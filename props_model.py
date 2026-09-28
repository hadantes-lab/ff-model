"""
props_model.py
==============
Player prop simulator core: turns a player's recent game log into a
predictive distribution for a single stat, then reads probabilities off it.

WHY NOT NORMAL DRAWS
--------------------
simulate.py draws every stat from a normal distribution. That is a poor fit
for the stats props are actually written on:
  - counts (receptions, attempts, TDs, INTs) are discrete and right-skewed
    -> Poisson, or negative binomial when games vary more than Poisson allows
  - yardage is right-skewed with a fat tail of big games
    -> gamma (mean and variance matched to the player's recent games)
  - anytime TD is just P(rushing + receiving TDs >= 1) from the count model

HOW A PROJECTION IS BUILT
-------------------------
1. Take the player's last MAX_GAMES games (across seasons), weight recent
   games more (exponential decay, HALFLIFE games).
2. Weighted mean and variance -> then inflate variance by the uncertainty in
   the mean itself (few games = wide distribution).
3. Scale mean and spread by an opponent multiplier: how much the defense
   allows to that position vs league average, shrunk hard toward 1.0 so a
   couple of games can't swing the projection.
4. Simulate N_SIMS games from the fitted distribution.

WHAT THIS DOES NOT DO (yet)
---------------------------
  - No correlation between stats (a QB's yards and his WR's yards are
    simulated independently), no game script, weather, or injury-driven
    role changes. The market prices those; this model does not.
"""

import hashlib
import json
import pathlib
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

SEED = 2026
N_SIMS = 20000
HALFLIFE = 5          # games; recent form weighs more
MIN_GAMES = 3         # need at least this many games to project anyone
MAX_GAMES = 17        # look-back window
OPP_SHRINK_GAMES = 6  # opponent effect is trusted more as its sample grows
OPP_STRENGTH = 0.6    # ...and never applied at full strength
OPP_CLIP = (0.8, 1.25)
CURRENT_SEASON_WEIGHT = 3.0   # this season's games count triple in defensive strength: defenses change year to year
LOG_GAMES = 10        # recent games exported for the hit-rate strip
TD_PRIOR_GAMES = 8    # pseudo-games of position-average TD rate blended into a player's own
COUNT_PRIOR_GAMES = 1  # weak floor for low-volume counts (see fit_player)
MIN_MEAN = {"count": 0.05, "yards": 1.0, "td": 0.0}
# How much of the model's disagreement with the market is believed. Against real lines the raw model
# differed from the market by ~12 percentage points on a typical prop, when a genuine edge is a few
# points at most -- so most of that gap is the model missing roles/injuries/game script the market
# knows. This cap is a judgment call, NOT a fitted value; fitting it needs closing lines plus results.
LAMBDA_MAX = 0.15
LAMBDA_FULL_N = 8     # games' worth of (recency-weighted) history needed to earn the full weight   # below this a player has no measurable role


CALIBRATION_FILE = pathlib.Path(__file__).parent / "model_calibration.json"


def load_calibration(path=CALIBRATION_FILE):
    """Bias-calibration and game-environment coefficients fitted by tune_props.py, or None."""
    try:
        return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def environment_effect(calibration, market):
    """{a, b} for a market's game-environment effect, already dampened by its out-of-sample slope."""
    e = ((calibration or {}).get("environment") or {}).get(market)
    return None if not e else {"a": e["a"] * e["damp"], "b": e["b"] * e["damp"]}


@dataclass(frozen=True)
class Spec:
    label: str
    cols: tuple
    kind: str  # "yards" | "count" | "td"


# Odds API market key -> how to build the stat from nflverse weekly columns
MARKETS = {
    "player_pass_yds":           Spec("Pass Yds",      ("passing_yards",), "yards"),
    "player_pass_tds":           Spec("Pass TDs",      ("passing_tds",), "count"),
    "player_pass_attempts":      Spec("Pass Att",      ("attempts",), "count"),
    "player_pass_completions":   Spec("Completions",   ("completions",), "count"),
    "player_pass_interceptions": Spec("Interceptions", ("passing_interceptions",), "count"),
    "player_rush_yds":           Spec("Rush Yds",      ("rushing_yards",), "yards"),
    "player_rush_attempts":      Spec("Rush Att",      ("carries",), "count"),
    "player_receptions":         Spec("Receptions",    ("receptions",), "count"),
    "player_reception_yds":      Spec("Rec Yds",       ("receiving_yards",), "yards"),
    "player_rush_reception_yds": Spec("Rush+Rec Yds",  ("rushing_yards", "receiving_yards"), "yards"),
    "player_anytime_td":         Spec("Anytime TD",    ("rushing_tds", "receiving_tds"), "td"),
}

# Minimum recent average for a player to count as a "regular" for a stat. Used when estimating
# league-wide effects (bias, game environment): a backup QB whose 2 attempts a game come in
# garbage time, or a WR4 with an occasional catch, has erratic outcomes that swamp the signal
# (this cost us a wrong-signed result before it was added). Roughly the props-worthy population.
ROLE_FLOOR = {
    "player_pass_yds": 100.0, "player_pass_tds": 0.8, "player_pass_attempts": 15.0,
    "player_pass_completions": 10.0, "player_pass_interceptions": 0.3,
    "player_rush_yds": 20.0, "player_rush_attempts": 5.0, "player_receptions": 2.0,
    "player_reception_yds": 20.0, "player_rush_reception_yds": 25.0, "player_anytime_td": 0.0,
}

# cheapest useful set first: what most people mean by "player props"
CORE_MARKETS = [
    "player_pass_yds", "player_pass_tds", "player_rush_yds",
    "player_receptions", "player_reception_yds",
]   # anytime TD is a long shot dominated by noise, so it is opt-in (--markets / --all-markets)


def normalize_name(name: str) -> str:
    """Sportsbook and nflverse spell players slightly differently (Jr., periods, apostrophes)."""
    name = re.sub(r"[.'\-]", "", name.lower())
    name = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", name.strip())
    return re.sub(r"\s+", " ", name).strip()


# ---- history ------------------------------------------------------------
def build_history(player_stats: pd.DataFrame) -> pd.DataFrame:
    """Regular-season offensive player-games in chronological order."""
    df = player_stats[player_stats["season_type"] == "REG"].copy()
    df = df[df["position"].isin(["QB", "RB", "WR", "TE", "FB"])]
    for col in {c for s in MARKETS.values() for c in s.cols}:
        df[col] = df[col].fillna(0)
    return df.sort_values(["season", "week"]).reset_index(drop=True)


def stat_series(df: pd.DataFrame, spec: Spec) -> pd.Series:
    return df[list(spec.cols)].sum(axis=1)


# ---- fitting ------------------------------------------------------------
def recency_weights(n: int, halflife: float = HALFLIFE) -> np.ndarray:
    """Oldest game first; the most recent game gets weight 1.0."""
    ages = np.arange(n)[::-1]
    return 0.5 ** (ages / halflife)


def fit_moments(values, weights):
    """
    Weighted mean and *predictive* variance.
    The variance is inflated by 1/n_eff because the mean is itself estimated
    from a handful of games -- otherwise 4 lucky games look like certainty.
    """
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mean = float(np.average(x, weights=w))
    n_eff = float(w.sum() ** 2 / (w ** 2).sum())
    var = float(np.average((x - mean) ** 2, weights=w))
    if n_eff > 1:
        var *= n_eff / (n_eff - 1)          # small-sample correction
    return mean, var * (1 + 1 / n_eff), n_eff


def opponent_multipliers(history: pd.DataFrame, spec: Spec) -> dict:
    """
    {(opponent, position): multiplier}: what a defense allows to a position
    group per game vs the league average for that group, shrunk toward 1.

    Games from the newest season in `history` count CURRENT_SEASON_WEIGHT times as much as
    older ones, so a defense's first weeks this year (injuries, new scheme, new players)
    move its rating quickly instead of being drowned out by last year.
    """
    df = history.copy()
    df["_v"] = stat_series(df, spec)
    if "season" not in df:
        df["season"] = 0
    per_game = df.groupby(["opponent_team", "game_id", "position"]).agg(
        _v=("_v", "sum"), season=("season", "first")).reset_index()
    per_game["_w"] = np.where(per_game["season"] == per_game["season"].max(), CURRENT_SEASON_WEIGHT, 1.0) \
        if per_game["season"].nunique() > 1 else 1.0
    league = per_game.groupby("position")["_v"].mean()
    out = {}
    for (opp, pos), g in per_game.groupby(["opponent_team", "position"]):
        base = league.get(pos, 0.0)
        if base <= 0:
            continue
        n_w = float(g["_w"].sum())
        raw = float(np.average(g["_v"], weights=g["_w"])) / base
        trust = n_w / (n_w + OPP_SHRINK_GAMES)
        mult = 1 + trust * OPP_STRENGTH * (raw - 1)
        out[(opp, pos)] = float(np.clip(mult, *OPP_CLIP))
    return out


def position_priors(history: pd.DataFrame, spec: Spec) -> dict:
    """
    Regression targets by position, among players with a real role (not scrubs):
      - counts: the per-game average of the stat
      - TDs:    touchdowns per touch (targets + carries), so a low-usage player is regressed toward
                what his *usage* supports, not toward a starter's scoring rate.
    """
    df = history[(history["targets"] + history["carries"] + history["attempts"]) >= 3]
    if spec.kind == "td":
        touches = (df["targets"] + df["carries"]).groupby(df["position"]).sum()
        rate = (stat_series(df, spec).groupby(df["position"]).sum() / touches).replace([np.inf, -np.inf], np.nan).dropna()
        return rate.to_dict()
    return stat_series(df, spec).groupby(df["position"]).mean().to_dict()


def market_weight(n_eff: float, lambda_max: float = LAMBDA_MAX) -> float:
    """
    Share of the model's disagreement with the market that is trusted; thinner history -> less.
    `lambda_max` (the ceiling at full history) defaults to the static LAMBDA_MAX but can be
    overridden per market once props_tracker.py's weekly refinement has enough real tracked
    outcomes to justify a different ceiling (see live_calibration.json).
    """
    return lambda_max * min(1.0, max(0.0, n_eff) / LAMBDA_FULL_N)


def blend_toward_market(model_p: float, market_p: float, n_eff: float, lambda_max: float = None) -> float:
    """Shrink a model probability toward the market's: market + w * (model - market)."""
    return market_p + market_weight(n_eff, LAMBDA_MAX if lambda_max is None else lambda_max) * (model_p - market_p)


def fit_player(history: pd.DataFrame, player_id: str, spec: Spec, opp_mult: float = 1.0, priors=None,
               calib=None, env_fn=None):
    """
    -> dict(mean, var, n_games, n_eff, log) or None if too little history.
    `history` must already be sorted chronologically (build_history does).

    TDs are rare events, so a player's own TD history is mostly noise: 0 TDs
    in 17 games does not mean a 0% chance. Their rate is pulled toward what his
    recent touches would produce at the position's TD-per-touch rate
    (TD_PRIOR_GAMES pseudo-games of it) and modeled as Poisson.

    `calib` ({a, b}) maps the projection through  actual ~ a + b*projection, fitted on walk-forward
    backtests: players chosen for their recent production run ~5% hot (regression to the mean).
    `env_fn(games, weights)` returns the game-environment multiplier (see game_context.py).

    Low-volume counts get the same idea, more weakly: a back with no catches in
    5 games is not a 0% chance of one, so a count mean *below* the position
    average is nudged up by COUNT_PRIOR_GAMES pseudo-games. High-volume players
    are left alone. A player whose mean is essentially zero has no measurable
    role, so there is nothing to simulate and None is returned.
    """
    g = history[history["player_id"] == player_id].tail(MAX_GAMES)
    if len(g) < MIN_GAMES:
        return None
    vals = stat_series(g, spec).to_numpy()
    mean, var, n_eff = fit_moments(vals, recency_weights(len(vals)))
    prior = (priors or {}).get(g["position"].iloc[-1])
    if prior is not None:
        if spec.kind == "td":
            touches = float(np.average((g["targets"] + g["carries"]).to_numpy(float), weights=recency_weights(len(g))))
            mean = (n_eff * mean + TD_PRIOR_GAMES * touches * prior) / (n_eff + TD_PRIOR_GAMES)
        elif spec.kind == "count" and mean < prior:
            mean = (n_eff * mean + COUNT_PRIOR_GAMES * prior) / (n_eff + COUNT_PRIOR_GAMES)
    if mean < MIN_MEAN[spec.kind]:
        return None
    mean *= opp_mult
    var *= opp_mult ** 2
    if spec.kind == "td":
        var = mean
    if calib and spec.kind != "td" and mean > 0:
        cal = calib["a"] + calib["b"] * mean
        if cal > 0:
            var *= (cal / mean) ** 2          # keep the coefficient of variation when the mean moves
            mean = cal
    env = None
    if env_fn is not None and spec.kind != "td":
        env = env_fn(g, recency_weights(len(g)))
        mean *= env["mult"]
        var *= env["mult"] ** 2
    recent = g.tail(LOG_GAMES)
    log = [[int(s), int(w), o, round(float(v), 1)]
           for s, w, o, v in zip(recent["season"], recent["week"], recent["opponent_team"],
                                 stat_series(recent, spec))]
    return {"mean": mean, "var": var, "n_games": int(len(g)), "n_eff": round(n_eff, 1), "log": log, "env": env}


# ---- simulation ---------------------------------------------------------
def draw(kind: str, mean: float, var: float, rng: np.random.Generator, n: int = N_SIMS) -> np.ndarray:
    """Draw n games from the distribution that suits this kind of stat."""
    if mean <= 1e-9:
        return np.zeros(n)
    if kind in ("count", "td"):
        if var <= mean * 1.02:
            return rng.poisson(mean, n).astype(float)
        r = mean ** 2 / (var - mean)         # negative binomial: over-dispersed counts
        p = r / (r + mean)
        return rng.negative_binomial(r, p, n).astype(float)
    k = mean ** 2 / max(var, 1e-9)           # gamma: skewed yardage
    return rng.gamma(k, var / mean, n)


def fair_line(kind: str, draws: np.ndarray):
    """
    The model's OWN "prediction line" -- the half-point line where its simulated distribution
    alone is closest to a coin flip -- computed independently of any market line, so it can be
    checked against the real posted line once one is pulled. None for "td" (a yes/no
    probability, not a line).
    """
    if kind == "td":
        return None
    if kind == "yards":
        median = float(np.median(draws))
        line = float(np.floor(median * 2 + 0.5) / 2)
        return line + 0.5 if line == int(line) else line
    k = int(np.floor(np.median(draws)))
    candidates = (max(0.5, k - 0.5), k + 0.5)
    return min(candidates, key=lambda c: abs(float((draws > c).mean()) - 0.5))


def line_probs(draws: np.ndarray, line: float) -> dict:
    """P(over), P(push), P(under) for a line. Pushes only exist on whole-number lines."""
    over = float((draws > line).mean())
    push = float((draws == line).mean())
    return {"over": over, "push": push, "under": max(0.0, 1.0 - over - push)}


def distribution_summary(kind: str, draws: np.ndarray) -> dict:
    """
    Compact form of the simulated distribution the web page can re-read at
    ANY line: a pmf for counts, 0-100th percentiles for yardage.
    """
    if kind in ("count", "td"):
        kmax = int(np.ceil(np.quantile(draws, 0.9995))) + 1
        pmf = np.bincount(draws.astype(int), minlength=kmax + 1)[: kmax + 1] / len(draws)
        return {"pmf": [round(float(p), 4) for p in pmf]}
    q = np.quantile(draws, np.linspace(0, 1, 101))
    return {"q": [round(float(v), 1) for v in q]}


def rng_for(*parts) -> np.random.Generator:
    """Deterministic per-prop RNG so re-running the export doesn't shuffle results."""
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return np.random.default_rng([SEED, int.from_bytes(digest[:8], "little")])
