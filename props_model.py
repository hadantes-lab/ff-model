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
LOG_GAMES = 10        # recent games exported for the hit-rate strip
TD_PRIOR_GAMES = 8    # pseudo-games of position-average TD rate blended into a player's own
COUNT_PRIOR_GAMES = 1  # weak floor for low-volume counts (see fit_player)
MIN_MEAN = {"count": 0.05, "yards": 1.0, "td": 0.0}   # below this a player has no measurable role


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

# cheapest useful set first: what most people mean by "player props"
CORE_MARKETS = [
    "player_pass_yds", "player_pass_tds", "player_rush_yds",
    "player_receptions", "player_reception_yds", "player_anytime_td",
]


def normalize_name(name: str) -> str:
    """Sportsbook and nflverse spell players slightly differently (Jr., periods, apostrophes)."""
    name = re.sub(r"[.'\-]", "", name.lower())
    name = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", name.strip())
    return re.sub(r"\s+", " ", name).strip()


# ---- history ------------------------------------------------------------
def build_history(player_stats: pd.DataFrame) -> pd.DataFrame:
    """Regular-season offensive player-games in chronological order."""
    df = player_stats[player_stats["season_type"] == "REG"].copy()
    df = df[df["position"].isin(["QB", "RB", "WR", "TE"])]
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
    """
    df = history.copy()
    df["_v"] = stat_series(df, spec)
    per_game = df.groupby(["opponent_team", "game_id", "position"])["_v"].sum().reset_index()
    league = per_game.groupby("position")["_v"].mean()
    out = {}
    for (opp, pos), g in per_game.groupby(["opponent_team", "position"]):
        base = league.get(pos, 0.0)
        if base <= 0:
            continue
        raw = g["_v"].mean() / base
        trust = len(g) / (len(g) + OPP_SHRINK_GAMES)
        mult = 1 + trust * OPP_STRENGTH * (raw - 1)
        out[(opp, pos)] = float(np.clip(mult, *OPP_CLIP))
    return out


def position_priors(history: pd.DataFrame, spec: Spec) -> dict:
    """Per-game average of a stat by position, among players with a real role (not scrubs)."""
    df = history[(history["targets"] + history["carries"] + history["attempts"]) >= 3]
    return stat_series(df, spec).groupby(df["position"]).mean().to_dict()


def fit_player(history: pd.DataFrame, player_id: str, spec: Spec, opp_mult: float = 1.0, priors=None):
    """
    -> dict(mean, var, n_games, n_eff, log) or None if too little history.
    `history` must already be sorted chronologically (build_history does).

    TDs are rare events, so a player's own TD history is mostly noise: 0 TDs
    in 17 games does not mean a 0% chance. Their rate is pulled toward the
    position average (TD_PRIOR_GAMES pseudo-games of it) and modeled as Poisson.

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
            mean = (n_eff * mean + TD_PRIOR_GAMES * prior) / (n_eff + TD_PRIOR_GAMES)
        elif spec.kind == "count" and mean < prior:
            mean = (n_eff * mean + COUNT_PRIOR_GAMES * prior) / (n_eff + COUNT_PRIOR_GAMES)
    if mean < MIN_MEAN[spec.kind]:
        return None
    mean *= opp_mult
    var *= opp_mult ** 2
    if spec.kind == "td":
        var = mean
    recent = g.tail(LOG_GAMES)
    log = [[int(s), int(w), o, round(float(v), 1)]
           for s, w, o, v in zip(recent["season"], recent["week"], recent["opponent_team"],
                                 stat_series(recent, spec))]
    return {"mean": mean, "var": var, "n_games": int(len(g)), "n_eff": round(n_eff, 1), "log": log}


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
