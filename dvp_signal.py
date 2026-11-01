"""
dvp_signal.py
=============
Defense-vs-position as a number a projection can use, built so it can be tested without peeking.

For a stat (receptions, rush yards, DraftKings points...) and a position group (QB / RB / WR / TE), every game gives one number per
offense: what that position group produced against that defense. From those games, as of any point in time and using only EARLIER
games, a defense gets four readings, each a ratio to the league (1.10 = 10% more than the league average is allowed):

    sea    the season so far
    l4     the last four games (across the season boundary: the first weeks of a year lean on the end of last year)
    prev   all of last season
    adj    the season so far, schedule-adjusted: what the defense allowed divided by what the offenses it faced were EXPECTED to
           produce (their own season-to-date average, blended with last season's), so a defense that only met good offenses is not
           marked down for it

dvp_study.py asks whether any of these explain the miss between a player's projection and his result. dvp_track.py logs them weekly.
"""

import numpy as np
import pandas as pd

GROUPS = ("QB", "RB", "WR", "TE")
GROUP_OF = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE"}
PRIOR_GAMES = 4          # pseudo-games of last season (or the league) in an offense's expected output
READINGS = ("sea", "l4", "prev", "adj")


def game_table(history: pd.DataFrame, values: pd.Series) -> pd.DataFrame:
    """
    One row per (offense, defense, game, position group): the sum of `values` (aligned to `history`) over that group's players.
    Columns: season, week, off, def, grp, v.  A group that produced nothing in a game it played counts as zero, not a missing game.
    """
    h = history.assign(_v=values.to_numpy())
    h = h[h["position"].isin(GROUP_OF)].assign(grp=lambda d: d["position"].map(GROUP_OF))
    g = (h.groupby(["season", "week", "team", "opponent_team", "grp"])["_v"].sum().reset_index()
          .rename(columns={"team": "off", "opponent_team": "def", "_v": "v"}))
    # every group exists in every game, even at zero
    games = g[["season", "week", "off", "def"]].drop_duplicates()
    full = games.merge(pd.DataFrame({"grp": list(GROUPS)}), how="cross")
    return full.merge(g, on=["season", "week", "off", "def", "grp"], how="left").fillna({"v": 0.0})


def with_expectation(gt: pd.DataFrame) -> pd.DataFrame:
    """
    Add `e`: what each offense was expected to produce in that game from its own EARLIER games: the season-to-date mean blended with
    last season's mean (PRIOR_GAMES pseudo-games), or, with no last season, just the games so far (none at all: no expectation).
    """
    gt = gt.sort_values(["grp", "off", "season", "week"]).reset_index(drop=True)
    league = gt.groupby(["grp", "season"])["v"].mean()                       # only ever read for a PREVIOUS season
    key = gt.groupby(["grp", "off", "season"])
    n = key.cumcount().to_numpy(float)
    prior_sum = (key["v"].cumsum() - gt["v"]).to_numpy(float)
    team_season = gt.groupby(["grp", "off", "season"])["v"].mean()
    prev = np.array([team_season.get((grp, off, season - 1), np.nan) if (grp, off, season - 1) in team_season.index
                     else league.get((grp, season - 1), np.nan)
                     for grp, off, season in zip(gt["grp"], gt["off"], gt["season"])], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        blended = (prior_sum + PRIOR_GAMES * prev) / (n + PRIOR_GAMES)
        own_only = prior_sum / n                                              # no last season to lean on: just the games so far
    gt["e"] = np.where(np.isnan(prev), np.where(n > 0, own_only, np.nan), blended)       # NaN: nothing earlier to expect anything from
    return gt


def readings_asof(gt: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """
    {sea, n_sea, l4, prev, adj} for every (defense, group) using only games before (season, week). `gt` is with_expectation's output.
    Index: (grp, def). Ratios are to the league mean over the same games; missing readings are NaN.
    """
    stamp = gt["season"] * 100 + gt["week"]
    prior = gt[stamp < season * 100 + week]
    sea = prior[prior["season"] == season]
    out = {}
    for grp in GROUPS:
        p, s = prior[prior["grp"] == grp], sea[sea["grp"] == grp]
        last = p[p["season"] == season - 1]
        cur_mean = s["v"].mean() if len(s) else np.nan
        lg_recent = p.sort_values(["season", "week"]).tail(32 * 4)["v"].mean() if len(p) else np.nan
        cols = {}
        if len(s):
            by = s.groupby("def")
            cols["sea"] = by["v"].mean() / cur_mean
            cols["n_sea"] = by["v"].size().astype(float)
            known = s[s["e"].notna()].groupby("def")
            cols["adj"] = known["v"].sum() / known["e"].sum()
        if len(p) and lg_recent:
            cols["l4"] = p.sort_values(["season", "week"]).groupby("def").tail(4).groupby("def")["v"].mean() / lg_recent
        if len(last):
            cols["prev"] = last.groupby("def")["v"].mean() / last["v"].mean()
        df = pd.DataFrame(cols)
        for c in ("sea", "n_sea", "l4", "prev", "adj"):
            if c not in df:
                df[c] = np.nan
        out[grp] = df[["sea", "n_sea", "l4", "prev", "adj"]]
    return pd.concat(out, names=["grp", "def"])


def shrunk(ratio, n, k=6.0):
    """Pull a ratio toward 1 by its sample size: n games earn n / (n + k) of the way."""
    return 1 + (ratio - 1) * (n / (n + k))


# ---- production: turn the readings into the multiplier a projection uses --------------------------------------------------
import json
import pathlib

WEIGHTS_FILE = pathlib.Path(__file__).parent / "dvp_weights.json"
FACTOR_CLIP = (0.88, 1.12)       # never move a projection more than 12% for the defense alone


def load_weights(path=None) -> dict:
    """{market: weight} from dvp_weights.json (written by `dvp_study.py --write`); markets with weight 0 or a missing file get nothing."""
    path = pathlib.Path(path or WEIGHTS_FILE)
    if not path.exists():
        return {}
    try:
        return {m: float(w) for m, w in json.loads(path.read_text(encoding="utf-8")).get("weights", {}).items() if float(w) > 0}
    except (ValueError, OSError):
        return {}


def excess(r: pd.Series) -> float:
    """The factor dvp_study.py validated: the mean excess over 1 of the season-so-far, last-four and schedule-adjusted readings, each shrunk by its sample (a missing reading counts as 0)."""
    n = 0.0 if pd.isna(r["n_sea"]) else float(r["n_sea"])
    parts = [shrunk(r["sea"], n) - 1 if pd.notna(r["sea"]) else 0.0,
             shrunk(r["l4"], SHRINK_FIXED_L4) - 1 if pd.notna(r["l4"]) else 0.0,
             shrunk(r["adj"], n) - 1 if pd.notna(r["adj"]) else 0.0]
    return sum(parts) / 3


SHRINK_FIXED_L4 = 4


def factors(history: pd.DataFrame, values: pd.Series, weight: float, season: int) -> dict:
    """{(defense, position): multiplier} for one market as of NOW (every game in `history` is earlier than the one being projected)."""
    gt = with_expectation(game_table(history, values))
    rd = readings_asof(gt, season, 99)
    out = {}
    for (grp, defense), r in rd.iterrows():
        f = float(np.clip(1 + weight * excess(r), *FACTOR_CLIP))
        for pos, g in GROUP_OF.items():
            if g == grp:
                out[(defense, pos)] = f
    return out


def apply_to_multipliers(mults: dict, history: pd.DataFrame, specs: dict, stat_fn, season: int, weights: dict = None) -> dict:
    """
    Multiply the validated DvP factor into mults[market][(defense, position)] in place, for the markets that earned a weight.
    -> {market: weight} actually applied.
    """
    weights = load_weights() if weights is None else weights
    applied = {}
    for m, w in weights.items():
        if m not in mults or m not in specs or w <= 0:
            continue
        f = factors(history, stat_fn(history, specs[m]), w, season)
        base = mults[m]
        for key, v in f.items():
            base[key] = base.get(key, 1.0) * v
        applied[m] = w
    return applied
