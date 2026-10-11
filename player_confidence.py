"""
player_confidence.py
====================
How well has the model done on THIS player this season, and how much should that change how much we trust it
on him this week?

THE RECORD
----------
Every graded prop in tracking/props_log.csv says which side the model leaned (its raw probability of the over,
before blending with the market) and whether the result went that way. Per (season, player, market) this keeps:
    n, hits          how many times the model's side was right
    shrunk hit rate  (hits + PRIOR_GAMES x 50%) / (n + PRIOR_GAMES): three games can't make a streak count
    accuracy         the typical size of the miss in standard deviations of the model's own forecast (|z|;
                     a well-calibrated forecast averages ~0.8), from the projection and spread logged alongside
                     each prediction (available for props logged from this feature onward)

THE GRADE (0-10)
----------------
grade = 10 x (half the shrunk hit rate + half the accuracy score), or the hit rate alone until accuracy data
exists. 5.0 is "no information" -- a player with no tracked games is a 5, and a short record stays close to 5.

HOW IT ENTERS THE EDGE, AND WHY IT IS CALIBRATED RATHER THAN CHOSEN
-------------------------------------------------------------------
The model's blend weight on a prop is multiplied by

    1 + strength x (grade - 5) / 5            (a 10 -> x(1+strength), a 0 -> x(1-strength))

`strength` is MEASURED, not assumed: `market_trust.py` regresses "did the model's side hit next time?" on the
shrunk record going in, over ~9,000 past props (2024-25), and sets strength > 0 only if the record predicts the
next result with a confidence interval that excludes zero. At the time of writing it does not: the model's side
hit 50.6% of the time whether the player's record was poor, average or strong (correlation -0.003), so strength is
0 and the grade is shown but moves nothing. If the tracked data ever shows the record has predictive power, the
same study turns the factor on, with no code change.
"""

import pathlib

import numpy as np
import pandas as pd

RECORD_FILE = pathlib.Path(__file__).parent / "tracking" / "player_model_record.csv"

PRIOR_GAMES = 6            # a player's record counts as this many games at 50% before his own results
EXPECTED_ABS_Z = 0.8       # mean |z| of a perfectly calibrated normal forecast
MAX_STRENGTH = 0.5         # the grade can move the model's weight by at most +/-50%
GRADE_COLUMNS = ["season", "player_id", "market", "player", "n", "hits", "hit_rate", "shrunk_rate", "n_acc", "mean_abs_z",
                 "accuracy_score", "grade"]


def shrunk_rate(hits: float, n: float, prior_games: float = PRIOR_GAMES) -> float:
    return (hits + 0.5 * prior_games) / (n + prior_games)


def accuracy_score(sum_abs_z: float, n_acc: int, prior_games: float = PRIOR_GAMES) -> float:
    """0..1 with 0.5 = as calibrated as expected, higher = the model's forecasts on him have been tighter than its stated spread."""
    m = (sum_abs_z + prior_games * EXPECTED_ABS_Z) / (n_acc + prior_games)
    return float(np.clip(0.5 + (EXPECTED_ABS_Z - m) / (2 * EXPECTED_ABS_Z), 0.0, 1.0))


def grade_from(hits: float, n: float, sum_abs_z: float = 0.0, n_acc: int = 0) -> float:
    """The 0-10 grade; 5.0 for a player with no tracked games."""
    rec = shrunk_rate(hits, n)
    score = rec if n_acc <= 0 else 0.5 * rec + 0.5 * accuracy_score(sum_abs_z, n_acc)
    return round(float(np.clip(10 * score, 0.0, 10.0)), 1)


def multiplier(grade: float, strength: float) -> float:
    """Factor applied to the model's blend weight. strength = 0 -> always 1.0."""
    s = float(np.clip(strength, 0.0, MAX_STRENGTH))
    return 1.0 + s * (float(grade) - 5.0) / 5.0


def player_records(log_df: pd.DataFrame, season: int = None) -> pd.DataFrame:
    """
    One row per (season, player, market) from the graded tracking log. The model's side is its RAW lean (model_over
    above or below 50%); pushes, DNPs, games and rows with no clear lean are left out.
    """
    if log_df is None or log_df.empty:
        return pd.DataFrame(columns=GRADE_COLUMNS)
    d = log_df[(log_df["pos"] != "GAME") & log_df["result"].isin(["over", "under"]) & log_df["model_over"].notna()].copy()
    if season is not None:
        d = d[d["season"] == season]
    d = d[d["model_over"] != 0.5]
    if d.empty:
        return pd.DataFrame(columns=GRADE_COLUMNS)
    d["hit"] = ((d["model_over"] > 0.5) & (d["result"] == "over")) | ((d["model_over"] < 0.5) & (d["result"] == "under"))
    if "mean" in d and "sd" in d:
        ok = d["mean"].notna() & d["sd"].notna() & (d["sd"] > 0) & d["actual"].notna()
        d["abs_z"] = np.where(ok, (d["actual"].astype(float) - d["mean"]).abs() / d["sd"], np.nan)
    else:
        d["abs_z"] = np.nan
    rows = []
    for (s, pid, m), g in d.groupby(["season", "player_id", "market"]):
        n, hits = len(g), int(g["hit"].sum())
        z = g["abs_z"].dropna()
        rows.append({"season": int(s), "player_id": pid, "market": m, "player": g["player"].iloc[-1], "n": n, "hits": hits,
                     "hit_rate": hits / n, "shrunk_rate": shrunk_rate(hits, n), "n_acc": int(len(z)),
                     "mean_abs_z": float(z.mean()) if len(z) else np.nan,
                     "accuracy_score": accuracy_score(float(z.sum()), len(z)) if len(z) else np.nan,
                     "grade": grade_from(hits, n, float(z.sum()), len(z))})
    return pd.DataFrame(rows, columns=GRADE_COLUMNS)


def lookup(records: pd.DataFrame, season: int) -> dict:
    """{(player_id, market): {grade, n, hits, hit_rate, mean_abs_z}} for one season."""
    out = {}
    if records is None or records.empty:
        return out
    for r in records[records["season"] == season].itertuples(index=False):
        out[(r.player_id, r.market)] = {"grade": float(r.grade), "n": int(r.n), "hits": int(r.hits), "hit_rate": float(r.hit_rate),
                                        "mean_abs_z": None if pd.isna(r.mean_abs_z) else float(r.mean_abs_z)}
    return out


def confidence_for(rec: dict, strength: float) -> dict:
    """The block shown on a prop: the player's record, grade, and the factor it applies to the model's weight."""
    if rec is None:
        rec = {"grade": 5.0, "n": 0, "hits": 0, "hit_rate": None, "mean_abs_z": None}
    return dict(rec, strength=float(strength), multiplier=round(multiplier(rec["grade"], strength), 3))


# ---- measuring `strength` --------------------------------------------------------------
def persistence(df: pd.DataFrame, id_cols=("pid", "market"), min_prior: int = 3) -> dict:
    """
    Does the model's record on a player predict whether it hits him next time? `df` needs one row per
    prop, ordered in time within each player-market, with a boolean/0-1 `hit`. For each row only EARLIER rows of the
    same player-market are used (shrunk with PRIOR_GAMES). -> {slope, se, ci, n}: the slope of next-hit on
    (shrunk prior rate - 0.5); 0 means the record carries no information.
    """
    d = df.sort_values(list(id_cols) + [c for c in ("season", "week") if c in df]).copy()
    g = d.groupby(list(id_cols), sort=False)["hit"]
    n_prior = g.cumcount()
    hits_prior = g.transform(lambda s: s.shift().expanding().sum()).fillna(0)
    d["prior"] = (hits_prior + 0.5 * PRIOR_GAMES) / (n_prior + PRIOR_GAMES) - 0.5
    d = d[n_prior >= min_prior]
    if len(d) < 200:
        return {"slope": 0.0, "se": None, "ci": [None, None], "n": int(len(d))}
    x, y = d["prior"].to_numpy(float), d["hit"].to_numpy(float)
    X = np.c_[x, np.ones(len(x))]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    se = float(np.sqrt(np.linalg.inv(X.T @ X)[0, 0] * resid.var(ddof=2)))
    return {"slope": float(beta[0]), "se": se, "ci": [float(beta[0] - 1.96 * se), float(beta[0] + 1.96 * se)], "n": int(len(d))}


def strength_from(p: dict) -> float:
    """
    The weight swing the grade is allowed: positive only if the record's predictive slope is statistically above
    zero. A slope of 0.25 (the record moves the chance of a hit by a quarter-point per point of record) earns the
    full MAX_STRENGTH; anything whose 95% interval includes zero earns none.
    """
    lo = p["ci"][0]
    if lo is None or lo <= 0:
        return 0.0
    return float(np.clip(p["slope"] / 0.25 * MAX_STRENGTH, 0.0, MAX_STRENGTH))
