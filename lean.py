"""
lean.py
=======
The simple view of a player prop: which side the model leans, how confident it is, whether that kind of lean has been hitting, and
whether the defense backs it up.

LEAN
----
The side of the line the RAW model simulation favors, at the line you would actually play (Underdog's when the pull had it, else the
sportsbook consensus line). Raw, because the blended number is the market's own probability nudged by a few points, so every prop sits
near 50% and cannot say how strongly the model leans.

CONFIDENCE  (a measured hit rate, not a feeling)
------------------------------------------------
Confidence is the chance the lean hits, read off a two-variable logistic curve fitted to this season's graded props in
tracking/props_log.csv: how strongly the raw model leans (|P(over) - 50%|), and whether the market's own side agrees. It was checked
out of sample, week by week (fit on earlier weeks only, scored on the next): the predicted rates matched the actual ones across the
range and beat a flat 50-50 guess on Brier score, though only slightly. What the log shows, which is why those two inputs:
  - the harder the raw model leans, the more often it is right (about 50% when it barely leans, 57% when it leans hard);
  - a lean the market also favors hits about 56% of the time, a lean AGAINST the market about 49%.
The fit is refreshed from the log on every run, so the numbers move as results arrive; with too few graded props it falls back to the
last measured curve.

HITTING AT A GOOD CLIP, AND WHAT IT IS WORTH
--------------------------------------------
Two records are shown, and they are not equally trustworthy:
  - the TYPE of lean (same market, same side, with or without the market's agreement) this season. Tested week by week this predicts
    the next week's hit rate weakly (correlation about +0.06; the best-record types hit ~55%, the worst ~47%).
  - the PLAYER's own record on that market. Tested the same way it does NOT predict the next result (the model's side hit about 50%
    whether his record so far was great or poor), consistent with the earlier test in player_confidence.py.
A type that has been MISSING (a shrunk record of 46% or worse over 25+ picks, e.g. passing-yards unders this season) is flagged "cold"
and its grade is capped at "low". A prop is flagged "hot" when its type has a good shrunk record with enough picks behind it, or the player's record is strong. The
flag says what has happened, and the page says which of the two it is so the weaker one is not mistaken for the stronger.

MATCHUP
-------
Does the opposing defense point the same way? A soft defense (ranks 23-32 in what it allows to that position on that stat) backs an
over, a tough one (1-10) backs an under. Tested on the same graded props, "the matchup backs the lean" hit about 54% against 53% when
it did not, a lift too small to move the confidence (adding it did not improve the out-of-sample score), so it is shown as a
highlight, not folded into the number.
"""

import math

import numpy as np
import pandas as pd

MIN_FIT_ROWS = 600                 # fewer graded props than this and the last measured curve is used
L2 = 3.0                           # ridge on the two slopes, so a lopsided week cannot swing them
S_SCALE = 5.0                      # the strength feature is entered as 5 x |P(over) - 0.5|
CONF_CLIP = (0.45, 0.66)
SHRINK = 0.9                       # the curve's top bucket ran ~2 points under its stated rate out of sample, so it is pulled 10% toward 50%
HIGH, MEDIUM = 0.575, 0.54         # confidence grades
# last measured curve (2026 weeks 3-10, 3,630 graded props); used until the log is large enough to refit
DEFAULT_CAL = {"b0": -0.1776, "b_s": 0.1878, "b_agree": 0.281, "n": 0, "base_rate": 0.527, "source": "default"}

SEG_PRIOR = 30                     # a type's record counts as this many picks at 50% before its own results
SEG_MIN_N = 25
SEG_HOT = 0.58                     # shrunk hit rate that marks a type of lean as hitting at a good clip
SEG_COLD = 0.46                    # ... and one that has been missing (the grade is capped at "low" for these)
PLAYER_MIN_N, PLAYER_HOT = 5, 0.80
RANK_SOFT, RANK_TOUGH = 23, 10     # defense-vs-position ranks (1 = allows the fewest, 32 = the most)
OPP_MULT_BAND = 0.03               # fallback when the defense has no rank on this stat: the projection's own opponent adjustment


# ---- the log -> examples ------------------------------------------------------------------
def examples(log_df: pd.DataFrame) -> pd.DataFrame:
    """Graded, non-TD, player props with a model and a market probability -> columns market, side, agree, s, hit, week."""
    if log_df is None or len(log_df) == 0:
        return pd.DataFrame(columns=["market", "side", "agree", "s", "hit", "week"])
    d = log_df[log_df["result"].isin(["over", "under"]) & (log_df["kind"] != "td") & (log_df["pos"] != "GAME")
               & log_df["model_over"].notna() & log_df["market_over"].notna()].copy()
    d = d[d["model_over"] != 0.5]
    d["side"] = np.where(d["model_over"] > 0.5, "over", "under")
    d["agree"] = (d["side"] == np.where(d["market_over"] > 0.5, "over", "under")).astype(float)
    d["s"] = (d["model_over"] - 0.5).abs()
    d["hit"] = (d["side"] == d["result"]).astype(float)
    return d[["market", "side", "agree", "s", "hit", "week", "season"]].reset_index(drop=True)


def _features(s, agree):
    s, agree = np.asarray(s, float), np.asarray(agree, float)
    return np.c_[np.ones(len(s)), S_SCALE * s, agree]


def _fit_logistic(X, y, l2=L2, iters=60):
    b = np.zeros(X.shape[1])
    pen = np.diag([0.0] + [1.0] * (X.shape[1] - 1))
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + l2 * pen
        step = np.linalg.solve(H, X.T @ (y - p) - l2 * (pen @ b))
        b += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return b


def fit_calibration(log_df: pd.DataFrame, min_rows: int = MIN_FIT_ROWS) -> dict:
    ex = examples(log_df)
    if len(ex) < min_rows:
        return dict(DEFAULT_CAL)
    b = _fit_logistic(_features(ex["s"], ex["agree"]), ex["hit"].to_numpy(float))
    return {"b0": round(float(b[0]), 4), "b_s": round(float(b[1]), 4), "b_agree": round(float(b[2]), 4), "n": int(len(ex)),
            "base_rate": round(float(ex["hit"].mean()), 4), "source": "fitted"}


def walk_forward(log_df: pd.DataFrame, min_rows: int = MIN_FIT_ROWS) -> dict:
    """
    The out-of-sample check: for each week, fit on EARLIER weeks only and score that week. -> {n, brier, brier_flat, buckets}, where
    each bucket is [predicted low, high, n, mean predicted, actual hit rate]; None if there is not enough history.
    """
    ex = examples(log_df)
    ex = ex.sort_values(["season", "week"]) if len(ex) else ex
    preds, ys = [], []
    for (season, week), te in (ex.groupby(["season", "week"]) if len(ex) else []):
        tr = ex[(ex["season"] < season) | ((ex["season"] == season) & (ex["week"] < week))]
        if len(tr) < min_rows:
            continue
        b = _fit_logistic(_features(tr["s"], tr["agree"]), tr["hit"].to_numpy(float))
        preds += list(1 / (1 + np.exp(-_features(te["s"], te["agree"]) @ b)))
        ys += list(te["hit"])
    if not ys:
        return None
    p, y = np.array(preds), np.array(ys)
    buckets = []
    for lo, hi in [(0, 0.5), (0.5, 0.53), (0.53, 0.56), (0.56, 0.6), (0.6, 1.01)]:
        m = (p >= lo) & (p < hi)
        if m.sum():
            buckets.append([lo, hi, int(m.sum()), round(float(p[m].mean()), 3), round(float(y[m].mean()), 3)])
    return {"n": int(len(y)), "brier": round(float(((p - y) ** 2).mean()), 5), "brier_flat": round(float(((y.mean() - y) ** 2).mean()), 5),
            "base_rate": round(float(y.mean()), 3), "buckets": buckets}


def confidence(cal: dict, s: float, agree: bool) -> float:
    z = cal["b0"] + cal["b_s"] * S_SCALE * s + cal["b_agree"] * (1.0 if agree else 0.0)
    return float(np.clip(0.5 + SHRINK * (1 / (1 + math.exp(-z)) - 0.5), *CONF_CLIP))


def grade(conf: float) -> str:
    return "high" if conf >= HIGH else "medium" if conf >= MEDIUM else "low"


# ---- records ------------------------------------------------------------------------------
def segment_records(log_df: pd.DataFrame, season: int = None) -> dict:
    """{(market, side, agree): {n, hits, rate (shrunk)}} for the season's graded leans."""
    ex = examples(log_df)
    if season is not None and len(ex):
        ex = ex[ex["season"] == season]
    out = {}
    for (m, side, ag), g in ex.groupby(["market", "side", "agree"]) if len(ex) else []:
        n, hits = len(g), int(g["hit"].sum())
        out[(m, side, bool(ag))] = {"n": n, "hits": hits, "rate": round((hits + 0.5 * SEG_PRIOR) / (n + SEG_PRIOR), 4)}
    return out


def matchup(p: dict, side: str, dvp: dict) -> dict:
    """
    Does the opposing defense point the same way as the lean?  -> {state: supports|against|neutral|None, rank, note}
    Uses the defense's rank in what it allows this position on this stat; falls back to the projection's own opponent multiplier.
    """
    pos = {"FB": "RB"}.get(p.get("pos"), p.get("pos"))
    cell = ((dvp or {}).get(p.get("opp"), {}).get(pos, {}) or {}).get(p.get("market"))
    if cell and cell.get("rank"):
        r = int(cell["rank"])
        soft, tough = r >= RANK_SOFT, r <= RANK_TOUGH
        if soft or tough:
            backs = (side == "over") == soft
            what = f"{p['opp']} allow the {_ordinal(33 - r if soft else r)} {'most' if soft else 'fewest'} {p.get('label', 'stat').lower()} to {pos}s"
            return {"state": "supports" if backs else "against", "rank": r, "note": what}
        return {"state": "neutral", "rank": r, "note": f"{p['opp']} rank {_ordinal(r)} of 32 against {pos}s on this stat: no edge either way"}
    m = p.get("opp_mult")
    if m is not None and abs(m - 1) >= OPP_MULT_BAND:
        backs = (side == "over") == (m > 1)
        return {"state": "supports" if backs else "against", "rank": None,
                "note": f"the projection moves {(m - 1) * 100:+.0f}% for {p['opp']}'s defense"}
    return {"state": None, "rank": None, "note": None}


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


# ---- the lean on a prop -------------------------------------------------------------------
def play_line(p: dict) -> tuple:
    """(line, book, raw P(over) at that line, market-anchored side at that line) for the line you would play."""
    pk = p.get("pickem")
    dfs = p.get("dfs") or []
    ud = next((d for d in dfs if pk and pk.get("book") and d["book"] == pk["book"]), None)
    if ud is not None:
        raw_over = ud["over"]
        line, book = ud["line"], ud["book"]
    else:
        raw_over, line, book = p.get("model_over"), p.get("line"), None
    anchored = pk["side"] if pk else ("over" if (p.get("over") or 0) >= (p.get("under") or 0) else "under")
    return line, book, raw_over, anchored


def lean_for(p: dict, cal: dict, segs: dict, dvp: dict, player_rec: dict = None) -> dict:
    """The `lean` block of one prop, or None for a prop with no over/under (anytime TDs) or no market to compare to."""
    if p.get("kind") == "td" or p.get("model_over") is None or p.get("market_over") is None or p.get("model_only"):
        return None
    line, book, raw_over, anchored = play_line(p)
    if raw_over is None or abs(raw_over - 0.5) < 1e-9:
        return None
    side = "over" if raw_over > 0.5 else "under"
    s = abs(raw_over - 0.5)
    agree = side == anchored
    conf = confidence(cal, s, agree)
    mu = matchup(p, side, dvp)
    seg = segs.get((p["market"], side, agree))
    seg_hot = bool(seg and seg["n"] >= SEG_MIN_N and seg["rate"] >= SEG_HOT)
    pr = player_rec if player_rec and player_rec.get("n", 0) >= PLAYER_MIN_N else None
    player_hot = bool(pr and pr["hits"] / pr["n"] >= PLAYER_HOT)
    seg_cold = bool(seg and seg["n"] >= SEG_MIN_N and seg["rate"] <= SEG_COLD)
    why = ([f"hitting: {seg['hits']} of {seg['n']} {side} leans in this market {'with' if agree else 'against'} the market this season"] if seg_hot else []) + \
          ([f"model right on {p['player']}'s {p.get('label', 'prop')} {pr['hits']} of {pr['n']} times this season"] if player_hot else [])
    return {"side": side, "line": line, "book": book, "raw_prob": round(raw_over if side == "over" else 1 - raw_over, 4),
            "confidence": round(conf, 4), "grade": "low" if seg_cold else grade(conf), "agree": agree,
            "matchup": mu, "segment": None if not seg else {k: seg[k] for k in ("n", "hits", "rate")}, "seg_hot": seg_hot,
            "player": None if not pr else {"n": pr["n"], "hits": pr["hits"]}, "player_hot": player_hot,
            "hot": seg_hot or player_hot, "hot_why": why, "cold": seg_cold,
            "cold_why": f"cold: only {seg['hits']} of {seg['n']} {side} leans in this market {'with' if agree else 'against'} the market have hit this season" if seg_cold else None}


def annotate(props: list, log_df: pd.DataFrame, dvp: dict, conf_records: dict = None, season: int = None) -> dict:
    """Adds p["lean"] to every prop in place. -> {calibration, walk_forward, n} for the snapshot."""
    cal = fit_calibration(log_df)
    segs = segment_records(log_df, season)
    n = 0
    for p in props:
        rec = (conf_records or {}).get((p.get("player_id"), p.get("market")))
        p["lean"] = lean_for(p, cal, segs, dvp, rec)
        n += p["lean"] is not None
    return {"calibration": cal, "walk_forward": walk_forward(log_df), "n": n,
            "segments": [{"market": m, "side": s, "agree": a, **v} for (m, s, a), v in sorted(segs.items(), key=lambda kv: -kv[1]["rate"])]}
