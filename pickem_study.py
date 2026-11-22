"""
pickem_study.py
===============
Would single-game stacks of the model's leans have paid? Replays this season's graded props week by week with NO look-ahead:
for each week, the confidence curve and the cold/hot records are fitted on EARLIER weeks only, every game's legs are built the way the
page builds them (a lean the market agrees with, confidence >= parlays.PICKEM_MIN_CONF, not a cold type), and the entries the page would
have offered (parlays._stack_candidates, correlations halved) are scored against what happened.

Two kinds of single-game entry are compared at 3 and 4 picks:
    stack   the combination parlays.py picks (best chance with the correlations counted)
    plain   simply the k most confident legs of the game, no correlation search
and a slate-wide one for scale: the k most confident legs overall, one per game.

    python pickem_study.py          # prints the table

What it can and cannot say: the log holds the sportsbook line, not Underdog's, so a leg's result is "did the lean win at the line the
model was pricing", and the payout multiples are the assumed ones in parlays.PICKEM_PAYOUTS. A season so far is ~80 games per entry
size, which moves a hit rate by about +/-5 points: this is a sanity check, not proof.
"""

import numpy as np
import pandas as pd

import lean
import parlays as pl


def _legs_for_week(te: pd.DataFrame, cal: dict, segs: dict) -> list:
    legs = []
    for r in te.itertuples():
        agree = r.side == ("over" if r.market_over > 0.5 else "under")
        conf = lean.confidence(cal, r.s, agree)
        seg = segs.get((r.market, r.side, agree))
        cold = bool(seg and seg["n"] >= lean.SEG_MIN_N and seg["rate"] <= lean.SEG_COLD)
        if not agree or cold or conf < pl.PICKEM_MIN_CONF:
            continue
        legs.append({"id": f"{r.player_id}|{r.market}|{r.side}", "player_id": r.player_id, "team": r.team, "game": r.game, "pos": r.pos,
                     "market": r.market, "side": r.side, "prob": conf, "score": conf - 0.5, "hit": int(r.hit)})
    legs.sort(key=lambda l: -l["score"])
    return legs


def stack_backtest(log_df: pd.DataFrame, corr=None, min_rows: int = lean.MIN_FIT_ROWS) -> dict:
    corr = corr or pl.CorrModel.load()
    cs = pl.ScaledCorr(corr)
    d = log_df[log_df["result"].isin(["over", "under"]) & (log_df["kind"] != "td") & (log_df["pos"] != "GAME")
               & log_df["model_over"].notna() & log_df["market_over"].notna() & (log_df["model_over"] != 0.5)].copy()
    if d.empty:
        return None
    d["side"] = np.where(d["model_over"] > 0.5, "over", "under")
    d["s"] = (d["model_over"] - 0.5).abs()
    d["hit"] = (d["side"] == d["result"]).astype(int)
    d = d.sort_values(["season", "week"])
    rows = []
    for (season, week), te in d.groupby(["season", "week"]):
        past = d[(d["season"] < season) | ((d["season"] == season) & (d["week"] < week))]
        if len(past) < min_rows:
            continue
        cal = lean.fit_calibration(past, min_rows)
        segs = lean.segment_records(past, int(season))
        legs = _legs_for_week(te, cal, segs)
        for k in pl.STACK_SIZES:
            stacks = {c[1][0]["game"]: c for c in pl._stack_candidates(legs, cs, k)}
            for game, (j, combo) in stacks.items():
                same, seen = [], set()                       # the plain set: the most confident legs, one per player (an entry cannot hold a player twice)
                for l in legs:
                    if l["game"] == game and l["player_id"] not in seen:
                        seen.add(l["player_id"])
                        same.append(l)
                    if len(same) == k:
                        break
                rows.append({"week": int(week), "k": k, "kind": "stack", "pred": j, "hit": int(all(l["hit"] for l in combo))})
                rows.append({"week": int(week), "k": k, "kind": "plain", "pred": float(np.prod([l["prob"] for l in same])),
                             "hit": int(all(l["hit"] for l in same))})
            one = {}
            for l in legs:
                one.setdefault(l["game"], l)
            top = sorted(one.values(), key=lambda l: -l["prob"])[:k]
            if len(top) == k:
                rows.append({"week": int(week), "k": k, "kind": "slate", "pred": float(np.prod([l["prob"] for l in top])),
                             "hit": int(all(l["hit"] for l in top))})
    if not rows:
        return None
    r = pd.DataFrame(rows)
    out = []
    for (kind, k), g in r.groupby(["kind", "k"]):
        mult = pl.PICKEM_PAYOUTS["standard"].get(int(k))
        out.append({"kind": kind, "k": int(k), "entries": int(len(g)), "hits": int(g["hit"].sum()), "rate": round(float(g["hit"].mean()), 4),
                    "predicted": round(float(g["pred"].mean()), 4), "multiplier": mult,
                    "breakeven": None if not mult else round(1 / mult, 4)})
    return {"weeks": sorted(r["week"].unique().tolist()), "rows": out}


if __name__ == "__main__":
    import props_tracker
    res = stack_backtest(props_tracker._load())
    if not res:
        print("not enough graded props yet")
    else:
        print(f"weeks {res['weeks']}")
        print(f"{'kind':8}{'k':>3}{'entries':>9}{'hits':>6}{'rate':>8}{'predicted':>11}{'breakeven':>11}{'ROI at assumed x':>18}")
        for x in res["rows"]:
            roi = "" if not x["multiplier"] else f"{x['rate'] * x['multiplier'] - 1:+.0%}"
            print(f"{x['kind']:8}{x['k']:>3}{x['entries']:>9}{x['hits']:>6}{x['rate']:>8.1%}{x['predicted']:>11.1%}{(x['breakeven'] or 0):>11.1%}{roi:>18}")
