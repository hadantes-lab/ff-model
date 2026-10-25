"""
parlay_track.py
===============
Log every parlay the page recommends and grade it afterwards, so the weekly report can say how they did and the results
build up like everything else.

A parlay is graded from its legs: a prop leg from the graded prop in tracking/props_log.csv (its actual stat against the
line the parlay used), a moneyline leg from the final score, a teaser leg from the final margin against the teased line.
A leg that was a push or a DNP is dropped and the payout repriced; a parlay with any losing leg lost; one with every leg
settled and no loss cashed.

What the report reads from this: parlays cashed against how many the stated joint probabilities expected, how many LEGS
hit against how many were expected (a much steadier measure than a handful of parlays, since each parlay is mostly a
coin flip repeated four or five times), and the profit per $1 at the prices shown.
"""

import json
import math
import pathlib

import numpy as np
import pandas as pd

import parlays as pl

PARLAY_LOG = pathlib.Path(__file__).parent / "tracking" / "parlay_log.csv"
COLUMNS = ["season", "week", "id", "kind", "n_legs", "price", "decimal", "joint_prob", "ev", "legs", "first_logged", "logged_at"]
LEG_KEYS = ("id", "kind", "player_id", "player", "team", "opp", "game", "market", "label", "side", "line", "price", "prob", "spread", "teased", "week")
TEASER_DECIMAL = pl.TEASER_PRICES["2-team at -120"]
KINDS = {"props": "Prop parlays", "ml": "Moneyline parlays", "teaser": "Wong teasers (pairs)"}


def _leg(l: dict) -> dict:
    return {k: l.get(k) for k in LEG_KEYS if l.get(k) is not None}


def rows_from_block(block: dict, season: int, week: int, now: str) -> list:
    """Every recommended parlay (props, moneyline, and each Wong teaser pair) as a log row."""
    rows = []

    def add(kind, legs, price, decimal, joint, ev, wk=None):
        rows.append({"season": season, "week": int(wk or week), "id": pl._parlay_id(legs), "kind": kind, "n_legs": len(legs),
                     "price": price, "decimal": round(decimal, 4), "joint_prob": round(float(joint), 5), "ev": round(float(ev), 5),
                     "legs": json.dumps([_leg(l) for l in legs]), "first_logged": now, "logged_at": now})

    for p in (block.get("props") or {}).get("parlays", []):
        add("props", p["legs"], p["price"], p["decimal"], p["joint_prob"], p["ev"])
    for p in (block.get("ml") or {}).get("parlays", []):
        wk = min([l.get("week") for l in p["legs"] if l.get("week")] or [week])
        add("ml", p["legs"], p["price"], p["decimal"], p["joint_prob"], p["ev"], wk)
    t = block.get("teasers") or {}
    seen = set()
    pairs = list(t.get("pairs") or []) + list(((t.get("round_robin") or {}).get("pairs")) or [])
    for pr in pairs:
        pid = pl._parlay_id(pr["legs"])
        if pid in seen:
            continue
        seen.add(pid)
        wk = min([l.get("week") for l in pr["legs"] if l.get("week")] or [week])
        add("teaser", pr["legs"], -120, TEASER_DECIMAL, pr["prob"], pr["ev"]["2-team at -120"], wk)
    return rows


def load(path=None) -> pd.DataFrame:
    path = pathlib.Path(path or PARLAY_LOG)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path)


def log_parlays(rows: list, path=None) -> int:
    """Upsert by (season, week, id): a re-run before kickoff refreshes the line and price, and keeps the first-logged time."""
    path = pathlib.Path(path or PARLAY_LOG)
    if not rows:
        return 0
    old = load(path)
    new = pd.DataFrame(rows, columns=COLUMNS)
    if not old.empty:
        first = {(r.season, r.week, r.id): r.first_logged for r in old.itertuples()}
        new["first_logged"] = [first.get((r.season, r.week, r.id), r.first_logged) for r in new.itertuples()]
    both = pd.concat([old, new], ignore_index=True).drop_duplicates(subset=["season", "week", "id"], keep="last")
    path.parent.mkdir(exist_ok=True)
    both.to_csv(path, index=False)
    return len(new)


# ---- grading -----------------------------------------------------------------------------------
def grade_leg(leg: dict, season: int, week: int, props_log: pd.DataFrame, sched: pd.DataFrame) -> str:
    """'win' | 'loss' | 'push' | 'void' | 'pending' for one leg."""
    kind = leg.get("kind")
    if kind == "prop":
        r = props_log[(props_log["season"] == season) & (props_log["player_id"] == leg["player_id"]) & (props_log["market"] == leg["market"])
                      & (props_log["week"] == week)]
        if r.empty:
            return "pending"
        r = r.iloc[-1]
        if r["result"] == "dnp":
            return "void"
        if pd.isna(r["actual"]) or r["result"] not in ("over", "under", "push"):
            return "pending"
        a, line = float(r["actual"]), float(leg["line"])
        if a == line:
            return "push"
        return "win" if (a > line) == (leg["side"] == "over") else "loss"
    away, home = leg["game"].split(" @ ")
    g = sched[(sched["season"] == season) & (sched["home_team"] == home) & (sched["away_team"] == away)]
    g = g[g["week"] == week] if "week" in g and (g["week"] == week).any() else g
    if g.empty or pd.isna(g.iloc[0]["home_score"]):
        return "pending"
    g = g.iloc[0]
    margin = float(g["home_score"] - g["away_score"])
    team_margin = margin if leg["team"] == home else -margin
    if kind == "ml":
        return "push" if team_margin == 0 else ("win" if team_margin > 0 else "loss")
    if kind == "wong":
        edge = team_margin + float(leg["spread"]) + pl.TEASER_POINTS
        return "push" if edge == 0 else ("win" if edge > 0 else "loss")
    return "pending"


def grade_parlay(row, props_log: pd.DataFrame, sched: pd.DataFrame) -> dict:
    legs = json.loads(row.legs)
    res = [grade_leg(l, int(row.season), int(row.week), props_log, sched) for l in legs]
    out = {"legs": legs, "leg_results": res, "legs_won": res.count("win"), "legs_lost": res.count("loss"),
           "legs_pending": res.count("pending"), "legs_dropped": res.count("push") + res.count("void")}
    if "loss" in res:
        out.update(status="lost", profit=-1.0)
    elif "pending" in res:
        out.update(status="pending", profit=None)
    elif res.count("win") == 0:
        out.update(status="push", profit=0.0)
    else:
        if row.kind == "teaser":
            d = TEASER_DECIMAL if res.count("win") == len(res) else 1.0
        else:
            d = float(np.prod([pl.dec(l["price"]) for l, r in zip(legs, res) if r == "win"]))
        out.update(status="cashed" if d > 1 else "push", profit=d - 1)
    return out


def section(parlay_log: pd.DataFrame, props_log: pd.DataFrame, sched: pd.DataFrame, season: int, week: int) -> dict:
    """The report's parlay section for one week."""
    d = parlay_log[(parlay_log["season"] == season) & (parlay_log["week"] == week)] if parlay_log is not None and len(parlay_log) else pd.DataFrame()
    groups, items = {}, []
    for r in d.itertuples():
        g = grade_parlay(r, props_log, sched)
        legs_p = [l["prob"] for l in g["legs"] if l.get("prob") is not None]
        k = groups.setdefault(r.kind, {"kind": r.kind, "title": KINDS.get(r.kind, r.kind), "n": 0, "cashed": 0, "lost": 0, "pending": 0, "push": 0,
                                       "expected_cash": 0.0, "var": 0.0, "profit": 0.0, "legs_won": 0, "legs_graded": 0, "expected_legs": 0.0,
                                       "price_sum": 0.0})
        if g["status"] == "pending":
            k["pending"] += 1
        else:
            k["n"] += 1
            k[{"cashed": "cashed", "lost": "lost", "push": "push"}[g["status"]]] += 1
            k["expected_cash"] += float(r.joint_prob)
            k["var"] += float(r.joint_prob) * (1 - float(r.joint_prob))
            k["profit"] += g["profit"] or 0.0
        for leg, res in zip(g["legs"], g["leg_results"]):
            if res in ("win", "loss"):
                k["legs_graded"] += 1
                k["legs_won"] += res == "win"
                k["expected_legs"] += float(leg.get("prob") or 0.0)
        items.append({"kind": r.kind, "id": r.id, "price": int(r.price), "status": g["status"], "joint_prob": float(r.joint_prob), "ev": float(r.ev),
                      "legs_won": g["legs_won"], "legs_lost": g["legs_lost"], "legs_pending": g["legs_pending"],
                      "legs": [{"label": (l.get("player") or l.get("team")) + (f" {l.get('label')} {l.get('side')} {l.get('line')}" if l.get("kind") == "prop"
                                else (" ML" if l.get("kind") == "ml" else f" teased to {l.get('teased', 0):+g}")), "result": res}
                               for l, res in zip(g["legs"], g["leg_results"])]})
    for k in groups.values():
        k["z"] = (k["cashed"] - k["expected_cash"]) / math.sqrt(k["var"]) if k["var"] > 0 else None
    sec = {"key": "parlays", "title": "Parlays", "groups": groups, "items": items, "n_logged": len(items)}
    sec["writeup"] = blurb(sec)
    return sec


def blurb(sec: dict) -> str:
    if not sec["n_logged"]:
        return "No parlays were logged for this week (parlay logging started after it, or the slate was too thin to build any)."
    parts = []
    for k in sec["groups"].values():
        if not (k["n"] or k["pending"]):
            continue
        txt = f"{k['title']}: {k['cashed']} of {k['n']} settled cashed" + (f" ({k['pending']} still pending)" if k["pending"] else "")
        if k["n"]:
            txt += f"; the stated chances expected {k['expected_cash']:.1f}"
        if k["legs_graded"]:
            txt += f". Individual legs: {k['legs_won']} of {k['legs_graded']} hit against {k['expected_legs']:.1f} expected"
        if k["n"]:
            txt += f". Flat $1 on each returned {k['profit']:+.2f} ({k['profit'] / k['n']:+.0%} per parlay)"
        parts.append(txt + ".")
    parts.append("A parlay needs every leg, so a week of four or five of them says very little; the leg-level numbers and the season-to-date table are the ones to read.")
    return " ".join(parts)
