"""
review_extras.py
================
Two more parts of the Monday report, kept out of weekly_report.py so that file stays about the core records:

* the week's TOP 10 prop picks (the ones the page ranked by expected value): how they did, in dollars at the prices logged
  and against what their stated chances expected, plus the season-to-date record of every top-10 pick so far;
* STREAKS: players whose prop the model's lean has been right week after week ("the model called Josh Allen's pass attempts
  right four weeks running"), each next to how many streaks that long chance alone would produce, because with several
  hundred player-market series tracked a handful of three-week streaks happen on luck. The report says which it is.

Everything is read from tracking/props_log.csv (one row per prop per week, graded after the games).
"""

import math

import numpy as np
import pandas as pd

LABELS = {
    "player_pass_yds": "Pass Yds", "player_pass_tds": "Pass TDs", "player_pass_attempts": "Pass Att", "player_pass_completions": "Completions",
    "player_pass_interceptions": "Interceptions", "player_rush_yds": "Rush Yds", "player_rush_attempts": "Rush Att",
    "player_receptions": "Receptions", "player_reception_yds": "Rec Yds", "player_rush_reception_yds": "Rush+Rec Yds",
    "player_reception_longest": "Long Rec", "player_rush_longest": "Long Rush", "player_anytime_td": "Anytime TD",
}
MIN_STREAK = 3
MAX_WEEK_GAP = 2                # consecutive for a player means the next logged week, allowing one bye in between
CHANCE_HIT = 0.55               # how often the model's lean is right by luck-plus-a-little: used for the "expected by chance" figure


def _props(log: pd.DataFrame, season: int) -> pd.DataFrame:
    d = log[(log["season"] == season) & log["market"].astype(str).str.startswith("player_")]
    if "pos" in d:
        d = d[d["pos"] != "GAME"]
    return d[d["result"].isin(["over", "under"])].copy()


def _side(row) -> str:
    if row.get("pick") in ("over", "under"):
        return row["pick"]
    return "over" if row["model_over"] > 0.5 else "under"


def _dec(price) -> float:
    price = float(price)
    return 1 + (price / 100 if price > 0 else 100 / -price)


def top10_section(log: pd.DataFrame, season: int, week: int):
    """The week's top-10 picks graded, their record and profit, and the season to date. None if the log has no top ranks."""
    if "top_rank" not in log:
        return None
    d = _props(log, season)
    d = d[d["top_rank"].notna()].copy()
    if d.empty:
        return None
    d["side"] = d.apply(_side, axis=1)
    d["hit"] = d["side"] == d["result"]
    d["p"] = np.where(d["side"] == "over", d["model_over"], 1 - d["model_over"])

    def profit(r):
        if pd.isna(r.get("pick_price")):
            return np.nan
        return _dec(r["pick_price"]) - 1 if r["hit"] else -1.0
    d["profit"] = d.apply(profit, axis=1)

    def summarize(x):
        n = len(x)
        w = int(x["hit"].sum())
        priced = x["profit"].dropna()
        return {"n": n, "wins": w, "losses": n - w, "expected_wins": float(x["p"].sum()), "profit": float(priced.sum()) if len(priced) else None,
                "n_priced": int(len(priced))}

    wk = d[d["week"] == week].sort_values("top_rank")
    if wk.empty:
        return None
    items = [{"rank": int(r["top_rank"]), "player": r["player"], "market": LABELS.get(r["market"], r["market"]), "side": r["side"],
              "line": float(r["line"]), "actual": None if pd.isna(r["actual"]) else float(r["actual"]), "result": "win" if r["hit"] else "loss",
              "prob": float(r["p"]), "profit": None if pd.isna(r["profit"]) else float(r["profit"])} for _, r in wk.iterrows()]
    by_week = {int(w): summarize(g) for w, g in d.groupby("week")}
    out = {"week": summarize(wk), "season": summarize(d), "weeks": len(by_week), "items": items}
    out["writeup"] = top10_blurb(out)
    return out


def top10_blurb(sec: dict) -> str:
    w, s = sec["week"], sec["season"]
    parts = [f"The week's top 10 prop picks went {w['wins']}-{w['losses']}"
             + (f", about {w['expected_wins']:.1f} wins by their stated chances" if w["n"] else "") + "."]
    if w["profit"] is not None and w["n_priced"]:
        parts.append(f"At the prices logged, $1 flat on each returned {w['profit']:+.2f} ({w['profit'] / w['n_priced']:+.0%} per pick).")
    hits = [i for i in sec["items"] if i["result"] == "win"]
    misses = [i for i in sec["items"] if i["result"] == "loss"]
    if hits:
        b = max(hits, key=lambda i: abs((i["actual"] or 0) - i["line"]))
        parts.append(f"Best call: {b['player']} {b['market']} {b['side']} {b['line']:g}, finished {b['actual']:g}.")
    if misses:
        m = max(misses, key=lambda i: abs((i["actual"] or 0) - i["line"]))
        parts.append(f"Biggest miss: {m['player']} {m['market']} {m['side']} {m['line']:g}, finished {m['actual']:g}.")
    if s["n"] > w["n"]:
        parts.append(f"Season to date over {sec['weeks']} weeks: {s['wins']}-{s['losses']} ({s['wins'] / s['n']:.0%}), against {s['expected_wins']:.1f} expected wins"
                     + (f" and {s['profit']:+.2f} units at logged prices." if s["profit"] is not None else "."))
    parts.append("Ten picks a week is a small sample: judge the model on the season line, not one week.")
    return " ".join(parts)


def streaks(log: pd.DataFrame, season: int, week: int, min_len: int = MIN_STREAK) -> dict:
    """
    Current streaks of the model's lean being right on one player's market in consecutive weeks, through `week`.
    -> {"current": [...], "n_series", "observed", "expected_by_chance"} (current sorted longest first).
    """
    d = _props(log, season)
    d = d[(d["week"] <= week) & (d["model_over"] != 0.5)].copy()
    if d.empty:
        return {"current": [], "n_series": 0, "observed": 0, "expected_by_chance": 0.0, "writeup": "No graded props yet this season."}
    d["lean"] = np.where(d["model_over"] > 0.5, "over", "under")
    d["correct"] = d["lean"] == d["result"]
    d = d.sort_values(["player_id", "market", "week"]).drop_duplicates(["player_id", "market", "week"], keep="last")
    cur, expected, n_series = [], 0.0, 0
    for (pid, market), g in d.groupby(["player_id", "market"]):
        g = g.reset_index(drop=True)
        if len(g) < min_len:
            continue
        n_series += 1
        expected += CHANCE_HIT ** min_len
        run = []
        for i in range(len(g) - 1, -1, -1):
            if not g.loc[i, "correct"]:
                break
            if run and run[-1]["week"] - g.loc[i, "week"] > MAX_WEEK_GAP:
                break
            r = g.loc[i]
            run.append({"week": int(r["week"]), "side": r["lean"], "line": float(r["line"]), "actual": None if pd.isna(r["actual"]) else float(r["actual"])})
        if len(run) >= min_len and run[0]["week"] == int(g["week"].max()) and int(g["week"].max()) >= week - MAX_WEEK_GAP + 1:
            cur.append({"player": g.loc[0, "player"], "team": g.loc[0, "team"], "market": market, "label": LABELS.get(market, market),
                        "length": len(run), "games": list(reversed(run))})
    cur.sort(key=lambda x: (-x["length"], x["player"]))
    sec = {"current": cur[:12], "n_series": n_series, "observed": len(cur), "expected_by_chance": expected}
    sec["writeup"] = streaks_blurb(sec, min_len)
    return sec


def streaks_blurb(sec: dict, min_len: int = MIN_STREAK) -> str:
    if not sec["n_series"]:
        return f"No player-market has {min_len} graded weeks yet, so there are no streaks to report."
    if not sec["current"]:
        return (f"No player's market has the model's lean right {min_len} weeks running as of this week "
                f"({sec['n_series']} player-markets have been tracked that long; about {sec['expected_by_chance']:.0f} such streaks would be expected on luck).")
    top = sec["current"][:3]
    names = "; ".join(f"{s['player']} {s['label']} ({s['length']} in a row)" for s in top)
    over = sec["observed"] - sec["expected_by_chance"]
    verdict = ("about what chance alone gives" if abs(over) <= max(2.0, 0.3 * sec["expected_by_chance"])
               else "more than chance alone gives" if over > 0 else "fewer than chance alone gives")
    return (f"The model's lean has been right {min_len}+ weeks running on {sec['observed']} player-markets: {names}. "
            f"Of {sec['n_series']} player-markets tracked that long, about {sec['expected_by_chance']:.0f} such streaks would come from luck ({verdict}), "
            f"so a streak is a talking point, not a signal; the model's record on a player has not predicted his next result.")
