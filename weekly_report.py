"""
weekly_report.py
================
The Monday report: how every kind of pick did last week, a short write-up per market, and what the results
are teaching the model.

WHAT IT GRADES (from tracking/props_log.csv, which records every prediction BEFORE the game and is graded after)
------------------------------------------------------------------------------------------------------------
* Moneyline winners (straight up), by tier S-F. Uses the picks we actually logged (`game_su`); for a week from
  before they were logged it reconstructs them from the closing spread and says so.
* Game spread picks, game total picks, and the power-rating spread (`game_rank_spread`).
* Player props, per market, two ways: the picks the page recommended (positive expected value after shrinking
  toward the market, or the week's top 10), and every prop where the model's own raw probability for its side
  was at least HIGH_CONF ("high confidence").

For each group: right/wrong/pushes, the win rate, and the win rate the model's own probabilities EXPECTED, so
"7-3" can be read as lucky, unlucky or about right (a z-score against the probabilities it stated). Where the
projection was logged, how close the model's number and the market's line each landed to the real result.

THE WRITE-UPS
-------------
Generated from the numbers, in plain words, with one rule: they say what happened and whether it is within normal
variation. A handful of picks is never called skill or failure. See `blurb`.

THE LEARNING LOOP
-----------------
Each run upserts the week's per-market results into tracking/weekly_summary.csv (so season-to-date and trends are
exact), and improve.py then re-measures how much to trust the model per market from ALL the evidence (the 2024-25
backtest plus everything tracked since), changing a weight only when the same strict rules that set it say so.
The report lists exactly what, if anything, changed and why.

    python weekly_report.py                 # the latest week with graded games
    python weekly_report.py --week 4        # a specific week
    python weekly_report.py --html          # also write reports/week-NN.html and web/weekly_report.html
"""

import argparse
import datetime
import html
import json
import math
import pathlib
import sys

import numpy as np
import pandas as pd

import straight_up

MIN_FOR_SHARPER = 30               # games needed before the report says whether the model or the market was sharper
HIGH_CONF = 0.60                    # the model's raw probability for its side at or above this = "high confidence"
SUMMARY_FILE = pathlib.Path(__file__).parent / "tracking" / "weekly_summary.csv"
REPORT_DIR = pathlib.Path(__file__).parent / "reports"
LATEST_HTML = pathlib.Path(__file__).parent / "web" / "weekly_report.html"
SUMMARY_COLUMNS = ["season", "week", "group", "n", "wins", "losses", "pushes", "expected_wins", "var", "brier_model", "brier_market",
                   "n_acc", "mae_model", "mae_market", "generated"]

GAME_GROUPS = {"game_spread": "Game spread picks", "game_total": "Game total picks", "game_rank_spread": "Power-rating spread picks"}
PROP_LABELS = {"player_pass_yds": "Passing yards", "player_pass_tds": "Passing TDs", "player_pass_attempts": "Pass attempts",
               "player_pass_completions": "Pass completions", "player_pass_interceptions": "Interceptions",
               "player_rush_yds": "Rushing yards", "player_rush_attempts": "Rush attempts", "player_receptions": "Receptions",
               "player_reception_yds": "Receiving yards", "player_rush_reception_yds": "Rush + receiving yards",
               "player_reception_longest": "Longest reception", "player_rush_longest": "Longest rush"}


# ---- grading a group of picks ---------------------------------------------------------
def _side_prob(df: pd.DataFrame) -> pd.Series:
    """The model's stated probability for the side it leaned (>= 0.5 by construction)."""
    return np.maximum(df["model_over"], 1 - df["model_over"])


def _leaned_over(df: pd.DataFrame) -> pd.Series:
    return df["model_over"] > 0.5


def outcome_of(side_over: bool, result: str) -> str:
    """'win' / 'loss' / 'push' for a pick on the over (True) or under (False) side."""
    if result == "push":
        return "push"
    return "win" if (side_over and result == "over") or (not side_over and result == "under") else "loss"


def record(df: pd.DataFrame, side_over=None, probs=None) -> dict:
    """
    Right/wrong tally for rows already graded. `side_over` (bool Series) is the side picked on each row (default: the
    model's raw lean); `probs` the stated probability of that side (default: the raw lean's). Pushes are excluded from
    the rate. -> n, wins, losses, pushes, win_rate, expected_wins, expected_rate, z, brier_model, brier_market.
    """
    if df is None or df.empty:
        return {"n": 0, "wins": 0, "losses": 0, "pushes": 0, "win_rate": None, "expected_wins": 0.0, "expected_rate": None,
                "var": 0.0, "z": None, "brier_model": None, "brier_market": None}
    d = df.copy()
    so = _leaned_over(d) if side_over is None else side_over
    pr = _side_prob(d) if probs is None else probs
    res = [outcome_of(bool(o), r) for o, r in zip(so, d["result"])]
    d["_o"] = res
    live = d[d["_o"] != "push"]
    wins, losses = int((live["_o"] == "win").sum()), int((live["_o"] == "loss").sum())
    n = wins + losses
    p = pr.loc[live.index].astype(float)
    exp, var = float(p.sum()), float((p * (1 - p)).sum())
    z = None if var <= 0 else (wins - exp) / math.sqrt(var)
    out = {"n": n, "wins": wins, "losses": losses, "pushes": int((d["_o"] == "push").sum()),
           "win_rate": wins / n if n else None, "expected_wins": exp, "expected_rate": exp / n if n else None, "var": var, "z": z,
           "brier_model": None, "brier_market": None}
    if "market_over" in live and live["market_over"].notna().all() and n:
        y = (live["result"] == "over").astype(float)
        out["brier_model"] = float(((live["model_over"] - y) ** 2).mean())
        out["brier_market"] = float(((live["market_over"] - y) ** 2).mean())
    return out


def accuracy(df: pd.DataFrame) -> dict:
    """Where the projection was logged: typical miss of the model's number and of the market's line vs the real result."""
    if df is None or df.empty or "mean" not in df:
        return {"n_acc": 0, "mae_model": None, "mae_market": None}
    d = df.dropna(subset=["mean", "actual"])
    d = d[d["kind"] != "td"] if "kind" in d else d
    if d.empty:
        return {"n_acc": 0, "mae_model": None, "mae_market": None}
    return {"n_acc": int(len(d)), "mae_model": float((d["actual"].astype(float) - d["mean"].astype(float)).abs().mean()),
            "mae_market": float((d["actual"].astype(float) - d["line"].astype(float)).abs().mean())}


def luck_phrase(z) -> str:
    """Plain words for how a result compares with what the model's own probabilities expected."""
    if z is None:
        return "too few picks to compare with expectation"
    if abs(z) < 1.0:
        return "right about where its own probabilities expected"
    if abs(z) < 2.0:
        return "a bit " + ("better" if z > 0 else "worse") + " than its probabilities expected, within normal variation"
    return ("clearly better" if z > 0 else "clearly worse") + " than its probabilities expected"


def _rec_str(r: dict) -> str:
    return f"{r['wins']}-{r['losses']}" + (f"-{r['pushes']}" if r["pushes"] else "")


def _pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.0f}%"


# ---- building the report ---------------------------------------------------------------
def _graded(log: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    d = log[(log["season"] == season) & (log["week"] == week) & log["result"].isin(["over", "under", "push"])]
    return d.copy()


def moneyline_section(log: pd.DataFrame, sched: pd.DataFrame, season: int, week: int) -> dict:
    """Straight-up picks by tier. Logged picks if there are any for this week, else rebuilt from the closing spread."""
    d = _graded(log, season, week)
    d = d[d["market"] == "game_su"]
    source = "logged"
    entries = []
    if not d.empty:
        for r in d.itertuples(index=False):
            p_home = float(r.model_over)
            pick_home = p_home >= 0.5
            prob = p_home if pick_home else 1 - p_home
            home_won = r.result == "over"
            tie = r.result == "push"
            entries.append({"game": r.game, "pick": r.team if pick_home else r.opp, "prob": prob, "tier": straight_up.tier_of(prob),
                            "result": "push" if tie else ("win" if home_won == pick_home else "loss"), "score": None})
    elif sched is not None and not sched.empty:
        source = "reconstructed"
        for e in straight_up.build_slate(sched[sched["season"] == season], week, season=season):
            if e["status"] != "final":
                continue
            entries.append({"game": e["game"], "pick": e["pick"], "prob": e["pick_prob"], "tier": e["tier"],
                            "result": "push" if e["correct"] is None else ("win" if e["correct"] else "loss"),
                            "score": f"{e['home_score']}-{e['away_score']}"})
    entries.sort(key=lambda e: (straight_up.TIER_ORDER.index(e["tier"]), -e["prob"], e["game"]))
    tiers = []
    for t in straight_up.TIER_ORDER:
        es = [e for e in entries if e["tier"] == t]
        if not es:
            continue
        w, l = sum(e["result"] == "win" for e in es), sum(e["result"] == "loss" for e in es)
        tiers.append({"tier": t, "range": straight_up.tier_range(t), "n": w + l, "wins": w, "losses": l,
                      "expected_wins": float(sum(e["prob"] for e in es if e["result"] != "push"))})
    w, l = sum(e["result"] == "win" for e in entries), sum(e["result"] == "loss" for e in entries)
    exp = float(sum(e["prob"] for e in entries if e["result"] != "push"))
    var = float(sum(e["prob"] * (1 - e["prob"]) for e in entries if e["result"] != "push"))
    return {"key": "moneyline", "title": "Moneyline winners (straight up)", "source": source, "entries": entries, "tiers": tiers,
            "n": w + l, "wins": w, "losses": l, "pushes": sum(e["result"] == "push" for e in entries), "win_rate": w / (w + l) if w + l else None,
            "expected_wins": exp, "var": var, "z": (w - exp) / math.sqrt(var) if var > 0 else None}


def game_section(log: pd.DataFrame, season: int, week: int, market: str) -> dict:
    d = _graded(log, season, week)
    d = d[d["market"] == market]
    # the pick the page showed: the recommended side when there was one, else the model's raw lean
    side_over = d["pick"].map({"over": True, "under": False})
    side_over = side_over.where(side_over.notna(), _leaned_over(d)).astype(bool)
    probs = pd.Series(np.where(side_over, d["model_over"], 1 - d["model_over"]), index=d.index)
    probs = probs.clip(lower=0.5)
    rec = record(d, side_over, probs)
    items = []
    for i, r in d.iterrows():
        pick_txt = ("home cover" if side_over[i] else "away cover") if market != "game_total" else ("Over" if side_over[i] else "Under")
        items.append({"game": r["game"], "line": float(r["line"]), "pick": pick_txt, "prob": float(probs[i]),
                      "actual": None if pd.isna(r["actual"]) else float(r["actual"]), "result": outcome_of(bool(side_over[i]), r["result"])})
    side_share = float(side_over.mean()) if len(side_over) else None             # share of picks on the over / home side
    return dict(rec, key=market, title=GAME_GROUPS[market], items=items, side_share=side_share, **accuracy(d))


def prop_section(log: pd.DataFrame, season: int, week: int, market: str) -> dict:
    d = _graded(log, season, week)
    d = d[(d["market"] == market) & (d["pos"] != "GAME")]
    lean_over = _leaned_over(d)
    rec_all = record(d)
    hi = d[_side_prob(d) >= HIGH_CONF]
    rec_hi = record(hi)
    if "pick_ev" in d:
        recommended = d[d["pick"].notna() & (d["pick_ev"] > 0)]
    else:
        recommended = d.iloc[0:0]
    if "top_rank" in d:
        recommended = pd.concat([recommended, d[d["top_rank"].notna()]]).drop_duplicates(subset=["player_id", "market"])
    rec_pick = record(recommended, recommended["pick"].map({"over": True, "under": False}).astype(bool),
                      pd.Series(np.where(recommended["pick"] == "over", recommended["model_over"], 1 - recommended["model_over"]),
                                index=recommended.index).clip(lower=0.5)) if len(recommended) else record(recommended)
    top = hi.assign(_p=_side_prob(hi)).sort_values("_p", ascending=False).head(6)
    items = [{"player": r["player"], "game": r["game"], "line": float(r["line"]), "pick": "Over" if r["model_over"] > 0.5 else "Under",
              "prob": float(r["_p"]), "actual": None if pd.isna(r["actual"]) else float(r["actual"]),
              "result": outcome_of(r["model_over"] > 0.5, r["result"])} for _, r in top.iterrows()]
    return {"key": market, "title": PROP_LABELS.get(market, market), "all": rec_all, "high": rec_hi, "recommended": rec_pick,
            "items": items, **accuracy(d)}


def build_report(log: pd.DataFrame, sched: pd.DataFrame, season: int, week: int, history: pd.DataFrame = None,
                 learning: list = None, generated: str = None) -> dict:
    """The whole report for one week as plain data (see render_markdown / render_html for the readable forms)."""
    sections = [moneyline_section(log, sched, season, week)]
    sections += [game_section(log, season, week, m) for m in GAME_GROUPS]
    g = _graded(log, season, week)
    prop_markets = sorted({m for m in g["market"] if m.startswith("player_")}, key=lambda m: list(PROP_LABELS).index(m) if m in PROP_LABELS else 99)
    props = [prop_section(log, season, week, m) for m in prop_markets]
    rep = {"season": season, "week": week, "generated": generated or datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "games_graded": int(len(g[g["market"].isin(GAME_GROUPS)]) // max(1, len(GAME_GROUPS))), "sections": sections, "props": props,
           "learning": learning or []}
    # the log only holds games that have been graded; say plainly what is still to come
    due = log[(log["season"] == season) & (log["week"] == week) & log["result"].isna() & log["market"].isin(list(GAME_GROUPS) + ["game_su"])]
    rep["games_pending"] = int(due["game"].nunique())
    rep["season_to_date"] = season_to_date(history, rep) if history is not None else {}
    for sec in sections + props:
        sec["writeup"] = blurb(sec)
    rep["headline"] = headline(rep)
    return rep


# ---- season to date ---------------------------------------------------------------------
def summary_rows(rep: dict) -> list:
    """The week's per-group numbers as rows for tracking/weekly_summary.csv."""
    rows = []
    gen = rep["generated"]

    def row(group, r, acc=None):
        acc = acc or {}
        rows.append({"season": rep["season"], "week": rep["week"], "group": group, "n": r.get("n", 0), "wins": r.get("wins", 0),
                     "losses": r.get("losses", 0), "pushes": r.get("pushes", 0), "expected_wins": round(r.get("expected_wins", 0.0), 3),
                     "var": round(r.get("var", 0.0), 3), "brier_model": r.get("brier_model"), "brier_market": r.get("brier_market"),
                     "n_acc": acc.get("n_acc", 0), "mae_model": acc.get("mae_model"), "mae_market": acc.get("mae_market"), "generated": gen})
    for s in rep["sections"]:
        row(s["key"], s, s)
    for p in rep["props"]:
        row(p["key"] + ":all", p["all"], p)
        row(p["key"] + ":high", p["high"])
        row(p["key"] + ":picks", p["recommended"])
    return rows


def upsert_summary(rows: list, path=None) -> pd.DataFrame:
    """One row per (season, week, group); re-running a week replaces it."""
    path = pathlib.Path(path or SUMMARY_FILE)
    new = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    old = pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame(columns=SUMMARY_COLUMNS)
    both = pd.concat([old, new], ignore_index=True).drop_duplicates(subset=["season", "week", "group"], keep="last")
    both = both.sort_values(["season", "week", "group"]).reset_index(drop=True)
    path.parent.mkdir(exist_ok=True)
    both.to_csv(path, index=False)
    return both


def season_to_date(history: pd.DataFrame, rep: dict) -> dict:
    """{group: {n, wins, losses, win_rate, expected_wins, z, weeks}} through this week, from the summary history."""
    if history is None or history.empty:
        return {}
    h = history[(history["season"] == rep["season"]) & (history["week"] <= rep["week"])]
    out = {}
    for g, d in h.groupby("group"):
        n, w = int(d["n"].sum()), int(d["wins"].sum())
        exp, var = float(d["expected_wins"].sum()), float(d["var"].sum())
        out[g] = {"n": n, "wins": w, "losses": int(d["losses"].sum()), "win_rate": w / n if n else None, "expected_wins": exp,
                  "z": (w - exp) / math.sqrt(var) if var > 0 else None, "weeks": int(d["week"].nunique())}
    return out


# ---- the words ---------------------------------------------------------------------------
def blurb(sec: dict) -> str:
    """A short plain-language write-up for one market, from its numbers."""
    k = sec["key"]
    if k == "moneyline":
        if not sec["n"]:
            return "No finished games to grade yet."
        best = [t for t in sec["tiers"] if t["tier"] in ("S", "A", "B")]
        top_n = sum(t["n"] for t in best)
        top_w = sum(t["wins"] for t in best)
        txt = (f"Went {sec['wins']}-{sec['losses']}" + (f"-{sec['pushes']}" if sec["pushes"] else "") + f" ({_pct(sec['win_rate'])}) on straight-up winners; "
               f"the market's probabilities expected about {sec['expected_wins']:.1f} wins, so that is {luck_phrase(sec['z'])}.")
        if top_n:
            txt += f" The clear favorites (S, A and B tiers) went {top_w}-{top_n - top_w}."
        misses = [e for e in sec["entries"] if e["result"] == "loss"]
        if misses:
            worst = max(misses, key=lambda e: e["prob"])
            txt += f" Biggest miss: {worst['pick']} ({worst['tier']} tier, {_pct(worst['prob'])}) in {worst['game']}."
        if sec["source"] == "reconstructed":
            txt += " (These picks were rebuilt from closing spreads, since this week predates logging them.)"
        return txt
    if k in GAME_GROUPS:
        if not sec["n"]:
            return "No graded games yet for this market."
        txt = (f"{_rec_str(sec)} ({_pct(sec['win_rate'])}) against the posted line. The model's own probabilities expected "
               f"{sec['expected_wins']:.1f} wins, so that is {luck_phrase(sec['z'])}.")
        if sec.get("brier_model") is not None:
            better = sec["brier_model"] < sec["brier_market"]
            if sec["n"] >= MIN_FOR_SHARPER:
                txt += (f" On probability quality the model scored {sec['brier_model']:.3f} against the market's {sec['brier_market']:.3f} "
                        f"(lower is better), so the {'model' if better else 'market'} was the sharper of the two.")
            else:
                txt += (f" On probability quality the model scored {sec['brier_model']:.3f} against the market's {sec['brier_market']:.3f}, "
                        f"but with only {sec['n']} games that is too few to say which was sharper.")
        share = sec.get("side_share")
        if share is not None and sec["n"] >= 8 and max(share, 1 - share) >= 0.85:
            side = ("Over" if share >= 0.5 else "Under") if k == "game_total" else ("the home side" if share >= 0.5 else "the road side")
            txt += (f" Watch out: {round(max(share, 1 - share) * (sec['n'] + sec['pushes']))} of {sec['n'] + sec['pushes']} picks were on {side}. "
                    f"A lean that one-sided usually means a built-in bias, not that many games independently pointed the same way.")
        if k == "game_total":
            txt += " Totals are the hardest game market to beat; a week of 15 games says very little."
        return txt
    # a prop market
    a, hi, rp = sec["all"], sec["high"], sec["recommended"]
    if not a["n"]:
        return "No graded props yet for this market."
    parts = [f"Across {a['n']} graded props the model's raw lean was right {_pct(a['win_rate'])} ({_rec_str(a)}); it expected {_pct(a['expected_rate'])}."]
    if hi["n"]:
        parts.append(f"High-confidence picks (raw probability {int(HIGH_CONF * 100)}%+): {_rec_str(hi)} ({_pct(hi['win_rate'])}) against an expected {_pct(hi['expected_rate'])}, {luck_phrase(hi['z'])}.")
    else:
        parts.append(f"No props reached the {int(HIGH_CONF * 100)}% high-confidence bar.")
    if rp["n"]:
        parts.append(f"The picks the page recommended went {_rec_str(rp)}.")
    if a.get("brier_model") is not None:
        verdict = ("model sharper" if a["brier_model"] < a["brier_market"] else "market sharper") if a["n"] >= MIN_FOR_SHARPER else "too few to call"
        parts.append(f"Probability quality: model {a['brier_model']:.3f} vs market {a['brier_market']:.3f} ({verdict}).")
    if sec.get("n_acc"):
        parts.append(f"The model's projection missed by {sec['mae_model']:.1f} on average and the posted line by {sec['mae_market']:.1f}.")
    return " ".join(parts)


def headline(rep: dict) -> str:
    ml = rep["sections"][0]
    bits = []
    if ml["n"]:
        bits.append(f"moneyline {ml['wins']}-{ml['losses']}")
    for s in rep["sections"][1:3]:
        if s["n"]:
            bits.append(f"{s['title'].lower().replace(' picks', '')} {_rec_str(s)}")
    hi_n = sum(p["high"]["n"] for p in rep["props"])
    hi_w = sum(p["high"]["wins"] for p in rep["props"])
    if hi_n:
        bits.append(f"high-confidence props {hi_w}-{hi_n - hi_w}")
    if not bits:
        return f"Week {rep['week']}: nothing graded yet."
    pending = f" ({rep['games_pending']} games still to be graded)" if rep.get("games_pending") else ""
    return f"Week {rep['week']}: " + ", ".join(bits) + "." + pending


# ---- rendering ---------------------------------------------------------------------------
def render_markdown(rep: dict) -> str:
    L = [f"# Week {rep['week']}, {rep['season']}: how the picks did", "", rep["headline"], ""]
    ml = rep["sections"][0]
    L += [f"## {ml['title']}", "", ml["writeup"], ""]
    if ml["tiers"]:
        L += ["| Tier | Range | Right | Wrong | Expected right |", "|---|---|---|---|---|"]
        L += [f"| {t['tier']} | {t['range']} | {t['wins']} | {t['losses']} | {t['expected_wins']:.1f} |" for t in ml["tiers"]]
        L += [""]
    for e in ml["entries"]:
        mark = {"win": "right", "loss": "WRONG", "push": "tie"}[e["result"]]
        L.append(f"- {e['tier']} {e['pick']} ({_pct(e['prob'])}) in {e['game']}: {mark}")
    L.append("")
    for s in rep["sections"][1:]:
        L += [f"## {s['title']}", "", s["writeup"], ""]
        fmt = (lambda v: f"{v:g}") if s["key"] == "game_total" else (lambda v: f"{v:+g}")
        L += [f"- {i['game']}: {i['pick']} {fmt(i['line'])}, {_pct(i['prob'])}: {i['result']}" for i in s["items"]] + [""]
    L += ["## Player props", ""]
    for p in rep["props"]:
        L += [f"### {p['title']}", "", p["writeup"], ""]
        L += [f"- {i['player']}: {i['pick']} {i['line']:g} ({_pct(i['prob'])}), result {i['actual']}: {i['result']}" for i in p["items"]] + [""]
    if rep["season_to_date"]:
        L += ["## Season to date", "", "| Group | Record | Win rate | Expected |", "|---|---|---|---|"]
        for g, v in sorted(rep["season_to_date"].items()):
            if (":" in g and not g.endswith(":high")) or not v["n"]:
                continue
            L.append(f"| {g} | {v['wins']}-{v['losses']} | {_pct(v['win_rate'])} | {v['expected_wins']:.1f} wins |")
        L.append("")
    L += ["## What the data is teaching the model", ""]
    L += [f"- {x}" for x in rep["learning"]] if rep["learning"] else ["- Nothing changed this week."]
    return "\n".join(L) + "\n"


def render_html(rep: dict, archive: list = None) -> str:
    e = html.escape
    tier_cls = lambda t: "t" + t
    css = """
:root{--bg:#12151b;--surface:#1b2028;--s2:#242a34;--border:#333b48;--text:#e9eaed;--dim:#98a0ac;--gold:#d9a441;--pos:#5cb88a;--neg:#d9705f}
@media(prefers-color-scheme:light){:root{--bg:#f4f2ee;--surface:#fff;--s2:#ece9e2;--border:#d8d3c8;--text:#201d17;--dim:#5c574c;--gold:#a1731d;--pos:#226b48;--neg:#b5432f}}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 "IBM Plex Sans",system-ui,sans-serif}
main{max-width:860px;margin:0 auto;padding:36px 18px 70px}h1{font:700 32px "Barlow Condensed",sans-serif;text-transform:uppercase;margin:0 0 4px}
h1 span{color:var(--gold)}h2{font:700 22px "Barlow Condensed",sans-serif;text-transform:uppercase;color:var(--gold);margin:30px 0 4px}
h3{font-size:15px;margin:18px 0 2px}.head{color:var(--dim);margin:0 0 6px}.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 16px;margin:10px 0}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:5px 8px;border-top:1px solid var(--border)}th{color:var(--dim);font-size:11px;text-transform:uppercase;border-top:none}
.ok{color:var(--pos);font-weight:700}.bad{color:var(--neg);font-weight:700}.dim{color:var(--dim)}.tier{display:inline-block;min-width:22px;text-align:center;border-radius:5px;padding:0 6px;font-weight:700;background:var(--s2)}
.tS{background:var(--gold);color:#16130a}.tA{background:var(--pos);color:#fff}a{color:var(--gold)}ul{padding-left:18px}li{margin:3px 0}
"""
    res = lambda r: {"win": '<span class="ok">right</span>', "loss": '<span class="bad">wrong</span>', "push": "tie/push"}[r]
    out = [f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
           f"<title>Weekly Report</title><link rel=\"stylesheet\" href=\"https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@700&family=IBM+Plex+Sans:wght@400;600&display=swap\"><style>{css}</style></head><body><main>",
           f'<h1><span>Week {rep["week"]}</span> report, {rep["season"]}</h1><p class="head">{e(rep["headline"])}</p>']
    ml = rep["sections"][0]
    out.append(f'<h2>{e(ml["title"])}</h2><div class="card"><p>{e(ml["writeup"])}</p>')
    if ml["tiers"]:
        out.append("<table><tr><th>Tier</th><th>Win chance</th><th>Right</th><th>Wrong</th><th>Expected right</th></tr>" + "".join(
            f'<tr><td><span class="tier {tier_cls(t["tier"])}">{t["tier"]}</span></td><td>{e(t["range"])}</td><td>{t["wins"]}</td><td>{t["losses"]}</td><td>{t["expected_wins"]:.1f}</td></tr>' for t in ml["tiers"]) + "</table>")
    out.append("<ul>" + "".join(f'<li><span class="tier {tier_cls(x["tier"])}">{x["tier"]}</span> {e(x["pick"])} ({_pct(x["prob"])}) in {e(x["game"])}: {res(x["result"])}' + (f' <span class="dim">{e(x["score"])}</span>' if x.get("score") else "") + "</li>" for x in ml["entries"]) + "</ul></div>")
    for s in rep["sections"][1:]:
        out.append(f'<h2>{e(s["title"])}</h2><div class="card"><p>{e(s["writeup"])}</p><ul>' + "".join(
            f'<li>{e(i["game"])}: {e(i["pick"])} {(format(i["line"], "g") if s["key"] == "game_total" else format(i["line"], "+g"))} ({_pct(i["prob"])}): {res(i["result"])}</li>' for i in s["items"]) + "</ul></div>")
    out.append("<h2>Player props</h2>")
    for p in rep["props"]:
        out.append(f'<div class="card"><h3>{e(p["title"])}</h3><p>{e(p["writeup"])}</p>' + ("<ul>" + "".join(
            f'<li>{e(i["player"])}: {e(i["pick"])} {i["line"]:g} ({_pct(i["prob"])}), result {i["actual"]}: {res(i["result"])}</li>' for i in p["items"]) + "</ul>" if p["items"] else "") + "</div>")
    std = {g: v for g, v in rep["season_to_date"].items() if (":" not in g or g.endswith(":high")) and v["n"]}
    if std:
        out.append("<h2>Season to date</h2><div class=\"card\"><table><tr><th>Group</th><th>Record</th><th>Win rate</th><th>Expected wins</th><th>Weeks</th></tr>" + "".join(
            f'<tr><td>{e(g)}</td><td>{v["wins"]}-{v["losses"]}</td><td>{_pct(v["win_rate"])}</td><td>{v["expected_wins"]:.1f}</td><td>{v["weeks"]}</td></tr>' for g, v in sorted(std.items())) + "</table></div>")
    out.append("<h2>What the data is teaching the model</h2><div class=\"card\"><ul>" + ("".join(f"<li>{e(x)}</li>" for x in rep["learning"]) if rep["learning"] else "<li>Nothing changed this week.</li>") + "</ul></div>")
    if archive:
        out.append("<h2>Earlier weeks</h2><p>" + " &middot; ".join(f'<a href="{e(a["href"])}">Week {a["week"]}</a>' for a in archive) + "</p>")
    out.append(f'<p class="dim">Generated {e(rep["generated"])}. Picks are logged before kickoff and graded after; a few picks in a week say very little, so read the season-to-date table, not one week.</p></main></body></html>')
    return "".join(out)


# ---- running it --------------------------------------------------------------------------
def latest_graded_week(log: pd.DataFrame, season: int) -> int:
    g = log[(log["season"] == season) & log["result"].isin(["over", "under", "push"])]
    return int(g["week"].max()) if not g.empty else None


MOSTLY_DONE = 0.6        # a week is "reportable" once this share of its games have been graded


def default_week(log: pd.DataFrame, sched: pd.DataFrame, season: int):
    """
    The week to report on: the latest one where most of its games are graded. On Sunday morning Thursday's game is graded
    but the week is not, so the report still shows the previous week; on Monday the weekend's games are in. Falls back to
    the latest week with anything graded.
    """
    g = log[(log["season"] == season) & log["market"].isin(list(GAME_GROUPS) + ["game_su"]) & log["result"].isin(["over", "under", "push"])]
    if g.empty:
        return latest_graded_week(log, season)
    total = sched[(sched["season"] == season) & (sched["game_type"] == "REG")].groupby("week").size() if sched is not None and len(sched) else {}
    for week in sorted(g["week"].unique(), reverse=True):
        n_games = int(g[g["week"] == week]["game"].nunique())
        denom = int(total.get(week, 0)) if len(total) else 0
        if denom and n_games / denom >= MOSTLY_DONE:
            return int(week)
    return latest_graded_week(log, season)


def main():
    import nflreadpy as nfl
    import props_tracker

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--week", type=int)
    ap.add_argument("--season", type=int)
    ap.add_argument("--html", action="store_true", help="write reports/week-NN.html + web/weekly_report.html + reports/week-NN.md")
    args = ap.parse_args()
    season = args.season or int(nfl.get_current_season())
    log = props_tracker._load()
    sched = nfl.load_schedules([season]).to_pandas()
    sched = sched[sched["game_type"] == "REG"]
    week = args.week or default_week(log, sched, season)
    if week is None:
        print("Nothing graded yet this season.")
        return
    rep = build_report(log, sched, season, week)
    hist = upsert_summary(summary_rows(rep))
    rep["season_to_date"] = season_to_date(hist, rep)
    try:
        import improve
        rep["learning"] = improve.run(log, season, week, rep["season_to_date"])
    except Exception as ex:                                   # the report must still go out if the learning step trips
        rep["learning"] = [f"The learning step did not run this week ({ex})."]
    print(render_markdown(rep))
    if args.html:
        REPORT_DIR.mkdir(exist_ok=True)
        weeks = sorted(int(p.stem.split("-")[1]) for p in REPORT_DIR.glob("week-*.html"))
        archive = [{"week": w, "href": f"week-{w:02d}.html"} for w in sorted(set(weeks + [week]), reverse=True)]
        page = render_html(rep, archive)
        (REPORT_DIR / f"week-{week:02d}.html").write_text(page, encoding="utf-8")
        (REPORT_DIR / f"week-{week:02d}.md").write_text(render_markdown(rep), encoding="utf-8")
        LATEST_HTML.write_text(page, encoding="utf-8")
        print(f"Wrote reports/week-{week:02d}.html/.md and {LATEST_HTML}", file=sys.stderr)


if __name__ == "__main__":
    main()
