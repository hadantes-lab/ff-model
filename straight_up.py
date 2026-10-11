"""
straight_up.py
==============
Moneyline winners (straight up, no point spread) with a confidence ranking, for confidence pools: pick the
winner of every game in the week, then give each pick a point value from 1 to N (N = games in the week), the most
sure thing getting N. You score the points on every pick that wins, so the ORDER matters as much as the picks.

WHERE THE WIN PROBABILITIES COME FROM (tested, 2021-26, 1,424 games)
-------------------------------------------------------------------
    market moneyline (devigged)        66.6% of winners right   Brier 0.2124
    market SPREAD turned into a prob   66.6%                    Brier 0.2122     <- used
    our power ratings                  63.2%                    Brier 0.2262
    moneyline + 10% of our ratings     66.5%                    Brier 0.2124     (any weight on our model made it worse)
The spread-based number is as good as the moneyline (the two correlate at 0.995), so no extra Odds API credits are
spent on moneylines. Our own simulation's win probability is shown beside the pick for reference and has ZERO weight.

CONVERSION
----------
P(home wins) = Phi(-home_spread / SPREAD_SD) with the odds-sheet sign (negative = home favored). SPREAD_SD is the
spread of an NFL game's final margin around its posted spread, ~11.5 points (it fit 11.25 on 2018-20 and 11.75 on
2021-26). Checked against what happened, the result is well calibrated: teams priced 60-70% won 62.8%, 70-80% won
72.7%, 80-90% won 85.1%. Ranking a week's games by it scored 70.8% of the maximum possible pool points, against
66.1% for the same picks in random order.

THE SLATE
---------
A pool covers the whole week, including games already played, so `build_slate` uses the schedule: a game still to
come uses the live consensus spread, anything else its closing line from nflverse, and finished games carry their
result. Finished games keep their place in the ranking (they take a point slot like any other).
"""

import math

import pandas as pd

SPREAD_SD = 11.5


def win_prob(home_spread: float, sd: float = SPREAD_SD) -> float:
    """P(home team wins) from the posted home spread (negative = home favored)."""
    return 0.5 * (1 + math.erf((-home_spread / sd) / math.sqrt(2)))


def rank_slate(entries: list) -> list:
    """
    Order a week's games by confidence and assign pool points: most confident = N, least = 1. Each entry needs
    `p_home`. Adds: pick (team), pick_prob, points, rank (1 = most confident). Ties (identical probability) are
    broken by the bigger spread then the game name, so the order never flips between runs.
    """
    for e in entries:
        home_pick = e["p_home"] >= 0.5
        e["pick"] = e["home"] if home_pick else e["away"]
        e["pick_side"] = "home" if home_pick else "away"
        e["pick_prob"] = e["p_home"] if home_pick else 1 - e["p_home"]
        e["opponent"] = e["away"] if home_pick else e["home"]
    ordered = sorted(entries, key=lambda e: (-round(e["pick_prob"], 6), -abs(e.get("spread") or 0), e["game"]))
    n = len(ordered)
    for i, e in enumerate(ordered):
        e["rank"] = i + 1
        e["points"] = n - i
    return ordered


def _kickoff_utc(gameday, gametime) -> str:
    """nflverse gives Eastern local date and time; the tracker and page want UTC ('...Z')."""
    t = gametime if isinstance(gametime, str) else "13:00"
    return pd.Timestamp(f"{gameday} {t}", tz="America/New_York").tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def build_slate(sched: pd.DataFrame, week: int, live_spreads: dict = None, model_p: dict = None, season: int = None,
                sd: float = SPREAD_SD) -> list:
    """
    Every regular-season game of `week`, ranked. `live_spreads` {(home, away): home_spread} supplies the current line for
    games still to come; `model_p` {(home, away): our simulated P(home wins)} is carried along for display only.
    Each entry: game, week, home, away, kickoff, spread (home, odds sign), source ('live' | 'closing'), status
    ('upcoming' | 'final'), p_home, p_model, winner/correct for finished games, plus rank_slate's pick/points/rank.
    """
    live_spreads, model_p = live_spreads or {}, model_p or {}
    s = sched[(sched["week"] == week) & (sched["game_type"] == "REG")]
    if season is not None:
        s = s[s["season"] == season]
    entries = []
    for r in s.itertuples(index=False):
        key = (r.home_team, r.away_team)
        final = pd.notna(r.home_score) and pd.notna(r.away_score)
        if not final and key in live_spreads and live_spreads[key] is not None:
            spread, source = float(live_spreads[key]), "live"
        elif pd.notna(r.spread_line):
            spread, source = -float(r.spread_line), "closing"          # nflverse: positive = home favored
        else:
            continue                                                   # no line at all (not yet posted): can't rank it
        e = {"game": f"{r.away_team} @ {r.home_team}", "week": int(week), "home": r.home_team, "away": r.away_team,
             "kickoff": _kickoff_utc(r.gameday, getattr(r, "gametime", None)),
             "spread": spread, "source": source, "status": "final" if final else "upcoming",
             "p_home": win_prob(spread, sd), "p_model": model_p.get(key)}
        if final:
            e["home_score"], e["away_score"] = int(r.home_score), int(r.away_score)
            e["winner"] = r.home_team if r.home_score > r.away_score else r.away_team if r.away_score > r.home_score else "TIE"
        entries.append(e)
    ranked = rank_slate(entries)
    for e in ranked:
        if e["status"] == "final":
            e["correct"] = None if e["winner"] == "TIE" else e["winner"] == e["pick"]
    return ranked


def pool_summary(entries: list) -> dict:
    """What the ranked slate is worth: expected points, and the points already won/lost on finished games."""
    n = sum(e["points"] for e in entries)
    expected = sum(e["points"] * e["pick_prob"] for e in entries)
    won = sum(e["points"] for e in entries if e.get("correct") is True)
    lost = sum(e["points"] for e in entries if e.get("correct") is False)
    return {"games": len(entries), "max_points": n, "expected_points": round(expected, 1),
            "expected_share": round(expected / n, 3) if n else None, "won": won, "lost": lost,
            "final": sum(1 for e in entries if e["status"] == "final")}
