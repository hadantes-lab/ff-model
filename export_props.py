"""
export_props.py
===============
Pulls this week's player prop lines from The Odds API, simulates every prop
with props_model, and writes web/props_snapshot.json for the shareable page
(web/build_props_page.py turns it into one HTML file).

    python export_props.py                    # core markets, real lines
    python export_props.py --all-markets      # every offensive market (more credits)
    python export_props.py --sample           # DEMO lines made up from the model -- no key, no credits

Real lines need ODDS_API_KEY (see odds_api.py).
"""

import argparse
import datetime
import json
import pathlib
import sys

import numpy as np
import nflreadpy as nfl

import odds_api
from props_model import (
    MARKETS, CORE_MARKETS, MIN_GAMES, build_history, distribution_summary, draw, fit_player,
    line_probs, normalize_name, opponent_multipliers, position_priors, rng_for,
)

OUT = pathlib.Path(__file__).parent / "web" / "props_snapshot.json"
ABBR_TO_NAME = {v: k for k, v in odds_api.TEAM_ABBR.items()}
THIN_GAMES = 6


def log(msg):
    print(msg, file=sys.stderr)


# ---- player lookup ------------------------------------------------------
def build_roster_index(history):
    """normalized name -> [latest row per player]; a player's team is where he last played."""
    last = history.groupby("player_id", as_index=False).tail(1)
    idx = {}
    for _, r in last.iterrows():
        idx.setdefault(normalize_name(r["player_display_name"]), []).append(r)
    return idx


def find_player(idx, name, teams):
    hits = [r for r in idx.get(normalize_name(name), []) if r["team"] in teams]
    return hits[0] if len(hits) == 1 else None


def injury_map(season):
    try:
        inj = nfl.load_injuries([season]).to_pandas()
    except Exception:
        return {}
    inj = inj[inj["week"] == inj["week"].max()]
    inj = inj[inj["report_status"].notna()]
    return dict(zip(inj["gsis_id"], inj["report_status"]))


# ---- demo lines ---------------------------------------------------------
def sample_event_odds(event, history, idx, rng, priors):
    """
    Fabricated sportsbook lines, *derived from the model itself* with random
    noise, in the same shape the Odds API returns. For exercising the pipeline
    and page only -- these are NOT real market prices.
    """
    teams = {odds_api.TEAM_ABBR[event["home_team"]], odds_api.TEAM_ABBR[event["away_team"]]}
    # rank by average usage over the last 4 games, so a backup who threw one
    # garbage-time pass doesn't get a starter's line
    recent = history.groupby("player_id", as_index=False).tail(4)
    usage = recent.groupby("player_id")[["attempts", "carries", "targets"]].mean()
    last = history.groupby("player_id", as_index=False).tail(1).drop(columns=["attempts", "carries", "targets"])
    last = last.join(usage, on="player_id")
    last = last[last["team"].isin(teams)]
    picks = []
    for pos, col, n, mkts in [
        ("QB", "attempts", 1, ["player_pass_yds", "player_pass_tds"]),
        ("RB", "carries", 2, ["player_rush_yds", "player_receptions", "player_anytime_td"]),
        ("WR", "targets", 3, ["player_receptions", "player_reception_yds", "player_anytime_td"]),
        ("TE", "targets", 1, ["player_receptions", "player_reception_yds", "player_anytime_td"]),
    ]:
        for t in teams:
            g = last[(last["team"] == t) & (last["position"] == pos)]
            for _, r in g.sort_values(col, ascending=False).head(n).iterrows():
                picks.append((r, mkts))

    books = {}
    for r, mkts in picks:
        for m in mkts:
            fit = fit_player(history, r["player_id"], MARKETS[m], 1.0, priors.get(m))
            if not fit:
                continue
            spec = MARKETS[m]
            if spec.kind == "td":
                line = None
                base_p = 1 - np.exp(-fit["mean"])
            else:
                # books post a line that makes the over ~a coin flip: the *median*, which for
                # skewed stats sits below the mean
                d = draw(spec.kind, fit["mean"], fit["var"], rng, 4000)
                if spec.kind == "yards":
                    line = float(np.floor(np.median(d) * rng.normal(1.0, 0.05) * 2 + 0.5) / 2)
                    line = line + 0.5 if line == int(line) else line
                else:
                    k = int(np.floor(np.median(d)))
                    line = min((k - 0.5, k + 0.5), key=lambda c: abs(float((d > c).mean()) - 0.5))
                    line = max(0.5, line)
                    if abs(float((d > line).mean()) - 0.5) > 0.2:
                        continue        # no realistic coin-flip line for a near-zero role
            for i, book in enumerate(["DemoBook A", "DemoBook B", "DemoBook C"]):
                mk = books.setdefault(book, {}).setdefault(m, [])
                if line is None:
                    p = float(np.clip(base_p * rng.normal(1.0, 0.06), 0.03, 0.95))
                    price = int(round(-100 * p / (1 - p))) if p > 0.5 else int(round(100 * (1 - p) / p))
                    mk.append({"name": "Yes", "description": r["player_display_name"], "price": price})
                else:
                    j = int(rng.choice([-5, 0, 0, 5, 10]))
                    mk.append({"name": "Over", "description": r["player_display_name"],
                               "price": -110 + j, "point": line})
                    mk.append({"name": "Under", "description": r["player_display_name"],
                               "price": -110 - j, "point": line})
    return {"bookmakers": [
        {"title": b, "markets": [{"key": k, "outcomes": o} for k, o in ms.items()]}
        for b, ms in books.items()]}


# ---- pricing math -------------------------------------------------------
def ev_per_dollar(win_prob, lose_prob, american):
    """Expected profit per $1 staked; a push returns the stake."""
    return win_prob * (odds_api.american_to_decimal(american) - 1) - lose_prob


def build_prop(event, home, away, player_row, market, cons, fit, mult, inj):
    spec = MARKETS[market]
    line = cons["point"]
    rng = rng_for(player_row["player_id"], market)
    draws = draw(spec.kind, fit["mean"], fit["var"], rng)
    p = line_probs(draws, line)

    market_over = cons["fair_over"]
    if market_over is None and cons["best_over"]:
        market_over = odds_api.american_to_prob(cons["best_over"]["price"])   # includes the vig

    ev_o = ev_per_dollar(p["over"], p["under"], cons["best_over"]["price"]) if cons["best_over"] else None
    ev_u = ev_per_dollar(p["under"], p["over"], cons["best_under"]["price"]) if cons["best_under"] else None
    sides = [(ev, s) for ev, s in ((ev_o, "over"), (ev_u, "under")) if ev is not None]
    pick_ev, pick = max(sides) if sides else (None, None)

    team = player_row["team"]
    opp = away if team == home else home
    return {
        "player": player_row["player_display_name"], "team": team, "opp": opp,
        "home": team == home, "pos": player_row["position"],
        "game": f"{away} @ {home}", "commence": event["commence_time"],
        "market": market, "label": spec.label, "kind": spec.kind, "line": line,
        "mean": round(fit["mean"], 2), "sd": round(float(np.sqrt(fit["var"])), 2),
        "opp_mult": round(mult, 3), "n_games": fit["n_games"], "thin": fit["n_games"] < THIN_GAMES,
        "injury": inj,
        "over": round(p["over"], 4), "push": round(p["push"], 4), "under": round(p["under"], 4),
        "fair_over": cons["fair_over"], "market_over": None if market_over is None else round(market_over, 4),
        "edge": None if market_over is None else round((p["over"] - market_over) * 100, 1),
        "best_over": cons["best_over"], "best_under": cons["best_under"],
        "ev_over": None if ev_o is None else round(ev_o, 4),
        "ev_under": None if ev_u is None else round(ev_u, 4),
        "pick": pick, "pick_ev": None if pick_ev is None else round(pick_ev, 4),
        "books": cons["books"], "log": fit["log"], "dist": distribution_summary(spec.kind, draws),
    }


# ---- main ---------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", action="store_true", help="demo lines derived from the model (no API)")
    ap.add_argument("--markets", help="comma-separated Odds API market keys")
    ap.add_argument("--all-markets", action="store_true", help="every offensive market")
    ap.add_argument("--max-events", type=int, help="only the first N games (saves credits)")
    ap.add_argument("--days", type=int, default=7, help="games starting within N days")
    args = ap.parse_args()

    season = nfl.get_current_season()
    week = nfl.get_current_week()
    markets = (args.markets.split(",") if args.markets
               else list(MARKETS) if args.all_markets else CORE_MARKETS)
    bad = [m for m in markets if m not in MARKETS]
    if bad:
        sys.exit(f"Unknown market(s): {bad}. Choose from: {list(MARKETS)}")

    log(f"Loading nflverse history (season {season}, week {week})...")
    history = build_history(nfl.load_player_stats([season - 1, season]).to_pandas())
    idx = build_roster_index(history)
    inj = injury_map(season)
    mults = {m: opponent_multipliers(history, MARKETS[m]) for m in markets}
    priors = {m: position_priors(history, MARKETS[m]) for m in markets if MARKETS[m].kind != "yards"}

    credits = None
    if args.sample:
        sched = nfl.load_schedules([season]).to_pandas()
        wk = sched[(sched["week"] == week) & (sched["gameday"] >= str(datetime.date.today()))]
        events = [{"id": r["game_id"], "commence_time": f"{r['gameday']}T17:00:00Z",
                   "home_team": ABBR_TO_NAME[r["home_team"]], "away_team": ABBR_TO_NAME[r["away_team"]]}
                  for _, r in wk.iterrows()]
        log(f"SAMPLE MODE: {len(events)} games, demo lines made up from the model.")
    else:
        key = odds_api.load_key()
        events, headers = odds_api.fetch_events(key, args.days)
        credits = odds_api.credits_line(headers)
        if args.max_events:
            events = events[: args.max_events]
        log(f"{len(events)} games in the next {args.days} days. Worst-case cost this run: "
            f"{len(events) * len(markets)} credits ({len(events)} games x {len(markets)} markets; "
            f"cached games cost 0). {credits}")

    props, unmatched, skipped = [], set(), 0
    for ev in events:
        home, away = odds_api.TEAM_ABBR[ev["home_team"]], odds_api.TEAM_ABBR[ev["away_team"]]
        if args.sample:
            odds = sample_event_odds(ev, history, idx, rng_for("sample", ev["id"]), priors)
        else:
            odds, cached = odds_api.fetch_event_props(key, ev["id"], markets)
            log(f"  {away} @ {home}: {'cached' if cached else 'fetched'}")
        cons = odds_api.consensus(odds_api.flatten(odds))
        for (name, market), c in cons.items():
            if market not in MARKETS:
                continue
            row = find_player(idx, name, {home, away})
            if row is None:
                unmatched.add(name)
                continue
            status = inj.get(row["player_id"])
            if status in ("Out", "Doubtful"):
                skipped += 1
                continue
            opp = away if row["team"] == home else home
            mult = mults.get(market, {}).get((opp, row["position"]), 1.0)
            fit = fit_player(history, row["player_id"], MARKETS[market], mult, priors.get(market))
            if fit is None:
                skipped += 1
                continue
            props.append(build_prop(ev, home, away, row, market, c, fit, mult, status))

    log(f"{len(props)} props simulated; {skipped} skipped (out/doubtful or under {MIN_GAMES} games); "
        f"{len(unmatched)} names not matched to a player.")
    if unmatched:
        log("  unmatched: " + ", ".join(sorted(unmatched)[:12]) + (" ..." if len(unmatched) > 12 else ""))

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "season": int(season), "week": int(week), "sample": bool(args.sample),
        "credits": credits, "markets": markets, "n_games": len(events), "props": props,
    }, separators=(",", ":")), encoding="utf-8")
    log(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
