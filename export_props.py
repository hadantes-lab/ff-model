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
import re
import sys

import numpy as np
import nflreadpy as nfl

import game_context as gc
import odds_api
from matchups import CoverageModel, load_coverage_targets
from props_model import (
    MARKETS, CORE_MARKETS, MIN_GAMES, blend_toward_market, environment_effect, load_calibration, market_weight, build_history, distribution_summary, draw, fit_player,
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
def sample_event_odds(event, history, idx, rng, priors, markets):
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
            if m not in markets:
                continue
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
    # a made-up pick'em book: one line per prop, sometimes a step away from the sportsbook line
    pickem = {}
    for m, outs in books["DemoBook A"].items():
        for o in outs:
            if o["name"] == "Over" and rng.random() < 0.6:
                step = 1.0 if MARKETS[m].kind == "count" else 2.0 if MARKETS[m].kind == "yards" else 0.0
                pt = o["point"] + float(rng.choice([-1, 0, 1])) * step
                pickem.setdefault(m, []).append(
                    {"name": "Over", "description": o["description"], "price": -110, "point": max(0.5, pt)})
    if pickem:
        books["DemoUnderdog"] = pickem
    return {"bookmakers": [
        {"title": b, "key": b.lower().replace(" ", "_"),
         "markets": [{"key": k, "outcomes": o} for k, o in ms.items()]}
        for b, ms in books.items()]}


# ---- pricing math -------------------------------------------------------
def ev_per_dollar(win_prob, lose_prob, american):
    """Expected profit per $1 staked; a push returns the stake."""
    return win_prob * (odds_api.american_to_decimal(american) - 1) - lose_prob


def build_prop(event, home, away, player_row, market, cons, fit, mult, inj, matchup=None):
    env = fit.get("env")
    spec = MARKETS[market]
    line = cons["point"]
    rng = rng_for(player_row["player_id"], market)
    draws = draw(spec.kind, fit["mean"], fit["var"], rng)
    p = line_probs(draws, line)

    market_over = cons["fair_over"]
    if market_over is None and cons["best_over"]:
        market_over = odds_api.american_to_prob(cons["best_over"]["price"])   # includes the vig

    # The raw model disagrees with real markets far more than real edges exist, so everything
    # downstream (edge, EV, the pick) uses the model shrunk toward the market. Raw stays visible.
    raw = dict(p)
    if market_over is not None:
        adj_over = blend_toward_market(p["over"], market_over, fit["n_eff"])
        p = {"over": adj_over, "push": p["push"], "under": max(0.0, 1.0 - p["push"] - adj_over)}

    ev_o = ev_per_dollar(p["over"], p["under"], cons["best_over"]["price"]) if cons["best_over"] else None
    ev_u = ev_per_dollar(p["under"], p["over"], cons["best_under"]["price"]) if cons["best_under"] else None
    sides = [(ev, s) for ev, s in ((ev_o, "over"), (ev_u, "under")) if ev is not None]
    pick_ev, pick = max(sides) if sides else (None, None)

    dfs = []
    for d in cons.get("dfs", []):
        dp = line_probs(draws, d["line"])
        dfs.append({"book": d["book"], "line": d["line"], "gap": round(d["line"] - line, 2),
                    "over": round(dp["over"], 4), "under": round(dp["under"], 4), "push": round(dp["push"], 4)})

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
        "model_over": round(raw["over"], 4), "model_under": round(raw["under"], 4),
        "gap": None if market_over is None else round((raw["over"] - market_over) * 100, 1),
        "trust": round(market_weight(fit["n_eff"]), 3),
        "fair_over": cons["fair_over"], "market_over": None if market_over is None else round(market_over, 4),
        "edge": None if market_over is None else round((p["over"] - market_over) * 100, 1),
        "best_over": cons["best_over"], "best_under": cons["best_under"],
        "ev_over": None if ev_o is None else round(ev_o, 4),
        "ev_under": None if ev_u is None else round(ev_u, 4),
        "pick": pick, "pick_ev": None if pick_ev is None else round(pick_ev, 4),
        "matchup": matchup,
        "env": None if not env or "total" not in env else {
            "mult": round(env["mult"], 4), "total": env["total"], "margin": env["margin"],
            "avg_total": round(env["avg_total"], 1), "avg_margin": round(env["avg_margin"], 1),
            "implied": round((env["total"] + env["margin"]) / 2, 1),
            "implied_opp": round((env["total"] - env["margin"]) / 2, 1)},
        "books": cons["books"], "dfs": dfs, "log": fit["log"], "dist": distribution_summary(spec.kind, draws),
    }


# ---- main ---------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", action="store_true", help="demo lines derived from the model (no API)")
    ap.add_argument("--markets", help="comma-separated Odds API market keys")
    ap.add_argument("--all-markets", action="store_true", help="every offensive market")
    ap.add_argument("--books", help="comma-separated bookmaker keys, e.g. draftkings,fanduel,underdog "
                                    "(up to 10 cost the same as one region; overrides the default 'us' region)")
    ap.add_argument("--no-matchups", action="store_true",
                    help="skip the man/zone coverage matchup (saves ~1 min of play-by-play loading)")
    ap.add_argument("--max-events", type=int, help="only the first N games (saves credits)")
    ap.add_argument("--days", type=int, default=7, help="games starting within N days")
    args = ap.parse_args()

    season = nfl.get_current_season()
    week = nfl.get_current_week()
    markets = (args.markets.split(",") if args.markets
               else list(MARKETS) if args.all_markets else CORE_MARKETS)
    books = [b.strip() for b in args.books.split(",")] if args.books else None
    bad = [m for m in markets if m not in MARKETS]
    if bad:
        sys.exit(f"Unknown market(s): {bad}. Choose from: {list(MARKETS)}")

    log(f"Loading nflverse history (season {season}, week {week})...")
    history = build_history(nfl.load_player_stats([season - 1, season]).to_pandas())
    history = gc.attach_context(history, gc.load_game_lines([season - 1, season]))   # totals/spreads of past games
    calibration = load_calibration()
    log("Calibration: " + ("loaded" if calibration else "NOT FOUND (run tune_props.py --write) -- using raw projections"))
    idx = build_roster_index(history)
    inj = injury_map(season)
    mults = {m: opponent_multipliers(history, MARKETS[m]) for m in markets}
    coverage = None
    if not args.no_matchups and "player_reception_yds" in markets:
        log("Loading man/zone coverage tags (2024-25) for receiver matchups...")
        coverage = CoverageModel(load_coverage_targets())
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
        regions = -(-len(books) // 10) if books else 1
        log(f"{len(events)} games in the next {args.days} days. Worst-case cost this run: "
            f"{len(events) * len(markets) * regions} credits ({len(events)} games x {len(markets)} markets x "
            f"{regions} region(s); only markets that return data are charged, cached games cost 0). {credits}")

    # this week's game totals and spreads: schedule lines for the demo, one bulk Odds API call otherwise
    if args.sample:
        gl = {r["game_id"]: {"total": r["total_line"], "home_spread": -r["spread_line"]}
              for _, r in nfl.load_schedules([season]).to_pandas().iterrows()
              if r["total_line"] == r["total_line"] and r["spread_line"] == r["spread_line"]}
    else:
        try:
            gl = odds_api.consensus_game_lines(odds_api.fetch_game_lines(key)[0])
        except odds_api.OddsApiError as e:
            log(f"  no game lines ({e}); game environment adjustment skipped")
            gl = {}

    props, unmatched, skipped = [], set(), 0
    for ev in events:
        home, away = odds_api.TEAM_ABBR[ev["home_team"]], odds_api.TEAM_ABBR[ev["away_team"]]
        if args.sample:
            odds = sample_event_odds(ev, history, idx, rng_for("sample", ev["id"]), priors, markets)
        else:
            odds, cached = odds_api.fetch_event_props(key, ev["id"], markets, bookmakers=books)
            log(f"  {away} @ {home}: {'cached' if cached else 'fetched'}")
        cons = odds_api.consensus(odds_api.flatten(odds))
        for (name, market), c in cons.items():
            if market not in MARKETS or re.search(r"(D/ST|Defense)$", name):
                continue                # team-defense TD props are not player props
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
            eff = environment_effect(calibration, market)
            g_line = gl.get(ev["id"], {})
            env_fn = None
            if eff and g_line.get("total") is not None and g_line.get("home_spread") is not None:
                team_margin = -g_line["home_spread"] if row["team"] == home else g_line["home_spread"]
                env_fn = lambda g, w, e=eff, t=g_line["total"], m=team_margin: gc.env_multiplier(g, w, t, m, e)
            matchup = None
            if coverage is not None and market == "player_reception_yds":
                matchup = coverage.for_receiver(row["player_id"], opp)     # scheme fit, on top of defense strength
            fit = fit_player(history, row["player_id"], MARKETS[market],
                             mult * (matchup["mult"] if matchup else 1.0), priors.get(market),
                             (calibration or {}).get("calibration", {}).get(market), env_fn)
            if fit is None:
                skipped += 1
                continue
            props.append(build_prop(ev, home, away, row, market, c, fit, mult, status, matchup))

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
