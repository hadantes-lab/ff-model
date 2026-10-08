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

import depth_chart as dc
import game_context as gc
import game_odds as go
import odds_api
import props_tracker
import team_ratings
import team_stats
from matchups import CoverageModel, load_coverage_targets
from props_model import (
    MARKETS, CORE_MARKETS, MIN_GAMES, blend_toward_market, environment_effect, fair_line, load_calibration,
    market_weight, build_history, distribution_summary, draw, fit_player,
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
                d = draw(spec.kind, fit["mean"], fit["var"], rng, 4000)
                fair = fair_line(spec.kind, d)
                if spec.kind == "yards":
                    # nudge the model's own fair line by book-to-book noise, for demo realism
                    line = float(np.floor(fair * rng.normal(1.0, 0.05) * 2 + 0.5) / 2)
                    line = line + 0.5 if line == int(line) else line
                else:
                    line = fair
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


def build_prop(event, home, away, player_row, market, cons, fit, mult, inj, matchup=None, promo=1.0,
               lambda_override=None):
    env = fit.get("env")
    spec = MARKETS[market]
    line = cons["point"]
    rng = rng_for(player_row["player_id"], market)
    draws = draw(spec.kind, fit["mean"], fit["var"], rng)
    p = line_probs(draws, line)

    # The model's own "prediction line" -- computed from the simulated distribution alone, with
    # no knowledge of where the book set the real line -- checked against it once pulled.
    predicted_line = fair_line(spec.kind, draws)
    line_gap = None if predicted_line is None else round(predicted_line - line, 2)

    market_over = cons["fair_over"]
    if market_over is None and cons["best_over"]:
        market_over = odds_api.american_to_prob(cons["best_over"]["price"])   # includes the vig

    # The raw model disagrees with real markets far more than real edges exist, so everything
    # downstream (edge, EV, the pick) uses the model shrunk toward the market. Raw stays visible.
    raw = dict(p)
    if market_over is not None:
        adj_over = blend_toward_market(p["over"], market_over, fit["n_eff"], lambda_override)
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
        "player_id": player_row["player_id"],
        "player": player_row["player_display_name"], "team": team, "opp": opp,
        "home": team == home, "pos": player_row["position"],
        "game": f"{away} @ {home}", "commence": event["commence_time"],
        "market": market, "label": spec.label, "kind": spec.kind, "line": line,
        "mean": round(fit["mean"], 2), "sd": round(float(np.sqrt(fit["var"])), 2),
        "predicted_line": predicted_line, "line_gap": line_gap,
        "opp_mult": round(mult, 3), "promoted": round(promo, 3) if promo > 1.001 else None,
        "n_games": fit["n_games"], "thin": fit["n_games"] < THIN_GAMES,
        "injury": inj,
        "over": round(p["over"], 4), "push": round(p["push"], 4), "under": round(p["under"], 4),
        "model_over": round(raw["over"], 4), "model_under": round(raw["under"], 4),
        "gap": None if market_over is None else round((raw["over"] - market_over) * 100, 1),
        "trust": round(market_weight(fit["n_eff"], lambda_override) if lambda_override is not None
                      else market_weight(fit["n_eff"]), 3),
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


def build_game(event, home, away, home_fits, away_fits, line, prices, live_weights=None):
    """Sides and totals, priced the same bottom-up way as player props: simulate, sum, compare."""
    rng = rng_for(event["id"], "game")
    home_pts, away_pts = go.simulate_game_scores(home_fits, away_fits, rng)
    raw = go.spread_total_probs(home_pts, away_pts, line["home_spread"], line["total"])
    n_eff = go.fits_n_eff(home_fits, away_fits)

    # The model's own "prediction line" from the raw simulation alone, in the same sign
    # convention as the real posted line, checked against it once pulled.
    predicted_spread = round(-raw["proj_margin"] * 2) / 2
    predicted_total = round(raw["proj_total"] * 2) / 2
    spread_gap = round(predicted_spread - line["home_spread"], 2)
    total_gap = round(predicted_total - line["total"], 2)

    live_weights = live_weights or {}
    spread_override = (live_weights.get("game_spread") or {}).get("suggested_weight")
    total_override = (live_weights.get("game_total") or {}).get("suggested_weight")
    home_cover = go.blend_game_prob(raw["home_cover"], prices.get("fair_home_cover"), n_eff, spread_override)
    over = go.blend_game_prob(raw["over"], prices.get("fair_over"), n_eff, total_override)
    away_cover = max(0.0, 1 - home_cover - raw["home_push"])
    under = max(0.0, 1 - over - raw["total_push"])

    def pick_side(p_a, price_a, name_a, p_b, price_b, name_b):
        sides = []
        if price_a:
            sides.append((ev_per_dollar(p_a, 1 - p_a, price_a["price"]), name_a, price_a))
        if price_b:
            sides.append((ev_per_dollar(p_b, 1 - p_b, price_b["price"]), name_b, price_b))
        return max(sides) if sides else (None, None, None)

    spread_ev, spread_pick, spread_price = pick_side(
        home_cover, prices.get("best_home"), "home", away_cover, prices.get("best_away"), "away")
    total_ev, total_pick, total_price = pick_side(
        over, prices.get("best_over"), "over", under, prices.get("best_under"), "under")

    return {
        "game": f"{away} @ {home}", "home": home, "away": away, "commence": event["commence_time"],
        "home_spread": line["home_spread"], "total": line["total"],
        "proj_home": round(raw["proj_home"], 1), "proj_away": round(raw["proj_away"], 1),
        "proj_margin": round(raw["proj_margin"], 1), "proj_total": round(raw["proj_total"], 1),
        "predicted_spread": predicted_spread, "predicted_total": predicted_total,
        "spread_gap": spread_gap, "total_gap": total_gap,
        "n_players": len(home_fits) + len(away_fits), "n_eff": round(n_eff, 1),
        "spread_trust": round((go.GAME_LAMBDA_MAX if spread_override is None else spread_override)
                              * min(1.0, n_eff / go.GAME_LAMBDA_FULL_N), 3),
        "total_trust": round((go.GAME_LAMBDA_MAX if total_override is None else total_override)
                             * min(1.0, n_eff / go.GAME_LAMBDA_FULL_N), 3),
        "home_cover": round(home_cover, 4), "away_cover": round(away_cover, 4),
        "model_home_cover": round(raw["home_cover"], 4), "fair_home_cover": prices.get("fair_home_cover"),
        "over": round(over, 4), "under": round(under, 4),
        "model_over": round(raw["over"], 4), "fair_over": prices.get("fair_over"),
        "best_home": prices.get("best_home"), "best_away": prices.get("best_away"),
        "best_over": prices.get("best_over"), "best_under": prices.get("best_under"),
        "spread_pick": spread_pick, "spread_pick_price": spread_price,
        "spread_pick_ev": None if spread_ev is None else round(spread_ev, 4),
        "total_pick": total_pick, "total_pick_price": total_price,
        "total_pick_ev": None if total_ev is None else round(total_ev, 4),
    }


# ---- ranking the picks ----------------------------------------------------
TOP_PROPS = 10
TOP_PROP_MAX_GAP = 12      # the page already flags |model - market| >= 12 as "model far from market"


def rank_top_props(props, n=TOP_PROPS):
    """
    The week's best prop picks, stamped in place as `top_rank` 1..n (others get None).
    Ranked by the EV of the recommended side, using the probabilities already shrunk toward the
    market. Left out: anytime TDs (long shots, mostly noise), thin samples, injured players, and
    props where the raw model sits far from the market (most likely model error, not edge). At most
    one pick per player, so one hot player can't fill the list.
    """
    for p in props:
        p["top_rank"] = None
    ok = [p for p in props
          if p.get("pick_ev") is not None and p["pick_ev"] > 0 and p.get("pick")
          and p["kind"] != "td" and not p.get("thin") and not p.get("injury")
          and (p.get("gap") is None or abs(p["gap"]) < TOP_PROP_MAX_GAP)]
    seen, top = set(), []
    for p in sorted(ok, key=lambda p: -p["pick_ev"]):
        if p["player_id"] in seen:
            continue
        seen.add(p["player_id"])
        top.append(p)
        if len(top) == n:
            break
    for i, p in enumerate(top):
        p["top_rank"] = i + 1
    return top


def game_pick_entries(g):
    """The (up to) two picks for one game -- spread and total -- each with its win probability."""
    out = []
    # spread: the EV-best priced side if there is one, else whichever side the model likes more
    side = g.get("spread_pick") or ("home" if g["home_cover"] >= g["away_cover"] else "away")
    home = side == "home"
    rk = g.get("rank") or {}
    rank_side = None if rk.get("rank_home_cover") is None else ("home" if rk["rank_home_cover"] > 0.5 else "away")
    out.append({"type": "Spread", "game": g["game"], "week": g.get("week"),
                "pick": f"{g['home'] if home else g['away']} {(g['home_spread'] if home else -g['home_spread']):+g}",
                "prob": g["home_cover"] if home else g["away_cover"],
                "ev": g.get("spread_pick_ev") if g.get("spread_pick") else None,
                "price": g.get("spread_pick_price") if g.get("spread_pick") else None,
                "rank_agrees": None if rank_side is None else rank_side == side})
    side = g.get("total_pick") or ("over" if g["over"] >= g["under"] else "under")
    out.append({"type": "Total", "game": g["game"], "week": g.get("week"),
                "pick": f"{side.title()} {g['total']:g}",
                "prob": g["over"] if side == "over" else g["under"],
                "ev": g.get("total_pick_ev") if g.get("total_pick") else None,
                "price": g.get("total_pick_price") if g.get("total_pick") else None,
                "rank_agrees": None})
    return out


def rank_game_picks(games):
    """
    Rank every spread and total pick on the page (in-window games) by confidence = the model's
    probability for the picked side after shrinking toward the market. -> the ranked list; each
    game also gets `pick_ranks` {"Spread": n, "Total": n}. `rank_agrees` marks spread picks the
    power-rating spread also supports -- shown as a tag, not folded into the number, because the
    backtest found the rating adds no information beyond the line.
    """
    entries = []
    for g in games:
        g["pick_ranks"] = {}
        if g.get("in_window", True):
            entries.extend(game_pick_entries(g))
    entries.sort(key=lambda e: (-e["prob"], -(e["ev"] if e["ev"] is not None else -9)))
    by_game = {g["game"]: g for g in games}
    for i, e in enumerate(entries):
        e["rank"] = i + 1
        by_game[e["game"]]["pick_ranks"][e["type"]] = i + 1
    return entries


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
    live_weights = props_tracker.load_live_calibration()
    if live_weights:
        log(f"Live calibration: refined trust weight for {len(live_weights)} market(s) "
            f"from tracked results ({', '.join(live_weights)})")
    idx = build_roster_index(history)
    inj = injury_map(season)
    mults = {m: opponent_multipliers(history, MARKETS[m]) for m in markets}
    coverage = None
    if not args.no_matchups and "player_reception_yds" in markets:
        log("Loading man/zone coverage tags (2024-25) for receiver matchups...")
        coverage = CoverageModel(load_coverage_targets())
    priors = {m: position_priors(history, MARKETS[m]) for m in markets if MARKETS[m].kind != "yards"}
    # needed to roll players up into team totals for game sides/totals, regardless of which
    # prop markets were actually requested from the Odds API
    for m in ("player_pass_yds", "player_rush_yds", "player_anytime_td"):
        mults.setdefault(m, opponent_multipliers(history, MARKETS[m]))
    priors.setdefault("player_anytime_td", position_priors(history, MARKETS["player_anytime_td"]))

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

    # this week's game totals and spreads: schedule lines for the demo, one bulk Odds API call
    # otherwise (also used below to price game sides/totals -- no extra credit cost)
    game_prices = {}
    if args.sample:
        gl = {r["game_id"]: {"total": r["total_line"], "home_spread": -r["spread_line"]}
              for _, r in nfl.load_schedules([season]).to_pandas().iterrows()
              if r["total_line"] == r["total_line"] and r["spread_line"] == r["spread_line"]}
        game_events = events
    else:
        try:
            raw_game_lines = odds_api.fetch_game_lines(key)[0]
            gl = odds_api.consensus_game_lines(raw_game_lines)
            game_prices = odds_api.consensus_game_prices(raw_game_lines)
            # the whole week's games, not just the --days window player props are limited to --
            # this bulk call costs the same flat few credits no matter how many games it covers,
            # so pricing every upcoming game (and being able to spot a Monday-vs-later-pull line
            # move on it) is free.
            game_events = [e for e in raw_game_lines if not odds_api.has_started(e)]   # never price live games
        except odds_api.OddsApiError as e:
            log(f"  no game lines ({e}); game environment adjustment and game odds skipped")
            gl = {}
            game_events = []

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
            promo = dc.promotion_multiplier(history, row["team"], row["position"], row["player_id"], inj)
            fit = fit_player(history, row["player_id"], MARKETS[market],
                             mult * (matchup["mult"] if matchup else 1.0) * promo, priors.get(market),
                             (calibration or {}).get("calibration", {}).get(market), env_fn)
            if fit is None:
                skipped += 1
                continue
            lambda_override = (live_weights.get(market) or {}).get("suggested_weight")
            props.append(build_prop(ev, home, away, row, market, c, fit, mult, status, matchup, promo, lambda_override))

    log(f"{len(props)} props simulated; {skipped} skipped (out/doubtful or under {MIN_GAMES} games); "
        f"{len(unmatched)} names not matched to a player.")
    if unmatched:
        log("  unmatched: " + ", ".join(sorted(unmatched)[:12]) + (" ..." if len(unmatched) > 12 else ""))

    profiles, lg_ypp, lg_plays, power = {}, 5.4, 62.0, None
    try:   # informational matchup stats + power rankings; a data hiccup here must never block the odds page
        tg = team_stats.team_game_stats(team_stats.load_pbp(range(int(season) - 2, int(season) + 1)))
        profiles, lg_ypp, lg_plays = team_stats.current_profiles(tg), float(tg["ypp"].mean()), float(tg["plays"].mean())
        sched_all = nfl.load_schedules(list(range(int(season) - 2, int(season) + 1))).to_pandas()
        rgames = team_ratings.build_games(tg, sched_all[sched_all["game_type"] == "REG"])
        power = team_ratings.power_table(rgames, int(season), None)
        log(f"Power rankings built from {len(rgames)} games (top: "
            + ", ".join(f"{r['rank']}. {r['team']}" for r in power["table"][:3]) + ")")
    except Exception as e:
        log(f"Team stat profiles / power rankings unavailable ({e}); game cards will omit them.")

    sched_now = None
    if not args.sample:
        try:
            sched_now = nfl.load_schedules([season]).to_pandas()
        except Exception as e:
            log(f"Schedule unavailable ({e}); games will be tagged with the current week.")
    window_end = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=args.days)

    games = []
    for ev in game_events:
        home, away = odds_api.TEAM_ABBR[ev["home_team"]], odds_api.TEAM_ABBR[ev["away_team"]]
        line = gl.get(ev["id"], {})
        if line.get("total") is None or line.get("home_spread") is None:
            continue   # no real line to compare a simulated score to
        home_fits = go.build_team_fits(history, home, away, mults, priors, calibration,
                                       line["total"], -line["home_spread"], inj_status=inj)
        away_fits = go.build_team_fits(history, away, home, mults, priors, calibration,
                                       line["total"], line["home_spread"], inj_status=inj)
        if not home_fits or not away_fits:
            continue
        game = build_game(ev, home, away, home_fits, away_fits, line, game_prices.get(ev["id"], {}), live_weights)
        # the bulk game-lines call covers every posted game (next week's too), so the week comes
        # from the schedule, and only games inside the --days window are shown on the page
        game["week"] = props_tracker.week_for_game(sched_now, home, away, ev["commence_time"], default=int(week))
        game["in_window"] = bool(args.sample or datetime.datetime.fromisoformat(
            ev["commence_time"].replace("Z", "+00:00")) < window_end)
        game["rank"] = None
        if power is not None and home in power["ranks"] and away in power["ranks"]:
            game["rank"] = team_ratings.project_game(
                power, home, away, line["home_spread"], qb_out_home=dc.team_qb_out(history, home, inj),
                qb_out_away=dc.team_qb_out(history, away, inj))
            hc = game["rank"]["rank_home_cover"]
            side = "home" if hc > 0.5 else "away"
            price = game.get("best_home") if side == "home" else game.get("best_away")
            game["rank_pick"], game["rank_pick_price"] = side, price
            game["rank_pick_ev"] = None if not price else round(
                ev_per_dollar(hc if side == "home" else 1 - hc, (1 - hc) if side == "home" else hc, price["price"]), 4)
        game["profile"] = team_stats.matchup_profile(profiles, home, away, lg_ypp, lg_plays)
        games.append(game)
    log(f"{len(games)} games priced for sides/totals.")
    top = rank_top_props(props)
    game_picks = rank_game_picks(games)
    log(f"Top {len(top)} props and {len(game_picks)} ranked game picks.")
    rating_backtest = None
    bt = pathlib.Path(__file__).parent / "web" / "rating_backtest.json"
    if bt.exists():
        rating_backtest = json.loads(bt.read_text(encoding="utf-8"))

    snapshot = {
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "season": int(season), "week": int(week), "sample": bool(args.sample),
        "credits": credits, "markets": markets, "n_games": len(events), "props": props, "games": games,
        "game_picks": game_picks, "rating_backtest": rating_backtest,
        "power": None if power is None else {"table": power["table"], "hfa": round(power["hfa"], 2),
                                              "per_rank": round(team_ratings.PER_RANK, 3)},
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(snapshot, separators=(",", ":")), encoding="utf-8")
    log(f"Wrote {OUT}")

    if not args.sample:
        n = props_tracker.log_predictions(snapshot)
        log(f"Tracker: logged/updated {n} props for later grading (tracking/props_log.csv)")


if __name__ == "__main__":
    main()
