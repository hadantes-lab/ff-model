"""
backfill_history.py
===================
Fill the line-history backlog (tracking/line_history.csv) with PAST player-prop lines from the Odds
API's historical endpoints, so the hit-rate chart can show "Line then" for games from before we
started collecting, and so the model's edges can finally be checked against real closing lines.

THIS COSTS MONEY, SO NOTHING IS SPENT BY DEFAULT
------------------------------------------------
The historical endpoints are paid-plan only and charge 10 credits per market per game per snapshot
(the event list costs 1 per call, ~1 per week). So the commands are:

    python backfill_history.py plan  --seasons 2024 2025     # what it WOULD cost. No API calls, no key needed.
    python backfill_history.py probe                          # does this key's plan include historical data? <=1 credit.
    python backfill_history.py run --seasons 2025 --execute --max-credits 15000

`run` without --execute prints the plan and stops. With --execute it stops itself at --max-credits (measured from
the API's own per-call cost header, not just the estimate), and it can be re-run: every (game, snapshot) already
fetched is recorded in tracking/backfill_state.json and skipped, so an interrupted or capped run resumes.

WHAT IT STORES
--------------
Only our derived numbers, exactly as line_history.py does: the consensus (median-book) line, the no-vig
probability, the best price on each side, how many books. Never per-book prices or book names (public repo; the
Odds API's terms bar republishing an odds board). The raw per-book responses exist only in memory.

SNAPSHOTS
---------
Default: one snapshot per game, 30 minutes before kickoff (the closing line). `--offsets 0.5 120` adds one 5 days
out (an opening-ish line). Player props are only available from 2023-05-03, so seasons before 2023 are refused.

PLAYER MATCHING
---------------
Sportsbook names are matched to nflverse players the same way the live export does, restricted to players who
had a game that season up to that week, with the player's team that week.
"""

import argparse
import datetime
import json
import pathlib
import sys

import pandas as pd

import line_history
import odds_api

STATE_FILE = pathlib.Path(__file__).parent / "tracking" / "backfill_state.json"
EARLIEST_SEASON = 2023                    # player props history starts 2023-05-03
CREDITS_PER_MARKET = 10                   # historical event odds, per market returned, per region
EVENTS_CALL_CREDITS = 1
DEFAULT_OFFSETS_H = (0.5,)


def log(msg):
    print(msg, file=sys.stderr)


# ---- schedule -------------------------------------------------------------------
def commence_utc(row) -> pd.Timestamp:
    """Kickoff in UTC from the schedule's local gameday + gametime (Eastern, which nflverse uses)."""
    t = str(row["gametime"]) if pd.notna(row.get("gametime")) else "13:00"
    return pd.Timestamp(f"{row['gameday']} {t}", tz="America/New_York").tz_convert("UTC")


def season_games(sched: pd.DataFrame, season: int) -> pd.DataFrame:
    g = sched[(sched["season"] == season) & (sched["game_type"] == "REG")].copy()
    g["kickoff"] = [commence_utc(r) for _, r in g.iterrows()]
    return g.sort_values("kickoff").reset_index(drop=True)


def estimate(n_games: int, n_markets: int, n_offsets: int, n_weeks: int, regions: int = 1) -> dict:
    """Upper-bound credits: every market assumed to return data for every game."""
    odds = n_games * n_markets * regions * CREDITS_PER_MARKET * n_offsets
    events = n_weeks * n_offsets * EVENTS_CALL_CREDITS
    return {"games": n_games, "snapshots": n_games * n_offsets, "odds_credits": odds, "event_list_credits": events,
            "total_credits": odds + events}


# ---- state ----------------------------------------------------------------------
def load_state(path=None) -> dict:
    path = pathlib.Path(path or STATE_FILE)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"done": [], "credits_spent": 0}


def save_state(state: dict, path=None):
    path = pathlib.Path(path or STATE_FILE)
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(state, indent=1), encoding="utf-8")


def snap_key(event_id, offset_h) -> str:
    return f"{event_id}|{offset_h}"


# ---- API ------------------------------------------------------------------------
def last_cost(headers) -> int:
    try:
        return int(headers.get("x-requests-last"))
    except (TypeError, ValueError):
        return 0


def historical_events(key, date_iso, t_from, t_to):
    """-> (events, headers). One call lists the games that were upcoming at `date_iso` (1 credit)."""
    wrapper, headers = odds_api._get(f"/historical/sports/{odds_api.SPORT}/events",
                                     {"date": date_iso, "commenceTimeFrom": t_from, "commenceTimeTo": t_to}, key)
    return wrapper.get("data", []), headers


def historical_event_odds(key, event_id, date_iso, markets):
    """-> (wrapper with 'timestamp' and 'data', headers) for one game at one snapshot."""
    return odds_api._get(f"/historical/sports/{odds_api.SPORT}/events/{event_id}/odds",
                         {"date": date_iso, "regions": "us", "markets": ",".join(markets), "oddsFormat": "american"}, key)


def match_event(events, home, away, kickoff: pd.Timestamp):
    """The listed event for this pairing starting within a day of `kickoff`, or None."""
    for e in events:
        try:
            h, a = odds_api.TEAM_ABBR[e["home_team"]], odds_api.TEAM_ABBR[e["away_team"]]
        except KeyError:
            continue
        if h == home and a == away and abs(pd.Timestamp(e["commence_time"]) - kickoff) < pd.Timedelta(days=1):
            return e
    return None


# ---- rows -----------------------------------------------------------------------
def rows_from_event_odds(wrapper, game, idx, markets, season, find_player, MARKETS) -> list:
    """
    One line_history row per matched (player, market) in a historical event-odds response. `game` is
    {home, away, week, commence}; `idx` the roster index for that season/week.
    """
    pulled_at = wrapper.get("timestamp")
    cons = odds_api.consensus(odds_api.flatten(wrapper.get("data") or {}))
    rows = []
    for (name, market), c in cons.items():
        if market not in markets or market not in MARKETS or name.endswith(("D/ST", "Defense")):
            continue
        row = find_player(idx, name, {game["home"], game["away"]})
        if row is None:
            continue
        fair = c["fair_over"]
        if fair is None and c["best_over"]:
            fair = odds_api.american_to_prob(c["best_over"]["price"])
        opp = game["away"] if row["team"] == game["home"] else game["home"]
        rows.append({"pulled_at": pulled_at, "season": season, "week": game["week"], "kind": "prop", "market": market,
                     "player_id": row["player_id"], "player": row["player_display_name"], "team": row["team"],
                     "opp": opp, "game": f"{game['away']} @ {game['home']}", "commence": game["commence"],
                     "line": c["point"], "fair_over": fair,
                     "best_over": c["best_over"]["price"] if c["best_over"] else None,
                     "best_under": c["best_under"]["price"] if c["best_under"] else None,
                     "n_books": c.get("n_books", len(c["books"])), "mean": None, "model_over": None, "injury": None})
    return rows


# ---- driver ---------------------------------------------------------------------
def run(args, key=None, history_loader=None, sched=None, out_path=None, state_path=None):
    """Fetch what the plan says, within the cap. Returns a summary dict. Injectable for tests."""
    import nflreadpy as nfl
    from export_props import build_roster_index, find_player
    from props_model import CORE_MARKETS, EXTENDED_MARKETS, MARKETS, build_history

    markets = list(CORE_MARKETS) + (list(EXTENDED_MARKETS) if args.markets == "extended" else [])
    markets = [m for m in dict.fromkeys(markets) if not m.endswith("_longest")]      # longest markets need pbp-built history
    sched = sched if sched is not None else nfl.load_schedules(list(args.seasons)).to_pandas()
    state = load_state(state_path)
    spent_start = state["credits_spent"]
    summary = {"calls": 0, "rows": 0, "skipped_done": 0, "unmatched_games": 0, "stopped": None}
    bad = [s for s in args.seasons if s < EARLIEST_SEASON]
    if bad:
        raise SystemExit(f"Player-prop history only starts in {EARLIEST_SEASON} (May 3, 2023); drop {bad}.")

    for season in args.seasons:
        games = season_games(sched, season)
        ps = (history_loader or (lambda s: build_history(nfl.load_player_stats([s]).to_pandas())))(season)
        for week, wk in games.groupby("week"):
            events = None
            for _, row in wk.iterrows():
                for off in args.offsets:
                    k = None
                    # find the odds-api event once per week (1 credit), then every game's snapshot
                    if events is None:
                        t0 = wk["kickoff"].min() - pd.Timedelta(hours=3)
                        t1 = wk["kickoff"].max() + pd.Timedelta(hours=3)
                        events, hdr = historical_events(key, t0.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                                        t0.strftime("%Y-%m-%dT%H:%M:%SZ"), t1.strftime("%Y-%m-%dT%H:%M:%SZ"))
                        state["credits_spent"] += last_cost(hdr) or EVENTS_CALL_CREDITS
                        summary["calls"] += 1
                    ev = match_event(events, row["home_team"], row["away_team"], row["kickoff"])
                    if ev is None:
                        summary["unmatched_games"] += 1
                        break
                    k = snap_key(ev["id"], off)
                    if k in state["done"]:
                        summary["skipped_done"] += 1
                        continue
                    if state["credits_spent"] - spent_start + len(markets) * CREDITS_PER_MARKET > args.max_credits:
                        summary["stopped"] = f"credit cap {args.max_credits} reached at {season} week {week}"
                        save_state(state, state_path)
                        return summary
                    snap = (row["kickoff"] - pd.Timedelta(hours=off)).strftime("%Y-%m-%dT%H:%M:%SZ")
                    wrapper, hdr = historical_event_odds(key, ev["id"], snap, markets)
                    state["credits_spent"] += last_cost(hdr)
                    summary["calls"] += 1
                    hist = ps[ps["week"] <= week]
                    game = {"home": row["home_team"], "away": row["away_team"], "week": int(week),
                            "commence": ev["commence_time"]}
                    rows = rows_from_event_odds(wrapper, game, build_roster_index(hist), markets, season,
                                                find_player, MARKETS)
                    summary["rows"] += line_history.append_rows(rows, out_path)
                    state["done"].append(k)
                    save_state(state, state_path)
        log(f"season {season} done ({summary['rows']} rows so far, {state['credits_spent'] - spent_start} credits)")
    summary["credits_spent"] = state["credits_spent"] - spent_start
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="plan", choices=["plan", "probe", "run"])
    ap.add_argument("--seasons", type=int, nargs="+", default=[datetime.date.today().year - 1])
    ap.add_argument("--offsets", type=float, nargs="+", default=list(DEFAULT_OFFSETS_H),
                    help="hours before kickoff for each snapshot (0.5 = closing line)")
    ap.add_argument("--markets", choices=["core", "extended"], default="core")
    ap.add_argument("--execute", action="store_true", help="actually spend credits (run only)")
    ap.add_argument("--max-credits", type=int, default=0, help="hard cap on credits this run may spend (required with --execute)")
    args = ap.parse_args()

    import nflreadpy as nfl
    from props_model import CORE_MARKETS, EXTENDED_MARKETS

    markets = [m for m in dict.fromkeys(list(CORE_MARKETS) + (list(EXTENDED_MARKETS) if args.markets == "extended" else []))
               if not m.endswith("_longest")]

    if args.command == "probe":
        key = odds_api.load_key()
        date = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30)).strftime("%Y-%m-%dT12:00:00Z")
        try:
            events, hdr = historical_events(key, date, date[:10] + "T00:00:00Z",
                                            (pd.Timestamp(date) + pd.Timedelta(days=40)).strftime("%Y-%m-%dT00:00:00Z"))
        except odds_api.OddsApiError as e:
            print(f"NOT AVAILABLE on this key's plan: {e}")
            return
        print(f"Historical data IS available on this key ({len(events)} events listed; {odds_api.credits_line(hdr)}).")
        return

    sched = nfl.load_schedules(list(args.seasons)).to_pandas()
    total = {"games": 0, "snapshots": 0, "odds_credits": 0, "event_list_credits": 0, "total_credits": 0}
    for s in args.seasons:
        g = season_games(sched, s)
        e = estimate(len(g), len(markets), len(args.offsets), g["week"].nunique())
        print(f"{s}: {e['games']} games x {len(markets)} markets x {len(args.offsets)} snapshot(s) -> up to {e['total_credits']:,} credits")
        for k in total:
            total[k] += e[k]
    print(f"TOTAL up to {total['total_credits']:,} credits for {total['snapshots']} snapshots "
          f"(markets: {', '.join(m.replace('player_', '') for m in markets)}). An upper bound: markets with no data aren't charged.")
    st = load_state()
    print(f"Already fetched (resumable): {len(st['done'])} snapshots, {st['credits_spent']:,} credits spent to date.")
    if args.command == "plan" or not args.execute:
        print("Nothing was spent. To run it: add --execute --max-credits N (a paid Odds API plan is required).")
        return
    if args.max_credits <= 0:
        raise SystemExit("--execute needs --max-credits N, a hard cap on what this run may spend.")
    summary = run(args, key=odds_api.load_key())
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
