"""
collect_lines.py
================
The cheap off-day collector. The main refresh (export_props.py) only runs Sunday, Monday and
Thursday and pulls props for the next 24 hours, so a game's line is seen a handful of times. Game
spreads and totals for the WHOLE slate cost a flat ~2 Odds API credits in one call, regardless of
how many games, so this script pulls them on the other days too and appends them to the backlog:

    python collect_lines.py            # game lines -> tracking/line_history.csv
                                       # + the power rankings -> tracking/power_history.csv

No simulation, no props, no site rebuild, so it takes seconds and costs ~2 credits per run (the
props pull is the expensive one, and it stays on the main schedule). It never touches
tracking/props_log.csv, which only the main refresh writes.
"""

import sys

import nflreadpy as nfl

import line_history
import odds_api
import props_tracker
import team_ratings
import team_stats


def log(msg):
    print(msg, file=sys.stderr)


def main():
    key = odds_api.load_key()
    season = int(nfl.get_current_season())
    pulled_at = line_history.now_iso()

    raw, headers = odds_api.fetch_game_lines(key)
    log(f"Game lines pulled. {odds_api.credits_line(headers)}")
    # a game in progress has LIVE lines; they would corrupt "closing line", so skip them
    events = [e for e in raw if not odds_api.has_started(e)]
    lines, prices = odds_api.consensus_game_lines(events), odds_api.consensus_game_prices(events)

    sched = nfl.load_schedules([season]).to_pandas()
    wk_default = int(nfl.get_current_week())
    week_for = lambda h, a, c: props_tracker.week_for_game(sched, h, a, c, default=wk_default)
    rows = line_history.rows_from_game_lines(events, lines, prices, week_for, season, pulled_at)
    n = line_history.append_rows(rows)
    log(f"Backlog: appended {n} game-line rows for {len(events)} upcoming games.")

    try:
        tg = team_stats.team_game_stats(team_stats.load_pbp(range(season - 2, season + 1)))
        sched3 = nfl.load_schedules(list(range(season - 2, season + 1))).to_pandas()
        games = team_ratings.build_games(tg, sched3[sched3["game_type"] == "REG"])
        n = team_ratings.snapshot_power_history(games, season, pulled_at)
        log(f"Backlog: stored {n} power-ranking rows.")
    except Exception as e:                       # the lines are already saved; rankings are a bonus here
        log(f"Power rankings not stored this run ({e}).")


if __name__ == "__main__":
    main()
