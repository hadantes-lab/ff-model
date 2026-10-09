"""
Tests for backfill_history.py with the Odds API faked: the credit cap, resuming, matching games and
players, what gets stored, and that nothing is spent without --execute.

Run:  python -m unittest discover -s tests -v
"""

import argparse
import pathlib
import tempfile
import unittest
from unittest import mock

import pandas as pd

import backfill_history as bf
import line_history as lh


class Hdr(dict):
    pass


def _sched():
    rows = []
    for wk, (day, home, away) in enumerate([("2024-09-08", "NE", "BUF"), ("2024-09-15", "GB", "CHI"),
                                            ("2024-09-22", "NO", "ATL")], start=1):
        rows.append({"season": 2024, "week": wk, "game_type": "REG", "gameday": day, "gametime": "13:00",
                     "home_team": home, "away_team": away})
    return pd.DataFrame(rows)


def _history(season):
    rows = []
    for wk in (1, 2, 3):
        for pid, name, team in (("qb_ne", "Test Quarterback", "NE"), ("wr_gb", "Test Receiver", "GB"),
                                ("qb_no", "Third Passer", "NO")):
            rows.append({"player_id": pid, "player_display_name": name, "team": team, "position": "QB", "season": season, "week": wk})
    return pd.DataFrame(rows)


def _odds_wrapper(player, ts="2024-09-08T16:30:00Z"):
    def book(title, o, u):
        return {"title": title, "key": title.lower(), "markets": [{"key": "player_pass_yds", "outcomes": [
            {"name": "Over", "description": player, "price": o, "point": 250.5},
            {"name": "Under", "description": player, "price": u, "point": 250.5}]}]}
    return {"timestamp": ts, "data": {"id": "x", "bookmakers": [book("BookA", -110, -110), book("BookB", -105, -115)]}}


class Env:
    """Patch the two API functions; count calls and charge a fixed cost per call."""

    def __init__(self, test, events=None, cost=10):
        self.calls = {"events": 0, "odds": 0}
        self.cost = cost
        names = {"NE": "New England Patriots", "BUF": "Buffalo Bills", "GB": "Green Bay Packers", "CHI": "Chicago Bears",
                 "NO": "New Orleans Saints", "ATL": "Atlanta Falcons"}
        sched = _sched()
        self.events = events if events is not None else [
            {"id": f"e{r.week}", "commence_time": f"{r.gameday}T17:00:00Z", "home_team": names[r.home_team],
             "away_team": names[r.away_team]} for r in sched.itertuples()]
        test.enterContext(mock.patch.object(bf, "historical_events", self._events))
        test.enterContext(mock.patch.object(bf, "historical_event_odds", self._odds))

    def _events(self, key, date, t_from, t_to):
        self.calls["events"] += 1
        lo, hi = pd.Timestamp(t_from), pd.Timestamp(t_to)
        return [e for e in self.events if lo <= pd.Timestamp(e["commence_time"]) <= hi], Hdr({"x-requests-last": "1"})

    def _odds(self, key, event_id, date, markets):
        self.calls["odds"] += 1
        player = {"e1": "Test Quarterback", "e2": "Nobody Matches", "e3": "Third Passer"}[event_id]
        return _odds_wrapper(player, date), Hdr({"x-requests-last": str(self.cost)})


class TestBackfill(unittest.TestCase):
    def setUp(self):
        d = pathlib.Path(tempfile.mkdtemp())
        self.out, self.state = d / "hist.csv", d / "state.json"

    def args(self, **kw):
        base = dict(seasons=[2024], offsets=[0.5], markets="core", max_credits=10_000)
        base.update(kw)
        return argparse.Namespace(**base)

    def go(self, env, **kw):
        return bf.run(self.args(**kw), key="k", history_loader=_history, sched=_sched(),
                      out_path=self.out, state_path=self.state)

    def test_stores_matched_players_with_the_snapshot_time_and_derived_numbers_only(self):
        env = Env(self)
        s = self.go(env)
        d = lh.load(self.out)
        self.assertEqual(set(d["player_id"]), {"qb_ne", "qb_no"})               # week 2's name matches nobody: no row
        self.assertEqual(s["rows"], 2)
        r = d[d["player_id"] == "qb_ne"].iloc[0]
        self.assertEqual((r["season"], r["week"], r["market"], r["line"], r["n_books"]), (2024, 1, "player_pass_yds", 250.5, 2))
        self.assertEqual(r["team"], "NE")
        self.assertEqual(r["opp"], "BUF")
        self.assertEqual(r["pulled_at"], "2024-09-08T16:30:00Z")                # kickoff 17:00Z minus 30 min
        self.assertEqual(r["best_over"], -105)
        self.assertTrue(0.45 < r["fair_over"] < 0.55)
        text = self.out.read_text(encoding="utf-8")
        self.assertNotIn("BookA", text)
        self.assertNotIn("BookB", text)

    def test_one_event_list_call_per_week_and_one_odds_call_per_game(self):
        env = Env(self)
        self.go(env)
        self.assertEqual(env.calls, {"events": 3, "odds": 3})

    def test_rerun_resumes_and_spends_nothing_more(self):
        env = Env(self)
        self.go(env)
        env2 = Env(self)
        s = self.go(env2)
        self.assertEqual(env2.calls["odds"], 0)
        self.assertEqual(s["skipped_done"], 3)
        self.assertEqual(len(lh.load(self.out)), 2)                              # nothing duplicated

    def test_the_credit_cap_stops_the_run_and_the_next_run_continues(self):
        env = Env(self, cost=50)
        # 5 markets x 10 = 50 per game; a cap of 120 allows 2 games (+ event-list calls), not 3
        s = self.go(env, max_credits=120)
        self.assertIn("credit cap", s["stopped"])
        self.assertLess(env.calls["odds"], 3)
        done_after_first = len(bf.load_state(self.state)["done"])
        env2 = Env(self, cost=50)
        s2 = self.go(env2, max_credits=10_000)
        self.assertEqual(env2.calls["odds"], 3 - done_after_first)
        self.assertIsNone(s2["stopped"])

    def test_spend_is_tracked_from_the_apis_own_cost_header(self):
        env = Env(self, cost=37)
        self.go(env)
        self.assertEqual(bf.load_state(self.state)["credits_spent"], 3 * 1 + 3 * 37)

    def test_game_missing_from_the_events_list_is_skipped_not_fatal(self):
        env = Env(self)
        env.events = [e for e in env.events if e["id"] != "e3"]
        s = self.go(env)
        self.assertEqual(s["unmatched_games"], 1)
        self.assertEqual(env.calls["odds"], 2)

    def test_seasons_before_player_props_existed_are_refused(self):
        with self.assertRaises(SystemExit):
            bf.run(self.args(seasons=[2022]), key="k", history_loader=_history, sched=_sched().assign(season=2022),
                   out_path=self.out, state_path=self.state)

    def test_multiple_offsets_make_multiple_snapshots(self):
        env = Env(self)
        s = self.go(env, offsets=[0.5, 120])
        self.assertEqual(env.calls["odds"], 6)
        self.assertEqual(len(bf.load_state(self.state)["done"]), 6)


class TestPlanAndHelpers(unittest.TestCase):
    def test_estimate_is_ten_credits_per_market_per_game_plus_event_lists(self):
        e = bf.estimate(272, 5, 1, 18)
        self.assertEqual(e["odds_credits"], 272 * 5 * 10)
        self.assertEqual(e["event_list_credits"], 18)
        self.assertEqual(e["total_credits"], 13_618)
        self.assertEqual(bf.estimate(272, 5, 2, 18)["snapshots"], 544)

    def test_kickoff_converts_eastern_to_utc(self):
        ts = bf.commence_utc({"gameday": "2024-09-08", "gametime": "13:00"})      # EDT = UTC-4
        self.assertEqual(ts.strftime("%Y-%m-%dT%H:%M"), "2024-09-08T17:00")
        winter = bf.commence_utc({"gameday": "2024-12-08", "gametime": "13:00"})   # EST = UTC-5
        self.assertEqual(winter.strftime("%H:%M"), "18:00")

    def test_match_event_needs_both_teams_and_a_close_kickoff(self):
        ev = [{"id": "a", "home_team": "New England Patriots", "away_team": "Buffalo Bills", "commence_time": "2024-09-08T17:00:00Z"}]
        k = pd.Timestamp("2024-09-08T17:00:00Z")
        self.assertEqual(bf.match_event(ev, "NE", "BUF", k)["id"], "a")
        self.assertIsNone(bf.match_event(ev, "BUF", "NE", k))
        self.assertIsNone(bf.match_event(ev, "NE", "BUF", k + pd.Timedelta(days=3)))     # a different meeting of the same pair

    def test_plan_command_never_calls_the_api_or_writes(self):
        import io
        import contextlib
        with mock.patch.object(bf.odds_api, "load_key", side_effect=AssertionError("plan must not need a key")), \
             mock.patch.object(bf, "historical_events", side_effect=AssertionError("no API call")), \
             mock.patch("sys.argv", ["backfill_history.py", "plan", "--seasons", "2024"]):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                bf.main()
        self.assertIn("Nothing was spent", buf.getvalue())

    def test_execute_without_a_cap_is_refused(self):
        with mock.patch("sys.argv", ["backfill_history.py", "run", "--seasons", "2024", "--execute"]):
            with self.assertRaises(SystemExit):
                bf.main()


class TestClosingMapFeedsTheChart(unittest.TestCase):
    def test_backfilled_closing_lines_become_line_then(self):
        rows = [{"pulled_at": "2024-09-08T16:30:00Z", "season": 2024, "week": 1, "kind": "prop", "market": "player_pass_yds",
                 "player_id": "qb_ne", "commence": "2024-09-08T17:00:00Z", "line": 249.5, "fair_over": 0.5},
                {"pulled_at": "2024-09-08T16:30:00Z", "season": 2024, "week": 1, "kind": "game", "market": "game_total",
                 "player_id": "GAME_NE_BUF_total", "commence": "2024-09-08T17:00:00Z", "line": 44.5, "fair_over": 0.5}]
        m = lh.closing_map(pd.DataFrame(rows))
        self.assertEqual(m[("qb_ne", "player_pass_yds")], {(2024, 1): 249.5})
        self.assertNotIn(("GAME_NE_BUF_total", "game_total"), m)                  # games aren't player props
        self.assertEqual(lh.closing_map(pd.DataFrame(columns=lh.COLUMNS)), {})


class TestPlanErrorMessage(unittest.TestCase):
    def test_free_plan_401_is_explained_as_a_plan_limit_not_a_bad_key(self):
        import io
        import urllib.error
        import odds_api

        body = b'{"message":"Historical odds are only available on paid usage plans.","error_code":"HISTORICAL_UNAVAILABLE_ON_FREE_USAGE_PLAN"}'
        err = urllib.error.HTTPError("u", 401, "Unauthorized", {}, io.BytesIO(body))
        with mock.patch("urllib.request.urlopen", side_effect=err):
            with self.assertRaises(odds_api.OddsApiError) as cm:
                odds_api._get("/historical/x", {}, "secretkey")
        self.assertIn("paid Odds API plan", str(cm.exception))
        self.assertNotIn("rejected", str(cm.exception))
        self.assertNotIn("secretkey", str(cm.exception))                          # the key never appears in errors


if __name__ == "__main__":
    unittest.main()
