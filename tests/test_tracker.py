"""
Tests for props_tracker.py: logging, grading, and the report it builds.

Run:  python -m unittest discover -s tests -v
"""

import datetime
import sys
import types
import unittest
from unittest import mock

import pandas as pd

import props_tracker as pt

PAST_KICKOFF = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=2)).isoformat(timespec="seconds")


def _snapshot(sample=False, props=None):
    return {"season": 2026, "week": 3, "sample": sample, "props": props if props is not None else [
        {"player_id": "p1", "player": "Test Back", "team": "AAA", "opp": "BBB", "pos": "RB",
         "game": "AAA @ BBB", "commence": PAST_KICKOFF, "market": "player_rush_yds",
         "kind": "yards", "line": 60.5, "model_over": 0.55, "model_under": 0.45, "market_over": 0.50,
         "gap": 5.0, "trust": 0.1, "pick": "over", "pick_ev": 0.03,
         "best_over": {"price": -110, "book": "X"}, "best_under": {"price": -110, "book": "X"}},
    ]}


def _game_snapshot(spread_pick="home", games=None):
    return {"season": 2026, "week": 3, "sample": False, "props": [], "games": games if games is not None else [
        {"game": "KC @ BUF", "home": "BUF", "away": "KC", "commence": PAST_KICKOFF,
         "home_spread": -2.5, "total": 47.5, "model_home_cover": 0.63, "model_over": 0.55,
         "fair_home_cover": 0.52, "fair_over": 0.49, "trust": 0.2,
         "spread_pick": spread_pick, "spread_pick_price": {"price": -108, "book": "FanDuel"}, "spread_pick_ev": 0.045,
         "total_pick": "over", "total_pick_price": {"price": -105, "book": "BetMGM"}, "total_pick_ev": 0.028},
    ]}


class TestLogging(unittest.TestCase):
    def setUp(self):
        self.tmp = pd.io.common.get_handle  # unused; just isolate LOG_FILE per test
        self._orig = pt.LOG_FILE
        pt.LOG_FILE = pt.LOG_FILE.parent / "_test_props_log.csv"
        if pt.LOG_FILE.exists():
            pt.LOG_FILE.unlink()

    def tearDown(self):
        if pt.LOG_FILE.exists():
            pt.LOG_FILE.unlink()
        pt.LOG_FILE = self._orig

    def test_sample_snapshots_are_never_logged(self):
        self.assertEqual(pt.log_predictions(_snapshot(sample=True)), 0)
        self.assertFalse(pt.LOG_FILE.exists())

    def test_logs_pick_price_from_the_recommended_side(self):
        n = pt.log_predictions(_snapshot())
        self.assertEqual(n, 1)
        df = pt._load()
        self.assertEqual(df.iloc[0]["pick_price"], -110)
        self.assertEqual(df.iloc[0]["player"], "Test Back")

    def test_rerunning_updates_in_place_rather_than_duplicating(self):
        pt.log_predictions(_snapshot())
        snap2 = _snapshot()
        snap2["props"][0]["line"] = 65.5      # the line moved before kickoff
        pt.log_predictions(snap2)
        df = pt._load()
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["line"], 65.5)

    def test_props_without_a_player_id_are_skipped_not_crashed(self):
        bad = _snapshot(props=[{"market": "player_rush_yds", "player": "No Id"}])
        self.assertEqual(pt.log_predictions(bad), 0)


class TestGrading(unittest.TestCase):
    def setUp(self):
        self._orig = pt.LOG_FILE
        pt.LOG_FILE = pt.LOG_FILE.parent / "_test_props_log.csv"
        pt.log_predictions(_snapshot())      # commence 2026-09-27T17:00:00Z: safely in the past for grading

    def tearDown(self):
        if pt.LOG_FILE.exists():
            pt.LOG_FILE.unlink()
        pt.LOG_FILE = self._orig

    def _fake_nfl(self, rows, schedule_rows=None):
        mod = types.SimpleNamespace(
            load_player_stats=lambda seasons: types.SimpleNamespace(to_pandas=lambda: pd.DataFrame(rows)),
            load_schedules=lambda seasons: types.SimpleNamespace(to_pandas=lambda: pd.DataFrame(schedule_rows or [])))
        return mock.patch.dict(sys.modules, {"nflreadpy": mod})

    def test_grades_a_finished_game_as_over_or_under(self):
        with self._fake_nfl([{"player_id": "p1", "season": 2026, "week": 3, "season_type": "REG", "rushing_yards": 80.0}]):
            n = pt.grade_pending(delay_hours=0)
        self.assertEqual(n, 1)
        df = pt._load()
        self.assertEqual(df.iloc[0]["result"], "over")
        self.assertEqual(df.iloc[0]["actual"], 80.0)

    def test_exact_line_is_a_push(self):
        with self._fake_nfl([{"player_id": "p1", "season": 2026, "week": 3, "season_type": "REG", "rushing_yards": 60.5}]):
            pt.grade_pending(delay_hours=0)
        self.assertEqual(pt._load().iloc[0]["result"], "push")

    def test_player_missing_from_results_is_marked_dnp_not_left_hanging(self):
        # a real week's stats always have columns; p1 just isn't in them (inactive, bye, bad id)
        with self._fake_nfl([{"player_id": "someone_else", "season": 2026, "week": 3, "season_type": "REG", "rushing_yards": 10.0}]):
            n = pt.grade_pending(delay_hours=0)
        self.assertEqual(n, 0)
        self.assertEqual(pt._load().iloc[0]["result"], "dnp")

    def test_future_games_are_not_graded_yet(self):
        future = _snapshot(props=[{**_snapshot()["props"][0], "player_id": "p2", "commence": "2099-01-01T00:00:00Z"}])
        pt.log_predictions(future)
        with self._fake_nfl([{"player_id": "p1", "season": 2026, "week": 3, "season_type": "REG", "rushing_yards": 80.0}]):
            pt.grade_pending(delay_hours=0)
        df = pt._load().set_index("player_id")
        self.assertTrue(pd.isna(df.loc["p2", "result"]))

    def test_anytime_td_grades_on_one_or_more_not_the_line(self):
        pt.log_predictions(_snapshot(props=[{**_snapshot()["props"][0], "player_id": "p3", "market": "player_anytime_td",
                                            "kind": "td", "line": 0.5}]))
        with self._fake_nfl([{"player_id": "p3", "season": 2026, "week": 3, "season_type": "REG",
                              "rushing_tds": 1.0, "receiving_tds": 0.0}]):
            pt.grade_pending(delay_hours=0)
        self.assertEqual(pt._load().set_index("player_id").loc["p3", "result"], "over")


class TestGameTracking(unittest.TestCase):
    def setUp(self):
        self._orig = pt.LOG_FILE
        pt.LOG_FILE = pt.LOG_FILE.parent / "_test_props_log.csv"

    def tearDown(self):
        if pt.LOG_FILE.exists():
            pt.LOG_FILE.unlink()
        pt.LOG_FILE = self._orig

    def _fake_nfl(self, schedule_rows):
        mod = types.SimpleNamespace(
            load_player_stats=lambda seasons: types.SimpleNamespace(to_pandas=lambda: pd.DataFrame([])),
            load_schedules=lambda seasons: types.SimpleNamespace(to_pandas=lambda: pd.DataFrame(schedule_rows)))
        return mock.patch.dict(sys.modules, {"nflreadpy": mod})

    def test_logs_both_spread_and_total_with_over_under_vocabulary(self):
        n = pt.log_predictions(_game_snapshot(spread_pick="home"))
        self.assertEqual(n, 2)
        df = pt._load().set_index("market")
        self.assertEqual(df.loc["game_spread", "pick"], "over")     # "home" translates to "over"
        self.assertEqual(df.loc["game_spread", "line"], -2.5)
        self.assertEqual(df.loc["game_total", "pick"], "over")
        self.assertEqual(df.loc["game_total", "line"], 47.5)

    def test_away_pick_translates_to_under(self):
        pt.log_predictions(_game_snapshot(spread_pick="away"))
        self.assertEqual(pt._load().set_index("market").loc["game_spread", "pick"], "under")

    def test_spread_grading_uses_the_correct_sign(self):
        # BUF (home) favored by 2.5 (home_spread -2.5); BUF wins by 10 -> comfortably covers -> "over"
        pt.log_predictions(_game_snapshot())
        rows = [{"home_team": "BUF", "away_team": "KC", "week": 3, "home_score": 27, "away_score": 17}]
        with self._fake_nfl(rows):
            n = pt.grade_pending(delay_hours=0)
        self.assertEqual(n, 2)
        df = pt._load().set_index("market")
        self.assertEqual(df.loc["game_spread", "result"], "over")
        self.assertEqual(df.loc["game_spread", "actual"], 10.0)

    def test_spread_grading_when_favorite_wins_but_doesnt_cover(self):
        # BUF favored by 2.5, wins by only 1 -> does NOT cover -> "under"
        pt.log_predictions(_game_snapshot())
        rows = [{"home_team": "BUF", "away_team": "KC", "week": 3, "home_score": 21, "away_score": 20}]
        with self._fake_nfl(rows):
            pt.grade_pending(delay_hours=0)
        self.assertEqual(pt._load().set_index("market").loc["game_spread", "result"], "under")

    def test_spread_push_at_the_exact_margin(self):
        pt.log_predictions(_game_snapshot())
        rows = [{"home_team": "BUF", "away_team": "KC", "week": 3, "home_score": 22.5, "away_score": 20.0}]
        with self._fake_nfl(rows):
            pt.grade_pending(delay_hours=0)
        self.assertEqual(pt._load().set_index("market").loc["game_spread", "result"], "push")

    def test_total_grading(self):
        pt.log_predictions(_game_snapshot())  # total line 47.5
        rows = [{"home_team": "BUF", "away_team": "KC", "week": 3, "home_score": 30, "away_score": 24}]  # 54 total
        with self._fake_nfl(rows):
            pt.grade_pending(delay_hours=0)
        df = pt._load().set_index("market")
        self.assertEqual(df.loc["game_total", "result"], "over")
        self.assertEqual(df.loc["game_total", "actual"], 54.0)

    def test_unfinished_game_is_left_pending_not_marked_dnp(self):
        # the game market's actual result requires a *final score*, unlike a player prop, which
        # has an unambiguous "he didn't play" fallback -- an in-progress game must wait, not fail
        pt.log_predictions(_game_snapshot())
        with self._fake_nfl([{"home_team": "BUF", "away_team": "KC", "week": 3, "home_score": None, "away_score": None}]):
            n = pt.grade_pending(delay_hours=0)
        self.assertEqual(n, 0)
        self.assertTrue(pd.isna(pt._load().iloc[0]["result"]))

    def test_player_grading_is_unaffected_when_no_games_are_pending(self):
        # regression check: grading pure player props must not require load_schedules at all
        pt.log_predictions(_snapshot())
        mod = types.SimpleNamespace(load_player_stats=lambda seasons: types.SimpleNamespace(
            to_pandas=lambda: pd.DataFrame([{"player_id": "p1", "season": 2026, "week": 3,
                                            "season_type": "REG", "rushing_yards": 80.0}])))
        with mock.patch.dict(sys.modules, {"nflreadpy": mod}):
            n = pt.grade_pending(delay_hours=0)   # would raise AttributeError if load_schedules were called
        self.assertEqual(n, 1)

    def test_game_rows_are_excluded_from_player_history(self):
        pt.log_predictions(_game_snapshot())
        with self._fake_nfl([{"home_team": "BUF", "away_team": "KC", "week": 3, "home_score": 27, "away_score": 17}]):
            pt.grade_pending(delay_hours=0)
        self.assertEqual(pt.build_player_history(), {})


class TestReport(unittest.TestCase):
    def setUp(self):
        self._orig = pt.LOG_FILE
        pt.LOG_FILE = pt.LOG_FILE.parent / "_test_props_log.csv"

    def tearDown(self):
        if pt.LOG_FILE.exists():
            pt.LOG_FILE.unlink()
        pt.LOG_FILE = self._orig

    def _graded_df(self, rows):
        base = {"season": 2026, "week": 3, "kind": "yards", "model_over": 0.5, "market": "player_rush_yds"}
        pt._save(pd.DataFrame([{**base, **r} for r in rows]))

    def test_empty_log_reports_zero_gracefully(self):
        rep = pt.build_report()
        self.assertEqual(rep["n_graded"], 0)

    def test_win_rate_and_roi_only_count_graded_picks(self):
        self._graded_df([
            {"player_id": "1", "pick": "over", "pick_price": -110, "result": "over"},   # win: +0.909
            {"player_id": "2", "pick": "over", "pick_price": -110, "result": "under"},  # loss: -1
            {"player_id": "3", "pick": "under", "pick_price": 120, "result": "under"},  # win: +1.2
            {"player_id": "4", "pick": None, "pick_price": None, "result": "over"},     # no pick: excluded
        ])
        rep = pt.build_report()
        self.assertEqual(rep["n_graded"], 4)
        self.assertEqual(rep["n_picks"], 3)
        self.assertAlmostEqual(rep["pick_win_rate"], 2 / 3, places=3)
        self.assertAlmostEqual(rep["roi_per_dollar"], (0.909090909 - 1 + 1.2) / 3, places=3)

    def test_push_neither_wins_nor_loses(self):
        self._graded_df([{"player_id": "1", "pick": "over", "pick_price": -110, "result": "push"}])
        rep = pt.build_report()
        self.assertEqual(rep["roi_per_dollar"], 0.0)

    def test_week_review_returns_everything_logged_for_that_week_graded_or_not(self):
        self._graded_df([
            {"player_id": "1", "player": "A", "season": 2026, "week": 3, "pick": "over", "pick_price": -110, "result": "over"},
            {"player_id": "2", "player": "B", "season": 2026, "week": 3, "pick": None, "pick_price": None, "result": "dnp"},
            {"player_id": "3", "player": "C", "season": 2026, "week": 3, "pick": "under", "pick_price": -110, "result": float("nan")},
            {"player_id": "4", "player": "D", "season": 2026, "week": 4, "pick": "over", "pick_price": -110, "result": "under"},
        ])
        g, season, week = pt.week_review(2026, 3)
        self.assertEqual(season, 2026)
        self.assertEqual(week, 3)
        self.assertEqual(set(g["player_id"]), {"1", "2", "3"})
        row1 = g.set_index("player_id").loc["1"]
        self.assertEqual(row1["hit_pick"], 1.0)

    def test_week_review_defaults_to_the_latest_graded_week(self):
        self._graded_df([
            {"player_id": "1", "player": "A", "season": 2026, "week": 3, "pick": "over", "pick_price": -110, "result": "over"},
            {"player_id": "2", "player": "B", "season": 2026, "week": 5, "pick": None, "pick_price": None, "result": float("nan")},
        ])
        g, season, week = pt.week_review()
        self.assertEqual(week, 3)          # week 5's only row is ungraded, so it's not "the latest graded week"

    def test_week_review_on_an_empty_log_does_not_crash(self):
        g, season, week = pt.week_review()
        self.assertTrue(g.empty)
        pt.print_week_review(g, season, week)   # should just print a message, not raise

    def test_player_history_groups_by_player_and_market_and_drops_dnp(self):
        self._graded_df([
            {"player_id": "1", "player": "A", "team": "AAA", "pos": "WR", "opp": "X", "market": "player_reception_yds",
             "season": 2026, "week": 1, "line": 50.5, "actual": 62.0, "result": "over"},
            {"player_id": "1", "player": "A", "team": "AAA", "pos": "WR", "opp": "Y", "market": "player_reception_yds",
             "season": 2026, "week": 2, "line": 48.5, "actual": 40.0, "result": "under"},
            {"player_id": "1", "player": "A", "team": "AAA", "pos": "WR", "opp": "Z", "market": "player_reception_yds",
             "season": 2026, "week": 3, "line": 55.5, "actual": None, "result": "dnp"},
        ])
        hist = pt.build_player_history()
        self.assertEqual(set(hist), {"1"})
        m = hist["1"]["markets"]["player_reception_yds"]
        self.assertEqual(m["label"], "Rec Yds")
        self.assertEqual(len(m["games"]), 2)              # the dnp week is excluded
        self.assertEqual([g["week"] for g in m["games"]], [1, 2])   # sorted chronologically

    def test_player_history_on_an_empty_log_is_an_empty_dict(self):
        self.assertEqual(pt.build_player_history(), {})

    def test_calibration_table_sums_to_roughly_all_rows(self):
        # model_over is always a real probability in [0, 1] in production
        self._graded_df([{"player_id": str(i), "pick": None, "pick_price": None,
                          "result": "over" if i % 2 else "under", "model_over": 0.05 + 0.045 * i}
                         for i in range(20)])
        rep = pt.build_report()
        self.assertEqual(sum(c["n"] for c in rep["calibration_adjusted"]), 20)


if __name__ == "__main__":
    unittest.main()
