"""
Tests for straight_up.py (win probabilities from the spread, confidence ranking, the weekly slate) and for the
tracker logging and grading the straight-up picks.

Run:  python -m unittest discover -s tests -v
"""

import sys
import types
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import props_tracker as pt
import straight_up as su


class TestWinProb(unittest.TestCase):
    def test_sign_convention_negative_spread_means_home_favored(self):
        self.assertGreater(su.win_prob(-7.0), 0.5)
        self.assertLess(su.win_prob(+7.0), 0.5)
        self.assertEqual(su.win_prob(0.0), 0.5)

    def test_symmetric_and_monotonic(self):
        self.assertAlmostEqual(su.win_prob(-3.0) + su.win_prob(+3.0), 1.0, places=9)
        ps = [su.win_prob(s) for s in (-14, -10, -7, -3, 0, 3, 7, 10, 14)]
        self.assertEqual(ps, sorted(ps, reverse=True))

    def test_values_match_the_nfl_rules_of_thumb(self):
        # a 3-point favorite wins ~60%, a touchdown favorite ~73%, a two-score favorite ~84%
        self.assertAlmostEqual(su.win_prob(-3.0), 0.60, delta=0.02)
        self.assertAlmostEqual(su.win_prob(-7.0), 0.73, delta=0.02)
        self.assertAlmostEqual(su.win_prob(-14.0), 0.89, delta=0.03)


def _entry(game, home, away, spread):
    return {"game": game, "home": home, "away": away, "spread": spread, "p_home": su.win_prob(spread)}


class TestRanking(unittest.TestCase):
    def test_most_confident_gets_the_most_points_and_points_are_a_permutation(self):
        es = su.rank_slate([_entry("B @ A", "A", "B", -3.0), _entry("D @ C", "C", "D", -10.0), _entry("F @ E", "E", "F", +1.0)])
        self.assertEqual([e["game"] for e in es], ["D @ C", "B @ A", "F @ E"])
        self.assertEqual([e["points"] for e in es], [3, 2, 1])
        self.assertEqual([e["rank"] for e in es], [1, 2, 3])
        self.assertEqual(sorted(e["points"] for e in es), [1, 2, 3])

    def test_a_road_favorite_is_picked_and_ranked_by_the_probability_of_the_pick(self):
        es = su.rank_slate([_entry("B @ A", "A", "B", +9.0), _entry("D @ C", "C", "D", -4.0)])
        top = es[0]
        self.assertEqual((top["pick"], top["pick_side"], top["opponent"]), ("B", "away", "A"))      # home +9 -> the visitor
        self.assertAlmostEqual(top["pick_prob"], 1 - su.win_prob(9.0))
        self.assertGreater(top["pick_prob"], es[1]["pick_prob"])

    def test_ties_break_deterministically_so_the_order_never_flips(self):
        a = su.rank_slate([_entry("Z @ Y", "Y", "Z", -3.0), _entry("B @ A", "A", "B", -3.0), _entry("D @ C", "C", "D", +3.0)])
        b = su.rank_slate(list(reversed([_entry("Z @ Y", "Y", "Z", -3.0), _entry("B @ A", "A", "B", -3.0), _entry("D @ C", "C", "D", +3.0)])))
        self.assertEqual([e["game"] for e in a], [e["game"] for e in b])

    def test_a_pick_em_picks_the_home_team_at_low_confidence(self):
        e = su.rank_slate([_entry("B @ A", "A", "B", 0.0)])[0]
        self.assertEqual(e["pick"], "A")
        self.assertEqual(e["points"], 1)


def _sched():
    rows = [
        # still to come, with a different line posted in the sheet than the live one
        {"season": 2026, "week": 5, "game_type": "REG", "gameday": "2026-10-11", "gametime": "13:00", "home_team": "AAA", "away_team": "BBB",
         "home_score": np.nan, "away_score": np.nan, "spread_line": 3.0},
        # finished: the favorite (CCC, home by 10) LOST
        {"season": 2026, "week": 5, "game_type": "REG", "gameday": "2026-10-08", "gametime": "20:15", "home_team": "CCC", "away_team": "DDD",
         "home_score": 16, "away_score": 24, "spread_line": 10.0},
        # finished in a tie
        {"season": 2026, "week": 5, "game_type": "REG", "gameday": "2026-10-11", "gametime": "13:00", "home_team": "EEE", "away_team": "FFF",
         "home_score": 20, "away_score": 20, "spread_line": 1.0},
        # no line posted at all
        {"season": 2026, "week": 5, "game_type": "REG", "gameday": "2026-10-12", "gametime": "20:15", "home_team": "GGG", "away_team": "HHH",
         "home_score": np.nan, "away_score": np.nan, "spread_line": np.nan},
        # a different week and a postseason game: never in this slate
        {"season": 2026, "week": 6, "game_type": "REG", "gameday": "2026-10-18", "gametime": "13:00", "home_team": "III", "away_team": "JJJ",
         "home_score": np.nan, "away_score": np.nan, "spread_line": 4.0},
        {"season": 2026, "week": 5, "game_type": "POST", "gameday": "2026-10-11", "gametime": "13:00", "home_team": "KKK", "away_team": "LLL",
         "home_score": np.nan, "away_score": np.nan, "spread_line": 4.0},
    ]
    return pd.DataFrame(rows)


class TestSlate(unittest.TestCase):
    def setUp(self):
        self.slate = {e["game"]: e for e in su.build_slate(_sched(), 5, live_spreads={("AAA", "BBB"): -6.5}, model_p={("AAA", "BBB"): 0.61}, season=2026)}

    def test_only_this_weeks_regular_season_games_with_a_line_are_ranked(self):
        self.assertEqual(set(self.slate), {"BBB @ AAA", "DDD @ CCC", "FFF @ EEE"})

    def test_an_upcoming_game_uses_the_live_line_not_the_sheets(self):
        e = self.slate["BBB @ AAA"]
        self.assertEqual((e["source"], e["spread"], e["status"]), ("live", -6.5, "upcoming"))
        self.assertEqual(e["pick"], "AAA")
        self.assertAlmostEqual(e["p_model"], 0.61)
        self.assertNotIn("winner", e)

    def test_a_finished_game_keeps_its_slot_uses_the_closing_line_and_records_the_result(self):
        e = self.slate["DDD @ CCC"]
        self.assertEqual((e["source"], e["spread"], e["status"]), ("closing", -10.0, "final"))   # nflverse +10 home = odds sign -10
        self.assertEqual((e["pick"], e["winner"], e["correct"]), ("CCC", "DDD", False))
        self.assertEqual(e["points"], 3)                                                           # the biggest favorite: the top slot

    def test_a_tie_is_neither_right_nor_wrong(self):
        e = self.slate["FFF @ EEE"]
        self.assertEqual(e["winner"], "TIE")
        self.assertIsNone(e["correct"])

    def test_kickoff_is_converted_from_eastern_to_utc(self):
        self.assertEqual(self.slate["DDD @ CCC"]["kickoff"], "2026-10-09T00:15:00Z")               # 8:15 PM EDT = 00:15 UTC next day
        self.assertEqual(self.slate["BBB @ AAA"]["kickoff"], "2026-10-11T17:00:00Z")

    def test_summary_counts_points_won_and_lost_and_the_expectation(self):
        es = list(self.slate.values())
        sm = su.pool_summary(es)
        self.assertEqual((sm["games"], sm["max_points"], sm["final"]), (3, 6, 2))
        self.assertEqual((sm["won"], sm["lost"]), (0, 3))                                          # only the CCC loss counts as lost
        self.assertAlmostEqual(sm["expected_points"], round(sum(e["points"] * e["pick_prob"] for e in es), 1))

    def test_an_empty_week_is_an_empty_slate(self):
        self.assertEqual(su.build_slate(_sched(), 17, season=2026), [])


class TestTrackerLogsAndGradesThePicks(unittest.TestCase):
    def setUp(self):
        self._orig = pt.LOG_FILE
        pt.LOG_FILE = pt.LOG_FILE.parent / "_test_props_log.csv"
        slate = su.build_slate(_sched(), 5, {("AAA", "BBB"): -6.5}, season=2026)
        self.snap = {"season": 2026, "week": 5, "sample": False, "props": [], "games": [], "su_pool": {"entries": slate}}

    def tearDown(self):
        if pt.LOG_FILE.exists():
            pt.LOG_FILE.unlink()
        pt.LOG_FILE = self._orig

    def test_only_games_still_to_come_are_logged_as_straight_up_picks(self):
        self.assertEqual(pt.log_predictions(self.snap), 1)                          # the finished and tied games are not
        r = pt._load().iloc[0]
        self.assertEqual((r["market"], r["player_id"], r["line"], r["kind"]), ("game_su", "GAME_AAA_BBB_su", 0.0, "su"))
        self.assertAlmostEqual(r["model_over"], su.win_prob(-6.5))
        self.assertEqual(r["pick"], "over")                                         # the home team is the pick
        self.assertEqual(r["commence"], "2026-10-11T17:00:00Z")

    def test_a_road_pick_is_logged_as_under(self):
        self.snap["su_pool"]["entries"] = su.build_slate(_sched(), 5, {("AAA", "BBB"): +4.0}, season=2026)
        pt.log_predictions(self.snap)
        self.assertEqual(pt._load().iloc[0]["pick"], "under")

    def test_the_pick_is_graded_on_whether_the_home_team_won(self):
        pt.log_predictions(self.snap)
        df = pt._load()
        df["commence"] = "2020-01-01T17:00:00Z"                                     # long past, so it is due for grading
        pt._save(df)
        final = [{"home_team": "AAA", "away_team": "BBB", "week": 5, "home_score": 24, "away_score": 17}]
        mod = types.SimpleNamespace(
            load_player_stats=lambda seasons: types.SimpleNamespace(to_pandas=lambda: pd.DataFrame([])),
            load_schedules=lambda seasons: types.SimpleNamespace(to_pandas=lambda: pd.DataFrame(final)))
        with mock.patch.dict(sys.modules, {"nflreadpy": mod}):
            self.assertEqual(pt.grade_pending(delay_hours=0), 1)
        r = pt._load().iloc[0]
        self.assertEqual((r["result"], r["actual"]), ("over", 7.0))                  # home won by 7
        final[0].update(home_score=10, away_score=13)
        df = pt._load(); df["result"] = np.nan; df["actual"] = np.nan; df["graded_at"] = np.nan; pt._save(df)
        with mock.patch.dict(sys.modules, {"nflreadpy": mod}):
            pt.grade_pending(delay_hours=0)
        self.assertEqual(pt._load().iloc[0]["result"], "under")                      # home lost: the pick (over) missed


if __name__ == "__main__":
    unittest.main()
