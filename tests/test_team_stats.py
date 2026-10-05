"""
Tests for team_stats.py (per-game team stats, leak-free rolling features, matchup expectation)
and game_factors.py (correlation + walk-forward machinery).

Run:  python -m unittest discover -s tests -v
"""

import unittest

import numpy as np
import pandas as pd

import game_factors as gf
import team_stats as ts


def _pbp():
    """Game G1: AAA (home) vs BBB. AAA: 3 passes, 2 runs, a kneel; BBB: 2 plays. Hand-countable."""
    rows = []

    def play(team, opp, ptype, yds, fd=0, down=1, conv=0, fail=0, sd=0, qtr=1, drive=1, top="2:00", kneel=0):
        rows.append(dict(game_id="G1", season=2026, week=1, season_type="REG", posteam=team, defteam=opp,
                         play_type=ptype, yards_gained=yds, first_down=fd, down=down, third_down_converted=conv,
                         third_down_failed=fail, xpass=0.5, score_differential=sd, qtr=qtr, drive=drive,
                         drive_time_of_possession=top, qb_kneel=kneel, qb_spike=0, sack=0))
    play("AAA", "BBB", "pass", 10, fd=1)
    play("AAA", "BBB", "run", 4)
    play("AAA", "BBB", "pass", 6, down=3, conv=1, fd=1)
    play("AAA", "BBB", "run", 0, down=3, fail=1)
    play("AAA", "BBB", "pass", 20, qtr=4, sd=-7)                 # 4th quarter: not "neutral"
    play("AAA", "BBB", "run", -1, kneel=1, qtr=4)                # kneel: excluded entirely
    play("AAA", "BBB", "punt", 0)                                 # not a pass/run
    play("BBB", "AAA", "run", 5, fd=1, drive=2, top="1:30")
    play("BBB", "AAA", "pass", 15, drive=2, top="1:30")
    return pd.DataFrame(rows)


class TestTeamGameStats(unittest.TestCase):
    def setUp(self):
        self.tg = ts.team_game_stats(_pbp()).set_index("team")

    def test_counts_exclude_kneels_and_non_plays(self):
        a = self.tg.loc["AAA"]
        self.assertEqual(a["plays"], 5)
        self.assertEqual(a["yards"], 40)
        self.assertAlmostEqual(a["ypp"], 8.0)
        self.assertEqual(a["first_downs"], 2)

    def test_third_down_rate(self):
        a = self.tg.loc["AAA"]
        self.assertEqual(a["third_att"], 2)
        self.assertAlmostEqual(a["third_rate"], 0.5)

    def test_neutral_state_is_tied_or_leading_through_q3(self):
        a = self.tg.loc["AAA"]
        self.assertEqual(a["neutral_plays"], 4)                    # the Q4 trailing pass is out
        self.assertAlmostEqual(a["neutral_pass_rate"], 0.5)
        self.assertAlmostEqual(a["proe"], 0.0)                     # xpass fixture is 0.5

    def test_time_of_possession_counts_each_drive_once(self):
        self.assertEqual(self.tg.loc["AAA", "top_sec"], 120)       # one drive, not one per play
        self.assertEqual(self.tg.loc["BBB", "top_sec"], 90)

    def test_allowed_columns_come_from_the_opponent(self):
        self.assertEqual(self.tg.loc["AAA", "yards_allowed"], self.tg.loc["BBB", "yards"])
        self.assertEqual(self.tg.loc["BBB", "plays_allowed"], 5)
        self.assertAlmostEqual(self.tg.loc["BBB", "ypp_allowed"], 8.0)

    def test_top_parser_handles_garbage(self):
        self.assertEqual(ts._top_to_seconds("3:33"), 213)
        self.assertTrue(np.isnan(ts._top_to_seconds(None)))
        self.assertTrue(np.isnan(ts._top_to_seconds("n/a")))


def _team_games(n=6, team="AAA"):
    return pd.DataFrame({"season": 2026, "week": range(1, n + 1), "game_id": [f"g{i}" for i in range(n)],
                         "team": team, "plays": [60.0] * (n - 1) + [100.0]})


class TestRollingFeatures(unittest.TestCase):
    def test_a_games_own_result_never_enters_its_features(self):
        tg = _team_games()                               # last game is a 100-play outlier
        r = ts.rolling_features(tg, cols=["plays"]).set_index("game_id")
        self.assertAlmostEqual(r.loc["g5", "r_plays"], 60.0)          # outlier excluded from its own features
        self.assertTrue(np.isnan(r.loc["g0", "r_plays"]) if "r_plays" in r.loc["g0"] else True)
        self.assertEqual(r.loc["g0", "r_n"], 0)
        self.assertEqual(r.loc["g3", "r_n"], 3)

    def test_recent_games_weigh_more(self):
        tg = _team_games()
        tg.loc[tg.index[:3], "plays"] = 40.0
        tg.loc[tg.index[3:5], "plays"] = 80.0
        r = ts.rolling_features(tg, cols=["plays"]).set_index("game_id")
        self.assertGreater(r.loc["g5", "r_plays"], 60.0)              # leans toward the recent 80s over the older 40s

    def test_current_profile_includes_latest_game(self):
        prof = ts.current_profiles(_team_games(), cols=["plays"])["AAA"]
        self.assertEqual(prof["r_n"], 6)
        self.assertGreater(prof["r_plays"], 60.0)


class TestMatchup(unittest.TestCase):
    def test_expected_ypp_adjusts_for_opposing_defense(self):
        off = {"r_plays": 64, "r_ypp": 5.5}
        weak_def = {"r_plays_allowed": 66, "r_ypp_allowed": 6.0}
        avg_def = {"r_plays_allowed": 66, "r_ypp_allowed": 5.4}
        self.assertAlmostEqual(ts.matchup_expectation(off, weak_def, 5.4, 62)["exp_ypp"], 6.1)
        self.assertAlmostEqual(ts.matchup_expectation(off, avg_def, 5.4, 62)["exp_ypp"], 5.5)

    def test_expected_plays_average_own_pace_and_opponent_allowed(self):
        e = ts.matchup_expectation({"r_plays": 60, "r_ypp": 5}, {"r_plays_allowed": 70, "r_ypp_allowed": 5}, 5, 62)
        self.assertEqual(e["exp_plays"], 65)
        self.assertEqual(e["exp_yards"], 325)

    def test_profile_is_none_without_both_teams(self):
        self.assertIsNone(ts.matchup_profile({"AAA": {"r_n": 3}}, "AAA", "BBB", 5.4, 62))


class TestCorrelationMachinery(unittest.TestCase):
    def test_perfect_correlation_and_significance(self):
        x = pd.Series(np.arange(100.0))
        r, n, p = gf.corr_with_p(x, 2 * x + 1)
        self.assertAlmostEqual(r, 1.0, places=6)
        self.assertEqual(n, 100)
        self.assertLess(p, 1e-6)

    def test_constant_or_tiny_input_is_nan_not_a_crash(self):
        self.assertTrue(np.isnan(gf.corr_with_p(pd.Series([1.0] * 50), pd.Series(np.arange(50.0)))[0]))
        self.assertTrue(np.isnan(gf.corr_with_p(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]))[0]))

    def test_walk_forward_finds_a_real_signal_and_rejects_noise(self):
        rng = np.random.default_rng(0)
        n = 1200
        gt = pd.DataFrame({"season": np.repeat([2021, 2022, 2023, 2024], n // 4),
                           "signal": rng.normal(size=n), "noise": rng.normal(size=n),
                           "total_line": 45 + rng.normal(size=n)})
        gt["resid"] = 3 * gt["signal"] + rng.normal(scale=3, size=n)          # genuinely predictable
        wf = gf.walk_forward(gt, ["signal", "noise"], "resid", "total_line", 2023)
        self.assertGreater(wf["pooled"]["r2_oos"], 0.1)
        gt["resid"] = rng.normal(scale=3, size=n)                              # nothing to find
        wf = gf.walk_forward(gt, ["signal", "noise"], "resid", "total_line", 2023)
        self.assertLess(wf["pooled"]["r2_oos"], 0.02)

    def test_walk_forward_never_trains_on_the_season_it_scores(self):
        gt = pd.DataFrame({"season": [2021] * 250 + [2022] * 250, "x": np.arange(500.0), "total_line": 45.0,
                           "resid": np.r_[np.zeros(250), np.full(250, 100.0)] + np.random.default_rng(1).normal(size=500)})
        wf = gf.walk_forward(gt, ["x"], "resid", "total_line", 2022, extra=())
        self.assertEqual([f["season"] for f in wf["folds"]], [2022])           # 2021 is training only
        self.assertLess(wf["pooled"]["r2_oos"], 0.0)                           # 2022's level was unseeable from 2021

    def test_wilson_interval_brackets_the_rate(self):
        lo, hi = gf.wilson(55, 100)
        self.assertLess(lo, 0.55)
        self.assertGreater(hi, 0.55)


if __name__ == "__main__":
    unittest.main()
