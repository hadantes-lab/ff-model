"""
Tests for the defense-vs-position signal (dvp_signal.py) and its weekly log (dvp_track.py): no look-ahead, the arithmetic of each
reading, and the log's upsert / grading / accuracy.

Run:  python -m unittest discover -s tests -v
"""

import pathlib
import tempfile
import unittest

import numpy as np
import pandas as pd

import dvp_signal as ds
import dvp_track as dt


def _rows(season, week, off, deff, wr_targets, rb_carries=10):
    """One offense's game: a WR and an RB, plus an FB that must count as an RB."""
    base = {"season": season, "week": week, "team": off, "opponent_team": deff}
    return [
        {**base, "position": "WR", "targets": wr_targets, "carries": 0, "receptions": 0, "receiving_yards": 0, "rushing_yards": 0,
         "attempts": 0, "completions": 0, "passing_yards": 0, "passing_tds": 0, "rushing_tds": 0, "receiving_tds": 0},
        {**base, "position": "RB", "targets": 0, "carries": rb_carries, "receptions": 0, "receiving_yards": 0, "rushing_yards": 0,
         "attempts": 0, "completions": 0, "passing_yards": 0, "passing_tds": 0, "rushing_tds": 0, "receiving_tds": 0},
        {**base, "position": "FB", "targets": 0, "carries": 1, "receptions": 0, "receiving_yards": 0, "rushing_yards": 0,
         "attempts": 0, "completions": 0, "passing_yards": 0, "passing_tds": 0, "rushing_tds": 0, "receiving_tds": 0},
    ]


def _history(spec):
    """spec: [(season, week, offense, defense, wr_targets)]"""
    return pd.DataFrame([r for s in spec for r in _rows(*s)])


class TestSignal(unittest.TestCase):
    def setUp(self):
        # four teams, season 2024, weeks 1-3; A and B are defenses
        self.spec = [
            (2024, 1, "X", "A", 10), (2024, 1, "Y", "B", 6),
            (2024, 2, "Y", "A", 12), (2024, 2, "X", "B", 4),
            (2024, 3, "X", "A", 14), (2024, 3, "Y", "B", 5),
        ]
        self.h = _history(self.spec)
        self.gt = ds.with_expectation(ds.game_table(self.h, self.h["targets"]))

    def test_game_table_groups_and_zero_fills(self):
        gt = ds.game_table(self.h, self.h["carries"])
        wk1 = gt[(gt["week"] == 1) & (gt["off"] == "X")]
        self.assertEqual(set(wk1["grp"]), set(ds.GROUPS))
        self.assertEqual(float(wk1[wk1["grp"] == "RB"]["v"].iloc[0]), 11.0)       # RB 10 carries + FB 1
        self.assertEqual(float(wk1[wk1["grp"] == "QB"]["v"].iloc[0]), 0.0)        # no QB in the data: a zero, not a missing game

    def test_expectation_uses_only_earlier_games(self):
        wr = self.gt[self.gt["grp"] == "WR"]
        x = wr[wr["off"] == "X"].sort_values("week")
        self.assertTrue(np.isnan(float(x["e"].iloc[0])))                      # week 1 of the first season: nothing earlier to expect from
        self.assertAlmostEqual(float(x["e"].iloc[1]), 10.0)                   # the games so far (no last season to blend)
        self.assertAlmostEqual(float(x["e"].iloc[2]), (10 + 4) / 2)

    def test_future_games_never_change_a_reading(self):
        before = ds.readings_asof(self.gt, 2024, 3)
        later = self.spec + [(2024, 4, "X", "A", 99)]
        gt2 = ds.with_expectation(ds.game_table(_history(later), _history(later)["targets"]))
        after = ds.readings_asof(gt2, 2024, 3)
        pd.testing.assert_frame_equal(before, after)
        changed = self.spec[:-1] + [(2024, 3, "Y", "B", 500)]                   # the very week being predicted
        gt3 = ds.with_expectation(ds.game_table(_history(changed), _history(changed)["targets"]))
        pd.testing.assert_frame_equal(before, ds.readings_asof(gt3, 2024, 3))

    def test_season_reading_is_a_ratio_to_the_league(self):
        r = ds.readings_asof(self.gt, 2024, 3).loc[("WR", "A")]
        # before week 3: A faced X (10) and Y (12); B faced Y (6) and X (4); league mean 8
        self.assertAlmostEqual(r["sea"], 11 / 8)
        self.assertEqual(r["n_sea"], 2)
        self.assertAlmostEqual(ds.readings_asof(self.gt, 2024, 3).loc[("WR", "B"), "sea"], 5 / 8)

    def test_adjusted_reading_discounts_a_strong_schedule(self):
        # only games where the offense had something earlier to be expected from count (week 1 of the first season has none)
        r = ds.readings_asof(self.gt, 2024, 3).loc[("WR", "A")]
        wr = self.gt[(self.gt["grp"] == "WR") & (self.gt["def"] == "A") & (self.gt["week"] < 3) & self.gt["e"].notna()]
        self.assertAlmostEqual(r["adj"], wr["v"].sum() / wr["e"].sum())

    def test_last_season_reading_needs_a_last_season(self):
        self.assertTrue(np.isnan(ds.readings_asof(self.gt, 2024, 3).loc[("WR", "A"), "prev"]))
        spec = [(2023, 1, "X", "A", 20), (2023, 1, "Y", "B", 10)] + self.spec
        gt = ds.with_expectation(ds.game_table(_history(spec), _history(spec)["targets"]))
        r = ds.readings_asof(gt, 2024, 3)
        self.assertAlmostEqual(r.loc[("WR", "A"), "prev"], 20 / 15)

    def test_shrunk(self):
        self.assertAlmostEqual(ds.shrunk(1.5, 6, 6), 1.25)
        self.assertEqual(ds.shrunk(1.5, 0, 6), 1.0)


class TestProduction(unittest.TestCase):
    def setUp(self):
        spec = [(2024, 1, "X", "A", 10), (2024, 1, "Y", "B", 6), (2024, 2, "Y", "A", 12), (2024, 2, "X", "B", 4),
                (2024, 3, "X", "A", 14), (2024, 3, "Y", "B", 5)]
        self.h = _history(spec)

    def test_a_soft_defense_lifts_and_a_tough_one_lowers(self):
        f = ds.factors(self.h, self.h["targets"], 0.5, 2024)
        self.assertGreater(f[("A", "WR")], 1.0)
        self.assertLess(f[("B", "WR")], 1.0)
        self.assertEqual(f[("A", "WR")], f[("A", "WR")])
        self.assertIn(("A", "RB"), f)
        self.assertIn(("A", "FB"), f)                                           # a fullback gets his position group's factor

    def test_weight_scales_the_move_and_the_clip_bounds_it(self):
        small = ds.factors(self.h, self.h["targets"], 0.1, 2024)[("A", "WR")]
        big = ds.factors(self.h, self.h["targets"], 0.5, 2024)[("A", "WR")]
        self.assertGreater(big - 1, small - 1)
        huge = ds.factors(self.h, self.h["targets"], 50.0, 2024)[("A", "WR")]
        self.assertEqual(huge, ds.FACTOR_CLIP[1])

    def test_apply_multiplies_only_markets_with_a_weight(self):
        mults = {"player_receptions": {("A", "WR"): 1.1}, "player_rush_yds": {("A", "WR"): 1.1}}
        specs = {"player_receptions": object(), "player_rush_yds": object()}
        applied = ds.apply_to_multipliers(mults, self.h, specs, lambda h, s: h["targets"], 2024, weights={"player_receptions": 0.5})
        self.assertEqual(applied, {"player_receptions": 0.5})
        self.assertGreater(mults["player_receptions"][("A", "WR")], 1.1)
        self.assertEqual(mults["player_rush_yds"][("A", "WR")], 1.1)

    def test_no_weights_file_means_no_change(self):
        self.assertEqual(ds.load_weights(pathlib.Path(tempfile.gettempdir()) / "no_such_weights.json"), {})
        mults = {"player_receptions": {("A", "WR"): 1.1}}
        ds.apply_to_multipliers(mults, self.h, {"player_receptions": object()}, lambda h, s: h["targets"], 2024, weights={})
        self.assertEqual(mults["player_receptions"][("A", "WR")], 1.1)

    def test_a_season_with_no_games_yet_leans_on_last_season_and_never_errors(self):
        f = ds.factors(self.h, self.h["targets"], 0.5, 2025)                    # 2025 has no games: only the last-four reading exists
        self.assertTrue(all(0.88 <= v <= 1.12 for v in f.values()))


class TestLog(unittest.TestCase):
    def setUp(self):
        spec = [(2024, w, o, d, t) for w, rows in {
            1: [("X", "A", 10), ("Y", "B", 6)], 2: [("Y", "A", 12), ("X", "B", 4)], 3: [("X", "A", 14), ("Y", "B", 5)]}.items()
            for o, d, t in rows]
        self.h = _history(spec)
        self.tmp = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.tmp.name) / "dvp_log.csv"

    def tearDown(self):
        self.tmp.cleanup()

    def test_snapshot_logs_each_defense_by_position_and_stat(self):
        rows = dt.snapshot_rows(self.h, 2024, 3, "now")
        keys = {(r["defense"], r["grp"], r["stat"]) for r in rows}
        self.assertIn(("A", "WR", "tgt"), keys)
        self.assertIn(("B", "RB", "rush_att"), keys)
        self.assertNotIn(("A", "QB", "rush_att"), keys)                          # only the stats listed for each group
        self.assertTrue(all(np.isnan(r["allowed"]) for r in rows))

    def test_grade_fills_what_was_allowed_and_a_rerun_keeps_the_grade(self):
        n = dt.log_snapshot(dt.snapshot_rows(self.h, 2024, 3, "t1"), self.path)
        self.assertGreater(n, 0)
        self.assertGreater(dt.grade(self.h, self.path), 0)
        g = dt.load(self.path)
        row = g[(g["defense"] == "A") & (g["grp"] == "WR") & (g["stat"] == "tgt")].iloc[0]
        self.assertEqual(row["allowed"], 14.0)                                  # A allowed X's 14 WR targets in week 3
        self.assertAlmostEqual(row["league"], (14 + 5) / 2)
        dt.log_snapshot(dt.snapshot_rows(self.h, 2024, 3, "t2"), self.path)     # a later run of the same week
        g2 = dt.load(self.path)
        self.assertEqual(len(g2), len(g))
        self.assertEqual(g2[(g2["defense"] == "A") & (g2["grp"] == "WR") & (g2["stat"] == "tgt")].iloc[0]["allowed"], 14.0)

    def test_ungraded_weeks_stay_blank(self):
        future = dt.snapshot_rows(self.h, 2024, 4, "now")
        dt.log_snapshot(future, self.path)
        self.assertEqual(dt.grade(self.h, self.path), 0)

    def test_accuracy_rewards_a_reading_that_predicts(self):
        rng = np.random.default_rng(1)
        rows = []
        for w in range(1, 9):
            for d in range(32):
                reading = 0.7 + 0.6 * d / 31
                rows.append({"season": 2024, "week": w, "defense": f"T{d}", "grp": "WR", "stat": "rec", "n": 5, "sea": reading,
                             "l4": reading, "prev": rng.uniform(0.7, 1.3), "adj": reading, "allowed": 10 * (reading + rng.normal(0, 0.08)),
                             "league": 10.0, "logged_at": "x"})
        acc = dt.accuracy(pd.DataFrame(rows, columns=dt.COLUMNS))
        self.assertGreater(acc["readings"]["sea"]["corr"], 0.8)
        self.assertLess(abs(acc["readings"]["prev"]["corr"]), 0.3)
        self.assertGreater(acc["readings"]["sea"]["soft"], 0.15)
        self.assertLess(acc["readings"]["sea"]["tough"], -0.15)

    def test_blurb_reads_with_no_graded_week(self):
        self.assertIsNone(dt.section(2024, 3, self.path))


if __name__ == "__main__":
    unittest.main()
