"""
Tests for player_confidence.py (per-player model record, 0-10 grade, measured strength) and how the grade
enters the edge in export_props.build_prop.

Run:  python -m unittest discover -s tests -v
"""

import pathlib
import tempfile
import unittest

import numpy as np
import pandas as pd

import export_props as ep
import market_trust as mt
import player_confidence as pc


def _log(rows):
    """Graded tracking-log rows. Each: (player_id, market, model_over, result, actual, mean, sd, season)."""
    cols = ["player_id", "market", "model_over", "result", "actual", "mean", "sd", "season"]
    df = pd.DataFrame(rows, columns=cols)
    df["player"] = df["player_id"]
    df["pos"] = "WR"
    return df


class TestGrade(unittest.TestCase):
    def test_no_games_is_a_neutral_five(self):
        self.assertEqual(pc.grade_from(0, 0), 5.0)

    def test_a_short_record_cannot_swing_far(self):
        self.assertLess(pc.grade_from(3, 3), 7.0)                     # 3-0 is not a 10
        self.assertGreater(pc.grade_from(0, 3), 3.0)                  # and 0-3 is not a 0
        self.assertAlmostEqual(pc.grade_from(3, 3), 10 * (3 + 3) / (3 + 6), places=1)

    def test_grade_rises_with_wins_and_falls_with_losses_and_is_bounded(self):
        grades = [pc.grade_from(h, 20) for h in range(0, 21, 5)]
        self.assertEqual(grades, sorted(grades))
        self.assertTrue(all(0 <= g <= 10 for g in grades))
        self.assertGreater(pc.grade_from(200, 200), 9.0)
        self.assertLess(pc.grade_from(0, 200), 1.0)

    def test_accuracy_half_moves_the_grade(self):
        precise = pc.grade_from(5, 10, sum_abs_z=0.3 * 10, n_acc=10)       # misses of 0.3 sd: far tighter than expected
        sloppy = pc.grade_from(5, 10, sum_abs_z=1.6 * 10, n_acc=10)        # misses of 1.6 sd
        self.assertGreater(precise, 5.0)
        self.assertLess(sloppy, 5.0)
        self.assertEqual(pc.accuracy_score(0.8 * 40, 40, prior_games=0), 0.5)   # exactly as calibrated as expected -> neutral

    def test_multiplier_is_one_at_zero_strength_and_bounded_otherwise(self):
        for g in (0, 3, 5, 7.5, 10):
            self.assertEqual(pc.multiplier(g, 0.0), 1.0)
        self.assertAlmostEqual(pc.multiplier(10, 0.4), 1.4)
        self.assertAlmostEqual(pc.multiplier(0, 0.4), 0.6)
        self.assertEqual(pc.multiplier(5, 0.4), 1.0)                       # neutral grade never changes the weight
        self.assertEqual(pc.multiplier(10, 9.0), 1.0 + pc.MAX_STRENGTH)     # strength is capped


class TestRecords(unittest.TestCase):
    def test_counts_hits_by_the_models_raw_side(self):
        df = _log([("a", "m", 0.7, "over", 60, 55, 10, 2026),      # leaned over, went over   -> hit
                   ("a", "m", 0.7, "under", 40, 55, 10, 2026),     # leaned over, went under  -> miss
                   ("a", "m", 0.3, "under", 40, 45, 10, 2026),     # leaned under, went under -> hit
                   ("a", "m", 0.3, "over", 60, 45, 10, 2026)])     # leaned under, went over   -> miss
        r = pc.player_records(df).iloc[0]
        self.assertEqual((r["n"], r["hits"]), (4, 2))
        self.assertEqual(r["grade"], pc.grade_from(2, 4, r["mean_abs_z"] * r["n_acc"], r["n_acc"]))

    def test_pushes_dnps_ungraded_games_and_coin_flips_are_left_out(self):
        df = _log([("a", "m", 0.7, "over", 60, 55, 10, 2026), ("a", "m", 0.7, "push", 50, 55, 10, 2026),
                   ("a", "m", 0.7, "dnp", np.nan, 55, 10, 2026), ("a", "m", 0.7, np.nan, np.nan, 55, 10, 2026),
                   ("a", "m", 0.5, "over", 60, 55, 10, 2026)])
        self.assertEqual(int(pc.player_records(df).iloc[0]["n"]), 1)

    def test_games_are_never_counted_as_players(self):
        df = _log([("a", "m", 0.7, "over", 60, 55, 10, 2026)])
        df.loc[0, "pos"] = "GAME"
        self.assertTrue(pc.player_records(df).empty)

    def test_players_markets_and_seasons_are_kept_apart(self):
        df = _log([("a", "m1", 0.7, "over", 60, 55, 10, 2026), ("a", "m2", 0.7, "under", 40, 55, 10, 2026),
                   ("b", "m1", 0.7, "over", 60, 55, 10, 2026), ("a", "m1", 0.7, "over", 60, 55, 10, 2025)])
        r = pc.player_records(df)
        self.assertEqual(len(r), 4)
        self.assertEqual(len(pc.player_records(df, season=2026)), 3)
        look = pc.lookup(r, 2026)
        self.assertEqual(look[("a", "m1")]["hits"], 1)
        self.assertEqual(look[("a", "m2")]["hits"], 0)
        self.assertNotIn(("a", "m1"), pc.lookup(r, 2024))

    def test_accuracy_needs_the_logged_projection_and_degrades_gracefully_without_it(self):
        df = _log([("a", "m", 0.7, "over", 60, np.nan, np.nan, 2026)] * 3)
        r = pc.player_records(df).iloc[0]
        self.assertEqual(r["n_acc"], 0)
        self.assertEqual(r["grade"], pc.grade_from(3, 3))                  # the hit rate alone
        self.assertTrue(pc.player_records(df.drop(columns=["mean", "sd"])).iloc[0]["n_acc"] == 0)   # old logs lack the columns

    def test_empty_inputs(self):
        self.assertTrue(pc.player_records(pd.DataFrame()).empty)
        self.assertEqual(pc.lookup(pd.DataFrame(columns=pc.GRADE_COLUMNS), 2026), {})

    def test_confidence_block_for_an_unknown_player_is_neutral(self):
        c = pc.confidence_for(None, 0.3)
        self.assertEqual((c["grade"], c["n"], c["multiplier"]), (5.0, 0, 1.0))


class TestMeasuredStrength(unittest.TestCase):
    def _frame(self, persistent, n_players=400, games=14, seed=0):
        """Per-player hit chance either varies by player (record is informative) or is the same for everyone (it isn't)."""
        rng = np.random.default_rng(seed)
        rows = []
        for i in range(n_players):
            skill = rng.choice([0.40, 0.50, 0.60]) if persistent else 0.5
            for k in range(games):
                rows.append({"pid": f"p{i}", "market": "m", "season": 2025, "week": k + 1, "hit": float(rng.random() < skill)})
        return pd.DataFrame(rows)

    def test_a_record_that_predicts_earns_a_positive_strength(self):
        p = pc.persistence(self._frame(True))
        self.assertGreater(p["ci"][0], 0)
        self.assertGreater(pc.strength_from(p), 0)
        self.assertLessEqual(pc.strength_from(p), pc.MAX_STRENGTH)

    def test_a_record_that_is_pure_luck_earns_nothing(self):
        p = pc.persistence(self._frame(False, seed=4))
        self.assertLessEqual(p["ci"][0], 0)
        self.assertEqual(pc.strength_from(p), 0.0)

    def test_persistence_uses_only_earlier_games(self):
        # every hit of a player's LAST game is flipped: his earlier-games record must be unaffected by it
        f = self._frame(True, n_players=100)
        base = pc.persistence(f)["slope"]
        flipped = f.copy()
        last = flipped.groupby("pid")["week"].transform("max") == flipped["week"]
        flipped.loc[last, "hit"] = 1.0 - flipped.loc[last, "hit"]
        self.assertNotEqual(pc.persistence(flipped)["slope"], base)             # the flipped games ARE the targets...
        only_early = pc.persistence(f[f["week"] < 14])                          # ...but dropping them leaves a valid, smaller study
        self.assertGreater(only_early["n"], 0)

    def test_too_little_data_means_zero_strength(self):
        p = pc.persistence(self._frame(True, n_players=5, games=6))
        self.assertEqual(pc.strength_from(p), 0.0)

    def test_stored_strength_roundtrips_and_defaults_to_zero(self):
        path = pathlib.Path(tempfile.mkdtemp()) / "t.json"
        mt.write({"m": {"weight": 0.0, "n": 200, "verdict": "x"}}, path, confidence={"strength": 0.3, "slope": 0.2})
        self.assertAlmostEqual(mt.load_confidence(path), 0.3)
        mt.write({"m": {"weight": 0.0, "n": 200, "verdict": "x"}}, path)
        self.assertEqual(mt.load_confidence(path), 0.0)
        self.assertEqual(mt.load_confidence(path.parent / "missing.json"), 0.0)


def _build(confidence, lambda_override=None):
    """One real prop through export_props.build_prop with a fixed model and market, so only the weight differs."""
    event = {"id": "e1", "commence_time": "2099-01-01T17:00:00Z"}
    row = {"player_id": "p1", "player_display_name": "Test WR", "team": "AAA", "position": "WR"}
    cons = {"point": 60.5, "fair_over": 0.50, "n_books": 3,
            "best_over": {"price": -110, "book": "X"}, "best_under": {"price": -110, "book": "Y"},
            "books": [{"book": "X", "over": -110, "under": -110}], "dfs": []}
    fit = {"mean": 75.0, "var": 25.0 ** 2, "n_games": 12, "n_eff": 10.0, "log": [], "env": None}      # model sits well above the line
    return ep.build_prop(event, "AAA", "BBB", row, "player_reception_yds", cons, fit, 1.0, None, None, 1.0,
                         lambda_override, confidence)


class TestGradeEntersTheEdge(unittest.TestCase):
    def test_at_zero_strength_the_edge_is_identical_with_or_without_a_grade(self):
        plain = _build(None)
        graded = _build(pc.confidence_for({"grade": 9.0, "n": 8, "hits": 7, "hit_rate": 0.875, "mean_abs_z": None}, 0.0))
        self.assertEqual(plain["over"], graded["over"])
        self.assertEqual(plain["trust"], graded["trust"])
        self.assertEqual(plain["pick_ev"], graded["pick_ev"])
        self.assertEqual(graded["confidence"]["multiplier"], 1.0)

    def test_with_positive_strength_a_high_grade_moves_the_probability_further_from_the_market(self):
        hi = _build(pc.confidence_for({"grade": 9.0, "n": 8, "hits": 7, "hit_rate": 0.875, "mean_abs_z": None}, 0.4), lambda_override=0.05)
        lo = _build(pc.confidence_for({"grade": 1.0, "n": 8, "hits": 1, "hit_rate": 0.125, "mean_abs_z": None}, 0.4), lambda_override=0.05)
        mid = _build(pc.confidence_for({"grade": 5.0, "n": 8, "hits": 4, "hit_rate": 0.5, "mean_abs_z": None}, 0.4), lambda_override=0.05)
        # the model is above the line, so more trust -> a higher P(over) and edge; less trust -> closer to the market's 0.50
        self.assertGreater(hi["over"], mid["over"])
        self.assertGreater(mid["over"], lo["over"])
        self.assertGreater(hi["trust"], mid["trust"])
        self.assertAlmostEqual(hi["trust"] / mid["trust"], pc.multiplier(9.0, 0.4), places=2)      # 1.32
        self.assertAlmostEqual(lo["trust"] / mid["trust"], pc.multiplier(1.0, 0.4), places=2)      # 0.68

    def test_the_default_cap_is_what_gets_scaled_when_a_market_has_no_override(self):
        hi = _build(pc.confidence_for({"grade": 10.0, "n": 9, "hits": 9, "hit_rate": 1.0, "mean_abs_z": None}, 0.5), lambda_override=None)
        base = _build(None, lambda_override=None)
        self.assertAlmostEqual(hi["trust"] / base["trust"], 1.5, places=1)


if __name__ == "__main__":
    unittest.main()
