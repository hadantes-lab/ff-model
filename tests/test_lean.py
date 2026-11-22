"""
Tests for lean.py: the lean (raw model side at the play line), the calibrated confidence, hot/cold records and the matchup check, plus
pickem_study's no-look-ahead replay.

Run:  python -m unittest discover -s tests -v
"""

import unittest

import numpy as np
import pandas as pd

import lean
import pickem_study as ps

CAL = {"b0": -0.18, "b_s": 0.19, "b_agree": 0.28, "n": 1000, "base_rate": 0.527, "source": "fitted"}


def _log(n=900, seed=1, weeks=8):
    """A synthetic graded log where a stronger raw lean and market agreement really do hit more often."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        mo = float(np.clip(rng.normal(0.5, 0.15), 0.05, 0.95))
        mk = float(np.clip(mo + rng.normal(0, 0.12), 0.05, 0.95))
        side = "over" if mo > 0.5 else "under"
        agree = side == ("over" if mk > 0.5 else "under")
        p_hit = 0.5 + 0.25 * abs(mo - 0.5) + (0.04 if agree else -0.02)
        hit = rng.random() < p_hit
        res = side if hit else ("under" if side == "over" else "over")
        rows.append({"season": 2026, "week": 3 + i % weeks, "player_id": f"p{i % 60}", "market": "player_receptions" if i % 2 else "player_rush_yds",
                     "player": f"P{i % 60}", "pos": "WR", "kind": "count", "model_over": mo, "market_over": mk, "result": res, "line": 4.5})
    return pd.DataFrame(rows)


def _prop(**kw):
    p = {"player_id": "a", "player": "A", "pos": "WR", "team": "HOM", "opp": "AWY", "market": "player_receptions", "label": "Receptions", "kind": "count",
         "line": 4.5, "model_over": 0.66, "market_over": 0.54, "over": 0.56, "under": 0.44, "opp_mult": 1.0,
         "pickem": {"line": 4.5, "book": None, "side": "over", "prob": 0.56, "agree": True}, "dfs": []}
    p.update(kw)
    return p


class TestCalibration(unittest.TestCase):
    def test_examples_skip_games_tds_ungraded_and_no_market(self):
        d = _log(50)
        extra = d.iloc[:4].copy()
        extra["kind"] = ["td", "count", "count", "count"]
        extra["pos"] = ["WR", "GAME", "WR", "WR"]
        extra["result"] = ["over", "over", "push", "over"]
        extra["market_over"] = [0.5, 0.5, 0.5, np.nan]
        ex = lean.examples(pd.concat([d, extra]))
        self.assertEqual(len(ex), len(lean.examples(d)))

    def test_fit_recovers_that_strength_and_agreement_help(self):
        cal = lean.fit_calibration(_log(4000), min_rows=100)
        self.assertEqual(cal["source"], "fitted")
        self.assertGreater(cal["b_s"], 0)
        self.assertGreater(cal["b_agree"], 0)

    def test_too_little_data_falls_back_to_the_default_curve(self):
        self.assertEqual(lean.fit_calibration(_log(100))["source"], "default")
        self.assertEqual(lean.fit_calibration(pd.DataFrame())["source"], "default")

    def test_confidence_rises_with_strength_and_agreement_and_is_clipped(self):
        self.assertGreater(lean.confidence(CAL, 0.3, True), lean.confidence(CAL, 0.05, True))
        self.assertGreater(lean.confidence(CAL, 0.15, True), lean.confidence(CAL, 0.15, False))
        self.assertLessEqual(lean.confidence({"b0": 9, "b_s": 9, "b_agree": 9}, 0.5, True), lean.CONF_CLIP[1])
        self.assertGreaterEqual(lean.confidence({"b0": -9, "b_s": 0, "b_agree": 0}, 0.0, False), lean.CONF_CLIP[0])

    def test_grades(self):
        self.assertEqual((lean.grade(0.60), lean.grade(0.55), lean.grade(0.51)), ("high", "medium", "low"))

    def test_walk_forward_scores_each_week_on_earlier_weeks_only(self):
        wf = lean.walk_forward(_log(2400), min_rows=300)
        self.assertGreater(wf["n"], 0)
        self.assertLess(wf["n"], 2400)                                   # the first weeks only train
        self.assertLess(wf["brier"], wf["brier_flat"] + 1e-3)
        self.assertIsNone(lean.walk_forward(_log(100), min_rows=300))


class TestRecords(unittest.TestCase):
    def test_segment_records_are_shrunk_toward_half(self):
        d = pd.DataFrame([{"season": 2026, "week": 3, "market": "player_receptions", "player_id": f"p{i}", "pos": "WR", "kind": "count", "model_over": 0.3,
                           "market_over": 0.4, "result": "under"} for i in range(10)])
        seg = lean.segment_records(d, 2026)[("player_receptions", "under", True)]
        self.assertEqual((seg["n"], seg["hits"]), (10, 10))
        self.assertAlmostEqual(seg["rate"], (10 + 15) / 40, places=4)      # ten wins do not make a 100% type


class TestMatchup(unittest.TestCase):
    DVP = {"AWY": {"WR": {"player_receptions": {"rank": 28, "pg": 6.1, "lg": 5.0}}}, "TUF": {"WR": {"player_receptions": {"rank": 4, "pg": 3.0, "lg": 5.0}}},
           "MID": {"WR": {"player_receptions": {"rank": 15, "pg": 5.0, "lg": 5.0}}}}

    def test_soft_backs_an_over_tough_backs_an_under(self):
        p = _prop()
        self.assertEqual(lean.matchup(p, "over", self.DVP)["state"], "supports")
        self.assertEqual(lean.matchup(p, "under", self.DVP)["state"], "against")
        t = _prop(opp="TUF")
        self.assertEqual(lean.matchup(t, "under", self.DVP)["state"], "supports")
        self.assertEqual(lean.matchup(t, "over", self.DVP)["state"], "against")
        self.assertEqual(lean.matchup(_prop(opp="MID"), "over", self.DVP)["state"], "neutral")

    def test_the_note_names_the_rank_in_words(self):
        self.assertIn("5th most", lean.matchup(_prop(), "over", self.DVP)["note"])          # rank 28 of 32 = the 5th-softest
        self.assertIn("4th fewest", lean.matchup(_prop(opp="TUF"), "under", self.DVP)["note"])

    def test_falls_back_to_the_opponent_multiplier_then_to_nothing(self):
        self.assertEqual(lean.matchup(_prop(opp="NEW", opp_mult=1.06), "over", {})["state"], "supports")
        self.assertEqual(lean.matchup(_prop(opp="NEW", opp_mult=0.94), "over", {})["state"], "against")
        self.assertIsNone(lean.matchup(_prop(opp="NEW", opp_mult=1.01), "over", {})["state"])

    def test_ordinals(self):
        self.assertEqual([lean._ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 23)], ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "23rd"])


class TestLeanFor(unittest.TestCase):
    def test_lean_is_the_raw_side_and_agree_compares_it_with_the_markets(self):
        ln = lean.lean_for(_prop(), CAL, {}, {})
        self.assertEqual((ln["side"], ln["agree"], ln["line"]), ("over", True, 4.5))
        self.assertAlmostEqual(ln["raw_prob"], 0.66)
        against = lean.lean_for(_prop(pickem={"line": 4.5, "book": None, "side": "under", "prob": 0.52, "agree": False}, over=0.48, under=0.52), CAL, {}, {})
        self.assertEqual((against["side"], against["agree"]), ("over", False))
        self.assertLess(against["confidence"], ln["confidence"])

    def test_the_underdog_line_is_the_play_line(self):
        p = _prop(pickem={"line": 3.5, "book": "Underdog", "side": "over", "prob": 0.6, "agree": True},
                  dfs=[{"book": "Underdog", "line": 3.5, "over": 0.78, "under": 0.22}])
        ln = lean.lean_for(p, CAL, {}, {})
        self.assertEqual((ln["line"], ln["book"]), (3.5, "Underdog"))
        self.assertAlmostEqual(ln["raw_prob"], 0.78)

    def test_an_under_lean_reports_the_under_probability(self):
        ln = lean.lean_for(_prop(model_over=0.30, market_over=0.45, pickem={"line": 4.5, "book": None, "side": "under", "prob": 0.55, "agree": True}), CAL, {}, {})
        self.assertEqual(ln["side"], "under")
        self.assertAlmostEqual(ln["raw_prob"], 0.70)

    def test_tds_model_only_and_marketless_props_have_no_lean(self):
        self.assertIsNone(lean.lean_for(_prop(kind="td"), CAL, {}, {}))
        self.assertIsNone(lean.lean_for(_prop(model_only=True), CAL, {}, {}))
        self.assertIsNone(lean.lean_for(_prop(market_over=None), CAL, {}, {}))

    def test_hot_cold_and_player_streaks(self):
        segs = {("player_receptions", "over", True): {"n": 80, "hits": 52, "rate": 0.60}}
        ln = lean.lean_for(_prop(), CAL, segs, {})
        self.assertTrue(ln["hot"] and ln["seg_hot"] and not ln["cold"])
        self.assertIn("52 of 80", ln["hot_why"][0])
        cold = lean.lean_for(_prop(), CAL, {("player_receptions", "over", True): {"n": 60, "hits": 24, "rate": 0.42}}, {})
        self.assertTrue(cold["cold"])
        self.assertEqual(cold["grade"], "low")
        streak = lean.lean_for(_prop(), CAL, {}, {}, {"n": 6, "hits": 5})
        self.assertTrue(streak["player_hot"] and streak["hot"])
        self.assertFalse(lean.lean_for(_prop(), CAL, {}, {}, {"n": 3, "hits": 3})["hot"])             # three games are not a streak
        self.assertFalse(lean.lean_for(_prop(), CAL, {"x": 1}, {}, {"n": 6, "hits": 4})["hot"])

    def test_a_small_segment_never_counts_as_hot(self):
        segs = {("player_receptions", "over", True): {"n": 10, "hits": 9, "rate": 0.7}}
        self.assertFalse(lean.lean_for(_prop(), CAL, segs, {})["hot"])

    def test_annotate_sets_every_prop_and_reports_the_check(self):
        props = [_prop(player_id="a"), _prop(player_id="b", kind="td")]
        info = lean.annotate(props, _log(900), {}, {}, 2026)
        self.assertIsNotNone(props[0]["lean"])
        self.assertIsNone(props[1]["lean"])
        self.assertEqual(info["n"], 1)
        self.assertIn("calibration", info)
        import json
        json.dumps(info)


class TestReplay(unittest.TestCase):
    def test_the_replay_has_no_rows_without_enough_history(self):
        self.assertIsNone(ps.stack_backtest(_log(200), min_rows=2000))

    def test_the_replay_scores_stacks_and_plain_sets(self):
        d = _log(6000, weeks=8)
        d["game"] = [f"A{i % 5} @ H{i % 5}" for i in range(len(d))]
        d["team"] = [f"H{i % 5}" for i in range(len(d))]
        res = ps.stack_backtest(d, min_rows=1500)
        self.assertIsNotNone(res)
        kinds = {(r["kind"], r["k"]) for r in res["rows"]}
        self.assertTrue({("stack", 3), ("plain", 3)} <= kinds)
        for r in res["rows"]:
            self.assertLessEqual(r["hits"], r["entries"])
            self.assertAlmostEqual(r["rate"], r["hits"] / r["entries"], places=3)
        self.assertGreater(min(res["weeks"]), 3)                        # the first weeks only train


if __name__ == "__main__":
    unittest.main()
