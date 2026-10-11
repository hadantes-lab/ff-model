"""
Tests for market_trust.py (measuring how much to trust the model per market), export_props.resolve_trust, and
volume_model.py (team-level game script -> volume).

Run:  python -m unittest discover -s tests -v
"""

import json
import pathlib
import tempfile
import unittest

import numpy as np
import pandas as pd

import export_props as ep
import market_trust as mt
import volume_model as vm


def _frame(n, model_skill, seed=0, market="player_x"):
    """n props: the truth is a latent probability; the market sees it with noise; the model sees it with `model_skill` noise."""
    rng = np.random.default_rng(seed)
    true = rng.uniform(0.3, 0.7, n)
    over = (rng.random(n) < true).astype(float)
    p_mkt = np.clip(true + rng.normal(0, 0.03, n), 0.05, 0.95)
    p_model = np.clip(true + rng.normal(0, model_skill, n), 0.02, 0.98)
    return pd.DataFrame({"market": market, "p_mkt": p_mkt, "p_m_env": p_model, "over": over})


class TestChooseWeight(unittest.TestCase):
    def test_brier_change_sign(self):
        d = pd.DataFrame({"p_mkt": [0.5, 0.5], "p_m": [0.9, 0.9], "over": [1.0, 1.0]})
        self.assertTrue((mt.brier_change(d, "p_m", 0.5) < 0).all())              # moving toward what happened helps
        d["over"] = 0.0
        self.assertTrue((mt.brier_change(d, "p_m", 0.5) > 0).all())              # moving away hurts
        self.assertTrue((mt.brier_change(d, "p_m", 0.0) == 0).all())             # weight 0 is the market itself

    def test_a_model_no_better_than_noise_gets_zero(self):
        d = _frame(1500, model_skill=0.25, seed=1)                                 # model is mostly noise around the truth
        r = mt.choose_weight(d, "p_m_env")
        self.assertEqual(r["weight"], 0.0)

    def test_a_clearly_better_model_earns_a_real_weight_but_is_capped(self):
        rng = np.random.default_rng(2)
        n = 6000
        true = rng.uniform(0.25, 0.75, n)
        over = (rng.random(n) < true).astype(float)
        d = pd.DataFrame({"market": "m", "over": over,
                          "p_mkt": np.clip(true + rng.normal(0, 0.12, n), 0.05, 0.95),          # a sloppy market
                          "p_m_env": np.clip(true + rng.normal(0, 0.02, n), 0.02, 0.98)})       # a sharp model
        r = mt.choose_weight(d, "p_m_env")
        self.assertGreater(r["weight"], mt.FLOOR)
        self.assertLessEqual(r["weight"], mt.WEIGHT_CAP)

    def test_too_few_lines_means_no_weight(self):
        self.assertIsNone(mt.choose_weight(_frame(40, 0.05), "p_m_env")["weight"])

    def test_write_and_load_roundtrip(self):
        path = pathlib.Path(tempfile.mkdtemp()) / "t.json"
        res = {"player_a": {"weight": 0.0, "n": 500, "verdict": "none"}, "player_b": {"weight": 0.05, "n": 300, "verdict": "some"},
               "player_c": {"weight": None, "n": 20, "verdict": "too few lines"}}
        mt.write(res, path, early={"player_a": {"qualifies": True}, "player_b": {"qualifies": False}})
        self.assertEqual(mt.load_weights(path), {"player_a": 0.0, "player_b": 0.05})      # a market with no verdict is left out
        self.assertEqual(list(mt.load_early(path)), ["player_a"])
        self.assertEqual(mt.load_weights(path.parent / "missing.json"), {})                 # no file: fall back to the default cap


class TestEarlyTable(unittest.TestCase):
    def _x(self, n, edge_wins, seed=3, market="player_m"):
        """Opening prices -110/-110. If `edge_wins`, the side the model points to really does win ~60% of the time."""
        rng = np.random.default_rng(seed)
        line = np.full(n, 50.0)
        gap = rng.normal(0, 0.2, n)                                   # model's relative gap vs the line
        model = line * (1 + gap)
        p_win = 0.6 if edge_wins else 0.5
        pointed_over = gap > 0
        win = rng.random(n) < p_win
        goes_over = np.where(pointed_over, win, ~win)
        actual = np.where(goes_over, 60.0, 40.0)
        return pd.DataFrame({"market": market, "open_line": line, "m_env": model, "actual": actual,
                             "open_over": -110.0, "open_under": -110.0})

    def test_a_real_early_edge_qualifies(self):
        # strong picks win 60% at -110 (ROI ~ +14%); with 2000 props / ~500 strong picks the interval clears zero
        r = mt.early_table(self._x(2000, True), n_tests=1)["player_m"]
        self.assertTrue(r["qualifies"])
        self.assertGreater(r["roi"], 0.05)

    def test_a_coin_flip_does_not(self):
        r = mt.early_table(self._x(2000, False), n_tests=1)["player_m"]
        self.assertFalse(r["qualifies"])

    def test_the_multiple_testing_correction_makes_it_harder(self):
        weak = self._x(1200, True, seed=5)
        weak.loc[weak.index[::2], "actual"] = np.where(weak.loc[weak.index[::2], "m_env"] > 50, 40.0, 60.0)   # dilute to a mild edge
        one = mt.early_table(weak, n_tests=1)["player_m"]
        many = mt.early_table(weak, n_tests=40)["player_m"]
        self.assertLessEqual(many["roi_ci"][0], one["roi_ci"][0])                     # a wider interval, lower bound only drops
        self.assertLessEqual(int(many["qualifies"]), int(one["qualifies"]))

    def test_roi_uses_the_real_prices_not_a_flat_minus_110(self):
        x = self._x(2000, True)
        x["open_over"] = -200.0                                                       # the side is priced far worse
        x["open_under"] = -200.0
        r = mt.early_table(x, n_tests=1)["player_m"]
        self.assertAlmostEqual(r["hit_rate"], 0.6, delta=0.05)
        self.assertLess(r["roi"], 0.0)                                                # 60% at -200 loses money (needs 66.7%)
        self.assertFalse(r["qualifies"])

    def test_pushes_and_missing_prices_are_excluded(self):
        x = self._x(1000, True)
        x.loc[x.index[:100], "open_over"] = np.nan
        x.loc[x.index[100:200], "actual"] = 50.0                                      # lands exactly on the line
        self.assertLessEqual(mt.early_table(x, n_tests=1)["player_m"]["n"], 800)


class TestResolveTrust(unittest.TestCase):
    def test_live_refinement_wins_then_static_then_none(self):
        live = {"player_a": {"suggested_weight": 0.2}}
        static = {"player_a": 0.0, "player_b": 0.05}
        self.assertEqual(ep.resolve_trust("player_a", live, static), 0.2)             # tracked results override
        self.assertEqual(ep.resolve_trust("player_b", live, static), 0.05)
        self.assertIsNone(ep.resolve_trust("player_c", live, static))                 # unknown market: default cap applies

    def test_a_live_weight_of_exactly_zero_is_respected(self):
        self.assertEqual(ep.resolve_trust("player_a", {"player_a": {"suggested_weight": 0.0}}, {"player_a": 0.05}), 0.0)

    def test_zero_weight_means_the_market_price_comes_back_unchanged(self):
        from props_model import blend_toward_market
        self.assertEqual(blend_toward_market(0.9, 0.5, n_eff=20, lambda_max=0.0), 0.5)


def _team_games(seasons=(2025, 2026), weeks=8, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in seasons:
        for w in range(1, weeks + 1):
            for team in ("AAA", "BBB", "CCC", "DDD"):
                opp = {"AAA": "BBB", "BBB": "AAA", "CCC": "DDD", "DDD": "CCC"}[team]
                margin = float(rng.normal(0, 7)) * (1 if team in ("AAA", "CCC") else -1)
                total = float(rng.normal(45, 4))
                # a favorite runs more and throws less, by construction
                rows.append({"season": s, "week": w, "team": team, "opp": opp, "is_home": int(team in ("AAA", "CCC")),
                             "margin": margin, "total_line": total, "total": total, "game_id": f"{s}{w}{team}",
                             "attempts": 35 - 0.3 * margin + rng.normal(0, 3), "carries": 26 + 0.3 * margin + rng.normal(0, 3),
                             "att_allowed": 34 + rng.normal(0, 2), "car_allowed": 26 + rng.normal(0, 2)})
    return pd.DataFrame(rows)


class TestVolumeModel(unittest.TestCase):
    def setUp(self):
        self.g = vm.add_opponent_features(vm.add_rolling(_team_games()))

    def test_rolling_features_never_include_the_game_itself(self):
        g = vm.add_rolling(_team_games())
        t = g[g["team"] == "AAA"].reset_index(drop=True)
        spike = t.copy()
        spike.loc[10, "attempts"] = 500.0                                              # a huge value in game 10
        again = vm.add_rolling(pd.concat([spike, g[g["team"] != "AAA"]], ignore_index=True))
        a = again[again["team"] == "AAA"].reset_index(drop=True)
        self.assertEqual(a.loc[10, "r_att"], t.loc[10, "r_att"])                        # game 10's own features unchanged
        self.assertGreater(a.loc[11, "r_att"], t.loc[11, "r_att"])                      # but the NEXT game sees it

    def test_the_fitted_script_effect_has_the_right_signs(self):
        att = vm.fit_final(self.g, "att")["coef"]["d_margin"]
        car = vm.fit_final(self.g, "car")["coef"]["d_margin"]
        self.assertLess(att, 0)                                                         # more expected margin: fewer attempts
        self.assertGreater(car, 0)                                                      # ...and more carries

    def test_predict_ratio_moves_the_right_way(self):
        fit = vm.fit_final(self.g, "car")
        row = self.g.dropna(subset=["r_car", "opp_ratio_car", "d_margin", "d_ln_total"]).iloc[[0]].copy()
        lo, hi = row.copy(), row.copy()
        lo["d_margin"], lo["d_margin_sq"] = -10.0, -1.0
        hi["d_margin"], hi["d_margin_sq"] = 10.0, 1.0
        c, icpt = np.array([fit["coef"][f] for f in fit["features"]]), fit["intercept"]
        self.assertGreater(vm.predict_ratio(hi, "car", c, icpt)[0], vm.predict_ratio(lo, "car", c, icpt)[0])

    def test_walk_forward_only_trains_on_earlier_seasons(self):
        w = vm.walk_forward(self.g, "att", first_test=2026)
        self.assertEqual(set(w["season"]), {2026})
        self.assertTrue((w["model"] > 0).all())


if __name__ == "__main__":
    unittest.main()
