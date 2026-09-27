"""
Tests for the pure math in props_model.py and the parsing in odds_api.py.

Run:  python -m unittest discover -s tests -v
"""

import unittest

import numpy as np
import pandas as pd

import odds_api
import game_context as gc
from props_model import (
    LAMBDA_MAX, environment_effect, load_calibration, MARKETS, blend_toward_market, distribution_summary, draw, fit_moments, fit_player,
    line_probs, market_weight, normalize_name, opponent_multipliers, position_priors, recency_weights, rng_for,
)


class TestDistributions(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_count_draws_match_mean_and_are_integers(self):
        d = draw("count", 4.0, 6.0, self.rng, 50000)      # over-dispersed -> negative binomial
        self.assertAlmostEqual(d.mean(), 4.0, delta=0.1)
        self.assertAlmostEqual(d.var(), 6.0, delta=0.4)
        self.assertTrue(np.all(d == d.astype(int)))

    def test_underdispersed_counts_fall_back_to_poisson(self):
        d = draw("count", 2.0, 1.0, self.rng, 50000)
        self.assertAlmostEqual(d.var(), 2.0, delta=0.15)

    def test_yardage_draws_match_moments_and_are_right_skewed(self):
        d = draw("yards", 70.0, 30.0 ** 2, self.rng, 50000)
        self.assertAlmostEqual(d.mean(), 70.0, delta=1.0)
        self.assertAlmostEqual(d.std(), 30.0, delta=1.0)
        self.assertGreater(d.mean(), np.median(d))          # skew: median sits below the mean
        self.assertTrue(np.all(d >= 0))

    def test_zero_mean_is_all_zeros(self):
        self.assertTrue(np.all(draw("yards", 0.0, 0.0, self.rng, 100) == 0))

    def test_rng_is_deterministic_per_prop_and_distinct_across_props(self):
        a1 = rng_for("p1", "player_rush_yds").random()
        a2 = rng_for("p1", "player_rush_yds").random()
        b = rng_for("p2", "player_rush_yds").random()
        self.assertEqual(a1, a2)
        self.assertNotEqual(a1, b)


class TestMarketBlend(unittest.TestCase):
    def test_zero_history_gets_zero_weight_and_full_history_is_capped(self):
        self.assertEqual(market_weight(0), 0.0)
        self.assertEqual(market_weight(50), LAMBDA_MAX)
        self.assertLess(market_weight(3), market_weight(6))

    def test_blend_moves_toward_the_market_but_keeps_direction(self):
        adj = blend_toward_market(0.70, 0.50, 20)
        self.assertGreater(adj, 0.50)                          # still leans the model's way...
        self.assertLess(adj, 0.55)                             # ...but only a little of the 20-pt gap survives
        self.assertLess(blend_toward_market(0.30, 0.50, 20), 0.50)

    def test_agreement_is_unchanged_and_thin_samples_defer_to_the_market(self):
        self.assertAlmostEqual(blend_toward_market(0.5, 0.5, 20), 0.5)
        self.assertLess(abs(blend_toward_market(0.8, 0.5, 2) - 0.5), abs(blend_toward_market(0.8, 0.5, 20) - 0.5))


class TestLineProbs(unittest.TestCase):
    def test_half_point_line_never_pushes(self):
        p = line_probs(np.array([1, 2, 3, 4, 5.0]), 2.5)
        self.assertEqual(p["push"], 0.0)
        self.assertAlmostEqual(p["over"] + p["under"], 1.0)
        self.assertAlmostEqual(p["over"], 0.6)

    def test_whole_number_line_pushes(self):
        p = line_probs(np.array([1, 2, 3, 3, 3.0]), 3)
        self.assertAlmostEqual(p["push"], 0.6)
        self.assertAlmostEqual(p["over"], 0.0)
        self.assertAlmostEqual(p["over"] + p["push"] + p["under"], 1.0)

    def test_distribution_summary_reproduces_line_probs(self):
        rng = np.random.default_rng(1)
        counts = draw("count", 5.0, 8.0, rng, 40000)
        pmf = distribution_summary("count", counts)["pmf"]
        self.assertAlmostEqual(sum(pmf[6:]), line_probs(counts, 5.5)["over"], places=2)
        self.assertAlmostEqual(sum(pmf), 1.0, places=2)
        q = distribution_summary("yards", draw("yards", 60, 900, rng, 40000))["q"]
        self.assertEqual(len(q), 101)
        self.assertEqual(q, sorted(q))


class TestFitting(unittest.TestCase):
    def test_recency_weights_favor_recent_games(self):
        w = recency_weights(6)
        self.assertEqual(w[-1], 1.0)
        self.assertTrue(np.all(np.diff(w) > 0))

    def test_predictive_variance_shrinks_with_more_games(self):
        few = fit_moments([50, 70, 60], recency_weights(3))
        many_vals = [50, 70, 60] * 5
        many = fit_moments(many_vals, np.ones(len(many_vals)))
        # same spread of values, but 15 games pin the mean down better than 3
        self.assertGreater(few[1], many[1])

    def test_thin_history_returns_none(self):
        h = pd.DataFrame({"player_id": ["a", "a"], "season": [2026, 2026], "week": [1, 2],
                          "opponent_team": ["X", "Y"], "position": ["WR", "WR"],
                          "receiving_yards": [40.0, 60.0]})
        self.assertIsNone(fit_player(h, "a", MARKETS["player_reception_yds"]))

    def test_td_with_zero_history_is_not_zero_probability(self):
        rows = []
        for w in range(1, 9):                    # scoreless WR seeing 6 targets a game
            rows.append(dict(player_id="a", season=2026, week=w, opponent_team="X", position="WR",
                             rushing_tds=0.0, receiving_tds=0.0, targets=6.0, carries=0.0))
        h = pd.DataFrame(rows)
        fit = fit_player(h, "a", MARKETS["player_anytime_td"], 1.0, {"WR": 0.05})   # 0.05 TD per touch
        self.assertGreater(fit["mean"], 0.05)    # regressed toward 6 touches x 0.05 = 0.3/game

    def test_td_prior_follows_usage_not_the_position_average(self):
        def player(pid, touches):
            return [dict(player_id=pid, season=2026, week=w, opponent_team="X", position="WR",
                         rushing_tds=0.0, receiving_tds=0.0, targets=touches, carries=0.0) for w in range(1, 9)]
        h = pd.DataFrame(player("star", 10.0) + player("scrub", 1.0))
        rate = {"WR": 0.05}
        star = fit_player(h, "star", MARKETS["player_anytime_td"], 1.0, rate)["mean"]
        scrub = fit_player(h, "scrub", MARKETS["player_anytime_td"], 1.0, rate)["mean"]
        self.assertGreater(star, 4 * scrub)      # a low-usage player is not lifted to a starter's rate

    def test_position_prior_for_tds_is_per_touch(self):
        rows = [dict(player_id="a", season=2026, week=w, opponent_team="X", position="WR", rushing_tds=0.0,
                     receiving_tds=1.0 if w % 2 else 0.0, targets=10.0, carries=0.0, attempts=0.0) for w in range(1, 11)]
        rate = position_priors(pd.DataFrame(rows), MARKETS["player_anytime_td"])
        self.assertAlmostEqual(rate["WR"], 5 / 100)          # 5 TDs on 100 touches

    def test_low_volume_count_is_not_zero_probability(self):
        rows = [dict(player_id="a", season=2026, week=w, opponent_team="X", position="RB", receptions=0.0)
                for w in range(1, 6)]                    # back with no catches in 5 games
        fit = fit_player(pd.DataFrame(rows), "a", MARKETS["player_receptions"], 1.0, {"RB": 2.0})
        self.assertGreater(fit["mean"], 0.2)
        self.assertGreater(line_probs(draw("count", fit["mean"], fit["var"], np.random.default_rng(0), 20000), 0.5)["over"], 0.1)

    def test_high_volume_count_is_not_dragged_toward_the_average(self):
        rows = [dict(player_id="a", season=2026, week=w, opponent_team="X", position="WR", receptions=8.0)
                for w in range(1, 11)]
        fit = fit_player(pd.DataFrame(rows), "a", MARKETS["player_receptions"], 1.0, {"WR": 4.0})
        self.assertAlmostEqual(fit["mean"], 8.0, places=6)

    def test_player_with_no_role_is_not_projected(self):
        rows = [dict(player_id="a", season=2026, week=w, opponent_team="X", position="WR", receiving_yards=0.0)
                for w in range(1, 8)]
        self.assertIsNone(fit_player(pd.DataFrame(rows), "a", MARKETS["player_reception_yds"]))

    def test_opponent_multiplier_direction_and_bounds(self):
        rows = []
        for g, (opp, yards) in enumerate([("SOFT", 120)] * 8 + [("STIFF", 40)] * 8 + [("AVG", 80)] * 8):
            rows.append(dict(opponent_team=opp, game_id=f"g{g}", position="WR", receiving_yards=float(yards)))
        m = opponent_multipliers(pd.DataFrame(rows), MARKETS["player_reception_yds"])
        self.assertGreater(m[("SOFT", "WR")], 1.0)
        self.assertLess(m[("STIFF", "WR")], 1.0)
        self.assertTrue(all(0.8 <= v <= 1.25 for v in m.values()))

    def test_normalize_name_bridges_sportsbook_spellings(self):
        self.assertEqual(normalize_name("Kenneth Walker III"), normalize_name("Kenneth Walker"))
        self.assertEqual(normalize_name("Ja'Marr Chase"), normalize_name("JaMarr Chase"))
        self.assertEqual(normalize_name("A.J. Brown"), normalize_name("AJ Brown"))


def _event(books):
    return {"bookmakers": [{"title": b, "markets": [{"key": "player_rush_yds", "outcomes": [
        {"name": "Over", "description": "Test Back", "price": o, "point": pt},
        {"name": "Under", "description": "Test Back", "price": u, "point": pt},
    ]}]} for b, o, u, pt in books]}


class TestOddsParsing(unittest.TestCase):
    def test_main_line_is_the_one_most_books_offer(self):
        ev = _event([("A", -110, -110, 64.5), ("B", -110, -110, 64.5), ("C", -105, -115, 66.5)])
        c = odds_api.consensus(odds_api.flatten(ev))[("Test Back", "player_rush_yds")]
        self.assertEqual(c["point"], 64.5)
        self.assertEqual(c["n_books"], 2)

    def test_fair_probability_removes_the_vig(self):
        ev = _event([("A", -110, -110, 64.5)])
        c = odds_api.consensus(odds_api.flatten(ev))[("Test Back", "player_rush_yds")]
        self.assertAlmostEqual(c["fair_over"], 0.5, places=4)     # 52.4% each side before vig removal

    def test_best_price_picks_the_highest_payout_per_side(self):
        ev = _event([("A", -110, -110, 64.5), ("B", -105, -125, 64.5), ("C", -120, +100, 64.5)])
        c = odds_api.consensus(odds_api.flatten(ev))[("Test Back", "player_rush_yds")]
        self.assertEqual(c["best_over"], {"price": -105, "book": "B"})
        self.assertEqual(c["best_under"], {"price": 100, "book": "C"})

    def test_anytime_td_has_no_fair_prob_and_uses_half_point(self):
        ev = {"bookmakers": [{"title": "A", "markets": [{"key": "player_anytime_td", "outcomes": [
            {"name": "Yes", "description": "Test Back", "price": 150}]}]}]}
        c = odds_api.consensus(odds_api.flatten(ev))[("Test Back", "player_anytime_td")]
        self.assertIsNone(c["fair_over"])
        self.assertEqual(c["point"], 0.5)
        self.assertEqual(c["best_over"]["price"], 150)

    def test_team_map_covers_all_32_teams_uniquely(self):
        self.assertEqual(len(odds_api.TEAM_ABBR), 32)
        self.assertEqual(len(set(odds_api.TEAM_ABBR.values())), 32)

    def test_missing_key_gives_actionable_error(self):
        import os
        old = os.environ.pop("ODDS_API_KEY", None)
        try:
            # only meaningful when no .env is present; otherwise the key is legitimately found
            import pathlib
            if not (pathlib.Path(odds_api.__file__).parent / ".env").exists():
                with self.assertRaises(odds_api.OddsApiError) as cm:
                    odds_api.load_key()
                self.assertIn("ODDS_API_KEY", str(cm.exception))
        finally:
            if old is not None:
                os.environ["ODDS_API_KEY"] = old


class TestPickemBooks(unittest.TestCase):
    def _event(self):
        def book(key, title, o, u, pt):
            return {"key": key, "title": title, "markets": [{"key": "player_rush_yds", "outcomes": [
                {"name": "Over", "description": "Test Back", "price": o, "point": pt},
                {"name": "Under", "description": "Test Back", "price": u, "point": pt}]}]}
        return {"bookmakers": [
            book("draftkings", "DraftKings", -110, -110, 64.5),
            book("fanduel", "FanDuel", -115, -105, 64.5),
            book("underdog", "Underdog", -137, -105, 61.5),       # fixed-payout entry "prices", different line
            book("prizepicks", "PrizePicks", -119, -119, 62.5),
        ]}

    def test_classifier(self):
        self.assertTrue(odds_api.is_dfs("underdog", "Underdog"))
        self.assertTrue(odds_api.is_dfs("prizepicks", "PrizePicks"))
        self.assertTrue(odds_api.is_dfs("draftkings_pick6", "DraftKings Pick6"))
        self.assertFalse(odds_api.is_dfs("draftkings", "DraftKings"))
        self.assertFalse(odds_api.is_dfs("fanduel", "FanDuel"))

    def test_pickem_lines_never_touch_the_sportsbook_consensus(self):
        c = odds_api.consensus(odds_api.flatten(self._event()))[("Test Back", "player_rush_yds")]
        self.assertEqual(c["point"], 64.5)                          # not dragged to 61.5/62.5
        self.assertEqual(c["n_books"], 2)
        self.assertEqual({b["book"] for b in c["books"]}, {"DraftKings", "FanDuel"})
        self.assertAlmostEqual(c["fair_over"], 0.5, delta=0.02)     # only the two real books
        self.assertEqual(c["dfs"], [{"book": "PrizePicks", "line": 62.5}, {"book": "Underdog", "line": 61.5}])

    def test_pickem_only_props_are_not_priced(self):
        ev = {"bookmakers": [self._event()["bookmakers"][2]]}
        self.assertEqual(odds_api.consensus(odds_api.flatten(ev)), {})

    def test_line_without_a_point_is_ignored_not_crashed(self):
        ev = self._event()
        ev["bookmakers"][2]["markets"][0]["outcomes"] = [{"name": "Higher", "description": "Test Back", "price": -110}]
        c = odds_api.consensus(odds_api.flatten(ev))[("Test Back", "player_rush_yds")]
        self.assertEqual([d["book"] for d in c["dfs"]], ["PrizePicks"])


class TestCalibrationAndEnvironment(unittest.TestCase):
    def _games(self, n=10, yards=60.0, total=44.0, margin=0.0):
        return pd.DataFrame([dict(player_id="a", season=2026, week=w, opponent_team="X", position="WR",
                                  receiving_yards=yards, total=total, margin=margin) for w in range(1, n + 1)])

    def test_calibration_pulls_a_hot_projection_back_and_keeps_the_spread_proportional(self):
        h = self._games()
        raw = fit_player(h, "a", MARKETS["player_reception_yds"])
        cal = fit_player(h, "a", MARKETS["player_reception_yds"], calib={"a": 2.0, "b": 0.85})
        self.assertAlmostEqual(cal["mean"], 2.0 + 0.85 * raw["mean"])
        self.assertAlmostEqual(np.sqrt(cal["var"]) / cal["mean"], np.sqrt(raw["var"]) / raw["mean"], places=6)

    def test_environment_bumps_a_shootout_and_docks_a_slog(self):
        h = self._games(total=44.0)
        eff = {"a": 0.5, "b": 0.0}
        base = fit_player(h, "a", MARKETS["player_reception_yds"])["mean"]
        hi = fit_player(h, "a", MARKETS["player_reception_yds"], env_fn=lambda g, w: gc.env_multiplier(g, w, 51.0, 0.0, eff))
        lo = fit_player(h, "a", MARKETS["player_reception_yds"], env_fn=lambda g, w: gc.env_multiplier(g, w, 38.0, 0.0, eff))
        self.assertGreater(hi["mean"], base * 1.03)          # a 51 total vs his usual 44
        self.assertLess(lo["mean"], base * 0.97)
        self.assertGreater(hi["env"]["mult"], 1.0)

    def test_environment_is_neutral_when_the_game_looks_like_his_usual_games(self):
        h = self._games(total=44.0, margin=3.0)
        m = gc.env_multiplier(h, np.ones(len(h)), 44.0, 3.0, {"a": 0.5, "b": 0.02})
        self.assertAlmostEqual(m["mult"], 1.0, places=6)

    def test_environment_is_capped_and_missing_lines_mean_no_adjustment(self):
        h = self._games(total=44.0)
        self.assertLessEqual(gc.env_multiplier(h, np.ones(len(h)), 200.0, 0.0, {"a": 5.0, "b": 0.0})["mult"], gc.MULT_CLIP[1])
        self.assertEqual(gc.env_multiplier(h, np.ones(len(h)), None, None, {"a": 0.5, "b": 0.0})["mult"], 1.0)
        self.assertEqual(gc.env_multiplier(h, np.ones(len(h)), 50.0, 0.0, None)["mult"], 1.0)

    def test_td_props_ignore_calibration_and_environment(self):
        rows = [dict(player_id="a", season=2026, week=w, opponent_team="X", position="WR", rushing_tds=0.0,
                     receiving_tds=1.0 if w % 3 == 0 else 0.0, targets=6.0, carries=0.0, total=44.0, margin=0.0)
                for w in range(1, 10)]
        h = pd.DataFrame(rows)
        base = fit_player(h, "a", MARKETS["player_anytime_td"])["mean"]
        same = fit_player(h, "a", MARKETS["player_anytime_td"], calib={"a": 9, "b": 0}, env_fn=lambda g, w: {"mult": 2.0})
        self.assertAlmostEqual(base, same["mean"])

    def test_environment_effect_is_dampened_by_out_of_sample_strength(self):
        cal = {"environment": {"m": {"a": 0.8, "b": 0.02, "damp": 0.5}}}
        self.assertEqual(environment_effect(cal, "m"), {"a": 0.4, "b": 0.01})
        self.assertIsNone(environment_effect(cal, "other"))
        self.assertIsNone(environment_effect(None, "m"))

    def test_missing_calibration_file_is_not_fatal(self):
        self.assertIsNone(load_calibration("does-not-exist.json"))


class TestGameLines(unittest.TestCase):
    def test_consensus_uses_medians_and_the_home_teams_spread(self):
        games = [{"id": "g1", "home_team": "Detroit Lions", "away_team": "New York Jets", "bookmakers": [
            {"markets": [{"key": "totals", "outcomes": [{"name": "Over", "point": 48.5}, {"name": "Under", "point": 48.5}]},
                         {"key": "spreads", "outcomes": [{"name": "Detroit Lions", "point": -7.5}, {"name": "New York Jets", "point": 7.5}]}]},
            {"markets": [{"key": "totals", "outcomes": [{"name": "Over", "point": 49.5}]},
                         {"key": "spreads", "outcomes": [{"name": "Detroit Lions", "point": -6.5}]}]},
            {"markets": [{"key": "totals", "outcomes": [{"name": "Over", "point": 49.0}]},
                         {"key": "spreads", "outcomes": [{"name": "Detroit Lions", "point": -7.0}]}]},
        ]}]
        out = odds_api.consensus_game_lines(games)["g1"]
        self.assertEqual(out["total"], 49.0)
        self.assertEqual(out["home_spread"], -7.0)

    def test_game_without_lines_yields_none_not_a_crash(self):
        out = odds_api.consensus_game_lines([{"id": "g", "home_team": "A", "away_team": "B", "bookmakers": []}])
        self.assertEqual(out["g"], {"total": None, "home_spread": None})

    def test_implied_team_totals(self):
        team, opp = gc.implied_team_totals(50.0, 3.0)         # favored by 3 in a 50 total
        self.assertEqual((team, opp), (26.5, 23.5))


if __name__ == "__main__":
    unittest.main()
