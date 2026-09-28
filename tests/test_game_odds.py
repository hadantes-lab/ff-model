"""
Tests for game_odds.py (team scoring, correlated simulation, spread/total probabilities)
and odds_api.py's game-price consensus.

Run:  python -m unittest discover -s tests -v
"""

import unittest

import numpy as np
import pandas as pd

import game_odds as go
import odds_api


class TestTeamPoints(unittest.TestCase):
    def test_matches_real_nfl_scoring_distribution(self):
        # sanity anchor: league-average rush/pass yards and TDs should reproduce the real
        # league-average score (~22.5) and spread (~9.9), not just an arbitrary number
        rng = np.random.default_rng(0)
        n = 60000
        rush = rng.normal(120, 20, n)
        pas = rng.normal(230, 40, n)
        td = rng.poisson(2.5, n).astype(float)
        pts = go.team_points(rush, pas, td, rng)
        self.assertAlmostEqual(pts.mean(), 22.5, delta=1.5)
        self.assertAlmostEqual(pts.std(), 9.9, delta=1.0)

    def test_more_yards_and_touchdowns_score_more(self):
        rng = np.random.default_rng(1)
        n = 20000
        lo = go.team_points(np.full(n, 80.0), np.full(n, 180.0), np.full(n, 1.0), rng)
        hi = go.team_points(np.full(n, 150.0), np.full(n, 280.0), np.full(n, 4.0), rng)
        self.assertGreater(hi.mean(), lo.mean())

    def test_points_are_never_negative(self):
        rng = np.random.default_rng(2)
        pts = go.team_points(np.zeros(5000), np.zeros(5000), np.zeros(5000), rng)
        self.assertTrue(np.all(pts >= 0))

    def test_noise_is_right_skewed_like_the_real_residual(self):
        rng = np.random.default_rng(3)
        noise = go._resid_draw(200000, rng)
        self.assertAlmostEqual(noise.mean(), 0.0, delta=0.05)
        self.assertAlmostEqual(noise.std(), go.RESID_STD, delta=0.1)
        self.assertGreater(float(pd.Series(noise).skew()), 0.3)


class TestSimulation(unittest.TestCase):
    def test_shared_script_correlates_a_teams_players(self):
        # two identical-role players on the same team should move together across sims far more
        # than two players simulated with independent (unscripted) draws
        rng = np.random.default_rng(4)
        n = 20000
        script = np.clip(rng.normal(1.0, go.SCRIPT_SD, n), *go.SCRIPT_CLIP)
        a = go._draw_scripted(60, 400, script, rng)
        b = go._draw_scripted(50, 300, script, rng)
        corr_scripted = np.corrcoef(a, b)[0, 1]
        a2 = go._draw_scripted(60, 400, np.ones(n), np.random.default_rng(5))
        b2 = go._draw_scripted(50, 300, np.ones(n), np.random.default_rng(6))
        corr_unscripted = np.corrcoef(a2, b2)[0, 1]
        self.assertGreater(corr_scripted, corr_unscripted + 0.1)

    def test_full_game_simulation_runs_and_is_sane(self):
        home = [{"pass_yd": {"mean": 240, "var": 3000}, "td": {"mean": 1.2, "var": 1.2}},
                {"rush_yd": {"mean": 90, "var": 500}, "td": {"mean": 0.6, "var": 0.6}}]
        away = [{"pass_yd": {"mean": 210, "var": 2500}, "td": {"mean": 1.0, "var": 1.0}},
                {"rush_yd": {"mean": 70, "var": 400}, "td": {"mean": 0.5, "var": 0.5}}]
        hp, ap = go.simulate_game_scores(home, away, np.random.default_rng(7), n=10000)
        self.assertEqual(len(hp), 10000)
        self.assertTrue(np.all(hp >= 0) and np.all(ap >= 0))
        self.assertGreater(hp.mean(), ap.mean())     # home has the better projection here

    def test_empty_roster_still_returns_a_score_from_the_intercept_and_noise(self):
        hp, ap = go.simulate_game_scores([], [], np.random.default_rng(8), n=5000)
        self.assertEqual(len(hp), 5000)
        self.assertTrue(np.all(hp >= 0))


class TestSpreadTotalProbs(unittest.TestCase):
    def test_favored_team_covers_more_often(self):
        rng = np.random.default_rng(9)
        home = rng.normal(27, 6, 50000)
        away = rng.normal(20, 6, 50000)
        p = go.spread_total_probs(home, away, -3.5, 45.5)
        self.assertGreater(p["home_cover"], 0.5)
        self.assertAlmostEqual(p["home_cover"] + p["away_cover"] + p["home_push"], 1.0, places=6)

    def test_over_under_sum_to_one(self):
        rng = np.random.default_rng(10)
        home, away = rng.normal(24, 5, 20000), rng.normal(21, 5, 20000)
        p = go.spread_total_probs(home, away, -2.5, 100.5)   # an unreachable total -> ~all under
        self.assertLess(p["over"], 0.02)
        self.assertAlmostEqual(p["over"] + p["under"] + p["total_push"], 1.0, places=6)

    def test_projections_match_simple_means(self):
        home, away = np.full(1000, 24.0), np.full(1000, 17.0)
        p = go.spread_total_probs(home, away, -6.5, 41.5)
        self.assertAlmostEqual(p["proj_home"], 24.0)
        self.assertAlmostEqual(p["proj_away"], 17.0)
        self.assertAlmostEqual(p["proj_margin"], 7.0)
        self.assertAlmostEqual(p["proj_total"], 41.0)


class TestMarketBlend(unittest.TestCase):
    def test_low_n_eff_defers_almost_entirely_to_the_market(self):
        adj = go.blend_game_prob(0.87, 0.50, n_eff=3)
        self.assertLess(abs(adj - 0.50), 0.05)

    def test_high_n_eff_keeps_more_of_the_models_view(self):
        low = go.blend_game_prob(0.70, 0.50, n_eff=3)
        high = go.blend_game_prob(0.70, 0.50, n_eff=200)
        self.assertGreater(high, low)
        self.assertLessEqual(go.blend_game_prob(0.70, 0.50, n_eff=10_000), 0.50 + go.GAME_LAMBDA_MAX * 0.20)

    def test_lambda_max_override_replaces_the_static_default(self):
        default = go.blend_game_prob(0.90, 0.50, n_eff=100)
        overridden = go.blend_game_prob(0.90, 0.50, n_eff=100, lambda_max=0.40)
        self.assertGreater(overridden, default)

    def test_no_market_probability_leaves_the_model_unchanged(self):
        self.assertEqual(go.blend_game_prob(0.63, None, n_eff=50), 0.63)

    def test_fits_n_eff_ignores_missing_fits(self):
        fits = [{"pass_yd": {"n_eff": 10}, "rush_yd": None, "td": {"n_eff": 6}}]
        self.assertAlmostEqual(go.fits_n_eff(fits), 8.0)
        self.assertEqual(go.fits_n_eff([]), 0.0)


class TestRosterSelection(unittest.TestCase):
    def _history(self):
        rows = []
        for w in range(1, 9):
            rows.append(dict(player_id="qb1", team="AAA", position="QB", season=2026, attempts=32, carries=3, targets=0, week=w))
            rows.append(dict(player_id="qb2", team="AAA", position="QB", season=2026, attempts=8, carries=1, targets=0, week=w))
            rows.append(dict(player_id="rb1", team="AAA", position="RB", season=2026, attempts=0, carries=18, targets=3, week=w))
            rows.append(dict(player_id="rb2", team="AAA", position="RB", season=2026, attempts=0, carries=4, targets=1, week=w))
            rows.append(dict(player_id="wr_new", team="AAA", position="WR", season=2026, attempts=0, carries=0, targets=6, week=w if w > 6 else None))
        return pd.DataFrame([r for r in rows if r["week"] is not None])

    def test_picks_the_right_positions_by_usage(self):
        picks = go.select_offense(self._history(), "AAA")
        self.assertIn("qb1", picks)
        self.assertIn("rb1", picks)

    def test_requires_a_minimum_number_of_games(self):
        # wr_new only has 2 games of history -- below MIN_USAGE_GAMES
        picks = go.select_offense(self._history(), "AAA")
        self.assertNotIn("wr_new", picks)

    def test_inactive_player_is_excluded_so_the_backup_is_selected_instead(self):
        picks = go.select_offense(self._history(), "AAA", inactive_ids={"qb1"})
        self.assertNotIn("qb1", picks)
        self.assertIn("qb2", picks)

    def test_without_the_exclusion_the_backup_would_not_be_picked(self):
        # sanity check that the fixture actually needs the exclusion to matter
        picks = go.select_offense(self._history(), "AAA")
        self.assertIn("qb1", picks)
        self.assertNotIn("qb2", picks)   # ROSTER_SLOTS takes only 1 QB, and qb1 outranks qb2


class TestBuildTeamFitsWithInjuries(unittest.TestCase):
    def _history(self):
        rows = []
        for w in range(1, 11):
            rows.append(dict(player_id="starter", team="AAA", position="QB", season=2026, week=w,
                             opponent_team="XXX", attempts=34, carries=2, targets=0, passing_yards=240,
                             rushing_yards=10, rushing_tds=0.1, receiving_tds=0.0))
            rows.append(dict(player_id="backup", team="AAA", position="QB", season=2026, week=w,
                             opponent_team="XXX", attempts=6, carries=1, targets=0, passing_yards=35,
                             rushing_yards=2, rushing_tds=0.0, receiving_tds=0.0))
        return pd.DataFrame(rows)

    def test_inactive_starter_is_not_in_the_simulated_roster(self):
        h = self._history()
        fits = go.build_team_fits(h, "AAA", "BBB", {}, {}, inj_status={"starter": "Out"})
        # only the backup's pass_yd fit should be present, scaled up toward the team's own rate
        pass_means = [pf["pass_yd"]["mean"] for pf in fits if pf.get("pass_yd")]
        self.assertEqual(len(pass_means), 1)
        self.assertGreater(pass_means[0], 35)   # boosted well above his own backup-level rate

    def test_healthy_starter_is_not_boosted(self):
        h = self._history()
        fits = go.build_team_fits(h, "AAA", "BBB", {}, {}, inj_status={})
        pass_means = [pf["pass_yd"]["mean"] for pf in fits if pf.get("pass_yd")]
        self.assertEqual(len(pass_means), 1)
        self.assertAlmostEqual(pass_means[0], 240, delta=5)


class TestGamePriceConsensus(unittest.TestCase):
    def _event(self, home_prices, away_prices, over_prices, under_prices, point=(-3.0, 44.5)):
        books = {}
        for name, p in home_prices.items():
            books.setdefault(name, {"title": name, "markets": []})
            books[name]["markets"].append({"key": "spreads", "outcomes": [
                {"name": "Home", "point": point[0], "price": p},
                {"name": "Away", "point": -point[0], "price": away_prices[name]}]})
        for name, p in over_prices.items():
            books.setdefault(name, {"title": name, "markets": []})
            books[name]["markets"].append({"key": "totals", "outcomes": [
                {"name": "Over", "point": point[1], "price": p},
                {"name": "Under", "point": point[1], "price": under_prices[name]}]})
        return {"id": "g1", "home_team": "Home", "away_team": "Away", "bookmakers": list(books.values())}

    def test_fair_probabilities_and_best_prices(self):
        ev = self._event({"A": -150, "B": -145}, {"A": 130, "B": 125}, {"A": -110, "B": -105}, {"A": -110, "B": -115})
        c = odds_api.consensus_game_prices([ev])["g1"]
        self.assertEqual(c["home_spread"], -3.0)
        self.assertEqual(c["total"], 44.5)
        self.assertGreater(c["fair_home_cover"], 0.5)     # -150ish favorite covers more than half the time
        self.assertEqual(c["best_home"]["price"], -145)   # least-negative (highest payout) home price
        self.assertEqual(c["best_over"]["price"], -105)

    def test_missing_markets_do_not_crash(self):
        ev = {"id": "g2", "home_team": "H", "away_team": "A", "bookmakers": []}
        c = odds_api.consensus_game_prices([ev])["g2"]
        self.assertIsNone(c["fair_home_cover"])
        self.assertIsNone(c["best_home"])


if __name__ == "__main__":
    unittest.main()
