"""
Tests for matchups.py (man/zone coverage matchups) and the recency-weighted defense strength.

Run:  python -m unittest discover -s tests -v
"""

import unittest

import numpy as np
import pandas as pd

from matchups import (
    MAN, MULT_CAP, ZONE, CoverageModel, league_zone_man_ratio, matchup_multiplier, receiver_splits,
    team_zone_rates,
)
from props_model import CURRENT_SEASON_WEIGHT, MARKETS, opponent_multipliers


def _targets(rows):
    return pd.DataFrame(rows, columns=["season", "defteam", "receiver_player_id", "coverage", "yards_gained"])


class TestSchemeRates(unittest.TestCase):
    def test_zone_rate_is_shrunk_toward_the_league_and_ordered(self):
        rows = [(2025, "ZON", "r", ZONE, 7)] * 90 + [(2025, "ZON", "r", MAN, 7)] * 10 \
             + [(2025, "MAN", "r", ZONE, 7)] * 20 + [(2025, "MAN", "r", MAN, 7)] * 80
        league, teams = team_zone_rates(_targets(rows))
        self.assertAlmostEqual(league, 0.55)
        self.assertGreater(teams["ZON"], league)
        self.assertLess(teams["MAN"], league)
        self.assertLess(teams["ZON"], 0.90)          # carried forward only in part (SCHEME_TRUST), not all the way to 90%

    def test_newer_season_counts_more(self):
        old_zone = [(2024, "T", "r", ZONE, 7)] * 50 + [(2025, "T", "r", MAN, 7)] * 50
        new_zone = [(2024, "T", "r", MAN, 7)] * 50 + [(2025, "T", "r", ZONE, 7)] * 50
        self.assertGreater(team_zone_rates(_targets(new_zone))[1]["T"], team_zone_rates(_targets(old_zone))[1]["T"])


class TestReceiverSplits(unittest.TestCase):
    def test_requires_enough_targets_on_both_sides(self):
        rows = [(2025, "X", "a", ZONE, 8)] * 30 + [(2025, "X", "a", MAN, 5)] * 30 \
             + [(2025, "X", "b", ZONE, 8)] * 30 + [(2025, "X", "b", MAN, 5)] * 3
        s = receiver_splits(_targets(rows))
        self.assertIn("a", s)
        self.assertNotIn("b", s)                      # only 3 targets against man: not usable
        self.assertAlmostEqual(s["a"]["zone_ypt"], 8.0)

    def test_league_ratio(self):
        rows = [(2025, "X", "a", ZONE, 8)] * 10 + [(2025, "X", "a", MAN, 4)] * 10
        self.assertAlmostEqual(league_zone_man_ratio(_targets(rows)), 2.0)


class TestMultiplier(unittest.TestCase):
    def split(self, zone_ypt, man_ypt, n=100):
        return {"zone_n": n, "zone_ypt": zone_ypt, "man_n": n, "man_ypt": man_ypt}

    def test_no_split_means_no_adjustment(self):
        self.assertEqual(matchup_multiplier(None, 0.75, 0.60, 1.1)["mult"], 1.0)

    def test_average_receiver_is_unaffected_by_scheme(self):
        m = matchup_multiplier(self.split(7.3, 6.7), 0.75, 0.60, 7.3 / 6.7)      # split == league split
        self.assertAlmostEqual(m["mult"], 1.0, places=6)

    def test_zone_beater_gains_against_zone_heavy_defense_and_loses_against_man_heavy(self):
        s = self.split(9.0, 5.0)
        self.assertGreater(matchup_multiplier(s, 0.75, 0.60, 1.09)["mult"], 1.0)
        self.assertLess(matchup_multiplier(s, 0.45, 0.60, 1.09)["mult"], 1.0)

    def test_zone_struggler_is_the_mirror_image(self):
        s = self.split(5.0, 9.0)
        self.assertLess(matchup_multiplier(s, 0.75, 0.60, 1.09)["mult"], 1.0)
        self.assertGreater(matchup_multiplier(s, 0.45, 0.60, 1.09)["mult"], 1.0)

    def test_thin_samples_are_trusted_less(self):
        big = matchup_multiplier(self.split(9.0, 5.0, n=400), 0.75, 0.60, 1.09)
        small = matchup_multiplier(self.split(9.0, 5.0, n=15), 0.75, 0.60, 1.09)
        self.assertGreater(big["reliability"], small["reliability"])
        self.assertGreater(big["mult"] - 1, small["mult"] - 1)

    def test_adjustment_is_small_and_capped(self):
        m = matchup_multiplier(self.split(30.0, 1.0, n=10_000), 1.0, 0.0, 1.0)     # absurd inputs
        self.assertLessEqual(m["mult"], 1 + MULT_CAP)
        typical = matchup_multiplier(self.split(9.0, 5.0, n=80), 0.72, 0.60, 1.09)
        self.assertLess(abs(typical["mult"] - 1), 0.03)          # typical effect is a couple of percent

    def test_coverage_model_falls_back_to_neutral_for_unknown_players_and_teams(self):
        rows = [(2025, "AAA", "a", ZONE, 8)] * 30 + [(2025, "AAA", "a", MAN, 5)] * 30
        cm = CoverageModel(_targets(rows))
        block = cm.for_receiver("nobody", "ZZZ")
        self.assertEqual(block["mult"], 1.0)
        self.assertIsNone(block["split"])
        self.assertEqual(block["opp_zone"], round(cm.league_zone, 3))


class TestRecencyWeightedDefense(unittest.TestCase):
    def _history(self):
        rows = []
        for g in range(8):        # last season: SOFT gave up 120
            rows += [dict(opponent_team="SOFT", game_id=f"o{g}", position="WR", season=2025, receiving_yards=120.0),
                     dict(opponent_team="AVG", game_id=f"a{g}", position="WR", season=2025, receiving_yards=80.0)]
        for g in range(3):        # this season: SOFT has clamped down to 40
            rows += [dict(opponent_team="SOFT", game_id=f"n{g}", position="WR", season=2026, receiving_yards=40.0),
                     dict(opponent_team="AVG", game_id=f"b{g}", position="WR", season=2026, receiving_yards=80.0)]
        return pd.DataFrame(rows)

    def test_current_season_games_count_extra(self):
        spec = MARKETS["player_reception_yds"]
        weighted = opponent_multipliers(self._history(), spec)[("SOFT", "WR")]
        flat = self._history().assign(season=2025)                       # same games, no recency
        unweighted = opponent_multipliers(flat, spec)[("SOFT", "WR")]
        self.assertLess(weighted, unweighted)                            # the new weeks pull the rating down faster
        self.assertGreater(CURRENT_SEASON_WEIGHT, 1.0)


if __name__ == "__main__":
    unittest.main()
