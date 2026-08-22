"""
test_model.py
=============
Unit tests for the pure math helpers in simulate.py and backtest.py.

These cover the parts of the model that are easiest to get subtly wrong
and hardest to notice by eye: odds conversion, vig removal, and the
empty-input edge case in the backtest summary.

Run:  python -m unittest tests.test_model -v
"""

import unittest

from simulate import american_to_prob, remove_vig, find_edge
from backtest import evaluate_bets
import pandas as pd


class TestAmericanToProb(unittest.TestCase):
    def test_negative_odds(self):
        # -110 is the standard "vig" price -> 110/210
        self.assertAlmostEqual(american_to_prob(-110), 110 / 210)

    def test_positive_odds(self):
        # +150 -> 100/250
        self.assertAlmostEqual(american_to_prob(150), 100 / 250)

    def test_even_money(self):
        self.assertAlmostEqual(american_to_prob(100), 0.5)


class TestRemoveVig(unittest.TestCase):
    def test_normalizes_to_one(self):
        # two -110 sides imply slightly more than 100% combined (the vig)
        p_over = american_to_prob(-110)
        p_under = american_to_prob(-110)
        no_vig_over, no_vig_under = remove_vig(p_over, p_under)
        self.assertAlmostEqual(no_vig_over + no_vig_under, 1.0)
        self.assertAlmostEqual(no_vig_over, 0.5)


class TestFindEdge(unittest.TestCase):
    def test_positive_edge_flagged(self):
        # model thinks 60% likely, market prices it at ~52.4% (-110) -> should flag
        result = find_edge(0.60, -110)
        self.assertGreater(result["edge_pts"], 0)
        self.assertTrue(result["bet"])

    def test_no_edge_not_flagged(self):
        # model roughly agrees with the market -> should not flag
        result = find_edge(american_to_prob(-110), -110)
        self.assertFalse(result["bet"])


class TestEvaluateBets(unittest.TestCase):
    def test_empty_bet_log_is_handled(self):
        result = evaluate_bets(pd.DataFrame())
        self.assertIn("note", result)


if __name__ == "__main__":
    unittest.main()
