"""
Tests for the extra markets: choose_markets (the credit guard), the longest-play data built from
play-by-play, and how build_history treats it.

Run:  python -m unittest discover -s tests -v
"""

import sys
import types
import unittest
from unittest import mock

import pandas as pd

import export_props as ep
import props_model as pm


class TestChooseMarkets(unittest.TestCase):
    BASE = list(pm.CORE_MARKETS)

    def test_off_keeps_the_base_set(self):
        m, note = ep.choose_markets(self.BASE, "off", 20000, 14)
        self.assertEqual(m, self.BASE)

    def test_on_adds_every_extra_regardless_of_credits(self):
        m, _ = ep.choose_markets(self.BASE, "on", 3, 14)
        self.assertEqual(set(m), set(self.BASE) | set(pm.EXTENDED_MARKETS))
        self.assertEqual(len(m), len(set(m)))                                  # no duplicates

    def test_auto_on_the_free_plan_never_adds_extras(self):
        # 500 credits a month is the whole free plan: a 14-game slate would leave far under the reserve
        m, note = ep.choose_markets(self.BASE, "auto", 301, 14)
        self.assertEqual(m, self.BASE)
        self.assertIn("skipped", note)
        m, _ = ep.choose_markets(self.BASE, "auto", 500, 1)                     # even a full free plan and one game
        self.assertEqual(m, self.BASE)

    def test_auto_on_a_paid_plan_adds_them_while_headroom_remains(self):
        m, note = ep.choose_markets(self.BASE, "auto", 19000, 14)
        self.assertEqual(set(m), set(self.BASE) | set(pm.EXTENDED_MARKETS))
        self.assertIn("on", note)
        cost = 14 * (len(self.BASE) + len(pm.EXTENDED_MARKETS))
        m, _ = ep.choose_markets(self.BASE, "auto", cost + ep.EXTENDED_MIN_CREDITS - 1, 14)    # one credit short
        self.assertEqual(m, self.BASE)
        m, _ = ep.choose_markets(self.BASE, "auto", cost + ep.EXTENDED_MIN_CREDITS, 14)
        self.assertEqual(set(m), set(self.BASE) | set(pm.EXTENDED_MARKETS))

    def test_unknown_credit_balance_is_treated_as_unsafe(self):
        m, note = ep.choose_markets(self.BASE, "auto", None, 14)
        self.assertEqual(m, self.BASE)
        self.assertIn("unknown", note)

    def test_every_extended_market_is_modeled_and_has_a_role_floor(self):
        for m in pm.EXTENDED_MARKETS:
            self.assertIn(m, pm.MARKETS)
            self.assertIn(m, pm.ROLE_FLOOR)


def _pbp():
    rows = []

    def play(week, ptype, yards, receiver=None, rusher=None, complete=0, kneel=0):
        rows.append({"season": 2026, "week": week, "season_type": "REG", "play_type": ptype, "complete_pass": complete,
                     "receiver_player_id": receiver, "rusher_player_id": rusher, "yards_gained": yards, "qb_kneel": kneel})
    play(1, "pass", 12, receiver="wr1", complete=1)
    play(1, "pass", 47, receiver="wr1", complete=1)            # the longest catch
    play(1, "pass", 80, receiver="wr1", complete=0)            # incomplete: yards_gained is 0 in real data, but must not count
    play(1, "run", 7, rusher="rb1")
    play(1, "run", 23, rusher="rb1")                           # the longest rush
    play(1, "run", -1, rusher="qb1", kneel=1)                  # kneel: not a rush
    play(1, "run", 9, rusher="qb1")                            # a scramble counts
    play(2, "pass", 5, receiver="wr1", complete=1)
    return pd.DataFrame(rows)


class TestLongestPlays(unittest.TestCase):
    def _load(self):
        mod = types.SimpleNamespace(load_pbp=lambda seasons: types.SimpleNamespace(to_pandas=_pbp))
        with mock.patch.dict(sys.modules, {"nflreadpy": mod}):
            return pm.load_longest([2026])

    def test_longest_catch_and_rush_per_player_game(self):
        L = self._load().set_index(["week", "player_id"])
        self.assertEqual(L.loc[(1, "wr1"), "long_rec"], 47)
        self.assertEqual(L.loc[(2, "wr1"), "long_rec"], 5)
        self.assertEqual(L.loc[(1, "rb1"), "long_rush"], 23)
        self.assertEqual(L.loc[(1, "qb1"), "long_rush"], 9)                    # scramble yes, kneel no

    def test_incomplete_passes_do_not_count_as_a_catch(self):
        L = self._load().set_index(["week", "player_id"])
        self.assertNotEqual(L.loc[(1, "wr1"), "long_rec"], 80)

    def test_build_history_merges_them_and_defaults_to_zero_without(self):
        ps = pd.DataFrame([
            {"season": 2026, "week": 1, "season_type": "REG", "player_id": "wr1", "position": "WR", "receptions": 2},
            {"season": 2026, "week": 1, "season_type": "REG", "player_id": "rb1", "position": "RB", "carries": 3}])
        for col in {c for spec in pm.MARKETS.values() for c in spec.cols} - {"long_rec", "long_rush"}:
            if col not in ps:
                ps[col] = 0.0                                                   # the other stat columns build_history needs
        h = pm.build_history(ps, self._load()).set_index("player_id")
        self.assertEqual(h.loc["wr1", "long_rec"], 47)
        self.assertEqual(h.loc["rb1", "long_rush"], 23)
        self.assertEqual(h.loc["rb1", "long_rec"], 0)                          # a back with no catch: 0, not NaN
        bare = pm.build_history(ps)                                            # callers that never load pbp still work
        self.assertTrue((bare["long_rec"] == 0).all() and (bare["long_rush"] == 0).all())

    def test_longest_markets_read_the_new_columns(self):
        self.assertEqual(pm.MARKETS["player_reception_longest"].cols, ("long_rec",))
        self.assertEqual(pm.MARKETS["player_rush_longest"].cols, ("long_rush",))


if __name__ == "__main__":
    unittest.main()
