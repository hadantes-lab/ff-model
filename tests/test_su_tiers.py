"""
Tests for the straight-up tiers (S best ... F): the probability bands, each tier's historical record, how a
week's picks are assigned, and the page's grouping logic (run under node when it is installed).

Run:  python -m unittest discover -s tests -v
"""

import json
import pathlib
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
import pandas as pd

import straight_up as su

TEMPLATE = pathlib.Path(__file__).resolve().parent.parent / "web" / "props_template.html"


class TestTierBands(unittest.TestCase):
    def test_boundaries_belong_to_the_higher_tier(self):
        for p, tier in ((0.99, "S"), (0.84, "S"), (0.8399, "A"), (0.76, "A"), (0.7599, "B"), (0.70, "B"), (0.6999, "C"),
                        (0.65, "C"), (0.6499, "D"), (0.60, "D"), (0.5999, "E"), (0.55, "E"), (0.5499, "F"), (0.50, "F")):
            self.assertEqual(su.tier_of(p), tier, p)

    def test_seven_tiers_best_first_and_every_probability_gets_one(self):
        self.assertEqual(su.TIER_ORDER, list("SABCDEF"))
        floors = [f for _, f in su.TIERS]
        self.assertEqual(floors, sorted(floors, reverse=True))
        self.assertEqual({su.tier_of(p) for p in np.linspace(0.5, 1.0, 501)}, set("SABCDEF"))

    def test_higher_probability_never_means_a_lower_tier(self):
        order = [su.TIER_ORDER.index(su.tier_of(p)) for p in np.linspace(0.5, 0.99, 200)]
        self.assertEqual(order, sorted(order, reverse=True))

    def test_range_labels(self):
        self.assertEqual(su.tier_range("S"), "84%+")
        self.assertEqual(su.tier_range("A"), "76-84%")
        self.assertEqual(su.tier_range("E"), "55-60%")
        self.assertEqual(su.tier_range("F"), "under 55%")

    def test_a_spread_lands_in_the_expected_tier(self):
        # a 3-point favorite (~60%) is D; a touchdown favorite (~73%) is B; a pick-em is F
        self.assertEqual(su.tier_of(su.win_prob(-3.0)), "D")
        self.assertEqual(su.tier_of(su.win_prob(-7.0)), "B")
        self.assertEqual(su.tier_of(su.win_prob(0.0)), "F")
        self.assertEqual(su.tier_of(1 - su.win_prob(+14.0)), "S")                 # a road team laying 14


def _games(spreads_and_results):
    """nflverse-style rows: (home spread in nflverse sign, home score, away score)."""
    rows = [{"season": 2025, "game_type": "REG", "home_score": h, "away_score": a, "spread_line": sp}
            for sp, h, a in spreads_and_results]
    return pd.DataFrame(rows)


class TestTierHistory(unittest.TestCase):
    def test_records_each_tier_by_the_favorites_result(self):
        # nflverse +14 = home favored by 14 -> S tier. Home wins: right. Home loses: wrong.
        h = su.tier_history(_games([(14, 30, 10), (14, 31, 7), (14, 10, 20), (14, 24, 17)]), first_season=2025)
        self.assertEqual(h["S"]["n"], 4)
        self.assertAlmostEqual(h["S"]["win_rate"], 0.75)
        self.assertAlmostEqual(h["S"]["share"], 1.0)

    def test_a_road_favorite_is_judged_by_the_road_teams_result(self):
        h = su.tier_history(_games([(-14, 10, 30), (-14, 28, 14)]), first_season=2025)     # away favored by 14
        self.assertAlmostEqual(h["S"]["win_rate"], 0.5)                                       # one right, one wrong

    def test_ties_count_half_and_unfinished_or_unlined_games_are_left_out(self):
        h = su.tier_history(_games([(14, 20, 20), (14, np.nan, np.nan), (np.nan, 20, 10)]), first_season=2025)
        self.assertEqual(h["S"]["n"], 1)
        self.assertAlmostEqual(h["S"]["win_rate"], 0.5)
        self.assertEqual(su.tier_history(pd.DataFrame(columns=["season", "game_type", "home_score", "away_score", "spread_line"])), {})

    def test_a_tier_with_no_games_is_absent(self):
        self.assertNotIn("F", su.tier_history(_games([(14, 30, 10)]), first_season=2025))


class TestSlateTiers(unittest.TestCase):
    def test_every_entry_has_a_tier_matching_its_probability_and_the_summary_counts_them(self):
        es = [{"game": f"{c} @ {h}", "home": h, "away": c, "spread": sp, "p_home": su.win_prob(sp), "status": "upcoming"}
              for h, c, sp in (("A", "B", -14.0), ("C", "D", -7.0), ("E", "F", -3.0), ("G", "H", 0.0))]
        ranked = su.rank_slate(es)
        self.assertEqual([e["tier"] for e in ranked], ["S", "B", "D", "F"])
        for e in ranked:
            self.assertEqual(e["tier"], su.tier_of(e["pick_prob"]))
        self.assertEqual(su.pool_summary(ranked)["tiers"], {"S": 1, "B": 1, "D": 1, "F": 1})
        self.assertAlmostEqual(su.pool_summary(ranked)["expected_winners"], round(sum(e["pick_prob"] for e in ranked), 1))

    def test_numeric_pool_points_are_still_there_for_anyone_who_needs_them(self):
        es = su.rank_slate([{"game": "B @ A", "home": "A", "away": "B", "spread": -10.0, "p_home": su.win_prob(-10.0)},
                            {"game": "D @ C", "home": "C", "away": "D", "spread": -1.0, "p_home": su.win_prob(-1.0)}])
        self.assertEqual([e["points"] for e in es], [2, 1])


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class TestPageGrouping(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        src = TEMPLATE.read_text(encoding="utf-8")
        cls.block = src[src.index("/* SU:BEGIN"):src.index("/* SU:END */")]

    def _run(self, expr):
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
            f.write(self.block + f"\nconsole.log(JSON.stringify({expr}));")
        out = subprocess.run(["node", f.name], capture_output=True, text=True, timeout=30)
        pathlib.Path(f.name).unlink()
        self.assertEqual(out.returncode, 0, out.stderr)
        return json.loads(out.stdout)

    TIERS = [{"tier": t, "range": "r", "blurb": "b"} for t in "SABCDEF"]
    ES = [{"game": "g1", "pick": "AAA", "tier": "B", "pick_prob": 0.72}, {"game": "g2", "pick": "BBB", "tier": "S", "pick_prob": 0.9},
          {"game": "g3", "pick": "CCC", "tier": "B", "pick_prob": 0.74}, {"game": "g4", "pick": "DDD", "tier": "F", "pick_prob": 0.52},
          {"game": "g5", "pick": "EEE", "tier": "B", "pick_prob": 0.72}]

    def test_groups_come_out_best_tier_first_and_empty_tiers_are_dropped(self):
        g = self._run(f"suGroups({json.dumps(self.ES)}, {json.dumps(self.TIERS)})")
        self.assertEqual([x["tier"] for x in g], ["S", "B", "F"])

    def test_within_a_tier_the_most_confident_is_first_and_ties_are_stable(self):
        g = self._run(f"suGroups({json.dumps(self.ES)}, {json.dumps(self.TIERS)})")
        b = [x for x in g if x["tier"] == "B"][0]
        self.assertEqual([e["pick"] for e in b["games"]], ["CCC", "AAA", "EEE"])        # .74, then the .72s by game name

    def test_copy_text_lists_the_picks_by_tier(self):
        txt = self._run(f"suCopyText(suGroups({json.dumps(self.ES)}, {json.dumps(self.TIERS)}))")
        self.assertEqual(txt, "S: BBB\nB: CCC, AAA, EEE\nF: DDD")

    def test_no_games_no_groups(self):
        self.assertEqual(self._run(f"suGroups([], {json.dumps(self.TIERS)})"), [])


if __name__ == "__main__":
    unittest.main()
