"""
Tests for the Defense vs Position table: the per-game numbers context_data.dvp_table builds, and the ranking/label logic in the
DVP block of web/props_template.html (run under node when it is installed).

Run:  python -m unittest discover -s tests -v
"""

import json
import pathlib
import shutil
import subprocess
import tempfile
import unittest

import pandas as pd

import context_data as cd

TEMPLATE = pathlib.Path(__file__).resolve().parent.parent / "web" / "props_template.html"


def _row(team, opp, season, week, pos, **kw):
    d = {"team": team, "opponent_team": opp, "season": season, "week": week, "position": pos, "attempts": 0, "completions": 0, "passing_yards": 0,
         "passing_tds": 0, "carries": 0, "rushing_yards": 0, "rushing_tds": 0, "targets": 0, "receptions": 0, "receiving_yards": 0, "receiving_tds": 0}
    d.update(kw)
    return d


def _history():
    rows = []
    for wk, (att, tgt) in enumerate([(30, 20), (40, 30), (35, 25), (45, 35), (50, 40)], start=1):   # DEF faces OFF every week
        rows += [_row("OFF", "DEF", 2026, wk, "QB", attempts=att, completions=att // 2, passing_yards=att * 7, passing_tds=2, carries=3, rushing_yards=10),
                 _row("OFF", "DEF", 2026, wk, "WR", targets=tgt, receptions=tgt // 2, receiving_yards=tgt * 8, receiving_tds=1),
                 _row("OFF", "DEF", 2026, wk, "RB", carries=20, rushing_yards=90, rushing_tds=1, targets=5, receptions=4, receiving_yards=30),
                 _row("DEF", "OFF", 2026, wk, "QB", attempts=25)]
    rows += [_row("OFF", "DEF", 2025, 1, "QB", attempts=20), _row("DEF", "OFF", 2025, 1, "QB", attempts=22)]
    return pd.DataFrame(rows)


class TestDvpTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = cd.dvp_table(_history(), 2026)

    def test_season_window_is_per_game_team_totals(self):
        a = self.t["windows"]["season"]["ALL"]["DEF"]
        self.assertEqual(a["g"], 5)
        self.assertAlmostEqual(a["pass_att"], 40.0)                     # (30+40+35+45+50)/5
        self.assertAlmostEqual(a["rush_att"], 23.0)                     # QB 3 + RB 20
        self.assertAlmostEqual(a["tgt"], (20 + 30 + 25 + 35 + 40) / 5 + 5)

    def test_position_views_only_count_that_position(self):
        qb = self.t["windows"]["season"]["QB"]["DEF"]
        self.assertAlmostEqual(qb["pass_att"], 40.0)
        self.assertEqual(qb["tgt"], 0.0)
        wr = self.t["windows"]["season"]["WR"]["DEF"]
        self.assertEqual(wr["pass_att"], 0.0)
        self.assertAlmostEqual(wr["tgt"], 30.0)

    def test_last4_uses_the_last_four_games(self):
        a = self.t["windows"]["last4"]["ALL"]["DEF"]
        self.assertEqual(a["g"], 4)
        self.assertAlmostEqual(a["pass_att"], (40 + 35 + 45 + 50) / 4)

    def test_previous_season_window(self):
        self.assertAlmostEqual(self.t["windows"]["prev"]["ALL"]["DEF"]["pass_att"], 20.0)

    def test_a_position_that_did_nothing_in_a_game_counts_as_zero_not_missing(self):
        h = _history()
        h = h[~((h["opponent_team"] == "DEF") & (h["week"] == 5) & (h["position"] == "WR"))]       # no WR stat line in week 5
        wr = cd.dvp_table(h, 2026)["windows"]["season"]["WR"]["DEF"]
        self.assertEqual(wr["g"], 5)
        self.assertAlmostEqual(wr["tgt"], (20 + 30 + 25 + 35) / 5)

    def test_game_by_game_and_through_week(self):
        g = self.t["games"]["DEF"]
        self.assertEqual([x[0] for x in g], [1, 2, 3, 4, 5])
        self.assertEqual(g[0][1], "OFF")
        self.assertEqual(g[0][2]["QB"][self.t["stats"].index("pass_att")], 30.0)
        self.assertEqual(self.t["through"], 5)
        json.dumps(self.t)

    def test_rows_without_a_defense_or_position_are_ignored(self):
        h = pd.concat([_history(), pd.DataFrame([_row("OFF", "DEF", 2026, 1, "K", attempts=99)])], ignore_index=True)
        self.assertAlmostEqual(cd.dvp_table(h, 2026)["windows"]["season"]["ALL"]["DEF"]["pass_att"], 40.0)


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class TestDvpBlock(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        src = TEMPLATE.read_text(encoding="utf-8")
        cls.block = src[src.index("/* DVP:BEGIN"):src.index("/* DVP:END */")]

    def _run(self, expr):
        code = self.block + f"\nconsole.log(JSON.stringify({expr}));"
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
            f.write(code)
        out = subprocess.run(["node", f.name], capture_output=True, text=True, timeout=30)
        pathlib.Path(f.name).unlink()
        self.assertEqual(out.returncode, 0, out.stderr)
        return json.loads(out.stdout)

    def test_ordinals(self):
        self.assertEqual(self._run("[1, 2, 3, 4, 11, 12, 13, 21, 22, 23, 32].map(ordinal)"),
                         ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "23rd", "32nd"])

    def test_rank_one_is_the_most_allowed(self):
        rows = [{"team": "A", "x": 10}, {"team": "B", "x": 30}, {"team": "C", "x": 20}]
        self.assertEqual(self._run(f"dvRanks({json.dumps(rows)}, 'x')"), {"A": 3, "B": 1, "C": 2})

    def test_ties_share_the_better_rank(self):
        rows = [{"team": "A", "x": 5}, {"team": "B", "x": 5}, {"team": "C", "x": 1}]
        self.assertEqual(self._run(f"dvRanks({json.dumps(rows)}, 'x')"), {"A": 1, "B": 1, "C": 3})

    def test_labels_read_most_least_or_of_32(self):
        self.assertEqual(self._run("[1, 10, 11, 22, 23, 32].map(r => dvRankLabel(r, 32))"),
                         ["1st most", "10th most", "11th of 32", "22nd of 32", "10th least", "1st least"])

    def test_columns_by_position(self):
        self.assertEqual(self._run("DV_COLS.WR"), ["tgt", "rec", "rec_yds", "rec_td"])
        self.assertEqual(len(self._run("DV_COLS.ALL")), 11)
        self.assertEqual(self._run("DV_COLS.QB")[0], "pass_att")

    def test_rows_come_from_the_chosen_window_and_position(self):
        table = {"windows": {"last4": {"WR": {"DET": {"g": 4, "tgt": 20.0}}}}}
        self.assertEqual(self._run(f"dvRows({json.dumps(table)}, 'last4', 'WR')"), [{"team": "DET", "g": 4, "tgt": 20.0}])
        self.assertEqual(self._run(f"dvRows({json.dumps(table)}, 'season', 'WR')"), [])


if __name__ == "__main__":
    unittest.main()
