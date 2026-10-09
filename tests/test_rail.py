"""
Tests for the left-rail browser's grouping logic (the RAIL block in web/props_template.html), run under
node when it is installed. The DOM rendering around it is checked in a real browser, not here.

Run:  python -m unittest discover -s tests -v
"""

import json
import pathlib
import shutil
import subprocess
import tempfile
import unittest

TEMPLATE = pathlib.Path(__file__).resolve().parent.parent / "web" / "props_template.html"


def _prop(player, pos, team, game, market, line, commence="2026-10-11T17:00:00Z", pid=None, **kw):
    p = {"player": player, "player_id": pid or player, "pos": pos, "team": team, "game": game, "market": market,
         "line": line, "label": market, "kind": "yards", "commence": commence, "top_rank": None}
    p.update(kw)
    return p


PROPS = [
    _prop("Star QB", "QB", "HOM", "AWY @ HOM", "player_pass_tds", 1.5),
    _prop("Star QB", "QB", "HOM", "AWY @ HOM", "player_pass_yds", 250.5),
    _prop("Star QB", "QB", "HOM", "AWY @ HOM", "player_rush_yds", 12.5),
    _prop("Lead RB", "RB", "HOM", "AWY @ HOM", "player_rush_yds", 70.5, top_rank=3),
    _prop("Slot WR", "WR", "AWY", "AWY @ HOM", "player_reception_yds", 55.5),
    _prop("Big WR", "WR", "AWY", "AWY @ HOM", "player_reception_yds", 80.5),
    _prop("Tight End", "TE", "AWY", "AWY @ HOM", "player_receptions", 4.5),
    _prop("Other QB", "QB", "NYY", "BOS @ NYY", "player_pass_yds", 230.5, commence="2026-10-11T20:25:00Z"),
    _prop("Early WR", "WR", "ABC", "DEF @ ABC", "player_reception_yds", 60.5, commence="2026-10-11T13:00:00Z"),
]
LINES = [{"game": "AWY @ HOM", "home_spread": -3.5, "total": 44.5}]


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class TestRailModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        src = TEMPLATE.read_text(encoding="utf-8")
        cls.block = src[src.index("/* RAIL:BEGIN"):src.index("/* RAIL:END */")]

    def _run(self, expr):
        code = self.block + f"\nconsole.log(JSON.stringify({expr}));"
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
            f.write(code)
        out = subprocess.run(["node", f.name], capture_output=True, text=True, timeout=30)
        pathlib.Path(f.name).unlink()
        self.assertEqual(out.returncode, 0, out.stderr)
        return json.loads(out.stdout)

    def model(self, q="", lines=LINES):
        return self._run(f"railModel({json.dumps(PROPS)}, {json.dumps(lines)}, {json.dumps(q)})")

    def test_games_are_ordered_by_kickoff_with_prop_counts(self):
        m = self.model()
        self.assertEqual([g["game"] for g in m], ["DEF @ ABC", "AWY @ HOM", "BOS @ NYY"])
        self.assertEqual([g["n"] for g in m], [1, 7, 1])

    def test_game_lines_attach_by_game_name_and_are_null_when_missing(self):
        m = {g["game"]: g for g in self.model()}
        self.assertEqual((m["AWY @ HOM"]["spread"], m["AWY @ HOM"]["total"]), (-3.5, 44.5))
        self.assertIsNone(m["BOS @ NYY"]["spread"])
        self.assertEqual((m["AWY @ HOM"]["away"], m["AWY @ HOM"]["home"]), ("AWY", "HOM"))

    def test_a_player_appears_once_with_a_featured_prop_by_position(self):
        g = [g for g in self.model() if g["game"] == "AWY @ HOM"][0]
        names = [pl["name"] for pl in g["players"]]
        self.assertEqual(names.count("Star QB"), 1)
        qb = [pl for pl in g["players"] if pl["name"] == "Star QB"][0]
        self.assertEqual((qb["feat"]["market"], qb["n"]), ("player_pass_yds", 3))        # pass yards, not pass TDs
        rb = [pl for pl in g["players"] if pl["name"] == "Lead RB"][0]
        self.assertTrue(rb["top"])                                                     # carries the top-10 star
        te = [pl for pl in g["players"] if pl["name"] == "Tight End"][0]
        self.assertEqual(te["feat"]["market"], "player_receptions")                    # falls back to what he has

    def test_players_sort_by_position_then_biggest_line(self):
        g = [g for g in self.model() if g["game"] == "AWY @ HOM"][0]
        self.assertEqual([pl["name"] for pl in g["players"]], ["Star QB", "Lead RB", "Big WR", "Slot WR", "Tight End"])

    def test_search_by_player_keeps_only_matching_games_and_players(self):
        m = self.model("slot")
        self.assertEqual([g["game"] for g in m], ["AWY @ HOM"])
        self.assertEqual([pl["name"] for pl in m[0]["players"]], ["Slot WR"])
        self.assertTrue(m[0]["searching"])

    def test_search_by_team_or_game_name(self):
        by_team = self.model("nyy")
        self.assertEqual([g["game"] for g in by_team], ["BOS @ NYY"])
        by_game = self.model("awy @ hom")
        self.assertEqual(len(by_game[0]["players"]), 5)                                # whole game matches: all its players

    def test_no_match_and_empty_input(self):
        self.assertEqual(self.model("zzz"), [])
        self.assertEqual(self._run("railModel([], [], '')"), [])


if __name__ == "__main__":
    unittest.main()
