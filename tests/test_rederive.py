"""
Tests for rederive_props.py: reading the data back out of a built page, re-deriving the leans and entries from it, and re-rendering.

Run:  python -m unittest discover -s tests -v
"""

import json
import pathlib
import sys
import tempfile
import unittest

import rederive_props as rp
from tests.test_lean import _log, _prop

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "web"))
import build_props_page as bp                      # noqa: E402

TEMPLATE = "<script>\nconst DATA = __PROPS_JSON__;\nconst TRACK = __TRACK_RECORD_JSON__;\nconst PLAYER_HISTORY = __PLAYER_HISTORY_JSON__;\n</script>"


def _data():
    props = []
    for g in range(3):
        for i in range(4):
            props.append(_prop(player_id=f"g{g}p{i}", player=f"P{g}{i}", team=f"H{g}", opp=f"A{g}", game=f"A{g} @ H{g}",
                               market=["player_receptions", "player_reception_yds", "player_rush_yds", "player_pass_yds"][i],
                               pickem={"line": 4.5, "book": "Underdog", "side": "over", "prob": 0.58, "agree": True},
                               dfs=[{"book": "Underdog", "line": 4.5, "over": 0.70 - 0.01 * i, "under": 0.30}]))
    games = [{"game": f"A{g} @ H{g}", "home": f"H{g}", "away": f"A{g}", "home_spread": -3.0, "total": 45.0, "week": 11, "in_window": True} for g in range(3)]
    return {"season": 2026, "week": 11, "props": props, "games": games, "context": {"dvp": {}}, "social": None}


class TestRederive(unittest.TestCase):
    def test_blobs_round_trip_through_the_render(self):
        html = bp.render(TEMPLATE, json.dumps(_data()), json.dumps({"n_graded": 3}), json.dumps({"x": [1, 2]}))
        blobs = rp.extract_blobs(html)
        self.assertEqual(json.loads(blobs["DATA"])["week"], 11)
        self.assertEqual(json.loads(blobs["TRACK"]), {"n_graded": 3})
        self.assertEqual(json.loads(blobs["PLAYER_HISTORY"]), {"x": [1, 2]})

    def test_a_closing_script_tag_inside_the_data_does_not_break_extraction(self):
        d = _data()
        d["props"][0]["player"] = "</script><b>x"
        html = bp.render(TEMPLATE, json.dumps(d))
        self.assertNotIn("</script><b>", html)
        self.assertEqual(json.loads(rp.extract_blobs(html)["DATA"])["props"][0]["player"], "</script><b>x")

    def test_a_page_without_the_data_is_an_error(self):
        with self.assertRaises(ValueError):
            rp.extract_blobs("<html></html>")

    def test_rederive_adds_leans_entries_and_the_replay(self):
        d = rp.rederive(_data(), _log(900))
        self.assertTrue(all(p["lean"] for p in d["props"]))
        self.assertGreaterEqual(len(d["parlays"]["pickem"]["entries"]), 1)
        self.assertIn("lean_info", d)
        self.assertIn("pickem_backtest", d)
        json.dumps(d)

    def test_the_cli_rewrites_a_page_in_place(self):
        tmp = pathlib.Path(tempfile.mkdtemp())
        tpl = tmp / "t.html"
        tpl.write_text(TEMPLATE, encoding="utf-8")
        page = tmp / "p.html"
        page.write_text(bp.render(TEMPLATE, json.dumps(_data())), encoding="utf-8")
        rp.main([str(page), "--template", str(tpl), "--no-schedule"])
        out = json.loads(rp.extract_blobs(page.read_text(encoding="utf-8"))["DATA"])
        self.assertTrue(out["props"][0]["lean"])


if __name__ == "__main__":
    unittest.main()
