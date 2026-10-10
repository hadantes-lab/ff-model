"""
Tests for espn_lines.py: reading ESPN's prop-bet structure (priced Over/Under pair = the main line, unpriced
alternate-line ladders ignored), the athlete-id mapping, provider fallback, resuming, and the validation
against our own logged lines. The ESPN API is faked; shapes copied from real responses.

Run:  python -m unittest discover -s tests -v
"""

import json
import pathlib
import tempfile
import unittest

import pandas as pd

import espn_lines as el


def _price(american):
    return {"american": f"{american:+d}" if american > 0 else str(american), "value": 1.9, "decimal": 1.9}


def item(aid, type_name, cur_line, open_line=None, side=None, price=-110, open_price=None, updated="2025-09-07T23:31Z"):
    cur, opn = {"target": {"value": cur_line}}, {"target": {"value": open_line if open_line is not None else cur_line}}
    if side:
        cur[side] = _price(price)
        opn[side] = _price(open_price if open_price is not None else price)
    return {"athlete": {"$ref": f"http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2025/athletes/{aid}?lang=en"},
            "type": {"id": "7", "name": type_name}, "current": cur, "open": opn, "lastUpdated": updated}


REC = "Total Receiving Yards (incl. overtime)"


def london_items():
    """Drake London's receiving yards as ESPN lists them: an unpriced ladder plus the priced main-line pair."""
    ladder = [item("4426502", REC, v) for v in (29.5, 49.5, 69.5, 99.5, 119.5)]
    main = [item("4426502", REC, 80.5, open_line=74.5, side="over", price=-120, open_price=-120),
            item("4426502", REC, 80.5, open_line=74.5, side="under", price=-110, open_price=-110)]
    return ladder + main


class TestMainLines(unittest.TestCase):
    def test_the_priced_pair_is_the_main_line_and_the_ladder_is_ignored(self):
        v = el.main_lines(london_items())[("4426502", "player_reception_yds")]
        self.assertEqual((v["open_line"], v["close_line"]), (74.5, 80.5))
        self.assertEqual((v["close_over"], v["close_under"]), (-120.0, -110.0))
        self.assertEqual(len(el.main_lines(london_items())), 1)                      # six items in, one prop out

    def test_an_unpriced_ladder_is_ambiguous_and_dropped(self):
        self.assertEqual(el.main_lines([item("1", REC, 49.5), item("1", REC, 69.5)]), {})

    def test_draftkings_shape_the_same_unpriced_line_listed_twice_is_the_main_line(self):
        # this season's provider lists the one line twice per player prop (over and under), with no prices
        two = [item("8439", "Total Passing Yards (incl. overtime)", 214.5, open_line=211.5) for _ in range(2)]
        v = el.main_lines(two)[("8439", "player_pass_yds")]
        self.assertEqual((v["open_line"], v["close_line"]), (211.5, 214.5))
        self.assertIsNone(v["close_over"])
        self.assertFalse(v["priced"])
        one = el.main_lines(two[:1])[("8439", "player_pass_yds")]                       # listed once is fine too
        self.assertEqual(one["close_line"], 214.5)
        self.assertTrue(el.main_lines(london_items())[("4426502", "player_reception_yds")]["priced"])

    def test_one_sided_listing_still_gives_the_line(self):
        v = el.main_lines([item("1", REC, 55.5, side="over", price=105)])[("1", "player_reception_yds")]
        self.assertEqual(v["close_line"], 55.5)
        self.assertEqual(v["close_over"], 105.0)                                       # "+105" parsed as a number
        self.assertIsNone(v["close_under"])

    def test_two_priced_lines_picks_the_one_nearest_a_coin_flip(self):
        items = [item("1", REC, 60.5, side="over", price=-110), item("1", REC, 60.5, side="under", price=-110),
                 item("1", REC, 90.5, side="over", price=+250), item("1", REC, 90.5, side="under", price=-400)]
        self.assertEqual(el.main_lines(items)[("1", "player_reception_yds")]["close_line"], 60.5)

    def test_market_names_map_and_unknown_types_are_skipped(self):
        for name, market in (("Total Passing Yards (incl. overtime)", "player_pass_yds"),
                             ("Longest Reception (incl. overtime)", "player_reception_longest"),
                             ("Longest Rush (incl. overtime)", "player_rush_longest"),
                             ("Total Carries (incl. overtime)", "player_rush_attempts"),
                             ("Total Rushing Plus Receiving Yards (incl. overtime)", "player_rush_reception_yds")):
            got = el.main_lines([item("1", name, 20.5, side="over"), item("1", name, 20.5, side="under")])
            self.assertEqual(list(got), [("1", market)])
        self.assertEqual(el.main_lines([item("1", "Anytime Touchdown Scorer", 0.5, side="over")]), {})
        self.assertEqual(el.main_lines([{"type": {"name": REC}, "current": {"target": {"value": 5}, "over": _price(-110)}}]), {})  # team prop, no athlete


class TestRows(unittest.TestCase):
    def test_rows_use_the_nflverse_id_and_skip_unmapped_or_inactive_players(self):
        mapping = el.espn_to_gsis(pd.DataFrame({"espn_id": [4426502.0, 999.0, None], "gsis_id": ["00-LONDON", "00-OTHER", "00-NOESPN"]}))
        self.assertEqual(mapping, {"4426502": "00-LONDON", "999": "00-OTHER"})
        lines = el.main_lines(london_items() + [item("999", REC, 40.5, side="over"), item("999", REC, 40.5, side="under"),
                                                item("555", REC, 40.5, side="over")])
        week = pd.DataFrame([{"player_id": "00-LONDON", "player_display_name": "Drake London", "team": "ATL", "week": 1}])
        game = {"season": 2025, "week": 1, "home": "ATL", "away": "TB", "commence": "2025-09-07T13:00:00", "espn": "401772830"}
        rows = el.rows_for_game(game, "58", lines, mapping, week)
        self.assertEqual(len(rows), 1)                                               # 999 has no row that week; 555 is unmapped
        r = rows[0]
        self.assertEqual((r["player_id"], r["player"], r["team"], r["opp"], r["market"]), ("00-LONDON", "Drake London", "ATL", "TB", "player_reception_yds"))
        self.assertEqual((r["open_line"], r["close_line"], r["provider"]), (74.5, 80.5, "58"))


class TestCollect(unittest.TestCase):
    def setUp(self):
        d = pathlib.Path(tempfile.mkdtemp())
        self.out, self.state = d / "espn.csv", d / "state.json"
        self.sched = pd.DataFrame([
            {"season": 2025, "week": 1, "game_type": "REG", "gameday": "2025-09-07", "gametime": "13:00", "home_team": "ATL",
             "away_team": "TB", "home_score": 20, "away_score": 23, "espn": 401772830.0},
            {"season": 2025, "week": 2, "game_type": "REG", "gameday": "2025-09-14", "gametime": "13:00", "home_team": "NO",
             "away_team": "SF", "home_score": 13, "away_score": 21, "espn": 401772831.0},
            {"season": 2025, "week": 3, "game_type": "REG", "gameday": "2025-09-21", "gametime": "13:00", "home_team": "GB",
             "away_team": "CHI", "home_score": None, "away_score": None, "espn": 401772832.0}])      # not played yet
        self.ids = pd.DataFrame({"espn_id": [4426502.0], "gsis_id": ["00-LONDON"]})
        self.hist = lambda s: pd.DataFrame([{"player_id": "00-LONDON", "player_display_name": "Drake London", "team": "ATL", "week": 1}])
        self.calls = []

    def fake_get(self, url, **kw):
        self.calls.append(url)
        m = __import__("re").search(r"events/(\d+)/competitions/\d+/odds/(\d+)/propBets\?limit=\d+&page=(\d+)", url)
        if not m:
            return None
        event, prov, page = m.group(1), m.group(2), int(m.group(3))
        if event == "401772830" and prov == "100":                                    # props live under the second provider
            return {"count": len(london_items()), "pageCount": 1, "items": london_items()}
        return {"count": 0, "pageCount": 0, "items": []}

    def go(self, **kw):
        return el.collect([2025], sched=self.sched, ids=self.ids, history_loader=self.hist, get=self.fake_get,
                          out_path=self.out, state_path=self.state, **kw)

    def test_finds_the_provider_that_has_props_and_writes_the_main_line(self):
        s = self.go()
        self.assertEqual(s["providers"], {"100": 1})                                  # tried 58 first, fell back to 100
        d = el.load(self.out)
        self.assertEqual(len(d), 1)
        self.assertEqual((d.iloc[0]["player"], d.iloc[0]["open_line"], d.iloc[0]["close_line"]), ("Drake London", 74.5, 80.5))

    def test_unplayed_games_are_skipped_and_games_without_props_are_not_retried(self):
        s = self.go()
        self.assertEqual(s["games"], 2)
        self.assertEqual(s["no_data"], 1)
        self.assertFalse(any("401772832" in u for u in self.calls))
        before = len(self.calls)
        s2 = self.go()                                                                 # resume: everything done
        self.assertEqual(s2["skipped_done"], 2)
        self.assertEqual(len(self.calls), before)
        self.assertEqual(len(el.load(self.out)), 1)                                    # nothing duplicated

    def test_closing_map_feeds_line_then(self):
        self.go()
        self.assertEqual(el.closing_map(el.load(self.out)), {("00-LONDON", "player_reception_yds"): {(2025, 1): 80.5}})
        self.assertEqual(el.closing_map(pd.DataFrame(columns=el.COLUMNS)), {})


class TestValidate(unittest.TestCase):
    def test_compares_with_our_own_logged_lines(self):
        espn = pd.DataFrame([
            {"season": 2026, "week": 3, "player_id": "a", "market": "player_pass_yds", "open_line": 240.5, "close_line": 250.5},
            {"season": 2026, "week": 3, "player_id": "b", "market": "player_receptions", "open_line": 4.5, "close_line": 5.5},
            {"season": 2026, "week": 3, "player_id": "c", "market": "player_rush_yds", "open_line": 60.5, "close_line": 61.5}])
        mine = pd.DataFrame([
            {"season": 2026, "week": 3, "player_id": "a", "market": "player_pass_yds", "pos": "QB", "line": 250.5, "opening_line": 245.5},
            {"season": 2026, "week": 3, "player_id": "b", "market": "player_receptions", "pos": "WR", "line": 5.0, "opening_line": 4.5},
            {"season": 2026, "week": 3, "player_id": "GAME_X", "market": "game_total", "pos": "GAME", "line": 45.5, "opening_line": 45.5}])
        v = el.validate(espn, mine)
        self.assertEqual(v["n"], 2)                                                    # c isn't in our log; the game row is excluded
        self.assertAlmostEqual(v["exact_close"], 0.5)
        self.assertAlmostEqual(v["mean_abs_diff"], 0.25)
        self.assertAlmostEqual(v["within_half_point"], 1.0)
        self.assertEqual(el.validate(espn.iloc[0:0], mine), {"n": 0})


class TestDedupeAnd404(unittest.TestCase):
    def test_dedupe_removes_a_games_rows_written_twice_and_keeps_the_rest(self):
        path = pathlib.Path(tempfile.mkdtemp()) / "e.csv"
        row = {c: None for c in el.COLUMNS}
        row.update({"season": 2025, "week": 1, "player_id": "p", "market": "player_pass_yds", "close_line": 250.5})
        other = dict(row, player_id="q")
        el.append_rows([row, other, row], path)
        self.assertEqual(el.dedupe(path), 1)
        self.assertEqual(len(el.load(path)), 2)
        self.assertEqual(el.dedupe(path), 0)

    def test_a_404_is_not_retried(self):
        import urllib.error
        from unittest import mock
        calls = []

        def boom(req, timeout=None):
            calls.append(1)
            raise urllib.error.HTTPError("u", 404, "Not Found", {}, None)
        with mock.patch("urllib.request.urlopen", boom), mock.patch("time.sleep"):
            self.assertIsNone(el.http_get("https://example.invalid/x", cache=False))
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
