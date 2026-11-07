"""
Tests for free_picks.py: the three source parsers (against page structures like the live ones), projections becoming picks only past an edge, each source
isolated from the others' failures, the latest-call view, and the simulation-vs-majority tagging.

Run:  python -m unittest discover -s tests -v
"""

import json
import pathlib
import tempfile
import unittest

import numpy as np
import pandas as pd

import free_picks as fp
import social_picks as sp
from tests.test_social_picks import GAMES

NOW = pd.Timestamp("2026-11-07T15:00:00Z")
DET_MIA = GAMES[GAMES["game_id"] == "2026_09_DET_MIA"].iloc[0]                       # DET -3 at MIA, total 47.5


def _ps(name, slug, tier, away, home, when, pick):
    return (f'<div><a class="x" href="/handicappers/{slug}/"><img src="/a.png"/><span class="n">{name}</span></a>'
            f'<span>{tier}</span><div>{away}</div><div>{home}</div><span>NFL</span><span>{when}</span><div>{pick}</div>'
            f'<span>Point Spread Pick</span><p>Read more</p></div>')


class TestModelPicks(unittest.TestCase):
    def test_a_pick_needs_an_edge(self):
        # DET -3 at MIA: home_spread is +3 (MIA is the dog). A predicted home margin of -1 means MIA covers by 2.
        p = fp.model_picks("m", DET_MIA, -1.0, 3.0)
        self.assertEqual([(x["market"], x["side"], x["line"]) for x in p], [("spread", "MIA", 3.0)])
        p = fp.model_picks("m", DET_MIA, -6.5, 3.0)                                      # the model has DET by 6.5: DET covers -3
        self.assertEqual([(x["market"], x["side"], x["line"]) for x in p], [("spread", "DET", -3.0)])
        self.assertEqual(fp.model_picks("m", DET_MIA, -3.5, 3.0), [])                    # half a point of edge is not a pick

    def test_total_picks(self):
        over = fp.model_picks("m", DET_MIA, -3.0, 3.0, 50.0, 47.5)
        under = fp.model_picks("m", DET_MIA, -3.0, 3.0, 45.0, 47.5)
        self.assertEqual([(x["market"], x["side"]) for x in over], [("total", "over")])
        self.assertEqual([(x["market"], x["side"]) for x in under], [("total", "under")])
        self.assertEqual(fp.model_picks("m", DET_MIA, -3.0, 3.0, 47.0, 47.5), [])

    def test_picks_carry_the_game_and_source(self):
        p = fp.model_picks("espn-fpi", DET_MIA, -1.0, 3.0)[0]
        self.assertEqual((p["source"], p["platform"], p["game_id"], p["game"]), ("espn-fpi", "web", "2026_09_DET_MIA", "DET @ MIA"))


class TestCovers(unittest.TestCase):
    PAGE = ("<div>Computer Picks Trends Predicted Score DET DET &#x2BC6; 23.00 @ 22.00 MIA MIA Lions vs Dolphins &middot; Prediction &amp; Game Odds "
            "Team Predicted Score Spread Total ML DET DET 23.00 -3.0 50% Kalshi Sports o47.5 52% Novig 60% Novig "
            "MIA MIA 22.00 +3.0 51% Kalshi Sports u47.5 52% Novig 40% Novig Lions are 4-0 ATS Computer Picks Trends</div>")

    def test_the_computer_pick_becomes_a_spread_and_a_total_pick(self):
        p = fp.parse_covers(self.PAGE, GAMES, NOW)
        got = {(x["market"], x["side"], x["line"]) for x in p}
        self.assertEqual(got, {("spread", "MIA", 3.0), ("total", "under", 47.5)})          # DET by 1 vs -3 and 45 vs 47.5: both are 1+ point picks
        self.assertTrue(all(x["source"] == "covers-computer" for x in p))

    def test_a_close_call_is_not_a_pick(self):
        page = self.PAGE.replace("23.00", "25.00").replace("22.00", "22.50")            # DET by 2.5 vs -3 is half a point; 47.5 vs 47.5
        self.assertEqual(fp.parse_covers(page, GAMES, NOW), [])

    def test_an_unknown_page_gives_nothing(self):
        self.assertEqual(fp.parse_covers("<html>nothing</html>", GAMES, NOW), [])

    def test_a_game_that_is_not_on_the_schedule_is_skipped(self):
        self.assertEqual(fp.parse_covers(self.PAGE.replace("DET DET", "SEA SEA"), GAMES, NOW), [])


class TestPickswise(unittest.TestCase):
    def test_spread_and_total_by_handicapper(self):
        page = (_ps("Aaron Taylor", "aaron-taylor", "Max", "Detroit Lions", "Miami Dolphins", "Sun, Nov 8 1:00 PM", "Spread - Detroit Lions -3 -3 -110")
                + _ps("Aaron Taylor", "aaron-taylor", "Medium", "Detroit Lions", "Miami Dolphins", "Sun, Nov 8 1:00 PM", "Over 48.5 o47.5 -108"))
        p = fp.parse_pickswise(page, GAMES, NOW)
        self.assertEqual({(x["source"], x["market"], x["side"], x["line"]) for x in p},
                         {("pickswise:aaron-taylor", "spread", "DET", -3.0), ("pickswise:aaron-taylor", "total", "over", 48.5)})

    def test_the_underdog_side_and_a_plus_line(self):
        page = _ps("Kevin Rogers", "kevin-rogers", "Fair", "New England Patriots", "Green Bay Packers", "Sun, Nov 8 4:25 PM", "Spread - New England Patriots +6.5 +6.5 -110")
        p = fp.parse_pickswise(page, GAMES, NOW)
        self.assertEqual([(x["side"], x["line"]) for x in p], [("NE", 6.5)])

    def test_props_and_articles_are_not_game_picks(self):
        prop = _ps("Sam", "sam", "Max", "Detroit Lions", "Miami Dolphins", "Sun, Nov 8 1:00 PM", "Jahmyr Gibbs Over 55.5 Rushing Yards")
        article = '<a href="/handicappers/andrew-wilsher/"><span>Andrew Wilsher</span></a> NFL Nov 6, 2026 NFL Week 9 Best Bets: Top Picks Read more'
        self.assertEqual(fp.parse_pickswise(prop + article, GAMES, NOW), [])

    def test_a_handicapper_listed_twice_counts_once_per_market(self):
        card = _ps("Aaron Taylor", "aaron-taylor", "Max", "Detroit Lions", "Miami Dolphins", "Sun, Nov 8 1:00 PM", "Spread - Detroit Lions -3 -3 -110")
        self.assertEqual(len(fp.parse_pickswise(card + card, GAMES, NOW)), 1)

    def test_a_number_far_from_the_market_is_dropped(self):
        card = _ps("Aaron Taylor", "aaron-taylor", "Max", "Detroit Lions", "Miami Dolphins", "Sun, Nov 8 1:00 PM", "Spread - Detroit Lions -12 -3 -110")
        self.assertEqual(fp.parse_pickswise(card, GAMES, NOW), [])

    def test_a_game_after_the_window_is_ignored(self):
        card = _ps("Aaron Taylor", "aaron-taylor", "Max", "Detroit Lions", "Miami Dolphins", "Sun, Nov 8 1:00 PM", "Spread - Detroit Lions -3 -3 -110")
        self.assertEqual(fp.parse_pickswise(card, GAMES, pd.Timestamp("2026-10-20T00:00:00Z")), [])


class TestESPN(unittest.TestCase):
    def _get(self, margin):
        sb = {"events": [{"id": "1", "status": {"type": {"state": "pre"}}, "competitions": [{"competitors": [
            {"homeAway": "home", "team": {"abbreviation": "MIA"}}, {"homeAway": "away", "team": {"abbreviation": "DET"}}]}]},
            {"id": "2", "status": {"type": {"state": "post"}}, "competitions": [{"competitors": [
                {"homeAway": "home", "team": {"abbreviation": "BAL"}}, {"homeAway": "away", "team": {"abbreviation": "JAX"}}]}]}]}
        pred = {"homeTeam": {"statistics": [{"name": "teamPredPtDiff", "value": margin}]}}

        def get(url):
            return json.dumps(pred if "predictor" in url else sb)
        return get

    def test_fpi_margin_against_the_line(self):
        p = fp.fetch_espn_fpi(GAMES, NOW, self._get(-1.0))                       # MIA by -1 with +3: MIA covers
        self.assertEqual([(x["source"], x["market"], x["side"], x["line"]) for x in p], [("espn-fpi", "spread", "MIA", 3.0)])

    def test_finished_games_and_no_edge_give_nothing(self):
        self.assertEqual(fp.fetch_espn_fpi(GAMES, NOW, self._get(-3.2)), [])

    def test_espn_abbreviations_map_to_our_codes(self):
        codes = {"LA", "WAS", "DET"}
        self.assertEqual(fp.norm_code("WSH", codes), "WAS")
        self.assertEqual(fp.norm_code("LAR", codes), "LA")
        self.assertEqual(fp.norm_code("DET", codes), "DET")
        self.assertIsNone(fp.norm_code("XYZ", codes))


class TestRun(unittest.TestCase):
    def test_one_failing_source_does_not_take_the_others_down(self):
        def good(games, now, get):
            return fp.model_picks("good", DET_MIA, -1.0, 3.0)

        def bad(games, now, get):
            raise ValueError("page changed")
        old, fp.POLITE_DELAY = fp.POLITE_DELAY, 0
        try:
            r = fp.fetch_all(GAMES, NOW, sources={"good": good, "bad": bad, "empty": lambda g, n, t: []})
        finally:
            fp.POLITE_DELAY = old
        self.assertEqual(len(r["picks"]), 1)
        self.assertTrue(r["sources"]["good"]["ok"])
        self.assertFalse(r["sources"]["bad"]["ok"])
        self.assertIn("page changed", r["sources"]["bad"]["note"])
        self.assertTrue(r["sources"]["empty"]["ok"])
        self.assertIn("no picks", r["sources"]["empty"]["note"])

    def test_log_rows_are_the_social_log_shape_and_grade_like_any_pick(self):
        picks = fp.model_picks("espn-fpi", DET_MIA, -1.0, 3.0)
        rows = fp.rows_for_log(picks, NOW)
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "l.csv"
            self.assertEqual(sp.log_picks(rows, path), 1)
            self.assertEqual(sp.log_picks(rows, path), 0)                                  # the next run's identical pick is not a second one
            df = sp.load(path)
            self.assertEqual((df.iloc[0]["platform"], df.iloc[0]["source"]), ("web", "espn-fpi"))
            games = GAMES.copy()
            games.loc[games["game_id"] == "2026_09_DET_MIA", ["home_score", "away_score"]] = [20.0, 21.0]     # DET won by 1: MIA +3 covers
            self.assertEqual(sp.grade(games, path), 1)
            self.assertEqual(sp.load(path).iloc[0]["result"], "win")

    def test_latest_round_trips_with_missing_values(self):
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "latest.json"
            fp.save_latest({"generated": "x", "sources": {}, "picks": [{"source": "a", "line": float("nan"), "odds": None}]}, path)
            self.assertEqual(fp.load_latest(path)["picks"][0]["source"], "a")
            self.assertEqual(fp.load_latest(pathlib.Path(d) / "missing.json")["picks"], [])


def _row(source, side, market="spread", line=-3.0, game="DET @ MIA", gid="2026_09_DET_MIA", kick="2026-11-08T18:00:00+00:00", result=np.nan):
    return {"source": source, "platform": "web", "post_id": "p", "posted_at": "2026-11-07T10:00:00+00:00", "season": 2026, "week": 9, "game_id": gid,
            "game": game, "kickoff": kick, "market": market, "side": side, "line": line, "odds": np.nan, "text": "", "result": result,
            "profit": np.nan, "graded_at": np.nan}


class TestAgreement(unittest.TestCase):
    def eff(self, rows, latest=None, log=None):
        df = pd.DataFrame(log if log is not None else [], columns=sp.COLUMNS)
        return fp.upcoming(df, {"picks": rows if latest is None else latest}, 2026, now="2026-11-07T15:00:00Z")

    def test_the_latest_call_replaces_a_sources_first_one(self):
        first = [_row("covers", "DET")]
        eff = fp.upcoming(pd.DataFrame(first, columns=sp.COLUMNS), {"picks": [_row("covers", "MIA", line=3.0)]}, 2026, now="2026-11-07T15:00:00Z")
        self.assertEqual(eff[["source", "side"]].values.tolist(), [["covers", "MIA"]])

    def test_picks_on_started_games_are_dropped_and_x_picks_stay(self):
        log = [_row("handle", "DET"), _row("old", "DET", kick="2026-11-01T18:00:00+00:00")]
        eff = fp.upcoming(pd.DataFrame(log, columns=sp.COLUMNS), {"picks": []}, 2026, now="2026-11-07T15:00:00Z")
        self.assertEqual(eff["source"].tolist(), ["handle"])

    def test_graded_picks_are_not_upcoming(self):
        eff = fp.upcoming(pd.DataFrame([_row("a", "DET", result="win")], columns=sp.COLUMNS), {"picks": []}, 2026, now="2026-11-07T15:00:00Z")
        self.assertTrue(eff.empty)

    def entries(self, side="DET", total="over"):
        return [{"game": "DET @ MIA", "type": "Spread", "side": side, "pick": f"{side} -3", "prob": 0.53, "rank": 2},
                {"game": "DET @ MIA", "type": "Total", "side": total, "pick": f"{total.title()} 47.5", "prob": 0.52, "rank": 5}]

    def test_a_majority_backs_the_pick(self):
        rows = [_row(s, "DET") for s in ("a", "b", "c", "d")] + [_row("e", "MIA", line=3.0)]
        e = self.entries()
        n = fp.annotate(e, self.eff(rows))
        self.assertEqual(n, 1)
        f = e[0]["free"]
        self.assertEqual((f["agree"], f["against"], f["of"], f["backed"], f["majority_side"]), (4, 1, 5, True, "DET"))
        self.assertEqual(f["sources"], ["a", "b", "c", "d"])
        self.assertEqual(e[1]["free"]["of"], 0)                                           # nobody has a total pick on this game
        self.assertFalse(e[1]["free"]["backed"])

    def test_a_majority_against_is_not_backed(self):
        rows = [_row(s, "MIA", line=3.0) for s in ("a", "b", "c")] + [_row("d", "DET")]
        e = self.entries()
        fp.annotate(e, self.eff(rows))
        self.assertFalse(e[0]["free"]["backed"])
        self.assertEqual(e[0]["free"]["majority_side"], "MIA")

    def test_two_picks_are_too_few_to_be_a_majority(self):
        e = self.entries()
        fp.annotate(e, self.eff([_row("a", "DET"), _row("b", "DET")]))
        self.assertFalse(e[0]["free"]["backed"])

    def test_an_even_split_is_not_a_majority(self):
        e = self.entries()
        fp.annotate(e, self.eff([_row("a", "DET"), _row("b", "DET"), _row("c", "MIA", line=3.0), _row("d", "MIA", line=3.0)]))
        self.assertFalse(e[0]["free"]["backed"])
        self.assertIsNone(e[0]["free"]["majority_side"])

    def test_totals_are_matched_by_side(self):
        rows = [_row(s, "over", market="total", line=47.5) for s in ("a", "b", "c")]
        e = self.entries(total="over")
        fp.annotate(e, self.eff(rows))
        self.assertTrue(e[1]["free"]["backed"])

    def test_the_block_lists_backed_picks_best_confidence_first(self):
        rows = [_row(s, "DET") for s in ("a", "b", "c")] + [_row(s, "over", market="total", line=47.5) for s in ("a", "b", "c")]
        e = self.entries()
        fp.annotate(e, self.eff(rows))
        b = fp.block(e, {"generated": "g", "sources": {"covers": {"ok": True, "picks": 3, "note": None}}})
        self.assertEqual([x["type"] for x in b["backed"]], ["Spread", "Total"])
        json.dumps(b)


if __name__ == "__main__":
    unittest.main()
