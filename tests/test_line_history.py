"""
Tests for the backlog: line_history.py (append-only pull history, closing lines) and the weekly
power-ranking history in team_ratings.py.

Run:  python -m unittest discover -s tests -v
"""

import pathlib
import tempfile
import unittest

import pandas as pd

import collect_lines as cl
import line_history as lh
import team_ratings as tr
from test_ratings import TEAMS, _league


def _snapshot():
    prop = {"player_id": "p1", "player": "Test Back", "team": "AAA", "opp": "BBB", "game": "BBB @ AAA",
            "commence": "2026-10-11T17:00:00Z", "market": "player_rush_yds", "line": 60.5, "market_over": 0.51,
            "best_over": {"price": -105, "book": "SECRET BOOK"}, "best_under": {"price": -115, "book": "OTHER BOOK"},
            "books": [{"book": "x"}, {"book": "y"}, {"book": "z"}], "mean": 64.2, "model_over": 0.55, "injury": None}
    game = {"game": "BBB @ AAA", "home": "AAA", "away": "BBB", "commence": "2026-10-11T17:00:00Z", "week": 6,
            "home_spread": -3.5, "total": 44.5, "fair_home_cover": 0.52, "fair_over": 0.49,
            "best_home": {"price": -108, "book": "H"}, "best_away": {"price": -112, "book": "A"},
            "best_over": {"price": -110, "book": "O"}, "best_under": {"price": -110, "book": "U"},
            "proj_margin": 4.1, "proj_total": 46.0, "model_home_cover": 0.55, "model_over": 0.57}
    return {"season": 2026, "week": 5, "props": [prop], "games": [game]}


class TestRows(unittest.TestCase):
    def test_prop_and_game_rows_from_a_snapshot(self):
        rows = lh.rows_from_snapshot(_snapshot(), "2026-10-09T12:00:00+00:00")
        self.assertEqual(len(rows), 3)                                         # 1 prop + game spread + game total
        prop = [r for r in rows if r["kind"] == "prop"][0]
        self.assertEqual((prop["line"], prop["fair_over"], prop["best_over"], prop["best_under"], prop["n_books"]),
                         (60.5, 0.51, -105, -115, 3))
        self.assertEqual(prop["mean"], 64.2)
        spread = [r for r in rows if r["market"] == "game_spread"][0]
        self.assertEqual(spread["week"], 6)                                    # the game's own week, not the snapshot's
        self.assertEqual((spread["line"], spread["best_over"], spread["best_under"]), (-3.5, -108, -112))   # over == home
        total = [r for r in rows if r["market"] == "game_total"][0]
        self.assertEqual((total["line"], total["fair_over"], total["mean"]), (44.5, 0.49, 46.0))

    def test_no_book_names_are_ever_stored(self):
        for r in lh.rows_from_snapshot(_snapshot()):
            self.assertNotIn("book", "".join(r.keys()).replace("n_books", ""))
            self.assertNotIn("SECRET BOOK", str(list(r.values())))

    def test_game_line_rows_without_a_model(self):
        events = [{"id": "e1", "home_team": "Home Team", "away_team": "Away Team", "commence_time": "2026-10-11T17:00:00Z"},
                  {"id": "e2", "home_team": "Home Team", "away_team": "Away Team", "commence_time": "2026-10-18T17:00:00Z"}]
        lines = {"e1": {"total": 45.5, "home_spread": -2.5}, "e2": {"total": None, "home_spread": None}}
        prices = {"e1": {"fair_home_cover": 0.5, "fair_over": 0.5, "best_home": {"price": -110}, "best_away": {"price": -110},
                         "best_over": {"price": -105}, "best_under": {"price": -115}}}
        rows = lh.rows_from_game_lines(events, lines, prices, lambda h, a, c: 5, 2026, "2026-10-09T13:00:00+00:00",
                                       abbr={"Home Team": "HOM", "Away Team": "AWY"})
        self.assertEqual(len(rows), 2)                                         # e2 has no line: skipped, not logged as blank
        self.assertEqual({r["player_id"] for r in rows}, {"GAME_HOM_AWY_spread", "GAME_HOM_AWY_total"})
        self.assertEqual([r["week"] for r in rows], [5, 5])
        self.assertTrue(all(r["mean"] is None and r["model_over"] is None for r in rows))


class TestAppendOnly(unittest.TestCase):
    def setUp(self):
        self.path = pathlib.Path(tempfile.mkdtemp()) / "hist.csv"

    def test_appends_never_overwrite_and_header_is_written_once(self):
        rows = lh.rows_from_snapshot(_snapshot(), "2026-10-09T12:00:00+00:00")
        self.assertEqual(lh.append_rows(rows, self.path), 3)
        later = lh.rows_from_snapshot(_snapshot(), "2026-10-10T12:00:00+00:00")
        later[0]["line"] = 62.5                                                # the line moved
        lh.append_rows(later, self.path)
        text = self.path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(text), 1 + 6)
        self.assertEqual(sum(l.startswith("pulled_at") for l in text), 1)
        df = lh.load(self.path)
        self.assertEqual(sorted(df[df["kind"] == "prop"]["line"]), [60.5, 62.5])   # both pulls survive

    def test_empty_inputs_are_safe(self):
        self.assertEqual(lh.append_rows([], self.path), 0)
        self.assertFalse(self.path.exists())
        self.assertTrue(lh.load(self.path).empty)
        self.assertEqual(lh.summary(lh.load(self.path)), {"rows": 0, "pulls": 0})

    def test_seed_from_log_is_idempotent(self):
        log = pd.DataFrame([{"logged_at": "2026-10-01T10:00:00+00:00", "season": 2026, "week": 4, "player_id": "p1",
                             "market": "player_rush_yds", "player": "A", "team": "AAA", "opp": "BBB", "pos": "RB",
                             "game": "BBB @ AAA", "commence": "2026-10-04T17:00:00Z", "line": 55.5, "market_over": 0.5,
                             "model_over": 0.52}])
        self.assertEqual(lh.seed_from_log(log, self.path), 1)
        self.assertEqual(lh.seed_from_log(log, self.path), 0)
        self.assertEqual(len(lh.load(self.path)), 1)


class TestClosingLines(unittest.TestCase):
    def _hist(self):
        base = {"season": 2026, "week": 5, "kind": "prop", "market": "player_rush_yds", "player_id": "p1",
                "commence": "2026-10-11T17:00:00Z", "fair_over": 0.5}
        pulls = [("2026-10-06T13:00:00+00:00", 60.5, 0.50), ("2026-10-09T13:00:00+00:00", 62.5, 0.48),
                 ("2026-10-11T16:00:00+00:00", 64.5, 0.46), ("2026-10-11T18:30:00+00:00", 99.5, 0.10)]   # last is LIVE
        return pd.DataFrame([dict(base, pulled_at=t, line=l, fair_over=f) for t, l, f in pulls])

    def test_open_close_and_move(self):
        c = lh.closing_lines(self._hist()).iloc[0]
        self.assertEqual((c["opening_line"], c["closing_line"], c["n_pulls"], c["move"]), (60.5, 64.5, 3, 4.0))
        self.assertEqual((c["open_fair"], c["close_fair"]), (0.5, 0.46))

    def test_pulls_after_kickoff_are_never_the_close(self):
        c = lh.closing_lines(self._hist()).iloc[0]
        self.assertNotEqual(c["closing_line"], 99.5)

    def test_separate_props_do_not_mix(self):
        h = pd.concat([self._hist(), self._hist().assign(player_id="p2", line=10.5)], ignore_index=True)
        c = lh.closing_lines(h).set_index("player_id")
        self.assertEqual(c.loc["p2", "opening_line"], 10.5)
        self.assertEqual(c.loc["p1", "opening_line"], 60.5)

    def test_empty(self):
        self.assertTrue(lh.closing_lines(pd.DataFrame(columns=lh.COLUMNS)).empty)

    def test_summary_counts(self):
        s = lh.summary(self._hist())
        self.assertEqual((s["rows"], s["pulls"], s["props"], s["games"]), (4, 4, 1, 0))


class TestPowerHistory(unittest.TestCase):
    def setUp(self):
        self.path = pathlib.Path(tempfile.mkdtemp()) / "power.csv"
        self.games = _league({t: float(i) - 16 for i, t in enumerate(TEAMS)}, seasons=(2025, 2026), weeks=6)

    def test_snapshot_stores_one_row_per_team_and_is_replaced_not_duplicated(self):
        self.assertEqual(tr.snapshot_power_history(self.games, 2026, "t1", self.path), 32)
        self.assertEqual(tr.snapshot_power_history(self.games, 2026, "t2", self.path), 32)      # same week re-run
        d = tr.load_power_history(self.path)
        self.assertEqual(len(d), 32)
        self.assertEqual(set(d["computed_at"]), {"t2"})                                         # latest wins
        self.assertEqual(set(d["through_week"]), {6})
        self.assertEqual(sorted(d["rank"]), list(range(1, 33)))

    def test_a_new_week_adds_rows_without_touching_earlier_weeks(self):
        tr.snapshot_power_history(self.games, 2026, "t1", self.path)
        earlier = self.games[~((self.games["season"] == 2026) & (self.games["week"] == 6))]
        # pretend week 6 hasn't been played yet: the stored rows for week 5 must survive a later week-6 snapshot
        tr.upsert_power_history(tr.power_rows(tr.power_table(earlier, 2026, None), 2026, 5, "t0"), self.path)
        d = tr.load_power_history(self.path)
        self.assertEqual(sorted(d["through_week"].unique()), [5, 6])
        self.assertEqual(len(d), 64)

    def test_backfill_covers_every_completed_week_with_only_earlier_games(self):
        n = tr.backfill_power_history(self.games, 2026, "bf", self.path)
        self.assertEqual(n, 6 * 32)
        d = tr.load_power_history(self.path)
        self.assertEqual(sorted(d["through_week"].unique()), [1, 2, 3, 4, 5, 6])
        # a week-3 ranking must not change if week 4+ results are rewritten (no future leakage)
        wk3 = d[d["through_week"] == 3].set_index("team")["rating"]
        tampered = self.games.copy()
        tampered.loc[(tampered["season"] == 2026) & (tampered["week"] >= 4), "home_score"] += 80
        other = pathlib.Path(tempfile.mkdtemp()) / "p2.csv"
        tr.backfill_power_history(tampered, 2026, "bf", other)
        wk3b = tr.load_power_history(other)
        wk3b = wk3b[wk3b["through_week"] == 3].set_index("team")["rating"]
        pd.testing.assert_series_equal(wk3.sort_index(), wk3b.sort_index())

    def test_first_season_filter(self):
        n = tr.backfill_power_history(self.games, 2026, "bf", self.path)
        self.assertEqual(set(tr.load_power_history(self.path)["season"]), {2026})


def _raw_event(eid, home, away, commence, spread=-3.0, total=45.0):
    """An Odds API bulk game-lines event with two books, in the real response shape."""
    def book(title, hp, ap, op, up):
        return {"title": title, "key": title.lower(), "markets": [
            {"key": "spreads", "outcomes": [{"name": home, "point": spread, "price": hp},
                                            {"name": away, "point": -spread, "price": ap}]},
            {"key": "totals", "outcomes": [{"name": "Over", "point": total, "price": op},
                                           {"name": "Under", "point": total, "price": up}]}]}
    return {"id": eid, "home_team": home, "away_team": away, "commence_time": commence,
            "bookmakers": [book("BookA", -110, -110, -105, -115), book("BookB", -108, -112, -110, -110)]}


class TestCollector(unittest.TestCase):
    """collect_lines.store_game_lines on a realistic response (this is the path the off-day workflow runs)."""
    SCHED = pd.DataFrame([
        {"season": 2026, "week": 6, "gameday": "2099-01-01", "home_team": "NE", "away_team": "BUF"},
        {"season": 2026, "week": 5, "gameday": "2000-01-01", "home_team": "NO", "away_team": "ATL"}])

    def setUp(self):
        self.path = pathlib.Path(tempfile.mkdtemp()) / "h.csv"

    def test_stores_spread_and_total_rows_with_the_games_own_week(self):
        raw = [_raw_event("e1", "New England Patriots", "Buffalo Bills", "2099-01-01T17:00:00Z")]
        n = cl.store_game_lines(raw, self.SCHED, 2026, 5, "2026-10-09T13:00:00+00:00", self.path)
        self.assertEqual(n, 2)
        d = lh.load(self.path).set_index("market")
        self.assertEqual(d.loc["game_spread", "line"], -3.0)
        self.assertEqual(d.loc["game_total", "line"], 45.0)
        self.assertEqual(set(d["week"]), {6})                                  # from the schedule, not the default 5
        self.assertEqual(d.loc["game_spread", "best_over"], -108)              # best home price
        self.assertTrue(0.45 < d.loc["game_total", "fair_over"] < 0.55)
        self.assertNotIn("BookA", self.path.read_text(encoding="utf-8"))       # no book names on disk

    def test_games_already_in_progress_are_skipped(self):
        raw = [_raw_event("live", "New Orleans Saints", "Atlanta Falcons", "2000-01-01T17:00:00Z"),
               _raw_event("e1", "New England Patriots", "Buffalo Bills", "2099-01-01T17:00:00Z")]
        self.assertEqual(cl.store_game_lines(raw, self.SCHED, 2026, 5, "t", self.path), 2)
        self.assertEqual(set(lh.load(self.path)["game"]), {"BUF @ NE"})

    def test_empty_response_stores_nothing(self):
        self.assertEqual(cl.store_game_lines([], self.SCHED, 2026, 5, "t", self.path), 0)
        self.assertFalse(self.path.exists())

    def test_two_pulls_build_a_trail_with_a_move(self):
        a = [_raw_event("e1", "New England Patriots", "Buffalo Bills", "2099-01-01T17:00:00Z", spread=-3.0)]
        b = [_raw_event("e1", "New England Patriots", "Buffalo Bills", "2099-01-01T17:00:00Z", spread=-4.5)]
        cl.store_game_lines(a, self.SCHED, 2026, 5, "2026-10-09T13:00:00+00:00", self.path)
        cl.store_game_lines(b, self.SCHED, 2026, 5, "2026-10-10T13:00:00+00:00", self.path)
        c = lh.closing_lines(lh.load(self.path))
        sp = c[c["market"] == "game_spread"].iloc[0]
        self.assertEqual((sp["opening_line"], sp["closing_line"], sp["n_pulls"], sp["move"]), (-3.0, -4.5, 2, -1.5))


if __name__ == "__main__":
    unittest.main()
