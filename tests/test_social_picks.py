"""
Tests for social_picks.py: reading picks out of posts, the rules for what counts, grading, the shrunk record and weights, the consensus, the
check that a good record predicts anything, and the X client (against a fake API).

Run:  python -m unittest discover -s tests -v
"""

import datetime
import json
import os
import pathlib
import tempfile
import unittest

import numpy as np
import pandas as pd

import social_picks as sp


def _sched(rows=None):
    base = [
        # (game_id, week, gameday, gametime ET, away, home, spread_line (home favored +), total, away_score, home_score)
        ("2026_09_DET_MIA", 9, "2026-11-08", "13:00", "DET", "MIA", -3.0, 47.5, np.nan, np.nan),
        ("2026_09_NE_GB", 9, "2026-11-08", "16:25", "NE", "GB", 6.5, 43.5, np.nan, np.nan),
        ("2026_09_CLE_NO", 9, "2026-11-08", "13:00", "CLE", "NO", 2.5, 41.0, np.nan, np.nan),
        ("2026_09_JAX_BAL", 9, "2026-11-05", "20:15", "JAX", "BAL", 4.0, 45.5, 10.0, 17.0),          # Thursday, already played
    ]
    rows = rows or base
    return pd.DataFrame([{"game_id": r[0], "season": 2026, "week": r[1], "gameday": r[2], "gametime": r[3], "away_team": r[4], "home_team": r[5],
                          "spread_line": r[6], "total_line": r[7], "away_score": r[8], "home_score": r[9], "game_type": "REG"} for r in rows])


GAMES = sp.games_from_schedule(_sched())
SAT = "2026-11-07T15:00:00Z"                                                       # Saturday morning, before every Sunday game


def parse(text, when=SAT, games=GAMES):
    return sp.parse_picks(text, when, games)


class TestTime(unittest.TestCase):
    def test_et_offset_changes_on_the_first_sunday_of_november(self):
        self.assertEqual(sp.et_offset(datetime.date(2026, 10, 31)), -4)
        self.assertEqual(sp.et_offset(datetime.date(2026, 11, 1)), -5)
        self.assertEqual(sp.et_offset(datetime.date(2026, 9, 13)), -4)

    def test_kickoffs_are_utc(self):
        k = GAMES.set_index("game_id")["kickoff"]
        self.assertEqual(k["2026_09_DET_MIA"], pd.Timestamp("2026-11-08T18:00:00Z"))        # 1:00 pm EST
        self.assertEqual(k["2026_09_JAX_BAL"], pd.Timestamp("2026-11-06T01:15:00Z"))        # 8:15 pm EST on Nov 5

    def test_a_timestamp_with_no_zone_is_eastern(self):
        self.assertEqual(sp.to_utc("2026-11-07 10:00"), pd.Timestamp("2026-11-07T15:00:00Z"))
        self.assertEqual(sp.to_utc("2026-11-07T10:00:00-05:00"), pd.Timestamp("2026-11-07T15:00:00Z"))


class TestParsing(unittest.TestCase):
    def test_spread_by_nickname(self):
        p = parse("Lions -3 \U0001f512 love this")
        self.assertEqual(len(p), 1)
        self.assertEqual((p[0]["market"], p[0]["side"], p[0]["line"], p[0]["game_id"]), ("spread", "DET", -3.0, "2026_09_DET_MIA"))
        self.assertIsNone(p[0]["odds"])

    def test_the_odds_after_a_spread_are_read(self):
        p = parse("Patriots +6.5 (-105)")
        self.assertEqual((p[0]["side"], p[0]["line"], p[0]["odds"]), ("NE", 6.5, -105))

    def test_unicode_minus_and_pick_em(self):
        self.assertEqual(parse("Lions −3")[0]["line"], -3.0)
        self.assertEqual(parse("Dolphins PK")[0]["line"], 0.0)

    def test_abbreviations_next_to_a_number(self):
        p = parse("DET -3 and NE +6.5")
        self.assertEqual({(x["side"], x["line"]) for x in p}, {("DET", -3.0), ("NE", 6.5)})

    def test_a_capital_word_that_is_a_team_code_is_not_a_pick(self):
        self.assertEqual(parse("NO LOCK today, no bet"), [])
        self.assertEqual(parse("WAS this a fade? TEN more minutes"), [])

    def test_a_moneyline_with_and_without_a_price(self):
        a, b = parse("Dolphins ML +130")[0], parse("Packers moneyline")[0]
        self.assertEqual((a["market"], a["side"], a["odds"], a["line"]), ("moneyline", "MIA", 130, None))
        self.assertEqual((b["side"], b["odds"]), ("GB", None))

    def test_a_ml_price_is_not_read_as_a_spread(self):
        p = parse("Dolphins ML -150")
        self.assertEqual([(x["market"], x["odds"]) for x in p], [("moneyline", -150)])
        self.assertEqual(parse("Lions -110"), [])

    def test_totals_need_a_game(self):
        p = parse("Lions/Dolphins over 47.5")
        self.assertEqual([(x["market"], x["side"], x["line"], x["game_id"]) for x in p], [("total", "over", 47.5, "2026_09_DET_MIA")])
        self.assertEqual(parse("Over 47.5"), [])                                              # no game named
        self.assertEqual(parse("Lions/Packers over 47.5"), [])                                # not a real matchup

    def test_a_spread_and_total_in_one_post(self):
        p = parse("Lions -3, Under 47.5")
        self.assertEqual({(x["market"], x["side"]) for x in p}, {("spread", "DET"), ("total", "under")})

    def test_abbreviation_pair_for_a_total(self):
        p = parse("NE/GB u43.5")
        self.assertEqual([(x["market"], x["side"], x["line"]) for x in p], [("total", "under", 43.5)])

    def test_several_picks_on_several_lines(self):
        p = parse("Lions -3\nPatriots +6.5\nSaints -2.5 ")
        self.assertEqual({x["side"] for x in p}, {"DET", "NE", "NO"})

    def test_things_that_are_not_game_picks_are_skipped(self):
        for t in ["Lions -3 Dolphins -3 parlay", "Lions -3 teaser 6 pts", "Lions -3 cashed ✅", "Recap: Patriots +6.5 hit",
                  "Lions -3 last week was a winner", "Gibbs over 55.5 rushing yards", "Lions to win the division +400 futures"]:
            self.assertEqual(parse(t), [], t)

    def test_a_prop_on_one_line_does_not_hide_a_pick_on_another(self):
        p = parse("Lions -3\nGibbs over 55.5 rush yards")
        self.assertEqual([(x["market"], x["side"]) for x in p], [("spread", "DET")])


class TestWhatCounts(unittest.TestCase):
    def test_after_kickoff_does_not_count(self):
        self.assertEqual(parse("Lions -3", "2026-11-08T18:30:00Z"), [])

    def test_too_early_does_not_count(self):
        self.assertEqual(parse("Lions -3", "2026-11-02T15:00:00Z"), [])                    # six days out
        self.assertEqual(len(parse("Lions -3", "2026-11-03T15:00:00Z")), 1)                # five days out is fine

    def test_a_post_picks_the_next_game_not_one_already_played(self):
        # Jaguars played Thursday; a Saturday post about them has no game within the window
        self.assertEqual(parse("Jaguars +4", SAT), [])
        self.assertEqual(parse("Jaguars +4", "2026-11-05T15:00:00Z")[0]["game_id"], "2026_09_JAX_BAL")

    def test_the_number_must_be_near_the_market(self):
        self.assertEqual(parse("Lions -9"), [])
        self.assertEqual(len(parse("Lions -5.5")), 1)                                      # within three points of -3
        self.assertEqual(parse("Lions/Dolphins over 60"), [])

    def test_a_team_name_with_nothing_after_it_is_not_a_pick(self):
        self.assertEqual(parse("The Lions are going to look great Sunday"), [])


class TestLog(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.tmp.name) / "social.csv"

    def tearDown(self):
        self.tmp.cleanup()

    def _post(self, source, text, when, pid):
        return {"source": source, "post_id": pid, "posted_at": when, "text": text}

    def test_a_flip_does_not_replace_the_first_pick(self):
        a = sp.rows_from_posts([self._post("@Sharp", "Lions -3", "2026-11-07T14:00:00Z", "1")], GAMES)
        b = sp.rows_from_posts([self._post("sharp", "Dolphins +3", "2026-11-07T20:00:00Z", "2")], GAMES)
        sp.log_picks(a, self.path)
        sp.log_picks(b, self.path)
        df = sp.load(self.path)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["side"], "DET")
        self.assertEqual(df.iloc[0]["source"], "sharp")                                    # handles are lowercased, '@' dropped

    def test_a_repost_is_one_pick_and_an_earlier_post_wins(self):
        later = sp.rows_from_posts([self._post("x", "Lions -3", "2026-11-07T20:00:00Z", "9")], GAMES)
        earlier = sp.rows_from_posts([self._post("x", "Lions -3", "2026-11-07T10:00:00Z", "3")], GAMES)
        sp.log_picks(later, self.path)
        self.assertEqual(sp.log_picks(earlier, self.path), 0)
        self.assertEqual(sp.load(self.path).iloc[0]["post_id"], "3")

    def test_a_grade_survives_a_later_run(self):
        rows = sp.rows_from_posts([self._post("x", "Jaguars +4", "2026-11-05T15:00:00Z", "1")], GAMES)
        sp.log_picks(rows, self.path)
        self.assertEqual(sp.grade(GAMES, self.path), 1)
        sp.log_picks(rows, self.path)
        self.assertEqual(sp.load(self.path).iloc[0]["result"], "loss")                     # BAL won 17-10: JAX +4 does not cover

    def test_import_reads_a_csv_in_eastern_time(self):
        f = pathlib.Path(self.tmp.name) / "p.csv"
        f.write_text("source,posted_at,text\n@a,2026-11-07 09:00,Lions -3\n@b,2026-11-07 09:30,nothing here\n", encoding="utf-8")
        r = sp.import_csv(f, GAMES, self.path)
        self.assertEqual(r, {"posts": 2, "picks_found": 1, "added": 1})
        self.assertEqual(sp.load(self.path).iloc[0]["posted_at"], "2026-11-07T14:00:00+00:00")


class TestGrading(unittest.TestCase):
    G = pd.Series({"home": "BAL", "away": "JAX", "home_score": 17.0, "away_score": 10.0})

    def row(self, market, side, line=None, odds=None):
        return pd.Series({"market": market, "side": side, "line": line, "odds": odds})

    def test_spread(self):
        self.assertEqual(sp.grade_pick(self.row("spread", "BAL", -6.5), self.G)[0], "win")             # won by 7
        self.assertEqual(sp.grade_pick(self.row("spread", "BAL", -7), self.G), ("push", 0.0))
        self.assertEqual(sp.grade_pick(self.row("spread", "BAL", -7.5), self.G)[0], "loss")
        self.assertEqual(sp.grade_pick(self.row("spread", "JAX", 7.5), self.G)[0], "win")
        self.assertEqual(sp.grade_pick(self.row("spread", "JAX", 6.5), self.G)[0], "loss")

    def test_total(self):
        self.assertEqual(sp.grade_pick(self.row("total", "over", 26.5), self.G)[0], "win")            # 27 points
        self.assertEqual(sp.grade_pick(self.row("total", "under", 26.5), self.G)[0], "loss")
        self.assertEqual(sp.grade_pick(self.row("total", "over", 27), self.G), ("push", 0.0))

    def test_units(self):
        res, p = sp.grade_pick(self.row("spread", "BAL", -6.5), self.G)
        self.assertAlmostEqual(p, 100 / 110)
        self.assertAlmostEqual(sp.grade_pick(self.row("spread", "BAL", -6.5, 150), self.G)[1], 1.5)
        self.assertEqual(sp.grade_pick(self.row("spread", "JAX", 6.5), self.G), ("loss", -1.0))

    def test_moneyline_has_no_unit_result_without_a_price(self):
        self.assertEqual(sp.grade_pick(self.row("moneyline", "BAL"), self.G), ("win", None))
        self.assertEqual(sp.grade_pick(self.row("moneyline", "JAX"), self.G), ("loss", None))
        self.assertAlmostEqual(sp.grade_pick(self.row("moneyline", "JAX", None, 200), self.G)[1], -1.0)
        self.assertAlmostEqual(sp.grade_pick(self.row("moneyline", "BAL", None, -150), self.G)[1], 100 / 150)

    def test_an_unfinished_game_is_not_graded(self):
        g = self.G.copy()
        g["home_score"] = np.nan
        self.assertEqual(sp.grade_pick(self.row("spread", "BAL", -3), g), (None, None))


class TestSkill(unittest.TestCase):
    def test_beta_sf(self):
        self.assertAlmostEqual(sp.beta_sf(0.5, 20, 20), 0.5, places=2)
        self.assertGreater(sp.beta_sf(0.5238, 40, 20), sp.beta_sf(0.5238, 30, 30))

    def test_a_hot_start_earns_nothing_until_enough_picks(self):
        self.assertEqual(sp.skill(5, 0)["weight"], 0.0)
        self.assertEqual(sp.skill(5, 0)["status"], "unproven")
        self.assertEqual(sp.skill(9, 0)["weight"], 0.0)

    def test_a_proven_winner_is_weighted_and_a_coin_flip_is_not(self):
        good, flat = sp.skill(20, 10), sp.skill(15, 15)
        self.assertGreater(good["weight"], 0.5)
        self.assertEqual(flat["weight"], 0.0)
        self.assertEqual(good["status"], "tracked")
        self.assertGreater(good["p_beats"], flat["p_beats"])

    def test_shrinkage_beats_a_small_sample(self):
        self.assertGreater(sp.skill(60, 40)["shrunk"], sp.skill(12, 8)["shrunk"])           # the same 60% is believed more over 100 picks than 20
        self.assertLess(sp.skill(6, 0)["shrunk"], 0.6)                                      # 100% over six picks is nowhere near 100%

    def test_weight_is_capped_at_one(self):
        self.assertEqual(sp.skill(80, 20)["weight"], 1.0)


def _log(rows):
    """rows: (source, week, market, side, line, result, posted_at, kickoff) -> a social log DataFrame."""
    recs = []
    for i, (src, wk, mkt, side, line, res, posted, kick) in enumerate(rows):
        recs.append({"source": src, "platform": "x", "post_id": str(i), "posted_at": posted, "season": 2026, "week": wk, "game_id": f"g{wk}_{i}",
                     "game": "AAA @ BBB", "kickoff": kick, "market": mkt, "side": side, "line": line, "odds": np.nan, "text": "",
                     "result": res if res else np.nan, "profit": (100 / 110 if res == "win" else -1.0 if res == "loss" else 0.0 if res else np.nan),
                     "graded_at": "x" if res else np.nan})
    return pd.DataFrame(recs, columns=sp.COLUMNS)


class TestBoard(unittest.TestCase):
    def history(self):
        rows = []
        day = pd.Timestamp("2026-09-13T17:00:00Z")
        for i in range(30):                                                                # 'pro': 20-10, 'flat': 15-15, 'new': 3-0
            k = day + pd.Timedelta(days=i)
            rows.append(("pro", 1 + i // 3, "spread", "AAA", -3.0, "win" if i % 3 else "loss", (k - pd.Timedelta(hours=20)).isoformat(), k.isoformat()))
            rows.append(("flat", 1 + i // 3, "total", "over", 45.5, "win" if i % 2 else "loss", (k - pd.Timedelta(hours=20)).isoformat(), k.isoformat()))
        for i in range(3):
            k = day + pd.Timedelta(days=i)
            rows.append(("new", 1, "spread", "AAA", -3.0, "win", (k - pd.Timedelta(hours=20)).isoformat(), k.isoformat()))
        return _log(rows)

    def test_source_table_orders_by_weight_and_flags_status(self):
        t = {r["source"]: r for r in sp.source_table(self.history(), 2026)}
        self.assertEqual((t["pro"]["wins"], t["pro"]["losses"]), (20, 10))
        self.assertEqual(t["pro"]["status"], "tracked")
        self.assertEqual(t["new"]["status"], "unproven")
        self.assertEqual(t["new"]["weight"], 0.0)
        self.assertEqual(sp.source_table(self.history(), 2026)[0]["source"], "pro")
        self.assertAlmostEqual(t["pro"]["units"], 20 * 100 / 110 - 10, places=6)

    def test_moneyline_stays_out_of_the_record(self):
        h = self.history()
        extra = _log([("pro", 9, "moneyline", "AAA", np.nan, "win", "2026-11-01T00:00:00+00:00", "2026-11-02T00:00:00+00:00")])
        t = {r["source"]: r for r in sp.source_table(pd.concat([h, extra]), 2026)}
        self.assertEqual((t["pro"]["wins"], t["pro"]["losses"]), (20, 10))

    def test_consensus_scores_by_proven_weight_and_skips_games_already_played(self):
        h = self.history()
        future = pd.Timestamp("2026-12-06T18:00:00Z")
        up = _log([("pro", 14, "spread", "AAA", -2.5, None, "2026-12-05T12:00:00+00:00", future.isoformat()),
                   ("new", 14, "spread", "AAA", -2.5, None, "2026-12-05T13:00:00+00:00", future.isoformat()),
                   ("flat", 14, "spread", "BBB", 2.5, None, "2026-12-05T14:00:00+00:00", future.isoformat())])
        up["game_id"] = "g14"
        both = pd.concat([h, up], ignore_index=True)
        out = sp.consensus(both, 2026, now="2026-12-05T20:00:00Z")
        self.assertEqual(len(out), 1)                                                      # graded games are not in it
        items = {i["side"]: i for i in out[0]["items"]}
        self.assertGreater(items["AAA"]["score"], 0.5)
        self.assertEqual(items["BBB"]["score"], 0.0)
        self.assertEqual(items["AAA"]["n"], 2)
        self.assertEqual(items["AAA"]["backers"][0]["source"], "pro")                      # best-weighted first
        self.assertEqual(sp.consensus(both, 2026, now="2026-12-07T00:00:00Z"), [])         # after kickoff: nothing left to show

    def test_a_good_record_predicts_nothing_in_pure_noise(self):
        rng = np.random.default_rng(3)
        rows, t0 = [], pd.Timestamp("2026-09-10T17:00:00Z")
        for s in range(30):
            for i in range(40):
                k = t0 + pd.Timedelta(days=i * 2 + s * 0.01)
                rows.append((f"s{s}", 1 + i // 4, "spread", "AAA", -3.0, "win" if rng.random() < 0.5 else "loss",
                             (k - pd.Timedelta(hours=20)).isoformat(), k.isoformat()))
        t = sp.skill_test(_log(rows), 2026)
        self.assertGreater(t["hot_n"], 50)
        self.assertLess(abs(t["hot_win"] - 0.5), 0.07)
        self.assertLess(abs(t["z"]), 3.0)

    def test_skill_test_only_uses_what_was_known_at_the_post(self):
        # the first MIN_PICKS picks of every source are classified with fewer than MIN_PICKS prior picks, so none can be 'hot'
        rows, k0 = [], pd.Timestamp("2026-09-10T17:00:00Z")
        for i in range(sp.MIN_PICKS):
            k = k0 + pd.Timedelta(days=i)
            rows.append(("ace", 1 + i, "spread", "AAA", -3.0, "win", (k - pd.Timedelta(hours=20)).isoformat(), k.isoformat()))
        t = sp.skill_test(_log(rows), 2026)
        self.assertEqual(t["hot_n"], 0)
        self.assertEqual(t["rest_n"], sp.MIN_PICKS)

    def test_block_is_json_and_section_reads(self):
        h = self.history()
        block = sp.build_block(2026, now="2026-12-05T00:00:00Z", df=h, config={"x_handles": ["a"], "max_posts_per_run": 200},
                               state={"since": {}, "posts_read": 10, "est_cost_usd": 0.05})
        json.dumps(block)
        self.assertEqual(block["handles"], 1)
        sec = sp.section(2026, 3, h)
        self.assertIn("graded picks this season", sec["writeup"])
        self.assertIsNone(sp.section(2026, 1, _log([]).iloc[0:0]))


class TestReport(unittest.TestCase):
    def test_the_weekly_report_renders_the_social_section(self):
        import weekly_report as wr
        from tests.test_weekly_report import _sched as report_sched
        from tests.test_weekly_report import TestGameSections
        rep = wr.build_report(TestGameSections()._games(3), report_sched(), 2026, 4)
        rep["social"] = {"week": 4, "writeup": "Week 4: 6 tracked picks went 4-2.", "tab": sp.source_table(TestBoard().history(), 2026)}
        md, html = wr.render_markdown(rep), wr.render_html(rep, [])
        self.assertIn("## Social picks", md)
        self.assertIn("@pro: 20-10", md)
        self.assertIn("Social picks</h2>", html)
        self.assertIn("@pro: 20-10", html)


class FakeX:
    """A stand-in for the X API: pages of posts newest first."""

    def __init__(self, posts_by_query, fail=None):
        self.posts, self.calls, self.fail = posts_by_query, [], fail

    def __call__(self, url, params, token):
        self.calls.append((url, dict(params), token))
        if self.fail:
            return self.fail, {"title": "Too Many Requests"}
        pool = self.posts
        if params.get("since_id"):
            pool = [p for p in pool if int(p["id"]) > int(params["since_id"])]
        start = int(params.get("next_token", 0))
        n = int(params["max_results"])
        page = pool[start:start + n]
        body = {"data": page, "includes": {"users": [{"id": "u1", "username": "SharpGuy"}]}}
        if start + n < len(pool):
            body["meta"] = {"next_token": str(start + n)}
        return 200, body


def _tweets(n, start=100):
    return [{"id": str(start + n - i), "author_id": "u1", "created_at": "2026-11-07T14:00:00.000Z", "text": "Lions -3"} for i in range(n)]


class TestX(unittest.TestCase):
    def test_chunks_stay_under_the_query_limit(self):
        handles = [f"handle_number_{i:03d}" for i in range(60)]
        chunks = sp.chunk_handles(handles)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(sum(chunks, []), handles)
        self.assertTrue(all(len(" OR ".join(f"from:{h}" for h in c)) <= sp.MAX_QUERY for c in chunks))

    def test_fetch_pages_through_and_remembers_where_it_stopped(self):
        api, state = FakeX(_tweets(150)), {"since": {}}
        posts, note = sp.fetch_posts(["sharpguy"], "TOKEN", state, 500, api)
        self.assertEqual((len(posts), note), (150, None))
        self.assertEqual(posts[0]["source"], "SharpGuy")
        self.assertEqual(state["posts_read"], 150)
        self.assertAlmostEqual(state["est_cost_usd"], 150 * sp.X_COST_PER_POST)
        again, _ = sp.fetch_posts(["sharpguy"], "TOKEN", state, 500, api)                  # nothing newer: nothing read, nothing paid
        self.assertEqual(again, [])
        self.assertEqual(state["posts_read"], 150)
        self.assertEqual(api.calls[-1][1]["since_id"], "250")

    def test_the_cap_stops_a_run_and_says_so(self):
        api, state = FakeX(_tweets(150)), {"since": {}}
        posts, note = sp.fetch_posts(["sharpguy"], "TOKEN", state, 100, api)
        self.assertEqual(len(posts), 100)
        self.assertIn("cap", note)
        self.assertLessEqual(state["posts_read"], 100)

    def test_an_error_is_reported_not_raised(self):
        posts, note = sp.fetch_posts(["a"], "TOKEN", {"since": {}}, 100, FakeX([], fail=429))
        self.assertEqual(posts, [])
        self.assertIn("429", note)

    def test_the_token_goes_only_in_the_auth_argument(self):
        api = FakeX(_tweets(3))
        sp.fetch_posts(["a"], "SECRET", {"since": {}}, 100, api)
        url, params, tok = api.calls[0]
        self.assertEqual(tok, "SECRET")
        self.assertNotIn("SECRET", url + json.dumps(params))

    def test_refresh_without_a_token_spends_nothing_and_still_grades(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            old = os.environ.pop("X_BEARER_TOKEN", None)
            try:
                sp.log_picks(sp.rows_from_posts([{"source": "x", "post_id": "1", "posted_at": "2026-11-05T15:00:00Z", "text": "Ravens -6.5"}], GAMES), d / "l.csv")
                api = FakeX(_tweets(5))
                r = sp.refresh(GAMES, api, d / "l.csv", d / "s.json", {"x_handles": ["a"], "max_posts_per_run": 100})
            finally:
                if old is not None:
                    os.environ["X_BEARER_TOKEN"] = old
            self.assertEqual(api.calls, [])
            self.assertIn("X_BEARER_TOKEN", r["note"])
            self.assertEqual(r["graded"], 1)

    def test_refresh_with_a_token_logs_picks_and_state(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            os.environ["X_BEARER_TOKEN"] = "T"
            try:
                tw = [{"id": "500", "author_id": "u1", "created_at": "2026-11-07T14:00:00.000Z", "text": "Lions -3 and Patriots +6.5"}]
                r = sp.refresh(GAMES, FakeX(tw), d / "l.csv", d / "s.json", {"x_handles": ["sharpguy"], "max_posts_per_run": 100})
            finally:
                os.environ.pop("X_BEARER_TOKEN", None)
            self.assertEqual((r["posts"], r["picks"], r["added"]), (1, 2, 2))
            self.assertEqual(sp.load_state(d / "s.json")["posts_read"], 1)
            self.assertEqual(set(sp.load(d / "l.csv")["source"]), {"sharpguy"})

    def test_timeline_backfill(self):
        def get(url, params, token):
            if url.endswith("/users/by"):
                return 200, {"data": [{"id": "7", "username": "Cap"}]}
            return 200, {"data": [{"id": "1", "created_at": "2026-09-13T12:00:00Z", "text": "x"}]}
        posts, note = sp.fetch_timelines(["cap"], "T", "2026-09-01", 50, get)
        self.assertEqual((len(posts), note), (1, None))
        self.assertEqual(posts[0]["source"], "Cap")

    def test_config_merges_the_file_and_the_environment(self):
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "c.json"
            f.write_text(json.dumps({"x_handles": ["@One", "two"], "max_posts_per_run": 50}), encoding="utf-8")
            os.environ["X_HANDLES"] = "two, Three"
            try:
                c = sp.load_config(f)
            finally:
                os.environ.pop("X_HANDLES", None)
            self.assertEqual(c, {"x_handles": ["one", "two", "three"], "max_posts_per_run": 50})
            self.assertEqual(sp.load_config(pathlib.Path(d) / "missing.json")["x_handles"], [])


if __name__ == "__main__":
    unittest.main()
