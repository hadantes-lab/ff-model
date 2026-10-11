"""
Tests for weekly_report.py (grading each kind of pick, the write-ups, season to date, which week to report) and
improve.py (the learning loop: re-measuring trust weights from tracked results, drift flags, the change log).

Run:  python -m unittest discover -s tests -v
"""

import pathlib
import tempfile
import unittest

import numpy as np
import pandas as pd

import improve
import market_trust as mt
import weekly_report as wr


def _row(**kw):
    base = {"season": 2026, "week": 4, "player_id": "p", "market": "player_receptions", "player": "P", "team": "AAA", "opp": "BBB",
            "pos": "WR", "game": "BBB @ AAA", "commence": "2026-10-04T17:00:00Z", "kind": "count", "line": 4.5, "model_over": 0.7,
            "model_under": 0.3, "market_over": 0.5, "gap": None, "trust": 0.05, "pick": None, "pick_price": None, "pick_ev": None,
            "result": "over", "actual": 6.0, "mean": np.nan, "sd": np.nan, "top_rank": np.nan}
    base.update(kw)
    return base


def _log(rows):
    return pd.DataFrame(rows)


class TestRecord(unittest.TestCase):
    def test_outcome_by_side_and_result(self):
        self.assertEqual(wr.outcome_of(True, "over"), "win")
        self.assertEqual(wr.outcome_of(True, "under"), "loss")
        self.assertEqual(wr.outcome_of(False, "under"), "win")
        self.assertEqual(wr.outcome_of(False, "over"), "loss")
        self.assertEqual(wr.outcome_of(True, "push"), "push")

    def test_record_counts_wins_losses_and_excludes_pushes_from_the_rate(self):
        d = _log([_row(result="over"), _row(result="over"), _row(result="under"), _row(result="push")])    # model leaned over on all
        r = wr.record(d)
        self.assertEqual((r["wins"], r["losses"], r["pushes"], r["n"]), (2, 1, 1, 3))
        self.assertAlmostEqual(r["win_rate"], 2 / 3)
        self.assertAlmostEqual(r["expected_wins"], 3 * 0.7)                              # its stated 70% on each live pick
        self.assertAlmostEqual(r["expected_rate"], 0.7)

    def test_the_models_side_is_its_raw_lean_either_way(self):
        d = _log([_row(model_over=0.3, result="under"), _row(model_over=0.2, result="over")])
        r = wr.record(d)
        self.assertEqual((r["wins"], r["losses"]), (1, 1))
        self.assertAlmostEqual(r["expected_wins"], 0.7 + 0.8)                             # probabilities of the side it LEANED

    def test_z_score_is_against_the_probabilities_it_stated(self):
        lucky = wr.record(_log([_row(result="over")] * 40))                              # 40-0 at a stated 70%
        self.assertGreater(lucky["z"], 3)
        unlucky = wr.record(_log([_row(result="under")] * 40))
        self.assertLess(unlucky["z"], -3)
        typical = wr.record(_log([_row(result="over")] * 28 + [_row(result="under")] * 12))   # exactly 70%
        self.assertAlmostEqual(typical["z"], 0.0, places=6)

    def test_brier_compares_the_model_with_the_market(self):
        d = _log([_row(model_over=0.9, market_over=0.5, result="over"), _row(model_over=0.9, market_over=0.5, result="over")])
        r = wr.record(d)
        self.assertLess(r["brier_model"], r["brier_market"])

    def test_empty_group_is_safe(self):
        r = wr.record(pd.DataFrame())
        self.assertEqual((r["n"], r["win_rate"], r["z"]), (0, None, None))

    def test_luck_phrase_thresholds(self):
        self.assertIn("about where", wr.luck_phrase(0.4))
        self.assertIn("within normal variation", wr.luck_phrase(1.5))
        self.assertIn("worse", wr.luck_phrase(-1.5))
        self.assertTrue(wr.luck_phrase(-2.5).startswith("clearly worse"))
        self.assertTrue(wr.luck_phrase(2.5).startswith("clearly better"))
        self.assertIn("too few", wr.luck_phrase(None))

    def test_accuracy_needs_the_logged_projection(self):
        d = _log([_row(mean=5.0, actual=6.0, line=4.5), _row(mean=np.nan, actual=2.0, line=4.5)])
        a = wr.accuracy(d)
        self.assertEqual(a["n_acc"], 1)
        self.assertAlmostEqual(a["mae_model"], 1.0)
        self.assertAlmostEqual(a["mae_market"], 1.5)
        self.assertEqual(wr.accuracy(pd.DataFrame({"x": [1]}))["n_acc"], 0)


def _sched():
    rows = []
    for i, (h, a, hs, as_) in enumerate([("AAA", "BBB", 24, 17), ("CCC", "DDD", 10, 20), ("EEE", "FFF", 20, 20)]):
        rows.append({"season": 2026, "week": 4, "game_type": "REG", "gameday": "2026-10-04", "gametime": "13:00", "home_team": h, "away_team": a,
                     "home_score": hs, "away_score": as_, "spread_line": [3.0, 10.0, 1.0][i]})
    return pd.DataFrame(rows)


class TestMoneyline(unittest.TestCase):
    def test_logged_picks_are_graded_by_tier(self):
        # logged: AAA 80% to win at home (A tier) and it did; CCC 85% (S) and it lost
        log = _log([_row(market="game_su", pos="GAME", player="BBB @ AAA", game="BBB @ AAA", team="AAA", opp="BBB", model_over=0.80, result="over"),
                    _row(market="game_su", pos="GAME", player="DDD @ CCC", game="DDD @ CCC", team="CCC", opp="DDD", model_over=0.85, result="under")])
        s = wr.moneyline_section(log, _sched(), 2026, 4)
        self.assertEqual(s["source"], "logged")
        self.assertEqual((s["wins"], s["losses"]), (1, 1))
        by = {t["tier"]: t for t in s["tiers"]}
        self.assertEqual((by["A"]["wins"], by["A"]["losses"]), (1, 0))
        self.assertEqual((by["S"]["wins"], by["S"]["losses"]), (0, 1))
        self.assertEqual([e["tier"] for e in s["entries"]], ["S", "A"])                  # best tier first

    def test_a_road_pick_and_a_tie(self):
        log = _log([_row(market="game_su", pos="GAME", game="DDD @ CCC", team="CCC", opp="DDD", model_over=0.2, result="under"),    # picked road DDD; home lost
                    _row(market="game_su", pos="GAME", game="FFF @ EEE", team="EEE", opp="FFF", model_over=0.6, result="push")])
        s = wr.moneyline_section(log, _sched(), 2026, 4)
        e = {x["game"]: x for x in s["entries"]}
        self.assertEqual((e["DDD @ CCC"]["pick"], e["DDD @ CCC"]["result"]), ("DDD", "win"))
        self.assertEqual(e["FFF @ EEE"]["result"], "push")
        self.assertEqual(s["pushes"], 1)

    def test_a_week_with_no_logged_picks_is_rebuilt_from_closing_spreads_and_says_so(self):
        s = wr.moneyline_section(_log([_row(market="player_receptions")]), _sched(), 2026, 4)
        self.assertEqual(s["source"], "reconstructed")
        e = {x["game"]: x for x in s["entries"]}
        self.assertEqual(e["BBB @ AAA"]["pick"], "AAA")                                    # nflverse +3 home = home favored
        self.assertEqual(e["BBB @ AAA"]["result"], "win")
        self.assertEqual(e["DDD @ CCC"]["result"], "loss")                                 # the 10-point favorite lost
        self.assertIn("rebuilt from closing spreads", wr.blurb(s))


class TestGameSections(unittest.TestCase):
    def _games(self, n, over_pick=True, result="over", market="game_total"):
        return _log([_row(market=market, pos="GAME", game=f"G{i} @ H{i}", player_id=f"GAME_{i}", model_over=0.7 if over_pick else 0.3,
                          result=result, line=45.0, actual=50.0) for i in range(n)])

    def test_picks_use_the_recommended_side_when_there_is_one_else_the_raw_lean(self):
        d = _log([_row(market="game_spread", pos="GAME", game="A @ B", model_over=0.7, pick="under", result="under", line=-3.0),     # recommended the other way
                  _row(market="game_spread", pos="GAME", game="C @ D", model_over=0.7, pick=None, result="over", line=-3.0)])
        s = wr.game_section(d, 2026, 4, "game_spread")
        self.assertEqual((s["wins"], s["losses"]), (2, 0))

    def test_a_one_sided_lean_is_called_out(self):
        s = wr.game_section(self._games(16), 2026, 4, "game_total")
        s["writeup"] = wr.blurb(s)
        self.assertIn("16 of 16 picks were on Over", s["writeup"])
        self.assertIn("built-in bias", s["writeup"])

    def test_a_balanced_slate_is_not_flagged(self):
        d = pd.concat([self._games(8), self._games(8, over_pick=False, result="under")], ignore_index=True)
        d["game"] = [f"G{i} @ H{i}" for i in range(len(d))]
        self.assertNotIn("built-in bias", wr.blurb(wr.game_section(d, 2026, 4, "game_total")))

    def test_the_report_will_not_call_a_winner_on_a_small_sample(self):
        d = self._games(14, market="game_spread")
        d["market_over"] = 0.5
        s = wr.game_section(d, 2026, 4, "game_spread")
        self.assertIn("too few to say which was sharper", wr.blurb(s))
        big = self._games(40, market="game_spread"); big["market_over"] = 0.5
        self.assertIn("was the sharper of the two", wr.blurb(wr.game_section(big, 2026, 4, "game_spread")))

    def test_totals_are_formatted_without_a_plus_sign(self):
        rep = wr.build_report(self._games(3), _sched(), 2026, 4)
        self.assertNotIn("Over +", wr.render_markdown(rep))
        self.assertNotIn("Over +", wr.render_html(rep))


class TestProps(unittest.TestCase):
    def _props(self):
        rows = []
        for i in range(10):      # five high-confidence leans (p=.8): four right; five weak leans (p=.55): two right
            rows.append(_row(player_id=f"h{i}", player=f"H{i}", model_over=0.8, result="over" if i < 4 else "under"))
        for i in range(5):
            rows.append(_row(player_id=f"w{i}", player=f"W{i}", model_over=0.55, result="over" if i < 2 else "under"))
        return _log(rows)

    def test_high_confidence_is_the_raw_probability_bar(self):
        s = wr.prop_section(self._props().assign(result=lambda d: d["result"]), 2026, 4, "player_receptions")
        self.assertEqual(s["all"]["n"], 15)
        self.assertEqual(s["high"]["n"], 10)
        self.assertEqual((s["high"]["wins"], s["high"]["losses"]), (4, 6))
        self.assertTrue(all(i["prob"] >= wr.HIGH_CONF for i in s["items"]))

    def test_recommended_picks_are_the_positive_ev_side_and_top_ranked_props(self):
        d = self._props()
        d["pick"] = None
        d["pick_ev"] = np.nan
        d.loc[0, ["pick", "pick_ev"]] = ["over", 0.03]                    # recommended, and it won
        d.loc[1, ["pick", "pick_ev"]] = ["over", -0.02]                   # a pick with negative EV is not a recommendation
        d["top_rank"] = np.nan
        d.loc[12, "top_rank"] = 3                                        # a published top-10 prop (a weak lean that lost)
        s = wr.prop_section(d, 2026, 4, "player_receptions")
        self.assertEqual(s["recommended"]["n"], 2)

    def test_a_market_the_model_got_badly_wrong_is_described_as_such(self):
        d = _log([_row(player_id=f"p{i}", model_over=0.85, result="under") for i in range(30)])
        s = wr.prop_section(d, 2026, 4, "player_receptions")
        s["writeup"] = wr.blurb(s)
        self.assertIn("clearly worse than its probabilities expected", s["writeup"])

    def test_no_props_reached_the_high_confidence_bar(self):
        d = _log([_row(player_id=f"p{i}", model_over=0.55) for i in range(5)])
        s = wr.prop_section(d, 2026, 4, "player_receptions")
        self.assertIn("No props reached", wr.blurb(s))


class TestReportAndHistory(unittest.TestCase):
    def setUp(self):
        self.log = pd.concat([_log([_row(market="game_su", pos="GAME", game="BBB @ AAA", team="AAA", opp="BBB", model_over=0.8, result="over")]),
                              _log([_row(market="game_spread", pos="GAME", game="BBB @ AAA", model_over=0.6, result="over", line=-3.0)]),
                              self._pending(),
                              TestProps()._props()], ignore_index=True)

    @staticmethod
    def _pending():
        return _log([_row(market="game_total", pos="GAME", game="DDD @ CCC", result=np.nan, actual=np.nan)])

    def test_headline_and_pending_games(self):
        rep = wr.build_report(self.log, _sched(), 2026, 4)
        self.assertTrue(rep["headline"].startswith("Week 4:"))
        self.assertIn("moneyline 1-0", rep["headline"])
        self.assertEqual(rep["games_pending"], 1)
        self.assertIn("still to be graded", rep["headline"])

    def test_every_section_has_a_writeup(self):
        rep = wr.build_report(self.log, _sched(), 2026, 4)
        for s in rep["sections"] + rep["props"]:
            self.assertTrue(s["writeup"])

    def test_summary_upsert_is_idempotent_and_season_to_date_adds_weeks(self):
        path = pathlib.Path(tempfile.mkdtemp()) / "s.csv"
        rep4 = wr.build_report(self.log, _sched(), 2026, 4)
        wr.upsert_summary(wr.summary_rows(rep4), path)
        h = wr.upsert_summary(wr.summary_rows(rep4), path)                               # a re-run replaces, never doubles
        self.assertEqual(h.duplicated(subset=["season", "week", "group"]).sum(), 0)
        log5 = self.log.assign(week=5)
        rep5 = wr.build_report(log5, _sched(), 2026, 5)
        h = wr.upsert_summary(wr.summary_rows(rep5), path)
        std = wr.season_to_date(h, rep5)
        self.assertEqual(std["player_receptions:high"]["weeks"], 2)
        self.assertEqual(std["player_receptions:high"]["n"], 20)                          # 10 high-confidence props in each of two weeks
        self.assertEqual(wr.season_to_date(h, rep4)["player_receptions:high"]["weeks"], 1)    # asking "through week 4" ignores week 5

    def test_renderers_escape_html_and_include_every_section(self):
        log = self.log.copy()
        log.loc[log["player_id"] == "h0", "player"] = "<script>alert(1)</script>"
        rep = wr.build_report(log, _sched(), 2026, 4)
        page = wr.render_html(rep, [{"week": 4, "href": "week-04.html"}])
        self.assertNotIn("<script>alert", page)
        self.assertIn("Moneyline winners", page)
        self.assertIn("Earlier weeks", page)
        md = wr.render_markdown(rep)
        self.assertIn("## Moneyline winners (straight up)", md)
        self.assertIn("## What the data is teaching the model", md)


class TestDefaultWeek(unittest.TestCase):
    def _log_for(self, graded_games_by_week):
        rows = []
        for wk, n in graded_games_by_week.items():
            rows += [_row(market="game_spread", pos="GAME", week=wk, game=f"G{i} @ H{i}", result="over") for i in range(n)]
        return _log(rows)

    def _sched_weeks(self, weeks):
        return pd.DataFrame([{"season": 2026, "week": w, "game_type": "REG"} for w in weeks for _ in range(15)])

    def test_sunday_morning_still_reports_the_previous_full_week(self):
        # week 4 fully graded; week 5 has only Thursday's game graded
        self.assertEqual(wr.default_week(self._log_for({4: 15, 5: 1}), self._sched_weeks([4, 5]), 2026), 4)

    def test_monday_reports_the_weekend(self):
        self.assertEqual(wr.default_week(self._log_for({4: 15, 5: 14}), self._sched_weeks([4, 5]), 2026), 5)

    def test_falls_back_to_the_latest_graded_week_and_handles_nothing_graded(self):
        self.assertEqual(wr.default_week(self._log_for({4: 3}), self._sched_weeks([4]), 2026), 4)
        self.assertIsNone(wr.default_week(_log([_row(result=np.nan)]), self._sched_weeks([4]), 2026))


class TestLearningLoop(unittest.TestCase):
    def _setup(self):
        d = pathlib.Path(tempfile.mkdtemp())
        trust = d / "trust.json"
        mt.write({"player_receptions": {"weight": 0.0, "n": 500, "verdict": "none"}}, trust)
        return d, trust

    def _evidence(self, n=600, seed=0):
        rng = np.random.default_rng(seed)
        true = rng.uniform(0.3, 0.7, n)
        return pd.DataFrame({"market": "player_receptions", "p_mkt": np.clip(true + rng.normal(0, 0.03, n), .05, .95),
                             "p_m_env": np.clip(true + rng.normal(0, 0.25, n), .02, .98), "over": (rng.random(n) < true).astype(float)})

    def _live(self, n, skilled):
        """Tracked graded props. If `skilled`, the model's side wins far more often than the market's price implies."""
        rng = np.random.default_rng(5)
        rows = []
        for i in range(n):
            truth = rng.uniform(0.3, 0.7)
            over = rng.random() < truth
            rows.append(_row(player_id=f"p{i}", model_over=float(np.clip(truth + rng.normal(0, 0.02 if skilled else 0.25), .02, .98)),
                             market_over=float(np.clip(truth + rng.normal(0, 0.15 if skilled else 0.03), .05, .95)),
                             result="over" if over else "under"))
        return _log(rows)

    def test_live_evidence_keeps_only_graded_priced_props(self):
        log = _log([_row(), _row(result=np.nan), _row(market_over=np.nan), _row(pos="GAME", market="game_spread"), _row(result="push")])
        self.assertEqual(len(improve.live_evidence(log)), 1)

    def test_too_little_live_data_changes_nothing(self):
        d, trust = self._setup()
        notes = improve.update_weights(self._live(improve.MIN_LIVE_PER_MARKET - 1, True), self._evidence(), trust, d / "l.csv", "2026-10-12")
        self.assertEqual(notes, [])
        self.assertEqual(mt.load_weights(trust)["player_receptions"], 0.0)
        self.assertFalse((d / "l.csv").exists())

    def test_evidence_that_a_market_is_better_raises_its_weight_and_logs_it(self):
        d, trust = self._setup()
        notes = improve.update_weights(self._live(3000, True), self._evidence(300), trust, d / "l.csv", "2026-10-12")
        self.assertEqual(len(notes), 1)
        self.assertGreater(mt.load_weights(trust)["player_receptions"], 0.0)
        self.assertLessEqual(mt.load_weights(trust)["player_receptions"], mt.WEIGHT_CAP)
        log = pd.read_csv(d / "l.csv")
        self.assertEqual((log.iloc[0]["market"], log.iloc[0]["old_weight"]), ("player_receptions", 0.0))
        self.assertEqual(len(improve.recent_changes("2026-10-13", 7, d / "l.csv")), 1)
        self.assertEqual(improve.recent_changes("2026-12-01", 7, d / "l.csv"), [])         # old changes age out of the report

    def test_a_small_wobble_is_ignored(self):
        d, trust = self._setup()
        mt.write({"player_receptions": {"weight": 0.05, "n": 500, "verdict": "x"}}, trust)
        # evidence whose measured weight is also the floor: |change| < MIN_CHANGE, so the file is left alone
        notes = improve.update_weights(self._live(60, False), self._evidence(900), trust, d / "l.csv", "2026-10-12")
        self.assertEqual(notes, [])
        self.assertEqual(mt.load_weights(trust)["player_receptions"], 0.05)

    def test_other_sections_of_the_weights_file_are_preserved(self):
        d, trust = self._setup()
        mt.write({"player_receptions": {"weight": 0.0, "n": 500, "verdict": "none"}}, trust, early={"player_x": {"qualifies": False}}, confidence={"strength": 0.0})
        improve.update_weights(self._live(3000, True), self._evidence(300), trust, d / "l.csv", "2026-10-12")
        import json
        payload = json.loads(trust.read_text(encoding="utf-8"))
        self.assertIn("early", payload)
        self.assertIn("confidence", payload)

    def test_drift_flags_only_for_big_deviations_on_enough_picks(self):
        std = {"player_receptions:high": {"n": 107, "wins": 62, "losses": 45, "z": -2.8},
               "moneyline": {"n": 16, "wins": 9, "losses": 7, "z": -2.5},                # too few picks
               "game_spread": {"n": 60, "wins": 33, "losses": 27, "z": 0.4},             # within noise
               "player_rush_yds:all": {"n": 90, "wins": 40, "losses": 50, "z": -3.0}}    # not a published group
        flags = improve.drift_flags(std)
        self.assertEqual(len(flags), 1)
        self.assertIn("player_receptions:high", flags[0])
        self.assertIn("raw confidence runs too high", flags[0])

    def test_run_always_says_something(self):
        d, trust = self._setup()
        notes = improve.run(pd.DataFrame(columns=["pos"]), 2026, 4, trust_path=trust, log_path=d / "l.csv", evidence=self._evidence(50), today="2026-10-12")
        self.assertIn("No market's evidence moved", notes[0])

    def test_evidence_roundtrip_is_compact(self):
        path = pathlib.Path(tempfile.mkdtemp()) / "e.csv"
        frame = self._evidence(50)
        frame["extra"] = 1
        self.assertEqual(improve.save_evidence_from_frame(frame, path), 50)
        back = improve.load_evidence(path)
        self.assertEqual(list(back.columns), ["market", "p_mkt", "p_m_env", "over"])
        self.assertEqual(len(back), 50)


if __name__ == "__main__":
    unittest.main()
