"""
Tests for the second feature round: the market-anchored totals model, the single-slate filter, teaser curve and same-week
pairs, Underdog pick'em entries (build, EV, grading), the weekly credit budget, the targets projection, and the
weekly review sections (top 10, streaks, target accuracy).

Run:  python -m unittest discover -s tests -v
"""

import datetime
import json
import pathlib
import tempfile
import unittest

import numpy as np
import pandas as pd

import export_props as ep
import parlay_track as pt
import parlays as pl
import review_extras as rx
import target_track as tt
import totals_model as tm
from tests.test_parlays import CORR, _game, _sched


# ---- totals model -----------------------------------------------------------------------------
class TestTotalsModel(unittest.TestCase):
    M = {"dome": 0.9, "wind": -1.7, "mean_dome": 0.28, "mean_wind": 0.15, "sd": 13.2, "shrink": 0.75}

    def test_features(self):
        self.assertEqual(tm.features("dome", 20)["dome"], 1.0)
        self.assertEqual(tm.features("closed", None)["wind"], 0.0)
        self.assertEqual(tm.features("outdoors", 12)["wind"], 1.0)
        self.assertEqual(tm.features("outdoors", 11.9)["wind"], 0.0)
        self.assertEqual(tm.features("outdoors", None)["wind"], 0.0)
        self.assertEqual(tm.features("outdoors", float("nan"))["wind"], 0.0)

    def test_windy_games_go_down_and_domes_up_relative_to_the_average_game(self):
        calm = tm.adjustment(self.M, "outdoors", 4, 60)
        windy = tm.adjustment(self.M, "outdoors", 18, 40)
        dome = tm.adjustment(self.M, "dome")
        self.assertLess(windy, calm - 1.0)
        self.assertGreater(dome, calm + 0.5)
        self.assertLess(abs(calm), 0.3)                       # a calm outdoor game is within a fraction of a point of the line

    def test_unknown_roof_retractable_is_between_the_two_cases(self):
        dome, calm = tm.adjustment(self.M, "dome"), tm.adjustment(self.M, "outdoors", 3, 70)
        mid = tm.adjustment(self.M, "outdoors", 3, 70, retractable=True)
        self.assertTrue(min(dome, calm) < mid < max(dome, calm))

    def test_over_probability_is_anchored_to_the_market_not_to_the_model(self):
        self.assertAlmostEqual(tm.over_probability(0.0), 0.5)
        self.assertAlmostEqual(tm.over_probability(0.0, fair_over=0.52), 0.52)
        self.assertLess(tm.over_probability(-1.7, 13.2), 0.5)
        self.assertLess(abs(tm.over_probability(-1.7, 13.2) - 0.5), 0.06)         # a wind effect moves a total a few points, never to 80%
        self.assertLessEqual(tm.over_probability(40.0), 0.97)

    def test_apply_replaces_the_inflated_simulation_total(self):
        g = {"game": "A @ B", "total": 44.5, "proj_total": 58.0, "proj_home": 30.0, "proj_away": 28.0, "fair_over": 0.5}
        n = tm.apply([g], {"A @ B": {"indoors": False, "roof": "outdoors", "wind_mph": 20, "gust_mph": 28, "temp_f": 40}}, self.M)
        self.assertEqual(n, 1)
        self.assertEqual(g["sim_total"], 58.0)
        self.assertLess(abs(g["proj_total"] - 44.5), 3.0)                              # anchored to the posted total, not 58
        self.assertAlmostEqual(g["proj_home"] + g["proj_away"], g["proj_total"], delta=0.2)
        self.assertLess(g["over"], 0.5)
        self.assertAlmostEqual(g["over"] + g["under"], 1.0, places=3)
        self.assertEqual(g["total_basis"], "wind")

    def test_apply_without_weather_is_the_posted_line(self):
        g = {"game": "A @ B", "total": 47.0, "proj_total": 61.0}
        tm.apply([g], {}, self.M)
        self.assertAlmostEqual(g["predicted_total"], 47.0, delta=0.5)
        self.assertAlmostEqual(g["over"], 0.5, delta=0.02)

    def test_fit_on_synthetic_games_recovers_a_planted_wind_effect(self):
        rng = np.random.default_rng(0)
        n = 4000
        roof = rng.choice(["outdoors", "dome"], n, p=[.7, .3])
        wind = rng.choice([3, 8, 15, 20], n)
        line = rng.normal(45, 4, n)
        extra = np.where((roof == "outdoors") & (wind >= 12), -2.0, 0.0)
        tot = line + extra + rng.normal(0, 13, n)
        sched = pd.DataFrame({"season": rng.choice(range(2014, 2026), n), "week": 1, "game_type": "REG", "total_line": line,
                              "home_score": tot / 2, "away_score": tot / 2, "roof": roof, "wind": wind, "temp": 50})
        m = tm.fit(sched)
        self.assertLess(m["wind"], -1.0)
        self.assertGreater(m["n"], 3000)
        self.assertIn("backtest", m)

    def test_shipped_coefficients_have_the_expected_signs(self):
        m = tm.load()
        self.assertLess(m["wind"], 0)
        self.assertGreater(m["dome"], 0)
        self.assertGreater(m["sd"], 10)


# ---- slate and teasers ---------------------------------------------------------------------------
def _ev(i, home, away, when):
    return {"id": f"e{i}", "home_team": home, "away_team": away, "commence_time": when}


class TestSlate(unittest.TestCase):
    def test_slate_week_is_the_earliest_week_still_to_play(self):
        self.assertEqual(pl.slate_week([{"week": 9}, {"week": 8}, {"week": 8}]), 8)
        self.assertIsNone(pl.slate_week([{"week": None}]))
        self.assertIsNone(pl.slate_week([]))

    def test_filter_to_slate_drops_next_weeks_thursday_game(self):
        sched = pd.DataFrame([
            {"home_team": "GB", "away_team": "CAR", "gameday": "2026-11-01", "week": 8},
            {"home_team": "BAL", "away_team": "JAX", "gameday": "2026-11-05", "week": 9},
        ])
        evs = [_ev(1, "Green Bay Packers", "Carolina Panthers", "2026-11-01T18:00:00Z"),
               _ev(2, "Baltimore Ravens", "Jacksonville Jaguars", "2026-11-06T01:15:00Z")]
        keep, slate = ep.filter_to_slate(evs, sched)
        self.assertEqual(slate, 8)
        self.assertEqual([e["id"] for e in keep], ["e1"])

    def test_filter_without_a_schedule_changes_nothing(self):
        evs = [_ev(1, "Green Bay Packers", "Carolina Panthers", "2026-11-01T18:00:00Z")]
        self.assertEqual(ep.filter_to_slate(evs, None), (evs, None))

    def test_build_all_ignores_games_from_other_weeks(self):
        mk = lambda name, wk, spread: {"game": name, "home": name[-1], "away": name[0], "home_spread": spread, "total": 44.0, "in_window": True, "week": wk,
                                       **{k: v for k, v in _game("x", 0.6, -140, 120).items() if k.startswith(("fair", "best"))}}
        games = [mk("A @ B", 8, 2.5), mk("C @ D", 8, -1.5), mk("E @ F", 9, 2.5)]
        out = pl.build_all([], games, None, CORR, _sched([(2025, 7.5, 3)] * 5), 2026, 8)
        self.assertEqual(out["slate_week"], 8)
        for leg in out["teasers"]["legs"]:
            self.assertEqual(leg["week"], 8)
        for p in out["ml"]["parlays"]:
            self.assertTrue(all(l["week"] == 8 for l in p["legs"]))


class TestTeaserCurve(unittest.TestCase):
    def setUp(self):
        rows = []
        for sp in (2.5, 1.5, 7.5, 3.5):
            rows += [(2024, sp, 3)] * 40            # home favored by sp wins by 3: covers its own teased line only when it was the dog...
        self.sched = _sched(rows)

    def test_curve_covers_half_points_and_shrinks_to_the_normal_curve(self):
        c = pl.teaser_curve(self.sched, first=2024)
        self.assertIn(2.5, c)
        self.assertIn(-14.0, c)
        self.assertNotIn(0.0, c)
        sparse = c[13.5]
        self.assertEqual(sparse["n"], 0)
        self.assertAlmostEqual(sparse["rate"], float(pl.ND.cdf(pl.TEASER_POINTS / 13.5)), places=6)      # flat ~67% off the key numbers
        for v in c.values():
            self.assertTrue(0.0 < v["rate"] < 1.0)

    def test_snap_spread(self):
        self.assertEqual(pl.snap_spread(2.5), 2.5)
        self.assertEqual(pl.snap_spread(2.4), 2.5)
        self.assertIsNone(pl.snap_spread(2.25))
        self.assertEqual(pl.snap_spread(2.4), 2.5)
        self.assertEqual(pl.snap_spread(-7.0), -7.0)

    def test_legs_include_non_wong_spreads_only_with_a_curve(self):
        games = [{"game": "A @ B", "home": "B", "away": "A", "home_spread": -3.5, "week": 8, "commence": "x"}]
        hist = pl.wong_history(_sched([(2024, 7.5, 3)] * 5), recent_from=2024, first=2024)
        self.assertEqual(pl.wong_legs(games, hist), [])
        curve = pl.teaser_curve(_sched([(2024, 3.5, 3)] * 5), first=2024)
        legs = pl.wong_legs(games, hist, curve)
        self.assertEqual(len(legs), 2)
        self.assertTrue(all(not l["wong"] for l in legs))

    def test_pairs_never_mix_weeks(self):
        mk = lambda i, wk, p: {"id": f"g{i}|wong|T", "game": f"g{i}", "team": f"T{i}", "opp": "O", "spread": 2.5, "teased": 8.5, "bucket": "dog", "prob": p,
                               "kind": "wong", "week": wk, "playable": True, "wong": True}
        t = pl.build_teasers([mk(1, 8, .78), mk(2, 9, .77), mk(3, 8, .70)])
        for pr in t["pairs"]:
            self.assertEqual(pr["legs"][0]["week"], pr["legs"][1]["week"])
        self.assertEqual(len(t["pairs"]), 1)


# ---- pick'em -----------------------------------------------------------------------------------------
def _pk_prop(pid, market="player_receptions", side="over", prob=0.60, agree=True, team=None, game=None, **kw):
    team = team or f"T{pid}"
    d = {"player_id": pid, "player": pid.upper(), "team": team, "opp": "ZZZ", "game": game or f"ZZZ @ {team}", "pos": "WR", "market": market, "label": market,
         "kind": "count", "line": 4.5, "over": prob, "under": 1 - prob, "model_over": 0.6 if side == "over" else 0.4,
         "pickem": {"line": 4.5, "book": "Underdog", "side": side, "prob": prob, "sportsbook_prob": prob - 0.02, "line_edge": 0.02, "agree": agree, "grade": "high"}}
    d.update(kw)
    return d


class TestPickem(unittest.TestCase):
    def test_pickem_entry_uses_underdogs_line_and_agreement(self):
        p = {"over": 0.55, "under": 0.45}
        raw = {"over": 0.62, "under": 0.38}
        dfs = [{"book": "Underdog Fantasy", "line": 3.5, "over_adj": 0.66, "under_adj": 0.34},
               {"book": "PrizePicks", "line": 4.5, "over_adj": 0.55, "under_adj": 0.45}]
        e = ep.pickem_entry(p, raw, 4.5, dfs, "count")
        self.assertEqual((e["book"], e["line"], e["side"]), ("Underdog Fantasy", 3.5, "over"))
        self.assertAlmostEqual(e["line_edge"], 0.11, places=3)
        self.assertTrue(e["agree"])
        self.assertEqual(e["grade"], "high")

    def test_pickem_entry_falls_back_to_the_sportsbook_line(self):
        e = ep.pickem_entry({"over": 0.45, "under": 0.55}, {"over": 0.40, "under": 0.60}, 4.5, [], "count")
        self.assertIsNone(e["book"])
        self.assertEqual((e["side"], e["line"]), ("under", 4.5))
        self.assertEqual(e["line_edge"], 0.0)
        self.assertIsNone(ep.pickem_entry({"over": .5, "under": .5}, {"over": .5, "under": .5}, 0.5, [], "td"))

    def test_hit_distribution_sums_to_one_and_matches_independent_binomial(self):
        d = pl.hit_distribution([0.6] * 4, np.eye(4), n=200_000)
        self.assertAlmostEqual(sum(d), 1.0, places=6)
        self.assertAlmostEqual(d[4], 0.6 ** 4, delta=0.004)
        self.assertAlmostEqual(d[3], 4 * 0.6 ** 3 * 0.4, delta=0.006)

    def test_positive_correlation_raises_the_all_hit_chance(self):
        R = np.full((4, 4), 0.4)
        np.fill_diagonal(R, 1.0)
        self.assertGreater(pl.hit_distribution([0.6] * 4, R, n=200_000)[4], 0.6 ** 4 + 0.01)

    def test_evs_use_the_payout_tables(self):
        dist = [0.0, 0.0, 0.0, 0.0, 1.0]
        ev = pl.pickem_evs(dist)
        self.assertAlmostEqual(ev["standard"]["ev"], 9.0)
        self.assertAlmostEqual(ev["flex"]["ev"], 5.0)
        half = pl.pickem_evs([0.5, 0, 0, 0, 0.5])
        self.assertAlmostEqual(half["standard"]["ev"], 4.0)
        self.assertIsNone(pl.pickem_evs([1, 0, 0, 0, 0, 0, 0])["standard"])      # no 6-pick table: nothing invented

    def test_breakeven_per_leg(self):
        self.assertAlmostEqual(pl.pickem_breakeven(4), 0.1 ** 0.25, places=6)
        self.assertAlmostEqual(pl.pickem_breakeven(5), 0.05 ** 0.2, places=6)

    def test_legs_need_agreement_and_a_chance_and_skip_thin_injured_and_model_only(self):
        props = [_pk_prop("a"), _pk_prop("b", agree=False), _pk_prop("c", prob=0.50), _pk_prop("d", thin=True), _pk_prop("e", injury="Questionable"),
                 _pk_prop("f", model_only=True)]
        legs = pl.pickem_legs(props, [])
        self.assertEqual([l["player_id"] for l in legs], ["a"])
        relaxed = pl.pickem_legs(props, [], require_agree=False)
        self.assertEqual({l["player_id"] for l in relaxed}, {"a", "b"})

    def test_volume_markets_get_the_bigger_script_bonus(self):
        game = {"game": "ZZZ @ Ta", "home": "Ta", "away": "ZZZ", "home_spread": -7.0, "total": 44.0}      # Ta favored by 7
        run = _pk_prop("a", market="player_rush_attempts", team="Ta", game="ZZZ @ Ta", pickem={"line": 17.5, "book": "Underdog", "side": "over", "prob": 0.6,
                                                                                            "sportsbook_prob": 0.6, "line_edge": 0.0, "agree": True, "grade": "high"})
        yds = _pk_prop("b", market="player_rush_yds", team="Ta", game="ZZZ @ Ta", pickem=dict(run["pickem"]))
        legs = {l["player_id"]: l for l in pl.pickem_legs([run, yds], [game])}
        self.assertEqual(legs["a"]["script"], 1)
        self.assertTrue(legs["a"]["volume"])
        self.assertFalse(legs["b"]["volume"])
        self.assertGreater(legs["a"]["score"], legs["b"]["score"])

    def test_build_pickem_returns_confidence_then_upside_entries(self):
        props = [_pk_prop(f"p{i}", prob=0.68 - 0.01 * i) for i in range(12)]
        legs = pl.pickem_legs(props, [])
        out = pl.build_pickem(legs, CORR)
        self.assertEqual(len(out["entries"]), 4)
        self.assertEqual({e["label"] for e in out["entries"]}, {"High confidence", "High upside"})
        for e in out["entries"]:
            self.assertEqual(e["n_legs"], 4 if e["label"] == "High confidence" else 5)
            self.assertEqual(len({l["player_id"] for l in e["legs"]}), e["n_legs"])
            self.assertAlmostEqual(sum(e["hit_distribution"]), 1.0, places=2)
            self.assertIn(e["best_type"], ("standard", "flex"))
        probs = [e["avg_leg_prob"] for e in out["entries"]]
        self.assertEqual(probs, sorted(probs, reverse=True))                 # sorted by confidence
        ids = [{l["id"] for l in e["legs"]} for e in out["entries"]]
        for i in range(4):
            for j in range(i + 1, 4):
                self.assertLessEqual(len(ids[i] & ids[j]), pl.MAX_SHARED_LEGS)
        json.dumps(out)

    def test_build_all_adds_a_pickem_block(self):
        props = [_pk_prop(f"p{i}", prob=0.62) for i in range(10)]
        out = pl.build_all(props, [], None, CORR, None, 2026, 8)
        self.assertEqual(len(out["pickem"]["entries"]), 4)
        json.dumps(out)
        self.assertEqual(pl.build_all([], [], None, CORR, None, 2026, 8)["pickem"]["entries"], [])


class TestPickemTracking(unittest.TestCase):
    def _block(self, entry):
        legs = [{"id": f"p{i}|player_receptions|over", "kind": "pickem", "player_id": f"p{i}", "player": f"P{i}", "team": "T", "opp": "O", "game": "O @ T",
                 "market": "player_receptions", "label": "Receptions", "side": "over", "line": 4.5, "prob": 0.6, "week": 8} for i in range(4)]
        e = {"legs": legs, "joint_prob": 0.13, "ev": -0.1, "best_type": entry, "standard": {"multiplier": 10.0}}
        return {"pickem": {"entries": [e]}}

    def _grade(self, entry, results):
        rows = pt.rows_from_block(self._block(entry), 2026, 8, "2026-11-01T12:00:00+00:00")
        path = pathlib.Path(tempfile.mkdtemp()) / "pl.csv"
        pt.log_parlays(rows, path)
        df = pt.load(path)
        log = pd.DataFrame([{"season": 2026, "week": 8, "player_id": f"p{i}", "market": "player_receptions", "result": r, "actual": a}
                            for i, (r, a) in enumerate(results)])
        return pt.grade_parlay(next(df.itertuples()), log, pd.DataFrame())

    def test_rows_carry_kind_and_entry(self):
        r = pt.rows_from_block(self._block("flex"), 2026, 8, "t")[0]
        self.assertEqual((r["kind"], r["entry"], r["n_legs"]), ("pickem", "flex", 4))

    def test_standard_entry_needs_every_pick(self):
        hit = [("over", 6)] * 4
        self.assertEqual(self._grade("standard", hit)["profit"], 9.0)
        miss = [("over", 6)] * 3 + [("under", 2)]
        self.assertEqual(self._grade("standard", miss)["status"], "lost")

    def test_flex_entry_pays_for_one_miss(self):
        g = self._grade("flex", [("over", 6)] * 3 + [("under", 2)])
        self.assertEqual(g["status"], "cashed")
        self.assertAlmostEqual(g["profit"], 0.5)
        self.assertEqual(self._grade("flex", [("over", 6)] * 2 + [("under", 2)] * 2)["status"], "lost")

    def test_a_dnp_pick_shrinks_the_entry(self):
        g = self._grade("standard", [("over", 6)] * 3 + [("dnp", None)])
        self.assertEqual(g["status"], "cashed")
        self.assertEqual(g["profit"], 5.0)                  # a 3-pick standard entry pays 6x

    def test_pending_until_every_pick_is_graded(self):
        g = self._grade("standard", [("over", 6)] * 3)
        self.assertEqual(g["status"], "pending")


# ---- credits -----------------------------------------------------------------------------------------
class TestBudget(unittest.TestCase):
    def test_budget_spreads_what_is_left_over_the_weeks_left(self):
        b = ep.props_budget(500, 13, 13, today=datetime.date(2026, 11, 1))
        self.assertGreater(b, 60)
        self.assertLess(b, 500)
        late = ep.props_budget(500, 13, 13, today=datetime.date(2026, 11, 26))      # one game week left: nearly everything is spendable
        self.assertGreater(late, b)

    def test_a_small_run_gets_a_small_share(self):
        sunday = ep.props_budget(300, 13, 14, today=datetime.date(2026, 11, 8))
        thursday = ep.props_budget(300, 1, 14, today=datetime.date(2026, 11, 8))
        self.assertGreater(sunday, thursday)

    def test_budget_never_dips_into_the_reserve_or_goes_negative(self):
        self.assertEqual(ep.props_budget(5, 13, 13, today=datetime.date(2026, 11, 8)), 0)
        self.assertLessEqual(ep.props_budget(100, 13, 13, today=datetime.date(2026, 11, 30)), 100 - ep.DEFAULT_RESERVE - ep.GAME_LINES_COST)
        self.assertIsNone(ep.props_budget(None, 13, 13))

    def test_december_rolls_into_january(self):
        self.assertIsNotNone(ep.props_budget(300, 5, 5, today=datetime.date(2026, 12, 20)))

    def test_markets_follow_priority_and_the_budget(self):
        pri = ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8"]
        self.assertEqual(ep.choose_markets_by_budget(pri, 52, 13)[0], pri[:4])           # 52 credits / 13 games = 4 markets
        self.assertEqual(ep.choose_markets_by_budget(pri, 500, 13)[0], pri)
        self.assertEqual(len(ep.choose_markets_by_budget(pri, 10, 13)[0]), 3)            # the floor; games are dropped instead
        self.assertEqual(ep.choose_markets_by_budget(pri, None, 13)[0], pri)

    def test_plan_pull_respects_the_run_budget(self):
        evs = [{"id": i, "commence_time": f"2026-11-0{i + 1}T18:00:00Z"} for i in range(8)]
        pull, skip, note = ep.plan_props_pull(evs, 5, 400, budget=20)
        self.assertEqual(len(pull), 4)
        self.assertEqual(len(skip), 4)
        self.assertIn("budgeted", note)
        pull, skip, _ = ep.plan_props_pull(evs, 5, 400)
        self.assertEqual(len(pull), 8)

    def test_default_books_stay_within_one_region(self):
        self.assertLessEqual(len(ep.DEFAULT_BOOKS), 10)
        self.assertIn("underdog", ep.DEFAULT_BOOKS)


# ---- review sections -------------------------------------------------------------------------------------
def _log_row(pid, week, line, actual, model_over, market="player_receptions", top_rank=None, pick=None, price=None, season=2026):
    res = "over" if actual > line else "under" if actual < line else "push"
    return {"season": season, "week": week, "player_id": pid, "player": pid.upper(), "market": market, "pos": "WR", "team": "T", "line": line,
            "actual": actual, "result": res, "model_over": model_over, "top_rank": top_rank, "pick": pick, "pick_price": price}


class TestReview(unittest.TestCase):
    def test_top10_records_profit_and_season(self):
        log = pd.DataFrame([
            _log_row("a", 7, 4.5, 6, 0.62, top_rank=1, pick="over", price=-110),
            _log_row("b", 7, 4.5, 3, 0.62, top_rank=2, pick="over", price=-110),
            _log_row("c", 6, 2.5, 4, 0.60, top_rank=1, pick="over", price=150),
            _log_row("d", 7, 5.5, 9, 0.30),                                   # not a top pick
        ])
        s = rx.top10_section(log, 2026, 7)
        self.assertEqual((s["week"]["wins"], s["week"]["losses"]), (1, 1))
        self.assertAlmostEqual(s["week"]["profit"], 100 / 110 - 1, places=3)
        self.assertEqual(s["season"]["n"], 3)
        self.assertEqual(s["weeks"], 2)
        self.assertEqual([i["rank"] for i in s["items"]], [1, 2])
        self.assertIn("1-1", s["writeup"])
        self.assertIn("Season to date", s["writeup"])

    def test_top10_is_none_without_ranks(self):
        log = pd.DataFrame([_log_row("a", 7, 4.5, 6, 0.62)])
        self.assertIsNone(rx.top10_section(log, 2026, 7))
        self.assertIsNone(rx.top10_section(pd.DataFrame([{"season": 2026, "week": 7, "market": "player_receptions", "result": "over"}]), 2026, 7))

    def test_streak_counts_consecutive_correct_leans(self):
        rows = [_log_row("a", w, 4.5, 6, 0.62) for w in (3, 4, 5, 6)]            # over leans, all hit
        rows += [_log_row("b", w, 4.5, 6, 0.62) for w in (3, 4)] + [_log_row("b", 5, 4.5, 2, 0.62), _log_row("b", 6, 4.5, 6, 0.62)]   # b broke at 5
        s = rx.streaks(pd.DataFrame(rows), 2026, 6)
        names = {x["player"]: x["length"] for x in s["current"]}
        self.assertEqual(names, {"A": 4})
        self.assertEqual(s["n_series"], 2)
        self.assertIn("4 in a row", s["writeup"])

    def test_a_streak_must_run_through_the_report_week(self):
        rows = [_log_row("a", w, 4.5, 6, 0.62) for w in (1, 2, 3)]
        self.assertEqual(rx.streaks(pd.DataFrame(rows), 2026, 8)["current"], [])

    def test_pushes_and_dnps_do_not_count_or_break_a_streak(self):
        rows = [_log_row("a", w, 4.5, 6, 0.62) for w in (3, 4, 6)] + [_log_row("a", 5, 4.5, 4.5, 0.62)]   # week 5 pushed
        s = rx.streaks(pd.DataFrame(rows), 2026, 6)
        self.assertEqual(s["current"][0]["length"], 3)

    def test_under_lean_streaks(self):
        rows = [_log_row("a", w, 50.5, 30, 0.35, market="player_reception_yds") for w in (4, 5, 6)]
        s = rx.streaks(pd.DataFrame(rows), 2026, 6)
        self.assertEqual(s["current"][0]["games"][0]["side"], "under")
        self.assertEqual(s["current"][0]["label"], "Rec Yds")

    def test_empty_logs_do_not_crash(self):
        empty = pd.DataFrame(columns=["season", "week", "player_id", "player", "market", "pos", "team", "line", "actual", "result", "model_over", "top_rank"])
        self.assertEqual(rx.streaks(empty, 2026, 8)["current"], [])
        self.assertIsNone(rx.top10_section(empty, 2026, 8))


class TestTargetTracking(unittest.TestCase):
    def setUp(self):
        self.path = pathlib.Path(tempfile.mkdtemp()) / "t.csv"
        t = [{"player_id": f"p{i}", "player": f"P{i}", "team": "T", "pos": "WR", "game": "O @ T", "mean": 6.0, "sd": 2.5, "line": 5.5, "target_rate": 5.0}
             for i in range(4)]
        self.rows = tt.rows_from_snapshot(t, 2026, 8, "2026-11-01T12:00:00+00:00")

    def test_log_is_an_upsert_and_keeps_grades(self):
        self.assertEqual(tt.log_projections(self.rows, self.path), 4)
        df = tt.load(self.path)
        df.loc[0, "actual"] = 7.0
        df.to_csv(self.path, index=False)
        tt.log_projections(self.rows, self.path)
        df = tt.load(self.path)
        self.assertEqual(len(df), 4)
        self.assertEqual(df.loc[df["player_id"] == "p0", "actual"].iloc[0], 7.0)

    def test_grade_fills_actuals_from_player_stats(self):
        tt.log_projections(self.rows, self.path)
        ps = pd.DataFrame([{"season": 2026, "week": 8, "season_type": "REG", "player_id": f"p{i}", "targets": 4 + i} for i in range(3)])
        self.assertEqual(tt.grade(ps, self.path), 3)
        df = tt.load(self.path)
        self.assertEqual(df["actual"].notna().sum(), 3)                    # p3 did not play: stays blank

    def test_accuracy_compares_with_the_recent_average(self):
        tt.log_projections(self.rows, self.path)
        ps = pd.DataFrame([{"season": 2026, "week": 8, "season_type": "REG", "player_id": f"p{i}", "targets": a} for i, a in enumerate([6, 7, 5, 6])])
        tt.grade(ps, self.path)
        acc = tt.accuracy(tt.load(self.path))
        self.assertEqual(acc["n"], 4)
        self.assertAlmostEqual(acc["mae"], 0.5)
        self.assertAlmostEqual(acc["mae_recent_avg"], 1.0)
        sec = tt.section(2026, 8, self.path)
        self.assertIn("better", sec["writeup"])

    def test_section_is_none_with_nothing_graded(self):
        tt.log_projections(self.rows, self.path)
        self.assertIsNone(tt.section(2026, 8, self.path))


if __name__ == "__main__":
    unittest.main()
