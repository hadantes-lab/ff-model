"""
Tests for parlays.py (probability math, correlation, legs, prop/moneyline parlays, Wong teasers), parlay_track.py
(logging and grading), the Odds API moneyline parsing, and the credit-aware props pull.

Run:  python -m unittest discover -s tests -v
"""

import json
import math
import pathlib
import tempfile
import unittest

import numpy as np
import pandas as pd

import export_props as ep
import odds_api
import parlay_track as pt
import parlays as pl
import weekly_report as wr


# ---- math ----------------------------------------------------------------------------------
class TestMath(unittest.TestCase):
    def test_odds_conversions_round_trip(self):
        for a in (-110, -250, 100, 150, 300, -105):
            self.assertAlmostEqual(pl.dec(a), pl.dec(pl.to_american(pl.dec(a))), places=1)
        self.assertAlmostEqual(pl.dec(-110), 1.9091, places=3)
        self.assertEqual(pl.to_american(2.5), 150)
        self.assertEqual(pl.to_american(1.5), -200)
        self.assertEqual(pl.fmt_american(150), "+150")
        self.assertEqual(pl.fmt_american(-110), "-110")

    def test_normal_cdf_matches_known_values(self):
        for x, want in ((0, 0.5), (1.96, 0.975), (-1.0, 0.1587), (3.0, 0.99865)):
            self.assertAlmostEqual(float(pl.ncdf(x)), want, places=4)

    def test_bivariate_normal_against_exact_values(self):
        self.assertAlmostEqual(pl.bvn_cdf(0, 0, 0.5), 1 / 3, places=3)               # 1/4 + asin(rho)/(2 pi)
        self.assertAlmostEqual(pl.bvn_cdf(0, 0, -0.5), 1 / 6, places=3)
        self.assertAlmostEqual(pl.bvn_cdf(0.3, -0.2, 0.0), float(pl.ncdf(0.3) * pl.ncdf(-0.2)), places=6)
        self.assertAlmostEqual(pl.bvn_cdf(9, 9, 0.7), 1.0, places=4)

    def test_pair_lift_is_above_one_for_positive_and_below_for_negative_correlation(self):
        self.assertGreater(pl.pair_lift(0.6, 0.6, 0.4), 1.0)
        self.assertLess(pl.pair_lift(0.6, 0.6, -0.4), 1.0)
        self.assertEqual(pl.pair_lift(0.6, 0.6, 0.0), 1.0)

    def test_joint_probability_independent_correlated_and_degenerate(self):
        self.assertAlmostEqual(pl.joint_probability([0.6, 0.6, 0.6], np.eye(3)), 0.216, delta=0.004)
        pos = pl.joint_probability([0.6, 0.6], [[1, .5], [.5, 1]])
        neg = pl.joint_probability([0.6, 0.6], [[1, -.5], [-.5, 1]])
        self.assertGreater(pos, 0.36)
        self.assertLess(neg, 0.36)
        self.assertEqual(pl.joint_probability([0.7], [[1.0]]), 0.7)
        # an impossible correlation matrix is repaired rather than crashing
        bad = [[1, .99, -.99], [.99, 1, .99], [-.99, .99, 1]]
        self.assertTrue(0 <= pl.joint_probability([.6, .6, .6], bad) <= 1)

    def test_joint_probability_is_deterministic(self):
        R = [[1, .3], [.3, 1]]
        self.assertEqual(pl.joint_probability([.55, .6], R), pl.joint_probability([.55, .6], R))


# ---- correlation -----------------------------------------------------------------------------------
def _leg(pid="p1", team="AAA", game="BBB @ AAA", market="player_pass_yds", pos="QB", side="over", price=-110, prob=0.6, **kw):
    d = {"id": f"{pid}|{market}|{side}", "kind": "prop", "player_id": pid, "player": pid.upper(), "team": team, "opp": "BBB" if team == "AAA" else "AAA",
         "game": game, "market": market, "label": market, "pos": pos, "side": side, "line": 50.5, "price": price, "prob": prob, "script": 0,
         "script_note": "", "score": prob - 0.5, "edge": 0.0, "quality": 0.5}
    d.update(kw)
    return d


CORR = pl.CorrModel({"teammates|pass_yds/QB|reception_yds/WRTE": {"rho": 0.3, "n": 1000},
                     "teammates|pass_yds/QB|rush_yds/RB": {"rho": -0.1, "n": 800},
                     "same_player|reception_yds/WRTE|receptions/WRTE": {"rho": 0.76, "n": 900},
                     "opponents|rush_yds/RB|rush_yds/RB": {"rho": -0.12, "n": 600}})


class TestCorrModel(unittest.TestCase):
    qb = _leg("qb", market="player_pass_yds", pos="QB")
    wr_ = _leg("wr", market="player_reception_yds", pos="WR")
    rb = _leg("rb", market="player_rush_yds", pos="RB")

    def test_teammates_and_the_role_collapse_wr_te(self):
        self.assertAlmostEqual(CORR.rho(self.qb, self.wr_), 0.3)
        te = _leg("te", market="player_reception_yds", pos="TE")
        self.assertAlmostEqual(CORR.rho(self.qb, te), 0.3)                             # WR and TE share a role
        self.assertAlmostEqual(CORR.rho(self.wr_, self.qb), 0.3)                       # order does not matter

    def test_an_under_flips_the_sign_and_two_unders_flip_back(self):
        under_wr = _leg("wr", market="player_reception_yds", pos="WR", side="under")
        under_qb = _leg("qb", market="player_pass_yds", pos="QB", side="under")
        self.assertAlmostEqual(CORR.rho(self.qb, under_wr), -0.3)
        self.assertAlmostEqual(CORR.rho(under_qb, under_wr), 0.3)

    def test_negative_script_correlation_and_opponents(self):
        self.assertAlmostEqual(CORR.rho(self.qb, self.rb), -0.1)
        rb_opp = _leg("rb2", team="BBB", market="player_rush_yds", pos="RB")
        self.assertAlmostEqual(CORR.rho(self.rb, rb_opp), -0.12)

    def test_different_games_and_unknown_pairs_are_independent(self):
        other = _leg("wr2", team="CCC", game="DDD @ CCC", market="player_reception_yds", pos="WR")
        self.assertEqual(CORR.rho(self.qb, other), 0.0)
        wr_rush = _leg("wr3", market="player_rush_yds", pos="WR")
        self.assertEqual(CORR.rho(self.qb, wr_rush), 0.0)

    def test_same_player_is_recognized(self):
        a = _leg("wr", market="player_reception_yds", pos="WR")
        b = _leg("wr", market="player_receptions", pos="WR")
        self.assertAlmostEqual(CORR.rho(a, b), 0.76)

    def test_no_file_means_no_correlation(self):
        self.assertEqual(pl.CorrModel.load(pathlib.Path(tempfile.mkdtemp()) / "none.json").rho(self.qb, self.wr_), 0.0)

    def test_the_shipped_file_has_the_headline_relationships(self):
        shipped = pl.CorrModel.load()
        self.assertGreater(shipped.rho(self.qb, self.wr_), 0.2)                         # QB yards with his receiver's yards
        self.assertLess(shipped.rho(self.qb, self.rb), 0.0)                             # ...and against his RB's rushing


class TestEstimateCorrelations(unittest.TestCase):
    def _pairs(self, rho, n, seed=0, rel="teammates"):
        rng = np.random.default_rng(seed)
        z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n)
        return pd.DataFrame({"rel": rel, "ka": [("player_pass_yds", "QB")] * n, "kb": [("player_reception_yds", "WR")] * n, "za": z[:, 0], "zb": z[:, 1]})

    def test_recovers_a_known_correlation(self):
        est = pl.estimate_correlations(self._pairs(0.4, 3000))
        self.assertAlmostEqual(est["teammates|pass_yds/QB|reception_yds/WRTE"]["rho"], 0.4, delta=0.04)

    def test_noise_is_shrunk_toward_zero_and_tiny_groups_are_dropped(self):
        weak = pl.estimate_correlations(self._pairs(0.0, 400, seed=3))
        self.assertLess(abs(weak["teammates|pass_yds/QB|reception_yds/WRTE"]["rho"]), 0.08)
        self.assertEqual(pl.estimate_correlations(self._pairs(0.5, 100)), {})                # under min_n

    def test_non_finite_values_do_not_break_it(self):
        d = self._pairs(0.4, 500)
        d.loc[0, "za"] = np.nan
        d.loc[1, "zb"] = np.inf
        self.assertEqual(len(pl.estimate_correlations(d)), 1)


# ---- script and quality -----------------------------------------------------------------------------
class TestScriptAndQuality(unittest.TestCase):
    def test_rushing_follows_the_lead_and_passing_follows_the_deficit(self):
        self.assertEqual(pl.script_alignment("player_rush_yds", "over", +6)[0], 1)
        self.assertEqual(pl.script_alignment("player_rush_yds", "over", -6)[0], -1)
        self.assertEqual(pl.script_alignment("player_rush_yds", "under", -6)[0], 1)
        self.assertEqual(pl.script_alignment("player_pass_yds", "over", -5)[0], 1)
        self.assertEqual(pl.script_alignment("player_pass_yds", "over", +9)[0], -1)
        self.assertEqual(pl.script_alignment("player_pass_yds", "under", +9)[0], 1)

    def test_small_margins_are_neutral_and_high_totals_help_overs(self):
        self.assertEqual(pl.script_alignment("player_rush_yds", "over", 1)[0], 0)
        self.assertEqual(pl.script_alignment("player_receptions", "over", 0, total=50)[0], 1)
        self.assertEqual(pl.script_alignment("player_receptions", "under", 0, total=38)[0], 1)
        self.assertEqual(pl.script_alignment("player_receptions", "over", 0, total=38)[0], 0)
        self.assertTrue(pl.script_alignment("player_rush_yds", "over", 6)[1])                # carries an explanation

    def test_market_quality_is_a_shrunk_hit_rate_of_the_high_confidence_picks(self):
        s = pd.DataFrame([{"season": 2026, "group": "player_rush_yds:high", "n": 100, "wins": 70},
                          {"season": 2026, "group": "player_pass_yds:high", "n": 100, "wins": 30},
                          {"season": 2026, "group": "player_rush_yds:all", "n": 100, "wins": 99},
                          {"season": 2025, "group": "player_receptions:high", "n": 100, "wins": 100}])
        q = pl.market_quality(s, 2026)
        self.assertAlmostEqual(q["player_rush_yds"], (70 + 10) / 120)
        self.assertLess(q["player_pass_yds"], 0.5)
        self.assertNotIn("player_receptions", q)                                           # a different season
        self.assertEqual(pl.market_quality(None), {})
        self.assertEqual(pl.quality_factor(0.0), 0.5)
        self.assertEqual(pl.quality_factor(1.0), 1.3)


# ---- prop legs -----------------------------------------------------------------------------------------
def _prop(pid="p1", market="player_receptions", over=0.62, under=0.38, raw=0.7, team="AAA", game="BBB @ AAA", price_o=-130, price_u=110, **kw):
    d = {"player_id": pid, "player": pid.upper(), "team": team, "opp": "BBB", "game": game, "pos": "WR", "market": market, "label": "Rec",
         "kind": "count", "line": 4.5, "over": over, "under": under, "model_over": raw, "market_over": 0.6, "thin": False, "injury": None,
         "best_over": {"price": price_o, "book": "X"}, "best_under": {"price": price_u, "book": "Y"}}
    d.update(kw)
    return d


GAMES = [{"game": "BBB @ AAA", "home": "AAA", "away": "BBB", "home_spread": -7.0, "total": 49.0, "in_window": True}]


class TestPropLegs(unittest.TestCase):
    def test_the_more_likely_side_at_the_best_price_becomes_the_leg(self):
        leg = pl.prop_legs([_prop()], GAMES)[0]
        self.assertEqual((leg["side"], leg["price"], leg["book"]), ("over", -130, "X"))
        self.assertAlmostEqual(leg["edge"], 0.62 * pl.dec(-130) - 1)

    def test_filters(self):
        cases = [_prop(thin=True), _prop(injury="Questionable"), _prop(kind="td"), _prop(raw=0.4),       # the model leans the other way
                 _prop(price_o=-400), _prop(over=0.505, under=0.495, raw=0.7)]                             # below the floor
        self.assertEqual(pl.prop_legs(cases, GAMES), [])
        self.assertEqual(pl.prop_legs([_prop(best_over=None, best_under=None)], GAMES), [])              # no price on either side

    def test_a_better_market_and_a_fitting_script_rank_higher(self):
        good = _prop("a", market="player_rush_yds", over=0.60, raw=0.7)
        bad = _prop("b", market="player_pass_yds", over=0.60, raw=0.7)
        legs = pl.prop_legs([bad, good], GAMES, {"player_rush_yds": 0.65, "player_pass_yds": 0.35})
        self.assertEqual(legs[0]["player_id"], "a")

    def test_script_tag_uses_the_teams_own_margin(self):
        home = pl.prop_legs([_prop("h", market="player_rush_yds", team="AAA")], GAMES)[0]       # AAA is favored by 7: leans on the run
        away = pl.prop_legs([_prop("v", market="player_rush_yds", team="BBB")], GAMES)[0]       # BBB is the underdog
        self.assertEqual(home["script"], 1)
        self.assertEqual(away["script"], -1)


# ---- prop parlays --------------------------------------------------------------------------------------
def _pool(n_games=6, per_game=2):
    legs = []
    for g in range(n_games):
        for k in range(per_game):
            legs.append(_leg(f"p{g}{k}", team=f"T{g}", game=f"O{g} @ T{g}", market="player_receptions", pos="WR", prob=0.62 - 0.01 * (g + k), score=0.12 - 0.01 * (g + k)))
    return legs


class TestPropParlays(unittest.TestCase):
    def test_parlays_are_4_or_5_legs_plus_money_and_sorted_by_edge(self):
        r = pl.build_prop_parlays(_pool(), pl.CorrModel({}))
        self.assertTrue(r["parlays"])
        for p in r["parlays"]:
            self.assertIn(p["n_legs"], (4, 5))
            self.assertGreaterEqual(p["decimal"], pl.MIN_PARLAY_DECIMAL)
            self.assertEqual(p["kind"], "props")
        evs = [p["ev"] for p in r["parlays"]]
        self.assertEqual(evs, sorted(evs, reverse=True))

    def test_one_leg_per_player_and_at_most_two_per_game(self):
        r = pl.build_prop_parlays(_pool(n_games=3, per_game=4), pl.CorrModel({}))
        for p in r["parlays"]:
            self.assertEqual(len({l["player_id"] for l in p["legs"]}), p["n_legs"])
            games = [l["game"] for l in p["legs"]]
            self.assertLessEqual(max(games.count(g) for g in set(games)), pl.MAX_PER_GAME)

    def test_negatively_correlated_same_game_legs_are_never_paired(self):
        qb = _leg("qb", market="player_pass_yds", pos="QB", prob=0.65, score=0.2)
        rb = _leg("rb", market="player_rush_yds", pos="RB", prob=0.65, score=0.2)             # corr(QB yds, RB yds) = -0.1 in CORR
        legs = [qb, rb] + [_leg(f"x{i}", team=f"T{i}", game=f"O{i} @ T{i}", market="player_receptions", pos="WR", prob=0.6, score=0.1) for i in range(6)]
        r = pl.build_prop_parlays(legs, CORR)
        for p in r["parlays"]:
            ids = {l["player_id"] for l in p["legs"]}
            self.assertFalse({"qb", "rb"} <= ids)

    def test_positive_correlation_lifts_the_joint_probability_above_the_naive_product(self):
        qb = _leg("qb", market="player_pass_yds", pos="QB", prob=0.6)
        wr_ = _leg("wr", market="player_reception_yds", pos="WR", prob=0.6)
        others = [_leg(f"x{i}", team=f"T{i}", game=f"O{i} @ T{i}", market="player_receptions", pos="WR", prob=0.55, score=0.05) for i in range(3)]
        r = pl.build_prop_parlays([qb, wr_] + others, CORR, sizes=(4,))
        both = [p for p in r["parlays"] if {"qb", "wr"} <= {l["player_id"] for l in p["legs"]}]
        self.assertTrue(both)
        self.assertGreater(both[0]["joint_prob"], both[0]["naive_prob"] * 1.02)
        self.assertGreater(both[0]["correlation_lift"], 1.02)
        self.assertEqual(len(both[0]["same_game"]), 1)

    def test_ev_is_joint_probability_times_payout_minus_one(self):
        p = pl.build_prop_parlays(_pool(), pl.CorrModel({}))["parlays"][0]
        self.assertAlmostEqual(p["ev"], p["joint_prob"] * p["decimal"] - 1, places=6)
        self.assertAlmostEqual(p["breakeven_prob"], 1 / p["decimal"], places=6)
        self.assertAlmostEqual(p["decimal"], float(np.prod([pl.dec(l["price"]) for l in p["legs"]])), places=2)

    def test_recommended_parlays_do_not_overlap_much(self):
        r = pl.build_prop_parlays(_pool(8), pl.CorrModel({}))
        sets = [{l["id"] for l in p["legs"]} for p in r["parlays"]]
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                self.assertLessEqual(len(sets[i] & sets[j]), pl.MAX_SHARED_LEGS)
        self.assertLessEqual(len(r["parlays"]), 4)

    def test_too_few_legs_gives_no_parlays_and_the_page_gets_a_reason(self):
        r = pl.build_prop_parlays(_pool(n_games=1, per_game=2), pl.CorrModel({}))
        self.assertEqual(r["parlays"], [])
        block = pl.build_all([_prop()], GAMES, None, CORR, None, 2026, 8)
        self.assertEqual(block["props"]["parlays"], [])
        self.assertIn("light slate", block["notes"][0])


# ---- moneyline parlays ---------------------------------------------------------------------------------
def _game(name, fair, home_price, away_price, home="H", away="A"):
    return {"game": name, "home": home, "away": away, "fair_home_win": fair, "best_home_ml": {"price": home_price, "book": "X"},
            "best_away_ml": {"price": away_price, "book": "Y"}}


class TestMoneylineParlays(unittest.TestCase):
    def games(self):
        return [_game(f"A{i} @ H{i}", 0.62 + 0.02 * i, -150 - 10 * i, 130 + 10 * i, home=f"H{i}", away=f"A{i}") for i in range(6)]

    def test_the_favorite_is_the_leg_at_its_best_price(self):
        legs = pl.moneyline_legs([_game("A @ H", 0.30, 250, -300, "H", "A")])
        self.assertEqual((legs[0]["team"], legs[0]["price"], legs[0]["side"]), ("A", -300, "away"))
        self.assertAlmostEqual(legs[0]["prob"], 0.70)
        self.assertAlmostEqual(legs[0]["edge"], 0.70 * pl.dec(-300) - 1)

    def test_games_without_moneylines_are_skipped(self):
        self.assertEqual(pl.moneyline_legs([{"game": "A @ H", "home": "H", "away": "A", "fair_home_win": None}]), [])

    def test_parlays_are_independent_products_in_different_games_and_varied_in_size(self):
        r = pl.build_ml_parlays(pl.moneyline_legs(self.games()))
        self.assertTrue(r["parlays"])
        self.assertEqual(len({p["n_legs"] for p in r["parlays"]}) , len(r["parlays"]))          # one of each size
        for p in r["parlays"]:
            self.assertAlmostEqual(p["joint_prob"], float(np.prod([l["prob"] for l in p["legs"]])), places=9)
            self.assertAlmostEqual(p["ev"], p["joint_prob"] * p["decimal"] - 1, places=6)
            self.assertEqual(len({l["game"] for l in p["legs"]}), p["n_legs"])
            self.assertGreaterEqual(p["decimal"], 2.0)
        sets = [{l["id"] for l in p["legs"]} for p in r["parlays"]]
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                self.assertLessEqual(len(sets[i] & sets[j]), 1)

    def test_shopping_a_better_price_raises_the_edge(self):
        base = pl.moneyline_legs([_game("A @ H", 0.40, 180, -200, "H", "A")])[0]
        shop = pl.moneyline_legs([_game("A @ H", 0.40, 180, -185, "H", "A")])[0]
        self.assertGreater(shop["edge"], base["edge"])


# ---- Wong teasers --------------------------------------------------------------------------------------
def _sched(rows):
    """rows: (season, home spread in NFLVERSE sign, home margin)."""
    return pd.DataFrame([{"season": s, "game_type": "REG", "home_score": 20 + m if m > 0 else 20, "away_score": 20 if m > 0 else 20 - m,
                          "spread_line": sp} for s, sp, m in rows])


class TestWong(unittest.TestCase):
    def test_a_teased_line_covers_by_the_right_margin(self):
        # nflverse +7.5 = home favored by 7.5 -> odds sign -7.5, teased to -1.5: covers if the home team wins by 2+
        rows = [(2024, 7.5, 2), (2024, 7.5, 1), (2024, 7.5, 10), (2024, 7.5, -3)]
        h = pl.wong_history(_sched(rows), recent_from=2024, first=2024)
        self.assertEqual(h[-7.5]["n_recent"], 4)
        self.assertAlmostEqual(h[-7.5]["rate_recent"], 0.5)                       # margins 2 and 10 cover; 1 and -3 do not

    def test_underdog_bucket_covers_unless_the_dog_loses_by_more_than_the_teased_cushion(self):
        # home underdog +2.5 teased to +8.5: covers if the home team loses by 8 or fewer (or wins)
        rows = [(2024, -2.5, -8), (2024, -2.5, -9), (2024, -2.5, 3), (2024, -2.5, -1)]
        h = pl.wong_history(_sched(rows), recent_from=2024, first=2024)
        self.assertAlmostEqual(h[2.5]["rate_recent"], 0.75)

    def test_rate_used_is_shrunk_toward_the_pooled_recent_rate(self):
        rows = [(2024, 7.5, 10)] * 3 + [(2024, -2.5, 3)] * 30 + [(2024, -2.5, -10)] * 30        # half the dog games fail to cover
        h = pl.wong_history(_sched(rows), recent_from=2024, first=2024)
        self.assertLess(h[-7.5]["rate"], 1.0)                                      # 3 covers in 3 is not 100%
        self.assertGreater(h[-7.5]["rate"], h["_pooled_recent"] - 0.01)

    def test_only_exact_buckets_become_legs_with_both_sides_checked(self):
        hist = {-7.5: {"name": "fav -7.5 to -1.5", "rate": 0.70, "n_recent": 50, "rate_recent": 0.7, "rate_all": 0.74, "n_all": 170},
                1.5: {"name": "dog +1.5 to +7.5", "rate": 0.76, "n_recent": 70, "rate_recent": 0.77, "rate_all": 0.8, "n_all": 120}}
        games = [{"game": "B @ A", "home": "A", "away": "B", "home_spread": -7.5},       # home -7.5 (bucket), away +7.5 (not a bucket)
                 {"game": "D @ C", "home": "C", "away": "D", "home_spread": -1.5},       # away +1.5 (bucket)
                 {"game": "F @ E", "home": "E", "away": "F", "home_spread": -3.0}]       # nothing
        legs = pl.wong_legs(games, hist)
        self.assertEqual({(l["team"], l["spread"]) for l in legs}, {("A", -7.5), ("D", 1.5)})
        d = [l for l in legs if l["team"] == "D"][0]
        self.assertEqual(d["teased"], 7.5)
        self.assertTrue(d["playable"])                                                  # 76% > the 73.9% break-even at -120
        self.assertFalse([l for l in legs if l["team"] == "A"][0]["playable"])         # 70% is not

    def test_pair_ev_math_at_both_prices(self):
        a = {"id": "a", "game": "g1", "team": "A", "prob": 0.76, "playable": True}
        b = {"id": "b", "game": "g2", "team": "B", "prob": 0.76, "playable": True}
        pr = pl.teaser_pair(a, b)
        self.assertAlmostEqual(pr["prob"], 0.5776)
        self.assertAlmostEqual(pr["ev"]["2-team at -120"], 0.5776 * (1 + 100 / 120) - 1)
        self.assertGreater(pr["ev"]["2-team at -110"], pr["ev"]["2-team at -120"])
        self.assertGreater(pr["ev"]["2-team at -120"], 0)

    def test_break_even_per_leg_is_what_the_docs_say(self):
        t = pl.build_teasers([])
        self.assertAlmostEqual(t["breakeven_leg"]["2-team at -120"], 0.739, places=3)
        self.assertAlmostEqual(t["breakeven_leg"]["2-team at -110"], 0.724, places=3)

    def test_round_robin_needs_three_playable_legs_and_covers_every_pair_once(self):
        mk = lambda i, p: {"id": f"l{i}", "game": f"g{i}", "team": f"T{i}", "prob": p, "playable": p >= 0.739}
        two = pl.build_teasers([mk(1, .77), mk(2, .76), mk(3, .70)])
        self.assertIsNone(two["round_robin"])                                           # only two clear the bar
        four = pl.build_teasers([mk(1, .78), mk(2, .77), mk(3, .76), mk(4, .75), mk(5, .70)])
        rr = four["round_robin"]
        self.assertEqual(rr["n_pairs"], 6)                                              # 4 choose 2
        self.assertEqual(len(rr["legs"]), 4)
        self.assertAlmostEqual(rr["total_ev"]["2-team at -120"], sum(p["ev"]["2-team at -120"] for p in rr["pairs"]))
        self.assertEqual(len({(p["legs"][0]["id"], p["legs"][1]["id"]) for p in rr["pairs"]}), 6)

    def test_one_side_per_game_in_pairs(self):
        a1 = {"id": "a1", "game": "g1", "team": "A", "prob": 0.8, "playable": True}
        a2 = {"id": "a2", "game": "g1", "team": "B", "prob": 0.79, "playable": True}          # the same game
        b = {"id": "b", "game": "g2", "team": "C", "prob": 0.77, "playable": True}
        pairs = pl.build_teasers([a1, a2, b])["pairs"]
        for p in pairs:
            self.assertNotEqual(p["legs"][0]["game"], p["legs"][1]["game"])


class TestBuildAll(unittest.TestCase):
    def test_every_part_degrades_without_data(self):
        out = pl.build_all([], [], None, pl.CorrModel({}), None, 2026, 8)
        self.assertEqual(out["props"]["parlays"], [])
        self.assertEqual(out["ml"]["parlays"], [])
        self.assertEqual(out["teasers"]["legs"], [])
        self.assertTrue(out["notes"])
        json.dumps(out)                                                                # and it is JSON-serializable

    def test_output_is_json_serializable_with_real_content(self):
        props = [_prop(f"p{i}", team=f"T{i}", game=f"O{i} @ T{i}") for i in range(8)]
        games = [{"game": f"O{i} @ T{i}", "home": f"T{i}", "away": f"O{i}", "home_spread": -3.0, "total": 45.0, "in_window": True, "week": 8,
                  **{k: v for k, v in _game("x", 0.6, -140, 120).items() if k.startswith(("fair", "best"))}} for i in range(8)]
        out = pl.build_all(props, games, None, CORR, _sched([(2025, 7.5, 3)]), 2026, 8)
        json.dumps(out)
        self.assertTrue(out["props"]["parlays"])
        self.assertTrue(out["ml"]["parlays"])


# ---- tracking and grading ------------------------------------------------------------------------------
def _block_props():
    legs = [_leg(f"p{i}", team=f"T{i}", game=f"O{i} @ T{i}", market="player_receptions", pos="WR", prob=0.6, price=-120, week=8) for i in range(4)]
    for l in legs:
        l["line"] = 4.5
    return {"props": {"parlays": [pl._summary(legs, 0.1296, 0.1296, "props")]}, "ml": {"parlays": []}, "teasers": {"pairs": [], "round_robin": None}}


def _props_log(results):
    """results: {player_id: (result, actual)}"""
    rows = [{"season": 2026, "week": 8, "player_id": pid, "market": "player_receptions", "result": r, "actual": a} for pid, (r, a) in results.items()]
    return pd.DataFrame(rows)


class TestParlayTracking(unittest.TestCase):
    def setUp(self):
        self.path = pathlib.Path(tempfile.mkdtemp()) / "pl.csv"
        self.rows = pt.rows_from_block(_block_props(), 2026, 8, "2026-10-25T12:00:00+00:00")
        pt.log_parlays(self.rows, self.path)
        self.log = pt.load(self.path)
        self.sched = pd.DataFrame(columns=["season", "week", "home_team", "away_team", "home_score", "away_score"])

    def grade(self, results):
        return pt.grade_parlay(next(self.log.itertuples()), _props_log(results), self.sched)

    def test_logging_upserts_and_keeps_the_first_logged_time(self):
        later = [dict(r, logged_at="2026-10-25T15:00:00+00:00", first_logged="2026-10-25T15:00:00+00:00", price=900) for r in self.rows]
        pt.log_parlays(later, self.path)
        d = pt.load(self.path)
        self.assertEqual(len(d), 1)
        self.assertEqual(d.iloc[0]["first_logged"], "2026-10-25T12:00:00+00:00")
        self.assertEqual(int(d.iloc[0]["price"]), 900)                                   # the latest price wins

    def test_all_legs_over_cashes_at_the_product_of_prices(self):
        g = self.grade({f"p{i}": ("over", 6.0) for i in range(4)})
        self.assertEqual((g["status"], g["legs_won"]), ("cashed", 4))
        self.assertAlmostEqual(g["profit"], pl.dec(-120) ** 4 - 1, places=3)

    def test_one_miss_loses_even_while_others_are_pending(self):
        g = self.grade({"p0": ("under", 2.0), "p1": ("over", 6.0)})
        self.assertEqual((g["status"], g["profit"]), ("lost", -1.0))

    def test_unsettled_legs_keep_it_pending(self):
        g = self.grade({"p0": ("over", 6.0), "p1": ("over", 6.0), "p2": (np.nan, np.nan)})
        self.assertEqual(g["status"], "pending")
        self.assertEqual(g["legs_pending"], 2)                                           # p2 unsettled, p3 not in the log yet

    def test_a_push_or_dnp_leg_is_dropped_and_the_payout_repriced(self):
        g = self.grade({"p0": ("over", 6.0), "p1": ("over", 6.0), "p2": ("push", 4.5), "p3": ("dnp", np.nan)})
        self.assertEqual((g["status"], g["legs_won"], g["legs_dropped"]), ("cashed", 2, 2))
        self.assertAlmostEqual(g["profit"], pl.dec(-120) ** 2 - 1, places=3)

    def test_everything_voided_is_a_push(self):
        g = self.grade({f"p{i}": ("push", 4.5) for i in range(4)})
        self.assertEqual((g["status"], g["profit"]), ("push", 0.0))

    def test_moneyline_and_teaser_legs_are_graded_from_the_final_score(self):
        sched = pd.DataFrame([{"season": 2026, "week": 8, "home_team": "H", "away_team": "A", "home_score": 24, "away_score": 20}])
        ml = {"kind": "ml", "game": "A @ H", "team": "H", "price": -150}
        self.assertEqual(pt.grade_leg(ml, 2026, 8, pd.DataFrame(), sched), "win")
        self.assertEqual(pt.grade_leg(dict(ml, team="A"), 2026, 8, pd.DataFrame(), sched), "loss")
        # H won by 4: an away +2.5 teased to +8.5 covers (loses by 4); a home -7.5 teased to -1.5 also covers (wins by 4 > 1.5)
        self.assertEqual(pt.grade_leg({"kind": "wong", "game": "A @ H", "team": "A", "spread": 2.5}, 2026, 8, pd.DataFrame(), sched), "win")
        self.assertEqual(pt.grade_leg({"kind": "wong", "game": "A @ H", "team": "H", "spread": -7.5}, 2026, 8, pd.DataFrame(), sched), "win")
        # a home -8.5 teased to -2.5 wins at +4 too, but a home -7.5 teased to -1.5 fails on a 1-point win
        close = sched.assign(home_score=21, away_score=20)
        self.assertEqual(pt.grade_leg({"kind": "wong", "game": "A @ H", "team": "H", "spread": -7.5}, 2026, 8, pd.DataFrame(), close), "loss")
        self.assertEqual(pt.grade_leg(ml, 2026, 8, pd.DataFrame(), sched.assign(home_score=np.nan)), "pending")

    def test_section_and_report_integration(self):
        log = _props_log({f"p{i}": ("over" if i < 3 else "under", 6.0 if i < 3 else 2.0) for i in range(4)})
        sec = pt.section(self.log, log, self.sched, 2026, 8)
        k = sec["groups"]["props"]
        self.assertEqual((k["n"], k["lost"], k["legs_won"], k["legs_graded"]), (1, 1, 3, 4))
        self.assertAlmostEqual(k["expected_legs"], 4 * 0.6)
        self.assertAlmostEqual(k["profit"], -1.0)
        self.assertIn("1 of", sec["writeup"].replace("0 of 1", "1 of"))
        self.assertIn("Individual legs: 3 of 4 hit", sec["writeup"])
        # the report carries it, summarizes it, and the season-to-date table can read it
        rep = wr.build_report(pd.DataFrame(columns=wr_cols()), pd.DataFrame(), 2026, 8, parlay_log=self.log)
        self.assertIsNotNone(rep["parlays"])
        self.assertIn("## Parlays", wr.render_markdown(rep))
        self.assertIn("Parlays", wr.render_html(rep))

    def test_empty_week_says_so(self):
        sec = pt.section(self.log, pd.DataFrame(), self.sched, 2026, 9)
        self.assertEqual(sec["n_logged"], 0)
        self.assertIn("No parlays were logged", sec["writeup"])


def wr_cols():
    return ["season", "week", "player_id", "market", "player", "team", "opp", "pos", "game", "commence", "kind", "line", "model_over",
            "model_under", "market_over", "gap", "trust", "pick", "pick_price", "pick_ev", "logged_at", "actual", "result", "graded_at"]


# ---- Odds API moneylines and the credit guard ---------------------------------------------------------
class TestMoneylineParsing(unittest.TestCase):
    def _event(self):
        def book(title, spread_h, spread_a, ml_h, ml_a):
            return {"title": title, "key": title.lower(), "markets": [
                {"key": "spreads", "outcomes": [{"name": "Home Team", "point": -3.0, "price": spread_h}, {"name": "Away Team", "point": 3.0, "price": spread_a}]},
                {"key": "totals", "outcomes": [{"name": "Over", "point": 44.5, "price": -110}, {"name": "Under", "point": 44.5, "price": -110}]},
                {"key": "h2h", "outcomes": [{"name": "Home Team", "price": ml_h}, {"name": "Away Team", "price": ml_a}]}]}
        return {"id": "e1", "home_team": "Home Team", "away_team": "Away Team", "commence_time": "2099-01-01T17:00:00Z",
                "bookmakers": [book("A", -110, -110, -150, 130), book("B", -110, -110, -145, 125)]}

    def test_best_prices_and_fair_probability(self):
        p = odds_api.consensus_game_prices([self._event()])["e1"]
        self.assertEqual(p["best_home_ml"]["price"], -145)                               # the less negative price is better
        self.assertEqual(p["best_away_ml"]["price"], 130)
        self.assertEqual(p["n_ml_books"], 2)
        self.assertTrue(0.55 < p["fair_home_win"] < 0.65)
        pa, pb = odds_api.american_to_prob(-150), odds_api.american_to_prob(130)
        self.assertAlmostEqual(p["fair_home_win"], round(((pa / (pa + pb)) + (odds_api.american_to_prob(-145) / (odds_api.american_to_prob(-145) + odds_api.american_to_prob(125)))) / 2, 4))

    def test_an_event_without_moneylines_still_prices_the_spread(self):
        e = self._event()
        for b in e["bookmakers"]:
            b["markets"] = [m for m in b["markets"] if m["key"] != "h2h"]
        p = odds_api.consensus_game_prices([e])["e1"]
        self.assertIsNone(p["fair_home_win"])
        self.assertIsNone(p["best_home_ml"])
        self.assertEqual(p["home_spread"], -3.0)


class TestCreditGuard(unittest.TestCase):
    EVENTS = [{"id": str(i), "commence_time": f"2026-10-25T{13 + i // 4:02d}:{(i % 4) * 15:02d}:00Z"} for i in range(13)][::-1]

    def test_a_short_balance_keeps_the_soonest_games_and_a_reserve(self):
        pull, skip, note = ep.plan_props_pull(self.EVENTS, 5, 64)
        self.assertEqual((len(pull), len(skip)), (10, 3))
        self.assertLessEqual(max(e["commence_time"] for e in pull), min(e["commence_time"] for e in skip))      # soonest first
        spent = len(pull) * 5 + ep.GAME_LINES_COST
        self.assertGreaterEqual(64 - spent, ep.DEFAULT_RESERVE)
        self.assertIn("reserve", note)

    def test_plenty_of_credits_pulls_everything(self):
        pull, skip, _ = ep.plan_props_pull(self.EVENTS, 5, 400)
        self.assertEqual((len(pull), len(skip)), (13, 0))

    def test_unknown_balance_pulls_everything_as_before(self):
        self.assertEqual(len(ep.plan_props_pull(self.EVENTS, 5, None)[0]), 13)

    def test_too_few_credits_pulls_nothing_not_a_negative_number(self):
        pull, skip, _ = ep.plan_props_pull(self.EVENTS, 5, 5)
        self.assertEqual((len(pull), len(skip)), (0, 13))

    def test_more_markets_cost_more_games(self):
        a = len(ep.plan_props_pull(self.EVENTS, 5, 100)[0])
        b = len(ep.plan_props_pull(self.EVENTS, 11, 100)[0])
        self.assertGreater(a, b)


if __name__ == "__main__":
    unittest.main()
