"""
Tests for team_ratings.py (power rankings, rank -> spread rule, QB adjustment), depth_chart.team_qb_out,
and the pick-ranking helpers in export_props.py.

Run:  python -m unittest discover -s tests -v
"""

import unittest

import numpy as np
import pandas as pd

import depth_chart as dc
import export_props as ep
import team_ratings as tr

TEAMS = [f"T{i:02d}" for i in range(1, 33)]


def _league(true_rating, seasons=(2025, 2026), weeks=8, seed=0, noise=10.0):
    """A synthetic league: results generated from known ratings, to check the fit recovers them."""
    rng = np.random.default_rng(seed)
    rows, gid = [], 0
    for season in seasons:
        for week in range(1, weeks + 1):
            order = rng.permutation(TEAMS)
            for h, a in zip(order[0::2], order[1::2]):
                margin = true_rating[h] - true_rating[a] + 1.9 + rng.normal(scale=noise)
                gid += 1
                rows.append({"game_id": f"g{gid}", "season": season, "week": week, "home_team": h, "away_team": a,
                             "home_score": 24 + margin / 2, "away_score": 24 - margin / 2, "location": "Home",
                             "spread_line": np.nan,
                             "epa_pp_h": 0.0, "epa_pp_a": 0.0, "epa_pp_allowed_h": 0.0, "epa_pp_allowed_a": 0.0,
                             "ypp_h": 5.0, "ypp_a": 5.0, "giveaways_h": 1, "giveaways_a": 1})
    g = pd.DataFrame(rows)
    g["t"] = (g["season"] - g["season"].min()) * tr.WEEKS_PER_SEASON + g["week"]
    return g


class TestRankRule(unittest.TestCase):
    def test_fifteenth_is_neutral_and_first_is_minus_six(self):
        self.assertAlmostEqual(tr.rank_margin(15, 15, hfa=0), 0.0)
        self.assertAlmostEqual(tr.rank_margin(1, 15, hfa=0), 6.0)         # home margin +6 == spread -6
        self.assertAlmostEqual(tr.rank_margin(15, 1, hfa=0), -6.0)

    def test_linear_and_adds_home_field(self):
        self.assertAlmostEqual(tr.rank_margin(1, 32, hfa=0), 31 * 6 / 14)
        self.assertAlmostEqual(tr.rank_margin(10, 20, hfa=2.0), 10 * 6 / 14 + 2.0)
        self.assertAlmostEqual(tr.rank_margin(10, 20, hfa=2.0, neutral=True), 10 * 6 / 14)

    def test_cover_probability_uses_the_odds_sign_convention(self):
        # home favored by 3 (spread -3): a predicted margin of exactly +3 is a coin flip
        self.assertAlmostEqual(tr.cover_prob(3.0, -3.0), 0.5)
        self.assertGreater(tr.cover_prob(10.0, -3.0), 0.5)
        self.assertLess(tr.cover_prob(-1.0, -3.0), 0.5)


class TestFit(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.true = {t: float(v) for t, v in zip(TEAMS, rng.normal(scale=4.0, size=32))}
        mean = np.mean(list(self.true.values()))
        self.true = {t: v - mean for t, v in self.true.items()}
        self.games = _league(self.true)

    def test_recovers_known_ratings_and_home_field(self):
        r, hfa = tr.fit_ratings(self.games, TEAMS, float(self.games["t"].max()) + 1, halflife=100)
        a, b = np.array([r[t] for t in TEAMS]), np.array([self.true[t] for t in TEAMS])
        self.assertGreater(np.corrcoef(a, b)[0, 1], 0.75)   # noisy synthetic league; a broken fit gives ~0
        self.assertAlmostEqual(hfa, 1.9, delta=1.2)

    def test_strength_of_schedule_is_adjusted(self):
        # a team that beats weak opponents by 7 must rate below one that beats strong ones by 7
        rows = []
        for i in range(30):
            rows.append(("STRONG_WIN", "STRONG_OPP", 7))
            rows.append(("WEAK_WIN", "WEAK_OPP", 7))
            rows.append(("STRONG_OPP", "WEAK_OPP", 14))                    # ties the two opponent groups together
        g = pd.DataFrame([{"game_id": f"g{i}", "season": 2026, "week": 1 + i % 8, "home_team": h, "away_team": a,
                           "home_score": 30 + m, "away_score": 30, "location": "Home"} for i, (h, a, m) in enumerate(rows)])
        g["t"] = g["week"].astype(float)
        teams = sorted(set(g["home_team"]) | set(g["away_team"]))
        r, _ = tr.fit_ratings(g, teams, 20.0, halflife=100, lam=0.5)
        self.assertGreater(r["STRONG_WIN"], r["WEAK_WIN"])

    def test_ratings_only_use_games_before_the_cutoff(self):
        early, _ = tr.fit_ratings(self.games, TEAMS, 3.0)
        g2 = self.games.copy()
        late = g2["t"] >= 3.0
        g2.loc[late, "home_score"] += 50                                   # wildly different "future"
        again, _ = tr.fit_ratings(g2, TEAMS, 3.0)
        self.assertEqual(early, again)

    def test_ranks_are_a_permutation_with_best_first(self):
        r, _ = tr.fit_ratings(self.games, TEAMS, 99.0)
        ranks = tr.rank_teams(r)
        self.assertEqual(sorted(ranks.values()), list(range(1, 33)))
        best = max(r, key=r.get)
        self.assertEqual(ranks[best], 1)

    def test_no_history_is_neutral_not_a_crash(self):
        r, hfa = tr.fit_ratings(self.games.iloc[0:0], TEAMS, 5.0)
        self.assertTrue(all(v == 0.0 for v in r.values()))
        self.assertEqual(hfa, tr.HOME_FIELD)


class TestPowerTableAndProjection(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2)
        true = dict(zip(TEAMS, rng.normal(scale=4.0, size=32)))
        self.games = _league(true)
        self.pt = tr.power_table(self.games, 2026, None)

    def test_table_has_every_team_ranked_once_with_changes(self):
        t = self.pt["table"]
        self.assertEqual([r["rank"] for r in t], list(range(1, 33)))
        self.assertTrue(all(r["change"] is not None for r in t))
        self.assertEqual(sum(r["change"] for r in t), 0)                   # rank moves net out
        for key in ("sos_rank", "off_rank", "def_rank"):
            self.assertEqual(sorted(r[key] for r in t), list(range(1, 33)))
        w, l = map(int, t[0]["record"].split("-")[:2])
        self.assertEqual(w + l, t[0]["games"])

    def test_project_game_applies_the_rank_rule_and_qb_adjustment(self):
        ranks = self.pt["ranks"]
        best, mid = [t for t, r in ranks.items() if r == 1][0], [t for t, r in ranks.items() if r == 15][0]
        base = tr.project_game(self.pt, best, mid, neutral=True)
        self.assertAlmostEqual(base["rank_margin"], 6.0, places=1)
        self.assertEqual(base["rank_spread"], -6.0)
        out_home = tr.project_game(self.pt, best, mid, neutral=True, qb_out_home=True)
        self.assertAlmostEqual(out_home["rank_margin"], 6.0 - tr.QB_CHANGE_POINTS, places=1)
        out_away = tr.project_game(self.pt, best, mid, neutral=True, qb_out_away=True)
        self.assertAlmostEqual(out_away["rank_margin"], 6.0 + tr.QB_CHANGE_POINTS, places=1)

    def test_rank_gap_and_cover_probability_vs_posted_spread(self):
        ranks = self.pt["ranks"]
        best, mid = [t for t, r in ranks.items() if r == 1][0], [t for t, r in ranks.items() if r == 15][0]
        g = tr.project_game(self.pt, best, mid, home_spread=-3.0, neutral=True)
        self.assertEqual(g["rank_gap"], -3.0)                              # rank says -6, market says -3
        self.assertGreater(g["rank_home_cover"], 0.5)


class TestQbOut(unittest.TestCase):
    def _hist(self):
        rows = [dict(player_id="starter", team="AAA", position="QB", season=2026, week=w, attempts=34) for w in range(1, 9)]
        rows += [dict(player_id="backup", team="AAA", position="QB", season=2026, week=w, attempts=2) for w in range(5, 9)]
        return pd.DataFrame(rows)

    def test_flags_when_the_regular_starter_is_out(self):
        self.assertTrue(dc.team_qb_out(self._hist(), "AAA", {"starter": "Out"}))
        self.assertTrue(dc.team_qb_out(self._hist(), "AAA", {"starter": "Doubtful"}))

    def test_not_flagged_when_healthy_questionable_or_only_the_backup_is_hurt(self):
        self.assertFalse(dc.team_qb_out(self._hist(), "AAA", {}))
        self.assertFalse(dc.team_qb_out(self._hist(), "AAA", {"starter": "Questionable"}))
        self.assertFalse(dc.team_qb_out(self._hist(), "AAA", {"backup": "Out"}))

    def test_unknown_team_is_false(self):
        self.assertFalse(dc.team_qb_out(self._hist(), "ZZZ", {"starter": "Out"}))


def _prop(pid, ev, kind="count", **kw):
    p = {"player_id": pid, "pick": "over", "pick_ev": ev, "kind": kind, "thin": False, "injury": None, "gap": 1.0}
    p.update(kw)
    return p


class TestTopProps(unittest.TestCase):
    def test_ranks_by_ev_one_per_player_and_caps_at_ten(self):
        props = [_prop(f"p{i}", 0.01 + i * 0.001) for i in range(15)]
        props.append(_prop("p14", 0.99))                                   # second market for the same player
        top = ep.rank_top_props(props)
        self.assertEqual(len(top), 10)
        self.assertEqual(top[0]["pick_ev"], 0.99)
        self.assertEqual(len({p["player_id"] for p in top}), 10)
        self.assertEqual([p["top_rank"] for p in top], list(range(1, 11)))
        self.assertEqual(sum(p["top_rank"] is not None for p in props), 10)

    def test_excludes_td_thin_injured_negative_ev_and_far_from_market(self):
        props = [_prop("td", 0.5, kind="td"), _prop("thin", 0.5, thin=True), _prop("hurt", 0.5, injury="Questionable"),
                 _prop("neg", -0.02), _prop("far", 0.5, gap=15.0), _prop("nopick", 0.5, pick=None),
                 _prop("good", 0.02)]
        top = ep.rank_top_props(props)
        self.assertEqual([p["player_id"] for p in top], ["good"])


def _game(name, home_cover, over, **kw):
    g = {"game": name, "home": "HHH", "away": "AAA", "home_spread": -3.0, "total": 45.0,
         "home_cover": home_cover, "away_cover": 1 - home_cover, "over": over, "under": 1 - over,
         "spread_pick": None, "total_pick": None, "in_window": True, "week": 5}
    g.update(kw)
    return g


class TestGamePicks(unittest.TestCase):
    def test_ranks_all_picks_by_probability(self):
        games = [_game("A", 0.56, 0.52), _game("B", 0.40, 0.61), _game("C", 0.51, 0.49)]
        picks = ep.rank_game_picks(games)
        self.assertEqual(len(picks), 6)
        self.assertEqual([p["rank"] for p in picks], list(range(1, 7)))
        probs = [p["prob"] for p in picks]
        self.assertEqual(probs, sorted(probs, reverse=True))
        self.assertEqual(picks[0]["pick"], "Over 45")                      # B's 61% over
        self.assertEqual(picks[1]["pick"], "AAA +3")                       # B's away side at 60%
        self.assertEqual(games[1]["pick_ranks"]["Total"], 1)

    def test_out_of_window_games_are_not_ranked(self):
        games = [_game("A", 0.56, 0.52), _game("LATER", 0.9, 0.9, in_window=False)]
        picks = ep.rank_game_picks(games)
        self.assertEqual({p["game"] for p in picks}, {"A"})
        self.assertEqual(games[1]["pick_ranks"], {})

    def test_uses_the_ev_pick_when_priced_and_notes_rank_agreement(self):
        g = _game("A", 0.52, 0.5, spread_pick="home", spread_pick_ev=0.03, spread_pick_price={"price": -108, "book": "X"},
                  rank={"rank_home_cover": 0.6})
        spread = [p for p in ep.rank_game_picks([g]) if p["type"] == "Spread"][0]
        self.assertEqual(spread["pick"], "HHH -3")
        self.assertTrue(spread["rank_agrees"])
        g["rank"] = {"rank_home_cover": 0.4}
        self.assertFalse([p for p in ep.rank_game_picks([g]) if p["type"] == "Spread"][0]["rank_agrees"])


if __name__ == "__main__":
    unittest.main()
