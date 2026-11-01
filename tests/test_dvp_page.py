"""
Tests for the Defense vs Position table: the per-game numbers context_data.dvp_table builds, and the ranking/label logic in the
DVP block of web/props_template.html (run under node when it is installed).

Run:  python -m unittest discover -s tests -v
"""

import json
import pathlib
import shutil
import subprocess
import tempfile
import unittest

import pandas as pd

import context_data as cd

TEMPLATE = pathlib.Path(__file__).resolve().parent.parent / "web" / "props_template.html"


def _row(team, opp, season, week, pos, **kw):
    d = {"team": team, "opponent_team": opp, "season": season, "week": week, "position": pos, "attempts": 0, "completions": 0, "passing_yards": 0,
         "passing_tds": 0, "carries": 0, "rushing_yards": 0, "rushing_tds": 0, "targets": 0, "receptions": 0, "receiving_yards": 0, "receiving_tds": 0}
    d.update(kw)
    return d


def _history():
    rows = []
    for wk, (att, tgt) in enumerate([(30, 20), (40, 30), (35, 25), (45, 35), (50, 40)], start=1):   # DEF faces OFF every week
        rows += [_row("OFF", "DEF", 2026, wk, "QB", attempts=att, completions=att // 2, passing_yards=att * 7, passing_tds=2, carries=3, rushing_yards=10),
                 _row("OFF", "DEF", 2026, wk, "WR", targets=tgt, receptions=tgt // 2, receiving_yards=tgt * 8, receiving_tds=1),
                 _row("OFF", "DEF", 2026, wk, "RB", carries=20, rushing_yards=90, rushing_tds=1, targets=5, receptions=4, receiving_yards=30),
                 _row("DEF", "OFF", 2026, wk, "QB", attempts=25)]
    rows += [_row("OFF", "DEF", 2025, 1, "QB", attempts=20), _row("DEF", "OFF", 2025, 1, "QB", attempts=22)]
    return pd.DataFrame(rows)


class TestDvpTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = cd.dvp_table(_history(), 2026)

    def test_season_window_is_per_game_team_totals(self):
        a = self.t["windows"]["season"]["ALL"]["DEF"]
        self.assertEqual(a["g"], 5)
        self.assertAlmostEqual(a["pass_att"], 40.0)                     # (30+40+35+45+50)/5
        self.assertAlmostEqual(a["rush_att"], 23.0)                     # QB 3 + RB 20
        self.assertAlmostEqual(a["tgt"], (20 + 30 + 25 + 35 + 40) / 5 + 5)

    def test_position_views_only_count_that_position(self):
        qb = self.t["windows"]["season"]["QB"]["DEF"]
        self.assertAlmostEqual(qb["pass_att"], 40.0)
        self.assertEqual(qb["tgt"], 0.0)
        wr = self.t["windows"]["season"]["WR"]["DEF"]
        self.assertEqual(wr["pass_att"], 0.0)
        self.assertAlmostEqual(wr["tgt"], 30.0)

    def test_last4_uses_the_last_four_games(self):
        a = self.t["windows"]["last4"]["ALL"]["DEF"]
        self.assertEqual(a["g"], 4)
        self.assertAlmostEqual(a["pass_att"], (40 + 35 + 45 + 50) / 4)

    def test_previous_season_window(self):
        self.assertAlmostEqual(self.t["windows"]["prev"]["ALL"]["DEF"]["pass_att"], 20.0)

    def test_a_position_that_did_nothing_in_a_game_counts_as_zero_not_missing(self):
        h = _history()
        h = h[~((h["opponent_team"] == "DEF") & (h["week"] == 5) & (h["position"] == "WR"))]       # no WR stat line in week 5
        wr = cd.dvp_table(h, 2026)["windows"]["season"]["WR"]["DEF"]
        self.assertEqual(wr["g"], 5)
        self.assertAlmostEqual(wr["tgt"], (20 + 30 + 25 + 35) / 5)

    def test_game_by_game_and_through_week(self):
        g = self.t["games"]["DEF"]
        self.assertEqual([x[0] for x in g], [1, 2, 3, 4, 5])
        self.assertEqual(g[0][1], "OFF")
        self.assertEqual(g[0][2]["QB"][self.t["stats"].index("pass_att")], 30.0)
        self.assertEqual(self.t["through"], 5)
        json.dumps(self.t)

    def test_rows_without_a_defense_or_position_are_ignored(self):
        h = pd.concat([_history(), pd.DataFrame([_row("OFF", "DEF", 2026, 1, "K", attempts=99)])], ignore_index=True)
        self.assertAlmostEqual(cd.dvp_table(h, 2026)["windows"]["season"]["ALL"]["DEF"]["pass_att"], 40.0)


def _two_defense_history():
    """DEF1 faces the pass-happy OFF1 (50 attempts every week); DEF2 faces the run-heavy OFF2 (20 attempts a week). Same raw 'allowed' pattern otherwise."""
    rows = []
    for wk in range(1, 5):
        rows += [_row("OFF1", "DEF1", 2026, wk, "QB", attempts=50), _row("DEF1", "OFF1", 2026, wk, "QB", attempts=30),
                 _row("OFF2", "DEF2", 2026, wk, "QB", attempts=20), _row("DEF2", "OFF2", 2026, wk, "QB", attempts=30)]
    return pd.DataFrame(rows)


class TestScheduleAdjustment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = cd.dvp_table(_two_defense_history(), 2026)["windows"]["season"]["ALL"]

    def test_raw_values_are_just_the_average(self):
        self.assertEqual(self.w["DEF1"]["pass_att"], 50.0)
        self.assertEqual(self.w["DEF2"]["pass_att"], 20.0)

    def test_a_defense_that_met_a_normal_offense_reads_as_the_league_average(self):
        # OFF1 throws 50 every week, so DEF1 allowing 50 is exactly what that offense always does: adjusted = the league mean
        league = sum(r["pass_att"] for r in self.w.values()) / len(self.w)
        self.assertAlmostEqual(self.w["DEF1"]["adj"]["pass_att"], league, places=1)
        self.assertAlmostEqual(self.w["DEF2"]["adj"]["pass_att"], league, places=1)

    def test_adjustment_moves_a_defense_toward_the_league_when_its_schedule_was_extreme(self):
        raw_gap = abs(self.w["DEF1"]["pass_att"] - self.w["DEF2"]["pass_att"])
        adj_gap = abs(self.w["DEF1"]["adj"]["pass_att"] - self.w["DEF2"]["adj"]["pass_att"])
        self.assertLess(adj_gap, raw_gap)

    def test_the_offense_columns_are_the_teams_own_output(self):
        self.assertEqual(self.w["DEF1"]["off"]["pass_att"], 30.0)           # DEF1's own offense threw 30 a game
        self.assertEqual(self.w["OFF1"]["off"]["pass_att"], 50.0)

    def test_a_single_game_opponent_with_no_other_games_falls_back_to_raw(self):
        h = pd.DataFrame([_row("A", "B", 2026, 1, "QB", attempts=40), _row("B", "A", 2026, 1, "QB", attempts=30)])
        w = cd.dvp_table(h, 2026)["windows"]["season"]["ALL"]
        self.assertEqual(w["B"]["adj"]["pass_att"], w["B"]["pass_att"])


def _tg_rows():
    """Two teams that only play each other, twice a season-long pair of games plus a third team to give baselines."""
    rows = []
    for gid, (a, b, ya, yb) in enumerate([("A", "B", 6.0, 4.0), ("A", "C", 6.0, 5.0), ("B", "C", 4.0, 5.0)] * 2):
        for team, opp, y, y_allowed in ((a, b, ya, yb), (b, a, yb, ya)):
            rows.append({"game_id": f"g{gid}", "season": 2026, "week": gid + 1, "team": team, "opp": opp, "plays": 60.0, "yards": 60.0 * y, "ypp": y,
                         "first_downs": 20.0, "third_rate": 0.4, "top_sec": 1800.0, "pass_rate": 0.55, "neutral_pass_rate": 0.55, "proe": 0.0,
                         "epa_pp": 0.0, "giveaways": 1.0, "plays_allowed": 60.0, "yards_allowed": 60.0 * y_allowed, "ypp_allowed": y_allowed,
                         "first_downs_allowed": 20.0, "third_rate_allowed": 0.4, "epa_pp_allowed": 0.0, "giveaways_allowed": 1.0})
    return pd.DataFrame(rows)


class TestTeamBreakdown(unittest.TestCase):
    def test_raw_and_adjusted_exist_for_every_window_and_team(self):
        import team_stats as ts
        b = ts.team_breakdown(_tg_rows(), 2026)
        for win in ("season", "last4", "prev"):
            self.assertIn(win, b["windows"])
        a = b["windows"]["season"]["A"]
        self.assertAlmostEqual(a["off"]["ypp"], 6.0)
        self.assertEqual(a["g"], 4)
        self.assertIn("ypp", a["off_adj"])
        self.assertIn("ypp", a["def_adj"])
        self.assertAlmostEqual(a["off"]["third_rate"], 40.0)             # shown as a percentage
        self.assertAlmostEqual(a["off"]["top_min"], 30.0)
        self.assertNotIn("pass_rate", a["def"])                          # no opponent-side version of pass rate

    def test_an_offense_that_beat_strong_defenses_is_adjusted_up(self):
        import team_stats as ts
        w = ts.team_breakdown(_tg_rows(), 2026)["windows"]["season"]
        # B's defense allows 6.0 ypp... A scores 6.0 against defenses that allow 4-5: A's adjusted rate is above the league mean
        league = sum(w[t]["off"]["ypp"] for t in w) / len(w)
        self.assertGreater(w["A"]["off_adj"]["ypp"], league)

    def test_last4_takes_each_teams_latest_games(self):
        import team_stats as ts
        t = _tg_rows()
        t.loc[(t["team"] == "A") & (t["week"] <= 2), "ypp"] = 9.0
        b = ts.team_breakdown(t, 2026, last_n=2)["windows"]
        self.assertGreater(b["season"]["A"]["off"]["ypp"], b["last4"]["A"]["off"]["ypp"])


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class TestDvpBlock(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        src = TEMPLATE.read_text(encoding="utf-8")
        cls.block = src[src.index("/* DVP:BEGIN"):src.index("/* DVP:END */")]

    def _run(self, expr):
        code = self.block + f"\nconsole.log(JSON.stringify({expr}));"
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
            f.write(code)
        out = subprocess.run(["node", f.name], capture_output=True, text=True, timeout=30)
        pathlib.Path(f.name).unlink()
        self.assertEqual(out.returncode, 0, out.stderr)
        return json.loads(out.stdout)

    def test_ordinals(self):
        self.assertEqual(self._run("[1, 2, 3, 4, 11, 12, 13, 21, 22, 23, 32].map(ordinal)"),
                         ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "23rd", "32nd"])

    def test_rank_one_is_the_most_allowed(self):
        rows = [{"team": "A", "x": 10}, {"team": "B", "x": 30}, {"team": "C", "x": 20}]
        self.assertEqual(self._run(f"dvRanks({json.dumps(rows)}, 'x')"), {"A": 3, "B": 1, "C": 2})

    def test_ties_share_the_better_rank(self):
        rows = [{"team": "A", "x": 5}, {"team": "B", "x": 5}, {"team": "C", "x": 1}]
        self.assertEqual(self._run(f"dvRanks({json.dumps(rows)}, 'x')"), {"A": 1, "B": 1, "C": 3})

    def test_labels_read_most_least_or_of_32(self):
        self.assertEqual(self._run("[1, 10, 11, 22, 23, 32].map(r => dvRankLabel(r, 32))"),
                         ["1st most", "10th most", "11th of 32", "22nd of 32", "10th least", "1st least"])

    def test_columns_by_position(self):
        self.assertEqual(self._run("DV_COLS.WR"), ["tgt", "rec", "rec_yds", "rec_td"])
        self.assertEqual(len(self._run("DV_COLS.ALL")), 11)
        self.assertEqual(self._run("DV_COLS.QB")[0], "pass_att")

    def test_ranks_accept_a_function_for_the_adjusted_numbers(self):
        rows = [{"team": "A", "x": 10, "adj": {"x": 30}}, {"team": "B", "x": 30, "adj": {"x": 10}}]
        self.assertEqual(self._run(f"dvRanks({json.dumps(rows)}, r => r.adj.x)"), {"A": 1, "B": 2})

    def test_headline_picks_whichever_of_passing_or_rushing_is_furthest_from_the_middle(self):
        row = {"pass_yds": 222.6, "rush_yds": 115.0}
        ranks = {"pass_yds": {"GB": 24}, "rush_yds": {"GB": 16}}                  # 9th least vs a middling rushing rank
        out = self._run(f"dvHeadline('GB', {json.dumps(row)}, {json.dumps(ranks)}, 32, false)")
        self.assertEqual(out, "Packers defense has allowed 222.6 passing yards per game (9th least)")
        ranks = {"pass_yds": {"GB": 15}, "rush_yds": {"GB": 7}}
        out = self._run(f"dvHeadline('GB', {json.dumps(row)}, {json.dumps(ranks)}, 32, false)")
        self.assertEqual(out, "Packers defense has allowed 115.0 rushing yards per game (7th most)")

    def test_headline_uses_the_adjusted_value_when_asked(self):
        row = {"pass_yds": 300.0, "rush_yds": 100.0, "adj": {"pass_yds": 280.0, "rush_yds": 105.0}}
        ranks = {"pass_yds": {"X": 2}, "rush_yds": {"X": 16}}
        self.assertIn("280.0 passing yards", self._run(f"dvHeadline('X', {json.dumps(row)}, {json.dumps(ranks)}, 32, true)"))

    def test_expected_volume_moves_half_way_to_each_side(self):
        self.assertAlmostEqual(self._run("dvExpected(40, 36, 32)"), 32 + 0.5 * 4 + 0.5 * 8)
        self.assertAlmostEqual(self._run("dvExpected(32, 32, 32)"), 32.0)
        self.assertAlmostEqual(self._run("dvExpected(40, 36, 32, 0)"), 32.0)

    def test_every_team_has_a_nickname(self):
        names = self._run("TEAM_NAMES")
        self.assertEqual(len(names), 32)
        self.assertEqual(names["LA"], "Rams")

    def test_rows_come_from_the_chosen_window_and_position(self):
        table = {"windows": {"last4": {"WR": {"DET": {"g": 4, "tgt": 20.0}}}}}
        self.assertEqual(self._run(f"dvRows({json.dumps(table)}, 'last4', 'WR')"), [{"team": "DET", "g": 4, "tgt": 20.0}])
        self.assertEqual(self._run(f"dvRows({json.dumps(table)}, 'season', 'WR')"), [])


if __name__ == "__main__":
    unittest.main()
