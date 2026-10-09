"""
Tests for the hit-rate chart: the Python that builds each prop's game log (export_props) and the
page's pure JavaScript math (web/props_template.html, run under node when it is installed).

Run:  python -m unittest discover -s tests -v
"""

import json
import pathlib
import shutil
import subprocess
import tempfile
import unittest

import pandas as pd

import export_props as ep

TEMPLATE = pathlib.Path(__file__).resolve().parent.parent / "web" / "props_template.html"


def _history():
    rows = []
    for w, (yds, tds) in enumerate([(50, 0), (80, 1), (30, 0), (120, 2)], start=1):
        rows.append(dict(player_id="p1", team="AAA", opponent_team=f"O{w}", season=2026, week=w, position="WR",
                         receiving_yards=yds, receiving_tds=tds, rushing_yards=0, rushing_tds=0, rushing_fumbles_lost=0))
    rows.append(dict(player_id="other", team="BBB", opponent_team="X", season=2026, week=1, position="WR",
                     receiving_yards=9, receiving_tds=0, rushing_yards=0, rushing_tds=0))
    return pd.DataFrame(rows)


class TestGameLog(unittest.TestCase):
    def setUp(self):
        sched = pd.DataFrame([
            {"season": 2026, "week": 1, "home_team": "AAA", "away_team": "O1", "gameday": "2026-09-13"},
            {"season": 2026, "week": 2, "home_team": "O2", "away_team": "AAA", "gameday": "2026-09-20"}])
        self.sched_map = ep.schedule_lookup(sched)

    def test_schedule_lookup_marks_home_and_away(self):
        self.assertEqual(self.sched_map[(2026, 1, "AAA")], (1, "2026-09-13"))
        self.assertEqual(self.sched_map[(2026, 2, "AAA")], (0, "2026-09-20"))

    def test_log_has_one_row_per_game_in_order_with_the_right_stat(self):
        log = ep.game_log_for(_history(), "p1", "player_reception_yds", self.sched_map, {})
        self.assertEqual([g[3] for g in log], [50.0, 80.0, 30.0, 120.0])
        self.assertEqual([g[1] for g in log], [1, 2, 3, 4])
        self.assertEqual(log[0][2], "O1")
        self.assertEqual(log[0][4:6], [1, "2026-09-13"])
        self.assertEqual(log[1][4], 0)                                        # away in week 2
        self.assertIsNone(log[2][4])                                          # schedule has no week-3 row: unknown, not wrong

    def test_only_that_players_games_are_included(self):
        log = ep.game_log_for(_history(), "p1", "player_reception_yds", self.sched_map, {})
        self.assertNotIn(9.0, [g[3] for g in log])
        self.assertEqual(ep.game_log_for(_history(), "nobody", "player_reception_yds", self.sched_map, {}), [])

    def test_lines_then_attach_only_to_games_that_had_one(self):
        then = ep.lines_then_lookup(pd.DataFrame([
            {"player_id": "p1", "market": "player_reception_yds", "season": 2026, "week": 2, "line": 61.5},
            {"player_id": "p1", "market": "player_rush_yds", "season": 2026, "week": 3, "line": 5.5}]))
        log = ep.game_log_for(_history(), "p1", "player_reception_yds", self.sched_map, then)
        self.assertEqual([g[6] for g in log], [None, 61.5, None, None])       # the rush line never leaks in

    def test_lines_then_ignores_missing_values_and_empty_logs(self):
        self.assertEqual(ep.lines_then_lookup(pd.DataFrame()), {})
        self.assertEqual(ep.lines_then_lookup(None), {})
        self.assertEqual(ep.lines_then_lookup(pd.DataFrame(
            [{"player_id": "p", "market": "m", "season": 2026, "week": 1, "line": float("nan")}])), {})

    def test_log_is_capped(self):
        rows = [dict(player_id="p", team="AAA", opponent_team="O", season=2025 + w // 18, week=w % 18 + 1, position="WR",
                     receiving_yards=w, receiving_tds=0, rushing_yards=0, rushing_tds=0) for w in range(60)]
        log = ep.game_log_for(pd.DataFrame(rows), "p", "player_reception_yds", {}, {})
        self.assertEqual(len(log), ep.GAME_LOG_MAX)
        self.assertEqual(log[-1][3], 59.0)                                    # keeps the most recent


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class TestHitRateJs(unittest.TestCase):
    """Runs the template's HITRATE block under node so the chart's numbers are tested, not eyeballed."""

    @classmethod
    def setUpClass(cls):
        src = TEMPLATE.read_text(encoding="utf-8")
        a, b = src.index("/* HITRATE:BEGIN"), src.index("/* HITRATE:END */")
        cls.block = src[a:b]

    def _run(self, expr):
        code = self.block + f"\nconsole.log(JSON.stringify({expr}));"
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
            f.write(code)
        out = subprocess.run(["node", f.name], capture_output=True, text=True, timeout=30)
        pathlib.Path(f.name).unlink()
        self.assertEqual(out.returncode, 0, out.stderr)
        return json.loads(out.stdout)

    # [season, week, opp, value, home, date, line_then]
    G = [[2025, 17, "A", 40, 1, None, None], [2026, 1, "B", 60, 0, None, 50.5], [2026, 2, "C", 50.5, 1, None, 55.5],
         [2026, 3, "D", 80, 0, None, None], [2026, 4, "E", 20, 1, None, 25.5]]

    def test_hit_rate_counts_overs_unders_and_pushes(self):
        st = self._run(f"hitStats({json.dumps(self.G)}, 50.5, 'yards', 'now')")
        self.assertEqual((st["over"], st["under"], st["push"]), (2, 2, 1))      # 60, 80 over; 40, 20 under; 50.5 push
        self.assertAlmostEqual(st["pct"], 0.5)                                    # pushes excluded from the rate
        self.assertAlmostEqual(st["avg"], 50.1)
        self.assertEqual(st["median"], 50.5)

    def test_line_then_uses_each_games_own_line_and_falls_back_to_todays(self):
        st = self._run(f"hitStats({json.dumps(self.G)}, 50.5, 'yards', 'then')")
        # 40 vs 50.5 under | 60 vs 50.5 over | 50.5 vs 55.5 under | 80 vs 50.5 over | 20 vs 25.5 under
        self.assertEqual(st["results"], ["under", "over", "under", "over", "under"])

    def test_touchdowns_are_any_td_regardless_of_line(self):
        g = [[2026, 1, "A", 0, 1, None, None], [2026, 2, "B", 1, 1, None, None], [2026, 3, "C", 2, 0, None, None]]
        st = self._run(f"hitStats({json.dumps(g)}, 0.5, 'td', 'now')")
        self.assertEqual((st["over"], st["under"]), (2, 1))

    def test_empty_selection_is_safe(self):
        st = self._run("hitStats([], 10, 'yards', 'now')")
        self.assertEqual(st["n"], 0)
        self.assertIsNone(st["pct"])
        self.assertIsNone(st["avg"])

    def test_window_and_home_away_filters(self):
        g = json.dumps(self.G)
        self.assertEqual(len(self._run(f"hitSelect({g}, {{win: 3, ha: 'all'}})")), 3)
        self.assertEqual([x[2] for x in self._run(f"hitSelect({g}, {{win: 'season', ha: 'all'}})")], ["B", "C", "D", "E"])
        self.assertEqual([x[2] for x in self._run(f"hitSelect({g}, {{win: 'all', ha: 'home'}})")], ["A", "C", "E"])
        self.assertEqual([x[2] for x in self._run(f"hitSelect({g}, {{win: 'all', ha: 'away'}})")], ["B", "D"])
        # the window applies AFTER the home/away filter: the last 2 HOME games, not home games among the last 2
        self.assertEqual([x[2] for x in self._run(f"hitSelect({g}, {{win: 2, ha: 'home'}})")], ["C", "E"])


if __name__ == "__main__":
    unittest.main()
