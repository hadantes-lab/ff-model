"""
Tests for context_data.py (weather, target share, defense vs position, weather backlog) and the
page's pure context functions (the TARGETS block in web/props_template.html, run under node).

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


def _forecast():
    times = [f"2026-10-11T{h:02d}:00" for h in range(10, 24)]
    return {"hourly": {"time": times,
                       "temperature_2m": [50 + h for h in range(14)], "wind_speed_10m": [5.0 + h for h in range(14)],
                       "wind_gusts_10m": [9.0 + h for h in range(14)], "precipitation_probability": [0, 0, 10, 40, 20] + [0] * 9,
                       "weather_code": [0, 1, 3, 61, 3] + [0] * 9}}


class TestWeather(unittest.TestCase):
    def test_conditions_cover_the_three_hours_from_kickoff(self):
        wx = cd.parse_forecast(_forecast(), "2026-10-11T13:00:00Z")               # hours 13, 14, 15
        self.assertEqual(wx["temp_f"], round((53 + 54 + 55) / 3))
        self.assertEqual(wx["wind_mph"], 10)                                       # the worst hour (8, 9, 10), not the mean
        self.assertEqual(wx["precip_pct"], 40)
        self.assertEqual(wx["sky"], "Light rain")                                  # the worst sky in the window

    def test_kickoff_outside_the_forecast_is_none(self):
        self.assertIsNone(cd.parse_forecast(_forecast(), "2026-11-30T17:00:00Z"))
        self.assertIsNone(cd.parse_forecast({}, "2026-10-11T13:00:00Z"))

    def test_international_venue_beats_the_home_team(self):
        london = cd.resolve_site("JAX", "Tottenham Hotspur Stadium")
        self.assertAlmostEqual(london[1], -0.07, places=1)                         # London, not Jacksonville
        self.assertAlmostEqual(cd.resolve_site("JAX", "EverBank Stadium")[1], -81.64, places=1)
        self.assertEqual(cd.resolve_site("GB", None), cd.TEAM_COORDS["GB"])
        self.assertIsNone(cd.resolve_site("XXX", "Some Park"))

    def test_every_team_has_coordinates(self):
        self.assertEqual(len(cd.TEAM_COORDS), 32)

    def test_domes_skip_the_forecast_entirely(self):
        def boom(*a):
            raise AssertionError("must not fetch a forecast for an indoor game")
        for roof in ("dome", "closed"):
            w = cd.weather_for_game("NO", "2026-10-11T17:00:00Z", roof, "Caesars Superdome", fetch=boom)
            self.assertTrue(w["indoors"])
        self.assertFalse(cd.is_indoors("outdoors"))
        self.assertFalse(cd.is_indoors("open"))                                    # an open retractable roof is outdoors

    def test_retractable_roof_with_unknown_status_is_marked_not_treated_as_outdoors(self):
        fx = lambda lat, lon: _forecast()
        unknown = cd.weather_for_game("ARI", "2026-10-11T13:00:00Z", float("nan"), "State Farm Stadium", fetch=fx)
        self.assertTrue(unknown["retractable"])
        self.assertFalse(cd.weather_for_game("ARI", "2026-10-11T13:00:00Z", "open", "State Farm Stadium", fetch=fx)["retractable"])
        self.assertFalse(cd.weather_for_game("GB", "2026-10-11T13:00:00Z", float("nan"), "Lambeau Field", fetch=fx)["retractable"])
        # an Arizona "home" game played abroad is a genuinely outdoor venue
        abroad = cd.weather_for_game("ARI", "2026-10-11T13:00:00Z", float("nan"), "Estadio Azteca", fetch=fx)
        self.assertFalse(abroad["retractable"])

    def test_outdoor_game_gets_a_card_and_failures_give_none(self):
        w = cd.weather_for_game("GB", "2026-10-11T13:00:00Z", "outdoors", "Lambeau Field", fetch=lambda lat, lon: _forecast())
        self.assertFalse(w["indoors"])
        self.assertEqual(w["stadium"], "Lambeau Field")
        self.assertIsNone(cd.weather_for_game("GB", "2026-10-11T13:00:00Z", "outdoors", "Lambeau Field", fetch=lambda *a: None))
        self.assertIsNone(cd.weather_for_game("XXX", "2026-10-11T13:00:00Z", "outdoors", "Nowhere", fetch=lambda *a: _forecast()))

    def test_schedule_row_picks_the_nearest_rematch(self):
        s = pd.DataFrame([{"home_team": "NE", "away_team": "BUF", "gameday": "2026-09-20", "week": 3},
                          {"home_team": "NE", "away_team": "BUF", "gameday": "2026-12-27", "week": 17}])
        self.assertEqual(cd.schedule_row(s, "NE", "BUF", "2026-12-27T18:00:00Z")["week"], 17)
        self.assertEqual(cd.schedule_row(s, "NE", "BUF", "2026-09-20T17:00:00Z")["week"], 3)
        self.assertIsNone(cd.schedule_row(s, "NYJ", "BUF", "2026-09-20T17:00:00Z"))

    def test_collect_weather_skips_games_without_a_schedule_row(self):
        s = pd.DataFrame([{"home_team": "GB", "away_team": "CHI", "gameday": "2026-10-11", "week": 5,
                           "roof": "outdoors", "stadium": "Lambeau Field"}])
        cards, meta = cd.collect_weather([("CHI @ GB", "GB", "CHI", "2026-10-11T13:00:00Z"),
                                          ("X @ Y", "Y", "X", "2026-10-11T13:00:00Z")], s, fetch=lambda *a: _forecast())
        self.assertEqual(list(cards), ["CHI @ GB"])
        self.assertEqual(meta["CHI @ GB"]["week"], 5)


class TestWeatherBacklog(unittest.TestCase):
    def test_rows_are_appended_never_overwritten(self):
        path = pathlib.Path(tempfile.mkdtemp()) / "w.csv"
        cards = {"CHI @ GB": {"temp_f": 60, "wind_mph": 8, "gust_mph": 14, "precip_pct": 10, "sky": "Clear",
                              "indoors": False, "roof": "outdoors", "stadium": "Lambeau Field"}}
        meta = {"CHI @ GB": {"week": 5, "commence": "2026-10-11T17:00:00Z"}}
        for stamp in ("2026-10-08T13:00:00+00:00", "2026-10-10T13:00:00+00:00"):
            self.assertEqual(cd.append_weather_history(cd.weather_rows(cards, meta, 2026, stamp), path), 1)
        lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 3)                                            # header + two pulls
        self.assertEqual(cd.append_weather_history([], path), 0)


def _pbp():
    rows = []

    def tgt(team, week, receiver, yl):
        rows.append({"season": 2026, "week": week, "season_type": "REG", "posteam": team, "play_type": "pass",
                     "receiver_player_id": receiver, "yardline_100": yl})
    for wk in (1, 2):
        tgt("AAA", wk, "r1", 40); tgt("AAA", wk, "r1", 15); tgt("AAA", wk, "r2", 8); tgt("AAA", wk, "r3", 60)
    tgt("AAA", 2, "r1", 5)
    rows.append({"season": 2026, "week": 1, "season_type": "REG", "posteam": "AAA", "play_type": "run",
                 "receiver_player_id": None, "yardline_100": 3})                        # a run is not a target
    rows.append({"season": 2026, "week": 1, "season_type": "REG", "posteam": "AAA", "play_type": "pass",
                 "receiver_player_id": None, "yardline_100": 30})                       # a sack/throwaway: no receiver
    return pd.DataFrame(rows)


def _hist():
    rows = []
    for pid, name, wks in (("r1", "One", (1, 2)), ("r2", "Two", (1, 2)), ("r3", "Three", (1, 2)), ("r4", "Rare", (1,))):
        for wk in wks:
            rows.append({"player_id": pid, "player_display_name": name, "team": "AAA", "position": "WR", "season": 2026, "week": wk})
    return pd.DataFrame(rows)


class TestTargetTables(unittest.TestCase):
    def setUp(self):
        self.t = cd.target_tables(_pbp(), _hist(), 2026, ["AAA", "ZZZ"])["AAA"]

    def test_team_weeks_count_only_real_targets(self):
        self.assertEqual(self.t["weeks"], [[1, 4, 2, 1], [2, 5, 3, 2]])             # [week, targets, RZ, inside-10]

    def test_players_have_per_game_splits_and_gp_includes_zero_target_games(self):
        r1 = [p for p in self.t["players"] if p["id"] == "r1"][0]
        self.assertEqual(r1["g"], [[1, 2, 1, 0], [2, 3, 2, 1]])
        self.assertEqual(self.t["players"][0]["id"], "r1")                          # sorted by total targets

    def test_players_under_the_target_floor_and_unknown_teams_are_left_out(self):
        self.assertNotIn("r4", [p["id"] for p in self.t["players"]])                # no targets at all
        self.assertEqual(cd.target_tables(_pbp(), _hist(), 2026, ["ZZZ"]), {})


class TestDefenseVsPosition(unittest.TestCase):
    def test_ranks_one_means_allows_the_fewest(self):
        from props_model import MARKETS

        rows = []
        for opp, yds in (("TOUGH", 40), ("MID", 70), ("SOFT", 100)):
            for wk in (1, 2, 3):
                rows += [{"opponent_team": opp, "position": "WR", "season": 2026, "week": wk, "receiving_yards": yds / 2,
                          "receptions": 3, "receiving_tds": 0, "player_id": f"{opp}{wk}{i}"} for i in range(2)]
        d = cd.defense_vs_position(pd.DataFrame(rows), 2026, {"player_reception_yds": MARKETS["player_reception_yds"]})
        self.assertEqual([d[t]["WR"]["player_reception_yds"]["rank"] for t in ("TOUGH", "MID", "SOFT")], [1, 2, 3])
        self.assertEqual(d["SOFT"]["WR"]["player_reception_yds"]["pg"], 100.0)      # all WRs summed per game
        self.assertAlmostEqual(d["MID"]["WR"]["player_reception_yds"]["lg"], 70.0)

    def test_falls_back_to_last_season_until_there_are_enough_games(self):
        from props_model import MARKETS

        rows = [{"opponent_team": "NEW", "position": "WR", "season": 2025, "week": w, "receiving_yards": 100.0,
                 "receptions": 0, "receiving_tds": 0, "player_id": f"a{w}"} for w in range(1, 11)]
        rows += [{"opponent_team": "NEW", "position": "WR", "season": 2026, "week": 1, "receiving_yards": 0.0,
                  "receptions": 0, "receiving_tds": 0, "player_id": "b"}]
        d = cd.defense_vs_position(pd.DataFrame(rows), 2026, {"player_reception_yds": MARKETS["player_reception_yds"]})
        cell = d["NEW"]["WR"]["player_reception_yds"]
        self.assertEqual(cell["n"], 11)                                             # 1 game this season is too few: both seasons used
        self.assertAlmostEqual(cell["pg"], 100 * 10 / 11, places=1)


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class TestContextJs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        src = TEMPLATE.read_text(encoding="utf-8")
        cls.block = src[src.index("/* TARGETS:BEGIN"):src.index("/* TARGETS:END */")]

    def _run(self, expr):
        code = self.block + f"\nconsole.log(JSON.stringify({expr}));"
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
            f.write(code)
        out = subprocess.run(["node", f.name], capture_output=True, text=True, timeout=30)
        pathlib.Path(f.name).unlink()
        self.assertEqual(out.returncode, 0, out.stderr)
        return json.loads(out.stdout)

    TD = {"weeks": [[1, 30, 3, 1], [2, 40, 5, 2], [4, 30, 2, 1], [5, 20, 4, 3]],             # the team had a bye in week 3
          "players": [{"id": "a", "name": "Alpha", "pos": "WR", "g": [[1, 10, 1, 0], [2, 20, 3, 1], [4, 6, 0, 0], [5, 12, 3, 2]]},
                      {"id": "b", "name": "Beta", "pos": "TE", "g": [[1, 5, 0, 0], [2, 0, 0, 0], [4, 10, 1, 1]]},
                      {"id": "c", "name": "Gamma", "pos": "RB", "g": [[1, 0, 0, 0], [2, 0, 0, 0]]}]}

    def test_season_window_totals_and_shares(self):
        r = self._run(f"targetRows({json.dumps(self.TD)}, 'season')")
        self.assertEqual(r["team"], {"tgt": 120, "rz": 14, "in10": 7})
        a = r["rows"][0]
        self.assertEqual((a["name"], a["gp"], a["tgt"], a["rz"]), ("Alpha", 4, 48, 7))
        self.assertAlmostEqual(a["share"], 48 / 120)
        self.assertAlmostEqual(a["avg"], 12.0)
        self.assertAlmostEqual(a["rzShare"], 7 / 14)
        self.assertNotIn("Gamma", [x["name"] for x in r["rows"]])                          # zero targets: not listed

    def test_windows_are_the_teams_last_n_games_so_a_bye_does_not_count(self):
        r = self._run(f"targetRows({json.dumps(self.TD)}, 2)")                              # weeks 4 and 5
        self.assertEqual(r["team"]["tgt"], 50)
        self.assertEqual([(x["name"], x["gp"], x["tgt"]) for x in r["rows"]], [("Alpha", 2, 18), ("Beta", 1, 10)])

    def test_no_data_is_null(self):
        self.assertIsNone(self._run("targetRows(null, 'season')"))
        self.assertIsNone(self._run("targetRows({weeks: [], players: []}, 3)"))

    def test_matchup_rank_colors_and_weather_flags(self):
        self.assertEqual([self._run(f"dvpClass({r})") for r in (1, 10, 11, 22, 23, 32)],
                         ["tough", "tough", "mid", "mid", "soft", "soft"])
        self.assertEqual(self._run("weatherFlags({indoors: false, wind_mph: 16, gust_mph: 20, temp_f: 30, precip_pct: 60})"),
                         ["windy", "freezing", "rain likely"])
        self.assertEqual(self._run("weatherFlags({indoors: false, wind_mph: 5, gust_mph: 24, temp_f: 50, precip_pct: 20})"), [])
        self.assertEqual(self._run("weatherFlags({indoors: true})"), [])
        self.assertEqual(self._run("weatherFlags({indoors: false, wind_mph: 8, gust_mph: 26, temp_f: 50, precip_pct: 0})"), ["windy"])
        self.assertEqual(self._run("weatherFlags({indoors: false, retractable: true, wind_mph: 27, gust_mph: 36, temp_f: 20, precip_pct: 90})"), [])


if __name__ == "__main__":
    unittest.main()
