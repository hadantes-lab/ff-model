"""
Tests for youtube_picks.py against a fake YouTube Data API: channel lookup (paid for once), new uploads only, picks read from titles and descriptions
under the same rules as any post, one failing channel not stopping the rest, the quota count, and the notes importer.

Run:  python -m unittest discover -s tests -v
"""

import json
import os
import pathlib
import tempfile
import unittest

import pandas as pd

import social_picks as sp
import youtube_picks as yp
from tests.test_social_picks import GAMES

PUBLISHED = "2026-11-07T14:00:00Z"                         # a Saturday morning, before every Sunday game in the fixture schedule


class FakeYouTube:
    """channels.list / playlistItems.list / videos.list over a small catalogue."""

    def __init__(self, uploads, descriptions=None, fail=None):
        self.uploads, self.desc, self.fail, self.calls = uploads, descriptions or {}, fail or {}, []

    def __call__(self, path, params, key):
        self.calls.append((path, dict(params), key))
        if path in self.fail:
            return self.fail[path]
        if path == "channels":
            h = params["forHandle"].lstrip("@")
            if h not in self.uploads:
                return 200, {"items": []}
            return 200, {"items": [{"id": f"UC_{h}", "snippet": {"title": f"{h} channel"}, "contentDetails": {"relatedPlaylists": {"uploads": f"UU_{h}"}}}]}
        if path == "playlistItems":
            h = params["playlistId"][3:]
            return 200, {"items": [{"snippet": {"title": t, "description": "(cut short)", "publishedAt": PUBLISHED},
                                    "contentDetails": {"videoId": vid, "videoPublishedAt": PUBLISHED}} for vid, t in self.uploads[h]]}
        if path == "videos":
            return 200, {"items": [{"id": v, "snippet": {"description": self.desc.get(v, "")}} for v in params["id"].split(",")]}
        raise AssertionError(path)


def _state():
    return {"channels": {}, "seen": [], "units": 0, "last_fetch": None, "last_note": None}


class TestFetch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = pathlib.Path(self.tmp.name) / "log.csv"

    def tearDown(self):
        self.tmp.cleanup()

    def api(self):
        return FakeYouTube({"sharp": [("v1", "Week 9 Best Bets"), ("v2", "Lions -3 is my best bet")],
                            "talk": [("v3", "Patriots +6.5 | Week 9 NFL picks")]},
                           {"v1": "Lions -3 and Under 47.5 for Sunday", "v2": "", "v3": "subscribe"})

    def test_picks_come_from_titles_and_descriptions_as_the_channel(self):
        state = _state()
        r = yp.fetch(["sharp", "talk"], "KEY", GAMES, state, self.api(), self.log)
        self.assertEqual((r["new_videos"], r["picks_found"], r["added"]), (3, 4, 3))          # Lions -3 appears in two of sharp's videos: one pick
        df = sp.load(self.log)
        got = {(x.source, x.market, x.side) for x in df.itertuples()}
        self.assertEqual(got, {("yt:sharp", "spread", "DET"), ("yt:sharp", "total", "under"), ("yt:talk", "spread", "NE")})
        self.assertEqual(set(df["platform"]), {"youtube"})
        self.assertEqual(df[df["market"] == "total"].iloc[0]["post_id"], "v1")

    def test_a_channels_first_call_on_a_game_stands(self):
        # v1's description has Lions -3; v2's title is the same pick: one pick on the game, not two
        yp.fetch(["sharp"], "KEY", GAMES, _state(), self.api(), self.log)
        df = sp.load(self.log)
        self.assertEqual(len(df[(df["source"] == "yt:sharp") & (df["market"] == "spread")]), 1)

    def test_only_new_videos_are_read_and_the_channel_lookup_is_paid_once(self):
        state, api = _state(), self.api()
        yp.fetch(["sharp"], "KEY", GAMES, state, api, self.log)
        first_calls = [c[0] for c in api.calls]
        self.assertEqual(first_calls, ["channels", "playlistItems", "videos"])
        self.assertEqual(state["units"], 3)
        yp.fetch(["sharp"], "KEY", GAMES, state, api, self.log)                       # nothing new
        self.assertEqual([c[0] for c in api.calls][3:], ["playlistItems"])            # no lookup, no description call
        self.assertEqual(state["units"], 4)

    def test_a_video_title_with_no_pick_costs_nothing(self):
        api = FakeYouTube({"c": [("v1", "NFL Week 9 Predictions and Analysis")]}, {"v1": "like and subscribe"})
        r = yp.fetch(["c"], "KEY", GAMES, _state(), api, self.log)
        self.assertEqual((r["new_videos"], r["picks_found"], r["added"]), (1, 0, 0))
        self.assertEqual(len(r["videos"]), 1)                                         # but it is still listed for the page

    def test_one_bad_channel_does_not_stop_the_rest(self):
        r = yp.fetch(["nosuch", "talk"], "KEY", GAMES, _state(), self.api(), self.log)
        self.assertEqual(r["picks_found"], 1)
        self.assertIn("no channel found for @nosuch", r["note"])

    def test_quota_exhaustion_stops_and_says_so(self):
        api = FakeYouTube({"a": [], "b": []}, fail={"channels": (403, {"error": {"errors": [{"reason": "quotaExceeded"}]}})})
        state = _state()
        r = yp.fetch(["a", "b"], "KEY", GAMES, state, api, self.log)
        self.assertIn("quotaExceeded", r["note"])
        self.assertEqual(len(api.calls), 1)                                           # stopped at the first channel, did not hammer the rest

    def test_the_key_never_appears_in_the_state_or_latest_files(self):
        state = _state()
        r = yp.fetch(["sharp"], "SECRETKEY", GAMES, state, self.api(), self.log)
        p = pathlib.Path(self.tmp.name)
        yp.save_state(state, p / "s.json")
        yp.save_latest(r["videos"], p / "l.json")
        self.assertNotIn("SECRETKEY", (p / "s.json").read_text(encoding="utf-8") + (p / "l.json").read_text(encoding="utf-8"))

    def test_a_pick_posted_after_kickoff_does_not_count(self):
        old = PUBLISHED
        api = FakeYouTube({"c": [("v1", "Lions -3 lock")]})
        api_items = api.__call__

        def late(path, params, key):
            status, body = api_items(path, params, key)
            if path == "playlistItems":
                for it in body["items"]:
                    it["contentDetails"]["videoPublishedAt"] = "2026-11-08T19:00:00Z"        # after the 1:00 pm ET kickoff
            return status, body
        r = yp.fetch(["c"], "KEY", GAMES, _state(), late, self.log)
        self.assertEqual(r["picks_found"], 0)
        self.assertEqual(old, PUBLISHED)


class TestConfigAndBlock(unittest.TestCase):
    def test_channels_merge_the_file_and_environment_without_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "c.json"
            f.write_text(json.dumps({"youtube_channels": ["@One", "two"]}), encoding="utf-8")
            os.environ["YOUTUBE_CHANNELS"] = "TWO, three"
            try:
                self.assertEqual(yp.load_channels(f), ["One", "two", "three"])
            finally:
                os.environ.pop("YOUTUBE_CHANNELS", None)
            self.assertEqual(yp.load_channels(pathlib.Path(d) / "missing.json"), [])

    def test_the_block_is_json_and_names_resolved_channels(self):
        state = _state()
        state["channels"]["sharp"] = {"id": "UC1", "title": "Warren Sharp", "uploads": "UU1"}
        b = yp.block(["sharp", "other"], state, {"generated": "g", "videos": [{"title": "t"}]})
        json.dumps(b)
        self.assertEqual([c["title"] for c in b["channels"]], ["Warren Sharp", None])

    def test_source_ids(self):
        self.assertEqual(yp.source_id("@RossTuckerNFL"), "yt:rosstuckernfl")


class TestNotes(unittest.TestCase):
    TEXT = ("# heard on the Saturday show\n"
            "SharpFootballAnalysis | 2026-11-07 09:30 | Lions -3, Under 47.5 | https://youtu.be/abc\n"
            "\n"
            "RossTuckerNFL | 2026-11-07T14:00:00Z | Patriots +6.5\n")

    def test_notes_parse_into_posts(self):
        p = yp.parse_notes(self.TEXT)
        self.assertEqual([(x["source"], x["post_id"]) for x in p], [("yt:sharpfootballanalysis", "https://youtu.be/abc"), ("yt:rosstuckernfl", "note-4")])

    def test_a_malformed_line_says_which(self):
        with self.assertRaises(ValueError) as cm:
            yp.parse_notes("only | two")
        self.assertIn("line 1", str(cm.exception))

    def test_notes_are_logged_and_graded_like_any_pick(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            (d / "n.txt").write_text(self.TEXT, encoding="utf-8")
            r = yp.import_notes(d / "n.txt", GAMES, d / "l.csv")
            self.assertEqual(r, {"notes": 2, "picks_found": 3, "added": 3})
            df = sp.load(d / "l.csv")
            self.assertEqual(set(df["platform"]), {"youtube"})
            self.assertEqual(df[df["source"] == "yt:sharpfootballanalysis"].iloc[0]["posted_at"], "2026-11-07T14:30:00+00:00")   # 9:30 ET
            games = GAMES.copy()
            games.loc[games["game_id"] == "2026_09_DET_MIA", ["home_score", "away_score"]] = [14.0, 24.0]    # DET -3 wins, under 47.5 wins
            self.assertEqual(sp.grade(games, d / "l.csv"), 2)


if __name__ == "__main__":
    unittest.main()
