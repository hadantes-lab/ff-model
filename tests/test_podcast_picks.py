"""
Tests for podcast_picks.py: spoken numbers, the transcript cleaner, the pick-cue rules (a team and a number in one breath is not a pick), the RSS reader
and the end-to-end fetch against a fake feed and transcript.

Run:  python -m unittest discover -s tests -v
"""

import json
import pathlib
import tempfile
import unittest
import urllib.error

import pandas as pd

import podcast_picks as pp
import social_picks as sp
from tests.test_social_picks import GAMES

NOW = pd.Timestamp("2026-11-07T15:00:00Z")
POSTED = "2026-11-06T12:00:00Z"                           # Friday noon: before every game in the fixture schedule


class TestSpokenNumbers(unittest.TestCase):
    def test_words_to_number(self):
        for spoken, n in [("seven", 7), ("six and a half", 6.5), ("forty seven", 47), ("forty-seven and a half", 47.5), ("twenty one point five", 21.5),
                          ("3.5", 3.5), ("ten", 10), ("thirteen", 13), ("twenty", 20), ("3", 3)]:
            self.assertEqual(pp.words_to_number(spoken), n, spoken)
        self.assertIsNone(pp.words_to_number("banana"))
        self.assertIsNone(pp.words_to_number("twenty twenty"))

    def test_normalize(self):
        self.assertEqual(pp.normalize_spoken("the Dolphins plus seven at home"), "the Dolphins +7 at home")
        self.assertEqual(pp.normalize_spoken("Saints minus 3."), "Saints -3.")
        self.assertEqual(pp.normalize_spoken("Packers minus six and a half"), "Packers -6.5")
        self.assertEqual(pp.normalize_spoken("the under forty seven and a half"), "the under 47.5")
        self.assertEqual(pp.normalize_spoken("Over 51 for sure"), "over 51 for sure")
        self.assertEqual(pp.normalize_spoken("plus banana"), "plus banana")                     # not a number: left alone

    def test_a_price_is_not_swallowed_as_a_line(self):
        self.assertIn("+580", pp.normalize_spoken("pays out plus 580 at DraftKings"))


class TestCleaner(unittest.TestCase):
    VTT = ("WEBVTT - NFL Week 9 Betting Preview\n\n"
           "0:00:00.370 --> 0:00:04.689\n<v Speaker 1>The Action Network Podcast.\n\n"
           "0:00:04.769 --> 0:00:06.190\n<v Speaker 2>I'm going with the Dolphins\n\n"
           "7\n00:00:09.120 --> 00:00:18.980\n<v Speaker 3>plus three and a half. [Music]\n")

    def test_vtt_becomes_running_text(self):
        t = pp.clean_transcript(self.VTT)
        self.assertEqual(t, "The Action Network Podcast. I'm going with the Dolphins plus three and a half.")

    def test_srt_and_plain_speaker_labels(self):
        srt = "1\n00:00:01,000 --> 00:00:03,000\nSpeaker 1: Lions minus three\n\n2\n00:00:03,500 --> 00:00:05,000\nDoug: I like it."
        self.assertEqual(pp.clean_transcript(srt), "Lions minus three I like it.")


class TestSpokenPicks(unittest.TestCase):
    def picks(self, text):
        return pp.spoken_picks(text, POSTED, GAMES)

    def got(self, text):
        return {(p["market"], p["side"], p["line"]) for p in self.picks(text)}

    def test_a_card_read_aloud(self):
        text = ("So Doug's going with the Packers minus six and a half. Evan is going with the Dolphins plus three and a half. "
                "And I'm going to take the under forty seven and a half in the Lions game.")
        self.assertEqual(self.got(text), {("spread", "GB", -6.5), ("spread", "MIA", 3.5), ("total", "under", 47.5)})

    def test_a_description_without_a_cue_is_not_a_pick(self):
        self.assertEqual(self.picks("Detroit is a minus three favorite and the total is forty seven and a half."), [])

    def test_negations_and_questions_are_not_picks(self):
        for t in ["I don't like the Packers minus six and a half.", "I wouldn't touch the Dolphins plus three and a half.",
                  "Would I take the Dolphins plus three and a half?", "I'm going to fade the Lions minus three."]:
            self.assertEqual(self.picks(t), [], t)

    def test_a_far_number_is_not_this_games_line(self):
        self.assertEqual(self.picks("I like the Dolphins plus seven."), [])                    # market MIA +3: seven is four points away

    def test_cue_words_in_other_sentences_do_not_leak(self):
        self.assertEqual(self.picks("I like the weather in Miami. The Dolphins plus three and a half is the number."), [])

    def test_a_show_that_posted_after_kickoff_gets_nothing(self):
        self.assertEqual(pp.spoken_picks("I like the Dolphins plus three and a half.", "2026-11-08T19:00:00Z", GAMES), [])


def _rss(items):
    body = "".join(
        f"<item><title>{t}</title><guid>{g}</guid><link>https://shows.example/{g}</link><pubDate>{d}</pubDate><description>{desc}</description>"
        + (f'<podcast:transcript url="{tr}" type="text/vtt" />' if tr else "") + "</item>"
        for t, g, d, desc, tr in items)
    return ('<?xml version="1.0"?><rss version="2.0" xmlns:podcast="https://podcastindex.org/namespace/1.0">'
            f"<channel><title>Show</title>{body}</channel></rss>")


FRIDAY = "Fri, 06 Nov 2026 12:00:00 +0000"
VTT = ("WEBVTT\n\n00:00:01.000 --> 00:00:05.000\n<v Speaker 1>Okay my picks. I'm going with the Dolphins plus three and a half\n\n"
       "00:00:05.000 --> 00:00:09.000\n<v Speaker 1>and I like the Packers minus six and a half.\n")


class TestFeed(unittest.TestCase):
    def test_parse_feed(self):
        eps = pp.parse_feed(_rss([("Week 9 picks", "g1", FRIDAY, "Lions -3 lock", "https://t.example/1.vtt"), ("No notes", "g2", FRIDAY, "", None)]))
        self.assertEqual([e["guid"] for e in eps], ["g1", "g2"])
        self.assertEqual(eps[0]["published"], "2026-11-06T12:00:00+00:00")
        self.assertEqual(eps[0]["transcripts"], [("https://t.example/1.vtt", "text/vtt")])
        self.assertEqual(eps[1]["transcripts"], [])

    def test_transcript_type_preference(self):
        self.assertEqual(pp.pick_transcript([("a", "application/srt"), ("b", "text/vtt"), ("c", "text/plain")]), "b")
        self.assertIsNone(pp.pick_transcript([]))


class TestFetch(unittest.TestCase):
    SHOWS = [{"id": "an", "name": "Action", "feed": "https://f.example/an.xml"}, {"id": "other", "name": "Other", "feed": "https://f.example/other.xml"}]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = pathlib.Path(self.tmp.name) / "l.csv"
        self.state = {"seen": [], "last_fetch": None, "last_note": None}

    def tearDown(self):
        self.tmp.cleanup()

    def getter(self, extra=None):
        pages = {"https://f.example/an.xml": _rss([("NFL Week 9 Betting Preview", "ep1", FRIDAY, "Talking about the games", "https://t.example/ep1.vtt"),
                                                   ("Old show", "ep0", "Mon, 12 Oct 2026 12:00:00 +0000", "x", None)]),
                 "https://f.example/other.xml": _rss([("Week 9 Best Bets", "o1", FRIDAY, "Lions -3 is the play", None)]),
                 "https://t.example/ep1.vtt": VTT, **(extra or {})}

        def get(url):
            v = pages[url]
            if isinstance(v, Exception):
                raise v
            return v
        return get

    def test_picks_come_from_the_transcript_and_show_notes_as_the_show(self):
        r = pp.fetch(self.SHOWS, GAMES, self.state, self.getter(), self.log, NOW)
        df = sp.load(self.log)
        got = {(x.source, x.market, x.side, x.line) for x in df.itertuples()}
        self.assertEqual(got, {("pod:an", "spread", "MIA", 3.5), ("pod:an", "spread", "GB", -6.5), ("pod:other", "spread", "DET", -3.0)})
        self.assertEqual(set(df["platform"]), {"podcast"})
        self.assertEqual(r["episodes"], 2)                                            # the October episode is outside the lookback
        self.assertIn("ep1", self.state["seen"])
        self.assertNotIn("ep0", self.state["seen"])

    def test_an_episode_is_read_once(self):
        pp.fetch(self.SHOWS, GAMES, self.state, self.getter(), self.log, NOW)
        again = pp.fetch(self.SHOWS, GAMES, self.state, self.getter(), self.log, NOW)
        self.assertEqual((again["episodes"], again["picks_found"], again["added"]), (0, 0, 0))

    def test_one_bad_feed_or_transcript_does_not_stop_the_rest(self):
        get = self.getter({"https://f.example/other.xml": urllib.error.URLError("down")})
        r = pp.fetch(self.SHOWS, GAMES, self.state, get, self.log, NOW)
        self.assertIn("other: feed not read", r["note"])
        self.assertEqual(len(sp.load(self.log)), 2)                                   # the Action Network picks still came through
        self.state["seen"] = []
        r = pp.fetch(self.SHOWS, GAMES, self.state, self.getter({"https://t.example/ep1.vtt": urllib.error.URLError("gone")}), self.log, NOW)
        self.assertIn("transcript", r["note"])

    def test_the_latest_list_notes_transcripts_and_picks(self):
        r = pp.fetch(self.SHOWS, GAMES, self.state, self.getter(), self.log, NOW)
        by = {e["id"]: e for e in r["latest"] if e["title"] != "Old show"}
        self.assertTrue(by["an"]["transcript"])
        self.assertEqual(by["an"]["picks"], 2)
        self.assertFalse(by["other"]["transcript"])
        self.assertEqual(by["other"]["picks"], 1)
        self.assertEqual(by["other"]["url"], "https://shows.example/o1")

    def test_state_latest_and_block_round_trip(self):
        r = pp.fetch(self.SHOWS, GAMES, self.state, self.getter(), self.log, NOW)
        d = pathlib.Path(self.tmp.name)
        pp.save_state(self.state, d / "s.json")
        pp.save_latest(r["latest"], d / "l.json")
        self.assertEqual(pp.load_state(d / "s.json")["seen"], self.state["seen"])
        b = pp.block(self.SHOWS, pp.load_state(d / "s.json"), pp.load_latest(d / "l.json"))
        json.dumps(b)
        self.assertEqual([s["name"] for s in b["shows"]], ["Action", "Other"])
        self.assertEqual(pp.load_state(d / "missing.json")["seen"], [])


if __name__ == "__main__":
    unittest.main()
