"""
podcast_picks.py
================
Picks from betting podcasts, read from what the PUBLISHER hands out in the show's own RSS feed: the episode's show notes and, for shows that
publish one, its transcript (a `podcast:transcript` link: WebVTT, SubRip or plain text). No audio is downloaded or transcribed, and nothing
outside the feed is scraped.

    python podcast_picks.py fetch       # what the refresh workflow runs
    python podcast_picks.py check       # which configured shows publish transcripts, and their latest episodes

WHAT COUNTS AS A SPOKEN PICK
----------------------------
A transcript is conversation, and a team and a number in the same breath are often not a pick ("Detroit is a minus seven favorite"). A sentence is
read for picks only when it
  - carries a pick cue: "I like", "I'm taking", "going with", "best bet", "my play", "I'll lay", "backing", "leaning" ...
  - and carries none of the negations: "don't like", "wouldn't", "fade", "stay away", "pass on" ..., and is not a question
and then the usual rules apply (social_picks.py): an explicit spread, total or moneyline, posted before kickoff and within the week's window, the number
near the market's, and the show's FIRST call on a game counts. Spoken numbers are converted ("plus seven" -> +7, "minus six and a half" -> -6.5,
"forty seven and a half" -> 47.5). The pick belongs to the SHOW (pod:<id>): transcripts usually do not say which host said it. This is a heuristic
reader of speech, so it will miss picks (phrased unusually) and may misread a few; the show's graded record is what tells you whether its picks are
worth anything, and a reader that is noise would simply show a coin-flip record.
Show notes are read like a post (title + description). Many shows put no pick there; those that name their picks in prose are caught the same way.
"""

import argparse
import datetime
import email.utils
import html
import json
import os
import pathlib
import re
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

import pandas as pd

import social_picks as sp

ROOT = pathlib.Path(__file__).parent
STATE = ROOT / "tracking" / "podcast_state.json"
LATEST = ROOT / "tracking" / "podcast_latest.json"
USER_AGENT = "ff-model-personal-research/1.0 (+https://github.com/hadantes-lab/ff-model)"
LOOKBACK_DAYS = 8
KEEP_SEEN = 400
SHOWN_PER_FEED = 4
MAX_BYTES = 40_000_000                     # a long-running show's feed lists every episode ever (Action Network's is several MB); transcripts are far smaller
TRANSCRIPT_TYPES = ("text/vtt", "text/plain", "application/srt", "application/x-subrip")        # in order of preference
NS = {"pc": "https://podcastindex.org/namespace/1.0", "itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd"}

CUE = re.compile(r"\b(going with|go with|gonna go with|i like|i love|i'm taking|i am taking|i'll take|i will take|taking the|take the|my pick|my picks|"
                 r"best bet|my play|my bet|i'm laying|i'll lay|laying the|lay the|i'm on the|i'm on|lean|leaning|back the|backing|i'm betting|i'll bet|"
                 r"bet the|play the|i'm playing|i'm going to take|i'd take|i would take|locking in|i'm locking|my lock)\b", re.I)
NEGATION = re.compile(r"\b(don't like|do not like|didn't like|wouldn't|would not|won't|not touching|stay away|staying away|fade|fading|pass on|passing on|"
                      r"hate|can't back|cannot back|not a fan|avoid|no way|not betting|not taking|not going with|don't love|not laying)\b", re.I)

_UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen "
                                     "eighteen nineteen".split())}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70}
_WORD = "|".join(sorted(list(_UNITS) + list(_TENS), key=len, reverse=True))
_NUM = rf"(?:\d{{1,3}}(?:\.\d)?|(?:{_WORD})(?:[\s-](?:{_WORD}))?)(?:\s+and\s+a\s+half|\s+point\s+(?:five|5))?"
_SIGNED = re.compile(rf"\b(?P<sign>plus|minus)\s+(?P<num>{_NUM})\b", re.I)
_OVERUNDER = re.compile(rf"\b(?P<side>over|under)\s+(?P<num>{_NUM})\b", re.I)


# ---- spoken numbers -----------------------------------------------------------------------------------------------------
def words_to_number(spoken: str):
    """'six and a half' -> 6.5, 'forty seven' -> 47, '3.5' -> 3.5, 'twenty-one point five' -> 21.5; None if it is not a number."""
    s = spoken.lower().strip()
    half = 0.0
    m = re.search(r"\s+and\s+a\s+half$|\s+point\s+(?:five|5)$", s)
    if m:
        half, s = 0.5, s[:m.start()]
    if re.fullmatch(r"\d{1,3}(?:\.\d)?", s):
        return float(s) + half
    parts = re.split(r"[\s-]+", s)
    if len(parts) == 1 and parts[0] in _UNITS:
        return _UNITS[parts[0]] + half
    if parts[0] in _TENS:
        if len(parts) == 1:
            return _TENS[parts[0]] + half
        if len(parts) == 2 and parts[1] in _UNITS and _UNITS[parts[1]] < 10:
            return _TENS[parts[0]] + _UNITS[parts[1]] + half
    return None


def _fmt(n: float) -> str:
    return str(int(n)) if n == int(n) else str(n)


def normalize_spoken(text: str) -> str:
    """'Dolphins plus seven' -> 'Dolphins +7'; 'minus six and a half' -> '-6.5'; 'under forty seven and a half' -> 'under 47.5'."""
    def signed(m):
        n = words_to_number(m["num"])
        return m.group(0) if n is None else ("+" if m["sign"].lower() == "plus" else "-") + _fmt(n)

    def total(m):
        n = words_to_number(m["num"])
        return m.group(0) if n is None else f"{m['side'].lower()} {_fmt(n)}"
    return _OVERUNDER.sub(total, _SIGNED.sub(signed, text))


# ---- transcripts --------------------------------------------------------------------------------------------------------
def clean_transcript(raw: str) -> str:
    """WebVTT / SubRip / plain transcript -> one running text: timings, cue numbers, speaker and markup tags removed."""
    out = []
    for line in raw.splitlines():
        t = line.strip()
        if not t or t.upper().startswith(("WEBVTT", "NOTE", "STYLE", "REGION")) or "-->" in t or re.fullmatch(r"\d+", t):
            continue
        if re.fullmatch(r"\[?\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?\]?", t):
            continue
        t = re.sub(r"<[^>]+>", "", t)                                   # <v Speaker 1>, <c>, <i>
        t = re.sub(r"^\s*(?:\[?Speaker \d+\]?:?|[A-Z][A-Za-z .]{0,25}:)\s+", "", t)           # 'Speaker 2:' / 'Doug:' prefixes
        t = re.sub(r"\[(?:music|applause|laughter|inaudible)[^\]]*\]", " ", t, flags=re.I)
        out.append(t)
    return re.sub(r"\s+", " ", " ".join(out)).strip()


def spoken_picks(text: str, posted_at, games: pd.DataFrame) -> list:
    """Game picks in a running transcript, under the cue rules in the module docstring."""
    out = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if not CUE.search(sentence) or NEGATION.search(sentence) or sentence.rstrip().endswith("?"):       # a question is not a pick
            continue
        out += sp.parse_picks(normalize_spoken(sentence), posted_at, games)
    return out


# ---- feeds --------------------------------------------------------------------------------------------------------------
def fetch_url(url: str, timeout: int = 60) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read(MAX_BYTES).decode("utf-8", errors="replace")


def parse_feed(xml_text: str) -> list:
    """Episodes: {guid, title, published (UTC ISO), link, description, transcripts: [(url, type)]}, newest first as published."""
    root = ET.fromstring(xml_text.encode("utf-8"))
    out = []
    for it in root.findall("./channel/item"):
        try:
            pub = email.utils.parsedate_to_datetime(it.findtext("pubDate") or "")
            pub = (pub if pub.tzinfo else pub.replace(tzinfo=datetime.timezone.utc)).astimezone(datetime.timezone.utc).isoformat()
        except (TypeError, ValueError):
            continue
        desc = html.unescape(re.sub(r"<[^>]+>", " ", it.findtext("description") or ""))
        out.append({"guid": (it.findtext("guid") or it.findtext("link") or it.findtext("title") or "").strip(), "title": (it.findtext("title") or "").strip(),
                    "published": pub, "link": (it.findtext("link") or "").strip(), "description": re.sub(r"\s+", " ", desc).strip(),
                    "transcripts": [(html.unescape(t.get("url", "")), (t.get("type") or "").lower()) for t in it.findall("pc:transcript", NS) if t.get("url")]})
    return out


def pick_transcript(options: list):
    for kind in TRANSCRIPT_TYPES:
        for url, t in options:
            if t == kind:
                return url
    return options[0][0] if options else None


def load_podcasts(config_path=None) -> list:
    path = pathlib.Path(config_path or sp.CONFIG)
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cfg = {}
    return [p for p in cfg.get("podcasts", []) if p.get("id") and p.get("feed")]


def load_state(path=None) -> dict:
    try:
        s = json.loads(pathlib.Path(path or STATE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        s = {}
    return {"seen": s.get("seen", []), "last_fetch": s.get("last_fetch"), "last_note": s.get("last_note")}


def save_state(state: dict, path=None):
    path = pathlib.Path(path or STATE)
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({**state, "seen": state["seen"][-KEEP_SEEN:]}, indent=1), encoding="utf-8")


def fetch(shows: list, games: pd.DataFrame, state: dict, get=fetch_url, log_path=None, now: pd.Timestamp = None) -> dict:
    """
    New episodes (published in the last LOOKBACK_DAYS, not yet read) of every show -> picks from the transcript when there is one and from the
    show notes always, logged as pod:<id>. One show failing is noted and the rest go on. -> {episodes, picks_found, added, latest, note}.
    """
    now = now or pd.Timestamp.now(tz="UTC")
    seen, posts, latest, notes, parsed = set(state["seen"]), [], [], [], {}
    for show in shows:
        sid = f"pod:{show['id']}"
        try:
            eps = parsed[show["id"]] = parse_feed(get(show["feed"]))
        except (urllib.error.URLError, ET.ParseError, ValueError, OSError) as ex:
            notes.append(f"{show['id']}: feed not read ({type(ex).__name__})")
            continue
        for ep in eps:
            age = now - pd.Timestamp(ep["published"])
            if age > pd.Timedelta(days=LOOKBACK_DAYS) or ep["guid"] in seen:
                continue
            seen.add(ep["guid"])
            posts.append({"source": sid, "post_id": ep["guid"], "posted_at": ep["published"], "platform": "podcast",
                          "text": ep["title"] + "\n" + ep["description"]})
            url = pick_transcript(ep["transcripts"])
            if url:
                try:
                    text = clean_transcript(get(url))
                except (urllib.error.URLError, ValueError, OSError) as ex:
                    notes.append(f"{show['id']}: transcript for '{ep['title'][:40]}' not read ({type(ex).__name__})")
                    continue
                for pk in spoken_picks(text, ep["published"], games):
                    posts.append({"source": sid, "post_id": ep["guid"], "posted_at": ep["published"], "platform": "podcast",
                                  "pre_parsed": pk})
    rows = []
    for p in posts:
        if "pre_parsed" in p:
            pk = p["pre_parsed"]
            rows.append({"source": p["source"], "platform": "podcast", "post_id": p["post_id"], "posted_at": sp.to_utc(p["posted_at"]).isoformat(), **pk,
                         "text": "spoken: " + f"{pk['side']} {pk['line'] if pk['line'] is not None else ''}".strip(), "result": float("nan"),
                         "profit": float("nan"), "graded_at": float("nan")})
        else:
            rows += sp.rows_from_posts([p], games, "podcast")
    added = sp.log_picks(rows, log_path)
    # the page's list of recent episodes, with how many picks were read from each
    per_ep = {}
    for r in rows:
        per_ep[r["post_id"]] = per_ep.get(r["post_id"], 0) + 1
    for show in shows:
        for ep in parsed.get(show["id"], [])[:SHOWN_PER_FEED]:
            latest.append({"show": show.get("name") or show["id"], "id": show["id"], "source": f"pod:{show['id']}", "title": ep["title"],
                           "published": ep["published"], "url": ep["link"] or None, "transcript": bool(ep["transcripts"]),
                           "picks": per_ep.get(ep["guid"]) or None})
    state["seen"] = list(seen)
    state["last_fetch"] = now.isoformat()
    state["last_note"] = "; ".join(notes) or None
    return {"episodes": len({p["post_id"] for p in posts}), "picks_found": len(rows), "added": added, "latest": latest, "note": state["last_note"]}


def save_latest(latest: list, path=None):
    path = pathlib.Path(path or LATEST)
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                                "episodes": sorted(latest, key=lambda e: e["published"], reverse=True)}, indent=1), encoding="utf-8")


def load_latest(path=None) -> dict:
    try:
        return json.loads(pathlib.Path(path or LATEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"generated": None, "episodes": []}


def block(shows: list, state: dict = None, latest: dict = None) -> dict:
    state = load_state() if state is None else state
    latest = load_latest() if latest is None else latest
    return {"shows": [{"id": s["id"], "name": s.get("name") or s["id"]} for s in shows], "episodes": latest.get("episodes", []),
            "last_fetch": state.get("last_fetch"), "note": state.get("last_note")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["fetch", "check"])
    args = ap.parse_args()
    shows = load_podcasts()
    if not shows:
        print("no podcasts listed (social_sources.json \"podcasts\")")
        return
    if args.command == "check":
        for s in shows:
            try:
                eps = parse_feed(fetch_url(s["feed"]))
            except Exception as ex:
                print(f"{s['id']}: {type(ex).__name__}: {ex}")
                continue
            n = sum(bool(e["transcripts"]) for e in eps)
            print(f"{s['id']:16s} {len(eps)} episodes, {n} with transcripts; latest: {eps[0]['title'][:70] if eps else None}")
        return
    import nflreadpy as nfl
    games = sp.load_games([int(nfl.get_current_season())])
    state = load_state()
    r = fetch(shows, games, state)
    save_state(state)
    save_latest(r["latest"])
    sp.ensure_files()
    print({k: v for k, v in r.items() if k != "latest"}, f"graded {sp.grade(games)}")


if __name__ == "__main__":
    main()
