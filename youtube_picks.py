"""
youtube_picks.py
================
Picks from YouTube channels, through the one route YouTube permits: the official Data API v3 (a free key; the daily quota of 10,000 units is
never close, a refresh uses about 3 units a channel).

    python youtube_picks.py fetch        # what the refresh workflow runs: new uploads from each channel, picks found in titles/descriptions
    python youtube_picks.py check        # resolve the channels and show who they are and their latest videos (run this once, to verify the list)
    python youtube_picks.py notes FILE   # picks you heard in a video, one per line: channel | when | picks | link

WHAT THIS CAN AND CANNOT READ
-----------------------------
Most picks on these channels are SPOKEN. The captions are not available through the official API (downloading them needs the channel owner's
sign-in), and the endpoints that caption scrapers use are ones YouTube's robots.txt disallows, as it does the RSS feeds and the search page.
So nothing here listens to a video. What it does read is each new video's title and description, with exactly the rules used for X posts
(social_picks.py): an explicit spread, total or moneyline, posted before kickoff, near the market's number, a source's first call counts.
Many channels put no picks there, and then the log gets nothing from them; the Social Picks tab still lists their new videos so you can jump to them.
When a host says "I'm on the Lions minus three", `notes` is how it gets into the record, graded like any other pick under the channel's name.

SETUP
-----
List channels in social_sources.json ("youtube_channels": ["@handle", ...]) or the YOUTUBE_CHANNELS variable, and set the YOUTUBE_API_KEY secret
(Google Cloud console -> APIs -> YouTube Data API v3 -> create a key; no billing needed). Without a key nothing is fetched.
"""

import argparse
import datetime
import json
import os
import pathlib
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

import social_picks as sp

ROOT = pathlib.Path(__file__).parent
STATE = ROOT / "tracking" / "youtube_state.json"
LATEST = ROOT / "tracking" / "youtube_latest.json"
API = "https://www.googleapis.com/youtube/v3"
RECENT = 15                    # uploads read per channel per run
KEEP_SEEN = 600
VIDEOS_SHOWN = 6               # per channel, on the page
SHOW_DAYS = 14


def source_id(handle: str) -> str:
    return "yt:" + handle.lstrip("@").lower()


def load_channels(config_path=None) -> list:
    path = pathlib.Path(config_path or sp.CONFIG)
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cfg = {}
    handles = [h.lstrip("@") for h in cfg.get("youtube_channels", [])]
    handles += [h.strip().lstrip("@") for h in os.environ.get("YOUTUBE_CHANNELS", "").split(",") if h.strip()]
    seen, out = set(), []
    for h in handles:
        if h.lower() not in seen:
            seen.add(h.lower())
            out.append(h)
    return out


def load_state(path=None) -> dict:
    try:
        s = json.loads(pathlib.Path(path or STATE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        s = {}
    return {"channels": s.get("channels", {}), "seen": s.get("seen", []), "units": s.get("units", 0),
           "last_fetch": s.get("last_fetch"), "last_note": s.get("last_note")}


def save_state(state: dict, path=None):
    path = pathlib.Path(path or STATE)
    path.parent.mkdir(exist_ok=True)
    state = {**state, "seen": state["seen"][-KEEP_SEEN:]}
    path.write_text(json.dumps(state, indent=1), encoding="utf-8")


# ---- the API ------------------------------------------------------------------------------------------------------------
def _http(path: str, params: dict, key: str):
    url = f"{API}/{path}?" + urllib.parse.urlencode({**params, "key": key})
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "ff-model-personal-research/1.0"}), timeout=30) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode("utf-8"))
        except (ValueError, OSError):
            body = {}
        return e.code, body


class ApiError(Exception):
    pass


def _call(path: str, params: dict, key: str, state: dict, get) -> dict:
    status, body = get(path, params, key)
    state["units"] = state.get("units", 0) + 1                      # every list call costs 1 quota unit
    if status != 200:
        reason = ((body.get("error") or {}).get("errors") or [{}])[0].get("reason") or (body.get("error") or {}).get("message") or ""
        raise ApiError(f"YouTube API returned {status}" + (f" ({reason})" if reason else ""))
    return body


def resolve(handle: str, key: str, state: dict, get=_http) -> dict:
    """{id, title, uploads} for a handle, cached: the lookup is paid for once."""
    cached = state["channels"].get(handle.lower())
    if cached:
        return cached
    body = _call("channels", {"part": "snippet,contentDetails", "forHandle": "@" + handle}, key, state, get)
    items = body.get("items") or []
    if not items:
        raise ApiError(f"no channel found for @{handle}")
    it = items[0]
    info = {"id": it["id"], "title": it["snippet"]["title"], "uploads": it["contentDetails"]["relatedPlaylists"]["uploads"], "handle": handle}
    state["channels"][handle.lower()] = info
    return info


def recent_uploads(info: dict, key: str, state: dict, get=_http) -> list:
    body = _call("playlistItems", {"part": "snippet,contentDetails", "playlistId": info["uploads"], "maxResults": RECENT}, key, state, get)
    out = []
    for it in body.get("items", []):
        sn = it["snippet"]
        out.append({"video_id": it["contentDetails"]["videoId"], "title": sn.get("title", ""), "description": sn.get("description", ""),
                    "published": it["contentDetails"].get("videoPublishedAt") or sn.get("publishedAt")})
    return out


def full_descriptions(ids: list, key: str, state: dict, get=_http) -> dict:
    """Complete descriptions (the playlist listing can cut them short), 50 videos a call."""
    out = {}
    for i in range(0, len(ids), 50):
        body = _call("videos", {"part": "snippet", "id": ",".join(ids[i:i + 50])}, key, state, get)
        for it in body.get("items", []):
            out[it["id"]] = it["snippet"].get("description", "")
    return out


# ---- fetch --------------------------------------------------------------------------------------------------------------
def fetch(handles: list, key: str, games: pd.DataFrame, state: dict, get=_http, log_path=None) -> dict:
    """
    New uploads from every channel -> picks found in title+description (logged as source yt:<handle>) and the latest-videos list for the page.
    One channel failing (wrong handle, quota) is noted and the others go on.
    """
    notes, videos, posts = [], [], []
    seen = set(state["seen"])
    for h in handles:
        try:
            info = resolve(h, key, state, get)
            ups = recent_uploads(info, key, state, get)
        except ApiError as ex:
            notes.append(f"@{h}: {ex}")
            if "quota" in str(ex).lower():
                break
            continue
        new = [u for u in ups if u["video_id"] not in seen]
        try:
            desc = full_descriptions([u["video_id"] for u in new], key, state, get) if new else {}
        except ApiError as ex:
            notes.append(f"@{h}: {ex}")
            desc = {}
        found = {}
        for u in new:
            text = u["title"] + "\n" + desc.get(u["video_id"], u["description"])
            posts.append({"source": source_id(h), "post_id": u["video_id"], "posted_at": u["published"], "text": text, "platform": "youtube"})
            seen.add(u["video_id"])
        rows = sp.rows_from_posts([p for p in posts if p["source"] == source_id(h)], games, "youtube")
        for r in rows:
            found[r["post_id"]] = found.get(r["post_id"], 0) + 1
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=SHOW_DAYS)
        for u in ups[:VIDEOS_SHOWN]:
            if u["published"] and pd.Timestamp(u["published"]) >= cutoff:
                videos.append({"channel": info["title"], "handle": h, "source": source_id(h), "title": u["title"], "published": u["published"],
                               "url": f"https://www.youtube.com/watch?v={u['video_id']}", "picks": found.get(u["video_id"], 0) or None})
    rows = sp.rows_from_posts(posts, games, "youtube")
    added = sp.log_picks(rows, log_path)
    state["seen"] = list(seen)
    state["last_fetch"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    state["last_note"] = "; ".join(notes) or None
    return {"new_videos": len(posts), "picks_found": len(rows), "added": added, "videos": videos, "note": state["last_note"]}


def save_latest(videos: list, path=None):
    path = pathlib.Path(path or LATEST)
    path.parent.mkdir(exist_ok=True)
    videos = sorted(videos, key=lambda v: v["published"] or "", reverse=True)
    path.write_text(json.dumps({"generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "videos": videos}, indent=1),
                    encoding="utf-8")


def load_latest(path=None) -> dict:
    try:
        return json.loads(pathlib.Path(path or LATEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"generated": None, "videos": []}


def block(handles: list, state: dict = None, latest: dict = None) -> dict:
    """What the Social Picks tab shows for YouTube."""
    state = load_state() if state is None else state
    latest = load_latest() if latest is None else latest
    return {"channels": [{"handle": h, "title": (state["channels"].get(h.lower()) or {}).get("title")} for h in handles],
            "videos": latest.get("videos", []), "key": bool(os.environ.get("YOUTUBE_API_KEY")), "units": state.get("units", 0),
            "last_fetch": state.get("last_fetch"), "note": state.get("last_note")}


# ---- picks you heard ----------------------------------------------------------------------------------------------------
def parse_notes(text: str) -> list:
    """
    One pick note per line:  channel | when | picks | link(optional).  `when` is when you heard it (a time with no zone is US Eastern); picks is free
    text read the same way as a post ("Lions -3, Over 47.5"). Blank lines and lines starting with # are skipped.
    """
    posts = []
    for n, line in enumerate(text.splitlines()):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            raise ValueError(f"line {n + 1}: expected  channel | when | picks | link")
        link = parts[3] if len(parts) > 3 and parts[3] else f"note-{n + 1}"
        posts.append({"source": source_id(parts[0]), "posted_at": parts[1], "text": parts[2], "post_id": link, "platform": "youtube"})
    return posts


def import_notes(path, games: pd.DataFrame, log_path=None) -> dict:
    posts = parse_notes(pathlib.Path(path).read_text(encoding="utf-8"))
    rows = sp.rows_from_posts(posts, games, "youtube")
    return {"notes": len(posts), "picks_found": len(rows), "added": sp.log_picks(rows, log_path)}


# ---- command line -------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["fetch", "check", "notes"])
    ap.add_argument("file", nargs="?")
    args = ap.parse_args()
    handles, key, state = load_channels(), os.environ.get("YOUTUBE_API_KEY"), load_state()

    if args.command == "notes":
        if not args.file:
            raise SystemExit("usage: youtube_picks.py notes FILE")
        import nflreadpy as nfl
        games = sp.load_games([int(nfl.get_current_season())])
        print(import_notes(args.file, games))
        print(f"graded {sp.grade(games)} picks")
        return
    if not key or not handles:
        print("no YOUTUBE_API_KEY set" if not key else "no channels listed (social_sources.json youtube_channels, or YOUTUBE_CHANNELS)")
        state["last_note"] = "YOUTUBE_API_KEY not set; nothing fetched" if not key else "no channels listed"
        save_state(state)
        if not LATEST.exists():
            save_latest([])
        return
    if args.command == "check":
        for h in handles:
            try:
                info = resolve(h, key, state)
                ups = recent_uploads(info, key, state)
                print(f"@{h}: {info['title']}  ({info['id']})")
                for u in ups[:3]:
                    print(f"    {u['published'][:10]}  {u['title'][:90]}")
            except ApiError as ex:
                print(f"@{h}: {ex}")
        save_state(state)
        return
    import nflreadpy as nfl
    games = sp.load_games([int(nfl.get_current_season())])
    result = fetch(handles, key, games, state)
    save_state(state)
    save_latest(result["videos"])
    sp.ensure_files()
    print({k: v for k, v in result.items() if k != "videos"}, f"graded {sp.grade(games)}", f"quota used so far: {state['units']} units")


if __name__ == "__main__":
    main()
