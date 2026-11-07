"""
free_picks.py
=============
Picks that cost nothing to read, pulled on every refresh, and the check of our own simulation against them.

    python free_picks.py fetch        # what the refresh workflow runs: read every source, log the picks, save tracking/free_picks_latest.json

SOURCES (each is a public page or API that answers a plain, honestly labelled request; nothing is logged into and no bot check is dodged)
  espn-fpi         ESPN's Football Power Index game projection (public JSON): predicted margin per game
  covers-computer  Covers' computer pick: predicted score per game, with its spread and total
  pickswise:<name> Pickswise's named handicappers: an explicit spread or total, at the number they picked at
  pt-median        ThePredictionTracker's median of ~60 computer models (spread and total), from its published CSV downloads
  nfelo, sagarin   two of those models on their own: nfelo (PredictionTracker's most accurate in 2024) and Sagarin
  sbd-ai           Sports Betting Dime's A.I. model spread picks (the week's picks article, found from their NFL hub page)
The ~60 PredictionTracker models are correlated with each other and with the market, so they enter as three voices, not sixty.
Not used: Dimers (serves its page to browsers only), NFLPickWatch (loads its picks from a private API), Vegas Insider (premium-locked alternate lines).

A projection becomes a pick only where it differs from the market line by MODEL_EDGE points or more; a human's pick is taken as stated. Every pick goes in
the same log as the X picks (tracking/social_picks.csv, platform "web"), so each source is graded by us on its own record and weighted by it
exactly the same way (see social_picks.py). The log keeps a source's FIRST call on a game; the page uses each source's LATEST (tracking/free_picks_latest.json),
because that is what the source says now.

THE HIGHLIGHT
-------------
For every pick our simulation makes (a spread and a total per game), count the sources that took the same side. When at least MIN_FREE picks exist on that
game and market and more than half back our side, the pick is "backed by the free-pick majority", and it is flagged on the page.
"""

import argparse
import datetime
import html
import json
import math
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

import social_picks as sp

ROOT = pathlib.Path(__file__).parent
LATEST = ROOT / "tracking" / "free_picks_latest.json"
USER_AGENT = "ff-model-personal-research/1.0 (+https://github.com/hadantes-lab/ff-model)"
MODEL_EDGE = 1.0                  # points a model's number must beat the market line by to count as a pick
MIN_FREE = 3                      # picks on a game and market before there is a "majority" to agree with
WEB_LEAD_HOURS = 192              # free sources publish early in the week: read picks for games up to 8 days out
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
ESPN_PREDICTOR = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{id}/competitions/{id}/predictor"
COVERS_URL = "https://www.covers.com/picks/nfl"
PICKSWISE_URL = "https://www.pickswise.com/nfl/predictions/"
PT_SPREADS = "http://www.thepredictiontracker.com/nflpredictions.csv"
PT_TOTALS = "http://www.thepredictiontracker.com/nfltotals.csv"
PT_SPREAD_MODELS = {"pt-median": "linemedian", "nfelo": "linenfelo", "sagarin": "linesag"}      # voice -> column (home-favored margin)
PT_TOTAL_MODELS = {"pt-median": "totmed", "sagarin": "totsag"}
SBD_HUB = "https://www.sportsbettingdime.com/nfl/"
POLITE_DELAY = 0.3                # seconds between requests to the same site

_NAME_RE = re.compile(rf"(?<![\w@])(?i:{sp._NAMES})(?![A-Za-z])")


# ---- http / text --------------------------------------------------------------------------------------------------------
def fetch(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def html_text(raw: str) -> str:
    """Visible text of an HTML fragment: scripts and styles dropped, tags to spaces, entities decoded, whitespace collapsed."""
    raw = re.sub(r"<script.*?</script>|<style.*?</style>", " ", raw, flags=re.S)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", raw))).strip()


def team_mentions(text: str) -> list:
    """Team codes in the order they are named (a nickname or city; 'Los Angeles Rams' is the Rams, 'New York Giants' the Giants)."""
    out = []
    for m in _NAME_RE.finditer(text):
        code = sp.NAME_TO_CODE[m.group(0).lower()]
        if not out or out[-1] != code:
            out.append(code)
    return out


def norm_code(token: str, codes: set):
    """A site's team abbreviation as our schedule's code (WSH -> WAS, LAR -> LA); None if it is not a team we know."""
    t = token.strip().upper()
    if t in codes:
        return t
    return sp.ABBREVIATIONS.get(t)


# ---- turning projections into picks -------------------------------------------------------------------------------------
def web_pick(source: str, g, market: str, side: str, line, odds=None) -> dict:
    return {"source": source, "platform": "web", **sp._pick(g, market, side, None if line is None else float(line), odds)}


def model_picks(source: str, g, margin_home: float, home_spread: float, total_pred: float = None, total_line: float = None) -> list:
    """
    Spread and total picks from a projection. `margin_home` is the predicted home margin; `home_spread` the market's (negative = home favored),
    so the home side covers by margin_home + home_spread. A pick needs MODEL_EDGE points of difference from the line.
    """
    out = []
    if home_spread is not None and not math.isnan(home_spread):
        edge = margin_home + home_spread
        if edge >= MODEL_EDGE:
            out.append(web_pick(source, g, "spread", g["home"], home_spread))
        elif edge <= -MODEL_EDGE:
            out.append(web_pick(source, g, "spread", g["away"], -home_spread))
    if total_pred is not None and total_line is not None and not math.isnan(total_line):
        edge = total_pred - total_line
        if edge >= MODEL_EDGE:
            out.append(web_pick(source, g, "total", "over", total_line))
        elif edge <= -MODEL_EDGE:
            out.append(web_pick(source, g, "total", "under", total_line))
    return out


def market_home_spread(g) -> float:
    sl = float(g["spread_line"])
    return float("nan") if math.isnan(sl) else -sl


def sane(g, market: str, line) -> bool:
    """A number far from the market's is another game, another sport or a typo."""
    if line is None:
        return True
    if market == "spread":
        return True
    tl = float(g["total_line"]) if not pd.isna(g["total_line"]) else float("nan")
    return 30 <= float(line) <= 75 and (math.isnan(tl) or abs(float(line) - tl) <= sp.TOTAL_TOLERANCE)


def sane_spread(g, side: str, line) -> bool:
    mkt = sp.team_line(g, side)
    return math.isnan(mkt) or abs(float(line) - mkt) <= sp.SPREAD_TOLERANCE


# ---- ESPN FPI -----------------------------------------------------------------------------------------------------------
def fetch_espn_fpi(games: pd.DataFrame, now: pd.Timestamp, get=fetch) -> list:
    sb = json.loads(get(ESPN_SCOREBOARD))
    codes = set(games["home"]) | set(games["away"])
    out = []
    for ev in sb.get("events", []):
        if ev.get("status", {}).get("type", {}).get("state") != "pre":
            continue
        comp = ev["competitions"][0]
        side = {c["homeAway"]: norm_code(c["team"]["abbreviation"], codes) for c in comp["competitors"]}
        if not side.get("home") or not side.get("away"):
            continue
        g = sp.find_game(games, side["home"], now, side["away"], WEB_LEAD_HOURS)
        if g is None:
            continue
        time.sleep(POLITE_DELAY)
        try:
            pred = json.loads(get(ESPN_PREDICTOR.format(id=ev["id"])))
        except (urllib.error.URLError, ValueError):
            continue
        stats = {s["name"]: s["value"] for s in pred.get("homeTeam", {}).get("statistics", [])}
        if "teamPredPtDiff" not in stats:
            continue
        out += model_picks("espn-fpi", g, float(stats["teamPredPtDiff"]), market_home_spread(g))
    return out


# ---- Covers computer picks ----------------------------------------------------------------------------------------------
_COVERS = re.compile(
    r"Team Predicted Score Spread Total ML (?P<a>[A-Z]{2,4}) (?P=a) (?P<as>\d{1,2}\.\d+) (?P<sp>[+-]?\d{1,2}(?:\.\d)?|PK)\b.*?"
    r"[ou](?P<tot>\d{2,3}(?:\.\d)?)\b.*?(?P<h>[A-Z]{2,4}) (?P=h) (?P<hs>\d{1,2}\.\d+)")


def parse_covers(raw: str, games: pd.DataFrame, now: pd.Timestamp) -> list:
    text = html_text(raw)
    codes = set(games["home"]) | set(games["away"])
    out = []
    for chunk in re.split(r"Computer Picks Trends", text):
        m = _COVERS.search(chunk)
        if not m:
            continue
        away, home = norm_code(m["a"], codes), norm_code(m["h"], codes)
        if not away or not home:
            continue
        g = sp.find_game(games, home, now, away, WEB_LEAD_HOURS)
        if g is None:
            continue
        away_line = 0.0 if m["sp"] == "PK" else float(m["sp"])                       # the spread as listed for the away team
        margin_home = float(m["hs"]) - float(m["as"])
        total_line = float(m["tot"])
        picks = model_picks("covers-computer", g, margin_home, -away_line, float(m["as"]) + float(m["hs"]), total_line)
        out += [p for p in picks if (p["market"] != "spread" or sane_spread(g, p["side"], p["line"])) and sane(g, p["market"], p["line"])]
    return out


def fetch_covers(games: pd.DataFrame, now: pd.Timestamp, get=fetch) -> list:
    return parse_covers(get(COVERS_URL), games, now)


# ---- Pickswise handicappers ---------------------------------------------------------------------------------------------
_PS_WHEN = re.compile(r"NFL (?:Sun|Mon|Tue|Wed|Thu|Fri|Sat), \w+ \d+ \d+:\d+ [AP]M (?P<pick>.{0,90})")
_PS_SPREAD = re.compile(r"^Spread - (?P<team>.+?) (?P<line>[+-]\d{1,2}(?:\.\d)?)(?: |$)")
_PS_TOTAL = re.compile(r"^(?P<side>Over|Under) (?P<line>\d{2,3}(?:\.\d)?)(?: |$)")


def parse_pickswise(raw: str, games: pd.DataFrame, now: pd.Timestamp) -> list:
    """One pick per handicapper per game and market. Player props and articles are not game picks and fall out of the patterns."""
    out, seen = [], set()
    for chunk in re.split(r'href="/handicappers/', raw)[1:]:
        slug = chunk[:chunk.index("/")] if "/" in chunk else ""
        text = html_text(chunk)[:600]
        m = _PS_WHEN.search(text)
        if not slug or not m:
            continue
        before = text[:m.start()]
        teams = team_mentions(before)
        if len(teams) < 2:
            continue
        away, home = teams[-2], teams[-1]
        g = sp.find_game(games, home, now, away, WEB_LEAD_HOURS)
        if g is None:
            continue
        source, pick = f"pickswise:{slug}", m["pick"]
        s, t = _PS_SPREAD.match(pick), _PS_TOTAL.match(pick)
        made = None
        if s:
            side = (team_mentions(s["team"]) or [None])[0]
            if side in (g["home"], g["away"]) and sane_spread(g, side, float(s["line"])):
                made = web_pick(source, g, "spread", side, float(s["line"]))
        elif t and sane(g, "total", float(t["line"])):
            made = web_pick(source, g, "total", t["side"].lower(), float(t["line"]))
        if made and (source, g["game_id"], made["market"]) not in seen:
            seen.add((source, g["game_id"], made["market"]))
            out.append(made)
    return out


def fetch_pickswise(games: pd.DataFrame, now: pd.Timestamp, get=fetch) -> list:
    return parse_pickswise(get(PICKSWISE_URL), games, now)


# ---- ThePredictionTracker: published CSVs of ~60 computer models ---------------------------------------------------------
def _pt_teams(row, games, now):
    """The game for a PredictionTracker row ('Jacksonville' at 'Baltimore'), or None."""
    road, home = team_mentions(str(row["road"])), team_mentions(str(row["home"]))
    if not road or not home:
        return None
    return sp.find_game(games, home[0], now, road[0], WEB_LEAD_HOURS)


def parse_predictiontracker(spreads_csv: str, totals_csv: str, games: pd.DataFrame, now: pd.Timestamp) -> list:
    """
    Each row is a game with every model's predicted home margin (home favored = positive, as nflverse) and the market line in `line`.
    Three voices come out of it: the median of all models, nfelo and Sagarin; a pick needs MODEL_EDGE points against the market line.
    """
    import io
    out = []
    sp_df = pd.read_csv(io.StringIO(spreads_csv))
    for _, row in sp_df.iterrows():
        g = _pt_teams(row, games, now)
        if g is None or pd.isna(row.get("line")):
            continue
        market_home = -float(row["line"])                      # home_spread: negative = home favored
        for name, col in PT_SPREAD_MODELS.items():
            if col in row and not pd.isna(row[col]):
                picks = model_picks(name, g, float(row[col]), market_home)
                out += [p for p in picks if sane_spread(g, p["side"], p["line"])]
    if totals_csv:
        tot = pd.read_csv(io.StringIO(totals_csv))
        for _, row in tot.iterrows():
            g = _pt_teams(row, games, now)
            if g is None or pd.isna(row.get("line")):
                continue
            for name, col in PT_TOTAL_MODELS.items():
                if col in row and not pd.isna(row[col]):
                    picks = model_picks(name, g, 0.0, float("nan"), float(row[col]), float(row["line"]))
                    out += [p for p in picks if sane(g, "total", p["line"])]
    return out


def fetch_predictiontracker(games: pd.DataFrame, now: pd.Timestamp, get=fetch) -> list:
    spreads = get(PT_SPREADS)
    time.sleep(POLITE_DELAY)
    try:
        totals = get(PT_TOTALS)
    except (urllib.error.URLError, ValueError):
        totals = ""
    return parse_predictiontracker(spreads, totals, games, now)


# ---- Sports Betting Dime A.I. spread picks -------------------------------------------------------------------------------
_SBD_PICK = re.compile(r"ATS Pick Pick: (?P<code>[A-Z]{2,4}) (?P<line>[+-]\d{1,2}(?:\.\d)?)")


def sbd_week(games: pd.DataFrame, now: pd.Timestamp):
    """The week whose games are next to kick off."""
    up = games[games["kickoff"] > now]
    return None if up.empty else int(up.sort_values("kickoff").iloc[0]["week"])


def sbd_article_url(hub_html: str, week: int):
    m = re.search(rf'href="([^"]*/week-{week}-nfl-picks-against-the-spread[^"]*)"', hub_html)
    if not m:
        return None
    url = html.unescape(m.group(1))
    return url if url.startswith("http") else "https://www.sportsbettingdime.com" + url


def parse_sbd(raw: str, games: pd.DataFrame, now: pd.Timestamp) -> list:
    codes = set(games["home"]) | set(games["away"])
    out, seen = [], set()
    for m in _SBD_PICK.finditer(html_text(raw)):
        team = norm_code(m["code"], codes)
        if not team:
            continue
        g = sp.find_game(games, team, now, None, WEB_LEAD_HOURS)
        if g is None or g["game_id"] in seen or not sane_spread(g, team, float(m["line"])):
            continue
        seen.add(g["game_id"])
        out.append(web_pick("sbd-ai", g, "spread", team, float(m["line"])))
    return out


def fetch_sbd(games: pd.DataFrame, now: pd.Timestamp, get=fetch) -> list:
    week = sbd_week(games, now)
    if week is None:
        return []
    url = sbd_article_url(get(SBD_HUB), week)
    if not url:
        return []
    time.sleep(POLITE_DELAY)
    return parse_sbd(get(url), games, now)


SOURCES = {"espn-fpi": fetch_espn_fpi, "covers-computer": fetch_covers, "pickswise": fetch_pickswise,
           "predictiontracker": fetch_predictiontracker, "sbd-ai": fetch_sbd}


# ---- run it -------------------------------------------------------------------------------------------------------------
def fetch_all(games: pd.DataFrame, now: pd.Timestamp = None, get=fetch, sources: dict = None) -> dict:
    """Every source, each isolated: one that fails or changes its page costs only its own picks. -> {generated, sources: {name: {ok, picks, note}}, picks}."""
    now = now or pd.Timestamp.now(tz="UTC")
    picks, status = [], {}
    for name, fn in (sources or SOURCES).items():
        try:
            got = fn(games, now, get)
            status[name] = {"ok": True, "picks": len(got), "note": None if got else "no picks found (page changed, or no games yet)"}
            picks += got
        except Exception as ex:                                   # a changed page must never stop the refresh
            status[name] = {"ok": False, "picks": 0, "note": f"{type(ex).__name__}: {str(ex)[:120]}"}
        time.sleep(POLITE_DELAY)
    return {"generated": now.isoformat(), "sources": status, "picks": picks}


def rows_for_log(picks: list, now: pd.Timestamp) -> list:
    stamp = now.isoformat()
    return [{**{k: p[k] for k in ("source", "platform", "game_id", "game", "season", "week", "kickoff", "market", "side", "line", "odds")},
             "post_id": f"web-{now.date()}-{p['source']}-{p['game_id']}-{p['market']}", "posted_at": stamp,
             "text": f"{p['source']}: {p['side']} {p['line'] if p['line'] is not None else ''}".strip(), "result": np.nan, "profit": np.nan,
             "graded_at": np.nan} for p in picks]


def save_latest(result: dict, path=None):
    path = pathlib.Path(path or LATEST)
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=1, default=lambda o: None if o is None or (isinstance(o, float) and math.isnan(o)) else str(o)), encoding="utf-8")


def load_latest(path=None) -> dict:
    try:
        return json.loads(pathlib.Path(path or LATEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"generated": None, "sources": {}, "picks": []}


# ---- the current picks, and our simulation against them -----------------------------------------------------------------
def upcoming(df: pd.DataFrame, latest: dict, season: int, now=None) -> pd.DataFrame:
    """
    Every pick on a game still to be played, one row per source, game and market: the log's picks (X accounts, imports) with each web source's
    LATEST call in place of its first, since that is what it says now.
    """
    now = sp.to_utc(now) if now is not None else pd.Timestamp.now(tz="UTC")
    d = df[(df["season"] == season) & df["result"].isna()].copy()
    lat = pd.DataFrame(latest.get("picks", []))
    if len(lat):
        for c in sp.COLUMNS:
            if c not in lat:
                lat[c] = np.nan
        lat = lat[sp.COLUMNS]
        key = ["source", "game_id", "market"]
        mask = d.set_index(key).index.isin(lat.set_index(key).index)
        d = pd.concat([d[~mask], lat], ignore_index=True)
    d = d[pd.to_datetime(d["kickoff"], utc=True, format="ISO8601") > now]
    return d.reset_index(drop=True)


def annotate(entries: list, eff: pd.DataFrame) -> int:
    """
    Tag each simulation pick (`entries`: game, type 'Spread'|'Total', side) with how the sources on that game and market lined up:
    entry["free"] = {agree, against, of, majority_side, backed, sources}. `backed` means MIN_FREE+ picks and more than half on our side.
    -> how many are backed.
    """
    n_backed = 0
    for e in entries:
        market = e["type"].lower()
        d = eff[(eff["game"] == e["game"]) & (eff["market"] == market)]
        of = len(d)
        agree = int((d["side"] == e.get("side")).sum())
        sides = d["side"].value_counts()
        e["free"] = {"agree": agree, "against": of - agree, "of": of, "majority_side": sides.index[0] if of and sides.iloc[0] * 2 > of else None,
                     "backed": bool(of >= MIN_FREE and agree * 2 > of),
                     "sources": sorted(d.loc[d["side"] == e.get("side"), "source"].tolist())}
        n_backed += e["free"]["backed"]
    return n_backed


def block(entries: list, latest: dict) -> dict:
    """What the page needs: per-source status and the simulation picks the majority backs (best confidence first)."""
    backed = [e for e in entries if e.get("free", {}).get("backed")]
    return {"generated": latest.get("generated"), "sources": latest.get("sources", {}), "min_free": MIN_FREE,
            "model_edge": MODEL_EDGE, "backed": [{"game": e["game"], "type": e["type"], "pick": e["pick"], "prob": e["prob"], "rank": e.get("rank"),
                                                   **e["free"]} for e in sorted(backed, key=lambda x: -x["prob"])]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["fetch"])
    ap.parse_args()
    import nflreadpy as nfl
    season = int(nfl.get_current_season())
    games = sp.load_games([season])
    now = pd.Timestamp.now(tz="UTC")
    result = fetch_all(games, now)
    save_latest(result)
    added = sp.log_picks(rows_for_log(result["picks"], now))
    graded = sp.grade(games)
    for name, s in result["sources"].items():
        print(f"{name:18s} {'ok ' if s['ok'] else 'FAIL'} {s['picks']:3d} picks  {s['note'] or ''}")
    print(f"{len(result['picks'])} picks read, {added} new in the log, {graded} graded")


if __name__ == "__main__":
    main()
