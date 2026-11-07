"""
social_picks.py
===============
Picks from X (and anything you paste in), graded by us, with the sources that have actually been winning this season weighted up.

    python social_picks.py refresh                 # what the refresh workflow runs: pull new posts from X, log the picks, grade
    python social_picks.py import picks.csv        # picks from anywhere: columns source, posted_at, text  (see below)
    python social_picks.py grade                   # grade whatever has finished
    python social_picks.py report                  # the leaderboard and the "does a good record predict anything" test
    python social_picks.py backfill --since 2026-09-01 --max-posts 1500 --yes     # one-off: this season's history from the timelines

HOW A PICK COUNTS
-----------------
A post is read for game picks: a spread ("Lions -3.5"), a total ("Over 47.5" next to a team or a matchup) or a moneyline ("Lions ML -150").
Player props, parlays, teasers, futures and recaps ("cashed", "last week") are skipped. A pick counts only if
  - it was posted BEFORE kickoff, and no more than MAX_LEAD_HOURS before it (so a Monday recap is never read as next Sunday's pick),
  - the number in the post is close to the market's (a post about another sport or another game does not slip in),
  - it is the source's FIRST pick on that game and market (flipping to the other side later changes nothing).
Every pick is graded at the number in the post, flat 1 unit, -110 unless the post gives a price. A source's record is what WE graded from its
own posts, never the record on its profile.

HOW A SOURCE IS WEIGHTED
------------------------
Win rate on spread and total picks is pulled toward 50% by PRIOR_PICKS pseudo-picks (a 5-1 start is mostly luck), then compared with the 52.4%
that -110 requires. A source with fewer than MIN_PICKS graded picks is "unproven" and carries no weight; one at or below break-even carries none either.
The report also asks the honest question: do picks from sources that looked good at the time win more than the rest? (They usually do not by
much, which is why this only ever re-orders what you read and never moves a projection.)

X ACCESS
--------
Reading posts through X's API is pay-per-use ($0.005 a post read, no free tier). Set the X_BEARER_TOKEN secret and list handles in
social_sources.json (or the X_HANDLES variable). Only posts newer than the last pull are fetched, and a run reads at most `max_posts_per_run`
(default 200, about $1). Without a token nothing is fetched and nothing is spent; the importer still works.

IMPORT FILE
-----------
A CSV with columns  source,posted_at,text  (posted_at ISO, e.g. 2026-11-07T14:05:00Z; with no timezone it is read as US Eastern). Optional: platform.
"""

import argparse
import datetime
import json
import math
import os
import pathlib
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent
LOG = ROOT / "tracking" / "social_picks.csv"
STATE = ROOT / "tracking" / "social_state.json"
CONFIG = ROOT / "social_sources.json"
COLUMNS = ["source", "platform", "post_id", "posted_at", "season", "week", "game_id", "game", "kickoff", "market", "side", "line", "odds",
           "text", "result", "profit", "graded_at"]

BREAKEVEN = 110 / 210              # win rate -110 needs
DEFAULT_ODDS = -110
PRIOR_PICKS = 40                   # pseudo-picks at 50% pulled into every source's record
MIN_PICKS = 10                     # graded spread/total picks before a source carries any weight
FULL_WEIGHT_EDGE = 0.06            # shrunk win rate this far above break-even earns full weight
MAX_LEAD_HOURS = 132               # a pick is read as being about a game kicking off within this long after the post
SPREAD_TOLERANCE = 3.0             # the post's spread may differ from the market's by this much
TOTAL_TOLERANCE = 5.0
X_COST_PER_POST = 0.005
X_API = "https://api.x.com/2"
MAX_QUERY = 440

SKIP_TWEET = re.compile(r"\b(parlay|teaser|sgp|futures?|win total|mvp|cashed|graded|recap|last week|yesterday|results?)\b", re.I)
SKIP_LINE = re.compile(r"\b(yds|yards|receptions?|rushing|passing|receiving|touchdowns?|tds?|anytime|attempts?|completions?|ints|sacks|longest|props?)\b", re.I)

NICKNAMES = {
    "ARI": ["cardinals", "cards", "arizona"], "ATL": ["falcons", "atlanta"], "BAL": ["ravens", "baltimore"], "BUF": ["bills", "buffalo"],
    "CAR": ["panthers", "carolina"], "CHI": ["bears", "chicago"], "CIN": ["bengals", "cincinnati", "cincy"], "CLE": ["browns", "cleveland"],
    "DAL": ["cowboys", "dallas"], "DEN": ["broncos", "denver"], "DET": ["lions", "detroit"], "GB": ["packers", "pack", "green bay"],
    "HOU": ["texans", "houston"], "IND": ["colts", "indianapolis", "indy"], "JAX": ["jaguars", "jags", "jacksonville"],
    "KC": ["chiefs", "kansas city"], "LAC": ["chargers", "bolts"], "LA": ["rams"], "LV": ["raiders", "las vegas"],
    "MIA": ["dolphins", "fins", "miami"], "MIN": ["vikings", "vikes", "minnesota"], "NE": ["patriots", "pats", "new england"],
    "NO": ["saints", "new orleans"], "NYG": ["giants"], "NYJ": ["jets"], "PHI": ["eagles", "philly", "philadelphia"],
    "PIT": ["steelers", "pittsburgh"], "SEA": ["seahawks", "seattle"], "SF": ["49ers", "niners", "san francisco"],
    "TB": ["buccaneers", "bucs", "tampa bay", "tampa"], "TEN": ["titans", "tennessee"], "WAS": ["commanders", "washington"],
}
# written abbreviations; only believed when they sit next to a number or ML, or in a "DET/GB" pair (several are ordinary words in capitals)
ABBREVIATIONS = {c: c for c in NICKNAMES}
ABBREVIATIONS.update({"LAR": "LA", "WSH": "WAS", "JAC": "JAX", "KAN": "KC", "NWE": "NE", "SFO": "SF", "TAM": "TB", "GNB": "GB", "NOR": "NO"})
ABBREVIATIONS.pop("LA")                      # "LA" alone could be the Rams or the Chargers
NAME_TO_CODE = {n: c for c, names in NICKNAMES.items() for n in names}

_NAMES = "|".join(re.escape(n) for n in sorted(NAME_TO_CODE, key=len, reverse=True))
_ABBRS = "|".join(sorted(ABBREVIATIONS, key=len, reverse=True))
TEAM = rf"(?:(?i:{_NAMES})|{_ABBRS})"
_START = r"(?<![\w@])#?"
SPREAD_RE = re.compile(rf"{_START}(?P<team>{TEAM})(?![A-Za-z])\s*[\(\[]?\s*(?P<line>[+-]\s?\d{{1,2}}(?:\.\d)?|(?i:pk|pick'?em))(?![\d/])\s*[\)\]]?")
ML_RE = re.compile(rf"{_START}(?P<team>{TEAM})(?![A-Za-z])\s*[\(\[]?\s*(?i:ml|moneyline|money line)\b\s*[\)\]]?(?:\s*[\(\[]?(?P<odds>[+-]\d{{3,4}})[\)\]]?)?")
TOTAL_RE = re.compile(r"(?<![\w.])(?P<side>over|under|ov|un|o|u)\s?(?P<line>\d{2,3}(?:\.\d)?)(?![\d])", re.I)
ODDS_RE = re.compile(r"^\s*[\(\[]?\s*(?P<odds>[+-]\d{3})\b")
PAIR_RE = re.compile(rf"{_START}(?P<a>{TEAM})(?![A-Za-z])\s*(?:/|@|-|vs\.?|v\.?|at|\|)\s*#?(?P<b>{TEAM})(?![A-Za-z])")


def _code(token: str) -> str:
    t = token.strip().lstrip("#")
    if t in ABBREVIATIONS:
        return ABBREVIATIONS[t]
    return NAME_TO_CODE[t.lower()]


# ---- time ---------------------------------------------------------------------------------------------------------------
def _nth_sunday(year: int, month: int, n: int) -> datetime.date:
    d = datetime.date(year, month, 1)
    d += datetime.timedelta(days=(6 - d.weekday()) % 7 + 7 * (n - 1))
    return d


def et_offset(day: datetime.date) -> int:
    """UTC offset of US Eastern on a date: -4 from the second Sunday of March to the first Sunday of November, else -5."""
    return -4 if _nth_sunday(day.year, 3, 2) <= day < _nth_sunday(day.year, 11, 1) else -5


def to_utc(value) -> pd.Timestamp:
    """Any timestamp as UTC; one with no timezone is US Eastern."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return (ts - pd.Timedelta(hours=et_offset(ts.date()))).tz_localize("UTC")
    return ts.tz_convert("UTC")


# ---- games --------------------------------------------------------------------------------------------------------------
def games_from_schedule(sched: pd.DataFrame) -> pd.DataFrame:
    """nflverse schedule rows -> game_id, season, week, home, away, kickoff (UTC), market lines and final scores."""
    s = sched[sched["game_type"] == "REG"] if "game_type" in sched else sched
    kick = []
    for day, tm in zip(s["gameday"].astype(str), s["gametime"].astype(str)):
        d = datetime.date.fromisoformat(day[:10])
        try:
            local = pd.Timestamp(f"{day[:10]} {tm[:5]}")
        except ValueError:
            local = pd.Timestamp(f"{day[:10]} 13:00")
        kick.append((local - pd.Timedelta(hours=et_offset(d))).tz_localize("UTC"))
    return pd.DataFrame({"game_id": s["game_id"].to_numpy(), "season": s["season"].to_numpy(), "week": s["week"].to_numpy(),
                         "home": s["home_team"].to_numpy(), "away": s["away_team"].to_numpy(), "kickoff": kick,
                         "spread_line": s["spread_line"].to_numpy(float), "total_line": s["total_line"].to_numpy(float),
                         "home_score": s["home_score"].to_numpy(float), "away_score": s["away_score"].to_numpy(float)})


def load_games(seasons) -> pd.DataFrame:
    import nflreadpy as nfl
    return games_from_schedule(nfl.load_schedules(list(seasons)).to_pandas())


def find_game(games: pd.DataFrame, team: str, posted: pd.Timestamp, opponent: str = None):
    """The first game `team` plays after `posted`, within MAX_LEAD_HOURS (and against `opponent` when given); None if there is none."""
    g = games[((games["home"] == team) | (games["away"] == team)) & (games["kickoff"] > posted)
              & (games["kickoff"] - posted <= pd.Timedelta(hours=MAX_LEAD_HOURS))]
    if opponent:
        g = g[(g["home"] == opponent) | (g["away"] == opponent)]
    return None if g.empty else g.sort_values("kickoff").iloc[0]


def team_line(game, team: str) -> float:
    """The market spread from `team`'s side (negative = favorite). nflverse's spread_line is positive when the HOME team is favored."""
    sl = float(game["spread_line"])
    return float("nan") if math.isnan(sl) else (-sl if team == game["home"] else sl)


# ---- parsing ------------------------------------------------------------------------------------------------------------
def _norm(text: str) -> str:
    return re.sub(r"[−–—]", "-", text)


def _parse_line(token: str) -> float:
    t = token.replace(" ", "").lower()
    return 0.0 if t in ("pk", "pickem", "pick'em") else float(t)


def parse_picks(text: str, posted_at, games: pd.DataFrame) -> list:
    """
    Game picks in one post, as dicts (game_id, game, kickoff, week, season, market, side, line, odds). Anything unclear is left out:
    a missed pick costs nothing, a wrong one would pollute a source's record.
    """
    text = _norm(text or "")
    if SKIP_TWEET.search(text):
        return []
    posted = to_utc(posted_at)
    out = []
    for seg in re.split(r"[\n;|•]", text):
        if not seg.strip() or SKIP_LINE.search(seg):
            continue
        taken = []                                                         # spans already read as a spread or ML pick
        for m in ML_RE.finditer(seg):
            g = find_game(games, _code(m["team"]), posted)
            if g is None:
                continue
            odds = int(m["odds"]) if m["odds"] else None
            out.append(_pick(g, "moneyline", _code(m["team"]), None, odds))
            taken.append(m.span())
        for m in SPREAD_RE.finditer(seg):
            if any(a <= m.start() < b for a, b in taken):
                continue
            team, line = _code(m["team"]), _parse_line(m["line"])
            if abs(line) > 25:
                continue
            g = find_game(games, team, posted)
            if g is None:
                continue
            mkt = team_line(g, team)
            if not math.isnan(mkt) and abs(line - mkt) > SPREAD_TOLERANCE:
                continue
            odds = ODDS_RE.match(seg[m.end():])
            out.append(_pick(g, "spread", team, line, int(odds["odds"]) if odds else None))
            taken.append(m.span())
        out += _totals(seg, posted, games, taken)
    return out


def _totals(seg: str, posted, games, taken) -> list:
    """Totals need a game: the matchup in the same line ("DET/GB", "Lions vs Packers") or, failing that, exactly one team."""
    hits = [m for m in TOTAL_RE.finditer(seg) if not any(a <= m.start() < b for a, b in taken)]
    if not hits:
        return []
    pair = PAIR_RE.search(seg)
    teams = []
    if pair:
        teams = [_code(pair["a"]), _code(pair["b"])]
    else:
        teams = sorted({_code(m.group(0)) for m in re.finditer(rf"{_START}(?i:{_NAMES})(?![A-Za-z])", seg)})
    games_hit = {}
    if len(teams) == 2:
        g = find_game(games, teams[0], posted, teams[1])
        if g is not None:
            games_hit[g["game_id"]] = g
    elif len(teams) == 1:
        g = find_game(games, teams[0], posted)
        if g is not None:
            games_hit[g["game_id"]] = g
    if len(games_hit) != 1:
        return []
    g = next(iter(games_hit.values()))
    out = []
    for m in hits:
        line = float(m["line"])
        tl = float(g["total_line"]) if "total_line" in g and not pd.isna(g["total_line"]) else float("nan")
        if not 30 <= line <= 75 or (not math.isnan(tl) and abs(line - tl) > TOTAL_TOLERANCE):
            continue
        side = "over" if m["side"].lower() in ("over", "ov", "o") else "under"
        odds = ODDS_RE.match(seg[m.end():])
        out.append(_pick(g, "total", side, line, int(odds["odds"]) if odds else None))
    return out


def _pick(g, market: str, side: str, line, odds):
    return {"game_id": g["game_id"], "game": f"{g['away']} @ {g['home']}", "season": int(g["season"]), "week": int(g["week"]),
            "kickoff": g["kickoff"].isoformat(), "market": market, "side": side, "line": line, "odds": odds}


# ---- the log ------------------------------------------------------------------------------------------------------------
def load(path=None) -> pd.DataFrame:
    path = pathlib.Path(path or LOG)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path, dtype={"post_id": str})
    for c in COLUMNS:
        if c not in df:
            df[c] = np.nan
    df = df[COLUMNS].copy()
    for c in ("result", "graded_at", "text", "side", "game", "kickoff", "posted_at"):
        df[c] = df[c].astype(object)
    return df


def log_picks(rows: list, path=None) -> int:
    """
    Add picks. A source's first pick on a game and market stands (the earlier post wins, so a later flip or a repost changes nothing) and a
    grade already recorded is kept. -> rows added.
    """
    path = pathlib.Path(path or LOG)
    if not rows:
        return 0
    old = load(path)
    new = pd.DataFrame(rows, columns=COLUMNS)
    both = pd.concat([old, new], ignore_index=True)
    both["_t"] = pd.to_datetime(both["posted_at"], utc=True, format="ISO8601")
    both = both.sort_values(["_t", "post_id"], kind="stable").drop_duplicates(["source", "game_id", "market"], keep="first")
    added = len(both) - len(old)
    path.parent.mkdir(exist_ok=True)
    both.drop(columns="_t").sort_values(["season", "week", "source"]).to_csv(path, index=False)
    return max(added, 0)


def rows_from_posts(posts: list, games: pd.DataFrame, platform: str = "x") -> list:
    """posts: [{source, post_id, posted_at, text}] -> log rows."""
    rows = []
    for p in posts:
        for pick in parse_picks(p["text"], p["posted_at"], games):
            rows.append({"source": str(p["source"]).lstrip("@").lower(), "platform": p.get("platform", platform), "post_id": str(p.get("post_id", "")),
                         "posted_at": to_utc(p["posted_at"]).isoformat(), **pick, "text": re.sub(r"\s+", " ", p["text"])[:280],
                         "result": np.nan, "profit": np.nan, "graded_at": np.nan})
    return rows


def import_csv(path, games: pd.DataFrame, log_path=None) -> dict:
    df = pd.read_csv(path, dtype=str).fillna("")
    for c in ("source", "posted_at", "text"):
        if c not in df:
            raise SystemExit(f"{path} needs a '{c}' column")
    posts = [{"source": r["source"], "posted_at": r["posted_at"], "text": r["text"], "platform": r.get("platform") or "import",
              "post_id": r.get("post_id") or f"import-{i}"} for i, r in df.iterrows()]
    rows = rows_from_posts(posts, games, "import")
    return {"posts": len(posts), "picks_found": len(rows), "added": log_picks(rows, log_path)}


# ---- grading ------------------------------------------------------------------------------------------------------------
def payout(odds) -> float:
    """Profit on a 1-unit win at American `odds` (default -110)."""
    o = DEFAULT_ODDS if odds is None or (isinstance(odds, float) and math.isnan(odds)) else float(odds)
    return o / 100 if o > 0 else 100 / -o


def grade_pick(row, game) -> tuple:
    """-> (result, profit) for a final game; profit is None for a moneyline pick with no price (no honest unit result)."""
    hs, as_ = float(game["home_score"]), float(game["away_score"])
    if math.isnan(hs) or math.isnan(as_):
        return None, None
    odds = None if pd.isna(row["odds"]) else float(row["odds"])
    if row["market"] == "spread":
        margin = (hs - as_) if row["side"] == game["home"] else (as_ - hs)
        diff = margin + float(row["line"])
    elif row["market"] == "total":
        diff = (hs + as_ - float(row["line"])) * (1 if row["side"] == "over" else -1)
    else:
        diff = (hs - as_) * (1 if row["side"] == game["home"] else -1)
        if diff == 0:
            return "push", 0.0
        return ("win", payout(odds) if odds is not None else None) if diff > 0 else ("loss", -1.0 if odds is not None else None)
    if diff == 0:
        return "push", 0.0
    return ("win", payout(odds)) if diff > 0 else ("loss", -1.0)


def grade(games: pd.DataFrame, path=None, now: str = None) -> int:
    """Grade every logged pick whose game is final. -> picks newly graded."""
    path = pathlib.Path(path or LOG)
    df = load(path)
    todo = df[df["result"].isna()]
    if todo.empty:
        return 0
    by = games.set_index("game_id")
    n = 0
    stamp = now or datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    for i, r in todo.iterrows():
        if r["game_id"] not in by.index:
            continue
        res, profit = grade_pick(r, by.loc[r["game_id"]])
        if res is None:
            continue
        df.at[i, "result"], df.at[i, "profit"], df.at[i, "graded_at"] = res, np.nan if profit is None else profit, stamp
        n += 1
    if n:
        df.to_csv(path, index=False)
    return n


# ---- who is any good ----------------------------------------------------------------------------------------------------
def beta_sf(x: float, a: float, b: float) -> float:
    """P(rate > x) for a Beta(a, b) posterior, by numeric integration (no scipy)."""
    grid = np.linspace(1e-6, 1 - 1e-6, 4001)
    logpdf = (a - 1) * np.log(grid) + (b - 1) * np.log1p(-grid)
    pdf = np.exp(logpdf - logpdf.max())
    return float(pdf[grid > x].sum() / pdf.sum())


def skill(wins: int, losses: int) -> dict:
    """Shrunk win rate, the chance the source's true rate beats -110, and the weight that earns (0 below MIN_PICKS or break-even)."""
    n = wins + losses
    a, b = wins + PRIOR_PICKS / 2, losses + PRIOR_PICKS / 2
    shrunk = a / (a + b)
    weight = 0.0 if n < MIN_PICKS else float(np.clip((shrunk - BREAKEVEN) / FULL_WEIGHT_EDGE, 0, 1))
    return {"n": n, "shrunk": shrunk, "p_beats": beta_sf(BREAKEVEN, a, b), "weight": weight, "status": "tracked" if n >= MIN_PICKS else "unproven"}


def source_table(df: pd.DataFrame, season: int) -> list:
    """One row per source for a season: spread/total record (what the weight rests on), units, moneyline separately. Best weight first."""
    d = df[(df["season"] == season)]
    rows = []
    for src, g in d.groupby("source"):
        scored = g[g["market"].isin(["spread", "total"]) & g["result"].notna()]
        w, l, p = (int((scored["result"] == r).sum()) for r in ("win", "loss", "push"))
        ml = g[(g["market"] == "moneyline") & g["profit"].notna()]
        row = {"source": src, "wins": w, "losses": l, "pushes": p, "picks": int(len(g)), "pending": int(g["result"].isna().sum()),
               "win_pct": w / (w + l) if w + l else None, "units": float(scored["profit"].sum()),
               "roi": float(scored["profit"].sum() / (w + l + p)) if w + l + p else None,
               "ml_record": f"{int((ml['result'] == 'win').sum())}-{int((ml['result'] == 'loss').sum())}", "ml_units": float(ml["profit"].sum()),
               **skill(w, l)}
        rows.append(row)
    return sorted(rows, key=lambda r: (-r["weight"], -r["p_beats"], r["source"]))


def weights(df: pd.DataFrame, season: int) -> dict:
    return {r["source"]: r["weight"] for r in source_table(df, season)}


def consensus(df: pd.DataFrame, season: int, now=None) -> list:
    """
    Picks on games still to be played, grouped by game and market, each side scored by the summed weight of the proven sources behind it.
    -> [{game, kickoff, items: [{market, side, line, score, backers: [...]}]}] with the most-backed game first.
    """
    now = to_utc(now) if now is not None else pd.Timestamp.now(tz="UTC")
    tab = {r["source"]: r for r in source_table(df, season)}
    d = df[(df["season"] == season) & df["result"].isna()].copy()
    d = d[pd.to_datetime(d["kickoff"], utc=True, format="ISO8601") > now]
    out = []
    for gid, g in d.groupby("game_id"):
        items = []
        for (market, side), s in g.groupby(["market", "side"]):
            backers = [{"source": r.source, "line": None if pd.isna(r.line) else float(r.line),
                        "odds": None if pd.isna(r.odds) else int(r.odds), "weight": tab[r.source]["weight"], "status": tab[r.source]["status"],
                        "record": f"{tab[r.source]['wins']}-{tab[r.source]['losses']}"} for r in s.itertuples()]
            backers.sort(key=lambda b: -b["weight"])
            items.append({"market": market, "side": side, "line": next((b["line"] for b in backers if b["line"] is not None), None),
                          "score": round(sum(b["weight"] for b in backers), 3), "n": len(backers), "backers": backers})
        items.sort(key=lambda i: (-i["score"], -i["n"]))
        out.append({"game_id": gid, "game": g["game"].iloc[0], "kickoff": g["kickoff"].iloc[0], "week": int(g["week"].iloc[0]), "items": items,
                    "score": max(i["score"] for i in items)})
    return sorted(out, key=lambda x: (-x["score"], x["kickoff"]))


def skill_test(df: pd.DataFrame, season: int) -> dict:
    """
    Does a good record mean anything? Each graded spread/total pick is sorted by whether its source, AT THE TIME OF THE POST (only picks whose games
    had finished by then), had MIN_PICKS+ graded picks and a shrunk win rate above break-even. -> win rates of the two groups and a z-score.
    """
    d = df[(df["season"] == season) & df["market"].isin(["spread", "total"]) & df["result"].isin(["win", "loss"])].copy()
    if d.empty:
        return {"n": 0}
    d["posted"] = pd.to_datetime(d["posted_at"], utc=True, format="ISO8601")
    d["kick"] = pd.to_datetime(d["kickoff"], utc=True, format="ISO8601")
    hot, rest = [], []
    for src, g in d.groupby("source"):
        for r in g.itertuples():
            prior = g[(g["kick"] + pd.Timedelta(hours=4)) <= r.posted]            # games that were over (and so graded) by the time of the post
            w, l = int((prior["result"] == "win").sum()), int((prior["result"] == "loss").sum())
            s = skill(w, l)
            (hot if s["n"] >= MIN_PICKS and s["shrunk"] > BREAKEVEN else rest).append(1.0 if r.result == "win" else 0.0)
    out = {"n": len(hot) + len(rest), "hot_n": len(hot), "rest_n": len(rest)}
    out["hot_win"] = float(np.mean(hot)) if hot else None
    out["rest_win"] = float(np.mean(rest)) if rest else None
    if len(hot) >= 20 and len(rest) >= 20:
        p = (sum(hot) + sum(rest)) / (len(hot) + len(rest))
        se = math.sqrt(p * (1 - p) * (1 / len(hot) + 1 / len(rest)))
        out["z"] = (out["hot_win"] - out["rest_win"]) / se if se else 0.0
    return out


# ---- the page and the report -------------------------------------------------------------------------------------------
def load_state(path=None) -> dict:
    path = pathlib.Path(path or STATE)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"since": {}, "posts_read": 0, "est_cost_usd": 0.0, "last_pull": None, "last_note": None}


def save_state(state: dict, path=None):
    path = pathlib.Path(path or STATE)
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(state, indent=1), encoding="utf-8")


def load_config(path=None) -> dict:
    path = pathlib.Path(path or CONFIG)
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cfg = {}
    handles = [h.lstrip("@").lower() for h in cfg.get("x_handles", [])]
    handles += [h.strip().lstrip("@").lower() for h in os.environ.get("X_HANDLES", "").split(",") if h.strip()]
    return {"x_handles": list(dict.fromkeys(handles)), "max_posts_per_run": int(cfg.get("max_posts_per_run", 200))}


def ensure_files(log_path=None, state_path=None, state: dict = None):
    log_path, state_path = pathlib.Path(log_path or LOG), pathlib.Path(state_path or STATE)
    if not log_path.exists():
        log_path.parent.mkdir(exist_ok=True)
        pd.DataFrame(columns=COLUMNS).to_csv(log_path, index=False)
    if not state_path.exists():
        save_state(state or load_state(state_path), state_path)


def build_block(season: int, now=None, df: pd.DataFrame = None, config: dict = None, state: dict = None) -> dict:
    """What the props page shows on its Social Picks tab."""
    df = load() if df is None else df
    config = load_config() if config is None else config
    state = load_state() if state is None else state
    return {"season": season, "handles": len(config["x_handles"]), "max_posts": config["max_posts_per_run"], "min_picks": MIN_PICKS, "breakeven": round(BREAKEVEN, 4),
            "prior_picks": PRIOR_PICKS, "sources": source_table(df, season), "games": consensus(df, season, now),
            "test": skill_test(df, season), "x": {"token": bool(os.environ.get("X_BEARER_TOKEN")), "posts_read": state.get("posts_read", 0),
                                                  "est_cost_usd": round(state.get("est_cost_usd", 0.0), 2), "last_pull": state.get("last_pull"),
                                                  "note": state.get("last_note")}}


def section(season: int, week: int, df: pd.DataFrame = None):
    """The weekly report section, or None when nothing has been tracked."""
    df = load() if df is None else df
    d = df[df["season"] == season]
    if d.empty:
        return None
    wk = d[(d["week"] == week) & d["market"].isin(["spread", "total"]) & d["result"].isin(["win", "loss", "push"])]
    tab = source_table(df, season)
    test = skill_test(df, season)
    return {"week": int(week), "tab": tab[:8], "writeup": blurb(wk, tab, test, week)}


def blurb(wk: pd.DataFrame, tab: list, test: dict, week: int) -> str:
    parts = []
    if len(wk):
        w, l = int((wk["result"] == "win").sum()), int((wk["result"] == "loss").sum())
        parts.append(f"Week {week}: {len(wk)} tracked spread and total picks from {wk['source'].nunique()} sources went {w}-{l}"
                     + (f" ({w / (w + l):.0%}; -110 needs {BREAKEVEN:.1%})" if w + l else "") + ".")
    tracked = [r for r in tab if r["status"] == "tracked"]
    if tracked:
        best = tracked[0]
        parts.append(f"{len(tracked)} sources have {MIN_PICKS}+ graded picks this season; the best is @{best['source']} at {best['wins']}-{best['losses']}"
                     f" (shrunk to {best['shrunk']:.1%}, {best['p_beats']:.0%} chance its true rate beats the vig).")
    else:
        parts.append(f"No source has {MIN_PICKS} graded picks yet, so none carries any weight.")
    if test.get("hot_n") and test.get("rest_n"):
        z = test.get("z")
        verdict = "" if z is None else (" That gap is " + ("distinguishable from luck." if abs(z) >= 2 else "within luck."))
        parts.append(f"Picks from sources that looked good at the time won {test['hot_win']:.0%} ({test['hot_n']} picks) against {test['rest_win']:.0%} "
                     f"({test['rest_n']}) for the rest.{verdict}")
    parts.append("Records are graded by us from each source's own posts, flat 1 unit at the stated price (-110 if none).")
    return " ".join(parts)


# ---- X ------------------------------------------------------------------------------------------------------------------
def _http_get(url: str, params: dict, token: str):
    req = urllib.request.Request(url + ("?" + urllib.parse.urlencode(params) if params else ""),
                                 headers={"Authorization": f"Bearer {token}", "User-Agent": "ff-model-social-picks"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode("utf-8"))
        except (ValueError, OSError):
            body = {}
        return e.code, body


def chunk_handles(handles: list) -> list:
    """Handles grouped so each search query stays under X's length limit."""
    chunks, cur = [], []
    for h in handles:
        if cur and len(" OR ".join(f"from:{x}" for x in cur + [h])) > MAX_QUERY:
            chunks.append(cur)
            cur = []
        cur.append(h)
    return chunks + ([cur] if cur else [])


def _posts_from(body: dict) -> list:
    users = {u["id"]: u["username"] for u in (body.get("includes") or {}).get("users", [])}
    return [{"source": users.get(t.get("author_id"), t.get("author_id")), "post_id": t["id"], "posted_at": t["created_at"], "text": t["text"],
             "platform": "x"} for t in body.get("data", [])]


def fetch_posts(handles: list, token: str, state: dict, max_posts: int, get=_http_get) -> tuple:
    """
    New posts from `handles` since the last pull (X's recent search, last 7 days), at most `max_posts` read. Updates `state` (per-query since_id,
    posts read, estimated cost). -> (posts, note) where note says why a pull stopped early, or None.
    """
    posts, note, read = [], None, 0
    for chunk in chunk_handles(handles):
        if read >= max_posts:
            note = f"stopped at the {max_posts}-post cap; some handles were not read this run"
            break
        key = ",".join(chunk)
        params = {"query": "(" + " OR ".join(f"from:{h}" for h in chunk) + ") -is:retweet -is:reply",
                  "tweet.fields": "created_at,author_id", "expansions": "author_id", "user.fields": "username"}
        if state["since"].get(key):
            params["since_id"] = state["since"][key]
        newest = int(state["since"].get(key) or 0)
        while True:
            params["max_results"] = max(10, min(100, max_posts - read))          # X's minimum page is 10
            status, body = get(f"{X_API}/tweets/search/recent", params, token)
            if status != 200:
                note = f"X API returned {status}" + (f": {body.get('title') or body.get('detail')}" if body else "")
                break
            got = _posts_from(body)
            read += len(got)
            posts += got
            newest = max([newest] + [int(p["post_id"]) for p in got])
            nxt = (body.get("meta") or {}).get("next_token")
            if not nxt or not got:
                break
            if read >= max_posts:
                note = f"stopped at the {max_posts}-post cap; older posts in the window were skipped"
                break
            params["next_token"] = nxt
        if newest:
            state["since"][key] = str(newest)
        if note:
            break
    state["posts_read"] = state.get("posts_read", 0) + read
    state["est_cost_usd"] = round(state.get("est_cost_usd", 0.0) + read * X_COST_PER_POST, 4)
    return posts, note


def fetch_timelines(handles: list, token: str, since: str, max_posts: int, get=_http_get) -> tuple:
    """One-off backfill: each handle's own posts since a date, through the timeline endpoint (newest first, up to ~3,200 per account)."""
    status, body = get(f"{X_API}/users/by", {"usernames": ",".join(handles)}, token)
    if status != 200:
        return [], f"X API returned {status} looking up users"
    posts = []
    for u in body.get("data", []):
        token_next = None
        while len(posts) < max_posts:
            params = {"max_results": max(5, min(100, max_posts - len(posts))), "exclude": "retweets,replies", "tweet.fields": "created_at,author_id",
                      "start_time": f"{since}T00:00:00Z"}
            if token_next:
                params["pagination_token"] = token_next
            status, page = get(f"{X_API}/users/{u['id']}/tweets", params, token)
            if status != 200:
                return posts, f"X API returned {status} reading @{u['username']}"
            for t in page.get("data", []):
                posts.append({"source": u["username"], "post_id": t["id"], "posted_at": t["created_at"], "text": t["text"], "platform": "x"})
            token_next = (page.get("meta") or {}).get("next_token")
            if not token_next:
                break
    return posts[:max_posts], None


# ---- command line -------------------------------------------------------------------------------------------------------
def refresh(games: pd.DataFrame = None, get=_http_get, log_path=None, state_path=None, config: dict = None) -> dict:
    """Pull new X posts (when there is a token and handles), log their picks, grade what has finished."""
    config = config or load_config()
    state = load_state(state_path)
    season = int(games["season"].max()) if games is not None else None
    if games is None:
        import nflreadpy as nfl
        season = int(nfl.get_current_season())
        games = load_games([season])
    token = os.environ.get("X_BEARER_TOKEN")
    summary = {"posts": 0, "picks": 0, "added": 0, "note": None}
    if token and config["x_handles"]:
        posts, note = fetch_posts(config["x_handles"], token, state, config["max_posts_per_run"], get)
        rows = rows_from_posts(posts, games)
        summary.update(posts=len(posts), picks=len(rows), added=log_picks(rows, log_path), note=note)
        state["last_pull"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        state["last_note"] = note
        save_state(state, state_path)
    elif not token:
        summary["note"] = "X_BEARER_TOKEN not set; nothing pulled"
    else:
        summary["note"] = "no handles listed (social_sources.json or X_HANDLES); nothing pulled"
    summary["graded"] = grade(games, log_path)
    ensure_files(log_path, state_path, state)                  # the workflow commits both files, so they exist from the first run
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["refresh", "import", "grade", "report", "backfill"])
    ap.add_argument("file", nargs="?")
    ap.add_argument("--since", help="backfill: earliest date, YYYY-MM-DD")
    ap.add_argument("--max-posts", type=int, default=1000)
    ap.add_argument("--yes", action="store_true", help="backfill: confirm the spend")
    args = ap.parse_args()
    import nflreadpy as nfl
    season = int(nfl.get_current_season())
    games = load_games([season])

    if args.command == "refresh":
        print(refresh(games))
    elif args.command == "grade":
        print(f"graded {grade(games)} picks")
    elif args.command == "import":
        if not args.file:
            raise SystemExit("usage: social_picks.py import picks.csv")
        print(import_csv(args.file, games))
        print(f"graded {grade(games)} picks")
    elif args.command == "backfill":
        token, cfg = os.environ.get("X_BEARER_TOKEN"), load_config()
        if not token or not cfg["x_handles"] or not args.since:
            raise SystemExit("backfill needs X_BEARER_TOKEN, handles in social_sources.json / X_HANDLES, and --since YYYY-MM-DD")
        cost = args.max_posts * X_COST_PER_POST + 0.01 * len(cfg["x_handles"])
        print(f"Will read up to {args.max_posts} posts from {len(cfg['x_handles'])} accounts: up to about ${cost:.2f} of X API credit.")
        if not args.yes:
            raise SystemExit("Re-run with --yes to spend it.")
        posts, note = fetch_timelines(cfg["x_handles"], token, args.since, args.max_posts)
        rows = rows_from_posts(posts, games)
        print({"posts": len(posts), "picks_found": len(rows), "added": log_picks(rows), "note": note})
        state = load_state()
        state["posts_read"] = state.get("posts_read", 0) + len(posts)
        state["est_cost_usd"] = round(state.get("est_cost_usd", 0.0) + len(posts) * X_COST_PER_POST, 4)
        save_state(state)
        print(f"graded {grade(games)} picks")
    else:
        df = load()
        for r in source_table(df, season):
            rec = f"{r['wins']}-{r['losses']}" + (f"-{r['pushes']}" if r["pushes"] else "")
            print(f"{r['source']:22s} {rec:9s} units {r['units']:+6.1f}  shrunk {r['shrunk']:.1%}  P(beats vig) {r['p_beats']:.0%}  "
                  f"weight {r['weight']:.2f}  {r['status']}")
        print(blurb(df[df["season"] == season].head(0), source_table(df, season), skill_test(df, season), 0))


if __name__ == "__main__":
    main()
