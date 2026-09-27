"""
odds_api.py
===========
Thin client for The Odds API (https://the-odds-api.com) player-prop markets,
plus the parsing that turns its response into one row per player prop with a
market consensus probability and the best available price.

KEY HANDLING
------------
The key comes from the ODDS_API_KEY environment variable, or a line
`ODDS_API_KEY=...` in a .env file next to this script (git-ignored). It is
never printed, logged, or written to the cache.

CREDIT COST
-----------
Listing events is free. Each event's props call costs
(number of markets) x (number of regions) credits. Responses are cached on
disk, so re-running within CACHE_MINUTES spends nothing.
"""

import calendar
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from simulate import american_to_prob

BASE = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"
CACHE_DIR = pathlib.Path(__file__).parent / "web" / "cache"
CACHE_MINUTES = 30

# The Odds API uses full team names; nflverse uses abbreviations.
TEAM_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}


# Pick'em / DFS operators (DraftKings Pick6, PrizePicks, Underdog, Dabble...) post one line with
# fixed-payout entries, not two-way prices. They are kept OUT of the sportsbook consensus (their
# "prices" aren't odds) and reported separately as extra lines to compare against.
DFS_MARKERS = ("underdog", "prizepicks", "pick6", "dabble")


def is_dfs(book_key, book_title=""):
    name = f"{book_key} {book_title}".lower()
    return any(m in name for m in DFS_MARKERS)


class OddsApiError(RuntimeError):
    pass


# ---- key + HTTP ---------------------------------------------------------
def load_key():
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if key:
        return key
    env_file = pathlib.Path(__file__).parent / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "ODDS_API_KEY" and value.strip():
                return value.strip().strip("'\"")
    raise OddsApiError(
        "No Odds API key found. Set the ODDS_API_KEY environment variable, or put\n"
        "  ODDS_API_KEY=your_key\n"
        "in a file named .env next to odds_api.py (it is git-ignored)."
    )


def _get(path, params, key):
    """GET -> (json, response headers). The key is added here and never surfaced in errors."""
    query = urllib.parse.urlencode({**params, "apiKey": key})
    req = urllib.request.Request(f"{BASE}{path}?{query}", headers={"User-Agent": "ff-model"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.load(resp), resp.headers
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:300]
        hint = {
            401: "the key was rejected -- check it is copied correctly",
            403: "this key's plan may not include that market",
            422: "a requested market isn't available for this sport/plan",
            429: "rate limited or out of credits",
        }.get(e.code, "")
        raise OddsApiError(f"Odds API HTTP {e.code} on {path}: {hint} {body}".strip()) from None
    except urllib.error.URLError as e:
        raise OddsApiError(f"Could not reach the Odds API: {e.reason}") from None


def credits_line(headers):
    left, used = headers.get("x-requests-remaining"), headers.get("x-requests-used")
    return f"credits remaining: {left} (used this period: {used})"


# ---- fetching (with cache) ---------------------------------------------
def _cached(name, fetch):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f = CACHE_DIR / f"{name}.json"
    if f.exists() and time.time() - f.stat().st_mtime < CACHE_MINUTES * 60:
        return json.loads(f.read_text(encoding="utf-8")), True
    data = fetch()
    f.write_text(json.dumps(data), encoding="utf-8")
    return data, False


def fetch_events(key, days_ahead=7):
    """Upcoming/live games starting within `days_ahead` days. Free to call."""
    events, headers = _get(f"/sports/{SPORT}/events", {}, key)
    cutoff = time.time() + days_ahead * 86400
    keep = [e for e in events
            if calendar.timegm(time.strptime(e["commence_time"], "%Y-%m-%dT%H:%M:%SZ")) < cutoff]
    return keep, headers


def fetch_event_props(key, event_id, markets, regions="us", bookmakers=None):
    """
    -> (event odds json, from_cache).
    Cost = markets that actually return data x regions, and every group of up to 10
    `bookmakers` counts as one region (bookmakers, when given, overrides regions).
    """
    where = "bk-" + "-".join(sorted(bookmakers)) if bookmakers else regions
    tag = f"{event_id}_{where}_{'-'.join(sorted(m.replace('player_', '') for m in markets))}"

    def fetch():
        params = {"markets": ",".join(markets), "oddsFormat": "american"}
        params.update({"bookmakers": ",".join(bookmakers)} if bookmakers else {"regions": regions})
        data, headers = _get(f"/sports/{SPORT}/events/{event_id}/odds", params, key)
        print(f"  {credits_line(headers)}", file=sys.stderr)
        return data

    return _cached(tag, fetch)


# ---- parsing ------------------------------------------------------------
def american_to_decimal(odds):
    return 1 + (odds / 100 if odds > 0 else 100 / -odds)


def flatten(event_odds):
    """One row per (book, market, player, side)."""
    rows = []
    for bk in event_odds.get("bookmakers", []):
        for mk in bk.get("markets", []):
            for o in mk.get("outcomes", []):
                if not o.get("description"):
                    continue
                rows.append({
                    "book": bk["title"], "book_key": bk.get("key", bk["title"]).lower(),
                    "market": mk["key"], "player": o["description"],
                    "side": o["name"].lower(), "point": o.get("point"), "price": o["price"],
                })
    return rows


def consensus(rows):
    """
    Collapse flattened rows to one entry per (player, market) at its *main*
    line -- the line offered by the most books (ties: the one closest to a
    coin flip). Returns {(player, market): {...}}.
    """
    by_pm, dfs_by_pm = {}, {}
    for r in rows:
        dest = dfs_by_pm if is_dfs(r["book_key"], r["book"]) else by_pm
        dest.setdefault((r["player"], r["market"]), []).append(r)

    out = {}
    for (player, market), rs in by_pm.items():
        points = {}
        for r in rs:
            points.setdefault(r["point"], []).append(r)

        def closeness(pt):
            overs = [american_to_prob(r["price"]) for r in points[pt] if r["side"] in ("over", "yes")]
            return abs((sum(overs) / len(overs) if overs else 0.5) - 0.5)

        point = sorted(points, key=lambda p: (-len({r["book"] for r in points[p]}), closeness(p)))[0]
        line_rows = points[point]

        books = {}
        for r in line_rows:
            books.setdefault(r["book"], {})[r["side"]] = r["price"]

        over_key = "over" if any("over" in b for b in books.values()) else "yes"
        fair = []
        for b in books.values():
            if over_key == "over" and "over" in b and "under" in b:
                po, pu = american_to_prob(b["over"]), american_to_prob(b["under"])
                fair.append(po / (po + pu))         # per-book vig removal
        over_prices = [(american_to_decimal(b[over_key]), b[over_key], name)
                       for name, b in books.items() if over_key in b]
        under_prices = [(american_to_decimal(b["under"]), b["under"], name)
                        for name, b in books.items() if "under" in b]
        best_over = max(over_prices) if over_prices else None
        best_under = max(under_prices) if under_prices else None

        out[(player, market)] = {
            "point": 0.5 if point is None else float(point),
            "fair_over": round(sum(fair) / len(fair), 4) if fair else None,
            "n_books": len(books),
            "best_over": {"price": best_over[1], "book": best_over[2]} if best_over else None,
            "best_under": {"price": best_under[1], "book": best_under[2]} if best_under else None,
            "books": [{"book": n, "over": b.get(over_key), "under": b.get("under")}
                      for n, b in sorted(books.items())],
            "dfs": _dfs_lines(dfs_by_pm.get((player, market), [])),
        }
    return out


def _dfs_lines(rows):
    """[{book, line}] -- one posted line per pick'em book (the prices are ignored on purpose)."""
    lines = {}
    for r in rows:
        if r["point"] is not None:
            lines.setdefault(r["book"], float(r["point"]))
    return [{"book": b, "line": pt} for b, pt in sorted(lines.items())]
