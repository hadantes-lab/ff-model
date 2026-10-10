"""
espn_lines.py
=============
A FREE source of past player-prop lines: ESPN's public (undocumented) odds API. For a finished game it
still serves the sportsbook that ESPN displayed -- ESPN BET for 2024-25, DraftKings from this season --
with each player prop's OPENING and CURRENT line and the over/under prices. That is the same thing the
paid Odds API historical endpoints sell (backfill_history.py), for one book instead of a consensus.

WHAT IT IS, AND IS NOT
----------------------
* One book, not a median across books, so its line can differ from the consensus by a half point.
* "Current" on a finished game is the last line before the market locked at kickoff -- we treat it as the
  closing line, and `validate` checks that against the lines we logged ourselves before games started.
* Unofficial: ESPN doesn't document or promise these endpoints, so this collector fails soft (a game
  with no data is skipped) and is meant for personal modeling. It keeps no raw responses in the repo.
* Alternate-line ladders ("to get 60+ yards") have no prices and are ignored; only the priced Over/Under
  pair is the main line.

HOW
---
Each game's ESPN event id comes from the nflverse schedule; ESPN athlete ids map to nflverse player ids through
nflreadpy's player-id table, so no per-player lookups. Results go to `tracking/espn_lines.csv` (separate from
line_history.csv so the single-book provenance stays explicit), and the hit-rate chart's "Line then" uses them
for any game our own pulls don't cover.

    python espn_lines.py collect --seasons 2024 2025      # fetch past games (resumable)
    python espn_lines.py validate                         # compare with the lines we logged ourselves
    python espn_lines.py summary
"""

import argparse
import csv
import hashlib
import json
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request

import pandas as pd

CORE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"
PROVIDERS = ["58", "100", "40"]                       # ESPN BET, DraftKings (ESPN's partner since late 2025), DraftKings legacy
OUT_FILE = pathlib.Path(__file__).parent / "tracking" / "espn_lines.csv"
STATE_FILE = pathlib.Path(__file__).parent / "tracking" / "espn_state.json"
CACHE_DIR = pathlib.Path(__file__).parent / "web" / "cache" / "espn"          # git-ignored
COLUMNS = ["season", "week", "game", "commence", "espn_event", "provider", "player_id", "player", "team", "opp", "market",
           "open_line", "close_line", "open_over", "open_under", "close_over", "close_under", "last_updated", "priced"]
KEY = ["season", "week", "player_id", "market"]

# ESPN prop type name (before "(incl. overtime)") -> our market key
TYPE_TO_MARKET = {
    "Total Passing Yards": "player_pass_yds", "Total Passing Touchdowns": "player_pass_tds",
    "Total Passing Attempts": "player_pass_attempts", "Total Pass Completions": "player_pass_completions",
    "Total Passing Interceptions": "player_pass_interceptions", "Total Rushing Yards": "player_rush_yds",
    "Total Carries": "player_rush_attempts", "Total Receptions": "player_receptions",
    "Total Receiving Yards": "player_reception_yds", "Total Rushing Plus Receiving Yards": "player_rush_reception_yds",
    "Longest Reception": "player_reception_longest", "Longest Rush": "player_rush_longest",
}


def log(msg):
    print(msg, file=sys.stderr)


# ---- HTTP (polite, cached, fails soft) ---------------------------------------------
def http_get(url: str, cache=True, retries=2, pause=0.12):
    """-> parsed JSON or None. Responses are cached on disk (git-ignored) so reruns cost nothing."""
    url = url.replace("http://", "https://")
    path = CACHE_DIR / (hashlib.sha1(url.encode()).hexdigest() + ".json")
    if cache and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (personal modeling project)"})
            with urllib.request.urlopen(req, timeout=25) as r:
                data = json.load(r)
            if cache:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data), encoding="utf-8")
            time.sleep(pause)
            return data
        except urllib.error.HTTPError as e:
            if e.code == 404:                      # a definitive "no such thing": don't wait and retry
                return None
            time.sleep(1.0 * (attempt + 1))
        except Exception:
            time.sleep(1.0 * (attempt + 1))
    return None


def fetch_prop_items(event_id: str, provider: str, get=http_get) -> list:
    base = f"{CORE}/events/{event_id}/competitions/{event_id}/odds/{provider}/propBets"
    first = get(f"{base}?limit=1000&page=1")
    if not first or not first.get("count"):
        return []
    items = list(first["items"])
    for pg in range(2, int(first.get("pageCount", 1)) + 1):
        page = get(f"{base}?limit=1000&page={pg}")
        items += page["items"] if page else []
    return items


def pick_provider(event_id: str, get=http_get):
    """The first provider in PROVIDERS that has props for this event -> (provider, items)."""
    for prov in PROVIDERS:
        items = fetch_prop_items(event_id, prov, get)
        if items:
            return prov, items
    return None, []


# ---- parsing -----------------------------------------------------------------------
def athlete_id(item) -> str:
    m = re.search(r"athletes/(\d+)", (item.get("athlete") or {}).get("$ref", ""))
    return m.group(1) if m else None


def market_of(item):
    name = re.sub(r"\s*\(incl\. overtime\)\s*", "", item["type"]["name"]).strip()
    return TYPE_TO_MARKET.get(name)


def _american(obj):
    if not obj:
        return None
    a = obj.get("american")
    try:
        return float(str(a).replace("+", ""))
    except (TypeError, ValueError):
        return None


def main_lines(items: list) -> dict:
    """
    {(espn athlete id, market): {open_line, close_line, open_over, open_under, close_over, close_under,
    last_updated, priced}} -- one main line per player prop. ESPN serves two shapes:
      * ESPN BET (2024-25): an unpriced ladder of alternate lines ("to get 60+ yards") PLUS a priced Over entry and
        Under entry at the main line -> the priced pair is the main line, with its prices (priced = True);
      * DraftKings (this season): the same single line listed once or twice per player prop, no prices (priced = False).
    A prop with several DIFFERENT unpriced lines and no priced pair is an ambiguous ladder and is skipped.
    """
    by = {}
    for it in items:
        mk, aid = market_of(it), athlete_id(it)
        cur = it.get("current") or {}
        if mk is None or aid is None or "target" not in cur:
            continue
        by.setdefault((aid, mk), []).append(it)
    out = {}
    for key, entries in by.items():
        overs = [it for it in entries if "over" in it["current"]]
        unders = [it for it in entries if "under" in it["current"]]
        priced = bool(overs or unders)
        if priced:
            # the main line is the one both sides are priced at; if several, the one nearest a coin flip
            lines = {it["current"]["target"]["value"] for it in overs} & {it["current"]["target"]["value"] for it in unders}
            pool = lines or {it["current"]["target"]["value"] for it in overs + unders}

            def balance(line):
                o = [it for it in overs if it["current"]["target"]["value"] == line]
                return abs((_american(o[0]["current"].get("over")) or -110) + 110) if o else 999
            line = sorted(pool, key=lambda l: (balance(l), l))[0]
            o = next((it for it in overs if it["current"]["target"]["value"] == line), None)
            u = next((it for it in unders if it["current"]["target"]["value"] == line), None)
            ref = o or u
        else:
            if len({it["current"]["target"]["value"] for it in entries}) != 1:
                continue                                           # an unpriced ladder: can't tell which rung is the main line
            ref, o, u = entries[0], None, None                     # (DraftKings lists the one line twice, over and under)
            line = ref["current"]["target"]["value"]
        out[key] = {
            "open_line": ref["open"]["target"]["value"] if "target" in (ref.get("open") or {}) else None,
            "close_line": float(line),
            "open_over": _american(((o or {}).get("open") or {}).get("over")),
            "open_under": _american(((u or {}).get("open") or {}).get("under")),
            "close_over": _american(((o or {}).get("current") or {}).get("over")),
            "close_under": _american(((u or {}).get("current") or {}).get("under")),
            "last_updated": ref.get("lastUpdated"), "priced": priced,
        }
    return out


# ---- rows --------------------------------------------------------------------------
def espn_to_gsis(ids: pd.DataFrame) -> dict:
    """ESPN athlete id (str) -> nflverse gsis id, from nflreadpy's player-id table."""
    d = ids.dropna(subset=["espn_id", "gsis_id"])
    return {str(int(float(e))): g for e, g in zip(d["espn_id"], d["gsis_id"])}


def rows_for_game(game: dict, provider: str, lines: dict, mapping: dict, week_rows: pd.DataFrame) -> list:
    """
    One row per (player, market). `week_rows` is the player-stats rows for this season/week: they supply the
    player's name and team, and a player with no row that week (inactive) is skipped.
    """
    info = week_rows.drop_duplicates("player_id").set_index("player_id")
    rows = []
    for (aid, market), v in lines.items():
        pid = mapping.get(aid)
        if pid is None or pid not in info.index:
            continue
        team = info.loc[pid, "team"]
        rows.append({"season": game["season"], "week": game["week"], "game": f"{game['away']} @ {game['home']}",
                     "commence": game["commence"], "espn_event": game["espn"], "provider": provider, "player_id": pid,
                     "player": info.loc[pid, "player_display_name"], "team": team,
                     "opp": game["away"] if team == game["home"] else game["home"], "market": market, **v})
    return rows


def append_rows(rows: list, path=None) -> int:
    path = pathlib.Path(path or OUT_FILE)
    if not rows:
        return 0
    path.parent.mkdir(exist_ok=True)
    fresh = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if fresh:
            w.writeheader()
        w.writerows(rows)
    return len(rows)


def dedupe(path=None) -> int:
    """Drop repeated (season, week, player, market) rows, keeping the last. -> rows removed. (A run stopped between
    writing a game's rows and recording it as done would leave that game's rows twice.)"""
    path = pathlib.Path(path or OUT_FILE)
    df = load(path)
    out = df.drop_duplicates(subset=KEY, keep="last")
    if len(out) != len(df):
        out.to_csv(path, index=False)
    return len(df) - len(out)


def load(path=None) -> pd.DataFrame:
    path = pathlib.Path(path or OUT_FILE)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path)


def closing_map(df: pd.DataFrame = None) -> dict:
    """{(player_id, market): {(season, week): closing line}} -- for the chart's "Line then" fallback."""
    df = load() if df is None else df
    out = {}
    for r in df.itertuples(index=False):
        if r.close_line == r.close_line:
            out.setdefault((r.player_id, r.market), {})[(int(r.season), int(r.week))] = float(r.close_line)
    return out


# ---- driver ------------------------------------------------------------------------
def collect(seasons, sched=None, ids=None, history_loader=None, get=http_get, out_path=None, state_path=None,
            limit_games=None) -> dict:
    """Fetch every finished game in `seasons` not already done. Resumable; fails soft per game."""
    import nflreadpy as nfl
    from props_model import build_history

    sched = sched if sched is not None else nfl.load_schedules(list(seasons)).to_pandas()
    mapping = espn_to_gsis(ids if ids is not None else nfl.load_ff_playerids().to_pandas())
    sp = pathlib.Path(state_path or STATE_FILE)
    state = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {"done": []}
    summary = {"games": 0, "rows": 0, "no_data": 0, "skipped_done": 0, "providers": {}}
    for season in seasons:
        hist = (history_loader or (lambda s: build_history(nfl.load_player_stats([s]).to_pandas())))(season)
        g = sched[(sched["season"] == season) & (sched["game_type"] == "REG") & sched["home_score"].notna() & sched["espn"].notna()]
        for r in g.sort_values(["week", "gameday"]).itertuples(index=False):
            eid = str(int(r.espn))
            if eid in state["done"]:
                summary["skipped_done"] += 1
                continue
            if limit_games is not None and summary["games"] >= limit_games:
                break
            prov, items = pick_provider(eid, get)
            summary["games"] += 1
            if not items:
                summary["no_data"] += 1
                state["done"].append(eid)                  # nothing to get for this game; don't retry forever
                sp.parent.mkdir(exist_ok=True)
                sp.write_text(json.dumps(state), encoding="utf-8")
                continue
            lines = main_lines(items)
            game = {"season": int(season), "week": int(r.week), "home": r.home_team, "away": r.away_team,
                    "commence": f"{r.gameday}T{r.gametime}:00" if isinstance(r.gametime, str) else str(r.gameday), "espn": eid}
            rows = rows_for_game(game, prov, lines, mapping, hist[hist["week"] == r.week])
            summary["rows"] += append_rows(rows, out_path)
            summary["providers"][prov] = summary["providers"].get(prov, 0) + 1
            state["done"].append(eid)
            sp.parent.mkdir(exist_ok=True)
            sp.write_text(json.dumps(state), encoding="utf-8")
    summary["duplicates_removed"] = dedupe(out_path)
    return summary


def validate(espn: pd.DataFrame, log_df: pd.DataFrame, hist: pd.DataFrame = None) -> dict:
    """
    Compare ESPN's closing/opening lines with the ones we logged ourselves (Odds API consensus, pulled before
    kickoff) on the props both cover. Tells us how far a single book's line sits from the consensus.
    """
    mine = log_df[log_df["pos"] != "GAME"][["season", "week", "player_id", "market", "line", "opening_line"]]
    m = espn.merge(mine, on=KEY, how="inner")
    if m.empty:
        return {"n": 0}
    d = (m["close_line"] - m["line"]).abs()
    yards = m["market"].isin(["player_pass_yds", "player_rush_yds", "player_reception_yds", "player_rush_reception_yds"])
    out = {"n": int(len(m)), "exact_close": float((d == 0).mean()), "within_half_point": float((d <= 0.5).mean()),
           "mean_abs_diff": float(d.mean()), "corr": float(m["close_line"].corr(m["line"])),
           "yards_exact": float((d[yards] == 0).mean()) if yards.any() else None,
           "yards_mean_abs_diff": float(d[yards].mean()) if yards.any() else None}
    o = m.dropna(subset=["open_line", "opening_line"])
    if len(o):
        od = (o["open_line"] - o["opening_line"]).abs()
        out["opening_exact"] = float((od == 0).mean())
        out["opening_mean_abs_diff"] = float(od.mean())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["collect", "validate", "summary"])
    ap.add_argument("--seasons", type=int, nargs="+", default=[2025])
    ap.add_argument("--limit-games", type=int, help="stop after N games (for a quick test)")
    args = ap.parse_args()
    if args.command == "collect":
        s = collect(args.seasons, limit_games=args.limit_games)
        print(json.dumps(s, indent=1))
    elif args.command == "validate":
        import props_tracker
        print(json.dumps(validate(load(), props_tracker._load()), indent=1))
    else:
        d = load()
        print({"rows": len(d), "games": int(d["espn_event"].nunique()) if len(d) else 0,
               "seasons": sorted(int(x) for x in d["season"].unique()) if len(d) else [],
               "providers": d["provider"].value_counts().to_dict() if len(d) else {},
               "markets": d["market"].value_counts().to_dict() if len(d) else {}})


if __name__ == "__main__":
    main()
