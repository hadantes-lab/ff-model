"""
line_history.py
===============
The backlog: an APPEND-ONLY record of every line pull, so the season builds a dataset nobody can
buy back later (the Odds API charges for historical props, and only offers them from a point in time).

`tracking/props_log.csv` (props_tracker.py) keeps ONE row per prop and overwrites it on each pull, so
it holds the opening and latest line but not the trail between. This file keeps the trail:

    one row per prop (or game spread/total) per pull:
    pulled_at, season, week, kind, market, player_id, player, team, opp, game, commence,
    line, fair_over, best_over, best_under, n_books, mean, model_over, injury

From it you can derive, for any prop or game:
    opening line  = first row            closing line = last row pulled before kickoff
    line movement = the whole path       how the market's no-vig price moved (fair_over)
and, joined to results, how far lines move toward the truth. `closing_lines()` does that.

WHAT IS STORED, AND WHAT IS NOT
-------------------------------
Only our own derived numbers: the consensus (median) line, the no-vig probability, the single best
price on each side, and how many books were in the consensus. NOT stored: per-book prices or book
names. The repo is public and the Odds API's terms bar republishing an odds board, so a per-book
history would not be OK here -- if that is ever wanted it belongs in private storage.

Rows are only ever appended; nothing here edits or deletes earlier pulls. `seed_from_log` fills in
the pulls that predate this file from the tracking log's existing rows (one row per prop, at the
time it was logged).
"""

import csv
import datetime
import pathlib

import pandas as pd

HISTORY_FILE = pathlib.Path(__file__).parent / "tracking" / "line_history.csv"
COLUMNS = ["pulled_at", "season", "week", "kind", "market", "player_id", "player", "team", "opp", "game", "commence",
           "line", "fair_over", "best_over", "best_under", "n_books", "mean", "model_over", "injury"]
KEY = ["season", "week", "player_id", "market"]


def _price(obj):
    return None if not obj else obj.get("price")


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def rows_from_snapshot(snapshot: dict, pulled_at: str = None) -> list:
    """Every prop and priced game in an export_props snapshot, as history rows."""
    pulled_at = pulled_at or now_iso()
    rows = []
    for p in snapshot.get("props", []):
        if "player_id" not in p:
            continue
        rows.append({"pulled_at": pulled_at, "season": snapshot["season"], "week": snapshot["week"], "kind": "prop",
                     "market": p["market"], "player_id": p["player_id"], "player": p["player"], "team": p["team"],
                     "opp": p["opp"], "game": p["game"], "commence": p["commence"], "line": p["line"],
                     "fair_over": p.get("market_over"), "best_over": _price(p.get("best_over")),
                     "best_under": _price(p.get("best_under")), "n_books": len(p.get("books") or []),
                     "mean": p.get("mean"), "model_over": p.get("model_over"), "injury": p.get("injury")})
    for g in snapshot.get("games", []):
        week = g.get("week", snapshot["week"])
        for market, key, line, fair, bo, bu, mean, model in (
                ("game_spread", "spread", g["home_spread"], g.get("fair_home_cover"), g.get("best_home"),
                 g.get("best_away"), g.get("proj_margin"), g.get("model_home_cover")),
                ("game_total", "total", g["total"], g.get("fair_over"), g.get("best_over"), g.get("best_under"),
                 g.get("proj_total"), g.get("model_over"))):
            rows.append({"pulled_at": pulled_at, "season": snapshot["season"], "week": week, "kind": "game",
                         "market": market, "player_id": f"GAME_{g['home']}_{g['away']}_{key}", "player": g["game"],
                         "team": g["home"], "opp": g["away"], "game": g["game"], "commence": g["commence"],
                         "line": line, "fair_over": fair, "best_over": _price(bo), "best_under": _price(bu),
                         "n_books": None, "mean": mean, "model_over": model, "injury": None})
    return rows


def rows_from_game_lines(events, lines, prices, week_for, season, pulled_at=None, abbr=None) -> list:
    """
    Game spread/total rows straight from a bulk game-lines pull -- no model needed, so this is the
    cheap off-day collector's path. `events` is the raw Odds API list, `lines`/`prices` are
    odds_api.consensus_game_lines / consensus_game_prices, `week_for(home, away, commence)` -> week.
    """
    import odds_api

    abbr = abbr or odds_api.TEAM_ABBR
    pulled_at = pulled_at or now_iso()
    rows = []
    for ev in events:
        ln = lines.get(ev["id"], {})
        if ln.get("total") is None or ln.get("home_spread") is None:
            continue
        home, away = abbr[ev["home_team"]], abbr[ev["away_team"]]
        pr = prices.get(ev["id"], {})
        week = week_for(home, away, ev["commence_time"])
        for market, key, line, fair, bo, bu in (
                ("game_spread", "spread", ln["home_spread"], pr.get("fair_home_cover"), pr.get("best_home"), pr.get("best_away")),
                ("game_total", "total", ln["total"], pr.get("fair_over"), pr.get("best_over"), pr.get("best_under"))):
            rows.append({"pulled_at": pulled_at, "season": season, "week": week, "kind": "game", "market": market,
                         "player_id": f"GAME_{home}_{away}_{key}", "player": f"{away} @ {home}", "team": home,
                         "opp": away, "game": f"{away} @ {home}", "commence": ev["commence_time"], "line": line,
                         "fair_over": fair, "best_over": _price(bo), "best_under": _price(bu), "n_books": None,
                         "mean": None, "model_over": None, "injury": None})
    return rows


def append_rows(rows: list, path=None) -> int:
    """Append rows (header written once). Never rewrites existing lines. -> rows written."""
    path = pathlib.Path(path or HISTORY_FILE)
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


def load(path=None) -> pd.DataFrame:
    path = pathlib.Path(path or HISTORY_FILE)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path)


def seed_from_log(log_df: pd.DataFrame, path=None) -> int:
    """
    One-time backfill from the tracking log: its rows are one pull each (at `logged_at`), so each
    becomes a history row. Skips anything already present, so it is safe to run twice.
    """
    if log_df is None or log_df.empty:
        return 0
    have = load(path)
    seen = set(zip(have["pulled_at"], have["season"], have["week"], have["player_id"], have["market"])) if len(have) else set()
    rows = []
    for r in log_df.itertuples(index=False):
        if (r.logged_at, r.season, r.week, r.player_id, r.market) in seen:
            continue
        rows.append({"pulled_at": r.logged_at, "season": r.season, "week": r.week,
                     "kind": "game" if r.pos == "GAME" else "prop", "market": r.market, "player_id": r.player_id,
                     "player": r.player, "team": r.team, "opp": r.opp, "game": r.game, "commence": r.commence,
                     "line": r.line, "fair_over": r.market_over, "best_over": None, "best_under": None,
                     "n_books": None, "mean": None, "model_over": r.model_over, "injury": None})
    rows.sort(key=lambda r: r["pulled_at"])
    return append_rows(rows, path)


def closing_lines(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per prop/game: opening line (first pull), closing line (last pull strictly before kickoff),
    how many pulls, and the move between them. Pulls after kickoff are live lines and are ignored.
    """
    if df.empty:
        return pd.DataFrame(columns=KEY + ["opening_line", "closing_line", "n_pulls", "move", "open_fair", "close_fair"])
    d = df.copy()
    d["_t"] = pd.to_datetime(d["pulled_at"], utc=True, errors="coerce")
    d["_k"] = pd.to_datetime(d["commence"], utc=True, errors="coerce")
    d = d[d["_t"].notna() & (d["_k"].isna() | (d["_t"] < d["_k"]))].sort_values("_t")
    g = d.groupby(KEY, sort=False)
    out = g.agg(opening_line=("line", "first"), closing_line=("line", "last"), n_pulls=("line", "size"),
                open_fair=("fair_over", "first"), close_fair=("fair_over", "last"),
                first_pull=("_t", "first"), last_pull=("_t", "last")).reset_index()
    out["move"] = (out["closing_line"] - out["opening_line"]).round(3)
    return out


def closing_map(df: pd.DataFrame = None) -> dict:
    """
    {(player_id, market): {(season, week): closing line}} for props, from the backlog (including anything
    backfilled from the historical API). Feeds the hit-rate chart's "Line then" view.
    """
    df = load() if df is None else df
    c = closing_lines(df[df["kind"] == "prop"]) if len(df) else closing_lines(df)
    out = {}
    for r in c.itertuples(index=False):
        out.setdefault((r.player_id, r.market), {})[(int(r.season), int(r.week))] = float(r.closing_line)
    return out


def summary(df: pd.DataFrame) -> dict:
    """How much backlog there is: rows, distinct pulls, props/games covered, date span, per week."""
    if df.empty:
        return {"rows": 0, "pulls": 0}
    return {"rows": int(len(df)), "pulls": int(df["pulled_at"].nunique()),
            "props": int(df[df["kind"] == "prop"][KEY].drop_duplicates().shape[0]),
            "games": int(df[df["kind"] == "game"][KEY].drop_duplicates().shape[0]),
            "first": str(df["pulled_at"].min()), "last": str(df["pulled_at"].max()),
            "per_week": {int(w): int(n) for w, n in df.groupby("week").size().items()}}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Inspect or seed the line-history backlog.")
    ap.add_argument("command", choices=["summary", "seed", "closing"])
    args = ap.parse_args()
    if args.command == "seed":
        import props_tracker

        print(f"seeded {seed_from_log(props_tracker._load())} rows from the tracking log")
    elif args.command == "closing":
        c = closing_lines(load())
        print(c.sort_values("move", key=abs, ascending=False).head(20).to_string(index=False))
    else:
        print(summary(load()))
