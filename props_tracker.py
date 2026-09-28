"""
props_tracker.py
=================
Closes the loop the backtest can't: tune_props.py validates the projection engine against
past STATS, but there are no historical prop LINES to check the model's edges against. This
logs what the model actually said before each game, then grades it once results are in.

Stores only our own numbers (line, our probabilities, the one price used for the pick) --
never a book-by-book odds board -- so this is an analytical record of this tool's own
performance, not a redistribution of the underlying market data.

    python props_tracker.py log                  # called by export_props.py after a real (non-sample) run
    python props_tracker.py grade                # fills in results for games that finished 5+ hours ago
    python props_tracker.py review               # audit the most recently graded week
    python props_tracker.py review --week 4       # audit a specific week (--season to disambiguate)
    python props_tracker.py report                # prints the calibration/accuracy report
    python props_tracker.py report --json          # also writes web/track_record.json and
                                                    # web/player_history.json for the page

tracking/props_log.csv is the persistent record. GitHub Actions runs are ephemeral, so the
workflow commits this file back to the repo after each run. The "line" it stores is whatever
was live when that run pulled it -- the closing line only if a run happened to land right
before kickoff, so treat it as "last line seen," not a guaranteed close.
"""

import argparse
import datetime
import json
import pathlib
import sys

import numpy as np
import pandas as pd

from props_model import MARKETS

LOG_FILE = pathlib.Path(__file__).parent / "tracking" / "props_log.csv"
REPORT_FILE = pathlib.Path(__file__).parent / "web" / "track_record.json"
HISTORY_FILE = pathlib.Path(__file__).parent / "web" / "player_history.json"
GRADE_DELAY_HOURS = 5      # a game is assumed final this long after kickoff
KEY = ["season", "week", "player_id", "market"]

COLUMNS = KEY + [
    "player", "team", "opp", "pos", "game", "commence", "kind", "line",
    "model_over", "model_under", "market_over", "gap", "trust",
    "pick", "pick_price", "pick_ev",
    "logged_at", "actual", "result", "graded_at",
]


def _load() -> pd.DataFrame:
    if LOG_FILE.exists():
        return pd.read_csv(LOG_FILE, dtype={"player_id": str})
    return pd.DataFrame(columns=COLUMNS)


def _save(df: pd.DataFrame):
    LOG_FILE.parent.mkdir(exist_ok=True)
    df.to_csv(LOG_FILE, index=False)


# ---- logging --------------------------------------------------------------
def log_predictions(snapshot: dict) -> int:
    """Upsert every prop in a real (non-sample) snapshot into the log, keyed by KEY."""
    if snapshot.get("sample"):
        return 0
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    rows = []
    for p in snapshot["props"]:
        if "player_id" not in p:
            continue
        pick_price = None
        if p.get("pick") == "over" and p.get("best_over"):
            pick_price = p["best_over"]["price"]
        elif p.get("pick") == "under" and p.get("best_under"):
            pick_price = p["best_under"]["price"]
        rows.append({
            "season": snapshot["season"], "week": snapshot["week"], "player_id": p["player_id"],
            "market": p["market"], "player": p["player"], "team": p["team"], "opp": p["opp"],
            "pos": p["pos"], "game": p["game"], "commence": p["commence"], "kind": p["kind"],
            "line": p["line"], "model_over": p["model_over"], "model_under": p["model_under"],
            "market_over": p.get("market_over"), "gap": p.get("gap"), "trust": p.get("trust"),
            "pick": p.get("pick"), "pick_price": pick_price, "pick_ev": p.get("pick_ev"),
            "logged_at": now, "actual": np.nan, "result": np.nan, "graded_at": np.nan,
        })
    if not rows:
        return 0
    new = pd.DataFrame(rows)
    old = _load()
    combined = pd.concat([old, new], ignore_index=True)
    combined = combined.drop_duplicates(subset=KEY, keep="last")   # this run's numbers win
    _save(combined)
    return len(new)


# ---- grading ----------------------------------------------------------------
def _actual_stat(stats_row, market) -> float:
    return float(sum(stats_row.get(c, 0.0) or 0.0 for c in MARKETS[market].cols))


def grade_pending(delay_hours=GRADE_DELAY_HOURS) -> int:
    """Fill in `actual`/`result` for logged props whose game started 5+ hours ago. -> rows graded."""
    import nflreadpy as nfl

    df = _load()
    if df.empty:
        return 0
    for col in ("actual", "result", "graded_at"):        # may be all-NaN (float64) until first graded
        df[col] = df[col].astype(object)
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=delay_hours)
    commence = pd.to_datetime(df["commence"], utc=True, errors="coerce")
    pending = df["actual"].isna() & commence.notna() & (commence < cutoff)
    if not pending.any():
        return 0

    graded_now = 0
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    for (season, week), idx in df[pending].groupby(["season", "week"]).groups.items():
        stats = nfl.load_player_stats([int(season)]).to_pandas()
        if not stats.empty:
            stats = stats[(stats["season"] == season) & (stats["week"] == week) & (stats["season_type"] == "REG")]
        by_pid = {r["player_id"]: r for _, r in stats.iterrows()} if not stats.empty else {}
        for i in idx:
            pid, market = df.at[i, "player_id"], df.at[i, "market"]
            row = by_pid.get(pid)
            if row is None:
                df.at[i, "result"] = "dnp"           # inactive, bye, or a name/id mismatch
                df.at[i, "graded_at"] = now
                continue
            actual = _actual_stat(row, market)
            df.at[i, "actual"] = actual
            line = df.at[i, "line"]
            if df.at[i, "kind"] == "td":
                df.at[i, "result"] = "over" if actual >= 1 else "under"
            else:
                df.at[i, "result"] = "over" if actual > line else ("push" if actual == line else "under")
            df.at[i, "graded_at"] = now
            graded_now += 1
    _save(df)
    return graded_now


# ---- reporting ----------------------------------------------------------------
def _pick_profit(row):
    if row["result"] == "push" or pd.isna(row["pick_price"]):
        return 0.0
    won = row["result"] == row["pick"]
    price = row["pick_price"]
    payout = price / 100 if price > 0 else 100 / -price
    return payout if won else -1.0


def calibration_table(g: pd.DataFrame, col: str, bins=(0, 0.40, 0.50, 0.60, 1.01)) -> list:
    """[{range, n, predicted, actual}]: does col's probability match the real hit rate?"""
    over_hit = np.where(g["result"] == "over", 1.0, np.where(g["result"] == "push", 0.5, 0.0))
    cut = pd.cut(g[col], bins=bins, right=False)
    out = []
    for interval, idx in g.groupby(cut, observed=True).groups.items():
        if len(idx) == 0:
            continue
        out.append({"range": f"{interval.left:.0%}-{interval.right:.0%}" if interval.right <= 1 else f"{interval.left:.0%}+",
                    "n": int(len(idx)), "predicted": round(float(g.loc[idx, col].mean()), 3),
                    "actual": round(float(over_hit[g.index.get_indexer(idx)].mean()), 3)})
    return out


def build_report(min_n=1) -> dict:
    df = _load()
    g = df[df["result"].notna() & (df["result"] != "dnp")].copy()
    if g.empty:
        return {"n_graded": 0, "updated": None}

    g["profit"] = g.apply(_pick_profit, axis=1)
    picks = g[g["pick"].notna()]
    by_market = []
    for m, gm in g.groupby("market"):
        pm = picks[picks["market"] == m]
        by_market.append({
            "market": m, "n": int(len(gm)),
            "pick_n": int(len(pm)), "pick_win_rate": None if pm.empty else round(float((pm["result"] == pm["pick"]).mean()), 3),
            "roi": None if pm.empty else round(float(pm["profit"].mean()), 4),
        })

    return {
        "updated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "n_graded": int(len(g)), "n_picks": int(len(picks)),
        "pick_win_rate": None if picks.empty else round(float((picks["result"] == picks["pick"]).mean()), 3),
        "roi_per_dollar": None if picks.empty else round(float(picks["profit"].mean()), 4),
        "by_market": by_market,
        "calibration_adjusted": calibration_table(g[g["kind"] != "td"], "model_over"),
        "note": "roi_per_dollar is flat $1-per-pick return at the price recorded when the pick was logged, "
                "not a simulated bankroll. Small samples early on will be noisy.",
    }


def week_review(season=None, week=None):
    """
    All logged props for one week, graded or not, for manual review. Defaults to the most
    recently GRADED week so `--week` can be left off day-to-day. -> (DataFrame, season, week).
    """
    df = _load()
    if df.empty:
        return df, season, week
    graded = df[df["result"].notna()]
    if week is None:
        pool = graded if not graded.empty else df
        if season is not None:
            pool = pool[pool["season"] == season]
        if pool.empty:
            return pool, season, week
        season = int(pool["season"].max()) if season is None else season
        week = int(pool[pool["season"] == season]["week"].max())
    elif season is None:
        season = int(df["season"].max())
    g = df[(df["season"] == season) & (df["week"] == week)].copy()
    g["hit_pick"] = np.where(g["pick"].isna() | g["result"].isna(), np.nan, (g["result"] == g["pick"]).astype(float))
    return g.sort_values(["market", "player"]), season, week


def print_week_review(g: pd.DataFrame, season, week):
    if g is None or g.empty:
        print(f"No props logged for {season or '?'} week {week or '?'} yet.")
        return
    ungraded = int(g["result"].isna().sum())
    dnp = int((g["result"] == "dnp").sum())
    print(f"Week {week}, {season} -- {len(g)} props logged"
          + (f", {ungraded} not yet graded" if ungraded else "") + (f", {dnp} dnp" if dnp else ""))
    cols = ["player", "market", "line", "actual", "result", "pick", "model_over", "market_over"]
    with pd.option_context("display.width", 140, "display.max_rows", None):
        print(g[cols].to_string(index=False, na_rep="—"))
    picks = g[g["pick"].notna() & g["result"].notna() & (g["result"] != "dnp")]
    if len(picks):
        print(f"\nPicks graded: {len(picks)}  win rate {float((picks['result'] == picks['pick']).mean()):.1%}")


def build_player_history(min_games=1) -> dict:
    """{player_id: {player, team, pos, markets: {market: {label, kind, games: [...]}}}} from graded history."""
    df = _load()
    g = df[df["result"].notna() & (df["result"] != "dnp")].copy()
    if g.empty:
        return {}
    out = {}
    for pid, pg in g.sort_values(["season", "week"]).groupby("player_id"):
        markets = {}
        for m, mg in pg.groupby("market"):
            if len(mg) < min_games or m not in MARKETS:
                continue
            spec = MARKETS[m]
            markets[m] = {"label": spec.label, "kind": spec.kind, "games": [
                {"season": int(r.season), "week": int(r.week), "opp": r.opp,
                 "line": float(r.line), "actual": float(r.actual), "result": r.result,
                 "model_over": None if pd.isna(r.model_over) else float(r.model_over)}
                for r in mg.itertuples()
            ]}
        if markets:
            last = pg.iloc[-1]
            out[pid] = {"player": last["player"], "team": last["team"], "pos": last["pos"], "markets": markets}
    return out


def print_report(rep: dict):
    if not rep["n_graded"]:
        print("No graded props yet. Run `python props_tracker.py grade` after some games have finished.")
        return
    print(f"Graded props: {rep['n_graded']}  |  picks made: {rep['n_picks']}")
    if rep["n_picks"]:
        print(f"Pick win rate: {rep['pick_win_rate']:.1%}   ROI per $1: {rep['roi_per_dollar']:+.1%}")
    print(f"\n{'market':28s} {'n':>5s} {'picks':>6s} {'win%':>7s} {'roi':>8s}")
    for r in rep["by_market"]:
        w = "—" if r["pick_win_rate"] is None else f"{r['pick_win_rate']:.1%}"
        roi = "—" if r["roi"] is None else f"{r['roi']:+.1%}"
        print(f"{r['market']:28s} {r['n']:5d} {r['pick_n']:6d} {w:>7s} {roi:>8s}")
    print("\nCalibration (adjusted model P(over) vs. how often it actually went over):")
    print(f"{'range':>10s} {'n':>6s} {'predicted':>10s} {'actual':>8s}")
    for c in rep["calibration_adjusted"]:
        print(f"{c['range']:>10s} {c['n']:6d} {c['predicted']:10.1%} {c['actual']:8.1%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["log", "grade", "report", "review"])
    ap.add_argument("--json", action="store_true", help="with report: also write web/track_record.json and web/player_history.json")
    ap.add_argument("--week", type=int, help="with review: which week (default: most recently graded)")
    ap.add_argument("--season", type=int, help="with review: which season (default: current)")
    args = ap.parse_args()

    if args.cmd == "log":
        import json as _json
        snap = _json.loads((pathlib.Path("web") / "props_snapshot.json").read_text(encoding="utf-8"))
        n = log_predictions(snap)
        print(f"Logged/updated {n} props.", file=sys.stderr)
    elif args.cmd == "grade":
        n = grade_pending()
        print(f"Graded {n} props.", file=sys.stderr)
    elif args.cmd == "review":
        g, season, week = week_review(args.season, args.week)
        print_week_review(g, season, week)
    elif args.cmd == "report":
        rep = build_report()
        print_report(rep)
        if args.json:
            REPORT_FILE.write_text(json.dumps(rep, separators=(",", ":")), encoding="utf-8")
            HISTORY_FILE.write_text(json.dumps(build_player_history(), separators=(",", ":")), encoding="utf-8")
            print(f"\nWrote {REPORT_FILE} and {HISTORY_FILE}", file=sys.stderr)


if __name__ == "__main__":
    main()
