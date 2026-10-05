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
    "opening_line", "line_move",
    "model_over", "model_under", "market_over", "gap", "trust",
    "pick", "pick_price", "pick_ev",
    "logged_at", "actual", "result", "graded_at",
]
BIG_MOVE = {"yards": 3.0, "count": 1.0, "spread": 1.5, "total": 2.0}  # per-kind threshold to flag a line move


def _load() -> pd.DataFrame:
    if LOG_FILE.exists():
        return pd.read_csv(LOG_FILE, dtype={"player_id": str})
    return pd.DataFrame(columns=COLUMNS)


def _save(df: pd.DataFrame):
    LOG_FILE.parent.mkdir(exist_ok=True)
    df.to_csv(LOG_FILE, index=False)


# ---- logging --------------------------------------------------------------
def week_for_game(sched: pd.DataFrame, home: str, away: str, commence, default=None):
    """
    The NFL week a game belongs to, from the schedule: the row for this home/away pairing whose
    date is closest to `commence` (a pairing can repeat in a season, e.g. a rematch). The Odds
    API's bulk game-lines call returns every posted game, including NEXT week's, so a game's week
    cannot be assumed to be the current one. -> `default` if the schedule has no such game.
    """
    if sched is None or sched.empty:
        return default
    m = sched[(sched["home_team"] == home) & (sched["away_team"] == away)]
    if m.empty:
        return default
    when = pd.to_datetime(commence, utc=True, errors="coerce")
    if pd.isna(when):
        return int(m["week"].iloc[0]) if len(m) == 1 else default
    days = (pd.to_datetime(m["gameday"], utc=True) - when).abs()
    return int(m.loc[days.idxmin(), "week"])


def relabel_game_weeks(sched: pd.DataFrame) -> int:
    """One-time repair: re-derive each logged game row's week from the schedule (earlier versions
    stamped every game with the current week). Moves rows between weeks only; never drops any.
    -> rows changed."""
    df = _load()
    if df.empty:
        return 0
    changed = 0
    for i in df.index[df["market"].isin(("game_spread", "game_total"))]:
        wk = week_for_game(sched[sched["season"] == df.at[i, "season"]], df.at[i, "team"], df.at[i, "opp"],
                           df.at[i, "commence"], default=df.at[i, "week"])
        if wk != df.at[i, "week"]:
            df.at[i, "week"] = wk
            changed += 1
    if changed:
        _save(df.drop_duplicates(subset=KEY, keep="last"))
    return changed


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
    for g in snapshot.get("games", []):
        # Spread picks are logged in the same over/under/push vocabulary as everything else,
        # with "over" standing for "home covers" -- so grading, ROI, and calibration need no
        # special case for games at all (the page itself still shows "home"/"away").
        spread_pick = {"home": "over", "away": "under"}.get(g.get("spread_pick"))
        for market, key, line, model_over, model_under, pick, pick_price_obj, pick_ev, trust in (
            ("game_spread", "spread", g["home_spread"], g["model_home_cover"], 1 - g["model_home_cover"],
             spread_pick, g.get("spread_pick_price"), g.get("spread_pick_ev"), g.get("spread_trust")),
            ("game_total", "total", g["total"], g["model_over"], 1 - g["model_over"],
             g.get("total_pick"), g.get("total_pick_price"), g.get("total_pick_ev"), g.get("total_trust")),
        ):
            rows.append({
                "season": snapshot["season"], "week": g.get("week", snapshot["week"]),
                "player_id": f"GAME_{g['home']}_{g['away']}_{key}",
                "market": market, "player": g["game"], "team": g["home"], "opp": g["away"],
                "pos": "GAME", "game": g["game"], "commence": g["commence"], "kind": key,
                "line": line, "model_over": model_over, "model_under": model_under,
                "market_over": g.get("fair_home_cover") if key == "spread" else g.get("fair_over"),
                "gap": None, "trust": trust,
                "pick": pick, "pick_price": pick_price_obj["price"] if pick_price_obj else None, "pick_ev": pick_ev,
                "logged_at": now, "actual": np.nan, "result": np.nan, "graded_at": np.nan,
            })
    if not rows:
        return 0

    # opening_line is set once and carried forward on every later pull, so a Monday line and a
    # Thursday line for the same prop/game can be compared even though the row itself gets
    # upserted (see the class docstring: only games/props that happen to be pulled more than
    # once actually get this -- mainly games, whose bulk line call covers the whole week cheaply).
    old = _load()
    opening_by_key = {}
    if not old.empty:
        for r in old.itertuples(index=False):
            k = tuple(getattr(r, c) for c in KEY)
            prior = getattr(r, "opening_line", None) if "opening_line" in old.columns else None
            opening_by_key[k] = prior if prior == prior else r.line   # NaN-safe fallback to its own line
    for row in rows:
        k = tuple(row[c] for c in KEY)
        row["opening_line"] = opening_by_key.get(k, row["line"])
        row["line_move"] = (None if row["line"] is None or row["opening_line"] is None
                            else round(row["line"] - row["opening_line"], 3))

    new = pd.DataFrame(rows)
    combined = pd.concat([old, new], ignore_index=True)
    combined = combined.drop_duplicates(subset=KEY, keep="last")   # this run's numbers win
    _save(combined)
    return len(new)


def significant_moves(min_by_kind=BIG_MOVE):
    """Logged props/games whose line has moved more than a kind-appropriate threshold since
    the first pull, largest move first. Mostly populated for games (see log_predictions);
    player props typically get pulled only once under the --days-limited schedule."""
    df = _load()
    if df.empty or "line_move" not in df.columns or "opening_line" not in df.columns:
        return df.iloc[0:0]
    moved = df[df["line_move"].notna() & (df["line_move"] != 0)].copy()
    if moved.empty:
        return moved
    threshold = moved["kind"].map(min_by_kind).fillna(1.0)
    big = moved[moved["line_move"].abs() >= threshold].copy()
    return big.reindex(big["line_move"].abs().sort_values(ascending=False).index)


# ---- grading ----------------------------------------------------------------
def _actual_stat(stats_row, market) -> float:
    return float(sum(stats_row.get(c, 0.0) or 0.0 for c in MARKETS[market].cols))


def _grade_over_under(df, i, actual, now):
    df.at[i, "actual"] = actual
    line = df.at[i, "line"]
    if df.at[i, "kind"] == "td":
        df.at[i, "result"] = "over" if actual >= 1 else "under"
    else:
        df.at[i, "result"] = "over" if actual > line else ("push" if actual == line else "under")
    df.at[i, "graded_at"] = now


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

        final = {}
        if df.loc[idx, "market"].isin(("game_spread", "game_total")).any():
            sched = nfl.load_schedules([int(season)]).to_pandas()
            sched = sched[sched["week"] == week] if not sched.empty else sched
            final = {(r["home_team"], r["away_team"]): (r["home_score"], r["away_score"])
                    for _, r in sched.iterrows() if pd.notna(r["home_score"])} if not sched.empty else {}

        for i in idx:
            market = df.at[i, "market"]
            if market in ("game_spread", "game_total"):
                score = final.get((df.at[i, "team"], df.at[i, "opp"]))   # team/opp = home/away for game rows
                if score is None:
                    continue   # game hasn't finished yet even though kickoff has passed -- try again later
                home_score, away_score = score
                if market == "game_total":
                    _grade_over_under(df, i, float(home_score + away_score), now)
                else:
                    # home_spread < 0 means home favored; home covers iff margin > -home_spread.
                    # The stored "line" is the natural home_spread value (e.g. "-2.5"), so this
                    # can't reuse _grade_over_under's plain `actual > line` -- grade directly.
                    margin = float(home_score - away_score)
                    threshold = -df.at[i, "line"]
                    df.at[i, "actual"] = margin
                    df.at[i, "result"] = "over" if margin > threshold else ("push" if margin == threshold else "under")
                    df.at[i, "graded_at"] = now
                graded_now += 1
                continue
            row = by_pid.get(df.at[i, "player_id"])
            if row is None:
                df.at[i, "result"] = "dnp"           # inactive, bye, or a name/id mismatch
                df.at[i, "graded_at"] = now
                continue
            _grade_over_under(df, i, _actual_stat(row, market), now)
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


# ---- weekly calibration refinement ------------------------------------------
LIVE_CALIBRATION_FILE = pathlib.Path(__file__).parent / "live_calibration.json"
REFINE_MIN_N = 50          # graded, priced picks a market needs before its trust weight can be refit
REFINE_MIN_IMPROVEMENT = 0.02   # the refit weight must beat the current default's Brier score by this much
REFINE_WEIGHT_BOUNDS = (0.0, 0.5)   # a refined weight is always clamped into this range, however good the fit looks


def _brier(pred, actual):
    return float(np.mean((pred - actual) ** 2))


def refine_trust_weights(min_n=REFINE_MIN_N, current_weight=0.15):
    """
    For each market with >= min_n graded, priced picks, grid-search the market-blend trust
    weight (see props_model.blend_toward_market / game_odds.blend_game_prob) that would have
    minimized Brier score against real tracked outcomes so far, and compare it to
    `current_weight` (the static default currently in use). A market only gets a suggested
    override if it clears REFINE_MIN_IMPROVEMENT over that default -- otherwise a handful of
    close scores from a middling sample would flip the weight around for no real reason.

    Needs BOTH the raw model probability and the market's own fair probability logged, which is
    only true for props/games where a real market price existed (sample runs and pick'em-only
    props are excluded automatically since those never populate market_over).
    -> {market: {"n", "current_brier", "suggested_weight", "suggested_brier"}} for markets with
    enough data to refine; markets below min_n are left out entirely (still use the static default).
    """
    df = _load()
    g = df[df["result"].notna() & (df["result"] != "dnp") & df["market_over"].notna()
          & df["model_over"].notna()].copy()
    if g.empty:
        return {}
    out = {}
    for m, gm in g.groupby("market"):
        if len(gm) < min_n:
            continue
        actual = np.where(gm["result"] == "over", 1.0, np.where(gm["result"] == "push", 0.5, 0.0))
        raw, mkt = gm["model_over"].to_numpy(float), gm["market_over"].to_numpy(float)
        current_brier = _brier(mkt + current_weight * (raw - mkt), actual)
        grid = np.linspace(*REFINE_WEIGHT_BOUNDS, 26)
        scored = [(w, _brier(mkt + w * (raw - mkt), actual)) for w in grid]
        best_w, best_brier = min(scored, key=lambda t: t[1])
        if current_brier - best_brier < REFINE_MIN_IMPROVEMENT:
            continue   # not a clear enough win over the default to trust yet
        out[m] = {"n": int(len(gm)), "current_brier": round(current_brier, 4),
                  "suggested_weight": round(float(best_w), 3), "suggested_brier": round(best_brier, 4)}
    return out


def write_live_calibration(min_n=REFINE_MIN_N):
    refined = refine_trust_weights(min_n)
    LIVE_CALIBRATION_FILE.write_text(json.dumps({
        "updated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "min_n": min_n, "weights": refined,
        "note": "Per-market override for the market-blend trust weight, refit weekly from "
                "props_tracker.py's own tracked results (not the historical-stats backtest "
                "tune_props.py uses). A market only appears once it has enough graded picks "
                "and the refit clearly beats the static default; see REFINE_MIN_N/"
                "REFINE_MIN_IMPROVEMENT in props_tracker.py.",
    }, indent=1), encoding="utf-8")
    return refined


def load_live_calibration():
    try:
        return json.loads(LIVE_CALIBRATION_FILE.read_text(encoding="utf-8")).get("weights", {})
    except (OSError, ValueError):
        return {}


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
        "big_moves": [
            {"player": r.player, "market": r.market, "opening_line": r.opening_line,
             "line": r.line, "move": r.line_move}
            for r in significant_moves().head(10).itertuples()
        ],
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
    g = df[(df["pos"] != "GAME") & df["result"].notna() & (df["result"] != "dnp")].copy()
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
    if rep.get("big_moves"):
        print("\nBiggest line moves since first pulled:")
        for m in rep["big_moves"]:
            print(f"  {m['player']:24s} {m['market']:16s} {m['opening_line']:>7.1f} -> {m['line']:<7.1f} ({m['move']:+.1f})")


def print_moves(df: pd.DataFrame):
    if df is None or df.empty:
        print("No line moves logged yet (most props are only pulled once under the current schedule; "
              "games are pulled every run, so check back after a couple of scheduled runs).")
        return
    print(f"{len(df)} prop(s)/game(s) with a notable line move since first pulled:")
    cols = ["player", "market", "opening_line", "line", "line_move", "season", "week"]
    with pd.option_context("display.width", 140, "display.max_rows", None):
        print(df[cols].to_string(index=False, na_rep="—"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["log", "grade", "report", "review", "moves", "refine"])
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
    elif args.cmd == "moves":
        print_moves(significant_moves())
    elif args.cmd == "refine":
        refined = write_live_calibration()
        if not refined:
            print(f"No market has {REFINE_MIN_N}+ graded, priced picks with a clear-enough "
                  f"improvement yet -- keeping the static default everywhere.", file=sys.stderr)
        else:
            print(f"Refined trust weight for {len(refined)} market(s):", file=sys.stderr)
            for m, r in refined.items():
                print(f"  {m:28s} n={r['n']:4d}  weight -> {r['suggested_weight']:.2f}  "
                      f"(brier {r['current_brier']:.4f} -> {r['suggested_brier']:.4f})", file=sys.stderr)
        print(f"Wrote {LIVE_CALIBRATION_FILE}", file=sys.stderr)
    elif args.cmd == "report":
        rep = build_report()
        print_report(rep)
        if args.json:
            REPORT_FILE.write_text(json.dumps(rep, separators=(",", ":")), encoding="utf-8")
            HISTORY_FILE.write_text(json.dumps(build_player_history(), separators=(",", ":")), encoding="utf-8")
            print(f"\nWrote {REPORT_FILE} and {HISTORY_FILE}", file=sys.stderr)


if __name__ == "__main__":
    main()
