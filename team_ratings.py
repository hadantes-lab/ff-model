"""
team_ratings.py
===============
Weekly power rankings (1st to 32nd), schedule-adjusted, and the point spread they imply.

HOW A RATING IS BUILT
---------------------
Every past game says "home team performed X better than the away team". A rating system finds the
one number per team that best explains all those results at once, so beating a strong team counts
for more than beating a weak one -- that is the strength-of-schedule adjustment, done jointly
rather than as an after-the-fact tweak:

    performance_margin(game) ~ rating(home) - rating(away) + home_field

solved as a recency-weighted ridge regression over a team's last ~2 seasons. The ridge penalty pulls
every team toward average, hardest when there is little data, so in week 1 a team is still mostly
its prior-season self and by week 8 mostly this season's -- the "use the first weeks of data"
behaviour falls out of the math instead of being hand-tuned.

WHAT "PERFORMANCE" MEANS (tested, not assumed -- see `python team_ratings.py`)
-----------------------------------------------------------------------------
Final score margin is noisy (a fluky return TD counts as much as a dominant drive), so the target
can instead be built from the stats behind the score. `TARGETS` lists the candidates; the
backtest scores each one by how well its ratings predict the NEXT week's results. Takeaways and
giveaways (the Samford model's inputs) are included as an option and tested like everything else.

RANK -> SPREAD
--------------
The user-specified convention: the 15th-ranked team is neutral, and the best team is
`POINTS_AT_TOP` (6) points better than that 15th team, linearly:

    points_per_rank = 6 / 14
    rank_margin(home vs away) = (away_rank - home_rank) * points_per_rank + home_field

so #1 at the 15th team is -6 (+ home field), and #1 vs #32 is about -13.3.
"""

import pathlib

import numpy as np
import pandas as pd

import team_stats as ts

POINTS_AT_TOP = 6.0           # the best team vs the 15th-ranked team, neutral field
NEUTRAL_RANK = 15
PER_RANK = POINTS_AT_TOP / (NEUTRAL_RANK - 1)
HOME_FIELD = 1.9              # fitted each week alongside the ratings; only the no-data fallback
WEEKS_PER_SEASON = 18.0       # only for ordering time; byes make the exact value unimportant
RATING_HALFLIFE_WEEKS = 8.0     # walk-forward best of 4-40 (a flat optimum: 6-10 are within noise)
RATING_LAMBDA = 6.0

# name -> function(game row) -> home-perspective performance margin in points
QB_CHANGE_POINTS = 2.4        # a team playing someone other than its usual starter: ~2.4 pts against our
                              # rating (t = 4.0, 2021-26); the market already prices it, so this is a rating fix
GAME_MARGIN_SD = 13.2         # std of (actual - predicted margin) in the walk-forward: margin -> cover chance
POINTS_PER_EPA_PLAY = 62.0    # ~plays per team-game: converts EPA/play into points per game
POINTS_PER_TURNOVER = 4.0     # a turnover costs roughly this many points of expected scoring


def _margin(g):
    return g["home_score"] - g["away_score"]


def _epa_margin(g):
    return POINTS_PER_EPA_PLAY * ((g["epa_pp_h"] - g["epa_pp_a"]))


def _epa_def_aware(g):
    # offense-minus-defense on each side: what each side did, minus what its opponent's offense did
    return POINTS_PER_EPA_PLAY * ((g["epa_pp_h"] - g["epa_pp_allowed_h"]) - (g["epa_pp_a"] - g["epa_pp_allowed_a"])) / 2


def _ypp_margin(g):
    return 9.0 * (g["ypp_h"] - g["ypp_a"])


def _turnover_margin(g):
    return POINTS_PER_TURNOVER * (g["giveaways_a"] - g["giveaways_h"])


TARGETS = {
    "points": _margin,
    "epa": _epa_margin,
    "yards_per_play": _ypp_margin,
    "points+epa": lambda g: 0.5 * _margin(g) + 0.5 * _epa_margin(g),
    "epa+turnovers": lambda g: _epa_margin(g) + _turnover_margin(g),
    "points+epa+turnovers": lambda g: 0.4 * _margin(g) + 0.4 * _epa_margin(g) + 0.2 * (_epa_margin(g) + _turnover_margin(g)),
}


# ---- data --------------------------------------------------------------------
def build_games(tg: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    """One row per completed regular-season game: schedule fields + both teams' game stats."""
    cols = ["epa_pp", "epa_pp_allowed", "ypp", "giveaways"]
    h = tg[["game_id", "team"] + cols].rename(columns={"team": "home_team", **{c: f"{c}_h" for c in cols}})
    a = tg[["game_id", "team"] + cols].rename(columns={"team": "away_team", **{c: f"{c}_a" for c in cols}})
    g = sched.merge(h, on=["game_id", "home_team"]).merge(a, on=["game_id", "away_team"])
    g = g[g["home_score"].notna()].copy()
    g["t"] = (g["season"] - g["season"].min()) * WEEKS_PER_SEASON + g["week"]
    return g.sort_values(["season", "week"]).reset_index(drop=True)


# ---- rating fit ---------------------------------------------------------------
def fit_ratings(games: pd.DataFrame, teams, as_of_t: float, target="points", halflife=RATING_HALFLIFE_WEEKS,
                lam=RATING_LAMBDA, fit_hfa=True):
    """
    Ratings from every game before time `as_of_t`. -> ({team: rating}, home_field_points).
    Weighted ridge: minimize sum_w (y - (r_h - r_a + hfa*home))^2 + lam * sum r^2.
    """
    fn = TARGETS[target] if isinstance(target, str) else target
    past = games[games["t"] < as_of_t]
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    if past.empty:
        return {t: 0.0 for t in teams}, HOME_FIELD
    X = np.zeros((len(past), n + 1))
    hi = past["home_team"].map(idx).to_numpy()
    ai = past["away_team"].map(idx).to_numpy()
    X[np.arange(len(past)), hi] = 1.0
    X[np.arange(len(past)), ai] = -1.0
    neutral = (past["location"] == "Neutral").to_numpy() if "location" in past else np.zeros(len(past), bool)
    X[:, n] = np.where(neutral, 0.0, 1.0)
    y = fn(past).to_numpy(float)
    w = 0.5 ** ((as_of_t - past["t"].to_numpy(float)) / halflife)
    ok = np.isfinite(y)
    X, y, w = X[ok], y[ok], w[ok]
    pen = np.eye(n + 1) * lam
    pen[n, n] = 0.0 if fit_hfa else 1e9               # home field is not shrunk (or is pinned to 0)
    A = X.T @ (X * w[:, None]) + pen
    b = X.T @ (w * y)
    beta = np.linalg.solve(A, b)
    hfa = float(beta[n]) if fit_hfa else HOME_FIELD
    r = beta[:n]
    r = r - r.mean()
    return {t: float(v) for t, v in zip(teams, r)}, hfa


def rank_teams(ratings: dict) -> dict:
    """{team: rank} with 1 = best."""
    order = sorted(ratings, key=lambda t: -ratings[t])
    return {t: i + 1 for i, t in enumerate(order)}


def rank_margin(home_rank: int, away_rank: int, hfa=HOME_FIELD, neutral=False, per_rank=PER_RANK) -> float:
    """Predicted home margin from ranks alone (the user's convention; see module docstring)."""
    return (away_rank - home_rank) * per_rank + (0.0 if neutral else hfa)


# ---- walk-forward backtest ------------------------------------------------------
def walk_forward(games: pd.DataFrame, target="points", first_season=2021, **fit_kw) -> pd.DataFrame:
    """
    For every game in every week from `first_season` on, the prediction made using ONLY earlier
    games: rating-based margin, rank-based margin, and the market spread to compare with.
    """
    teams = sorted(set(games["home_team"]) | set(games["away_team"]))
    out = []
    for (season, week), wk in games[games["season"] >= first_season].groupby(["season", "week"]):
        t0 = float(wk["t"].min())
        ratings, hfa = fit_ratings(games, teams, t0, target=target, **fit_kw)
        ranks = rank_teams(ratings)
        for r in wk.itertuples(index=False):
            neutral = getattr(r, "location", "") == "Neutral"
            out.append({"game_id": r.game_id, "season": season, "week": week, "home": r.home_team, "away": r.away_team,
                        "pred_margin": ratings[r.home_team] - ratings[r.away_team] + (0.0 if neutral else hfa),
                        "rank_margin": rank_margin(ranks[r.home_team], ranks[r.away_team], hfa, neutral),
                        "home_rank": ranks[r.home_team], "away_rank": ranks[r.away_team],
                        "actual_margin": r.home_score - r.away_score,
                        "market_margin": r.spread_line if pd.notna(r.spread_line) else np.nan})
    return pd.DataFrame(out)


def score(pred: pd.DataFrame, col="pred_margin") -> dict:
    """MAE and how much of the miss vs the market the rating explains."""
    d = pred.dropna(subset=["actual_margin"])
    res = {"n": int(len(d)), "mae": float((d["actual_margin"] - d[col]).abs().mean()),
           "rmse": float(np.sqrt(((d["actual_margin"] - d[col]) ** 2).mean()))}
    m = d.dropna(subset=["market_margin"])
    if len(m):
        res["market_mae"] = float((m["actual_margin"] - m["market_margin"]).abs().mean())
        res["market_rmse"] = float(np.sqrt(((m["actual_margin"] - m["market_margin"]) ** 2).mean()))
        res["corr_with_market"] = float(m[col].corr(m["market_margin"]))
        # does the rating know anything the market doesn't? corr of (rating - market) with (actual - market)
        res["corr_edge_vs_miss"] = float((m[col] - m["market_margin"]).corr(m["actual_margin"] - m["market_margin"]))
    return res


# ---- the weekly table + per-game projection ---------------------------------------
def week_t(games: pd.DataFrame, season: int, week: int) -> float:
    """The time index (see build_games) of a given week."""
    return float((season - games["season"].min()) * WEEKS_PER_SEASON + week)


def cover_prob(margin: float, home_spread: float, sd=GAME_MARGIN_SD) -> float:
    """P(home covers) if the home margin is Normal(margin, sd). Home covers iff margin > -home_spread."""
    from math import erf, sqrt
    z = (margin - (-home_spread)) / sd
    return 0.5 * (1 + erf(z / sqrt(2)))


def power_table(games: pd.DataFrame, season: int, week=None, target="points") -> dict:
    """
    The weekly power rankings from every game before (season, week).
    -> {"ratings", "ranks", "hfa", "table": [row per team, best first], "prev_ranks"}.
    Each row: rank, last-week rank and change, rating (points vs an average team), record, point
    differential per game, strength of schedule (the average rating of opponents already played
    this season, ranked), and EPA/play on offense and defense (ranked) -- all through last week.
    """
    teams = sorted(set(games["home_team"]) | set(games["away_team"]))
    # week=None: "as of right now" -- after the latest completed game, whatever week that was
    t_now = week_t(games, season, week) if week is not None else float(games["t"].max()) + 0.5
    ratings, hfa = fit_ratings(games, teams, t_now, target=target)
    ranks = rank_teams(ratings)
    this = games[(games["season"] == season) & (games["t"] < t_now)]
    prev_t = float(this["t"].max()) if not this.empty else None          # the most recent completed week
    prev_ranks = rank_teams(fit_ratings(games, teams, prev_t, target=target)[0]) if prev_t is not None else {}

    rows = []
    for team in teams:
        h = this[this["home_team"] == team]
        a = this[this["away_team"] == team]
        margins = list(h["home_score"] - h["away_score"]) + list(a["away_score"] - a["home_score"])
        opps = list(h["away_team"]) + list(a["home_team"])
        w, l, tie = sum(m > 0 for m in margins), sum(m < 0 for m in margins), sum(m == 0 for m in margins)
        off = list(h["epa_pp_h"]) + list(a["epa_pp_a"])
        dfn = list(h["epa_pp_a"]) + list(a["epa_pp_h"])
        rows.append({"team": team, "rank": ranks[team], "prev_rank": prev_ranks.get(team),
                     "change": None if team not in prev_ranks else prev_ranks[team] - ranks[team],
                     "rating": round(ratings[team], 2), "record": f"{w}-{l}" + (f"-{tie}" if tie else ""),
                     "games": len(margins), "pd_per_game": round(float(np.mean(margins)), 1) if margins else None,
                     "sos": round(float(np.mean([ratings[o] for o in opps])), 2) if opps else None,
                     "off_epa": round(float(np.nanmean(off)), 3) if off else None,
                     "def_epa": round(float(np.nanmean(dfn)), 3) if dfn else None})
    for key, name, reverse in (("sos", "sos_rank", True), ("off_epa", "off_rank", True), ("def_epa", "def_rank", False)):
        vals = sorted([r for r in rows if r[key] is not None], key=lambda r: r[key], reverse=reverse)
        for i, r in enumerate(vals):
            r[name] = i + 1
    rows.sort(key=lambda r: r["rank"])
    return {"ratings": ratings, "ranks": ranks, "hfa": hfa, "table": rows, "prev_ranks": prev_ranks}


def project_game(pt: dict, home: str, away: str, home_spread=None, qb_out_home=False, qb_out_away=False,
                 neutral=False) -> dict:
    """
    The ranking-based view of one game. `rank_margin` follows the user's rank -> points rule;
    `rating_margin` uses the fitted rating points directly (shown for comparison). A missing
    regular quarterback shifts both by QB_CHANGE_POINTS. `rank_spread` is in the odds convention
    (negative = home favored), rounded to the half point.
    """
    hr, ar = pt["ranks"][home], pt["ranks"][away]
    qb = QB_CHANGE_POINTS * (int(bool(qb_out_away)) - int(bool(qb_out_home)))
    margin = rank_margin(hr, ar, pt["hfa"], neutral) + qb
    rating_margin = pt["ratings"][home] - pt["ratings"][away] + (0.0 if neutral else pt["hfa"]) + qb
    out = {"home_rank": hr, "away_rank": ar, "rank_margin": round(margin, 2),
           "rank_spread": round(-margin * 2) / 2, "rating_margin": round(rating_margin, 2), "qb_adjust": qb}
    if home_spread is not None:
        out["rank_gap"] = round(out["rank_spread"] - home_spread, 2)           # negative: rank model likes home more
        out["rank_home_cover"] = round(cover_prob(margin, home_spread), 4)
    return out


# ---- weekly history (the backlog) -----------------------------------------------------
POWER_HISTORY_FILE = pathlib.Path(__file__).parent / "tracking" / "power_history.csv"
POWER_COLUMNS = ["season", "through_week", "computed_at", "team", "rank", "prev_rank", "rating", "record",
                 "pd_per_game", "sos", "off_epa", "def_epa", "hfa"]
POWER_KEY = ["season", "through_week", "team"]


def power_rows(pt: dict, season: int, through_week: int, computed_at: str) -> list:
    """The power table as history rows: the rankings through `through_week` of `season`."""
    return [{"season": season, "through_week": through_week, "computed_at": computed_at, "team": r["team"],
             "rank": r["rank"], "prev_rank": r["prev_rank"], "rating": r["rating"], "record": r["record"],
             "pd_per_game": r["pd_per_game"], "sos": r["sos"], "off_epa": r["off_epa"], "def_epa": r["def_epa"],
             "hfa": round(pt["hfa"], 2)} for r in pt["table"]]


def load_power_history(path=None) -> pd.DataFrame:
    path = pathlib.Path(path or POWER_HISTORY_FILE)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=POWER_COLUMNS)
    return pd.read_csv(path)


def upsert_power_history(rows: list, path=None) -> int:
    """
    Keep one row per (season, through_week, team): a re-run of the same week replaces its rows (the
    rankings can shift slightly if a late stat correction lands), everything else is untouched.
    """
    if not rows:
        return 0
    path = pathlib.Path(path or POWER_HISTORY_FILE)
    path.parent.mkdir(exist_ok=True)
    new = pd.DataFrame(rows, columns=POWER_COLUMNS)
    old = load_power_history(path)
    both = pd.concat([old, new], ignore_index=True).drop_duplicates(subset=POWER_KEY, keep="last")
    both = both.sort_values(["season", "through_week", "rank"]).reset_index(drop=True)
    both.to_csv(path, index=False)
    return len(new)


def snapshot_power_history(games: pd.DataFrame, season: int, computed_at: str, path=None) -> int:
    """Store the rankings as of right now (through the latest completed week of `season`)."""
    done = games[games["season"] == season]
    if done.empty:
        return 0
    through = int(done["week"].max())
    return upsert_power_history(power_rows(power_table(games, season, None), season, through, computed_at), path)


def backfill_power_history(games: pd.DataFrame, first_season: int, computed_at: str, path=None) -> int:
    """
    Rebuild the rankings as they stood after every completed week since `first_season`. Each week
    uses only the games before it, exactly as a live run would have, so the backfill is honest.
    """
    n = 0
    for season in sorted(games["season"].unique()):
        if season < first_season:
            continue
        for week in sorted(games[games["season"] == season]["week"].unique()):
            pt = power_table(games, int(season), int(week) + 1)
            n += upsert_power_history(power_rows(pt, int(season), int(week), computed_at), path)
    return n


# ---- do the article-inspired context features add anything? ---------------------------
CONTEXT_FEATURES = ["dqb", "rest_diff", "thursday", "dome", "windy", "cold", "turf"]


def context_table(games: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """
    Walk-forward predictions + game context: did the home/away team start someone other than its
    usual QB (the most common starter of its previous 6 games), rest-day gap, Thursday, dome, wind,
    cold, artificial turf. `dqb` > 0 means the AWAY team lacks its usual QB (good for the home side).
    """
    long = []
    for r in games.itertuples(index=False):
        long.append((r.t, r.game_id, r.home_team, r.home_qb_id))
        long.append((r.t, r.game_id, r.away_team, r.away_qb_id))
    L = pd.DataFrame(long, columns=["t", "game_id", "team", "qb"]).sort_values("t")
    changed = {}
    for team, d in L.groupby("team"):
        d = d.reset_index(drop=True)
        for i in range(len(d)):
            prev = d["qb"].iloc[max(0, i - 6):i].dropna()
            usual = prev.mode().iloc[0] if len(prev) >= 3 else None
            changed[(d["game_id"][i], team)] = None if usual is None else int(d["qb"][i] != usual)
    m = pred.copy()
    m["chg_h"] = [changed.get((r.game_id, r.home)) for r in m.itertuples()]
    m["chg_a"] = [changed.get((r.game_id, r.away)) for r in m.itertuples()]
    m = m.dropna(subset=["chg_h", "chg_a"]).merge(
        games[["game_id", "home_rest", "away_rest", "weekday", "roof", "surface", "temp", "wind"]], on="game_id")
    m["dqb"] = m["chg_a"].astype(float) - m["chg_h"].astype(float)
    m["rest_diff"] = (m["home_rest"] - m["away_rest"]).clip(-7, 7)
    m["thursday"] = (m["weekday"] == "Thursday").astype(float)
    m["dome"] = m["roof"].isin(["dome", "closed"]).astype(float)
    m["windy"] = (m["wind"].fillna(0) >= 15).astype(float)
    m["cold"] = (m["temp"].fillna(70) <= 35).astype(float)
    m["turf"] = (~m["surface"].fillna("grass").str.contains("grass", case=False)).astype(float)
    return m


def context_effects(m: pd.DataFrame, target: str) -> pd.DataFrame:
    """OLS of a residual (actual - rating margin, or actual - market margin) on the context features."""
    d = m.dropna(subset=CONTEXT_FEATURES + [target])
    X = np.c_[d[CONTEXT_FEATURES].to_numpy(float), np.ones(len(d))]
    y = d[target].to_numpy(float)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ beta
    se = np.sqrt(np.diag(res @ res / (len(d) - X.shape[1]) * np.linalg.inv(X.T @ X)))
    return pd.DataFrame({"coef": beta[:-1], "se": se[:-1], "t": beta[:-1] / se[:-1]}, index=CONTEXT_FEATURES)


# ---- CLI: backtest report ----------------------------------------------------------
def main():
    import argparse
    import json
    import pathlib
    import nflreadpy as nfl

    ap = argparse.ArgumentParser(description="Walk-forward test of the power-rating system.")
    ap.add_argument("--first", type=int, default=2018)
    ap.add_argument("--json", action="store_true", help="write web/rating_backtest.json")
    ap.add_argument("--backfill-power", action="store_true",
                    help="rebuild tracking/power_history.csv for every completed week since 2021 and exit")
    args = ap.parse_args()
    season = nfl.get_current_season()
    tg = ts.team_game_stats(ts.load_pbp(range(args.first, season + 1)))
    sched = nfl.load_schedules(list(range(args.first, season + 1))).to_pandas()
    g = build_games(tg, sched[sched["game_type"] == "REG"])
    if args.backfill_power:
        import datetime
        stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        n = backfill_power_history(g, 2021, stamp)
        print(f"power history: wrote {n} rows -> {POWER_HISTORY_FILE}")
        return

    print("Which performance measure builds the best ratings? (each week predicted from earlier games only)")
    print(f"  {'target':24s} {'MAE':>6s} {'rank-rule MAE':>14s}   market MAE")
    rows = {}
    for name in TARGETS:
        p = walk_forward(g, name)
        a, b = score(p), score(p, "rank_margin")
        rows[name] = {"mae": round(a["mae"], 3), "rank_mae": round(b["mae"], 3),
                      "corr_edge_vs_miss": round(a["corr_edge_vs_miss"], 3)}
        print(f"  {name:24s} {a['mae']:6.2f} {b['mae']:14.2f}   {a['market_mae']:.2f}")
    p = walk_forward(g, "points")
    sc, sr = score(p), score(p, "rank_margin")
    x, y = (p["away_rank"] - p["home_rank"]).to_numpy(float), p["actual_margin"].to_numpy(float)
    slope, icpt = np.linalg.lstsq(np.c_[x, np.ones(len(x))], y, rcond=None)[0]
    print(f"\nChosen: points target, half-life {RATING_HALFLIFE_WEEKS:g} weeks, ridge {RATING_LAMBDA:g}.")
    print(f"  n={sc['n']} games 2021+: rating MAE {sc['mae']:.2f}, rank-rule MAE {sr['mae']:.2f}, market MAE {sc['market_mae']:.2f}")
    print(f"  fitted points per rank step {slope:.3f} (your 6-point rule: {PER_RANK:.3f}); home field {icpt:.2f}")
    print(f"  edge vs market: corr(rating - market, actual - market) = {sc['corr_edge_vs_miss']:+.3f}  (0 = no information beyond the line)")
    m = context_table(g, p)
    m["res_rating"] = m["actual_margin"] - m["pred_margin"]
    m["res_market"] = m["actual_margin"] - m["market_margin"]
    print(f"\nContext features (n={len(m)}): effect in points on the home margin, t-stat in brackets")
    print(f"  {'':10s} {'vs our rating':>16s} {'vs the market':>16s}")
    ctx_r, ctx_m = context_effects(m, "res_rating"), context_effects(m, "res_market")
    for f in CONTEXT_FEATURES:
        print(f"  {f:10s} {ctx_r.at[f, 'coef']:+8.2f} [{ctx_r.at[f, 't']:+5.1f}] {ctx_m.at[f, 'coef']:+8.2f} [{ctx_m.at[f, 't']:+5.1f}]")
    print("  (a feature that matters to our rating but not to the market is something the market already prices in)")
    if args.json:
        out = pathlib.Path(__file__).parent / "web" / "rating_backtest.json"
        out.write_text(json.dumps({
            "n": sc["n"], "seasons": f"2021-{season}", "mae": round(sc["mae"], 2), "rank_rule_mae": round(sr["mae"], 2),
            "market_mae": round(sc["market_mae"], 2), "fitted_points_per_rank": round(float(slope), 3),
            "rule_points_per_rank": round(PER_RANK, 3), "home_field": round(float(icpt), 2),
            "corr_edge_vs_miss": round(sc["corr_edge_vs_miss"], 3), "targets": rows,
            "qb_change_points": QB_CHANGE_POINTS, "halflife_weeks": RATING_HALFLIFE_WEEKS,
            "context_vs_rating_t": {f: round(float(ctx_r.at[f, "t"]), 2) for f in CONTEXT_FEATURES},
            "context_vs_market_t": {f: round(float(ctx_m.at[f, "t"]), 2) for f in CONTEXT_FEATURES}}, indent=1), encoding="utf-8")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
