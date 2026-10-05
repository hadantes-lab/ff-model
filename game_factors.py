"""
game_factors.py
===============
Which team stats relate to a game's total (over/under) and side (spread) -- and, more to the
point, which ones the posted line does NOT already account for.

THE QUESTION THAT MATTERS
-------------------------
Pace and efficiency stats correlate with total points almost by construction (more plays at more
yards per play is more points). That is not a finding. The posted line already knows it. What would
be useful is a stat that predicts how far the game lands from the line:

    residual = actual total - posted total          (and margin - spread for sides)

so every feature is tested three ways against:
    actual total   - does it relate to scoring at all (expected: yes)
    posted total   - has the market already priced it (expected: mostly yes)
    residual       - does it predict what the market missed (the only one that pays)

Every feature is PRE-GAME: a recency-weighted average of the teams' prior games only (see
team_stats.rolling_features), so a game's own result never leaks into its features.

OUT-OF-SAMPLE TEST
------------------
With ~14 features, a few will look "significant" by luck alone, so in-sample correlations are
reported with that caveat and the real test is walk-forward: fit a regularized model on earlier
seasons, predict the residual for the next, and score it on games it never saw -- R^2 against
just predicting the average residual, and how often the lean toward over/under was right
(break-even at standard -110 pricing is 52.4%).

    python game_factors.py                 # prints the report
    python game_factors.py --json          # also writes web/game_factors.json
"""

import argparse
import json
import math
import pathlib
import sys

import numpy as np
import pandas as pd

import team_stats as ts

MIN_PRIOR_GAMES = 3          # both teams need this many prior games for a game to be used
BREAKEVEN = 0.5238           # win rate needed at -110
OUT = pathlib.Path(__file__).parent / "web" / "game_factors.json"

# game-level features for TOTALS (combined across both teams) and their plain-English labels
TOTAL_FEATURES = {
    "plays_sum": "Combined plays per game",
    "sec_per_play": "Seconds per play (avg; lower = faster pace)",
    "yards_sum": "Combined yards per game",
    "ypp_off": "Yards per play, offense (avg)",
    "ypp_def": "Yards per play allowed, defense (avg)",
    "exp_ypp": "Expected yards per play in this matchup (avg)",
    "exp_plays": "Expected plays in this matchup (sum)",
    "exp_yards": "Expected yards in this matchup (sum)",
    "first_downs_sum": "Combined first downs per game",
    "third_rate": "Third-down conversion rate (avg)",
    "third_allowed": "Third-down rate allowed (avg)",
    "neutral_pass_rate": "Neutral-state pass rate (avg)",
    "xpass": "Expected pass rate, neutral state (avg)",
    "proe": "Pass rate over expected, neutral state (avg)",
}
# game-level features for SIDES (home minus away)
SIDE_FEATURES = {
    "yards_diff": "Yards per game, home - away",
    "ypp_diff": "Yards per play, home - away",
    "exp_ypp_diff": "Expected yards per play in this matchup, home - away",
    "exp_yards_diff": "Expected yards in this matchup, home - away",
    "first_downs_diff": "First downs per game, home - away",
    "third_diff": "Third-down rate, home - away",
    "top_diff": "Time of possession (sec), home - away",
    "proe_diff": "Pass rate over expected, home - away",
}


# ---- data --------------------------------------------------------------------
def load_schedule(seasons) -> pd.DataFrame:
    import nflreadpy as nfl

    s = nfl.load_schedules(list(seasons)).to_pandas()
    s = s[s["game_type"] == "REG"] if "game_type" in s.columns else s
    cols = ["game_id", "season", "week", "home_team", "away_team", "home_score", "away_score",
            "total_line", "spread_line"]
    return s[cols].copy()


def build_game_table(tg: pd.DataFrame, sched: pd.DataFrame, min_prior=MIN_PRIOR_GAMES) -> pd.DataFrame:
    """One row per game: pre-game features for both teams, the posted lines, and the result."""
    roll = ts.rolling_features(tg)
    roll = roll[roll["r_n"] >= min_prior]
    h = roll.rename(columns=lambda c: c if c in ("game_id",) else f"{c}_h")
    a = roll.rename(columns=lambda c: c if c in ("game_id",) else f"{c}_a")
    g = sched.merge(h, left_on=["game_id", "home_team"], right_on=["game_id", "team_h"]).drop(columns="team_h")
    g = g.merge(a, left_on=["game_id", "away_team"], right_on=["game_id", "team_a"]).drop(columns="team_a")

    lg_ypp = float(tg["ypp"].mean())
    mean = lambda x, y: (g[x] + g[y]) / 2

    # expected plays/ypp/yards for each side of THIS matchup (see team_stats.matchup_expectation)
    exp_plays_h = (g["r_plays_h"] + g["r_plays_allowed_a"]) / 2
    exp_plays_a = (g["r_plays_a"] + g["r_plays_allowed_h"]) / 2
    exp_ypp_h = g["r_ypp_h"] + (g["r_ypp_allowed_a"] - lg_ypp)
    exp_ypp_a = g["r_ypp_a"] + (g["r_ypp_allowed_h"] - lg_ypp)

    f = pd.DataFrame({"game_id": g["game_id"], "season": g["season"], "week": g["week"]})
    f["plays_sum"] = g["r_plays_h"] + g["r_plays_a"]
    f["sec_per_play"] = mean("r_sec_per_play_h", "r_sec_per_play_a")
    f["yards_sum"] = g["r_yards_h"] + g["r_yards_a"]
    f["ypp_off"] = mean("r_ypp_h", "r_ypp_a")
    f["ypp_def"] = mean("r_ypp_allowed_h", "r_ypp_allowed_a")
    f["exp_ypp"] = (exp_ypp_h + exp_ypp_a) / 2
    f["exp_plays"] = exp_plays_h + exp_plays_a
    f["exp_yards"] = exp_plays_h * exp_ypp_h + exp_plays_a * exp_ypp_a
    f["first_downs_sum"] = g["r_first_downs_h"] + g["r_first_downs_a"]
    f["third_rate"] = mean("r_third_rate_h", "r_third_rate_a")
    f["third_allowed"] = mean("r_third_rate_allowed_h", "r_third_rate_allowed_a")
    f["neutral_pass_rate"] = mean("r_neutral_pass_rate_h", "r_neutral_pass_rate_a")
    f["xpass"] = mean("r_neutral_xpass_h", "r_neutral_xpass_a")
    f["proe"] = mean("r_proe_h", "r_proe_a")

    f["yards_diff"] = g["r_yards_h"] - g["r_yards_a"]
    f["ypp_diff"] = g["r_ypp_h"] - g["r_ypp_a"]
    f["exp_ypp_diff"] = exp_ypp_h - exp_ypp_a
    f["exp_yards_diff"] = exp_plays_h * exp_ypp_h - exp_plays_a * exp_ypp_a
    f["first_downs_diff"] = g["r_first_downs_h"] - g["r_first_downs_a"]
    f["third_diff"] = g["r_third_rate_h"] - g["r_third_rate_a"]
    f["top_diff"] = g["r_top_sec_h"] - g["r_top_sec_a"]
    f["proe_diff"] = g["r_proe_h"] - g["r_proe_a"]

    f["total_line"], f["spread_line"] = g["total_line"], g["spread_line"]
    done = g["home_score"].notna() & g["total_line"].notna() & g["spread_line"].notna()
    f["total_actual"] = np.where(done, g["home_score"] + g["away_score"], np.nan)
    f["margin_actual"] = np.where(done, g["home_score"] - g["away_score"], np.nan)
    f["resid_total"] = f["total_actual"] - f["total_line"]
    f["resid_margin"] = f["margin_actual"] - f["spread_line"]          # spread_line > 0: home favored
    return f.sort_values(["season", "week"]).reset_index(drop=True)


# ---- correlations ------------------------------------------------------------
def corr_with_p(x: pd.Series, y: pd.Series):
    """Pearson r, n, and a two-sided p-value (normal approximation to the t, fine at n in the hundreds)."""
    d = pd.concat([x, y], axis=1).dropna()
    n = len(d)
    if n < 10 or d.iloc[:, 0].std() == 0 or d.iloc[:, 1].std() == 0:
        return float("nan"), n, float("nan")
    r = float(d.iloc[:, 0].corr(d.iloc[:, 1]))
    t = r * math.sqrt((n - 2) / max(1e-12, 1 - r * r))
    return r, n, math.erfc(abs(t) / math.sqrt(2))


def correlation_table(gt: pd.DataFrame, features: dict, targets: dict) -> pd.DataFrame:
    """Rows = features, columns = r (and p) against each target in `targets` {label: column}."""
    rows = []
    for f, label in features.items():
        row = {"feature": f, "label": label}
        for tname, col in targets.items():
            r, n, p = corr_with_p(gt[f], gt[col])
            row[f"r_{tname}"], row[f"p_{tname}"], row["n"] = r, p, n
        rows.append(row)
    return pd.DataFrame(rows)


def top_vs_plays(tg: pd.DataFrame) -> dict:
    """
    Time of possession vs. plays, same game. Naive correlation is high, but TOP is partly just
    plays x seconds-per-play and partly a symptom of efficiency (good offenses sustain drives), so
    the partial correlation holding yards-per-play and third-down rate fixed is reported too.
    """
    d = tg[["plays", "top_sec", "ypp", "third_rate", "sec_per_play"]].dropna()
    naive = float(d["plays"].corr(d["top_sec"]))
    X = np.c_[d["ypp"], d["third_rate"], np.ones(len(d))]
    resid = lambda y: y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    partial = float(np.corrcoef(resid(d["plays"].to_numpy(float)), resid(d["top_sec"].to_numpy(float)))[0, 1])
    # across teams (season-long habits, not single games)
    team = tg.groupby(["season", "team"])[["plays", "top_sec", "sec_per_play"]].mean().dropna()
    return {"n_team_games": int(len(d)), "r_game": naive, "r_game_partial": partial,
            "r_team_season": float(team["plays"].corr(team["top_sec"])),
            "r_spp_plays": float(d["plays"].corr(d["sec_per_play"]))}


# ---- walk-forward out-of-sample test -------------------------------------------
def _standardize(train: np.ndarray, test: np.ndarray):
    mu, sd = np.nanmean(train, axis=0), np.nanstd(train, axis=0)
    sd[sd == 0] = 1.0
    return (train - mu) / sd, (test - mu) / sd


def _ridge_fit(X, y, lam):
    n, p = X.shape
    A = X.T @ X + lam * np.eye(p)
    return np.linalg.solve(A, X.T @ (y - y.mean())), float(y.mean())


def _choose_lambda(X, y, grid=(30, 100, 300, 1000, 3000), folds=5, seed=0):
    """Pick the ridge strength by K-fold CV inside the TRAINING data only."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    best, best_mse = grid[-1], np.inf
    for lam in grid:
        err = 0.0
        for k in range(folds):
            te = idx[k::folds]
            tr = np.setdiff1d(idx, te)
            Xtr, Xte = _standardize(X[tr], X[te])
            w, b = _ridge_fit(Xtr, y[tr], lam)
            err += float(np.sum((Xte @ w + b - y[te]) ** 2))
        if err < best_mse:
            best, best_mse = lam, err
    return best


def wilson(wins: int, n: int, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = wins / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def walk_forward(gt: pd.DataFrame, features: list, target: str, line_col: str, first_test_season: int,
                 extra=("total_line",)) -> dict:
    """
    Train on every earlier season, predict `target` (a residual) for the next, over all test seasons.
    -> per-season and pooled out-of-sample metrics. `extra` columns (e.g. the line itself) are
    added as features, so the model can learn that the market over- or under-shoots at extremes.
    """
    cols = list(features) + [c for c in extra if c in gt.columns]
    d = gt.dropna(subset=cols + [target]).reset_index(drop=True)
    folds, pooled = [], []
    for s in sorted(d["season"].unique()):
        if s < first_test_season:
            continue
        tr, te = d[d["season"] < s], d[d["season"] == s]
        if len(tr) < 200 or len(te) < 20:
            continue
        lam = _choose_lambda(tr[cols].to_numpy(float), tr[target].to_numpy(float))
        Xtr, Xte = _standardize(tr[cols].to_numpy(float), te[cols].to_numpy(float))
        w, b = _ridge_fit(Xtr, tr[target].to_numpy(float), lam)
        pred = Xte @ w + b
        y = te[target].to_numpy(float)
        sse_model = float(np.sum((y - pred) ** 2))
        sse_base = float(np.sum((y - tr[target].mean()) ** 2))
        f = {"season": int(s), "n": int(len(te)), "lambda": lam,
             "r2_oos": 1 - sse_model / max(sse_base, 1e-12),
             "corr": float(np.corrcoef(pred, y)[0, 1]),
             "coefs": dict(zip(cols, [float(x) for x in w]))}
        folds.append(f)
        pooled.append(pd.DataFrame({"pred": pred, "y": y, "line": te[line_col].to_numpy(float), "season": s}))
    if not pooled:
        return {"folds": [], "pooled": None}
    p = pd.concat(pooled, ignore_index=True)
    sse_model = float(np.sum((p["y"] - p["pred"]) ** 2))
    sse_base = float(np.sum((p["y"] - p["y"].mean()) ** 2))
    live = p[p["y"] != 0].copy()          # pushes (residual exactly 0) are not bets
    live["lean_right"] = np.sign(live["pred"]) == np.sign(live["y"])
    top = live[live["pred"].abs() >= live["pred"].abs().quantile(0.8)]
    pooled_out = {
        "n": int(len(p)), "r2_oos": 1 - sse_model / max(sse_base, 1e-12), "corr": float(np.corrcoef(p["pred"], p["y"])[0, 1]),
        "lean_accuracy_all": float(live["lean_right"].mean()), "n_bets_all": int(len(live)),
        "lean_ci_all": wilson(int(live["lean_right"].sum()), len(live)),
        "lean_accuracy_top20": float(top["lean_right"].mean()), "n_bets_top20": int(len(top)),
        "lean_ci_top20": wilson(int(top["lean_right"].sum()), len(top)),
        "mean_abs_pred": float(p["pred"].abs().mean()),
    }
    return {"folds": folds, "pooled": pooled_out}


# ---- report ------------------------------------------------------------------
def build_report(gt: pd.DataFrame, tg: pd.DataFrame, first_test_season=2023) -> dict:
    played = gt.dropna(subset=["total_actual", "total_line"])
    tot = correlation_table(played, TOTAL_FEATURES,
                            {"actual": "total_actual", "line": "total_line", "resid": "resid_total"})
    side = correlation_table(played, SIDE_FEATURES,
                             {"margin": "margin_actual", "spread": "spread_line", "resid": "resid_margin"})
    wf_total = walk_forward(played, list(TOTAL_FEATURES), "resid_total", "total_line", first_test_season)
    wf_side = walk_forward(played, list(SIDE_FEATURES), "resid_margin", "spread_line", first_test_season,
                           extra=("spread_line",))
    line_vs_actual = {
        "n": int(len(played)),
        "r_line_actual": float(played["total_line"].corr(played["total_actual"])),
        "mean_resid_total": float(played["resid_total"].mean()),
        "std_resid_total": float(played["resid_total"].std()),
        "over_rate": float((played["resid_total"] > 0).sum() / max(1, (played["resid_total"] != 0).sum())),
    }
    return {"n_games": int(len(played)), "seasons": sorted(int(s) for s in played["season"].unique()),
            "line_vs_actual": line_vs_actual, "total_table": tot.to_dict("records"),
            "side_table": side.to_dict("records"), "top_vs_plays": top_vs_plays(tg),
            "walk_forward_total": wf_total, "walk_forward_side": wf_side,
            "n_features_tested": len(TOTAL_FEATURES) + len(SIDE_FEATURES)}


def print_report(rep: dict):
    lv = rep["line_vs_actual"]
    print(f"{rep['n_games']} games with posted lines, seasons {rep['seasons'][0]}-{rep['seasons'][-1]}")
    print(f"Posted total vs actual: r = {lv['r_line_actual']:.2f} | mean miss {lv['mean_resid_total']:+.2f} pts "
          f"| typical miss {lv['std_resid_total']:.1f} pts | over rate {lv['over_rate']:.1%}")
    k = rep["n_features_tested"]
    print(f"\nTOTALS: correlation of each pre-game stat with ... (|r| above ~{2 / math.sqrt(rep['n_games']):.3f} is ~2 SE)")
    print(f"  {'feature':44s} {'actual':>7s} {'posted':>7s} {'MISS':>7s} {'p(miss)':>8s}")
    for r in sorted(rep["total_table"], key=lambda r: -abs(r["r_resid"])):
        print(f"  {r['label']:44s} {r['r_actual']:+7.3f} {r['r_line']:+7.3f} {r['r_resid']:+7.3f} {r['p_resid']:8.3f}")
    print(f"\nSIDES: home-minus-away stats vs. ...")
    print(f"  {'feature':44s} {'margin':>7s} {'spread':>7s} {'MISS':>7s} {'p(miss)':>8s}")
    for r in sorted(rep["side_table"], key=lambda r: -abs(r["r_resid"])):
        print(f"  {r['label']:44s} {r['r_margin']:+7.3f} {r['r_spread']:+7.3f} {r['r_resid']:+7.3f} {r['p_resid']:8.3f}")
    print(f"\n{k} stats tested in total, so ~{k * 0.05:.1f} would hit p<0.05 by chance alone.")

    t = rep["top_vs_plays"]
    print(f"\nTIME OF POSSESSION vs PLAYS ({t['n_team_games']} team-games):")
    print(f"  same game r = {t['r_game']:+.2f}; holding yards/play and 3rd-down rate fixed r = {t['r_game_partial']:+.2f}")
    print(f"  across team-seasons r = {t['r_team_season']:+.2f}; seconds-per-play vs plays r = {t['r_spp_plays']:+.2f}")

    for name, wf in (("TOTALS", rep["walk_forward_total"]), ("SIDES", rep["walk_forward_side"])):
        print(f"\nWALK-FORWARD ({name}): fit on earlier seasons, scored on the next, never on games it was fit to")
        for f in wf["folds"]:
            print(f"  {f['season']}: n={f['n']:4d}  out-of-sample R^2 {f['r2_oos']:+.3f}  corr {f['corr']:+.3f}  (ridge lambda {f['lambda']})")
        p = wf["pooled"]
        if p:
            print(f"  pooled: n={p['n']}  R^2 {p['r2_oos']:+.3f}  corr {p['corr']:+.3f}")
            lo, hi = p["lean_ci_all"]
            print(f"  lean correct on {p['lean_accuracy_all']:.1%} of {p['n_bets_all']} games (95% CI {lo:.1%}-{hi:.1%}; break-even {BREAKEVEN:.1%})")
            lo, hi = p["lean_ci_top20"]
            print(f"  strongest 20% of leans: {p['lean_accuracy_top20']:.1%} of {p['n_bets_top20']} (95% CI {lo:.1%}-{hi:.1%})")


def main():
    import nflreadpy as nfl

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--first", type=int, default=2021)
    ap.add_argument("--json", action="store_true", help="also write web/game_factors.json")
    args = ap.parse_args()
    season = nfl.get_current_season()
    tg = ts.team_game_stats(ts.load_pbp(range(args.first, season + 1)))
    gt = build_game_table(tg, load_schedule(range(args.first, season + 1)))
    rep = build_report(gt, tg)
    print_report(rep)
    if args.json:
        OUT.write_text(json.dumps(rep, separators=(",", ":"), default=float), encoding="utf-8")
        print(f"\nWrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
