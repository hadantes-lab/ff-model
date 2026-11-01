"""
dvp_study.py
============
Does defense-vs-position explain what players do beyond the projection we already make?

For every player-game in the test seasons it takes the walk-forward projection from backtest_props (recency-weighted average times the
existing opponent multiplier, which is itself a cruder version of DvP), then asks whether the four DvP readings in dvp_signal.py
(season so far, last four games, last season, schedule-adjusted) explain the part of the result the projection missed.

  miss = actual - projection          x = projection * (reading - 1)          miss ~ c * x

c = 1 means the reading is worth exactly its face value, c = 0 means nothing. Standard errors are clustered by game. The honest test is
out of sample: coefficients fitted on earlier seasons applied to the next, measured by mean absolute error against the projection alone.

    python dvp_study.py                      # ~10 minutes, caches to the temp folder
    python dvp_study.py --seasons 2019 2020 2021 2022 2023 2024 2025 --refresh
"""

import argparse
import pathlib
import sys
import tempfile

import numpy as np
import pandas as pd

import backtest_props as bt
import context_data as cd
import dvp_signal as ds
import props_model as pm

CACHE = pathlib.Path(tempfile.gettempdir()) / "ff_dvp_study_cache.pkl"
STUDY = ["player_pass_yds", "player_pass_attempts", "player_pass_completions", "player_pass_tds", "player_rush_attempts",
         "player_rush_yds", "player_targets", "player_receptions", "player_reception_yds", "fp_dk", "fp_ppr"]
FANTASY = {"fp_dk": pm.Spec("DK points", ("dk",), "yards"), "fp_ppr": pm.Spec("PPR points", ("ppr",), "yards")}
FLOORS = {"fp_dk": 6.0, "fp_ppr": 6.0}
SHRINK_FIXED = {"l4": 4, "prev": 17}


def study_markets():
    allm = {**pm.ALL_MARKETS, **FANTASY}
    return {m: allm[m] for m in STUDY}


def build(seasons):
    """Player-game records with the existing projection and every DvP reading attached."""
    h = bt.load(seasons)
    h[["dk", "fd", "ppr"]] = cd.fantasy_points(h)
    markets = study_markets()
    pm.ROLE_FLOOR.update(FLOORS)
    bt.MARKETS = markets
    weeks = sorted({(int(a), int(b)) for a, b in h.loc[h["season"].isin(seasons) & (h["week"] >= 4), ["season", "week"]].to_numpy()})
    opp_lookup = {}
    for i, (s, w) in enumerate(weeks):
        prior = h[(h["season"] * 100 + h["week"]) < (s * 100 + w)]
        opp_lookup[(s, w)] = {m: pm.opponent_multipliers(prior, sp) for m, sp in markets.items()}
        print(f"  opponent tables {i + 1}/{len(weeks)}", end="\r", file=sys.stderr)
    df = bt.collect(h, seasons, opp_lookup)
    df["grp"] = df["pos"].map(ds.GROUP_OF)
    df = df[df["grp"].notna()]

    parts = []
    for m, spec in markets.items():
        gt = ds.with_expectation(ds.game_table(h, pm.stat_series(h, spec)))
        rd = []
        for s, w in weeks:
            r = ds.readings_asof(gt, s, w).reset_index().assign(season=s, week=w, market=m)
            rd.append(r)
        parts.append(pd.concat(rd, ignore_index=True))
    rdg = pd.concat(parts, ignore_index=True).rename(columns={"def": "opp_team"})
    df = df.merge(rdg, on=["market", "season", "week", "grp", "opp_team"], how="left")
    return df


def design(df):
    """Shrunk readings turned into the regressors x = projection * (reading - 1)."""
    p0 = df["mu"] * df["opp"]
    n = df["n_sea"].fillna(0)
    cols = {"opp": df["mu"] * (df["opp"] - 1)}
    for r in ds.READINGS:
        ratio = df[r]
        k = n if r in ("sea", "adj") else pd.Series(SHRINK_FIXED[r], index=df.index)
        cols[r] = df["mu"] * (ds.shrunk(ratio, k) - 1)
    return p0, pd.DataFrame(cols).fillna(0.0)


def clustered(y, X, cl):
    """OLS without intercept, standard errors clustered by `cl`. -> (coef, se)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    xtx_inv = np.linalg.pinv(X.T @ X)
    b = xtx_inv @ X.T @ y
    e = y - X @ b
    meat = np.zeros((X.shape[1],) * 2)
    for _, idx in pd.Series(np.arange(len(y))).groupby(np.asarray(cl)):
        s = X[idx.to_numpy()].T @ e[idx.to_numpy()]
        meat += np.outer(s, s)
    v = xtx_inv @ meat @ xtx_inv
    return b, np.sqrt(np.clip(np.diag(v), 0, None))


def report(df):
    d = df.dropna(subset=["adj", "sea"]).copy()
    d["cl"] = d["season"].astype(str) + "-" + d["week"].astype(str) + "-" + d["opp_team"].astype(str)
    p0, X = design(d)
    print(f"\n{len(d):,} player-games, {d['season'].min()}-{d['season'].max()}\n")
    print("Does each reading explain the miss?  coefficient (t-stat); 1.0 = worth its face value, 0 = nothing")
    print("(first column: the opponent multiplier already in the projection; the rest are each tried ALONE on top of it)")
    names = ["opp", *ds.READINGS]
    print(f"{'market':26s} {'n':>7s} " + " ".join(f"{n:>13s}" for n in names))
    rows = []
    for m, g in list(d.groupby("market")) + [("ALL", d)]:
        ix = d.index.get_indexer(g.index)
        cells, res = [], {}
        for r in names:
            cols = ["opp"] if r == "opp" else ["opp", r]
            b, se = clustered(g["actual"] - g["mu"], X.iloc[ix][cols], d["cl"].iloc[ix])
            c, t = b[-1], b[-1] / se[-1] if se[-1] > 0 else 0
            res[r] = (c, t)
            cells.append(f"{c:+6.2f} ({t:+5.1f})")
        rows.append((m, len(g), res))
        print(f"{m:26s} {len(g):7d} " + " ".join(f"{c:>13s}" for c in cells))
    return d, p0, X


def out_of_sample(d, p0, X, readings=("sea", "l4", "prev", "adj")):
    """Fit coefficients on earlier seasons, apply to the next; compare mean absolute error with the projection as it is today."""
    print("\nOUT OF SAMPLE: mean absolute error vs the current projection (negative = better)")
    seasons = sorted(d["season"].unique())
    if len(seasons) < 2:
        print("  need two seasons")
        return
    cols = ["opp", *readings]
    out = {}
    for m, g in list(d.groupby("market")) + [("ALL", d)]:
        ae0, ae1 = [], []
        for s in seasons[1:]:
            tr, te = g[g["season"] < s], g[g["season"] == s]
            if len(tr) < 400 or not len(te):
                continue
            b, _ = clustered(tr["actual"] - tr["mu"], X.loc[tr.index, cols], tr["cl"])
            base = te["mu"] + X.loc[te.index, "opp"]                       # = mu * opp, the projection today
            new = te["mu"] + X.loc[te.index, cols].to_numpy() @ b          # opp re-weighted, DvP readings added
            ae0.append((base - te["actual"]).abs()), ae1.append((new - te["actual"]).abs())
        if ae0:
            a0, a1 = pd.concat(ae0), pd.concat(ae1)
            out[m] = (len(a0), a0.mean(), a1.mean())
    for m, (n, a0, a1) in out.items():
        print(f"  {m:26s} n={n:7d}  today {a0:6.3f}  with DvP {a1:6.3f}  ({(a1 / a0 - 1) * 100:+5.2f}%)")


def quintiles(d, p0, X):
    """The plain-language version: bucket by combined DvP score and see what players did against projection."""
    comb = X[["sea", "l4", "prev", "adj"]].sum(axis=1) / d["mu"]
    d = d.assign(score=comb, rel=(d["actual"] - p0) / p0.clip(lower=0.1))
    print("\nPlayers vs the CURRENT projection, bucketed by the defense's combined DvP score (sum of the four readings' excess)")
    for m, g in list(d.groupby("market")) + [("ALL", d)]:
        g = g.assign(q=pd.qcut(g["score"].rank(method="first"), 5, labels=False))
        r = g.groupby("q").apply(lambda x: (x["actual"].sum() / (x["mu"] * x["opp"]).sum() - 1) * 100)
        print(f"  {m:26s} " + "  ".join(f"{v:+5.1f}%" for v in r.to_numpy()) + "   (softest defenses last)")


WEIGHTS_FILE = pathlib.Path(__file__).parent / "dvp_weights.json"
WEIGHT_SHRINK = 0.8          # winner's curse: the fitted coefficient is believed at 80%
WEIGHT_CAP = 0.6
MIN_GAIN = 0.001             # a market is switched on only if the out-of-sample squared error falls by at least 0.1%
MIN_T = 2.0


def combined(X):
    """The single factor production uses: the mean excess of the season-so-far, last-four and schedule-adjusted readings (last season's left out)."""
    return (X["sea"] + X["l4"] + X["adj"]) / 3


def fit_weights(d, X):
    """
    -> {market: {"weight", "coef", "t", "oos_mse_change", "n", "on"}}. One coefficient c per market: miss of today's projection ~ c * mu * (combined - 1),
    fitted on every season, then judged out of sample (coefficient from earlier seasons applied to the next). A market is on only if it
    shows a positive coefficient with |t| >= MIN_T and the out-of-sample squared error falls by MIN_GAIN.
    """
    X = X.assign(comb=combined(X))
    out = {}
    for m, g in d.groupby("market"):
        y = g["actual"] - g["mu"] * g["opp"]
        b, se = clustered(y, X.loc[g.index, ["comb"]], g["cl"])
        e0, e1 = [], []
        for s in sorted(d["season"].unique())[1:]:
            tr, te = g[g["season"] < s], g[g["season"] == s]
            if len(tr) < 400 or not len(te):
                continue
            bt_, _ = clustered(tr["actual"] - tr["mu"] * tr["opp"], X.loc[tr.index, ["comb"]], tr["cl"])
            base = te["mu"] * te["opp"]
            e0.append(base - te["actual"]), e1.append(base + X.loc[te.index, "comb"] * bt_[0] - te["actual"])
        chg = float((pd.concat(e1) ** 2).mean() / (pd.concat(e0) ** 2).mean() - 1) if e0 else 0.0
        t = float(b[0] / se[0]) if se[0] > 0 else 0.0
        on = b[0] > 0 and t >= MIN_T and chg <= -MIN_GAIN
        out[m] = {"weight": round(float(min(b[0] * WEIGHT_SHRINK, WEIGHT_CAP)), 3) if on else 0.0, "coef": round(float(b[0]), 3),
                  "t": round(t, 2), "oos_mse_change": round(chg, 5), "n": int(len(g)), "on": bool(on)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", default=[2020, 2021, 2022, 2023, 2024, 2025])
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--write", action="store_true", help="fit and save dvp_weights.json (the markets DvP is applied to, and how hard)")
    args = ap.parse_args()
    if CACHE.exists() and not args.refresh:
        df = pd.read_pickle(CACHE)
        print("using cache", CACHE, file=sys.stderr)
    else:
        df = build(args.seasons)
        df.to_pickle(CACHE)
    d, p0, X = report(df)
    out_of_sample(d, p0, X)
    quintiles(d, p0, X)
    w = fit_weights(d, X)
    print("\nWEIGHTS (market: weight applied | pooled coefficient, t, out-of-sample squared-error change)")
    for m, v in w.items():
        print(f"  {m:26s} {'ON ' if v['on'] else 'off'} {v['weight']:.2f} | coef {v['coef']:+.2f}  t {v['t']:+.1f}  oos {v['oos_mse_change'] * 100:+.2f}%  n={v['n']}")
    if args.write:
        import json
        WEIGHTS_FILE.write_text(json.dumps({"seasons": [int(args.seasons[0]), int(args.seasons[-1])], "shrink_games": 6,
                                            "weights": {m: v["weight"] for m, v in w.items()}, "evidence": w}, indent=1), encoding="utf-8")
        print("wrote", WEIGHTS_FILE)


if __name__ == "__main__":
    main()
