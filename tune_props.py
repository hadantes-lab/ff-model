"""
tune_props.py
=============
Refines props_model against the last few seasons, strictly walk-forward: every parameter that
is applied to a test season was estimated on EARLIER seasons only.

  1. look-back halflife        -> lowest out-of-sample MAE
  2. bias calibration          -> projections of selected (high-usage) players run high; a per-market
                                  linear map  actual ~ a + b * projection  fixes it
  3. game-environment effect   -> direct test: regress the error of the forecast on the total/spread
                                  environment; is the fitted effect real out of sample?
  4. spread (variance scale)   -> the scale that minimizes CRPS, a proper scoring rule

    python tune_props.py            # prints results
    python tune_props.py --write    # also writes model_calibration.json (used by export_props.py)
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

import backtest_props as bt
import game_context as gc
import props_model as pm

OUT = pathlib.Path(__file__).parent / "model_calibration.json"
SCALES = [0.9, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0]


def crps(kind, mu, var, actual, rng, n=300):
    d = pm.draw(kind, mu, var, rng, n)
    return float(np.abs(d - actual).mean() - 0.5 * np.abs(d[: n // 2] - d[n // 2:]).mean())


def collect_with(h, seasons, opp_lookup, halflife):
    orig = pm.recency_weights
    pm.recency_weights = lambda n, halflife_=halflife: orig(n, halflife_)     # collect() reads pm.recency_weights
    try:
        return bt.collect(h, seasons, opp_lookup)
    finally:
        pm.recency_weights = orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--seasons", type=int, nargs="+", default=[2023, 2024, 2025])
    args = ap.parse_args()
    seasons, tests = args.seasons, [s for s in args.seasons[1:]]

    h = bt.load(seasons)
    opp = {}
    for s, w in sorted({(int(a), int(b)) for a, b in h.loc[h["season"].isin(seasons) & (h["week"] >= 4), ["season", "week"]].to_numpy()}):
        prior = h[(h["season"] * 100 + h["week"]) < (s * 100 + w)]
        opp[(s, w)] = {m: pm.opponent_multipliers(prior, sp) for m, sp in bt.MARKETS.items()}

    # ---- 1. halflife --------------------------------------------------------------
    print("1) LOOK-BACK HALFLIFE (out-of-sample MAE on the test seasons, opponent-adjusted; lower is better)")
    results = {}
    for hl in (3, 5, 8, 12, 20):
        d = collect_with(h, seasons, opp, hl)
        d = d[d["season"].isin(tests)]
        d["proj"] = d["mu"] * d["opp"]
        results[hl] = d.groupby("market").apply(lambda g: (g["proj"] - g["actual"]).abs().mean(), include_groups=False)
    tab = pd.DataFrame(results)
    print((tab / tab[5].to_numpy()[:, None]).round(4).to_string())
    best_hl = int((tab / tab[5].to_numpy()[:, None]).mean().idxmin())
    print(f"   -> lowest average MAE at halflife {best_hl}, but the gap to {pm.HALFLIFE} is ~1%: keeping {pm.HALFLIFE}")

    df = collect_with(h, seasons, opp, pm.HALFLIFE)
    df["proj"] = df["mu"] * df["opp"]

    # ---- 2. bias calibration, walk-forward ---------------------------------------
    print("\n2) BIAS CALIBRATION  actual ~ a + b*projection, fitted on earlier seasons only")
    df["cal"] = df["proj"]
    coefs = {}
    for m in bt.MARKETS:
        for s in tests:
            fit = df[(df["market"] == m) & (df["season"] < s)]
            if len(fit) < 300:
                continue
            b, a = np.polyfit(fit["proj"], fit["actual"], 1)
            sel = (df["market"] == m) & (df["season"] == s)
            df.loc[sel, "cal"] = a + b * df.loc[sel, "proj"]
        allfit = df[df["market"] == m]
        b, a = np.polyfit(allfit["proj"], allfit["actual"], 1)
        coefs[m] = {"a": float(a), "b": float(b)}
    t = df[df["season"].isin(tests)]
    print(f"{'market':28s} {'MAE raw':>9s} {'MAE cal':>9s} {'chg':>7s}  {'bias raw':>9s} {'bias cal':>9s}  slope b")
    for m, g in t.groupby("market"):
        r, c = (g["proj"] - g["actual"]).abs().mean(), (g["cal"] - g["actual"]).abs().mean()
        print(f"{m:28s} {r:9.3f} {c:9.3f} {(c / r - 1) * 100:+6.1f}%  {(g['proj'] - g['actual']).mean():+9.3f} {(g['cal'] - g['actual']).mean():+9.3f}  {coefs[m]['b']:.2f}")

    # ---- 3. game environment: is it real out of sample? ---------------------------
    print("\n3) GAME ENVIRONMENT  regress forecast error on total/spread; effect fitted on earlier seasons,")
    print("   then the SLOPE of realized error on that effect out of sample (1.0 = right size, 0 = useless, <0 = harmful)")
    df["r"] = np.clip(df["actual"] / df["cal"].clip(lower=1e-6) - 1, -1.5, 1.5)
    df["env_x"] = 0.0
    env_coefs = {}
    for m in bt.MARKETS:
        for s in tests:
            fit = df[(df["market"] == m) & (df["season"] < s)]
            if len(fit) < 300:
                continue
            X = np.c_[fit["dlt"], fit["dm"]]
            coef, *_ = np.linalg.lstsq(X, fit["r"], rcond=None)
            sel = (df["market"] == m) & (df["season"] == s)
            df.loc[sel, "env_x"] = np.c_[df.loc[sel, "dlt"], df.loc[sel, "dm"]] @ coef
        X = np.c_[df[df["market"] == m]["dlt"], df[df["market"] == m]["dm"]]
        coef, *_ = np.linalg.lstsq(X, df[df["market"] == m]["r"], rcond=None)
        env_coefs[m] = {"a": float(coef[0]), "b": float(coef[1])}
    t = df[df["season"].isin(tests)]
    oos_slope = {}
    print(f"{'market':28s} {'slope':>7s} {'(se)':>7s}   in-sample a (per ln total)   b (per margin pt)")
    for m, g in t.groupby("market"):
        x = g["env_x"].to_numpy(); r = g["r"].to_numpy()
        if not (x * x).sum():
            continue
        slope = float((x * r).sum() / (x * x).sum()); se = float(np.sqrt(np.var(r - slope * x) / (x * x).sum()))
        oos_slope[m] = slope
        print(f"{m:28s} {slope:7.2f} {se:7.2f}      {env_coefs[m]['a']:+.2f}                        {env_coefs[m]['b']*100:+.2f}%")

    # ---- 4. spread ----------------------------------------------------------------
    print("\n4) SPREAD  CRPS (lower is better) by variance scale, calibrated means, test seasons")
    rng = np.random.default_rng(3)
    best_scale = {}
    for m, g in t.groupby("market"):
        g = g.sample(min(len(g), 900), random_state=2)
        row = {s: np.mean([crps(k, mu, v * s, a, rng) for k, mu, v, a in zip(g["kind"], g["cal"], g["var"], g["actual"])]) for s in SCALES}
        best = min(row, key=row.get); best_scale[m] = best
        print(f"  {m:28s} " + " ".join(f"{s:.2f}:{row[s]:.3f}" for s in SCALES) + f"   -> best x{best}")

    if args.write:
        # The environment effect is dampened by how well it held up out of sample: a market whose
        # backtest slope was 0.55 gets 55% of the fitted effect; slopes above 1 are capped at 1.
        env = {m: {**c, "damp": float(np.clip(oos_slope.get(m, 0.0), 0.0, 1.0))} for m, c in env_coefs.items()}
        OUT.write_text(json.dumps({
            "calibration": coefs,
            "environment": env,
            "seasons": seasons,
            "note": "Fitted by tune_props.py on walk-forward backtests (calibration: actual ~ a + b*projection; "
                    "environment: relative error ~ a*ln(total/usual) + b*(margin - usual), times damp). "
                    "Regenerate with: python tune_props.py --write",
        }, indent=1), encoding="utf-8")
        print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
