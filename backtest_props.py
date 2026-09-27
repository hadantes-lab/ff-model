"""
backtest_props.py
=================
Walk-forward test of the props projections against what actually happened.

For every player-game in the test seasons (default 2024-25, weeks 4-18) it projects each stat
using ONLY games before kickoff, then compares to the real result. Betting lines for past
player props are not available, so this measures what we can measure honestly:

  - accuracy of the projected mean  (MAE / bias)         -> does an adjustment help?
  - calibration of the distribution (PIT interval coverage) -> is the spread right?

and compares model variants side by side:

  base     recency-weighted average of his recent games
  +opp     ... times the opponent-strength multiplier
  +env(x)  ... times a game-environment multiplier from the game's total and spread, with the
           coefficients estimated ONLY on seasons before the test season (no peeking)

    python backtest_props.py                    # ~1-3 minutes
    python backtest_props.py --seasons 2023 2024 2025
"""

import argparse
import sys

import numpy as np
import pandas as pd
import nflreadpy as nfl

import game_context as gc
import props_model as pm

MARKETS = {m: s for m, s in pm.MARKETS.items() if s.kind != "td"}


def load(seasons):
    first = min(seasons) - 3
    ps = nfl.load_player_stats(list(range(first, max(seasons) + 1))).to_pandas()
    hist = pm.build_history(ps)
    hist = hist[hist["week"] <= 18]
    lines = gc.load_game_lines(range(first, max(seasons) + 1))
    h = gc.attach_context(hist, lines).dropna(subset=["total", "margin"])
    h = h.sort_values(["season", "week"]).reset_index(drop=True)
    return h


def univariate_effects(h, markets):
    """Total-only within-player elasticity (the simple version) for comparison."""
    out = {}
    for m, spec in markets.items():
        d = h.assign(_v=pm.stat_series(h, spec))
        d = d.groupby("player_id").filter(lambda x: len(x) >= 8 and x["_v"].mean() >= pm.ROLE_FLOOR[m])
        if len(d) < 500:
            continue
        g = d.groupby("player_id")
        rel = d["_v"] / g["_v"].transform("mean") - 1
        x = np.log(d["total"]) - g["total"].transform(lambda s: np.log(s).mean())
        out[m] = {"a": float((rel * x).sum() / (x * x).sum()), "b": 0.0}
    return out


def collect(h, seasons, opp_lookup):
    """One record per (player-game, market) with everything needed to score any variant."""
    recs = []
    test = h[(h["season"].isin(seasons)) & (h["week"] >= 4)]
    for pid, g in h.groupby("player_id"):
        g = g.reset_index(drop=True)
        n = len(g)
        if n <= pm.MIN_GAMES:
            continue
        vals = {m: pm.stat_series(g, s).to_numpy() for m, s in MARKETS.items()}
        ln_total, margin = np.log(g["total"].to_numpy()), g["margin"].to_numpy()
        for k in np.flatnonzero(g["season"].isin(seasons).to_numpy() & (g["week"].to_numpy() >= 4)):
            lo = max(0, k - pm.MAX_GAMES)
            if k - lo < pm.MIN_GAMES:
                continue
            w = pm.recency_weights(k - lo)
            s_, wk_ = int(g.at[k, "season"]), int(g.at[k, "week"])
            avg_lt = float(np.average(ln_total[lo:k], weights=w))
            avg_mg = float(np.average(margin[lo:k], weights=w))
            for m, spec in MARKETS.items():
                mu, var, n_eff = pm.fit_moments(vals[m][lo:k], w)
                if mu < pm.ROLE_FLOOR[m]:
                    continue
                opp = opp_lookup[(s_, wk_)][m].get((g.at[k, "opponent_team"], g.at[k, "position"]), 1.0)
                recs.append((pid, m, s_, wk_, spec.kind, float(vals[m][k]), mu, var, n_eff, opp,
                             ln_total[k] - avg_lt, margin[k] - avg_mg))
    cols = ["pid", "market", "season", "week", "kind", "actual", "mu", "var", "n_eff", "opp", "dlt", "dm"]
    return pd.DataFrame(recs, columns=cols)


def score(df, mean_col):
    e = df[mean_col] - df["actual"]
    return float(e.abs().mean()), float(e.mean())


def pit(df, mean_col, var_scale=1.0, n=600, seed=7):
    """Central-interval coverage of the fitted distribution at 50/80/90% (should match the label)."""
    rng = np.random.default_rng(seed)
    u = np.empty(len(df))
    for i, (mu, var, a, kind) in enumerate(zip(df[mean_col], df["var"] * var_scale, df["actual"], df["kind"])):
        d = pm.draw(kind, mu, var, rng, n)
        below, equal = (d < a).mean(), (d == a).mean()
        u[i] = below + rng.random() * equal if kind == "count" else (d <= a).mean()
    cov = {c: float(((u > (1 - c) / 2) & (u < 1 - (1 - c) / 2)).mean()) for c in (0.5, 0.8, 0.9)}
    return cov, float(u.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", default=[2024, 2025])
    args = ap.parse_args()
    seasons = args.seasons

    print("loading...", file=sys.stderr)
    h = load(seasons)
    print(f"{len(h)} player-games with betting-line context", file=sys.stderr)

    # defensive-strength tables as of each test week (only earlier games), exactly as production builds them
    opp_lookup = {}
    for s, w in sorted({(int(a), int(b)) for a, b in h.loc[h["season"].isin(seasons) & (h["week"] >= 4), ["season", "week"]].to_numpy()}):
        prior = h[(h["season"] * 100 + h["week"]) < (s * 100 + w)]
        opp_lookup[(s, w)] = {m: pm.opponent_multipliers(prior, sp) for m, sp in MARKETS.items()}
    df = collect(h, seasons, opp_lookup)
    print(f"{len(df)} projections across {df['market'].nunique()} markets", file=sys.stderr)

    # game-environment coefficients estimated strictly BEFORE each test season
    df["a_multi"] = df["b_multi"] = df["a_uni"] = 0.0
    for s in seasons:
        prior = h[h["season"] < s]
        multi = gc.fit_context_effects(prior, MARKETS)
        uni = univariate_effects(prior, MARKETS)
        for m in MARKETS:
            sel = (df["season"] == s) & (df["market"] == m)
            if m in multi:
                df.loc[sel, "a_multi"], df.loc[sel, "b_multi"] = multi[m]["a"], multi[m]["b"]
            if m in uni:
                df.loc[sel, "a_uni"] = uni[m]["a"]

    clip = lambda x: np.clip(x, *gc.MULT_CLIP)
    df["m_base"] = df["mu"]
    df["m_opp"] = df["mu"] * df["opp"]
    df["m_env_multi"] = df["mu"] * df["opp"] * clip(1 + df["a_multi"] * df["dlt"] + df["b_multi"] * df["dm"])
    df["m_env_uni"] = df["mu"] * df["opp"] * clip(1 + df["a_uni"] * df["dlt"])
    variants = ["m_base", "m_opp", "m_env_multi", "m_env_uni"]

    print("\nMEAN ABSOLUTE ERROR of the projected mean (lower is better; % change vs base)")
    print(f"{'market':28s} {'n':>6s} " + " ".join(f"{v[2:]:>13s}" for v in variants))
    for m, g in df.groupby("market"):
        base_mae = score(g, "m_base")[0]
        cells = []
        for v in variants:
            mae = score(g, v)[0]
            cells.append(f"{mae:7.2f} ({(mae / base_mae - 1) * 100:+4.1f}%)" if v != "m_base" else f"{mae:7.2f}       ")
        print(f"{m:28s} {len(g):6d} " + " ".join(cells))

    print("\nBIAS of the projected mean (projection minus actual; ~0 is unbiased)")
    for m, g in df.groupby("market"):
        print(f"  {m:28s} base {score(g, 'm_base')[1]:+6.2f}   +opp {score(g, 'm_opp')[1]:+6.2f}")

    print("\nCALIBRATION of the fitted distribution (central intervals should cover 50% / 80% / 90%)")
    for m, g in df.groupby("market"):
        cov, mean_u = pit(g.sample(min(len(g), 1500), random_state=1), "m_opp")
        print(f"  {m:28s} 50%->{cov[0.5]:.2f}  80%->{cov[0.8]:.2f}  90%->{cov[0.9]:.2f}   mean PIT {mean_u:.2f} (0.50 = unbiased)")
    df.to_pickle("web/cache_backtest.pkl") if False else None
    return df


if __name__ == "__main__":
    main()
