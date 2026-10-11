"""
market_trust.py
===============
How much should the model's opinion count against the market's, market by market? Until we had real past
lines this was a judgment call (one global cap, LAMBDA_MAX = 0.15). With ~13,000 historical closing lines
(espn_lines.py: 2024-25 ESPN BET, with Over/Under prices) it can be MEASURED:

    p_blend = p_market + w * (p_model - p_market)

for each market, find the weight w in a grid that minimizes Brier score against what actually happened, using
the market's own no-vig probability as the baseline. w = 0 means "the model adds nothing; just use the
market"; w = 1 means "trust the model fully".

    python market_trust.py            # run the study, print the table
    python market_trust.py --write    # also write trust_weights.json (read by export_props.py)

WHAT IT WRITES, AND HOW CAUTIOUSLY
----------------------------------
A weight only goes above the floor if the improvement over the market is statistically convincing (bootstrap
95% interval excludes zero); a market where the model merely looks a bit better keeps the floor, and one where
it looks worse gets zero. Weights are capped at WEIGHT_CAP. The file is the *static prior*: once the live tracker
has 50+ graded picks for a market, props_tracker's own refinement overrides it, as before.

Honest limits: 2024-25 only, one book (ESPN BET) as "the market", and prices at the close.
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

TRUST_FILE = pathlib.Path(__file__).parent / "trust_weights.json"
GRID = (0.0, 0.05, 0.10, 0.15, 0.25, 0.40)
WEIGHT_CAP = 0.25          # even a convincing edge is capped: the sample is two seasons of one book
FLOOR = 0.05               # a market where the model isn't clearly worse keeps a small voice
MIN_N = 150


def brier_change(df: pd.DataFrame, col: str, w: float) -> np.ndarray:
    """Per-prop change in squared error from blending w of the model into the market (negative = helps)."""
    p = df["p_mkt"] + w * (df[col] - df["p_mkt"])
    return ((p - df["over"]) ** 2 - (df["p_mkt"] - df["over"]) ** 2).to_numpy()


def bootstrap_ci(v: np.ndarray, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(v, len(v)).mean() for _ in range(n)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def choose_weight(df: pd.DataFrame, col: str) -> dict:
    """The blend weight for one market's rows -> {weight, best_grid, change, ci, n, verdict}."""
    n = len(df)
    if n < MIN_N:
        return {"weight": None, "n": n, "verdict": "too few lines"}
    scores = {w: float(brier_change(df, col, w).mean()) for w in GRID}
    best = min(scores, key=scores.get)
    lo, hi = bootstrap_ci(brier_change(df, col, best)) if best > 0 else (0.0, 0.0)
    if best > 0 and hi < 0:
        w, verdict = min(best, WEIGHT_CAP), "model adds information (interval excludes zero)"
    elif best > 0:
        w, verdict = FLOOR, "model looks slightly better but not conclusively"
    else:
        w, verdict = 0.0, "model is no better than the market"
    return {"weight": w, "best_grid": best, "change_at_best": scores[best], "ci": [lo, hi], "n": n,
            "change_at_current_cap": scores[0.15], "verdict": verdict}


def study(x: pd.DataFrame, col: str = "p_m_env") -> dict:
    """Run choose_weight for every market (and position group) in the prepared frame."""
    out = {}
    for m, d in x.groupby("market"):
        out[m] = choose_weight(d, col)
    return out


EARLY_QUANTILE = 0.75       # "strong edge" = the largest quarter of projection gaps vs the opening line
FAMILYWISE_ALPHA = 0.05    # across ALL markets tested (Bonferroni), so one lucky market can't qualify by chance
EARLY_MIN_PICKS = 100


def wilson(k: int, n: int, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def _profit(won: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """Profit per $1 staked at decimal odds `dec` (a loss is -1, a win dec - 1)."""
    return np.where(won, dec - 1.0, -1.0)


def early_table(x: pd.DataFrame, n_tests: int = None) -> dict:
    """
    Per market: when the model's projection sits far from the OPENING line, does betting the side it points to
    make money at the opening PRICES? `x` needs actual, open_line, open_over, open_under and m_env (model mean).
    "Strong" = the largest quarter of gaps (|gap| / line). A market earns early-edge flags only if the ROI's
    bootstrap interval, widened for the number of markets tested (Bonferroni at FAMILYWISE_ALPHA), excludes zero.
    -> {market: {qualifies, rel_threshold, roi, roi_ci, hit_rate, n_picks, n}}.
    """
    import odds_api

    d = x.dropna(subset=["open_line", "m_env", "open_over", "open_under"]).copy()
    d = d[d["actual"] != d["open_line"]]
    markets = [m for m, g in d.groupby("market") if len(g) >= MIN_N]
    alpha = FAMILYWISE_ALPHA / max(1, n_tests or len(markets))
    out = {}
    rng = np.random.default_rng(0)
    for m in markets:
        g = d[d["market"] == m]
        rel = (g["m_env"] - g["open_line"]) / g["open_line"]
        thr = float(rel.abs().quantile(EARLY_QUANTILE))
        big = (rel.abs() >= thr).to_numpy()
        over = (rel > 0).to_numpy()
        won = np.where(over, (g["actual"] > g["open_line"]).to_numpy(), (g["actual"] < g["open_line"]).to_numpy())
        price = np.where(over, g["open_over"].to_numpy(), g["open_under"].to_numpy())
        dec = np.array([odds_api.american_to_decimal(p) for p in price])
        v = _profit(won, dec)[big]
        n = int(big.sum())
        if n < EARLY_MIN_PICKS:
            continue
        boots = np.array([rng.choice(v, n).mean() for _ in range(4000)])
        lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out[m] = {"qualifies": bool(lo > 0), "rel_threshold": thr, "roi": float(v.mean()), "roi_ci": [float(lo), float(hi)],
                  "hit_rate": float(won[big].mean()), "n_picks": n, "n": int(len(g)), "alpha_per_market": alpha}
    return out


def confidence_study(x: pd.DataFrame) -> dict:
    """
    Measure how strongly the model's record on a player should move the weight on that player. `x` is the
    comparison frame (see build_frame): its raw model probability `p_m_env` and the real outcome `over` define each
    row's hit. -> {strength, slope, ci, n} where strength > 0 only if the record predicts the next hit.
    """
    import player_confidence as pc

    d = x.copy()
    d["hit"] = (((d["p_m_env"] > 0.5) & (d["over"] == 1)) | ((d["p_m_env"] < 0.5) & (d["over"] == 0))).astype(float)
    d = d[d["p_m_env"] != 0.5]
    p = pc.persistence(d)
    return dict(p, strength=pc.strength_from(p))


def write(results: dict, path=None, note="", early: dict = None, confidence: dict = None):
    path = pathlib.Path(path or TRUST_FILE)
    payload = {"weights": {m: r["weight"] for m, r in results.items() if r.get("weight") is not None},
               "evidence": {m: {k: r[k] for k in ("n", "verdict", "change_at_best", "ci", "best_grid") if k in r}
                            for m, r in results.items()},
               "early": early or {},
               "confidence": confidence or {"strength": 0.0},
               "note": note or "Static prior for the market-blend weight, fitted by market_trust.py on 2024-25 ESPN BET closing "
                               "lines. `early` marks the markets whose big projection gaps vs the OPENING line have won (see early_table)."}
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return path


def load_confidence(path=None) -> float:
    """How strongly the 0-10 player grade moves the model's weight (0 until the record is shown to predict anything)."""
    path = pathlib.Path(path or TRUST_FILE)
    if not path.exists():
        return 0.0
    return float((json.loads(path.read_text(encoding="utf-8")).get("confidence") or {}).get("strength", 0.0))


def load_early(path=None) -> dict:
    """{market: {...}} for the markets that qualify for early-edge flags, or {}."""
    path = pathlib.Path(path or TRUST_FILE)
    if not path.exists():
        return {}
    return {m: v for m, v in json.loads(path.read_text(encoding="utf-8")).get("early", {}).items() if v.get("qualifies")}


def load_weights(path=None) -> dict:
    """{market: weight} from trust_weights.json, or {} if it doesn't exist (export_props then uses LAMBDA_MAX)."""
    path = pathlib.Path(path or TRUST_FILE)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("weights", {})


# ---- building the frame (needs the backtest harness + espn_lines) -----------------------
def build_frame(seasons=(2024, 2025)) -> pd.DataFrame:
    import backtest_props as bt
    import game_context as gc
    import odds_api
    import props_model as pm
    import espn_lines as el

    h = bt.load(list(seasons))
    bt.MARKETS = {m: s for m, s in bt.MARKETS.items() if s.kind != "td"}
    opp_lookup = {}
    for s, w in sorted({(int(a), int(b)) for a, b in h.loc[h["season"].isin(seasons) & (h["week"] >= 4), ["season", "week"]].to_numpy()}):
        prior = h[(h["season"] * 100 + h["week"]) < (s * 100 + w)]
        opp_lookup[(s, w)] = {m: pm.opponent_multipliers(prior, sp) for m, sp in bt.MARKETS.items()}
    df = bt.collect(h, list(seasons), opp_lookup)
    df["a"] = df["b"] = 0.0
    for s in seasons:
        multi = gc.fit_context_effects(h[h["season"] < s], bt.MARKETS)
        for m in bt.MARKETS:
            sel = (df["season"] == s) & (df["market"] == m)
            if m in multi:
                df.loc[sel, "a"], df.loc[sel, "b"] = multi[m]["a"], multi[m]["b"]
    clip = lambda v: np.clip(v, *gc.MULT_CLIP)
    df["m_env"] = df["mu"] * df["opp"] * clip(1 + df["a"] * df["dlt"] + df["b"] * df["dm"])

    e = el.load().rename(columns={"player_id": "pid"})
    e = e[(e["priced"] == True) & e["close_over"].notna() & e["close_under"].notna()]
    x = df.merge(e[["season", "week", "pid", "market", "open_line", "open_over", "open_under", "close_line", "close_over", "close_under"]],
                 on=["season", "week", "pid", "market"])
    x = x[x["actual"] != x["close_line"]].copy()
    po, pu = x["close_over"].map(odds_api.american_to_prob), x["close_under"].map(odds_api.american_to_prob)
    x["p_mkt"] = po / (po + pu)
    x["over"] = (x["actual"] > x["close_line"]).astype(float)
    rng = np.random.default_rng(3)

    def p_over(row):
        mean = row["m_env"]
        var = row["var"] * (mean / row["mu"]) ** 2 if row["mu"] > 0 else row["var"]
        d = pm.draw(row["kind"], mean, var, rng, 3000)
        return float((d > row["close_line"]).mean() + 0.5 * (d == row["close_line"]).mean())
    x["p_m_env"] = [p_over(r) for _, r in x.iterrows()]

    # the same comparison at the OPENING line, where the market has not yet absorbed game script
    o = x.dropna(subset=["open_line", "open_over", "open_under"]).copy()
    o = o[o["actual"] != o["open_line"]]
    po, pu = o["open_over"].map(odds_api.american_to_prob), o["open_under"].map(odds_api.american_to_prob)
    o["p_mkt"] = po / (po + pu)
    o["over"] = (o["actual"] > o["open_line"]).astype(float)
    o["close_line"] = o["open_line"]                     # p_over() reads the line from this column
    o["p_m_env"] = [p_over(r) for _, r in o.iterrows()]
    x.attrs["open"] = o
    return x


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="write trust_weights.json")
    ap.add_argument("--seasons", type=int, nargs="+", default=[2024, 2025])
    args = ap.parse_args()
    print("building the comparison (model vs real closing lines)...", file=sys.stderr)
    x = build_frame(args.seasons)
    res = study(x)
    print(f"{len(x)} props with prices and results | overall market Brier {((x.p_mkt - x.over) ** 2).mean():.4f}\n")
    print(f"{'market':28s} {'n':>5s} {'Brier chg @0.15':>16s} {'best w':>7s} {'chg @ best':>11s} {'95% CI':>20s}  {'weight':>6s}  verdict")
    for m, r in sorted(res.items()):
        if r["weight"] is None:
            print(f"{m:28s} {r['n']:5d}  {r['verdict']}")
            continue
        print(f"{m:28s} {r['n']:5d} {r['change_at_current_cap']:+16.4f} {r['best_grid']:7.2f} {r['change_at_best']:+11.4f} "
              f"[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}]  {r['weight']:6.2f}  {r['verdict']}")
    early = early_table(x)
    print("\nEARLY-WEEK SIGNAL: strongest quarter of projection gaps vs the OPENING line")
    for m, r in sorted(early.items()):
        print(f"  {m:28s} won {r['hit_rate']:.1%} of {r['n_picks']}  ROI {r['roi']:+.1%} [{r['roi_ci'][0]:+.1%}, {r['roi_ci'][1]:+.1%}]  gap >= {r['rel_threshold']:.1%} of the line"
              f"  -> {'QUALIFIES' if r['qualifies'] else 'no'}")
    conf = confidence_study(x)
    ci = conf["ci"]
    print(f"\nPLAYER-RECORD CONFIDENCE: does the model's record on a player predict whether it hits him next? "
          f"slope {conf['slope']:+.3f}" + (f" [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci[0] is not None else "") +
          f" over {conf['n']} props -> strength {conf['strength']:.2f}")
    if args.write:
        print("\nwrote", write(res, early=early, confidence=conf))


if __name__ == "__main__":
    main()
