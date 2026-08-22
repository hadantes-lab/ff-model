"""
backtest.py
===========
The reality check. A betting model that hasn't been backtested is just numbers
with confidence. This measures whether your model would have BEATEN THE MARKET
on past games — the only honest evidence it works.

WHAT IT DOES
------------
1. Walk through historical weeks you did NOT use to build the projection.
2. For each game/prop, generate the model's probability as if it were live.
3. Compare to the actual closing line and the actual result.
4. Track: hit rate, ROI, and — most importantly — CLV (closing line value).

WHY CLV MATTERS MOST
--------------------
Beating the closing line is the single best predictor of long-term profit,
because the closing line is the sharpest number the market produces. If your
model consistently bets sides that the line later moves toward, you have a real
edge even before results come in. Hit rate is noisy over small samples; CLV
signals skill faster.

HONESTY GUARDRAILS built in:
  - out-of-sample only (never test on data you trained on)
  - accounts for vig (a 52.4% hit rate at -110 is break-even, not winning)
  - reports sample size loudly (small samples lie)

Run:  python backtest.py
"""

import numpy as np
import pandas as pd

BREAKEVEN_110 = 0.524   # you must clear this at standard -110 juice to profit


def evaluate_bets(bet_log: pd.DataFrame):
    """
    bet_log columns expected:
      model_prob, implied_prob, odds (american), won (1/0), closing_prob
    Returns summary stats with honest framing.
    """
    n = len(bet_log)
    if n == 0:
        return {"note": "No bets cleared the edge threshold — that's a valid (conservative) result."}

    hit_rate = bet_log["won"].mean()

    # ROI: sum of profits / total staked (flat $1 bets)
    def profit(row):
        if row["won"] == 1:
            o = row["odds"]
            return (100 / -o) if o < 0 else (o / 100)
        return -1.0
    profits = bet_log.apply(profit, axis=1)
    roi = profits.sum() / n

    # CLV: did we beat the closing line? (our implied < closing implied = we got a better price)
    clv = (bet_log["closing_prob"] - bet_log["implied_prob"]).mean()
    clv_beat_rate = (bet_log["implied_prob"] < bet_log["closing_prob"]).mean()

    return {
        "n_bets": n,
        "hit_rate": round(float(hit_rate), 4),
        "breakeven_needed": BREAKEVEN_110,
        "beat_breakeven": bool(hit_rate > BREAKEVEN_110),
        "roi": round(float(roi), 4),
        "avg_clv_pts": round(float(clv * 100), 2),
        "clv_positive_rate": round(float(clv_beat_rate), 4),
        "verdict": _verdict(hit_rate, roi, clv, n),
    }


def _verdict(hit_rate, roi, clv, n):
    if n < 50:
        return f"SAMPLE TOO SMALL ({n} bets). Need 200+ before trusting anything."
    signals = []
    if clv > 0:
        signals.append("positive CLV (good sign — beating closing line)")
    else:
        signals.append("negative CLV (warning — market disagrees)")
    if roi > 0:
        signals.append(f"positive ROI ({roi:+.1%})")
    else:
        signals.append(f"negative ROI ({roi:+.1%})")
    return " ; ".join(signals)


def walk_forward_backtest(all_weeks, build_fn, price_fn, result_fn):
    """
    Generic walk-forward harness. You wire in three functions:
      build_fn(train_weeks) -> projections   (uses only prior weeks)
      price_fn(week)         -> DataFrame of market lines + closing lines
      result_fn(week)        -> actual outcomes

    This structure GUARANTEES out-of-sample testing: week N is always predicted
    using only weeks < N. That's the discipline that keeps a backtest honest.
    """
    from simulate import simulate_prop, find_edge

    bets = []
    for i, week in enumerate(all_weeks):
        if i < 4:   # need a few weeks of history to start
            continue
        train = all_weeks[:i]
        projections = build_fn(train)
        lines = price_fn(week)
        results = result_fn(week)

        for _, row in lines.iterrows():
            proj = projections.get(row["player"])
            if not proj:
                continue
            sim = simulate_prop(proj, row["stat"], row["line"])
            if not sim:
                continue
            side_prob = sim["p_over"] if row["side"] == "over" else sim["p_under"]
            edge = find_edge(side_prob, row["odds"])
            if edge["bet"]:
                actual = results.get((row["player"], row["stat"]))
                if actual is None:
                    continue
                won = (actual > row["line"]) == (row["side"] == "over")
                bets.append({
                    "week": week, "player": row["player"], "stat": row["stat"],
                    "side": row["side"], "line": row["line"], "odds": row["odds"],
                    "model_prob": side_prob, "implied_prob": edge["implied_prob"],
                    "closing_prob": row.get("closing_prob", edge["implied_prob"]),
                    "won": int(won),
                })
    return evaluate_bets(pd.DataFrame(bets))


if __name__ == "__main__":
    # DEMO: synthetic bet log so you see how results read before wiring live data.
    np.random.seed(1)
    n = 120
    # simulate a model with a small real edge
    demo = pd.DataFrame({
        "odds": np.random.choice([-110, -115, +100, -105], n),
        "implied_prob": np.random.uniform(0.47, 0.55, n),
    })
    # give it a slight true edge -> ~54% hit
    demo["won"] = (np.random.random(n) < 0.54).astype(int)
    demo["closing_prob"] = demo["implied_prob"] + np.random.uniform(-0.01, 0.03, n)

    summary = evaluate_bets(demo)
    print("=== BACKTEST SUMMARY (demo data) ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("\nReplace the demo data with walk_forward_backtest(...) on real weeks.")
