"""
improve.py
==========
The learning loop behind the Monday report: use what has been graded to re-measure how much the model deserves to
be trusted, and say plainly what changed.

HOW THE MODEL LEARNS FROM ITS RESULTS
-------------------------------------
Every graded prop in tracking/props_log.csv records the model's raw probability, the market's no-vig probability,
and what happened. `market_trust.py` turned 14,777 such comparisons from 2024-25 (ESPN BET closing lines) into a
blend weight per market -- how much of the model's disagreement with the market to believe -- and stored the
backtest rows compactly in tracking/trust_evidence.csv. Each week this module:

  1. adds the newly graded, tracked props to that evidence (a market's weight is then re-measured on ALL of it);
  2. re-runs the same rules (Brier score against the market's own probability over a grid of weights; a weight above
     the floor only if the improvement's bootstrap interval excludes zero; capped at market_trust.WEIGHT_CAP);
  3. applies a new weight only if it moves by at least MIN_CHANGE, records the change in tracking/learning_log.csv,
     and rewrites trust_weights.json (which export_props.py reads on its next run);
  4. flags any group of picks running far from what the model's own probabilities expected (|z| >= FLAG_Z over at
     least FLAG_MIN picks) so a drifting market gets looked at.

Nothing here can push a weight past WEIGHT_CAP or above the floor without statistical evidence, and a week with
too little new data changes nothing -- by design, so the model cannot chase noise.
"""

import datetime
import json
import pathlib

import numpy as np
import pandas as pd

import market_trust as mt

EVIDENCE_FILE = pathlib.Path(__file__).parent / "tracking" / "trust_evidence.csv"
LEARNING_LOG = pathlib.Path(__file__).parent / "tracking" / "learning_log.csv"
MIN_CHANGE = 0.025            # a re-measured weight must differ by at least this to replace the current one
MIN_LIVE_PER_MARKET = 40      # tracked props needed before a market's weight is re-measured at all
FLAG_Z = 2.0
FLAG_MIN = 30


def live_evidence(log: pd.DataFrame) -> pd.DataFrame:
    """Graded, tracked props as evidence rows: market, p_mkt (the market's no-vig P(over)), p_m_env (the model's raw P(over)), over."""
    if log is None or log.empty:
        return pd.DataFrame(columns=["market", "p_mkt", "p_m_env", "over"])
    d = log[(log["pos"] != "GAME") & log["result"].isin(["over", "under"]) & log["market_over"].notna() & log["model_over"].notna()]
    d = d[d["kind"] != "td"] if "kind" in d else d
    return pd.DataFrame({"market": d["market"].to_numpy(), "p_mkt": d["market_over"].astype(float).to_numpy(),
                         "p_m_env": d["model_over"].astype(float).to_numpy(), "over": (d["result"] == "over").astype(float).to_numpy()})


def load_evidence(path=None) -> pd.DataFrame:
    path = pathlib.Path(path or EVIDENCE_FILE)
    if not path.exists():
        return pd.DataFrame(columns=["market", "p_mkt", "p_m_env", "over"])
    return pd.read_csv(path)[["market", "p_mkt", "p_m_env", "over"]]


def save_evidence_from_frame(frame: pd.DataFrame, path=None) -> int:
    """One-time: store the backtest comparison (market_trust.build_frame's output) compactly. -> rows written."""
    path = pathlib.Path(path or EVIDENCE_FILE)
    out = frame[["market", "p_mkt", "p_m_env", "over"]].copy()
    out[["p_mkt", "p_m_env"]] = out[["p_mkt", "p_m_env"]].round(4)
    path.parent.mkdir(exist_ok=True)
    out.to_csv(path, index=False)
    return len(out)


def log_change(msg: dict, path=None):
    path = pathlib.Path(path or LEARNING_LOG)
    path.parent.mkdir(exist_ok=True)
    row = pd.DataFrame([msg])
    if path.exists() and path.stat().st_size:
        row = pd.concat([pd.read_csv(path), row], ignore_index=True)
    row.to_csv(path, index=False)


def update_weights(log: pd.DataFrame, evidence: pd.DataFrame = None, trust_path=None, log_path=None, today=None) -> list:
    """
    Re-measure each market's blend weight on backtest + tracked evidence and apply real changes.
    -> plain-language lines describing what changed (empty if nothing).
    """
    evidence = load_evidence() if evidence is None else evidence
    live = live_evidence(log)
    if live.empty:
        return []
    today = today or datetime.date.today().isoformat()
    pooled = pd.concat([evidence, live], ignore_index=True)
    trust_path = pathlib.Path(trust_path or mt.TRUST_FILE)
    old = mt.load_weights(trust_path)
    notes = []
    changed = {}
    for market, d in pooled.groupby("market"):
        n_live = int((live["market"] == market).sum())
        if n_live < MIN_LIVE_PER_MARKET:
            continue
        r = mt.choose_weight(d, "p_m_env")
        if r.get("weight") is None:
            continue
        cur = old.get(market)
        if cur is not None and abs(r["weight"] - cur) < MIN_CHANGE:
            continue
        changed[market] = r["weight"]
        label = market.replace("player_", "").replace("_", " ")
        before = "the default" if cur is None else f"{cur:.2f}"
        notes.append(f"{label}: the model's weight against the market moved from {before} to {r['weight']:.2f} "
                     f"({r['verdict']}; measured on {r['n']:,} props, {n_live} of them tracked this season).")
        log_change({"date": today, "market": market, "old_weight": cur, "new_weight": r["weight"], "n": r["n"], "n_live": n_live,
                    "verdict": r["verdict"]}, log_path)
    if changed and trust_path.exists():
        payload = json.loads(trust_path.read_text(encoding="utf-8"))
        payload.setdefault("weights", {}).update(changed)
        payload["last_updated"] = today
        trust_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return notes


def drift_flags(season_to_date: dict) -> list:
    """Groups whose season-to-date results sit far from what the model's own probabilities expected."""
    out = []
    for g, v in sorted(season_to_date.items()):
        if v.get("z") is None or v["n"] < FLAG_MIN or abs(v["z"]) < FLAG_Z or ":" in g and not g.endswith(":high"):
            continue
        way = "above" if v["z"] > 0 else "below"
        out.append(f"{g}: {v['wins']}-{v['losses']} over {v['n']} picks is {abs(v['z']):.1f} standard deviations {way} what the model's own probabilities "
                   f"expected; " + ("worth watching." if v["z"] > 0 else "its raw confidence runs too high here. That is what blending toward the market "
                                    "is for, so the real question is the weight (see above), not the raw number."))
    return out


def recent_changes(today=None, days: int = 7, path=None) -> list:
    """Weight changes the loop made in the last `days` days, from the learning log, so a re-run of the report still shows them."""
    path = pathlib.Path(path or LEARNING_LOG)
    if not path.exists() or not path.stat().st_size:
        return []
    d = pd.read_csv(path)
    today = pd.Timestamp(today or datetime.date.today())
    d = d[pd.to_datetime(d["date"]) >= today - pd.Timedelta(days=days - 1)]
    out = []
    for r in d.itertuples(index=False):
        label = str(r.market).replace("player_", "").replace("_", " ")
        before = "the default" if pd.isna(r.old_weight) else f"{r.old_weight:.2f}"
        out.append(f"{label}: the model's weight against the market moved from {before} to {r.new_weight:.2f} "
                   f"({r.verdict}; measured on {int(r.n):,} props, {int(r.n_live)} of them tracked this season).")
    return out


def run(log: pd.DataFrame, season: int, week: int, season_to_date: dict = None, trust_path=None, log_path=None,
        evidence: pd.DataFrame = None, today=None) -> list:
    """What the report's learning section says. Safe to call every week (paths are injectable for tests)."""
    notes = update_weights(log, evidence, trust_path, log_path, today)
    notes = notes + [m for m in recent_changes(today, 7, log_path) if m not in notes]     # earlier runs this week count too
    if season_to_date:
        notes += drift_flags(season_to_date)
    if not notes:
        notes = ["No market's evidence moved enough to change how much the model is trusted, and nothing is running unusually far from expectation."]
    return notes
