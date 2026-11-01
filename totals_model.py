"""
totals_model.py
===============
The over/under model: the market's total, adjusted for what the market has been measured to miss.

WHY IT IS BUILT THIS WAY
------------------------
The first version rolled player projections up into team points (game_odds.py). Checked against the posted totals it ran
about 10 points high on average (54.9 projected vs 44.4 posted on one real slate) and called nearly every game an over, so
its "prediction line" was wrong by a touchdown and a half before it ever said anything about value.

So the question became: is there ANY information about a game's total that the closing line does not already hold?
Tested walk-forward on 3,771 regular-season games (2012-2026, closing totals from nflverse, each season predicted using
only earlier ones), against `actual total - closing total`:

* the closing total's own error is 10.4 points on average (SD 13.1), with no tilt by season, week or line level
  (actual = 1.02 x line - 0.1);
* teams' decayed scoring for and against, plays per game (pace), offensive and defensive EPA per play, pass rate,
  home/away, primetime: none of them predicts the miss out of sample (R^2 -0.003 to +0.000, win rate 49-51%);
* two things do, a little: DOME / closed-roof games finish about 1 point ABOVE the closing total (3 standard errors, higher
  in 12 of 15 seasons, though betting every dome over won only 52% walk-forward, about break-even), and outdoor games with
  wind of 12+ mph finish about 1.5-2 points BELOW it (56.9% unders walk-forward, n = 281; 57% over all 566). Temperature
  carries nothing once wind is in.

So the prediction is the market's total plus a weather adjustment, fitted by least squares on those two flags, centered so the
average game gets no adjustment, and shrunk for the fact that a forecast is noisier than the measured game-day wind the fit used. A game with no weather effect has a
prediction equal to the line and a ~50% pick, which is the honest answer for most games. Probabilities use the real spread
of the miss (SD ~13 points, slightly right-skewed), not the simulation's.

The bottom-up simulation is still used for player props; it simply no longer sets a game total.

    python totals_model.py --write      # refit and rewrite totals_model.json (with the backtest numbers the page quotes)
"""

import json
import math
import pathlib
import sys
from statistics import NormalDist

import numpy as np
import pandas as pd

FILE = pathlib.Path(__file__).parent / "totals_model.json"
ND = NormalDist()
FIRST_SEASON = 2012
WIND_HIGH = 12.0           # mph at or above which an outdoor game counts as windy
FORECAST_SHRINK = 0.75     # forecast wind/temperature are noisier than the measured game-day values the fit used
MIN_GAMES_FIT = 500
DEFAULT = {"dome": 0.9, "wind": -1.7, "mean_dome": 0.28, "mean_wind": 0.1, "sd": 13.1, "shrink": FORECAST_SHRINK}   # used if the file is missing


def features(roof, wind, temp=None) -> dict:
    """Model inputs for one game from its roof and (forecast or measured) wind in mph: whether it is indoors, whether it is windy."""
    if str(roof).lower() in ("dome", "closed"):
        return {"dome": 1.0, "wind": 0.0}
    windy = wind is not None and not (isinstance(wind, float) and math.isnan(wind)) and float(wind) >= WIND_HIGH
    return {"dome": 0.0, "wind": 1.0 if windy else 0.0}


def _frame(sched: pd.DataFrame) -> pd.DataFrame:
    g = sched[(sched["game_type"] == "REG") & sched["total_line"].notna() & sched["home_score"].notna()
              & (sched["season"] >= FIRST_SEASON)].copy()
    g["resid"] = g["home_score"] + g["away_score"] - g["total_line"]
    f = pd.DataFrame([features(r, w, t) for r, w, t in zip(g["roof"], g["wind"], g["temp"])], index=g.index)
    return pd.concat([g[["season", "week", "total_line", "resid"]], f], axis=1)


def _ols(X: np.ndarray, y: np.ndarray, lam: float = 5.0):
    """Ridge regression with an intercept (the intercept is not penalized)."""
    Xb = np.column_stack([np.ones(len(X)), X])
    pen = lam * np.eye(Xb.shape[1])
    pen[0, 0] = 0.0
    return np.linalg.solve(Xb.T @ Xb + pen, Xb.T @ y)


COLS = ("dome", "wind")


def fit(sched: pd.DataFrame) -> dict:
    """Coefficients (points per unit of each feature, relative to a no-weather game), the residual SD, and the walk-forward check."""
    f = _frame(sched)
    if len(f) < MIN_GAMES_FIT:
        return dict(DEFAULT, n=int(len(f)))
    b = _ols(f[list(COLS)].to_numpy(float), f["resid"].to_numpy(float))
    resid = f["resid"].to_numpy(float)
    out = {"dome": round(float(b[1]), 3), "wind": round(float(b[2]), 3), "shrink": FORECAST_SHRINK,
           "mean_dome": round(float(f["dome"].mean()), 3), "mean_wind": round(float(f["wind"].mean()), 3),
           "sd": round(float(resid.std()), 2), "skew": round(float(pd.Series(resid).skew()), 2), "mean_miss": round(float(np.abs(resid).mean()), 2),
           "n": int(len(f)), "seasons": [int(f["season"].min()), int(f["season"].max())]}
    out["backtest"] = walk_forward(f)
    return out


def walk_forward(f: pd.DataFrame, first_test: int = 2019) -> dict:
    """Each season predicted from earlier seasons only: how often the weather side beat the closing total, and the closing line's own error."""
    rows = []
    for season in sorted(f["season"].unique()):
        if season < first_test:
            continue
        tr, te = f[f["season"] < season], f[f["season"] == season]
        b = _ols(tr[list(COLS)].to_numpy(float), tr["resid"].to_numpy(float))
        centered = te[list(COLS)].to_numpy(float) - tr[list(COLS)].to_numpy(float).mean(axis=0)
        adj = (centered @ b[1:]) * FORECAST_SHRINK
        rows.append(pd.DataFrame({"adj": adj, "resid": te["resid"].to_numpy(float)}))
    if not rows:
        return {}
    P = pd.concat(rows)
    P = P[P["resid"] != 0]
    res = {"n": int(len(P)), "line_mae": round(float(P["resid"].abs().mean()), 2),
           "model_mae": round(float((P["resid"] - P["adj"]).abs().mean()), 2),
           "over_rate": round(float((P["resid"] > 0).mean()), 3)}
    for thr in (0.3, 1.0):
        sel = P[P["adj"].abs() >= thr]
        res[f"picks_{thr:g}"] = {"n": int(len(sel)), "win_rate": round(float((np.sign(sel["adj"]) == np.sign(sel["resid"])).mean()), 3) if len(sel) else None}
    return res


def load(path=None) -> dict:
    try:
        return json.loads(pathlib.Path(path or FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return dict(DEFAULT)


def adjustment(model: dict, roof=None, wind=None, temp=None, retractable: bool = False) -> float:
    """Points to add to the posted total. A retractable-roof stadium whose roof position isn't known gets half the effect of each case."""
    x = features(roof, wind)
    shrink = float(model.get("shrink", FORECAST_SHRINK))
    mean = {k: float(model.get("mean_" + k, DEFAULT["mean_" + k])) for k in COLS}
    pts = lambda f: sum(float(model.get(k, DEFAULT[k])) * (f[k] - mean[k]) for k in COLS) * shrink
    if retractable and not x["dome"]:       # roof position unknown: half indoors, half as the weather says
        return 0.5 * pts(x) + 0.5 * pts({"dome": 1.0, "wind": 0.0})
    return pts(x)


def over_probability(adj: float, sd: float = 13.1, fair_over: float = None) -> float:
    """P(over): the market's own no-vig chance (0.5 if unknown) moved by what the adjustment shifts a normal miss of SD `sd`."""
    base = 0.5 if fair_over is None else float(fair_over)
    return min(0.97, max(0.03, base + (ND.cdf(adj / sd) - 0.5)))


def weather_inputs(card: dict, roof=None):
    """(roof, wind_mph, temp_f, retractable) from a context_data weather card, or Nones."""
    if not card:
        return roof, None, None, False
    if card.get("indoors"):
        return card.get("roof") or "dome", None, None, False
    wind = card.get("gust_mph") if card.get("wind_mph") is None else max(card["wind_mph"], 0.7 * (card.get("gust_mph") or 0))
    return card.get("roof") or "outdoors", wind, card.get("temp_f"), bool(card.get("retractable"))


def apply(games: list, weather: dict, model: dict = None) -> int:
    """
    Replace each game's total model with the market-anchored one, in place (the simulation's own total is kept as `sim_total`).
    `weather` is {game label: weather card}. -> games adjusted.
    """
    model = model or load()
    sd = float(model.get("sd", DEFAULT["sd"]))
    n = 0
    for g in games:
        line = g.get("total")
        if line is None:
            continue
        roof, wind, temp, retr = weather_inputs((weather or {}).get(g["game"]))
        adj = adjustment(model, roof, wind, temp, retr)
        g["sim_total"] = g.get("proj_total")
        over = over_probability(adj, sd, g.get("fair_over"))
        g["total_adj"] = round(adj, 2)
        g["predicted_total"] = round((line + adj) * 2) / 2
        g["proj_total"] = round(line + adj, 1)
        g["total_gap"] = round(g["predicted_total"] - line, 2)
        g["model_over"], g["over"], g["under"] = round(over, 4), round(over, 4), round(1 - over, 4)
        g["total_trust"] = 1.0
        g["total_basis"] = ("dome" if adj > 0.4 else "wind" if adj < -0.4 else "none")
        # the simulated team scores ran ~10 points high: rescale them to the new total, keeping the simulated split
        sh, sa = g.get("proj_home"), g.get("proj_away")
        if sh and sa and (sh + sa) > 0:
            k = g["proj_total"] / (sh + sa)
            g["proj_home"], g["proj_away"] = round(sh * k, 1), round(sa * k, 1)
        n += 1
    return n


def main():
    if "--write" not in sys.argv:
        print(json.dumps(load(), indent=1))
        return
    import nflreadpy as nfl
    sched = nfl.load_schedules(list(range(FIRST_SEASON, nfl.get_current_season() + 1))).to_pandas()
    m = fit(sched)
    FILE.write_text(json.dumps(m, indent=1), encoding="utf-8")
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
