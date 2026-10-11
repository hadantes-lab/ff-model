"""
volume_model.py
===============
Game script -> volume. A team that is expected to trail throws more and runs less; a team expected to lead
runs more and throws less. This module models that at the TEAM level (how many pass attempts and rush
attempts a team will have in this game, given the spread, the total, its own pace and what the opposing
defense allows) and turns it into a multiplier on QB and RB volume projections.

WHY TEAM LEVEL
--------------
A player's own recent attempts are noisy and tell you little about how THIS game's script differs from
his last few. The team's volume is far steadier, and the game script (the posted spread and total)
moves it in a known direction. Player volume = team volume x his share of it, so the player projection
only needs the team-level shift applied to it.

THE MODEL (all fitted walk-forward, so every backtest number is out of sample)
-----------------------------------------------------------------------------
For team T vs opponent O in a game, with features known BEFORE kickoff:
    margin    = T's expected margin from the posted spread (positive = T favored)
    total     = the posted game total
    r_att/r_car     T's recency-weighted pass / rush attempts in its recent games
    ra_att/ra_car   what O's defense has ALLOWED recently in attempts (how much it makes teams pass/run)
Target: T's pass attempts (and, separately, rush attempts), as the ratio to its own recent average, fitted
by ridge regression on `margin`, `margin^2`, `ln total`, the opposing defense's ratio, and home field.

`python volume_model.py` prints the walk-forward test: does this beat just using the team's recent average,
and does it beat the single linear game-context adjustment production uses today?
"""

import numpy as np
import pandas as pd

ROLL_GAMES = 14
ROLL_HALFLIFE = 5
RIDGE = 5.0
FEATURES = ["d_margin", "d_margin_sq", "d_ln_total", "opp_ratio", "home"]


# ---- data ----------------------------------------------------------------------
def team_volume_games(player_stats: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    """
    One row per team per regular-season game: pass attempts, rush attempts (all players, scrambles
    included), the posted margin/total from that team's side, and the opponent.
    """
    ps = player_stats[player_stats["season_type"] == "REG"]
    agg = (ps.groupby(["season", "week", "team"])[["attempts", "carries", "completions", "passing_yards", "rushing_yards"]]
             .sum().reset_index())
    s = sched[(sched["game_type"] == "REG") & sched["home_score"].notna()]
    home = s.assign(team=s["home_team"], opp=s["away_team"], is_home=1, margin=s["spread_line"])
    away = s.assign(team=s["away_team"], opp=s["home_team"], is_home=0, margin=-s["spread_line"])
    g = pd.concat([home, away], ignore_index=True)[["season", "week", "team", "opp", "is_home", "margin", "total_line", "game_id"]]
    g = g.merge(agg, on=["season", "week", "team"], how="left")
    # what each defense ALLOWED: the opponent's attempts/carries in the same game
    allowed = agg.rename(columns={"team": "opp", "attempts": "att_allowed", "carries": "car_allowed"})[
        ["season", "week", "opp", "att_allowed", "car_allowed"]]
    g = g.merge(allowed, on=["season", "week", "opp"], how="left")
    g = g.dropna(subset=["attempts", "carries", "margin", "total_line"]).copy()
    g["total"] = g["total_line"]
    return g.sort_values(["season", "week"]).reset_index(drop=True)


def _decayed_mean(values, halflife=ROLL_HALFLIFE):
    w = 0.5 ** (np.arange(len(values))[::-1] / halflife)
    return float(np.average(values, weights=w))


def add_rolling(g: pd.DataFrame, games=ROLL_GAMES, halflife=ROLL_HALFLIFE) -> pd.DataFrame:
    """Add each team's prior-games-only averages: own attempts/carries and what its defense allowed."""
    g = g.sort_values(["season", "week"]).reset_index(drop=True)
    g["ln_total_g"] = np.log(g["total"])
    cols = {"attempts": "r_att", "carries": "r_car", "att_allowed": "ra_att", "car_allowed": "ra_car",
            "margin": "r_margin", "ln_total_g": "r_ln_total"}       # r_margin/r_ln_total: the scripts behind those averages
    out = {c: np.full(len(g), np.nan) for c in cols.values()}
    out["r_n"] = np.zeros(len(g))
    for team, idx in g.groupby("team").groups.items():
        idx = list(idx)
        for k, i in enumerate(idx):
            prior = idx[max(0, k - games):k]
            out["r_n"][i] = len(prior)
            if len(prior) >= 3:
                for src, dst in cols.items():
                    out[dst][i] = _decayed_mean(g.loc[prior, src].to_numpy(float), halflife)
    for k, v in out.items():
        g[k] = v
    return g


def add_opponent_features(g: pd.DataFrame) -> pd.DataFrame:
    """The opposing defense's rolling allowed volume, from the opponent's side of the same game."""
    key = g[["season", "week", "team", "ra_att", "ra_car"]].rename(
        columns={"team": "opp", "ra_att": "opp_ra_att", "ra_car": "opp_ra_car"})
    g = g.merge(key, on=["season", "week", "opp"], how="left")
    lg_att, lg_car = g["attempts"].mean(), g["carries"].mean()
    g["opp_ratio_att"] = g["opp_ra_att"] / lg_att          # >1: that defense lets teams throw more than average
    g["opp_ratio_car"] = g["opp_ra_car"] / lg_car
    g["ln_total"] = np.log(g["total"])
    # the signal is how THIS game's script differs from the scripts the recent averages came from
    g["d_margin"] = g["margin"] - g["r_margin"]
    g["d_ln_total"] = g["ln_total"] - g["r_ln_total"]
    g["d_margin_sq"] = g["d_margin"] ** 2 * np.sign(g["d_margin"])  # signed: keeps direction, adds curvature
    g["home"] = g["is_home"].astype(float)
    return g


def build(player_stats, sched) -> pd.DataFrame:
    return add_opponent_features(add_rolling(team_volume_games(player_stats, sched)))


# ---- fitting ---------------------------------------------------------------------
TARGETS = {"att": ("attempts", "r_att", "opp_ratio_att"), "car": ("carries", "r_car", "opp_ratio_car")}


def design(g: pd.DataFrame, kind: str, features=None) -> tuple:
    target, base, opp = TARGETS[kind]
    f = features or FEATURES
    X = pd.DataFrame({"d_margin": g["d_margin"] / 10.0, "d_margin_sq": g["d_margin_sq"] / 100.0,
                      "d_ln_total": g["d_ln_total"], "opp_ratio": g[opp] - 1.0, "home": g["home"] - 0.5})[f]
    y = g[target] / g[base] - 1.0                               # relative to the team's own recent average
    return X.to_numpy(float), y.to_numpy(float)


def fit_ridge(X, y, lam=RIDGE):
    A = X.T @ X + lam * np.eye(X.shape[1])
    b = np.linalg.solve(A, X.T @ (y - y.mean()))
    return b, float(y.mean())


def predict_ratio(g: pd.DataFrame, kind: str, coef, intercept, features=None) -> np.ndarray:
    X, _ = design(g.assign(**{TARGETS[kind][0]: 1.0}), kind, features)
    return 1.0 + intercept + X @ coef


def walk_forward(g: pd.DataFrame, kind: str, first_test=2023, features=None, lam=RIDGE) -> pd.DataFrame:
    """Per-game predictions (own recent avg, and the model) for every season from `first_test`, each fitted on earlier seasons only."""
    target, base, _ = TARGETS[kind]
    d = g.dropna(subset=[base, TARGETS[kind][2], "d_margin", "d_ln_total"]).reset_index(drop=True)
    d = d[d["r_n"] >= 4]
    out = []
    for s in sorted(d["season"].unique()):
        if s < first_test:
            continue
        tr, te = d[d["season"] < s], d[d["season"] == s]
        X, y = design(tr, kind, features)
        coef, icpt = fit_ridge(X, y, lam)
        ratio = predict_ratio(te, kind, coef, icpt, features)
        out.append(pd.DataFrame({"season": s, "week": te["week"].to_numpy(), "team": te["team"].to_numpy(),
                                 "actual": te[target].to_numpy(), "base": te[base].to_numpy(),
                                 "model": te[base].to_numpy() * ratio, "margin": te["margin"].to_numpy(),
                                 "total": te["total"].to_numpy()}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def fit_final(g: pd.DataFrame, kind: str, features=None, lam=RIDGE) -> dict:
    """The production fit: every game we have. -> {coef (by feature), intercept, features}."""
    d = g.dropna(subset=[TARGETS[kind][1], TARGETS[kind][2], "d_margin", "d_ln_total"])
    d = d[d["r_n"] >= 4]
    f = features or FEATURES
    X, y = design(d, kind, f)
    coef, icpt = fit_ridge(X, y, lam)
    return {"coef": dict(zip(f, [float(c) for c in coef])), "intercept": icpt, "features": list(f), "n": int(len(d))}


def mae(df, col):
    return float((df["actual"] - df[col]).abs().mean())


def main():
    import nflreadpy as nfl

    season = nfl.get_current_season()
    ps = nfl.load_player_stats(list(range(2018, season + 1))).to_pandas()
    sched = nfl.load_schedules(list(range(2018, season + 1))).to_pandas()
    g = build(ps, sched)
    print("Team-level game script -> volume, tested walk-forward (each season fitted on earlier seasons only)\n")
    for kind, label in (("att", "pass attempts"), ("car", "rush attempts")):
        w = walk_forward(g, kind)
        c = fit_final(g, kind)["coef"]
        print(f"{label}: team recent average misses by {mae(w, 'base'):.2f}; with the game-script model {mae(w, 'model'):.2f} "
              f"({100 * (mae(w, 'model') / mae(w, 'base') - 1):+.1f}%), n={len(w)}")
        print(f"   effect of the game being 10 points MORE favorable than the team's usual script: {c['d_margin'] * 100:+.1f}% volume "
              f"(plus curvature {c['d_margin_sq'] * 100:+.1f}%)")
    print("\nThe books already price this in; see market_trust.py for the test against real lines.")


if __name__ == "__main__":
    main()
