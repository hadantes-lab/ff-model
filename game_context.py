"""
game_context.py
===============
Game environment: the betting total and spread for the game a prop is in.

A player's recent stats came from games with their own totals and spreads. If this week's
game is projected to be a shootout (total 50+), passing volume and yardage should be higher
than his average game; a 38-point slog should be lower; a big favorite runs more and passes
less. The market prices all of that; a game-log model does not, unless we tell it.

HOW MUCH?  Measured, not guessed.  For each stat, regress a player's game-to-game deviation
from his own average on how far that game's total (log) and expected margin were from the
totals/margins of his own games (nflverse schedules carry historical closing lines):

    relative_dev = a * ln(total / his usual total) + b * (margin - his usual margin)

Sanity anchors from 2023-25 (within team / within QB): a higher total means more team
passing yards (elasticity about +0.6) and fewer rushing attempts (about -0.3) -- more passing,
less rushing. Coefficients are then shrunk by their own statistical strength (t^2 / (1 + t^2)),
so a stat with no reliable relationship gets no adjustment.

Applied as a multiplier on the projection mean, capped at +/-15%.
"""

import numpy as np
import pandas as pd

from props_model import ROLE_FLOOR

MULT_CLIP = (0.85, 1.15)
MIN_PLAYER_GAMES = 8
MIN_ROWS = 500


def load_game_lines(seasons) -> pd.DataFrame:
    """Historical/scheduled lines from nflverse: total and spread (positive = home favored)."""
    import nflreadpy as nfl

    s = nfl.load_schedules(list(seasons)).to_pandas()
    return s[["game_id", "season", "week", "home_team", "away_team", "total_line", "spread_line"]]


def attach_context(history: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """Add `total` and `margin` (the player's team's expected margin; + = favored) to each player-game."""
    h = history.merge(lines[["game_id", "home_team", "total_line", "spread_line"]], on="game_id", how="left")
    h["total"] = h["total_line"]
    h["margin"] = np.where(h["team"] == h["home_team"], h["spread_line"], -h["spread_line"])
    return h.drop(columns=["total_line", "spread_line", "home_team"])


def implied_team_totals(total, margin):
    """(team, opponent) implied points from a game total and the team's expected margin."""
    return (total + margin) / 2, (total - margin) / 2


def fit_context_effects(history_ctx: pd.DataFrame, markets: dict) -> dict:
    """
    {market: {"a": total elasticity, "b": margin effect per point, "n", "t_a", "t_b"}} after
    significance shrinkage. `history_ctx` comes from attach_context; `markets` is MARKETS.
    """
    from props_model import stat_series

    h = history_ctx.dropna(subset=["total", "margin"]).copy()
    out = {}
    for m, spec in markets.items():
        if spec.kind == "td":
            continue
        h["_v"] = stat_series(h, spec)
        d = h.groupby("player_id").filter(lambda x: len(x) >= MIN_PLAYER_GAMES and x["_v"].mean() >= ROLE_FLOOR[m])
        d = d.reset_index(drop=True)
        if len(d) < MIN_ROWS:
            continue
        g = d.groupby("player_id")
        rel = (d["_v"] / g["_v"].transform("mean") - 1).to_numpy()
        dlt = (np.log(d["total"]) - g["total"].transform(lambda s: np.log(s).mean())).to_numpy()
        dm = (d["margin"] - g["margin"].transform("mean")).to_numpy()
        X = np.c_[dlt, dm]
        coef, *_ = np.linalg.lstsq(X, rel, rcond=None)
        resid = rel - X @ coef
        se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * resid.var())
        t = coef / se
        w = t ** 2 / (1 + t ** 2)                      # weak evidence -> pulled to zero
        out[m] = {"a": float(coef[0] * w[0]), "b": float(coef[1] * w[1]), "n": int(len(d)),
                  "t_a": float(t[0]), "t_b": float(t[1])}
    return out


def env_multiplier(player_games: pd.DataFrame, weights, total_now, margin_now, effects) -> dict:
    """
    Multiplier on a player's mean for THIS game's environment relative to the environment of the
    games his recent stats came from (weighted the same way as his projection).
    """
    if effects is None or total_now is None or margin_now is None:
        return {"mult": 1.0}
    g = player_games.assign(_w=np.asarray(weights, dtype=float)).dropna(subset=["total", "margin"])
    if g.empty:
        return {"mult": 1.0}
    avg_ln_total = float(np.average(np.log(g["total"]), weights=g["_w"]))
    avg_margin = float(np.average(g["margin"], weights=g["_w"]))
    mult = 1 + effects["a"] * (np.log(total_now) - avg_ln_total) + effects["b"] * (margin_now - avg_margin)
    return {
        "mult": float(np.clip(mult, *MULT_CLIP)),
        "total": float(total_now), "margin": float(margin_now),
        "avg_total": float(np.exp(avg_ln_total)), "avg_margin": avg_margin,
    }
