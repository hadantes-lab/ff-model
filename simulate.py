"""
simulate.py
===========
The Monte Carlo engine. Takes player projections (mean + std per stat) and
simulates thousands of game-worlds, then reads probabilities off the results.

THIS IS THE HEART OF THE MODEL. Same engine does three jobs:
  1. PLAYER PROPS  — simulate one player's stat, count how often it beats the line
  2. TOTALS        — sum all players into a game total, compare to the O/U line
  3. SPREADS       — team A points minus team B points, compare to the spread

The key insight (why props feed spreads): a team's score is just the sum of its
players' simulated stats converted to points. Simulate the players, and the
spread falls out for free.

CORRELATION NOTE: real games are correlated — when a QB throws for 350, his WRs
eat too. Independent simulation understates blowouts and shootouts. We add a
simple per-team "game script" factor each sim to capture this. It's a
first-order fix, not a full covariance matrix, but it matters.

Run:  python simulate.py
"""

import numpy as np

N_SIMS = 20000     # simulations per event; 10-25k is the industry norm


def american_to_prob(odds):
    """Convert American odds to implied probability (includes the vig)."""
    if odds < 0:
        return -odds / (-odds + 100)
    return 100 / (odds + 100)


def remove_vig(prob_over, prob_under):
    """
    Books bake in a margin (vig). The two sides sum to >100%.
    Normalize so they sum to 1 — this is the market's TRUE implied probability,
    which is what you compare your model against.
    """
    total = prob_over + prob_under
    return prob_over / total, prob_under / total


def draw_stat(mean, std, discrete=False, n=N_SIMS, script=None):
    """
    Draw n simulated values of a stat.
    - continuous stats (yards): normal, clamped at 0
    - discrete stats (TDs, receptions): normal then rounded, clamped at 0
    - script: optional per-sim multiplier array (game-script correlation)
    """
    if mean <= 0:
        return np.zeros(n)
    draws = np.random.normal(mean, std, n)
    if script is not None:
        draws *= script
    draws = np.clip(draws, 0, None)
    if discrete:
        draws = np.round(draws)
    return draws


DISCRETE = {"rush_td", "rec_td", "receptions", "targets", "rush_att"}


def simulate_prop(player_proj, stat, line, n=N_SIMS):
    """
    Prop probability. Returns model's P(over) and P(under) plus the sim mean.
    """
    d = player_proj["stats"].get(stat)
    if d is None:
        return None
    draws = draw_stat(d["mean"], d["std"], discrete=(stat in DISCRETE), n=n)
    p_over = float((draws > line).mean())
    return {
        "stat": stat,
        "line": line,
        "sim_mean": round(float(draws.mean()), 2),
        "p_over": round(p_over, 4),
        "p_under": round(1 - p_over, 4),
    }


def stat_to_points(stat, value):
    """Convert a raw stat outcome to fantasy-independent SCORING points.
    Used for team-score assembly (spreads/totals). Rough NFL scoring:
    ~ a TD = 6 (+ where relevant), yardage doesn't score directly but proxies drives.
    For team scoring we approximate points from TDs + field-goal proxy."""
    if stat in ("rush_td", "rec_td"):
        return value * 6.0
    return 0.0  # yards don't directly score; TDs carry team points here


def simulate_game(team_a_players, team_b_players, spread_line, total_line, n=N_SIMS):
    """
    Simulate a full game by summing player outcomes into team scores.
    Adds a per-sim game-script factor per team for correlation.

    team_x_players: list of player_proj dicts (from projections.py)
    spread_line: points, negative = team_a favored (e.g. -3.5)
    total_line: combined points O/U

    Returns spread + total probabilities vs the given lines.
    """
    # game-script correlation: one shared multiplier per team per sim
    script_a = np.random.normal(1.0, 0.18, n)
    script_b = np.random.normal(1.0, 0.18, n)
    script_a = np.clip(script_a, 0.4, 1.8)
    script_b = np.clip(script_b, 0.4, 1.8)

    def team_points(players, script):
        pts = np.zeros(n)
        # base scoring from simulated TDs
        for p in players:
            for stat in ("rush_td", "rec_td"):
                d = p["stats"].get(stat)
                if d:
                    pts += draw_stat(d["mean"], d["std"], discrete=True, n=n, script=script) * 6.0
        # add a yardage-driven field goal / drive proxy so scores are realistic
        yards = np.zeros(n)
        for p in players:
            for stat in ("rush_yards", "rec_yards"):
                d = p["stats"].get(stat)
                if d:
                    yards += draw_stat(d["mean"], d["std"], n=n, script=script)
        pts += yards / 45.0   # ~ every 45 team yards ≈ 1 additional point (drives/FGs)
        return pts

    a = team_points(team_a_players, script_a)
    b = team_points(team_b_players, script_b)

    margin = a - b            # positive = team_a wins by this
    total = a + b

    # spread_line negative means team_a favored by that many; a covers if margin > -spread
    a_covers = float((margin > -spread_line).mean())
    over = float((total > total_line).mean())

    return {
        "proj_a": round(float(a.mean()), 1),
        "proj_b": round(float(b.mean()), 1),
        "proj_margin": round(float(margin.mean()), 1),
        "proj_total": round(float(total.mean()), 1),
        "spread_line": spread_line,
        "p_a_covers": round(a_covers, 4),
        "p_b_covers": round(1 - a_covers, 4),
        "total_line": total_line,
        "p_over": round(over, 4),
        "p_under": round(1 - over, 4),
    }


def find_edge(model_prob, american_odds):
    """
    The payoff step. Compare model probability to the market's vig-free implied
    probability. Positive edge = model thinks it's more likely than the price.

    Returns edge in percentage points and expected value per $1 staked.
    """
    implied = american_to_prob(american_odds)
    edge = model_prob - implied
    # EV per $1: p*payout - (1-p)*1
    if american_odds < 0:
        payout = 100 / -american_odds
    else:
        payout = american_odds / 100
    ev = model_prob * payout - (1 - model_prob)
    return {
        "model_prob": round(model_prob, 4),
        "implied_prob": round(implied, 4),
        "edge_pts": round(edge * 100, 2),
        "ev_per_dollar": round(ev, 4),
        "bet": edge > 0.02,   # only flag if edge clears ~2pts (covers noise)
    }


if __name__ == "__main__":
    # DEMO with fake projections so you can see the shapes without live data.
    np.random.seed(42)
    demo_player = {"stats": {
        "rush_yards": {"mean": 68, "std": 22},
        "rush_td":    {"mean": 0.6, "std": 0.7},
    }}

    print("=== PROP DEMO: rush yards o/u 64.5 ===")
    r = simulate_prop(demo_player, "rush_yards", 64.5)
    print(r)
    print("edge vs -115 over:", find_edge(r["p_over"], -115))

    print("\n=== GAME DEMO: spread -3.5, total 47.5 ===")
    team_a = [demo_player, demo_player, demo_player]  # stand-ins
    team_b = [demo_player, demo_player]
    g = simulate_game(team_a, team_b, spread_line=-3.5, total_line=47.5)
    for k, v in g.items():
        print(f"  {k}: {v}")
