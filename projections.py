"""
projections.py
==============
Builds per-player stat projections from real nflverse play-by-play data.

This is the FOUNDATION of the whole model. Everything downstream — prop
probabilities, spreads, totals — is built by simulating these projections.

A "projection" here is NOT a single number. It's a mean AND a spread (volatility),
because betting is entirely about the distribution of outcomes, not the average.
Two players averaging 60 rush yards are completely different bets if one ranges
40-80 and the other ranges 5-140.

HOW IT WORKS
------------
1. Pull recent play-by-play (real data, via nflreadpy).
2. For each player, compute their per-game history of each stat.
3. Weight recent games more heavily (form matters).
4. Adjust for opponent strength (defense faced) and expected volume.
5. Output mean + standard deviation per stat -> feeds the simulator.

Run:  python projections.py
"""

import numpy as np
import pandas as pd

# nflreadpy is the modern Python interface to nflverse data.
# Install with:  pip install nflreadpy
try:
    import nflreadpy as nfl
except ImportError:
    nfl = None
    print("[!] nflreadpy not installed. Run:  pip install nflreadpy")
    print("    (The code below shows exactly how the real data flows in.)")


# ---- CONFIG -------------------------------------------------------------
SEASONS = [2024, 2025]     # seasons of history to pull
RECENCY_HALFLIFE = 4       # games; recent form weighted, older games decay
MIN_GAMES = 3              # need at least this many games to project a player


# ---- STATS WE PROJECT ---------------------------------------------------
# Mapped to the raw play-by-play columns they're derived from.
SKILL_STATS = {
    "rush_yards":   "rushing_yards",
    "rush_att":     "rush_attempt",
    "rush_td":      "rush_touchdown",
    "rec_yards":    "receiving_yards",
    "receptions":   "complete_pass",   # when player is the receiver
    "targets":      "pass_attempt",    # when player is targeted
    "rec_td":       "pass_touchdown",  # receiving TD
}


def load_pbp(seasons=SEASONS):
    """Pull real play-by-play. Millions of rows, one call."""
    if nfl is None:
        raise RuntimeError("nflreadpy required. pip install nflreadpy")
    pbp = nfl.load_pbp(seasons).to_pandas()
    # keep only real offensive plays
    pbp = pbp[(pbp["pass_attempt"] == 1) | (pbp["rush_attempt"] == 1)].copy()
    return pbp


def recency_weights(n_games, halflife=RECENCY_HALFLIFE):
    """
    Most recent game = weight 1.0, decaying by halflife.
    A player's last 4 games tell you more than games from last season.
    """
    ages = np.arange(n_games)[::-1]          # 0 = most recent
    return 0.5 ** (ages / halflife)


def player_game_logs(pbp):
    """
    Collapse play-by-play into per-player, per-game stat lines.
    Returns a tidy frame: one row per (player, game, stat).
    """
    rows = []

    # --- rushing ---
    rush = pbp[pbp["rush_attempt"] == 1]
    g = rush.groupby(["rusher_player_id", "rusher_player_name", "game_id", "week", "posteam", "defteam"])
    for (pid, name, gid, wk, team, opp), grp in g:
        rows.append(dict(pid=pid, name=name, game=gid, week=wk, team=team, opp=opp,
                         rush_yards=grp["rushing_yards"].sum(),
                         rush_att=len(grp),
                         rush_td=grp["rush_touchdown"].sum()))

    # --- receiving ---
    rec = pbp[pbp["pass_attempt"] == 1]
    g = rec.groupby(["receiver_player_id", "receiver_player_name", "game_id", "week", "posteam", "defteam"])
    for (pid, name, gid, wk, team, opp), grp in g:
        if pid is None or (isinstance(pid, float) and np.isnan(pid)):
            continue
        rows.append(dict(pid=pid, name=name, game=gid, week=wk, team=team, opp=opp,
                         rec_yards=grp["receiving_yards"].sum(),
                         receptions=grp["complete_pass"].sum(),
                         targets=len(grp),
                         rec_td=grp["pass_touchdown"].sum()))

    logs = pd.DataFrame(rows).fillna(0)
    # merge rushing + receiving rows for same player-game
    logs = logs.groupby(["pid", "name", "game", "week", "team", "opp"], as_index=False).sum()
    return logs


def defense_adjustments(logs):
    """
    How much does each defense suppress or inflate a stat vs league average?
    Returns a multiplier per (defense, stat). >1 = gives up more than average.
    This is the 'opponent strength' the pros insist you must include.
    """
    adj = {}
    for stat in ["rush_yards", "rec_yards", "receptions", "rush_td", "rec_td"]:
        if stat not in logs:
            continue
        league_avg = logs[stat].mean()
        if league_avg == 0:
            continue
        by_def = logs.groupby("opp")[stat].mean()
        adj[stat] = (by_def / league_avg).to_dict()
    return adj


def project_player(player_logs, opp, def_adj):
    """
    Turn one player's game history into a projection: mean + std per stat.
    - recency-weighted mean (recent form)
    - opponent defense adjustment
    - std captures their real volatility (this is what makes it a *distribution*)
    """
    player_logs = player_logs.sort_values("week")
    n = len(player_logs)
    if n < MIN_GAMES:
        return None

    w = recency_weights(n)
    proj = {}
    for stat in ["rush_yards", "rush_att", "rush_td", "rec_yards", "receptions", "targets", "rec_td"]:
        if stat not in player_logs:
            continue
        vals = player_logs[stat].to_numpy(dtype=float)
        # weighted mean
        mean = np.average(vals, weights=w)
        # weighted std (volatility) — the crucial second number
        var = np.average((vals - mean) ** 2, weights=w)
        std = np.sqrt(var)
        # opponent adjustment
        mult = def_adj.get(stat, {}).get(opp, 1.0)
        mean *= mult
        proj[stat] = {"mean": round(float(mean), 2),
                      "std": round(float(max(std, mean * 0.15)), 2)}  # floor on std
    return proj


def build_projections(opponent_map=None):
    """
    Main entry: returns {player_name: {stat: {mean, std}}} for the upcoming week.
    opponent_map: optional {player_name: opponent_team} for this week's matchups.
    """
    pbp = load_pbp()
    logs = player_game_logs(pbp)
    def_adj = defense_adjustments(logs)

    projections = {}
    for (pid, name), plog in logs.groupby(["pid", "name"]):
        # default: use most recent opponent if no map supplied
        opp = (opponent_map or {}).get(name, plog.sort_values("week")["opp"].iloc[-1])
        proj = project_player(plog, opp, def_adj)
        if proj:
            projections[name] = {"opp": opp, "stats": proj}
    return projections


if __name__ == "__main__":
    if nfl is None:
        print("\nInstall nflreadpy first, then re-run to pull live data.\n")
    else:
        print("Pulling real nflverse play-by-play... (first run downloads, then caches)")
        projs = build_projections()
        print(f"\nBuilt projections for {len(projs)} players.\n")
        # show a few examples
        for name in list(projs)[:5]:
            print(name, "vs", projs[name]["opp"])
            for stat, d in projs[name]["stats"].items():
                print(f"   {stat:12s} mean={d['mean']:6.1f}  std={d['std']:5.1f}")
            print()
