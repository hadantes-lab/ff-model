"""
depth_chart.py
===============
Detects when a player is stepping into a bigger role than his own recent history reflects --
most commonly a backup starting because the usual starter is out or inactive -- and produces a
multiplier that scales his volume-driven projection toward what the OFFENSE, not the
individual, has recently done at that position. Without this, a backup QB who has thrown 12
passes all season in mop-up duty gets projected off HIS OWN rate even on a night he is the
confirmed starter, which understates him by a lot.

HOW IT WORKS
------------
1. usage_rate() -- the team's recency-weighted average per-game USAGE at a position (pass
   attempts for QB, carries for RB, targets for WR/TE), regardless of who it was.
2. For one player, compare HIS OWN recency-weighted usage rate to that team rate.
3. He is judged the newly-promoted starter only if every teammate at his position with MORE
   recent usage than him is Out/Doubtful/IR this week (from the injury report). If a
   more-used, presumably-healthy teammate exists, nothing changes -- he is still the backup.
4. If promoted, his volume projection is scaled up toward the team's rate: never down, and
   capped (PROMOTION_CAP) since even a confirmed new starter won't instantly play like a
   whole offense's long-run habits, and PROMOTION_TRUST holds back some of that gap too.

This does NOT touch per-touch efficiency (yards per carry, catch rate, TD rate) -- those still
come from his own history, or the position-average prior fit_player() already regresses thin
samples toward. It only corrects his projected VOLUME, applied as one more multiplier
alongside opponent strength and the coverage matchup, the same way this codebase already
composes adjustments.
"""

import numpy as np

USAGE_COL = {"QB": "attempts", "RB": "carries", "WR": "targets", "TE": "targets"}
OUT_STATUSES = ("Out", "Doubtful", "IR")
PROMOTION_CAP = 3.0     # a new starter is never projected at more than 3x his own recent rate
PROMOTION_TRUST = 0.7   # ...and only this much of the gap to the team's rate is closed
MIN_TEAMMATE_GAMES = 3  # a teammate's usage only overrides this player's "starter" status with this much history


def _weighted_rate(values, halflife):
    from props_model import recency_weights
    if len(values) == 0:
        return 0.0
    return float(np.average(values, weights=recency_weights(len(values), halflife)))


def usage_rate(history, team, position, halflife=None, max_games=8):
    """The team's recency-weighted average per-game usage at a position -> (rate, n_games)."""
    from props_model import HALFLIFE

    col = USAGE_COL.get(position)
    if col is None:
        return 0.0, 0
    df = history[(history["team"] == team) & (history["position"] == position)]
    if df.empty or col not in df:
        return 0.0, 0
    per_game = (df.groupby(["season", "week"])[col].sum()
               .reset_index().sort_values(["season", "week"]).tail(max_games))
    if per_game.empty:
        return 0.0, 0
    return _weighted_rate(per_game[col].to_numpy(float), halflife or HALFLIFE), len(per_game)


def player_usage_rate(history, player_id, position, halflife=None, max_games=None):
    """This player's own recency-weighted average per-game usage -> (rate, n_games)."""
    from props_model import HALFLIFE, MAX_GAMES

    col = USAGE_COL.get(position)
    if col is None:
        return 0.0, 0
    g = history[history["player_id"] == player_id].tail(max_games or MAX_GAMES)
    if g.empty or col not in g:
        return 0.0, 0
    vals = g[col].fillna(0).to_numpy(float)
    return _weighted_rate(vals, halflife or HALFLIFE), len(g)


def is_promoted_starter(history, team, position, player_id, inj_status_by_player):
    """
    True iff at least one teammate at his position has more recent usage than he does, AND
    every one of those more-used teammates is Out/Doubtful/IR this week.

    The "at least one" requirement is what makes this safe: without it, the team's single
    highest-usage player at a position would always vacuously "have no more-used teammate" and
    get flagged as promoted whenever any backup happens to be hurt. It also avoids a fixed
    share-of-team-usage threshold, which sounds like a cleaner check but isn't -- a QB's
    starter usually takes >90% of the team's attempts (a sharp signal), while RB carries or
    WR/TE targets routinely split 40/30/20 across a healthy, uninjured group, so any single
    "already a starter" percentage misreads a normal committee as a promotion (verified against
    a real slate before shipping: this exact bug flagged Chicago's starting RB and WRs as
    "promoted" with no RB/WR/TE actually listed as Out that week).
    """
    if position not in USAGE_COL:
        return False
    own_rate, own_n = player_usage_rate(history, player_id, position)
    if own_n == 0:
        return False
    teammates = history[(history["team"] == team) & (history["position"] == position)
                        & (history["player_id"] != player_id)]["player_id"].unique()
    found_more_used = False
    for tm in teammates:
        tm_rate, tm_n = player_usage_rate(history, tm, position)
        if tm_n < MIN_TEAMMATE_GAMES or tm_rate <= own_rate:
            continue                                  # not more-used, or too little history to judge
        found_more_used = True
        if inj_status_by_player.get(tm) not in OUT_STATUSES:
            return False                               # a more-used, presumably-healthy teammate exists
    return found_more_used


def promotion_multiplier(history, team, position, player_id, inj_status_by_player):
    """
    -> 1.0 unless this player is judged a newly-promoted starter (see is_promoted_starter),
    in which case a multiplier >= 1.0 scaling his volume toward the team's recent rate.
    """
    if not is_promoted_starter(history, team, position, player_id, inj_status_by_player):
        return 1.0
    own_rate, _ = player_usage_rate(history, player_id, position)
    team_rate, team_n = usage_rate(history, team, position)
    if team_n == 0 or own_rate <= 0:
        return 1.0
    ratio = min(team_rate / own_rate, PROMOTION_CAP)
    return float(max(1.0, 1 + PROMOTION_TRUST * (ratio - 1)))
