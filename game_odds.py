"""
game_odds.py
============
Rolls the same per-player projections that price props up into team point totals, to price
game sides (spread) and totals the identical bottom-up way: simulate the players, sum to a
team score, compare the simulated distribution to the market's line.

TEAM POINTS FROM PROJECTIONS
-----------------------------
Team points are not simulated drive-by-drive; that would need modeling field position,
red-zone conversion, kicking, and defense/special-teams scores. Instead, points are
regressed directly on what player projections roll up into -- rushing yards, passing yards,
and offensive touchdowns (rushing + receiving; a passing TD is the same event as a receiver's
TD, so counting both would double it) -- fitted on 2021-25 team-games (see tune_props.py):

    points = 2.09 + 0.0297*rush_yds + 0.0156*pass_yds + 5.60*off_td + noise

R^2 = 0.81, MAE = 3.35 points -- a large improvement on a naive yards-per-point guess
(MAE 14.2 for the simpler "yards/45" proxy the auction draft board still uses). The
unexplained ~19% is mostly defensive/special-teams touchdowns, missed kicks, and 2-point
tries, none of which are simulated here. `noise` is drawn to match the real residual's shape
(mean 0, std 4.32, right-skewed -- a team can gain extra points from a pick-six but can't lose
points it already scored) via a shifted gamma.

CORRELATION
-----------
A team's players are not simulated independently. Each simulated game draws one shared
"game script" multiplier per team -- how well that specific simulated game went for its
offense -- applied to every one of that team's players before they're summed. Without this,
script-driven blowouts and shootouts are understated. This mirrors simulate.py's older,
simpler team-total function (the auction draft board), carried over here.

WHAT THIS DOES NOT KNOW
------------------------
No defense/special-teams scoring, no kicking accuracy, no 2-point tries, and it only includes
players a roster-selection heuristic judges relevant this week (recent usage). It also has not
yet been checked against real spread/total lines and results -- only the points formula itself
is validated. props_tracker.py logs and grades these picks the same way it does player props,
so that check accumulates over time.
"""

import numpy as np

import depth_chart
import game_context as gc
from props_model import MARKETS, environment_effect, fit_player

N_SIMS = 20000
# How much of the raw simulation's disagreement with the market is trusted, the same shrinkage
# idea props_model.py uses and for the same reason: summing many small-sample player fits (see
# TD_PRIOR_GAMES in props_model.py) into a team total compounds their noise, and early in a
# season a few red-hot or ice-cold games can swing a whole roster's projected total by a lot
# more than a real week-to-week swing. This constant is a judgment call, not a fitted value --
# there is no historical closing-line data to fit it against -- and should come down once
# props_tracker.py has logged enough graded game picks to check it.
GAME_LAMBDA_MAX = 0.20
GAME_LAMBDA_FULL_N = 40   # combined avg. n_eff across both rosters' fits needed for full trust
TEAM_PTS_COEF = {"rush_yds": 0.0297, "pass_yds": 0.0156, "off_td": 5.6035, "intercept": 2.0915}
RESID_STD = 4.32
RESID_SKEW = 0.82
SCRIPT_SD = 0.18          # per-team game-script spread, matching simulate.py's existing constant
SCRIPT_CLIP = (0.4, 1.8)
MIN_USAGE_GAMES = 3       # a contributor needs at least this many recent games to be simulated

# roster-selection heuristic: (position, usage column, how many to take, ranked by usage)
ROSTER_SLOTS = [("QB", "attempts", 1), ("RB", "carries", 3), ("WR", "targets", 4), ("TE", "targets", 2)]


def _resid_draw(n, rng):
    """Zero-mean noise matching the real points residual: mean 0, std RESID_STD, skew RESID_SKEW."""
    shape = (2 / RESID_SKEW) ** 2
    scale = RESID_STD / np.sqrt(shape)
    return rng.gamma(shape, scale, n) - shape * scale


def team_points(rush_yds, pass_yds, off_td, rng):
    """Vectorized over sims: rush_yds/pass_yds/off_td are already-scripted per-sim team totals."""
    n = len(rush_yds)
    c = TEAM_PTS_COEF
    base = c["intercept"] + c["rush_yds"] * rush_yds + c["pass_yds"] * pass_yds + c["off_td"] * off_td
    return np.clip(base + _resid_draw(n, rng), 0, None)


def _draw_scripted(mean, var, script, rng, discrete=False):
    """Normal draw (matching simulate.py's team-total approach) scaled by the shared script array."""
    draws = np.clip(rng.normal(mean, np.sqrt(max(var, 0)), len(script)) * script, 0, None)
    return np.round(draws) if discrete else draws


def simulate_team_offense(player_fits, script, rng, n):
    """
    player_fits: [{"rush_yd": fit|None, "pass_yd": fit|None, "td": fit|None}, ...] (a `fit` is
    props_model.fit_player's return dict). -> (rush_yds, pass_yds, off_td), each length n.
    """
    rush, pas, td = np.zeros(n), np.zeros(n), np.zeros(n)
    for pf in player_fits:
        if pf.get("rush_yd"):
            rush += _draw_scripted(pf["rush_yd"]["mean"], pf["rush_yd"]["var"], script, rng)
        if pf.get("pass_yd"):
            pas += _draw_scripted(pf["pass_yd"]["mean"], pf["pass_yd"]["var"], script, rng)
        if pf.get("td"):
            td += _draw_scripted(pf["td"]["mean"], pf["td"]["var"], script, rng, discrete=True)
    return rush, pas, td


def simulate_game_scores(home_fits, away_fits, rng, n=N_SIMS):
    """-> (home_points, away_points), each an array of length n."""
    script_h = np.clip(rng.normal(1.0, SCRIPT_SD, n), *SCRIPT_CLIP)
    script_a = np.clip(rng.normal(1.0, SCRIPT_SD, n), *SCRIPT_CLIP)
    rh, ph, th = simulate_team_offense(home_fits, script_h, rng, n)
    ra, pa, ta = simulate_team_offense(away_fits, script_a, rng, n)
    return team_points(rh, ph, th, rng), team_points(ra, pa, ta, rng)


def fits_n_eff(*fit_lists):
    """Average n_eff across every fitted stat in one or more team rosters, for the market blend's trust."""
    vals = [f["n_eff"] for fits in fit_lists for pf in fits for f in pf.values() if f]
    return float(np.mean(vals)) if vals else 0.0


def blend_game_prob(model_p, market_p, n_eff, lambda_max=None):
    """
    Shrink a simulated probability toward the market's, trusted more with more history behind
    it. `lambda_max` overrides GAME_LAMBDA_MAX once props_tracker.py's weekly refinement has
    enough real graded game picks to justify a different ceiling for this market.
    """
    if market_p is None:
        return model_p
    weight = (GAME_LAMBDA_MAX if lambda_max is None else lambda_max) * min(1.0, max(0.0, n_eff) / GAME_LAMBDA_FULL_N)
    return market_p + weight * (model_p - market_p)


def spread_total_probs(home_pts, away_pts, home_spread, total_line):
    """
    home_spread: negative = home favored (the Odds API's convention: the home team's own
    spread outcome). -> dict of cover/over probabilities plus the simulated projections.
    """
    margin = home_pts - away_pts
    total = home_pts + away_pts
    home_cover = float((margin > -home_spread).mean())
    home_push = float((margin == -home_spread).mean())
    over = float((total > total_line).mean())
    total_push = float((total == total_line).mean())
    return {
        "home_cover": home_cover, "home_push": home_push, "away_cover": max(0.0, 1 - home_cover - home_push),
        "over": over, "total_push": total_push, "under": max(0.0, 1 - over - total_push),
        "proj_home": float(home_pts.mean()), "proj_away": float(away_pts.mean()),
        "proj_margin": float(margin.mean()), "proj_total": float(total.mean()),
    }


# ---- roster selection + fitting --------------------------------------------------------
def select_offense(history, team, min_games=MIN_USAGE_GAMES, inactive_ids=()):
    """
    This team's likely contributors this week, by recent usage. -> [player_id, ...].
    `history` must have the columns build_history() produces. `inactive_ids` (Out/Doubtful/IR
    this week) are excluded before ranking, so a hurt starter's history doesn't crowd out the
    healthy backup actually playing -- without this, select_offense would keep picking last
    week's starter by usage alone even after he's ruled out.
    """
    recent = history[(history["team"] == team) & (~history["player_id"].isin(inactive_ids))]
    recent = recent.groupby("player_id", as_index=False).tail(4)
    usage = recent.groupby(["player_id", "position"]).agg(
        attempts=("attempts", "mean"), carries=("carries", "mean"), targets=("targets", "mean"),
        games=("attempts", "size")).reset_index()
    usage = usage[usage["games"] >= min_games]
    picks = []
    for pos, col, n in ROSTER_SLOTS:
        pool = usage[usage["position"] == pos].sort_values(col, ascending=False)
        picks += pool["player_id"].head(n).tolist()
    return picks


def build_team_fits(history, team, opp, mults, priors, calibration=None, game_total=None, team_margin=None,
                    inj_status=None):
    """
    Fit each selected player's rush_yd / pass_yd / td distributions -- opponent-adjusted,
    market-calibrated, and game-environment-adjusted using EACH market's own fitted
    coefficients (pass yards and rush yards respond to the game script differently, so this
    resolves environment_effect() separately per stat rather than sharing one function).
    `mults`/`priors` are keyed by market, as export_props.py already builds them; `calibration`
    is the object load_calibration() returns. `game_total`/`team_margin` (this team's own
    expected margin, + = favored) come from the real posted line for this game.
    -> [{"rush_yd": fit|None, "pass_yd": fit|None, "td": fit|None}, ...]
    """
    def env_fn_for(market):
        eff = environment_effect(calibration, market)
        if not eff or game_total is None or team_margin is None:
            return None
        return lambda g, w, e=eff, t=game_total, m=team_margin: gc.env_multiplier(g, w, t, m, e)

    inj_status = inj_status or {}
    inactive_ids = {pid for pid, status in inj_status.items() if status in depth_chart.OUT_STATUSES}
    fits = []
    for pid in select_offense(history, team, inactive_ids=inactive_ids):
        row = history[history["player_id"] == pid]
        if row.empty:
            continue
        pos = row["position"].iloc[-1]
        promo = depth_chart.promotion_multiplier(history, team, pos, pid, inj_status)
        block = {}
        for market, key, eligible in [
            ("player_pass_yds", "pass_yd", pos == "QB"),
            ("player_rush_yds", "rush_yd", pos in ("QB", "RB", "WR", "TE")),
        ]:
            if not eligible:
                continue
            m = mults.get(market, {}).get((opp, pos), 1.0)
            calib = (calibration or {}).get("calibration", {}).get(market)
            block[key] = fit_player(history, pid, MARKETS[market], m * promo, None, calib, env_fn_for(market))
        m = mults.get("player_anytime_td", {}).get((opp, pos), 1.0)
        block["td"] = fit_player(history, pid, MARKETS["player_anytime_td"], m * promo, priors.get("player_anytime_td"))
        if any(block.values()):
            fits.append(block)
    return fits
