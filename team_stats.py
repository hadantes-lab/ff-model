"""
team_stats.py
=============
Team-level pace and efficiency stats, one row per team per game, built from nflverse
play-by-play -- the raw material for checking which team stats actually relate to game totals
(and sides), and for showing each side's profile on the Game Lines page.

STATS (all per game, offense unless marked "allowed" = what the opponent's offense did):
  plays            pass + run plays (sacks count as pass plays; kneels, spikes, penalties-only
                   "no play" rows and special teams are excluded)
  yards, ypp       net yards (sack yardage included) and yards per play
  first_downs      first downs gained on those plays
  third_att/conv   third-down attempts and conversions -> third_rate
  top_sec          time of possession in seconds (summed drive clocks); sec_per_play = top/plays
  neutral_*        pass rate and mean expected-pass probability (nflverse `xpass`) when the
                   offense is tied or leading, quarters 1-3 (the user's "neutral" definition, minus
                   the 4th quarter, when leading teams run the clock rather than playing a neutral game)
  proe             pass rate over expected, same neutral situations (percentage points)
  *_allowed        the same figures from the opponent's side of the ball (the defense's results)

PRE-GAME FEATURES
-----------------
`rolling_features` gives, for every team-game, a recency-weighted average of that team's
PRIOR games only (up to ROLL_GAMES, spanning seasons, newest weighted most) -- so a feature
for week 5 never contains week 5. That is the discipline that keeps any correlation found
from being "the answer leaked into the question".

"EXPECTED" YARDS PER PLAY
-------------------------
For a matchup, a side's expected yards per play is its offense's rate plus how much its opponent's
defense differs from league average (off + opp_def_allowed - league), and expected plays
average its own pace with the opponent's plays-allowed. Expected yards = plays x ypp.
"""

import numpy as np
import pandas as pd

ROLL_GAMES = 12
ROLL_HALFLIFE = 4

STAT_COLS = ["plays", "yards", "ypp", "first_downs", "third_att", "third_conv", "third_rate", "top_sec",
             "sec_per_play", "neutral_plays", "neutral_pass_rate", "neutral_xpass", "proe", "pass_rate"]
ALLOWED_FROM = ["plays", "yards", "ypp", "first_downs", "third_rate"]


def _top_to_seconds(s):
    """'3:33' -> 213.0 ; missing/blank -> NaN."""
    try:
        m, sec = str(s).split(":")
        return int(m) * 60 + int(sec)
    except (ValueError, AttributeError):
        return np.nan


def load_pbp(seasons) -> pd.DataFrame:
    import nflreadpy as nfl

    cols = ["game_id", "season", "week", "season_type", "posteam", "defteam", "play_type", "pass", "rush", "sack",
            "yards_gained", "first_down", "down", "third_down_converted", "third_down_failed", "xpass",
            "score_differential", "qtr", "drive", "drive_time_of_possession", "qb_kneel", "qb_spike"]
    pbp = nfl.load_pbp(list(seasons)).to_pandas()
    pbp = pbp[[c for c in cols if c in pbp.columns]]
    return pbp[pbp["season_type"] == "REG"].copy()


def team_game_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week, game_id, team) with offensive stats and the `*_allowed` columns."""
    live = pbp[(pbp["play_type"].isin(["pass", "run"])) & pbp["posteam"].notna()].copy()
    live = live[(live["qb_kneel"].fillna(0) == 0) & (live["qb_spike"].fillna(0) == 0)]
    live["is_pass"] = (live["play_type"] == "pass").astype(float)
    live["yards_gained"] = live["yards_gained"].fillna(0.0)

    g = live.groupby(["season", "week", "game_id", "posteam", "defteam"])
    agg = g.agg(plays=("play_type", "size"), yards=("yards_gained", "sum"),
                first_downs=("first_down", "sum"), pass_rate=("is_pass", "mean")).reset_index()

    third = live[live["down"] == 3].groupby("game_id posteam".split()).agg(
        third_conv=("third_down_converted", "sum"), third_fail=("third_down_failed", "sum")).reset_index()
    third["third_att"] = third["third_conv"] + third["third_fail"]
    agg = agg.merge(third[["game_id", "posteam", "third_att", "third_conv"]], on=["game_id", "posteam"], how="left")
    agg[["third_att", "third_conv"]] = agg[["third_att", "third_conv"]].fillna(0.0)
    agg["third_rate"] = np.where(agg["third_att"] > 0, agg["third_conv"] / agg["third_att"].clip(lower=1), np.nan)
    agg["ypp"] = agg["yards"] / agg["plays"]

    # neutral game state: tied or leading, quarters 1-3
    neutral = live[(live["score_differential"] >= 0) & (live["qtr"] <= 3)]
    n = neutral.groupby(["game_id", "posteam"]).agg(
        neutral_plays=("play_type", "size"), neutral_pass_rate=("is_pass", "mean"),
        neutral_xpass=("xpass", "mean")).reset_index()
    agg = agg.merge(n, on=["game_id", "posteam"], how="left")
    agg["proe"] = (agg["neutral_pass_rate"] - agg["neutral_xpass"]) * 100

    # time of possession from drive clocks (one clock value per drive)
    drives = (pbp[pbp["posteam"].notna() & pbp["drive_time_of_possession"].notna()]
              .drop_duplicates(["game_id", "drive", "posteam"])[["game_id", "drive", "posteam", "drive_time_of_possession"]])
    drives["sec"] = drives["drive_time_of_possession"].map(_top_to_seconds)
    top = drives.groupby(["game_id", "posteam"])["sec"].sum().reset_index().rename(columns={"sec": "top_sec"})
    agg = agg.merge(top, on=["game_id", "posteam"], how="left")
    agg["sec_per_play"] = agg["top_sec"] / agg["plays"]

    agg = agg.rename(columns={"posteam": "team", "defteam": "opp"})
    opp = agg[["game_id", "team"] + ALLOWED_FROM].rename(
        columns={"team": "opp", **{c: f"{c}_allowed" for c in ALLOWED_FROM}})
    out = agg.merge(opp, on=["game_id", "opp"], how="left")
    return out.sort_values(["season", "week", "team"]).reset_index(drop=True)


def rolling_features(tg: pd.DataFrame, cols=None, games=ROLL_GAMES, halflife=ROLL_HALFLIFE) -> pd.DataFrame:
    """
    For every team-game row, the recency-weighted mean of that team's PRIOR games (never the
    game itself) for each column -> columns named `r_<col>`; `r_n` is how many prior games fed it.
    """
    cols = cols or (STAT_COLS + [f"{c}_allowed" for c in ALLOWED_FROM])
    tg = tg.sort_values(["season", "week"]).reset_index(drop=True)
    out = []
    for team, g in tg.groupby("team", sort=False):
        g = g.sort_values(["season", "week"]).reset_index(drop=True)
        rows = []
        for i in range(len(g)):
            prior = g.iloc[max(0, i - games):i]
            feat = {"game_id": g.at[i, "game_id"], "team": team, "r_n": len(prior)}
            if len(prior):
                w = 0.5 ** (np.arange(len(prior))[::-1] / halflife)
                for c in cols:
                    v = prior[c].to_numpy(float)
                    ok = ~np.isnan(v)
                    feat[f"r_{c}"] = float(np.average(v[ok], weights=w[ok])) if ok.any() else np.nan
            rows.append(feat)
        out.append(pd.DataFrame(rows))
    return pd.concat(out, ignore_index=True)


def current_profiles(tg: pd.DataFrame, **kw) -> dict:
    """{team: {r_<col>: value}} -- each team's rolling profile through its LATEST completed game,
    i.e. what a not-yet-played game's pre-game features would be."""
    cols = kw.pop("cols", None) or (STAT_COLS + [f"{c}_allowed" for c in ALLOWED_FROM])
    games, halflife = kw.get("games", ROLL_GAMES), kw.get("halflife", ROLL_HALFLIFE)
    out = {}
    for team, g in tg.sort_values(["season", "week"]).groupby("team"):
        prior = g.tail(games)
        w = 0.5 ** (np.arange(len(prior))[::-1] / halflife)
        prof = {"r_n": len(prior)}
        for c in cols:
            v = prior[c].to_numpy(float)
            ok = ~np.isnan(v)
            prof[f"r_{c}"] = float(np.average(v[ok], weights=w[ok])) if ok.any() else np.nan
        out[team] = prof
    return out


def league_means(tg: pd.DataFrame, col: str) -> float:
    return float(tg[col].mean())


def matchup_expectation(off: dict, opp: dict, lg_ypp: float, lg_plays: float) -> dict:
    """
    One side's expected plays / yards-per-play / yards against a specific opponent.
    `off`/`opp` are rolling-feature dicts (r_<col>). Plays average the offense's own pace with
    the opponent's plays-allowed; ypp is offense + (opponent defense - league).
    """
    plays = np.nanmean([off.get("r_plays"), opp.get("r_plays_allowed")])
    ypp = off.get("r_ypp", np.nan) + (opp.get("r_ypp_allowed", np.nan) - lg_ypp)
    return {"exp_plays": float(plays), "exp_ypp": float(ypp), "exp_yards": float(plays * ypp)}


def _r(x, nd=1):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def matchup_profile(profiles: dict, home: str, away: str, lg_ypp: float, lg_plays: float) -> dict | None:
    """
    The two teams' stat lines side by side for the Game Lines card, plus each side's expected
    plays / yards-per-play / yards for THIS matchup. None if either team has no profile yet.
    Informational: game_factors.py found the posted lines already absorb these stats.
    """
    if home not in profiles or away not in profiles:
        return None

    def side(team, opp):
        p, o = profiles[team], profiles[opp]
        exp = matchup_expectation(p, o, lg_ypp, lg_plays)
        return {"plays": _r(p.get("r_plays")), "yards": _r(p.get("r_yards"), 0), "ypp": _r(p.get("r_ypp"), 2),
                "ypp_allowed": _r(p.get("r_ypp_allowed"), 2), "first_downs": _r(p.get("r_first_downs")),
                "third_rate": _r(100 * p["r_third_rate"]) if np.isfinite(p.get("r_third_rate", np.nan)) else None,
                "neutral_pass_rate": _r(100 * p["r_neutral_pass_rate"]) if np.isfinite(p.get("r_neutral_pass_rate", np.nan)) else None,
                "proe": _r(p.get("r_proe")), "top_min": _r(p.get("r_top_sec", np.nan) / 60, 1),
                "exp_plays": _r(exp["exp_plays"]), "exp_ypp": _r(exp["exp_ypp"], 2), "exp_yards": _r(exp["exp_yards"], 0),
                "games": int(p["r_n"])}
    h, a = side(home, away), side(away, home)
    return {"home": h, "away": a,
            "exp_total_yards": None if None in (h["exp_yards"], a["exp_yards"]) else round(h["exp_yards"] + a["exp_yards"]),
            "exp_total_plays": None if None in (h["exp_plays"], a["exp_plays"]) else round(h["exp_plays"] + a["exp_plays"], 1)}
