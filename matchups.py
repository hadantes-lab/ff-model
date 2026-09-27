"""
matchups.py
===========
Man-vs-zone coverage matchups for receivers.

The idea: some receivers produce differently against man and zone coverage, and
defenses differ in how much zone they play, so "WR X vs a zone-heavy defense" may
shift a receiving-yards projection.

WHAT THE DATA SAYS (2024-25, NFL participation data joined to play-by-play)
--------------------------------------------------------------------------
  - Defenses genuinely differ: zone on 47% (DEN) to 74% (GB) of pass plays; league 60%.
  - But a defense's zone rate only correlates ~0.42 from one season to the next.
  - Zone coverage gives up more yards per target league-wide (7.3 vs 6.7).
  - A receiver's own zone-vs-man gap is mostly NOISE: across 90 receivers with enough
    targets both years, year-over-year correlation of (zone yds/target - man yds/target)
    was only ~0.16.

So the effect is real in principle and small in practice. Both halves are shrunk hard
toward "no effect" (see SPLIT_K and SCHEME_TRUST), and the adjustment is capped, so a
typical matchup moves a projection by 1-3%.

LIMITS
------
  - Coverage tags are published only through 2025. The 2026 scheme is *estimated* from
    2024-25 tendencies (2025 weighted more); 2026 results still flow in through the
    opponent-strength multiplier in props_model.py, which weights the current season.
  - It only adjusts receiving yards, using yards per target.
"""

import numpy as np
import pandas as pd

ZONE, MAN = "ZONE_COVERAGE", "MAN_COVERAGE"

SPLIT_K = 180        # targets' worth of prior: reliability = n / (n + K), n = the thinner side's targets
SCHEME_TRUST = 0.5   # share of a team's (weighted) deviation from league-average zone rate carried forward
SEASON_WEIGHTS = {2024: 1.0, 2025: 2.0}   # newer coverage tendencies count more
MULT_CAP = 0.06      # never move a projection more than +/-6%
MIN_SIDE_TARGETS = 10


def load_coverage_targets(seasons=(2024, 2025)) -> pd.DataFrame:
    """One row per pass attempt with the defense's man/zone tag. Needs nflreadpy (heavy: play-by-play)."""
    import nflreadpy as nfl

    pt = nfl.load_participation(list(seasons)).to_pandas()
    pbp = nfl.load_pbp(list(seasons)).to_pandas()
    pbp = pbp[(pbp["pass_attempt"] == 1) & (pbp["sack"] != 1)]
    pbp = pbp[["game_id", "play_id", "season", "defteam", "receiver_player_id", "yards_gained", "complete_pass"]]
    tags = pt[["nflverse_game_id", "play_id", "defense_man_zone_type"]].rename(
        columns={"nflverse_game_id": "game_id", "defense_man_zone_type": "coverage"})
    df = pbp.merge(tags, on=["game_id", "play_id"], how="left")
    return df[df["coverage"].isin([ZONE, MAN])].reset_index(drop=True)


def team_zone_rates(targets: pd.DataFrame, season_weights=SEASON_WEIGHTS, trust=SCHEME_TRUST):
    """-> (league zone rate, {team: estimated zone rate}), season-weighted and shrunk toward the league."""
    t = targets.assign(_z=(targets["coverage"] == ZONE).astype(float),
                       _w=targets["season"].map(season_weights).fillna(1.0))
    league = float(np.average(t["_z"], weights=t["_w"]))
    out = {}
    for team, g in t.groupby("defteam"):
        raw = float(np.average(g["_z"], weights=g["_w"]))
        out[team] = league + trust * (raw - league)
    return league, out


def receiver_splits(targets: pd.DataFrame, min_side=MIN_SIDE_TARGETS) -> dict:
    """{player_id: {zone_n, zone_ypt, man_n, man_ypt}} pooled across seasons (yards per target)."""
    t = targets.dropna(subset=["receiver_player_id"])
    g = t.groupby(["receiver_player_id", "coverage"])["yards_gained"].agg(["size", "mean"]).unstack()
    out = {}
    for pid, r in g.iterrows():
        zn, mn = r[("size", ZONE)], r[("size", MAN)]
        if pd.isna(zn) or pd.isna(mn) or zn < min_side or mn < min_side:
            continue
        out[pid] = {"zone_n": int(zn), "zone_ypt": float(r[("mean", ZONE)]),
                    "man_n": int(mn), "man_ypt": float(r[("mean", MAN)])}
    return out


def league_zone_man_ratio(targets: pd.DataFrame) -> float:
    """League yards-per-target vs zone divided by vs man (>1: zone gives up more per target)."""
    y = targets.groupby("coverage")["yards_gained"].mean()
    return float(y[ZONE] / y[MAN])


def matchup_multiplier(split, opp_zone_rate, league_zone_rate, league_ratio) -> dict:
    """
    Expected relative change in a receiver's yards per target against this defense's scheme,
    versus an average scheme, from HIS zone/man tendency only (the general "zone gives up more"
    effect is already inside the opponent-strength multiplier).

      raw = ln( (his zone ypt / his man ypt) / league ratio )      > 0: better vs zone than typical
      delta = reliability * raw                                      shrunk hard; reliability = n/(n+K)
      mult = (z_opp*e^delta + 1 - z_opp) / (z_lg*e^delta + 1 - z_lg)

    Returns {"mult", "delta", "reliability", ...}; mult is exactly 1.0 with no usable split.
    """
    if not split:
        return {"mult": 1.0, "delta": 0.0, "reliability": 0.0}
    n = min(split["zone_n"], split["man_n"])
    reliability = n / (n + SPLIT_K)
    raw = float(np.log((max(split["zone_ypt"], 0.1) / max(split["man_ypt"], 0.1)) / league_ratio))
    delta = reliability * raw
    e = float(np.exp(delta))
    mult = (opp_zone_rate * e + 1 - opp_zone_rate) / (league_zone_rate * e + 1 - league_zone_rate)
    mult = float(np.clip(mult, 1 - MULT_CAP, 1 + MULT_CAP))
    return {"mult": mult, "delta": delta, "reliability": reliability}


class CoverageModel:
    """Everything the exporter needs, built once from the coverage-tagged targets."""

    def __init__(self, targets: pd.DataFrame):
        self.league_zone, self.team_zone = team_zone_rates(targets)
        self.splits = receiver_splits(targets)
        self.ratio = league_zone_man_ratio(targets)

    def for_receiver(self, player_id, opp_team) -> dict:
        """Matchup block for one receiver against one defense, ready to export."""
        split = self.splits.get(player_id)
        opp_zone = self.team_zone.get(opp_team, self.league_zone)
        m = matchup_multiplier(split, opp_zone, self.league_zone, self.ratio)
        return {
            "mult": round(m["mult"], 4), "reliability": round(m["reliability"], 2),
            "opp_zone": round(opp_zone, 3), "league_zone": round(self.league_zone, 3),
            "split": None if not split else {k: (round(v, 2) if isinstance(v, float) else v) for k, v in split.items()},
        }
