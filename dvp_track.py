"""
dvp_track.py
============
The weekly defense-vs-position log, and the check on whether it predicts anything.

Every refresh logs what each defense's readings looked like BEFORE the week's games (tracking/dvp_log.csv: the season so far, the last
four games, last season and the schedule-adjusted reading, for the stats a prop bettor looks at, by position group). After the games it
fills in what each defense actually allowed that week. The Monday report then answers, from the log alone:

  - did the softest defenses by each reading actually allow more than the toughest ones the next game?
  - how correlated is each reading with what was allowed? (a year of weeks is what makes this number mean something)

and, for the player-level question (does a soft matchup lift a player above his projection?), dvp_study.py replays every game since 2019.

    python dvp_track.py            # grade what has finished and print the season's check
"""

import pathlib

import numpy as np
import pandas as pd

import context_data as cd
import dvp_signal as ds

LOG = pathlib.Path(__file__).parent / "tracking" / "dvp_log.csv"
COLUMNS = ["season", "week", "defense", "grp", "stat", "n", "sea", "l4", "prev", "adj", "allowed", "league", "logged_at"]
# the stats worth logging for each position group (what the props and pick'em apps post), keyed as in context_data.DVP_COLS
STATS = {
    "QB": ["pass_att", "cmp", "pass_yds", "dk"],
    "RB": ["rush_att", "rush_yds", "tgt", "rec", "rec_yds", "dk"],
    "WR": ["tgt", "rec", "rec_yds", "dk"],
    "TE": ["tgt", "rec", "rec_yds", "dk"],
}
READINGS = ("sea", "l4", "prev", "adj")


def load(path=None) -> pd.DataFrame:
    path = pathlib.Path(path or LOG)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path)
    for c in COLUMNS:
        if c not in df:
            df[c] = np.nan
    return df[COLUMNS]


def stat_values(history: pd.DataFrame) -> pd.DataFrame:
    """The per-player-game columns of cd.DVP_STATS (attempts, targets, DK points...) aligned to `history`."""
    v = pd.DataFrame(index=history.index)
    for k, c in cd.DVP_COLS.items():
        v[k] = history[c].fillna(0.0) if c in history else 0.0
    v[["dk", "fd", "ppr"]] = cd.fantasy_points(history)
    return v


def game_tables(history: pd.DataFrame) -> dict:
    """{stat: game table with the offense's expectation}, for every stat in STATS."""
    vals = stat_values(history)
    need = sorted({s for stats in STATS.values() for s in stats})
    return {s: ds.with_expectation(ds.game_table(history, vals[s])) for s in need}


def snapshot_rows(history: pd.DataFrame, season: int, week: int, now: str, tables: dict = None) -> list:
    """Readings for every defense as of the start of `week`, using only earlier games."""
    tables = tables or game_tables(history)
    rows = []
    for stat in sorted(tables):
        rd = ds.readings_asof(tables[stat], season, week)
        for grp, stats in STATS.items():
            if stat not in stats or grp not in rd.index.get_level_values(0):
                continue
            for defense, r in rd.loc[grp].iterrows():
                rows.append({"season": season, "week": week, "defense": defense, "grp": grp, "stat": stat,
                             "n": r["n_sea"], **{k: round(float(r[k]), 4) if pd.notna(r[k]) else np.nan for k in READINGS},
                             "allowed": np.nan, "league": np.nan, "logged_at": now})
    return rows


def log_snapshot(rows: list, path=None) -> int:
    """Upsert by (season, week, defense, grp, stat): a re-run before kickoff refreshes the readings and keeps a grade already there."""
    path = pathlib.Path(path or LOG)
    if not rows:
        return 0
    old, new = load(path), pd.DataFrame(rows, columns=COLUMNS)
    both = pd.concat([old, new], ignore_index=True)
    key = ["season", "week", "defense", "grp", "stat"]
    graded = both.dropna(subset=["allowed"]).drop_duplicates(key, keep="last").set_index(key)[["allowed", "league"]]
    both = both.drop_duplicates(key, keep="last").set_index(key)
    both.loc[graded.index.intersection(both.index), ["allowed", "league"]] = graded.loc[graded.index.intersection(both.index)]
    path.parent.mkdir(exist_ok=True)
    both.reset_index().to_csv(path, index=False)
    return len(new)


def grade(history: pd.DataFrame, path=None, tables: dict = None) -> int:
    """Fill in what each defense actually allowed (and the league's average that week) for logged weeks that have been played."""
    path = pathlib.Path(path or LOG)
    df = load(path)
    if df.empty:
        return 0
    todo = df[df["allowed"].isna()]
    if todo.empty:
        return 0
    tables = tables or game_tables(history)
    done = set(zip(history["season"], history["week"]))
    n = 0
    for stat, g in todo.groupby("stat"):
        gt = tables[stat]
        league = gt.groupby(["season", "week", "grp"])["v"].mean()
        allowed = gt.set_index(["season", "week", "def", "grp"])["v"]
        for i, r in g.iterrows():
            if (int(r["season"]), int(r["week"])) not in done:
                continue
            a = allowed.get((int(r["season"]), int(r["week"]), r["defense"], r["grp"]))
            if a is None:                                                      # on a bye or not played
                continue
            df.at[i, "allowed"] = float(a)
            df.at[i, "league"] = float(league.get((int(r["season"]), int(r["week"]), r["grp"]), np.nan))
            n += 1
    if n:
        df.to_csv(path, index=False)
    return n


def accuracy(df: pd.DataFrame) -> dict:
    """
    From graded rows: per reading, the pooled correlation between (reading - 1) and (allowed / league that week - 1), and how the
    softest and toughest quarter of defenses by that reading fared against the league the following game.
    """
    d = df.dropna(subset=["allowed", "league"])
    d = d[d["league"] > 0]
    if d.empty:
        return {"n": 0}
    d = d.assign(ratio=d["allowed"] / d["league"])
    out = {"n": int(len(d)), "weeks": int(d.groupby(["season", "week"]).ngroups), "readings": {}}
    for r in READINGS:
        x = d.dropna(subset=[r])
        x = x[x["ratio"].notna()]
        if len(x) < 30:
            continue
        # rank within each (week, position, stat) so a reading is compared with its own peers
        x = x.assign(rk=x.groupby(["season", "week", "grp", "stat"])[r].rank(pct=True))
        soft, tough = x[x["rk"] >= 0.75], x[x["rk"] <= 0.25]
        out["readings"][r] = {"corr": float(np.corrcoef(x[r] - 1, x["ratio"] - 1)[0, 1]), "n": int(len(x)),
                              "soft": float(soft["ratio"].mean() - 1), "tough": float(tough["ratio"].mean() - 1)}
    return out


def section(season: int, week: int, path=None):
    df = load(path)
    df = df[df["season"] == season]
    wk, sd = accuracy(df[df["week"] == week]), accuracy(df[df["week"] <= week])
    if not sd["n"]:
        return None
    return {"week": wk, "season": sd, "writeup": blurb(wk, sd, week)}


NAMES = {"sea": "season-to-date", "l4": "last-four-games", "prev": "last-season", "adj": "schedule-adjusted"}


def blurb(wk: dict, sd: dict, week: int) -> str:
    parts = []
    best = None
    for r, v in sd.get("readings", {}).items():
        if best is None or v["corr"] > sd["readings"][best]["corr"]:
            best = r
    if wk.get("readings") and "adj" in wk["readings"]:
        v = wk["readings"]["adj"]
        parts.append(f"Week {week}: the schedule-adjusted ratings' softest quarter of defenses allowed {v['soft']:+.0%} versus the league, the toughest quarter {v['tough']:+.0%}.")
    if best:
        v = sd["readings"][best]
        parts.append(f"Season to date ({sd['weeks']} weeks, {sd['n']:,} defense-position-stat games): the {NAMES[best]} rating tracked what defenses allowed best "
                     f"(correlation {v['corr']:+.2f}; the softest quarter allowed {v['soft']:+.0%} against the league, the toughest {v['tough']:+.0%}).")
    parts.append("A reading only helps a projection if it predicts the NEXT game, which is what this checks; "
                 "the player-level test (dvp_study.py) replays every game since 2019.")
    return " ".join(parts)


if __name__ == "__main__":
    import nflreadpy as nfl
    import props_model as pm

    season = int(nfl.get_current_season())
    ps = nfl.load_player_stats(list(range(season - 1, season + 1))).to_pandas()
    hist = pm.build_history(ps)
    print(f"graded {grade(hist)} logged readings")
    print(accuracy(load().query("season == @season")))
