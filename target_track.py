"""
target_track.py
===============
Targets are the one projected stat no sportsbook in the Odds API posts a line on, so there is no market to grade a pick
against. What can be tracked is the projection itself: logged before the games (tracking/target_proj.csv), graded after,
and compared with the simple alternative of "a player's recent average", which is the thing a projection has to beat
to be worth reading. The Monday report quotes the result.

    python target_track.py            # grade what has finished and print the season's accuracy
"""

import math
import pathlib

import numpy as np
import pandas as pd

LOG = pathlib.Path(__file__).parent / "tracking" / "target_proj.csv"
COLUMNS = ["season", "week", "player_id", "player", "team", "pos", "game", "mean", "sd", "line", "recent_avg", "logged_at", "actual"]


def load(path=None) -> pd.DataFrame:
    path = pathlib.Path(path or LOG)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path)
    for c in COLUMNS:
        if c not in df:
            df[c] = np.nan
    return df[COLUMNS]


def rows_from_snapshot(targets: list, season: int, week: int, now: str) -> list:
    return [{"season": season, "week": int(t.get("week") or week), "player_id": t["player_id"], "player": t["player"], "team": t["team"],
             "pos": t["pos"], "game": t["game"], "mean": t["mean"], "sd": t["sd"], "line": t["line"], "recent_avg": t.get("target_rate"),
             "logged_at": now, "actual": np.nan} for t in targets or []]


def log_projections(rows: list, path=None) -> int:
    """Upsert by (season, week, player): a re-run before kickoff refreshes the projection and keeps any grade already there."""
    path = pathlib.Path(path or LOG)
    if not rows:
        return 0
    old = load(path)
    new = pd.DataFrame(rows, columns=COLUMNS)
    both = pd.concat([old, new], ignore_index=True)
    both["_k"] = list(zip(both["season"], both["week"], both["player_id"]))
    graded = both.dropna(subset=["actual"]).drop_duplicates("_k", keep="last").set_index("_k")["actual"]
    both = both.drop_duplicates("_k", keep="last")
    both["actual"] = [graded.get(k, a) for k, a in zip(both["_k"], both["actual"])]
    path.parent.mkdir(exist_ok=True)
    both.drop(columns="_k").to_csv(path, index=False)
    return len(new)


def grade(player_stats: pd.DataFrame, path=None) -> int:
    """Fill `actual` from nflverse weekly player stats (a player who didn't play stays blank). -> rows newly graded."""
    path = pathlib.Path(path or LOG)
    df = load(path)
    if df.empty:
        return 0
    ps = player_stats[player_stats["season_type"] == "REG"][["season", "week", "player_id", "targets"]]
    key = {(int(r.season), int(r.week), r.player_id): float(r.targets) for r in ps.itertuples(index=False)}
    n = 0
    for i, r in df.iterrows():
        if pd.isna(r["actual"]):
            v = key.get((int(r["season"]), int(r["week"]), r["player_id"]))
            if v is not None:
                df.at[i, "actual"] = v
                n += 1
    if n:
        df.to_csv(path, index=False)
    return n


def accuracy(df: pd.DataFrame) -> dict:
    """MAE of the projection and of the recent average, bias, and how often the actual fell inside +/-1 SD (about 68% if calibrated)."""
    d = df.dropna(subset=["actual", "mean"])
    if d.empty:
        return {"n": 0}
    err = d["actual"] - d["mean"]
    base = d.dropna(subset=["recent_avg"])
    out = {"n": int(len(d)), "mae": float(err.abs().mean()), "bias": float(err.mean()),
           "within_1sd": float((err.abs() <= d["sd"]).mean()), "n_base": int(len(base))}
    if len(base):
        out["mae_projection_on_base"] = float((base["actual"] - base["mean"]).abs().mean())
        out["mae_recent_avg"] = float((base["actual"] - base["recent_avg"]).abs().mean())
    return out


def section(season: int, week: int, path=None):
    df = load(path)
    df = df[df["season"] == season]
    wk = accuracy(df[df["week"] == week])
    sd = accuracy(df[df["week"] <= week])
    if not sd["n"]:
        return None
    return {"week": wk, "season": sd, "writeup": blurb(wk, sd, week)}


def blurb(wk: dict, sd: dict, week: int) -> str:
    parts = []
    if wk.get("n"):
        parts.append(f"Week {week}: {wk['n']} target projections graded, off by {wk['mae']:.2f} targets on average"
                     + (f" against {wk['mae_recent_avg']:.2f} for each player's recent average" if wk.get("mae_recent_avg") is not None else "") + ".")
    parts.append(f"Season to date ({sd['n']} projections): {sd['mae']:.2f} targets off on average"
                 + (f" versus {sd['mae_recent_avg']:.2f} for the recent average ({'better' if sd['mae'] < sd['mae_recent_avg'] else 'no better'})" if sd.get("mae_recent_avg") is not None else "")
                 + f"; bias {sd['bias']:+.2f}; {sd['within_1sd']:.0%} landed within one standard deviation of the projection (68% if its spread is right).")
    parts.append("No book posts a targets line, so this tracks the projection itself, not a betting result.")
    return " ".join(parts)


if __name__ == "__main__":
    import nflreadpy as nfl
    season = int(nfl.get_current_season())
    n = grade(nfl.load_player_stats([season]).to_pandas())
    print(f"graded {n} new target projections")
    df = load()
    print(accuracy(df[df["season"] == season]))
