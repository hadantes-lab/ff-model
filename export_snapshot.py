"""
export_snapshot.py
===================
One-off exporter: runs auction_engine's real data pull + auction value
calc, merges in current FantasyPros consensus draft rankings (ADP/ECR)
via nflreadpy, and dumps the result to JSON for the shareable HTML
version of the tool.

Run:  python export_snapshot.py > web/snapshot.json
"""

import json
import re
import sys
import datetime

import pandas as pd
import nflreadpy as nfl

from auction_engine import load_player_points, compute_auction_values, archetype


def normalize(name):
    name = name.lower()
    name = re.sub(r"[.'\-]", "", name)
    name = re.sub(r"\s+(jr|sr|ii|iii|iv|v)\.?$", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


players = load_player_points()
players = compute_auction_values(players)
players["arch"] = players.apply(archetype, axis=1)

# current-season expert consensus draft rankings (ADP proxy), via nflverse's
# FantasyPros scrape -- "redraft-overall" is the standard, cross-position
# ranking real snake drafts are based on.
adp_raw = nfl.load_ff_rankings("draft").to_pandas()
adp = adp_raw[
    (adp_raw["page_type"] == "redraft-overall") & (adp_raw["pos"].isin(["QB", "RB", "WR", "TE"]))
].copy()
adp["_key"] = adp["player"].map(normalize)
adp = adp.sort_values("ecr")

adp_map = {}
for _, r in adp.iterrows():
    key = r["_key"]
    if key in adp_map:
        continue  # keep the best (lowest ecr) row on duplicate names
    adp_map[key] = {
        "ecr": round(float(r["ecr"]), 2),
        "sd": round(float(r["sd"]), 2),
        "bye": int(r["bye"]) if pd.notna(r["bye"]) else None,
    }

out = []
for _, r in players.sort_values("value", ascending=False).iterrows():
    a = adp_map.get(normalize(r["name"]))
    out.append({
        "name": r["name"],
        "pos": r["pos"],
        "team": r["team"],
        "proj_points": round(float(r["proj_points"]), 1),
        "floor": int(r["floor"]),
        "ceiling": int(r["ceiling"]),
        "value": int(r["value"]),
        "arch": r["arch"],
        "adp": a["ecr"] if a else None,
        "adp_sd": a["sd"] if a else None,
        "bye": a["bye"] if a else None,
    })

matched = sum(1 for p in out if p["adp"] is not None)
print(f"[export_snapshot] ADP matched {matched}/{len(out)} players", file=sys.stderr)

print(json.dumps({
    "generated": datetime.date.today().isoformat(),
    "adp_scrape_date": str(adp["scrape_date"].iloc[0]) if len(adp) else None,
    "players": out,
}, separators=(",", ":")))
