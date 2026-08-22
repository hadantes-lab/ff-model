"""
export_snapshot.py
===================
One-off exporter: runs auction_engine's real data pull + auction value
calc, dumps the result to JSON for the shareable HTML version of the tool.

Run:  python export_snapshot.py > snapshot.json
"""

import json
import datetime
from auction_engine import load_player_points, compute_auction_values, archetype, ARCHE_RULES

players = load_player_points()
players = compute_auction_values(players)
players["arch"] = players.apply(archetype, axis=1)

out = []
for _, r in players.sort_values("value", ascending=False).iterrows():
    out.append({
        "name": r["name"],
        "pos": r["pos"],
        "team": r["team"],
        "proj_points": round(float(r["proj_points"]), 1),
        "floor": int(r["floor"]),
        "ceiling": int(r["ceiling"]),
        "value": int(r["value"]),
        "arch": r["arch"],
    })

print(json.dumps({
    "generated": datetime.date.today().isoformat(),
    "players": out,
}, indent=None))
