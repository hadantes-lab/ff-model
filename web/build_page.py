"""
build_page.py
==============
Combines auction_board_template.html with a fresh snapshot.json into a
single self-contained auction_board.html you can open directly or
re-publish as a shareable page.

Run (from the ff-model folder):
   python export_snapshot.py > web/snapshot.json
   python web/build_page.py
"""

import json
import pathlib

here = pathlib.Path(__file__).parent

snap = json.loads((here / "snapshot.json").read_text(encoding="utf-8"))
template = (here / "auction_board_template.html").read_text(encoding="utf-8")

html = template.replace("__PLAYERS_JSON__", json.dumps(snap["players"], separators=(",", ":")))
html = html.replace("__SNAPSHOT_DATE__", json.dumps(snap["generated"]))

out = here / "auction_board.html"
out.write_text(html, encoding="utf-8")
print(f"Wrote {out} ({len(snap['players'])} players, snapshot {snap['generated']})")
