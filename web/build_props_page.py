"""
build_props_page.py
===================
Combines props_template.html with web/props_snapshot.json (and, if present,
web/track_record.json and web/player_history.json) into one self-contained
web/props.html you can open directly or publish.

    python export_props.py            # or: python export_props.py --sample
    python web/build_props_page.py
"""

import json
import pathlib

here = pathlib.Path(__file__).parent


def _read_or_default(name, default):
    f = here / name
    return f.read_text(encoding="utf-8") if f.exists() else default


snap = (here / "props_snapshot.json").read_text(encoding="utf-8")
template = (here / "props_template.html").read_text(encoding="utf-8")
track = _read_or_default("track_record.json", "null")
history = _read_or_default("player_history.json", "{}")

# "</" inside embedded JSON could end the <script> block early
def embed(raw):
    return json.dumps(json.loads(raw), separators=(",", ":")).replace("</", "<\\/")


html = (template
        .replace("__PROPS_JSON__", embed(snap))
        .replace("__TRACK_RECORD_JSON__", embed(track))
        .replace("__PLAYER_HISTORY_JSON__", embed(history)))
out = here / "props.html"
out.write_text(html, encoding="utf-8")
print(f"Wrote {out} ({out.stat().st_size / 1024:.0f} KB, {len(json.loads(snap)['props'])} props, "
      f"{len(json.loads(history))} players with season history)")
