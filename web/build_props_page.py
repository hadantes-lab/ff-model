"""
build_props_page.py
===================
Combines props_template.html with web/props_snapshot.json (and, if present,
web/track_record.json) into one self-contained web/props.html you can open
directly or publish.

    python export_props.py            # or: python export_props.py --sample
    python web/build_props_page.py
"""

import json
import pathlib

here = pathlib.Path(__file__).parent

snap = (here / "props_snapshot.json").read_text(encoding="utf-8")
template = (here / "props_template.html").read_text(encoding="utf-8")
track_file = here / "track_record.json"
track = track_file.read_text(encoding="utf-8") if track_file.exists() else "null"

# "</" inside embedded JSON could end the <script> block early
payload = json.dumps(json.loads(snap), separators=(",", ":")).replace("</", "<\\/")
track_payload = json.dumps(json.loads(track), separators=(",", ":")).replace("</", "<\\/")
html = template.replace("__PROPS_JSON__", payload).replace("__TRACK_RECORD_JSON__", track_payload)
out = here / "props.html"
out.write_text(html, encoding="utf-8")
print(f"Wrote {out} ({out.stat().st_size / 1024:.0f} KB, {len(json.loads(snap)['props'])} props)")
