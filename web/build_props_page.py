"""
build_props_page.py
===================
Combines props_template.html with web/props_snapshot.json into one
self-contained web/props.html you can open directly or publish.

    python export_props.py            # or: python export_props.py --sample
    python web/build_props_page.py
"""

import json
import pathlib

here = pathlib.Path(__file__).parent

snap = (here / "props_snapshot.json").read_text(encoding="utf-8")
template = (here / "props_template.html").read_text(encoding="utf-8")

# "</" inside embedded JSON could end the <script> block early
payload = json.dumps(json.loads(snap), separators=(",", ":")).replace("</", "<\\/")
out = here / "props.html"
out.write_text(template.replace("__PROPS_JSON__", payload), encoding="utf-8")
print(f"Wrote {out} ({out.stat().st_size / 1024:.0f} KB, {len(json.loads(snap)['props'])} props)")
