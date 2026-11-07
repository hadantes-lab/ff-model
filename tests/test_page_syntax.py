"""
The props page is one big inline script; a stray backtick in any tab's code breaks every tab and the Python tests never run it.
This parses the whole script with node (skipped where node is not installed).

Run:  python -m unittest discover -s tests -v
"""

import pathlib
import re
import shutil
import subprocess
import tempfile
import unittest

TEMPLATE = pathlib.Path(__file__).parent.parent / "web" / "props_template.html"


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class TestPageScript(unittest.TestCase):
    def test_the_inline_script_parses(self):
        html = TEMPLATE.read_text(encoding="utf-8")
        js = max((m.group(1) for m in re.finditer(r"<script>(.*?)</script>", html, re.S)), key=len)
        js = js.replace("__PROPS_JSON__", "{}").replace("__TRACK_RECORD_JSON__", "null").replace("__PLAYER_HISTORY_JSON__", "{}")
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "page.js"
            f.write_text(js, encoding="utf-8")
            r = subprocess.run(["node", "--check", str(f)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[:800])


if __name__ == "__main__":
    unittest.main()
