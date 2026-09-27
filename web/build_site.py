"""
build_site.py
=============
Assembles the folder GitHub Pages serves from whichever pages have been built:

    site/index.html         landing page linking to whatever exists
    site/draft/index.html   the draft board   (web/auction_board.html)
    site/props/index.html   the props page    (web/props.html)

Only finished HTML pages are copied -- never the raw JSON snapshots, which
would amount to redistributing the underlying odds/ranking data as files.

    python web/build_site.py [output_dir]
"""

import pathlib
import shutil
import sys

here = pathlib.Path(__file__).parent
out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else here.parent / "site"

PAGES = [
    ("draft", "auction_board.html", "Draft Board", "Auction and snake draft boards with ADP tiers, PPR and league-size settings."),
    ("props", "props.html", "Props Simulator", "Simulated player props against sportsbook lines, with a what-if calculator."),
]

if out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True)

links = []
for slug, src, title, blurb in PAGES:
    page = here / src
    if page.exists():
        (out / slug).mkdir()
        shutil.copy(page, out / slug / "index.html")
        links.append((slug, title, blurb))

items = "\n".join(
    f'    <a class="card" href="{slug}/"><h2>{title}</h2><p>{blurb}</p></a>' for slug, title, blurb in links
) or "    <p>Nothing has been published yet.</p>"

(out / "index.html").write_text(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>FF Model</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@700&family=IBM+Plex+Sans:wght@400;600&display=swap">
<style>
  :root {{ --bg:#12151b; --surface:#1b2028; --border:#333b48; --text:#e9eaed; --dim:#98a0ac; --gold:#d9a441; }}
  @media (prefers-color-scheme: light) {{ :root {{ --bg:#f4f2ee; --surface:#fff; --border:#d8d3c8; --text:#201d17; --dim:#5c574c; --gold:#a1731d; }} }}
  body {{ margin:0; background:var(--bg); color:var(--text); font:16px/1.5 "IBM Plex Sans",system-ui,sans-serif; }}
  main {{ max-width:720px; margin:0 auto; padding:56px 20px; }}
  h1 {{ font:700 34px "Barlow Condensed",sans-serif; text-transform:uppercase; letter-spacing:.02em; margin:0 0 6px; }}
  h1 span {{ color:var(--gold); }}
  .sub {{ color:var(--dim); margin:0 0 30px; }}
  .card {{ display:block; background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:16px 18px;
    margin-bottom:12px; text-decoration:none; color:inherit; }}
  .card:hover, .card:focus-visible {{ border-color:var(--gold); outline:none; }}
  .card h2 {{ font:700 22px "Barlow Condensed",sans-serif; text-transform:uppercase; margin:0; color:var(--gold); }}
  .card p {{ margin:2px 0 0; color:var(--dim); font-size:14px; }}
  footer {{ margin-top:30px; font-size:12.5px; color:var(--dim); }}
  footer a {{ color:var(--gold); }}
</style></head><body><main>
  <h1><span>FF</span> Model</h1>
  <p class="sub">NFL fantasy and props tools, rebuilt from nflverse data on game days.</p>
{items}
  <footer><a href="https://github.com/hadantes-lab/ff-model">Source on GitHub</a></footer>
</main></body></html>
""", encoding="utf-8")
print(f"Wrote {out} with: {', '.join(s for s, *_ in links) or 'nothing'}")
