"""
rederive_props.py
=================
Puts the current page template and the lean / Underdog-entry logic onto a props page that is ALREADY PUBLISHED, without pulling any
odds (zero Odds API credits). The finished page carries its own data (the props, the lines, the games), so this reads it back out,
re-derives the parts that depend only on that data and on the tracking log (the leans, the Underdog entries, the entry replay), and
renders the page again with the current template.

    python rederive_props.py site/props/index.html            # rewrites that page in place
    python rederive_props.py old.html --out new.html

Used by the Republish workflow. The lines, probabilities and games are exactly the ones the original run pulled; nothing is re-simulated.
"""

import argparse
import json
import pathlib
import re
import sys

import pandas as pd

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "web"))

BLOBS = {"DATA": "__PROPS_JSON__", "TRACK": "__TRACK_RECORD_JSON__", "PLAYER_HISTORY": "__PLAYER_HISTORY_JSON__"}


def extract_blobs(html: str) -> dict:
    """{"DATA": text, "TRACK": text, "PLAYER_HISTORY": text}: the JSON the page embeds as `const NAME = ...;`."""
    dec = json.JSONDecoder()
    out = {}
    for name in BLOBS:
        m = re.search(r"const " + name + r" = ", html)
        if not m:
            raise ValueError(f"the page has no embedded {name}")
        obj, _ = dec.raw_decode(html[m.end():])
        out[name] = json.dumps(obj, separators=(",", ":"))
    return out


def rederive(data: dict, log_df: pd.DataFrame, summary=None, sched=None) -> dict:
    """Leans, entries and the replay, from the data a page already holds. Returns the updated copy."""
    import lean
    import parlays
    import pickem_study
    import player_confidence

    season, week = int(data["season"]), int(data["week"])
    conf_records = player_confidence.lookup(player_confidence.player_records(log_df, season), season)
    data["lean_info"] = lean.annotate(data["props"], log_df, (data.get("context") or {}).get("dvp"), conf_records, season)
    data["parlays"] = parlays.build_all(data["props"], data.get("games") or [], summary, sched=sched, season=season, week=week)
    try:
        data["pickem_backtest"] = pickem_study.stack_backtest(log_df)
    except Exception as ex:                                              # the replay is a footnote; never block the page for it
        print(f"entry replay skipped ({ex})", file=sys.stderr)
        data["pickem_backtest"] = None
    return data


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("page", help="a built props page (props/index.html)")
    ap.add_argument("--out", help="write here instead of over the input")
    ap.add_argument("--template", default=str(HERE / "web" / "props_template.html"))
    ap.add_argument("--no-schedule", action="store_true", help="skip loading the schedule history (teaser legs are then left empty)")
    args = ap.parse_args(argv)

    import props_tracker
    import build_props_page

    page = pathlib.Path(args.page)
    blobs = extract_blobs(page.read_text(encoding="utf-8"))
    data = json.loads(blobs["DATA"])
    log_df = props_tracker._load()
    summary, sched = None, None
    try:
        import weekly_report
        summary = pd.read_csv(weekly_report.SUMMARY_FILE) if weekly_report.SUMMARY_FILE.exists() else None
    except Exception:
        pass
    if not args.no_schedule:
        try:
            import nflreadpy as nfl
            sched = nfl.load_schedules(list(range(2014, int(data["season"]) + 1))).to_pandas()
        except Exception as ex:
            print(f"schedule history unavailable ({ex}); teaser legs left out", file=sys.stderr)
    data = rederive(data, log_df, summary, sched)
    template = pathlib.Path(args.template).read_text(encoding="utf-8")
    html = build_props_page.render(template, json.dumps(data), blobs["TRACK"], blobs["PLAYER_HISTORY"])
    out = pathlib.Path(args.out) if args.out else page
    out.write_text(html, encoding="utf-8")
    n = sum(1 for p in data["props"] if p.get("lean"))
    print(f"Wrote {out}: {n} of {len(data['props'])} props with a lean, {len(data['parlays']['pickem']['entries'])} Underdog entries.")


if __name__ == "__main__":
    main()
