"""The frozen command line: `--events --out [--ladder-state] [--record-dir] [--title]`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .html import render
from .reader import EmptyRunRecord, load_record

#: The rendered page must stay under this on the 35k-step shakedown record (PACKET-DASH-1).
SIZE_CAP_BYTES = 400_000


def main(argv: list[str] | None = None) -> int:
    """Render one run record to one HTML file; 2 on a record that refuses."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", type=Path, required=True, help="the run's JSONL event stream")
    ap.add_argument("--ladder-state", type=Path, default=None,
                    help="the run's eval_ladder_state.json (games and wr per rung per round)")
    ap.add_argument("--record-dir", type=Path, default=None,
                    help="the run-record directory holding collate_dumps/ and logs; omitted, "
                         "the firings input is unmeasured")
    ap.add_argument("--out", type=Path, required=True, help="the HTML file to write")
    ap.add_argument("--title", default=None, help="page title (default: the events file's name)")
    args = ap.parse_args(argv)
    try:
        rec = load_record(args.events, args.ladder_state, args.record_dir)
    except EmptyRunRecord as exc:
        print(f"run_dashboard: refused: {exc}", file=sys.stderr)
        return 2
    page = render(rec, args.title or f"mantis run record — {args.events.name}")
    args.out.write_text(page, encoding="utf-8")
    size = args.out.stat().st_size
    print(f"wrote {args.out} ({size} bytes) from {len(rec.events)} events")
    if size > SIZE_CAP_BYTES:
        print(f"run_dashboard: WARNING the page is {size} bytes, over the {SIZE_CAP_BYTES} cap",
              file=sys.stderr)
    return 0
