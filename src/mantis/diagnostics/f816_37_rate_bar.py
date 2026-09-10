"""Pre-registered memory-corruption rate bar: condemn a host on rate or on location.

Two channels, because "outside the wire arrays" cannot be read from collate dumps alone — a
`GraphContractError` is by construction about the wire, so a dump scan would score a
weights-side corruption as a clean run:

  * IN-WIRE      `collate_dumps/collate_dump_*.json`; these count toward the 3-in-12h rate.
  * OUT-OF-WIRE  checkpoint content-hash mismatches and ring-header / index failures, read
                 from the run's log. Any one condemns immediately, whatever the rate.

Exit codes: 0 = bar clear; 1 = condemned, relocate; 2 = refusing to report, because the scan
could not prove it ran. It lives here rather than in `tools/` so this CLI and the dashboard
panel share one copy of the arithmetic.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

WINDOW_SEC = 12 * 3600
MAX_IN_WINDOW = 3

#: Log signatures for corruption OUTSIDE the wire arrays. Each condemns on its own.
_OUT_OF_WIRE = (
    (re.compile(r"content hash \S+ disagrees with the filename sha8"), "weights"),
    (re.compile(r"anchor sha256 mismatch"), "weights"),
    (re.compile(r"ring header|ring_header|buffer header"), "ring headers"),
    (re.compile(r"index out of range|out-of-range index|illegal move"), "indices"),
)


@dataclass(frozen=True)
class Firing:
    """One recorded corruption event."""

    when: float
    channel: str
    where: str
    detail: str


def _dump_time(path: Path, sidecar: dict[str, object]) -> float:
    """Return epoch seconds for a dump, from its filename stamp, else its mtime.

    The stamp is authoritative because it survives a file copy; mtime does not.
    """
    m = re.search(r"_(\d{10,})\.json$", path.name)
    if m:
        return int(m.group(1)) / 1000.0
    ts = sidecar.get("timestamp")
    if isinstance(ts, (int, float)):
        return float(ts)
    return path.stat().st_mtime


def scan_dumps(record: Path) -> list[Firing]:
    """Return every in-wire firing under a run record, oldest first.

    Raises:
        json.JSONDecodeError: a sidecar is unreadable. Not swallowed — an unparseable dump is
            evidence, not a zero.
    """
    out: list[Firing] = []
    for path in sorted(record.rglob("collate_dump_*.json")):
        sidecar = json.loads(path.read_text(encoding="utf-8"))
        out.append(
            Firing(
                when=_dump_time(path, sidecar),
                channel="in-wire",
                where=str(sidecar.get("path", "unknown")),
                detail=str(sidecar.get("error", ""))[:160],
            )
        )
    return sorted(out, key=lambda f: f.when)


def scan_logs(record: Path) -> list[Firing]:
    """Return every out-of-wire signature in the run's logs, oldest first."""
    out: list[Firing] = []
    for path in sorted(record.rglob("*.log")):
        mtime = path.stat().st_mtime
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            for pattern, where in _OUT_OF_WIRE:
                if pattern.search(line):
                    out.append(
                        Firing(when=mtime, channel="out-of-wire", where=where,
                               detail=line.strip()[:160])
                    )
                    break
    return sorted(out, key=lambda f: f.when)


def worst_window(firings: list[Firing]) -> tuple[int, float]:
    """Return the largest count in any WINDOW_SEC window, and that window's start.

    Rolling over firing times, not calendar buckets: four firings spanning 11 h 59 m violate an
    "in any 12 h" bar even when they straddle two clock buckets.
    """
    best, at = 0, 0.0
    for i, anchor in enumerate(firings):
        n = sum(1 for f in firings[i:] if f.when - anchor.when <= WINDOW_SEC)
        if n > best:
            best, at = n, anchor.when
    return best, at


def evaluate(record: Path) -> int:
    """Print the bar's reading and return its exit code."""
    if not record.is_dir():
        print(f"f816-37 rate bar: REFUSING — {record} is not a directory, so nothing was "
              "scanned and a clean report would be a phantom.", file=sys.stderr)
        return 2

    dumps, logs = scan_dumps(record), scan_logs(record)
    scanned = sum(1 for _ in record.rglob("*"))
    if scanned == 0:
        print(f"f816-37 rate bar: REFUSING — {record} is empty; an empty run record is not a "
              "clean one.", file=sys.stderr)
        return 2

    print(f"R342(b)(iv) rate bar over {record} ({scanned} entries scanned)")
    print(f"  in-wire firings      {len(dumps)}")
    print(f"  out-of-wire firings  {len(logs)}")
    for f in dumps + logs:
        print(f"    [{f.channel}] {f.where}: {f.detail}")

    if logs:
        print(f"\nCONDEMNED (R342(b)(iv)): {len(logs)} firing(s) OUTSIDE the wire arrays — "
              f"{', '.join(sorted({f.where for f in logs}))}. The bar condemns on location "
              "alone, with no reference to the rate and no further ruling needed. Relocate.",
              file=sys.stderr)
        return 1

    n, at = worst_window(dumps)
    print(f"  worst 12 h window    {n} (limit {MAX_IN_WINDOW})")
    if n > MAX_IN_WINDOW:
        print(f"\nCONDEMNED (R342(b)(iv)): {n} in-wire firings inside one 12 h window starting "
              f"at epoch {at:.0f}, over the pre-registered limit of {MAX_IN_WINDOW}. No further "
              "ruling needed. Relocate.", file=sys.stderr)
        return 1

    print("\nBAR CLEAR: the host may keep training under R342(b).")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry.

    Raises:
        SystemExit: argparse exits on a malformed command line.
    """
    ap = argparse.ArgumentParser(description="R342(b)(iv) rate bar over a run record")
    ap.add_argument("record", type=Path, help="run record directory (holds collate_dumps/ and logs)")
    return evaluate(ap.parse_args(argv).record)


if __name__ == "__main__":
    raise SystemExit(main())
