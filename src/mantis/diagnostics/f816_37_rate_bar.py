"""R342(b)(iv) — the pre-registered rate bar that CONDEMNS the host with no further ruling.

R342 stays R341(a)'s condemnation and lets this host train, on conditions. This is condition
(iv), and it is the one with teeth: *"more than 3 firings in any 12 h of the run, or any firing
outside the wire arrays (weights, ring headers, indices), CONDEMNS the host again with no
further ruling needed — the operator relocates."*

TWO CHANNELS, BECAUSE "OUTSIDE THE WIRE ARRAYS" CANNOT BE READ FROM COLLATE DUMPS ALONE.
A `GraphContractError` is by construction about the wire, so a scan of `collate_dumps/` can
only ever report in-wire firings and would score a weights-side corruption as a clean run.
The two channels are therefore:

  * IN-WIRE   `collate_dumps/collate_dump_*.json` — the 1-in-1 checks on the three collate
              paths (R342(b)(i)). These count toward the 3-in-12h rate.
  * OUT-OF-WIRE  a checkpoint content-hash mismatch (the weights channel, R342(b)(ii)), and
              ring-header / index failures, read from the run's log. ANY of these condemns
              immediately, without reference to the rate.

WHY A RATE AND NOT A COUNT. One firing is the bound run6 starts under and R342(a) accepts it as
an uptime cost. Four in twelve hours is a different machine state, and the bar exists so that
judgement is made by arithmetic on the record rather than by whoever is watching at the time.

EXIT CODES. 0 = bar clear; 1 = CONDEMNED, relocate; 2 = REFUSING to report — the scan could not
prove it ran. A checker that finds nothing because it looked nowhere must fail, not pass.

IT LIVES UNDER `mantis.diagnostics` AND NOT `tools/` because two surfaces read it — this CLI and
the dashboard's R342(b)(v) panel — and `tools/` is not an importable package. The alternative was
a `sys.path` write, which R5 forbids outright, or a second copy of the arithmetic, which is the
exact way the bar and the panel would come to disagree about how many firings a run had.
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
    """Epoch seconds for a dump, from its filename stamp, else its mtime.

    `write_collate_dump` names files `collate_dump_{round_id}_{ms}.json`, so the stamp is
    authoritative and survives a file copy; mtime does not, and is the fallback only.
    """
    m = re.search(r"_(\d{10,})\.json$", path.name)
    if m:
        return int(m.group(1)) / 1000.0
    ts = sidecar.get("timestamp")
    if isinstance(ts, (int, float)):
        return float(ts)
    return path.stat().st_mtime


def scan_dumps(record: Path) -> list[Firing]:
    """Every in-wire firing under a run record, oldest first.

    Raises:
        json.JSONDecodeError: a sidecar is unreadable. Deliberately NOT swallowed — a dump that
            cannot be parsed is evidence that must be looked at, not a zero.
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
    """Every out-of-wire signature in the run's logs, oldest first."""
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
    """Largest count in any WINDOW_SEC window, and that window's start.

    Rolling over firing times rather than fixed calendar buckets: a bar stated as "in any 12 h"
    is violated by four firings spanning 11 h 59 m even when they straddle two clock buckets.
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
