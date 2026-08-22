"""`python -m mantis.diagnostics.eval_child_memory` — has the eval-child term CONVERGED?

WHAT THIS REPLACES, and why it is a tool rather than a paragraph. `F816_10_BOX_PROCEDURE.md`
STEP 1d samples `nvidia-smi` per process during "any burst that reaches one eval round" and
takes the maximum. That procedure produced **0.881 GiB** (41 samples), then **1.1855** (709
samples, running maximum flat for the final 30% of a 24-minute drive) — and the strengthened
STEP 4 then measured **3.5293**, falsifying the mint the 1.1855 was carried into. The sitting's
own words: *a term measured by watching until it looks flat is not a bound*
(`RECAL_EXIT_2026-08-22.md` §11b).

The replacement is not a longer look. It is a STATED STOPPING RULE, applied by something other
than the person who wants the answer:

    PLATEAU      no round in the trailing window set a new maximum exceeding the previous
                 running maximum by more than `--band-pct`
    GROWING      one did
    (refusal)    fewer than `--plateau-rounds` measured rounds exist

Both parameters are REQUIRED and have no defaults. A default here would be a stopping rule
nobody chose, applied to a mint-critical term — which is the class this tool exists to end, one
layer up.

EXIT CODES, because a sitting gates on them:
    0   PLATEAU
    1   GROWING
    2   REFUSED — no rounds of the expected kind, too few rounds, or an unreadable input

**IT FAILS CLOSED, and that is the load-bearing half.** No events of the expected kind is rc 2
with a named refusal, never "0 rounds, plateau". A marker file with no markers is rc 2. The
sitting's own `peaks.py` split polls on an assumption about a CSV header, collapsed a whole run
into one poll and reported **1 392 GiB on a 16 GiB card**; a plausible-looking wrong number
would have been minted against. A reader that guesses is worse than no reader.

TWO TRANSPORTS, ONE RULE. `--events` reads the run's own JSONL event stream
(`eval_round_device_memory`, emitted by `mantis.eval.pipeline` from the child's own payload);
`--markers` reads captured child stdout (`MANTIS_EVAL_MEM` lines) for a sitting that has only a
burst log. Exactly one may be given: a tool that silently preferred one input over another
would answer about a file the operator did not think it was reading.

UNMEASURED ROUNDS ARE REPORTED, NEVER DROPPED. A round whose child had no CUDA counters ships
`available: false` with every counter `null`. It is listed, counted and named in the readout,
and excluded from the verdict by name — because silently dropping it would bias the series
without saying so, and "we had no counters that round" is itself a finding about the drive.

EVERY FIGURE CARRIES ITS SAMPLING LIMIT. Rounds observed and the wall seconds they cover print
beside the peaks, per the block's own convention.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mantis.eval.child_memory import EVENT, MARKER, parse_marker_lines

PLATEAU = "PLATEAU"
GROWING = "GROWING"

#: The refusal exit code. `2` is argparse's own usage-error code, and this tool's refusals are
#: the same kind of thing: the caller did not give it something it can answer about.
RC_REFUSED = 2

GIB = 1024 ** 3


class NoRoundsFoundError(ValueError):
    """The input carried no rounds of the expected kind."""


class InsufficientRoundsError(ValueError):
    """Fewer measured rounds than the stopping rule needs. NOT a verdict."""


@dataclass(frozen=True)
class RoundReading:
    """One round's readout. `peak_bytes` is `None` exactly when `available` is False."""

    round_id: str
    step: int | None
    available: bool
    peak_bytes: int | None
    reserved_peak_bytes: int | None
    wall_sec: float | None
    phases: tuple[str, ...]


def _reading_from_payload(payload: Any, *, round_id: str, step: int | None) -> RoundReading:
    phases = payload.get("phases") or []
    stamps = [p.get("t_mono_sec") for p in phases if p.get("t_mono_sec") is not None]
    return RoundReading(
        round_id=round_id,
        step=step,
        available=bool(payload.get("available")),
        peak_bytes=payload.get("round_peak_allocated_bytes"),
        reserved_peak_bytes=payload.get("round_peak_reserved_bytes"),
        wall_sec=(max(stamps) - min(stamps)) if stamps else None,
        phases=tuple(str(p.get("phase")) for p in phases),
    )


def read_rounds_from_events(text: str) -> list[RoundReading]:
    """Extract the per-round readings from a JSONL event stream.

    A line that is not JSON is SKIPPED — a run log carries lines that are not events, and
    refusing on one would make the tool unusable against real captures. The refusal comes from
    finding no ROUNDS, which is the thing the verdict actually needs, not from finding a line
    the reader could not parse.
    """
    rounds: list[RoundReading] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("event") != EVENT:
            continue
        payload = event.get("device_memory")
        if not isinstance(payload, dict):
            raise NoRoundsFoundError(
                f"an {EVENT!r} event carries no `device_memory` object: {line[:160]!r}. The "
                "payload is the measurement; an event without one is a producer defect, and "
                "reading past it would report a series with a hole nobody could see."
            )
        rounds.append(_reading_from_payload(
            payload, round_id=str(event.get("round_id")), step=event.get("step"),
        ))
    if not rounds:
        raise NoRoundsFoundError(
            f"no {EVENT!r} events found. This tool does not fall back to a substitute "
            "reading: an eval-child term inferred from something other than the child's own "
            "counters is exactly what has under-measured it on every previous occasion."
        )
    return rounds


def read_rounds_from_markers(text: str) -> list[RoundReading]:
    """Extract the per-round readings from captured child stdout markers.

    `parse_marker_lines` refuses a file with no markers; this function then groups the phase
    records by `round_id` IN FIRST-APPEARANCE ORDER. Order matters — the verdict is about a
    trailing window — and the child writes its marks in phase order, so first appearance is
    round order.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in parse_marker_lines(text):
        grouped.setdefault(str(record.get("round_id")), []).append(record)
    rounds: list[RoundReading] = []
    for round_id, records in grouped.items():
        peaks = [r.get("max_memory_allocated_bytes") for r in records]
        reserved = [r.get("max_memory_reserved_bytes") for r in records]
        measured = [p for p in peaks if p is not None]
        rounds.append(RoundReading(
            round_id=round_id,
            step=None,
            available=any(bool(r.get("available")) for r in records),
            peak_bytes=max(measured) if measured else None,
            reserved_peak_bytes=max([r for r in reserved if r is not None], default=None),
            wall_sec=(max(r["t_mono_sec"] for r in records)
                      - min(r["t_mono_sec"] for r in records)) if records else None,
            phases=tuple(str(r.get("phase")) for r in records),
        ))
    return rounds


def classify(peaks: list[float] | list[int], *, plateau_rounds: int, band_pct: float) -> str:
    """Apply the stopping rule to a series of per-round peaks, oldest first.

    The window is TRAILING and the comparison is against the running maximum BEFORE each
    round in it. Growth outside the window is history rather than a live trend — otherwise a
    series that ever rose could never converge, and the rule would have exactly one arm.

    Refuses rather than answering on a series shorter than the window. That refusal is the
    reason this is a function and not an eyeball: the shape it declines to judge is exactly
    the shape STEP 1d has twice reported a bound from.
    """
    if plateau_rounds < 1:
        raise ValueError(f"plateau_rounds must be >= 1, got {plateau_rounds}")
    if len(peaks) < plateau_rounds:
        raise InsufficientRoundsError(
            f"the stopping rule needs {plateau_rounds} measured rounds and the series has "
            f"{len(peaks)}. A verdict from fewer is the reading this tool exists to refuse: "
            "the eval-child term looked converged at 41 samples and at 709, and was 2.98x "
            "larger the first time a round was allowed to complete."
        )
    window_start = len(peaks) - plateau_rounds
    running = max(peaks[:window_start]) if window_start else peaks[0]
    for index in range(window_start, len(peaks)):
        value = peaks[index]
        if value > running * (1.0 + band_pct / 100.0):
            return GROWING
        running = max(running, value)
    return PLATEAU


def _fmt_gib(value: int | float | None) -> str:
    return "unmeasured" if value is None else f"{value / GIB:.4f} GiB"


def _render(rounds: list[RoundReading], *, plateau_rounds: int, band_pct: float,
            verdict: str | None, refusal: str | None, out: Any) -> None:
    measured = [r for r in rounds if r.available and r.peak_bytes is not None]
    unmeasured = [r for r in rounds if r not in measured]
    wall = sum(r.wall_sec or 0.0 for r in rounds)
    print(f"rule: plateau_rounds={plateau_rounds} band_pct={band_pct:g}", file=out)
    print(
        f"sample: rounds_observed={len(rounds)} rounds_measured={len(measured)} "
        f"rounds_unmeasured={len(unmeasured)} wall_sec={wall:.1f}",
        file=out,
    )
    for reading in rounds:
        flag = "" if reading.available and reading.peak_bytes is not None else "  [unmeasured]"
        print(
            f"  {reading.round_id:>24}  step={reading.step}  "
            f"peak_allocated={_fmt_gib(reading.peak_bytes)}  "
            f"peak_reserved={_fmt_gib(reading.reserved_peak_bytes)}  "
            f"phases={len(reading.phases)}  wall_sec="
            f"{'n/a' if reading.wall_sec is None else format(reading.wall_sec, '.1f')}{flag}",
            file=out,
        )
    if measured:
        largest = max(r.peak_bytes or 0 for r in measured)
        print(
            f"largest measured round peak: {_fmt_gib(largest)} over {len(measured)} measured "
            f"round(s) — the sampling limit is that count and that wall time, not a bound",
            file=out,
        )
    if verdict is not None:
        print(f"VERDICT: {verdict}", file=out)
    if refusal is not None:
        print(f"REFUSED: {refusal}", file=out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mantis.diagnostics.eval_child_memory", description=__doc__,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--events", help="run event JSONL carrying eval_round_device_memory")
    source.add_argument("--markers", help=f"captured child stdout carrying {MARKER} lines")
    # NO DEFAULTS, deliberately (see the module docstring): a stopping rule nobody chose,
    # applied to a mint-critical term, is the defect this tool exists to end one layer up.
    parser.add_argument("--plateau-rounds", type=int, required=True,
                        help="trailing window, in measured rounds")
    parser.add_argument("--band-pct", type=float, required=True,
                        help="a new maximum beyond this %% of the running max is GROWING")
    args = parser.parse_args(argv)

    path = Path(args.events or args.markers)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"REFUSED: cannot read {path}: {exc}", file=sys.stderr)
        return RC_REFUSED
    try:
        rounds = (read_rounds_from_events(text) if args.events
                  else read_rounds_from_markers(text))
    except (NoRoundsFoundError, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return RC_REFUSED

    peaks = [r.peak_bytes for r in rounds if r.available and r.peak_bytes is not None]
    try:
        verdict = classify(peaks, plateau_rounds=args.plateau_rounds, band_pct=args.band_pct)
    except InsufficientRoundsError as exc:
        _render(rounds, plateau_rounds=args.plateau_rounds, band_pct=args.band_pct,
                verdict=None, refusal=str(exc), out=sys.stdout)
        print(f"REFUSED: {exc}", file=sys.stderr)
        return RC_REFUSED
    _render(rounds, plateau_rounds=args.plateau_rounds, band_pct=args.band_pct,
            verdict=verdict, refusal=None, out=sys.stdout)
    return 0 if verdict == PLATEAU else 1


__all__ = [
    "EVENT",
    "GROWING",
    "PLATEAU",
    "RC_REFUSED",
    "InsufficientRoundsError",
    "NoRoundsFoundError",
    "RoundReading",
    "classify",
    "main",
    "read_rounds_from_events",
    "read_rounds_from_markers",
]

if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
