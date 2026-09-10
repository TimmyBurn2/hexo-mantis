# R8 justify: the stopping RULE, the two readers that feed it and the renderer are one unit — a
# verdict is a function of a stated sample, so a reader living apart from the rule could hand it
# a series shaped differently from the one the rule was pre-registered against.
"""`python -m mantis.diagnostics.eval_child_memory` — has the eval-child term CONVERGED?

A STATED STOPPING RULE, applied by something other than the person who wants the answer, in
place of "sample until the maximum looks flat" — which reported 0.881 GiB (41 samples) and then
1.1855 GiB (709 samples) for a term later measured at 3.5293 GiB.

    PLATEAU      no round in the trailing window set a new maximum exceeding the previous
                 running maximum by more than `--band-pct`
    GROWING      one did
    (refusal)    fewer than `--plateau-rounds` measured rounds exist

Both parameters are REQUIRED and have no defaults. Exit 0 PLATEAU, 1 GROWING, 2 REFUSED (no
rounds of the expected kind, too few rounds, or an unreadable input): it FAILS CLOSED, never
"0 rounds, plateau", because a reader that guesses is worse than no reader — an earlier one
collapsed a whole run into one poll and reported 1 392 GiB on a 16 GiB card.

TWO TRANSPORTS, ONE RULE: `--events` reads the run's own JSONL event stream and `--markers`
captured child stdout, and exactly one may be given. A round whose child had no CUDA counters
is listed, counted and named but excluded from the verdict, and every figure prints the rounds
observed and the wall seconds they cover.
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

#: The refusal exit code, `2` for argparse's own reason: the caller gave nothing answerable.
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
    """Extract the per-round readings from a JSONL event stream; a non-JSON line is SKIPPED,
    since the refusal comes from finding no ROUNDS.
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
    """Extract the per-round readings from captured child stdout markers, grouped by `round_id`
    IN FIRST-APPEARANCE ORDER — the verdict is about a trailing window, and the child writes its
    marks in phase order.
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
            # Absent stamps are filtered: `max()` over a list containing `None` raises
            # TypeError, which escaped `--markers` and exited with GROWING's code.
            wall_sec=_span(r.get("t_mono_sec") for r in records),
            phases=tuple(str(r.get("phase")) for r in records),
        ))
    return rounds


def _span(stamps: Any) -> float | None:
    """`max - min` over the stamps that EXIST, or `None` when fewer than two do; one stamp is an
    instant, not a span."""
    values = [float(s) for s in stamps if s is not None]
    return (max(values) - min(values)) if len(values) >= 2 else None


def classify(peaks: list[float] | list[int], *, plateau_rounds: int, band_pct: float) -> str:
    """Apply the stopping rule to a series of per-round peaks, oldest first.

    The window is TRAILING and each round is compared against the running maximum BEFORE it, or
    a series that ever rose could never converge. A series shorter than the window is REFUSED.
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
    # Summed over the rounds that HAVE a wall, with the count printed beside it: counting an
    # unmeasurable round as zero seconds made the printed total an understatement.
    timed = [r.wall_sec for r in rounds if r.wall_sec is not None]
    wall_str = f"{sum(timed):.1f} over {len(timed)}/{len(rounds)} timed round(s)" if timed \
        else f"unmeasured (0/{len(rounds)} rounds carry a wall)"
    print(f"rule: plateau_rounds={plateau_rounds} band_pct={band_pct:g}", file=out)
    print(
        f"sample: rounds_observed={len(rounds)} rounds_measured={len(measured)} "
        f"rounds_unmeasured={len(unmeasured)} wall_sec={wall_str}",
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
    # NO DEFAULTS, deliberately: a stopping rule nobody chose, applied to a mint-critical term.
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
