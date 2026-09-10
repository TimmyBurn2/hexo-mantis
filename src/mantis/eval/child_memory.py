"""`DeviceMemoryProbe` — the eval child's own device-memory readout.

The child's memory term is phase- and posture-dependent: one process per round, a gate block
that puts a second model on the card only once a promotion exists, and a per-move
`release_cuda_cache()` that makes the peak a within-move demand spike rather than a climb. So
the probe marks EVERY phase, keeps running maxima that are never reset mid-round, and ships two
sinks — the round result's `device_memory` payload and one stdout marker line per phase, which
is what aligns an external `nvidia-smi` sample to a phase.

`available: false` is a PRESENT key with every counter `null`: "the device has no counters" and
"the code never ran" must not look the same to a reader. `t_mono_sec` is `null` on that arm
because a cpu round's result dict is pinned byte-stable across two runs of the same spec.
"""
from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Mapping
from typing import Any

#: The stdout marker. One line per phase, `MARKER<space><json>`; reader and tests both key on it.
MARKER = "MANTIS_EVAL_MEM"

#: The event the parent emits from the child's payload.
EVENT = "eval_round_device_memory"

_COUNTER_KEYS = (
    "max_memory_allocated_bytes",
    "max_memory_reserved_bytes",
    "memory_allocated_bytes",
    "memory_reserved_bytes",
)


def torch_cuda_reader(device: str) -> Callable[[], dict[str, int]]:
    """Return the production counter reader for `device`.

    The torch access lives in `mantis.util.device`, not here: `tests/eval/test_pipeline_isolation.py`
    bans every `.cuda` attribute under `src/mantis/eval/` outside the child entry point.
    """
    from mantis.util.device import cuda_memory_counters

    def _read() -> dict[str, int]:
        return cuda_memory_counters(device)

    return _read


def make_probe(device: str, *, round_id: str, out: Any = None) -> DeviceMemoryProbe:
    """Build the probe for `device`, arming it only where the counters exist.

    Availability is decided once, here, and recorded on the payload; re-deriving it per mark
    would produce a payload whose halves disagree about what it measured.
    """
    read_fn: Callable[[], dict[str, int]] | None = None
    try:
        from mantis.util.device import cuda_counters_available

        available = cuda_counters_available(device)
    except Exception:  # noqa: BLE001 — an unimportable torch is "no counters", not fatal
        available = False
    if available:
        read_fn = torch_cuda_reader(device)
    return DeviceMemoryProbe(
        device=str(device), round_id=round_id, available=available, read_fn=read_fn,
        out=sys.stdout if out is None else out,
    )


class DeviceMemoryProbe:
    """Phase-boundary device-memory marks for ONE eval round.

    The counter source is injected so the contract is testable without a GPU. Running maxima
    are kept across the round because nothing an eval child runs resets torch's peak counters.
    """

    def __init__(
        self,
        *,
        device: str,
        round_id: str,
        available: bool,
        read_fn: Callable[[], Mapping[str, int]] | None,
        clock: Callable[[], float] = time.monotonic,
        out: Any = None,
    ) -> None:
        self._device = device
        self._round_id = round_id
        self._available = bool(available) and read_fn is not None
        self._read_fn = read_fn
        self._clock = clock
        self._out = out
        self._t0 = clock()
        self._phases: list[dict[str, Any]] = []
        self._peak_alloc: int | None = None
        self._peak_reserved: int | None = None

    def mark(self, phase: str) -> dict[str, Any]:
        """Record one phase boundary; write its marker line; return the record."""
        record: dict[str, Any] = {"phase": phase, "t_mono_sec": None}
        if self._available and self._read_fn is not None:
            record["t_mono_sec"] = round(self._clock() - self._t0, 6)
            live = dict(self._read_fn())
            alloc = int(live["max_memory_allocated_bytes"])
            reserved = int(live["max_memory_reserved_bytes"])
            self._peak_alloc = alloc if self._peak_alloc is None else max(self._peak_alloc, alloc)
            self._peak_reserved = (
                reserved if self._peak_reserved is None
                else max(self._peak_reserved, reserved)
            )
            record.update({
                "max_memory_allocated_bytes": self._peak_alloc,
                "max_memory_reserved_bytes": self._peak_reserved,
                "memory_allocated_bytes": int(live["memory_allocated_bytes"]),
                "memory_reserved_bytes": int(live["memory_reserved_bytes"]),
            })
        else:
            record.update(dict.fromkeys(_COUNTER_KEYS))
        self._phases.append(record)
        self._emit_marker(record)
        return record

    def _emit_marker(self, record: Mapping[str, Any]) -> None:
        if self._out is None:
            return
        line = dict(record)
        line["round_id"] = self._round_id
        line["device"] = self._device
        line["available"] = self._available
        # Flushed: the child may be SIGKILLed at a hard cap, and a buffered marker is a
        # measurement that did not survive the thing it measured.
        print(f"{MARKER} {json.dumps(line, sort_keys=True)}", file=self._out, flush=True)

    def payload(self) -> dict[str, Any]:
        """The round's readout, for the worker result's `device_memory` key."""
        return {
            "available": self._available,
            "device": self._device,
            "round_id": self._round_id,
            "phases": list(self._phases),
            "round_peak_allocated_bytes": self._peak_alloc,
            "round_peak_reserved_bytes": self._peak_reserved,
        }


def parse_marker_lines(text: str) -> list[dict[str, Any]]:
    """Recover the marker records from captured child output. FAILS CLOSED.

    Raises:
        ValueError: no marker line is present, or a marker payload is not readable json.
    """
    records: list[dict[str, Any]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith(MARKER):
            continue
        body = line[len(MARKER):].strip()
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{MARKER} line carries no readable json payload ({exc}); refusing to guess "
                f"at it: {body[:120]!r}"
            ) from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"{MARKER} payload must be a json object, got {type(parsed).__name__}")
        records.append(parsed)
    if not records:
        raise ValueError(
            f"no {MARKER} lines found. This reader does not fall back to guessing at a file's "
            "shape: the substitute reading it would produce is indistinguishable from a "
            "measurement, and that is how a 1 392 GiB peak was once reported for a 16 GiB card."
        )
    return records


__all__ = [
    "EVENT",
    "MARKER",
    "DeviceMemoryProbe",
    "make_probe",
    "parse_marker_lines",
    "torch_cuda_reader",
]
