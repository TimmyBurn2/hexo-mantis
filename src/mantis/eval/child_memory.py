"""`DeviceMemoryProbe` — the eval child's own device-memory readout (RECAL-PREP, R308(g)(ii)).

WHY THIS EXISTS, in the numbers that ordered it. The `eval_child` term of the memory partition
has been measured three times and grown every time: **0.881 GiB** (41 samples), **1.1855**
(709 samples, running maximum flat for the final 30% of a 24-minute drive), **3.5293** (a
3 600 s burst with rounds allowed to COMPLETE). The 2.98x jump falsified a mint. The sitting's
own conclusion, R308(a)(i)'s first ordered finding, is that *STEP 1d as the procedure writes it
cannot measure the term it exists to measure* — and the reason is structural, not a matter of
watching longer.

THE STRUCTURE, DERIVED FROM THIS TREE AND NOT INHERITED FROM THE SITTING'S ATTRIBUTION.

  * The eval child is ONE SPAWN-CONTEXT PROCESS PER ROUND (`pipeline.py::_spawn_worker`).
    Nothing accumulates across rounds inside it, so across-round growth is growth in what a
    round DOES — never a leak in a long-lived process.
  * The GATE BLOCK is the only phase that puts a SECOND model and a SECOND
    `LocalInferenceEngine` on the card (`worker.py::_play_gate_block`), and it is skipped
    WHOLE while `best_snapshot is None` or the gate is not scheduled. So a round before the
    first promotion measures a strictly smaller term than one after it: the term is
    POSTURE-dependent, and a reading taken from a run that never promoted is a floor.
  * `DeployHeadPlayer.select_move` calls `release_cuda_cache()` after EVERY move
    (`arena/deploy_head.py`, pinned by `tests/arena/test_deploy_head_vram.py`). That is why an
    external sampler sees the child sawtooth between roughly 400 and 3 600 MiB rather than
    climb — the peak is a within-move DEMAND peak, not retained cache, and it belongs to
    whichever phase was running when it was taken.

The sitting attributed the growth to "the ladder rungs, which load further bot models". That
attribution has no subject at this sha and this module does not repeat it — see
`RECAL_PREP_FINDINGS.md` in the governance workspace, filed and deliberately NOT fixed here.

WHAT FOLLOWS FOR THE INSTRUMENT: a term that is phase- and posture-dependent cannot be read at
round boundaries alone. This probe marks EVERY phase, keeps RUNNING MAXIMA that are never reset
mid-round, and records the round's posture beside the numbers, so the re-sit can attribute a
peak instead of watching a total.

TWO SINKS, DELIBERATELY.

  1. The round result (`device_memory`), which the parent emits as `eval_round_device_memory`
     from the CHILD's own payload — the structured channel, and the one
     `mantis.diagnostics.eval_child_memory` reads.
  2. One MARKER LINE per phase on the child's stdout. The burst log captures child stdout, and
     this is what carries phase-boundary timestamps so an external `nvidia-smi -l` sample can
     be aligned to a phase. Marker-delimited because the sitting's own `peaks.py` guessed at a
     file's shape, collapsed a whole run into one poll and produced **1 392 GiB on a 16 GiB
     card** — a plausible-looking wrong number that would have been minted against. The reader
     REFUSES a file with no markers rather than reporting from a file it did not understand.

UNCONDITIONAL, with no config key. A knob here would be a second authority over whether a
measurement exists at all, and the cost is four counter reads per phase boundary.

`available: false` IS A PRESENT KEY, NOT A MISSING ONE. On a device with no CUDA counters the
payload ships with every counter `null`. "The device has no counters" and "the code never ran"
must not look the same to a reader; the first is a measurement, the second is a defect.

`t_mono_sec` IS A MEASUREMENT AND IS `null` ON THE UNAVAILABLE ARM, which looks fussy and is
not. The round result carries this payload, and `tests/eval/test_graph_round_encoding.py::
test_dense_v6_round_is_byte_stable_and_deterministic` pins that two runs of the SAME spec return
EQUAL result dicts (only `worker_pid` is popped). A wall-clock reading on a cpu round — where
there is nothing to measure — would break that determinism for a number that measures nothing.
On the available arm the counters are non-deterministic anyway, so no byte-stability claim is
being weakened; it is being kept exactly where it exists.
"""
from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Mapping
from typing import Any

#: The stdout marker. One line per phase, `MARKER<space><json>`. Public because the reader and
#: the tests both key on it, and a marker that two modules spell separately is not a marker.
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
    """The production counter source for `device`, via the module that owns `torch.cuda` for
    the paths a guard fences off.

    The four numbers come from `mantis.util.device.cuda_memory_counters` — two high-water
    (`max_memory_*`) and two instantaneous. The high-water pair is the one that matters: the
    child's peak is a WITHIN-MOVE demand spike (per-move `release_cuda_cache()` is what makes
    it a sawtooth), so a probe that only sampled at phase boundaries would read the troughs.

    THE TORCH ACCESS IS NOT IN THIS FILE, and that is an isolation law rather than a style
    choice: `tests/eval/test_pipeline_isolation.py` bans every `.cuda` attribute in
    `src/mantis/eval/*.py` except the child entry point, because an in-process CUDA eval path
    on the PARENT side is what that law makes unrepresentable. This module is imported by the
    child, so wording the guard around it would be the exact "write documents around a gate"
    shape this repo refuses. The counters live in `mantis.util.device`, which owns `torch.cuda`
    for the FENCED paths — not for the whole repo, which is measurably false
    (`selfplay/graph_collate.py`, `selfplay/inference_server.py`, `train/subsystems.py` and
    `diagnostics/fusion_calibrate.py` all touch it directly).

    The probe additionally keeps a RUNNING MAXIMUM over the two high-water fields. Nothing that
    runs in an eval child calls `reset_peak_memory_stats` — the callers are
    `diagnostics/fusion_calibrate.py`, a train-side oracle and (since WORKER-SWEEP)
    `mantis.util.device.reset_cuda_peak_counters`, which `mantis.diagnostics.worker_sweep` uses
    to open its own per-round windows in its own process. The names are given rather than
    counted, per derive-or-delete: this sentence used to say "the two call sites" and a third
    arrived. The guard is cheap and its absence would be invisible: a foreign reset would
    silently turn every later phase's figure into a fragment of the round, and a figure that
    FELL would be read as memory being released.
    """
    from mantis.util.device import cuda_memory_counters

    def _read() -> dict[str, int]:
        return cuda_memory_counters(device)

    return _read


def make_probe(device: str, *, round_id: str, out: Any = None) -> DeviceMemoryProbe:
    """Build the probe for `device`, arming it only where the counters exist.

    The availability decision is made ONCE, here, and recorded on the payload — never
    re-derived per mark, where a mid-round change would produce a payload whose halves
    disagree about what it measured.
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

    Constructed with its counter source injected: the contract is "read these numbers at a
    phase boundary and keep the running maxima", and that contract is testable on any host.
    Whether CUDA reports the right bytes is torch's business, and a suite that asserted it
    would need a GPU the CI tier does not have — the GPU-side reading is the RE-SIT's
    measurement, and this module must not pretend to take it.
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
        # `flush` because the child may be SIGKILLed at a hard cap and a buffered marker is a
        # measurement that did not survive the thing it was measuring.
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

    A file with no markers RAISES, and a marker line whose payload is not JSON RAISES. Both
    are the `peaks.py` lesson: a reader that guesses at a shape it does not recognise produces
    a number nobody can distinguish from a measurement.
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
