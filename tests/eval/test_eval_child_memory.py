"""Pin the eval child's per-phase device-memory readout.

The budget term grew every time it was measured — 0.881 GiB at 41 samples, 1.1855 at 709, 3.5293
once rounds were allowed to complete — because a term measured by watching until it looks flat is
not a bound. The readout is per PHASE and carries the posture beside the numbers: the child is one
process per round, and the GATE BLOCK, the only phase that puts a second engine on the card, is
skipped whole while there is no anchor.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from mantis.eval.child_memory import (
    MARKER,
    DeviceMemoryProbe,
    parse_marker_lines,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER_SRC = REPO_ROOT / "src" / "mantis" / "eval" / "worker.py"


class _FakeCounters:
    """Stand in deterministically for the four device counters: the probe's contract is testable
    on any host, while whether the device reports the right bytes needs a GPU."""

    def __init__(self, series: list[tuple[int, int]]) -> None:
        self._series = list(series)
        self._i = -1

    def step(self) -> None:
        self._i = min(self._i + 1, len(self._series) - 1)

    def allocated(self) -> int:
        return self._series[max(self._i, 0)][0]

    def reserved(self) -> int:
        return self._series[max(self._i, 0)][1]


def _probe(series, *, out=None):
    counters = _FakeCounters(series)
    ticks = iter(range(1000))

    def _read():
        counters.step()
        # The fake reports its instantaneous pair as the high-water too, so what the rows below
        # observe is the probe's OWN running maximum.
        return {
            "max_memory_allocated_bytes": counters.allocated(),
            "max_memory_reserved_bytes": counters.reserved(),
            "memory_allocated_bytes": counters.allocated(),
            "memory_reserved_bytes": counters.reserved(),
        }

    return DeviceMemoryProbe(
        device="cuda", round_id="r000001_1", available=True,
        read_fn=_read, clock=lambda: float(next(ticks)), out=out,
    )


def test_cm03_the_round_peak_is_a_running_maximum_across_phases():
    probe = _probe([(10, 100), (50, 500), (20, 200)])
    probe.mark("round_start")
    probe.mark("gate_block")
    probe.mark("round_end")
    payload = probe.payload()
    assert payload["round_peak_allocated_bytes"] == 50
    assert payload["round_peak_reserved_bytes"] == 500


def test_cm03_each_phase_carries_the_running_max_at_that_boundary():
    """Prove each phase carries the running max: a figure that fell means a reset."""
    probe = _probe([(10, 100), (50, 500), (20, 200)])
    for phase in ("round_start", "gate_block", "round_end"):
        probe.mark(phase)
    maxima = [p["max_memory_allocated_bytes"] for p in probe.payload()["phases"]]
    assert maxima == sorted(maxima), maxima
    assert maxima == [10, 50, 50]


def test_cm03_the_instantaneous_pair_is_recorded_beside_the_maxima():
    """Prove the instantaneous pair sits beside the maxima: where they disagree the larger
    governs, and a reader cannot apply that rule against one number."""
    probe = _probe([(10, 100), (50, 500), (20, 200)])
    for phase in ("round_start", "gate_block", "round_end"):
        probe.mark(phase)
    last = probe.payload()["phases"][-1]
    assert last["memory_allocated_bytes"] == 20
    assert last["max_memory_allocated_bytes"] == 50


def test_cm01_phases_are_recorded_in_the_order_they_were_marked():
    probe = _probe([(1, 1)] * 5)
    for phase in ("round_start", "gate_block", "rung:sealbot_d5", "random_floor", "round_end"):
        probe.mark(phase)
    assert [p["phase"] for p in probe.payload()["phases"]] == [
        "round_start", "gate_block", "rung:sealbot_d5", "random_floor", "round_end",
    ]


def test_cm01_the_monotonic_clock_is_recorded_so_an_external_sampler_can_be_aligned():
    """Prove the monotonic clock is recorded, so an external sampler's spikes can be joined to
    the phase they belonged to."""
    probe = _probe([(1, 1)] * 3)
    for phase in ("round_start", "gate_block", "round_end"):
        probe.mark(phase)
    stamps = [p["t_mono_sec"] for p in probe.payload()["phases"]]
    assert stamps == sorted(stamps)
    assert len(set(stamps)) == 3


def test_cm02_an_unavailable_device_still_emits_the_payload_with_every_counter_null():
    probe = DeviceMemoryProbe(
        device="cpu", round_id="r1", available=False,
        read_fn=None, clock=lambda: 0.0, out=None,
    )
    probe.mark("round_start")
    probe.mark("round_end")
    payload = probe.payload()
    assert payload["available"] is False
    assert payload["device"] == "cpu"
    assert payload["round_peak_allocated_bytes"] is None
    assert len(payload["phases"]) == 2
    for phase in payload["phases"]:
        assert phase["max_memory_allocated_bytes"] is None
        assert phase["phase"]


def test_cm02_the_payload_key_set_is_the_same_on_both_arms():
    """Prove the payload key set is the same on both arms; a reader that has to branch on which
    keys exist will eventually branch wrong."""
    available = _probe([(1, 1)])
    available.mark("round_start")
    unavailable = DeviceMemoryProbe(device="cpu", round_id="r1", available=False,
                                    read_fn=None, clock=lambda: 0.0, out=None)
    unavailable.mark("round_start")
    assert available.payload().keys() == unavailable.payload().keys()
    assert (available.payload()["phases"][0].keys()
            == unavailable.payload()["phases"][0].keys())


def test_cm04_each_mark_writes_one_marker_line_of_json(capsys):
    import sys

    probe = _probe([(10, 100), (50, 500)], out=sys.stdout)
    probe.mark("round_start")
    probe.mark("gate_block")
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith(MARKER)]
    assert len(lines) == 2
    body = json.loads(lines[1][len(MARKER):].strip())
    assert body["phase"] == "gate_block"
    assert body["round_id"] == "r000001_1"
    assert body["max_memory_allocated_bytes"] == 50


def test_cm04_the_reader_refuses_a_file_with_no_markers(tmp_path):
    """Prove the reader refuses a file with no markers; one that guessed at a file's shape once
    produced 1 392 GiB on a 16 GiB card."""
    victim = tmp_path / "nothing.log"
    victim.write_text("some ordinary run output\nwith no markers at all\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        parse_marker_lines(victim.read_text(encoding="utf-8"))
    assert MARKER in str(exc.value)


def test_cm04_the_reader_recovers_exactly_what_the_probe_wrote(capsys):
    import sys

    probe = _probe([(10, 100), (50, 500), (20, 200)], out=sys.stdout)
    for phase in ("round_start", "gate_block", "round_end"):
        probe.mark(phase)
    recovered = parse_marker_lines(capsys.readouterr().out)
    assert [r["phase"] for r in recovered] == ["round_start", "gate_block", "round_end"]
    assert [r["max_memory_allocated_bytes"] for r in recovered] == [10, 50, 50]


def test_cm04_a_marker_line_with_unparsable_json_is_a_named_refusal_not_a_skip():
    with pytest.raises(ValueError) as exc:
        parse_marker_lines(f"{MARKER} {{not json}}\n")
    assert "json" in str(exc.value).lower()


def _engine_constructions_per_function(source: str) -> dict[str, int]:
    """Count engine constructions inside each top-level function, from structure not text."""
    tree = ast.parse(source)
    counts: dict[str, int] = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        n = 0
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call):
                func = inner.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                if name == "LocalInferenceEngine":
                    n += 1
        counts[node.name] = n
    return counts


def test_cm05_the_gate_block_is_the_only_phase_that_builds_a_second_engine():
    """Prove the gate block is the only phase that builds a second engine; one added elsewhere
    would silently make the instrument's phase attribution incomplete."""
    counts = _engine_constructions_per_function(WORKER_SRC.read_text(encoding="utf-8"))
    builders = {name: n for name, n in counts.items() if n}
    assert builders == {"run_round": 1, "_play_gate_block": 1}, builders


def test_cm05_the_census_has_a_positive_control():
    planted = (
        "def _play_new_phase(spec):\n"
        "    e = LocalInferenceEngine(1, 2)\n"
        "    return e\n"
    )
    assert _engine_constructions_per_function(planted) == {"_play_new_phase": 1}


def test_cm05_the_gate_block_is_skipped_whole_when_there_is_no_anchor():
    """Prove the gate block is skipped whole with no anchor, so a round before the first
    promotion measures a strictly smaller term and cannot bound the one after it."""
    from mantis.eval import worker

    source = ast.parse(WORKER_SRC.read_text(encoding="utf-8"))
    gate = next(n for n in source.body
                if isinstance(n, ast.FunctionDef) and n.name == "_play_gate_block")
    first = gate.body[1] if isinstance(gate.body[0], ast.Expr) else gate.body[0]
    assert isinstance(first, ast.If), (
        "the anchor check must be the gate block's FIRST statement: a second engine built "
        "before it would make the no-anchor round pay for an anchor it does not have"
    )
    assert isinstance(first.body[0], ast.Return)
    assert worker._play_gate_block is not None
