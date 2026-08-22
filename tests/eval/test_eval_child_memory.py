"""⊕ RECAL-PREP item 2 — the eval child's device-memory readout (R308(g)(ii)).

Written by ORACLE **before** the feature exists; every row below was red first.

WHAT THE SITTING MEASURED, AND WHY A ROUND-BOUNDARY READING CANNOT SEE IT. The `eval_child`
budget term was measured three times and grew every time: 0.881 GiB (41 samples), 1.1855 (709
samples), 3.5293 (rounds allowed to complete). `RECAL_EXIT_2026-08-22.md` §11b states the
consequence — *a term measured by watching until it looks flat is not a bound*. This suite
pins the instrument that replaces the watching.

THE STRUCTURE THE INSTRUMENT IS SHAPED BY, and each half is asserted here rather than assumed:

  * the eval child is ONE SPAWN-CONTEXT PROCESS PER ROUND (`eval/pipeline.py::_spawn_worker`),
    so nothing accumulates ACROSS rounds inside it — across-round growth is growth in what a
    round DOES, which is why the readout is per PHASE and not per round boundary;
  * the GATE BLOCK is the only phase that puts a SECOND model and a SECOND
    `LocalInferenceEngine` on the card (`eval/worker.py::_play_gate_block`), and it is skipped
    whole while there is no anchor — so the term is POSTURE-dependent, and the readout carries
    the posture beside the numbers or a reader cannot attribute a peak.

The defect each row is the ONLY witness to:

- **CM-01** — a readout that exists but is never taken at the phase that moves the number.
- **CM-02** — "absent" and "unmeasured" looking the same to a reader. On a device with no
  counters the payload is PRESENT with `available: false` and every counter `null`.
- **CM-03** — a peak that is reset between phases, so no figure is the round's peak.
- **CM-04** — the marker channel drifting from the structured one, or a marker line that a
  reader has to guess the shape of (the sitting's `peaks.py`: a whole run collapsed into one
  poll and produced 1 392 GiB on a 16 GiB card).
- **CM-05** — the claim "the gate block is the second-engine phase" being prose. It is derived
  from the tree by an `ast` census, with a positive control.
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
    """A deterministic stand-in for `torch.cuda`'s four counters.

    A FAKE and not a monkeypatch of torch: the probe's contract is "read these four numbers
    at a phase boundary and keep the running maxima", and that contract is testable on any
    host. Whether CUDA reports the right bytes is torch's business, and asserting it here
    would need a GPU the CI tier does not have — the GPU-side reading is the RE-SIT's
    measurement, and this suite must not pretend to take it.
    """

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
        # The fake reports its instantaneous pair as the high-water too, so the probe's own
        # running maximum is what the rows below observe. A real `torch.cuda` reader supplies
        # a genuine high-water; the probe's guard is what keeps a figure from ever falling.
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


# ── CM-03: running maxima, never reset ───────────────────────────────────────────────────
def test_cm03_the_round_peak_is_a_running_maximum_across_phases():
    probe = _probe([(10, 100), (50, 500), (20, 200)])
    probe.mark("round_start")
    probe.mark("gate_block")
    probe.mark("round_end")
    payload = probe.payload()
    assert payload["round_peak_allocated_bytes"] == 50
    assert payload["round_peak_reserved_bytes"] == 500


def test_cm03_each_phase_carries_the_running_max_at_that_boundary():
    """A phase whose figure fell would mean the counters were reset under the reader."""
    probe = _probe([(10, 100), (50, 500), (20, 200)])
    for phase in ("round_start", "gate_block", "round_end"):
        probe.mark(phase)
    maxima = [p["max_memory_allocated_bytes"] for p in probe.payload()["phases"]]
    assert maxima == sorted(maxima), maxima
    assert maxima == [10, 50, 50]


def test_cm03_the_instantaneous_pair_is_recorded_beside_the_maxima():
    """Both readings, because the block's rule is that where they disagree the larger
    governs — and a reader cannot apply that rule against one number."""
    probe = _probe([(10, 100), (50, 500), (20, 200)])
    for phase in ("round_start", "gate_block", "round_end"):
        probe.mark(phase)
    last = probe.payload()["phases"][-1]
    assert last["memory_allocated_bytes"] == 20
    assert last["max_memory_allocated_bytes"] == 50


# ── CM-01: every phase is marked, in order ───────────────────────────────────────────────
def test_cm01_phases_are_recorded_in_the_order_they_were_marked():
    probe = _probe([(1, 1)] * 5)
    for phase in ("round_start", "gate_block", "rung:sealbot_d5", "random_floor", "round_end"):
        probe.mark(phase)
    assert [p["phase"] for p in probe.payload()["phases"]] == [
        "round_start", "gate_block", "rung:sealbot_d5", "random_floor", "round_end",
    ]


def test_cm01_the_monotonic_clock_is_recorded_so_an_external_sampler_can_be_aligned():
    """The one thing an in-process counter cannot give the sitting is WHICH PHASE a sampled
    spike belonged to. The timestamps are what join the two records."""
    probe = _probe([(1, 1)] * 3)
    for phase in ("round_start", "gate_block", "round_end"):
        probe.mark(phase)
    stamps = [p["t_mono_sec"] for p in probe.payload()["phases"]]
    assert stamps == sorted(stamps)
    assert len(set(stamps)) == 3


# ── CM-02: absent and unmeasured are distinguishable ─────────────────────────────────────
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
    """A reader that has to branch on which keys exist will eventually branch wrong."""
    available = _probe([(1, 1)])
    available.mark("round_start")
    unavailable = DeviceMemoryProbe(device="cpu", round_id="r1", available=False,
                                    read_fn=None, clock=lambda: 0.0, out=None)
    unavailable.mark("round_start")
    assert available.payload().keys() == unavailable.payload().keys()
    assert (available.payload()["phases"][0].keys()
            == unavailable.payload()["phases"][0].keys())


# ── CM-04: the marker channel ────────────────────────────────────────────────────────────
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
    """The `peaks.py` lesson, made structural: a reader that guesses at a file's shape
    produced 1 392 GiB on a 16 GiB card and would have been minted against."""
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


# ── CM-05: the gate block is the second-engine phase, DERIVED from the tree ──────────────
def _engine_constructions_per_function(source: str) -> dict[str, int]:
    """Count `LocalInferenceEngine(...)` constructions inside each top-level function.

    STRUCTURE, not text: a call through an alias or an attribute is still a `Call` whose
    callee name is what this reads, and a docstring naming the class is not a construction.
    """
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
    """The posture-dependence of the eval-child term, derived rather than asserted.

    `run_round` builds the candidate engine once; `_play_gate_block` builds a SECOND one for
    the anchor. Every other phase helper builds none. If a future change adds an engine to
    another phase, the instrument's phase attribution silently stops being complete — and
    this row is what says so.
    """
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
    """The other half of the posture: with no anchor the second engine is never built, so a
    round before the first promotion measures a strictly smaller term than one after it.
    This is why STEP 1d's "any burst that reaches one eval round" cannot bound the term."""
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
