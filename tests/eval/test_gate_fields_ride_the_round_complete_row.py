"""The gate's rule fields ride `eval_round_complete.gate` (CARD-EVAL-GATE-FIELDS-IN-STREAM).

Until this row `pairs_played`, `stopped`, `llr` and the gate block's wall lived in the child's
`result.json` and the in-process routed result only, so a stream reader (the dashboard, the
eval census) could not see HOW a round stopped without the spool. The producer is
`EvalPipeline._success_result` on the success route and `_broken_result` on the A-3 partial
route; the projection is `mantis.eval.rounds.gate_stream_fields`, read here through the real
methods. The planted break (a projection that drops a field) reds the row by name.
"""
from __future__ import annotations

from typing import Any

import pytest

from mantis.eval.errors import EvalBrokenReason
from mantis.eval.rounds import GATE_STREAM_FIELDS, gate_stream_fields

pytest.importorskip("torch")

_CHILD_GATE = {
    "wr_screen": 0.616, "wr_confirm": 0.616, "n_screen": 176, "n_confirm": 0, "n_pooled": 176,
    "escalated": False, "elo_ci_lower_boot": 0.05, "low_power": False, "eff_n": 176,
    "reason": "", "deploy_matched": True, "promoted": True,
    "rule": "gsprt", "llr": 3.42, "pairs_played": 88, "stopped": "accept", "wall_sec": 5473.2,
}


class _Sink:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, payload: dict) -> None:
        self.events.append(dict(payload))


class _FakePipeline:
    """`_success_result` / `_broken_result` lifted off the class; the code exercised is production."""

    def __init__(self, sink: _Sink) -> None:
        self._sink = sink
        self._round_counter = 0
        self._floor_checked_total = 0
        self._floor_skipped_total = 0

    def _emit_posture_events(self, inflight: Any, raw: Any) -> None: ...


def _complete(events: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [e for e in events if e.get("event") == "eval_round_complete"]
    assert len(rows) == 1, rows
    return rows[0]


def test_the_projection_names_the_rule_fields_and_reads_absent_ones_as_None() -> None:
    assert {"rule", "pairs_played", "stopped", "llr", "wall_sec"} <= set(GATE_STREAM_FIELDS)
    projected = gate_stream_fields(_CHILD_GATE)
    assert projected is not None
    assert (projected["rule"], projected["pairs_played"], projected["stopped"], projected["llr"],
            projected["wall_sec"]) == ("gsprt", 88, "accept", 3.42, 5473.2)
    assert gate_stream_fields(None) is None, "no gate ran: no gate mapping, never a zeroed one"
    partial = gate_stream_fields({"promoted": True})
    assert partial is not None and partial["pairs_played"] is None, (
        "an A-3 partial lacking a field reads None; a subscript here would kill the poller"
    )


def test_a_successful_round_carries_the_childs_gate_fields_on_the_stream() -> None:
    from mantis.eval.pipeline import EvalPipeline

    sink = _Sink()
    raw = {"rungs": {}, "gate": dict(_CHILD_GATE), "random": {"games": 0, "wr": None},
           "skipped_rungs": [], "worker_pid": 7}
    EvalPipeline._success_result(
        _FakePipeline(sink), {"round_id": "r000001_3000", "step": 3000, "round_idx": 1}, raw,
        wall_sec=7700.0,
    )
    row = _complete(sink.events)
    assert row["gate"] == {name: _CHILD_GATE[name] for name in GATE_STREAM_FIELDS}, row["gate"]
    assert row["wall_sec"] == 7700.0 and row["gate"]["wall_sec"] == 5473.2, (
        "the round's wall and the gate block's wall ride the same row, so the split is readable"
    )
    assert row["promoted"] is True


def test_a_round_with_no_gate_carries_gate_None() -> None:
    from mantis.eval.pipeline import EvalPipeline

    sink = _Sink()
    raw = {"rungs": {}, "gate": None, "random": {"games": 4, "wr": 0.5}, "skipped_rungs": []}
    EvalPipeline._success_result(
        _FakePipeline(sink), {"round_id": "r000001_3000", "step": 3000, "round_idx": 1}, raw,
        wall_sec=12.0,
    )
    row = _complete(sink.events)
    assert row["gate"] is None and row["promoted"] is None


def test_a_broken_round_with_an_A3_partial_carries_the_partial_gate_fields(tmp_path, monkeypatch) -> None:
    """The A-3 route: the gate phase persisted its verdict, the child died after it, and the
    row still says how the gate stopped."""
    from types import SimpleNamespace

    import mantis.eval.pipeline as pipeline_mod
    from mantis.eval.pipeline import EvalPipeline

    monkeypatch.setattr(pipeline_mod, "read_partial_gate", lambda path, *, step: dict(_CHILD_GATE))
    monkeypatch.setattr(pipeline_mod, "read_progress", lambda spec: None)
    sink = _Sink()
    spec = SimpleNamespace(result_path=str(tmp_path / "r.json"))
    EvalPipeline._broken_result(
        _FakePipeline(sink), {"round_id": "r000002_6000", "step": 6000, "spec": spec},
        reason=EvalBrokenReason.KILLED, exit_code=-9, wall_sec=9000.0, phase="worker_exit",
    )
    row = _complete(sink.events)
    assert row["games_total"] is None and row["promoted"] is True
    assert row["gate"]["stopped"] == "accept" and row["gate"]["pairs_played"] == 88


def test_a_broken_round_with_no_partial_carries_gate_None(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    import mantis.eval.pipeline as pipeline_mod
    from mantis.eval.pipeline import EvalPipeline

    monkeypatch.setattr(pipeline_mod, "read_partial_gate", lambda path, *, step: None)
    monkeypatch.setattr(pipeline_mod, "read_progress", lambda spec: None)
    sink = _Sink()
    spec = SimpleNamespace(result_path=str(tmp_path / "r.json"))
    EvalPipeline._broken_result(
        _FakePipeline(sink), {"round_id": "r000002_6000", "step": 6000, "spec": spec},
        reason=EvalBrokenReason.RESULT_MISSING, exit_code=0, wall_sec=1.0, phase="worker_exit",
    )
    assert _complete(sink.events)["gate"] is None


def test_the_PLANTED_break_a_projection_that_drops_a_field_reds_the_row(monkeypatch) -> None:
    """Mutation self-test: the producer row above is not vacuous."""
    import mantis.eval.pipeline as pipeline_mod
    from mantis.eval.pipeline import EvalPipeline

    def _mutant(gate):
        out = gate_stream_fields(gate)
        if out is not None:
            out.pop("pairs_played")
        return out

    monkeypatch.setattr(pipeline_mod, "gate_stream_fields", _mutant)
    sink = _Sink()
    raw = {"rungs": {}, "gate": dict(_CHILD_GATE), "random": {"games": 0, "wr": None},
           "skipped_rungs": []}
    EvalPipeline._success_result(
        _FakePipeline(sink), {"round_id": "r000001_3000", "step": 3000, "round_idx": 1}, raw,
        wall_sec=7700.0,
    )
    with pytest.raises(AssertionError):
        assert _complete(sink.events)["gate"] == {name: _CHILD_GATE[name] for name in GATE_STREAM_FIELDS}
