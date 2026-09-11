"""Prove the fused-forward split leaves the graph inference failure seam untouched.

Everything the split adds sits INSIDE the existing inner `try`, so a planner refusal, a collate
error and a real OOM all land on the same handler, log line and failure submission. The negative
half is pinned too: a mid-plan failure submits NOTHING and fails EVERY id uniformly, with no OOM
handler and no retry — catching the OOM to halve the cap makes the allocation bound unprovable.
"""
from __future__ import annotations

import ast
import logging
from pathlib import Path

import pytest
import torch

import _fused_graph_harness as H

_ALL_IDS = [1, 2, 3, 4]


def _assert_uniform_seam_failure(batcher, *, n_ids: int) -> None:
    """Assert the seam's whole contract: nothing submitted, one failure, every id."""
    assert batcher.results == [], (
        f"a failing pop submitted results anyway: {len(batcher.results)} submit(s). The "
        "submit happens only after ALL parts succeed, so a mid-plan failure means nothing "
        "was submitted and there is no partial-success bookkeeping to get wrong.")
    assert len(batcher.failures) == 1, (
        f"expected exactly ONE failure submission covering the whole pop; got "
        f"{len(batcher.failures)} — a per-part failure path is a NEW failure path")
    ids, msg = batcher.failures[0]
    assert ids == list(range(1, n_ids + 1)), (
        f"the failure did not cover every waiter: {ids}. A waiter left unresolved blocks its "
        "worker until the stall watchdog fires (LAW-16), which reports the wrong subsystem.")
    assert msg.startswith("Graph inference failed: "), (
        f"the failure message left the EXISTING wrapper: {msg!r}")


@pytest.mark.parametrize("member", ["edges", "nodes"])
def test_fg7_01_an_over_cap_graph_dies_through_the_existing_seam(
    monkeypatch, caplog, member: str
) -> None:
    """Prove an over-cap graph is run-fatal through the existing seam, with every waiter
    released — never a truncation, never a drop."""
    payload = H.build_payload([2, 3, 40, 2])
    ec, nc = H.per_graph_counts(payload)
    cap_e = int(ec.max()) - 1 if member == "edges" else 10 ** 9
    cap_n = int(nc.max()) - 1 if member == "nodes" else 10 ** 9
    with caplog.at_level(logging.ERROR):
        _server, batcher, net = H.drive_one_pop(
            monkeypatch, payload, max_fused_edges=cap_e, max_fused_nodes=cap_n)

    _assert_uniform_seam_failure(batcher, n_ids=len(_ALL_IDS))
    assert "graph_inference_forward_failed" in caplog.text, (
        "the refusal did not reach the EXISTING log line the post-mortem greps for")
    assert "FusedGraphOverCap" in caplog.text, (
        "the log line does not name the refusal TYPE, so the failure is indistinguishable "
        "from a collate error or a NaN in the run record")
    assert net.calls == [], (
        "the planner refuses BEFORE any device allocation; a forward that ran anyway means "
        "the plan was computed after the collate, which is the post-collate design §4.1(1) "
        "rejects")
    assert batcher.closed == 1, "the loop must still close its batcher on exit"


def test_fg7_02_an_out_of_memory_error_in_the_forward_dies_through_the_same_seam(
    monkeypatch, caplog
) -> None:
    """Prove a genuine OOM in the forward is not special-cased and rides the same handler."""
    payload = H.build_payload([2, 3, 4, 2])
    net = H.SentinelGraphNet(oom_on_call=1)
    with caplog.at_level(logging.ERROR):
        _server, batcher, _ = H.drive_one_pop(monkeypatch, payload, net=net)

    _assert_uniform_seam_failure(batcher, n_ids=len(_ALL_IDS))
    assert "OutOfMemoryError" in caplog.text, (
        "the OOM lost its type on the way to the log line")
    assert len(net.calls) == 1, (
        f"the forward was attempted {len(net.calls)} times — an OOM retry is the silent "
        "catch-and-retry R276(f) forbids by name")


def test_fg7_04_a_mid_plan_failure_submits_nothing_and_fails_every_id(
    monkeypatch, caplog
) -> None:
    """Prove a mid-plan failure submits nothing and fails every id. The model call count is the
    instrument: a failing pop produces no results for anything else to assert against."""
    payload = H.build_payload([2, 3, 4, 2])
    ec, _nc = H.per_graph_counts(payload)
    cap_e = int(ec[0]) + int(ec[1])       # graphs 0+1, then 2, then 3
    net = H.SentinelGraphNet(oom_on_call=2)
    with caplog.at_level(logging.ERROR):
        _server, batcher, _ = H.drive_one_pop(
            monkeypatch, payload, max_fused_edges=cap_e, max_fused_nodes=10 ** 9, net=net)

    _assert_uniform_seam_failure(batcher, n_ids=len(_ALL_IDS))
    assert len(net.calls) == 2, (
        f"the plan ran {len(net.calls)} forwards; expected exactly 2 — part 1 succeeded, "
        "part 2 died, and NOTHING after it may run (no continue-past-the-failure, no retry)")
    assert "graph_inference_forward_failed" in caplog.text


def test_fg7_04_the_successful_parts_output_is_discarded_not_submitted(
    monkeypatch, caplog
) -> None:
    """Prove the successful part's output is discarded rather than submitted on its own: a partial
    submit is a half-served pop `fail_remaining` cannot undo."""
    payload = H.build_payload([2, 3, 4, 2])
    ec, _nc = H.per_graph_counts(payload)
    with caplog.at_level(logging.ERROR):
        _server, batcher, _ = H.drive_one_pop(
            payload=payload, monkeypatch=monkeypatch,
            max_fused_edges=int(ec[0]) + int(ec[1]), max_fused_nodes=10 ** 9,
            net=H.SentinelGraphNet(oom_on_call=2))
    assert batcher.results == [], (
        "the first part's probs were submitted before the plan finished — the ONE submit "
        "happens after every part has run (design §4.1 property 3)")


#: The loop's failure surface since A4-4 spans the loop and its two pipeline stages plus the
#: one failure path they share; the census walks their union, not the loop body alone.
_LOOP_FUNCTIONS = ("_run_graph_loop", "_launch_pop", "_retire", "_retire_or_fail", "_fail_pop")


def _graph_loop_ast() -> ast.Module:
    src = (Path(__file__).resolve().parents[2] / "src" / "mantis" / "selfplay"
           / "inference_server.py").read_text(encoding="utf-8")
    found = {node.name: node for node in ast.walk(ast.parse(src))
             if isinstance(node, ast.FunctionDef) and node.name in _LOOP_FUNCTIONS}
    missing = [name for name in _LOOP_FUNCTIONS if name not in found]
    assert not missing, f"{missing} not found in inference_server.py"
    return ast.Module(body=[found[name] for name in _LOOP_FUNCTIONS], type_ignores=[])


def test_fg7_03_no_new_failure_path_is_introduced(monkeypatch, caplog) -> None:
    """Census the shipped loop's failure surface: one submit of each kind, exactly two broad
    handlers, every narrow handler ending in a bare `raise`, and no OOM-specific handler.

    Counting handlers flatly would be a proxy — red on a harmless observe-and-re-raise arm, green
    if a broad arm became a degrade.
    """
    fn = _graph_loop_ast()
    calls = [n.func.attr for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert calls.count("submit_graph_inference_failure") == 1, (
        f"the loop has {calls.count('submit_graph_inference_failure')} failure-submit call "
        "sites; a second one is a SECOND failure path over one seam")
    assert calls.count("submit_graph_inference_results") == 1, (
        f"the loop has {calls.count('submit_graph_inference_results')} result-submit call "
        "sites; the ONE submit per pop is what makes the FFI's self-consistency checks hold "
        "against the UNSLICED `legal_offsets`")
    handlers = [n for n in ast.walk(fn) if isinstance(n, ast.ExceptHandler)]
    # Observing arms — those that only look and re-raise — are unlimited; what is capped is the
    # broad arms that route a failure over the seam.
    def _is_broad(h: ast.ExceptHandler) -> bool:
        return h.type is None or ast.unparse(h.type) in {"Exception", "BaseException"}

    broad = [h for h in handlers if _is_broad(h)]
    assert len(broad) == 3, (
        f"the loop carries {len(broad)} BROAD except handlers; HEAD carries 3 — the launch arm "
        "and the retire arm, each routing its OWN pop's failure through `_fail_pop`, and the "
        "outer loop guard. A fourth broad catch on this path is the catch-and-degrade R276(f) "
        "forbids")
    for handler in [h for h in handlers if not _is_broad(h)]:
        reraises = any(
            isinstance(n, ast.Raise) and n.exc is None for n in ast.walk(handler)
        )
        assert reraises, (
            f"the narrow handler for `{ast.unparse(handler.type) if handler.type else ''}` does "
            "not end in a bare `raise`. An arm that observes a failure and lets execution "
            "continue is the catch-and-degrade R276(f) forbids, whatever it logs on the way")
    for handler in handlers:
        named = ast.unparse(handler.type) if handler.type is not None else ""
        assert "OutOfMemoryError" not in named and "MemoryError" not in named, (
            f"an OOM-specific handler was added ({named}). The only reason to catch an OOM "
            "specifically is to retry, and a retry on a memory failure is the silent "
            "catch-and-retry R276(f) forbids by name")

    payload = H.build_payload([2, 3, 40, 2])
    ec, _nc = H.per_graph_counts(payload)
    with caplog.at_level(logging.ERROR):
        _server, batcher, _ = H.drive_one_pop(
            monkeypatch, payload, max_fused_edges=int(ec.max()) - 1,
            max_fused_nodes=10 ** 9)
    assert len(batcher.failures) == 1, (
        "the census's behavioural twin: the one failure path is LIVE, not merely unique")


def test_fg7_05_the_failure_names_the_config_key_the_operator_must_change(
    monkeypatch, caplog
) -> None:
    """Prove the worker-facing failure message, not just the log line, names the inference key
    the operator must change."""
    payload = H.build_payload([2, 3, 40, 2])
    ec, _nc = H.per_graph_counts(payload)
    with caplog.at_level(logging.ERROR):
        _server, batcher, _ = H.drive_one_pop(
            monkeypatch, payload, max_fused_edges=int(ec.max()) - 1,
            max_fused_nodes=10 ** 9)
    assert len(batcher.failures) == 1, (
        "the over-cap graph did not reach the seam at all — the caps were never enforced")
    _ids, msg = batcher.failures[0]
    assert "inference.fused_graph_caps.max_fused_edges" in msg, (
        f"the worker-facing failure does not name the key to re-mint: {msg!r}")
    assert "train.microbatch_caps" not in msg, (
        "the worker-facing failure names the TRAIN key — false provenance (D-2/R73)")


def test_fg7_05_a_healthy_pop_is_the_clean_twin(monkeypatch) -> None:
    """Prove the clean twin: the same rig with non-binding caps submits results and no failure,
    so the rows above are not passing because the harness fails everything."""
    payload = H.build_payload([2, 3, 4, 2])
    _server, batcher, net = H.drive_one_pop(monkeypatch, payload)
    assert batcher.failures == [], f"the clean twin failed: {batcher.failures}"
    assert len(batcher.results) == 1
    assert len(net.calls) == 1
    assert not isinstance(net.calls, torch.Tensor)  # guards a copy-paste of the wrong name
