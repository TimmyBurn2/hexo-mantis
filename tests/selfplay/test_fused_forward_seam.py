"""⊕ F-816-10 F7 — the R276 seam is UNTOUCHED by the split.

Written by ORACLE-WRITE **before** the feature exists. The design's central failure claim
(§4.1 property 5) is that everything the split adds sits INSIDE the existing inner `try`, so
the planner's refusal, a collate error and a real `OutOfMemoryError` all land on the SAME
`except Exception` -> `graph_inference_forward_failed` log line ->
`submit_graph_inference_failure(request_ids, ...)` -> `GraphQueue::fail_remaining` ->
`InferenceSeamFailure` at the worker. **No OOM handler, no retry, no catch-and-degrade, no new
failure path.**

That claim is only worth anything if the NEGATIVE half is pinned too, so it is: a mid-plan
failure must submit NOTHING and fail EVERY id UNIFORMLY. R276(f)'s own words are "loud and
counted, no silent catch-and-retry" — and the shape this packet is most at risk of inventing
is exactly the tempting one: catch the OOM, halve the cap, try again. Design §4.3 refuses it
on `graph_wire_split.py`'s recorded grounds (tune-to-green at runtime, R61; and it makes the
peak-allocation bound unprovable).

The defect each row is the ONLY witness to:

- **FG7-01** — a `FusedGraphOverCap` caught and turned into a skip, a truncation or a drop.
- **FG7-02** — an `except torch.cuda.OutOfMemoryError` arm added anywhere in the loop. The
  ONLY reason to add one is to retry, and a retry on a memory failure is the silent
  catch-and-retry R276(f) forbids by name.
- **FG7-03** — a per-part submit. It would leave the ids of the successful parts resolved and
  the rest failed, which is partial-success bookkeeping on a path that has none, and it would
  break the FFI's `lo[n] == probs.len()` self-consistency check on every part but the last.
- **FG7-04** — a retry loop. The model call count is the instrument: a retry runs part `k`
  twice and is invisible to every result assertion, because there are no results.
- **FG7-05** — a failure that stops naming itself. The log line is the field name the run's
  own post-mortem greps for; a reworded one is a defect nobody finds.
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
    """The seam's whole contract in one place: nothing submitted, ONE failure, EVERY id."""
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


# ═══ FG7-01 — the refusal travels the seam ═══════════════════════════════════════════════
@pytest.mark.parametrize("member", ["edges", "nodes"])
def test_fg7_01_an_over_cap_graph_dies_through_the_existing_seam(
    monkeypatch, caplog, member: str
) -> None:
    """FG7-01 — a single graph over either member is RUN-FATAL through the R276 seam: loud,
    typed, named, and every waiter released. Never a truncation, never a drop."""
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


# ═══ FG7-02/04 — a real OOM travels the same seam, mid-plan ══════════════════════════════
def test_fg7_02_an_out_of_memory_error_in_the_forward_dies_through_the_same_seam(
    monkeypatch, caplog
) -> None:
    """FG7-02 — a genuine `torch.cuda.OutOfMemoryError` raised inside the forward is NOT
    special-cased. It rides the same `except Exception` as everything else.

    This row is a REGRESSION pin as much as a new one: it holds at HEAD, and its job is to
    stay held while the split is introduced directly above it."""
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
    """FG7-04 — THE negative row. Part 1 of 3 succeeds, part 2 OOMs, part 3 never runs.

    Nothing is submitted, every id fails, and the model is called exactly twice: once for the
    part that worked and once for the part that died. A third call would be a retry; a fourth
    would be a retry loop. This is the only row that can see either, because a failing pop
    produces no results to assert against."""
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
    """FG7-04 second limb, stated as its own claim because it is the one an implementer is
    most likely to get wrong while "improving" the design: the part that SUCCEEDED before the
    failure must not be submitted on its own.

    A partial submit would resolve some waiters with real policies and fail the rest, which is
    a half-served pop the Rust side has no vocabulary for — and `fail_remaining` would then
    leave the already-set waiters untouched (`queues/graph.rs`), so the run would continue on
    a pop it half-served."""
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


# ═══ FG7-03/05 — the negatives, named ════════════════════════════════════════════════════
def _graph_loop_ast() -> ast.FunctionDef:
    src = (Path(__file__).resolve().parents[2] / "src" / "mantis" / "selfplay"
           / "inference_server.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == "_run_graph_loop":
            return node
    raise AssertionError("`_run_graph_loop` not found in inference_server.py")


def test_fg7_03_no_new_failure_path_is_introduced(monkeypatch, caplog) -> None:
    """FG7-03 — the seam SURFACE is unchanged, censused over the shipped source.

    A behavioural row cannot see a second failure route that this rig never reaches, and it
    cannot see a `try` that swallows an OOM at a nesting level the fakes never exercise. So
    this is an `ast` census — a grep cannot tell a call from a string (R93/DR-11):

    * exactly ONE `submit_graph_inference_failure` call and ONE `submit_graph_inference_results`
      call in the whole loop (the ONE submit per pop, design §4.1 property 3);
    * exactly TWO `except` handlers, the inner and the outer that HEAD already carries — a
      third is a new arm, and the only reason to add one on this path is to retry;
    * no handler naming `OutOfMemoryError` or `MemoryError` anywhere.

    This row is GREEN at authorship and that is its job: it is the pin that must STAY held
    while the split is written directly inside the `try` it censuses."""
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
    assert len(handlers) == 2, (
        f"the loop carries {len(handlers)} except handlers; HEAD carries 2 (inner + outer) "
        "and the split adds none — a third arm is the catch-and-degrade R276(f) forbids")
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
    """FG7-05 — the message that reaches the WORKER (not just the log) names the inference
    key. `submit_graph_inference_failure`'s `error_msg` is what surfaces as
    `InferenceSeamFailure` at the worker and in the run's own post-mortem; a refusal that
    names only "over cap" sends the operator to the wrong knob (R73 name-truth)."""
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
    """FG7-05 second limb — the LAW-07 clean twin: the same rig with caps that do not bind
    submits RESULTS and no failure, so the seam rows above are not passing because the harness
    fails everything."""
    payload = H.build_payload([2, 3, 4, 2])
    _server, batcher, net = H.drive_one_pop(monkeypatch, payload)
    assert batcher.failures == [], f"the clean twin failed: {batcher.failures}"
    assert len(batcher.results) == 1
    assert len(net.calls) == 1
    assert not isinstance(net.calls, torch.Tensor)  # guards a copy-paste of the wrong name
