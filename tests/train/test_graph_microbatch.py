# >300 justify (R8): the rows are ONE claim — "the `train.microbatch_caps` split is the un-split
# step, bounded" — and share ONE rig: the same real `HexgBuffer`, the same replayed wire, the same
# tiny `GnnNet` trainer and the same caps arithmetic. Splitting them would fork that rig.
"""The config-typed edge/node cap plus gradient-accumulating micro-batching.

No row below is a median, a mean or any other majority statistic, and none exists in this file:
the F1 median-form statistic reads exactly 0.0 against a defect confined to <=50% of graphs, which
is precisely the shape a micro-batch split produces.

The EXACT rows are `torch.equal`, per-graph identities, exact counts or a pre-registered `rtol` on
an exact-equivalence claim, and the numerically exact ones run under
`_microbatch_harness.deterministic_algorithms()`. The BAND rows are pre-registered RELATIVE bands
on quantities that are not exact and carry three bands each.

The buffer, the wire, the partition, the slice, the collate, the losses, the optimizer, the
scheduler, the events and the filesystem are real; the ARCH and the SINK are not. Nothing here
fakes the caps, the split, the normalisation or the resolver.
"""
from __future__ import annotations

import inspect
import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

import _microbatch_harness as H
from mantis.monitor.config import MonitorConfig
from mantis.config.resolve.microbatch import (
    MicrobatchCapsSpec,
    MissingMicrobatchCapsError,
    resolve_microbatch_caps,
)
from mantis.encoding import lookup
from mantis.model import build_net
from mantis.model.dist65 import binned_value_loss
from mantis.selfplay.graph_collate import collate_graph_batch, graph_wire_from_rust
from mantis.selfplay.graph_wire_split import (
    GraphEmptyBatchError,
    GraphMicroBatchOverCap,
    plan_microbatches,
    slice_graph_wire,
    slice_targets,
)
from mantis.train.coordinator import dispatch as dispatch_mod
from mantis.train.coordinator.dispatch import run_declared_train_step
from mantis.train.coordinator.step import StepCoordinator
from mantis.train.losses import graph_loss_denominators, ragged_policy_ce
from mantis.train.trainer.core import Trainer
_TRAINLESS_GRAPH = {"identity": {"encoding": H.GRAPH_ENCODING, "representation": "graph"}}


def _offsets(counts: np.ndarray) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)


def _reference_plan(ec, nc, max_edges: int, max_nodes: int) -> list[tuple[int, int]]:
    """An INDEPENDENT transcription of the design's greedy rule, written from the design text
    rather than from the implementation: two implementations of one stated rule disagree exactly
    where the rule was misread."""
    parts: list[tuple[int, int]] = []
    start, acc_e, acc_n = 0, 0, 0
    for i in range(len(ec)):
        if (acc_e + ec[i] > max_edges or acc_n + nc[i] > max_nodes) and i > start:
            parts.append((start, i))
            start, acc_e, acc_n = i, 0, 0
        acc_e += ec[i]
        acc_n += nc[i]
    parts.append((start, len(ec)))
    return parts


def _assert_partition_properties(ec, nc, max_edges: int, max_nodes: int,
                                 parts: tuple[tuple[int, int], ...]) -> None:
    b = len(ec)
    # (ii) contiguous ordered cover of [0, B)
    assert parts, "a non-empty batch must produce at least one part"
    assert parts[0][0] == 0 and parts[-1][1] == b
    for (a0, a1), (b0, _) in zip(parts, parts[1:], strict=False):
        assert a0 < a1, f"empty or inverted part {(a0, a1)}"
        assert a1 == b0, f"parts {(a0, a1)} and {(b0, _)} are not contiguous"
    assert parts[-1][0] < parts[-1][1]
    # (iii) nothing dropped or duplicated
    covered = [g for g0, g1 in parts for g in range(g0, g1)]
    assert covered == list(range(b)), "the parts are not an ordered cover of [0, B)"
    # (i) every part within BOTH members
    for g0, g1 in parts:
        assert int(ec[g0:g1].sum()) <= max_edges, f"part {(g0, g1)} breaches max_edges"
        assert int(nc[g0:g1].sum()) <= max_nodes, f"part {(g0, g1)} breaches max_nodes"
    # (iv) M minimal for the STATED greedy rule: every part but the last is maximal, since adding
    # the next graph would breach a member. A partition that split early passes (i)-(iii).
    for g0, g1 in parts[:-1]:
        assert (int(ec[g0:g1 + 1].sum()) > max_edges
                or int(nc[g0:g1 + 1].sum()) > max_nodes), (
            f"part {(g0, g1)} is not maximal — graph {g1} fits and was split off anyway")
    assert list(parts) == _reference_plan(ec, nc, max_edges, max_nodes)


def test_of2_1_partition_properties_over_randomised_inputs() -> None:
    """The four properties on 100% of >=200 randomised inputs. A counter-example is a HALT, not a
    rate: a partition property is binary."""
    rng = np.random.default_rng(H.SEED)
    checked = 0
    for _ in range(240):
        b = int(rng.integers(1, 24))
        ec = rng.integers(1, 500, size=b).astype(np.int64)
        nc = rng.integers(1, 60, size=b).astype(np.int64)
        max_edges = int(rng.integers(int(ec.max()), int(ec.sum()) + 1))
        max_nodes = int(rng.integers(int(nc.max()), int(nc.sum()) + 1))
        parts = plan_microbatches(_offsets(ec), _offsets(nc), max_edges, max_nodes)
        _assert_partition_properties(ec, nc, max_edges, max_nodes, parts)
        checked += 1
    assert checked >= 200, f"the row requires >=200 randomised inputs; ran {checked}"


@pytest.mark.parametrize("case", ["max", "sum", "b1", "all_equal", "one_dominant"])
@pytest.mark.parametrize("member", ["edges", "nodes"])
def test_of2_1_enumerated_boundary_cases(case: str, member: str) -> None:
    """The enumerated boundaries, for BOTH members. `cap == max(counts)` is where the
    `>=`-for-`>` off-by-one lands, so it is in the bank rather than left to the randomiser."""
    if case == "b1":
        ec, nc = np.array([7], dtype=np.int64), np.array([3], dtype=np.int64)
    elif case == "all_equal":
        ec, nc = np.full(6, 10, dtype=np.int64), np.full(6, 4, dtype=np.int64)
    elif case == "one_dominant":
        ec = np.array([1, 1, 97, 1], dtype=np.int64)
        nc = np.array([1, 1, 41, 1], dtype=np.int64)
    else:
        ec = np.array([5, 9, 3, 9, 2], dtype=np.int64)
        nc = np.array([4, 2, 7, 1, 7], dtype=np.int64)
    counts = ec if member == "edges" else nc
    if case in ("max", "b1", "all_equal", "one_dominant"):
        cap = int(counts.max())
    else:
        cap = int(counts.sum())
    max_edges = cap if member == "edges" else int(ec.sum())
    max_nodes = cap if member == "nodes" else int(nc.sum())
    parts = plan_microbatches(_offsets(ec), _offsets(nc), max_edges, max_nodes)
    _assert_partition_properties(ec, nc, max_edges, max_nodes, parts)


@pytest.mark.parametrize("member", ["edges", "nodes"])
def test_of2_1_cap_one_below_the_largest_graph_raises(member: str) -> None:
    """`cap == max(counts) - 1` is out of domain: no split rescues a single graph, so it RAISES
    rather than producing a part that breaches its own bound."""
    ec = np.array([5, 9, 3], dtype=np.int64)
    nc = np.array([4, 2, 7], dtype=np.int64)
    counts = ec if member == "edges" else nc
    max_edges = int(counts.max()) - 1 if member == "edges" else int(ec.sum())
    max_nodes = int(counts.max()) - 1 if member == "nodes" else int(nc.sum())
    with pytest.raises(GraphMicroBatchOverCap):
        plan_microbatches(_offsets(ec), _offsets(nc), max_edges, max_nodes)


def test_of2_2_slice_fidelity_deterministic_mode_exact() -> None:
    """Bit-exact slice fidelity against the FULL collated batch, with every part through the real
    `collate_graph_batch(semantic="full")`. `torch.equal`, never a tolerance: this is index
    arithmetic and "close" is meaningless."""
    buf = H.ragged_graph_buffer(8)
    wire, targets = buf.sample_graph_batch(6, augment=False, recent_frac=0.0)
    # ONE read of the wire, then everything off the payload: `take()` MOVES the buffers into numpy,
    # so a second read of the pyclass raises. The payload is freely re-readable.
    payload = graph_wire_from_rust(wire)
    ec, nc = H.per_graph_counts(payload)
    b = int(payload.n_graphs)
    parts = plan_microbatches(payload.edge_offsets, payload.node_offsets,
                              int(ec.max()) * 2, int(nc.sum()) + 1)
    assert len(parts) >= 2, "the fixture must actually split or this row is vacuous"

    kw = dict(expected_version=1, trunk_size=H.GSPEC.trunk_size,
              win_length=H.GSPEC.win_length, node_feat_dim=H.GSPEC.node_feat_dim,
              edge_feat_dim=H.GSPEC.edge_feat_dim, device="cpu", semantic="full")
    with H.deterministic_algorithms():
        # The PAYLOAD, not the pyclass: the wire was consumed by the read above, and this is what
        # `_graph_step` collates too — it never hands the raw wire to the collate.
        full = collate_graph_batch(payload, target_argmax_cells=targets.target_argmax_cells,
                                   **kw)
        no = payload.node_offsets
        eo = payload.edge_offsets
        lo = payload.legal_offsets
        seen = 0
        for g0, g1 in parts:
            sub = slice_graph_wire(payload, g0, g1)
            tsl = slice_targets(targets, payload.legal_offsets, g0, g1)
            # every part passes the full 18-check contract, on its own
            part = collate_graph_batch(sub, target_argmax_cells=tsl.target_argmax_cells, **kw)
            n0, n1 = int(no[g0]), int(no[g1])
            e0, e1 = int(eo[g0]), int(eo[g1])
            l0, l1 = int(lo[g0]), int(lo[g1])
            assert torch.equal(part.x, full.x[n0:n1])
            assert torch.equal(part.edge_attr, full.edge_attr[e0:e1])
            assert torch.equal(part.edge_index, full.edge_index[:, e0:e1] - n0)
            # `legal_mask` is retired; its content here is the gather split parity below, and the
            # claim it added is asserted globally in the collate-masking authority suite.
            assert torch.equal(part.node_offsets, full.node_offsets[g0:g1 + 1] - n0)
            assert torch.equal(part.legal_offsets, full.legal_offsets[g0:g1 + 1] - l0)
            assert torch.equal(part.legal_node_gather,
                               full.legal_node_gather[l0:l1] - n0)
            # `policy_dst_slot` is retired from the batch and re-expressed on the WIRE, which is
            # where the bridge reads it and where the split's correctness is consumed.
            assert np.array_equal(np.asarray(sub.policy_dst_slot),
                                  np.asarray(payload.policy_dst_slot)[l0:l1])
            assert torch.equal(part.n_stones, full.n_stones[g0:g1])
            # `current_player` / `window_center` are likewise re-expressed on the WIRE, which is
            # where the assemble path reads them.
            assert np.array_equal(np.asarray(sub.current_player),
                                  np.asarray(payload.current_player)[g0:g1])
            assert np.array_equal(np.asarray(sub.window_center),
                                  np.asarray(payload.window_center)[g0 * 2:g1 * 2])
            assert part.n_graphs == g1 - g0
            # the four target arrays and the argmax-cell sequence (MB-20's kill surface)
            assert np.array_equal(np.asarray(tsl.policy_target),
                                  np.asarray(targets.policy_target)[l0:l1])
            assert np.array_equal(np.asarray(tsl.outcomes),
                                  np.asarray(targets.outcomes)[g0:g1])
            assert np.array_equal(np.asarray(tsl.value_valid),
                                  np.asarray(targets.value_valid)[g0:g1])
            assert np.array_equal(np.asarray(tsl.is_full_search),
                                  np.asarray(targets.is_full_search)[g0:g1])
            assert list(tsl.target_argmax_cells) == list(targets.target_argmax_cells)[g0:g1]
            assert len(tsl.target_argmax_cells) == g1 - g0
            seen += g1 - g0
    assert seen == b, "the parts did not cover every graph exactly once"


def _algebra_fixture(b: int = 12, per_graph_legal: int = 5):
    rng = np.random.default_rng(H.SEED)
    counts = np.full(b, per_graph_legal, dtype=np.int64)
    offsets = torch.tensor(_offsets(counts), dtype=torch.long)
    lg = int(counts.sum())
    logits = torch.tensor(rng.standard_normal(lg), dtype=torch.float32)
    target = torch.tensor(rng.random(lg), dtype=torch.float32)
    for g in range(b):                                  # per-graph unit mass
        seg = target[g * per_graph_legal:(g + 1) * per_graph_legal]
        target[g * per_graph_legal:(g + 1) * per_graph_legal] = seg / seg.sum()
    bin_logits = torch.tensor(rng.standard_normal((b, 65)), dtype=torch.float32)
    outcomes = torch.tensor(rng.choice([-1.0, 0.0, 1.0], size=b), dtype=torch.float32)
    return logits, target, offsets, bin_logits, outcomes, counts


#: The MIXED masks are deliberately UNBALANCED across every split boundary this row uses. Measured
#: at HEAD before the fix: with a mask alternating `[1,0,1,0,...]` the `1/M` and `B_m/B` weightings
#: agree to 7.3e-08 at k=2, because every micro-batch then carries the same mask count and the
#: wrong denominator cancels — so a balanced mask would report coverage this row does not have.
_IFS = {"ones": lambda b: torch.ones(b, dtype=torch.uint8),
        "mixed": lambda b: torch.tensor([1] * (b - 4) + [0] * 4, dtype=torch.uint8),
        "zeros": lambda b: torch.zeros(b, dtype=torch.uint8),
        "none": lambda b: None}
_VV = {"mixed": lambda b: torch.tensor([1, 1] + [0] * (b - 5) + [1, 1, 1], dtype=torch.uint8),
       "zeros": lambda b: torch.zeros(b, dtype=torch.uint8),
       "none": lambda b: None}


@pytest.mark.parametrize("k", [2, 3, 12])
@pytest.mark.parametrize("vv_name", sorted(_VV))
@pytest.mark.parametrize("ifs_name", sorted(_IFS))
def test_of2_3a_split_normalisation_equals_unsplit_deterministic_mode(
        ifs_name: str, vv_name: str, k: int) -> None:
    """Un-split vs the sum of split-and-denominator-weighted parts, fp32, NO model, across mask
    VALUE x mask PRESENCE. `rtol=1e-6 / atol=1e-8` is a floating-point associativity bound on an
    algebraically identical quantity, not a discrepancy budget."""
    b = 12
    logits, target, offsets, bin_logits, outcomes, counts = _algebra_fixture(b)
    ifs = _IFS[ifs_name](b)
    vv = _VV[vv_name](b)

    with H.deterministic_algorithms():
        unsplit = (ragged_policy_ce(logits, target, offsets, full_search_mask=ifs)
                   + binned_value_loss(bin_logits, outcomes, value_mask=vv))
        p_den, v_den = graph_loss_denominators(ifs, vv, n_graphs=b)
        assert bin_logits.shape[0] == b        # the design's own precondition, at the call
        total = torch.zeros((), dtype=torch.float32)
        bounds = list(range(0, b + 1, b // k))
        for g0, g1 in zip(bounds, bounds[1:], strict=False):
            l0, l1 = int(_offsets(counts)[g0]), int(_offsets(counts)[g1])
            sub_off = offsets[g0:g1 + 1] - offsets[g0]
            total = total + ragged_policy_ce(
                logits[l0:l1], target[l0:l1], sub_off,
                full_search_mask=None if ifs is None else ifs[g0:g1],
                denominator=p_den,
            ) + binned_value_loss(
                bin_logits[g0:g1], outcomes[g0:g1],
                value_mask=None if vv is None else vv[g0:g1],
                denominator=v_den,
            )
    assert torch.allclose(total, unsplit, rtol=1e-6, atol=1e-8), (
        f"split {float(total)!r} != un-split {float(unsplit)!r} "
        f"(is_full_search={ifs_name}, value_valid={vv_name}, k={k}) — the split is training "
        "on a different objective")


def test_of2_3a_prime_the_denominator_asymmetry_is_pinned_on_a_non_binary_mask() -> None:
    """The two denominators are DIFFERENT quantities and the implementation says so: the policy
    denominator sums mask VALUES, the value denominator COUNTS true entries. They agree only while
    the masks are strictly 0/1, so the mask used here is deliberately NOT 0/1."""
    ifs = torch.tensor([2, 0, 3], dtype=torch.float32)     # sum of VALUES = 5
    vv = torch.tensor([2, 0, 3], dtype=torch.float32)      # count of TRUE  = 2
    p_den, v_den = graph_loss_denominators(ifs, vv, n_graphs=3)
    assert p_den == 5.0, f"policy denominator must be the sum of mask VALUES; got {p_den}"
    assert v_den == 2.0, f"value denominator must be the COUNT of TRUE entries; got {v_den}"
    assert p_den != v_den, "a symmetric implementation cannot distinguish these"
    # the clamp floor, on both, and the `None` arms falling back to the graph count
    z = torch.zeros(4, dtype=torch.uint8)
    assert graph_loss_denominators(z, z, n_graphs=4) == (1.0, 1.0)
    assert graph_loss_denominators(None, None, n_graphs=7) == (7.0, 7.0)


def test_of2_3a_prime_bin_logits_row_count_is_asserted_at_the_call(tmp_path) -> None:
    """`bin_logits.shape[0] == n_graphs` is CHECKED at the call, not assumed: the
    `value_valid is None` arm sets the value denominator to the graph count while
    `binned_value_loss` reduces over `bin_logits` ROWS, so an unchecked mismatch would make
    `graph_loss_denominators` a second authority over a count it does not own."""
    trainer = H.tiny_graph_trainer(tmp_path)
    buf = H.uniform_graph_buffer(8)
    replay = H.ReplayWireBuffer(buf, 4)

    real_forward = trainer.model.forward_batch

    def _short_bin_logits(*a: Any, **kw: Any):
        policy_logits, value, bin_logits = real_forward(*a, **kw)
        return policy_logits, value, bin_logits[:-1]        # one row short

    trainer.model.forward_batch = _short_bin_logits
    caps = H.non_binding_caps(replay.wire)
    with pytest.raises(ValueError, match="bin_logits"):
        run_declared_train_step(
            trainer, replay, H.GSPEC, batch_size=4, augment=False, recency_weight=0.0,
            recent_buffer=None, sample_threads_provider=lambda: 1,
                            fast_policy_weight_provider=lambda: 0.0,
            caps_provider=lambda: MicrobatchCapsSpec(max_edges=caps[0], max_nodes=caps[1]))


#: Warm-up steps taken on BOTH arms, with identical non-binding caps, before the measured step.
#:
#: At a COLD AdamW the per-parameter difference between the arms is dominated by {~0, 2*lr}, so
#: OF2-3d's <=1.0e-4 envelope is unreachable there: every reading that fires its ABORT sits at
#: exactly 2*lr, 200% of the update the envelope asks a 10% bound on. Measured cold over 20 trials,
#: `max|dtheta|` takes three values — 1.5e-08, 6.588e-04, 2.000e-03 — landing PASS 7 / DISCLOSE 9 /
#: ABORT 4, because AdamW's first step collapses to `lr * sign(g)`. Three warm-up steps restore
#: proportionality: over 20 trials `max|dtheta|` becomes 1.371e-05 .. 3.523e-05, PASS 20/20. NO
#: ENVELOPE IS MOVED — what changed is the instrument's optimizer state.
_WARMUP_STEPS = 3


def _two_arm_step(tmp_path, m: int):
    """One M=1 step and one M=k step over the SAME wire, from the SAME weights and the same warmed
    optimizer state, so the only difference at the measured step is the micro-batch partition."""
    buf = H.uniform_graph_buffer(8)
    replay = H.ReplayWireBuffer(buf, 4)
    non_binding = H.non_binding_caps(replay.wire)
    out = []
    for caps in (non_binding, H.caps_for_exactly(replay.wire, m)):
        trainer = H.tiny_graph_trainer(tmp_path)
        for _ in range(_WARMUP_STEPS):
            run_declared_train_step(
                trainer, replay, H.GSPEC, batch_size=4, augment=False, recency_weight=0.0,
                recent_buffer=None, sample_threads_provider=lambda: 1,
                            fast_policy_weight_provider=lambda: 0.0,
                caps_provider=lambda: MicrobatchCapsSpec(max_edges=non_binding[0],
                                                         max_nodes=non_binding[1]))
        info = run_declared_train_step(
            trainer, replay, H.GSPEC, batch_size=4, augment=False, recency_weight=0.0,
            recent_buffer=None, sample_threads_provider=lambda: 1,
                            fast_policy_weight_provider=lambda: 0.0,
            caps_provider=lambda c=caps: MicrobatchCapsSpec(max_edges=c[0], max_nodes=c[1]))
        out.append((info, H.grad_vector(trainer.model), H.param_vector(trainer.model)))
    return out


@pytest.mark.parametrize("m", [2, 4])
def test_of2_3bcd_split_step_matches_the_unsplit_step(tmp_path, m: int) -> None:
    """Loss and grad-norm relative deltas, gradient cosine and post-step parameter deltas against
    the pre-registered envelopes, THREE BANDS each: the middle band is a PASS-WITH-DISCLOSURE and
    is reported, not failed. Both arms are WARMED first (see `_WARMUP_STEPS`), no envelope was
    moved, and DISCLOSED: this harness is where the split's regrouping effects are SMALLEST."""
    (one, g1, p1), (split, gk, pk) = _two_arm_step(tmp_path, m)
    d_loss = abs(split["loss"] - one["loss"]) / max(abs(one["loss"]), 1e-12)
    d_gn = abs(split["grad_norm"] - one["grad_norm"]) / max(abs(one["grad_norm"]), 1e-12)
    cos = float(torch.nn.functional.cosine_similarity(g1, gk, dim=0))
    d_theta = float((pk - p1).abs().max())
    print(f"OF2-3b/c/d M={m}: dloss={d_loss:.3e} dgrad_norm={d_gn:.3e} cos={cos:.9f} "
          f"max|dtheta|={d_theta:.3e}")
    # ABORT thresholds (PREREG_DFIX §4, unmoved)
    assert d_loss <= 1.5e-1, f"OF2-3b ABORT: |dloss|/|loss| = {d_loss:.3e} > 1.5e-1"
    assert d_gn <= 1.5e-1, f"OF2-3b ABORT: |dgrad_norm|/grad_norm = {d_gn:.3e} > 1.5e-1"
    assert cos >= 0.99, f"OF2-3c ABORT: gradient cosine = {cos:.9f} < 0.99"
    assert d_theta <= 1.0e-3, f"OF2-3d ABORT: max|dtheta| = {d_theta:.3e} > 1.0e-3"
    # PASS bands — a result in the middle band is REPORTED, which is what the row asks for
    for name, value, ok in (("OF2-3b dloss", d_loss, d_loss <= 5.0e-2),
                            ("OF2-3b dgrad_norm", d_gn, d_gn <= 5.0e-2),
                            ("OF2-3c cosine", cos, cos >= 0.999),
                            ("OF2-3d max|dtheta|", d_theta, d_theta <= 1.0e-4)):
        if not ok:
            print(f"PASS-WITH-DISCLOSURE M={m}: {name} = {value:.3e} is inside its ABORT "
                  "threshold but outside its PASS band (PREREG_DFIX §4)")


def _drive_with_spies(tmp_path, m: int, *, checkpoint_interval: int = 1):
    buf = H.uniform_graph_buffer(8)
    replay = H.ReplayWireBuffer(buf, 4)
    sink = H.SpySink()
    trainer = H.tiny_graph_trainer(tmp_path, sink=sink,
                                   checkpoint_interval=checkpoint_interval)
    opt_spy = H.OptimizerSpy(trainer.optimizer)
    sched_spy = H.SchedulerSpy(trainer.scheduler)
    caps = H.non_binding_caps(replay.wire) if m == 1 else H.caps_for_exactly(replay.wire, m)
    before = trainer.step
    clip_calls: list[int] = []
    real_clip = torch.nn.utils.clip_grad_norm_

    def _counting_clip(*a: Any, **kw: Any):
        clip_calls.append(1)
        return real_clip(*a, **kw)

    torch.nn.utils.clip_grad_norm_ = _counting_clip
    try:
        info = run_declared_train_step(
            trainer, replay, H.GSPEC, batch_size=4, augment=False, recency_weight=0.0,
            recent_buffer=None, sample_threads_provider=lambda: 1,
                            fast_policy_weight_provider=lambda: 0.0,
            caps_provider=lambda: MicrobatchCapsSpec(max_edges=caps[0], max_nodes=caps[1]))
    finally:
        torch.nn.utils.clip_grad_norm_ = real_clip
    return SimpleNamespace(info=info, sink=sink, trainer=trainer, opt=opt_spy,
                           sched=sched_spy, before=before, clips=len(clip_calls),
                           ckpts=sorted((tmp_path / "ckpt").glob("*.ckpt")))


@pytest.mark.parametrize("m", [1, 2, 4])
def test_of2_4_one_optimizer_step_and_five_keys_at_every_m(tmp_path, m: int) -> None:
    """ONE of everything per training step, at M in {1, 2, 4}, and the returned dict carries all
    five keys at every M. The key-presence half is not decoration: the coordinator's grad-norm gate
    reads `float(loss_info.get("grad_norm", 0.0))`, so a branch returning a dict without
    `grad_norm` silently feeds an armed abort a `0.0` that always passes its threshold."""
    r = _drive_with_spies(tmp_path, m)
    assert r.opt.zero_grads == 1, f"M={m}: {r.opt.zero_grads} zero_grad calls, want 1"
    assert r.opt.steps == 1, f"M={m}: {r.opt.steps} optimizer.step calls, want 1 (MB-7)"
    assert r.sched.steps == 1, f"M={m}: {r.sched.steps} scheduler.step calls, want 1"
    assert r.trainer.step - r.before == 1, f"M={m}: trainer.step moved by {r.trainer.step - r.before} (MB-8)"
    assert len(r.sink.named("trainer_step")) == 1, f"M={m}: not exactly one trainer_step event"
    assert len(r.ckpts) == 1, (
        f"M={m}: {len(r.ckpts)} .ckpt files at checkpoint_interval=1 — the R173/CS2 periodic "
        "seam must fire ONCE per training step, not once per micro-batch")
    assert len(r.sink.named("periodic_checkpoint_save")) == 1
    for key in ("loss", "policy_loss", "value_loss", "grad_norm", "lr"):
        assert key in r.info, (
            f"M={m}: the returned dict omits {key!r} — a missing 'grad_norm' silently "
            "disarms grad_norm_hard_abort through coordinator/step.py's .get(\"grad_norm\", 0.0)")
    assert set(r.info) == {"loss", "policy_loss", "value_loss", "grad_norm", "lr"}
    assert math.isfinite(r.info["grad_norm"]) or math.isnan(r.info["grad_norm"])
    assert r.sink.named("trainer_step")[0]["microbatches"] == m


@pytest.mark.parametrize("m", [1, 2, 4])
def test_of2_4_the_ema_update_fires_exactly_once_per_training_step(tmp_path, m: int) -> None:
    """The SEVENTH count — the EMA update, which the sibling row above cannot reach: the default
    fixture declares no `ema` block, so `ema_model is None` and the EMA branch never executes at
    any M. This drive enables EMA for real and counts the updates."""
    buf = H.uniform_graph_buffer(8)
    replay = H.ReplayWireBuffer(buf, 4)
    trainer = H.ema_graph_trainer(tmp_path, update_every=1)
    assert trainer.ema_model is not None, (
        "premise: this leg needs EMA actually enabled, or it re-creates the gap it closes")
    updates: list[int] = []
    real_update = trainer.ema_model.update_parameters

    def _counting_update(*a: Any, **kw: Any):
        updates.append(1)
        return real_update(*a, **kw)

    trainer.ema_model.update_parameters = _counting_update
    caps = H.non_binding_caps(replay.wire) if m == 1 else H.caps_for_exactly(replay.wire, m)
    before = trainer.step
    run_declared_train_step(
        trainer, replay, H.GSPEC, batch_size=4, augment=False, recency_weight=0.0,
        recent_buffer=None, sample_threads_provider=lambda: 1,
                            fast_policy_weight_provider=lambda: 0.0,
        caps_provider=lambda: MicrobatchCapsSpec(max_edges=caps[0], max_nodes=caps[1]))
    assert trainer.step - before == 1
    assert len(updates) == 1, (
        f"M={m}: the EMA update fired {len(updates)} times in ONE training step. It is a TAIL "
        "statement (DESIGN §3.6 lists it beside `self.step += 1`); inside the accumulation "
        "loop it would fire once per MICRO-BATCH and smooth the weights M times per step.")


@pytest.mark.parametrize("m", [1, 2, 4])
def test_of2_5_clip_grad_norm_is_called_exactly_once_for_any_m(tmp_path, m: int) -> None:
    """Clipping is NONLINEAR in the whole gradient, so it happens ONCE, after the accumulation:
    per-micro clipping would feed `grad_norm_hard_abort` the norm of a FRACTION of the gradient.
    A call COUNT cannot be absorbed by variance, which is why it is the primary assertion."""
    r = _drive_with_spies(tmp_path, m)
    assert r.clips == 1, f"M={m}: clip_grad_norm_ called {r.clips} times, want exactly 1"


def test_of2_5_grad_norm_matches_the_unsplit_steps_norm(tmp_path) -> None:
    """The reported `grad_norm` is the norm of the ACCUMULATED gradient, within the prereg's
    5.0e-2 of the UN-SPLIT step on the SAME wire and never of a self-consistent value: the
    last-micro-batch defect reads about `1 - 1/M` off that comparison and exactly 0 off a
    self-comparison."""
    (one, _, _), (split, _, _) = _two_arm_step(tmp_path, 4)
    rel = abs(split["grad_norm"] - one["grad_norm"]) / max(abs(one["grad_norm"]), 1e-12)
    assert rel <= 5.0e-2, f"|dgrad_norm|/grad_norm = {rel:.3e} > 5.0e-2 (MB-10's surface)"


@pytest.mark.parametrize("m", [1, 2, 4])
def test_of2_6_graph_trainer_step_event_carries_the_counter_and_its_caps(tmp_path,
                                                                         m: int) -> None:
    """LAW-18: the lever logs its own fire-rate in-run and the CAPS travel beside it, since a
    fire-rate of 1 is uninterpretable without the bound that produced it. `M` is computed HERE from
    the wire's own per-graph counts and never read back out of the event."""
    r = _drive_with_spies(tmp_path, m)
    ev = r.sink.named("trainer_step")[0]
    assert ev["representation"] == "graph"
    for key in ("microbatches", "edges", "nodes", "caps_max_edges", "caps_max_nodes"):
        assert key in ev, f"the graph trainer_step event omits {key!r} (LAW-18)"
        assert isinstance(ev[key], int) and not isinstance(ev[key], bool)
    replay = H.ReplayWireBuffer(H.uniform_graph_buffer(8), 4)
    ec, nc = H.per_graph_counts(replay.wire)
    assert ev["microbatches"] == m, f"counter says {ev['microbatches']}, the partition gives {m}"
    assert ev["microbatches"] >= 1
    assert ev["edges"] == int(ec.sum()) and ev["nodes"] == int(nc.sum())
    caps = H.non_binding_caps(replay.wire) if m == 1 else H.caps_for_exactly(replay.wire, m)
    assert (ev["caps_max_edges"], ev["caps_max_nodes"]) == caps


@pytest.mark.parametrize("member", ["max_edges", "max_nodes"])
def test_of2_7_a_single_over_cap_graph_raises_and_nothing_partial_happens(tmp_path,
                                                                         member: str) -> None:
    """Never a silent truncation, never a silent drop, for BOTH members. The absence assertions are
    not redundant: moving the check after `optimizer.zero_grad()` still raises and still corrupts
    the step, and only an ABSENCE assertion sees it."""
    buf = H.uniform_graph_buffer(8)
    replay = H.ReplayWireBuffer(buf, 4)
    ec, nc = H.per_graph_counts(replay.wire)
    caps = (int(ec.max()) - 1, int(nc.sum()) + 1) if member == "max_edges" else \
           (int(ec.sum()) + 1, int(nc.max()) - 1)
    sink = H.SpySink()
    trainer = H.tiny_graph_trainer(tmp_path, sink=sink, checkpoint_interval=1)
    spy = H.OptimizerSpy(trainer.optimizer)
    before = trainer.step
    with pytest.raises(GraphMicroBatchOverCap) as exc:
        run_declared_train_step(
            trainer, replay, H.GSPEC, batch_size=4, augment=False, recency_weight=0.0,
            recent_buffer=None, sample_threads_provider=lambda: 1,
                            fast_policy_weight_provider=lambda: 0.0,
            caps_provider=lambda: MicrobatchCapsSpec(max_edges=caps[0], max_nodes=caps[1]))
    message = str(exc.value)
    assert "graph 0" in message or "graph index 0" in message, message
    assert str(int(ec[0])) in message, f"the message must name the edge count: {message}"
    assert str(int(nc[0])) in message, f"the message must name the node count: {message}"
    assert member in message, f"the message must name WHICH member: {message}"
    assert str(caps[0] if member == "max_edges" else caps[1]) in message, message
    assert f"train.microbatch_caps.{member}" in message, (
        f"the message must name the config key path: {message}")
    # nothing partial happened
    assert trainer.step == before
    assert spy.steps == 0 and spy.zero_grads == 0
    assert sink.named("trainer_step") == []
    assert sorted((tmp_path / "ckpt").glob("*.ckpt")) == []


def test_of2_11_partition_boundaries_are_identical_over_100_repeats() -> None:
    """The partition is a pure function of `(ec, nc, caps)`: byte-identical boundaries over 100
    repeats, so a host-state dependence shows as a single differing tuple. Disclosed asymmetry: a
    bin-packing reorder is still DETERMINISTIC, so only the ordered-cover property can see it."""
    ec = np.array([5, 9, 3, 9, 2, 11, 4], dtype=np.int64)
    nc = np.array([4, 2, 7, 1, 7, 3, 5], dtype=np.int64)
    first = plan_microbatches(_offsets(ec), _offsets(nc), 16, 12)
    for _ in range(99):
        assert plan_microbatches(_offsets(ec), _offsets(nc), 16, 12) == first
    assert len(first) >= 2


def test_of2_11_records_whether_deterministic_mode_rejects_index_add(tmp_path, capsys) -> None:
    """RECORDED, NOT GATED. `index_add_`'s CUDA backward is an atomic scatter-add torch documents
    as nondeterministic; whether THIS build rejects it under determinism is data this phase prints
    rather than a property it asserts, since asserting it would gate on a torch detail."""
    buf = H.uniform_graph_buffer(8)
    replay = H.ReplayWireBuffer(buf, 4)
    caps = H.non_binding_caps(replay.wire)
    outcome = "accepted"
    try:
        with H.deterministic_algorithms():
            trainer = H.tiny_graph_trainer(tmp_path)
            run_declared_train_step(
                trainer, replay, H.GSPEC, batch_size=4, augment=False, recency_weight=0.0,
                recent_buffer=None, sample_threads_provider=lambda: 1,
                            fast_policy_weight_provider=lambda: 0.0,
                caps_provider=lambda: MicrobatchCapsSpec(max_edges=caps[0],
                                                         max_nodes=caps[1]))
    except RuntimeError as exc:                      # noqa: BLE001 — recorded, then re-read
        outcome = f"rejected: {exc}"
    print(f"OF2-11 determinism observation (device=cpu, torch={torch.__version__}): {outcome}")
    assert outcome  # the row's content is the RECORD; there is nothing here to gate


def _coordinator(full_config: dict, trainer: Any, buffer: Any) -> StepCoordinator:
    """A real `StepCoordinator` over the given `full_config`. The collaborators this row does not
    exercise are `None`; the ONE fake is the step config, a namespace rather than the frozen
    `StepCoordinatorConfig`, and `_run_training_step` reads only three fields off it."""
    return StepCoordinator(
        monitor_cfg=MonitorConfig(),
        trainer=trainer, buffer=buffer, pretrained_buffer=None, recent_buffer=None,
        pool=None, eval_pipeline=None, subsystems=None, anchor_state=None, shutdown=None,
        eval_model=None, bufs=None,
        config=SimpleNamespace(selfplay_stall_timeout_sec=1800.0),
        full_config=full_config)


def test_of2_15b_a_graph_route_without_the_block_raises_by_name() -> None:
    """An absent cap on the graph route is a NAMED raise, never a default: a cap that silently
    becomes absent-and-unbounded REPORTS AS PRESENT, which is the phantom-gate class and the exit
    ruled out in advance."""
    for cfg, level in ((dict(_TRAINLESS_GRAPH), "train"),
                       ({"identity": _TRAINLESS_GRAPH["identity"], "train": {}},
                        "microbatch_caps"),
                       ({"identity": _TRAINLESS_GRAPH["identity"],
                          "train": {"microbatch_caps": {"max_edges": 10}}}, "max_nodes")):
        with pytest.raises(MissingMicrobatchCapsError) as exc:
            resolve_microbatch_caps(cfg)
        assert level in str(exc.value), (
            f"the error must name the missing level {level!r}: {exc.value}")
        assert "train.microbatch_caps" in str(exc.value)


def test_of2_15b_the_graph_route_propagates_the_named_absence(tmp_path) -> None:
    """`_graph_step` does not wrap it, does not catch it and has no fallback arm, so the named
    error reaches the caller of `run_declared_train_step`."""
    coord = _coordinator(dict(_TRAINLESS_GRAPH), None, None)
    trainer = H.tiny_graph_trainer(tmp_path)
    replay = H.ReplayWireBuffer(H.uniform_graph_buffer(8), 4)
    with pytest.raises(MissingMicrobatchCapsError):
        run_declared_train_step(trainer, replay, H.GSPEC, batch_size=4, augment=False,
                                recency_weight=0.0, recent_buffer=None, sample_threads_provider=lambda: 1,
                            fast_policy_weight_provider=lambda: 0.0,
                                caps_provider=coord._microbatch_caps)


def test_of2_15_the_resolver_is_memoised_and_reads_the_config_once() -> None:
    """`_microbatch_caps` mirrors `_step_spec` in MEMOISATION, so the resolver runs once per
    coordinator however many steps a burst takes."""
    cfg = {"identity": _TRAINLESS_GRAPH["identity"],
           "train": {"microbatch_caps": {"max_edges": 11, "max_nodes": 7}}}
    coord = _coordinator(cfg, None, None)
    first = coord._microbatch_caps()
    assert first is coord._microbatch_caps()
    assert (first.max_edges, first.max_nodes) == (11, 7)


def test_of2_16_zero_graphs_plan_to_zero_parts() -> None:
    """`plan_microbatches` returns `()` at `B == 0`. A naive reading of the greedy loop appends a
    trailing part unconditionally and yields `[(0, 0)]`: one EMPTY part, which would then collate
    a zero-graph batch and produce a gradient-free loss."""
    empty = np.array([0], dtype=np.int64)
    assert plan_microbatches(empty, empty, 10, 10) == ()


def test_of2_16_a_zero_part_step_raises_before_zero_grad(tmp_path) -> None:
    """A step with no graphs cannot produce a gradient, so the only honest outcomes are a raise or
    a silent no-op, and a silent no-op would let a run report steps it never took. DECLARED
    DEFENSIVE: reachability through the coordinator's `min_buf_size` gate is UNVERIFIED."""
    sink = H.SpySink()
    trainer = H.tiny_graph_trainer(tmp_path, sink=sink, checkpoint_interval=1)
    spy = H.OptimizerSpy(trainer.optimizer)
    before = trainer.step
    with pytest.raises(GraphEmptyBatchError):
        trainer.train_step_from_graph_batch(
            parts=(), policy_denominator=1.0, value_denominator=1.0,
            total_edges=0, total_nodes=0, caps_max_edges=1, caps_max_nodes=1)
    assert trainer.step == before
    assert spy.zero_grads == 0 and spy.steps == 0
    assert sink.named("trainer_step") == []
    assert sorted((tmp_path / "ckpt").glob("*.ckpt")) == []
