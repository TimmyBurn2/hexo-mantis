# >300 justify (R8): ONE claim — that the peak allocation of one graph training step is bounded
# BY CONSTRUCTION by the two members of `train.microbatch_caps` — and every row below is a leg
# of that one claim over one shared rig. The device-free legs and the GPU legs must not be
# separated: the construction is the evidence and the measurement only corroboration, and the
# GPU leg's own premise (M = 1) makes it unable to detect the laziness conjunct 3 catches.
"""OF2-10 — the STRUCTURAL bound on one graph training step's peak allocation.

This file carries the card's success criterion, and "got further" is banned as evidence: the
claim is that peak allocation of one graph training step is bounded by the two members of
`train.microbatch_caps`, shown by CONSTRUCTION over an adversarial bank. Burst survival is
corroboration only and never appears here.

Leg 1 (CI, device-free) is the adversarial bank: parts summing to exactly `max_edges`, to
`max_edges - 1`, to `max_edges + 1`, a HIGH-N / LOW-E member whose split MUST be node-driven,
and a single-graph batch — an implementation accumulating EDGES ONLY passes every edge
assertion here and REDs exactly that member. Conjunct 3 (CI) is that only ONE micro-batch is
ever resident. Leg 2 (GPU only, loud skip elsewhere) is the measured `max_memory_allocated`
delta at `(E, N) ~ caps`, with leg 2b repeating it at ~2x the caps, where the caps BIND.
"""
from __future__ import annotations

import weakref
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import _microbatch_harness as H
from mantis.config.loader import load_config
from mantis.config.resolve.microbatch import MicrobatchCapsSpec, resolve_microbatch_caps
from mantis.model import arch_from_spec_and_config, build_net
from mantis.selfplay.graph_wire_split import GraphMicroBatchOverCap, plan_microbatches
from mantis.train.coordinator.dispatch import run_declared_train_step
from mantis.train.trainer.core import Trainer

_CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def _offsets(counts: np.ndarray) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)


def _bank() -> dict[str, tuple[np.ndarray, np.ndarray, int, int, dict[str, Any]]]:
    """`name -> (ec, nc, max_edges, max_nodes, expectations)`."""
    ec = np.array([40, 35, 25, 50, 30], dtype=np.int64)      # sum 180
    nc = np.array([8, 7, 5, 10, 6], dtype=np.int64)          # sum 36
    high_n_low_e = (np.array([2, 2, 2, 2, 2, 2], dtype=np.int64),
                    np.array([90, 90, 90, 90, 90, 90], dtype=np.int64))
    return {
        # a part sums to EXACTLY max_edges: the `>` / `>=` boundary MB-1 lands on
        "sum_equals_max_edges": (ec, nc, 100, 1000, {"min_parts": 2}),
        # one below: the same partition boundary, from the other side
        "sum_one_below_max_edges": (ec, nc, 99, 1000, {"min_parts": 2}),
        # one above
        "sum_one_above_max_edges": (ec, nc, 101, 1000, {"min_parts": 2}),
        # HIGH-N / LOW-E: the split must be NODE-driven — every part's edge sum is far under
        # the edge member, so an edges-only accumulator returns M = 1 here (MB-19)
        "node_driven": (*high_n_low_e, 10_000, 200, {"min_parts": 2, "node_driven": True}),
        # a single-graph batch: exactly one part, and no split is possible
        "single_graph": (np.array([7], dtype=np.int64), np.array([3], dtype=np.int64),
                         7, 3, {"exact_parts": 1}),
    }


@pytest.mark.parametrize("case", sorted(_bank()))
def test_of2_10_leg1_both_members_hold_on_every_part(case: str) -> None:
    """OF2-10 leg 1 — every part is within BOTH members and the high-N/low-E split is
    node-driven, which with the sizing model `peak ~ a + b*E + c*N` bounds every step."""
    ec, nc, max_edges, max_nodes, want = _bank()[case]
    parts = plan_microbatches(_offsets(ec), _offsets(nc), max_edges, max_nodes)
    for g0, g1 in parts:
        assert int(ec[g0:g1].sum()) <= max_edges, f"{case}: part {(g0, g1)} breaches max_edges"
        assert int(nc[g0:g1].sum()) <= max_nodes, f"{case}: part {(g0, g1)} breaches max_nodes"
    covered = [g for g0, g1 in parts for g in range(g0, g1)]
    assert covered == list(range(len(ec))), f"{case}: the parts are not an ordered cover"
    if "exact_parts" in want:
        assert len(parts) == want["exact_parts"]
    if "min_parts" in want:
        assert len(parts) >= want["min_parts"], (
            f"{case}: {len(parts)} parts, want >= {want['min_parts']}")
    if want.get("node_driven"):
        assert len(parts) >= 2, (
            f"{case}: M = {len(parts)} — an edges-only accumulator produces exactly this "
            "(MB-19); the node term is what must have driven the split")
        for g0, g1 in parts:
            assert int(ec[g0:g1].sum()) < 0.5 * max_edges, (
                f"{case}: part {(g0, g1)} has edge sum {int(ec[g0:g1].sum())}, not far under "
                f"max_edges={max_edges} — the split was not node-driven and this member has "
                "stopped testing what it exists to test")


@pytest.mark.parametrize("member", ["max_edges", "max_nodes"])
def test_of2_10_leg1_the_over_cap_member_raises(member: str) -> None:
    """OF2-10 leg 1, out of domain — a single graph over either member RAISES rather than
    yielding a part that breaches its own bound."""
    ec = np.array([40, 35, 25, 50, 30], dtype=np.int64)
    nc = np.array([8, 7, 5, 10, 6], dtype=np.int64)
    max_edges = 49 if member == "max_edges" else 1000
    max_nodes = 9 if member == "max_nodes" else 1000
    with pytest.raises(GraphMicroBatchOverCap) as exc:
        plan_microbatches(_offsets(ec), _offsets(nc), max_edges, max_nodes)
    assert member in str(exc.value)


def test_of2_10_leg1_the_bank_covers_the_three_named_edge_boundaries() -> None:
    """OF2-10 leg 1's premise — the bank still contains the members prereg named."""
    bank = _bank()
    ec = bank["sum_equals_max_edges"][0]
    # a CONTIGUOUS PREFIX sums to exactly the cap on the `sum_equals_max_edges` member
    sums = np.cumsum(ec)
    assert 100 in set(int(s) for s in sums) or any(
        int(ec[i:j].sum()) == 100 for i in range(len(ec)) for j in range(i + 1, len(ec) + 1)), (
        "no window of the bank sums to exactly max_edges — the `>`/`>=` boundary is untested")
    assert set(bank) >= {"sum_equals_max_edges", "sum_one_below_max_edges",
                         "sum_one_above_max_edges", "node_driven", "single_graph"}


def _max_concurrently_live_parts(trainer, replay, caps, batch_size: int) -> int:
    """Drive one real training step and return the MAXIMUM number of `GraphStepInputs` alive at
    the same moment.

    Laziness is a LIVENESS property of the Python object graph: the finalizer is registered on
    the returned object's `x` TENSOR, not on the `GraphStepInputs` wrapper, and under
    refcounting the drop is deterministic at the `del` in the accumulation loop. Watching the
    wrapper cannot see the defect, because a mutation that collates eagerly and hands out a
    fresh wrapper per call keeps every micro-batch's tensors resident. Measured on this rig:

        SHIPPED     finalize-on-WRAPPER -> 1     finalize-on-obj.x -> 1
        ANALOGUE-A  finalize-on-WRAPPER -> 1     finalize-on-obj.x -> 4   <- the bytes
    """
    live = 0
    peak = 0
    real_step = trainer.train_step_from_graph_batch

    def _observe(make):
        def _materialise():
            nonlocal live, peak
            obj = make()
            live += 1
            peak = max(peak, live)

            def _released(_ref=None) -> None:
                nonlocal live
                live -= 1

            # ON THE TENSOR, NOT THE WRAPPER — see the docstring's measurement.
            weakref.finalize(obj.x, _released)
            return obj

        return _materialise

    def _wrapped(*, parts, **kw):
        return real_step(parts=tuple(_observe(m) for m in parts), **kw)

    trainer.train_step_from_graph_batch = _wrapped
    run_declared_train_step(
        trainer, replay, H.GSPEC, batch_size=batch_size, augment=False, recency_weight=0.0,
        recent_buffer=None, caps_provider=lambda: caps, sample_threads_provider=lambda: 1,
                            fast_policy_weight_provider=lambda: 0.0)
    return peak


@pytest.mark.parametrize("m", [2, 4])
def test_of2_10_only_one_microbatch_is_resident_at_a_time(tmp_path, m: int) -> None:
    """CONJUNCT 3 OF THE BOUND — only one micro-batch is ever resident.

    Break it and the step allocates the WHOLE un-split batch — the 2.49-2.56x overshoot the card
    exists to close — while `microbatches`, `edges`, `nodes` and every cadence count stay exactly
    correct. Shipped code reads 1; removing the `del` reads 2; collating eagerly reads M.

    The eager mutation SURVIVED the whole bank (131 passed, 0 failed) while four artifacts named
    the GPU leg as its only detector — that leg's premise puts the batch under both members,
    i.e. M = 1, where an eager and a lazy `parts` are the same program.
    """
    buf = H.uniform_graph_buffer(8)
    replay = H.ReplayWireBuffer(buf, 4)
    e_cap, n_cap = H.caps_for_exactly(replay.wire, m)
    caps = MicrobatchCapsSpec(max_edges=e_cap, max_nodes=n_cap)
    trainer = H.tiny_graph_trainer(tmp_path)
    peak = _max_concurrently_live_parts(trainer, replay, caps, batch_size=4)
    plan = plan_microbatches(np.asarray(replay.wire.edge_offsets),
                             np.asarray(replay.wire.node_offsets), e_cap, n_cap)
    assert len(plan) == m, (
        f"premise: this leg needs M = {m} micro-batches to have anything to observe; the "
        f"partition produced {len(plan)}. At M = 1 eager and lazy are the same program and "
        "this row would pass vacuously — which is exactly how MB-17 survived the bank")
    assert peak == 1, (
        f"M={m}: {peak} micro-batches' node tensors were resident SIMULTANEOUSLY, want 1. "
        "`parts` is a "
        "Sequence of ZERO-ARG CALLABLES so that only one micro-batch's tensors are ever live; "
        "materialising them eagerly (MB-17) holds all M at once and defeats the cap while "
        "every count-based oracle stays green. A reading of M is eager materialisation; a "
        "reading of 2 is the previous part outliving the next `make()` — the `del` in the "
        "accumulation loop is what makes it 1.")


#: THE SIZING PASS'S BUDGET, in GiB, with its derivation. Every term carries the sha it was
#: measured at and the regime it was measured in; an untagged term is UNMEASURED, not inherited.
#: All four below were measured at 24ae93e, POST-Design-A:
#:
#:     15.479 GiB usable card (torch.cuda.mem_get_info)
#:   -  0.261      CUDA context (total-free on an idle card)
#:   -  2.244      self-play/inference resident share at the minted `fused_graph_caps` (160)
#:   -  0.881      eval child on `worker_device: cuda` (peak over one live eval round)
#:   = 12.094      device left to the trainer
#:   / 1.2278      measured fragmentation ratio (DEFAULT allocator posture)
#:   = 9.850 GiB   allocatable
#:
#: The ARMED constant is derived from what the step DRAWS, not from what the card can spare.
#: Sitting 9 sampled the r8/GnnArchV2 step delta 60 times on the real ring — min 7.405569,
#: p50 7.907190, p90 8.186055, p95 8.358100, MAX 8.638102 — and the allowance is over the MAX,
#: because a partition is a bound and an OOM at the 5th-percentile step ends the run:
#:
#:     8.638102 (measured MAX, 60 steps, batch 256, real r8 ring) x 1.03 = 8.897245
#:     armed FLOORED to 8.8972 — 47 KiB below the derivation, conservative on both sides
#:
#: The partition term is `budget x frag`, the PERMISSION and not the draw; it still closes with
#: 0.1536 GiB of cap to spare. NAMED RESIDUAL: the partition declares 14.965 GiB against a
#: 15.479 GiB card, so a fragmentation ratio above ~1.30 would close it.
#: `test_graph_microbatch_authority.py` carries its OWN `_SIZING_BUDGET_BYTES` for the SIZING
#: FRONTIER, ruled to STAY UNMOVED — do not move one assuming the other follows.
_SIZING_BUDGET_GIB = 8.8972
_SIZING_BUDGET_BYTES = int(_SIZING_BUDGET_GIB * 1024 ** 3)

#: PREREG_DFIX §4, OF2-10 leg 2: PASS is "<= the budget with >= 15% margin".
_REQUIRED_MARGIN = 0.15

#: How close to the minted caps the fixture must get before a peak measurement means anything.
#: MEASURED here: nodes reach 99.9% of `max_nodes`, edges 84.6% of `max_edges` (mean in-degree
#: 22.4 against the cap pair's 26.5, so the NODE member binds). The floors sit under those with
#: slack — their job is to catch a fixture that has stopped reaching the regime at all.
_MIN_EDGE_FRACTION = 0.75
_MIN_NODE_FRACTION = 0.95


def _run5_caps() -> MicrobatchCapsSpec:
    """run5's OWN minted caps, through the real loader and resolver — DERIVED, never transcribed."""
    return resolve_microbatch_caps(load_config(_CONFIGS / "run6.yaml").model_dump())


def _cap_regime_batch(caps: MicrobatchCapsSpec):
    """A wire sized to sit just under BOTH minted members, with the graph count derived from the
    caps and the fixture's own per-graph counts so it tracks a re-mint."""
    probe = H.uniform_graph_buffer(8)
    ec, nc = H.per_graph_counts(probe.sample_graph_batch(4, augment=False, recent_frac=0.0)[0])
    per_e, per_n = int(ec[0]), int(nc[0])
    n_graphs = min(caps.max_edges // per_e, caps.max_nodes // per_n)
    buf = H.uniform_graph_buffer(n_graphs + 8)
    return H.ReplayWireBuffer(buf, n_graphs), n_graphs


def test_of2_10_leg2_fixture_reaches_the_minted_cap_regime() -> None:
    """OF2-10 leg 2's PREMISE, device-free so CI carries it: a peak measured far below the caps
    bounds nothing, so a fixture change that shrinks `(E, N)` is caught here."""
    caps = _run5_caps()
    replay, n_graphs = _cap_regime_batch(caps)
    ec, nc = H.per_graph_counts(replay.wire)
    e_total, n_total = int(ec.sum()), int(nc.sum())
    e_frac, n_frac = e_total / caps.max_edges, n_total / caps.max_nodes
    print(f"OF2-10 leg 2 regime: {n_graphs} graphs, E={e_total} ({e_frac:.1%} of "
          f"max_edges={caps.max_edges}), N={n_total} ({n_frac:.1%} of "
          f"max_nodes={caps.max_nodes})")
    assert e_total <= caps.max_edges and n_total <= caps.max_nodes, (
        "the regime batch must sit UNDER both members — otherwise the split fires and the "
        "measurement is of a micro-batch, not of the cap")
    assert e_frac >= _MIN_EDGE_FRACTION, (
        f"the fixture reaches only {e_frac:.1%} of max_edges; below {_MIN_EDGE_FRACTION:.0%} a "
        "peak measurement bounds a toy, not the cap")
    assert n_frac >= _MIN_NODE_FRACTION, (
        f"the fixture reaches only {n_frac:.1%} of max_nodes; below {_MIN_NODE_FRACTION:.0%} a "
        "peak measurement bounds a toy, not the cap")


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="OF2-10 leg 2 measures the max_memory_allocated DELTA over one real "
                           "graph training step at (E, N) ~ the MINTED caps, with run5's own "
                           f"arch, against the sizing pass's {_SIZING_BUDGET_GIB} GiB budget; "
                           "it needs the CUDA "
                           "device the sizing pass measured. LOUD SKIP: the MEASURED half does "
                           "NOT run here and this phase claims no peak-allocation measurement "
                           "from CI. What DOES run device-free: leg 1's structural bound, the "
                           "fixture-reach premise, and the LIVENESS row above. CORRECTED (R96, "
                           "RED-TEAM F-RT-1): this reason used to claim leg 2 was 'the ONLY "
                           "detector of MB-17'. That was FALSE — leg 2 runs at M = 1, where an "
                           "eager and a lazy `parts` are the same program, so it cannot detect "
                           "MB-17 at all. MB-17's detector is "
                           "`test_of2_10_only_one_microbatch_is_resident_at_a_time`, which is "
                           "device-free and runs in CI.")
def test_of2_10_leg2_peak_allocation_is_under_the_sizing_budget(tmp_path) -> None:
    """OF2-10 leg 2 — the MEASURED half, and the clause for closing CARD-RUN5-GPU-OOM.

    Three bands: PASS is `<= budget` with `>= 15%` margin; PASS-WITH-DISCLOSURE is within budget
    under that margin and prints the number; over budget is an ABORT — halt and re-size. Real
    here: run5's own minted caps and arch through the real loader and resolver, the real
    dispatcher, partition and collate, and a `(E, N)` just under both members; the measurement
    is the DELTA across the step. DISCLOSED: the buffer is SYNTHETIC at mean in-degree 22.4
    against run5's measured 26.8, the node member binds first so `E` reaches only ~85% of
    `max_edges`, and the co-resident eval child appears in NO number here.
    """
    caps = _run5_caps()
    replay, n_graphs = _cap_regime_batch(caps)
    ec, nc = H.per_graph_counts(replay.wire)
    e_total, n_total = int(ec.sum()), int(nc.sum())

    run5_cfg = load_config(_CONFIGS / "run6.yaml").model_dump()
    arch = arch_from_spec_and_config(H.GSPEC, run5_cfg)
    torch.manual_seed(H.SEED)
    trainer = Trainer(build_net(arch), H.graph_config(), arch=arch,
                      checkpoint_dir=tmp_path / "ckpt", device=torch.device("cuda"),
                      train_hparams=H.graph_hparams())

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = int(torch.cuda.max_memory_allocated())
    run_declared_train_step(
        trainer, replay, H.GSPEC, batch_size=n_graphs, augment=False, recency_weight=0.0,
        recent_buffer=None, caps_provider=lambda: caps, sample_threads_provider=lambda: 1,
                            fast_policy_weight_provider=lambda: 0.0)
    torch.cuda.synchronize()
    peak_delta = int(torch.cuda.max_memory_allocated()) - before
    margin = (_SIZING_BUDGET_BYTES - peak_delta) / _SIZING_BUDGET_BYTES
    print(f"OF2-10 leg 2: peak_delta={peak_delta} B ({peak_delta / 1024 ** 3:.3f} GiB) "
          f"vs budget {_SIZING_BUDGET_BYTES} B ({_SIZING_BUDGET_GIB} GiB), "
          f"margin={margin:.1%}; measured at E={e_total} "
          f"({e_total / caps.max_edges:.1%} of max_edges), N={n_total} "
          f"({n_total / caps.max_nodes:.1%} of max_nodes), arch={arch}")
    assert peak_delta <= _SIZING_BUDGET_BYTES, (
        f"ABORT — peak {peak_delta / 1024 ** 3:.3f} GiB EXCEEDS the sizing budget "
        f"{_SIZING_BUDGET_GIB} GiB at E={e_total}, N={n_total}. The bound is not where the "
        "sizing pass says it is: HALT and re-size (PREREG_DFIX §4, OF2-10 leg 2).")
    if margin < _REQUIRED_MARGIN:
        print(f"PASS-WITH-DISCLOSURE: margin {margin:.1%} is inside the budget but below the "
              f"{_REQUIRED_MARGIN:.0%} the row asks for — reported, per PREREG_DFIX §4.")


#: Leg 2b's ceiling on `peak(2x) / peak(1x)`. It catches peak scaling with the INPUT rather
#: than the cap, which reads ~2.0; NOT calibrated on the box, so 1.25 is not a measurement.
_ACCUM_PEAK_RATIO_CEILING = 1.25


def test_of2_10_leg2b_premise_the_doubled_batch_actually_binds_the_caps() -> None:
    """Leg 2b's PREMISE, device-free so CI carries it: a premise that only runs where the test
    runs is a premise nobody checks. If the doubled batch stops splitting, leg 2b silently
    becomes a second copy of leg 2 and the accumulation loop goes unmeasured again."""
    caps = _run5_caps()
    _single, n_single = _cap_regime_batch(caps)
    replay = H.ReplayWireBuffer(H.uniform_graph_buffer(2 * n_single + 8), 2 * n_single)
    ec, nc = H.per_graph_counts(replay.wire)
    plan = plan_microbatches(np.asarray(replay.wire.edge_offsets),
                             np.asarray(replay.wire.node_offsets),
                             caps.max_edges, caps.max_nodes)
    print(f"leg 2b premise: {2 * n_single} graphs, E={int(ec.sum())} "
          f"({int(ec.sum()) / caps.max_edges:.1f}x), N={int(nc.sum())} "
          f"({int(nc.sum()) / caps.max_nodes:.1f}x) -> M={len(plan)}")
    assert len(plan) >= 2, (
        f"the doubled batch produced M={len(plan)}: leg 2b would measure an UNSPLIT step and "
        "the accumulation loop's peak would go unmeasured, which is exactly the hole leg 2b "
        "was added to close")
    for g0, g1 in plan:
        assert int(ec[g0:g1].sum()) <= caps.max_edges
        assert int(nc[g0:g1].sum()) <= caps.max_nodes


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="OF2-10 leg 2b measures peak allocation at 1x AND at ~2x the caps "
                           "and compares them; it needs the same CUDA device. LOUD SKIP: the "
                           "device-free liveness row and leg 2b's premise row both run here, "
                           "so a skip is a missing NUMBER, not a missing detector.")
def test_of2_10_leg2b_doubling_the_input_does_not_move_the_peak(tmp_path) -> None:
    """OF2-10 leg 2b — a step at ~2x the caps costs no more than a step at 1x, which is the true
    statement of the bound.

    The first version asserted only `peak <= budget` while its name promised a comparison, which
    left ~14% (1.161 GiB) of slack a leak could hide in. It now measures BOTH steps in one
    process, so a leak that scales peak with the INPUT reads ~2.0. Leg 2b's own disclosure: the
    budget's fragmentation divisor was measured on the UN-SPLIT program.
    """
    caps = _run5_caps()
    single, n_single = _cap_regime_batch(caps)
    doubled = H.ReplayWireBuffer(H.uniform_graph_buffer(2 * n_single + 8), 2 * n_single)
    run5_cfg = load_config(_CONFIGS / "run6.yaml").model_dump()
    arch = arch_from_spec_and_config(H.GSPEC, run5_cfg)

    def _peak(replay, batch_size: int, tag: str) -> int:
        torch.manual_seed(H.SEED)
        trainer = Trainer(build_net(arch), H.graph_config(), arch=arch,
                          checkpoint_dir=tmp_path / f"ckpt_{tag}", device=torch.device("cuda"),
                          train_hparams=H.graph_hparams())
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = int(torch.cuda.max_memory_allocated())
        run_declared_train_step(
            trainer, replay, H.GSPEC, batch_size=batch_size, augment=False,
            recency_weight=0.0, recent_buffer=None, caps_provider=lambda: caps, sample_threads_provider=lambda: 1,
                            fast_policy_weight_provider=lambda: 0.0)
        torch.cuda.synchronize()
        peak = int(torch.cuda.max_memory_allocated()) - before
        del trainer
        torch.cuda.empty_cache()
        return peak

    peak_1x = _peak(single, n_single, "1x")
    peak_2x = _peak(doubled, 2 * n_single, "2x")
    ratio = peak_2x / max(peak_1x, 1)
    margin = (_SIZING_BUDGET_BYTES - peak_2x) / _SIZING_BUDGET_BYTES
    print(f"OF2-10 leg 2b: peak_1x={peak_1x / 1024 ** 3:.3f} GiB, "
          f"peak_2x={peak_2x / 1024 ** 3:.3f} GiB, ratio={ratio:.3f}, "
          f"budget margin at 2x={margin:.1%}")
    assert ratio <= _ACCUM_PEAK_RATIO_CEILING, (
        f"ABORT — doubling the input moved the peak by {ratio:.2f}x (ceiling "
        f"{_ACCUM_PEAK_RATIO_CEILING}). The cap is meant to bound the step regardless of how "
        "big the sampled batch is; a ratio near 2.0 means the micro-batches are resident "
        "together and the split is not bounding anything.")
    assert peak_2x <= _SIZING_BUDGET_BYTES, (
        f"ABORT — the 2x step peaked at {peak_2x / 1024 ** 3:.3f} GiB, over the "
        f"{_SIZING_BUDGET_GIB} GiB budget: HALT and re-size.")
    if margin < _REQUIRED_MARGIN:
        print(f"PASS-WITH-DISCLOSURE: margin {margin:.1%} below the "
              f"{_REQUIRED_MARGIN:.0%} the row asks for — reported, per PREREG_DFIX §4.")
