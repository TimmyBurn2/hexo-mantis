# >300 justify (R8): ONE claim — OF2-10, that the peak allocation of one graph training step
# is bounded BY CONSTRUCTION by the two members of `train.microbatch_caps` — and every row
# below is a leg of that one claim over one shared rig (`_microbatch_harness`, the enumerated
# `_bank`, the run5 caps). The device-free legs and the GPU legs must not be separated: R179
# makes the construction the evidence and the measurement corroboration, and a reader who
# meets leg 2 on its own will read a survived burst as the bar, which is precisely what R179
# bans. The R96 correction recorded below is the same argument in the other direction — leg
# 2's premise puts the batch under both caps, i.e. M = 1, so it cannot detect MB-17 at all
# and conjunct 3's residency probe is the only detector. That is checkable only while the
# two sit in one file.
"""⊕ WP12-R dispatch 6 phase F2 — OF2-10, the STRUCTURAL bound (DESIGN_DFIX §5.2,
PREREG_DFIX §4, R179).

**This file carries the card's success criterion.** R179 bans "got further" as evidence: the
claim CARD-RUN5-GPU-OOM is closed on is that *peak allocation of one graph training step is
bounded by the two members of `train.microbatch_caps`*, and that peak allocation is bounded by
the caps must be shown by CONSTRUCTION over an adversarial bank, not by a burst that happened
not to die. Burst survival is corroboration only, and never appears here.

**Leg 1 (CI, device-free)** — the adversarial bank: parts summing to exactly `max_edges`, to
`max_edges - 1`, to `max_edges + 1`, a **high-N / low-E member whose split MUST be
node-driven**, and a single-graph batch. Both members hold on every part; the over-cap member
raises.

**The third clause is what makes leg 1 not a tautology.** An implementation that accumulates
EDGES ONLY passes every edge assertion in this file and REDs exactly one member — the
high-N/low-E one, where it produces M = 1 against a required M >= 2 (MB-19). Rev-1 of the
design would have shipped precisely that mutation as its implementation, which is why the
node term is in the bank rather than in a sentence.

**Conjunct 3 — only ONE micro-batch is ever resident (CI, device-free)** —
`test_of2_10_only_one_microbatch_is_resident_at_a_time`. Break this and the step allocates the
whole un-split batch while every count-based oracle stays green.

**Leg 2 (box / GPU only, loud skip elsewhere with grounds printed)** — the measured
`max_memory_allocated` delta over one training step at `(E, N) ~ caps`, against the sizing
pass's budget. **Leg 2b** repeats the measurement at ~2x the caps, where the caps BIND, so the
accumulation loop's own peak is measured too.

> **[R96 CORRECTION, RED_TEAM_DFIX_B F-RT-1.]** This docstring used to say leg 2 *"is the ONLY
> detector of MB-17 … laziness is a memory property and no CI oracle can see it"*. **Both
> halves were FALSE.** Leg 2's premise puts the batch UNDER both members, i.e. **M = 1**, and
> at M = 1 an eager and a lazy `parts` are the same program — it cannot detect MB-17 anywhere.
> And laziness is a **liveness** property of the Python object graph, which a device-free
> `weakref.finalize` probe reads directly. MB-17 SURVIVED the whole bank (131 passed, 0
> failed) while six artifacts said it was covered. Conjunct 3's detector is the CI row named
> above, 160 lines below this sentence.
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


# ── the adversarial bank, ENUMERATED (never randomised: each member is a named boundary) ──
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
    """OF2-10 leg 1 — the bound, by construction. Every part is within BOTH members, and the
    high-N/low-E member's split is node-driven.

    "Peak allocation is bounded by the caps" reduces to this statement plus the sizing pass's
    measured cost model `peak ~ a + b*E + c*N`: if no micro-batch ever exceeds `max_edges` or
    `max_nodes`, then no step ever allocates more than the model's value at the caps. The
    model is the sizing pass's; the partition half is this row's."""
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
    """OF2-10 leg 1, the out-of-domain half — a single graph over either member has no split
    that rescues it, so it RAISES rather than yielding a part that breaches its own bound. A
    bound that silently admits one over-bound part is not a bound."""
    ec = np.array([40, 35, 25, 50, 30], dtype=np.int64)
    nc = np.array([8, 7, 5, 10, 6], dtype=np.int64)
    max_edges = 49 if member == "max_edges" else 1000
    max_nodes = 9 if member == "max_nodes" else 1000
    with pytest.raises(GraphMicroBatchOverCap) as exc:
        plan_microbatches(_offsets(ec), _offsets(nc), max_edges, max_nodes)
    assert member in str(exc.value)


def test_of2_10_leg1_the_bank_covers_the_three_named_edge_boundaries() -> None:
    """OF2-10 leg 1 premise — the bank actually contains the members PREREG named. An
    adversarial bank that quietly lost its boundary cases is an adversarial bank in name."""
    bank = _bank()
    ec = bank["sum_equals_max_edges"][0]
    # a CONTIGUOUS PREFIX sums to exactly the cap on the `sum_equals_max_edges` member
    sums = np.cumsum(ec)
    assert 100 in set(int(s) for s in sums) or any(
        int(ec[i:j].sum()) == 100 for i in range(len(ec)) for j in range(i + 1, len(ec) + 1)), (
        "no window of the bank sums to exactly max_edges — the `>`/`>=` boundary is untested")
    assert set(bank) >= {"sum_equals_max_edges", "sum_one_below_max_edges",
                         "sum_one_above_max_edges", "node_driven", "single_graph"}


# ── conjunct 3: ONLY ONE MICRO-BATCH IS EVER RESIDENT (device-free) ──────────────────────
def _max_concurrently_live_parts(trainer, replay, caps, batch_size: int) -> int:
    """Drive one real training step and return the MAXIMUM number of `GraphStepInputs` alive
    at the same moment.

    Laziness is not a memory property that only an allocator can see — it is a LIVENESS
    property of the Python object graph, and CPython makes it directly observable. Each
    `parts` callable is wrapped and the finalizer is registered **on the returned object's `x`
    TENSOR, not on the `GraphStepInputs` wrapper**; the counter goes up on materialisation and
    down when that tensor is collected. Under refcounting the drop is deterministic at the
    `del` in the accumulation loop, so the reading is exact, not statistical.

    **WHY THE TENSOR AND NOT THE WRAPPER — measured, and it is the difference between an
    oracle and a decoration (RED-TEAM ANALOGUE-A).** A mutation that collates eagerly and then
    hands out a FRESH wrapper per call keeps every micro-batch's tensors resident while each
    wrapper dies immediately. Watching the wrapper reads **1** and passes; watching `obj.x`
    reads **M**. Measured on this rig, both forms, both programs:

        SHIPPED     finalize-on-WRAPPER -> 1     finalize-on-obj.x -> 1
        ANALOGUE-A  finalize-on-WRAPPER -> 1     finalize-on-obj.x -> 4   <- the bytes

    Counting wrappers would have been the same class of defect this row exists to close, one
    level in: an oracle that cannot see what it claims to measure. The bytes live on the
    tensors, so the tensors are what is watched.

    Nothing about the step is faked: the real dispatcher builds the real callables, and the
    wrapper only observes what it is handed on the way past."""
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

            # ON THE TENSOR, NOT THE WRAPPER — see the docstring's measurement. A fresh
            # wrapper around already-collated tensors dies immediately and would read 1.
            weakref.finalize(obj.x, _released)
            return obj

        return _materialise

    def _wrapped(*, parts, **kw):
        return real_step(parts=tuple(_observe(m) for m in parts), **kw)

    trainer.train_step_from_graph_batch = _wrapped
    run_declared_train_step(
        trainer, replay, H.GSPEC, batch_size=batch_size, augment=False, recency_weight=0.0,
        recent_buffer=None, caps_provider=lambda: caps, sample_threads_provider=lambda: 1)
    return peak


@pytest.mark.parametrize("m", [2, 4])
def test_of2_10_only_one_microbatch_is_resident_at_a_time(tmp_path, m: int) -> None:
    """**CONJUNCT 3 OF THE BOUND, and until this row it had NO detector anywhere.**

    The bound this card rests on is a composition of three statements: (1) every micro-batch is
    within both members — OF2-10 leg 1, device-free; (2) peak allocation at the caps is under
    the sizing budget — leg 2, GPU; and (3) **only one micro-batch is ever resident**. Break
    (3) and the step allocates the WHOLE un-split batch — the 2.49-2.56x overshoot the card
    exists to close — while `microbatches`, `edges`, `nodes` and every cadence count stay
    exactly correct. `trainer/core.py` says so in its own docstring: a `Sequence` of
    already-collated batches *"would hold every micro-batch resident at once, defeating the cap
    while passing every count-based oracle."*

    **WHY THIS ROW EXISTS NOW.** MB-17 (`parts = [make() for make in parts]`) was registered
    with leg 2 named as its only detector, in four separate artifacts. That was FALSE, and
    measurably so: leg 2's premise asserts the batch sits UNDER both members, which by
    `plan_microbatches`' own contract means **M = 1** — and at M = 1 an eager and a lazy
    `parts` are the SAME PROGRAM. RED-TEAM ran MB-17 against the whole bank and it SURVIVED
    (131 passed, 0 failed), together with its unbanked sibling (the `del` removed from the
    accumulation loop). The stated grounds for having no CI oracle — *"laziness is a memory
    property; no CI oracle can see it"* — were refuted by a fourteen-line device-free probe.

    This row kills both, on CPU, in under a second: shipped code reads **1**, the `del`-removal
    sibling reads **2**, MB-17 reads **M**."""
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


# ── leg 2: the box ────────────────────────────────────────────────────────────────────────
#: **THE SIZING PASS'S BUDGET, in GiB, with its derivation.** RE-FITTED at the F-816-10/-12
#: box sitting (R281(d), R282(b), R283) against terms measured THAT sitting. Every term below
#: carries **the sha it was measured at and the regime it was measured in** — R281(d)(ii)'s
#: dated-premise convention, adopted because the superseded derivation (kept below) read as a
#: current measurement for as long as it did precisely by carrying neither tag.
#:
#: **A term with no sha and no regime tag is UNMEASURED, not inherited.**
#:
#:     15.479 GiB usable card   [measured 24ae93e, POST-Design-A, torch.cuda.mem_get_info]
#:   -  0.261      CUDA context [measured 24ae93e, POST-Design-A, total-free on an idle card]
#:   -  2.244      SELF-PLAY/INFERENCE RESIDENT SHARE at the minted `inference.fused_graph_caps`
#:                 [measured 24ae93e, POST-Design-A, fused batch up to
#:                  n_workers x leaf_batch_size = 160 — this SUPERSEDES the 1.287 GiB figure
#:                  measured at 528eb37, which was taken when `submit_graph_and_wait` put
#:                  exactly ONE graph in flight per worker]
#:   -  0.881      eval child on `worker_device: cuda`
#:                 [measured 24ae93e, POST-Design-A, peak over one live eval round]
#:   = 12.094      device left to the trainer
#:   / 1.2278      measured fragmentation ratio
#:                 [measured 24ae93e, POST-Design-A, DEFAULT allocator posture]
#:   = 9.850 GiB   allocatable
#:
#: **THE RE-FIT CONFIRMED THIS CONSTANT RATHER THAN MOVING IT** — 9.431 sat 0.419 GiB under the
#: 9.850 the measured terms admitted — and **R326(b) HAS NOW MOVED IT ANYWAY, on a different
#: ground: the permission was never DERIVED from what the trainer needs.** Both derivations above
#: compute what the CARD can spare; neither asks what the step DRAWS. Measured across two
#: sittings on two shas, the step draws **7 992 252 928 B = 7.443 GiB**, byte-identical. The
#: cap-permitted ceiling was `9.431 x frag = 11.777 GiB`, **77.4 % of the whole card for a step
#: that uses 7.443** — a 4.33 GiB gap between permitted and drawn, and it is that gap, not any
#: measurement, that refused the partition at sittings 3 and 4.
#:
#:     7.443 GiB   MEASURED trainer peak, this oracle, reproduced to the byte
#:                 [RECAL_SITTING3_RECORD:354 and RECAL_SITTING4_RECORD:275]
#:   x 1.129       stated allowance, +12.9 %
#:   = 8.40 GiB    the permission, ARMED by the operator's RECAL-SITTING-5 forwarding
#:
#: **THE CEILING CONVENTION IS UNCHANGED:** the partition term is still `budget x frag`, the
#: PERMISSION and not the draw. Only the permission shrank. The closing boundary — the largest
#: budget at which conjunct-2 headroom still equals M — was re-derived independently at
#: pre-flight as **8.4669** against the sitting's 8.4666, and 8.40 clears it by 0.067.
#:
#: **WHAT THIS COSTS THIS ROW, so the next reader is not surprised by it:** the margin below is
#: budget-relative, so 7.443 under 8.40 reads **11.4 %** where it read 21.1 % under 9.431. That
#: is under `_REQUIRED_MARGIN` and the row therefore reports PASS-WITH-DISCLOSURE. It is
#: arithmetic, not a regression: 12.9 % over the NEED is 11.4 % of the BUDGET.
#:
#: **On the old 0.85 margin, which is no longer applied here.** It was sized for two NAMED
#: unknowns. One — the eval child — is now a measured, subtracted term. The other — a
#: fragmentation swing — is what the partition's remaining headroom covers, and it is the
#: named residual risk of this sizing: at the minted pair the WHOLE partition declares
#: 14.965 GiB against a 15.479 GiB card (+0.514 GiB), so a fragmentation ratio above ~1.30
#: would close it. The live pre-mint validation run peaked at 12.05 GiB of card, 3.87 GiB
#: below the wall, because the trainer does not reach this ceiling in practice (its measured
#: peak that sitting was 7.447 GiB at 84.6% of `max_edges`).
#:
#: THE SUPERSEDED DERIVATION, kept verbatim because R281(d) is a ruling about how this comment
#: is written and deleting the evidence would delete the lesson (`wp/WP12R/MEASUREMENT_SIZING.md`,
#: RTX 5080, torch 2.11.0+cu128, two live boots at `528eb37`, ratified R193):
#:
#:     15.479 - 0.330 (CUDA context) - 1.287 (self-play cache) = 13.862 reserved
#:     / 1.2493 = 11.096 allocatable * 0.85 margin = 9.431 GiB
#:
#: The 1.287 GiB line is the F-816-12 defect itself: one member of a partition moved by a large
#: factor and its partner kept the value fitted before the move, while the oracle below — which
#: measures the trainer ALONE — stayed green through it.
#:
#: Transcribed ONCE, here, with the arithmetic beside it so a reader can re-derive it rather
#: than trust it. If the operator re-sizes, this constant and the EDGE-CAP row move together.
#:
#: **AND A SECOND CONSTANT DOES NOT MOVE WITH IT — READ `test_graph_microbatch_authority.py`
#: BEFORE ASSUMING IT SHOULD.** That file carries its own `_SIZING_BUDGET_BYTES` for the SIZING
#: FRONTIER, and R326(b) deliberately left it where it was. The two are the same nominal
#: quantity in two different denominations, and the disagreement is recorded there, not resolved.
_SIZING_BUDGET_GIB = 8.40
_SIZING_BUDGET_BYTES = int(_SIZING_BUDGET_GIB * 1024 ** 3)

#: PREREG_DFIX §4, OF2-10 leg 2: PASS is "<= the budget with >= 15% margin".
_REQUIRED_MARGIN = 0.15

#: How close to the minted caps the fixture must get before a peak measurement means anything.
#: MEASURED on this fixture: nodes reach **99.9%** of `max_nodes` and edges **84.6%** of
#: `max_edges` — the fixture's mean in-degree is 3316/148 = 22.4 against the cap pair's ratio
#: of 26.5, so the NODE member is what binds and edges land below their own cap. The floors
#: are set under those measurements with a little slack, NOT at them: their job is to catch a
#: fixture that has stopped reaching the regime at all. The first shipped version of this leg
#: measured peak on a 32-graph toy — roughly five orders below the sized caps — and asserted
#: `peak > 0`, which is green whatever the allocation is; these floors are what make that
#: unconstructible.
_MIN_EDGE_FRACTION = 0.75
_MIN_NODE_FRACTION = 0.95


def _run5_caps() -> MicrobatchCapsSpec:
    """run5's OWN minted caps, through the real loader and the real resolver — DERIVED, never
    transcribed, so the operator's mint act moves this leg with it and pins nothing."""
    return resolve_microbatch_caps(load_config(_CONFIGS / "run5.yaml").model_dump())


def _cap_regime_batch(caps: MicrobatchCapsSpec):
    """A wire sized to sit just under BOTH minted members — the `(E, N) ~ caps` the row names.

    The graph count is derived from the caps and the fixture's own per-graph counts, so it
    tracks a re-mint instead of being a magic number."""
    probe = H.uniform_graph_buffer(8)
    ec, nc = H.per_graph_counts(probe.sample_graph_batch(4, augment=False, recent_frac=0.0)[0])
    per_e, per_n = int(ec[0]), int(nc[0])
    n_graphs = min(caps.max_edges // per_e, caps.max_nodes // per_n)
    buf = H.uniform_graph_buffer(n_graphs + 8)
    return H.ReplayWireBuffer(buf, n_graphs), n_graphs


def test_of2_10_leg2_fixture_reaches_the_minted_cap_regime() -> None:
    """OF2-10 leg 2's PREMISE, checked device-free so CI carries it.

    A peak-allocation measurement taken far below the caps bounds nothing about the caps. This
    leg asserts the fixture actually reaches the regime BEFORE the box spends a GPU on it — so
    if a fixture change silently shrinks `(E, N)`, that is caught here, in CI, rather than
    surfacing as a comfortable green from the measured leg nobody can re-run."""
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
    """OF2-10 leg 2 — the MEASURED half, and PREREG §0.1's clause 2 for closing
    CARD-RUN5-GPU-OOM.

    Three bands, all PREREG_DFIX §4's: PASS is `<= budget` with `>= 15%` margin;
    PASS-WITH-DISCLOSURE is within budget under 15% margin, and prints the number and the
    margin; over budget is an ABORT — HALT and re-size (one re-size is a finding and is
    disclosed AS a re-size; a second is a design failure).

    WHAT IS REAL HERE: run5's OWN minted caps (through the real loader and resolver), run5's
    OWN arch (`hidden=128, num_layers=4`, resolved from `configs/run5.yaml`), the real
    dispatcher, the real partition, the real collate and a `(E, N)` that sits just under both
    minted members. The measurement is the DELTA across the step
    (`reset_peak_memory_stats` -> `max_memory_allocated`), not an absolute reading.

    DISCLOSED, so a green is not read for more than it is: (a) the buffer is SYNTHETIC and its
    mean in-degree is 22.4 against run5's measured 26.8, so bytes-per-edge here is not
    guaranteed to equal production's 2203.57; (b) the node member binds first on this fixture,
    so `E` reaches ~85% of `max_edges` while `N` reaches ~100% of `max_nodes` — the leg bounds
    peak AT THE (E, N) IT REPORTS, and the fraction of each cap is printed beside the result;
    (c) the eval child on `eval.worker_device: cuda` is co-resident in production and appears
    in NO number here — it is what the 0.85 margin in the budget is partly for."""
    caps = _run5_caps()
    replay, n_graphs = _cap_regime_batch(caps)
    ec, nc = H.per_graph_counts(replay.wire)
    e_total, n_total = int(ec.sum()), int(nc.sum())

    run5_cfg = load_config(_CONFIGS / "run5.yaml").model_dump()
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
        recent_buffer=None, caps_provider=lambda: caps, sample_threads_provider=lambda: 1)
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


#: Leg 2b's ceiling on `peak(2x) / peak(1x)`. The DEFECT SIGNATURE it exists to catch is peak
#: scaling with the INPUT rather than with the cap — a leak reads ~2.0. NOT calibrated on the
#: box (no CUDA here), so it is set to catch that signature with room, not to be a tight
#: envelope; the first box run's job is to REPORT the real ratio, and if it lands near 1.0 this
#: constant should tighten to a measured value. Stated so nobody reads 1.25 as a measurement.
_ACCUM_PEAK_RATIO_CEILING = 1.25


def test_of2_10_leg2b_premise_the_doubled_batch_actually_binds_the_caps() -> None:
    """Leg 2b's PREMISE, device-free, so CI carries it.

    Moved out of the GPU-skipped body under the dispatcher's pre-commit review: leg 2b's
    `len(plan) >= 2` assert used to sit INSIDE the `skipif`, which is the one place this
    phase's own F-RT-1 lesson had not been applied — a premise that only runs where the test
    runs is a premise nobody checks. If the doubled batch ever stops splitting, leg 2b silently
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
    """OF2-10 leg 2b — **a step at ~2x the caps costs no more than a step at 1x**, which is
    the true statement of the bound. Added under RED-TEAM F-RT-1; the comparison itself added
    under the dispatcher's pre-commit review.

    RENAMED AND TIGHTENED. The first version asserted only `peak <= budget` while its name
    promised a comparison against the 1x step — and at the predicted 9.318 GiB against the
    9.431 GiB budget OF THAT TIME that left **~14% (1.161 GiB) of slack a leak could hide in**.
    (R326(b) has since re-derived the budget to 8.40; the figure above is left at the value the
    defect was measured against, because re-computing a historical slack at a later budget would
    describe a comparison nobody ran.) It now measures
    BOTH steps in the same process and compares them directly, so the name and the assertion
    say the same thing. A leak that scales peak with the INPUT reads ~2.0 and cannot hide in
    the budget's headroom.

    Disclosures inherited from leg 2 (synthetic in-degree 22.4 vs run5's 26.8; the eval child
    absent from every number) plus leg 2b's own: **the budget's fragmentation divisor (1.2493)
    was measured on the UN-SPLIT program**, so the post-fix allocation pattern is UNVERIFIED
    and this leg is the first instrument that would show it."""
    caps = _run5_caps()
    single, n_single = _cap_regime_batch(caps)
    doubled = H.ReplayWireBuffer(H.uniform_graph_buffer(2 * n_single + 8), 2 * n_single)
    run5_cfg = load_config(_CONFIGS / "run5.yaml").model_dump()
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
            recency_weight=0.0, recent_buffer=None, caps_provider=lambda: caps, sample_threads_provider=lambda: 1)
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
