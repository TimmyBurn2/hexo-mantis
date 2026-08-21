# >300 justify (R8). NO LINE COUNT is stated, per G-DFIX-4 and R192(e)'s derive-or-delete:
# R8 asks for a one-line justification, not a tally, and a number that must be re-edited
# whenever a row is added will eventually be wrong and then be read as evidence. (This
# file's first version stated one and it was already false at submission.)
# The fourteen rows are ONE claim — "the
# `train.microbatch_caps` split is the un-split step, bounded" — and they share ONE rig: the
# same real `HexgBuffer`, the same replayed wire, the same tiny `GnnNet` trainer and the same
# caps arithmetic. Splitting them would fork that rig (the fork-and-drift failure
# `tests/train/test_periodic_checkpoint.py:5-8` argues against) and would separate the
# partition properties from the normalisation identity they are the precondition of. The
# executable content is a minority: the rest is the per-row "what defect is this the ONLY
# witness to" rationale LAW-07 asks of each row, and the reachability note R166 asks of each
# drive.
"""⊕ WP12-R dispatch 6 phase F2 — the config-typed edge/node cap + gradient-accumulating
micro-batching (DESIGN_DFIX §3, PREREG_DFIX §4, CARD-RUN5-GPU-OOM).

**R191 BINDS EVERY ROW HERE.** F1's median-form statistic reads **exactly 0.0** against a
2.6e+3 defect confined to <=50% of graphs, and a micro-batch split produces precisely that
minority-subset shape: a partition bug, a mis-weighted denominator or a dropped part corrupts
the graphs in SOME parts and leaves the rest bit-identical. So no row below is a median, a
mean, or any other majority statistic — grep-verified, none exists in this file.

Two kinds of row, named separately because they are not the same kind of evidence. **The exact
rows** — OF2-1, OF2-2, OF2-3a/a', OF2-4, OF2-5's count, OF2-6, OF2-7, OF2-11, OF2-13, OF2-15,
OF2-16 — are `torch.equal`, per-graph identities, exact counts or a pre-registered `rtol` on an
EXACT-equivalence claim, and the ones whose exactness is numeric run under
`_microbatch_harness.deterministic_algorithms()`, whose scope is the test and whose name says
so. **The band rows** — OF2-3b/c/d and OF2-5's second limb — are pre-registered RELATIVE bands
on quantities that are not exact (the split regroups floating-point sums), they do NOT run
under determinism, and they carry three bands each. R191 is satisfied by both kinds; it is not
satisfied by calling the second kind the first.

The defect each row is the ONLY witness to:

- **OF2-1** — a partition that silently drops, duplicates or reorders graphs. Properties over
  >=200 randomised `(ec, nc, caps)` inputs plus the enumerated boundary cases, because a
  partition property is binary and a spot check is not a property.
- **OF2-2** — corrupted rebasing or offset arithmetic in the slice, and an unsliced
  `target_argmax_cells` (which makes `collate_graph_batch` raise on EVERY part,
  `graph_collate.py:586-590`). The only row that reads the slice against the full batch.
- **OF2-3a** — the naive-averaging trap: a split that trains on a DIFFERENT objective. The
  configuration cross is mask VALUE x mask PRESENCE because `1/M` weighting is CORRECT when
  every graph is full-search and every row is value-valid, and wrong otherwise — a
  single-configuration oracle ships MB-4.
- **OF2-3a'** — a SYMMETRIC `graph_loss_denominators`, which passes every other row here
  while encoding a latent divergence: HEAD's two denominators are different quantities (a sum
  of mask VALUES at `losses.py:101-102`, a count of TRUE at `dist65.py:58-62`) and they agree
  only while the masks are strictly 0/1.
- **OF2-3b/c/d** — the algebra right and the plumbing wrong.
- **OF2-4** — M optimizer steps per training step, the R173/CS2 periodic-save seam firing M
  times, and **a missing `grad_norm` key silently disarming `grad_norm_hard_abort` through
  `coordinator/step.py`'s grad-norm gate reading `loss_info.get("grad_norm", 0.0)`**. Swept over M in {1, 2, 4} so the
  key-presence guarantee is tested on the M=1 path production takes when the caps do not bind.
- **OF2-5** — an armed gate's input rescaled by M. Clipping is NONLINEAR in the whole
  gradient, so per-micro clipping feeds `grad_norm_hard_abort` the norm of a FRACTION.
- **OF2-6** — a cap that fires with no in-run evidence (LAW-18, R164's rider).
- **OF2-7** — a silent drop or truncation of an out-of-domain graph (R114's clause), and a
  half-executed step.
- **OF2-11** — a partition depending on host state.
- **OF2-13** — the `GRAPH_FORBIDDEN_NONZERO_WEIGHTS` ban, which has **NO behavioural producer
  at HEAD**: `tests/config/test_train_entropy.py:73-81` only asserts the string is absent from
  the schema module, a duplication guard (R4/LAW-07).
- **OF2-15** — the route-scoped resolution. (a) is a REGRESSION pin against the fix's own
  earlier defect — an eager `caps=self._microbatch_caps()` at `coordinator/step.py:930` would
  resolve `full_config["train"]` on BOTH representations and break the four frozen grid
  coordinators that carry no `train` key; (b) is the ⊕ half, a graph route with no
  `train.microbatch_caps` failing BY NAME instead of defaulting to uncapped.
- **OF2-16** — DEFENSIVE. A zero-graph step silently returning a no-op result dict, i.e. a run
  reporting steps it never took. **Reachability through the coordinator's `min_buf_size` gate
  is UNVERIFIED** (DESIGN_DFIX §3.3 names the settling measurement); this row is a producer
  test for a GUARD, not for a live route, and may not be cited as evidence that `B == 0`
  occurs.

**What is real and what is not.** Real: the buffer, the wire, the partition, the slice,
`collate_graph_batch` (`semantic="full"`, all 18 checks, on every part), the losses, the
optimizer, the scheduler, the events, the filesystem. Fake: the ARCH (tiny `GnnNet`) and the
SINK (a spy). **Nothing here fakes the caps, the split, the normalisation or the resolver** —
and OF2-15 does not even fake the config: it uses the literal train-less `full_config` five
frozen files construct.
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
from mantis._engine import ReplayBuffer
from mantis.config.resolve.microbatch import (
    MicrobatchCapsSpec,
    MissingMicrobatchCapsError,
    resolve_microbatch_caps,
)
from mantis.encoding import lookup
from mantis.model import CnnArch, build_net
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

GRID_ENCODING = "v6_live2_ls"
_DSPEC = lookup(GRID_ENCODING)

#: The exact train-less `full_config` five FROZEN files construct (`test_clean_stop_save.py:254`,
#: `test_eval_result_routing.py:198`, `test_terminal_eval_rc.py:330`,
#: `test_target_counter_events.py:242`). Copied as a LITERAL, not imported: it is the shape
#: OF2-15 is about, and a shared constant could be edited to make the row pass.
_TRAINLESS_GRID = {"identity": {"encoding": GRID_ENCODING, "representation": "grid"}}
_TRAINLESS_GRAPH = {"identity": {"encoding": H.GRAPH_ENCODING, "representation": "graph"}}


# ═══ OF2-1 — partition properties ════════════════════════════════════════════════════════
def _offsets(counts: np.ndarray) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)


def _reference_plan(ec, nc, max_edges: int, max_nodes: int) -> list[tuple[int, int]]:
    """An INDEPENDENT transcription of DESIGN_DFIX §3.3's greedy rule, written from the design
    text rather than from the implementation. Two implementations of one stated rule disagree
    exactly where the rule was misread."""
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
    # (iv) M minimal for the STATED greedy rule: every part but the last is maximal — adding
    # the next graph would breach a member. This is the property that makes the count minimal
    # rather than merely legal; a partition that split early passes (i)-(iii) and fails here.
    for g0, g1 in parts[:-1]:
        assert (int(ec[g0:g1 + 1].sum()) > max_edges
                or int(nc[g0:g1 + 1].sum()) > max_nodes), (
            f"part {(g0, g1)} is not maximal — graph {g1} fits and was split off anyway")
    assert list(parts) == _reference_plan(ec, nc, max_edges, max_nodes)


def test_of2_1_partition_properties_over_randomised_inputs() -> None:
    """OF2-1 — the four properties on 100% of >=200 randomised inputs. A counter-example is a
    HALT, not a rate: a partition property is binary."""
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
    """OF2-1 — the enumerated boundaries, for BOTH members. `cap == max(counts)` is where
    MB-1's `>=`-for-`>` off-by-one lands, so it is a member of the bank rather than a value
    the randomiser might happen to draw."""
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
    """OF2-1/OF2-7 boundary — `cap == max(counts) - 1` is out of domain: no split rescues a
    single graph, so it RAISES rather than producing a part that breaches its own bound."""
    ec = np.array([5, 9, 3], dtype=np.int64)
    nc = np.array([4, 2, 7], dtype=np.int64)
    counts = ec if member == "edges" else nc
    max_edges = int(counts.max()) - 1 if member == "edges" else int(ec.sum())
    max_nodes = int(counts.max()) - 1 if member == "nodes" else int(nc.sum())
    with pytest.raises(GraphMicroBatchOverCap):
        plan_microbatches(_offsets(ec), _offsets(nc), max_edges, max_nodes)


# ═══ OF2-2 — slice fidelity ══════════════════════════════════════════════════════════════
def test_of2_2_slice_fidelity_deterministic_mode_exact() -> None:
    """OF2-2 — bit-exact slice fidelity against the FULL collated batch, and every part
    through the real `collate_graph_batch(semantic="full")`.

    `torch.equal`, never a tolerance: this is index arithmetic and "close" is meaningless.
    Reachability (R166): the wire, the payload conversion, the slice and the collate are the
    production statements `dispatch.py::_graph_step` executes."""
    buf = H.ragged_graph_buffer(8)
    wire, targets = buf.sample_graph_batch(6, augment=False, recent_frac=0.0)
    payload = graph_wire_from_rust(wire)
    ec, nc = H.per_graph_counts(wire)
    b = int(wire.n_graphs)
    parts = plan_microbatches(payload.edge_offsets, payload.node_offsets,
                              int(ec.max()) * 2, int(nc.sum()) + 1)
    assert len(parts) >= 2, "the fixture must actually split or this row is vacuous"

    kw = dict(expected_version=1, trunk_size=H.GSPEC.trunk_size,
              win_length=H.GSPEC.win_length, node_feat_dim=H.GSPEC.node_feat_dim,
              edge_feat_dim=H.GSPEC.edge_feat_dim, device="cpu", semantic="full")
    with H.deterministic_algorithms():
        full = collate_graph_batch(wire, target_argmax_cells=targets.target_argmax_cells, **kw)
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
            # (was: `part.legal_mask == full.legal_mask[n0:n1]`.) `legal_mask` is retired by
            # RQ-16 / R297(c). Its CONTENT here is the row two below — the gather split parity —
            # and the one thing it added beyond that, namely that no legal node outside a graph's
            # CSR slice lands inside that graph's node range, is now asserted globally and
            # directly by `test_the_gather_and_the_CSR_agree_per_graph_segment`
            # (tests/selfplay/test_graph_collate_masking_authority.py). Rebuilding both masks
            # from gathers this block already asserts equal would be a tautology, not a check.
            assert torch.equal(part.node_offsets, full.node_offsets[g0:g1 + 1] - n0)
            assert torch.equal(part.legal_offsets, full.legal_offsets[g0:g1 + 1] - l0)
            assert torch.equal(part.legal_node_gather,
                               full.legal_node_gather[l0:l1] - n0)
            assert torch.equal(part.policy_dst_slot, full.policy_dst_slot[l0:l1])
            assert torch.equal(part.n_stones, full.n_stones[g0:g1])
            assert torch.equal(part.current_player, full.current_player[g0:g1])
            assert torch.equal(part.window_center, full.window_center[g0:g1])
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


# ═══ OF2-3a / OF2-3a' — the normalisation algebra ════════════════════════════════════════
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


#: The MIXED masks are deliberately UNBALANCED across every split boundary this row uses.
#: MEASURED at HEAD before the fix: with a mask alternating `[1,0,1,0,...]` the `1/M` and
#: `B_m/B` weightings are correct to 7.3e-08 at k=2 — every micro-batch then carries the same
#: mask count, so the wrong denominator cancels. A balanced mask would have made the k=2 cell
#: GREEN against MB-4 and MB-5 and the row would have reported coverage it did not have.
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
    """OF2-3a — un-split vs the sum of split-and-denominator-weighted parts, fp32, NO model,
    across mask VALUE x mask PRESENCE.

    This is the exact-equivalence claim, so the tolerance is `rtol=1e-6 / atol=1e-8` on a
    quantity that is algebraically identical — it is a floating-point-associativity bound, not
    a discrepancy budget. MB-4 (`1/M` weighting) and MB-5 (`B_m/B` weighting) are GREEN on the
    all-ones cell and RED on every mixed cell, which is why the cross exists."""
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
    """OF2-3a' — the two denominators are DIFFERENT quantities and the implementation says so.

    HEAD's policy denominator is a sum of mask VALUES (`losses.py:101-102` casts the mask to
    the loss dtype and sums it); HEAD's value denominator is a COUNT of TRUE entries
    (`dist65.py:58-62` divides by `kept.numel()`). They agree only while the masks are
    strictly 0/1 — which they are in production (uint8 at `dispatch.py:126-128`) — so a
    SYMMETRIC implementation passes every other row in this file while encoding a latent
    divergence. The mask here is deliberately NOT 0/1, which is the only input on which the
    two expressions can be told apart."""
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
    """OF2-3a' second limb — `bin_logits.shape[0] == n_graphs` is CHECKED at the call, not
    assumed. The `value_valid is None` arm sets the value denominator to the graph count while
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
            recent_buffer=None,
            caps_provider=lambda: MicrobatchCapsSpec(max_edges=caps[0], max_nodes=caps[1]))


# ═══ OF2-3b/c/d — end to end through the real Trainer ════════════════════════════════════
#: Warm-up steps taken on BOTH arms, with identical non-binding caps, before the measured step.
#:
#: **WHY THIS EXISTS, IN ONE LINE: at a COLD AdamW the per-parameter difference between the
#: arms is DOMINATED BY {~0, 2*lr}, so OF2-3d's <=1.0e-4 envelope is unreachable in that
#: regime — the readings that fire its ABORT are all at exactly 2*lr, which is 200% of the
#: update the envelope's own justification calls O(lr) and asks a 10% bound on.** The warm-up
#: is not a convenience that makes a number pass and must never be read as one: it puts the
#: optimizer in the regime the envelope was written for, and the regime run5 occupies after its
#: first few steps. Deleting it does not make this row stricter; it makes the row measure a
#: statistic that cannot express the property (RULED: conformance, not amendment — the envelope
#: is PREREG_DFIX §4's and is UNCHANGED).
#:
#: **CORRECTION 1, from REVIEW-impl, applied here because this is the artifact readers read
#: (R96): the quantisation is NOT to exactly two values.** An earlier version of this comment
#: said "SIGN-QUANTISED to {0, 2*lr}" and then, four lines later, reported three discrete
#: measured values — the middle one, 6.588e-04, is 0.66*lr and lands in the PASS-WITH-
#: DISCLOSURE band, so the cold statistic CAN take an intermediate value. The sentence
#: contradicted its own measurement one paragraph apart. "Dominated by {~0, 2*lr}, with an
#: intermediate value observed" is what the numbers support, and the ruling survives it
#: because every ABORT firing sat at exactly 2*lr.
#:
#: **CORRECTION 2: what warming gives up, ARGUED rather than assumed.** Warming does remove one
#: sensitivity — an isolated sign flip on a near-zero-gradient parameter reads 2*lr cold and
#: reads small warm. That is not lost DEFECT coverage, and the same measurement is why: those
#: flips occur between arms whose loss is BIT-IDENTICAL and whose gradient cosine is 0.999993,
#: i.e. they are the noise channel, not the defect channel. The defect channel is carried by
#: OF2-3b (loss), OF2-3c (cosine) and OF2-5 (the clip COUNT, which no numeric regime can
#: absorb) — none of which the warm-up touches. MB-4 and MB-6 are measured RED with it in
#: place.
#:
#: Not a tolerance and not an envelope — it is the OPTIMIZER STATE the pre-registered
#: statistic presumes.
#:
#: MEASURED, at a COLD optimizer, over 20 trials: `max|dtheta|` takes three discrete values —
#: 1.5e-08, 6.588e-04 and 2.000e-03 — and lands PASS 7 / DISCLOSE 9 / ABORT 4. The mechanism
#: was measured, not guessed: AdamW's FIRST step has `v_hat ~ g^2`, so the update collapses to
#: `lr * sign(g)` and discards gradient magnitude entirely. The 2.000e-03 reading is one
#: parameter of 2834 whose gradient flipped sign between the arms (+5.259e-04 vs -5.500e-04,
#: on a parameter whose |g| is 3.1x BELOW the median |g|), giving |dtheta| = 2*lr = 1.999963e-03
#: to five significant figures. Only 2 of 2834 parameters exceeded 1e-4 at all.
#:
#: So at step 1 the statistic has no resolution: it reads ~0 or ~2*lr depending on whether any
#: single near-zero-gradient parameter flips, and it would ship a 20%-flaky ABORT that fires
#: for a reason which is NOT the split. PREREG's own registered justification for the 1.0e-4
#: bound is *"updates are O(lr) = O(1e-3), so this is a 10% bound on the update"* — which
#: presumes an update PROPORTIONAL to the gradient, exactly what AdamW's first step is not.
#: Three warm-up steps populate the second moment and restore that proportionality: measured
#: over 20 trials, `max|dtheta|` becomes 1.371e-05 .. 3.523e-05, PASS 20/20.
#:
#: **NO ENVELOPE IS MOVED.** All three bands below are PREREG_DFIX §4's, unchanged. What
#: changed is the instrument's optimizer state (R61: the threshold is not tuned; the
#: measurement is made to measure what it says it measures).
_WARMUP_STEPS = 3


def _two_arm_step(tmp_path, m: int):
    """One M=1 step and one M=k step over the SAME wire, from the SAME weights AND the same
    warmed optimizer state. Both arms take the identical warm-up sequence, so the ONLY
    difference between them at the measured step is the micro-batch partition."""
    buf = H.uniform_graph_buffer(8)
    replay = H.ReplayWireBuffer(buf, 4)
    non_binding = H.non_binding_caps(replay.wire)
    out = []
    for caps in (non_binding, H.caps_for_exactly(replay.wire, m)):
        trainer = H.tiny_graph_trainer(tmp_path)
        for _ in range(_WARMUP_STEPS):
            run_declared_train_step(
                trainer, replay, H.GSPEC, batch_size=4, augment=False, recency_weight=0.0,
                recent_buffer=None,
                caps_provider=lambda: MicrobatchCapsSpec(max_edges=non_binding[0],
                                                         max_nodes=non_binding[1]))
        info = run_declared_train_step(
            trainer, replay, H.GSPEC, batch_size=4, augment=False, recency_weight=0.0,
            recent_buffer=None,
            caps_provider=lambda c=caps: MicrobatchCapsSpec(max_edges=c[0], max_nodes=c[1]))
        out.append((info, H.grad_vector(trainer.model), H.param_vector(trainer.model)))
    return out


@pytest.mark.parametrize("m", [2, 4])
def test_of2_3bcd_split_step_matches_the_unsplit_step(tmp_path, m: int) -> None:
    """OF2-3b/c/d — loss and grad-norm relative deltas, gradient cosine and post-step
    parameter deltas, against PREREG_DFIX §4's envelopes, THREE BANDS each.

    The middle band is a PASS-WITH-DISCLOSURE and is reported, not failed: asserting only the
    PASS band would fire a HALT on a result the prereg explicitly accepts with disclosure.

    Both arms are WARMED first (`_WARMUP_STEPS`) because a COLD AdamW's first step is
    `lr*sign(g)`: the per-parameter arm difference is then dominated by `{~0, 2*lr}` (with an
    intermediate value observed), and every reading that fires OF2-3d's ABORT sits at exactly
    `2*lr` — 200% of the update its `<=1.0e-4` envelope asks a 10% bound on. See
    `_WARMUP_STEPS` for the measurement, for what warming gives up, and for why that residue
    is the noise channel rather than the defect channel. **No envelope here was moved** — all
    three bands are PREREG_DFIX §4's.

    DISCLOSED (PREREG §4, finding 6): this harness is where the split's summation-regrouping
    and bf16-accumulation effects are SMALLEST. BF2-5 is the production-scale instrument; if
    it cannot run, the production-scale claim is UNVERIFIED and may not be made."""
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


# ═══ OF2-4 / OF2-5 — cadence and clip-once ═══════════════════════════════════════════════
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
            recent_buffer=None,
            caps_provider=lambda: MicrobatchCapsSpec(max_edges=caps[0], max_nodes=caps[1]))
    finally:
        torch.nn.utils.clip_grad_norm_ = real_clip
    return SimpleNamespace(info=info, sink=sink, trainer=trainer, opt=opt_spy,
                           sched=sched_spy, before=before, clips=len(clip_calls),
                           ckpts=sorted((tmp_path / "ckpt").glob("*.ckpt")))


@pytest.mark.parametrize("m", [1, 2, 4])
def test_of2_4_one_optimizer_step_and_five_keys_at_every_m(tmp_path, m: int) -> None:
    """OF2-4 — ONE of everything per training step, at M in {1, 2, 4}, **and the returned dict
    carries all five keys at every M**.

    The key-presence half is not decoration. `coordinator/step.py`'s grad-norm gate reads
    `float(loss_info.get("grad_norm", 0.0))` and fires `grad_norm_hard_abort` off it, so a branch that returns a dict WITHOUT `grad_norm` silently feeds an armed
    run-safety abort a `0.0` that always passes its threshold — and nothing else in this
    repository notices (MB-22). Sweeping M includes M=1, the path production takes whenever
    the caps do not bind, so the guarantee is not tested only on the exotic branch."""
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
    """OF2-4's SEVENTH count — the EMA update, which the row's registered PASS column names
    (`1 / 1 / 1 / 1 / 1 / +1 / 1`) and which the sibling row above cannot reach.

    MEASURED, and this is why the leg exists separately: `tiny_graph_trainer` builds from
    `configs/dev_example.yaml`, which declares no `ema` block, so `ema_model is None` and
    `trainer/core.py`'s EMA branch NEVER EXECUTES in any default-fixture row — at any M. An
    EMA update moved inside the accumulation loop (the MB-8 shape, one row over) would have
    fired M times per training step and reded nothing behaviourally. This drive enables EMA
    for real and counts the updates; `update_parameters` is additionally named in OF2-9 leg
    2's `_FORBIDDEN_IN_LOOP`, so the same mutation dies structurally too, at any M and
    whatever the fixture's EMA posture."""
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
        recent_buffer=None,
        caps_provider=lambda: MicrobatchCapsSpec(max_edges=caps[0], max_nodes=caps[1]))
    assert trainer.step - before == 1
    assert len(updates) == 1, (
        f"M={m}: the EMA update fired {len(updates)} times in ONE training step. It is a TAIL "
        "statement (DESIGN §3.6 lists it beside `self.step += 1`); inside the accumulation "
        "loop it would fire once per MICRO-BATCH and smooth the weights M times per step.")


@pytest.mark.parametrize("m", [1, 2, 4])
def test_of2_5_clip_grad_norm_is_called_exactly_once_for_any_m(tmp_path, m: int) -> None:
    """OF2-5 — clipping is NONLINEAR in the whole gradient, so it happens ONCE, after the
    accumulation. Per-micro clipping would feed `grad_norm_hard_abort` the norm of a FRACTION
    of the gradient and rescale a live abort threshold by an operator-invisible M (MB-6).

    A call COUNT cannot be absorbed by variance, which is why it is the primary assertion and
    the numeric comparison below is the second detector rather than the only one."""
    r = _drive_with_spies(tmp_path, m)
    assert r.clips == 1, f"M={m}: clip_grad_norm_ called {r.clips} times, want exactly 1"


def test_of2_5_grad_norm_matches_the_unsplit_steps_norm(tmp_path) -> None:
    """OF2-5 second limb — the reported `grad_norm` is the norm of the ACCUMULATED gradient,
    within PREREG's 5.0e-2. Compared against the UN-SPLIT step on the SAME wire, never against
    a self-consistent value: MB-10 (return the last micro-batch's own norm) reads
    approximately `1 - 1/M` off that comparison and exactly 0 off a self-comparison."""
    (one, _, _), (split, _, _) = _two_arm_step(tmp_path, 4)
    rel = abs(split["grad_norm"] - one["grad_norm"]) / max(abs(one["grad_norm"]), 1e-12)
    assert rel <= 5.0e-2, f"|dgrad_norm|/grad_norm = {rel:.3e} > 5.0e-2 (MB-10's surface)"


# ═══ OF2-6 — the LAW-18 counter ══════════════════════════════════════════════════════════
@pytest.mark.parametrize("m", [1, 2, 4])
def test_of2_6_graph_trainer_step_event_carries_the_counter_and_its_caps(tmp_path,
                                                                         m: int) -> None:
    """OF2-6 — LAW-18: the lever logs its own fire-rate in-run, and the CAPS travel beside it.

    A fire-rate of 1 is uninterpretable without the bound that produced it, and an event
    carrying the numerator without the denominator is the shape R164 rejected. `M` is
    computed HERE from the wire's own per-graph counts (MB-11's kill surface: a hard-coded
    `microbatches: 1` in the payload), never read back out of the event."""
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


def test_of2_6_the_dense_event_carries_none_of_the_five_keys(tmp_path) -> None:
    """OF2-6 second limb — the five keys are GRAPH-ROUTE ONLY. Asserted as an ABSENCE, which
    a "keys present" oracle would not see (MB-12)."""
    sink = H.SpySink()
    torch.manual_seed(H.SEED)
    arch = CnnArch(board_size=int(_DSPEC.board_size), in_channels=int(_DSPEC.n_planes),
                   filters=8, res_blocks=1)
    trainer = Trainer(build_net(arch), H.minted_config("sustained_kcluster.yaml"),
                      arch=arch, checkpoint_dir=tmp_path / "dense", device=torch.device("cpu"),
                      train_hparams=H.graph_hparams(), sink=sink)
    run_declared_train_step(trainer, _dense_buffer(), _DSPEC, batch_size=4, augment=False,
                            recency_weight=0.0, recent_buffer=None,
                            caps_provider=_never_called_provider)
    ev = sink.named("trainer_step")[0]
    assert ev["representation"] == "grid"
    for key in ("microbatches", "edges", "nodes", "caps_max_edges", "caps_max_nodes"):
        assert key not in ev, f"the DENSE trainer_step event carries {key!r} (MB-12)"


def _never_called_provider() -> MicrobatchCapsSpec:
    raise AssertionError(
        "the grid arm invoked caps_provider — the caps are graph-route only and the grid "
        "route must not be able to reach them (DESIGN_DFIX §3.11.1, F2-ABORT-5)")


def _dense_buffer(n_records: int = 8, capacity: int = 64) -> ReplayBuffer:
    rb = ReplayBuffer(capacity, GRID_ENCODING)
    s = int(_DSPEC.board_size)
    n_cells = s * s
    for i in range(n_records):
        state = np.zeros((int(_DSPEC.n_planes), s, s), dtype=np.float16)
        state[0, 0, i % s] = 1.0
        chain = np.zeros((6, s, s), dtype=np.float16)
        policy = np.zeros(int(_DSPEC.policy_stride), dtype=np.float32)
        policy[i % n_cells] = 1.0
        own = np.zeros(n_cells, dtype=np.uint8)
        wl = np.zeros(n_cells, dtype=np.uint8)
        rb.push(state, chain, policy, 1.0 if i % 2 == 0 else -1.0, own, wl)
    return rb


# ═══ OF2-7 — the out-of-domain graph ═════════════════════════════════════════════════════
@pytest.mark.parametrize("member", ["max_edges", "max_nodes"])
def test_of2_7_a_single_over_cap_graph_raises_and_nothing_partial_happens(tmp_path,
                                                                         member: str) -> None:
    """OF2-7 — R114's clause: never a silent truncation, never a silent drop.

    Run for BOTH members, because an edges-only check passes the edge arm and leaves the node
    arm unguarded. The absence assertions are the second half and they are not redundant:
    MB-14 moves the check AFTER `optimizer.zero_grad()`, which still raises and still corrupts
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
            recent_buffer=None,
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


# ═══ OF2-11 — determinism ════════════════════════════════════════════════════════════════
def test_of2_11_partition_boundaries_are_identical_over_100_repeats() -> None:
    """OF2-11 — the partition is a pure function of `(ec, nc, caps)`. Byte-identical
    boundaries over 100 repeats; a host-state dependence shows as a single differing tuple.

    Disclosed asymmetry (MB-2): a bin-packing reorder is still DETERMINISTIC, so this row
    cannot see it. Only OF2-1's ordered-cover property can."""
    ec = np.array([5, 9, 3, 9, 2, 11, 4], dtype=np.int64)
    nc = np.array([4, 2, 7, 1, 7, 3, 5], dtype=np.int64)
    first = plan_microbatches(_offsets(ec), _offsets(nc), 16, 12)
    for _ in range(99):
        assert plan_microbatches(_offsets(ec), _offsets(nc), 16, 12) == first
    assert len(first) >= 2


def test_of2_11_records_whether_deterministic_mode_rejects_index_add(tmp_path, capsys) -> None:
    """OF2-11 second limb — RECORDED, NOT GATED. `index_add_`'s CUDA backward is an atomic
    scatter-add that torch documents as nondeterministic; whether THIS build rejects it under
    `use_deterministic_algorithms(True)` is data this phase prints rather than a property it
    asserts. Asserting it would gate on a torch implementation detail."""
    buf = H.uniform_graph_buffer(8)
    replay = H.ReplayWireBuffer(buf, 4)
    caps = H.non_binding_caps(replay.wire)
    outcome = "accepted"
    try:
        with H.deterministic_algorithms():
            trainer = H.tiny_graph_trainer(tmp_path)
            run_declared_train_step(
                trainer, replay, H.GSPEC, batch_size=4, augment=False, recency_weight=0.0,
                recent_buffer=None,
                caps_provider=lambda: MicrobatchCapsSpec(max_edges=caps[0],
                                                         max_nodes=caps[1]))
    except RuntimeError as exc:                      # noqa: BLE001 — recorded, then re-read
        outcome = f"rejected: {exc}"
    print(f"OF2-11 determinism observation (device=cpu, torch={torch.__version__}): {outcome}")
    assert outcome  # the row's content is the RECORD; there is nothing here to gate


# ═══ OF2-13 — the armed graph-weights refusal ════════════════════════════════════════════
def test_of2_13_a_nonzero_forbidden_weight_raises_before_any_state_moves(tmp_path) -> None:
    """OF2-13 — the FIRST behavioural producer the `GRAPH_FORBIDDEN_NONZERO_WEIGHTS` ban has
    ever had (R4/LAW-07). At HEAD the only test naming the ban is
    `tests/config/test_train_entropy.py:73-81`, which asserts the string is ABSENT from the
    schema module — a duplication guard, not a producer. Deleting the loop (MB-23) reds
    nothing anywhere in the repository at HEAD."""
    buf = H.uniform_graph_buffer(8)
    replay = H.ReplayWireBuffer(buf, 4)
    caps = H.non_binding_caps(replay.wire)
    sink = H.SpySink()
    trainer = H.tiny_graph_trainer(tmp_path, sink=sink, ownership_weight=0.5)
    spy = H.OptimizerSpy(trainer.optimizer)
    before = trainer.step
    with pytest.raises(ValueError, match="ownership_weight"):
        run_declared_train_step(
            trainer, replay, H.GSPEC, batch_size=4, augment=False, recency_weight=0.0,
            recent_buffer=None,
            caps_provider=lambda: MicrobatchCapsSpec(max_edges=caps[0], max_nodes=caps[1]))
    assert trainer.step == before
    assert spy.zero_grads == 0 and spy.steps == 0
    assert sink.named("trainer_step") == []


# ═══ OF2-15 — the ROUTE-SCOPED resolution ════════════════════════════════════════════════
def _coordinator(full_config: dict, trainer: Any, buffer: Any) -> StepCoordinator:
    """A real `StepCoordinator` over the given `full_config`. The collaborators this row does
    not exercise are `None`; the ONE fake beside them is the step config (a namespace rather
    than the frozen `StepCoordinatorConfig`, whose every field is required by invariant —
    `coordinator/config.py:283`). `_run_training_step` reads only `batch_size`, `augment` and
    `recency_weight` off it, and none of the three is this row's subject."""
    return StepCoordinator(
        trainer=trainer, buffer=buffer, pretrained_buffer=None, recent_buffer=None,
        pool=None, eval_pipeline=None, subsystems=None, anchor_state=None, shutdown=None,
        eval_model=None, bufs=None,
        config=SimpleNamespace(selfplay_stall_timeout_sec=1800.0),
        full_config=full_config)


def test_of2_15a_a_grid_step_over_a_trainless_config_never_resolves_the_caps(tmp_path,
                                                                            monkeypatch) -> None:
    """OF2-15(a) — the REGRESSION pin against this fix's own earlier shape.

    Five FROZEN files build a `StepCoordinator` whose `full_config` carries no `train` key,
    and at least `test_clean_stop_save.py` provably runs real steps through that call. An
    eager `caps=self._microbatch_caps()` at `coordinator/step.py:930` — Python evaluates every
    argument before the call — would resolve `full_config["train"]` on BOTH representations
    and raise `KeyError: 'train'` there (MB-26). The provider is passed UNCALLED and only the
    graph arm invokes it, so the grid route cannot read the caps: a property of the call
    graph, checkable from two signatures.

    GREEN at HEAD and green after — declared, so it is not claimed as more than it is."""
    calls: list[Any] = []
    real = resolve_microbatch_caps

    def _spy(cfg: Any):
        calls.append(cfg)
        return real(cfg)

    monkeypatch.setattr("mantis.config.resolve.microbatch.resolve_microbatch_caps", _spy)
    monkeypatch.setattr("mantis.train.coordinator.step.resolve_microbatch_caps", _spy,
                        raising=False)
    torch.manual_seed(H.SEED)
    arch = CnnArch(board_size=int(_DSPEC.board_size), in_channels=int(_DSPEC.n_planes),
                   filters=8, res_blocks=1)
    trainer = Trainer(build_net(arch), H.minted_config("sustained_kcluster.yaml"),
                      arch=arch, checkpoint_dir=tmp_path / "dense", device=torch.device("cpu"),
                      train_hparams=H.graph_hparams())
    coord = _coordinator(dict(_TRAINLESS_GRID), trainer, _dense_buffer())
    cfg = SimpleNamespace(batch_size=4, augment=False, recency_weight=0.0)
    info = coord._run_training_step(cfg)         # a REAL grid training step
    assert "loss" in info and trainer.step == 1
    assert calls == [], (
        "a grid training step resolved train.microbatch_caps — the caps are graph-route only "
        "and five frozen coordinators carry no `train` key (MB-26, F2-ABORT-5)")


def test_of2_15a_the_grid_arm_does_not_take_the_provider_at_all() -> None:
    """OF2-15(a), structural half — "the grid arm cannot read the caps" is a CALL-GRAPH
    property, not a convention a future edit can break silently. Two signatures say so."""
    graph_params = inspect.signature(dispatch_mod._graph_step).parameters
    grid_params = inspect.signature(dispatch_mod._grid_step).parameters
    assert "caps_provider" in graph_params
    assert "caps_provider" not in grid_params, (
        "_grid_step accepts caps_provider — under the design the grid arm is not GIVEN the "
        "provider, so a grid run structurally cannot reach the caps (DESIGN_DFIX §3.11.1)")
    top = inspect.signature(run_declared_train_step).parameters["caps_provider"]
    assert top.default is inspect.Parameter.empty, (
        "caps_provider has a default — a default is a code-side default for a config-derived "
        "value and a caller that forgot it would silently get an UNCAPPED step (R1)")


def test_of2_15b_a_graph_route_without_the_block_raises_by_name() -> None:
    """OF2-15(b) — the ⊕ half. An absent cap on the graph route is a NAMED raise, never a
    default. MB-27's mutation is `full_config.get("train", {})`: a cap that silently becomes
    absent-and-unbounded REPORTS AS PRESENT, which is the phantom-gate class (R4/LAW-07) and
    the exit the dispatcher ruled out in advance (F2-ABORT-5)."""
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
    """OF2-15(b) second limb — `_graph_step` does not wrap it, does not catch it and has no
    fallback arm, so the named error reaches the caller of `run_declared_train_step`."""
    coord = _coordinator(dict(_TRAINLESS_GRAPH), None, None)
    trainer = H.tiny_graph_trainer(tmp_path)
    replay = H.ReplayWireBuffer(H.uniform_graph_buffer(8), 4)
    with pytest.raises(MissingMicrobatchCapsError):
        run_declared_train_step(trainer, replay, H.GSPEC, batch_size=4, augment=False,
                                recency_weight=0.0, recent_buffer=None,
                                caps_provider=coord._microbatch_caps)


def test_of2_15_the_resolver_is_memoised_and_reads_the_config_once() -> None:
    """OF2-15 third limb — `_microbatch_caps` mirrors `_step_spec` (`step.py:936-942`) in
    MEMOISATION, so the resolver runs once per coordinator however many steps a burst takes."""
    cfg = {"identity": _TRAINLESS_GRAPH["identity"],
           "train": {"microbatch_caps": {"max_edges": 11, "max_nodes": 7}}}
    coord = _coordinator(cfg, None, None)
    first = coord._microbatch_caps()
    assert first is coord._microbatch_caps()
    assert (first.max_edges, first.max_nodes) == (11, 7)


# ═══ OF2-16 — the empty batch (DEFENSIVE) ════════════════════════════════════════════════
def test_of2_16_zero_graphs_plan_to_zero_parts() -> None:
    """OF2-16 — `plan_microbatches` returns `()` at `B == 0`. A naive reading of the greedy
    loop appends a trailing part unconditionally and yields `[(0, 0)]`: one EMPTY part, which
    would then collate a zero-graph batch and produce a gradient-free loss."""
    empty = np.array([0], dtype=np.int64)
    assert plan_microbatches(empty, empty, 10, 10) == ()


def test_of2_16_a_zero_part_step_raises_before_zero_grad(tmp_path) -> None:
    """OF2-16 — a step with no graphs cannot produce a gradient, so the only honest outcomes
    are a raise or a silent no-op, and a silent no-op would let a run report steps it never
    took (LAW-14's posture, MB-28).

    **DECLARED DEFENSIVE.** Whether `sample_graph_batch` can return `n_graphs == 0` through
    the coordinator's `min_buf_size` gate is UNVERIFIED — DESIGN_DFIX §3.3 names the settling
    measurement and this row does not stand in for it. This is a producer test for a GUARD,
    and it may not be cited as evidence that `B == 0` occurs."""
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
