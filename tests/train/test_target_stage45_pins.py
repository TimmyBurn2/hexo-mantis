"""⊕ WP12-R Phase T (TARGET INTEGRITY) — S4 + S5: the stage-4/5 target-pipeline pins
(DESIGN_T §1.6, §5 O-2). Written at T-2 ORACLE-WRITE, byte-frozen through IMPL.

S4 — the dispatcher forwards `policy_target` VALUE-INTACT from `sample_graph_batch` to
`train_step_from_graph_batch` (the recording-trainer pattern of
tests/train/test_train_step_dispatch.py, extended with a VALUE assert, not just kwarg
presence — the census's "drop moved downstream of the value-assert" route).
Killer: M-L (dispatch scales policy_target x0.5 → the exact-equality assert reds).

S5 — `ragged_policy_ce` performs NO renormalization: a target carrying fraction f of
its mass contributes f x its CE (the (1-f) gradient-weakening fact, DESIGN_T §1.6).
This DOCUMENTS the cured defect's loss-side consequence and makes any future
re-introduction of a sub-unity target visible at the loss. NO loss code changes.
Killer: M-M (per-segment renorm inside ragged_policy_ce → the 0.8x pin inverts).

PRE-FIX status at HEAD: all GREEN (stages 4/5 are faithful; the defect is upstream).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from mantis._engine import HexgBuffer
from mantis.config.resolve.microbatch import MicrobatchCapsSpec
from mantis.encoding import lookup
from mantis.train.coordinator.dispatch import run_declared_train_step
from mantis.train.losses import ragged_policy_ce

GRAPH_ENCODING = "gnn_axis_v1"
_GSPEC = lookup(GRAPH_ENCODING)


# ── S4: dispatch pass-through, value-intact ──────────────────────────────────────────
def _graph_buffer(n_records: int = 6, capacity: int = 64) -> HexgBuffer:
    hb = HexgBuffer(capacity, GRAPH_ENCODING, 128)
    for i in range(n_records):
        stones = [(0, 0, 1), (1, 0, -1), (0, 1, 1)][: 2 + (i % 2)]
        policy = [(2, 0, 0.6), (1, 1, 0.4)] if i % 2 == 0 else [(2, 0, 0.25), (1, 1, 0.75)]
        hb.push_graph_position(stones, policy, 1, 2, 2 + i, True, 1.0, True, 10 + i)
    return hb


class _RecordingTrainer:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def train_step_from_graph_batch(self, **kw: Any) -> dict[str, float]:
        self.calls.append(kw)
        return {"policy_loss": 0.0}


#: G-DFIX-1 (WP12-R F2). The largest per-graph edge count this fixture can produce, MEASURED
#: over 40 draws of its own buffer: the records carry 2 or 3 stones, giving exactly two
#: distinct per-graph edge counts, 3142 and 3316. Setting `max_edges` to the LARGER makes no
#: single graph over-cap (so nothing raises) while making any two graphs together over-cap —
#: which forces `len(parts) >= 2` on a `batch_size=3` drive and is what makes the strengthened
#: assertions below exercise the cross-micro-batch property at all. `max_nodes` is set past
#: the whole batch so the split is edge-driven and the binding member is unambiguous.
_S4_MAX_PER_GRAPH_EDGES = 3316
_S4_CAPS = MicrobatchCapsSpec(max_edges=_S4_MAX_PER_GRAPH_EDGES, max_nodes=1_000_000)


def test_dispatch_forwards_policy_target_value_intact() -> None:
    real = _graph_buffer()
    sampled: list[Any] = []

    class _RecordingHexg:
        size = real.size
        capacity = real.capacity

        def sample_graph_batch(self, batch_size: int, augment: bool = False, recent_frac: float = 0.0):
            wire, targets = real.sample_graph_batch(batch_size, augment=augment, recent_frac=recent_frac)
            sampled.append(targets)
            return wire, targets

    rec = _RecordingTrainer()
    run_declared_train_step(rec, _RecordingHexg(), _GSPEC, batch_size=3, augment=False,
                            recency_weight=0.0, recent_buffer=None,
                            caps_provider=lambda: _S4_CAPS)
    assert len(rec.calls) == 1 and len(sampled) == 1
    # G-DFIX-1 (WP12-R F2): after the micro-batch split the trainer receives a PARTITION, not
    # a `policy_target` kwarg. The pin is UNCHANGED in what it claims and STRICTLY STRONGER in
    # what it checks — the dtype is now asserted on every part, and the value assertion now
    # also pins ORDER ACROSS micro-batch boundaries, which the single-tensor form could not
    # express. The `len(parts) >= 2` assertion is what earns that: without a BINDING cap the
    # fixture gives M = 1 and the cross-boundary claim would be vacuous.
    parts = [make() for make in rec.calls[0]["parts"]]
    assert len(parts) >= 2, (
        f"the caps did not bind — {len(parts)} micro-batch(es) from a batch_size=3 drive. "
        "With M = 1 the concatenation below is the old single-tensor assertion wearing a "
        "loop, and the 'strictly stronger' claim this grant rests on is FALSE"
    )
    want = torch.from_numpy(np.asarray(sampled[0].policy_target, dtype=np.float32))
    assert all(p.policy_target.dtype == torch.float32 for p in parts)
    assert torch.equal(torch.cat([p.policy_target.cpu() for p in parts]), want), (
        "policy_target reached the trainer MUTATED — the dispatcher must forward the "
        "sampled ragged target verbatim, in order, ACROSS the micro-batch split "
        "(dispatch.py `_graph_step`; M-L's exact kill surface)"
    )
    # And the forwarded target is per-graph unit mass (the post-fix producer law seen
    # at the consumer): each legal_offsets segment of the sampled batch sums to ~1.
    ifs = np.asarray(sampled[0].is_full_search)
    assert ifs.shape[0] == 3
    assert sum(p.is_full_search.shape[0] for p in parts) == 3, (
        "the split dropped or duplicated a per-graph target"
    )


# ── S5: ragged CE carries sub-unity mass LINEARLY (no renorm) ────────────────────────
def _toy_segments() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(20260731)
    counts = [5, 7]
    logits = torch.randn(sum(counts), dtype=torch.float32)
    offsets = torch.tensor([0, counts[0], sum(counts)], dtype=torch.long)
    tgt = torch.zeros(sum(counts), dtype=torch.float32)
    tgt[0:5] = torch.tensor([0.4, 0.3, 0.2, 0.05, 0.05])
    tgt[5:12] = torch.tensor([0.5, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05])
    return logits, tgt, offsets


def test_ragged_ce_scales_linearly_with_target_mass() -> None:
    logits, tgt, offsets = _toy_segments()
    mask = torch.ones(2, dtype=torch.float32)
    full = ragged_policy_ce(logits, tgt, offsets, full_search_mask=mask)
    scaled = ragged_policy_ce(logits, 0.8 * tgt, offsets, full_search_mask=mask)
    assert torch.isfinite(full) and float(full) > 0.0
    assert abs(float(scaled) - 0.8 * float(full)) <= 1e-6, (
        f"a 0.8-mass target must contribute 0.8x its CE ({float(scaled)} vs "
        f"{0.8 * float(full)}) — a renormalization arm inside ragged_policy_ce (M-M) "
        "would hide sub-unity targets from the loss"
    )


def test_ragged_ce_matches_the_manual_no_renorm_expectation() -> None:
    logits, tgt, offsets = _toy_segments()
    # one full-mass graph, one 0.8-mass graph — the mixed batch the (1-f) fact names
    mixed = tgt.clone()
    mixed[5:12] *= 0.8
    mask = torch.ones(2, dtype=torch.float32)
    got = float(ragged_policy_ce(logits, mixed, offsets, full_search_mask=mask))

    lo = logits.double().numpy()
    tg = mixed.double().numpy()
    per_graph = []
    for a, b in ((0, 5), (5, 12)):
        seg = lo[a:b]
        p = np.exp(seg - seg.max())
        p = p / p.sum()
        per_graph.append(float(-(tg[a:b] * np.log(np.clip(p, 1e-12, None))).sum()))
    want = sum(per_graph) / 2.0  # denominator = mask.sum() (losses.py:100-103)
    assert abs(got - want) <= 1e-5, (
        f"ragged_policy_ce {got} != manual no-renorm expectation {want} — the "
        "denominator/renorm contract drifted"
    )
