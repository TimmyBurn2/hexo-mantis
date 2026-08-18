"""WPSC Phase 3 SC-B6 — masking single-authority (A4; DESIGN_P3.md §7). `legal_mask`
(dense bool) and `legal_offsets` (CSR) are two VIEWS of the identical Rust-computed legal
set, constructed together in `collate_graph_batch` (`graph_collate.py`) and never
independently re-derived downstream. `train_step_from_graph_batch` (`trainer/core.py`)
and `ragged_policy_ce`/`GnnNet.forward_batch` consume them as required, no-default
parameters — there is no signature-level path to omit-and-fall-back-to-a-recomputed-mask.

R284 UPDATE: `GnnNet.forward_batch`'s required view is now `legal_index` (the wire's
`legal_node_gather`) rather than `legal_mask` — the third co-derived view of the SAME
Rust-computed legal set, and the one the ragged output is ordered by. `legal_mask` remains on
`GraphBatch` and remains constructed in the same act from the same array, so the two rows below
that assert mask/CSR agreement are unaffected and still assert what they always did.

`tests/selfplay/test_gnn_seam_smoke.py`/`test_buffer_facade.py` read first per DESIGN_P3.
md §7.2 — neither covers this exact masking-consistency/AST-scan surface, so this is a
genuinely new file. Reuses the session-scoped `payload_fields` fixture from `tests/
selfplay/conftest.py` (the "b6" golden — a REAL 6-graph fixture, sha-pinned, not a
from-scratch synthetic payload) rather than hand-building a wire payload.

GREEN-guard (DESIGN_P3.md §7.2): all three assertion classes already hold against HEAD's
current code — a producer test for an EXISTING correct structure (LAW-07), not a
RED-at-import pin. Stays in the tree unstaged.
"""
from __future__ import annotations

import inspect
import textwrap

from mantis.model.gnn import GnnNet
from mantis.selfplay.graph_collate import GraphWirePayload, collate_graph_batch
from mantis.train.losses import ragged_policy_ce
from mantis.train.trainer.core import Trainer


def test_legal_mask_and_legal_offsets_are_one_set_total_count(payload_fields) -> None:
    batch = collate_graph_batch(GraphWirePayload(**payload_fields("b6")))
    assert int(batch.legal_mask.sum().item()) == int(batch.legal_offsets[-1].item())


def test_legal_mask_and_legal_offsets_agree_per_graph_segment(payload_fields) -> None:
    batch = collate_graph_batch(GraphWirePayload(**payload_fields("b6")))
    assert batch.n_graphs >= 2, "the b6 golden must carry >=2 graphs for a real per-segment check"
    for i in range(batch.n_graphs):
        lo, hi = int(batch.node_offsets[i]), int(batch.node_offsets[i + 1])
        segment_true = int(batch.legal_mask[lo:hi].sum().item())
        csr_count = int(batch.legal_offsets[i + 1] - batch.legal_offsets[i])
        assert segment_true == csr_count, f"graph {i}: mask/CSR disagree"


def test_train_step_from_graph_batch_never_reassigns_legal_mask_or_offsets() -> None:
    """AST/text-level guarantee (§7.2 item 2, the "test-caught" half of A4's phrasing):
    the training-step function body contains no `legal_mask =`/`legal_offsets =`
    statement after the `def` line — the loss and the model forward literally receive
    the identical object the caller (the graph-batch producer) constructed."""
    src = textwrap.dedent(inspect.getsource(Trainer.train_step_from_graph_batch))
    # Scan the whole function body (skip the `def ...(` signature lines themselves,
    # which end at the first line containing the closing `) -> ...:`).
    lines = src.splitlines()
    sig_end = next(i for i, line in enumerate(lines) if line.rstrip().endswith(":"))
    body_text = "\n".join(lines[sig_end + 1:])
    assert "legal_mask =" not in body_text
    assert "legal_offsets =" not in body_text


def test_ragged_policy_ce_and_forward_batch_require_the_legal_set_no_default() -> None:
    """The "unconstructible" half of A4: a caller cannot construct a valid call without
    supplying the legal set — no accidental omit-and-fall-back path.

    RENAMED at R284 (R73 name-truth: a test name is a behavioural claim). `forward_batch`'s
    required legal-set view is now `legal_index` — the wire's `legal_node_gather` — rather than
    `legal_mask`, because the boolean mask forced a `nonzero` and with it a host-device
    synchronization on the serve thread's hot path (P-MASK). **The A4 property this row exists
    for is unchanged and is checked on the same parameter POSITION**: no default, so there is
    still no signature-level path to omit the legal set and fall back to a recomputed one. What
    moved is WHICH co-derived view the net requires, not whether it requires one.

    The alternative considered and rejected was an OPTIONAL `legal_index` with a mask fallback:
    that is precisely the omit-and-fall-back shape this file bans, and a caller that forgot it
    would silently get the slow path (`plan/R284_PERF_DESIGN.md` §1.5)."""
    ce_params = inspect.signature(ragged_policy_ce).parameters
    assert ce_params["legal_offsets"].default is inspect.Parameter.empty

    fwd_params = inspect.signature(GnnNet.forward_batch).parameters
    assert "legal_mask" not in fwd_params, (
        "a `legal_mask` parameter beside `legal_index` would be two authorities for one set"
    )
    assert fwd_params["legal_index"].default is inspect.Parameter.empty
