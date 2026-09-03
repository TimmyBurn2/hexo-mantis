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


def test_the_gather_and_the_CSR_are_one_set_total_count(payload_fields, wire_geometry) -> None:
    """A4 re-expressed against the GATHER (R297(c)), and it asserts MORE than it used to.

    The old form was `legal_mask.sum() == legal_offsets[-1]`. That was two assertions wearing one
    face, because `legal_mask` was a SCATTER of the gather (`legal_mask_np[gather] = True`): the
    count agreed only if the gather's length matched the CSR **and** the gather had no duplicate
    entries — a duplicate would scatter twice into one cell and sink the sum below the length.

    Re-expressed naively as `len(gather) == legal_offsets[-1]`, the uniqueness half is silently
    lost. Both halves are therefore asserted explicitly below, which is a strictly stronger test
    than the one it replaces and names the property instead of implying it through a scatter.
    """
    batch = collate_graph_batch(GraphWirePayload(**payload_fields("b6")), expected_version=1,
                                **wire_geometry)
    gather = batch.legal_node_gather
    assert int(gather.numel()) == int(batch.legal_offsets[-1].item()), (
        "the gather and the CSR disagree on the size of the legal set"
    )
    assert int(gather.unique().numel()) == int(gather.numel()), (
        "the gather contains a duplicate node — the old mask form caught this implicitly, via a "
        "scatter that collapsed the repeat; it is asserted directly here so it cannot be lost"
    )


def test_the_gather_and_the_CSR_agree_per_graph_segment(payload_fields, wire_geometry) -> None:
    """The per-graph half, re-expressed. Each graph's slice of the gather must fall inside that
    graph's node range and must be exactly as long as the CSR says."""
    batch = collate_graph_batch(GraphWirePayload(**payload_fields("b6")), expected_version=1,
                                **wire_geometry)
    for i in range(int(batch.n_graphs)):
        lo, hi = int(batch.node_offsets[i]), int(batch.node_offsets[i + 1])
        c0, c1 = int(batch.legal_offsets[i]), int(batch.legal_offsets[i + 1])
        segment = batch.legal_node_gather[c0:c1]
        assert int(segment.numel()) == c1 - c0
        assert bool(((segment >= lo) & (segment < hi)).all()), (
            f"graph {i}: a gather entry points outside its own node range [{lo}, {hi}) — the "
            "block-diagonal offset contract is what makes the ragged output re-assemblable"
        )


def test_the_legal_set_is_never_RE_DERIVED_inside_its_consumers() -> None:
    """A4's single-authority half, re-pointed at a consumer that actually has the set.

    **THE OLD FORM WAS VACUOUS AND THIS IS THE FINDING, not a refactor.** It scanned
    `train_step_from_graph_batch`'s source text for `legal_mask =` / `legal_offsets =`. That
    function's signature is `parts`/denominators/caps — it never had either name in it, so the
    assertion could not fail: a guard green because there is nothing there to be red about, which
    is the phantom class (LAW-07). The subject moved when the trainer was partitioned and the
    guard did not move with it.

    Re-pointed at `ragged_policy_ce`, which genuinely takes `legal_offsets` as a required
    parameter, and derived from the **AST** rather than from a substring (R296(f)): a text scan
    for `"legal_offsets ="` also matches a comment, a docstring, or `legal_offsets == x`.
    """
    import ast
    import inspect
    import textwrap

    from mantis.train.losses import ragged_policy_ce

    tree = ast.parse(textwrap.dedent(inspect.getsource(ragged_policy_ce)))
    params = {p.arg for p in tree.body[0].args.args} | {p.arg for p in tree.body[0].args.kwonlyargs}
    assert "legal_offsets" in params, (
        "the re-pointed guard has lost ITS subject too — `ragged_policy_ce` no longer takes the "
        "legal set, so this test is now the vacuous thing it was written to replace"
    )
    rebound = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        and node.id == "legal_offsets"
    ]
    assert not rebound, (
        f"`legal_offsets` is re-assigned inside ragged_policy_ce at line(s) {rebound}; the legal "
        "set has one authority (the wire) and a consumer that rebuilds it is a second"
    )


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
