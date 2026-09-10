"""Hold the legal set to ONE authority across its co-derived views.

The dense mask, the CSR offsets and the gather are views of the identical Rust-computed legal
set, constructed together in `collate_graph_batch` and never re-derived downstream; every
consumer takes its view as a required, no-default parameter, so there is no signature-level
path to omit it and fall back to a recomputed one.
"""
from __future__ import annotations

import inspect
import textwrap

from mantis.model.gnn import GnnNet
from mantis.selfplay.graph_collate import GraphWirePayload, collate_graph_batch
from mantis.train.losses import ragged_policy_ce
from mantis.train.trainer.core import Trainer


def test_the_gather_and_the_CSR_are_one_set_total_count(payload_fields, wire_geometry) -> None:
    """Prove the gather and the CSR are one set: same total count AND no duplicate entries.

    The mask form implied uniqueness through a scatter that collapsed repeats; a naive
    `len(gather) == legal_offsets[-1]` would lose that half silently, so both are explicit.
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
    """Prove each graph's gather slice sits inside its own node range and matches the CSR length."""
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
    """Prove the legal set is never re-derived inside a consumer that actually receives it.

    The subject is asserted present first: a scan pointed at a function that never had the name
    is a guard with nothing to be red about. Derived from the AST, because a text scan for
    `"legal_offsets ="` also matches a comment, a docstring, or `legal_offsets == x`.
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
    """Prove both consumers require the legal set with no default, so omitting it is
    unconstructible rather than a silent fallback.

    A `legal_mask` parameter beside `legal_index` would be two authorities for one set, and an
    optional `legal_index` with a mask fallback would hand a forgetful caller the slow path.
    """
    ce_params = inspect.signature(ragged_policy_ce).parameters
    assert ce_params["legal_offsets"].default is inspect.Parameter.empty

    fwd_params = inspect.signature(GnnNet.forward_batch).parameters
    assert "legal_mask" not in fwd_params, (
        "a `legal_mask` parameter beside `legal_index` would be two authorities for one set"
    )
    assert fwd_params["legal_index"].default is inspect.Parameter.empty
