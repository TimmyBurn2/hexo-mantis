"""P-MASK output-parity oracle — the sync-free gather is BYTE-IDENTICAL.

`forward_batch` gathers the legal-node rows with `index_select` instead of a boolean mask. Both
are pure row copies, so the assertion is `torch.equal` and NOT `allclose`: a tolerance passes
through the one failure that can occur here, a row REORDERING. The reference arm re-expresses
the OLD formulation through the net's public parts, and `forward_single`, still on the mask
form, is a second cross-formulation reference at B=1.

MUTATION: the wrong-gather rows drive the SAME production forward with a corrupted index and
assert the oracle SEES it — an oracle that cannot fail on a wrong gather is not evidence.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from mantis.encoding import lookup
from mantis.model import GnnArch, GnnNet, build_net
from mantis.selfplay.graph_collate import (
    GraphWirePayload,
    collate_graph_batch,
    stone_mask_from_batch,
)

_ENC = "gnn_axis_v1"

#: The committed collate payload bank. The selfplay conftest exposes the same files through a
#: session fixture, but cross-test imports are barred and a conftest fixture does not reach a
#: sibling directory, so the two-line loader is duplicated. The files are the authority.
_COLLATE = Path(__file__).resolve().parents[1] / "fixtures" / "selfplay" / "collate"


@pytest.fixture(scope="module")
def payload_fields():
    """Factory -> a FRESH `GraphWirePayload` ctor-kwarg dict; arrays are copied per call because
    the mutation rows corrupt their payload in place."""
    import json
    scalars_all = json.loads(
        (_COLLATE / "collate_expectations.json").read_text(encoding="utf-8")
    )["payloads"]

    def _load(stem: str) -> dict:
        with np.load(_COLLATE / f"{stem}_payload.npz") as z:
            fields: dict = {k: z[k].copy() for k in z.files}
        fields.update({k: int(v) for k, v in scalars_all[stem]["scalars"].items()})
        return fields
    return _load


def _net(spec) -> GnnNet:
    torch.manual_seed(20260818)
    net = build_net(GnnArch(in_dim=spec.node_feat_dim, edge_dim=spec.edge_feat_dim))
    assert isinstance(net, GnnNet)
    net.eval()
    return net


def _batch(payload_fields, stem: str):
    spec = lookup(_ENC)
    batch = collate_graph_batch(
        GraphWirePayload(**payload_fields(stem)),
        expected_version=1,
        trunk_size=spec.trunk_size,
        win_length=spec.win_length,
        node_feat_dim=spec.node_feat_dim,
        edge_feat_dim=spec.edge_feat_dim,
        device="cpu",
    )
    return spec, batch


def _legal_mask(batch) -> torch.Tensor:
    """The dense boolean view of the legal set, built HERE from the gather — which does not cost
    independence, because the collate built the same mask by scattering that same gather. What
    these rows compare is two INDEXING FORMULATIONS over one index set."""
    mask = torch.zeros(batch.x.shape[0], dtype=torch.bool)
    mask[batch.legal_node_gather.to(torch.int64)] = True
    return mask


def _reference_logits(net: GnnNet, batch, stone_mask) -> torch.Tensor:
    """The OLD line, through public parts: `emb[legal_mask]` -> policy MLP."""
    emb = net.node_embeddings(batch.x, batch.edge_index, batch.edge_attr)
    return net.policy_head.mlp(emb[_legal_mask(batch)]).squeeze(-1)


@pytest.mark.parametrize("stem", ["b1", "b6"])
@pytest.mark.parametrize("autocast", [False, True], ids=["fp32", "bf16-autocast"])
def test_gather_is_byte_identical_to_the_boolean_mask(payload_fields, stem, autocast) -> None:
    spec, batch = _batch(payload_fields, stem)
    net = _net(spec)
    stone_mask = stone_mask_from_batch(batch)
    with torch.inference_mode(), torch.autocast(
        device_type="cpu", dtype=torch.bfloat16, enabled=autocast
    ):
        got, _value, _bins = net.forward_batch(
            batch.x, batch.edge_index, batch.edge_attr,
            batch.legal_node_gather, stone_mask, batch.node_offsets,
        )
        ref = _reference_logits(net, batch, stone_mask)
    assert got.shape == ref.shape
    assert got.dtype == ref.dtype
    assert torch.equal(got, ref), (
        "P-MASK parity: index_select gather is not byte-identical to the boolean mask "
        f"(max |diff| = {float((got.float() - ref.float()).abs().max()):.3e})"
    )


@pytest.mark.parametrize("stem", ["b1", "b6"])
def test_value_and_bin_logits_are_untouched_by_the_gather_change(payload_fields, stem) -> None:
    """The value head reads `stone_mask`, not the legal gather; pinned so a future edit routing
    the value head through the legal path is caught here rather than by a strength number."""
    spec, batch = _batch(payload_fields, stem)
    net = _net(spec)
    stone_mask = stone_mask_from_batch(batch)
    with torch.inference_mode():
        _logits, value, bins = net.forward_batch(
            batch.x, batch.edge_index, batch.edge_attr,
            batch.legal_node_gather, stone_mask, batch.node_offsets,
        )
        emb = net.node_embeddings(batch.x, batch.edge_index, batch.edge_attr)
        from mantis.model.gnn import _node_offsets_to_batch_vec, segment_mean_with_fallback
        bv = _node_offsets_to_batch_vec(batch.node_offsets)
        pooled = segment_mean_with_fallback(emb, stone_mask, bv, batch.n_graphs)
        ref_value, ref_bins = net.value_head(pooled)
    assert torch.equal(value, ref_value)
    assert torch.equal(bins, ref_bins)


def test_forward_single_is_the_independent_cross_formulation_reference(payload_fields) -> None:
    """`forward_single` still gathers with the BOOLEAN MASK, so at B=1 it must agree with the
    batched gather to the batched path's pooling tolerance — a ~5e-7 accumulation-order
    difference. The POLICY logits are byte-identical and asserted exactly."""
    spec, batch = _batch(payload_fields, "b1")
    assert batch.n_graphs == 1
    net = _net(spec)
    stone_mask = stone_mask_from_batch(batch)
    with torch.inference_mode():
        batched, b_value, _b = net.forward_batch(
            batch.x, batch.edge_index, batch.edge_attr,
            batch.legal_node_gather, stone_mask, batch.node_offsets,
        )
        single, s_value, _s = net.forward_single(
            batch.x, batch.edge_index, batch.edge_attr, _legal_mask(batch), stone_mask,
        )
    assert torch.equal(batched, single), "policy logits must be byte-identical at B=1"
    assert torch.allclose(b_value.reshape(()), s_value.reshape(()), atol=1e-5)


@pytest.mark.parametrize(
    "corrupt,label",
    [
        (lambda g: torch.flip(g, dims=(0,)), "reversed"),
        (lambda g: torch.roll(g, 1, dims=0), "rolled-by-one"),
        # The label says exactly what the lambda does: `[g[:-1], g[:1]]` puts the FIRST row last.
        # The obvious-looking `[g[:-1], g[-1:]]` is the IDENTITY — a mutation that cannot fail.
        (lambda g: torch.cat([g[:-1], g[:1]]), "last-row-replaced-by-the-first"),
    ],
)
def test_a_wrong_gather_is_SEEN_by_this_oracle(payload_fields, corrupt, label) -> None:
    """MUTATION: the production forward is driven with a deliberately wrong index and the
    oracle's own assertion must fail. The corruptions are order-only or membership-only, so only
    the byte-exact ordered comparison catches them."""
    spec, batch = _batch(payload_fields, "b6")
    net = _net(spec)
    stone_mask = stone_mask_from_batch(batch)
    wrong = corrupt(batch.legal_node_gather)
    assert wrong.shape == batch.legal_node_gather.shape, "the mutation must not change length"
    with torch.inference_mode():
        got, _v, _b = net.forward_batch(
            batch.x, batch.edge_index, batch.edge_attr, wrong, stone_mask, batch.node_offsets,
        )
        ref = _reference_logits(net, batch, stone_mask)
    assert not torch.equal(got, ref), (
        f"MUTATION {label} was NOT seen: a wrong gather produced byte-identical logits, so "
        "this oracle is not evidence for the right one"
    )


def test_the_gather_is_strictly_increasing_on_every_committed_fixture(payload_fields) -> None:
    """The invariant byte-equality rests on, asserted against the fixtures rather than argued
    from the producer: `emb[bool]` returns rows in ascending row index and `index_select` in the
    index's order, and the two agree exactly when the gather is strictly increasing."""
    for stem in ("b0", "b1", "b6", "empty_legal"):
        g = np.asarray(payload_fields(stem)["legal_node_gather"])
        assert g.size == 0 or bool(np.all(np.diff(g) > 0)), f"{stem}: gather not increasing"
