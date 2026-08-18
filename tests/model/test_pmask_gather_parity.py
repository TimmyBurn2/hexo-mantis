"""P-MASK output-parity oracle (R284(b)) — the sync-free gather is BYTE-IDENTICAL.

`GnnNet.forward_batch` gathers the legal-node rows with `emb.index_select(0, legal_index)`
instead of the boolean-mask `emb[legal_mask]` it used before. The two are pure row copies that
perform no arithmetic, so the correct assertion is `torch.equal`, NOT `allclose`: a tolerance
is the weaker claim and would pass through the one failure that can actually occur here — a row
REORDERING, which changes no value and every prior-to-cell pairing.

The reference arm is the OLD FORMULATION re-expressed through the net's own public parts
(`node_embeddings` -> `policy_head.mlp(emb[legal_mask])`), so this compares two formulations
rather than one implementation with itself. `forward_single` is deliberately left on the
boolean-mask form in production code for the same reason and is exercised here as the second,
independent cross-formulation reference at B=1.

Both autocast arms run, and the bf16 arm is LAW-06's regime (graph path = bf16, pinned). The
gather is dtype-agnostic — it copies whatever dtype `emb` carries — so byte-equality is the
claim in every arm, not merely in fp32.

MUTATION (LAW-07, and the dispatch's "RED-verified against a deliberately wrong gather"): the
`_wrong_gather_*` rows drive the SAME production `forward_batch` with a deliberately corrupted
index — reversed, rolled, and one row replaced — and assert the oracle SEES it. A parity oracle
that cannot fail on a wrong gather is not evidence for a right one.
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

#: The committed collate payload bank. `tests/selfplay/conftest.py` exposes the same files
#: through a session fixture, but R5 bars cross-test imports and a conftest fixture does not
#: reach a sibling directory — so the two-line loader is DUPLICATED here rather than shared,
#: on the precedent `tests/eval/test_rung_seat_off_window.py` states in terms ("duplicated from
#: the frozen file rather than imported — R5 bars cross-test imports"). The files are the
#: authority; this is a `np.load`, not a second copy of the data.
_COLLATE = Path(__file__).resolve().parents[1] / "fixtures" / "selfplay" / "collate"


@pytest.fixture(scope="module")
def payload_fields():
    """Factory -> a FRESH `GraphWirePayload` ctor-kwarg dict. Arrays are copied per call for
    the reason the selfplay conftest states: the mutation rows below corrupt their payload in
    place and a shared buffer would leak one row's corruption into the next."""
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


def _reference_logits(net: GnnNet, batch, stone_mask) -> torch.Tensor:
    """The OLD line, through public parts: `emb[legal_mask]` -> policy MLP."""
    emb = net.node_embeddings(batch.x, batch.edge_index, batch.edge_attr)
    return net.policy_head.mlp(emb[batch.legal_mask]).squeeze(-1)


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
    """The value head reads `stone_mask`, not the legal gather. Pinned so a future edit that
    routes the value head through the legal path is caught by THIS oracle rather than by a
    strength number six weeks later."""
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
    """`forward_single` still gathers with the BOOLEAN MASK (production, unchanged). At B=1 it
    must agree with the batched index gather to the batched path's own pooling tolerance —
    `allclose`, not `equal`, and the reason is stated rather than assumed: `forward_single`
    pools with `emb[stone_mask].mean(0)` while `forward_batch` uses segment pooling, a
    documented ~5e-7 accumulation-order difference (`gnn.py` module docstring). The POLICY
    logits, which are what the gather produces, are byte-identical — asserted separately and
    exactly."""
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
            batch.x, batch.edge_index, batch.edge_attr, batch.legal_mask, stone_mask,
        )
    assert torch.equal(batched, single), "policy logits must be byte-identical at B=1"
    assert torch.allclose(b_value.reshape(()), s_value.reshape(()), atol=1e-5)


@pytest.mark.parametrize(
    "corrupt,label",
    [
        (lambda g: torch.flip(g, dims=(0,)), "reversed"),
        (lambda g: torch.roll(g, 1, dims=0), "rolled-by-one"),
        # R73 name-truth: the label says exactly what the lambda does. Written as
        # `[g[:-1], g[:1]]` — the FIRST row replacing the last. The obvious-looking
        # `[g[:-1], g[-1:]]` is the IDENTITY and would be a mutation that can never fail.
        (lambda g: torch.cat([g[:-1], g[:1]]), "last-row-replaced-by-the-first"),
    ],
)
def test_a_wrong_gather_is_SEEN_by_this_oracle(payload_fields, corrupt, label) -> None:
    """MUTATION. The production forward is driven with a deliberately wrong index; the oracle's
    own assertion must fail. Corruptions are order-only or membership-only, so a length check
    or a set check would NOT catch them — only the byte-exact ordered comparison does, which is
    the reason the oracle asserts `torch.equal`."""
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
    """The invariant P-MASK's byte-equality rests on, asserted against the fixtures rather than
    argued from the producer. `emb[bool]` returns rows in ascending row index; `index_select`
    returns them in the index's order; the two agree exactly when this holds."""
    for stem in ("b0", "b1", "b6", "empty_legal"):
        g = np.asarray(payload_fields(stem)["legal_node_gather"])
        assert g.size == 0 or bool(np.all(np.diff(g) > 0)), f"{stem}: gather not increasing"
