# >300 justify (R8): each witness is a claim plus the measurement that falsifies it, and split
# across files one side can be weakened without the other going red.
"""The pre-registered behavioral witnesses for `GnnNetV2`.

Common frame, load-bearing: CPU, seeded, both nets at the SAME widths, `eval()`, `no_grad()`,
random init. These compare FUNCTION FORM. NOTHING HERE IS A STRENGTH CLAIM in either direction
— static probes have passed while self-play collapsed to 0-1 %.
"""
from __future__ import annotations

import pytest
import torch

from mantis.model import GnnArch, GnnArchV2, build_net, net_param_hash
from mantis.model.gnn_v2 import GnnNetV2

#: An instrument parameter: it fixes WHICH random nets are compared, and no claim uses it.
_SEED = 20260830

#: Identical across the two arches — a width difference would confound the comparison.
_WIDTHS = {"in_dim": 11, "edge_dim": 5, "hidden": 8, "num_layers": 2,
           "policy_hidden": 8, "value_hidden": 8}


def _nets() -> tuple[torch.nn.Module, torch.nn.Module]:
    """One V1 and one V2 at the same widths, each from the same seed."""
    torch.manual_seed(_SEED)
    v1 = build_net(GnnArch(**_WIDTHS)).eval()
    torch.manual_seed(_SEED)
    v2 = build_net(GnnArchV2(**_WIDTHS)).eval()
    return v1, v2


def _star_graph(n_real: int, n_stones: int = 3) -> dict:
    """Build a synthetic graph: `n_real` real nodes plus ONE dummy wired to all of them.
    It reproduces the wire's dummy topology; it is not a position and implies no reachability."""
    dummy = n_real
    n = n_real + 1
    src = torch.cat([torch.arange(n_real), torch.full((n_real,), dummy)])
    dst = torch.cat([torch.full((n_real,), dummy), torch.arange(n_real)])
    edge_index = torch.stack([src, dst])
    stone_mask = torch.zeros(n, dtype=torch.bool)
    stone_mask[:n_stones] = True
    legal_mask = torch.zeros(n, dtype=torch.bool)
    legal_mask[n_stones:n_real] = True
    return {
        "x": torch.randn(n, _WIDTHS["in_dim"]),
        "edge_index": edge_index,
        "edge_attr": torch.zeros(edge_index.shape[1], _WIDTHS["edge_dim"]),
        "stone_mask": stone_mask,
        "legal_mask": legal_mask,
        "legal_index": legal_mask.nonzero(as_tuple=True)[0],
    }


def _value_of(net, g: dict) -> torch.Tensor:
    with torch.no_grad():
        _policy, value, _bins = net.forward_batch(
            g["x"], g["edge_index"], g["edge_attr"], g["legal_index"], g["stone_mask"]
        )
    return value


def _readout_value(net, emb: torch.Tensor, masks: dict, is_v2: bool) -> torch.Tensor:
    """Drive the READOUT from a trunk embedding, leaving the trunk out of the path."""
    from mantis.model.gnn import segment_mean_with_fallback
    from mantis.model.gnn_v2 import segment_max_with_fallback

    with torch.no_grad():
        batch_vec = torch.zeros(emb.shape[0], dtype=torch.long)
        mean = segment_mean_with_fallback(emb, masks["stone"], batch_vec, 1)
        pooled = (
            torch.cat((mean, segment_max_with_fallback(emb, masks["real"], batch_vec, 1)), -1)
            if is_v2
            else mean
        )
        return net.value_head(pooled)[0]


def _readout_masks(n_real: int, n_stones: int) -> dict:
    n = n_real + 1
    stone = torch.zeros(n, dtype=torch.bool)
    stone[:n_stones] = True
    legal = torch.zeros(n, dtype=torch.bool)
    legal[n_stones:n_real] = True
    return {"stone": stone, "legal": legal, "real": stone | legal, "dummy_row": n_real,
            "legal_row": n_stones}


def test_W_A1_a_spike_that_raises_the_MAX_without_moving_the_MEAN_moves_only_V2() -> None:
    """V2's readout sees a spike on a real non-stone node; V1's stone-masked mean cannot.

    Spiking a NON-stone node makes V1's delta EXACTLY zero, which removes the confound that the
    two value heads have different fan-in and so are not comparable in |Δvalue| magnitude.
    """
    v1, v2 = _nets()
    masks = _readout_masks(n_real=64, n_stones=3)
    torch.manual_seed(_SEED)
    emb = torch.randn(65, _WIDTHS["hidden"] * _WIDTHS["num_layers"])

    spiked = emb.clone()
    spiked[masks["legal_row"]] += 25.0

    d_v1 = float((_readout_value(v1, spiked, masks, False)
                  - _readout_value(v1, emb, masks, False)).abs().max())
    d_v2 = float((_readout_value(v2, spiked, masks, True)
                  - _readout_value(v2, emb, masks, True)).abs().max())
    assert d_v1 == 0.0, (
        f"V1's value moved by {d_v1} under a spike on a NON-stone node. The measurement's whole "
        "premise is that the stone-masked mean cannot see this node; if it can, the comparison "
        "below is confounded and this witness is not reading what it claims"
    )
    assert d_v2 > d_v1, (
        f"V2 moved by {d_v2} and V1 by {d_v1}. The max half is not reaching the value head — "
        "either the concat is not the vector the head consumes, or the max spans the wrong set"
    )


def test_W_A1_control_a_spike_on_the_DUMMY_row_moves_NEITHER_net() -> None:
    """Negative control: without it the row above passes on a V2 whose max spans the dummy."""
    v1, v2 = _nets()
    masks = _readout_masks(n_real=64, n_stones=3)
    torch.manual_seed(_SEED)
    emb = torch.randn(65, _WIDTHS["hidden"] * _WIDTHS["num_layers"])
    spiked = emb.clone()
    spiked[masks["dummy_row"]] += 25.0

    for net, is_v2 in ((v1, False), (v2, True)):
        moved = float((_readout_value(net, spiked, masks, is_v2)
                       - _readout_value(net, emb, masks, is_v2)).abs().max())
        assert moved == 0.0, f"the dummy row moved a readout by {moved}"


def test_W_A1_control_the_delta_of_an_UNPERTURBED_graph_is_zero_for_both() -> None:
    """The measured quantity is a DELTA — two differently-shaped heads never match on value."""
    v1, v2 = _nets()
    g = _star_graph(n_real=64)
    assert float((_value_of(v1, g) - _value_of(v1, g)).abs().max()) == 0.0
    assert float((_value_of(v2, g) - _value_of(v2, g)).abs().max()) == 0.0


def test_W_A2_the_readout_IGNORES_the_dummy_row_and_SEES_a_real_one() -> None:
    """The max ignores the dummy row and sees a real one; the second arm stops the first
    passing vacuously. Driven at the readout, since message passing would carry a dummy
    perturbation into real nodes and move both arms for a reason that is not the max."""
    from mantis.model.gnn_v2 import segment_max_with_fallback

    n_real, d = 8, 4
    emb = torch.zeros(n_real + 1, d)
    batch_vec = torch.zeros(n_real + 1, dtype=torch.long)
    real_mask = torch.zeros(n_real + 1, dtype=torch.bool)
    real_mask[:n_real] = True

    baseline = segment_max_with_fallback(emb, real_mask, batch_vec, 1)

    dummy_hot = emb.clone()
    dummy_hot[n_real] = 99.0
    assert torch.equal(segment_max_with_fallback(dummy_hot, real_mask, batch_vec, 1), baseline), (
        "the dummy row moved the max, so the max spans the dummy — the readout is reading a "
        "node that is an artefact of the wire rather than a fact about the position"
    )

    real_hot = emb.clone()
    real_hot[0] = 99.0
    assert not torch.equal(
        segment_max_with_fallback(real_hot, real_mask, batch_vec, 1), baseline
    ), "a real row did not move the max either — this witness is measuring nothing"


def test_W_A2_the_real_mask_is_DERIVED_from_the_wires_two_masks() -> None:
    """The real mask is built from `stone | legal`, not a literal `N-1`: the latter is right
    only until the builder reorders the rows."""
    stone = torch.tensor([True, True, False, False, False])
    legal = torch.tensor([2, 3])
    real = GnnNetV2.real_mask_from_batch(stone, legal)
    assert real.tolist() == [True, True, True, True, False]
    assert stone.tolist() == [True, True, False, False, False], "the input mask was mutated"


def test_W_A2_the_max_FALLS_BACK_to_all_nodes_when_none_are_masked() -> None:
    """The max falls back to all nodes when none are masked; otherwise a graph with no real
    nodes hands the value head a dtype floor, which reads as a number."""
    from mantis.model.gnn_v2 import segment_max_with_fallback

    emb = torch.tensor([[1.0], [5.0], [2.0]])
    none_masked = torch.zeros(3, dtype=torch.bool)
    out = segment_max_with_fallback(emb, none_masked, torch.zeros(3, dtype=torch.long), 1)
    assert float(out[0, 0]) == 5.0


def test_W_A3_forward_single_AGREES_with_forward_batch_on_one_graph() -> None:
    """V2's deploy twin agrees with its batched path, against V1's own gap measured here: the
    MAX half is order-independent, so V2 must not be worse than V1's ~5e-7 drift."""
    v1, v2 = _nets()
    g = _star_graph(n_real=32)
    args = (g["x"], g["edge_index"], g["edge_attr"])

    def gap(net, legal_arg):
        with torch.no_grad():
            _p_b, v_b, _l_b = net.forward_batch(*args, g["legal_index"], g["stone_mask"])
            _p_s, v_s, _l_s = net.forward_single(*args, legal_arg, g["stone_mask"])
        return float((v_b.squeeze() - v_s).abs().max())

    v1_gap = gap(v1, g["legal_mask"])
    v2_gap = gap(v2, g["legal_mask"])
    assert v2_gap <= max(v1_gap, 1e-5), (
        f"V2's batched/deploy gap is {v2_gap} against V1's {v1_gap}. The max is "
        "order-independent, so it contributes no drift term — a larger gap means the two V2 "
        "paths are not computing the same readout"
    )


def _dummy_agg_norms(net, counts: tuple[int, ...]) -> list[float]:
    """Return `‖agg[dummy]‖₂` at the first conv, hooked on the one aggregation authority: a
    re-implementation could agree with itself while the real one drifted."""
    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        captured.append(output.detach())

    handle = net.representation.convs[0].register_forward_hook(hook)
    norms: list[float] = []
    try:
        for n_real in counts:
            captured.clear()
            g = _star_graph(n_real=n_real)
            _value_of(net, g)
            norms.append(float(captured[0][n_real].norm()))
    finally:
        handle.remove()
    return norms


def test_W_C1_the_dummy_row_GROWS_with_N_under_V1_and_is_FLATTER_under_V2() -> None:
    """V2 degree-normalizes the dummy's incoming aggregation, so `‖agg[dummy]‖` stops tracking
    the real-node count where V1's grows.

    Two-sided: V2's ratio at V1's means the normalization is not in the path; V1's ratio near 1
    means the hazard has no subject on this wire, and that is a finding, not a test to relax.
    """
    v1, v2 = _nets()
    counts = (16, 128, 1024)

    torch.manual_seed(_SEED)
    v1_norms = _dummy_agg_norms(v1, counts)
    torch.manual_seed(_SEED)
    v2_norms = _dummy_agg_norms(v2, counts)

    v1_ratio = v1_norms[-1] / max(v1_norms[0], 1e-12)
    v2_ratio = v2_norms[-1] / max(v2_norms[0], 1e-12)
    assert v1_ratio > 2.0, (
        f"V1's dummy aggregation norm went {v1_norms} over {counts}, a ratio of {v1_ratio}. "
        "FALSIFIER (ii) HAS FIRED: the unnormalised sum does NOT grow with the real-node count "
        "on this wire, so GNN-3's size-generalisation hazard has no subject here and candidate "
        "C(i) is unmotivated on the evidence. Report it; do not weaken this bound."
    )
    assert v2_ratio < v1_ratio, (
        f"V2's ratio {v2_ratio} is not below V1's {v1_ratio} (norms {v2_norms} vs {v1_norms}). "
        "FALSIFIER (i): the degree normalization is not in the aggregation path"
    )


def test_W_C2_real_node_aggregation_is_UNTOUCHED_when_the_dummy_is_absent() -> None:
    """With the dummy's edges removed the two trunks are byte-identical from equal weights —
    a global mean aggregation would destroy GINE's injectivity and look the same from
    outside."""
    v1, v2 = _nets()
    v2.load_state_dict(
        {k: v.clone() for k, v in v1.state_dict().items() if not k.startswith("value_head.")},
        strict=False,
    )
    g = _star_graph(n_real=32)
    keep = (g["edge_index"][0] != 32) & (g["edge_index"][1] != 32)
    edge_index = g["edge_index"][:, keep]
    edge_attr = g["edge_attr"][keep]

    with torch.no_grad():
        emb_v1 = v1.node_embeddings(g["x"], edge_index, edge_attr)
        emb_v2 = v2.node_embeddings(g["x"], edge_index, edge_attr, ~g["stone_mask"] & ~g["legal_mask"])
    assert torch.equal(emb_v1, emb_v2), (
        "with the dummy's edges gone the two trunks disagree, so C(i) is touching real-node "
        "aggregation — which is the global-mean change the memo explicitly does not propose"
    )


def test_W_ID1_the_canonical_hash_SEPARATES_the_two_arches_and_is_STABLE() -> None:
    """`net_param_hash` separates the two arches and is stable across two builds of one — a
    golden over an unstable identity would mean nothing."""
    v1, v2 = _nets()
    assert net_param_hash(v1) != net_param_hash(v2)
    again_v1, again_v2 = _nets()
    assert net_param_hash(v1) == net_param_hash(again_v1)
    assert net_param_hash(v2) == net_param_hash(again_v2)


#: V2's golden, measured twice at the widths and seed above; a digest that moves without the
#: arch moving voids every cross-drive comparison.
_V2_GOLDEN = "620b2ada10d3d0c4d5372c7ea2a2297666c5843adbfbebc37e7d05855bebdf73"


def test_the_V2_golden_holds() -> None:
    _v1, v2 = _nets()
    assert net_param_hash(v2) == _V2_GOLDEN, (
        "V2's canonical parameter hash moved. If the arch genuinely changed, the golden moves "
        "with it IN THE SAME COMMIT; if it did not, this is the determinism gate firing"
    )


def test_the_V2_golden_is_NOT_the_V1_golden() -> None:
    """A golden slot holding V1's digest would pass every row above and denominate the wrong
    net."""
    v1, _v2 = _nets()
    assert net_param_hash(v1) != _V2_GOLDEN


@pytest.mark.parametrize("net_kind", ["v1", "v2"])
def test_both_arches_FORWARD_on_a_real_wire_position(net_kind) -> None:
    """Drive both nets on a wire the engine actually produced, so no witness above rests on
    synthetic input alone."""
    import mantis.encoding as encoding
    from mantis._engine import Board, HexgBuffer
    from mantis.selfplay.graph_collate import collate_graph_batch

    spec = next(s for s in encoding.all_specs() if s.is_graph)
    board = Board.with_encoding_name(spec.name)
    for i in range(6):
        board.apply_move(i, 0)
    legal = board.legal_moves()
    buffer = HexgBuffer(2, spec.name, 8)
    buffer.push_graph_position(
        board.get_stones(), [(legal[0][0], legal[0][1], 1.0)],
        board.current_player, board.moves_remaining, board.ply, True, 0.0, True, 8,
    )
    wire, _targets = buffer.sample_graph_batch(1, augment=False)
    batch = collate_graph_batch(
        wire, trunk_size=spec.trunk_size, win_length=spec.win_length,
        node_feat_dim=spec.node_feat_dim, edge_feat_dim=spec.edge_feat_dim,
    )
    stone_mask = torch.zeros(batch.x.shape[0], dtype=torch.bool)
    stone_mask[: int(batch.n_stones.sum())] = True

    widths = {**_WIDTHS, "in_dim": spec.node_feat_dim, "edge_dim": spec.edge_feat_dim}
    torch.manual_seed(_SEED)
    net = build_net(GnnArch(**widths) if net_kind == "v1" else GnnArchV2(**widths)).eval()
    with torch.no_grad():
        policy, value, bins = net.forward_batch(
            batch.x, batch.edge_index, batch.edge_attr,
            batch.legal_node_gather, stone_mask, batch.node_offsets,
        )
    assert policy.shape[0] == batch.legal_node_gather.shape[0]
    assert value.shape == (1, 1)
    assert bins.shape == (1, widths["n_value_bins"] if "n_value_bins" in widths else 65)
    assert torch.isfinite(value).all() and torch.isfinite(policy).all()
