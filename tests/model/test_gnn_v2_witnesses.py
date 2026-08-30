# >300 justify (R8): each witness is a claim V2 makes plus the measurement that would falsify
# it, and the pair only means anything together — a witness whose falsifier lives in another file
# can be weakened on one side without the other side going red.
"""The PRE-REGISTERED behavioral witnesses for `GnnNetV2` (SEAM-B1 Leg 2).

Registered BEFORE any V2 line existed, in the governance workspace at
`plan/SEAM_B1_LEG2_PREREG.md` §2, and reproduced here as the tests that read them. Each witness
names the CLAIM, the MEASUREMENT, and the OUTCOME THAT FALSIFIES the claim — the Q-C10 shape.

THE FRAME, common to every witness and load-bearing: CPU, seeded, both nets built at the SAME
declared widths, `eval()`, `no_grad()`, **random init**. These are comparisons of FUNCTION FORM.
NOTHING HERE IS A STRENGTH CLAIM in either direction, and no row may be cited as evidence that
V2 plays better — F-01 is the standing fence, where static probes passed while self-play
collapsed to 0–1 %. Strength is operator-only, on the box, post-mint.

TWO OF THESE WITNESSES CAN FALSIFY A CANDIDATE'S PREMISE, not just its implementation, and that
is deliberate. W-C1(ii) reads V1's own dummy-aggregation growth curve — which the WP-AXIS2 memo
records as UNMEASURED in-tree — and a flat V1 curve would mean the hazard C(i) targets does not
exist on this wire. That outcome was registered as a live possibility before the number was
seen, so if it fires it is a finding, not a test to relax.
"""
from __future__ import annotations

import pytest
import torch

from mantis.model import GnnArch, GnnArchV2, build_net, net_param_hash
from mantis.model.gnn_v2 import GnnNetV2

#: The seed every witness builds under. An instrument parameter: it fixes WHICH random nets are
#: compared, and no claim below depends on its value.
_SEED = 20260830

#: Widths. Small enough to be a default-tier test, and identical across the two arches — the
#: comparison is of readout and aggregation, so any width difference would confound it.
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
    """A synthetic graph with `n_real` real nodes plus ONE dummy wired bidirectionally to all.

    SYNTHETIC AND LABELLED AS SUCH. It reproduces the wire's dummy topology
    (`crates/mantis-graph/src/lib.rs`: the dummy is bidirectionally connected to every real node
    with all-zero attrs) at node counts a constructed position need not reach, because the claim
    under test is about how the aggregation SCALES. It is not a position and no reachability is
    implied by it — T6's rule about never letting a synthetic point pass for a reachable one.
    """
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


# ── W-A1 — the readout consumes a MAX statistic V1 cannot see ────────────────────────────
def _readout_value(net, emb: torch.Tensor, masks: dict, is_v2: bool) -> torch.Tensor:
    """Drive the READOUT from a given trunk embedding, through the net's own value head.

    The perturbation the witness needs is on the trunk EMBEDDING, so the trunk is not in the
    path — see the disclosure on the witness below for why that is what was registered and why
    it matters.
    """
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
    """CLAIM: V2's readout is sensitive to the maximum over real nodes; V1's mean is not.
    MEASUREMENT: raise the TRUNK EMBEDDING of one node that is real but NOT a stone, so the
    stone-masked mean is untouched by construction. FALSIFIER: V2's |Δvalue| not strictly
    greater than V1's.

    DISCLOSURE — this witness's FIRST implementation fired, and the diagnosis is recorded here
    rather than smoothed away, because a pre-registered falsifier that fires is the one moment
    the record earns its cost. That implementation deviated from the registered measurement in
    two ways at once: it perturbed the INPUT rather than the trunk embedding, so the pre-norm
    LayerNorm attenuated the spike and message passing spread it; and it spiked a STONE node,
    which sits in the mean's 3-element denominator and therefore moves the mean MORE than it
    moves a max over 64 nodes. It also compared |Δvalue| across two nets whose value heads have
    different fan-in (V2's is 2x wide, so its init scale is ~1/sqrt(2) of V1's) — a confound
    that would have made the comparison unreadable even had the spike been placed correctly.
    Implemented as registered, V1's delta is EXACTLY zero, which is what removes the confound:
    no head-scale difference can make zero the larger number.

    The spike magnitude and node counts are INSTRUMENT PARAMETERS, not thresholds on a subject.
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
    """The negative control. Without it, W-A1 passes on a V2 whose max spans every row
    including the wire's own artefact — which would be a different net making a different claim.
    """
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
    """The measured quantity is a DELTA; two differently-shaped value heads do not produce the
    same value, so "the values match" could never have been this control."""
    v1, v2 = _nets()
    g = _star_graph(n_real=64)
    assert float((_value_of(v1, g) - _value_of(v1, g)).abs().max()) == 0.0
    assert float((_value_of(v2, g) - _value_of(v2, g)).abs().max()) == 0.0


# ── W-A2 — the max spans REAL nodes, dummy excluded, and the exclusion is DERIVED ─────────
def test_W_A2_the_readout_IGNORES_the_dummy_row_and_SEES_a_real_one() -> None:
    """CLAIM: the max is taken over `real = stone | legal`, so the dummy is excluded — and the
    exclusion is derived from the wire's two masks, since the wire carries no `real_mask`.
    MEASUREMENT: two arms. FALSIFIER: the value moves when the DUMMY row alone is made extreme
    (the dummy is in the max), or it does NOT move when a REAL row is made extreme (the readout
    ignores everything and the first arm passes vacuously).

    Driven at the readout rather than through the trunk: message passing would carry a dummy
    perturbation into real nodes and both arms would move for a reason that is not the max.
    """
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
    """The derivation half. A literal `N-1` would be right today — the dummy IS the last row —
    and would be a code-side constant standing in for a wire fact the moment the builder
    reorders. So the mask is built from `stone | legal` and this drives that construction."""
    stone = torch.tensor([True, True, False, False, False])
    legal = torch.tensor([2, 3])
    real = GnnNetV2.real_mask_from_batch(stone, legal)
    assert real.tolist() == [True, True, True, True, False]
    assert stone.tolist() == [True, True, False, False, False], "the input mask was mutated"


def test_W_A2_the_max_FALLS_BACK_to_all_nodes_when_none_are_masked() -> None:
    """The fallback exists so both halves of the readout degenerate the same way; without it a
    graph with no real nodes would hand the value head a dtype floor, which reads as a number."""
    from mantis.model.gnn_v2 import segment_max_with_fallback

    emb = torch.tensor([[1.0], [5.0], [2.0]])
    none_masked = torch.zeros(3, dtype=torch.bool)
    out = segment_max_with_fallback(emb, none_masked, torch.zeros(3, dtype=torch.long), 1)
    assert float(out[0, 0]) == 5.0


# ── W-A3 — batched and deploy readouts agree on the new branch ────────────────────────────
def test_W_A3_forward_single_AGREES_with_forward_batch_on_one_graph() -> None:
    """CLAIM: V2's deploy twin computes the same readout as its batched path, in the sense V1's
    pair is already held to. MEASUREMENT: one graph through both. FALSIFIER: a disagreement
    exceeding V1's own, measured here rather than assumed — the MEAN half carries V1's ~5e-7
    accumulation-order drift and the MAX half adds no drift term, so V2 must not be worse."""
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


# ── W-C1 — the dummy's aggregation is flat in N where V1's grows ──────────────────────────
def _dummy_agg_norms(net, counts: tuple[int, ...]) -> list[float]:
    """`‖agg[dummy]‖₂` at the FIRST conv, captured by a forward hook on the one aggregation
    authority. Hooked rather than recomputed: a Python re-implementation of the aggregation
    would be a second authority and could agree with itself while the real one drifted."""
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
    """CLAIM: V2 degree-normalizes the dummy's incoming aggregation, so its magnitude stops
    tracking the real-node count. MEASUREMENT: `‖agg[dummy]‖` at a ladder of node counts, from a
    hook on the conv itself. FALSIFIER, TWO-SIDED and both sides required:

      (i) V2's growth ratio at V1's — the normalization is not in the path;
      (ii) V1's ratio ≈ 1 — the hazard C(i) targets does not exist on this wire, in which case
           **C(i) is unmotivated on the evidence** and that is a finding to report rather than a
           test to relax. The memo records this curve as UNMEASURED in-tree, so (ii) was a live
           possibility when this was registered.

    The ladder is SYNTHETIC and labelled: it reproduces the wire's dummy topology at node counts
    chosen to span an order of magnitude, and no reachability is claimed for any of them.
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
    """CLAIM: C(i) normalizes the DUMMY's incoming sum only; real-node GINE sums and the count
    signal they carry are byte-identical to V1's. MEASUREMENT: remove the dummy's edges and
    forward both nets from identical weights. FALSIFIER: any difference at all.

    This is the witness that keeps C(i) from quietly becoming the global mean aggregation the
    memo explicitly does NOT propose — that would destroy GINE's injectivity premise, and it
    would look exactly like C(i) from the outside.
    """
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


# ── W-ID1 — V2 has its own canonical identity ─────────────────────────────────────────────
def test_W_ID1_the_canonical_hash_SEPARATES_the_two_arches_and_is_STABLE() -> None:
    """CLAIM: `net_param_hash` denominates V2 distinctly. FALSIFIER: equal hashes across the two
    arches (the identity does not separate them), or unequal hashes across two builds of one
    arch (the identity is not stable, and a golden over it would mean nothing)."""
    v1, v2 = _nets()
    assert net_param_hash(v1) != net_param_hash(v2)
    again_v1, again_v2 = _nets()
    assert net_param_hash(v1) == net_param_hash(again_v1)
    assert net_param_hash(v2) == net_param_hash(again_v2)


#: V2's OWN GOLDEN, at its own slot. Measured on the implementation this commit lands, at the
#: widths and seed above, twice. It denominates the identity primitive for V2 the way
#: `test_net_param_hash_promotion.py`'s literal does for V1: a digest that moved without the
#: arch moving would void every cross-drive comparison R317(c)(i) makes.
_V2_GOLDEN = "620b2ada10d3d0c4d5372c7ea2a2297666c5843adbfbebc37e7d05855bebdf73"


def test_the_V2_golden_holds() -> None:
    _v1, v2 = _nets()
    assert net_param_hash(v2) == _V2_GOLDEN, (
        "V2's canonical parameter hash moved. If the arch genuinely changed, the golden moves "
        "with it IN THE SAME COMMIT; if it did not, this is the determinism gate firing"
    )


def test_the_V2_golden_is_NOT_the_V1_golden() -> None:
    """PK2's second half. A golden slot that happened to hold V1's digest would pass every test
    above and denominate the wrong net."""
    v1, _v2 = _nets()
    assert net_param_hash(v1) != _V2_GOLDEN


@pytest.mark.parametrize("net_kind", ["v1", "v2"])
def test_both_arches_FORWARD_on_a_real_wire_position(net_kind) -> None:
    """The synthetic star graph is a topology, not a position. This drives both nets on a wire
    the engine actually produced, so no witness above rests on synthetic input alone."""
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
