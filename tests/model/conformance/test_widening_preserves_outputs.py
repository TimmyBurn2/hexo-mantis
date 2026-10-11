"""The width doubling serves its parent's outputs on the golden r8 positions for every graph kind; its split noise lets copies train apart."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from mantis._engine import HexgBuffer
from mantis.encoding import lookup
from mantis.model import ARCH_KINDS, ARCH_KINDS_BY_REPRESENTATION, build_net
from mantis.model.widen import WidenError, widen_arch, widen_net, widen_state
from mantis.selfplay.graph_collate import GraphBatch, collate_graph_batch, stone_mask_from_batch

ENC = "gnn_axis_r8"
GOLDEN = Path(__file__).resolve().parents[2] / "fixtures" / "graph_build" / "positions_r8.txt"
TOL = 1e-4
NEIGHBOURS = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))


def golden_positions(n: int) -> list[tuple[int, int, list[tuple[int, int, int]]]]:
    """The first `n` recorded positions: (to move, stones left this turn, stones)."""
    out = []
    for line in GOLDEN.read_text(encoding="utf-8").splitlines():
        head, *fields = line.split()
        if head != "S":
            continue
        v = [int(f) for f in fields]
        out.append((v[0], v[1], [tuple(v[3 + 3 * i: 6 + 3 * i]) for i in range(v[2])]))
        if len(out) == n:
            break
    return out


def golden_batch(positions: list[tuple[int, int, list[tuple[int, int, int]]]], *, sample: int, seed: int) -> GraphBatch:
    """`sample` graphs drawn by the production sampler from a ring holding `positions`, collated as the trainer does."""
    spec = lookup(ENC)
    buffer = HexgBuffer(max(len(positions), 8), ENC, 8)
    for to_move, left, stones in positions:
        taken = {(q, r) for q, r, _ in stones}
        cell = next((q + dq, r + dr) for q, r, _ in stones for dq, dr in NEIGHBOURS if (q + dq, r + dr) not in taken)
        buffer.push_graph_position(stones, [(cell[0], cell[1], 1.0)], to_move, left, len(stones), True, 0.0, True,
                                   len(stones) + 1)
    buffer.seed_sampler(seed)
    wire, _ = buffer.sample_graph_batch(sample, augment=False)
    return collate_graph_batch(wire, trunk_size=spec.trunk_size, win_length=spec.win_length,
                               node_feat_dim=spec.node_feat_dim, edge_feat_dim=spec.edge_feat_dim, device="cpu")


def parent(kind: str, hidden: int, layers: int) -> tuple[torch.nn.Module, Any]:
    """A seeded parent and its declared arch; every weight, LayerNorm gain and GINE eps is off its init, so no rule holds by accident."""
    spec = lookup(ENC)
    torch.manual_seed(20261014)
    arch = ARCH_KINDS[kind](in_dim=int(spec.node_feat_dim), edge_dim=int(spec.edge_feat_dim), hidden=hidden,
                            num_layers=layers)
    net = build_net(arch)
    with torch.no_grad():
        for name, t in net.state_dict().items():
            if name.endswith(".eps"):
                t.fill_(0.3)
            else:
                t.add_(torch.randn_like(t) * 0.1)
    return net.eval(), arch


def heads(net: torch.nn.Module, b: GraphBatch) -> tuple[torch.Tensor, ...]:
    with torch.no_grad():
        out = net.forward_batch_heads(b.x, b.edge_index, b.edge_attr, b.legal_node_gather, stone_mask_from_batch(b),
                                      b.node_offsets)
    return tuple(t for t in out if t is not None)


def single(net: torch.nn.Module, b: GraphBatch) -> tuple[torch.Tensor, ...]:
    legal = torch.zeros(b.x.shape[0], dtype=torch.bool)
    legal[b.legal_node_gather] = True
    return tuple(net.forward_single(b.x, b.edge_index, b.edge_attr, legal, stone_mask_from_batch(b)))


def max_gap(lhs: tuple[torch.Tensor, ...], rhs: tuple[torch.Tensor, ...]) -> float:
    assert len(lhs) == len(rhs) and all(a.shape == b.shape for a, b in zip(lhs, rhs, strict=True))
    return max(float((a - b).abs().max()) for a, b in zip(lhs, rhs, strict=True))


@pytest.mark.parametrize("kind", ARCH_KINDS_BY_REPRESENTATION["graph"])
def test_the_doubled_net_serves_its_parents_outputs_on_the_golden_positions(kind: str, derived: Any) -> None:
    net, arch = parent(kind, hidden=16, layers=2)
    wide = widen_net(net, arch, seed=7)
    assert widen_arch(arch).hidden == 32 and type(wide) is type(net)
    assert wide.state_dict()["representation.input_proj.weight"].shape[0] == 32
    positions = golden_positions(512)[::40]
    gaps = []
    for i in range(len(positions)):
        b = golden_batch(positions[i:i + 1], sample=1, seed=i)
        gaps += [max_gap(heads(net, b), heads(wide, b)), max_gap(single(net, b), single(wide, b))]
    b = golden_batch(positions, sample=24, seed=99)
    gaps.append(max_gap(heads(net, b), heads(wide, b)))
    derived(f"widen.{kind}.max_gap", max(gaps))
    assert max(gaps) <= TOL


def test_the_shape_of_record_doubles_within_the_pin() -> None:
    """4 layers of 128 to 256: policy, aux, value and bins within the pin on golden positions, batched and single."""
    net, arch = parent("GnnArchV2SoftPolicy", hidden=128, layers=4)
    wide = widen_net(net, arch, seed=20261014)
    positions = golden_positions(4)
    b = golden_batch(positions, sample=8, seed=5)
    assert max_gap(heads(net, b), heads(wide, b)) <= TOL
    one = golden_batch(positions[:1], sample=1, seed=0)
    assert max_gap(single(net, one), single(wide, one)) <= TOL


def test_copies_read_alike_and_their_readers_split_unevenly_into_the_parents_weight() -> None:
    net, _ = parent("GnnArchV2SoftPolicy", hidden=8, layers=2)
    sd, h = net.state_dict(), 8
    wide = widen_state(sd, hidden=h, seed=3)
    w = wide["representation.convs.1.nn.0.weight"]
    assert torch.equal(w[:h], w[h:]), "a unit's two copies read the same inputs"
    pair_sum = w[:, :h].double() + w[:, h:].double()
    assert torch.allclose(pair_sum[:h], sd["representation.convs.1.nn.0.weight"].double(), atol=1e-6)
    assert not torch.allclose(w[:, :h], w[:, h:]), "the split noise makes the copies' outgoing weights differ"
    jk = wide["policy_head.mlp.0.weight"]
    assert jk.shape == (sd["policy_head.mlp.0.weight"].shape[0], 2 * 2 * h), "every JK block doubled"
    assert torch.allclose(jk[:, :h].double() + jk[:, h:2 * h].double(), sd["policy_head.mlp.0.weight"][:, :h].double(),
                          atol=1e-6)
    assert torch.equal(wide["representation.norms.0.weight"], sd["representation.norms.0.weight"].repeat(2))
    assert torch.equal(wide["value_head.fc2_bins.weight"], sd["value_head.fc2_bins.weight"])
    assert all(torch.equal(wide[k], widen_state(sd, hidden=h, seed=3)[k]) for k in wide), "the seed fixes the noise"


@pytest.mark.parametrize("noise", [0.0, 0.1])
def test_only_the_split_noise_lets_the_copies_train_apart(noise: float) -> None:
    """With an even split the copies get the same gradient forever; the noise separates them at the first step."""
    net, arch = parent("GnnArchV2SoftPolicy", hidden=8, layers=2)
    wide = widen_net(net, arch, seed=11, noise=noise).train()
    b = golden_batch(golden_positions(3), sample=3, seed=1)
    policy, value, bins, aux, _future = wide.forward_batch_heads(b.x, b.edge_index, b.edge_attr, b.legal_node_gather,
                                                        stone_mask_from_batch(b), b.node_offsets)
    (policy.square().mean() + value.square().mean() + bins.square().mean() + aux.square().mean()).backward()
    with torch.no_grad():
        for p in wide.parameters():
            p.sub_(0.1 * p.grad)
    w = wide.state_dict()["representation.convs.0.nn.0.weight"]
    apart = float((w[:8] - w[8:]).abs().max())
    assert apart > 1e-4 if noise else apart <= 1e-7


def test_a_tensor_without_a_rule_and_a_sign_flipping_noise_are_refused() -> None:
    sd = dict(parent("GnnArchV2", hidden=8, layers=1)[0].state_dict())
    with pytest.raises(WidenError, match="noise"):
        widen_state(sd, hidden=8, seed=0, noise=0.5)
    with pytest.raises(WidenError, match="trunk's 6"):
        widen_state(sd, hidden=6, seed=0)
    sd["representation.gate.weight"] = torch.zeros(8, 8)
    with pytest.raises(WidenError, match="representation.gate.weight"):
        widen_state(sd, hidden=8, seed=0)
