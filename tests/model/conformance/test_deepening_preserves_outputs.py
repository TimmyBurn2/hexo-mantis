"""The depth growth serves its parent's outputs in fp64 on the golden r8 positions for every graph kind; its new layers pass x through and train off it."""
from __future__ import annotations

import dataclasses

import pytest
import torch

from mantis.model import ARCH_KINDS_BY_REPRESENTATION
from mantis.model.deepen import DeepenError, deepen_arch, deepen_net, deepen_state, deepen_tag
from mantis.selfplay.graph_collate import GraphBatch, stone_mask_from_batch

from test_widening_preserves_outputs import TOL, golden_batch, golden_positions, max_gap, parent


def fp64(b: GraphBatch) -> GraphBatch:
    return dataclasses.replace(b, x=b.x.double(), edge_attr=b.edge_attr.double())


@torch.no_grad()
def outputs(net: torch.nn.Module, b: GraphBatch) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Every head through `forward_batch_heads`, and through the deploy twin when `b` is one graph, plus the trunk's JK concat."""
    trunk: list[torch.Tensor] = []
    hook = net.representation.register_forward_hook(lambda _m, _i, out: trunk.append(out))
    try:
        mask = stone_mask_from_batch(b)
        got = [t for t in net.forward_batch_heads(b.x, b.edge_index, b.edge_attr, b.legal_node_gather, mask,
                                                  b.node_offsets) if t is not None]
        if b.node_offsets.shape[0] == 2:
            legal = torch.zeros(b.x.shape[0], dtype=torch.bool)
            legal[b.legal_node_gather] = True
            got += net.forward_single(b.x, b.edge_index, b.edge_attr, legal, mask)
    finally:
        hook.remove()
    return tuple(got), trunk[0]


def pin_gap(net: torch.nn.Module, deep: torch.nn.Module, b: GraphBatch, *, layers: int, hidden: int) -> float:
    """The worst fp64 |Δ| over every head, the parent's JK slots in the deeper trunk and each new slot against the last one it passes through."""
    (want, trunk), (got, deep_trunk) = outputs(net.double(), fp64(b)), outputs(deep.double(), fp64(b))
    old = layers * hidden
    new = deep_trunk[:, old:].reshape(deep_trunk.shape[0], -1, hidden)
    return max(max_gap(want, got), float((deep_trunk[:, :old] - trunk).abs().max()),
               float((new - trunk[:, None, old - hidden:]).abs().max()))


@pytest.mark.parametrize("added", [1, 2])
@pytest.mark.parametrize("kind", ARCH_KINDS_BY_REPRESENTATION["graph"])
def test_the_deeper_net_serves_its_parents_outputs_in_fp64_on_the_golden_positions(kind: str, added: int,
                                                                                   derived) -> None:
    net, arch = parent(kind, hidden=16, layers=2)
    deep = deepen_net(net, arch, layers=2 + added, seed=7)
    assert deepen_arch(arch, 2 + added).num_layers == 2 + added and type(deep) is type(net)
    assert len(deep.representation.convs) == 2 + added
    positions = golden_positions(512)[::40]
    gaps = [pin_gap(net, deep, golden_batch(positions[i:i + 1], sample=1, seed=i), layers=2, hidden=16)
            for i in range(len(positions))]
    gaps.append(pin_gap(net, deep, golden_batch(positions, sample=24, seed=99), layers=2, hidden=16))
    derived(f"deepen.{kind}.+{added}.max_gap", max(gaps))
    assert max(gaps) <= TOL


@pytest.mark.parametrize("layers", [5, 6])
def test_both_candidate_depths_of_the_shape_of_record_hold_the_pin(layers: int, derived) -> None:
    """4 layers of 256 to 5 and to 6, the future head and the soft policy carried: every output and the trunk within the pin."""
    net, arch = parent("GnnArchV2SoftPolicyFuture", hidden=256, layers=4)
    deep = deepen_net(net, arch, layers=layers, seed=20261023)
    positions = golden_positions(4)
    gaps = [pin_gap(net, deep, golden_batch(positions, sample=8, seed=5), layers=4, hidden=256),
            pin_gap(net, deep, golden_batch(positions[:1], sample=1, seed=0), layers=4, hidden=256)]
    derived(f"deepen.record.{layers}.max_gap", max(gaps))
    assert max(gaps) <= TOL


@pytest.mark.parametrize("kind", ARCH_KINDS_BY_REPRESENTATION["graph"])
def test_the_planted_break_random_init_new_layers_fails_the_pin(kind: str, derived) -> None:
    """The known-bad keeps the new layers and their head columns at their fresh draws: the pin must red on it."""
    net, arch = parent(kind, hidden=16, layers=2)
    bad = deepen_net(net, arch, layers=3, seed=7, known_bad=True)
    gap = pin_gap(net, bad, golden_batch(golden_positions(512)[::40], sample=24, seed=99), layers=2, hidden=16)
    derived(f"deepen.{kind}.known_bad_gap", gap)
    assert gap > TOL


def test_the_parents_tensors_are_carried_and_only_the_new_ones_follow_the_rule() -> None:
    net, arch = parent("GnnArchV2SoftPolicyFuture", hidden=8, layers=2)
    sd, h = net.state_dict(), 8
    deep = deepen_net(net, arch, layers=4, seed=3).state_dict()
    assert all(torch.equal(deep[k], sd[k]) for k in sd if k in deep and deep[k].shape == sd[k].shape)
    for i in (2, 3):
        assert not deep[f"representation.convs.{i}.nn.2.weight"].any() and not deep[f"representation.convs.{i}.nn.2.bias"].any()
        assert deep[f"representation.convs.{i}.nn.0.weight"].any(), "the first message linear keeps its fresh draw"
        assert torch.equal(deep[f"representation.norms.{i}.weight"], torch.ones(h))
    for head in ("policy_head", "aux_policy_head", "future_head"):
        w = deep[f"{head}.mlp.0.weight"]
        assert w.shape[1] == 4 * h and torch.equal(w[:, :2 * h], sd[f"{head}.mlp.0.weight"]) and not w[:, 2 * h:].any()
    fc1, parent_fc1 = deep["value_head.fc1.weight"], sd["value_head.fc1.weight"]
    assert torch.equal(fc1[:, :2 * h], parent_fc1[:, :2 * h]) and torch.equal(fc1[:, 4 * h:6 * h], parent_fc1[:, 2 * h:])
    assert not fc1[:, 2 * h:4 * h].any() and not fc1[:, 6 * h:].any(), "the mean and the max groups each gain zero columns"
    again = deepen_net(net, arch, layers=4, seed=3).state_dict()
    assert all(torch.equal(deep[k], again[k]) for k in deep), "the seed fixes the fresh draws"
    assert deepen_tag(arch.num_layers, 4) == "+deepen(2->4)"
    assert deepen_tag(arch.num_layers, 4, known_bad=True) == "+deepen-known-bad(2->4)"


def test_the_new_layer_trains_off_the_identity_once_its_head_columns_move() -> None:
    """Step one moves only the zero columns (nothing reads the new slot yet); step two reaches the new layer's last linear."""
    net, arch = parent("GnnArchV2SoftPolicyFuture", hidden=8, layers=2)
    deep = deepen_net(net, arch, layers=3, seed=11).train()
    b = golden_batch(golden_positions(3), sample=3, seed=1)
    silent, columns = [], []
    for _ in range(2):
        deep.zero_grad()
        out = deep.forward_batch_heads(b.x, b.edge_index, b.edge_attr, b.legal_node_gather, stone_mask_from_batch(b),
                                       b.node_offsets)
        sum(t.square().mean() for t in out if t is not None).backward()
        with torch.no_grad():
            for p in deep.parameters():
                if p.grad is not None:
                    p.sub_(0.1 * p.grad)
        sd = deep.state_dict()
        silent.append(float(sd["representation.convs.2.nn.2.weight"].abs().max()))
        columns.append(float(sd["policy_head.mlp.0.weight"][:, 16:].abs().max()))
    assert silent[0] == 0.0 and columns[0] > 0.0
    assert silent[1] > 0.0


def test_a_depth_adding_nothing_a_tensor_without_a_rule_and_a_parent_of_another_depth_are_refused() -> None:
    net, arch = parent("GnnArchV2", hidden=8, layers=2)
    with pytest.raises(DeepenError, match="adds none"):
        deepen_arch(arch, 2)
    fresh = deepen_net(net, arch, layers=3, seed=0).state_dict()
    with pytest.raises(DeepenError, match="already has layer 1"):
        deepen_state(net.state_dict(), fresh, hidden=8, layers=1)
    with pytest.raises(DeepenError, match="no such tensor"):
        deepen_state({k: v for k, v in net.state_dict().items() if "convs.1." not in k}, fresh, hidden=8, layers=2)
    for key in ("representation.gate.weight", "representation.convs.2.gate.weight"):
        with pytest.raises(DeepenError, match=key.replace(".", r"\.")):
            deepen_state(net.state_dict(), {**fresh, key: torch.zeros(8, 8)}, hidden=8, layers=2)
    with pytest.raises(DeepenError, match="no place"):
        deepen_state({**net.state_dict(), "aux_policy_head.mlp.0.bias": torch.zeros(8)}, fresh, hidden=8, layers=2)
