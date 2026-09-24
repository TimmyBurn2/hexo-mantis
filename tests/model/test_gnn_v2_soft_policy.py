"""`GnnArchV2SoftPolicy` (R366(b)): a sibling kind whose net serves V2's outputs exactly and evaluates its second policy head only on the trainer's entry."""
from __future__ import annotations

import torch

from mantis.encoding import lookup
from mantis.eval.snapshot import load_model_snapshot, write_model_snapshot
from mantis.model import (
    ARCH_KINDS,
    GnnArch,
    GnnArchV2,
    GnnArchV2SoftPolicy,
    GnnNetV2,
    GnnNetV2SoftPolicy,
    build_net,
)

_SPEC = lookup("gnn_axis_v1")
_W = dict(in_dim=int(_SPEC.node_feat_dim), edge_dim=int(_SPEC.edge_feat_dim), hidden=8, num_layers=2,
          policy_hidden=8, value_hidden=8)


def _batch(seed: int = 7) -> tuple[torch.Tensor, ...]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(12, _W["in_dim"], generator=g)
    edge_index = torch.tensor([[0, 1, 2, 3, 6, 7, 8, 9], [1, 2, 3, 0, 7, 8, 9, 6]], dtype=torch.long)
    edge_attr = torch.randn(8, _W["edge_dim"], generator=g)
    legal_index = torch.tensor([1, 2, 3, 7, 8, 9], dtype=torch.long)
    stone_mask = torch.zeros(12, dtype=torch.bool)
    stone_mask[[0, 6]] = True
    node_offsets = torch.tensor([0, 6, 12], dtype=torch.long)
    return x, edge_index, edge_attr, legal_index, stone_mask, node_offsets


def test_the_kind_is_a_sibling_that_dispatches_to_its_own_net() -> None:
    assert not issubclass(GnnArchV2SoftPolicy, GnnArchV2) and not issubclass(GnnArchV2SoftPolicy, GnnArch)
    assert ARCH_KINDS["GnnArchV2SoftPolicy"] is GnnArchV2SoftPolicy
    net = build_net(GnnArchV2SoftPolicy(**_W))
    assert type(net) is GnnNetV2SoftPolicy and isinstance(net, GnnNetV2)
    assert "aux_policy_head.mlp.0.weight" in net.state_dict()


def test_served_outputs_are_V2s_exactly_and_the_aux_head_is_inert_at_deploy() -> None:
    """A V2 net and a soft-policy net with the SAME trunk and heads serve identical outputs."""
    torch.manual_seed(1)
    v2 = build_net(GnnArchV2(**_W)).eval()
    soft = build_net(GnnArchV2SoftPolicy(**_W)).eval()
    soft.load_state_dict(v2.state_dict(), strict=False)
    with torch.no_grad():
        a = v2.forward_batch(*_batch())
        b = soft.forward_batch(*_batch())
        p, v, bins, aux = soft.forward_batch_heads(*_batch())
    for lhs, rhs in zip(a, b, strict=True):
        assert torch.equal(lhs, rhs)
    assert torch.equal(p, a[0]) and torch.equal(v, a[1]) and torch.equal(bins, a[2])
    assert aux.shape == p.shape and not torch.equal(aux, p), "the aux head is its own function"
    x, ei, ea, li, sm, _ = _batch()
    legal_mask = torch.zeros(12, dtype=torch.bool)
    legal_mask[li] = True
    single_v2 = v2.forward_single(x[:6], ei[:, :4], ea[:4], legal_mask[:6], sm[:6])
    single_soft = soft.forward_single(x[:6], ei[:, :4], ea[:4], legal_mask[:6], sm[:6])
    for lhs, rhs in zip(single_v2, single_soft, strict=True):
        assert torch.equal(lhs, rhs)


def test_the_kind_round_trips_through_the_eval_snapshot(tmp_path) -> None:
    net = build_net(GnnArchV2SoftPolicy(**_W)).eval()
    write_model_snapshot(net, tmp_path / "s.pt")
    back = load_model_snapshot(tmp_path / "s.pt")
    assert type(back) is GnnNetV2SoftPolicy and type(back.arch) is GnnArchV2SoftPolicy
    with torch.no_grad():
        assert torch.equal(back.forward_batch(*_batch())[0], net.forward_batch(*_batch())[0])
