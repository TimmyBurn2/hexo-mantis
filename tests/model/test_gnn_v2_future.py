"""`GnnArchV2SoftPolicyFuture`: the soft-policy net's outputs served exactly, plus a quiet three-way head per legal node."""
from __future__ import annotations

import torch

from mantis.eval.snapshot import load_model_snapshot, write_model_snapshot
from mantis.model import (
    ARCH_KINDS,
    GnnArchV2SoftPolicy,
    GnnArchV2SoftPolicyFuture,
    GnnNetV2SoftPolicy,
    GnnNetV2SoftPolicyFuture,
    build_net,
)
from mantis.model.gnn_v2_future import FUTURE_CLASSES, FUTURE_HEAD, FUTURE_PRIOR_LOGITS
from test_gnn_v2_soft_policy import _W, _batch


def test_the_kind_is_a_sibling_that_builds_the_soft_policy_net_plus_the_head() -> None:
    assert not issubclass(GnnArchV2SoftPolicyFuture, GnnArchV2SoftPolicy)
    assert ARCH_KINDS["GnnArchV2SoftPolicyFuture"] is GnnArchV2SoftPolicyFuture
    net = build_net(GnnArchV2SoftPolicyFuture(**_W))
    assert type(net) is GnnNetV2SoftPolicyFuture and isinstance(net, GnnNetV2SoftPolicy)
    assert net.future_heads() == (FUTURE_HEAD,) and net.policy_heads() == ("policy_head", "aux_policy_head")
    assert build_net(GnnArchV2SoftPolicy(**_W)).future_heads() == ()


def test_served_outputs_are_the_soft_policy_nets_exactly_and_the_head_serves_nothing() -> None:
    """Same weights, same served outputs on both serving faces; only the trainer's entry evaluates the head."""
    torch.manual_seed(1)
    soft = build_net(GnnArchV2SoftPolicy(**_W)).eval()
    fut = build_net(GnnArchV2SoftPolicyFuture(**_W)).eval()
    missing = fut.load_state_dict(soft.state_dict(), strict=False).missing_keys
    assert sorted(missing) == sorted(k for k in fut.state_dict() if k.startswith(FUTURE_HEAD + "."))
    with torch.no_grad():
        for a, b in zip(soft.forward_batch(*_batch()), fut.forward_batch(*_batch()), strict=True):
            assert torch.equal(a, b)
        x, ei, ea, legal, stones, _ = _batch()
        one = slice(0, 6)
        mask = torch.zeros(6, dtype=torch.bool)
        mask[[1, 2, 3]] = True
        for a, b in zip(soft.forward_single(x[one], ei[:, :4], ea[:4], mask, stones[one]),
                        fut.forward_single(x[one], ei[:, :4], ea[:4], mask, stones[one]), strict=True):
            assert torch.equal(a, b)
        heads = fut.forward_batch_heads(*_batch())
    assert all(torch.equal(a, b) for a, b in zip(heads[:4], soft.forward_batch_heads(*_batch())[:4], strict=True))
    assert heads[4] is not None and heads[4].shape == (len(_batch()[3]), FUTURE_CLASSES)


def test_the_head_starts_quiet_at_its_prior() -> None:
    """Zero last layer: every legal cell starts at the prior's logits and the trunk gets no gradient from the head."""
    net = build_net(GnnArchV2SoftPolicyFuture(**_W))
    future = net.forward_batch_heads(*_batch())[4]
    assert torch.allclose(future, torch.tensor(FUTURE_PRIOR_LOGITS).expand_as(future))
    future.sum().backward()
    assert all(p.grad is None or not p.grad.any() for n, p in net.named_parameters() if n.startswith("representation."))


def test_the_kind_round_trips_through_a_snapshot(tmp_path) -> None:
    net = build_net(GnnArchV2SoftPolicyFuture(**_W))
    path = tmp_path / "snap.pt"
    write_model_snapshot(net, path)
    back = load_model_snapshot(path)
    assert type(back) is GnnNetV2SoftPolicyFuture
    assert all(torch.equal(a, b) for a, b in zip(net.state_dict().values(), back.state_dict().values(), strict=True))
