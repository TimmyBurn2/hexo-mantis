"""`GnnArchV2` reaches its own net and its own rehydration — the two hazards filed against it.

Both were found by the WP-AXIS2 design review, filed, and never closed until V2 landed. Both are
SILENT failures: nothing raises, the wrong net is built, and every downstream number is a
measurement of V1 wearing a V2 label.

  * **The `build_net` isinstance twin.** A `GnnArchV2` that subclassed `GnnArch` would satisfy
    `isinstance(arch, GnnArch)` and build V1's net. Two independent guards below: V2 is not a
    subclass, AND its dispatch branch precedes V1's, so either one alone would hold.
  * **The rehydration discriminator.** `_arch_from_dict` dispatched on `representation` alone,
    and V2 declares `representation="graph"` because it consumes the same wire — so a V2 stamp
    rehydrated as `GnnArch` and the loader rebuilt V1 for a V2 checkpoint. LAW-12 is about
    exactly this: one loader, and a stamp that means what it says.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

from mantis.model import GnnArch, GnnArchV2, GnnNet, GnnNetV2, build_net
from mantis.model.arch import CnnArch, RepresentationMismatch
from mantis.train.checkpoints import _arch_from_dict, _arch_to_dict

_V2 = GnnArchV2(in_dim=11, edge_dim=5, hidden=8, num_layers=2, policy_hidden=8, value_hidden=8)
_V1 = GnnArch(in_dim=11, edge_dim=5, hidden=8, num_layers=2, policy_hidden=8, value_hidden=8)


def test_GnnArchV2_is_a_SIBLING_of_GnnArch_and_not_a_subclass() -> None:
    """The first guard, and the one that makes the second a second line rather than the only
    line. `issubclass` here is not pedantry: it is the exact predicate `build_net` runs."""
    assert not issubclass(GnnArchV2, GnnArch)
    assert not issubclass(GnnArch, GnnArchV2)
    assert not isinstance(_V2, GnnArch), (
        "a GnnArchV2 that satisfies isinstance(arch, GnnArch) reaches build_net's V1 branch and "
        "is built as V1, silently"
    )


def test_build_net_gives_each_graph_arch_its_OWN_net() -> None:
    """The behavioural half. `type(...) is` and not `isinstance`, deliberately — `GnnNetV2`
    subclasses `GnnNet`, so an isinstance assertion here would pass on the wrong net."""
    assert type(build_net(_V2)) is GnnNetV2
    assert type(build_net(_V1)) is GnnNet


def test_the_V2_branch_PRECEDES_the_V1_branch_in_the_dispatch() -> None:
    """The second guard, checked structurally rather than by comment. If V2 were ever made a
    subclass, branch order is what would still route it correctly — so the order is a property
    worth pinning, not a formatting accident."""
    import ast
    import inspect

    import mantis.model.build as build_module

    source = inspect.getsource(build_module.build_net)
    names = [
        node.args[1].id
        for node in ast.walk(ast.parse(source.strip()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "isinstance"
        and len(node.args) == 2
        and isinstance(node.args[1], ast.Name)
    ]
    assert "GnnArchV2" in names and "GnnArch" in names
    assert names.index("GnnArchV2") < names.index("GnnArch"), (
        f"build_net tests {names} in that order; GnnArch before GnnArchV2 is the ordering under "
        "which a subclassed V2 builds V1's net"
    )


def test_the_declared_arch_is_the_HANDLE_the_built_net_carries() -> None:
    """PK1 — the witness stamp as a CONSTRUCTION PRECONDITION. `is`, not `==`: a copy would be a
    second authority for the run's identity, which is what `build_net`'s own comment says."""
    net = build_net(_V2)
    assert net.arch is _V2
    assert "arch" not in net.state_dict(), (
        "the arch handle leaked into the state dict; LAW-12's checkpoint key set must not move"
    )


@pytest.mark.parametrize("arch", [_V1, _V2, CnnArch(board_size=19, in_channels=8)])
def test_every_arch_ROUND_TRIPS_through_the_checkpoint_serializer(arch) -> None:
    """LAW-12's core claim for the widened union: what goes in comes back as itself."""
    assert _arch_from_dict(_arch_to_dict(arch)) == arch


def test_a_V2_STAMP_does_NOT_rehydrate_as_V1() -> None:
    """The filed hazard, driven. Before the discriminator landed this returned a `GnnArch` whose
    fields all matched, so nothing downstream could tell — the loader then built V1's net for a
    V2 checkpoint and every reading taken from it was mislabelled."""
    payload = _arch_to_dict(_V2)
    assert payload["representation"] == "graph", (
        "the premise of the hazard: V2 shares V1's representation because it shares the wire"
    )
    assert payload["arch_kind"] == "GnnArchV2"
    rehydrated = _arch_from_dict(payload)
    assert type(rehydrated) is GnnArchV2
    assert type(build_net(rehydrated)) is GnnNetV2


def test_the_REPRESENTATION_alone_no_longer_separates_the_two_graph_arches() -> None:
    """The negative control that keeps the test above honest: it must be the discriminator doing
    the work, not some incidental difference in the serialized fields."""
    v1_fields = dataclasses.asdict(_V1)
    v2_fields = dataclasses.asdict(_V2)
    assert v1_fields == v2_fields, (
        "the two arches serialize to identical FIELD dicts, which is why `arch_kind` had to be "
        "added — if this ever stops being true, say so here rather than letting the difference "
        "quietly become the discriminator"
    )


def test_a_LEGACY_stamp_with_no_arch_kind_rehydrates_as_V1() -> None:
    """A stamp written before the discriminator existed IS a V1 stamp — at the time it was
    written, no other graph arch existed. That is a fact about history, which is what makes it a
    sound fallback rather than a default."""
    legacy = dataclasses.asdict(_V1)
    assert "arch_kind" not in legacy
    assert type(_arch_from_dict(legacy)) is GnnArch


def test_an_UNKNOWN_arch_kind_is_REFUSED_rather_than_approximated() -> None:
    """A checkpoint from a newer build must not be rebuilt as the nearest thing that fits."""
    payload = _arch_to_dict(_V2) | {"arch_kind": "GnnArchNext"}
    with pytest.raises(RepresentationMismatch, match="REFUSED rather than rebuilt"):
        _arch_from_dict(payload)


def test_a_LEGACY_dict_that_does_NOT_FIT_its_arch_is_refused() -> None:
    """The other half of the fallback: `representation` names the class, and if the fields do
    not fit that class the answer is a named refusal, never a coercion (LAW-11's shape)."""
    legacy = dataclasses.asdict(_V1) | {"a_field_that_never_existed": 1}
    with pytest.raises(RepresentationMismatch, match="never coerced"):
        _arch_from_dict(legacy)


def test_a_stamp_with_NEITHER_discriminator_is_refused() -> None:
    with pytest.raises(RepresentationMismatch, match="expected 'grid' or 'graph'"):
        _arch_from_dict({"representation": "hypergraph"})


def test_the_two_graph_nets_have_DIFFERENT_state_dict_shapes_at_the_value_head() -> None:
    """W-ID1's structural half: V2 pools two statistics, so its value head is twice as wide.
    A V2 checkpoint loaded into a V1 net would therefore fail loudly on shape — which is worth
    knowing, because it means the rehydration hazard was silent only at the ARCH layer."""
    torch.manual_seed(0)
    v1_sd = build_net(_V1).state_dict()
    torch.manual_seed(0)
    v2_sd = build_net(_V2).state_dict()
    assert v1_sd["value_head.fc1.weight"].shape[1] * 2 == v2_sd["value_head.fc1.weight"].shape[1]


def test_the_REPRESENTATION_keys_are_IDENTICAL_so_BC_warmstart_survives() -> None:
    """The property that separates candidates A/C(i) from candidate B, asserted rather than
    asserted-in-prose. `load_representation_policy_from_bc` raises on ANY key mismatch under
    `representation.` / `policy_head.`, so a V2 whose trunk keys had moved would have silently
    cost the warmstart path its subject."""
    torch.manual_seed(0)
    v1_sd = build_net(_V1).state_dict()
    torch.manual_seed(0)
    v2_sd = build_net(_V2).state_dict()
    for prefix in ("representation.", "policy_head."):
        v1_keys = {k: v.shape for k, v in v1_sd.items() if k.startswith(prefix)}
        v2_keys = {k: v.shape for k, v in v2_sd.items() if k.startswith(prefix)}
        assert v1_keys == v2_keys, f"{prefix} keys/shapes differ between V1 and V2"
