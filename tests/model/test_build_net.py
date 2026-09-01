"""O3 — model construction per representation.

`build_net(arch)` produces the exact state-dict KEY SET per registered encoding
(frozen old-side goldens, COPIED from capture #3), and the spec/config→arch adapter
enforces LAW-11 (no dense-by-default): an absent/unknown representation, a graph
value_head_type≠dist65, or a graph spec missing node/edge geometry all raise
`RepresentationMismatch`.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mantis.encoding import lookup
from mantis.model import (
    CnnArch,
    GnnArch,
    RepresentationMismatch,
    arch_from_spec_and_config,
    build_net,
)

_KEYS_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "value_probes" / "statedict_keys"


def _golden_keys(name: str) -> set[str]:
    return set((_KEYS_DIR / f"{name}.txt").read_text().split())


def _keyset(net) -> set[str]:
    return set(net.state_dict().keys())


def test_grid_keysets_match_golden() -> None:
    # scalar arm (default) → v6 / v6w25.
    for enc in ("v6", "v6w25"):
        arch = arch_from_spec_and_config(lookup(enc), {})
        assert isinstance(arch, CnnArch)
        assert _keyset(build_net(arch)) == _golden_keys(enc)


def test_v6_live2_ls_both_arms_match_golden() -> None:
    spec = lookup("v6_live2_ls")
    dist_arch = arch_from_spec_and_config(spec, {"value_head_type": "dist65"})
    assert _keyset(build_net(dist_arch)) == _golden_keys("v6_live2_ls")
    scalar_arch = arch_from_spec_and_config(spec, {})
    assert _keyset(build_net(scalar_arch)) == _golden_keys("v6_live2_ls_scalar")
    # dist65 = scalar ∪ {value_fc2_bins.{weight,bias}} exactly.
    assert _golden_keys("v6_live2_ls") - _golden_keys("v6_live2_ls_scalar") == {
        "value_fc2_bins.weight", "value_fc2_bins.bias",
    }


def test_graph_keyset_matches_golden() -> None:
    arch = arch_from_spec_and_config(lookup("gnn_axis_v1"), {})
    assert isinstance(arch, GnnArch)
    assert _keyset(build_net(arch)) == _golden_keys("gnn_axis_v1")


def test_no_killed_branch_keys_on_any_constructed_net() -> None:
    killed = ("cluster_pool.", "global_encoder.", "gpool_bias_branch.")
    for enc in ("v6", "v6w25", "v6_live2_ls", "gnn_axis_v1", "gnn_axis_r8"):
        arch = arch_from_spec_and_config(lookup(enc), {})
        for k in _keyset(build_net(arch)):
            assert not k.startswith(killed), f"{enc}: killed-branch key {k!r}"


# ── RepresentationMismatch / no-dense-default (LAW-11) ────────────────────────


def test_absent_representation_raises_no_grid_default() -> None:
    spec = SimpleNamespace(name="stub", board_size=19, n_planes=8)  # no `representation`
    with pytest.raises(RepresentationMismatch):
        arch_from_spec_and_config(spec, {})


def test_unknown_representation_raises() -> None:
    spec = SimpleNamespace(name="weird", representation="hypergraph")
    with pytest.raises(RepresentationMismatch):
        arch_from_spec_and_config(spec, {})


def test_graph_non_dist65_value_head_raises() -> None:
    spec = lookup("gnn_axis_v1")
    with pytest.raises(RepresentationMismatch):
        arch_from_spec_and_config(spec, {"value_head_type": "scalar"})


def test_graph_missing_geometry_raises() -> None:
    spec = SimpleNamespace(name="g", representation="graph", node_feat_dim=None, edge_feat_dim=None)
    with pytest.raises(RepresentationMismatch):
        arch_from_spec_and_config(spec, {})


def test_build_net_rejects_non_arch() -> None:
    with pytest.raises(RepresentationMismatch):
        build_net(object())  # type: ignore[arg-type]


def test_grid_dist65_requires_65_bins() -> None:
    with pytest.raises(ValueError):
        build_net(CnnArch(board_size=19, in_channels=4, value_head_type="dist65", n_value_bins=51))
