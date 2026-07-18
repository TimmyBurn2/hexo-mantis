"""Cross-encoding round-trip, parameterised over the registry.

Coverage per registered encoding:
  1. Registry stable-instance identity (`lookup(name) is lookup(name) is spec`).
  2. Rust↔Python helper parity over the derived shape accessors.
  3. The UNIFIED `detect_encoding_from_state_dict` (LOCKED #7): marker/stamp
     beats shape/filename; deterministic shape fallback; strict-raises.

DEFERRED (tracked-not-silent → WP9/WP10): the HexTacToeNet-forward leg and the
real-checkpoint torch-load leg need the unported `model`/`train` layers. The
detector cases here are torch-free (a fake tensor exposes only `.shape`/`.dim`).
"""
from __future__ import annotations

import pytest

from mantis import _engine
from mantis.encoding import all_specs, lookup
from mantis.encoding.resolvers import detect_encoding_from_state_dict

_REGISTERED: list[str] = sorted(s.name for s in all_specs())

# Derived shape accessors that the Python shim exposes over the compiled spec.
_HELPER_FIELDS: tuple[str, ...] = (
    "n_cells",
    "state_stride",
    "chain_stride",
    "aux_stride",
    "policy_stride",
    "n_source_planes",
)


class _FakeTensor:
    """Minimal torch-free stand-in — the detector reads only `.shape`/`.dim()`."""

    def __init__(self, *shape: int) -> None:
        self._shape = shape

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def dim(self) -> int:
        return len(self._shape)


def _grid_state(in_ch: int, n_actions: int | None) -> dict:
    state: dict = {"trunk.input_conv.weight": _FakeTensor(64, in_ch, 3, 3)}
    if n_actions is not None:
        state["policy_fc.weight"] = _FakeTensor(n_actions, 64)
    return state


def _gnn_state() -> dict:
    return {
        "representation.input_proj.weight": _FakeTensor(128, 11),
        "representation.input_proj.bias": _FakeTensor(128),
        "policy_head.mlp.0.weight": _FakeTensor(128, 512),
    }


# ── 1. Registry stable-instance identity ────────────────────────────────────


@pytest.mark.parametrize("name", _REGISTERED)
def test_lookup_stable_instance(name: str) -> None:
    spec = lookup(name)
    assert lookup(name) is spec, f"{name}: lookup() returned a non-stable instance"


# ── 2. Rust↔Python helper parity ────────────────────────────────────────────


@pytest.mark.parametrize("name", _REGISTERED)
def test_helper_parity_shim_vs_engine(name: str) -> None:
    py = lookup(name)
    rs = _engine.RegistrySpec.from_registry(name)
    for field in _HELPER_FIELDS:
        assert getattr(py, field) == getattr(rs, field), (
            f"{name}.{field}: shim {getattr(py, field)!r} != engine {getattr(rs, field)!r}"
        )
    assert tuple(py.kept_plane_indices) == tuple(rs.kept_plane_indices)


# ── 3. Unified detector — grid shape fallback ───────────────────────────────


def test_detect_v6_by_shape() -> None:
    spec = detect_encoding_from_state_dict(_grid_state(8, 362), "model.pt", strict=False)
    assert spec is not None and spec.name == "v6"


def test_detect_v6w25_by_n_actions() -> None:
    spec = detect_encoding_from_state_dict(_grid_state(8, 626), "model.pt", strict=False)
    assert spec is not None and spec.name == "v6w25"


def test_detect_v6_live2_ls_by_shape() -> None:
    spec = detect_encoding_from_state_dict(_grid_state(4, 362), "model.pt", strict=False)
    assert spec is not None and spec.name == "v6_live2_ls"


def test_detect_partial_conv_wrapped_key() -> None:
    state = _grid_state(8, 362)
    state["trunk.input_conv.conv.weight"] = state.pop("trunk.input_conv.weight")
    spec = detect_encoding_from_state_dict(state, "model.pt", strict=False)
    assert spec is not None and spec.name == "v6"


def test_detect_pma_policy_mlp_key_when_policy_fc_absent() -> None:
    state = {
        "trunk.input_conv.weight": _FakeTensor(64, 8, 3, 3),
        "cluster_pool.policy_mlp.2.weight": _FakeTensor(626, 64),
    }
    spec = detect_encoding_from_state_dict(state, "model.pt", strict=False)
    assert spec is not None and spec.name == "v6w25"


# ── 3b. Marker/stamp beats shape (and filename is NOT a signal — the KILL) ───


def test_detect_graph_marker_beats_shape() -> None:
    spec = detect_encoding_from_state_dict(_gnn_state(), "model.pt", strict=False)
    assert spec is not None and spec.name == "gnn_axis_v1"
    assert spec.representation == "graph"


def test_detect_graph_marker_wins_strict_too() -> None:
    spec = detect_encoding_from_state_dict(_gnn_state(), "model.pt", strict=True)
    assert spec is not None and spec.name == "gnn_axis_v1"


def test_detect_stamp_beats_shape() -> None:
    # An embedded encoding_name stamp wins over the grid shape (which says v6).
    state = _grid_state(8, 362)
    state["metadata"] = {"encoding_name": "v6w25"}
    spec = detect_encoding_from_state_dict(state, "model.pt", strict=False)
    assert spec is not None and spec.name == "v6w25"


def test_detect_filename_is_not_a_signal() -> None:
    """The filename says v6w25 but the shape (8, 362) is unambiguously v6 — the
    KILL means the filename NEVER overrides the marker/shape resolution."""
    spec = detect_encoding_from_state_dict(_grid_state(8, 362), "model_v6w25.pt", strict=False)
    assert spec is not None and spec.name == "v6"


# ── 3c. Deterministic fallback — ambiguity/miss strict-raises ────────────────


def test_detect_lenient_no_conv_returns_none() -> None:
    assert detect_encoding_from_state_dict({}, "model.pt", strict=False) is None


def test_detect_strict_no_conv_raises() -> None:
    with pytest.raises(ValueError):
        detect_encoding_from_state_dict({}, "model.pt", strict=True)


def test_detect_strict_unsupported_in_ch_raises() -> None:
    with pytest.raises(ValueError, match="in_channels"):
        detect_encoding_from_state_dict(_grid_state(99, 100), "model.pt", strict=True)


def test_detect_lenient_unsupported_in_ch_returns_none() -> None:
    assert detect_encoding_from_state_dict(_grid_state(99, 100), "model.pt", strict=False) is None


def test_detect_strict_ambiguous_shape_raises() -> None:
    """in_ch=8 with no n_actions probe matches BOTH v6 and v6w25 — with no
    filename tiebreak (the KILL), strict must RAISE on the ambiguity."""
    state = {"trunk.input_conv.weight": _FakeTensor(64, 8, 3, 3)}
    with pytest.raises(ValueError, match="ambiguous"):
        detect_encoding_from_state_dict(state, "model.pt", strict=True)


def test_detect_lenient_ambiguous_shape_returns_none() -> None:
    state = {"trunk.input_conv.weight": _FakeTensor(64, 8, 3, 3)}
    assert detect_encoding_from_state_dict(state, "model.pt", strict=False) is None
