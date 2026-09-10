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
from mantis.encoding.resolvers import (
    AmbiguousGraphMarkerError,
    detect_encoding_from_state_dict,
)

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


# 1. Registry stable-instance identity


@pytest.mark.parametrize("name", _REGISTERED)
def test_lookup_stable_instance(name: str) -> None:
    spec = lookup(name)
    assert lookup(name) is spec, f"{name}: lookup() returned a non-stable instance"


# 2. Rust↔Python helper parity


@pytest.mark.parametrize("name", _REGISTERED)
def test_helper_parity_shim_vs_engine(name: str) -> None:
    py = lookup(name)
    rs = _engine.RegistrySpec.from_registry(name)
    for field in _HELPER_FIELDS:
        assert getattr(py, field) == getattr(rs, field), (
            f"{name}.{field}: shim {getattr(py, field)!r} != engine {getattr(rs, field)!r}"
        )
    assert tuple(py.kept_plane_indices) == tuple(rs.kept_plane_indices)


# 3. Unified detector — the grid shape fallback is RETIRED (R346(f))


def test_the_grid_shape_fallback_is_gone_and_an_unstamped_grid_shape_refuses() -> None:
    """The shape fallback resolved a DENSE state dict by `(in_channels, n_actions)`. Both the
    encodings it discriminated between and the arch that produced those keys are deleted, so
    the branch is gone: a state dict carrying neither a stamp nor the graph marker now REFUSES
    under `strict` and answers `None` otherwise, whatever its conv widths say.

    This is the inverse of the rows it replaces. They asserted that a shape RESOLVED; this
    asserts that it no longer can — which is what stops a future reader re-deriving an
    encoding from bytes that no longer determine one (LAW-11)."""
    dense_shaped = {
        "trunk.input_conv.weight": _FakeTensor(64, 8, 3, 3),
        "policy_fc.weight": _FakeTensor(362, 64),
    }
    assert detect_encoding_from_state_dict(dense_shaped, "model.pt", strict=False) is None
    with pytest.raises(ValueError, match="neither an encoding stamp nor the graph marker"):
        detect_encoding_from_state_dict(dense_shaped, "model.pt", strict=True)


# ── 3b. Marker/stamp beats shape (and filename is NOT a signal — the KILL) ───


def test_detect_graph_marker_REFUSES_once_more_than_one_graph_encoding_is_registered() -> None:
    """The marker says GRAPH; it has never said WHICH graph (R328(c)).

    These two rows used to assert `gnn_axis_v1` outright, and they were right for exactly as
    long as one graph row existed. R328(b) registered `gnn_axis_r8`, and the two differ ONLY in
    a geometry no checkpoint stamp records — so the branch now REFUSES by name rather than
    resolving an unstamped checkpoint to whichever graph row was written first.
    """
    graph = [s for s in all_specs() if s.representation == "graph"]
    assert len(graph) > 1, (
        "this row's subject is the AMBIGUITY; with one graph encoding registered the marker "
        "is determinate again and `test_detect_graph_marker_resolves_when_unique` is the row "
        "that applies"
    )
    for strict in (False, True):
        with pytest.raises(AmbiguousGraphMarkerError, match="never said WHICH graph"):
            detect_encoding_from_state_dict(_gnn_state(), "model.pt", strict=strict)


def test_detect_graph_marker_resolves_when_unique(monkeypatch) -> None:
    """The positive control: pruned back to ONE graph row, the marker resolves again.

    Without this, the row above would pass on a branch that raised unconditionally."""
    from mantis.encoding import resolvers

    only = [s for s in all_specs() if s.name == "gnn_axis_v1"]
    monkeypatch.setattr(resolvers, "_graph_specs", lambda: only)
    spec = detect_encoding_from_state_dict(_gnn_state(), "model.pt", strict=False)
    assert spec is not None and spec.name == "gnn_axis_v1"
    assert spec.representation == "graph"


def test_detect_stamp_beats_the_graph_marker() -> None:
    """A stamp wins over every other signal. The shape it used to out-rank is retired
    (R346(f)); the marker is what is left to out-rank, and the stamp names WHICH graph row
    the marker cannot."""
    state = _gnn_state()
    state["metadata"] = {"encoding_name": "gnn_axis_r8"}
    spec = detect_encoding_from_state_dict(state, "model.pt", strict=False)
    assert spec is not None and spec.name == "gnn_axis_r8"


def test_detect_filename_is_not_a_signal() -> None:
    """The filename says gnn_axis_r8; the STAMP says gnn_axis_v1 — the KILL means the
    filename NEVER overrides the stamp/marker resolution."""
    state = _gnn_state()
    state["metadata"] = {"encoding_name": "gnn_axis_v1"}
    spec = detect_encoding_from_state_dict(state, "model_gnn_axis_r8.pt", strict=False)
    assert spec is not None and spec.name == "gnn_axis_v1"


# 3c. The miss arms — strict raises, lenient answers None


def test_detect_lenient_no_marker_returns_none() -> None:
    assert detect_encoding_from_state_dict({}, "model.pt", strict=False) is None


def test_detect_strict_no_marker_raises() -> None:
    with pytest.raises(ValueError):
        detect_encoding_from_state_dict({}, "model.pt", strict=True)
