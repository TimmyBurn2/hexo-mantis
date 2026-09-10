"""Suite E — the replay-buffer facade (`mantis.selfplay.buffers`).

The kind resolution (E-01/E-02), the zero-copy identity verdict (E-03) and the passthrough
surface (E-05) all bind the SAME module and share the recording stub. The dense arms —
the `ReplayBuffer` half of the kind cross-check, the `push_dense_many` identity verdict,
the `#C3e` outcome-band capture and the HEXB/HEXG cross-magic rejection — went with the
dense path (R346(f)); `BufferKind` now has one member.

The per-buffer-kind rule this suite still holds: the graph `HexgBuffer` genuinely has no
`outcome_in_range_count` on EITHER side, so the missing attribute must PROPAGATE and keep
the caller's NaN fallback reachable. A fabricated number here is a FAIL (an undeclared
behaviour change).

E-07 (`test_pool_pushes_through_the_facade`) is NOT in this file: it asserts on
`WorkerPool` construction, and `pool.py` is a later slice. Recorded as owed in
`wp/WPSP/IMPL_NOTES_S2.md` so it is not silently dropped.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from mantis._engine import HexgBuffer
from mantis.encoding import lookup
from mantis.model import RepresentationMismatch
from mantis.selfplay.buffers import BufferKind, BufferKindMismatch, ReplayFacade
_DRAW_BAND = (-0.75, -0.45)
_GRAPH_SPEC = lookup("gnn_axis_v1")


@dataclass
class _FakeSpec:
    """A spec-shaped stub; only `representation` is read by the facade."""

    representation: str | None
    name: str = "fake_spec"


class _RecordingBuffer:
    """Records every forwarded call WITHOUT touching the arrays.

    Deliberately duck-typed (neither engine class), so the facade's mislabel guard —
    which rejects by the WRONG engine class rather than by an allowlist — accepts it.
    """

    def __init__(self) -> None:
        self.graph_calls: list[tuple[tuple, dict]] = []
        self.other: list[tuple[str, tuple]] = []
        self.size = 7
        self.capacity = 11

    def push_graph_position(self, *args, **kwargs) -> None:
        self.graph_calls.append((args, kwargs))

    def resize(self, new_capacity: int) -> None:
        self.other.append(("resize", (new_capacity,)))

    def save_to_path(self, path: str) -> None:
        self.other.append(("save_to_path", (path,)))

    def load_from_path(self, path: str) -> int:
        self.other.append(("load_from_path", (path,)))
        return 3

    def set_weight_schedule(self, thresholds, weights, default_weight) -> None:
        self.other.append(("set_weight_schedule", (thresholds, weights, default_weight)))


# ── E-01 — closed match, no wildcard arm ─────────────────────────────────────────
def test_kind_from_spec_closed_match() -> None:
    assert BufferKind.from_spec(_GRAPH_SPEC) is BufferKind.GRAPH


@pytest.mark.parametrize("rep", ["grid", "dense", "GRAPH", "", None, "hex", "canvas"])
def test_kind_from_spec_unknown_representation_raises(rep) -> None:
    """LAW-11: an unknown/absent representation is an ERROR, never a silent default.

    `"grid"` leads the list and is the sharpest member: it is the one value that USED to be
    answered, and answering it now would hand a graph buffer back for a dense declaration —
    the inverted dense-by-default arm. `"dense"` and the mis-cased `"GRAPH"` are near-misses,
    which must not be coerced either."""
    with pytest.raises(RepresentationMismatch):
        BufferKind.from_spec(_FakeSpec(representation=rep))


def test_kind_from_spec_no_attribute_raises() -> None:
    with pytest.raises(RepresentationMismatch):
        BufferKind.from_spec(object())


def test_matched_kind_raw_pairs_construct() -> None:
    """LAW-07 clean twin: the guard must not reject the CORRECT pairing."""
    graph = ReplayFacade(_GRAPH_SPEC, HexgBuffer(capacity=8, encoding="gnn_axis_v1", visit_capacity=128))
    assert graph.kind is BufferKind.GRAPH
    # The raw handle is held, not copied or re-wrapped.
    assert isinstance(graph.raw, HexgBuffer)


def test_zero_copy_passthrough_graph_arm() -> None:
    rec = _RecordingBuffer()
    facade = ReplayFacade(_GRAPH_SPEC, rec)
    record = (
        [(0, 0, 1)],
        [(0, 1, 0.5)],
        1,
        2,
        3,
        True,
        0.5,
        True,
        7,
    )
    facade.push_graph_position(*record, game_id=-1)

    assert len(rec.graph_calls) == 1
    args, kwargs = rec.graph_calls[0]
    assert kwargs == {"game_id": -1}
    assert len(args) == len(record)
    for i, original in enumerate(record):
        assert args[i] is original, f"graph record element {i} was not forwarded verbatim"


def test_facade_module_imports_no_numpy() -> None:
    """The zero-copy grep, made mechanical: the facade module performs no array
    operations at all, so it CANNOT copy. A `numpy` import appearing here is the first
    sign the veneer grew a body."""
    import ast
    import inspect

    import mantis.selfplay.buffers as buffers_mod

    assert not hasattr(buffers_mod, "np")
    tree = ast.parse(inspect.getsource(buffers_mod))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "numpy" not in imported, f"the facade imported numpy: {sorted(imported)}"
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "np" not in names, "the facade references `np` — it is no longer copy-free"


# ── E-05 — passthrough surface + per-buffer-kind composition parity ──────────────
def test_passthrough_surface_forwards() -> None:
    rec = _RecordingBuffer()
    facade = ReplayFacade(_GRAPH_SPEC, rec)
    assert facade.size == 7
    assert facade.capacity == 11
    facade.resize(99)
    facade.save_to_path("buffer.hexg")
    assert facade.load_from_path("buffer.hexg") == 3
    facade.set_weight_schedule([1, 2], [0.5, 1.0], 1.0)
    assert rec.other == [
        ("resize", (99,)),
        ("save_to_path", ("buffer.hexg",)),
        ("load_from_path", ("buffer.hexg",)),
        ("set_weight_schedule", ([1, 2], [0.5, 1.0], 1.0)),
    ]


def test_graph_arm_missing_getter_propagates() -> None:
    """E-05(iii): the graph buffer genuinely has no `outcome_in_range_count` on EITHER
    side. The facade must let the `AttributeError` out so the caller's NaN fallback stays
    reachable — fabricating a number here is an undeclared behaviour change."""
    facade = ReplayFacade(_GRAPH_SPEC, HexgBuffer(capacity=8, encoding="gnn_axis_v1", visit_capacity=128))
    assert not hasattr(facade.raw, "outcome_in_range_count")
    with pytest.raises(AttributeError):
        facade.outcome_in_range_count(*_DRAW_BAND)
