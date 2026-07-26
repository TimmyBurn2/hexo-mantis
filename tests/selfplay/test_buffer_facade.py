"""Suite E — the replay-buffer facade (`mantis.selfplay.buffers`).

>300 justify: one facade, one per-buffer-kind rule. The kind resolution (E-01/E-02), the
zero-copy identity verdict (E-03), the wrong-kind guard (E-04), the passthrough surface
and the dense/graph split of `outcome_in_range_count` (E-05) and the cross-magic
rejection (E-06) all bind the SAME two-class module and share the fill recipe and the
recording stub; splitting them would duplicate the fixtures that carry the captured
numbers.

IMPL-written (non-⊕) per DESIGN §b. Every number asserted against old-side capture comes
from the dispatcher's `#C3e` probe as restated in PREREG §3 (DISPATCHER CORRECTION 2):
the dense fill recipe (capacity 64, encoding v6, 40 rows, outcomes
10×−0.5 / 8×−0.7 / 10×+1.0 / 12×−1.0), the draw band `[-0.75, -0.45)` and the resulting
`outcome_in_range_count == 18` → `draw_target_fraction == 0.45`.

The per-buffer-kind rule this suite exists to hold (PREREG §3, superseding E-05):

  * dense `ReplayBuffer` — the getter is PRESENT on both sides; the real fraction must
    come out. NaN here is a FAIL (a phantom input into a registered event field).
  * graph `HexgBuffer`  — the getter is genuinely ABSENT on both sides; the missing
    attribute must PROPAGATE so the caller's NaN fallback stays reachable. A fabricated
    number here is a FAIL (an undeclared behaviour change).

E-07 (`test_pool_pushes_through_the_facade`) is NOT in this file: it asserts on
`WorkerPool` construction, and `pool.py` is a later slice. Recorded as owed in
`wp/WPSP/IMPL_NOTES_S2.md` so it is not silently dropped.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from mantis._engine import HexgBuffer, ReplayBuffer
from mantis.encoding import lookup
from mantis.model import RepresentationMismatch
from mantis.selfplay.buffers import BufferKind, BufferKindMismatch, ReplayFacade

# #C3e fill recipe + probe (PREREG §3 CORRECTION 2 table).
_FILL_CAPACITY = 64
_FILL_ROWS = 40
_DRAW_BAND = (-0.75, -0.45)
_EXPECTED_IN_BAND = 18
_EXPECTED_DRAW_FRACTION = 0.45

_GRID_SPEC = lookup("v6")
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
        self.dense_calls: list[tuple[tuple, dict]] = []
        self.graph_calls: list[tuple[tuple, dict]] = []
        self.other: list[tuple[str, tuple]] = []
        self.size = 7
        self.capacity = 11

    def push_many(self, *args, **kwargs) -> None:
        self.dense_calls.append((args, kwargs))

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


class _NoOutcomeCountBuffer:
    """Stub WITHOUT `outcome_in_range_count` — the pre-dating-wheel arm the capture used
    to exercise the fallback branch on the dense path."""

    def __init__(self, size: int = _FILL_ROWS, capacity: int = _FILL_CAPACITY) -> None:
        self.size = size
        self.capacity = capacity


def _filled_dense_buffer() -> ReplayBuffer:
    """Rebuild the #C3e deterministic fill on a real engine `ReplayBuffer`."""
    n = _FILL_ROWS
    buf = ReplayBuffer(capacity=_FILL_CAPACITY, encoding="v6")
    outcomes = np.empty(n, dtype=np.float32)
    outcomes[0:10] = -0.5
    outcomes[10:18] = -0.7
    outcomes[18:28] = 1.0
    outcomes[28:40] = -1.0
    pols = np.zeros((n, 362), dtype=np.float32)
    pols[:, 0] = 1.0
    buf.push_many(
        np.zeros((n, 8, 19, 19), dtype=np.float16),
        np.zeros((n, 6, 19, 19), dtype=np.float16),
        pols,
        outcomes,
        np.zeros((n, 361), dtype=np.uint8),
        np.zeros((n, 361), dtype=np.uint8),
        np.full(n, 25, dtype=np.uint16),
        np.ones(n, dtype=np.uint8),
        np.arange(n, dtype=np.uint16),
        value_target_valid=np.ones(n, dtype=np.uint8),
    )
    return buf


def _dense_push_arrays(n: int = 3) -> dict[str, np.ndarray]:
    """Distinct array objects, one per `push_dense_many` parameter (identity probes)."""
    return {
        "states": np.zeros((n, 8, 19, 19), dtype=np.float16),
        "chain_planes": np.zeros((n, 6, 19, 19), dtype=np.float16),
        "policies": np.zeros((n, 362), dtype=np.float32),
        "outcomes": np.zeros(n, dtype=np.float32),
        "ownership": np.zeros((n, 361), dtype=np.uint8),
        "winning_line": np.zeros((n, 361), dtype=np.uint8),
        "game_lengths": np.zeros(n, dtype=np.uint16),
        "is_full_search": np.ones(n, dtype=np.uint8),
        "position_indices": np.arange(n, dtype=np.uint16),
        "value_target_valid": np.ones(n, dtype=np.uint8),
    }


# ── E-01 — closed match, no wildcard arm ─────────────────────────────────────────
def test_kind_from_spec_closed_match() -> None:
    assert BufferKind.from_spec(_GRID_SPEC) is BufferKind.GRID
    assert BufferKind.from_spec(_GRAPH_SPEC) is BufferKind.GRAPH


@pytest.mark.parametrize("rep", ["dense", "GRID", "", None, "hex", "canvas"])
def test_kind_from_spec_unknown_representation_raises(rep) -> None:
    """LAW-11: an unknown/absent representation is an ERROR, never a silent dense
    default. `"GRID"` and `"dense"` are in the list on purpose — near-misses must not be
    coerced."""
    with pytest.raises(RepresentationMismatch):
        BufferKind.from_spec(_FakeSpec(representation=rep))


def test_kind_from_spec_no_attribute_raises() -> None:
    with pytest.raises(RepresentationMismatch):
        BufferKind.from_spec(object())


# ── E-02 — kind/raw cross-check at construction ──────────────────────────────────
def test_kind_raw_type_crosscheck_graph_buffer_under_grid_spec() -> None:
    raw = HexgBuffer(capacity=8, encoding="gnn_axis_v1")
    with pytest.raises(BufferKindMismatch):
        ReplayFacade(_GRID_SPEC, raw)


def test_kind_raw_type_crosscheck_dense_buffer_under_graph_spec() -> None:
    raw = ReplayBuffer(capacity=8, encoding="v6")
    with pytest.raises(BufferKindMismatch):
        ReplayFacade(_GRAPH_SPEC, raw)


def test_matched_kind_raw_pairs_construct() -> None:
    """LAW-07 clean twin: the guard must not reject the CORRECT pairing."""
    dense = ReplayFacade(_GRID_SPEC, ReplayBuffer(capacity=8, encoding="v6"))
    graph = ReplayFacade(_GRAPH_SPEC, HexgBuffer(capacity=8, encoding="gnn_axis_v1"))
    assert dense.kind is BufferKind.GRID
    assert graph.kind is BufferKind.GRAPH
    # The raw handle is held, not copied or re-wrapped.
    assert isinstance(dense.raw, ReplayBuffer)
    assert isinstance(graph.raw, HexgBuffer)


# ── E-03 — zero-copy passthrough (a VERDICT, not a claim) ────────────────────────
def test_zero_copy_passthrough_dense_arm() -> None:
    rec = _RecordingBuffer()
    facade = ReplayFacade(_GRID_SPEC, rec)
    arrays = _dense_push_arrays()
    facade.push_dense_many(**arrays)

    assert len(rec.dense_calls) == 1
    args, kwargs = rec.dense_calls[0]
    forwarded = list(args) + list(kwargs.values())
    # Every forwarded object is the caller's object — identity, not equality.
    for name, original in arrays.items():
        assert any(captured is original for captured in forwarded), (
            f"{name} was not forwarded by identity — the facade copied or re-wrapped it"
        )
    assert len(forwarded) == len(arrays)


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


# ── E-04 — wrong-kind push dies with a named error ───────────────────────────────
def test_wrong_kind_method_raises_graph_push_on_grid_facade() -> None:
    facade = ReplayFacade(_GRID_SPEC, _RecordingBuffer())
    with pytest.raises(BufferKindMismatch):
        facade.push_graph_position([(0, 0, 1)], [(0, 1, 1.0)], 1, 2, 0, True, 0.0, True, 1)


def test_wrong_kind_method_raises_dense_push_on_graph_facade() -> None:
    rec = _RecordingBuffer()
    facade = ReplayFacade(_GRAPH_SPEC, rec)
    with pytest.raises(BufferKindMismatch):
        facade.push_dense_many(**_dense_push_arrays())
    # And nothing reached the raw handle — the guard fires BEFORE the forward.
    assert rec.dense_calls == []


# ── E-05 — passthrough surface + per-buffer-kind composition parity ──────────────
def test_passthrough_surface_forwards() -> None:
    rec = _RecordingBuffer()
    facade = ReplayFacade(_GRID_SPEC, rec)
    assert facade.size == 7
    assert facade.capacity == 11
    facade.resize(99)
    facade.save_to_path("buffer.hexb")
    assert facade.load_from_path("buffer.hexb") == 3
    facade.set_weight_schedule([1, 2], [0.5, 1.0], 1.0)
    assert rec.other == [
        ("resize", (99,)),
        ("save_to_path", ("buffer.hexb",)),
        ("load_from_path", ("buffer.hexb",)),
        ("set_weight_schedule", ([1, 2], [0.5, 1.0], 1.0)),
    ]


def test_dense_arm_returns_the_real_draw_fraction_not_nan() -> None:
    """E-05(ii): the getter is PRESENT on the dense buffer both sides. On the #C3e fill
    the band `[-0.75, -0.45)` holds 18 of 40 rows → 0.45. NaN here = FAIL."""
    facade = ReplayFacade(_GRID_SPEC, _filled_dense_buffer())
    assert facade.size == _FILL_ROWS
    assert facade.capacity == _FILL_CAPACITY

    lo, hi = _DRAW_BAND
    in_band = facade.outcome_in_range_count(lo, hi)
    assert in_band == _EXPECTED_IN_BAND
    fraction = in_band / facade.size
    assert fraction == pytest.approx(_EXPECTED_DRAW_FRACTION)
    assert not np.isnan(fraction)


@pytest.mark.parametrize(
    ("lo", "hi", "expected"),
    [
        (-0.75, -0.45, 18),
        (-0.55, -0.45, 10),
        (-0.6, -0.4, 10),
        (-1.05, -0.95, 12),
        (0.95, 1.05, 10),
        (-2.0, 2.0, 40),
        (0.0, 0.0, 0),
    ],
)
def test_dense_outcome_band_probe_matches_capture(lo, hi, expected) -> None:
    """The dispatcher's 7-point probe of the getter's own semantics, re-run through the
    facade so the passthrough is proven to change no number."""
    facade = ReplayFacade(_GRID_SPEC, _filled_dense_buffer())
    assert facade.outcome_in_range_count(lo, hi) == expected


def test_graph_arm_missing_getter_propagates() -> None:
    """E-05(iii): the graph buffer genuinely has no `outcome_in_range_count` on EITHER
    side. The facade must let the `AttributeError` out so the caller's NaN fallback stays
    reachable — fabricating a number here is an undeclared behaviour change."""
    facade = ReplayFacade(_GRAPH_SPEC, HexgBuffer(capacity=8, encoding="gnn_axis_v1"))
    assert not hasattr(facade.raw, "outcome_in_range_count")
    with pytest.raises(AttributeError):
        facade.outcome_in_range_count(*_DRAW_BAND)


def test_fallback_branch_is_reachable_from_an_attribute_less_stub() -> None:
    """E-05(iv): the fallback branch still exists and is exercisable on the DENSE arm
    too, via a buffer stub that pre-dates the getter — the shape the caller's
    `except AttributeError` NaN arm is written for."""
    facade = ReplayFacade(_GRID_SPEC, _NoOutcomeCountBuffer())
    assert facade.size == _FILL_ROWS
    try:
        facade.outcome_in_range_count(*_DRAW_BAND)
    except AttributeError:
        draw_target_fraction = float("nan")
    else:  # pragma: no cover — the stub has no getter by construction
        pytest.fail("the attribute-less stub must not answer outcome_in_range_count")
    assert np.isnan(draw_target_fraction)


# ── E-06 — cross-magic load is rejected loudly ───────────────────────────────────
def test_cross_magic_load_rejected_loud(tmp_path) -> None:
    """The HEXB/HEXG magic check is the engine's byte-level gate; the facade must
    propagate it unswallowed at the seam."""
    hexb_path = tmp_path / "dense.hexb"
    hexg_path = tmp_path / "graph.hexg"

    dense = ReplayFacade(_GRID_SPEC, _filled_dense_buffer())
    dense.save_to_path(str(hexb_path))

    graph_raw = HexgBuffer(capacity=8, encoding="gnn_axis_v1")
    graph_raw.push_graph_position(
        [(0, 0, 1), (1, 0, -1)], [(0, 1, 1.0)], 1, 2, 0, True, 0.0, True, 1,
    )
    graph = ReplayFacade(_GRAPH_SPEC, graph_raw)
    graph.save_to_path(str(hexg_path))

    with pytest.raises(Exception) as dense_err:
        dense.load_from_path(str(hexg_path))
    assert dense_err.value is not None

    with pytest.raises(Exception) as graph_err:
        graph.load_from_path(str(hexb_path))
    assert graph_err.value is not None

    # LAW-07 clean twin: the SAME call on the matching format succeeds.
    assert dense.load_from_path(str(hexb_path)) >= 0
    assert graph.load_from_path(str(hexg_path)) >= 0
