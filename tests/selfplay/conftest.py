"""Shared golden loaders for the selfplay oracle suites.

Imports no `mantis.selfplay.*` — the suites do that, so this file stays import-clean while they
are RED. Every golden is dispatcher-captured old-side truth under `tests/fixtures/selfplay/`
(manifest-tracked, sha-pinned); nothing is synthesized.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from _wire_geometry import COLLATE_FIXTURE_ENCODING, geometry_kwargs, spec_for

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "selfplay"
COLLATE_DIR = FIXTURES / "collate"
DRAIN_DIR = FIXTURES / "drain"
POOL_DIR = FIXTURES / "pool"
INSTR_DIR = FIXTURES / "instrumentation"

# The 13 wire arrays + 3 scalars of the graph payload, in the old capture's field order.
PAYLOAD_ARRAY_FIELDS: tuple[str, ...] = (
    "node_feat", "node_coords", "edge_index", "edge_attr", "node_offsets",
    "edge_offsets", "legal_offsets", "legal_node_gather", "policy_dst_slot",
    "n_nodes_checksum", "n_stones", "window_center", "current_player",
)
PAYLOAD_SCALAR_FIELDS: tuple[str, ...] = ("contract_version", "builder_impl", "n_graphs")

# `empty_legal` is the old file's hand-built 1-stone/0-legal single-graph payload.
_PAYLOAD_NPZ = {
    "b6": COLLATE_DIR / "b6_payload.npz",
    "b1": COLLATE_DIR / "b1_payload.npz",
    "b0": COLLATE_DIR / "b0_payload.npz",
    "empty_legal": COLLATE_DIR / "empty_legal_payload.npz",
}
_COLLATED_NPZ = {
    "b6": COLLATE_DIR / "b6_collated.npz",
    "b1": COLLATE_DIR / "b1_collated.npz",
    # semantic="full" and semantic="off" produced byte-identical B=0 output; ONE golden serves both.
    "b0": COLLATE_DIR / "b0_collated.npz",
}


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


@pytest.fixture(scope="session")
def collate_expectations() -> dict[str, Any]:
    """Payload + collate metadata and the ADV expectation table."""
    return json.loads((COLLATE_DIR / "collate_expectations.json").read_text())


@pytest.fixture(scope="session")
def drain_goldens() -> dict[str, Any]:
    """The scripted drain/push golden, 5 variants."""
    return json.loads((DRAIN_DIR / "drain_goldens.json").read_text())


@pytest.fixture(scope="session")
def encoding_resolve_golden() -> dict[str, Any]:
    """`_resolve_encoding_for_pool` outcome per registered encoding."""
    return json.loads((POOL_DIR / "encoding_resolve.json").read_text())


@pytest.fixture(scope="session")
def runner_config_goldens() -> dict[str, Any]:
    """The SelfPlayRunnerConfig ctor-kwarg/attr golden."""
    return json.loads((POOL_DIR / "runner_config_goldens.json").read_text())


@pytest.fixture(scope="session")
def pure_function_battery() -> dict[str, Any]:
    """22 move histories × the instrumentation pure functions."""
    return json.loads((INSTR_DIR / "pure_function_battery.json").read_text())


@pytest.fixture(scope="session")
def _payload_bank() -> dict[str, dict[str, np.ndarray]]:
    return {name: _load_npz(path) for name, path in _PAYLOAD_NPZ.items()}


@pytest.fixture(scope="session")
def collated_golden() -> dict[str, dict[str, np.ndarray]]:
    """The captured OLD collate outputs, as numpy arrays keyed by stem then tensor name."""
    return {name: _load_npz(path) for name, path in _COLLATED_NPZ.items()}


@pytest.fixture(scope="session")
def hotpath_golden() -> dict[str, np.ndarray]:
    """The seeded logits, old `segment_softmax` output, old `stone_mask`."""
    return _load_npz(COLLATE_DIR / "b6_hotpath.npz")


@pytest.fixture
def payload_fields(
    _payload_bank: dict[str, dict[str, np.ndarray]],
    collate_expectations: dict[str, Any],
) -> Callable[[str], dict[str, Any]]:
    """Factory → a FRESH ctor-kwarg dict for `GraphWirePayload`.

    Arrays are copied per call: every ADV test mutates its payload in place, so a shared buffer
    would leak one test's corruption into the next.
    """
    def make(name: str) -> dict[str, Any]:
        arrays = _payload_bank[name]
        scalars = collate_expectations["payloads"][name]["scalars"]
        fields: dict[str, Any] = {f: int(scalars[f]) for f in PAYLOAD_SCALAR_FIELDS}
        fields.update({f: arrays[f].copy() for f in PAYLOAD_ARRAY_FIELDS})
        return fields

    return make


@pytest.fixture(scope="session")
def wire_geometry(_payload_bank: dict[str, dict[str, np.ndarray]]) -> dict[str, int]:
    """The four geometry kwargs `collate_graph_batch` REQUIRES, read off the registry.

    The row-to-capture association is DERIVED, not declared: the captured `node_feat` and
    `edge_attr` arrays must divide by the row's dims, so a re-capture at a row with different
    dims fails here rather than collating under the wrong geometry.
    """
    spec = spec_for(COLLATE_FIXTURE_ENCODING)
    arrays = _payload_bank["b6"]
    n_nodes = arrays["node_coords"].size // 2
    n_edges = arrays["edge_index"].size // 2
    assert arrays["node_feat"].size == n_nodes * spec.node_feat_dim, (
        f"the captured payload does not carry {COLLATE_FIXTURE_ENCODING}'s node_feat_dim "
        f"({spec.node_feat_dim}): {arrays['node_feat'].size} floats over {n_nodes} nodes"
    )
    assert arrays["edge_attr"].size == n_edges * spec.edge_feat_dim, (
        f"the captured payload does not carry {COLLATE_FIXTURE_ENCODING}'s edge_feat_dim "
        f"({spec.edge_feat_dim}): {arrays['edge_attr'].size} floats over {n_edges} edges"
    )
    return geometry_kwargs(COLLATE_FIXTURE_ENCODING)


@pytest.fixture(scope="session")
def collect_data_input() -> tuple[np.ndarray, ...]:
    """The scripted `collect_data()` 10-tuple the capture fed the old drain loop.

    Order is the old return order: feats, chain, pols, vals, plies, own, wl, ifs, pidx, vv.
    Row 1 is the ply-capped row (vv=0); row 2 is the quick-search row (ifs=0).
    """
    z = _load_npz(DRAIN_DIR / "collect_data_input.npz")
    return tuple(z[k] for k in
                 ("feats", "chain", "pols", "vals", "plies", "own", "wl", "ifs", "pidx", "vv"))


@pytest.fixture(scope="session")
def dense_pushed() -> dict[str, np.ndarray]:
    """Captured `push_many` positional/keyword arrays + the 4 recent-buffer pushes."""
    return _load_npz(DRAIN_DIR / "dense_pushed.npz")


@pytest.fixture(scope="session")
def graph_pushed() -> dict[str, np.ndarray]:
    """Captured `push_graph_position` arrays (3 rows, in order)."""
    return _load_npz(DRAIN_DIR / "graph_pushed.npz")


@pytest.fixture(scope="session")
def graph_rows_input() -> list[tuple[Any, ...]]:
    """The scripted `collect_graph_data()` rows, rebuilt from the capture recipe.

    Each row is opaque tuple data the pool forwards verbatim plus a TRAILING runner game id, the
    one field the drain reads and translates. The first two rows share game 100 and the third is
    game 101, so two positions of one game must land under ONE allocated buffer id — a fixture
    where every row had its own game would pass a per-ROW allocator. Rebuilt rather than loaded
    so the input identity objects are ours to assert on.
    """
    return [
        (np.arange(6, dtype=np.float32), np.arange(4, dtype=np.int64), 3, 0.5, 100),
        (np.arange(6, dtype=np.float32) + 10.0, np.arange(4, dtype=np.int64) + 1, 4, -1.0, 100),
        (np.arange(6, dtype=np.float32) + 20.0, np.arange(4, dtype=np.int64) + 2, 5, 0.0, 101),
    ]
