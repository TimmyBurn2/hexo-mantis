"""Shared loaders for the WP-SP ⊕ oracle suites (tests/selfplay/).

Written by ORACLE-WRITE **before** any `mantis.selfplay` port code exists. This conftest
imports ONLY already-present layers (stdlib + numpy) and NEVER `mantis.selfplay.*` — so it
collects cleanly while the six ⊕ suites are RED at import. The suites import
`mantis.selfplay.*`, which does not exist until IMPL lands; that is the correct
oracle-first state (PREREG §3 global rule).

Every golden here is dispatcher-captured old-side truth promoted into
`tests/fixtures/selfplay/` (manifest-tracked, sha-pinned). Nothing is synthesized: if a
value is not in the capture it is not asserted (see wp/WPSP/ORACLE_NOTES.md §gaps).

Root conftest already installs the autouse `_reseed` fixture — this file does NOT re-seed
and does NOT touch sys.path or sys.modules (R5/LAW-17).
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

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

# Payload fixture stem → npz file. `empty_legal` is the old file's hand-built
# 1-stone/0-legal single-graph payload (A-14).
_PAYLOAD_NPZ = {
    "b6": COLLATE_DIR / "b6_payload.npz",
    "b1": COLLATE_DIR / "b1_payload.npz",
    "b0": COLLATE_DIR / "b0_payload.npz",
    "empty_legal": COLLATE_DIR / "empty_legal_payload.npz",
}
_COLLATED_NPZ = {
    "b6": COLLATE_DIR / "b6_collated.npz",
    "b1": COLLATE_DIR / "b1_collated.npz",
    # semantic="full" and semantic="off" produced byte-identical B=0 output old-side
    # (per-tensor sha-equal); ONE golden, both arms assert against it.
    "b0": COLLATE_DIR / "b0_collated.npz",
}


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


# ── captured goldens (session-scoped; read once) ──────────────────────────────────────
@pytest.fixture(scope="session")
def collate_expectations() -> dict[str, Any]:
    """#C1/#C2/#C2b/#C2c — payload + collate metadata and the ADV expectation table."""
    return json.loads((COLLATE_DIR / "collate_expectations.json").read_text())


@pytest.fixture(scope="session")
def drain_goldens() -> dict[str, Any]:
    """#C3b — the scripted drain/push golden, 5 variants."""
    return json.loads((DRAIN_DIR / "drain_goldens.json").read_text())


@pytest.fixture(scope="session")
def encoding_resolve_golden() -> dict[str, Any]:
    """#C3a — `_resolve_encoding_for_pool` outcome per registered encoding."""
    return json.loads((POOL_DIR / "encoding_resolve.json").read_text())


@pytest.fixture(scope="session")
def runner_config_goldens() -> dict[str, Any]:
    """#C3d — the SelfPlayRunnerConfig ctor-kwarg/attr golden (KILLed fields removed)."""
    return json.loads((POOL_DIR / "runner_config_goldens.json").read_text())


@pytest.fixture(scope="session")
def pure_function_battery() -> dict[str, Any]:
    """#C3c — 22 move histories × the instrumentation pure functions."""
    return json.loads((INSTR_DIR / "pure_function_battery.json").read_text())


# ── payload / collated array banks ────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def _payload_bank() -> dict[str, dict[str, np.ndarray]]:
    return {name: _load_npz(path) for name, path in _PAYLOAD_NPZ.items()}


@pytest.fixture(scope="session")
def collated_golden() -> dict[str, dict[str, np.ndarray]]:
    """The captured OLD collate outputs, as numpy arrays keyed by stem then tensor name."""
    return {name: _load_npz(path) for name, path in _COLLATED_NPZ.items()}


@pytest.fixture(scope="session")
def hotpath_golden() -> dict[str, np.ndarray]:
    """#C1(i)-hot: the seeded logits, old `segment_softmax` output, old `stone_mask`."""
    return _load_npz(COLLATE_DIR / "b6_hotpath.npz")


@pytest.fixture
def payload_fields(
    _payload_bank: dict[str, dict[str, np.ndarray]],
    collate_expectations: dict[str, Any],
) -> Callable[[str], dict[str, Any]]:
    """Factory → a FRESH ctor-kwarg dict for `GraphWirePayload` (arrays copied per call).

    Copies matter: every ADV test mutates its payload in place, and a shared buffer would
    leak one test's corruption into the next (the old capture harness cloned for the same
    reason). Kept out of the test modules so the not-yet-written dataclass is imported
    only there — this conftest stays import-clean while the suites are RED.
    """
    def make(name: str) -> dict[str, Any]:
        arrays = _payload_bank[name]
        scalars = collate_expectations["payloads"][name]["scalars"]
        fields: dict[str, Any] = {f: int(scalars[f]) for f in PAYLOAD_SCALAR_FIELDS}
        fields.update({f: arrays[f].copy() for f in PAYLOAD_ARRAY_FIELDS})
        return fields

    return make


# ── drain fixture inputs (#C3b scripted `collect_data` rows + captured push args) ──────
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

    Recipe (`drain_goldens.json._constants.graph_rows_recipe`): 3 opaque tuples
    (f32[6], i64[4], int, float) — the pool forwards each verbatim and inspects nothing.
    Rebuilt (not loaded) so the *input* identity objects are ours: C-02 asserts the drain
    forwards these exact objects, so they must be constructible test-side.
    """
    return [
        (np.arange(6, dtype=np.float32), np.arange(4, dtype=np.int64), 3, 0.5),
        (np.arange(6, dtype=np.float32) + 10.0, np.arange(4, dtype=np.int64) + 1, 4, -1.0),
        (np.arange(6, dtype=np.float32) + 20.0, np.arange(4, dtype=np.int64) + 2, 5, 0.0),
    ]
