"""Checks 15-16 in the Rust pack refuse exactly what the numpy semantic layer refused, arm for arm.

The numpy predicate is kept here, transcribed, as the oracle; it runs beside the pack over captured and built wires.
"""
from __future__ import annotations

import threading
import time
from collections import Counter
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from _wire_geometry import geometry_kwargs

from mantis import _engine
from mantis.selfplay.graph_collate import (
    GatherNotLegalNode,
    GraphContractError,
    GraphWirePayload,
    ScatterSlotCanonicalMismatch,
    collate_graph_batch,
    graph_wire_from_rust,
    reset_semantic_canary,
)

_POSITIONS = Path(__file__).resolve().parents[1] / "fixtures" / "graph_build" / "positions_r8.txt"
_I32_MIN, _I32_MAX = -(2**31), 2**31 - 1
#: Each arm's phrase in the pack's message, so a refusal is matched to the oracle's arm, not only its class.
_ARMS = {"range": "stone or dummy node", "count": "n_nodes_checksum - n_stones - 1",
         "slot": "canonical window slot"}
_CLASSES = {GatherNotLegalNode.__name__, ScatterSlotCanonicalMismatch.__name__}
#: The pack's three shapes: attribute rows, vocabulary codes, and codes padded by one graph.
_PACKS = ((False, False), (True, False), (True, True))


def _reference(f: dict[str, Any], trunk: int, node_feat_dim: int) -> tuple[str, str] | None:
    """Checks 15-16 as the semantic layer ran them in numpy, transcribed: `(class, arm)` or None."""
    gather, legal_offsets, node_offsets = f["legal_node_gather"], f["legal_offsets"], f["node_offsets"]
    if gather.size == 0:
        return None
    stones = f["n_stones"].astype(np.int64)
    graph = np.repeat(np.arange(legal_offsets.size - 1, dtype=np.int64), np.diff(legal_offsets))
    if np.any(gather < node_offsets[graph] + stones[graph]) or np.any(gather >= node_offsets[graph + 1] - 1):
        return (GatherNotLegalNode.__name__, "range")
    if not np.array_equal(np.diff(legal_offsets), f["n_nodes_checksum"].astype(np.int64) - stones - 1):
        return (GatherNotLegalNode.__name__, "count")
    coord = f["node_coords"].reshape(f["node_feat"].size // node_feat_dim, 2).astype(np.int64)[gather]
    centre = f["window_center"].reshape(-1, 2).astype(np.int64)[graph]
    half = (trunk - 1) // 2
    wq, wr = coord[:, 0] - centre[:, 0] + half, coord[:, 1] - centre[:, 1] + half
    inside = (wq >= 0) & (wq < trunk) & (wr >= 0) & (wr < trunk)
    canon = np.where(inside, wq * trunk + wr, -1).astype(np.int64)
    if not np.array_equal(canon, f["policy_dst_slot"].astype(np.int64)):
        return (ScatterSlotCanonicalMismatch.__name__, "slot")
    return None


def _pack_verdict(f: dict[str, Any], geometry: dict[str, int], coded: bool, padded: bool) -> tuple[str, str] | None:
    """The collate's refusal with the semantic layer off, as `(class, message)`, or None."""
    pad = None
    if padded:
        n, e = f["node_feat"].size // geometry["node_feat_dim"], f["edge_attr"].size // geometry["edge_feat_dim"]
        pad = (int(f["n_graphs"]) + 1, n + 3, e + 2, f["legal_node_gather"].size + 2)
    try:
        collate_graph_batch(GraphWirePayload(**f), device="cpu", semantic="off", coded_edges=coded,
                            pad_to=pad, **geometry)
    except GraphContractError as exc:
        return (type(exc).__name__, str(exc))
    return None


def _assert_parity(f: dict[str, Any], geometry: dict[str, int], label: str, tally: Counter) -> None:
    """Every pack shape gives the oracle's verdict, or refuses by a check 4-13 that ran before it as it always did."""
    want = _reference(f, geometry["trunk_size"], geometry["node_feat_dim"])
    for coded, padded in _PACKS:
        got = _pack_verdict(f, geometry, coded, padded)
        if got is not None and got[0] not in _CLASSES:
            tally["earlier"] += 1
            continue
        where = f"{label} (coded={coded}, padded={padded})"
        if want is None:
            assert got is None, f"{where}: numpy accepted, the pack refused {got}"
            tally["accepted"] += 1
            continue
        assert got is not None, f"{where}: numpy refused {want}, the pack accepted"
        assert got[0] == want[0] and _ARMS[want[1]] in got[1], f"{where}: numpy {want}, the pack {got}"
        tally[want[1]] += 1


def _with(f: dict[str, Any], name: str, edit: Any) -> dict[str, Any]:
    """`f` with a fresh copy of array `name` after `edit(copy)`."""
    arr = f[name].copy()
    edit(arr)
    return {**f, name: arr}


def _free_slot(f: dict[str, Any], g: int, trunk: int) -> int:
    """A window slot no legal node of graph `g` holds."""
    lo = f["legal_offsets"]
    used = set(f["policy_dst_slot"][int(lo[g]):int(lo[g + 1])].tolist())
    return next(s for s in range(trunk * trunk) if s not in used)


def _planted(f: dict[str, Any], trunk: int) -> list[tuple[str, dict[str, Any]]]:
    """One corruption per arm and two the oracle accepts, each placed so checks 4-13 still hold."""
    lo, no, ns, slots = f["legal_offsets"], f["node_offsets"], f["n_stones"], f["policy_dst_slot"]
    b = int(f["n_graphs"])
    g = next(k for k in range(b) if ns[k] > 0 and lo[k + 1] - lo[k] > 1)
    first, last = int(lo[g]), int(lo[g + 1]) - 1
    inside = next(i for i in range(first, last + 1) if slots[i] >= 0)
    cell = 2 * int(f["legal_node_gather"][inside])
    out = [
        ("gather onto a stone", _with(f, "legal_node_gather", lambda a: a.__setitem__(first, no[g]))),
        ("gather onto the dummy", _with(f, "legal_node_gather", lambda a: a.__setitem__(last, no[g + 1] - 1))),
        ("a stone uncounted", _with(f, "n_stones", lambda a: a.__setitem__(g, a[g] - 1))),
        ("a phantom stone", _with(f, "n_stones", lambda a: a.__setitem__(g, a[g] + 1))),
        ("centre moved along q", _with(f, "window_center", lambda a: a.__setitem__(2 * g, a[2 * g] + 1))),
        ("centre moved along r", _with(f, "window_center", lambda a: a.__setitem__(2 * g + 1, a[2 * g + 1] - 1))),
        ("an in-window slot as the sentinel", _with(f, "policy_dst_slot", lambda a: a.__setitem__(inside, -1))),
        ("a slot moved to a free one",
         _with(f, "policy_dst_slot", lambda a: a.__setitem__(inside, _free_slot(f, g, trunk)))),
        ("a legal cell moved", _with(f, "node_coords", lambda a: a.__setitem__(cell, a[cell] + 1))),
        ("a graph at the i32 extremes, every slot the sentinel",
         _with(_with(f, "window_center", lambda a: a.__setitem__(slice(2 * g, 2 * g + 2), (_I32_MIN, _I32_MAX))),
               "policy_dst_slot", lambda a: a.__setitem__(slice(first, last + 1), -1))),
    ]
    dropped = {**f, "legal_node_gather": np.delete(f["legal_node_gather"], first),
               "policy_dst_slot": np.delete(slots, first),
               "legal_offsets": np.where(np.arange(b + 1) > g, lo - 1, lo).astype(np.int64)}
    out.append(("a legal node dropped", dropped))
    outside = np.flatnonzero(slots == -1)
    if outside.size:
        i = int(outside[0])
        owner = int(np.searchsorted(lo, i, side="right")) - 1
        row = int(f["legal_node_gather"][i])
        out.append(("an off-window cell given a free slot",
                    _with(f, "policy_dst_slot", lambda a: a.__setitem__(i, _free_slot(f, owner, trunk)))))
        out.append(("an off-window cell at the i32 extremes",
                    _with(f, "node_coords", lambda a: a.__setitem__(slice(2 * row, 2 * row + 2), (_I32_MAX, _I32_MIN)))))
    return out


def _fuzzed(f: dict[str, Any], rng: np.random.Generator, trials: int) -> list[tuple[str, dict[str, Any]]]:
    """`trials` single-entry edits of the arrays checks 15-16 read, seeded."""
    names = ("legal_node_gather", "n_stones", "policy_dst_slot", "window_center", "node_coords")
    out = []
    for t in range(trials):
        name = names[int(rng.integers(len(names)))]
        if f[name].size == 0:
            continue
        i = int(rng.integers(f[name].size))
        delta = int(rng.choice([-1000, -3, -2, -1, 1, 2, 3, 1000]))
        info = np.iinfo(f[name].dtype)
        out.append((f"fuzz {t}: {name}[{i}] {delta:+d}", _with(
            f, name, lambda a, i=i, d=delta, info=info: a.__setitem__(i, min(max(int(a[i]) + d, info.min), info.max)))))
    return out


def _corpus() -> list[tuple[list[tuple[int, int, int]], int, int]]:
    """The recorded radius-8 positions as `(stones, to move, stones left)` requests."""
    out = []
    for line in _POSITIONS.read_text(encoding="utf-8").splitlines():
        head, *rest = line.split()
        if head == "M":
            continue
        v = [int(x) for x in rest]
        stones = v[3:]
        assert head == "S" and len(stones) == 3 * v[2], line[:40]
        out.append(([tuple(stones[k:k + 3]) for k in range(0, len(stones), 3)], v[0], v[1]))
    assert len(out) == 512, f"{len(out)} positions; the corpus holds 512"
    return out


def _built_wire(encoding: str, positions: list[Any]) -> dict[str, Any]:
    """One fused wire of `positions` off the production batcher; its waiters are failed once it is popped."""
    batcher = _engine.InferenceBatcher(encoding_spec=_engine.RegistrySpec.from_registry(encoding))
    refused: list[BaseException] = []

    def submit() -> None:
        try:
            batcher.submit_graphs_and_wait(positions, 1)
        except ValueError as exc:
            refused.append(exc)

    thread = threading.Thread(target=submit, daemon=True)
    thread.start()
    ids: list[int] = []
    wire: Any = None
    deadline = time.monotonic() + 30.0
    while not ids and time.monotonic() < deadline:
        popped, wire = batcher.next_graph_batch(len(positions), 200)
        ids = list(popped)
    batcher.submit_graph_inference_failure(ids, "popped for the pack parity test")
    thread.join(timeout=30.0)
    batcher.close()
    assert len(ids) == len(positions) and len(refused) == 1 and not thread.is_alive()
    payload = graph_wire_from_rust(wire)
    return {fl.name: getattr(payload, fl.name) for fl in dataclass_fields(payload)}


def test_the_pack_matches_numpy_on_the_captured_payloads(payload_fields) -> None:
    """The captured wires and their planted corruptions: every pack shape refuses what numpy refused, by arm."""
    geometry, tally = geometry_kwargs(), Counter()
    for name in ("b6", "b1", "b0"):
        _assert_parity(payload_fields(name), geometry, name, tally)
    for name in ("b6", "b1"):
        for label, f in _planted(payload_fields(name), geometry["trunk_size"]):
            _assert_parity(f, geometry, f"{name}: {label}", tally)
    assert tally["earlier"] == 0, f"a planted corruption tripped a check before 15: {tally}"
    assert all(tally[arm] > 0 for arm in (*_ARMS, "accepted")), tally


@pytest.mark.parametrize("encoding", ["gnn_axis_v1", "gnn_axis_r8"])
def test_the_pack_matches_numpy_on_built_wires_planted_and_fuzzed(encoding: str) -> None:
    """Wires built from the recorded positions at each shipped radius, planted and fuzzed alike."""
    geometry, tally, rng = geometry_kwargs(encoding), Counter(), np.random.default_rng(20261006)
    corpus = _corpus()[::16]
    for k in range(0, len(corpus), 8):
        f = _built_wire(encoding, corpus[k:k + 8])
        cases = [("clean", f), *_planted(f, geometry["trunk_size"]), *_fuzzed(f, rng, 24)]
        for label, case in cases:
            _assert_parity(case, geometry, f"{encoding} wire {k // 8}: {label}", tally)
    assert all(tally[arm] > 0 for arm in (*_ARMS, "accepted")), tally


def test_a_trunk_the_wire_was_not_built_at_is_refused_on_both_sides(payload_fields) -> None:
    """The pack reads the caller's trunk, not one of its own: a wrong trunk moves every in-window slot."""
    geometry = {**geometry_kwargs(), "trunk_size": geometry_kwargs()["trunk_size"] + 6}
    f = payload_fields("b6")
    assert _reference(f, geometry["trunk_size"], geometry["node_feat_dim"]) == (
        ScatterSlotCanonicalMismatch.__name__, "slot")
    _assert_parity(f, geometry, "b6 at a wider trunk", Counter())


@pytest.mark.parametrize(("semantic", "period"), [("off", 64), ("canary", 64), ("full", 64)])
def test_checks_15_and_16_refuse_every_batch_whatever_the_canary(payload_fields, semantic: str, period: int) -> None:
    """In the pack they run on every batch, stricter than any canary period, in every semantic mode."""
    reset_semantic_canary()
    for _ in range(4):
        f = payload_fields("b6")
        f["window_center"][0] += 1
        with pytest.raises(ScatterSlotCanonicalMismatch):
            collate_graph_batch(GraphWirePayload(**f), device="cpu", semantic=semantic,
                                canary_period=period, **geometry_kwargs())
    reset_semantic_canary()
