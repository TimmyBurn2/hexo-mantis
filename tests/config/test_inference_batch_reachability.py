"""Q-FIND-1 — the collector's saturation threshold against the in-flight supply the same
config provisions.

The Rust collector waits for `inference_batch_size / 2` queued graphs
(`GraphQueue::pop_graph_batch_blocking`) before serving a forward. The in-flight supply is
`n_workers * <graphs a worker can have queued at once>`: ONE under the serial per-graph
submit, and the whole `leaf_batch_size` under `GraphQueue::submit_graphs_and_wait`. When
the threshold exceeds the supply the loop can NEVER satisfy its condition — every forward
runs to the `inference_max_wait_ms` deadline and then serves whatever happens to be
queued, which is what Q-FIND-1 measured as a 1/64 = 1.5625% fill. Nothing in the schema,
in any gate, or in any test asserted the relation before this file.

The `/ 2` divisor is a Rust literal with no Python accessor, so it is restated here ONCE,
with its behavioural authority pinned on the Rust side rather than here:
`crates/mantis-selfplay/tests/queue_roundtrip.rs::a_reachable_threshold_returns_before_the_deadline`
(batch_size 8, supply 8 ⇒ returns off the threshold) and
`::an_unreachable_threshold_still_serves_on_the_deadline` (batch_size 64, supply 1 ⇒ runs
to the deadline and still serves). If the divisor moves, those two red.

SCOPE (ruling R263): the knob move that would make every minted config reachable is a
SEPARATE, separately-benched package and is deliberately NOT made here. What this file
asserts is what the dispatch change alone delivers, plus a named tripwire on the configs
that are still starved — so the gap is visible and un-ignorable rather than silent.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mantis.config.loader import discover_configs, load_config

_REPO = Path(__file__).resolve().parents[2]
_CONFIGS_DIR = _REPO / "configs"

# `pop_graph_batch_blocking`: `let threshold = batch_size / 2;`
_THRESHOLD_DIVISOR = 2


def _graph_config_paths() -> list[Path]:
    """Every minted config whose representation is `graph` — the only ones the graph
    queue serves. `discover_configs` is the ONE discovery authority (R71), not a sixth
    flat glob."""
    return [p for p in discover_configs(_CONFIGS_DIR) if load_config(p).identity.representation == "graph"]


def _knobs(config: Any) -> dict[str, int]:
    return {
        "n_workers": config.selfplay.n_workers,
        "leaf_batch_size": config.selfplay.leaf_batch_size,
        "inference_batch_size": config.inference.inference_batch_size,
        "inference_max_wait_ms": config.inference.inference_max_wait_ms,
    }


def _threshold(knobs: dict[str, int]) -> int:
    return knobs["inference_batch_size"] // _THRESHOLD_DIVISOR


def _serial_supply(knobs: dict[str, int]) -> int:
    """Pre-change: one blocking submit per worker ⇒ one graph in flight per worker."""
    return knobs["n_workers"]


def _batched_supply(knobs: dict[str, int]) -> int:
    """Post-change: one `submit_graphs_and_wait` per worker ⇒ the whole leaf batch."""
    return knobs["n_workers"] * knobs["leaf_batch_size"]


def _reachable(knobs: dict[str, int], supply: int) -> bool:
    return _threshold(knobs) <= supply


def _ledger() -> list[tuple[str, dict[str, int], int, int, int]]:
    rows = []
    for path in _graph_config_paths():
        knobs = _knobs(load_config(path))
        rows.append(
            (path.name, knobs, _threshold(knobs), _serial_supply(knobs), _batched_supply(knobs))
        )
    return rows


def _ledger_text() -> str:
    return "\n".join(
        f"  {name}: threshold {thr} (inference_batch_size {k['inference_batch_size']} // 2), "
        f"serial supply {ser} (n_workers), batched supply {bat} "
        f"(n_workers {k['n_workers']} x leaf_batch_size {k['leaf_batch_size']}) "
        f"-> {'REACHABLE' if thr <= bat else 'STARVED'}"
        for name, k, thr, ser, bat in _ledger()
    )


@pytest.mark.parametrize("config_path", _graph_config_paths(), ids=lambda p: p.name)
def test_the_batched_submit_raises_the_in_flight_supply_on_every_graph_config(
    config_path: Path,
) -> None:
    """The dispatch change must not be a no-op on any shipped graph config.

    A config minted with `leaf_batch_size: 1` would provision the same supply before and
    after — the fix would land and measure nothing, and the in-run `occupancy` histogram
    would stay pinned on its `{"1": N}` bucket with no defect anywhere to find.
    """
    knobs = _knobs(load_config(config_path))
    assert knobs["leaf_batch_size"] > 1, (
        f"{config_path.name}: leaf_batch_size {knobs['leaf_batch_size']} makes the batched "
        "submit a no-op — one graph per worker in flight either way"
    )
    assert _batched_supply(knobs) == _serial_supply(knobs) * knobs["leaf_batch_size"]
    assert _batched_supply(knobs) > _serial_supply(knobs)


def test_the_batched_submit_makes_the_threshold_reachable_where_the_serial_one_could_not() -> None:
    """The config-level headline: on at least one minted graph config the batched submit
    flips the collector threshold from unreachable to reachable.

    Not every config flips — the ones that do not are the prereg BATCHING row's business
    (R263 scopes the knob move out of this change) and the next test names them. But if NO
    shipped config flips, the fix has no config it can be measured on at all, and the
    failure message below prints the whole derived ledger rather than a bare `False`.
    """
    rows = _ledger()
    assert rows, "no graph configs discovered — the ledger is vacuous"
    flipped = [
        name
        for name, knobs, _thr, ser, bat in rows
        if not _reachable(knobs, ser) and _reachable(knobs, bat)
    ]
    assert flipped, (
        "no minted graph config gains a reachable collector threshold from the batched "
        "submit — the fix cannot be measured in-run on anything shipped:\n" + _ledger_text()
    )


@pytest.mark.parametrize("config_path", _graph_config_paths(), ids=lambda p: p.name)
def test_a_still_starved_graph_config_is_starved_only_on_the_worker_supply_axis(
    config_path: Path,
) -> None:
    """The named tripwire on the half R263 scopes out.

    A graph config whose threshold is still unreachable after the dispatch change must be
    starved for exactly ONE reason: it provisions a single worker. That is the axis the
    prereg BATCHING row moves (`n_workers`, or `inference_batch_size` downward). A config
    that raised `n_workers` and is STILL starved moved the knob without clearing the
    threshold — a half-applied prereg row, which is precisely the state that would read as
    "the batching fix did not work" in the run.
    """
    knobs = _knobs(load_config(config_path))
    if _reachable(knobs, _batched_supply(knobs)):
        return
    assert knobs["n_workers"] == 1, (
        f"{config_path.name}: n_workers {knobs['n_workers']} was raised but the collector "
        f"threshold {_threshold(knobs)} still exceeds the batched supply "
        f"{_batched_supply(knobs)} — every forward still burns the "
        f"{knobs['inference_max_wait_ms']} ms deadline. Finish the prereg BATCHING row "
        f"(raise n_workers to >= {-(-_threshold(knobs) // knobs['leaf_batch_size'])} or "
        f"lower inference_batch_size to <= {2 * _batched_supply(knobs)}):\n" + _ledger_text()
    )


def test_the_reachability_predicate_is_not_vacuous() -> None:
    """Self-test (LAW-07): the predicate must REFUSE a starved provisioning and ACCEPT a
    provisioned one. Without this, a discovery that found zero graph configs — or a
    predicate that returned `True` unconditionally — would let every assertion above pass
    while asserting nothing."""
    assert _graph_config_paths(), "no graph configs discovered — the parametrization is vacuous"

    starved = {
        "n_workers": 1, "leaf_batch_size": 8,
        "inference_batch_size": 64, "inference_max_wait_ms": 10,
    }
    assert _threshold(starved) == 32
    assert _serial_supply(starved) == 1
    assert _batched_supply(starved) == 8
    assert not _reachable(starved, _batched_supply(starved)), (
        "a 1-worker/8-leaf config against a 64-slot collector MUST read as starved — "
        "this is the Q-FIND-1 shape itself"
    )

    provisioned = dict(starved, n_workers=4)
    assert _batched_supply(provisioned) == 32
    assert _reachable(provisioned, _batched_supply(provisioned))
    # ... and the SAME provisioning is still starved under the serial submit, which is the
    # whole claim the dispatch change makes.
    assert not _reachable(provisioned, _serial_supply(provisioned))
