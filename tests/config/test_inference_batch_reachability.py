"""The collector's saturation threshold against the in-flight supply the same config provisions.

The Rust collector waits for `inference_batch_size / 2` queued graphs before serving a
forward. In-flight supply is `n_workers` under the serial per-graph submit and
`n_workers * leaf_batch_size` under `submit_graphs_and_wait`; when the threshold exceeds
the supply, every forward runs to the `inference_max_wait_ms` deadline and serves whatever
happens to be queued (measured once at a 1/64 fill).

The `/ 2` divisor is a Rust literal with no Python accessor, restated here once; its
behavioural authority is `crates/mantis-selfplay/tests/queue_roundtrip.rs`
(`a_reachable_threshold_returns_before_the_deadline` and
`an_unreachable_threshold_still_serves_on_the_deadline`), which red if the divisor moves.

The knob move that would make every minted config reachable is a separate, separately
benched package: this file asserts what the dispatch change alone delivers plus a named
tripwire on the configs still starved.
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
    """Return every minted config whose representation is `graph`, via the one discovery authority."""
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
    """Return the serial-submit supply: one graph in flight per worker."""
    return knobs["n_workers"]


def _batched_supply(knobs: dict[str, int]) -> int:
    """Return the batched-submit supply: one whole leaf batch in flight per worker."""
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
    """Prove the batched submit raises in-flight supply on every shipped graph config.

    A config minted with `leaf_batch_size: 1` provisions the same supply either way, so the
    change would land and measure nothing.
    """
    knobs = _knobs(load_config(config_path))
    assert knobs["leaf_batch_size"] > 1, (
        f"{config_path.name}: leaf_batch_size {knobs['leaf_batch_size']} makes the batched "
        "submit a no-op — one graph per worker in flight either way"
    )
    assert _batched_supply(knobs) == _serial_supply(knobs) * knobs["leaf_batch_size"]
    assert _batched_supply(knobs) > _serial_supply(knobs)


def test_the_batched_submit_makes_the_threshold_reachable_where_the_serial_one_could_not() -> None:
    """Prove at least one minted graph config flips from unreachable to reachable.

    Configs that do not flip are the next test's business; if none flips, the change has no
    shipped config it can be measured on at all.
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
    """Prove a still-starved graph config is starved only because it provisions one worker.

    A config that raised `n_workers` and is still starved moved the knob without clearing
    the threshold, which reads in-run as "the batching fix did not work".
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
    """Prove the reachability predicate refuses a starved provisioning and accepts a supplied one.

    Zero discovered configs, or a predicate stuck at True, would let every assertion above
    pass while asserting nothing.
    """
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
    # The same provisioning is still starved under the serial submit — the whole claim.
    assert not _reachable(provisioned, _serial_supply(provisioned))
