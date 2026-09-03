"""PERF-TRANCHE-1 G-2 — the eval path's collector geometry comes from CONFIG, not a literal.

Ledger F-2: `LocalInferenceEngine` built its graph `InferenceServer` from a dict literal
carrying `inference_batch_size: 64` and `inference_max_wait_ms: 10`. The literal's own
comment already stated the argument against exactly this for the cap beside it — *"a cap
written here would be a SECOND authority over one byte budget, on the one construction path
with no config to be the first"* — and the ledger measured what the un-threaded pair costs
on the arm LAW-15 reads a promotion bar off: at the single-stream deploy head, supply 8
against a collector threshold the literal set to 32, **1.76 of the eval path's 5.30 ms/sim,
33 %**, is the collector's own deadline.

The planted break for every row here is the literal: pin a value in
`inference_local.py`'s dict and the constructed server stops following the config.
"""
from __future__ import annotations

import pytest
import torch

from mantis._engine import RegistrySpec
from mantis.config.resolve.fused_graph_caps import FusedGraphCapsSpec
from mantis.config.resolve.inference_batching import (
    InferenceBatchingSpec,
    MissingInferenceBatchingError,
    resolve_inference_batching,
)
from mantis.selfplay.inference_local import LocalInferenceEngine

_CPU = torch.device("cpu")
_GRAPH_SPEC = RegistrySpec.from_registry("gnn_axis_v1")
_GRID_SPEC = RegistrySpec.from_registry("v6")
_CAPS = FusedGraphCapsSpec(max_fused_edges=4_500_000, max_fused_nodes=170_000)


class _Net(torch.nn.Module):
    """The smallest thing the engine will hold; no forward is driven here."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)


def _graph_engine(batching: InferenceBatchingSpec, max_in_flight: int) -> LocalInferenceEngine:
    return LocalInferenceEngine(
        _Net(), _CPU, encoding_spec=_GRAPH_SPEC, fused_graph_caps=_CAPS,
        inference_batching=batching, max_in_flight=max_in_flight, amp_dtype="bf16",
    )


@pytest.mark.parametrize(
    "batch_size, max_wait_ms",
    [(64, 10), (16, 3), (128, 25)],
)
def test_the_graph_server_takes_its_batching_from_the_threaded_spec(
    batch_size: int, max_wait_ms: int
) -> None:
    """The constructed server FOLLOWS the spec — three distinct values, not just run5's.

    One value would pass against a literal that happened to equal it; that is precisely the
    coincidence `test_deploy_matched_hparam_coincidence.py` was written to police, and this
    row exists to make policing unnecessary. A literal cannot follow three values.
    """
    engine = _graph_engine(
        InferenceBatchingSpec(inference_batch_size=batch_size,
                              inference_max_wait_ms=max_wait_ms),
        max_in_flight=8,
    )
    try:
        server = engine._graph_server
        assert server is not None, "a graph engine must construct a graph server"
        assert int(server._batch_size) == batch_size, (
            f"the server's pop width is {server._batch_size}, not the threaded "
            f"{batch_size} — it is reading a literal again (ledger F-2)"
        )
        assert int(server._max_wait_ms) == max_wait_ms, (
            f"the server's pop deadline is {server._max_wait_ms}, not the threaded "
            f"{max_wait_ms} — it is reading a literal again (ledger F-2)"
        )
    finally:
        engine.close()


@pytest.mark.parametrize("supply", [1, 8, 96])
def test_the_graph_batcher_takes_the_declared_supply(supply: int) -> None:
    """G-1's relation reaches the eval route: the batcher is TOLD what it can ever hold.

    Without this, a single-stream deploy head submits `leaf_batch_size` graphs against a
    saturation threshold of `batch_size / 2` and every forward runs to the deadline — the
    same unreachable relation ledger F-1 measured on the self-play route, and 33 % of the
    eval path's ms/sim (F-2).
    """
    engine = _graph_engine(InferenceBatchingSpec(64, 10), max_in_flight=supply)
    try:
        assert engine._graph_batcher is not None
        assert engine._graph_batcher.graph_max_in_flight == supply, (
            "the eval batcher was not told its supply, so the collector's threshold falls "
            "back to the half-batch and a single-stream head pays the deadline on every pop"
        )
    finally:
        engine.close()


def test_a_graph_engine_refuses_an_absent_batching_spec() -> None:
    """`None` is the GRID arm. A graph engine handed it raises rather than falling back."""
    with pytest.raises(ValueError, match="inference_batching"):
        LocalInferenceEngine(
            _Net(), _CPU, encoding_spec=_GRAPH_SPEC, fused_graph_caps=_CAPS,
            inference_batching=None, max_in_flight=8, amp_dtype="bf16",
        )


def test_a_grid_engine_carries_none_and_builds_no_graph_server() -> None:
    """The `None` arm is a real posture, not an oversight: a grid route opens no collector."""
    engine = LocalInferenceEngine(
        _Net(), _CPU, encoding_spec=_GRID_SPEC, fused_graph_caps=None,
        inference_batching=None, max_in_flight=0, amp_dtype="bf16",
    )
    try:
        assert engine._graph_server is None
        assert engine._graph_batcher is None
    finally:
        engine.close()


def test_the_resolver_refuses_an_absent_member() -> None:
    """LAW-11 at the read path: absent is a named error, never a default."""
    good = {"inference": {"inference_batch_size": 64, "inference_max_wait_ms": 10}}
    assert resolve_inference_batching(good) == InferenceBatchingSpec(64, 10)
    for missing in ("inference_batch_size", "inference_max_wait_ms"):
        partial = {"inference": {k: v for k, v in good["inference"].items() if k != missing}}
        with pytest.raises(MissingInferenceBatchingError, match=missing):
            resolve_inference_batching(partial)
    with pytest.raises(MissingInferenceBatchingError, match="no `inference` section"):
        resolve_inference_batching({})


def test_the_round_spec_carries_the_batching_across_the_process_seam() -> None:
    """The eval child is a SEPARATE PROCESS; a field that does not round-trip never arrives.

    Rehydration is the failure this pins: `asdict` flattens the frozen dataclass to a plain
    mapping, and a field missing from `_REHYDRATED_SPEC_FIELDS` comes back as that mapping
    and fails at the child's first attribute read — inside a subprocess whose stderr nobody
    is reading.
    """
    import dataclasses
    import json

    from mantis.eval.rounds import GateSpec, RoundSpec

    batching = InferenceBatchingSpec(inference_batch_size=32, inference_max_wait_ms=7)
    spec = RoundSpec(
        round_id="r1", step=1, candidate_snapshot="c.pt", best_snapshot=None, best_step=None,
        encoding="gnn_axis_v1", worker_device="cpu",
        gate=GateSpec(stride=1, screen_games=2, confirm_games=2, promotion_winrate=0.55,
                      screen_confirm_lo=0.5, deploy_sims=8, opening_book="none",
                      bootstrap_resamples=10, min_distinct_per_pair=1, seed_base=1,
                      run_gate=False),
        rung_jobs=[], random_floor_games=0, random_model_sims=1, sealbot_model_sims=1,
        kraken_model_sims=1, strix_model_sims=1, seed_base=1, round_timeout_sec=1.0,
        result_path="r.json", progress_path="p.txt", ladder_bootstrap_resamples=10,
        ladder_bootstrap_ci_level=0.95, ladder_bootstrap_seed=1,
        ply_cap_adjudication=None, strength_floor=None, fused_graph_caps=_CAPS,
        inference_batching=batching, leaf_batch_size=8, c_visit=50.0, c_scale=1.0, amp_dtype="bf16", max_plies=128, leaf_build_threads=1,
    )
    back = RoundSpec.from_dict(json.loads(json.dumps(dataclasses.asdict(spec))))
    assert isinstance(back.inference_batching, InferenceBatchingSpec), (
        f"the batching came back as {type(back.inference_batching).__name__}, not the "
        "dataclass — the child would raise on its first attribute read, in a subprocess"
    )
    assert back.inference_batching == batching
    assert back == spec

    # The `None` arm round-trips as `None`, not as a rehydration failure.
    grid = dataclasses.replace(spec, inference_batching=None, fused_graph_caps=None)
    assert RoundSpec.from_dict(json.loads(json.dumps(dataclasses.asdict(grid)))) == grid
