"""One served forward's CPU stage never blocks on the device: `set_sync_debug_mode("error")` over `_launch_pop`."""
from __future__ import annotations

from collections.abc import Callable
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from mantis._engine import Board, InferenceBatcher
from mantis.config.census import production_configs
from mantis.config.loader import load_config
from mantis.config.resolve.edge_geometry_check import resolve_edge_geometry_check
from mantis.encoding import lookup
from mantis.model import arch_from_spec_and_config, build_net
from mantis.selfplay.graph_collate import collate_graph_batch, graph_wire_from_rust
from mantis.selfplay.inference_server import InferenceServer

_REPO = Path(__file__).resolve().parents[2]
_GAME = [(-2, 2), (1, 2), (3, 1), (-2, 1), (-2, 0), (-2, -2), (-2, -1), (-1, 1), (-1, 0), (0, 0),
         (-3, 2), (0, 1), (-3, 1), (-4, 1), (1, 1), (-4, 2), (-1, -1), (0, -2)]

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(),
                                 reason="LOUD SKIP — the served forward's device syncs exist only on CUDA")]


def _positions(encoding: str) -> list[tuple[list[tuple[int, int, int]], int, int]]:
    board, out = Board.with_encoding_name(encoding), []
    for q, r in _GAME:
        out.append((list(board.get_stones()), int(board.current_player), int(board.moves_remaining)))
        board.apply_move(q, r)
    return out


def _serve_with_checked_launch(compile_trunk: bool, plant: Callable[[], None] | None = None,
                               checked: int = 2) -> list[BaseException]:
    """Serve two pops through the real server; pop `checked`'s launch runs under sync-debug "error" (the first captures its bucket)."""
    if compile_trunk:
        torch._dynamo.reset()  # an earlier test's cache entries would otherwise stand in for this compile
    config = load_config(production_configs(_REPO)[0]).model_dump()
    # One pop per submit, so the checked launch is the second submission whole and never a warm-up's tail.
    config["inference"]["inference_batch_size"] = max(int(config["inference"]["inference_batch_size"]), len(_GAME))
    spec = lookup(config["identity"]["encoding"])
    net = build_net(arch_from_spec_and_config(spec, config)).cuda()
    batcher = InferenceBatcher(encoding_spec=spec)
    server = InferenceServer(net, torch.device("cuda"), config, batcher=batcher, encoding_spec=spec,
                             edge_geometry_check=resolve_edge_geometry_check(config),
                             compile_trunk=compile_trunk)
    launch, calls, caught = server._launch_pop, [], []
    frames_before = server.batch_timing_snapshot()["compile"]["frames_ok"]
    if checked == 1:
        # Compile, autotune and library handles happen here, eagerly: the checked first pop is left with the capture.
        _prewarm(server, spec, _positions(config["identity"]["encoding"]))

    def checked_launch(*args: Any) -> Any:
        calls.append(1)
        if len(calls) != checked:
            return launch(*args)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            if plant is not None:
                plant()
            return launch(*args)
        except RuntimeError as exc:
            caught.append(exc)
            raise
        finally:
            torch.cuda.set_sync_debug_mode("default")

    server._launch_pop = checked_launch  # type: ignore[method-assign]
    server.start()
    positions = _positions(config["identity"]["encoding"])
    try:
        batcher.submit_graphs_and_wait(positions, 1)
        try:
            batcher.submit_graphs_and_wait(positions, 1)
        except ValueError:
            assert caught, "the checked pop failed with no sync error recorded"
    finally:
        server.stop()
        server.join(timeout=30.0)
    # Dynamo's frame counter is process-global, so only this server's delta says it compiled.
    frames = server.batch_timing_snapshot()["compile"]["frames_ok"] - frames_before
    assert (frames > 0) == compile_trunk, f"compile_trunk={compile_trunk} but {frames} compiled frame(s)"
    assert len(calls) == 2, f"{len(calls)} pops for two submissions; the checked launch was not one whole submission"
    return caught


def _prewarm(server: InferenceServer, spec: Any, positions: list[Any]) -> None:
    batcher = InferenceBatcher(encoding_spec=spec)
    submit = threading.Thread(target=batcher.submit_graphs_and_wait, args=(positions, 1))
    submit.start()
    ids, wire = batcher.next_graph_batch(len(positions), 100)
    while not ids:
        ids, wire = batcher.next_graph_batch(len(positions), 100)
    payload = graph_wire_from_rust(wire)
    batch = collate_graph_batch(payload, device="cuda", semantic="off", trunk_size=spec.trunk_size,
                                win_length=spec.win_length, node_feat_dim=spec.node_feat_dim,
                                edge_feat_dim=spec.edge_feat_dim, coded_edges=True)
    with torch.inference_mode():
        server._serve(batch)
    torch.cuda.synchronize()
    counts = np.diff(np.asarray(payload.legal_offsets))
    batcher.submit_graph_inference_results(ids, np.repeat(1.0 / counts, counts).astype(np.float32),
                                           np.asarray(payload.legal_offsets), np.zeros(len(ids), np.float32))
    submit.join(timeout=10.0)


def _capturing_pop(compile_trunk: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """The first pop under sync-debug "error", which cannot see the allocator's own syncs: the cache release is counted."""
    releases: list[int] = []
    release = torch.cuda.empty_cache

    def counted() -> None:
        releases.append(1)
        release()

    monkeypatch.setattr(torch.cuda, "empty_cache", counted)
    caught = _serve_with_checked_launch(compile_trunk, checked=1)
    assert not caught, f"a capturing pop synchronised with the device: {caught}"
    assert len(releases) == 1, f"{len(releases)} cache releases for the one bucket the pop captured"


def test_a_pop_that_captures_its_bucket_launches_without_a_host_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bucket's warm-ups and capture run on the serving thread's first pop: its one device wait is the cache release."""
    _capturing_pop(False, monkeypatch)


@pytest.mark.slow
def test_a_compiled_pop_that_captures_its_bucket_launches_without_a_host_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same first pop through the compiled trunk the box serves with."""
    _capturing_pop(True, monkeypatch)


def test_the_instrument_reds_on_a_planted_sync() -> None:
    """PLANTED BREAK: an `.item()` inside the checked window must be caught, or a green below proves nothing."""
    caught = _serve_with_checked_launch(False, plant=lambda: torch.ones(1, device="cuda").sum().item())
    assert caught and "synchroniz" in str(caught[0]), f"the planted sync was not caught: {caught}"


def test_one_served_forward_launches_without_a_host_sync() -> None:
    """The eager trunk: collate, H2D, forward and the queued D2H never wait on the device."""
    caught = _serve_with_checked_launch(False)
    assert not caught, f"the served forward synchronized: {caught[0]}"


@pytest.mark.slow
def test_one_compiled_served_forward_launches_without_a_host_sync() -> None:
    """The compiled trunk the box serves with: the same property through Inductor's kernels."""
    caught = _serve_with_checked_launch(True)
    assert not caught, f"the compiled served forward synchronized: {caught[0]}"
