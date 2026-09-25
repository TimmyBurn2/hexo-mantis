"""One served forward's CPU stage never blocks on the device: `set_sync_debug_mode("error")` over `_launch_pop`."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch

from mantis._engine import InferenceBatcher
from mantis.config.census import production_configs
from mantis.config.loader import load_config
from mantis.config.resolve.edge_geometry_check import resolve_edge_geometry_check
from mantis.encoding import lookup
from mantis.model import arch_from_spec_and_config, build_net
from mantis.selfplay.inference_server import InferenceServer

_REPO = Path(__file__).resolve().parents[2]
_GAME = [(-2, 2), (1, 2), (3, 1), (-2, 1), (-2, 0), (-2, -2), (-2, -1), (-1, 1), (-1, 0), (0, 0),
         (-3, 2), (0, 1), (-3, 1), (-4, 1), (1, 1), (-4, 2), (-1, -1), (0, -2)]

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="LOUD SKIP — the served forward's device syncs exist only on CUDA")


def _positions(encoding: str) -> list[tuple[list[tuple[int, int, int]], int, int]]:
    from mantis._engine import Board

    board, out = Board.with_encoding_name(encoding), []
    for q, r in _GAME:
        out.append((list(board.get_stones()), int(board.current_player), int(board.moves_remaining)))
        board.apply_move(q, r)
    return out


def _serve_with_checked_launch(compile_trunk: bool,
                               plant: Callable[[], None] | None = None) -> list[BaseException]:
    """Serve two pops through the real server; the SECOND pop's launch runs under sync-debug "error"."""
    config = load_config(production_configs(_REPO)[0]).model_dump()
    spec = lookup(config["identity"]["encoding"])
    net = build_net(arch_from_spec_and_config(spec, config)).cuda()
    batcher = InferenceBatcher(encoding_spec=spec)
    server = InferenceServer(net, torch.device("cuda"), config, batcher=batcher, encoding_spec=spec,
                             edge_geometry_check=resolve_edge_geometry_check(config),
                             compile_trunk=compile_trunk)
    launch, calls, caught = server._launch_pop, [], []

    def checked(*args: Any) -> Any:
        calls.append(1)
        if len(calls) != 2:
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

    server._launch_pop = checked  # type: ignore[method-assign]
    frames_before = server.batch_timing_snapshot()["compile"]["frames_ok"]
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
    assert len(calls) >= 2, f"only {len(calls)} pop(s) launched; the checked launch never ran"
    return caught


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
