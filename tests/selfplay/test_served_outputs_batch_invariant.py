"""A position's served evaluation does not depend on the batch it rode in — so an exact-cache hit is bit-identical."""
from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from mantis._engine import Board, InferenceBatcher
from mantis.config.census import production_configs
from mantis.config.loader import load_config
from mantis.config.resolve.compile_trunk import resolve_compile_trunk
from mantis.encoding import lookup
from mantis.model import arch_from_spec_and_config, build_net
from mantis.selfplay.inference_server import InferenceServer

_REPO = Path(__file__).resolve().parents[2]
_GAMES = [[(-2, 2), (1, 2), (3, 1), (-2, 1), (-2, 0), (-2, -2), (-2, -1), (-1, 1), (-1, 0), (0, 0), (-3, 2)],
          [(0, 0), (1, 0), (0, 1), (2, -1), (-1, 1), (3, -2), (-2, 2), (1, 1), (4, -2), (-1, 2), (0, 2), (2, 0)]]

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="LOUD SKIP — the served path's batch invariance is a CUDA-kernel property")


def _positions(encoding: str) -> list[tuple[list[tuple[int, int, int]], int, int]]:
    out = []
    for game in _GAMES:
        board = Board.with_encoding_name(encoding)
        for q, r in game:
            out.append((list(board.get_stones()), int(board.current_player), int(board.moves_remaining)))
            board.apply_move(q, r)
    return out


def test_every_grouping_serves_every_position_bit_identically() -> None:
    """Whole batch, one by one, odd chunks and shuffled: each position's policy and value are equal in every grouping."""
    config = load_config(production_configs(_REPO)[0]).model_dump()
    spec = lookup(config["identity"]["encoding"])
    torch.manual_seed(20260925)
    net = build_net(arch_from_spec_and_config(spec, config)).cuda()
    torch._dynamo.reset()
    batcher = InferenceBatcher(encoding_spec=spec)
    server = InferenceServer(net, torch.device("cuda"), config, batcher=batcher, encoding_spec=spec,
                             compile_trunk=resolve_compile_trunk(config))
    positions = _positions(config["identity"]["encoding"])
    order = list(range(len(positions)))
    random.Random(7).shuffle(order)
    server.start()
    try:
        whole = batcher.submit_graphs_and_wait(positions, 1)
        alone = [batcher.submit_graphs_and_wait([p], 1)[0] for p in positions]
        chunked = [r for i in range(0, len(positions), 7)
                   for r in batcher.submit_graphs_and_wait(positions[i:i + 7], 1)]
        shuffled = dict(zip(order, batcher.submit_graphs_and_wait([positions[i] for i in order], 1), strict=True))
    finally:
        server.stop()
        server.join(timeout=30.0)
    for name, got in (("alone", alone), ("chunks of 7", chunked),
                      ("shuffled", [shuffled[i] for i in range(len(positions))])):
        diffs = [i for i, (a, b) in enumerate(zip(whole, got, strict=True)) if a != b]
        assert not diffs, f"{name}: positions {diffs} serve differently than in the whole batch"
