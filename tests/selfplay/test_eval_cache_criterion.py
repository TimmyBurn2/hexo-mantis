"""The cache criterion, controls first: a replay stays inside the served path's own batch-size spread; a side-blind key and a stale net fail it."""
from __future__ import annotations

import copy
import json
import os
from collections.abc import Callable, Sequence
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
from mantis.train.checkpoints import load_checkpoint

_REPO = Path(__file__).resolve().parents[2]
_GAMES = [[(-2, 2), (1, 2), (3, 1), (-2, 1), (-2, 0), (-2, -2), (-2, -1), (-1, 1), (-1, 0), (0, 0), (-3, 2)],
          [(0, 0), (1, 0), (0, 1), (2, -1), (-1, 1), (3, -2), (-2, 2), (1, 1), (4, -2), (-1, 2), (0, 2), (2, 0)],
          [(0, 0), (-1, 1), (1, -1), (2, -2), (-2, 2), (0, 1), (0, -1), (1, 1), (-1, -1), (3, -3), (-3, 3)]]
_BATCH_SIZES = range(1, 65)
#: Where the measured spread and the three verdicts are written when set (a records path, never the tree).
_RECORD_ENV = "MANTIS_EVAL_CACHE_CRITERION_RECORD"

Position = tuple[list[tuple[int, int, int]], int, int]
Served = tuple[list[float], float]

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="LOUD SKIP — the batch-size spread is a property of the CUDA served path")


def _positions(encoding: str) -> list[Position]:
    """Every position of the fixed games, each followed by its side-swapped twin."""
    out: list[Position] = []
    for game in _GAMES:
        board = Board.with_encoding_name(encoding)
        for q, r in game:
            stones = [(int(a), int(b), int(c)) for a, b, c in board.get_stones()]
            player, left = int(board.current_player), int(board.moves_remaining)
            if stones:
                out.append((stones, player, left))
                out.append((stones, -player, left))
            board.apply_move(q, r)
    return out


def _serve(batcher: InferenceBatcher, positions: Sequence[Position], size: int) -> list[Served]:
    """Each position's policy row and value when served in pops of `size`."""
    out: list[Served] = []
    for i in range(0, len(positions), size):
        for dense, _overflow, value in batcher.submit_graphs_and_wait(list(positions[i:i + size]), 1):
            out.append((list(dense), float(value)))
    return out


def _spread(by_size: dict[int, list[Served]], n: int) -> list[tuple[float, float]]:
    """Per position: the max-min range of the value and of every policy entry across batch sizes."""
    spread = []
    for p in range(n):
        values = [by_size[b][p][1] for b in by_size]
        rows = torch.tensor([by_size[b][p][0] for b in by_size])
        spread.append((max(values) - min(values), float((rows.max(0).values - rows.min(0).values).max())))
    return spread


def _violations(keys: Sequence[str], stored: Sequence[Served], fresh: Sequence[Served],
                spread: Sequence[tuple[float, float]]) -> list[int]:
    """Positions whose replay (the first entry stored under their key) leaves their own spread."""
    table: dict[str, Served] = {}
    for key, served in zip(keys, stored, strict=True):
        table.setdefault(key, served)
    out = []
    for p, (key, (row, value)) in enumerate(zip(keys, fresh, strict=True)):
        hit_row, hit_value = table[key]
        dev_row = max((abs(a - b) for a, b in zip(hit_row, row, strict=True)), default=0.0)
        if abs(hit_value - value) > spread[p][0] or dev_row > spread[p][1]:
            out.append(p)
    return out


def _base_net(config: dict, spec: object) -> torch.nn.Module:
    """The parent when MANTIS_PERF_CHECKPOINT names it, else a seeded fresh net of the config's arch."""
    parent = os.environ.get("MANTIS_PERF_CHECKPOINT")
    if parent:
        ck = load_checkpoint(Path(parent))
        assert ck.metadata.arch is not None
        net = build_net(ck.metadata.arch)
        net.load_state_dict(ck.model_state)
        return net
    torch.manual_seed(20260926)
    return build_net(arch_from_spec_and_config(spec, config))


def _drifted(net: torch.nn.Module, rel: float) -> torch.nn.Module:
    """A copy with every float weight scaled by `1 + rel`: roughly one optimizer step's drift."""
    out = copy.deepcopy(net)
    with torch.no_grad():
        for p in out.parameters():
            if p.is_floating_point():
                p.mul_(1.0 + rel)
    return out


def _with_server(config: dict, spec: object, net: torch.nn.Module,
                 body: Callable[[InferenceBatcher], object]) -> object:
    torch._dynamo.reset()
    batcher = InferenceBatcher(encoding_spec=spec)
    server = InferenceServer(net.cuda(), torch.device("cuda"), config, batcher=batcher, encoding_spec=spec,
                             compile_trunk=resolve_compile_trunk(config))
    server.start()
    try:
        return body(batcher)
    finally:
        server.stop()
        server.join(timeout=30.0)


def test_the_criterion_passes_the_real_key_and_fails_both_known_bad_paths() -> None:
    config = load_config(production_configs(_REPO)[0]).model_dump()
    encoding = config["identity"]["encoding"]
    spec = lookup(encoding)
    positions = _positions(encoding)

    def measure(batcher: InferenceBatcher) -> tuple[dict[int, list[Served]], list[str]]:
        return {b: _serve(batcher, positions, b) for b in _BATCH_SIZES}, batcher.eval_cache_keys(positions)

    base = _base_net(config, spec)
    by_size, keys = _with_server(config, spec, base, measure)  # type: ignore[misc]
    newer = _with_server(config, spec, _drifted(base, 1e-3), lambda b: _serve(b, positions, 64))
    spread = _spread(by_size, len(positions))
    board_keys = [repr(sorted(stones)) for stones, _player, _left in positions]

    correct = {b: _violations(keys, by_size[64], by_size[b], spread) for b in _BATCH_SIZES}
    incomplete = _violations(board_keys, by_size[64], by_size[1], spread)
    stale = _violations(keys, by_size[64], newer, spread)  # type: ignore[arg-type]

    record = os.environ.get(_RECORD_ENV)
    if record:
        Path(record).write_text(json.dumps({
            "net": os.environ.get("MANTIS_PERF_CHECKPOINT") or "seeded fresh net",
            "positions": len(positions),
            "value_spread_max": max(s[0] for s in spread),
            "policy_spread_max": max(s[1] for s in spread),
            "correct_violations": sum(len(v) for v in correct.values()),
            "incomplete_key_violations": len(incomplete),
            "stale_net_violations": len(stale),
        }, indent=1), encoding="utf-8")

    distinct = {(repr(sorted(stones)), player, left) for stones, player, left in positions}
    assert len(set(keys)) == len(distinct), "the real key conflates two distinct positions"
    assert len(set(board_keys)) < len(distinct), "the control's premise: twins share a board key"
    assert incomplete, "the criterion cannot see a key blind to the side to move: it is void"
    assert stale, "the criterion cannot see a replay to a newer net: it is void"
    assert not any(correct.values()), f"the real key leaves the spread: {correct}"
