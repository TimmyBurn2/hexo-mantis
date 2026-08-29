"""PERF-BASELINE §2 E — the deploy head's single-stream cost, decomposed.

One uncontended self-play game driven by `DeployHeadPlayer` over the production graph
decode (`LocalInferenceEngine.infer_batch_ls` -> the ONE `InferenceServer`), at
`eval.gate.deploy_sims` and the config's own `selfplay.leaf_batch_size` (R318(b)).

`LocalInferenceEngine` hardcodes `perf_timing: False` in its own server dict literal, so the
server is subclassed here to arm the diagnostic timers. Nothing else about the path changes.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import GpuSampler, git_sha, stats, write_json  # noqa: E402

from mantis._engine import Board  # noqa: E402
from mantis.arena.deploy_head import DeployHeadPlayer  # noqa: E402
from mantis.config.loader import load_config  # noqa: E402
from mantis.config.resolve.fused_graph_caps import resolve_fused_graph_caps  # noqa: E402
from mantis.selfplay.hparams import resolve_pool_encoding  # noqa: E402
from mantis.diagnostics.worker_sweep import build_sweep_net  # noqa: E402
from mantis.model import arch_from_spec_and_config  # noqa: E402

TIMES: dict[str, list[float]] = defaultdict(list)
LEAVES = [0]


def arm_perf_server(sync_cuda: bool) -> None:
    """Force the diagnostic timers on for the server `LocalInferenceEngine` builds."""
    from mantis.selfplay import inference_server as mod

    base = mod.InferenceServer

    class PerfServer(base):  # type: ignore[misc,valid-type]
        def __init__(self, *a: Any, **kw: Any) -> None:
            super().__init__(*a, **kw)
            self._perf_timing = True
            self._perf_sync_cuda = sync_cuda

    mod.InferenceServer = PerfServer


def timed(name: str, fn: Any) -> Any:
    def wrapper(*a: Any, **kw: Any) -> Any:
        t0 = time.perf_counter()
        out = fn(*a, **kw)
        TIMES[name].append((time.perf_counter() - t0) * 1e3)
        return out
    return wrapper


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/run5.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-moves", type=int, default=40)
    ap.add_argument("--sync-cuda", action="store_true")
    args = ap.parse_args()

    arm_perf_server(args.sync_cuda)
    from mantis.eval.worker import _graph_expand_fn  # noqa: PLC0415 — after the patch
    from mantis.selfplay.inference_local import LocalInferenceEngine  # noqa: PLC0415

    config = load_config(args.config)
    raw = config.model_dump()
    device = torch.device("cuda")
    spec = resolve_pool_encoding(raw, arch=None).registry_spec
    arch = arch_from_spec_and_config(spec, raw)
    model = build_sweep_net(config, arch, device)
    caps = resolve_fused_graph_caps(raw)
    engine = LocalInferenceEngine(model, device, encoding_spec=spec, fused_graph_caps=caps)
    server = engine._graph_server  # noqa: SLF001 — diagnostic read

    n_sims = config.eval.gate.deploy_sims
    k = config.selfplay.leaf_batch_size
    engine.infer_batch_ls = timed("infer_batch_ls", engine.infer_batch_ls)
    base_expand = _graph_expand_fn(engine, spec)

    def expand(tree: Any, leaves: list[Any]) -> None:
        LEAVES[0] += len(leaves)
        base_expand(tree, leaves)

    player = DeployHeadPlayer(expand_fn=timed("expand_fn", expand), n_sims=n_sims,
                              leaf_batch_size=k)

    record: dict[str, Any] = {
        "sha": git_sha(),
        "regime": {"deploy_sims": n_sims, "leaf_batch_size": k, "encoding": spec.name,
                   "sync_cuda": args.sync_cuda,
                   "server": {"inference_batch_size": 64, "inference_max_wait_ms": 10,
                              "note": "LocalInferenceEngine's own hardcoded server dict"}},
    }
    sampler = GpuSampler()
    try:
        board = Board.with_encoding_name(spec.name)
        player.new_game()
        # One warm move so trace/allocator/first-batch costs stay out of the sample.
        warm_move = player.select_move(board)
        board.apply_move(*warm_move)
        TIMES.clear()
        LEAVES[0] = 0
        py_before = server.perf_stage_snapshot()
        bt_before = server.batch_timing_snapshot()
        sampler.start()
        move_ms: list[float] = []
        t0 = time.monotonic()
        moves = 0
        while moves < args.max_moves and not board.check_win() and board.legal_moves():
            t = time.perf_counter()
            move = player.select_move(board)
            move_ms.append((time.perf_counter() - t) * 1e3)
            board.apply_move(*move)
            moves += 1
        wall = time.monotonic() - t0
        sampler.stop()
        py_after = server.perf_stage_snapshot()
        bt_after = server.batch_timing_snapshot()
        sampler.join(timeout=5.0)

        sims = LEAVES[0]
        stages = {}
        for name, row in py_after["stages"].items():
            prev = py_before["stages"].get(name, {"count": 0, "total_ms": 0.0})
            count = row["count"] - prev["count"]
            total = row["total_ms"] - prev["total_ms"]
            stages[name] = {"count": count, "total_ms": total,
                            "mean_ms": (total / count) if count else None,
                            "ms_per_sim": total / sims if sims else None}
        record.update({
            "moves_played": moves, "wall_sec": wall, "sims_total": sims,
            "ms_per_move": stats(move_ms),
            "ms_per_sim": (wall * 1e3 / sims) if sims else None,
            "sims_per_move": sims / moves if moves else None,
            "python_stages": stages,
            "python_stage_total_ms_per_sim":
                sum(v["total_ms"] for v in stages.values()) / sims if sims else None,
            "caller_stages_ms_per_sim": {
                name: sum(v) / sims for name, v in TIMES.items()} if sims else None,
            "caller_stages_total_ms": {name: sum(v) for name, v in TIMES.items()},
            "batch_timing_before": bt_before,
            "batch_timing_after": bt_after,
            "gpu": sampler.summary(),
        })
    finally:
        engine.close()

    write_json(args.out, record)
    print(f"eval single-stream: {record.get('ms_per_sim')} ms/sim over "
          f"{record.get('sims_total')} sims, {record.get('moves_played')} moves")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
