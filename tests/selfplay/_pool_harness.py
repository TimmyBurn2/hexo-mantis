"""The hand-built WorkerPool config and factory the surface and lifecycle suites share."""
from __future__ import annotations

from typing import Any

import torch

from _fused_caps import CAPS_DICT
from mantis._engine import HexgBuffer
from mantis.encoding import lookup
from mantis.model import GnnArch, build_net
from mantis.selfplay.pool import WorkerPool


def pool_cfg(
    encoding: str = "gnn_axis_v1", *, n_simulations: int = 8, fast_sims: int = 8,
    **over: Any
) -> dict[str, Any]:
    # `selfplay`/`inference`/`train` are nested schema-shaped sections, so the `from_config`
    # readers take no flat dict with a top-level fallback. `over` still layers onto `selfplay`.
    selfplay: dict[str, Any] = {
        "search": {"kind": "puct"}, "n_workers": 1, "leaf_batch_size": 8, "max_game_moves": 128,
        "c_visit": 50.0,
        "c_scale": 1.0, "q_rescale": True, "gumbel_m": 16, "gumbel_explore_moves": 10, "search_stats_every": 8,
        "results_queue_cap": 10_000, "random_opening_plies": 0,
        "log_investigation_metrics": True,
        "mcts": {"n_simulations": n_simulations, "c_puct": 1.5, "fpu_reduction": 0.25,
                 "quiescence_enabled": True, "quiescence_blend_2": 0.3,
                 "dirichlet_alpha": 0.3, "dirichlet_epsilon": 0.25, "dirichlet_enabled": True},
        "playout_cap": {"fast_sims": fast_sims, "fast_prob": 0.0, "standard_sims": 0,
                        "full_search_prob": 0.0, "n_sims_quick": 0, "n_sims_full": 0,
                        "temperature_threshold_compound_moves": 0, "temp_min": 0.5},
    }
    selfplay.update(over)
    inference = {
        "inference_batch_size": 4, "inference_max_wait_ms": 10,
        "edge_geometry_check": "inline", "compile_trunk": False,
        # The graph arm resolves the fused-forward memory bound at construction; NON-BINDING
        # BY CONSTRUCTION here, since these suites are about wiring and nothing asserts the M.
        "fused_graph_caps": CAPS_DICT,
    }
    train = {"draw_reward": -0.5, "ply_cap_value": -0.5}
    return {"encoding": encoding, "deploy": {"search": {"kind": "puct"}}, "selfplay": selfplay,
            "inference": inference, "train": train}


def graph_pool(
    *,
    device: torch.device | None = None,
    buffer: Any = None,
    capacity: int = 256,
    visit_capacity: int = 128,
    n_simulations: int = 8,
    fast_sims: int = 8,
    **pool_over: Any,
) -> WorkerPool:
    """A graph WorkerPool over a fresh 1-layer GNN; `buffer` replaces the raw buffer when given,
    and `pool_over` forwards to the WorkerPool constructor."""
    spec = lookup("gnn_axis_v1")
    arch = GnnArch(in_dim=spec.node_feat_dim, edge_dim=spec.edge_feat_dim,
                   hidden=16, num_layers=1)
    raw = buffer if buffer is not None else HexgBuffer(
        capacity=capacity, encoding="gnn_axis_v1", visit_capacity=visit_capacity)
    return WorkerPool(
        build_net(arch), pool_cfg(n_simulations=n_simulations, fast_sims=fast_sims),
        device or torch.device("cpu"), raw, arch=arch,
        **pool_over,
    )
