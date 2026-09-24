"""The schema tests' valid section builders, derived off dev_example's resolved train block."""
from __future__ import annotations

from pathlib import Path

from mantis.config.loader import load_config

_REPO = Path(__file__).resolve().parents[2]

MINTED_TRAIN: dict = load_config(_REPO / "configs" / "dev_example.yaml").train.model_dump()


def train_block(**over: object) -> dict:
    return dict(MINTED_TRAIN, **over)


def eval_block() -> dict:
    return {
        "random_model_sims": 96, "max_plies": 128, "random_floor_games": 0, "worker_device": "cuda",
        "round_timeout_sec": 3600.0, "worker_kill_grace_sec": 10.0,
        "ply_cap_adjudication": None, "strength_floor": None,
        "gate": {
            "stride": 1, "screen_games": 80, "confirm_games": 128, "promotion_winrate": 0.55,
            "screen_confirm_lo": 0.44, "deploy_sims": 150, "opening_book": "book_v1_s20260625_p4",
            "bootstrap_resamples": 1000, "min_distinct_per_pair": 10, "seed_base": 20260625, "sequential": None,
        },
    }


def selfplay_block() -> dict:
    return {
        "search": {"kind": "puct"}, "n_workers": 1, "leaf_batch_size": 8, "max_game_moves": 128,
        "c_visit": 50.0,
        "c_scale": 1.0, "q_rescale": True, "gumbel_m": 16, "gumbel_explore_moves": 10, "search_stats_every": 8,
        "results_queue_cap": 10_000, "random_opening_plies": 0,
        "log_investigation_metrics": True,
        "mcts": {"n_simulations": 50, "c_puct": 1.5, "fpu_reduction": 0.25,
                 "quiescence_enabled": True, "quiescence_blend_2": 0.3,
                 "dirichlet_alpha": 0.3, "dirichlet_epsilon": 0.25, "dirichlet_enabled": True},
        "playout_cap": {"fast_sims": 50, "fast_prob": 0.0, "standard_sims": 0,
                        "full_search_prob": 0.0, "n_sims_quick": 0, "n_sims_full": 0,
                        "temperature_threshold_compound_moves": 0, "temp_min": 0.5},
    }


def inference_block() -> dict:
    return {
        "inference_batch_size": 64, "inference_max_wait_ms": 10,
        # `inference.fused_graph_caps` is a REQUIRED block, and the pair here is the
        # template's NON-BINDING-BY-CONSTRUCTION value, so nothing exercises a split; the
        # real configs are pinned by tests/config/test_fused_graph_caps_authority.py.
        "fused_graph_caps": {"max_fused_edges": 57149441, "max_fused_nodes": 1785921},
    }


def monitor_block(**over: object) -> dict:
    base = {
        # R242 (ADJ-D12): the ARMING cadence, schema-only and required.
        "gate_interval": 1000,
        "alert_entropy_min": 1.0, "collapse_threshold_nats": 1.5, "alert_grad_norm_max": 10.0,
        "alert_loss_increase_window": 3, "axis_warn": 0.45, "axis_alert": 0.50,
        "heartbeat_deadline_train_step_sec": 1800.0,
        "heartbeat_deadline_inference_dispatch_sec": 1800.0,
        "heartbeat_deadline_selfplay_drain_sec": 1800.0,
        "heartbeat_deadline_eval_round_sec": 1800.0,
        "heartbeat_poll_interval_sec": 5.0, "heartbeat_file_interval_sec": 15.0,
        "heartbeat_close_out_deadline_sec": 14400.0, "heartbeat_fire_effect_timeout_sec": 30.0,
        "supervisor_stale_after_sec": 900.0, "supervisor_poll_interval_sec": 30.0,
        "supervisor_kill_grace_sec": 30.0, "supervisor_max_relaunches": 5,
        "actor_lag_threshold_steps": 100, "actor_lag_abort_enabled": False,
        "drain": {"final_eval_drain_timeout_sec": 900.0,
                  "eval_final_drain_safety_factor": 3.0,
                  "eval_final_drain_hard_cap_sec": 14400.0,
                  "terminal_eval_hard_cap_sec": 14400.0},
        "disk_guard": {"interval_sec": 60.0, "warn_gb": 10.0, "fail_gb": 5.0},
    }
    base.update(over)
    return base
