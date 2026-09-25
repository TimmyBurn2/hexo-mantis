"""Shared fixtures for the `tests/train/` suites.

This conftest imports only torch and `mantis.model`/`mantis.encoding`/`mantis.config`, never
`mantis.train.*`, so it collects cleanly. The root conftest installs the autouse reseed
fixture; this file does not re-seed and does not touch sys.modules.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from _fused_caps import CAPS_DICT
from mantis.config.loader import load_config
from mantis.config.schema import ARCH_SCOPED_KEYS
from mantis.encoding import lookup
from mantis.model import GnnArch, arch_from_spec_and_config, build_net
from _spy import SpyEventSink

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
TRAIN_FIXTURES = FIXTURES / "train"
ANCHOR_KEYS_FILE = FIXTURES / "value_probes" / "statedict_keys" / "gnn_axis_v1.txt"

# The ONE registered representation; the grid path is gone.
GRAPH_ENCODING = "gnn_axis_v1"
KILLED_PREFIXES = ("cluster_pool.", "global_encoder.", "gpool_bias_branch.")


@pytest.fixture
def spy_sink() -> SpyEventSink:
    return SpyEventSink()


def make_tiny_arch() -> GnnArch:
    """Build the tiny graph net: the registered wire dims at toy widths."""
    from _microbatch_harness import tiny_graph_arch

    return tiny_graph_arch()


@pytest.fixture
def tiny_arch() -> GnnArch:
    return make_tiny_arch()


@pytest.fixture
def tiny_net(tiny_arch: GnnArch) -> torch.nn.Module:
    return build_net(tiny_arch)


def make_optim_scaler_sched(
    net: torch.nn.Module, *, lr: float = 1e-3, t_max: int = 1000, eta_min: float = 1e-5
):
    """Build a two-param-group AdamW (the golden's `param_groups==2`), a CPU GradScaler with
    real state, and a CosineAnnealingLR."""
    decay = [p for _, p in net.named_parameters() if p.ndim >= 2]
    no_decay = [p for _, p in net.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 1e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )
    scaler = torch.amp.GradScaler("cpu", enabled=True)  # non-empty state_dict on CPU
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max, eta_min=eta_min)
    return opt, scaler, sched


@pytest.fixture
def optim_scaler_sched(tiny_net: torch.nn.Module):
    return make_optim_scaler_sched(tiny_net)


# The legacy read path resolves arch from the encoding, so this is the FULL registry arch at
# the incumbent widths, not a tiny net.
@pytest.fixture(scope="session")
def full_graph_net() -> torch.nn.Module:
    return build_net(arch_from_spec_and_config(lookup(GRAPH_ENCODING), {}))


@pytest.fixture
def full_graph_state(full_graph_net: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return a fresh shallow copy of the full `gnn_axis_v1` state dict."""
    return dict(full_graph_net.state_dict())


def _make_eval_block() -> dict[str, Any]:
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


# Derived from a minted config rather than restated, so a new `train.*` key costs no edit here
# and cannot disagree with the schema.
_MINTED_TRAIN: dict[str, Any] = load_config(
    Path(__file__).resolve().parents[2] / "configs" / "dev_example.yaml").train.model_dump()


def _make_train_block(**over: Any) -> dict[str, Any]:
    return dict(_MINTED_TRAIN, **over)


def _make_selfplay_block(**over: Any) -> dict[str, Any]:
    base = {
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
    base.update(over)
    return base


def _make_inference_block(**over: Any) -> dict[str, Any]:
    base = {
        "inference_batch_size": 64, "inference_max_wait_ms": 10,
        # A non-binding-by-construction pair: no split is exercised here.
        "fused_graph_caps": CAPS_DICT,
    }
    base.update(over)
    return base


def _make_monitor_block(**over: Any) -> dict[str, Any]:
    base = {
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
        "drain": {
            "final_eval_drain_timeout_sec": 900.0, "eval_final_drain_safety_factor": 3.0,
            "eval_final_drain_hard_cap_sec": 14400.0, "terminal_eval_hard_cap_sec": 14400.0,
        },
        "disk_guard": {"interval_sec": 60.0, "warn_gb": 10.0, "fail_gb": 5.0},
    }
    base.update(over)
    return base


def make_run_config(encoding: str = GRAPH_ENCODING, representation: str = "graph",
                    run_id: str = "run5") -> dict[str, Any]:
    """Build a complete, schema-valid RunConfig dict.

    Arch-scoped blocks are dropped for the representation that does not have them, driven from
    `ARCH_SCOPED_KEYS` rather than by deleting names.
    """
    config = {
        "schema_version": 1,
        "eval_enabled": True,
        # A required top-level leaf; `null` is the placeholder, refused at boot on a cuda
        # process.
        "allocator_posture": None,
        "run_id": run_id,
        "seed": 20260718,
        "identity": {"encoding": encoding, "representation": representation},
        # The tiny arch's own widths (v35): 4 x 128 over a 16 x 1 net is the drift the writer refuses.
        "model": {"gnn": {"hidden": 16, "num_layers": 1}, "aux_soft_policy": None},
        "deploy": {"search": {"kind": "puct"}},
        "eval": _make_eval_block(),
        "train": _make_train_block(),
        "selfplay": _make_selfplay_block(),
        "inference": _make_inference_block(),
        "monitor": _make_monitor_block(),
    }
    for key in ARCH_SCOPED_KEYS:
        if representation != key.arch:
            config[key.section].pop(key.field, None)
    return config


# Every TrainHParams field is required, so the ONE factory (`_microbatch_harness.graph_hparams`)
# layers overrides over a full set and a test passes only the fields it cares about.
def make_full_train_hparams(**over: Any):
    from _microbatch_harness import graph_hparams

    return graph_hparams(**over)


@pytest.fixture
def full_train_hparams():
    return make_full_train_hparams


@pytest.fixture
def valid_config() -> dict[str, Any]:
    return make_run_config()


@pytest.fixture
def invalid_config() -> dict[str, Any]:
    """Return a config that fails schema validation: complete, plus one unknown key."""
    cfg = make_run_config()
    cfg["__unknown_knob__"] = True
    return cfg


def make_metadata_kwargs(arch: GnnArch, *, encoding_name: str = GRAPH_ENCODING,
                         run_id: str = "runa", corpus_sha256: str | None = None
                         ) -> dict[str, Any]:
    """Build the metadata stamp inputs; `created_utc`/`commit_sha` are stamped once by
    `save_checkpoint`, and supplying them here would be a restamp."""
    mk: dict[str, Any] = {"encoding_name": encoding_name, "run_id": run_id, "arch": arch}
    if corpus_sha256 is not None:
        mk["corpus_sha256"] = corpus_sha256
    return mk


@pytest.fixture
def metadata_kwargs(tiny_arch: GnnArch) -> dict[str, Any]:
    return make_metadata_kwargs(tiny_arch)


# Callable fixtures, so a test can vary encoding/run_id without importing conftest by name.
@pytest.fixture
def mk_config():
    return make_run_config


@pytest.fixture
def mk_meta():
    return make_metadata_kwargs


@pytest.fixture(scope="session")
def resume_goldens() -> dict[str, Any]:
    return json.loads((TRAIN_FIXTURES / "resume_goldens.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def legacy_shapes() -> dict[str, Any]:
    return json.loads((TRAIN_FIXTURES / "legacy_payload_shapes.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def anchor_key_set() -> set[str]:
    return set(ANCHOR_KEYS_FILE.read_text(encoding="utf-8").split())
