"""Shared fixtures for the WP10 ⊕⊕ conformance suites (tests/train/).

>300 justify: one shared fixture module for one directory's suites — the spies, the tiny-net
+ optim/scaler/sched builders, the full `v6_live2_ls` net, and the `RunConfig`/`TrainHParams`
block factories (`train`/`selfplay`/`inference`/`monitor`) all have to stay co-located so
every `tests/train/` suite draws its config shape from ONE place; splitting them would let
two copies of a block factory drift apart. WPSC Phase 2 SC-A1/SC-A2's `train:`/`selfplay:`
reshape of `make_run_config` plus the new `full_train_hparams` fixture factory
(DESIGN_P2.md §2.1) is what pushed this file past the cap.

This conftest imports ONLY already-present layers (torch + mantis.model / mantis.encoding
/ mantis.config) and NEVER `mantis.train.*` — so it collects cleanly while the two suites
are RED (the suites import `mantis.train.*`, which does not exist until IMPL; that is the
correct oracle-first state). Helper spies (EventSink / clock / call recorders) are plain
duck-typed classes: the injected `EventSink` is a structural Protocol (single `emit` method),
so a bare class with `.emit` satisfies it without importing the not-yet-written Protocol.

Root conftest already installs the autouse `_reseed` fixture (random/numpy/torch) — this
file does NOT re-seed and does NOT touch sys.modules (R5/LAW-17).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from mantis.config.loader import load_config
from mantis.config.schema import ARCH_SCOPED_KEYS
from mantis.encoding import lookup
from mantis.model import CnnArch, arch_from_spec_and_config, build_net

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
TRAIN_FIXTURES = FIXTURES / "train"
ANCHOR_KEYS_FILE = FIXTURES / "value_probes" / "anchor_keys" / "v6_live2.txt"

# Encoding used for the grid checkpoint tests: v6_live2_ls (grid, 19, 4 planes) is the O3b
# PASS anchor lineage and a registered encoding.
GRID_ENCODING = "v6_live2_ls"
KILLED_PREFIXES = ("cluster_pool.", "global_encoder.", "gpool_bias_branch.")


# ── spies (duck-typed; satisfy the structural EventSink / callable seams) ─────────────────
class SpyEventSink:
    """Records every emitted event Mapping. Satisfies the structural `EventSink` Protocol
    (single `emit(event: Mapping)` method). The event NAME travels under the `event` key
    (mantis emit convention; cf. `mantis.config.emit.ResolvedConfig.to_event_payload`)."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event: Any) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("event") == name]

    def has(self, name: str) -> bool:
        return any(e.get("event") == name for e in self.events)


class FakeClock:
    """Controllable monotonic clock: `clock()` returns the current fake time `t`."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def spy_sink() -> SpyEventSink:
    return SpyEventSink()


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()


# ── tiny nets + optim/scaler/scheduler (real torch objects) ──────────────────────────────
def make_tiny_arch() -> CnnArch:
    """The DESIGN §b tiny net: build_net(CnnArch(filters=16, res_blocks=1, ...))."""
    return CnnArch(board_size=19, in_channels=4, filters=16, res_blocks=1)


@pytest.fixture
def tiny_arch() -> CnnArch:
    return make_tiny_arch()


@pytest.fixture
def tiny_net(tiny_arch: CnnArch) -> torch.nn.Module:
    return build_net(tiny_arch)


def make_optim_scaler_sched(
    net: torch.nn.Module, *, lr: float = 1e-3, t_max: int = 1000, eta_min: float = 1e-5
):
    """Two-param-group AdamW (weight-decay split → the golden's `param_groups==2`) + a CPU
    GradScaler with real state + a CosineAnnealingLR."""
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


# ── full v6_live2_ls net (registry arch) — the strict-load target for legacy / O3b tests ──
# The bare O3b anchor and the legacy read path resolve arch from the encoding → the FULL
# registry arch (filters=128, res_blocks=12), NOT a tiny net. Built once per session.
@pytest.fixture(scope="session")
def full_ls_net() -> torch.nn.Module:
    return build_net(arch_from_spec_and_config(lookup(GRID_ENCODING), {}))


@pytest.fixture
def full_ls_state(full_ls_net: torch.nn.Module) -> dict[str, torch.Tensor]:
    """A fresh shallow copy of the full v6_live2_ls state dict (147 keys, O3b-clean)."""
    return dict(full_ls_net.state_dict())


# WP11-A schema extension: eval.gate/eval.ladder are now required fields (design §c.1).
def _make_eval_block() -> dict[str, Any]:
    return {
        "random_model_sims": 96, "sealbot_model_sims": 128, "kraken_model_sims": 128,
        "strix_model_sims": 128, "random_floor_games": 0, "worker_device": "cuda",
        "round_timeout_sec": 3600.0, "worker_kill_grace_sec": 10.0,
        "ply_cap_adjudication": None, "strength_floor": None,
        "gate": {
            "stride": 1, "screen_games": 80, "confirm_games": 128, "promotion_winrate": 0.55,
            "screen_confirm_lo": 0.44, "deploy_sims": 150, "opening_book": "book_v1_s20260625_p4",
            "bootstrap_resamples": 1000, "min_distinct_per_pair": 10, "seed_base": 20260625,
        },
        "ladder": {
            "rungs": [{"name": "sealbot_d5", "bot": "sealbot", "variant": "d5", "depth": 5,
                      "opponent_sims": None, "opening_book": "book_v1_s20260625_p4",
                      "deploy_matched": True, "games_max": 32}],
            "round_games": 64, "min_games_per_active_rung": 4, "graduation_wr_lower_ci": 0.75,
            "graduation_consec_rounds": 3, "activation_wr_lower_ci": 0.65,
            "calibration_every_k_rounds": 4, "calibration_games": 8,
            "bootstrap_resamples": 1000, "bootstrap_ci_level": 0.95,
            "bt_prior_games": 1.0, "bootstrap_seed": 1234,
        },
    }


# WPSC Phase 2 SC-A1: `train:` is a required RunConfig section.
# WPMINT Phase K-A stage 0: the block is DERIVED from a MINTED config rather than restated —
# eleven files carried a hand-written copy, so a new `train.*` key cost eleven edits and gave
# eleven chances to disagree with the schema. `dev_example.yaml`'s resolved block was measured
# byte-identical to the census it replaces (which was itself the zero-behavior-change
# TrainHParams-dataclass-default carry-over, DESIGN_P2.md §1.1/§2), so the swap changes nothing.
_MINTED_TRAIN: dict[str, Any] = load_config(
    Path(__file__).resolve().parents[2] / "configs" / "dev_example.yaml").train.model_dump()


def _make_train_block(**over: Any) -> dict[str, Any]:
    return dict(_MINTED_TRAIN, **over)


# WPSC Phase 2 SC-A2: `selfplay:` gains `mcts:`/`playout_cap:` sub-blocks + many new required
# scalars; `legal_move_radius_schedule` is GONE (DESIGN_P2.md §5); `inference:` is a new
# required top-level section. Zero-behavior-change values carried over (DESIGN_P2.md §1.2).
def _make_selfplay_block(**over: Any) -> dict[str, Any]:
    base = {
        "n_workers": 1, "leaf_batch_size": 8, "max_game_moves": 128,
        "inference_pool_size": None, "completed_q_values": False, "c_visit": 50.0,
        "c_scale": 1.0, "gumbel_mcts": False, "gumbel_m": 16, "gumbel_explore_moves": 10,
        "results_queue_cap": 10_000, "random_opening_plies": 0, "rotation_enabled": True,
        "forced_win_policy_enabled": False, "forced_win_policy_depth": 2,
        "forced_win_policy_weight": 1.0, "solver_enabled": False, "solver_depth": 16,
        "solver_node_budget": 50_000, "solver_neighbor_dist": 2, "solver_visit_weight": 0.3,
        "seed_fraction": 0.0, "seed_corpus_path": None, "log_investigation_metrics": True,
        "instrumentation_enabled": False,
        "mcts": {"n_simulations": 50, "c_puct": 1.5, "fpu_reduction": 0.25,
                 "quiescence_enabled": True, "quiescence_blend_2": 0.3,
                 "dirichlet_alpha": 0.3, "dirichlet_epsilon": 0.25, "dirichlet_enabled": True},
        "playout_cap": {"fast_sims": 50, "fast_prob": 0.0, "standard_sims": 0,
                        "full_search_prob": 0.0, "n_sims_quick": 0, "n_sims_full": 0,
                        "zoi_enabled": False, "zoi_lookback": 16, "zoi_margin": 5,
                        "temperature_threshold_compound_moves": 0, "temp_min": 0.5},
    }
    base.update(over)
    return base


def _make_inference_block(**over: Any) -> dict[str, Any]:
    base = {
        "inference_batch_size": 64, "inference_max_wait_ms": 10, "trace_inference": True,
        "compile_inference": False, "compile_inference_mode": "default",
        "compile_inference_dynamic": True, "perf_timing": False, "perf_sync_cuda": False,
        # F-816-10: `inference.fused_graph_caps` is a REQUIRED block. The pair here is
        # the template's NON-BINDING-BY-CONSTRUCTION value, so nothing in this file
        # exercises a split; the R119 `null` placeholder is pinned by
        # tests/config/test_fused_graph_caps_authority.py against the real configs.
        "fused_graph_caps": {"max_fused_edges": 57149441, "max_fused_nodes": 1785921},
    }
    base.update(over)
    return base


# WPSC Phase 2 SC-A3: `monitor:` is now a required RunConfig section — every value below is
# the zero-behavior-change `mantis.monitor.config.MonitorConfig` dataclass default carried
# over verbatim (DESIGN_P2.md §4.2), plus the 4 `DrainCapsConfig` fields (§4.3).
def _make_monitor_block(**over: Any) -> dict[str, Any]:
    base = {
        # R242 (ADJ-D12): the ARMING cadence, schema-only and required.
        "gate_interval": 1000,
        "alert_entropy_min": 1.0, "collapse_threshold_nats": 1.5, "alert_grad_norm_max": 10.0,
        "alert_loss_increase_window": 3, "wr_hard_abort_enabled": False,
        "wr_rolling_consecutive_evals": 2, "wr_rolling_threshold": 0.10,
        "wr_rolling_min_step": 20000, "wr_collapse_from_peak_ratio": 0.5,
        "wr_collapse_min_step": 25000, "wr_collapse_consecutive_evals": 3,
        "wr_early_death_threshold": 0.05, "wr_early_death_min_step": 15000,
        "axis_warn": 0.45, "axis_alert": 0.50,
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


# ── schema-valid / invalid config snapshots (validated against config-schema v1 on write) ─
def make_run_config(encoding: str = GRID_ENCODING, representation: str = "grid",
                    run_id: str = "run5") -> dict[str, Any]:
    """A complete, schema-v1-valid RunConfig dict (the envelope `config` snapshot).

    ARCH-SCOPED BLOCKS ARE DROPPED FOR THE REPRESENTATION THAT DOES NOT HAVE THEM (R322(d)).
    The factory's default is `grid`, and its `train` block is derived from a GRAPH config's
    dump, so before B2 it produced a grid config carrying two graph-only cap blocks — exactly
    the shape `RunConfig` now refuses. Driven from `ARCH_SCOPED_KEYS` rather than by deleting
    two names, so a third scoped block needs no edit here.
    """
    config = {
        "schema_version": 1,
        "eval_enabled": True,
        # RECAL-PREP (R308(g)(i)): a REQUIRED top-level leaf. `null` is R119's
        # placeholder — refused at boot on a cuda process, valued only by the
        # re-calibration sitting under R282(b).
        "allocator_posture": None,
        "run_id": run_id,
        "seed": 20260718,
        "identity": {"encoding": encoding, "representation": representation},
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


# ── shared TrainHParams factory (DESIGN_P2.md §2.1 recommendation) — every TrainHParams
# field is now required (no dataclass default, R-TRAINCONFIG-SCHEMA closure); this factory
# returns the zero-behavior-change values with **overrides layered on, so a test passes only
# the fields it cares about instead of enumerating all ~24 at every call site.
def make_full_train_hparams(**over: Any):
    from mantis.train.trainer.core import TrainHParams

    base = dict(
        lr=1e-3, weight_decay=1e-4, grad_clip=1.0, fp16=True, lr_schedule="cosine",
        total_steps=1_000_000, scheduler_t_max=None, eta_min=5e-4, min_lr=None,
        checkpoint_interval=0, completed_q_values=False, policy_prune_frac=0.0,
        entropy_reg_weight=0.0, aux_opp_reply_weight=0.0, uncertainty_weight=0.0,
        ownership_weight=0.0, threat_weight=0.0, aux_chain_weight=0.0, ply_index_weight=0.0,
        threat_pos_weight=1.0, value_target="pure_outcome_z",
        policy_target="raw_visit_distribution", draw_reward=-0.5, ply_cap_value=-0.5,
    )
    base.update(over)
    return TrainHParams(**base)


@pytest.fixture
def full_train_hparams():
    return make_full_train_hparams


@pytest.fixture
def valid_config() -> dict[str, Any]:
    return make_run_config()


@pytest.fixture
def invalid_config() -> dict[str, Any]:
    """A config that fails schema v1 (extra=forbid): a complete config + one unknown key."""
    cfg = make_run_config()
    cfg["__unknown_knob__"] = True
    return cfg


# ── metadata_kwargs (the stamp inputs to save_checkpoint; encoding_name REQUIRED) ─────────
def make_metadata_kwargs(arch: CnnArch, *, encoding_name: str = GRID_ENCODING,
                         run_id: str = "runa", corpus_sha256: str | None = None
                         ) -> dict[str, Any]:
    """The metadata stamp inputs. `created_utc`/`commit_sha` are stamped ONCE by
    save_checkpoint (NOT supplied here — supplying them is the restamp error, T-CK-10)."""
    mk: dict[str, Any] = {"encoding_name": encoding_name, "run_id": run_id, "arch": arch}
    if corpus_sha256 is not None:
        mk["corpus_sha256"] = corpus_sha256
    return mk


@pytest.fixture
def metadata_kwargs(tiny_arch: CnnArch) -> dict[str, Any]:
    return make_metadata_kwargs(tiny_arch)


# ── factory fixtures (callables, so a test can vary encoding/run_id without importing
#    conftest by name — R5/LAW-17 keeps the collection style import-hack-free) ─────────────
@pytest.fixture
def mk_config():
    return make_run_config


@pytest.fixture
def mk_meta():
    return make_metadata_kwargs


@pytest.fixture
def mk_optim():
    return make_optim_scaler_sched


# ── committed goldens (manifest-tracked) ─────────────────────────────────────────────────
@pytest.fixture(scope="session")
def resume_goldens() -> dict[str, Any]:
    return json.loads((TRAIN_FIXTURES / "resume_goldens.json").read_text())


@pytest.fixture(scope="session")
def legacy_shapes() -> dict[str, Any]:
    return json.loads((TRAIN_FIXTURES / "legacy_payload_shapes.json").read_text())


@pytest.fixture(scope="session")
def anchor_key_set() -> set[str]:
    return set(ANCHOR_KEYS_FILE.read_text().split())
