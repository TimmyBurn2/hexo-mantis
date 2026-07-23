"""Training-loop event BUILDERS (WP10 §a.3 IMPROVE — route through the injected EventSink).

Ported behaviour-exact from the old `training/events.py`, but every payload is funnelled
through the injected `EventSink.emit` instead of importing `monitoring.events.emit_event` —
the DAG stays clean (no `train → monitor` hard edge). Payload SHAPES are unchanged. The
step-coordinator/probe collaborators (`pool`/`gpu_monitor`/`early_game_probe`/`tb_writer`)
are duck-typed (`Any`) so this module carries no top-level monitor import; the real
alert-RULE evaluation + LAW-07 producer tests are WP13.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from mantis.train.axis_distribution import compute_axis_fractions
from mantis.train.emit import EventSink, emit_via

_LOG = logging.getLogger(__name__)

# The old monitoring.early_game_probe threshold, inlined (the probe is WP13; only the numeric
# gate is needed here to preserve the warn-log behaviour).
EARLY_GAME_ENTROPY_WARN_THRESHOLD: float = 4.5

# CONFRES S2: PUCT-descent-specific cluster stats — always-keyed (value under PUCT, None under
# Gumbel) so the iteration_complete schema is regime-STABLE.
_REGIME_GATED_CLUSTER_STAT_KEYS = (
    "mcts_root_concentration",
    "cluster_value_std_mean",
    "cluster_policy_disagreement_mean",
    "cluster_variance_sample_count",
)


def regime_gated_cluster_stats(rstats: Any, puct_regime: bool) -> dict[str, Any]:
    """The PUCT-descent-specific cluster stats, always keyed: value under PUCT, `None` under
    Gumbel. Schema-stable (S2) — the keys are never dropped."""
    if not puct_regime:
        return {k: None for k in _REGIME_GATED_CLUSTER_STAT_KEYS}
    return {
        "mcts_root_concentration": rstats.mcts_mean_root_concentration,
        "cluster_value_std_mean": rstats.cluster_value_std_mean,
        "cluster_policy_disagreement_mean": rstats.cluster_policy_disagreement_mean,
        "cluster_variance_sample_count": rstats.cluster_variance_sample_count,
    }


def replay_pretrain_events(log_dir: str | Path, sink: EventSink) -> None:
    """Replay up to 500 pretrain `training_step` events into the sink on resume."""
    pretrain_log = Path(log_dir) / "pretrain.jsonl"
    if not pretrain_log.exists():
        return
    replay_evs: list[dict] = []
    try:
        with open(pretrain_log) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if d.get("event") == "train_step" and d.get("phase") == "pretrain":
                        replay_evs.append({
                            "event": "training_step",
                            "step": d.get("step"),
                            "loss_total": d.get("loss"),
                            "loss_policy": d.get("policy_loss"),
                            "loss_value": d.get("value_loss"),
                            "loss_aux": d.get("aux_opp_reply_loss"),
                            "policy_entropy": d.get("policy_entropy"),
                            "value_accuracy": d.get("value_accuracy"),
                            "lr": d.get("lr"),
                            "grad_norm": d.get("grad_norm"),
                            "corpus_mix": d.get("corpus_mix", {"pretrain": 1.0, "self_play": 0.0}),
                            "phase": "pretrain",
                        })
                except Exception:  # noqa: BLE001 — a malformed line must not abort replay
                    pass
    except Exception as e:  # noqa: BLE001
        _LOG.warning("pretrain_replay_failed: %s", e)
        return
    if replay_evs:
        _LOG.info("replaying_pretrain_events: count=%d", len(replay_evs[-500:]))
        for ev in replay_evs[-500:]:
            emit_via(sink, ev)


def emit_axis_distribution(
    train_step: int,
    pool: Any,
    config: dict[str, Any],
    baseline: dict[str, float],
    tb_writer: Any,
    sink: EventSink,
) -> Optional[float]:
    """Compute + emit selfplay axis-distribution metrics through the injected sink."""
    recent_games = pool.recent_move_histories
    if not recent_games:
        return None

    metrics = compute_axis_fractions(recent_games)
    axis_q, axis_r, axis_s = metrics["axis_q"], metrics["axis_r"], metrics["axis_s"]
    axis_max = metrics["axis_max"]

    mon_cfg = config.get("monitors", {})
    axis_warn = float(mon_cfg.get("axis_warn", 0.45))
    axis_alert = float(mon_cfg.get("axis_alert", 0.50))
    max_frac = max(axis_q, axis_r, axis_s)

    if max_frac >= axis_alert:
        _LOG.warning(
            "axis_distribution_alert: step=%d axis_max=%s max_frac=%.4f (>= %.2f, n_games=%d)",
            train_step, axis_max, max_frac, axis_alert, len(recent_games),
        )
    elif max_frac >= axis_warn:
        _LOG.warning(
            "axis_distribution_warn: step=%d axis_max=%s max_frac=%.4f (>= %.2f, n_games=%d)",
            train_step, axis_max, max_frac, axis_warn, len(recent_games),
        )

    emit_via(sink, {
        "event": "axis_distribution",
        "step": train_step,
        "axis_q": axis_q,
        "axis_r": axis_r,
        "axis_s": axis_s,
        "axis_max": axis_max,
        "n_games": len(recent_games),
    })

    if tb_writer is not None:
        tb_metrics: dict[str, float] = {
            "axis_dist/axis_q": axis_q,
            "axis_dist/axis_r": axis_r,
            "axis_dist/axis_s": axis_s,
        }
        for label in ("axis_q", "axis_r", "axis_s"):
            if label in baseline:
                tb_metrics[f"axis_dist_delta/{label}"] = metrics[label] - baseline[label]
        try:
            tb_writer.log_step(train_step, tb_metrics)
        except Exception as _tb_err:  # noqa: BLE001
            _LOG.warning("axis_distribution_tb_failed: step=%d error=%s", train_step, _tb_err)

    return axis_q


def emit_training_events(
    train_step: int,
    loss_info: dict[str, float],
    w_pre: float,
    games_played: int,
    last_iter_games: int,
    pool: Any,
    buffer: Any,
    gpu_monitor: Any,
    config: dict[str, Any],
    mcts_config: dict[str, Any],
    capacity: int,
    games_per_hour_fn: Any,
    qfire_delta: int,
    sink: EventSink,
    early_game_probe: Optional[Any] = None,
    trainer_model: Optional[Any] = None,
    solver_deltas: Optional[dict[str, Any]] = None,
) -> None:
    """Emit `training_step` + `iteration_complete` events through the injected sink."""
    policy_entropy = float(loss_info.get("policy_entropy", 0.0))
    value_accuracy = float(loss_info.get("value_accuracy", 0.0))
    grad_norm = float(loss_info.get("grad_norm", float("nan")))
    lr = float(loss_info.get("lr", 0.0))

    probe_metrics: dict[str, Any] = {}
    if early_game_probe is not None and trainer_model is not None:
        try:
            probe_metrics = early_game_probe.compute(trainer_model)
            if probe_metrics["early_game_entropy_mean"] > EARLY_GAME_ENTROPY_WARN_THRESHOLD:
                _LOG.warning(
                    "early_game_entropy_high: step=%d entropy_mean=%.4f (>= %.2f)",
                    train_step, probe_metrics["early_game_entropy_mean"],
                    EARLY_GAME_ENTROPY_WARN_THRESHOLD,
                )
        except Exception as _egp_err:  # noqa: BLE001
            _LOG.warning("early_game_probe_failed: step=%d error=%s", train_step, _egp_err)
            probe_metrics = {}

    training_step_event: dict[str, Any] = {
        "event": "training_step",
        "step": train_step,
        "loss_total": float(loss_info["loss"]),
        "loss_policy": float(loss_info["policy_loss"]),
        "loss_value": float(loss_info["value_loss"]),
        "loss_aux": float(loss_info.get("opp_reply_loss", 0.0)),
        "loss_ownership": float(loss_info.get("ownership_loss", 0.0)),
        "loss_threat": float(loss_info.get("threat_loss", 0.0)),
        "loss_chain": float(loss_info.get("chain_loss", 0.0)),
        "avg_sigma": float(loss_info.get("avg_sigma", 0.0)),
        "policy_entropy": policy_entropy,
        "policy_entropy_pretrain": float(loss_info.get("policy_entropy_pretrain", float("nan"))),
        "policy_entropy_selfplay": float(loss_info.get("policy_entropy_selfplay", float("nan"))),
        "policy_entropy_recent": float(loss_info.get("policy_entropy_recent", float("nan"))),
        "policy_target_entropy": float(loss_info.get("policy_target_entropy", 0.0)),
        "n_rows_policy_loss": int(loss_info.get("n_rows_policy_loss", 0)),
        "n_rows_total": int(loss_info.get("n_rows_total", 0)),
        "value_accuracy": value_accuracy,
        "lr": lr,
        "grad_norm": grad_norm,
        "quiescence_fires_per_step": qfire_delta,
    }
    if probe_metrics:
        training_step_event.update(probe_metrics)
    if solver_deltas:
        training_step_event.update(solver_deltas)
    emit_via(sink, training_step_event)

    gph = games_per_hour_fn()
    avg_gl = pool.avg_game_length if hasattr(pool, "avg_game_length") else 0.0
    pph = gph * avg_gl if avg_gl > 0 else 0.0
    rstats = pool.runner_stats()
    _puct_regime = not pool.gumbel_mcts
    iteration_complete_event: dict[str, Any] = {
        "event": "iteration_complete",
        "step": train_step,
        "games_total": games_played,
        "games_this_iter": games_played - last_iter_games,
        "games_per_hour": round(gph, 1),
        "positions_per_hour": round(pph, 1),
        "avg_game_length": round(avg_gl, 1),
        "win_rate_p0": round(float(pool.x_winrate), 4),
        "win_rate_p1": round(float(pool.o_winrate), 4),
        "draw_rate": round(float(pool.draws / games_played), 4) if games_played > 0 else 0.0,
        "sims_per_sec": pool.sims_per_sec or 0.0,
        "buffer_size": buffer.size,
        "buffer_capacity": buffer.capacity,
        "corpus_selfplay_frac": round(1.0 - w_pre, 4),
        "batch_fill_pct": pool.batch_fill_pct,
        "mcts_mean_depth": rstats.mcts_mean_depth,
    }
    iteration_complete_event.update(regime_gated_cluster_stats(rstats, _puct_regime))
    emit_via(sink, iteration_complete_event)
