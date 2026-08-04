# R8 >300 justify (402, MEASURED by `wc -l`; was 330 at WP12-R Phase O, 293 before it —
# SF-7): the file builds and emits BOTH per-boundary payloads (`training_step` and
# `iteration_complete`). WP12R Step 3 narration (R210) SPLIT the builder into
# `emit_training_step_event` + `emit_iteration_complete_event` so the two halves can live on
# different cadences (training_step stays `log_interval`-gated; iteration_complete emits per
# coordinator step), keeping the OLD combined `emit_training_events` as a thin wrapper for the
# `tests/train/test_target_counter_events.py:486` signature pin. The 4 WARN rules run on the
# `training_step` payload that was actually emitted, so a rule can never fire on a shape the
# event stream does not carry (LAW-07 — the alert and its producer are the same object).
# Phase O adds the `target_integrity` block; WP12R R218 rider 1 adds the `Q-O-TWO-POOL-READS`
# collapse (`rstats` passed into `emit_iteration_complete_event` so it does NOT make its own
# `pool.runner_stats()` call — ONE atomic snapshot, the straddle ELIMINATED).
"""Training-loop event BUILDERS (WP10 §a.3 IMPROVE — route through the injected EventSink).

Ported behaviour-exact from the old `training/events.py`, but every payload is funnelled
through the injected `EventSink.emit` instead of importing `monitoring.events.emit_event` —
the DAG stays clean (no `train → monitor` hard edge). Payload SHAPES are unchanged. The
probe collaborators (`gpu_monitor`/`early_game_probe`/`tb_writer`) are duck-typed (`Any`)
so this module carries no top-level monitor import; the real alert-RULE evaluation +
LAW-07 producer tests are WP13. `pool` is typed by `PoolTelemetryLike` below (WPCLEAN
Phase PC, R106).
"""
from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from mantis.train.axis_distribution import compute_axis_fractions
from mantis.train.emit import EventSink, emit_via

_LOG = logging.getLogger(__name__)


@runtime_checkable
class PoolTelemetryLike(Protocol):
    """The narrow READ-side pool surface this module consumes (WPCLEAN Phase PC, R106 —
    the CENSUS_C C-6/C-7 nine, completed against `selfplay/pool.py`'s concretes).

    Deliberately NOT folded into `coordinator.config.WorkerPoolLike` (the R106 design
    question, decided with grounds): the coordinator's pool reads are load-bearing control
    flow (watchdog arm / health abort / draw-rate gate) while these nine are emission-only,
    and their failure postures differ — an absent
    operational member is an abort-integrity bug, an absent telemetry member breaks a
    payload (`avg_game_length` is even hasattr-guarded to 0.0, ported behaviour). Defined
    HERE rather than in `coordinator/config.py` because the seam layer describes the shape
    it consumes (the `DrawRateAbortLike` precedent) and because `coordinator/__init__`
    imports `step`, which imports this module — the reverse import would cycle. Pinned by
    `tests/train/test_trainer_seam_conformance.py`'s widened matrix.

    WP12-R Phase O correction (SF-7 — the sentence above used to end "the two member sets
    are disjoint", and R164 made that FALSE rather than leaving it to rot): `runner_stats`
    is now declared on BOTH protocols, because the coordinator reads the same snapshot to
    build `iteration_complete`'s `target_integrity` block. The SPLIT still stands on its
    other two grounds — different failure postures and different holders — and one shared
    read is not a merge; what would be a merge is folding the other eight emission-only
    members into the control-flow protocol.
    """

    gumbel_mcts: bool
    avg_game_length: float
    x_winrate: float
    o_winrate: float
    draws: int
    sims_per_sec: float
    batch_fill_pct: float
    recent_move_histories: list[list[tuple[int, int]]]

    def runner_stats(self) -> Any: ...  # RunnerStats — Any keeps the no-`train → selfplay` edge

# The old monitoring.early_game_probe threshold, inlined. The PROBE itself is DEFER/ARCH
# (file_map row 64; the F-27/F-30 class — a probe is never a run-gate: the value-spread
# canary stayed green through a 33%→5% WR collapse). Re-entry requires a LAW-02
# re-validation, not a wiring commit. Only the numeric gate is needed here to preserve the
# warn-log behaviour for a duck-typed probe a caller may still inject.
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
    pool: PoolTelemetryLike,
    monitor_cfg: Any,
    baseline: dict[str, float],
    tb_writer: Any,
    sink: EventSink,
) -> float | None:
    """Compute + emit selfplay axis-distribution metrics through the injected sink.

    `monitor_cfg` is a `MonitorConfig`-like (duck-typed — `train/events.py` is NOT one of
    the three declared `train → monitor` import sites): the warn/alert thresholds have ONE
    authority, the dataclass field. The old inline
    `config.get("monitors", {}).get("axis_warn", 0.45)` code-side defaults are DEAD — a
    duplicated default authority is exactly what §5/R1 forbids.
    """
    recent_games = pool.recent_move_histories
    if not recent_games:
        return None

    metrics = compute_axis_fractions(recent_games)
    axis_q, axis_r, axis_s = metrics["axis_q"], metrics["axis_r"], metrics["axis_s"]
    axis_max = metrics["axis_max"]

    axis_warn = float(monitor_cfg.axis_warn)
    axis_alert = float(monitor_cfg.axis_alert)
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


def emit_training_step_event(
    train_step: int,
    loss_info: dict[str, float],
    qfire_delta: int | None,
    sink: EventSink,
    early_game_probe: Any | None = None,
    trainer_model: Any | None = None,
    solver_deltas: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build + emit the `training_step` event (WP13-A §c.4) and RETURN its payload.

    Split out of `emit_training_events` (WP12R Step 3 narration, R210): `training_step` is
    the per-`log_interval`-boundary payload the 4 WARN rules read
    (`monitor.rules.emit_training_step_alerts`). The rules run on the payload that was
    actually emitted, so a rule can never fire on a shape the event stream does not carry
    (LAW-07 — the alert and its producer are the same object). Stays `log_interval`-gated
    at the coordinator call site (R210: "training_step alerting stays gated").
    """
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
        # None = NOT MEASURED (no quiescence-counter producer exists new-side; the
        # solver-delta half is DEFER/ARCH). The key stays for schema stability, but it
        # must never carry a fabricated 0 — see docs/contracts/event_manifest.md.
        "quiescence_fires_per_step": qfire_delta,
    }
    if probe_metrics:
        training_step_event.update(probe_metrics)
    if solver_deltas:
        training_step_event.update(solver_deltas)
    emit_via(sink, training_step_event)
    return training_step_event


def emit_iteration_complete_event(
    train_step: int,
    w_pre: float,
    games_played: int,
    last_iter_games: int,
    pool: PoolTelemetryLike,
    buffer: Any,
    config: dict[str, Any],
    mcts_config: dict[str, Any],
    capacity: int,
    games_per_hour_fn: Any,
    steps_per_hour_fn: Any | None,
    target_integrity: Mapping[str, Any],
    rstats: Any,
    sink: EventSink,
) -> None:
    """Build + emit the `iteration_complete` event (WP13-A §c.4, WP12-R Phase O).

    Split out of `emit_training_events` (WP12R Step 3 narration, R210): `iteration_complete`
    is the per-iteration counter payload (`games_total`, `buffer_size`,
    `corpus_selfplay_frac`, `batch_fill_pct`, the `target_integrity` block). R210: "games_total
    is a per-iteration counter, not a training-logging event" — emitted per coordinator step
    (per burst), NOT `log_interval`-gated.

    `rstats` (R218 rider 1, `Q-O-TWO-POOL-READS` collapse): the `RunnerStats` snapshot from
    `StepCoordinator._target_integrity_report`, passed IN so this builder does NOT make its
    own `pool.runner_stats()` call. The collapse ELIMINATES the straddle — the
    `target_integrity` block and the `mcts_mean_depth`/cluster block become ONE atomic read
    instead of two microseconds-apart reads that could straddle a game boundary. This is a
    SEMANTIC CHANGE (more correct, not a no-op): the two blocks are now guaranteed-consistent
    on one snapshot.
    """
    gph = games_per_hour_fn()
    avg_gl = pool.avg_game_length if hasattr(pool, "avg_game_length") else 0.0
    pph = gph * avg_gl if avg_gl > 0 else 0.0
    _puct_regime = not pool.gumbel_mcts
    iteration_complete_event: dict[str, Any] = {
        "event": "iteration_complete",
        "step": train_step,
        "games_total": games_played,
        "games_this_iter": games_played - last_iter_games,
        "games_per_hour": round(gph, 1),
        # R29 gap metric (b): the coordinator's own step rate over the same clock as (a).
        # None = NOT MEASURED (no producer injected), never a fabricated 0 — the same
        # doctrine as `quiescence_fires_per_step`.
        "steps_per_hour": (round(float(steps_per_hour_fn()), 1)
                           if steps_per_hour_fn is not None else None),
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
        # WP12-R Phase O (R164/LAW-18): the three Phase-T target-integrity counters reach
        # the ONE channel here, each as {total, delta, per_position} beside the
        # `positions_delta` denominator the rate is taken over. Nested so the three travel
        # together and cannot crosswire; built by the coordinator, which owns the previous
        # boundary's readings (`StepCoordinator._target_integrity_report`).
        "target_integrity": dict(target_integrity),
    }
    iteration_complete_event.update(regime_gated_cluster_stats(rstats, _puct_regime))
    emit_via(sink, iteration_complete_event)


def emit_training_events(
    train_step: int,
    loss_info: dict[str, float],
    w_pre: float,
    games_played: int,
    last_iter_games: int,
    pool: PoolTelemetryLike,
    buffer: Any,
    gpu_monitor: Any,
    config: dict[str, Any],
    mcts_config: dict[str, Any],
    capacity: int,
    games_per_hour_fn: Any,
    qfire_delta: int | None,
    sink: EventSink,
    early_game_probe: Any | None = None,
    trainer_model: Any | None = None,
    solver_deltas: dict[str, Any] | None = None,
    steps_per_hour_fn: Any | None = None,
    *,
    target_integrity: Mapping[str, Any],
) -> dict[str, Any]:
    """Emit `training_step` + `iteration_complete` events through the injected sink and
    RETURN the `training_step` payload.

    RETAINED as a thin wrapper (WP12R Step 3 narration, R210): the production coordinator
    now calls `emit_training_step_event` + `emit_iteration_complete_event` directly (the
    two halves live on different cadences after R210 — `training_step` stays
    `log_interval`-gated, `iteration_complete` emits per coordinator step). This wrapper
    keeps the OLD combined signature so `tests/train/test_target_counter_events.py:486`'s
    `inspect.signature(emit_training_events).parameters` assertion stays green. It has no
    production caller after the split; retiring it is a wider change out of scope here.

    `target_integrity` (WP12-R Phase O, R164) is REQUIRED, keyword-only and carries NO
    default, and that is the whole difference from `solver_deltas` two lines above — a
    defaulted parameter with zero callers passing it, whose payload keys therefore silently
    never appear and whose absence no test can see. A parameter default is a MIGRATED
    authority, not an absent one (`run.py:366-372`, MF-2 Attack B): with no default, a
    caller that forgets it is a `TypeError` at the first `log_interval` boundary, loudly.
    (`solver_deltas` itself is left byte-untouched here — its fix needs a semantics decision
    about denominators and about the documented-unproduced `quiescence_fires_per_step`, and
    taking that inside this commit would be the scope widening R119 forbids.)

    The return is what the 4 WARN rules read (`monitor.rules.emit_training_step_alerts`):
    the rules run on the payload that was actually emitted, so a rule can never fire on a
    shape the event stream does not carry (LAW-07 — the alert and its producer are the
    same object)."""
    training_step_event = emit_training_step_event(
        train_step, loss_info, qfire_delta, sink,
        early_game_probe=early_game_probe, trainer_model=trainer_model,
        solver_deltas=solver_deltas,
    )
    rstats = pool.runner_stats()
    emit_iteration_complete_event(
        train_step, w_pre, games_played, last_iter_games, pool, buffer,
        config, mcts_config, capacity, games_per_hour_fn, steps_per_hour_fn,
        target_integrity, rstats, sink,
    )
    return training_step_event
