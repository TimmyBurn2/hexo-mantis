# R8 >300 justify: the file builds and emits BOTH per-boundary payloads (`training_step` and
# `iteration_complete`), split so the two halves can live on different cadences — training_step
# stays `log_interval`-gated, iteration_complete emits per coordinator step. The WARN rules run
# on the `training_step` payload that was actually emitted, so a rule can never fire on a shape
# the event stream does not carry: the alert and its producer are the same object.
"""Training-loop event BUILDERS — every payload routes through the injected `EventSink`.

Nothing imports the monitor package, so the DAG keeps no `train -> monitor` hard edge; the
probe collaborators are duck-typed for the same reason. `pool` is typed by
`PoolTelemetryLike` below.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from mantis.train.axis_distribution import compute_axis_fractions
from mantis.train.emit import EventSink, emit_via

_LOG = logging.getLogger(__name__)


@runtime_checkable
class PoolTelemetryLike(Protocol):
    """The narrow READ-side pool surface this module consumes. Deliberately NOT folded into the
    coordinator's `WorkerPoolLike`: those pool reads are load-bearing control flow while these are
    emission-only, and their failure postures differ. Defined HERE because `coordinator/__init__`
    imports `step`, which imports this module, so the reverse would cycle; `runner_stats` is on
    BOTH protocols because the coordinator reads the same snapshot, and one shared read is not a
    merge."""

    search_kind: str
    avg_game_length: float
    x_winrate: float
    o_winrate: float
    # The payload reads the SHARE, computed by the pool under its own lock against the same
    # `games_completed` the two win rates use. Declaring the raw count while consuming a
    # coordinator-side denominator is what let the two drift into a fraction above 1.
    draw_rate: float
    sims_per_sec: float
    batch_fill_pct: float
    # The batching instrument behind `batch_fill_pct`'s single ratio. Optional on the SOURCE,
    # not on the payload: a source that does not produce it publishes `None`, which means "no
    # producer" and which a consumer must never read as zero.
    inference_batch_timing: Mapping[str, Any] | None
    # R349(c): `{rows, graph_rows, per_1000}` — the alpha = 1.0 count over graph rows pushed.
    alpha_full: Mapping[str, Any]
    recent_move_histories: list[list[tuple[int, int]]]

    def runner_stats(self) -> Any: ...  # RunnerStats — Any keeps the no-`train → selfplay` edge

# The old early-game-probe threshold, inlined. The PROBE itself is deferred — a probe is never
# a run-gate, since the value-spread canary stayed green through a 33% to 5% WR collapse — and
# re-entry needs a re-validation, not a wiring commit. Only the numeric gate is needed here to
# preserve the warn-log behaviour for a duck-typed probe a caller may still inject.
EARLY_GAME_ENTROPY_WARN_THRESHOLD: float = 4.5

#: The `trainer_step` key R347(a)'s per-row tail mass alpha travels under. One spelling
#: authority for the key.
GUMBEL_TAIL_MASS_KEY = "gumbel_tail_mass"


def tail_mass_block(values: Any) -> dict[str, Any]:
    """The in-run reading of the sparse Gumbel row's tail mass alpha.

    ALPHA IS THE PART OF THE TRAINING TARGET THE ROW DID NOT STORE: the trainer rebuilds it from
    its own detached current prior, so a run with alpha near 0 trains on stored targets while one
    near 1 trains almost entirely on its own prior — and nothing in the loss curve says which.
    Reported as the step's own DISTRIBUTION, because the mean of a bimodal alpha names neither
    mode. NO ROWS omits the key, since a keyed `None` would claim a producer that did not run;
    PUCT ROWS report a MEASURED zero.

    Args:
        values: the step's per-row alphas, any sequence of floats.

    Returns:
        `{GUMBEL_TAIL_MASS_KEY: {...}}`, or `{}` when there are no rows.
    """
    alphas = [float(v) for v in values]
    if not alphas:
        return {}
    alphas.sort()
    n = len(alphas)

    def _at(q: float) -> float:
        return alphas[min(n - 1, max(0, int(round((n - 1) * q))))]

    return {GUMBEL_TAIL_MASS_KEY: {
        "n_rows": n,
        "mean": sum(alphas) / n,
        "p50": _at(0.5),
        "p90": _at(0.9),
        "max": alphas[-1],
        "rows_with_tail": sum(1 for a in alphas if a > 0.0),
    }}


def regime_gated_cluster_stats(rstats: Any, puct_regime: bool) -> dict[str, Any]:
    """The PUCT-descent-specific root-concentration stat: a value under PUCT, `None` under Gumbel,
    so the `iteration_complete` schema is regime-STABLE. The cluster-variance fields it carried
    died with the dense search arm that produced them.

    Args:
        rstats: the runner-stats snapshot.
        puct_regime: whether the run's search kind is PUCT.

    Returns:
        The one-key block.
    """
    return {
        "mcts_root_concentration": rstats.mcts_mean_root_concentration if puct_regime else None
    }


def emit_axis_distribution(
    train_step: int,
    pool: PoolTelemetryLike,
    monitor_cfg: Any,
    baseline: dict[str, float],
    tb_writer: Any,
    sink: EventSink,
) -> float | None:
    """Compute and emit selfplay axis-distribution metrics through the injected sink.

    `monitor_cfg` is duck-typed, since this module is not one of the declared
    `train -> monitor` import sites: the warn/alert thresholds have ONE authority, the
    dataclass field, and the old inline `.get(..., 0.45)` code-side defaults are dead.
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

    # DEMOTED TO A METRIC on alert-fatigue grounds: the burst fired this on 193 of 193
    # emissions, and a warning that fires every time trains its reader to skip the channel. The
    # THRESHOLD is what is wrong (a minted 0.5 against a ~0.33 pigeonhole floor over three
    # axes), so the comparison is still MADE and PUBLISHED and only the log level moves —
    # deleting it would throw away the measurement the re-derivation needs.
    band = "alert" if max_frac >= axis_alert else ("warn" if max_frac >= axis_warn else "ok")
    if band != "ok":
        _LOG.info(
            "axis_distribution_%s: step=%d axis_max=%s max_frac=%.4f (>= %.2f, n_games=%d) — "
            "METRIC, not an alert (R343(e)): the threshold is under re-derivation",
            band, train_step, axis_max, max_frac,
            axis_alert if band == "alert" else axis_warn, len(recent_games),
        )

    emit_via(sink, {
        "event": "axis_distribution",
        "step": train_step,
        "axis_q": axis_q,
        "axis_r": axis_r,
        "axis_s": axis_s,
        "axis_max": axis_max,
        "n_games": len(recent_games),
        # The band the dashboard draws: the reading survives the demotion, the interruption
        # does not.
        "axis_band": band,
        "axis_warn_threshold": axis_warn,
        "axis_alert_threshold": axis_alert,
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


def measured(loss_info: Mapping[str, Any], key: str) -> float | None:
    """An absent loss key is NOT MEASURED (`None`), never a fabricated number. What that costs
    when it is not applied: `policy_entropy` defaulted to `0.0` with no producer, so the collapse
    rule fired at every `log_interval` of every run while masking a real collapse behind identical
    text. A fabricated default is what makes an absent arm unreachable."""
    value = loss_info.get(key)
    return None if value is None else float(value)


def measured_int(loss_info: Mapping[str, Any], key: str) -> int | None:
    """`measured` for the integer counters — absence is `None`, not a count of zero."""
    value = loss_info.get(key)
    return None if value is None else int(value)


def emit_training_step_event(
    train_step: int,
    loss_info: dict[str, float],
    qfire_delta: int | None,
    sink: EventSink,
    early_game_probe: Any | None = None,
    trainer_model: Any | None = None,
    solver_deltas: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and emit the `training_step` event and RETURN its payload.

    The per-`log_interval`-boundary payload the WARN rules read. The rules run on the payload
    that was actually emitted, so a rule can never fire on a shape the event stream does not
    carry. Stays `log_interval`-gated at the coordinator call site.
    """
    policy_entropy = measured(loss_info, "policy_entropy")
    value_accuracy = measured(loss_info, "value_accuracy")
    grad_norm = measured(loss_info, "grad_norm")
    lr = measured(loss_info, "lr")

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
        # Every row below is `None` when its producer did not supply it. The three
        # `policy_entropy_*` rows travelled as JSON `NaN` — not valid JSON at all — and the
        # rest as `0.0`/`0`, indistinguishable from a measured zero.
        "loss_aux": measured(loss_info, "opp_reply_loss"),
        "loss_ownership": measured(loss_info, "ownership_loss"),
        "loss_threat": measured(loss_info, "threat_loss"),
        "loss_chain": measured(loss_info, "chain_loss"),
        "avg_sigma": measured(loss_info, "avg_sigma"),
        "policy_entropy": policy_entropy,
        "policy_entropy_pretrain": measured(loss_info, "policy_entropy_pretrain"),
        "policy_entropy_selfplay": measured(loss_info, "policy_entropy_selfplay"),
        "policy_entropy_recent": measured(loss_info, "policy_entropy_recent"),
        "policy_target_entropy": measured(loss_info, "policy_target_entropy"),
        "n_rows_policy_loss": measured_int(loss_info, "n_rows_policy_loss"),
        "n_rows_total": measured_int(loss_info, "n_rows_total"),
        "value_accuracy": value_accuracy,
        "lr": lr,
        "grad_norm": grad_norm,
        # None = NOT MEASURED: no quiescence-counter producer exists. The key stays for
        # schema stability but must never carry a fabricated 0.
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
    """Build and emit `iteration_complete` — the per-iteration counter payload, emitted per
    coordinator step and NOT `log_interval`-gated, because games_total is a per-iteration counter
    rather than a training-logging event. `rstats` is passed IN so this builder makes no
    `pool.runner_stats()` call of its own, which eliminates the straddle: both blocks read ONE
    atomic snapshot instead of two taken microseconds apart."""
    # Each of the three is `None` when its inputs were not measured: a rate over zero elapsed
    # time, a mean over zero completed games, and a product of either are all absences, and
    # each used to be published as a hard `0.0`, which reads as a measured stall.
    gph = games_per_hour_fn()
    avg_gl = getattr(pool, "avg_game_length", None)
    pph = (gph * avg_gl) if (gph is not None and avg_gl is not None and avg_gl > 0) else None
    _puct_regime = pool.search_kind == "puct"
    iteration_complete_event: dict[str, Any] = {
        "event": "iteration_complete",
        "step": train_step,
        "games_total": games_played,
        "games_this_iter": games_played - last_iter_games,
        "games_per_hour": round(gph, 1) if gph is not None else None,
        # The coordinator's own step rate over the same clock. None = NOT MEASURED (no
        # producer injected), never a fabricated 0.
        "steps_per_hour": (round(float(steps_per_hour_fn()), 1)
                           if steps_per_hour_fn is not None else None),
        "positions_per_hour": round(pph, 1) if pph is not None else None,
        "avg_game_length": round(avg_gl, 1) if avg_gl is not None else None,
        "win_rate_p0": round(float(pool.x_winrate), 4),
        "win_rate_p1": round(float(pool.o_winrate), 4),
        # Read the SHARE off the pool, not `pool.draws / games_played`: the old form paired a
        # live numerator with a snapshot frozen near the top of `step()` while the feeder kept
        # draining, and emitted 1.3333 / 1.5 / 1.125 on the shakedown burn — a fraction above 1.
        # The three outcome shares now share a denominator and sum to 1.
        "draw_rate": round(float(pool.draw_rate), 4),
        # `or 0.0` turned the not-yet-measured `None` into a measured zero, which is the whole
        # finding in one operator.
        "sims_per_sec": pool.sims_per_sec,
        "buffer_size": buffer.size,
        "buffer_capacity": buffer.capacity,
        "corpus_selfplay_frac": round(1.0 - w_pre, 4),
        "batch_fill_pct": pool.batch_fill_pct,
        # The inference batching instrument — the collector wait, the collate cost and the
        # served-batch occupancy DISTRIBUTION that `batch_fill_pct`'s single ratio cannot
        # resolve. Rides `iteration_complete` because that is already the seam for
        # inference-server stats and because it emits on neither interval knob. `None` = the
        # source has no producer for it, never a fabricated zero block.
        "inference_batching": getattr(pool, "inference_batch_timing", None),
        # R349(c): rows whose explicit entries carry no target mass, per 1,000 graph rows
        # pushed since boot. `None` = the source has no producer for it, never a zero.
        "gumbel_alpha_full": getattr(pool, "alpha_full", None),
        "mcts_mean_depth": rstats.mcts_mean_depth,
        # The target-integrity counters plus the SEAM conjunct of the same class reach the ONE
        # channel here, each as {total, delta, per_position} beside the `positions_delta`
        # denominator. Nested so they travel together and cannot crosswire; built by the
        # coordinator, which owns the previous boundary's readings.
        "target_integrity": dict(target_integrity),
    }
    iteration_complete_event.update(regime_gated_cluster_stats(rstats, _puct_regime))
    emit_via(sink, iteration_complete_event)


# THE PRE-SPLIT WRAPPER IS DELETED. `emit_training_events` stood here, kept as a thin wrapper
# only to satisfy one test's `inspect.signature` assertion — a signature pin on a shape
# production does not produce, which kept a second entry to the event stream alive. The pin
# moved to `emit_iteration_complete_event`, where `target_integrity` actually lives.

