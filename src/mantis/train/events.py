# R8 >300 justify: the file builds and emits BOTH per-boundary payloads (`training_step` and
# `iteration_complete`). WP12R Step 3 narration (R210) SPLIT the builder into
# `emit_training_step_event` + `emit_iteration_complete_event` so the two halves can live on
# different cadences (training_step stays `log_interval`-gated; iteration_complete emits per
# coordinator step). The OLD combined `emit_training_events` was kept as a thin wrapper for
# ONE test's signature pin and is DELETED (AUDIT-1 F-47) — the pin moved to the live builder.
# The 4 WARN rules run on the
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

import logging
from collections.abc import Mapping
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

    search_kind: str
    avg_game_length: float
    x_winrate: float
    o_winrate: float
    # F-816-2: was `draws: int`. The raw count is no longer reached from here — the payload
    # reads the SHARE, computed by the pool under its own lock against the same
    # `games_completed` the two win rates use. Declaring the count while consuming a
    # coordinator-side denominator is what let the two drift into a fraction above 1.
    draw_rate: float
    sims_per_sec: float
    batch_fill_pct: float
    # Q3 (LAW-18): the batching instrument behind `batch_fill_pct`'s single ratio.
    # Optional on the SOURCE, not on the payload — a telemetry source that does not
    # produce it publishes `None`, which the event-manifest convention defines as "no
    # producer" and which a consumer must never read as zero.
    inference_batch_timing: Mapping[str, Any] | None
    recent_move_histories: list[list[tuple[int, int]]]

    def runner_stats(self) -> Any: ...  # RunnerStats — Any keeps the no-`train → selfplay` edge

# The old monitoring.early_game_probe threshold, inlined. The PROBE itself is DEFER/ARCH
# (file_map row 64; the F-27/F-30 class — a probe is never a run-gate: the value-spread
# canary stayed green through a 33%→5% WR collapse). Re-entry requires a LAW-02
# re-validation, not a wiring commit. Only the numeric gate is needed here to preserve the
# warn-log behaviour for a duck-typed probe a caller may still inject.
EARLY_GAME_ENTROPY_WARN_THRESHOLD: float = 4.5

#: The `trainer_step` key R347(a)'s per-row tail mass alpha travels under. One spelling
#: authority for the key.
GUMBEL_TAIL_MASS_KEY = "gumbel_tail_mass"


def tail_mass_block(values: Any) -> dict[str, Any]:
    """R347(a)/LAW-18 — the in-run reading of the sparse Gumbel row's tail mass alpha.

    ALPHA IS THE PART OF THE TRAINING TARGET THE ROW DID NOT STORE. The trainer rebuilds it
    from its own detached current prior, so the reconstruction is exact only to the extent
    that prior still resembles the one that recorded the row — and alpha is how much of each
    target rides on that. A run with alpha near 0 is training on stored targets; a run with
    alpha near 1 is training almost entirely on its own prior, and nothing in the loss curve
    says which. This is the line that does, from step 0.

    Reported as the step's own DISTRIBUTION rather than a mean: the mean of a bimodal alpha
    (early plies with a narrow legal set, late plies with a wide one) names neither mode.

    Three arms, matching the shape the sibling blocks carry:
      NO ROWS — the key is OMITTED. A step with no graph rows has no alpha, and a keyed
        `None` would claim a producer that did not run.
      PUCT ROWS — every alpha is 0.0 and the block says so truthfully; the arm stores no
        tail, and a zero here is a MEASURED zero, not an absence (R249).
      GUMBEL ROWS — the quantiles of the step's own alphas.

    Args:
        values: the step's per-row alphas, any sequence of floats (a numpy array, a list, or
            a concatenation of per-micro-batch slices).

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
    """The PUCT-descent-specific root-concentration stat: a value under PUCT, `None` under
    Gumbel (CONFRES S2, so the `iteration_complete` schema is regime-STABLE).

    The three cluster-variance fields it also carried are DELETED with the dense search arm
    that produced them (R346(f)): `mcts_root_concentration` is accumulated once per search in
    `play_one_move`, path-independently, which is why it is the one that survives.

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

    # R343(e) — DEMOTED TO A METRIC, and the ground is alert fatigue rather than a belief that
    # the number does not matter. The R342 burst fired `axis_distribution_alert` on 193 of 193
    # emissions: a warning that fires every time carries no information and trains its reader to
    # skip the whole channel, including the warnings that do. The THRESHOLD is what is wrong
    # (`run6.yaml` mints `axis_alert: 0.5`, and three axes make ~0.33 the floor by pigeonhole),
    # and re-deriving it needs the shakedown's own distribution — carded, `AUDIT-1 F-01`
    # cross-referenced. Until then the comparison is still MADE and still PUBLISHED, on the
    # event where a dashboard reads it; only the log level moves. Deleting the comparison would
    # have thrown away the measurement the re-derivation needs.
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
        # The band the dashboard draws. Carried on the event because that is what "demoted to a
        # dashboard metric" means: the reading survives the demotion, the interruption does not.
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
    """An absent loss key is NOT MEASURED (`None`), never a fabricated number.

    `docs/contracts/event_manifest.md`'s unproduced-field convention. AUDIT-1 F-01 is what
    it costs when the convention is not applied here: `policy_entropy` defaulted to `0.0`,
    no trainer tail produces the key, and `check_entropy_collapse` therefore fired
    `entropy_collapse` at every `log_interval` of every run — while masking a real collapse
    behind identical text. Every WARN rule already has an absent arm; a fabricated default
    is what made those arms unreachable.
    """
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
    """Build + emit the `training_step` event (WP13-A §c.4) and RETURN its payload.

    Split out of `emit_training_events` (WP12R Step 3 narration, R210): `training_step` is
    the per-`log_interval`-boundary payload the 4 WARN rules read
    (`monitor.rules.emit_training_step_alerts`). The rules run on the payload that was
    actually emitted, so a rule can never fire on a shape the event stream does not carry
    (LAW-07 — the alert and its producer are the same object). Stays `log_interval`-gated
    at the coordinator call site (R210: "training_step alerting stays gated").
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
        # `policy_entropy_*` rows travelled as JSON `NaN` — which is not valid JSON at all —
        # and the rest as `0.0`/`0`, indistinguishable from a measured zero (AUDIT-1 F-01,
        # F-28 rows INST-C01/C02/C03).
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
    # AUDIT-1 F-28/C07. Each of the three is `None` when its inputs were not measured, on
    # the same doctrine `steps_per_hour` below already states. A rate over zero elapsed time,
    # a mean over zero completed games, and a product of either, are all absences — and each
    # of them used to be published as a hard `0.0`, which reads as a measured stall.
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
        # R29 gap metric (b): the coordinator's own step rate over the same clock as (a).
        # None = NOT MEASURED (no producer injected), never a fabricated 0 — the same
        # doctrine as `quiescence_fires_per_step`.
        "steps_per_hour": (round(float(steps_per_hour_fn()), 1)
                           if steps_per_hour_fn is not None else None),
        "positions_per_hour": round(pph, 1) if pph is not None else None,
        "avg_game_length": round(avg_gl, 1) if avg_gl is not None else None,
        "win_rate_p0": round(float(pool.x_winrate), 4),
        "win_rate_p1": round(float(pool.o_winrate), 4),
        # F-816-2: read the SHARE off the pool, not `pool.draws / games_played`. The old
        # form paired a live numerator with `_games_played`, a snapshot frozen near the top
        # of `step()` while the feeder kept draining, and emitted 1.3333 / 1.5 / 1.125 on the
        # shakedown burn — a fraction above 1. Same straddle class as R218 rider 1, one
        # payload over. The three outcome shares now share a denominator and sum to 1.
        "draw_rate": round(float(pool.draw_rate), 4),
        # `or 0.0` turned the not-yet-measured `None` into a measured zero, which is the
        # whole finding in one operator.
        "sims_per_sec": pool.sims_per_sec,
        "buffer_size": buffer.size,
        "buffer_capacity": buffer.capacity,
        "corpus_selfplay_frac": round(1.0 - w_pre, 4),
        "batch_fill_pct": pool.batch_fill_pct,
        # Q3 (LAW-18): the inference batching instrument — the collector wait, the
        # collate cost and the served-batch occupancy DISTRIBUTION that `batch_fill_pct`'s
        # single ratio cannot resolve. Rides `iteration_complete` because that is already
        # the established seam for inference-server stats (`batch_fill_pct` itself) and
        # because it emits per coordinator step, on neither interval knob. `None` = the
        # telemetry source has no producer for it (event_manifest unproduced-field
        # convention), never a fabricated zero block.
        "inference_batching": getattr(pool, "inference_batch_timing", None),
        "mcts_mean_depth": rstats.mcts_mean_depth,
        # WP12-R Phase O (R164/LAW-18): the three Phase-T target-integrity counters — plus
        # R275(b)'s `inference_failures_total`, the SEAM conjunct of the same class — reach
        # the ONE channel here, each as {total, delta, per_position} beside the
        # `positions_delta` denominator the rate is taken over. Nested so the three travel
        # together and cannot crosswire; built by the coordinator, which owns the previous
        # boundary's readings (`StepCoordinator._target_integrity_report`).
        "target_integrity": dict(target_integrity),
    }
    iteration_complete_event.update(regime_gated_cluster_stats(rstats, _puct_regime))
    emit_via(sink, iteration_complete_event)


# ══ THE PRE-SPLIT WRAPPER IS DELETED (AUDIT-1 F-47) ═════════════════════════════════════
# `emit_training_events` stood here — the combined `training_step` + `iteration_complete`
# emitter that R210 split in two, because the halves run on DIFFERENT cadences
# (`training_step` stays `log_interval`-gated; `iteration_complete` emits per coordinator
# step). It was RETAINED as a thin wrapper with a stated reason: to keep one test's
# `inspect.signature(emit_training_events)` assertion green. Its own docstring said so.
#
# That is a signature pin on a shape production does not produce, and it kept a second entry
# to the event stream alive to satisfy it. The pin moved to `emit_iteration_complete_event`,
# where `target_integrity` actually lives, and the wrapper went. The two builders above are
# the emitters; `coordinator/step.py` has called them directly since the split.

