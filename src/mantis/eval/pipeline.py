"""EvalPipeline — async kick + persistent poller + bounded drains (design §a.3/§c.3).

Out-of-process eval inference ONLY: no `device`/`model` kwargs, so an in-process CUDA path is
unrepresentable; the worker is spawned under `get_context("spawn")` and every join is
timeout-bounded (isolation laws 1 + 2). The ONE poller thread beats `heartbeat("eval_round")`
every tick with or without an in-flight round, so a between-round gap cannot false-fire the
watchdog; round PROGRESS is bounded separately by `round_timeout_sec`.

>300 justify: one isolation-law seam sharing one in-flight round record and one mailbox —
splitting kick from poller from drain would scatter that state machine.
"""
from __future__ import annotations

import json
import logging
import math
import multiprocessing
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mantis.bots.resolve import SKIP_REASON_MARKERS
from mantis.config.resolve.eval_posture import (
    resolve_ply_cap_adjudication,
    resolve_strength_floor,
)
from mantis.config.resolve.fused_graph_caps import FusedGraphCapsSpec
from mantis.config.resolve.inference_batching import InferenceBatchingSpec
from mantis.eval.bt import fit_bt, predict_p
from mantis.eval.channel_health import RoundReading as ChannelRoundReading
from mantis.eval.channel_health import assess as assess_channel
from mantis.eval.child_memory import EVENT as EVAL_DEVICE_MEMORY_EVENT
from mantis.eval.errors import EvalBrokenReason, LadderStateError, ResultContractError
from mantis.eval.ladder import LadderState
from mantis.eval.promote import DeployTagHooks, apply_gate_decision
from mantis.eval.rounds import (
    GameRecordTarget,
    GateSpec,
    RoundSpec,
    RungJob,
    build_round_result,
    validate_worker_result,
)
from mantis.eval.snapshot import write_model_snapshot

_LOG = logging.getLogger(__name__)

#: The poller thread's fixed tick — small enough that an idle beat is observable in a few
#: tens of milliseconds, cheap enough to run for a whole round's life.
_POLL_TICK_SEC = 0.02

#: Mirrors `mantis.config.schema._EVAL_TIMEOUT_CEILING_SEC`: `Process.join` raises
#: `OverflowError` on a non-finite timeout, so every join here bounds its own regardless.
_JOIN_TIMEOUT_CEILING_SEC = 86400.0


def _bounded_join_timeout(timeout: float) -> float:
    """Clamp `timeout` to a finite, non-negative value `Process.join()` can always accept.
    Non-finite values collapse to the one-day ceiling. Never raises."""
    if not math.isfinite(timeout):
        return _JOIN_TIMEOUT_CEILING_SEC
    return max(0.0, min(timeout, _JOIN_TIMEOUT_CEILING_SEC))


@dataclass(frozen=True)
class DrainCaps:
    """The 4 drain-cap fields lifted from `StepCoordinatorConfig`; each gains a live consumer
    here."""

    final_eval_drain_timeout_sec: float
    eval_final_drain_safety_factor: float
    eval_final_drain_hard_cap_sec: float
    terminal_eval_hard_cap_sec: float


def drain_budget_sec(caps: DrainCaps) -> float:
    """`min(final_eval_drain_timeout_sec * eval_final_drain_safety_factor,
    eval_final_drain_hard_cap_sec)` — the mid-run/teardown `drain_pending` bound."""
    return min(
        caps.final_eval_drain_timeout_sec * caps.eval_final_drain_safety_factor,
        caps.eval_final_drain_hard_cap_sec,
    )


def drain_or_kill(
    proc: Any, *, budget_sec: float, worker_kill_grace_sec: float, clock: Callable[[], float]
) -> EvalBrokenReason | None:
    """Bounded join -> terminate -> bounded join -> kill -> bounded join."""
    del clock  # the caller advances/consults its own clock; every join below is bounded
    proc.join(_bounded_join_timeout(budget_sec))
    if not proc.is_alive():
        return None
    proc.terminate()
    proc.join(_bounded_join_timeout(worker_kill_grace_sec))
    proc.kill()
    proc.join(_bounded_join_timeout(worker_kill_grace_sec))
    return EvalBrokenReason.JOIN_TIMEOUT


def _emit(sink: Any, payload: Mapping[str, Any]) -> None:
    if sink is not None:
        sink.emit(dict(payload))


def emit_round_started(
    sink: Any, *, round_id: str, step: int, scheduled: Mapping[str, int],
    gate_scheduled: bool, ts: float,
) -> dict[str, Any]:
    payload = {
        "event": "eval_round_started", "round_id": round_id, "step": step,
        "scheduled": dict(scheduled), "gate_scheduled": bool(gate_scheduled), "ts": ts,
    }
    _emit(sink, payload)
    return payload


def emit_round_complete(
    sink: Any, *, round_id: str, step: int, wall_sec: float, games_total: int | None,
    promoted: bool | None, wr_sealbot: float | None, progress: dict[str, Any] | None = None,
    wr_sealbot_ci_lower: float | None = None, wr_sealbot_ci_upper: float | None = None,
) -> dict[str, Any]:
    """Build the round-complete payload; `games_total` is `None` on a broken round."""
    payload = {
        "event": "eval_round_complete", "round_id": round_id, "step": step,
        "wall_sec": wall_sec, "games_total": games_total, "promoted": promoted,
        "wr_sealbot": wr_sealbot, "progress": progress,
        "wr_sealbot_ci_lower": wr_sealbot_ci_lower,
        "wr_sealbot_ci_upper": wr_sealbot_ci_upper,
    }
    _emit(sink, payload)
    return payload


def emit_round_skipped_busy(sink: Any, *, step: int, in_flight_round_id: str) -> dict[str, Any]:
    payload = {"event": "eval_round_skipped_busy", "step": step, "in_flight_round_id": in_flight_round_id}
    _emit(sink, payload)
    return payload


def emit_strength_floor(
    sink: Any, *, round_id: str, step: int, floor: Mapping[str, Any],
    checked_total: int, skipped_total: int,
) -> dict[str, Any]:
    """The strength floor's ONE event — the probe's verdict AND its in-run fire rate."""
    payload = {
        "event": "eval_strength_floor", "round_id": round_id, "step": step,
        **dict(floor), "checked_total": checked_total, "skipped_total": skipped_total,
    }
    _emit(sink, payload)
    return payload


def emit_ply_cap_adjudication(
    sink: Any, *, round_id: str, step: int, adjudication: Mapping[str, Any],
) -> dict[str, Any]:
    """The ply-cap criterion's in-run fire rate, one event per armed round."""
    payload = {
        "event": "eval_ply_cap_adjudication", "round_id": round_id, "step": step,
        **dict(adjudication),
    }
    _emit(sink, payload)
    return payload


#: The CLOSED skip-reason partition, in the resolver's own order. A reason nothing recognises
#: is a loud failure, never a fifth bucket invented at emission time.
SKIP_REASON_CLASSES: tuple[str, ...] = (
    "operator_authorized", "vendor_absent", "build_absent", "load_failed",
)


def _classify_skip_reason(reason: str) -> str | None:
    """The reason's class, or None when nothing recognises it."""
    if set(SKIP_REASON_MARKERS) != set(SKIP_REASON_CLASSES):
        raise ResultContractError(
            f"the skip-class partition disagrees with its markers: classes "
            f"{sorted(SKIP_REASON_CLASSES)} vs markers {sorted(SKIP_REASON_MARKERS)}. Two "
            f"authorities for one closed set is exactly what the marker mapping exists to "
            f"prevent, so the disagreement is fatal rather than resolved in favour of either."
        )
    matched = [name for name, marker in SKIP_REASON_MARKERS.items() if marker in reason]
    if len(matched) != 1:
        return None
    return matched[0]


def emit_device_memory(
    sink: Any, *, round_id: str, step: int, device_memory: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish the CHILD's own device-memory readout, from the child's payload."""
    payload = {
        "event": EVAL_DEVICE_MEMORY_EVENT, "round_id": round_id, "step": step,
        "device_memory": dict(device_memory),
    }
    _emit(sink, payload)
    return payload


def emit_rung_skip_events(round_id: str, skipped: list[Mapping[str, str]], sink: Any) -> None:
    """Per skipped rung: an `eval_rung_skipped` event, an ERROR log line, and one
    `eval_rung_skip_class` counter event carrying the running per-class count."""
    counts: dict[str, int] = dict.fromkeys(SKIP_REASON_CLASSES, 0)
    for entry in skipped:
        payload = {"event": "eval_rung_skipped", "round_id": round_id,
                   "rung": entry["rung"], "reason": entry["reason"]}
        _emit(sink, payload)
        _LOG.error("eval_rung_skipped round_id=%s rung=%s reason=%s",
                   round_id, entry["rung"], entry["reason"])

        reason_class = _classify_skip_reason(entry["reason"])
        if reason_class is None:
            _LOG.error(
                "eval_rung_skip_class_unclassified round_id=%s rung=%s reason=%s — no "
                "SKIP_REASON_MARKERS entry matches this reason, so it is counted in NO class; "
                "the partition is closed and a fifth bucket is not invented here",
                round_id, entry["rung"], entry["reason"],
            )
            continue
        counts[reason_class] += 1
        _emit(sink, {
            "event": "eval_rung_skip_class", "round_id": round_id, "rung": entry["rung"],
            "reason_class": reason_class, "class_count": counts[reason_class],
        })


def _result_tmp_path(result_path: str) -> Path:
    """The `.tmp` the worker writes for `result_path`, derived the way the worker derives it."""
    return Path(result_path + ".tmp")


def _remove_result_tmp(result_path: str) -> None:
    """Delete THIS round's `<result>.json.tmp` once its writer is gone.

    Removes only the `.tmp`, only for a child the caller has confirmed dead, and NEVER
    fatally: it runs in `_finalize_round`'s un-caught prologue, where a raise would kill
    the poller thread silently.
    """
    try:
        _result_tmp_path(result_path).unlink(missing_ok=True)
    except OSError:
        _LOG.debug("stale eval result tmp not removed: %s.tmp", result_path, exc_info=True)


def _drop_result_tmp_if_writer_gone(inflight: dict[str, Any]) -> None:
    """Remove one round's `.tmp` iff its worker is confirmed dead — the ONE decision, shared
    by `_finalize_round` and `stop()`."""
    spec = inflight.get("spec")
    proc = inflight.get("proc")
    if spec is None or proc is None or proc.is_alive():
        return
    _remove_result_tmp(spec.result_path)


def read_progress(spec: Any) -> dict[str, Any] | None:
    """The CHILD's last per-game progress row, or `None` if it wrote none."""
    path = getattr(spec, "progress_path", None)
    if not path:
        return None
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            return row
    return None


def _worker_entry(spec_path: str, result_path: str) -> None:
    """The spawn-ctx `Process` target (module-level so spawn can pickle-by-reference).

    Torch/worker imports stay LAZY, and the FIRST statement asks the kernel to kill this
    process when its parent dies — the path that once left a child holding 458 MiB.
    """
    from mantis.train.lifecycle.signals import arm_parent_death_signal

    arm_parent_death_signal()

    from mantis.eval.worker import worker_main

    worker_main(spec_path, result_path)


class EvalPipeline:
    """Satisfies `EvalPipelineLike` (coordinator/config.py:61-74) exactly."""

    def __init__(
        self,
        *,
        eval_cfg: Any,
        caps: DrainCaps,
        encoding: str,
        fused_graph_caps: FusedGraphCapsSpec | None,
        inference_batching: InferenceBatchingSpec | None,
        leaf_batch_size: int,
        max_plies: int,
        c_visit: float,
        c_scale: float,
        search_kind: str,
        gumbel_m: int,
        leaf_build_threads: int = 1,
        run_id: str,
        spool_dir: str | Path,
        game_record_dir: str | Path,
        allocator_posture: str | None = None,
        ladder_state_path: str | Path,
        promotion: DeployTagHooks,
        sink: Any = None,
        heartbeat: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        mp_ctx_name: str = "spawn",
    ) -> None:
        # A whitelist equality on the context NAME STRING, before any directory is made.
        # Never on the context OBJECT: five suites monkeypatch `get_context`.
        if mp_ctx_name != "spawn":
            raise ValueError(
                f"mp_ctx_name={mp_ctx_name!r} is refused; only 'spawn' is supported. TWO "
                "independent reasons, both structural: (1) the worker needs its OWN CUDA "
                "context (this module's docstring) and a forked child inherits a poisoned "
                "one; (2) `_worker_entry` arms PR_SET_PDEATHSIG (F-816-14) and the kernel "
                "signals on the death of the thread that CREATED the child — under "
                "'forkserver' that is a thread of the forkserver process, not the trainer, "
                "so the arming would track the wrong process and either fire early or never. "
                "This is not a permanent bar: a future caller with a real need may lift it, "
                "but must re-derive the arming's parent identity first and say so here."
            )
        self._eval_cfg = eval_cfg
        self._caps = caps
        self._encoding = encoding
        #: The graph forward's memory bound, resolved ONCE in the PARENT. Required with no
        #: default: the eval child is a SECOND allocator no in-process bound can see. `None`
        #: is the GRID arm.
        self._fused_graph_caps = fused_graph_caps
        #: The deploy head's MCTS leaf-batch width. NOT defaulted: a default would be a search
        #: regime nobody minted standing in for the one the net was trained under.
        self._leaf_batch_size = int(leaf_batch_size)
        #: The run's `selfplay.max_game_moves`. NOT defaulted: a default is the
        #: `DEFAULT_MAX_PLIES = 128` module constant put back on an operator-owed prereg axis.
        self._max_plies = int(max_plies)
        #: `selfplay.{c_visit, c_scale}`, the deploy head's sigma terms. NOT defaulted: a
        #: default is `DeployHeadPlayer`'s own `50.0`/`1.0` put back one layer out.
        self._c_visit = float(c_visit)
        self._c_scale = float(c_scale)
        #: The run's `search.kind` and `selfplay.gumbel_m`. NOT defaulted: the eval head's
        #: regime used to come from `DeployHeadPlayer`'s body, which the config never stated.
        self._search_kind = str(search_kind)
        self._gumbel_m = int(gumbel_m)
        #: The graph collector's batching geometry. NOT defaulted: these two were LITERALS in
        #: the child's hand-made server dict, and a default would put them back.
        self._inference_batching = inference_batching
        #: The eval leaf-graph build's WIDTH, derived ONCE in the parent. DEFAULTED TO 1
        #: because 1 is the SERIAL path, the exact-parity control. A producer test over
        #: `run.py`'s AST keeps it from silently disabling: a build that stopped being threaded
        #: shows up as correct results with 95 % of the eval path back in a serial loop.
        self._leaf_build_threads = max(1, int(leaf_build_threads))
        #: The allocator REGIME the round's caps were fitted under. It carries a DEFAULT
        #: because the safety is "a cuda child handed no token RAISES", which `None` can only
        #: fail, never excuse. `None` is the not-cuda arm.
        self._allocator_posture = allocator_posture
        self._run_id = run_id
        self._spool_dir = Path(spool_dir)
        self._spool_dir.mkdir(parents=True, exist_ok=True)
        #: The run's game-record directory, THREADED from the composition root — deriving it
        #: from `spool_dir.parent` would be a second authority for a path `mantis.run` owns.
        self._game_record_dir = Path(game_record_dir)
        # Sidecars live in a SIBLING directory: spool_dir holds ONLY snapshot (.pt) files, the
        # LAW-12 one-loader carve-out a test pins by torch.load()ing everything there. SCOPED
        # BY `run_id` because round ids are a per-run counter, so two runs sharing an out-dir
        # wrote identical sidecar names; the schema constrains `run_id`, so a sanitizer here
        # would be a second authority for it.
        self._work_dir = self._spool_dir.parent / f"{self._spool_dir.name}.work" / self._run_id
        self._work_dir.mkdir(parents=True, exist_ok=True)
        # The litter sweep, the ONLY handle the "the run itself was SIGKILLed" case has. Its
        # precondition — no live writer at construction — is structural because the dir derives
        # from `--out-dir` AND `run_id`; from the out-dir alone it was false exactly when two
        # runs shared one. Only `.tmp` is in scope, and no age threshold, which would be an
        # unmeasured constant against a race the precondition already excludes.
        for stale in self._work_dir.glob("*_result.json.tmp"):
            try:
                stale.unlink(missing_ok=True)
            except OSError:
                _LOG.debug("stale eval result tmp not swept: %s", stale, exc_info=True)
        self._ladder_state_path = Path(ladder_state_path)
        self._promotion = promotion
        self._sink = sink
        self._heartbeat = heartbeat
        self._clock = clock
        self._mp_ctx_name = mp_ctx_name

        self._lock = threading.Lock()
        self._inflight: dict[str, Any] | None = None
        self._mailbox: list[dict[str, Any]] = []
        self._round_counter = 0
        self._last_p_hat: dict[str, float] = {}
        # The EXTERNAL channel's history and consecutive-flag counter. IN-MEMORY, resetting on
        # resume: every reading is also emitted, so the durable series is the event stream.
        self._external_history: list[ChannelRoundReading] = []
        self._degradation_flags = 0
        #: Times `_finalize_round` was re-entered for an already-finalised round and the
        #: duplicate was SUPPRESSED. Non-zero means a promotion is being double-counted.
        self._double_finalize_suppressed = 0
        #: The strength floor's fire rate as a pair — probed rounds and the subset
        #: short-circuited. Both stay 0 when the config mints `eval.strength_floor: null`.
        self._floor_checked_total = 0
        self._floor_skipped_total = 0

        # LAZY: the ladder state is only needed once a round is kicked, so a pipeline built to
        # exercise the poller alone need not hand a fully-populated `eval_cfg.ladder` up front.
        self._ladder_state: LadderState | None = None

        self._stop_event = threading.Event()
        self._poller = threading.Thread(
            target=self._poll_loop, name="eval-pipeline-poller", daemon=True,
        )
        self._poller.start()

    def _ensure_ladder_state(self) -> LadderState:
        if self._ladder_state is None:
            self._ladder_state = self._load_or_init_ladder_state()
        return self._ladder_state

    def _load_or_init_ladder_state(self) -> LadderState:
        # LAW-14: a load failure (corrupt/unreadable state file) RAISES — it must never
        # silently discard graduation streaks or saturation history by "starting fresh".
        if self._ladder_state_path.exists():
            return LadderState.load(self._ladder_state_path, ladder_cfg=self._eval_cfg.ladder)
        return LadderState.initial(self._eval_cfg.ladder)

    def _beat(self, source: str) -> None:
        if self._heartbeat is not None:
            self._heartbeat(source)

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            self._beat("eval_round")
            with self._lock:
                inflight = self._inflight
            if inflight is None:
                self._stop_event.wait(_POLL_TICK_SEC)
                continue
            proc = inflight["proc"]
            proc.join(_POLL_TICK_SEC)
            if not proc.is_alive():
                self._finalize_round(inflight)
                continue
            elapsed = self._clock() - inflight["t0"]
            if elapsed > self._eval_cfg.round_timeout_sec:
                self._escalate_and_finalize(inflight)

    def _escalate_and_finalize(self, inflight: dict[str, Any]) -> None:
        # `_bounded_join_timeout` is the ONLY guard between this call and a real OverflowError:
        # this runs from `_poll_loop`, outside the catch-all, so a raise kills the poller.
        proc = inflight["proc"]
        proc.terminate()
        proc.join(_bounded_join_timeout(self._eval_cfg.worker_kill_grace_sec))
        proc.kill()
        proc.join(_bounded_join_timeout(self._eval_cfg.worker_kill_grace_sec))
        # The round exceeded `round_timeout_sec`, its PROGRESS budget; the joins above are the
        # kill sequence, not the cause. `JOIN_TIMEOUT` here misreported that as a stuck child.
        self._finalize_round(inflight, escalated_reason=EvalBrokenReason.ROUND_TIMEOUT)

    def run_evaluation(
        self, model: Any, step: int, best: Any, *, full_config: dict[str, Any],
        best_model_step: int | None, ignore_stride: bool = False,
    ) -> dict[str, Any]:
        if ignore_stride:
            return self._run_terminal_sync(model, step, best, best_model_step=best_model_step)

        with self._lock:
            if self._inflight is not None:
                in_flight_id = self._inflight["round_id"]
                emit_round_skipped_busy(self._sink, step=step, in_flight_round_id=in_flight_id)
                return {"kicked": False, "round_id": in_flight_id, "step": step, "reason": "busy"}
            round_idx = self._round_counter + 1
            round_id = f"r{round_idx:06d}_{step}"
            self._round_counter = round_idx
            spec, scheduled, gate_scheduled, candidate_path = self._build_round_spec(
                model, step, best, round_id=round_id, round_idx=round_idx, terminal=False,
            )
            proc = self._spawn_worker(spec)
            self._inflight = {
                "round_id": round_id, "step": step, "proc": proc, "spec": spec,
                "t0": self._clock(), "round_idx": round_idx,
                "candidate_snapshot_path": str(candidate_path),
            }
        emit_round_started(
            self._sink, round_id=round_id, step=step, scheduled=scheduled,
            gate_scheduled=gate_scheduled, ts=time.time(),
        )
        return {"kicked": True, "round_id": round_id, "step": step, "reason": None}

    @property
    def round_counter(self) -> int:
        """Rounds this pipeline has kicked. READ-ONLY; `restore_round_state` is the writer."""
        return self._round_counter

    @property
    def last_p_hat(self) -> dict[str, float]:
        """The last observed per-rung win rates, as a copy. READ-ONLY."""
        return dict(self._last_p_hat)

    def restore_round_state(self, *, round_counter: int, last_p_hat: Mapping[str, float]) -> None:
        """Resume the round counter and `p_hat` a stopped process left behind.

        Raises:
            ValueError: `round_counter` is negative, or lower than the counter already reached."""
        if round_counter < 0:
            raise ValueError(f"round_counter must be >= 0, got {round_counter}")
        if round_counter < self._round_counter:
            raise ValueError(
                f"refusing to restore round_counter {round_counter} below the counter already "
                f"reached ({self._round_counter}) — round ids are monotonic within a run and "
                "reusing one would overwrite a result already on disk"
            )
        self._round_counter = int(round_counter)
        self._last_p_hat = {str(k): float(v) for k, v in last_p_hat.items()}

    def _assess_external_channel(
        self, result: Mapping[str, Any], *, round_id: str, step: int,
    ) -> None:
        """Read the saturation label and the degradation flag, and publish them."""
        wr = result.get("wr_sealbot")
        games = result.get("wr_sealbot_games")
        rung = result.get("wr_sealbot_rung")
        if wr is None or not games or rung is None:
            # No sealbot rung recorded a game this round. NOT a zero: a round the instrument
            # did not play is absent from the series, never a loss.
            return
        self._external_history.append(ChannelRoundReading(
            round_idx=self._round_counter, games=int(games),
            wins=int(round(float(wr) * int(games))),
            promoted=bool(result.get("promoted")), rung=str(rung),
        ))
        health = assess_channel(
            self._external_history,
            bootstrap_resamples=self._eval_cfg.ladder.bootstrap_resamples,
            bootstrap_ci_level=self._eval_cfg.ladder.bootstrap_ci_level,
            bootstrap_seed=self._eval_cfg.ladder.bootstrap_seed,
            previous_consecutive_flags=self._degradation_flags,
        )
        self._degradation_flags = health.consecutive_degradation_flags
        if health.saturated:
            _LOG.info(
                "eval_channel_saturated round_id=%s rung=%s pooled_wr=%.4f over %d games — "
                "the rung has stopped discriminating; strength claims answer to the NEXT rung",
                round_id, rung, health.pooled_wr, health.pooled_games,
            )
        if health.degraded:
            _LOG.warning(
                "eval_channel_degraded round_id=%s rung=%s pooled_wr=%.4f vs running max "
                "%.4f while promotions continue (flag %d of 2 before an architect read) — "
                "WARN-ONLY for run6 (G-3)",
                round_id, rung, health.pooled_wr, health.running_max_wr or 0.0,
                health.consecutive_degradation_flags,
            )
        _emit(self._sink, {
            "event": "eval_channel_health", "round_id": round_id, "step": step,
            "rung": rung, "label": health.label, "pooled_wr": health.pooled_wr,
            "pooled_games": health.pooled_games, "rounds_pooled": health.rounds_pooled,
            "ci_lower": health.ci_lower, "ci_upper": health.ci_upper,
            "running_max_wr": health.running_max_wr, "saturated": health.saturated,
            "degraded": health.degraded,
            "consecutive_degradation_flags": health.consecutive_degradation_flags,
        })

    def _current_p_hat(self) -> dict[str, float]:
        if self._last_p_hat:
            return dict(self._last_p_hat)
        return {rung.name: 0.5 for rung in self._eval_cfg.ladder.rungs}

    def _build_round_spec(
        self, model: Any, step: int, best: Any, *, round_id: str, round_idx: int, terminal: bool,
    ) -> tuple[RoundSpec, dict[str, int], bool, Path]:
        cfg = self._eval_cfg
        candidate_path = self._spool_dir / f"{round_id}_candidate.pt"
        write_model_snapshot(model, candidate_path)
        best_path: Path | None = None
        if best is not None:
            best_path = self._spool_dir / f"{round_id}_best.pt"
            write_model_snapshot(best, best_path)

        if terminal:
            alloc = {rung.name: rung.games_max for rung in cfg.ladder.rungs}
        else:
            # Scheduling is the design's pre-registered activation law, verbatim: a dormant
            # rung stays dormant until its predecessor clears `activation_wr_lower_ci`.
            alloc = self._ensure_ladder_state().allocate_games(round_idx, self._current_p_hat())

        run_gate = (best is not None) and (round_idx % cfg.gate.stride == 0 or terminal)
        rung_jobs = [
            RungJob(
                name=rung.name, bot=rung.bot, variant=rung.variant, depth=rung.depth,
                opponent_sims=rung.opponent_sims, opening_book=rung.opening_book,
                deploy_matched=rung.deploy_matched, games=int(alloc.get(rung.name, 0)),
            )
            for rung in cfg.ladder.rungs
        ]
        gate_spec = GateSpec(
            stride=cfg.gate.stride, screen_games=cfg.gate.screen_games,
            confirm_games=cfg.gate.confirm_games, promotion_winrate=cfg.gate.promotion_winrate,
            screen_confirm_lo=cfg.gate.screen_confirm_lo, deploy_sims=cfg.gate.deploy_sims,
            opening_book=cfg.gate.opening_book, bootstrap_resamples=cfg.gate.bootstrap_resamples,
            min_distinct_per_pair=cfg.gate.min_distinct_per_pair, seed_base=cfg.gate.seed_base,
            run_gate=run_gate,
        )
        result_path = self._work_dir / f"{round_id}_result.json"
        progress_path = self._work_dir / f"{round_id}_progress.txt"
        spec = RoundSpec(
            round_id=round_id, round_index=int(round_idx), step=step,
            candidate_snapshot=str(candidate_path),
            best_snapshot=(str(best_path) if best_path is not None else None),
            best_step=None, encoding=self._encoding, worker_device=cfg.worker_device,
            gate=gate_spec, rung_jobs=rung_jobs, random_floor_games=cfg.random_floor_games,
            random_model_sims=cfg.random_model_sims, sealbot_model_sims=cfg.sealbot_model_sims,
            seed_base=cfg.gate.seed_base, round_timeout_sec=cfg.round_timeout_sec,
            result_path=str(result_path), progress_path=str(progress_path),
            ladder_bootstrap_resamples=cfg.ladder.bootstrap_resamples,
            ladder_bootstrap_ci_level=cfg.ladder.bootstrap_ci_level,
            ladder_bootstrap_seed=cfg.ladder.bootstrap_seed,
            game_record=GameRecordTarget(record_dir=str(self._game_record_dir),
                                        run_id=self._run_id),
            # The two early-strength postures, resolved through their ONE read path and
            # carried to the child. Both are `None` for every committed config.
            ply_cap_adjudication=resolve_ply_cap_adjudication(cfg),
            strength_floor=resolve_strength_floor(cfg),
            # Resolved once in the parent: the child has no `RunConfig` and builds its graph
            # server from a hand-made dict, so this is the only route to the second allocator.
            fused_graph_caps=self._fused_graph_caps,
            # Same seam and same reason: the deploy head must search at the width the net's
            # targets were generated at, and the child cannot read the config to find it.
            leaf_batch_size=self._leaf_batch_size,
            # Same seam: the child's DENSE autocast had no `dtype=` and ran at torch's device
            # default on the path LAW-15 reads the bar off; the ply cap was a module constant.
            max_plies=self._max_plies,
            # Same seam and same reason: two REQUIRED schema keys the deploy head was never
            # given, so it searched at its own signature defaults.
            c_visit=self._c_visit, c_scale=self._c_scale,
            search_kind=self._search_kind, gumbel_m=self._gumbel_m,
            # Same seam: the child's graph server wrote its pop width and pop deadline as
            # literals, and 33 % of the eval path's ms/sim was the deadline one of them set.
            inference_batching=self._inference_batching,
            # Same seam and same reason: the leaf build's width is a HOST reservation and the
            # child has no config to derive one from.
            leaf_build_threads=self._leaf_build_threads,
            # Same seam. Read straight off `cfg` rather than cached: it is one int with no
            # resolver, and a cached copy is the second authority these rows exist to remove.
            concurrency=cfg.concurrency,
            # Same seam, same reason: a posture is a property of the PROCESS environment, so
            # the parent's boot assertion says nothing about the child's.
            allocator_posture=self._allocator_posture,
        )
        return spec, dict(alloc), run_gate, candidate_path

    def _spawn_worker(self, spec: RoundSpec) -> Any:
        # THE FIRST STATEMENT, ahead of the spec write: the single choke point both callers
        # pass through, so the invariant lives in one place rather than in two copies.
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError(
                "eval worker spawn attempted from thread "
                f"{threading.current_thread().name!r}, not the main thread. `_worker_entry` "
                "arms PR_SET_PDEATHSIG (F-816-14) and the kernel tracks the CREATING THREAD, "
                "so a child spawned from a short-lived thread is SIGKILLed the moment that "
                "thread returns — a premature kill of a LIVE eval round, which is strictly "
                "worse than the orphan the arming prevents. Both call sites "
                "(`run_evaluation`, `_run_terminal_sync`) are reached inline from "
                "`run_training_loop` under `compose_run`, which is main-thread-only because "
                "it calls `signal.signal`. If the eval kick ever moves onto the poller "
                "thread, the arming in `_worker_entry` must be re-derived FIRST."
            )
        spec_path = self._work_dir / f"{spec.round_id}_spec.json"
        spec_path.write_text(json.dumps(spec.to_dict()))
        ctx = multiprocessing.get_context(self._mp_ctx_name)
        # typeshed's BaseContext omits Process (it lives on the concrete contexts); every real
        # context returned by get_context has it.
        proc = ctx.Process(  # pyright: ignore[reportAttributeAccessIssue]
            target=_worker_entry, args=(str(spec_path), spec.result_path),
            kwargs={}, daemon=True,
        )
        proc.start()
        from mantis.train.lifecycle.signals import register_child
        register_child(proc)
        return proc

    def poll_completed(self) -> dict | list | None:
        with self._lock:
            if not self._mailbox:
                return None
            return self._mailbox.pop(0)

    def drain_pending(self) -> dict | list | None:
        with self._lock:
            inflight = self._inflight
        if inflight is None:
            return None
        proc = inflight["proc"]
        if proc.is_alive():
            budget = drain_budget_sec(self._caps)
            reason = drain_or_kill(
                proc, budget_sec=budget, worker_kill_grace_sec=self._eval_cfg.worker_kill_grace_sec,
                clock=self._clock,
            )
            if reason is not None:
                return self._finalize_round(inflight, escalated_reason=reason)
        return self._finalize_round(inflight)

    def _finalize_round(
        self, inflight: dict[str, Any], *, escalated_reason: EvalBrokenReason | None = None,
    ) -> dict[str, Any] | None:
        # ONCE-ONLY. `_poll_loop` and the drain both read `self._inflight` before it is cleared
        # at the END of this method, so both can finalise the SAME dict — appending twice,
        # persisting twice, promoting off one round's games counted as two. The latch lives on
        # the `inflight` dict because it must be per-ROUND: a `self`-level flag would need a
        # reset between rounds, and a missed reset disables the guard forever.
        with self._lock:
            if inflight.get("_finalized"):
                self._double_finalize_suppressed += 1
                _LOG.warning(
                    "eval_round_double_finalize_suppressed round_id=%s step=%s",
                    inflight.get("round_id"), inflight.get("step"),
                )
                _emit(self._sink, {
                    "event": "eval_round_double_finalize_suppressed",
                    "round_id": inflight.get("round_id"), "step": inflight.get("step"),
                    "suppressed_total": self._double_finalize_suppressed,
                })
                # The first finalise's result if it has already been produced; `None` while
                # it is still in flight, which the callers already treat as "nothing ready".
                return inflight.get("_result")
            inflight["_finalized"] = True

        proc = inflight["proc"]
        from mantis.train.lifecycle.signals import unregister_child
        unregister_child(proc)
        # The child died, the run lives; all four finalising routes converge here. Non-raising
        # by construction, so a deletion failure can never manufacture a broken round.
        _drop_result_tmp_if_writer_gone(inflight)
        wall_sec = max(self._clock() - inflight["t0"], 0.0)
        exit_code = getattr(proc, "exitcode", None)

        # The whole round-completion decision runs under one catch-all: any uncaught exception
        # becomes a delivered `eval_broken(round_completion_error)` instead of propagating out
        # of the poller thread, where silent death stops the heartbeat and hangs the run to the
        # watchdog staleness deadline.
        try:
            if escalated_reason is not None:
                # `phase` is a FUNCTION of the reason, so it stays on the payload: a constant
                # "drain" would send a supervisor triaging a round timeout to the drain budget.
                phase = ("round_timeout" if escalated_reason is EvalBrokenReason.ROUND_TIMEOUT
                         else "drain")
                result = self._broken_result(inflight, reason=escalated_reason, exit_code=exit_code,
                                             wall_sec=wall_sec, phase=phase)
            elif exit_code is not None and exit_code != 0:
                reason = (EvalBrokenReason.KILLED if exit_code < 0
                          else EvalBrokenReason.EXIT_NONZERO)
                result = self._broken_result(inflight, reason=reason, exit_code=exit_code,
                                             wall_sec=wall_sec, phase="worker_exit")
            else:
                result = self._read_worker_result(inflight, exit_code=exit_code, wall_sec=wall_sec)
        except Exception as exc:  # noqa: BLE001 -- deliberate catch-all, see docstring above
            # The traceback is logged at the raising site, not in the emitter: `_LOG.exception`
            # is only correct with a live exception, and the other call sites have none.
            detail = repr(exc)
            _LOG.exception(
                "eval_round_completion_failed round_id=%s step=%s detail=%s",
                inflight["round_id"], inflight["step"], detail,
            )
            result = self._broken_result(
                inflight, reason=EvalBrokenReason.ROUND_COMPLETION_ERROR,
                exit_code=getattr(inflight["proc"], "exitcode", None), wall_sec=wall_sec,
                phase="round_completion", detail=detail, exception_class=type(exc).__name__,
            )

        with self._lock:
            self._inflight = None
            self._mailbox.append(result)
            # Cache under the same lock the guard reads, so a suppressed second caller that
            # arrives after this point gets the real result rather than `None`.
            inflight["_result"] = result
        return result

    def _read_worker_result(
        self, inflight: dict[str, Any], *, exit_code: int | None, wall_sec: float,
    ) -> dict[str, Any]:
        spec: RoundSpec = inflight["spec"]
        result_path = Path(spec.result_path)
        try:
            if not result_path.is_file():
                raise FileNotFoundError(str(result_path))
            raw = json.loads(result_path.read_text())
            validate_worker_result(raw)
        except FileNotFoundError:
            return self._broken_result(inflight, reason=EvalBrokenReason.RESULT_MISSING,
                                       exit_code=exit_code, wall_sec=wall_sec,
                                       phase="worker_exit")
        except (ValueError, ResultContractError, OSError):
            return self._broken_result(inflight, reason=EvalBrokenReason.RESULT_INVALID,
                                       exit_code=exit_code, wall_sec=wall_sec,
                                       phase="worker_exit")
        return self._success_result(inflight, raw, wall_sec=wall_sec)

    def _broken_result(
        self, inflight: dict[str, Any], *, reason: EvalBrokenReason, exit_code: int | None,
        wall_sec: float, phase: str, detail: str | None = None,
        exception_class: str | None = None,
    ) -> dict[str, Any]:
        """THE broken-round emitter — one event, one payload builder, one `build_round_result`
        call site for all seven routes."""
        payload: dict[str, Any] = {
            "event": "eval_broken", "round_id": inflight["round_id"], "step": inflight["step"],
            "reason": reason, "exit_code": exit_code, "phase": phase,
        }
        if exception_class is not None:
            payload["exception_class"] = exception_class
        if detail is not None:
            payload["detail"] = detail
        _emit(self._sink, payload)
        _LOG.error("eval_broken round_id=%s step=%s reason=%s", inflight["round_id"],
                  inflight["step"], reason.value)
        result = build_round_result(
            step=inflight["step"], round_id=inflight["round_id"],
            rungs_config=self._eval_cfg.ladder.rungs, rung_results={}, gate_result=None,
            skipped_rungs=[], bt={"ratings": {}, "p_hat": {}}, schedule_next={},
            eval_round_wall_sec=wall_sec, reason=reason, detail=detail, random_wr=None,
        )
        emit_round_complete(
            self._sink, round_id=inflight["round_id"], step=inflight["step"], wall_sec=wall_sec,
            # None, never 0 — a broken round MEASURED nothing, and a count here is a default
            # a reader will mistake for one (it already was).
            games_total=None, promoted=False, wr_sealbot=result["wr_sealbot"],
            progress=read_progress(inflight.get("spec")),
        )
        return result

    def _success_result(
        self, inflight: dict[str, Any], raw: dict[str, Any], *, wall_sec: float,
    ) -> dict[str, Any]:
        round_idx = inflight["round_idx"]
        rungs_raw: dict[str, Any] = raw.get("rungs", {})
        gate_raw = raw.get("gate")
        random_raw = raw.get("random") or {"games": 0, "wr": None}
        skipped_rungs = raw.get("skipped_rungs", [])

        ladder_results = {
            name: {"games": info.get("games", 0), "wr": info.get("wr"), "ci_lo": info.get("wr_ci_lower")}
            for name, info in rungs_raw.items()
        }
        # The worker has no `LadderState` and stamped `"status": "active"` on every rung,
        # saturated ones included. Read BEFORE `record_round`: the status it was PLAYED under.
        played_under = self._ensure_ladder_state()
        for name, info in rungs_raw.items():
            try:
                info["status"] = played_under.status(name)
            except KeyError:
                # a rung the worker played that the ladder does not know: absent, not "active"
                info["status"] = None
        self._ensure_ladder_state().record_round(round_idx, ladder_results, sink=self._sink)
        try:
            self._ensure_ladder_state().save(self._ladder_state_path)
        except LadderStateError:
            # LAW-14: a persistence failure is run-fatal. The games already played go with it,
            # since the ladder's on-disk state did not durably advance.
            _LOG.exception("ladder_state_persist_failed round_id=%s", inflight["round_id"])
            return self._broken_result(
                inflight, reason=EvalBrokenReason.LADDER_PERSIST_FAILED, exit_code=None,
                wall_sec=wall_sec, phase="ladder_persist",
                detail=f"ladder state persist failed: {self._ladder_state_path}",
            )

        # Fold the best/gate entity into the SAME global BT fit: a fit that omits it still
        # recovers rung-vs-candidate ratings but never the candidate-vs-best gap.
        rung_entities = [
            rung.name for rung in self._eval_cfg.ladder.rungs if rung.name in rungs_raw
        ]
        entities = ["candidate"]
        if gate_raw:
            entities.append("best")
        entities += rung_entities
        n = len(entities)
        wins_matrix = np.zeros((n, n), dtype=np.float64)
        if gate_raw:
            idx = entities.index("best")
            n_pooled = int(gate_raw.get("n_pooled") or 0)
            wr_confirm = gate_raw.get("wr_confirm")
            if n_pooled > 0 and wr_confirm is not None:
                cand_wins = wr_confirm * n_pooled
                wins_matrix[0, idx] += cand_wins
                wins_matrix[idx, 0] += n_pooled - cand_wins
        for name in rung_entities:
            idx = entities.index(name)
            info = rungs_raw[name]
            games = int(info.get("games", 0))
            wr = info.get("wr")
            if games > 0 and wr is not None:
                cand_wins = wr * games
                wins_matrix[0, idx] += cand_wins
                wins_matrix[idx, 0] += games - cand_wins
        ratings = fit_bt(wins_matrix, prior_games=self._eval_cfg.ladder.bt_prior_games)
        p_hat = {name: predict_p(ratings, 0, entities.index(name)) for name in rung_entities}
        self._last_p_hat = p_hat

        schedule_next = (
            self._ensure_ladder_state().allocate_games(round_idx + 1, self._current_p_hat()) if p_hat else {}
        )

        games_total = sum(int(info.get("games", 0)) for info in rungs_raw.values())
        games_total += int(random_raw.get("games", 0) or 0)
        if gate_raw:
            games_total += int(gate_raw.get("n_pooled") or gate_raw.get("n_screen") or 0)
        # On a REFUSED strength floor the probe plays the only games of the round, so the terms
        # above sum to 0 while `eval_strength_floor.games` reports N. This is the round's games.
        floor_raw = raw.get("strength_floor")
        if floor_raw:
            games_total += int(floor_raw.get("games", 0) or 0)

        result = build_round_result(
            step=inflight["step"], round_id=inflight["round_id"],
            rungs_config=self._eval_cfg.ladder.rungs, rung_results=rungs_raw,
            gate_result=gate_raw, skipped_rungs=skipped_rungs,
            bt={"ratings": {name: float(ratings[i]) for i, name in enumerate(entities)}, "p_hat": p_hat},
            schedule_next=schedule_next, eval_round_wall_sec=wall_sec, reason=None,
            detail=None, random_wr=random_raw.get("wr"), worker_pid=raw.get("worker_pid"),
            candidate_snapshot_path=inflight.get("candidate_snapshot_path"),
            # The floor's verdict reaches the LAW-15 gate ONLY through this mapping.
            # `_emit_posture_events` reads the same `raw` key; neither is the other's source,
            # so a floor payload that stops arriving silences both rather than staling one.
            strength_floor=raw.get("strength_floor"),
        )
        # The identity is published by the producer out of one walk; this is the AGREEMENT
        # CHECK over it — an independent walk that refuses when the two disagree.
        self._check_the_sealbot_rung_identity(
            rungs_raw, result, round_id=inflight["round_id"],
        )

        emit_round_complete(
            self._sink, round_id=inflight["round_id"], step=inflight["step"], wall_sec=wall_sec,
            games_total=games_total,
            # `promoted: False` used to cover gate-refused, gate-not-scheduled and no-anchor
            # alike. A decision was taken iff the worker returned a gate result.
            promoted=(result["promoted"] if gate_raw else None),
            wr_sealbot=result["wr_sealbot"],
            # The ROUND CI travels with the win rate it belongs to; it was already bootstrapped
            # and published per rung, and simply stopped at the round RESULT.
            wr_sealbot_ci_lower=result.get("wr_sealbot_ci_lower"),
            wr_sealbot_ci_upper=result.get("wr_sealbot_ci_upper"),
            progress=read_progress(inflight.get("spec")),
        )
        self._assess_external_channel(result, round_id=inflight["round_id"],
                                      step=inflight["step"])
        emit_rung_skip_events(inflight["round_id"], skipped_rungs, self._sink)
        device_memory = raw.get("device_memory")
        if device_memory is not None:
            emit_device_memory(
                self._sink, round_id=inflight["round_id"], step=inflight["step"],
                device_memory=device_memory,
            )
        self._emit_posture_events(inflight, raw)
        return result

    def _check_the_sealbot_rung_identity(
        self, rungs_raw: Mapping[str, Any], result: Mapping[str, Any], *, round_id: str,
    ) -> None:
        """Refuse a round whose published sealbot identity disagrees with an independent walk.

        Raises:
            ResultContractError: the independent walk disagrees on the rung, its WR or its
                game count."""
        expected: tuple[Any, Any, Any] = (None, None, None)
        for rung in self._eval_cfg.ladder.rungs:
            if getattr(rung, "bot", None) != "sealbot":
                continue
            info = rungs_raw.get(rung.name)
            if info is None or int(info.get("games", 0)) <= 0:
                continue
            expected = (info.get("wr"), rung.name, int(info.get("games", 0)))
            break
        published = (
            result.get("wr_sealbot"), result.get("wr_sealbot_rung"),
            result.get("wr_sealbot_games"),
        )
        if published != expected:
            raise ResultContractError(
                f"round {round_id}: the independent ladder walk derives "
                f"(wr, rung, games) = {expected!r} for the first sealbot rung with games "
                f"this round, but the round result publishes {published!r}. "
                "`mantis.eval.rounds._first_sealbot_wr` and this walk have drifted."
            )

    def _emit_posture_events(self, inflight: dict[str, Any], raw: Mapping[str, Any]) -> None:
        """The two armed-posture channels, driven by the worker payload's OWN key set."""
        floor = raw.get("strength_floor")
        if floor is not None:
            self._floor_checked_total += 1
            if not floor.get("passed", False):
                self._floor_skipped_total += 1
            emit_strength_floor(
                self._sink, round_id=inflight["round_id"], step=inflight["step"], floor=floor,
                checked_total=self._floor_checked_total, skipped_total=self._floor_skipped_total,
            )
        adjudication = raw.get("ply_cap_adjudication")
        if adjudication is not None:
            emit_ply_cap_adjudication(
                self._sink, round_id=inflight["round_id"], step=inflight["step"],
                adjudication=adjudication,
            )

    def _run_terminal_sync(
        self, model: Any, step: int, best: Any, *, best_model_step: int | None,
    ) -> dict[str, Any]:
        round_idx = self._round_counter + 1
        round_id = f"r{round_idx:06d}_{step}_terminal"
        self._round_counter = round_idx
        spec, scheduled, gate_scheduled, candidate_path = self._build_round_spec(
            model, step, best, round_id=round_id, round_idx=round_idx, terminal=True,
        )
        proc = self._spawn_worker(spec)
        # The terminal round emitted `eval_round_complete` with no `eval_round_started`, while
        # the `eval_round_wall` row names the PAIR as its producer.
        emit_round_started(
            self._sink, round_id=round_id, step=step, scheduled=scheduled,
            gate_scheduled=gate_scheduled, ts=time.time(),
        )
        inflight = {
            "round_id": round_id, "step": step, "proc": proc, "spec": spec,
            "t0": self._clock(), "round_idx": round_idx,
            "candidate_snapshot_path": str(candidate_path),
        }
        reason = drain_or_kill(
            proc, budget_sec=self._caps.terminal_eval_hard_cap_sec,
            worker_kill_grace_sec=self._eval_cfg.worker_kill_grace_sec, clock=self._clock,
        )
        # `_finalize_round` returns `None` only when another route already finalised, and this
        # `inflight` is never published to `self._inflight` — asserted, not assumed.
        result = (self._finalize_round(inflight, escalated_reason=reason) if reason is not None
                  else self._finalize_round(inflight))
        assert result is not None, (
            "the terminal-eval round was finalised twice — its inflight record is local to "
            "this call and must be unreachable from the poller and the drain"
        )
        return result

    def apply_gate_decision(self, result: Mapping[str, Any]) -> int | None:
        return apply_gate_decision(self._promotion, result)

    def stop(self) -> None:
        self._stop_event.set()
        if self._poller.is_alive():
            self._poller.join(5.0)
        with self._lock:
            inflight = self._inflight
        if inflight is not None:
            proc = inflight["proc"]
            from mantis.train.lifecycle.signals import unregister_child
            unregister_child(proc)
            if proc.is_alive():
                proc.terminate()
                proc.join(_bounded_join_timeout(self._eval_cfg.worker_kill_grace_sec))
                if proc.is_alive():
                    proc.kill()
                    proc.join(_bounded_join_timeout(self._eval_cfg.worker_kill_grace_sec))
            # The teardown route's litter sweep: this method never calls `_finalize_round` and
            # runs on EVERY run exit, the commonest producer the per-round unlink cannot reach.
            _drop_result_tmp_if_writer_gone(inflight)


def build_eval_pipeline(
    *,
    eval_cfg: Any,
    coordinator_cfg_caps: DrainCaps,
    encoding: str,
    fused_graph_caps: FusedGraphCapsSpec | None,
    inference_batching: InferenceBatchingSpec | None,
    leaf_batch_size: int,
    max_plies: int,
    c_visit: float,
    c_scale: float,
    search_kind: str,
    gumbel_m: int,
    run_id: str,
    spool_dir: str | Path,
    game_record_dir: str | Path,
    ladder_state_path: str | Path,
    promotion: DeployTagHooks,
    leaf_build_threads: int = 1,
    allocator_posture: str | None = None,
    sink: Any = None,
    heartbeat: Callable[[str], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
    mp_ctx: str = "spawn",
) -> EvalPipeline:
    """The ONE constructor — NO `device`, NO `model` parameter (isolation law 1: an in-process
    CUDA eval path is unrepresentable)."""
    return EvalPipeline(
        eval_cfg=eval_cfg, caps=coordinator_cfg_caps, encoding=encoding,
        fused_graph_caps=fused_graph_caps, inference_batching=inference_batching,
        leaf_batch_size=leaf_batch_size, max_plies=max_plies,
        c_visit=c_visit, c_scale=c_scale,
        search_kind=search_kind, gumbel_m=gumbel_m,
        leaf_build_threads=leaf_build_threads,
        run_id=run_id,
        allocator_posture=allocator_posture,
        spool_dir=spool_dir, game_record_dir=game_record_dir,
        ladder_state_path=ladder_state_path, promotion=promotion,
        sink=sink, heartbeat=heartbeat, clock=clock, mp_ctx_name=mp_ctx,
    )


__all__ = [
    "DrainCaps",
    "EvalPipeline",
    "build_eval_pipeline",
    "drain_budget_sec",
    "drain_or_kill",
    "emit_device_memory",
    "emit_ply_cap_adjudication",
    "emit_round_complete",
    "emit_round_skipped_busy",
    "emit_round_started",
    "emit_rung_skip_events",
    "emit_strength_floor",
    "read_progress",
    "SKIP_REASON_CLASSES",
]
