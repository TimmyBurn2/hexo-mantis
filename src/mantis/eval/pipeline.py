"""EvalPipeline — async kick + persistent poller + bounded drains (design §a.3/§c.3
pipeline.py). Out-of-process eval inference ONLY: `build_eval_pipeline` has NO `device`/
`model` constructor kwargs — an in-process CUDA path is unrepresentable here. The worker
subprocess is spawned under `multiprocessing.get_context("spawn")` (own CUDA context);
every subprocess join is timeout-bounded (isolation laws 1 + 2).

The pipeline owns exactly ONE persistent poller/keepalive thread (started at
`build_eval_pipeline`, stopped only by `stop()`) that beats `heartbeat("eval_round")`
EVERY tick, with or without an in-flight round — so a between-round gap can never
false-fire the watchdog; round PROGRESS is bounded separately by `round_timeout_sec`.

>300 justify: one isolation-law seam (kick/ack, spawn-context, join-boundedness, the
persistent poller, drain/kill escalation, round-result assembly) sharing one in-flight
round record and one mailbox — splitting the kick path from the poller from the drain
path would scatter the exact state machine the run3 45h livelock class exists to bound.
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

from mantis.eval.bt import fit_bt, predict_p
from mantis.eval.errors import LadderStateError, ResultContractError
from mantis.eval.ladder import LadderState
from mantis.eval.promote import DeployTagHooks, apply_gate_decision
from mantis.eval.rounds import (
    GateSpec,
    RoundSpec,
    RungJob,
    build_round_result,
    validate_worker_result,
)
from mantis.eval.snapshot import write_model_snapshot

_LOG = logging.getLogger(__name__)

#: The poller thread's fixed tick — small enough that an idle beat is observable in a few
#: tens of milliseconds (heartbeat tests), cheap enough to run for a whole round's life.
_POLL_TICK_SEC = 0.02

#: RED-TEAM-2 F-RT2-1 (BLOCKER fix), layer 2 (structural, defense-in-depth): mirrors
#: `mantis.config.schema._EVAL_TIMEOUT_CEILING_SEC`. `multiprocessing.Process.join`
#: cannot accept a non-finite timeout (raises `OverflowError` deep inside
#: `selectors.select()`'s `math.ceil(timeout*1e3)`) — schema validation (layer 1) makes a
#: non-finite `worker_kill_grace_sec`/drain-budget value unreachable through a config
#: load, but every `proc.join(...)` call site in this module bounds its timeout
#: defensively regardless: isolation law 2 ("every subprocess join is timeout-bounded")
#: must hold unconditionally, not only for schema-validated inputs — a future non-YAML
#: config source, a hand-built test fixture, or an arithmetic bug upstream (e.g.
#: `drain_budget_sec`'s multiply) must never be able to smuggle a non-finite value into a
#: real `Process.join()` and silently kill the poller thread the way F1's original
#: failure mode did.
_JOIN_TIMEOUT_CEILING_SEC = 86400.0


def _bounded_join_timeout(timeout: float) -> float:
    """Clamp `timeout` to a finite, non-negative value `Process.join()` can always accept.
    `inf`/`-inf`/`nan` (not `math.isfinite`) collapse to the one-day ceiling; a finite value
    is clamped to `[0.0, _JOIN_TIMEOUT_CEILING_SEC]`. Never raises."""
    if not math.isfinite(timeout):
        return _JOIN_TIMEOUT_CEILING_SEC
    return max(0.0, min(timeout, _JOIN_TIMEOUT_CEILING_SEC))


# ── R-DRAIN-HARDCAP: DrainCaps + the join-bound arithmetic (P-1, pre-registered WIRE) ────
@dataclass(frozen=True)
class DrainCaps:
    """The 4 drain-cap fields lifted from `StepCoordinatorConfig` (coordinator/config.py:
    176-180) — every field gains a live consumer here (R-DRAIN-HARDCAP-CONSUMERS)."""

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
) -> tuple[bool, str]:
    """Bounded join -> (if still alive) terminate -> bounded join -> kill -> bounded join.
    Returns `(broken, reason)`; every join carries a bound (isolation law 2)."""
    del clock  # the caller advances/consults its own clock; every join below is bounded
    proc.join(_bounded_join_timeout(budget_sec))
    if not proc.is_alive():
        return False, "clean_exit"
    proc.terminate()
    proc.join(_bounded_join_timeout(worker_kill_grace_sec))
    proc.kill()
    proc.join(_bounded_join_timeout(worker_kill_grace_sec))
    return True, "join_timeout"


# ── pure event builders (sink.emit + return the exact emitted payload) ──────────────────
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
    sink: Any, *, round_id: str, step: int, wall_sec: float, games_total: int,
    promoted: bool, wr_sealbot: float | None,
) -> dict[str, Any]:
    payload = {
        "event": "eval_round_complete", "round_id": round_id, "step": step,
        "wall_sec": wall_sec, "games_total": games_total, "promoted": promoted,
        "wr_sealbot": wr_sealbot,
    }
    _emit(sink, payload)
    return payload


def emit_round_skipped_busy(sink: Any, *, step: int, in_flight_round_id: str) -> dict[str, Any]:
    payload = {"event": "eval_round_skipped_busy", "step": step, "in_flight_round_id": in_flight_round_id}
    _emit(sink, payload)
    return payload


def emit_rung_skip_events(round_id: str, skipped: list[Mapping[str, str]], sink: Any) -> None:
    """Per skipped rung: ONE `eval_rung_skipped` event AND one ERROR log line — never
    silent (all three: event + log + the caller's own `skipped_rungs` record)."""
    for entry in skipped:
        payload = {"event": "eval_rung_skipped", "round_id": round_id,
                   "rung": entry["rung"], "reason": entry["reason"]}
        _emit(sink, payload)
        _LOG.error("eval_rung_skipped round_id=%s rung=%s reason=%s",
                   round_id, entry["rung"], entry["reason"])


def _worker_entry(spec_path: str, result_path: str) -> None:
    """The spawn-ctx `Process` target (module-level so spawn can pickle-by-reference).
    Torch/worker imports stay LAZY — this function body is the only place the parent
    process's import of `mantis.eval.pipeline` ever touches `mantis.eval.worker`."""
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
        run_id: str,
        spool_dir: str | Path,
        ladder_state_path: str | Path,
        promotion: DeployTagHooks,
        sink: Any = None,
        heartbeat: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        mp_ctx_name: str = "spawn",
    ) -> None:
        self._eval_cfg = eval_cfg
        self._caps = caps
        self._encoding = encoding
        self._run_id = run_id
        self._spool_dir = Path(spool_dir)
        self._spool_dir.mkdir(parents=True, exist_ok=True)
        # Spec/result/progress sidecar files live in a SIBLING directory, never nested
        # under spool_dir: spool_dir holds ONLY model snapshot (.pt) files — the LAW-12
        # one-loader carve-out this WP pins (test_snapshots_are_not_checkpoints walks
        # every file under spool_dir and torch.load()s it).
        self._work_dir = self._spool_dir.parent / f"{self._spool_dir.name}.work"
        self._work_dir.mkdir(parents=True, exist_ok=True)
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

        # LAZY: the ladder state is only ever needed once a round is actually kicked
        # (`_build_round_spec`/`_success_result`) — deferring construction means a
        # pipeline built for a narrow purpose (e.g. only exercising the heartbeat poller)
        # need not hand a fully-populated `eval_cfg.ladder` up front.
        self._ladder_state: LadderState | None = None

        self._stop_event = threading.Event()
        self._poller = threading.Thread(
            target=self._poll_loop, name="eval-pipeline-poller", daemon=True,
        )
        self._poller.start()

    # ── construction helpers ─────────────────────────────────────────────────────────
    def _ensure_ladder_state(self) -> LadderState:
        if self._ladder_state is None:
            self._ladder_state = self._load_or_init_ladder_state()
        return self._ladder_state

    def _load_or_init_ladder_state(self) -> LadderState:
        # LAW-14: a load failure (corrupt/unreadable state file) RAISES — it must never
        # silently discard graduation streaks/saturation history by "starting fresh" [M-1].
        if self._ladder_state_path.exists():
            return LadderState.load(self._ladder_state_path, ladder_cfg=self._eval_cfg.ladder)
        return LadderState.initial(self._eval_cfg.ladder)

    def _beat(self, source: str) -> None:
        if self._heartbeat is not None:
            self._heartbeat(source)

    # ── the persistent poller thread ────────────────────────────────────────────────────
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
        # F-RT2-1 layer 2: `_bounded_join_timeout` is the ONLY guard between this call
        # and a real OverflowError -- this method is invoked directly from `_poll_loop`,
        # entirely OUTSIDE `_finalize_round`'s F1 layer-2 catch-all, so an uncaught
        # exception here would kill the poller thread silently exactly like F1's original
        # failure mode (RED-TEAM-2 F-RT2-1).
        proc = inflight["proc"]
        proc.terminate()
        proc.join(_bounded_join_timeout(self._eval_cfg.worker_kill_grace_sec))
        proc.kill()
        proc.join(_bounded_join_timeout(self._eval_cfg.worker_kill_grace_sec))
        self._finalize_round(inflight, escalated_reason="join_timeout")

    # ── kick / ack ───────────────────────────────────────────────────────────────────
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
            # Deviation #3 REVERTED (dispatcher ruling, FIX-PASS Part 4): scheduling
            # semantics are the design's pre-registered STATE §5 activation law, verbatim
            # — no pipeline-level top-up for dormant rungs. A dormant rung behind an
            # unresolvable predecessor stays dormant (0 games) until its predecessor's
            # own measured round clears `activation_wr_lower_ci`; the e2e fixture (Part 3)
            # makes rung0 a RESOLVABLE stub so activation flows lawfully instead.
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
            round_id=round_id, step=step, candidate_snapshot=str(candidate_path),
            best_snapshot=(str(best_path) if best_path is not None else None),
            best_step=None, encoding=self._encoding, worker_device=cfg.worker_device,
            gate=gate_spec, rung_jobs=rung_jobs, random_floor_games=cfg.random_floor_games,
            random_model_sims=cfg.random_model_sims, sealbot_model_sims=cfg.sealbot_model_sims,
            kraken_model_sims=cfg.kraken_model_sims, strix_model_sims=cfg.strix_model_sims,
            seed_base=cfg.gate.seed_base, round_timeout_sec=cfg.round_timeout_sec,
            result_path=str(result_path), progress_path=str(progress_path),
            ladder_bootstrap_resamples=cfg.ladder.bootstrap_resamples,
            ladder_bootstrap_ci_level=cfg.ladder.bootstrap_ci_level,
            ladder_bootstrap_seed=cfg.ladder.bootstrap_seed,
        )
        return spec, dict(alloc), run_gate, candidate_path

    def _spawn_worker(self, spec: RoundSpec) -> Any:
        spec_path = self._work_dir / f"{spec.round_id}_spec.json"
        spec_path.write_text(json.dumps(spec.to_dict()))
        ctx = multiprocessing.get_context(self._mp_ctx_name)
        # typeshed's BaseContext omits Process (it lives on the concrete contexts);
        # every real context returned by get_context has it.
        proc = ctx.Process(  # pyright: ignore[reportAttributeAccessIssue]
            target=_worker_entry, args=(str(spec_path), spec.result_path),
            kwargs={}, daemon=True,
        )
        proc.start()
        return proc

    # ── mailbox / bounded drains ───────────────────────────────────────────────────────
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
            broken, reason = drain_or_kill(
                proc, budget_sec=budget, worker_kill_grace_sec=self._eval_cfg.worker_kill_grace_sec,
                clock=self._clock,
            )
            if broken:
                return self._finalize_round(inflight, escalated_reason=reason)
        return self._finalize_round(inflight)

    def _finalize_round(
        self, inflight: dict[str, Any], *, escalated_reason: str | None = None,
    ) -> dict[str, Any]:
        proc = inflight["proc"]
        wall_sec = max(self._clock() - inflight["t0"], 0.0)
        exit_code = getattr(proc, "exitcode", None)

        # F1 fix, layer 2 (isolation law 2, structural): the entire round-completion
        # decision (including scheduling `allocate_games` for next round, deep inside
        # `_read_worker_result` -> `_success_result`) runs under one catch-all. ANY
        # uncaught exception here — not just the KeyError RED_TEAM's Finding F1 reproduced
        # — converts to a named `eval_broken(round_completion_error)` result that IS
        # delivered (mailbox append below always runs), instead of propagating out of the
        # poller thread (silent thread death -> `poll_completed()` returns None forever ->
        # the `eval_round` heartbeat stops -> up to the watchdog staleness deadline of
        # silent hang; RED_TEAM.md Finding F1 consequences). The round terminates loudly
        # every time, never a hang, never a silent skip.
        try:
            if escalated_reason is not None:
                result = self._broken_result(inflight, reason=escalated_reason, exit_code=exit_code,
                                             wall_sec=wall_sec, phase="drain")
            elif exit_code is not None and exit_code != 0:
                reason = "killed" if exit_code < 0 else "exit_nonzero"
                result = self._broken_result(inflight, reason=reason, exit_code=exit_code,
                                             wall_sec=wall_sec, phase="worker_exit")
            else:
                result = self._read_worker_result(inflight, exit_code=exit_code, wall_sec=wall_sec)
        except Exception as exc:  # noqa: BLE001 -- deliberate catch-all, see docstring above
            result = self._round_completion_error_result(inflight, exc, wall_sec=wall_sec)

        with self._lock:
            self._inflight = None
            self._mailbox.append(result)
        return result

    def _round_completion_error_result(
        self, inflight: dict[str, Any], exc: Exception, *, wall_sec: float,
    ) -> dict[str, Any]:
        """The F1 layer-2 catch-all result: reason names the exception CLASS
        (`round_completion_error`, not a bare "something broke"), the event AND the routed
        `error` string both carry `repr(exc)` — never a swallowed exception, never a bare
        log line (isolation law 2)."""
        detail = repr(exc)
        _emit(self._sink, {
            "event": "eval_broken", "round_id": inflight["round_id"], "step": inflight["step"],
            "reason": "round_completion_error", "exit_code": getattr(inflight["proc"], "exitcode", None),
            "phase": "round_completion", "exception_class": type(exc).__name__, "detail": detail,
        })
        _LOG.exception(
            "eval_broken round_id=%s step=%s reason=round_completion_error detail=%s",
            inflight["round_id"], inflight["step"], detail,
        )
        result = build_round_result(
            step=inflight["step"], round_id=inflight["round_id"],
            rungs_config=self._eval_cfg.ladder.rungs, rung_results={}, gate_result=None,
            skipped_rungs=[], bt={"ratings": {}, "p_hat": {}}, schedule_next={},
            eval_round_wall_sec=wall_sec, eval_broken=True,
            error=f"round_completion_error: {detail}", random_wr=None,
        )
        emit_round_complete(
            self._sink, round_id=inflight["round_id"], step=inflight["step"], wall_sec=wall_sec,
            games_total=0, promoted=False, wr_sealbot=result["wr_sealbot"],
        )
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
            return self._broken_result(inflight, reason="result_missing", exit_code=exit_code,
                                       wall_sec=wall_sec, phase="worker_exit")
        except (ValueError, ResultContractError, OSError):
            return self._broken_result(inflight, reason="result_invalid", exit_code=exit_code,
                                       wall_sec=wall_sec, phase="worker_exit")
        return self._success_result(inflight, raw, wall_sec=wall_sec)

    def _broken_result(
        self, inflight: dict[str, Any], *, reason: str, exit_code: int | None,
        wall_sec: float, phase: str,
    ) -> dict[str, Any]:
        _emit(self._sink, {
            "event": "eval_broken", "round_id": inflight["round_id"], "step": inflight["step"],
            "reason": reason, "exit_code": exit_code, "phase": phase,
        })
        _LOG.error("eval_broken round_id=%s step=%s reason=%s", inflight["round_id"],
                  inflight["step"], reason)
        result = build_round_result(
            step=inflight["step"], round_id=inflight["round_id"],
            rungs_config=self._eval_cfg.ladder.rungs, rung_results={}, gate_result=None,
            skipped_rungs=[], bt={"ratings": {}, "p_hat": {}}, schedule_next={},
            eval_round_wall_sec=wall_sec, eval_broken=True, error=reason, random_wr=None,
        )
        emit_round_complete(
            self._sink, round_id=inflight["round_id"], step=inflight["step"], wall_sec=wall_sec,
            games_total=0, promoted=False, wr_sealbot=result["wr_sealbot"],
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
        self._ensure_ladder_state().record_round(round_idx, ladder_results, sink=self._sink)
        try:
            self._ensure_ladder_state().save(self._ladder_state_path)
        except LadderStateError:
            # LAW-14: a persistence failure is run-fatal — it must surface as a named
            # round failure (the eval_broken-class path), never degrade to a log line
            # [M-1]. The games ALREADY PLAYED this round are discarded along with it: the
            # ladder's own on-disk state of record (activation/graduation streaks) did not
            # durably advance, so reporting those games as a normal success would silently
            # drift the in-memory state ahead of the persisted state.
            _LOG.exception("ladder_state_persist_failed round_id=%s", inflight["round_id"])
            return self._broken_result(
                inflight, reason="ladder_persist_failed", exit_code=None,
                wall_sec=wall_sec, phase="ladder_persist",
            )

        # M-5: fold the best/gate entity into the SAME global BT fit (design §a.3 bt.py —
        # "ONE global fit across candidate + best + all rungs") — the gate's pooled W/L
        # anchors the Elo scale to best exactly like a rung would; a fit that omits it
        # still recovers rung-vs-candidate ratings but never the candidate-vs-best gap.
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

        result = build_round_result(
            step=inflight["step"], round_id=inflight["round_id"],
            rungs_config=self._eval_cfg.ladder.rungs, rung_results=rungs_raw,
            gate_result=gate_raw, skipped_rungs=skipped_rungs,
            bt={"ratings": {name: float(ratings[i]) for i, name in enumerate(entities)}, "p_hat": p_hat},
            schedule_next=schedule_next, eval_round_wall_sec=wall_sec, eval_broken=False,
            error=None, random_wr=random_raw.get("wr"), worker_pid=raw.get("worker_pid"),
            candidate_snapshot_path=inflight.get("candidate_snapshot_path"),
        )
        emit_round_complete(
            self._sink, round_id=inflight["round_id"], step=inflight["step"], wall_sec=wall_sec,
            games_total=games_total, promoted=result["promoted"], wr_sealbot=result["wr_sealbot"],
        )
        emit_rung_skip_events(inflight["round_id"], skipped_rungs, self._sink)
        return result

    # ── terminal (synchronous, ignore_stride) ───────────────────────────────────────────
    def _run_terminal_sync(
        self, model: Any, step: int, best: Any, *, best_model_step: int | None,
    ) -> dict[str, Any]:
        round_idx = self._round_counter + 1
        round_id = f"r{round_idx:06d}_{step}_terminal"
        self._round_counter = round_idx
        spec, _scheduled, _gate_scheduled, candidate_path = self._build_round_spec(
            model, step, best, round_id=round_id, round_idx=round_idx, terminal=True,
        )
        proc = self._spawn_worker(spec)
        inflight = {
            "round_id": round_id, "step": step, "proc": proc, "spec": spec,
            "t0": self._clock(), "round_idx": round_idx,
            "candidate_snapshot_path": str(candidate_path),
        }
        broken, reason = drain_or_kill(
            proc, budget_sec=self._caps.terminal_eval_hard_cap_sec,
            worker_kill_grace_sec=self._eval_cfg.worker_kill_grace_sec, clock=self._clock,
        )
        if broken:
            return self._finalize_round(inflight, escalated_reason=reason)
        return self._finalize_round(inflight)

    # ── gate-decision delegation (the ONE call site lives in promote.py) ────────────────
    def apply_gate_decision(self, result: Mapping[str, Any]) -> int | None:
        return apply_gate_decision(self._promotion, result)

    # ── teardown ─────────────────────────────────────────────────────────────────────
    def stop(self) -> None:
        self._stop_event.set()
        if self._poller.is_alive():
            self._poller.join(5.0)
        with self._lock:
            inflight = self._inflight
        if inflight is not None:
            proc = inflight["proc"]
            if proc.is_alive():
                proc.terminate()
                proc.join(_bounded_join_timeout(self._eval_cfg.worker_kill_grace_sec))


def build_eval_pipeline(
    *,
    eval_cfg: Any,
    coordinator_cfg_caps: DrainCaps,
    encoding: str,
    run_id: str,
    spool_dir: str | Path,
    ladder_state_path: str | Path,
    promotion: DeployTagHooks,
    sink: Any = None,
    heartbeat: Callable[[str], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
    mp_ctx: str = "spawn",
) -> EvalPipeline:
    """The ONE constructor — NO `device`, NO `model` parameter (isolation law 1: an
    in-process CUDA eval path is unrepresentable; models arrive only through
    `run_evaluation`'s protocol args and are IMMEDIATELY serialized-and-dropped)."""
    return EvalPipeline(
        eval_cfg=eval_cfg, caps=coordinator_cfg_caps, encoding=encoding, run_id=run_id,
        spool_dir=spool_dir, ladder_state_path=ladder_state_path, promotion=promotion,
        sink=sink, heartbeat=heartbeat, clock=clock, mp_ctx_name=mp_ctx,
    )


__all__ = [
    "DrainCaps",
    "EvalPipeline",
    "build_eval_pipeline",
    "drain_budget_sec",
    "drain_or_kill",
    "emit_round_complete",
    "emit_round_skipped_busy",
    "emit_round_started",
    "emit_rung_skip_events",
]
