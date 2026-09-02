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

from mantis.bots.resolve import SKIP_REASON_MARKERS
from mantis.config.resolve.eval_posture import (
    resolve_ply_cap_adjudication,
    resolve_strength_floor,
)
from mantis.config.resolve.fused_graph_caps import FusedGraphCapsSpec
from mantis.config.resolve.inference_batching import InferenceBatchingSpec
from mantis.eval.bt import fit_bt, predict_p
from mantis.eval.child_memory import EVENT as EVAL_DEVICE_MEMORY_EVENT
from mantis.eval.errors import EvalBrokenReason, LadderStateError, ResultContractError
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
) -> EvalBrokenReason | None:
    """Bounded join -> (if still alive) terminate -> bounded join -> kill -> bounded join.

    Returns the typed reason the drain escalated with, or `None` when the child exited
    inside its budget; every join carries a bound (isolation law 2).

    WP12-R Phase O (R152/R79): the old `(bool, str)` return was two authorities for one
    fact — a `True` beside a healthy-drain spelling, and a `False` beside `join_timeout`,
    were both constructible — and the healthy half was a reason spelling no member spells.
    Absence IS the clean state, so there is no second field left to disagree.
    """
    del clock  # the caller advances/consults its own clock; every join below is bounded
    proc.join(_bounded_join_timeout(budget_sec))
    if not proc.is_alive():
        return None
    proc.terminate()
    proc.join(_bounded_join_timeout(worker_kill_grace_sec))
    proc.kill()
    proc.join(_bounded_join_timeout(worker_kill_grace_sec))
    return EvalBrokenReason.JOIN_TIMEOUT


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
    sink: Any, *, round_id: str, step: int, wall_sec: float, games_total: int | None,
    promoted: bool | None, wr_sealbot: float | None, progress: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """R319(e)(i): `games_total` is `int | None`, and `None` is the BROKEN-round value.

    AUDIT-1 F-28/B04: `promoted` is `bool | None`, and `None` means NO PROMOTION DECISION WAS
    TAKEN — the gate was not scheduled this round, or there was no best anchor to play, or the
    round broke before the gate block. `False` means the gate ran and refused. Those are
    different facts about the run and both used to be `false`.

    It used to be a hardcoded `0` on every broken path, which is a DEFAULT WEARING A
    MEASUREMENT'S CLOTHES: a reader cannot tell "the round played no games" from "the round
    was killed before it could report", and RECAL-SITTING-3 published the former having
    measured only the latter (§8.1 of the sitting record). `None` is unreadable as a count, so
    the mistake is not available to make twice.

    `progress` carries the child's last per-game progress row when one exists (R319(e)(ii)) —
    so a broken round now says HOW FAR it got instead of nothing at all.
    """
    payload = {
        "event": "eval_round_complete", "round_id": round_id, "step": step,
        "wall_sec": wall_sec, "games_total": games_total, "promoted": promoted,
        "wr_sealbot": wr_sealbot, "progress": progress,
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
    """The strength floor's ONE event — the probe's verdict AND its in-run fire rate.

    One event rather than a pass channel and a separate skip channel, because they are one
    fact ("what did this round's floor probe decide") and two channels for one fact is how
    the two drift. LAW-18 wants a FIRE RATE and not a flag, so both running totals ride the
    payload: `checked_total` is every armed round that probed, `skipped_total` the subset
    that short-circuited. A reader with only the second cannot tell a floor that never fires
    from a floor that never ran.

    Emitted ONLY on an armed round — the disarmed posture produces no `strength_floor` key on
    the worker result, so this builder is never reached and the stream is byte-unchanged.
    """
    payload = {
        "event": "eval_strength_floor", "round_id": round_id, "step": step,
        **dict(floor), "checked_total": checked_total, "skipped_total": skipped_total,
    }
    _emit(sink, payload)
    return payload


def emit_ply_cap_adjudication(
    sink: Any, *, round_id: str, step: int, adjudication: Mapping[str, Any],
) -> dict[str, Any]:
    """The ply-cap criterion's in-run fire rate (LAW-18), one event per armed round.

    The tally is the ADJUDICATOR's own count of the capped games it saw and how it resolved
    them, carried up from the worker rather than re-derived from the aggregates — the
    aggregates record a winner, not whether a rule or a criterion produced it. Emitted only
    on an armed round, for the same reason as the floor event above.
    """
    payload = {
        "event": "eval_ply_cap_adjudication", "round_id": round_id, "step": step,
        **dict(adjudication),
    }
    _emit(sink, payload)
    return payload


#: The CLOSED skip-reason partition (WP12-R Phase A, DESIGN_A §2.7(4)). Order is the
#: resolver's own: the R139-ruled skip first, then the three ways a sealbot rung can fail to
#: resolve. The set is closed on purpose — a reason nothing recognises must be a loud failure,
#: never a fifth bucket invented at emission time, because the whole value of the partition is
#: that "4 rungs skipped as ruled" and "6 rungs skipped because the box is misconfigured" stop
#: looking identical to every consumer.
SKIP_REASON_CLASSES: tuple[str, ...] = (
    "operator_authorized", "vendor_absent", "build_absent", "load_failed",
)


def _classify_skip_reason(reason: str) -> str | None:
    """The reason's class, or None when nothing recognises it.

    Classification is by marker substring against `mantis.bots.resolve.SKIP_REASON_MARKERS` —
    the SAME literals the refusal strings are built from, imported rather than re-transcribed,
    so a resolver whose wording drifted out of this classifier's reach cannot do it silently.
    Exactly one marker must match: zero is an unrecognised reason and two would mean the
    partition stopped partitioning, and both are reported the same way (loudly, with no
    counter event) rather than guessed at.
    """
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
    """Publish the CHILD's own device-memory readout (RECAL-PREP, R308(g)(ii)).

    Emitted from the child's payload rather than from anything the parent measures: the term
    this exists to bound is the child's, in the child's own process and CUDA context, and a
    parent-side reading of it is exactly the substitute reading that has under-measured it
    three times. Unconditional whenever the payload arrives — including the `available: false`
    arm, where every counter is `null`. A reader must be able to tell "this round had no
    counters" from "nobody looked", and dropping the unmeasured rounds would silently bias the
    series a growth verdict is taken over.
    """
    payload = {
        "event": EVAL_DEVICE_MEMORY_EVENT, "round_id": round_id, "step": step,
        "device_memory": dict(device_memory),
    }
    _emit(sink, payload)
    return payload


def emit_rung_skip_events(round_id: str, skipped: list[Mapping[str, str]], sink: Any) -> None:
    """Per skipped rung: an `eval_rung_skipped` event, an ERROR log line, AND — new in WP12-R
    Phase A — one `eval_rung_skip_class` counter event carrying the running per-class count.

    The fourth channel exists because R164 made LAW-18 mean IN-RUN FIRE-RATE, not "there is a
    log line somewhere". The first three channels all answer "some rungs skipped"; none of
    them answers the question R143 says has to be legible WHILE the run is going — whether
    these are the skips the operator AUTHORISED. Hence one counter event per rung, emitted
    alongside its skip through the same injected sink: a single aggregate at round end would
    be precisely the "log line somewhere" R164 ruled out.
    """
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
    """The `.tmp` the worker writes for `result_path`, derived the way the worker derives it.

    `worker.py` spells it `target.with_suffix(target.suffix + ".tmp")`, which for a
    `…_result.json` target is exactly `str(target) + ".tmp"`. Two derivations of one fact in
    two files is a drift risk, so the two spellings are pinned against each other by a producer
    row rather than trusted — the cutover battery is going to edit the worker's write lines.
    """
    return Path(result_path + ".tmp")


def _remove_result_tmp(result_path: str) -> None:
    """Delete THIS round's `<result>.json.tmp` once its writer is gone (F-816-20 item 3a).

    THE LITTER. The worker writes `tmp.write_text(...)` then `tmp.replace(target)`. A kill
    between those two lines leaves the `.tmp` on disk FOREVER: round ids are unique, so nothing
    ever writes that name again. Atomicity is intact — a reader sees the complete old file or
    nothing — so this is litter, never corruption, but litter with no expiry.

    It only ever removes the `.tmp`, NEVER the target, so the worker's tmp+replace stays
    exactly as atomic as it was. Callers guard on the child being confirmed dead: unlinking a
    tmp a LIVE writer is about to `replace()` would turn litter into a failed round, which is
    the fix being worse than the defect.

    NEVER FATAL, and that is load-bearing rather than polite: this runs in `_finalize_round`'s
    un-caught prologue, outside the catch-all that converts anything to
    `eval_broken(round_completion_error)`. A round is not broken by a file we could not delete,
    and an exception escaping here would kill the poller thread silently.
    """
    try:
        _result_tmp_path(result_path).unlink(missing_ok=True)
    except OSError:
        _LOG.debug("stale eval result tmp not removed: %s.tmp", result_path, exc_info=True)


def _drop_result_tmp_if_writer_gone(inflight: dict[str, Any]) -> None:
    """Remove one round's `.tmp` iff its worker is confirmed dead. The ONE decision, shared by
    the two teardown routes that reach it (`_finalize_round` and `stop()`).

    `spec` is read through `.get` and checked FIRST so the liveness call is short-circuited
    when a record carries no spec: both call sites sit in un-caught prologues, and a `KeyError`
    or `AttributeError` there would kill the poller thread — the silent-thread-death class the
    catch-all further down exists to prevent, and which it cannot reach from here.
    """
    spec = inflight.get("spec")
    proc = inflight.get("proc")
    if spec is None or proc is None or proc.is_alive():
        return
    _remove_result_tmp(spec.result_path)


def read_progress(spec: Any) -> dict[str, Any] | None:
    """R319(e)(ii): the CHILD's last per-game progress row, or `None` if it wrote none.

    Read-only and total: any failure to read returns `None`. This is reporting, and a
    diagnostic file must never be able to fail a round that was otherwise fine — the same
    reasoning the child's writer disables itself on `OSError` rather than raising.

    ESCALATION SEMANTICS ARE UNCHANGED BY THIS FUNCTION AND MUST STAY SO (R319(e)(ii)): no
    caller may branch on the value. It exists so a killed round can say how far it got, which
    is exactly what nobody could tell during RECAL-SITTING-3's two 3600 s drives.
    """
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
    Torch/worker imports stay LAZY — this function body is the only place the parent
    process's import of `mantis.eval.pipeline` ever touches `mantis.eval.worker`.

    F-816-14 (R284(f)): the FIRST thing it does is ask the kernel to kill it when its parent
    dies. `daemon=True` and the `stop()` teardown below both cover the paths where the parent
    RUNS; this covers the path where the parent is killed outright, which is the one that left a
    child holding 458 MiB of the card after a SIGTERM. Armed before the torch import so the
    window in which this process is both heavy and unreapable is as short as it can be."""
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
        leaf_build_threads: int = 1,
        run_id: str,
        spool_dir: str | Path,
        allocator_posture: str | None = None,
        ladder_state_path: str | Path,
        promotion: DeployTagHooks,
        sink: Any = None,
        heartbeat: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        mp_ctx_name: str = "spawn",
    ) -> None:
        # F-816-20 item 2. A WHITELIST equality on the NAME STRING, checked at construction
        # and BEFORE any directory is made: a round is a long way into a run, so a refusal
        # that arrives thirty minutes in is a worse instrument than one that arrives at boot,
        # and a pipeline that is going to be refused should not leave a work dir behind.
        # Keyed on the name and never on the context OBJECT, deliberately: five suites
        # monkeypatch `multiprocessing.get_context` and take this default, so a check that
        # inspected the returned context would red every one of them.
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
        #: The graph inference forward's memory bound (F-816-10 D-1), resolved ONCE in the
        #: PARENT through its one read path and carried to every round's `RoundSpec`. REQUIRED
        #: and keyword-only, with no default, because the eval child is a SECOND allocator on
        #: the same card that no in-process bound can see: a default here would be a value
        #: nobody minted, standing in for the one the operator measured. `None` is the GRID
        #: arm, where there is no fused graph forward to bound.
        self._fused_graph_caps = fused_graph_caps
        #: The deploy head's MCTS leaf-batch width (R318(b)), carried to every round's
        #: `RoundSpec`. NOT defaulted, for `fused_graph_caps`' reason: it is the config's own
        #: `selfplay.leaf_batch_size`, and a default here would be a search regime nobody
        #: minted standing in for the one the net was trained under.
        self._leaf_batch_size = int(leaf_batch_size)
        #: The graph collector's batching geometry (PERF-TRANCHE-1 G-2, ledger F-2), resolved
        #: ONCE in the parent and carried to every round's `RoundSpec`. NOT defaulted, for
        #: `leaf_batch_size`' reason: these two knobs were LITERALS in the child's hand-made
        #: server dict, and a default here would put them straight back.
        self._inference_batching = inference_batching
        #: The eval leaf-graph build's WIDTH (NIGHTRUN-1 E1), derived ONCE in the parent by
        #: `resolve_leaf_build_threads` and carried to every round's `RoundSpec`.
        #: DEFAULTED TO 1, and the reason is the same one `HexgBuffer.sample_graph_batch`'s
        #: `n_threads` carries: 1 is the SERIAL path — the exact-parity control and the
        #: behaviour that shipped — not a host reservation this layer invented. What keeps
        #: that from becoming a silently-disabled lever is a PRODUCER TEST over `run.py`'s
        #: own AST (`tests/eval/test_leaf_build_threads_wiring.py`), because a widened build
        #: that stopped being threaded would show up as nothing at all: correct results,
        #: 95 % of the eval path back in a serial loop.
        self._leaf_build_threads = max(1, int(leaf_build_threads))
        #: The allocator REGIME the round's caps were fitted under (RECAL-PREP, R308(g)(i)),
        #: resolved ONCE in the parent and carried to every round's `RoundSpec`. Unlike
        #: `fused_graph_caps` this one carries a DEFAULT, and the reason is stated rather than
        #: convenient: the safety here is not "a value is present" but "a cuda child that was
        #: handed no token RAISES" (`assert_posture_token`), so `None` cannot excuse an
        #: assertion — it can only fail one. `None` is the not-cuda arm.
        self._allocator_posture = allocator_posture
        self._run_id = run_id
        self._spool_dir = Path(spool_dir)
        self._spool_dir.mkdir(parents=True, exist_ok=True)
        # Spec/result/progress sidecar files live in a SIBLING directory, never nested
        # under spool_dir: spool_dir holds ONLY model snapshot (.pt) files — the LAW-12
        # one-loader carve-out this WP pins (test_snapshots_are_not_checkpoints walks
        # every file under spool_dir and torch.load()s it).
        #
        # SCOPED BY `run_id` (RQ-13 / clause (l)). Nothing locks an out-dir, and round ids are
        # a PER-RUN counter (`r{round_idx:06d}_{step}`), so two runs sharing one out-dir wrote
        # identical sidecar filenames into one directory. `run_id` is safe to put in a path by
        # TYPE rather than by inspection: the schema constrains it to `^[a-z0-9][a-z0-9_\-]*$`
        # (`config/schema/core.py`), so it cannot contain a separator or a `..` — a sanitizer
        # here would be a second authority for a constraint pydantic already enforces.
        self._work_dir = self._spool_dir.parent / f"{self._spool_dir.name}.work" / self._run_id
        self._work_dir.mkdir(parents=True, exist_ok=True)
        # F-816-20 item 3a, the sweep. This is the ONLY handle the "the run itself was
        # SIGKILLed" case has: no code ran in that process, so the only thing that can ever
        # clean its litter is a LATER pipeline over the same work dir — which is exactly what
        # a `--resume-from` relaunch into the same `--out-dir` is. The precondition is that at
        # CONSTRUCTION this pipeline has no live writer, and a second pipeline over one work
        # dir is already impossible: `build_eval_pipeline` has one call site, once per
        # process, and the dir is derived from `--out-dir` AND `run_id`. The second half is
        # new with RQ-13 and it is the half that makes the precondition structural: derived
        # from the out-dir alone, the claim was false exactly when two runs shared one, which
        # is the case A5 raised and nothing prevents. A relaunch still reaches its OWN litter
        # because `--resume-from` supplies the same `--config`, hence the same run_id and the
        # same out-dir (`run.py`: `run_id = config.run_id`, `log_dir = out_dir / "logs"`). Only the `.tmp` suffix is in
        # scope — results, specs, progress sidecars and snapshots are never touched. No age
        # threshold: that would be an unmeasured constant bought to solve a race the
        # precondition already excludes.
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
        #: Times `_finalize_round` was re-entered for a round it had already finalised and
        #: the duplicate was SUPPRESSED. Reads 0 in a healthy run. Non-zero means the poll
        #: loop and a drain both reached the same in-flight round — see the guard in
        #: `_finalize_round` for why that double-counts a promotion (LAW-18).
        self._double_finalize_suppressed = 0
        #: The strength floor's LAW-18 fire rate, as a pair rather than a single count:
        #: `checked` is every armed round that probed, `skipped` the subset the probe
        #: short-circuited. Both stay 0 for the whole life of a pipeline whose config mints
        #: `eval.strength_floor: null`, because the worker result then carries no floor key
        #: at all. A skip count without its denominator cannot tell "the floor never fires"
        #: from "the floor never ran", which is precisely the distinction LAW-18 was written
        #: about.
        self._floor_checked_total = 0
        self._floor_skipped_total = 0

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
        # The round exceeded `round_timeout_sec`, its PROGRESS budget. The joins above are
        # the kill sequence, not the cause — this path reported `JOIN_TIMEOUT` until R316(c),
        # which told an operator the child would not exit when it had in fact been killed for
        # running long. The genuine join timeout is `_drain_escalate`'s, and it keeps the name.
        self._finalize_round(inflight, escalated_reason=EvalBrokenReason.ROUND_TIMEOUT)

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
            # The two early-strength postures, resolved through their ONE read path (R1/LAW-08)
            # and carried to the child. Both are `None` for every committed config.
            ply_cap_adjudication=resolve_ply_cap_adjudication(cfg),
            strength_floor=resolve_strength_floor(cfg),
            # Resolved once in the parent (F-816-10 D-1), not re-read here: the child has no
            # `RunConfig` and its `LocalInferenceEngine` builds its graph server from a
            # hand-made dict, so this is the only way the bound reaches the second allocator.
            fused_graph_caps=self._fused_graph_caps,
            # R318(b), same seam and same reason: the deploy head must search at the width the
            # net's targets were generated at, and the child cannot read the config to find it.
            leaf_batch_size=self._leaf_batch_size,
            # G-2, same seam and same reason: the child's graph server wrote its pop width
            # and pop deadline as literals, and 33 % of the eval path's ms/sim was the
            # deadline one of them set (ledger F-2).
            inference_batching=self._inference_batching,
            # NIGHTRUN-1 E1, same seam and same reason: the leaf build's width is a HOST
            # reservation and the child has no config to derive one from.
            leaf_build_threads=self._leaf_build_threads,
            # Same seam, same reason: a posture is a property of the PROCESS environment, so
            # the parent's boot assertion says nothing about the child's, and the child has no
            # config to resolve one from.
            allocator_posture=self._allocator_posture,
        )
        return spec, dict(alloc), run_gate, candidate_path

    def _spawn_worker(self, spec: RoundSpec) -> Any:
        # F-816-20 item 1. THE FIRST STATEMENT, ahead of the spec write: the single choke
        # point both callers pass through, so the invariant lives in one place rather than in
        # two copies free to drift.
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
        # typeshed's BaseContext omits Process (it lives on the concrete contexts);
        # every real context returned by get_context has it.
        proc = ctx.Process(  # pyright: ignore[reportAttributeAccessIssue]
            target=_worker_entry, args=(str(spec_path), spec.result_path),
            kwargs={}, daemon=True,
        )
        proc.start()
        from mantis.train.lifecycle.signals import register_child
        register_child(proc)
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
        # ONCE-ONLY (item 5(c)). Two independent routes finalise: `_poll_loop` (proc no
        # longer alive, or round timeout via `_escalate_and_finalize`) and the drain
        # (`drain_pending` / the terminal-eval path). Both read `self._inflight` and then
        # act, and `self._inflight` is not cleared until the END of this method — so both
        # can hold the SAME dict and finalise it twice. That is not a harmless repeat: it
        # appends the round's result to the mailbox TWICE, calls `unregister_child` twice,
        # persists the ladder twice, and — through `apply_gate_decision` on the second copy
        # — can promote off one round's games counted as two.
        #
        # The latch lives on the `inflight` dict rather than on `self`, because it must be
        # per-ROUND: a `self`-level flag would have to be reset between rounds and a missed
        # reset silently disables the guard forever.
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
                # it is still in flight, which `drain_pending`/`poll_completed` already
                # treat as "nothing ready" (their declared `dict | list | None`).
                return inflight.get("_result")
            inflight["_finalized"] = True

        proc = inflight["proc"]
        from mantis.train.lifecycle.signals import unregister_child
        unregister_child(proc)
        # F-816-20 item 3a, case 1 — the child died, the run lives. Every finalising route
        # (`_poll_loop`, `_escalate_and_finalize`, `drain_pending`, `_run_terminal_sync`)
        # converges here, so one guarded unlink at this point covers all four. It is placed in
        # this un-caught prologue deliberately: it has been made non-raising by construction,
        # and a deletion failure must never manufacture a broken round.
        _drop_result_tmp_if_writer_gone(inflight)
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
                # `phase` is a FUNCTION of the reason, which is why it stays on the payload rather
                # than folding into the enum. Both escalating routes arrive here: the drain's
                # genuine join timeout, and `_poll_loop`'s round-budget kill. A constant "drain"
                # would send a supervisor triaging a round timeout to look at the drain budget.
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
            # The traceback is logged HERE, at the raising site, and not inside the one
            # emitter: `_LOG.exception` is only correct with a live exception in flight,
            # and `_broken_result`'s other call sites (the drain and worker-exit arms
            # above) have none. This is the shape the sibling exception-bearing route
            # (`ladder_persist_failed`) already uses at `_success_result`'s catch, so both
            # routes log identically and the emitter stays uniform across all seven
            # reasons. `repr(exc)` says WHAT was raised; only the traceback says WHERE, and
            # on a catch-all that is the whole diagnostic value — "never a swallowed
            # exception, never a bare log line" (isolation law 2).
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
        """THE broken-round emitter — one event, one payload builder, one
        `build_round_result` call site for every one of the seven routes (R152).

        `detail`/`exception_class` are the two payload extras the round-completion route
        carries; they are emitted iff present, so the payload of the other six routes is
        byte-unchanged. The typed `reason` is written ONCE and read by both the event and
        the routed mapping, so the stream and `promote.py` cannot disagree.
        """
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
            # R319(e)(i): None, never 0 — a broken round MEASURED nothing, and a count here is
            # a default a reader will mistake for one (it already was, §8.1).
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
        # AUDIT-1 F-28/B02. The worker child has no `LadderState`, so it used to stamp
        # `"status": "active"` on every rung — including a SATURATED rung playing its
        # off-cadence calibration games. The status is read HERE, from the one authority, and
        # BEFORE `record_round`: what a reader wants is the status the rung was PLAYED under,
        # not the one recording this round's result produced.
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
            # LAW-14: a persistence failure is run-fatal — it must surface as a named
            # round failure (the eval_broken-class path), never degrade to a log line
            # [M-1]. The games ALREADY PLAYED this round are discarded along with it: the
            # ladder's own on-disk state of record (activation/graduation streaks) did not
            # durably advance, so reporting those games as a normal success would silently
            # drift the in-memory state ahead of the persisted state.
            _LOG.exception("ladder_state_persist_failed round_id=%s", inflight["round_id"])
            return self._broken_result(
                inflight, reason=EvalBrokenReason.LADDER_PERSIST_FAILED, exit_code=None,
                wall_sec=wall_sec, phase="ladder_persist",
                detail=f"ladder state persist failed: {self._ladder_state_path}",
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
        # AUDIT-1 F-05. The strength-floor probe plays REAL games, and on a REFUSED floor it
        # plays the only games of the round: the worker returns from PHASE 0 with `gate=None`,
        # `rungs={}` and `random.games = 0`, so the three terms above sum to a computed 0 while
        # `eval_strength_floor.games` beside it reports N. `games_total` is the round's games,
        # not its ladder games — and a fabricated 0 here is exactly the misread RECAL §8.1 paid
        # for. Run5 and shakedown both ARM the floor, so this is reachable, not hypothetical.
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
            # R324(d). The floor's verdict has to reach the LAW-15 gate, and the ONLY route
            # from the worker child to that gate is this mapping. `_emit_posture_events`
            # below reads the same `raw` key for the event channel; neither is the other's
            # source, so a floor payload that stops arriving silences BOTH rather than
            # leaving one of them reporting a stale verdict.
            strength_floor=raw.get("strength_floor"),
        )
        emit_round_complete(
            self._sink, round_id=inflight["round_id"], step=inflight["step"], wall_sec=wall_sec,
            games_total=games_total,
            # AUDIT-1 F-28/B04. `promoted: False` used to cover three different rounds — the
            # gate ran and refused, the gate was not scheduled, there was no best anchor to
            # play against — and a reader counting "rounds that failed the gate" counted all
            # three. A promotion DECISION was taken iff the worker returned a gate result;
            # derived HERE from the worker payload, because `mantis.eval.rounds` is a frozen
            # producer under R118/A-1 (PREREG_A §8 abort 8) and this repair is not the act
            # that lifts a freeze.
            promoted=(result["promoted"] if gate_raw else None),
            wr_sealbot=result["wr_sealbot"],
            progress=read_progress(inflight.get("spec")),
        )
        emit_rung_skip_events(inflight["round_id"], skipped_rungs, self._sink)
        device_memory = raw.get("device_memory")
        if device_memory is not None:
            emit_device_memory(
                self._sink, round_id=inflight["round_id"], step=inflight["step"],
                device_memory=device_memory,
            )
        self._emit_posture_events(inflight, raw)
        return result

    def _emit_posture_events(self, inflight: dict[str, Any], raw: Mapping[str, Any]) -> None:
        """The two armed-posture channels, driven by the worker payload's OWN key set.

        Presence of the key IS the arming evidence, and it comes from the child that actually
        played the round — not from the parent's config read, which would report "armed" for
        a round the worker never applied it to. On a disarmed run neither key exists, neither
        counter moves and neither event is emitted, so the stream is byte-identical.
        """
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

    # ── terminal (synchronous, ignore_stride) ───────────────────────────────────────────
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
        # AUDIT-1 F-28/B05. The terminal round emitted `eval_round_complete` with no
        # `eval_round_started` beside it, while the `eval_round_wall` manifest row names the
        # PAIR as its producer — so the one round that runs at the very end of a run, whose
        # wall time is exactly what the drain budget is judged on, had no start timestamp to
        # subtract from. The kick path has emitted it since the beginning; this is the same
        # emit, at the same point in the sequence (after the spawn, before the drain).
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
        # `_finalize_round` returns `None` only when the round was ALREADY finalised by
        # another route. `inflight` here is local to this call and has never been published
        # to `self._inflight`, so no other route can hold it and the suppression arm is
        # structurally unreachable — asserted rather than assumed, so a future change that
        # DOES publish it fails loudly instead of returning a `None` the caller unpacks.
        result = (self._finalize_round(inflight, escalated_reason=reason) if reason is not None
                  else self._finalize_round(inflight))
        assert result is not None, (
            "the terminal-eval round was finalised twice — its inflight record is local to "
            "this call and must be unreachable from the poller and the drain"
        )
        return result

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
            from mantis.train.lifecycle.signals import unregister_child
            unregister_child(proc)
            if proc.is_alive():
                proc.terminate()
                proc.join(_bounded_join_timeout(self._eval_cfg.worker_kill_grace_sec))
                if proc.is_alive():
                    proc.kill()
                    proc.join(_bounded_join_timeout(self._eval_cfg.worker_kill_grace_sec))
            # F-816-20 item 3a on the teardown route (review RC-2). This method NEVER calls
            # `_finalize_round` — it runs its own terminate -> join -> kill -> join — and it is
            # called unconditionally from `compose_run`'s teardown on EVERY run exit. A round
            # in flight at ordinary shutdown is routine, not pathological, so without this line
            # the commonest producer of the litter is the one route the per-round unlink does
            # not reach. Same guard, same decision function: a writer that survived the kill
            # keeps its tmp.
            _drop_result_tmp_if_writer_gone(inflight)


def build_eval_pipeline(
    *,
    eval_cfg: Any,
    coordinator_cfg_caps: DrainCaps,
    encoding: str,
    fused_graph_caps: FusedGraphCapsSpec | None,
    inference_batching: InferenceBatchingSpec | None,
    leaf_batch_size: int,
    run_id: str,
    spool_dir: str | Path,
    ladder_state_path: str | Path,
    promotion: DeployTagHooks,
    leaf_build_threads: int = 1,
    allocator_posture: str | None = None,
    sink: Any = None,
    heartbeat: Callable[[str], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
    mp_ctx: str = "spawn",
) -> EvalPipeline:
    """The ONE constructor — NO `device`, NO `model` parameter (isolation law 1: an
    in-process CUDA eval path is unrepresentable; models arrive only through
    `run_evaluation`'s protocol args and are IMMEDIATELY serialized-and-dropped)."""
    return EvalPipeline(
        eval_cfg=eval_cfg, caps=coordinator_cfg_caps, encoding=encoding,
        fused_graph_caps=fused_graph_caps, inference_batching=inference_batching,
        leaf_batch_size=leaf_batch_size, leaf_build_threads=leaf_build_threads,
        run_id=run_id,
        allocator_posture=allocator_posture,
        spool_dir=spool_dir, ladder_state_path=ladder_state_path, promotion=promotion,
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
