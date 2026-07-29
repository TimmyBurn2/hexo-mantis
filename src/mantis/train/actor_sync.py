"""ActorSync — continuous, unconditional actor weight sync (WP-UNFREEZE, R49).

Pushes the learner's current inference weights into the injected sync target on a
step-modulo cadence, unconditionally: nothing else participates in the decision to
sync. Owns THE ONE readable ``actor_ckpt_step`` field — ``maybe_sync``'s success path
is its single producer. Holds no reference to any other subsystem; the no-cross-read
law is enforced structurally by tests/train/test_actor_sync_isolation.py.

Write discipline (producer honesty, R4/LAW-14): ``_actor_step`` advances ONLY after
both target calls return. A raising target leaves the recorded step untouched — the
lag invariant then reports the truth (the actor did not get the weights) — and the
exception travels straight up to the coordinator step (fail loud, no swallow).

Thread-safety: ``_actor_step`` is a plain int written by the coordinator thread and
read by the watchdog thread; each write is a single reference assignment under the
GIL, so no lock is needed.

LAW-18: the cadence is a lever under test — every sync emits an ``actor_sync`` event
carrying the lever's own fire-rate fields through the injected sink.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any


class ActorSync:
    """The train-side sync engine. The coordinator is the only driver (one
    ``maybe_sync`` call per training step): no thread, no file poll, no discovery."""

    def __init__(
        self,
        *,
        target: Any,
        state_dict_fn: Callable[[], dict],
        step_fn: Callable[[], int],
        cadence_steps: int,
        sink: Any,
        run_id: str,
    ) -> None:
        self._target = target
        self._state_dict_fn = state_dict_fn
        self._step_fn = step_fn
        self._cadence_steps = int(cadence_steps)
        self._sink = sink
        self.run_id = run_id
        # Pre-first-sync posture: a watchdog read in this window sees lag 0, never a
        # false fire (the ctor snapshot IS the learner's current step).
        self._actor_step = int(step_fn())
        self._synced_once = False
        self._sync_count = 0

    def actor_ckpt_step(self) -> int:
        """THE one readable field: the training step whose weights the actor runs."""
        return self._actor_step

    def maybe_sync(self, train_step: int) -> bool:
        """One coordinator-driven check per training step. The FIRST call syncs
        unconditionally (establishes the invariant without assuming boot parity
        between the injected target and the learner); thereafter sync iff
        ``train_step`` sits on the cadence boundary."""
        step = int(train_step)
        if self._synced_once and step % self._cadence_steps != 0:
            return False
        started = time.monotonic()
        lag_pre = int(self._step_fn()) - self._actor_step
        # One sync = weights, then step, then record — in that order. The target's
        # weight swap is synchronous, so post-call the actor IS running these weights
        # and the recorded step is honest by construction.
        self._target.sync_inference_weights(self._state_dict_fn())
        self._target.update_checkpoint_step(step)
        self._actor_step = step
        self._synced_once = True
        self._sync_count += 1
        self._sink.emit({
            "event": "actor_sync",
            "step": step,
            "actor_ckpt_step": self._actor_step,
            "lag_steps_pre_sync": lag_pre,
            "cadence_steps": self._cadence_steps,
            "sync_count": self._sync_count,
            "duration_ms": round((time.monotonic() - started) * 1000.0, 3),
        })
        return True


__all__ = ["ActorSync"]
