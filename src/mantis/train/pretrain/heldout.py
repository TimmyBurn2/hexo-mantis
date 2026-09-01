"""Held-out policy-loss monitoring and the patience stop for BC pretrain (R328(d)).

WHY A MODULE AND NOT A FEW LINES IN `graph_route`. The stopping rule is the ONE stated risk
bound on bootstrap posture (A) — the filed adjudication's §3(A) says the over-fit risk *"is
bounded by a knob the operator already controls, how long you pretrain"* — so it is the thing
that has to be provable on its own, with planted breaks, before any pretrain consumes it. A
rule woven into the training loop can only be tested by running a training loop.

THE ESTIMATOR IS HONEST ABOUT BEING ONE. `HexgBuffer.sample_graph_batch` takes no seed, so a
held-out pass is `ceil(plies / batch_size)` SAMPLED batches — ring-equivalent in expectation,
covering about 63 % of distinct rows with replacement, and NOT an exact epoch loss. Its own
step-to-step noise is therefore measurable and is measured (`measure_noise`), because a
stopping rule whose noise exceeds the improvement it looks for is not a stopping rule. The
`min_delta` that decides "improved" is set from that measurement, never guessed.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

_LOG = logging.getLogger(__name__)


class HeldOutError(RuntimeError):
    """The held-out monitor cannot do its job: an empty ring, or a contaminated one."""


@dataclass
class PatienceStop:
    """Best-so-far tracking with a patience counter over held-out evaluations.

    `min_delta` is a REQUIRED improvement, not a tolerance on equality: with `>` alone, a loss
    that drifts down by a millionth of the estimator's own noise resets patience forever and
    the stop can never fire. That is the failure this class exists to not have.
    """

    patience: int
    min_delta: float
    best: float = math.inf
    best_at_step: int = -1
    since_best: int = 0
    evaluations: int = 0
    fired: bool = False

    def observe(self, loss: float, *, step: int) -> bool:
        """Record one held-out reading; return True iff the run should stop now."""
        self.evaluations += 1
        if loss < self.best - self.min_delta:
            self.best, self.best_at_step, self.since_best = loss, step, 0
        else:
            self.since_best += 1
            if self.since_best >= self.patience:
                self.fired = True
        return self.fired

    def counters(self) -> dict[str, Any]:
        """LAW-18: the lever under test reports its own state in-run."""
        return {
            "heldout_evaluations": self.evaluations,
            "heldout_best_policy_loss": None if self.best == math.inf else self.best,
            "heldout_best_at_step": self.best_at_step,
            "heldout_evals_since_best": self.since_best,
            "heldout_patience": self.patience,
            "heldout_min_delta": self.min_delta,
            "heldout_stop_fired": self.fired,
        }


@dataclass
class HeldOutMonitor:
    """Evaluates the held-out ring's POLICY loss on a cadence and owns the stop decision.

    `eval_batches` is DERIVED from the ring, not chosen: `ceil(plies / batch_size)`, so the
    pass is one ring-equivalent of samples whatever the ring's size. A fixed batch count would
    be a different amount of evidence on every corpus.
    """

    ring: Any
    spec: Any
    batch_size: int
    eval_every: int
    stop: PatienceStop
    caps_provider: Any
    sample_threads_provider: Any
    eval_batches: int = 0
    plies: int = 0
    history: list[tuple[int, float]] = field(default_factory=list)

    @classmethod
    def build(cls, *, ring: Any, spec: Any, plies: int, batch_size: int, eval_every: int,
              patience: int, min_delta: float, caps_provider: Any,
              sample_threads_provider: Any) -> HeldOutMonitor:
        """Construct with `eval_batches` derived from the held-out ring's own ply count.

        Raises:
            HeldOutError: the ring is empty, or the cadence would never fire.
        """
        if plies <= 0:
            raise HeldOutError(
                "the held-out ring holds zero plies — a held-out loss over nothing is a "
                "number with no producer (R69), not a small number."
            )
        if eval_every <= 0:
            raise HeldOutError(
                f"eval_every={eval_every} never fires, so the stop can never fire and the "
                "budget silently becomes the only bound."
            )
        return cls(
            ring=ring, spec=spec, batch_size=batch_size, eval_every=eval_every,
            stop=PatienceStop(patience=patience, min_delta=min_delta),
            caps_provider=caps_provider, sample_threads_provider=sample_threads_provider,
            eval_batches=max(1, math.ceil(plies / batch_size)), plies=plies,
        )

    def evaluate(self, trainer: Any) -> float:
        """Mean held-out POLICY loss over one ring-equivalent of sampled batches."""
        from mantis.train.coordinator.dispatch import run_declared_eval_step  # noqa: PLC0415

        total = 0.0
        for _ in range(self.eval_batches):
            info = run_declared_eval_step(
                trainer, self.ring, self.spec, batch_size=self.batch_size,
                caps_provider=self.caps_provider,
                sample_threads_provider=self.sample_threads_provider,
            )
            total += float(info["policy_loss"])
        return total / self.eval_batches

    def measure_noise(self, trainer: Any, *, repeats: int = 2) -> float:
        """Spread of the estimator on an UNCHANGED model — the floor `min_delta` must clear.

        Run BEFORE the first optimizer step. Any difference between these readings is the
        sampler's, not the model's, because nothing moved in between.
        """
        readings = [self.evaluate(trainer) for _ in range(repeats)]
        spread = max(readings) - min(readings)
        _LOG.info("bc_heldout_noise readings=%s spread=%.6g", readings, spread)
        return spread

    def maybe_evaluate(self, trainer: Any, *, step: int) -> tuple[bool, float | None]:
        """Evaluate if `step` is on the cadence; return `(should_stop, loss_or_None)`."""
        if step % self.eval_every:
            return self.stop.fired, None
        loss = self.evaluate(trainer)
        self.history.append((step, loss))
        should_stop = self.stop.observe(loss, step=step)
        _LOG.info("bc_heldout step=%d policy_loss=%.6f best=%.6f since_best=%d stop=%s",
                  step, loss, self.stop.best, self.stop.since_best, should_stop)
        return should_stop, loss
