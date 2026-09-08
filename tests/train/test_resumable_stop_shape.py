"""⊕ R343(c) — the ORDERED stop's shape: what a signal stop skips, and what it must not.

R343(c) orders "SIGTERM → drain under kill-grace → checkpoint → exit 0". At HEAD the fourth
step did not follow the third: `close_out` ran the terminal battery synchronously under
`terminal_eval_hard_cap_sec` (14400 s in `run6.yaml`), MEASURED once at t+67 min and still
running while the checkpoint had been safe for an hour (`ADJUDICATION_QUEUE.md`, F-R-P2B-4). On
the 12 h block R343(f) sets, that is a third of the run spent closing a run about to be
reopened.

The defect each row is the ONLY witness to:

* a stop that still costs four hours — so a RESUMABLE stop must skip the battery;
* **a stop that skips the battery when it should not**, which is the dangerous direction and
  the one a first cut got wrong. `shutdown_save` does NOT mean "an operator asked": the disk
  guard SIGTERMs its own process, and its rule is recorded AFTER `close_out` by design, so
  neither `shutdown_save` nor `abort_rule` can see it in the epilogue. The default is False and
  these rows pin it;
* the ring persisted on the one abort that fires because the disk is full — a large write onto
  a full disk, out of a LAW-14 path.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from mantis.train.coordinator import drain


class _Pipeline:
    def __init__(self) -> None:
        self.evaluations = 0

    def run_evaluation(self, *a: Any, **kw: Any) -> dict[str, Any]:
        self.evaluations += 1
        return {"eval_broken_reason": None}


def _coord(pipeline: _Pipeline) -> Any:
    return SimpleNamespace(
        eval_pipeline=pipeline, config=SimpleNamespace(terminal_eval_enabled=True),
        anchor_state=SimpleNamespace(best_model=object(), best_model_step=3),
        _train_step=1234, _sink=None, full_config={}, eval_model=object(),
        record_terminal_eval_reason=lambda reason: None,
    )


def test_a_resumable_stop_skips_the_terminal_battery() -> None:
    p = _Pipeline()
    assert drain.run_terminal_eval(_coord(p), resumable_stop=True) is None
    assert p.evaluations == 0, (
        "a run being stopped to be RESUMED has not closed, so it must not spend up to "
        "terminal_eval_hard_cap_sec running its closing measurement"
    )


def test_the_default_runs_the_battery() -> None:
    """THE PLANTED BREAK, and it is the direction that matters: anything that does not
    POSITIVELY assert a resumable stop keeps the terminal battery. A default of True here
    would silently delete the terminal round from every abort and from the clean terminus."""
    p = _Pipeline()
    drain.run_terminal_eval(_coord(p))
    assert p.evaluations == 1


def test_an_explicitly_non_resumable_stop_runs_the_battery() -> None:
    p = _Pipeline()
    drain.run_terminal_eval(_coord(p), resumable_stop=False)
    assert p.evaluations == 1


def test_the_skip_never_overrides_terminal_eval_disabled() -> None:
    """The config's own switch stays the outer authority — the skip is a narrowing of when the
    battery runs, never a widening."""
    p = _Pipeline()
    coord = _coord(p)
    coord.config = SimpleNamespace(terminal_eval_enabled=False)
    assert drain.run_terminal_eval(coord, resumable_stop=False) is None
    assert p.evaluations == 0
