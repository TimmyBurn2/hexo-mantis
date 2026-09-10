"""The BC stopping mechanism: forward-only held-out loss, and a patience stop.

The stopping rule is the ONE stated risk bound on the bootstrap posture — the value head's
over-fit risk is bounded by how long you pretrain — and the knob did not exist until it was built
here, before the first BC run.

THE ROWS ARE THE BREAKS: a stopping rule has two symmetric failure modes that both look healthy
from outside, one that can never fire (the run silently becomes budget-bound) and one that fires
on noise (the run stops before it learns).
"""
from __future__ import annotations

import math

import pytest

from mantis.train.pretrain.heldout import HeldOutError, HeldOutMonitor, PatienceStop


def test_pb6_a_worse_reading_does_not_reset_patience() -> None:
    """If a WORSE loss reset the counter, the stop could never fire."""
    stop = PatienceStop(patience=2, min_delta=0.0)
    assert not stop.observe(1.0, step=1)
    assert not stop.observe(2.0, step=2)          # worse — must COUNT, not reset
    assert stop.observe(3.0, step=3), "two non-improving readings must fire patience=2"
    assert stop.fired and stop.best == 1.0 and stop.best_at_step == 1


def test_pb7_an_improvement_smaller_than_min_delta_is_NOT_progress() -> None:
    """With `>` alone, drift inside the estimator's own noise resets patience forever."""
    stop = PatienceStop(patience=2, min_delta=0.01)
    assert not stop.observe(1.0, step=1)
    assert not stop.observe(0.999, step=2)        # improvement, but below min_delta
    assert stop.observe(0.998, step=3), "sub-min_delta drift must not count as progress"
    assert stop.best == 1.0, "best must not move on a sub-threshold improvement"


def test_a_real_improvement_DOES_reset_patience() -> None:
    """The positive control. Without it both rows above pass on a stop that always fires."""
    stop = PatienceStop(patience=2, min_delta=0.01)
    assert not stop.observe(1.0, step=1)
    assert not stop.observe(0.90, step=2)
    assert not stop.observe(0.80, step=3)
    assert not stop.fired and stop.since_best == 0 and stop.best_at_step == 3


def test_the_counters_report_enough_to_tell_the_two_endings_apart() -> None:
    """A run that hit its ceiling and one that stopped early must be distinguishable without
    arithmetic on the logs."""
    stop = PatienceStop(patience=1, min_delta=0.0)
    stop.observe(1.0, step=10)
    stop.observe(2.0, step=20)
    c = stop.counters()
    assert c["heldout_stop_fired"] is True
    assert c["heldout_best_policy_loss"] == 1.0 and c["heldout_best_at_step"] == 10
    assert c["heldout_evaluations"] == 2 and c["heldout_evals_since_best"] == 1
    assert c["heldout_patience"] == 1 and c["heldout_min_delta"] == 0.0


def test_a_monitor_that_never_observed_reports_no_best_rather_than_a_sentinel() -> None:
    """`inf` in a log reads as a measurement. `None` says no reading was taken."""
    assert PatienceStop(patience=1, min_delta=0.0).counters()["heldout_best_policy_loss"] is None


def _monitor(**kw):
    base = dict(ring=object(), spec=object(), plies=1000, batch_size=100, eval_every=10,
                patience=2, min_delta=0.01, caps_provider=lambda: None,
                sample_threads_provider=lambda: 1,
                fast_policy_weight_provider=lambda: 0.0)
    base.update(kw)
    return HeldOutMonitor.build(**base)


def test_an_EMPTY_heldout_ring_is_REFUSED() -> None:
    """A held-out loss over nothing is a number with no producer, not a small number."""
    with pytest.raises(HeldOutError, match="zero plies"):
        _monitor(plies=0)


def test_a_cadence_that_NEVER_FIRES_is_REFUSED() -> None:
    """`eval_every <= 0` silently turns the budget into the only bound."""
    with pytest.raises(HeldOutError, match="never fires"):
        _monitor(eval_every=0)


def test_the_pass_length_is_DERIVED_from_the_ring_and_not_chosen() -> None:
    """`ceil(plies / batch_size)` — one ring-equivalent of samples whatever the corpus.

    A fixed batch count would be a different amount of evidence on every corpus, and the
    estimator's noise would then move with the data rather than with the design."""
    assert _monitor(plies=1000, batch_size=100).eval_batches == 10
    assert _monitor(plies=1001, batch_size=100).eval_batches == 11   # ceil, not floor
    assert _monitor(plies=1, batch_size=100).eval_batches == 1       # never zero


def test_the_cadence_gates_evaluation(monkeypatch) -> None:
    """Off-cadence steps must not evaluate: an evaluation per step would make the held-out
    pass the dominant cost and the 'forward-only' claim meaningless."""
    m = _monitor(eval_every=5)
    calls = []
    monkeypatch.setattr(m, "evaluate", lambda trainer: (calls.append(1), 1.0)[1])
    for step in range(1, 11):
        m.maybe_evaluate(object(), step=step)
    assert len(calls) == 2, f"expected evaluations at steps 5 and 10, got {len(calls)}"
    assert [s for s, _ in m.history] == [5, 10]


def test_measure_noise_reports_the_SPREAD_of_repeated_readings(monkeypatch) -> None:
    """The number `min_delta` has to clear. Measured on an unchanged model, so any difference
    is the sampler's."""
    m = _monitor()
    readings = iter([0.50, 0.53, 0.51])
    monkeypatch.setattr(m, "evaluate", lambda trainer: next(readings))
    assert m.measure_noise(object(), repeats=3) == pytest.approx(0.03)


def test_pb8_the_train_ring_hazard_is_DEFENDED_where_the_ring_is_CHOSEN() -> None:
    """A POINTER to where the train-ring hazard is actually defended.

    A monitor handed the training ring reports a loss that falls forever and every row above still
    passes, because they are properties of the monitor and not of WHICH ring it holds. The defence
    lives where the ring is CHOSEN: the encoder's partition, asserted disjoint and exhaustive in
    `tests/data/test_bootstrap_split_and_truncation.py`, and the CLI's `split_part` guard, driven
    by `tests/train/test_bc_pretrain_cli_stopflags.py`.
    """
    m = _monitor()
    assert m.ring is not None
    import inspect

    from mantis.train.pretrain import cli
    assert "split_part" in inspect.getsource(cli.pretrain), (
        "the CLI's held-out ring guard has gone; PB-8 is undefended and this pointer is a lie"
    )
