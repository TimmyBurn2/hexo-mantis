"""R328(d) — the BC stopping mechanism: forward-only held-out loss, and a patience stop.

WHY THIS SUITE EXISTS AT ALL. The stopping rule is the ONE stated risk bound on bootstrap
posture (A): the filed adjudication's §3(A) says the value head's over-fit risk *"is bounded by
a knob the operator already controls — how long you pretrain"*. MINT-CLOSE Leg 2 then measured
that the knob did not exist: `run_graph_pretrain` evaluated nothing and `run_declared_train_step`
IS an optimizer step. R328(d) rules it BUILT and **suite-proven with planted breaks before any
pretrain consumes it** — which is this file, written before the first BC run.

THE ROWS ARE THE BREAKS. Each names the mutation it exists to catch, because a stopping rule has
two symmetric failure modes that both look healthy from outside: one that can never fire (the run
silently becomes budget-bound) and one that fires on noise (the run stops before it learns).
"""
from __future__ import annotations

import math

import pytest

from mantis.train.pretrain.heldout import HeldOutError, HeldOutMonitor, PatienceStop


# ═══ PB-6 / PB-7 — the patience rule's two symmetric deaths ══════════════════════════════
def test_pb6_a_worse_reading_does_not_reset_patience() -> None:
    """PB-6. If a WORSE loss reset the counter, the stop could never fire."""
    stop = PatienceStop(patience=2, min_delta=0.0)
    assert not stop.observe(1.0, step=1)
    assert not stop.observe(2.0, step=2)          # worse — must COUNT, not reset
    assert stop.observe(3.0, step=3), "two non-improving readings must fire patience=2"
    assert stop.fired and stop.best == 1.0 and stop.best_at_step == 1


def test_pb7_an_improvement_smaller_than_min_delta_is_NOT_progress() -> None:
    """PB-7, the mirror of PB-6: with `>` alone, drift inside the estimator's own noise
    resets patience forever and the stop can never fire either."""
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
    """LAW-18. A run that hit its ceiling and a run that stopped early must be
    distinguishable without arithmetic on the logs."""
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


# ═══ the monitor's construction refusals ═════════════════════════════════════════════════
def _monitor(**kw):
    base = dict(ring=object(), spec=object(), plies=1000, batch_size=100, eval_every=10,
                patience=2, min_delta=0.01, caps_provider=lambda: None,
                sample_threads_provider=lambda: 1)
    base.update(kw)
    return HeldOutMonitor.build(**base)


def test_an_EMPTY_heldout_ring_is_REFUSED() -> None:
    """A held-out loss over nothing is a number with no producer, not a small number."""
    with pytest.raises(HeldOutError, match="zero plies"):
        _monitor(plies=0)


def test_a_cadence_that_NEVER_FIRES_is_REFUSED() -> None:
    """`eval_every <= 0` silently turns the budget into the only bound — the failure this
    whole mechanism was built because of."""
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


# ═══ PB-8 — the break that would leave every other row green ═════════════════════════════
def test_pb8_the_train_ring_hazard_is_DEFENDED_where_the_ring_is_CHOSEN() -> None:
    """PB-8, and it is a POINTER now rather than the disclosure it was when written.

    A monitor handed the training ring reports a loss that falls forever, and every row above
    still passes: patience arithmetic, cadence, derivation and refusals are all properties of
    the monitor, not of WHICH ring it holds. So the defence cannot live here — it lives where
    the ring is CHOSEN, and it now exists in two places:

      * the ENCODER's partition, asserted disjoint and exhaustive by game-hash set in
        `tests/data/test_bootstrap_split_and_truncation.py`; and
      * the CLI's `split_part` guard, which reads the provenance sidecar and refuses a ring
        that does not declare itself held-out — driven by
        `tests/train/test_bc_pretrain_cli_stopflags.py`.

    This row stays as the pointer between them, because a reader who finds the stopping rule
    first will otherwise assume this file covers the hazard it names.
    """
    m = _monitor()
    assert m.ring is not None
    import inspect

    from mantis.train.pretrain import cli
    assert "split_part" in inspect.getsource(cli.pretrain), (
        "the CLI's held-out ring guard has gone; PB-8 is undefended and this pointer is a lie"
    )
