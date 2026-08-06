"""Item 6 pins — a NaN must not be able to destroy the model in silence.

THE CASCADE (falsified row F-11, measured). A non-finite microbatch loss backwards into a
non-finite clip coefficient, and `clip_and_step` then writes NaN into EVERY weight. One bad
step and the model is gone; the run continues and keeps reporting numbers.

WHAT WAS WRONG. The guard existed on the PRETRAIN path (`pretrain/trainer.py`) and not on
the graph path that actually trains run5. Worse, every layer that could have reported the
condition was filtering it out:

  * `check_grad_norm_spike` carried a `gn == gn` filter — "a NaN must never trip the
    instability alert" — so the alert was quietest exactly when the weights had just been
    corrupted, while a merely large finite norm alerted;
  * `emit_training_step_alerts` dropped a non-finite `loss_total` from the loss window
    (correct: a NaN poisons every later comparison) and reported it nowhere (not correct);
  * the `grad_norm_hard_abort` gate read `math.isfinite(gn) and gn > threshold`, EXCLUDING
    NaN from the abort. Unbounded is above any threshold; it was treated as "unknown, so
    assume fine".

So the one condition that destroys a model outright was the one condition nothing reported.
These pins drive the REAL trainer step and the REAL rules — the guard and its three
reporting paths, each with the mutation half that proves it stays quiet on healthy training.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import torch

from mantis.monitor.config import MonitorConfig
from mantis.monitor.rules import check_grad_norm_spike, check_nonfinite_loss

import _microbatch_harness as H  # the shared graph-step harness (rootdir-relative, house convention)


def _graph_step(trainer: Any, buffer: Any) -> dict[str, float]:
    """One real graph training step, through the PRODUCTION dispatch.

    `dispatch._graph_step` is what the coordinator calls: it samples the wire, plans the
    microbatches, collates each part and calls `train_step_from_graph_batch`. Driving it
    (rather than hand-building `parts`) keeps the guard under test on the path that actually
    runs (R155: production path, production parameters).
    """
    from mantis.config.resolve.microbatch import MicrobatchCapsSpec
    from mantis.train.coordinator.dispatch import _graph_step as production_graph_step

    wire, _targets = buffer.sample_graph_batch(4, augment=False, recent_frac=0.0)
    max_edges, max_nodes = H.non_binding_caps(wire)
    return production_graph_step(
        trainer, buffer, H.GSPEC,
        batch_size=4, augment=False, recency_weight=0.0, recent_buffer=None,
        caps_provider=lambda: MicrobatchCapsSpec(max_edges=max_edges, max_nodes=max_nodes),
    )


# ── the guard on the production graph step ─────────────────────────────────────────────


def test_a_nonfinite_microbatch_loss_is_skipped_and_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real step, with the real guard, against a genuinely non-finite loss.

    The NaN is injected at `ragged_policy_ce` — the actual term that produced it in the F-11
    incident (a 0x-inf in the aux CE) — so the guard is exercised where it sits rather than
    through a stubbed `loss`.
    """
    import mantis.train.trainer.core as core

    trainer = H.tiny_graph_trainer(tmp_path, sink=H.SpySink())
    buffer = H.uniform_graph_buffer()
    before = H.param_vector(trainer.model).clone()

    real = core.ragged_policy_ce
    monkeypatch.setattr(
        core, "ragged_policy_ce",
        lambda *a, **k: real(*a, **k) * float("nan"))

    _graph_step(trainer, buffer)

    assert trainer.nonfinite_loss_microbatches > 0, (
        "the non-finite microbatch was NOT counted — the skip is silent, and a run dropping "
        "every microbatch looks identical to a healthy one on loss alone (LAW-18)"
    )
    after = H.param_vector(trainer.model)
    assert torch.isfinite(after).all(), (
        "the model contains non-finite weights: the NaN reached clip_and_step and was "
        "written into the parameters. This is the F-11 cascade the guard exists to stop."
    )
    assert torch.equal(before, after), (
        "weights moved on a step whose every microbatch was non-finite — the skip did not "
        "actually skip the backward"
    )


def test_a_healthy_step_counts_nothing_and_does_move_the_weights(tmp_path: Path) -> None:
    """Mutation half. Without it, `always skip` would pass the test above.

    Mechanism: the guard fires only on `not torch.isfinite(loss)`, so a finite loss must
    leave the counter at 0 AND must still take its optimizer step. A counter that always
    increments, or a guard that always skips, reports nothing and trains nothing.
    """
    trainer = H.tiny_graph_trainer(tmp_path, sink=H.SpySink())
    buffer = H.uniform_graph_buffer()
    before = H.param_vector(trainer.model).clone()

    result = _graph_step(trainer, buffer)

    assert trainer.nonfinite_loss_microbatches == 0, "counted a NaN that never happened"
    assert trainer.nonfinite_grad_steps == 0
    assert math.isfinite(result["loss"])
    assert not torch.equal(before, H.param_vector(trainer.model)), (
        "a healthy step did not update the weights — the guard is skipping unconditionally"
    )


def test_the_loss_info_contract_stays_five_keys(tmp_path: Path) -> None:
    """OF2-9 is not broken by the counters: they ride the EVENT, not the return.

    `loss_info` is read by the coordinator's gates and by checkpoint metadata, so widening
    it would change a contract those readers pin. The counters belong on the event stream.
    """
    trainer = H.tiny_graph_trainer(tmp_path, sink=H.SpySink())
    result = _graph_step(trainer, H.uniform_graph_buffer())
    assert set(result) == {"loss", "policy_loss", "value_loss", "grad_norm", "lr"}


def test_the_counters_reach_the_event_stream(tmp_path: Path) -> None:
    """LAW-18: a counter nothing can read in-run is not an instrument."""
    sink = H.SpySink()
    trainer = H.tiny_graph_trainer(tmp_path, sink=sink)
    _graph_step(trainer, H.uniform_graph_buffer())
    steps = [e for e in sink.events if e.get("event") == "training_step"]
    assert steps, "no training_step event emitted"
    assert "nonfinite_loss_microbatches" in steps[-1]
    assert "nonfinite_grad_steps" in steps[-1]


# ── the three reporting paths that used to filter NaN out ──────────────────────────────


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_nonfinite_grad_norm_fires_the_instability_alert(value: float) -> None:
    """Reversed by item 6: this used to be pinned as `must never trip`."""
    assert check_grad_norm_spike({"grad_norm": value}, MonitorConfig()) is not None


def test_a_finite_grad_norm_below_the_bar_still_does_not_fire() -> None:
    """Mutation half: the rule must not have become `always fire`."""
    assert check_grad_norm_spike({"grad_norm": 0.5}, MonitorConfig()) is None
    assert check_grad_norm_spike({}, MonitorConfig()) is None, (
        "an ABSENT grad_norm is a missing reading, not a bad one — it must stay silent"
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_a_nonfinite_loss_fires_its_rule(value: float) -> None:
    assert check_nonfinite_loss({"loss_total": value}, MonitorConfig()) is not None


def test_a_finite_or_absent_loss_does_not_fire_the_nonfinite_rule() -> None:
    cfg = MonitorConfig()
    assert check_nonfinite_loss({"loss_total": 3.0}, cfg) is None
    assert check_nonfinite_loss({}, cfg) is None
    assert check_nonfinite_loss({"loss_total": True}, cfg) is None, (
        "a bool is not a loss reading; it must not be treated as one"
    )


def test_the_hard_abort_no_longer_excludes_a_nonfinite_grad_norm() -> None:
    """Source census on the gate condition (item 6's third path).

    The gate lives inside a long coordinator method that needs a full StepCoordinator to
    drive; the CONDITION is the whole of the change, so it is asserted structurally. The old
    form `math.isfinite(step_gn) and step_gn > ...` is what excluded NaN from the abort.
    """
    src = (Path(__file__).resolve().parents[2]
           / "src" / "mantis" / "train" / "coordinator" / "step.py").read_text(encoding="utf-8")
    assert "if not math.isfinite(step_gn) or step_gn > cfg.hard_gn_threshold:" in src, (
        "the grad-norm hard abort does not treat a non-finite norm as exceeding the "
        "threshold — a NaN gradient cannot fire the abort"
    )
    assert "if math.isfinite(step_gn) and step_gn > cfg.hard_gn_threshold:" not in src, (
        "the old NaN-excluding condition is back"
    )
