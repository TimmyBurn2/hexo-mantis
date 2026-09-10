"""A NaN must not be able to destroy the model in silence.

The cascade, measured: a non-finite microbatch loss backwards into a non-finite clip
coefficient, and `clip_and_step` writes NaN into EVERY weight — one bad step and the model is
gone, while the run keeps reporting numbers. The guard existed on the pretrain path and not on
the graph path that actually trains, and every reporting layer filtered the condition out: the
spike check carried a `gn == gn` filter, so the alert was quietest exactly when the weights had
just been corrupted; the alert emitter dropped a non-finite `loss_total` and reported it
nowhere; and the hard-abort gate treated unbounded as "unknown, so assume fine". These pins
drive the REAL trainer step and the REAL rules, each with its mutation half.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import torch

from mantis.monitor.config import MonitorConfig
from mantis.monitor.rules import check_grad_norm_spike, check_nonfinite_loss

import _microbatch_harness as H  # the shared graph-step harness (rootdir-relative)


def _graph_step(trainer: Any, buffer: Any) -> dict[str, float]:
    """One real graph training step through the PRODUCTION dispatch: `dispatch._graph_step` is
    what the coordinator calls, so driving it keeps the guard on the path that actually runs."""
    from mantis.config.resolve.microbatch import MicrobatchCapsSpec
    from mantis.train.coordinator.dispatch import _graph_step as production_graph_step

    wire, _targets = buffer.sample_graph_batch(4, augment=False, recent_frac=0.0)
    max_edges, max_nodes = H.non_binding_caps(wire)
    return production_graph_step(
        trainer, buffer, H.GSPEC,
        batch_size=4, augment=False, recency_weight=0.0, recent_buffer=None,
        caps_provider=lambda: MicrobatchCapsSpec(max_edges=max_edges, max_nodes=max_nodes),
        sample_threads_provider=lambda: 1,
                            fast_policy_weight_provider=lambda: 0.0,
    )


def test_a_nonfinite_microbatch_loss_is_skipped_and_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real step, with the real guard, against a genuinely non-finite loss, injected at
    `ragged_policy_ce` — the term that produced it in the incident — rather than a stubbed loss."""
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
    """Mutation half — without it, "always skip" would pass the test above: a finite loss must
    leave the counter at 0 AND still take its optimizer step."""
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
    """The counters ride the EVENT, not the return: `loss_info` is a contract the coordinator's
    gates and checkpoint metadata pin."""
    trainer = H.tiny_graph_trainer(tmp_path, sink=H.SpySink())
    result = _graph_step(trainer, H.uniform_graph_buffer())
    assert set(result) == {"loss", "policy_loss", "value_loss", "grad_norm", "lr"}


def test_the_counters_reach_the_event_stream(tmp_path: Path) -> None:
    """LAW-18: a counter nothing can read in-run is not an instrument."""
    sink = H.SpySink()
    trainer = H.tiny_graph_trainer(tmp_path, sink=sink)
    _graph_step(trainer, H.uniform_graph_buffer())
    steps = [e for e in sink.events if e.get("event") == "trainer_step"]
    assert steps, "no trainer_step event emitted"
    assert "nonfinite_loss_microbatches" in steps[-1]
    assert "nonfinite_grad_steps" in steps[-1]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_nonfinite_grad_norm_fires_the_instability_alert(value: float) -> None:
    """Reversed: this used to be pinned as "must never trip"."""
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


def test_the_hard_abort_gap_is_documented_and_pin_bound() -> None:
    """KNOWN GAP, deliberately left open and pinned so it cannot close by accident.

    Making a non-finite grad norm trip `grad_norm_hard_abort` was implemented and REVERTED: the
    manifest SOURCE-PINS this exact comparison, so a change to the gate's decision must force a
    re-adjudication rather than a quiet edit, and the row is DEFERRED and knowingly DISARMED, so
    firing regardless of the threshold would partially arm it — an operator-only change. A NaN
    is still caught by the trainer guard and the alert rules; only the backstop stays gated.
    """
    src = (Path(__file__).resolve().parents[2]
           / "src" / "mantis" / "train" / "coordinator" / "step.py").read_text(encoding="utf-8")
    assert "if math.isfinite(step_gn) and step_gn > cfg.hard_gn_threshold:" in src, (
        "the grad-norm abort's comparison changed. That comparison is an R56 SOURCE PIN in "
        "`mantis/config/armed_aborts.py` — re-adjudicate the manifest row, do not edit the "
        "pin, and do not change this test to match (ADJ-D13)"
    )
    manifest = (Path(__file__).resolve().parents[2]
                / "src" / "mantis" / "config" / "armed_aborts.py").read_text(encoding="utf-8")
    assert "if math.isfinite(step_gn) and step_gn > cfg.hard_gn_threshold:" in manifest, (
        "the manifest no longer pins this comparison — the two sides have drifted apart, "
        "which is the state R56's scan exists to make impossible"
    )
