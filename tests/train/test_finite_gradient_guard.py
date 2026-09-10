"""A non-finite GRADIENT must not reach `optimizer.step`.

Distinct from `test_nonfinite_guard.py`, which pins the guard on a non-finite LOSS: a finite
loss can backward into a NaN or inf `.grad` (a `sqrt(0)` derivative, a `0 x inf` in a fused
kernel, an fp32 overflow in an accumulation), and that path once had no guard at all.

When every microbatch is skipped, `.grad` stays at zero, `clip_grad_norm_` returns a finite
`0.0` and the step takes its optimizer, scheduler and EMA moves on a gradient that does not
exist. A fresh optimizer hides that (the delta is numerically zero); accumulated momentum
does not, because Adam's update on a zero gradient is a decayed `exp_avg`. So the
all-skipped row takes a healthy step first — the momentum is the instrument.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import torch

import _microbatch_harness as H  # the shared graph-step harness


def _graph_step(trainer: Any, buffer: Any) -> dict[str, float]:
    """Take one real graph training step through the production dispatch."""
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


def _poison_one_gradient(model: torch.nn.Module) -> Any:
    """Make one parameter's gradient non-finite while every loss stays finite.

    A backward hook puts the NaN where a `sqrt(0)` derivative would: in `.grad`, after
    backward, before the optimizer reads it. The forward and autograd graph stay finite.
    """
    param = next(p for p in model.parameters() if p.requires_grad)
    return param.register_hook(lambda g: g * float("nan"))


def _optimizer_clock(optimizer: Any) -> list[float]:
    """Return every `step` counter Adam keeps in its own state: the clock the guard must not move."""
    return [float(s["step"]) for s in optimizer.state.values() if "step" in s]


def test_a_nonfinite_gradient_from_a_finite_loss_never_reaches_the_optimizer(
    tmp_path: Path,
) -> None:
    """Prove a finite loss with a non-finite `.grad` moves no weights and no step counter."""
    trainer = H.tiny_graph_trainer(tmp_path, sink=H.SpySink())
    buffer = H.uniform_graph_buffer()
    handle = _poison_one_gradient(trainer.model)
    try:
        before = H.param_vector(trainer.model).clone()
        step_before = trainer.step

        result = _graph_step(trainer, buffer)

        after = H.param_vector(trainer.model)
        assert torch.isfinite(after).all(), (
            "non-finite weights: the NaN gradient reached `optimizer.step` and was written "
            "into the parameters. The loss was finite the whole way — this is the half the "
            "loss guard cannot see."
        )
        assert torch.equal(before, after), (
            "weights moved on a step whose gradient was non-finite"
        )
        assert trainer.step == step_before, (
            "`self.step` advanced on a step that took no optimizer step — the run would "
            "report a step it never took (LAW-14) and the checkpoint cadence would fire on it"
        )
        assert not math.isfinite(result["grad_norm"]), (
            "`loss_info` reported a finite grad_norm for a non-finite gradient"
        )
    finally:
        handle.remove()


def test_the_skipped_step_is_named_counted_and_on_the_event_stream(tmp_path: Path) -> None:
    """Prove the skipped step is counted and named on the in-run event stream."""
    sink = H.SpySink()
    trainer = H.tiny_graph_trainer(tmp_path, sink=sink)
    handle = _poison_one_gradient(trainer.model)
    try:
        _graph_step(trainer, H.uniform_graph_buffer())
    finally:
        handle.remove()

    assert trainer.skipped_steps == 1, (
        f"the skipped step was not counted (skipped_steps={trainer.skipped_steps!r}) — a run "
        "dropping every step looks identical to a healthy one on loss alone"
    )
    skipped = sink.named("trainer_step_skipped")
    assert skipped, (
        "no `trainer_step_skipped` event: the counter exists only in memory, so nothing "
        "in-run can read it (LAW-18)"
    )
    assert skipped[-1]["reason"] == "nonfinite_gradient"
    assert skipped[-1]["skipped_steps"] == 1


def test_an_all_skipped_microbatch_set_advances_no_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove an all-skipped microbatch set takes no step, scheduler tick or optimizer-state move.

    The healthy step first is the instrument: on a fresh Adam a zero gradient produces a
    numerically zero update, so only accumulated momentum makes the defect visible.
    """
    import mantis.train.trainer.core as core

    trainer = H.tiny_graph_trainer(tmp_path, sink=H.SpySink())
    buffer = H.uniform_graph_buffer()

    _graph_step(trainer, buffer)  # build the momentum
    before = H.param_vector(trainer.model).clone()
    step_before = trainer.step
    lr_before = trainer.optimizer.param_groups[0]["lr"]
    clock_before = _optimizer_clock(trainer.optimizer)
    assert clock_before, "Adam kept no per-parameter step state — the clock has no instrument"

    real = core.ragged_policy_ce
    monkeypatch.setattr(core, "ragged_policy_ce", lambda *a, **k: real(*a, **k) * float("nan"))

    result = _graph_step(trainer, buffer)

    assert trainer.step == step_before, "`self.step` advanced with no contributing microbatch"
    assert _optimizer_clock(trainer.optimizer) == clock_before, (
        "the optimizer's own step state advanced on a gradient that does not exist — Adam's "
        "bias correction now believes in a step nothing fed"
    )
    assert trainer.optimizer.param_groups[0]["lr"] == lr_before, (
        "the scheduler advanced on an all-skipped step, so the LR curve is ahead of the "
        "gradient count that is supposed to drive it"
    )
    assert torch.equal(before, H.param_vector(trainer.model)), (
        "weights moved on a step where every microbatch was skipped — the decayed momentum "
        "was applied as though a gradient had arrived"
    )
    assert not math.isfinite(result["grad_norm"]), (
        "an all-skipped step reported a finite grad_norm (a 0.0 over zeroed grads), which "
        "reads to every downstream gate as a healthy, exceptionally stable step"
    )


def test_a_healthy_step_steps_advances_the_clock_and_counts_nothing(tmp_path: Path) -> None:
    """Prove a healthy step still steps, so an unconditional skip cannot satisfy the rows above."""
    sink = H.SpySink()
    trainer = H.tiny_graph_trainer(tmp_path, sink=sink)
    buffer = H.uniform_graph_buffer()
    before = H.param_vector(trainer.model).clone()
    step_before = trainer.step

    result = _graph_step(trainer, buffer)

    assert trainer.step == step_before + 1, "a healthy step did not advance `self.step`"
    assert _optimizer_clock(trainer.optimizer), "the optimizer never stepped"
    assert trainer.skipped_steps == 0, "counted a skip that never happened"
    assert math.isfinite(result["grad_norm"])
    assert not torch.equal(before, H.param_vector(trainer.model)), (
        "a healthy step did not update the weights — the guard is skipping unconditionally"
    )
    assert not sink.named("trainer_step_skipped"), (
        "a healthy step emitted a skip event"
    )


def test_the_loss_info_contract_stays_five_keys_on_a_skipped_step(tmp_path: Path) -> None:
    """Prove a skipped step still returns all five `loss_info` keys: the skip rides the event.

    The hard-abort consumer reads `grad_norm` off this return, so dropping a key would
    change what the abort compares.
    """
    trainer = H.tiny_graph_trainer(tmp_path, sink=H.SpySink())
    handle = _poison_one_gradient(trainer.model)
    try:
        result = _graph_step(trainer, H.uniform_graph_buffer())
    finally:
        handle.remove()
    assert set(result) == {"loss", "policy_loss", "value_loss", "grad_norm", "lr"}
