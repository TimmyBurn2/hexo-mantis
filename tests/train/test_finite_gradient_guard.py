"""R345(b)(1) — a non-finite GRADIENT must not reach `optimizer.step`.

THE DISTINCTION FROM `test_nonfinite_guard.py`, WHICH IS THE WHOLE LEG. That suite pins the
guard on a non-finite LOSS: a microbatch whose loss is NaN is skipped before its backward.
This one pins the guard on a non-finite GRADIENT, which is a different quantity reached by a
different route — a perfectly finite loss can backward into a NaN or inf `.grad` (a
`sqrt(0)` derivative, a `0 x inf` in a fused kernel, an fp32 overflow in an accumulation).
At HEAD that path had no guard at all: `clip_and_step` ran `clip_grad_norm_` and
`optimizer.step()` unconditionally, and the `if not math.isfinite(grad_norm)` branch in the
graph tail executed AFTER the optimizer had already written a NaN-scaled update into every
weight. The counter it incremented was a post-mortem.

THE SECOND HALF IS THE CLOCK. When every microbatch is skipped the loop leaves `.grad` at
zero, `clip_grad_norm_` returns a finite `0.0`, and the step took its optimizer step, its
`self.step` increment, its scheduler step and its EMA update on a gradient that does not
exist. With a fresh optimizer the weight delta is numerically zero and the defect is
invisible; with momentum already accumulated it is not, because Adam's update on a zero
gradient is `exp_avg` decayed, not nothing. So `test_an_all_skipped_step_advances_no_clock`
takes a healthy step FIRST — the momentum is the instrument.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import torch

import _microbatch_harness as H  # the shared graph-step harness (rootdir-relative, house convention)


def _graph_step(trainer: Any, buffer: Any) -> dict[str, float]:
    """One real graph training step through the PRODUCTION dispatch (R155)."""
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
    """Make ONE parameter's gradient non-finite while every loss stays finite.

    A backward hook on the parameter, which is the real mechanism and not a stub of one: the
    forward, the loss and the whole autograd graph are untouched and finite, and the NaN
    enters exactly where a `sqrt(0)` derivative or a fused-kernel `0 x inf` would put it —
    in `.grad`, after backward, before the optimizer reads it.
    """
    param = next(p for p in model.parameters() if p.requires_grad)
    return param.register_hook(lambda g: g * float("nan"))


def _optimizer_clock(optimizer: Any) -> list[float]:
    """Every `step` counter Adam keeps in its own state — the clock the guard must not move."""
    return [float(s["step"]) for s in optimizer.state.values() if "step" in s]


# ── the gradient guard ──────────────────────────────────────────────────────────────────
def test_a_nonfinite_gradient_from_a_finite_loss_never_reaches_the_optimizer(
    tmp_path: Path,
) -> None:
    """The leg's subject: finite loss, non-finite `.grad`, weights must not move."""
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
    """LAW-18: a lever under test logs its own fire-rate in-run, under its own name."""
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
    """Every microbatch non-finite ⇒ no step, no scheduler tick, no optimizer-state move.

    The healthy step first is the instrument, not scene-setting: on a FRESH Adam a zero
    gradient produces a numerically zero update, so the defect hides. With momentum already
    accumulated the decayed `exp_avg` moves the weights, and the all-skipped step becomes
    visible as the weight change it should never have made.
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


# ── mutation half: none of the above may be satisfied by refusing to train ──────────────
def test_a_healthy_step_steps_advances_the_clock_and_counts_nothing(tmp_path: Path) -> None:
    """Without this, `always skip` passes every assertion above.

    Mechanism: the guard fires only on a non-finite pre-clip gradient norm and only on a
    zero-contribution microbatch set, so a healthy step must move the weights, advance
    `self.step`, advance Adam's own state, and leave `skipped_steps` at 0.
    """
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
    """OF2-9's single tail survives the guard: the skip rides the EVENT, not the return.

    `train/coordinator/step.py` reads `loss_info.get("grad_norm", 0.0)` and feeds it to
    `grad_norm_hard_abort`, whose comparison is an R56 SOURCE PIN. A skipped step must
    therefore still return all five keys, with `grad_norm` non-finite — which the pinned
    `math.isfinite(step_gn) and ...` already handles by resetting its consecutive counter,
    exactly as it does today for a NaN. The pin is not touched by this leg.
    """
    trainer = H.tiny_graph_trainer(tmp_path, sink=H.SpySink())
    handle = _poison_one_gradient(trainer.model)
    try:
        result = _graph_step(trainer, H.uniform_graph_buffer())
    finally:
        handle.remove()
    assert set(result) == {"loss", "policy_loss", "value_loss", "grad_norm", "lr"}
