"""The step's one read after its backward: host-found value rows are the value mask, and each microbatch's scalars land in their own totals."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

import _microbatch_harness as H
import mantis.train.trainer.core as core
from mantis.config.resolve.microbatch import MicrobatchCapsSpec
from mantis.model.dist65 import N_VALUE_BINS, binned_value_loss
from mantis.train.coordinator.dispatch import _build_graph_parts, run_declared_train_step
from mantis.train.events import tail_mass_block


def test_the_value_rows_loss_is_the_value_mask_loss_forward_and_backward() -> None:
    gen = torch.Generator().manual_seed(7)
    logits = torch.randn(37, N_VALUE_BINS, generator=gen, requires_grad=True)
    outcome = torch.empty(37).uniform_(-1.0, 1.0, generator=gen)
    mask = (torch.rand(37, generator=gen) > 0.4).to(torch.uint8)
    rows = torch.from_numpy(np.flatnonzero(mask.numpy()))

    def by_rows(r: torch.Tensor) -> torch.Tensor:
        return binned_value_loss(logits.index_select(0, r), outcome.index_select(0, r), denominator=11.0)

    masked = binned_value_loss(logits, outcome, value_mask=mask, denominator=11.0)
    picked = by_rows(rows)
    assert torch.equal(masked, picked)
    assert torch.equal(torch.autograd.grad(masked, logits)[0], torch.autograd.grad(picked, logits)[0])
    # PLANTED BREAK: one valid row dropped must move the loss.
    assert not torch.equal(masked, by_rows(rows[:-1]))


def test_each_part_carries_its_value_mask_as_host_found_rows(tmp_path) -> None:
    trainer = H.tiny_graph_trainer(tmp_path)
    replay = H.ReplayWireBuffer(H.uniform_graph_buffer(8), 4)
    caps = H.caps_for_exactly(replay.wire, 2)
    built = _build_graph_parts(trainer, replay, H.GSPEC, batch_size=4, augment=False,
                               caps_provider=lambda: MicrobatchCapsSpec(*caps), sample_threads_provider=lambda: 1)
    for make in built["parts"]:
        inputs = make()
        assert torch.equal(inputs.value_rows, torch.from_numpy(np.flatnonzero(inputs.value_valid.numpy())))


def _step_with_known_scalars(tmp_path, monkeypatch, m: int) -> tuple[dict[str, Any], Any]:
    """One aux-head step at `m` microbatches whose loss scalars are distinct exact constants per microbatch."""
    sink = H.SpySink()
    trainer = H.soft_policy_graph_trainer(tmp_path, sink=sink)
    replay = H.ReplayWireBuffer(H.uniform_graph_buffer(8), 4)
    real_ce, real_aux, seen = core.ragged_policy_ce_and_entropies, trainer._aux_soft_policy_terms, []

    def ce(*a: Any, **k: Any):
        loss, ent, model_ent = real_ce(*a, **k)
        i = len(seen)
        seen.append(i)
        return loss * 0 + (1.0 + i), ent * 0 + (10.0 + i), model_ent * 0 + (100.0 + i)

    def aux(*a: Any, **k: Any):
        loss, kl = real_aux(*a, **k)
        i = len(seen) - 1
        return loss * 0 + (1000.0 + i), kl * 0 + (2000.0 + i)

    monkeypatch.setattr(core, "ragged_policy_ce_and_entropies", ce)
    monkeypatch.setattr(trainer, "_aux_soft_policy_terms", aux)
    caps = H.caps_for_exactly(replay.wire, m)
    run_declared_train_step(trainer, replay, H.GSPEC, batch_size=4, augment=False,
                            caps_provider=lambda: MicrobatchCapsSpec(*caps), sample_threads_provider=lambda: 1)
    (event,) = sink.named("trainer_step")
    return event, replay.targets


def _totals_land(event: dict[str, Any], targets: Any, m: int) -> bool:
    return (event["microbatches"] == m
            and event["policy_loss"] == sum(1.0 + i for i in range(m))
            and event["policy_target_entropy"] == sum(10.0 + i for i in range(m))
            and event["policy_entropy"] == sum(100.0 + i for i in range(m))
            and event["aux_soft_policy_loss"] == sum(1000.0 + i for i in range(m))
            and event["aux_soft_policy_kl_hard_vs_soft"] == sum(2000.0 + i for i in range(m))
            and all(np.isfinite(event[k]) for k in event if k.endswith("_grad_norm"))
            and all(event[k] == v for k, v in tail_mass_block(
                np.asarray(targets.tail_mass, dtype=np.float32).tolist()).items()))


@pytest.mark.parametrize("m", [2, 4])
def test_every_microbatch_scalar_lands_in_its_own_total(tmp_path, monkeypatch, m: int) -> None:
    event, targets = _step_with_known_scalars(tmp_path, monkeypatch, m)
    assert _totals_land(event, targets, m), event


def test_a_read_sliced_one_off_is_caught(tmp_path, monkeypatch) -> None:
    """PLANTED BREAK: the step's one read handed back rotated by one value must miss the totals."""
    real = core.clip_and_step

    def rotated(*a: Any, **k: Any):
        grad_norm, rest = real(*a, **k)
        return grad_norm, rest[1:] + rest[:1]

    monkeypatch.setattr(core, "clip_and_step", rotated)
    event, targets = _step_with_known_scalars(tmp_path, monkeypatch, 2)
    assert not _totals_land(event, targets, 2)
