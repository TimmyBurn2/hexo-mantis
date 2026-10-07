"""A quick row's policy target never reaches the train step, so an empty value-only row trains exactly what a full one would."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

import _microbatch_harness as H
from mantis._engine import HexgBuffer
from mantis.config.resolve.microbatch import MicrobatchCapsSpec
from mantis.train.coordinator.dispatch import run_declared_train_step

_STONES = [(0, 0, 1), (1, 0, -1), (0, 1, 1)]
_ANCHOR = [(2, 0, 0.6), (1, 1, 0.4)]
_KEYS = ("loss", "policy_loss", "value_loss", "policy_target_entropy", "aux_soft_policy_loss",
         "aux_soft_policy_kl_hard_vs_soft")


def _ring(probed_target: list[tuple[int, int, float]], *, probed_full: bool) -> HexgBuffer:
    """Rows 0-3 full rows on one target; rows 4-7 the probed rows, a decided root's +1 root value on each."""
    buf = HexgBuffer(64, H.GRAPH_ENCODING, 128)
    for i in range(8):
        probed = i >= 4
        buf.push_graph_position(_STONES, probed_target if probed else _ANCHOR, 1, 2, 2 + i,
                                probed_full if probed else True, 1.0 if i % 2 == 0 else -1.0, True, 10 + i,
                                tail_mass=0.0, root_value=1.0 if probed else 0.0, root_value_valid=probed)
    return buf


def _step(tmp_path: Path, buf: HexgBuffer) -> tuple[Any, dict[str, Any], torch.Tensor]:
    """One seeded soft-policy train step: the sampled targets, the `trainer_step` event, the step's gradient."""
    buf.seed_sampler(H.SEED)
    replay = H.ReplayWireBuffer(buf, 8)
    sink = H.SpySink()
    trainer = H.soft_policy_graph_trainer(tmp_path, sink=sink)
    with H.deterministic_algorithms():
        run_declared_train_step(
            trainer, replay, H.GSPEC, batch_size=8, augment=False,
            caps_provider=lambda: MicrobatchCapsSpec(*H.non_binding_caps(replay.wire)),
            sample_threads_provider=lambda: 1)
    (event,) = sink.named("trainer_step")
    return replay.targets, event, H.grad_vector(trainer.model)


def test_a_quick_rows_empty_target_trains_exactly_what_a_full_distribution_would(tmp_path: Path) -> None:
    """PLANTED BREAK: weigh the quick arm's policy rows (`graph_policy_row_weights` all ones) and every key moves."""
    empty_targets, empty, empty_grad = _step(tmp_path / "a", _ring([], probed_full=False))
    _full_targets, filled, filled_grad = _step(tmp_path / "b", _ring([(1, 1, 1.0)], probed_full=False))
    quick = np.asarray(empty_targets.is_full_search) == 0
    assert quick.any() and not quick.all(), "the seeded draw holds both a value-only row and a full row"
    assert int(np.asarray(empty_targets.explicit_mask).sum()) == 2 * int((~quick).sum()), "only full rows store entries"
    for key in _KEYS:
        assert empty[key] == filled[key], f"{key}: {empty[key]} against {filled[key]}"
    assert torch.equal(empty_grad, filled_grad)


def test_the_same_swap_on_a_full_row_moves_the_step(tmp_path: Path) -> None:
    """The control: the comparison above sees a target change wherever the policy is trained."""
    _t, anchor, anchor_grad = _step(tmp_path / "a", _ring(_ANCHOR, probed_full=True))
    _t, moved, moved_grad = _step(tmp_path / "b", _ring([(1, 1, 1.0)], probed_full=True))
    assert anchor["policy_loss"] != moved["policy_loss"]
    assert anchor["aux_soft_policy_loss"] != moved["aux_soft_policy_loss"]
    assert not torch.equal(anchor_grad, moved_grad)
