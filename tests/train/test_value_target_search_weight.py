"""`train.value_target_search_weight`: the value target is w·v_search + (1−w)·z on a row with a search value, z on the rest."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import _microbatch_harness as H
import numpy as np
import pytest
import torch

from mantis._engine import HexgBuffer
from mantis.config.resolve.microbatch import MicrobatchCapsSpec
from mantis.train import losses
from mantis.train.coordinator.dispatch import run_declared_eval_step, run_declared_train_step

_STONES = [(0, 0, 1), (1, 0, -1), (0, 1, 1)]
_VISITS = [(2, 0, 0.6), (1, 1, 0.4)]
_CAPS = MicrobatchCapsSpec(max_edges=100_000_000, max_nodes=4_000_000)
# (z, root value or None, value_valid). Row 4's game has no winner, so its value is masked whatever its root value.
# Row 5's root value equals its z, so the mix reaches it without moving it.
_ROWS = [(1.0, 0.5, True), (-1.0, None, True), (1.0, None, True), (-1.0, -0.25, True), (1.0, -0.5, False),
         (-1.0, -1.0, True)]
# The same rows' targets at weight 0.25 (asymmetric, so a swapped 1 − w shows), each exact in float32.
_WEIGHT = 0.25
_MIXED = [0.875, -1.0, 1.0, -0.8125, 0.625, -1.0]


def _ring(*, roots: bool = True, outcomes: list[float] | None = None) -> HexgBuffer:
    """The six rows, one game each, sampler seeded: two rings built alike draw the same rows in the same order."""
    buf = HexgBuffer(64, H.GRAPH_ENCODING, 128)
    for i, (z, v, valid) in enumerate(_ROWS):
        kw: dict[str, Any] = {"root_value": v, "root_value_valid": True} if roots and v is not None else {}
        z = z if outcomes is None else outcomes[i]
        buf.push_graph_position(_STONES[: 2 + i % 2], _VISITS, 1, 30, 2 + i, True, z, valid, 10 + i, game_id=i, **kw)
    buf.seed_sampler(H.SEED)
    return buf


def _value_loss(tmp_path: Path, ring: HexgBuffer, weight: float, *, train: bool) -> float:
    trainer = H.tiny_graph_trainer(tmp_path, value_target_search_weight=weight, checkpoint_interval=0)
    step: Callable[..., dict[str, float]] = run_declared_train_step if train else run_declared_eval_step
    kw: dict[str, Any] = {"augment": False} if train else {}
    out = step(trainer, ring, H.GSPEC, batch_size=len(_ROWS), caps_provider=lambda: _CAPS,
               sample_threads_provider=lambda: 1, **kw)
    return float(out["value_loss"])


def _assert_the_mix_trains_its_own_target(tmp_path: Path) -> None:
    """At weight > 0 the train step's value loss IS the weight 0 loss of the same rows relabelled with their mixed targets."""
    mixed = _value_loss(tmp_path, _ring(), _WEIGHT, train=True)
    relabelled = _value_loss(tmp_path, _ring(roots=False, outcomes=_MIXED), 0.0, train=True)
    assert mixed == relabelled, f"the weight {_WEIGHT} loss {mixed} is not its target's {relabelled}"
    assert mixed != _value_loss(tmp_path, _ring(), 0.0, train=True), "weight > 0 moved nothing"


def test_at_weight_zero_the_value_target_is_the_outcome_tensor_itself() -> None:
    z = torch.tensor([1.0, -1.0, 0.0])
    assert losses.value_target(z, torch.tensor([0.5, 0.0, -0.25]), torch.tensor([1, 0, 1], dtype=torch.uint8), 0.0) is z


def test_the_mix_reaches_only_rows_with_a_root_value() -> None:
    z = torch.tensor([1.0, -1.0, 1.0, -1.0])
    v = torch.tensor([0.5, 0.0, 0.0, 1.0])
    ok = torch.tensor([1, 0, 0, 1], dtype=torch.uint8)
    assert losses.value_target(z, v, ok, 0.5).tolist() == [0.75, -1.0, 1.0, 0.0]
    assert losses.value_target(z, v, ok, 1.0).tolist() == [0.5, -1.0, 1.0, 1.0]


def test_at_weight_zero_the_trainer_is_byte_equal_whatever_the_rows_root_values(tmp_path: Path) -> None:
    """The v2 trainer at weight 0: rows with and without root values train to the same bits, losses and weights alike."""
    def run(ring: HexgBuffer) -> tuple[list[dict[str, str]], bytes]:
        trainer = H.tiny_graph_trainer(tmp_path, value_target_search_weight=0.0, checkpoint_interval=0)
        replay = H.ReplayWireBuffer(ring, len(_ROWS))
        steps = []
        for _ in range(3):
            out = run_declared_train_step(trainer, replay, H.GSPEC, batch_size=len(_ROWS), augment=False,
                                          caps_provider=lambda: _CAPS, sample_threads_provider=lambda: 1)
            steps.append({k: float(out[k]).hex() for k in ("loss", "policy_loss", "value_loss", "grad_norm")})
        return steps, H.param_vector(trainer.model).numpy().tobytes()

    with_roots, without = run(_ring()), run(_ring(roots=False))
    assert with_roots[0] == without[0]
    assert with_roots[1] == without[1], "the weights moved differently at weight 0"


def test_a_weight_above_zero_mixes_only_rows_with_a_root_value_and_the_planted_break_reds(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """PLANTED BREAK: a mix that ignores the flag trains a row with no root value toward (1−w)·z; the checker reds."""
    _assert_the_mix_trains_its_own_target(tmp_path)
    from mantis.train.trainer import core

    monkeypatch.setattr(core, "value_target", lambda z, v, ok, w: w * v + (1.0 - w) * z)
    with pytest.raises(AssertionError, match="is not its target's"):
        _assert_the_mix_trains_its_own_target(tmp_path)


def test_the_eval_step_reads_z_whatever_the_weight(tmp_path: Path) -> None:
    """A reading must not reward the lever: the forward-only value loss at weight > 0 is the weight 0 one, bit for bit."""
    assert _value_loss(tmp_path, _ring(), _WEIGHT, train=False) == _value_loss(tmp_path, _ring(), 0.0, train=False)


def test_the_step_event_counts_the_value_rows_the_mix_reaches_and_moves(tmp_path: Path) -> None:
    """The mix's fire-rate: value-supervised rows with a root value (reached), those with v != z (moved), the weight echoed."""
    sink = H.SpySink()
    trainer = H.tiny_graph_trainer(tmp_path, sink=sink, value_target_search_weight=_WEIGHT, checkpoint_interval=0)
    replay = H.ReplayWireBuffer(_ring(), len(_ROWS))
    run_declared_train_step(trainer, replay, H.GSPEC, batch_size=len(_ROWS), augment=False,
                            caps_provider=lambda: _CAPS, sample_threads_provider=lambda: 1)
    t = replay.targets
    rooted = np.asarray(t.root_value_valid) != 0
    assert int(rooted.sum()) == 4 and sorted(np.asarray(t.outcomes).tolist()) == sorted(z for z, _v, _ok in _ROWS)
    event = sink.named("trainer_step")[0]
    assert (event["root_value_rows"], event["root_value_rows_moved"]) == (3, 2)
    assert event["value_target_search_weight"] == _WEIGHT
