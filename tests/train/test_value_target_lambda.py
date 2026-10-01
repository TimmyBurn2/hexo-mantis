"""`train.value_target_lambda`: the value target is λ·v_search + (1−λ)·z on a row with a root value, z on the rest."""
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
_ROWS = [(1.0, 0.5, True), (-1.0, None, True), (1.0, None, True), (-1.0, -0.25, True), (1.0, -0.5, False),
         (-1.0, 1.0, True)]
# The same rows' targets at λ = 0.5, each exact in float32.
_MIXED_AT_HALF = [0.75, -1.0, 1.0, -0.625, 0.25, 0.0]


def _ring(*, roots: bool = True, outcomes: list[float] | None = None) -> HexgBuffer:
    """The six rows, one game each, sampler seeded: two rings built alike draw the same rows in the same order."""
    buf = HexgBuffer(64, H.GRAPH_ENCODING, 128)
    for i, (z, v, valid) in enumerate(_ROWS):
        kw: dict[str, Any] = {"root_value": v, "root_value_valid": True} if roots and v is not None else {}
        z = z if outcomes is None else outcomes[i]
        buf.push_graph_position(_STONES[: 2 + i % 2], _VISITS, 1, 30, 2 + i, True, z, valid, 10 + i, game_id=i, **kw)
    buf.seed_sampler(H.SEED)
    return buf


def _value_loss(tmp_path: Path, ring: HexgBuffer, lam: float, *, train: bool) -> float:
    trainer = H.tiny_graph_trainer(tmp_path, value_target_lambda=lam, checkpoint_interval=0)
    step: Callable[..., dict[str, float]] = run_declared_train_step if train else run_declared_eval_step
    kw: dict[str, Any] = {"augment": False} if train else {}
    out = step(trainer, ring, H.GSPEC, batch_size=len(_ROWS), caps_provider=lambda: _CAPS,
               sample_threads_provider=lambda: 1, **kw)
    return float(out["value_loss"])


def _assert_the_mix_trains_its_own_target(tmp_path: Path) -> None:
    """At λ = 0.5 the value loss IS the λ = 0 loss of the same rows relabelled with their mixed targets, both paths."""
    for train in (False, True):
        mixed = _value_loss(tmp_path, _ring(), 0.5, train=train)
        relabelled = _value_loss(tmp_path, _ring(roots=False, outcomes=_MIXED_AT_HALF), 0.0, train=train)
        assert mixed == relabelled, f"train={train}: the λ = 0.5 loss {mixed} is not its target's {relabelled}"
        assert mixed != _value_loss(tmp_path, _ring(), 0.0, train=train), "λ > 0 moved nothing"


def test_at_lambda_zero_the_value_target_is_the_outcome_tensor_itself() -> None:
    z = torch.tensor([1.0, -1.0, 0.0])
    assert losses.value_target(z, torch.tensor([0.5, 0.0, -0.25]), torch.tensor([1, 0, 1], dtype=torch.uint8), 0.0) is z


def test_the_mix_reaches_only_rows_with_a_root_value() -> None:
    z = torch.tensor([1.0, -1.0, 1.0, -1.0])
    v = torch.tensor([0.5, 0.0, 0.0, 1.0])
    ok = torch.tensor([1, 0, 0, 1], dtype=torch.uint8)
    assert losses.value_target(z, v, ok, 0.5).tolist() == [0.75, -1.0, 1.0, 0.0]
    assert losses.value_target(z, v, ok, 1.0).tolist() == [0.5, -1.0, 1.0, 1.0]


def test_at_lambda_zero_the_trainer_is_byte_equal_whatever_the_rows_root_values(tmp_path: Path) -> None:
    """The v2 trainer at λ = 0: rows with and without root values train to the same bits, losses and weights alike."""
    def run(ring: HexgBuffer) -> tuple[list[dict[str, str]], bytes]:
        trainer = H.tiny_graph_trainer(tmp_path, value_target_lambda=0.0, checkpoint_interval=0)
        replay = H.ReplayWireBuffer(ring, len(_ROWS))
        steps = []
        for _ in range(3):
            out = run_declared_train_step(trainer, replay, H.GSPEC, batch_size=len(_ROWS), augment=False,
                                          caps_provider=lambda: _CAPS, sample_threads_provider=lambda: 1)
            steps.append({k: float(out[k]).hex() for k in ("loss", "policy_loss", "value_loss", "grad_norm")})
        return steps, H.param_vector(trainer.model).numpy().tobytes()

    with_roots, without = run(_ring()), run(_ring(roots=False))
    assert with_roots[0] == without[0]
    assert with_roots[1] == without[1], "the weights moved differently at λ = 0"


def test_a_lambda_above_zero_mixes_only_rows_with_a_root_value_and_the_planted_break_reds(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """PLANTED BREAK: a mix that ignores the flag trains a row with no root value toward (1−λ)·z; the checker reds."""
    _assert_the_mix_trains_its_own_target(tmp_path)
    from mantis.train.trainer import core

    monkeypatch.setattr(core, "value_target", lambda z, v, ok, lam: lam * v + (1.0 - lam) * z)
    with pytest.raises(AssertionError, match="is not its target's"):
        _assert_the_mix_trains_its_own_target(tmp_path)


def test_the_step_event_counts_the_value_rows_the_mix_reaches(tmp_path: Path) -> None:
    """`root_value_rows`, the mix's fire-rate, counts the step's value-supervised rows with a root value."""
    sink = H.SpySink()
    trainer = H.tiny_graph_trainer(tmp_path, sink=sink, value_target_lambda=0.5, checkpoint_interval=0)
    replay = H.ReplayWireBuffer(_ring(), len(_ROWS))
    run_declared_train_step(trainer, replay, H.GSPEC, batch_size=len(_ROWS), augment=False,
                            caps_provider=lambda: _CAPS, sample_threads_provider=lambda: 1)
    t = replay.targets
    rooted = np.asarray(t.root_value_valid) != 0
    assert int(rooted.sum()) == 4 and sorted(np.asarray(t.outcomes).tolist()) == sorted(z for z, _v, _ok in _ROWS)
    assert sink.named("trainer_step")[0]["root_value_rows"] == 3
