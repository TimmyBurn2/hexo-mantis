"""`selfplay.decided_tails.value_weight`: a decided tail's value row weighs w in the z CE, every other row 1."""
from __future__ import annotations

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
from mantis.train.trainer.core import TrainHParams

_STONES = [(0, 0, 1), (1, 0, -1), (0, 1, 1)]
_VISITS = [(2, 0, 0.6), (1, 1, 0.4)]
_CAPS = MicrobatchCapsSpec(max_edges=100_000_000, max_nodes=4_000_000)
_ROWS = 32


def _ring() -> HexgBuffer:
    """Every fourth row a decided tail's value-only row the side to move loses; every other row searched and won."""
    buf = HexgBuffer(2 * _ROWS, H.GRAPH_ENCODING, 128)
    for i in range(_ROWS):
        if i % 4 == 0:
            buf.push_graph_position(_STONES, [], 1, 1, 4 + i, False, -1.0, True, 40, game_id=i, proven_tail=True)
        else:
            buf.push_graph_position(_STONES[: 2 + i % 2], _VISITS, 1, 30, 2 + i, True, 1.0, True, 10 + i, game_id=i)
    buf.seed_sampler(H.SEED)
    return buf


class _TargetsView:
    """The sampler's targets with named members replaced; every other member read through."""

    def __init__(self, targets: Any, **over: Any) -> None:
        self._targets, self._over = targets, over

    def __getattr__(self, name: str) -> Any:
        over = self.__dict__["_over"]
        return over[name] if name in over else getattr(self.__dict__["_targets"], name)


def _replay(**over: Any) -> H.ReplayWireBuffer:
    replay = H.ReplayWireBuffer(_ring(), _ROWS)
    if over:
        replay._pair = (replay._pair[0], _TargetsView(replay.targets, **over))
    return replay


def _tails(replay: H.ReplayWireBuffer) -> np.ndarray:
    return np.asarray(replay.targets.proven_tail) != 0


def _step(tmp_path: Path, replay: Any, weight: float, *, train: bool = True) -> tuple[dict[str, float], Any]:
    """One production step from the seeded tiny trainer at tail weight `weight`: its losses and its state after."""
    trainer = H.tiny_graph_trainer(tmp_path, tail_value_weight=weight, checkpoint_interval=0)
    if not train:
        return run_declared_eval_step(trainer, replay, H.GSPEC, batch_size=_ROWS, caps_provider=lambda: _CAPS,
                                      sample_threads_provider=lambda: 1), trainer.model.state_dict()
    out = run_declared_train_step(trainer, replay, H.GSPEC, batch_size=_ROWS, augment=False,
                                  caps_provider=lambda: _CAPS, sample_threads_provider=lambda: 1)
    return out, trainer.model.state_dict()


def _same_state(a: Any, b: Any) -> bool:
    return all(torch.equal(a[k], b[k]) for k in a)


def test_the_sampler_carries_each_rows_tail_flag() -> None:
    """The flag rides its own row through the shuffled, augmented draw: tail rows are exactly the lost value-only ones."""
    buf = _ring()
    seen = np.zeros(2, dtype=np.int64)
    for _ in range(8):
        _wire, targets = buf.sample_graph_batch(_ROWS, augment=True, n_threads=2)
        tail = np.asarray(targets.proven_tail) != 0
        assert (tail == (np.asarray(targets.outcomes) == -1.0)).all()
        assert (tail == (np.asarray(targets.is_full_search) == 0)).all()
        seen += [int(tail.sum()), int((~tail).sum())]
    assert seen.min() > 0, f"a draw of 256 rows saw {seen.tolist()} tail/other rows"


def test_weight_zero_is_the_step_with_the_tails_value_rows_masked(tmp_path: Path) -> None:
    """0 masks: the step is the weight-1 step over the same rows with each tail row's value_valid zeroed, bit for bit."""
    replay = _replay()
    tail = _tails(replay)
    assert 0 < int(tail.sum()) < _ROWS
    zero, zero_state = _step(tmp_path, replay, 0.0)
    masked_vv = np.where(tail, 0, np.asarray(replay.targets.value_valid)).astype(np.uint8)
    masked, masked_state = _step(tmp_path, _replay(value_valid=masked_vv), 1.0)
    assert zero["value_loss"] == masked["value_loss"] and zero["policy_loss"] == masked["policy_loss"]
    assert _same_state(zero_state, masked_state)


def test_weight_one_leaves_the_flag_inert(tmp_path: Path) -> None:
    """At 1 a flagged row trains as any other: the step equals the step over the same rows with no flag set."""
    replay = _replay()
    one, one_state = _step(tmp_path, replay, 1.0)
    plain, plain_state = _step(tmp_path, _replay(proven_tail=np.zeros(_ROWS, dtype=np.uint8)), 1.0)
    assert one == plain and _same_state(one_state, plain_state)


def _assert_weighted_mean(tmp_path: Path, w: float) -> None:
    """The z CE at `w` is (A + w·B) / (n + w·t): A, n over the other rows, B, t over the tails, read at weight 1."""
    replay = _replay()
    tail = _tails(replay)
    n, t = int((~tail).sum()), int(tail.sum())
    masked_vv = np.where(tail, 0, np.asarray(replay.targets.value_valid)).astype(np.uint8)
    others = _step(tmp_path, _replay(value_valid=masked_vv), 1.0)[0]["value_loss"]
    every = _step(tmp_path, replay, 1.0)[0]["value_loss"]
    a, b = n * others, (n + t) * every - n * others
    assert abs(others - every) > 1e-3, "the fixture's tail and other rows read one CE: the check would see nothing"
    got = _step(tmp_path, replay, w)[0]["value_loss"]
    want = (a + w * b) / (n + w * t)
    assert got == pytest.approx(want, rel=1e-5), f"the z CE at w={w} is {got}, not the weighted mean {want}"


def test_a_fractional_weight_is_the_weighted_mean_and_the_planted_break_reds(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """PLANTED BREAK: a dispatcher whose weights ignore the flag trains the tails at 1 and reds the check."""
    _assert_weighted_mean(tmp_path, 0.1)
    monkeypatch.setattr(losses, "tail_value_weights", lambda proven_tail, weight: np.ones(
        np.asarray(proven_tail).shape, dtype=np.float32))
    with pytest.raises(AssertionError, match="weighted mean"):
        _assert_weighted_mean(tmp_path, 0.1)


def test_the_eval_step_never_weighs(tmp_path: Path) -> None:
    """A reading must not move with the lever: the forward-only loss at 0.1 is the one at 1, bit for bit."""
    replay = _replay()
    assert _step(tmp_path, replay, 0.1, train=False)[0] == _step(tmp_path, replay, 1.0, train=False)[0]


def test_the_hparams_read_the_block_and_its_absence() -> None:
    """Off reads 1 (a preloaded tail row then trains as any other); an armed block reads its weight; no row refuses."""
    config = H.graph_config()
    assert config["selfplay"]["decided_tails"] is None
    assert TrainHParams.from_config(config).tail_value_weight == 1.0
    config["selfplay"]["decided_tails"] = {"value_weight": 0.1}
    assert TrainHParams.from_config(config).tail_value_weight == 0.1
    del config["selfplay"]["decided_tails"]
    with pytest.raises(KeyError, match="decided_tails"):
        TrainHParams.from_config(config)
