"""`train.value_mask_redraw_p`: each train step keeps a Bernoulli(p) subset of the value-valid rows, re-drawn every step."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import _microbatch_harness as H
import numpy as np
import pytest

from mantis._engine import HexgBuffer
from mantis.config.resolve.microbatch import MicrobatchCapsSpec
from mantis.train import losses
from mantis.train.coordinator.dispatch import run_declared_eval_step, run_declared_train_step
from mantis.train.trainer.core import TrainHParams

_STONES = [(0, 0, 1), (1, 0, -1), (0, 1, 1)]
_VISITS = [(2, 0, 0.6), (1, 1, 0.4)]
_CAPS = MicrobatchCapsSpec(max_edges=100_000_000, max_nodes=4_000_000)
_P = 0.125
_SEED = 20261004
_ROWS = 64


def _ring(n: int = _ROWS, *, valid: list[bool] | None = None) -> HexgBuffer:
    """`n` one-row games, every row value-valid unless `valid` says otherwise; the sampler seeded."""
    buf = HexgBuffer(2 * n, H.GRAPH_ENCODING, 128)
    for i in range(n):
        ok = True if valid is None else valid[i]
        buf.push_graph_position(_STONES[: 2 + i % 2], _VISITS, 1, 30, 2 + i, True, 1.0 if i % 3 else -1.0, ok,
                                10 + i, game_id=i)
    buf.seed_sampler(H.SEED)
    return buf


def _masks(p: float, seed: int, steps: range, n: int = _ROWS) -> np.ndarray:
    """The kept rows of `n` value-valid rows at each step, one boolean row per step."""
    vv = np.ones(n, dtype=np.uint8)
    return np.stack([losses.redraw_value_mask(vv, p, seed, s) != 0 for s in steps])


def _assert_rows_are_dropped(kept: np.ndarray) -> None:
    """At p = 1/8 the kept share is p within four binomial SDs, and no step keeps every row."""
    share = float(kept.mean())
    sd = (_P * (1 - _P) / kept.size) ** 0.5
    assert abs(share - _P) < 4 * sd, f"kept share {share:.4f} is not p = {_P}"
    assert not kept.all(axis=1).any(), "a step kept every row"


def _assert_redrawn_every_step(kept: np.ndarray) -> None:
    """Consecutive steps draw different subsets, and every row is kept at about p across steps (a fixed subset reads 0 or 1)."""
    assert all((kept[i] != kept[i + 1]).any() for i in range(len(kept) - 1)), "two consecutive steps drew one subset"
    per_row = kept.mean(axis=0)
    assert per_row.min() > 0.0 and per_row.max() < 0.4, f"per-row keep rates span {per_row.min()}..{per_row.max()}"


def test_at_one_eighth_rows_are_dropped_and_the_planted_break_reds(monkeypatch: pytest.MonkeyPatch) -> None:
    """PLANTED BREAK: a mask that drops no row at p = 1/8 reds the checker."""
    _assert_rows_are_dropped(_masks(_P, _SEED, range(400)))
    monkeypatch.setattr(losses, "redraw_value_mask", lambda vv, p, seed, step: vv)
    with pytest.raises(AssertionError, match="is not p"):
        _assert_rows_are_dropped(_masks(_P, _SEED, range(400)))


def test_the_subset_is_redrawn_every_step_and_the_planted_break_reds(monkeypatch: pytest.MonkeyPatch) -> None:
    """PLANTED BREAK: a subset drawn once (the step left out of the key) reds the checker."""
    _assert_redrawn_every_step(_masks(_P, _SEED, range(400)))
    real = losses.redraw_value_mask
    monkeypatch.setattr(losses, "redraw_value_mask", lambda vv, p, seed, step: real(vv, p, seed, 0))
    with pytest.raises(AssertionError, match="one subset"):
        _assert_redrawn_every_step(_masks(_P, _SEED, range(400)))


def test_the_draw_is_a_function_of_the_seed_and_the_step_alone() -> None:
    """A resumed trainer at step s draws step s's subset; another seed draws another stream; an invalid row stays out."""
    assert (_masks(_P, _SEED, range(10, 20)) == _masks(_P, _SEED, range(10, 20))).all()
    assert (_masks(_P, _SEED, range(10, 20)) != _masks(_P, _SEED + 1, range(10, 20))).any()
    vv = np.array([1, 0] * 32, dtype=np.uint8)
    for s in range(50):
        assert not losses.redraw_value_mask(vv, _P, _SEED, s)[1::2].any()


def _step(tmp_path: Path, ring: Any, mask: tuple[float, int] | None, *, train: bool = True,
          sink: Any = None) -> dict[str, float]:
    trainer = H.tiny_graph_trainer(tmp_path, value_mask=mask, checkpoint_interval=0, sink=sink)
    if not train:
        return run_declared_eval_step(trainer, ring, H.GSPEC, batch_size=_ROWS, caps_provider=lambda: _CAPS,
                                      sample_threads_provider=lambda: 1)
    return run_declared_train_step(trainer, ring, H.GSPEC, batch_size=_ROWS, augment=False,
                                   caps_provider=lambda: _CAPS, sample_threads_provider=lambda: 1)


def test_a_kept_row_is_weighted_through_the_whole_batch_denominator_as_reg1s_harness(tmp_path: Path) -> None:
    """REG-1's semantics: the masked step IS the step with the dropped rows' value_valid zeroed, a kept row ≈ 1/p."""
    replay = H.ReplayWireBuffer(_ring(), _ROWS)
    masked = _step(tmp_path, replay, (_P, _SEED))
    kept = losses.redraw_value_mask(np.asarray(replay.targets.value_valid), _P, _SEED, 0)
    assert 0 < int(kept.sum()) < _ROWS
    relabelled = H.ReplayWireBuffer(_ring(), _ROWS)
    relabelled._pair = (relabelled._pair[0], _TargetsView(relabelled.targets, value_valid=kept))
    plain = _step(tmp_path, relabelled, None)
    assert masked["value_loss"] == plain["value_loss"], "the masked step is not REG-1's zeroed-row step"
    assert masked["policy_loss"] == plain["policy_loss"], "the mask moved the policy loss"
    assert masked["value_loss"] != _step(tmp_path, H.ReplayWireBuffer(_ring(), _ROWS), None)["value_loss"]


class _TargetsView:
    """The sampler's targets with `value_valid` replaced; every other member read through."""

    def __init__(self, targets: Any, **over: Any) -> None:
        self._targets, self._over = targets, over

    def __getattr__(self, name: str) -> Any:
        over = self.__dict__["_over"]
        return over[name] if name in over else getattr(self.__dict__["_targets"], name)


def test_the_eval_step_never_masks(tmp_path: Path) -> None:
    """A reading must not move with the lever: the forward-only loss under the mask is the unmasked one, bit for bit."""
    ring = H.ReplayWireBuffer(_ring(), _ROWS)
    assert _step(tmp_path, ring, (_P, _SEED), train=False) == _step(tmp_path, ring, None, train=False)


def test_at_p_zero_no_mask_is_drawn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Off is off: at p = 0 the hparams carry no mask and the draw is never called, so the step is HEAD's."""
    assert H.graph_hparams().value_mask is None
    config = H.graph_config()
    config["train"]["value_mask_redraw_p"] = 0.0
    assert TrainHParams.from_config(config).value_mask is None
    config["train"]["value_mask_redraw_p"] = _P
    assert TrainHParams.from_config(config).value_mask == (_P, int(config["seed"]))

    def _refuse(*_a: Any) -> Any:
        raise AssertionError("the mask was drawn at p = 0")

    monkeypatch.setattr(losses, "redraw_value_mask", _refuse)
    _step(tmp_path, H.ReplayWireBuffer(_ring(), _ROWS), None)


def test_the_step_event_counts_the_valid_and_kept_rows_and_echoes_p(tmp_path: Path) -> None:
    """The mask's fire-rate: value-valid rows, the rows kept of them, and p, on every step event."""
    valid = [i % 4 != 0 for i in range(_ROWS)]
    sink = H.SpySink()
    replay = H.ReplayWireBuffer(_ring(valid=valid), _ROWS)
    _step(tmp_path, replay, (_P, _SEED), sink=sink)
    event = sink.named("trainer_step")[0]
    n_valid = int((np.asarray(replay.targets.value_valid) != 0).sum())
    kept = losses.redraw_value_mask(np.asarray(replay.targets.value_valid), _P, _SEED, 0)
    assert (event["value_rows_valid"], event["value_rows_kept"]) == (n_valid, int((kept != 0).sum()))
    assert event["value_rows_kept"] < event["value_rows_valid"] == sum(valid)
    assert event["value_mask_redraw_p"] == _P

    off = H.SpySink()
    _step(tmp_path, H.ReplayWireBuffer(_ring(valid=valid), _ROWS), None, sink=off)
    event = off.named("trainer_step")[0]
    assert event["value_rows_kept"] == event["value_rows_valid"] == sum(valid)
    assert event["value_mask_redraw_p"] == 0.0
