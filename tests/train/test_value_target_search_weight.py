"""`train.value_target_search_weight` at `end_bin`: (1 − w)·z over the masked rows plus w·the search value's end-bin CE over every value-valid row carrying one."""
from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import _microbatch_harness as H
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from mantis._engine import HexgBuffer
from mantis.config.resolve.microbatch import MicrobatchCapsSpec
from mantis.model.dist65 import scalar_to_two_hot
from mantis.train import losses
from mantis.train.coordinator import dispatch
from mantis.train.coordinator.dispatch import run_declared_eval_step, run_declared_train_step
from mantis.train.trainer import core

_STONES = [(0, 0, 1), (1, 0, -1), (0, 1, 1)]
_VISITS = [(2, 0, 0.6), (1, 1, 0.4)]
_CAPS = MicrobatchCapsSpec(max_edges=100_000_000, max_nodes=4_000_000)
# (z, root value or None, value_valid). Row 4's game has no winner, so its value is masked whatever its root value.
# Row 5's root value equals its z, so the term reaches it without moving it.
_ROWS = [(1.0, 0.5, True), (-1.0, None, True), (1.0, None, True), (-1.0, -0.25, True), (1.0, -0.5, False),
         (-1.0, -1.0, True)]
_END_BIN = "end_bin"
_MASK = (0.5, 7)


def _ring(*, roots: list[float | None] | None = None, valid: list[bool] | None = None) -> HexgBuffer:
    """The six rows, one game each, sampler seeded; `roots` / `valid` replace the rows' root values / value flags."""
    buf = HexgBuffer(64, H.GRAPH_ENCODING, 128)
    for i, (z, v, ok) in enumerate(_ROWS):
        v = v if roots is None else roots[i]
        kw: dict[str, Any] = {"root_value": v, "root_value_valid": True} if v is not None else {}
        buf.push_graph_position(_STONES[: 2 + i % 2], _VISITS, 1, 30, 2 + i, True, z, ok if valid is None else valid[i],
                                10 + i, game_id=i, **kw)
    buf.seed_sampler(H.SEED)
    return buf


def _value_loss(tmp_path: Path, ring: HexgBuffer, weight: float, *, train: bool = True,
                mask: tuple[float, int] | None = None) -> float:
    """One step's value loss from a fresh seeded trainer: every call reads the same rows at the same weights."""
    trainer = H.tiny_graph_trainer(tmp_path, value_target_search_weight=weight, checkpoint_interval=0, value_mask=mask,
                                   value_target_search_encoding=_END_BIN if weight > 0 else None)
    step: Callable[..., dict[str, float]] = run_declared_train_step if train else run_declared_eval_step
    kw: dict[str, Any] = {"augment": False} if train else {}
    out = step(trainer, ring, H.GSPEC, batch_size=len(_ROWS), caps_provider=lambda: _CAPS,
               sample_threads_provider=lambda: 1, **kw)
    return float(out["value_loss"])


def _uniform_ring() -> HexgBuffer:
    """Eight rows of one shape, so caps split the step exactly: rows 6–7 carry no search value, rows 1 and 4 no value."""
    buf = HexgBuffer(64, H.GRAPH_ENCODING, 128)
    for i in range(8):
        kw: dict[str, Any] = {"root_value": 0.25 * (i % 4) - 0.5, "root_value_valid": True} if i < 6 else {}
        buf.push_graph_position(_STONES, _VISITS, 1, 30, 2 + i, True, 1.0 if i % 2 == 0 else -1.0, i not in (1, 4),
                                10 + i, game_id=i, **kw)
    buf.seed_sampler(H.SEED)
    return buf


def _split_value_loss(tmp_path: Path, m: int) -> float:
    """The armed step's value loss with the mask on, the batch cut into exactly `m` parts."""
    replay = H.ReplayWireBuffer(_uniform_ring(), 8)
    caps = H.caps_for_exactly(replay.wire, m)
    trainer = H.tiny_graph_trainer(tmp_path / f"m{m}", value_target_search_weight=0.5, checkpoint_interval=0,
                                   value_target_search_encoding=_END_BIN, value_mask=_MASK)
    out = run_declared_train_step(trainer, replay, H.GSPEC, batch_size=8, augment=False,
                                  caps_provider=lambda: MicrobatchCapsSpec(*caps), sample_threads_provider=lambda: 1)
    return float(out["value_loss"])


def _assert_the_parts_sum_to_the_whole_step(tmp_path: Path) -> None:
    """Each term divides by its WHOLE step's rows in every part, so a split step reads the un-split loss."""
    whole = _split_value_loss(tmp_path, 1)
    for m in (2, 4):
        assert _split_value_loss(tmp_path, m) == pytest.approx(whole, rel=1e-5), m


_REAL_MASK = losses.redraw_value_mask


def _after_the_mask(value_valid: Any, root_value_valid: Any) -> np.ndarray:
    """The planted break of the term's rows: read AFTER the step-0 value mask, so it loses what the mask dropped."""
    return (_REAL_MASK(np.asarray(value_valid), *_MASK, 0) != 0) & (np.asarray(root_value_valid) != 0)


def _rooted(v: float) -> list[float | None]:
    """Every row that carries a root value carries `v`."""
    return [None if r is None else v for _z, r, _ok in _ROWS]


def _assert_linear_in_the_teacher(tmp_path: Path) -> None:
    """The end-bin CE is linear in the win probability: L(½) = ¾·L(+1) + ¼·L(−1) on the same rows at w = 1."""
    up, down, half = (_value_loss(tmp_path, _ring(roots=_rooted(v)), 1.0) for v in (1.0, -1.0, 0.5))
    assert math.isclose(half, 0.75 * up + 0.25 * down, rel_tol=0, abs_tol=1e-6), (half, up, down)


def _assert_the_mask_is_z_s_alone(tmp_path: Path, unmasked: float) -> None:
    """At w = 1 (z out of the loss) the value mask moves nothing: the term trains every value-valid row with a search value."""
    assert _value_loss(tmp_path, _ring(), 1.0, mask=_MASK) == unmasked


def _assert_no_root_no_term(tmp_path: Path) -> None:
    """At w = 1 a row with no search value is outside the term: dropping its value flag moves nothing."""
    unrooted_out = [ok and v is not None for _z, v, ok in _ROWS]
    assert _value_loss(tmp_path, _ring(valid=unrooted_out), 1.0) == _value_loss(tmp_path, _ring(), 1.0)


def test_at_weight_zero_the_trainer_is_byte_equal_whatever_the_rows_root_values(tmp_path: Path) -> None:
    """The trainer at weight 0: rows with and without root values train to the same bits, losses and weights alike."""
    def run(ring: HexgBuffer) -> tuple[list[dict[str, str]], bytes]:
        trainer = H.tiny_graph_trainer(tmp_path, value_target_search_weight=0.0, checkpoint_interval=0)
        replay = H.ReplayWireBuffer(ring, len(_ROWS))
        steps = []
        for _ in range(3):
            out = run_declared_train_step(trainer, replay, H.GSPEC, batch_size=len(_ROWS), augment=False,
                                          caps_provider=lambda: _CAPS, sample_threads_provider=lambda: 1)
            steps.append({k: float(out[k]).hex() for k in ("loss", "policy_loss", "value_loss", "grad_norm")})
        return steps, H.param_vector(trainer.model).numpy().tobytes()

    no_roots: list[float | None] = [None] * len(_ROWS)
    with_roots, without = run(_ring()), run(_ring(roots=no_roots))
    assert with_roots[0] == without[0]
    assert with_roots[1] == without[1], "the weights moved differently at weight 0"


def test_a_teacher_equal_to_z_on_every_value_row_trains_exactly_the_z_loss(tmp_path: Path) -> None:
    """v = z = ±1 is one-hot z on the end bins and the term's denominator is z's (no mask): w = ½ IS w = 0, bit for bit."""
    zs: list[float | None] = [z for z, _v, _ok in _ROWS]
    assert _value_loss(tmp_path, _ring(roots=zs), 0.5) == _value_loss(tmp_path, _ring(roots=zs), 0.0)


def test_the_end_bin_term_is_linear_in_the_teacher_and_the_planted_two_hot_reds(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """PLANTED BREAK: the old knob's encoding, two-hot of a fractional value, targets interior bins; the checker reds."""
    _assert_linear_in_the_teacher(tmp_path)

    def two_hot(bin_logits: torch.Tensor, value: torch.Tensor, denominator: float) -> torch.Tensor:
        logp = F.log_softmax(bin_logits.float(), dim=-1)
        return -(scalar_to_two_hot(value) * logp).sum() / denominator
    monkeypatch.setattr(core, "_end_bin_value_loss", two_hot)
    with pytest.raises(AssertionError):
        _assert_linear_in_the_teacher(tmp_path)


def test_the_value_mask_is_z_s_alone_and_the_planted_break_reds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """PLANTED BREAK: the term's rows read after the mask lose what the mask dropped; the checker reds."""
    unmasked = _value_loss(tmp_path, _ring(), 1.0)
    _assert_the_mask_is_z_s_alone(tmp_path, unmasked)
    monkeypatch.setattr(losses, "search_value_rows", _after_the_mask)
    with pytest.raises(AssertionError):
        _assert_the_mask_is_z_s_alone(tmp_path, unmasked)


def test_a_row_with_no_search_value_is_outside_the_term_and_the_planted_break_reds(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """PLANTED BREAK: a term that ignores the root flag trains a row toward its default 0; the checker reds."""
    _assert_no_root_no_term(tmp_path)
    monkeypatch.setattr(losses, "search_value_rows", lambda vv, rv: np.asarray(vv) != 0)
    with pytest.raises(AssertionError):
        _assert_no_root_no_term(tmp_path)


def test_the_parts_sum_to_the_whole_step_and_a_per_part_denominator_reds(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """PLANTED BREAK: a part dividing by its own term rows weighs the step by how it was split; the checker reds."""
    _assert_the_parts_sum_to_the_whole_step(tmp_path)
    real = dispatch.GraphStepInputs

    def per_part(**kw: Any) -> Any:
        return real(**{**kw, "teacher_denominator": max(float(kw["teacher_rows"].numel()), 1.0)})
    monkeypatch.setattr(dispatch, "GraphStepInputs", per_part)
    with pytest.raises(AssertionError):
        _assert_the_parts_sum_to_the_whole_step(tmp_path)


def test_the_eval_step_reads_z_whatever_the_weight(tmp_path: Path) -> None:
    """A reading must not reward the lever: the forward-only value loss at weight > 0 is the weight 0 one, bit for bit."""
    assert _value_loss(tmp_path, _ring(), 0.5, train=False) == _value_loss(tmp_path, _ring(), 0.0, train=False)


def _assert_the_event_counts_the_rows_before_the_mask(tmp_path: Path) -> None:
    """Value-valid rows with a root value, read before the mask (reached), v != z among them (moved), w echoed."""
    sink = H.SpySink()
    trainer = H.tiny_graph_trainer(tmp_path, sink=sink, value_target_search_weight=0.25, checkpoint_interval=0,
                                   value_target_search_encoding=_END_BIN, value_mask=_MASK)
    replay = H.ReplayWireBuffer(_ring(), len(_ROWS))
    run_declared_train_step(trainer, replay, H.GSPEC, batch_size=len(_ROWS), augment=False,
                            caps_provider=lambda: _CAPS, sample_threads_provider=lambda: 1)
    event = sink.named("trainer_step")[0]
    assert (event["root_value_rows"], event["root_value_rows_moved"]) == (3, 2)
    assert event["value_target_search_weight"] == 0.25


def test_the_step_event_counts_the_rows_the_term_reaches_before_the_mask_and_the_planted_break_reds(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """PLANTED BREAK: counting after the mask reads fewer rows — the mask drops a rooted row here by construction."""
    t = H.ReplayWireBuffer(_ring(), len(_ROWS)).targets
    vv, rooted = np.asarray(t.value_valid), np.asarray(t.root_value_valid) != 0
    assert int(rooted.sum()) == 4 and sorted(np.asarray(t.outcomes).tolist()) == sorted(z for z, _v, _ok in _ROWS)
    assert int(_after_the_mask(vv, rooted).sum()) != 3, "the mask keeps every rooted row: the read proves nothing"
    _assert_the_event_counts_the_rows_before_the_mask(tmp_path)
    monkeypatch.setattr(losses, "search_value_rows", _after_the_mask)
    with pytest.raises(AssertionError):
        _assert_the_event_counts_the_rows_before_the_mask(tmp_path)


@pytest.mark.parametrize(("weight", "encoding", "match"), [(0.5, None, "exactly at weight 0"),
                                                         (0.0, _END_BIN, "exactly at weight 0"),
                                                         (0.5, "two_hot", "not an encoding the trainer has")])
def test_a_weight_and_an_encoding_that_disagree_are_refused(weight: float, encoding: str | None, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        H.graph_hparams(value_target_search_weight=weight, value_target_search_encoding=encoding)
