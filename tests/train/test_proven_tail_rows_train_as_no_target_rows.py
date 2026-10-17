"""A decided tail's rows train exactly as the no-target rows they are, flag or no flag, under every production arch kind."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch

import _microbatch_harness as H
from mantis._engine import HexgBuffer
from mantis.config.census import production_configs
from mantis.config.loader import load_config
from mantis.config.resolve.microbatch import MicrobatchCapsSpec
from mantis.train.coordinator.dispatch import run_declared_train_step

_REPO = Path(__file__).resolve().parents[2]

_STONES = [(0, 0, 1), (1, 0, -1), (0, 1, 1)]
_ANCHOR = [(2, 0, 0.6), (1, 1, 0.4)]
_KEYS = ("loss", "policy_loss", "value_loss", "aux_soft_policy_loss", "aux_soft_policy_kl_hard_vs_soft")
_KINDS: dict[str, Callable[..., Any]] = {
    "GnnArchV2SoftPolicy": H.soft_policy_graph_trainer,
    "GnnArchV2SoftPolicyFuture": H.future_graph_trainer,
}
#: The arch kinds the configs census runs; one with no trainer harness above fails the harness check below.
_CENSUS = sorted({str(load_config(path).identity.arch_kind) for path in production_configs(_REPO)})


def _future(i: int) -> list[tuple[int, int, int]]:
    """Twenty placements from ply 3 of a twenty-turn game, P1's two then P2's two."""
    return [(q, 2 + (q + i) % 2, 1 if k % 4 < 2 else -1) for k, q in enumerate(range(-10, 10))]


def _ring(winner_entry: list[tuple[int, int, float]], *, tail: bool, flagged: bool) -> HexgBuffer:
    """Rows 0-3 full rows on one target; rows 4-7 the probed rows, a tail's winner (even, one entry, +1) and loser (odd, value-only, −1)."""
    buf = HexgBuffer(64, H.GRAPH_ENCODING, 128)
    for i in range(8):
        probed = i >= 4
        winner = i % 2 == 0
        visits = (winner_entry if winner else []) if probed and tail else (winner_entry if probed else _ANCHOR)
        buf.push_graph_position(_STONES, visits, 1, 2, 3, not (probed and tail), 1.0 if winner else -1.0, True, 20,
                                game_id=i, tail_mass=0.0, root_value=(1.0 if winner else -1.0) if probed else 0.0,
                                root_value_valid=probed, future_stones=_future(i), proven_tail=probed and flagged)
    return buf


def _step(tmp_path: Path, kind: str, buf: HexgBuffer) -> tuple[dict[str, Any], torch.Tensor]:
    """One seeded train step of `kind` over `buf`: its `trainer_step` event and its gradient."""
    buf.seed_sampler(H.SEED)
    replay = H.ReplayWireBuffer(buf, 8)
    sink = H.SpySink()
    trainer = _KINDS[kind](tmp_path, sink=sink)
    with H.deterministic_algorithms():
        run_declared_train_step(
            trainer, replay, H.GSPEC, batch_size=8, augment=False,
            caps_provider=lambda: MicrobatchCapsSpec(*H.non_binding_caps(replay.wire)),
            sample_threads_provider=lambda: 1)
    (event,) = sink.named("trainer_step")
    return event, H.grad_vector(trainer.model)


def test_every_census_kind_has_a_harness() -> None:
    assert _CENSUS and set(_CENSUS) <= set(_KINDS), f"census kinds without a trainer harness: {set(_CENSUS) - set(_KINDS)}"


@pytest.mark.parametrize("kind", _CENSUS)
def test_a_tail_row_trains_as_its_unflagged_and_value_only_twins(tmp_path: Path, kind: str) -> None:
    """PLANTED BREAK: weigh a winner row's entry anywhere in the step, or read the flag, and a key moves."""
    flagged, flagged_grad = _step(tmp_path / "a", kind, _ring([(1, 1, 1.0)], tail=True, flagged=True))
    plain, plain_grad = _step(tmp_path / "b", kind, _ring([(1, 1, 1.0)], tail=True, flagged=False))
    moved, moved_grad = _step(tmp_path / "c", kind, _ring([(2, 0, 1.0)], tail=True, flagged=True))
    empty, empty_grad = _step(tmp_path / "d", kind, _ring([], tail=True, flagged=True))
    for key in _KEYS:
        assert flagged[key] == plain[key] == moved[key] == empty[key], (kind, key, flagged[key], empty[key])
    if kind == "GnnArchV2SoftPolicyFuture":
        assert flagged["aux_future_rows"] > 0 and flagged["aux_future_ce"] == empty["aux_future_ce"]
    for grad in (plain_grad, moved_grad, empty_grad):
        assert torch.equal(flagged_grad, grad)


@pytest.mark.parametrize("kind", _CENSUS)
def test_the_same_entry_swap_on_a_full_row_moves_the_step(tmp_path: Path, kind: str) -> None:
    """The control: where the policy trains, the swap the tail rows shrug off moves every policy key."""
    anchor, anchor_grad = _step(tmp_path / "a", kind, _ring([(1, 1, 1.0)], tail=False, flagged=False))
    moved, moved_grad = _step(tmp_path / "b", kind, _ring([(2, 0, 1.0)], tail=False, flagged=False))
    assert anchor["policy_loss"] != moved["policy_loss"]
    assert anchor["aux_soft_policy_loss"] != moved["aux_soft_policy_loss"]
    assert not torch.equal(anchor_grad, moved_grad)
