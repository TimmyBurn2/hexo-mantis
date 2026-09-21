"""run10's ONE hypothesis (R366(b)): LR cosine 1e-3 → 1e-4 over 108 000 steps, pinned off the minted rows through the trainer's own scheduler; and torch's cosine is PERIODIC past its horizon, which the STOP LAW (read 45k games THEN STOP) is what holds."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mantis.config.loader import load_config
from mantis.model import GnnArchV2SoftPolicy, build_net
from mantis.train.trainer.core import Trainer, TrainHParams

_REPO = Path(__file__).resolve().parents[2]
_RUN10 = _REPO / "configs" / "run10.yaml"
_HORIZON = 108_000


def _trainer(tmp_path: Path) -> Trainer:
    """A tiny net driven by run10's OWN train rows: the schedule is the config's, not a literal."""
    config = load_config(_RUN10).model_dump()
    hp = TrainHParams.from_config(config)
    assert (hp.lr, hp.lr_schedule, hp.scheduler_t_max, hp.eta_min) == (1e-3, "cosine", _HORIZON, 1e-4)
    arch = GnnArchV2SoftPolicy(in_dim=11, edge_dim=5, hidden=4, num_layers=1, policy_hidden=4, value_hidden=4)
    return Trainer(build_net(arch), config, arch=arch, checkpoint_dir=tmp_path, device=torch.device("cpu"),
                   train_hparams=hp)


def _lr_at(trainer: Trainer, step: int, *, from_step: int = 0) -> float:
    assert trainer.scheduler is not None
    for _ in range(step - from_step):
        trainer.optimizer.step()
        trainer.scheduler.step()
    return float(trainer.optimizer.param_groups[0]["lr"])


def test_the_cosine_reads_1e3_at_0_the_midpoint_at_54k_and_1e4_at_the_horizon(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)
    assert _lr_at(trainer, 0) == pytest.approx(1e-3)
    assert _lr_at(trainer, 54_000) == pytest.approx(5.5e-4, rel=1e-6)
    assert _lr_at(trainer, _HORIZON, from_step=54_000) == pytest.approx(1e-4, rel=1e-6)


def test_the_horizon_is_the_mint_row_and_not_total_steps(tmp_path: Path) -> None:
    """`train.total_steps` stays 1 000 000 by "byte-equal elsewhere"; the anneal rides `scheduler_t_max`."""
    config = load_config(_RUN10)
    assert config.train.total_steps == 1_000_000 and config.train.scheduler_t_max == _HORIZON
    assert config.train.eta_min == 1e-4 and config.train.lr == 1e-3


def test_past_the_horizon_torch_rises_again_within_the_stop_laws_window_by_under_1e6(tmp_path: Path) -> None:
    """Torch's `CosineAnnealingLR` is periodic: the run stops after the 45k-games cell (≈ 1 500 steps past 108k), over which the LR climbs by < 1e-6 — stated, not coded around."""
    trainer = _trainer(tmp_path)
    at_horizon = _lr_at(trainer, _HORIZON)
    after_cell = _lr_at(trainer, _HORIZON + 1_500, from_step=_HORIZON)
    assert after_cell > at_horizon, "premise: torch's cosine rises past T_max"
    assert after_cell - at_horizon < 1e-6
    far = _lr_at(trainer, 2 * _HORIZON, from_step=_HORIZON + 1_500)
    assert far == pytest.approx(1e-3, rel=1e-4), "and at 2 × T_max it is back at the peak — the STOP LAW is the guard"
