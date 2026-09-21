"""The LR schedule is the config's rows through the trainer's own scheduler, and it never rises: cosine to `eta_min` at the horizon, HELD there after it (torch's parent is periodic past `T_max`), so the STOP LAW governs games, not the learning rate."""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from mantis.config.loader import discover_configs, load_config
from mantis.model import ARCH_KINDS, INCUMBENT_ARCH_KIND, build_net
from mantis.train.lr_schedule import FlooredCosineAnnealingLR
from mantis.train.trainer.core import Trainer, TrainHParams

_REPO = Path(__file__).resolve().parents[2]
_CONFIGS = _REPO / "configs"
_COSINE = sorted(p.name for p in discover_configs(_CONFIGS) if load_config(p).train.lr_schedule == "cosine")


def _lr_sweep(scheduler: torch.optim.lr_scheduler.LRScheduler, steps: int) -> list[float]:
    """The LR at every step in `[0, steps]`, one `scheduler.step()` per training step as the trainer takes it."""
    optimizer = scheduler.optimizer
    optimizer.step()
    out = [float(optimizer.param_groups[0]["lr"])]
    for _ in range(steps):
        scheduler.step()
        out.append(float(optimizer.param_groups[0]["lr"]))
    return out


def _floored(t_max: int, eta_min: float, *, floored: bool = True) -> torch.optim.lr_scheduler.LRScheduler:
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    cls = FlooredCosineAnnealingLR if floored else CosineAnnealingLR
    return cls(optimizer, T_max=t_max, eta_min=eta_min)


def test_the_floor_holds_where_torchs_cosine_rises_again() -> None:
    """The planted break: the parent class climbs back to the peak at 2 × T_max; the floored one reads `eta_min` from the horizon on."""
    parent = _lr_sweep(_floored(100, 1e-4, floored=False), 200)
    assert parent[200] == pytest.approx(1e-3, rel=1e-6), "premise: torch's cosine is periodic past T_max"
    ours = _lr_sweep(_floored(100, 1e-4), 200)
    assert ours[:101] == pytest.approx(parent[:101]), "the two agree everywhere the parent is monotone"
    assert all(lr == 1e-4 for lr in ours[100:]), "from the horizon on, exactly eta_min"
    assert all(b <= a for a, b in zip(ours, ours[1:])), "never rises"


def test_the_floor_survives_a_resume_from_the_parents_state() -> None:
    """A stamp written by the un-floored class restores into the floored one (same state format) and keeps the floor."""
    old = _floored(100, 1e-4, floored=False)
    _lr_sweep(old, 150)
    new = _floored(100, 1e-4)
    new.optimizer.step()
    new.load_state_dict(old.state_dict())
    new.step()
    assert float(new.optimizer.param_groups[0]["lr"]) == 1e-4


@pytest.mark.parametrize("name", _COSINE)
def test_every_cosine_config_reads_its_own_rows_and_never_rises_inside_its_budget(name: str, tmp_path: Path) -> None:
    """Off each config's rows through the trainer: `lr` at 0, the midpoint at T/2, `eta_min` at T, monotone non-increasing over the whole `max_train_steps` budget."""
    config = load_config(_CONFIGS / name)
    hp = TrainHParams.from_config(config.model_dump())
    horizon = hp.scheduler_t_max if hp.scheduler_t_max is not None else hp.total_steps
    assert horizon is not None
    kind = ARCH_KINDS[config.identity.arch_kind or INCUMBENT_ARCH_KIND[config.identity.representation]]
    arch = kind(in_dim=11, edge_dim=5, hidden=4, num_layers=1, policy_hidden=4, value_hidden=4)
    trainer = Trainer(build_net(arch), config.model_dump(), arch=arch, checkpoint_dir=tmp_path,
                      device=torch.device("cpu"), train_hparams=hp)
    assert isinstance(trainer.scheduler, FlooredCosineAnnealingLR)
    budget = int(config.train.max_train_steps)
    sweep = _lr_sweep(trainer.scheduler, max(budget, horizon))
    assert sweep[0] == pytest.approx(hp.lr)
    assert sweep[horizon // 2] == pytest.approx(hp.eta_min + (hp.lr - hp.eta_min) * (1 + math.cos(math.pi * (horizon // 2) / horizon)) / 2, rel=1e-6)
    assert sweep[horizon] == pytest.approx(hp.eta_min, rel=1e-6)
    assert all(b <= a + 1e-15 for a, b in zip(sweep, sweep[1:])), f"{name}: the LR rose inside its budget"
    assert sweep[budget] == pytest.approx(hp.eta_min if budget >= horizon else sweep[budget], rel=1e-6)
