"""The LR schedule is the config's rows through the trainer's own scheduler and never rises: cosine to `eta_min` at the horizon, held there after it, so the STOP LAW governs games, not the LR."""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from mantis.config.loader import discover_configs, load_config
from mantis.encoding import lookup
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


def _schedule(name: str) -> tuple[float, float, int, int]:
    """A config's `(lr, eta_min, horizon, budget)` off its own rows."""
    hp = TrainHParams.from_config(load_config(_CONFIGS / name).model_dump())
    horizon = hp.scheduler_t_max if hp.scheduler_t_max is not None else hp.total_steps
    assert horizon is not None
    return (hp.lr, hp.eta_min, int(horizon), int(load_config(_CONFIGS / name).train.max_train_steps))


_SCHEDULES = sorted({_schedule(name) for name in _COSINE})


@pytest.mark.parametrize("name", _COSINE)
def test_every_cosine_config_builds_the_floored_schedule_from_its_own_rows(name: str, tmp_path: Path) -> None:
    """The trainer's scheduler is the floored class at the config's horizon and floor — no literal in between."""
    config = load_config(_CONFIGS / name)
    hp = TrainHParams.from_config(config.model_dump())
    spec = lookup(config.identity.encoding)
    kind = ARCH_KINDS[config.identity.arch_kind or INCUMBENT_ARCH_KIND[config.identity.representation]]
    arch = kind(in_dim=spec.node_feat_dim, edge_dim=spec.edge_feat_dim, hidden=4, num_layers=1, policy_hidden=4, value_hidden=4)
    trainer = Trainer(build_net(arch), config.model_dump(), arch=arch, checkpoint_dir=tmp_path,
                      device=torch.device("cpu"), train_hparams=hp)
    lr, eta_min, horizon, _budget = _schedule(name)
    assert isinstance(trainer.scheduler, FlooredCosineAnnealingLR)
    assert (trainer.scheduler.T_max, trainer.scheduler.eta_min, set(trainer.scheduler.base_lrs)) == (horizon, eta_min, {lr})


@pytest.mark.parametrize("schedule", _SCHEDULES, ids=[f"lr{s[0]}-eta{s[1]}-T{s[2]}-budget{s[3]}" for s in _SCHEDULES])
def test_every_distinct_schedule_never_rises_inside_its_budget(schedule: tuple[float, float, int, int]) -> None:
    """Off each distinct `(lr, eta_min, horizon, budget)` the configs declare: `lr` at 0, the midpoint at T/2, `eta_min` from T on, monotone non-increasing over the whole budget — one sweep per schedule, not per config."""
    lr, eta_min, horizon, budget = schedule
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=lr)
    sweep = _lr_sweep(FlooredCosineAnnealingLR(optimizer, T_max=horizon, eta_min=eta_min), max(budget, horizon))
    assert sweep[0] == pytest.approx(lr)
    assert sweep[horizon // 2] == pytest.approx(eta_min + (lr - eta_min) * (1 + math.cos(math.pi * (horizon // 2) / horizon)) / 2, rel=1e-6)
    assert all(lr_ == pytest.approx(eta_min, rel=1e-6) for lr_ in sweep[horizon:]), "eta_min from the horizon on"
    assert all(b <= a + 1e-15 for a, b in zip(sweep, sweep[1:])), "the LR rose inside the budget"
