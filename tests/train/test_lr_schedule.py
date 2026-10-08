"""The LR schedule is the config's rows through the trainer's own scheduler: cosine to `eta_min` at the horizon, held there after it, so the STOP LAW governs games, not the LR; a declared second cycle is its one rise, config-owned through a resume."""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from mantis.config.loader import discover_configs, load_config
from mantis.encoding import lookup
from mantis.model import ARCH_KINDS, INCUMBENT_ARCH_KIND, build_net
from mantis.train.checkpoints import resume_trainer, save_checkpoint
from mantis.train.lr_schedule import FlooredCosineAnnealingLR, LrCycle
from mantis.train.orchestrator import build_resume_config_overrides
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


def _floored(t_max: int, eta_min: float, *, floored: bool = True,
             cycle: LrCycle | None = None) -> torch.optim.lr_scheduler.LRScheduler:
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    if not floored:
        return CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    return FlooredCosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min, cycle=cycle)


def _cosine(lr: float, eta_min: float, k: int, t_max: int) -> float:
    return eta_min + (lr - eta_min) * (1 + math.cos(math.pi * k / t_max)) / 2


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


def _schedule(name: str) -> tuple[float, float, int, int, LrCycle | None]:
    """A config's `(lr, eta_min, horizon, budget, cycle)` off its own rows."""
    hp = TrainHParams.from_config(load_config(_CONFIGS / name).model_dump())
    return (hp.lr, hp.eta_min, int(hp.scheduler_t_max), int(load_config(_CONFIGS / name).train.max_train_steps),
            hp.lr_cycle)


_SCHEDULES = sorted({_schedule(name) for name in _COSINE}, key=repr)


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
    lr, eta_min, horizon, _budget, cycle = _schedule(name)
    assert isinstance(trainer.scheduler, FlooredCosineAnnealingLR)
    assert (trainer.scheduler.T_max, trainer.scheduler.eta_min, set(trainer.scheduler.base_lrs)) == (horizon, eta_min, {lr})
    assert trainer.scheduler.cycle == cycle


@pytest.mark.parametrize("schedule", _SCHEDULES,
                         ids=[f"lr{s[0]}-eta{s[1]}-T{s[2]}-budget{s[3]}-cycle{s[4] is not None}" for s in _SCHEDULES])
def test_every_distinct_schedule_rises_only_at_its_declared_cycle(
        schedule: tuple[float, float, int, int, LrCycle | None]) -> None:
    """Off each distinct schedule the configs declare: `lr` at 0, the midpoint at T/2, `eta_min` from T on, then a declared cycle's `lr` at its start, its midpoint and its floor, also through a restore at its start; monotone non-increasing everywhere but that one step — one sweep per schedule, not per config."""
    lr, eta_min, horizon, budget, cycle = schedule
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=lr)
    steps = max(budget, horizon) if cycle is None else max(budget, horizon, cycle.start + cycle.t_max + 1)
    sweep = _lr_sweep(FlooredCosineAnnealingLR(optimizer, T_max=horizon, eta_min=eta_min, cycle=cycle), steps)
    end = len(sweep) if cycle is None else cycle.start
    assert sweep[0] == pytest.approx(lr)
    if horizon // 2 < end:
        assert sweep[horizon // 2] == pytest.approx(_cosine(lr, eta_min, horizon // 2, horizon), rel=1e-6)
    assert all(lr_ == pytest.approx(eta_min, rel=1e-6) for lr_ in sweep[horizon:end]), "eta_min from the horizon on"
    rises = [i + 1 for i, (a, b) in enumerate(zip(sweep, sweep[1:])) if b > a + 1e-15]
    if cycle is None:
        assert rises == [], "the LR rose inside the budget"
        return
    assert rises == ([cycle.start] if cycle.lr > sweep[cycle.start - 1] else []), "a rise other than the cycle's start"
    assert sweep[cycle.start] == pytest.approx(cycle.lr)
    assert sweep[cycle.start + cycle.t_max // 2] == pytest.approx(
        _cosine(cycle.lr, cycle.eta_min, cycle.t_max // 2, cycle.t_max), rel=1e-6)
    assert all(lr_ == pytest.approx(cycle.eta_min, rel=1e-6) for lr_ in sweep[cycle.start + cycle.t_max:])
    first = FlooredCosineAnnealingLR(torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=lr),
                                     T_max=horizon, eta_min=eta_min)
    _lr_sweep(first, cycle.start)
    resumed = FlooredCosineAnnealingLR(torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=lr),
                                       T_max=horizon, eta_min=eta_min, cycle=cycle)
    resumed.load_state_dict(first.state_dict())
    assert resumed.optimizer.param_groups[0]["lr"] == cycle.lr, "restored at the start, before its first step"
    assert _lr_sweep(resumed, cycle.t_max + 1) == [cycle.lr_at(cycle.start + k) for k in range(cycle.t_max + 2)]


_CYCLE = LrCycle(start=60, lr=5e-4, eta_min=1e-4, t_max=20)


def test_a_cycle_restarts_at_its_start_and_floors_after_its_horizon() -> None:
    """Live through the start: the first cycle's trace bit for bit before it, then the cycle's closed form, then its floor."""
    plain = _lr_sweep(_floored(40, 1e-4), 100)
    cycled = _lr_sweep(_floored(40, 1e-4, cycle=_CYCLE), 100)
    assert cycled[:60] == plain[:60], "the first cycle is unchanged before the start"
    assert cycled[60:] == [_CYCLE.lr_at(60 + k) for k in range(41)]
    assert cycled[60] == 5e-4 and cycled[70] == pytest.approx(3e-4) and set(cycled[80:]) == {1e-4}


def test_the_cycle_is_the_configs_and_never_the_checkpoints() -> None:
    """The saved state carries no cycle; a floored state restored at the start re-points the optimizer at the cycle's lr before the first resumed step, where torch's restore leaves the optimizer's own lr."""
    old = _floored(40, 1e-4)
    _lr_sweep(old, 60)
    state = old.state_dict()
    assert "cycle" not in _floored(40, 1e-4, cycle=_CYCLE).state_dict()
    assert old.optimizer.param_groups[0]["lr"] == 1e-4, "premise: the saved run sits at its floor"
    new = _floored(40, 1e-4, cycle=_CYCLE)
    new.load_state_dict(state)
    assert new.optimizer.param_groups[0]["lr"] == 5e-4, "the first resumed step runs at the cycle's lr"


def test_a_resume_mid_cycle_continues_the_live_trace() -> None:
    """A state saved seven steps into the cycle restores into the same trace the live scheduler takes."""
    live = _floored(40, 1e-4, cycle=_CYCLE)
    trace = _lr_sweep(live, 67)
    resumed = _floored(40, 1e-4, cycle=_CYCLE)
    resumed.load_state_dict(live.state_dict())
    assert resumed.optimizer.param_groups[0]["lr"] == trace[67]
    assert _lr_sweep(resumed, 30)[1:] == [_CYCLE.lr_at(67 + k) for k in range(1, 31)]


def _nested(**train: object) -> dict:
    """The minted dev config's dump with `train.*` leaves replaced, at the tiny net's widths."""
    cfg = load_config(_CONFIGS / "dev_example.yaml").model_dump()
    cfg["train"].update(train)
    cfg["model"]["gnn"] = {"hidden": 16, "num_layers": 1}
    return cfg


def test_the_launchers_resume_runs_the_launch_configs_cycle(tmp_path: Path, tiny_net: torch.nn.Module,
                                                            optim_scaler_sched: tuple, metadata_kwargs: dict) -> None:
    """Through `build_resume_config_overrides` and `resume_trainer`, as `init_trainer` resumes: a full checkpoint saved at its floor with no cycle in its stamp, resumed under a launch that declares one, steps on the cycle."""
    optimizer, scaler, _ = optim_scaler_sched
    scheduler = FlooredCosineAnnealingLR(optimizer, T_max=40, eta_min=1e-4)
    for _ in range(60):
        optimizer.step()
        scheduler.step()
    baked = _nested(lr=1e-3, scheduler_t_max=40, eta_min=1e-4, lr_cycle=None)
    path = save_checkpoint(model=tiny_net, optimizer=optimizer, scaler=scaler,
                           scheduler=scheduler, step=60, config=baked, metadata_kwargs=metadata_kwargs,
                           checkpoint_dir=tmp_path, kind="full")
    launch = _nested(lr=1e-3, scheduler_t_max=40, eta_min=1e-4,
                     lr_cycle={"start_step": 60, "lr": 5e-4, "eta_min": 1e-4, "t_max": 20})
    trainer = resume_trainer(Trainer, path, config_overrides=build_resume_config_overrides(launch, launch), declared_encoding=None)
    assert trainer.step == 60 and trainer.scheduler.cycle == _CYCLE
    assert _lr_sweep(trainer.scheduler, 25) == [_CYCLE.lr_at(60 + k) for k in range(26)]

