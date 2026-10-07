"""The trainer's cosine schedule: torch's `CosineAnnealingLR` is PERIODIC past `T_max`; this one holds `eta_min` there."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR


@dataclass(frozen=True)
class LrCycle:
    """A second cosine cycle: from step `start`, `lr` annealed to `eta_min` over `t_max` steps, then `eta_min`."""

    start: int
    lr: float
    eta_min: float
    t_max: int

    def lr_at(self, step: int) -> float:
        """The cycle's closed-form LR at `step` (>= `start`)."""
        k = min(step - self.start, self.t_max)
        return self.eta_min + (self.lr - self.eta_min) * (1 + math.cos(math.pi * k / self.t_max)) / 2


class FlooredCosineAnnealingLR(CosineAnnealingLR):
    """Cosine from the base LR to `eta_min` over `T_max`, then `eta_min`; from a declared `cycle`'s start, that cycle.

    The cycle is the config's: it stays out of `state_dict`, and a restore past its start re-points the optimizer at it.
    """

    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.0, last_epoch: int = -1,
                 *, cycle: LrCycle | None = None) -> None:
        self.cycle = cycle
        super().__init__(optimizer, T_max=T_max, eta_min=eta_min, last_epoch=last_epoch)

    def _active_cycle(self) -> LrCycle | None:
        cycle = self.cycle
        return cycle if cycle is not None and self.last_epoch >= cycle.start else None

    def get_lr(self) -> list[float | Tensor]:
        if (cycle := self._active_cycle()) is not None:
            return [cycle.lr_at(self.last_epoch) for _ in self.base_lrs]
        if self.last_epoch >= self.T_max:
            return [float(self.eta_min) for _ in self.base_lrs]
        return super().get_lr()

    def _get_closed_form_lr(self) -> list[float | Tensor]:
        if (cycle := self._active_cycle()) is not None:
            return [cycle.lr_at(self.last_epoch) for _ in self.base_lrs]
        if self.last_epoch >= self.T_max:
            return [float(self.eta_min) for _ in self.base_lrs]
        return super()._get_closed_form_lr()

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state.pop("cycle", None)
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        if (cycle := self._active_cycle()) is not None:
            lr = cycle.lr_at(self.last_epoch)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            self._last_lr = [lr for _ in self.optimizer.param_groups]
