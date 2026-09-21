"""The trainer's cosine schedule: torch's `CosineAnnealingLR` is PERIODIC past `T_max`; this one holds `eta_min` there."""
from __future__ import annotations

from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR


class FlooredCosineAnnealingLR(CosineAnnealingLR):
    """Cosine from the base LR to `eta_min` over `T_max`, then `eta_min` for every later step — monotone non-increasing by construction, same state format as the parent."""

    def get_lr(self) -> list[float | Tensor]:
        if self.last_epoch >= self.T_max:
            return [float(self.eta_min) for _ in self.base_lrs]
        return super().get_lr()

    def _get_closed_form_lr(self) -> list[float | Tensor]:
        if self.last_epoch >= self.T_max:
            return [float(self.eta_min) for _ in self.base_lrs]
        return super()._get_closed_form_lr()
