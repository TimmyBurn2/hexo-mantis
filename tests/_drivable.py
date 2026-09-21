"""The composition seam's drivable trainer double — ONE copy for every root/wiring test (R367(a)); importable from any test directory through the root conftest's own path, no `sys.path` write."""
from __future__ import annotations

from typing import Any


class DrivableTrainerStub:
    """The composition seam's trainer double, the ONE copy (R367(a)): the declared entry points plus `device`, with `actor_sd` and `inference_sd` DISTINCT so a root that hands the deploy view to the actors reds; `on_step` fires after each step, `saves` records every checkpoint call."""

    def __init__(self, *, step: int = 0, grad_norm: float = 0.1, on_step: Any = None, model: Any = None) -> None:
        self.step = step
        self.device = "cpu"
        self.model = object() if model is None else model
        self.grad_norm = grad_norm
        self.on_step = on_step
        self.saves: list = []
        self.actor_sd: dict = {"w": "ACTOR-SENTINEL"}
        self.inference_sd: dict = {"w": "DEPLOY-SENTINEL"}

    def loss_info(self) -> dict[str, float]:
        return {"loss": 1.0, "policy_loss": 0.6, "value_loss": 0.4, "grad_norm": self.grad_norm,
                "policy_entropy": 2.0, "value_accuracy": 0.5, "lr": 1e-3,
                "opp_reply_loss": 0.0, "loss_total": 1.0}

    def train_step_from_tensors(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        self.step += 1
        if self.on_step is not None:
            self.on_step(self.step)
        return self.loss_info()

    def train_step_from_graph_batch(self, **kwargs: Any) -> dict[str, float]:
        return self.train_step_from_tensors()

    def inference_state_dict(self) -> dict:
        return self.inference_sd

    def actor_state_dict(self) -> dict:
        return self.actor_sd

    def deploy_module(self) -> Any:
        return self.model

    def save_checkpoint(self, loss_info: Any) -> Any:
        self.saves.append(loss_info)
        return None

