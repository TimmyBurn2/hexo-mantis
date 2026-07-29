"""BootstrapTrainer — the corpus→net pretraining loop (WP10 §a.7 PORT of `pretrain_trainer.py`).

Behaviour-exact relocation of the Phase-4.0 pretrain loop, rehomed onto the new-repo seams:
  * losses via ``mantis.train.losses`` (the Slice-2 loss family);
  * training-step events via the injected ``EventSink`` (``mantis.train.emit``) — no `monitoring`
    import (DAG-clean);
  * killed-branch gate surfacing DROPPED — the falsified `cluster_pool` / `gpool_bias` global-gate
    scalars (F-04/F-05, O3b) do not exist on the new `build_net` CNN, so their per-step readout is
    severed (not a reachable numeric path);
  * the pretrain artifacts stay in their LEGACY bare/full shape (the T-CK-31/32 read shapes) — the
    envelope-v2 writer is the run5 training-loop concern, not the bootstrap pretrain concern.

The negative-step accounting (`self.step` counts up from `-total_pretrain_steps` toward 0; the
`step_budget` exit uses the delta so it is sign-independent) is preserved verbatim.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.optim as optim

# Canonical stub-exported locations — `torch.amp` itself does not re-export for type checkers.
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

from mantis.train.emit import emit_via
from mantis.train.losses import (
    compute_aux_loss,
    compute_chain_loss,
    compute_policy_loss,
    compute_total_loss,
    compute_value_loss,
    fp16_backward_step,
)

_LOG = logging.getLogger(__name__)


def _base_model(net: Any) -> Any:
    return getattr(net, "_orig_mod", net)


class BootstrapTrainer:
    """Pretraining loop using the Phase-4.0 loss function: FP16 AMP, aux opponent-reply head,
    grad clip to 1.0, policy valid-masking, label smoothing, cosine LR schedule."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: dict,
        device: torch.device,
        checkpoint_dir: Path,
        *,
        arch: Any = None,
        sink: Any = None,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.arch = arch
        self._sink = sink
        self.step = 0

        fp16 = bool(config.get("fp16", True)) and device.type == "cuda"
        self.fp16 = fp16

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=float(config.get("lr", 0.002)),
            weight_decay=float(config.get("weight_decay", 0.0001)),
        )
        self.scaler = GradScaler(device=device.type, enabled=fp16)

        total_steps = int(config.get("pretrain_total_steps", 50_000))
        eta_min = float(config.get("pretrain_eta_min", 1e-5))
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, total_steps), eta_min=eta_min,
        )

    def train_epoch(
        self,
        loader: torch.utils.data.DataLoader,
        label_smoothing: float = 0.05,
        aux_weight: float = 0.15,
        chain_weight: float = 0.0,
        step_budget: int | None = None,
        start_step: int | None = None,
        log_interval: int = 50,
    ) -> dict[str, float]:
        """One full pass over the dataloader (yields (states, chain_planes, policies, outcomes)).

        Returns a dict with loss / policy_loss / value_loss / opp_reply_loss / chain_loss. The
        `step_budget` exit uses the `self.step - budget_origin` delta so it is sign-independent
        (the negative-step convention). See the module docstring.
        """
        budget_origin = start_step if start_step is not None else self.step
        self.model.train()
        total: dict[str, float] = {
            "loss": 0.0, "policy_loss": 0.0,
            "value_loss": 0.0, "opp_reply_loss": 0.0,
            "chain_loss": 0.0,
        }
        n_batches = 0

        for batch in loader:
            states, chain_planes, policies, outcomes = batch
            states = states.to(self.device)
            chain_planes = chain_planes.to(self.device)
            policies = policies.to(self.device)
            outcomes = outcomes.to(self.device)

            if label_smoothing > 0.0:
                n_actions = policies.shape[1]
                policies = policies * (1.0 - label_smoothing) + label_smoothing / n_actions

            self.optimizer.zero_grad()

            use_chain = chain_weight > 0.0
            with autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.fp16,
            ):
                fwd = self.model(states, aux=True, chain=use_chain)
                log_policy, _value, v_logit, opp_reply = fwd[0], fwd[1], fwd[2], fwd[3]
                chain_pred = fwd[4] if use_chain else None

                policy_valid = policies.sum(dim=1) > 1e-6
                policy_loss = compute_policy_loss(log_policy, policies, policy_valid, self.device)
                value_loss = compute_value_loss(v_logit, outcomes)
                opp_reply_loss = compute_aux_loss(opp_reply, policies, policy_valid, self.device)

                chain_loss = None
                if use_chain and chain_pred is not None:
                    chain_loss = compute_chain_loss(chain_pred, chain_planes.float())

                loss = compute_total_loss(
                    policy_loss,
                    value_loss,
                    opp_reply_loss,
                    aux_weight,
                    chain_loss=chain_loss,
                    chain_weight=chain_weight,
                )

            # Defense-in-depth: skip backward+step on a non-finite forward loss (fp16 overflow
            # cascade guard — a NaN clip_coef would otherwise write NaN to every weight).
            if not torch.isfinite(loss):
                self.skipped_nonfinite_steps = getattr(self, "skipped_nonfinite_steps", 0) + 1
                if self.skipped_nonfinite_steps <= 5 or self.skipped_nonfinite_steps % 50 == 0:
                    _LOG.warning(
                        "skipped_nonfinite_loss step=%s n_skipped=%s loss=%s",
                        self.step + 1, self.skipped_nonfinite_steps, float(loss.detach().item()),
                    )
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.step += 1
                continue

            grad_norm = fp16_backward_step(loss, self.optimizer, self.scaler, self.model, self.fp16)

            self.scheduler.step()
            self.step += 1

            step_loss = loss.item()
            total["loss"] += step_loss
            total["policy_loss"] += policy_loss.item()
            total["value_loss"] += value_loss.item()
            total["opp_reply_loss"] += opp_reply_loss.item()
            if chain_loss is not None:
                total["chain_loss"] += chain_loss.item()
            n_batches += 1

            if log_interval > 0 and self.step % log_interval == 0:
                policy_entropy = -torch.sum(torch.exp(log_policy) * log_policy, dim=1).mean().item()
                value_accuracy = (
                    torch.sign(v_logit.squeeze()) == torch.sign(outcomes)
                ).float().mean().item()
                lr = float(self.optimizer.param_groups[0]["lr"])
                emit_via(self._sink, {
                    "event": "training_step",
                    "step": self.step,
                    "loss_total": float(step_loss),
                    "loss_policy": float(policy_loss.item()),
                    "loss_value": float(value_loss.item()),
                    "loss_aux": float(opp_reply_loss.item()),
                    "loss_chain": float(chain_loss.item()) if chain_loss is not None else 0.0,
                    "policy_entropy": policy_entropy,
                    "value_accuracy": value_accuracy,
                    "lr": lr,
                    "grad_norm": grad_norm,
                    "corpus_mix": {"pretrain": 1.0, "self_play": 0.0},
                    "phase": "pretrain",
                })
                _LOG.info(
                    "train_step step=%s phase=pretrain loss=%.4f policy=%.4f value=%.4f "
                    "aux=%.4f lr=%s grad_norm=%.4f",
                    self.step, step_loss, policy_loss.item(), value_loss.item(),
                    opp_reply_loss.item(), lr, grad_norm,
                )

            if step_budget is not None and (self.step - budget_origin) >= step_budget:
                break

        n = max(n_batches, 1)
        return {k: v / n for k, v in total.items()}

    def save_checkpoint(self, inf_out: Path | None = None) -> Path:
        """Save a full pretrain checkpoint (LEGACY shape — the T-CK-31 full-v1 read shape) plus a
        bare inference-weights file (the T-CK-32 bare-anchor shape). The bootstrap pretrain
        artifact is a legacy anchor by nature; the run5 envelope-v2 writer is a distinct concern."""
        step = self.step
        ckpt_path = self.checkpoint_dir / f"pretrain_{abs(step):08d}.pt"
        base = _base_model(self.model)
        encoding = self.config.get("encoding")
        enc_name = encoding if isinstance(encoding, str) else None
        payload = {
            "step": int(step),
            "model_state": base.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "config": dict(self.config),
            "metadata": {"encoding_name": enc_name} if enc_name is not None else {},
        }
        torch.save(payload, ckpt_path)
        inf_path = Path(inf_out) if inf_out is not None else Path("checkpoints") / "bootstrap_model.pt"
        inf_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(base.state_dict(), inf_path)
        _LOG.info("checkpoint_saved path=%s inference=%s", str(ckpt_path), str(inf_path))
        return ckpt_path
