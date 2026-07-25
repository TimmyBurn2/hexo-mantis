"""The Trainer — one gradient step + envelope-v2 checkpoint IO (WP10 §a.4/§c.7 PORT).

>300 justify: the `Trainer` owns the full per-step training surface (optimizer / scaler /
scheduler / EMA lifecycle, the dense-CNN step, the graph-GNN step, and the checkpoint
save/load delegates) — ONE cohesive responsibility kept in one file so the training-step
numeric contract is greppable. Behaviour-exact relocation of `hexo_rl/training/trainer.py`,
routed through the new-repo seams, with the ratified WP10 amendments + KILL severances:

  * `save_checkpoint` writes envelope v2 via `checkpoints.save_checkpoint(kind="full", ...)`
    → `{run_id}_{step:08d}_{sha8}.ckpt` (the old `checkpoint_{step}.pt`/`inference_only.pt`/
    EMA-sidecar names are replaced); `load_checkpoint` delegates to `checkpoints.resume_trainer`
    (§c.7). No shape-inference — arch travels on `self.arch` (a WP9 declared dataclass).
  * autocast dtype is REPRESENTATION-AWARE off the DECLARED arch (`self.arch.representation`
    → `amp_dtype_for`), NOT a `model_representation(module)` sniff (WP9-deleted, §c.4). The
    graph path is bf16-pinned (LAW-06).
  * DEFINITE-KILL branches SEVERED (never re-enter): `per_class_target_temperature` (F-02/F-28),
    `track_b_grad_attribution`/`track_b_buffer_snapshot` (F-22..F-33).
  * The display/metrics half (per-source entropy split, perf probes, value-spread canary,
    structlog train_step lines) DEFERS→WP13 — training-side events route through the injected
    `EventSink`; the reachable NUMERIC path (forward/loss/backward/optim/scheduler/EMA/step) is
    behaviour-exact.

Training hyperparameters ARE first-class `TrainConfig` schema fields (R-TRAINCONFIG-SCHEMA
closure, WPSC Phase 2 SC-A1): the minted `configs/*.yaml` `train:` section is the sole default
authority (R1). `TrainHParams` is now the RUNTIME dataclass a resolver (`from_config`) builds
from the validated `RunConfig.train` section — it is no longer an independent default
authority; every field is a REQUIRED constructor argument (no `=` default on the dataclass).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

from mantis.encoding import resolve_from_config
from mantis.model import (
    ModelArch,
    amp_dtype_for,
    arch_from_spec_and_config,
    binned_value_loss as _binned_value_loss,
)
from mantis.train import checkpoints
from mantis.train.emit import emit_via
from mantis.train.losses import (
    chain_loss_with_fire_rate,
    compute_aux_loss,
    compute_kl_policy_loss,
    compute_policy_loss,
    compute_ply_index_loss,
    compute_total_loss,
    compute_uncertainty_loss,
    compute_value_loss,
    fp16_backward_step,
    ragged_policy_ce,
)

_LOG = logging.getLogger(__name__)

# Graph configs must zero every aux/entropy weight — GnnNet ships policy + dist65 value only
# (no ownership/threat/chain/opp-reply/uncertainty/ply-index heads, no entropy regularizer).
GRAPH_FORBIDDEN_NONZERO_WEIGHTS: tuple[str, ...] = (
    "aux_opp_reply_weight", "uncertainty_weight", "ownership_weight",
    "threat_weight", "aux_chain_weight", "ply_index_weight", "entropy_reg_weight",
)


def build_param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Split params for AdamW weight decay: 2D+ weights decay, 1D params / biases don't
    (nanoGPT / KataGo precedent; the no-decay group counters late-training plasticity loss).
    Yields exactly TWO param groups (pinned by T-CK-19 `param_groups==2`)."""
    decay: list[torch.Tensor] = []
    no_decay: list[torch.Tensor] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


@dataclass(frozen=True)
class TrainHParams:
    """Training hyperparameters — the RUNTIME object a `Trainer` (and every test bypassing
    `.from_config`) constructs and passes around. R-TRAINCONFIG-SCHEMA closure (WPSC Phase 2
    SC-A1): every field is a REQUIRED constructor argument — NO Python dataclass default.
    `TrainConfig` (the schema, `mantis.config.schema.train`) is the sole default authority
    (R1); a dataclass default here would be a second, independently-editable authority for
    the same knob, exactly what R1 forbids. `.from_config` builds this from a validated
    `RunConfig.train` section (`config["train"]`), never from flat legacy keys.
    """

    lr: float
    weight_decay: float
    grad_clip: float
    fp16: bool
    lr_schedule: str
    total_steps: int
    scheduler_t_max: Optional[int]
    eta_min: float
    min_lr: Optional[float]
    checkpoint_interval: int
    completed_q_values: bool
    policy_prune_frac: float
    entropy_reg_weight: float
    aux_opp_reply_weight: float
    uncertainty_weight: float
    ownership_weight: float
    threat_weight: float
    aux_chain_weight: float
    ply_index_weight: float
    threat_pos_weight: float
    value_target: str
    policy_target: str
    draw_reward: float
    ply_cap_value: float

    @classmethod
    def from_config(cls, config: Any) -> "TrainHParams":
        """Build hparams from a validated `RunConfig`-shaped mapping's `train` section. No
        flat-key fallback: `config['train']` (a `TrainConfig.model_dump()`-shaped, no-terminal-
        default dict — every field present) is REQUIRED. `value_target`/`policy_target`'s
        single-variant Literals are asserted here (V-NOOP-eligible reads, T-D/T-B/R34) and
        `policy_target` is cross-validated against `completed_q_values` on both sides."""
        cfg = config if isinstance(config, dict) else {}
        train = cfg.get("train")
        if not isinstance(train, dict):
            raise ValueError(
                "TrainHParams.from_config: config['train'] is required (R-TRAINCONFIG-SCHEMA "
                "closure) — no flat legacy training keys are read anymore."
            )
        if train["value_target"] != "pure_outcome_z":
            raise ValueError(f"train.value_target: unsupported {train['value_target']!r}")
        _assert_policy_target_consistency(train, cfg.get("selfplay") or {})
        fields = {f for f in cls.__dataclass_fields__}
        kwargs = {k: train[k] for k in fields}
        return cls(**kwargs)


def _assert_policy_target_consistency(train: Dict[str, Any], selfplay: Dict[str, Any]) -> None:
    """T-B/R34 cross-check: `train.policy_target` must agree with `train.completed_q_values`
    and (once `selfplay` carries the field — SC-A2) `selfplay.completed_q_values`. One
    decision, not two independently-editable knobs (ADJUDICATION_QUEUE closing note). The
    `RunConfig`-level `model_validator` (schema/core.py) enforces this at schema-validate
    time; this is the defensive runtime assertion at the actual `.from_config` consumer for
    a caller that hands `TrainHParams.from_config` a dict never routed through
    `RunConfig.model_validate`."""
    raw = train["policy_target"] == "raw_visit_distribution"
    train_off = not train["completed_q_values"]
    if raw != train_off:
        raise ValueError(
            "train.policy_target disagrees with train.completed_q_values — "
            f"policy_target={train['policy_target']!r}, "
            f"completed_q_values={train['completed_q_values']!r}."
        )
    if "completed_q_values" in selfplay:
        selfplay_off = not selfplay["completed_q_values"]
        if raw != selfplay_off:
            raise ValueError(
                "train.policy_target disagrees with selfplay.completed_q_values — "
                f"policy_target={train['policy_target']!r}, "
                f"selfplay.completed_q_values={selfplay['completed_q_values']!r}."
            )


class Trainer:
    """Manages one training step and checkpoint IO.

    Args:
        model:  a net built by `mantis.model.build_net(arch)` (HexTacToeNet / GnnNet).
        config: the WP8 RunConfig snapshot (nested; schema-validated on checkpoint write) OR a
                legacy flat config (resume).
        arch:   the declared arch dataclass (the SOLE arch source at save). Derived from
                `config` when omitted.
        checkpoint_dir / device: as named.
        train_hparams: explicit `TrainHParams`; derived from `config` when omitted.
        sink:   the injected `EventSink` (training-side events + the O-CHAIN fire-rate report).
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        *,
        arch: Optional[ModelArch] = None,
        checkpoint_dir: str | Path = "checkpoints",
        device: Optional[torch.device] = None,
        train_hparams: Optional[TrainHParams] = None,
        sink: Any = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.config = config
        self._sink = sink
        self.arch: ModelArch = arch if arch is not None else self._derive_arch(config)
        self.hp = train_hparams if train_hparams is not None else TrainHParams.from_config(config)

        # autocast dtype off the DECLARED arch representation (§c.4; no module sniff).
        representation = getattr(self.arch, "representation", "grid")
        self.amp_dtype = amp_dtype_for(representation, config if isinstance(config, dict) else {})

        # fp16 is CUDA-only (matches old: disabled on CPU); bf16 needs no scaler.
        fp16_requested = bool(self.hp.fp16)
        if fp16_requested and self.device.type != "cuda":
            fp16_requested = False
        self.fp16 = fp16_requested
        self._scaler_enabled = self.fp16 and self.amp_dtype == torch.float16
        # autocast is enabled on the fp16 grid path (CUDA) AND on the bf16 graph path (LAW-06,
        # incl. CPU — the graph regime is bf16-pinned, not fp16-gated).
        self._autocast_enabled = self._scaler_enabled or (self.amp_dtype == torch.bfloat16)

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = optim.AdamW(
            build_param_groups(self.model, float(self.hp.weight_decay)),
            lr=float(self.hp.lr),
        )
        self.scheduler = self._build_scheduler()

        from mantis.train.ema import build_ema_model, resolve_ema_config
        _ema_enabled, _ema_decay, self.ema_update_every = resolve_ema_config(
            config if isinstance(config, dict) else {}
        )
        self.ema_model = (
            build_ema_model(getattr(self.model, "_orig_mod", self.model), decay=_ema_decay)
            if _ema_enabled else None
        )

        self.scaler = GradScaler(device=self.device.type, enabled=self._scaler_enabled)

        self.step = 0
        # CONFRES F1(A) back-prop: keys the resume F1 defer preserved (empty on a fresh run).
        self.f1_deferred_keys: "frozenset[str]" = frozenset()
        self.loaded_from_full_checkpoint = False
        self.ckpt_had_value_fc2_bins = False

        _pos_w = float(self.hp.threat_pos_weight)
        self._threat_pos_weight: Optional[torch.Tensor] = (
            torch.tensor(_pos_w, dtype=torch.float32, device=self.device) if _pos_w != 1.0 else None
        )

    # ── construction helpers ────────────────────────────────────────────────────────────
    @staticmethod
    def _derive_arch(config: Any) -> ModelArch:
        cfg = dict(config) if isinstance(config, dict) else {}
        return arch_from_spec_and_config(_resolve_spec(cfg), cfg)

    def _build_scheduler(self):
        schedule = str(self.hp.lr_schedule or "none").lower()
        if schedule in {"none", "off", "disabled"}:
            return None
        if schedule == "cosine":
            t_max = self.hp.scheduler_t_max if self.hp.scheduler_t_max is not None else self.hp.total_steps
            if t_max is None:
                raise ValueError("lr_schedule: cosine requires total_steps / scheduler_t_max.")
            min_lr = self.hp.eta_min if self.hp.eta_min is not None else self.hp.min_lr
            if min_lr is None:
                raise ValueError("lr_schedule: cosine requires eta_min / min_lr.")
            return CosineAnnealingLR(self.optimizer, T_max=max(1, int(t_max)),
                                     eta_min=float(min_lr), last_epoch=-1)
        raise ValueError(f"Unsupported lr_schedule: {schedule}")

    def _base_model(self) -> nn.Module:
        return getattr(self.model, "_orig_mod", self.model)

    # ── dense (CNN) training step ─────────────────────────────────────────────────────────
    def train_step_from_tensors(
        self,
        states: np.ndarray,
        policies: np.ndarray,
        outcomes: np.ndarray,
        chain_planes: Optional[Any] = None,
        ownership_targets: Optional[Any] = None,
        threat_targets: Optional[Any] = None,
        is_full_search: Optional[Any] = None,
        n_pretrain: int = 0,
        n_recent: int = 0,
        position_indices: Optional[Any] = None,
        value_target_valid: Optional[Any] = None,
    ) -> Dict[str, float]:
        """One gradient update from pre-built numpy arrays (dense grid path)."""
        return self._train_on_batch(
            states, policies, outcomes,
            chain_planes=chain_planes, ownership_targets=ownership_targets,
            threat_targets=threat_targets, is_full_search=is_full_search,
            n_pretrain=n_pretrain, n_recent=n_recent,
            position_indices=position_indices, value_target_valid=value_target_valid,
        )

    def _train_on_batch(
        self,
        states: np.ndarray,
        policies: np.ndarray,
        outcomes: np.ndarray,
        chain_planes: Optional[Any] = None,
        ownership_targets: Optional[Any] = None,
        threat_targets: Optional[Any] = None,
        is_full_search: Optional[Any] = None,
        n_pretrain: int = 0,
        n_recent: int = 0,
        position_indices: Optional[Any] = None,
        value_target_valid: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Core dense step: forward, loss, backward, optimizer step (behaviour-exact; the
        KILLED per-class-temperature + track_b branches are SEVERED)."""
        from mantis.train.aux_decode import decode_ownership, decode_winning_line, mask_aux_rows

        hp = self.hp
        aux_weight = float(hp.aux_opp_reply_weight)
        uncertainty_weight = float(hp.uncertainty_weight)
        ownership_weight = float(hp.ownership_weight)
        threat_weight = float(hp.threat_weight)
        chain_weight = float(hp.aux_chain_weight)
        ply_index_weight = float(hp.ply_index_weight)
        entropy_weight = float(hp.entropy_reg_weight)

        states_t = torch.from_numpy(states).to(self.device)
        if not self.fp16:
            states_t = states_t.float()
        policies_t = torch.from_numpy(policies).to(self.device)
        outcomes_t = torch.from_numpy(outcomes).to(self.device)
        full_search_mask_t: Optional[torch.Tensor] = None
        if is_full_search is not None:
            full_search_mask_t = torch.from_numpy(
                np.asarray(is_full_search, dtype=np.uint8)).to(self.device).bool()
        value_mask_t: Optional[torch.Tensor] = None
        if value_target_valid is not None:
            value_mask_t = torch.from_numpy(
                np.asarray(value_target_valid, dtype=np.uint8)).to(self.device).bool()

        prune_frac = float(hp.policy_prune_frac)
        if prune_frac > 0.0:
            policies_t = _prune_policy_targets(policies_t, prune_frac)

        self.optimizer.zero_grad()

        batch_n = int(states.shape[0])
        assert 0 <= n_pretrain <= batch_n, f"n_pretrain={n_pretrain} out of [0, {batch_n}]"
        use_ownership = ownership_weight > 0.0 and ownership_targets is not None
        use_threat = threat_weight > 0.0 and threat_targets is not None
        own_t = decode_ownership(ownership_targets, self.device) if use_ownership else None
        thr_t = decode_winning_line(threat_targets, self.device) if use_threat else None

        with autocast(device_type=self.device.type, dtype=self.amp_dtype,
                      enabled=self._autocast_enabled):
            use_aux = aux_weight > 0.0
            use_uncertainty = uncertainty_weight > 0.0
            use_chain = chain_weight > 0.0
            use_ply_index = ply_index_weight > 0.0 and position_indices is not None

            fwd = self.model(states_t, aux=use_aux, uncertainty=use_uncertainty,
                             ownership=use_ownership, threat=use_threat,
                             chain=use_chain, ply_index=use_ply_index)
            log_policy, value, value_aux = fwd[0], fwd[1], fwd[2]
            base = self._base_model()
            is_dist65 = getattr(base, "value_head_type", "scalar") == "dist65"
            v_logit = (torch.atanh(value.detach().clamp(-0.999999, 0.999999))
                       if is_dist65 else value_aux)
            _idx = 3
            opp_reply = fwd[_idx] if use_aux else None
            _idx += 1 if use_aux else 0
            sigma2 = fwd[_idx] if use_uncertainty else None
            _idx += 1 if use_uncertainty else 0
            own_pred = fwd[_idx] if use_ownership else None
            _idx += 1 if use_ownership else 0
            thr_pred = fwd[_idx] if use_threat else None
            _idx += 1 if use_threat else 0
            chain_pred = fwd[_idx] if use_chain else None
            _idx += 1 if use_chain else 0
            ply_pred = fwd[_idx] if use_ply_index else None

            policy_valid = policies_t.sum(dim=1) > 1e-6
            if bool(hp.completed_q_values):
                policy_loss = compute_kl_policy_loss(log_policy, policies_t, policy_valid,
                                                     self.device, full_search_mask=full_search_mask_t)
            else:
                policy_loss = compute_policy_loss(log_policy, policies_t, policy_valid,
                                                  self.device, full_search_mask=full_search_mask_t)
            if is_dist65:
                value_loss = _binned_value_loss(value_aux, outcomes_t, value_mask=value_mask_t)
            else:
                value_loss = compute_value_loss(value_aux, outcomes_t, value_mask=value_mask_t)

            opp_reply_loss = (
                compute_aux_loss(opp_reply, policies_t, policy_valid, self.device,
                                 full_search_mask=full_search_mask_t) if use_aux else None
            )
            entropy_bonus = None
            if entropy_weight > 0.0:
                p_fp32 = torch.exp(log_policy.float())
                entropy_bonus = torch.special.entr(p_fp32).sum(dim=-1).mean()
            unc_loss = (
                compute_uncertainty_loss(sigma2, outcomes_t, value.detach())
                if use_uncertainty else None
            )
            aux_skip_full_pretrain = n_pretrain >= batch_n
            own_loss = None
            if use_ownership and own_pred is not None and own_t is not None and not aux_skip_full_pretrain:
                own_pred_m, own_t_m = mask_aux_rows(own_pred, own_t, n_pretrain)
                own_loss = nn.functional.mse_loss(own_pred_m.squeeze(1), own_t_m)
            thr_loss = None
            if use_threat and thr_pred is not None and thr_t is not None and not aux_skip_full_pretrain:
                thr_pred_m, thr_t_m = mask_aux_rows(thr_pred, thr_t, n_pretrain)
                thr_loss = nn.functional.binary_cross_entropy_with_logits(
                    thr_pred_m.squeeze(1), thr_t_m, pos_weight=self._threat_pos_weight)
            # Q13-aux chain loss WITH the in-run fire-rate self-report (O-CHAIN, LAW-07/18).
            chain_loss = None
            if chain_pred is not None and chain_planes is not None:
                chain_target = torch.from_numpy(np.asarray(chain_planes)).to(self.device).float()
                chain_loss = chain_loss_with_fire_rate(
                    chain_pred, chain_target, chain_weight, sink=self._sink, step=self.step + 1)
            ply_index_loss = None
            if use_ply_index and ply_pred is not None and position_indices is not None:
                pos_idx_t = torch.from_numpy(np.asarray(position_indices)).to(self.device)
                ply_index_loss = compute_ply_index_loss(ply_pred, pos_idx_t)

            loss = compute_total_loss(
                policy_loss, value_loss, opp_reply_loss, aux_weight,
                entropy_bonus, entropy_weight, unc_loss, uncertainty_weight,
                own_loss, ownership_weight, thr_loss, threat_weight,
                chain_loss, chain_weight, ply_index_loss, ply_index_weight,
            )

        if not torch.isfinite(loss):
            self.optimizer.zero_grad()
            if (self.fp16 and self.scaler.is_enabled()
                    and getattr(self.scaler, "_scale", None) is not None):
                decayed = float(self.scaler.get_scale() * self.scaler.get_backoff_factor())
                self.scaler.update(new_scale=decayed)
            return {"loss": float("nan"), "policy_loss": float("nan"),
                    "value_loss": float("nan"), "grad_norm": float("nan"),
                    "lr": self.optimizer.param_groups[0]["lr"], "value_accuracy": float("nan")}

        grad_norm = fp16_backward_step(loss, self.optimizer, self.scaler, self.model,
                                       self._scaler_enabled, max_grad_norm=float(hp.grad_clip))
        self.step += 1
        if self.scheduler is not None and math.isfinite(grad_norm):
            self.scheduler.step()
        if (self.ema_model is not None and math.isfinite(grad_norm)
                and self.step % self.ema_update_every == 0):
            self.ema_model.update_parameters(self._base_model())

        with torch.no_grad():
            pred_win = (v_logit.squeeze(1) > 0).float()
            value_accuracy = (pred_win == (outcomes_t > 0).float()).float().mean().item()
        lr = self.optimizer.param_groups[0]["lr"]

        result: Dict[str, float] = {
            "loss": loss.item(), "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(), "grad_norm": grad_norm,
            "value_accuracy": value_accuracy, "lr": lr,
        }
        if chain_loss is not None:
            result["chain_loss"] = chain_loss.item()
        emit_via(self._sink, {"event": "training_step", "step": self.step,
                              "representation": "grid", **result})

        interval = int(hp.checkpoint_interval)
        if interval > 0 and self.step % interval == 0:
            self.save_checkpoint(result)
        return result

    # ── graph (GNN) training step — the numeric core, bench + injected-buffer driver ──────
    def train_step_from_graph_batch(
        self,
        *,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        legal_mask: torch.Tensor,
        stone_mask: torch.Tensor,
        node_offsets: torch.Tensor,
        legal_offsets: torch.Tensor,
        policy_target: torch.Tensor,
        outcomes: torch.Tensor,
        value_valid: Optional[torch.Tensor] = None,
        is_full_search: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """One gradient update from an ALREADY-collated graph batch (block-diagonal tensors).

        The numeric core of the old `_train_on_batch`-graph sibling — factored so the bench
        (fixed synthetic batch) and any injected self-play graph buffer drive the SAME
        forward/loss/backward path. bf16 autocast (LAW-06). GnnNet ships policy + dist65 value
        only, so there is no aux/entropy branch (standing §6.3)."""
        for key in GRAPH_FORBIDDEN_NONZERO_WEIGHTS:
            if float(getattr(self.hp, key, 0.0)) != 0.0:
                raise ValueError(
                    f"train_step_from_graph_batch: {key} is nonzero on a graph run — GnnNet "
                    "ships policy + dist65 value only (no aux heads / entropy). Zero every "
                    "GRAPH_FORBIDDEN_NONZERO_WEIGHTS key (standing §6.3).")
        self.optimizer.zero_grad()
        with autocast(device_type=self.device.type, dtype=self.amp_dtype,
                      enabled=self._autocast_enabled):
            policy_logits, _value, bin_logits = self.model.forward_batch(
                x, edge_index, edge_attr, legal_mask, stone_mask, node_offsets=node_offsets)
            policy_loss = ragged_policy_ce(policy_logits, policy_target, legal_offsets,
                                           full_search_mask=is_full_search)
            value_loss = _binned_value_loss(bin_logits, outcomes, value_mask=value_valid)
            loss = policy_loss + value_loss

        grad_norm = fp16_backward_step(loss, self.optimizer, self.scaler, self.model,
                                       self._scaler_enabled, max_grad_norm=float(self.hp.grad_clip))
        self.step += 1
        if self.scheduler is not None and math.isfinite(grad_norm):
            self.scheduler.step()
        if (self.ema_model is not None and math.isfinite(grad_norm)
                and self.step % self.ema_update_every == 0):
            self.ema_model.update_parameters(self._base_model())
        lr = self.optimizer.param_groups[0]["lr"]
        result = {"loss": loss.item(), "policy_loss": policy_loss.item(),
                  "value_loss": value_loss.item(), "grad_norm": grad_norm, "lr": lr}
        emit_via(self._sink, {"event": "training_step", "step": self.step,
                              "representation": "graph", **result})
        return result

    # ── checkpoint IO ─────────────────────────────────────────────────────────────────────
    def inference_state_dict(self) -> Dict[str, torch.Tensor]:
        """The state_dict self-play / eval / promotion consume (EMA weights when EMA is on)."""
        if self.ema_model is not None:
            return self.ema_model.state_dict()
        return self._base_model().state_dict()

    def _resolve_encoding_name(self) -> Optional[str]:
        try:
            return _resolve_spec(dict(self.config)).name
        except Exception as exc:  # noqa: BLE001 — surfaced, but a resolvable config is required
            _LOG.error("checkpoint_encoding_resolve_failed error=%s", exc)
            return None

    def save_checkpoint(self, loss_info: Optional[Dict[str, float]] = None) -> Path:
        """Write an envelope-v2 FULL checkpoint via the ONE writer (§c.7): filename
        `{run_id}_{step:08d}_{sha8}.ckpt`, immutable stamp, config schema-validated on write.
        `encoding_name` from the registry resolver, `arch` from `self.arch`."""
        cfg = self.config if isinstance(self.config, dict) else {}
        metadata_kwargs = {
            "encoding_name": self._resolve_encoding_name(),
            "run_id": cfg.get("run_id"),
            "arch": self.arch,
            "corpus_sha256": cfg.get("corpus_sha256"),
        }
        return checkpoints.save_checkpoint(
            model=self.model, optimizer=self.optimizer, scaler=self.scaler,
            scheduler=self.scheduler, step=self.step, config=cfg,
            metadata_kwargs=metadata_kwargs, checkpoint_dir=self.checkpoint_dir, kind="full",
        )

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        checkpoint_dir: Optional[str | Path] = None,
        device: Optional[torch.device] = None,
        fallback_config: Optional[Dict[str, Any]] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        declared_keys: Optional["frozenset | set"] = None,
        sink: Any = None,
    ) -> "Trainer":
        """Restore a Trainer — thin delegate to `checkpoints.resume_trainer` (§c.7)."""
        return checkpoints.resume_trainer(
            cls, checkpoint_path, fallback_config=fallback_config,
            config_overrides=config_overrides, declared_keys=declared_keys,
            sink=sink, device=device,
        )


def _resolve_spec(config: Any):
    """Resolve the encoding spec from a config, bridging the WP8 NESTED `identity.encoding`
    shape and the legacy FLAT `encoding` shape. `resolve_from_config` only reads the flat key
    (defaulting to v6), so a WP8 config's declared encoding must be lifted from `identity` first
    — this keeps `metadata.encoding_name` consistent with `config.identity.encoding` (the loader's
    stamp-source check, T-CK-30)."""
    cfg = dict(config) if isinstance(config, dict) else {}
    ident = cfg.get("identity")
    if isinstance(ident, dict) and isinstance(ident.get("encoding"), str) and "encoding" not in cfg:
        from mantis.encoding import lookup
        return lookup(ident["encoding"])
    return resolve_from_config(cfg)


def _prune_policy_targets(pi: torch.Tensor, threshold_frac: float) -> torch.Tensor:
    """Zero policy-target entries at/below `threshold_frac · max(row)`, renormalize
    (behaviour-exact with the old `prune_policy_targets`; sharpens MCTS visit targets)."""
    if threshold_frac <= 0.0:
        return pi
    max_vals = pi.max(dim=-1, keepdim=True).values
    mask = pi > (threshold_frac * max_vals)
    pruned = pi * mask
    return pruned / pruned.sum(dim=-1, keepdim=True).clamp(min=1e-8)
