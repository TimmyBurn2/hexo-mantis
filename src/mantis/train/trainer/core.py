"""The Trainer — one gradient step + envelope-v2 checkpoint IO (WP10 §a.4/§c.7 PORT).

>300 justify: the `Trainer` owns the full per-step training surface (optimizer / scaler /
scheduler / EMA lifecycle, the dense-CNN step, the graph-GNN step, the periodic-checkpoint
seam `_maybe_periodic_checkpoint` — THE one reader of `train.checkpoint_interval`, which
both step tails call (R173) — and the checkpoint save/load delegates) — ONE cohesive
responsibility kept in one file so the training-step numeric contract is greppable.
Behaviour-exact relocation of `hexo_rl/training/trainer.py`,
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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Canonical stub-exported locations — `torch.amp` itself does not re-export for type checkers.
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

from mantis.encoding import resolve_from_config
from mantis.model import (
    ModelArch,
    amp_dtype_for,
    arch_from_spec_and_config,
)
from mantis.model import (
    binned_value_loss as _binned_value_loss,
)
from mantis.selfplay.graph_wire_split import GraphEmptyBatchError
from mantis.train import checkpoints
from mantis.train.emit import emit_via
from mantis.train.losses import (
    backward_accumulate,
    chain_loss_with_fire_rate,
    clip_and_step,
    compute_aux_loss,
    compute_kl_policy_loss,
    compute_ply_index_loss,
    compute_policy_loss,
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
    scheduler_t_max: int | None
    eta_min: float
    min_lr: float | None
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
    def from_config(cls, config: Any) -> TrainHParams:
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


def _assert_policy_target_consistency(train: dict[str, Any], selfplay: dict[str, Any]) -> None:
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
        config: dict[str, Any],
        *,
        arch: ModelArch | None = None,
        checkpoint_dir: str | Path = "checkpoints",
        device: torch.device | None = None,
        train_hparams: TrainHParams | None = None,
        sink: Any = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.config = config
        self._sink = sink
        self.arch: ModelArch = arch if arch is not None else self._derive_arch(config)
        self.hp = train_hparams if train_hparams is not None else TrainHParams.from_config(config)

        # autocast dtype off the DECLARED arch representation (§c.4; no module sniff).
        # R30b: hard key access, no fallback — config["train"] is already required by
        # TrainHParams.from_config's own no-fallback read (Phase 2 precedent).
        # AUDIT-1 F-35: attribute access, no default (LAW-11 — no dense-by-default).
        representation = self.arch.representation
        self.amp_dtype = amp_dtype_for(representation, config["train"]["amp_dtype"])

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
        # AUDIT-1 F-06: the block is REQUIRED, so it is read from the config as given. The
        # `{}` fallback that stood here made an absent block indistinguishable from a declared
        # OFF — which is exactly how the lever stayed silently disabled.
        _ema_enabled, _ema_decay, self.ema_update_every = resolve_ema_config(config)
        self.ema_model = (
            build_ema_model(getattr(self.model, "_orig_mod", self.model), decay=_ema_decay)
            if _ema_enabled else None
        )

        self.scaler = GradScaler(device=self.device.type, enabled=self._scaler_enabled)

        self.step = 0
        #: Non-finite guard counters (item 6). Both MUST read 0 in a healthy run. Non-zero
        #: means a NaN/inf was produced and suppressed — the F-11 cascade caught early
        #: rather than after it had written NaN into every weight.
        self.nonfinite_loss_microbatches = 0
        self.nonfinite_grad_steps = 0
        # CONFRES F1(A) back-prop: keys the resume F1 defer preserved (empty on a fresh run).
        self.f1_deferred_keys: frozenset[str] = frozenset()
        self.loaded_from_full_checkpoint = False
        self.ckpt_had_value_fc2_bins = False

        _pos_w = float(self.hp.threat_pos_weight)
        self._threat_pos_weight: torch.Tensor | None = (
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
        chain_planes: Any | None = None,
        ownership_targets: Any | None = None,
        threat_targets: Any | None = None,
        is_full_search: Any | None = None,
        n_pretrain: int = 0,
        n_recent: int = 0,
        position_indices: Any | None = None,
        value_target_valid: Any | None = None,
    ) -> dict[str, float]:
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
        chain_planes: Any | None = None,
        ownership_targets: Any | None = None,
        threat_targets: Any | None = None,
        is_full_search: Any | None = None,
        n_pretrain: int = 0,
        n_recent: int = 0,
        position_indices: Any | None = None,
        value_target_valid: Any | None = None,
    ) -> dict[str, float]:
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
        full_search_mask_t: torch.Tensor | None = None
        if is_full_search is not None:
            full_search_mask_t = torch.from_numpy(
                np.asarray(is_full_search, dtype=np.uint8)).to(self.device).bool()
        value_mask_t: torch.Tensor | None = None
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
        # The redundant `is not None` restates the use_* definitions above so the
        # None-exclusion is visible to the type checker.
        own_t = (decode_ownership(ownership_targets, self.device)
                 if use_ownership and ownership_targets is not None else None)
        thr_t = (decode_winning_line(threat_targets, self.device)
                 if use_threat and threat_targets is not None else None)

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

            opp_reply_loss = None
            if use_aux:
                # `aux=True` forward contract: fwd carries the opp_reply head output.
                assert opp_reply is not None
                opp_reply_loss = compute_aux_loss(opp_reply, policies_t, policy_valid,
                                                  self.device,
                                                  full_search_mask=full_search_mask_t)
            entropy_bonus = None
            if entropy_weight > 0.0:
                p_fp32 = torch.exp(log_policy.float())
                entropy_bonus = torch.special.entr(p_fp32).sum(dim=-1).mean()
            unc_loss = None
            if use_uncertainty:
                # `uncertainty=True` forward contract: fwd carries the sigma2 head output.
                assert sigma2 is not None
                unc_loss = compute_uncertainty_loss(sigma2, outcomes_t, value.detach())
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

        result: dict[str, float] = {
            "loss": loss.item(), "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(), "grad_norm": grad_norm,
            "value_accuracy": value_accuracy, "lr": lr,
        }
        if chain_loss is not None:
            result["chain_loss"] = chain_loss.item()
        # `trainer_step`, NOT `training_step`: the coordinator's `log_interval`-gated
        # narration owns the `training_step` literal (train/events.py, the manifest's one
        # documented shape). This is the trainer's OWN per-step diagnostic row — delivered
        # in production since F-R-P2B-2's sink threading — and one literal = one shape, so
        # it emits under its own name (F-P4 review blocker: a second, non-conforming,
        # ungated producer of the canonical literal at ~log_interval:1 volume).
        emit_via(self._sink, {"event": "trainer_step", "step": self.step,
                              "representation": "grid", **result})

        self._maybe_periodic_checkpoint(result)
        return result

    # ── graph (GNN) training step — the numeric core, bench + injected-buffer driver ──────
    def eval_step_from_graph_batch(
        self,
        *,
        parts: Sequence[Callable[[], Any]],
        policy_denominator: float,
        value_denominator: float,
        total_edges: int,
        total_nodes: int,
        caps_max_edges: int,
        caps_max_nodes: int,
    ) -> dict[str, float]:
        """FORWARD-ONLY loss over a partitioned graph batch — no gradient, no state (R328(d)).

        THE SIBLING OF `train_step_from_graph_batch`, sharing its `parts` contract and its
        denominators so the two numbers are commensurable. What is ABSENT is absent rather
        than skipped: no `zero_grad`, no `backward`, no `clip_and_step`, no `self.step`
        increment, no scheduler step, no EMA update, no `_maybe_periodic_checkpoint`. A held-out
        reading that moved any of those would be a training step wearing an evaluation's name.

        THE MODE IS RESTORED IN A `finally`, and that is load-bearing rather than tidy: an
        evaluation that left the model in `eval()` would change every LATER training step's
        dropout and normalisation behaviour, and the loss curve would look BETTER for it —
        a corruption that reads as an improvement.

        THE RETURN DELIBERATELY OMITS `grad_norm` AND `lr`. `train/coordinator/step.py` reads
        `loss_info.get("grad_norm", 0.0)` and feeds it to `grad_norm_hard_abort`; an eval dict
        carrying that key could be routed there by a later edit and would report a gradient
        that was never computed. Omitting it means such a routing raises rather than silently
        passing a fabricated zero.

        Args:
            parts: zero-arg callables, each materialising one micro-batch (lazily, so only one
                is resident at a time — the same bound the training step relies on).
            policy_denominator: the whole batch's policy denominator.
            value_denominator: the whole batch's value denominator.
            total_edges: edges across the un-split batch, for the caller's records.
            total_nodes: nodes across the un-split batch, for the caller's records.
            caps_max_edges: the resolved micro-batch edge cap.
            caps_max_nodes: the resolved micro-batch node cap.

        Returns:
            `{"loss", "policy_loss", "value_loss"}`, summed over the parts exactly as the
            training step sums them.

        Raises:
            GraphEmptyBatchError: `parts` is empty, so there is nothing to evaluate.
        """
        del total_edges, total_nodes, caps_max_edges, caps_max_nodes  # recorded by the caller
        if len(parts) == 0:
            raise GraphEmptyBatchError(
                "eval_step_from_graph_batch: zero micro-batches — the sampled batch holds no "
                "graphs, so there is no loss to read. Raised rather than returning 0.0, which "
                "a patience stop would read as the best score ever achieved."
            )
        was_training = self.model.training
        loss_total = policy_total = value_total = 0.0
        try:
            self.model.eval()
            with torch.no_grad():
                for make in parts:
                    inputs = make()
                    with autocast(device_type=self.device.type, dtype=self.amp_dtype,
                                  enabled=self._autocast_enabled):
                        policy_logits, _value, bin_logits = self.model.forward_batch(  # pyright: ignore[reportCallIssue]
                            inputs.x, inputs.edge_index, inputs.edge_attr, inputs.legal_index,
                            inputs.stone_mask, node_offsets=inputs.node_offsets)
                        policy_loss = ragged_policy_ce(
                            policy_logits, inputs.policy_target, inputs.legal_offsets,
                            full_search_mask=inputs.is_full_search,
                            denominator=policy_denominator)
                        value_loss = _binned_value_loss(
                            bin_logits, inputs.outcomes, value_mask=inputs.value_valid,
                            denominator=value_denominator)
                        loss = policy_loss + value_loss
                    if torch.isfinite(loss):
                        loss_total += loss.item()
                        policy_total += policy_loss.item()
                        value_total += value_loss.item()
                    del inputs, policy_logits, bin_logits, policy_loss, value_loss, loss
        finally:
            if was_training:
                self.model.train()
        return {"loss": loss_total, "policy_loss": policy_total, "value_loss": value_total}

    def train_step_from_graph_batch(
        self,
        *,
        parts: Sequence[Callable[[], Any]],
        policy_denominator: float,
        value_denominator: float,
        total_edges: int,
        total_nodes: int,
        caps_max_edges: int,
        caps_max_nodes: int,
    ) -> dict[str, float]:
        """One gradient update from a PARTITIONED graph batch (WP12-R F2, CARD-RUN5-GPU-OOM).

        The numeric core of the old `_train_on_batch`-graph sibling. bf16 autocast (LAW-06).
        GnnNet ships policy + dist65 value only, so there is no aux/entropy branch
        (standing §6.3).

        `parts` is a Sequence of ZERO-ARG CALLABLES and that is load-bearing. `Sequence` gives
        `len()` for the LAW-18 counter without consuming anything; the callables make
        materialisation LAZY, which is the whole bound — a `Sequence` of already-collated
        batches would hold every micro-batch resident at once, defeating the cap while passing
        every count-based oracle. The loop materialises one, uses it, and drops the reference
        before the next `make()`.

        ONE OPTIMIZER STEP PER TRAINING STEP. `zero_grad` once before the loop, `backward`
        once per part, `clip_and_step` ONCE after it, one `self.step` increment, one scheduler
        step, one EMA update, one `trainer_step` event and one `_maybe_periodic_checkpoint`
        call (R173's seam is untouched and still fires exactly once per training step).
        Clipping is nonlinear in the whole gradient and `grad_norm` is an armed gate's input,
        so clipping once is a correctness requirement, not a preference.

        SINGLE TAIL: exactly one `return`, the last statement, returning a dict literal with
        exactly `loss`, `policy_loss`, `value_loss`, `grad_norm`, `lr`. There is no early
        return on any path — the degenerate-mask cases return finite-or-`nan` through this
        same tail, and the two failure paths RAISE. That matters because
        `train/coordinator/step.py` reads `loss_info.get("grad_norm", 0.0)`: a path returning
        a dict without the key would silently feed `grad_norm_hard_abort` a `0.0` that always
        passes its threshold.
        """
        for key in GRAPH_FORBIDDEN_NONZERO_WEIGHTS:
            if float(getattr(self.hp, key, 0.0)) != 0.0:
                raise ValueError(
                    f"train_step_from_graph_batch: {key} is nonzero on a graph run — GnnNet "
                    "ships policy + dist65 value only (no aux heads / entropy). Zero every "
                    "GRAPH_FORBIDDEN_NONZERO_WEIGHTS key (standing §6.3).")
        if len(parts) == 0:
            raise GraphEmptyBatchError(
                "train_step_from_graph_batch: zero micro-batches — the sampled batch holds no "
                "graphs, so this step cannot produce a gradient. Raised BEFORE zero_grad, so "
                "no optimizer state moved: a silent no-op here would let the run report a "
                "step it never took (LAW-14)."
            )
        self.optimizer.zero_grad()
        loss_total = 0.0
        policy_total = 0.0
        value_total = 0.0
        for make in parts:
            inputs = make()
            with autocast(device_type=self.device.type, dtype=self.amp_dtype,
                          enabled=self._autocast_enabled):
                # nn.Module.__getattr__ types dynamic attrs as Tensor | Module;
                # `forward_batch` is GnnNet's real method.
                policy_logits, _value, bin_logits = self.model.forward_batch(  # pyright: ignore[reportCallIssue]
                    inputs.x, inputs.edge_index, inputs.edge_attr, inputs.legal_index,
                    inputs.stone_mask, node_offsets=inputs.node_offsets)
                if int(bin_logits.shape[0]) != int(inputs.n_graphs):
                    raise ValueError(
                        f"train_step_from_graph_batch: bin_logits has "
                        f"{int(bin_logits.shape[0])} rows for {int(inputs.n_graphs)} graphs. "
                        "`binned_value_loss` reduces over bin_logits ROWS while the value "
                        "denominator is computed from the per-GRAPH mask, so a mismatch "
                        "would silently make the denominator a second authority over a count "
                        "it does not own.")
                policy_loss = ragged_policy_ce(policy_logits, inputs.policy_target,
                                               inputs.legal_offsets,
                                               full_search_mask=inputs.is_full_search,
                                               denominator=policy_denominator)
                value_loss = _binned_value_loss(bin_logits, inputs.outcomes,
                                                value_mask=inputs.value_valid,
                                                denominator=value_denominator)
                loss = policy_loss + value_loss
            # Non-finite guard on the PRODUCTION graph step (item 6), mirroring the pretrain
            # trainer's (`pretrain/trainer.py`). Without it a single NaN/inf microbatch loss
            # backwards into a NaN clip coefficient, which writes NaN to EVERY weight — the
            # model is destroyed in one step and the run continues reporting numbers. That is
            # falsified row F-11's exact cascade (0×−inf in aux CE → NaN total loss → BN
            # poisoning), and the pretrain path was guarded while the path that trains run5
            # was not.
            #
            # The microbatch is SKIPPED, not zeroed: its gradient contribution is undefined,
            # and dropping it keeps the remaining microbatches' step valid. Counted so the
            # skip is never silent — a run quietly dropping half its microbatches looks
            # exactly like a healthy one on loss alone (LAW-18).
            if not torch.isfinite(loss):
                self.nonfinite_loss_microbatches += 1
                if (self.nonfinite_loss_microbatches <= 5
                        or self.nonfinite_loss_microbatches % 50 == 0):
                    _LOG.warning(
                        "skipped_nonfinite_loss step=%s n_skipped=%s loss=%s",
                        self.step + 1, self.nonfinite_loss_microbatches,
                        float(loss.detach().item()),
                    )
                del inputs, policy_logits, bin_logits, policy_loss, value_loss, loss
                continue
            backward_accumulate(loss, self.scaler, self._scaler_enabled)
            loss_total += loss.item()
            policy_total += policy_loss.item()
            value_total += value_loss.item()
            del inputs, policy_logits, bin_logits, policy_loss, value_loss, loss

        grad_norm = clip_and_step(self.optimizer, self.scaler, self.model,
                                  self._scaler_enabled, float(self.hp.grad_clip))
        self.step += 1
        if self.scheduler is not None and math.isfinite(grad_norm):
            self.scheduler.step()
        if (self.ema_model is not None and math.isfinite(grad_norm)
                and self.step % self.ema_update_every == 0):
            self.ema_model.update_parameters(self._base_model())
        if not math.isfinite(grad_norm):
            # A non-finite grad norm means `clip_and_step` scaled by a NaN/inf coefficient.
            # Counted for the same reason as above, and carried in the payload so the monitor
            # rules can SEE it — before item 6 every NaN was filtered out of the alerts and
            # out of the hard abort, so the one condition that destroys a model outright was
            # the one condition nothing reported.
            self.nonfinite_grad_steps += 1
            _LOG.warning("nonfinite_grad_norm step=%s n=%s grad_norm=%s",
                         self.step, self.nonfinite_grad_steps, grad_norm)
        lr = self.optimizer.param_groups[0]["lr"]
        # `result` stays the FIVE-key loss_info contract (OF2-9: one tail, five keys). The
        # non-finite counters ride the EVENT, not the return: `loss_info` is consumed by the
        # coordinator's gates and by checkpoint metadata, and widening it would change a
        # contract those readers pin — while the event stream is where LAW-18 in-run counters
        # belong anyway.
        result = {"loss": loss_total, "policy_loss": policy_total,
                  "value_loss": value_total, "grad_norm": grad_norm, "lr": lr}
        # `trainer_step`, not `training_step` — same reason as the dense tail's emit above.
        emit_via(self._sink, {"event": "trainer_step", "step": self.step,
                              "representation": "graph", **result,
                              "microbatches": len(parts), "edges": int(total_edges),
                              "nodes": int(total_nodes),
                              "caps_max_edges": int(caps_max_edges),
                              "caps_max_nodes": int(caps_max_nodes),
                              "nonfinite_loss_microbatches": self.nonfinite_loss_microbatches,
                              "nonfinite_grad_steps": self.nonfinite_grad_steps})
        self._maybe_periodic_checkpoint(result)
        return result

    # ── checkpoint IO ─────────────────────────────────────────────────────────────────────
    def inference_state_dict(self) -> dict[str, torch.Tensor]:
        """The state_dict self-play / eval / promotion consume (EMA weights when EMA is on)."""
        if self.ema_model is not None:
            return self.ema_model.state_dict()
        return self._base_model().state_dict()

    def _resolve_encoding_name(self) -> str | None:
        try:
            return _resolve_spec(dict(self.config)).name
        except Exception as exc:  # noqa: BLE001 — surfaced, but a resolvable config is required
            _LOG.error("checkpoint_encoding_resolve_failed error=%s", exc)
            return None

    def _maybe_periodic_checkpoint(self, loss_info: dict[str, float] | None) -> Path | None:
        """THE periodic-checkpoint seam — the ONE reader of `train.checkpoint_interval`.

        R173: the interval read and the periodic write live here and nowhere else, so the
        dense step and the graph step share ONE authority for the cadence rather than each
        carrying its own (`R1`: two independently-editable readers of one key is the
        duplicated-authority class). `0` disables — the value most shipped configs still
        mint (`config/schema/train.py:231`, `ge=0`); `shakedown_20260807.yaml` is the first
        production config to mint a nonzero value. A positive `N` fires at `N, 2N, 3N, …`
        against `self.step`, which is the POST-increment step number in both callers
        (the dense tail and the graph tail), so the boundary is the step whose gradient
        update the artefact contains.

        rule 3 / LAW-12: the write is `self.save_checkpoint`, the SAME entry legs 2 and 3
        call, so the artefact rides the one stamp path — validated config, stamp built once,
        `{run_id}_{step:08d}_{sha8}.ckpt`. This method authors no second write surface.

        LAW-14: a failure is NOT caught. `_write_v2_payload` counts it on
        `checkpoints.persist_errors_total` and re-raises; that counter is the persist-fatal
        watchdog's registered input. A swallow here would be the silent-except LAW-14 bans,
        and would let a run report a cadence it never wrote.

        LAW-18: the event lands AFTER the write and carries the WRITER's returned path —
        `loop.py`'s pre-emit ordering is deliberately NOT mirrored (the argument is
        `coordinator/step.py::_clean_stop_save`'s, and it applies unchanged to a leg that
        fires N times instead of once: a pre-emit puts a false record in the stream on every
        failed write).
        """
        interval = int(self.hp.checkpoint_interval)
        if interval <= 0 or self.step % interval != 0:
            return None
        path = self.save_checkpoint(loss_info)
        emit_via(self._sink, {
            "event": "periodic_checkpoint_save",
            "step": self.step,
            "interval": interval,
            "representation": self.arch.representation,
            "path": None if path is None else str(path),
        })
        return path

    def save_checkpoint(self, loss_info: dict[str, float] | None = None) -> Path:
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
        checkpoint_dir: str | Path | None = None,
        device: torch.device | None = None,
        fallback_config: dict[str, Any] | None = None,
        config_overrides: dict[str, Any] | None = None,
        declared_keys: frozenset | set | None = None,
        sink: Any = None,
    ) -> Trainer:
        """Restore a Trainer — thin delegate to `checkpoints.resume_trainer` (§c.7)."""
        return checkpoints.resume_trainer(
            cls, checkpoint_path, fallback_config=fallback_config,
            config_overrides=config_overrides, declared_keys=declared_keys,
            sink=sink, device=device,
        )


def _resolve_spec(config: Any):
    """Resolve the encoding spec from a config. Both the WP8 NESTED `identity.encoding` shape
    and the legacy FLAT `encoding` shape are read by `resolve_from_config` itself, which is the
    ONE authority for where an encoding may be declared (TD-4 / CARD-POOL-ENCODING-BRIDGE); this
    function is now just the non-dict coercion the Trainer's own callers need. Keeps
    `metadata.encoding_name` consistent with `config.identity.encoding` (the loader's
    stamp-source check, T-CK-30) — unchanged, and now by the same route every other caller
    takes rather than by a private bridge only this module had."""
    return resolve_from_config(dict(config) if isinstance(config, dict) else {})


def _prune_policy_targets(pi: torch.Tensor, threshold_frac: float) -> torch.Tensor:
    """Zero policy-target entries at/below `threshold_frac · max(row)`, renormalize
    (behaviour-exact with the old `prune_policy_targets`; sharpens MCTS visit targets)."""
    if threshold_frac <= 0.0:
        return pi
    max_vals = pi.max(dim=-1, keepdim=True).values
    mask = pi > (threshold_frac * max_vals)
    pruned = pi * mask
    return pruned / pruned.sum(dim=-1, keepdim=True).clamp(min=1e-8)
