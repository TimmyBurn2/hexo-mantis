"""Shared loss computation for the Trainer + pretrain (WP10 §a.4 PORT; old training/losses.py).

>300 justify: the full policy/value/aux loss family (dense CE, ragged graph CE, dist65
value, uncertainty, ownership/threat, the surviving Q13 `chain_head` smooth-L1, ply-index,
the total-loss combiner, and the fp16 backward step) is ONE concern — the trainer's loss
math — kept together so the numeric contract is greppable in one file. Behaviour-exact
relocation; two additions with reasons:
  * `_segment_softmax` is inlined (old `ragged_policy_ce` up-imported it from
    `selfplay/graph_collate.py`, which is not part of the WP10 train surface) — the same
    numerically-stable per-graph softmax, verbatim.
  * `chain_loss_with_fire_rate` wraps `compute_chain_loss` with an in-run fire-rate
    SELF-REPORT on the `chain_planes` target (LAW-07/LAW-18, WP9-owed O-CHAIN): weight>0
    fires + logs its fire-rate; weight 0 reports 0. The loss MATH is untouched.

Architecture spec (docs/01_architecture.md §2):
    L = L_policy + L_value + w_aux·L_opp_reply + w_unc·L_uncertainty (+ chain/ply/own/threat)
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from mantis.train.emit import emit_via


# ── graph ragged-CE helper (inlined segment softmax) ───────────────────────────────────
def _segment_softmax(logits: torch.Tensor, legal_offsets: torch.Tensor) -> torch.Tensor:
    """Numerically-stable per-graph softmax over each graph's legal nodes.

    `logits` is the flat `[Lg_total]` per-legal-node tensor `GnnNet.forward_batch` returns;
    `legal_offsets` is the `[B+1]` CSR pointer segmenting it per graph. Returns `[Lg_total]`
    probs summing to 1 within each segment. Vectorized (scatter_reduce_/scatter_add_), no
    Python per-graph loop. Inlined verbatim from the old selfplay/graph_collate.segment_softmax.
    """
    counts = legal_offsets[1:] - legal_offsets[:-1]
    b = int(legal_offsets.shape[0]) - 1
    seg = torch.repeat_interleave(
        torch.arange(b, device=logits.device, dtype=torch.long), counts
    )
    seg_max = torch.full((b,), float("-inf"), dtype=logits.dtype, device=logits.device)
    seg_max.scatter_reduce_(0, seg, logits, reduce="amax", include_self=False)
    ex = torch.exp(logits - seg_max[seg])
    denom = torch.zeros(b, dtype=logits.dtype, device=logits.device)
    denom.scatter_add_(0, seg, ex)
    return ex / denom[seg]


# ── policy losses ──────────────────────────────────────────────────────────────────────
def compute_policy_loss(
    log_policy: torch.Tensor,
    target_policy: torch.Tensor,
    valid_mask: torch.Tensor,
    device: torch.device,
    full_search_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cross-entropy policy loss, masked on zero-policy rows and quick-search positions."""
    combined = valid_mask
    if full_search_mask is not None:
        combined = valid_mask & full_search_mask.bool()
    if combined.any():
        return -(target_policy[combined] * log_policy[combined]).sum(dim=1).mean()
    return torch.zeros(1, device=device, dtype=torch.float32).squeeze()


def ragged_policy_ce(
    policy_logits: torch.Tensor,
    policy_target: torch.Tensor,
    legal_offsets: torch.Tensor,
    full_search_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Ragged per-legal-node policy CE for the GNN graph branch — the no-drop replacement
    for the dense-362 `compute_policy_loss`. Per graph: log_softmax over its legal-node
    segment, then -Σ target·logp, masked by is_full_search (quick-search rows contribute
    value only).

    fp32 cast at entry mirrors `binned_value_loss`: under autocast `torch.exp` inside the
    segment softmax autopromotes to fp32 but an fp16 `policy_logits` drags fp16 along and
    the scatter_add dtype-mismatches (WP5b BREAK-1); the cast also fixes the fp16 log-clamp
    underflow (1e-12 flushing to 0 → log(0) → NaN).
    """
    policy_logits = policy_logits.to(torch.float32)
    device = policy_logits.device
    b = int(legal_offsets.shape[0]) - 1
    if b == 0 or policy_logits.numel() == 0:
        return torch.zeros((), device=device, dtype=torch.float32)
    probs = _segment_softmax(policy_logits, legal_offsets)
    logp = torch.log(probs.clamp(min=1e-12))
    per_node = -(policy_target * logp)  # (Lg,)
    counts = legal_offsets[1:] - legal_offsets[:-1]
    seg = torch.repeat_interleave(
        torch.arange(b, device=device, dtype=torch.long), counts
    )
    per_graph = torch.zeros(b, device=device, dtype=per_node.dtype)
    per_graph.scatter_add_(0, seg, per_node)  # (B,)
    if full_search_mask is not None:
        mask = full_search_mask.reshape(-1).to(per_graph.dtype)
        denom = mask.sum().clamp_min(1.0)
        return (per_graph * mask).sum() / denom
    return per_graph.mean()


def compute_kl_policy_loss(
    log_policy: torch.Tensor,
    target_policy: torch.Tensor,
    valid_mask: torch.Tensor,
    device: torch.device,
    full_search_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """KL(target || model) policy loss for completed-Q targets — identical gradients to CE,
    more interpretable value (0 when the model matches the target)."""
    combined = valid_mask
    if full_search_mask is not None:
        combined = valid_mask & full_search_mask.bool()
    if combined.any():
        tgt = target_policy[combined]
        log_model = log_policy[combined]
        log_tgt = torch.log(tgt.clamp(min=1e-8)).clamp(min=-100.0)  # fp16-safe
        return (tgt * (log_tgt - log_model)).sum(dim=1).mean()
    return torch.zeros(1, device=device, dtype=torch.float32).squeeze()


# ── value loss (scalar head) ─────────────────────────────────────────────────────────────
def compute_value_loss(
    value_logit: torch.Tensor,
    outcome: torch.Tensor,
    value_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Numerically-stable BCE via `binary_cross_entropy_with_logits`; outcomes {-1,+1}→{0,1}.

    DRAW-MASK: `value_mask` 0 rows (ply-capped horizon truncation, a false label) are excluded
    from numerator AND denominator; `None` = all rows contribute (pretrain corpus path)."""
    value_target = (outcome + 1.0) / 2.0
    logit = value_logit.squeeze(1)
    if value_mask is None:
        return nn.functional.binary_cross_entropy_with_logits(logit, value_target)
    per_row = nn.functional.binary_cross_entropy_with_logits(logit, value_target, reduction="none")
    mask = value_mask.reshape(-1).bool()
    combined = per_row[mask]
    if combined.numel() == 0:
        return torch.zeros((), device=per_row.device, dtype=per_row.dtype)
    return combined.mean()


# ── aux heads ────────────────────────────────────────────────────────────────────────────
def compute_aux_loss(
    aux_logit: torch.Tensor,
    target_policy: torch.Tensor,
    valid_mask: torch.Tensor,
    device: torch.device,
    full_search_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Opponent-reply auxiliary loss (policy-shaped; same MCTS visit targets, same gate)."""
    combined = valid_mask
    if full_search_mask is not None:
        combined = valid_mask & full_search_mask.bool()
    if combined.any():
        valid_targets = target_policy[combined]
        valid_logits = aux_logit[combined]
        safe_log = valid_logits.clamp(min=-100.0)
        return -(valid_targets * safe_log).sum(dim=1).mean()
    return torch.zeros(1, device=device, dtype=torch.float32).squeeze()


def compute_chain_loss(
    chain_pred: torch.Tensor,
    chain_target: torch.Tensor,
    legal_mask: Optional[torch.Tensor] = None,
    huber_delta: float = 1.0,
) -> torch.Tensor:
    """Q13-aux smooth-L1 (Huber) loss on 6 chain-length planes.

    `chain_pred`/`chain_target`: (B, 6, H, W). `legal_mask`: optional float mask broadcastable
    to (B, 6, H, W) (a (B,1,H,W) or (B,H,W) mask broadcasts across the 6 planes); `None` → every
    cell contributes (reduction="mean"). Targets live in [0,1] after /6.0 normalization.
    """
    if legal_mask is None:
        return torch.nn.functional.smooth_l1_loss(
            chain_pred.float(), chain_target.float(), beta=huber_delta, reduction="mean",
        )
    per_cell = torch.nn.functional.smooth_l1_loss(
        chain_pred.float(), chain_target.float(), beta=huber_delta, reduction="none",
    )
    mask = legal_mask.float()
    if mask.dim() == per_cell.dim() - 1:
        mask = mask.unsqueeze(1)
    mask_b = mask.expand_as(per_cell)
    return (per_cell * mask_b).sum() / mask_b.sum().clamp_min(1.0)


def chain_target_fire_rate(
    chain_target: torch.Tensor, legal_mask: Optional[torch.Tensor] = None
) -> float:
    """Fraction of batch rows whose `chain_planes` target carries signal (any nonzero over
    the 6 planes) — the fire-rate the in-run self-report publishes (LAW-18). When `legal_mask`
    is given, a row counts as firing iff it has a nonzero target on a legal (mask>0) cell."""
    b = int(chain_target.shape[0])
    if b == 0:
        return 0.0
    t = chain_target.float()
    if legal_mask is not None:
        m = legal_mask.float()
        if m.dim() == t.dim() - 1:
            m = m.unsqueeze(1)
        t = t * m.expand_as(t)
    active = t.reshape(b, -1).abs().amax(dim=1) > 0
    return active.float().mean().item()


def chain_loss_with_fire_rate(
    chain_pred: torch.Tensor,
    chain_target: torch.Tensor,
    weight: float,
    *,
    legal_mask: Optional[torch.Tensor] = None,
    huber_delta: float = 1.0,
    sink: Any = None,
    step: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Compute the `chain_head` smooth-L1 loss AND self-report its fire-rate in-run
    (LAW-07/LAW-18, O-CHAIN). At ``weight <= 0`` the lever is OFF: no loss is returned and
    the report publishes ``fire_rate = 0.0`` (a disabled lever stays VISIBLE). At
    ``weight > 0`` the loss fires and the report carries the fire-rate measured on the
    ``chain_planes`` target (guarding the F-10 silent-wrong-sub-buffer class). The loss
    MATH is `compute_chain_loss` unchanged — this only adds the self-report seam."""
    if weight <= 0.0:
        emit_via(sink, {"event": "aux_chain_loss", "weight": float(weight),
                        "fired": False, "fire_rate": 0.0, "step": step})
        return None
    loss = compute_chain_loss(chain_pred, chain_target, legal_mask=legal_mask,
                              huber_delta=huber_delta)
    fire_rate = chain_target_fire_rate(chain_target, legal_mask=legal_mask)
    emit_via(sink, {"event": "aux_chain_loss", "weight": float(weight), "fired": True,
                    "fire_rate": fire_rate, "loss": float(loss.detach().float()), "step": step})
    return loss


def compute_uncertainty_loss(
    sigma2: torch.Tensor,
    z_targets: torch.Tensor,
    value_detached: torch.Tensor,
) -> torch.Tensor:
    """Huber loss for the value-uncertainty head — predicts squared value error (bounded
    everywhere; the head learns the magnitude of the value head's error). Gradient flows
    only through the head params — caller passes a detached value tensor."""
    z = z_targets.float().unsqueeze(1)
    target = (z - value_detached.float()).pow(2)
    return torch.nn.functional.smooth_l1_loss(sigma2.float(), target, beta=1.0, reduction="mean")


def compute_ply_index_loss(
    ply_pred: torch.Tensor,
    position_indices: torch.Tensor,
) -> torch.Tensor:
    """Huber loss on normalized ply index (target = clamp(position_indices/100, 0, 1)) —
    forces the trunk to encode game-time progress."""
    target = (position_indices.float() / 100.0).clamp(0.0, 1.0).unsqueeze(1)
    return torch.nn.functional.smooth_l1_loss(ply_pred.float(), target, beta=1.0, reduction="mean")


# ── total-loss combiner + backward step ─────────────────────────────────────────────────
def compute_total_loss(
    policy_loss: torch.Tensor,
    value_loss: torch.Tensor,
    aux_loss: Optional[torch.Tensor] = None,
    aux_weight: float = 0.0,
    entropy_bonus: Optional[torch.Tensor] = None,
    entropy_weight: float = 0.0,
    uncertainty_loss: Optional[torch.Tensor] = None,
    uncertainty_weight: float = 0.0,
    ownership_loss: Optional[torch.Tensor] = None,
    ownership_weight: float = 0.0,
    threat_loss: Optional[torch.Tensor] = None,
    threat_weight: float = 0.0,
    chain_loss: Optional[torch.Tensor] = None,
    chain_weight: float = 0.0,
    ply_index_loss: Optional[torch.Tensor] = None,
    ply_index_weight: float = 0.0,
) -> torch.Tensor:
    """Combine policy, value, aux, entropy, uncertainty, ownership, threat, chain, ply-index."""
    total = policy_loss + value_loss
    if aux_loss is not None and aux_weight > 0.0:
        total = total + aux_weight * aux_loss
    if entropy_bonus is not None and entropy_weight > 0.0:
        total = total - entropy_weight * entropy_bonus
    if uncertainty_loss is not None and uncertainty_weight > 0.0:
        total = total + uncertainty_weight * uncertainty_loss
    if ownership_loss is not None and ownership_weight > 0.0:
        total = total + ownership_weight * ownership_loss
    if threat_loss is not None and threat_weight > 0.0:
        total = total + threat_weight * threat_loss
    if chain_loss is not None and chain_weight > 0.0:
        total = total + chain_weight * chain_loss
    if ply_index_loss is not None and ply_index_weight > 0.0:
        total = total + ply_index_weight * ply_index_loss
    return total


def fp16_backward_step(
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    model: nn.Module,
    fp16: bool,
    max_grad_norm: float = 1.0,
) -> float:
    """Backward pass with optional FP16 gradient scaling + clipping. Returns the pre-clip
    gradient norm (the diagnostic signal)."""
    if fp16:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm).item()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm).item()
        optimizer.step()
    return grad_norm
