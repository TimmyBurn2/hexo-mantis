"""Shared loss computation for the Trainer + pretrain.

>300 justify (R8): the full policy/value/aux loss family — dense CE, ragged graph CE, dist65
value, uncertainty, ownership/threat, chain, ply-index, the total-loss combiner and the fp16
backward step — is ONE concern, the trainer's loss math, kept together so the numeric contract is
greppable in one file. `_segment_softmax` is inlined rather than up-imported from the self-play
collate, which is not part of the train surface.

Architecture spec (docs/01_architecture.md §2):
    L = L_policy + L_value + w_aux·L_opp_reply + w_unc·L_uncertainty (+ chain/ply/own/threat)
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

# Canonical stub-exported location — `torch.amp` itself does not re-export for type checkers.
from torch.amp.grad_scaler import GradScaler

from mantis.train.emit import emit_via


def _segment_softmax(logits: torch.Tensor, legal_offsets: torch.Tensor) -> torch.Tensor:
    """Numerically-stable per-graph softmax over each graph's legal nodes: flat `[Lg_total]`
    logits segmented by the `[B+1]` CSR `legal_offsets`, returning probs summing to 1 within each
    segment. Vectorized (scatter_reduce_/scatter_add_), no Python per-graph loop."""
    counts = legal_offsets[1:] - legal_offsets[:-1]
    b = int(legal_offsets.shape[0]) - 1
    seg = torch.repeat_interleave(
        torch.arange(b, device=logits.device, dtype=torch.long), counts,
        output_size=int(logits.shape[0]),
    )
    seg_max = torch.full((b,), float("-inf"), dtype=logits.dtype, device=logits.device)
    seg_max.scatter_reduce_(0, seg, logits, reduce="amax", include_self=False)
    ex = torch.exp(logits - seg_max[seg])
    denom = torch.zeros(b, dtype=logits.dtype, device=logits.device)
    denom.scatter_add_(0, seg, ex)
    return ex / denom[seg]


def compute_policy_loss(
    log_policy: torch.Tensor,
    target_policy: torch.Tensor,
    valid_mask: torch.Tensor,
    device: torch.device,
    full_search_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy policy loss, masked on zero-policy rows and quick-search positions."""
    combined = valid_mask
    if full_search_mask is not None:
        combined = valid_mask & full_search_mask.bool()
    if combined.any():
        return -(target_policy[combined] * log_policy[combined]).sum(dim=1).mean()
    return torch.zeros(1, device=device, dtype=torch.float32).squeeze()


def graph_policy_row_weights(
    is_full_search: Any, fast_policy_weight: float
) -> torch.Tensor:
    """Per-row POLICY weight for a graph batch — the one authority.

    A full-search row weighs 1 and a fast-arm row weighs `fast_policy_weight`, so the fast arm
    trains VALUE always and POLICY at a declared weight instead of being gated out entirely.

    Args:
        is_full_search: per-row full-search flags, `[B]`. `Any` because the ONE production caller
            holds them as NUMPY pre-collate; `torch.as_tensor` is a no-op on a tensor.
        fast_policy_weight: `train.fast_policy_weight`, resolved by its own resolver.

    Returns:
        A flat float32 tensor `[B]`, evaluated ONCE per step and read by BOTH the numerator and
        the denominator, because two evaluations of one rule can disagree.

    Raises:
        ValueError: `fast_policy_weight` is negative or not finite — a negative policy weight
            would train the fast arm AWAY from its own target.
    """
    if not math.isfinite(fast_policy_weight) or fast_policy_weight < 0.0:
        raise ValueError(
            f"fast_policy_weight must be finite and >= 0, got {fast_policy_weight!r} "
            "(train.fast_policy_weight)"
        )
    ifs = torch.as_tensor(is_full_search).reshape(-1).to(torch.float32)
    return ifs + fast_policy_weight * (1.0 - ifs)


def graph_loss_denominators(
    is_full_search: Any,
    value_valid: Any,
    n_graphs: int,
) -> tuple[float, float]:
    """`(policy_denominator, value_denominator)` for ONE training step's WHOLE batch.

    The micro-batch split is equivalent to the un-split step only if every micro-batch divides by
    the denominator the UN-SPLIT batch would have used. Weighting by `B_m/B` or by `1/M` is wrong:
    neither denominator is the graph count, and no single scalar per micro-batch is right for both.

    THE TWO EXPRESSIONS ARE DELIBERATELY ASYMMETRIC BECAUSE THE LOSSES ARE — policy sums the mask
    VALUES, value counts TRUE entries — and they agree only while the masks are strictly 0/1, so a
    SYMMETRIC implementation would pass every behavioural oracle while encoding a latent
    divergence. Measured on the mask `[2, 0, 3]`: 5.0 vs 2.0.

    The `None` arms fall back to the graph count; the value arm's fallback is only correct while
    `bin_logits` has one row per graph, which the caller ASSERTS at the call site.
    """
    if is_full_search is None:
        p_den = float(n_graphs)
    else:
        p_den = max(float(torch.as_tensor(is_full_search).sum()), 1.0)
    if value_valid is None:
        v_den = float(n_graphs)
    else:
        v_den = max(float(torch.as_tensor(value_valid).reshape(-1).bool().sum()), 1.0)
    return p_den, v_den


def ragged_policy_ce(
    policy_logits: torch.Tensor,
    policy_target: torch.Tensor,
    legal_offsets: torch.Tensor,
    full_search_mask: torch.Tensor | None = None,
    denominator: float | None = None,
    explicit_mask: torch.Tensor | None = None,
    tail_mass: torch.Tensor | None = None,
) -> torch.Tensor:
    """Ragged per-legal-node policy CE for the GNN graph branch — the no-drop replacement for the
    dense-362 `compute_policy_loss`.

    Per graph: log_softmax over its legal-node segment, then -Σ target·logp, masked by
    is_full_search so quick-search rows contribute value only. The fp32 cast at entry stops an fp16
    `policy_logits` dragging fp16 into the scatter_add, and fixes the fp16 log-clamp underflow.

    `denominator`, when supplied, makes the reduction `numerator_sum / denominator` — how ONE
    micro-batch divides by the WHOLE step's denominator so the parts sum to the un-split loss.

    `explicit_mask` / `tail_mass` reconstruct the SPARSE Gumbel row from THIS model's prior. THE
    DETACH IS THE MECHANISM: built from a live `probs` the tail would be a function of the
    parameters, and the CE gradient would pick up a second term pushing the prior toward whatever
    it already is — a self-referential objective on most of the legal set.

    Raises:
        ValueError: exactly one of `explicit_mask` / `tail_mass` was supplied — a tail mass with
            no support set has no set to spread over.
    """
    if (explicit_mask is None) != (tail_mass is None):
        raise ValueError(
            "ragged_policy_ce: explicit_mask and tail_mass are one argument in two parts "
            "(R347(a)) — supply both or neither"
        )
    policy_logits = policy_logits.to(torch.float32)
    device = policy_logits.device
    b = int(legal_offsets.shape[0]) - 1
    if b == 0 or policy_logits.numel() == 0:
        return torch.zeros((), device=device, dtype=torch.float32)
    probs = _segment_softmax(policy_logits, legal_offsets)
    logp = torch.log(probs.clamp(min=1e-12))
    counts = legal_offsets[1:] - legal_offsets[:-1]
    seg = torch.repeat_interleave(
        torch.arange(b, device=device, dtype=torch.long), counts
    )
    target = policy_target
    if explicit_mask is not None and tail_mass is not None:
        # DETACHED: the reconstructed tail is a target, not a term of the model.
        tail_prior = probs.detach() * (
            1.0 - explicit_mask.reshape(-1).to(probs.dtype)
        )
        tail_denom = torch.zeros(b, device=device, dtype=tail_prior.dtype)
        tail_denom.scatter_add_(0, seg, tail_prior)
        # A graph whose legal set is entirely explicit has an empty tail; the clamp keeps the
        # divide finite and the numerator is zero there anyway.
        scale = tail_mass.reshape(-1).to(tail_prior.dtype) / tail_denom.clamp_min(1e-12)
        target = policy_target + tail_prior * scale[seg]
    per_node = -(target * logp)  # (Lg,)
    per_graph = torch.zeros(b, device=device, dtype=per_node.dtype)
    per_graph.scatter_add_(0, seg, per_node)  # (B,)
    if full_search_mask is not None:
        mask = full_search_mask.reshape(-1).to(per_graph.dtype)
        if denominator is not None:
            return (per_graph * mask).sum() / denominator
        denom = mask.sum().clamp_min(1.0)
        return (per_graph * mask).sum() / denom
    if denominator is not None:
        return per_graph.sum() / denominator
    return per_graph.mean()


def compute_kl_policy_loss(
    log_policy: torch.Tensor,
    target_policy: torch.Tensor,
    valid_mask: torch.Tensor,
    device: torch.device,
    full_search_mask: torch.Tensor | None = None,
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


def compute_value_loss(
    value_logit: torch.Tensor,
    outcome: torch.Tensor,
    value_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Numerically-stable BCE via `binary_cross_entropy_with_logits`; outcomes {-1,+1}→{0,1}.
    `value_mask` 0 rows (ply-capped horizon truncation, a false label) are excluded from numerator
    AND denominator; `None` means all rows contribute."""
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


def compute_aux_loss(
    aux_logit: torch.Tensor,
    target_policy: torch.Tensor,
    valid_mask: torch.Tensor,
    device: torch.device,
    full_search_mask: torch.Tensor | None = None,
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
    legal_mask: torch.Tensor | None = None,
    huber_delta: float = 1.0,
) -> torch.Tensor:
    """Smooth-L1 (Huber) loss on 6 chain-length planes. `chain_pred`/`chain_target` are
    (B, 6, H, W); `legal_mask` is an optional float mask broadcastable to that shape, and `None`
    means every cell contributes. Targets live in [0,1]."""
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
    chain_target: torch.Tensor, legal_mask: torch.Tensor | None = None
) -> float:
    """Fraction of batch rows whose `chain_planes` target carries signal — the fire-rate the
    in-run self-report publishes (LAW-18). With `legal_mask`, a row fires iff it has a nonzero
    target on a legal cell."""
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
    legal_mask: torch.Tensor | None = None,
    huber_delta: float = 1.0,
    sink: Any = None,
    step: int | None = None,
) -> torch.Tensor | None:
    """Compute the `chain_head` smooth-L1 loss AND self-report its fire-rate in-run. At
    ``weight <= 0`` the lever is OFF: no loss, and the report publishes ``fire_rate = 0.0``, so a
    disabled lever stays VISIBLE. The loss MATH is `compute_chain_loss` unchanged."""
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
    """Huber loss for the value-uncertainty head, predicting squared value error. Gradient flows
    only through the head params — the caller passes a detached value tensor."""
    z = z_targets.float().unsqueeze(1)
    target = (z - value_detached.float()).pow(2)
    return torch.nn.functional.smooth_l1_loss(sigma2.float(), target, beta=1.0, reduction="mean")


def compute_ply_index_loss(
    ply_pred: torch.Tensor,
    position_indices: torch.Tensor,
) -> torch.Tensor:
    """Huber loss on normalized ply index, forcing the trunk to encode game-time progress."""
    target = (position_indices.float() / 100.0).clamp(0.0, 1.0).unsqueeze(1)
    return torch.nn.functional.smooth_l1_loss(ply_pred.float(), target, beta=1.0, reduction="mean")


def compute_total_loss(
    policy_loss: torch.Tensor,
    value_loss: torch.Tensor,
    aux_loss: torch.Tensor | None = None,
    aux_weight: float = 0.0,
    entropy_bonus: torch.Tensor | None = None,
    entropy_weight: float = 0.0,
    uncertainty_loss: torch.Tensor | None = None,
    uncertainty_weight: float = 0.0,
    ownership_loss: torch.Tensor | None = None,
    ownership_weight: float = 0.0,
    threat_loss: torch.Tensor | None = None,
    threat_weight: float = 0.0,
    chain_loss: torch.Tensor | None = None,
    chain_weight: float = 0.0,
    ply_index_loss: torch.Tensor | None = None,
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


def backward_accumulate(loss: torch.Tensor, scaler: GradScaler, fp16: bool) -> None:
    """The BACKWARD half of `fp16_backward_step` — accumulate into `.grad`, step nothing. No
    `zero_grad` inside, which is the point: the loop body accumulates and the caller zeroes once
    before it and steps once after it."""
    if fp16:
        scaler.scale(loss).backward()
    else:
        loss.backward()


def clip_and_step(
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    model: nn.Module,
    fp16: bool,
    max_grad_norm: float,
) -> float:
    """The CLIP+STEP half — run ONCE per training step, on the ACCUMULATED gradient. Clipping is
    NONLINEAR in the whole gradient and the pre-clip norm it returns is an ARMED GATE'S INPUT, so
    clipping per micro-batch would rescale a live abort threshold by an invisible M."""
    if fp16:
        scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm).item()
    if not math.isfinite(grad_norm):
        # The pre-clip norm is a COMPLETE detector for a non-finite gradient: it is
        # `sqrt(sum of squares)`, so a finite norm means every entry is finite.
        # `clip_grad_norm_` has already multiplied the gradients by a non-finite clip coefficient
        # by the time we read it, so the refusal must be here: those gradients are discarded and
        # `optimizer.step()` is the one thing that does not run.
        optimizer.zero_grad(set_to_none=True)
        if fp16:
            # The scaler still gets its update, so its inf-driven backoff keeps working;
            # `scaler.step` is what is skipped, not the scaler's bookkeeping.
            scaler.update()
        return grad_norm
    if fp16:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    return grad_norm


def fp16_backward_step(
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    model: nn.Module,
    fp16: bool,
    max_grad_norm: float = 1.0,
) -> float:
    """Backward pass with optional FP16 gradient scaling + clipping, returning the pre-clip
    gradient norm. DECOMPOSED, NOT FORKED: exactly the composition of `backward_accumulate` and
    `clip_and_step`, the same statements in the same order on the same objects."""
    backward_accumulate(loss, scaler, fp16)
    return clip_and_step(optimizer, scaler, model, fp16, max_grad_norm)
