"""Shared loss computation for the Trainer + pretrain: the ragged policy CE, the binned value loss
and the chain head; the trainer sums `policy_loss + value_loss` (`trainer/core.py`)."""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

# Canonical stub-exported location — `torch.amp` itself does not re-export for type checkers.
from torch.amp.grad_scaler import GradScaler

from mantis.selfplay.graph_collate import segment_ids, segment_softmax, segment_sum
from mantis.util.constants import is_alpha_full


def sparse_tail(
    prior_probs: torch.Tensor,
    explicit_mask: torch.Tensor,
    tail_mass: torch.Tensor,
    legal_offsets: torch.Tensor,
) -> torch.Tensor:
    """The SPARSE row's tail (R347(a)): `tail_mass` spread over the non-explicit legal nodes in proportion to the DETACHED prior — the ONE construction the CE, the soft target and the KL row share; an all-explicit graph has an empty tail."""
    b = int(legal_offsets.shape[0]) - 1
    seg = segment_ids(legal_offsets, total=int(prior_probs.shape[0]))
    tail_prior = prior_probs.detach() * (1.0 - explicit_mask.reshape(-1).to(prior_probs.dtype))
    scale = tail_mass.reshape(-1).to(tail_prior.dtype) / segment_sum(tail_prior, seg, b).clamp_min(1e-12)
    return tail_prior * scale[seg]


def rebuild_sparse_target(
    policy_target: torch.Tensor,
    prior_probs: torch.Tensor,
    explicit_mask: torch.Tensor,
    tail_mass: torch.Tensor,
    legal_offsets: torch.Tensor,
) -> torch.Tensor:
    """The hard policy target as the CE trains on it: the explicit masses plus `sparse_tail`."""
    return policy_target.reshape(-1) + sparse_tail(prior_probs, explicit_mask, tail_mass, legal_offsets)


def graph_policy_row_weights(
    is_full_search: Any, fast_policy_weight: float
) -> torch.Tensor:
    """Per-row POLICY weight for a graph batch, the one authority: 1 on a full-search row,
    `fast_policy_weight` on a fast-arm row (weighted, not gated). A flat float32 `[B]`, evaluated
    ONCE per step and read by both the numerator and the denominator.

    Raises:
        ValueError: `fast_policy_weight` negative or not finite — it would train the fast arm AWAY.
    """
    if not math.isfinite(fast_policy_weight) or fast_policy_weight < 0.0:
        raise ValueError(
            f"fast_policy_weight must be finite and >= 0, got {fast_policy_weight!r} "
            "(train.fast_policy_weight)"
        )
    ifs = torch.as_tensor(is_full_search).reshape(-1).to(torch.float32)
    return ifs + fast_policy_weight * (1.0 - ifs)


def exclude_alpha_full_rows(policy_row_weight: torch.Tensor, tail_mass: Any) -> tuple[torch.Tensor, int]:
    """Zero the policy weight of every row at alpha = 1.0 (R350(e): "play none of the searched
    moves" is not a target), BEFORE the denominator reads the vector; returns `(weights, n_excluded)`.
    Raises ValueError when the two vectors disagree in length.
    """
    alpha = torch.as_tensor(tail_mass).reshape(-1)
    if alpha.shape != policy_row_weight.shape:
        raise ValueError(
            f"exclude_alpha_full_rows: tail_mass has {alpha.shape[0]} rows for "
            f"{policy_row_weight.shape[0]} row weights"
        )
    full = torch.tensor([is_alpha_full(a) for a in alpha.tolist()], dtype=torch.bool)
    return policy_row_weight * (~full).to(policy_row_weight.dtype), int(full.sum().item())


def policy_loss_weight_at(step: int, warmup_steps: int) -> float:
    """R350(b)(iii): 0.0 while `step < warmup_steps` (0-based, the step about to be taken), else 1.0.

    Raises:
        ValueError: a negative step or warm-up length.
    """
    if step < 0 or warmup_steps < 0:
        raise ValueError(f"policy_loss_weight_at: step={step}, warmup_steps={warmup_steps} must be >= 0")
    return 0.0 if step < warmup_steps else 1.0


def graph_loss_denominators(
    is_full_search: Any,
    value_valid: Any,
    n_graphs: int,
) -> tuple[float, float]:
    """`(policy_denominator, value_denominator)` for ONE step's WHOLE batch, so every micro-batch
    divides by what the un-split batch would have (never `B_m/B` or `1/M`). DELIBERATELY asymmetric:
    policy SUMS the mask values, value COUNTS true entries — they agree only on a strict 0/1 mask
    (measured `[2, 0, 3]`: 5.0 vs 2.0). The `None` arms fall back to the graph count; the value arm's
    is right only with one `bin_logits` row per graph, asserted by the caller.
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
    """`ragged_policy_ce_and_entropies`'s CE alone — the loss term. See that function."""
    ce, _h_target, _h_model = ragged_policy_ce_and_entropies(
        policy_logits, policy_target, legal_offsets, full_search_mask=full_search_mask,
        denominator=denominator, explicit_mask=explicit_mask, tail_mass=tail_mass,
    )
    return ce


def ragged_policy_ce_and_entropies(
    policy_logits: torch.Tensor,
    policy_target: torch.Tensor,
    legal_offsets: torch.Tensor,
    full_search_mask: torch.Tensor | None = None,
    denominator: float | None = None,
    explicit_mask: torch.Tensor | None = None,
    tail_mass: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ragged per-legal-node policy CE for the GNN graph branch and — reduced the SAME way, DETACHED —
    the target's entropy (`CE - H` is KL(target || policy), R350(b)(iv)) and the model's own (B-4).

    Per graph: log_softmax over its legal segment, `-Σ target·logp`, masked by `full_search_mask`;
    `denominator` makes ONE micro-batch divide by the WHOLE step's so the parts sum to the un-split
    loss; `explicit_mask` / `tail_mass` rebuild the SPARSE Gumbel row's tail from THIS model's
    DETACHED prior (live, the CE gradient would push the prior toward itself).

    Raises:
        ValueError: exactly one of `explicit_mask` / `tail_mass` was supplied — a tail mass with no set.
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
        zero = torch.zeros((), device=device, dtype=torch.float32)
        return zero, zero.clone(), zero.clone()
    probs = segment_softmax(policy_logits, legal_offsets)
    logp = torch.log(probs.clamp(min=1e-12))
    seg = segment_ids(legal_offsets, total=int(probs.shape[0]))
    target = policy_target
    if explicit_mask is not None and tail_mass is not None:
        target = rebuild_sparse_target(policy_target, probs, explicit_mask, tail_mass, legal_offsets)
    per_graph = segment_sum(-(target * logp), seg, b)
    with torch.no_grad():
        t = target.detach()
        entropy_graph = segment_sum(-(t * torch.log(t.clamp(min=1e-12))), seg, b)
        p = probs.detach()
        model_graph = segment_sum(-(p * torch.log(p.clamp(min=1e-12))), seg, b)

    def _reduce(values: torch.Tensor) -> torch.Tensor:
        if full_search_mask is not None:
            mask = full_search_mask.reshape(-1).to(values.dtype)
            if denominator is not None:
                return (values * mask).sum() / denominator
            return (values * mask).sum() / mask.sum().clamp_min(1.0)
        if denominator is not None:
            return values.sum() / denominator
        return values.mean()

    return _reduce(per_graph), _reduce(entropy_graph), _reduce(model_graph)


def soft_policy_target(
    policy_target: torch.Tensor,
    legal_offsets: torch.Tensor,
    explicit_mask: torch.Tensor,
    tail_mass: torch.Tensor,
    prior_probs: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """The aux head's target, detached: the searched target's EXPLICIT entries at `1/temperature`, renormalised to the explicit mass, plus the hard target's own tail (the temperature stays OFF the tail — the measured reason is RUN10_PREREG §1a); Raises: ValueError — `temperature` not above 1 (at 1 the head reads armed while dead)."""
    if not math.isfinite(temperature) or temperature <= 1.0:
        raise ValueError(f"soft_policy_target: temperature must be > 1, got {temperature!r}")
    with torch.no_grad():
        target = policy_target.to(torch.float32).reshape(-1)
        explicit = explicit_mask.reshape(-1).to(torch.bool)
        b = int(legal_offsets.shape[0]) - 1
        seg = segment_ids(legal_offsets, total=int(target.shape[0]))
        alpha = tail_mass.reshape(-1).to(torch.float32)
        sharpened = torch.where(explicit & (target > 0),
                                torch.exp(torch.log(target.clamp_min(1e-30)) / temperature),
                                torch.zeros_like(target))
        explicit_mass = (1.0 - alpha).clamp_min(0.0)
        soft_explicit = sharpened * (explicit_mass / segment_sum(sharpened, seg, b).clamp_min(1e-30))[seg]
        prior32 = prior_probs.detach().to(torch.float32).reshape(-1)
        return soft_explicit + sparse_tail(prior32, explicit_mask, tail_mass, legal_offsets)


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


def backward_accumulate(loss: torch.Tensor, scaler: GradScaler, fp16: bool) -> None:
    """The BACKWARD half of the fp16 step — accumulate into `.grad`, step nothing. No `zero_grad`
    inside, which is the point: the loop body accumulates and the caller zeroes once before it
    and steps once after it."""
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

