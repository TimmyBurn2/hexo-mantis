"""CNN head math + K-cluster MinMaxPool + aux-head factories.

The head LAYERS stay direct attributes of `HexTacToeNet` (state-dict keys rooted
at the parent, byte-identical to the old min/max net). This module holds only the
head FORWARD math (free functions taking layer references) + the stateless
`MinMaxPool` reduction + factory functions that build the aux-head submodules. The
gpool-bias `policy_bias`/`value_bias` args are dropped (that branch is killed; the
default was `None`, so the forward is byte-identical).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mantis.model.dist65 import decode_binned_value


def min_max_window_head(
    out: torch.Tensor,
    *,
    policy_conv: nn.Conv2d,
    policy_fc: nn.Linear,
    value_fc1: nn.Linear,
    value_fc2: nn.Linear,
    value_head_type: str = "scalar",
    value_fc2_bins: nn.Linear | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-window min_max head for `has_pass_slot=true` encodings.

    Args:
        out:        `(N, C, H, W)` trunk output. `N` is `B` when called from
                    `HexTacToeNet.forward` (one window per board) or `K` when called
                    from `aggregated_forward_K` (K cluster windows for a board).
        policy_conv / policy_fc / value_fc1 / value_fc2:
                    the trained head layers attached to `HexTacToeNet` — passed by
                    reference so state-dict keys stay rooted at the parent module.

    Returns:
        `(log_policy, value, value_aux)` per-window:
          * log_policy:  `(N, n_actions)` log-softmax probabilities.
          * value:       `(N, 1)` scalar in [-1, 1] (scalar = tanh(v_logit);
                         dist65 = E[softmax(bin_logits) · support]).
          * value_aux:   scalar → `(N, 1)` raw pre-tanh v_logit; dist65 → `(N, 65)`
                         bin logits.
    """
    # Policy branch — 1×1 conv → ReLU → flatten → FC → log_softmax.
    p = F.relu(policy_conv(out))
    p = p.flatten(1)
    p_logits = policy_fc(p)
    log_policy = F.log_softmax(p_logits, dim=1)

    # Value branch — global avg + max pool → cat → FC → ReLU → FC → tanh.
    v_avg = out.mean(dim=(2, 3))
    v_max = out.amax(dim=(2, 3))
    v = torch.cat([v_avg, v_max], dim=1)
    v = F.relu(value_fc1(v))
    if value_head_type == "dist65":
        assert value_fc2_bins is not None, "value_fc2_bins required for dist65 head"
        bin_logits = value_fc2_bins(v)                    # (N, 65)
        value = decode_binned_value(bin_logits)           # (N, 1) in [-1,1]
        return log_policy, value, bin_logits
    v_logit = value_fc2(v)
    value = torch.tanh(v_logit)
    return log_policy, value, v_logit


class MinMaxPool(nn.Module):
    """Stateless K-cluster reduction: value=min(K) + policy=scatter-max(K) in prob
    space, mirroring the engine `records::aggregate_policy` semantics. No learnable
    parameters; cluster-token features are accepted but ignored."""

    def forward(
        self,
        cluster_tokens: torch.Tensor,        # (B, K, dim) — accepted, unused
        per_cluster_logits: torch.Tensor,    # (B, K, n_actions) raw logits
        per_cluster_values: torch.Tensor,    # (B, K, 1)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del cluster_tokens  # min/max pools the per-cluster head outputs, not tokens.
        # `per_cluster_values` are post-tanh; pool them and recover a pre-tanh proxy
        # via atanh (clamped for well-behaved gradients).
        value = per_cluster_values.min(dim=1).values             # (B, 1)
        value_logit = torch.atanh(value.clamp(-0.999999, 0.999999))

        # Policy: per-cluster softmax → max across K → renormalise → log.
        probs_K = F.softmax(per_cluster_logits, dim=-1)          # (B, K, A)
        max_probs = probs_K.max(dim=1).values                    # (B, A)
        max_probs = max_probs.clamp_min(1e-12)
        max_probs = max_probs / max_probs.sum(dim=-1, keepdim=True)
        log_policy = max_probs.log()                             # (B, A)
        return log_policy, value, value_logit


# ── aux-head factories (state-dict keys root at the parent attribute) ──────────


def make_value_var(filters: int) -> nn.Sequential:
    """Value-uncertainty head (training only) — predicts squared error of value.
    Keys: `value_var.2.{weight,bias}` (Linear at index 2)."""
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(filters, 1),
        nn.Softplus(),
    )


def make_ownership_head(filters: int) -> nn.Sequential:
    """Ownership head (training only) — per-cell stone affiliation ∈ (-1, 1).
    Keys: `ownership_head.0.{weight,bias}`."""
    return nn.Sequential(
        nn.Conv2d(filters, 1, kernel_size=1),
        nn.Tanh(),
    )


def make_threat_head(filters: int) -> nn.Conv2d:
    """Threat head (training only) — per-cell winning-line membership logits.
    Keys: `threat_head.{weight,bias}`."""
    return nn.Conv2d(filters, 1, kernel_size=1)


def make_chain_head(filters: int) -> nn.Conv2d:
    """Q13-aux chain-length head (training only) — predicts the 6 chain-length
    planes from trunk features (smooth-L1 in the trainer). The surviving Q13 form.
    Keys: `chain_head.{weight,bias}`."""
    return nn.Conv2d(filters, 6, kernel_size=1)


def make_ply_index_head(filters: int) -> nn.Sequential:
    """Ply-index head (training only) — predicts normalized position index ∈ [0, 1].
    Keys: `ply_index_head.2.{weight,bias}`, `ply_index_head.4.{weight,bias}`."""
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(filters, 64),
        nn.ReLU(inplace=True),
        nn.Linear(64, 1),
        nn.Sigmoid(),
    )
