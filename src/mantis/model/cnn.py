# >300 (LAW-17): the single CNN control net (trunk + SE/Residual blocks + policy/value/
# aggregated-K heads + 5 aux heads + input_channels knob) is one cohesive module; a
# split would scatter one net's construction+forward across files with no reader benefit.
"""HexTacToeNet — ResNet backbone (SE blocks) + dual-pool value head + policy head
+ opponent-reply and KataGo-style aux heads, on the reachable min/max K-cluster path.

Grid architecture (e.g. v6: 8-plane × 19×19):
  Input:  (B, in_channels, H, W) tensor.
  Trunk:  Conv → GN → ReLU → res_blocks × ResidualBlock(SE).
  Policy: Conv2d(filters→2, 1×1) → ReLU → FC(2·H·W → H·W+1) → log_softmax.
  Value:  GAP+max → FC(2C→256) → ReLU → FC(256→1) → Tanh  (scalar), or → 65-bin
          logits decoded to E[softmax·support] (dist65).
  Aux (training only): opp_reply, value_var, ownership, threat, chain (Q13), ply_index.

The v8 / PMA-pool / gpool-bias / canvas-realness branches of the old net are
STRIPPED: every registered grid encoding is `has_pass_slot=true`, `pool_type=min_max`,
so those branches are unreachable AND consumer-less (F-03/F-04/F-05/F-13). On the one
reachable path the ported forward runs byte-identical torch ops to the old min/max
path (NN-forward parity oracle). Head layers stay direct attributes so state-dict
keys are byte-identical to the old min/max net; the head math lives in `cnn_heads`.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from mantis.model.arch import CnnArch
from mantis.model.cnn_heads import (
    MinMaxPool,
    make_chain_head,
    make_ownership_head,
    make_ply_index_head,
    make_threat_head,
    make_value_var,
    min_max_window_head,
)
from mantis.model.dist65 import N_VALUE_BINS

_log = logging.getLogger(__name__)

MODEL_GN_GROUPS = 8  # GroupNorm group count; filters must be divisible by this
_GN_GROUPS = MODEL_GN_GROUPS

# Required wire planes for the v6-family `input_channels` subset knob — every
# variant must include these or the model has no stone information (plane 0 = cur
# ply-0, plane 4 = opp ply-0 in the 8-plane HEXB v6 wire format).
_REQUIRED_INPUT_CHANNELS: tuple[int, ...] = (0, 4)
_WIRE_CHANNELS: int = 8  # HEXB v6 wire format plane count (the input_channels domain)


def _validate_input_channels(channels: Sequence[int]) -> list[int]:
    """Validate a variant's `input_channels` list; fail loudly on misconfig."""
    if not isinstance(channels, (list, tuple)):
        raise ValueError(
            f"input_channels must be a list/tuple, got {type(channels).__name__}."
        )
    canon: list[int] = []
    for i, c in enumerate(channels):
        try:
            ci = int(c)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"input_channels[{i}] is not an integer: {c!r}. {exc}") from exc
        if ci < 0 or ci >= _WIRE_CHANNELS:
            raise ValueError(
                f"input_channels[{i}]={ci} out of range [0, {_WIRE_CHANNELS})."
            )
        if ci in canon:
            raise ValueError(f"input_channels has duplicate index {ci}.")
        canon.append(ci)
    for required in _REQUIRED_INPUT_CHANNELS:
        if required not in canon:
            raise ValueError(
                f"input_channels missing required plane {required} "
                f"(plane 0 = cur ply-0, plane 4 = opp ply-0). Configured: {canon}."
            )
    return canon


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""

    def __init__(self, channels: int, reduction_ratio: int = 4) -> None:
        super().__init__()
        mid = max(channels // reduction_ratio, 1)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        s = x.mean(dim=(2, 3))              # (B, C) — squeeze
        s = F.relu(self.fc1(s))              # (B, C//r)
        s = torch.sigmoid(self.fc2(s))       # (B, C)
        return x * s.view(b, c, 1, 1)       # scale


class ResidualBlock(nn.Module):
    def __init__(self, filters: int, se_reduction_ratio: int = 4) -> None:
        super().__init__()
        assert filters % _GN_GROUPS == 0, (
            f"filters={filters} must be divisible by num_groups={_GN_GROUPS}"
        )
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(_GN_GROUPS, filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(_GN_GROUPS, filters)
        self.se = SEBlock(filters, se_reduction_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + residual)


class Trunk(nn.Module):
    """ResNet trunk: input conv → GN → ReLU → `nn.Sequential` of `ResidualBlock`s.

    Sequential-only (the gpool-splice ModuleList path is killed with v8), so the
    v6-family state_dicts load byte-exact (`trunk.tower.N.*`)."""

    def __init__(
        self,
        in_channels: int,
        filters: int,
        res_blocks: int,
        se_reduction_ratio: int = 4,
    ) -> None:
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, filters, 3, padding=1, bias=False)
        self.input_gn = nn.GroupNorm(_GN_GROUPS, filters)
        self.tower = nn.Sequential(
            *[ResidualBlock(filters, se_reduction_ratio) for _ in range(res_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.input_gn(self.input_conv(x)))
        return self.tower(out)


class HexTacToeNet(nn.Module):
    """ResNet trunk (min/max K-cluster grid net), constructed from a declared
    `CnnArch`. State-dict keys are byte-identical to the old min/max net."""

    def __init__(self, arch: CnnArch) -> None:
        super().__init__()
        board_size = int(arch.board_size)
        filters = int(arch.filters)
        res_blocks = int(arch.res_blocks)
        se_reduction_ratio = int(arch.se_reduction_ratio)
        value_head_type = arch.value_head_type
        n_value_bins = int(arch.n_value_bins)
        spatial = board_size * board_size

        self.board_size = board_size
        self.filters = filters
        self.res_blocks = res_blocks
        self.value_head_type = value_head_type

        # `input_channels`: the v6-family plane-subset knob (state-dict-neutral when
        # None). When set, forward slices `x[:, input_channels]` before the trunk.
        if arch.input_channels is not None:
            channels = _validate_input_channels(list(arch.input_channels))
            if int(arch.in_channels) != len(channels):
                raise ValueError(
                    f"in_channels={arch.in_channels} disagrees with "
                    f"len(input_channels)={len(channels)}."
                )
            self._input_channels: list[int] | None = list(channels)
            self.register_buffer(
                "input_channel_index",
                torch.tensor(channels, dtype=torch.long),
                persistent=True,
            )
            self.in_channels = len(channels)
        else:
            self._input_channels = None
            self.input_channel_index = None  # type: ignore[assignment]
            self.in_channels = int(arch.in_channels)

        # Every registered grid encoding has a pass slot → n_actions = spatial + 1.
        self.n_actions: int = spatial + 1

        self.trunk = Trunk(
            in_channels=self.in_channels,
            filters=filters,
            res_blocks=res_blocks,
            se_reduction_ratio=se_reduction_ratio,
        )

        # Policy + opponent-reply heads (FC head — no normalization: 2 channels).
        self.policy_conv = nn.Conv2d(filters, 2, 1)
        self.policy_fc = nn.Linear(2 * spatial, spatial + 1)
        self.opp_reply_conv = nn.Conv2d(filters, 2, 1)
        self.opp_reply_fc = nn.Linear(2 * spatial, spatial + 1)

        # Value head — global avg+max pooling.
        _VALID_VH_TYPES = {"scalar", "dist65"}
        if value_head_type not in _VALID_VH_TYPES:
            raise ValueError(f"value_head_type={value_head_type!r} not in {_VALID_VH_TYPES}")
        self.value_fc1 = nn.Linear(2 * filters, 256)
        self.value_fc2 = nn.Linear(256, 1)
        if value_head_type == "dist65":
            if n_value_bins != N_VALUE_BINS:
                raise ValueError(
                    f"n_value_bins={n_value_bins} != {N_VALUE_BINS}; the dist65 head "
                    "is 65-fixed (VALUE_SUPPORT is linspace(-1,1,65))."
                )
            self.value_fc2_bins: nn.Linear | None = nn.Linear(256, n_value_bins)
        else:
            self.value_fc2_bins = None

        # Aux heads (training only — never called from inference/MCTS).
        self.value_var = make_value_var(filters)
        self.ownership_head = make_ownership_head(filters)
        self.threat_head = make_threat_head(filters)
        self.chain_head = make_chain_head(filters)      # the surviving Q13 form
        self.ply_index_head = make_ply_index_head(filters)

    def forward(
        self,
        x: torch.Tensor,
        aux: bool = False,
        uncertainty: bool = False,
        ownership: bool = False,
        threat: bool = False,
        chain: bool = False,
        ply_index: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Base return (all flags False) — 3-tuple `(log_policy, value, value_logit)`.
        Additional outputs appended in order: opp_reply, sigma2, ownership_pred,
        threat_pred, chain_pred, ply_pred. Never pass the flags from inference/MCTS."""
        if self._input_channels is not None:
            assert self.input_channel_index is not None
            x = x.index_select(1, self.input_channel_index)

        out = self.trunk(x)

        log_policy, value, v_logit = min_max_window_head(
            out,
            policy_conv=self.policy_conv,
            policy_fc=self.policy_fc,
            value_fc1=self.value_fc1,
            value_fc2=self.value_fc2,
            value_head_type=self.value_head_type,
            value_fc2_bins=self.value_fc2_bins,
        )

        extras: list[torch.Tensor] = []
        if aux:
            o = F.relu(self.opp_reply_conv(out))
            o = o.flatten(1)
            extras.append(F.log_softmax(self.opp_reply_fc(o), dim=1))
        if uncertainty:
            extras.append(self.value_var(out))
        if ownership:
            extras.append(self.ownership_head(out))
        if threat:
            extras.append(self.threat_head(out))
        if chain:
            extras.append(self.chain_head(out))
        if ply_index:
            extras.append(self.ply_index_head(out))

        if not extras:
            return log_policy, value, v_logit
        return (log_policy, value, v_logit, *extras)

    @torch.no_grad()
    def aggregated_forward_K(
        self, x_K: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """K>1 inference entry point: run the per-cluster head then the model-side
        min/max reduction (scatter-max-in-prob-space + value-min), returning the
        standard 3-tuple with leading dim 1 (the aggregated sample)."""
        if self._input_channels is not None:
            assert self.input_channel_index is not None
            x_K = x_K.index_select(1, self.input_channel_index)

        out = self.trunk(x_K)                                     # (K, C, H, W)
        per_logp, per_val, _per_vlogit = min_max_window_head(
            out,
            policy_conv=self.policy_conv,
            policy_fc=self.policy_fc,
            value_fc1=self.value_fc1,
            value_fc2=self.value_fc2,
            value_head_type=self.value_head_type,
            value_fc2_bins=self.value_fc2_bins,
        )
        pool = MinMaxPool()
        return pool(
            cluster_tokens=out.mean(dim=(2, 3)).unsqueeze(0),    # (1, K, C) — unused
            per_cluster_logits=per_logp.unsqueeze(0),             # (1, K, A)
            per_cluster_values=per_val.unsqueeze(0),              # (1, K, 1)
        )


def compile_model(model: HexTacToeNet, mode: str = "default") -> HexTacToeNet:
    try:
        if "TORCHINDUCTOR_CACHE_DIR" not in os.environ:
            cache_dir = Path(".torchinductor-cache").resolve()
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
        compiled = torch.compile(model, mode=mode)  # pyright: ignore[reportUnknownMemberType]
        _log.info("torch.compile applied successfully (mode=%s)", mode)
        return compiled  # type: ignore[return-value]
    except Exception as exc:  # noqa: BLE001 — compile is best-effort; fall back to eager
        _log.warning("torch.compile failed, continuing without compilation: %s", exc)
        return model
