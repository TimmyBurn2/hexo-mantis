"""Net2Net width doubling of a graph net: trunk units copied, their readers split, the parent's function kept (docs/design/WIDEN_2026-10-09.md)."""
from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping

import torch
import torch.nn as nn
from torch import Tensor

from mantis.model.arch import ModelArch
from mantis.model.build import build_net

__all__ = ["SPLIT_NOISE", "WidenError", "widen_arch", "widen_net", "widen_state"]

#: A weight w reading a copied unit becomes w·(½ + ε) and w·(½ − ε), ε ~ U(−SPLIT_NOISE, +SPLIT_NOISE) per element: the
#: pair sums to w, so no output moves, and the copies' outgoing weights differ, so their gradients differ from step 1.
SPLIT_NOISE = 0.1

_COPY, _KEEP = "copy", "keep"
#: Each tensor's rule as (its output dim, its input dim): `copy` doubles that dim, `split` doubles an input dim that
#: reads copied units, `keep` leaves it. A tensor no rule matches is refused, so a new module cannot widen silently.
_RULES: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = tuple((re.compile(p), r) for p, r in (
    (r"representation\.(input_proj|edge_proj)\.weight", (_COPY, _KEEP)),
    (r"representation\.convs\.\d+\.(lin|nn\.0|nn\.2)\.weight", (_COPY, "split")),
    (r"representation\.(input_proj|edge_proj|convs\.\d+\.(lin|nn\.0|nn\.2))\.bias", (_COPY,)),
    (r"representation\.(norms\.\d+|final_norm)\.(weight|bias)", (_COPY,)),
    (r"representation\.convs\.\d+\.eps", (_KEEP,)),
    (r"(policy_head|aux_policy_head)\.mlp\.0\.weight|value_head\.fc1\.weight", (_KEEP, "split")),
    (r"(policy_head|aux_policy_head)\.mlp\.(0\.bias|2\.weight|2\.bias)|value_head\.(fc1\.bias|fc2_bins\.(weight|bias))",
     (_KEEP,)),
))


class WidenError(ValueError):
    """A tensor the doubling has no rule for, a width it does not read, or a split noise that could flip a copy's sign."""


def widen_arch(arch: ModelArch) -> ModelArch:
    """The same arch kind at twice the trunk width; every other field as declared."""
    return dataclasses.replace(arch, hidden=2 * arch.hidden)


def _rule(key: str) -> tuple[str, ...]:
    for pattern, rule in _RULES:
        if pattern.fullmatch(key):
            return rule
    raise WidenError(f"no doubling rule for {key!r}: a module the widening does not know")


def _copies(blocks: int, hidden: int) -> tuple[Tensor, Tensor]:
    """For `blocks` concatenated hidden vectors doubled to `[x, x]` each: every new index's parent index and its copy's sign."""
    j = torch.arange(blocks * 2 * hidden)
    block, within = j // (2 * hidden), j % (2 * hidden)
    return block * hidden + within % hidden, 1.0 - 2.0 * (within // hidden).to(torch.float64)


def widen_state(state: Mapping[str, Tensor], *, hidden: int, seed: int, noise: float = SPLIT_NOISE) -> dict[str, Tensor]:
    """The parent's state doubled from trunk width `hidden` on CPU, the noise drawn from `seed` over the sorted keys; Raises: WidenError — a key without a rule, a width `hidden` does not divide, `noise` outside [0, ½)."""
    if not 0.0 <= noise < 0.5:
        raise WidenError(f"split noise {noise} must lie in [0, 0.5): a share at or below 0 flips a copy's sign")
    gen = torch.Generator().manual_seed(seed)
    out_rows, _ = _copies(1, hidden)
    wide: dict[str, Tensor] = {}
    for key in sorted(state):
        rule, t = _rule(key), state[key].detach().cpu()
        if rule[0] == _COPY and t.shape[0] != hidden:
            raise WidenError(f"{key}: {t.shape[0]} output units, not the trunk's {hidden}")
        if len(rule) == 1 or rule[1] == _KEEP:
            wide[key] = t.index_select(0, out_rows) if rule[0] == _COPY else t.clone()
            continue
        if t.dim() != 2 or t.shape[1] % hidden:
            raise WidenError(f"{key}: input width {tuple(t.shape)} is not whole blocks of the trunk's {hidden}")
        cols, sign = _copies(t.shape[1] // hidden, hidden)
        eps = (torch.rand(t.shape, generator=gen, dtype=torch.float64) * 2.0 - 1.0) * noise
        w = t.double()[:, cols] * (0.5 + sign[None, :] * eps[:, cols])
        wide[key] = (w.index_select(0, out_rows) if rule[0] == _COPY else w).to(t.dtype)
    return wide


def widen_net(net: nn.Module, arch: ModelArch, *, seed: int, noise: float = SPLIT_NOISE) -> nn.Module:
    """`net`, declared as `arch`, doubled into a CPU net of the same kind in eval mode; Raises: WidenError — as `widen_state`; RepresentationMismatch — `arch` is no kind `build_net` has; RuntimeError — the doubled state does not load strictly."""
    wide = build_net(widen_arch(arch))
    wide.load_state_dict(widen_state(net.state_dict(), hidden=arch.hidden, seed=seed, noise=noise), strict=True)
    return wide.eval()
