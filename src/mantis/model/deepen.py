"""Depth growth of a graph net: GINE layers appended after the parent's, each the identity at birth, the parent's function kept (docs/design/DEEPEN_2026-10-23.md)."""
from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping

import torch
import torch.nn as nn
from torch import Tensor

from mantis.model.arch import ModelArch
from mantis.model.build import build_net

__all__ = ["DeepenError", "deepen_arch", "deepen_net", "deepen_state", "deepen_tag"]

_LAYER = re.compile(r"representation\.(convs|norms)\.(\d+)\.(.+)")
#: Every tensor a layer has: one the rules do not name is refused, so a new module cannot deepen silently.
_LAYER_PARTS = frozenset({("convs", f"{m}.{p}") for m in ("lin", "nn.0", "nn.2") for p in ("weight", "bias")}
                         | {("convs", "eps"), ("norms", "weight"), ("norms", "bias")})
#: A new layer's last message linear starts at zero, so its conv adds nothing and the residual's ReLU returns x
#: unchanged: every layer's output is already a ReLU's.
_SILENT = frozenset({("convs", "nn.2.weight"), ("convs", "nn.2.bias")})
#: The tensors that read the JK concat, as whole groups of `layers · hidden` columns (V2's value head reads mean ⊕ max).
_JK_READERS = re.compile(r"(policy_head|aux_policy_head|future_head)\.mlp\.0\.weight|value_head\.fc1\.weight")
_CARRIED = re.compile(r"representation\.(input_proj|edge_proj|final_norm)\.(weight|bias)"
                      r"|(policy_head|aux_policy_head|future_head)\.mlp\.(0\.bias|2\.weight|2\.bias)"
                      r"|value_head\.(fc1\.bias|fc2_bins\.(weight|bias))")


class DeepenError(ValueError):
    """A tensor the deepening has no rule for, a parent that is not the declared depth or shape, or a depth that adds no layer."""


def deepen_arch(arch: ModelArch, layers: int) -> ModelArch:
    """The same arch kind `layers` GINE layers deep, every other field as declared; Raises: DeepenError — `layers` adds no layer."""
    if layers <= arch.num_layers:
        raise DeepenError(f"{layers} layers adds none to the parent's {arch.num_layers}: deepening appends layers")
    return dataclasses.replace(arch, num_layers=layers)


def deepen_tag(parent_layers: int, layers: int, *, known_bad: bool = False) -> str:
    """The lineage tag a deepened stamp carries after its parent's lineage; the known-bad's says so, so it cannot pass for a parent."""
    return f"+deepen{'-known-bad' if known_bad else ''}({parent_layers}->{layers})"


def _jk_reader(key: str, t: Tensor, init: Tensor, old: int, known_bad: bool) -> Tensor:
    """The parent's reader `t` with each group's new JK columns zero (`init`'s fresh draw under `known_bad`)."""
    groups = t.shape[1] // old if t.dim() == 2 and t.shape[1] % old == 0 else 0
    if not groups or t.shape[0] != init.shape[0] or init.shape[1] % groups or init.shape[1] // groups <= old:
        raise DeepenError(f"{key}: {tuple(t.shape)} is not whole JK groups of {old} under the deeper {tuple(init.shape)}")
    wide = init.shape[1] // groups
    out = init.clone() if known_bad else torch.zeros_like(init)
    for g in range(groups):
        out[:, g * wide:g * wide + old] = t[:, g * old:(g + 1) * old]
    return out


def deepen_state(parent: Mapping[str, Tensor], fresh: Mapping[str, Tensor], *, hidden: int, layers: int,
                 known_bad: bool = False) -> dict[str, Tensor]:
    """`fresh`, a deeper net's initial state, carrying the parent's `layers` layers and heads, each new layer's last message linear and JK reader columns zero; `known_bad` keeps their fresh draws; Raises: DeepenError — a key without a rule, a trunk not `hidden` wide, a parent not `layers` deep or not of the deeper net's shape."""
    for name, state in (("parent", parent), ("deeper net", fresh)):
        proj = state.get("representation.input_proj.weight")
        if proj is None or proj.shape[0] != hidden:
            raise DeepenError(f"the {name}'s trunk is not {hidden} wide: input_proj {None if proj is None else tuple(proj.shape)}")
    deep: dict[str, Tensor] = {}
    for key in sorted(fresh):
        init, m = fresh[key].detach().cpu(), _LAYER.fullmatch(key)
        known = (m[1], m[3]) in _LAYER_PARTS if m else bool(_JK_READERS.fullmatch(key) or _CARRIED.fullmatch(key))
        if not known:
            raise DeepenError(f"no deepening rule for {key!r}: a module the deepening does not know")
        if m is not None and int(m[2]) >= layers:
            if key in parent:
                raise DeepenError(f"{key}: the parent already has layer {m[2]}, so it is not {layers} layers deep")
            deep[key] = torch.zeros_like(init) if (m[1], m[3]) in _SILENT and not known_bad else init.clone()
            continue
        if key not in parent:
            raise DeepenError(f"{key}: the parent has no such tensor, so it is not {layers} layers of this kind")
        t = parent[key].detach().cpu()
        if _JK_READERS.fullmatch(key):
            deep[key] = _jk_reader(key, t, init, layers * hidden, known_bad)
        elif t.shape != init.shape:
            raise DeepenError(f"{key}: the parent's {tuple(t.shape)} is not the deeper net's {tuple(init.shape)}")
        else:
            deep[key] = t.clone()
    stray = sorted(set(parent) - set(fresh))
    if stray:
        raise DeepenError(f"the parent carries {stray}, which the deeper net has no place for")
    return deep


def deepen_net(net: nn.Module, arch: ModelArch, *, layers: int, seed: int, known_bad: bool = False) -> nn.Module:
    """`net`, declared as `arch`, grown to `layers` GINE layers in a CPU net of the same kind in eval mode, the new layers' fresh draws from `seed`; Raises: DeepenError — as `deepen_arch` and `deepen_state`; RepresentationMismatch — `arch` is no kind `build_net` has; RuntimeError — the deeper state does not load strictly."""
    deep_arch = deepen_arch(arch, layers)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)  # not torch.manual_seed: that also reseeds CUDA, which the fork leaves
        deep = build_net(deep_arch)
    deep.load_state_dict(deepen_state(net.state_dict(), deep.state_dict(), hidden=arch.hidden, layers=arch.num_layers,
                                      known_bad=known_bad), strict=True)
    return deep.eval()
