"""Canonical model identity: the parameter hash that denominates a constructed net."""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import torch


def net_param_hash(model: torch.nn.Module) -> str:
    """Return the SHA-256 over a constructed net's parameters, post-seed and pre-play.

    Sorted by name so state_dict ordering is not load-bearing, and over raw parameter bytes
    with shape and dtype, so a differing dtype or layout is still a differing hash.

    Args:
        model: the module whose `state_dict()` is the identity being denominated.

    Returns:
        The hex digest, stable across processes for a net built from the same seed.
    """
    return state_dict_param_hash(model.state_dict())


def state_dict_param_hash(state: Mapping[str, Any]) -> str:
    """Return the same hash as `net_param_hash`, over a state dict not attached to a module.

    This is the one denomination of "are these the same weights?", so a launch-anchor pin and
    a sweep's reported hash are comparable. `_orig_mod.` / `module.` wrapper prefixes are
    canonicalised away, since a `torch.compile` or DDP wrapper is not a different net.

    Args:
        state: a `name -> tensor` mapping (a `state_dict()`, or one loaded from disk).

    Returns:
        The hex digest, equal to `net_param_hash` of any module with these weights.
    """
    digest = hashlib.sha256()
    for canon, raw in sorted((_canonical_key(k), k) for k in state):
        tensor = state[raw]
        digest.update(canon.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _canonical_key(key: str) -> str:
    """Strip `torch.compile` / DDP wrapper prefixes so wrapped and unwrapped copies hash equal."""
    changed = True
    while changed:
        changed = False
        for prefix in ("_orig_mod.", "module."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key
