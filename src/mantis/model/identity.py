"""Canonical model identity: the parameter hash that denominates a constructed net.

Promoted out of `mantis.diagnostics.worker_sweep` by R321(d) — a determinism gate, a mint
denomination and a provenance primitive do not live in a diagnostic module.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import torch


def net_param_hash(model: torch.nn.Module) -> str:
    """SHA-256 over a constructed net's parameters, POST-SEED PRE-PLAY (R317(c)(i)).

    This is the gate F-RESIT-10's repair claimed and the throughput band could only proxy:
    same seed must mean the SAME net, exactly, not "close on a throughput reading that also
    carries OS scheduling noise". Sorted by name so dict/state_dict ordering is not
    load-bearing; raw parameter bytes, not a repr, so two numerically-identical tensors under
    a different dtype or layout would still be caught by their own dtype/shape difference.

    Args:
        model: the module whose `state_dict()` is the identity being denominated.

    Returns:
        The hex digest, stable across processes for a net built from the same seed.
    """
    return state_dict_param_hash(model.state_dict())


def state_dict_param_hash(state: Mapping[str, Any]) -> str:
    """The SAME hash as `net_param_hash`, over a state dict that is not attached to a module.

    AUDIT-1 F-32. `train/anchor.py` carried a SECOND parameter identity — `state_dict_sha256`,
    canonicalised keys plus raw bytes, no shape and no dtype — and used it for the launch-anchor
    pin. Two functions answering "are these the same weights?" that CANNOT agree by
    construction: a run's `expected_anchor_sha256` and the `net_param_hash` a sweep or an
    acceptance witness reports were never comparable, so the pin could not be cross-checked
    against any recorded observable. One denomination now, and this is the entry point for the
    callers that hold bytes rather than a module.

    `_orig_mod.` / `module.` wrapper prefixes are canonicalised away, because a `torch.compile`
    or DDP wrapper is not a different net — that half of the deleted function was right and is
    kept.

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
    """Strip `torch.compile` / DDP wrapper prefixes so a wrapped and an unwrapped copy of one
    net hash equal."""
    changed = True
    while changed:
        changed = False
        for prefix in ("_orig_mod.", "module."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key
