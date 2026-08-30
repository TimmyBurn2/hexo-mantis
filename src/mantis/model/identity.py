"""Canonical model identity: the parameter hash that denominates a constructed net.

Promoted out of `mantis.diagnostics.worker_sweep` by R321(d) — a determinism gate, a mint
denomination and a provenance primitive do not live in a diagnostic module.
"""
from __future__ import annotations

import hashlib

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
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
