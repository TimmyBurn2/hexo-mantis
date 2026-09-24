"""Seeding, as a leaf: `random`, `numpy` and `torch` from one integer seed, never at import time.

A run's ONE boot site is `mantis.run.build_run_collaborators`, its first statement, before any
RNG-consuming object exists.

It lives in `mantis.util` because `mantis.diagnostics.worker_sweep` also seeds and is
structurally barred from importing anything under `mantis.train`.
"""
from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed `random`, `numpy` and `torch` (CPU and all CUDA devices) from one integer seed.

    Idempotent, but call it exactly once per real boot, before any model, optimizer or other
    RNG-consuming object is constructed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


__all__ = ["seed_everything"]
