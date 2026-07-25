"""The ONE determinism boot site (R30a, DESIGN_P2.md §7). Seeds stdlib `random` + `numpy` +
`torch` (CPU and all CUDA devices) from `cfg.seed`, exactly once per boot, at the single
site named below — never at import time (a module-level seed call would fire on every
`import mantis.train.determinism`, including a bare test collection import, which is
exactly the ambient-nondeterminism failure mode R30a exists to close).

Call site: `mantis.run.main()` (`run.py`), immediately after `load_config` succeeds and
before any RNG-consuming object (model/optimizer) is constructed — see `run.py`'s own
docstring note for why this is currently the earliest REAL `cfg.seed`-bearing entry point.
"""
from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed `random`, `numpy`, and `torch` (CPU + all CUDA devices via `torch.manual_seed`,
    per torch's own docs) from a single integer seed. Idempotent — safe to call more than
    once (e.g. a resume boot re-seeding to a reproducible point); the ONE determinism boot
    site (R30a) — call exactly once per real boot, before any model/optimizer/RNG-consuming
    object is constructed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


__all__ = ["seed_everything"]
