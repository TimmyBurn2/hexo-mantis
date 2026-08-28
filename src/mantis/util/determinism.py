"""Seeding, as a leaf. Seeds stdlib `random` + `numpy` + `torch` (CPU and all CUDA devices)
from an integer seed — never at import time (a module-level seed call would fire on every
import, including a bare test collection import, which is exactly the ambient-nondeterminism
failure mode R30a exists to close).

**R30a's ONE BOOT SITE is unchanged and is `mantis.run.build_run_collaborators`** — the FIRST
statement of the ONE collaborator builder, before any RNG-consuming object (model / optimizer /
pool) exists. WPMAIN made that the single site: the launcher and the mint preflight's boot child
used to seed separately at their own entry points, which is two boot sites for a determinism law
that says one. **That rule is about a RUN's boot and still has exactly one site.**

**WHY THIS FILE LIVES IN `mantis.util` AND NOT `mantis.train`, which is where it started.**
`mantis.diagnostics.worker_sweep` must seed the network it builds per rung (F-RESIT-10: unseeded,
every rung of the pre-registered ladder raced a different random net and the knee rule's ranking
column carried an uncontrolled draw). The sweep is guaranteed **trainer-unreachable by import at
any scope** — R309(g), enforced structurally by
`tests/diagnostics/test_worker_sweep_reachability.py` — and importing ANY module under
`mantis.train` executes `mantis/train/__init__.py`, which was **measured** to pull eight training
modules into `sys.modules` (`emit`, `lifecycle`, `disk_guard`, `heartbeat_watchdog`, `signals`,
`watchdog`, …). So the seeding helper could not stay where it was without either weakening that
ban or the sweep growing a second seeding authority.

It moved rather than being duplicated, and it moved HERE because this is where torch-consuming
leaves live: `mantis.util`'s own package docstring keeps `__init__` free of re-exports precisely
so a submodule import does not drag torch into package init, and `mantis.util.device` is the
sibling this file now matches. Seeding was never a training concern; it is a process-global
RNG concern, and the package it sits in now says so.
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
