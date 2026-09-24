"""The one determinism context for test scope; the model and train harnesses re-export it.

Bare-importable from every test directory through the tests-root conftest, so a targeted run
of either harness's suite resolves it without the cross-directory import neither can make.
"""
from __future__ import annotations

import contextlib
import os

import torch


@contextlib.contextmanager
def deterministic_algorithms():
    """TEST SCOPE ONLY: enable `torch.use_deterministic_algorithms(True)` for the block, then
    restore the ambient setting exactly — the mode is process-global, so leaking it would
    silently change the numerics of every sibling test. Nothing in `src/mantis/` calls this.
    `CUBLAS_WORKSPACE_CONFIG` is set if absent and restored.
    """
    was_enabled = torch.are_deterministic_algorithms_enabled()
    had_cublas = "CUBLAS_WORKSPACE_CONFIG" in os.environ
    old_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if not had_cublas:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        torch.use_deterministic_algorithms(True)
        yield
    finally:
        torch.use_deterministic_algorithms(was_enabled)
        if had_cublas:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = old_cublas
        else:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
