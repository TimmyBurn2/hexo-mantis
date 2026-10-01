"""Arms for `tests/test_cuda_group.py`, collected only when a child pytest names this file."""
from __future__ import annotations

import os
from pathlib import Path

import pytest


def _record(tag: str) -> None:
    Path(os.environ["MANTIS_CUDA_ARMS_DIR"], tag).write_text(os.environ.get("PYTEST_XDIST_WORKER", "main"),
                                                     encoding="utf-8")


@pytest.mark.cuda
@pytest.mark.parametrize("i", range(6))
def test_a_cuda_arm_records_its_worker(i: int) -> None:
    _record(f"cuda-{i}")


@pytest.mark.parametrize("i", range(12))
def test_a_plain_arm_records_its_worker(i: int) -> None:
    _record(f"plain-{i}")


def test_an_unmarked_arm_allocates_on_the_gpu() -> None:
    import torch

    torch.zeros(1024, device="cuda")
