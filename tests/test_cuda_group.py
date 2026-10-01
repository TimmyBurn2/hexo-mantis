"""Every `cuda`-marked test runs on ONE xdist worker in gate 3a (an 8 GB card OOMed under four), and an unmarked GPU allocation reds."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
ARMS = "tests/_cuda_group_arms.py"


def _child(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """One child pytest over the arms, recording into `tmp_path`."""
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *args, ARMS],
        cwd=REPO_ROOT, env={**os.environ, "MANTIS_CUDA_ARMS_DIR": str(tmp_path)},
        capture_output=True, text=True, check=False,
    )


def _workers(tmp_path: Path, prefix: str) -> set[str]:
    return {p.read_text() for p in tmp_path.glob(f"{prefix}-*")}


def test_every_cuda_test_lands_on_one_worker_under_gate_3a_distribution(tmp_path: Path) -> None:
    """PLANTED BREAK: drop the root conftest's grouping hook and the six cuda arms land on several workers."""
    proc = _child(tmp_path, "-n", "3", "--dist", "loadgroup", "-k", "records_its_worker")
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-1000:]
    assert len(list(tmp_path.glob("cuda-*"))) == 6 and len(list(tmp_path.glob("plain-*"))) == 12
    assert len(_workers(tmp_path, "plain")) > 1, "the plain arms ran on one worker: the child was not parallel"
    assert len(_workers(tmp_path, "cuda")) == 1, f"the cuda arms ran on {_workers(tmp_path, 'cuda')}"


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="LOUD SKIP — an allocation needs a GPU")
def test_an_unmarked_gpu_allocation_reds_the_test(tmp_path: Path) -> None:
    """PLANTED BREAK: drop `_gpu_work_carries_the_cuda_mark` and the unmarked allocation passes."""
    proc = _child(tmp_path, "-k", "unmarked_arm_allocates")
    assert proc.returncode != 0, "an unmarked GPU allocation passed: the guard is gone"
    assert "allocated on the GPU without the `cuda` mark" in proc.stdout, proc.stdout[-2000:]


def test_gate_3a_and_make_test_distribute_by_group() -> None:
    """The group means one worker only under `--dist loadgroup`, so both default-tier invocations carry it."""
    for path, needle in (("tools/ci_gates/run_all.sh", '-m "not integration and not slow" -n 8'),
                         ("Makefile", '-m "not integration and not slow" -n 8')):
        lines = [ln for ln in (REPO_ROOT / path).read_text().splitlines() if needle in ln]
        assert lines and all("--dist loadgroup" in ln for ln in lines), (path, lines)
