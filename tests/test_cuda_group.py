"""Every `cuda`-marked test runs on ONE xdist worker in gates 3a and 3b (an 8 GB card OOMed under four), and an unmarked GPU allocation reds."""
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
    return {p.read_text(encoding="utf-8") for p in tmp_path.glob(f"{prefix}-*")}


def test_every_cuda_test_lands_on_one_worker_and_an_integration_file_stays_on_one(tmp_path: Path) -> None:
    """PLANTED BREAK: drop the hook's cuda group (or its per-file integration group) and those arms land on several workers."""
    proc = _child(tmp_path, "-m", "", "-n", "3", "--dist", "loadgroup", "-k", "records_its_worker")
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-1000:]
    counts = {tag: len(list(tmp_path.glob(f"{tag}-*"))) for tag in ("cuda", "plain", "integration")}
    assert counts == {"cuda": 6, "plain": 12, "integration": 6}, counts
    assert len(_workers(tmp_path, "plain")) > 1, "the plain arms ran on one worker: the child was not parallel"
    assert len(_workers(tmp_path, "cuda")) == 1, f"the cuda arms ran on {_workers(tmp_path, 'cuda')}"
    assert len(_workers(tmp_path, "integration")) == 1, (
        f"one integration file's arms ran on {_workers(tmp_path, 'integration')}: `loadfile`'s locality is lost"
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="LOUD SKIP — an allocation needs a GPU")
def test_an_unmarked_gpu_allocation_reds_the_test(tmp_path: Path) -> None:
    """PLANTED BREAK: drop `_gpu_work_carries_the_cuda_mark` and the unmarked allocation passes."""
    proc = _child(tmp_path, "-k", "unmarked_arm_allocates")
    assert proc.returncode != 0, "an unmarked GPU allocation passed: the guard is gone"
    assert "allocated on the GPU without the `cuda` mark" in proc.stdout, proc.stdout[-2000:]


def test_both_pytest_tiers_distribute_by_group() -> None:
    """The group means one worker only under `--dist loadgroup`, so every parallel tier invocation carries it."""
    for path in ("tools/ci_gates/run_all.sh", "Makefile"):
        for needle in ('-m "not integration and not slow" -n 8', "-m integration -n 4"):
            lines = [ln for ln in (REPO_ROOT / path).read_text(encoding="utf-8").splitlines() if needle in ln]
            assert lines and all("--dist loadgroup" in ln for ln in lines), (path, lines)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="LOUD SKIP — a CUDA context needs a GPU")
def test_collecting_the_tree_creates_no_cuda_context() -> None:
    """PLANTED BREAK: read a device property in a module-level `skipif` and every xdist worker makes a context at collection."""
    probe = ("import sys, pytest, torch; rc = pytest.main(['--collect-only', '-q', '-m', '', '-p', 'no:cacheprovider', 'tests']); "
             "print('CUDA_INITIALISED', torch.cuda.is_initialized()); sys.exit(rc)")
    proc = subprocess.run([sys.executable, "-c", probe], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-1000:]
    assert "CUDA_INITIALISED False" in proc.stdout, proc.stdout[-2000:]
