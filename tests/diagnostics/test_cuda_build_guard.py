"""R328(e) — the CPU-only-torch guard, and the matmul that is the actual proof.

WHY THIS SUITE EXISTS. `uv sync` on the GPU box replaces a CUDA torch with `2.11.0+cpu`,
because `pyproject.toml` pins torch to the PyTorch CPU wheel index. The MINT-CLOSE session hit
it, repaired it by hand, and verified the repair with a real matmul rather than with
`torch.cuda.is_available()`. R328(e) turns that into mechanism.

The suite runs entirely on a CPU-only host: the refusal arm is this host's real state, and the
CUDA arms drive the REAL residual arithmetic with `Tensor.cuda` monkeypatched, so what is
exercised is the tolerance and the comparison rather than a mock of them.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import torch

from mantis.diagnostics import cuda_build_guard as guard

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "mantis.diagnostics.cuda_build_guard", *args],
        capture_output=True, text=True, check=False)


# ═══ the refusal arm — this host's real state ════════════════════════════════════════════
@pytest.mark.skipif(torch.version.cuda is not None,
                    reason="loud skip: this host has a CUDA torch build, so the CPU-only "
                           "refusal arm has no subject here")
def test_r328e_01_a_cpu_only_build_refuses_by_name_and_names_the_recurring_cause() -> None:
    """A `+cpu` wheel refuses, and the refusal says WHY it will happen again.

    A refusal that says only "no CUDA" sends the reader to the driver. The cause here is a
    committed index pin, so the message names it: the next `uv sync` undoes the repair."""
    with pytest.raises(guard.CudaBuildRefusal) as excinfo:
        guard.assert_cuda_build()
    message = str(excinfo.value)
    assert "CPU-ONLY" in message, f"the refusal does not name the build class: {message!r}"
    assert "pyproject" in message, (
        f"the refusal does not name the recurring cause, so a reader repairs it once and is "
        f"surprised by the next sync: {message!r}")


@pytest.mark.skipif(torch.version.cuda is not None,
                    reason="loud skip: CUDA build present; the CLI refusal arm has no subject")
def test_r328e_02_the_cli_exits_2_and_prints_no_pass_line() -> None:
    """Exit 2 (fusion_calibrate's named-refusal code), and nothing that reads as a green."""
    proc = _run()
    assert proc.returncode == 2, f"expected 2, got {proc.returncode}\n{proc.stdout}{proc.stderr}"
    assert "REFUSED" in proc.stderr, f"the refusal is not named: {proc.stderr!r}"
    assert "PASS" not in proc.stdout, f"a refusing run printed a pass line: {proc.stdout!r}"


@pytest.mark.skipif(torch.version.cuda is not None,
                    reason="loud skip: CUDA build present; the JSON refusal arm has no subject")
def test_r328e_03_json_refusal_carries_the_verdict_and_the_build() -> None:
    """`--json` refusals are machine-readable, so the box procedure can capture the state."""
    proc = _run("--json")
    assert proc.returncode == 2
    payload = json.loads(proc.stderr)
    assert payload["verdict"] == "REFUSED"
    assert payload["cuda_toolkit"] is None
    assert payload["torch_version"] == torch.__version__


# ═══ the matmul arm — the half `is_available()` cannot answer ════════════════════════════
def _patch_cuda_transport(monkeypatch: pytest.MonkeyPatch, perturbation: float) -> None:
    """Make `Tensor.cuda()` a CPU identity, optionally corrupting one operand's transport."""
    state = {"calls": 0}

    def fake_cuda(self: torch.Tensor, *_args: object, **_kwargs: object) -> torch.Tensor:
        state["calls"] += 1
        return self + perturbation if state["calls"] == 1 else self.clone()

    monkeypatch.setattr(torch.Tensor, "cuda", fake_cuda, raising=False)


def test_r328e_04_a_wrong_cuda_result_refuses_even_though_the_build_check_passed(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """THE MUTATION SELF-TEST (LAW-07): the guard's value is exactly this row.

    The build half is forced to pass, so the only thing standing between a broken card and a
    green is the arithmetic. A guard resting on `torch.cuda.is_available()` reports green here.
    """
    monkeypatch.setattr(guard, "assert_cuda_build", lambda: {"torch_version": "fake"})
    _patch_cuda_transport(monkeypatch, perturbation=1.0)
    with pytest.raises(guard.CudaBuildRefusal, match="disagrees with the CPU reference"):
        guard.assert_cuda_matmul(n=64)


def test_r328e_05_a_correct_cuda_result_passes_and_reports_its_residual(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The positive control, without which row 04 could pass on a guard that always raises."""
    _patch_cuda_transport(monkeypatch, perturbation=0.0)
    residual = guard.assert_cuda_matmul(n=64)
    assert residual == 0.0, f"an unperturbed transport should be exact, got {residual}"


def test_r328e_06_the_tolerance_is_load_bearing_on_both_sides(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A residual just inside `atol` passes and one just outside refuses.

    Pins that `atol` is a comparison and not decoration — an `atol` the code never reads would
    leave rows 04 and 05 both passing."""
    _patch_cuda_transport(monkeypatch, perturbation=0.0)
    assert guard.assert_cuda_matmul(n=8, atol=1e-6) == 0.0
    _patch_cuda_transport(monkeypatch, perturbation=0.5)
    with pytest.raises(guard.CudaBuildRefusal):
        guard.assert_cuda_matmul(n=8, atol=1e-6)


def test_r328e_07_a_raising_cuda_op_refuses_rather_than_propagating(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A driver-level RuntimeError becomes the guard's named refusal, not a stack trace."""
    def boom(self: torch.Tensor, *_a: object, **_k: object) -> torch.Tensor:
        raise RuntimeError("CUDA error: no kernel image is available for execution")
    monkeypatch.setattr(torch.Tensor, "cuda", boom, raising=False)
    with pytest.raises(guard.CudaBuildRefusal, match="raised rather than computing"):
        guard.assert_cuda_matmul(n=8)


def test_r328e_08_a_zero_device_build_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    """`cuda_available` True with `device_count` 0 is a real state and it is not a green."""
    monkeypatch.setattr(guard, "torch_build", lambda: {
        "torch_version": "2.11.0+cu128", "cuda_toolkit": "12.8",
        "cuda_available": True, "device_count": 0})
    with pytest.raises(guard.CudaBuildRefusal, match="device_count 0"):
        guard.assert_cuda_build()


# ═══ the claim the refusal makes about the tree ══════════════════════════════════════════
def test_r328e_09_the_refusals_pyproject_claim_is_true_of_the_shipped_pyproject() -> None:
    """The message says the pin causes this. Derived from the file, so it cannot become a lie.

    Read STRUCTURALLY out of `[tool.uv.sources]` / `[[tool.uv.index]]` rather than grepped for
    a string, so a rename of the index does not quietly pass (R296(f))."""
    data = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    index_name = data["tool"]["uv"]["sources"]["torch"]["index"]
    urls = {entry["name"]: entry["url"] for entry in data["tool"]["uv"]["index"]}
    assert index_name in urls, f"torch names index {index_name!r}, which is not declared"
    assert urls[index_name].rstrip("/").endswith("/cpu"), (
        f"the guard's refusal blames a CPU wheel index; torch resolves to {urls[index_name]!r}. "
        "If this pin has moved, the refusal message must move with it.")
