"""R347(d) — the two START pre-flight HALTs, proven to fire and proven not to fire vacuously.

Both are decided BEFORE the boot, in `preflight_mint._assert_start_halts`, and both are named
outcomes with their own rc rather than a line in a green report. The CUDA halt is conditioned on
what the RUN declares, never on sniffing the host: this box carries an NVIDIA card AND a
deliberate `+cpu` wheel for the WP9 forward-parity regime, so host GPU presence would be the
wrong predicate and would red a dev machine that is behaving correctly.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from mantis.config.loader import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
GATES = REPO_ROOT / "tools" / "ci_gates"


def _tool() -> object:
    spec = importlib.util.spec_from_file_location("_start_halts_tool", GATES / "preflight_mint.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = _tool()


def _config(name: str = "run6.yaml"):
    return load_config(REPO_ROOT / "configs" / name)


def test_an_ephemeral_run_directory_HALTS_with_its_own_rc(tmp_path: Path) -> None:
    """THE PIN. `workspace_is_volume = false` used to be a note; it is now rc 16 before the
    boot, so the run that would have been lost never starts."""
    config = _config()
    mounts = tmp_path / "mounts"
    mounts.write_text(f"dev0 / ext4 rw 0 0\ndev1 {tmp_path} tmpfs rw 0 0\n", encoding="utf-8")
    from mantis.diagnostics import workspace_durability

    original = workspace_durability.MOUNTS
    workspace_durability.MOUNTS = mounts
    try:
        with pytest.raises(TOOL.PreflightWorkspaceNotDurableError) as caught:
            TOOL._assert_start_halts(config, tmp_path / "run", {})
    finally:
        workspace_durability.MOUNTS = original
    assert caught.value.rc == 16, f"the named outcome is rc 16; got {caught.value.rc}"
    assert "tmpfs" in str(caught.value)


def test_a_durable_run_directory_records_its_evidence_and_does_not_halt(tmp_path: Path) -> None:
    """The control: without it the halt could be unconditional, and the report would carry no
    statement of WHICH mount was accepted."""
    config = _config()
    mounts = tmp_path / "mounts"
    mounts.write_text(f"dev0 / ext4 rw 0 0\ndev1 {tmp_path} xfs rw 0 0\n", encoding="utf-8")
    from mantis.diagnostics import workspace_durability

    original = workspace_durability.MOUNTS
    workspace_durability.MOUNTS = mounts
    report: dict = {}
    try:
        with pytest.raises(TOOL.PreflightCudaBuildError):
            TOOL._assert_start_halts(config, tmp_path / "run", report)
    finally:
        workspace_durability.MOUNTS = original
    assert report["workspace"]["verdict"] == "DURABLE"
    assert report["workspace"]["fstype"] == "xfs"


@pytest.mark.skipif(torch.version.cuda is not None,
                    reason="loud skip: this host has a CUDA torch build, so the refusal arm "
                           "has no subject here")
def test_a_cuda_config_on_a_cpu_torch_HALTS_with_its_own_rc(tmp_path: Path) -> None:
    """THE PIN. run6 declares `train.device: cuda`; a `+cpu` wheel cannot run it, and the
    downgrade recurs on every `uv sync` because the index pin is committed."""
    config = _config()
    assert "cuda" in {config.train.device, config.eval.worker_device}, (
        "this pin needs a config that declares cuda; run6 does")
    mounts = tmp_path / "mounts"
    mounts.write_text(f"dev0 / ext4 rw 0 0\ndev1 {tmp_path} xfs rw 0 0\n", encoding="utf-8")
    from mantis.diagnostics import workspace_durability

    original = workspace_durability.MOUNTS
    workspace_durability.MOUNTS = mounts
    try:
        with pytest.raises(TOOL.PreflightCudaBuildError) as caught:
            TOOL._assert_start_halts(config, tmp_path / "run", {})
    finally:
        workspace_durability.MOUNTS = original
    assert caught.value.rc == 17, f"the named outcome is rc 17; got {caught.value.rc}"
    message = str(caught.value)
    assert "cuda" in message and config.run_id in message


def test_a_cpu_config_records_not_run_rather_than_asserting_cuda(tmp_path: Path) -> None:
    """The condition is what the RUN declares. A cpu-declaring config must reach a report line
    saying the CUDA question was not asked — a silent skip and a pass look identical."""
    config = _config().model_copy(
        update={"train": _config().train.model_copy(update={"device": "cpu"}),
                "eval": _config().eval.model_copy(update={"worker_device": "cpu"})})
    mounts = tmp_path / "mounts"
    mounts.write_text(f"dev0 / ext4 rw 0 0\ndev1 {tmp_path} xfs rw 0 0\n", encoding="utf-8")
    from mantis.diagnostics import workspace_durability

    original = workspace_durability.MOUNTS
    workspace_durability.MOUNTS = mounts
    report: dict = {}
    try:
        TOOL._assert_start_halts(config, tmp_path / "run", report)
    finally:
        workspace_durability.MOUNTS = original
    assert report["cuda_build"]["verdict"] == "not_run"
    assert "no cuda device declared" in report["cuda_build"]["reason"]


def test_the_two_halts_hold_distinct_rcs_outside_the_reserved_band() -> None:
    """A halt that shares a code with another outcome is a halt a supervisor cannot read."""
    codes = {TOOL.PreflightWorkspaceNotDurableError.rc, TOOL.PreflightCudaBuildError.rc}
    assert len(codes) == 2
    assert not codes & set(TOOL.RESERVED_CODES), "a START halt must not collide with the run's own"
    others = {cls.rc for name, cls in vars(TOOL).items()
              if name.startswith("Preflight") and name.endswith("Error")
              and isinstance(cls, type) and cls.rc not in codes}
    assert not codes & others, f"a START halt rc collides with another named outcome: {others}"
