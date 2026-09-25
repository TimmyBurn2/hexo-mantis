"""The two START HALTs, proven to fire and not to fire vacuously: the CUDA halt (rc 17) before
the boot, keyed on what the RUN declares; the mirror-receipts halt (rc 16) after it,
on the boot's own bundle and first shard — the volume arm is DELETED."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mantis.config.census import production_configs
from mantis.config.loader import load_config
from mantis.diagnostics import mirror_receipts as D
from mantis.train import bundle_receipts as R
from mantis.util import mirror_receipts as U
from mantis.util.hashing import sha256_file
from _toolpath import load_module_by_path

REPO_ROOT = Path(__file__).resolve().parents[2]
GATES = REPO_ROOT / "tools" / "ci_gates"


def _tool() -> object:
    return load_module_by_path("_start_halts_tool", GATES / "preflight_mint.py")


TOOL = _tool()


def _config(name: str):
    return load_config(REPO_ROOT / "configs" / name)


def _cuda_config():
    """The first production config that declares a cuda device, or None if the census holds none."""
    for path in production_configs(REPO_ROOT):
        config = load_config(path)
        if "cuda" in {config.train.device, config.eval.worker_device}:
            return config
    return None


def _receipt_run_dir(root: Path, run_id: str) -> None:
    checkpoints = root / D.CHECKPOINTS_SUBDIR
    for manifest in D.complete_bundles(checkpoints):
        for path in R.bundle_member_paths(manifest, checkpoints):
            U.write_receipt(path, mirrored_sha256=sha256_file(path),
                            mirrored_bytes=path.stat().st_size, cycle=1, mirror_id="test")
    shard = D.first_closed_shard(root / D.GAMES_SUBDIR, run_id)
    assert shard is not None
    U.write_receipt(shard, mirrored_sha256=sha256_file(shard),
                    mirrored_bytes=shard.stat().st_size, cycle=1, mirror_id="test")


def test_an_unreceipted_run_directory_HALTS_with_its_own_rc(
    tmp_path: Path, synthetic_run_dir,
) -> None:
    """THE PIN: an unreceipted burst is rc 16 — a run whose artifacts would die never starts."""
    config = _config("smoke_preflight_armed.yaml")
    run_dir = synthetic_run_dir(tmp_path / "run", run_id=config.run_id)
    args = SimpleNamespace(receipt_wait_sec=0.0)
    with pytest.raises(TOOL.PreflightMirrorReceiptsError) as caught:
        TOOL._assert_mirror_receipts_halt(config, run_dir, args, {})
    assert caught.value.rc == 16, f"the named outcome is rc 16; got {caught.value.rc}"
    assert "not receipted" in str(caught.value) and "waited 0 s" in str(caught.value)


def test_a_receipted_run_directory_records_its_evidence_and_does_not_halt(
    tmp_path: Path, synthetic_run_dir,
) -> None:
    """The control: receipted, the halt passes and the reading names what was proven."""
    config = _config("smoke_preflight_armed.yaml")
    run_dir = synthetic_run_dir(tmp_path / "run", run_id=config.run_id, step=16, shards=2)
    _receipt_run_dir(run_dir, config.run_id)
    report: dict = {}
    TOOL._assert_mirror_receipts_halt(config, run_dir, SimpleNamespace(receipt_wait_sec=0.0), report)
    workspace = report["workspace"]
    assert workspace["verdict"] == U.MIRRORED_VERDICT
    assert workspace["bundle"]["step"] == 16 and len(workspace["bundle"]["files"]) == 4
    assert workspace["shard"]["name"].endswith("seg0001_2026091201.jsonl")


def test_a_receipt_written_for_other_bytes_still_HALTS(tmp_path: Path, synthetic_run_dir) -> None:
    """Killer: a receipt keyed by name. The ring's receipt certifies other bytes while the bundle
    still verifies against its manifest, so the ONLY thing refusing is the receipt's hash."""
    config = _config("smoke_preflight_armed.yaml")
    run_dir = synthetic_run_dir(tmp_path / "run", run_id=config.run_id)
    _receipt_run_dir(run_dir, config.run_id)
    ring = next((run_dir / D.CHECKPOINTS_SUBDIR).glob("*.ring.bin"))
    U.write_receipt(ring, mirrored_sha256="0" * 64, mirrored_bytes=ring.stat().st_size,
                    cycle=2, mirror_id="test")
    with pytest.raises(TOOL.PreflightMirrorReceiptsError, match="ring.bin.*sha256"):
        TOOL._assert_mirror_receipts_halt(config, run_dir, SimpleNamespace(receipt_wait_sec=0.0), {})


@pytest.mark.skipif(torch.version.cuda is not None,
                    reason="loud skip: this host has a CUDA torch build, so the refusal arm "
                           "has no subject here")
def test_a_cuda_config_on_a_cpu_torch_HALTS_with_its_own_rc() -> None:
    """THE PIN. A production config declares a cuda device; a `+cpu` wheel cannot run it, and the
    downgrade recurs on every bare `uv sync` because the default group is the CPU wheel."""
    config = _cuda_config()
    assert config is not None, "this pin needs a production config that declares cuda"
    with pytest.raises(TOOL.PreflightCudaBuildError) as caught:
        TOOL._assert_cuda_build_halt(config, {})
    assert caught.value.rc == 17, f"the named outcome is rc 17; got {caught.value.rc}"
    message = str(caught.value)
    assert "cuda" in message and config.run_id in message


@pytest.mark.skipif(torch.version.cuda is None,
                    reason="loud skip: this host has no CUDA torch build, so the pass arm has "
                           "no subject here")
def test_a_cuda_config_on_a_cuda_torch_records_the_build_and_does_not_halt() -> None:
    report: dict = {}
    config = _cuda_config()
    assert config is not None, "this pass arm needs a production config that declares cuda"
    TOOL._assert_cuda_build_halt(config, report)
    assert report["cuda_build"]["cuda_available"] is True


def test_a_cpu_config_records_not_run_rather_than_asserting_cuda() -> None:
    """The condition is what the RUN declares. A cpu-declaring config must reach a report line
    saying the CUDA question was not asked — a silent skip and a pass look identical."""
    cuda = _cuda_config()
    assert cuda is not None, "the flip needs a production config that declares cuda to flip"
    config = cuda.model_copy(
        update={"train": cuda.train.model_copy(update={"device": "cpu"}),
                "eval": cuda.eval.model_copy(update={"worker_device": "cpu"})})
    report: dict = {}
    TOOL._assert_cuda_build_halt(config, report)
    assert report["cuda_build"]["verdict"] == "not_run"
    assert "no cuda device declared" in report["cuda_build"]["reason"]


def test_the_two_halts_hold_distinct_rcs_outside_the_reserved_band() -> None:
    """A halt that shares a code with another outcome is a halt a supervisor cannot read."""
    codes = {TOOL.PreflightMirrorReceiptsError.rc, TOOL.PreflightCudaBuildError.rc}
    assert len(codes) == 2
    assert not codes & set(TOOL.RESERVED_CODES), "a START halt must not collide with the run's own"
    others = {cls.rc for name, cls in vars(TOOL).items()
              if name.startswith("Preflight") and name.endswith("Error")
              and isinstance(cls, type) and cls.rc not in codes}
    assert not codes & others, f"a START halt rc collides with another named outcome: {others}"


def test_the_volume_arm_is_gone_from_the_tree() -> None:
    """A mount-table reader anywhere in the shipped package is the deleted arm growing back."""
    hits = [path for path in (REPO_ROOT / "src" / "mantis").rglob("*.py")
            if "/proc/mounts" in path.read_text(encoding="utf-8")]
    assert hits == [], f"the mount-table reader is back: {hits}"
