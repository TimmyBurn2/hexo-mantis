"""R349(b): the puller as a LOCAL loop — a directory stands in for the ssh alias, the same
rsync and receipt writer the operator's machine runs against the box."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from mantis.diagnostics import mirror_receipts as D
from mantis.util import mirror_receipts as U

REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOL = REPO_ROOT / "tools" / "mirror_pull.py"


def _tool():
    spec = importlib.util.spec_from_file_location("mirror_pull", _TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = _tool()


def test_one_cycle_receipts_the_bundle_and_the_closed_shard_beside_the_source(
    tmp_path: Path, synthetic_run_dir,
) -> None:
    source = synthetic_run_dir(tmp_path / "box", shards=2)
    mirror = tmp_path / "mirror"
    summary = TOOL.run_cycle(str(source), mirror, "synth", cycle=1, mirror_id="op")
    assert summary["bundles_receipted"] == [40]
    assert len(summary["shards_receipted"]) == 2
    # The SOURCE (the box) now proves its own artifacts; the mirror holds a readable bundle.
    reading = D.require_mirror_receipts(source, "synth")
    assert reading["verdict"] == U.MIRRORED_VERDICT
    assert D.unreceipted_bundle_steps(source / D.CHECKPOINTS_SUBDIR) == []
    assert D.require_mirror_receipts(mirror, "synth")["bundle"]["step"] == 40
    receipt = U.read_receipt(U.receipt_path_for(
        D.first_closed_shard(source / D.GAMES_SUBDIR, "synth")))
    assert receipt["mirror_id"] == "op" and receipt["cycle"] == 1


def test_a_second_cycle_is_idempotent_and_a_new_bundle_is_picked_up(
    tmp_path: Path, synthetic_run_dir,
) -> None:
    source = synthetic_run_dir(tmp_path / "box")
    mirror = tmp_path / "mirror"
    TOOL.run_cycle(str(source), mirror, "synth", cycle=1, mirror_id="op")
    again = TOOL.run_cycle(str(source), mirror, "synth", cycle=2, mirror_id="op")
    assert again["bundles_receipted"] == [] and again["shards_receipted"] == []
    synthetic_run_dir(tmp_path / "box", step=80)
    third = TOOL.run_cycle(str(source), mirror, "synth", cycle=3, mirror_id="op")
    assert third["bundles_receipted"] == [80]


def test_a_partially_transferred_shard_is_not_receipted(tmp_path: Path, synthetic_run_dir) -> None:
    """Killer: receipting by index row alone certifies a short copy intact."""
    source = synthetic_run_dir(tmp_path / "box")
    mirror = tmp_path / "mirror"
    TOOL.pull_down(str(source), mirror)
    shard = D.first_closed_shard(mirror / D.GAMES_SUBDIR, "synth")
    assert shard is not None
    shard.write_bytes(shard.read_bytes()[:-3])
    assert TOOL.receipt_shards(mirror, "synth", cycle=1, mirror_id="op") == []


def test_a_bundle_that_does_not_verify_in_the_mirror_is_not_receipted(
    tmp_path: Path, synthetic_run_dir,
) -> None:
    source = synthetic_run_dir(tmp_path / "box")
    mirror = tmp_path / "mirror"
    TOOL.pull_down(str(source), mirror)
    ring = next((mirror / D.CHECKPOINTS_SUBDIR).glob("*.ring.bin"))
    ring.write_bytes(b"torn")
    assert TOOL.receipt_bundles(mirror, cycle=1, mirror_id="op") == []


def test_receipts_never_flow_down_and_only_receipts_flow_up(
    tmp_path: Path, synthetic_run_dir,
) -> None:
    source = synthetic_run_dir(tmp_path / "box")
    (source / "checkpoints" / "stale.receipt.json").write_text("{}", encoding="utf-8")
    mirror = tmp_path / "mirror"
    TOOL.pull_down(str(source), mirror)
    assert not (mirror / "checkpoints" / "stale.receipt.json").exists()
    (mirror / "checkpoints" / "not_a_receipt.txt").write_text("x", encoding="utf-8")
    TOOL.push_receipts(mirror, str(source))
    assert not (source / "checkpoints" / "not_a_receipt.txt").exists()


def test_the_cli_once_mode_reports_the_cycle(tmp_path: Path, synthetic_run_dir, capsys) -> None:
    source = synthetic_run_dir(tmp_path / "box")
    rc = TOOL.main(["--source", str(source), "--mirror", str(tmp_path / "m"), "--run-id", "synth",
                    "--once", "--mirror-id", "op"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["bundles_receipted"] == [40]


def test_a_missing_source_is_a_transport_failure_not_a_silent_pass(tmp_path: Path) -> None:
    with pytest.raises(TOOL.MirrorTransportError):
        TOOL.pull_down(str(tmp_path / "absent"), tmp_path / "m")


def test_a_bare_checkpoint_is_receipted_only_when_its_payload_hashes_to_its_name(
    tmp_path: Path, synthetic_run_dir,
) -> None:
    """The puller verifies a bare checkpoint the loader's way — `content_sha8` against its name."""
    import torch

    source = synthetic_run_dir(tmp_path / "box", bundle=False)
    mirror = tmp_path / "mirror"
    summary = TOOL.run_cycle(str(source), mirror, "synth", cycle=1, mirror_id="op")
    assert summary["bundles_receipted"] == [] and len(summary["checkpoints_receipted"]) == 1
    assert D.require_mirror_receipts(source, "synth")["checkpoint"]["name"].endswith(".ckpt")
    # A checkpoint whose bytes do not hash to its name is a torn or foreign copy: no receipt.
    forged = source / "checkpoints" / "synth_00000099_deadbeef.ckpt"
    torch.save({"schema_version": 2, "step": 99}, forged)
    again = TOOL.run_cycle(str(source), mirror, "synth", cycle=2, mirror_id="op")
    assert again["checkpoints_receipted"] == []
    assert not U.receipt_path_for(forged).exists()
