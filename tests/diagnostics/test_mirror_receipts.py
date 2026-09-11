"""R349(b): a receipt proves one byte string left the box and the START halt demands receipts
for the burst's bundle and first shard; the deleted volume arm's replacement must bite the same
way — other bytes refused, a missing receipt refused, the pass naming what it read."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mantis.diagnostics import mirror_receipts as D
from mantis.train.bundle import complete_bundles
from mantis.util import mirror_receipts as U
from mantis.util.hashing import sha256_file


def _receipt_everything(root: Path, run_id: str = "synth") -> None:
    """Receipt every bundle file and closed shard as the puller would, from the same bytes."""
    checkpoints = root / D.CHECKPOINTS_SUBDIR
    for manifest in complete_bundles(checkpoints):
        for path in D.bundle_member_paths(manifest, checkpoints):
            U.write_receipt(path, mirrored_sha256=sha256_file(path),
                            mirrored_bytes=path.stat().st_size, cycle=1, mirror_id="test")
    shard = D.first_closed_shard(root / D.GAMES_SUBDIR, run_id)
    assert shard is not None
    U.write_receipt(shard, mirrored_sha256=sha256_file(shard), mirrored_bytes=shard.stat().st_size,
                    cycle=1, mirror_id="test")


def test_a_receipt_verifies_only_against_the_bytes_it_was_written_for(tmp_path: Path) -> None:
    artifact = tmp_path / "a.bin"
    artifact.write_bytes(b"x" * 1000)
    U.write_receipt(artifact, mirrored_sha256=sha256_file(artifact), mirrored_bytes=1000,
                    cycle=3, mirror_id="op")
    reading = U.verify_receipt(artifact)
    assert reading["verified_sha256"] == sha256_file(artifact) and reading["cycle"] == 3
    artifact.write_bytes(b"x" * 999 + b"y")
    with pytest.raises(U.MirrorReceiptError, match="sha256"):
        U.verify_receipt(artifact)
    artifact.write_bytes(b"x" * 10)
    with pytest.raises(U.MirrorReceiptError, match="bytes"):
        U.verify_receipt(artifact)


def test_a_missing_or_foreign_receipt_is_refused_by_name(tmp_path: Path) -> None:
    artifact = tmp_path / "a.bin"
    artifact.write_bytes(b"abc")
    with pytest.raises(U.MirrorReceiptError, match="no receipt"):
        U.verify_receipt(artifact)
    other = tmp_path / "b.bin"
    other.write_bytes(b"abc")
    U.write_receipt(other, mirrored_sha256=sha256_file(other), mirrored_bytes=3, cycle=1,
                    mirror_id="op")
    U.receipt_path_for(other).rename(U.receipt_path_for(artifact))
    with pytest.raises(U.MirrorReceiptError, match="names 'b.bin'"):
        U.verify_receipt(artifact)


def test_the_halt_refuses_an_unreceipted_run_directory_and_names_the_gap(
    tmp_path: Path, synthetic_run_dir,
) -> None:
    with pytest.raises(D.MirrorReceiptsMissingError, match="no complete resume bundle"):
        D.require_mirror_receipts(tmp_path, "synth")
    synthetic_run_dir(tmp_path)
    with pytest.raises(D.MirrorReceiptsMissingError, match="step 40 is not receipted"):
        D.require_mirror_receipts(tmp_path, "synth")
    assert D.unreceipted_bundle_steps(tmp_path / D.CHECKPOINTS_SUBDIR) == [40]


def test_the_halt_passes_when_bundle_and_first_shard_are_receipted(
    tmp_path: Path, synthetic_run_dir,
) -> None:
    synthetic_run_dir(tmp_path, shards=2)
    _receipt_everything(tmp_path)
    reading = D.require_mirror_receipts(tmp_path, "synth")
    assert reading["verdict"] == U.MIRRORED_VERDICT
    assert reading["bundle"]["step"] == 40
    assert set(reading["bundle"]["files"]) == {
        p.name for p in D.bundle_member_paths(
            complete_bundles(tmp_path / D.CHECKPOINTS_SUBDIR)[0], tmp_path / D.CHECKPOINTS_SUBDIR)
    }
    assert reading["shard"]["name"].endswith("seg0001_2026091201.jsonl"), (
        "the FIRST closed shard is the one demanded, not the newest")
    assert D.unreceipted_bundle_steps(tmp_path / D.CHECKPOINTS_SUBDIR) == []


def test_a_receipt_for_a_rewritten_member_no_longer_covers_it(
    tmp_path: Path, synthetic_run_dir,
) -> None:
    """Killer: a receipt keyed by NAME alone — the bytes could change after mirroring."""
    synthetic_run_dir(tmp_path)
    _receipt_everything(tmp_path)
    shard = D.first_closed_shard(tmp_path / D.GAMES_SUBDIR, "synth")
    assert shard is not None
    with shard.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"late": True}) + "\n")
    with pytest.raises(D.MirrorReceiptsMissingError, match="shard .* is not receipted"):
        D.require_mirror_receipts(tmp_path, "synth")


def test_the_wait_spends_its_budget_then_raises_the_last_refusal(tmp_path: Path) -> None:
    with pytest.raises(D.MirrorReceiptsMissingError, match="waited 0 s"):
        D.await_mirror_receipts(tmp_path, "synth", wait_sec=0.0, poll_sec=0.01)


def test_the_cli_mirrors_the_halt(tmp_path: Path, synthetic_run_dir, capsys) -> None:
    synthetic_run_dir(tmp_path)
    assert D.main([str(tmp_path), "--run-id", "synth", "--json"]) == 2
    assert json.loads(capsys.readouterr().err)["verdict"] == "REFUSED"
    _receipt_everything(tmp_path)
    assert D.main([str(tmp_path), "--run-id", "synth", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == U.MIRRORED_VERDICT


def test_a_clean_completion_with_no_bundle_is_proven_on_its_checkpoint(
    tmp_path: Path, synthetic_run_dir,
) -> None:
    """R137's third leg writes no bundle; killer: the halt that demanded one (three box rows red)."""
    synthetic_run_dir(tmp_path, bundle=False)
    with pytest.raises(D.MirrorReceiptsMissingError, match="checkpoint .* is not receipted"):
        D.require_mirror_receipts(tmp_path, "synth")
    ckpt = D.stamped_checkpoints(tmp_path / D.CHECKPOINTS_SUBDIR)[-1]
    U.write_receipt(ckpt, mirrored_sha256=sha256_file(ckpt), mirrored_bytes=ckpt.stat().st_size,
                    cycle=1, mirror_id="test")
    shard = D.first_closed_shard(tmp_path / D.GAMES_SUBDIR, "synth")
    assert shard is not None
    U.write_receipt(shard, mirrored_sha256=sha256_file(shard), mirrored_bytes=shard.stat().st_size,
                    cycle=1, mirror_id="test")
    reading = D.require_mirror_receipts(tmp_path, "synth")
    assert reading["verdict"] == U.MIRRORED_VERDICT and "bundle" not in reading
    assert reading["checkpoint"]["name"] == ckpt.name
