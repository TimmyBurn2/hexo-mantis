"""R345(b)(3) — a checkpoint that is not a continuation point is not a resume.

THREE DEFECTS, ONE LEG. At HEAD a stop wrote three files with three different durability
postures and no relationship between them:

  * the CHECKPOINT went straight to its final path (`torch.save(payload, path)`), with no
    temp file, no `fsync` and no rename — a kill mid-write leaves a `.ckpt` that exists, is
    named for a content hash it does not have, and fails to load;
  * the RING was worse. `HexgBuffer::save_to_path_impl` opened with `File::create`, which
    TRUNCATES the existing file before the first new byte is written, so a kill mid-save
    leaves no ring at all rather than the previous one — the single most expensive file in
    the run, written in the one way that destroys the old copy first;
  * the SIDECAR was already atomic, which is what makes the other two visible as choices
    rather than as house style.

And nothing tied them together: `_maybe_periodic_checkpoint` wrote the checkpoint ALONE, so
every periodic artefact was weights plus optimizer state with no ring and no sidecar — not a
continuation point, whatever its name suggested. A run killed between periodic saves resumed
from a checkpoint whose ring refilled from empty, which is what R343(c) forbids.

THE MANIFEST IS THE COMMIT. Members are written to temp, fsynced and renamed; the manifest
naming them and their hashes is written LAST, by the same atomic route. A bundle without its
manifest is incomplete BY CONSTRUCTION and a resume will not take it — which is the property
that makes "two complete bundles retained" a meaningful guarantee rather than a file count.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch

import _microbatch_harness as H
from mantis.train import bundle as B


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _member_files(directory: Path) -> set[str]:
    return {p.name for p in directory.iterdir() if p.is_file()}


# ── atomic publication ──────────────────────────────────────────────────────────────────
def test_a_failed_write_leaves_the_previous_file_intact(tmp_path: Path) -> None:
    """The property `File::create` and a bare `torch.save` both give up.

    The writer raises halfway; the target must still hold the OLD bytes, and no temp file may
    be left behind to be mistaken for a partial artefact later.
    """
    target = tmp_path / "thing.bin"
    B.atomic_write(target, lambda handle: handle.write(b"first"))
    assert target.read_bytes() == b"first"

    def _explode(handle: Any) -> None:
        handle.write(b"second, but only ha")
        raise RuntimeError("planted mid-write failure")

    with pytest.raises(RuntimeError, match="planted mid-write failure"):
        B.atomic_write(target, _explode)

    assert target.read_bytes() == b"first", (
        "the interrupted write reached the target — it was not written through a temp file"
    )
    assert _member_files(tmp_path) == {"thing.bin"}, (
        f"a temp file survived the failure: {_member_files(tmp_path)}"
    )


def test_the_checkpoint_is_written_through_a_temp_file(tmp_path: Path) -> None:
    """The real writer, watched: the final path must never be the one being appended to."""
    seen: list[str] = []
    real_open = B.atomic_write

    trainer = H.tiny_graph_trainer(tmp_path, sink=H.SpySink())

    def _watch(path: Path, writer: Any) -> Path:
        seen.append(Path(path).name)
        return real_open(path, writer)

    import mantis.train.checkpoints as checkpoints

    original = checkpoints.atomic_write
    checkpoints.atomic_write = _watch
    try:
        written = trainer.save_checkpoint(None)
    finally:
        checkpoints.atomic_write = original

    assert seen == [written.name], (
        f"the checkpoint write did not go through the atomic publisher (saw {seen})"
    )
    assert written.is_file()


# ── the bundle ──────────────────────────────────────────────────────────────────────────
def test_publishing_a_bundle_writes_the_manifest_last(tmp_path: Path) -> None:
    """Order is the whole contract: the manifest's existence is the commit."""
    order: list[str] = []
    ckpt = tmp_path / "run_00000010_deadbeef.ckpt"
    ckpt.write_bytes(b"checkpoint bytes")
    ring = tmp_path / "run_00000010_deadbeef.ring.bin"
    sidecar = tmp_path / "run_00000010_deadbeef.resume.json"

    def _ring_writer(path: Path) -> None:
        order.append("ring")
        Path(path).write_bytes(b"ring bytes")

    def _sidecar_writer(path: Path) -> None:
        order.append("sidecar")
        Path(path).write_text("{}", encoding="utf-8")

    manifest_path = B.publish_bundle(
        checkpoint_path=ckpt, run_id="run", step=10,
        write_ring=_ring_writer, ring_path=ring,
        write_sidecar=_sidecar_writer, sidecar_path=sidecar,
    )

    assert manifest_path.name.endswith(B.BUNDLE_SUFFIX)
    assert order == ["ring", "sidecar"], f"members were not written before the manifest: {order}"
    manifest = B.read_manifest(manifest_path)
    assert manifest.checkpoint.sha256 == _sha(ckpt)
    assert manifest.ring is not None and manifest.ring.sha256 == _sha(ring)
    assert manifest.sidecar.sha256 == _sha(sidecar)


def test_a_bundle_whose_member_changed_underneath_is_refused(tmp_path: Path) -> None:
    """A hash in the manifest that nothing compares against is a phantom gate (LAW-07)."""
    manifest_path = _publish_a_bundle(tmp_path, step=10)
    manifest = B.read_manifest(manifest_path)
    B.verify_bundle(manifest, tmp_path)  # the control: it verifies before we touch it

    ring = tmp_path / manifest.ring.name
    original = ring.read_bytes()

    # SAME LENGTH, different bytes — so the hash is the only thing that can catch it. A
    # length-changing tamper would be caught by the size field and prove nothing about the
    # hash, which is the check that has to work when a file is silently corrupted in place
    # (F-816-37's own signature is a single flipped exponent bit).
    ring.write_bytes(bytes(b ^ 0xFF for b in original))
    with pytest.raises(B.BundleError, match="sha256"):
        B.verify_bundle(manifest, tmp_path)

    ring.write_bytes(original[:-1])
    with pytest.raises(B.BundleError, match="bytes"):
        B.verify_bundle(manifest, tmp_path)


def test_a_bundle_whose_member_is_corrupt_is_not_offered_as_a_resume_point(
    tmp_path: Path,
) -> None:
    """The manifest EXISTING is not the same fact as the bundle being intact.

    Found by a mutation that removed `verify_bundle` from `complete_bundles` and passed every
    other test in this file: the hashes were written, compared on demand, and never consulted
    on the path a resume actually takes. A silently corrupted ring — F-816-37's own signature
    is a single flipped exponent bit, which changes no length — would have been handed to a
    resume as a valid continuation point.
    """
    _publish_a_bundle(tmp_path, step=10)
    _publish_a_bundle(tmp_path, step=20)
    newest = B.newest_complete_bundle(tmp_path)
    assert newest is not None and newest.step == 20  # the control

    ring = tmp_path / newest.ring.name
    ring.write_bytes(bytes(b ^ 0x01 for b in ring.read_bytes()))  # same length, one bit

    fallen_back = B.newest_complete_bundle(tmp_path)
    assert fallen_back is not None and fallen_back.step == 10, (
        "the corrupted step-20 bundle was still offered; its manifest exists, but its ring "
        "does not hash to what the manifest records, so it is not a resume point"
    )
    assert [m.step for m in B.complete_bundles(tmp_path)] == [10]


def test_a_bundle_with_no_manifest_is_not_a_bundle(tmp_path: Path) -> None:
    """The half-written case, which is the one that matters at 3 a.m."""
    manifest_path = _publish_a_bundle(tmp_path, step=10)
    manifest_path.unlink()
    assert B.newest_complete_bundle(tmp_path) is None, (
        "a checkpoint with no manifest was offered as a resume point"
    )


def test_the_newest_complete_bundle_is_the_one_offered(tmp_path: Path) -> None:
    _publish_a_bundle(tmp_path, step=10)
    _publish_a_bundle(tmp_path, step=30)
    _publish_a_bundle(tmp_path, step=20)
    newest = B.newest_complete_bundle(tmp_path)
    assert newest is not None and newest.step == 30, (
        "bundles were ordered by something other than their step"
    )


# ── retention ───────────────────────────────────────────────────────────────────────────
def test_two_complete_bundles_are_retained_and_older_ones_go_whole(tmp_path: Path) -> None:
    """Retention deletes BUNDLES, not files: a surviving orphan member is the failure mode."""
    for step in (10, 20, 30, 40):
        _publish_a_bundle(tmp_path, step=step)

    removed = B.prune_bundles(tmp_path, keep=2)

    kept = sorted(m.step for m in B.complete_bundles(tmp_path))
    assert kept == [30, 40], f"retention kept {kept}"
    assert removed, "prune reported removing nothing while two bundles disappeared"
    survivors = _member_files(tmp_path)
    for step in (10, 20):
        assert not any(f"{step:08d}" in name for name in survivors), (
            f"step {step}'s bundle left members behind: "
            f"{sorted(n for n in survivors if f'{step:08d}' in n)}"
        )


def test_retention_never_drops_below_one_complete_bundle(tmp_path: Path) -> None:
    """Mutation half. `keep=2` with only one bundle present must delete nothing."""
    _publish_a_bundle(tmp_path, step=10)
    assert B.prune_bundles(tmp_path, keep=2) == []
    assert B.newest_complete_bundle(tmp_path) is not None


def test_an_incomplete_bundle_is_never_counted_as_one_of_the_two(tmp_path: Path) -> None:
    """Otherwise "keep 2" retains one usable bundle and one carcass."""
    _publish_a_bundle(tmp_path, step=10)
    _publish_a_bundle(tmp_path, step=20)
    torn = _publish_a_bundle(tmp_path, step=30)
    torn.unlink()  # step 30 is now manifest-less: written, never committed

    B.prune_bundles(tmp_path, keep=2)

    kept = sorted(m.step for m in B.complete_bundles(tmp_path))
    assert kept == [10, 20], (
        f"the torn bundle was counted toward the retention budget (kept {kept}), so the run "
        "retained one usable bundle and one that cannot be resumed from"
    )


# ── fixture ─────────────────────────────────────────────────────────────────────────────
def _publish_a_bundle(directory: Path, *, step: int) -> Path:
    ckpt = directory / f"run_{step:08d}_abcd1234.ckpt"
    ckpt.write_bytes(f"checkpoint {step}".encode())
    return B.publish_bundle(
        checkpoint_path=ckpt, run_id="run", step=step,
        write_ring=lambda p: Path(p).write_bytes(f"ring {step}".encode()),
        ring_path=directory / f"run_{step:08d}_abcd1234{B.RING_SUFFIX}",
        write_sidecar=lambda p: Path(p).write_text(
            json.dumps({"step": step}), encoding="utf-8"),
        sidecar_path=directory / f"run_{step:08d}_abcd1234.resume.json",
    )
