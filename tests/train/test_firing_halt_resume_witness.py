"""A run-fatal halt leaves the last bundle resumable, ring hash equal.

A wire-corruption detector raises out of the collate path and must stop the run — but it must
not take the resume point with it. A bundle is published and its ring hash recorded, then a
save is interrupted partway through writing the ring. The assertions are the resume contract
exactly: the previous bundle still verifies, its ring is byte-identical, and the hash the
manifest recorded is still the hash on disk.

This drives the halt's SHAPE — an exception out of the ring write — not the corruption that
produces it, so the planted failure is a plain raise and is labelled as one.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from mantis.train import bundle as B


class _PlantedFiring(RuntimeError):
    """Stands in for the `GraphContractError` a real firing raises — same shape, no wire."""


def _publish(directory: Path, *, step: int, ring_bytes: bytes) -> Path:
    ckpt = directory / f"run_{step:08d}_abcd1234.ckpt"
    ckpt.write_bytes(f"checkpoint {step}".encode())
    return B.publish_bundle(
        checkpoint_path=ckpt, run_id="run", step=step,
        write_ring=lambda p: Path(p).write_bytes(ring_bytes),
        ring_path=B.ring_path_for(ckpt),
        write_sidecar=lambda p: Path(p).write_text(f'{{"step": {step}}}', encoding="utf-8"),
        sidecar_path=Path(str(ckpt) + ".resume.json"),
    )


def test_a_firing_during_the_next_ring_write_leaves_the_previous_bundle_intact(
    tmp_path: Path,
) -> None:
    """Planted firing: the step-10 bundle still resumes, ring hash equal."""
    ring_bytes = b"the ring that must survive" * 512
    _publish(tmp_path, step=10, ring_bytes=ring_bytes)
    before = B.newest_complete_bundle(tmp_path)
    assert before is not None and before.ring is not None
    recorded_hash = before.ring.sha256
    ring_on_disk = tmp_path / before.ring.name
    assert hashlib.sha256(ring_on_disk.read_bytes()).hexdigest() == recorded_hash

    next_ckpt = tmp_path / "run_00000020_abcd1234.ckpt"
    next_ckpt.write_bytes(b"checkpoint 20")

    def _fires_partway(path: Path) -> None:
        Path(path).write_bytes(b"half a ring and then")
        raise _PlantedFiring("F-816-37-shaped halt during the ring write")

    with pytest.raises(_PlantedFiring):
        B.publish_bundle(
            checkpoint_path=next_ckpt, run_id="run", step=20,
            write_ring=_fires_partway, ring_path=B.ring_path_for(next_ckpt),
            write_sidecar=lambda p: Path(p).write_text("{}", encoding="utf-8"),
            sidecar_path=Path(str(next_ckpt) + ".resume.json"),
        )

    after = B.newest_complete_bundle(tmp_path)
    assert after is not None, (
        "the halt left NO complete bundle — the run cannot resume, which is the whole failure "
        "this leg exists to make impossible"
    )
    assert after.step == 10, (
        f"the torn step-20 bundle was offered as a resume point (got step {after.step}); its "
        "manifest never committed, so it must not be visible at all"
    )
    assert after.ring is not None and after.ring.sha256 == recorded_hash, (
        "the surviving bundle's manifest no longer records the ring hash it was published with"
    )
    assert hashlib.sha256(ring_on_disk.read_bytes()).hexdigest() == recorded_hash, (
        "THE RING CHANGED. The previous ring was destroyed by an interrupted write of the "
        "next one — the `File::create` truncation defect, which is exactly what this leg "
        "removed."
    )
    B.verify_bundle(after, tmp_path)


def test_the_torn_bundles_members_do_not_impersonate_a_resume_point(tmp_path: Path) -> None:
    """A half-written ring is left on disk deliberately, so what must hold is that nothing
    OFFERS it as anybody's ring."""
    _publish(tmp_path, step=10, ring_bytes=b"good ring")
    next_ckpt = tmp_path / "run_00000020_abcd1234.ckpt"
    next_ckpt.write_bytes(b"checkpoint 20")
    with pytest.raises(_PlantedFiring):
        B.publish_bundle(
            checkpoint_path=next_ckpt, run_id="run", step=20,
            write_ring=lambda p: (_ for _ in ()).throw(_PlantedFiring("halt")),
            ring_path=B.ring_path_for(next_ckpt),
            write_sidecar=lambda p: Path(p).write_text("{}", encoding="utf-8"),
            sidecar_path=Path(str(next_ckpt) + ".resume.json"),
        )

    assert [m.step for m in B.complete_bundles(tmp_path)] == [10]


def test_a_halt_after_the_manifest_commits_keeps_the_new_bundle(tmp_path: Path) -> None:
    """Mutation half: the commit point is the MANIFEST, so an unconditional fall back to the
    older bundle does not satisfy this row."""
    _publish(tmp_path, step=10, ring_bytes=b"old ring")
    _publish(tmp_path, step=20, ring_bytes=b"new ring")
    newest = B.newest_complete_bundle(tmp_path)
    assert newest is not None and newest.step == 20, (
        "a fully published bundle was not offered — the fallback is unconditional"
    )
