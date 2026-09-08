# >300 justify (R8): one subject — what makes a set of files a resume point — and its parts
# are inseparable. `atomic_write` is the durability primitive every member is published
# through, the manifest is the commit record over those members, `verify_bundle` is the read
# side of the hashes `publish_bundle` writes, and `prune_bundles` is retention expressed in
# terms of completeness rather than filenames. Splitting them would put the hash that is
# written in one file and the hash that is checked in another, which is the shape LAW-07 calls
# a phantom gate; and retention that did not read completeness would keep carcasses.
"""The resume BUNDLE — checkpoint + ring + sidecar, published by a manifest last (R345(b)(3)).

WHY A BUNDLE AND NOT THREE SAVES. A resume needs three artefacts to agree: the weights and
optimizer state, the replay ring they were trained against, and the per-stop facts the
envelope cannot carry (`train/resume_state.py`). Written independently they can disagree in
every combination a crash can produce — a checkpoint from step N beside a ring from step N-1,
a sidecar naming a checkpoint that was never finished. The manifest makes the set atomic at
the only level that matters: it is written LAST, so a bundle either has one (and its members
are all present and hash as recorded) or it is not a bundle and no resume will take it.

WHY THE MEMBERS ARE PER-STEP RATHER THAN CANONICAL NAMES. The ring used to live at one
`replay_buffer.bin` that every save overwrote, so "keep the previous bundle" was impossible
even in principle — the previous ring was gone the moment the next save started. Naming each
member for its checkpoint is what makes retention a choice.

DURABILITY IS `fsync`-ORDERED, NOT RENAME-ORDERED. `os.replace` is atomic with respect to
readers, but a rename made durable before the data it points at leaves a crash window where
the name resolves to unwritten blocks. Every member is fsynced BEFORE its rename, and the
containing directory is fsynced after, so the manifest cannot become visible ahead of the
bytes it certifies.
"""
from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO

_LOG = logging.getLogger(__name__)

#: Bumped when a field's MEANING changes, never when one is added — `from_dict` reads every
#: field explicitly, so an older manifest missing one is refused rather than defaulted (R1).
BUNDLE_VERSION = 1

#: Appended to the checkpoint's own filename, so the `{run_id}_{step:08d}_{sha8}` grammar
#: `checkpoints.checkpoint_filename` owns still parses the stem each member is derived from,
#: and no member can be mistaken for a checkpoint by a directory scan.
BUNDLE_SUFFIX = ".bundle.json"
RING_SUFFIX = ".ring.bin"

_HASH_CHUNK = 1 << 20
_STEP_RE = re.compile(r"^.+_(\d{8})_[0-9a-f]+$")


class BundleError(RuntimeError):
    """A bundle is absent, incomplete, or does not hash as its manifest records."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: str | Path, writer: Callable[[BinaryIO], Any]) -> Path:
    """Publish `path` atomically: write a temp sibling, fsync it, rename, fsync the directory.

    `writer` receives an open binary handle and writes the whole payload to it. If it raises,
    the temp file is removed and `path` keeps whatever it held — which is the property a bare
    `open(path, "wb")` cannot give and the one `File::create` on the Rust ring gave up.

    The temp name is unique per writer and call (pid + random), because a shared fixed
    `<name>.tmp` makes two concurrent writers race each other's temp file — the defect
    `monitor/heartbeat.py:250` records having hit.

    Args:
        path: the final path to publish.
        writer: called with the open temp handle; writes the payload.

    Returns:
        The published path.

    Raises:
        OSError: the temp file could not be written, synced or renamed.
        Exception: whatever `writer` raises, after the temp file is removed.
    """
    target = Path(path)
    tmp = target.with_name(
        f"{target.name}.{os.getpid()}.{int.from_bytes(os.urandom(4), 'big'):08x}.tmp"
    )
    try:
        with open(tmp, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _fsync_dir(target.parent)
    return target


def atomic_publish_existing(path: str | Path) -> None:
    """Make an already-written file durable: fsync the file, then its directory.

    For members a foreign writer produced in place — the Rust ring, which does its own
    temp-and-rename inside the engine. The bytes are there; this is the barrier that makes
    them survive the crash the manifest is about to claim they survive.

    Raises:
        OSError: the file could not be opened or synced.
    """
    target = Path(path)
    fd = os.open(target, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_dir(target.parent)


def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@dataclasses.dataclass(frozen=True)
class BundleMember:
    """One file in a bundle: its name (never a path — bundles are directory-relative), its
    hash and its size. The size is not redundant with the hash: it makes a truncation
    visible in the manifest itself, before anything reads a megabyte to find out."""

    name: str
    sha256: str
    bytes: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: Any) -> BundleMember:
        if not isinstance(payload, dict):
            raise BundleError(f"bundle member is {type(payload).__name__}, not an object")
        try:
            return cls(name=str(payload["name"]), sha256=str(payload["sha256"]),
                       bytes=int(payload["bytes"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise BundleError(f"bundle member is malformed: {exc}") from exc


@dataclasses.dataclass(frozen=True)
class BundleManifest:
    """The commit record. Its presence is what makes a set of files a resume point."""

    version: int
    run_id: str
    step: int
    checkpoint: BundleMember
    ring: BundleMember | None
    sidecar: BundleMember

    def members(self) -> list[BundleMember]:
        return [m for m in (self.checkpoint, self.ring, self.sidecar) if m is not None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version, "run_id": self.run_id, "step": self.step,
            "checkpoint": self.checkpoint.to_dict(),
            "ring": None if self.ring is None else self.ring.to_dict(),
            "sidecar": self.sidecar.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> BundleManifest:
        """Rehydrate, reading every field EXPLICITLY — no `.get(key, default)` anywhere (R1).

        Raises:
            BundleError: the payload is not an object, is a version this build does not read,
                or is missing a field.
        """
        if not isinstance(payload, dict):
            raise BundleError(f"bundle manifest is {type(payload).__name__}, not an object")
        version = payload.get("version")
        if version != BUNDLE_VERSION:
            raise BundleError(
                f"bundle manifest version {version!r} is not {BUNDLE_VERSION} — this build "
                "cannot read it, and reading it partially would resume into an undeclared state"
            )
        try:
            ring_raw = payload["ring"]
            return cls(
                version=version,
                run_id=str(payload["run_id"]),
                step=int(payload["step"]),
                checkpoint=BundleMember.from_dict(payload["checkpoint"]),
                ring=None if ring_raw is None else BundleMember.from_dict(ring_raw),
                sidecar=BundleMember.from_dict(payload["sidecar"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise BundleError(f"bundle manifest is malformed: {exc}") from exc


def manifest_path_for(checkpoint_path: str | Path) -> Path:
    return Path(checkpoint_path).with_name(Path(checkpoint_path).name + BUNDLE_SUFFIX)


def ring_path_for(checkpoint_path: str | Path) -> Path:
    return Path(checkpoint_path).with_name(Path(checkpoint_path).name + RING_SUFFIX)


def _member_of(path: Path) -> BundleMember:
    return BundleMember(name=path.name, sha256=sha256_file(path), bytes=path.stat().st_size)


def publish_bundle(
    *,
    checkpoint_path: str | Path,
    run_id: str,
    step: int,
    write_ring: Callable[[Path], Any] | None,
    ring_path: str | Path | None,
    write_sidecar: Callable[[Path], Any],
    sidecar_path: str | Path,
) -> Path:
    """Write the ring and the sidecar beside an ALREADY-WRITTEN checkpoint, then the manifest.

    The checkpoint is written first and by its own writer (`checkpoints.save_checkpoint`), so
    it arrives here as a fact; the two other members are produced by the callables, and the
    manifest — hashes and sizes for all three — is published last through `atomic_write`.

    `write_ring` may be `None` for a bundle that deliberately carries no ring (a bare weights
    save). That is recorded as `ring: null` in the manifest, which is a DIFFERENT fact from a
    ring that failed to write: the latter raises out of `write_ring` and no manifest appears.

    Args:
        checkpoint_path: the checkpoint, already published.
        run_id: the run that produced it.
        step: the training step the bundle continues from.
        write_ring: called with `ring_path`; writes the ring. `None` for no ring.
        ring_path: where the ring goes. Required when `write_ring` is given.
        write_sidecar: called with `sidecar_path`; writes the sidecar.
        sidecar_path: where the sidecar goes.

    Returns:
        The manifest path.

    Raises:
        BundleError: the checkpoint is missing, or `write_ring` was given without a path.
        OSError: a member or the manifest could not be written.
    """
    ckpt = Path(checkpoint_path)
    if not ckpt.is_file():
        raise BundleError(
            f"publish_bundle: {ckpt} does not exist — the checkpoint is written by its own "
            "writer and must be a fact before a bundle can certify it"
        )
    if write_ring is not None and ring_path is None:
        raise BundleError("publish_bundle: write_ring was given without a ring_path")

    ring_member: BundleMember | None = None
    if write_ring is not None and ring_path is not None:
        rp = Path(ring_path)
        write_ring(rp)
        atomic_publish_existing(rp)
        ring_member = _member_of(rp)

    sp = Path(sidecar_path)
    write_sidecar(sp)
    atomic_publish_existing(sp)

    manifest = BundleManifest(
        version=BUNDLE_VERSION, run_id=str(run_id), step=int(step),
        checkpoint=_member_of(ckpt), ring=ring_member, sidecar=_member_of(sp),
    )
    import json

    payload = json.dumps(manifest.to_dict(), indent=2, sort_keys=True).encode("utf-8")
    return atomic_write(manifest_path_for(ckpt), lambda handle: handle.write(payload))


def read_manifest(path: str | Path) -> BundleManifest:
    """Read one manifest.

    Raises:
        BundleError: the file is not JSON, or is not a manifest this build reads.
        OSError: the file could not be read.
    """
    import json

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except ValueError as exc:
        raise BundleError(f"{Path(path).name}: not valid JSON: {exc}") from exc
    return BundleManifest.from_dict(payload)


def verify_bundle(manifest: BundleManifest, directory: str | Path) -> None:
    """Every member present, the right size, and hashing to what the manifest recorded.

    Raises:
        BundleError: a member is missing, truncated, or hashes differently.
    """
    base = Path(directory)
    for member in manifest.members():
        path = base / member.name
        if not path.is_file():
            raise BundleError(
                f"bundle at step {manifest.step}: member {member.name} is missing"
            )
        actual_size = path.stat().st_size
        if actual_size != member.bytes:
            raise BundleError(
                f"bundle at step {manifest.step}: member {member.name} is {actual_size} bytes, "
                f"manifest says {member.bytes} — truncated or replaced"
            )
        actual = sha256_file(path)
        if actual != member.sha256:
            raise BundleError(
                f"bundle at step {manifest.step}: member {member.name} sha256 {actual} does "
                f"not match the manifest's {member.sha256}"
            )


def complete_bundles(directory: str | Path) -> list[BundleManifest]:
    """Every bundle in `directory` whose manifest parses AND whose members all verify.

    A manifest that does not parse, or whose members disagree with it, is NOT a bundle — it is
    logged and skipped, because the alternative is offering a resume point that will fail on
    read. Sorted by step, oldest first.
    """
    base = Path(directory)
    if not base.is_dir():
        return []
    found: list[BundleManifest] = []
    for path in sorted(base.glob(f"*{BUNDLE_SUFFIX}")):
        try:
            manifest = read_manifest(path)
            verify_bundle(manifest, base)
        except (BundleError, OSError) as exc:
            _LOG.warning("bundle_incomplete path=%s reason=%s", path.name, exc)
            continue
        found.append(manifest)
    found.sort(key=lambda m: m.step)
    return found


def newest_complete_bundle(directory: str | Path) -> BundleManifest | None:
    """The highest-step complete bundle, or `None` when there is none to resume from."""
    bundles = complete_bundles(directory)
    return bundles[-1] if bundles else None


def prune_bundles(directory: str | Path, *, keep: int = 2) -> list[str]:
    """De-commit all but the `keep` newest COMPLETE bundles. Returns the names removed.

    **THE CHECKPOINT IS NEVER DELETED.** Retention removes the manifest, the ring and the
    sidecar; the `.ckpt` stays. Four grounds, and the first two are binding:

    * R3 / LAW-12 make the checkpoint the ARTEFACT OF RECORD — run-id and content hash in its
      own filename, stamped once, immutable. Retention is a disk-budget mechanism and has no
      business deleting provenance.
    * R345(d) requires STRENGTH-FRONTIER-1 to measure *"run6's own frozen checkpoints (steps
      ~5k, ~12k, 25k)"*. A retention that deleted checkpoints would destroy two of those three
      before the block ended — the ruling's own later clause, defeated by its earlier one.
    * The budget exists for the RING (~24 MB each); a checkpoint is a fraction of that, so
      deleting it buys almost nothing.
    * Anchors and `strip_and_restamp` read old checkpoints; sweeping them breaks inputs a
      later leg still needs.

    A de-committed bundle stops being a RESUME POINT — its manifest is gone, so
    `complete_bundles` no longer sees it — while remaining a readable artefact. That is the
    distinction the two words are for.

    Completeness is what is counted, and that is the point: counting manifests would let a
    torn bundle occupy one of the two retention slots, leaving the run with one usable resume
    point while its own retention policy reported two.

    Members of an incomplete bundle are deliberately NOT swept here. Sweeping unreferenced
    files is a different operation with a different failure mode (it can delete a bundle
    mid-publication), and it is carded rather than smuggled into retention.

    Raises:
        ValueError: `keep` is below 1 — a retention policy that keeps no resume point is not
            a retention policy.
    """
    if keep < 1:
        raise ValueError(f"prune_bundles: keep={keep} must be >= 1")
    base = Path(directory)
    bundles = complete_bundles(base)
    if len(bundles) <= keep:
        return []
    removed: list[str] = []
    for manifest in bundles[: len(bundles) - keep]:
        # The manifest goes FIRST: it is the commit, so removing it de-commits the bundle
        # before any member disappears. A crash midway then leaves an incomplete bundle that
        # `complete_bundles` already refuses, never a manifest pointing at a deleted member.
        manifest_name = None
        for path in base.glob(f"*{BUNDLE_SUFFIX}"):
            try:
                if read_manifest(path).step == manifest.step:
                    manifest_name = path.name
                    path.unlink()
                    break
            except (BundleError, OSError):
                continue
        for member in manifest.members():
            if member.name == manifest.checkpoint.name:
                continue  # the artefact of record — see this function's docstring
            (base / member.name).unlink(missing_ok=True)
            removed.append(member.name)
        if manifest_name is not None:
            removed.append(manifest_name)
        _LOG.info("bundle_pruned step=%d members=%d", manifest.step, len(manifest.members()))
    return removed


def step_of(path: str | Path) -> int | None:
    """The step encoded in a `{run_id}_{step:08d}_{sha8}` stem, or `None` if it does not parse.

    DERIVED from the filename grammar `checkpoints.checkpoint_filename` owns rather than
    transcribed, so a grammar change reds here instead of silently mis-ordering bundles.
    """
    match = _STEP_RE.match(Path(path).stem)
    return int(match.group(1)) if match else None


__all__ = [
    "BUNDLE_SUFFIX",
    "BUNDLE_VERSION",
    "BundleError",
    "BundleManifest",
    "BundleMember",
    "RING_SUFFIX",
    "atomic_publish_existing",
    "atomic_write",
    "complete_bundles",
    "manifest_path_for",
    "newest_complete_bundle",
    "prune_bundles",
    "publish_bundle",
    "read_manifest",
    "ring_path_for",
    "sha256_file",
    "step_of",
    "verify_bundle",
]
