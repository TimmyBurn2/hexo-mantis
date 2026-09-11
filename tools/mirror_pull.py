"""The puller (R349(b)), run on the OPERATOR'S machine: each cycle rsyncs the run directory
down, verifies bundles against their manifests and closed shards against the index's sizes,
receipts the mirrored bytes and rsyncs the receipts up. `--source` is an rsync spec; no host here."""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from mantis.diagnostics.mirror_receipts import CHECKPOINTS_SUBDIR, GAMES_SUBDIR, bundle_member_paths
from mantis.monitor.game_record import index_filename
from mantis.train.bundle import BundleError, complete_bundles
from mantis.train.bundle_receipts import CHECKPOINT_NAME_RE, stamped_checkpoints
from mantis.util.hashing import sha256_file
from mantis.util.mirror_receipts import (
    RECEIPT_SUFFIX,
    MirrorReceiptError,
    read_receipt,
    receipt_path_for,
    write_receipt,
)

_LOG = logging.getLogger("mirror_pull")

#: Never pulled: a writer's temp files. Never pulled DOWN: receipts are this tool's own output.
_PULL_EXCLUDES = ("*.tmp", f"*{RECEIPT_SUFFIX}", f"*{RECEIPT_SUFFIX}.*")


class MirrorTransportError(RuntimeError):
    """rsync is absent or a transfer failed; the cycle is retried, never skipped."""


def _rsync(src: str, dst: str, *extra: str) -> None:
    binary = shutil.which("rsync")
    if binary is None:
        raise MirrorTransportError("rsync is not installed on this machine")
    cmd = [binary, "-a", "--partial", *extra, src.rstrip("/") + "/", dst.rstrip("/") + "/"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise MirrorTransportError(
            f"rsync rc {proc.returncode}: {' '.join(cmd)}\n{proc.stderr.strip()[-2000:]}")


def pull_down(source: str, mirror: Path) -> None:
    """Mirror the run directory from `source` into `mirror`, receipts and temp files excluded."""
    mirror.mkdir(parents=True, exist_ok=True)
    _rsync(source, str(mirror), *(f"--exclude={pattern}" for pattern in _PULL_EXCLUDES))


def push_receipts(mirror: Path, source: str) -> None:
    """Send every receipt under `mirror` back beside its artifact under `source`."""
    _rsync(str(mirror), source, "--include=*/", f"--include=*{RECEIPT_SUFFIX}", "--exclude=*",
           "--prune-empty-dirs")


def _receipt_current(path: Path, digest: str) -> bool:
    try:
        return read_receipt(receipt_path_for(path))["sha256"] == digest
    except MirrorReceiptError:
        return False


def receipt_bundles(mirror: Path, *, cycle: int, mirror_id: str) -> list[int]:
    """Receipt every complete bundle (`complete_bundles` re-verified the MIRRORED members); new steps."""
    directory = mirror / CHECKPOINTS_SUBDIR
    receipted: list[int] = []
    for manifest in complete_bundles(directory):
        wrote = False
        for path in bundle_member_paths(manifest, directory):
            digest = sha256_file(path)
            if _receipt_current(path, digest):
                continue
            write_receipt(path, mirrored_sha256=digest, mirrored_bytes=path.stat().st_size,
                          cycle=cycle, mirror_id=mirror_id)
            wrote = True
        if wrote:
            receipted.append(manifest.step)
    return receipted


def _checkpoint_verifies(path: Path) -> bool:
    """The copy hashes to the `content_sha8` its filename carries (the loader's own check)."""
    import torch  # lazy: the checkpoint arm is the only reason the puller needs torch

    from mantis.train.checkpoints import content_sha8

    match = CHECKPOINT_NAME_RE.match(path.name)
    if match is None:
        return False
    try:
        payload = torch.load(path, weights_only=True, map_location="cpu")
    except Exception as exc:  # noqa: BLE001 — a copy torch cannot read is not receipted, and why is logged
        _LOG.warning("checkpoint unreadable in the mirror path=%s: %s", path.name, exc)
        return False
    return isinstance(payload, dict) and content_sha8(payload) == match.group("sha8")


def receipt_checkpoints(mirror: Path, *, cycle: int, mirror_id: str) -> list[str]:
    """Receipt every bare stamped checkpoint whose payload hashes to its name; returns the names."""
    receipted: list[str] = []
    for path in stamped_checkpoints(mirror / CHECKPOINTS_SUBDIR):
        digest = sha256_file(path)
        if _receipt_current(path, digest):
            continue
        if not _checkpoint_verifies(path):
            continue
        write_receipt(path, mirrored_sha256=digest, mirrored_bytes=path.stat().st_size,
                      cycle=cycle, mirror_id=mirror_id)
        receipted.append(path.name)
    return receipted


def closed_shards(record_dir: Path, run_id: str) -> list[tuple[Path, int]]:
    """`(shard path, bytes at close)` for every row of the mirrored index."""
    index = record_dir / index_filename(run_id)
    if not index.is_file():
        return []
    rows: list[tuple[Path, int]] = []
    for line in index.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("record") == "shard_closed" and row.get("shard"):
            rows.append((record_dir / str(row["shard"]), int(row.get("bytes", -1))))
    return rows


def receipt_shards(mirror: Path, run_id: str, *, cycle: int, mirror_id: str) -> list[str]:
    """Receipt every closed shard whose mirrored size equals the indexed size at close (a
    shorter copy is a partial transfer and waits); returns the names."""
    receipted: list[str] = []
    for path, indexed_bytes in closed_shards(mirror / GAMES_SUBDIR, run_id):
        if not path.is_file() or path.stat().st_size != indexed_bytes:
            continue
        digest = sha256_file(path)
        if _receipt_current(path, digest):
            continue
        write_receipt(path, mirrored_sha256=digest, mirrored_bytes=indexed_bytes,
                      cycle=cycle, mirror_id=mirror_id)
        receipted.append(path.name)
    return receipted


def run_cycle(source: str, mirror: Path, run_id: str, *, cycle: int,
              mirror_id: str) -> dict[str, Any]:
    """One pull → verify → receipt → push cycle; returns its summary.

    Raises:
        MirrorTransportError: a transfer failed (the caller decides whether to retry).
    """
    started = time.monotonic()
    pull_down(source, mirror)
    try:
        bundles = receipt_bundles(mirror, cycle=cycle, mirror_id=mirror_id)
    except (BundleError, OSError) as exc:
        _LOG.error("bundle receipting failed cycle=%d: %s", cycle, exc)
        bundles = []
    checkpoints = receipt_checkpoints(mirror, cycle=cycle, mirror_id=mirror_id)
    shards = receipt_shards(mirror, run_id, cycle=cycle, mirror_id=mirror_id)
    push_receipts(mirror, source)
    return {"cycle": cycle, "bundles_receipted": bundles, "checkpoints_receipted": checkpoints,
            "shards_receipted": shards, "seconds": round(time.monotonic() - started, 2)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True,
                        help="rsync source of the run directory: `alias:/path` or a directory")
    parser.add_argument("--mirror", required=True, type=Path, help="local mirror directory")
    parser.add_argument("--run-id", required=True, help="the run's id (names its shard index)")
    parser.add_argument("--mirror-id", default=socket.gethostname(),
                        help="stamped into each receipt; default: this machine's hostname")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="one cycle, then exit")
    mode.add_argument("--interval-sec", type=float,
                      help="loop: one cycle every N seconds until interrupted")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stop = {"flag": False}

    def _stop(_signum: int, _frame: Any) -> None:
        stop["flag"] = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    cycle = 0
    failures = 0
    while not stop["flag"]:
        cycle += 1
        try:
            summary = run_cycle(args.source, args.mirror, args.run_id, cycle=cycle,
                                mirror_id=args.mirror_id)
        except MirrorTransportError as exc:
            failures += 1
            _LOG.error("cycle %d transport failure (%d so far): %s", cycle, failures, exc)
            if args.once:
                return 2
        else:
            print(json.dumps(summary, sort_keys=True), flush=True)
            if args.once:
                return 0
        deadline = time.monotonic() + float(args.interval_sec)
        while not stop["flag"] and time.monotonic() < deadline:
            time.sleep(1.0)
    _LOG.info("puller stopped after %d cycle(s), %d transport failure(s)", cycle, failures)
    return 0


if __name__ == "__main__":
    sys.exit(main())
