"""START pre-flight, the mirror arm (R349(b)): R347(d)'s volume halt is deleted (no host on
offer has a volume) and the puller's receipts are the rule — the preflight demands them for the
burst's bundle and first shard, so the loop is proven, not assumed."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from mantis.monitor.game_record import index_filename
from mantis.train.bundle import complete_bundles
from mantis.train.bundle_receipts import (
    bundle_member_paths,
    stamped_checkpoints,
    unreceipted_bundle_steps,
    unreceipted_members,
)
from mantis.util.mirror_receipts import MIRRORED_VERDICT, MirrorReceiptError, verify_receipt

#: Where a run writes each artifact class, relative to its run directory.
CHECKPOINTS_SUBDIR = "checkpoints"
GAMES_SUBDIR = Path("logs") / "games"


class MirrorReceiptsMissingError(RuntimeError):
    """The run directory's required artifacts are not (all) receipted. Names what is missing."""


def first_closed_shard(record_dir: str | Path, run_id: str) -> Path | None:
    """The run's FIRST closed shard, read off its index; `None` before one closes.

    Raises:
        OSError: the index exists but could not be read.
    """
    index = Path(record_dir) / index_filename(run_id)
    if not index.is_file():
        return None
    for line in index.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("record") == "shard_closed" and row.get("shard"):
            return Path(record_dir) / str(row["shard"])
    return None


def require_mirror_receipts(run_dir: str | Path, run_id: str) -> dict[str, Any]:
    """The newest complete bundle (else the newest stamped checkpoint) and the first closed shard
    are BOTH receipted; the stamp's reading names which artefact it read.

    Raises:
        MirrorReceiptsMissingError: no artefact of record or closed shard, or a receipt absent/wrong.
    """
    root = Path(run_dir).expanduser().resolve()
    checkpoint_dir = root / CHECKPOINTS_SUBDIR
    bundles = complete_bundles(checkpoint_dir)
    artefact: dict[str, Any]
    if bundles:
        newest = bundles[-1]
        missing = unreceipted_members(newest, checkpoint_dir)
        if missing:
            raise MirrorReceiptsMissingError(
                f"the bundle at step {newest.step} is not receipted: "
                + "; ".join(f"{name}: {why}" for name, why in missing.items()))
        artefact = {"bundle": {"step": newest.step, "files": {
            path.name: verify_receipt(path)["verified_sha256"]
            for path in bundle_member_paths(newest, checkpoint_dir)}}}
    else:
        # A clean completion writes a checkpoint and NO bundle (R137's third leg).
        checkpoints = stamped_checkpoints(checkpoint_dir)
        if not checkpoints:
            raise MirrorReceiptsMissingError(
                f"no complete resume bundle and no stamped checkpoint under {checkpoint_dir}: "
                "the run must write one before its mirroring can be proven")
        newest_ckpt = checkpoints[-1]
        try:
            ckpt_receipt = verify_receipt(newest_ckpt)
        except MirrorReceiptError as exc:
            raise MirrorReceiptsMissingError(
                f"checkpoint {newest_ckpt.name} is not receipted: {exc}") from exc
        artefact = {"checkpoint": {"name": newest_ckpt.name,
                                   "sha256": ckpt_receipt["verified_sha256"]}}
    shard = first_closed_shard(root / GAMES_SUBDIR, run_id)
    if shard is None:
        raise MirrorReceiptsMissingError(
            f"no closed game-record shard for run_id {run_id!r} under {root / GAMES_SUBDIR}")
    try:
        shard_receipt = verify_receipt(shard)
    except MirrorReceiptError as exc:
        raise MirrorReceiptsMissingError(f"shard {shard.name} is not receipted: {exc}") from exc
    return {
        "verdict": MIRRORED_VERDICT,
        "run_dir": str(root),
        **artefact,
        "shard": {"name": shard.name, "sha256": shard_receipt["verified_sha256"]},
    }


def await_mirror_receipts(run_dir: str | Path, run_id: str, *, wait_sec: float,
                          poll_sec: float = 5.0) -> dict[str, Any]:
    """`require_mirror_receipts`, retried until it passes or `wait_sec` elapses.

    Raises:
        MirrorReceiptsMissingError: the last refusal, once the wait is spent.
    """
    deadline = time.monotonic() + max(0.0, float(wait_sec))
    while True:
        try:
            return require_mirror_receipts(run_dir, run_id)
        except MirrorReceiptsMissingError as exc:
            if time.monotonic() >= deadline:
                raise MirrorReceiptsMissingError(
                    f"{exc} (waited {wait_sec:.0f} s for the puller's receipts)") from exc
        time.sleep(poll_sec)


__all__ = [
    "CHECKPOINTS_SUBDIR", "GAMES_SUBDIR", "MirrorReceiptsMissingError", "await_mirror_receipts",
    "bundle_member_paths", "first_closed_shard", "require_mirror_receipts",
    "unreceipted_bundle_steps", "unreceipted_members",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mantis.diagnostics.mirror_receipts",
        description="Refuse a run directory whose bundle and first shard are not receipted.")
    parser.add_argument("run_dir", help="the run directory (holds checkpoints/ and logs/games/)")
    parser.add_argument("--run-id", required=True, help="the run's id, for the shard index")
    parser.add_argument("--json", action="store_true", help="emit the reading as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        reading = require_mirror_receipts(Path(args.run_dir), args.run_id)
    except MirrorReceiptsMissingError as exc:
        payload = {"run_dir": args.run_dir, "verdict": "REFUSED", "reason": str(exc)}
        print(json.dumps(payload) if args.json else f"MIRROR-RECEIPTS HALT: {exc}",
              file=sys.stderr)
        return 2
    artefact = (f"bundle step {reading['bundle']['step']}" if "bundle" in reading
                else f"checkpoint {reading['checkpoint']['name']}")
    print(json.dumps(reading) if args.json else
          f"MIRROR-RECEIPTS PASS: {artefact} and shard {reading['shard']['name']} are "
          f"receipted under {reading['run_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
