"""START pre-flight: refuse a run directory that a reboot would erase.

The rule has two arms — the run dir sits on a persistent volume, or every bundle and shard
rsyncs off-box hash-verified within one checkpoint interval. Only the volume arm is decidable
here: the mirror arm's evidence comes from off-box tooling, and a gate input with no in-repo
producer is refused, so the refusal names that arm instead of accepting a flag for it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

#: Filesystems whose contents do not survive the machine; `overlay` is a container's writable
#: layer, which dies with the container even when the image is durable.
EPHEMERAL_FSTYPES = frozenset(
    {"tmpfs", "ramfs", "devtmpfs", "overlay", "overlayfs", "aufs", "squashfs", "ramdisk"}
)

#: Where the kernel publishes the mount table. Parameterised so the parse has a mutation test.
MOUNTS = Path("/proc/mounts")
#: A test drive points the check at a planted table; the reading names the table it used and
#: `mantis.run` refuses a stamp taken from any table but `MOUNTS`, so the seam launches nothing.
MOUNTS_ENV = "MANTIS_PREFLIGHT_MOUNTS_TABLE"


def resolve_mounts_table() -> Path:
    """The mount table a preflight reads: `MOUNTS_ENV` when set, else the kernel's `MOUNTS`."""
    override = os.environ.get(MOUNTS_ENV)
    return Path(override) if override else MOUNTS


class WorkspaceNotDurableError(RuntimeError):
    """The run directory is on a filesystem a reboot erases. Carries the mount it refused."""


def _mount_table(mounts: Path | None = None) -> list[tuple[str, str]]:
    """Return `(mount_point, fstype)` for every mount, longest mount point first.

    Args:
        mounts: the mount table to parse; `None` reads `MOUNTS` at call time, so a rebound
            module constant is obeyed.

    Returns:
        Mount points paired with their filesystem type, ordered so the first entry whose path
        is a prefix of a target is that target's backing mount.

    Raises:
        WorkspaceNotDurableError: the mount table is unreadable, so durability is UNKNOWN and
            an unknown must halt rather than pass.
    """
    mounts = MOUNTS if mounts is None else mounts
    try:
        raw = mounts.read_text(encoding="utf-8")
    except OSError as exc:
        raise WorkspaceNotDurableError(
            f"the mount table {mounts} is unreadable ({exc}), so whether the run directory "
            "survives a reboot cannot be decided. An undecided durability halts: the failure "
            "this check exists for loses a whole run, and silence is what it looked like."
        ) from exc
    rows: list[tuple[str, str]] = []
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) >= 3:
            rows.append((fields[1].replace("\\040", " "), fields[2]))
    return sorted(rows, key=lambda row: len(row[0]), reverse=True)


def backing_mount(path: Path, mounts: Path | None = None) -> tuple[str, str]:
    """Return the mount point and filesystem type that back `path`.

    Args:
        path: any path, existing or not; it is resolved first.
        mounts: the mount table to parse.

    Returns:
        `(mount_point, fstype)`.

    Raises:
        WorkspaceNotDurableError: no mount point contains the path, or the table is unreadable.
    """
    resolved = path.expanduser().resolve()
    for mount_point, fstype in _mount_table(mounts):
        mount = Path(mount_point)
        if resolved == mount or mount in resolved.parents:
            return mount_point, fstype
    raise WorkspaceNotDurableError(
        f"no mount in {MOUNTS if mounts is None else mounts} contains {resolved}: the backing "
        "filesystem is unidentifiable and durability cannot be decided."
    )


def assert_durable(path: Path, mounts: Path | None = None) -> dict[str, object]:
    """Halt unless the run directory's filesystem survives the machine.

    Args:
        path: the run/out directory a run would write into.
        mounts: the mount table to parse.

    Returns:
        A report naming the path, its backing mount and fstype, and the table it was read from.

    Raises:
        WorkspaceNotDurableError: the backing filesystem is ephemeral, or durability could
            not be decided at all.
    """
    resolved = path.expanduser().resolve()
    mount_point, fstype = backing_mount(resolved, mounts)
    if fstype in EPHEMERAL_FSTYPES:
        raise WorkspaceNotDurableError(
            f"the run directory {resolved} is backed by {mount_point} ({fstype}), which does "
            "not survive the machine. R347(d): the run dir sits on a persistent volume, or "
            "every bundle and shard rsyncs off-box hash-verified within one checkpoint "
            "interval, PROVEN before START. The mirror arm is not provable from this "
            "repository — its evidence comes from off-box tooling that is not here, and a "
            "gate input with no producer is refused (LAW-07). Move the run directory onto a "
            "persistent volume."
        )
    return {"path": str(resolved), "mount_point": mount_point,
            "fstype": fstype, "verdict": "DURABLE",
            "mounts_table": str(MOUNTS if mounts is None else mounts)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mantis.diagnostics.workspace_durability",
        description="Refuse a run directory a reboot would erase.")
    parser.add_argument("path", help="the run/out directory a run would write into")
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = assert_durable(Path(args.path))
    except WorkspaceNotDurableError as exc:
        payload = {"path": args.path, "verdict": "REFUSED", "reason": str(exc)}
        print(json.dumps(payload) if args.json else f"WORKSPACE-DURABILITY HALT: {exc}",
              file=sys.stderr)
        return 2
    print(json.dumps(report) if args.json else
          f"WORKSPACE-DURABILITY PASS: {report['path']} on {report['mount_point']} "
          f"({report['fstype']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
