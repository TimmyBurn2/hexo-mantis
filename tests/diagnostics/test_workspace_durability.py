"""R347(d) — the START pre-flight HALT on a run directory a reboot erases.

`workspace_is_volume = false` was a noted risk with no mechanism: the run wrote into ephemeral
storage, the machine went away, and the loss looked exactly like silence. Every arm below is
driven against a SYNTHETIC mount table so the halt is exercised on any host, and the two
undecided cases (unreadable table, path under no mount) are asserted to HALT rather than pass —
an unknown durability is the case the risk was actually made of.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from mantis.diagnostics import workspace_durability as wd


def _table(tmp_path: Path, rows: list[tuple[str, str]]) -> Path:
    path = tmp_path / "mounts"
    path.write_text(
        "".join(f"dev{i} {point} {fstype} rw 0 0\n" for i, (point, fstype) in enumerate(rows)),
        encoding="utf-8")
    return path


def test_an_ephemeral_backing_filesystem_HALTS(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    mounts = _table(tmp_path, [("/", "ext4"), (str(tmp_path), "tmpfs")])
    with pytest.raises(wd.WorkspaceNotDurableError) as caught:
        wd.assert_durable(run_dir, mounts)
    message = str(caught.value)
    assert "tmpfs" in message and str(run_dir.resolve()) in message
    assert "persistent volume" in message, "the halt must name the repair"
    assert "LAW-07" in message, "the mirror arm is refused for a NAMED reason, not silently"


def test_a_durable_backing_filesystem_PASSES(tmp_path: Path) -> None:
    """The control. Without it the halt could be an unconditional raise."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    mounts = _table(tmp_path, [("/", "ext4"), (str(tmp_path), "xfs")])
    report = wd.assert_durable(run_dir, mounts)
    assert report["verdict"] == "DURABLE"
    assert report["fstype"] == "xfs"
    assert report["mount_point"] == str(tmp_path)


def test_the_NEAREST_mount_decides_not_the_first_one_listed(tmp_path: Path) -> None:
    """An ephemeral mount nested inside a durable one is the one the run writes to. Ordering by
    mount-point length is the mechanism; a plain scan order would report the parent."""
    run_dir = tmp_path / "outer" / "inner" / "run"
    run_dir.mkdir(parents=True)
    mounts = _table(tmp_path, [(str(tmp_path / "outer" / "inner"), "tmpfs"),
                               (str(tmp_path / "outer"), "ext4"),
                               ("/", "ext4")])
    with pytest.raises(wd.WorkspaceNotDurableError, match="tmpfs"):
        wd.assert_durable(run_dir, mounts)


def test_an_unreadable_mount_table_HALTS_rather_than_passing(tmp_path: Path) -> None:
    """An undecided durability is not a green one."""
    with pytest.raises(wd.WorkspaceNotDurableError, match="unreadable"):
        wd.assert_durable(tmp_path, tmp_path / "no-such-mounts")


def test_a_path_under_no_mount_HALTS(tmp_path: Path) -> None:
    mounts = _table(tmp_path, [("/nowhere-at-all", "ext4")])
    with pytest.raises(wd.WorkspaceNotDurableError, match="unidentifiable"):
        wd.assert_durable(tmp_path, mounts)


def test_a_mount_point_containing_a_space_is_parsed(tmp_path: Path) -> None:
    """/proc/mounts octal-escapes a space; splitting on whitespace without un-escaping would
    make such a mount unmatchable and silently fall through to its parent."""
    spaced = tmp_path / "a b"
    spaced.mkdir()
    path = tmp_path / "mounts"
    path.write_text(f"dev0 / ext4 rw 0 0\ndev1 {str(spaced).replace(' ', chr(92) + '040')} "
                    "tmpfs rw 0 0\n", encoding="utf-8")
    with pytest.raises(wd.WorkspaceNotDurableError, match="tmpfs"):
        wd.assert_durable(spaced, path)


def _live_mount(*, ephemeral: bool) -> str:
    rows = wd._mount_table()
    for point, fstype in rows:
        if (fstype in wd.EPHEMERAL_FSTYPES) is ephemeral and Path(point).is_dir():
            return point
    pytest.skip(f"loud skip: this host has no {'ephemeral' if ephemeral else 'durable'} mount "
                "to drive the CLI arm against")


def test_the_real_mount_table_parses_and_decides() -> None:
    """The parse runs against the kernel's own table, not only a synthetic one, and the
    verdict it reaches agrees with the table it read."""
    point, fstype = wd.backing_mount(Path.cwd())
    assert fstype and Path(point).is_dir()
    if fstype in wd.EPHEMERAL_FSTYPES:
        with pytest.raises(wd.WorkspaceNotDurableError, match=fstype):
            wd.assert_durable(Path.cwd())
    else:
        assert wd.assert_durable(Path.cwd())["fstype"] == fstype


def test_the_cli_refuses_with_rc_2_and_says_so() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "mantis.diagnostics.workspace_durability",
         _live_mount(ephemeral=True), "--json"],
        capture_output=True, text=True, check=False)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert json.loads(proc.stderr)["verdict"] == "REFUSED"


def test_the_cli_passes_with_rc_0_on_a_durable_path() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "mantis.diagnostics.workspace_durability",
         _live_mount(ephemeral=False)],
        capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.startswith("WORKSPACE-DURABILITY PASS:"), proc.stdout
