"""Sweep the preflight oracles' probe paths out of the tree, before and after the session.

The oracle that drives the mint preflight with an in-repo `--out-dir` has no `try/finally`, so a
failing guard leaves an untracked artifact directory behind and every later run then fails on
the precondition instead of on the guard. The sweep touches exactly the two paths named below.
"""
from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from mantis.diagnostics.workspace_durability import MOUNTS_ENV

REPO_ROOT = Path(__file__).resolve().parents[2]
#: The probe paths from the out-dir and symlink oracles, kept as literals rather than imported
#: so this file is not a consumer of a frozen oracle's internals.
PROBES = (REPO_ROOT / "_preflight_oracle_outdir",
          REPO_ROOT / "_preflight_symlink_probe")


def _sweep() -> None:
    """Remove each probe path, loudly — a silent sweep leaves the poisoned tree it exists to
    prevent, and the next failure is then blamed on the guard.

    The symlink arm runs FIRST because `Path.is_dir()` follows symlinks while `shutil.rmtree`
    refuses them: from a session-scoped autouse fixture that raise was measured at 195 collection
    errors across all of `tests/tools/`.
    """
    for probe in PROBES:
        if probe.is_symlink():
            try:
                probe.unlink()
            except OSError as exc:
                raise RuntimeError(
                    f"could not unlink the preflight probe symlink {probe}: {exc}. Remove it "
                    "by hand — while it exists the probe path is not usable by the oracle."
                ) from exc
            continue
        if not probe.is_dir():
            continue
        try:
            shutil.rmtree(probe)
        except OSError as exc:
            raise RuntimeError(
                f"could not sweep the preflight probe path {probe}: {exc}. It must be removed "
                "by hand — while it exists, `test_an_out_dir_inside_the_repo_is_refused` fails "
                "on its PRECONDITION rather than on the guard it exists to witness, and the "
                "tree carries an untracked artifact directory (R7 / gate 6)."
            ) from exc


# THE one preflight wall-clock budget: a fixed bar on a real boot must clear the slowest host
# that runs it, so it is derived from completed drives rather than transcribed per test.
# Measured: dev host 161.6 s idle / 200.1 s under 698% load (2026-08-18, 16 cores); migration box
# 447 s; CI runner 1205 s (2026-08-19, the slowest budget-governed row, after 900 was measured
# red there at a 915.4 s truncation). 1800 covers the slowest measured row at ~1.5x.
PREFLIGHT_BUDGET_SEC = 1800.0

#: Derived, never transcribed: if `subprocess.run(timeout=...)` fires first the tool never writes
#: the rc-40 report the tests read, and the harness timeout hides the tool's verdict.
PREFLIGHT_HARNESS_CEILING_SEC = PREFLIGHT_BUDGET_SEC + 120.0


@pytest.fixture(scope="session")
def preflight_budget_sec() -> float:
    """Serve the one preflight wall-clock budget to sibling test modules."""
    return PREFLIGHT_BUDGET_SEC


@pytest.fixture(scope="session")
def preflight_harness_ceiling_sec() -> float:
    """Serve the harness ceiling, always above the tool budget."""
    return PREFLIGHT_HARNESS_CEILING_SEC


@pytest.fixture(scope="session", autouse=True)
def _preflight_probe_path_is_not_left_in_the_tree():
    """Sweep before as well as after, so an earlier session's leftovers cannot fail the guard's
    oracle for an unrelated reason."""
    _sweep()
    yield
    _sweep()


@pytest.fixture(scope="module")
def planted_durable_mounts(tmp_path_factory) -> Iterator[Path]:
    """Point a subprocess preflight at a planted table calling `/` durable (pytest's tmp base is
    tmpfs or overlay); module-scoped so a module-scoped preflight fixture sees it."""
    table = tmp_path_factory.mktemp("mounts") / "mounts"
    table.write_text("dev0 / ext4 rw 0 0\n", encoding="utf-8")
    patch = pytest.MonkeyPatch()
    patch.setenv(MOUNTS_ENV, str(table))
    try:
        yield table
    finally:
        patch.undo()
