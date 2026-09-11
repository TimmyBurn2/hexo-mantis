"""Sweep the preflight oracles' probe paths out of the tree, before and after the session.

The oracle that drives the mint preflight with an in-repo `--out-dir` has no `try/finally`, so a
failing guard leaves an untracked artifact directory behind and every later run then fails on
the precondition instead of on the guard. The sweep touches exactly the two paths named below.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

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


def _load_puller():
    spec = importlib.util.spec_from_file_location("_conftest_mirror_pull",
                                                  REPO_ROOT / "tools" / "mirror_pull.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_dirs_under(base: Path) -> list[tuple[Path, str]]:
    """`(run dir, run_id)` for every directory under `base` holding a resume-bundle manifest."""
    found: dict[Path, str] = {}
    for manifest in base.rglob("checkpoints/*.bundle.json"):
        try:
            run_id = str(json.loads(manifest.read_text(encoding="utf-8"))["run_id"])
        except (OSError, ValueError, KeyError):
            continue
        found.setdefault(manifest.parents[1], run_id)
    return sorted(found.items())


@pytest.fixture(scope="module")
def local_puller(tmp_path_factory) -> Iterator[Path]:
    """The R349(b) loop in miniature: the REAL puller as a LOCAL loop over every run directory
    a preflight child writes under pytest's tmp base; module-scoped for the preflight fixtures."""
    puller = _load_puller()
    base = Path(tmp_path_factory.getbasetemp())
    mirrors = tmp_path_factory.mktemp("mirror")
    stop = threading.Event()

    def loop() -> None:
        cycle = 0
        while not stop.is_set():
            cycle += 1
            for run_dir, run_id in _run_dirs_under(base):
                if mirrors in run_dir.parents:
                    continue
                mirror = mirrors / f"{run_dir.parent.name}_{run_dir.name}"
                try:
                    puller.run_cycle(str(run_dir), mirror, run_id, cycle=cycle,
                                     mirror_id="local_puller")
                except puller.MirrorTransportError:
                    continue
            stop.wait(1.0)

    thread = threading.Thread(target=loop, name="local_puller", daemon=True)
    thread.start()
    try:
        yield mirrors
    finally:
        stop.set()
        thread.join(timeout=30)
