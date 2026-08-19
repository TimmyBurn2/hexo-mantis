"""Tree hygiene for the gate-script oracles under `tests/tools/`.

SF-I7, closed as far as it can be closed WITHOUT touching a byte-frozen oracle.

`tests/tools/test_preflight_mint.py:701-712` (`test_an_out_dir_inside_the_repo_is_refused`,
hash `bd8e65e682c6a2dc`) drives the mint preflight with `--out-dir <repo>/_preflight_oracle_
outdir` and opens with the precondition `assert not inside.exists()`. When the guard under
test FAILS, the tool creates that directory — with `logs/` and `checkpoints/` inside it — and
the oracle has no `try/finally` to remove it. Two consequences, both measured by REVIEW-impl
this phase: an untracked artifact directory is manufactured inside the tree (R7 / gate 6, by
the very gate that exists to prevent exactly that), and every SUBSEQUENT run fails on the
PRECONDITION rather than on the guard — so the sole witness stops witnessing after its first
genuine failure.

The correct fix is a `try/finally` + `rmtree` inside that test. It is byte-frozen, editing it
is an R43 event, and reshaping a frozen oracle to fit a fix destroys the evidence value of
every stage before it — so the fix pass STOPPED there and this is the remedy that does not
touch it. `CARD-PREFLIGHT-ORACLE-OUTDIR-CLEANUP` carries the real one for whoever lifts the
freeze.

Deliberately narrow: it removes exactly the two probe paths named below, exactly when each is
a directory, and it never touches anything else in the tree.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
#: The literal probe path from `test_an_out_dir_inside_the_repo_is_refused`. Kept as a
#: literal rather than imported, because importing it would make this file a consumer of the
#: frozen oracle's internals and couple the two more tightly than a name in a docstring.
#:
#: The second entry is ADJ-13 F-2's symlink probe target: `_checked_out_dir` compared an
#: `abspath`-normalised `--out-dir` against a `.resolve()`d git toplevel, so a SYMLINK
#: pointing inside the tree was not refused and the tool created its target. The test owns a
#: `try/finally`; this sweep is the backstop for the case where the guard regresses and the
#: test dies before its `finally` runs.
PROBES = (REPO_ROOT / "_preflight_oracle_outdir",
          REPO_ROOT / "_preflight_symlink_probe")


def _sweep() -> None:
    """ADJ-13 N-2: LOUD. This was `shutil.rmtree(..., ignore_errors=True)` — a sweep whose
    failure mode is silence leaves exactly the poisoned tree the file exists to prevent, and
    the next session's failure is then attributed to the guard rather than to the sweep
    (LAW-14's shape: a cleanup that cannot fail out loud is a cleanup nobody can trust).

    Recheck R-7 — the SYMLINK arm, and it is not a nicety. `Path.is_dir()` FOLLOWS symlinks
    while `shutil.rmtree` REFUSES them ("Cannot call rmtree on a symbolic link"), so a symlink
    at either probe path sent the loud version straight into its own `RuntimeError` — from a
    session-scoped autouse fixture, i.e. **195 collection errors across all of `tests/tools/`**,
    including every row of gate 11's corpus, which has nothing to do with the preflight.
    Measured. A backstop that can take out 195 unrelated tests is not an improvement on the
    silent version, so the symlink is `unlink`ed (the correct removal for a symlink) BEFORE the
    directory test — and loudness is kept for the case it was added for.
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


# ── the ONE preflight wall-clock budget (R46 loop, R284(f)) ───────────────────────────
#
# THREE tests drove the real preflight tool with three separately-transcribed `--timeout-sec`
# constants — 300, 400, 400 — and a fourth number (`subprocess.run(timeout=500)`) coupled to one
# of them. Four authorities for one quantity, none of them derived from anything, all of them
# silently host-dependent: the budget bounds a REAL boot + burst + terminal eval, so what it
# actually encodes is "how fast is the machine", which is not a property any test can transcribe.
#
# The 300 went red on the migration box (`ADJUDICATION_QUEUE.md`: rc 40, the tool's own timeout,
# child rc 48). MEASURED HERE, 2026-08-18, on a 16-core host:
#
#     the same row, ALONE on an idle host          161.6 s   (passed)
#     the same row, under 698% synthetic CPU load  200.1 s   (passed)
#
# So this host does NOT reproduce the failure, with or without load — which means the box's
# >300 s is HARDWARE, not contention, and no amount of deflaking here can find it. The one
# number that IS a measurement of the slow host is the sitting record's completed drive: the
# same command, given 900 s, exited rc 0 in **447 s**.
#
# The budget is therefore set to cover the SLOWEST MEASURED HOST with margin, in one place, with
# its grounds beside it. It is NOT a speedup and does not pretend to be: it is the honest
# statement that a fixed wall-clock bar on a real boot must clear the slowest machine that runs
# it, and 300 did not. Raising it costs nothing on a passing run (this host exits at ~162 s) and
# buys a longer wait only on a genuinely hung one.
#
# LAW-15's shape, one level down: a bar must be a reproducible instrument, and a bar that passes
# on one host and fails on another is measuring the host.
#
# 2026-08-19: the slowest host that runs this tier is now the GitHub CI runner, and 900 is
# measured RED there — three budget rows truncated at 915.4 s by the tool's own rc 40 (run
# 32214255298, public check-run annotations), while the SAME 16-step burst that takes ~155 s
# here completed on that runner in 1190 s and 1198 s (the two run-launcher rows in the same
# run's digest — the only budget-shaped workloads that finished there). True budget-row
# duration on the runner is therefore ~1200 s, and the prior 900 was a sub-measurement guess.
# THIS VALUE IS A MEASUREMENT POSTURE: 3000 is deliberately high so the next CI run measures
# every budget row's true runner duration through the public digest instead of truncating it
# at the bar; the follow-up commit derives the final budget from those readings (slowest
# measured row, with margin), closing the guess-the-host-speed loop that produced both the
# 300 -> 900 move and this one.
PREFLIGHT_BUDGET_SEC = 3000.0

#: The harness ceiling that must always exceed the tool's own budget, DERIVED rather than
#: transcribed — if `subprocess.run(timeout=...)` fires first, the tool never gets to write the
#: rc-40 report the tests read, and the failure is reported as a harness timeout instead of as
#: the tool's verdict.
PREFLIGHT_HARNESS_CEILING_SEC = PREFLIGHT_BUDGET_SEC + 120.0


@pytest.fixture(scope="session")
def preflight_budget_sec() -> float:
    """THE preflight wall-clock budget. A fixture and not an import: R5 bars cross-test imports,
    and a conftest fixture is the sanctioned way to share one value across sibling test modules."""
    return PREFLIGHT_BUDGET_SEC


@pytest.fixture(scope="session")
def preflight_harness_ceiling_sec() -> float:
    """The harness ceiling, always > the tool budget (derived above, never transcribed)."""
    return PREFLIGHT_HARNESS_CEILING_SEC


@pytest.fixture(scope="session", autouse=True)
def _preflight_probe_path_is_not_left_in_the_tree():
    """Swept BEFORE as well as after: a poisoned tree left by an earlier session must not
    make the guard's oracle fail for a reason unrelated to the guard."""
    _sweep()
    yield
    _sweep()
