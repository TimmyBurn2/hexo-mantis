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

Deliberately narrow: it removes exactly the one probe path the frozen oracle names, exactly
when that path is a directory, and it never touches anything else in the tree.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
#: The literal probe path from `test_an_out_dir_inside_the_repo_is_refused`. Kept as a
#: literal rather than imported, because importing it would make this file a consumer of the
#: frozen oracle's internals and couple the two more tightly than a name in a docstring.
PROBE = REPO_ROOT / "_preflight_oracle_outdir"


def _sweep() -> None:
    if PROBE.is_dir():
        shutil.rmtree(PROBE, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def _preflight_probe_path_is_not_left_in_the_tree():
    """Swept BEFORE as well as after: a poisoned tree left by an earlier session must not
    make the guard's oracle fail for a reason unrelated to the guard."""
    _sweep()
    yield
    _sweep()
