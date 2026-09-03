"""AUDIT-1 F-08 — every process entry installs THE mantis stderr handler.

THE DEFECT. `monitor/logging_setup.py::configure_logging` — whose own docstring calls itself
"the ONE mantis stderr handler" — had ZERO callers. `mantis.run`, `mantis.monitor.supervise`,
the eval worker child and every diagnostic installed no root handler at all, so Python's
`lastResort` dropped every `logger.info` and printed WARNING+ unformatted and unattributed.

WHAT WAS INVISIBLE, and this is why it is an R1 finding rather than a cosmetic one:

* every INFO diagnostic a run emits, for the whole run;
* the registry-sha handshake's SKIP reason (`_LOG.info` in `mantis.encoding`), whose own
  docstring says the skip is "NEVER a silent pass" — it was exactly that;
* REPAIR-1's new `disk_guard_error` warning, which F-11 landed so a dead disk guard would be
  visible. It went to an unformatted lastResort line at best.

`pretrain/cli.py` used `logging.basicConfig` instead — two bootstraps for one sink, one of them
dead, which is the duplicate-authority shape with the live one losing.

NOT ASSERTED HERE: that any particular line reaches a terminal. What is asserted is that the
handler EXISTS on the root logger after each entry runs its setup, which is the thing whose
absence made every line above unobservable.
"""
from __future__ import annotations

import ast
import logging
from pathlib import Path

import pytest

from mantis.monitor.logging_setup import _MANTIS_HANDLER, configure_logging

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src" / "mantis"
#: Every process entry point: `python -m mantis.run`, the supervisor, the eval child, and the
#: pretrain CLI. Each is a separate PROCESS, so each needs its own installation — a handler on
#: the parent's root logger says nothing about a spawned child's.
_ENTRIES = {
    "run.py": "main",
    "monitor/supervise.py": "main",
    "eval/worker.py": "_main",
    "train/pretrain/cli.py": None,  # module-level in `pretrain()`; matched by source census
}


@pytest.fixture(autouse=True)
def _restore_root_handlers():
    """This suite installs handlers on the ROOT logger; leaving them behind would change every
    later test's logging. Snapshot and restore."""
    root = logging.getLogger()
    saved, level = list(root.handlers), root.level
    yield
    root.handlers[:] = saved
    root.setLevel(level)


def test_configure_logging_installs_a_tagged_handler() -> None:
    root = logging.getLogger()
    root.handlers[:] = []
    handler = configure_logging()
    assert handler in root.handlers
    assert any(getattr(h, _MANTIS_HANDLER, False) for h in root.handlers), (
        "the installed handler carries no mantis tag, so the idempotence check that removes a "
        "previous one cannot find it"
    )


def test_it_is_idempotent_so_a_relaunch_does_not_double_every_line() -> None:
    root = logging.getLogger()
    root.handlers[:] = []
    configure_logging()
    configure_logging()
    tagged = [h for h in root.handlers if getattr(h, _MANTIS_HANDLER, False)]
    assert len(tagged) == 1, f"{len(tagged)} mantis handlers — a resumed run would double-log"


def test_an_info_line_actually_reaches_the_handler(caplog) -> None:
    """The behavioural half: with the handler installed the root logger is at INFO, which is
    what `lastResort` was NOT — it drops everything below WARNING."""
    root = logging.getLogger()
    root.handlers[:] = []
    configure_logging()
    assert root.level <= logging.INFO, (
        "the root logger is above INFO, so every `logger.info` in the tree is still dropped — "
        "which is the state AUDIT-1 F-08 found"
    )


@pytest.mark.parametrize("rel", sorted(_ENTRIES))
def test_every_process_entry_installs_the_handler(rel: str) -> None:
    """Source census, because the alternative is launching four processes. Each entry module
    must both IMPORT and CALL `configure_logging` — importing without calling is the state the
    module was in for its whole life."""
    tree = ast.parse((_SRC / rel).read_text(encoding="utf-8"), filename=rel)
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "configure_logging"
    ]
    assert calls, (
        f"{rel} never calls `configure_logging`. It is a process entry, so nothing else will "
        "install a handler for it and every INFO line it emits is dropped (AUDIT-1 F-08)."
    )


def test_basic_config_is_not_used_anywhere_in_src() -> None:
    """The second bootstrap is gone. `logging.basicConfig` and `configure_logging` are two
    authorities over one sink, and the tree had both with the live one losing."""
    offenders = [
        f"{path.relative_to(_SRC)}:{node.lineno}"
        for path in sorted(_SRC.rglob("*.py"))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "basicConfig"
    ]
    assert not offenders, (
        f"`logging.basicConfig` under src/: {offenders}. `configure_logging` is the one sink "
        "bootstrap (AUDIT-1 F-08)."
    )
