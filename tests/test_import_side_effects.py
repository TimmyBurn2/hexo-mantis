"""Importing any `mantis` module writes nothing to the working directory."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_IMPORT_ALL = """
import importlib, pkgutil, mantis
for info in pkgutil.walk_packages(mantis.__path__, "mantis."):
    importlib.import_module(info.name)
"""


def test_importing_every_module_leaves_the_cwd_empty(tmp_path: Path) -> None:
    """Killer: a module-level `mkdir` (the old corpus metrics made `reports/` at import)."""
    done = subprocess.run([sys.executable, "-c", _IMPORT_ALL], cwd=tmp_path,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    assert sorted(p.name for p in tmp_path.iterdir()) == []
