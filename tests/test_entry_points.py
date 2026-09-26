"""The launch surface is `python -m mantis.*`: no tracked Python file lives outside src/, tests/ or tools/."""
from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_ROOTS = ("src/mantis/", "tests/", "tools/")


def _loose(paths: list[str]) -> list[str]:
    """The Python files outside the package, the tests and the dev tooling."""
    return [p for p in paths if p.endswith(".py") and not p.startswith(_ROOTS)]


def test_no_tracked_python_file_is_a_loose_script() -> None:
    tracked = subprocess.run(["git", "ls-files"], cwd=_REPO, check=True, capture_output=True,
                             text=True).stdout.splitlines()
    assert _loose(tracked) == []


def test_every_console_script_enters_the_package() -> None:
    project = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    targets = project.get("scripts", {}).values()
    assert [t for t in targets if not t.startswith("mantis.")] == []


def test_a_loose_script_is_caught() -> None:
    assert _loose(["run_me.py", "scripts/train.py", "src/mantis/run.py"]) == [
        "run_me.py", "scripts/train.py"]
