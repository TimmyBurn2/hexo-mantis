"""B-5 (R355(e)): no `uv run` in the Makefile or any tools/ script may re-sync the venv (CPU torch on the box)."""
from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_UV_RUN = re.compile(r"(?:\$\(UV\)|\buv)\s+run\b")
#: The one script whose job IS the sync.
_SYNC_IS_THE_POINT = {"gate_01_fresh_sync.sh"}


def makefile_offenders(text: str) -> list[str]:
    """Recipe lines running `uv run` without `--no-sync` or a leading `UV_NO_SYNC=1`."""
    out: list[str] = []
    for n, line in enumerate(text.split("\n"), 1):
        if _UV_RUN.search(line) and "--no-sync" not in line and "UV_NO_SYNC=1" not in line:
            out.append(f"Makefile:{n}")
    return out


def script_offenders(name: str, text: str) -> list[str]:
    """A script exports `UV_NO_SYNC=1` before its first `uv run`; `--no-sync`/`--no-project` runs pass."""
    if name in _SYNC_IS_THE_POINT:
        return []
    exported_at = None
    for n, line in enumerate(text.split("\n"), 1):
        if re.match(r"\s*export\s+UV_NO_SYNC=1\b", line):
            exported_at = n
            break
    for n, line in enumerate(text.split("\n"), 1):
        if line.lstrip().startswith("#"):
            continue
        if "--no-sync" in line or "--no-project" in line:
            continue
        if _UV_RUN.search(line) and (exported_at is None or n < exported_at):
            return [f"{name}:{n}"]
    return []


def test_the_checkers_bite() -> None:
    assert makefile_offenders("test:\n\t$(UV) run pytest\n") == ["Makefile:2"]
    assert makefile_offenders("test:\n\t$(UV) run --no-sync pytest\n") == []
    assert script_offenders("x.sh", "uv run python -c 1\nexport UV_NO_SYNC=1\n") == ["x.sh:1"]
    assert script_offenders("x.sh", "export UV_NO_SYNC=1\nuv run python -c 1\n") == []
    assert script_offenders("x.sh", "uv run --no-project python setup.py\n") == []


def test_no_uv_run_in_the_makefile_or_the_gate_scripts_resyncs() -> None:
    offenders = makefile_offenders((_REPO / "Makefile").read_text(encoding="utf-8"))
    for script in sorted((_REPO / "tools").rglob("*.sh")):
        offenders += script_offenders(script.name, script.read_text(encoding="utf-8"))
    assert offenders == [], f"a bare `uv run` re-syncs the venv to the CPU wheel: {offenders}"
