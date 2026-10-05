"""The census around DASH-2: no `src/mantis` module reaches `tools`, importing the package opens no socket, nothing writes but the freeze."""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = REPO_ROOT / "tools" / "dash"
#: The one module allowed to write a file: the freeze's `--out`.
WRITERS = {"cli.py"}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def test_no_mantis_module_imports_tools_or_the_dash_package():
    offenders = [str(p.relative_to(REPO_ROOT)) for p in (REPO_ROOT / "src" / "mantis").rglob("*.py")
                 if {n.split(".")[0] for n in _imports(p)} & {"tools", "dash"}]
    assert offenders == []


def test_importing_every_dash_module_binds_no_socket():
    script = (
        "import importlib, socket, sys\n"
        "from pathlib import Path\n"
        "from mantis.util.loadpkg import load_tools_package\n"
        "def refuse(*a, **k):\n    raise SystemExit('a socket was bound at import')\n"
        "socket.socket.bind = refuse\n"
        "root = Path(sys.argv[1])\n"
        "load_tools_package('dash', repo_root=root)\n"
        "pkg = root / 'tools' / 'dash'\n"
        "for p in sorted(pkg.rglob('*.py')):\n"
        "    importlib.import_module('.'.join(('dash', *p.relative_to(pkg).with_suffix('').parts)).removesuffix('.__init__'))\n"
        "print('ok')\n")
    out = subprocess.run([sys.executable, "-c", script, str(REPO_ROOT)], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr[-2000:]


@pytest.mark.parametrize("path", sorted(PACKAGE.rglob("*.py")), ids=lambda p: str(p.relative_to(PACKAGE)))
def test_no_module_but_the_cli_writes_a_file(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    writes = [node.lineno for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and (node.func.attr in ("write_text", "write_bytes", "mkdir", "unlink", "rename", "touch")
                   or (node.func.attr == "open" and any(isinstance(a, ast.Constant) and isinstance(a.value, str)
                                                        and set(a.value) & set("wax+") for a in node.args)))]
    allowed = path.name in WRITERS and path.parent == PACKAGE
    assert allowed or writes == [], f"{path.name} writes at lines {writes}"
