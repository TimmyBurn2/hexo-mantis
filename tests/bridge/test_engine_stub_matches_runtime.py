"""The one `_engine.pyi` sits where the wheel installs it and declares exactly the runtime surface."""
from __future__ import annotations

import ast
import tomllib
from pathlib import Path

from mantis import _engine

_REPO = Path(__file__).resolve().parents[2]
_BRIDGE = _REPO / "crates" / "mantis-bridge"
_MATURIN = tomllib.loads((_BRIDGE / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["maturin"]
_PYTHON_SOURCE = _BRIDGE / _MATURIN["python-source"]
_STUB = _PYTHON_SOURCE / (_MATURIN["module-name"].replace(".", "/") + ".pyi")


def _names(body: list[ast.stmt]) -> set[str]:
    """The public names a stub body binds: defs, classes, annotated and plain assignments."""
    names: set[str] = set()
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return {name for name in names if not name.startswith("_")}


def _stub(path: Path) -> tuple[set[str], dict[str, set[str]]]:
    """The stub's module-level names and, per declared class, its public members."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    classes = {node.name: _names(node.body) for node in tree.body if isinstance(node, ast.ClassDef)}
    return _names(tree.body), classes


def _runtime_members(cls: type) -> set[str]:
    """Public members the class itself defines; `Exception.args` and kin are not its surface."""
    return {name for name in vars(cls) if not name.startswith("_")}


def test_the_one_stub_is_the_wheel_path_pyright_reads() -> None:
    pyright = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["pyright"]
    assert _STUB.is_file(), f"no stub at the path the wheel installs: {_STUB}"
    assert _REPO / pyright["stubPath"] == _PYTHON_SOURCE
    assert sorted(p.relative_to(_REPO) for p in (_REPO / "src").rglob("_engine.pyi")) == []


def test_the_stub_declares_exactly_the_module_surface() -> None:
    declared, _classes = _stub(_STUB)
    exported = {name for name in dir(_engine) if not name.startswith("_")}
    assert sorted(declared - exported) == [], "declared in the stub, absent from the runtime"
    assert sorted(exported - declared) == [], "exported by the runtime, absent from the stub"


def test_every_stub_class_declares_exactly_its_runtime_members() -> None:
    _declared, classes = _stub(_STUB)
    drift = {}
    for name, declared in classes.items():
        live = _runtime_members(getattr(_engine, name))
        if declared != live:
            drift[name] = {"stub_only": sorted(declared - live),
                           "runtime_only": sorted(live - declared)}
    assert drift == {}, f"stub/runtime member drift: {drift}"
