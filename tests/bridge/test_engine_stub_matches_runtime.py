"""Both `_engine.pyi` stubs declare exactly the surface the compiled `mantis._engine` exports."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from mantis import _engine

_REPO = Path(__file__).resolve().parents[2]
_STUBS = (_REPO / "src" / "mantis" / "_engine.pyi",
          _REPO / "crates" / "mantis-bridge" / "python" / "mantis" / "_engine.pyi")


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


@pytest.mark.parametrize("path", _STUBS, ids=lambda p: str(p.relative_to(_REPO)))
def test_the_stub_declares_exactly_the_module_surface(path: Path) -> None:
    declared, _classes = _stub(path)
    exported = {name for name in dir(_engine) if not name.startswith("_")}
    assert sorted(declared - exported) == [], "declared in the stub, absent from the runtime"
    assert sorted(exported - declared) == [], "exported by the runtime, absent from the stub"


@pytest.mark.parametrize("path", _STUBS, ids=lambda p: str(p.relative_to(_REPO)))
def test_every_stub_class_declares_exactly_its_runtime_members(path: Path) -> None:
    _declared, classes = _stub(path)
    drift = {}
    for name, declared in classes.items():
        live = _runtime_members(getattr(_engine, name))
        if declared != live:
            drift[name] = {"stub_only": sorted(declared - live),
                           "runtime_only": sorted(live - declared)}
    assert drift == {}, f"stub/runtime member drift: {drift}"
