"""Load a `tools/<name>` directory as a package by path — `tools/` is not a package and `sys.path` is never written."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_REPO_ROOT = Path(__file__).resolve().parents[3]


def load_tools_package(name: str, *, repo_root: Path | None = None) -> ModuleType:
    """Import `<repo>/tools/<name>` as the top-level module `name`, once per process. Raises: ImportError."""
    if name in sys.modules:
        return sys.modules[name]
    pkg = (_REPO_ROOT if repo_root is None else Path(repo_root)) / "tools" / name
    init = pkg / "__init__.py"
    if not init.is_file():
        raise ImportError(f"no tools package {name!r} under {pkg}")
    spec = importlib.util.spec_from_file_location(
        name, init, submodule_search_locations=[str(pkg)])
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the {name} package from {pkg}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


__all__ = ["load_tools_package"]
