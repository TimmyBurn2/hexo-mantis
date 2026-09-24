"""Spec-load a repo file as a top-level module by path; sys.path is never written."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_module_by_path(name: str, path: Path) -> ModuleType:
    """Import `path` as the top-level module `name`. Raises: ImportError if unloaded."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the module {name!r} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
