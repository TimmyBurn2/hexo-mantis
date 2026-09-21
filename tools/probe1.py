"""PROBE-1's command (R365(b)); the implementation is `tools/probe1`, loaded by path with `analyzer` (no `sys.path` write)."""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path


def _package(name: str):
    """Load `tools/<name>` by path — `tools/` is not a package and `sys.path` is not touched."""
    if name in sys.modules:
        return sys.modules[name]
    pkg = Path(__file__).resolve().parent / name
    spec = importlib.util.spec_from_file_location(name, pkg / "__init__.py", submodule_search_locations=[str(pkg)])
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the {name} package from {pkg}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_package("analyzer")
_package("probe1")
_cli = importlib.import_module("probe1.cli")

main = _cli.main

if __name__ == "__main__":
    raise SystemExit(main())
