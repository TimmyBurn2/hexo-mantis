"""The position analyzer command (ANALYZER-1); the implementation is `tools/analyzer`, this is the CLI."""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path


def _package():
    """Load `tools/analyzer` by path — `tools/` is not a package and `sys.path` is not touched."""
    if "analyzer" in sys.modules:
        return sys.modules["analyzer"]
    pkg = Path(__file__).resolve().parent / "analyzer"
    spec = importlib.util.spec_from_file_location(
        "analyzer", pkg / "__init__.py", submodule_search_locations=[str(pkg)])
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the analyzer package from {pkg}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["analyzer"] = module
    spec.loader.exec_module(module)
    return module


_package()
_cli = importlib.import_module("analyzer.cli")

main = _cli.main

if __name__ == "__main__":
    raise SystemExit(main())
