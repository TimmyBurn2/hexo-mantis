"""The game viewer command (R352(g)); the implementation is `tools/viewer`, this is the CLI."""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path


def _package():
    """Load `tools/viewer` by path — `tools/` is not a package and `sys.path` is not touched."""
    if "viewer" in sys.modules:
        return sys.modules["viewer"]
    pkg = Path(__file__).resolve().parent / "viewer"
    spec = importlib.util.spec_from_file_location(
        "viewer", pkg / "__init__.py", submodule_search_locations=[str(pkg)])
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the viewer package from {pkg}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["viewer"] = module
    spec.loader.exec_module(module)
    return module


_package()
_cli = importlib.import_module("viewer.cli")

main = _cli.main

if __name__ == "__main__":
    raise SystemExit(main())
