"""The run dashboard command (R333(d)); the implementation is `tools/dashboard`, this is the CLI."""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path


def _package():
    """Load `tools/dashboard` by path — `tools/` is not a package and `sys.path` is not touched."""
    if "dashboard" in sys.modules:
        return sys.modules["dashboard"]
    pkg = Path(__file__).resolve().parent / "dashboard"
    spec = importlib.util.spec_from_file_location(
        "dashboard", pkg / "__init__.py", submodule_search_locations=[str(pkg)])
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the dashboard package from {pkg}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["dashboard"] = module
    spec.loader.exec_module(module)
    return module


_package()
_cli = importlib.import_module("dashboard.cli")
_health = importlib.import_module("dashboard.health")

main = _cli.main
SIZE_CAP_BYTES = _cli.SIZE_CAP_BYTES
#: R349(b), cited by docs/contracts/event_manifest.md under this module's name.
MIRROR_LAG_WARN_BUNDLES = _health.MIRROR_LAG_WARN_BUNDLES

if __name__ == "__main__":
    raise SystemExit(main())
