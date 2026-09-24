"""The run dashboard command (R333(d)); the implementation is `tools/dashboard`, this is the CLI."""
from __future__ import annotations

import importlib

from mantis.util.loadpkg import load_tools_package

load_tools_package("dashboard")
_cli = importlib.import_module("dashboard.cli")
_health = importlib.import_module("dashboard.health")

main = _cli.main
SIZE_CAP_BYTES = _cli.SIZE_CAP_BYTES
#: R349(b), cited by docs/contracts/event_manifest.md under this module's name.
MIRROR_LAG_WARN_BUNDLES = _health.MIRROR_LAG_WARN_BUNDLES

if __name__ == "__main__":
    raise SystemExit(main())
