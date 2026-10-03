"""The run monitor's command; the implementation is `tools/run_monitor`, loaded by path (no `sys.path` write)."""
from __future__ import annotations

import importlib

from mantis.util.loadpkg import load_tools_package

load_tools_package("run_monitor")
_cli = importlib.import_module("run_monitor.cli")

main = _cli.main

if __name__ == "__main__":
    raise SystemExit(main())
