"""PROBE-1's command (R365(b)); the implementation is `tools/probe1`, loaded by path with `analyzer` (no `sys.path` write)."""
from __future__ import annotations

import importlib

from mantis.util.loadpkg import load_tools_package

load_tools_package("analyzer")
load_tools_package("probe1")
_cli = importlib.import_module("probe1.cli")

main = _cli.main

if __name__ == "__main__":
    raise SystemExit(main())
