"""The position analyzer command (ANALYZER-1); the implementation is `tools/analyzer`, this is the CLI."""
from __future__ import annotations

import importlib

from mantis.util.loadpkg import load_tools_package

load_tools_package("analyzer")
_cli = importlib.import_module("analyzer.cli")

main = _cli.main

if __name__ == "__main__":
    raise SystemExit(main())
