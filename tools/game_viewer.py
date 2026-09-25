"""The game viewer command; the implementation is `tools/viewer`, this is the CLI."""
from __future__ import annotations

import importlib

from mantis.util.loadpkg import load_tools_package

load_tools_package("viewer")
_cli = importlib.import_module("viewer.cli")

main = _cli.main

if __name__ == "__main__":
    raise SystemExit(main())
