"""The value instrument's command; the implementation is `tools/value_instrument`, loaded by path (no `sys.path` write)."""
from __future__ import annotations

import importlib

from mantis.util.loadpkg import load_tools_package

load_tools_package("value_instrument")
_cli = importlib.import_module("value_instrument.cli")

main = _cli.main

if __name__ == "__main__":
    raise SystemExit(main())
