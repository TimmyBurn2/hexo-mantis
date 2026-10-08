"""The distillation harness's command; the implementation is `tools/distill`, loaded by path (no `sys.path` write)."""
from __future__ import annotations

import importlib

from mantis.util.loadpkg import load_tools_package

load_tools_package("distill")
_cli = importlib.import_module("distill.cli")

main = _cli.main

if __name__ == "__main__":
    raise SystemExit(main())
