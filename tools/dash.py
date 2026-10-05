"""The DASH-2 command; the implementation is `tools/dash`, loaded relative to this file so a second checkout serves itself."""
from __future__ import annotations

import importlib
from pathlib import Path

from mantis.util.loadpkg import load_tools_package

load_tools_package("dash", repo_root=Path(__file__).resolve().parents[1])
_cli = importlib.import_module("dash.cli")

main = _cli.main

if __name__ == "__main__":
    raise SystemExit(main())
