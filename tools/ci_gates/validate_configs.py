"""CI gate 7: every file under configs/ schema-validates.

Globs configs/**/*.yaml + *.yml; an EMPTY glob is a failure ("no configs found") so the
gate can never go vacuous. Prints `OK <path>` per valid file; collects all failures
before exiting. Exit 0 all valid; 1 any failure; 2 on internal error.
"""
import sys
from pathlib import Path

import yaml
from pydantic import ValidationError

from mantis.config.loader import load_config


def main() -> int:
    configs_dir = Path("configs")
    files = sorted(configs_dir.glob("**/*.yaml")) + sorted(configs_dir.glob("**/*.yml"))
    if not files:
        print("no configs found under configs/ — gate 7 must never be vacuous", file=sys.stderr)
        return 1
    failures = 0
    for path in files:
        try:
            load_config(path)
        except (ValidationError, yaml.YAMLError, TypeError, OSError) as exc:
            print(f"FAIL {path}: {exc}", file=sys.stderr)
            failures += 1
        else:
            print(f"OK {path}")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception as exc:  # noqa: BLE001 — top-level backstop: internal error -> exit 2 (gate contract)
        print(f"internal error: {exc}", file=sys.stderr)
        rc = 2
    sys.exit(rc)
