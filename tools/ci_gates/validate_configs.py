"""CI gate 7: every file under configs/ schema-validates.

Discovery is `mantis.config.loader.discover_configs` — the ONE enumeration, shared with gate
12's declaration partition (R71 / ADJ-13 F-1). It used to be an inline `**/*.yaml` + `**/*.yml`
glob here and a FLAT `*.yaml` glob there, so a file this gate blessed could be invisible to gate
12 and never audited.

Under R75's shared-authority invariant that enumeration is **name-agnostic** — every path under
`configs/` that is not a real directory — so this gate now states a stronger law than "the
`.yaml` files parse": **every file under `configs/` is a complete, schema-valid config.** A
stray note, a `.gitkeep` or an editor's `run5.yaml.bak` is a FAILURE here, and that is correct
rather than a false red: the loader reads by content, so any name-based exemption is exactly the
gap four escapes walked through. Notes belong in `docs/`. An EMPTY discovery is a failure
("no configs found") so the gate can never go vacuous. Prints `OK <path>` per valid file;
collects all failures before exiting. Exit 0 all valid; 1 any failure; 2 on internal error.
"""
import sys
from pathlib import Path

import yaml
from pydantic import ValidationError

from mantis.config.loader import discover_configs, load_config


def main() -> int:
    configs_dir = Path("configs")
    files = discover_configs(configs_dir)
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
