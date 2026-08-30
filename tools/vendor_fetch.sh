#!/usr/bin/env bash
# `make vendor`: reads vendor/pins.toml; clones each pin into vendor/external/<name>
# (gitignored) at its exact sha and applies the optional tracked patch. Empty pin table
# is honest empty behavior (exit 0), not a gate.
set -euo pipefail
python3 - <<'PYEOF'
import subprocess
import sys
import tomllib
from pathlib import Path

pins = tomllib.loads(Path("vendor/pins.toml").read_text(encoding="utf-8")).get("pins", {})
if not pins:
    print("vendor: no pins declared; nothing to fetch")
    sys.exit(0)
for name, spec in pins.items():
    url, sha = spec["url"], spec["sha"]
    dest = Path("vendor/external") / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        subprocess.run(["git", "clone", url, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "fetch", "--all"], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", sha], check=True)
    patch = spec.get("patch")
    if patch:
        subprocess.run(["git", "-C", str(dest), "apply", str(Path(patch).resolve())], check=True)
    print(f"vendor: {name} @ {sha[:12]} ready")
PYEOF
