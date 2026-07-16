#!/usr/bin/env bash
# CI gate 8: registry sha handshake + audit. Auto-arming stub: the registry does not
# exist until the encoding port lands (accepted debt, recorded in that work package).
# The sha-handshake comparison (_engine.registry_sha() vs sha256 of the on-disk TOML)
# is appended here by the work package that ships registry_sha(); the existence trigger
# below means a registry cannot land without this gate going live.
set -euo pipefail
if [ ! -f crates/mantis-encoding/registry.toml ]; then
  echo "gate 8: registry not yet ported; gate is armed and will trigger automatically"
  echo "        when crates/mantis-encoding/registry.toml exists (accepted WP3 debt)."
  exit 0
fi
uv run python -m mantis.encoding audit
