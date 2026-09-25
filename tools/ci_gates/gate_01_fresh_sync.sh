#!/usr/bin/env bash
# CI gate 1: a fresh CLONE (the checkout already has build artifacts) syncs and calls COMPILED surface.
# `registry_sha_hex()` must stay a symbol other gates hold down: tests/bridge/test_surface.py,
# crates/mantis-bridge/src/encoding.rs and _engine.pyi pin it; gate 8 uses `registry_sha()`, not this.
set -euo pipefail
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
git clone --quiet . "$tmp/clone"
cd "$tmp/clone"
uv sync --locked
uv run python - <<'PY'
from mantis import _engine

assert _engine.__doc__, "extension imported but carries no docstring"

sha = _engine.registry_sha_hex()
assert isinstance(sha, str) and len(sha) == 64, f"registry_sha_hex() returned {sha!r}"
_engine.Board()  # constructing a Board proves the core kernels actually linked

print(f"gate01 OK: fresh clone built a working extension; registry sha {sha[:12]}")
PY
