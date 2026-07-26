#!/usr/bin/env bash
# CI gate 1 (repo_design §9.1): fresh clone -> uv sync builds the extension -> it WORKS.
# "clone-and-run is the product", so this must clone; running in the existing checkout
# would test a tree that already has build artifacts and is not what a user gets.
#
# The smoke assertion deliberately exercises COMPILED surface rather than just importing:
# an import can succeed against a stale/partial build.
#
# `registry_sha_hex()` is chosen because it cannot be renamed or deleted without another
# gate failing loudly in the same run. It is pinned in four places, by gates 2 and 3:
#   tests/bridge/test_surface.py:29,73        (MODULE_FNS census + a value assertion)
#   crates/mantis-bridge/src/encoding.rs:213  (registry_sha_hex_matches_raw_bytes)
#   crates/mantis-bridge/python/mantis/_engine.pyi:650
# NOTE: gate 8 (tools/ci_gates/registry_gate.sh:34) uses the sibling `registry_sha()` —
# the raw 32-byte digest — NOT this hex form, so gate 8 is NOT what protects this symbol.
# That distinction is recorded because getting it wrong is how a smoke check ends up
# asserting a symbol nothing else holds down, which is exactly the failure this gate
# suffered from WP7 to WPUF-2: it asserted a `hello()` scaffold symbol that had been
# deleted, and nothing noticed for the whole migration.
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
