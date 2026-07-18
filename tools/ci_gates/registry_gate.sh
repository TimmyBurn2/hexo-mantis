#!/usr/bin/env bash
# CI gate 8: registry-sha handshake (ARMED as of WP7) + audit CLI.
#
# Handshake sub-check — LIVE. At `import mantis.encoding` the on-disk registry.toml is
# hashed and compared to the compiled `_engine.registry_sha()`; a drift HARD-ERRORS, so a
# stale extension cannot serve a stale registry. This gate re-proves that comparison AND
# carries its own LAW-07 mutation self-test (a drifted TOML must raise) so the check can
# never silently become a phantom gate. The pytest producer is
# tests/encoding/test_registry_sha_handshake.py.
#
# audit exit-0 sub-check — DEFERRED to the cutover gate. `python -m mantis.encoding audit`
# is unsatisfiable mid-migration (checkpoints/ data/ configs/variants absent -> sections
# 2/3/4 warn; the section-5 hardcode scan is not yet allowlist-clean), and
# migration_plan.md places "audit CLI exit 0" in the cutover battery, not per-WP. It arms
# here once the tree is populated post-cutover.
set -euo pipefail

REG="crates/mantis-encoding/src/registry.toml"
[ -f "$REG" ] || { echo "gate 8: FAIL — expected registry at $REG (ported in WP3); absent."; exit 1; }

# Handshake sub-check + inline LAW-07 mutation self-test (imports the shipped helper).
uv run python - "$REG" <<'PY'
import hashlib, pathlib, shutil, sys, tempfile
from mantis import _engine
from mantis.encoding import (
    EncodingRegistryError,
    _registry_sha_handshake,
    _resolve_registry_toml,
)

reg = pathlib.Path(sys.argv[1])
# 1) the shipped on-disk registry matches the compiled sha (the live handshake).
disk = hashlib.sha256(reg.read_bytes()).digest()
assert disk == _engine.registry_sha(), (
    "gate 8: on-disk registry.toml sha != compiled _engine.registry_sha()"
)
# 2) in-repo resolution actually finds the on-disk TOML (not the installed-skip path).
assert _resolve_registry_toml() is not None, "gate 8: _resolve_registry_toml() failed in-repo"
# 3) LAW-07 mutation self-test — the guard BITES on a drifted registry.
with tempfile.TemporaryDirectory() as d:
    mutated = pathlib.Path(d) / "registry.toml"
    shutil.copyfile(reg, mutated)
    mutated.write_bytes(mutated.read_bytes() + b"\n# drift injected by gate 8 self-test\n")
    try:
        _registry_sha_handshake(mutated)
    except EncodingRegistryError:
        pass
    else:
        raise SystemExit("gate 8: mutation self-test did NOT bite — handshake is a phantom gate")
print("gate 8: registry-sha handshake ARMED + PASS; mutation self-test BITES.")
PY

echo "gate 8: 'python -m mantis.encoding audit' exit-0 sub-check DEFERRED to the cutover gate"
echo "        (unsatisfiable mid-migration; migration_plan cutover battery owns it)."
