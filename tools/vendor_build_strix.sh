#!/usr/bin/env bash
# The strix rung's BUILD step (RUNG-2, R352(e)): `make vendor` fetches the pin; this builds
# strix's OWN venv (its torch is separate from ours: CPU wheel, its Rust engine `hexo_rs` via
# maturin, the `train` extra for torch_geometric) inside the vendored tree. Refuses before
# building on a missing tree or a drifted sha, because a venv over the wrong source plays a
# different engine. Idempotent: `uv sync` on a warm tree is a no-op.
set -euo pipefail

cd "$(dirname "$0")/.."
DEST="vendor/external/hexo-strix"

if [ ! -d "$DEST/.git" ]; then
  echo "vendor-build: $DEST is not a fetched pin; run \`make vendor\` from the repo root" >&2
  exit 2
fi

PINNED_SHA="$(python3 -c '
import tomllib, pathlib
pins = tomllib.loads(pathlib.Path("vendor/pins.toml").read_text(encoding="utf-8"))["pins"]
print(pins["hexo-strix"]["sha"])
')"
HEAD_SHA="$(git -C "$DEST" rev-parse HEAD)"
if [ "$HEAD_SHA" != "$PINNED_SHA" ]; then
  echo "vendor-build: hexo-strix tree is at $HEAD_SHA, pins.toml says $PINNED_SHA; re-run \`make vendor\`" >&2
  exit 3
fi

# strix's pyproject makes cpu/cuda/rocm torch mutually exclusive groups; the rung plays on the
# CPU by design (the card belongs to the run) and `--all-packages --extra train` pulls the
# workspace members (hexo_rs is built by maturin) and torch_geometric.
(cd "$DEST" && uv sync --group cpu --all-packages --extra train)
"$DEST/.venv/bin/python" -c 'import hexo_rs, hexo_a0.model, torch_geometric; print("vendor-build: strix venv ok, torch", __import__("torch").__version__)'
mkdir -p vendor/external/strix_models
echo "vendor-build: place the pinned checkpoint under vendor/external/strix_models/ (see vendor/pins.toml [pins.hexo-strix])"
