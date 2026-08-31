#!/usr/bin/env bash
# The sealbot rung's BUILD step, pinned. `make vendor` fetches and patches; nothing built the
# extension, so the rung refused for a step that lived only in a record (R324(c)).
#
# It refuses BEFORE building on every precondition — tree absent, sha drifted, patch not
# applied — because a build over the wrong source produces a working `.so` that plays a
# different engine, and no downstream oracle can tell.
set -euo pipefail

cd "$(dirname "$0")/.."
DEST="vendor/external/sealbot"
CURRENT="$DEST/current"

if [ ! -d "$DEST/.git" ]; then
  echo "vendor-build: $DEST is not a fetched pin; run \`make vendor\` from the repo root" >&2
  exit 2
fi

PINNED_SHA="$(python3 -c '
import tomllib, pathlib
pins = tomllib.loads(pathlib.Path("vendor/pins.toml").read_text(encoding="utf-8"))["pins"]
print(pins["sealbot"]["sha"])
')"
HEAD_SHA="$(git -C "$DEST" rev-parse HEAD)"
if [ "$HEAD_SHA" != "$PINNED_SHA" ]; then
  echo "vendor-build: sealbot tree is at $HEAD_SHA, pins.toml says $PINNED_SHA; re-run \`make vendor\`" >&2
  exit 3
fi

# The patch, asserted by its EFFECT rather than by `git diff` being non-empty: a non-empty
# diff proves an edit, not this edit. `-march=native` is the LAW-15 surface DESIGN_A §2.6
# removes; `WIN_THRESHOLD` is what the depth receipt reads.
if grep -q 'march=native' "$CURRENT/setup.py"; then
  echo "vendor-build: -march=native still present in $CURRENT/setup.py — the pinned patch is NOT applied; re-run \`make vendor\`" >&2
  exit 4
fi
if ! grep -q 'WIN_THRESHOLD' "$CURRENT/minimax_bot.cpp"; then
  echo "vendor-build: WIN_THRESHOLD not exported in $CURRENT/minimax_bot.cpp — the pinned patch is NOT applied; re-run \`make vendor\`" >&2
  exit 4
fi

# `--no-project` IS LOAD-BEARING, and it was found the expensive way. Without it, `uv run`
# executed from inside the repo tree RE-SYNCS THE PROJECT VENV — which on this project swaps
# the CUDA torch wheel for the CPU one (the standing torch-select hazard). A NIGHTRUN-1 box
# measurement silently fell back to CPU mid-mission because this line ran between two arms of
# an A/B. The vendored engine's build has nothing to do with the mantis environment, so it
# must not touch it.
( cd "$CURRENT" && uv run --no-project --with pybind11 --with setuptools python setup.py build_ext --inplace )

SO="$(find "$CURRENT" -maxdepth 1 -name 'minimax_cpp*.so' -print -quit)"
if [ -z "$SO" ]; then
  echo "vendor-build: the build reported success but no minimax_cpp*.so landed under $CURRENT/" >&2
  exit 5
fi
echo "vendor-build: sealbot @ ${PINNED_SHA:0:12} built"
echo "vendor-build:   $SO"
echo "vendor-build:   sha256 $(sha256sum "$SO" | cut -d' ' -f1)  bytes $(stat -c%s "$SO")"
