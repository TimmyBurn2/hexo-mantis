#!/usr/bin/env bash
# Gate 21: the browser package's wasm build (pinned bindgen CLI), its types, and its node tests (W2 in wasm and its
# break, the Worker's loop, the arena adapter) on a tiny stamp the one exporter writes.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
uv="${UV:-uv}"
(cd "$root/web" && npm ci --no-audit --no-fund --loglevel=error)
make -C "$root" --no-print-directory web
(cd "$root/web" && npx tsc --noEmit)
fixture="$(mktemp -d)"
trap 'rm -rf "$fixture"' EXIT
(cd "$root" && ORT_DISABLE_TELEMETRY=1 $uv run --no-sync python tests/tools/_web_stamps.py "$fixture" >/dev/null)
(cd "$root/web" && MANTIS_WEB_FIXTURE="$fixture" node --test test/*.test.ts)
