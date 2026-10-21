#!/usr/bin/env bash
# Release (design §10), after the operator's words on the licence, the name and the stamp of record: the stamp exported
# on a clean tree into dist/net, the package built and its node tests run on the real files, the WebGPU witness on
# this desk, then `npm pack`. Publishing is the operator's act; this hands over the tarball and its sha256.
# Usage: web/scripts/release.sh STAMP.ckpt
set -euo pipefail
stamp="$1"
web="$(cd "$(dirname "$0")/.." && pwd)"
root="$(cd "$web/.." && pwd)"
if [ -n "$(git -C "$root" status --porcelain --untracked-files=no)" ]; then
    echo "release.sh: the tree has tracked changes; a release exports from a clean commit" >&2
    exit 2
fi
(cd "$web" && npm ci --no-audit --no-fund --loglevel=error)
make -C "$root" --no-print-directory web
reference="$(mktemp --suffix=.json)"
trap 'rm -f "$reference"' EXIT
(cd "$root" && ORT_DISABLE_TELEMETRY=1 uv run --no-sync python tools/export_web.py --checkpoint "$stamp" \
    --out "$web/dist/net" --reference "$reference")
manifest="$(basename "$stamp" .ckpt).manifest.json"
node -e '
const m = require(process.argv[1]);
if (m.exporter.dirty !== false) { console.error("release.sh: the manifest says the export ran on a dirty tree"); process.exit(2); }
if (!m.files.every((f) => f.passes_its_line !== false || f.dtype === "fp16")) { console.error("release.sh: a shipped file misses its line"); process.exit(1); }
' "$web/dist/net/$manifest"
printf '{"manifest": "%s"}\n' "$manifest" > "$web/dist/net/index.json"
rm -f "$web/dist/net/"*.fp16.onnx  # fp16 ships only after its own read (R399(a)); run13's 120k missed it
(cd "$web" && MANTIS_WEB_FIXTURE="$web/dist" node --test test/*.test.ts)
(cd "$web" && node scripts/webgpu.mjs dist/net "$manifest" "$reference" "$web/dist/net/webgpu_witness.json")
node -e '
const r = require(process.argv[1]);
const failed = !r.ok || r.thirdPartyFetches.length > 0 || !Object.values(r.witness).every((w) => w.pass);
if (failed) { console.error("release.sh: the WebGPU witness failed", JSON.stringify(r.witness)); process.exit(1); }
' "$web/dist/net/webgpu_witness.json"
(cd "$web" && npm pack --pack-destination "$web/dist")
sha256sum "$web"/dist/*.tgz
