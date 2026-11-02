#!/usr/bin/env bash
# Release (design §10): the stamp exported on a clean tree into dist/net, the package built and its node tests run on
# the real files, the WebGPU witness on this desk, then `npm pack` with the licence and the third-party notices.
# The tarball and its sha256 go to a GitHub release; `private` keeps it off the npm registry.
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
ortdir="$(mktemp -d)"
notices=("$web/LICENSE" "$web/THIRD_PARTY_LICENSES.txt" "$web/RUST_LIBRARY_NOTICES.html")
trap 'rm -rf "$reference" "$ortdir" "${notices[@]}"' EXIT
cp "$root/LICENSE" "$web/LICENSE"
# MIT asks every copy to carry its notices: ONNX Runtime Web's with its runtime's third parties, Rust's standard library
# and each crate compiled into the wasm. ORT's npm package ships neither file, so they come from its tag, pinned here.
# The pinned toolchain's own notices cover the standard library and what it compiles in (hashbrown, dlmalloc, Unicode).
cp "$(cd "$root" && rustc --print sysroot)/share/doc/rust/COPYRIGHT-library.html" "$web/RUST_LIBRARY_NOTICES.html"
ort="$(node -p 'require(process.argv[1]).version' "$web/node_modules/onnxruntime-web/package.json")"
if [ "$ort" != 1.30.0 ]; then
    echo "release.sh: onnxruntime-web $ort's notices are unpinned: pin their sha256 here" >&2
    exit 2
fi
for f in LICENSE ThirdPartyNotices.txt; do
    curl -sSfL -o "$ortdir/$f" "https://raw.githubusercontent.com/microsoft/onnxruntime/v$ort/$f"
done
(cd "$ortdir" && sha256sum --check --quiet) <<'EOF'
2f07c72751aed99790b8a4869cf2311df85a860b22ded05fa22803587a48922c  LICENSE
143764b952fdb1a7c69ce653bfba74a7744d6a8a573bfb73e235fba356c83de3  ThirdPartyNotices.txt
EOF
mit="$(tail -n +5 "$root/LICENSE")"
{
    printf 'Where a component offers MIT among its licences, it is used here under MIT.\n\n'
    printf '== onnxruntime-web %s (MIT), then the third-party notices of ONNX Runtime %s\n\n' "$ort" "$ort"
    cat "$ortdir/LICENSE" "$ortdir/ThirdPartyNotices.txt"
    printf '\n== the Rust standard library and its own dependencies: RUST_LIBRARY_NOTICES.html, beside this file\n'
    (cd "$root" && cargo tree --locked -p mantis-player --target wasm32-unknown-unknown -e normal,no-proc-macro \
        --prefix none -f '{p}|{l}') | sed 's/ (\*)$//' | LC_ALL=C sort -u | grep -v '^mantis-' |
        while IFS='|' read -r pkg licence; do
            case "$licence" in
                "MIT" | "MIT OR Apache-2.0" | "Apache-2.0 OR MIT" | "Apache-2.0/MIT" | "Unlicense OR MIT" | \
                    "BSD-2-Clause OR Apache-2.0 OR MIT" | "(MIT OR Apache-2.0) AND Unicode-3.0") ;;
                *) echo "release.sh: $pkg is $licence, a licence this script never read: add it" >&2; exit 1 ;;
            esac
            dir="$(ls -d "${CARGO_HOME:-$HOME/.cargo}"/registry/src/*/"${pkg% v*}-${pkg##* v}" | head -1)"
            printf '\n== %s (%s)\n\n' "$pkg" "$licence"
            # Every licence file but Apache's, the alternative not taken; one of them must be the MIT text.
            mapfile -t texts < <(find "$dir" -maxdepth 1 -name 'LICENSE*' ! -name 'LICENSE-APACHE*' | LC_ALL=C sort)
            if [ "${#texts[@]}" -gt 0 ]; then
                grep -q "Permission is hereby granted" "${texts[@]}" ||
                    { echo "release.sh: $pkg's licence files hold no MIT text" >&2; exit 1; }
                awk 1 "${texts[@]}"
            else
                authors="$(sed -n 's/^authors = \[\(.*\)\]$/\1/p' "$dir/Cargo.toml" | tr -d '"')"
                [ -n "$authors" ] || { echo "release.sh: $pkg ships no licence file and names no authors" >&2; exit 1; }
                printf 'MIT License\n\nCopyright (c) %s\n\n%s\n' "$authors" "$mit"
            fi
        done
} > "$web/THIRD_PARTY_LICENSES.txt"
(cd "$root" && ORT_DISABLE_TELEMETRY=1 uv run --no-sync python tools/export_web.py --checkpoint "$stamp" \
    --out "$web/dist/net" --reference "$reference")
manifest="$(basename "$stamp" .ckpt).manifest.json"
node -e '
const m = require(process.argv[1]);
if (m.exporter.dirty !== false) { console.error("release.sh: the manifest says the export ran on a dirty tree"); process.exit(2); }
if (!m.files.every((f) => f.passes_its_line !== false || f.dtype === "fp16")) { console.error("release.sh: a shipped file misses its line"); process.exit(1); }
' "$web/dist/net/$manifest"
printf '{"manifest": "%s"}\n' "$manifest" > "$web/dist/net/index.json"
rm -f "$web/dist/net/"*.fp16.onnx  # fp16 ships only once a net passes its own fp16 line
(cd "$web" && MANTIS_WEB_FIXTURE="$web/dist" node --test test/*.test.ts)
# The witness record names this desk's browser and GPU: it stays beside the tarball, never in it.
(cd "$web" && node scripts/webgpu.mjs dist/net "$manifest" "$reference" "$web/dist/webgpu_witness.json")
node -e '
const r = require(process.argv[1]);
const failed = !r.ok || r.thirdPartyFetches.length > 0 || !Object.values(r.witness).every((w) => w.pass);
if (failed) { console.error("release.sh: the WebGPU witness failed", JSON.stringify(r.witness)); process.exit(1); }
' "$web/dist/webgpu_witness.json"
(cd "$web" && npm pack --pack-destination "$web/dist")
sha256sum "$web"/dist/*.tgz
