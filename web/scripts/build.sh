#!/usr/bin/env bash
# The engine's wasm module and its bindgen glue into web/pkg/, then the package's JS into web/dist/: simd128 is a
# portable target feature every evergreen browser has (the engine refuses where it is missing), never a target-cpu.
set -euo pipefail
web="$(cd "$(dirname "$0")/.." && pwd)"
root="$(cd "$web/.." && pwd)"
want="$(awk '/^name = "wasm-bindgen"$/ { getline; gsub(/"/, "", $3); print $3 }' "$root/Cargo.lock")"
bindgen="$(cd "$root" && mise which wasm-bindgen 2>/dev/null || command -v wasm-bindgen || true)"
have="$([ -n "$bindgen" ] && "$bindgen" --version | awk '{ print $2 }' || true)"
if [ "$have" != "$want" ]; then
    echo "build.sh: the wasm-bindgen CLI is '${have:-absent}', the lock's crate is $want: run 'mise install' (mise.toml pins it)" >&2
    exit 2
fi
(cd "$root" && RUSTFLAGS="-C target-feature=+simd128" cargo rustc -p mantis-player --target wasm32-unknown-unknown \
    --release --locked --crate-type cdylib)
rm -rf "$web/pkg"
"$bindgen" --target web --out-dir "$web/pkg" "$root/target/wasm32-unknown-unknown/release/mantis_player.wasm"
[ -d "$web/node_modules" ] || exit 0  # the engine alone: gate 4's check and a fresh clone before `npm ci`
cd "$web"
rm -rf dist
npx tsc -p tsconfig.build.json
cp -r pkg dist/pkg
cp demo/index.html dist/demo/
mkdir -p dist/ort
# The WebGPU bundle and the one runtime it loads (its glue and its wasm), from the integrator's own origin (`env.wasm.wasmPaths`).
cp node_modules/onnxruntime-web/dist/ort.webgpu.bundle.min.mjs node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.{mjs,wasm} dist/ort/
