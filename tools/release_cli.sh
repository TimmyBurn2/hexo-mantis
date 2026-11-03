#!/usr/bin/env bash
# Usage: tools/release_cli.sh NET_DIR OUT_DIR — mantis-cli built clean for this host, with NET_DIR's one manifest and weights.
set -euo pipefail
net="$(cd "$1" && pwd)"
out="$(mkdir -p "$2" && cd "$2" && pwd)"
root="$(cd "$(dirname "$0")/.." && pwd)"
if [ -n "$(git -C "$root" status --porcelain --untracked-files=no)" ]; then
    echo "release_cli.sh: the tree has tracked changes; a release builds from a clean commit" >&2
    exit 2
fi
host="$(rustc -vV | sed -n 's/^host: //p')"
case "$host" in
    x86_64-unknown-linux-gnu) platform=linux-x86_64 ;;
    *) echo "release_cli.sh: no bundle is defined for $host" >&2; exit 2 ;;
esac
manifests=("$net"/*.manifest.json)
if [ "${#manifests[@]}" -ne 1 ] || [ ! -f "${manifests[0]}" ]; then
    echo "release_cli.sh: $net holds ${#manifests[@]} manifests, not one" >&2
    exit 2
fi
stem="$(basename "${manifests[0]}" .manifest.json)"
version="$(cd "$root" && cargo pkgid -p mantis-player | sed 's/.*[#@]//')"
name="mantis-cli-$version-$platform"
(cd "$root" && cargo build --release --locked -p mantis-player --bin mantis-cli)
stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
mkdir "$stage/$name"
cp "$root/target/release/mantis-cli" "${manifests[0]}" "$net/$stem.weights.bin" "$root/LICENSE" "$stage/$name/"
# MIT asks every copy to carry its notices: the Rust standard library's (the pinned toolchain's own file) and each crate's.
cp "$(cd "$root" && rustc --print sysroot)/share/doc/rust/COPYRIGHT-library.html" "$stage/$name/RUST_LIBRARY_NOTICES.html"
{
    printf 'Where a component offers MIT among its licences, it is used here under MIT.\n\n'
    printf '== the Rust standard library and its own dependencies: RUST_LIBRARY_NOTICES.html, beside this file\n'
    "$root/tools/rust_notices.sh" mantis-player "$host"
} > "$stage/$name/THIRD_PARTY_LICENSES.txt"
# The bundle as unpacked: the binary finds the manifest beside it, checks the weights by sha256 and answers.
if ! printf 'six\nquit\n' | "$stage/$name/mantis-cli" --threads 1 | grep -qx sixok; then
    echo "release_cli.sh: the unpacked bundle did not start" >&2
    exit 1
fi
tar --sort=name --owner=0 --group=0 --numeric-owner --mtime="@$(git -C "$root" log -1 --format=%ct)" \
    -C "$stage" -czf "$out/$name.tar.gz" "$name"
(cd "$out" && sha256sum "$name.tar.gz" | tee "$name.tar.gz.sha256")
