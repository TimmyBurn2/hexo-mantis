#!/usr/bin/env bash
# Usage: tools/rust_notices.sh PACKAGE TARGET — every compiled crate's notice, MIT where offered; both releases ship it.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
mit="$(tail -n +5 "$root/LICENSE")"
(cd "$root" && cargo tree --locked -p "$1" --target "$2" -e normal,no-proc-macro --prefix none -f '{p}|{l}') |
    sed 's/ (\*)$//' | LC_ALL=C sort -u | grep -v '^mantis-' |
    while IFS='|' read -r pkg licence; do
        case "$licence" in
            "MIT" | "MIT OR Apache-2.0" | "Apache-2.0 OR MIT" | "Apache-2.0/MIT" | "Unlicense OR MIT" | \
                "BSD-2-Clause OR Apache-2.0 OR MIT" | "(MIT OR Apache-2.0) AND Unicode-3.0") ;;
            *) echo "rust_notices.sh: $pkg is $licence, a licence this script never read: add it" >&2; exit 1 ;;
        esac
        dir="$(ls -d "${CARGO_HOME:-$HOME/.cargo}"/registry/src/*/"${pkg% v*}-${pkg##* v}" | head -1)"
        printf '\n== %s (%s)\n\n' "$pkg" "$licence"
        # Every licence file but Apache's, the alternative not taken; one of them must be the MIT text.
        mapfile -t texts < <(find "$dir" -maxdepth 1 -name 'LICENSE*' ! -name 'LICENSE-APACHE*' | LC_ALL=C sort)
        if [ "${#texts[@]}" -gt 0 ]; then
            grep -q "Permission is hereby granted" "${texts[@]}" ||
                { echo "rust_notices.sh: $pkg's licence files hold no MIT text" >&2; exit 1; }
            awk 1 "${texts[@]}"
        else
            authors="$(sed -n 's/^authors = \[\(.*\)\]$/\1/p' "$dir/Cargo.toml" | tr -d '"')"
            [ -n "$authors" ] || { echo "rust_notices.sh: $pkg ships no licence file and names no authors" >&2; exit 1; }
            printf 'MIT License\n\nCopyright (c) %s\n\n%s\n' "$authors" "$mit"
        fi
    done
