#!/usr/bin/env bash
# THE LOCAL GATE SET. Every gate CLAUDE.md lists, with the arguments CI passes, in one place.
#
# WHY THIS EXISTS (AUDIT-1 F-09). R311(b) suspended remote CI and made LOCAL GREEN the gate.
# But there was no local runner: `make test` runs the default tier plus `cargo test`, which
# does NOT compile `[[bench]]` targets, and `make bench` compiles exactly one of the eight.
# `cargo clippy --workspace --all-targets --locked -- -D clippy::all` existed ONLY in
# `.github/workflows/ci.yml`. So every "full local gate set" since the suspension excluded
# `-D clippy::all` — including `incompatible_msrv`, the guard on the 1.87 floor — and never
# compiled the seven bench targets standing behind `tools/bench_floors.toml`'s 28 floors.
# CLAUDE.md's own rule is "nothing lives only in workflow YAML".
#
# THE WITNESS, measured 2026-09-03 with `clippy::len_zero` planted in
# `crates/mantis-selfplay/benches/queue_fuse_bench.rs`:
#     cargo clippy --workspace --locked -- -D clippy::all   rc 0   GREEN
#     make bench                                            rc 0   GREEN
#     cargo test --workspace --locked --no-run              rc 0   GREEN, and its target list
#                                                                  names NO bench binary
#     make lint.rust  (clippy --all-targets)                rc 2   RED, naming clippy::len_zero
#                                                                  in bench "queue_fuse_bench"
# So `--all-targets` is the ONLY thing in the repo that compiles seven of the eight bench
# targets, and before this file nothing local passed it.
#
# WHAT IT IS NOT. It is not a second authority over what a gate CHECKS: every row below shells
# out to the same script or make target CI invokes, with the same arguments. A gate's logic
# lives in its own file; this file only says WHICH gates there are and RUNS them all.
#
# ORDER. Cheap and structural first, so a typo reds in seconds rather than after the tier.
# NOTHING short-circuits: every gate runs even after one reds, because a run that stops at the
# first failure tells you about one gate when you wanted to know about seventeen.
#
# GATE 1 IS DELIBERATELY NOT HERE by default. Its fresh-clone `uv sync` is the one check no
# local run reproduces cheaply (CLAUDE.md records the accepted cost), and it takes minutes.
# `--with-fresh-sync` opts into it.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 2

UV=${UV:-uv}
WITH_FRESH_SYNC=0
BASE_REF=${GATE_BASE_REF:-origin/dev}
ONLY=""

usage() {
    cat <<'USAGE'
usage: tools/ci_gates/run_all.sh [--with-fresh-sync] [--base REF] [--only SUBSTRING]

  --with-fresh-sync  also run gate 1 (fresh-clone uv sync; minutes, not seconds)
  --base REF         the diff base gates 6 and 17 measure against (default origin/dev,
                     or $GATE_BASE_REF)
  --only SUBSTRING   run only the gates whose label contains SUBSTRING
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --with-fresh-sync) WITH_FRESH_SYNC=1; shift ;;
        --base) BASE_REF=$2; shift 2 ;;
        --only) ONLY=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "run_all: unknown argument $1" >&2; usage >&2; exit 2 ;;
    esac
done

PASSED=()
FAILED=()

run_gate() {
    local label=$1; shift
    if [ -n "$ONLY" ] && [[ "$label" != *"$ONLY"* ]]; then
        return 0
    fi
    printf '\n\033[1m── %s ──\033[0m\n' "$label"
    printf '   $ %s\n' "$*"
    local start rc
    start=$SECONDS
    "$@"
    rc=$?
    if [ $rc -eq 0 ]; then
        printf '   \033[32mGREEN\033[0m  (%ss)\n' "$((SECONDS - start))"
        PASSED+=("$label")
    else
        printf '   \033[31mRED rc=%s\033[0m  (%ss)\n' "$rc" "$((SECONDS - start))"
        FAILED+=("$label (rc $rc)")
    fi
}

# ── rust ──────────────────────────────────────────────────────────────────────────────
[ $WITH_FRESH_SYNC -eq 1 ] && \
    run_gate "gate 1: fresh-clone uv sync builds the extension" \
        bash tools/ci_gates/gate_01_fresh_sync.sh

run_gate "gate 2a: cargo test workspace" \
    cargo test --workspace --locked
# `--all-targets` is the load-bearing flag: without it the seven non-smoke bench targets are
# never compiled by ANY local command, and the 28 floors in tools/bench_floors.toml stand
# behind code nothing builds.
run_gate "gate 2b: clippy (-D clippy::all, --all-targets)" \
    cargo clippy --workspace --all-targets --locked -- -D clippy::all
run_gate "gate 4: wasm check (mantis-graph dep-free)" \
    make check.wasm
run_gate "gate 5: bench smoke (stub criterion bench)" \
    make bench

# ── python ────────────────────────────────────────────────────────────────────────────
run_gate "gate 3a: pytest default tier" \
    $UV run pytest -m "not integration and not slow"
run_gate "gate 3b: pytest integration tier" \
    $UV run pytest -m integration
run_gate "gate 3c: collected-test count non-decreasing" \
    bash tools/ci_gates/test_count_gate.sh
run_gate "gate 7: every configs/ file schema-validates" \
    $UV run python tools/ci_gates/validate_configs.py
run_gate "gate 8: registry sha handshake + audit" \
    bash tools/ci_gates/registry_gate.sh
run_gate "gate 9: import-DAG check" \
    $UV run python tools/check_import_dag.py src/mantis
run_gate "gate 11: no silent encoding-fallback arms" \
    $UV run python tools/ci_gates/silent_encoding_gate.py
run_gate "gate 12: armed-abort manifest audit" \
    $UV run python tools/ci_gates/preflight_mint.py --audit-only
run_gate "gate 13: contract-doc drift (run config schema)" \
    $UV run python tools/ci_gates/contract_doc_gate.py
run_gate "gate 14: curated lint/type gate (R98)" \
    bash tools/ci_gates/lint_gate.sh --self-test
run_gate "gate 15: R8 justification headers" \
    $UV run python tools/ci_gates/r8_header_gate.py
run_gate "gate 16: no encoding-less text I/O" \
    $UV run python tools/ci_gates/encoding_io_gate.py

# ── hygiene (diff-scoped) ─────────────────────────────────────────────────────────────
run_gate "gate 6: artifact rejection" \
    python3 tools/ci_gates/artifact_gate.py --base "$BASE_REF"
run_gate "gate 10: no Makefile/doc reference to untracked paths" \
    python3 tools/ci_gates/check_tracked_refs.py
run_gate "gate 17: no host content in the tree (rule 7)" \
    python3 tools/ci_gates/rule7_gate.py --base "$BASE_REF"

# ── the screen ────────────────────────────────────────────────────────────────────────
printf '\n\033[1m══ LOCAL GATE SET ══\033[0m\n'
printf '   base ref for the diff-scoped gates: %s\n' "$BASE_REF"
if [ $WITH_FRESH_SYNC -eq 0 ]; then
    printf '   gate 1 NOT RUN (pass --with-fresh-sync). Its fresh-clone uv sync is the one\n'
    printf '   check no local run reproduces; CLAUDE.md records that as an accepted cost.\n'
fi
printf '   green: %s\n' "${#PASSED[@]}"
if [ ${#FAILED[@]} -eq 0 ]; then
    printf '   \033[32mALL GREEN\033[0m\n'
    exit 0
fi
printf '   \033[31mRED: %s\033[0m\n' "${#FAILED[@]}"
for entry in "${FAILED[@]}"; do
    printf '     - %s\n' "$entry"
done
exit 1
