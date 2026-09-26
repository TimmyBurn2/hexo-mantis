#!/usr/bin/env bash
# THE LOCAL GATE SET: every gate with CI's arguments; remote CI is suspended, so this is the gate.
# Each row calls the gate's own script or make target (no second authority); nothing short-circuits.
# Gate 1 (minutes) and the slow tier (deselected by BOTH pytest tiers) are opt-in; omission prints every run.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 2

UV=${UV:-uv}
WITH_FRESH_SYNC=0
WITH_SLOW=0
BASE_REF=${GATE_BASE_REF:-origin/dev}
ONLY=""

usage() {
    cat <<'USAGE'
usage: tools/ci_gates/run_all.sh [--with-fresh-sync] [--with-slow] [--base REF]
                                 [--only SUBSTRING]

  --with-fresh-sync  also run gate 1 (fresh-clone uv sync; minutes, not seconds)
  --with-slow        also run the `slow` tier, which BOTH pytest tiers deselect and which
                     therefore no gate otherwise executes (R333(b)). Run it at every
                     packet exit.
  --base REF         the diff base gates 6 and 17 measure against (default origin/dev,
                     or $GATE_BASE_REF)
  --only SUBSTRING   run only the gates whose label contains SUBSTRING
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --with-fresh-sync) WITH_FRESH_SYNC=1; shift ;;
        --with-slow) WITH_SLOW=1; shift ;;
        --base) BASE_REF=$2; shift 2 ;;
        --only) ONLY=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "run_all: unknown argument $1" >&2; usage >&2; exit 2 ;;
    esac
done

# THE GATES NEVER RE-SYNC THE VENV: a bare `uv run` swaps a box's CUDA torch for the CPU wheel
# mid-set. Syncing is the caller's step (`make build` or `make build.cuda`).
export UV_NO_SYNC=1
printf 'run_all: venv gated AS BUILT (UV_NO_SYNC=1): torch %s\n' \
    "$($UV run python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo absent)"

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

# ── rust arm: runs BESIDE the python arm, its log printed whole once both finish ──────────
RUST_ARM=$(mktemp -d) || exit 2
trap 'rm -rf "$RUST_ARM"' EXIT
# Job control gives the arm its own process group, so a signal can take down cargo and its tests.
set -m
(
    PASSED=()
    FAILED=()
    [ $WITH_FRESH_SYNC -eq 1 ] && \
        run_gate "gate 1: fresh-clone uv sync builds the extension" \
            bash tools/ci_gates/gate_01_fresh_sync.sh

    run_gate "gate 2a: cargo test workspace" \
        cargo test --workspace --locked
    # `--all-targets` is load-bearing: nothing else local compiles the non-smoke bench targets
    # that stand behind tools/bench_floors.toml's floors.
    run_gate "gate 2b: clippy (-D clippy::all, --all-targets)" \
        cargo clippy --workspace --all-targets --locked -- -D clippy::all
    run_gate "gate 4: wasm check (mantis-graph dep-free)" \
        make check.wasm
    run_gate "gate 5: bench smoke (stub criterion bench)" \
        make bench
    printf '%s\n' "${PASSED[@]}" > "$RUST_ARM/passed"
    printf '%s\n' "${FAILED[@]}" > "$RUST_ARM/failed"
    : > "$RUST_ARM/done"
) > "$RUST_ARM/log" 2>&1 &
RUST_PID=$!
set +m
trap 'trap - INT TERM; kill -TERM -- -"$RUST_PID" 2>/dev/null; exit 130' INT TERM

# ── python ────────────────────────────────────────────────────────────────────────────
run_gate "gate 3a: pytest default tier" \
    $UV run pytest -m "not integration and not slow" -n 8
run_gate "gate 3b: pytest integration tier" \
    $UV run pytest -m integration
# Not numbered: it is not one of the numbered gates. It is the tier those gates do
# not reach, and it says so in the summary whether or not it ran.
[ $WITH_SLOW -eq 1 ] && \
    run_gate "slow tier (opt-in, --with-slow): the tests BOTH tiers deselect" \
        $UV run pytest -m slow
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

wait "$RUST_PID"
RUST_RC=$?
trap - INT TERM
if [ -s "$RUST_ARM/log" ]; then
    printf '\n\033[1m══ rust arm (ran beside the python arm) ══\033[0m\n'
    cat "$RUST_ARM/log"
fi
if [ -e "$RUST_ARM/done" ]; then
    while IFS= read -r row; do [ -n "$row" ] && PASSED+=("$row"); done < "$RUST_ARM/passed"
    while IFS= read -r row; do [ -n "$row" ] && FAILED+=("$row"); done < "$RUST_ARM/failed"
else
    FAILED+=("rust arm (exit $RUST_RC before its gates finished)")
fi

# ── hygiene (diff-scoped) ─────────────────────────────────────────────────────────────
run_gate "gate 6: artifact rejection" \
    python3 tools/ci_gates/artifact_gate.py --base "$BASE_REF"
run_gate "gate 10: no Makefile/doc reference to untracked paths" \
    python3 tools/ci_gates/check_tracked_refs.py
run_gate "gate 17: no host content in the tree (rule 7)" \
    python3 tools/ci_gates/rule7_gate.py --base "$BASE_REF"
run_gate "gate 18: rustfmt on the Rust files touched since base" \
    python3 tools/ci_gates/rustfmt_touched_gate.py --base "$BASE_REF"
run_gate "gate 19: every commit since base is one line, no body, no trailer" \
    python3 tools/ci_gates/commit_convention_gate.py --base "$BASE_REF"

# ── the screen ────────────────────────────────────────────────────────────────────────
printf '\n\033[1m══ LOCAL GATE SET ══\033[0m\n'
printf '   base ref for the diff-scoped gates: %s\n' "$BASE_REF"
if [ $WITH_FRESH_SYNC -eq 0 ]; then
    printf '   gate 1 NOT RUN (pass --with-fresh-sync). Its fresh-clone uv sync is the one\n'
    printf '   check no local run reproduces; CLAUDE.md records that as an accepted cost.\n'
fi
if [ $WITH_SLOW -eq 0 ]; then
    printf '   SLOW TIER NOT RUN (pass --with-slow). Both pytest tiers DESELECT it, so these\n'
    printf '   tests were executed by nothing in this run. R333(b): run it at every packet exit.\n'
else
    printf '   slow tier RAN (--with-slow) — the tier both pytest tiers deselect.\n'
fi
printf '   green: %s\n' "${#PASSED[@]}"
printf '   wall: %ss\n' "$SECONDS"
if [ ${#FAILED[@]} -eq 0 ]; then
    printf '   \033[32mALL GREEN\033[0m\n'
    exit 0
fi
printf '   \033[31mRED: %s\033[0m\n' "${#FAILED[@]}"
for entry in "${FAILED[@]}"; do
    printf '     - %s\n' "$entry"
done
exit 1
