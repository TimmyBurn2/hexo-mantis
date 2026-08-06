#!/usr/bin/env bash
# Phase P1(a) — self-play burst profiling harness. PREP ONLY: this script is committed so the
# box run is a one-liner, but NOTHING here has been executed against a box (rule 7: no host
# specifics; R170: the box run is operator-only).
#
# WHY THIS EXISTS. LAW-09 is explicit that a performance change starts with a profile, not an
# impression: "Profile first (flamegraph / py-spy; DHAT for allocation-rate hunting);
# profiling builds = release + debug symbols (`profiling` profile)." The headline claim under
# investigation — "the GNN implementation is extremely slow" — is ASSERTED, and STATE §2.5's
# on-record observations (box tier-1 games_per_hour ~19; a CPU twin too slow to finish a game
# at 50 sims in 5 min) are consistent with it but do not localise it. This produces the
# localisation.
#
# PRODUCTION PATH, PRODUCTION PARAMETERS (R155/R160). The config is passed in and is expected
# to be a MINTED config — the same file a real run would use. There is deliberately no
# --sims / --workers / --device override: a profile taken at parameters no run uses measures
# a program no run executes, and that is how a preflight once false-cleared a 16 GiB GPU wall
# (CARD-RUN5-GPU-OOM). If the shape must change, change it in a minted config and say so in
# the prereg.
#
# NO THRESHOLDS, NO VERDICTS. This script collects; it decides nothing. The gain brackets and
# abort thresholds are the operator's, pre-registered BEFORE the numbers land
# (plan/RUN5_MINT_PREREG.md; the skeleton is tools/perf_prereg_skeleton.md). Reading a
# threshold off a profile you have already seen is a post-hoc threshold.
#
# USAGE
#   tools/profile_selfplay.sh --config configs/run5.yaml --out-dir <run-stamped dir> \
#                            [--duration-sec N] [--rust]
#
# OUTPUTS (all under <out-dir>/, which should be run-stamped by the caller)
#   pyspy_selfplay.svg        py-spy sampling flamegraph, whole process tree
#   pyspy_selfplay.speedscope py-spy speedscope profile (better for reading GIL-held frames)
#   torch_selfplay/           torch.profiler trace dir (chrome trace + stacks), if enabled
#   flamegraph_selfplay.svg   cargo-flamegraph over the Rust engine, if --rust
#   PROFILE_ENV.txt           what was measured: shas, config identity, tool versions
set -euo pipefail

CONFIG=""; OUT_DIR=""; DURATION_SEC=300; DO_RUST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --config)       CONFIG="$2"; shift 2 ;;
    --out-dir)      OUT_DIR="$2"; shift 2 ;;
    --duration-sec) DURATION_SEC="$2"; shift 2 ;;
    --rust)         DO_RUST=1; shift ;;
    *) echo "profile_selfplay: unknown argument $1" >&2; exit 2 ;;
  esac
done

# R1 posture at the boundary: both inputs required, neither defaulted. A defaulted --out-dir
# is how two profiles end up overwriting each other in one directory.
[ -n "$CONFIG" ]  || { echo "profile_selfplay: --config is required (a MINTED config)" >&2; exit 2; }
[ -n "$OUT_DIR" ] || { echo "profile_selfplay: --out-dir is required (run-stamped)" >&2; exit 2; }
[ -f "$CONFIG" ]  || { echo "profile_selfplay: config not found: $CONFIG" >&2; exit 2; }

mkdir -p "$OUT_DIR"

# ── provenance first, so a profile can never be read without knowing what produced it ──────
# LAW-09's triangulation rule and R98 both depend on this: a flamegraph with no attached
# commit/config identity cannot be compared to anything later.
{
  echo "profile_kind=selfplay_burst"
  echo "config=$CONFIG"
  echo "config_sha256=$(sha256sum "$CONFIG" | cut -d' ' -f1)"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "git_dirty=$(test -n "$(git status --porcelain)" && echo yes || echo no)"
  echo "duration_sec=$DURATION_SEC"
  echo "rustc=$(rustc --version 2>/dev/null || echo absent)"
  echo "py_spy=$(py-spy --version 2>/dev/null || echo absent)"
  echo "python=$(uv run python --version 2>/dev/null || echo absent)"
  echo "torch=$(uv run python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo absent)"
  echo "cuda_available=$(uv run python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo unknown)"
} > "$OUT_DIR/PROFILE_ENV.txt"
echo "profile_selfplay: provenance -> $OUT_DIR/PROFILE_ENV.txt"

# ── the run under profile ──────────────────────────────────────────────────────────────────
# `python -m mantis.run` is THE entry point (CLAUDE.md); nothing here constructs a bespoke
# harness, because a harness is a second composition authority and would profile a program
# the run does not execute.
RUN_OUT="$OUT_DIR/run"
mkdir -p "$RUN_OUT"
RUN_CMD=(uv run python -m mantis.run --config "$CONFIG" --out-dir "$RUN_OUT")

if command -v py-spy >/dev/null 2>&1; then
  # --subprocesses: self-play workers and the eval worker are separate processes; profiling
  #   only the parent measures the coordinator and misses the thing under suspicion.
  # --idle: GIL-blocked and IO-blocked frames are KEPT. For a "too slow" investigation the
  #   waiting frames are the finding as often as the running ones — marshal+queue wait is a
  #   pre-registered hotspot precisely because it is invisible without this.
  # --nonblocking: do not stop the target to sample; a stopped target changes the timing.
  echo "profile_selfplay: py-spy record (${DURATION_SEC}s) ..."
  py-spy record --subprocesses --idle --nonblocking --rate 100 \
      --duration "$DURATION_SEC" --format flamegraph \
      --output "$OUT_DIR/pyspy_selfplay.svg" -- "${RUN_CMD[@]}" || \
      echo "profile_selfplay: py-spy flamegraph FAILED (recorded, not swallowed)"
  py-spy record --subprocesses --idle --nonblocking --rate 100 \
      --duration "$DURATION_SEC" --format speedscope \
      --output "$OUT_DIR/pyspy_selfplay.speedscope" -- "${RUN_CMD[@]}" || \
      echo "profile_selfplay: py-spy speedscope FAILED (recorded, not swallowed)"
else
  echo "profile_selfplay: py-spy ABSENT — no Python profile produced. This is a MISSING" >&2
  echo "                  MEASUREMENT, not a passing run; install py-spy and re-run." >&2
fi

# ── Rust side (optional): where the per-leaf graph rebuild and the segment ops live ────────
if [ "$DO_RUST" = "1" ]; then
  if command -v cargo-flamegraph >/dev/null 2>&1 || cargo flamegraph --version >/dev/null 2>&1; then
    # The `profiling` profile is release + debug symbols (LAW-09's own words). A release
    # build without symbols yields a flamegraph of hex addresses.
    echo "profile_selfplay: cargo flamegraph over the selfplay bench ..."
    CARGO_PROFILE=profiling cargo flamegraph --profile profiling \
        -o "$OUT_DIR/flamegraph_selfplay.svg" \
        -p mantis-selfplay --bench graph_build_bench -- --bench || \
        echo "profile_selfplay: cargo-flamegraph FAILED (recorded, not swallowed)"
  else
    echo "profile_selfplay: cargo-flamegraph ABSENT — no Rust profile produced." >&2
  fi
fi

echo "profile_selfplay: DONE. Artifacts in $OUT_DIR"
echo "profile_selfplay: NOTE — this script produces MEASUREMENTS ONLY. It asserts no verdict,"
echo "                  and no gain bracket or abort threshold may be chosen after reading it."
