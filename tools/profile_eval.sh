#!/usr/bin/env bash
# Phase P1(b) — eval-at-deploy-sims profiling harness. PREP ONLY: committed so the box run is
# a one-liner; NOT executed here (rule 7: no host specifics; R170: box runs are operator-only).
#
# WHY A SEPARATE SCRIPT FROM THE SELF-PLAY ONE. The eval path is a different program with a
# different suspicion attached. STATE §2.5's sharpest on-record observation is eval-side:
# V-0 eval at deploy_sims=150 completed ZERO games in 15 minutes (~1.6 s/move plus
# cache-release cost), and the 20.12 s/round eval-wall figure is of UNKNOWN REGIME (R235 watch
# item, unreconciled). Those are eval numbers; profiling a self-play burst cannot explain
# them. The eval worker also runs OUT OF PROCESS under a spawn context with its own CUDA
# context (`eval/pipeline.py`'s isolation laws), so it must be sampled with --subprocesses or
# it is not sampled at all.
#
# DEPLOY-MATCHED, NOT CONVENIENT (LAW-15). The eval sims come from the MINTED config's own
# deploy decision. There is deliberately no --sims flag: LAW-15 makes deploy-matched eval the
# promotion bar and says a missing deploy decision blocks promotion rather than falling back
# to a proxy regime — a profile taken at a proxy regime measures a bar nothing promotes on.
# V-1's cache-release cost is IN this measurement by construction (it is on the deploy path,
# `arena/deploy_head.py`, landed 3be49d4); separating it is a later, pre-registered step.
#
# NO THRESHOLDS, NO VERDICTS. Collection only. Brackets and aborts are pre-registered by the
# operator BEFORE the numbers exist (tools/perf_prereg_skeleton.md).
#
# USAGE
#   tools/profile_eval.sh --config configs/run5.yaml --out-dir <run-stamped dir> \
#                        [--duration-sec N]
#
# OUTPUTS (under <out-dir>/)
#   pyspy_eval.svg          py-spy flamegraph across the eval worker subprocess tree
#   pyspy_eval.speedscope   py-spy speedscope profile
#   torch_eval/             torch.profiler trace (CUDA kernels + shapes), if torch is present
#   PROFILE_ENV.txt         provenance
set -euo pipefail

CONFIG=""; OUT_DIR=""; DURATION_SEC=900
while [ $# -gt 0 ]; do
  case "$1" in
    --config)       CONFIG="$2"; shift 2 ;;
    --out-dir)      OUT_DIR="$2"; shift 2 ;;
    --duration-sec) DURATION_SEC="$2"; shift 2 ;;
    *) echo "profile_eval: unknown argument $1" >&2; exit 2 ;;
  esac
done

[ -n "$CONFIG" ]  || { echo "profile_eval: --config is required (a MINTED config)" >&2; exit 2; }
[ -n "$OUT_DIR" ] || { echo "profile_eval: --out-dir is required (run-stamped)" >&2; exit 2; }
[ -f "$CONFIG" ]  || { echo "profile_eval: config not found: $CONFIG" >&2; exit 2; }

mkdir -p "$OUT_DIR"

# The eval path only runs if the config says so. Refuse loudly rather than produce an empty
# profile that reads as "eval is fast".
EVAL_ON=$(uv run python -c "
from mantis.config.loader import load_config
print(bool(load_config('$CONFIG').eval_enabled))
" 2>/dev/null || echo "unknown")
if [ "$EVAL_ON" != "True" ]; then
  echo "profile_eval: REFUSING — eval_enabled is '$EVAL_ON' in $CONFIG." >&2
  echo "              A run with eval off produces no eval work, and an empty eval profile" >&2
  echo "              would read as 'the eval path is cheap'. Use a config with eval on." >&2
  exit 3
fi

{
  echo "profile_kind=eval_at_deploy_sims"
  echo "config=$CONFIG"
  echo "config_sha256=$(sha256sum "$CONFIG" | cut -d' ' -f1)"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "git_dirty=$(test -n "$(git status --porcelain)" && echo yes || echo no)"
  echo "duration_sec=$DURATION_SEC"
  echo "eval_enabled=$EVAL_ON"
  # The deploy sims are the regime this profile CLAIMS. Recorded so a later reader cannot
  # mistake which regime produced the numbers — R235's 20.12 s/round figure is unreconciled
  # precisely because its regime was not recorded with it.
  echo "deploy_sims=$(uv run python -c "
from mantis.config.loader import load_config
c = load_config('$CONFIG')
print(getattr(getattr(c.eval, 'deploy', None), 'sims', 'UNRECORDED'))
" 2>/dev/null || echo UNRECORDED)"
  echo "py_spy=$(py-spy --version 2>/dev/null || echo absent)"
  echo "torch=$(uv run python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo absent)"
  echo "cuda_available=$(uv run python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo unknown)"
} > "$OUT_DIR/PROFILE_ENV.txt"
echo "profile_eval: provenance -> $OUT_DIR/PROFILE_ENV.txt"

RUN_OUT="$OUT_DIR/run"
mkdir -p "$RUN_OUT"
RUN_CMD=(uv run python -m mantis.run --config "$CONFIG" --out-dir "$RUN_OUT")

if command -v py-spy >/dev/null 2>&1; then
  # --subprocesses is NOT optional here: the eval worker is spawned as a separate process
  # (spawn context, own CUDA context). Without it this profiles the coordinator waiting.
  # --idle keeps the waiting frames, which on the eval path is where a 1.6 s/move is most
  # likely to be visible (queue wait, cache release, H2D).
  echo "profile_eval: py-spy record (${DURATION_SEC}s) ..."
  py-spy record --subprocesses --idle --nonblocking --rate 100 \
      --duration "$DURATION_SEC" --format flamegraph \
      --output "$OUT_DIR/pyspy_eval.svg" -- "${RUN_CMD[@]}" || \
      echo "profile_eval: py-spy flamegraph FAILED (recorded, not swallowed)"
  py-spy record --subprocesses --idle --nonblocking --rate 100 \
      --duration "$DURATION_SEC" --format speedscope \
      --output "$OUT_DIR/pyspy_eval.speedscope" -- "${RUN_CMD[@]}" || \
      echo "profile_eval: py-spy speedscope FAILED (recorded, not swallowed)"
else
  echo "profile_eval: py-spy ABSENT — no Python profile produced. MISSING MEASUREMENT." >&2
fi

echo "profile_eval: DONE. Artifacts in $OUT_DIR"
echo "profile_eval: NOTE — MEASUREMENTS ONLY. No verdict is asserted, and no gain bracket or"
echo "              abort threshold may be chosen after reading these numbers."
