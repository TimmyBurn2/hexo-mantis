#!/usr/bin/env bash
# PERF-BASELINE §2 B/F — the four self-play windows. Serial: they contend for one card.
set -uo pipefail
cd /workspace/hexo-mantis
OUT=/workspace/perfbase
mkdir -p "$OUT"
export MANTIS_PERF_STAGES=1
PY=.venv/bin/python

run() {  # label n_workers extra...
  local label=$1 nw=$2; shift 2
  echo "=== $label ==="
  $PY tools/perf_baseline/drive.py --n-workers "$nw" \
      --warmup-sec 30 --window-sec 120 --label "$label" \
      --out "$OUT/$label.json" --scratch "$OUT/cfg" "$@" 2>&1 | tail -6
  sleep 10
}

run w1_nosync   1  --capture-dir "$OUT/wires_w1"  --capture-limit 400
run w12_nosync 12  --capture-dir "$OUT/wires_w12" --capture-limit 400
run w1_sync     1  --sync-cuda
run w12_sync   12  --sync-cuda
echo "=== ALL DRIVES DONE ==="
