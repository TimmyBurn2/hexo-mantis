#!/usr/bin/env bash
# PERF-BASELINE — the remaining measurements. Serial: they contend for one card.
set -uo pipefail
cd /workspace/hexo-mantis
OUT=/workspace/perfbase
PY=.venv/bin/python
export MANTIS_PERF_STAGES=1

echo "=== A: silicon floor + kernel table (real captured wires) ==="
$PY tools/perf_baseline/silicon_floor.py --wire-dir "$OUT/wires_w12" \
    --out "$OUT/silicon_floor.json" 2>&1 | tail -40
sleep 5

echo "=== C: seam, null-model server ==="
$PY tools/perf_baseline/seam.py --out "$OUT/seam.json" --iters 150 2>&1 | tail -20
sleep 5

echo "=== E: eval single-stream, uncontended ==="
$PY tools/perf_baseline/eval_stream.py --out "$OUT/eval_stream.json" --max-moves 25 2>&1 | tail -10
sleep 5
$PY tools/perf_baseline/eval_stream.py --out "$OUT/eval_stream_sync.json" --max-moves 25 --sync-cuda 2>&1 | tail -10
sleep 5

echo "=== B: py-spy over a contended drive ==="
MANTIS_PERF_STAGES=1 $PY tools/perf_baseline/drive.py --n-workers 12 \
    --warmup-sec 25 --window-sec 90 --label w12_pyspy \
    --out "$OUT/w12_pyspy.json" --scratch "$OUT/cfg" > "$OUT/w12_pyspy.log" 2>&1 &
DRIVE_PID=$!
sleep 40
.venv/bin/py-spy record --pid "$DRIVE_PID" --duration 60 --rate 120 --threads \
    --format raw --output "$OUT/pyspy_w12.folded" 2>&1 | tail -5
wait $DRIVE_PID
tail -3 "$OUT/w12_pyspy.log"
sleep 5

echo "=== D: trainer step ==="
$PY tools/perf_baseline/trainer_step.py --out "$OUT/trainer_step.json" \
    --fill-sec 240 --fill-workers 12 --steps 25 2>&1 | tail -35

echo "=== REST DONE ==="
