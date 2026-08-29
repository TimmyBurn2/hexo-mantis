#!/usr/bin/env bash
# PERF-BASELINE pass 3 — a REPRESENTATIVE wire capture, the floor re-measured on it, and the seam.
set -uo pipefail
cd /workspace/hexo-mantis
OUT=/workspace/perfbase
PY=.venv/bin/python
export MANTIS_PERF_STAGES=1

echo "=== capture drive (n_workers=12, long window, strided single-graph capture) ==="
rm -rf "$OUT/wires_rep"
$PY tools/perf_baseline/drive.py --n-workers 12 --warmup-sec 20 --window-sec 330 \
    --label w12_capture --out "$OUT/w12_capture.json" --scratch "$OUT/cfg" \
    --capture-dir "$OUT/wires_rep" --capture-limit 700 --capture-stride 3 --capture-per-pop 2 \
    2>&1 | tail -4
echo "captured: $(ls "$OUT/wires_rep" | wc -l) single graphs, $(du -sh "$OUT/wires_rep" | cut -f1)"
sleep 8

echo "=== A2: silicon floor on the representative sample ==="
$PY tools/perf_baseline/silicon_floor.py --wire-dir "$OUT/wires_rep" \
    --out "$OUT/silicon_floor_rep.json" --graph-cap 700 2>&1 | tail -45
sleep 5

echo "=== C: seam, null-model server (uniform probs) ==="
$PY tools/perf_baseline/seam.py --out "$OUT/seam.json" --iters 120 2>&1 | tail -20

sleep 5

echo "=== E2: eval single-stream, DEEPER game (graph size grows with ply) ==="
$PY tools/perf_baseline/eval_stream.py --out "$OUT/eval_stream_deep.json" --max-moves 64 2>&1 | tail -6

sleep 5

echo "=== SYNC PROBE: is the forward wall its own work or a pipeline drain? ==="
$PY tools/perf_baseline/sync_probe.py --wire-dir "$OUT/wires_rep" \
    --out "$OUT/sync_probe.json" 2>&1 | tail -10
sleep 5

echo "=== D: trainer step ==="
$PY tools/perf_baseline/trainer_step.py --out "$OUT/trainer_step.json" \
    --fill-sec 240 --fill-workers 12 --steps 25 2>&1 | tail -40

echo "=== PASS 3 DONE ==="
