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

echo "=== PASS 3 DONE ==="
