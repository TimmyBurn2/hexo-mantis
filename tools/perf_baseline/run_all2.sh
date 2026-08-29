#!/usr/bin/env bash
# PERF-BASELINE — rebuild, then the seven-stage re-drives plus every remaining measurement.
set -uo pipefail
cd /workspace/hexo-mantis
OUT=/workspace/perfbase
PY=.venv/bin/python
export PATH="$HOME/.cargo/bin:$PATH"
export MANTIS_PERF_STAGES=1

echo "=== REBUILD ==="
uv sync 2>&1 | tail -3
bash /root/restore_cuda.sh 2>&1 | tail -2
uv pip install -q py-spy nvidia-ml-py 2>&1 | tail -1
$PY -c "import mantis._engine as e; print('stages:', list(e.selfplay_perf_snapshot()['stages'].keys()))"

echo "=== B7: seven-stage re-drives ==="
for nw in 1 12; do
  $PY tools/perf_baseline/drive.py --n-workers $nw --warmup-sec 30 --window-sec 120 \
      --label "w${nw}_s7" --out "$OUT/w${nw}_s7.json" --scratch "$OUT/cfg" 2>&1 | tail -3
  sleep 8
done

echo "=== A: silicon floor + kernel table ==="
$PY tools/perf_baseline/silicon_floor.py --wire-dir "$OUT/wires_w12" \
    --out "$OUT/silicon_floor.json" 2>&1 | tail -45
sleep 5

echo "=== C: seam, null-model server ==="
$PY tools/perf_baseline/seam.py --out "$OUT/seam.json" --iters 150 2>&1 | tail -20
sleep 5

echo "=== E: eval single-stream ==="
$PY tools/perf_baseline/eval_stream.py --out "$OUT/eval_stream.json" --max-moves 25 2>&1 | tail -8
sleep 5
$PY tools/perf_baseline/eval_stream.py --out "$OUT/eval_stream_sync.json" --max-moves 25 --sync-cuda 2>&1 | tail -8
sleep 5

echo "=== B: py-spy over a contended drive (py-spy is the PARENT) ==="
.venv/bin/py-spy record --rate 120 --threads --format raw \
    --output "$OUT/pyspy_w12.folded" -- \
    $PY tools/perf_baseline/drive.py --n-workers 12 \
    --warmup-sec 20 --window-sec 100 --label w12_pyspy \
    --out "$OUT/w12_pyspy.json" --scratch "$OUT/cfg" 2>&1 | tail -6
sleep 5

echo "=== D: trainer step ==="
$PY tools/perf_baseline/trainer_step.py --out "$OUT/trainer_step.json" \
    --fill-sec 240 --fill-workers 12 --steps 25 2>&1 | tail -40

echo "=== ALL DONE ==="
