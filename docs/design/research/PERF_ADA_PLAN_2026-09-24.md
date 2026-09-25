# PERF-ADA — making the 4080S box the run box (plan, 2026-09-24)
References re-verified against dev fc37f3f2 (post-SLIM merge) 2026-09-25; measurements unchanged (taken on 51eddda5).

Living plan. Evidence is filed beside it; this file says what has to happen, in what order, and who decides.

## 0. Where this starts

- **Operator decision (2026-09-24):** the run box is the 4080S box (RTX 4080 SUPER 16 GB, Ryzen 9 9950X
  16C/32T, 60 GB disk, no persistent volume). The 5080 + 3900X candidate that passed admission (2 364 leaves/s,
  0.96×) was declined on cost. Development cost is accepted; the work rides the pruning programme.
- **The measured gap:** `tools/bench_server.py` alone at B 64 reads 1 816 leaves/s against the reference
  box's 2 466 (0.74×; R367(e) admits at ≥ 0.85×). Not the tree (the reference tree reads 1 847 here), not PCIe
  (≈ 5 %), not the serving knobs (B 16…256 and 32/48 workers × 8/16 leaves all ≤ B 64's reading).
- **The cause:** the GPU. bf16 `index_add_` (the GINE message aggregation) is 55 % of GPU time; on Ada the bf16
  atomic is emulated, on Blackwell it is native (microbench 6.25 ms vs 1.40 ms per 1M×128 aggregation).
  fp32 accumulation reads 2.43 ms here, a dst-sorted segment sum 2.11 ms, and both are more accurate than the
  bf16 path (max error 0.375 vs fp32).
- **Not a cost:** run10's step-0 net (45k + fresh aux head) serves as run8 does (1 802); the aux head is not
  evaluated in serving.

## 0a. Post-merge status (2026-09-25, dev fc37f3f2)

- **The findings hold on the merged tree.** SLIM-FIX touched none of the code they rest on: `gine.py`,
  `gnn_v2.py`, `graph_wire_split.py`, the bridge's `inference.rs` and `configs/run10.yaml` are unchanged since
  9839becd; `inference_server.py`, `graph_collate.py`, `bench_server.py` and the Rust builder changed in comments
  only. The sync is still `gnn_v2.py:118` (called at :183), the bf16 aggregation `gine.py:64`, the two timers
  `inference_server.py:714-716` / `:832-835`, the blind probe `bench_server.py:117-118`, the dummy edges
  `crates/mantis-graph/src/lib.rs:740-748`. The box ran 51eddda5, which equals 9839becd under src/, crates/,
  tools/ and configs/. All four docs carry a re-verification line; ~55 citations were moved, 0 findings are stale.
- **`configs/run8.yaml` is gone** (deleted at 8b00b4dd). The box's cells read it; any future re-read of the
  admission cell uses `configs/run10.yaml` + the 45k checkpoint — its `inference` section is identical to what
  the box read and `selfplay` differs only in `mcts.dirichlet_enabled` (true → false), which `bench_server` never
  reaches (it serves stored positions, no search); the bench rebuilds the net from the checkpoint's stamp.
- **R368(i) binds this plan.** Until run10 STARTs, no leg may change run10's resolved config, the stamp or
  checkpoint format, or trainer/search/eval numerics. The operator's direction is that run10 WAITS for the perf
  work (2026-09-25). The two cannot both hold for leg 3: see §2 ask 5 and the leg table's R368(i) column.
- **STATE is stale on one line:** it says no box is rented; the operator rented the 4080S box on 2026-09-24
  (the box checkout sits on 51eddda5, the pre-merge slim tip; run10's launch tip is R368(i)'s to name).

## 1. Evidence

| Report | What it settles |
|---|---|
| `research/PERF_ADA_CODEBASE_2026-09-24.md` | every serving/training lever in the code, file:line, priced on this host; the outside-the-box set; the profiling hypotheses |
| `research/PERF_ADA_ONLINE_2026-09-24.md` | primary sources: bf16 atomics by compute capability, PyTorch's index_add path, scatter alternatives and determinism, overlap/CUDA graphs, Zen 5 / AVX-512, profiling in an unprivileged container |
| `measurements/PERF_ADA_PROFILE_2026-09-24.md` | mixed Python+Rust flamegraphs (serving, real self-play burst), trainer step profile, the fp32/segment A/B in the real server with numerics, the native-build A/B, cheap outside-the-box probes |

Raw artifacts (flamegraphs, speedscope, logs) live in the operator's mirror, never in the tree (R7).

### 1b. What the evidence says (all three reports in; the profile's numbers are MEASURED on the box)

- **A hidden per-forward sync is the first bound.** `real[legal_index] = True` (`gnn_v2.py:118`) is a pageable
  H2D that blocks the server thread until the GPU drains: 78 % of the server thread in the bench, 67 % in the
  real loop. `index_fill_` builds the same mask without it: launch 32.4 → 8.9 ms, +12 %, bit-identical. It is
  the mechanism behind F-47's "the halves never overlap" on this host.
- **The aggregation is the second.** `index_put_(accumulate=True)` (PyTorch's sort-based path: fp32 sum, one
  rounding) is deterministic, +56 % alone and **+87 % with the sync fix** (1 808 → 3 381), and speeds the
  trainer step 10 % at the same peak memory. A plain fp32 buffer is as fast in serving but OOMs the trainer at
  its 4.5 M-edge microbatch — so it cannot be the one implementation.
- **The real self-play loop** (sync fix + fp32 sum, no trainer): 2 010 → 3 691 leaves/s, 696 → 1 320 games/h,
  still GPU-bound (99.9 %). The reference box's whole run8 in-run rate was 3 208.
- **The current bf16 path is not repeatable** (the same batch forwarded three times differs by up to 0.11 in
  value), and `bench_server`'s determinism probe cannot see it: it serves one opening position four times, so
  it reads a constant for every kernel. The probe must be replaced before any lever is judged by it.
- **35.8 % of served leaves are exact repeats** in steady state — an exact per-net eval cache is the largest
  lever after the two above (≤ ×1.5 turns/s, INFERRED upper bound).
- **Measured null:** `target-cpu=native`, CPU inference on the spare cores, the fixed-slot neighbour table
  (slower than the fused fp32 path), torch intra-op threads (0 leaves/s, but frees 2.6–5.2 cores), a
  copy stream or second server stream (≤ 3–8 % left once the GPU is 92–99.9 % busy, INFERRED), CARD-PERF-4's
  Rust collate (0 today: CPU stage 8.5–9 ms against a 13 ms GPU stage, INFERRED).
- **Corrections to earlier readings:** `launch` and `gpu_wait` are timed on different threads, so
  "launch − gpu_wait" was never the CPU stage; PERF-3's "the CPU launch stage is the bound" was read off the
  same timer while the same sync existed, so CARD-PERF-4's premise needs re-reading before it is designed.
- **Not measured:** Rust-frame profiles (py-spy `--native` starved the target), the trainer inside the loop
  (estimated ~28 % of the GPU at 2.4 steps/game — the twin reads it). cu130 / newer torch: measured, §1a.

### 1a. The CUDA wheel (operator question, 2026-09-24)

The pin is `torch==2.11.0` from the cu128 index, and cu128 ends at 2.11 (checked against the PyTorch wheel
index 2026-09-24: cu128 2.10–2.11; cu129 to 2.13; cu130 2.10–2.14; cu132 2.13–2.14). The box's driver
(580-series, CUDA ≤ 13.0) runs cu130 and not cu132 (forward compatibility is datacenter-only). So
`2.11.0+cu130` is available with the torch pin unchanged — a one-line index swap plus `uv.lock`. Native bf16
atomics are sm_90+ in hardware, so no toolkit removes the sm_89 CAS loop; the swap is justified by staying on a
live index. To measure on the box after the profile (a `/tmp` venv, one job at a time): the scatter microbench
and the B-64 cell under `2.11.0+cu130` and `2.14.0+cu130` (whether a newer PyTorch changed the bf16
`index_add_` path or Inductor's sm < 9.0 fallback).

**Measured on the box 2026-09-24 (B-64 cell, run8 cfg (GONE at fc37f3f2: `configs/run8.yaml` was deleted at `8b00b4dd` per R368(e); the file the box read is `git show 51eddda5:configs/run8.yaml`) + 45k, IDLE; logs in the operator's mirror):**

| torch | bf16 `index_add_` microbench (1M×128, random) | B-64 cell | CUDA arch list |
|---|---|---|---|
| 2.11.0+cu128 (current) | 6.19 ms | 1 808 (mean of 4) | sm_75/80/86/90/100/120 — no sm_89 |
| 2.11.0+cu130 | 6.16 ms | 1 839 (noise) | same |
| 2.14.0+cu130 | 2.16 ms, but max error vs fp32 0.50 (was 0.375) | **1 193 (−35 %)** | same |

2.14 routes bf16 `index_add_` to a new `vectorized_scatter_add_kernel`: fast on the random microbench, but
70 % of GPU time and ~8.7 ms per call on the real batches (2.11: ~4.2 ms) — the real graphs' dst hot spots
(the dummy node) punish it. So the version bump is NOT a free baseline: it regresses exactly the op leg 3
removes. Order therefore: the cu130 index swap on 2.11 now (neutral, measured); legs 2–3 on 2.11; the torch
version bump AFTER leg 3, re-measured with `index_put_` in place (whose 2.14 path is not yet measured).

**Operator direction (2026-09-24):** cu128 was only ever the old host's driver ceiling (CUDA 12.8); torch may move
up with the perf work if it pays. Default target: the newest torch on cu130 (2.14.0 today), CPU and CUDA groups
bumped TOGETHER — one torch version across the tree, so the CPU-tested code is the code the box runs. What
the bump costs, and must be re-verified as its own commit before any perf lever lands on top of it: the WP9
CPU parity regime (captured on torch 2.11 / MKL / AVX512 — its goldens may need a re-capture, which is a
ruling), the LAW-06 bf16 parity pin and drift bands on the GPU, `compile_trunk` behaviour under the newer
Inductor (recompiles, the `dynamic=True` guidance), the determinism probe, and the admission B-64 cell
re-read on the bumped tree as the new baseline every lever is measured against.

## 2. Rulings the operator owns (before any code)

1. **Admission:** R367(e) admits this box only at ≥ 0.85×. Either the fix lifts it over the line (re-read the
   same cell), or the operator amends the criterion — the ruling states which.
2. **LAW-06:** fp32 accumulation inside the bf16 graph path is a numerics change on a pinned law (mixed
   precision: bf16 storage and GEMMs, fp32 sum). The parity test that pins LAW-06 moves with the ruling.
3. **run10's order — DECIDED by the operator 2026-09-25: run10 waits.** The prereg's §6 sequence
   (preflight → EMA cell → stamp → twin → START) is unchanged; it runs on whichever tip ask 5 admits.
4. **Placement:** its own packet (PERF-ADA, absorbing CARD-PERF-4's scope where they overlap) or a wave of the
   pruning programme, now that SLIM-FIX is merged.
5. **R368(i) versus "run10 waits":** R368(i) forbids trainer/search numerics changes before run10 STARTs, and
   leg 3 is one by design. Either (A) R368(i) is amended to admit named perf legs before run10 (then run10's
   validated base moves to the post-leg tip, its prereg's baselines re-read there, the full gate set green incl.
   the slow tier); or (B) R368(i) stands, run10 starts on the frozen base at the box's 0.74× (which needs ask 1's
   criterion amended) and legs 3–5 land for run11. Legs 0, 2 and 4 fit inside R368(i) either way (below).
6. **The cu130 swap under R368(i):** same torch version, different CUDA runtime — cuBLAS/cuDNN kernel selection
   may differ, so bit-identity is not guaranteed. Either it is shown bit-identical on the box (the repeat
   witness of leg 0, same batches, both wheels) and rides with legs 0/2, or it waits with leg 3.

## 3. The legs, in order (each: test-first, one commit per change, a LAW-09 bench, a fresh review)

| Leg | Change | Measured | Numerics | Ruling needed | Inside R368(i)? |
|---|---|---|---|---|---|
| 0 | **A real repeat witness**: `bench_server`'s probe serves distinct positions and a same-input repeat; the no-sync assertion (`set_sync_debug_mode("error")` over one served forward) as a test | — (the instrument every later leg is judged by) | none | no | yes (tools + tests) |
| 1 | **cu130 index swap on torch 2.11** (§1a) | neutral (1 839 vs 1 808) | kernel selection may differ | no | only if shown bit-identical (ask 6) |
| 2 | **The sync**: `real_mask_from_batch` via `index_fill_` | +12 % alone | bit-identical | no | yes, with a bit-identity test |
| 3 | **The aggregation**: GINE sum via `index_put_(accumulate=True)`, server and trainer (one implementation) | +87 % with leg 2; trainer −10 % | deterministic, fp32 sum, one rounding | LAW-06 (fp32 accumulation) and the precedent (config key vs new arch kind) | **no** (ask 5) |
| 4 | **Hygiene**: one intra-op thread in the serving process, a blocking CUDA event on the retirer | 0 leaves/s; 3–6 cores freed for workers + trainer | none | no | yes |
| 4b | **torch version bump** (newest cu130, CPU + CUDA together) with `index_put_` in place; new baseline | 2.14 as-is −35 % (the bf16 `index_add_` it removes) | parity re-verified | WP9 goldens if they move | **no** |
| 5 | **Exact eval cache** (per net version, keyed on the exact position) | 35.8 % steady repeat rate; ≤ ×1.5 turns/s (INFERRED) | exact | yes (search semantics, the served-copy boundary) | **no** (search path) |

Legs 0 + 2 alone (inside R368(i)) measured +12 % in the cell; the admission line (≥ 2 096) is crossed only with
leg 3 (1 808 → 2 017 with leg 2; → 3 381 with legs 2 + 3).

After leg 3 the admission cell is re-read (the plan's line: ≥ 2 096; the measured stack reads ~3 380, i.e.
~1.37× the old reference box). Everything the evidence priced at ≈ 0 (§1b) is parked with its number, and
CARD-PERF-4 is re-read against the corrected timer before anyone designs it. The strix follower on the
operator's desktop is a run-ops choice, independent of the legs (it works against the mirror as-is; its
regime label reads the box's mirrored heartbeat and would say CONTENDED).

## 4. Implementation legs

- Each lever is its own leg on its own branch, test-first, one commit per change, a LAW-09 bench per commit.
- Each leg ends with a FRESH read-only review agent (R10 review gate), findings fixed before merge, report
  a local record outside the tree (R369(f); repaired in place, this line said `docs/audits/`).
- The exit sweep is `make gates.exit` on the leg's tip; box numbers are re-read on the box, IDLE.

## 5. Box validation

- `bench_server.py` B-curve IDLE on the new tree, the same-cell determinism probe = 0, the admission line
  re-read against 2 466.
- Then run10's §6 sequence as pre-registered.

## 6. Risks carried

- The 45k parent was trained under bf16 accumulation; a more accurate sum serves slightly different numbers
  from the same weights. The A/B's numerics section states how different; the ruling decides whether that
  needs anything beyond the parity pin.
- The bench under-reads in-run serving and has no trainer beside it; the profile's real burst is the check
  that the GPU, not the workers, is the in-run bound on this box.
- The container has no kernel profilers; Rust frames come from py-spy's native unwinding only.
