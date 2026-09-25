# PERF-ADA profile — serving, self-play and trainer on the 4080S box (2026-09-24)
References re-verified against dev fc37f3f2 (post-SLIM merge) 2026-09-25; measurements unchanged (taken on 51eddda5).

One profiling agent, one box session (16:10–17:52 UTC, ≈ 1 h 42 min of box wall). Box: RTX 4080 SUPER
(cc 8.9) + Ryzen 9 9950X (16 cores / 32 threads), `torch 2.11.0+cu128`, CPython 3.14.7, tree `51eddda5`
(branch `box`, clean before and after). SM clock read 2 805 MHz in every cell, cgroup `nr_throttled` 0 in every
cell (P7 killed: no CPU throttling). Every cell below is **IDLE** (one job on the box at a time, GPU 0 % / 1 MiB
before each chain) unless marked otherwise. Nothing in the repo was edited; every A/B is a monkeypatch in a scratch
driver. Units: **leaves/s** = served leaves per second (PERF-3's unit); "cell" = `tools/bench_server.py`'s
`run_cell` at B 64, 32 workers × 8 leaves, run8 config (GONE at fc37f3f2: `configs/run8.yaml` was deleted at `8b00b4dd` per R368(e); the file the box read is `git show 51eddda5:configs/run8.yaml`) + `run8_00045000_3bdedf76.ckpt`, compiled trunk,
`checker_thread`; "burst" = the real `WorkerPool` (Rust runner + `InferenceServer` + feeder), run8 config, the 45k
weights, no trainer, no stamp. Base reference = the mean of four unsampled base cells, **1 808 leaves/s**
(1 876 / 1 787 / 1 775 / 1 795; range ±4 %).

## R369 packet ledger (PERF-ADA legs, 2026-09-25)

**Void under R369(e).** Every `bench_server` determinism reading taken before L0 is void: the old probe served
`positions[:4]` alone, so each cell read one constant batch (−0.0155016… in all 17 cells of item 3, and every
PERF-3 probe line). PERF-3's "the CPU launch stage is the bound" is void too: it was read off `launch − gpu_wait`,
two timers on different threads, while the mask sync (finding 1) inflated `launch`. CARD-PERF-4 waits on a re-read.

**The L0 instrument.** `bench_server` now serves `--probe` (64) pairwise-distinct positions twice, back to back on
the idle warm server after the load, and compares every output EXACTLY (`probe_repeat`); the measured window is
`--windows` (5) sub-windows, whose quartiles are the IQR, and `--baseline` reads each B against an earlier record
(faster = the new q1 above the baseline's q3). The no-sync witness is
`tests/selfplay/test_served_forward_no_sync.py`: sync-debug "error" over the second pop's `_launch_pop`, with a
planted `.item()` that must red; on the desktop 3070 at L0 it catches exactly `real[legal_index] = True`, and
`index_fill_` clears it. Out of its reach: an allocator-growth sync (`cudaMalloc` / `cudaHostAlloc` on a cache miss),
which sync-debug mode does not flag; the checked pop repeats the first pop's shapes, so its caches hit.

**L2's error criterion: v1 was pre-stated, failed its own control, and was replaced by E before the new path was
measured.** v1 (pooled max |Δ| vs fp32 over 8 real batches ≤ the current path's pooled repeat spread) fails on the
CURRENT path itself: desktop 3070 logit 1.40 > 1.0; box 4080S 4 of 10 single draws; the literal L0 form on the box
(64 probe positions, served compiled) value 0.0962 > 0.0657 and prob 0.0191 > 0.0153. Today's error vs fp32 is
mostly systematic (the other bf16 ops), so an error-vs-jitter band at the output is unsatisfiable. A red-team review
selected **E** (the session adopts it; UNRATIFIED by the operator):
- **(ii-a) gate, at the aggregation**: on every GINE layer's real `(msg, dst, n, divisor)` captured from the
  parent's 8 real B-64 batches (eager bf16 forward), the production aggregation, eager and compiled, returns bf16,
  is exactly 0 where the fp64 sum is 0, is within one bf16 ulp (`2^(e−8)`, `frexp` of the fp64 sum) of it
  elsewhere, and its pooled max |Δ| vs fp64 is ≤ the pre-L2 sum's pooled range over 5 repeats.
- **(ii-b) guard, end to end**: pooled mean |Δlogit| vs the fp32 forward ≤ the pre-L2 path's worst of 5 repeats.
- **Controls first**: the pre-L2 sum must break the ulp bound and jitter (the instrument is live); an exact fp32
  oracle must pass (ii-a) eager and compiled and (ii-b). Both hosts. HALT if production fails (i), (ii-a) or (ii-b).

**Finding 5 is corrected (repaired in place, R311(c)).** On bf16, `index_put_(accumulate=True)` rounds after every
add in a fixed order: op-level max |Δ| vs fp64 on the desktop 3070 is 586.6, identical to bf16 `index_add_`; only an
fp32 buffer cast once stays within one bf16 ulp (4.0). The box's `agg_micro.json` shows the same (185.6 vs 179.6).
L2 therefore implements R369(b) as ONE custom op, `mantis::gine_aggregate`: an fp32 buffer summed by the
deterministic sorted `index_put_` in bounded edge chunks, divided in fp32, rounded once; its backward is the bf16
gather; opaque to Inductor, so the compiled server cannot lower it to fp32 atomics (non-deterministic, item 3).
The trainer-gradient repeat (witness (i)) then still differed on the desktop: the gather `x[src]` has a graph
aggregation as its gradient (bf16 atomics over the repeated sources), so L2 also routes it through ONE op,
`mantis::gine_gather`, whose backward is `gine_aggregate` over `src`; the forward is the same `index_select`. On CPU
both ops sum by `index_add_` (serial): CPU `index_put_(accumulate=True)` races (repeat |Δvalue| 1.9e-8 measured).

**(iii), re-stated before L2 was measured.** Two runs of the same path at the cap differ by 18 MiB (allocator state),
so the two paths are alternated twice in one process and max(production) ≤ max(pre-L2) is the rule.

**LAW-09 bench ledger** (bench_server, B 64, IDLE box, `--windows 5`; expected gain and abort threshold stated
before the bench):

| leg / commit | expected | abort below | benched sha | median leaves/s [IQR] | vs parent |
|---|---|---|---|---|---|
| L0 instrument (parent of L1) | 0 (tools and tests only) | — | `c9474c85` | 1 853 [1 768, 1 869]; repeat DIFFERS, max \|Δvalue\| 0.0657, \|Δp\| 0.0153 | — |
| L1 `index_fill_` mask | +8 … +15 % (item 3: +12 %) | median < 0.97 × parent, or slower beyond the IQR | `a25183cd` | 2 101 [1 994, 2 106]; `launch` 31.4 → 8.9 ms; repeat DIFFERS 0.0671 / 0.0148 | **+13.4 %**, faster beyond the IQR |
| L2 fp32 sorted aggregation (custom op) | +30 … +65 % over L1 (item 3 idxput + nosync: +87 % over base, but fp32 values and chunking cost more) | median < 0.97 × L1, or slower beyond the IQR | `6efa0c2e` | 1 934 [1 907, 2 003]; `launch` 8.9 → 11.4 ms, `gpu_wait` 25.3 → 26.6 ms; repeat: value EXACT, prob 1.2e-7 (the fp32 `segment_sum` of the served softmax) | **−7.9 %: PAST THE ABORT LINE (2 038) — HALT** |
| B3 fixed-order readout sums (`segment_reduce`, fp32) | +2 … +8 % over L2 (the pools and served softmax sum ~2× faster than their atomics) | median < 0.97 × L2 (1 876), or slower beyond the IQR | `d5605bf2` | 1 949 [1 902, 2 027]; repeat **EXACT** (value and prob) | +0.7 % (noise; below the expected band, above abort) |
| L2′ fused CSR aggregation (B1) | +62 … +110 % over L1 (think agent INF from the box forward times) | median < 0.97 × L1 (2 038), or slower beyond the IQR; below +40 % (2 941) stop and profile | `c9392c95` | **3 881** [3 869, 3 905]; `gpu_wait` 25.3 → 7.4 ms, `launch` 12.3 ms (now the bound); repeat EXACT | **+84.7 % over L1** (2.09× L0) |
| L2″ per-layer recompute in training (the operator's remedy for (iii)) | trainer peak at the cap −3 … −4 GiB; step +10 … +20 % over B1 (381 → 420–460 ms); serving unchanged (inference path untouched) | peak not under both (iii) and the OF2-10 budget; step > 543 ms (slower than before the packet); serving median < 0.97 × B1 | `aa143027` | serving **3 982** [3 952, 3 998], repeat EXACT; trainer peak **4.69 GiB** (was 8.94; OF2-10 margin 47 %); step **407 ms** (was 547) | serving +2.6 % over B1 (noise); peak −4.2 GiB; step −25 % vs before the packet |
| L3 torch 2.14.0+cu130 (Triton 3.8), a /tmp venv over the recompute tip | land only if faster beyond the IQR (R369(c)) | not faster | `9f4e8768` + 2.14 | 3 849 [3 784, 3 864], repeat EXACT; determinism, compile, no-sync and goldens 27/27 green | −3.3 % (noise): **NOT LANDED**, the tree stays on 2.11+cu128 |
| L3b torch 2.11.0+cu130 (operator's ask: the runtime alone) | land only if faster beyond the IQR | not faster | `9f4e8768` + 2.11+cu130 | 3 827 [3 760, 3 827], repeat EXACT; 27/27 green; `launch` 13.1 vs 12.2 ms | −3.9 %: **NOT LANDED**; both cu130 builds read 3–4 % under cu128 with a slower launch stage |
| H2 + H3 hot-path hardening (named errors, one search helper set), integrated | no change (a hot-path edit may not regress) | median < 0.97 × the recompute tip | `9e4ed476` | 3 981 [3 968, 4 017], repeat EXACT | −0.0 % |
| H4 `par::map_in_order` vs its inlined parent (criterion `graph_build_bench`, box) | revert if the shared form is slower where production runs | — | tip vs tip + revert | 64 leaves: 1 thread revert **+10.0 %** slower, 4 threads revert −6.3 %, 8 threads ±1 %; 8 leaves ±1 % | production runs the sample fan-out at **1 thread** on this box (`max(1, cpu − n_workers − 1)` = 1), where the shared form is faster: **KEPT**, the conditional revert dropped |
| L4 exact per-net eval cache (R369(d)) | lands only if served outputs are bit-identical with and without it | — | branch `perf-ada-l4` (`94784c4a`), NOT on perf-ada | the served path is NOT batch-invariant (3070, the parent, 64 distinct positions vs the whole batch): groups of 1 → 38–39/64 differ (max \|Δvalue\| 0.0076, \|Δp\| 2e-3); groups of 4–16 → 33–35/64 (\|Δvalue\| 0.0075); groups of 32 → 0–1/64 | **NOT LANDED**: a hit replays a value from another batch size; the bf16 value-head GEMM's kernel choice varies with B |

The repeat probe reads DIFFERS on both rows by construction: the mask is bit-identical, and the non-determinism is the bf16 `index_add_` aggregation L2 replaces.

**L2 on the box (4080S, `6efa0c2e`):** every witness green — (i) exact eager / compiled / trainer gradients; (ii-a)
0 outputs past one ulp, max |Δ| 2.42 vs the pre-L2 spread 16; (ii-b) 0.045222 vs 0.051383; (iii) peak 8.877 vs
8.878 GiB; OF2-10 legs 2/2b 8.877 GiB (margin 0.2 %), ratio 1.000. Trainer step A/B (run10 cfg, 45k weights and
ring, alternated): 530.7 / 529.9 ms pre-L2 vs 529.6 / 533.4 ms production — no gain (item 3's −10 % was the bf16
sorted path); peaks 8.92 / 8.71 vs 8.88 / 8.83 GiB. The serving bench regresses past its abort line: HALT.
**HALT diagnosis (measurement only, same cell, 60 s × 3 windows, scratch patches of the op's module globals):**
committed 2 051 [1 837, 2 088]; one chunk 2 294; plain `index_select` gather 2 218; plain gather + one chunk
**2 485** [2 483, 2 596] (+18 % over L1). Both costs are real: the opaque gather blocks Inductor's gather+add+relu
fusion, and 2^18-edge chunks multiply the sort and copy launches (`launch` 12.7 → 9.2 ms, `gpu_wait` 27.5 → 18.8
ms). Even the best deterministic fp32 variant sits far below item 3's 3 378–3 393, which was the NON-deterministic
fused fp32 atomic path (or the bf16 sorted path, 3 381, which does not accumulate in fp32). Served probabilities
still differ by 6e-8–1.2e-7 on a repeat in every variant: the served softmax's fp32 `segment_sum` (and the value
pool's `index_add_`) are atomic; the value head output is exact.

**After the HALT (operator, 2026-09-25): resume with the fused CSR op (B1) replacing the committed L2 ops.** A
read-only think agent root-caused the regression: the dummy hub's ~1000-deep serial run in ATen's sorted
`index_put_`, paid once per 2^18-edge chunk; the opaque gather blocking Inductor's fusion; the fp32 copy of the
messages. Its prototype (desktop): 4-layer aggregation 3.3–3.9 ms vs 44.1 committed and 7.1 fp32 atomics; full
forward 21.6 ms vs 28.3 fp32 atomics; bitwise equal to the committed op. The operator's conditional on determinism
(accept non-determinism only if no deterministic fp32 path comes within ~10 % of the fp32-atomic path) is decided
by B1's box reading. The later levers are carded (CARDS.md, "Opened by the PERF-ADA packet").

**B1 on the box (`c9392c95`):** (i) exact served and trained; (ii-a) 0 outputs past one ulp, max |Δ| 4 vs spread
12–16; (ii-b) 0.045222 vs 0.051124; no-sync green. Trainer step A/B 543 → **381 ms (−30 %)**. **(iii) FAILS:** peak at
the cap 8.939 → 9.103 GiB (+168 MiB), and OF2-10 leg 2 now exceeds its 8.897 GiB sizing budget (margin −2.3 %). The
rise is the op saving `xs` (each layer's bf16 [N, H] input) for its backward: 169 904 × 128 × 2 B × 4 layers =
166 MiB, which the bf16 path never saved (it saved the message, as B1 saves `e`). A HALT line of the packet; the
remedy is the operator's call (see the exit record).

**B1's box microbench (IDLE 4080S, one real B-64 batch of 1.09 M edges, compiled trunk, full served forward):**
L1 bf16 `index_add_` 34.1 ms (not repeatable); committed L2 35.6 ms; fp32 atomics fused 17.6 ms (not repeatable);
**B1 fused CSR 11.8 ms** (repeatable); B1 + edge table 6.5 ms (repeatable). B1 is 1.5x faster than the
non-deterministic fp32 path, so the operator's conditional does not fire: determinism stays (R369(b) unamended).
B1 changes three served facts, stated: the degree divisor is exact fp32 (the bf16 path rounded 779 to 780 on the
dummy); the per-edge add is rounded to bf16 inside the kernel exactly as autocast's add was; CPU keeps its edge
order and its serial `index_add_`, so the CPU goldens stay bit-identical.

## Findings first

1. **The serving bound on this box is a hidden per-forward host↔device sync, then the bf16 atomics.**
   `GnnNetV2.real_mask_from_batch` (`src/mantis/model/gnn_v2.py:118`, `real[legal_index] = True`; removed at L1, `a25183cd`) copies a
   pageable 0-dim CPU tensor to the device on every forward, which blocks the server thread until the stream drains.
   py-spy puts **78.4 %** of the server thread's wall on that one line in the cell (2 761 samples) and **66.9 %**
   in-run. `torch.cuda.set_sync_debug_mode` flags the same line. The server thread's CPU-only share is **13.4 %**
   (≈ 4.6 ms of a 34.6 ms pop). The rest is 8.2 % queue wait and 78.4 % waiting on the GPU.
2. **"launch − gpu_wait" was never the CPU stage.** With the sync removed (`index_fill_`, bit-identical mask), the
   `launch` timer drops from **32.4 to 8.9 ms/pop**. The server thread's run share drops from 0.89 to 0.34. The cell
   reads **2 017 leaves/s (+12 %)** and GPU utilisation rises from 87 % to 95 %. This settles the two research
   reports: the code has a two-thread pipeline, but this sync serialises it (F-47's "the halves never overlap" is
   true here, through a different mechanism). The sync fix is what lets it overlap.
3. **fp32 accumulation in the GINE sum: 2 857 leaves/s (+58 %), and 3 378–3 393 with the sync fix (+87–88 %).**
   Under the compiled trunk, Inductor fuses gather + add + relu + fp32 scatter into one Triton kernel. On the real
   B-64 batch that kernel takes **0.99 ms per layer, against 5.45 ms** for the compiled bf16 path (still the ATen
   CAS-loop `indexFuncLargeIndex`). Eager, the same change gives only 1 472 → 1 788 (+21 %).
4. **In the real self-play loop the stacked fix nearly doubles throughput.** The burst with fp32acc + sync fix +
   OMP-1 reads **3 691 leaves/s vs 2 010**, **1 320 vs 696 games/h** and **28.6 vs 15.9 compound turns/s**.
   GPU utilisation is 99.9 % vs 97.1 %. The burst is GPU-bound both before and after; the 32 Rust workers use
   **1.4 cores** (base) and 2.7 cores (fixed) of 16.
5. **`index_put_(accumulate=True)` is the deterministic twin of fp32acc.** It serves at **2 824 leaves/s alone and
   3 381 with the sync fix**, with **exactly 0** run-to-run difference on real batches. It uses PyTorch's sort-based
   path, which accumulates in fp32 and rounds once [CORRECTED: on bf16 it rounds after every add; see the R369 ledger]. It also speeds the trainer: **469 vs 523 ms/step** (−10 %),
   CUDA kernels 269 vs 321 ms, same 8.81 GiB peak.
6. **Today's production bf16 path is not run-to-run deterministic.** The same 9 real batches forwarded 3× give a
   max |Δvalue| of **0.105–0.115** and a max |Δlogit| of **0.75–0.875**, eager and compiled. The bench's own
   "determinism probe" (4 positions served alone) reads −0.0155016… in **every** variant. It cannot see this.
   CARD-PERF-4's "the probe must read 0" needs a real same-input repeat witness.
7. **Numerics of the levers stay inside the base path's own noise.** Against an fp32 (no-autocast) reference, the
   mean |Δvalue| over 9 batches is 0.0089 for base (compiled), 0.0073 for fp32acc and segsum, and 0.0100 for idxput.
   p99 |Δvalue| is 0.074 / 0.059 / 0.059 / 0.072. Every variant is smaller than base's own repeat spread.
8. **fp32acc in the TRAINER is a memory hazard with no speed gain.** At the production microbatch cap (4.5 M edges)
   it **OOMs** the 16 GB card: 14.18 GiB allocated, against **8.81 GiB** for base. At a halved cap it peaks at 9.27
   vs 4.99 GiB (+4.3 GiB) and step wall is unchanged (517 vs 520 ms). The serving path does not have this problem:
   peak allocated at B 64 is 0.81 GiB in both base and fp32acc (fused, no fp32 temporary).
9. **Trainer step at run8 shape: 523 ms wall, 321 ms CUDA kernels, bf16 `index_add_` = 38 % of kernel time.**
   This is batch 256 off the 45k ring, cap 4.5 M edges. At run10's 2.4 steps/game that costs ≈ 15 % of GPU time at
   base games/h and ≈ 28 % at the fixed rate (INFERRED). While serving, the GPU is already at 97 %, so in-run the
   trainer displaces serving roughly 1:1.
10. **Eval-cache census: 35.8 % of served leaves in steady state are exact repeats of an already-evaluated
    position.** This is global across workers, over a 240-s window after 2 min of warm-up. It is not caused by the
    openings: the first 2 min read 34.6 %. An LRU of 65 536 entries catches 34.5 % and one of 4 096 catches 25.9 %.
    Only 1.7 % are duplicates inside the same pop.
11. **Nulls, each measured.** `target-cpu=native` gives 1 861 vs 1 808 in the cell and 1 931 vs 2 010 in the burst.
    OMP-1 (`torch.set_num_threads(1)`) gives 1 799 vs 1 808, but frees **2.6–5.2 cores** that the collate's parallel
    `pin_memory` pool spin-waits away. CPU bf16 inference on the idle cores reaches **143 leaves/s at 8 threads**
    (448 ms per B-64 batch), +3.9 % of the fixed rate for half the cores.
12. **The wheel has no sm_89 code.** `torch.cuda.get_arch_list()` = sm_75/80/86/90/100/120 on a (8, 9) device, so
    the 4080S runs sm_86 SASS. bf16 `atomicAdd` there is the CAS loop. Its cost tracks the data: 7.67 ms with no
    zero messages, 3.99 ms at the real 97.5 % zero fraction, and **12.1 ms with a dummy-node hot spot**, while fp32
    stays flat at 2.83 ms. This is why random-init weights (dense messages) serve slower than trained ones.

## Item 1 — Serving: baseline cell and where the server thread's wall goes

Commands (cwd = the box checkout root; scratch scripts in `/tmp/perf-ada/`, copies in the mirror):

```
.venv/bin/python /tmp/perf-ada/serve_cell.py base 120 20 compile /tmp/perf-ada/ab_base_compile.json
py-spy record --threads --idle --nonblocking -r 100 --format raw -o /tmp/perf-ada/serve_pyspy.txt -- \
  .venv/bin/python /tmp/perf-ada/serve_cell.py base 120 20 compile /tmp/perf-ada/pyspy_serve_cell.json
PERF_SYNCDBG=1 .venv/bin/python /tmp/perf-ada/serve_cell.py base 30 10 compile /tmp/perf-ada/ab2_syncdbg.json
```

`serve_cell.py` loads `tools/bench_server.py` by `importlib` and calls its `run_cell` unchanged. It hooks
`InferenceServer.batch_timing_snapshot`, the two window marks, to add per-thread `/proc/self/task/*/{stat,schedstat}`
deltas, `/proc/stat` box CPU, nvidia-smi at 200 ms, `torch.cuda.memory_stats` deltas and cgroup `cpu.stat`.

**The sampler: what worked and what did not.** `py-spy record --native` at 250 Hz, run with the sampler as the
parent, **starved the target**: the child got 1 s of CPU in 8 min of wall and was killed. A 20-Hz `--native` retry,
time-boxed at 240 s, did not finish either. Native unwinding over this process's roughly 40 Python threads plus
libtorch's symbol tables is not usable here. The profile is therefore **Python-level, `--nonblocking`**, which never
pauses the target: the cell read 1 851 leaves/s under the sampler against 1 808 unsampled. At that load py-spy
achieved ~18 Hz per thread, 2 761 samples. Rust worker threads are not Python threads and do not appear in it; their
time comes from schedstat below. The symbol-bearing build (`CARGO_PROFILE_RELEASE_STRIP=none
CARGO_PROFILE_RELEASE_DEBUG=line-tables-only`) was made in the one worktree and served at the main tree's rate
(1 795). Its Rust symbols went unused because native unwinding failed. **GIL wait: NOT MEASURED** (Python-level
sampling cannot see `take_gil`).

**Cell numbers (IDLE, 120 s):** 1 876 leaves/s, B 64.0, cycle 34.1 ms, `queue_wait` 3.0, `collate` 4.0, `launch`
31.1, `gpu_wait` 24.5 ms. GPU utilisation 88 %, 202 W, box CPU 17.8 %. Peak allocated 0.81 GiB (reserved 4.2–6.8
GiB, which grows with dynamic shapes). `num_alloc_retries` 0 and 4–10 device allocations per 60-s window (P2
killed: the allocator does not churn).

| thread (py-spy, share of its own wall) | where |
|---|---|
| **server** (`_run_graph_loop`) | **78.4 % `gnn_v2.py:118` `real[legal_index] = True` (GPU sync)** · 8.2 % `next_graph_batch` (queue wait) · 7.2 % `collate_graph_batch` (2.3 % pinned copy `graph_collate.py:437`, ≈ 2 % checks 11–13) · ≈ 6 % rest (forward Python/launches, `segment_softmax` 0.5 %, `stone_mask` 0.7 %, `slice_graph_wire` 0.4 %) |
| retirer (`_PopRetirer`) | 76.1 % `event.synchronize()` (a **spin** wait: schedstat run share 0.68–0.87 core) · 18.6 % idle · 5 % dispatch |
| checker (`_EdgeGeometryChecker`) | 22.2 % `verify_edge_geometry` · 77.5 % idle |
| 32 bench feeders | 92.6 % blocked in `submit_graphs_and_wait` |
| 15 OMP threads (comm `inference-serve`, inherited from the server thread) | not Python; schedstat **2.6 cores** of spin (the parallel `pin_memory` in `_device_copier`, `graph_collate.py:437`) |

`torch.cuda.set_sync_debug_mode("warn")` names two synchronizing call sites in the served forward:
`gnn_v2.py:118`, which fires every forward, and `dist65.py:46`, which fires once to fill a cache. **P1 answered:**
the block inside `_launch_pop` is a synchronous pageable H2D copy inside `real_mask_from_batch`, at the top of the
forward. It is not the allocator (0 retries), not a pinned allocation, and not `.item()`/`nonzero`. **The true
CPU-only share of the server thread is 13.4 %** (4.6 ms/pop, sampled). After the fix, the `launch` timer reads
8.5–9.1 ms/pop. The gap between the two (the collate grows from 4.0 to 5.9 ms once the host runs ahead) is not
separated here.

## Item 2 — The real self-play loop, and the trainer step

Command (the pool is built as `mantis.diagnostics.worker_sweep.build_sweep_pool` builds it, but from the 45k
weights; no trainer, no stamp, no run dir):

```
.venv/bin/python /tmp/perf-ada/burst.py 420 30 /tmp/perf-ada/burst_clean.json
.venv/bin/python /tmp/perf-ada/burst.py 360 30 /tmp/perf-ada/burst_fixed.json nocensus fp32acc+nosync+omp1
```

Steady window = from 120 s to the end, which drops the compile and the synchronised openings.

| burst (IDLE, run8 cfg + 45k, 32 workers) | window | leaves/s | games/h | turns/s | leaves/turn | B | cycle ms | `launch` / `gpu_wait` / `queue_wait` ms | GPU util | box CPU |
|---|---|---|---|---|---|---|---|---|---|---|
| base, release build | 300 s, 58 games | **2 010** | **696** | 15.9 | 127 | 50.1 | 24.9 | 22.5 / 21.3 / 2.5 | **97.1 %** | 21.4 % |
| base, `target-cpu=native` | 240 s, 42 games | 1 931 | 630 | 15.4 | 125 | 50.1 | 26.0 | 23.3 / 22.2 / 2.7 | 97.2 % | 21.0 % |
| base + census hasher | 240 s, 46 games | 1 968 | 690 | 15.2 | 129 | 50.1 | 25.5 | — | 96.8 % | 21.5 % |
| **fp32acc + sync fix + OMP-1** | 240 s, 88 games | **3 691** | **1 320** | **28.6** | 129 | 38.6 | 10.5 | 5.3 / 9.8 / 3.7 | **99.9 %** | 15.6 % |

Games per window are few, so games/h carries about ±15 % sampling noise. turns/s and leaves/s are the stable
figures: 30-s windows range 1 886–2 269 (base) and 3 532–3 984 (fixed).

**Where the process's CPU goes in-run** (schedstat run share over the base window, in cores): server thread 0.99
(spinning in the same sync; py-spy over a 30-s in-run slice puts 66.9 % of it on `gnn_v2.py:118`, 11.2 % in queue
wait and 9.5 % in collate). Retirer 0.87 (spin), OMP pool 3.2 (spin), checker 0.32, **the 32 Rust workers together
1.4** (≈ 4–6 % each). The workers are therefore ≈ 95 % blocked waiting on inference: the worker CPU per leaf is
0.7 ms in a 16-ms-per-turn loop. The MCTS-versus-graph-build split inside Rust is **NOT MEASURED** (py-spy cannot
unwind natively here; see item 1).

**Bench vs in-run:** 2 010 in-run vs 1 808 in the cell. In-run pops are smaller (B 50) and early graphs smaller.

**Is the in-run bound the GPU, the server thread or the workers?** The GPU, at 97.1 % utilisation, and it stays the
GPU after the fix at 99.9 %. The server thread looks 99 % busy in-run, but that is the sync spin, not work. The
workers use 1.4 of 16 cores.

**The trainer step** (the STARTPATH §C method: production `init_trainer`, `run_declared_train_step`, 3 warm-up
steps, 20 timed steps, 10 profiled; run8 config with `identity.warm_start` nulled and the 45k weights loaded; ring =
the 45k bundle's 100 000 rows; batch 256; `sample_threads` 1; IDLE):

```
.venv/bin/python /tmp/perf-ada/trainer_prof.py base /tmp/perf-ada/trainer
PERF_MAXEDGES=2250000 .venv/bin/python /tmp/perf-ada/trainer_prof.py fp32acc /tmp/perf-ada/trainer
```

| variant | microbatch cap (edges) | step wall median | CUDA kernel ms/step | bf16/fp32 `index_add` share of kernels | peak allocated |
|---|---|---|---|---|---|
| base | 4.5 M (production) | **523 ms** | **321** | **38.1 %** | **8.81 GiB** |
| base | 2.25 M | 520 ms | 318 | 38.6 % | 4.99 GiB |
| fp32acc | 2.25 M | 517 ms | 316 | 24.0 % | 9.27 GiB |
| **fp32acc** | **4.5 M** | **OOM** (14.18 GiB allocated, 278 MiB request refused) | — | — | — |
| **idxput** | 4.5 M | **469 ms** | **269** | 21.6 % | 8.81 GiB |
| segsum | 4.5 M | OOM (the first, unfixed-accounting pass) | — | — | — |

The kernel totals here count CUDA kernel rows only. The first pass summed ATen op rows and kernel rows together
and read 643 ms, a double count. Wall is 0.2 s above kernel time. That gap is CPU-side (ring sampling, collate,
checks) and was not decomposed. F-44's reading, that the trainer is not the in-run bound, still holds as a regime
fact at run8's cadence. At run10's 2.4 steps/game and the fixed games/h, trainer GPU time is ≈ 3 168 steps/h ×
0.32 s ≈ 28 % of the GPU (INFERRED). That makes the trainer the next thing the serving gain has to share the card
with.

## Item 3 — A/B: the aggregation variants in the real server

Command, per variant: `.venv/bin/python /tmp/perf-ada/serve_cell.py <variant> {120|60} {20|15} compile <out>`.
Variants are class-level monkeypatches in `/tmp/perf-ada/patches.py`:

- `fp32acc`: `agg` is an fp32 `zeros` + `index_add_(msg.float())`, divided in fp32, cast back to the autocast
  dtype.
- `segsum`: dst-sort once per forward on the GPU (argsort + searchsorted, shared by the four layers), then
  `torch.segment_reduce(..., "sum", offsets=)` in fp32. In a real design the sort moves into the Rust collate: it
  costs 1.21 ms per pop on the GPU.
- `idxput`: `new_zeros().index_put_((dst,), msg, accumulate=True)`.
- `det`: base under `torch.use_deterministic_algorithms(True, warn_only=True)`.
- `+nosync`: `real_mask_from_batch` → `stone_mask.clone().index_fill_(0, legal_index, True)`.
- `+omp1`: `torch.set_num_threads(1)`.

All cells IDLE, B 64.0 throughout.

| variant | leaves/s (cells) | vs 1 808 | cycle ms | `launch` ms | `gpu_wait` ms | GPU util | box CPU | server run share | peak alloc (serve) |
|---|---|---|---|---|---|---|---|---|---|
| base | 1 876 / 1 787 / 1 775 / 1 795 | — | 34.1–36.1 | 31.1–32.5 | 24.0–24.5 | 87–88 % | 17.5 % | 0.88–0.90 | 0.81 GiB |
| base, eager (no compile) | 1 472 | −19 % | 43.5 | 40.0 | 30.5 | 92 % | 15.5 % | 0.91 | — |
| **base + nosync** | **2 017** | **+12 %** | 31.7 | **8.9** | 25.7 | **95 %** | 19.1 % | **0.34** | 0.81 GiB |
| base + omp1 | 1 799 | 0 % | 35.6 | 32.4 | 24.0 | 87 % | **9.4 %** | 0.89 | 0.81 GiB |
| **fp32acc** | **2 860 / 2 853** | **+58 %** | 22.4 | 19.1 | 13.0 | 79 % | 24.0 % | 0.83 | 0.81 GiB |
| fp32acc, eager | 1 788 | (+21 % vs eager base) | 35.8 | 32.5 | 24.4 | 88 % | 17.5 % | 0.90 | — |
| **fp32acc + nosync** | **3 378** | **+87 %** | 18.9 | 8.5 | 12.9 | 92 % | 29.2 % | 0.55 | 0.81 GiB |
| **fp32acc + nosync + omp1** | **3 393** | **+88 %** | 18.9 | 8.5 | 13.2 | **97 %** | **12.3 %** | 0.54 | 0.81 GiB |
| segsum | 2 637 / 2 629 | +46 % | 24.3 | 21.1 | 14.7 | 81–82 % | 22.6 % | 0.85 | 1.45 GiB |
| **idxput** | **2 824** | **+56 %** | 22.7 | 19.6 | 12.4 | 75 % | 23.7 % | 0.82 | 0.86 GiB |
| **idxput + nosync** | **3 381** | **+87 %** | 18.9 | 9.1 | 12.9 | 87 % | 29.3 % | 0.57 | 0.86 GiB |
| det | 2 244 | +24 % | 28.5 | 25.5 | 16.5 | 81 % | 24.9 % | 0.85 | 1.20 GiB |
| base, `target-cpu=native` (item 4) | 1 861 | +3 % (noise) | 34.4 | 31.3 | 24.7 | 84 % | 17.6 % | 0.89 | 0.81 GiB |

**Numerics** (`numerics.py`). Nine batches were captured from a live eager cell: the 4-position probe, one of 32
graphs and seven of 64 (335 k–1.21 M edges). Each variant was then forwarded 3× eager and 3× compiled under bf16
autocast, and compared with an fp32 no-autocast reference (`ref32`, whose own repeat spread is 2e-5 logit and
4e-6 value):

| variant / mode | repeat max \|Δlogit\| / \|Δvalue\| | vs fp32 ref: max \|Δv\| · mean \|Δv\| · p99 \|Δv\| · mean \|Δlogit\| | forward median (captured batches) | peak alloc |
|---|---|---|---|---|
| base eager | **0.75 / 0.109** | 0.087 · 0.0082 · 0.065 · 0.052 | 39.3 ms | 1.52 GiB |
| base compiled | **0.75 / 0.105** | 0.102 · 0.0089 · 0.074 · 0.048 | 31.2 ms | 0.95 GiB |
| fp32acc eager | **0 / 0** | 0.118 · 0.0101 · 0.086 · 0.045 | 31.8 ms | 1.84 GiB |
| fp32acc compiled | 0.125 / 0.003 | 0.092 · 0.0073 · 0.059 · 0.039 | **17.5 ms** | 0.95 GiB |
| segsum eager | **0 / 0** | 0.118 · 0.0101 · 0.086 · 0.045 | 31.5 ms | 2.15 GiB |
| segsum compiled | **0 / 0** | 0.092 · 0.0073 · 0.059 · 0.039 | 19.6 ms | 1.51 GiB |
| idxput eager | **0 / 0** | 0.118 · 0.0087 · 0.065 · 0.051 | 25.2 ms | 1.52 GiB |
| idxput compiled | **0 / 0** | 0.125 · 0.0100 · 0.072 · 0.047 | **16.8 ms** | 0.99 GiB |
| det eager / compiled | **0 / 0** | as idxput (the same kernel path) | 32.6 / 22.7 ms | 1.59 / 1.29 GiB |

What this confirms and what it kills:

- **Confirmed:** the aggregation is the lever, and under compile the win is the Triton fusion, not only the
  accumulator dtype. On the real batch the ops time as follows. Fused compiled fp32 path: 0.99 ms. Eager fp32
  `index_add_`: 2.88. Current bf16: 4.60. Compiled bf16 (`index_add_` not lowered): 5.45.
- **Killed:** the premise that fp32 accumulation costs accuracy or determinism. It is closer to the fp32 reference
  than base, and more repeatable.
- **Open for a ruling, not measured further:** the regime pins in `tests/model/test_gine_*` and the bf16
  parity/null-distribution bands move with any of these variants (LAW-06's autocast dtype itself is untouched).

**The same-cell determinism probe:** `bench_server`'s `probe_values` read −0.015501638874411583 on all four
positions in **all 17 cells** of every variant. The probe's four positions evaluate identically and are too small to
exercise the atomics. It reads "0" whatever the kernel does. The repeat test above is the witness that bites: base
fails it at 0.105; idxput, segsum and det pass at exactly 0.

**Aggregation microbench on ONE real B-64 batch** (`agg_micro.py`; batch 4 of the capture: 49 406 nodes,
1 210 868 edges, H 128, layer-1 `msg`; 30 reps after 3 warm; IDLE). Real-batch facts: `msg` zero fraction
**97.5 %**; dummy in-edges 4.1 % of edges (64 dummies, in-degree up to 997); max real in-degree 31 (so the ELL table
width K is 31).

| op on the real (dst, msg) | ms | repeat max \|Δ\| |
|---|---|---|
| bf16 `index_add_` (current) | 4.60 | 12.0 (non-deterministic) |
| bf16 `index_add_`, dst-sorted input | 5.06 | 13.0 |
| bf16 `index_add_`, dummy edges removed | 3.88 | 0.5 |
| fp32 `index_add_` + cast | 2.88 | 0 |
| `index_put_(accumulate=True)` bf16 (sort-based) | **1.16** | **0** |
| bf16 `index_add_` under `use_deterministic_algorithms` | 2.12 | 0 |
| `segment_reduce` fp32, pre-sorted | 2.50 | 0 |
| argsort + searchsorted + permute (once per pop) | 1.21 | — |
| ELL fixed-slot (K = 31 gathers, fp32) + dummy `index_add_` | 1.95 | 0 |
| ELL dense gather + sum | 4.25 | 0 |
| msg compute + bf16 `index_add_`, eager / compiled | 7.42 / 5.45 | 11 / 18 |
| **msg compute + fp32 `index_add_`, compiled (Triton-fused)** | **0.99** | 0 |

| synthetic bf16 `index_add_` at the real shape | bf16 ms | fp32 ms |
|---|---|---|
| 0 % zeros | 7.67 | 2.83 |
| 50 % zeros | 7.17 | 2.83 |
| 90 % zeros | 5.72 | 2.83 |
| 97.5 % zeros (the real fraction) | 3.99 | 2.83 |
| 0 % zeros + one hot row per graph (880 edges each) | **12.14** | 2.83 |

This answers the research's P3 and P12 and the online report's "(3)": the bf16 cost tracks the zero fraction and
the hot spot (the CAS loop), and the fp32 cost is flat. **The ELL table (lever 3)** measures 1.95 ms. That is below
eager fp32 but twice the compiled fused fp32 path, which is already at 0.99 ms. On this evidence ELL buys nothing
that `compile` + fp32 does not, and costs a wire contract.

## Item 4 — `target-cpu=native`

In the one worktree (`/tmp/wt-prof`, the same `51eddda5`). The venv was **copied** from the main tree: this box's
`uv` runs without a cache, and a fresh `make build.cuda` failed to download torch (a connect timeout to the wheel
index). `uv sync` then reinstalled only the two local packages. The native build used
`RUSTFLAGS="-C target-cpu=native" uv sync --extra cuda --no-group cpu --reinstall-package mantis-engine`. That is
`make build.native`'s semantics with the CUDA extra; the Makefile's own line would revert torch to CPU (33 s build).

| | leaves/s | `launch` − `gpu_wait` ms | burst leaves/s (turns/s) |
|---|---|---|---|
| release (portable) | 1 808 (4 cells) | 6.6–8.5 (not a CPU measure: see item 1) | 2 010 (15.9) |
| `target-cpu=native` | 1 861 | 6.6 | 1 931 (15.4) |

**Null** on both instruments: the cell is +3 % inside the ±4 % base spread, and the burst is −4 % inside its
window-to-window spread. It cannot matter while the GPU is the bound and the Rust workers use 1.4 cores. Consistent
with the research's lever 13. Afterwards: `git worktree remove --force /tmp/wt-prof && git worktree prune`,
`git status` 0 lines, `df -h /` 51 G free.

## Item 5 — Cheap probes outside the box's current design

- **Eval-cache hit rate (P8, measured).** `burst.py 360 30 … census base`: a GIL-releasing hasher thread
  (blake2b-96 over each graph's `node_feat` + `node_coords` + `current_player`, i.e. the net's whole input) hashed
  762 180 served leaves.

  | window | leaves | hit rate (seen before in the process) |
  |---|---|---|
  | first 120 s | 289 786 | 34.6 % |
  | after 120 s (steady) | 472 394 | **35.8 %** |
  | whole run, LRU 4 096 / 65 536 / 1 M | 762 180 | 25.9 % / 34.5 % / 35.4 % |
  | duplicates inside one pop | — | 1.7 % |

  The census cost 0.1 core, and the burst read 1 968 vs 2 010 leaves/s. **Price (INFERRED):** an exact cache in
  front of `submit` removes 34.5 % of GPU work at 65 k entries. At a GPU bound that is ×1/(1 − 0.345) ≈ **×1.53
  turns/s** on top of any kernel lever. Two unmeasured discounts apply. Hits are valid only within one net version,
  and the served weights change at each actor sync. And part of the rate is repeats inside one worker's own
  successive searches, which tree reuse could also recover. Kill line P8 ("< 8 %"): **not killed**.
- **GPU headroom for the trainer while serving.** None: the burst runs the GPU at 97.1 % (base) and 99.9 % (fixed)
  with no trainer. See item 2 for the trainer's share.
- **CPU inference on the idle cores (P9-adjacent)** (`cpu_infer.py`, AVX-512 capability, eager, on the captured
  B-64 batches):

  | threads | fp32 | bf16 |
  |---|---|---|
  | 4 | 1 182 ms per batch (54 leaves/s) | 502 ms (128 leaves/s) |
  | 8 | 874 ms (73 leaves/s) | 448 ms (143 leaves/s) |
  | 16 | 859 ms (74 leaves/s) | 454 ms (141 leaves/s) |

  **Price:** +143 leaves/s for 8 cores = +7.9 % of base and +3.9 % of the fixed in-run rate. It also adds a
  second numeric path (CPU bf16) beside the GPU's. Not worth it.
- **CUDA stream / pipeline overlap.** In base the server thread is idle, blocked on the GPU, 78 % of the time. The
  GPU idles ≈ 12 % (88 % utilisation) because each pop's kernel launches start only after the previous pop has
  drained. After the sync fix the server runs 34–57 %, and GPU utilisation is 92–97 % in the cell and 99.9 % in the
  burst. **A second stream or a second server has nothing to take: the GPU is saturated once the sync is gone.** P4's
  two-process test was therefore not run. A copy stream for H2D (P10) is worth at most the remaining 3–8 % GPU idle
  in the cell and ≈ 0 in the burst.
- **Spin-waiters (P7-adjacent, measured as cores).** The retirer's `event.synchronize()` spins 0.7–0.97 core. The
  collate's parallel `pin_memory` leaves 15 OMP threads spinning 2.6–5.2 cores. OMP-1 removes the latter at 0
  leaves/s cost (table above). Neither is a throughput lever on this box because the workers are not CPU-starved.
  Both are free cores for a co-located strix cell or the trainer's CPU side.

## Ranked lever table

All figures are measured on this box, IDLE, leaves/s against 1 808 (cell) or 2 010 (burst), unless marked INFERRED.

| # | lever | measured effect | numerics | size | falsified.md row |
|---|---|---|---|---|---|
| 1 | **Remove the per-forward sync**: `real_mask_from_batch` via `index_fill_` (`gnn_v2.py:118`) | cell **+12 %** alone. With lever 2: +18 % over lever 2 alone (2 857 → 3 378). `launch` 32.4 → 8.9 ms | bit-identical (same mask) | S (one line + a no-sync test, e.g. `set_sync_debug_mode("error")` over one served forward) | none. It is the mechanism behind F-47's "halves never overlap" on this host |
| 2a | **GINE sum via `index_put_(accumulate=True)`** | cell +56 % (with #1: **+87 %**); trainer −10 % wall / −16 % kernels; peak unchanged | **deterministic (0)**, fp32 accumulate, one rounding; mean \|Δv\| vs fp32 0.0100 (base 0.0089) | S | none (F-21 is a custom-kernel/dense-path row) |
| 2b | GINE sum in fp32 (`index_add_` into fp32, compiled → fused Triton) | cell +58 % (with #1: **+87–88 %**); burst with #1 + OMP-1: **+84 % leaves/s, +80 % turns/s, 1 320 vs 696 games/h** | closest to fp32 (mean \|Δv\| 0.0073); near-deterministic compiled (0.003) | S in serving, **not for the trainer: OOM at the 4.5 M cap, +4.3 GiB at 2.25 M, 0 speed** | none |
| 2c | dst-sorted `segment_reduce` (sort once per pop) | cell +46 % | deterministic (0); mean \|Δv\| 0.0073 | M (sort to the Rust collate / wire) | none |
| 2d | `use_deterministic_algorithms(True)` | cell +24 % | deterministic | S but global (touches every op) | none |
| 3 | Exact eval cache (Rust, per net version) | census **35.8 %** steady hit rate → ×1.53 turns/s at a GPU bound (INFERRED, upper bound) | exact (same input, same net) | M | none. F-17/F-19 scope notes exclude it |
| 4 | OMP-1 in the serving process / no parallel pin | 0 leaves/s; **2.6–5.2 cores** freed | none | S | none |
| 5 | Blocking CUDA event on the retirer | 0 leaves/s (INFERRED); **0.7–0.97 core** freed | none | S | none |
| 6 | ELL fixed-slot aggregation | microbench 1.95 ms vs 0.99 for #2b compiled | deterministic | L (wire contract) | none. **Dominated by #2** |
| 7 | CPU bf16 inference on idle cores | +143 leaves/s for 8 cores (+3.9 % after #1+#2) | second numeric path | M | none. Not worth it |
| 8 | `target-cpu=native` | null (+3 % cell, −4 % burst, both in noise) | — | S, host-local (LAW-13) | none |
| 9 | Copy stream / second server stream (P4/P10) | GPU already 92–99.9 % after #1 → ≤ 3–8 % (INFERRED) | bit-identical | M | F-47 (more in flight) |
| 10 | CARD-PERF-4 Rust collate + pinned buffer | the CPU stage is 8.5–9 ms against a 13 ms GPU stage after #1+#2 → 0 today (INFERRED) | must pass a REAL repeat witness (see finding 6) | L | S-PREFUSE (R336(a)) adjacent |

Stacking note: #1 + #2 are measured together (+87–88 % cell, +84 % burst). #3 multiplies turns/s at the same
leaves/s (INFERRED). In a run, the trainer's GPU share (≈ 28 % at run10's cadence and the fixed games/h, INFERRED)
comes out of the same card. #2a is the one variant that is fast in both serving and training and is deterministic.

## Artifacts (the operator's mirror, `box-2026-09-24/profile/`; not tracked, R7)

- **Stacks:** `serve_pyspy.txt` / `.svg` / `.speedscope.json`; `burst_pyspy.txt` / `.svg` / `.speedscope.json`.
  These are Python-level and `--nonblocking`; the burst slice covers 30 s because its first launch crashed at the
  first snapshot on a driver bug, fixed afterwards. `serve_pyspy_analysis.json` and `burst_pyspy_analysis.json`
  come from `analyze_stacks.py`. Its "cuda launch" sub-buckets are a regex artefact (they match `_launch_pop`);
  use the leaf-line table.
- **Cells:** `ab_*.json`, `ab2_*.json`, `ab3_*.json`, `wt_base_compile.json`, `native_base_compile.json`,
  `pyspy_serve_cell.json`, `ab2_syncdbg.json` + `syncdbg.log`; `smoke_*.json` (CONTENDED with a build, not used).
- **Bursts:** `burst_clean.json`, `burst_native.json`, `burst_census.json` (first census, unwindowed),
  `burst_census2.json`, `burst_fixed.json`.
- **Numerics / microbench / trainer / CPU:** `numerics_compare.json`, `agg_micro.json`, `trainer/trainer_*.json`,
  `chain4.out` (trainer OOM text), `clean.out` (CPU inference).
- **Scripts:** `serve_cell.py`, `burst.py`, `patches.py`, `probes_common.py`, `numerics.py`, `agg_micro.py`,
  `trainer_prof.py`, `cpu_infer.py`, `analyze_stacks.py`, `collapsed_to_speedscope.py`, and the chain shell scripts
  with their logs.
- **Not kept (size or derived):** the captured batches (285 MB), the pickled positions and the raw numerics tensors.
