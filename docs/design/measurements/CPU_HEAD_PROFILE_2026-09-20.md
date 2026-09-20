# CPU DEPLOY HEAD PROFILE — leaf-batch histogram and forward ms at 256 sims (R363 §0(5), 2026-09-20)

Ordered by R363 §0(5): numbers before any design. Nothing here is a design, a lever or a proposal;
the section at the end names what the numbers rule out and what they leave open, and stops.

## 1. What was measured, and how

The head is the ladder's `MantisBackend` (`tools/ladder/backends.py`) — the SAME construction the
ladder bot and the follower's cell use: `configs/run8.yaml`'s deploy seam (`deploy.search.kind:
puct`, `eval.gate.deploy_sims: 256`, `selfplay.leaf_batch_size: 8`, `inference.inference_batch_size:
64` / `inference_max_wait_ms: 10`, σ off) on `run8_00030000_6e45edb0.ckpt` (step 30 000, net
`f16bf6c9…`), `LocalInferenceEngine` → `InferenceBatcher` → `InferenceServer` on CPU, eager (the
local engine does not thread `compile_trunk`). The instrument wraps
`LocalInferenceEngine.infer_batch_ls` — ONE call per `select_leaves` batch of the PUCT drive, the
engine boundary: queue wait + collate + forward + the Rust expand are all inside it — recording the
leaf count and wall of every call, and reads `InferenceServer.batch_timing_snapshot()` at the end
(the server's own queue-wait, collate, launch and occupancy instruments, cumulative). Positions: the
head plays ITSELF from `book_v1_s20260625_p4` openings 0–3 (translated onto the origin), plies 4 → 30,
so early, middle and late positions all appear; every position is searched twice from a fresh tree —
the unit cell (256 sims) and the "play" cell (64 sims), the 256 move applied. The harness is a
one-file scratch script (≈ 120 lines: build the backend, monkeypatch `infer_batch_ls`, loop the
openings, dump JSON) kept out of the tree; everything below is derived from its JSON.

Host: AMD Ryzen 7 3700X (8 cores / 16 threads), torch 2.11.0+cpu, Python 3.13, the workstation
otherwise IDLE (no run, no other search). Three thread cells: 8 torch threads (four openings, 101
stones — the ladder shakedown's setting), 16 and 4 (opening 0 only, 26 stones each).

## 2. The numbers — 8 threads, 101 stones

| | unit (256 sims) | play (64 sims) |
|---|---|---|
| wall per STONE, median (mean) | **5 007 ms** (4 790) | **1 233 ms** (1 208) |
| wall per compound turn (two stones), median | **10.0 s** | **2.47 s** |
| engine-boundary calls per stone, median | 41 | 13 |
| calls total / leaves total | 4 178 / 25 198 | 1 281 / 7 715 |
| share of the wall inside the engine call | 99.3 % | 99.4 % |
| leaves actually spent per stone | 256 in 97 of 101 (255, 96, 13 on decided positions) | 64 in 99 of 101 (13, 2) |

**Leaf-batch histogram (unit cell), calls by leaves returned by `select_leaves(8)`:**

| leaves | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| calls | 728 | 133 | 162 | 166 | 142 | 146 | 140 | 2 561 |
| share of calls | 17.4 % | 3.2 % | 3.9 % | 4.0 % | 3.4 % | 3.5 % | 3.4 % | **61.3 %** |
| median ms per call | 22.5 | 38.7 | 52.6 | 81.1 | 101.2 | 119.2 | 134.2 | **158.0** |
| median ms per LEAF | 22.5 | 19.3 | 17.5 | 20.3 | 20.2 | 19.9 | 19.2 | **19.8** |

81.3 % of all leaves ride in full 8-leaf batches. The 728 one-leaf calls (the root evaluation of
every stone plus cold-tree returns) are 17 % of the calls and 2.8 % of the leaves; they cost 16.3 s of
the cell's 481 s (3.4 %). The play cell's histogram has the same shape (8 leaves in 44 % of calls, 1
leaf in 32 %; per-leaf 19.3 / 22.5 ms).

**The per-leaf cost is FLAT in batch size**: 22.5 ms alone, 19.8 ms per leaf in a batch of 8 — a
batch of 8 costs 7.0× a batch of 1. It grows with the position: 8-leaf calls read **16.9 ms/leaf at
4–9 stones, 18.4 at 10–19, 20.5 at 20–29** (n = 639 / 1 031 / 891), and the stone wall follows it
(4 422 → 4 767 → 5 345 ms median). The server's fusion instrument gives the graph sizes behind that:
2 735 nodes and 69 615 fused edges per pop at 5.8 leaves per pop — ≈ 470 nodes and ≈ 12 k edges per
leaf under `gnn_axis_r8`.

**The server's own split (5 459 pops, cumulative):** `queue_wait` mean **7.98 ms** per pop (min 0.49,
max 10.5; total 43.5 s = **7.2 % of the wall** — one submitter of ≤ 8 leaves never reaches the 32-leaf
wake, so nearly every pop runs to the 10 ms deadline); `collate` 0.32 ms per pop (1.7 s total);
`launch` (collate + the CPU forward) mean **103 ms** per pop, 562 s total; `gpu_wait` 0 (CPU,
synchronous); occupancy 5.8 of 64 (fill 9 %); no fusion splits, no cap hits.

**The play cell's argmax agreed with the unit cell's in 58 of 101 positions.** At 64 sims the head
is a different player, not a faster copy; the receipt's `search.preset` label exists for that.

## 3. Thread cells — opening 0, 26 stones each

| torch threads | stone wall median (unit) | 8-leaf call median | ms per leaf at 8 | stone wall median (play) |
|---|---|---|---|---|
| 8 (the 101-stone cell, same opening's 26 stones) | **4 551 ms** | 139.7 ms | 17.5 | 1 131 ms |
| 16 | 5 452 ms (+20 %) | 168.7 ms | 21.1 | 1 354 ms |
| 4 | 5 261 ms (+16 %) | 163.0 ms | 20.4 | 1 332 ms |

(n = 699 eight-leaf calls per row; the 8-thread row is opening 0's 26 stones out of the 101-stone cell.) **The forward does not scale with torch threads**: halving 8 → 4 costs 16 %, doubling 8 → 16 (SMT siblings on 8 cores) costs 20 %, and the one-leaf call reads 21–25 ms in every cell. Whatever bounds the ≈ 20 ms per leaf, it is not the core count — per-op dispatch and memory-bound message passing over ≈ 12 k edges are the candidates, and the split that would tell them apart was not measured here. The 4-thread cell overlapped ≈ 10 s of ladder unit tests on other cores; its medians over 26 stones are not moved by that.

## 4. What the numbers say, and no more

- The CPU deploy head at 256 sims is **bound by the net's forward**: ≈ 20 ms per leaf, linear in
  the leaves, 99 % of the wall, and NOT moved by torch threads (4 ≈ 8 ≈ 16 within 20 %). There is no
  batching economy to collect on this host — a batch of 8 already costs 7× a batch of 1 — so
  `leaf_batch_size`, the pop wake and the fill are not where the time is, and neither is the core
  count. The ladder shakedown's 13 s per compound turn (2026-09-19, strix on the same host) is this
  10.0 s plus contention.
- The one instrument term that is NOT forward is the batcher's deadline: 7.2 % of the wall, from a
  10 ms `inference_max_wait_ms` that a single submitter always runs out. Named, not ordered.
- 64 sims is 4.06× faster per stone (the per-leaf cost is identical) and a different argmax in 43
  of 101 positions. That is what the "play" preset buys and what it is not.
- Per-leaf cost rises 21 % from the opening to the 20-stone middlegame with the graph size; late
  games will be slower still.
- Left open, not measured here: the forward's own split (message passing vs heads vs per-op
  dispatch vs the Python seam — the thread cells say it is not FLOPs-bound, they do not say what it
  is), the net's cost at other widths, a CUDA card (this host has none), and any redesign. R363
  §0(5) ordered numbers before design; these are the numbers.
