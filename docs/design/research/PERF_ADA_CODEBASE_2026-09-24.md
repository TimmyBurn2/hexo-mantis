# PERF-ADA codebase map — serving and training levers on a 4080 SUPER + 9950X host (2026-09-24)
References re-verified against dev fc37f3f2 (post-SLIM merge) 2026-09-25; measurements unchanged (taken on 51eddda5).

Read-only map of branch `perf-ada` at `9839becd`. Nothing under `src/`, `crates/`, `tools/`, `tests/` or
`configs/` was edited or run. Tags: **MEASURED** = a record or today's operator session measured it (cited);
**INFERRED** = derived here from code and measured numbers (arithmetic shown); **UNKNOWN** = nobody has
measured it and the code cannot settle it. "Today" = measured 2026-09-24 (operator session) on the 4080S box.
The torch facts about Inductor are read from the installed torch 2.11.0 Python sources (the local venv is the
`+cpu` build; the `_inductor` Python is the same file set in the `+cu128` build — INFERRED).

Per-pop GPU budget used throughout (B 64, the 4080S, today's torch.profiler shares): total kernel time
**T ≈ 30.5 ms** = bf16 `index_add_` 4 × 4.2 = 16.8 ms (55 %) / triton gather+relu 3.4 ms (11 %) / bf16 GEMM
3.2 ms (10.5 %) / `scatter_reduce` 2.3 ms (7.4 %) / H2D 1.5 ms (4.8 %) / rest 3.4 ms. Cycle 35.3 ms at
92–94 % util. While the device is the bound, cycle ≈ 35.3 × T′/30.5 and leaves/s = 64 / cycle; the floor is the
CPU stage, ≈ 10.4 ms (queue_wait 3.0 + the operator's "launch − gpu_wait" 7.4 — see §1.3 for why that
subtraction is itself an inference).

## Findings first

- **The 4080S box is GPU-bound and the GPU is bound on emulated bf16 atomics.** Inductor never compiles a bf16
  `index_add_`: `torch/_inductor/decomposition.py:227-241` returns `NotImplemented` for any bf16 `index_add`
  outside fbcode (pytorch issue 137425), so the compiled trunk calls ATen's `indexFuncLargeIndex<BFloat16>` on
  EVERY device; on sm_89 that kernel's atomic is a CAS loop, on sm_120 it is native (INFERRED; consistent with
  today's 6.25 vs 1.40 ms microbench). The 4 calls are `src/mantis/model/gine.py:63-64`.
- **bf16 accumulation buys nothing downstream.** `out = agg + (1.0 + self.eps) * x` (`gine.py:69`) adds the
  bf16 `agg` to the fp32 LayerNorm output (autocast promotes `layer_norm` to fp32 on CUDA), so `out` is
  already fp32: an fp32 accumulator changes no downstream dtype, only the sum's precision and its kernel.
  The dummy node's in-degree (~650) is also rounded to bf16 before the division (`gine.py:68`, `gnn_v2.py:79-83`).
- **Top lever: move the GINE sum off bf16 atomics** — fp32 accumulate (Inductor then lowers it to a Triton
  scatter, `lowering.py:4078-4086`, and can fuse the gather/add/relu producer) or a dst-sorted segment sum.
  INFERRED +30 % to +55 % leaves/s alone (1 816 → ≈ 2 360–2 780), which clears R367(e)'s admission line
  (0.85 × 2 466 = 2 096). No production path is sorted today (§2.3).
- **Bolder, same direction: a fixed-slot (ELL) neighbour table.** Every real node has at most 30 axis in-edges
  ((3 axes × 2 signs × 5 depths), `crates/mantis-graph/src/lib.rs:687-735`) + 1 dummy edge, and the edge
  attribute is a function of (slot, neighbour kind) — so aggregation can be a dense gather-sum with no atomics,
  no edge_attr on the wire (609 → ≈ 107 KB/leaf H2D), deterministic order. INFERRED ceiling ≥ 2× (the edge ops
  are ~24 of the 30.5 ms); cheap to falsify offline from a collate dump (§6.1).
- **The pipeline exists; F-47's "halves never overlap" is stale.** A4-4 (`c888c3b7`) split the server into a
  CPU-stage thread and a retire thread with `Semaphore(2)` (`src/mantis/selfplay/inference_server.py:248-281,
  704-717`). But there is exactly ONE CUDA stream in the whole tree (no `torch.cuda.Stream` anywhere under
  `src/`), so H2D never overlaps compute, and serving and the trainer share that stream.
- **Contradiction with the given framing:** `gpu_wait` is not a part of `launch` — they are timed on two
  different threads (`inference_server.py:714-716` vs `832-835`). "launch − gpu_wait = CPU part" holds only if
  `_launch_pop` blocks on the device; the code contains no explicit sync there, yet 32.2 ms on a faster CPU says
  it does block somewhere. WHERE is UNKNOWN and is profiling question P1. The same caveat reaches back to
  PERF-3's 5080 reading that "the CPU stage is the whole lever" (CARD-PERF-4's premise).
- **Pooling and heads are eager, and their segment ops are atomics too:** 5 fp32 `index_add_` + 2 fp32
  `scatter_reduce_(amax)` into only B = 64 rows (`src/mantis/model/gnn.py:65-73`, `gnn_v2.py:50-58`), while
  `node_offsets` (sorted CSR) is already on the device. A `segment_reduce` over `node_offsets` is exact for the
  max half. INFERRED +8 %.
- **The trainer pays the same atomics twice** (forward `index_add_` + the backward of `index_select`, which is an
  `index_add_` of bf16 grads), eager, on the same GPU and stream. INFERRED step ≈ 0.47 s GPU at run10 shape on
  this card → ≈ 15 % of the GPU in-run at 2.4 steps/game; that share grows after a serving-only fix.
- **CPU-side levers are ≈ 0 on this host until the GPU time per pop falls below ≈ 10 ms**: `make build.native`,
  CARD-PERF-4's Rust collate, the retirer's spin, worker/thread counts. CARD-PERF-4 becomes the next lever only
  after a ≥ 2× GPU gain.
- **The strix follower can run on another machine as-is**: it discovers checkpoints from the mirrored event
  stream and `checkpoints/` glob (`tools/strix_follower.py:74, 94-99`); the puller rsyncs the whole run dir
  without `--delete` (`tools/mirror_pull.py:59`). Worth ≈ +3 % wall at run10's 36 000 cadence (39 % of run8's
  wall had a cell or round in flight under run8's promotion-triggered cells).
- **An NN-eval cache across plies is open ground** (F-17/F-19's scope notes exclude eval caching): the tree AND
  its transposition table are cleared every ply (`crates/mantis-selfplay/src/runner/search_drive.rs:504`,
  `crates/mantis-search/src/mcts/mod.rs:157-172`). ESTIMATE 6–20 % of leaves re-evaluate a position the same
  worker evaluated last ply; a one-counter burst falsifies it.

## 1. The serving chain, step by step

### 1.1 Who does what (production: `WorkerPool` → `InferenceServer`; one process with the trainer)

| # | step | thread | GIL | device sync | overlap | where |
|---|---|---|---|---|---|---|
| 1 | select leaves (Gumbel: ≈ 3.5 leaves per round trip, MEASURED `PERF_INVESTIGATION_2026-09-11.md` §2.1), build one axis graph per leaf SERIALLY, submit the round, block | 32 Rust worker threads | never (Rust threads) | none | across workers only | `crates/mantis-selfplay/src/runner/search_drive.rs:264-300`; `crates/mantis-selfplay/src/queues/graph.rs:196` |
| 2 | pop: wake at `saturation_threshold` = min(B/2, n_workers×leaf_batch) = 32, or the 10-ms deadline; take ≤ B | server thread | released (`py.detach`) | none | yes (vs device) | `crates/mantis-bridge/src/inference.rs:372`; `queues/graph.rs:55-62, 78-105` |
| 3 | per-id in-flight bookkeeping (clones `policy_dst_slot`, collects legal coords) | server | **HELD** | none | yes | `inference.rs:374-416` |
| 4 | block-diagonal fuse `GraphWire::from_axis_graphs` (u32 → i64 widening, `edge_index` = `[src ‖ dst]`) | server | released | none | yes | `inference.rs:419-423`; `crates/mantis-selfplay/src/queues/wire.rs:65-147` |
| 5 | `take()` moves arrays into numpy (steps 2–5 are the `queue_wait` timer) | server | held | none | yes | `inference_server.py:683-687`; `graph_collate.py:242-273` |
| 6 | `retirer.slots.acquire()` — depth 2 (NOT inside any timer) | server | released while blocked | waits for retire of pop N−1 | — | `inference_server.py:704` |
| 7 | plan fused parts (1 part at B 64: `fusion_splits` 0.06 %, MEASURED PERF-3) | server | held | none | yes | `inference_server.py:746-752`; `graph_wire_split.py` |
| 8 | collate: structural checks (always) + semantic checks EVERY pop (`collate_check_period=1`, protected set "1-in-1 collate checks"); check 14 captured for the checker thread | server | mostly held (numpy) | none | yes | `pool.py:153`; `graph_collate.py:314-417, 446-608, 617-641` |
| 9 | H2D: 7 arrays, pinned staging (`pin_memory()` for ≥ 1 MB via torch's intra-op pool, `np.copyto` below) + `non_blocking` | server | held/released per op | none explicit | CPU copy overlaps device; the DMA does NOT overlap compute (one stream) | `graph_collate.py:420-442` |
| 10 | `stone_mask_from_batch` (`output_size=` → sync-free) | server | — | none | yes | `graph_collate.py:747-766` |
| 11 | forward launch under `_weights_lock`, `inference_mode`, bf16 autocast: compiled trunk + eager readout/heads | server | held for Python, released inside ops | none explicit (A4-2 removed them) | kernels queue | `inference_server.py:795-811` |
| 12 | `segment_softmax` (fp32, eager), value `.float()` | server | — | none | queue | `inference_server.py:814-815`; `graph_collate.py:729-744` |
| 13 | D2H `non_blocking` into fresh pinned host tensors; `torch.cuda.Event().record()` (steps 7–13 are the `launch` timer) | server | — | none | queue | `inference_server.py:149-155, 818, 825-827` |
| 14 | `event.synchronize()` (spins under the default schedule, GIL released) — the `gpu_wait` timer | retire thread | released | **the one sync** | — | `inference_server.py:832-835` |
| 15 | finiteness gate on host copies, concatenate | retire | held | none | — | `inference_server.py:844-857` |
| 16 | `submit_graph_inference_results`: per graph `assemble_ls_from_gnn_probs` + wake waiters | retire | **HELD** (no `py.detach`) | none | — | `inference.rs:429-505` |
| 17 | hand check 14 to the checker; `verify_edge_geometry` runs GIL-free | checker thread | released (`py.detach`) | none | yes | `inference_server.py:866-867`; `crates/mantis-bridge/src/graph_contract.rs:229` |

The trainer (MainThread of the same process) issues its step onto the same default stream; the eval gate child
and strix cells are separate processes with their own CUDA contexts (time-slicing).

### 1.2 Pipelining — what exists and what does not

- EXISTS (MEASURED at introduction, `PERF_A4_2026-09-11.md` §5: +52 % over serial eager on the 5080): CPU
  stage of pop N+1 (steps 2–13) runs while the device executes pop N; the retire thread dispatches N the moment
  its event fires. `_PIPELINE_DEPTH = 2` (`inference_server.py:250`). F-47's mechanism ("the CPU half and the
  GPU half never overlap") describes the pre-A4-4 tree; CARD-PERF-3 records R360(a)'s annotation of that (GONE at fc37f3f2: the SPENT CARD-PERF-3 row left `CARDS.md` at `b13e324d`; R360(a) itself stands in `RULINGS.md`, the R360 entry and the ANNOTATION under R359's foot).
- DOES NOT EXIST: a second CUDA stream (H2D of N+1 is queued behind N's last kernel → the copy engine and the
  SMs alternate: ≈ 1.5 ms/pop of SM idle, MEASURED share 4.8 %); double-buffered persistent pinned buffers
  (every pop allocates from torch's caching host allocator and memcpys numpy → pinned, `graph_collate.py:433-440`);
  collate in Rust (CARD-PERF-4, OPENED by R366(e), nothing built); stream priorities between trainer and server.

### 1.3 What the timers can and cannot say on this host (contradiction flagged)

`launch` brackets `_launch_pop` on the server thread (`inference_server.py:714-716`); `gpu_wait` brackets the
retire thread's `event.synchronize()` (`832-835`). They are different threads, so "launch 32.2 (of which
gpu_wait 24.8)" is not a containment the code guarantees. Today's cycle 35.3 = queue_wait 3.0 + launch 32.2
+ ≈ 0.1 means `slots.acquire()` never waits — so the server thread is blocked INSIDE `_launch_pop` for most of
the device time, although no line in steps 7–13 syncs explicitly. INFERRED from the arithmetic: pop N's
device work ends ≈ 24.8 ms after its launch returns; the next launch starts 3.0 ms after that return and lasts
32.2 ms, i.e. it returns ≈ 10.4 ms after pop N retires — consistent with an implicit device-coupled wait early
in the launch. Candidates (UNKNOWN which): caching-allocator growth or retry under dynamic shapes
(`cudaMalloc`/`cuMemMap` with `allocator_posture: expandable_segments`, `configs/run10.yaml` line
`allocator_posture`; `cudaFree` on an alloc retry synchronizes), `cudaHostAlloc` growth in the pinned host
allocator, a hidden sync in a fallback op, or GIL waits against the bench's 32 Python feeder threads
(`tools/bench_server.py:107-113`). It changes nothing about the bound (the GPU is 92–94 % busy either way),
but it does mean the 5080 reading "CPU stage 0.34 ms/leaf flat in B" (PERF-3 step 3) may also contain device
wait — the premise under CARD-PERF-4 is weaker than recorded.

## 2. The GNN forward on the GPU

### 2.1 Every op, its dtype under LAW-06 autocast, and whether it is compiled

Serving calls `forward_batch` (`inference_server.py:803`); `GnnNetV2SoftPolicy` inherits it, so run10's aux
head is never evaluated in serving (`gnn_v2.py:233-260`) — consistent with today's step-0 reading.
`torch.compile(self.model.representation, dynamic=True)` compiles the TRUNK ONLY (`inference_server.py:405-413`);
everything in `_readout` after the trunk, the heads, `segment_softmax` and `decode_binned_value` run eager.

| stage | op | dtype | atomics? | compiled? | where |
|---|---|---|---|---|---|
| readout prep | `real = stone_mask.clone(); real[legal_index] = True` | bool | no | no | `gnn_v2.py:115-119, 183` |
| trunk | in-degree `index_add_(0, dst, ones)` over E | **fp32** (x is the fp32 node input) | yes, E adds, 650 into each dummy | yes (fp32 → Triton atomic) | `gnn_v2.py:77-83` |
| trunk | `input_proj` 11→128, `edge_proj` 5→128 over E | bf16 out | no | yes | `gine.py:91-93` / `gnn_v2.py:85-86` |
| per layer ×4 | `LayerNorm` | fp32 (autocast fp32 op) | no | yes | `gnn_v2.py:90` |
| per layer ×4 | `e = lin(projected_edge_attr)` 128→128 over **E rows** | bf16 GEMM | no | yes | `gine.py:59` (the 10.5 % GEMM) |
| per layer ×4 | `xs = x.to(bf16)`; `msg = relu(xs[src] + e)` | bf16 | no | yes (the 11 % triton gather+relu) | `gine.py:60-61` |
| per layer ×4 | `agg.index_add_(0, dst, msg)` | **bf16** | **yes — ATen fallback, CAS on sm_89** | NO (fallback) | `gine.py:63-64` (55 %) |
| per layer ×4 | `agg / divisor.to(bf16)` (dummy degree-normalised) | bf16 | no | yes | `gine.py:67-68` |
| per layer ×4 | `out = agg + (1+eps)·x` → fp32; `nn` MLP (2 Linear) | fp32 in, bf16 GEMMs | no | yes | `gine.py:69-70` |
| trunk out | `final_norm` per layer + `cat` → emb (N, 512) | **fp32** | no | yes | `gnn_v2.py:95-96` |
| readout | `emb.index_select(legal_index)` | fp32 | no | no | `gnn_v2.py:185` |
| readout | `batch_vec` via `repeat_interleave(output_size=)` | i64 | no | no | `gnn.py:36-44` |
| pool mean | `emb*mask`, 2 × `index_add_` (N,512) + 2 × (N,) into **B rows** | fp32 | yes, ~650-way contention per row | no | `gnn.py:63-78` |
| pool max | `masked_fill` (N,512), 2 × `scatter_reduce_(amax)` into B rows, 1 × `index_add_` counts | fp32 | yes (CAS max) | no | `gnn_v2.py:25-59` (the 7.4 %) |
| heads | policy MLP 512→128→1 over Lg; value 1024→32→65 | bf16 GEMMs | no | no | `gnn_v2.py:157-158`; `gnn.py:28-33` |
| value | softmax·support, fp32 | fp32 | no | no | `dist65.py:50-55` |
| output | `segment_softmax`: `scatter_reduce_(amax)` + `scatter_add_` over Lg | fp32 | yes (small) | no | `graph_collate.py:729-744` |

bf16 atomics count: exactly 4 per pop (the operator's count is right); fp32 atomics: 1 in the trunk + 6 in the
readout + 2 in the softmax.

### 2.2 Why Inductor does not save the aggregation, and what would change it

- `torch/_inductor/decomposition.py:227-241`: bf16 `index_add` → `NotImplemented` → ATen kernel. This holds on
  the 5080 too; there the atomic is native (`torch/_inductor/utils.py:3215-3222` gates bf16 atomic support on
  compute capability ≥ 9.0 for `index_put`/scatter lowering). INFERRED: the 4.5× microbench ratio is the CAS
  emulation, and the neighbouring bf16 columns of one row share a 32-bit word, so adjacent threads contend on
  the same CAS target by construction.
- fp32 accumulator: `index_put` with `accumulate` on fp32 is NOT a fallback (`lowering.py:4078-4086`; the
  lowering even casts `values` to the accumulator dtype), so a compiled trunk would emit a Triton scatter with
  native fp32 atomics and can fuse the gather/add/relu producer into it (no `msg` [E,128] materialisation:
  256 MB write + 256 MB read per layer at B 64). UNKNOWN until the generated code is read (P3).
- `segment_reduce` is a declared Inductor fallback (`lowering.py:3006-3007`): it runs as an ATen kernel inside
  the compiled graph, no graph break (INFERRED).
- Eager (the trainer, the eval child, the deploy head): `index_add_` requires matching dtypes, so fp32 accumulate
  needs `msg.float()` — an [E,128] fp32 transient (512 MB at B 64; ≈ 2.3 GB at the trainer's 4.5 M-edge
  microbatch cap). That is the allocation class of CARD-RUN5-GPU-OOM (the 8.94 GiB request pinned by
  `tests/model/test_gine_gather_regime.py`); the trainer needs compile, a custom op, or the ELL form.

### 2.3 Edge order on the wire today

- Builder (`crates/mantis-graph/src/lib.rs:687-748`): for each real node i in node order (stones sorted by
  (q, r), then legal cells sorted), for each axis, each sign, each depth: push (i → j) AND (j → i); dedup by
  (src, axis, sign, d) keeping the FIRST (`:738`); then the dummy pairs (dummy → i, i → dummy) for every real i
  (`:740-748`). Neither src-sorted nor dst-sorted; pairs interleave.
- Fuse (`crates/mantis-selfplay/src/queues/wire.rs:65-147`): per-graph blocks, contiguous, globally offset,
  `[src(E) ‖ dst(E)]` int64, plus `edge_offsets`/`node_offsets` CSR over GRAPHS only. So: graph-blocked yes,
  per-node CSR no.
- The order is pinned byte-exact by `crates/mantis-graph/tests/graph_parity.rs` (oracle fixtures
  `tests/fixtures/graph_parity/`; re-verified fc37f3f2: the goldens `inputs.bin` + `raw/` stay, the source corpus
  `wpa_positions.json` was deleted at `b27a49d7`) and `crates/mantis-selfplay/tests/queue_fuse_pin.rs`; `docs/contracts/graph_wire.md`
  states membership/ordering rules for gathers and per-graph metadata, not edge order.

### 2.4 What dst-sorted edges + CSR row pointers would cost — three placements

| placement | cost | who pays | contract impact | INFERRED |
|---|---|---|---|---|
| (a) builder: counting sort by dst per graph, emit `row_ptr` | O(E + N) per graph ≈ 16 k × 28 B moved, ≈ 20–40 µs vs the 0.85 ms build floor (`tools/bench_floors.toml` `axis_graph_build/per_position`, 9900X) → +3–5 % worker CPU | 32 worker threads (90 % blocked, MEASURED PERF_INVESTIGATION §2.4) | breaks `graph_parity.rs` goldens (oracle order); D6 augmentation (`crates/mantis-selfplay/src/replay/sym.rs`) must preserve it; wire contract version bump | off the critical path |
| (b) fuse: sort inside `from_axis_graphs` (`wire.rs:65`) | same work, on the server thread inside `py.detach` (inside `queue_wait`) | server thread, ≈ 64 × 30 µs ≈ 2 ms/pop | builder goldens untouched; `queue_fuse_pin.rs` + wire contract move; ONE place for serving AND the replay sample path (the fuse serves both, `wire.rs:61-62`, the `from_axis_graphs` doc) | fine while GPU-bound (CPU floor ≈ 10.4 → 12.4 ms < T′) |
| (c) GPU, once per pop: `perm = argsort(dst, stable)`, permute `edge_index`/`edge_attr` once, `row_ptr` by bincount+cumsum; each of the 4 layers then segment-sums | ≈ 0.3–0.5 ms/pop for 1 M keys + 36 MB permute | device | no wire change at all | smallest blast radius, deterministic |

## 3. The trainer on the same GPU

- **Shape** (`configs/run10.yaml` `train:`): `batch_size 256`, `microbatch_caps {max_edges 4 500 000, max_nodes
  170 000}` → one microbatch at ≈ 16 k edges/graph (4.1 M edges), sometimes two; `augment: true` (D6 in the Rust
  sample, GIL-free: `crates/mantis-bridge/src/hexg.rs:155`); collate `semantic="full"` every part
  (`src/mantis/train/coordinator/dispatch.py:202-214`).
- **Dtype/path**: eager (no compile on the learner; the compile is server-private by design,
  `inference_server.py:307-309`), bf16 autocast (`src/mantis/train/trainer/core.py:388-389`), `forward_batch_heads`
  (run10 evaluates the aux head, `gnn_v2.py:243-260`). Forward: the same 4 bf16 `index_add_`; backward: each
  `index_select` gather's backward is an `index_add_` of bf16 grads (4 more CAS-atomic calls), plus the
  `index_add_` backward = a gather. Syncs: `tail_mass.tolist()` then 5–7 `.item()` per microbatch
  (`core.py:385-387, 430-444`); MEASURED 31 syncs/step on the 5080 (`MEASUREMENT_STARTPATH_2026-09-11.md` §C).
- **Cadence**: `training_steps_per_game 2.4`, `max_train_burst 8`, `actor_sync_cadence_steps 2` (weights into the
  server's own copy under `_weights_lock`, `pool.py:77-85`, `inference_server.py:449-461`).
- **Records**: the 5080 step at run6 shape = 0.244 s wall, 0.156 s GPU, `index_add_` 34 ms (22 %),
  `index_select` backward 17 ms (MEASURED, STARTPATH §C). F-44's "92 % asleep" is run6's regime only
  (falsified.md F-44 regime annotation, R365(e)). Contention in-run: rounds cost the trainer 24–32 % and a strix
  cell 60–64 % in steps/h (MEASURED, `EVAL_COST_2026-09-19.md` §(iv)); serving beside a cell 27–54 % of alone,
  the device stage unchanged — the CPU is what a cell takes (PERF-3 step 3).
- **INFERRED step on the 4080S at run10 shape**: atomics (34 + 17 ms) × 4.46 (today's bf16 ratio) + the rest
  (105 ms) × 1.30 (960/736 GB/s) = 364 ms, × 1.29 (16.1 k vs 12.5 k edges/graph) ≈ **0.47 s GPU/step** (± 50 %;
  P5 measures it).
- **INFERRED in-run share**: leaves/game ≈ 88 plies × 129 ≈ 11 350 (PERF-3 88 plies; 127–130 leaves/position,
  PERF_INVESTIGATION §2.1). Trainer GPU share s solves s = 2.4 × L(1−s) × 0.47 / 11 350 with L = 1 816 →
  s = 0.153: in-run serving ≈ 1 540 leaves/s, ≈ 490 games/h, ≈ 1 170 steps/h. After a serving-only fix
  (L ≈ 2 440): s ≈ 0.195, in-run ≈ 1 960. Fixing the trainer's atomics too (step → ≈ 0.30 s): s ≈ 0.134, in-run
  ≈ 2 110 (+7.6 % on top).
- **What the aggregation fix does for it**: the forward half transfers directly (4 calls); the backward half does
  NOT unless the gather's backward also leaves bf16 `index_add_` — a custom autograd function, a compiled
  training trunk, or the ELL form with a reverse-slot table (§6.1) whose backward is also a gather.

## 4. CPU side on a 16-core Zen 5

- **`make build.native`** (`Makefile:15-16`, env-only `RUSTFLAGS=-C target-cpu=native`, R2/LAW-13; release is
  `lto = "fat"`, `codegen-units = 1`, `panic = "unwind"`, `Cargo.toml:40-43`). Without it the extension targets
  baseline x86-64 (SSE2). Vectorisable Rust loops: the fuse's widening copies (`wire.rs:105-140`), check 14's
  per-edge recompute (`graph_contract.rs`), the builder's feature fill; the builder's walk and hash probes are
  branchy and gain little (INFERRED). None is on this host's critical path → ≈ 0 leaves/s today. The 28 bench
  floors (re-verified fc37f3f2: `grep -c '^\[floor\.' tools/bench_floors.toml` counts 23, as at 9839becd) are toolchain/host-attested, so a native build is never a floor-comparable build.
- **Threads**: 32 Rust workers (≈ 0.9 ms CPU/leaf, ≈ 1.6 cores at 1 816 leaves/s, INFERRED from MEASURED 0.89 ms,
  PERF_INVESTIGATION §3); server, retire (spins in `event.synchronize()`, 77 % busy on the 5080, PERF-A4 §5),
  checker (52 %), trainer, torch intra-op pool (no `set_num_threads` anywhere in `src/`; the pool spins after each
  `pin_memory()` region, ≈ 4–6 cores on the 12-core box, PERF-A4 §4/§8). Against a 30.7-CPU cgroup quota the
  risk is CFS throttling during a strix cell, not steady-state (P7).
- **F-46**: repaired at `13562ce1` (`graph_contract.rs:229`, `py.detach` over the borrowed slices), +17 %
  MEASURED (falsified.md F-46 repair annotation); `checker_thread` is armed in `configs/run10.yaml`. Nothing
  left there.
- **CARD-PERF-4** (Rust collate + pinned host buffer + one launch per pop): OPENED by R366(e), design only after a
  ruling, nothing built; falsifier `tools/bench_server.py` ≥ 1.4× on the same machine with the bf16 witness at 0.
  On a GPU-bound host its gain is 0 until T′ < ≈ 10 ms; R336(a) refuted the worker-side pre-fuse (S-PREFUSE,
  falsified.md annotation) — a Rust collate must not re-propose that shape.
- **Event spin**: `torch.cuda.Event()` at `inference_server.py:826`; `torch.cuda.Event(blocking=True)` makes the
  retire thread sleep instead of spin. Frees ≈ 0.8 core, adds µs latency; 0 leaves/s here unless P7 shows
  throttling.

## 5. Other levers in the code

- **CUDA graphs**: `inference.fused_graph_caps {1 373 143 edges, 56 645 nodes}` (run10) are per-PART memory
  caps (`inference_server.py:394-399`), and 80 % of pops sit in [512 k, 1 M) edges (MEASURED PERF-3). Padding to
  the cap = +37–170 % edge work; graphs save CPU launch time (106 launches, PERF-A4 §5), which is not the bound
  → negative on this host. Bucketed shapes + `reduce-overhead` only matter once the CPU binds.
- **H2D**: 39 MB/pop at 26.8 GB/s ≈ 1.46 ms (today), serial with compute on the one stream. Two cheap forms:
  (i) a copy stream + `wait_event` so pop N+1's DMA runs under pop N's kernels (+≈ 4.5 %, bit-identical);
  (ii) shrink the wire: `edge_index` int32 (16 → 8 B/edge), `edge_attr` as a u8 code (20 → 1 B/edge) → ≈ 12 MB/pop.
- **D2H**: probs + values ≈ 80 KB/pop into fresh pinned tensors; one event sync per pop on the retirer; no lever.
- **Pinned reuse**: the caching host allocator reuses blocks; the per-pop numpy → pinned memcpy (≈ 36 MB) is CPU
  work off the bound here.
- **Eval gate child**: its own `InferenceServer`, eager, own CUDA context (PERF-A4 §7); at run10's
  `eval_interval 36 000` a round every ≈ 31 h (INFERRED from ≈ 1 170 steps/h), costing 24–32 % of the trainer
  while it runs (MEASURED) → ≈ 1–3 % of wall.
- **Strix follower off-box**: triggers come from `events_<run_id>_seg*.jsonl` in `--run-dir/logs`
  (`strix_follower.py:67-91`), the checkpoint from the event's `path` if it exists locally, else
  `checkpoints/{run_id}_{step:08d}_*.ckpt` (`:94-99`) — both present in a mirror (`mirror_pull.py:59`, no
  `--delete`, so sidecars written into the mirror survive). Caveat: `regime()` (`:106-119`) labels a cell
  CONTENDED when any mirrored heartbeat is < 300 s old, which on another machine reads the box's heartbeat → a
  mislabel, not a failure (the label is a record field; a flag or a local-heartbeat rule fixes it). A cell's 8
  CPU-torch strix drivers need ≈ 20 cores (PERF-3 step 3: ≈ 2.5 each) → ≈ 2.5× longer wall on an 8-core desktop,
  0 box cost. Value at run10's 36 000 cadence: one cell (1.2–1.6 h, MEASURED 4 056–5 682 s) per ≈ 31 h at 0.38× →
  ≈ 0.9 h / 31 h ≈ **+3 % wall**.
- **Deploy/ladder CPU head** (`CPU_HEAD_PROFILE_2026-09-20.md`, 3700X): ≈ 20 ms per leaf, flat in batch AND in
  threads (MEASURED). INFERRED cause: the CPU forward runs fp32 (autocast only `enabled=on_cuda`,
  `inference_server.py:799`), and ATen's CPU `index_add_` iterates edges serially — which is exactly "flat in
  threads". A dense ELL/CSR sum parallelises over nodes; P9 confirms or kills.

## 6. Outside the box

Each: mechanism in the code, price, cheapest falsifier.

### 6.1 Per-node aggregation: the fixed-slot (ELL) neighbour table
- **Mechanism**: in-edges of a real node come from ≤ 30 distinct (axis, sign, depth) slots — one cell per
  displacement — plus 1 dummy edge; the graph is symmetric by construction (every pair is pushed both ways,
  `lib.rs:709-716`; INFERRED symmetric after dedup since both endpoints' walks find each other). The attribute
  of edge (slot, neighbour) is (axis one-hot, signed distance, neighbour's player) → ≤ 91 codes, which is the
  A4-5 codebook. So `agg[j] = Σ_s relu(x[nbr[j,s]] + T_layer[slot s, kind(nbr)]) · valid[j,s]`: a gather into
  [N, 31, 128] reduced over dim 1 — no atomics, fixed summation order (deterministic), fp32 accumulation free,
  no `edge_attr`, no [E,128] GEMM. The backward of the gather is the reverse-slot gather (symmetry), also
  atomic-free.
- **Price (INFERRED)**: per layer the gather reads N × 31 × 256 B ≈ 331 MB at B 64 with x (10.7 MB bf16) L2-resident
  on a 64 MB-L2 card; the edge work (≈ 24 of 30.5 ms: index_add 16.8 + gather/relu 3.4 + GEMM 3.2 + edge_proj
  share) could fall to ≈ 2–4 ms → T′ ≈ 9–11 ms, i.e. the CPU floor binds: **≥ 2×**, and the H2D falls
  609 → ≈ 107 KB/leaf (nbr int32 78 KB + x 28.6 KB) which also shrinks the CPU stage. Padding waste: 31 slots vs
  ≈ 24.7 used (≈ 25 %).
- **Cost**: L — wire contract v2 (or a derived table computed at fuse), builder/oracle goldens, a new arch
  kind or a ruling (R309's rule as A4-5 applied it: numerics-changing forward = new kind or nothing), trainer
  path, D6 augmentation. Checkpoints load unchanged (same parameters, same function up to summation order).
- **Cheap falsifier (S, off-box code, no tree edit)**: take a collate dump (`src/mantis/selfplay/collate_dump.py`
  writes the wire) or a `bench_server` pop, derive `nbr`/`slot` in numpy from `edge_index`/`edge_attr`, run
  both forms through `torch.compile` on the 4080S, compare outputs (should agree within the bf16 null) and time
  them. Kill if the ELL layer is not ≥ 3× faster than the current layer.

### 6.2 Remove the dummy hub from the edge list
- 2 × n_real of each graph's ≈ 16.1 k edges are dummy edges (`lib.rs:740-748`, ≈ 8 %); the ≈ 650 edges INTO the
  dummy all hit ONE row — the worst CAS contention in the batch. Both directions have zero attributes, so
  dummy-in = per-graph mean over real nodes of relu(x_i + c_layer) (the divisor already makes it a mean) and
  dummy-out = one per-graph vector broadcast to every real node: two segment ops over `node_offsets`.
- **Price**: ≥ 8 % of the edge work (≈ 1.9 ms/pop, +6 %) plus the contention relief, UNKNOWN (P3 isolates it).
  Numerics: sum order only. M, touches the same seams as 6.1 but not the wire.

### 6.3 NN-eval cache across plies (per worker)
- The tree and TT are rebuilt every ply (`search_drive.rs:504`, `mcts/mod.rs:157-172`); the next root and part of
  the chosen child's subtree were evaluated one ply earlier. Sequential halving gives the chosen candidate ≈ 19/64
  (quick) or ≈ 86/320 (full) visits (INFERRED from PERF_INVESTIGATION §2.1's round widths) → ≈ 27–30 % of last
  ply's leaves lie under the new root. ESTIMATE hit rate 10–20 %; with the cache keyed by `model_version` (actor
  sync every 2 steps ≈ every 5–6 s vs ≈ 2.3 s per worker-ply, INFERRED) ≈ 40 % of ply pairs straddle a swap →
  ≈ 6–12 %.
- **Price**: positions/h × 1/(1−h) → +6–25 % with no GPU change. S–M (Rust, a bounded FxHashMap per worker keyed
  by zobrist + side/moves-remaining + model_version, cleared per game). Staleness-by-one would be a new actor-lag
  regime → a ruling + LAW-18 fire-rate counter.
- **Falsifier (S)**: a counter burst — for each evaluated leaf, whether its hash was evaluated by the same worker
  in the previous ply (and the previous version); kill below 8 %. Not in falsified.md: F-17/F-19's scope notes
  exclude eval caching and transposition reuse explicitly.

### 6.4 Is the GPU really saturated? Two servers / two streams
- nvidia-smi's 92–94 % is "a kernel was running", not SM throughput; the pooling kernels write into 64 rows and
  CAS kernels serialise. If SMs idle inside "busy", a second stream (two server loops, or trainer on a low-priority
  stream) would fill them.
- **Falsifier (S)**: two `bench_server.py` processes on the card at once; aggregate ≥ 1.1 × 1 816 ⇒ headroom exists
  (then a second server stream is an M lever); ≈ 1 816 ⇒ saturated, kill.

### 6.5 CPU inference on the idle Zen 5 cores
- MEASURED on the 3700X: the CPU forward is ≈ 40 leaves/s whole-machine (PERF-3 step 2: 382 ms per B 16 pop,
  1 589 ms per B 64). ×2 cores × ≈ 1.3 AVX-512 → ≈ 100–150 leaves/s ≈ +5–8 %, while taking the cores the CPU stage
  and workers need and serving fp32 numerics beside bf16 ones. **Low**; unless 6.1 also fixes the CPU path.
- **Falsifier (S)**: `tools/bench_server.py --device cpu --no-compile --batch-sizes 16` on the 9950X; kill below
  200 leaves/s.

### 6.6 Moving work off the box to the operator's desktop (3700X + 8 GB sm_86)
- **Strix follower**: works as-is (§5), +3 % at run10's cadence; S (operational).
- **Trainer**: frees ≈ 15 % of the box GPU (§3) → serving ceiling +18 % (1/0.847); the 3070 has the same CAS
  atomics and 448 GB/s, so its step ≈ 2× the 4080S (INFERRED) → ≈ 30 % duty, it keeps up. But the learner, the
  ring, ActorSync (every 2 steps), resume bundles, LAW-16 lifecycle and the actor-lag abort are one-process
  contracts: **L, high risk, not before the aggregation fix** (which is worth more for less). F-44's regime note
  applies: its "trainer is asleep" is run6's, not run10's.

### 6.7 Quantised eval (F-21's ordered fallback) and a smaller graph
- int8/fp8 touches the GEMMs (10.5 %) and halves x's gather width; the atomics stay → ≤ +10 %, heavy numerics.
  **Low.** F-21's row is scoped to the dense path; its fallback order (compile → smaller net → quantised) is
  advice, not a verdict here.
- Radius 8 → 6 is ≈ −22 % edges (run6's r6 graphs: 12.5 k edges vs 16.1 k) but it is an identity-key change
  (`identity.encoding`, LAW-11) and a different net — a run11 design input, not a serving lever.

## 7. Profiling hypotheses (for the box profiler agent)

| # | question | how | confirms / kills |
|---|---|---|---|
| P1 | Where does the server thread block inside `_launch_pop`? | `torch.cuda.set_sync_debug_mode("warn")` in a `bench_server` cell; py-spy `--native --threads` on `inference-server`; look for `cudaStreamSynchronize`/`cudaEventSynchronize`/`cudaMalloc`/`cudaFree`/`cuMemMap`/`cudaHostAlloc`/`take_gil` frames under `_launch_pop` | a device-wait frame ≈ 20 ms/pop ⇒ the launch timer holds device time (and PERF-3's 5080 CPU-stage reading needs the same check); pure CPU frames ⇒ the CPU stage is genuinely ≈ 32 ms and the GPU-bound reading is wrong |
| P2 | Allocator churn under dynamic shapes | `torch.cuda.memory_stats()` deltas over a 200-s cell: `num_alloc_retries`, `num_device_alloc/free`; host allocator stats; run with and without `expandable_segments` | retries > 0 or device allocs per pop ⇒ a hidden sync source; fix = pre-reserve / stable sizes |
| P3 | Attribution of the 4.2 ms bf16 call | microbench on REAL pops: bf16 `index_add_` on (i) the real `edge_index`, (ii) dummy edges removed, (iii) dst-sorted, (iv) fp32 accumulator eager, (v) fp32 under `torch.compile` with `TORCH_LOGS=output_code` (is the gather/relu fused into the Triton scatter?), (vi) `segment_reduce` on sorted | ranks §6.2 vs §2.4 vs fp32; (v) ≤ 1.5 ms ⇒ fp32+compile is the S lever |
| P4 | SM saturation | two concurrent `bench_server` processes (§6.4); if counters are permitted, `ncu` on `indexFuncLargeIndex` for achieved occupancy | aggregate ≈ 1 816 ⇒ saturated |
| P5 | The trainer step on this card at run10 shape | STARTPATH §C's 10-step torch.profiler at batch 256 off a run8/run10 ring; peak memory with an fp32 accumulator | GPU/step vs the INFERRED 0.47 s; `index_add_` fwd/bwd shares; whether 2.3 GB fp32 transients fit beside serving |
| P6 | Trainer vs serving in-run | 20-min twin: trainer kernel time / wall, served leaves/s in-run vs the bench | s vs the INFERRED 0.15; confirms the in-run discount |
| P7 | cgroup throttling | `cpu.stat` `nr_throttled`/`throttled_usec` during a bench cell and during a strix cell; retire-thread CPU | throttling ⇒ blocking events + intra-op thread cap become levers |
| P8 | NN-eval cache hit rate | a counter build (not landed): per evaluated leaf, hash seen in the same worker's previous ply, with/without version equality | < 8 % ⇒ kill §6.3 |
| P9 | CPU forward anatomy (deploy head) | torch.profiler on the CPU `forward_batch` of 1 and 8 leaves on the 9950X at 1/8/16 threads | `index_add_` dominant and serial ⇒ ELL/CSR fixes the ladder head too |
| P10 | The H2D bubble | torch.profiler trace: gap between pop N's last kernel and N+1's first | ≈ 1.5 ms ⇒ a copy stream buys ≈ 4.5 % |
| P11 | The eager readout | kernel times of the 2 `scatter_reduce_(amax)`, the 5 fp32 `index_add_`, `masked_fill`, `emb*mask` | sizes lever C |
| P12 | The weight dependence | the same profile on random-init vs trained (1 245 vs 1 816 today) | the delta lives in `index_add_` ⇒ it is the CAS path; elsewhere ⇒ a second mechanism |

## 8. Ranked lever table

Gains are leaves/s on the 4080S bench cell (alone, B 64) unless stated; arithmetic from the per-pop budget above;
all INFERRED unless marked. "Admission" = 2 096.

| # | lever | where (file:line) | expected gain on this host (arithmetic) | numerics | size | falsified.md | prerequisites |
|---|---|---|---|---|---|---|---|
| 1 | GINE sum in fp32 (accumulator fp32; compiled trunk → Triton fp32 scatter, eager → `msg.float()`) | `src/mantis/model/gine.py:63-64, 67-69` | 4 × (4.2 − 2.43) = 7.1 ms → T′ 23.4 → cycle 27.1 → **2 360 (+30 %)**; with the in-model proportional 4.2/6.25 scaling (1.63 ms/call) → 2 735 (+51 %); if compile fuses gather+relu in, up to ≈ +80 % (UNKNOWN, P3) | not bit-identical; strictly more accurate (bf16 max err 0.375, today); atomics order still nondeterministic; LAW-06's autocast dtype untouched, but `test_gine_gather_regime.py` (the MA-4 "agg dtype is the gather dtype" assertion) and `test_gine_bf16_drift.py` / `test_bf16_parity_nulldist.py` bands move; A4-3's precedent (a key, parity inside the device's own null) or R309's (new arch kind) — a ruling | S (serving via compile) / M (trainer, memory) | none; F-21 is dense-path-scoped | ruling (plan §2.2); P3, P5; a key if device-conditional (the 5080 reads fp32 slower eager: 2.30 vs 1.40) — R1/LAW-08, gate 13's schema doc |
| 2 | dst-sorted CSR segment sum, sort once per pop on the GPU | `gine.py:63-64`; `gnn_v2.py:66-96` (permute once before the layer loop) | 4 × (4.2 − 2.11) − 0.5 sort = 7.9 ms → cycle 26.2 → **2 440 (+35 %)**; proportional → 2 780 (+53 %) | deterministic order (a determinism gain: probe can read exactly 0); fp32 input needed for fp32 sum (same transient as #1 eager) | M | none | ruling; `test_gine_gather_regime.py` "exactly one `index_add_`" pin moves; builder/wire untouched |
| 3 | ELL fixed-slot aggregation (+ codebook implied) | `crates/mantis-graph/src/lib.rs:687-748`, `wire.rs:65-147`, `gine.py:41-70`, collate | edge work ≈ 24 → ≈ 2–4 ms → T′ ≈ 9–11 ms → CPU-floor bound: **≥ 2× (≈ 3 700+)**, H2D 609 → 107 KB/leaf | deterministic; sum order only; same weights | L | none (S-PREFUSE is a different shape) | offline microbench (§6.1); contract v2; new arch kind or ruling; trainer backward via reverse slots |
| 4 | readout pooling via `segment_reduce` on `node_offsets` (+ compile the readout) | `gnn.py:47-78`, `gnn_v2.py:25-59` | ≈ 2.3 ms (+ an UNKNOWN share of the 5 fp32 `index_add_`) → cycle ≈ 32.6 → **≈ 1 960 (+8 %)**, stacks with 1/2 | max half bit-identical; mean half fp32 reorder (~1e-7); `forward_single` untouched | S | none | P11 |
| 5 | H2D off the compute stream (copy stream + `wait_event`) or wire shrink (int32 index, u8 edge code) | `graph_collate.py:425-442`; `wire.rs:25-27` | 1.46 ms/pop → **+4.5 %** | bit-identical (stream form); dtype form exact for indices | S (stream) / M (wire) | none | `test_collate_h2d_pinned_parity.py` extended to the stream |
| 6 | Edge codebook table (A4-5) | `gine.py:59`, `gnn_v2.py:86` | GEMM 3.2 ms + `projected_edge_attr` traffic ≈ 3.5 ms → **+11 %** (subsumed by #3) | bit-exact on sm_120, 1.6e-2–6.3e-2 off on sm_86 (MEASURED, PERF-A4 §6); sm_89 UNKNOWN | M | none (PERF-A4 §6 deferred it) | a parity probe on sm_89 |
| 7 | Dummy hub out of the edge list | `lib.rs:740-748`; `gnn_v2.py:77-83` | ≥ 8 % of edge work ≈ 1.9 ms → **≥ +6 %**, contention relief UNKNOWN | sum order only | M | none | P3 (ii) |
| 8 | NN-eval cache across plies | `search_drive.rs:504`; `mcts/mod.rs:157-172`; `selection.rs:305-320` | h ≈ 6–20 % → **+6–25 % positions/h** at equal leaves/s | cached value from another batch (inside the bf16 null); staleness regime if version-tolerant | S–M | none (F-17/F-19 scope notes exclude it) | P8; ruling on staleness; LAW-18 counter |
| 9 | Trainer atomics (custom fwd/bwd or compiled training trunk) | `src/mantis/train/trainer/core.py:383-446`; `gine.py` | in-run: s 0.195 → 0.134 after #1/#2 → **+7.6 % in-run** | as #1/#2 | M–L | F-44 (run6 regime only, R365(e)) | P5, P6 |
| 10 | Strix follower on the desktop against the mirror | `tools/strix_follower.py:67-119`; `tools/mirror_pull.py:59` | one 1.2–1.6 h cell per ≈ 31 h at 0.38× → **≈ +3 % wall** at run10's cadence | none | S (operational; fix the regime label) | none | operator placement |
| 11 | Second server stream / two servers | `inference_server.py:634-729` | UNKNOWN; P4 decides | bit-identical per pop | M | F-47 (more in flight — measured null here too: 48×8 and 32×16 change nothing today) | P4 |
| 12 | Blocking CUDA event on the retirer | `inference_server.py:826` | 0 leaves/s; ≈ 0.8 core back | none | S | none | P7 |
| 13 | `make build.native` | `Makefile:15-16` | ≈ 0 while GPU-bound | bit-identical Rust (float paths unaffected unless FMA contraction — Rust does not contract by default) | S | none | host-local only (LAW-13) |
| 14 | CARD-PERF-4 Rust collate + persistent pinned buffer | `graph_collate.py:314-442`; `inference.rs:363-425` | 0 until T′ < ≈ 10 ms; after #1+#4+#5 (T′ ≈ 19 ms) still 0; after #3 it is THE next lever | must read the bf16 witness at 0 | L | S-PREFUSE (R336(a)) is adjacent, not the same | a ruling (CARD-PERF-4) |
| 15 | CUDA graphs (static shapes by padding to caps) | `inference_server.py:394-413` | negative: +37–170 % edge work to save CPU launches that do not bind | padding must be exact-zero | M | none | — |
| 16 | CPU inference on Zen 5 | `inference_local.py` / `bench_server.py --device cpu` | ≈ 100–150 leaves/s → **+5–8 %**, contends with the CPU stage | fp32 beside bf16 | M | none | §6.5 falsifier |
| 17 | Trainer to the desktop | the one-process run (`run.py`, ActorSync) | ceiling +18 % | none | L, high risk | F-44 regime note | after #1–#3 |
| 18 | Quantised eval | `gine.py`, heads | ≤ +10 % | heavy | M | F-21's fallback order (advisory) | — |

Stacked (INFERRED, device-bound): #1 or #2 (−7.9) + #4 (−2.3) + #5 (−1.5) + #6 (−3.5) → T′ ≈ 15 ms → cycle
≈ 17.5 ms → ≈ 3 650 leaves/s, at which point the CPU stage (≈ 10.4 ms) is within 2× and CARD-PERF-4 becomes live.
#3 reaches the same place in one lever with fewer seams touched on the device side and more on the wire.

## 9. Contradictions with the facts as given

1. **"launch 32.2 (of which gpu_wait 24.8)"** — not a containment by construction (two threads, §1.3). The
   subtraction is informative only because `_launch_pop` evidently blocks on the device; the code has no explicit
   sync there, so the blocking site is UNKNOWN (P1).
2. **F-47's "halves never overlap"** — superseded at HEAD by A4-4's two-thread pipeline; what does not overlap is
   the H2D DMA with compute (one stream).
3. **"4 bf16 `index_add_` calls per batch"** — correct, but the forward also runs 1 fp32 `index_add_` in the trunk
   and 6 fp32 atomics-based reductions in the eager readout plus 2 in `segment_softmax`; the `scatter_reduce`
   7.4 % is eager, not compiled.
4. **The 5080 comparison** — the 5080 also ran the ATen fallback (Inductor refuses bf16 `index_add` on every
   device); the difference is the atomic, not the code path. And PERF-3's "the CPU stage is the whole lever" on the
   5080 read `launch` as pure CPU — the same timer ambiguity applies there.
5. **Everything else checks**: 609 KB/leaf = 16 B/edge (`edge_index` i64) + 20 B/edge (`edge_attr`) × 15.7 k +
   28.6 KB `x` ≈ 599 KB + offsets/gathers; the pinned `non_blocking` path is `graph_collate.py:425-442`; run10's aux
   head is not evaluated in serving (`inference_server.py:803` → `GnnNetV2.forward_batch`).
