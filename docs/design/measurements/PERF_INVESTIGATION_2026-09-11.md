<!-- Filed from the performance investigation's REPORT.md (two read-only agents, 2026-09-11). Its reproduction scripts and raw harvests live in the dispatcher's scratchpad (perf_investigation/) and on the box under /workspace/oc7/pi and /workspace/runs/pi_*; the SUMMARY is the one-page form at the foot. Every number here is sourced to a script or log named in §8. -->

# Where the wall clock of a run6 training block goes — performance investigation (2026-09-11)

Scope: the run6 regime (`configs/run6.yaml`; profiled trees `4c4a83d4` and `7a97fa80`, HEAD `2302aa40`
— `git log 7a97fa80..HEAD -- src/ crates/ configs/` touches only `mirror_receipts.py` and
`bundle_receipts.py`, nothing on a hot path). Nothing under `src/`, `crates/`, `tools/`, `tests/`,
`configs/` was edited; every box artefact is a throwaway minted outside `configs/`. Every number
below names its instrument, window and denominator; a number nobody measured is marked ESTIMATE.
LAW-05: falsified.md rows F-17/F-19 (incremental legal-set maintenance on descent paths), F-21 (a
custom CUDA kernel for the DENSE path — its own text says the kernel "solves GNN variable-size
ragged batching", which is exactly the graph path's problem, so the graph path is outside the row),
F-44 (the trainer is the bound) and F-45 (α = 1.0 rows) are cited where relevant and not re-proposed.

This document merges a predecessor's drafts (parts 1, 3, 4) with the box measurements that were
still running when it was killed (part 2 = §4 here). Where the arms CONTRADICT the predecessor's
model, the contradiction is stated and the measured number wins — §4.4 and Appendix A.

## 0. The one-paragraph answer

A run6 block is 25,001 trainer steps at ONE step per completed game (`train.training_steps_per_game
1.0`, `max_train_burst 1`; 0.97–0.99 steps per game measured), so the block's wall clock is
`25,001 × plies-per-game ÷ positions-per-hour`; the trainer thread is asleep 92 % (F-44) and
nothing on it is on the critical path. Positions/hour is set by ONE thread: the inference server,
which runs a strictly serial chain per pop — pop+fuse (Rust) → collate incl. check 14 (CPU, GIL
held) → forward (GPU, the thread spin-waits) → softmax/D2H/dispatch — and that thread is **90.6 %
busy at run6's 16 workers and 99.0 % at 32** (schedstat, §4.2). Its cost is LINEAR in leaves:
≈ 2.7 ms per pop + 0.52 ms per leaf (fit over the arms, §4.3) — there is no batching economy past
~35 leaves, because every term (check 14, the seven H2D copies, the GINE message pass over 440k
edges) scales with the edge count. Doubling the workers therefore bought +5 % leaves/s and pinned
the thread at 99 %; doubling the batch bought −3 %. The GPU is busy 52–58 % of wall (47 % of it the
server's forward, 6 % the trainer) and idle the rest, because the CPU half and the GPU half of the
chain never overlap. Consequently every lever is a MILLISECOND OFF THAT CHAIN, and they convert
≈ 1:1 into games/h today, not after some future saturation: (1) overlap the CPU half with the GPU
half (a two-stage pipeline; ceiling max(CPU, GPU) instead of their sum, ×1.6–1.8 ESTIMATE);
(2) check 14 off the chain WITH the GIL released — the landed `checker_thread` posture gained
nothing in PERF-3b and LOSES 12.6 % at 32 workers because the Rust verifier holds the GIL for its
whole 3–6 ms and the server thread waits for it (§4.4, arm `w32ct`); (3) the compiled trunk (−31 % of the forward, measured off-line); (4) the folded/tabled
edge linear and pinned H2D. The eval child that shares the GPU for 12–22 min of every ~54 min costs the self-play loop
**one fifth of its throughput while it runs** (arm `w16ev`, §4.5: −20.7 % leaves/s, two CUDA
contexts time-slicing plus 8 CPU threads) — ≈ 5–10 % of the block, the one block-level term no
record had measured; batching the child's inference is the lever.

## 1. Instruments and provenance

| id | instrument | window / denominator | where |
|---|---|---|---|
| P1 | `py-spy record --rate 50 --idle --threads --native --format raw` as the PARENT of the START-path burst (`drive_burst.py startpath_cd.yaml`), tree `4c4a83d4` | 97,554 main-thread samples = 1,951 s; 97,430 server-thread samples = 1,949 s; whole burst incl. the 6.4-min ring fill | `scratchpad/startpath/pyspy_idle_native.raw`; classifier `perf_investigation/pyspy_budget.py` → `pyspy_budget.txt` |
| P2 | the same burst's event stream (`iteration_complete.inference_batching`, `game_complete`, `target_integrity.positions_delta`) | 86,100 served pops over 1,942 s; 680 games; steps 1→496 | `perf_investigation/box_artefacts/events_startpath_cd_seg0001.jsonl`; `events_rates.py` |
| P3 | PERF-3b checker-thread A/B event streams (`inline` vs `checker_thread`, 120 steps each) | 150-s windows | `box_artefacts/events_perf3b_ab_*.jsonl` |
| P4 | ten-step `torch.profiler` of the production trainer step (R349(d)) | 10 steps, batch 256, box idle | box `/workspace/oc7/trainer_profile/`; `MEASUREMENT_STARTPATH_2026-09-11.md` §C |
| P5 | per-thread `/proc/<pid>/task/*/schedstat` sampler (5 s) + `nvidia-smi` at 1 Hz beside every arm | 150–450 s after arm start (inside the stepping window of every arm) | `box/thread_sampler.py`, `box/analyze_threads.py`, `arms_summary.py`; `box_results/pi_*/threads.jsonl`, `gpu_util.csv` |
| P6 | server-forward microbench at the served-batch shape off the START burst's step-400 ring (19,669 rows), `torch.profiler` + CUDA events, four code variants by monkeypatch (nothing under `src/` edited) | 40 timed serves per (variant, B); B ∈ {16, 35, 64, 128} | `box/server_forward_bench.py`; `box_results/bench.log`, `results.json` |
| P7 | config-only arms `w16` / `w32` / `w32b128` (`selfplay.n_workers` 16→32, `inference.inference_batch_size` 64→128), 150 steps each after a 1,024-row fill, tree `7a97fa80` | the stepping window steps 1→150 (403–487 s) | `box/pi_*.yaml` (minted by `startpath/mint_throwaway.py`, never under `configs/`), `box/run_arm.sh`, `box/run_all_pi.sh`; `box_results/pi_*/logs/events_*.jsonl`; `arms_summary.py` |
| P8 | this session's two arms: `w32ct` (32 workers + `inference.edge_geometry_check: checker_thread`) and `w16ev` (16 workers, `train.eval_interval 100`, 250 steps — one eval round overlapping the stepping window) | as P7 | `box/pi_w32ct.yaml`, `box/pi_w16ev.yaml`, `box/run_all_pi2.sh`; `box_results/pi_w32ct/`, `pi_w16ev/` |
| B1 | committed criterion floors `tools/bench_floors.toml` | box CPU (Ryzen 9 9900X, 12 cores / 24 threads) | `axis_graph_build/per_position` = 0.850 ms |

What P1 does NOT contain: the Rust worker threads. py-spy samples only threads registered with the
interpreter; a `std::thread::spawn`ed Rust thread never appears (research §1). The record holds
seven threads: MainThread, inference-server, selfplay-stats, eval-pipeline-poller,
heartbeat-watchdog, disk-guard and one unnamed autograd device thread. The earlier "16 Rust workers
99.6 % in a futex wait" reading cannot have come from this file (`grep -c futex` = 0); the worker
side is measured by P5. P1 also carries an OBSERVER EFFECT: the START burst served 1,546 leaves/s
under a 50-Hz native-unwinding sampler, the un-sampled `w16` arm of the same regime 1,687 (+9 %);
P1's SHARES are used below, its absolute rates are not.

Box facts that bound what can be measured: `torch 2.11.0+cu128`, RTX 5080 (sm_120),
`perf_event_paranoid 4`, `yama/ptrace_scope 1`, 24 logical CPUs, no `nsys` (only `ncu`, which needs
GPU performance-counter permission an unprivileged container normally lacks), the extension is
built without debug symbols.

## 2. The structure: one server thread, a closed loop of workers, and a cost linear in leaves

### 2.1 What one self-play move costs in inference round trips

`run_mcts_search` (`crates/mantis-selfplay/src/runner/search_drive.rs:375`) opens every search with
a ONE-leaf round trip (the root), then under `SearchKind::Gumbel` issues one round trip per
sequential-halving ROUND: `MctxRootState::round_batch` returns every candidate alive at the
current considered-visit level (`crates/mantis-search/src/mcts/gumbel_mctx.rs:104`), so a round is
`m, m/2, …` wide as `considered_visits_sequence(m, n)` dictates (`seq_halving.rs:27`). At run6's
values:

| budget | rounds (width × count) | round trips incl. root | leaves per round trip |
|---|---|---|---|
| quick, n = 64 (75 % of moves) | 16×1, 8×1, 4×3, 2×7, 2×7 (last truncated) | 1 + 19 = 20 | 3.2 |
| full, n = 320 (25 % of moves) | 16×4, 8×9, 4×19, 2×39, 2×15 | 1 + 86 = 87 | 3.7 |
| PCR mean (128 leaves / move) | | ≈ 37 | **≈ 3.5** |

A worker never has more than 16 leaves in flight and on average ≈ 3.5; 54 of a full search's 86
rounds are TWO leaves wide. `leaf_batch_size: 8` is inert under Gumbel except as the
`max_in_flight` clamp on the pop threshold. Inside a round the worker builds its leaves' graphs
SERIALLY (`infer_and_expand_graph`, `search_drive.rs:268–329`: a `build_leaf_graph` per leaf, then
ONE `submit_graphs_and_wait`) and blocks until the whole round returns. Cross-check from the arms:
127–130 leaves per position (P7, three arms) = the mean-128 budget plus the root leaf.

### 2.2 The pop rule

`GraphInner::pop_graph_batch_blocking` (`crates/mantis-selfplay/src/queues/graph.rs:85`) returns as
soon as the queue holds `saturation_threshold(batch_size, max_in_flight) = min(64/2, 16×8) = 32`
graphs, or after `max_wait_ms = 10`, and takes up to `batch_size`. Occupancy histograms (P7; a
bucket is a power-of-two LOWER bound): `w16` {32: 24,252, 64: 78, 16: 31} mean 34.9 — the server
takes the first 32–40 the instant the 32nd arrives; `w32` {32: 11,158, 64: 5,354} mean 54.9 — a
third of the pops find a FULL batch of 64 already waiting; `w32b128` {64: 15,372, 128: 17} mean 67.1.
The "fill 54.5 %" in `iteration_complete` is this policy, not starvation.

### 2.3 What `queue_wait` measures

`queue_wait` wraps the whole `next_graph_batch` call (`inference_server.py:618–622`), which is
(`crates/mantis-bridge/src/inference.rs:405–475`): the GIL-free pop, the per-id `in_flight`
bookkeeping under a lock, and the GIL-free block-diagonal FUSE `GraphWire::from_axis_graphs` —
so it has a floor that is work, not waiting: min 0.40 ms (`w16`), 0.56 ms (`w32`), 1.12 ms
(`w32b128`), growing with the pop. The schedstat split (§4.2) puts the server's true not-running
time at 1.9 ms per 20.7-ms cycle at `w16` and 0.3 ms per 31-ms cycle at `w32`; the remainder of
`queue_wait` (≈ 1.0 / 2.4 ms) is the fuse plus, under `checker_thread`, the wait to RE-ACQUIRE THE
GIL after the detached pop (§4.4).

### 2.4 The closed loop, read correctly

Little's law holds — leaves/s = leaves in flight ÷ round-trip time — but the arms show which side
binds. `w16`: 16 × 3.5 ≈ 56 leaves in flight, 1,687 leaves/s → RTT ≈ 33 ms, of which the worker's
own think time is ≈ 3 ms (§3.4: 0.89 ms of worker CPU per leaf × 3.5) and ≈ 30 ms is spent queued
or in service at the server. `w32`: ≈ 112 in flight, 1,771 leaves/s → RTT ≈ 63 ms. Doubling the
customers doubled the queue and left the throughput where it was: the textbook signature of a
saturated server in a closed network. The predecessor read the 14.5 % `next_graph_batch` share as
slack to be filled by more workers (+25–40 % pre-registered); the arm measured +5 %, and §4.2 shows
why: ≈ 40 % of that share was the fuse and the `in_flight` bookkeeping, the ≈ 9 % of the cycle that was
truly idle is exactly what the extra workers filled, and the server thread was already 90.6 % busy.

## 3. Time budgets, thread by thread

### 3.1 The block (the only wall clock that matters)

| term | value | source |
|---|---|---|
| steps per game | 0.97–0.99 (1 per game by `training_steps_per_game 1.0` / `max_train_burst 1`; the loss is two games completing inside one 0.1-s coordinator sleep) | P2 0.986; P7 0.968–0.980 |
| positions/h in the stepping window, run6 regime (16 workers) | 46,800 (13.0/s, `w16`); 43,200 (12.0/s) under py-spy in the START burst | P7, P2 |
| plies per game | 38.6 (START burst, steps 1–496); 30–42 in the 150-step arms (early-training games are short and noisy) | P2, P7 |
| games/h, stepping window | 1,105–1,120 at 38.6 plies (START); 1,344 at 34.8 plies (`w16`) | P2, P7 |
| block of 25,001 steps | ≈ 20.6 h at 46,800 positions/h and 38.6 plies (22.6 h at the py-spy'd rate), + 5–10 % for the eval rounds (§4.5) ≈ 21.6–22.7 h — the number is a function of game LENGTH, which moves with training | arithmetic |
| ring fill before the first step | 6.4 min (`min_buf_size 4096` ÷ 38.6 rows/game ÷ games/h) | STARTPATH §A |
| eval rounds | every 1,000 steps (≈ 54 min) a child process shares the GPU: 168–296 games × 4.1–4.4 s = 11.5–21.6 min per round → the child is alive 21–40 % of the block; the self-play rate DURING a round is **0.79×** (§4.5) → the block is 5–10 % longer than the self-play rate alone predicts (ESTIMATE) | PERF-3b, P8 |

### 3.2 The inference-server thread, per pop

Shares are P1's (of 1,949 s of server-thread samples at a 22.6-ms cycle); the ms column is
rescaled to the un-sampled `w16` cycle of 20.7 ms (48.3 pops/s, 34.9 leaves). The CPU/GPU/idle
split is P5's (§4.2), the collate and forward ms are P7's and P6's.

| phase | P1 share | ms per 20.7-ms pop | CPU, GPU-wait or idle | note |
|---|---|---|---|---|
| A `next_graph_batch` (pop + fuse) | 14.5 % | 2.9 (P7 `queue_wait`) = ≈ 1.9 idle + ≈ 1.0 fuse | idle + CPU | the idle is the only slack in the thread; it is gone at 32 workers |
| B `graph_wire_from_rust` + `plan_fused_forwards` + `slice_graph_wire` | 3.0 % | 0.6 | CPU | one fusion part per pop at B = 64 (0 splits in 24k pops); 32 splits in 15k pops at B = 128 |
| C1 check 14 `verify_edge_geometry` — Rust, GIL HELD, serial per-edge loop (`graph_contract.rs:210`) | 18.7 % | 3.0–3.9 (P6 2.98 at B = 35 warm; P1 scaled 3.9) | CPU | ≈ 7–9 ns per edge over 437k edges; the builder already ran `verify_contract` on every graph in the worker, so this is the post-marshal re-check (F-816-37) — 1-in-1 by `pool.py:141`, not a config knob |
| C2 collate tensor build: seven `torch.from_numpy(...).to(device)` from PAGEABLE memory | 8.1 % | 1.7 | CPU + copy | `x` 17k×11 f32 0.76 MB, `edge_index` 2×437k i64 7.0 MB, `edge_attr` 437k×5 f32 8.7 MB ≈ 16.5 MB per pop through the pageable staging path (each `.to()` = copy + stream sync) |
| C3 + C4 structural (13 numpy passes) + semantic 15/16 | 4.0 % | 0.8 | CPU | |
| D `stone_mask_from_batch` | 1.6 % | 0.3 | CPU | |
| E1 `_node_offsets_to_batch_vec` → `repeat_interleave(counts)` → `.item()` → `cudaStreamSynchronize` (`gnn.py:44`, called at `gnn_v2.py:165` AFTER the trunk and the policy head) | **32.2 %** | **6.7** | GPU-wait (the thread SPINS) | this sync ABSORBS the trunk's GPU time — it is the forward's cost appearing on the CPU thread, not 7 ms of bloat; `output_size=n_total` (known on the host, `gnn_v2.py:155`) removes the sync but not the time (P6 V1 = V0) |
| E2 kernel launches for trunk/heads/pooling | 8.5 % | 1.8 | CPU | 216 launches per forward (P6) at ≈ 8 µs each, eager dispatch |
| E3 `decode_binned_value`: `VALUE_SUPPORT.to(device)` — a pageable 65-float H2D every forward, a second full sync (`dist65.py:39`) | 5.2 % | 1.1 | GPU-wait | absorbs pooling + value head |
| F `segment_softmax`, `isfinite` gate (third sync), `.cpu().numpy()` D2H | 2.0 % | 0.4 | CPU + sync | |
| G `submit_graph_inference_results` (Rust, GIL held) | 1.4 % | 0.3 | CPU | |
| H other | 0.9 % | 0.2 | | |

Regrouped against P5 (§4.2): **CPU work ≈ 10.5–11 ms (51–53 %)**, **GPU-bound spin ≈ 7.8–8.3 ms
(38–40 %)** (the launch time straddles the two), **idle ≈ 1.9 ms (9 %)**. CPU and GPU never overlap: one thread, one stream, and three syncs inside
the forward plus the seven blocking H2D copies before it. GIL contention under `inline` is nil
(`gil_scoped_acquire` frames 0.13 % of the server thread; the main thread is asleep 92 %).

Served-batch shape (P2 means over 86,100 pops): 34.9 graphs, 17,210 nodes, 437,480 edges — ≈ 493
nodes and 12,530 edges per leaf, 25.4 edges per node, of which the dummy node's bidirectional
edges are ≈ 8 %.

### 3.3 The trainer main thread (P1, 1,951 s)

| where | share | per step (496 steps) |
|---|---|---|
| `time.sleep` at `coordinator/step.py:453` (O5, no new games) and `:447` (O4, ring fill), via `RealClock.sleep` (`config.py:154`) | 91.8 % | 3,611 ms |
| `loss.item()` sync (`core.py:397`) — absorbs the GPU forward+backward | 1.8 % | 72 ms |
| `sample_graph_batch` ring rebuild (Rust, GIL released, 7 threads) (`dispatch.py:194`) | 1.5 % | 57 ms |
| trainer forward's own `repeat_interleave` sync (`gnn.py:44`) | 1.4 % | 54 ms |
| check 14 on the trainer's collate (`graph_collate.py:167`) | 1.1 % | 42 ms |
| policy CE / losses (`losses.py:380,395,412`) | 0.5 % | 19 ms |
| everything else (slice, dist65, collate tensors, weight sync `inference_server.py:376`) | 1.9 % | 75 ms |

In-run step ≈ 0.32 s; standalone (P4) 0.244 s of which CUDA kernels 0.156 s and 31 stream syncs
(`tail_mass.tolist()`, `isfinite(loss)`, three `.item()` per micro-batch); `index_add_` 22 %, GEMMs
16 %, 7.8 GB of activations per step. GPU duty from the trainer ≈ 0.156 s × 0.37 steps/s ≈ 6 % of
wall (P4 × P7). None of it is on the block's critical path at this cadence (F-44). Under 32
workers `resolve_sample_threads` = `max(1, 24 − 32 − 1)` = 1 serialises the ring rebuild
(`config/resolve/sample_threads.py:38`): main-thread busy 8.3 % → 11.4 % (P5), steps/h unchanged.

### 3.4 The workers, the drain and the rest

The 16 worker threads are 7.9–12.0 % busy each, 1.58 cores in total (P5, `w16`); at 32 workers
3.9–7.9 % each, 1.77 cores. Worker CPU per leaf = 1.58 cores ÷ 1,687 leaves/s ≈ **0.89 ms**
(`w32`: 0.94), against the committed builder floor of 0.85 ms per `build_axis_graph` at 490 nodes
(B1) — the worker's CPU is essentially the graph build; selection, backup and completed-Q are the
remainder. Each worker's remaining ≈ 90 % is the futex wait in `submit_graphs_and_wait`.
`selfplay-stats` (the drain) is 99.1 % in `time.sleep`; `eval-pipeline-poller`,
`heartbeat-watchdog`, `disk-guard`: asleep; the autograd device thread: idle. The whole process
uses 2.5–2.8 of 24 logical CPUs.

### 3.5 GPU duty (P5, measured)

`nvidia-smi utilization.gpu` over 300 s of stepping: `w16` 52.8 % (p10 47, p90 57), `w32` 57.6 %,
`w32b128` 53.5 %. Of `w16`'s 52.8 %, the trainer is ≈ 6 % (P4 × step rate) and the server ≈ 47 %
→ ≈ 9.7 ms of GPU-busy per 20.7-ms pop, against P6's 8.75-ms GPU-event for a B = 35 forward
(the in-run forward shares the device with the trainer's kernels). The GPU is idle ≈ 45 % of wall.

## 4. The box measurements (P5–P8) — the part the predecessor never wrote

### 4.1 P6 — the server forward at the served shape, four variants (`bench.log`, `results.json`)

CPU side, per batch (median of 10; `collate(off)` = structural checks + tensor build + H2D,
`collate(full)` adds the semantic layer with check 14 inline):

| B | nodes | edges | `collate(off)` | check 14 alone | `collate(full)` |
|---|---|---|---|---|---|
| 16 | 8,213 | 209,000 | 0.73 ms | 1.52 ms | 2.37 ms |
| 35 | 16,011 | 407,072 | 1.22 ms | 2.98 ms | 4.41 ms |
| 64 | 29,542 | 752,716 | 2.35 ms | 5.50 ms | 8.31 ms |
| 128 | 55,864 | 1,418,812 | 4.74 ms | 10.66 ms | 16.70 ms |

Linear in edges throughout: check 14 ≈ 7.3 ns per edge, the rest of collate ≈ 3.2 ns per edge.

Serve wall (collate excluded: `forward_batch` under `inference_mode` + bf16 autocast, `segment_softmax`,
`isfinite` gate, D2H — median of 40), GPU-event time of the forward alone, and per-serve profiler
counts:

| variant | B = 16 | B = 35 | B = 64 | B = 128 | launches / syncs per serve |
|---|---|---|---|---|---|
| V0 as shipped | 4.99 ms (GPU 4.73) | 9.06 ms (GPU 8.75) | 16.25 ms (GPU 15.90) | 30.21 ms (GPU 29.85) | 217 / 22 |
| V1 no implicit syncs (`output_size=`, support cached on device) | 4.99 | 8.97 | 16.22 | 30.12 | 215 / 17 |
| V3 V1 + `conv.lin ∘ edge_proj` folded to one 5→128 linear per layer | 4.58 | 8.30 (−8 %) | 15.18 (−7 %) | 28.08 (−7 %) | 216 / 17 |
| V2 V1 + `torch.compile(representation, dynamic=True)` (compile 4.4 s) | 3.33 (−33 %) | **6.26 (−31 %)** | **11.32 (−30 %)** | 20.92 (−31 %) | 106 / 22 |

Outputs match V0 to bf16 noise at every B (`value_first` within 2e-4, `probs_sum` = B). The forward
is linear in B above ≈ 35: 0.26 → 0.24 ms per graph, with ≈ 1.2 ms fixed (launches + syncs) — the
GPU has all the parallelism it can use at 400k edges, so a bigger pop buys nothing on the device.

Kernel shares of the V0 B = 35 forward (10 serves, `Self CUDA` 84.8 ms → 8.48 ms per serve):
`index_add_` (GINE sum-aggregation, bf16 atomics, `indexFuncLargeIndex`) 29.8 %; `add` 17.5 %;
`relu` 12.1 %; `gather`/`index_select` 10.0 %; GEMMs (`addmm`, cutlass bf16 tensor-op) 13.7 %;
`scatter_reduce` (readout max) 6.6 %; layer_norm 2.8 %; copies 3.1 %. The eager message pass
(gather + add + relu + index_add) is **69 % of kernel time** and materialises three to four
(E, 128) bf16 tensors per layer (≈ 104 MB each at E = 407k); its bandwidth floor is ≈ 0.6–0.9 ms
per forward at the card's ≈ 960 GB/s against ≈ 5.9 ms measured (ESTIMATE of the floor from the
tensor sizes). `cudaStreamSynchronize` is 82 % of the serve's self-CPU time — the thread spins.

### 4.2 P5 — per-thread accounting (schedstat run-time ÷ wall, 150–450 s of each arm)

| arm | server thread busy | wchan of the server thread | workers (each) | workers total | main thread | all threads | GPU util |
|---|---|---|---|---|---|---|---|
| `w16` | **90.6 %** | running 95 %, `futex_wait` 5 % | 7.9–12.0 % | 1.58 cores | 8.3 % | 2.49 cores | 52.8 % |
| `w32` | **99.0 %** | running 100 % | 3.9–7.9 % | 1.77 cores | 11.4 % | 2.77 cores | 57.6 % |
| `w32b128` | 95.3 % | | 3.9–7.9 % | 1.75 cores | | 2.71 cores | 53.5 % |

The `w16` figure is stationary: per 5-s sample over the whole stepping window (81 samples) it is 90.7 % ± 0.9
(min 88.5, max 93.6), not an average of two regimes. "Busy" is time the thread was RUNNING, which includes the spin in `cudaStreamSynchronize` (the CUDA
default schedule spins while contexts < cores; 24 logical CPUs, two contexts): it is the length of
the serial chain, not CPU work alone. Subtracting the server's GPU-busy share (≈ 47 % of wall at
`w16`) gives CPU work ≈ 44 % plus the launch time overlapped with kernels — the §3.2 regrouping.

### 4.3 P7 — the three config-only arms (`arms_summary.py`, stepping window steps 1→150)

| arm | pops/s | occupancy | **leaves/s** | cycle per pop | `queue_wait` | collate per part | **positions/s** | games/h (plies/game) | fusion splits |
|---|---|---|---|---|---|---|---|---|---|
| `w16` (run6) | 48.3 | 34.9 | **1,687** | 20.7 ms | 2.87 ms | 6.16 ms | **13.01** | 1,344 (34.8) | 0 |
| `w32` | 32.3 | 54.9 | **1,771 (+5.0 %)** | 31.0 ms | 2.68 ms | 9.70 ms | **13.92 (+7.0 %)** | 1,358 (36.9) | 0 |
| `w32b128` | 25.7 | 67.1 | **1,721 (−2.8 % vs `w32`)** | 39.0 ms | 5.21 ms | 11.50 ms | **13.35** | 1,139 (42.2) | 32 |

Single 7–8-minute windows; positions/s wobbles ±5 % between 60-s windows inside one arm
(`events_rates.py … 60`), so +5–7 % is "small and positive", not a precise number. What is precise
is the shape: collate per part is 0.171–0.177 ms per graph in all three arms (6.16/34.9, 9.70/54.9,
11.50/67.1 — linear), and the cycle fits **≈ 2.7 ms + 0.515 ms per leaf** (`w16`/`w32`;
`w32b128` sits 1.7 ms above the line — its 32 fusion splits at the 1.37M-edge cap and the larger
working set). Per leaf the server costs 0.594 ms at `w16` and 0.565 at `w32`: the batching economy
between 35 and 55 leaves per pop is 5 %, and it is the whole of what 16 more workers bought.
`w32b128`'s threshold of 64 makes the server wait for fill again (`queue_wait` 5.2 ms, busy 95 %),
and the 128-cap pops split at the fused-edge cap: the arm is NEGATIVE against `w32`, as pre-registered
(L11 in the predecessor's table).

### 4.4 P8 arm `w32ct` — 32 workers + `checker_thread`: is PERF-3b's null the GIL or slack?

Pre-registered before the harvest (`w32ct_prereg.txt`): A (GIL) = leaves/s within ±6 % of `w32`,
`queue_wait` +2.5–5 ms, collate −3–5 ms, server busy DROPS; B (slack, the predecessor's reading) =
leaves/s ≥ +12 %, busy ≈ 99 %; C (thrash) = leaves/s ≤ −6 %.

| arm (stepping window, steps 1→150) | pops/s | occupancy | leaves/s | cycle | `queue_wait` (min) | collate / part | positions/s | server busy | server wchan | checker thread | GPU util |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `w32` (`inline`) | 32.3 | 54.9 | 1,771 | 31.0 ms | 2.68 ms (0.56) | 9.70 ms | 13.92 | 99.0 % | running 100 % | — | 57.6 % |
| `w32ct` (`checker_thread`) | 28.2 | 54.8 | **1,547 (−12.6 %)** | 35.4 ms | **12.09 ms (2.71)** | **4.07 ms** | 12.20 (−12 %) | **73.4 %** | running 70 %, `futex_wait` 30 % | 30.9 % busy | 51.5 % |

`edge_geometry_check` sub-block: 20,315 parts deferred, 0 inline fallbacks, 0 failures — the posture
did exactly what it says. Outcome **C, carrying A's signature**: the check left the collate timer
(−5.6 ms per pop, the cost of check 14 at B ≈ 55 per P6), the server thread's blocked time rose by
9.4 ms per pop (99.0 → 73.4 % running; `futex_wait` 30 % of samples — CPython's GIL is a
futex-backed condition), the cycle grew 4.4 ms and throughput fell 12.6 %. Even the FASTEST pop now
waits 2.7 ms (`queue_wait` min 0.56 → 2.71 ms): after the detached pop returns, the server must
re-acquire the GIL, and the checker holds it for the whole Rust call, which cannot be pre-empted.
The checker thread costs 30.9 % of a core (≈ 11 ms per pop, twice the 5.6 ms it removed —
cold-cache re-reads and the slice materialisation are its overhead, off the chain). Per-60-s
windows (`events_rates.py … 60`): the first two ran at 1,680–1,690 leaves/s (−5 %), the remaining
seven at 1,450–1,560 (−12 to −18 %) as games lengthened (48 plies/game in this arm vs 37).

Re-reading P3's raw streams over their whole stepping windows (steps 1→120) with the same
aggregation: `inline` 46.4 pops/s × 35.0 = 1,626 leaves/s, cycle 21.6 ms, `queue_wait` 3.02 ms
(min 0.40), collate 6.33 ms, 12.47 positions/s; `checker_thread` 45.7 × 35.0 = 1,600 leaves/s
(−1.6 %), cycle 21.9 ms, `queue_wait` 7.05 ms (**min 2.20**), collate 2.79 ms, 12.32 positions/s
(−1.2 %). The same signature at 16 workers, damped by the 9 % idle; PERF-3b's "+3.2 % positions/h,
−4.6 % wall" came from a 150-s window and from shorter games in that arm (120 steps in 332 s vs
394 s at equal positions/s), not from the lever.

PERF-3b's reading of its own A/B ("the serving loop was not the binding constraint … the pop wait
rises to absorb the freed time") is therefore corrected: at 16 workers the 9 % idle absorbed most of
the GIL wait and the net was ≈ 0; at 32 workers there is no idle to absorb it and the posture is
NEGATIVE. **The landed `checker_thread` posture must not be armed as it stands.** What the arm
proves for L2: the check's milliseconds DO leave the chain when moved (−5.6 ms measured in-run),
so a verifier that releases the GIL (`py.detach` around `verify_edge_geometry_impl` with a `Send`
wrapper of the six slices — the per-edge loop reads only two precomputed node tables,
`graph_contract.rs:64–84`) converts them at the full rate: 31.0 → ≈ 25.4 ms per pop at `w32`,
**+18–22 %** pre-registered; alternatively a 4-way `std::thread::scope` chunking of the per-edge
loop INLINE (no new thread, no GIL handoff, no backlog path) cuts it 5.6 → ≈ 1.5 ms, **+13–15 %**.

### 4.5 P8 arm `w16ev` — run6's regime with an eval round overlapping the stepping window

Arm: run6's 16 workers, `train.eval_interval 100`, 250 steps (`pi_w16ev.yaml`); the round opened at
step 100 (303 s), the step-200 round was `eval_round_skipped_busy`, the round closed at 1,025 s —
**722 s = 12.0 min** for the strength-floor probe (4) + ladder + gate screen + random floor at
`eval.concurrency 8`, `worker_device cuda`, `deploy_sims 160` (`MANTIS_EVAL_MEM`: the child's
`max_memory_allocated` 38 MB; its CUDA context shows as +555 MiB in `nvidia-smi`). `fill_45.py`:

| window | steps | wall | pops/s | occupancy | leaves/s | cycle | `queue_wait` | collate/part | positions/s | GPU util / VRAM max | server busy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| before the round | 1→99 | 215 s | 50.0 | 34.9 | **1,741** | 20.0 ms | 2.67 ms | 6.06 ms | **13.43** | 53.7 % / 7,856 MiB | 90.7 % |
| DURING the round | 100→240 | 721 s | 39.5 | 35.0 | **1,381 (−20.7 %)** | 25.3 ms (+26 %) | 3.53 ms | 6.99 ms (+15 %) | **10.67 (−20.6 %)** | 58.0 % / 8,411 MiB | 91.1 % |
| after the round | 241→250 | 37 s | 44.3 | 34.9 | 1,545 | 22.6 ms | 3.21 ms | 6.54 ms | 11.85 | 55.4 % / 8,078 MiB | 90.8 % |

(The first 138 s of the round cost −13 %; the loss deepened as the round's phases ramped. The
"after" window is 37 s and only partly recovered. Against `w16`'s whole-window 1,687 the loss is
−18 %; against its own before-window −21 %.)

**The self-play loop loses one fifth of its throughput while the eval child is alive.** The server
thread is as busy as ever (91 %) — its chain simply got 5.3 ms longer per pop: the GPU half is
time-sliced against a SECOND CUDA CONTEXT issuing a stream of tiny single-graph forwards (8 child
workers × `forward_single`, ≈ 200 launches each — a context switch per slice, not bandwidth: the
child allocates 38 MB), and the CPU half slows 15 % (the child's 8 threads and their cache
footprint; collate 6.06 → 6.99 ms). Block arithmetic (ESTIMATE from this one round and PERF-3b's
round lengths): a round of 12–22 min every ≈ 54 min at 0.8× the self-play rate costs
0.25 × (12–22) ≈ 3–5.5 min per 1,000 steps → **the block is 5–10 % longer than its self-play rate
alone predicts (1–2 h of 20.6)**, and the round itself runs ≈ 8 % slower under contention
(PERF-3b: 1,083 s idle vs 1,168 s mid-run). Levers, in §6 as L9: (a) batch the child's inference
(one batched forward per step across its 8 games instead of 8 × `forward_single`) — the
contention is per-launch and per-context-slice, so ×8 fewer launches should remove most of it
(pre-registered: during-round loss −20 % → ≤ −8 %; ABORT if still ≤ −15 %); (b) `eval.worker_device:
cpu` — config-only, trades a longer round for a quieter GPU (pre-registered: loss → ≈ −5 % from CPU
contention alone; the round ≈ 2–3× longer, so the block-level trade is roughly neutral — measure
both before choosing); (c) fewer rounds (`eval_interval`) — a regime decision, not a lever.

## 5. Bloat, dead waiting, real work — the classification

"Real work" = the leaf forward's arithmetic + the collation the contract requires + the tree work.
"Dead waiting" = a thread blocked on another that is itself not saturated. "Bloat" = work that
exists only because of how the code is arranged. Per pop at `w16` (20.7 ms, 34.9 leaves).

| item | class | per pop | why |
|---|---|---|---|
| server: pop wait | dead wait (structural) | 1.9 ms (9 %) | the closed loop at 16 workers; 0.3 ms at 32 |
| server: fuse in `next_graph_batch` | real work (marshalling) | ≈ 1.0 ms | GIL-free Rust; scales with the pop |
| server: GPU forward (trunk + heads), the thread spinning | real work, INEFFICIENT | 8.3–9.7 ms (40–47 %) | 69 % of kernel time is the eager message pass at ≈ 6–9× its bandwidth floor; GEMMs 14 % |
| server: the per-layer edge GEMM `conv.lin(edge_proj(a))` | bloat | ≈ 0.7 ms | `Linear ∘ Linear` is one Linear (V3: −8 %); the edge attribute is CATEGORICAL — axis one-hot (3) × signed_dist ±1..±5 (10) × src_player (3) ≤ 90 rows + the all-zero dummy row — so each layer's `e` is a ≤ 91×128 TABLE, not a GEMM over 437k rows |
| server: check 14 inline | duplicate work by governance, ON the chain | 3.0–3.9 ms (15–19 %) | every graph already passed the builder's always-on `verify_contract`; the re-check is the post-marshal instrument (F-816-37), 1-in-1 by `pool.py:141`. The landed `checker_thread` moves it off the collate timer but NOT off the chain: the Rust verifier holds the GIL for the whole check (§4.4) |
| server: pageable H2D (7 copies, 16.5 MB) | bloat (mechanism) | ≈ 1.7 ms | pinned staging + `non_blocking` overlaps it; under a codebook `edge_attr` need not travel at all (one byte per edge) |
| server: `repeat_interleave` `.item()`, `VALUE_SUPPORT.to(device)`, `isfinite` syncs | bloat (mechanism); free today, the WALL for pipelining | ≈ 0 ms of idle GPU today (P6 V1 = V0) | they forbid CPU/GPU overlap: with them the thread cannot collate pop N+1 while pop N runs |
| server: 216 eager kernel launches | bloat (mechanism) | ≈ 1.8 ms CPU, hidden only when nothing syncs | compile halves them (P6 V2: 106); CUDA graphs need bucketed shapes |
| server: structural + semantic numpy checks, slice, plan, stone mask, softmax, submit | real work (contract) | ≈ 2.0 ms | linear in B |
| workers: blocked in `submit_graphs_and_wait` | dead wait (structural, harmless) | ≈ 90 % of each worker | the server is the bottleneck; a worker costs 0.89 ms of CPU per leaf, 16 of them use 1.6 cores of 24 |
| workers: `build_axis_graph` per leaf, serial in the round | real work, OFF the critical path today | 0.85 ms per leaf (B1) | `build_leaf_graphs_batch(n_threads)` exists (`inference.rs:611`) and is unused on this path; matters only once RTT binds, which it does not (§2.4) |
| trainer: 92 % asleep | dead wait by DESIGN (cadence) | 3.6 s per step | `training_steps_per_game 1.0`; CARD-TRAINER-CADENCE, the architect's |
| trainer: 31 syncs per step, `index_add_` 22 %, 7.8 GB activations | inefficiency OFF the critical path | 0 wall | binds only if games/h ≈ ×4 or the cadence changes |
| GPU: idle ≈ 45 % of wall | dead wait (structural) | | the CPU half of the chain |
| eval child sharing the GPU 21–40 % of the block | contention (two CUDA contexts + 8 CPU threads) | −20.7 % leaves/s while a round runs (measured, §4.5) → block +5–10 % (ESTIMATE) | the child's forwards are single-graph and unbatched; a context switch per slice |

## 6. Levers, ranked by expected gain × confidence ÷ cost

LAW-09: each is a PROPOSAL with a pre-registered expected gain, an abort threshold and the cheapest
falsifier; nothing here is landed. Gains are on **leaves/s of the serving loop** = positions/h =
the block's wall clock at the 1-step-per-game cadence. The server's cost per leaf at `w16` is
S = 0.594 ms (≈ 0.30 CPU + 0.24 GPU-bound + 0.055 idle, §3.2/§4.2); a lever that removes Δ ms per
leaf gains S/(S − Δ) − 1 TODAY, because the thread is 90–99 % busy. Unmeasured gains are ESTIMATES
from measured ms; "measured" means the ms was measured in P6/P7/P8 and only its conversion is
inferred.

| # | lever | kind | Δ per leaf / expected gain | confidence | cost | cheapest falsifier / abort |
|---|---|---|---|---|---|---|
| L1 | **Two-stage server pipeline**: collate pop N+1 (CPU) while pop N's forward runs (GPU) — on ONE thread as a software pipeline (launch N asynchronously into pinned outputs + a CUDA event, then pop/collate N+1, then wait event N and dispatch N) or on two threads/streams; prerequisites: pinned staging + `non_blocking` H2D, `output_size=n_total`, the support constant resident on the device, finiteness checked on the host copy | code, large | cycle → max(CPU, GPU) + ≈ 0.03: **S 0.594 → ≈ 0.33 (+80 %) ESTIMATE upper bound; +40–60 % pre-registered** | medium (the mechanism is textbook and every torch op that syncs releases the GIL; the risks are the "one part resident" memory bound becoming two, and the failure-path guarantees) | high: `_run_graph_loop` restructure, LAW-18 counters, parity | prototype OFF-TREE in `server_forward_bench.py` (two batches ping-ponged) → serve throughput; ABORT if < +25 % over the serial loop; in-run it needs ≥ 32 workers (L5) |
| L2 | **Check 14 off the chain for real**: release the GIL in `verify_edge_geometry` (take the six `as_slice()`s, `py.detach` with a `Send` wrapper of pointer+len — research §5) so the landed `checker_thread` posture actually runs concurrently; or chunk the per-edge loop over 2–4 scoped threads (no cross-edge state) and keep it inline | code, small (Rust, one function) + a mint row | −0.085–0.11 ms → **+17–23 %** | medium-high: `w32ct` (§4.4) proved the ms leave the chain (collate −5.6 ms in-run) and that the GIL is what eats them (server blocked 27 % of wall, −12.6 % throughput) — the landed posture must NOT be armed as it stands | small | `w32` + `checker_thread` with the detached verifier: pops/s 32 → ≥ 37, `queue_wait` flat; ABORT if < +8 % |
| L3 | **Compile the trunk** (`torch.compile(representation, dynamic=True)`; P6 V2 −31 % serve wall at every B, compile 4.4 s, outputs = V0 to bf16 noise) | code, medium | −0.08 ms → **+16 %** (measured ms) | medium-high (risks: recompiles under shape drift — `mark_dynamic` on N/E/Lg, `recompile_limit` = 8 — and LAW-06 parity) | medium: a flag on the server's model construction, a parity test at bf16 tolerance, `load_state_dict` in place keeps the compiled graph valid across weight swaps | an in-run arm with the compiled server; ABORT on a recompile storm (> 8) or a parity miss |
| L4 | **Edge-attribute codebook**: `table_l = conv.lin(edge_proj(A))` over the ≤ 91 distinct attribute rows, `e = table_l[code]`, the code from `edge_attr` in a few pointwise device ops (or on the wire later); under compile the lookup fuses into the message kernel and `e` is never materialised. L4a alone = the V3 fold | code, medium (inference-side wrapper; the trainer keeps the GEMM or uses `F.embedding`) | V3 fold: −0.022 ms → +4 % (measured); table: −0.04 more → +7 % ESTIMATE | high for the fold, medium for the table | small / medium | extend P6 with a V4; ABORT if < −8 % serve wall at B = 64 |
| L5 | **`selfplay.n_workers` 16 → 32** (a mint row) | config | +5 % leaves/s, +7 % positions/s (MEASURED, `w32`); the PREREQUISITE for L1 (two pops in flight) | high | 0 code; +1.3 GiB RSS; the ring rebuild goes serial (`resolve_sample_threads` assumes a worker owns a core — it uses 5 %) | done (P7); re-run with L1 |
| L6 | **Pinned staging + `non_blocking` H2D** for the seven collate tensors (or hand the fused wire over as one pinned buffer from Rust) | code, small | −0.03 ms → +5 % ESTIMATE (P1's 1.7 ms per pop minus the copy itself) | medium | small | P6 with a pinned variant; ABORT if < −0.5 ms per pop at B = 35 |
| L7 | **Fused message pass** (Triton or a custom kernel: gather x[src] + table[code] + relu + segment-sum in ONE pass; with a per-batch dst-sort a CSR segment sum replaces the bf16 atomics — the builder emits edges per source walk, not dst-sorted, `mantis-graph/src/lib.rs:695–752`) — the graph path is OUTSIDE F-21's scope by that row's own words | code, large | GPU 0.24 → ≈ 0.12 ms per leaf → +28 % standalone ESTIMATE; overlaps with L3 (compile already fuses gather+add+relu) | medium-low (the bandwidth floor is arithmetic; reaching it is engineering; bf16 atomics are native on sm_120) | high | a Triton kernel in the bench against V2; ABORT if < −25 % of V2's forward at B = 64 |
| L8 | `max_train_burst` 1 → ≥ 2 / `training_steps_per_game` as the architect rules | config | keeps steps ≡ games once games/h grows: 0.97–0.99 today → ≈ 0.95 at ×2 ESTIMATE; the CADENCE itself is CARD-TRAINER-CADENCE | high | mint row | `steps_per_hour / games_per_hour` in `iteration_complete` |
| L9 | **Eval child contention** (measured: −20.7 % leaves/s for the 12–22 min of every ≈ 54 min a round runs, §4.5): (a) batch the child's inference across its 8 games; (b) `eval.worker_device: cpu` (config-only, longer rounds); (c) `eval_interval` (regime) | (a) code, medium; (b) config | block −5–10 % of wall (ESTIMATE: 0.25 × round length per 1,000 steps); (a) recovers most, (b) trades round length | high that the term exists (measured); medium on (a)'s recovery | (a) medium; (b) 0 | (a): a `w16ev`-style arm with the batched child, ABORT if the during-round loss is still ≤ −15 %; (b): the same arm with `worker_device: cpu`, read BOTH the during-round rate and the round wall |
| L10 | Trainer step: 31 syncs → ~3, `index_add_` → CSR/segment sum, compile | code | 0 on the block at this cadence; −30–50 % of the step's 0.24 s if it ever binds (INVESTIGATION-1 item 2) | medium | medium | P4 rerun |
| L11 | `inference_batch_size` 128 | config | **−3 % (MEASURED, `w32b128` vs `w32`)** — not proposed | high | | done |
| L12 | Parallel leaf build inside the worker (`build_leaf_graphs_batch`) | code, small | 0 today (RTT does not bind, §2.4); relevant only once the server is ≈ ×3 faster | high that it is 0 today | | after L1–L3 |
| L13 | int32 `edge_index` / int8 codes on the wire | contract change | ≈ −0.5–1 ms per pop of copy; subsumed by L1/L6 | | high (registry, check 14) | not proposed now |

Rank by expected gain × confidence ÷ cost (labels are stable, the order is the ranking): **L1, L2, L3,
L4, L5, L9, L6, L7, L8, L10**; L11–L13 are not proposed. Not proposed (LAW-05): incremental legal-set
maintenance on descent paths (F-17/F-19); INCR-GRAPH stays a parked candidate with its own
inequality; the trainer as a bound (F-44).

### 6.1 The stacked picture (model, ESTIMATE — every step has its own falsifier above)

| stage | S (server ms per leaf) | leaves/s | positions/h | block at 38.6 plies |
|---|---|---|---|---|
| today (`w16`) | 0.594 | 1,687 | 46,800 | 20.6 h |
| L5 (32 workers, idle gone) | 0.565 | 1,771 | 49,100 (measured) | 19.7 h |
| + L2 (check 14 off the chain, GIL released) | 0.47 | 2,130 | 59,000 | 16.4 h |
| + L3 + L4a (compiled trunk, folded edge linear) | 0.37 | 2,700 | 75,000 | 12.9 h |
| + L6 (pinned H2D) | 0.34 | 2,940 | 82,000 | 11.8 h |
| + L1 (pipelined: max(CPU ≈ 0.17, GPU ≈ 0.15) + 0.03) | ≈ 0.20 | ≈ 5,000 | ≈ 140,000 | ≈ 7 h; needs ≈ 64 workers and L8 |

At the last row the trainer runs ≈ 1.1 steps/s × 0.32 s = 35 % of its thread and ≈ 17 % GPU duty
(which slows the server's GPU half by about that) — still off the path, but L8 becomes necessary
and L10 starts to matter.

## 7. What could not be measured, and why

1. **Where inside a worker the 0.89 ms per leaf goes.** py-spy cannot see a Rust `std::thread`;
   `perf`/`samply` need `perf_event_paranoid ≤ 2` (the box has 4); Yama scope 1 allows only a
   parent to trace. P5 gives the busy fraction; the split (build vs. selection vs. backup) is
   inferred from B1. A `CARGO_PROFILE_RELEASE_DEBUG=true` build under `gdb -batch` thread
   backtraces from a parent gdb would give it, at the cost of a rebuild on the box.
2. **CPU work vs. GPU spin inside the server thread's 90.6 %.** schedstat cannot tell a spinning
   `cudaStreamSynchronize` from work; the split in §3.2 is P1's shares + P5's GPU util. A
   `torch.profiler` pass INSIDE the running server (not the bench) would settle it — not taken
   because it needs a code hook.
3. **The self-play rate during an eval round** — measured now (§4.5, one round of 12 min at step 100 of a
   fresh warm start); a FULLY escalated 18–22-min round (screen passed → 128 confirm games) and the
   per-phase profile (ladder vs gate vs floor) are not separated.
4. **The transposition-table hit rate** inside a search (2-stone turns transpose): the TT exists
   (`MCTSTree.transposition_table`, per game), a hit skips the NN, and there is no counter (a
   LAW-18 gap, not a lever).
5. **Trainer GPU time IN-RUN**: P4 is standalone (0.244 s) vs. 0.32 s in-run; the difference is
   attributed to sharing the device with the server and the ring rebuild running with fewer
   threads — not separated.
6. **Per-leaf RTT distribution**: `queue_wait` is the server's, not a leaf's; RTT is derived by
   Little's law (§2.4).
7. **Kernel-level GPU counters**: no `nsys`; `ncu` is present but unprivileged containers refuse
   the performance counters — not attempted.
8. **Run-to-run noise of a 150-step arm**: ±5 % on positions/s between 60-s windows of one arm;
   every P7/P8 delta under ≈ 8 % is "sign and rough size", not a number.

## 8. Reproduction

- P1/P2: `perf_investigation/pyspy_budget.py scratchpad/startpath/pyspy_idle_native.raw 50` →
  `pyspy_budget.txt`; `events_rates.py box_artefacts/events_startpath_cd_seg0001.jsonl 240`.
- P3: `events_rates.py box_artefacts/events_perf3b_ab_{inline,checker_thread}_seg0001.jsonl 150`.
- P5/P7/P8: `python3 arms_summary.py box_results/pi_w16 box_results/pi_w32 box_results/pi_w32b128
  box_results/pi_w32ct box_results/pi_w16ev`; per-thread detail `box/analyze_threads.py
  box_results/pi_<arm>/threads.jsonl 150 450`; GPU `gpu_util_summary.py box_results/pi_<arm>/gpu_util.csv 150 300`;
  windowed rates `events_rates.py box_results/pi_<arm>/logs/events_pi_<arm>_seg0001.jsonl 60`.
- P6: on the box, `.venv/bin/python /workspace/oc7/pi/server_forward_bench.py <ring.bin> <out>`
  (ring = the START burst's step-400 bundle ring); results in `box_results/bench.log`, `results.json`.
- Arms: `box/run_all_pi.sh` / `box/run_all_pi2.sh` → `box/run_arm.sh <arm>` → `drive_burst.py` on
  `box/pi_<arm>.yaml` (minted by `startpath/mint_throwaway.py` with `train.max_train_steps=150
  train.min_buf_size=1024 train.terminal_eval_enabled=false train.draw_rate_abort=null` plus the
  arm's row; `pi_w32ct.yaml` carries `inference.edge_geometry_check: checker_thread` inserted after
  the mint because the leaf is optional and absent from the template).
- P8 `w32ct`: `w32ct_prereg.txt` (the decision rule, written before the harvest); `w16ev`: `fill_45.py`
  → `w16ev_table.txt` (before/during/after windows keyed on `eval_round_started`/`_complete`).
- Schedule arithmetic (§2.1): `considered_visits_sequence(16, 63)` and `(16, 319)` from
  `crates/mantis-search/src/mcts/seq_halving.rs`; rounds = runs of equal considered level.
- The fit in §4.3: cycle_ms = 2.7 + 0.515 × occupancy from (`w16`: 20.7, 34.9) and (`w32`: 31.0, 54.9).

## Appendix A — what the predecessor's drafts got wrong, and which measurement decided it

1. "The first lever is MORE LEAVES IN FLIGHT (+25–40 %)": `w32` measured +5 % leaves/s and a server
   thread at 99 % (§4.2, §4.3). The 14.5 % `next_graph_batch` share was read as idle; ≈ 40 %
   of it is the block-diagonal fuse and bookkeeping (GIL-free Rust) and the schedstat says the thread
   slept only 9 % of the cycle — which is what 16 more workers filled.
2. "Server levers are worth ×1.5–2.5 more AFTER saturation": they are worth their full value NOW
   — the thread is saturated at 16 workers.
3. "GPU idle ≈ 55 %, server 38 % + trainer 5 % (ESTIMATE)": measured 52.8 % busy (§3.5).
4. PERF-3b's checker-thread null was read as "the server was not the binding constraint"; the
   arithmetic of P3 (collate −3.3 ms, `queue_wait` +4.3 ms, cycle +0.5 ms) is the signature of a
   thread waiting to re-acquire the GIL the checker holds — §4.4's arm is the test.
5. Part 2 of the drafts never existed on disk; §4 is what it was to hold (P5/P6/P7).
6. A register candidate for whoever owns falsified.md (not written there by this investigation):
   "R347(e)'s `checker_thread` posture takes check 14 off the serving critical path" — falsified
   by `w32ct` (§4.4): the check leaves the collate timer and the server thread waits for the GIL
   instead, −12.6 % leaves/s at 32 workers, ≈ 0 at 16 where idle absorbs it. Mechanism: the Rust
   verifier holds the GIL (`graph_contract.rs:210`, no `detach`). The posture's LAW-18 counters
   (20,315 deferred, 0 fallbacks) prove it ran as designed; the design is what is wrong.

## Appendix B — online research (gathered 2026-09-11; cited in the text as "research §n")


**§1.** py-spy: `--native` symbolizes native frames of PYTHON threads only; pure native threads (Rust std::thread with no PyThreadState) are never sampled (python_spy.rs walks the PyThreadState list; issue benfred/py-spy#332 open; PR #637 `--native-all` unmerged). `--idle` keeps threads whose /proc state is not `R`. `--gil` keeps only samples from the GIL-holder thread. Stripped cdylibs symbolize to bare addresses; compile the extension with symbols for names. Sources: https://github.com/benfred/py-spy (README "How do you detect if a thread is idle", "Can py-spy profile native extensions"), https://github.com/benfred/py-spy/issues/332, https://github.com/benfred/py-spy/pull/637, https://www.benfrederickson.com/profiling-native-python-extensions-with-py-spy/

**§2.** torch sync detection: sync markers = cudaStreamSynchronize / cudaMemcpyAsync D2H under aten::item / aten::_local_scalar_dense, aten::nonzero, repeat_interleave without output_size, `.to()` without non_blocking. `torch.repeat_interleave(..., output_size=)` "will avoid stream synchronization needed to calculate output shape" (https://docs.pytorch.org/docs/2.14/generated/torch.repeat_interleave.html). Pageable `.to(device)` (non_blocking=False): PyTorch issues cudaMemcpyAsync then cudaStreamSynchronize — blocking (https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html; https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html). `torch.cuda.set_sync_debug_mode("warn"|"error")` is a prototype covering PyTorch-routed syncs (https://docs.pytorch.org/docs/2.14/generated/torch.cuda.set_sync_debug_mode.html; https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/sync-free-code.html).

**§3.** Scatter vs CSR: torch_scatter `segment_csr` "the fastest method ... fully-deterministic", needs destination-sorted index (https://pytorch-scatter.readthedocs.io/en/latest/functions/segment_csr.html); PyG `utils.segment` falls back to `torch._segment_reduce(offsets=)`; PyG SparseTensor/EdgeIndex sorted adjacency avoids atomics (https://pytorch-geometric.readthedocs.io/en/latest/notes/sparse_tensor.html). bf16 atomics: native `atom.add.bf16` needs sm_90+; older archs CAS loops (pytorch/pytorch#84981, triton-lang/triton#2708). sm_120 (RTX 5080) qualifies. GeoT segment reduction: 1.8x operator / 1.68x end-to-end over PyG baselines (https://arxiv.org/abs/2404.03019).

**§4.** torch.compile dynamic shapes: `dynamic=None` automatic; `mark_dynamic`; `torch._dynamo.config.recompile_limit` (alias cache_size_limit) default 8 then eager fallback (http://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html). Inductor lowers index_add/scatter_add to Triton `tl.atomic_add`, bf16 only on CC >= 9.0 (else ATen fallback) — torch/_inductor/utils.py `needs_fallback_due_to_atomic_add_limitations`. CUDA graphs (reduce-overhead / CUDAGraph Trees) re-record per unique shape: "pad input tensors to a few fixed tensor shapes" (https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html); vLLM precedent `cudagraph_capture_sizes` + pad (https://docs.vllm.ai/en/stable/design/cuda_graphs/).

**§5.** PyO3: `Python::detach` (0.26 rename of allow_threads) releases the GIL; "always call detach in situations that spawn worker threads" (https://pyo3.rs/latest/parallelism.html). `PyReadonlyArray` is !Send and GIL-bound (https://docs.rs/numpy/latest/numpy/struct.PyReadonlyArray.html); pattern: take `as_slice()` outside, pass a Send wrapper of the raw pointer+len into `detach` (https://github.com/PyO3/rust-numpy/discussions/363).

**§6.** Rust perf book: hoist allocations out of hot loops, `with_capacity`, FxHash for integer keys, `debug = "line-tables-only"` for profiles (https://nnethercote.github.io/perf-book/). `samply`/`perf` need perf_event_paranoid <= 1-2; the box has 4 (Ubuntu's "disallow all unprivileged") so neither works (https://github.com/mstange/samply; https://lwn.net/Articles/696216/). Yama ptrace_scope=1: only descendants may be traced (https://docs.kernel.org/admin-guide/LSM/Yama.html) — hence py-spy/gdb as the PARENT.

**§7.** Serving design precedent: KataGo self-play `numGameThreads = 128`, `numSearchThreads = 1`, `nnMaxBatchSize = 128` — "the number of game threads ... is very large, probably far larger than the number of cores ... intentional, as each thread only currently runs synchronously with respect to neural net queries, so a large number of parallel games is needed to take advantage of batching" (https://github.com/lightvector/KataGo/blob/master/SelfplayTraining.md; cpp/configs/training/selfplay1.cfg). KataGo's server pops greedily: blocks until non-empty then takes up to the cap, no timer (cpp/neuralnet/nneval.cpp). Lc0 multiplexing backend batches across game threads; larger minibatch trades strength for nps (https://lczero.org/blog/2019/04/backend-configuration/). AlphaZero.jl: batching across game simulations replaces in-tree virtual-loss parallelism; requires batch_size <= num_workers (https://github.com/jonathan-laurent/AlphaZero.jl/issues/71). mctx: sequential halving's per-game leaf parallelism shrinks m -> 2 across phases; mctx/Pgx vectorise over B = 1024 games per step instead (https://github.com/google-deepmind/mctx; https://arxiv.org/abs/2303.17503).

**§8.** /proc per-thread accounting: `/proc/<pid>/task/<tid>/schedstat` = run ns, runqueue-wait ns, timeslices (https://docs.kernel.org/scheduler/sched-stats.html); `stat` fields 14/15 utime/stime in clock ticks, parse after the last `)` (https://man7.org/linux/man-pages/man5/proc_pid_stat.5.html).

**§9.** CUDA host-side synchronisation behaviour: with the default `cudaDeviceScheduleAuto`, a blocking sync SPINS when the number of active CUDA contexts is below the number of logical processors, else it yields — which is why the server thread reads as 99 % RUNNING in schedstat while waiting for the GPU (https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html, `cudaSetDeviceFlags`). `torch.cuda.Stream` + `torch.cuda.Event` with pinned output buffers and `non_blocking=True` D2H is the stock way to let one Python thread keep launching while an earlier batch drains (https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-streams; the pin-memory tutorial in §2).

**§10.** GIL and a second serving thread: "libraries like NumPy and PyTorch release the GIL when
dispatching operations to low-level numerical computation routines" (PEP 703,
https://peps.python.org/pep-0703/), so a torch op that blocks in `cudaStreamSynchronize` does not
hold the GIL — but the Python-level glue between ops does, and pytorch/pytorch#77139 records a
significant throughput loss from GIL contention between the main thread and ONE extra Python thread
(the DataLoader pin thread) (https://github.com/pytorch/pytorch/issues/77139). This is why L1's
first form is a software pipeline on the ONE server thread (events + pinned buffers), and why the
`checker_thread` arm behaved as §4.4 shows: the Rust verifier never dispatches to anything that
releases the GIL.

---

# The one-page SUMMARY

# SUMMARY — where a run6 training block's wall clock goes (2026-09-11; full record: REPORT.md)

**The block.** 25,001 steps at ONE step per completed game (`training_steps_per_game 1.0`, 0.97–0.99
measured) → wall = 25,001 × plies/game ÷ positions/h = **≈ 20.6 h** at run6's measured 46,800
positions/h and 38.6 plies (P7 `w16`, P2), **plus 5–10 % for the eval rounds** (self-play runs at 0.79×
while a round is alive, arm `w16ev`) → ≈ 21.6–22.7 h. The trainer thread is asleep 92 % (F-44); the
block is priced by the self-play serving loop, and that loop is ONE Python thread.

**The time budget** (the inference-server thread, per pop of 34.9 leaves at 16 workers; cycle
20.7 ms = 48.3 pops/s; sources: py-spy shares P1, schedstat P5, events P7, microbench P6):

| where | ms | % | class |
|---|---|---|---|
| GPU forward (trunk + heads), thread spin-waiting in `cudaStreamSynchronize` | 8.3–9.7 | 40–47 | real work, 69 % of it an eager GINE message pass at ≈ 6–9× its bandwidth floor |
| check 14 `verify_edge_geometry` (Rust, GIL held, 1-in-1 by `pool.py:141`) | 3.0–3.9 | 15–19 | duplicate of the builder's `verify_contract`, on the chain |
| collate tensors: 7 pageable H2D copies, 16.5 MB | 1.7 | 8 | mechanism bloat |
| eager kernel launches (216 per forward) | 1.8 | 9 | mechanism bloat (compile halves it) |
| fuse + slice + plan + checks + stone mask + softmax + submit | 3.0 | 14 | contract work, linear in B |
| idle (waiting for the 32nd leaf) | 1.9 | 9 | the only slack — gone at 32 workers |

(The launch row partly overlaps the GPU row; the rows sum to the 20.7-ms cycle within ± 1 ms.)

The thread is **90.6 % busy at 16 workers, 99.0 % at 32** (schedstat); the GPU is busy 52.8 % of
wall (47 % server, 6 % trainer) and idle the rest because the CPU half and the GPU half of the chain
never overlap. The server's cost is LINEAR in leaves — 2.7 ms per pop + 0.515 ms per leaf (fit over
the arms) — so **32 workers bought +5 % leaves/s (measured), batch 128 bought −3 % (measured)**.
The 16 workers use 1.6 cores of 24 (0.89 ms of CPU per leaf, ≈ all `build_axis_graph`); the process
uses 2.5 cores. Every lever is therefore a millisecond off the server chain, worth its full value
today. (This overturns the predecessor's draft, which ranked "more workers" first at +25–40 %.)

**Top 10 levers** (LAW-09 proposals; gain = on leaves/s = positions/h = block wall; S = 0.594 ms per leaf today):

| # | lever | expected gain | conf. | cost | cheapest falsifier / abort |
|---|---|---|---|---|---|
| 1 | Two-stage server pipeline: collate pop N+1 while pop N's forward runs (pinned + `non_blocking` H2D, `output_size=`, support on device, finiteness on the host copy) | +40–60 % pre-reg (ceiling +80 % ESTIMATE: max(CPU, GPU) instead of the sum) | medium | high | off-tree prototype in `server_forward_bench.py`; ABORT < +25 % over serial; needs 32+ workers |
| 2 | Check 14 off the chain WITH the GIL released in `verify_edge_geometry` (`py.detach` + Send slice wrapper) so `checker_thread` runs concurrently; or 2–4 scoped threads inline | +17–23 % | med-high (arm `w32ct`: the ms leave the chain, −5.6 ms collate; the landed posture is −12.6 % because the Rust verifier holds the GIL — do NOT arm it as is) | small | `w32`+`checker_thread`+detached verifier: pops/s 32 → ≥ 37; ABORT < +8 % |
| 3 | `torch.compile(representation, dynamic=True)` — P6: −31 % serve wall at every B, outputs = eager to bf16 noise | +16 % | med-high | medium | in-run arm; ABORT on recompile storm (> 8) or parity miss |
| 4 | Edge-attribute codebook (≤ 91 distinct rows → per-layer 91×128 table; V3 fold alone measured −8 % of the forward) | +4 % fold (measured) + 7 % table (ESTIMATE) | high / medium | small / medium | P6 V4; ABORT < −8 % at B = 64 |
| 5 | `selfplay.n_workers` 32 — a mint row; the prerequisite for #1 | +5 % (MEASURED) | high | 0 | done |
| 6 | Eval-child contention: self-play runs at **0.79×** while a round is alive (MEASURED, arm `w16ev`: −20.7 % leaves/s over a 12-min round; two CUDA contexts time-slicing + 8 CPU threads) — (a) batch the child's inference across its 8 games; (b) `eval.worker_device: cpu` (config-only, longer rounds) | block −5–10 % of wall (ESTIMATE from the measured 0.79× and 12–22-min rounds every ≈ 54 min) | high the term exists; medium on (a) | (a) medium; (b) 0 | `w16ev`-style arm with the batched child; ABORT if the during-round loss is still ≤ −15 % |
| 7 | Pinned staging + `non_blocking` for the 7 collate tensors | +5 % ESTIMATE | medium | small | P6 pinned variant; ABORT < −0.5 ms/pop |
| 8 | Fused message-pass kernel (gather + table + relu + segment-sum in one pass; dst-sort → CSR instead of bf16 atomics); graph path is outside F-21's scope | +28 % standalone ESTIMATE, overlaps #3 (largest raw gain, lowest confidence) | med-low | high | Triton kernel vs V2 in the bench; ABORT < −25 % of V2 |
| 9 | `max_train_burst` ≥ 2 so steps stay ≡ games as games/h grows (the cadence itself is CARD-TRAINER-CADENCE, the architect's) | keeps 0.97–0.99 from drifting to ≈ 0.95 at ×2 (ESTIMATE) | high | mint row | `steps_per_hour / games_per_hour` |
| 10 | Trainer step: 31 syncs → ~3, `index_add_` → segment sum, compile | 0 on the block today; −30–50 % of the 0.24-s step if it ever binds | medium | medium | P4 rerun |

Not proposed: `inference_batch_size 128` (measured −3 %), incremental legal sets (F-17/F-19),
parallel leaf build in the worker (RTT does not bind: 0 today).

Stacked (ESTIMATE, each step has its own falsifier): #5 → #2 → #3+#4a → #7 → #1 takes S from 0.594
to ≈ 0.20 ms per leaf: 1,687 → ≈ 5,000 leaves/s, the block from 20.6 h to ≈ 7 h, at which point
the trainer's 0.32-s step is 35 % of its thread and #9 is mandatory.

**Could not measure, and why.** (1) Inside a worker (py-spy sees no Rust thread; `perf_event_paranoid
4`, Yama 1 → only busy fractions from `/proc`). (2) CPU work vs GPU spin inside the server's 90.6 %
(schedstat cannot tell; needs an in-server `torch.profiler` hook). (3) Only ONE eval round (12 min, step 100 of a fresh warm start) was crossed; a fully escalated
18–22-min round and the per-phase profile are not separated. (4) TT hit rate (no counter, LAW-18 gap). (5) In-run trainer GPU
share (0.32 s in-run vs 0.244 s standalone, unseparated). (6) Per-leaf RTT (only the server's
`queue_wait` exists; derived by Little's law). (7) Kernel counters (no `nsys`; `ncu` unprivileged).
(8) Noise: a 150-step arm resolves ± 5 %; deltas under 8 % are sign-only.
