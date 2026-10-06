# SEARCH-PERF-1: why our search does fewer playouts per second than Six's, and what is left to gain

Research packet SEARCH-PERF-1, written for the operator on 2026-10-06. The tree studied is `dev` 3ee61745; the branch
sits on `dev` a87f48b2, which adds only two governance commits. Nothing here changes `src/`, `crates/`, `configs/` or
`tools/`.

Labels:
- **MEASURED** carries its record.
- **ESTIMATED** carries its reasoning.
- **"Derived"** marks an ESTIMATED figure composed by arithmetic from measured parts.

Two regimes are kept apart throughout:
- **Regime 1, single-game play:** equal-time evals and cells on the GPU, and the ladder deploy on a CPU VPS. The GPU
  figures are for **one game per engine (concurrency 1)**, as the equal-time probe and the ladder run. Rung cells run 8
  game threads on one engine, and there several levers rarely fire (#1, #3, #7 in §3).
- **Regime 2, self-play throughput:** 32 workers batched on one GPU, with the trainer in the same process.

How the work was done:
- **Phase 0:** the lead read our search end to end, and the two rivals' engines statically.
- **Phase 1:** five experts covered GPU latency, MCTS algorithms, the systems boundary, the tactics solver and the CPU
  deploy.
- **Phase 2:** five fresh red teams attacked every option and updated their verdicts once the measurements landed.
- **Review:** a fresh reviewer checked this report against every record.
- **Training:** a sixth fresh red team checked the training section against run11's own events, and a box-B loop A/B
  with the trainer tied to games measured it (§3.5).
- **Measurements:**
  - the lead's, on the desktop (RTX 3070 + Ryzen 7 3700X) under the cell lock;
  - box B's wait-0 probes, from the RUN11-GO session;
  - box B's turn profiles and loop A/B, this leg's own bench under the operator's leave.
- **Restarts:** three experts were killed mid-work by a false-positive API safeguard. B was re-run from scratch; A and E
  resumed from the files their first runs had saved.

## 0. The answer in brief

**Why we do fewer playouts per second than Six.** It is not the network: ours is about 16× smaller, and the GPU is busy
only 17–21 % of a single-game turn as configured. Four things cost us, in order of size.

1. **The collector deadline.** The server wakes on 8 queued leaves or a 10 ms window it opens at its last pop. One
   game's batch is usually thinner than 8 (table hits, solver-ended descents and collisions take slots), so most round
   trips wait out the rest of that window: about **4.5 ms per round trip** on average (S1).
   - Waking on submission takes **30–35 % off a GPU turn** (desktop) and **12 % off a CPU stone** (MEASURED).
   - Box B's wait-0 probe reads 1.74–1.78× at 64–256/stone.
2. **Every round trip is CPU-heavy and pays for padding.**
   - Each leaf's 521-node, 13 k-edge axis graph is built from scratch and **serially**, 0.60 ms per leaf, because
     `leaf_build_threads` subtracts self-play's 32 workers and resolves to 1.
   - The eval path's 1-in-1 semantic checks cost ~0.22 ms per graph.
   - Almost every pop of 1–8 leaves replays the same 4 867-node CUDA-graph bucket. About 70 % of a one-leaf pop's GPU
     time is padding, and ~49 % of all nodes replayed in a 256/stone turn are padding.
   - On the CPU ladder, each layer of the message pass materialises four fresh [E, 128] fp32 temporaries (~7 MB per leaf
     each): 85 % of the CPU forward.
   - Six writes 8×25×25 planes in microseconds and calls ONNX Runtime in-thread.
3. **We make ~1.7× the round trips we need.** Contested stones carry 4.1–4.7 network leaves per trip; a filled trip
   carries 8.
   - At 64/stone (the ladder's budget) the main cause is a defect: virtual loss has the wrong sign at every second-stone
     node, so a pending child looks like a win, later descents collide with it, and each collision ends the batch.
   - At 256 the defect is about half the shortfall, and at 1 024 about a third. The rest is table and solver descents
     taking batch slots.
   - Fixing the sign cut round trips 36 % at 64/stone at the same budget (MEASURED counts: real net, 16 turns, CPU).
     Full rounds would cut 39–40 % (a bound from measured counters).
   - Six batches up to 32 descents per flush.
4. **We re-search what we already searched.** Each stone starts a fresh tree with no eval cache on the deploy path.
   22–34 % of a turn's network evaluations repeat positions already evaluated, mostly by the turn's first stone
   (MEASURED at 256/stone). Another 5–8 % repeat inside a single batch. Six keeps its tree across stones and turns.

**The tactics solver is not a cause.** On our probe positions the full block makes a turn faster: 139 against 228 ms
plain at 64/stone, because proofs replace network calls. The whole tactics area is ≤ 7 % of a turn there; expect up to
~2× that share on X-cell positions.

**The gap in fair units.** On contested turns Six makes ~3× (16–64/stone) to ~5× (256–1 024) our fresh network
evaluations per second as configured, and 1.5–3.7× at wait 0. Six's side is ESTIMATED from its code.

**Does it make training faster? Mostly not, and that is fine.**
- **The single-game-only fixes (#1, #3, #7, #9, #13–#15) reach a training run only through its gate rounds:** ≤ 0.3 % of
  throughput.
- **They are worth doing for play alone,** as the operator said in session (§7.1 item 3). At equal time, RUN11-GO's
  box-B cells moved from 0.391 to 0.637 against Strix at 1 s per turn when the batching wait was removed. Against Six
  they moved from 0.047 to 0.045 at 0.9 s.
- **Training speed is a different loop.** run11's self-play is latency-bound, and its in-process trainer costs it ~25 %.
  That is derived from MEASURED per-burst rates in run11's own events, and box B measured it directly at −25 % with the
  trainer switched off. The operator parked it; §3.5.3 sets it out.
- **The self-play levers are modest once the trainer's feedback is counted:**
  - the faster builder (#4): +2.5 to +4.5 % games/h;
  - skipping the search at quick-arm decided roots (#10): +1 to +3 %;
  - building a worker's round in parallel (#19), new in this leg: measured on box B at −1 %, so killed in that form
    (§3.5.7);
  - Sequential Halving round fusion (#16, a strength-class change): 0 to +14 %.
- **The largest per-leaf cut is the next run's pruned encoding (#18),** which shrinks the trainer's work too.

**What to do, in order** (§5; the operator's word in this session is in §7.1).
- **Leg 1, the free fixes: go ahead** (operator). Wake on submission, the parallel leaf build, small CUDA-graph buckets,
  the byte-identical builder, and the ladder's CPU message sum.
  - Four are bit-identical. Small buckets stay inside the served path's own spread (|Δp| ~1e-17).
  - MEASURED: a desktop 256/stone turn 543 → 293 ms through the build fix (~246–261 ms with small buckets, derived).
  - MEASURED: box B at 64/stone 187 → 77 ms.
  - MEASURED: the CPU ladder 2.4× per stone.
- **Leg 2: the per-game eval cache (#6), allowed by the operator, its ruling text owed.** Then the budget-aware early
  stop (#14; STRENGTH-class, ruling owed) as a per-use switch, set per consumer (§7.2(b)).
- **Leg 3: the protected checks during the forward (#5),** wanted by the operator once proven safe and free (§7.3).
- **Leg 4: the round-fill fix (#13),** which repairs a virtual-loss defect. It changes the search, so it needs a ruling
  and equal-time cells (§3.6.1).
- **Leg 5, training speed on a box:**
  - the builder in the loop;
  - #10;
  - C8;
  - the trainer's ring sample width.

  The trainer's interference waits for the operator.
- **Then:**
  - one tree per turn (#15) and round fusion (#16), as strength tests;
  - the knob re-tune sweep;
  - the next run's encoding (#18).

**Everything still open is set out in §7**: decisions with options and a recommendation, the measurements not made, and
the defects found. The rulings owed:
- R370(c) extended to eval paths (#6);
- the 1-in-1 collate posture and F-816-37's dump-on-fire (#5);
- the served-sims witness beside R378(c) (#14, #15);
- the deploy-unit change, on the R378(b) precedent (#13, #15);
- R346(c)'s round definition and the Gumbel round-width pin (#16);
- R275(b)'s exporter conjunct, for #10's zero-visit export;
- retiring the producer-side `verify_contract` where check 14 runs 1-in-1 (§7.5 D11).

## 1. Three engines, one playout and one turn

Read from source at `dev` 3ee61745 (ours) and statically from the vendored trees (Six: `vendor/external/six/engine/src`;
Strix: `vendor/external/hexo-strix/hexo-rs/hexo-mcts/src/mcts` and `tools/strix_driver.py`, our driver). Nothing of
either rival was run for this report.

### 1.1 Ours — the deploy head (regime 1)

`DeployHeadPlayer` (`src/mantis/arena/deploy_head.py`) drives the PyO3 `MCTSTree` from Python. A **turn is two
independent stone searches.**

- **Per stone:**
  - `tree.new_game(board)` resets the root, the in-search table and the solver table (`mcts/mod.rs:179-200`,
    `tactics_wiring.rs:282`).
  - `root_offence()` runs the root solver (8 turns, ≤ 20 000 nodes), only at moves_remaining 2, so once per turn.
  - Then the search proper:
    - the root is evaluated alone, as its own round trip;
    - `select_leaves(min(8, left))` → `expand_fn` until the descents (served + table hits + inline solver terminals)
      reach the budget;
    - the most visited child is chosen;
    - `torch.cuda.empty_cache()` is called.
  - `root_audit()` vets the choice: at moves_remaining 2, k 4 candidates × m 4 second stones at ≤ 2 000 nodes each, ≤ 40
    000 in all; at moves_remaining 1, one proof per candidate.
- **Per descent** (`selection.rs`):
  - PUCT with virtual loss over every legal cell as a child. The cap is 1 024; the radius-8 legal set is median 498 on
    our recorded positions.
  - Two passes over the children per level: the FPU's explored-mass sum, then the argmax.
  - `tactics_leaf`: `analyze` at every leaf, and the leaf solver (3 turns, 256 nodes) at quiet moves_remaining 2 leaves.
    This runs **before** the table probe, so table hits pay it too.
  - Then the in-search table probe.
  - Then the leaf is queued. The board is cloned three times (`selection.rs`, `bridge/src/mcts.rs:244`), each clone with
    a dirty legal cache.
- **Per round trip:**
  - In Python, `board.get_stones()` for each leaf.
  - In Rust (`bridge/src/inference.rs:552`), `build_leaf_graphs_batch`, serial: `leaf_build_threads` resolves to `max(1,
    cpus − selfplay.n_workers − 1)` = 1 on every host we run.
    - Each axis graph is built from scratch: median 521 nodes, 13 372 edges, about 509 KB of wire.
    - The builder recomputes the legal set; expansion recomputes it again.
  - The whole batch is enqueued under one lock. **No eval cache on this path.**
  - The Python `InferenceServer` thread pops when `min(64/2, max_in_flight 8)` = 8 leaves are queued or after **10 ms**.
    It then collates: the eval worker's period 1 (the full semantic layer every pop) and check 14 inline.
  - The forward is a replay of the smallest of 12 CUDA-graph buckets that holds the pop (bf16). The smallest holds 4 867
    nodes, so one leaf is padded about 9×.
  - The D2H goes to pinned buffers; the retire thread syncs the event, checks finiteness and wakes the waiter.
  - The 362-float dense policy crosses PyO3 into Python lists and back into Rust; the expansion sorts all legal cells by
    prior and writes ~500 child nodes; the table insert clones the policy.
- **Per game:** a 4 M-node pool (144 MB, pattern-filled), an 8 MiB solver table and a ~5 MB grid are allocated.
- **Ladder (CPU):** `tools/ladder/backends.py` builds the same engine on the CPU: fp32, eager, no CUDA graphs, collate
  1-in-64. `gine_message_sum` runs `index_select + add + relu + index_add_` and materialises four fresh [E, 128] fp32
  temporaries per layer, about 7.2 MB each per leaf.

### 1.2 Ours — self-play (regime 2)

`crates/mantis-selfplay/src/runner/`: 32 OS threads, one game each, each blocking on its own leaves.

- **Per stone:**
  - a fresh tree and the root offence;
  - a Gumbel search, including at a **decided root** (10.3 % of run11's positions are searched only to record a target);
  - Gumbel-Top-k with m 16, then Sequential Halving;
  - each round is one forced descent per surviving candidate, submitted together. Widths run 16, 8, 4, 2; run11 averages
    3.18 descents per round and ~40 rounds per searched stone (full search 97, quick 21), plus the root's own;
  - the audit.
- **Per leaf:**
  - a Zobrist key and the **exact eval cache**: 32 768 entries, shared by the workers, per net version. Each weight sync
    clears a shard on its first put of the new version (`queues/eval_cache.rs:174-178`); run11 syncs every 50 steps, ~40
    s. Hits are 17–35 % (32 % in run11);
  - a graph build on the worker thread on a miss, **one miss at a time**;
  - one submit per round.
- **Server:** the same, with collate period 1, check 14 on the checker thread, the trunk compiled and a saturation
  threshold of 32. Pops average about 36.
- **Trainer:** shares the GPU; its ring rebuild resolves to 1 sample thread on a 32-CPU box with 32 workers (PERF-3's
  drivers hard-coded 10).

### 1.3 Six (static reading)

One `go nodes N` per turn, from `position radius r moves …`. Our adapter: `cacheEntries 0`, `reuseTree` on (default),
`newgame` per game.

- **Per turn** (`Mcts::search`, `mcts.cpp:619-726`):
  - The root runs `analyze` plus the threat solver (20 000 nodes, 8 turns; a 25 % time share only under `movetime`).
  - The tree is **reused**: the moves since the last search are followed into the kept tree and compacted, otherwise a
    fresh tree is started.
  - Phase 1: `run` until **root visits, reused ones included,** reach 0.75 N. The first stone is the most visited child,
    proven wins first.
  - The stone is placed; a one-stone finish plays at once.
  - Phase 2: `run(first, …, N)` **until the chosen child's visits reach N**, continuing its subtree. The second stone is
    that child's most visited child.
  - New visits per turn: (0.75 N − reused) + (N − V1), between 1.0 N and 1.75 N.
  - The tree and its threat-solver table are kept to the next turn; the table is cleared only at `newgame`.
- **Per descent** (`gather`):
  - PUCT over **at most 40 children** (the top 40 by prior), one pass. FPU is the parent's net value − 0.25.
  - Incremental `place`/`undo` on one board.
  - At a new leaf: `analyze` (finish +1, unblockable −1, forced blocks), the threat solver at 64 nodes over 2 turns, the
    expansion cache, then `queue`.
  - `fillPlanes` writes 8 planes × 25×25 straight into the batch buffer, a few µs.
- **Per round:**
  - `run` stops a gather batch at 32 descents or at the first collision with a pending leaf.
  - `flush` calls ONNX Runtime once, synchronously: CUDA EP, fp32, input copied from host.
  - Per leaf, a partial sort keeps the top 40 by exp(logit − top); priors are renormalised.
  - One C++ thread, no FFI, no Python.

### 1.4 Strix (as our driver runs it; static reading)

`hexo_rs.gumbel_mcts_with_diagnostics` per **stone**, m 16, c_visit 50, no Gumbel noise, seed 0, a fresh tree, no reuse.

- The root VCF solver: 6 turns, 2 000 nodes. No leaf solver.
- Sequential Halving with virtual loss 0, so the serial loop: one descent per surviving candidate, batched into one
  Python `eval_fn` call.
- Heap nodes with `FxHashMap` children and a cloned `GameState` per node.
- `eval_fn`: a Python axis-graph build of about 1.7 ms per position, a PyG `Batch`, then the forward: 1.9 ms at B1 and
  8.1 ms at B64 on the 5070 Ti.

### 1.5 Side by side

| | ours (deploy) | Six (play) | Strix (as driven) |
|---|---|---|---|
| search unit | per stone, 2 searches a turn | per turn; the 2nd stone continues the chosen child | per stone |
| budget unit | descents: served + table + inline (R376(e), R378(c)) | root visits incl. reused; the 2nd stone tops up to N | root visits |
| tree reuse | none | across turns + within the turn | none |
| NN reuse | in-search table, cleared per stone | 32 k expansion cache (off in our runs) | none |
| children per node | all legal (≤ 1 024; median ~500) | top 40 by prior | all legal |
| leaves per round trip | ≤ 8, about 4–6 actual | ≤ 32 descents, ends at first collision | ≤ 16, halving |
| featurisation | Rust graph from scratch, 0.60 ms/leaf, serial | dense 8×25×25 planes, µs | Python graph, ~1.7 ms |
| inference | queue → Python server → CUDA-graph replay → retire thread; a 10 ms deadline | ORT Run, synchronous, in-thread | torch eager + PyG |
| collate checks | full semantic layer + check 14 inline every pop (eval) | none | none |
| decode | 362 floats → Python lists → Rust; full sort | 625 logits, partial sort top 40 | Python lists |
| leaf solver | 3 turns / 256, before the table probe | 2 turns / 64 | off |
| root solver | 8 / 20 000 once a turn | 8 / 20 000 once a turn | VCF 6 / 2 000 per stone |
| audit | ≤ 40 000 nodes a stone | none | none |
| solver table | cleared per stone | kept per game | per call |
| language | Python loop over a Rust tree + a Python server | C++ | Python eval over a Rust tree |

## 2. Our measured profile: where a turn's time goes

**Hosts and tags.**
- **The desktop:** RTX 3070 (sm_86) + Ryzen 7 3700X, torch 2.11 + cu128, run11@108k, configs/run11a2.yaml's deploy
  block.
  - Sessions ran under the cell lock: 14:37–14:40 CEST (session 1) and 15:47–16:01 (session 2a).
  - Every job is tagged CONTENDED by the strict rule: a foreign, mostly idle Six process sat on the card, and the CPU
    load was 1.3–4.4.
- **Box B:** RTX 5070 Ti + Ryzen 9 5900XT, idle, from the RUN11-GO probes.

**Samples.** The desktop turn profiles play the two stones of recorded turn starts (moves_remaining 2, ply ≥ 8), drawn
from a pool of 512 box-B cell positions; every arm plays the same positions. Turns per arm:

| Run | Sample |
|---|---|
| Turn profiles | 40 / 30 / 12 turns at 64 / 256 / 1 024 per stone (after 6 / 6 / 3 warm-up turns) |
| A's move witness | 12 turn starts |
| E's CPU deploy | 24 positions at 64, 12 at 256 |
| The virtual-loss check (real net) | 16 turns on CPU |
| B's game replays | 8 games at 256, 3 at 1 024 |
| C-M3 | 24 games, turns 1–4 |

**Statistics.** The turn-time distribution is bimodal (decided turns take ~10 ms), so the desktop compares MEANS. The
box-B probes report medians of real game turns.

### 2.1 A single-game turn on the GPU (concurrency 1)

| Budget per stone | Desktop, mean ms, as configured | Wake (#1) | + parallel build (#2) | Box B median s, as configured | Box B, wait 0 | Six, same games |
|---|---|---|---|---|---|---|
| 64 | 210 | 139 | 110 | 0.28 | 0.16 | 0.144 s (128/turn) |
| 256 | 543 | 375 | 293 | 0.95 | 0.53–0.55 | 0.296 s (512/turn) |
| 1 024 | 1 676 | 1 174 | — | 2.8–3.3 | 1.18–1.72 | 0.882 s (2 048/turn) |

The three desktop arms ran **identical searches**: the same descents, served leaves, table hits and revisits per turn.
- S1: `lead/measure/S1_RESULTS.md`.
- A's M2 witness: 12/12 identical moves and root values at 64 and 256 (`lead/measure/S2_RESULTS.md`).
- Box B: `mantis-mirror/run11/versus/box_b/timing` and `timing_wait0c`, solvers on, the main pairings.

**Box B, the same recorded turn starts** (MEASURED in this leg: 30 turns, run11@108k, concurrency 1, 5070 Ti + 5900XT).
The search counters are identical in every arm: 54.8 / 169.3 evaluations and 14.3 / 38.9 round trips per turn.

| Budget per stone | As configured, mean (median) | Wake (#1) | + 8 build threads (#2) | Overall |
|---|---|---|---|---|
| 64 | 187 ms (151) | 95 (50) | 77 (51) | 2.4× |
| 256 | 487 ms (177) | 316 (72) | 220 (63) | 2.2× |

On box B the GPU replays a small pop in ~0.95 ms against ~2.4 ms on the 3070. The CPU side (collate ~1.5–1.9 ms per pop,
the build) is therefore a larger share there, and small buckets (#3) have less to remove.

Box B's wait-10 / wait-0 ratio by pairing (v Six / v Strix): 1.74 / 1.78 at 64, 1.78 / 1.75 at 256, 1.92 / 2.37 at 1
024. The wake saves **4.5 / 4.3 / 4.7 ms per round trip** at 64 / 256 / 1 024 (desktop: the turn saving divided by round
trips).

**The configured 256/stone turn (543 ms), split by the packet's categories.** Derived from S1; the parts overlap a
little, so they are not exactly additive:

| Category | Time per turn |
|---|---|
| The collector's wait (removed by #1) | ~169 ms |
| Featurisation (the graph build) | ~82 ms removable by threads, more in total |
| Inference: collate incl. checks | ~83 ms (39 × 2.13), ~35 ms of it the semantic layer |
| Inference: GPU wait | ~104 ms (39 round trips × 2.66 ms) |
| Search (`select_leaves`: descents, `analyze`, the leaf solver) | 19 ms |
| Decode and expand (PyO3 + Rust) | 18 ms |
| Root solver + audit | 2.6 ms |
| Python around the submit | 2.4 ms |

After #1 and #2 (mean 293 ms):

| Part | Time per turn | Share |
|---|---|---|
| Network round trips (build + serve) | 250 ms | 85 % |
| `select_leaves` | 18 ms | 6 % |
| Expansion | 17 ms | 6 % |
| Root offence + audit | 2.6 ms | 0.9 % |
| Python | 2 ms | 0.7 % |

There are 39 round trips per turn, each with **4.3 network leaves**. Each costs ~2.2 ms of CPU launch (collate ~2.0) and
~2.5 ms of GPU wait. The GPU is busy 17–21 % of a turn as configured, and 32–43 % after the fixes (A's M2).

### 2.2 One round trip, decomposed

Desktop, median ms (`S1_RESULTS.md` §A; `S2_RESULTS.md`, A's M1 and M3). The rows overlap: the retire contains the GPU
replay, and the server's window overlaps the caller's build. **Read them as stage costs, not a sum.**

| Stage | B1 | B8 | B64 | Lever |
|---|---|---|---|---|
| Caller-visible collector wait, as configured | ≈ 6.7 (11.40 − 4.75; the server's own window reads 10.1, but it opens before the submission) | ≈ 0 (threshold met) | 0 | #1 |
| Leaf build, serial | 0.60 per leaf | 4.8 | 38 | #2 (÷ threads), #4 (−45 %) |
| Pop + fuse into the wire | 0.04 | 0.53 | 3.95 | |
| Collate: pack | 0.26 tight / 0.61 padded | 0.62 | 4.3 | |
| Collate: semantic layer (1-in-1 on eval paths) | 0.28 (check 14 0.19) | 1.76 (1.60) | 13.6 (12.9) | #5 |
| GPU replay of the bucket | 2.38 (4 867-node bucket) | 2.18 | 16.8 | #3 |
| GPU replay at a tight shape / a 1 025-node rung | 0.76 / 0.85 | 2.05 | 13.3 | #3 |
| Retire (event sync incl. the replay, host checks, wake) | 2.39 | 2.17 | 17.4 | |
| Whole evaluate: configured / wake / wake + 8 build threads | 11.4 / 4.75 / 4.60 | 11.5 / 12.0 / 7.45 | 88.6 / 90.4 / 54.3 | |

**The forward itself is small.** A pop of 1–8 leaves almost always replays the same padded bucket for ~2.2–2.4 ms. M1's
linear fit (0.44 ms + 0.40 µs per padded node) and the tight one-leaf graph (0.76 against 2.38 ms) put **~70 % of a
one-leaf pop's GPU time** in padding. Over a 256/stone turn the replays carry 5.89 M padded nodes for 3.02 M real ones
(~49 % padding).

**Everything else is CPU, mostly per leaf:**
- The build, 0.60 ms per leaf. It splits edge walk 37 %, legal set 23 %, threat features 15 %, the producer verify 13 %
  (C-M1). A byte-identical rewrite measures −45 %.
- The semantic checks, ~0.22 ms per graph.
- The fuse and the pack.

**Output invariance on the served path.** A position's output does not depend on the pop it rides in: |Δvalue| 0.0 and
|Δp| ≤ 1e-17, with 1–2 of 512 positions differing at all over B 1–64 (A's M5, sm_86). The eager tight path does vary,
with |Δvalue| up to 0.025 at B 32–64.

### 2.3 Why Six does more, normalised

**Six's unit.** `go nodes N` counts root visits, reused ones included. Phase 1 caps root visits at 0.75 N; phase 2 tops
the chosen child up to N. New visits per turn are max(0, 0.75N − R) + max(0, N − V1): 1.0–1.75 N without reuse, from
0.25 N with it (`mcts.cpp:385-390, 659-714`).

**Our unit.** Our descents count table hits (10–25 % of contested descents) and revisits of proven terminals. Revisits
are 1.3 / 1.9 / 10.1 % of contested descents at 64 / 256 / 1 024; in decided stones they are 52 of 64 to 952 of 1 024
descents, and decided stones cost ~10–15 ms.

**Fresh network evaluations per second, contested turns** (our root |v| ≤ 0.9):

| | 16–64/stone | 256/stone | 1 024/stone |
|---|---|---|---|
| Six over ours, as configured | ~3× | ~5× | ~5× |
| Six over ours, wait 0 | 1.5–1.9× (64) | 2.6–3.7× | 2.8–3.6× |

Six's counts are ESTIMATED from its code; ours are MEASURED from the probe records (expert B, red team B). Six's net is
~16× ours.

**The gap is cost per evaluated leaf and round trips per turn.**
- Six fills up to 32 descents per flush, from 8×25×25 planes written in microseconds, through ONNX Runtime in-thread.
- We fill 4.1–4.7 network leaves per trip on contested stones (B's counters). Each leaf is a 13 k-edge graph carried
  through three Python threads.

### 2.4 The search's own structure (desktop, real net)

| Per contested stone | 64 | 256 | 1 024 |
|---|---|---|---|
| NN leaves per round trip | 4.09 | 4.49 | 4.70 |
| Round trips, actual / if every trip filled to 8 | 13.7 / 8.3 | 40.9 / 24.3 | 137.7 / 82.2 |
| Select calls short of their ask | 51 % | 33 % | 19 % |
| Share of the shortfall that is collisions (the rest: table/solver descents taking slots) | 78 % | 49 % | 36 % |
| Budget a budget-aware early stop would save | 8.3 % | 18.5 % | 22.8 % |
| Second stone's evaluations already made by the first stone (real games) | — | 39.9 % | 50.4 % |
| Same-batch duplicate evaluations within a stone (a transposition queued twice before either returns) | 8.1 % | 8.2 % | 4.9 % |

On real game sequences (C-M3, turns 1–4), a per-game cache would replay **33.8 %** of a turn's network evaluations at
256/stone. S1's independent turn starts read 25 % (within-turn only), and B's game replays ~22 % at the turn level (red
team C's arithmetic).

**The collisions are the virtual-loss frame defect.**
- With the penalty applied in the chooser's frame, every select call fills: 8.00 of 8 on the stub, and round trips fall
  39 % and 31 % at 64 and 256.
- With the real net at 64/stone, round trips per turn fall 14.1 → 9.0 (−36 %) and NN leaves per trip rise 3.42 → 5.35
  (`lead/vl/VL_CHECK.md`; functional counts, 16 turns, CPU).

**The tactics block is a net time saver on these positions.** At 64/stone with the wake, the full block takes 139 ms
against 228 ms plain (61 against 106 evaluations per turn). The whole tactics area is ≤ 7 % of a turn after #1 and #2.
These positions are terminal-rich, so revisits run high and leaf-solver work light; on X-cell positions expect the
TACTICS-DEPLOY L5 ratio, 0.864× plain, and up to ~2× the area's share.

### 2.5 The CPU ladder

**The forward** (8 threads, batch 8, E's M1): 156.8 ms.
- The message pass is 133.5 ms of it (85 %): `add` 45.8, `index_select` 38.8, ReLU 26.5 and `scatter_add` 14.6 ms, over
  the per-layer [E, 128] fp32 temporaries.
- Node MLPs 4.2 ms; readout 10.2 ms.

**Per leaf today:** 18.7 ms at B1, 16.7 at B4, 19.5 at B8.
- On the 3700X this is flat in batch and in threads, as R363 found. Other hosts differ: on box B the per-leaf cost rises
  ~3× from B1 to B8, and the 9950X scales 1.35× from 4 to 8 threads.
- The cause is memory traffic. An allocator swap fixes B1's page faults (18.8 → ~9 ms) but not B8's traffic.

**With E1a + E2** (the chunked, order-preserving sum and the edge-table hoist; bit-identical): 7.8–9.1 ms per leaf,
which is 2.0–2.5×. That is 1.6–1.75× on a 2-core VPS stand-in and 2.45× on one thread.

**The ladder's deploy per stone** (E's M5; 0 differing stones in every variant):

| Setting | Configured | Wake | + E1a + E2 | + 4 build threads | Speedup |
|---|---|---|---|---|---|
| 64 sims, 8 threads | 542 ms | 479 | 242 | 224 | 2.4× |
| 256 sims, 8 threads | 1 795 ms | — | — | 697 | 2.6× |
| VPS stand-in, 64 sims | 577 ms | — | — | 301 | 1.9× |

### 2.6 Self-play (regime 2): not re-measured here

The loop's numbers are PERF-2's and PERF-3's, on box A (4080S + 9950X):
- 235 k positions/h alone, 149–156 k with a fixed-rate trainer (PERF-2; PERF-3's loop read ~159 k);
- pops of B ~36;
- workers ~85 % blocked;
- 8 of 32 CPUs busy;
- per served leaf: select 98 µs, build 622 µs, expand 92 µs.

**Four facts from this leg:**
- **Every self-play pop runs the semantic layer 1-in-1** (`selfplay/pool.py:145-152`), not 1-in-64. Check 14 runs on the
  checker thread.
- **`gpu_wait` includes the retire thread's GIL reacquisition.** PERF-2's own box records nevertheless put the trainer's
  cost on the GPU: device wait +2.66 ms against a GIL-sensitive launch +0.40 ms per pop. A's M4 probe agrees. A
  GIL-holding background would be catastrophic (a pop at 234 against ~20 ms). PERF-2's isolated loop showed none, but
  run11's events put +2.65 ms per pop on the untimed slot/GIL span while the trainer steps (§3.5.3).
- **The production trainer rebuilds its ring on 1 sample thread on box A** (`sample_threads.py:69` subtracts 32 workers
  from 32 CPUs). PERF-3's drivers hard-coded 10, so PERF-3's trainer readings do not represent production.
- **10.3 % of self-play positions are decided roots,** searched only for their row (run11's events).

**Why none was measured here.** A self-play measurement on the desktop would not transfer to box A: 16 against 32
threads, a 3070 against a 4080S. The operator also ruled out a worker sweep for this leg.

## 3. The options, ranked

### 3.0 How to read the tables

**Merged duplicates, counted once:**
- A1 = C1 = E3's wake.
- C2 = E3's build width.
- B4 = C4.
- A6 = C9 + CARD-PERF-DST-SORT.
- The virtual-loss fix, B3's refill and the first-collision stop form one round-fill leg.
- C6, PLAUSIBLE by red team C, is folded into #5 as its serve-first form: #5 contains it and dominates it (red team A).
  The milliseconds are the same, so the saving is counted once.

**Overlaps to remember when stacking:**
- #2, #4 and the wire's builder half share the same serial-build milliseconds. After #2, only the slowest leaf's build
  stays on the critical path.
- #5 and the wire both count check 14.
- #15 contains most of #14's first-stone saving, and its evaluation saving lies inside #6's.
- D5 contains D4.
- D1's within-turn saving lies inside #15 and overlaps #6.

**Verdicts** are the red teams' final ones, after session 2.

**Desktop turn percentages** are means at 256/stone at concurrency 1, after the rows above them unless the row says
otherwise:
- base: 543 ms as configured;
- 375 after #1;
- 293 after #2.

**Out of scope:** knob values, by the operator's direction (§4, §5).

### 3.1 Regime 1, single-game play. SPEED: the same moves at a fixed budget

| # | Option (source ids) | Expected gain | Evidence | Red team | Cost | Risk | How to prove it |
|---|---|---|---|---|---|---|---|
| 1 | **Wake on submission** (A1 = C1 = E3a): a single submitter's batch pops at once, with no 10 ms window and no spin. | **GPU turn −30 to −35 %** (MEASURED: S1 + A's M2); **box B −35 % at 256/stone, −49 % at 64** (MEASURED in this leg). Box B wait-0 proxy 1.74 / 1.78× at 64, 1.78 / 1.75× at 256, 1.92 / 2.37× at 1 024 (v Six / v Strix; MEASURED). **CPU ladder −12 % per stone** (MEASURED). Self-play 0. | S1: 210 → 139 ms (64), 543 → 375 (256), 1 676 → 1 174 (1 024), identical search counters, 4.3–4.7 ms saved per round trip. A's M2: 12/12 identical moves and root values. E's M5. | CONFIRMED | Small: the queue's wake rule (`queues/graph.rs:61-115`) or the engine's declared supply. | At concurrency 8 (cells) it barely applies: it changes nothing while all 8 games have leaves queued and only shifts timing when 2–7 do. In self-play the threshold is met already. The one risk is under-declaring submitters on a shared engine. | Deploy-24 byte-equal moves. LAW-18: the wake's fire count. LAW-09: turn −25 % at 256/stone; abort under −12 %. |
| 2 | **Parallel leaf build on standalone single-game hosts** (C2 = E3b): resolve `leaf_build_threads` for the host running one game. Today it subtracts self-play's 32 workers and resolves to 1. | **GPU turn a further −21 to −22 %** (MEASURED: 375 → 293 at 256; 139 → 110 at 64); **box B a further −30 % at 256, −19 % at 64** (MEASURED). **CPU a further −7 % per stone after #9** (MEASURED). Self-play 0. | S1; A's M2 (12/12 identical); E's M5. The build is 0.60 ms per leaf serial on the 3700X (A's M3). | CONFIRMED | Tiny (`config/resolve/leaf_build_threads.py`). Also fix `par.rs:34` dropping a named panic payload. | Bit-identical (`map_in_order`). Only standalone hosts widen. The in-run eval beside self-play keeps the reservation (`src/mantis/run.py:641-643`); widening it would take CPU from run11's workers. | As #1. Pin that self-play's reservation is unchanged. LAW-09: a further −15 %; abort under −7 %. |
| 3 | **Small pops replay small buckets** (A3): continue the bucket ladder below 4 096 nodes, with a ≥ 1 024-node floor. | **~1.0 ms per pop → turn −11 to −16 %** (derived: measured per-pop parts, S1's pop histogram, a modelled padded-node cut 5 039 → ~3 100 per pop). Self-play 0; CPU 0. | M1's ladder probe: one-leaf replay 0.85 / 1.07 / 1.44 / 2.37 ms at 1 025 / 1 601 / 2 501 / 4 867 nodes. ~70 % of a one-leaf pop's GPU time is padding; pops average 4.3 leaves. | CONFIRMED | Small (`selfplay/served_graphs.py`'s ladder; more captures). | Inside the served path's own spread, not proven bit-identical: on sm_86 an extended ladder down to 655 nodes moved 2 of 512 positions by \|Δp\| 1e-17 (Δvalue 0, no argmax/sign flips), and production moves 1 of 512 at B ≥ 16. | At the ≥ 1 024 floor, on each card that serves cells or the gate (sm_86 desktop; sm_89 box A): 0 differences, or differences inside production's own pop spread with 0 argmax/sign flips. LAW-18: captures and rung per pop. LAW-09: −8 %; abort under −4 %. |
| 4 | **Byte-identical faster graph builder** (C3): dense legal grid, BFS distances, dedup at emission, one-pass verify. | **Build −45 %** (MEASURED). **Turn −3 to −4 %** after #2 (−12 % without #2; derived). **Self-play +4 to +6 % positions/h** (ESTIMATED from PERF-3 L2's calibration; open loop, a real run +2.5 to +4.5 %, §3.5.5). | C-M1: 588 → 322 µs per build, 0 mismatches over 520 cases × 8 variants. Phases: edge walk 37 %, legal set 23 %, threat features 15 % (the next target), producer verify 13 % (against its comment's "under 3 %"). | CONFIRMED | Medium (`crates/mantis-graph`). | Byte-identical by parity. | The criterion bench; an interleaved 2×2 loop A/B on a box (CARD-PERF-LOOP-SPREAD). Loop line +3 %; abort under +1 %. |
| 5 | **The protected checks while the forward is in flight** (A2 ⊇ C6): two steps. (b) Checks 15–16 move into the Rust pack's gather pass, GIL-free, still before the launch (both regimes). (a) On single-game engines only, check 14 (already GIL-free) runs on the retire thread between the launch and the release; self-play keeps it on its checker thread. | **About −12 %** (derived: ~0.9 ms per pop × 39). Self-play 0 to +1.5 % in a real run, from (b) only (§3.5.5). CPU ladder 0: its forward is synchronous. | M1 / M3: the checks cost 0.28 / 0.51 / 0.92 / 1.73 ms at B1 / 2 / 4 / 8; check 14 is 68–95 % of it, and the GPU replay covers them at every pop size. F-46 and its repair annotation: a checker posture only helps once the verifier runs GIL-free. | PLAUSIBLE: **ruling owed** (the protected 1-in-1 checks; F-816-37's dump-on-fire) | Medium. | Keeps 1-in-1, dump-on-fire and halt-on-fire; C6's serve-first form loses halt-on-fire. The protected F-816-37 plant is refused by the coded pack before any check, so it cannot vouch for this. | A new check-14-only, in-vocabulary plant reds on the LAST pop of a round. LAW-18: the posture counter. LAW-09: ≥ 8 % at 256 over #1 + #2; abort under 5 %; deploy-24 byte-equal. |
| 6 | **Per-game exact deploy eval cache** (C4 = B4): per player, reset each game, never on the shared queue; it must also dedup keys already pending in a batch. | **CPU ladder −15 to −19 %** at 64/stone today, −10 to −14 % after #9, more at 256. **GPU −5 to −20 %** after #1–2, decided by the all-hit round-trip share (uncounted). Both ESTIMATED; the replay share is MEASURED. | Replay share at 256/stone: 33.8 % of a turn's evaluations on real game sequences (C-M3); 25 % within-turn on independent turn starts (S1); 39.9 % of the second stone's evaluations (B's replays, 8 games; 50.4 % at 1 024, 3 games). Same-batch duplicates add 5–8 %. | PLAUSIBLE: **ruling owed** (R370(c) extended to eval paths) | Small–medium (the self-play cache exists). | GPU: the served path replays identical values (A's M5, sm_86). CPU ladder: the eager path is batch-variant in the value (\|Δvalue\| ≤ 7.2e-7), so there the cache stays inside R370(c)'s spread but is not identical. | Same moves on 200 positions (expected 100 % on GPU). CPU same-move rate and turn ≤ −10 %. One X-cell re-read (≥ 99 % of games move-identical). LAW-18: hits and GPU evaluations. |
| 7 | **In-thread serving for one submitter** (A4): no queue hop, server thread or retire thread. | −5 to −10 % if the cross-session gap holds, −1 to −3 % if only the handoffs go (ESTIMATED). | M1: an inline pop is 0.6–1.2 ms cheaper than the threaded round trip, but S1 and S2 were different sessions. | PLAUSIBLE | Medium. | Low. | A same-session A/B; a per-pop saving ≥ 0.15 ms, else drop. |
| 8 | **Engine plumbing**: the exact first-collision stop (red team B's new finding); a leaner wire (A6 = C9 + dst-sort); a Rust-driven deploy loop (C5); a serving-only bf16 weight copy; solver micro-work (D3, D4, D5). | The stop: at most `select_leaves`' 4.5–6 % of a turn (a bound). Wire −2 to −4 %; C5 −2 to −3 %; weight copy −2 to −4 % of a small pop's GPU time; ≤ 1 % each for D3–D5 (ESTIMATED). Self-play: wire +1 to +4 %. | The stop: after an overlap the tree is back in the same state, so every later attempt repeats the same path (`selection.rs:271-292, 336-412`). M3's fuse/pack bases. S1's Python and expand buckets. `served_outputs` runs autocast with its cache off, so every replay re-casts every Linear. S1 bounds the tactics area at ≤ 7 % of a turn. | Stop: new, exact by red team B's reading. The rest: PLAUSIBLE, low value | Medium–high (the wire changes a contract and the protected plant). | The wire moves a one-bit fault's catch from the pack to check 14, which the ladder runs 1-in-64. | Per change, a bench with a pre-stated line. |

### 3.1b Regime 1, CPU ladder. SPEED

| # | Option | Expected gain | Evidence | Red team | Cost | Risk | How to prove it |
|---|---|---|---|---|---|---|---|
| 9 | **Order-preserving CPU message sum** (E1a): a chunked `index_add_` with no per-layer [E, 128] fp32 temporaries. Plus the **edge-table hoist** (E2), shipped with #1–2 as one package. | **2.06–2.51× per leaf** at 8 threads, 1.59–1.76× on a 2-core VPS stand-in, 2.45× on one thread (MEASURED). **Deploy per stone: 2.42× at 64 sims, 2.58× at 256, 1.92× on the stand-in** (MEASURED, 0 differing stones). E2 alone ≤ 2 %. | E's M1: the message pass is 85 % of the CPU forward. E's M4 and M5. Bit-identity proven from the torch 2.11 source and measured on every pop. | CONFIRMED (E1a, E3); E2 PLAUSIBLE | Small–medium. | LAW-06: it must be THE CPU branch for server and trainer alike, with per-thread buffers. The forward then stops scaling past 4 threads (~60 % serial); per-chunk dispatch and the readout (14 %) come next. | A `torch.equal` pin: `tests/model/test_forward_parity.py`'s 1e-6 golden would pass a reordered sum. A CPU bench line ≥ 2× per leaf at B 4–8. Re-read on the real VPS, where both ladder bots share 4 vCPUs. Drop E2 if the bench cannot resolve it. |

### 3.2 Regime 2, self-play throughput

| # | Option | Kind | Expected gain | Evidence | Red team | Cost | Risk | How to prove it |
|---|---|---|---|---|---|---|---|---|
| 10 | **No search at quick-arm decided roots** (D7a), optionally also decided_lost roots. | SPEED for quick-arm roots (target- and move-neutral). STRENGTH-class for decided_lost (it plays a different losing stone). | **+2 to +4 % games/h**; +4 to +6 % with decided_lost roots (ESTIMATED, open loop; a real run +1 to +3 %, §3.5.5). | 10.3 % of run11's positions are decided roots searched only for their row (`search_drive.rs:629-642`). Quick rows are value-only (`losses.py:64-69`); lost roots record no policy target (R377(f)). | PLAUSIBLE | Small, plus a writer change: `refuse_zero_visit_export` (`crates/mantis-selfplay/src/records.rs:182`) refuses a zero-visit export today. | Full-arm decided roots (D7b) are ruled out by R239 / R377(f) / R378(e). | LAW-18 rows first, then interleaved loop A/B pairs. |
| 11 | **The builder (#4), the Gumbel interior selector computed once (C8), the leaner wire (#8).** | SPEED | +4 to +6 %, +0.5 to +2 %, +1 to +4 % positions/h (ESTIMATED, open loop; a real run ×0.6–0.7, §3.5.5). | PERF-3 L2's calibration: −385 µs of worker CPU per leaf gave +5.6 to +17 %. | #4 CONFIRMED; C8 and wire PLAUSIBLE | Medium. | Bit-identical (#4, C8). | An interleaved 2×2 loop A/B on a box. |
| 12 | **The trainer on an SM-limited green-context stream**, against serving contention: K7's arrangement, judged structural by red team A. A GIL-free serving dispatch (A7) was killed on PERF-2's split; run11's events re-open it (§4, §7.2(i)). | SPEED for serving | 0 to +20 % positions/h open loop (ESTIMATED upper bound); a real run 0 to +15 %, sign not guaranteed (§3.5.5). | PERF-2's isolated loop put the trainer's cost on the GPU (device wait +2.66 ms against launch +0.40 ms per pop); A's M4 agrees. run11's events disagree: +1.5 ms launch and +2.65 ms slot/GIL per pop while the trainer steps (§3.5.3). F-44's R365(e) note: the trainer's duty in this regime is unread. | PLAUSIBLE | Medium (a beta API in torch 2.11). | An SM-limited trainer can fall behind run11's steps-per-game cadence, which changes training dynamics. | A box-loop A/B with `gpu_wait` split into CUDA-event device time and GIL wait, and the trainer's step rate read beside it. |

### 3.3 STRENGTH options: the search returns something else

Each needs a strength test at **equal time**: fixed-depth budgets matched to the measured contested-turn wall, as LAW-15
requires. The rulers are X = Six gen30 at 16 nodes and S = Strix at r8, 288 pairs each, with power stated per LAW-19. A
288-pair cell reaches 0.8 power only at |Δ| ≥ ~0.5 logit, so a non-inferiority claim needs an ~800-pair equal-sims
head-to-head at −0.15 logit.

| # | Option | Regime | Expected gain | Evidence | Red team | Cost | Risk | How to prove it |
|---|---|---|---|---|---|---|---|---|
| 13 | **The round-fill leg.** Fix the virtual-loss frame at second-stone nodes (new; red team B, verified by the lead), and refill rounds out of order (B3, lc0). The exact first-collision stop goes with Leg 1 (#8). | 1 | **The frame fix alone: −36 % round trips at 64/stone** (MEASURED: real net, 16 turns, CPU counts). −35 / −20 / −6 % at 64 / 256 / 1 024 by red team B's model (ESTIMATED). **Full rounds would cut 39–40 %** (a bound from measured counters). Frame fix + refill modelled at −41 to −43 % (ESTIMATED). GPU wall: −17 to −20 % of a contested stone after #1 with the serial build, −25 to −30 % after #2 (ESTIMATED); #3 and #5 then lower it. CPU ladder: 0 to −3 %, more after #9 (ESTIMATED). Self-play 0. | `node.rs:78-86` with `selection.rs:155-156`; the module doc says the opposite (`mcts/mod.rs:7-9`). `lead/vl/VL_CHECK.md`. B's counters: 4.1–4.7 leaves per trip; collisions are 78 / 49 / 36 % of the shortfall. | Frame: a CONFIRMED defect. Leg: PLAUSIBLE | Trivial (frame, stop); small (refill). | A deploy-instrument unit change: cells, the in-run gate and the ladder all move. No test pins the sign today. | A unit test pinning the sign at both stone levels; equal-time cells on X and S, or the ~800-pair non-inferiority test; then the speed read: contested wall at 256 ≥ −15 %, abort under −6 %. |
| 14 | **Budget-aware early stop** (B2): stop when the visit leader can no longer be overtaken (lc0 smart pruning). | 1 only (Gumbel's target needs the whole halving schedule) | **−8 / −18 / −23 % of a contested stone's descents** at 64 / 256 / 1 024 (MEASURED); wall about proportional (ESTIMATED). | B's counters: **0 of 156 stones changed their move, before or after the audit** (95 % upper bound ≈ 1.9 %). | PLAUSIBLE: **ruling owed** (the served-sims witness `last_sims == n`; distinguish it from R378(c)'s "the plain head's early end is a defect") | Small. | STRENGTH-class by construction: only the pre-audit leader is fixed. The audit takes its fallbacks in visit order (`tactics_root.rs:48-62, 125-134, 386-398`), and the shared solver table is filled less, so the played move can change (0 observed). It also truncates the visits that analysis reads (§7.2(b)). | A 512-position witness at 256 (bounds flips at ≈ 0.6 %); LAW-18 stops fired and descents saved; an IQR-gated turn bench; a strength screen only if flips appear. |
| 15 | **Within-turn subtree reuse** (B1): one tree per turn; the second stone continues the chosen child (Six, AlphaGo Zero, lc0, KataGo). | 1 | −15 / −22 / −28 % of a contested turn's descents against today (MEASURED chosen-child shares 0.31 / 0.44 / 0.56). **Beyond #6: GPU −9 to −15 % at 256** (ESTIMATED); CPU ~0 beyond #6. | B's measurements; Six's `mcts.cpp:686-691`. | PLAUSIBLE: **ruling owed** (the witness) | Medium. | Overlaps #6, so choose the SPEED form first. The audit's shared table can flip budget-bound verdicts. | Equal-time cells on X and S. |
| 16 | **Self-play Sequential Halving round fusion** (B6). | 2 | Round trips per stone −51 % (exact from the schedule). Positions/h +0 to +20 % (ESTIMATED, open loop; server- and GPU-capped; a real run 0 to +14 %, §3.5.5). | Six fuses its halving reps. F-47: leaves in flight bought +5 % when the server was the bound; red team B says it does not transfer as is, but the transfer must be measured. | PLAUSIBLE: **ruling owed** | Medium. | Breaks the protected `a_gumbel_round_is_exactly_the_halving_phase_wide` and R346(c)'s round definition. Moves targets through the completed-Q at each halving. | A twin with the starvation exams, then panel cells. Rank it against leaves in flight at the re-tune. |
| 17 | **Smaller levers:** MCGS (B5); proof propagation at 1 024-class budgets (B9); a per-game solver table (D1). | 1 (D1 both) | ~0 wall, +0 to +0.1 logit (low prior); speed ≤ −2 to −6 % with strength unmeasured; −0.2 to −0.7 % (ESTIMATED). | CARD-MCGS-DEPLOY (28.6 % table hits); B's revisit split; D's soundness proof. | PLAUSIBLE, low value | Medium–large (B5). | D1 changes played moves (a live-instrument change). | Equal-time cells. |
| 18 | **Prune empty→empty edges** (E4; Strix's `prune_empty_edges`): a new encoding. | Both | 84 % of our edges join two empty cells. Pruning leaves ~2 000 edges per leaf against 13.4 k: the largest single cut to per-leaf GPU and CPU work in both regimes (edge counts MEASURED; effect ESTIMATED). | E's census over 512 positions. | PLAUSIBLE (the next run) | Large: a new row in `crates/mantis-encoding/src/registry.toml`, a schema key, LAW-08 / LAW-11 / LAW-12, gate 11, a new net. | A capacity and strength question. | A fine-tune pair (pruned against an equal-recipe unpruned control, LAW-19), then equal-time cells. |

### 3.4 What the stack buys (derived; factors stated)

**Desktop GPU, 256/stone, mean, concurrency 1.**

| Step | Turn time | How derived |
|---|---|---|
| As configured | 543 ms | MEASURED |
| #1 | 375 ms | MEASURED |
| #2 | 293 ms | MEASURED |
| #3 | 246–261 ms | ×0.84–0.89 |
| #5 | 211–226 ms | −0.9 ms × 39 pops; red team A: still covered after #3 |
| #6 | 169–214 ms | ×0.80–0.95, ruling owed |
| The round-fill leg (STRENGTH) | 118–178 ms at best | ×0.70–0.83: −25 to −30 % after #2 by red team B, less once #3 and #5 cut the per-trip cost (ESTIMATED) |

**Box B.**
- MEASURED on the recorded turn starts: 487 → 220 ms mean at 256/stone with #1 + #2 (2.2×), and 187 → 77 ms at 64/stone
  (2.4×).
- Applied to box B's real-game median at 256/stone (0.95 s as configured), that gives ~0.43 s (derived).
- #3 adds less on box B, whose GPU replays a small pop in ~0.95 ms.

**Against Six's 0.30 s at 512/turn:** that is a per-turn wall at a nominal budget, and Six's 512 means 1.0–1.75 × 512
new visits without reuse (§2.3). The comparison is indicative, not like for like.

**The CPU ladder.** Desktop, 64/stone, 8 threads, mean per stone: 542 → **224 ms** (#1 + #2 + #9, MEASURED), and ~190–
200 ms with #6 (×0.86–0.90). The VPS stand-in, 4 threads: 577 → **301 ms** (MEASURED).

**Rung cells at concurrency 8:** #1 barely applies, #3 rarely applies and #7 does not apply. Their gains are at most
those of #2, #4, #5 and #6.

### 3.5 What a real training run gains (regime 2 in practice)

The operator asked whether these changes also make real training faster, not only single games. This section answers
from three sources:
- **run11's own event stream:** segment 3, 28.4 h, mirrored to the desktop and read there; the box was not touched.
- **A fresh red team** (batch "training"), which re-derived every number from the events and simulated the trainer's
  step rule.
- **A box-B loop A/B,** run under the operator's leave on 2026-10-06 (§3.5.7).

#### 3.5.1 The loop today (MEASURED, run11 on box A)

| run11, 27.4 h after the first hour | |
|---|---|
| Games/h; trainer steps/h | 1 891; 4 508 (1 903; 4 536 without the 4 gate rounds) |
| Steps per game | 2.384 realised against `train.training_steps_per_game` 2.4; the burst cap of 8 dropped 0.66 % of steps |
| Server pops/s; leaves per pop | 90; 36.1. 95.4 % of pops fire at the 32-leaf threshold, 4.6 % at the 10 ms deadline |
| Server per pop: queue wait | 4.93 ms = the wait for leaves (~3.0–3.4) + the GIL-free Rust fuse of the whole pop (~1.5–2), with GIL-held bookkeeping and two GIL re-entries (`bridge/src/inference.rs:355-405`) |
| Server per pop: launch | 4.15 ms, collate 3.75 of it |
| Server per pop: untimed | 1.99 ms: the pipeline-slot acquire and GIL re-entries |
| Server cycle | 11.07 ms |
| GPU wait per pop (retire thread) | 7.40 ms: the residual device wait plus one GIL re-entry, not the forward's length |
| Eval-cache hits | 32 % of served leaves, within one net version. A weight sync every 50 steps (~40 s) clears each shard on its first put of the new version (`queues/eval_cache.rs:174-178`) |
| Gumbel rounds | 3.18 descents per round (2.59 served, 0.59 solver-terminal; 1.76 GPU misses). ~40 rounds per searched stone (full search 97, quick 21), plus the root's own round trip |
| Trainer | 0.547 s per back-to-back step (0.654 s during gate rounds); busy 0.69 of wall |
| In-run gate rounds | 4 in 28.4 h, 711–1 800 s each: 4.5 % of wall. Self-play runs 15–25 % slower while they play |

#### 3.5.2 What binds

**Latency, not a saturated stage.** PERF-3's L0 profile on box A found this, and run11's counters repeat it.
- **The server.** It waits for leaves 27–31 % of its cycle: ~35 % while the trainer is idle, ~23 % while it steps.
  `queue_wait` also contains the pop's Rust fuse, so it is not all waiting. F-47 records the misreading of exactly this
  timer.
- **The workers** are ~85 % blocked on their round trips.
- **The pops.** Each carries ~15–18 workers' rounds. Two pops' worth exceeds the 32 workers, so the loop ping-pongs: the
  next threshold needs leaves that sit in the in-flight pop.

**What a worker pays per round:** its own serial CPU (select, key, build on a miss, expand) plus one round trip (queue,
launch, GPU, retire). **The largest inflation of that round trip today is the trainer** (§3.5.3).

#### 3.5.3 The trainer's interference with self-play (MEASURED; parked by the operator, an open item in §7)

**Method (red team, reproducible from the events).**
- Every `iteration_complete` closes one trainer burst. 39 919 steady intervals were used; the first hour and the gate
  rounds ± 5 min were excluded.
- Busy share = (last step ts − first step ts + 0.547 s) / interval.
- The server's counters were differenced over each interval.

| Trainer busy | Hours | GPU evals/s | Positions/h | Cycle ms | Queue wait | Launch | GPU wait | Untimed |
|---|---|---|---|---|---|---|---|---|
| 0.0–0.3 | 3.09 | 3 957 | 198.7 k | 9.03 | 5.04 | 3.37 | 5.94 | 0.62 |
| 0.4–0.5 | 2.66 | 3 719 | 193.9 k | 9.69 | 4.85 | 3.66 | 6.41 | 1.18 |
| 0.6–0.7 | 1.92 | 3 391 | 178.4 k | 10.66 | 4.83 | 4.02 | 7.12 | 1.81 |
| 0.8–0.9 | 1.35 | 2 957 | 153.1 k | 12.23 | 4.94 | 4.51 | 8.27 | 2.77 |
| 0.9–1.0 | 10.07 | 2 784 | 139.7 k | 13.04 | 4.90 | 4.87 | 8.81 | 3.27 |

**Result.**
- Weighted least squares: GPU evals/s = 4 388 − 1 606 × busy, at a mean busy share of 0.687 and a mean rate of 3 285/s.
- **Trainer-free, self-play would run ~+33 % faster. Time-averaged, the in-process trainer costs ~25 % of self-play
  throughput, and ~37 % while it steps.** These are derived from the fit; its intercept extrapolates past the lowest
  bin's MEASURED 3 957/s.
- Served leaves/s, rounds/s and pops/s agree.

**Confounds checked.**
- At matched interval lengths the split holds: 3 865 / 3 415 / 2 793 evals/s at busy < 0.5 / 0.5–0.8 / ≥ 0.8 for 2.4–3.5
  s intervals.
- The hit share (31.5–33.1 %) and evals per round (1.71–1.80) are flat.
- Graph size runs against the finding: idle intervals carry larger graphs (699 against 645 nodes per leaf).
- The pool drains rows continuously at ~10 Hz, not at game ends.
- **Limit:** the split is observational. The busy share is set by game completions, not randomised.

**The channel is not only GPU time.** With the trainer stepping, the server's cycle grows +4.0 ms:
- launch +1.5 ms;
- the untimed slot/GIL span +2.65 ms;
- queue wait flat.

The retirer's GPU wait grows +2.9 ms on the retire thread, concurrent with the server's cycle. PERF-2's isolated split
(launch +0.40 ms, device +2.66 ms) does not describe run11's trainer, whose step does:
- two microbatches, each with the full semantic collate (`train/coordinator/dispatch.py:204-215`);
- the D6 augmentation;
- a value-mask redraw;
- sampling a 500 k ring.

The retirer's result submit also holds the GIL while it assembles ~36 policies and wakes the waiters one by one
(`bridge/src/inference.rs:412-487`, no `py.detach`).

**What is known about remedies** (none tried here). PERF-2's L4 found that a highest-priority serving stream does not
move it: −1 % positions/h against a +10 % line (`CARDS.md:140-146`).
- **Read the GPU/GIL split first,** with a py-spy `--gil` or schedstat sample in run11's regime, as the 2026-09-11
  investigation did. Choosing among the remedies below depends on it. Box A is live and off-limits, so this needs the
  operator's leave for a read-only sample there, or a box-B replica of the loop. The same split decides A7, a GIL-free
  serving dispatch: its own revival line (≥ 1 ms per pop of non-device wait) now looks met.
- **A separate trainer process** (CUDA MPS, or another GPU) removes the GIL channel entirely. It changes the run's
  process layout and actor sync (a cost, and a design question for the overarching agent).
- **A GIL-free trainer collate.** The trainer runs the full semantic collate on every microbatch by code
  (`dispatch.py:204-215`). The protected 1-in-1 pins read the eval worker and the self-play pool
  (`tests/eval/test_f816_37_instrument.py:85-106`), so a port that keeps the checks needs no ruling. A cheaper trainer
  posture would need one.
- **An SM-limited trainer (#12)** reaches only the device share, and its sign is not guaranteed. A slower trainer raises
  its duty, the burst rule then drops steps (~2.31 per game at 0.80 duty), and it holds the GIL longer per step.
- **The retirer's result submit** could move out of the GIL (a smaller, separate change).
- **The card's other untried options:** pacing the trainer against serving, and the trainer's kernel sizes
  (`CARDS.md:140-146`).
- **#18,** the pruned encoding, shrinks the trainer's bytes as well as serving's.

**The ceiling.** Removing all of the interference gives ~+33 % games/h. Keeping up would need a trainer duty of ~0.92.
The burst rule drops the excess, so steps/h rise ~+26 % at a realised duty of ~0.86. Beyond it, the trainer's own step
is the lever (§3.5.4).

**History.** It is the same order as CARD-PERF-TRAINER-CONTENTION's "a third". PERF-2 read 235 k positions/h alone
against 149–156 k with a fixed-rate trainer, −34 to −37 %; run11's while-stepping cost (−37 %) matches it, now in
production's own regime. F-44's R365(e) note said this regime's trainer duty was unread; it is now read: 0.69.

**Box B** measured it directly, the trainer switched off against on: −25.1 % positions/h and −22.1 % GPU positions/s,
through the same channels (§3.5.7).

#### 3.5.4 How a self-play gain reaches a run's training steps

**The rule.**
- The trainer follows games; self-play is never throttled. O5 sleeps 0.1 s while no game is new
  (`train/coordinator/step.py:444-448`).
- O6 runs `min(floor(carry + 2.4 × new games), 8)` steps (`:450-456`, `train/mixing.py:11-20`).
- The carry keeps only the fraction, so steps above the burst cap are dropped, never owed.

**Simulated** with the measured step times and per-worker game durations; it reproduces today at 2.367 steps per game
and 0.685 duty.

| Games/h × | 1.1 | 1.2 | 1.3 | 1.5 | 1.7 | 2.0 |
|---|---|---|---|---|---|---|
| Steps/h × | 1.090 | 1.173 | 1.245 | 1.353 | 1.414 | 1.448 |
| Steps per game | 2.34 | 2.31 | 2.27 | 2.14 | 1.97 | 1.71 |
| Trainer duty | 0.75 | 0.80 | 0.85 | 0.93 | 0.97 | 0.99 |

**The closed-loop discount.**
- Every extra game also brings trainer work, which slows serving: R = R0 (1 − c·d) with the measured c = 0.366 and d =
  0.687. So d ln R / d ln R0 ≈ 0.75, and the burst rule converts at ~0.9 for small gains.
- **An open-loop gain therefore reaches a real run's steps/h at ~×0.6–0.7.**
- PERF-3's L2 (+9.3 % positions/h, the calibration used for the self-play estimates) ran its trainer open loop at a
  fixed 1.2 steps/s. In a run it reads ~+6.8 % games/h, ~+6.2 % steps/h.

**Side effects a decision should see.**
- Faster self-play lowers the realised steps per game: 2.14 at 1.5×. That moves the run's sample reuse without a
  decision. STATE's [1.2, 2.4] envelope bounds the configured `training_steps_per_game`, not the realised rate. Raising
  `max_train_burst` to hold it is a re-mint with a ruling.
- `eval_interval` counts steps, so faster training makes gate rounds more frequent per hour. At +30 % steps/h they take
  ~5.8 % of wall instead of 4.5 %.
- The cap falls further when serving slows the step: +20 % during gate rounds.

**The cheapest cap lever is the trainer's ring sample.** It runs GIL-free under `py.detach`
(`bridge/src/hexg.rs:152-159`) on 1 thread in production (`config/resolve/sample_threads.py:69`). Widening it shortens
the step without GIL or GPU interference. It does take CPU from the workers, which the reservation
(`sample_threads.py:11-13`) exists to prevent, so it faces the same CPU question as #19.

#### 3.5.5 Lever by lever, for a real run

Read the real-run column. The open-loop column is what an A/B with a fixed-rate trainer would show.

| Lever | Real run, games/h (steps/h ≈ ×0.9) | Open loop | Basis | Red team |
|---|---|---|---|---|
| The single-game fixes: #1, #3, #7, #9, #13, #14, #15 | **≤ 0.3 %**; ceiling 0.88 % if gate rounds cost nothing | — | The rounds took 4.5 % of wall at a 19.6 % self-play deficit. At concurrency 8, #1, #3 and #7 barely apply; #9 is CPU-only. | CONFIRMED |
| #2 as specified (the deploy knob) | 0 | 0 | `resolve_leaf_build_threads` is consumed only by the eval child (`run.py:643`). | KILLED as a training lever (CONFIRMED for regime 1, §3.1) |
| **#19 (new): a worker builds its round's misses in parallel** | **≤ 0 on box B** (MEASURED: −1.2 % GPU positions/s, −0.9 % positions/h over four pairs). The red team's prior was +1 to +6 % | trainer off: −2.5 % / −4.8 % (MEASURED) | §3.5.6, §3.5.7 | **KILLED** in this form (box B); box A unmeasured |
| #4, the byte-identical builder | +2.5 to +4.5 % | +4 to +6 % | −45 % build MEASURED; L2's calibration | CONFIRMED |
| #10, no search at quick-arm decided roots | +1 to +3 % | +2 to +4 % | Quick-arm decided roots are ~7.7 % of positions; proven roots are cheap searches already. | PLAUSIBLE |
| #8, a leaner wire; C8, the selector computed once | +0.7 to +3 %; +0.3 to +1.5 % | +1 to +4 %; +0.5 to +2 % | Part of the fuse sits inside the queue-wait timer. | PLAUSIBLE |
| #5, the checks while the forward runs (self-play's share) | 0 to +1.5 % | — | Only the Python semantic layer moves (~1 ms per B-36 pop). It keeps the GIL unless ported (F-46). | PLAUSIBLE, ruling owed |
| #12, an SM-limited trainer | 0 to +15 %, sign not guaranteed | — | Reaches only the device share of the interference. | PLAUSIBLE |
| #16, Sequential Halving round fusion | 0 to +14 % | 0 to +20 % | Doubles leaves in flight per worker; capped by the retire/GPU stage under the trainer (73 % busy while it steps). | PLAUSIBLE, STRENGTH-class |
| #18, pruning empty→empty edges | The largest per-leaf cut, and the one lever that also shrinks the trainer's footprint, so it converts at or above its open-loop reading | — | 84 % of edges | PLAUSIBLE (next run) |
| **The trainer's interference** | up to +33 % games/h, ~+26 % steps/h | — | §3.5.3 | MEASURED; parked |
| Widening the trainer's ring sample | raises the cap (no gain until the trainer binds) | — | GIL-free, 1 thread today | PLAUSIBLE |
| Re-tune knobs: leaves in flight, the collector threshold, the cache size and the sync cadence | ≤ +15–19 % | ≤ +20–25 % | GPU utilisation is unread in run11. PERF-3's comparable loop read 74–78 %, and box B's loop ~60 % with the trainer and 48 % without (§3.5.7). The cache clears at every sync, so its size and the cadence are linked. | knob, for the sweep |

#### 3.5.6 #19 in detail (proposed in this leg)

**Today.** A self-play worker handles each leaf of its round in turn (`runner/search_drive.rs:325-413`):
1. it computes the key (`:337`);
2. it reads the shared cache (`:346`);
3. on a miss, it builds the axis graph on its own thread (`:350`).

It then submits all misses at once (`:371`) and stores the results after they return (`:400-413`).

**The change.** The misses of a round are built on up to N threads through `par.rs`'s `map_in_order`, in submission
order. Rounds with one miss stay serial.

**Identity.** It holds at the worker level:
- `map_in_order` returns index order;
- the cache `get` has no side effects (FIFO eviction, `queues/eval_cache.rs:153-162`);
- the puts stay after the results.

Run-level identity does not exist either way: the cache is shared across 32 workers and timing-dependent.

**Prerequisites.**
- `par.rs:34` must carry the panic payload. Otherwise a `verify_contract` panic turns from a counted worker panic
  (`runner/spawn.rs:30-39`) into a generic inference failure, and a different LAW-18 counter fires.
- A helper cap: 4 keeps ~85 % of the saving.
- A persistent per-worker helper, if spawning per round costs (~2 600 thread spawns/s at full use).
- A LAW-18 row: rounds built wide, and helpers per round.

**History it must answer.**
- NIGHTRUN-1 E1 (R325) excluded self-play on purpose: "each is already one of `n_workers` threads building its own
  leaves, so widening one takes threads from the others and double-counts the reservation"
  (`config/resolve/leaf_build_threads.py:9-11`).
- The 2026-09-11 investigation called the serial round build "OFF the critical path today … matters only once RTT binds"
  (`PERF_INVESTIGATION_2026-09-11.md:378`).
- Both came from run6's server-saturated regime (F-47). PERF-3 L0 and run11 show the loop is latency-bound now, so the
  premise has changed. Only a measurement can say whether the conclusion changes too.

**The estimate (ESTIMATED).**
- 1.76 misses per round × ~40 rounds (70.3 misses) × 0.622 ms gives 43.7 ms of serial build per stone. Fully parallel
  with no overhead, the critical path keeps ~one build per round: −0.21 to −0.23 ms per served leaf.
- SMT and burst crowding cut that to −0.08 to −0.17 ms. A retire wakes the 15–18 workers of one pop together: ~35–45
  runnable threads on 16 physical cores.
- The saving concentrates in the few wide rounds: 16- and 8-wide rounds, ~5 of ~40 per stone, hold ~60 % of it.

**Box B's reading** (§3.5.7): −1.2 % GPU positions/s over four pairs, and −2.5 % with the trainer off. It is killed in
this form.

#### 3.5.7 The box-B loop A/B (MEASURED, 2026-10-06 17:49–19:15 CEST)

**Setup.**
- **Host:** box B (RTX 5070 Ti + Ryzen 9 5900XT, 32 threads), in its own tree: a `git archive` of 3ee61745 plus the
  env-gated #19 patch (`boxb/round_build.patch`).
- **The loop:** configs/run11a2.yaml with 32 workers, serving run11@162k and training on its ring.
- **The trainer, closed loop as in production:**
  - `_steps_budget` at 2.4 steps per game, burst 8 and the fill ramp;
  - sample threads as production resolves them;
  - served weights synced every 50 steps.
- **Each run:** 2 min of warm-up, then six 1-min windows. A discarded priming run filled the compile cache first.
- **Arms:** A is the serial round build (`MANTIS_SCRATCH_ROUND_BUILD_THREADS` = 1, today's path); B uses up to 4 helper
  threads per round.
- **Order:** A B B A A B B A with the trainer, then A and B with it off.

| Run | Arm | Trainer | Positions/h | Games/h | GPU positions/s | Steps per game | Server cycle ms | GPU util |
|---|---|---|---|---|---|---|---|---|
| p1 | A | on | 156 520 | 1 890 | 3 064 | 2.35 | 12.02 | 62 % |
| p1 | B | on | 153 060 | 1 860 | 3 039 | 2.32 | 12.78 | 59 % |
| p2 | B | on | 152 860 | 1 780 | 2 946 | 2.37 | 12.99 | 63 % |
| p2 | A | on | 155 160 | 1 830 | 3 003 | 2.42 | 12.26 | 60 % |
| p3 | A | on | 153 870 | 1 700 | 2 979 | 2.40 | 12.33 | 60 % |
| p3 | B | on | 150 000 | 1 790 | 2 922 | 2.34 | 13.18 | 60 % |
| p4 | B | on | 156 920 | 1 810 | 3 021 | 2.33 | 12.42 | 61 % |
| p4 | A | on | 153 040 | 1 750 | 3 020 | 2.41 | 12.29 | 61 % |
| nt | A | off | 206 540 | 2 490 | 3 873 | — | 9.58 | 48 % |
| nt | B | off | 196 690 | 2 330 | 3 777 | — | 10.24 | 48 % |

**#19 does not pay in this form.**
- Over the four pairs, B against A:
  - GPU positions/s −1.2 % (pairs −1.9 % to 0.0 %), the steadier metric;
  - positions/h −0.9 % (−2.5 % to +2.5 %);
  - games/h +1.1 %, which rides game length.
- With the trainer off: −2.5 % GPU positions/s, −4.8 % positions/h.
- The red team's central +2.5 % lies outside every pair on the steadier metric.
- **The mechanism visible in the counters:**
  - arm B's pops are larger (38.2–38.6 positions against 37.0–37.2);
  - its server cycle is 0.5–0.9 ms longer.
  - Helpers that finish rounds together make arrivals burstier and crowd the Python server and retire threads, which
    eats the build saving.
- **Verdict: KILLED in the spawn-per-round form on box B.** It is unmeasured on box A, whose CPU differs. A persistent
  per-worker helper is the only form left, at low priority.

**The trainer's interference, read directly on a second host.**
- Arm A with the trainer against without it: positions/h 154.6 k against 206.5 k (−25.1 %); GPU positions/s 3 017
  against 3 873 (−22.1 %).
- Per pop with the trainer:
  - launch +1.45 ms;
  - the untimed span +1.2 ms;
  - GPU wait +2.0 ms;
  - queue wait flat.
- That is the size, and the channels, of run11's own split on box A (§3.5.3). Box B confirms it with the trainer
  switched off, not only by regression.

**Other readings.**
- **GPU utilisation:** ~60 % with the trainer and 48 % without. The GPU is not the bound of box B's loop.
- **Box B's trainer step:** 0.63–0.66 s, against box A's 0.547 s.
- **With the trainer,** box B's loop runs 1 700–1 890 games/h at 2.32–2.42 steps per game. That is close to run11's live
  1 891 and 2.384, so the closed-loop harness reproduces production's coupling.

**Records:** `mantis-records/search-perf-1/boxb/` holds the scripts, the patch, the per-run `loop.json`, the single-game
`r1_*.json` and the logs.

#### 3.5.8 How to prove a training lever

- **The A/B.** A box loop A/B at run11's config, with the trainer closed loop: 2.4 steps per game, burst 8, the
  production ring and sample threads. A fixed-rate trainer, as PERF-3 used, hides the feedback this section is about.
- **Interleaved pairs**, since two runs of one regime differ by ~5 % (CARD-PERF-LOOP-SPREAD).
- **What to read:**
  - positions/h beside games/h, since game length drifts (91.6 → 85.9 positions per game inside segment 3);
  - the per-burst busy split, as the interference meter.
- **Box A's own card** for the final number, because Ada and Blackwell differ (PERF-ADA).

### 3.6 The two changes that alter what the search returns, in detail

The operator asked for these two to be set out in full. Both can make the single-game search faster, and both change
which positions it looks at. A speed number alone cannot clear either one: each needs a ruling and a strength test.

#### 3.6.1 #13: the virtual-loss sign at second-stone nodes, plus refilling short rounds

**How the deploy search batches today.** Each GPU round trip carries up to 8 positions.
- `select_leaves(8)` runs up to 8 descents from the root. Each descent places a temporary **virtual loss** on its path,
  so the next descent sees that path as worse and goes elsewhere.
- When the results return, the virtual losses are removed and the real values backed up. Virtual loss never biases the
  final statistics; it only decides which positions share a round trip.

**The defect** (found by red team B, verified by the lead in `lead/vl/VL_CHECK.md`).
- `q_value_vl` subtracts the penalty in the child's own frame: (w − vl·penalty) / (n + vl) (`mcts/node.rs:78-86`).
- `puct_score` then negates the child's value when the parent's player is placing a second stone, because the child
  position belongs to the opponent (`mcts/selection.rs:155-156`).
- So at every second-stone parent the penalty is negated into a bonus. A pending child looks like a win to the player
  choosing, and the next descents run straight back into it.
- The module doc says virtual loss discourages (`mcts/mod.rs:7-9`). No test pins the sign, which is how it survived.

**What it costs today.**
- A descent that reaches a pending position (a collision) undoes its own virtual loss and rewinds. The tree is then in
  the same state, so every further attempt repeats the same path and collides again. The select call ends at its first
  collision, after burning its remaining attempts (`selection.rs:271-292, 336-412`).
- Contested stones therefore carry 4.1–4.7 network positions per round trip, not 8.
- Collisions are 78 / 49 / 36 % of that shortfall at 64 / 256 / 1 024 per stone. The rest is in-search table hits and
  solver-ended descents taking batch slots.

**The change.**
1. **Apply the penalty in the chooser's frame** at a second-stone parent: (−w − vl·penalty) / (n + vl). One line.
2. **Refill (B3, lc0's out-of-order evaluation).** When a select call comes back short because table hits and solver
   terminals took slots, keep selecting, with a bounded number of attempts, until the round carries 8 network positions.
3. **The exact first-collision stop** belongs with Leg 1 (§3.1 #8): it changes nothing returned, only wasted CPU.

**What it does to speed.**

| | 64/stone | 256/stone | 1 024/stone |
|---|---|---|---|
| Round trips, sign fix alone (real net, MEASURED counts: 16 turns, CPU) | −36 % (14.1 → 9.0 per turn; network positions per trip 3.42 → 5.35) | — | — |
| Round trips, sign fix alone (red team B's model, ESTIMATED) | −35 % | −20 % | −6 % |
| Today's network positions packed in full trips of 8 (B's counters; a reference, not a bound on the leg) | −39 % | −40 % | −40 % |
| The same from S1's `trips_if_filled_to_8` | −48 % | −42 % | −40 % |
| Round trips, sign fix + refill (modelled, ESTIMATED) | −43 % | −42 % | −41 % |

- **GPU turn wall:** −17 to −20 % of a contested stone after the wake fix with today's serial build, and −25 to −30 %
  after #2's parallel build, which moves build cost from each position to each trip (ESTIMATED, red team B). #3 and #5
  then make each trip cheaper and lower it.
- **CPU ladder:** 0 to −3 % today (ESTIMATED), more after #9 (the per-position cost there then varies with batch size).
- **Self-play:** none. Gumbel's interior selector ignores virtual loss, and its rounds are one forced descent per
  candidate.

**Why it is a STRENGTH change, and which way it may go.**
- **Different rounds:** each round would carry different positions: more spread out, chosen with staler statistics (the
  pending ones penalised, as intended).
- **The known trade-off:** classic virtual loss trades search quality per playout for parallelism. At equal playouts a
  round of 8 spread-out positions can be slightly worse than a narrow round of fresh ones, most at low budgets.
- **At equal time,** the fewer round trips buy more playouts. RUN11-GO's equal-time cells show playouts are worth a lot
  against Strix (0.391 → 0.637 at 1 s per turn when the wait was removed), but that cell bought ~2.9× the playouts (256
  → 754 per stone). #13's −17 to −30 % of wall is ~1.2–1.4×. Against Six the same wait removal read 0.047 → 0.045.
- **Expected:** neutral to slightly negative at equal playouts, positive at equal time. Unmeasured.

**Every PUCT deploy search moves:** cells, the in-run gate, the ladder and the probes. The strength series gets a break,
as with R378(b)'s unit change. run11's tree on box A must not move, so this applies to new cells and future runs only.
The desktop's run11 X-cell unit imports the main checkout's extension, so #13 must not reach that checkout while run11's
series runs. The parent cells would need re-reading with the new head to re-baseline.

**How it would play out.**
1. **A ruling:** the deploy instrument changes.
2. **A unit test** pinning the sign at both stone levels: a pending child must lose PUCT score under its own parent, at
   a first-stone and at a second-stone parent.
3. **The strength test, either of:**
   - equal-time cells on X (Six gen30 at 16 nodes) and S (Strix at r8), 288 pairs each, with fixed-depth budgets matched
     to the measured contested-turn wall against the Leg-1 head, as LAW-15 requires. At 288 pairs, 0.8 power reaches
     only |Δ| ≥ ~0.5 logit. At the expected effect (< 0.5 logit) a null is therefore void under LAW-19; only a win
     clears it. Strength claims read a panel of nets (LAW-19), e.g. run11's 108k and its neighbours, not one net;
   - an ~800-pair head-to-head at equal playouts for non-inferiority at −0.15 logit.
4. **LAW-18 counters:** overlaps per select call, network positions per round trip.
5. **The speed read,** with its LAW-09 lines: contested wall at 256/stone −15 %; abort if the saving is under 6 %.

**Cost:** trivial (the sign), small (the refill), small (the test). The cells take a few hours of the desktop 3070 each.

#### 3.6.2 #15: one tree per turn (the second stone continues the first stone's chosen subtree)

**Today.** A turn is two independent searches. After stone 1 is chosen, `tree.new_game(board)` discards the tree, the
in-search table and the solver table (`mcts/mod.rs:179-200`, `tactics_wiring.rs:282`). Stone 2 then starts from an empty
root.

**The change.** Keep the chosen first stone's subtree and make it stone 2's root. Its visits, values and expanded
children carry over. Six does this within the turn (`mcts.cpp:686-691`: phase 2 runs until the chosen child's visits
reach N), as do AlphaGo Zero, lc0 and KataGo across moves.

**How much carries over** (MEASURED, B's counters, on contested first stones: n = 20 / 19 / 10). Stone 1's search puts
31 / 44 / 56 % of its visits into the stone it chooses, at 64 / 256 / 1 024 per stone.

**What it does to speed.**
- −15 / −22 / −28 % of a contested turn's descents against today (derived from the MEASURED shares above, in the top-up
  form below).
- **Beyond #6** (which already removes the repeated network evaluations): −9 to −15 % on the GPU at 256 after the wake
  fix (ESTIMATED), less after the round-fill leg. The extra comes from the descents, tree building and round trips.
- About 0 extra on the CPU ladder, where evaluations dominate and #6 already removes the repeats.
- **Self-play:** none from this change.

**Two forms, both STRENGTH changes.**
- **Top-up, as Six does.** Stone 2's budget of N counts the inherited visits, and the search adds only N − V1. It is
  faster at the same total visits. But the inherited visits were gathered while stone 1's root priorities steered the
  search, so the visit distribution under the new root differs from a fresh search's.
- **Extra.** Stone 2 adds N new visits on top of the inherited ones. Same cost as today, more visits, probably stronger.
  This is a strength lever, not a speed lever.

**Rulings it needs.**
- **What "256 per stone" means:** whether inherited visits count. R378(c)'s served-sims witness (`last_sims == n`) has
  to be restated for a re-rooted search.
- **The tactics state:** whether the in-search table and the solver table persist for the turn. Keeping the solver table
  is D1's per-game table in miniature. The stone-2 audit would then run over a table with stone-1 contents, so a
  budget-bound audit verdict can flip (the same caveat as #14).

**Against #6, the per-game cache.**
- #6 removes the repeated evaluations with identical moves on the GPU path. The operator allowed it in this session.
- #15 removes more (the descents and the tree work) and gives stone 2 a head start, but it changes moves.

The recommended order is #6 first, then #15 as a strength test.

**How it would play out.**
1. **A ruling** on the unit and the witness.
2. **The bridge** gains a re-root call that keeps the pool, and the deploy head starts a tree per turn, not per stone.
3. **LAW-18 counters:** inherited visits at the stone-2 root; audit verdicts under a persisted table.
4. **Equal-time cells** on X and S under the same power rule as #13 (a null at < 0.5 logit is void at 288 pairs), or an
   ~800-pair equal-playout head-to-head, non-inferior at −0.15 logit.
5. **The speed read** at 256/stone after #6: contested wall −9 %; abort under −4 % (ESTIMATED line, set from the range
   above).

**Cost:** medium (the bridge API and the deploy head).

**Related ideas already killed:** cross-turn reuse (B7: previous-turn evaluations under the new root are 2.4 % at
256/stone) and a turn-scoped table with deferred backup (B4, which #6 duplicates).

## 4. Killed ideas, with the reason

| Idea (source) | Why it is dead |
|---|---|
| TensorRT, or ONNX Runtime's CUDA EP, for our GNN (A) | A custom op (`mantis::gine_message_sum`) over ragged graphs needs an export path. LAW-06 wants one implementation shared by server and trainer. The GPU is not the bound: 17–21 % busy as configured. F-21 orders the remedies if the forward ever binds: `torch.compile`, then a smaller net, then quantised eval. The first is measured at 6–8 % (A5, below). |
| fp16 serving (A) | fp16 GINE sum aggregation overflows (`inference_server.py:417`); LAW-06 pins bf16 with fp32 aggregation. |
| Serving the compiled trunk on eval engines (A5) | MEASURED only 6–8 % faster at small pops (M1). That is under its own 15 % line, about 0 on #3's smaller buckets, and it moves served outputs (root values up to 0.0101). |
| A GIL-free serving dispatch for self-play (A7) | PERF-2's own box-loop records put the trainer's cost on the GPU (device wait +2.66 ms against a GIL-sensitive launch +0.40 ms per pop); A's M4 agrees. **RE-OPENED:** run11's events supersede that basis for run11's regime (+1.5 ms launch and +2.65 ms slot/GIL per pop while the trainer steps, §3.5.3). A7's own revival line (≥ 1 ms per pop of non-device wait) now looks met; §7.2(i)'s GPU/GIL split decides it. |
| Dropping the per-stone `torch.cuda.empty_cache()`; list-decode tweaks alone; pinned-memory changes (A's K4–K6) | MEASURED at 0.02 ms per turn in the bucketed steady state; the decode and pinned copies are tenths of a ms per pop. |
| ONNX Runtime or OpenVINO for the CPU forward (E5); int8 or bf16 on the CPU (E6); `torch.compile` of the whole CPU trunk (E7) | The same export problem, and a second implementation. The CPU bound is memory traffic over [E, H] temporaries, not GEMM FLOPs, and #9 removes it inside torch. Quantisation would change served numbers for little. |
| The Inductor-compiled slot sweep (E1b) | MEASURED: 1.46× per leaf at B8 over #9's chunked sum, but 0.87–0.99× per stone. Its order holds only by an Inductor threshold, and it needs a compiler on the VPS. (Red team E withdrew its "LAW-06 hard block": the op's backward recomputes from its inputs.) |
| A turn-scoped table with deferred backup (B4) | A duplicate of #6 restricted to one turn; #15 contains its immediate form. |
| Cross-turn subtree reuse (B7) | MEASURED: previous-turn evaluations under the new root are 2.4 % at 256/stone, under B's own 3 % drop line. At 1 024 the 5.7 % is already served by #6's cross-turn hits. |
| Incremental axis-graph construction from the parent position (INCR-GRAPH, F-19's registered candidate) | Parked by expert C: its pre-registered falsifier, F-19's own inequality, fails from the root; a cached-parent variant does not fit in memory; and #4 cuts the build 45 % without touching the byte contract. |
| The two-phase leaf solve on a helper thread (D6) | It fails its own gate: the whole tactics area is ≤ 7 % of a turn (S1). A phase-2-proven leaf would also be cached in the in-search table as non-terminal (`backup.rs:502-508`). |
| D's own kills, upheld: Six's leaf budget 2/64; a wall-clock cap on the root solver; a Strix-style post-eval override; the audit's k and m as speed levers; any speed lever in the self-play solver | 2/64: −28 % proofs and +10.2 evaluations per stone. A clock cap breaks determinism (TACTICS_DESIGN §4.1), and Six's cap is inactive under `go nodes` anyway. The Strix override pays an evaluation per proven leaf. The audit is ~1 ms per turn. The self-play solver is ~1 % of worker wall. |
| Larger leaf batches on the strength of Cazenave's batch-MCTS table (B's citation) | That table measures a two-tree design whose main tree keeps ~24 nodes from ~200 inferences. It is no evidence about our leaf batch. |
| Knob values, out of scope for this leg (operator): `selfplay.n_workers` 32 → 48–64 (C7); leaf solver 3/128 or 4/256 (D2, D8); a top-40/64 child cap (B8); the ladder at 256/stone after the CPU fixes (E8) | Values, not structure: re-tune them in one sweep after the structural changes land. On C7: F-47 found more leaves in flight bought only +5 % when the server was the bound. On B8: the chosen move's prior rank has median 0 and is never ≥ 40, and the root mass beyond the top 40 is 0.05–0.08 %, so a cap would rarely bind at the root. Root mass says nothing about interior refutations, and F-40 found the net ~0-prior-blind on exactly those. E8's premise is MEASURED false: after #9 a 256-sim stone still costs 1.3× (desktop) and 1.7× (the VPS stand-in) what today's 64-sim stone does. |
| ESTIMATED gains that leaned on the biased first wait-0 probe ("2.2–3.5×", C's "−55 %", D's "10–15 % of a turn") | That probe paired ours with Strix at 64 for every budget. The clean A/B is 1.6–1.8× with solvers off and 1.75–2.4× with solvers on. Every such number is corrected above. |

C6, the serve-first checker posture, is not killed. It stays PLAUSIBLE and is folded into #5 (§3.0).

## 5. Recommended next legs, with their gates

Gains below are desktop, 256/stone, concurrency 1, unless stated. The operator's word from this session (§7.1) informs
the order.

**The pre-registered lines.**
- Every speed change takes one commit and one IQR-gated bench (LAW-09).
- Each line below is stated before the bench.
- A change aborts if it reads under its abort line.

**The hosts.**
- **Desktop measurements** run under the cell lock, as in this leg.
- **The sm_89 reads and the self-play loop A/Bs need a box.** Box A is live and off-limits, so the host is the
  operator's call.

  | Option | Cost | Leaves for later |
  |---|---|---|
  | A short rental | about 1 box-hour for the sm_89 reads (ESTIMATED); a few hours per loop A/B set | nothing |
  | Box A between runs | no rental | the box-A reads, until run11 ends |
  | Box B | no new rental; its rent runs until the operator destroys it | Blackwell + 5900XT, not box A's Ada + 9950X |

### Leg 1, DEPLOY-SPEED-1: the free fixes (regime 1, SPEED; the operator: go ahead)

**Contents.**
- #1, the wake on submission.
- #2, the parallel leaf build, on standalone single-game hosts only. Self-play's reservation and the in-run eval beside
  it stay as they are.
- #3, small buckets, with a ≥ 1 024-node floor.
- #4, the byte-identical builder, which also feeds Leg 5.
- #9, the order-preserving CPU message sum as the one CPU branch for server and trainer. The edge-table hoist (E2) is
  added only if its bench resolves it.
- The exact first-collision stop. It is not in the operator's list (§7.1 item 2). Red team B reads it as
  output-identical (the same positions and tree; only a mean-depth statistic moves). It is included on the overarching
  agent's call.

**Prerequisites:** `par.rs:34` carries the panic payload before #2 runs wide.

**Gates.**
- **Identity:**
  - deploy-24 byte-equal moves and root values at concurrency 1, GPU and CPU, for every change except #3;
  - `served_sims_exact`, the protected set and the full gate set green;
  - for #9, a `torch.equal` pin (the 1e-6 forward-parity golden would pass a reordered sum).
- **#3 against the served path's own spread:** at the ≥ 1 024 floor, on each card that serves cells or the gate: 0
  differences, or differences inside production's own pop spread with 0 argmax or sign flips.
- **LAW-18 counters:** the wake's fire count; captures and rung per pop.
- **LAW-09 lines (line; abort under):**

  | Change | Line | Abort under |
  |---|---|---|
  | #1 | turn −25 % | −12 % |
  | #2 | a further −15 % | −7 % |
  | #3 | −8 % (desktop; box B's ~1 ms replays leave it less to gain) | −4 % |
  | #4 | build −40 % (criterion); loop +3 % (box A/B) | build −30 %; loop +1 % |
  | #9 | ≥ 2× per leaf on the CPU at B 4–8 | 1.5× |
  | The collision stop | set after a `select_leaves` profile (profile first); bounded by `select_leaves`' 4.5–6 % of a turn | — |

- **Re-reads:**
  - #9 on the real VPS, where both ladder bots share 4 vCPUs;
  - an X cell at concurrency 8 for outcome identity. The wake barely applies there, so this is a witness, not a blocker.

**Expected.**
- **Desktop GPU** at 256/stone: 543 → 293 ms MEASURED through #2; ~246–261 ms with #3 (derived).
- **Box B,** MEASURED on the same turn starts at concurrency 1:
  - 487 → 220 ms at 256/stone (2.2×);
  - 187 → 77 ms at 64/stone (2.4×).
  - Applied to box B's real-game median at 256/stone, 0.95 s → ~0.43 s (derived).
- **The CPU ladder:** 2.4× per stone at 64 sims (MEASURED), 1.9× on the VPS stand-in.

### Leg 2, DEPLOY-CACHE-1: the per-game cache, then the early stop (regime 1)

**#6, the per-game exact cache.** The operator allowed it; the ruling text is owed (§7.2(a)).
- **Contents:**
  - one cache per player and per game;
  - exact keys;
  - cleared at a new game;
  - dedup of keys already pending in a batch.
- **Gates:**
  - same moves on 200 positions: expected 100 % on the GPU's served path;
  - on the CPU, a same-move rate (the eager CPU value is batch-variant at ≤ 7.2e-7), and the turn −10 % or better;
  - the all-hit round-trip share counted, for the GPU size (−5 to −20 %);
  - one X-cell re-read, ≥ 99 % of games move-identical;
  - LAW-18: hits and GPU evaluations.

**#14, the budget-aware early stop.** The operator found it interesting, with the analysis question answered in §7.2(b).
- **Form:** a per-use switch, set per consumer (§7.2(b)):
  - the ladder ON, with its receipt stamping the switch;
  - the dash's Analyzer view OFF;
  - rung cells and the in-run gate OFF, because their per-ply records carry visits and the series' unit;
  - equal-time probes OFF, unless their time-matching is redone with the stop on (then stamped).
- **The ruling:** the served-sims witness, kept distinct from R378(c)'s defect.
- **Gates:**
  - a 512-position witness at 256/stone, which bounds flips at ≈ 0.6 %;
  - LAW-18: stops fired and descents saved;
  - an IQR-gated turn bench;
  - a strength screen only if flips appear.

### Leg 3, CHECKS-OFF-PATH: #5 (SPEED; step (a) regime 1, step (b) both; wanted once proven safe and free)

**Contents:** §7.3's design.
- **(b) first:** checks 15–16 move into the Rust pack's gather pass, GIL-free, still before the launch.
- **(a) then, single-game engines only:** check 14, already GIL-free, runs on the retire thread between the launch and
  the release. Self-play keeps check 14 on its checker thread.
- No new thread, lock or wait edge. A failed check fails its pop through the existing failure path.
- A check-14-only, in-vocabulary corruption plant as the new control.

**Gates.**
- The plant reds on the last pop of a round, under a stress test, without a hang.
- The protected tests unchanged.
- A py-spy `--gil` sample showing no GIL held during the checks, at concurrency 1 and 8.
- A concurrency-8 cell-throughput read with no regression; the CPU path unchanged.
- LAW-18: the posture counter.
- Speed: −8 % at 256 over #1 + #2; abort under −5 %. Deploy-24 byte-equal.
- Self-play's 0 to +1.5 % is read in Leg 5's loop A/B, with the trainer-interference meter beside it.

### Leg 4, ROUND-FILL-1: #13 (regime 1, STRENGTH; a ruling first, §3.6.1)

**Contents:** the virtual-loss sign fix (a defect) with B3's out-of-order refill.

**Gates.**
- A unit test pinning the sign at both stone levels.
- Strength, either of:
  - equal-time cells on X and S, 288 pairs each, at fixed-depth budgets matched to the Leg-1 head's contested-turn wall
    (LAW-15), with LAW-19 power stated;
  - an ~800-pair equal-playout head-to-head, non-inferior at −0.15 logit.
- LAW-18: overlaps per select call; network positions per round trip.
- Speed: contested wall at 256 −15 %; abort under −6 %.

**The unit change:** the strength series and the in-run gate note it (the R378(b) precedent), and the parent cells are
re-read with the new head.

### Leg 5, TRAIN-SPEED-1 (regime 2; on a box)

**Contents.**
- #4 in the loop.
- Not #19: its spawn-per-round form measured −1 % on box B (§3.5.7). A persistent-helper form is the only one left, at
  low priority.
- #10 (D7a): its LAW-18 rows first, and the writer change for zero-visit exports.
- C8.
- The trainer's ring sample width, a cap lever.

The trainer's interference is parked by the operator. When it is taken up, its first step is a read-only GPU/GIL split
(§7.2(i)).

**Gates.**
- A closed-loop trainer: 2.4 steps per game, burst 8, the production ring and sample threads.
- Interleaved A/B pairs (CARD-PERF-LOOP-SPREAD).
- Positions/h beside games/h; the trainer-busy meter.
- The exams and the bands, as for any loop change.

**Decide first:** the sample-reuse drift and the gate cadence (§7.2(j), (k)).

### Afterwards

- **#15,** one tree per turn (§3.6.2), and **#16,** round fusion: strength tests, after their rulings.
- **The knob re-tune sweep,** as the operator anticipated ("a new sweep on optimal knobs may be needed if we get some
  results"): workers or games in flight, leaf batch, the collector's wait and threshold, solver budgets, child caps, the
  cache size with the sync cadence, and the ladder's budget. It is read once on the new structure.
- **For the next run's design:** #18, pruning empty→empty edges (84 % of edges). It is the largest single lever on
  per-leaf cost in both regimes, and the only one that also shrinks the trainer's footprint. It is an encoding and a
  net, not a speed patch.

## 6. Corrections, side findings, method and caveats

### 6.1 Corrections to the packet and to earlier records (each verified in source or in a record)

- **The packet's first "ours (wait 0)" line is biased.** That probe paired ours with Strix at 64 for every budget, and a
  weak opponent ends more turns by solver. The RUN11-GO session corrected the packet at 14:01 CEST. The clean A/B is
  1.6–1.8× with solvers off and 1.75–2.4× with solvers on (`box_b/timing_wait0c`).
- **"Six searches ~4× our nodes per second" compares unlike units.**
  - Six's `go nodes N` counts root visits, reused ones included, and tops the chosen child up to N: max(0, 0.75N − R) +
    max(0, N − V1) new visits per turn (`mcts.cpp:385-390, 659-714`).
  - Our descents include table hits and proven-terminal revisits.
  - §2.3 restates the gap in fresh evaluations.
- **The evaluate path's batch-64 plateau is not "max_in_flight caps each forward".** A single submitter's 64 leaves go
  in one pop (`queues/graph.rs:105-115, 256-263`); the plateau is the serial leaf build.
- **The packet's evaluate-path numbers (`versus/net_bench.py`) ran the semantic layer 1-in-64,** with no
  `collate_check_period`. Eval cells run it 1-in-1 with check 14 inline, about 0.2 ms more per graph (expert C).
- **The solver premise.**
  - "≈185 000 solver nodes per turn at 128/stone" is a cap, and an overcounted one: the root solve runs once per turn,
    and the stone-2 audit at most ~12 000 nodes.
  - The true cap is ~137 k at 128/stone, and the actual spend is ~1–3.4 % of it.
- **Self-play runs the semantic collate layer on every pop** (`selfplay/pool.py:145-152`, pinned by
  `tests/eval/test_f816_37_instrument.py`), not 1-in-64. Only the ladder runs the 1-in-64 canary.
- **R363's CPU profile and its "flat per-leaf cost".**
  - "Flat in batch and in threads" is a fact of the desktop's 3700X. On box B the per-leaf cost rises ~3× from B1 to B8
    (7.6 against 22.9 ms), and box A's 9950X scales 1.35× from 4 to 8 threads (`PERF_ADA_PROFILE_2026-09-24.md`).
  - The mechanism is memory traffic over the per-layer [E, 128] temporaries (85 % of the forward). Page faults explain
    part of the batch-1 cost only.
  - R363's 8-leaf calls never waited out the deadline, because the threshold was already 8. Its "32-leaf wake" is stale.
- **Graph sizes on the 512 recorded positions:** median 521 nodes / 13 372 edges (mean 555 / 14 122, max 1 169 / 28
  766), about 509 KB of wire per graph. R363's ~470 / 12 k was run8's sample.

### 6.2 Defects and loose ends found on the way (none fixed here)

1. **The virtual-loss frame at second-stone nodes** (§3.3 #13): `node.rs:78-86` with `selection.rs:155-156`, against the
   module doc `mcts/mod.rs:7-9`. It affects every PUCT deploy search: cells, the in-run gate, the ladder and the probes.
   Self-play's Gumbel interior selector ignores virtual loss.
2. **A select call ends at its first collision and then replays the same path until 4n attempts are spent**
   (`selection.rs:271-292, 336-412`).
3. **`leaf_build_threads` resolves to 1 on every standalone single-game path**, because it subtracts
   `selfplay.n_workers` (`config/resolve/leaf_build_threads.py`, via `sample_threads.py:69`).
4. **The production trainer's ring rebuild runs on 1 sample thread on box A** (`sample_threads.py:69`). PERF-3's drivers
   hard-coded 10 (`mantis-records/perf-3/drivers/p3_loop.py:172`, `p3_trainer.py:217`), so PERF-3's trainer readings ran
   a ~10× faster rebuild than production does.
5. **The main checkout's compiled extension (`_engine.abi3.so`, 2026-10-04 00:23 CEST) predates five commits:** 54528021
   and 9bd322d4 (`gumbel_m_quick`), 6ed55ef3 (PERF-3 L2), and fe9224fe and 7523eaef (the ring's root value).
   - The deploy path is functionally unchanged since, so this leg's numbers stand.
   - A self-play run through it would refuse `gumbel_m_quick`, and its eval-cache keys are pre-L2.
   - Rebuild only in a separate tree while cells import the checkout.
6. **`par.rs:34`'s `map_in_order` replaces a worker's panic payload with a generic string,** so a named
   `verify_contract` panic loses its name on the parallel build path (#2's path and the trainer's rebuild).
7. **The eval path has no reader for a deferred check-14 failure.** `LocalInferenceEngine.close()` drains and latches
   without raising; only `pool.py:341-349` reads one.
8. **The protected F-816-37 plant is refused by the coded pack before any check runs, on every serving path,** so it
   cannot vouch for a change to the semantic layer's posture.
9. **`gpu_wait` brackets `event.synchronize()`, but its closing timer runs after the retire thread reacquires the GIL.**
   In a loop with other Python threads it carries GIL time.
10. **A LAW-18 gap in the audit:** `audit_exhausted` counts only total-budget exhaustion (`tactics_root.rs:98`). An
    audit call that exhausts its own 2 000 nodes reads as Holds and is not counted.
11. **The producer-side `verify_contract` is 13.2 % of a graph build** (C-M1), against its comment's "under 3 %"
    (`mantis-graph` `lib.rs:903`). Where check 14 runs 1-in-1 it duplicates that check, and retiring it would save ~16 %
    of the build. That needs a ruling.
12. **Served outputs are pop-composition invariant on the CUDA-graph path:** |Δvalue| 0.0, |Δp| ≤ 1e-17 on the 3070 (A's
    M5). PERF-ADA's 0.0076 batch variance predates the CUDA-graph buckets. Verify per card.
13. **Stale text:**
    - `src/mantis/selfplay/inference_local.py:24-25` still says the deploy head's collector threshold is 32 (it is 8).
    - `mantis-records/run11/EXIT.md:346-347` says "fixed seeds and book_v1 make re-reads bit-identical". The parent X
      re-read is outcome-identical only: 2 of 580 games differ internally (red team C). RUN11-GO's session repaired it
      in place on 2026-10-06.

### 6.3 Method, agents and what failed

**Phase 0.** The lead read our search, the bridge, the server, the deploy head and the ladder backend from source. Six's
`mcts.cpp` / `planes.cpp` / `evaluator.cpp` / `main.cpp` and Strix's `hexo-mcts` were read statically. The anatomy went
to every agent and was repaired in place twice: the wait-0 correction, then the clean box-B probe.

**Phase 1.** Five experts: A GPU latency, B MCTS algorithms, C the systems / Rust–Python boundary, D the tactics solver,
E the CPU deploy.
- Each researched online with citations and wrote measurement scripts; none of them timed anything.
- A, B and E were killed mid-work by a false-positive API safeguard (`[reasoning_extraction]`). B was re-run from
  scratch in a workflow; A and E resumed as subagents from the files their first runs had saved.

**Phase 2.** Five fresh red teams, one per batch, attacked every option on five points:
- regime transfer;
- history and the falsified register;
- determinism and batch invariance;
- a SPEED claim that hides a STRENGTH change;
- an estimate that misses a mechanism.

All five updated their verdicts after session 2. Final tally: 7 CONFIRMED, 22 PLAUSIBLE, 25 KILLED, before merging
duplicates.

**Phase 3.** The lead merged and ranked the options. A fresh reviewer then checked the draft against every record and
found 16 must-fix and 35 should-fix items, all applied.

**The training question** (the operator, mid-leg).
- The lead read run11's own event stream from the mirror.
- A sixth fresh red team (batch "training") re-derived every number from the events and simulated the trainer's step
  rule. It returned 16 verdicts, 15 corrections and 11 new findings, all applied.
- The operator then cleared box B for a bench, once the RUN11-GO session (hexo-mantis-66) had copied off its records and
  confirmed nothing of its own ran there.
- The bench (`/tmp/spf1` on box B, 17:49–19:15 CEST) built its own CUDA venv from a `git archive` of 3ee61745 plus the
  env-gated #19 patch. It read single-game turns, then run11's loop with the trainer tied to games, in interleaved A/B
  pairs.

**Measurements.** The desktop measurements ran under `flock mantis-mirror/.cell.lock`, short, between run11's cells. The
box-B bench is described above.
- Session 1 (14:37–14:40 CEST): the lead's `eval_split` and `turn_profile`.
- Session 2a (15:47–16:01 CEST): the experts' scripts.
- The lead's functional virtual-loss check: counts only, on CPU.
- The patched engine ran from a separate CUDA venv on a `git archive` of 3ee61745 in `mantis-mirror/scratch/` (the
  operator's leave). The export has since been deleted.
- No worker sweep, per the operator's direction.

**Never touched or run:** box A and the repo were not touched. Box B ran only this leg's bench (above), under the
operator's leave. Rival code and weights were not run. The `phase4.log` that appeared untracked in the repo root at
15:09 CEST is not from this leg.

### 6.4 Biases and caveats

- **The host.**
  - The desktop is not box A: a Zen 2 8-core with an RTX 3070, against a 9950X with a 4080 SUPER. Ratios transfer better
    than milliseconds, and box B's probes give the second host.
  - Every desktop job is tagged CONTENDED: a foreign, mostly idle Six process held 432 MiB of the card (0–21 %
    utilisation), and the CPU load was 1.3–4.4.
  - One of expert B's CPU smokes overlapped two session-1 jobs; they were re-run and read within noise.
- **The positions.**
  - Desktop turn profiles use recorded turn starts from box-B cell games: terminal-rich, light in solver work, median 24
    stones. The sample sizes are in §2's header.
  - Turn times are bimodal, so means are compared.
  - Contested-turn numbers split on our own root value.
- **Concurrency.** The regime-1 GPU numbers are for one game per engine. Rung cells run 8 game threads per engine, and
  there #1 barely applies and #3 rarely applies.
- **Six's side is ESTIMATED from its code.** Its internals cannot be counted without running it beyond the vendored
  adapter.
- **Derived stacks multiply measured steps.** Interactions beyond the one measured (#1 + #2) are assumed independent,
  and the legs' own A/Bs are the record.
- **The STRENGTH options are unmeasured for strength.** Each needs its own cells at equal time (§3.3).
- **The box-B loop is not box A's.**
  - The card is Blackwell, not Ada; the CPU is a 5900XT, not a 9950X.
  - It ran run11@162k and its ring, with the trainer closed loop as in production.
  - Ratios transfer better than rates, and box A's own A/B is still owed.
- **The trainer-interference split is observational:** 39 919 intervals, by trainer activity. Its busy share follows
  game completions; it was not randomised.
- **Cells reproduce outcomes, not every internal.** The run11 parent X-cell re-read matched all 580 games' moves, but 2
  differ in a root value or visit split. With the served path composition-invariant, those are rare ties, not value
  noise.

### 6.5 Records

The leg's local records sit outside the tree, as for every leg, in `mantis-records/search-perf-1/`:
- `ANATOMY.md`;
- `experts/<key>/{RESULT.json, NOTES.md, scripts}`;
- `redteam/<key>_REDTEAM.{json, md}`;
- `lead/measure/`: `S1_RESULTS.md`, `S2_RESULTS.md`, the session JSONs and job logs, `eval_split.py`, `turn_profile.py`,
  `run_batch.sh`;
- `lead/vl/VL_CHECK.md`;
- `lead/train/`: `extract_run11_loop.py` with its outputs, the run11 extraction for §3.5;
- `redteam/training_REDTEAM.{json, md}`: the training red team, with its per-burst method in an appendix;
- `boxb/`: the box-B bench's scripts (`run_bench.sh`, `spf_loop.py`, `turn_profile.py`), the #19 scratch patch
  (`round_build.patch`), and every output (`r1_*.json`, `loop.json` per run, the logs);
- `report/`: the section drafts and `REVIEW.md`.

The box-B probes are in `mantis-mirror/run11/versus/box_b/{timing, timing_wait0c, speed}/`.

## 7. Open items for the overarching agent

This section collects everything still open, as the operator asked: "anything other that is still open I would want you
to document extensively in the report so the overarching agent could also decide / think that through". Each item gives
the question, the evidence, the options and this leg's recommendation, or says that it has none.

Rulings are the operator's. A session records the operator's word, never a ruling.

### 7.1 The operator's word in this session (2026-10-06), to be recorded formally

Quoted as written, in the order given.

1. **No knob sweep in this leg:** "I would not say we go diff worker counts this would need to be benchmarked etc we
   rather look for other obvious stuff we could do since a new sweep on optimal knobs may be needed if we get some
   results".
2. **Single-game speed has its own value:** "even if its only marginally in training benefits in single games played
   e.g. being faster would still always help etc / my call as operator no?"
3. **The bit-identical play fixes go ahead:** "but the other 'free' ones I would definetly do". In the list the operator
   was answering, these were #1 wake on submission, #2 parallel leaf build, #3 small buckets, #4 the faster builder and
   #9 the CPU message sum.
   - #3 is not strictly bit-identical: it moves values by ~1e-17, inside the served path's own spread. Its gate is "no
     argmax or sign flip at the ≥ 1 024-node floor", not "byte-equal".
   - The exact first-collision stop was not in that list (§5 Leg 1).
4. **#6, the per-game exact eval cache: "I would allow #6".** The formal ruling still has to be written: R370(c)
   extended to eval paths, with the scope and gates in 7.2(a).
5. **#14, the early stop:** "I also think that #14 could be interesting, but could that potentially affect if we want to
   use this for analysis where we also want to see different stones the engine would look at?" It does affect analysis.
   This leg recommends a per-use switch (7.2(b)).
6. **#5, the checks during the forward:** "#5 also sound good to me but we'd need to make sure it wont make deadlock or
   python lock problems etc e.g. see that it is save and does not cost us". The design and its proofs are in 7.3.
7. **The trainer's interference is parked:** "I want you to not go into the 'largest measured training term yet' but do
   note that in detail in the final report". It is set out in §3.5.3.
8. **The two search-changing options** (#13, #15): "I want you to note the 2 changes that would change extensively in
   the final report". They are in §3.6.
9. **Box B for the bench:** "you may use box b now for benching ig / talk to the other session when you can use it". The
   RUN11-GO session (hexo-mantis-66) confirmed box B was free after copying off its records. Destroying the box is the
   operator's act; this leg tells the operator when the bench is done (§3.5.7).

### 7.2 Decisions owed, with options and a recommendation

**(a) #6, the per-game exact eval cache (allowed; the ruling text is owed).**
- **Scope:** every deploy player on the GPU path: the cells, the in-run gate and the equal-time probes. The CPU ladder
  too, if accepted.
- **The cache:**
  - per player and per game, never shared across engines or games;
  - keyed exactly, the way self-play's cache keys;
  - cleared by the player's per-game reset (`DeployHeadPlayer.new_game`), never by `MCTSTree.new_game`, which runs at
    every stone;
  - it must dedup keys already pending in a batch (5–8 % of a stone's evaluations repeat inside one batch).
- **Identity:**
  - on the GPU's served path, a position's outputs do not depend on its batch (A's M5 on sm_86; verify per card, 7.4),
    so moves should be identical;
  - on the CPU ladder, the eager path's value moves by ≤ 7.2e-7 with batch makeup. There the cache stays inside
    R370(c)'s spread but is not identical.
- **Options for the CPU ladder:** accept its tiny non-identity, or keep the cache off on CPU.
- **Recommendation:** accept it, with a same-move-rate gate.
- **Gates:**
  - same moves on 200 positions (expected 100 % on the GPU);
  - a CPU same-move rate;
  - one X-cell re-read with ≥ 99 % of games move-identical;
  - LAW-18 rows: hits and GPU evaluations;
  - the all-hit round-trip share counted, which sizes the GPU gain (−5 to −20 %).

**(b) #14, the budget-aware early stop: where it is on, and what analysis loses.**
- **What it changes.**
  - The visit leader before the audit never changes, by construction.
  - The played move can. The audit takes its alternative first stones, and the chosen stone's second stones, in visit
    order (`tactics_root.rs:48-62, 125-134, 386-398`). An early stop also leaves a different solver table for the audit.
  - Measured: 0 of 156 stones changed, before or after the audit (95 % upper bound ≈ 1.9 %).
  - The runner-ups stop gaining visits, so their counts and values are less settled and their order can differ. That is
    both what analysis would lose and the channel through which the audit could pick differently.
- **Who reads the visits today:**
  - the dash's Analyzer view (`make dash`; `tools/dash/engine/engines.py:199-217`, `analysis.py:17-34`), which shows
    children by visits and `root_visits`;
  - the per-ply `root_value` and visited `visits` that `src/mantis/arena/match.py:170-186` writes for every deploy-head
    ply, in cells and on the promotion channel (the in-run gate). The dash's Games view
    (`tools/dash/readers/games.py:89-93`) and the re-read identity checks read them;
  - the ladder receipt's per-move `sims`, and `--replay`'s budget witness (`tools/ladder_bot.py:31-80`,
    `tools/ladder/receipt.py`). With the stop on, most stones would record below their configured sims, and
    `below_budget` would fire on most turns;
  - the protected served-sims witness (`served_sims_exact`).
- **Recommendation: a switch set per consumer.**
  - The ladder: ON. Its receipt stamps the switch, and `below_budget` must tell an early stop from a decided position.
  - The dash's Analyzer: OFF.
  - Rung cells and the in-run gate: OFF, because their records carry visits and the series' unit.
  - Equal-time probes: OFF, unless their time-matching is redone with the stop on (then stamped).
- **The ruling it needs:** R378(c) made "the head spends exactly its budget" the unit (`last_sims == n`). A deliberate
  stop needs a ruling that separates it from the early end R378(c) called a defect.
- **Gates:**
  - a 512-position witness at 256/stone (bounds flips at ≈ 0.6 %);
  - LAW-18 rows: stops fired and descents saved;
  - an IQR-gated turn bench;
  - a strength screen only if flips appear.
- **A separate lever it opens:** on a real clock (the ladder), the time saved on easy moves could go to hard ones. That
  is time management, unexamined here.

**(c) #5, the protected checks during the forward (wanted, on 7.3's conditions).** The ruling concerns the protected
1-in-1 collate checks and F-816-37's dump-on-fire.
- **What a ruling would grant:** on single-game engines, check 14 runs after the replay is launched and before the
  results are released, still 1-in-1 with dump-on-fire.
- **What it needs:** a new check-14-only, in-vocabulary corruption plant. The existing protected plant is refused by the
  coded pack before any check runs, so it cannot vouch for the moved check.
- **Recommendation:** land step (b) first. It needs no posture ruling and helps both regimes. Rule on step (a) once
  7.3's proofs read.

**(d) #13, the virtual-loss sign fix plus refill: a deploy-instrument change (§3.6.1).**
- **The ruling:** the series break, and the re-baselining of the parent cells.
- **The strength design to choose:** equal-time cells on X and S at 288 pairs (a null is void at the expected effect),
  or an ~800-pair equal-playout non-inferiority test.
- **Recommendation:** fix the sign in the same leg as a test that pins it, since it is a defect, but land it only
  through the strength test.

**(e) #15, one tree per turn (§3.6.2).**
- **The ruling:** the per-turn unit, and whether inherited visits count (Six's top-up) or not (extra visits).
- **Recommendation:** after #6, as a strength test.

**(f) #16, Sequential Halving round fusion in self-play.**
- **What it breaks:** the protected `a_gumbel_round_is_exactly_the_halving_phase_wide` and R346(c)'s round definition.
  It also moves policy targets.
- **The gain:** 0 to +14 % games/h in a real run (open loop 0 to +20 %).
- **What it needs:** a twin with the starvation exams, then panel cells.
- **Recommendation:** rank it against leaves in flight (a knob) at the re-tune; do not land it before.

**(g) #10 (D7a), no search at quick-arm decided roots.**
- **Status:** SPEED for quick-arm roots, +1 to +3 % in a run.
- **What it needs:**
  - a writer change, since `refuse_zero_visit_export` (`mantis-selfplay/src/records.rs:182`) refuses a zero-visit export
    today;
  - a ruling, because that change relaxes R275(b)'s exporter conjunct ("a zero-visit search cannot produce a target").
- **The extension:** the decided_lost roots are STRENGTH-class, because they play a different losing stone. Full-arm
  decided roots are ruled out by R239, R377(f) and R378(e).
- **Recommendation:** the quick-arm form only.

**(h) #3 on box A's card (sm_89).** The measurement is sm_86 only, and the read needs an sm_89 (Ada) card, since box A
is live.
- **Options:** a short rental (about 1 box-hour); box A between runs; or landing at the 1 024-node floor on sm_86
  evidence with the sm_89 read owed.
- **Recommendation:** the third, since the gate then reads it on whichever card serves first.

**(i) The trainer's interference (parked; §3.5.3) and A7.**
- **The first step:** a read-only GPU/GIL split of the trainer's cost in run11's regime (py-spy `--gil` or schedstat).
  Box A is live and off-limits, so this needs the operator's leave for a read-only sample there, or a box-B replica of
  the loop.
- **The same split decides A7,** a GIL-free serving dispatch. It was killed on PERF-2's split; its own revival line (≥ 1
  ms per pop of non-device wait) now looks met (§4).
- **Then the remedy path:**
  - a separate trainer process under MPS, or a second GPU;
  - a GIL-free trainer collate;
  - moving the retirer's result submit out of the GIL;
  - pacing the trainer against serving;
  - the trainer's kernel sizes;
  - an SM partition (#12, for the device share only);
  - #18, which shrinks both sides.
- **The ceiling:** ~+33 % games/h, ~+26 % steps/h.
- **Recommendation:** the read-only split first, when the operator takes it up.

**(j) The sample-reuse drift.**
- **The mechanism:** faster self-play lowers the realised steps per game under the burst cap: 2.14 at 1.5× (§3.5.4).
- **The envelope:** STATE's [1.2, 2.4] bounds the configured `training_steps_per_game`, not the realised rate. Raising
  `max_train_burst` is a re-mint with a ruling.
- **Options:** accept the drift; widen the trainer's ring sample to raise the cap; or re-mint the burst cap.
- **Recommendation:** widen the ring sample first; decide the rest before any regime-2 speed leg lands.

**(k) Gate cadence.**
- **The mechanism:** `eval_interval` counts steps, so a faster loop makes gate rounds more frequent per hour (4.5 % →
  ~5.8 % of wall at +30 % steps/h).
- **Options:** restate the cadence in hours, or accept it.
- **Recommendation:** none; this is the operator's call. The cost is small, and the single-game fixes shorten each
  round.

**(l) The deploy unit itself.**
- **The comparison:** Six searches per turn (one tree, a turn budget, reuse across turns). We search per stone with
  fresh trees.
- **The question:** whether the deploy head should move to a per-turn unit. This is bigger than #15 and touches every
  strength series.
- **Recommendation:** take it up in the next design pass, not in a speed leg.

**(m) The strength ruler's unit.**
- **The mechanism:** the speed work makes equal playouts mean less wall time.
- **The evidence:** at equal time we gain a lot against Strix (0.391 → 0.637 at 1 s per turn, wait removed) and nothing
  measurable against Six (0.047 → 0.045 at 0.9 s).
- **The question:** whether the strength rulers stay at fixed playouts or move to fixed time. The answer decides how
  much of this work reads as strength.
- **Recommendation:** none; this is the operator's call. Keep both readings until it is decided.

### 7.3 #5 made safe, and shown to cost nothing (the operator's condition)

**One design, the same in §3.1, §5 and here.**
- **(b) first, both regimes:** checks 15–16 move into the Rust pack's gather pass. They run with the GIL released and
  still before the launch (CARD-PERF-COLLATE-2; expert A's A2(b)).
- **(a) then, single-game engines only:**
  - check 14, already GIL-free under `py.detach` (`bridge/src/graph_contract.rs:229`), is captured at collate by the
    existing `deferred_edge_geometry` sink;
  - it runs on the retire thread between the launch and the release, before the results are submitted (expert A's
    design);
  - the server stays free to pop the next batch, which matters at concurrency 8, and the retirer would otherwise sit in
    `event.synchronize()`.
- **Self-play keeps check 14 on its checker thread.** Only (b) applies there: moving check 14 onto the serving path
  would add ~7 ms per B-36 pop.

**Device safety.** The structural checks 4–13 and the vocabulary lookup stay in the pack before the launch
(`selfplay/graph_collate.py:398-424`, `_PACK_ERRORS`). An unchecked batch therefore cannot fault the device: a failing
batch is computed but never served.

**No deadlock and no hang.**
- The design adds no thread, lock, timer or wait edge.
- A failed check raises inside the existing failure path:
  - a raise before `retirer.submit` releases the pipeline slot and fails this pop's waiters
    (`inference_server.py:742-744`);
  - a raise in `_retire` does the same through `_retire_or_fail` (`:905-910`) and the retirer's `finally` (`:277-278`).
- So every waiter of the pop is woken with the error, the slot is released, and the run halts with its dump, as today.
- Graph waiters have no timeout of their own (`queues/graph.rs:295-308`). This failure path is what rules out a hang,
  and the stress test proves it under a test timeout.

**GIL.**
- Check 14 is GIL-free, and checks 15–16 become GIL-free by (b). There is no GIL-held fallback.
- If (b)'s port cannot land, (a) does not land either: F-46 showed that a check holding the GIL against the server costs
  self-play 12.6 %, and the same would hold for cells at concurrency 8.

**Proof of safety.**
- A stress test: concurrent submitters at concurrency 1 and 8, with the new check-14-only plant on the last pop of a
  round. That pop's waiters must fail cleanly, the run must halt, and nothing may hang within the test timeout.
- The protected set green.
- py-spy `--gil` samples at concurrency 1 and 8, showing no GIL held during the checks.

**Proof of no cost.**
- The LAW-09 bench at 256/stone, concurrency 1: −8 % over #1 + #2, aborting under −5 %. Deploy-24 byte-equal.
- A concurrency-8 cell-throughput read with no regression.
- The CPU ladder path unchanged: its forward is synchronous, and its 1-in-64 canary is untouched.
- For (b) in self-play: a closed-loop loop A/B with no regression, with the trainer-interference meter (§3.5.3) read
  beside it.

### 7.4 Measurements owed (not made in this leg)

- **#3 on sm_89** (7.2(h)), and the 1 024-node floor's own read.
- **sm_89 composition invariance,** which #6's GPU identity rests on.
- **#6's all-hit round-trip share,** which sizes the GPU gain.
- **#5:** the GIL samples and the concurrency-8 read (7.3).
- **#14's 512-position witness.**
- **Strength cells** for #13 and #15 (and for #16 in self-play).
- **The round-fill leg's GPU wall read** (red team B: −17 to −20 % serial, −25 to −30 % after #2, unmeasured).
- **#4 on box A,** in a closed-loop A/B. #19 only in a persistent-helper form, at low priority: its spawn-per-round form
  read −1 % on box B (§3.5.7), a different card and CPU.
- **The trainer interference's GPU/GIL split** (7.2(i)).
- **GPU utilisation in run11:** unread. PERF-3's comparable loop read 74–78 %; box B's loop ~60 % with the trainer and
  48 % without.
- **The CPU ladder on the real VPS,** where both ladder bots share 4 vCPUs. The taskset stand-in has no neighbour.
- **The ladder as an instrument** (red team E):
  - a CPU fp32 against GPU bf16 agreement probe;
  - the ladder's 1-in-64 canary against the protected 1-in-1 pin.
- **E2's hoisted tables,** computed at the weight-swap point (red team E).
- **The `select_leaves` profile** that the collision stop's LAW-09 line waits on.
- **Unowned smaller options:**
  - #7's same-session A/B;
  - #8's leaner wire (a wire-contract and protected-plant decision), C5, the bf16 weight copy, and D3–D5.
- **Six's internals:** estimated from its code, never counted.
- **The tactics share on X-cell positions.** This leg's turn starts are terminal-rich. Expect up to ~2× the ≤ 7 %
  measured here, and the TACTICS-DEPLOY L5 ratio of 0.864× plain.

### 7.5 Defects and register repairs found (none fixed here)

Most are detailed in §6.2. This table is the consolidated list, with an owner for each. Rows are D1–D20 so that no "#"
collides with an option number.

| D | Item | Where | Suggested owner and action |
|---|---|---|---|
| D1 | Virtual loss becomes a bonus at second-stone parents | `mcts/node.rs:78-86` with `selection.rs:155-156` | Leg ROUND-FILL-1 (§3.6.1); a test pinning the sign |
| D2 | A select call burns its remaining 4n attempts after its first collision | `selection.rs:271-292, 336-412` | Leg 1 (exact), on the overarching agent's call (§5) |
| D3 | `leaf_build_threads` resolves to 1 on standalone single-game hosts | `config/resolve/leaf_build_threads.py` | Leg 1 (#2) |
| D4 | The production trainer's ring sample runs on 1 thread; PERF-3's drivers used 10 | `sample_threads.py:69`; `perf-3/drivers/p3_loop.py:172` | A cap lever (§3.5.4). PERF-3's trainer readings do not represent production |
| D5 | The main checkout's `_engine.abi3.so` predates five commits (54528021, 9bd322d4, 6ed55ef3, fe9224fe, 7523eaef) | desktop main checkout | Rebuild in a separate tree while cells import the checkout |
| D6 | `par.rs` drops a worker's panic payload | `par.rs:34` | Fix before #2 or #19 |
| D7 | The eval path has no reader for a deferred check-14 failure | `pool.py:341-349` is the only reader | Before any serve-first posture on eval engines (7.3's check-before-release design does not need it) |
| D8 | The protected F-816-37 plant cannot vouch for a semantic-layer posture change | the coded pack refuses it first | A check-14-only plant (#5) |
| D9 | `gpu_wait` includes a GIL re-entry; `queue_wait` includes the pop's Rust fuse | `inference_server.py:707-711, 857-860` | Read them as stage spans, not as device time or slack (F-47) |
| D10 | LAW-18 gap: `audit_exhausted` counts only total-budget exhaustion | `tactics_root.rs:98` | Count a per-call exhaustion |
| D11 | The producer-side `verify_contract` is 13.2 % of a build, against its comment's "under 3 %" | `mantis-graph/src/lib.rs:903` | Repair the comment. Retiring it where check 14 runs 1-in-1 (−16 % of the build) needs a ruling |
| D12 | The retirer's result submit holds the GIL | `bridge/src/inference.rs:412-487` | A trainer-interference channel (7.2(i)) |
| D13 | The eval cache is cleared at every actor sync (every 50 steps) | `queues/eval_cache.rs:174-178` | Cache size and sync cadence are linked knobs. §1.2 of this report is corrected |
| D14 | Stale text: "a collector threshold of 32" for the deploy head (it is 8) | `src/mantis/selfplay/inference_local.py:24-25` | A one-line repair |
| D15 | Stale text: the "32-leaf wake" in the CPU head profile | `docs/design/measurements/CPU_HEAD_PROFILE_2026-09-20.md` §2 | A one-line annotation |
| D16 | run11's EXIT said "fixed seeds and book_v1 make re-reads bit-identical"; the parent X re-read is outcome-identical (2 of 580 games differ internally) | `mantis-records/run11/EXIT.md` §10 | **Repaired** in place by RUN11-GO's session on 2026-10-06: EXIT §10 and its report page; its own-errors list §8 item 11; an annotation in `run11-fresh/PRESTATED.md` |
| D17 | Self-play's `queue_wait` read as slack in this leg's own first draft | §3.5 | Corrected; the same misreading F-47 records |
| D18 | F-44's R365(e) note says this regime's trainer duty is "UNREAD" | `docs/governance/falsified.md:138-147` | Annotate: run11 reads 0.69 (§3.5.3) |
| D19 | CARD-PERF-TRAINER-CONTENTION carries PERF-2's isolated numbers only | `docs/governance/CARDS.md:140-146` | Annotate: run11's −25 % time-averaged, −37 % while stepping, +33 % ceiling (§3.5.3) |
| D20 | A7's kill rests on PERF-2's split | §4 | Re-opened in §4; decided by 7.2(i)'s split |

### 7.6 Interactions and risks to think through

- **Speed versus strength against each opponent.** At equal time, speed buys a lot against Strix and nothing measurable
  against Six at these budgets. Closing the gap to Six needs strength (net, solver, search shape), not only speed.
- **Instrument continuity.**
  - Every deploy change moves the cells, the in-run gate and the ladder for future runs.
  - run11's box-A tree must not move, and the desktop's run11 cell unit imports the main checkout.
  - Bit-identical changes keep the series; #13 and #15 break it.
- **Regime-2 speedups change training dynamics** through the burst cap and the gate cadence (7.2(j), (k)), even when
  every change is bit-identical at the search level.
- **Hosts.** Box B (Blackwell + 5900XT) and the desktop (Ampere + 3700X) bracket box A (Ada + 9950X), but neither is box
  A. PERF-ADA found precision trade-offs differ between Ada and Blackwell.
- **Concurrency.** The single-game GPU numbers are for one game per engine. Rung cells run 8 game threads per engine,
  where #1 and #3 do little; the equal-time probe and the ladder get the most.
- **Selection in this report.** Options this leg proposed (#19, the first reading of the training section) were cut by
  its own red teams and its bench: #19 from a central +6 % to +2.5 % by red team, then to −1 % measured on box B; the
  training gains by ×0.6–0.7. Every estimate here should be read as a prior for a measurement, not as a result.
