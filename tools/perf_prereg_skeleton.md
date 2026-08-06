# Perf prereg skeleton (LAW-09) — hotspot rows, blank brackets

**Phase P2. PREP ONLY. Every gain-bracket and abort-threshold cell below is deliberately
BLANK and is the operator's to fill — BEFORE the profile runs, not after.**

LAW-09: *"One optimization = pre-registered hotspot list + expected gain bracket + abort
threshold. One change = one commit = one IQR-gated bench; parity oracles re-run after every
hot-path change; measure the end-to-end metric, not only the microbench."*

The ordering matters and is the whole point of a skeleton committed ahead of the numbers: a
bracket chosen after seeing the profile is a post-hoc threshold, and it can only ever be met.
The rows below name WHERE to look and WHAT would count as a finding; they assert no
magnitudes, because none have been measured by this dispatcher.

Harnesses: `tools/profile_selfplay.sh` (P1a), `tools/profile_eval.sh` (P1b). Neither has been
executed — the box run is operator-only (R170), and no host specifics appear in either script
(rule 7).

---

## 1. Falsified adjacency — READ BEFORE PROPOSING ANY OF THESE (LAW-05, and LAW-02 to cite one)

These rows are stated up front because three of the six hotspots below sit directly on top of
already-falsified work, and the register's rule is cite → state the context it was falsified
in → test whether that context transfers → only then keep or drop.

- **F-17** — *a sorted-Vec legal-move set beats the hash-set rebuild.* Falsified by bench:
  −32.5% sims/s. Mechanism: the ring loop pushes ~7× duplicate cells, so sort+dedup on the
  bloated array costs more than hash-with-inline-dedup insert.
- **F-18** — *the residual legal-move-set self-time is lookup-dominated.* Falsified by
  flamegraph: insert dominates (56.8% vs 27.7%). The earlier fix failed by **fix-design
  error, not by the assumed mechanism** — which is the trap to avoid repeating.
- **F-19** — *incremental legal-coverage delta maintenance amortizes below the once-per-leaf
  rebuild.* Falsified by bench: −49.5% sims/s. The delta runs per descent STEP
  (apply × depth + undo × 2·depth), not per leaf. **Corollary now standing as perf doctrine:
  build-once-per-leaf beats incremental deltas on descent paths.** Row 1 below is bound by it.
- **F-21** — *a custom CUDA kernel speeds the dense forward path.* Falsified + red-teamed.
  **SCOPE, and it cuts both ways:** the row is kernel/ragged-batch-perf **scoped to DENSE**,
  and says explicitly it *"does NOT adjudicate the axis-graph representation"*. So F-21 does
  NOT settle graph batching (row 3 is open); equally it is not a licence to reach for a
  custom kernel — its stated fallback order, if forward throughput binds, is
  **torch.compile → smaller net → quantized eval, in that order.**
- **R191 / R181** — oracle statistics: deterministic-mode exact assertions are preferred over
  calibrated bounds. Any bench assertion here inherits that.

**A change that contradicts F-17/F-18/F-19's build-once-per-leaf doctrine is REJECTED at
design stage unless the proposer performs the LAW-02 transfer test in writing.**

---

## 2. Pre-registered hotspot rows

Fill `expected gain bracket` and `abort threshold` BEFORE running P1. "Abort threshold" =
the measured result at or below which the line of work STOPS, no retry, no re-scope.

| # | hotspot | where it lives | why it is suspected | end-to-end metric (LAW-09: not only the microbench) | expected gain bracket | abort threshold |
|---|---|---|---|---|---|---|
| 1 | **per-leaf graph rebuild** | `mantis-graph` builder, called per MCTS leaf via `queues::build_leaf_graph` | The graph is rebuilt for every leaf evaluation; at 150 mean sims/move this is the highest-multiplicity Rust work on the self-play path. **BOUND BY F-19's corollary** — the fix is NOT an incremental delta; that exact class was measured at −49.5% sims/s. Any proposal here must be a cheaper REBUILD, not a delta. | self-play `games_per_hour` (on-record baseline ≈19, box tier 1) | ______ | ______ |
| 2 | **marshal + queue wait** | `mantis-selfplay::queues` (Dense/GraphQueue) ↔ `mantis-bridge::inference` ↔ Python `InferenceServer` | Every leaf crosses the FFI and a queue. Wait time is invisible without `--idle` sampling, which is why both P1 scripts pass it. Suspicion is that a large share of wall-clock is BLOCKED, not computing — which would redirect the whole investigation away from kernel-level work. | self-play `games_per_hour`; queue depth / dispatch age from the event stream | ______ | ______ |
| 3 | **GPU forward under F2 edge-cap / micro-batching** | `train/trainer/core.py::train_step_from_graph_batch`; `config/resolve/microbatch.py` caps | The F2 edge cap + micro-batching landed for MEMORY (CARD-RUN5-GPU-OOM, a 16 GiB wall), and a memory fix routinely costs throughput. Whether it does here is unmeasured. **F-21 is dense-scoped and does not settle this**; its fallback order (torch.compile → smaller net → quantized eval) is the sanctioned ladder if forward binds. | training steps/sec; the 41.66 ms/step new-side floor (WP10 bench pivot) | ______ | ______ |
| 4 | **V-1 cache-release cost** | `arena/deploy_head.py` — CUDA cache release at move boundaries (landed `3be49d4`) | V-1 fixed a real VRAM accumulation defect (VERDICT-A) by releasing the allocator cache at every move boundary. `torch.cuda.empty_cache()` is not free and this runs PER MOVE on the deploy path. STATE §2.5 already pairs it with the V-0 observation (~1.6 s/move at deploy_sims=150). **9C's V-1 LAW-09 bench is ABSENT from the repo record** (Phase 0.3) — so this row has a fix with no bench, which is what LAW-09 asks for on a hot path. | eval wall per round at deploy sims; the R235 20.12 s/round figure, whose REGIME must be recorded this time | ______ | ______ |
| 5 | **GIL-held Rust calls** | `mantis-bridge::buffer.rs`, `hexg.rs`, `runner.rs` | PyO3 calls hold the GIL unless they explicitly `py.detach()`. A long GIL-held Rust call serialises every Python thread in the process — the inference server, the drain loop, the coordinator. `inference.rs` already detaches on `pop_graph_batch`; whether the buffer and hexg paths do is unaudited. Presents as "everything is slow", not as one slow thing. | wall-clock share of GIL-blocked frames in the P1a speedscope | ______ | ______ |
| 6 | **segment ops at ~7k candidates** | ragged policy/value reductions — `ragged_policy_ce`, `_binned_value_loss`, `legal_offsets` segment scatter | Segment reductions over ragged offsets scale with the candidate count; ~7k candidates is the stated operating point. Cheap to measure, and if it is not hot the row closes and stops absorbing attention. | training steps/sec | ______ | ______ |

---

## 3. Protocol for each row, once brackets are filled

1. **Profile first** (P1a / P1b). Localise before proposing. A row whose profile shows it is
   not hot **closes** — that is a result, and a measured structural floor is a finding, not a
   failure (LAW-09's own words).
2. **One change = one commit = one IQR-gated bench.** No bundling.
3. **Re-run parity oracles after every hot-path change** — the graph/dense parity suites and
   the Q13 oracles, not just the bench.
4. **Measure the end-to-end metric**, not only the microbench. A microbench win that does not
   move `games_per_hour` or steps/sec is not a win.
5. **A single-run regression with no code mechanism on the touched path requires fresh-bench
   triangulation before any verdict** (LAW-09) — do not accept a lone bad run as evidence,
   and do not accept a lone good one either.
6. Bench floors: `tools/bench_floors.toml` carries 28 floors attested against rustc 1.97.1
   (`rust-toolchain.toml`). **Changing the toolchain invalidates all 28** — a bump is a
   perf-host event, not a local one.

## 4. What this skeleton does NOT do

It states no magnitudes, ranks no rows, and predicts no winner. The row ORDER above is
narrative (self-play path, then eval path, then cross-cutting), not a priority claim. The
operator's brackets are what turn it into a prereg; until then it is a list of places to look
with the falsified ledger stapled to the front.
