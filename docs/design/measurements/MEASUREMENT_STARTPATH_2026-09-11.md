# MEASUREMENT — the START path: α = 1.0 reconstructed, the trainer step profiled (R349(c),(d)), 2026-09-11

Instrument: ONE burst of run6's own regime on the box (Ryzen 9 9900X / RTX 5080, `torch
2.11.0+cu128`, tree `4c4a83d4`): `configs/run6.yaml`'s 27 deltas replayed onto the template
with `max_train_steps 600`, `checkpoint_interval 200`, `terminal_eval_enabled false`,
`draw_rate_abort null` (its `min_step 25000` is unreachable in 600 steps and the schema
refuses an unfireable arm), run in-process through `launch_run` UNDER `py-spy record --idle
--native --threads` at 50 Hz (the sampler as the parent, so Yama scope 1 needs no attach),
warm-started from the LAW-12 strip. Every row of §C is then reconstructed off-line from the
run's own `alpha_full_row` events with the warm-start net through the deploy head; §D adds a
ten-step `torch.profiler` pass on the burst's step-400 ring through the production step.

## A. What the burst did

| quantity | value |
|---|---|
| wall to first trainer step (ring fill to `min_buf_size 4096`) | 6.4 min |
| stepping window | 1,613 s, steps 1 → 496, games 178 → 680 |
| **steps/h in the stepping window** | **1,105** (games/h 1,120; **0.986 steps per game**) |
| batch fill / queue wait / collate (inference server, `edge_geometry_check: inline`) | 54.5 % / 3.19 ms / 7.08 ms |
| α = 1.0 rows (`iteration_complete.gumbel_alpha_full`) | 7 of 15,576 at step 296 → 24 of 23,118 at step 496 = **1.04 per 1,000 and rising** |
| **how the burst ended** | **step ≈ 496, `EmptyTarget … n_legal=0` at the ring push, the feeder died, run-fatal** |

The death is R349(c)'s question answered by the run itself: a sparse row whose 16 sampled
candidates ALL carried exactly zero mass had no explicit cell to store, the ring refused it as
the degenerate all-zero class, and `check_producer_health` took the run down. At this rate run6
would have died inside its first hour.

## B. The rows, reconstructed (25 `alpha_full_row` events, warm-start net, 3 Gumbel seeds each)

Every one of the 25 is at **`moves_remaining == 1`** (the turn's second stone); 21 quick
(64 sims), 4 full (320). Per row the reconstruction reads: raw root value `−0.85 … −1.00`,
root W/N within 0.06 of it, every visited child's Q (root perspective) in `−0.87 … −1.00`,
the children's own net values (opponent to move) `+0.83 … +0.997`. The 16 recorded explicit
masses per row run `1e−9 … 1e−33`; the 4 full rows recorded only 4, 8, 11 and 11 cells with
masses down to `1.4e−45` (the f32 denormal floor) — the rest had underflowed to 0 and been
dropped; the fatal row dropped all 16.

**Hypothesis 1 (a perspective error at the intermediate stone): REFUTED.** Root `−0.95` against
children `+0.98` from the opponent's side is the `q_sign = −1` flip working; W/N agrees with the
raw value on every row; `mctx_completed_qvalues` and the recorded masses agree with the f32
replay of `completed_q.rs` to the ULP.

**Hypothesis 2 (the early-value pathology): CONFIRMED, with the mechanism.** These are LOST
positions. The children — the opponent to move with a full turn in hand — sit at `≈ +0.99`
(a won-next-turn position is trivially legible), so every visited Q is `≈ −0.99 ± 0.005`,
while the root, one stone earlier and less certain, reads `−0.85 … −0.95`. Mctx's
`v_mix = (raw + Σn·q̄)/(Σn+1)` is therefore lifted ABOVE the visited Q's by
`(raw − q̄)/(Σn+1) ≈ 0.001`; whenever the 16 candidates cluster within that, every UNVISITED
child holds the completed maximum and the min-max rescale maps the span (`0.002 … 0.06`) onto
`(c_visit + max_n) · c_scale = 55 … 59` logits at 64 sims — the visited masses are `e^−28 …
e^−57`; at a full search's `max_n` the gap passes ~103 and the mass is exactly 0. The exclusive
`mr == 1` signature is the sign flip meeting this saturation asymmetry, not a bug in the flip.
The 75 re-searches reproduced α ≈ 0 (v_mix landed INSIDE the visited range by
`0.001 … 0.11`): the event needs the run's own Gumbel draw and its own step-N weights, and the
rate climbs as the value head saturates (7 → 24 rows over 200 steps).

**The defect (fixed, `f253e65e`, parity vector on both sides of the FFI):** `record_position_graph`
kept only `p > 0` cells — a rule that predates sparse rows — so a sampled candidate whose mass
underflowed left the explicit set (the trainer's mask then spread the tail over it, wrongly, on
every such row) and when all m underflowed the row was empty and refused. The explicit set is
now MEMBERSHIP: every support cell is stored, at zero mass too. `records::a_sparse_row_stores_
every_support_cell_even_at_zero_mass` and `test_an_all_tail_sparse_row_is_admitted_stored_at_
zero_and_trains_finite` are the vectors; the sampler already masked by membership
(`sample.rs`: "the mask is membership in the STORED map, not `prob > 0`").

**Left to the architect (D2, not a code defect):** the target these rows carry is rescaled
noise — in a decided position the improved policy puts everything on unsearched moves because
`c_scale 1.0` on top of Mctx's min-max rescale turns a `< 0.01` Q spread into a hard argmax.
The paper's σ(q̂) = (c_visit + max N)·c_scale·q̂ has NO per-node min-max (its `c_scale 1.0` is
for q̂ in its natural [−1, 1]); Mctx's default with the rescale is `value_scale 0.1`. Options,
each a regime change needing the PERF-3b re-measure: the paper's form; `c_scale 0.1`; a span
floor. At 1 per 1,000 rows the trainer's cost is nil (an all-tail row is a zero-gradient
self-target); the question is whether ~1 % of decided-position rows should train "not here".

## C. The trainer step — 92 % asleep (py-spy, 97,554 main-thread samples = 1,951 s)

| where the trainer thread's wall went | share | per step (496 steps) |
|---|---|---|
| `sleep(0.1)` at O5 "no new games since the last burst" | **74.9 %** | 2,946 ms |
| `sleep(0.5)` at O4 warm-up (ring below `min_buf_size`) | 16.9 % | 665 ms |
| **everything else** | **8.2 %** | **323 ms** |
| — `loss.item()` sync (`core.py:397`; absorbs the GPU forward+backward) | 1.8 % | 72 ms |
| — forward, CPU-side launches (`gnn_v2`/`gine`) | 1.5 % | 60 ms |
| — replay rebuild `sample_graph_batch` (Rust, GIL-free, 7 threads) | 1.5 % | 57 ms |
| — collate: the 1-in-1 semantic check 14 | 1.3 % | 53 ms |
| — backward (CPU wall) | 0.4 % | 17 ms |
| — collate (rest) / microbatch slice / value loss / actor sync / structural check / optimizer / policy CE + tail | | 15 / 9 / 8 / 7 / 5 / 4 / 2 ms |

**The 5.3 s/step of PERF-3b is not the trainer's cost; it is the mixing cadence.**
`train.training_steps_per_game: 1.0` with `max_train_burst: 1` takes ONE gradient step per
completed game (`_steps_budget`, `mixing.py`), so steps/h ≡ games/h and the thread waits in O5
between games. PERF-3b's 678 steps/h was the from-boot `iteration_complete.steps_per_hour`
average of a 13-min burst with a 6-min ring fill; in the stepping window this burst's rate is
**1,105 steps/h ≈ games/h**, which prices the **25,001-step block at ≈ 22.6 h**, not 37 h. The
sample-reuse ratio behind the cadence is 256 rows/step ÷ 38.6 rows/game ≈ **6.6 draws per row**.

**Ten steps under `torch.profiler` at run6 shape** (batch 256, the burst's 19,669-row ring, box
idle, GPU exclusive): **wall 0.244 s/step**; CUDA kernel time **0.156 s/step**; self CPU
0.161 s/step of which 0.139 s is `cudaStreamSynchronize` (31 syncs per step). Kernel shares per
step: `index_add_` (the GINE sum-aggregation, bf16 → `indexFuncLargeIndex`, atomics) 34 ms =
22 %; elementwise 19 + 11 ms; `mm`/`addmm` 25 ms = 16 %; `index_select` backward 17 ms;
`gather` 13 ms; `copy_` 14 ms; `add` 12 ms; `linear` allocates 7.8 GB of activations per step.
So the GPU part is ~160 ms, five times R349(d)'s "tens of milliseconds", and the message
passing (gather + scatter + their backwards ≈ 80 ms) costs three times the GEMMs — a real
INVESTIGATION-1 item 2 (segment-CSR sum, fp32 accumulation off the bf16 atomic path, fewer
syncs) — but at 92 % idle none of it moves the block's wall clock.

**R349(d)'s lever verdict: no code lever is S and ≥ ×2 inside the trainer, because the trainer
is not the bound.** A ×2 on the block's wall clock is either ×2 games/h (INVESTIGATION-1 item 3:
the serving ceiling, §E) or `training_steps_per_game 2.0` (a REGIME change — 13 draws per row —
the architect's, not a lever). The trainer term is measured at the mint regardless.

## D. The serving side, from the same record (INVESTIGATION-1 item 3's starting numbers)

The inference-server thread's 1,949 s: **32.1 % at `_node_offsets_to_batch_vec` (`gnn.py:44`)**
— `torch.repeat_interleave(…, counts)` with a device tensor of repeats, an implicit device→host
sync at the top of every forward (passing `output_size=` removes it); **18.7 % in check 14**
(`verify_edge_geometry`, the inline arm — the checker-thread lever is landed and NOT armed in
run6's mint); 14.5 % waiting in `next_graph_batch`; 4.9 % value decode; 6 % collate. The 16
Rust worker threads: **99.6 % in a futex wait** — the workers wait on the one server. The deploy
head (the eval instrument) spends only 28–40 of its 64-sim quick budget and 52–77 of 320 in
these decided positions (`gumbel_root_select` returns `None` early) — carried as an item, not
chased here.

## E. The mirror loop, proven over the alias (R349(b))

`tools/mirror_pull.py --source <alias>:/workspace/runs/startpath_cd --mirror … --run-id
startpath_cd --once`: cycle 1 in 11.1 s mirrored the run dir (9.4 MB) and receipted the
step-200 bundle's four files; the receipts landed beside the artifacts on the box and
`unreceipted_bundle_steps` read `[]` there. The first shard closed at the burst's death and is
receipted by the next cycle.

## F. Two more defects the burst surfaced, both fixed on this line

1. The launch-time anchor (`best_model.pt` at fresh init) was a BARE state dict; the resilient
   loader rebuilt the encoding's incumbent kind (`GnnArch`) and hit `size mismatch … [32, 1024]
   vs [32, 512]` on the V2 weights — a run6 resume before its first promotion died at the
   anchor. Every anchor is stamped now (`901139d9`).
2. The `gates.exit` set never ran a CUDA venv until `8dfa8b5f`; this burst's own `best_model.pt`
   is what exposed (1) — the class R349(a) names, again.

Provenance: box `/workspace/runs/startpath_cd/{drive.log,logs/events_startpath_cd_seg0001.jsonl,
pyspy_idle_native.raw}`, `/workspace/oc7/{alpha_probe.py,alpha_probe_warmstart.json,
trainer_profile.py,trainer_profile/}`; the dev box's scratchpad `startpath/` mirrors them.
