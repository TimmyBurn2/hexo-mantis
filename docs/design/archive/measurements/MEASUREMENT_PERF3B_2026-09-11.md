# MEASUREMENT — PERF-3b: the Gumbel regime's price on the rebuilt box (R348(d)), 2026-09-11

Instrument: the real composition root (`launch_run`) on a minted throwaway config = run6's own
28 deltas plus `search.kind gumbel`, `train.policy_target completed_improved_policy`,
`selfplay.mcts.n_simulations 320`, PCR `full_search_prob 0.25 / n_sims_full 320 /
n_sims_quick 64` (mean 128), `eval.gate.deploy_sims 160`, `gumbel_m 16`, `c_scale 1.0`,
`eval.concurrency 8`, 16 workers, warm-started from the LAW-12 strip of the BC artifact (same
net hash `2e72abd4…`). Box: Ryzen 9 9900X / RTX 5080, `torch 2.11.0+cu128`, tree `79b2b2c3`.
Driven in-process, not through `mantis.run`, because the box has no persistent volume and the
R347(d) halt refuses every preflight there (no stamp can exist) — stated, not hidden.

# READINGS

| quantity | value | window |
|---|---|---|
| **positions/h** (length-independent) | **43,400** | clean burst, steady after 5 min |
| **leaves/h** at the mean-128 regime (positions/h × 128) | **≈5.6M** | derived from the row above |
| games/h | 1,125 steady (1,470 whole-burst) | same; plies/game 38.6 steady (30.8 whole) |
| batch fill | 54.6% | `iteration_complete` |
| trainer | **678 steps/h ≈ 5.3 s/step** at 2.85M edges/step (batch 256) — CORRECTED 2026-09-11 (R311(c), in place): this is the from-boot `steps_per_hour` average of a 13-min burst with a 6-min ring fill, not the trainer's cost; the trainer thread is 92 % asleep and steps/h ≡ games/h at `training_steps_per_game 1.0` (0.986 measured over a 27-min stepping window, 1,105/h) — `MEASUREMENT_STARTPATH_2026-09-11.md` §C, falsified.md F-44 | `iteration_complete` |
| **25k-step block wall** | **≈37 h** — CORRECTED to **≈ 22.6 h** at the stepping-window rate (same note) | game-bound: 1 step/game by `train.training_steps_per_game`, the trainer idle between games |
| α (tail mass) per trainer step | mean 0.0006, p90 0.0002, **max 1.0**, 98.9% of rows carry a tail | `trainer_step.gumbel_tail_mass` |
| eval, fully escalated round at G=8, deploy 160/m16 | **264 games in 1,083 s = 4.10 s/game** (terminal round, uncontended) | vs R341(c)'s 7.09 s/game at PUCT-150 |
| eval, mid-run round | 264 games in 1,168 s (4.43 s/game), cut by the 200-step throwaway's close-out drain with only the random floor left | `eval_broken join_timeout` — an artefact of the burst's bound |
| terminal outcome | `wr_sealbot 0.281`, promoted, at step 200 from the BC warm start | one round, not a strength claim |
| **host memory, the K/MAX_NODES term** (R348(e)) | run process RSS **3.84 GiB at boot, 4.55 median, 4.70 max**; one process, 67 threads; GPU **7.66 GB max** during self-play + training (no eval round) | 120-step burst, 63 samples at 10 s |

## What replaces R347(b)'s prediction

R347(b) predicted ~450–640 games/h "serving-bound at ~4.4M leaves/h" and "~2–2.5 days per
25k-step block". Measured: **leaves/h ≈ 5.6M, so the serving prediction holds**; games/h is
~2× the prediction only because this net plays ~31–39-ply games, half the length the
prediction assumed — games/h is a length-dependent number and should not be the prereg's
wall-clock line. The block is **trainer-bound, not serving-bound**: ≈5.3 s per 256-row step
at 2.85M edges puts 25k steps at ≈37 h (≈1.5 days), inside the prediction.

## Caveats, in order of weight

1. The first burst's parent process survived its producer's death (the tail-mass defect, fixed
   at `79b2b2c3`) and overlapped the second burst for ~20 min; every self-play number here is
   from the third, **clean** burst on a box verified idle first. The eval numbers are from the
   second burst's rounds, which ran after the overlap ended.
2. `iteration_complete.sims_per_sec` bills at `n_sims_full` (320) per position under PCR and is
   an instantaneous per-drain rate (it read 3,198 / 6,381 / 12,705 across three samples); the
   leaves/h row is derived from positions/h × 128 instead.
3. **α max 1.0 rows exist** — rows whose 16 explicit entries carry no target mass at all. Not
   adjudicated here; it is the architect's reading against R347(a)'s "the completed-Q target is
   exact on the m sampled entries".
4. The checker-thread lever of R347(e) landed after these readings (`8443d0e5`,
   `inference.edge_geometry_check`); its A/B is recorded in §"A/B" below once taken. The Rust
   verifier holds the GIL (`PyReadonlyArray1` borrows, no `allow_threads`), which bounds what
   moving it to a thread can buy.
5. 150–200 steps from a BC warm start: the α and plies readings are early-training values.

## A/B — R347(e)'s checker-thread lever (`8443d0e5`), 120 steps per arm, same config and seed, box idle

| reading | `inline` | `checker_thread` | Δ |
|---|---|---|---|
| collate per part (the server-cycle term the lever targets) | 5.85 ms mean, 34,457 parts | **2.51 ms mean**, 31,209 parts | **−57%** (×2.3) |
| batcher queue wait per pop | 2.92 ms | 7.23 ms | +148% — the server now waits for fill |
| batch fill | 54.68% | 54.62% | unchanged |
| positions/h, steady | 41,069 | 42,369 | **+3.2%** |
| wall for 120 steps | 705.7 s | 673.0 s | −4.6% |
| steps/h (`iteration_complete`) | 613 | 643 | +4.9% |
| checks deferred / inline fallbacks / failures | — | 31,209 / 0 / 0 | every check ran, none dropped |

Reading: the lever does what R347(e) said on the server cycle — collate sheds 57% of its time
per part, more than the ×1.2 expected — but the END-TO-END metric moves ~3–5% because the serving
loop was not the binding constraint at this regime: fill stays at 55% and the pop wait rises to
absorb the freed time (LAW-09: the microbench is not the metric). Single run per arm; a 3% gap
is inside what two 12-minute bursts can resolve, so "≤ +5%" is the claim, not "+3.2%". The Rust
verifier still holds the GIL during the check; releasing it is the next lever on THIS thread, and
the 45% of batch capacity left unfilled is the larger one (HOT-14 / the carded server ceiling).
Arming `checker_thread` in run6 is a mint row; nothing here forces it.

## Artefacts

Box: the three burst directories under the box's workspace (`perf3b_run`, `perf3b_clean`, with
`out/logs/events_*.jsonl` and the drive logs); the analysis script beside them
(`analyze.py`); the A/B arms under `perf3b_ab/{inline,checker_thread}`; the host-memory burst under
`perf3b_hostmem` (`rss.log`); the minted throwaways `perf3b_*.yaml` in the dispatcher's scratchpad.
