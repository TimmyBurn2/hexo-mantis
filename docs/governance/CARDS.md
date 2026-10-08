# CARDS — the open work

Every card that is open at R346, with the ruling that last moved it. A card leaves this file by
being closed, refused or spent — never by going quiet. A closed card keeps its row, marked
CLOSED/LANDED with the date and the ruling, until a ruling sweeps its section (this sentence once
said closed cards are not kept here; ~20 were, so the practice is recorded rather than the claim);
the reasoning that closed one lives in its ruling entry in `docs/governance/RULINGS.md`.

Status words mean what they mean elsewhere in this repo: **BLOCKING** stops the thing it names;
**MINT-BLOCKING** stops a mint; **HELD** is waiting on a named event; **CARDED** is accepted work
with no date; **OWED** is a text or a value someone must supply.

## Opened by R347 (CLEANUP WAVE 2)

Both were found by running the gate set rather than by reading it, and both are pre-existing on
`dev` — neither was introduced by wave 2.

- **CARD-OC7-OVERRUN — DISCHARGED, not BLOCKING.** `tests/train/test_clean_stop_save.py`'s
  50-step OC-7 row and the AVX2 tier hang it exposed were root-caused as HOST, not code
  (`docs/design/measurements/MEASUREMENT_OC7_2026-09-11.md`): the dev box (Ryzen 7 3700X, AVX2,
  no AVX-512 BF16) ran LAW-06's bf16 autocast through ATen's generic path at 72× per GEMM. Fixed
  by CARD-TIER-HOST's carve-out — fp32 on `train.device: cpu`, landed at `9491b4d0` with its own
  parity test (`tests/train/test_law06_cpu_carveout.py`). OC-7 PASSES in 178.4 s inside its ceiling
  on the AVX-512 BF16 box (Ryzen 9 9900X, the measurement's row); on the AVX2 host the carve-out's
  fp32 path is what the parity test pins (no post-carve-out OC-7 row is measured there). `make gates.exit` completes: the 2026-09-21 exit sweep read the
  integration tier at 49 passed, none of it hanging.

- **CARD-GATE17-LOCAL-COUPLING — CARDED.** `tests/tools/test_gate_vacuity.py::test_an_empty_diff_degrades_WIDE_rather_than_printing_green`
  shells out to `rule7_gate.py --base HEAD` and asserts `returncode == 0`. Gate 17's local
  supplement is **untracked by design** (R312(e)), so this TRACKED test's outcome depends on
  whether an untracked file exists in the working directory and on what it contains: on a machine
  with the operator terms armed and any matching content in the tree, a tracked test fails. Seen
  live in wave 2 — the supplement was copied into a worktree branched before the redaction commit,
  and the test went red on content the tracked floor does not catch. The test was right and the
  tree was wrong, which is the good case; the bad case is the same coupling firing on a clean tree
  somewhere else. A vacuity test should assert the DEGRADE-WIDE behaviour without binding itself to
  the verdict of a scan whose pattern set it cannot see.

## Opened by the DEPLOY-2 packet (2026-10-08) — the floor read, evals on box A, the dash's units, run12's inputs

- **CARD-DEPLOY-2 — ORDERED by the architect's DEPLOY-2 packet under R388: branch `deploy-2` (`.wt/deploy-2`), on dev
  + fresh-2, box A for cells under the GPU-share cap, the desktop only to build.** A1 the floor read
  (CARD-FLOOR-READ-2); A2 the evals on box A (CARD-EVAL-ON-BOX-A); A3 the dash (CARD-DASH-UNITS); A4 run12's inputs
  (the S units of record on the GPU, the 1-ply plain start, the in-run gate's arena book, self-play's PUCT frame, the
  best-save rule); A5 gates at each leg, a fresh review, gates.exit, the push on the operator's word. run11's tree and
  config are excluded; a cell started without the cap halts the packet. Local records `mantis-records/deploy-2/`.
  EXITED 2026-10-08: A1 read (the floor binds), A2–A4 built, two fresh reviews fixed, gates.exit and gates green;
  exit record `mantis-records/deploy-2/EXIT.md`.
- **CARD-DASH-UNITS — OPENED by the DEPLOY-2 packet (A3): DASH-DYNAMIC lands and the dash reads its units from the
  records (the unit of record v legacy tags), served from box A behind the operator's tunnel.** Per save it shows rung
  16, S, the monitor's rows (GEN cf CE, T, gap, exams), positions/h and the gate rounds.
  BUILT 2026-10-08 (DEPLOY-2 A3): DASH-DYNAMIC landed; `--rule-unit` retired for the records' units file (`rule`,
  `second`, `legacy`, the rung state), read at every poll; the Run view lists every save with the ruler of record,
  the second ruler, GEN cf CE, T, gap, exams held, positions/h save to save and the gate round. Box A serves it
  (`mantis_dash`) from the deploy-2 tree.
- **CARD-RUNG-STEP-IN-TREE — OPENED by DEPLOY-2's reviews: R388(a)'s step (two consecutive saves above 0.7 → the next
  rung, never down) and the parent and anchor re-read at a step live only in box A's follower script.** They come into
  the tree, one implementation the follower and the dash share, before run12's going-forward line can meet a step.

## Opened by R389 (THE SLOPE, NOT THE FLOOR; THE CAPACITY PROBE; 2026-10-08) — DISTILL-1, the withdrawn floor, run12's replay row

- **CARD-DISTILL-1** (R389(c)). Label at least 1M positions from run11's rings with Six gen455's raw policy and value
  (MIT; R376(f): a means, lineage-tagged). Train 4×128, 4×256 and 8×128 at equal budget beside a 4×128 control on
  run11's own search targets and a permuted known-bad. Read held-out KL to the teacher; rung 16 and S as two-save panels.
  Pre-stated: distilled 4×128 above run11's best by +0.17 on both → run12 starts from it; only a wider net → run12 starts
  from that net; none → CARD-DENSE-1. run11 pauses one day for it.
- **CARD-DENSE-1** (R389(c), conditional). Design a hex-conv model kind behind the seam, only if DISTILL-1 finds no net
  above the line.
- **CARD-RUN12-REPLAY** (R389(d)). run12 carries replay_capacity 2M as a prereg row with its own two-save gap and T
  check; the supervised harness is the lab for architecture and auxiliary targets before a self-play run carries them.
- **CARD-FLOOR-READ-2 CLOSED** (R389(b)). Withdrawn: the slope, not the floor, is the read; run12's schedule is a plain
  cosine and no run is spent on it.
- **CARD-BOX-B-DESTROY** (R389(e)). No new box; box B is destroyed on the fresh bundle's verification (the operator's act).

## Opened by R388 (THE FLOOR READ, THE LINEAGE, THE ENCODING; 2026-10-08) — one rung, the second floor read, FRESH-2 accepted, run12, ENCODE-1

- **CARD-RULER-ONE-RUNG — ORDERED by R388(a): the ruler of record is one rung, Six gen455 at 16 nodes per turn v ours
  128/stone on the arena book (`ladder455_n16`).** It steps to 128 then 256 on two consecutive saves above 0.7, never
  down. S on the GPU is the second ruler; the S units of record move to the GPU (580/580 identical). Validity accepted
  (the 132k/156k pair unresolved on both). R387's N50 and equal-work parity read are withdrawn.
  The S units of record play on the GPU (DEPLOY-2 A4); `equal_work`'s GPU cells open a new series beside its CPU ones,
  a break in the label stated in the eval contract.
- **CARD-FLOOR-READ-2 — ORDERED by R388(b), pre-stated: four rung-16 cells of run11 at 228k, 240k, 252k and 264k.**
  RUN11-CYCLE-2's verdict stands as pre-registered (its window, a warm restart read at its high-LR start, was the
  architect's error). The four cells' mean above the plateau (−0.754 logit) by +0.17 → the floor binds and run12's
  schedule is one cosine 5e-4 → 1e-4 over 300k; else the floor stays and ENCODE-1 takes the budget.
  OWNED by DEPLOY-2 A1 (the packet calls it CARD-CYCLE-ANNEALED-READ): box A, one cell at a time under a GPU-share
  cap pre-stated before the first cell (local records `mantis-records/deploy-2/PRESTATED.md`), S beside at 240k and
  264k, the operator told the moment the read is in.
  READ 2026-10-08 (DEPLOY-2 A1): 228k/240k/252k/264k read 0.365/0.336/0.366/0.392 (logit −0.556/−0.681/−0.552/−0.437),
  576 games each, no forfeits, each checkpoint the monitor's sha. Mean −0.557 against the plateau −0.754: +0.197 over
  the +0.17 line → THE FLOOR BINDS, and run12's schedule is one cosine 5e-4 → 1e-4 over 300k. A thin pass: +0.027 over
  the line, inside the four cells' ±0.09; with the plateau's own error the climb reads +0.20 [+0.07, +0.33]. S at 240k
  and 264k is re-queued (CARD-EVAL-ON-BOX-A).
- **CARD-FRESH-2-FINAL-CELLS — ORDERED by R388(c), low priority: close FRESH-2's record. HANDED to DEPLOY-2 (box A)
  2026-10-08 with the commands, the sha and the one-instrument-per-pair rule (local records
  `mantis-records/run11-fresh/fresh2/FINAL_CELLS_HANDOFF.md`).** The final save 139 933 is read
  on rung 16, X and the old gen455@128 cell and paired with run11@132k (the pre-statement's annotation 5); B3 ("the
  lineage stands" on 96k/108k/120k, every instrument) is final when they land. Local records
  `mantis-records/run11-fresh/fresh2/`.
  DEPLOY-2 queues them on box A at the lowest priority, the run11@132k partners re-read on the same tree.
- **CARD-F64-NORMALISER — RATIFIED by R388(c): the improved policy's softmax normaliser accumulates in f64, landing before
  any from-scratch start.** The f32 sum dropped terms under half an ulp on roots with thousands of children
  (random-init boards), shipping targets summing to 1 + up to 2.8e-4, run-fatal under LAW-14. Commit `99a13798` on
  branch `fresh-2` (a unity test that fails on the old code; targets only, moves unchanged, no measurable cost).
  LANDS as `e7b9c9f3` with R388's record (a fresh review, no blockers).
- **CARD-PUCT-FRAME-SELFPLAY — ORDERED by R388(c): self-play's PUCT frame takes #13's fix (one implementation), pinned.**
  OWNED by DEPLOY-2 A4.
  BUILT 2026-10-08 (DEPLOY-2 A4): the switch is gone, both stone levels pinned, and a whole-tree golden recorded on the
  base tree holds Gumbel bit-identical (a planted interior break moves it).
- **CARD-RUN12 — ORDERED by R388(d): run12 = run11's best save by ladder-and-S agreement, on dev + fresh-2.** The arena
  book in the in-run gate; bookless evaluations from the origin stone alone (1 ply); 48 workers. It starts after
  CARD-FLOOR-READ-2's read, without waiting for ENCODE-1. Its going-forward line: four rung-16 cells above its parent's
  reading by +0.17, read every 12k; the two-read halts stay.
  Its best-save rule becomes a tool (DEPLOY-2 A4): the save first on rung 16 and S, else the first on rung 16 whose S
  interval holds the best S reading, the pick and both readings stated.
  Its schedule by CARD-FLOOR-READ-2: one cosine 5e-4 → 1e-4 over 300k. Its parent: `tools/best_save.py` over run11's
  rung-16 and S cells (on the re-base saves it picks 132k).
- **CARD-EVAL-ON-BOX-A — ORDERED by R388(e): cells, followers and the dash feed run on box A under a GPU-share cap.** The
  run's positions/h during cells is a row; above a 10 % daily cost an eval box replaces it.
  OWNED by DEPLOY-2 A2 (the packet's CARD-EVALS-ON-BOX): the cell followers (rung of record every 12k, S every 24k,
  the legacy gen455@128 tagged on two saves then stopped), the records writer and the dash feed as box A units; the
  desktop mirror best-effort.
  BUILT 2026-10-08 (DEPLOY-2 A2): supervisor units `mantis_cells` (the picker plays only checkpoints matching the
  monitor's sha; a retry moves its stale work aside; a cap stop re-queues without an attempt; no cell starts while run11
  is down) and `mantis_records` (sidecars, cost rows, the dash's units file). A rung cell takes ~25 min and holds run11
  at 153–168k/h against its cell-free 200.4k/h (−16 to −23 % while it plays, ~3.6 % of a day at one per 12k). S at four
  games in flight (one GPU strix process each) took the card to 203 MiB: 240k was stopped by the cap, 264k by hand,
  run11 unharmed; S now plays two in flight behind 5 GB free, its daily share read off its first cells. The cap and its
  four annotations: local records `mantis-records/deploy-2/PRESTATED.md`.
- **CARD-BOX-B-DESTROY — ORDERED by R388(e): box B is destroyed after the fresh bundle's mirror is sha-verified (the
  operator's act).** VERIFIED 2026-10-08 07:22 CEST: the 139 933 bundle in the mirror matches its manifest and box B
  (checkpoint `284abc88…`); box B's remaining FRESH-2 records are swept into `mantis-mirror/run11-fresh/`. Box B is
  ready for the operator.
- **CARD-ENCODE-1 — ORDERED by R388(f): prune empty→empty edges as a new encoding row behind the seam.** Registry,
  detector, conformance and a pre-registered witness; read by a fine-tune pair from run11's best save (pruned v unpruned
  control, equal recipe, 10k steps) on rung 16 and S at equal playouts, with the per-leaf and trainer-step cost beside;
  it adopts only if strength holds by the line. Supersedes CARD-RUN12-ENCODING's "#18 is run12's encoding".
  EXITED 2026-10-08, NOT PASS (branch `encode-1`, unpushed). The row `gnn_axis_r8_pruned` lands behind the seam (a
  required `empty_edges` key, the builder's verify, the mode carried to every leaf build, rings crossing only across the
  edge set) with the detector: `load_checkpoint`/`resume_trainer` take the declared encoding with no default and every
  serve/train path declares or adopts the stamp. Cost (desktop): edges per leaf 13 372 → 1 994, build −47 %, served per
  leaf −20 to −31 % GPU / −49 to −57 % CPU, trainer step −50 to −60 %. The pair (box A, run11 paused 13:26–18:26 CEST
  by the operator's word; parent 132k, 3-save panels, 576 games a cell): Δ rung 16 −0.41 [−0.55, −0.27], Δ S −0.62
  [−0.76, −0.49] against the −0.17 line; known-bad valid, control not degraded. The row stays a capability with no
  config consumer; local records `mantis-records/encode-1/EXIT.md`.
- **CARD-ENCODE-CONSUMER-CHECK — OPENED by ENCODE-1, CARDED: under a pruned spec the collate checks no edge membership**,
  so an empty pair a builder emitted would pass where `ProducerVerify::ConsumerEveryBatch` skips the builder's own
  verify. Defensive only (the builder cannot emit one); touches the protected collate.
- **CARD-PRUNED-CAPS — OPENED by ENCODE-1, CARDED for any pruned run: the fused and micro-batch caps and the served-graph
  bucket floors were fitted on kept graphs (~24 edges per node; pruned ~4)**, so pruned parts pad toward kept edge
  counts and E2's served gain is understated; a pruned run re-fits them (an arch-scoped mint).
- **CARD-LAW08-GATE11 — OPENED by ENCODE-1, OWED (an operator annotation): LAWS.md says LAW-08 is pinned "for encodings,
  gate 11", but gate 11 refuses silent encoding fallbacks and checks no consumer;** no test checks that a registered
  encoding is named by a config. `docs/contracts/registry.md` was repaired in place and states the pruned row's exception.
- **CARD-GIL-TEST-RACE — OPENED by ENCODE-1, CARDED: `tests/bridge/test_graph_wire_adv.py`'s GIL-release pin races** —
  its control's stall loop never runs when the worker finishes inside `start()`, so it read 0.0 ms twice inside a large
  batch and passed alone three times.
- **CARD-SEPARATE-TRAINER — CLOSED by R388(f): the drag is device time (GIL 11.7 %).**
- **CARD-PREFLIGHT-RANDOM-INIT-EVAL — OPENED by FRESH-2's exit, not built: a random-init config's preflight can time out
  in its terminal eval.** Its throwaway 101-step net can walk away (stones at the radius-8 edge, games to the 256-ply
  cap on ~22 000-cell graphs, ~2.5 s per ply): FRESH-2's first preflight timed out at 3 h. A guard (a ply-cap
  adjudication or a shorter cap for the preflight's terminal eval) wants a ruling before run12 or any from-scratch mint.
- **CARD-INTERIOR-ARGMAX-F32 — OPENED by FRESH-2's exit, not built: `mctx_interior_argmax_input` keeps an f32 softmax
  sum.** It feeds only an argmax (a uniform (1+ε) scale moves only ε-sized near-ties) and no unity check, so it was left
  as is; changing it would move search trajectories.

## Opened by R386 (THE FLOOR, THE FRESH LINE, THE INSTRUMENT; 2026-10-06) — run11 pauses, the instrument breaks once, DEPLOY-1

- **CARD-RUN11-CYCLE-2 — ORDERED by R386(a); EXITED 2026-10-07 (local records `mantis-records/run11-cycle-2/`): run11
  resumed 12:33 CEST from its 174 008 stop save on its second cosine cycle; the read says NO CLIMB — the floor was not
  the bound, run12 takes the budget.** The mint: `train.lr_cycle` {174 008, 5e-4, 1e-4, 54 000} (the operator kept the
  ruled 54k) and n_workers 48. The read (R387's rung 16 by the operator's word, pending the architect): 177k 0.293,
  180k 0.306, 183k 0.311, 186k 0.368 → mean −0.759 logit against the plateau −0.754, line −0.584. The post-resume
  saves are clean (exams held, no band miss, the gap under its line); the gate rounds at 180k and 216k promoted.
  Every pre-v57 tree refuses the resumed run's checkpoints (`train.lr_cycle` extra_forbidden); the cell trees carry
  the v57 schema.
- **CARD-CYCLE-ANNEALED-READ — CARDED by RUN11-CYCLE-2: a rung-16 cell at the cycle's annealed end (228k) and after.**
  The pre-registered window is the cycle's first 12k steps at LR 5.0 → 4.5e-4; 186k alone read +0.21 logit over the
  plateau (SE 0.08). A report row for the architect, not a re-opening of the verdict.
  SUPERSEDED by CARD-FLOOR-READ-2 (R388(b)): four rung-16 cells at 228k–264k, played by the DEPLOY-2 packet's A1.
- **CARD-TRAIN-SPEED-1 — ORDERED by R386(a); EXITED 2026-10-07: box A's closed-loop A/Bs (4 interleaved pairs each, the
  trainer closed-loop as production) and the knob sweep.** C8 (the Gumbel interior logits once, bit-identical,
  criterion −13 %) LANDED; #10 (quick-arm decided roots unsearched, value-only rows, `unsearched_decided_rows`) +1.1 %
  LANDED; DEPLOY-1's #4 builder +5.9 % LANDED; the ring-sample width 1 → 8 (step −23 %, positions/h −0.9 %) is a ready
  lever, NOT landed (CARD-RING-SAMPLE-WIDTH). The sweep (+3 % rule): n_workers 32 → 48 +5.7 % LANDED; batch 96,
  wait 5 ms, cache ×4 and sync cadence 75 did not clear it; solver budgets not run (~1 % of worker CPU). Realised
  steps/game stayed ≥ 2.3 throughout; exams and bands in on every A/B ring.
- **CARD-RING-SAMPLE-WIDTH — CARDED by TRAIN-SPEED-1: the trainer's ring sample at 8 threads, a cap lever.** It lands
  (a trainer-side width; the in-run eval's leaf-build reservation unmoved) only when a change drops realised
  steps/game below 2.3.
- **CARD-RUN11-FRESH-RESUME — ACCEPTED and CLOSED by R388(c): the lineage stands.** Resumed from 69 954 on 2026-10-07
  10:30 CEST, stopped at 139 933 on 2026-10-08 06:50 CEST on the operator's word; at matched steps 84k–120k it trails
  run11 on rung 16 (−1.04 to −1.39 logit), X and gen455@128; the final cells are CARD-FRESH-2-FINAL-CELLS. Was
  ORDERED by R386(b): RUN11-FRESH is VIABLE and resumes on box B for 48 h. Its cap band
  becomes a two-read halt at < 10 % with a slope row; its exam floors stay report-only until first passed. Its curve
  against run11's at matched steps, on (c)'s instrument, is run12's parent evidence. The FRESH-2 packet runs it in
  another session; the cap band's two-read rule is monitor code, owed before the resume.
- **CARD-INSTRUMENT-BREAK — ORDERED by R386(c): the instrument breaks once. DEPLOY-1's L4 builds it, as
  CARD-RULER-OF-RECORD and CARD-EVAL-OPENINGS-ARENA.** Its ruler is ANNOTATED by R387(a) and then R388(a): one rung,
  gen455 at 16 nodes per turn (CARD-RULER-ONE-RUNG).
  - The ruler of record is Six gen g at equal playouts per turn (ours 128 per stone, Six 256 per turn). g is the lowest
    generation that reads the shipped head inside [0.3, 0.7] on a 64-pair screen, re-picked upward with a three-save
    overlap when a panel leaves the band.
  - S stays the second ruler; equal-time cells are a report row.
  - Openings follow the arena protocol (the origin, plies drawn within hex distance 2, 5 plies, the 4-in-6 threat
    redraw, each opening twice with sides swapped) over 288 seeded draws; book_v1 retires.
  - The parent and three run11 saves are re-read on it, and the going-forward line is re-derived.
  - BUILT by DEPLOY-1 (2026-10-07); ANNOTATED by R387 and the operator's word (pending the architect): the ruler is ONE
    rung of Six gen455, `ladder455_n16`, stepping 16 → 128 → 256; S plays on the GPU. Re-base at rung 16: parent
    0.169, run11 108k/132k/156k 0.309/0.316/0.335 (anchor −1.591 logit, line −1.421). VALIDITY ACCEPTED by the
    operator's word (2026-10-07, for the architect to ratify): the strict four-rank match failed only on the 132k–156k
    pair, unresolved on both instruments; the three resolved pairs agree (tau 0.67). The ladder is the reading line; X
    stops.
- **CARD-DEPLOY-1 — ORDERED by R386(d): SEARCH-PERF-1's report
  (`docs/design/research/SEARCH_PERF_2026-10-06.md`) lands as DEPLOY-1. EXITED 2026-10-07 on branch `deploy-1`
  (push on the operator's word): L1 all six pass (a single-game GPU turn 535 → 251 ms), L2 move-identical, #13's pair
  Δ 0.000 logit (no halt), D11 −20.6 %, #5(b) a posture not a speed lever, #9 landed with its stand-in miss flagged,
  L5's VPS reads dropped (nothing runs on the VPS until a net is finished). Was: RUNNING 2026-10-06: the architect's DEPLOY-1
  packet, branch `deploy-1` in `.wt/deploy-1`, cells on the desktop, no box; legs L1 (the free fixes), L2 (the cache),
  L3 (the sign, #14's switch, #5's step (b), the producer-side verify), L4 (the instrument), L5 (the ladder's
  posture).**
  - The free fixes: #1–#4, #9 and the collision stop.
  - The per-game exact cache (#6): R370(c) extends to eval paths, since a per-player per-game exact cache is a
    served-output identity, not a reading.
  - The virtual-loss frame fix with B3's refill (#13), a defect. Its record is an equal-time pair against S before and
    after, and it rides (c)'s break.
  - The early stop (#14) as a per-consumer switch: on for the ladder, off for cells, the gate and analysis. The
    served-sims witness reads `last_sims == n`, or a fired stop with the leader fixed.
  - #5's step (b) now, and its step (a) after its proofs.
  - The producer-side `verify_contract` retires where check 14 runs 1-in-1.
  - #10 is permitted for quick-arm decided roots only (R275(b) relaxed for value-only rows).
  - #15, #16 and the per-turn deploy unit wait for run12's design; #18 is run12's encoding, landed behind the seam with
    a fine-tune pair.
  - Its exit states the two cards' owed rulings as one-line questions, which are then ruled (R386(f)).
- **CARD-STALE-EXTENSION — ORDERED by R386(f): the main checkout's stale extension is rebuilt in a separate tree.**
  perf-3-l1b stays a branch, and search-perf-research has been fast-forwarded (`dev` at `ef25e64f`).

## Opened by the DEPLOY-1 packet (2026-10-06) — the instrument's two halves, run12's encoding, the per-turn unit, the report's defects

- **CARD-RULER-OF-RECORD — OPENED by DEPLOY-1 for R386(c), OWNED by its L4: the ruler of record is Six gen g at equal
  playouts per turn (ours 128/stone, Six 256/turn).** L4 screens Six gens 150/200/250/300, 64 pairs each against
  run11@156k's shipped head; g is the lowest inside [0.3, 0.7]. It then re-reads the parent, run11@108k, @132k and
  @156k on (ruler g, the arena openings, L3's head), 288 pairs each, and S likewise once, and re-derives the
  going-forward line (the parent's new anchor + the line). Every old reading stays in the records with its instrument
  named. DONE as annotated under CARD-INSTRUMENT-BREAK: gens 150/200 read 0.180/0.211 (flat), the gen455 node screen
  0.344/0.336/0.273/0.242 at 16/32/64/128 (flat), so one rung at 16; the validity ruling is owed. RULED by R388(a):
  one rung, validity accepted (CARD-RULER-ONE-RUNG).
- **CARD-EVAL-OPENINGS-ARENA — OPENED by DEPLOY-1 for R386(c), OWNED by its L4: an arena-draw tool under tools/ (its
  path named here once tracked), the arena protocol exactly (the 18-cell region, odd plies, the 4-in-6 redraw, uniform
  over balanced draws), its unit tests mirroring the arena's `opening.test.ts` cases.** 288 seeded arena-5 openings are the cell book and "1 ply" the plain
  start; book_v1 retires, its file kept for re-reads. DONE: `tools/openings/arena_draw.py` matches the arena draw for
  draw (941 draws) and mirrors its 16 cases; `arena_s20261006_p5` has 0 solver-forced openings and no colour bias
  beyond chance. OPEN: "1 ply as the plain start" is not built (its reading is a question for the exit), and the in-run
  gate still names `book_v1` in every production config (a mint at the next config change).
  ANSWERED by R388(d): bookless evaluations start from the origin stone alone (the 1-ply draw), and the in-run gate
  takes the arena book; both OWNED by DEPLOY-2 A4.
  BUILT 2026-10-08 (DEPLOY-2 A4): a game with no opening plays `PLAIN_START`, pinned against the arena draw. The gate's
  book stays out of the dev template (moving it re-mints every config minted from it, run11's included); by the
  operator's word run12's mint states `eval.gate.opening_book: arena_s20261006_p5`, and the template moves when run11's
  configs retire.
- **CARD-RUN12-ENCODING — MOVED to CARD-ENCODE-1 by R388(f) (run12 starts without it). OPENED by DEPLOY-1 for R386(d):
  #18, pruning empty→empty edges, is run12's encoding.** 84 %
  of our edges join two empty cells (SEARCH_PERF §3.3 #18). It lands behind the seam (a new row in
  `crates/mantis-encoding/src/registry.toml`, a schema key, LAW-08/11/12, gate 11) with a fine-tune pair: pruned
  against an equal-recipe unpruned control (LAW-19), then equal-time cells. Not DEPLOY-1's to build.
- **CARD-DEPLOY-UNIT-PER-TURN — OPENED by DEPLOY-1 for R386(d): #15 (one tree per turn, the second stone continuing
  the chosen child) and SEARCH_PERF §7.2(l) (a per-turn deploy unit, Six's) wait for run12's design.** Both are
  STRENGTH changes that move every strength series; both need a ruling on what a stone's budget counts and on the
  served-sims witness for a re-rooted search (§3.6.2).
- **CARD-SEARCH-PERF-HYGIENE — OPENED by DEPLOY-1: the SEARCH_PERF §7.5 defect list, one card, each row with its
  owner.**
  DEPLOY-1's rows CLOSED 2026-10-07: D1 (the frame, gated to the deploy head and pinned at both stone levels), D2 (the
  first-collision stop), D3 (the standalone width), D6 (the panic payload), D11 (the verify skip, −20.6 % a build), D14
  (the stale threshold, gone with #1's rewrite), D15 (annotated in the CPU head profile).
  - D1 the virtual-loss frame at second-stone parents — DEPLOY-1 L3, with a test pinning the sign. R388(c) extends it
    to self-play's PUCT, one implementation (CARD-PUCT-FRAME-SELFPLAY).
  - D2 a select call burning its attempts after its first collision — DEPLOY-1 L1 (the exact stop).
  - D3 `leaf_build_threads` resolving to 1 on single-game hosts — DEPLOY-1 L1 (#2).
  - D4 the production trainer's ring sample on 1 thread (PERF-3's drivers used 10) — TRAIN-SPEED-1 (a cap lever).
  - D5 the main checkout's stale `_engine.abi3.so` — CARD-STALE-EXTENSION, rebuilt once DEPLOY-1's re-base frees the
    checkout from run11's cells.
  - D6 `par.rs` dropping a worker's panic payload — DEPLOY-1 L1, before #2 runs wide.
  - D7 no eval-path reader for a deferred check-14 failure — #5's step (a), after its proofs (R386(d)).
  - D8 the protected F-816-37 plant cannot vouch for a posture change — #5's step (a), its check-14-only plant.
  - D9 `gpu_wait` carrying a GIL re-entry and `queue_wait` the Rust fuse — TRAIN-SPEED-1 (read them as stage spans).
  - D10 `audit_exhausted` counting only total-budget exhaustion — OPEN, unowned (a LAW-18 row for the next leg that
    touches the audit).
  - D11 the producer-side `verify_contract` at 13.2 % of a build — DEPLOY-1 L3 (retired where check 14 runs 1-in-1,
    R386(d); its comment repaired).
  - D12 the retirer's result submit holding the GIL — TRAIN-SPEED-1, parked by R386(e).
  - D13 the self-play eval cache cleared at every actor sync — TRAIN-SPEED-1's knob sweep.
  - D14 `inference_local.py`'s stale "threshold of 32" — DEPLOY-1 L1.
  - D15 the CPU head profile's stale "32-leaf wake" — DEPLOY-1 L5 (a one-line annotation).
  - D16 run11's EXIT re-read wording — REPAIRED by RUN11-GO's session (2026-10-06).
  - D17 `queue_wait` read as slack in SEARCH-PERF-1's first draft — CORRECTED in the report.
  - D18 F-44's R365(e) note ("UNREAD" trainer duty) — TRAIN-SPEED-1, by annotation (run11 reads 0.69).
  - D19 CARD-PERF-TRAINER-CONTENTION carrying PERF-2's isolated numbers only — TRAIN-SPEED-1 (run11's −25 %).
  - D20 A7's kill resting on PERF-2's split — TRAIN-SPEED-1, decided by §7.2(i)'s split, parked by R386(e).

## Opened by R385 (THE RUN STARTS; THE RULE IS RETIRED; 2026-10-05) — run11 continues arm 2, RUN11-FRESH beside it

- **CARD-RUN11 (was CARD-RUN11-CONTINUE; the RUN11-GO packet's name) — PAUSED 2026-10-06 20:00 CEST at 174 008 by
  R386(a) (a resumable stop save right after the 174 000 bundle; CARD-RUN11-CYCLE-2 resumes it). RAN from 2026-10-05
  12:35 CEST (box A,
  `8d169abb`), ORDERED by R385(a):
  run11 continues arm 2 (2.4 steps/game, value mask 1/8; `run11a2`) from its final bundle (step 32 201, mirrored and
  hash-verified on the desktop) on box A, which runs nothing else (R385(f)).** Its config is unchanged, so its run id
  stays `run11a2` (prereg §13 A1); its envelope is §13 A5's and STATE's.
  The going-forward read PASSED at 60k (+0.64 logit against the +0.17 line; STATE has the cells) and at every read
  since, through +0.93 at 144k. The RUN11-GO packet EXITED 2026-10-06 and is ACCEPTED by R386(a). The LR schedule
  ended at 108k, and the rulers have flattened since. R386(a) pauses run11 at its next save and resumes it with a
  second cosine cycle (CARD-RUN11-CYCLE-2).
  RUN11-PRE's registered pick (arm 3) is set aside by ruling. The rate is a scalar inside the envelope [1.2, 2.4]: a
  memorisation gap above +0.05 at two consecutive saves drops it to 1.2 by STATE line. The going-forward read is the
  mean of four cells (the 32k final, 36k, 48k, 60k) above the parent's anchor by the line, then one cell every fourth
  save (R385(b)); the desktop plays the cells. Later levers land by the prereg's resume rule (R385(d)).
- **CARD-MONITOR-TWO-MISS — LANDED 2026-10-05 (`3a41ca6b`..`e7e9e848`, a fresh review's findings fixed), before
  either line launched. ORDERED by R385(b), owed before run11's monitor resumes with `--halt`: an exam floor halts
  on two consecutive misses (one miss arms, the next fires, a pass disarms).** `tools/run_monitor` at `0aca1e1a` fires
  a halting row on one miss. The same ruling makes L̄ report-only, reads arm comparisons as means over the last k saves
  (level, gap, policy CE) with cells pooled over ≥ 2 checkpoints, and makes matched work matched positions.
  RUN11-FRESH's floors report until first passed, then halt under this rule (R385(e)).
- **CARD-RUN11-FRESH — HALTED 2026-10-06 07:22 CEST at 69 000 by its cap band (0.052 against < 0.05; final save
  69 954, resumable, mirrored). VIABLE by B4:** its 48 h cell, the halt save (B3: the newest save at launch + 48 h),
  reads 0.609 against the parent's 0.594. This is a screen inside the noise, final at 2026-10-07 12:43 CEST unless
  resumed. R386(b) rules it VIABLE and resumes it on box B for 48 h (CARD-RUN11-FRESH-RESUME). Was RUNNING from
  2026-10-05 12:43 CEST (box B,
  `configs/run11fresh.yaml`). ORDERED by R385(e):
  rebootstrap answered by an arm, not for run11. On box B: arm 2's recipe and
  net shape, random init, no parent, an empty ring, the fill ramp.** Exam floors report-only until first passed, then
  halting under (b). The cap/draw abort is its early halt; if it fires inside the first hours, the BC start at F-07's
  setting replaces the random init. Read on the same monitor and cells; viable if it reaches the parent's X anchor
  within 48 h. Its result is the evidence for run12's parent, nothing more.
- **CARD-RUN11-STATUS-24H — ORDERED by R385(f): a one-screen status every 24 h** over run11, RUN11-FRESH and the cells.
  The first is the RUN11-GO exit report (2026-10-06, `mantis-records/run11/EXIT.md`).
- **CARD-QUICK-ARM-SIMS (the quick-arm lever, RUN11-PRE's arm 4, 32 sims at m 8) — CLOSED by R385(c):** 1.145×
  against the prereg's 1.15, and its final-save T4 row fired. Cheap placements are CARD-WITHIN-TURN-TREE-REUSE's.
- **CARD-RATE-FORK — OPENED by the RUN11-GO packet, later: a paired fork of run11 at one save into a 1.2 and a 2.4
  steps/game branch.** Not built and not ordered; it reads the rate at matched start, which RUN11-PRE could not.
- **CARD-DEPLOY-BATCH-WAIT — OPENED by the RUN11-GO exit, not built: a single-game inference posture for the ladder
  deploy and equal-time evals, apart from self-play's. OWNED by DEPLOY-1's L1 (#1, the wake on submission); the
  ladder's forward cap is its L5.** The inference server waits up to
  `inference.inference_max_wait_ms` (10, tuned for self-play) for company, and caps each forward at `max_in_flight` =
  `selfplay.leaf_batch_size` 8.
  - In one game the wait costs about 1.75× per turn at 64–256 sims/stone (box B, same pairings).
  - The cap holds batched throughput at about 1 200 positions/s on the GPU and 44/s on the CPU, where Strix reaches
    about 100/s.
  - Measured with scratch overrides only (EXIT.md §5.4–5.5).
- **CARD-SEARCH-PERF — CLOSED 2026-10-06: SEARCH-PERF-1 is ACCEPTED by R386(d) and lands as the DEPLOY-1 packet
  (CARD-DEPLOY-1). OPENED by the RUN11-GO exit: the SEARCH-PERF-1 research packet (another session), our search
  against Six's.** At equal time Six gen455 does about 2–4× our nodes per second with a net about 16× ours. Its report
  is the card's output.

## Opened by R384 (THE RUN STARTS FROM THE READ; 2026-10-04) — RUN11-PRE is run11's first leg

- **CARD-BUBBLE-STUDY — ACCEPTED by R384(c) as evidence on R383(f)'s terms (rank and card, never adopt)** (local report
  `mantis-records/research/BUBBLE_2026-10-02.md`). Its two corrections land as annotations: `book_v2` is CLOSED on
  CARD-EVAL-ROUND-OVERRUN and R354, and R381(c)'s carry is re-stated on R381. Its lesson is adopted: the production run
  is the default state from run11 on. The censuses' serial sequencing is the architect's error, on the ledger.
- **CARD-BUBBLE-RULER — RECOMMENDED by R384(e): Bubble 185k against S and X, 288 pairs each.** Running its weights is
  the operator's act. At or above our shipped head, it becomes the third ruler.
- **CARD-REG-1-PUSH — PUSHED 2026-10-04 (`origin/dev` = `ad3682d7`). ORDERED by R384(g): `reg-1` rebases onto `dev`,
  gates, pushes.** Rebased 2026-10-04 onto `dev` `9da49bdd`; local gates 24/24 green on `ad3682d7` before the push.
- **CARDED by R384(f), not built before the start** (the RUN11-PRE packet's names in brackets):
  - CARD-WITHIN-TURN-TREE-REUSE [CARD-TREE-REUSE-TURN] — within-turn tree reuse; RE-STATED by R385(c) as the
    cheap-placement lever, carded for a pinned resume after the start;
  - CARD-HEAD-TO-HEAD-LEVER-CELLS [CARD-H2H-LEVER-CELLS] — head-to-head lever cells;
  - CARD-PAIR-DIAGNOSTIC — the pair diagnostic;
  - CARD-SIX-TRAINER-READ — the Six-trainer read;
  - CARD-RULER-HYGIENE — ruler hygiene;
  - CARD-SEARCH-VALUE-TEACHER [CARD-VALUE-TEACHER-CALIBRATED] — the calibrated search-value teacher, gated on a warmed head
    (CARD-FRACTIONAL-VALUE-TARGET-WARMUP);
  - CARD-ALL-VETOED-TARGET — the all-vetoed target fix, evidence only; R380(b) stands.

## Opened by the PERF-3 packet (2026-10-04) — batch prep, the cache key, trainer syncs

- **CARD-PERF-3-PACKET — RATIFIED by R385(d); the arms ran on this code, so run11 needs no landing. EXITED 2026-10-04,
  merged and pushed at `e6d1fb61` (local records `mantis-records/perf-3/`): L2 and L3 LANDED, L1 CARDED. The production loop with the trainer reads 156 340 -> 170 940 positions/h (+9.3 %, IQRs
  [153 840, 160 785] and [164 250, 179 760]).** L0's profile ranked L2, L1, L3: the loop is latency-bound, popping at half
  a batch (B 36) with the workers ~85 % blocked on inference and ~8 of 32 CPUs busy.
  - L2 (`6ed55ef3`): the eval cache keys a leaf by a Zobrist over every input `build_leaf_graph` reads, before any
    build; a hit skips the build and no SHA-256 is taken. Per-leaf worker CPU of the touched stages 908 -> 523 µs,
    workers 5.0 -> 3.5 cores. A seeded 10 869-leaf drive misses exactly the leaves the graph hash missed. R370(c)'s "a hash of
    the encoded input" is annotated (A1, the operator's word to merge).
  - L3 (`f7d40604`): the trainer step's host syncs 23 -> M + 1 (2 at production shape); serving step 484.5 -> 396.8 ms
    (-18 %), idle flat; losses, events and weights byte-equal over 4 steps on sm_86 and sm_89.
  - L1: CARD-PERF-COLLATE-2.
- **CARD-TRAINER-ONE-READ-GUARD — CARDED by PERF-3: one device read per step needs the protected microbatch guard moved
  after the backward.** The guard skips a non-finite microbatch BEFORE its backward, so it reads the device once per
  microbatch. Deferring it (backward always, the verdict read with the clip norm, a redo of the finite parts when one is
  not) keeps the outcome but moves a protected mechanism: a ruling's call.
- **CARD-PERF-TRAINER-PREFETCH — CARDED by PERF-3's L0: an idle trainer's GPU is busy 66 % of its step.** The ring
  sample (40 ms) and the collate (45 ms, 35 of it Python semantic checks) run serially before each step's forward. A
  prefetching thread would overlap them with the previous step's GPU work; at a fixed trainer rate in the shared
  process its production effect is the GIL and CPU it moves, unmeasured.
- **CARD-PERF-LOOP-SPREAD — CARDED by PERF-3: the production loop's between-run spread is wider than its IQR.** Five
  base runs of one tree over one night read 148 200 - 162 190 positions/h (~±5 %) while each run's 1-minute IQR is
  ~3-6 %. A landing read of production takes interleaved A/B runs (two per arm) or pools them. The scratch
  per-leaf counters read ~5 % low themselves (32 workers' atomics on one cache line).
- **CARD-PERF-B128 — MEASURED by PERF-3's L4(b), no lever.** Under the bucketed served forward B 128 serves 4 374
  [4 347, 4 927] leaves/s against ~5 200 at B 64: the minted caps split a B-128 pop into two parts and the slice costs
  ~9 ms of a 20.9 ms launch. Its drift against B 64 on 512 positions: 3 logits differ (max 0.125), values exact, 0
  argmax or sign flips; LAW-06 (ii-b) 0.043450 at both, inside the 0.0511 line.

## Opened by the PERF-2 packet (2026-10-03)

- **CARD-PERF-TRAINER-CONTENTION — CARDED by PERF-2's L4: the trainer in the process costs serving a third.** At 1.2
  trainer steps/s the loop reads 149-156k positions/h against 235k alone, the device wait per pop 3.97 -> 6.6 ms. A
  highest-priority serving stream does not move it (-1 %). Untried: pacing the trainer against serving, a second
  process under MPS, the trainer's kernel sizes. PERF-3 (2026-10-04): its L3 removes the step's per-scalar syncs (each
  one waited on the shared stream's serving work), so the serving step falls -18 % and the in-loop step 0.392 ->
  0.357 s at the same 1.2 steps/s; its L4(a) reads the check-14 thread at 1.1 % of trainer wall (0.357 vs 0.353 s off,
  positions/h +1.7 % inside the IQR, 0.64 cores) — under the 5 % that would earn a ruling request. PARKED by R386(e)
  except its first step: a read-only GIL sample on the live loop, on the operator's leave. RUN11-CYCLE-2 (2026-10-06,
  on a replica of run11's loop, run11 being paused; the operator's choice): the GIL is held 11.7 % of wall — the
  server 5.5, the retirer 3.3 (its result submit), the trainer 1.3; the trainer's busy time is ~35 % its GIL-free ring
  sample and ~44 % waiting on its own GPU work. The interference is the device: GIL-side remedies reach ≤ ~3 % each, A7's
  revival line is unmet (~0.6 ms per pop); a 25 % lower trainer busy share (8 sample threads) moved self-play by −0.9 %.
- **CARD-PERF-COLLATE-2 — CARDED by PERF-2: after L3 the server's launch is its collate. PERF-3's L1 (a second server
  thread on the one queue, branch `perf-3-l1` `0e834fa5`) NOT LANDED: B-64 cell +5.0 % (A-B-A +4.3 %) against a +10 %
  line; one production pair +5.4 % (IQRs disjoint, inside the loop's between-run spread). Pops shrink (B 64 -> 54 in the
  cell, 37 -> 31 in the loop) as two consumers split the queue. A revival owes its review's findings: a LAW-18
  thread/overlap counter, a dump stamp unique across threads, the event manifest's `pipeline` block, locked test
  batchers. The ready-block design is out: a device block fused past the wire payload breaks what the 1-in-1 checks
  verify. CARDED WITH ITS BRANCH by R385(d): the revival `perf-3-l1b` (`402b0372` on `dev` `e6d1fb61`, every review
  finding fixed, gates 21/21, a fresh review clean) read NOT PROVEN by its pre-stated rule on box A, run8@45k with the
  trainer at 1.2 steps/s, medians of 5 one-minute windows: dev 172 320, L1 174 420, L1 188 580, dev 177 119
  positions/h. Each L1 loop above each dev loop FAILS (174 420 < 177 119); the pooled +3.9 % (+3.4 % whole-window)
  passes its +3 %; the LAW-18 counter was not read in production. Both windows put L1 at +4 to +5 %, inside the
  loop's ±5 % run-to-run spread (CARD-PERF-LOOP-SPREAD).** Per B-64 pop the pack takes
  3.51 ms and the semantic checks 1.83 of a 6.72 ms launch. Coding edges by value in the pack costs 1.45 ms over L1's
  copy; the builder emitting codes on the wire removes it (a wire contract change). The semantic checks 15-16 can
  follow the structural ones into the Rust pass. The pack's 4-thread split reads inside the IQR of one thread under 32
  workers: a removal candidate, with its own bench.
- **CARD-INDUCTOR-CACHE-SCHEMA — CARDED by PERF-2: torch's compile-cache key does not cover a custom op's schema.** A
  tree whose `mantis::gine_message_sum` takes six arguments loaded a later tree's compiled module (seven, `code=None`)
  from the shared `/tmp/torchinductor_<user>` and failed at its first compiled call: loud, never silent. A host that
  runs trees from both sides of PERF-2's L2 gives each its own `TORCHINDUCTOR_CACHE_DIR`.

## Opened by R383 (THE REUSE READING; 2026-10-02) — the last two reads before the mint

- **CARD-REG-1 (CENSUS-3's proposed CARD-VALUE-REGULARISER) — CLOSED: ACCEPTED by R384(a). run11's value loss takes
  the re-drawn per-step mask at p = 1/8; shrinkage, the head-LR factor and the fixed mask are killed for run11; the
  anchor stays unresolved. EXITED 2026-10-03 (local records `mantis-records/reg-1/`).**
  - **The pick by the hashed rule:** the re-drawn per-step value mask at 1/8 (keep 0.1305, ≈ 1 value exposure per label
    at 8 policy draws).
    - Δ −0.0251 [−0.032, −0.018] calibration-free on in-distribution held-out games, effect of record −0.018,
      Bonferroni-clear.
    - Policy Δ −0.0003. Calibrated exams 0.289 / 0.295, against floors of 0.154 / 0.100.
    - The whole gain sits in plies > 40.
  - **At 4 draws the direction holds:** −0.015.
  - **The other masks are detections:** fixed 1/8 −0.0185, re-drawn 1/4 −0.0173, fixed 1/4 −0.0140.
  - **R1 v R2:** "distinct boards", resolved but under 0.012.
  - **Two-hot shrinkage** kills the parent's value head (0 of 32 units).
  - **The value-head lr factor** shows no effect of 0.012 or more.
  - **The 0.5 parent anchor** is unresolved.
  - Census box-h 48.4 against the packet's 36, by the operator's extension.

  Was ORDERED by R383(d), MINT-BLOCKING: the regularisers at run11's reuse, the first of the last two reads before
  the mint.
  - **Data:** fresh parent games under run11's regime, read at run11's reuse (8 draws per row). GEN self-plays
    run8@45k frozen under arm A's tactics config (F2 out), with the v3 producer, ≤ 4 box-h.
  - **Instrument:** an IN-DISTRIBUTION held-out split (unseen games from the train actors' window, split by game id),
    so the gap reads memorisation and not drift from the parent, the confound CENSUS-3's EXIT names. The 48k ring
    stays the near-parent second held-out, reported.
  - **Design:** 5 seeds and a fixed-subset known-bad (Rbad: 25 % label corruption on the same rows every draw).
    Census jobs ≤ 36 box-h (the packet's grant, inside R383(g)'s 40); keep or stop is the operator's.
  - **Arms**, against a plain control:
    - the fixed per-game value mask, k ∈ {4, 8};
    - the re-drawn per-step value mask, p ∈ {1/4, 1/8};
    - the value-head LR factor, m ∈ {0.5, 0.25} (CARD-VALUE-HEAD-LR, folded here);
    - label shrinkage, ε ∈ {0.1, 0.25}, with its re-drawn noise twin as a report row;
    - a lagged-value anchor, 0.5·z + 0.5·the parent's raw value.
  - **Pick:** the best arm whose CI clears the control by the line, whose policy CE is within 0.012 of it, and whose
    calibrated exams hold the run8-panel floors. The top two re-read at 4 draws per row. If nothing clears, the step
    rate alone carries to RUN11-PRE.
  - **Its two free reads:**
    - the field's semantics on run8's `search_stats` (CARD-RING-V3-SEMANTICS);
    - the calibrated exam floors (CARD-EXAMS-CALIBRATED).
- **CARD-RUN11-PRE — CLOSED: ACCEPTED by R385(a), its registered pick (arm 3) set aside by ruling; run11 continues
  arm 2 (CARD-RUN11-CONTINUE). EXITED 2026-10-05 (local records `mantis-records/run11-pre/` EXIT.md + DONE_REPORT.md):
  four arms ran their registered hours; L1 pushed with the arming at `0aca1e1a`, gates.exit green on `6a1156b7`. Its
  packet is RUN11-PRE (architect, 2026-10-03). RE-SPECIFIED by R384(d): run11's first leg, not a twin.**
  - The packet's legs: L1 on the desktop (L1a the mask, L1b the ramp, L1c the field and the key, L1d the origin at
    deploy, L1e the run monitor, L1f the template and the four drafted arm configs, L1g the prereg); L2 the arms on two
    matched boxes, ≤ 40 box-h plus preflights.
  - Four arms from run8@45k under one pre-registration:
    1. 2.4 steps/game, mask off — the control;
    2. 2.4, mask 1/8;
    3. 1.2, mask 1/8;
    4. 1.2, mask 1/8, quick arm 32 sims at m 8.
  - Matched wall-clock on matched hardware (two matched boxes, R384(g); renting is the operator's).
  - Read on the value instrument (paired lagged read + the GEN anchor), policy CE on fresh games, calibrated exams, ring
    bands, positions/h; shipped-head cells v the parent at the end as a screen.
  - The pick rule and halts are hashed before any arm starts; the picked arm continues as run11 from its last save.
  - Arming the configs is the operator's word. Trainer contention is a reason to read the step rate live, not to wait
    (R384(b)).

  Was ORDERED by R383(d), MINT-BLOCKING, after CARD-REG-1's exit: the step rate read live at matched throughput on the
  instrument. REG-1's exit asks it to run with the picked mask on, plus a mask-off control (a
  1/8 value gradient may track a moving actor more slowly, which a fixed-data census cannot price).
- **CARD-VALUE-MASK-REDRAWN — RUN11-PRE's L1a builds it as `train.value_mask_redraw_p` (0 = off). ORDERED by R384(a)
  for RUN11-PRE's arms 2-4: run11's value loss uses it at p = 1/8. Was
  PROPOSED by REG-1's exit, not built: the mint's trainer knob for REG-1's pick.** It is a
  seeded per-step mask on the sampled batch's `value_valid` at keep p, applied ahead of the whole-batch value
  denominator, so a kept row carries 1/p weight. A plain value-loss weight is a different, untested lever.
- **CARD-FRACTIONAL-VALUE-TARGET-WARMUP — CARDED by R384(a): a fractional target needs a warmed head; shrinkage is
  killed for run11. Was PROPOSED by REG-1's exit: concentrated interior value targets can kill the
  parent's value head.**
  - The evidence: two-hot shrinkage (±0.8 / ±0.5) took a first loss of ≈ 73 and left 0 of 32 units.
  - The scope: λ > 0 mixes, CARD-SEARCH-VALUE-AUX and short-term value outputs need a head warm-up or a gentle start
    first.
  - REG-1's 0.5 anchor (spread fractional targets) survived, so the trigger is not mapped.
- **CARD-LABEL-NOISE — OPENED by the RUN11-PRE packet's forwarding (R384). PROPOSED by REG-1's exit: re-drawn ±1 flips at 0.25 read the best held-out value point estimate
  (−0.026), but leave the head badly underconfident (T 0.38).** It is a lever of its own, not run11's. Its packet follows REG-1's exit. CENSUS-3 read generalisation on fixed rings, not
  learning per hour.
- **CARD-WARMSTART-RING-FILL — RUN11-PRE's L1b builds it (`min_buf_size` stays 4096). ORDERED for the mint by
  R383(c).** Training steps per game scale with ring fill
  (tspg × rows/capacity), which holds draws per row at the steady state from the first row. R381(e)'s `min_buf_size`
  100k is withdrawn.
- **CARD-EXAMS-CALIBRATED — its floors are the HALTING ROWS of record by R384(a), halting on two consecutive misses by
  R385(b) (CARD-MONITOR-TWO-MISS). LANDED 2026-10-03 by REG-1's free
  read 0b: floors of record are T4 V 0.154 and DEF V_att
  0.100 (the mean − 3 SD over run8's seven panel saves, each net's T read on GEN's held-out ring).** On it, the parent
  reads 0.286 / 0.210 and the known-bad 0.11 / 0.09; the DEF separation is marginal. Was ORDERED by R383(a): T4 V and
  DEF V_att are read on calibrated values from now on, floors re-derived calibrated on run8's panel.
  - CENSUS-3's post-hoc read is V_cal = tanh(atanh V / T), with the net's held-out temperature.
  - On it the raw floor rewarded overconfidence: D8 0.382 → 0.208 calibrated, D2 0.352 → 0.321.
- **CARD-RING-V3-KEY — RUN11-PRE's L1c renames it `train.value_target_search_weight`. ORDERED by R383(e), before any
  mint; restated by R384(a).** `train.value_target_lambda` is renamed to weight the
  search (0 = z). run11 trains at search weight 0; the field is recorded.
- **CARD-RING-V3-SEMANTICS — RULED by R384(a): the field stores Σπ′·completedQ with the proof override; RUN11-PRE's
  L1c moves the producer. OPENED by
  R383(e): what the root value field holds. READ 2026-10-03 by REG-1's 0a: on
  run8's sampled undecided `search_stats` rows, Σπ′·completedQ beats root W/N against z by −0.011
  [−0.012, −0.010] in every band, and raw v is worst. The TD targets only read self-consistency. The producer follows
  at RUN11-PRE.** The field's semantics (root W/N v
  Σπ′·completedQ) are read on run8's `search_stats` at 0 box-h (REG-1's free read 0a), and the producer follows the
  read. If `search_stats` lacks the children's π′ or completed Q, the read waits for RUN11-PRE's producer.
- **CARD-FORGE2-ANNOTATIONS — LANDED 2026-10-02 by R383(f): FORGE-2's corrections to the record are annotations**
  (its report §5, `mantis-records/research/FORGE2_2026-10-01.md`, local).
  - R382(a)'s void covers raw MSE and MAE. It is annotated on R382's entry and on
    `docs/design/measurements/CENSUS1_2026-09-29.md` (C5's value MSE and C6's MAE). That doc also carries C5's
    coords-off Δ as policy CE, `is_full_search` not being the drawn arm, and `root_offence` running on both arms.
  - `value_targets.py` was deleted at `482684f2`, annotated where `docs/design/research/STRENGTH_RESEARCH_2_2026-09-21.md`
    cites it.
  - CARD-EMA-SHADOW carries the arming fact.
  - CARD-ORIGIN-RULE carries the untrained forced ply-0 row.
  - The temperature convention note sits on CARD-VALUE-INSTRUMENT.
  - The local records carry their own lines: the training-path audit and LEVERS (KataGo's td horizons, Lc0's sampling
    ratios, C15's ≈ 75 Elo, hex pairing at k ≥ 7, the §8 swing as cell noise) and CENSUS-2's regime-check rings.
- **CARDED by R383(f), from FORGE-2's survivors (derived numbers rank and card, never adopt):**
  - **CARD-VALUE-HEAD-LR — KILLED for run11 by R384(a). Was FOLDED into CARD-REG-1** as its value-head LR arm.
    - The head gets its own AdamW group.
    - `FlooredCosineAnnealingLR` holds one `eta_min` for every group, so the group's floor eta_min × m is set
      harness-side.
  - **CARD-SAVE-AVERAGING-OFFLINE — CARDED, no box-h before run11: snapshot averages of run11's own stamped saves,
    built offline with the existing averagers, each read against its own raw save.**
    - `train.ema` stays off: arming it is not a shadow (CARD-EMA-SHADOW).
    - Prior ≈ 0: EMA5 −0.10 logit on X, run8swa +0.26 on S, both void or descriptive.
    - It needs ≥ 15 non-overlapping pairs and a known-bad with a measured fall.
  - **CARD-DEPLOY-PUCT-ABLATION — CARDED as a deploy packet's census: c_puct 1.5 and fpu 0.25 were never ablated.**
    - The overconfident raw scale makes the deploy search ≈ 2.7× greedier near v ≈ 0 than a calibrated one.
    - It runs as a named c_puct/fpu ablation behind a desktop move-change pre-gate.
    - It needs a mirror known-bad and power stated before cells.
  - **CARD-SHORT-TERM-VALUE-POSTMINT — CARDED post-mint: short-term value / error outputs.** They are the
    precondition of uncertainty-weighted playouts, read on run11's own rings and adopted only at a re-mint. An aux
    leaves z's per-game gradient intact, so it is an enabler, not a value lever.
  - The segment rule as CARD-TACTICS-LABELS-ONLY's text (that card).
  The rest of FORGE-2's proposals are killed as run11 levers.

## Opened by R382 (THE BLIND METRIC; 2026-10-01) — the value instrument, then CENSUS-3

- **CARD-VALUE-INSTRUMENT — EXTENDED 2026-10-03 by REG-1 under its grant:
  - a cross-fitted Platt CE;
  - `calibrated`;
  - `compare` CIs with detection, TOST and per-band rows;
  - a `lagged` read between one run's saves (run11's monitor);
  - an `exams` read at the held-out temperature.
  It was LANDED 2026-10-02 as `tools/value_instrument` (read/compare; branch `census-3`), the value
  reading of record by R383(a); its T4 V and DEF V_att read on calibrated values from R383(a) (CARD-EXAMS-CALIBRATED).
  Was ORDERED by R382(b); it reads its known-bad before any arm runs.** Calibration-free
  held-out value CE with the temperature fitted on disjoint games, AUC, the temperature and the train/held-out gap as
  rows, by ply band (0–10, 11–40, > 40); T4 V and DEF V_att co-primary; rings the parent never trained on; ≥ 3 seeds;
  the known-bad reads worse by ≥ 2 seed SDs. The same rows are read at every run11 save. Raw held-out value CE rules
  on nothing (R382(a)).
  - The temperature convention (FORGE-2 §5.9): the instrument DIVIDES the logit, so T > 1 is overconfident. CENSUS-2
    multiplied it, so its 0.37 is 1/T. Witnesses state the convention or read convention-free.
- **CARD-CENSUS-3 — CLOSED by R383(a): EXITED 2026-10-02 (local records `mantis-records/census-3/`). I0 PASS; the
  parent's head stands (shape bounded at ≈ 0.009); 8 draws per row over-fits the value head on held-out games (D2 − D8
  −0.031, effect of record ≥ 0.022) and the 13-draw warm start is the worst point; decay and the 64-sim W/N target null
  at this regime. Its levers move to CARD-REG-1 and CARD-RUN11-PRE. Was ORDERED by R382(d), MINT-BLOCKING: nothing mints
  before it reads.** It reads, on
  CARD-VALUE-INSTRUMENT with 3 seeds and in order of mechanism (R382(c)): a position-specific value target (z mixed
  with the search's root value); the value label's reuse per game, read as a curve; the head's shape and activation
  re-initialised and warmed; weight decay. Its picks set run11's value rows (R381(e), re-pointed). It needs the box
  within a day, ≤ 32 box-h (R382(f)); keep or stop is the operator's.
- **CARD-RING-V3 — RATIFIED by R383(e); its key is renamed before any mint (CARD-RING-V3-KEY). EXITED 2026-10-01 on branch `ring-v3` (from origin/dev `80b54f5a`, code tip `de80ea35`), awaiting
  the operator's fast-forward; gates.exit green. Was ORDERED by R382(d), MINT-BLOCKING: the root value field lands with
  default-fill before the mint.** It discharges CARD-RING-MIGRATION's v2→v3 path for the field CARD-SEARCH-VALUE-AUX
  and CENSUS-3's first lever need.
  - HEXG v3 (contract #6, repo_design amendment): every row carries the search's root value in the row-mover's frame
    and its flag; a v2 ring loads with every flag 0, and its v3 re-save minus the two fields is the file byte for byte.
  - Self-play writes it: the search's W/N (bit-equal to the 1-in-N `search_stats` value, which stays a cross-check),
    or a decided root's proof value (+1, and -1 lost on cover); the drain, `push_graph` and the facade carry it.
  - `train.value_target_lambda` (run config v52, template and every config at 0.0): λ·v + (1−λ)·z on rows with a root
    value, z on the rest, in the TRAIN step; the eval step reads z whatever λ. At 0.0 the trainer is the v2 trainer
    bit for bit. Fire-rate `trainer_step.root_value_rows` / `root_value_rows_moved` with λ echoed (manifest v9).
  - Witnesses (local records): the sha-pinned held-out ring with run8@45k trains byte-equal across trees at λ = 0;
    all 135 v2 rings in the mirror load default-filled and byte-equal, 141 resume bundles and 8 parent stamps load.
- **CARD-HELDOUT-GAP-Z — PROPOSED by RING-V3's exit, not built: a train-side value CE against z.** At λ > 0 the
  `heldout_gap` event's train side is the mixed-target CE while its held-out side reads z, so `gap_value` is not z
  against z and rules on nothing; the fix publishes a z-CE from the train step, which moves the seven-key `loss_info`
  contract. Needed before any λ > 0 run reads its gap.
- **CARD-RESEARCH-FORGE-2 — ACCEPTED by R383(f): five survivors carded, the rest killed as run11 levers, its record
  corrections owed as annotations (CARD-FORGE2-ANNOTATIONS). Was ORDERED by R382(d), beside CENSUS-3, read-only.** Nothing it proposes is adopted without
  a census.

## Opened by R381 (THREE TWINS, ONE LESSON; 2026-10-01) — F2 killed, the desktop work before run11's mint

- **CARD-HYGIENE-1 — RATIFIED by R382(e), the operator-granted protected-set edits included; `dev` fast-forwarded
  to its tip `7978cf1b`. EXITED 2026-10-01 on branch `hygiene-1` from origin/dev `b1e34aa4`; gates.exit ALL GREEN on the card. Every item landed, H1–H9 and all of H5.
  The five that edit ruling-protected tests or their harness (H7, H9, `recency_weight`, `policy_loss_trough_abort`,
  screen/confirm) landed under the OPERATOR'S GRANT of 2026-10-01 to touch protected pins where each pinned invariant is
  kept; a fresh review judged every touched protected test kept or strengthened (the pair-statistics pin now binds the
  GSPRT's live aggregate and its LLR). The schema went 179 → 159 leaves (contract v51, event manifest v8; rc 49
  retired) and run10's resolved config lost exactly eighteen dead keys, no value changed. Exit record local.
  Was IN PROGRESS from 2026-10-01, no run and no cells, ORDERED by R381(d), on the desktop before the mint, in parallel
  with CARD-CENSUS-2.** Every change is a deletion, a mask or a standing law enforced, each with a pin; the protected set,
  the serving path (CARD-PERF-2), the engine's legal set (ORIGIN-1) and the model (CARD-CENSUS-2) are out of it. From
  the training-path audit (`reports/training_path_audit.md`, local, outside the tree), accepted by R381(d):
  - F5: a game ending short of the cap without a winner trains as a draw; a transient `RootExpansionFailed` turns a
    masked cap game into a trained one, uncounted;
  - F7: under tactics the tail can hand mass to a vetoed unvisited cell, or past 16 forced cells to non-blocking ones;
  - F11: the composed live loop with the production regime (Gumbel + PCR + aux) is never booted by a test tier;
  - F18: `--override-scheduler-horizon` is a no-op on a full resume;
  - F22: deploy PUCT constants come from bridge defaults, not the config;
  - the audit's §7 items 1–5: game-level fast games out, one sims key, F5 then `ply_cap_value` and `draw_reward` out,
    the dead learner knobs out, screen/confirm out;
  - F2 out: the row-wise mixing's code leaves the tree (R381(a));
  - the CUDA tests run serial (four xdist workers OOM the desktop's 8 GB card).
- **CARD-CENSUS-2 — CLOSED by R382(a): HALTED 2026-10-01 by its own known-bad, a success of LAW-19. Raw held-out
  value CE read calibration, not skill; its levers move to CARD-CENSUS-3 on CARD-VALUE-INSTRUMENT. Was ORDERED by
  R381(d), on the desktop before the mint, in parallel with CARD-HYGIENE-1.** It reads
  the value head's width and activation, with re-initialisation and warm-up (CARD-VALUE-HEAD-DEAD-OPENING; audit F3),
  and weight decay (audit F24: decoupled AdamW 1e-4 at lr 1e-3 is ≈ 1e-7 shrink per step). run11 mints both as it
  reads them (R381(e)).
- **CARD-EMA-SHADOW — CARDED for CARD-RUN11-DESIGN, not built, by R381(d): EMA as a shadow copy, read paired at
  saves.** R376(g) left EMA neither shown nor excluded; the audit's §8 reads it as the steadier of every reading.
  - ANNOTATED by R383(f) (FORGE-2 §5.13): arming `train.ema` is NOT a shadow. The EMA becomes the gate candidate,
    the anchor, the follower, the witness and the next warm start.
  - `train/ema.py`'s module docstring contradicts `actor_state_dict` on this.
  - A shadow needs its own wiring; offline averaging is CARD-SAVE-AVERAGING-OFFLINE.
- **CARD-TACTICS-LABELS-ONLY — CARDED by R381(c) as run11's second fallback, not built; FORGE-2's segment rule is its
  "relabel z" text by R383(f).** The audit's label-only
  design: proven rows relabel z, decided tails are played on the quick arm at value-only weight, and the tree decides
  no leaf; root vetoes stay the one in-search use. The first fallback is plain self-play under the tactics deploy
  (R376(d) as annotated).
  - **The relabel is FORGE-2's segment rule** (§2 item 5, Lc0's TB-rescoring rule), replacing "proven rows relabel z":
    - each row takes the verdict of the next proven turn-start row j ≥ i, its sign flipped only at turn handovers;
    - rows with no later proof keep z;
    - it is a sample of "self-play until the proof, perfect play after", sound as a sample, not exact before the
      proof.
  - **The old rule** changed 0.91 % of rows against the segment rule's 19 %, and it left "A wins" beside "B wins"
    inside one segment.
  - **Where it lands:** in `finalize_graph_outcome`, behind a cap guard (an anchor only if p + 4·turns + 1 ≤ 255),
    sharing one proof producer with RING-V3's override.
  - **Expected effect:** 0 rows under arm A, ≈ 19 % under plain self-play.
  - **Census, if the fallback is taken:** a per-ply sign-flip known-bad, and held-out arm-A games read on unproven
    rows by band.
- **CARD-RING-MIGRATION — LANDED for the root value field by CARD-RING-V3 (EXITED 2026-10-01): HEXG v3 reads v2 rings
  default-filled, so the sha-pinned held-out ring and the resume bundles' rings still load. Was ORDERED for the root
  value field as CARD-RING-V3 by R382(d). Was CARDED by R381(d), with CARD-SEARCH-VALUE-AUX, not built: a v2→v3 ring migration that
  default-fills a new field.** Today a new per-row field bumps `HEXG_VERSION` with no migration path, so every v2 ring
  (the sha-pinned held-out ring and the resume bundles' rings among them) stops reading (audit §6).

## Opened by R380 (THE FEED THAT PASSED; 2026-09-30) — F1 retracted, F2 under test

- **CARD-TACTICS-SELFPLAY-3 — CLOSED 2026-10-01 by R381(a) and (g): NOT PASS with no halting row fired; F2 is killed
  by screen and the feed of record is arm A's; closed at 6.75 box-h. Was EXITED 2026-10-01 NOT PASS; the architect rules next. Was IN PROGRESS: the
  TACTICS-SELFPLAY-3 packet (2026-09-30), worktree
  `.wt/tactics-selfplay-3` on branch `tactics-selfplay-3` (cut from `tactics-selfplay-2`'s tip `aa0c5304`), box T ≤ 7
  box-h. ORDERED by R380(d): one arm, A″ = arm A's feed with the row-wise mixing (F2), 12k steps, same parent
  (run8@45k) and seed, against TACTICS-SELFPLAY's recorded A and B saves.**
  - Its code starts from `tactics-selfplay-2`'s tip, lands F1's retraction (R380(b)) and merges as one branch (R380(f)).
  - The witnesses, in order:
    - the exams and ring bands at every save: a floor miss halts and kills F2. T4's "all" floors halt; the per-length
      rows and every ceiling report (R380(e));
    - T4 prior at 12k beside A@12k, F2's pre-registered direction;
    - R379(c)'s pass line on the shipped head against B@12k (0.477).
  - Throughput screens at 0.9× arm A's positions/h over the same steps (R380(e)).
  - If F2 dies, run11 takes arm A's feed as measured.
  - The box stays through it, within 7 box-h (R380(g)).
  - The exit (records `mantis-records/tactics-selfplay-3/EXIT.md`, local):
    - L1: F1 is retracted in code, and with F2 off the golden of feed rows is byte-equal to origin/dev's.
      `vetoed_all_rows` joins the move rows: every all-vetoed root, whatever its row holds. `emptied_target_rows`
      equals it while each such row records no policy.
    - The run lived. At 3k, 6k, 9k and 12k no halting row fired and every ring band was inside. Throughput read 0.979×
      A's positions/h over the same steps. The all-vetoed rows read 19.1 per 1 000 positions against A's 18.4, none
      given a policy. No proven ring row held under 0.5 on its proof.
    - The pass line missed: A″@12k − B@12k on the shipped head is +0.174 [−0.158, +0.505] logit (A″ 0.521, B 0.477).
    - Report-only:
      - A″ − A on the shipped head, −0.48 [−0.80, −0.16]: F2's screen reads negative, and decides nothing;
      - the parent line, −0.21 [−0.53, +0.11]: A″@12k is not offered, and run8@45k stands;
      - the T4 prior at 12k, 0.246 against A's 0.227.
    - 6.75 box-h of 7. The box has been idle since 2026-09-30 17:24Z, mirrored.

## Opened by R379 (THE SECOND TWIN; 2026-09-30) — the feed, the second twin, CENSUS-1 accepted

- **CARD-TACTICS-SELFPLAY-2 — CLOSED 2026-09-30 by the TACTICS-SELFPLAY-3 packet: the lane continues as
  CARD-TACTICS-SELFPLAY-3. Was RULED by R380(a): the halt stands; no reading of A′ is a strength reading and none of its
  saves is a parent candidate; F1 is retracted (R380(b)) and the lane continues as CARD-TACTICS-SELFPLAY-3. HALTED
  2026-09-30 at arm A′'s 3k save. Was IN PROGRESS: the
  TACTICS-SELFPLAY-2 packet (2026-09-30), worktree `.wt/tactics-selfplay-2`, box T ≤ 7 box-h. ORDERED by R379(c): one
  arm with R379(b)'s feed against TACTICS-SELFPLAY's recorded arms.**
  - The halt: T4 floor misses at 3k (overall P 0.127, V 0.133 against A's 0.249 / 0.528 and B's 0.239 / 0.565; 2-turn
    V 0.022) and the near-terminal band outside (0.068 < 0.119); the shutdown save at 3598 misses too. The defence exam
    held. No cell ran, so PASS-TO-RUN11 was not read. 1.54 box-h.
  - The feed reached the ring: no all-vetoed row went without a policy, and the weak-proof share read 0.0. Throughput
    read 0.876× A's positions/h over the same steps.
  - Read after the halt (report-only): A′'s proven endgames are about 4× shorter (root proofs 11.5 per 1k positions
    against A's 41.2; ring PROOF turn starts 282 against 1 191). The leading suspect is F1's move rule, which replaces
    the best hold with the re-search's winner. F1's move rule, its target and F2 are not separated by this data.
  - The build's reading of "every searched move is vetoed" (it cannot fire literally under Gumbel) is in
    `docs/design/TACTICS_DESIGN_2026-09-28.md` §6; the exit record is local
    (`mantis-records/tactics-selfplay-2/EXIT.md`). F1 builds the re-search (row `tactics_research_count`), F2 the row-wise mixing
  (row `tactics_mixed_rows`), and T runs arm A′ from run8@45k. Its pass line is the deploy-matched reading with the lower bound above zero. run11 arms tactics in self-play
  only on that pass, since R376(d) forbids a plain self-play under a tactics deploy.
  - The feed (R379(b)), at deploy and in self-play alike:
    - A root whose every searched move is vetoed is re-searched over the non-vetoed set at the same budget. That
      search's improved policy is the row's target, and its winner is the played move. This replaces the all-vetoed
      root's empty row (TACTICS-SELFPLAY's `emptied_target_rows`).
    - A proven root's target mixes the proof at α = 0.5 only where the searched mass on it reads below 0.5, row by
      row. This replaces R378(e)'s median rule.
  - The readings (R379(a)): the T4 and defence exams are the starvation instrument; the deploy-matched head is the
    strength reading for a tactics-era net (LAW-15, R378(b)); the net alone is a report-only diagnostic pooled over at
    least two saves.
  - The recorded arms are TACTICS-SELFPLAY's A and B (`mantis-records/tactics-selfplay/`, local; CARD-TACTICS-SELFPLAY).
- **CARD-PERF-2 — RATIFIED by R384(b), L1's miss and the operator's word recorded. LANDED 2026-10-03 by the PERF-2 packet (L1 `bef8f04f`, L2 `7ad5208a`, L3 `355c7dc9`, L3b `e6b9e0e8`; L4 not landed): the production self-play loop on the 4080S box reads 235 455 positions/h against 164 422 (+43 %, IQRs [230 895, 243 870] and [158 910, 169 215]; games/h 2 580 against 1 807 at 91.3 and 91.0 plies per game). Was: runs NOW, in parallel, by R383(b): throughput (independent games per step) is a value lever. Was: runs after CARD-HYGIENE-1, on the desktop before the mint (R381(d)). Its order RULED by R379(e), following CENSUS-1's C2: the copies first, then the edge table, then
  CUDA graphs.** R378(g) sequenced PERF-2 after CENSUS-1, which R379(d) accepts. C2 read the H2D copies as the largest
  part of the server's `launch` (33–38 %) and the compiled trunk's launch as nearly fixed per forward (≈ 3.5 ms)
  (`docs/design/measurements/CENSUS1_2026-09-29.md` §C2).
  - The copies: the levers' C1, one pinned buffer (`LEVERS_RESEARCH_2026-09-28.md`, local).
  - The edge table: CARD-PERF-EDGE-TABLE.
  - CUDA graphs: CARD-PERF-GRAPHS.
  - The bench (box, B 64, 5 windows, median [IQR] leaves/s): base 3 931 [3 643, 4 405]; L1 4 253 [4 176, 4 560]
    (+8.2 %, inside the IQR); L2 4 629 [4 587, 4 629] (+8.8 % over L1, beyond it); L3 5 130 [4 949, 5 691] (+10.8 %
    over L2, beyond it); L3b 5 029 [4 841, 5 834] (-2.0 %, inside). The server's launch per pop reads 11.94, 9.84,
    11.05 and 6.72 ms; its device wait 7.74, 7.84, 3.81 and 5.35 ms.
  - L1 missed its pre-registered +10 % line. The line was drawn from CENSUS-1's desktop split (copies 7.60 ms), and
    the box's base copies take 3.67 ms. L1 landed under L2 and L3 by the operator's word (2026-10-03). Its 4-thread
    pack reads 4 253 against 4 149 at one thread, inside the IQR.
  - The pins: L1 and L2 are byte-equal on the golden collate and on the 24 seeded deploy searches. L2's coded forward
    is bit-identical to the per-edge forward (sm_86 on 11 batches; sm_89 on the bench's 64 probe values). L3 is within
    LAW-06 (ii-b) (pooled mean |dlogit| 0.0452 eager and 0.0434 compiled against the 0.0511 line) with no argmax or
    value-sign flip, and the 24 deploy searches' moves are equal with root values within 0.0101.
  - Memory: the served path reserves 2.6 GB with buckets against 13.6 GB eager. With the trainer in the process the
    card peaks at 15.2 of 15.6 GiB reserved (4.96 GiB live), and the allocator retried at the ceiling 3-4 times in 8
    minutes. A capture cannot release cached blocks, so L3b releases them before each capture: one device wait per
    bucket per server.
  - L4 (a highest-priority serving stream, an event around the weight copy) read 9.86 against 9.96 ms serving latency
    at an equal 1.20 trainer steps/s: -1 % against a 10 % line, not landed (CARD-PERF-TRAINER-CONTENTION).
- **CARD-JK-LAST — DEAD for run11 by R381(d). Was CARDED for throughput by R379(d).** CENSUS-1's C5 read the JK-last shape at Δ −0.052 nats against
  4×256's −0.053, at a quarter of the parameters (229k against 996k); no shape passed the −0.07 line. It is carded as a
  throughput lever, not as growth.
- **CARD-CENSUS-1-PARKED — PARKED by R379(d).** Each needs a ruling to reopen:
  - growth: CARD-NET-EXPAND is HELD (C5: no shape wins on the frozen 45k ring);
  - the regret and restart family (C4);
  - the next-ply aux head (C5: it fails its line);
  - learning from Six's positions (R376(f) still governs any learning from Six's outputs).
  - Beside them, not parked: MCGS earns a deploy A/B later (CARD-MCGS-DEPLOY).
- **CARD-VALUE-HEAD-DEAD-OPENING — a capacity defect with no shown cost (R382(c)); the head's shape and activation
  are CARD-CENSUS-3's third lever. Was read by CARD-CENSUS-2 (R381(d)), run11 mints the head as it reads. Was CARDED for CARD-RUN11-DESIGN by R379(d): a value-head re-initialisation at warm
  start, with warm-up.** CENSUS-1's C4 found run8@45k's value head dead in the opening: 18.5 % of the 45k ring's value
  rows read the empty board's v = −0.015556, and 0 of `value_head.fc1`'s 32 ReLUs are live on opening rows. run10's
  warm start inherits the head. C5 read the value gap at +0.160.
- **CARD-QUICK-ARM-NOISE — CARDED, not built, by R381(d). Was CARDED for CARD-RUN11-DESIGN by R379(d): a noise-free quick arm.** CENSUS-1's C1: the
  64-sim quick search builds `MctxRootState::new` and draws root Gumbel noise whatever the arm drew, and from ply
  `gumbel_explore_moves` on the played move is `best_action`, whose score carries that noise.
- **CARD-SEARCH-VALUE-AUX — the 64-sim W/N target read NULL at CENSUS-3's regime by R383(a) (it closes the gap but
  lifts no held-out skill; W/N is pessimistic, −0.105 against z's +0.010); run11 trains at search weight 0 by R383(e).
  Was CARD-CENSUS-3's first lever by R382(c)-(d): z mixed with the search's root value, on
  CARD-RING-V3, whose exit lands the lever as `train.value_target_lambda` minted 0.0 (rings written before it carry
  no root value, so the lever reads on rings self-play writes after it). Was CARDED, not built, by R381(d), with a v2→v3 ring migration. Was CARDED for CARD-RUN11-DESIGN by R379(d): a search-value aux target from our own search.**
  CENSUS-1's C6: Six's search value beats our raw one off the proofs (DEF −0.097 on its non-proof rows, QU −0.058,
  MID −0.053). R376(f) governs any learning from Six's outputs; this card is our own search's value.
- **CARD-DECIDED-TAILS — PARKED by R381(d). Was CARDED for CARD-RUN11-DESIGN by R379(d): decided tails.** CENSUS-1's C3(b): the pooled
  proven tail is 0.463 of a game's plies, a games/h bound of 1.86×; the proven side converted 76.8 % of those games.
- **CARD-MCGS-DEPLOY — CARDED by R379(d): graph search earns a deploy A/B, later.** CENSUS-1's C3(d): table hits are
  28.6 % [27.9, 29.4] of descents at the plain deploy head's turn starts, above the 10 % line.

## Opened by the TACTICS-SELFPLAY packet (2026-09-29)

- **CARD-TACTICS-TARGET-FEED — CLOSED 2026-10-01: F1 (the re-search) retracted by R380(b), F2 (the row-wise mixing)
  killed by screen by R381(a); the feed of record is arm A's, no re-search and no mixing, and F2's code leaves the
  tree (CARD-HYGIENE-1). Was F2 ONLY since R380: the row-wise mixing of the proof, under test in
  CARD-TACTICS-SELFPLAY-3. Option (c) RETRACTED by R380(b): an all-vetoed root records no policy target and plays
  the audit's best hold, as arm A did; the row-wise mixing is under test in CARD-TACTICS-SELFPLAY-3 (R380(d)). Was
  RULED by R379(b), and the treatment of TACTICS-SELFPLAY-2 (CARD-TACTICS-SELFPLAY-2):
  the card's option (c) for the all-vetoed root, at deploy and in self-play alike, and the proven-root mixture decided
  row by row, replacing R378(e)'s median rule; TACTICS-SELFPLAY-2 builds them as F1 (the re-search) and F2 (the
  row-wise mixing). Was CARDED for the architect: what the tactics module teaches the policy in self-play.**
  The twin's arm A played the module's moves, but its rows teach them weakly.
  - Defence: when the audit vetoes every move the search visited, the row records no policy target (8 366 rows in
    arm A's window, 18 per 1 000 positions). The hold the game played teaches nothing, in exactly the positions where
    the net's policy is wrong. When the veto takes part of the mass (1 473 rows), the renormalised target carries the
    defence.
  - Offence: a proven root trains the searched target (R378(e)). On arm A's last ring, 35.5 % of those rows put under
    0.5 of their mass on the proof (P1 read 35 % on run8), because P1's rule read the median of a bimodal
    distribution. The α = 0.5 mixture would put at least 0.5 on the proof.
  - A hold is not a proven defence (the audit failed to refute it within budget; TACTICS-DESIGN's T3 found 16 of 20
    such moves lost when played out), so a bare one-hot on it would often teach a losing move.
  - Options, cheapest first:
    - (a) the net's prior with the vetoed cells removed;
    - (b) α · one-hot(hold) + (1 − α) · (a);
    - (c) a second search restricted to the non-vetoed moves.
  - Any of them changes the target law of R377(f) and R378(e), so it is a ruling and a new twin's treatment.
- **CARD-THROUGHPUT-IN-POSITIONS — CARDED (R379 does not rule it): a throughput rule reads positions produced per
  hour, not games per hour.**
  games/h reads game length as cost: in the twin, arm A's games ran 92.6 positions to arm B's 71.5. The rule read
  positions/h by the operator's word before arm B's data was read (RULES_T A3). A/B read 1.058 in positions and
  0.816 in games, so the choice did not decide the verdict. The per-game trainer budget (≈ 2.38 steps per game) means
  the arm with longer games trains on fresher data per step (replay ratio 6.60 vs 8.49); it is part of the treatment.
  PERF-2 read both units (+43 % positions/h and +43 % games/h at 91.0 and 91.3 plies per game): a serving change
  leaves game length alone, so the units agree there, and the rule decides only where a treatment moves game length.

## Opened by R378 (TACTICS-DEPLOY ratified; 2026-09-29) — tactics at deploy

- **CARD-SIMS-ACCOUNTING — CLOSED by TACTICS-SELFPLAY's P0 (`faa27c48`, branch `tactics-selfplay`); was OPEN
  (R378(c)): one simulation count for every head.** P0 reproduced the early end (the plain PUCT head stopped at 43 of
  256 descents at a revisited win) and removed it: a table descent counts toward n, and a terminal on the table path
  counts as an inline descent. All three loops (runner, deploy PUCT, deploy Gumbel) spend exactly N. The witness pins
  root visits == counted descents; its planted break reds. A descent that backs up a value is a simulation, whatever backed it: the net, the table or the solver.
  - The plain head's early end is a defect: TACTICS-DEPLOY's L5 read the plain deploy head ending 12 % of its searches
    short (CARD-TT-HIT-STARVATION, which this card fixes).
  - P0 reproduces it first, then removes it, with a planted-break test that reds.
  - The served-sims witness pins all three cases: a net leaf, a table hit and a solver terminal. The packet grants
    the witness its table-hit and solver-terminal cases (R376(e), R378(c)).
  - The A/B stands: TACTICS-DEPLOY's plain arm is not re-read.
- **CARD-GATE-DEPLOY-MATCHED — LANDED on branch `tactics-selfplay` (`a44c9b28`); OPENED by R378(d): the in-run gate
  is deploy-matched (LAW-15), so candidate and anchor play the same block.** The eval worker's `_pair` arms the best
  side with the round's block beside the candidate
  (`tests/eval/test_game_record_eval_channel.py::test_an_armed_round_arms_both_gate_sides_and_every_record_carries_the_candidates_rows`,
  `docs/contracts/game_record.md`); a ruler cell still arms our head alone. Before it, the tree armed the candidate
  alone. This was TACTICS-DEPLOY's exit item (3), from L4's review. It is owed before a run
  whose `deploy.search.tactics` is non-null plays a gate round. That means run11, and TACTICS-SELFPLAY's arm A if it
  plays one (run10's header sets `terminal_eval_enabled: true`). The ruler cells'
  candidate-only arm is not this seam (`docs/contracts/eval_instrument.md` v6).
- **CARD-CENSUS-1 — ACCEPTED by R379(d): its record is `docs/design/measurements/CENSUS1_2026-09-29.md`. Its parked
  items are CARD-CENSUS-1-PARKED, JK-last is CARD-JK-LAST, six items are carried to CARD-RUN11-DESIGN, and its C2 orders
  CARD-PERF-2 (R379(e)). Was ORDERED by R378(g): CENSUS-1 runs beside the TACTICS-SELFPLAY twin, and PERF-2 follows it.**
  Its three commits (`c4418ffd`, `7f18b478`, `986a9661`) ride branch `tactics-selfplay`, and it lent the desktop GPU
  for that branch's gates.exit. It
  is its own packet and session. Its background is LEVERS_RESEARCH (`mantis-records/research/LEVERS_RESEARCH_2026-09-28.md`,
  local), accepted with its §0 corrections, except that the in-run cache hit rate is measured: 24.2 % (the F5
  reading under the RUN10-PRESTART section).

## Opened by R377 (TACTICS-DESIGN accepted; 2026-09-28) — the tactics module

- **CARD-TACTICS-DEPLOY — RATIFIED by R378(a) and MERGED: `dev` is the branch's tip `0204edf9`. The read's block is
  the deploy block of record, and run11 arms it at its mint by the operator's word, in self-play and at deploy alike.
  EXITED 2026-09-29: the lever PASSES on both rulers and the audit LANDS.**
  - Lever X, full − plain: +0.957 logit [0.752, 1.162]; win rate ~0.33 → ~0.56 at 39k/45k/51k.
  - Lever S: +0.973 [0.663, 1.284].
  - Audit X, full − audit-off: +0.254 [0.049, 0.459].
  - The known-bad (the audit inverted) reads 0.052 against 0.351.
  - No game a found proof played was lost.
  - The wiring alone (no solver, no audit) reads +0.027 over plain, so the solvers and the audit carry the lift.
  - The read's block: leaf 256/3, root 20 000/8, audit 2 000/8, k 4, m 4, total 40 000.
  - The bench: no abort; the deploy wall 0.864x plain; self-play 1.08x descents/s.
  - The branch `tactics-deploy` was fast-forwarded into `dev` (this line said "`dev` is untouched" until R378's
    record). Every config mints `tactics: null`. Records: `mantis-records/tactics-deploy/` (EXIT.md,
    drivers/L5_BENCH.md, drivers/L6_READINGS.md).
  - R378 answers the exit's open items: the block by (a), Q5 by (c) (CARD-SIMS-ACCOUNTING), the gate by (d)
    (CARD-GATE-DEPLOY-MATCHED) and the build hash by (h). From the merge, the strength series plays the shipped head,
    and a read of the net alone plays `--arm plain` and says so (R378(b)).

  As ORDERED by R377(e), the lane's next packet: The packet reads X's floor first (one random-init net, 288 games), builds and benches the module,
  wires it at deploy with `search.tactics` null in both homes (the self-play path unchanged), and reads plain,
  full and audit-off at 39k, 45k and 51k on X then S, with the inverted audit on 45k as its known-bad, on the box
  within a grant of ≤ 12 box-h. It builds the module to `docs/design/TACTICS_DESIGN_2026-09-28.md` §4–§10, as
  R377 rules it (the doc's §11 and §12 are amended in place per R377):
  - Six's kind, a turn-level strictly forcing threat-space search, is the core. Our per-stone `TacticalSolver`
    is retired when the module lands (R377(a)).
  - v1 proves wins and can't-cover losses, Six's set (R377(b)); deeper loss proofs are CARD-TACTICS-DEEP-LOSS.
  - A descent that ends at a proven terminal is a simulation, and the served-sims witness counts it (R377(c)).
  - `apply_quiescence`'s override and its blend stay in v1 behind the shared analysis function (R377(d)).
  - The A/B reads the full module AND an audit-off arm against the plain parent on both rulers, X's floor first.
    The defence audit lands only if its arm earns its cost (R377(e)). The design prices the bundle at ≈ 5.5
    box-h and the audit-off arm at ≈ +3.2 (§11, §12 Q2).
  - §12 Q5 (TT-hit expansions stay uncounted, the design's named deviation from R376(e)) is not among R377's
    items. RULED by R378(c): a TT hit is a counted descent (CARD-SIMS-ACCOUNTING).
- **CARD-TACTICS-SELFPLAY — RULED by R379(a): the screen stands as written and the lever is not adopted from it; the
  lane continues as CARD-TACTICS-SELFPLAY-2 (R379(c)). EXITED 2026-09-29: NOT PASS-TO-RUN11. The pre-stated 12k screen
  fails, while every band held and throughput passed. Was IN PROGRESS: the TACTICS-SELFPLAY packet (2026-09-29),
  forwarded by R378; was HELD on CARD-TACTICS-DEPLOY's read.**
  - The screen: the net alone at 12k, A − B = −0.209 logit [−0.569, +0.151], against a bar of > −0.17 (X, 288
    games; A 0.222, B 0.260).
  - Report-only: the net alone at 9k +0.229 [−0.137, +0.596] and at 6k −0.795 [−1.240, −0.351]. Pooled over 9k and
    12k it reads +0.010 [−0.247, +0.267].
  - The shipped head at 12k: +0.676 [+0.342, +1.010] (A 0.642, B 0.477). What ships separates; the net alone does not.
  - The bands at every save of both arms: no floor miss, no ring band outside, no halt. The ceiling misses were value
    misses at 3k (A 1, B 3), from the shared parent.
  - Throughput by positions/h (RULES_T A3, the operator's word before arm B's data was read): A/B 1.058. games/h
    reads 0.816, because A's games run 30 % longer.
  - A's policy is fed weakly (CARD-TACTICS-TARGET-FEED). A1's re-read of P1 on arm A's ring: median 0.928, with
    35.5 % below 0.5.
  - The code is on branch `tactics-selfplay` (tip `5db3280b`, gates.exit green), which `dev` contains (this line did
    not say so until R379's record). Its records, with every bias the
    reading carries, are local in `mantis-records/tactics-selfplay/` (EXIT.md, EXIT_DRAFT.md).
  - Box: 11.86 box-h of jobs. A twin with the same module in the loop, against a plain twin. The design is
  `docs/design/TACTICS_DESIGN_2026-09-28.md` as R377 and R378 amend it. The session works in worktree
  `.wt/tactics-selfplay`, and its records go to `mantis-records/tactics-selfplay/`.
  - The targets (R377(f), amended in form by R378(e)): a proven root plays its proof, a vetoed move gets zero target
    mass, and a lost root records no policy target. The target at a proven root is the searched target where its
    proof-set mass reads ≥ 0.7 at the median, else the α = 0.5 mixture; never the bare two-hot.
  - Desktop legs, before the twin:
    - P0: CARD-SIMS-ACCOUNTING.
    - P1 (A9, F-53 with the new kind): over the proven roots in run8's 45k ring, the searched target's mass on the
      proof's cells picks the target form. It rebuilds PROBE-1's reading 4, which left with the per-stone solver
      (`bd7bcde9`).
    - P2 (A10): the defence exam's band.
    - P3 (A11): the ring's baselines, which set the ring-composition bands.
  - Box legs: P4 reads the in-run cache at sync cadence 2, 50 and 200 beside F5's 24.2 %. Then T: arm A (tactics in
    self-play and at deploy, at the block of record) and arm B (plain), 4 h each and one at a time, from run8@45k
    with the same seed.
  - The witness at every save (R377(f), R378(f)):
    - T4 by proof length, against a floor (mean − 3 SD) and an enrichment ceiling (mean + 3 SD), on the frozen exam
      (sha256 `2b2eb7f5…`). The panel's mean − 3 SD is mean prior 0.154 and mean value 0.255.
    - The defence exam, against P2's band; the ring-composition bands.
    - The throughput and tactics rows.
  - Pre-stated: PASS-TO-RUN11 iff every band holds at every save, arm A's throughput is ≥ 0.8× arm B's, and arm A's
    net alone is not below arm B's by more than 0.17 logit at 12k. A T4 or defence floor miss at any save is the
    starvation signature, and it halts arm A.
  - Grants: the self-play path accepts `selfplay.search.tactics`, the witness gains its two cases, and a census-exempt
    twin config is minted through run10's header; the box, ≤ 12 box-h for P4 and T. No production config, no START
    and no priced act.
- **CARD-ORIGIN-RULE — SPLIT by R384(f): the deploy head plays the origin on an empty board now (RUN11-PRE's L1d);
  the engine rule and canonicalisation land after run11's start by pinned resume. OPENED for the RUN11-PREP packet, whose leg ORIGIN-1
  lands it (R377(g)); it gains by R383(f): the forced ply-0 row does not train (FORGE-2 §5.8, as in Six).** The first stone
  is the origin, as in the
  official rule and Six: the empty board's legal set is {origin}. Records and books are canonicalised by
  translation on load, and the frozen fixtures are re-pinned under grant. It touches the protected set's arena
  legality, which R377(g) names. That naming stands in R377's Status line, and the LAWS bullet moves when ORIGIN-1
  lands (R378(h)).
- **CARD-TACTICS-DEEP-LOSS — CARDED, a later lever (R377(b)).** Loss proofs beyond can't-cover, e.g. the design's
  §12 Q4: a cover-2 position where every covering reply loses to a strict opponent win.
- **CARD-QUIESCENCE-BLEND-RETIRE — CARDED, its own leg (R377(d)).** Retire `apply_quiescence`'s heuristic blend
  (−0.3 for two fives against a two-stone turn), which is not exact.
- **CARD-NATURAL-BOOK — DEFERRED by R381(d). Was OPENED for the RUN11-PREP packet (R377(h)): a policy-drawn natural book becomes a second
  reading.** The ruler's book (`book_v1_s20260625_p4`) stays for the series.

## Opened by the TACTICS-DEPLOY packet (2026-09-28)

- **CARD-ZOBRIST-REFLECTION — CARDED: core's Zobrist keys a cell and its reflection through the origin alike, in
  1 of 8 cells beyond ±9.** `ZobristTable::get_for_pos` seeds an out-of-table cell with `q*M1 ^ r*M2 ^ p*M3`, and
  negating a number flips every bit above its lowest set one, so `(q, r)` and `(-q, -r)` collide whenever
  `tz(r) == tz(q) + 1` (2 680 pairs within ±64; L1's review). `Board::zobrist_hash` keys the MCTS transposition
  table with it. The tactics solver no longer does: it keys stones by its own injective `grid::stone_key`, after a
  table keyed by core's carried a proven win to a reflected position. The eval cache's key (PERF-3's L2) keys stones by
  its own injective word too, after its review met the same collision there. Fixing core re-mints the pinned values of
  `crates/mantis-core/tests/golden_replay.rs`.
- **CARD-TACTICS-BUILD-HASH — OPENED by R378(h); was CARDED: the engine does not carry the hash of the sources it
  was built from.** A
  receipt names the tree's tactics sources (`module_sha256`) and the engine's bytes (`engine_sha256`), read before a
  cell plays, but nothing ties the two: a build-time hash embedded in `mantis._engine` (a `build.rs`), compared at
  cell start with a refusal on mismatch, would. Found by TACTICS-DEPLOY's L4 review.
- **CARD-TACTICS-PLY-HORIZON — CARDED: a proof's turns are not capped by the plies a game has left.** A leaf or root
  proof whose six lands after the game's ply cap is not a win in that game. Deploy games rarely reach the cap;
  self-play's `max_moves_per_game` makes it a TACTICS-SELFPLAY question.
- **CARD-TT-HIT-STARVATION — FIXED by TACTICS-SELFPLAY's P0 (`faa27c48`). RULED by R378(c): a TT hit is a counted
  descent, and the plain head's early end is a defect, fixed by CARD-SIMS-ACCOUNTING. Was CARDED: a PUCT search can end short of its budget, tactics on or
  off.** A TT-hit
  expansion is an uncounted descent (the design's §12 Q5, unruled), and when every attempt of a `select_leaves`
  call (4n) is one, the call returns nothing and the budget loops (`run_mcts_search`, `_drive_puct`) stop. Found by
  TACTICS-DEPLOY's L2 review: a compact producer under-spent 8 of 8 tactics-off trials, and the witness's tactics-on
  case read short in 2 of 8 runs. It is now COUNTED, not fixed: the runner's `starved_searches` / `starved_descents`
  rows (Rust snapshot; not yet bridged or in the event manifest) and the deploy head's `last_sims`, and the
  witness's tactics-on cases assert served + inline + starved == N a search. Ruling Q5 by R376(e)'s letter (a TT hit
  is a counted descent) removes it; a loop that simply continued past a starved call would break `MAX_ARMED_SIMS`,
  which is derived from the 4n attempt cap.
  - MEASURED at deploy (TACTICS-DEPLOY L5, 300 positions of the X 45k cell): the PLAIN head ends 12 % of its searches
    short, by 139 descents on average and down to 2, a mean of 239 of 256. Any armed block, even the wiring alone,
    does not, because its decided leaves are inline descents. On X the wiring alone reads +0.027 logit over plain,
    so the shortfall costs no measurable strength there.

## Opened by R376 (DECIDE-1 accepted; 2026-09-28) — the tactics lane

- **CARD-TACTICS-LANE — ORDERED (R376(d)); it goes first, and run11 is designed after its deploy read
  (R376(c)). RE-AIMED 2026-09-28 to the TACTICS-DESIGN packet. It measures before designing: a census of
  both solvers, the two run head to head on the same positions, D1's specificity check and a baselined
  starvation witness. Its exit commits `TACTICS_DESIGN_2026-09-28.md` in docs/design/ and scopes two packets:
  TACTICS-DEPLOY (the module and its A/B on S and X) and TACTICS-SELFPLAY (a twin read by the witness).
  TACTICS-DESIGN EXITED 2026-09-28 and is ACCEPTED by R377(a): Six's kind is the core, and the lane continues
  as CARD-TACTICS-DEPLOY, then CARD-TACTICS-SELFPLAY. TACTICS-DEPLOY is RATIFIED by R378(a), and the lane is now
  carried by the TACTICS-SELFPLAY packet (2026-09-29, in progress).**
  ONE exact tactics module in Rust on the search path, used identically at deploy and in self-play
  (LAW-15). The search crate carried a net-free per-stone `TacticalSolver` (`crates/mantis-search/src/tactics/`,
  the solver F-53 read); one implementation per thing, so the lane's module replaces it (done in TACTICS-DEPLOY:
  the old solver lived through `bd7bcde9`).
  - Deploy lands first, read as an A/B against the plain parent on both rulers, at equal work, with its own
    known-bad and a power line (LAW-19). Counting (R376(e)): a solver terminal is a simulation, GPU
    evaluations are their own LAW-18 row, and served-sims exactness pins descents.
  - Self-play follows only with a pre-registered starvation witness, and with F-15, F-39, F-53 and R239
    re-validated under LAW-02 first.
  - Evidence (DECIDE-1): in 106 of S1's 233 losses, f 0.455 [0.392, 0.519], one of our turns allowed the
    opponent a strict ≤ 8-turn forced win where a safe turn existed (Six's `ThreatSolver`, 20 000 nodes); 100
    of them at our last turn before that run. f is an upper bound against false safes only. Switching Six's
    threat solver off moves the parent 0.351 → 0.594 against gen 30 and 0.115 → 0.286 against gen 455
    (≈ 1 logit at both).
  - Owed before f is taken at face value: a specificity check of D1's "safe" alternatives (a sample re-checked
    at a deeper budget, or played out). DISCHARGED by TACTICS-DESIGN's T3: 39 of 40 alternatives stay safe at
    12 turns / 60 000 nodes (f_corr 0.455 [0.385, 0.515]), but 16 of 20 played out with Six on both seats are
    still lost within 13 opponent turns.
- **CARD-RUN11-DESIGN — HELD on CARD-TACTICS-LANE's deploy read (R376(c)), now CARD-TACTICS-DEPLOY's (R377); that
  read is RATIFIED by R378(a). Its self-play tactics are arm A's design, CARRIED, not adopted (R381(c)); its mint rows
  are R381(e)'s, after CARD-HYGIENE-1, CARD-CENSUS-2 and CARD-PERF-2 (R381(d)); its value rows are CARD-CENSUS-3's
  picks, and nothing mints before CENSUS-3 reads and CARD-RING-V3 lands (R382(d)). R383: the parent's value head, wd
  1e-4 and search weight 0 (R383(a), (e)); `min_buf_size` withdrawn for the ring-fill warm start (R383(c)); the value
  regulariser and the step rate are CARD-REG-1's and CARD-RUN11-PRE's, the last two reads before the mint (R383(d)).
  R384: the re-drawn value mask at 1/8 and Σπ′·completedQ in the ring (R384(a)); run11 is RUN11-PRE's picked arm,
  continued from its last save (R384(d)); the production run is the default state from run11 on (R384(c)).** Until run11's
  mint replaces it,
  `configs/run10.yaml` stays the production config the instruments read; run10 will not START. The design
  carries:
  - Cooldown and EMA as screened levers with power lines (R376(g)): neither is shown nor excluded. DECIDE-1's
    cooldown screen read COOL 0.301 vs CTRL 0.266 on X over 4 origins (≈ +0.20 ± 0.30 logit), the S pair +0.009,
    and COOL − REV +0.025 (inconclusive: the lead is not attributed to the cool end); EMA5 read −0.020 against
    its members on X (void by power).
  - CARD-NET-EXPAND (run11's build, R367(c)). HELD by R379(d): growth is parked (CARD-CENSUS-1-PARKED).
  - Learning from Six's outputs only as a means (R376(f)): probes first, a run only on a pass, and a net that
    learned from Six carries it in its lineage. The standing goal is to surpass Six by self-play with exact
    tactics.
  - A random-opening share in self-play, as an arm, not a default (R377(h)).
  - CARD-ORIGIN-RULE, which lands in RUN11-PREP (R377(g)).
  - The deploy block of record (leaf 256/3, root 20 000/8, audit 2 000/8), armed at run11's mint by the operator's
    word, in self-play and at deploy alike (R378(a)), with the gate deploy-matched (R378(d),
    CARD-GATE-DEPLOY-MATCHED). In self-play as arm A's design, CARRIED, not adopted (R381(c)): run11's exams and ring
    bands halt it, and its process level on the shipped head against the parent's anchor reads it. The fallback is
    plain self-play under the tactics deploy (R376(d) as annotated), then the audit's label-only design.
  - The mint rows (R381(e)): aux weight 2.0 (the owed re-pick, inside its own band); `draw_reward` gone, with
    reason-3 rows masked; `min_buf_size` 100k rows (the warm-start draw count ≈ 13, C5's tested figure); sync cadence
    50; the value head and weight decay as CARD-CENSUS-2 reads them (re-pointed to CARD-CENSUS-3's picks by R382(d)). Reuse stays at 2.4 steps/game, read by run11's
    held-out gap (R381(d)). A scalar re-mint inside the pre-registered envelope is a STATE line, and the envelope is a
    prereg row (R381(f)).
  - Carried from CENSUS-1 by R379(d):
    - sync cadence 50 (TACTICS-SELFPLAY's P4 read the in-run cache hit at 33 % there, against 24.5 % at 2);
    - HL-Gauss as a screened lever (C5: it passes its lines, no gain resolved at one seed; any HL-Gauss line read on
      raw held-out value CE is VOID as evidence, R382(a));
    - a value-head re-initialisation at warm start, with warm-up (C4: the parent's value head is dead in the opening;
      C5: its value gap +0.160, read on raw held-out value CE; CARD-VALUE-HEAD-DEAD-OPENING);
    - a noise-free quick arm (C1: the 64-sim quick search draws root Gumbel noise; CARD-QUICK-ARM-NOISE);
    - decided tails (C3(b): a pooled proven tail of 0.463, a games/h bound of 1.86×; CARD-DECIDED-TAILS);
    - a search-value aux target from our own search (C6: Six's search value beats our raw one off the proofs;
      CARD-SEARCH-VALUE-AUX).
- **CARD-PACKET-POWER-LINE — LANDED 2026-09-28 in the packet rule (`docs/governance/COMMS_STYLE.md` item 4): a
  packet carries each band's power line and sets known-bad bars against the ruler's measured floor. Was CARDED
  (process; DECIDE-1's exit).** Three of DECIDE-1's banded rules were void by R375(d) before any reading (B1-N2, B3, the S
  confirmation as written), and B1-N1 needed a spread between nets that the panel then showed was absent.
- **CARD-DESKTOP-TMP-QUOTA — CARDED (ops; DECIDE-1's exit).** The desktop's `/tmp` is a RAM tmpfs with a per-user
  quota shared by every session. On 2026-09-27 a finished leg's 9.6 GB scratch filled it mid-leg. Four training
  arms died with EDQUOT, a staging copy stopped short, and every shell call lost its output. The note: leg data
  (nets, arms, games) lives in the mirror (`mantis-mirror/<leg>-<date>/`) from the start, and records in
  `mantis-records/<leg>/`. A session deletes its own scratch at exit, once the records name the copies.

## Opened by R375 (DECIDE-1's forward; 2026-09-27) — read the process, not the peak

- **CARD-STRIX-SEED-LUCK — CLOSED 2026-09-28, SUPERSEDED by R376(b): not separated, and no longer needed. DECIDE-1's
  noise nets were void (the KL-1.0 known-bad read 0.12 on X against a ≤ 0.05 bar); its panel reads the parent's
  0.191 as an outlier, and every bar prices from a panel. Was OWNED by DECIDE-1 (R375(c)); opened in
  RUN10-CONTROLS' exit record.** At the fixed
  seed a perturbation of KL 0.0067 (C4, the parent at LR ×0.01 for 101 steps) re-draws 255 of 288 games and a
  neighbour re-draws all of them. On the re-drawn games S1's wins fall from 42 to 22 (C4), 53 to 22 (42k) and
  55 to 27 (48k). Training damage and a lucky parent draw predict the same thing there. DECIDE-1 separates them
  with seeded weight noise on the parent at C4's and A6's distances (KL ≈ 0.007 and ≈ 0.09, with a KL ≈ 1
  known-bad), read on both rulers; the §4 instrument keeps its seed (R373(b)).
- **CARD-SIX-RUNG — LANDED 2026-09-28 by the SIX-RUNG packet (R375(a)); RATIFIED by R376(a): X, the second ruler.
  Recipe decisions read X; milestones read X and S. X's floor (a random-init net) is unmeasured; DECIDE-1's
  KL-1.0 noise net read 0.12 on it.** Six generation 30
  at 16 nodes, tactics as shipped, search cache off (`cacheEntries 0`), plays as `mantis.bots.six` beside strix.
  The repo `f2b5ec2`, release 1.2.0's engine and ONNX Runtime and the gen 30 and gen 455 networks are pinned by
  url and sha256 in `vendor/pins.toml` (`make vendor.six`), never tracked, and re-hashed at every engine start;
  the follower's `six30_16` unit writes `<ckpt>.six30_16.json`. The witness on the box (tree `46387ad0`)
  re-read the parent 0.351 [0.295, 0.406], SIX-SCOUT's reading exactly (101–187). The cache setting moves Six's
  turns in 12 of SIX-SCOUT's 288 games (CPU provider), so the operator restated the replay bar before the cell
  ran: 256 of the 276 cache-invariant games replayed (bar 249); 258 of 288 overall (the pre-stated bar was 260).
  Each of the 30 non-replays first diverges on our stone in a cache-invariant game (20) or on Six's in a
  cache-sensitive one (10). SIX-SCOUT's scratch at `vendor/external/six` (the desktop main checkout, the box)
  sits on the pin's clone path: `make vendor` refuses there until it is moved aside. Upstream has since tagged
  1.2.1 (`ba101e6`), which changes `engine/src/threats.cpp`; R375(a) admitted 1.2.0.
- **CARD-RANDOM-OPENINGS — RULED by R377(h): the ruler's book stays for the series, a policy-drawn natural book
  is a second reading (CARD-NATURAL-BOOK), and a random-opening share in self-play is a run11 arm, not a default
  (CARD-RUN11-DESIGN). Was CARDED: self-play draws no opening plies, while both rulers' books are random
  scatter.** `configs/run10.yaml` mints `selfplay.random_opening_plies: 0`, so every self-play game starts from
  the empty board. Both rulers' cells open from `book_v1_s20260625_p4`: four uniform-random plies from the
  empty board (`tools/mint_opening_book.py`).
- **CARD-COMPLETED-Q-DEFAULT — CARDED, latent: a node with no raw value completes against a silent 0.0.**
  `node_completed_qvalues` (`crates/mantis-search/src/mcts/policy.rs`) reads the node's raw value with
  `unwrap_or(0.0)`. The PUCT kind allocates no raw-value slots, and a Gumbel slot that was never written keeps
  its 0.0 initialisation; neither case is signalled. Latent: today's callers are the Gumbel kind's
  (`gumbel_mctx.rs`, `pick_best_mctx_interior`).

## Opened by the RUN10-AUX-PROBE packet (R373; 2026-09-27) — no suspect is needed: every net trained from the parent reads below it

- **CARD-RUN10-START-DAMAGE — the reading the design packet starts from (R373(c)). R376(b): the parent's 0.191 is
  an outlier (DECIDE-1's panel of run8's saves); run10 will not START (R376(c)).** Offline, on the launch tree,
  101 production steps from the parent on its own ring (minus the prereg's slice) read at strix @ r8: 0.069
  [0.042, 0.101] with the aux head at weight 4 (A1); 0.083 [0.049, 0.118] with no aux loss (A2); 0.115 [0.076,
  0.156] with the aux's trunk gradient stopped (A5); 0.066 [0.038, 0.094] with no aux loss and the parent's
  AdamW moments (A6). The parent reads 0.191 [0.146, 0.236], the preflight's online 101-step net 0.0625 [0.035,
  0.094]. None of the packet's three suspects is needed: not the online data (offline reproduces it), not the
  aux head (A2, A6), not the fresh optimizer (A6). Six of six nets read after the parent read below it (the four
  arms, the online net, the twin's 12k 0.118). Two readings remain and these cells cannot split them: run10's
  step loses a strength run8's own training kept, or the parent's 0.191 tops a noisy series (run7's neighbours
  3k apart read 0.142 → 0.094 → 0.163; the parent read 0.142 on the old tree). The split costs no training:
  run8's own neighbours (42k, 48k, in the mirror) at the prereg instrument on the launch tree. Beside it: every
  warm start in this lineage dipped early (run7 from the BC net's 0.083 to 0.045 at 15k; run8, no aux head, from
  run7@42k's 0.111–0.142 to 0.056 / 0.062 at 3k / 6k). The aux's trunk pull is second-order: at 1 000 offline
  steps it costs value fit in proportion to its weight (train value loss 0.467 at w 4, 0.455 at w 2, 0.438–0.441
  with no trunk path); zero-init does not remove it, a stopped gradient does. Records:
  `mantis-records/run10-aux-probe/`; the arms' checkpoints and the cells' games are in the mirror.
- **CARD-WARM-START-DROPS-MOMENTS — the warm start copies weights only; not the lever at 101 steps.** The parent
  checkpoint carries its AdamW state (46 tensors, Adam step 45 000, lr 9.975e-4); `apply_bc_warm_start` copies
  the tensors and `Trainer.__init__` builds a fresh AdamW. A6 prices the moments: 0.066 [0.038, 0.094] with
  them, A2 0.083 [0.049, 0.118] without — no measurable difference.
- **CARD-HELDOUT-PROXY-BLIND — the frozen-slice readings do not see the start's damage (R373(d): no stand-in).**
  On the prereg's slice, A1, A2 and A6 @ 101 read beside the parent (own value loss 0.528 / 0.509 / 0.508 vs
  0.516; value MSE vs the parent 0.065 / 0.045 / 0.046, the online net 0.234) while strix reads them beside the
  online net; no reading (policy KL, value MSE, own policy or value loss) orders the known nets as strix does.
  Two facts for any successor: the slice's 3 031 rows touch 1 109 of the ring's 1 239 games, so an offline arm
  trained on that ring is partly in-sample on it; and a candidate must order the online net, A1/A2/A5/A6 @ 101,
  the twin's 12k, the SWA net and the parent before it stands in. Untested candidate: the same readings on
  positions from the strix cells' own games.

## Opened by the RUN10-REPICK packet (R372; 2026-09-26) — the aux start is not recovered

- **CARD-RUN10-AUX-START-DESIGN — CLOSED 2026-09-27 (R375(b), DECIDE-1's forward): not implicated — RUN10-CONTROLS read every 101-step arm, with or without the aux, at or near run8's neighbourhood. Was BLOCKING run10's START (R372(c)); RE-AIMED 2026-09-27 by RUN10-AUX-PROBE (R373(c)).** At strix @ r8 on the launch tree the
  twin's last save (step 12 000, weight 4) reads 0.118 [0.083, 0.153] against the pooled parent 0.191 (576
  games; 0.194 [0.150, 0.244] on 304 distinct) — above the preflight's 101-step control 0.0625 [0.035,
  0.094], below the parent it warm-started from. The fresh aux head's early clipped updates cost strength that
  12 000 steps did not return. A design packet decides how the aux head starts; the weight re-pick waits on it.
  R373(a): the pooling is void (a same-seed cell replays its games); the parent reads 0.191 [0.146, 0.236].
  RUN10-AUX-PROBE: offline, the 101-step cost needs no aux head (A2, no aux loss, 0.083) and survives the
  parent's optimizer moments (A6, 0.066), so the attribution above to the head's clipped updates does not hold.
- **CARD-STRIX-CELL-REPLAY — SPENT 2026-09-26 by R373(a)(b): a same-seed cell witnesses determinism and adds no sample; the §4 instrument keeps its seed.** The equal-work cell's
  `seed_base` is fixed (20260625), and on an IDLE box its play is near-deterministic: the parent's second cell
  replayed 272 of S1's 288 games byte for byte and read 55/233 again. Its CI is the book's opening spread, not
  run-to-run noise; a cell that must tighten a point needs a different `seed_base` (or book slice), which is a
  change to the §4 instrument and the operator's.

## Opened by the RUN10-PRESTART packet (R371; 2026-09-26) — run10 halted at the witness

- **CARD-RUN10-AUX-WEIGHT-REPICK — the §1a rule fires; HELD on CARD-RUN10-AUX-START-DESIGN (R372(c): C not recovered; R373(c): the re-pick follows RUN10-AUX-PROBE's reading). That card CLOSED 2026-09-27; run10 is held by R375(c), and the re-pick follows DECIDE-1. run10 will not START
  (R376(c)), so no run10 re-mint follows.** Over the twin's settled half (steps 7 188–14 375)
  `aux_policy_head_grad_norm / policy_head_grad_norm` reads a median 2.92 (p10–p90 2.24–3.83; by quarter
  3.22 → 2.99 → 2.90 → 2.93, flat), outside [0.5, 2]. The rule re-picks inside [2, 8] by re-mint with its own
  preflight; weight 2 predicts ≈ 1.46 (the head norm scales with the weight, prereg §8's foreseen case). The
  re-mint, the new launch tree and a fresh twin are the operator's word. Beside it: the preflight's terminal
  regression guard read the 101-step burst net at 0.234 vs its anchor (16 pairs, reject) — the fresh head's
  early clipped updates cost strength; the re-picked weight's twin should read it again. RUN10-AUX-PROBE: the
  101-step cost needs no aux head (CARD-RUN10-START-DAMAGE); offline, weight 2 halves the head's pull on the
  trunk (settled aux/main head-norm ratio 1.8 at w 2 vs 3.5 at w 4, steps 11–101).
- **CARD-RUN10-ONE-HOT-3K — the pre-START audit misses on the step-3 000 ring.** `one_hot_share_full`
  0.3151 (< 0.30) and `_mr1` 0.4002 (< 0.40); `_mr2` 0.2276 passes. Reported beside it, not a substitute
  witness: 6k 0.305 / 0.387, 9k 0.297 / 0.379, 12k 0.290 / 0.373 — falling, no collapse. Whether the 3k
  ring is the right witness for a run whose head starts fresh is the operator's reading.
- **CARD-RUN10-PARENT-BARS — CLOSED 2026-09-28, SUPERSEDED by R376(b): the process level is ≈ 0.11 on S and ≈ 0.29
  on X (DECIDE-1's panel, run8 36k–54k), and every bar prices from a panel. Was: the parent re-read moved.** strix @ r8 on the launch tree, IDLE: 0.191 [0.146,
  0.236] (55/288) against the recorded 0.142 [0.104, 0.181] (CONTENDED, the old box, pre-PERF-ADA). R371(c)
  sends run10's bars and the EMA conditional to the operator; the EMA cell read 0.149 (waits under either).
  RUN10-AUX-PROBE: six of six nets read after the parent on this tree read below it; whether 0.191 tops a noisy
  series is CARD-RUN10-START-DAMAGE's next reading.
- **CARD-PREFLIGHT-CHILD-LOGGING — CLOSED 2026-09-26 (RUN10-REPICK, `d158d27e`): the child configures logging; a boot row reads `run_safety_built` in `child_stderr.log` and reds without it.** The preflight child's boot narration was lost. `preflight_mint.py
  --_boot` never calls `configure_logging`, so its INFO lines (`heldout_slice_opened`, the prereg §6 boot
  witness) reach no file; the twin's `mantis.run` boot logged it (ring, 100 000 rows, 12 batches, 3 000).
- **The F5 reading (closes CARD-F5-LOOP-WITH-TRAINER).** run10's twin, 4 h on the 4080S box, window 3.71 h
  after 15 min: 1 512 games/h, 3 623 steps/h, 4 048 served leaves/s (3 067 GPU evals), cache hits 24.2 %,
  replay_ratio 8.16; GPU 81 % — inference ≈ 36 % duty at the idle bench's 0.118 ms per eval, the trainer ≈ 45 %
  (derived, one process); card peak 6.35 GiB, run-tree RSS 10.0 GiB, box used 13.1 GiB, load 13.2 of 30.7 CPUs.
  CPU, offline × in-run rate: leaf build + key 646 µs per leaf ≈ 2.6 cores, of which SHA-256 126–215 µs ≈
  0.5–0.9 cores (CARD-PERF-CACHE-KEY-HASH's share); server launch 0.57, collate 0.29 core-s/s; ≈ 9 cores
  unattributed (search, trainer host side — no in-run timer; no profiler attaches on the box). No single
  lever is shown ≥ 20 % in-run (R371(e)): PERF-2 follows run10. R378(g) moves it: "CENSUS-1 runs beside the
  twin; PERF-2 follows it" (CARD-CENSUS-1), and it cites this 24.2 % as the measured in-run cache hit rate.

## Opened by the FINISH packet (R370; 2026-09-26)

Grounds and numbers: the FINISH records (timing, reviews) outside the tree; STATE names the tip.
- **CARD-DEV-SPEED — levers (1)-(3) LANDED 2026-09-26 by the DEV-SPEED packet (its ruling is owed);
  (4) and (5) stay CARDED.** On the desktop CUDA venv, back to back with the load logged: the integration
  tier 2 922 -> 466 s and `make gates.exit` 3 076 -> 627 s; three tip runs of the tier 435 / 443 / 539 s,
  zero failures. (1) The phase split: every fixed wait between phases is <= 2 s (the 120 s receipt wait
  clears on its first poll), so the seven boots were compute — a ~9 s/step CPU burst and a 190-290 s
  terminal eval round — and no wait was converted. (2) The armed-smoke, boot-convergence and
  foreign-litter rows read ONE module-scoped preflight boot (`tests/tools/test_preflight_armed_smoke.py`).
  (3) The four in-process wiring rows boot `configs/smoke_wiring.yaml` (census-exempt; pinned
  leaf-for-leaf to the armed smoke by `tests/config/test_smoke_wiring_config.py`): 355-768 -> 8-13 s
  each. The tier's long pole is now that one preflight boot (~430 s in-tier). Open: (4) impact selection
  while iterating, gates.exit staying the full set; (5) the box as a remote gate host.
- **CARD-F5-LOOP-WITH-TRAINER — CLOSED 2026-09-26 by run10's twin (RUN10-PRESTART section).** FINISH F5 HALTED. `python -m mantis.run` refuses a run10-derived
  config with a throwaway run id (`PreflightStampMissingError`, also with `--inherit-preflight
  configs/run10.yaml`: the box holds no preflight stamp at all), and the grant excluded minting and
  stamping. The loop-with-trainer reading (games/h, trainer steps/h, GPU split, peaks, one profile)
  is owed; it needs either run10's own preflight (which vests a stamp a twin can inherit) or a ruling
  on a stampless measurement path.
- **CARD-PERF-CACHE-KEY-HASH — LANDED 2026-10-04 by PERF-3's L2 (`6ed55ef3`): the key is a Zobrist over the builder's
  inputs taken before any build (per-leaf touched CPU 908 -> 523 µs, the SHA's 280 µs gone); R370(c) annotated (A1).
  Was: the eval cache's key cost.** The key is SHA-256 over the whole encoded
  graph (~0.35–0.6 MB per mid-game leaf, copied into one buffer first); with the cache on, the box loop
  went GPU-bound → CPU-bound (GPU 91 → 72–76 %). Whether the hash is part of the CPU bound is
  unmeasured: bench `GraphKey::of` or profile the workers, then try streaming the fields into the hasher
  or a 128-bit non-cryptographic hash (both still hash the encoded input, R370(c)).
- **CARD-OC7-REAIM — CLOSED 2026-09-26 (R372(a), `164fd10e`): the row drives the minted 200 with no override, 32.0 s on the desktop; a planted no-terminal-write break reds it.** Was OWED (operator). `tests/train/test_clean_stop_save.py`'s real-boot row drives 50
  steps because the armed smoke's minted 200 measured over the tier ceiling; on `smoke_wiring` the
  50-step drive takes ~13 s, so the row could drive its minted 200 with no deviation. Re-aiming moves
  `_OC7_BOUND`, which its assertions compare against — a change to what the test asserts, so it is the
  operator's.
- **CARD-LIVE-AUDIT-WORDING — CARDED.** `armed_abort_live_audit`'s ERROR log ("the config armed them and
  the producer did not run") fires for a row the config itself disarms (`draw_rate_collapse` on
  `smoke_wiring`) and for `disk_space_exhausted` on a run shorter than the disk guard's first sample;
  every wiring boot logs it. The event is right to list the rows; the sentence claims an arming the
  config does not carry.
- **CARD-BENCH-SERVER-WINDOW — CLOSED 2026-09-26 (RUN10-REPICK, `b95715b5`): an empty sub-window waits for its first pop under a 60 s hang bound; a stalled-snapshot row reds the old window with this error.** Was CARDED. `tests/tools/test_bench_server.py::test_a_cell_serves_every_leaf_its_workers_submitted`
  went red on the UNTOUCHED base in DEV-SPEED's before-sweep (`no pops in the window (pops=0, wall=0.503
  s)`, gate 3a beside the Rust arm): a 0.5 s window that a loaded host can pass with no pop. Green at
  the tip and in every other sweep on record; a timing-window flake, pre-existing.


## Opened by the PERF-ADA packet (R369; 2026-09-25) — the levers after L2

Grounds and numbers: the R369 ledger in `docs/design/measurements/PERF_ADA_PROFILE_2026-09-24.md`. Order is the
recommended one; each is its own leg with a LAW-09 bench.
- **CARD-PERF-CSR — the fused CSR aggregation (B1): CLOSED 2026-09-26 (R370(a), FINISH sweep) — DONE at `c9392c95` (src/mantis/model/_gine_triton.py), box 3 881 leaves/s, repeat exact.** One Triton op over
  dst-sorted edges: gather + edge add + relu fused, fp32 register sum, one rounding, no atomics, no fp32 [E, H];
  its backward the same sum over src order. Desktop prototype: 3.3–3.9 ms per 4 layers (committed L2 44.1,
  fp32 atomics 7.1), bitwise equal to the committed L2 op.
- **CARD-PERF-EDGE-TABLE — LANDED 2026-10-03 by PERF-2's L2 (`7ad5208a`): bit-identical to the per-edge forward on
  sm_86 and on the box's sm_89 probe, +8.8 % at B 64. Serving only: the coded path raises under grad, so training
  keeps the per-edge forward. Was: the edge-code table (B2); second in PERF-2's order by R379(e), after the copies
  (CARD-PERF-2).** The edge embedding has 91 distinct raw rows, so each layer's
  `lin(edge_proj(·))` is a lookup; the table is bitwise equal to the per-edge GEMM when padded to M ≥ 1024 (sm_86;
  sm_89 unverified). Box forward 11.8 → 6.5 ms on top of B1. Reaches past R369(b)'s "aggregation": its own leg.
  **DECIDED 2026-09-25 (dispatcher, on the operator's "think through if this should be done"): NOT in PERF-ADA.**
  It couples the model to the encoding's 91-row edge vocabulary (needs a loud guard), its bitwise equality rests
  on cuBLAS kernel choice at padded M (unverified on sm_89 and exposed to L3's torch move), its backward needs a
  deterministic per-code reduction, and once B1 lands the serving bound is expected to move to the CPU collate
  (CARD-PERF-4). Re-decide on the exit's real-loop reading: if the GPU is still the in-run bound, it is next.
- **CARD-PERF-READOUT — the deterministic readout (B3), OPERATOR-ENDORSED 2026-09-25. CLOSED 2026-09-26 (R370(a), FINISH sweep) — DONE at `d5605bf2` + `12f829f9`: fixed-order fp32 `segment_reduce`, repeat exact.** The served softmax's
  `segment_sum` and the value/mean pools sum by atomics (served probabilities jitter ~1e-7 on a repeat); a
  segment reduction over the CSR offsets makes them exact. A precondition for L4's bit-identity witness.
- **CARD-PERF-BATCH-INVARIANCE — served outputs that do not depend on the pop's size; then the eval cache. CLOSED 2026-09-26 (R370(c)) — the cache's precondition is now a replay inside the served path's own batch-size spread, not bit-identity; the cache landed under it at `373c31d1` (box loop +16–21 % positions/s, 35 % hits). Batch invariance itself stays unbuilt and is no longer anything's precondition.**
  R369(d)'s precondition failed: the value head's bf16 GEMM runs over the batch dimension and its kernel
  varies with B (a single-position pop serves |Δvalue| up to 0.0076 from the same position in a full pop).
  A fixed-M head (pad to one tile) or a batch-invariant kernel set makes it exact; the cache itself is built
  and parked on branch `perf-ada-l4` (sharded FIFO, full-position key, per-version, served/GPU counters,
  the runner field off by default so `served_sims_exact` stays unchanged). 35.8 % of leaves are repeats.
- **CARD-PERF-DST-SORT — dst-sorted edges from the Rust builder.** Removes the per-forward GPU argsort
  (0.5–0.8 ms/pop INF); a wire/golden contract change.
- **CARD-PERF-GRAPHS — LANDED 2026-10-03 by PERF-2's L3 (`355c7dc9`) and L3b (`e6b9e0e8`): twelve padded buckets at
  ratio 1.25 down to 4 096 nodes, one capture per bucket per server, +10.8 % at B 64 with the launch 11.05 -> 6.72 ms.
  Was: CUDA graphs for the serving forward; third in PERF-2's order by R379(e) (CARD-PERF-2).**
  Dynamic shapes need bucketing; worth it only once
  the server is CPU-bound, after CARD-PERF-4's re-read.

## Opened by R368 (SLIM-FIX; 2026-09-23)

- **CARD-RUST-HOTLOOP-EXPECTS — CLOSED by PERF-ADA H2 (e5658e4b, 1cfe2ced, 1e9b2e78, a80d0587; box bench flat, 3 981 vs 3 982): the `unwrap()`/`expect()` sites on hot loops, classified by R368(g)'s
  correctness pass and left as they are, because a named-error rewrite there is a hot-path change that needs
  LAW-09's one-change-one-bench.** `mantis-core` `Board::check_win` (a cell lookup `apply_move` has just set);
  `mantis-selfplay` `queues/graph.rs` (13 lock/condvar-poison expects on the per-batch inference path, plus the
  infallible `position()` in `submit_graphs_and_wait`); `runner/search_drive.rs` `infer_and_expand_graph`'s
  `win_length`/`graph_radius` expects (the fix is hoisting both into the worker's inference context at start)
  and `select_move`'s guarded fallback `choose().unwrap()`. `mantis-search` has none left.
- **CARD-POISON-STANCE — CLOSED by PERF-ADA H1 (a38d9dff, 1f1062b3, b3e09d7b; the planted panic in Drop keeps the save): one stance on a poisoned `Mutex` in the self-play runner.** Nine production
  `lock().expect(..)` sites (`runner/finalize.rs` ×2, `runner/mod.rs` latch / fatal read / `stop` / the two drain
  faces, `runner/spawn.rs`, `search_drive.rs`'s latch store) panic on a poisoned lock. `stop()` runs from `Drop`
  and cannot return a `Result`, and a latch must not itself fail. The measured recommendation (the W2 selfplay
  leg): latch, stop, finalize and spawn take `PoisonError::into_inner` (the panic that poisoned the lock is
  already counted by `worker_panics` and halts the run); the drain faces return a named error the bridge raises.
- **CARD-SEARCH-HOT-DUP — CLOSED by PERF-ADA H3 (5b423c5e, 453912ba, 68350493; monomorphic, box bench flat): the MCTS hot-path duplicates the census found (S-A-RUST-1-13, -14:
  shared selection/expansion helpers; -15: the `action_idx` decode re-typed inline).** Each lands only
  monomorphic (a shared fn, no kind flag), with LAW-09's bench and `search_kind_conformance.rs` as witness.
- **CARD-SCHEMA-KEY-RETIREMENT — CLOSED for `train.value_target` by PERF-ADA H5 (923a7882; both run8 parents load, `mantis.config.retired` is the one authority): retiring a schema key and its minted rows (first:
  `train.value_target`, S-A-CORE-2-14).** A retirement needs a loader witness over every mirrored parent
  stamp before the key leaves the schema; not before run10 STARTs (R368(i)).
- **CARD-CLUSTER-THRESHOLD-RESIDUE — CLOSED by PERF-ADA H6 (177d11df; the golden fixture keeps the field as capture provenance): `Board.cluster_threshold` is write-only since the cluster BFS
  went (R368 W2).** Plumbed from the registry (bridge `board.rs`, selfplay `game.rs`) into `BoardGeometry` and
  the golden-replay fixture's geometry, read by nothing; removal touches `BoardGeometry` and that fixture's field.
- **CARD-SEAM-2 — HOLD LIFTED 2026-09-27 by R375(e): may merge before any run starts. Was HELD (R368(j)): the seam for a kind that brings its own head, objective and config rows.**
  L-SEAM-01..04 fold into it; the design packet follows the SLIM-FIX phase; R368(j) merged the implementation
  only after run10 STARTs, and R375(e) lifts that hold.
- **CARD-W5-RESIDUE — CARDED: small residues the W5 leg found and did not fix, each its own class.**
  The graph drain goldens have no committed generator (the capture script was a scratch file; kept, it
  should be a `tools/` generator). `ResolvedPoolEncoding.board_size`/`trunk_size`/`n_kept_planes`
  (`src/mantis/selfplay/hparams.py`) have no `src/` reader but are golden-pinned. A supervisor
  poll-sleep plant hangs `tests/monitor/test_supervisor.py` instead of redding it. `tests/_determinism.py`'s
  context has no CPU witness. `disk_guard.keep_all`'s tombstone (`src/mantis/train/lifecycle/disk_guard.py`)
  could become a "deliberately absent" CLAUDE.md row; `RegimeKey`'s (`src/mantis/arena/regime.py`) `==` leg
  is not re-asserted. `test_coordinator_knobs_wiring.py::_real_graph_ring` duplicates
  `tests/train/_graph_drive.py::filled_hexg`. L-STYLE-04 (subprocess `text=True` without `encoding=`, needs
  a gate-16 widening ruling before it can be fixed). L-STYLE-10 (function-scope imports, fixed on contact,
  no tracking needed).
- **CARD-GAME-RECORD-STATUS-DRIFT — CARDED (found by REVIEW-W6): the game-record contract still says
  self-play search stats have no producer.** `docs/contracts/game_record.md`'s status line (and
  `docs/design/repo_design.md`'s game-record amendment, point 4) cite `CARD-GAME-RECORD-SELFPLAY-STATS`
  for "no producer on the self-play channel", but `selfplay.search_stats_every` is that producer
  (R355(d), `18eb4f4e`) and the same contract's `search_stats` section already describes it. Repaired on
  contact: the contract's text, version and tests move in one commit; repo_design's by an R9 amendment.

## Opened by R367 (DESIGN STANDARD + REVIEW GATE; SIZE CONDITIONAL WITHDRAWN; PRICE LAW; 2026-09-21)

- **CARD-NET-EXPAND — HELD by R379(d): growth is parked (CENSUS-1's C5 reads no shape winning on the frozen 45k ring;
  CARD-CENSUS-1-PARKED). Was run11's build: a FUNCTION-PRESERVING width/depth expansion of the trunk behind the
  seam, with a conformance section proving output equality at expansion.** OPENED by R367(c) from
  CARD-RUN10-SIZE-PARENT (CLOSED below): no shape-compatible parent exists for a 6×192 `GnnNetV2`, so the
  size row leaves run10 and returns as an EXPANSION — a wider/deeper net initialised FROM the 4×128 parent
  so that its served outputs equal the parent's at the moment of expansion, behind the arch seam
  (`identity.arch_kind` / `model.gnn`, v35) as a warm-start path rather than a new kind; the conformance
  suite gains a section that builds the parent, expands it and asserts output equality on the fixture
  batch. Design before code (R9): the expansion's mechanism and its identity proof are a design doc under
  `docs/design/` first; the expanded shape's leaves/s is `bench_server`'s reading on the box (IDLE, B 64)
  and is recorded before the row is armed. Not run10's; nothing perf rides it (R366(e)).
- **CARD-MECHANISM-SWEEP — the PRE-EXISTING inventory R367(a) now names, out of the R366 range and out of
  the fix leg's scope (REVIEW-1 F1.6, F4.2, F2.1).** Measured by REVIEW-1 at `19e8351d`: a `RUN5`
  constant bound to a different run's config (a name that lies about its file) in four test modules
  (gone by 2026-09-26: no `RUN5` symbol remains);
  32 test functions carrying a `runN` token in 16
  files; ≈ 800 `run5`/`run6` identifier tokens across `src`/`tests`/`tools`; the analyzer tests' (under `tests/tools/`)
  synthetic `"run9"` run ids; `crates/mantis-selfplay/tests/dirichlet_inert_on_gumbel.rs`'s run9 message;
  `tests/config/test_every_key_has_consumer.py`'s at_max_pairs note; the file-sha256 copies beside the one
  `sha256_file` (DONE, R368 W3: every src/ and tools/ file hash reads `mantis.util.hashing.sha256_file`, kept
  apart only `tools/audit_bootstrap_corpus.py` (mantis-free by design), `tools/ci_gates/preflight_mint_parent.py::_sha256`
  (gate code, protected) and `src/mantis/diagnostics/worker_sweep.py::_sha256` (OPEN — a thin wrapper that
  already delegates to `sha256_file`, so the fold is a call-site rename, not a re-implementation));
  the 21 remaining private `_Pool`/`_Buffer` fakes of the composition tests (the trainer stub is one since
  F2; the pool stub is one for the coordinator tests only) (DONE, R368 W5: the root composition, train
  wiring and config wiring pools are `tests/_drivable.py::DrivablePoolStub`, their buffers
  `tests/train/_graph_drive.py::GraphSampleBuffer` or `_drivable.BufferStub`; OPEN: the draw-rate
  files' pools (PZ-1) and those of augment_sym_counter, quiescence_fires_producer and rates_are_measured);
  the nine full-config literal dicts whose `model` line the fix leg left (each is a complete config a
  schema test owns); the eight remaining bare `class _Trainer:` fakes NOT on the shared
  `tests/_drivable.py::DrivableTrainerStub` (W5 folded most; re-grepped OPEN: test_abort_exit_signal.py,
  test_anchor_wiring.py, test_bc_graph_reroute.py, test_clean_stop_save.py, test_cluster_stat_wiring.py,
  test_gate_interval_decoupling.py, test_ply_cap_gate.py, test_policy_loss_trough_gate.py, all under
  `tests/train/`); REVIEW-1's F5.6 (the
  frozen held-out slice's buffer allocated at the training ring's capacity rather than the file's own
  rows; the fix is a header-reading constructor in the Rust bridge). Applied ON
  CONTACT (R316(e)'s rule for comments, extended by R367(a)) — a leg that touches one of these files fixes
  what it touches; no tree-wide pass is ordered.

## Opened by R366 (RUN10: THE POLICY-CAPACITY RUN; 2026-09-21)

- **CARD-PERF-4 — OPENED by R366(e), its own packet; NOTHING from it rides run10.** PERF-3 step 3 read
  the CPU launch stage at 0.34–0.39 ms per leaf at EVERY batch size (`PERF3_2026-09-18.md` §step 3: alone
  2 236 / 2 394 / 2 466 / 2 055 / 2 091 leaves/s at B 16 … 256, no knee, D-1 dead by its own falsifier), so
  the stage is linear in B and is the bound. The design: a Rust-side collate + a pinned host buffer + a
  single launch per pop (the fused graph as ONE H2D and ONE forward). `tools/bench_server.py` is its
  falsifier (the same B-curve on the same box, IDLE and CONTENDED, R361(d)); the pre-registered success
  line stays PERF-3's ≥ 1.4× on the same machine, and the determinism probe (the same-cell bf16 witness)
  must read 0 — a faster server that serves different numbers is not the same server. Design only after a
  ruling; the box work is a perf-host event, not run10's window. **WAITS on a re-read (R369(e)):** its
  premise was read off `launch − gpu_wait` while the mask sync inflated `launch`; the old probe is void and
  the same-input repeat probe replaces it (`PERF_ADA_PROFILE_2026-09-24.md`, R369 ledger).
- **CARD-RUN10-SIZE-PARENT — CLOSED by R367(c) 2026-09-21 into CARD-NET-EXPAND (run11).** Grounds: a
  6×192 `GnnNetV2` shares NO tensor shape with the 4×128 parent (`input_proj` 11→192, every conv
  192→192, the JK-cat readout 6 × 192 = 1 152 wide into both heads against 512) and `load_from_bc` is
  strict both ways, so the warm start cannot land — no shape-compatible parent exists. Stated in
  `RUN10_PREREG_2026-09-21.md` §2.
- **CARD-RUN10-WEIGHT-ENVELOPE — the aux head's weight, minted 4 inside [2, 8], picked in the twin.**
  The rule (R366 §0(2), prereg §1a): the SETTLED ratio `aux_policy_head_grad_norm / policy_head_grad_norm`
  on `trainer_step` over the twin's window in [0.5, 2] → 4 stands; outside → re-pick inside [2, 8] by
  re-mint with its own preflight; no value in the envelope reaching it → HALT. The CPU pre-read (prereg
  §1b/§8) established the shape the twin will see: a FRESH head starts at ≈ ln(legal) ≈ 6.2 nats of CE
  with its own gradient norm an order of magnitude above the trained main head's, so under `grad_clip`
  1.0 the first steps' clipped update is mostly the aux head's — the witness is the settled ratio, never
  the first steps'. Spent when the twin's reading is on the record.

## Opened by R365 (RUN9 NOT STARTED, PROBE-1, RUN10 DESIGNED FROM MEASUREMENTS; 2026-09-21)

- **CARD-PROBE-1 — ORDERED by R365(b), dev + mirror, 0 box-h, ≤ 2 dev-days; the record is ONE file,
  `docs/design/measurements/PROBE1_2026-09-21.md`.** Six readings on run8's mirrored checkpoints and
  rings, each row carrying its instrument, n, number, the run10 change it admits or kills and its
  PRE-STATED line (the packet's §2, copied into R365's entry): (1) the one-hot DECOMPOSER first — the
  full-arm share with tail-only (α = 1.0) rows excluded, bucketed by `moves_remaining`, on the 3k/15k/
  30k/45k/51k rings; the band is re-derived from it, never asserted (it corrects an instrument the others
  read); (2) E3 mr-CALIBRATION — dist65 E[v] vs realised z bucketed mr = 1 / mr = 2 on run8@45k over
  ≥ 20k ring positions (error(mr=1) ≥ 2× error(mr=2) → a named target, admits P-B4 / the proof lane,
  kills C2; within CI → the premise closes); (3) C3-1 GAP — train-vs-held-out policy/value loss on the
  18 checkpoints, held-out = a ring from a different step never sampled (gap < 0.1 nats, both high →
  UNDERFIT, admits 6×192 and reuse; > 0.3 → OVERFIT, kills reuse-8, admits window/augment/LR floor;
  between → report); (4) P-B2 PROOF RATE — the tactics solver at depth 6 / 2 000 nodes over ≥ 5k ring
  roots: proof rate, novelty (proof ≠ target argmax), ms/root (≥ 3 % AND ≥ 5 % AND ≤ 50 ms → the
  un-probed-hypothesis candidate; else dead, filed); (5) P-B1 KL — median KL(prior‖target) over full-arm
  rows on 45k (< 0.02 nats → the soft-policy head is dead; ≥ 0.1 → admitted, queue (vii)); (6) the
  SPREAD SERIES — the analyzer's symmetry spread over ≥ 12 positions × 18 checkpoints (falling while
  strix rises → CARD-ARCH-D6 stays parked; flat/rising → admitted to run11's design); (7) the EMA CELL at
  the box in run10's preflight window, 1.5 bh — the weight-average of run8's 30k–51k checkpoints vs
  strix 256/256, 288 games (> 0.181 → EMA deploy rides run10, a server-owned copy its prerequisite;
  ≤ 0.142 → dead; between → the LR floor is run10's hypothesis instead). Order 1 → 2/3/5 (one mirror
  pull) → 4 → 6; 7 at the box. Then R366 composes run10 under R365(c). **READ 2026-09-21, rows 1–6 in
  `PROBE1_2026-09-21.md` §8; SPENT by R366(a)–(b)** (the head admitted and built, the gap witness a
  producer, the band re-derived, proof-as-target and the mid-turn premise dead, ARCH-D6 parked). Row 7,
  the EMA cell, is run10's preflight window (prereg §2): the card's last open line.
- **CARD-RUN10-RULE — R365(c), carried until run10's ruling applies it.** A change whose mechanism a
  PROBE-1 reading supports in our regime may ride run10 without being the single swap; run10 carries at
  most ONE un-probed hypothesis. Attribution before the run replaces attribution by the run. This amends
  R359(c)'s one-swap-per-run order for run10 only; every change still carries its own in-run producer
  (LAW-18) before it is armed. **APPLIED by R366(b), 2026-09-21 — SPENT:** the probe-supported changes
  riding together are the soft-policy head (reading 5), the data regime from run9's mint (reading 3's
  witness beside it), the two conditionals (reading 7 and the bench); the ONE un-probed hypothesis is the
  LR anneal. Each carries its producer (`RUN10_PREREG_2026-09-21.md` §3).
- **CARD-E1-RULER-R6 — the cell R365 §0(3) ordered at the box, 1.2 bh; the tooling LANDED 2026-09-21
  (`d4804e58`).** The parent run8@45k (`3aef7883…`) at PUCT-256 vs strix 256 sims with the driver's
  `placement_radius` 6 (strix's own trained radius; every reading on record rides the driver's default
  8), 288 paired games, `book_v1_s20260625_p4`, the unit `ruler_r6` of `tools/strix_follower.py --once`
  (variant `<stem>:r6`, sidecar `.strix256_r6.json`). Pre-stated reading (`STRENGTH_RESEARCH_2` E1):
  within ± 4 pp of the r8 cell's 0.142 [0.104, 0.181] → the ruler is radius-stable and the caveat
  drops; a move ≥ the CI half-width → "strix @ r8" becomes a stated unit qualifier on every follower
  point and every bar priced in that unit. The instrument's own signature: at r6 every strix reply's
  legal set is a strict subset of our r8 fence, so the bot's `fence disagreement` finding fires on
  every strix move — recorded in the cell, gated by nothing, and it IS the change under test.
  **READ 2026-09-21 09:40 UTC (`PROBE1_2026-09-21.md` §E1): 0.080 [0.049, 0.115]** (23–265–0, median 41
  plies, 2 804 s, IDLE, 6 690 findings, 0 out-of-fence) against the r8 cell's 0.142 [0.104, 0.181] —
  a −6.2 pp move, the CIs disjoint: **"strix @ r8" IS a unit qualifier on every follower point on the
  record**; strix at its trained radius is the stronger ruler. The series and the parent choice stand
  (every point shares the unit); a bar quoted "vs strix" without the radius does not. Whether the
  follower's unit MOVES to r6 (a re-read of the parent series, ≈ 3 × 1.2 bh) is R366's; the receipt is
  mirrored, the card is READ and waits on that ruling. **R366(d), 2026-09-21: strix @ r8 STAYS the series
  ruler; r6 is a QUALIFIER, read once per run at block end if the box allows (1.2 bh). Every vs-strix
  citation on the record carries "strix @ r8" — the run8 series (0.104 / 0.135 / 0.142), the parent's
  baseline and run10's bars (0.192 / 0.142) are all r8 numbers. The card is CLOSED as an order and stays
  as the qualifier's statement.**

## Opened by R364 (RUN8 STOP, RUN9 = DATA REGIME, GATE AS REGRESSION GUARD; 2026-09-21)

- **CARD-ARCH-D6 — OPENED by R364(d), NOT ARMED; the lever after the knobs.** A D6-equivariant net
  behind the capability seam (the arch selector, `identity.arch_kind`; `GnnArchV2` is the incumbent).
  Witness BEFORE any training: the analyzer's symmetry spread per checkpoint (ANALYZER-1's first
  reading, R363(d): run8@18k spreads 0.146 over the 12 elements on a CHECK position, translation exact,
  argmax 12/12) — an equivariant net reads 0 by construction, and the detector that reads it is what
  lands first; then a conformance test (the 24 × 12 D6-lossless positions of
  `crates/mantis-graph/tests/d6_lossless.rs`, the net's outputs equal across the orbit to a stated
  tolerance) before a single training step. Order: after the queue's knobs (CARD-RUN9-QUEUE: (i) rides run9; (ii) LR waits on
  `PARAM_DISTANCE_2026-09-21.md`, run10's; (iv) prior temperature queued while one-hot is flat). Not a
  run9 change; not a run10 change without its own ruling. **R365(b), 2026-09-21:** the spread SERIES
  (≥ 12 positions × run8's 18 checkpoints) is PROBE-1's reading 6, with its line pre-stated there —
  spread falling while strix rises → this card stays parked; flat or rising → admitted to run11's design.
- **CARD-RUN9-STOP-LAW — the rule R364(a) states, carried until run9's own stop discharges it.** A run
  STOPS at its last pre-registered read unless a ruling extends it; run9's line says "read 45k-games
  THEN STOP" (`RUN9_PREREG_2026-09-19.md` §3). Nothing in the config stops the run (`max_train_steps`
  1 000 000, byte-equal by R364 §0(3)); the box session's SIGTERM does, once the deciding cell is read.
  Grounds: run8 ran 8 h past 45k on an omission (A1 under R363's foot). **R365(a), 2026-09-21: run9 is
  NOT STARTED; the rule carries UNSPENT to run10's line** (run8's own stop — 08:23:44 UTC at 55 170, one
  SIGTERM, `shutdown_save` — was the rule's first application).

## Opened by the LADDER-1 packet (2026-09-19, register head R362); moved by R363 (2026-09-20: the unit FIXED, the server items CARDED)

- **CARD-LADDER-RUNG — OPENED 2026-09-19: design + shakedown LANDED; the UNIT FIXED by R363(c) 2026-09-20; NOT an instrument of record until its admission cell is read.**
  `tools/ladder_bot.py` on the `tools/ladder/` package (client, wire, backends, session, receipt;
  `tools/ladder/vps/` carries the unit file, the CPU build recipe and the README): one process per
  registered bot on a HeXO server's bot API, the stream held (presence and availability ARE the
  connection — there is no registration endpoint), every move request answered through ONE backend
  (`mantis`: the deploy head on a named checkpoint, every knob the run config's; `strix`: the pin at
  256 sims, solver ON, noise off), one receipt per game keyed by net hash. Posture A: ladder games are
  EVAL, pinned by a test that the package imports no ring writer. Endpoint strings live in
  `tools/ladder/client.py` alone (pinned by test). The witnesses held BEFORE the reading: determinism
  (every live receipt replays move-for-move through its backend, 4 of 4), budget (read off
  `DeployHeadPlayer.last_sims` / `StrixBot.last_sims`, both added: 512 leaves per compound turn on
  every undecided position; a decided one stops early on BOTH backends — strix's solver at 3,
  mantis's exhausted tree at 427 — reported, reproduced, never a verdict), the receipt (LAW-07 planted
  break + mutation self-test). The shakedown: 20 games parent (42k) vs strix on the operator's server, Mantis 0 of 20 — which is 0 of 2
  DISTINCT games (LAW-04), consistent with the parent cell's 0.111 (P = 0.79) and a reading of NOTHING;
  0 rejections, 0 drops; mantis ≈ 13 s and strix ≈ 3.4 s per compound turn on the dev host's CPU. The record is
  `docs/design/measurements/LADDER_SHAKEDOWN_2026-09-19.md`; repo_design carries the amendment
  admitting a SECOND operator-side tool under `tools/`.
  **The deployed server (TimmyBurn2/HeXO `deploy` @ 8166053, verified live) is ahead of the spec
  file (Hexo-Bot-Api 0.2.0) in four places the client follows:** `firstPlayer` is
  challenger|challenged|random (not host|guest); a challenge carries no `rated` (every bot game is
  unrated); `challengeCanceled` carries `reason` and `Challenge.status` includes `expired`; a guest's
  `elo`/`profileId` are null. **Server findings for the operator:** the finished-game record's
  `moves[]` is racy within a compound turn (fire-and-forget `appendMove`; sort by `moveNumber`, which
  is right and starts at 2); house bots refuse bot challenges, so a ladder needs two stream bots; two
  deterministic bots from the auto-placed origin play the SAME game every time — the server has no
  opening variety, and a series needs it from the server, the challenger or per-game noise.
  **What remains:** the operator installs on the VPS; the witnesses on the VPS; ONE 288-game cell
  read beside a follower cell on the same checkpoint — impossible as a distinct-game count until the
  opening question is answered — then R363 or later admits or refuses the rung. Not a series, not a
  promotion input, not in any prereg until then.
  **R363(c)/(e), 2026-09-20 — the unit and the admission test.** The unit is `book_v1_s20260625_p4`
  PAIRED, the follower's own: an opening needs no fairness, a pair does. Opening index = match index
  (pair `m` = games `2m` and `2m+1`, colours swapped by the challenger's alternating `firstPlayer`,
  both games on opening `m` in the book's FILE order — no seed, no permutation), a CONVENTION BETWEEN
  OUR TWO BOTS because the server's challenge carries no opening field: both bots translate opening
  `m` onto the server's auto-placed origin and play its prefix before searching (the second player's
  first compound turn is plies 2–3 of the book; the first player's first compound turn is ply 4 plus
  ONE searched stone), a board that leaves the prefix is off-book and searched, the receipt records
  the opening (book, index, id, the stones it forced, where it went off-book) and `--replay` re-derives
  the forced stones from the receipt's index. The witnesses (determinism 4/4, budget 512 per compound
  turn, receipts keyed by net hash) are the instrument's BUDGET PROOF; the shakedown's reading (2
  distinct games) is a finding about openings, not the bots. **Admission:** ONE 288-game cell, the
  parent vs strix, IDLE (the VPS or the dev CPU), read beside the follower's parent cell 0.111
  [0.073, 0.149] on the same checkpoint — then a ruling admits or refuses. Not an instrument before
  that. The 64-sim "play" preset R363 §0(5) allows is the ladder tool's own row (`--preset play`),
  labelled on the receipt and never a unit reading.
- **CARD-LADDER-SERVER-ASKS — RECORDED 2026-09-20 (R363 §0(4)): the six server items are the
  OPERATOR'S; no ruling on them.** What the shakedown and the client found on the deployed server
  (`TimmyBurn2/HeXO` `deploy` @ 8166053) that only the server side can change, kept here so no session
  re-derives them or rules on them: **(1)** `firstPlayer` is challenger|challenged|random where the
  spec file (Hexo-Bot-Api 0.2.0) says host|guest; **(2)** a challenge carries no `rated` and every
  bot game is unrated server-side (the spec says otherwise; whether bot games should rate is the
  operator's); **(3)** `challengeCanceled` carries `reason` and `Challenge.status` includes `expired`,
  neither in the spec; **(4)** a guest's `elo`/`profileId` are null, the spec has them non-null;
  **(5)** the finished-game record's `moves[]` is racy within a compound turn (`appendMove` is
  fire-and-forget) while `moveNumber` is right and starts at 2 — the receipt sorts by `moveNumber`;
  **(6)** house bots (SealBot) refuse bot challenges ("played from the lobby dialog"), so a ladder
  needs two stream bots. The seventh finding — no opening variety between two deterministic bots — is
  NOT a server ask: R363(c) answered it on our side (the book prefix as a convention between our
  bots). Items (1)–(4) close by updating the spec file or the server, (5) and (6) by the server;
  none blocks the admission cell. The list was read off `LADDER_SHAKEDOWN_2026-09-19.md` §4 and the
  card above; the packet's own enumeration was not in the text this session received, so an item
  the operator meant differently is corrected here in one line.

## Opened by R361 (EVAL COST: STOP THE BLEED, COUNT, THEN REDESIGN); moved by R362 (RUN8 TO 30K, RUN9 EVAL ROWS)

- **CARD-EVAL-REDESIGN — RULED by R362(c) 2026-09-19; the rows LANDED on `dev` the same day.**
  Gate cadence 15 000 (run9's `train.eval_interval` mint row, recorded in
  `docs/design/measurements/RUN9_PREREG_2026-09-19.md`, minted at 30k with the parent); the
  sealbot rung DELETED in one commit with `sealbot_wr_abort`, `sealbot_wr_warn`, the ladder
  state, `eval_channel_health` and `wr_sealbot` (contract v33, repo_design amendment R362(c));
  gate sims 256 and the GSPRT 0.52/0.62 UNCHANGED (64 unmeasured; H0 0.50 withdrawn, R362(d));
  the gate's rule fields on the stream (the card below). Gate = internal comparator and parent
  selector; strix cells = external scale. **MOVED by R364(c), 2026-09-21: the gate is a REGRESSION
  GUARD on run9** — H0 0.42 / H1 0.52 (the zero-drift point 0.47: an equal candidate drifts to
  accept), cap 104, `eval.gate.sequential.at_max_pairs: promote` (contract v34 — the cap promotes; a
  GSPRT reject is the one way the anchor stays; run7/run8's files re-minted with `sign`, what they
  ran); the cadence 15 000 is GAMES (36 000 steps at 2.4); parent selection stays the strix triple
  (R363(b) applied). Grounds: r6–r17 all rejected against the 15k anchor under 0.52/0.62 (two cap
  rounds at 0.536 / 0.524 rejected by the sign) while strix rose 0.104 → 0.142 — a gate that cannot
  resolve the run's gain selects nothing and cost 51 % of run8's wall. What remains OPEN under this card: gate sims 64 as a
  MEASURED cell (a bench in run9's preflight window beside PERF-3 step 3, not a census) — the one
  pre-stated direction R362 neither adopted nor killed. The census's reading, kept for the
  record: what the census gives each pre-stated
  direction (R361(c)): promotion selects NOTHING in the self-play loop — the actor runs the learner's
  weights on a 2-step cadence and the promoted net is only the next round's anchor and the next
  run's parent — so the gate is an instrument; gate cadence 15 000 returns ≈ 11–12 h of run7's 84
  (25 rounds → 5; a round costs the trainer 24–32 % while it runs, 50 h of run7 was in one); the
  sealbot rung read 0.657 ± 0.045 flat over run7 while strix swung 0.045 → 0.170 → 0.111, at 34 %
  of eval wall, and did not separate promoted from rejected rounds — deleting it retires
  `sealbot_wr_abort` (DEFERRED, inert), `sealbot_wr_warn`, the ladder state, `eval_channel_health`
  and `wr_sealbot` in the same commit (R4/LAW-07); GSPRT H0 0.50 is −7 % pairs (≈ 1.3 h over run7)
  for an equal candidate's false-promotion 2 % → 5 % — the long end is the (μ0+μ1)/2 = 0.57
  midpoint, not H0; gate sims 64 is UNMEASURED (the gate game is the unit that costs: 38–40 s at
  256/256 over 91–93 plies; a 64/64 cell's s/game is a bench, not a census).
## Opened by R358 (RUN8 RE-MINT, THE NET-ONLY CELL, THE RUN9 QUEUE); moved by R359 (READINGS LANDED, QUEUE SOURCED, PERF-3 ISSUED)

- **CARD-RUN9-QUEUE — RECORDED, nothing armed (R358(f)); sourced and ORDERED by R359(c).** One
  swap per run, each carrying its own in-run producer before it is armed. The entries: (i) DATA
  REGIME — `replay_capacity` 500 000 with `training_steps_per_game` set so the MEASURED replay ratio
  (`ring_audit --events`, run7 3.6; shakedown8 READ 3.02) holds ≈ 8, strix's setpoint; (ii) LR —
  AdamW 2e-4 → 2e-5 cosine over the block horizon; (iii) the root proof solver — OUT of the queue
  2026-09-18 by the net_only cell's reading (solver ON 0.111 vs OFF 0.115, Δ +0.3 pp within CI; the
  PLAY-TIME solver only — the training-side proof-as-target hypothesis stayed untested, not refuted),
  until new evidence; (iv) PRIOR
  TEMPERATURE — KataGo's root policy softmax temperature 1.25 → 1.1 (g170), applied to the logits
  BEFORE the Gumbel top-m draw (source: KataGo's upstream `KataGoMethods.md`, read 2026-09-18);
  (v) sims; NEW (vi) POLICY-SURPRISE WEIGHTING — half the sampling weight uniform, half ∝
  KL(prior ‖ target) per row; producer = the in-run distribution of the sampled rows' KL (same
  source); NEW (vii) AUXILIARY SOFT-POLICY HEAD — a second policy head on target^(1/4)
  (renormalised), nominal weight 8, behind the seam with a producer test (same source: KataGo's
  T = 4, weight 8). A value-target swap is NOT a queue entry: KataGo's short-term value heads are the
  family the landed-UNARMED λ-return codec already was (`f049ef02`, imported by nothing; deleted by
  HYGIENE-1 at `482684f2`, recoverable from either). ORDER after run8's 15k/30k reading: (i), (vii), (ii), (vi),
  (iv), (v). run9's mint also DROPS the `selfplay.mcts.dirichlet_*` rows (R359(d): inert on the
  Gumbel arm by code, so a cosmetic) with one pin that the Gumbel arm never applies Dirichlet.
  FALSIFIED run8 → run9 = parent + A-2 + augment with run7's σ (rescale): σ leaves, augmentation
  stays (R358(b)). **R364(b), 2026-09-21: (i) RIDES run9** — `train.replay_capacity` 500 000,
  `train.training_steps_per_game` 2.4 inside the envelope [2.0, 3.0] (predicted reuse 7.7; the
  shakedown's `replay_ratio` decides whether it stands, `RUN9_PREREG_2026-09-19.md` §1b), `max_train_burst`
  8, the budget's remainder now CARRIED (`mantis.train.mixing`, a correctness fix the envelope needed:
  at HEAD one game per burst realised integers only); the `dirichlet_*` rows dropped to `false` with the
  pin landed (`dirichlet_root_fires`, `crates/mantis-selfplay/tests/dirichlet_inert_on_gumbel.rs`).
  (ii) LR WAITS on the parameter-distance test (READ: `PARAM_DISTANCE_2026-09-21.md` — the step flat,
  the coherence falling 0.15 → 0.03, the weight norm +49 %; the call is the operator's, run10's).
  (iv) prior temperature STAYS QUEUED (one-hot 27 % flat). The gate rows beside the swap: H0 0.42 /
  H1 0.52, cap 104, `at_max_pairs: promote` (CARD-EVAL-REDESIGN). **R365, 2026-09-21: run9 is NOT
  STARTED (a 288-game cell cannot resolve the run's slope, E4); its mint stands as run10's base, (i)
  un-armed and un-read.** The queue is re-ordered by measurement, not by list: PROBE-1 (CARD-PROBE-1)
  admits or kills (vii) by the KL line, the proof lane by the proof-rate line, size and reuse by the gap
  line; LR and EMA are decided by the EMA cell and the drift reading (R365(d)); run10 may carry every
  probe-supported change together and at most one un-probed hypothesis (CARD-RUN10-RULE). **R366(b),
  2026-09-21: (vii) RIDES run10** (`identity.arch_kind: GnnArchV2SoftPolicy`, `model.aux_soft_policy` T 4
  / weight 4 in [2, 8]; the target's temperature on the EXPLICIT entries with the tail carried, on a
  measurement — the whole-set form put 40 % of the soft mass on a ~1e-4 tail); **(i) rides from run9's
  mint** with the value-gap witness beside it; **(ii) IS run10's one hypothesis** as `scheduler_t_max`
  108 000 / `eta_min` 1e-4 (cosine 1e-3 → 1e-4 over the block, not the queue's 2e-4 → 2e-5); (iv) prior
  temperature, (v) sims and (vi) policy-surprise weighting STAY QUEUED; (iii) is dead (F-53).
- **PARKED by R358(e), one line each, not resurrected without a new measurement:** aux targets
  (ownership/score are Go quantities, no graph-native source, strix's q_head unmeasured); net size
  (strix is 6 % smaller and wins); curriculum (no ablation in any source, strix's r2 stage
  degenerate); opponent diversity (one asymmetric-game cumulative ablation); teacher signal (no
  measured later cost anywhere; the goal call is the operator's).
## Opened by R355 (REPAIR-A4)

- **CARD-STYLE-BACKLOG — the judgment half of the R346(f) census, held by the ratchet.** The
  mechanical half landed 2026-09-17 (63 stale R8 headers, nine three-line runs, the labelled
  separator rules, a cite-only line, a file-top banner). What is left is a POLICY call and
  site-by-site work, measured at HEAD by `comment_lint.py --measure` against the gated floors
  named in `tools/ci_gates/comment_length_floor.txt` (that file holds the current values, never
  transcribed here): `comment_excess_lines`, `banner_comment_lines`, `docstring_excess_lines`,
  `private_docstring_excess_lines` (R346(f) names PUBLIC APIs; whether a private symbol may carry
  a multi-line docstring is the architect's call — the measure exists so the answer can be driven,
  never presumed), `rust_doc_excess_lines`, `ruling_cite_lines` (R368(g); supersedes the retired
  `ruling_cite_comment_lines`), and `textfile_comment_excess_lines`. The parenthetical ruling cites
  in comments and docstrings were REVIEWED and kept: each attaches a ruling or law to the fact it
  grounds (a pool size, a refused default, a fatal latch), which is the provenance class R346(f)/
  R368(g) carve out — a strip would delete where a number came from. Also on contact, never as a
  pass: the F1 defer path (`declared_keys`/`declared_lr` no production caller passes, B-14 — its
  flat `RESUME_CHECKPOINT_OWNED_KEYS` is a golden-pinned contract row); `collate_graph_batch(device=None)`,
  the two segment softmaxes, the double `torch.load` (B-15). Every measure may fall and may never
  rise; the floor file is the record of progress.

## Opened by R352 (run7 kind, strix rung, viewer)

- **CARD-GUMBEL-HEAD-RESIDUE — which part of the Gumbel deploy head loses to the most-visited
  child at every σ (F-51's residue, R352(b)).** All three σ pairs read the 18k net within 7 pp of
  each other at 128 sims and within 2 pp at 512, halving per doubling, against PUCT's 0.774. The
  candidates in order of cheapness to test: `gumbel_m` 16 over a ≈ 355-move legal set (a 16-sample
  Gumbel top-k on a flat prior drops the best move with probability the paper's Go runs never
  paid); the sequential-halving schedule at 512 sims (how many rounds survive, what each finalist
  gets); the interior (non-root) selector; or a defect the Mctx parity fixtures do not reach (the
  pins cover completed-Q and the root pick on toy trees, not a 355-move root). One cell each, the
  18k net, 288 paired games, the same rung; the card is NOT a run7 question — as a TRAINER the
  head works. CARDED.

- **CARD-EVAL-ROUND-OVERRUN — the eval round outlasted its cadence; the overrun itself is FIXED
  (resume mint `ce0a8ff6`: gate and rung at 256, `eval.max_plies` its own row, the band 0.5, the
  GSPRT). Still OPEN, from the same review:** the eval child emits no `batch_timing_snapshot()`
  (LAW-18; needed before concurrency is touched), `book_v2` (CLOSED by R384(c): the "+11 pp of
  gate power" was the 2q(1−q) chance floor, so there is no power case), and the ruling that names "gate pair statistics" for the GSPRT.

## Opened by R350 (the block verdict)

- **CARD-STOP-DRAIN-VS-GRACE — a stop during an eval round is a SIGKILL after the save.**
  Repaired 2026-09-26: legs 1–2 landed at `49b7740d` (`abandon_pending`), so the headline no longer
  holds for a resumable stop; leg 3 (a mint relation between grace and teardown) is still open.
  Witnessed at run6's stop (2026-09-13 08:15:12 UTC, one SIGTERM to the supervisor): `shutdown_save`
  at +0.5 s, `resume_state_persisted` (the 35 084 bundle, complete) and `flush_pending_eval` at
  +1.4 s — then 30 s of `game_complete` events while `close_out` waited on the in-flight round-35
  eval child, and the supervisor's `monitor.supervisor_kill_grace_sec 30` SIGKILLed the run at
  +30 s. The drain's bound is `min(final_eval_drain_timeout_sec × safety, hard_cap)` = 2 700 s
  from the schema-default `monitor.drain` block run6 does not mint, so on any stop that lands
  inside a round (≈ 27 % of wall at run6's cadence) the teardown ladder past the drain — pool stop,
  the recorder's `stop()` that indexes the open shard — never runs. Cost this time: the hour's
  two open game shards (`games_run6_seg0001_2026091308.jsonl`, `games_run6_seg0036_2026091308.jsonl`,
  427 KB) carry no `shard_closed` row and are therefore unreceipted by the puller; the bytes are
  on disk and mirrored (line-buffered writes, `iter_run_games` scans shards). LAW-16's save is
  intact; the exit is not orderly. Fix shape: a RESUMABLE stop terminates the in-flight round at
  once (a resumed run re-kicks the boundary round its sidecar says was never kicked — the
  `eval_round_last_step` field, B-3/R355(e), 2026-09-16; the round's result is not owed)
  and closes the recorder before anything that can wait; and the supervisor's grace must exceed
  the child's worst orderly teardown, stated as a relation at the mint rather than two numbers.
- **CARD-DRAIN-POLLER-RACE — `drain_pending` can return `None` while the poller finalises the
  same round.** `EvalPipeline._finalize_round`'s once-only guard returns `inflight["_result"]`
  to the second caller, which is `None` while the first (the poller thread) is still between
  latching `_finalized` and storing the result — so a `close_out` drain that loses the race by a
  few milliseconds proceeds with no result routed, and the poller's mailbox copy is never read
  after the run stops (a promotion decided in the run's last round could go unapplied). Seen as
  a 1-in-~5 flake of `tests/eval/test_eval_broken.py::test_killed_worker_yields_eval_broken_and_clean_drain`
  on 2026-09-13 (the fake process is flipped dead just before the drain). Fix shape: the second
  finaliser WAITS (bounded by the kill grace) for `_result` rather than returning `None`.
- **CARD-WARMSTART-CONTROL — the R340 control's head set, read from the tree. CLOSED 2026-09-26 (R370(a), FINISH sweep) — DONE: the bc_tp/bc_full pair is read in STRENGTH_FRONTIER_1 §D; `identity.warm_start.reinit` landed at `bd94dd96`.** R350(a) states
  the burst copied ALL heads; `run6-mint` at `d3ba75e` carries the same trunk+policy seam run6
  booted with (the burst's log died with the box, archive v3.54). The frontier measures the head
  set as its own cell pair (`bc_tp` vs `bc_full`); the card closes on that reading.

## Opened by R349 (the START path)

Records: `docs/design/measurements/MEASUREMENT_STARTPATH_2026-09-11.md`; falsified.md F-44/F-45.

- **CARD-SHAKEDOWN-TIMEOUT-STOP — LAW-16's save did NOT fire under the R340 launcher's
  `timeout`; it DOES fire under a direct SIGTERM (the supervisor's path). OPEN, not a START hold.**
  The 4 h shakedown (2026-09-11 21:56 → 01:56 UTC, `/workspace/runs/shakedown`) ended by the
  launcher's `timeout 14400` and reported rc 124 "the success path" — but the run's events end
  2 s before the deadline with no `shutdown_requested` log line, no `shutdown_save`, no final
  bundle (the step-6000 periodic bundle stands); the process died silently within 3 s. The
  falsifier run 8 min later on the same tree (`/workspace/runs/sigtest`, `/workspace/oc7/
  sigtest.log`): the twin launched bare, ONE `kill -TERM` at step 2 → `shutdown_requested`,
  `shutdown_save`, a complete bundle at step 3, exit in 3 s. A toy shows GNU `timeout` delivers
  one coalesced SIGTERM to a sleeping child; what differs for the real run under `timeout`
  (group kill + PDEATHSIG armed at SIGKILL + a busy main thread + the eval child in the group)
  is NOT root-caused. Consequences now: the launcher's rc-124 claim is FALSE unless the events
  carry `shutdown_save` — it must assert that; every `timeout`-wrapped stop on record (R340,
  R343 leg 3) is suspect the same way. Falsifier: a 5-min twin under `timeout` vs `timeout
  --foreground`, `strace -f -e trace=signal` on the child. START is unaffected: the block stops
  through the supervisor (one SIGTERM, witnessed) and its bundles every 1 000 steps bound a loss.
- **CARD-TRAINER-CADENCE — the architect's.** Steps/h ≡ games/h by `train.training_steps_per_game
  1.0` / `max_train_burst 1` (6.6 draws per row); the trainer is 92 % idle (run6's regime — F-44's annotation, R365(e)). The block is ≈ 22.6 h
  at 1,105 steps/h. Raising the ratio halves the wall clock and doubles sample reuse — a regime
  decision, not a lever. INVESTIGATION-1 item 2 is re-aimed: "is the reuse ratio right" and the
  GPU step's own 156 ms (`index_add_` 22 %, GEMMs 16 %, 31 syncs) — not "why is the step 5 s".
- **CARD-SERVER-SYNC — INVESTIGATION-1 item 3, now the block's price.** The performance
  investigation (`docs/design/measurements/PERF_INVESTIGATION_2026-09-11.md`) measured the
  inference-server thread **90.6 % busy at 16 workers, 99.0 % at 32** (F-47): per 20.7-ms pop of
  34.9 leaves, 8–10 ms spinning in `cudaStreamSynchronize` for an eager GINE forward at 6–9× its
  bandwidth floor, 3–4 ms in check 14 with the GIL held, 1.7 ms pageable H2D, 1.8 ms of 216 eager
  launches, 3 ms of contract work, 1.9 ms idle. Its ranked levers, each a LAW-09 proposal with a
  falsifier: a two-stage pipeline (collate pop N+1 under pop N's forward, +40–60 % pre-reg),
  check 14 with the GIL released in Rust (+17–23 %), `torch.compile` of the trunk (−31 % serve
  wall measured in the microbench, +16 %), an edge-attribute codebook, pinned + `non_blocking`
  H2D, a fused message-pass kernel. `n_workers 32` measured +5 % alone (a mint row).
  PERF-A4 (branch `perf-a4`, `docs/design/measurements/PERF_A4_2026-09-11.md`) landed four of
  them as one commit each with a box arm apiece: the GIL-released verifier (+17 %), pinned
  H2D + sync removals (0, the prerequisite), `inference.compile_trunk` (+21 % serial), and a
  two-thread software pipeline (+52 % over serial) — 1,831 → 3,662 leaves/s at 32 workers
  (2.0×), the block ≈ 10.6–11.1 h ESTIMATE against 20.5–21.5 h; the codebook is bit-exact on the
  RTX 5080 only and is deferred with its numbers; the mint rows are in that record's §11.
- **CARD-EVAL-CONTENTION — a measured term nobody had priced.** While an eval round is alive
  self-play runs at **0.79×** (arm `w16ev`: −20.7 % leaves/s over a 12-min round; two CUDA
  contexts time-slicing plus the child's 8 CPU threads, the child allocating only 38 MB). At run6's
  cadence that is ≈ 5–10 % of the block's wall (ESTIMATE from 12–22-min rounds every ≈ 54 min).
  Levers: batch the child's inference across its 8 games, or `eval.worker_device: cpu` (a config
  row, longer rounds). Only one 12-min round was crossed; a fully escalated one is not separated.
  **RE-MEASURED on the block (25 rounds, 5.9 h alive of 21.8 h): self-play at 0.94× during a
  round at 32 workers behind the PERF-A4 pipeline — ≈ 440 steps, 1.7 % of the block, ≈ 22 min.
  The term is priced and small; lever (a) stays a proposal.**
- **CARD-DEPLOY-HEAD-BUDGET — item 4.** In decided positions the deploy head spends 28–40 of a
  64-sim budget and 52–77 of 320: `gumbel_root_select` returns `None` early. The eval instrument
  under-spends exactly where the position is settled; whether that moves a bar is unmeasured.
  SUSPENDED under LAW-02 by R355(b) (2026-09-16): the numbers were read through the defective
  Gumbel driver (A-1, the first-transposition stop); re-read from R355(c)'s cells.

## Opened by R348 (WAVE 3, leg 1)

Found by running things the dev box could not run — the integration tier and the Gumbel regime at
scale on the box. Records: `docs/design/measurements/MEASUREMENT_OC7_2026-09-11.md`,
`docs/design/archive/measurements/MEASUREMENT_PERF3B_2026-09-11.md` (archived 2026-09-17).

- **CARD-TIER-HOST residue — TEST-1's minted CUDA smoke profile is OWED.** The AVX2-host carve-out
  itself is landed (LAW-06's fp32-on-`train.device: cpu` amendment, `9491b4d0`, its own parity
  test); the fourth option named at the time, a minted CUDA smoke profile, is still owed.
- **CARD-ALPHA-MAX-ROWS — DISCRIMINATED under R349(c); the run-fatal half FIXED, the target
  half OWED to the architect.** 25 rows reconstructed (`MEASUREMENT_STARTPATH_2026-09-11.md`
  §B): all at `moves_remaining == 1`, all lost positions, the perspective flip CORRECT (F-45).
  Mechanism: value-saturation asymmetry lifts `v_mix` above the clustered visited Q's and
  Mctx's min-max rescale × `c_scale 1.0` underflows the explicit masses. The recorder dropped
  zero-mass sampled cells, so an all-underflow row was refused at the ring as `EmptyTarget` and
  killed the burst at step 496 — fixed at `f253e65e` with parity vectors on both sides of the
  FFI. **OWED (D2):** whether ~1 per 1,000 decided-position rows should train "not here" —
  the paper's σ without min-max, `c_scale 0.1`, or a span floor — each a regime change needing
  the PERF-3b re-measure. The per-1,000 count rides `iteration_complete.gumbel_alpha_full` and
  the dashboard from step 0.

## Open cards whose in-source markers are gone

Both were in-source markers until the 2026-09-11 style pass (`2649e0b9`) stripped the marker text.
The debt did not move with the text, so both are carried here as ordinary rows.

- **CARD-BUDGET-AUTHORITY-CONSOLIDATION — CARDED as debt by R327(d): two `_SIZING_BUDGET_*`
  authorities for one nominal quantity.** `tests/train/test_graph_microbatch_authority.py`'s
  `_SIZING_BUDGET_BYTES` bounds the sizing frontier's MODELLED per-micro-batch peak;
  `tests/train/test_graph_microbatch_bound.py`'s `_SIZING_BUDGET_GIB` bounds a MEASURED
  `max_memory_allocated` delta plus an allowance, and was re-sized by R326(b) and R338(c) while the
  first did not move. R327(d) ruled the first STAYS UNMOVED: moving it to the sibling's value reds the
  shipped, measured-green pair and demands a cap re-fit, which is circular. The debt closes only by
  RE-FITTING the frontier on current code, never by moving that constant.
- **CARD-CONFIG-DISCOVERY-ROOT — CARDED at ADJ-13's close (R71, R72; `4d11147d`): discovery binds
  `configs/` and nothing outside it.** `mantis.config.loader.discover_configs` sees every loadable
  file under `configs/`, but a loadable config in a scratch directory is still launchable by
  `python -m mantis.run --config` and is never discovered by gates 7 and 12. Deliberate:
  preflighting a candidate from a scratch directory is the normal case, and every `--config` route
  depends on it. The mint path is covered instead: `preflight_mint.py --config <path>` unions any
  named path into the audit set (`_audit_paths`).

## Reading the identifiers

Four traps, each of which has already misled a reader:

1. **`F-<number>` is five namespaces, not one.** The graves in `docs/governance/falsified.md`
   (derive the last row from the file, never transcribe it) · AUDIT-1's 52 findings · the session ledger's · a perf-ledger `F-10`
   corrected by R320 · R246's cross-language-parity `F-01`/`F-02`. **Three distinct `F-10`s
   exist.** Unpadded `F-1`..`F-6` are ADJ-13 / RED-TEAM findings closed as a class by R71/R72 and
   never mean `F-01`..`F-06`. Every `F-NN` here names its register. Graves are never cards.
2. **`CARD-LEVEL` is not an identifier.** Every hit is the phrase "CARD-LEVEL FACT" — a fact about
   the GPU card. There is no such card.
3. **`CARD-ANCHOR-WIRING` was DECLINED** — R55 ruled it "is therefore not created". It is not a card.
4. **Two ADJ number-spaces collide** (WPUF/WPAX-era against WP12-R-era) at ADJ-19 through ADJ-26
   and ADJ-29. Qualify by era or by the enclosing ruling.

## What held run6's START

run6 RAN its block from these holds (`RUN6_BLOCK_2026-09-12.md`) and was STOPPED by R350(a) at
35 084 steps; every hold below is DISCHARGED, kept as the record of what held its START.

- **REPAIR-A2 leg 5 — the MCTS root child cap. DISCHARGED.** Landed at `b9f5f509` (R347(c)):
  `MAX_CHILDREN_PER_NODE` moved from 192 to 1024 (crates/mantis-search/src/mcts/mod.rs), a 4M-node
  pool with the ceilings re-derived to 976/960. The 192 measurement that drove the move (a mean 88%
  of the policy's prior mass discarded on 99.97% of expansions) is superseded by the landed value.
- **`F-816-24` — CLOSED by R349(a).** Read at contact 2026-09-11 (R348(e)'s AUDIT-3 lead):
  `supervise.main` loads the minted config and resolves every `monitor.supervisor_*` through
  `resolve_monitor_config` since `c8bd7190` (2026-08-21), with a named refusal for a missing
  `--config`; `tests/monitor/test_supervisor_config_witness.py` carries the running-supervisor
  witnesses (integration) and an AST pin that the module constructs no `MonitorConfig` by any
  shape (default tier). R291(b).
- **The `supervisor_kill_grace_sec` reading — SETTLED at `a37d2e5e`.** The operator read it the
  other way: the re-mint reverts `600.0` to the template's `30.0` (R348(e), merged under R349).

Riding the run rather than holding it: **`F-816-37`**, below.

**`R341(b)` / `R319(d)` — DISCHARGED AT G=8 ONLY (R343(a)); still LIVE below G=8.** R341(b) withdrew R340(a)'s closure
because the round that must finish is the CONTENDED one. R341(c)'s G table then ran and G=8 was
armed on the operator's forward: **G=1 at 53.33 s/game and G=4 at 13.99 s/game both consume the
full 3600 s `round_timeout_sec` and return `wr_sealbot: null`; G=8 at 7.09 s/game completed a
fully escalated 264-game round in 1872.8 s, 78.8% of the 2376 s bar, with a real `wr_sealbot`**,
and the 4 h shakedown held a steady 900.6 s wall with no growth over seven completed rounds, zero
nulls, two promotions. run6 minted `eval.concurrency = 8`. **The row is discharged by the ARMED
VALUE, not by the geometry becoming safe:** lowering concurrency to 4 walks straight back into
the timeout (G=5–7 unmeasured), and a null round is not a slow reading. run10 arms `eval.concurrency: 8`; a re-mint
below 8 re-opens this row. R343(a).

## F-816-* findings

| id | subject | status | last moved |
|---|---|---|---|
| F-816-37 | run-fatal `EdgeAttrGeometryMismatch` at run6's minted geometry, not root-caused | OPEN. Converted into a 1-in-1 eval-path instrument with dump-on-fire (protected set); zero shakedown firings is explicitly NOT a close. Every firing on record is on the host R341 condemned and R342 downgraded to SUSPECT, and the work moved to a different box — the halt is spent, the class is not | R342(a) |
| F-816-27 | supervisor kill-grace CEILING absent (schema is `Field(ge=0)` only) | RULED; rides prereg row 19 to the operator | R338(c) |
| F-816-34 | vacuous knee band | RULED by R338(a)/(b): the halt ratified, and the widening's noise term corrected by mechanism to the noise-floor drive's rel-SE, never a within-round spread, the widening VOID for a ladder it still swallows. Filed 2026-09-04: PICK = 2 from `adjusted_threshold` 54.9167, widened below every rung's throughput (min passing 89.600; unwidened the rule picks 16); a pre-statable vacuity test is `adjusted_threshold < min passing throughput` | R338(a)/(b) |
| F-816-35 | r8 trainer need is a DISTRIBUTION | RULED by R338(c): the allowance is taken over the MAX (8.6381 GiB), not p95; `test_graph_microbatch_bound.py`'s `_SIZING_BUDGET_GIB` carries the re-sized value. Measured p50 7.9072 / p95 8.3581 / max 8.6381 GiB over 60 steps, exceeding both the minted `_SIZING_BUDGET_GIB = 8.40` and R330(b)'s armed 3% over FINISH-1's single-draw point (8.3341); not a halt alone — the peak is cap-bound and STEP 3 re-fits the caps — but which statistic the allowance is taken over is worth 0.75 GiB | R338(c) |
| F-816-36 | an unplayable rung sets every ring's composed visit capacity | CLOSED 2026-09-26 (R370(a), FINISH sweep) — GONE: `strix_256` and `eval.ladder` were deleted by R362(c) at `94b8286a`; `derived_visit_capacity` composes only from self-play sims | none |
| F-816-15 | `freeze_verify.py` red on 39 of 64 paths; audit-before-rebaseline | ORDERED as its own packet, never dispatched | R285(g)/R286(c) |
| F-816-19 | the run's own process is spawned unparented (PDEATHSIG class) | ORDERED PRE-MINT, no close | R285(h) |
| F-816-21 | test de-triplication — one stub in three files across two registers | RE-SEQUENCED behind RQ-1; owed inside the freeze packet | R288(d) |
| F-816-26 | parent/child config binding; a mismatch is a NAMED REFUSAL | RULED, queued behind Q3/Q4 | R306(d) |
| F-816-28 | preserve BOTH invariants or the primitive does not move | RULED BY PRINCIPLE, queued behind Q3/Q4 | R306(d) |
| F-816-30 | a skip guard must detect the MECHANISM, never a proxy | RULED; carried by PACKET_CI_RUNTIME, which forwards first | R306(d) |
| F-816-11 | arena/eval ply cap as an unconfigurable literal | CLOSED 2026-09-26 (R370(a), FINISH sweep) — DONE: `eval.max_plies` is a minted row (`79d832cd`) and `arena/match.py` takes it as a required argument | R338(d) |
| F-816-14 | the eval child survives its parent's SIGTERM holding 458 MiB | HALF-OPEN — the SIGKILL leg closed, the SIGTERM leg re-worded as F-Q6-8 | R300(d) |
| F-816-17 | dead `legal_mask` build | routing RATIFIED AS FILED, no close | R286(f) |
| F-816-1 | run5 death was a host event with no software error line | no close ever recorded | R268 |
| F-816-2 | independent card riding the VisitSlotsExceeded packet | CARDED, no close recorded | R274(e) |
| F-816-4 | thread-leak / process-global-state hazard class | no status ever recorded | R289(s) |
| F-816-6 | degenerate ply-cap flood, draw_rate 1.000 at bootstrap | MINT-CRITICAL headline, no close recorded | R269 |
| F-816-8 | `wppre-scratch` branch containment ground | no status recorded | R277(c) |
| F-R302-1 | trainer-forward OOM; allocator-reservation fragmentation | EXPLAINED, NOT CLOSED — "closes at a standing mint" | R315(a) |
| F-B1 | parent/child same-file config binding (`config_identity_sha256`) | LIVE as DESIGN input to the F-816-24 packet | R292(c) |
| F-Q6-1 | the flamegraph tool's own 12.4 GiB orphan (PDEATHSIG family, instrument side) | routed to the carry-over queue, "live until their rows close", not seen since | R300(d) |
| F-Q6-8 | save-then-exit did not hold under OOM (LAW-16) — the re-worded F-816-14 SIGTERM leg | OPEN; the close-out measurement is filed beside it | R302(d) |

## Named work items

| item | subject | status | last moved |
|---|---|---|---|
| DASH-2 | `make dash` (`tools/dash.py`, package `tools/dash/`), a read-only stdlib HTTP server over run records, loopback by default | LANDED 2026-10-05 (`293b3277..83a5c1e6`): Run, Games and Analyzer views over a run directory or its mirror, the run monitor's save records, the cell sidecars (every ruler its own series over its own parent; `--rule-unit`, `--ladder`) and stamped checkpoints, plus a one-file freeze of the Run view. `tools/dashboard`, `tools/viewer` and `tools/analyzer` are retired; the analyzer's engine layer lives on in `tools/dash/engine`. The R9 amendment is at the foot of repo_design.md, admitted docs-only by R386(a); box A's display is admitted read-only at the lowest priority, no GPU. Two fresh reviews' findings fixed (reports outside the tree). | R344(d) |
| RUNG-2 | new external rungs — strix first, shrimp second | strix LANDED (the frontier's cell, the `tools/strix_follower.py` equal-work 256/256 cell on every 15 000-step checkpoint (the per-promotion trigger withdrawn, R361(a))). shrimp HELD for an architect read on the R257 radius fence | R356(a), R359(e), R361(a) |
| INCR-GRAPH / S-INCR-GRAPH | incremental axis-graph construction from the parent position | PARKED, after being elevated to the top of the floor lane at R325. A CANDIDATE, not a plan: gated on a Rust-criterion box measurement, falsifier pre-registered as F-19's own inequality (`delta_cost x depth < build_cost`). Outside F-17/F-19's measured scope — see `docs/governance/falsified.md` | R335(e) |
| HOT-14 | cross-core ownership explains x1.66 of x6.84 | RE-OPENED when S-PREFUSE was refuted | R336(a) |
| S-BATTERY-G | eval battery concurrency capability | landed UNARMED; the CUDA arm is OWED at the mint's battery | R336(a) |
| S-CHECK17 | trainer-step check-17 bar | bar MISSED and BANKED at x1.18 — banked, not tuned toward | R336(a) |
| PERF-TRANCHE-1 residual | the 7.2% pre-control/ledger disagreement | OPEN as instrument hygiene; ledger absolute levels are not quotable without re-measurement | R320 |
| PERF-TRANCHE-2 | six items T2-1..T2-6 | EXECUTED — its findings are cited as landed evidence by R335 — but NO ratifying clause exists in either archive file | R334(e) |
| WP-AXIS2 | Phase 2 axis-graph arch, then a shakedown | LAST ORDERED, NEVER CONFIRMED. Neither the shakedown nor the R339 mint is ever labelled WP-AXIS2, so completion would be an inference, not a record | R335(g) |
| DASH-1 banked panels | average sims/move, held-out loss | 2 BANKED; drawn as stated gaps, never as zeros. Repaired 2026-09-26: held-out loss now HAS a producer (the `heldout_gap` event, event_manifest.md) — its panel is unwired, not producerless | R334(a) |
| R317(c)(ii) diagnostic | move-sequence-hash diagnostic | accepted as NON-BLOCKING DEBT, never shipped | R318 |
| AUDIT-1 P10 | lane-C design input, explicitly "not a packet" | still the architect's, undispatched | R331(d) |

## Owed texts and values

- **R227, R228, R267 — TEXTS OWED, operator residue.** ADJ-D2 covers R227/R228 and directs that
  they are NOT filled agent-side; it is load-bearing because it discharges R56/R133/R138. R267 has
  no section at all — the only record is a STATE digest line, deliberately not reconstructed
  because a digest line is not the ruling.
- **R226 / R229 / R243 — prereg rows owed:** two flagged at dispatch 8C, three 8B findings.
- **R349(c) / R350(e) — the three-row α = 1.0 reconstruction is OWED with its finding.** Three
  rows from the game record: `v_mix` vs max visited Q, which stone of the turn, the perspective
  sign at the root; a perspective error at the intermediate stone is the first hypothesis. The
  START-path measurement (§B, F-45) read 25 rows from a burst; the block's rows are not yet read.

## CARD-* named in governance

| card | subject | status |
|---|---|---|
| CARD-RUN5-GPU-OOM | GPU-OOM defect CLASS; the site set now includes the GNN training forward | OPEN as a class. Instance F-816-12 closed at the joint mint; the class row never closed. ANNOTATION 4 / R302(c) rider |
| CARD-CLEANSTOP-SAVE leg (b) | the `checkpoint_interval` prereg row (leg (a) discharged) | CLOSED 2026-09-26 (R370(a), FINISH sweep) — GONE: its RUN5_MINT_PREREG row has no document to land in, and the one production config mints `checkpoint_interval: 3000` |
| CARD-RESUME-LAUNCHER-FLAG | supervisor auto-resume / `--resume-from` launcher surface | BUILT IN CODE: `python -m mantis.run --resume-from <bundle>` is the ONE optional launcher flag (`resolve_bootstrap` fails a stale one at launch); run7 resumed through it on 2026-09-15. Supervisor AUTO-resume is not built; it was asked for by R343(c) (carded at `8d69cf69`), so it is not "not ordered" (repaired 2026-09-26); the row closes when a ruling names it |
| CARD-EVAL-CHANNEL-SPLIT | split the promotion and external eval cadences | OPEN, narrowed. R343(b)(v)'s conditional FIRED; R345(c) moved `gate.stride` to 3 and ledgered "the split that was already a key"; superseded for run7 — the gate runs at stride 1 on the 3 000-step cadence (R350(d), CARD-EVAL-CADENCE) |
| CARD-PROTOCOL-COMPLETE | complete protocol declarations, widen the AST conformance gate, LAW-16 sink/watchdog row | OPEN, pre-cutover, NOT mint-blocking |
| CARD-DENSE-EVAL-ADAPTER | wire `infer_batch_per_cluster` into the deploy-head decode | CLOSED 2026-09-26 (R370(a), FINISH sweep) — GONE: the function went with the grid path at `3dd20b49`; its control arm `v6_live2_ls` is pinned ABSENT in registry_census.rs |
| CARD-LINT-TYPE | ruff/pyright advisory type-debt backlog | OPEN debt row, deliberately kept out of the gate by R98 |
| CARD-PYRIGHT-STRICT | pyright strict-mode adoption as a post-cutover ratchet | OPEN; live marker in pyproject.toml's `[tool.pyright]` comment block |
| CARD-TORCH-INDEX | conditional torch index / uv extra for the CPU-wheel parity regime | CLOSED 2026-09-26 (R370(a), FINISH sweep) — DONE at `669b6008`: the `cuda` extra on pytorch-cu128, the `cpu` default group, `make build.cuda` |
| CARD-EVAL-CORESIDENCY | characterize eval-child steady VRAM for the co-residency prereg row | OPEN. The founding 8.21 GiB figure was superseded by R229(1) (unbounded, to 13.5 GiB) without naming the card |
| CARD-A10-CAP | whether an entropy term enters the graph loop at all | RECORDED, explicitly NOT executed. R335(b) makes entropy normalization a PRECONDITION on ever arming one |
| CARD-SEALBOT-BRANCHES | evaluate ramora0 branches (nnue) as a higher ladder rung | DEFERRED, not mint-relevant |
| CARD-MINPIN | the K-cluster min/max asymmetry, pending the matched-FLOP dense arm | CLOSED 2026-09-26 (R370(a), FINISH sweep) — GONE: no K-cluster aggregation exists (both registry rows `value_pool = "none"`, `is_multi_window = false`); the multi-window rows and the dense arm it waited on left with the grid path at `3dd20b49` |
| CARD-CHECK14-EDGE-GEOMETRY | the `verify_edge_geometry` collate check ("check 14", NOT CI gate 14) | NO STATUS EVER RULED. R336(e) separately CARDS check 14's 41.4 ms/part, not ordered |
| CARD-FRESHSYNC | gate 1 fresh-clone sync broken since WP7 | OPEN in governance, REPAIRED IN CODE — the gate pins `registry_sha_hex()` and describes the failure in the past tense |

## CARD-* that exist only as in-source markers

Zero governance mentions; their status comes from the code, not from a ruling. Derived by
`git grep -hoE "CARD-[A-Z0-9.-]+"` over `src`, `tests`, `tools` and `crates`, less what this file
already names.

- `CARD-ABORT-EXIT` — `heartbeat.DRAW_RATE_COLLAPSE_EXIT_CODE`'s cooperative (not OS) exit delivery,
  discharged by R84. `src/mantis/config/armed_aborts.py`
- `CARD-COORD-KNOBS` — a follow-up owed to the operator at run5's mint prereg.
  `src/mantis/config/armed_aborts.py`
- `CARD-CS2` — both step tails call the ONE periodic-checkpoint resolver, closing the graph-arm
  gap WP12-R found. `tests/test_run_launcher.py`
- `CARD-LINT-GATE` — the curated lint/type gate itself (R98). `tools/ci_gates/lint_gate.sh`
- `CARD-MINT-RESOLVE-PARENT-CONJUNCT` — the `max(learners) >= 1` mint-resolve conjunct.
  `tests/config/test_mint_and_diff.py`
- `CARD-ORPHAN-WORKERS` — SIGINT during self-play leaves zero descendant processes (R230), the
  second-signal path too. `tests/train/test_orphan_workers_census.py`
- `CARD-POOL-ENCODING-BRIDGE` — the pool/encoding bridge seam (WPBRIDGE Phase T, TD-4).
  `tests/selfplay/test_pool_encoding_bridge.py`
- `CARD-PREFLIGHT-CHILD-STDERR-BUDGET` — the preflight child's stderr tail is a bounded VIEW, not
  the full spool. `tests/tools/test_preflight_pfc_cards.py`
- `CARD-PREFLIGHT-OUTDIR-REUSE` — preflight out-dir reuse across phases.
  `tests/tools/test_preflight_pfc_cards.py`
- `CARD-RING-SAMPLER-SEED` — `seed_sampler` buys run-to-run reproducibility, not stop/resume
  continuity, by refusal rather than deferral. `crates/mantis-selfplay/tests/replay_sampler_seed.rs`
- `CARD-RUN-MAIN` — the six minted configs frozen at `b482243`, proving the WPMAIN re-mint
  ADDITIVE-only. `tests/fixtures/manifest.toml`
- `CARD-TRAINSTEP-ADAPTER` — a dormant contingency: TD-1 sits behind TD-4 on a CPU box and has
  never fired; the test pins its absence. `tests/tools/test_preflight_mint_process.py`

CLOSED, still cited from the tree. Their card rows left this file with R368(e)'s sweep; each line
names what closed it, so every live cite resolves here.

- `CARD-A4-MINT` — CLOSED 2026-09-11: the four PERF-A4 rows decided by matrix and minted,
  `0f20896e`. `tests/eval/test_deploy_matched_hparam_coincidence.py`
- `CARD-EVAL-GATE-FIELDS-IN-STREAM` — CLOSED 2026-09-19 by R362(c), `94b8286a`:
  `eval_round_complete.gate` carries the gate's rule fields.
  `tests/eval/test_gate_fields_ride_the_round_complete_row.py`
- `CARD-SEALBOT-TT-SEAT` — CLOSED 2026-09-15 on the A/B's reading, R353(b), `7e0a424c`
  (`SEALBOT_TT_AB_2026-09-14.md`); the rung itself was deleted by R362, and its dashboard row retired with the dashboard (DASH-2).
- `CARD-SELFPLAY-SEARCH-STATS` — LANDED 2026-09-16 by R355(d), `18eb4f4e`:
  `selfplay.search_stats_every` samples self-play games into the record. `tools/dash/views/games_text.py`
- `CARD-SERVER-OWNED-COPY` — LANDED 2026-09-21 by R366(b), `857187af`: the inference server serves
  its own copy, which `ActorSync` writes and the learner never reads. `src/mantis/train/checkpoints.py`
- `CARD-STAMP-FLOOR` — CLOSED 2026-09-11, decided by matrix, `652b9f02`: the preflight burst is a
  stop-step bound over the minted config. `tools/ci_gates/preflight_mint_parent.py`

## RQ-*

- **RQ-5** — an independent cross-model review of the `freeze_verify` mission diff. ORDERED
  findings-only, unexecuted. R289(c).
- **RQ-7** — `supervisor_kill_grace_sec`. SPLIT; the VALUE is a prereg row and is operator-owed.
- **RQ-16** — dead-transfer field removals (`node_coords` plus four TEST-ONLY LAW-08 rows).
  DISPOSITIONED PER FIELD, one commit per field, execution pending on the hygiene packet.
- **RQ-18** — the compiled-arm parity criterion, four legs with `k` pre-registered. Criterion
  RULED; the MEASUREMENT is open and is an architect/box item.
- **RQ-21** — `freeze_verify` becomes CI GATE 18. MINTED and RULED; **the wiring is OPEN — no
  gate-18 script exists in `tools/ci_gates/`.**
- **RQ-22** — whether freeze row 30 should be frozen at all. MINTED; evaluation open.
- **RQ-2, RQ-14** — UNDETERMINED. Their dispositions live in an off-repo batch document; RQ-2
  appears only inside the string "RQ-2..19" and RQ-14 has zero occurrences.

## ADJ-*

- **ADJ-D2** — the missing R227/R228 texts. OWED, operator, do NOT fill agent-side.
- **ADJ-12** — no disposition anywhere; cited only as "the ADJ-12/13 lesson made law".
- **ADJ-WP12R-18** — the requeue of ADJ-WP12R-11's contradicting oracle evidence. NO RECORDED
  RESOLUTION; exactly one occurrence in the whole corpus.

Every other ADJ row resolves to a numbered ruling.

## AUDIT-1 and session-ledger findings still open

- **F-39 (AUDIT-1)** — 34 bridge-signature defaults shadow config keys. REGISTERED, not banked:
  what shipped is an enumerated `REGISTERED_DEBT` that reds in both directions. R333(a).
- **F-11 (AUDIT-1)** — disk-guard arming, shape A plus `poll_once` age. Armed and ordered to LAND
  BEFORE THE MINT; run6 was minted at R339 and **neither archive file records that it landed.**
  This needs a reading against the tree before anyone treats it as done. R334(b).
- **F-51 (AUDIT-1)** — supplies the candidate mechanism per tranche item (HOT-04 / HOT-11 /
  HOT-14). Cited as evidence only; no clause ever disposes it. R334(e).
- **F-42 (AUDIT-1)** — the one-owner axis table. BANKED at the REPAIR-2 exit, bank accepted, no
  movement since. R333(a).
- **F-10 / F-10b (session ledger)** — a box run wedged at step 12. OPEN, but SCOPED AWAY from run6:
  the wedge is on `gnn_axis_v1` and run6 mints `gnn_axis_r8`. No ruling has closed it.

## Q-C*

**None open.** Q-C0 through Q-C10 are all ruled or closed, and Q-C6/Q-C7/Q-C8 never existed. A
stale line once read "Q-C1..Q-C4 are OPEN at the queue foot" — superseded by R304. It is recorded
here so nobody re-opens four closed questions from it.

## How this list was derived, and what it cannot cover

Built at R346 from the archive's live-marked rows, its curation log, the ruling entries in
`docs/governance/RULINGS.md`, and an in-source sweep for `CARD-` markers, then checked against the
tree for every value it names.

Two limits, stated rather than hidden:

1. **The archive's live-force section was never extended past R337.** R338, R339, R340, R344 and
   R345 have no row there, and R341(e) and R343(f) still sit in it marked LIVE carrying terms (a
   750-step cadence, a 12 h block) that R343(b) and R344 have already superseded. A card list built
   by walking that section alone would miss the newest holds and carry two dead ones. Everything
   above from R338 onward came from the ruling entries instead.
2. **Read the whole row, not its headline.** A row's bolded verdict is where it STARTED. R341(b)'s
   headline still reads "LIVE, MINT-BLOCKING, and HOST-INDEPENDENT"; its discharge is nine lines
   below, in the same row. This file got that row wrong once by stopping at the bold, and so did
   the ruling that closed it prematurely before that. The discharge, the scope and the superseding
   annotation all live at the FOOT of a row, never in its title.
3. **AUDIT-2's own card list is still not enumerable here.** The analysis text is now filed at
   `docs/audits/archive/AUDIT_2026-09-09.md`, but R345's "everything else in AUDIT-2 is CARDED with its
   priority" refers to a card list that travelled in the packet, not in the audit document. An
   unknown number of AUDIT-2 cards therefore remain outside this file, and it should not be read as
   complete.
