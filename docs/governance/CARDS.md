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
  parity test (`tests/train/test_law06_cpu_carveout.py`); OC-7 now PASSES in 178.4 s inside its
  ceiling on that host. `make gates.exit` completes: the 2026-09-21 exit sweep read the
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

## Opened by R368 (SLIM-FIX; 2026-09-23)

- **CARD-RUST-HOTLOOP-EXPECTS — CARDED: the `unwrap()`/`expect()` sites on hot loops, classified by R368(g)'s
  correctness pass and left as they are, because a named-error rewrite there is a hot-path change that needs
  LAW-09's one-change-one-bench.** `mantis-core` `Board::check_win` (a cell lookup `apply_move` has just set);
  `mantis-selfplay` `queues/graph.rs` (13 lock/condvar-poison expects on the per-batch inference path, plus the
  infallible `position()` in `submit_graphs_and_wait`); `runner/search_drive.rs` `infer_and_expand_graph`'s
  `win_length`/`graph_radius` expects (the fix is hoisting both into the worker's inference context at start)
  and `select_move`'s guarded fallback `choose().unwrap()`. `mantis-search` has none left.
- **CARD-POISON-STANCE — OWED: one stance on a poisoned `Mutex` in the self-play runner.** Nine production
  `lock().expect(..)` sites (`runner/finalize.rs` ×2, `runner/mod.rs` latch / fatal read / `stop` / the two drain
  faces, `runner/spawn.rs`, `search_drive.rs`'s latch store) panic on a poisoned lock. `stop()` runs from `Drop`
  and cannot return a `Result`, and a latch must not itself fail. The measured recommendation (the W2 selfplay
  leg): latch, stop, finalize and spawn take `PoisonError::into_inner` (the panic that poisoned the lock is
  already counted by `worker_panics` and halts the run); the drain faces return a named error the bridge raises.
- **CARD-SEARCH-HOT-DUP — CARDED (R368(h)): the MCTS hot-path duplicates the census found (S-A-RUST-1-13, -14:
  shared selection/expansion helpers; -15: the `action_idx` decode re-typed inline).** Each lands only
  monomorphic (a shared fn, no kind flag), with LAW-09's bench and `search_kind_conformance.rs` as witness.
- **CARD-SCHEMA-KEY-RETIREMENT — CARDED (R368(h)): retiring a schema key and its minted rows (first:
  `train.value_target`, S-A-CORE-2-14).** A retirement needs a loader witness over every mirrored parent
  stamp before the key leaves the schema; not before run10 STARTs (R368(i)).
- **CARD-CLUSTER-THRESHOLD-RESIDUE — CARDED: `Board.cluster_threshold` is write-only since the cluster BFS
  went (R368 W2).** Plumbed from the registry (bridge `board.rs`, selfplay `game.rs`) into `BoardGeometry` and
  the golden-replay fixture's geometry, read by nothing; removal touches `BoardGeometry` and that fixture's field.
- **CARD-SEAM-2 — HELD (R368(j)): the seam for a kind that brings its own head, objective and config rows.**
  L-SEAM-01..04 fold into it; the design packet follows the SLIM-FIX phase; the implementation merges only
  after run10 STARTs.
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

## Opened by R367 (DESIGN STANDARD + REVIEW GATE; SIZE CONDITIONAL WITHDRAWN; PRICE LAW; 2026-09-21)

- **CARD-NET-EXPAND — run11's build: a FUNCTION-PRESERVING width/depth expansion of the trunk behind the
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
  constant bound to a different run's config (a name that lies about its file) in four test modules;
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
  ruling; the box work is a perf-host event, not run10's window.
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
  family the landed-UNARMED λ-return codec already is (`src/mantis/model/value_targets.py`,
  `f049ef02`, imported by nothing). ORDER after run8's 15k/30k reading: (i), (vii), (ii), (vi),
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
  (LAW-18; needed before concurrency is touched), `book_v2` (the balanced book is worth +11 pp of
  gate power), and the ruling that names "gate pair statistics" for the GSPRT.

## Opened by R350 (the block verdict)

- **CARD-STOP-DRAIN-VS-GRACE — a stop during an eval round is a SIGKILL after the save.**
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
- **CARD-WARMSTART-CONTROL — the R340 control's head set, read from the tree.** R350(a) states
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

**`R341(b)` / `R319(d)` — DISCHARGED AT G=8 (R343(a)).** R341(b) withdrew R340(a)'s closure
because the round that must finish is the CONTENDED one. R341(c)'s G table then ran and G=8 was
armed on the operator's forward: **G=1 at 53.33 s/game and G=4 at 13.99 s/game both consume the
full 3600 s `round_timeout_sec` and return `wr_sealbot: null`; G=8 at 7.09 s/game completed a
fully escalated 264-game round in 1872.8 s, 78.8% of the 2376 s bar, with a real `wr_sealbot`**,
and the 4 h shakedown held a steady 900.6 s wall with no growth over seven completed rounds, zero
nulls, two promotions. run6 minted `eval.concurrency = 8`. R343(a).

## F-816-* findings

| id | subject | status | last moved |
|---|---|---|---|
| F-816-37 | run-fatal `EdgeAttrGeometryMismatch` at run6's minted geometry, not root-caused | OPEN. Converted into a 1-in-1 eval-path instrument with dump-on-fire (protected set); zero shakedown firings is explicitly NOT a close. Every firing on record is on the host R341 condemned and R342 downgraded to SUSPECT, and the work moved to a different box — the halt is spent, the class is not | R342(a) |
| F-816-27 | supervisor kill-grace CEILING absent (schema is `Field(ge=0)` only) | RULED; rides prereg row 19 to the operator | R338(c) |
| F-816-34 | vacuous knee band | FILED 2026-09-04, never adjudicated. PICK = 2 from `adjusted_threshold` 54.9167, widened below every rung's throughput (min passing 89.600; unwidened the rule picks 16); a pre-statable vacuity test is `adjusted_threshold < min passing throughput` | none |
| F-816-35 | r8 trainer need is a DISTRIBUTION | FILED, never adjudicated. Measured p50 7.9072 / p95 8.3581 / max 8.6381 GiB over 60 steps, exceeding both the minted `_SIZING_BUDGET_GIB = 8.40` and R330(b)'s armed 3% over FINISH-1's single-draw point (8.3341); not a halt alone — the peak is cap-bound and STEP 3 re-fits the caps — but which statistic the allowance is taken over is worth 0.75 GiB | none |
| F-816-36 | an unplayable rung sets every ring's composed visit capacity | FILED, never adjudicated. The retained `strix_256` rung can never play a game (only sealbot is pinned), so it sets the composed `visit_capacity` of every ring the run writes and refuses the corpus fit | none |
| F-816-15 | `freeze_verify.py` red on 39 of 64 paths; audit-before-rebaseline | ORDERED as its own packet, never dispatched | R285(g)/R286(c) |
| F-816-19 | the run's own process is spawned unparented (PDEATHSIG class) | ORDERED PRE-MINT, no close | R285(h) |
| F-816-21 | test de-triplication — one stub in three files across two registers | RE-SEQUENCED behind RQ-1; owed inside the freeze packet | R288(d) |
| F-816-26 | parent/child config binding; a mismatch is a NAMED REFUSAL | RULED, queued behind Q3/Q4 | R306(d) |
| F-816-28 | preserve BOTH invariants or the primitive does not move | RULED BY PRINCIPLE, queued behind Q3/Q4 | R306(d) |
| F-816-30 | a skip guard must detect the MECHANISM, never a proxy | RULED; carried by PACKET_CI_RUNTIME, which forwards first | R306(d) |
| F-816-11 | arena/eval ply cap as an unconfigurable literal | LIVE precondition, discharged IN FACT at HEAD but never closed | R338(d) |
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
| DASH-2 | `mantis dash serve`, a read-only stdlib HTTP server over the run record carrying the GAME VIEWER, loopback by default | ORDERED, NOT BUILT. Owes an R9 amendment to repo_design.md in the SAME commit as the code. One finding already booked: a concurrent block writes every progress row at BLOCK END, so from outside it is indistinguishable from a wedge. The OBSERVATORY design (`docs/design/observatory_design.md`, 2026-09-14) is this card's design; its phase-1 readers landed at `3a563574..4678537d` and were RETIRED from the tree on 2026-09-17 under R355(f) (one dashboard implementation stays: `tools/dashboard` + `tools/viewer`, which serve the box page) — revive them from history when this is built: they read run6's record in 3.0 s at 84 MB peak against the dashboard's 5.0 s at 729 MB, parity-tested against it | R344(d) |
| RUNG-2 | new external rungs — strix first, shrimp second | strix LANDED (the frontier's cell, the `tools/strix_follower.py` equal-work 256/256 cell on every checkpoint and promotion, ACCEPTED at 15k + promotions, R359(e)). shrimp HELD for an architect read on the R257 radius fence | R356(a), R359(e) |
| INCR-GRAPH / S-INCR-GRAPH | incremental axis-graph construction from the parent position | PARKED, after being elevated to the top of the floor lane at R325. A CANDIDATE, not a plan: gated on a Rust-criterion box measurement, falsifier pre-registered as F-19's own inequality (`delta_cost x depth < build_cost`). Outside F-17/F-19's measured scope — see `docs/governance/falsified.md` | R335(e) |
| HOT-14 | cross-core ownership explains x1.66 of x6.84 | RE-OPENED when S-PREFUSE was refuted | R336(a) |
| S-BATTERY-G | eval battery concurrency capability | landed UNARMED; the CUDA arm is OWED at the mint's battery | R336(a) |
| S-CHECK17 | trainer-step check-17 bar | bar MISSED and BANKED at x1.18 — banked, not tuned toward | R336(a) |
| PERF-TRANCHE-1 residual | the 7.2% pre-control/ledger disagreement | OPEN as instrument hygiene; ledger absolute levels are not quotable without re-measurement | R320 |
| PERF-TRANCHE-2 | six items T2-1..T2-6 | EXECUTED — its findings are cited as landed evidence by R335 — but NO ratifying clause exists in either archive file | R334(e) |
| WP-AXIS2 | Phase 2 axis-graph arch, then a shakedown | LAST ORDERED, NEVER CONFIRMED. Neither the shakedown nor the R339 mint is ever labelled WP-AXIS2, so completion would be an inference, not a record | R335(g) |
| DASH-1 banked panels | average sims/move, held-out loss | 2 BANKED with no producer at HEAD; drawn as stated gaps, never as zeros | R334(a) |
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
| CARD-CLEANSTOP-SAVE leg (b) | the `checkpoint_interval` prereg row (leg (a) discharged) | LIVE, pinned to the operator's prereg batch. AMBIGUOUS: run6 mints `checkpoint_interval: 1000` and the derived index was never updated |
| CARD-RESUME-LAUNCHER-FLAG | supervisor auto-resume / `--resume-from` launcher surface | BUILT IN CODE: `python -m mantis.run --resume-from <bundle>` is the ONE optional launcher flag (`resolve_bootstrap` fails a stale one at launch); run7 resumed through it on 2026-09-15. Supervisor AUTO-resume is not built and not ordered; the governance row closes when a ruling names it |
| CARD-EVAL-CHANNEL-SPLIT | split the promotion and external eval cadences | OPEN, narrowed. R343(b)(v)'s conditional FIRED; R345(c) moved `gate.stride` to 3 and ledgered "the split that was already a key"; superseded for run7 — the gate runs at stride 1 on the 3 000-step cadence (R350(d), CARD-EVAL-CADENCE) |
| CARD-PROTOCOL-COMPLETE | complete protocol declarations, widen the AST conformance gate, LAW-16 sink/watchdog row | OPEN, pre-cutover, NOT mint-blocking |
| CARD-DENSE-EVAL-ADAPTER | wire `infer_batch_per_cluster` into the deploy-head decode | OPEN — pre-Stage-0 BLOCKING, not mint-blocking |
| CARD-LINT-TYPE | ruff/pyright advisory type-debt backlog | OPEN debt row, deliberately kept out of the gate by R98 |
| CARD-PYRIGHT-STRICT | pyright strict-mode adoption as a post-cutover ratchet | OPEN; live marker in pyproject.toml's `[tool.pyright]` comment block |
| CARD-TORCH-INDEX | conditional torch index / uv extra for the CPU-wheel parity regime | OPEN, post-mint |
| CARD-EVAL-CORESIDENCY | characterize eval-child steady VRAM for the co-residency prereg row | OPEN. The founding 8.21 GiB figure was superseded by R229(1) (unbounded, to 13.5 GiB) without naming the card |
| CARD-A10-CAP | whether an entropy term enters the graph loop at all | RECORDED, explicitly NOT executed. R335(b) makes entropy normalization a PRECONDITION on ever arming one |
| CARD-SEALBOT-BRANCHES | evaluate ramora0 branches (nnue) as a higher ladder rung | DEFERRED, not mint-relevant |
| CARD-MINPIN | the K-cluster min/max asymmetry, pending the matched-FLOP dense arm | OPEN — no in-tree pin exists (`aggregate_cluster_values_min` has zero hits; both registry rows read `value_pool = "none"`); the asymmetry stays a flagged defect. See falsified.md F-04 |
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
