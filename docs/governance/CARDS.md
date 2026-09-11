# CARDS — the open work

Every card that is open at R346, with the ruling that last moved it. A card leaves this file by
being closed, refused or spent — never by going quiet. Closed cards are not kept here; the
reasoning that closed one lives in its ruling entry in `docs/governance/RULINGS.md`.

Status words mean what they mean elsewhere in this repo: **BLOCKING** stops the thing it names;
**MINT-BLOCKING** stops a mint; **HELD** is waiting on a named event; **CARDED** is accepted work
with no date; **OWED** is a text or a value someone must supply.

## Opened by R346 itself

All four came out of the CLEANUP ERA move and are now CLOSED, by wave 2. Kept as one line each
because a card leaves this file by being closed, not by going quiet.

- **CARD-CLAUDEMD-REPOINT — CLOSED, closing line in R348.** `CLAUDE.md` names
  `docs/governance/falsified.md`, `docs/governance/LAWS.md` and `docs/governance/archive/laws.md`;
  gate 10 is green on it.
- **CARD-GATE10-SCOPE — CLOSED.** `tools/ci_gates/check_tracked_refs.py` globs
  `docs/governance/` and carries a per-directory floor, so a dissolved directory can no longer be
  absorbed into a full one's count. `RULINGS.md` is exempt by declaration with grounds in the
  gate: it corrects only by annotation and ANNOTATION 7 deliberately preserves a wrong path
  string. `docs/registers/` is whitelisted as DISSOLVED and the entry refuses itself if the path
  is ever tracked again.
- **CARD-STALE-REGISTER-CITES — CLOSED.** All six source cites now name `docs/governance/`.
- **CARD-PHANTOM-TOOL-COMMENT — CLOSED.** `tools/ci_gates/tier_census.py` cites LAW-07 instead of
  two tools that never existed here.


## Opened by R347 (CLEANUP WAVE 2)

Both were found by running the gate set rather than by reading it, and both are pre-existing on
`dev` — neither was introduced by wave 2.

- **CARD-OC7-OVERRUN — BLOCKING (the integration tier, hence `make gates.exit`).**
  `tests/train/test_clean_stop_save.py::test_a_clean_run_at_the_minted_bound_leaves_one_stamped_checkpoint`
  (OC-7) drives a real 50-step run on 14 workers and **exceeds its own stated 300 s tier ceiling**:
  killed at the cap on `dev`, and uncapped it passed 900 s on both `dev` and `gumbel-3`. Its bound
  `_OC7_BOUND = 50` was fixed by a stated, host-relative decision rule — the largest member of
  {200, 100, 50, 32, 16} measuring <= 300 s **on the dev box** — and M-0 (PREREG_CS §5.3) measured
  50 at **240.2 s** on 2026-08-01. The measurement no longer holds on that same box. Not yet
  separated: host drift versus a real slowdown landing after 2026-08-01, a window that contains
  five `mcts/` commits dated 2026-09-08/09, all of wave 1 among them. Until it is separated the
  bound must NOT be re-aimed — the docstring records that the three measurements were taken before
  the row existed precisely so the bound could not be lowered to make a red go away. Wave 2
  proceeds with the integration tier NOT RUN and the exclusion stated at each merge, on operator
  direction; the separation belongs to AUDIT-3.
  **CORRECTION, measured after this card was first written: OC-7 is not the whole of it.** The
  tier with OC-7 deselected — 49 tests — ALSO failed to finish, hitting a 3 000 s cap on
  `gumbel-3`. So the tier is not one bad test on an otherwise healthy suite; it is >50 min of
  work at best, and CLAUDE.md's "the superset is ~35 min" is stale by a wide margin on this host.
  What is NOT yet known: whether the remaining 49 are merely slow (49 real 14-worker boots would
  explain it) or whether a second test is unbounded like OC-7. A `-v` per-test timing pass is the
  next measurement and it is owed. **The consequence to hold on to: while this stands,
  no `make gates.exit` run in this repository can complete, so "local green is the gate" is
  answered by a gate that never finishes — the false-clean class, in the time dimension.**
  **DISCRIMINATED 2026-09-11 under R348(b) — HOST, not code.** Record:
  `docs/design/measurements/MEASUREMENT_OC7_2026-09-11.md`. (i) SLOW, not hung: the dev box's
  main thread sits in a bf16 CPU GEMM inside the GNN backward at ~60 s/step (14 workers) and
  ~40 s/step (1 worker); (ii) the same row at HEAD on the box **PASSES in 178.4 s**, inside its
  ceiling, at the minted 14 workers; (iii) the 2026-08-01 tree runs at the same ~40 s/step on the
  dev box today, so no bisect. Mechanism measured, not inferred: the dev box (Ryzen 7 3700X,
  AVX2, no AVX-512 BF16) runs LAW-06's bf16 autocast through ATen's generic path — one GEMM at
  the trainer's edge shape is **72× slower than fp32** there — while the box has native bf16.
  Contributing: `1203f740` (2026-08-31) minted 14 workers into the smoke config the row drives.
  The first fact at contact was a different defect: the row FAILED in 46 s on DELETE-1's 10→8
  `GameResultRow` change that the Python drain never received (fixed `92643671`). The bound is
  NOT re-aimed: the docstring's rule yields 50 on the box and nothing on an AVX2 host, and
  which of (tier runs on the box / row marked `slow` / LAW-06 CPU carve-out) is the operator's.

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

## Opened by R349 (the START path)

Records: `docs/design/measurements/MEASUREMENT_STARTPATH_2026-09-11.md`; falsified.md F-44/F-45.

- **CARD-STAMP-FLOOR — BLOCKING START; a ruling, not code.** R348(c)'s trap (`mantis.run`
  refuses a config with no passing stamp) meets the schema's own reachability rule: run6 mints
  `train.draw_rate_abort.min_step 25000`, so `_apply_burst_override` refuses every burst below
  25 001 steps (rc 11, as on 2026-09-08) and the only stamp-writing preflight of run6 is the
  block itself (≈ 22.6 h at 1 105 steps/h). The shakedown config (R259's run class) carries the
  same floor. Options (STATE.md, Hold 1): a `sync_lag`-tier stamp for production configs whose
  full tier exceeds the shakedown length, with draw-rate reachability proven by the schema rule;
  an in-process shakedown plus a ruled START bypass; or paying the 22.6 h. Phase W and the
  post-rebuild cap calibration wait on the same launch path.
- **CARD-ALPHA-TARGET-FORM — the architect's (R349(c)'s D2).** In decided positions the
  completed-Q improved policy is rescaled noise: `c_scale 1.0` on top of Mctx's per-node min-max
  maps a `< 0.06` Q spread onto 55–150 logits, so ~1 row per 1,000 (rising with value
  saturation) trains "none of the 16 searched moves". Options, each a regime change needing the
  PERF-3b re-measure and a new parity vector: the paper's σ(q̂) without min-max; `c_scale 0.1`
  (Mctx's default with the rescale); a span floor. Not a hold on run6 — the trainer's cost of
  such a row is a zero-gradient self-target — but a target-quality question the count on the
  dashboard now measures.
- **CARD-TRAINER-CADENCE — the architect's.** Steps/h ≡ games/h by `train.training_steps_per_game
  1.0` / `max_train_burst 1` (6.6 draws per row); the trainer is 92 % idle. The block is ≈ 22.6 h
  at 1,105 steps/h. Raising the ratio halves the wall clock and doubles sample reuse — a regime
  decision, not a lever. INVESTIGATION-1 item 2 is re-aimed: "is the reuse ratio right" and the
  GPU step's own 156 ms (`index_add_` 22 %, GEMMs 16 %, 31 syncs) — not "why is the step 5 s".
- **CARD-SERVER-SYNC — INVESTIGATION-1 item 3's first item.** The inference-server thread spends
  32 % of its wall in `_node_offsets_to_batch_vec`'s `repeat_interleave(…, counts)` — an implicit
  device→host sync at the top of every forward (`output_size=` removes it) — and 19 % in check 14
  inline (the checker-thread lever is landed, unarmed in run6's mint); the 16 workers wait on the
  server 99.6 % of their wall. Measured, not yet benched (LAW-09 prereg first).
- **CARD-DEPLOY-HEAD-BUDGET — item 4.** In decided positions the deploy head spends 28–40 of a
  64-sim budget and 52–77 of 320: `gumbel_root_select` returns `None` early. The eval instrument
  under-spends exactly where the position is settled; whether that moves a bar is unmeasured.

## Opened by R348 (WAVE 3, leg 1)

All five were found by running things the dev box could not run — the integration tier and the
Gumbel regime at scale on the box — and every one is pre-existing on `dev`. Records:
`docs/design/measurements/MEASUREMENT_OC7_2026-09-11.md`, `MEASUREMENT_PERF3B_2026-09-11.md`.

- **CARD-BOX-VOLUME — CLOSED by R349(b) at `8e307af1`.** `/` and `/workspace` on the box are
  `overlay`; R347(d)'s rc 16 refused every preflight there and is DELETED. `tools/mirror_pull.py`
  (the operator's machine, an rsync spec, receipts beside each artifact on the box) is the arm;
  the preflight's rc 16 now demands receipts for the burst's bundle and first shard, `mantis.run`
  refuses a stamp without the `MIRRORED` verdict, and `resume_state_persisted.unreceipted_bundles`
  feeds the dashboard's two-interval warning. Proven over the alias on the START-path burst
  (cycle 11.1 s). Loss-on-recycle is the operator's RECORDED acceptance.
- **CARD-WARMSTART-STAMP-SCHEMA — CLOSED by R349(a) at `1793fee1`.** run6's BC artifact's stamped
  config no longer validated under the wave-2 schema (49 `extra_forbidden`); the LAW-12 strip
  (`checkpoints/bc/run6_00006500_ca1afb71.ckpt`, net hash unchanged) is now the minted
  `identity.warm_start.checkpoint`, and the default tier's one true red is green wherever the
  strip exists.
- **CARD-TIER-HOST — RULED by R349(a): the carve-out.** On an AVX2 host LAW-06's bf16 CPU trainer
  was emulated (72× per GEMM) and the tier could not finish. R349(a) takes the third option as a
  law amendment: fp32 on `train.device: cpu` (the autocast context, never the dtype pin), landed
  at `9491b4d0` with its own parity test; the tier runs on CUDA where a card exists (the dev box's
  3070 via `make build.cuda`, the box remotely). TEST-1's minted CUDA smoke profile is the fourth
  option and is still owed.
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
- **CARD-GATES-ON-CUDA-VENV — the gate set had never gated a CUDA venv.** Every `$UV run` in
  `run_all.sh` re-synced to the default groups, so every box gate run in history silently ran
  on the CPU wheel; fixed at `8dfa8b5f` (`UV_NO_SYNC=1`, torch build printed). Gated as built
  on `+cu128`, gate 3a shows rows whose expectations assume a CPU torch: the model puts itself
  on CUDA while the test's tensors stay on CPU ("Expected all tensors to be on the same
  device") in `tests/model/conformance/test_arch_states_its_memory_envelope.py` (8),
  `tests/model/test_gnn_v2_witnesses.py::test_both_arches_FORWARD_on_a_real_wire_position` (2),
  the `slow` tier's harness rows through `test_local_gate_runner`, and the ten-row control in
  `test_preflight_start_halts` (fixed in place). **CLOSED by R349(a)** — the four collate sites
  in the model oracles state `device="cpu"` at `8443d0e5` instead of letting
  `collate_graph_batch(device=None)` sniff CUDA beside a CPU net; all 150 rows of the affected
  suites pass on the box's `+cu128` venv.
- **CARD-CHECKER-THREAD-LEVER — the lever LANDED at `8443d0e5`; the A/B is the measurement.**
  `inference.edge_geometry_check` (schema default `inline`; `checker_thread` moves check 14 to
  one bounded checker thread after the batch is served, a full queue runs it inline, a failure
  dumps the same artifact and halts through `deferred_contract_failure` on the next pop and the
  pool health check). LAW-18 counters ride `iteration_complete.inference_batching.edge_geometry_check`.
  The Rust verifier holds the GIL, which bounds the gain; the A/B reading is in
  `docs/design/measurements/MEASUREMENT_PERF3B_2026-09-11.md`. Arming it in run6 is a mint row.

## Reading the identifiers

Cites below are `A:` for `docs/governance/archive/RULINGS_ACTIVE.md` and `R:` for
`docs/governance/archive/rulings_register.md`. Line numbers are unchanged by the R346 move, and
they are a starting point, not evidence — derive at point of use.

Four traps, each of which has already misled a reader:

1. **`F-<number>` is five namespaces, not one.** The graves in `docs/governance/falsified.md`
   (`F-01`..`F-43`) · AUDIT-1's 52 findings · the session ledger's · a perf-ledger `F-10`
   corrected by R320 · R246's cross-language-parity `F-01`/`F-02`. **Three distinct `F-10`s
   exist.** Unpadded `F-1`..`F-6` are ADJ-13 / RED-TEAM findings closed as a class by R71/R72 and
   never mean `F-01`..`F-06`. Every `F-NN` here names its register. Graves are never cards.
2. **`CARD-LEVEL` is not an identifier.** Every hit is the phrase "CARD-LEVEL FACT" — a fact about
   the GPU card. There is no such card.
3. **`CARD-ANCHOR-WIRING` was DECLINED** — R55 ruled it "is therefore not created". It is not a card.
4. **Two ADJ number-spaces collide** (WPUF/WPAX-era against WP12-R-era) at ADJ-19 through ADJ-26
   and ADJ-29. Qualify by era or by the enclosing ruling.

## What holds run6

run6 is minted and has never started.

- **REPAIR-A2 leg 5 — the MCTS root child cap. BLOCKING, the architect's.** Seven of eight legs
  landed; leg 5 halted with numbers rather than moving `MAX_CHILDREN_PER_NODE = 192`. Measured: the
  cap discards a mean 88% of the policy's prior mass on 99.97% of expansions; "the r8 legal
  maximum" is not a constant (355 median, 489 max clustered, up to 8142 sprawling); the witness
  "omitted mass reads 0" is unreachable at any feasible K, since K = 2048 still drops 25% and halves
  `MAX_ARMED_SIMS` to 122. The tree memory delta is EXACTLY ZERO — the pool is preallocated at
  `MAX_NODES` — so what K costs is the armed-sims ceiling, which gates configs. R345(b)(5); A:7687.
- **`F-816-24` — CLOSED by R349(a).** Read at contact 2026-09-11 (R348(e)'s AUDIT-3 lead):
  `supervise.main` loads the minted config and resolves every `monitor.supervisor_*` through
  `resolve_monitor_config` since `c8bd7190` (2026-08-21), with a named refusal for a missing
  `--config`; `tests/monitor/test_supervisor_config_witness.py` carries the running-supervisor
  witnesses (integration) and an AST pin that the module constructs no `MonitorConfig` by any
  shape (default tier). R291(b); A:2662.
- **The `supervisor_kill_grace_sec` reading — SETTLED at `a37d2e5e`.** The operator read it the
  other way: the re-mint reverts `600.0` to the template's `30.0` (R348(e), merged under R349).
  A:7718.

Riding the run rather than holding it: **`F-816-37`**, below.

**`R341(b)` / `R319(d)` — DISCHARGED AT G=8 ONLY (R343(a)). Not a hold on run6; still LIVE below
G=8.** R341(b) withdrew R340(a)'s closure because the round that must finish is the CONTENDED one.
R341(c)'s G table then ran and G=8 was armed on the operator's forward: **G=1 at 53.33 s/game and
G=4 at 13.99 s/game both consume the full 3600 s `round_timeout_sec` and return
`wr_sealbot: null`; G=8 at 7.09 s/game completed a fully escalated 264-game round in 1872.8 s,
78.8% of the 2376 s bar, with a real `wr_sealbot`**, and the 4 h shakedown held a steady 900.6 s
wall with no growth over seven completed rounds, zero nulls, two promotions. run6 mints
`eval.concurrency = 8`. **The row is discharged by the ARMED VALUE, not by the geometry becoming
safe** — lowering concurrency to 4 walks straight back into a timeout, and a null is not a slow
reading: witness (iii) fits an Elo slope over at least 5 rounds and cannot fit nulls, so a geometry
failure disarms one of run6's three success witnesses. R343(a); A:1927-1939, A:7249, A:7341.

## F-816-* findings

| id | subject | status | last moved | cite |
|---|---|---|---|---|
| F-816-37 | run-fatal `EdgeAttrGeometryMismatch` at run6's minted geometry, not root-caused | OPEN. Converted into a 1-in-1 eval-path instrument with dump-on-fire (protected set); zero shakedown firings is explicitly NOT a close. Every firing on record is on the host R341 condemned and R342 downgraded to SUSPECT, and the work moved to a different box — the halt is spent, the class is not | R342(a) | A:1926 |
| F-816-24 | bare `MonitorConfig()` — minted `supervisor_*` reach no process | CLOSED by R349(a); fixed at `c8bd7190` (2026-08-21), witnessed | R349(a) | A:2662 |
| F-816-27 | supervisor kill-grace CEILING absent (schema is `Field(ge=0)` only) | RULED; rides prereg row 19 to the operator | R338(c) | A:3498 |
| F-816-34 | vacuous knee band — PICK = 2 from a band widened below every rung | FILED 2026-09-04, never adjudicated | none | A:6739 |
| F-816-35 | r8 trainer need is a DISTRIBUTION exceeding `_SIZING_BUDGET_GIB` and R330(b)'s 3% | FILED, never adjudicated | none | A:6741 |
| F-816-36 | an unplayable rung sets every ring's composed visit capacity | FILED, never adjudicated | none | A:6745 |
| F-816-15 | `freeze_verify.py` red on 39 of 64 paths; audit-before-rebaseline | ORDERED as its own packet, never dispatched | R285(g)/R286(c) | A:3582 |
| F-816-19 | the run's own process is spawned unparented (PDEATHSIG class) | ORDERED PRE-MINT, no close | R285(h) | A:2533 |
| F-816-21 | test de-triplication — one stub in three files across two registers | RE-SEQUENCED behind RQ-1; owed inside the freeze packet | R288(d) | R:5058 |
| F-816-26 | parent/child config binding; a mismatch is a NAMED REFUSAL | RULED, queued behind Q3/Q4 | R306(d) | A:4048 |
| F-816-28 | preserve BOTH invariants or the primitive does not move | RULED BY PRINCIPLE, queued behind Q3/Q4 | R306(d) | A:4049 |
| F-816-30 | a skip guard must detect the MECHANISM, never a proxy | RULED; carried by PACKET_CI_RUNTIME, which forwards first | R306(d) | A:4042 |
| F-816-11 | arena/eval ply cap as an unconfigurable literal | LIVE precondition, discharged IN FACT at HEAD but never closed | R338(d) | A:3444 |
| F-816-14 | the eval child survives its parent's SIGTERM holding 458 MiB | HALF-OPEN — the SIGKILL leg closed, the SIGTERM leg re-worded as F-Q6-8 | R300(d) | A:2716 |
| F-816-17 | dead `legal_mask` build | routing RATIFIED AS FILED, no close | R286(f) | A:2592 |
| F-816-1 | run5 death was a host event with no software error line | no close ever recorded | R268 | A:2287 |
| F-816-2 | independent card riding the VisitSlotsExceeded packet | CARDED, no close recorded | R274(e) | R:4309 |
| F-816-4 | thread-leak / process-global-state hazard class | no status ever recorded | R289(s) | R:5034 |
| F-816-6 | degenerate ply-cap flood, draw_rate 1.000 at bootstrap | MINT-CRITICAL headline, no close recorded | R269 | A:2288 |
| F-816-8 | `wppre-scratch` branch containment ground | no status recorded | R277(c) | R:4270 |
| F-R302-1 | trainer-forward OOM; allocator-reservation fragmentation | EXPLAINED, NOT CLOSED — "closes at a standing mint" | R315(a) | A:1827 |
| F-B1 | parent/child same-file config binding (`config_identity_sha256`) | LIVE as DESIGN input to the F-816-24 packet | R292(c) | A:2683 |
| F-Q6-1 | the flamegraph tool's own 12.4 GiB orphan (PDEATHSIG family, instrument side) | routed to the carry-over queue, "live until their rows close", not seen since | R300(d) | A:2722 |
| F-Q6-8 | save-then-exit did not hold under OOM (LAW-16) — the re-worded F-816-14 SIGTERM leg | OPEN; the close-out measurement is filed beside it | R302(d) | A:2718 |

## Named work items

| item | subject | status | last moved |
|---|---|---|---|
| STRENGTH-FRONTIER-1 | measures the sims question (50 vs 96) at block end on run6's own frozen checkpoints | ORDERED, waits on block end | R345(d) |
| GUMBEL-REPAIR-1 | Gumbel repaired to Mctx invariants; lands DURING the block, enabled in no run until the frontier compares at equal NN work | ORDERED — lands, stays UNARMED. run6 mints `gumbel_mcts: false` | R345(d) |
| GAME-RECORD-1 | every game written from step 0; move list in axial coordinates, append-only length-delimited msgpack shards, no new hard dependency | ORDERED BEFORE THE START — a run that does not write its games cannot be viewed, replayed or mined | R344 |
| DASH-2 | `mantis dash serve`, a read-only stdlib HTTP server over the run record carrying the GAME VIEWER, loopback by default | ORDERED, NOT BUILT. Owes an R9 amendment to repo_design.md in the SAME commit as the code. One finding already booked: a concurrent block writes every progress row at BLOCK END, so from outside it is indistinguishable from a wedge | R344(d) |
| RUNG-2 | new external rungs — strix first, shrimp second | ORDERED as mid-run work, deferrable but not optional, sequenced AFTER DASH-2; shrimp HELD for an architect read on the R257 radius fence | R344(e) |
| INCR-GRAPH / S-INCR-GRAPH | incremental axis-graph construction from the parent position | PARKED, after being elevated to the top of the floor lane at R325. A CANDIDATE, not a plan: gated on a Rust-criterion box measurement, falsifier pre-registered as F-19's own inequality (`delta_cost x depth < build_cost`). Outside F-17/F-19's measured scope — see `docs/governance/falsified.md` | R335(e) |
| HOT-14 | cross-core ownership explains x1.66 of x6.84 | RE-OPENED when S-PREFUSE was refuted | R336(a) |
| S-BATTERY-G | eval battery concurrency capability | landed UNARMED; the CUDA arm is OWED at the mint's battery | R336(a) |
| S-CHECK17 | trainer-step check-17 bar | bar MISSED and BANKED at x1.18 — banked, not tuned toward | R336(a) |
| PERF-TRANCHE-1 residual | the 7.2% pre-control/ledger disagreement | OPEN as instrument hygiene; ledger absolute levels are not quotable without re-measurement | R320 |
| PERF-TRANCHE-2 | six items T2-1..T2-6 | EXECUTED — its findings are cited as landed evidence by R335 — but NO ratifying clause exists in either archive file | R334(e) |
| WP-AXIS2 | Phase 2 axis-graph arch, then a shakedown | LAST ORDERED, NEVER CONFIRMED. Neither the shakedown nor the R339 mint is ever labelled WP-AXIS2, so completion would be an inference, not a record | R335(g) |
| AUDIT-2 filing | the `AUDIT_2026-09-09.md` analysis text | **CLOSED.** Filed to `dev` at `428f3c8` as `docs/audits/AUDIT_2026-09-09.md`, 1661 lines. Read it with its own header caveat: the audit was taken at `97e814e3`, 29 commits behind `fb3725f`, so REPAIR-A2 and GUMBEL-REPAIR-1 both post-date it and its findings are not a statement about HEAD | R346 era |
| DASH-1 banked panels | average sims/move, held-out loss | 2 BANKED with no producer at HEAD; drawn as stated gaps, never as zeros | R334(a) |
| R317(c)(ii) diagnostic | move-sequence-hash diagnostic | accepted as NON-BLOCKING DEBT, never shipped | R318 |
| AUDIT-1 P10 | lane-C design input, explicitly "not a packet" | still the architect's, undispatched | R331(d) |

## Owed texts and values

- **R227, R228, R267 — TEXTS OWED, operator residue.** ADJ-D2 covers R227/R228 and directs that
  they are NOT filled agent-side; it is load-bearing because it discharges R56/R133/R138. R267 has
  no section at all — the only record is a STATE digest line, deliberately not reconstructed
  because a digest line is not the ruling. A:1367, A:3732.
- **run5 prereg values — SPENT.** R137 leg (b) `checkpoint_interval` and R147
  `eval.random_floor_games` were owed against run5's config, which minted them at `0`. R346(f)
  pruned that config out of the tree, so the rows have no subject; run6 mints `1000` and `20`.
- **R226 / R229 / R243 — prereg rows owed:** two flagged at dispatch 8C, three 8B findings.
- **R245(c) — the LAW-18 augmentation-group counter is OWED.** The per-record losslessness gate
  landed; the in-run fire-rate counter beside it did not.

## CARD-* named in governance

| card | subject | status |
|---|---|---|
| CARD-RUN5-GPU-OOM | GPU-OOM defect CLASS; the site set now includes the GNN training forward | OPEN as a class. Instance F-816-12 closed at the joint mint; the class row never closed. ANNOTATION 4 / R302(c) rider |
| CARD-CLEANSTOP-SAVE leg (b) | the `checkpoint_interval` prereg row (leg (a) discharged) | LIVE, pinned to the operator's prereg batch. AMBIGUOUS: run6 mints `checkpoint_interval: 1000` and the derived index was never updated |
| CARD-RESUME-LAUNCHER-FLAG | supervisor auto-resume / `--resume-from` launcher surface | OPEN — declared NOT BUILT on a CONTRACT, not a time-box; three admissible shapes carded |
| CARD-EVAL-CHANNEL-SPLIT | split the promotion and external eval cadences | OPEN, narrowed. R343(b)(v)'s conditional FIRED; R345(c) moved `gate.stride` to 3 and ledgered "the split that was already a key" |
| CARD-PROTOCOL-COMPLETE | complete protocol declarations, widen the AST conformance gate, LAW-16 sink/watchdog row | OPEN, pre-cutover, NOT mint-blocking |
| CARD-DENSE-EVAL-ADAPTER | wire `infer_batch_per_cluster` into the deploy-head decode | OPEN — pre-Stage-0 BLOCKING, not mint-blocking |
| CARD-LINT-TYPE | ruff/pyright advisory type-debt backlog | OPEN debt row, deliberately kept out of the gate by R98 |
| CARD-PYRIGHT-STRICT | pyright strict-mode adoption as a post-cutover ratchet | OPEN; live marker at `pyproject.toml:92` |
| CARD-MAXPLIES | `_DEFAULT_MAX_PLIES` schema promotion | OPEN in governance, CLOSED IN CODE — the symbol is gone and `max_plies` derives from `selfplay.max_game_moves` |
| CARD-TORCH-INDEX | conditional torch index / uv extra for the CPU-wheel parity regime | OPEN, post-mint |
| CARD-EVAL-CORESIDENCY | characterize eval-child steady VRAM for the co-residency prereg row | OPEN. The founding 8.21 GiB figure was superseded by R229(1) (unbounded, to 13.5 GiB) without naming the card |
| CARD-A10-CAP | whether an entropy term enters the graph loop at all | RECORDED, explicitly NOT executed. R335(b) makes entropy normalization a PRECONDITION on ever arming one |
| CARD-SEALBOT-BRANCHES | evaluate ramora0 branches (nnue) as a higher ladder rung | DEFERRED, not mint-relevant |
| CARD-MINPIN | the K-cluster min/max asymmetry, pending the matched-FLOP dense arm | PARTIAL — the parity pin landed; the asymmetry stays a flagged defect. See falsified.md F-04 |
| CARD-CHECK14-EDGE-GEOMETRY | the `verify_edge_geometry` collate check ("check 14", NOT CI gate 14) | NO STATUS EVER RULED. R336(e) separately CARDS check 14's 41.4 ms/part, not ordered |
| CARD-FRESHSYNC | gate 1 fresh-clone sync broken since WP7 | OPEN in governance, REPAIRED IN CODE — the gate pins `registry_sha_hex()` and describes the failure in the past tense |

## CARD-* that exist only as in-source markers

Zero governance mentions; their status comes from the code, not from a ruling.

- `CARD-BUDGET-AUTHORITY-CONSOLIDATION` — the second `_SIZING_BUDGET_*` authority; filed as debt by
  R327(d). `tests/train/test_graph_microbatch_authority.py`
- `CARD-GAME-RECORD-SELFPLAY-STATS` — per-position self-play stats are un-associable to a game
  without a hot-drain engine act (LAW-09). `src/mantis/monitor/game_record.py`
- `CARD-CONFIG-DISCOVERY-ROOT` — config discovery root for every `--config` route.
  `src/mantis/config/loader.py`
- `CARD-EXEMPT-CONFIGS-OPERATOR-CONFIRM` — awaits a ruling. `src/mantis/config/armed_aborts.py`
- `CARD-MINT-RESOLVE-PARENT-CONJUNCT` — the `max(learners) >= 1` mint-resolve conjunct.
  `tests/config/test_mint_and_diff.py`
- `CARD-PREFLIGHT-ORACLE-OUTDIR-CLEANUP` — oracle out-dir cleanup. `tests/tools/conftest.py`
- `CARD-DESIGN-P-3.4-ORDERING` — status unknown. `tools/ci_gates/preflight_mint.py`

## RQ-*

- **RQ-5** — an independent cross-model review of the `freeze_verify` mission diff. ORDERED
  findings-only, unexecuted. R289(c).
- **RQ-7** — `supervisor_kill_grace_sec`. SPLIT; the VALUE is a prereg row and is operator-owed.
- **RQ-8** — `MonitorConfig` schema-resident defaults. ESCALATED into F-816-24, above.
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

**None open.** Q-C0 through Q-C10 are all ruled or closed, and Q-C6/Q-C7/Q-C8 never existed. One
stale line survives in the archive at A:3901, still reading "Q-C1..Q-C4 are OPEN at the queue
foot" — superseded by R304 and never stamped the way the archive's §7 stamps its superseded rows.
It is recorded here so nobody re-opens four closed questions from it.

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
   `docs/audits/AUDIT_2026-09-09.md`, but R345's "everything else in AUDIT-2 is CARDED with its
   priority" refers to a card list that travelled in the packet, not in the audit document. An
   unknown number of AUDIT-2 cards therefore remain outside this file, and it should not be read as
   complete.
