# CARDS — the open work

Every card that is open at R346, with the ruling that last moved it. A card leaves this file by
being closed, refused or spent — never by going quiet. Closed cards are not kept here; the
reasoning that closed one lives in its ruling entry in `docs/governance/RULINGS.md`.

Status words mean what they mean elsewhere in this repo: **BLOCKING** stops the thing it names;
**MINT-BLOCKING** stops a mint; **HELD** is waiting on a named event; **CARDED** is accepted work
with no date; **OWED** is a text or a value someone must supply.

## Opened by R346 itself

These four came out of the CLEANUP ERA move and are recorded here because the move made them,
not because a ruling carded them.

- **CARD-CLAUDEMD-REPOINT — BLOCKING (CI gate 10).** `CLAUDE.md` names `docs/registers/falsified.md`
  (lines 6, 68) and `docs/registers/laws.md` (lines 7, 69, 75). R346 dissolved that directory, so
  gate 10 (`tools/ci_gates/check_tracked_refs.py`) reds on five lines. The replacements are
  `docs/governance/falsified.md` and `docs/governance/LAWS.md`, and line 75's "full text" pointer
  should read `docs/governance/archive/laws.md`, which is where the earned mechanisms went.
  **Only the operator may edit CLAUDE.md**; GOV-1 left it untouched deliberately.
- **CARD-GATE10-SCOPE.** Gate 10's own scope is `Makefile, README.md, CLAUDE.md,
  docs/contracts/*.md, docs/registers/*.md`. It still globs `docs/registers/`, which is now empty,
  and it does not scan `docs/governance/` at all — so the five new governance docs are outside the
  gate that exists to catch exactly their failure mode. The glob floor (`MIN_GLOB_FILES = 5`) is
  still met by `docs/contracts/`'s twelve files, so the gate does not raise; it just quietly checks
  less than it says it does, which is the AUDIT-1 F-26 class its own comment warns about. A `.py`
  change, out of a docs-only agent's remit.
- **CARD-STALE-REGISTER-CITES.** Six source files still cite the dissolved paths in docstrings,
  comments and one test fixture string. None is an executable dependency, so nothing fails; all six
  are misinformation a future reader trusts (SF-7). They are
  `src/mantis/diagnostics/worker_sweep.py:73`, `crates/mantis-graph/src/lib.rs:652`,
  `tests/model/conformance/test_arch_states_its_perf_floor.py:39`,
  `tests/model/conformance/test_leaf_forward_throughput_harness.py:38`,
  `tests/model/test_gine_gather_regime.py:10` (which also carries a `laws.md:37-39` line cite that
  the 60-line LAWS.md invalidates), and `tests/tools/test_rule7_gate.py:93`.
- **CARD-PHANTOM-TOOL-COMMENT.** `tools/ci_gates/tier_census.py:110` reasons by analogy to
  `ruling_census.py` and `sync_governance.py`. Neither tool has ever existed in this repository —
  they lived in the migration workspace, and R346 deleted that class of tooling. The argument the
  comment makes is sound; only its two witnesses are phantom.

## What holds run6

run6 is minted and has never started. Two things hold it, and one open class rides with it.

- **REPAIR-A2 leg 5 — the MCTS root child cap. BLOCKING, the architect's.** Seven of eight
  REPAIR-A2 legs landed; leg 5 halted with numbers rather than moving the cap. Measured: the cap
  discards a mean 88% of the policy's prior mass on 99.97% of expansions; "the r8 legal maximum" is
  not a constant (355 median, 489 max clustered, up to 8142 sprawling); the witness "omitted mass
  reads 0" is unreachable at any feasible K, since K = 2048 still drops 25% and halves
  `MAX_ARMED_SIMS` to 122. The tree memory delta is EXACTLY ZERO — the pool is preallocated at
  `MAX_NODES` — so what K costs is the armed-sims ceiling, which gates configs and moves run6's
  search. Last moved by R345.
- **`F-816-37` — a run-fatal `EdgeAttrGeometryMismatch` that is still not root-caused. OPEN, and
  it rides run6 rather than holding it.** Its R339(c)/R340 halt was raised on the host that R341
  condemned on signature and R342 then downgraded to SUSPECT on the operator's override; the work
  moved to a different box and the re-mint completed there. Every firing on record is on the old
  host, so the halt is spent — but the CLASS is not closed and nothing has root-caused it.
  Converted from a hunt into an instrument by R339(c): 1-in-1 on the eval path with dump-on-fire
  proven by a planted corruption, self-play deliberately left at the derived 1-in-64. It has fired
  three times. The signature is a single float32 exponent-LSB flip — **bit 23 in every corrupted
  word across every firing**, with all three corrupted words at byte offset congruent to 4 mod 8,
  the same half of the 64-bit bus. R340(c) discriminated software vs hardware and returned **BOTH
  NEGATIVE**: the software arm is the strong half (the mask appears in no spelling anywhere, no
  site views wire floats as integers, and `verify_contract` passes an exact-equality one-hot test
  on those bytes before the graph is emitted, leaving a memcpy-only corruption window); the
  hardware arm is weak, because 2.85 TiB over 81 cycles found zero errors but covered 32-42 GiB of
  60.53 GiB, and the ECC/MCE channel **returned no verdict and could not** — EDAC registers zero
  memory controllers, so that zero is a phantom gate, not a clean read. The host is un-convicted,
  not cleared. Two candidates are dead by measurement: the parallel leaf build (the firing driver
  passes no `leaf_build_threads`, so the engine took the fully serial arm) and the concurrency
  family generally (a race produces whole wrong values, not the same lone exponent bit three
  times). `F-816-37 dump-on-fire` is in the protected set. Last moved by R341/R342 on the host.
- **The `supervisor_kill_grace_sec` reading. OWED, one line.** `600.0` was armed on the reading
  that "only if the supervisor is used" describes when the value takes effect rather than
  conditioning the change. A one-line re-mint if the operator reads it the other way.

`R319(d)` — the gate round that could not finish inside `round_timeout_sec` — is no longer an
open question. R339(b) adjudicated it BY MEASUREMENT: the geometry does not move, the lever is
`eval.concurrency`, and the value is picked by a rule run on the box. run6 mints `concurrency: 8`.

## Owed texts and values

- **R267 — TEXT OWED.** No register section exists. The only surviving record is a STATE digest
  line ("eval posture mechanism inert, values operator"). Deliberately NOT reconstructed: a digest
  line is not the ruling. It is to be filled from the exported transcript, and it sits on the
  operator's residue list.
- **R147 / `eval.random_floor_games` on run5 — VALUE OPERATOR-OWED, BLOCKING until valued.**
  `configs/run5.yaml` still carries `random_floor_games: 0`. It is an armed value, therefore
  mint-prereg only, so no dispatcher may touch it — the config being "wrong" and staying untouched
  is correct behaviour, on the record. run6 is unaffected: it mints `20`.
- **R226 / R229 / R243 — prereg rows owed:** two flagged at dispatch 8C, three 8B findings.
- **R245(c) — the LAW-18 augmentation-group counter is OWED.** The per-record losslessness gate
  landed; the in-run fire-rate counter beside it did not.

## Carded work with no date

- **STRENGTH-FRONTIER-1** — measures the sims question at block end on run6's own frozen
  checkpoints. Ordered by R345, sequenced after the block.
- **GUMBEL-REPAIR-1** — lands to Mctx invariants during the block but is **enabled in no run**
  until the frontier compares it at equal NN work. run6 mints `gumbel_mcts: false`. Ordered by R345.
- **GAME-RECORD-1** — every game written from step 0, one record per game with the move list in
  axial coordinates, append-only length-delimited msgpack shards, no new hard dependency. Ordered
  by R344 BEFORE the start, on the ground that a run which does not write its games cannot be
  viewed, replayed or mined.
- **DASH-2** — `mantis dash serve`, a read-only stdlib HTTP server over the run record carrying the
  game viewer, loopback by default. Ordered by R344, design decided in the packet.
- **RUNG-2** — sequenced behind DASH-2; strix first, shrimp held for an architect read on R257's
  radius fence.
- **The promotion-interval split** — R344 confirmed 1000 on both channels and R345 re-ruled the
  cadence on arithmetic (`eval_interval` 1000, `gate.stride` 3). The split itself stays carded.
- **INCR-GRAPH** — incremental axis-graph construction from the parent position. A CANDIDATE, not a
  plan: registered as the neighbour F-19's grave explicitly does NOT cover, gated on a Rust-criterion
  box measurement, with its falsifier pre-registered as F-19's own inequality
  (`delta_cost x depth < build_cost`). See `docs/governance/falsified.md`, F-19's scope annotation.
- **The rest of AUDIT-2** — R345 carded everything it did not rule, each item with its priority,
  rather than adopting the audit wholesale. KLENT's search-free Shrimp target was REFUSED BY NAME,
  because it trusts an action-Q head this repo does not train.

## Audit findings still open

`F-<number>` is NOT one namespace. **Five different registers use it**, so every citation must
name which: the graves in `docs/governance/falsified.md` (`F-01`..`F-43`, which are graves and
never cards); AUDIT-1's 52 findings; the session ledger's; a perf-ledger `F-10` corrected by R320;
and R246's cross-language-parity `F-01`/`F-02`. Three distinct `F-10`s exist, and unpadded
`F-1`..`F-6` are ADJ-13 / RED-TEAM findings closed as a class by R71/R72 — they do NOT mean
`F-01`..`F-06`.

- **F-39 (AUDIT-1) — REGISTERED, not banked.** 34 bridge-signature defaults shadow config keys;
  what shipped is an enumerated `REGISTERED_DEBT` that reds in both directions. Accepted R333(a).
- **F-11 (AUDIT-1) — disk-guard arming. STATUS UNCONFIRMED.** R334(b) armed shape A and ordered it
  to LAND BEFORE THE MINT. run6 was minted at R339 and no record says it landed. The row needs a
  reading against the tree before anyone treats it as done.
- **F-51 (AUDIT-1) — never disposed.** Cited as evidence for the per-tranche candidate mechanism
  (HOT-04 / HOT-11 / HOT-14) in R334(e)'s landing note; no clause ever rules on it.
- **F-42 (AUDIT-1) — banked at the REPAIR-2 exit, bank accepted, no movement since.**
- **F-10 / F-10b (session ledger) — a box run wedged at step 12. OPEN but scoped away from run6:**
  the wedge is on `gnn_axis_v1` and run6 mints `gnn_axis_r8`. No ruling has closed it.
- **F-Q6-1 — the flamegraph tool's own 12.4 GiB orphan.** Marked live until its row closes, routed
  to the carry-over queue by R300(d), and not seen since.
- **WP-AXIS2 — last ordered, never confirmed.** It appears last at R335(g) as the step between the
  mint launcher and run6. Neither the shakedown nor the R339 mint is ever labelled WP-AXIS2, so
  "absorbed" would be an inference. Recorded as unconfirmed rather than closed.

## How this list was derived, and its one known limit

Built at R346 from the archive's LIVE-marked rows, the last two curation entries, and the ruling
entries in `docs/governance/RULINGS.md`, then checked against the tree for every value it names.

**The limit, stated rather than hidden:** the archive's live-force section was never extended past
R337 — R338, R339, R340, R344 and R345 have no row there, and R341(e) and R343(f) still sit in it
marked LIVE carrying terms (a 750-step cadence, a 12 h block) that R343(b) and R344 have already
superseded. So a card list built by walking that section alone would miss the newest holds and
carry two dead ones. The cards above for R338 onward come from the ruling entries instead. Anyone
extending this file should do the same, and should not treat the frozen archive as a live queue.
