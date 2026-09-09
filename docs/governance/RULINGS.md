# RULINGS — R23 to R346

One entry per ruling. From R346 these entries are **canonical**: an entry here is what the
ruling means, and it is what a session cites. The verbatim pre-R346 wording is frozen in
`docs/governance/archive/rulings_register.md` — that archive is the historical record and the
place to go when an entry is not enough, not a competing authority.

**Conventions this file carries forward.**

- Numbering continues from R345. The next ruling is R347.
- A ruling corrects only by ANNOTATION, never by silent edit. Where a later ruling moved an
  earlier one, the entry's `Status` line says so and the `Decision` carries the corrected fact
  with the correction named. Nothing here rewrites history into having always been right.
- Packets are not committed. A packet is working material; what survives it is its ruling entry
  here and whatever it changed in the tree.
- Exit facts go into `docs/governance/STATE.md`, not into this file.
- Coordinates go stale. Every file:line in the archive was measured once and most have drifted;
  derive a coordinate at point of use (R98) rather than carrying one out of a ruling.

**Status vocabulary.** `standing` — in force. `superseded by R___` / `amended by R___` — a later
ruling moved it; the later text governs. `spent` — its condition was discharged and it directs
nothing further. `annotated — ANNOTATION n` — the register carries a correction beside it, and
the corrected fact is in the Decision above.

**R23-R31 carry an older authority.** The archive's own header says the operator's
`STATE_2026-07-24_ADDENDUM_A.md` is authoritative for R23-R31 and wins on any conflict of
substance; the archive's sections for those are a dispatcher's glosses. That addendum is not in
this repository. Treat the R23-R31 entries below as the best available reading, not as final text.

**Coverage.** Every number from R23 to R345 has an entry except **R227** and **R228**: R227 was
never filled and R228 exists only as three inline merge-append blocks, both on operator direction.
Four entries record an absence rather than a decision, and say so in place: **R24, R29, R32** had
no section in the register (their text lives in an operator addendum that is not in this
repository), **R33** is superseded in full by R37, and **R267** is a documented GAP whose text is
still owed. **R279(g)-ANNEX** carries its own entry, as it did its own register section. That is
322 entries over 321 numbers.

### R346 — the CLEANUP ERA
Decision: verbatim below. This entry breaks the <= 10-line convention deliberately: the packet
made its own §1 the canonical home and directed that it be copied verbatim as the first entry
here, so condensing it would destroy the authority it was given. Every later entry keeps the cap.

> R346 — (a) The CLEANUP ERA is opened by operator direction. Its
> object: one search kind per name, one config a person can read, one
> home for governance, comments that say why and nothing else, and a
> tree small enough to audit. Deletion is the default for anything
> unarmed, unmeasured and unreferenced by a standing law; the burden is
> on keeping, not on removing.
> (b) run6 SEARCH: Gumbel, mctx dialect, no Dirichlet, playout-cap
> randomization. PROPOSED rows (operator's, armed at the re-mint):
> full 320 sims / m 16 at p = 0.25; fast 64 sims / m 8 at p = 0.75
> (mean 128 — the operator's number; 400/37 was rejected because a
> 37-sim fast arm is thinner than the paper's own low-sim regime);
> completed-Q improved policy is the target on every row, is_full_search
> recorded; value_scale 0.1, c_visit 50; eval deploy Gumbel 160 / m 16,
> candidate and anchor matched. Throughput is predicted, not promised:
> serving-bound at ~4.4M leaves/h the mean-128 regime yields ~640
> games/h; PERF-3 measures it and the candidate-parallel batching in (c)
> is what keeps "fast" true.
> (c) GUMBEL-2 (wave 1): the legacy dialect and the PUCT-root hybrid are
> DELETED; `search.kind ∈ {puct, gumbel}` replaces gumbel_mcts,
> gumbel_variant and both completed_q flags; the root samples and
> completes over the FULL legal prior vector WITHOUT materializing every
> child (children exist only for sampled/visited candidates); interior
> cap K = 1024 with omitted-mass telemetry; within a halving phase the
> surviving candidates' descents are issued as ONE leaf batch (m, m/2,
> … leaves per round trip — the Gumbel analogue of leaf_batch_size,
> with no virtual loss needed across candidates); PCR under Gumbel; the
> deploy head runs the same kind as self-play; served-sims witness
> holds under both kinds, root evaluation charged, N means N leaves.
> (d) INVARIANTS every deletion is proven against: net-param hash on the
> warm-start; served-sims exactness; the suite's conformance sections;
> 1-in-1 collate checks; arena legality; finite-gradient guard; resume
> bundle round-trip; gate pair statistics; F-816-37 dump-on-fire;
> strength_floor; draw-rate abort. These are the PROTECTED SET; it is
> listed in LAWS.md and is the whole of what "ruling-protected code"
> means from now on.
> (e) GOVERNANCE moves to hexo-mantis/docs/governance/: LAWS.md (the
> standing laws, ≤ 60 lines), STATE.md (minted values, armed rows,
> protected set, open cards, current phase — rewritten in place),
> RULINGS.md (one entry per ruling, ≤ 10 lines, canonical), falsified.md
> (kept, deduplicated), CARDS.md. The old register, ACTIVE, sitting
> records and plan/ are FROZEN into docs/governance/archive/ (one
> directory, one README, no tooling). Census, stamp, mirror and
> sync tooling are DELETED — one repo needs no mirror. Numbering
> continues; canonical texts live in RULINGS.md; packets are not
> committed; exit facts go into STATE.md.
> (f) CODE AND CONFIG (wave 2): the grid/dense path is deleted (tagged
> `archive/grid-path` first); parked, unmeasured features are deleted
> with their keys (solver, forced-win, ZOI, seed corpus, mixing, bot
> buffers, aux/entropy weights, dense rotation, trace/compile/perf
> fields, dead keys named by AUDIT-2); operational constants get schema
> defaults and leave the YAML; the run YAML shrinks to what a run
> decides (identity, search, train, eval, the measured caps); the
> resolved complete config is still written to the run dir and is still
> strict. configs/ keeps run6 and one smoke profile. Comments: ≤ 2 lines
> unless stating a non-obvious invariant; no narrative, no ruling
> numbers, no banners; one-line docstrings on public APIs; a comment-
> length lint ratchets the count down and never up.
> (g) PERF-3 (wave 1, box, measure only): profile end to end at PUCT-50
> and at the (b) regime on a throwaway config — leaves/s, batch fill,
> queue wait, server thread share, per-row insertion, eval tree
> allocation, cache release per eval move — falsified.md read first;
> output is a ranked ledger with the Amdahl bound per item, no code.
> (h) Wave 3: AUDIT-3 (fresh deep audit on the cleaned tree, the
> AUDIT-2 method) → REPAIR-A3 → re-mint on the rehearsed procedure
> (rows: (b), K, sealbot-only rungs, supervisor grace reverted) →
> shakedown → START. STRENGTH-FRONTIER-1 stands as designed, at block
> end, with Gumbel cells now the live arm and PUCT-50 the comparison.
> (i) Ledger: leg 5's "r8 legal maximum" was written as if a constant;
> it is not (355 median, 8,142 max) — architect's error, the K decision
> above is its correction. The AUDIT-2 file the packets cited never
> existed in either repo; the operator files it.
> ROUTE: land in both homes; wave 1 (4 agents, worktrees); wave 2 (2
> agents, sequential); wave 3; one exit screen per agent, one STATE.md.

Grounds: operator direction. The era's object is stated in (a); the burden of proof moves from
removal to retention.
Status: standing — (i)'s AUDIT-2 filing is DISCHARGED (`docs/audits/AUDIT_2026-09-09.md`,
`428f3c8`), and (e)'s census/stamp/mirror/sync deletion is a verified NO-OP: that tooling never
existed in this repository.

---

### R23 — gumbel fields become schema fields
Decision: gumbel_m, gumbel_explore_moves and gumbel_mcts become first-class SelfplayConfig
schema fields, replacing the code-side defaults 16/10/false that were reachable only through
the legacy unvalidated dict path (hparams.py:198-200,318-320). A later coordinate note
corrects the ruling's own stale cite: SelfplayConfig lives at
src/mantis/config/schema.py:228-236, not at the schema.py:66-75 the text names.
Grounds: defaults reachable only via an unvalidated path are a second default authority (R1).
Status: annotated — R39 coordinate note (hparams cites verified, schema cite corrected).

### R24 — WP-AXIS2 scope and its v2 block
Decision: No R24 section exists in this register — the register's authority header states the
addendum carries R24, R29 and R32 as rulings with no section here, and ADDENDUM_A §1 is the
authoritative text. As R54 cites it, R24 scoped WP-AXIS2 and blocked its DESIGN until
inputs/run3_findings_v2.md existed on disk; R54 lifted that block once the file arrived.
Grounds: not stated in the register.
Status: standing — no section in the register; text lives in ADDENDUM_A §1.

### R25 — A9 closure: radius schedule becomes a constant, override chain deleted
Decision: Commit A replaces legal_move_radius_schedule with a scalar `legal_move_radius: int | None`
(null = registry value), deletes RadiusStage and resolve_radius_from_schedule, collapses
require_offline_radius to constant-vs-override precedence, and re-mints the configs. Commit B deletes
the entire radius_override runtime chain (pool hooks, bridge setter/sentinel, selfplay atomic and
per-game branch, core test); WP8-F4 is CLOSED-OBSOLETED. A coordinate note corrects the ruling's own
cites for RadiusStage, checkpoints and the bridge setter.
Grounds: zero production callers, recon-verified — dead weight is deleted, not documented.
Status: amended by the pre-authorized WPSC_RESUME "SC-A4 / SC-B1 (R25 latitude)" — DESIGN may
instead land no config field at all, with the registry as sole authority.

### R26 — run5 radius stays registry-derived
Decision: run5's radius stays derived from the registry (gnn_axis_v1 = 6). The number 8 is
never to be written into any config or doc in that run.
Grounds: not stated in the register (recorded as context only).
Status: standing — reaffirmed by R238's radius-resurrection watch (the ledger holds r=6, never 8).

### R27 — falsified.md F-04 scope limit
Decision: The F-04 register entry is amended by docs(design) commit to read that F-04
falsified PMA-as-tested versus min/max only, and does NOT certify min versus mean+max
attention. The min/max asymmetry — value aggregation takes the worst cluster view while policy
aggregation takes the best-scoring view — is a flagged defect preserved pending the
matched-FLOP dense arm, pinned by CARD-MINPIN.
Grounds: the asymmetry is a defect to preserve under test, not a settled result.
Status: standing — carries a coordinate note: the site is
crates/mantis-selfplay/src/runner/search_drive.rs:311, verified byte-accurate.

### R28 — dense-by-default fallbacks become hard raises
Decision: The dense-by-default "v6" fallback arms become hard raises — the arms at
src/mantis/encoding/resolvers.py:43-54, the 373-395 legacy default, and the
crates/mantis-bridge/src/board.rs:279-283 `to_tensor()` fallback. A missing or unspecified
encoding RAISES a named error; it never silently becomes v6.
Grounds: LAW-05 and LAW-11 — no dense-by-default, absent encoding is an error.
Status: standing — all three coordinates verified accurate at HEAD.

### R29 — (no section in the register)
Decision: No R29 section exists in this register. The register's authority header records that
the operator's ADDENDUM_A carries R24, R29 and R32 as rulings with no section here, and that
the addendum is authoritative for the R23-R31 range; no substance for R29 is reproduced
anywhere in the register text.
Grounds: not stated in the register.
Status: standing — no section in the register; text lives in ADDENDUM_A.

### R30 — seed determinism and a single amp authority
Decision: (a) `seed` is wired to real determinism — torch, numpy and stdlib random seeded once
at orchestrator boot from cfg.seed, with a behavior-named test proving two boots at equal seed
produce identical first-batch tensors on CPU. (b) amp dtype collapses to ONE authority on the
schema seam, merging the runtime pin in src/mantis/model/amp.py:20-38 with the schema-side
resolve_amp_dtype; graph path = bf16 remains law (fp16 banned on graph), enforced at that
single authority with a test. Coordinate note corrects the ruling: `seed` is RunConfig.seed at
src/mantis/config/schema.py:244, not `:82`.
Grounds: two default authorities over one dtype fact can disagree (R1/LAW-08, LAW-06).
Status: annotated — coordinate note corrects the `seed` cite; amp.py cite verified.

### R31 — host access requires a per-dispatch grant
Decision: Agent host access requires an explicit per-dispatch grant; provider aliases never
cross into hexo-mantis. No grant was given for the WPSC run.
Grounds: not stated in the register (recorded as a gloss).
Status: standing — satisfied per-dispatch later by R112 (WPBOX) and R142 (WP12-R), never by
precedent.

### R32 — (no section in the register)
Decision: No R32 section exists in this register. The register's authority header records that
the operator's ADDENDUM_A carries R24, R29 and R32 as rulings with no section here, and that
the addendum is authoritative for the R23-R31 range; no substance for R32 is reproduced
anywhere in the register text.
Grounds: not stated in the register.
Status: standing — no section in the register; text lives in ADDENDUM_A.

### R33 — entropy knob as a mode/value pair (omitted)
Decision: No R33 section exists in this register — it is deliberately omitted. The register's
Superseded block records what it said: the entropy knob typed as
`{entropy_mode, entropy_value}` with `-0.005` minted as the value.
Grounds: its premise was falsified by Phase 0 target T-C (evidence in WPSC ADJ-01).
Status: superseded by R37 — in full; no section in the register.

### R34 — value/policy target become typed knobs
Decision: value-target and policy-target become TYPED knobs with resolvers, minted at CURRENT
semantics only. Single-variant enums are acceptable and intended — the choice is explicit per
run, never a default. No lambda-returns and no Grill LEARN in this run. Recon resolved the
variants: value = `pure_outcome_z`, policy = `raw_visit_distribution`.
Grounds: the v2 §10 non-settlement law — an identity choice is stated per run, never defaulted.
Status: standing — carries a recon note supplying the two resolved variants.

### R35 — LICENSE is MIT
Decision: The repository LICENSE is MIT.
Grounds: not stated in the register.
Status: spent — DISCHARGED by commit `5bd1d70` on `wpsc-scratch`.

### R36 — scratch-branch commit exception (WPSC only)
Decision: For the WPSC run only, agents commit ONLY to a new scratch branch `wpsc-scratch` cut
from dev HEAD; dev and main are never touched. One chunk = one commit with a final-form
conventional single-line subject, no trailers and no Co-Authored-By; doc/contract amendments
are separate docs(design) commits. The operator cherry-picks onto dev after review and the
scratch branch is then deleted. Every commit boundary must be gate-green BEFORE the commit.
Grounds: not stated in the register.
Status: superseded in part by R52 (its cherry-pick language, in favour of ff-merge); its
one-run exception is generalised into the standing pattern by R47.

### R37 — entropy knob typed as one non-negative float
Decision: The entropy knob is typed `entropy_reg_weight: float`, constrained non-negative at
schema (a negative value raises a named ValueError); there is NO mode enum and
max_entropy_fraction is descoped. Every minted config carries `entropy_reg_weight: 0.0`
explicitly. Documented sign law: a positive coefficient on a subtracted entropy BONUS, so
larger means more exploration pressure — the historical "-0.005" was a sign-leak from the loss
formula that never existed in this tree. GRAPH_FORBIDDEN_NONZERO_WEIGHTS stays untouched as the
sole graph-ban authority; core.py:120's default dies; CARD-A10-CAP is recorded, not executed.
Grounds: resolves ADJ-01 — R33's premise was falsified by Phase 0 target T-C.
Status: standing — supersedes R33 in full.

### R38 — temperature_schedule debt row is void at HEAD
Decision: The temperature_schedule debt row is VOID-AT-HEAD. Read the two named test functions
(tests/selfplay/test_pool_hparams_arms.py:196, tests/selfplay/test_pool_hparams.py:145): if
they assert real behavior of playout_cap.temperature_threshold_compound_moves plus temp_min
under a misleading name, RENAME them to behavior-named form; if they reference nothing real,
DELETE them with a log row. No new schema entity is invented either way.
Grounds: the row is void at HEAD, so the only live question is whether the tests pin real
behaviour.
Status: standing — resolves ADJ-02; amends SC-B7.

### R39 — recon reuse and Phase 1 ratification
Decision: Phase 0 recon is COMPLETE and REUSABLE — targets T-A..T-F are not re-run, and stage
prompts carry the CORRECTED coordinates from TARGET_RECON_REPORT.md §T-E plus the verbatim
debt rows from DEBT_DOSSIER.md, never the stale §R landmarks. Phase 1 is COMMITTED and
RATIFIED (`wpsc-scratch` = 5bd1d70, parent f1ad10f), including deviation 1 (running Phase 1
with ADJ-01 known) and deviation 3 (compressed pipeline), with a rider: compression is
permitted only for zero-behavior metadata commits and must always be disclosed.
Grounds: not stated in the register.
Status: standing.

### R40 — SC-B4 validator identifiers
Decision: Two named validators replace SC-B4's "A2/A8" citations. V-NOOP: every schema key must
be consumed by exactly one resolver/consumer, extending the existing O15 bijection test to all
fields added in the run — a key no resolver reads fails the test suite, not silently. V-PCR:
the playout-cap-randomization validator rejects full_sims == quick_sims (a no-op
"randomization") and degenerate full_fraction (<=0 or >=1 when both presets are set), with
named errors. The identifier "A2" is not used for validators anywhere in stage prompts.
Grounds: not stated in the register.
Status: standing — replaces SC-B4's "A2/A8" citations.

### R41 — optional-deps extras
Decision: The optional-deps extras group declares matplotlib, rich and scipy ONLY, each backed
by a recon-cited import site. structlog is imported nowhere and is NOT declared.
Grounds: every declared dependency must have a live import site.
Status: standing — ratifies HANDOFF-4.

### R42 — gate-evidence timestamps precede commits
Decision: Gate-evidence timestamps MUST precede the commit timestamp, and both are recorded in
COMMIT_MANIFEST.md. A commit made before its gate output was read is a breach. WPUF-2's Phase
M.4 extends the same discipline to a merge: R42 applies to the merge exactly as to a commit.
Grounds: the ordering is mechanically checkable, so an unread gate cannot be claimed as
evidence.
Status: standing [INLINE]

### R43 — frozen-oracle edits always queue, regardless of direction
Decision: Once an oracle suite is byte-frozen, ANY edit to it — whether it would tighten or
loosen the assertion — goes to ADJUDICATION_QUEUE.md rather than being made. Only the
parenthetical gloss survives on disk; the full text is owed by the operator, and a dispatcher
must not expand this ruling beyond that gloss.
Grounds: oracle-contradicts-design is ADJUDICATED, never satisfied (WPUF Global Rule 3).
Status: standing [SUMMARY-ONLY]

### R44 — CARD-FRESHSYNC; main-gate CI non-binding until fixed
Decision: Fix gate 1's fresh-clone sync, broken since WP7 (`_engine.hello()` failure), and
answer IN THE LOG why nothing surfaced it for the whole migration — whether the gate was not
run, or run and its result ignored; both the script and the CI wiring are fixed if both are
implicated. Until that commit lands, main-gate CI evidence is treated as non-binding; after
it, run the fresh-clone gate once end-to-end and record the evidence.
Grounds: the why-answer decides whether the repair is the script or the CI wiring.
Status: amended by R57 — the non-binding-CI clause becomes permanent by ruling, with the local
serialized sweep as the authority rather than CI. [INLINE]

### R45 — silent-v6 arm closed plus a pattern gate
Decision: Convert the fifth known silent-v6 fallback arm in src/mantis/train/pretrain/
validate.py to the established named-error pattern (MissingEncodingError / PanicException — no
new convention invented), and in the SAME commit add a repo-wide grep gate to the CI gate
family failing on the silent-fallback pattern class, its pattern set derived from the known
instances and enumerated in its header comment; escapes need a site justification.
Grounds: so a sixth silent-fallback arm cannot exist quietly.
Status: amended by WPUF-2 phases R1-R3 — the census is answered against post-merge dev HEAD,
and if the site is already closed R45 reduces to the pattern-gate half, with the header
enumerating all closed arms however many there turn out to be. [INLINE]

### R46 — flaky-test law: deflake or loud-quarantine
Decision: Deflake if the root cause is reachable within one revision loop; otherwise quarantine
LOUDLY — skip-with-named-reason, a debt-ledger row appended to DISPATCH_LOG handoffs, and the
test-count floor adjusted to the measured post-quarantine count with the quarantine named in
the commit message. Never a silent skip.
Grounds: a flaky test inside the floor count is a floor that lies.
Status: standing [INLINE]

### R47 — scratch-branch standing pattern
Decision: All commits go to a scratch branch cut from dev HEAD at entry (`wpuf-scratch` for
that run); dev and main are never touched. One chunk = one commit with a conventional
single-line subject and no trailers; doc/contract amendments are separate docs(design)
commits.
Grounds: generalises WPSC's one-run R36 exception into the standing pattern for every
subsequent WP.
Status: standing [INLINE]

### R48 — WP-UNFREEZE unblocked from run3_findings_v2; sources enumerated
Decision: run3_findings_v2.md is ABSENT and is NOT required — WP-UNFREEZE does not depend on
it, and any DESIGN question genuinely needing its section-level text goes to
ADJUDICATION_QUEUE.md rather than being guessed. The run's on-disk sources are enumerated:
STATE_2026-07-24.md §4 row A1/§5/§6, the rulings register, WPRECON RECON_REPORT §T5, WP11A
DISPATCH_LOG, and WPSC's TARGET_RECON_REPORT + DEBT_DOSSIER.
Grounds: not stated in the register.
Status: standing — carries a dated status note (2026-07-26): v2 has since ARRIVED on disk, so
the premise no longer holds while the ruling stands unchanged; v2 enters only through Phase A's
A-0 verify-read under R54. [INLINE]

### R49 — continuous actor sync is law; the old mode is unrepresentable
Decision: run3's gating deadlock leaves the tree. Actor weight sync becomes CONTINUOUS and
unconditional, and the promotion gate controls ONLY the deploy tag; the two seams have no
cross-read and the old sync-on-gate behavior is not representable. The enforcement rider bans
a mode knob — a config key that could restore sync-on-gate is itself an R49 breach.
Grounds: under the gated sync, run3 ran roughly 39% of training on a stale actor.
Status: standing [INLINE]

### R50 — first sanctioned behavior change; change-list discipline
Decision: DESIGN must enumerate every existing test from CENSUS (e) that pins the old behavior,
each with its disposition — rewrite to pin the new law, or delete with grounds. That list is
reviewed in REVIEW-design, and IMPL may not touch a pinned test that is not on it.
Grounds: WP-UNFREEZE is the migration's first sanctioned behavior change, and the change-list
is what keeps "sanctioned" from becoming "unbounded".
Status: amended by R128 — the R50 list is AMENDED after CENSUS.md §4's false negative is
corrected in every downstream artifact, and a subprocess-shape census law is added. [INLINE]

### R51 — minimal artifacts at entry-gate stops
Decision: When a hard entry gate fails, the dispatcher writes ONE queue entry and nothing else
— no scratch branch cut, no baseline sweep, no CENSUS, no tree mutation. Only the
parenthetical gloss survives on disk; the full text is owed by the operator.
Grounds: not stated in the register (gloss only; the compliant instance is WPUF ADJ-01).
Status: standing [SUMMARY-ONLY]

### R52 — ff-merge over cherry-pick, to preserve shas
Decision: Stacks are merged by fast-forward rather than cherry-picked, so commit shas are
preserved byte-identically.
Grounds: cherry-picking the WPSC stack would rewrite all 19 shas and invalidate every sha in
COMMIT_MANIFEST.md and every gate-evidence row bound to them.
Status: standing — supersedes the cherry-pick language in R36/WPSC_dispatch and in WPUF
Global Rule 2. [SUMMARY-ONLY]

### R53 — mechanical merge authority for WPUF-2 Phase M
Decision: A narrow one-run grant to act directly on dev: verify dev HEAD == f1ad10f and
`wpsc-scratch` tip == 101a9d3 with an exact parent chain; full pre-merge gate sweep;
`git merge --ff-only` (ff-only is LAW — never resolve, rebase or re-pick, queue and STOP); full
post-merge sweep confirming the floor file at 1535; both sweeps timestamped under R42; neither
scratch branch deleted, LICENSE untouched, WPSC ledger rows appended.
Grounds: the authority is mechanical, so every listed failure mode is a queue-and-STOP rather
than a judgment call.
Status: spent — a one-run grant that does not extend past Phase M; it replaces WPUF Global
Rule 2's entry gate and resolves ADJ-01. [INLINE]

### R54 — WP-AXIS2 stretch phase plus the v2 verify-read rider
Decision: Phase A (WP-AXIS2) runs as a STRETCH only if Phase U lands fully green. A-0 is a
find-only verify-read extracting run3_findings_v2 §4.1/§4.2 verbatim into V2_EXTRACTS.md under
a discrepancy law where STATE plus rulings WIN over the raw file and any contradiction queues.
A-1 designs a new registry entry `gnn_axis_v2` beside `gnn_axis_v1` — radius/win_length as
data-only values, rays=7 as GATED new code behind versioned dispatch — with the v1 path
byte-frozen and its parity suite untouched. Out of scope: any training run, any v1-vs-v2
strength claim, and dense-path pooling changes. Phase A drops first if the run runs short.
Grounds: unblocks R24's block on WP-AXIS2 DESIGN now that run3_findings_v2.md exists on disk.
Status: standing [INLINE]

### R55 — anchor wiring is chunk U-w, not a separate card
Decision: Anchor wiring is NOT a separate card. It is chunk U-w, the first commit of Phase U,
restoring run3 sync-on-gate semantics as the reference state, which subsequent U chunks replace
per R49. Riders: a port_ledger defect row for the parity gap; re-verify the R50 change-list
against CENSUS before freezing oracles; end-state oracles freeze unchanged. CARD-ANCHOR-WIRING
is therefore not created and the xfail(strict=True) regression-arming oracle is not landed.
Grounds: attribution windows are run-scoped and no run intervenes, so there is no window to
confound; bisect isolation is preserved by the chunk commit.
Status: standing — resolves ADJ-04, overturning both the dispatcher's recommendation and
DESIGN §0's endorsement of it. [INLINE]

### R56 — ten-arm count accepted; arm 8 registered-open
Decision: The ten-arm silent-fallback count is accepted. Arm 8 remains registered-open under
WP12-R with two riders: the gate-11 exemption comment cites the WP12-R handoff row, and a
committed test proves arm 8's reachable paths either fail loud or provably match the configured
encoding. If that test cannot be written, arm 8 escalates to a hard run5-mint blocker.
Grounds: not stated in the register.
Status: superseded — R228's merge append (2026-08-04) records R56 as SATISFIED-AND-SUPERSEDED:
arm 8 closed in `29f304b`, and the exemption comment and its debt row no longer exist. [INLINE]

### R57 — the local serialized sweep is the binding pre-cutover gate
Decision: The local serialized gate-sweep protocol with R42 timestamps is the BINDING
pre-cutover gate, recorded in the register. ruff and pyright are demoted to advisory in one
loud `ci:` commit carrying debt row CARD-LINT-TYPE (WP-R family). Remote plus green Actions
becomes an explicit cutover-battery item — satisfied or operator-waived.
Grounds: CI is not the authority; the local serialized sweep is.
Status: standing — resolves ADJ-02 and makes R44's non-binding-CI clause permanent by ruling
rather than by accident. [INLINE]

### R58 — WPUF-2 disclosures ratified; sweep serialization becomes law
Decision: The gate-11 rebuild is ratified, with the 31-evasion corpus committed as fixtures if
not already; registry_sha_hex gets an honest re-justification written in the log, and a
fix-commit if the choice fails it. Standing law: gate sweeps run serialized under exclusive
load, and a sweep taken under concurrent load is INVALID — not "interpreted with caution",
invalid.
Grounds: three of this run's sweeps were contaminated by a concurrently running load generator.
Status: standing [INLINE]

### R59 — run5 arms the actor-lag abort now, not at mint
Decision: run5.yaml arms the actor-lag abort immediately rather than at mint. Deliberate
disarming stays legal for smoke configs, while the minted production config defaults to armed.
Draw-rate arming is audited, not blind-flipped.
Grounds: with the abort disarmed a frozen actor emits an event and nothing else — the
silent-failure mode that cost run3 the run.
Status: standing — the audit clause surfaced that draw_rate_threshold had no config key at all
(a code-side StepCoordinatorConfig default), filed as ADJ-08 and resolved by R65. [INLINE]

### R60 — CARD-SMOKE-SEAM is one card, one design
Decision: The smoke-arm collapse, the ADJ-07 named error and the F-C re-anchor are ONE card
with one design, and the frozen composition oracle's renegotiation is pre-planned under R43
rather than tripped over mid-phase.
Grounds: deleting the smoke arm reds 13 tests across 4 files including a byte-frozen oracle,
and discovering that on the last available loop is what forced the deferral.
Status: standing [INLINE]

### R61 — run5 mint gains a hard preflight gate
Decision: run5's mint gains a hard preflight gate — the actual minted config booted through the
REAL composition root with REAL run-safety, asserting live sync events, live lag sourcing, and
an arming audit against the committed armed-abort manifest.
Grounds: production-only axes get production-time checks — the two axes no test can close
(_default_step_coordinator_config and build_run_safety) are varied at production time instead.
Status: standing — tightened by R64 (real model, real eval_enabled, boot walls are tree
defects) and its manifest content settled by R65's Option B. [INLINE]

### R62 — CARD-EVAL-CLOCK
Decision: The 90-second wall-clock eval-round tests are R46-class; inject a clock or
restructure them, pre-cutover. Recorded as an owned card, explicitly not in the WPAX run's
scope.
Grounds: the two tests carry a fixed 90s deadline with no injected clock and are the ones that
went red under contaminated load.
Status: standing [INLINE]

### R63 — staging is explicit paths, never `-A`
Decision: Git staging uses explicit paths, never `git add -A`.
Grounds: an `-A` sweep pulled WP14's R11-protected dirty contract doc into a Phase U commit —
there is no situation in which `git add -A` is the right command in this repo.
Status: standing [INLINE]

### R64 — Phase P posture: full production reality
Decision: The preflight runs in full production reality — real `build_net` model, real
`build_run_safety`, real coordinator config, and `eval_enabled` at the minted config's own
value. `eval_enabled=False` is BANNED as an escape. Any wall the real boot hits
(`terminal_eval`, `.arch`, or new) is a TREE DEFECT: fixed in-run if small and within loops,
queued if not — never designed around inside the tool.
Grounds: the preflight's job is to hit exactly these walls before run5 does.
Status: standing — tightens R61 rather than restating it. [INLINE]

### R65 — ADJ-08 resolved: Option B, plus a new Phase D
Decision: Option B is ratified — the armed-abort manifest carries typed `status` rows and
DEFERRED rows print loudly. A NEW Phase D lands after P and before A as CARD-DRAWRATE-KEY:
`draw_rate_threshold` and its arming become typed config through one resolver, the code-side
0.0 literal dies, run5 is re-minted armed, and the manifest row flips DEFERRED to required.
Phase priority is P > D > A.
Grounds: B is cheap to reverse into C, and a code-side default must not survive as a second
default authority beside the schema field (R1/LAW-08).
Status: standing [INLINE]

### R66 — ADJ-09 resolved: Option B; WP14 promoted
Decision: Option B is ratified — land S-4, amend repo_design.md, and the contract-doc update is
owed to WP14. WP14 is promoted to the next run, and R11 (the bar on touching
docs/contracts/run_config_schema.md) dissolves there.
Grounds: the R11 versus §4 precedence question is answered by expiry — the conflict cannot
recur past WP14.
Status: standing [INLINE]

### R67 — pre-authorized bounded R43 event
Decision: A bounded R43 exception is pre-authorized, landable any time after Phase P under §S-3
discipline (enumerated hunks, before/after hashes, dispatcher-verified, ONE commit): fold the
F-2 census into tests/train/test_actor_lag_watchdog.py — frozen at 5638b90db43866e6 — and
correct N-3's three stale sentences in that file, none of which any assertion depends on.
Grounds: with the census sited in a non-frozen sibling, deleting it and restoring the F-2
defect together yields zero failures — only the count floor catches it.
Status: standing — ratified as landed by R73, which adds the name-truth rule on top. [INLINE]

### R68 — subagent scratch hygiene; sweeps assert free disk
Decision: Every subagent brief mandates scratch cleanup, and every sweep asserts and records
free disk before starting; a sweep with no recorded free-disk figure is missing evidence, not
merely undocumented.
Grounds: a sweep had to be discarded because the box ran out of space under accumulated
per-stage scratchpad copies.
Status: amended by R76 — sweeps measure and record RAM (/tmp is a tmpfs, so free disk was the
wrong quantity), tree copies are barred from tmpfs, and subagent harnesses run single-worker
unless the brief grants otherwise. [INLINE]

### R69 — every "verified under X" claim names its mechanism
Decision: Every "verified under X" claim names its producing mechanism or is struck, and
repeating an unproduced claim is the same violation as making it — a reviewer inherits no
immunity from having read it upstream.
Grounds: three reports claimed the tier green "under random order" when no random-order plugin
was installed, so `-p no:randomly` was a no-op over one deterministic run.
Status: standing [INLINE]

### R70 — commit-on-FAIL ratified under a six-part standing test
Decision: Phase P's commit on a failed RED-TEAM is ratified, and committing on a failed
RED-TEAM becomes legal only when ALL six hold: (1) measured strictly-better-than-absent,
(2) findings latent not live, measured, (3) FAIL plus caveat disclosed into the artifact's
consumer surface, (4) scratch not dev, (5) loop budget exhausted, (6) residue adjudicated. The
caveat blocks any mint use of gate 12's rc or a preflight rc until it is discharged.
Grounds: all six held for this instance — `configs/` was measured to hold five top-level .yaml
files with no .yml and no subdirectories, so F-1's escape had no live instance.
Status: standing [INLINE]

### R71 — ADJ-13 authorized in full; the class-fix law
Decision: ADJ-13 is authorized in full (F-1, F-2, F-3, F-5, F-6 plus three nits) under R67's
discipline with a delta-scoped adversarial recheck rather than a full RED-TEAM. Class-fix law
(MF-7): every fix names its class, and flip-sets cover the CLASS BOUNDARY, not the demo input.
F-1's audit set derives from the loader's own discovery authority — one authority — with .yml,
subdirectory and novel-extension flip rows.
Grounds: the demonstrated fix closed `run6.yaml` while `run6.yml` and `configs/prod/` walked
straight through.
Status: standing [INLINE]

### R72 — per-conjunct flip-set coverage becomes standing law
Decision: Every conjunct of every shipped predicate must appear in some flip-set; this applies
to the ADJ-13 delta and to all subsequent phases.
Grounds: flip-sets were verified sufficient for the anti-shotgun property but never claimed
corpus completeness — the gap F-5 and F-6 fell through.
Status: standing — its instrument's disposition is settled by R77 (kept as evidence, plus a
lightweight repeatable gate-family check). [INLINE]

### R73 — R67 ratified; name-truth becomes law
Decision: R67's Option A, its unforecast fourth hunk (`import pytest`) and the rename are all
RATIFIED. Standing: name-truth is part of behavior-naming law, and a bounded frozen-edit grant
implicitly includes renames made necessary by the granted change, disclosed in the same event.
Grounds: a test name is a behavioural claim, so a name made false by a granted edit must be
corrected within that same edit.
Status: standing [INLINE]

### R74 — WP-R §9.10 CLOSED
Decision: WP-R §9.10 is CLOSED — the defect was closed in fact by revalidate_run_config in
Phase S, validated by RED-TEAM-2's 147-leaf model_dump fidelity sweep. The residue stands as
originally booked: model_construct fails loud at use.
Grounds: R67's item 2 struck the sentence that kept the card alive citing a hole that no longer
existed.
Status: standing [INLINE]

### R75 — the loader accept-set narrowing is DECLINED
Decision: The narrowing is DECLINED, reversing a production change that had already landed in
`4d11147`: `load_config`'s refusal of suffixes outside CONFIG_SUFFIXES comes OUT, discovery
widens so the shared-authority invariant holds (whatever the loader accepts, the audit must
see), constants without a live consumer die, and the tests asserting the refusal invert. Parked
as a WP-R row; reopening requires a concrete escape the invariant misses.
Grounds: the invariant closes the same class from the other side without constraining what a
run may be launched from, and the preflight covers the mint path shape-agnostically.
Status: standing — the only ruling in that run to reverse an already-landed production change.
[INLINE]

### R76 — R68 amended: measure RAM, not disk
Decision: Sweeps measure and record RAM; tree copies are barred from tmpfs; subagent harnesses
are single-worker unless the brief explicitly grants otherwise.
Grounds: /tmp is a tmpfs — it is RAM — so every artifact recording "16G available" was
measuring the wrong quantity while the box reached global OOM.
Status: standing — amends R68. [INLINE]

### R77 — the R72 instrument stays; a gate-family check opens
Decision: The R72 conjunct-coverage instrument stays in wp/WPAX/ as evidence, and a WP-R row
opens conjunct-coverage as a lightweight, repeatable gate-family check under the explicit
constraint "build small or don't build".
Grounds: the size constraint forecloses a coverage instrument growing into a second test
framework.
Status: standing [INLINE]

### R78 — Phase D authors exactly ONE knob
Decision: Phase D authors exactly one knob. The roughly 24 code-side coordinator knobs become
CARD-COORD-KNOBS — pre-run5-mint, its own card — whose first design question is a
preflight-JSON dump of the resolved coordinator config, making the unauthored values visible in
the mint record before deciding which become config.
Grounds: forecloses the scope creep the `_default_step_coordinator_config` seam invites.
Status: amended by R80 — "one knob" is clarified to the draw-rate abort FAMILY (threshold +
min_step + min_samples, one block, one resolver), not one literal field. [INLINE]

### R79 — single-authority arming; no boolean proxy beside a gating value
Decision: No boolean enable proxy may sit beside a gating value. Armed versus disarmed is a
property of the RESOLVED VALUE with explicit off-semantics — the off state is a value an
operator writes deliberately, never a default that happens to disable; the manifest row asserts
the resolved-value condition rather than the existence of a schema field; and the pin and the
manifest bind to the same fact.
Grounds: `draw_rate_threshold` already gates its own check, so a boolean beside it would be a
second authority over one fact and could contradict it (armed-by-boolean, disarmed-by-value).
Status: amended by R83 — the shape is fixed as None = disarmed (explicit), present value gt=0
and le=1. [INLINE]

### R80 — ADJ-14 closed; R78 clarified to the abort FAMILY
Decision: Recommendation A is strengthened-adopted: the draw-rate estimator defect closes with
BOTH guards — min-sample inclusion (`len(dq) >= min_samples`) and a non-zero `min_step`. R78 is
clarified: Phase D authors the draw-rate abort FAMILY (threshold, min_step, min_samples) as one
block with one resolver — not one literal field, and not the coordinator-knob card.
Grounds: instrumentation.py:365-371 included any worker with `len(dq) > 0`, so a single drawn
game per worker saturated the pool mean at 1.0 — the hazard is the inclusion rule, not the step
count, which a min_step guard alone could not fix.
Status: standing — clarifies R78; its per-worker min_samples guard is later DELETED by R92 in
favour of a pooled count-weighted statistic. [INLINE]

### R81 — ADJ-15: the narrow R43 grant is GIVEN, with conditions
Decision: One assertion may be re-pointed from "deferred row prints" to "deferred mechanism
works" via a synthetic manifest row; `_print_deferred_rows` survives. Conditions: §S-3
discipline, and a mutation self-test showing that killing `_print_deferred_rows` reds the
re-pointed test alone. Keeping a row deferred so the assertion stays true is REJECTED.
Grounds: without the mutation proof, re-pointing from real behaviour to a synthetic subject can
quietly turn a live pin into a self-satisfying one; shaping the shipped manifest to suit a test
inverts the instrument.
Status: standing — its "alone" condition is settled by R86 (not self-satisfying, no unrelated
casualty). [INLINE]

### R82 — run5 arms at 0.25
Decision: run5 arms at `draw_rate_threshold = 0.25`. `min_step` and `min_samples` are DESIGN's
to propose from measured deque geometry with stated grounds. All three are pre-registered at
mint prereg — the only place they may change.
Grounds: healthy draw rate is about 0.025% (ply-cap truncations only), so 0.25 sits four orders
of magnitude above healthy and well below a real collapse, with R80's guards preventing
estimator saturation.
Status: standing — 0.25 CARRIES OVER on unchanged basis under R92, amendable only by the
operator via prereg; the hard stop on these values binds all other actors. [INLINE]

### R83 — R79 amended; the named RED extended
Decision: The arming shape is fixed: None = disarmed (explicit), and a present value must be
gt=0 and le=1. The named RED extends to a builder-signature assertion (closing the
default-migration route the change list itself creates) and to `__post_init__` /
`object.__setattr__` resurrection of a frozen dataclass field. N-1's two pin-scan tests join
the enumerated change-list.
Grounds: `le=1` kills the "armed in the config, absent in effect" route a percent/fraction unit
slip creates, and the pin-scan tests would otherwise have been gutted silently by the flip.
Status: standing — amends R79. [INLINE]

### R84 — ADJ-16 ratified; CARD-ABORT-EXIT opened BLOCKING
Decision: `exit_code=None` is RATIFIED as truthful now. CARD-ABORT-EXIT opens PRE-RUN5-MINT and
BLOCKING: author the supervisor-visible mechanism (presumptively a registered exit code with
fail-fast family parity, but the card justifies its own choice), update the contract doc in the
SAME commit per R9, flip the manifest row from None to the authored value, and prove by
mutation that a fired abort is supervisor-distinguishable from a clean run.
Grounds: a fabricated exit code in a manifest a mint reads would be an unproduced number; the
mutation clause stops the card being satisfied by a number nothing produces.
Status: standing — the second run5-mint blocker, beside CARD-COORD-KNOBS. [INLINE]

### R85 — the draw-rate guard values accepted, with a Stage-0 revisit
Decision: The guards are ACCEPTED — `min_samples = 50` (equal to `_DRAW_RATE_WINDOW`, with an
`le=` pin and the 51-counterexample recorded) and `min_step = 25000` (minted precedent, plus
the missed-abort-is-cheaper asymmetry). Prereg note: `min_step` is revisable at mint prereg
once a real boot measures the early-run draw distribution, under Stage-0 discipline.
Grounds: accepted with the evidence gap named — min_step came from minted precedent rather than
a measured early-run distribution, so it is not frozen by being accepted.
Status: standing — its per-worker `min_samples` is later DELETED by R92 when the statistic
becomes pooled. [INLINE]

### R86 — R81's "alone" condition settled
Decision: R81's mutation condition means not self-satisfying and no unrelated casualty. A
five-node family casualty caused by re-basing a parametrized test is in-subject and therefore
satisfies the condition. Ratified.
Grounds: the census — one call site, no test naming the function, no unrelated casualty — is
what satisfies it.
Status: standing — settles R81. [INLINE]

### R87 — ADJ-17: the second bounded R43 grant is GIVEN
Decision: GRANTED under §S-3 discipline against 0f42484c1ce1c980, with hunks extracted verbatim
from the design's fenced block and R81's condition read per R86 (mutation-red per re-pointed
assertion, no unrelated casualty). A vacuous assertion is IN the enumeration: re-point it to a
live subject or delete it with grounds, never leave it passing emptily. The
land-keys-defer-flip is REJECTED.
Grounds: the manifest is a mint-read artifact and does not assert dead deferrals; an assertion
that goes vacuous is as much a hunk as one that goes red.
Status: standing — the last pre-authorized R43 exception of that run. [INLINE]

### R88 — DESIGN owes a frozen-impact census from Phase A onward
Decision: Standing from Phase A onward, DESIGN's change-list includes a frozen-impact census:
drive the flip against stand-ins, declare every frozen file touched, and request the R43 grants
IN the design. Binding boundary condition on Phase D's IMPL: the post-IMPL perimeter re-hash
must account for every frozen file, with each drift traced to its numbered grant beside the
hash — no bare hash lists.
Grounds: R43 fired twice in one phase and each correct stop cost a round trip because DESIGN
under-measured the blast radius; two reactive stops become one review-time adjudication.
Status: standing [INLINE]

### R89 — Phase D ratified, its compression residual SCHEDULED
Decision: Phase D is RATIFIED with its compression residual SCHEDULED, not accepted: its
RED-TEAM was dispatcher-driven with no fresh-context adversary and zero downstream adversarial
stages. The next dispatch opens with a delta-scoped fresh-context adversarial recheck of Phase
D's arming surfaces (Phase DR), and Phase D stays PASS-WITH-DISCLOSURE until that recheck
returns a clean verdict.
Grounds: nothing adversarial and fresh-context ever examined Phase D's arming surfaces.
Status: amended by R92 — the recheck returned FINDINGS, so the lift condition becomes DR-FIX
plus Phase DS passing a narrow adversarial re-verify against DR's battery; the caveat is
finally lifted by R97. [INLINE]

### R90 — delegation package for the WPMINT run
Decision: The dispatcher proceeds without operator round-trips where law is settled:
(a) settled-class frozen-oracle grants auto-granted under S-3 discipline plus the R81/R86
mutation condition, recorded as a queued adjudication would be; (b) commit-on-FAIL applies
R70's six conditions directly; (c) one third revision loop per phase, pre-authorized iff loops
1-2 converged and the fix scope is enumerated; (d) phase reordering with recorded grounds;
(e) STOP-class items halt only phases sharing their surface. HARD STOPS: new adjudication
classes, any dev mutation, scope widening, any change to run5's armed values (0.25/25000/50).
Grounds: not stated in the register.
Status: standing — renewed unchanged by R108 (WPCLEAN) and R119 (WPMAIN + WP12-R). [INLINE]

### R91 — WP14 mechanics; R11 dissolves
Decision: R11 dissolves inside the WPMINT run. The dirty docs/contracts/run_config_schema.md
working copy is WP14's INPUT — curated against the shipped tree and committed, never discarded.
From WP14's commit the tree-clean invariant is FULLY clean, and every gate asserting the R11
exception drops it in the same commit; ADJ-09's owed contract-doc update and all R9
same-commit debts settle in WP14. WP14's design question: a contract-doc drift gate deriving
from the live schema authority (the gate-12 pattern, never a parallel copy) — build small or
record why not.
Grounds: not stated in the register.
Status: standing — dissolves R11. [INLINE]

### R92 — ADJ-18/19 resolved; the pooled statistic authored; Phase DS opened
Decision: The draw-rate statistic becomes a pooled count-weighted rate (sum of draws over sum of completed, across the
union of worker windows). Below `N_pool_min` there is NO observation, never a healthy 0.0; zero-completion starvation
belongs to the stall family. `min_step` 25000 and `consec` 3 stand; per-worker `min_samples` is DELETED; `N_pool_min`
comes from DESIGN's measured geometry; threshold 0.25 carries over. Lands as Phase DS after DR-FIX, DR's
counterexamples (0.968 fires, 0.0319 silent) becoming permanent oracles.
Grounds: one mechanism kills both directions — count-weighting stops one worker carrying the mean, and removing the
inclusion bar means no worker can be excluded into invisibility.
Status: standing — amends R89's lift condition, dissolves ADJ-20, and supersedes R80/R85's per-worker min_samples.
[INLINE]

### R93 — drain-keys finding routed to Phase K, with conditions
Decision: RATIFIED with conditions — Phase K wires the four drain keys for real; every
consumer-registry citation K touches is verified by MUTATION (set the knob, observe the
consumer), never by grep; the SC-A3 "hardcaps wired" ledger row is corrected; and the
false-citation class is logged for WP-R so siblings are hunted rather than these four treated
as isolated. DR-6's `_leaf_paths` optionality fix is settled-class in DR-FIX.
Grounds: a registry string named a consumer path that `data.pop("drain")` had already
discarded, and a grep-verified citation cannot distinguish a reader from a pop.
Status: standing [INLINE]

### R94 — ADJ-21 ratified; the draw-rate floor validator stands
Decision: The floor validator stands. A sub-floor threshold requires a deliberate docs(design)
schema amendment, not a config edit, and the docstring must cite R94 and its basis so the next
reader finds the grounds at the constraint.
Grounds: healthy draw rate is about 0.00025, so the 0.02 floor sits roughly 80x above healthy —
far enough that no legitimate posture is lost, close enough not to be an arbitrary wall.
Status: standing [INLINE]

### R95 — validators assert only what they OBSERVE
Decision: The config-time check re-scopes to config-domain facts and surrenders the
reachability claim to runtime, where R92's no-observation rule already owns it; a validator's
name and message may assert only what its inputs can witness. It lands as DSV-2 on
wpmint-scratch with a confirmatory narrow adversarial pass against the post-fix state; Phase
D's caveat lifts on DSV-2's clean verdict; MERGE AFTER DSV-2, not before.
Grounds: a load-time validator cannot see which workers will report, so asserting reachability
was an overclaim no config-time arithmetic could make true.
Status: amended by R97 — its "clean verdict" condition is restated in place to its intent
(mechanism clean under adversarial re-verify, residual classes enumerated and covered). [INLINE]

### R96 — artifact-correction law
Decision: A withdrawn finding is corrected in EVERY artifact downstream agents consume, not
just in the ledger. DSV-2 verifies that the two production files carrying the DR-8-derived
claim now state the grepped truth, and any correction that did not land rides the DSV-2 commit
rather than being assumed.
Grounds: DR-8 was withdrawn in the ledger but not in RECHECK_D.md, so the next phase read it,
believed it, and shipped the false sentence into armed_aborts.py and coordinator/config.py.
Status: standing [INLINE]

### R97 — Phase D caveat LIFTED
Decision: The Phase D caveat is LIFTED and Phase D is PASS, full — mechanism clean under DR's
battery plus two consecutive fresh-context adversarial passes, with the residual
rationale-prose class enumerated at 286-claim coverage, fixed, and re-verified 0/30 and 0/18 in
the previously-failing sub-populations. R95's condition is amended in place to its intent.
Recorded explicitly: this is relocation-and-coverage of the failure surface, not attrition — no
unchanged surface was rerun for a green word.
Grounds: the lift rests on measured coverage rather than fatigue.
Status: standing — amends R95 and retires the PASS-WITH-DISCLOSURE label R89 attached and R92
re-scoped. [INLINE]

### R98 — CARD-LINT-GATE ratified, owned by WP-R
Decision: RATIFIED with constraints — a curated rule set seeded from rules that caught a real
defect here (the F601 class first), never the advisory backlog wholesale (that stays
CARD-LINT-TYPE's debt); no gate over a known-dirty baseline; a mutation self-test is mandatory.
Rider: derive-or-delete for mechanical facts in rationale prose, plus gate-13 citation
discipline wherever a fact is derivable.
Grounds: ruff's 20 x F601 on a duplicated registry block was readable for four commits while no
gate read it, and 17 of 17 residual findings were mechanical facts asserted in prose.
Status: standing [INLINE]

### R99 — WPMINT merge authorized
Decision: MERGE AUTHORIZED — the 10-commit wpmint-scratch stack onto dev on the standard
runbook (verify stack, quiet box, double sweep, ledger and register appends, scratch deleted).
The caveat ledger is empty and the remaining pre-mint items are operator acts, not engineering.
Grounds: no adjudication remains open against the stack — ADJ-18/19 (R92), ADJ-20 (dissolved),
ADJ-21 (R94) and ADJ-22 (R95) are closed, and the Phase D residual is lifted.
Status: spent [INLINE]

### R100 — push AUTHORIZED, with a proviso
Decision: Push is AUTHORIZED with the proviso that origin is the private remote under operator
control. R57 updates accordingly: "remote exists" is satisfied, "green Actions" remains a
cutover-battery gate, and Actions runs lint advisory-only per 4bc1c77 as intended.
Grounds: not stated in the register.
Status: standing — a FIDELITY PATCH: acted on by the WPTS push before it was registered, with
the verbatim text supplied retroactively by the operator on 2026-07-29. [INLINE]

### R101 — correction: one engineering item remained
Decision: On the record, one engineering item remained on the mint critical path — TD-4 /
CARD-POOL-ENCODING-BRIDGE (mode PREFLIGHT cannot burst). WPBRIDGE is dispatched with Phase T
(the fix, under R64 posture law, converging on the production path, mutation-tested) and Phase
R (a dev-box rehearsal in which a truthfully red rehearsal is a success, with zero first-time
code paths on the training box). After WPBRIDGE merges, the mint path is operator-only.
Grounds: not stated in the register.
Status: superseded in part by its own correction history — TD-1 also remained (see R102-R106),
and the mint path became operator-only at R107 by measured census. [INLINE]

### R102 — ADJ-23: WP-TRAINSTEP gets its own WP and full pipeline
Decision: TD-1 becomes WP-TRAINSTEP, its own WP with a full pipeline: TD-1 fixed through the
typed representation route (dense versus graph dispatch off the DECLARED representation, never
a buffer sniff); a SCOPED, mutation-tested conformance gate on the coordinator/trainer seam,
explicitly fenced off from the pyright backlog which stays CARD-LINT-TYPE; and a
behind-the-warmup-gate reachability census fully adjudicated BEFORE IMPL.
Grounds: the learner's dispatch seam gets a DESIGN, not a follow-up grant — and since a CPU box
cannot reach the warmup gate, how TD-1 will be observed must be settled before the fix is
written.
Status: standing [INLINE]

### R103 — ADJ-24 granted: an armed smoke config is minted
Decision: `smoke_preflight_armed.yaml` is GRANTED — minted via tooling so R1's
minted-never-hand-varied rule holds, with header-truth and a named live consumer (the second
burst oracle). It doubles as TD-1's end-to-end proof.
Grounds: every non-run5 config is deliberately disarmed under R59, so run5 was the only config
mode PREFLIGHT would burst, leaving no cheap rehearsal target and one unreachable oracle.
Status: standing [INLINE]

### R104 — ADJ-25: agreement-or-raise
Decision: Disagreeing dual-shape encoding configs are CORRUPT input, not a precedence question.
The one resolver keeps a single accept path, gains an agreement check, and RAISES rather than
silently picking a winner; `anchor.py` and `pretrain/validate.py` converge onto it. LAW-11's
arms are re-pinned by oracle.
Grounds: a narrowing can lose a raise as easily as a widening can, and precedence is the wrong
question to ask of corrupt input.
Status: standing [INLINE]

### R105 — WPBRIDGE closes PARTIAL-RATIFIED; merge authorized now
Decision: WPBRIDGE closes PARTIAL-RATIFIED — Phase T a full success, Phase R successful by
definition (truthfully red, with every wall named). MERGE AUTHORIZED now, standard runbook,
1-commit stack. The "partial" attaches to the WP's SCOPE, not its execution.
Grounds: good work does not wait on the next WP — a gate-green stack carrying no open
adjudication against itself merges on its own merit.
Status: spent [INLINE]

### R106 — ADJ-26: CARD-PROTOCOL-COMPLETE, pre-cutover, not mint-blocking
Decision: CARD-PROTOCOL-COMPLETE is opened pre-cutover and NOT mint-blocking: complete the
protocol declarations against concretes, THEN widen the AST conformance gate to all `*Like`
protocols, with the gate staying seam-scoped until the baseline is clean (per R98's
no-lying-gate law). Whether events.py gets a narrow read-side protocol or WorkerPoolLike grows
by nine is a design question, not a mandate. With this ruling the WPTS census holds no unruled
rows.
Grounds: called-and-undeclared with concretes present is type-system drift with zero runtime
effect — distinct from TD-1's called-and-absent.
Status: standing [INLINE]

### R107 — WPTS merge authorized
Decision: MERGE AUTHORIZED — the 3-commit wpts-scratch stack onto dev, standard runbook, then
push per R100. Post-merge, run5's critical path is operator-only by measured census: box
preflight, then the WP12-R decision, then prereg, then mint.
Grounds: not stated in the register.
Status: spent — its fidelity note (R100 and R101 cited but unregistered) was DISCHARGED
2026-07-29 when the operator supplied both texts verbatim. [INLINE]

### R108 — WPCLEAN delegation; the R90 package renews
Decision: The R90 delegation package renews unchanged for WPCLEAN. The phase-drop order when
short is PC, then LG/LT, then residuals, then R8, then NAME — dropping from the back. Fence:
any deletion touching R20-protected dense surfaces, the v6 question included, is
queue-with-recommendation and never auto-executed.
Grounds: the R20 dense frame is operator-locked.
Status: standing — renewed again by R119 for the WPMAIN and WP12-R runs. [INLINE]

### R109 — HANDOFF-1 closed; MIT confirmed
Decision: HANDOFF-1 is CLOSED — MIT is confirmed and has been in place since `5bd1d70`, and the
"hexo-mantis contributors" holder line is accepted as final unless the operator supplies a
name, which would ride the merge as a one-line commit.
Grounds: not stated in the register.
Status: spent — no name was supplied with R111's merge dispatch, so the default stands and no
copyright commit rode the merge. [INLINE]

### R110 — queue dispositions
Decision: CARD-PYRIGHT-STRICT is accepted as a post-cutover ratchet (one exclusion per commit,
with a zero-error proof each time). Q-NAME-1/2 recommendations are pre-approved iff they are
naming-only or reachability-only AND leave v6w25, multicluster and dense infrastructure
untouched — a dense-surface deletion requires the operator's explicit line. Q-PFC-SPLIT and
Q-PFC-R43 are approved in GROUND_PFC.md's stated order — flip-set pin regeneration first, then
the split plus frozen hunk under S-3/R43 — as the next run's opening phase.
Grounds: R20 is operator-locked.
Status: standing [INLINE]

### R111 — WPCLEAN merge authorized
Decision: MERGE AUTHORIZED — the 20-commit wpclean-scratch stack onto dev, standard runbook,
test-count floor 2163, pushed per R100.
Grounds: not stated in the register.
Status: spent — recorded with a sequencing note: run5 mints on gnn_axis_v1, WP-AXIS2 lands
post-mint, and gnn_axis_v2 is run6's pre-registered baseline change, never an in-run A/B.
[INLINE]

### R112 — host grant for the WPBOX run
Decision: Host access is GRANTED for the WPBOX dispatch — this host only, via the operator's
ssh alias, which is what satisfies R31. Rule 7 is absolute: provider name, alias and host paths
may appear in migration-workspace artifacts but NEVER in anything committed to hexo-mantis, and
bench provenance carries interpreter, numpy, rustc and CPU model only. The wiped instance's
freshly-pinned stack is THE prod baseline that R18/R21 deferred to.
Grounds: not stated in the register.
Status: spent — a this-host-only grant that does not travel; R142 re-grants it per-dispatch for
WP12-R rather than relying on this as precedent. Its verbatim text was a FIDELITY PATCH
discharged 2026-07-30, replacing a [DISPATCH-DERIVED] reconstruction. [INLINE]

### R113 — WPBOX merge authorized
Decision: MERGE AUTHORIZED — the 9-commit stack 45e6ee7 through b482243, standard runbook,
floor 2173, pushed (CB-4's green-Actions confirmation fires on it), scratch deleted. WPBOX
closes RATIFIED.
Grounds: the truthful OOM red, the first-ever GPU test executions and the prereg-held battery
are exactly what the box run existed to produce.
Status: spent — DISCHARGED 2026-07-30: fast-forwarded onto dev (aab268b..b482243), full gate
sweep green on the scratch tip (2171 passed / 2 skipped against the 2173 floor), pushed to
origin/dev, local wpbox-scratch deleted. [INLINE]

### R114 — CARD-RUN5-GPU-OOM routed to WP12-R
Decision: CARD-RUN5-GPU-OOM lands in WP12-R, tracing the single ~12.8 GiB allocation event to its tensor provenance; the fix
must be a capped, config-typed bound, never a silent truncation. Q-GAP-D-LATENCY and Q-GAP-C-EVAL-WALL ride its seam;
Q-TRAIN-STEPS-FLOOR stays blocked behind it. CORRECTED SINCE: R179 corrects the suspect list — the mechanism is the TRAINING
step, not inference — and ANNOTATION 4 widens the site set to the GNN training forward, R302(c) reading such an OOM as a
stale mint on a changed host.
Grounds: the original prime suspects (unbounded inference-server batch, or a graph exceeding the edge-cap's domain) are right
only when the OOM is inference-side.
Status: annotated — REGISTER-FOOT ANNOTATION 4; suspect list corrected by R179; the diagnostic direction is widened, not
withdrawn. [INLINE]

### R115 — CARD-RUN-MAIN opened; the run5 reachability frontier
Decision: CARD-RUN-MAIN opens as new, mint-critical work in its own small WP with a full
pipeline: wire `main()` to build trainer/pool/buffer, `compose_run`, and `run_until_stopped` in
production posture, with the preflight boot child converging onto the same path so there is ONE
composition authority. The WP still opens with a reachability re-census.
Grounds: the audit's measured headline — every subsystem works in isolation and
`python -m mantis.run` cannot start a run.
Status: standing [INLINE]

### R116 — audit dispositions batch
Decision: The dead TacticalSolver FFI export is deleted under the dead-weight law (riding any
stack); a glossary is written as step 0 of WP-LEAN-RENAME; skeleton contract docs pass to the
WP14 drift-gate's jurisdiction; the honor-system laws (LAW-03/04/09/10/15) get a post-cutover
enforcer card while the rest are process laws by nature; the solver quiet-move body is Track B
with F-35..40 as its mandate; and the `unimplemented!()` dense multi-window arms stay
dormant-annotated under the R20 frame.
Grounds: not stated in the register.
Status: standing [INLINE]

### R117 — WP-LEAN-RENAME approved as post-mint work
Decision: WP-LEAN-RENAME is approved as post-mint work, combining the lean-4 migration with the
tier-C registry renames per the audit's sequencing insight, under the audit's own constraints:
LAW-12 forbids re-stamping, so an alias layer is used rather than a history rewrite; gate 3's
floor is handled by the ratchet-to-measured law; persist/load.rs's "v6" is treated as the
compat constant it is; and the R20 frame is satisfied because v6_live2_ls lean-4 remains the
dense control arm. Its DESIGN returns to the operator for the final deletion sign-off.
Grounds: the audit's sequencing insight — the rename and the lean-4 migration share a surface.
Status: standing — a later dispatcher report that the tree contradicted R117 was accurate as
measurement but superseded as interpretation by R148. [INLINE]

### R118 — WP11-A close-out RATIFIED in full
Decision: RATIFIED in full — A-1 (wr_sealbot stays None until WP12-R Phase A populates it),
A-2/A-3 (run5's random-floor cadence and fresh-seed decision become named RUN5_MINT_PREREG
rows), oracles O-A..O-G (O-G: protocol keyword name permitted, attribute reads banned), e2e
option (b), deviations #2, #3 (revert) and #5, and the R-DRAIN-HARDCAP wiring. Handoff rows
route: encoding_spec blocker to WP12-R Phase B; _DEFAULT_MAX_PLIES, run.py's broad-except
re-tightening and emit coverage of eval knobs to WPMAIN Phase-0 item 5; F-RT2-2 to a WP-R row.
Grounds: not stated in the register.
Status: standing — a FIDELITY PATCH: issued in chat, never delivered as register-ready text,
and appended 2026-07-30 after WPMAIN's entry check caught the gap. [INLINE]

### R119 — R90/R108 delegation renewed for WPMAIN and WP12-R
Decision: The R90/R108 delegation package applies unchanged to the WPMAIN and WP12-R runs, with
the HARD STOPS verbatim: new adjudication classes; any dev mutation; scope widening beyond
carded scope; and any change to run5's armed values (0.25 / 25000 / 50 — mint-prereg-only, per
R82/R85/R92).
Grounds: not stated in the register.
Status: standing — a FIDELITY PATCH appended 2026-07-30 after the WPMAIN dispatch cited a
ruling the register did not carry. [INLINE]

### R120 — Q-EVALKEY: eval_enabled becomes a typed schema field
Decision: `eval_enabled` is promoted to a typed RunConfig schema field; the code-side default
True at the composition root dies; every minted config carries the key explicitly; the live
consumer is the one composition root, mutation-tested. R79 is not over-applied — eval_enabled
is itself the fact, not a proxy beside a gating value. R64 is unchanged: the preflight boots the
config's own value and may never force False. run5 mints True. The dispatch's "expected ZERO new
keys" is amended to exactly one key, with manifest impact stated in DESIGN.
Grounds: LAW-15 — a promotion bar with eval off is unrepresentable as a decision.
Status: amended by R122 (the budget widens to eval_enabled plus the disk_guard family) and
again by R126 (plus device). [INLINE]

### R121 — census dispositions; boot inversion accepted; run_until_stopped
Decision: Lift `_boot_main` out of tools/ci_gates/preflight_mint.py into src/mantis (the composition
root) and re-point both callers at it. Defects 1 and 2 are in scope: the root installs signal handlers
bound to the injected ShutdownState and constructs the DiskGuard, contract-tested. `run_until_stopped`
is wire-or-retire with grounds — it becomes the one loop or is deleted, never documented around.
Success criterion 1 amends to "boots through the one composer into the live run loop, bounded, clean stop".
Grounds: a tool owning the only real boot is the one-authority violation the work package exists to end.
Status: standing [INLINE]

### R122 — Q-DISKGUARD-KEYS granted as a FAMILY
Decision: Disk-guard keys are granted as ONE family — one config block, one resolver, three typed leaves,
minted at the previously dead literals 60/10/5. No enable boolean: LAW-16 makes the guard
always-constructed and R79 forbids a proxy beside gating values. R120's key budget amends to one key
(eval_enabled) plus one family (disk_guard); the three values are revisable at mint prereg.
Grounds: the literals were dead, so nothing has ever measured them; the behavior change is R121(b)'s
mandated construction, not the values.
Status: amended by R126 (budget widens to eval_enabled + disk_guard + device) [INLINE]

### R123 — Q-RUNID-PARAM provisionally endorsed; REVIEW-design charged
Decision: Deleting the `run_id` parameter is provisionally endorsed on the principle that no parameter
carries a config fact. REVIEW-design must first verify three things: where run_id is produced at HEAD and
that it is a mint-time fact; that no LAW-12 stamp path breaks; and that the deletion smuggles no default —
an absent run_id must RAISE by name. Any check fails and the deletion reverts to a parameter pending adjudication.
Grounds: same class as eval_enabled's removal, and it makes forcing a divergent value unrepresentable.
Status: standing [INLINE]

### R124 — Q-MAXPLIES-DEFER owner assigned
Decision: Promoting `_DEFAULT_MAX_PLIES` to a schema field becomes CARD-MAXPLIES, owned by WP-R,
pre-cutover, NOT mint-blocking. It is explicitly not folded into CARD-COORD-KNOBS — different seam
(eval, not coordinator) — and that card's scope stays as R78/R80 fixed it.
Grounds: code-side default present since WP11-A, no run5 dependency, and loading a mint-critical work
package with it violates drop discipline.
Status: standing [INLINE]

### R125 — Q-MF4-RC-PREDICATE: argued deletion ADOPTED
Decision: The unreachable except arm is deleted and MF-4's rc assertion goes with it. The mapping
alternative is rejected. Post-deletion, a RepresentationRouteError (if the closed enum is ever widened)
propagates as an uncaught loud failure — LAW-14 fail-loud, not a silent arm. Whoever widens the
representation enum must re-open child-seam routing in that same design.
Grounds: the arm is unreachable through the child CLI, so keeping it alive to feed a test is R116 dead
weight and the self-satisfying-test species.
Status: standing [INLINE]

### R126 — Q-DEVICE-AUTHORITY: device is a CONFIG FACT; the flag dies
Decision: Train device becomes a typed schema field — closed `Literal["cpu","cuda"]`, required, no
default, absent = named raise. The `--device` CLI flag dies on both callers. It is not a duplicate
authority beside `eval.worker_device` (split topology is legitimate). The key budget amends to
eval_enabled + disk_guard family + device; all configs re-mint mechanically.
Grounds: a CLI-only flag leaves a posture-divergence hole — a cpu preflight against a cuda run5
false-clears the GPU memory wall, the LAW-03 instrument-that-cannot-false-clear corollary.
Status: standing [INLINE]

### R127 — N-STRIPRESTAMP-PLACEHOLDER-R1 countersigned
Decision: Countersigned as proposed — the strip path's synthetic config gains `eval_enabled: True`,
disk_guard 60/10/5 and device at minted values, existing placeholder framing, stamp path untouched.
IMPL may derive the values from schema defaults if mechanical, else use literals with a comment citing R127.
Grounds: no third default authority is created, because stripped artifacts never boot runs.
Status: standing [INLINE]

### R128 — census correction; subprocess-shape census law
Decision: CENSUS.md §4's false negative is corrected in every downstream artifact and the R50 list is
AMENDED (two preflight process tests added with a rewrite-to-pin disposition, plus the three CUDA-boot
drives). STANDING LAW: reachability censuses over entry-point surfaces must cover subprocess, `-m` and
console-script invocation shapes, never import shapes only. IMPL's stop at the R50 bar was correct — the
bar worked, the list was wrong.
Grounds: an entry point's consumers are by nature invisible to import greps.
Status: standing [INLINE]

### R129 — frozen oracle re-pointed to truth; CARD-CLEANSTOP-SAVE opened
Decision: R43 grant — oracle O-B1 is re-pointed from "final checkpoint written" to the measured truth:
clean bounded stop, rc 0, O2 arm sets running=False, `checkpoints/` EMPTY because checkpoint_interval is 0,
clean-vs-aborted distinction intact. CARD-CLEANSTOP-SAVE opened (does clean completion trigger a final
save), pre-cutover, NOT mint-blocking — with the explicit trigger that run5 minting 0 escalates it to
mint-blocking on the spot.
Grounds: DESIGN §9's premise was false; the oracle must assert a positive truth and never go vacuous.
Status: standing — its own trigger fired (run5 mints 0); escalated to mint-blocking by R137 [INLINE]

### R130 — CUDA-boot drives: the bill converts to evidence
Decision: The three rc-40 drives re-point to a minted CPU-device config preserving the rc-40 property
(minted via tooling, never hand-varied); run5.yaml's local boot evidence moves to the box preflight. A NEW
positive oracle is required in the same pass: booting run5.yaml on a non-CUDA box fails LOUD in
init_trainer with parent rc 33. A raw torch raise is acceptable-loud, recorded, not re-wrapped in scope.
Grounds: pins R126's grounds as a permanent regression oracle so the device false-clear is dead by
construction and stays dead.
Status: standing [INLINE]

### R131 — sink-close deviation countersigned; protocol debt routed
Decision: The teardown deviation is countersigned as disclosed (§8's contract met, oracle O-D2 pins it),
but the forcing cause is named as debt: seven off-list suites stand in SimpleNamespace sinks lacking
`stop()`/`close()`. CARD-PROTOCOL-COMPLETE gains a row — complete the sink/watchdog protocol against
concretes, then lift the arm restriction so teardown runs unconditionally.
Grounds: production code contorting around under-implemented test fakes is the tail wagging the dog.
Status: standing [INLINE]

### R132 — Q-RT-DISKGUARD-RC0: bounded fix pass authorized, pre-merge
Decision: A disk-guard abort returning rc 0 does not merge as-is. One bounded fix pass: disk-guard abort
joins the fail-fast family with a registered exit code wired through `exit_code_for_abort`, contract doc in
the same commit, manifest row, and a mutation test proving a fired guard is supervisor-distinguishable from
a clean run. This is a new abort CODE, not an armed-value change; the R119 hard stop is untouched.
Grounds: a green that lies is R44-class evidence, and it appeared on the work package's own new subsystem.
Status: standing [INLINE]

### R133 — GnnNet call site deduped into WP12-R Phase B; class widened; rc rider
Decision: CARD-GNNNET-NO-FORWARD dedupes into WP12-R Phase B, whose scope WIDENS to the class "eval-side
model invocation assumes dense" — census every such call site and route all through representation
dispatch. Rider: terminal-round `eval_broken` becomes a registered nonzero rc; mid-run eval_broken stays
non-fatal. Until it lands, WPMAIN's rc-0 boot evidence carries the ledger caveat "rc 0 does not certify eval health".
Grounds: same defect, second call site — R71 class-fix law means the class boundary, not the two demo sites.
Status: standing — caveat DISCHARGED (Phase B 29f304b for cause; Phase O d0957f1 exit 48 for rc taxonomy), ratified by R167 [INLINE]

### R134 — Q-R8-DISKGUARD-FROZEN-HEADER granted
Decision: R43 grant, settled-class: the one-line R8 justification-header correction on the disk-guard
frozen file rides the R132 fix-pass commit under S-3 discipline.
Grounds: stale prose with no assertion depending on it (the N-3 precedent).
Status: standing [INLINE]

### R135 — WPMAIN MERGE AUTHORIZED, conditional
Decision: Merge authorized on R132's fix pass plus a targeted re-probe green: 6-commit wpmain-scratch into
dev on the standard runbook (verify stack, quiet box, double sweep with RAM+disk, R42 timestamps, floor to
measured, ledger and register appends, scratch deleted, push). The R133 caveat rides the ledger row
verbatim; WP12-R's entry gate then reads the new dev sha.
Grounds: not stated in the register.
Status: standing — executed as 7 commits (disclosed deviation), ratified by R136 [INLINE]

### R142 — host grant for the WP12-R run (Phase D)
Decision: Host access GRANTED for the WP12-R dispatch on the same box as WPBOX; R112's grant mechanism
carries over unchanged (this host only, via the operator's ssh alias). R31 is satisfied by this explicit
per-dispatch grant, NOT by R112's precedent. Rule 7 stays ABSOLUTE: provider name, alias and host paths may
appear in migration-workspace artifacts but never in anything committed to hexo-mantis.
Grounds: R112 was "this host only" and does not travel.
Status: standing — renumbered from R136 to resolve the operator/dispatcher numbering collision [DISPATCH-RECORDED]

### R143 — Phase A ladder: sealbot only, the rest authorized-skip
Decision: Phase A resolves the SEALBOT rungs only; the remaining four rungs (`kraken_raw`,
`kraken_mcts200`, `strix_128`, `strix_256`) stay loud-skip with grounds recorded per rung —
operator-authorized, not a dispatcher shortfall. Ladder target for run5 is 2/6 rungs live
(`sealbot_d5`, `sealbot_d6`), and EVAL_DECISION.md must say plainly that the Bradley-Terry fit rests on
two rungs of one engine family.
Grounds: operator direction, verbatim — "for now look for sealbot to pin and skip the rest".
Status: superseded in part by R139, which is the operative text (adds per-rung grounds + the sha-not-branch rider) [DISPATCH-RECORDED]

### R144 — CARD-GNNNET-NO-FORWARD absorbed into Phase B as a symptom
Decision: CARD-GNNNET-NO-FORWARD is not an independent defect and does not become its own card — the entry
census established one chain (eval/worker.py never threads `encoding_spec` → the engine binds the dense v6
default → `_is_graph` False → the dense arm calls `self.model(...)` → GnnNet has no `forward`). It is
absorbed into Phase B and closes when Phase B's graph eval round runs green. NO `forward` is added to GnnNet.
Grounds: adding forward would convert a loud failure into silent dense-shaped output from a graph net.
Status: standing — renumbered from R138; note the collision with the operator's own R138 (eval decode) [DISPATCH-RECORDED]

### R145 — the SealBot pin is ramora0/SealBot master
Decision: The run5 ladder anchor is pinned to `https://github.com/ramora0/SealBot.git` at sha
`c94749c21c16c3b072fff6da49762dd5f92f3986` (branch master as of 2026-03-31) — the stable public default
branch, not the newer experiment branches and not the private perf fork. Two consequences DESIGN must
carry: a vendor build step exists (C++/pybind11, must be rebuilt from source, loud-skip when absent), and
the adapter is a coordinate/turn-parity translation, not a wrapper.
Grounds: lineage verified upstream of the perf work; experiment branches can move and NNUE needs weights
vendor/pins.toml may not carry; upstream master exposes settable `max_depth`, so LAW-15 gets a reproducible bar.
Status: standing — CONFIRMED by the operator's R139, which adds the binding sha-not-branch rider [DISPATCH-RECORDED]

### R146 — ADJ-WP12R-5 measured BEFORE Phase A
Decision: The deploy-matching question is measured before Phase A opens, record-only and explicitly NOT
authorized to implement any remedy — it returns a verdict plus costed options. Phase order for the
remainder of WP12-R becomes ADJ-5 measurement → A → D(+riders), amending the dispatch's B → C → A → D.
Grounds: Phase A's EVAL_DECISION.md must state run5's promotion bar, and LAW-15 binds that bar to
deploy-matched eval, so a bar written while deploy-matching is unresolved rests on an unverified premise.
Status: spent — discharged 2026-07-31; the measurement returned DEPLOY-MATCHING BROKEN and the operator
ruled on it as R138 [DISPATCH-RECORDED]

### R136 — WPMAIN CLOSED, ratified in full
Decision: The WPMAIN close is ratified. The 7-commit deviation from R135's stated 6 is RATIFIED — a true
count with stable shas beats a squash matching stale text. RT-1's property-level fix and the
repo_design:101 producer are noted as debt paid; the R133 caveat rides the ledger row until Phase B
discharges it.
Grounds: preserving shas is exactly what R52 exists for.
Status: standing [INLINE]

### R137 — CARD-CLEANSTOP-SAVE: mint-blocking, both legs
Decision: R129's trigger fired on its own terms (run5 mints `checkpoint_interval: 0`; empty `checkpoints/`
observed on clean completion), so the card escalates to MINT-BLOCKING with two legs: (a) clean completion
triggers a final save as its OWN semantic — a third taxonomy leg beside periodic and shutdown_save, with a
mutation test that a clean 200-step run ends with exactly one final checkpoint; (b) RUN5_MINT_PREREG gains
a checkpoint_interval row, because 0 on a 25000-step run is not a legal production posture.
Grounds: a clean 25000-step run5 would otherwise write no checkpoint at all.
Status: amended by R173 — leg (b) now requires a nonzero value AND a live consumer on run5's route [INLINE]

### R138 — eval decode: Option A ADOPTED; deploy-matching restored
Decision: Option A adopted — eval decode consumes the overflow the producer already returns; self-play
semantics is THE authority (registry text right, eval decode wrong). Mint-blocking, lands on the WP12-R
eval seam. R71 class named: "eval-side consumption of the shared producer diverges from self-play
consumption" — census every divergence site, not the one demo line. Three oracles required: cross-FFI
parity, the dispersed-position regression, and ladder asymmetry dead against RandomBot.
Grounds: three independent sources agree on self-play semantics, and the divergence handicaps eval.
Status: standing — evidence statement DISCHARGED by merge append R228 (mechanism 6de393c); riders remain open and non-gating [INLINE]

### R139 — opponent pinning: operator decision recorded
Decision: run5's ladder is SealBot std (ramora0) plus a RandomBot floor; krakenbot is SKIPPED (weights not
cleanly accessible) and the strix-bot repo is SKIPPED (actively changing), each loud-skip with those
grounds per rung. Binding rider: a pin is a COMMIT SHA in vendor/pins.toml, never a branch name — "master"
is recorded as "sha X, master as of date". CARD-SEALBOT-BRANCHES opened, deferred, not mint-relevant.
Grounds: operator decision; anchoring a promotion bar to a movable branch is what LAW-10 exists to prevent.
Status: standing [INLINE]

### R140 — lean-4 and renames: R117 sequencing HOLDS; prep runs parallel
Decision: WP-LEAN-RENAME stays POST-MINT. Authorized now, find-only, zero tree mutation, in parallel:
(a) the glossary; (b) an evidence census for the 18-plane retirement assembled into a disposition doc
DRAFT that states the falsification record and mechanisms, never "worse/overengineered" as a vibe;
(c) a census of what the current dense control arm actually is, confirmed against the tree. The deletion
list returns for the operator's explicit sign-off line.
Grounds: run5 mints on gnn_axis_v1, so the dense path is off the mint's critical path; pulling renames
forward collides with WP12-R on the same seams for zero mint value.
Status: standing [INLINE]

### R141 — GAP-CENSUS authorized
Decision: One find-only census dispatch, cheap tier, run against HEAD 49e9efa plus STATE §2/§3 and every
open card and prereg row, producing MISSING.md — what stands between HEAD and mint, and between mint and
cutover, each row carrying an owner and a blocking status. No proposals, no tree mutation.
Grounds: not stated in the register.
Status: standing [INLINE]

### R147 — M-15: run5's RandomBot floor is ARMED; the config is wrong
Decision: run5's `eval.random_floor_games: 0` is WRONG — the RandomBot floor should be armed, and the
`4 -> 0` mint delta is superseded by R139, which names the floor as part of run5's ladder. Because
`random_floor_games` is an armed value, the re-mint is MINT-PREREG ONLY: it becomes a named prereg row with
grounds and the value stays the operator's at mint. EVAL_DECISION.md states a bar that includes the random floor.
Grounds: R139 makes the RandomBot floor part of the ladder, so a zeroed floor contradicts the ruled ladder.
Status: standing [DISPATCH-RECORDED]

### R148 — ADJ-11 RESOLVED: dense control arm is v6_live2_ls
Decision: The legacy production dense lineage is `v6_live2_ls`, and it IS the R20 matched-FLOP control arm,
consistent with R117. EVAL_DECISION and the R140(c) census consume this as ruled fact; any tree evidence
contradicting it goes to the queue, never silently re-pointed. The contradicting committed oracle
(tests/eval/test_graph_round_encoding.py, four sites naming `v6`) requeues as ADJ-WP12R-18, a naming defect
to correct under a grant.
Grounds: operator confirmation of the lineage; a ruled identification makes contradicting tree text a defect, not a dispute.
Status: standing [INLINE]

### R149 — second-half sequencing; CLEANSTOP rides the stack
Decision: Order is Phase Q (queue clearance) → Phase A, with Phase D parallel at will on the operator's
host grant. CARD-CLEANSTOP-SAVE rides wp12r-scratch as its own phase — one merge event, one runbook.
Unruled mint-relevant queue rows return to the architect in ONE batch, never serially. Operator amendment
in the same session: Phase Q is NOT run by the WP12-R dispatcher; a separate dispatcher with clean context
runs it, and the incumbent's duty is a self-contained handoff.
Grounds: a queue-clearance pass benefits from a reader who has not already concluded what each row means.
Status: standing [INLINE]

### R150 — WP12-R MERGE AUTHORIZED, conditional
Decision: Merge is authorized when all hold: Phases Q/A/CS/D green (D's success per the R114+R138 riders —
OOM class dead, burst past the death envelope, riders measured); the adjudication queue empty or every
residual row explicitly non-mint with grounds; RUN5_MINT_PREREG.md holding every blocking row with owner and
status; and the R133/R138 caveats discharged in the ledger. After merge the mint path is operator-only:
box preflight both tiers → prereg → mint.
Grounds: not stated in the register.
Status: amended by R174 (CS2 added to the roster) and R179 (merge waits on D-fix plus payloads 2–5) [INLINE]

### R151 — ADJ-4: CARD-DENSE-EVAL-ADAPTER, pre-Stage-0 blocking, NOT mint
Decision: The composed fact stands — run5's ruled dense control arm has never produced an eval result and
cannot until wired — but nothing on the mint path consumes dense eval, so CARD-DENSE-EVAL-ADAPTER is opened
as a HARD GATE on Stage 0 (which cannot open until it lands and LAW-10 anchors re-measure on it), not on the
mint. It is not ridden here, per drop discipline. Phase A is UNBLOCKED: EVAL_DECISION states control arm =
v6_live2_ls with eval wiring owed and the card named, refused-loud-by-name today.
Grounds: the promotion bar is graph candidate-vs-best and the ladder is SealBot+RandomBot, so the first
consumer of dense eval is the post-mint Stage 0 re-baseline.
Status: standing [INLINE]

### R152 — ADJ-7 plus Q-EVALBROKEN-RC0: ONE taxonomy, rides this stack
Decision: The two rows merge — both are "the parent cannot distinguish eval failures". One small design,
one authority: the `eval_broken` reason enum is the single source; every failure route produces a typed
reason logged in-run with its fire-rate; the parent maps reason → rc on the TERMINAL round only, mid-run
staying non-fatal under the watchdog's jurisdiction. Mutation proves each reason class is
supervisor-distinguishable. No second boolean or proxy beside the enum.
Grounds: two rows describing one indistinguishability defect must not grow two authorities.
Status: standing — rider added by R164 (the Phase-T counter family must reach the event stream in this landing) [INLINE]

### R153 — ADJ-10: characterize first; documented semantics is authority
Decision: Escalation was correct — training-target mass is not a dispatcher's to accept. Authority is
settled: the policy target is `raw_visit_distribution` and the targets do not inherit the off-window skip,
so an export dropping off-window mass diverges from its documented authority. Characterization comes FIRST
with a pre-registered instrument and verdict rule stated before measuring; any systematic drop confirms the
class defect regardless of size. MINT-BLOCKING until the measurement rules.
Grounds: n=1 does not carry a verdict, and run5 training on silently truncated targets moves R138's
handicap from eval to learning.
Status: standing [INLINE]

### R154 — ADJ-19: deletion authorized, conditioned
Decision: `resolve_anchor_path` and `resolve_arch` die under the dead-weight law, riding this stack,
PROVIDED the armed census's zero-ref evidence is recorded per deletion AND neither is an R20-protected
dense surface — if either is, it queues with a recommendation and the named-list exclusion stands
meanwhile. The anti-rot test survives the deletions.
Grounds: the anti-rot test is the law's enforcement, not the exclusions' registry.
Status: standing — its two conditions later fired and caught a defective zero-ref census and an R20 surface (ratified in R162) [INLINE]

### R155 — leg 2 mandated; measurement-path parity clause
Decision: The legal-set leg runs BEFORE anything else — `expand_and_backup_ls_at` driven over the same
1440-position sample for gnn_axis_v1, pre-registered before running. STANDING instrument clause: a drop
measurement clears an encoding ONLY when driven through that encoding's PRODUCTION expand path, so leg 1's
gnn_axis_v1 zero stays labeled non-production forever. Mint-blocking stands until leg 2 rules; refuted
unblocks that axis but the fix still rides, confirmed adds a prereg dropped-mass row.
Grounds: the LAW-03 false-clear corollary made mechanical.
Status: standing [INLINE]

### R156 — leg-1 conduct ratified; hypothesis binds the flip-set
Decision: Leg 1 is ratified — abort 1 fired and was honored (supersede, never merge, since two samples
under different generators are not one sample), the "0 affected cannot be read as clearance" framing is
correct hygiene, and the documented-semantics authority call stands. The cap-boundary hypothesis stays
labeled UNTESTED and is not consumed as explanation, but it BINDS the fix's flip-set: the 193–235 band,
sparse-coverage early plies, and deep-tail dispersed rows are all mandatory boundary coverage.
Grounds: the fix must not be able to pass on the demo region alone.
Status: standing — the cap-boundary hypothesis was KILLED in passing by R157 [INLINE]

### R157 — leg 2 CONFIRMED consumed; degenerate class named; contamination census
Decision: Leg 2 is ratified and RUN5 EXPOSURE CONFIRMED is consumed — mint BLOCKED on this axis, the fix
rides the stack, the prereg gains the dropped-mass row, and an exported-target parity oracle is required.
The all-zero rows are a NAMED SEPARATE CLASS, not folded into the mass statistic: the fix makes degenerate
export UNREPRESENTABLE. A contamination census rides the fix — enumerate every committed fixture, corpus,
bank or golden generated through `get_policy_ls`, then regenerate or label VOID-AS-ANCHOR.
Grounds: a policy target must be a valid distribution ALWAYS; a silently-shipped no-op row is not a statistic.
Status: amended by R161 — the degenerate-row count supersedes from 12 to 37* at the R160-verified regime [INLINE]

### R158 — fix scope: the WHOLE target pipeline is the class
Decision: The R71 class is "consumer diverges from the documented no-drop producer contract", and its
boundary is the ENTIRE target pipeline, not the Rust exporter alone: census `get_policy_ls` export →
wire/record format → replay buffer → Python batch assembly → loss, each stage getting a producer-consumer
parity check. Semantics: export equals the raw visit distribution over the FULL child set, mass sums to 1
within tolerance, off-window mass carried, never renormalized over a subset. Parity oracle both sides of the FFI.
Grounds: if the Python side consumes only the dense half, fixing the exporter merely moves the drop downstream.
Status: standing [INLINE]

### R159 — frozen-oracle pre-grant for the fix
Decision: Settled-class R43 grant, pre-authorized: frozen parity oracles and fixtures that pin the DROPPING
behavior as expected (graph_child_parity.rs and the child-parity fixtures are the named suspects) may be
re-pointed to the no-drop law citing R157/R158, under S-3 discipline — enumerated hunks, verbatim
extraction, before/after hashes, and the mutation condition. Any hunk that WEAKENS an assertion rather than
re-pointing it, and any oracle whose subject is not target/parity semantics, QUEUES rather than being edited.
Grounds: the dispatcher's refusal to re-point an unruled surface was right; this ruling supplies the ruling.
Status: standing [INLINE]

### R160 — sims/export-regime provenance; the table is not yet the grounds
Decision: "50 sims = run5's actual" is UNVERIFIED and conflicts with the recorded PCR frame (600@10% /
75@90%). Before DESIGN freezes, a census must: derive run5's sim regime from run5.yaml plus the
PCR/playout-cap config on disk; determine from the tree whether policy targets export from quick-arm moves
at all, cited by file:line; and re-run the binding table at the TRUE export regime(s). The prereg
dropped-mass row cites only production_path AND production_sims rows; the 50-sim table is retained as
mechanism evidence under its actual provenance label.
Grounds: derive-or-delete — never from memory of either side; if the tree contradicts STATE's PCR frame it queues.
Status: standing [INLINE]

### R161 — degenerate class superseded 12 to 37*; unconstructible ratified
Decision: R157's degenerate-row count supersedes to the R160-verified regime's measurement, with 37 at 50
sims standing as mechanism evidence meanwhile. Masking degenerate rows is REJECTED on the dilution grounds;
instead the fix makes a degenerate target UNCONSTRUCTIBLE — a target that is not a valid distribution
cannot be built (correct by construction or named raise). The MAX_VISITS guard is ratified as argued:
needed, not binding at run5.
Grounds: a count taken at an unverified sim regime cannot be the grounds for a prereg row.
Status: standing [INLINE]

### R162 — Phase T CLOSED, ratified in full; the record corrected
Decision: Phase T is ratified — the entry judgement call, STOP-1/STOP-2/DEV-1, the in-card RED-TEAM
rulings, FA-1..FA-3, the bench with its honest out-of-bracket intermediate, and T-4's dispositions including
both VOID-AS-ANCHOR labels. The 2c disclosure is ratified (a dense value-only sentinel masking degenerate
rows out of numerator AND denominator satisfies R161's no-dilution grounds). QN-1 deletion is authorized.
CORRECTION on the record: "remaining path to mint is operator territory" is FALSE and is struck in the
dispatch log — Phases A and D, CS and R152 are owed engineering.
Grounds: R154's two refusals are R154 executing as written; the false Phase-Q evidence is census-gap instance #5.
Status: standing [INLINE]

### R163 — ADJ-20: sims regime is a prereg decision; recommendation recorded
Decision: The flat-50 disk config versus STATE's PCR 600@10%/75@90% frame is an armed-value question —
prereg-only, operator-only. Architect recommendation, recorded not ruled: re-arm PCR 600/75 per the recorded
intent (compute-legal at mean ~128 against the 150 hard cap; the quick-arm policy mask means policy targets
come from full-search moves). Flat 50 is acceptable ONLY as a deliberate prereg line with grounds, never as
an accident of a zeroed block.
Grounds: armed values belong to the operator at mint prereg, so the architect recommends and does not rule.
Status: standing — the recommendation was adopted by R165 [INLINE]

### R164 — ADJ-21: LAW-18 means in-run; rider on R152
Decision: Test-visible-only counters FAIL LAW-18 — the law's text is in-run. Rider on R152: the Phase-T
counter family, including the §0b drift witness, must reach the EVENT STREAM in R152's landing, as one
taxonomy plus one observability commit family under R84 mutation discipline. R152 is confirmed pre-merge —
this makes it load-bearing, not merely owed.
Grounds: LAW-18's own provenance is a null read that was unreadable without in-run counters — exactly this case.
Status: standing [INLINE]

### R165 — sims regime: OPERATOR ADOPTS PCR 600/75
Decision: run5's sim regime is PCR re-armed at 600 full @ 10% / 75 quick @ 90% (mean ~128, under the 150
hard cap). The prereg line states this plus the quick-arm policy-mask semantics, and run5.yaml re-mints via
tooling AT prereg authoring, not before. Configurability is already satisfied — the PCR block is
schema-typed config with the V-PCR validator; nothing new is built. Pre-merge rider: the 600-sim
supplemental binding-table row runs on the existing leg-2 instrument so the prereg cites measured numbers at both arms.
Grounds: operator decision, per R163's recommendation and the original intent.
Status: standing [INLINE]

### R166 — mutation-mechanism law, standing from Phase CS
Decision: STANDING LAW — every mutation row carries a MECHANISM column stating what the mutation reaches
and why the pin can see it; reachability is stated, never assumed. The discipline back-propagates: a
discipline invented mid-phase applies to that phase's earlier tables in the same event.
Grounds: four instances in one phase earned the law, and the fifth was found exactly where the mid-phase fix
had not been back-propagated — LAW-07 producer-testing applied to mutations themselves.
Status: standing [INLINE]

### R167 — Phase O CLOSED, ratified
Decision: Phase O is ratified: three rows closed, the R133 caveat DISCHARGED with mechanism named (terminal
exit 48, seven pins, M-O8 two-sided, no-over-fire proven), and ten non-mint rows transferred to the queue.
The queue-transfer practice is RATIFIED AS STANDING — findings live in the adjudication queue, never only in
impl notes. Two red-team discoveries are logged as register-grade: 42 pre-registered mutations had never
been executed by any prior stage, and a renamed pyo3 getter shipped a fabricated 0 through a green tree.
Grounds: execution, not registration, is the evidence.
Status: standing [INLINE]

### R168 — CS second-FAIL adjudicated: bounded retry authorized
Decision: The two-FAIL stop rule targets an implementation that keeps failing; this one survived two full
adversarial passes with zero behavioral claims falsified, and FAIL-2 originated in the auditor's own
transcription (11/13, not 12/13). Neither verdict is relabelled — the FAILs stand in the record and the
retry DISCHARGES them. One bounded retry at exactly four workspace-text items, NO code and no sealed-file
edit; R166's back-propagation clause applies across the ENTIRE CS mutation table, not the two named cells.
Grounds: 26 lines is not a search radius for a class on its third instance in one card family.
Status: standing [INLINE]

### R169 — ADJ-24: liveness claims tie to their instrument
Decision: "2/6 live rungs" is not claimable bare. EVAL_DECISION.md instead states "2/6 resolve locally;
liveness unverified in CI; verified at box preflight" — covered/not_run honesty, claim bound to its
instrument. The liveness build RIDES the Phase-D box session (approved, cheap, batched under the existing
host grant), and its result upgrades the decision doc's line in place before the prereg cites it. No CI gate
is built for it pre-mint.
Grounds: a liveness claim with no runnable producer is the R69/LAW-07 class.
Status: standing [INLINE]

### R170 — dispatcher commit authority clarified; template text superseded
Decision: STANDING — a dispatcher MAY commit to the work-package-scoped scratch branch when (a) the verdict
chain at that boundary is PASS or R70's six-part test is met, (b) gate evidence precedes the commit,
(c) the commit's content is subagent-authored except transcribed measurements, and (d) the authority is
CITED in the ledger row, never inferred from precedent. dispatcher_template.md's "do NOT commit" is
WP3-era text SUPERSEDED. The MERGE is carved out: R150's runbook is the plan, not the trigger — the
ff-merge, scratch deletion and push execute only on a fresh architect go-line.
Grounds: "authorized in a ruling" and "authorized right now" are different things for an irreversible action.
Status: standing [INLINE]

### R171 — Phase A oracle bank ratified; instance counts recorded
Decision: The 10/10 bank freeze is ratified with all three recorded rulings: (a) the +48 band was a
withdrawn figure carried from conversation instead of re-derived from PREREG_A REV 5 (+52…+86), corrected in
the artifact — derive-or-delete instance #11; (b) the F-A2 table amendment needs no grant, since the frozen
nodes were already redding correctly and the defect was the table's silence; (c) the refused weak-red is the
bank's load-bearing property — 22/23 greened against a synthetic document, then seven mutations each redding
exactly its named row. IMPL proceeds against the frozen bank.
Grounds: derive from the artifact at point of use — the pattern is numbers taken from chat rather than the document.
Status: standing [INLINE]

### R172 — Phase A CLOSED, ratified; the ADJ-22 hold ENFORCED
Decision: Phase A's close is ratified in full, and the 7/3 manifest non-re-mint becomes the STANDING rule —
pre-IMPL hashes are the evidence the implementation did not shape its tests, and re-minting erases what the
freeze exists to prove. The liveness line shipping under abort 10 is covered/not_run honesty done right.
THE HOLD STANDS: Phase D does not open the box while ADJ-WP12R-22 is unread by the ruling authority — five
asks, five non-answers, so the row now blocks by ruling rather than by request.
Grounds: grants docs are the key, and Phases O/CS/A all ship this way.
Status: standing — the hold it enforced was LIFTED by R174 once R173 resolved ADJ-22; the 7/3 standing rule
survives unchanged. Text recovered verbatim from the session transcript under recorded provenance [INLINE]

### R173 — ADJ-22 RESOLVED: option (ii); R137 leg (b) amended; CARD-CS2
Decision: Option (ii) is taken against the dispatcher's recommendation. Leg (a) closes COMPLETION exposure
only, leaving crash-loss, an unexecutable LAW-10 threat probe, missing eval/promotion candidates, and a
minted `checkpoint_interval` with zero live consumers. R137 leg (b) is AMENDED: the prereg row requires a
nonzero value AND a live consumer on run5's route. CARD-CS2 rides this stack — wire the interval read and
periodic save into the graph step through ONE resolver, reusing Phase CS's save machinery unchanged.
Architect recommendation recorded: 5000, on LAW-10 cadence alignment; the value stays the operator's.
Grounds: a dead knob "recorded as inert with grounds" is documentation of a violation, not a remedy.
Status: standing [INLINE]

### R174 — Phase D go-line, sequenced
Decision: The ADJ-WP12R-22 hold LIFTS. CARD-CS2 lands on scratch FIRST, then Phase D opens the box, so the
25001-step burst exercises the periodic save under production reality and a checkpoint appearing mid-burst
becomes a burst assertion. D's four payloads are unchanged plus that assertion. Merge remains gated on the
architect's go-line, with CS2 added to R150's roster.
Grounds: production-only axes get production-time checks.
Status: standing [INLINE]

### R175 — final-mile close-out ratified; four decision items ruled
Decision: (a) TERMINUS COLLISION accepted as pinned — two stamped artifacts with distinct provenance
(periodic vs clean-stop) are unambiguous under LAW-12 and dedupe logic would buy nothing. (b) LAW-10 PROBE
ABSENT: CARD-THREAT-PROBE opened, NOT mint-blocking, owed pre-Stage-0, and the prereg gains an INSTRUMENT
INVENTORY line recording its absence. (c) SEALBOT LICENSE: no license at the pinned sha means fetch-and-run
locally for the operator's own eval only; SealBot source or artifacts never enter hexo-mantis. (d) "rule 3"
means CLAUDE.md hard rule 3, one-resolver-per-regime-knob.
Grounds: a law without its instrument is stated, never implied healthy; the conservative licence posture is the point.
Status: standing [INLINE]

### R176 — ADJ-23 plus queue discipline for the merge
Decision: ADJ-23's verbatim text has never reached the ruling authority, so the next dispatcher's FIRST
report opens with it verbatim. Provisional frame pending that text: if run5 has NO resume path,
buffer-not-saved is genuinely non-mint; if resume-from-checkpoint is a claimed capability it ESCALATES,
because CS2's weights without the buffer is half a resume. The ~25-row queue is audited to a classified
table (mint-relevant must be zero or ruled) BEFORE the merge roster is evaluated.
Grounds: the ADJ-22 lesson applies before it can repeat — no row is ruled from a mention.
Status: standing [INLINE]

### R177 — compute placement, parallel tracks, and the ast-derivation practice
Decision: Standing for the rest of the work package: heavy compute (bursts, box-floor benches, liveness
games, scored rounds) runs on the box; the local machine carries only the test/gate suite and serialized
sweeps. A box fitness check precedes every burst — disk/RAM/GPU recorded, quiet, stack pins verified against
STACK_PINS, drift recorded with a provenance note on every floor-referenced bench. TWO PARALLEL TRACKS
authorized: BOX (Phase D's five payloads) and MERGE-PREP (local — queue audit, roster pre-audit, prereg row
drafting, register verification R118–R177).
Grounds: the ast-derived firing-order practice is recorded as the reference mechanism for R166 compliance —
recommended, not mandated, until it proves out once more.
Status: standing [INLINE]

### R178 — ADJ-23 RESOLVED: delete the dead key; posture documented; CARD-RESUME
Decision: No resume path exists (measured at 982da03), so the non-mint branch holds: (a) DELETE
`train.buffer_save_interval` and the no-op `_try_save_buffer` gate arms, with a mechanical config re-mint
riding the deletion; (b) the prereg gains one posture line — run5's buffer is deliberately non-persistent,
crash means restart from scratch, warmup governed by the real prefill gate; (c) CARD-RESUME opens POST-MINT
owning weights, optimizer/scheduler, buffer persistence and launcher surface as ONE design. Wiring it now is REJECTED.
Grounds: a key minted into run5.yaml with zero reachable effect is the dead-knob class R1 exists to kill,
and half-resumes are the trap the work package just spent a phase killing.
Status: standing [INLINE]

### R179 — Phase D SPLIT; the fix is its own dispatch; amplifier sequenced inside it
Decision: D-DIAGNOSIS is CLOSED PASS — its mechanism overturns the card's own premise (training step, not
inference; E unbounded, batch_size bounds the wrong quantity; the fp32 gather amplifier confirmed on-box).
D-FIX becomes its OWN dispatch on the full pipeline. Sequencing inside it: the fp32 gather disposition is
DESIGN's first sub-decision (casting to bf16 roughly halves the dominant tensor and changes what cap value
the headroom curve yields), with its own prereg, parity oracle and commit, THEN the cap is sized once.
"The burst got further" is BANNED as evidence — success criteria are structural.
Grounds: death is stochastic (173/882/180 s), so survival is corroboration only; two changes need two
commits, two benches and one sizing pass.
Status: standing — it amends R150 (merge waits on D-fix plus payloads 2–5) [INLINE]

### R180 — corrections, recovery, and the probe lesson ratified
Decision: Entry-state corrections accepted (17 commits, freeze 47/66 with CS 0/2 drift authorized). The R172
register recovery with provenance is RATIFIED and vindicates the verify-don't-trust handoff clause. Track 2
is accepted in full, including the 54-row audit (not the architect's stale ~25) and the roster pre-audit with
its AMBER gap register. STANDING lesson from the fitness-probe disclosure: a fitness probe that can mutate
the thing it measures is not a fitness probe — box probes run read-only and assert the pins they found.
Grounds: the gap the recovery filled was real, and a probe that mutates its subject cannot certify it.
Status: standing [INLINE]

### R181 — ADJ-26: the STATISTIC is replaced; the band never moves
Decision: The parity oracle's max-form statistic reads up to 3.96e-1 on HEAD-vs-HEAD and therefore cannot
witness F1 — it measures index_add_ atomics. Moving the band is REJECTED, and the general rule is recorded:
a band is never moved on a statistic that cannot distinguish its subject from nothing. The oracle re-points
to a statistic that is ZERO on identical code, bounded by a null-calibrated envelope derived from the
MEASURED distribution, with the null measurement committed as a pinned artifact. Deterministic-path (CPU)
legs may assert equality outright; each leg's label says which it is.
Grounds: moving the band would green the tier and forfeit the witness.
Status: standing — sharpened by R191, which replaces the calibrated envelope with deterministic-mode exact
equality on CUDA legs [INLINE]

### R182 — BF1-2: HOLD SUSTAINED; one probe converts mechanism to root cause
Decision: The hold is correct — a plausible mechanism is not a root cause. One targeted, budget-bounded box
probe: vary bytes at fixed arithmetic, or read the profiler's memory-throughput bound on the gather
directly. Wall tracking bytes confirms bandwidth-bound; confirmed means BF1-2 is ACCEPTED with mechanism
named and F1's wall envelope updated to measured reality; not confirmed means escalate with the probe data.
Grounds: faster-than-envelope has burned this lineage before (LAW-15's wall-clock provenance).
Status: standing [INLINE]

### R183 — ADJ-25 routed; BF1-4 retired; the F3 question comes verbatim
Decision: (a) R178(a)'s deletion is ASSIGNED to this dispatcher as settled-class execution, own commit on
the stack, config re-mint via tooling, landing before the merge evidence package. (b) BF1-4's premise is
RETIRED BY MEASUREMENT — F1's effect shrinks at production in-degree because independent per-edge errors
average down, so the review's finding-6 hold is discharged in the opposite direction from expectation and the
design doc gets the correction. (c) The Phase-B roster question from F3 reaches the architect VERBATIM.
Grounds: the ADJ-22/23 rule is now unconditional — no row is ruled from a mention.
Status: standing [INLINE]

### R184 — G-R178A-1 ratified; grant-overlap reconciliation is the standing shape
Decision: The grant's execution is ratified — throwaway-worktree authorship, sealed file byte-identical, and
a patch that makes the census STRICTLY STRICTER (4 reads/3 sites down to 1/1). The explicit reconciliation
against the overlapping grant G-DFIX-2-C — overlap named, premise checked untouched, F2's implementer barred
from citing this grant, both grant numbers on the twice-amended manifest — is RATIFIED AS THE STANDING SHAPE
for overlapping grants on one file.
Grounds: overlapping grants on one file are reconciled in writing, never let pass.
Status: standing [INLINE]

### R185 — ADJ-27: floor lowering authorized; the ratchet-down law
Decision: STANDING LAW — the test-count floor may move DOWN only when all four hold: the deletions are RULED
with a register citation in the commit message; the delta is attributed test-by-test; the new floor is the
MEASURED clean-tree count at that commit; and the lowering is its own visible act, never absorbed into an
unrelated addition. ADJ-27 meets all four: floor 2532 → 2529 on grounds R178/R184, own commit line. The
INSERT-ONLY companion pattern is recorded beside it — when a gate blocks ruled work, the gate learns the
ruled class, it does not loosen.
Grounds: a floor that passes by coincidence of sequencing is a floor that lies.
Status: amended by R188(a), which clarifies clause (4) — a downward move rides the commit that CAUSES it [INLINE]

### R186 — ADJ-28: one freeze authority; disagreeing duplicates become a red
Decision: The tree-matching freeze row GOVERNS and the disagreeing EVALDECODE row is corrected to its sha
with an S-3 record. The sharper half is the instrument: the freeze checker is TAUGHT that a duplicate path
whose shas disagree must RED, with a mutation proving it, and the freeze registers are swept for further
duplicates in the same pass with the count reported either way.
Grounds: a file frozen twice at disagreeing shas is not two freezes, it is zero — and a verification that
reads clean over such rows trains readers to wave known-bad rows through.
Status: standing [INLINE]

### R187 — mint header defect: mint-relevant, fixed now
Decision: `mint_config.py` writing Python `str()` into headers (producing "None" for None, which fails
validation) breaks minted-config provenance and is MINT-RELEVANT. Fixed in its own small commit riding the
stack: proper serialization, a header round-trip test (mint → load → validate green, None handled), and a
check that no existing minted config carries a stringified-None header, re-minting mechanically any that do.
Grounds: minted-config provenance is the surface R1's "minted, never hand-varied" rests on, and the prereg
authoring path will mint configs.
Status: standing [INLINE]

### R188 — commit trio ratified; R185(4) clarified by direction
Decision: (a) R185(4)'s "its own visible act" means never absorbed into UNRELATED work — a downward move
rides the commit that CAUSES it, because splitting a downward ratchet leaves the intermediate commit red at
its own floor; this is adopted into R185's text. (b) Sequential falsification is accepted: a claim true at
its commit and narrowed later with an explicit cross-reference is R9's amendment trail. (c) The independent
PRE rebuild is the method lesson — two of four checks would have passed on a wrong PRE because the patch
derives FROM PRE.
Grounds: a convention that only worked upward was a latent assumption until this exposed it.
Status: standing [INLINE]

### R189 — R90c third loop GRANTED; MA-4's record corrected by ruling
Decision: R90c's conditions are met (loops converged REVISE(7)→…→REVISE(1); scope enumerated as two
sentences in two documents, no code, measurement, band or oracle), and the third loop is GRANTED at that
scope exactly. The substance, recorded so nobody re-litigates it: on the MA-4 path the RAISE is the detector
— `index_add_` raises on the Float/BFloat16 mismatch before `_OpRecorder`'s assertion is ever reached, so
the assertion is NOT the detector and the raise-path coverage is load-bearing and deletable by nobody.
Both skip paths are authorized as a settled-class fix with mutation proof each, beside the R90c scope.
Grounds: the recorder calls the op before it records, so the raise precedes the assertion.
Status: standing [INLINE]

### R190 — errors #8/#9 ratified as corrected; the notarisation clause
Decision: Error #8 (a correction misread as a deletion) was caught before it re-created R186's disagreement,
and its lesson is adopted: BUILD INSTRUMENTS RATHER THAN RESOLVING TO BE CAREFUL — the morning's checker
would have caught the evening's error regardless of anyone's diligence. Error #9 is graver: repeating an
unverified claim is the same violation as making it, and a dispatcher's log NOTARISES, so adoption grants
authority the claim never earned. Standing emphasis: derive-at-point-of-use binds HARDEST on claims that
flatter their maker and cost nothing to accept.
Grounds: that is the selection pressure that fills logs with convenient falsehoods.
Status: standing [INLINE]

### R191 — ADJ-29 RESOLVED: deterministic-mode assertion; the class is settled
Decision: FAIL upheld. The CUDA parity leg runs under `torch.use_deterministic_algorithms(True)` and asserts
EXACT equality on identical code (measured 15/15 exact zeros); determinism is TEST-SCOPE ONLY. The 5080 null
artifact is re-labeled DEVICE-SPECIFIC, and per-device envelope calibration is REJECTED as a treadmill. The
median form is blind to minority-subset defects, so F2's oracle uses deterministic-mode exact/per-graph
assertions. STANDING SETTLED-CLASS: "oracle re-point from nondeterminism-contaminated statistic to
deterministic-mode exact assertion" auto-applies under S-3 plus the mutation condition, no architect round-trip.
Grounds: an exact assertion beats any calibrated bound and is device-independent.
Status: standing [INLINE]

### R192 — R191 execution ratified; C2 SUSPENDED pending provenance
Decision: (a) Instance #11 accepted — a brief instructing a cite to a column the artifact does not contain
is the notarisation class, and the agent correcting the dispatch rather than complying is the right reflex.
(b) C2 is SUSPENDED as load-bearing until the provenance hypothesis (untrained-randn versus shipped fixture)
is VERIFIED by running both inputs through one probe on one device; the loser's numbers then get corrected.
(c) The near-miss discloses its method: ENUMERATE, don't guess. (e) The count-removal is the standing answer
to self-staling numbers.
Grounds: a figure that must be re-edited on every sibling addition will be wrong and be read as evidence.
Status: standing [INLINE]

### R193 — S3 ratified; the commit-message correction; cap recommendation recorded
Decision: S3 is CLOSED PASS — one sizing pass as ruled, which vindicated R179's sequencing with numbers
(sizing on the synthetic would have over-sized the cap ~22%). The 528eb37 commit-message correction lives in
the log and prereg row with a cross-reference (instance #12: a synthetic-sourced number written unlabelled).
Architect recommendation recorded, not ruled: `max_edges` 4,500,000 / `max_nodes` 170,000; values remain the
operator's at mint prereg. The prereg row's negative claim is ratified verbatim: run5 still does not fit —
the cap makes overshoot UNCONSTRUCTIBLE by splitting, it does not make the batch small.
Grounds: a number states its provenance or does not ship; margin is sized for two named unknowns, not padded.
Status: standing [INLINE]

### R194 — F2 GO
Decision: F2 implementation proceeds against the S3 sizing input, under dispatch_6's F2 text as amended by
R191 (deterministic-mode exact oracles, never the median statistic), R179 (structural success criteria,
"got further" banned) and R193's values-as-recommendation. The remaining map after F2 is F3 (Q-D-1), F4 box
payloads, the merge evidence package with the instance ledger, then STOP for the go-line.
Grounds: not stated in the register.
Status: standing — F2 was ratified by R195 [INLINE]

### R195 — F2 ratified; the freeze statement adopted; coupling pinned
Decision: F2/67120a4 is ratified — all three conjuncts of the bound now carry detectors, the three-layer
arming pin with measured residues (22x → 4.16x → sized) stands, and the commit's refusal to claim is the
honest line. The 1.20% headroom TIGHTNESS becomes a pinned coupling so nobody retunes one leg. The freeze
statement is ADOPTED VERBATIM for the merge package: "48 of 64 match, 16 pre-existing attributed drifts, one
earlier unattributed drift absorbed into a granted row with provenance recorded" — never "48/64, up from 47".
Grounds: a count that improves by absorption is not hygiene; the three-legged trace is what makes the row honest.
Status: standing — text recovered verbatim from the operator after it was found absent from the register [INLINE]

### R196 — SHAKEDOWN RUN authorized; the quarantine law
Decision: Inside S6, after the bench clears its bracket, the burst MAY extend into a bounded shakedown run at
run5's config shape under ABSOLUTE quarantine: run-id prefix "shakedown-", outputs outside any evidential
path, checkpoints stamped but never promotable, no strength or learning claim ever citing it, and no number
entering prereg grounds except OPERATIONAL measurements explicitly labeled shakedown-sourced. Sequencing is
unmoved — merge → box preflight → prereg → MINT.
Grounds: it buys real operational soak (LAW-16 legs, CS2 checkpoint cadence, live eval rounds, cap fire-rate)
but can never buy evidence about the net.
Status: standing [INLINE]

### R197 — fresh dispatcher for S6+S7; rollover at merge
Decision: The D-fix session's context is spent, so S6 and S7 run under a fresh dispatcher context with a
self-sufficient dispatch, carrying the same verify-don't-trust entry (R118–R196) and the same instance-ledger
duty (12+ entries, one line each, beside the roster at the go-line).
Grounds: not stated in the register.
Status: standing — text recovered verbatim from the operator after it was found absent from the register [INLINE]

### R198 — recovery and withdrawals ratified; the corroboration asymmetry
Decision: The R195–R197 register gap was an operator delivery error, caught by the fidelity law and recovered
under R172's provenance pattern — second application, ratified. The ASYMMETRY is adopted into the handoff
lineage verbatim: rulings that authorize actions or specify artifacts have no pre-existing form and cannot be
corroborated by measurement, so they are the MOST dangerous to infer and the first to verify. The Q-D7-2/3
withdrawals are ratified, and LAW-16's first production firing on real staleness is recorded as the
run3-class defense proving itself live.
Grounds: arithmetic on a dead producer is not a rate, and telemetry reporting an empty buffer is not an alarm.
Status: standing [INLINE]

### R199 — ADJ-ZERO-GAMES: mint-blocking, NOT merge-blocking; diagnosis framed
Decision: The zero-games defect is MINT-BLOCKING (a run that plays no games cannot train) and NOT
merge-blocking. Prereg'd suspects in order: (1) CUDA initialization inside spawned/forked self-play inference
workers; (2) the invocation/environment diff between the Repro-A session and this one. Diagnose first, one
discriminator per suspect, verdict before any fix. Production corroboration is owed at box preflight, which
ADJ-ZERO-GAMES gates; the shakedown deferral is ratified because R196's purchase list is empty at zero games.
Grounds: the same sha reached a training step earlier and produces zero now, so the delta lives outside the tree.
Status: standing — its "inference_dispatch never ticked" signature was later FALSIFIED as diagnostic garbage
by R207; ADJ-ZERO-GAMES closed as invisible-games by R220/R222 [INLINE]

### R200 — zero-games: three discriminators, answerable before any fix
Decision: Three diagnosis steps in order of information per minute: (1) STACK DUMP the hung burst (py-spy
plus native stacks) to find where `collect_graph_data` waits and whether Rust worker threads exist;
(2) RE-READ Repro A's and S3's own recorded artifacts to see whether `inference_dispatch` ticked there, and
derive the exact invocation of each boot from its provenance record — executable without the box; (3) a
runner startup census in the event stream of the hung arms. Verdict rule stated before running.
Grounds: one attach answers the causal direction, and an init-order deadlock leaves a fingerprint in startup events.
Status: standing — the "never ticked" signature it chased was FALSIFIED by R207; the defect was invisibility (R210) [INLINE]

### R201 — first-contact ratified; the shared error; S3's provenance question
Decision: The verdict was never consumed, so ADJ-ZERO-GAMES re-frames to first-contact — "self-play has not
yet been made to work on this box" — still mint-blocking, still not merge-blocking. The falsified lever was
built by BOTH parties (the architect inferred games from the OOM; the dispatcher reported it as
demonstrated), two ledger lines, one each. CARD-RUN5-GPU-OOM's diagnosis stands unshaken: the OOM was real
and in the training step. MANDATED cheap check from artifacts: what fed S3's buffers, with the prereg row
stating its feed source either way.
Grounds: a producer inferred from apparent output instead of its liveness record is the Q-D7-2 class one level up.
Status: superseded in its framing by R204(c) (box-first-contact → GLOBAL first-contact), which was itself
FALSIFIED and corrected by R207/R215 [INLINE]

### R202 — localization without ptrace: the producer narrates its own startup
Decision: Ptrace denial redirects rather than blocks. Three moves in order: (1) VERIFY the local premise that
the local launch played games on CPU, derived from WPMAIN's recorded event stream rather than trusted;
(2) THE CHEAP DISCRIMINATOR, no ptrace and no code — rerun the burst with `OMP_NUM_THREADS=1` /
`torch.set_num_threads(1)`, verdict rule stated before running; (3) if that refutes, STARTUP NARRATION lands
in-tree as PERMANENT instrumentation (registered lifecycle events, LAW-18/LAW-07 shape, mutation-tested).
Grounds: a producer whose startup is invisible is how this cost four sessions; that instrument should exist regardless.
Status: standing — the GIL-starvation hypothesis was closed as REFUTED by R204(a) [INLINE]

### R203 — three-way split; reduced-latitude dispatcher mode
Decision: Remaining work splits into three single-problem dispatches (8A zero-games, 8B box payloads, 8C
evidence package) run under REDUCED-LATITUDE MODE: no new rulings, no paraphrased rulings (quote verbatim or
STOP), no judgment calls beyond each dispatch's enumerated latitude, every ambiguity STOP-and-report, queue
rows verbatim always, forbidden-actions list binding. Channel-count and architecture renames are CONFIRMED
post-mint, with F-14's number being 8-of-18, not 4.
Grounds: not stated in the register.
Status: standing [INLINE]

### R204 — Step 2 adjudicated REFUTED; the defect declared GLOBAL
Decision: (a) The GIL-starvation hypothesis is REFUTED and closed — the pre-written rule's subject was the
tick and it did not tick, and Step 1's local evidence shows the defect on a CPU-only machine. (b) Q2 is moot:
dispatch 7's CPU arm was a tooling-minted twin differing from run5.yaml in exactly two keys. (c) ADJ-ZERO-GAMES
re-framed from box-first-contact to GLOBAL first-contact — "production self-play has never run in ANY
`python -m mantis.run` invocation". Clause (c) is the falsified one: self-play DID run (13 games,
corpus_selfplay_frac 1.0, 5 real training steps) and the defect was invisibility, not absence.
Grounds: clause (a) rests on device-independent local evidence; clause (c) rested on a census that read a counter with no live producer.
Status: FALSIFIED (clause c), corrected by R207/R215 [INLINE]

### R205 — Step 3 GO, re-scoped LOCAL-FIRST
Decision: The narration fingerprint is read on the LOCAL machine (CPU, smoke config, minutes per iteration);
the box is for CONFIRMATION after a fix, not for diagnosis. Phase 0 comes from artifacts before any code:
training_step event counts and batch provenance in the WPMAIN launch and the CS 475s runs — if training_step
is 0 locally the "trains its 200 steps" claims get corrected to loop-steps, if above 0 the feed is named.
Then narration lands per 8A Step 3's spec and the fix routes on the fingerprint.
Grounds: the defect reproduces locally, which makes the diagnosis 10x faster; the instrumentation is permanent either way.
Status: standing [INLINE]

### R206 — CARD-EVAL-CORESIDENCY opened: measured, prereg-relevant
Decision: The co-resident eval child was measured on the 16 GiB box before trainer allocation, and it does
not fit under R193's margin. Card scope: characterize eval-child steady VRAM (load spike versus resident),
then let the prereg decide with grounds — smaller co-residency, staggered scheduling, or
`eval.worker_device: cpu` for run5 (deploy-matching binds the eval REGIME, sims and semantics; device changes
wall, not verdicts). The EDGE-CAP row gains a co-residency line either way. Mint-blocking only through the
prereg row being honest.
Grounds: it is the exact named unknown R193's margin was sized around, now with a number.
Status: amended by R229 — the 8.21 GiB steady-state figure is SUPERSEDED by measurement showing eval-child
VRAM growing unboundedly to 13.5 GiB at deploy_sims=150 (no-gate baseline 386 MiB) [INLINE]

### R207 — R204(c) FALSIFIED and corrected; both census errors ledgered
Decision: The architect's "GLOBAL first-contact / self-play has never run in ANY invocation" is FALSIFIED by
measurement (13 games, corpus_selfplay_frac 1.0, 5 real training steps) and is corrected wherever cited. Both
roots of the error chain are recorded, and both are instances of one law: a census must first verify its
counter is a live producer. The dispatcher's STOP on contradicting a ruling premise was exactly right, third
time running.
Grounds: LAW-07 applied to censuses — a counter that is not a live producer measures nothing.
Status: standing — its two visibility-mechanism attributions were both CORRECTED by R215 (game_complete IS
emitted at pool_drain.py:177 but dropped by `sink=None`; iteration_complete is gated by log_interval, so the
production-visible games signal is actor_sync). The register entry stands verbatim with R215 as its correction of record [INLINE]

### R208 — MINT-BLOCKER: phantom heartbeat sources under an armed abort
Decision: `inference_dispatch` and `selfplay_drain` are registered heartbeat sources that no producer ever
ticks on the production path, sitting under a staleness watchdog that when armed killed a burst at its 1800 s
deadline — a false positive that would execute every healthy run5 at minute 30. MINT-BLOCKING, opened as
CARD-PHANTOM-BEAT: census every registered heartbeat source against its actual producer (mutation-tested),
wire or remove each phantom with grounds, and add a producer-liveness conjunct to the arming audit. The rc-34
box death is re-labeled: evidence about the watchdog, not about games.
Grounds: this is LAW-07's founding class — a phantom gate input arming an abort chain — live in the tree.
Status: standing — clarified by R225: the conjunct lives at the composition root, not inside `HeartbeatWatchdog.arm()` [INLINE]

### R209 — the fourth cell; game_complete folds into narration; Step 3 sequenced
Decision: Before any instrumentation, mint the run5 CPU twin via tooling (exactly `train.device` and
`eval.worker_device` flipped) and run it locally ~20 min, with the verdict rule fixed in advance: GAMES
(games_total > 0) means box-specific, so suspects become box assets/env; NO GAMES means config-shaped and
locally reproducible, so BISECT the smoke-to-run5 config delta by minted intermediates until the blocking
key(s) are named. Per-game `game_complete` emission is confirmed as an instrumentation gap and folds into
Step 3's narration scope, which lands AFTER the delta verdict. The box stays untouched until a local verdict exists.
Grounds: the defect gets named before it is fixed, and narration should be built knowing what it must witness.
Status: standing [INLINE]

### R210 — ADJ-ZERO-GAMES re-framed; Key-1 named; narration authorized as the fix chunk
Decision: The "zero-games" diagnosis on the CPU path was an INVISIBILITY artifact, not absence. Key 1 is
measured-confirmed: `train.log_interval=1000` gates `iteration_complete`, the sole carrier of `games_total`
(intermediate 1 produced actor_sync=13, learner_step=12 with iteration_complete 0 — healthy but invisible).
Fix route (a), decoupling `iteration_complete` from log_interval while training_step alerting stays gated, is
AUTHORIZED as the structural fix and heads the narration chunk (one design: decoupling, registered lifecycle
events, per-game game_complete). Route (b), lowering run5's log_interval, is REJECTED as a workaround.
Grounds: games_total is a per-iteration counter, not a training-logging event, so the coupling is the defect.
Status: standing — its "training_step alerting stays gated" clause is superseded in scope by R242, which
rules that clause governed games-visibility narration and must not be read as arming-cadence law [INLINE]

### R211 — Key-2 (n_simulations=50 CPU-slow) closed
Decision: `selfplay.mcts.n_simulations=50` producing no completed games in 5 minutes on CPU is a CPU-only
performance characteristic, not a production defect, since run5 targets GPU. Closed with no mint action and
no card; CPU repro of run5 self-play at production sims is not supported and must not be attempted.
Grounds: run5 targets GPU, so a CPU wall-clock characteristic is not evidence about the production path.
Status: standing [INLINE]

### R212 — Box game-production must be MEASURED, not inferred
Decision: The claim "the box may have been producing games all along" is an INFERENCE about the box from a
local CPU result and is refused. 8B must run a diagnostic box burst (low `log_interval=10`, a diagnostic
config, not the mint config) and MEASURE `iteration_complete.games_total` and `actor_sync`. Verdict rule
fixed in advance: either above zero means the box produces games and ADJ-ZERO-GAMES closes as invisible-games
on both paths; both zero means a second distinct defect and a re-bisect on box. The "zero-games" to
"invisible-games" re-frame is CONDITIONAL on this measurement.
Grounds: never infer a producer from apparent output — the R199/R207 class.
Status: standing — the measurement returned actor_sync=5 and the verdict was consumed by R220/R222 [INLINE]

### R213 — R96/R167 compliance owed before any further work
Decision: The agent reported findings in the dispatch report but did not append them verbatim to the
adjudication queue or correct downstream artifacts. Required before the narration chunk begins: (i) append
the bisect table and Key-1/Key-2 mechanism text verbatim to ADJUDICATION_QUEUE.md; (ii) add a falsified-claims
entry that "run5 CPU twin produces zero games" is FALSIFIED, with the box half marked UNRESOLVED pending
R212; (iii) reframe STATE §2A to CPU-path invisibility resolved / box-path measurement-owed; (iv) report the
corrections back. The instance goes on the instance ledger.
Grounds: findings live in the queue, not only in reports, and corrections apply in EVERY downstream artifact.
Status: standing — compliance accepted by R214 [INLINE]

### R214 — routing: both lanes in parallel; narration DESIGN starts now
Decision: The R213 compliance is accepted, and "begin narration DESIGN or route R212 first" is a false
dichotomy — both proceed now: (a) the Step 3 narration DESIGN chunk starts immediately, local, with no box
dependency; (b) the R212 box diagnostic burst routes to 8B in parallel. Rider: the ORACLE-WRITE stage must
assert, mutation-tested, that `iteration_complete` emits on every coordinator step at run5's
`log_interval=1000` while training_step alerting stays gated — the falsifying test for the fix itself.
Grounds: the decoupling fix is required on both paths; R212 only determines whether a second box defect exists.
Status: standing [INLINE]

### R215 — R207's mechanism corrected; the games signal is actor_sync
Decision: R207's core falsification STANDS (self-play ran in every healthy invocation), but both of its
visibility-mechanism attributions are CORRECTED in every downstream artifact: (a) `game_complete` IS emitted
at pool_drain.py:177 and is golden-pinned, but is DROPPED in production because the WorkerPool is constructed
with `sink=None`; (b) games do NOT "live in iteration_complete.games_total" as a production visibility claim,
because that event is gated by log_interval=1000. The production-visible games signal is `actor_sync`. R207
is not rewritten in the register — this ruling is its correction of record.
Grounds: pool_drain.py:177's emit plus golden pins contradict "never emitted", and R210 contradicts "live in iteration_complete".
Status: standing [INLINE]

### R216 — narration chunk re-authorized with corrected premises
Decision: The narration chunk is re-authorized with corrected scope: (i) decouple `iteration_complete`
emission from `_run_log_interval` in coordinator/step.py, train-side and independent; (ii)+(iii) share ONE
root — the production WorkerPool built with `sink=None` — so the fix is to inject a sink there;
`game_complete` needs NO re-wiring and must not break the C-03/J-05 goldens. Lifecycle events register
through the selfplay-local EventSink Protocol, NOT `mantis.train.emit.EventSink`; any adapter lives in
`mantis.run`. The constraint is FIXED: no `src/mantis/selfplay/` file imports `mantis.train.emit`.
Grounds: routing lifecycle events through the train sink would create the forbidden selfplay-to-train import edge.
Status: standing [INLINE]

### R217 — grant boundary: narration owns `sink=`, R208 owns `heartbeat=`
Decision: The production WorkerPool construction site is shared by two distinct defects, so the narration
chunk's grant covers ONLY the `sink=` keyword argument (plus the named pool_hooks/pool_drain/step edits) and
must NOT touch `heartbeat=`, which is R208's subject. The two chunks land on the scratch branch sequentially
(narration first, R208 after), each touching a different keyword, so no merge conflict. The narration IMPL
notes must state explicitly that `heartbeat=` was left at None, and RED-TEAM must probe that heartbeat
behavior did not change.
Grounds: two defects at one construction site need an explicit grant boundary or they collide.
Status: standing [INLINE]

### R218 — REVIEW-design PASS-WITH-FIXES accepted; ORACLE-WRITE authorized
Decision: The PASS-WITH-FIXES verdict is accepted and ORACLE-WRITE is authorized. Rider 1: the
Q-O-TWO-POOL-READS collapse is a SEMANTIC CHANGE — the two snapshots become ONE atomic read, and the design's
"the collapse does NOT change the straddle" is corrected here to "it ELIMINATES the straddle"; the oracle must
assert the shared snapshot and a mutation re-introducing the second `pool.runner_stats()` call must turn RED.
Rider 2: the log-interval boundary test's name now covers only the still-gated arms, so the IMPL docstring
must reflect the split; renaming is IMPL's call.
Grounds: the collapse is more correct, not neutral, so it must be stated as a behavior change rather than a no-op.
Status: standing [INLINE]

### R219 — ORACLE-WRITE accepted; IMPL authorized; O-N1b precondition-gate noted
Decision: The ORACLE-WRITE stage is accepted — two oracle files, 5 tests, RED/GREEN verified at HEAD (3 RED,
2 GREEN), goldens green, floor 2690 → 2695 non-decreasing, the R217 boundary holding, no production code
touched. IMPL is authorized. Rider: O-N1b's RED at HEAD is a PRECONDITION-gated RED, not the collapse
assertion firing, so IMPL must verify that after O-N1 lands, O-N1b's REAL assertion fires and its falsifying
mutation reds. IMPL notes must state heartbeat untouched, the collapse as a semantic change, and the boundary-test update.
Grounds: a precondition-gated RED is a correct oracle pattern — the collapse cannot be tested until the decoupling exists.
Status: standing [INLINE]

### R220 — R212 verdict: box produces games; ADJ-ZERO-GAMES closes
Decision: The R212 box measurement is COMPLETE: `actor_sync=5` satisfies the fixed verdict rule, so
ADJ-ZERO-GAMES is CLOSED as an invisibility defect on BOTH paths (CPU: log_interval gating; box: the same
plus the `sink=None` drop). The "zero-games" to "invisible-games" re-frame is now UNCONDITIONAL and may be
written into artifacts as settled. The falsified-claims ledger drops the box-half UNRESOLVED marker, and
STATE §2A reframes to ADJ-ZERO-GAMES CLOSED.
Grounds: the condition R212 placed on the re-frame — a box measurement — is satisfied.
Status: standing [INLINE]

### R221 — wedged-drain kill is the operator's call; finding carded
Decision: The wedged drain (16 minutes past SIGTERM, past its own 900 s ceiling, holding 9798 MiB of GPU,
with the outer timeout carrying no `--kill-after` so it will never escalate) is the same lifecycle-defect
class as the earlier orphan. The kill decision is the OPERATOR's alone, since host access is per-dispatch;
the RECOMMENDATION is to kill the chain. The finding is CARDED as a lifecycle defect
(drain-exceeds-ceiling plus timeout-without---kill-after), owned by the post-mint queue, NOT the narration chunk.
Grounds: the R212 measurement is immutable in the JSONL, so killing the process cannot change the verdict.
Status: standing — the kill was operator-authorized and recorded in R222 [INLINE]

### R222 — R212 lane-b COMPLETE; ADJ-ZERO-GAMES closed; one mint blocker down
Decision: The box measurement is ACCEPTED (`actor_sync=5 > 0` fires the verdict rule), ADJ-ZERO-GAMES is
CLOSED as invisible-games on both paths, and the re-frame is UNCONDITIONAL with corrections applied to the
falsified ledger, STATE and the queue. The operator-authorized kill is recorded and the lifecycle defect is
carded post-mint. `iteration_complete=0` at `log_interval=10` on the box CONFIRMS the decoupling defect exists
on both paths. No second distinct box defect. Remaining mint blocker: CARD-PHANTOM-BEAT (R208) only.
Grounds: the fixed verdict rule fired on measured values.
Status: standing [INLINE]

### R223 — narration chunk IMPL PASS; R208 is the last mint blocker
Decision: The narration chunk IMPL is ACCEPTED (PASS). All three parts landed: `iteration_complete` decoupled
and emitting per burst with training_step alerting still gated; the two-pool-reads collapse passing one
snapshot so there is ONE atomic `runner_stats()` call per emit; a `_DeferredSink` adapter injected where
`sink=None` was, bound after run-safety construction, with `heartbeat=None` verifiably untouched; six
lifecycle events through the selfplay-local Protocol with no forbidden import edge. All falsifying mutations
were driven both ways and restored green. The chunk is READY for the scratch-branch commit.
Grounds: gate evidence green (2701 collected against floor 2690, lint green, pyright 0 errors) and goldens intact.
Status: standing [INLINE]

### R224 — dispatcher errors corrected (floor 2690, not 2701)
Decision: Both STOP reports are ACCEPTED — the agents were right to stop, and the errors were the architect's.
Three corrections: (1) the on-disk floor is 2690, not the 2701 written in both dispatches, which confused the
collected count with the floor file — an R98 violation, corrected in both; (2) the 8C dispatch operates in
the migration workspace, not hexo-mantis; (3) the R-numbered register is the migration workspace's
`plan/rulings_register.md`, not the falsified ledger (F-numbered, now `docs/governance/falsified.md`).
Grounds: a number taken from memory rather than derived from the file at point of use.
Status: standing [INLINE]

### R225 — R208 STOP adjudicated: the conjunct is composition-root, not watchdog
Decision: Option (A). The producer-liveness conjunct R208 names lives at the COMPOSITION ROOT, not inside
`HeartbeatWatchdog.arm()`; the watchdog is correct as written and its existing test stays GREEN. The defect is
that `_BASE_WIRED_SOURCES` declares `inference_dispatch` and `selfplay_drain` wired while `heartbeat=None`
means they are not — the root LIES to the watchdog. Fix: a `_DeferredHeartbeat` adapter injected and bound
like the sink, plus a root-level assertion before `watchdog.start()` that every wired source has a bound
producer. The conjunct is arm/construction-time only; mid-run producer death stays a normal staleness fire.
Grounds: R208's "arming audit" means the arming FLOW, and both sources have real producers — none qualifies for removal.
Status: standing [INLINE]

### R226 — 8C MERGE_EVIDENCE.md accepted; 2 missing prereg rows flagged
Decision: The 8C merge-evidence deliverable is ACCEPTED. One mint blocker remains (CARD-PHANTOM-BEAT,
waiting on R208); ADJ-ZERO-GAMES is CLOSED. The prereg holds 13 of 15 rows, with buffer non-persistence
(R178) and the R206 co-residency line MISSING but correctly listed rather than authored. Freeze re-run is
47 OK / 17 MISMATCH / 0 MISSING with R195's statement quoted verbatim. The two missing rows are NOT
merge-blocking — they must be present at MINT authoring, not at the go-line.
Grounds: R178 is already a written ruling in the register, and the R206 line is pending the 8B measurement.
Status: standing [INLINE]

### R229 — post-merge: three 8B findings routed for prereg authoring
Decision: The WP12-R merge is COMPLETE and 8B is DONE (6 of 7 payloads executed, 1 stopped on a stale brief).
Three findings route to prereg authoring, none a mint blocker: (1) R206 eval-child VRAM — the prior 8.21 GiB
steady-state figure is SUPERSEDED, since at `deploy_sims=150` co-resident VRAM grows unboundedly to 13.5 GiB
against a 386 MiB no-gate baseline, leaving ~2.5 GiB headroom on a 16 GiB card; the accept-or-cap decision is
operator-only. (2) `best_model.pt` will not load — carded, post-mint queue. (3) Q-TRAIN-STEPS-FLOOR is 1 step.
Grounds: these are inputs the operator needs at MINT authoring, and the unbounded-growth profile changes the
co-residency picture materially.
Status: standing [INLINE]

### R230 — preflight ratified; CARD-ORPHAN-WORKERS opened MINT-RELEVANT
Decision: The Step-1 preflight PASS is ratified on its evidence (the R210 and R208 fixes live, LAW-16 legs on
SIGINT). Worker-pool children surviving parent SIGINT with CPU pinned, reproduced twice, is ruled a LAW-16
defect and MINT-RELEVANT: CARD-ORPHAN-WORKERS is fixed on a fresh scratch branch, full pipeline, scope =
pool teardown on signal (terminate → bounded-timeout join → kill escalation), with a mutation test that SIGINT
during active self-play leaves ZERO descendant processes. Box invocations of `uv` must always carry
`UV_NO_SYNC=1` or `--frozen`. The operator's request constitutes the per-dispatch box grant.
Grounds: a killed run5 must not leave the box poisoned for the next launch — save-then-exit that leaves 694%-CPU orphans has not exited.
Status: standing [INLINE]

### R231 — VRAM: discriminate before fixing; prereg fallback recorded
Decision: The card's frame is ADOPTED — Hypothesis A (allocator caching of variable-size graph batches)
versus B (reference leak) is DISCRIMINATED BY MEASUREMENT before any fix: allocated-versus-reserved at
boundaries, then the `empty_cache` probe. The fix routes on the verdict, full pipeline, after which the
co-residency curve is RE-MEASURED at `deploy_sims=150` and its number written into the R206 prereg row.
FALLBACK recorded if the investigation has no verdict at authoring time: `eval.worker_device: cpu` for run5;
`deploy_sims=150` is armed and untouchable throughout.
Grounds: a blind `empty_cache` would mask a leak.
Status: standing — VERDICT-A (allocator caching) was consumed by R233 [INLINE]

### R232 — perf characterization authorized, non-evidential
Decision: One CUDA perf session at run5's real shape, AFTER the 9A fix lands so eval co-residency is real,
measuring games/hour, training steps/hour, eval-round wall, steady VRAM in both processes, GPU utilization,
and the derived wall time to 25000 steps. R196's quarantine applies: operational numbers only, labeled, with
no net-evidence claim ever; outputs feed the prereg's estimates section and Stage-0 planning, and every
number ships its command and provenance line.
Grounds: not stated in the register.
Status: standing [INLINE]

### R233 — 9A interim ratified; location approved; torch-pin ruled
Decision: (a) VERDICT-A is consumed — allocator caching on variable-size graph batches, with move-boundary
necessity MEASURED (cache hits 8.4 GiB inside one game) rather than guessed. (b) The location deviation is
APPROVED: the defect class is deploy-head graph eval inference and `DeployHeadPlayer` is its grep-verified
home; the torch.compile reduce-overhead incompatibility rides DESIGN as a recorded hazard. (c) No pyproject
torch-pin change pre-mint — the CPU wheel index is the deliberate parity regime. Instead the box bootstrap is
a documented mechanical step and every box fitness probe ASSERTS torch build == cu128 before any run.
CARD-TORCH-INDEX opens post-mint.
Grounds: touching the wheel index now is new scope against the finish-line posture, and the probe already caught both incidents.
Status: standing [INLINE]

### R234 — F-9B-1 resolved: the row's sentence governs; the drift was the architect's
Decision: The EDGE-CAP row's wording — "the cap's job is to make the overshoot unconstructible by splitting,
not to make the batch small" — is the ORIGINAL Phase-T sentence, and R193's and the STATE doc's "verbatim
requirement" quoted the architect's own restatement of it. The original governs, no edit is made, F-9B-1 is
CLOSED, and an architect ledger line is recorded. Phase P is ratified in full, including the not-a-git-repo
observation, which was correctly reported rather than improvised around.
Grounds: a verbatim-requirement that cites a paraphrase as the verbatim is the R98 class pointed at its own author.
Status: standing [INLINE]

### R235 — finishing order, bench bracket, and two watch items
Decision: Order to done: LAW-09 bench (before/after moves-in-fixed-wall, with pre-registered acceptance —
move-wall cost up to +60% accepted outright, beyond that STOP with the numbers) → V-2 (round COMPLETES;
steady eval-child at or under ~2 GiB or STOP for the cpu fallback) → Phase L → close/sweep → go-line evidence
→ Phase X. Two WATCH items ride X: reconcile the 20.12 s/round eval-wall figure with the zero-games-in-15-min
observation at `deploy_sims=150` and label both regimes; and watch the trainer's reserved-memory curve during
the soak for the same variable-batch allocator signature, opening a NEW CARD with the curve if it grows.
Grounds: eval wall is not training throughput, but a doubling needs eyes; two numbers that cannot describe the
same regime must not both enter the prereg.
Status: standing [INLINE]

### R236 — mint PAUSED by operator intake (posture amendment)
Decision: The mint is PAUSED. The operator's act of opening this intake amends the finish-line posture of
record, and the pause holds until the intake's problem table is dispositioned per STATE §2. No new scope
beyond the dispositioned pre-mint bundle; prereg authoring resumes only after the fix branch merges under
R170 and the box preflight re-runs green on the new dev.
Grounds: not stated in the register.
Status: standing [INLINE]

### R237 — entry-verification record (gaps OPEN)
Decision: Three gaps are recorded as UNVERIFIED at this session's entry: (i) register rulings R118–R235,
because the register was not attached to this session — verify-and-fill under recovery-provenance headers at
next repo contact before any register write lands; (ii) 9C status (V-1 bench, V-2, Phase L, the scratch merge,
Phase X); (iii) the fix branch's name, base and contents, where the defect doc self-declares nothing
implemented, contradicting STATE §2, to be resolved by the operator. Rulings issued in this session are
register-ready blocks pending (i).
Grounds: not stated in the register.
Status: standing [INLINE]

### R238 — radius resurrection watch (LAW-02)
Decision: Both new research docs carry radius-8 framing (a formal identification built on d_hex <= 8 and
CNN-3 arithmetic using "radius 5 or 8"), but the falsified ledger holds run5 radius = 6, never 8. Structural
conclusions (state-dependent legal set, the locality caveat on strategy-stealing) survive at r=6, but any
numeric claim citing 8 — span-leak margins, candidate-set sizes, L(S) growth — must be re-derived from
`registry.toml` at point of use before it grounds a fix. No doc may enter the design record with an
un-annotated radius-8 number.
Grounds: LAW-02 — never drop a driver on an un-re-validated prior; the radius is a falsified-register fact.
Status: standing [INLINE]

### R239 — F-15 transfers as a design constraint on SYS-5
Decision: SYS-5 (quiescence into a proven MCTS-Solver +/-infinity backup with subtree termination) is
adjacent to falsified row F-15 (expansion-time forced-win short-circuit means the net never sees near-win
positions, so no fork learning). The context transfers as a HAZARD, not a kill: subtree termination re-creates
the starvation mechanism unless ExIt-style target injection is mandatory in the design. Any SYS-5 design
without proof-as-training-target is REJECTED at DESIGN stage.
Grounds: both independent reviews' top recommendation and the register's own F-38/F-39/F-40 convergent close
point at the same lever — new research and the falsified register agree.
Status: standing [INLINE]

### R240 — goal ordering of record (operator posture)
Decision: Work proceeds in three tracks, strictly ordered for merges and interleaved for prep: (1) CORRECTNESS — the pre-mint fix bundle (problem-table rows A1-A15 plus the LAW-07/18 debts) finishes on the existing fix branch, nothing architectural riding it; (2) ARCHITECTURE & EXTENSIBILITY — the PLAN-0 program (capability seam, conformance suite T1-T4, ragged-wire collapse, then arches) opens only after (1) merges and the intake's MEAS items report; (3) PERFORMANCE — a standing LAW-09 track, profiling harness prepped during (1), the box profile gating the sims/lever prereg lines, per-hotspot preregs thereafter.
The mint slot is unchanged: after (1) + preflight + prereg.
Grounds: not stated in the register.
Status: standing [INLINE]

### R241 — fix-branch adoption protocol
Decision: The unfinished fix branch is INVENTORIED before it is extended: every existing commit and diff is mapped against items A1-A15 and classified ADOPT (it passes its item's oracle) / FINISH / REDO. The inventory report lands in the adjudication queue before any new implementation starts.
Grounds: no assumption that partial work is correct — R155 applies to anything claiming to be a fix.
Status: standing [INLINE]

### R242 — ADJ-D12: gate cadence decoupled from narration cadence
Decision: The draw-rate and SealBot-WR hard aborts could not fire before step 1000 at run5's log_interval. Mechanism ruled: a new explicit schema key (e.g. monitor.gate_interval) with NO default, consumed by gate evaluation, abort sampling and monitor_gates emission, while train.log_interval reverts to narration-only; consec/threshold semantics are re-expressed in gate-interval units and the re-scaled armed values become operator prereg rows, not code.
This supersedes R210's "training_step alerting stays gated" clause in scope — R210 governed games-visibility narration and must not be read as arming-cadence law. FULL sub-pipeline (impl/review/red-team) is mandatory because this touches armed aborts.
Grounds: armed machinery with a blind first kilometer is the F-43 class landing on the abort path itself.
Status: standing [INLINE]

### R243 — ADJ-D13: item 6 disposition
Decision: The dispatcher's halt was correct. Final state: the non-finite guard, its counter and alert inclusion all stand, and the hard-abort arm stays exactly as the R56 pin has it — disarmed at 1e9. Arming NaN into the hard abort is a prereg row, not a code change. Commit 0a2b238's overstating subject is reworded by interactive rebase next session.
Grounds: the branch is unpushed and clean, so the clean-bisect rule applies.
Status: standing [INLINE]

### R244 — per-item verification law (new standing law)
Decision: Per-item verification henceforth runs (a) the item's declared pins, (b) the test files that reference any touched symbol, GREP-DERIVED and never guessed, and (c) a full default-tier checkpoint at minimum every 3 items and at EXIT. Ledger entry recorded with it: three regressions including 25 preflight failures from item 6, caught only by the EXIT full tier.
Grounds: three regressions surviving to EXIT proves that directories-touched is not directories-that-pin.
Status: standing — its evidence-hygiene rider enters the register as R244-a by R252 [INLINE]

### R245 — CNN-6 verdict: DROP confirmed
Decision: The lossless augmentation group has order 4, not 12; 8 of the 12 elements drop ~25% of cells each, so the dense path carries confirmed label noise. Two options were put to the operator — (a) restrict dense-arm augmentation to the order-4 subgroup pre-mint, or (b) keep 12-fold for run3 comparability — with the architect's lean on (a). The graph path is structurally unaffected (rotate_axial exact).
The operator's approval of the R261 go-package RULED it option (c): a conditional per-record symmetry gate on the dense arm, implemented during the mission, which does not block the graph shakedown.
Grounds: the control arm's job is to be a sound dense baseline, not a bug-compatible one.
Status: standing [INLINE]

### R246 — CNN-9 confirmed, card not mint-blocking
Decision: Four of the seven history planes carry opponent stones deterministically, but the control arm v6_live2_ls keeps planes {0,8,16,17} — no history planes — so the mint is untouched provided that plane set is confirmed AT POINT OF USE from registry.toml rather than assumed. The fix rides the cross-language-parity card (encoding F-01/F-02), post-mint.
Grounds: the defect cannot reach the minted control arm once the exemption is re-derived from the registry.
Status: standing — exemption later re-derived and the row DISCHARGED [INLINE]

### R247 — bootstrap corpus intake (card + manifest pin)
Decision: The HF corpus (human-only, rated, per-game Elo, sha256'd, encoding-free axial move lists, MIT) is ADOPTED as a sha-pinned external bootstrap artifact — outside the repo, manifest-indexed per R7 — pending an audit of the winner/coordinate convention mapping to mantis-core, dedupe overlap against the in-repo corpus by game_hash, Elo and length distributions, and two recorded selection biases: decisive-only (zero draw mass, a value-target bias) and >=20 moves (drops the short tactical wins F-15/F-38 care about). Whether run5's bootstrap points at it is a prereg row.
Grounds: it matches the F-06 canonical shape and is presumably the canonical corpus's published export.
Status: standing — the corpus is CERTIFIED as prereg grounds by R279(b) [INLINE]

### R248 — perf strategy of record
Decision: Two perf tracks, split by what is being optimized. Track P-loop (now, mint-gating) is the HARNESS, not the net — batch starvation is loop-level and architecture-independent, so it precedes the mint as infrastructure correctness. Track P-arch (post-seam) is the NET — kernel and arch-level optimization waits for PLAN-0 Stage 3, and PLAN-E's T6 makes us/leaf a permanent conformance column, which is development-with-perf-in-mind made structural.
Grounds: the K-reduction alone has four implementations today, so optimizing duplicated paths is paid twice and measured never.
Status: standing [INLINE]

### R249 — ADJ-D32: phantom cluster metrics
Decision: Accepted. derived_mean_f64 returning a hard 0.0 at zero count, with variance atomics never passed, is a channel asserting "perfect agreement" from zero samples — and it retroactively VOIDS Phase R's CNN-1 sigma-pull, whose zero-compute test reverts to NOT-RUN. Fix mechanism: zero count yields None, the emitter DROPS None fields rather than publishing them, with producer plus mutation tests. Combined with R250, on graph encodings these fields are absent rather than None-as-zero.
Grounds: F-10/LAW-07 class — a monitor input asserting a measurement no producer fed, live on run5's arm.
Status: standing [INLINE]

### R250 — encoding-conditional instruments
Decision: K, cluster variance, coverage and uncovered_forced_win are dense / K-cluster-path concepts, and K is structurally absent on gnn_axis_v1. Standing rule: an instrument for a mechanism an encoding DOES NOT HAVE is ABSENT from that encoding's event stream — never zero, never null-as-value. Item 10's two halves are implemented on the dense path and tick only while a K-cluster encoding is active; they remain owed as LAW-07/18 debts on the control arm.
The ruling's own path MAPPING was wrong and is corrected by R256 (uncovered_forced_win runs on the graph arm); the absence principle is untouched.
Grounds: a zero-reading instrument on an encoding without the mechanism is indistinguishable from a real measurement.
Status: standing — mapping half corrected by R256 [INLINE]

### R251 — ADJ-D22: cadence-based silent disarm closed
Decision: The armed-abort audit computes each armed row's EARLIEST POSSIBLE FIRE STEP from the live cadence keys and FAILS any row whose value exceeds a declared fraction of max_train_steps; that fraction is a schema constant with a live consumer and no code-side default. Deliberate disarm keeps exactly one spelling — the existing explicit R56-style pin — and a large interval is NEVER a sanctioned disarm. FULL pipeline.
Grounds: R242's defect class had simply relocated itself onto its own new knob.
Status: standing — generalized to per-axis sample clocks by R265 [INLINE]

### R252 — R244 rider ratified, with a process note
Decision: The evidence-hygiene rider to R244 is accepted on its measured grounds (a 110 s tier against contention-inflated wall times) and enters the register as R244-a by this act. Process fixed going forward: riders to rulings may be DRAFTED by dispatchers but enter the register only via an architect ruling.
Grounds: the rider was measured rather than asserted, and register entry stays an architect act.
Status: standing [INLINE]

### R253 — Q-FIND-1 disposition
Decision: Binding sequence: (1) the box run completes and batch_fill_pct is read against the pre-registered ~1.56% prediction under its fixed falsification criterion; (2) if the prediction holds, the batching fix (server-side multi-graph collation into the existing segment-batched forward) is AUTHORIZED as a pre-mint item, FULL pipeline, one commit; (3) one IQR-gated before/after bench at matched config, reporting sph, fill and util; (4) the resulting worker/batch/wait values become prereg rows. A falsified prediction sends the work back to the flamegraph with NO fix authorized.
ADJ-D17's correction is accepted as stated: the finding stands, the armed surface is the two dense configs.
Grounds: a fix is authorized by a pre-registered prediction surviving its own falsifier, never by a plausible mechanism.
Status: standing — clause (1) read under "Reading M" by R263, which authorizes Design A [INLINE]

### R254 — KLENT, re-affirmed
Decision: KLENT is a SAMPLE-EFFICIENCY lever, not a throughput lever: it reduces GPU-hours to a given strength (deflated to ~1.3-2.5x for this cost structure) and does not fill batches or raise sph. The binding constraint is throughput, whose levers are Q-FIND-1 (harness), SYS-4 / Gumbel low-n regimes (search budget) and worker/batch config. KLENT ingredients stay post-mint Track B in the ruled order — lambda-returns first, entropy normalization by log|A(s)|, reverse-KL anchoring. The 4x headline stays dead.
Grounds: confound discipline — the ingredients are separable and must not enter together.
Status: standing — refined by R258, which adds the acting-cost effect R254 omitted [INLINE]

### R255 — ADJ-D34: the boot guard derives its bound from the config
Decision: MAX_VISITS = 128 is a literal tunable on an armed path — a hard-rule-4 violation with a mint-blocking consequence. Mechanism: the Phase-T guard's capacity is DERIVED at composition time from the configured sims regime (max over PCR arms), with the schema validating the relation explicitly, so a regime the guard cannot honor is a mint-time error and never a boot surprise. No new literal, no default; the 600/75 values themselves stay prereg rows. FULL pipeline.
Grounds: Phase-T's integrity machinery cannot be bounded by a hand-written constant.
Status: standing — capacity derivation CLEARED and regime-tagged by R275(a); the "retirement clause" once cited to it is NOT in this text, and completed-Q-on-graph admissibility is downgraded to ASSERTED by R278(c) [INLINE]

### R256 — ADJ-D37: R250's mapping was wrong; the halt was right
Decision: The architect had ruled the forced-win-injection instrument onto the DENSE path; measurement shows the mechanism runs on run5's GRAPH legal_set arm and not on the shipped dense grids, so uncovered_forced_win lands on the graph path. Corrected rule: an instrument attaches to the mechanism's MEASURED LIVE PATH, not to the encoding family it was first described under, and R250's mapping is re-derived from code per instrument. R250's absence principle stands; a ledger instance is recorded.
Grounds: landing it as ruled would have produced an instrument reading zero exactly where the drops happen — the F-27 canary shape.
Status: standing [INLINE]

### R257 — Shrimp-Bot reference intake
Decision: Adopted as a reference implementation behind two hard fences: (i) RULES divergence — Shrimp-Bot is the radius-8 game and mantis run5 is radius-6, so architecture and loop patterns transfer while corpora, checkpoints and any radius-dependent arithmetic do not; (ii) ACTING-SCHEME divergence — their net acts search-free via the KLENT operator, and "never search-free at deploy" is an operator lock.
CORRECTED BY ANNOTATION, not by repair: ANNOTATION 9 QUALIFIES fence (ii)'s premise — search-free describes their TRAINING loop, while the deployed checkpoint searches (Gumbel sequential halving, 16-128 sims per stone), which corroborates the deploy lock rather than challenging it. ANNOTATION 10 NARROWS fence (i): the reference's stage S5 is stated at win_length 6 / placement_radius 6 on an unbounded board, this project's exact rule set, so S5-stage material is `[r6-MATCH]` and needs no radius re-derivation (no published figure is at S5, so no figure is released). ANNOTATION 11 INVERTS the radius clause once the registry actually carries the run6 radius-8 lineage: r8 reference material becomes IN-REGIME and r6-only material — including every run5 quantity read forward — carries the divergence note; nothing is thereby scaled or made true unmeasured, and R26 / run5-is-radius-6 stands untouched.
Grounds: their net is separable from their acting scheme, which is exactly PLAN-0's axis separation demonstrated in the wild.
Status: annotated — ANNOTATION 9, ANNOTATION 10, ANNOTATION 11 [INLINE]

### R258 — KLENT speed, refined
Decision: Two distinct speed effects exist and only one was in R254: (i) sample efficiency, ~1.3-2.5x on this cost structure, unchanged; (ii) ACTING COST — the reference bot's KLENT is search-free, so its cost per game is ~1 forward per placement against our 75-600 sims, and that is where its speed actually lives. Effect (ii) is fenced off for run5 by the operator's deploy lock. The admissible middle path, post-mint Track B, is KLENT-style improved-policy targets from a per-action Q head, shrinking the sims a good target needs; pulling lambda-returns forward is an operator prereg re-decision if post-batching ETA is unacceptable.
Grounds: run5 stays clean for confound discipline.
Status: standing — its premise about the reference bot's acting scheme is qualified by ANNOTATION 9 [INLINE]

### R259 — SHAKEDOWN-1 (authorized run class, non-mint)
Decision: A long graph-arm training run from the remediation build is AUTHORIZED as a non-mint run class, with a complete config minted as shakedown_*.yaml and a run-id unambiguously not run5. Purposes in value order: the first live soak of the just-fixed survivability machinery (checkpoint/resume, watchdog saves, promotion/anchor integrity, gate cadence) including a deliberate SIGTERM-save-resume cycle in hour one; the R253 clause-1 batch_fill_pct readout, which opens the Q-FIND-1 gate by itself; the before/after vehicle for the batching fix; and trajectory, ladder and checkpoint data for the WP-AXIS2 falsifiers.
Nothing from a shakedown is a strength claim beyond LAW-15's protocol rules, and no shakedown result arms a prereg value by itself.
Grounds: restarting a shakedown is free — there is no prereg integrity to protect.
Status: standing [INLINE]

### R260 — absence-mode autopilot (the long-horizon dispatcher)
Decision: The long-horizon dispatcher is the architect-level mechanism scaled down: durable state on disk, ephemeral context per packet, never compact — restart. Protocol: an autopilot directory holding MISSION.md (immutable orders), STATE.md (rewritten at every packet boundary with sha, run status, phase, next action), JOURNAL.md (append-only findings and decisions) and AUTHORIZATIONS.md (signed grants); each work packet is a FRESH session that reads MISSION + STATE + JOURNAL tail, executes one bounded packet, updates STATE and exits, with an outer wrapper relaunching.
The run never depends on agent liveness — LAW-16 makes it self-sufficient and the agent monitors the JSONL stream read-only, intervening only per the runbook. HALT-and-queue changes meaning in absence mode: halt the ITEM and continue the mission with the next unblocked one. All law discipline is unchanged.
Grounds: the mechanism already exists at architect level and only needs scaling down.
Status: standing [INLINE]

### R261 — go-package (operator-signed)
Decision: The operator-signed package covering the absence period: R245 is ruled option (c) — the dispatcher implements the per-record symmetry gate during the mission, dense arm only; a branch push grant for remediation and its children with dev frozen and no merges; a box grant of 8 days compute for shakedown and benches; restart authority to relaunch per runbook, capped at N unexplained-crash restarts before the run parks and the mission continues on non-run work; and standing forbiddens — no merge or push to dev, no mint, no armed-value or falsified-register edits, no host/config changes beyond the runbook, no force-push, no R20-surface changes.
Operator approval is recorded verbatim on 2026-08-07 ("operator approves those"), signing R259, R260 and this package.
Grounds: without the signed package the mission shrinks to local-only work.
Status: standing — the per-event grants inside it expired on completion [INLINE]

### R262 — hierarchical autopilot orchestration
Decision: MAIN is a thin Fable-class orchestrator whose first act is to write MISSION.md, STATE.md, JOURNAL.md and three track states, then launch sub-dispatchers SD-RUN, SD-FIX and SD-DEV with briefs derived from mission phases M0-M4, using Opus-class leaves for impl/design/diagnosis and Sonnet-class for mechanical/monitoring. Enforced: a capability floor plus cross-model review on FULL items (§2); SD-FIX is the SOLE code committer and leaves return diffs and transcripts only (§3); artifact-mediated hand-offs with pins re-run before acceptance (§4); at most 3 leaves per sub-dispatcher plus a tier and box lock file (§5); forbiddens bind every node and HALTs escalate inward and PARK rather than stall the mission (§6).
MAIN holds no implementation context: it orients each packet from MISSION + STATE + JOURNAL tail + repo at HEAD, rewrites STATE, and relaunches rather than compacts.
Grounds: R262's own numbered text exists nowhere on disk — the register carries the MISSION's APPLICATION of it, which is a rendering and not the ruling (the R267 precedent).
Status: standing — fidelity call ratified as CORRECT and standing by R272(a); supersedable only by the exported transcript [RECOVERY-PROVENANCE / SUMMARY-ONLY]

### R263 — R253 Reading M adopted; Design A authorized
Decision: The R253 clause-(1) gate was always about the MECHANISM and not the literal w1 bracket — occupancy capped at exactly n_workers, the collector threshold structurally unreachable, every pop burning its full wait — and that mechanism was confirmed at w20 to the integer, matching its own w20 prediction. Reading L is REJECTED and Design A is AUTHORIZED. LAW-09 rider: the ~15 ms/pop unattributed overhead gets flamegraph attribution in the SAME packet as Design A's bench, because it bounds Design A's ceiling and must not be discovered after the verdict; one change = one commit = one IQR bench still holds, since attribution is measurement, not change.
Grounds: Reading L would elevate a bracket's authoring context above the mechanism it predicted.
Status: standing [INLINE]

### R264 — controlled resume re-verification on the live burn
Decision: A deliberate SIGTERM-then-resume on the CURRENT ring is authorized during the live burn, as the cheapest live proof of the resume fix at f9f9eee and as the restoration of restart-with-ring capability. A failure is an explained crash, a fresh ring and a root-cause packet with the budget intact; until it passes, fresh-rings-only stands.
Grounds: soaking survivability is the burn's stated purpose.
Status: standing [INLINE]

### R265 — ADJ-D38 mechanism (generalizes D36/R251)
Decision: The WR-consec abort is unfireable because gate 12 audits its fire-step in the TRAINING-STEP clock while WR samples arrive in the EVAL-ROUND clock. Rule: the fireability audit computes each armed row's earliest possible fire in THAT AXIS'S OWN SAMPLE CLOCK — draw-rate in gate-interval steps, WR in eval rounds times their real cadence — derived from live cadence keys. D36's derivation pattern extends; values stay blank; FULL pipeline.
Grounds: no axis is auditable in a clock it does not tick in.
Status: standing [INLINE]

### R266 — ratifications
Decision: ADJ-D36 is ratified as landed with bit-identical behaviour preserved at <=32; D35 is closed per the S2-stands memo; findings F-R-P4-1 and F-R-P2B-2 are accepted as closed; and the R245(c) disclosures are accepted, with the LAW-18 augmentation-group counter OWED before any dense-arm training and ownable by a dispatcher.
Grounds: not stated in the register.
Status: standing [INLINE]

### R267 — GAP: no ruling text in the register
Decision: No R267 section exists. The register states the gap explicitly: the only surviving record is a STATE_2026-08-16 §5 digest line — "R267 eval posture mechanism inert, values operator" — corroborated by the autopilot STATE file ("PKG-5 eval posture (R267) COMPLETE, landed (inert)") and plan/EVAL_POSTURE_OPTIONS.md. It is deliberately NOT reconstructed, because a digest line is not the ruling, and it is to be filled from the exported chat transcript.
Grounds: a rendering is not the ruling — the precedent this gap sets, later applied to R262 and cited by R272(a).
Status: GAP — no register section; text still owed on the operator's residue list at R299(f)/R300(h)

### R268 — F-816-1: no crash slot consumed
Decision: Accepted as recommended: the last act was a normal step with no software error line, so a box-external cause of the OOM-killer class is more likely than one of ours, and NO unexplained-crash slot is consumed. Ring restarts are proven, so relaunch is cheap when wanted.
Grounds: nothing in the run record points at a fault of ours.
Status: standing [RECOVERED VERBATIM]

### R269 — F-816-6 is the headline; the bootstrap path becomes MINT-CRITICAL
Decision: Three findings in one row. (a) draw_rate pinned at 1.000 — at bootstrap-from-scratch strength neither side ever completes six, so every game runs to the ply cap at cap x sims forwards for zero learning signal, meaning part of "extremely slow" is actually "degenerate". (b) ply 0.00 at every check cannot be literally true of cap-length games — a suspected phantom channel of the F-10/R249 class, to be investigated before any ply-derived statistic is trusted. (c) Zero solver fires must be reconciled against config disarms before being read as a defect.
Consequence: the bootstrap path is promoted to MINT-CRITICAL — the R247 corpus (BC pretrain or corpus-mix warm start) is the mechanism that makes early self-play decisive, while ply-cap/draw-abort posture surgery only treats the symptom.
Grounds: run5-as-minted aborts on draw rate around step 25k either way, so the decision is owed now and has measured grounds.
Status: standing [RECOVERED VERBATIM]

### R270 — handoff; STATE_2026-08-16 supersedes STATE_2026-08-06
Decision: This architect context is past its reliable horizon and hands off; the STATE_2026-08-16 snapshot SUPERSEDES STATE_2026-08-06. One caution is flagged inside it: rulings R236-R270 were issued in-chat and may not all be present in the register, so the next session's FIRST act is verify-and-fill with the chat transcript as the recovery source, on the R172/R195 precedent.
Grounds: a context past its horizon hands off rather than continues ruling.
Status: standing [RECOVERED VERBATIM]

### R271 — register hygiene: archive/index split
Decision: (a) The rulings register is the append-only VERBATIM ARCHIVE — never compressed, rewritten or pruned; only recovery-provenance fills and status annotations touch it. (b) RULINGS_ACTIVE.md is the DERIVED working index; sessions seed from ACTIVE + laws.md + CLAUDE.md and consult the register at point of use, and absence from ACTIVE claims no forward force and deletes nothing — ACTIVE is never authority and the register wins on conflict. (c) A ruling stating a durable rule GRADUATES into laws.md or CLAUDE.md by normal amendment commit, its ACTIVE row collapsing to a pointer and then dropping.
(d) ACTIVE is curated at session close by the register-pen holder, one log line per curation. (e) The 14 single-ruling batch headers may be normalized in one mechanical docs commit, zero substance.
Grounds: an index that can be mistaken for authority is how a governance record drifts.
Status: standing [INLINE]

### R272 — R271 execution ratified
Decision: (a) The census, both foot appends, the 14-header normalization and the seed-range corrections are accepted as landed, and R262's [SUMMARY-ONLY] fidelity call is CORRECT and standing — a mission's application of a ruling is a rendering, not the ruling. (b) The R259 restore and the R23-R31 authority qualification are ratified, and the practice joins R271(d): every curation SPOT-CHECKS at least 5 index lines against verbatim register text, and an index line may never claim more than its ruling. (c) Two config-cited BLOCKING values are pinned to the prereg batch with their evidence — checkpoint_interval (run5.yaml mints 0, not a legal production posture) and random_floor_games (mints 0) — both authored, both operator-owed. (d) The dispatcher's R119 restraint on the armed R147 value was correct behavior, on the record.
Grounds: an overclaiming index line is the F-43 misstatement class in miniature.
Status: standing [INLINE]

### R273 — R272 execution ratified; hygiene program CLOSED
Decision: The git posture is accepted as standing — whitelist ignore, sha-manifests-in / blobs-out, snapshot-not-replay history, repo-local identity. The register-hygiene program is CLOSED: what remains is operator-input-gated rather than work-gated (the R267 transcript, the ADJ-D2 / R227-R228 text), plus the graduation-absorption queue, deferred as a batchable dispatcher packet rather than session work. R240's ordering now governs: correctness, then architecture, then perf.
An owed item is recorded beside it: one verbatim check of R160's index line when the register is next in the operator's context.
Grounds: the remaining items need operator input, not more session work.
Status: standing [INLINE]

### R274 — VisitSlotsExceeded routing (foot append, filled verbatim)
Decision: (a) Registered as F-816-9 from the exit-report evidence verbatim — an EXPLAINED run-fatal defect that consumes NO unexplained-crash slot. (b) MINT-BLOCKING, correctness class; under R240 it outranks and blocks the bench. (c) Neither the R255 check nor the producer is presumed right — Phase A measures which: capacity derived in the wrong axis/unit means fix the derivation and KEEP the check, while a producer genuinely writing mass beyond the sims bound puts the fix on the producer; a capacity raise without a measured mechanism is BANNED and deleting or softening the check is BANNED.
(d) Bench re-run protocol: after side pinned to 8ba2d0d against its parent, matched config, one change one bench. (e) F-816-2 rides the same packet as an independent card, separate commits, per-item verification, no shared-commit bundling. (f) The fix ships with the test the tiers lack — a deterministic ply-cap-length game through the production record path.
Grounds: LAW-14 keeps this run-fatal, and the measurement decides which side is wrong rather than the presumption.
Status: standing [INLINE]

### R275 — F-816-9 Phase A ratified; PRODUCER-BUG stands; Phase C GO
Decision: (a) The R255 capacity derivation is CLEARED and regime-tagged — valid for the current visit-limited target construction, re-derived with both pins if completed-Q-on-graph is adopted at prereg. (b) The class refines to two conjuncts: a SEAM (a failed inference never reports completed — run-fatal and loud, with inference_failures_total reaching the event stream in-run) and an EXPORTER (a zero-visit search cannot produce a target; refusal is loud and named), with flip-sets covering both plus the healthy ply-cap game and the exact-capacity boundary.
(c) The sims prereg row is BLOCKED on this fix landing; checkpoint interval, random floor, NaN arming, gate consecs and corpus are NOT blocked. (d) Trigger forensics ride the next box session — grep the five dead after-reps' logs before anything is deleted; a confirmed GPU-failure-under-batch-fusion routes to the CARD-RUN5-GPU-OOM class as its own item. (e) A post-fix bench re-run may legitimately yield numbers OR a loud named inference failure — the second is trigger confirmation, not a failed errand.
Grounds: at 600/75 the defect is silent, so tripwire sensitivity is exactly what blocks the sims row.
Status: standing [INLINE]

### R276 — Phase C exit adjudication (merge gate, grants, OOM routing, ply cap)
Decision: (a) f816-scratch is approved for merge conditional on the closure-typing cite. (b) Sequential reviewer isolation is adopted as an R262 rider — concurrent review VOIDS the later verdict unless re-verified in an isolated worktree. (c) A retroactive per-event R43 grant for the target_latch_propagation.rs edit, disclosed same-event, never precedent. (d) F-816-2 is reclassified telemetry-only and the correction propagates to every artifact carrying "feeds an armed abort". (e) The capacity guard is retained as defense-in-depth. (f) The OOM trigger is CONFIRMED as F-816-10 in the CARD-RUN5-GPU-OOM class, with a design-first memory-bounded fusion packet authorized — loud and counted, no silent catch-and-retry without a counter and a cap — and the bench unit openly redefined as parent vs (Design A + memory bound) as ONE deployable unit, since Design A cannot run without the bound.
(g) The ply cap stays 128 through bootstrap-fix validation; the cap VALUE is a prereg row decided JOINTLY with the adjudication criterion from the corpus audit's measured length distribution in plies, never from a precautionary round number.
Grounds: at draw_rate 1.000 the cap is the only thing bounding a worthless game's cost — 200/128 buys 1.56x more forwards per degenerate game for zero additional signal.
Status: standing [INLINE]

### R277 — merge gate ratified + grants
Decision: (a) The merge-gate HALT is RATIFIED as correct and load-bearing — the closure-typing hole was real, reachable and documented as intended behavior — and the fix in 461728b (discriminator as runner kill-switch, ordering running=false before either close, pool stops runner before server) is APPROVED. (b) Per-event operator grant: merge f816-scratch to dev fast-forward and push, gates green first, expiring on completion, never precedent. (c) Containment-gated cleanup: delete only branches verified fully contained in dev by ancestor check; anything not contained is reported, never deleted.
(d) _DEFAULT_MAX_PLIES is registered as F-816-11 in the R255 MAX_VISITS class — an unconfigurable literal, adjudicator-coupled, silently divergent against selfplay.max_game_moves — with the fix direction fixed by hard rule 3 (ONE resolver; eval reads self-play's seam; a deliberate split needs its own prereg key), riding the ply-cap prereg packet. (e) Routing protocol adopted: every ruling ships with a ROUTE plus a follow-up prompt.
Grounds: the classifier that reads no queue retires the wrong-queue class by construction.
Status: standing [INLINE]

### R278 — R277 execution ratified; R274 fill, audit certification, branch adjudication
Decision: (a) R277 execution ratified, the R274 NOT-FILLED halt and the exit-code self-catch recorded as the discipline working. (b) R274 is filled from the architect's verbatim text, with a rider to R277(e): every ruling's follow-up prompt CARRIES THE RULING TEXT VERBATIM for same-event register append. (c) Completed-Q-on-graph admissibility is DOWNGRADED to ASSERTED — the "retirement clause" traces to a STATE digest line, not to R255's registered text — exercisable at prereg only by operator re-affirmation. (d) The audit tool is certified-before-cited: fix OPEN-2 (manifest shape), OPEN-5 (elo pair) and OPEN-7 (game_id absent) and re-run before its numbers enter a prereg row.
(e) The five worktree-agent-* branches get per-branch content-equivalence adjudication (tree-equivalent means archive-tag then delete; unique content is reported). (f) The engine repo's origin switches to the SSH URL on this machine. (g) Evidence-citing reports verify cited files are tracked at commit time. (h) The ply cap stays 128 now, with cap x adjudication decided jointly on the certified distribution and F-816-11 as precondition.
Grounds: prereg grounds must be CERTIFIED, and an index digest is not register text — the third instance of the index-overreach class.
Status: standing [INLINE]

### R279 — corpus CERTIFIED as prereg grounds; audit-merge, disposal, identity and push grants
Decision: (a) R278 execution ratified; certified nearest-rank values govern (p99 263, p999 523) and the off-ladder hand-pass Elo quartiles are correctly withdrawn. (b) The corpus is now CERTIFIED GROUNDS for the bootstrap and ply-cap prereg rows — exit 0, sha match, 8698/8698 distinct, winner convention replay-verified, OPEN-2/5/7 closed with refusal-pinned amendments. (c) Archive-tag-and-delete GRANTED for a8dc82d6, dev strictly superseding by range-diff. (d) The engine repo gets a repo-local git identity.
(e) Per-event merge grant for r278-audit-scratch to dev after gates, pushed over SSH. (f) A standing push grant for the migration docs repo to its private remote, engine-repo pushes staying per-event. (g) Ply cap x bootstrap x adjudication decide as ONE matrix, values operator-owned, F-816-11 as implementation precondition, the turn-boundary value derived at point of use — NO value is registered by R279, and (g)'s "as stated above" antecedent was never supplied, so the register recorded a stub.
Grounds: a decision whose halves are valued separately is decided backwards.
Status: standing — (g)'s reasoning of record is supplied by the R279(g)-ANNEX under R280(b) [INLINE]

### R279(g)-ANNEX — the ply-cap x bootstrap matrix, self-contained
Decision: The ply-cap and bootstrap prereg rows decide as ONE matrix. (i) If a warm start is taken (BC-pretrain or corpus-mix), the self-play cap rises to the ~256-ply CLASS — exact value derived at point of use on the engine's turn boundary — covering p99 = 263 of certified human decisive lengths and cutting truncation label-noise from 7.335% to ~1%, with cost binding only on the tail once play is decisive. (ii) Posture-only means the cap STAYS 128, because in the degenerate regime every added ply is waste and the draw-abort bounds it. (iii) eval.ply_cap_adjudication arms in EITHER branch, with criterion and min_margin operator-valued and the seat-neutrality asymmetry disclosed. (iv) The certified tail is a FLOOR on the true tail (decisive-only bias). (v) F-816-11's single-resolver fix is the implementation precondition. (vi) The deploy/bridge cap is a separate knob, with training cap >= deploy expectation.
No VALUE is registered by the annex, and R276(g)/R278(h) are unamended — the cap STAYS 128 until the prereg row values it.
Grounds: R280(b) — register-bound ruling text carries no deictic references, and an annex under the issuing ruling's number is the sanctioned repair path.
Status: standing — REPLACES the R279(g) stub's force; the stub's honesty note stands unedited [INLINE, annex to R279(g)]

### R280 — R279 execution ratified; F-816-10 ratified + merge grant; rule-7 tag scan
Decision: (a) R279 execution is ratified: prereg-row creation was R119-clean dispatcher work, and the zero-headroom test-count floor is the ratchet working as designed. (b) The R279(g)-ANNEX repairs the dangling antecedent, with a protocol rider to R277(e): register-bound ruling text carries NO deictic references, and an annex under the issuing ruling's number is the repair path. (c) A rule-7 scan of the five public archive tags — trees, messages, diffs — precedes any further push, with hits deleted remotely and the local archive kept. (d) F-816-10 is RATIFIED, its equal-length transposition pin recorded as the packet's most valuable find, with a merge grant effective on operator forwarding.
(e) R43 ratification for the FROZEN-table edit is WITHHELD pending the two queue rows read verbatim; the edit stands meanwhile on its stated grounds. (f) A worktree prune grant, verify-then-remove. (g) The mint path restated: merge, box sitting, bench verdict, prereg batch, R61 preflight, MINT. Landed section headers STAND, and the forward-only header convention is "architect adjudication response, operator-ratified by forwarding".
Grounds: disclosed-but-unread is not granted.
Status: standing [INLINE]

### R281 — push hold ratified; rule-7 SANITIZE-FORWARD; gate 17; F-816-12 mint-critical
Decision: (a) The push hold is RATIFIED as correct — the scan's premise-falsification was surfaced, not absorbed. (b) Rule-7 disposition is SANITIZE-FORWARD: the fixture's host paths are neutralized in a FORWARD commit with the manifest sha updated and the consuming test verified path-insensitive first; a history rewrite is REJECTED because sha citations across the governance record are load-bearing, and residual old-history exposure is accepted as path-shape-only. (c) Gate 17 is AUTHORIZED — the scan's pattern set becomes a repo-local CI gate over added/modified files with one full-tree baseline at adoption.
(d) F-816-12 is RATIFIED MINT-CRITICAL in the CARD-RUN5-GPU-OOM class: the two caps are ONE partition and train.microbatch_caps and inference.fused_graph_caps are fitted JOINTLY from measured terms, derivation comments carry sha and regime tags at re-mint, and a boot-time partition assertion (measured card >= declared partition sum) is authorized; no separate code packet. (e) F-816-13 is GRANTED per the R276(c) shape, per-event, never precedent.
Grounds: a partition is one object — certifying its halves independently certifies nothing (the fourth instance of premise-moved-under-a-green-gate).
Status: standing — (b) and (e) are per-event grants, EXPIRED on completion [INLINE]

### R282 — R281 execution ratified; delegation boundary; BOX SITTING authorized
Decision: (a) R281 execution ratified, including gate 17's false-clean self-catch, the regex-defect self-test kill, the empty-EXEMPT deviation, and the untracked-supplement design (tracked floor, untracked ceiling). (b) DELEGATION BOUNDARY: the architect decides routine adjudications, grants and dispositions after pros-cons-and-red-team, on the record; PRESERVED OPERATOR-ONLY are box grants, run5 mint authorization and judgment-valued prereg rows, while measurement-derived cap values mint IN-SITTING under pre-registered acceptance (calibration falsifier PASS and partition inequality holding and the fitted pair in the tool's recommended form), any deviation halting with the numbers. (c) The BOX SITTING packet is authorized to a fresh dispatcher, box grant activated by operator forwarding, artifacts landing in the migration workspace only, with STEP 3 minting run5 AND shakedown and closing F-816-12.
CORRECTED BY ANNOTATION 12: clause (b) RESERVES NO PUSH — pushes appear nowhere in its operator-only list, and the contrary reading lived in the ACTIVE index rather than in this text; pushes are governed by R311(b) (full local gate set green, remote CI suspended), and R170's MERGE half is untouched.
Grounds: measurement-derived values may mint in-sitting because their acceptance criterion is pre-registered; judgment-valued ones cannot.
Status: annotated — ANNOTATION 12; (c)'s "band from QFIND1_READOUT §2" clause was unsatisfiable and is SUPERSEDED by R283(c), the rest of (c) standing [INLINE]

### R283 — box-sitting HALT ratified; bench pre-registration AUTHORED and binding
Decision: (a) The HALT is RATIFIED on both blockers, with three architect defects ledgered: the un-carried R282 body (violating the architect's own R278(b) rider), a rung-2 transcription error (the ladder says rung 1), and an instrument-shape mismatch in the before-side pin. (b) R282 lands via this prompt verbatim, plus a rider: a dispatch citing an UN-CARRIED ruling number HALTs at Task 0 by rule, not by judgment. (c) The bench pre-registration is AUTHORED and BINDING — instrument bench_side.sh burst 5x21 at matched config; before side the banked burst-shaped bench_before; P1 mechanism-engaged means after occupancy.max > 20 (P1 false means defect investigation and no perf verdict); P4 verdict on median gph against before 138.43 with BUILD >= 208, PROTOTYPE 166-208 with P1 true, and STOP/INVESTIGATE below 166 or on any after-median regression; P2 and P3 are recorded diagnostics, never gates; the declared unit is the bundle plus minted caps.
(d) The calibration falsifier substitution (synthetic-graph calibration, with STEP 4 gaining a real-graph peak-memory falsifier against the minted budget) is ACCEPTED and recorded, never silent. (e) Flamegraph wrap-launch only. (f) A per-event gate-17 reporting fix printing counts, never local contents. (h) F-816-12 remains OPEN until STEP 3.
Grounds: a pre-registration's legitimacy comes from being written BEFORE the measurement, not from where its numbers descend.
Status: standing [INLINE]

### R284 — LAW-09 bench verdict STOP/INVESTIGATE; perf packet targets ordered
Decision: The verdict is STOP/INVESTIGATE exactly as pre-registered — P1 TRUE, P2 FAIL, P3 PASS, P4 0.911x with non-overlapping IQRs. (a) The mechanism engaged and throughput REGRESSED; the mechanism of record is latency-per-leaf-batch-round-trip, not GPU fill, for fixed-worker MCTS self-play. (b) Investigation targets in order: P-MASK (a sync-free gather replacing boolean-mask indexing, output-parity-oracled), P-CHECKS (_check_structural moved to Rust or amortized to O(1) per batch), then the pure-forward microbench only if the first two do not close it. (c) The minted caps STAND — they are correctness, not perf, and zero OOM at peak closes F-816-10's validation. (d) Attribution escalation is DECLINED. (e) PCR 600/75 is unviable on measured numbers, so the sims row waits for the re-bench or mints in the 50-sims class. (f) F-816-14 (eval-child orphan surviving parent SIGTERM) is queued and the R46 loop is ORDERED on the preflight flake.
CORRECTED BY THE ANNOTATION AT R284's FOOT (ordered by R285(c)/(d)): clause (b)'s "hard rule 12" citation has NO referent — the doctrine home is repo_design.md §10's perf-doctrine bullet, whose "contract validation on marshaled arrays" covers _check_structural — and R284's canonical text is the register's appended block, the chat variant being a SUPERSEDED DRAFT; both are architect-ledger defects, annotated and never repaired.
Grounds: the serve thread's per-batch Python overhead raised latency, and no amount of fill compensates for that.
Status: annotated — R284-foot annotation (R285(c)/(d)); (e)'s "best taken AFTER the re-bench" sequencing SPENT by R300(c), the sims fork being open now [INLINE]

### R285 — R284 execution ratified; F-816-18 GRANTED; AUDIT-BEFORE-REBASELINE ordered
Decision: (a) The R284 packet is ratified — parity-oracle discipline, the pre-registered no-local-signal prediction, the orphan catch with re-measurement — and the sync-migration trap is named of record: the 36.22% box frame may be misattributed pipeline wait, and the re-bench's two falsifiable predictions are the decision instrument. (b) F-816-18 is GRANTED on bandwidth-floor reasoning, its prediction riding the re-bench. (c) The "hard rule 12" citation is corrected of record by ANNOTATION, never repair. (d) R284's canonical text is the register's appended block and a ruling has ONE TEXT — the packet-embedded one.
(e) F-816-16 is granted ONLY IF its subject is exactly the assertion-map delta (check 8 reformulation, check 9 range guard, check 13 addition, zero removals); any other content HALTs before merge. (f) A merge grant conditional on (e) resolving clean. (g) F-816-15 AUDIT-BEFORE-REBASELINE is ORDERED — all 39 reds classified before any manifest touch, its own packet. (h) A PDEATHSIG-class fix for the supervisor-unarmed orphan is ordered PRE-MINT. (i) The re-bench box event is authorized on merge.
Grounds: a blind re-baseline of the freeze registers would launder exactly what the tool exists to catch.
Status: standing — (e) resolved MISMATCH/HALT the same day so (f) and (i) did not vest; SUPERSEDED IN OUTCOME by R286(a)/(b)/(d), with (e)'s LAW untouched [INLINE]

### R286 — F-816-16 HALT ratified; both frozen edits GRANTED on their actual subjects
Decision: (a) The F-816-16 HALT is RATIFIED: the diff-not-description test caught two independent failures — the architect's mis-scoped conditional (R285(e) named a subject frozen nowhere) and the row's incomplete census (one of two frozen edits undisclosed, mechanism measured as a single-register hand-read while the verifier was red on both paths). (b) BOTH frozen edits are GRANTED on their ACTUAL subjects, per-event and never precedent, conditioned on the R43 row being amended to name both paths with the census-error mechanism recorded. (c) De-triplication is ORDERED — one shared stub module imported by all three eval test files, frozen once, riding the F-816-15 audit so freeze surgery happens exactly once.
(d) The R285(f) merge grant RE-VESTS on (b), gates already green. (e) The re-bench box event vests on the merge; the 0.16%-vs-0.31% prose discrepancy is ledgered on both sides with the record's figure governing and nothing in the argument turning on it. (f)/(g) The F-816-17 and F-816-20 routings are RATIFIED as filed.
Grounds: the signature change is forced by the production seam, the equal-count argument is checkable and enforced by check 13, and no assertion, golden or expectation moves.
Status: standing — (c) RE-SEQUENCED (not rescinded) by R288(d) to run after RQ-1 lands [INLINE]

### R287 — MISSION CLEAN-SWEEP established; perf lane of record
Decision: (a) R286 execution ratified, and the figure-provenance convention is adopted from its founding case: every measured figure carried across documents NAMES ITS PRODUCING RUN. (b) MISSION CLEAN-SWEEP is established under R260/R262 — an autonomous loop with durable on-disk state, ephemeral packet contexts, subagent leaves under sequential isolated review, and red-team-adjudicated decisions with pre-registered verdicts; mission-scoped grants signed by operator forwarding cover commits, gate-green merges to dev, pushes on both repos and remote-CI repair including workflow files (gate logic stays in tools/ — the workflow only calls it), while armed values, self-granted frozen edits, box access, architecture refactors and bootstrap/prereg decisions are ALWAYS excluded and queue instead.
(c) The perf lane of record: consume the re-bench readout, then the trainer-side profile, then conversion/pinned-H2D candidates with local mechanism proofs, then Rust criterion at run5 shape, with bucketed-compile design-only and graves fenced item by item. (d) D1 of the definition-of-done is remote CI VERIFIED green on dev HEAD — checked, not assumed. (f) Runaway-scope fences: an enumerated definition-of-done, an item-size tripwire, a checkpoint report at every merge, operator stop at any time.
Grounds: an autonomous loop needs its authority enumerated up front and its scope fenced, or it grows past both.
Status: standing — the mission-scoped grants expired with MISSION CLEAN-SWEEP [INLINE]

### R288 — MISSION CLEAN-SWEEP ratified; RQ-1 as a class; F-816-23 diff-scoped
Decision: (a) MISSION CLEAN-SWEEP is ratified in full, with the Q2 Option-B falsifier firing pre-execution, the CI evidence-surface-first method, the PDEATHSIG cycle with measured rcs, and the first trainer profile in project history on the record. (b) RQ-1 is GRANTED as a CLASS: the 34 REAL-DRIFT holds ratify held-to-OK wherever a path's annotation traces to a merged, reviewed packet, and any path whose citation fails to trace STAYS HELD. (c) F-816-23 is GRANTED diff-scoped to the exact diff carried on the row, per-event, never precedent, on one condition — the diff touches no production config and no arming authority, else HALT; this is the D1 key.
(d) R286(c) is RE-SEQUENCED, not rescinded: de-triplication proceeds AFTER RQ-1 lands. (e) Falsified-register scope ruling: F-17/F-19 are scoped to their measured subject (legal-set maintenance on descent paths, dense-era regime) and explicitly do NOT cover eval caching, root-level increments, transposition reuse, or incremental axis-graph construction from parent — the last registered as candidate INCR-GRAPH, gated on box measurement with the F-19 inequality as its pre-registered falsifier. (f) Batch adjudication is ORDERED: all RQ rows surfaced verbatim in ONE document for a single ruling pass.
Grounds: consolidation must consolidate RATIFIED content instead of laundering unratified content, and a fence is a re-litigation tax via the LAW-02 valve, never a prohibition.
Status: standing [INLINE]

### R289 — the RQ batch ruled; premise-verification made a precondition
Decision: 23 lettered clauses dispose of 26 RQ rows, all binding to each row's VERBATIM text under a standing precondition: before executing any lettered clause the executor verifies the clause's stated premise against the row's own text, and a mismatch HALTs that row back to the architect while the rest proceed. Substance: freeze pins reference COMMITTED content only, minted in the same commit that lands the frozen file (a), and a new freeze row may pin only reviewed-and-merged or explicitly granted content (b); an independent fresh-context cross-model review of freeze_verify.py is ORDERED (c); freeze_verify becomes CI GATE 18 as RQ-21 (d) and row 30's freeze is re-evaluated as RQ-22 (e); the supervisor kill-grace is SPLIT — floor derivation architect's, VALUE an operator prereg row (g); wrapper depth >= 2 REFUSES with a named error (i); the supervisor dies OF its signal, exiting 128+n, with rc-substitution banned (j); dead legal_mask removal proceeds as LAW-08 hygiene (n); bench floors are host-attested and cross-host comparisons inadmissible for verdicts (o); a compiled-arm parity criterion is pre-registered with no post-hoc widening (p); the mixed-batch/pretrained-buffer path is RESERVED, not dead (q); a CI-runtime packet is authorized (s); and the lint gate refuses LOUD on a missing interpreter (u).
(v) Protocol, PERMANENT: a ruling's text exists in EXACTLY ONE place — the packet that lands it — and chat summaries carry no clause text, closing the double-carriage class. (w) Consolidation into three packets: freeze-governance, hygiene, CI-runtime.
Grounds: the R286 diff-not-description rule, institutionalized — a disposition is only as good as the row text it was written against.
Status: annotated — ANNOTATION 1 (the packet preamble's "(t) ends the double-carriage class" is a MISREFERENCE; the clause is (v), and (t) disposes of RQ-b2/RQ-c) and ANNOTATION 2 (clause (a)'s second landing site, the ORACLE-WRITE leaf instructions, did not exist anywhere and was ordered CREATED by R290(d)(3)) [INLINE]

### R290 — R289 execution ratified; RQ-8 ESCALATED; freeze packet re-scoped
Decision: (a) The R289 execution is ratified, and the two premise-verification halts are the mechanism's proof of value — the RQ-16 halt prevented an architect-ordered deletion of code with live read-sites. (b) RQ-8 is ESCALATED and re-scoped: its original claim is FALSIFIED (all MonitorConfig fields are schema-resident) and the real defect is a bare-MonitorConfig() construction path that bypasses the minted config and has SILENTLY DISARMED an armed abort; before any packet inherits the row, the disarmed abort must be named, every bare-construction call site enumerated, and production reachability decided — a production-reachable bypass is filed immediately as MINT-BLOCKING, a test-only one routes to hygiene. (c) RQ-16 is re-scoped to PER-TENSOR adjudication with a reachability census over production entry points, since read-site grep alone proves nothing in either direction.
(d) The gate-18 HALT is ratified and the freeze-governance packet is RE-SCOPED into six ordered steps: rule-7 scan, relocation of registers and tool into the engine repo, freeze policy applied with the leaf-instructions document CREATED, de-triplication, gate 18 wired natively, then the RQ-22 row-30 evaluation. (e) An architect order naming a frozen path carries the R43 grant for exactly the ordered change, and same-act re-pinning is a TERM of every such grant. (f) Two register-foot annotations ordered. (g) Operator item: authenticate gh on the working host.
Grounds: the same-act discipline does not survive being known, only being checked — which is gate 18's mandate.
Status: standing [INLINE]

### R291 — F-816-24 RATIFIED MINT-BLOCKING; the LAW-08 citation defect fixed as a CLASS
Decision: (a) R290 execution ratified; the architect's premise conflation is ledgered, and the census method — grep AND AST, with a measured 31% grep over-report — becomes the standing method for construction censuses. (b) F-816-24 is RATIFIED MINT-BLOCKING: the supervisor entry surface reads no minted config, so prereg row 19's safety bound cannot discharge and config-authored grace values reach no process. A fix packet is ORDERED under full pipeline with cross-model review — a REQUIRED --config whose absence is a named error, monitor config resolved through the ONE resolver with bare construction unrepresentable or loudly refused, a LAW-07 producer test that launches the supervisor via -m with a distinctive minted grace value and asserts it LIVE in the process plus the refusal test, the landed signal posture untouched, and an exit carrying a 14-surface config-reachability census.
(c) The LAW-08 citation-prose defect is fixed as a CLASS: the consumer-citation checker verifies every arrow by SYMBOL REFERENCE, never by prose. (e) R289(c)'s freeze_verify review is slotted after relocation and before gate wiring. (h) Protocol maturation of R289(v): a landed ruling is READ FROM THE REGISTER at point of use and copied nowhere; dispatches cite the number and the file, never the text.
Grounds: a safety bound whose values never reach the process is unenforceable as shipped.
Status: standing [INLINE]

### R292 — the R79 duplicate-authority half SPLIT; hygiene stop LIFTED
Decision: (a) The packet-authorship overlap check — shared 7-word runs against the register, 32 found in the first draft and rewritten before landing — is ADOPTED as a standing check for every future packet, with each packet's commit subject recording its result. (b) The R79 duplicate-authority work SPLITS: the F-816-24 fix packet owns the supervisor site (the required --config, the single-resolver path, the refusal-or-unrepresentability mechanism as built there), while the hygiene packet owns the class-wide extension across every remaining bare-construction site and takes that ONE item up only after the F-816-24 packet merges; THE HYGIENE STOP IS LIFTED for every other hygiene item.
(c) The F-B1 re-opening is ratified as DESIGN input: a supervisor reading its own config re-opens the parent-child same-file binding question, config_identity_sha256 is the named candidate, and DESIGN answers it explicitly with RED-TEAM verifying the answer exists. (d) R285(d)'s two-copy check is NOTED-NOT-RETIRED — it stands at zero cost over the class R291(h) empties, catching regressions of the protocol itself.
Grounds: the anti-carriage discipline now has an instrument rather than an intention.
Status: standing [INLINE]

### R293 — reconciliation-grep convention; gh DISCHARGED; the architect desk EMPTY
Decision: (a) R292 execution ratified; the overlap check's first production use caught its own author (18 shared runs in the hygiene draft, 0 at landing), and one convention line is adopted: replacing a section carries a same-edit grep for references to the replaced content, reconciled in the same act. (b) The gh authentication evidence is accepted beside the clause with nothing edited, so freeze-governance step (2)'s precondition is DISCHARGED and the packet dispatches under its existing R290(d) authority — no new grant exists or is needed. (c) The architect desk is EMPTY: no open dispositions, no halted rows, no owed adjudications, with four packets forwardable in any order and the box sitting operator-scheduled; the owed R227/R228 and R267 texts and the prereg values remain exactly where they were.
Grounds: an empty desk is a statement about the QUEUE, not about the work.
Status: standing — (c)'s "in any order" superseded by R295(c)'s fixed dependency order, and further fixed for two packets by R306(d) [INLINE]

### R294 — the scope stretch adopted as principle; re-check-at-dispatch
Decision: (a) The R293 execution is ratified INCLUDING its disclosed scope stretch — repairing sites outside the landing's own files in the same act was correct, and it is adopted as principle: a convention lands together with the repairs its adoption sweep finds. (b) The re-check-at-dispatch instruction on discharged preconditions is ratified — a discharge is evidence at a moment, and dispatch re-verifies it. (c) The PAST-TENSE EXCLUSION is ratified: "(was: ...)" state history is provenance and never a stale live claim, so the reconciliation grep's scope is LIVE ASSERTIONS ONLY. (d) The desk remains empty and the board is unchanged from R293(c).
Grounds: an unbounded stale-claim grep would, over successive curations, consume the change log it exists to protect.
Status: standing [INLINE]

### R295 — MISSION FINISH-LINE established; sequencing, not authority
Decision: (a) R294 execution ratified with both conventions' first-run arguments on the record, including that boards are POINTED AT, never copied forward. (b) MISSION FINISH-LINE is established over the four authored packets (supervisor-config, hygiene, freeze-governance, CI-runtime), the hygiene item held behind the F-816-24 merge, and three closing deliverables; grants are the R287(b) mission pattern renewed by operator forwarding and NO new authority is issued — this mission adds SEQUENCING, not authority. (c) Dependency order is FIXED: F-816-24 dispatches first among engine-touching work with right-of-way, freeze-governance re-verifies the gh discharge at its own dispatch, CI-runtime runs its Task-0 staleness refresh before anything else, and the held hygiene item starts only after the F-816-24 merge lands on dev; parallelism is permitted exactly where collision tables RE-MEASURED at dispatch say it is.
(d) Closing deliverables: M3 — prereg row 19 flips to DISCHARGEABLE when the real-drive witness is merged and green; M4 — a fresh architect STATE snapshot superseding the 2026-08-16 lineage plus a refreshed handoff seed; M5 — CLEAN REPORT 2 ending with the who-owes-what list. (e) The mission ends when every packet's exit criteria are met with CI green, the held item is done and M3-M5 are delivered — or when the operator stops it.
Grounds: each packet's own hard limits, pipelines and exit criteria already govern inside it.
Status: standing [INLINE]

### R296 — Checkpoint 3 / Q1 exit ratified; F-816-25/-26/-27 ruled; STRUCTURE-NOT-TEXT adopted
Decision: (a) Checkpoint 3 and the Q1 exit are ratified with all six recorded decisions — the full-RunConfig load, the retained override flags with their re-argued justification, publish-don't-compare with F-816-26 filed, the declined census gate, the declined kill-grace ceiling (correctly refused as half of an open prereg row), and fix-forward over revert. (b) F-816-25 (the pretrain CLI shadows five minted train.* keys, three divergent at run5) is RULED an R79 duplicate-authority defect — the shadowing flags are removed or become config-only — and is a mission item SEQUENCED BEFORE any BC-pretrain execution. (c) F-816-26 is RULED: the child compares config_identity_sha256 against the parent-published value and a mismatch is a NAMED REFUSAL, never a warning. (d) F-816-27 rides prereg row 19 to the operator sitting; no agent authors either side of that inequality.
(e) Gate 14's interpreter is fixed by a REPO TOOLCHAIN PIN — the rust-toolchain.toml pattern applied to node — touching no host config, with the refusal arm still armed for hosts without mise. (f) STRUCTURE-NOT-TEXT is ADOPTED: verification mechanisms derive from structure (AST, types, reachability) and never from text matching, and every existing text-matching guard converts on contact. (g) The box sitting JOINS the mission as Q6 under the renewed R282(c) grant, run after the Q2-Q4 merges, raw tables and no verdict. (h) Token riders bind prose, never checks.
Grounds: a divergent shadow on the pretrain CLI would silently un-mint the bootstrap row's own values.
Status: standing [INLINE]

### R297 — checkpoint 6 ratified; NEGATIVE-SEARCH COROLLARY; RQ-16 dispositioned per field
Decision: (a) Checkpoint 6 is ratified including the disclosed grep mis-scope, with the census leaf's claims re-verified. (b) The NEGATIVE-SEARCH COROLLARY is ADOPTED as R296(f)'s completion: a negative text-search result is evidence about the PATTERN, never about the artifact, unless its scope and case posture are stated beside it — and negative-search sites join the M5 conversion census as a second axis. (c) RQ-16 dispositions bind the hygiene packet: node_coords is GENUINELY DEAD and deletable in one commit, while legal_mask, policy_dst_slot, window_center and current_player are test-only LAW-08 findings resolved PER FIELD — retire with test re-expression where the resolved contract covers it, wire a live consumer only where one is legitimately owed, one commit per field, NEVER a blanket act — with the perf measurement of the removals riding Q6 on the box.
(d) The R289 halt's premise-verification record is ANNOTATED, not rewritten: leg 1 is falsified by the resolved citation with both miss mechanisms stated, leg 2 untouched and sharpened. (e) Handoff plan of record: on CLEAN REPORT 2 plus the Q6 sitting record, the architect returns IN ONE TURN the re-bench verdict, the architecture-program starter prompt, the updated project instruction file and the memory update.
Grounds: a guard that matches text and a search that fails to match text are the same instrument read in opposite directions.
Status: standing [INLINE]

### R298 — DEFERRED RATIFICATION in force for the mission remainder
Decision: (a) Checkpoint 7 is ratified and both judgement calls become the per-field TEMPLATE: retired-key goldens are kept as evidence with the absence ASSERTED — a capture whose bytes are rewritten on code change has stopped being a capture, and a check that skips unmatched keys passes by not checking — and retired-set authority lives in ONE module. (b) DEFERRED RATIFICATION is in force for the rest of MISSION FINISH-LINE: checkpoint reports land in the governance repo and do NOT return to the operator, the mission continues without per-checkpoint architect turns, disclosed-and-reasoned judgement calls (disclosed same-act, mechanism stated, mutation-proved where a guard is involved) proceed under STANDING APPROVAL and are batch-ratified at mission end from the on-disk records, and the rulings queue stays monotone — file, skip, continue. Frozen edits without existing grants, armed values and prereg decisions remain ABSOLUTE exclusions: filed, never taken.
(c) EARLY RETURN occurs in exactly one case — every remaining queue item blocked; otherwise the next and final surfacing is CLEAN REPORT 2 together with the Q6 sitting record, answered in one turn. (d) One commit per remaining field; Q6 runs BOX_BLOCK.md verbatim, raw tables, no verdict.
Grounds: deferral is safe only because the records are re-derivable in full — every figure names its producing run, every guard is mutation-proved, every census is scripted.
Status: standing [INLINE]

### R299 — MISSION FINISH-LINE batch-ratified; the re-bench verdict NOT issued
Decision: (a) BATCH RATIFICATION per R298(b): every interim checkpoint of MISSION FINISH-LINE is ratified from the on-disk records including all disclosed judgement calls, the gate-3b caveat is accepted as stated (a claimed-green sibling set is not a completed integration tier), and the two REFUSALS — declining inadmissible-host bench numbers and declining to infer box identity from ssh config — are ratified as CORRECT and on the record. (b) F-816-29 is GRANTED diff-scoped to the row's one-keyword-argument diff, same-act re-pin as a term, per-event, never precedent. (c) F-816-28 is RULED BY PRINCIPLE: the resolution preserves BOTH invariants — a single exit-code-constant authority AND util's import-free property — and if no option satisfies both, the primitive does not move and the row records why.
(d) F-816-30 is RULED: a skip guard's discriminator must detect the MECHANISM its docstring names (the BLAS path in use), never a proxy, since platform.machine() is true on every runner and sees nothing; MECHANISM-NOT-PROXY is adopted as structure-not-text's sibling. (e) The RE-BENCH VERDICT is NOT ISSUED — no after side exists and Q6's structural block was correct — and remains owed against the UNCHANGED R283(c) band. (f) Handoff per R297(e), with the carry-over dispatch queue named and this session lineage closed.
Grounds: a process that ratifies only what was made teaches that not-making is invisible.
Status: standing — (e) SPENT by R300(b), which ISSUED the verdict; any text reading "the verdict is owed" is superseded from 2026-08-21 [INLINE]

### R300 — the re-bench verdict ISSUED (STOP/INVESTIGATE); Q6 adjudications; security disposition
Decision: (a) The ARCH-ERA verification census is ratified in full and the numbering HALT is resolved: the register head is R299 landed at f9988bd, and numbering continues. (b) THE RE-BENCH VERDICT IS ISSUED against the unchanged R283(c) band on the Q6 sitting record — P1 TRUE (occupancy 64 vs 20), P2 diagnostic fail, P3 86.3% diagnostic pass, P4 median 138.96 gph [137.20, 143.27] against before 138.43, BELOW the 166 line, so STOP/INVESTIGATE by the band's own mapping; the 0.911x regression is ERASED (parity, IQRs overlap), both P-MASK and P-CHECKS predictions are CONFIRMED (gather 36.34% to 0.03%, _check_structural 12.89% to 3.65%, collate 17.72 to 7.18 ms), neither pre-registered falsifier fired, and the prior investigation's targets are CLOSED. CORRECTED BY THE ANNOTATION AT R300's FOOT, never by repair: _check_structural's 3.65% is ABOVE its predicted 2-3 band, not "inside" it (CONFIRMED still stands, because the falsifier was "still near 12%"), and P2's median is 50.25 ms, not "~49 ms"; neither correction touches the verdict.
(c) The residual perf track is SCOPED and explicitly NOT mint-blocking — measure whether the 49.95%-idle serve thread is supply-limited before designing anything, candidate P-REPEAT for repeat_interleave (the third member of the size-from-contents sync class), C1-pinned staging live — and the operator's sims fork is OPEN NOW: PCR 600/75 unviable, the 50-sims class viable at parity throughput. (d) Q6 adjudications: Q3-D1 PASS closes PDEATHSIG on GPU, F-816-14's split is accepted (SIGKILL leg closed, SIGTERM leg re-worded as F-Q6-8), Q4d floors are clean within +-4% on the attested host, and F-Q6-1..8 route to the carry-over queue.
(e) SECURITY: the captured live token is COMPROMISED BY RULE because it reached a remote, the operator rotates it as the first act, forward redaction stands, and there is NO history rewrite (the repo is private, recent shas are load-bearing, rotation is the effective remedy); the migration workspace is the sanctioned home for box content, the pushed evidence stands, and a lightweight secret-scan convention is adopted. (f) Migration-repo remote operations use git-over-SSH ONLY, never the gh CLI. (g) The R299 session's two disclosed acts are ratified. (h) The carry-over queue and the operator residue (token rotation, prereg values, R227/R228 and R267 texts, the mint word) are recorded.
Grounds: a pre-registered band is read against the numbers and never adjusted to them — a mechanism working as predicted and the system not getting faster are compatible findings.
Status: annotated — the R300-foot annotation (two figure corrections) plus ANNOTATION 3, which ledgers them as the architect's; (e)'s rotation is SUPERSEDED IN MECHANISM by R301(c), the disposition itself unchanged [INLINE]

### R301 — R300 landing ratified; research copy ordered NOW; the workspace becomes LOCAL-COMMIT-ONLY
Decision: Ratifies the R300 landing and puts two figure errors on the architect's ledger, corrected by annotation only (per the ANNOTATION under R300's foot: `_check_structural` 3.65% is ABOVE its predicted 2-3% band, not inside it; P2's median is 50.25 ms, not ~49 ms). Orders the ~520 KB `tmp/research/` set copied at once into `mantis-migration/plan/research/` through the secret scan, provenance in the commit. OPERATOR RULING OF RECORD: the migration remote is DELETED and never pushed to again and the rented instance is DESTROYED AND RECREATED as the R300(e) token rotation, so the governance workspace is LOCAL-COMMIT-ONLY and every "governance push" line is satisfied by a local commit. Two conventions: process censuses redact token-bearing argv at capture time; a rented box carries NO repository credentials. Seam-recon findings accepted, Q-C1..Q-C4 routed, VAST-SETUP authorized.
Grounds: the leak was the instrument faithfully recording a command line, so the instrument learns redaction; and a remote that no longer answers cannot be the record's durable home.
Status: standing [INLINE]

### R302 — TORCH-SELECT ordered; HOST-COUPLED MINTS ruled as a class
Decision: Ratifies the R301-landing and VAST-REBUILD exits; the replacement 9900X host matches the attestation so bench floors are admissible again and the box is sitting-capable. Orders TORCH-SELECT: torch selection becomes host-correct at `uv sync` (GPU wheel where a compatible driver is detected, CPU default and fallback), the implicit-resync trap dies by construction, `restore_cuda.sh` retires, a producer test pins the default. Rules HOST-COUPLED MINTS as a class: memory caps (`train.microbatch_caps`, `inference.fused_graph_caps`, `_SIZING_BUDGET_GIB`) are host-attested exactly as bench floors are, so INSTANCE RECREATION VOIDS THEM and re-calibration becomes a permanent precondition; the burst's trainer-forward OOM is filed as the class's measured instance and CARD-RUN5-GPU-OOM's site wording is corrected. Files the close-out measurement beside F-Q6-8 as a LAW-16 question; adopts two conventions (box-resident instruction files are never read; process greps exclude the grepping process).
Grounds: a floor is void on a host change because the number describes the host, and a memory cap describes the host the same way.
Status: standing [INLINE]

### R303 — the RING QUESTION decided; PEN TRANSFER to the ARCH-ERA session
Decision: Ratifies the R302 landing and ADOPTS two instruments — `plan/ruling_census.py` and the version-stamp assertion. Ratifies the TORCH-SELECT option-kill: `[tool.uv] torch-backend` is schema-valid but inert for locked project workflows, established by cache-bypassed controls, so the conflicting-extras pattern proceeds to design. DECIDES the ring question: re-calibration runs on DISCLOSED SYNTHETIC graphs on the R283(d) precedent, with STEP 4 STRENGTHENED as the price — the validation burst must include TRAINING STEPS at the minted caps, not games alone, with peak memory read across both phases and both partition shares; generate-a-ring and defer are both DECLINED. PEN TRANSFER: the ARCH-ERA session holds FULL architect authority and the predecessor lineage closes with this ruling as its final act, the operator retaining a named residue.
Grounds: the measured OOM site is the GNN training forward, so a games-only falsifier could not fire on the failure it exists to catch.
Status: standing — (c)'s strengthened STEP 4 is DISCHARGED by R327(a) once the owed burst is paid [INLINE]

### R304 — Q-C1..Q-C4 ruled, closing the PLAN-C seam recon
Decision: (a) The capability surface lands AFTER the mint as its own thread; ArchCaps must REPLACE or DERIVE FROM the existing identity-key/schema/registry checks, never sit beside them. (b) `caps.exact_symmetries` does not land as one field: the board automorphism group is an exact RULES fact while model invariance is a MEASURED property of a trained net (FALSE for the GNN), so the field lands named for the rules fact and the orbit probe is ordered as precondition for any model-side symmetry claim. (c) §1.3 is NOT a site and CLOSES — the exhaustive match refuses a third representation at compile time, and replacing that with a runtime derivation is a downgrade. (d) `validate_against_state_dict` and `ShapeMismatchError` are DELETED on a zero-caller census; LAW-08 is explicitly NOT the grounds. (e) The ArchCaps DESIGN is authorized to draft now, sequenced to land after the mint.
Grounds: a second authority over the same relation is the duplicated-default class R1 exists to kill, and a validator whose green means "I could not see the thing I check" is worse than an absent check.
Status: standing — (b)'s field-naming order is CORRECTED by ANNOTATION 5 (the augmentation-facing field must declare whether THIS ARCH'S ENCODING commutes, and its name must say `encoding`), and the field is then DELETED outright by R307(b) with ANNOTATION 6 [AUTHORED]

### R305 — operator decisions recorded; residue reduced 7 to 5
Decision: Records four operator decisions of record: BACKUP is DECLINED and the flag RETIRES PERMANENTLY (the workspace stays single-copy by choice, and nothing re-raises it); ORIGIN removal on mantis-migration is executed and VERIFIED at this landing (`git remote -v` returns no remotes); ROUTING — all agent exit reports go to the ARCH-ERA architect session directly; TORCH — the GPU-detect / CPU-fallback directive is confirmed as the operator's own, already the R302(b) order. The operator residue reduces from seven items to five: forwarding the re-calibration block, the prereg values, the R227/R228 and R267 texts, the mint word, ssh-only next box.
Grounds: two residue items are discharged by measurement and by the operator's own word, so the list shrinks rather than being restated.
Status: standing [INLINE]

### R306 — mission PLANC-SEAM-M1 established under deferred ratification
Decision: Ratifies the R305 landing with the architect's stamp-model prediction error on the ledger. ADOPTS as standing practice: an instrument's MEASURED behaviour outranks any packet's prediction of it, and substring spot-checks strip markdown emphasis and blockquote markers BEFORE whitespace normalization. ESTABLISHES mission PLANC-SEAM-M1 (Leaf 0 the ArchCaps design review, then the R262 DESIGN/REVIEW/ORACLE/IMPL/REVIEW/RED-TEAM pipeline) under DEFERRED RATIFICATION in the R298(b) shape: disclosed judgement calls proceed under standing approval and are batch-ratified at mission end, while frozen edits without grants, armed values and prereg decisions stay ABSOLUTE exclusions. Sets binding constraints (nothing touches the mint path; R257 fences and R26 hold; box measurements are staged for operator forwarding) and the dispatch order of record, Q4 first.
Grounds: the flaky oracles and shrinking CI headroom are the substrate every other packet runs on, so they forward first.
Status: standing [INLINE]

### R307 — Q-C5: caps.exact_symmetries DELETED; the per-record gate is the sole symmetry authority
Decision: Ratifies checkpoint D1 / Leaf 0 as SOUND-WITH-REPAIRS. RULES Q-C5: `caps.exact_symmetries` is DELETED from the ArchCaps design, because the engine decides losslessness per record and per site on both arms, so a frozen per-arch set admits only two readings and both break (the union authorises augmentation the engine refuses; the intersection collapses to identity). The per-record gate IS the symmetry authority, singular. Orders ANNOTATION 6 under R304's foot. Sets two binding bounds on the F2-F9 repairs: no exit criterion replaces or re-keys the seven-name ban (an operator LOCK), and LAW-08 is STRUCK from the design's grounds wherever cited. Retires tier T2 for re-derivation and extends mission scope to PHASE 2 / WP-AXIS2, depth-to-done, with the hard exclusions restated.
Grounds: where the engine holds a per-record authority, no per-arch field may summarise it — a pointer duplicates as surely as a copy.
Status: standing [INLINE]

### R308 — the re-calibration sitting ratified as a SUCCESSFUL HALT; MINT-ON-BRANCH adopted
Decision: The mint was legal when made and is VOID now that STEP 4 falsified the partition on three clauses; caps stay void, F-R302-1 is EXPLAINED not closed (allocator reservation fragmentation under DEFAULT posture, 14.80 GiB reserved against 10.10 allocated), R61 stays unsatisfied. Two mechanism F-rows are ordered filed, chiefly that the eval-child budget term is ROUND-DEPENDENT, not constant. ADOPTS MINT-ON-BRANCH (STEP 3 lands on a branch; dev fast-forwards only after STEP 4 passes). RULES Q-C9 derive-or-delete inside the gate: the silent-encoding gate's `ENCODINGS` DERIVES from the encoding registry at point of use and asserts the derived set is non-empty and matches the registry count. Decides B(i) as forward-time symmetrization (key set untouched, so BC transfer survives); bars post-hoc arch stamping; sets the frozen-golden grant procedure; orders RECAL-PREP with the allocator posture becoming a minted regime knob.
Grounds: STEP 4's strengthening bought exactly what it was set to purchase, and a cap fitted on a destroyed container is a stale mint rather than a code defect.
Status: standing — (c)'s registry path is CORRECTED by ANNOTATION 7: the file is `crates/mantis-encoding/src/registry.toml`, not `crates/mantis-encoding/registry.toml`; nothing else in (c) moves [INLINE]

### R309 — Q-C10 behavioral-witness-or-no-landing; GnnNetV2 lands as a new arch
Decision: Ratifies the Phase-2 ORACLE and RECAL-PREP exits in full and ORDERS ANNOTATION 7 under R308's foot. RULES Q-C10: structural unfalsifiability CONVERTS the verification obligation and never waives it — a candidate no structural check can witness lands ONLY with a pre-registered BEHAVIORAL witness, and where neither is constructible the candidate DOES NOT LAND. DECIDES the detector dilemma: GnnNetV2 lands as a NEW arch and in-place modification of GnnNet is REJECTED, with deliberate detectors replacing the accidental ones (witness-stamp as a construction precondition; v2 mints its own golden; the fixtures manifest gains a reverse check; six bare-count gates convert to set equality). ISSUES the R308(f) grant against the presented diff. RECORDS the operator's n_workers prereg row (n_workers=1 REJECTED; bracket >1 through 14, point value by the knee rule). ORDERS WORKER-SWEEP as PHASE W of the re-sit.
Grounds: a change nothing can detect cannot be verified to exist; and caps fit at the config that will run, or they are stale at birth.
Status: standing — (e)'s push clause is COMPLETED by R310(c), and (g)'s "Phase W ahead of STEP 1" ordering is corrected by R310(e) to STEP 0 -> PHASE W -> STEP 1 [INLINE]

### R310 — F-R309-1 disposed; RED-BY-DESIGN refused as a posture; grant-on-a-measured-diff adopted
Decision: Grants the frozen edit on `tests/tools/test_preflight_mint_process.py` against the exact presented diff, with both arms asserting and the repair self-expiring at the sitting's mint; rejects the skipif extension (it would switch off R126's only witness) and rejects leaving the row red. THE RULE THAT GENERALIZES: a row whose subject is unavailable at HEAD carries a LIVE ARM at HEAD or a DECLARED skip — red-by-design is not a posture this tree carries, and an arm must assert something no other row asserts. Ratifies the Phase-W prep and RESOLVES the Phase-W ordering conflict against the architect's own text (STEP 0 posture A/B, then PHASE W, then STEP 1's four terms). ADOPTS grant-on-a-measured-diff: a presented diff is a diff that has been applied, driven, and planted-broken before the grant. Routes three pre-registration matters to the operator unmoved, with recommendations that are explicitly not pre-registrations.
Grounds: a red that is expected teaches every reader to expect red, so the next real red arrives to an audience already trained to look away.
Status: standing — (c)'s closing sentence ("Remote CI green on the pushed head is the confirming measurement") is WAIVED for that push by ANNOTATION 8 on the operator's direction, the local tiers recorded in its place; the sentence stands for every future push [INLINE]

### R311 — REMOTE CI SUSPENDED; field-ruling conditions; docs follow reality
Decision: Ratifies R310 and makes its shape the rule — an execution session MAY issue a field ruling when the item blocks progress, the evidence is complete and on disk, the ruling is disclosed as a field ruling in the act, and it is filed for architect ratification; operator locks are never field-ruled. REMOTE CI IS SUSPENDED by operator decision until the operator re-enables it: no push or merge waits on it and local green is the gate, with targeted tests while iterating and the full local gate set at leg exit and before any push. DOCS FOLLOW REALITY: a non-canonical working document that disagrees with verified state is repaired in place by whoever finds it, one line, no loop — register text still corrects only by annotation. SYMBOL, NOT LINE for packet references. Adopts plain comms, refreshes CLAUDE.md, grants the planc-conformance push, and records the operator's ADOPTION of the three worker-sitting preregs (F-WS-2, F-WS-7, F-WS-4).
Grounds: this changes WHEN gates run, never WHAT they check; the accepted cost is that gate 1's fresh-clone leg is the one check no local run reproduces.
Status: standing [INLINE]

### R312 — the governance mirror becomes a REDACTED DERIVATIVE
Decision: Ratifies the R311 landing including the mirror leak near-miss, and puts the rustfmt-gate claim and the stale re-sit prompt on the architect's ledger; supersessions of in-flight packets now ship as labeled amendments, never replacements. RULES that the public mirror is a REDACTED DERIVATIVE: `sync_governance.py` redacts through gate 17's own scan with stable placeholders before writing, `--check` verifies against the redacted transform, and the header names it derivative and points at canonical. Verbatim mirroring into a public repo is REFUSED; the operator may override only by making the repo private. Holds `--check` out of the gate set until Q4's gate protocol lands. Orders the conformance branch reset onto dev. Closes gate 17's worktree weakness LOUD: an absent local-terms file prints an OPERATOR-TERM ARM SKIPPED banner and the tool refuses to write any mirror without the full arm.
Grounds: a scanner that can be silently disarmed by a missing file is a mirror-writer that leaks quietly.
Status: standing [INLINE]

### R313 — a scan passing is not the output being clean; PIPE-EXIT and SHARED-TREE AMEND laws
Decision: Ratifies the R312 landing including the redaction-insufficiency repair — the gate-17-scan redaction passed while still carrying three account handles and a private repo name, and an independent grep, not the gate, caught it. Promotes that independent grep INTO `sync_governance.py`: after writing, it greps every supplement term against every written mirror and refuses on any hit, which with the refuse-without-arm rule makes mirror generation main-tree-only by construction. PIPE-EXIT LAW: no gate, scanner or verifier ever sits upstream of a pipe without `pipefail` or an explicit PIPESTATUS capture — an exit code never dies in a pipe; one-time sweep of gate wrappers, fix on contact. SHARED-TREE AMEND LAW: `--amend` only after re-reading HEAD in the same breath and verifying it is your own commit; in a shared tree prefer a new commit.
Grounds: the masked `secret_scan` rc 1 behind `| tail -1` is the pipe law's second producing instance, and the independent check is what authored the redaction catch.
Status: standing [INLINE]

### R314 — the shared-tree law generalizes to every commit; --check retargets to the dev ref
Decision: Ratifies the R313 execution, including the pipe-exit sweep's gate-14 catch (a type baseline that could announce itself unmeasured). GENERALIZES the shared-tree law, third instance in one family: verify BRANCH AND TIP before ANY commit in a shared worktree, not only before an amend; concurrent sessions commit from their own worktrees and the main checkout belongs to the session running the box event. Names the detached-worktree cherry-pick with byte-identical restoration of the foreign tip as the recovery pattern. RETARGETS `sync_governance.py --check`: mirrors compare against `git show dev:docs/governance/<name>`, never the working tree, because the mirror contract is about what dev carries; writing stays main-tree-only. One new control: parked on a non-dev branch, `--check` must still report dev's truth.
Grounds: the mirror contract is a statement about dev, so a working-tree comparison answers a question nobody asked.
Status: standing [INLINE]

### R315 — the re-sit ratified as a SUCCESSFUL HALT; the allocator posture SETTLED
Decision: The re-sit is ratified as a successful halt — blocked by the instrument, not the card. Every August FAIL clause now passes under `expandable_segments` and the REFUSED eval verdict is the tool working; nothing vests (caps stay void, F-R302-1 stays open as a posture artefact, R61 unsatisfied). SETTLES the allocator posture as `expandable_segments`, on the sitting's pre-declared mechanism criterion, to be minted with the caps at the next sitting. ORDERS RESIT-PREP-2 engine-side: a determinism control as a planted break; the eval-round timeout fixed at cause so a timeout exit FIRES LAW-15's gate rather than bypassing it; Δ8 rewritten against the measured nine-act mint; and conjunct 2 gaining a pre-registered MARGIN FLOOR ratified before any forwarding. The third sitting forwards only after that lands.
Grounds: the fragmentation divisor the partition arithmetic divides by is wrong 3.77x under DEFAULT and within 0.6% under expandable.
Status: standing [INLINE]

### R316 — the MARGIN FLOOR ratified at M = 0.35 GiB with its operative form pinned
Decision: Ratifies RESIT-PREP-2 in full. RATIFIES the margin floor at M = 0.35 GiB, derived from the single measured under-prediction (2026-08-22's joint peak exceeded its declared budget by 0.3352 GiB) rounded up, with the OPERATIVE FORM pinned: M is a SUBTRAHEND in the budget derivation BEFORE the fit — budget = (usable - trainer - eval_child - M) / frag — and conjunct 2 then asserts headroom >= M. Limits carried: n = 1, M is a forcing term and never a predictor, and correcting the 0.26 GiB double-count VOIDS M. Grants the frozen-file edit for the eval-broken reason rename, scoped to the one measured row. Accepts REFBOT-SCAN-1 and orders R257 ANNOTATED, never repaired. Adopts the operator's COMMENT STYLE direction into CLAUDE.md with the load-bearing-marker carve-out.
Grounds: a floor checked without the derivation subtracting it refuses every sitting mechanically, twice measured; and the armed value is the operator's, locked by his forwarding.
Status: standing — its (d) orders ANNOTATION 9, which QUALIFIES R257's "search-free" premise (it holds of Shrimp-Bot's training loop, not of the served artifact, which searches at deploy) [INLINE]

### R317 — the determinism control diagnosed at cause and RE-SPECIFIED on exact observables
Decision: Ratifies the RECAL-SITTING-3 HALT on every clause. Names THE DEFECT AT CAUSE: the determinism control certified through a PROXY — wall-clock `moves_per_min` conflates what the seed controls with what the machine controls — and its 1% band was a CROSS-REGIME CARRY from one quiet engine-side measurement (0.5821%, n=1) asked to certify a live-GPU drive whose own within-drive round noise is ~6%. RE-SPECIFIES the control, pre-registered before the measurement it will judge: (i) GATE = net-parameter hash equality across both control drives and every rung, no band, any inequality HALTs; (ii) DIAGNOSTIC, non-gating = per-drive move-sequence hash; (iii) REPORTED = the throughput spread with no band. Measures a NOISE FLOOR from four fresh same-seed drives at rung 4 and amends the knee rule strictly conservatively (the within set expands by 3-sigma; the expansion can only pull the pick toward fewer workers).
Grounds: R302(c)'s voiding logic extends across regimes, and the architect ratified past it at R316(a)(i) — that error is on the architect's ledger.
Status: standing — (d)'s stated assumption that rung-4 relative noise carries across rungs is MEASURED FALSE and corrected of record by R326(a); the widening's noise term is later corrected again by R338(b) [INLINE]

### R318 — the deploy head ADOPTS leaf batching as a train/deploy alignment
Decision: Ratifies the eval-stall investigation on every leg, the mechanism named at cause on three independent grounds. THE RULING: the deploy head adopts leaf batching — `select_leaves(k)` with k read from the SAME config knob self-play reads (`leaf_batch_size`, run5 value 8), so no new constant enters the tree. Grounds given in the ruling: the defect is the round-trip count and only batching reaches it; the net's targets are generated by k=8 leaf-batched search, so k=1 at deploy was an unexamined train/deploy mismatch; and the promotion protocol stays fair because both sides run one regime at fixed nodes. The per-move `release_cuda_cache` STAYS as the eval term's bounding mechanism. This is a change inside the R254/R258 locked surface and the operator's relay of the packet is his word on the lock for this change only. A falsifiable prediction rides as a tripwire, not a gate: round-trips fall ~8x and the first gate block completes well inside `eval.round_timeout_sec`.
Grounds: eval runs the deploy head, so aligning the shipped search with the regime the model is trained under keeps eval deploy-matched by construction.
Status: standing [INLINE]

### R319 — the CARD-LEVEL FACT banked; the eval-child term DECOUPLED from gate geometry
Decision: Ratifies the R318(d) tripwire HALT and accepts the retraction as exemplary — `games_total 0` was a hardcoded broken-path literal read as a measurement. BANKS the card-level fact: the pre-registered gate block (screen_games 80 x deploy_sims 150, ~1.5M leaf evaluations) costs on the order of an hour of this card's full self-play throughput, so `round_timeout_sec 3600` cannot hold that geometry on this card class under ANY implementation — the residual defect is the geometry/budget pair, not the code path. DECOUPLES the eval-child term from gate geometry on a pre-registered measured ground (the child plays games sequentially, so peak demand is per-game-structured), verified by a TWO-POINT INVARIANCE PROBE (a 4-game then an 8-game round, tolerance pre-stated) that HALTs on disagreement. Routes gate geometry vs budget to the operator as a prereg adjudication and holds RUN6 until it lands. Two field orders: the broken-round path stops reporting a valid-looking `games_total`, and `RoundSpec.progress_path` goes live minimally.
Grounds: a promotion gate that cannot complete inside its own budget is the defect class this era exists to end.
Status: standing — (d) is CLOSED BY MEASUREMENT by R340(a) as a weak-net artifact, and that closure is then WITHDRAWN by R341(b) as a cross-regime carry (standalone -> contended), so (d) is REOPENED and mint-blocking [INLINE]

### R320 — the gate-geometry adjudication WITHDRAWN; the eval outcome channel is a CONSTANT
Decision: Ratifies PERF-TRANCHE-1 with A3's refutation accepted as evidence and the 7.2% pre-control/ledger discrepancy left OPEN as instrument hygiene (ledger absolute levels are not quotable without re-measurement). WITHDRAWS the gate-geometry adjudication of 2026-08-29 as fatally defective, error at the architect's desk — authored without reading `plan/EVAL_POSTURE_OPTIONS.md`, whose §2 already states the outcome channel is a constant, and its arithmetic erred in the flattering direction. RATIFIES the harness check at mechanism: every game runs to the 128-ply cap, every config ships `ply_cap_adjudication` null, a capped game with no adjudicator scores draw and a draw scores 0.5, so WR is identically 0.500 with zero variance and the gate escalates unconditionally returning constant False. INVERTS the sequencing — ply-cap adjudication is the PRECONDITION of gate geometry — and adopts the §4.4 posture (longest_run_margin, seat-neutral, min_margin 1) FOR MEASUREMENT only.
Grounds: 0/13 is explained — it is not a slow gate, it is not a gate; and even on a healthy channel the shipped screen never screens (P(escalate | truly-50%) ~ 84.3%).
Status: standing [INLINE]

### R321 — the UNIVERSAL MODEL CONTRACT adopted as the design of record
Decision: ADOPTS `plan/SEAM_V1_DESIGN.md` and its governing principle (contract, not call graph; boundaries at batch level; hot paths compiled per-arch with zero runtime indirection; conformance proven offline). PLANC Phase-2 is re-scoped to it, GnnNetV2 becomes the first TENANT rather than the purpose, and the accept bar is testable: if a second arch still needs edits to the trainer, server, arena or config schema outside its own scope, the seam did not ship. CORRECTS two carried figures on the record — the "44% floor" is 53% of the post-tranche card (favourable), and "~80% idle card" understated its own measurement (~85-87% idle). ORDERS lane B0 (rebase the conformance suite, measured 33 ahead / 35 behind dev, onto dev and land it). Makes `_net_param_hash`'s promotion out of the diagnostics module a PRECONDITION on part 6. HOLDS the aux-head ban until the rule replaces the assertion. Elevates bootstrap BC-pretrain as both the early-strength fix and the earliest path to a calibratable gate.
Grounds: both shipped eval criteria are degenerate at this maturity, so the binding precondition is a checkpoint mature enough to produce a decidable position, not a better criterion.
Status: standing [INLINE]

### R322 — SEAM-B0/B1 ratified; R257's [r8] fence NARROWED by annotation; B2 scoped as three legs
Decision: Ratifies SEAM-B0's extras (the run-fatal PyRefMut race fix with its mutation self-test commended into practice) and SEAM-B1 in full, with W-C1 as the leg's finding: GNN-3's size-generalization hazard is measured at V1 dummy-aggregation norm 78.5x over a 64x node increase against V2's 2.49x. ORDERS R257 ANNOTATED, never repaired, narrowing its `[r8]` fence. Corrects five scout-flagged reference rows in place, one disclosed line each. ADOPTS B2 in three legs: LEG 1 kills T9's eight red config-partition rows by repair (ratchet rows deleted by the fix, a repair touching a MINTED row is a HALT) with candidate D landing an arch selector that leaves every production config byte-unchanged; LEG 2 migrates dense lineage by structurally derived reachability (archive with goldens and a grave note, or SURFACE with consumers named); LEG 3 lands two components UNARMED behind the contract — landing is not arming.
Grounds: strix stage S5 is win_length 6 / placement radius 6 / unbounded, which is this project's exact rule set, so S5-stage material transfers without a radius re-derivation.
Status: standing — its (b) orders ANNOTATION 10 [INLINE]

### R323 — SEAM-B2 ratified; the box lane ordered as three items
Decision: Ratifies SEAM-B2 in full — the eight-row repair with its ratchet deleted by the fix and its own introduced defect caught by an existing oracle; candidate D's mechanism with the incumbent pinned against every minted config; the archive verdicts (HeXONet and ValueHead archived with goldens and a proven fence; HexTacToeNet kept with consumers named); both codecs landed UNARMED with unarmedness asserted structurally; and the census's pre-registered branch firing. Commends the withdrawn W-L2 correction. ADOPTS candidate D's disposition: the arch-selector key enters production configs only as a MINTED ROW at run6's mint. ORDERS the box lane: repair the opponent pool (sitting 3 recorded sealbot, kraken and strix all UNRESOLVABLE at HEAD); implement and prove `eval.strength_floor` with a planted break, its arming VALUES proposed but RESERVED to the operator's sitting-4 forwarding; execute F-816-25's fix if field-rulable.
Grounds: a promotion gate is only as meaningful as its opponents, and sitting 4 does not forward against an empty ladder.
Status: standing [INLINE]

### R324 — SITTING4-PREP-1 ratified; sitting 3's sealbot cause CORRECTED on the record
Decision: Ratifies SITTING4-PREP-1: sealbot repaired at mechanism (BUILD_ABSENT), kraken and strix remaining R139 loud-skips, strength_floor landed with four biting planted breaks, and the F-816-25 field ruling ratified with its honest census of ELEVEN sites, not five, `eta_min`'s 50x divergence included. CORRECTS sitting 3's stated sealbot cause of record: the pin was always present, the BUILD was absent. Corrects the "~50 s" figure at its three carry sites in place — the measured basis is ~341 s at the single-eval-worker rate, producer named. Makes the sealbot repair REPRODUCIBLE: a pinned fetch-and-patch script with its sha lands in-tree so the box runs the same commands from the tree, not from memory. Holds the strength_floor values PROPOSED and NOT armed (probe_games 4, min_decisive_rate 0.25, min_winrate 0.0), their deciding ground ordered MEASURED this mission. Adopts NIGHTRUN as five severable legs.
Grounds: candidate-vs-random decisiveness had never been measured in any era, so the values could not be armed on argument.
Status: standing [INLINE]

### R325 — NIGHTRUN-1 ratified and the PUSH ordered; the strength_floor re-read as a gate-integrity guard
Decision: Ratifies NIGHTRUN-1 (E1 met at x1.905 with byte-exact parity) and ORDERS the held push — verify the sweep green at tip, push, verify `dev == origin/dev`. Records the eval-loop attribution (95.3% serial GIL-holding leaf build) and S-INCR-GRAPH's measured 141x margin as CONVERGING EVIDENCE, elevating INCR-GRAPH to the top of the floor lane. RECORDS the decisiveness finding — step-25 scored 0/20 vs random and a fresh-init net scored 0/20, identically — and RE-READS `strength_floor` as a GATE-INTEGRITY GUARD rather than a training-progress meter, informative exactly when a checkpoint can finish a game, with min_winrate 0.0 empirically vindicated. FILES the bootstrap adjudication as `plan/ADJUDICATION_BOOTSTRAP_POSTURE.md` with corpus-mix recorded UNREACHABLE on a graph run in three independent places, so `bot_batch_share 0.0` is structural fact. Completes BC capability UNARMED. The posture row stays OPERATOR-ONLY DECISION OWED.
Grounds: a filing that supersedes the architect's draft is the one canonical home, and execution of any pretrain still waits on the operator's posture word.
Status: standing [INLINE]

### R326 — THE CAP RULING: the trainer's sizing permission re-derived and PROPOSED at 8.40 GiB
Decision: Ratifies RECAL-SITTING-4 on a REPRODUCED card-level fact — every partition term within 17 KiB across 93 commits and a changed worker geometry, so the refusal is a property of the card and the pre-registered permission, not of any measurement. THE CAP RULING: the trainer's sizing permission was never derived; it is RE-DERIVED as measured need plus a stated allowance — peak 7.443 GiB reproduced to kilobytes across two sittings, plus 12.9%, PROPOSED at 8.40 GiB — with pre-flight independently re-deriving the closing boundary and a disagreement HALTing before ssh. GRANTS the `_MOVED_LEAVES` widening from run5-only to the seven R316(f) configs. GAP-4 FINAL: `load_pretrained_buffer` is DELETED with a grave line and an import-census oracle pinning the absence. EXCLUDES `sealbot_d6 x 32` from the default battery by ruling (30.9 s/move cannot finish inside its own timeout).
Grounds: the ceiling convention is unchanged — the term remains the permission; the permission shrinks to honesty. R317(d)'s carried-noise assumption is MEASURED FALSE and corrected of record: future ladders use per-rung noise.
Status: standing [INLINE]

### R327 — RECAL-SITTING-5 and THE MINT ratified; R303(c) DISCHARGED
Decision: Ratifies RECAL-SITTING-5 and the MINT: conjunct 2 closed at +0.362 against M under the honestly derived cap, and THE OWED BURST WAS PAID (joint peak 13.0596 against partition 14.8555, edge cap binding at its fitted value, zero OOMs) — so R303(c) is DISCHARGED. Issues the FROZEN-FILE GRANT for exactly the two rows of the presented mint diff and orders `sitting5-mint` merged to dev with a full local sweep and push. PINS `--margin` at 0.85 as procedure: moving to the measured affordability edge (0.8568) after observing it would be criterion movement, and the 0.86-refuses fact is RECORDED as the partition being tight and honest. Leaves the second budget authority UNMOVED and files consolidation as debt. AUTHORIZES BC-EXEC-1 under the §0.3 envelope with the ACCEPTANCE WITNESS pre-registered: post-BC vs random clears the armed strength_floor decisive rate and the longest-run distribution shows contested play.
Grounds: a failed witness re-adjudicates the recipe and returns here; the posture never silently degrades to (C).
Status: standing [INLINE]

### R328 — RADIUS 8 ADOPTED for the run6 lineage; ANNOTATION 11 ordered
Decision: Ratifies MINT-CLOSE Leg 1 and BOTH Leg-2 halts, records R279's audit gap (a replay that never builds a Board), and places the envelope defect — an early-stop term specified against a mechanism that exists nowhere — on the ARCHITECT'S ledger. ADOPTS RADIUS 8 for the run6 lineage by the operator's forwarding; R26 stands untouched as run5 history and R257/ANNOTATION-10 gains ANNOTATION 11 because the fence's radius clause INVERTS. Orders the registry change EXECUTED THROUGH THE SEAM as an identity-key change: both live arches re-proven by the full conformance suite at radius 8, floors re-benched, envelopes re-emitted, goldens re-captured with the r6 set archived, and the encoder re-run against the whole corpus. Orders the BC stopping mechanism BUILT and suite-proven before any pretrain consumes it, and `freeze_verify`'s 16 unexplained rows TRIAGED before any box leg.
Grounds: the corpus, the reference bots and the human ladder are radius-8, a strength claim is anchored to the ecosystem it plays in, and the 34.76% refusal with its wide-play bias is unacceptable in a bootstrap.
Status: standing — its (b) orders ANNOTATION 11, which dates the inversion to the registry (it takes effect when the registry actually carries radius 8 at HEAD, not on the day the annotation lands) [INLINE]

### R329 — RUN6-IDENTITY-1 ratified with the BC witness PASSED; the leg-order deviation corrected as provenance
Decision: Ratifies RUN6-IDENTITY-1 Legs 1-4: the two-row identity mechanism, the suite green at r8, the MAX_STONES amendment with `max_stones()` derived from the allocation, the full-corpus encode with truncation counted beside the game count, the stopping mechanism with its measured noise floor, and THE WITNESS PASSED — post-BC 20/20 decisive vs random with the floor cleared 4x. Records the `is_full_search` defect (policy loss silently masked on every corpus row) as caught by the held-out monitor on first print. ACCEPTS the leg-order deviation (BC before the mint) as MEASUREMENT and CORRECTS it as PROVENANCE: the checkpoint of record is re-produced on the minted world, after two witness caveats are repaired first — the control net seeds through the determinism seam, and longest-run reads stone colour from the engine and never ply parity. Fixes F-816-33 so gate 3c fails on any collection error.
Grounds: a checkpoint produced on a pre-mint world is not the checkpoint the minted run starts from, so the reading survives while the provenance is re-made.
Status: standing [INLINE]

### R330 — THE ALLOWANCE RULE pre-registered; pushes settled
Decision: Ratifies IDENTITY-CLOSEOUT and RE-READS the card-level fact as a fact about the PERMISSION, not the card — at zero allowance the r8 partition holds +0.424 GiB, the widest measured, and every allowance <= 4.89% closes it. PRE-REGISTERS THE ALLOWANCE RULE: allowance = max(2 x measured cross-sitting trainer-peak variance, 3%); four byte-identical sittings put the variance at ~0, so 3% is PROPOSED, must clear the boundary re-derived AFTER eval_child is measured at r8, and no allowance below 2% is ever proposed. Orders eval_child MEASURED at r8 by the R319(c) two-point probe. Gives PER-RUNG NOISE its mechanism (each rung's rel-SE from its own rounds; the within-set expansion uses the max rel-SE over the candidate set). Lands arch-selector plumbing engine-side before the mint — config-less call sites resolve the arch from the artifact's stamp, ONE authority. SETTLES pushes: R282(b) is annotated, pushes are governed by R311(b), dev pushes now.
Grounds: refusing to pick an allowance after reading the gate is commended; two omissions are the architect's, including a 12.9% r6 artifact re-applied without r8 grounds.
Status: standing — (d)'s per-rung noise term is later CORRECTED BY MECHANISM by R338(b) (the widening's noise term is the noise-floor drive's rel-SE, never a within-round spread); (f) orders ANNOTATION 12 [INLINE]

### R331 — FINISH-1 ratified; AUDIT-1 accepted and the P1-P11 order adopted
Decision: Ratifies FINISH-1: eval_child at r8 measures 1.857 GiB on its stated denominator, the 3% allowance now carries r8 grounds (+0.133 against M) and stands PROPOSED, arming only at the operator's mint forwarding. ACCEPTS AUDIT-1 as evidence — fifty-two findings across four risk tiers — and ADOPTS the P1-P11 order with three amendments: (i) F-19 is DECIDED BUILT, so run6's trainer initializes from a checkpoint named in the prereg by path and net hash, resolved through the artifact stamp by the one selector authority, with the step-0 net hash and the 20/20 BC reproduction as its witness; (ii) the brief's class-4 example is CORRECTED of record — `MAX_STONES` is a storage ceiling with one Rust owner exported through the bridge, not registry geometry, so the architect's example was wrong and the tree was right; (iii) findings from the three session-limited sweeps carry a lower-confidence mark. REPAIR-1 executes P1-P4.
Grounds: P1's F-01 and F-05 are the only rows producing wrong numbers on artifacts read today, and P2's F-09 makes "full local green" mean what R311(b) says.
Status: standing [INLINE]

### R332 — REPAIR-1 ratified; a freeze outlives its subject only by ruling
Decision: Ratifies REPAIR-1 — P1-P4 repaired with pinning tests in the same commits, the local runner making the gate set ONE command (nineteen gates, gate 1 opt-in and self-declaring), and the three self-introduced reds each caught by a gate and ledgered with their single cause named once. LIFTS the R118/A-1 FREEZE on `eval/rounds.py`: its subject merged and its certification is discharged, the freeze manifest and `freeze_verify` reconcile, and the pipeline-published identity with its agreement check becomes the pinning test. STANDING RULE: a freeze outlives its subject only by ruling. Rules F-11's arming MINT-CLASS — presented at pre-flight in one screen with two shapes costed, the architect ruling at exit, nothing arming in the leg. REPAIR-2 executes P5, then P8 with F-16 first and the F-19 warm-start entry BUILT, then P6 with F-08's EMA key landing as a schema twin, default off.
Grounds: a certification whose subject has merged is a standing constraint with nothing left to constrain, and only a ruling may end it.
Status: standing [INLINE]

### R333 — THE TIER LAW: a tier-scoped verification loop cannot see a tier-scoped call site
Decision: Ratifies REPAIR-2 — P5, P8 and P6 executed, the warm-start entry built with its hash-equality pin, F-16 closed so V2 survives an eval round, and the two audit repair lines that were wrong at contact (F-30's graph half, F-35's census root) CORRECTED of record rather than followed. THE LAW: a tier-scoped verification loop cannot see a tier-scoped call site, so a repair that makes a signature stricter must sweep call sites over the WHOLE TREE by structure before it is called done, and the SLOW TIER runs at every packet exit as an opt-in runner flag with its cost stated on every run, like gate 1. REPAIR-3 lands the runner change. F-11 and F-32 are ruled ON READING, the arm landing at the mint forwarding whichever shape is chosen. REPAIR-3 then executes P7, P9, the slow-tier runner change and DASH-1, the file-based run dashboard.
Grounds: the exit's red-then-green is the leg's finding — the stricter signature's call sites were invisible to the tier that was run.
Status: standing [INLINE]

### R334 — REPAIR-3 ratified; F-11 ARMED as SHAPE A; the RUN6 SUCCESS WITNESS pre-registered in shape
Decision: Ratifies REPAIR-3 (P7's one walker replacing seven hand copies; P9's two reverted deletions on ruling grounds; the slow tier at packet exit; DASH-1 with seven live panels and two banked) and executes the held merge and push so `dev == origin/dev` before any measurement. ARMS F-11 as SHAPE A: the arming predicate gains a producer-liveness operand with gate 12 untouched, plus `poll_once` reading the guard's own counters as a last-emit age for mid-run death; SHAPE B is REJECTED in its form because a monitor thread on a relaunch-class stall code is a crash loop into a filling volume. F-32 takes SHAPE A at the run6 mint with the pin DERIVED from `identity.warm_start` — one source, no hand-synced twin. F-18 takes option (i), the cross-crate pin against mantis-core's independent implementation. Pre-registers PERF-TRANCHE-2's six items with bars written BEFORE each A/B. Pre-registers the RUN6 SUCCESS WITNESS in shape: rounds complete from round one; a checkpoint beats sealbot_d5 at fixed nodes with a CI excluding 0.5; Elo against pinned external rungs slopes upward over the first third.
Grounds: N and the slope bar are operator prereg rows, so the witness is fixed in shape and left open in value.
Status: standing [INLINE]

### R335 — SCOUT-2 accepted; KLENT refused as a bundle; PERF-TRANCHE-3 ordered
Decision: Accepts SCOUT-2 as evidence and DISPOSES its six §6 contradictions by FOUR DIFFERENT INSTRUMENTS chosen by each document's status: an append-only annotation for the register row (F-43), one disclosed in-place line each for the two working docs, and an annotation for the input document. REFUSES KLENT as a bundle on COVERAGE, explicitly not stability: |A| is 3-15x outside the tested envelope, the cross-iteration buffer was never validated, epochs-per-buffer is unreported; lambda-returns is the first ablation candidate, reverse-KL second, and entropy normalization is a PRECONDITION on ever arming an entropy term rather than a fix. Fixes the served-sims overshoot AT CAUSE before the mint (53.46 served at 50 configured; a search stops at exactly N). Proposes `playout_cap` DISARMED for the run6 mint. Orders PERF-TRANCHE-3 in five severable legs and requires the RUN6 SUCCESS WITNESS's instrument PROVEN before step 1.
Grounds: the doc's status decides the instrument, not the finding's size; and a witness whose instrument has never fired is the F-27/F-30 class.
Status: standing [INLINE]

### R336 — THE SELECTOR ROW ruled; the mint act separated from the warm-start pin act
Decision: Ratifies PERF-TRANCHE-3 (served sims now equal `n_simulations` on the PUCT path; S-CHECK17 landed with its bar MISSED and BANKED at x1.18; S-PREFUSE REFUTED with HOT-14 re-opened; S-BATTERY-G landed unarmed) and ratifies the SITTING-7 pre-flight HALT AS A SUCCESS with all four blockers on the ARCHITECT'S ledger. THE SELECTOR ROW: `identity.encoding = gnn_axis_r8` and the arch kind is `GnnArchV2`, shaped as ONE validated config key on RunConfig consumed only by the one selector authority. THE MINT ACT is the caps' act (STEP 3 mints both caps, the allowance row, the identity rows and every armed prereg row); `identity.warm_start` is a SEPARATE PIN ACT — STEP 4a reproduces the checkpoint, 4b writes path + sha256 + net hash as its own commit, 4c runs the step-0 witness. Six prereg rows are RE-SHAPED to HEAD, two of them DROPPED (the NaN row and the collapse-abort row).
Grounds: a hash cannot be minted before the artifact exists, and an envelope armed against a missing mechanism is the same error twice.
Status: standing — (d)'s citation of BC-EXEC-1 for the five held-out terms is CORRECTED by ANNOTATION 13 (the terms are `BC_EXEC_2_PREREG.md`'s; BC-EXEC-1 records NO MECHANISM for two of them, and STEP 4a is TWO commands across two all-or-none flag groups, not one); (b)'s shape half is noted as an R330(e) restatement by ANNOTATION 14 [INLINE]

### R337 — THE FILL RULE of record; the BC-recipe mis-citation annotated, not patched
Decision: Ratifies SITTING-8's pre-flight HALT as a success and places the cause on the ARCHITECT'S ledger — the launcher's §0.4 read a PROPOSED value as an UNSIGNED one, so a table built for envelope arming turned fourteen filled cells into fourteen halting ones. THE FILL RULE OF RECORD: the operator's forwarding ARMS every proposed value in the filed table; only a row carrying NO proposed value halts; and rows that name no `run6.yaml` key (run length, slope bar, rungs, ablation queue) are RECORD rows written to `RUN6_MINT_PREREG.md` that never gate the mint. Orders the BC-recipe mis-citation ANNOTATED at the register foot rather than patched, execution reading BC-EXEC-2 while the clause's intent stands. REFINES Δ19: a non-partition, non-identity row whose key is absent at HEAD is RECORDED as "no key at HEAD" and the sitting proceeds; only a partition or identity row's absent key halts. SITTING-8 resumes at Δ12 under the vested grant, the box kept.
Grounds: a halt on every filled cell is a table asking for a signature it already has.
Status: standing — its (b) orders ANNOTATION 13 and notes ANNOTATION 14 [INLINE]

### R338 — SITTING-9's Phase-W halt ratified; the widening corrected by MECHANISM, no threshold edited
Decision: Ratifies SITTING-9's refusal to mint `n_workers = 2` and places the defect on the ARCHITECT'S ledger: R330(d)'s widening took the WITHIN-ROUND SPREAD as its noise term, but at ply cap 256 that spread is a MONOTONE DEPTH TREND (no game ends inside a round on the throwaway net), so the widened threshold fell below every rung and the selection rule stopped seeing its own data. CORRECTION BY MECHANISM, no threshold edited: the widening's noise term is the NOISE-FLOOR DRIVE's rel-SE, never a within-round spread; recompute from the sitting-9 data in hand, and if the corrected widening still puts the threshold below the weakest rung the widening is VOID for that ladder and the unwidened rule governs (expected pick: 16). Takes the trainer allowance over the MAX of the r8/V2 distribution (8.6381 GiB), not p95. BUDGETS the random floor at 20 games per round with `deploy_matched = False` disclosed beside every reading.
Grounds: a partition is a bound, and a percentile readable as a bound is the "absent is not zero" family — an OOM at the 5th-percentile step ends the run.
Status: standing [INLINE]

### R339 — SITTING-10 ratified; run6 MINTED; R319(d) adjudicated
Decision: Ratifies SITTING-10 and MINTS run6. The step-0 witness fired once on real games and its reading — the BC net INDISTINGUISHABLE from `sealbot_d5` at 32 games — is recorded as the BASELINE the success witness measures against, not as a strength claim. ADJUDICATES R319(d): the first real round (93 games, 50 min 22 s, unfinished) exceeds its timeout by measurement, but the gate GEOMETRY does not move; the lever is the landed, unarmed battery concurrency, so `eval.concurrency` becomes a schema key (default 1, serial path byte-exact) whose value is picked by RULE on the box — the smallest G in {4, 8} whose full round completes within 0.66 x the existing minimum timeout. Makes F-816-37 an INSTRUMENT, not a hunt: the eval-path collate check runs at 1-in-1 with DUMP-ON-FIRE. Two tooling laws: the governance mirror REFUSES to write when HEAD is not dev, and `git reset --hard` on a dirty tree is BANNED without a stash ref recorded first.
Grounds: the timeout is the stall watchdog's limit and lengthening it buys stalls, so concurrency is the lever and the timeout is not.
Status: standing [INLINE]

### R340 — R319(d) CLOSED by measurement; eval.concurrency PINNED at 1
Decision: Ratifies the R339 exit and CLOSES R319(d) BY MEASUREMENT on three readings: the round that could not finish was a WEAK-NET ARTIFACT — warm-started, every gate game ends in 16-17 median plies, the round fits its timeout at G=1 with 2.3x headroom, concurrency buys under 1% in the cap-length regime, and the strength_floor refusal caps that regime at ~310 s. PINS `eval.concurrency = 1` for run6, with the escalation PRE-REGISTERED (trigger is a measured round time, value is the measured G=4 row, arm is the operator's forward). DISCRIMINATES F-816-37's two live hypotheses rather than arguing them — HARDWARE (a host DRAM bit-lane fault, tested by memory stress plus EDAC/MCE logs) versus SOFTWARE (a masked write of 1<<23 reaching a float buffer, searched by text and by structure) — with all three verdicts pre-stated, including that HARDWARE POSITIVE CONDEMNS the box. The shakedown runs only on a cleared box, 1-in-1 on both collate paths.
Grounds: arming a value the round never needed is arming a default readable as a measurement.
Status: (a)'s closure of R319(d) is WITHDRAWN by R341(b) as a cross-regime carry (standalone -> contended), so R319(d) reopens; the rest stands [INLINE]

### R341 — the host CONDEMNED ON SIGNATURE; R340(a)'s closure WITHDRAWN
Decision: CONDEMNS the host on SIGNATURE rather than on a test that could not be run — three firings, five words, two code paths, every one float32 bit 23 at byte offset congruent to 4 mod 8, one bit lane of a 64-bit word, after a software arm that is negative and strong. Makes RELOCATION the decisive test with both branches pre-stated. WITHDRAWS R340(a)'s closure of R319(d) as a cross-regime carry on the architect's ledger: contended gate rounds are ~7x slower per game (4.0 -> 25.9 s) and hit the 3600 s timeout returning null, on which the slope witness cannot fire — mint-blocking and host-independent. Orders the contended G table (G in {1, 4, 8}) as shakedown-2's FIRST ACT with the mechanism expectation stated before the data and a no-code prereg fallback. MEASURES time-to-signal rather than estimating it. Proposes run length 12 h and gate cadence every 750 steps. Two LAWS: no box is torn down until the run record and driver logs are copied off and hashed; `pgrep -f` is BANNED in packets and scripts.
Grounds: a memory stress that finds nothing in 102 min on a no-ECC host with no EDAC does not clear an intermittent lane, and un-convicted is not the bar a host must clear to train on.
Status: (a)'s condemnation is STAYED, NOT PATCHED, by R342(a) — ANNOTATION under R341's foot: the host is SUSPECT, not condemned, the consequence withdrawn on a ground R341 did not weigh, and R341(a)'s reading of the signature evidence still stands. (e)'s armed 750-step cadence is SUPERSEDED, NOT WITHDRAWN, by R343(b) — second ANNOTATION under R341's foot: both channels move to 1000 because the measured rate rose to 1581 steps/h, so 1000 now yields 6 rounds in the first third. The rest stands [INLINE]

### R342 — R341(a) STAYED on the operator's override; five conditions under which the host trains
Decision: STAYS R341(a) on the operator's override and records that the condemnation OVERSHOT: at the measured firing rate (one in ~825 eval games, one in a shakedown, none in 2004 steps) and with 1-in-1 checks on both collate paths, a lane fault is an UPTIME cost, not a correctness breach — every firing is caught, dumped and halts. Records that discrimination-by-relocation was OVERSTATED as decisive, since only a FIRING elsewhere convicts software. Sets five CONDITIONS, each a check and not a memory: (i) 1-in-1 on both collate paths for the WHOLE run with dump-on-fire and halt-on-fire; (ii) every checkpoint sha256'd on write and verified on load, mismatch = halt; (iii) on a firing the run RESUMES from the last verified checkpoint, or the operator-restart cost is stated on the exit screen; (iv) a pre-registered RATE BAR — more than 3 firings in any 12 h, or any firing outside the wire arrays, CONDEMNS the host again with no further ruling; (v) firing count and location are dashboard lines. The 2026-09-06 rebuild voided the SITTING-10 caps, so the re-mint runs on this container.
Grounds: what is unchecked either fails loudly or perturbs one sample among millions, so the measured rate makes the fault an uptime cost.
Status: standing [INLINE]

### R343 — the EVAL INSTRUMENT SPLIT into two decoupled channels; RESUME-1 ordered before run6
Decision: Ratifies R341 and R342 as executed: the G table refuted R341(b) at G=8 ONLY, and 1-in-1 on all three paths returned ZERO firings, the first bound that means what R342(b) intends; `cuda_context` measured 18.7% larger on the same container, generalised into the law that NO PARTITION TERM IS EVER CARRIED, not even across a container on the same card. SPLITS EVAL into two decoupled channels: PROMOTION every 1000 steps for the whole run; EXTERNAL (sealbot_d5 at fixed nodes) every 1000 through the first third then every 2000, with a SATURATION RULE (pooled external WR >= 0.85 halves the cadence again and labels the reading SATURATED) and a DEGRADATION FLAG for the self-play-cycling signature, WARN-ONLY. Orders RUNG-2 as mid-run work. ORDERS RESUME-1 before run6, time-boxed one day, with the RING persisted on stop and reloaded on resume, never refilled from empty, and byte-identity witnesses pre-registered. Fixes the ANCHOR as the warm-start checkpoint of record; a self-seeded anchor is a pre-flight HALT. DEMOTES `axis_distribution_alert` to a dashboard metric.
Grounds: 32-game CIs cannot separate 0.85 from 0.95, so an instrument that has stopped discriminating is retired rather than believed; and an alert firing on every emission trains its reader to ignore alerts.
Status: standing — (f)'s 12 h block is SUPERSEDED, NOT WITHDRAWN, by R344 §0.5 (ANNOTATION under R343's foot): the block becomes a 25 001-step minimum, because at 1581 steps/h a 12 h block ends ~6 000 steps before two of its own armed aborts (`min_step: 25000`) could fire at all. (f)'s reasoning about extension-on-readings is untouched [INLINE]

### R344 — GAME-RECORD-1 ordered before the start; DASH-2 becomes a read-only server
Decision: Ratifies R343, holds the rate bar at R342(b)(iv)'s written 3-per-12 h because RESUME-1 delivered inside its box, and puts the anchor-pin collision between R343(c) and R343(d) on the ARCHITECT'S ledger with the fix (the pin follows launch mode). Orders the ring sampler SEEDED FROM `config.seed` — the cheaper alternative — and REFUSES capturing ChaCha state through rand's backend BY NAME as the coupling the crate's pin exists to prevent. ORDERS GAME-RECORD-1: every game saved from step 0 across self-play, promotion, external rung and random floor, one record per game with the move list in axial coordinates and per-position search stats on every eval-channel game and a 1-in-N sample of self-play, stored as append-only length-delimited msgpack shards per (run, hour) with fsync at close, no new hard dependency. ORDERS DASH-2: `mantis dash serve` as a read-only stdlib HTTP server over the run record, loopback by default, carrying the GAME VIEWER.
Grounds: a run that does not write its games cannot be viewed, replayed or mined, so the producer must exist at step 0.
Status: standing — (c)'s "1000 steps, both channels" is CORRECTED by R345, which holds `eval_interval` at 1000 and moves `gate.stride` to 3; §0.5's 25 001-step minimum supersedes R343(f)'s 12 h block [INLINE]

### R345 — AUDIT-2 accepted; run6 HELD for REPAIR-A2; sims and Gumbel separated
Decision: Accepts AUDIT-2 as evidence and rules FOUR of its findings RUN-BREAKING for a promoting, resumable run — a non-finite gradient reaching `optimizer.step`, an arena that scores moves it never checked against the legal set, periodic checkpoints that are not continuation points beside a ring truncated in place on write, and a gate CI that resamples games rather than opening pairs on openings that repeat every round — so run6 is HELD for REPAIR-A2, seven severable legs in a two-day box, each carrying a planted break and a mutation self-test. Re-rules the GATE CADENCE on arithmetic: `eval_interval` held at 1000 and `gate.stride` moved to 3, correcting R344(c)'s "1000 for both channels" with the split-that-was-already-a-key on the ARCHITECT'S ledger. SEPARATES sims from Gumbel: run6 runs PUCT at 50 as the minted, gate-armed control arm, 96 REFUSED on projection, STRENGTH-FRONTIER-1 measuring the question at block end on run6's own frozen checkpoints, and GUMBEL-REPAIR-1 landing to Mctx invariants during the block but enabled in no run. REFUSES KLENT's search-free Shrimp target BY NAME. Its (e) annotates LAW-10 as GRID-ERA — the thresholds were measured on the dense threat-logit head, which `GnnNet` does not have, so the law has no producer on run6's lineage.
Grounds: all seven legs verified LIVE at HEAD before execution, against AUDIT-1's comparable sweep where ~6 of 52 lines were already false at contact.
Status: standing — the register section carries the header summary, the provenance block and the premise verification only; the canonical quote-block lives in `PACKET_R345_AUDIT2_ADJUDICATION_REPAIR_A2.md` §1, which is not in this repo. One clause could not be executed as written: §0.1's filing of `plan/AUDIT_2026-09-09.md` is OWED, the analysis text never having been forwarded [SUMMARY-ONLY]

## Annotations inventory — the register's append-only corrections

Register text corrects ONLY by annotation, never by repair (R285(c) shape, R290(f) ledger).
Numbered annotations 1-14 are NOT stored in numeric order: 1-6 and 9-11 sit in the
`## REGISTER-FOOT ANNOTATIONS` block, 12 sits under R282, 7 under R308, 8 under R310, and
13-14 under R336. Five further annotations carry no number and sit under their ruling's foot.
Source: docs/governance/archive/rulings_register.md, plus laws.md for the LAW-10 annotation.

### ANNOTATION 1 — the (t)-for-(v) misreference in the R289 packet's framing prose
Corrects: R289 (framing prose only; no numbered clause is affected)
Fact: the packet's preamble cites "R289(t) below ends the double-carriage class"; the clause
that ends it is (v). (t) disposes of RQ-b2 and RQ-c and says nothing about carriage.
Ordered by: R290(f), promoting it from the R289 carriage note to the register foot

### ANNOTATION 2 — R289(a)'s phantom landing site
Corrects: R289(a)
Fact: (a) directs the freeze-practice amendment to land in "the conventions doc and the
ORACLE-WRITE leaf instructions". The conventions doc exists; NOTHING called the ORACLE-WRITE
leaf instructions existed anywhere, so half the order named an uneditable file.
Ordered by: R290(f); resolved forward by R290(d)(3), which orders the document CREATED

### ANNOTATION 3 — the two R300(b) figure corrections, ledgered as a POINTER
Corrects: nothing directly — it points at the ANNOTATION under R300's foot
Fact: R301(a) records the two R300(b) figure errors as the ARCHITECT'S, on the ledger. This row
is a pointer, never a second copy (R285 ONE-TEXT), so the ledger reader can find them.
Ordered by: R301(a), added by the R301 landing session

### ANNOTATION 4 — CARD-RUN5-GPU-OOM's site wording corrected from new measurement
Corrects: R114's diagnostic direction on CARD-RUN5-GPU-OOM (widened, not withdrawn)
Fact: the measured OOM site set is wider than "inference-side". The VAST-REBUILD burst OOM'd in
the GNN TRAINING FORWARD (`src/mantis/model/gine.py:71`, 4.77 GiB wanted / 3.29 free, peak
14798/16303 MiB) after self-play inference ran 95 s clean, so it is the trainer's term.
Ordered by: R302(c), which also re-types the class as a stale mint on a changed host

### ANNOTATION 5 — R304(b)'s split is TOO COARSE, and the defect is in its operative half
Corrects: R304(b)'s final two sentences on the field's naming
Fact: there are THREE facts under one name, not two — (i) the board automorphism group (a rules
constant, useless per-arch), (ii) whether THIS ARCH'S ENCODING commutes, (iii) whether the
trained NET is invariant (measured FALSE). Augmentation needs (i) AND (ii), not (iii), so the
augmentation-facing field declares (ii) and its NAME MUST SAY `encoding`. The holding survives.
Ordered by: the same architect session that wrote R304, 2026-08-22, on drafting against real code

### ANNOTATION 6 — fact (ii)'s TYPING is corrected: per-record and per-site, not static per-arch
Corrects: ANNOTATION 5's typing of fact (ii)
Fact: the engine decides losslessness PER RECORD AND PER SITE on both arms (hexg/sample.rs:172
12-or-1; sym.rs:134-140 12-or-4; game.rs:675-676 permanently 4; augment.py:214-229 per row), and
ANNOTATION 5's own citation is that gate. Consequence, ruled in R307(b): `caps.exact_symmetries`
is DELETED — the per-record gate is the symmetry authority, singular.
Ordered by: R307(c), appended by the MISSION PLANC-SEAM-M1 MAIN dispatcher, 2026-08-22

### ANNOTATION 7 — R308(c)'s registry PATH is corrected
Corrects: R308(c)
Fact: the file is `crates/mantis-encoding/src/registry.toml`, NOT
`crates/mantis-encoding/registry.toml` — the latter does not exist at HEAD. Every executor of (c)
derives from the former; nothing else in (c) moves. SECOND occurrence of the identical wrong
string, on the architect's ledger; CLAUDE.md's Map bullet, carrying no directory segment, invited it.
Ordered by: R309(b), appended 2026-08-22

### ANNOTATION 8 — R310(c)'s REMOTE-CI CONFIRMATION IS WAIVED BY THE OPERATOR
Corrects: R310(c)'s closing sentence, FOR THAT PUSH ONLY
Fact: the operator overruled watching remote CI. What replaces it is the LOCAL tiers run to
completion on the pushed tree (integration 39 passed / 10 skipped, default 3730 passed / 4
skipped, twelve gates rc 0), recorded where the run id would have been. Cost stated: the local
host is not the CI host and gate 1's fresh-clone `uv sync` is the one check no local run reproduces.
Ordered by: the operator's direction, recorded 2026-08-27 on the ANNOTATION 7 precedent

### ANNOTATION 9 — R257's "search-free" premise is QUALIFIED, not repaired
Corrects: R257's second fence (its PREMISE, not its holding)
Fact: the reference bot acts search-free in its TRAINING loop only; the DEPLOYED checkpoint
searches — Gumbel sequential halving at 16-128 sims per stone. The direction is favourable: an
artifact that searches at deploy CORROBORATES the R254/R258 deploy lock rather than challenging it.
Ordered by: R316(d), appended 2026-08-28

### ANNOTATION 10 — R257's [r8] FENCE IS NARROWED IN SCOPE, not lifted
Corrects: R257 fence (i)'s scope (R257 itself is NOT repaired)
Fact: strix's curriculum walks `win_length` and `placement_radius` on an unbounded board, and its
STAGE S5 is win_length 6 / placement_radius 6 / unbounded — this project's exact rule set. The
fence attaches to their PUBLISHED FIGURES (all r8 or synthetic) and to NON-S5 stages; S5-stage
material transfers without a radius re-derivation, labelled `[r6-MATCH]`. Fence (ii) untouched.
Ordered by: R322(b), appended 2026-08-30

### ANNOTATION 11 — R257's FENCE (i) RADIUS CLAUSE INVERTS
Corrects: the reading of R257 fence (i) from the run6 identity forward
Fact: what made the fence bite was a MISMATCH, not radius 8 as such. With run6 at radius 8,
r8-derived reference material is IN-REGIME and r6-only material now carries the divergence note —
including ANNOTATION 10's `[r6-MATCH]` class and every run5 quantity read forward. It does not
reach backwards (R26 and run5 stay radius 6) and licenses no unmeasured transfer.
Ordered by: R328(b); the inversion is DATED TO THE REGISTRY, effective only when the registry
actually carries radius 8 at HEAD (R328 Leg 1), not on the day the annotation landed

### ANNOTATION 12 — R282(b) RESERVES NO PUSH
Corrects: the reading of R282(b) (the clause text is untouched)
Fact: (b)'s preserved operator-only list reads "box grants, run5 mint authorization,
judgment-valued prereg rows"; PUSHES appear nowhere in it. The reservation lived in the ACTIVE
index (§2's R170 line and §6's R303(d) row), which was the index overclaiming. Pushes are governed
by R311(b) — full local gate set green. R170's MERGE half is untouched.
Ordered by: R330(f), appended 2026-09-02

### ANNOTATION 13 — R336(d) and launcher Δ18(4a) MIS-CITE BC-EXEC-1 for the five held-out terms
Corrects: R336(d)'s citation (its INTENT — a reproduction under the pinned recipe — stands)
Fact: BC-EXEC-1's recipe pins none of the five; its own rows record NO MECHANISM for held-out
split and early stop. The five are `BC_EXEC_2_PREREG.md`'s: split seed 328, heldout_frac 0.05,
eval_every 500, patience 3, min_delta measured at pre-flight. And "five flags" is itself off —
they reach a run through TWO tools and TWO all-or-none groups, so STEP 4a is TWO COMMANDS.
Ordered by: R337(b), appended 2026-09-04, on the ANNOTATION 7 / ANNOTATION 8 precedent

### ANNOTATION 14 — R336(b)'s SHAPE HALF restates a shape R330(e) had already ruled
Corrects: nothing — this is a LEDGER NOTE, not a correction; neither clause is wrong
Fact: R330(e) settled the arch-selector shape on 2026-09-02 and the engine BUILT it
(`stamped_arch_kind`, three call sites routing through it); R336(b) re-ruled it as though fresh
while citing R330(e), so no register reader is misled. One sentence IS new and stands: "a site
holding neither is a finding, not a third path". The cost: only the VALUE half was ever live.
Ordered by: R337(b), same act as ANNOTATION 13, 2026-09-04

### ANNOTATION under R284's foot — the citation correction and the double-carriage settlement
Corrects: R284(b)'s "hard rule 12" citation, and R284's carriage status
Fact: "hard rule 12" referenced the PREDECESSOR document; the current doctrine home is the
perf-doctrine bullet, `docs/design/repo_design.md` §10, first bullet. R284's canonical text is the
register's appended block; the chat variant is a SUPERSEDED DRAFT, not a second authority. Forward
rule from R285(d): a ruling has ONE text.
Ordered by: R285(c) and R285(d), appended 2026-08-18 by the R285 dispatcher

### ANNOTATION under R300's foot — two figures in R300(b) corrected against the sitting record
Corrects: R300(b)'s two figures (the verdict is untouched)
Fact: (1) `_check_structural` at 3.65% is ABOVE its predicted 2-3% band, not inside it — the
prediction's own falsifier was "still near 12%", so CONFIRMED stands. (2) P2's median is
`queue_wait.mean_ms` 50.2493 [47.9263, 50.8743], not "~49 ms"; R283(c) assigns P2 no gate either way.
Ordered by: R285(c)'s correct-by-annotation shape; filed by the R300 landing session, 2026-08-21

### ANNOTATION under R341's foot (1 of 2) — R341(a) IS STAYED, NOT PATCHED
Corrects: R341(a)'s CONSEQUENCE only; its text and its reading of the signature evidence stand
Fact: the operator overrode the condemnation on price/performance grounds, so THE HOST IS
SUSPECT, NOT CONDEMNED. The withdrawn consequence rests on a ground R341 did not weigh: at the
measured rate with 1-in-1 on both collate paths, every firing is caught, dumped and halts, making
a lane fault an UPTIME cost. R341 leg 0 is VOID for the R342 sitting; the condemned-host check
stays as built, disabled for that packet only and re-arming the moment R342(b)(iv) trips.
Ordered by: R342(a), 2026-09-07

### ANNOTATION under R341's foot (2 of 2) — R341(e)'s ARMED CADENCE IS SUPERSEDED, NOT WITHDRAWN
Corrects: R341(e)'s armed `train.eval_interval: 750` (and, by relation, `checkpoint_interval`)
Fact: R343(b) sets both eval channels at 1000, so the minted pair re-derives 750 -> 1000.
R341(e)'s arithmetic was correct on its own evidence (at ~1200 steps/h, 1000 gave only 4 rounds
in a 4 h first third, failing witness (iii)'s >= 5). What changed is the RATE: R342's shakedown
measured 1581 steps/h, giving 6 rounds at 1000. `checkpoint_interval` RE-DERIVES rather than being
re-authored, which is why R242 required that row be stated as a relation.
Ordered by: R343(b), 2026-09-08

### ANNOTATION under R343's foot — R343(f)'s 12 h BLOCK IS SUPERSEDED, NOT WITHDRAWN
Corrects: R343(f)'s block term only; its extension-on-readings reasoning is untouched
Fact: the block becomes a 25 001-STEP MINIMUM, extendable by resume. Ground is arithmetic:
`run6.yaml` mints `train.draw_rate_abort.min_step: 25000` and `monitor.wr_collapse_min_step:
25000`, and at 1581 steps/h a 12 h block is ~19 000 steps — it would end ~6 000 steps BEFORE two
of its own armed aborts could fire, which is LAW-07 in the time dimension. The first-third screen
moves with it, 4 h -> ~8 334 steps, carrying ~8 external points against the >= 5 bar.
Ordered by: R344 §0.5, 2026-09-08

### ANNOTATION under LAW-10 (in laws.md) — GRID-ERA
Corrects: LAW-10's applicability, not its criterion structure
Fact: every threshold in LAW-10 was measured on the DENSE threat-logit head. `GnnNet` ships
policy + dist65 value only and has NO threat head, so `train.threat_weight` is one of the
`GRAPH_FORBIDDEN_NONZERO_WEIGHTS` a graph config must zero. On run6's lineage the law has NO
PRODUCER and therefore gates nothing; the numbers are grid history and a graph-era threat probe
would re-anchor all four before the law binds again.
Ordered by: R345(e)

### Note — annotations ordered by rulings in range that live OUTSIDE the register
Not inventoried above because they are not register text: the append-only ANNOTATION under
`F-R-P2B-1`'s foot in `plan/ADJUDICATION_QUEUE.md` (cited by R303(a)); the F-43 ANNOTATION and the
`klent_assessment` §5.1 / Amendment 2 annotation ordered by R335(a); and the ANNOTATION discharging
the WPMAIN "rc 0 does not certify eval health" caveat, conditionally ordered by R318(f).
