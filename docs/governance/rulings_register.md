REDACTED DERIVATIVE — 7 fragment(s) replaced by stable placeholders under rule 7. Canonical: mantis-migration/plan/rulings_register.md @ c9af200, regenerated 2026-08-30.
NOT the authoritative text; never edit here; edits land in mantis-migration.
<!-- END MIRROR HEADER -->

# RULINGS REGISTER

> **AUTHORITY (updated 2026-07-26, WPUF-2 Phase M).** The operator's
> `plan/STATE_2026-07-24_ADDENDUM_A.md` **is now on disk and is AUTHORITATIVE for
> R23–R31.** The R23–R31 sections below are the WPSC dispatcher's glosses, retained for
> their verified coordinate corrections only; on ANY conflict of substance the addendum
> wins (its own header: "Authority: tier 3 alongside STATE_2026-07-24.md; on conflict,
> this addendum wins"). The addendum additionally carries **R24, R29, R32** — rulings that
> have no section in this file at all — and binds the §0 numbering fix (v2-internal Track A
> ids ≠ STATE §4 ids; klent §7 "A3" = STATE §4 row **A5**). Read the addendum first for
> anything in the R23–R34 range. R34–R41 below remain dispatcher-recorded from the WPSC
> sources; R42–R54 are appended at the foot of this file with per-ruling fidelity labels.
>
> **The register is current through R88** (updated 2026-07-28, WPAX Phase D). R55–R58 were
> appended by the WPUF-2 dispatcher from the operator's adjudication response; **R59–R63** by
> the WPAX dispatcher from `wp/WPAX/WPAX_dispatch.md`, all five **[INLINE]**; **R64–R69** by
> the WPAX dispatcher from the operator's mid-run adjudication response, all six **[INLINE]**.
> R64–R69 ratify Phase S as landed, close **ADJ-08** (R65) and **ADJ-09** (R66), and add two
> standing process laws (R68 scratch/disk hygiene, R69 named-mechanism claims). **R64–R69** were
> appended mid-run; **R70–R72** after Phase P landed, ratifying the commit-on-FAIL under a six-part
> standing test (R70), authorizing ADJ-13 with the class-fix law (R71), and making per-conjunct
> flip-set coverage standing law (R72).

Created 2026-07-25 by the WPSC dispatcher, per WPSC_RESUME.md "FIRST ACTION" (closes
HANDOFF-2 provisionally). Sources: `wp/WPSC/WPSC_dispatch.md` §R (verbatim) and
`wp/WPSC/WPSC_RESUME.md` (verbatim). **R33 is omitted — superseded in full by R37.**

---

## R23 — gumbel fields become schema fields

> R23: gumbel_m / gumbel_explore_moves / gumbel_mcts become first-class
> SelfplayConfig schema fields (they are currently code-side defaults 16/10/false
> reachable only via the legacy unvalidated dict path — hparams.py:198-200,318-320;
> schema.py:66-75 has no gumbel field).

*Coordinate note (R39): `hparams.py:198-200,318-320` verified accurate at HEAD;
`schema.py:66-75` is stale — `SelfplayConfig` is at `src/mantis/config/schema.py:228-236`.*

## R25 — A9 closure (radius schedule → constant; delete override chain)

> R25: A9 closure. Commit A: `legal_move_radius_schedule` → scalar
> `legal_move_radius: int | None` (null = registry value); RadiusStage
> (schema.py:52-56) deleted; resolve_radius_from_schedule deleted;
> require_offline_radius collapses to constant-vs-override precedence;
> emit.py:66-69,78 schedule block dies; configs/smoke_radius_curriculum.yaml
> RE-MINTED constant-radius on v6w25 (it is v6w25's ONLY live-config edge — do
> not retire it); all other configs re-minted mechanically (currently null stays
> null); test fallout per recon T8 §4 (test_regime_parity, test_resolve_radius,
> test_schema*, test_resolved_config_emit, test_every_key_has_consumer, plus
> key-rename fallout in checkpoints.py:654 and listed tests). Commit B: DELETE
> the entire radius_override runtime chain — pool_hooks.py:242-248,
> pool.py:274-276, bridge setter/sentinel (mantis-bridge runner.rs:141,179,220,
> 570-574), selfplay atomic + per-game branch (runner config.rs:81,146,
> mod.rs:119,234,246,308-309,338, game.rs:585-608), core radius_override.rs
> test. Grounds: zero production callers (recon-verified); dead weight is
> deleted, not documented. WP8-F4 (schedule step-ordering validation) is
> CLOSED-OBSOLETED by Commit A — do not implement it.

*Amended by WPSC_RESUME.md "SC-A4 / SC-B1 (R25 latitude, pre-authorized)" — DESIGN may
choose end shape (i) scalar field with a real wired consumer, or (ii) no config field at
all with the registry as sole authority. Coordinate corrections: RadiusStage is at
`schema.py:64-68`; the checkpoints reference is at `src/mantis/train/checkpoints.py:677`;
bridge setter is `runner.rs:572-574`.*

## R26 — run5 radius stays registry-derived

> R26 (context only): run5 radius stays registry-derived (gnn_axis_v1 = 6).
> Never write the number 8 into any config or doc in this run.

## R27 — falsified.md F-04 scope limit

> R27: falsified.md F-04 entry amended (docs(design) commit) to read, in
> substance: "F-04 falsified PMA-as-tested vs min/max. It does NOT certify min
> vs mean+max→attention (run3_findings_v2 §4.1). The min/max asymmetry — value
> aggregation takes the worst cluster view while policy aggregation takes the
> best-scoring view (search_drive.rs:311) — is a flagged defect preserved
> pending the matched-FLOP dense arm; see CARD-MINPIN pinning test."

*Coordinate note: the file is `crates/mantis-selfplay/src/runner/search_drive.rs`; line
311 verified byte-accurate.*

## R28 — dense-by-default fallbacks become hard raises

> R28: dense-by-default fallbacks become hard raises: the "v6" default arms in
> src/mantis/encoding/resolvers.py:43-54 (and the 373-395 legacy default) and
> the crates/mantis-bridge/src/board.rs:279-283 `to_tensor()` fallback. A
> missing/unspecified encoding RAISES a named error; it never silently becomes
> v6 (LAW-05 + LAW-11).

*All three coordinates verified accurate at HEAD.*

## R30 — seed determinism (a) and single amp authority (b)

> R30: (a) `seed` (schema.py:82) gets wired to real determinism — torch,
> numpy, and stdlib random seeded once at orchestrator boot from cfg.seed, with
> a behavior-named test proving two boots with equal seed produce identical
> first-batch tensors on CPU; (b) amp dtype collapses to ONE authority: the
> runtime pin in src/mantis/model/amp.py:20-38 (hardcoded graph→bf16 + raw
> non-schema dict key) and the schema-side resolve_amp_dtype (called only from
> emit) become one resolver on the schema seam; graph path = bf16 REMAINS LAW
> (rule 10; fp16 banned on graph) and is enforced at that single authority with
> a test.

*Coordinate note: `seed` is `RunConfig.seed` at `src/mantis/config/schema.py:244`, not
`:82`. `amp.py:20-38` verified accurate. "called only from emit" verified — sole
production caller `src/mantis/config/emit.py:79`.*

## R31 — host access (gloss)

Agent host access requires an explicit per-dispatch grant; provider aliases never cross
into hexo-mantis. *(No grant was given for the WPSC run.)*

## R34 — value/policy target typed knobs

> R34: value-target and policy-target become TYPED knobs with resolvers,
> minted at CURRENT semantics only (whatever Phase 0 finds those to be —
> expected: outcome-z value target; visit-derived or Gumbel-improved policy
> target). Single-variant enums are acceptable and intended: the choice is
> explicit per run, never a default (v2 §10 non-settlement law). Do NOT
> implement λ-returns or Grill LEARN in this run.

*Recon resolved the variants: value = `pure_outcome_z` (T-D), policy =
`raw_visit_distribution` (T-B).*

## R35 — LICENSE

> R35: LICENSE = MIT.

*DISCHARGED — commit `5bd1d70` on `wpsc-scratch`.*

## R36 — scratch-branch commit exception (= WPSC_dispatch Global Rule 1)

> COMMITS (R36 exception, this run only): agents commit ONLY to a new scratch
> branch `wpsc-scratch` cut from dev HEAD. dev and main are NEVER touched.
> One chunk = one commit, final-form conventional message (single-line subject,
> no trailers, no Co-Authored-By). docs/contract amendments = separate
> docs(design) commits. Operator cherry-picks onto dev after review; scratch is
> then deleted. Every commit boundary must be gate-green (rule 5 below) BEFORE
> the commit is made.

## R37 — entropy knob (resolves ADJ-01; supersedes R33 in full)

> 1. SC-A1 types the entropy knob as `entropy_reg_weight: float`, constrained
>    NON-NEGATIVE at schema (negative value = named ValueError). NO mode enum —
>    the `max_entropy_fraction` capability is descoped from this run entirely.
> 2. Every minted config carries `entropy_reg_weight: 0.0` explicitly — the
>    current measured behavior (core.py:120), true zero-behavior.
> 3. Field documentation states the sign law: positive coefficient on a
>    subtracted entropy BONUS (core.py:362-365, losses.py:285-286) — i.e. larger
>    value = more exploration pressure. The historical "−0.005" was a sign-leak
>    from the loss formula and never existed in this tree.
> 4. `GRAPH_FORBIDDEN_NONZERO_WEIGHTS` (core.py:69-72,459-464) is UNTOUCHED and
>    remains the sole authority for the graph-path ban. No duplicate load-time
>    check in this run.
> 5. The code-side default at core.py:120 dies with SC-A1's typing (rule-1
>    closure): the value comes from the validated config, nowhere else.
> 6. New Track B card CARD-A10-CAP (recorded, NOT executed here): decide whether
>    an entropy term enters the graph loop at all (guard lift is a deliberate
>    design act), and if yes, land |A|-plumbing at the loss site + the
>    log|A|-normalized form + the |A|=50-vs-400 pressure test. KLENT Amendment 2
>    attaches to that card.

## R38 — temperature_schedule disposition (resolves ADJ-02; amends SC-B7)

> The temperature_schedule debt row is VOID-AT-HEAD. Disposition: read the two
> test functions (tests/selfplay/test_pool_hparams_arms.py:196,
> tests/selfplay/test_pool_hparams.py:145). If they assert real behavior of the
> actual mechanism (playout_cap.temperature_threshold_compound_moves + temp_min)
> under a misleading name → RENAME to behavior-named form. If they reference
> nothing real → DELETE, with a log row. No new schema entity is invented.

## R39 — recon reuse + Phase 1 ratification (standing rules)

Recorded from WPSC_RESUME.md "RESUME STATE", which defines R39 inline:

> - Phase 0 recon is COMPLETE and REUSABLE. Do not re-run T-A..T-F. Stage prompts
>   carry the CORRECTED coordinates from TARGET_RECON_REPORT.md §T-E and the
>   verbatim debt rows from DEBT_DOSSIER.md — never the stale §R landmarks
>   (standing rule, R39).
> - Phase 1 is COMMITTED and RATIFIED: `wpsc-scratch` = `5bd1d70`, parent
>   `f1ad10f`. Deviation 1 (running Phase 1 after ADJ-01 was known) and
>   deviation 3 (compressed pipeline for a zero-behavior metadata commit) are
>   RATIFIED (R39). Rider on 3: compression is permitted only for zero-behavior
>   metadata commits and must always be disclosed.

## R40 — SC-B4 validator identifiers (replaces its "A2/A8" citations)

> - V-NOOP: every schema key must be consumed by exactly one resolver/consumer;
>   extends the existing O15 bijection test to ALL fields added in this run.
>   A key no resolver reads fails validation of the test suite, not silently.
> - V-PCR: playout-cap-randomization validator — rejects full_sims == quick_sims
>   (no-op "randomization") and degenerate full_fraction (≤0 or ≥1 when both
>   presets are set). Named errors.
> The identifier "A2" is not used for validators anywhere in stage prompts.

## R41 — optional-deps extras (ratifies HANDOFF-4)

> Optional-deps extras group declares matplotlib, rich, scipy ONLY (recon-cited
> import sites). structlog is imported nowhere and is NOT declared.

---

# R42–R54 — appended by the WPUF-2 dispatcher, 2026-07-26 (Phase M)

**Provenance discipline for this block.** `WPUF2_dispatch.md` instructs the dispatcher
to append "the texts of R42–R54 as inlined in this file and the WPUF chat record". The
WPUF chat record is NOT on disk in this workspace. Each ruling below is therefore
recorded at its **highest available fidelity**, and the fidelity is labelled:

- **[INLINE]** — reconstructed from operative text inlined in `wp/WPUF/WPUF_dispatch.md`
  or `wp/WPUF/WPUF2_dispatch.md`, with the citation given. This is binding text.
- **[SUMMARY-ONLY]** — the only text available on disk is the parenthetical gloss in
  `WPUF2_dispatch.md`'s REGISTER APPEND paragraph. Recorded verbatim as that gloss.
  **The operator should supply the full text on next review**; a dispatcher must not
  expand a SUMMARY-ONLY ruling beyond its gloss when resolving a question against it.

No ruling below was invented, widened, or narrowed by the dispatcher.

## R42 — gate-evidence timestamps precede commits [INLINE]

> Gate-evidence timestamps MUST precede the commit timestamp and both are recorded in
> COMMIT_MANIFEST.md (R42 — mechanically checkable; a commit made before its gate output
> was read is a breach).

*Source: `WPUF_dispatch.md` Global Rule 1. `WPUF2_dispatch.md` Phase M.4 extends R42
discipline to the merge itself: "R42 discipline applies to the merge exactly as to a
commit."*

## R43 — frozen-oracle edits always queue, regardless of direction [SUMMARY-ONLY]

> R43 (frozen-oracle edits always queue, regardless of direction).

*Source: `WPUF2_dispatch.md` REGISTER APPEND. Operative reading, consistent with
`WPUF_dispatch.md` Global Rule 3 ("oracle-contradicts-design is ADJUDICATED, never
satisfied"): once an oracle suite is byte-frozen, ANY edit to it — whether it would
tighten or loosen the assertion — goes to ADJUDICATION_QUEUE.md rather than being made.
Full text owed by the operator.*

## R44 — CARD-FRESHSYNC; main-gate CI non-binding until fixed [INLINE]

> Gate 1 fresh-clone sync has been broken since WP7 (`_engine.hello()` failure, per WPSC
> HANDOFF-9). Fix it, and answer IN THE LOG the question that matters more than the fix:
> why did nothing surface this for the whole migration — was the gate not being run, or
> run and its result ignored? That answer decides whether the repair is the script or the
> CI wiring, and BOTH get fixed if both are implicated. Until this commit lands, main-gate
> CI evidence is treated as non-binding (R44); after it, run the fresh-clone gate once
> end-to-end and record the evidence. Latitude: the why-answer is a log row, not an
> adjudication, whatever it turns out to be.

*Source: `WPUF_dispatch.md` PHASE R1 (verbatim).*

## R45 — silent-v6 card + pattern gate [INLINE]

> `src/mantis/train/pretrain/validate.py` carries the fifth known silent-v6 fallback arm
> (WPSC HANDOFF-10). Convert it to the same named-error pattern SC-B2 used
> (MissingEncodingError / PanicException on the FFI side — match the established
> convention, do not invent a new one). SAME COMMIT: add a repo-wide grep gate to the CI
> gate family that fails on the silent-fallback pattern class, so a sixth arm cannot exist
> quietly. The gate's pattern set is derived from the five known instances (enumerate them
> in the gate's header comment); false-positive escapes require a justification comment at
> the site, same mechanism as the 300-line soft cap.

*Source: `WPUF_dispatch.md` PHASE R2 (verbatim). Amended by `WPUF2_dispatch.md` PHASES
R1–R3: the CENSUS-first survival question is answered against the POST-MERGE dev HEAD; if
the site is already closed, R2 reduces to the pattern-gate half only, and the gate header
enumerates ALL closed arms "however many there turn out to be".*

## R46 — flaky-test law: deflake or loud-quarantine [INLINE]

> Deflake if root-causable within one revision loop; otherwise quarantine LOUDLY:
> skip-with-named-reason + a debt-ledger row appended to DISPATCH_LOG handoffs + the floor
> adjusted to the measured post-quarantine count with the quarantine named in the commit
> message. Never a silent skip — a flaky test inside the floor count is a floor that lies.

*Source: `WPUF_dispatch.md` PHASE R3 (verbatim). Subject: WPSC HANDOFF-8,
`tests/train/test_heartbeat_watchdog.py::test_gil_starvation_freezes_seq_and_the_supervisor_declares_it_stale`.*

## R47 — scratch-branch standing pattern [INLINE]

> COMMITS (R47): all commits go to scratch branch `wpuf-scratch`, cut from dev HEAD at
> entry. dev/main never touched. One chunk = one commit, conventional single-line subject,
> no trailers. docs/contract amendments = separate docs(design) commits.

*Source: `WPUF_dispatch.md` Global Rule 1. R47 generalises WPSC's R36 from a one-run
exception into the STANDING pattern for every subsequent WP. `WPUF2_dispatch.md` GLOBAL
RULES scopes it for this run: scratch branch for everything after Phase M; "Phase M is the
only action taken directly on dev, under R53's conditions."*

## R48 — WP-UNFREEZE unblocked from v2; sources enumerated [INLINE]

> run3_findings_v2.md remains ABSENT and is NOT required (ruling R48). Any DESIGN question
> that genuinely needs its §-level text → ADJUDICATION_QUEUE.md, never guess.

> On-disk sources this run consumes (read before Phase U):
>   plan/STATE_2026-07-24.md §4 row A1, §5, §6   (the A1 spec + run5 arming)
>   plan/rulings_register.md                      (R23–R50; this file inlines R47–R50)
>   wp/WPRECON/RECON_REPORT.md §T5                (actor-sync seam census)
>   wp/WP11A/DISPATCH_LOG.md                      (handoffs; marked gate-decision call site)
>   wp/WPSC/TARGET_RECON_REPORT.md, DEBT_DOSSIER.md (corrected landmarks; debt rows)

*Source: `WPUF_dispatch.md` header (verbatim, both blocks). **Status note (2026-07-26):**
`inputs/run3_findings_v2.md` has since ARRIVED on disk. R48's premise (v2 absent) no longer
holds, but its ruling — WP-UNFREEZE does not depend on v2 — stands unchanged, and Phase U's
source list is unaltered. v2 enters this run only through Phase A's A-0 VERIFY-READ under
R54, never through Phase U.*

## R49 — continuous sync is law; old mode unrepresentable [INLINE]

> Remove run3's gating deadlock from the tree. At HEAD (by deliberate zero-behavior port,
> WP-SP): the self-play actor updates its weights ONLY when the promotion gate passes. In
> run3 this ran ~39% of training on a stale actor. After this WP: actor weight sync is
> CONTINUOUS and unconditional; the promotion gate controls ONLY the deploy tag. These are
> two seams with no cross-read, and the old behavior is not representable (R49).

*Source: `WPUF_dispatch.md` PHASE U Mission (verbatim). Enforcement rider, same file U-1.5:
"NOTHING ELSE — no mode knob (R49), no sims, no radius." A config knob that could restore
sync-on-gate is itself an R49 breach.*

## R50 — first sanctioned behavior change; change-list discipline [INLINE]

> **R50 change-list**: enumerate every existing test from CENSUS (e) that pins old
> behavior, with its disposition (rewrite to pin the new law / delete with grounds). This
> list is reviewed in REVIEW-design; IMPL may not touch a pinned test that is not on it.

*Source: `WPUF_dispatch.md` U-1.6 (verbatim). WP-UNFREEZE is the migration's first
sanctioned behavior change — every prior WP was zero-behavior — so the change-list is the
mechanism that keeps "sanctioned" from becoming "unbounded": a pinned test may only be
rewritten or deleted if it appears on a REVIEWED list, and the confound-hygiene rider
(U-1.7) holds encode/search parity goldens outside the blast radius entirely.*

## R51 — minimal artifacts at entry-gate stops [SUMMARY-ONLY]

> R51 (minimal artifacts at entry-gate stops).

*Source: `WPUF2_dispatch.md` REGISTER APPEND. Operative reading, evidenced by the WPUF-1
run it governed: when a hard entry gate fails, the dispatcher writes ONE queue entry and
nothing else — no scratch branch cut, no baseline sweep, no CENSUS, no tree mutation (see
`wp/WPUF/ADJUDICATION_QUEUE.md` ADJ-01, which is the compliant instance). Full text owed by
the operator.*

## R52 — ff-merge over cherry-pick, to preserve shas [SUMMARY-ONLY]

> R52 (ff-merge over cherry-pick to preserve shas).

*Source: `WPUF2_dispatch.md` REGISTER APPEND. Operative reading, and the operative
mechanism of Phase M: cherry-picking the WPSC stack would rewrite all 19 shas, invalidating
every sha recorded in `wp/WPSC/COMMIT_MANIFEST.md` and every gate-evidence row bound to
them. A fast-forward merge preserves them byte-identically. This supersedes the
cherry-pick language in R36/WPSC_dispatch and in `WPUF_dispatch.md` Global Rule 2. Full
text owed by the operator.*

## R53 — mechanical merge authority (this run) [INLINE]

> 1. Verify: `dev` HEAD == f1ad10f; `wpsc-scratch` tip == 101a9d3 with exact parent chain
>    down to f1ad10f (no divergence). Any mismatch → queue + STOP.
> 2. Full gate sweep on wpsc-scratch tip (pre-merge evidence).
> 3. `git merge --ff-only wpsc-scratch` on dev. ff-only is LAW: if git refuses, do NOT
>    resolve, rebase, or re-pick — queue + STOP.
> 4. Full gate sweep on the new dev HEAD; confirm floor file = 1535; record both sweeps
>    with timestamps in COMMIT_MANIFEST.md (R42 discipline applies to the merge exactly as
>    to a commit).
> 5. Do NOT delete wpsc-scratch or wpsc-scratch-pre-fix (operator deletes after final
>    review). Do NOT touch LICENSE:3 (HANDOFF-1 stays open — operator supplies the name).
> 6. Append the WPSC ledger rows to plan/port_ledger.md with the now-on-dev shas.

*Source: `WPUF2_dispatch.md` PHASE M (verbatim). R53 is the narrow, one-run grant that lets
the dispatcher act directly on `dev` — the authority is MECHANICAL (verify, sweep,
fast-forward, sweep, record) and carries no latitude: every listed failure mode is a
queue-and-STOP, not a judgment call. It replaces `WPUF_dispatch.md` Global Rule 2's entry
gate, and thereby resolves ADJ-01. It does not extend to any phase after M.*

## R54 — WP-AXIS2 stretch + v2 verify-read rider [INLINE]

> PHASE A — WP-AXIS2 (STRETCH, R54) — only if Phase U lands fully green. Scope per ruling
> R24 (in the register + ADDENDUM_A §1):
> - A-0 VERIFY-READ (find-only): extract run3_findings_v2.md §4.1/§4.2 (and any section
>   defining "rays 7", pooling, global channel) VERBATIM into wp/WPAX2/V2_EXTRACTS.md.
>   Cross-check against STATE_2026-07-24.md §4 rows A2/A5/A6 and ADDENDUM_A. Discrepancy
>   law: STATE + rulings WIN over the raw file (the file predates the reconciliation); any
>   contradiction → queue, never a silent reopen of locked frames (multicluster stays;
>   dense stays per R20; compute budget hard; no sims increase).
> - A-1 DESIGN: new registry graph-schema entry `gnn_axis_v2` BESIDE gnn_axis_v1 —
>   radius/win_length as data-only entry values; rays=7 as GATED new code (WIN_AXES,
>   EDGE_FEAT_DIM, edge-cap formula) behind versioned dispatch; the gnn_axis_v1 call path
>   stays byte-frozen and its parity suite untouched; global-channel adequacy decided here
>   (existing zero-attr dummy vs learnable features) FROM the v2 extracts; A6 orbit-spread
>   instrument (per-checkpoint D6 orbit-spread measurement, threshold config-typed) rides
>   along.
> - Oracles: v1 parity suite untouched and green (the non-negotiable); v2 entry builds and
>   round-trips through the versioned dispatch; a config selecting a nonexistent axis
>   version raises named; edge-cap formula property-tested at rays=7; orbit-spread
>   instrument produces a measurement on a fixture checkpoint.
> - Out of scope: any training run, any v1-vs-v2 strength claim (that is run5 mint's
>   explicit decision), pooling changes on the DENSE path (A5 territory).

*Source: `WPUF2_dispatch.md` PHASE A (verbatim). Priority rider, same file PHASE U: "This
is the priority phase: if time or loops run short anywhere in this run, Phase A is what
gets dropped, never Phase U's rigor." R54 is what unblocks R24's "WP-AXIS2 DESIGN blocked
until inputs/run3_findings_v2.md exists on disk" (ADDENDUM_A §1) — the file now exists, and
A-0's verify-read with its discrepancy law is the sanctioned way to consume it.*

---

# R55–R58 — operator adjudication response, 2026-07-26 (WPUF-2)

Verbatim from the operator's ADJUDICATION RESPONSE. These resolve ADJ-02, ADJ-04 and ADJ-05
and ratify the WPUF-2 disclosures. **[INLINE]** — operative text supplied directly.

## R55 — anchor wiring is chunk U-w, not a separate card (resolves ADJ-04) [INLINE]

> ADJ-04 RESOLVED (R55): anchor wiring is NOT a separate card. It is chunk U-w, the first
> commit of Phase U: "fix(selfplay): wire actor anchor — port defect (WP-SP/WP11-A parity
> gap)", restoring run3 sync-on-gate semantics as the reference state; subsequent U chunks
> replace it per R49. Grounds: attribution windows are run-scoped and no run intervenes;
> bisect isolation is preserved by the chunk commit. Riders: (1) port_ledger defect row for
> the parity gap; (2) re-verify the R50 change-list against CENSUS before freezing oracles —
> record in the log that no existing test pinned sync timing and that this is how the total
> freeze survived; (3) end-state oracles freeze unchanged. DESIGN stands as verified; this is
> a chunk restructure, not a revision loop. ORACLE-WRITE and IMPL are unblocked.

*Dispatcher note: this OVERTURNS both the dispatcher's recommendation and DESIGN §0's
endorsement of it. The confound argument both made — "unfreeze is THE change", so the anchor
repair must not ride along — is answered by the operator's grounds: attribution windows are
**run-scoped**, and no run intervenes between the repair and the unfreeze, so there is no
window to confound. Bisect isolation, the other thing a separate card would have bought, is
preserved by making it its own chunk commit. `CARD-ANCHOR-WIRING` is therefore **not
created**; the `xfail(strict=True)` regression-arming oracle DESIGN specified is **not
landed** — the defect is fixed outright at U-w and the end-state oracles assert the real
post-unfreeze law.*

## R56 — ten-arm count accepted; arm 8 stays registered-open with two riders [INLINE]

> ADJ-05 RESOLVED (R56): ten-arm count accepted. Arm 8 remains registered-open under WP12-R,
> with riders: gate-11 exemption comment cites the WP12-R handoff row; add a committed test
> proving arm 8's reachable paths fail loud or provably match the configured encoding. If
> that test cannot be written, arm 8 escalates to hard run5-mint blocker.

**MERGE APPEND (R228, 2026-08-04):** R56 → SATISFIED-AND-SUPERSEDED (ADJ-3). Arm 8 closed
in `29f304b`; the gate-11 exemption comment and the debt row it pointed at no longer exist.
The ruling is not left pointing at a debt row that no longer exists. Text authored at
`PHASE_Q.md:286-287`; staged at the WP12-R merge commit per R150/R170 cell 7.

## R57 — the local serialized sweep is the binding pre-cutover gate (resolves ADJ-02) [INLINE]

> ADJ-02 RESOLVED (R57): the local serialized gate-sweep protocol with R42 timestamps is the
> BINDING pre-cutover gate (record in register). Demote ruff/pyright to advisory in one loud
> `ci:` commit carrying debt row CARD-LINT-TYPE (WP-R family). Remote + green Actions becomes
> an explicit cutover-battery item: satisfied or operator-waived.

*Consequence for R44: its "main-gate CI evidence is non-binding" clause is now permanent by
ruling rather than by accident — CI is not the authority, the local serialized sweep is.*

## R58 — WPUF-2 disclosures ratified; sweep serialization becomes standing law [INLINE]

> DISCLOSURES (R58): gate-11 rebuild ratified — commit the 31-evasion corpus as fixtures if
> not already committed. registry_sha_hex: write an honest re-justification in the log;
> fix-commit if the choice fails it. Standing law: gate sweeps serialized under exclusive
> load; concurrent-load results invalid.

*The standing-law clause generalises the dispatcher's own process error (three sweeps
contaminated by a concurrent load generator) into a rule binding on every future run: a gate
sweep and any deliberate load must never share a box, and a sweep taken under concurrent
load is INVALID — not "interpreted with caution", invalid.*

---

# R59–R63 — appended by the WPAX dispatcher at Phase M2, 2026-07-26

Source: `wp/WPAX/WPAX_dispatch.md` § "RULINGS EXECUTED THIS RUN", which supplies these five
texts verbatim for the register. Fidelity label on each is **[INLINE]** — the dispatch text
IS the ruling text, not a dispatcher gloss of a chat ruling.

## R59 — run5 arms the actor-lag abort now, not at mint [INLINE]

> R59: run5.yaml arms the actor-lag abort now, not at mint. Deliberate disarming remains
> legal for smoke configs; the minted production config defaults to armed. Draw-rate arming
> is audited (R61), not blind-flipped.

*Resolves ADJ-06 F-B (WPUF), which requested exactly this decision and recorded that with
the abort disarmed "a frozen actor emits an event and nothing else — the silent-failure mode
that cost run3 the run".*

*Dispatcher note, recorded because the audit clause earned its keep: the draw-rate half of
the arming instruction turned out to be **unsatisfiable at HEAD** — `draw_rate_threshold` is
a `StepCoordinatorConfig` code-side default with no config key at all, so it cannot be
armed, blind-flipped or otherwise. Filed as **ADJ-08**. R59's decision to audit rather than
flip is what surfaced this instead of shipping a config that looked armed.*

## R60 — CARD-SMOKE-SEAM is one card, one design [INLINE]

> R60: CARD-SMOKE-SEAM — smoke-arm collapse + ADJ-07 named error + F-C re-anchor are one
> card, one design, with the frozen composition oracle's renegotiation pre-planned under R43
> rather than tripped over.

*The "rather than tripped over" clause is the substance: WPUF's U-7 established that
deleting the smoke arm turns 13 tests red across 4 files including one byte-frozen oracle,
and that discovering this on the last available loop is what forced the deferral. R60 buys
the renegotiation up front, bounded by §S-3's enumeration discipline.*

## R61 — run5 mint gains a hard preflight gate [INLINE]

> R61: run5 mint gains a hard preflight gate: the actual minted config booted through the
> REAL composition root with REAL run-safety, asserting live sync events, live lag sourcing,
> and an arming audit against the committed armed-abort manifest. Production-only axes get
> production-time checks.

*This is the operational half of ADJ-06's sign-off, which recorded that the class-level
defenses "are not tests at all: structural … and operational". It targets the two axes WPUF
signed off as unclosable by any test — `_default_step_coordinator_config` (monkeypatched)
and `build_run_safety` (faked) — by varying them at production time instead.*

*See **ADJ-08**: assertion (c)'s manifest, as R61 fixes its initial content, cannot go green
on any config at HEAD. The dispatcher proceeds under Option B (typed `status` per manifest
row; `required` rows gate, `DEFERRED` rows print loudly — the R56 `KNOWN_DEBT` pattern), and
flags it for ratification.*

## R62 — CARD-EVAL-CLOCK [INLINE]

> R62: CARD-EVAL-CLOCK — the 90s wall-clock eval-round tests are R46-class; inject a clock or
> restructure; pre-cutover. (NOT in this run's scope; recorded as an owned card.)

*Records as an owned card what WPUF's Phase R3 found by accident and twice flagged as debt:
`tests/eval/test_round_end_to_end.py::test_full_headless_round_end_to_end` and
`::test_round_records_carry_regime_key_on_every_record` carry a fixed 90s wall-clock deadline
with no injected clock. R3 confirmed them "structurally different" from the GIL race it
deflaked and outside R46's named scope. R62 gives them an owner and a deadline. **No WPAX
phase touches them**; the run's own sweeps are the reason they must be fixed — they are the
tests that went red under contaminated load.*

## R63 — staging is explicit paths, never `-A` [INLINE]

> R63: staging is explicit paths, never -A.

*Generalises WPUF dispatcher process error #3 into standing law. That error swept
`docs/contracts/run_config_schema.md` — WP14's file, which R11 says must stay dirty and out
of every commit — into the Phase U IMPL commit; the tell was `git status` coming back
completely clean when the R11 file should always remain modified. It was corrected by
`git reset --soft HEAD~1` (orphaning `93e91e3` in favour of `60c3689`), but a dispatcher who
had not noticed would have handed another WP's uncommitted work to this one. R63 removes the
judgment call: there is no situation in which `git add -A` is the right command in this
repo.*

---

# R64–R69 — operator adjudication response, 2026-07-27 (WPAX mid-run)

Verbatim from the operator's ADJUDICATION RESPONSE, received mid-run between Phase S and
Phase P. These ratify Phase S as landed, resolve **ADJ-08** and **ADJ-09**, pre-authorize one
bounded R43 event, and add two standing process laws. Fidelity label on each is **[INLINE]** —
the operative text is supplied directly, not glossed from a chat summary.

**Phase S: RATIFIED as landed** (`ae5b1d0` + `94977cf`). Named explicitly in the ratification:
the disk-quota sweep discard, the in-loop tail-pin take, the last-loop production escalation,
the disclosed review compression, and the C-4 green-pin exception. Every one of those was a
dispatcher call disclosed in `wp/WPAX/DISPATCH_LOG.md` rather than a silent deviation; all
five are now ratified rather than merely unchallenged.

## R64 — Phase P posture: full production reality [INLINE]

> R64 — Phase P posture: full production reality. Real `build_net` model, real
> `build_run_safety`, real coordinator config, `eval_enabled` per the minted config's own
> value. `eval_enabled=False` is BANNED as an escape. Any wall the real boot hits
> (`terminal_eval`, `.arch`, or new) is a TREE DEFECT: fix in-run if small and within loops,
> queue if not — never design around it in the tool. The preflight's job is to hit exactly
> these walls before run5 does.

*This tightens R61 rather than restating it. R61 said "no fakes and no monkeypatches" of
`build_run_safety` and the coordinator config; R64 extends the ban to the model and to
`eval_enabled`, and — the operative half — reclassifies every boot wall as a defect in the
TREE, not an input the tool may route around. Phase S's own IMPL notes (D-5) recorded that the
`eval_enabled=True` drives needed `C-3`'s second `.arch` wall patched; R64 says the preflight
must walk into that class of wall deliberately, because run5's mint will.*

## R65 — ADJ-08 RESOLVED: Option B ratified, plus a new Phase D [INLINE]

> R65 — ADJ-08: Option B ratified (typed status rows; DEFERRED prints loud). NEW PHASE D
> after P, before A — CARD-DRAWRATE-KEY: `draw_rate_threshold` + arming become typed config
> (one resolver, code-side 0.0 literal dies per rule 1), wired through the strict construction
> site, run5 re-minted armed, manifest row flips DEFERRED→required. Full-but-light pipeline.
> Priority: P > D > A.

*The ADJ-08 table offered A / B / C; the operator took **B and then C**, sequenced. B is
ratified for what Phase P ships — so the manifest's typed `status` rows and the loud DEFERRED
print are built as designed — and CARD-DRAWRATE-KEY then closes the gap in its own phase,
flipping that same row to `required`. This is exactly the "cheap to reverse into C" property
the dispatcher argued for B on, being exercised. Note "code-side 0.0 literal dies per rule 1":
`train/coordinator/config.py:182`'s `draw_rate_threshold: float = 0.0` is not to survive as a
second default authority beside the schema field (R1 / LAW-08). Phase priority is restated as
**P > D > A**, which supersedes the dispatch's Global Rule 5 ordering only by inserting D — S
and P remain the never-dropped pair, and A still drops first.*

## R66 — ADJ-09 RESOLVED: Option B ratified; WP14 promoted [INLINE]

> R66 — ADJ-09: Option B ratified (land S-4, amend `repo_design.md`, contract-doc update owed
> to WP14). WP14 is promoted to the next run; R11 dissolves there.

*Ratifies as landed what `94977cf` already did. The forward half is the news: **WP14 is the
next run**, and R11 — the standing bar on touching `docs/contracts/run_config_schema.md` —
**dissolves there**. So the owed contract-doc update has a dated owner rather than an open
card, and the R11/§4 precedence question the queue asked to be settled generally is answered
by expiry: the conflict cannot recur past WP14.*

## R67 — pre-authorized bounded R43 event [INLINE]

> R67 — Pre-authorized bounded R43 event (S-3 discipline: enumerated hunks, before/after
> hashes, dispatcher-verified, one commit): fold the F-2 census into
> `tests/train/test_actor_lag_watchdog.py` and correct N-3's three stale sentences. Land any
> time after Phase P.

*Closes the debt the Phase S fix pass booked when it STOPPED rather than edit a frozen file:
the F-2 census landed in a non-frozen sibling with an authority note, and RED-TEAM-2's N-5
recorded the honest weakness of that siting (deleting the census **and** restoring the F-2
defect together yields 0 failures — only the count floor catches it). N-3 is the companion
correction: three stale sentences in the frozen file, **no assertion depending on any of
them**. The grant is bounded by §S-3's discipline verbatim — enumerate hunks with before/after
hashes, one commit — and it is the second and last pre-authorized exception to R43 in this
run. `tests/train/test_actor_lag_watchdog.py` is frozen at `5638b90db43866e6`.*

## R68 — subagent scratch hygiene; sweeps assert free disk [INLINE]

> R68 — Subagent briefs mandate scratch cleanup; sweeps assert + record free disk before
> starting.

*Standing law, generalising the disk-quota sweep discard that R64's preamble ratifies: a sweep
this run had to throw away because the box ran out of disk under accumulated per-stage
scratchpad copies. Two obligations, on two different actors — every subagent brief carries a
cleanup clause, and every sweep records free disk in its evidence row the way it already
records load average under R58. A sweep with no recorded free-disk figure is missing evidence,
not merely undocumented.*

## R69 — every "verified under X" claim names its mechanism [INLINE]

> R69 — Every "verified under X" claim names its producing mechanism or is struck; repeating
> an unproduced claim is the same violation.

*Standing law, and the register entry for N-7. Phase S's IMPL reported the tier green "under
random order"; REVIEW-impl repeated it; the dispatcher repeated it a third time in the log —
and RED-TEAM found **no random-order plugin is installed**, so `-p no:randomly` was a no-op and
three reports were one deterministic run. The second clause is the sharp one: **repeating** an
unproduced claim is the same violation as making it, so a reviewer inherits no immunity from
having read it upstream. This is LAW-07's producer-test rule lifted out of gate inputs and
applied to prose — a claim in a log is a gate input for whoever reads it next.*

---

# R70–R72 — operator adjudication response, 2026-07-28 (WPAX, post-Phase-P)

Verbatim from the operator's ADJUDICATION RESPONSE. These ratify the Phase P commit-on-FAIL,
authorize **ADJ-13** in full, and promote the dispatcher's R-P2 extension to standing law.
Fidelity label on each is **[INLINE]**.

## R70 — Phase P commit-on-FAIL RATIFIED, and the six-part test becomes standing rule [INLINE]

> R70 — Phase P commit-on-FAIL RATIFIED. Standing rule: committing on a failed RED-TEAM is legal
> only when ALL of: (1) measured strictly-better-than-absent, (2) findings latent not live
> (measured), (3) FAIL + caveat disclosed into the artifact's consumer surface, (4) scratch not
> dev, (5) loop budget exhausted, (6) residue adjudicated. All six held. Caveat blocks any mint
> use of gate 12 / preflight rc until discharged.

*The dispatcher offered three reasons for committing `72f0872`; the operator ratified on **six**,
and the three additions are the load-bearing ones. **(2) "latent not live (measured)"** is what
separates this from waving a finding through: `configs/` was measured to hold five top-level
`.yaml` files with no `.yml` and no subdirectories, so F-1's escape had no live instance.
**(3) disclosure into the artifact's CONSUMER surface** — not merely a log a later reader might
find, but the place a mint sign-off will actually look. **(6) residue adjudicated** — the findings
had to be filed as a decision request, not carried as a TODO. The final clause binds the mint:
**gate 12's rc 0 and any preflight rc are unusable as mint evidence until the caveat is
discharged**, which ADJ-13's fix commit is what discharges.*

## R71 — ADJ-13 AUTHORIZED in full; the class-fix law [INLINE]

> R71 — ADJ-13 AUTHORIZED in full (F-1, F-2, F-3, F-5, F-6 + three nits), R67 discipline +
> delta-scoped adversarial recheck. Class-fix law (MF-7): every fix names its class; flip-sets
> cover the class boundary, not the demo input. F-1's audit set derives from the loader's own
> discovery authority — one authority — with .yml / subdir / novel-extension flip rows.

*Three things, and the second outlives the card. **The authorization** covers all six findings plus
the three nits under R67's discipline (enumerated hunks, before/after hashes, dispatcher-verified,
ONE commit), with a **delta-scoped adversarial recheck** rather than a full RED-TEAM. **The
class-fix law** generalises MF-7's failure — the fix closed the reviewer's `run6.yaml` while
`run6.yml` and `configs/prod/` walked through — into a rule binding on every future fix: name the
class, and make the flip-set cover the **class boundary**, not the input that demonstrated it.
**F-1's shape is specified rather than left open**: the audit set derives from **the loader's own
discovery authority**, so gate 12 cannot hold a second hand-maintained answer to "what is a config"
(R1/LAW-08) — with `.yml`, subdirectory and novel-extension rows in the flip-set.*

## R72 — the R-P2 extension becomes standing law [INLINE]

> R72 — R-P2 extension is standing law: every conjunct of every shipped predicate appears in some
> flip-set. Applies to the ADJ-13 delta and all subsequent phases.

*Completes the ruling the dispatcher made under R-P2 and RED-TEAM bounded. R-P2 replaced DESIGN
§12's false "each mutation flips exactly one predicate" with non-empty, exactly-as-declared,
pairwise-distinct flip-sets — which RED-TEAM verified **sufficient** for the anti-shotgun property
but showed never claimed **corpus completeness**, the gap F-5 and F-6 fell through. R72 closes it
from the other end: coverage is measured **per conjunct of every shipped predicate**, so a
predicate can no longer ship a clause no mutation exercises.*

---


# R73–R77 — operator adjudication response, 2026-07-28 (WPAX, post-R67)

Verbatim from the operator's ADJUDICATION RESPONSE. These ratify R67 as landed, close a WP-R
card, **DECLINE a production change that had already landed**, amend R68, and rule on the R72
instrument. Fidelity label on each is **[INLINE]**.

## R73 — R67 ratified, and name-truth becomes law [INLINE]

> R73 — R67 Option A + fourth hunk + rename RATIFIED. Standing: name-truth is part of
> behavior-naming law; bounded frozen-edit grants implicitly include renames made necessary by
> the granted change, disclosed in the same event.

*Both halves of the disclosure are ratified — the unforecast fourth hunk (`import pytest`, which
parametrizing required) and the rename the dispatcher flagged as past the grant's letter. The
standing rule generalises it: **a test name is a behavioural claim**, so a name made false by a
granted edit must be corrected in that same edit, and a bounded frozen-edit grant carries that
correction implicitly. The dispatcher no longer needs to ask; it needs to **disclose in the same
event**, which is what happened here.*

## R74 — WP-R §9.10 CLOSED [INLINE]

> R74 — WP-R §9.10 CLOSED (defect closed in-fact by revalidate_run_config, Phase S; validated by
> the 147-leaf sweep). Residue as originally booked: model_construct fails loud at use.

*R67's item 2 struck the sentence that was keeping this card alive citing a hole that no longer
existed; R74 closes the card itself. Note the operator validates it on **RED-TEAM-2's 147-leaf
`model_dump()` fidelity sweep** — the same evidence DESIGN_P declined to inherit and re-derived
independently at 970 leaves. Both stand; the card closes on the original.*

## R75 — the loader accept-set narrowing is DECLINED [INLINE]

> R75 — Loader accept-set narrowing DECLINED. The shared-authority invariant (loader accepts ⇒
> audit sees) is the protection; preflight covers the mint path shape-agnostically. Parked as a
> WP-R row; reopening requires a concrete escape the invariant misses.

***This ruling reverses a production change that has already landed*** in `4d11147`, and it is
the only ruling in this run to do so. The corrective pass closed ADJ-13 F-1's class by making
`load_config` **refuse** any suffix outside `CONFIG_SUFFIXES` (recheck shape (b)); the dispatcher
flagged it as a shipped-package behaviour change and asked for ratification. **The operator
declines it.**

The substituted protection is the **shared-authority invariant: whatever the loader accepts, the
audit must see.** That closes the same class from the other side — the escapes
(`run6.yml` → `configs/prod/` → `run6.txt` → `run6.YAML`) all worked because the launchable set
was strictly larger than the discoverable set, and the invariant forbids that gap without
constraining what a run may be launched from. The mint path is covered **shape-agnostically** by
the preflight, so `tools/mint_config.py`'s suffix guard is not the mechanism either.

Consequences the dispatcher must execute: the `load_config` refusal comes **out**; discovery
widens so the invariant holds; any constant left without a live consumer dies (R1/LAW-08); and
the tests asserting the refusal invert. **Reopening requires a concrete escape the invariant
misses** — not an argument that one might exist.

## R76 — R68 AMENDED [INLINE]

> R76 — R68 AMENDED: sweeps measure+record RAM; no tree copies in tmpfs; subagent harnesses
> single-worker unless the brief grants otherwise.

*Adopts the dispatcher's proposal after the OOM incident. R68's original free-disk clause was
measuring the wrong quantity for the whole run: every stage reported `df -h /tmp` as disk hygiene,
but `/tmp` is a **tmpfs — it is RAM**, which is why the box reached global OOM while every artifact
recorded "16G available". Three clauses now: sweeps **measure and record RAM**; tree copies are
**barred from tmpfs**; harnesses are **single-worker by default**, and a brief must grant otherwise
explicitly. On this evidence the Phase S sweep loss that motivated R68 was the same memory
exhaustion misread as a disk quota.*

## R77 — the R72 instrument stays; a lightweight gate-family check is opened [INLINE]

> R77 — The R72 instrument stays in wp/WPAX/ as evidence. WP-R row opened: conjunct-coverage as a
> lightweight repeatable gate-family check — build small or don't build.

*Ratifies the dispatcher's judgment call to lift five files (136 KB) out of the deleted scratch
before removing it. The forward half is a scoped card: conjunct-coverage becomes a **repeatable**
check for the gate family, with an explicit size constraint — **"build small or don't build"** —
which forecloses the obvious failure mode of a coverage instrument growing into a second test
framework.*

---


# R78–R79 — operator adjudication response, 2026-07-28 (WPAX, Phase D authorization)

Verbatim from the operator's ADJUDICATION RESPONSE. These bound Phase D's scope to one knob and
fix the SHAPE of arming for the whole repo. Fidelity label on each is **[INLINE]**.

## R78 — Phase D authors exactly ONE knob [INLINE]

> R78 — Phase D authors exactly ONE knob. The ~24 code-side coordinator knobs are
> CARD-COORD-KNOBS (pre-run5-mint, own card). Preflight-JSON dump of resolved coordinator config
> is that card's first design question.

*Forecloses the scope creep the seam invites. `_default_step_coordinator_config()` supplies ~24
unauthored code-side knobs, and ADJ-08 recorded that seam as "one seam, three phases" — S-4 took
`stop_step`, Phase D takes `draw_rate_threshold`, and the rest are now a **named card with a
deadline** (pre-run5-mint) rather than an open invitation. The rider is a design steer the card
inherits: the **preflight's JSON dump of the resolved coordinator config** is where CARD-COORD-KNOBS
starts — i.e. make the unauthored values *visible in the mint record* before deciding which become
config.*

## R79 — single-authority arming; no boolean proxy beside a gating value [INLINE]

> R79 — Single-authority arming: no boolean enable proxy beside a gating value. Armed/disarmed is
> a property of the resolved value (explicit off-semantics), and the manifest row asserts the
> resolved-value condition — one fact, one authority, pin and manifest bound to the same thing.

***This overrules the obvious reading of R65 and the shape the in-repo precedent suggests.*** The
actor-lag row pairs a boolean (`monitor.actor_lag_abort_enabled`) with a threshold
(`monitor.actor_lag_threshold_steps`), and mirroring it for draw-rate — a `draw_rate_threshold`
plus a `draw_rate_abort_enabled` — is what a designer would reach for. **R79 forbids it.**
`draw_rate_threshold` already **gates its own check** (`step.py:421`: `if draw and
cfg.draw_rate_threshold > 0`), so a boolean beside it would be a **second authority over one
fact** (R1/LAW-08) and could contradict it — armed-by-boolean, disarmed-by-value.

Three binding consequences: **(1)** arming is a property of the **resolved value**, with
**explicit off-semantics** — the off state is a value an operator writes deliberately, never a
default that happens to disable; **(2)** the manifest row asserts the **resolved-value
condition**, not the existence of a schema field; **(3)** the pin and the manifest bind to **the
same fact**, which is what closes RED-TEAM's F-4 route where a schema key could be added while the
dataclass default survived and the gate still went green.*

---


# R80–R83 — operator adjudication response, 2026-07-28 (WPAX Phase D)

Verbatim from the operator's ADJUDICATION RESPONSE. These close **ADJ-14** and **ADJ-15**, fix
run5's armed value, **clarify R78** and **amend R79**. Fidelity label on each is **[INLINE]**.

## R80 — ADJ-14 closed; R78 clarified to the abort FAMILY [INLINE]

> R80 — ADJ-14: Recommendation A STRENGTHENED-ADOPTED. Estimator defect closed with BOTH guards:
> min-sample inclusion (len(dq) ≥ min_samples) and non-zero min_step. R78 clarified: Phase D
> authors the draw-rate abort FAMILY (threshold + min_step + min_samples, one block, one
> resolver) — not one literal field, and not the coordinator-knob card.

*Two things, and the second re-scopes the phase. **The estimator defect is closed at its root**:
REVIEW-design measured that `instrumentation.py:365-371` includes any worker with `len(dq) > 0` —
**one game** — so a single drawn game per worker saturates the pool mean at `1.0`, at or above
every legal threshold. A `min_step` guard alone would not have fixed that, because the hazard is
the *inclusion rule*, not the step count; both guards are required and the operator says so
explicitly. **And R78 is clarified rather than overruled**: "exactly ONE knob" meant one *abort
family*, not one literal field. Phase D authors three keys as **one block with one resolver** —
which is the same single-authority shape S-4 set, applied to a family. `CARD-COORD-KNOBS` keeps
the other ~24 and is untouched.*

## R81 — ADJ-15: the narrow R43 grant is GIVEN, with conditions [INLINE]

> R81 — ADJ-15: narrow R43 grant GIVEN. One assertion re-points from "deferred row prints" to
> "deferred mechanism works" via synthetic manifest= row. _print_deferred_rows survives.
> Conditions: S-3 discipline; mutation self-test (killing _print_deferred_rows reds the
> re-pointed test alone). Keep-a-row-deferred REJECTED.

*The third and last pre-authorized R43 exception of this run, and the narrowest — **one
assertion**. The condition is the interesting half: the re-pointed test must be shown, by
mutation, to **red alone** when `_print_deferred_rows` is killed. Without that, re-pointing an
assertion from a real behaviour to a synthetic one could quietly turn a live pin into a
self-satisfying one, which is the exact species (`assert x is True` satisfied by a constant) that
this run has caught three times. The rejected alternative — keeping a row deferred so the
assertion stays true — is rejected because it **inverts the instrument**: it would shape the
shipped manifest to suit a test.*

## R82 — run5 arms at 0.25 [INLINE]

> R82 — run5 arms at draw_rate_threshold = 0.25. Grounds: healthy draw rate ≈ 0.025% (ply-cap
> truncations only); collapse clears 0.25 with margin; R80 guards prevent estimator saturation.
> min_step / min_samples proposed by DESIGN from measured deque geometry with grounds. All three
> pre-registered at mint prereg — the only place they may change.

*Supplies the number the repo did not contain, **with its grounds** rather than as a bare value:
healthy play draws only on ply-cap truncation (≈0.025%), so `0.25` sits four orders of magnitude
above the healthy rate and still well below a real collapse — margin on both sides. The operator
supplies only the threshold; **`min_step` and `min_samples` are DESIGN's to propose from measured
deque geometry with stated grounds**, which keeps R69 intact rather than inviting two more invented
numbers. **All three are pre-registered at mint prereg, "the only place they may change"** — so
they are run-scoped constants, not tunables.*

## R83 — R79 AMENDED; the named RED extended [INLINE]

> R83 — R79 AMENDED: shape is None = disarmed (explicit), present value gt=0, le=1. Named RED
> extended: builder-signature assertion (no default migration route) +
> __post_init__/object.__setattr__ resurrection. N-1's two pin-scan tests join the enumerated
> change-list.

*Closes **MF-1** and **MF-2** by ruling rather than leaving them to the fix pass. **MF-1**: `le=1`
kills the "armed in the config, absent in effect" route — a threshold `> 1.0` passed `gt=0`,
audited **ARMED**, and could never fire, reachable by the very percent/fraction unit slip the
design cited as its own motivation. **MF-2**: the named RED must also pin the **builder
signature** (the default authority migrating to a parameter — a route the change list itself
creates) and the `__post_init__` / `object.__setattr__` resurrection of a frozen dataclass field.
**And N-1's two pin-scan tests are now enumerated change-list rows**: they `assert pinned, "no
pinned row means this test has no subject"`, the deferred row is the only pinned row, so the
flip would have gutted them silently.*

---


# R84–R86 — operator adjudication response, 2026-07-28 (WPAX Phase D, pre-ORACLE)

Verbatim from the operator's ADJUDICATION RESPONSE. These close **ADJ-16**, accept Phase D's
proposed guard values, and settle R81's "alone" condition. Fidelity label on each is **[INLINE]**.

## R84 — ADJ-16 ratified as truthful; CARD-ABORT-EXIT opened BLOCKING [INLINE]

> R84 — ADJ-16: exit_code=None RATIFIED as truthful now. CARD-ABORT-EXIT opened, PRE-RUN5-MINT,
> BLOCKING: author the supervisor-visible mechanism (presumptive: registered exit code in the
> process contract, fail-fast family parity — card justifies), contract doc updated same commit
> per R9, manifest row flipped None → authored value, mutation test proves a fired abort is
> supervisor-distinguishable from a clean run.

*Ratifies the **refusal to invent a `46`** — `exit_code=None` is truthful now, and a fabricated
code in a manifest a mint reads would have been the R1 class with an unproduced number. The
forward half is a **new BLOCKING pre-run5-mint card**, and it is fully specified rather than
gestured at: author the supervisor-visible mechanism (presumptively a registered exit code with
fail-fast family parity, but **the card justifies its own choice**), update the contract doc in the
**same commit** per R9, flip the manifest row `None` → the authored value, and prove by
**mutation** that a fired abort is distinguishable from a clean run. That last clause is what stops
the card being satisfied by a number that nothing produces.*

**Run5-mint blocker count: this is the second.** `CARD-COORD-KNOBS` (R80/ADJ-14) and
**`CARD-ABORT-EXIT`** both gate the mint.

## R85 — the guard values are accepted, with a Stage-0 revisit [INLINE]

> R85 — Guards ACCEPTED: min_samples = 50 (= _DRAW_RATE_WINDOW, le= pinned, 51-counterexample
> recorded), min_step = 25000 (minted precedent, missed-abort-cheaper asymmetry). Prereg note:
> min_step revisable at mint prereg once a real boot measures early-run draw distribution —
> Stage-0 discipline.

*Both proposals accepted as argued. The operator explicitly preserves the two things that made
them honest: `min_samples`'s **`le=` pin with its 51-counterexample** (at 51 the estimator can
never fire — MF-1's class on a second axis), and the design's own **stated boundary** on
`min_step` — measured geometry, but a choice from *minted precedent* rather than from a measured
early-run draw distribution. R85 converts that boundary into a scheduled revisit: **`min_step` is
revisable at mint prereg once a real boot measures the distribution**, under Stage-0 discipline.
The number is not frozen by being accepted; it is accepted *with its evidence gap named and
owned*.*

## R86 — R81's "alone" settled [INLINE]

> R86 — R81 condition means: not self-satisfying, no unrelated casualty. Five-node family casualty
> is in-subject. Ratified.

*The dispatcher flagged that killing `_print_deferred_rows` reds **five nodes in one family**
rather than literally one, because re-basing the `[×4]` parametrized test costs its subject too,
and declined to reinterpret the operator's own condition silently. R86 settles it: **"alone" means
not self-satisfying and no unrelated casualty** — an in-subject family is in-subject. The census
(one call site, no test naming the function, no unrelated casualty) is what satisfies it.*

---


# R87–R88 — operator adjudication response, 2026-07-28 (WPAX Phase D, pre-IMPL)

Verbatim from the operator's ADJUDICATION RESPONSE. These close **ADJ-17** and add a standing
design-stage obligation. Fidelity label on each is **[INLINE]**.

## R87 — ADJ-17: the second bounded R43 grant is GIVEN [INLINE]

> R87 — ADJ-17 GRANTED: §S-3 discipline against 0f42484c1ce1c980, hunks extracted verbatim from
> the design's fenced block, R81 condition under its R86 reading (mutation-red per re-pointed
> assertion, no unrelated casualty). The vacuous assertion is in the enumeration: re-point to a
> live subject or delete with grounds. Land-keys-defer-flip REJECTED — the manifest is a
> mint-read artifact; it does not assert dead deferrals.

*The fourth and final pre-authorized R43 exception of this run. Three clauses earn their place:
**"extracted verbatim from the design's fenced block"** — the same discipline ORACLE-WRITE used
for R81's hunk, which is why that hunk landed byte-exact rather than retyped-with-drift; **the
vacuous assertion is IN the enumeration** — an assertion that goes *vacuous* is as much a hunk as
one that goes red, and it must be re-pointed to a live subject or deleted **with grounds**, never
left to pass emptily (the class REVIEW-impl found in Phase P as `assert all(...)` over an empty
list); and the rejection is reasoned rather than flat — **the manifest is a mint-read artifact, so
it does not assert dead deferrals.***

## R88 — DESIGN owes a frozen-impact census, from Phase A onward [INLINE]

> R88 — Standing from Phase A onward: DESIGN's change-list includes a frozen-impact census (drive
> the flip against stand-ins, declare every frozen file touched, request R43 grants IN the
> design). Two reactive stops become one review-time adjudication.

*Generalises the pattern the dispatcher named: **R43 fired twice in one phase and both times the
stage stopped rather than bent** — MF-3 caught `test_preflight_mint.py` (→ R81), ORACLE-WRITE
caught `test_armed_abort_manifest.py` (→ R87). Each stop was correct behaviour and each cost a
round trip, because the flip's blast radius on frozen oracles was under-measured by DESIGN both
times. R88 moves the measurement forward: **drive the change against stand-ins at design time,
declare every frozen file touched, and request the grants IN the design** — converting two
reactive stops into one review-time adjudication. Note the technique is already proven in this
run: ORACLE-WRITE derived ADJ-17's four-row damage table against post-delta stand-ins **without
touching the file**, which is exactly the census R88 now requires of DESIGN.*

**Boundary condition, binding on Phase D's IMPL** *(operator, same response)*: the post-IMPL
perimeter re-hash must **account for every frozen file, with each drift traced to its numbered
grant beside the hash** — no bare hash lists.

---


## R89 — Phase D ratified, its compression residual SCHEDULED [INLINE]

> R89 — Phase D RATIFIED with its compression residual SCHEDULED, not accepted: Phase D's
> RED-TEAM was dispatcher-driven with no fresh-context adversary and zero downstream
> adversarial stages. The next dispatch opens with a delta-scoped fresh-context adversarial
> recheck of Phase D's arming surfaces (Phase DR); the Phase D caveat lifts ONLY on that
> recheck's clean verdict, and Phase D remains PASS-WITH-DISCLOSURE until it does.
> [Amended by R92: DR returned FINDINGS; the lift condition is now DR-FIX + Phase DS passing
> a narrow adversarial re-verify against DR's battery.]

*Supplied by the operator 2026-07-28, filling the gap this register had recorded as
`R89 — MISSING TEXT`. The recorded gap is now CLOSED; the earlier entry's substance — that
inferring the ruling from the dispatch's summary would have put dispatcher-authored prose
under an operator ruling number — stands as the reason it was left empty rather than
reconstructed. **The recheck ran and returned FINDINGS** (WPMINT Phase DR), so the clean-verdict
condition was NOT met and the caveat did not lift; R92's amendment re-scopes the lift rather
than waiving it.*

---



## R90 — delegation package for the WPMINT run [INLINE]

> R90 — Delegation package (WPMINT run): the dispatcher proceeds without
> operator round-trips where law is settled: (a) settled-class frozen-oracle
> grants (re-point/extend assertions whose subject the card deliberately
> changes) auto-granted under S-3 discipline — enumerated hunks, verbatim
> extraction, before/after hashes — plus the R81/R86 mutation condition (not
> self-satisfying, no unrelated casualty), recorded exactly as a queued
> adjudication would be; (b) commit-on-FAIL applies R70's six conditions
> directly; (c) one third revision loop per phase, pre-authorized iff loops
> 1–2 converged and the fix scope is enumerated; (d) phase reordering with
> recorded grounds; (e) STOP-class items halt only phases sharing their
> surface, independence argument recorded. HARD STOPS: new adjudication
> classes; any dev mutation; scope widening beyond carded scope; any change
> to run5's armed values (0.25 / 25000 / 50 — mint-prereg-only, R82/R85).

---


## R91 — WP14 mechanics; R11 dissolves [INLINE]

> R91 — WP14 mechanics: R11 dissolves inside the WPMINT run. The dirty
> docs/contracts/run_config_schema.md working copy is WP14's INPUT — its
> accumulated content is curated against the shipped tree and committed,
> never discarded. From WP14's commit the tree-clean invariant is FULLY
> CLEAN; every gate asserting the R11 exception drops it in the same commit;
> ADJ-09's owed contract-doc update and all R9 same-commit debts settle in
> WP14. WP14's design question: a contract-doc drift gate deriving from the
> live schema authority (gate-12 pattern, never a parallel copy) — build
> small or record why not.

*Both texts supplied verbatim by the operator in the WPAX→dev merge dispatch (2026-07-28) and
appended by that dispatcher; `wp/WPMINT/WPMINT_dispatch.md` was already written against them.
**R91 operational note for the merge itself:** the dirty `run_config_schema.md` working copy
survived the fast-forward byte-identical (sha256 `d4f99cfb…c99f7bd7`, verified both sides), and a
copy was taken before checkout. It is untouched by all 9 WPAX commits and identical between
`ca4569f` and `d0b3974`, so the merge could not have endangered it — but R91 makes it an INPUT
rather than debris, so the verification is recorded rather than assumed.*


# R92–R93 — operator adjudication response, 2026-07-28 (WPMINT, post-DR)

Verbatim from the operator's ADJUDICATION RESPONSE. These resolve **ADJ-18/ADJ-19**, dissolve
**ADJ-20**, ratify the **DR-11** routing, and amend **R89**'s lift condition. Fidelity label on
each is **[INLINE]**.

## R92 — ADJ-18/19 RESOLVED; the statistic is authored; Phase DS opened [INLINE]

> R92 — ADJ-18/19 RESOLVED. Statistic: pooled count-weighted rate
> (Σ draws / Σ completed over the union of worker windows). Insufficient evidence
> (Σ completed < N_pool_min) → NO observation, never a healthy 0.0; zero-completion starvation
> is the stall family's jurisdiction. Guards: min_step=25000 and consec=3 stand; per-worker
> min_samples DELETED; N_pool_min proposed by DESIGN from measured geometry, joins the prereg
> row. Threshold 0.25 CARRIES OVER — basis unchanged (R82 amendment by its issuing authority
> via prereg; the hard stop stands for all other actors). Lands as Phase DS after DR-FIX: DR's
> counterexamples become permanent regression oracles (0.968 fires, 0.0319 silent); R88 census;
> Phase D oracle re-points are settled-class citing R92. Caveat lift = DR-FIX + DS pass a narrow
> adversarial re-verify against DR's full battery.

*Answers both hard stops with ONE mechanism rather than patching each. **The statistic** —
Σ draws / Σ completed over the union of windows — kills ADJ-18 in both directions at once: it is
count-weighted, so a single included worker can no longer carry the pool mean (the 0.0319 false
positive), and no worker can be excluded into invisibility (the 0.968 false negative), because
there is no per-worker inclusion bar left to exclude anyone — **`min_samples` is DELETED**, not
re-bounded, which is also why **ADJ-20 dissolves rather than being answered**: the key whose floor
was in question no longer exists. **The empty case** is answered by TYPE, not by value: below
`N_pool_min` the gate makes NO OBSERVATION, so ADJ-19's "healthy 0.0 appended as a real
measurement" becomes unrepresentable, and zero-completion starvation is explicitly assigned to the
stall family instead of being silently absorbed here. **The armed values move by their own
authority**: 0.25 carries over on unchanged basis, and R82's amendment is made by the operator
via prereg — the register records that the HARD STOP on run5's armed values "stands for all other
actors", so the dispatcher's bar is unchanged. And the lift condition is re-scoped, not waived:
DR's two measured counterexamples become PERMANENT regression oracles, which converts the finding
that broke the recheck into the instrument that prevents its recurrence.*

## R93 — the drain-keys finding is routed to Phase K, with conditions [INLINE]

> R93 — Drain-keys finding routed to Phase K, RATIFIED, with conditions: K wires the four keys
> for real; every consumer-registry citation K touches is verified by MUTATION (set knob →
> observe consumer), not grep; the SC-A3 "hardcaps wired" ledger row is corrected; the
> false-citation class is logged for WP-R. DR-6's _leaf_paths optionality fix is settled-class in
> DR-FIX; generality note recorded.

*Ratifies the dispatcher's DR-11 routing and then hardens the METHOD that missed it. The
load-bearing condition is **"verified by MUTATION, not grep"**: DR-11 existed because a registry
string NAMED a consumer path that `data.pop("drain")` had already discarded, and a grep-verified
citation cannot tell a reader from a `pop`. Mutation — set the knob, observe the consumer — is the
only check that distinguishes them, and it is now binding on every citation Phase K touches.
Three consequences follow rather than one fix: the **SC-A3 ledger row** that recorded these
hardcaps as wired is corrected (the false claim outlived the code in the ledger too), and the
**false-citation class** is logged for WP-R so the sweep looks for siblings rather than treating
these four as isolated.*

---


# R94–R96 — operator adjudication response, 2026-07-29 (WPMINT close)

Verbatim from the operator's ADJUDICATION RESPONSE. These close **ADJ-21** and **ADJ-22**, and
add a standing law on corrections. Fidelity label on each is **[INLINE]**.

## R94 — ADJ-21 RATIFIED; the floor validator stands [INLINE]

> R94 — ADJ-21 RATIFIED. Floor validator stands; grounds recorded: healthy draw rate ≈ 0.00025,
> floor 0.02 ≈ 80× healthy; sub-floor thresholds require a deliberate docs(design) schema
> amendment. Docstring cites R94 + basis.

*Ratifies the constraint Phase DS added unbidden, and — as R82 did for `0.25` — supplies the
**number's grounds** rather than a bare approval. Healthy draw rate ≈ **0.00025**, so the induced
floor of `0.02` sits ≈ **80× above healthy**: far enough that no legitimate posture is lost, close
enough that it is not an arbitrary wall. The escape hatch is named and deliberately expensive —
a sub-floor threshold needs a **`docs(design)` schema amendment**, not a config edit, which is the
R9 shape applied to a bound. And the docstring must **cite R94 and the basis**, so the next reader
finds the grounds at the constraint rather than in a register they may not open.*

## R95 — ADJ-22 RESOLVED: validators assert only what they OBSERVE [INLINE]

> R95 — ADJ-22 RESOLVED: validators assert only what they observe. The config-time check
> re-scopes to config-domain facts; evidence sufficiency remains runtime's (R92 no-observation
> rule). Lands as DSV-2 on wpmint-scratch with the confirmatory narrow adversarial pass against
> the post-fix state. Phase D's caveat lifts on DSV-2's clean verdict. MERGE AFTER DSV-2, not
> before.

*Answers the fifth "armed but unable to observe" axis by **fixing the CLAIM rather than widening
the check** — which is the only honest move available, because the thing ADJ-22 exposed is not a
missing test but a validator saying more than it can know. A load-time validator cannot see which
workers will report; asserting **reachability** was therefore an overclaim, and no amount of
config-time arithmetic can make it true. So the check keeps its (genuine) config-domain
arithmetic — you cannot require more evidence than the configured windows can physically hold —
and **surrenders the reachability claim to runtime**, where R92's no-observation rule already owns
it. This is R73 name-truth applied to a validator's own name, and it generalises: **a validator's
name and message may assert only what its inputs can witness.**

Two procedural clauses matter as much as the fix. It **lands as DSV-2**, not as a follow-up card —
so the run that opened the axis closes it. And **MERGE AFTER DSV-2, not before**, with Phase D's
caveat lifting on DSV-2's clean verdict: the caveat that has survived R89, Phase DR and DS-VERIFY
does not get discharged by a merge.*

## R96 — artifact-correction law [INLINE]

> R96 — Artifact-correction law: a withdrawn finding is corrected in every artifact downstream
> agents consume, not just the ledger. DSV-2 verifies the two production files carrying the
> DR-8-derived claim now state the grepped truth; if not, the correction rides the DSV-2 commit.

*Promotes the WPMINT dispatcher's own process failure into standing law. DR-8 was measured false
and withdrawn correctly — **in the ledger**; `RECHECK_D.md`, the artifact the next agent's brief
pointed at, was left unamended, so Phase DS read it, believed it, and shipped the false sentence
into `armed_aborts.py` and `coordinator/config.py`. DS-VERIFY caught it as DSV-2 and it blocked a
caveat lift. The law's force is in the word **every**: the correction must reach each artifact a
downstream agent consumes, and the ledger is not one of them. The verification clause closes the
loop rather than trusting the fix — **DSV-2 checks the two production files**, and a correction
that did not land rides the DSV-2 commit instead of being assumed.*

---


# R97–R99 — operator adjudication response, 2026-07-29 (WPMINT close-out)

Verbatim from the operator's ADJUDICATION RESPONSE. These **lift the Phase D caveat**, ratify
`CARD-LINT-GATE`, and authorize the merge. Fidelity label on each is **[INLINE]**.

## R97 — Phase D caveat LIFTED [INLINE]

> R97 — Phase D caveat LIFTED. Grounds: mechanism clean under DR's battery + two consecutive
> fresh-context adversarial passes; residual failing class (rationale prose) enumerated at
> 286-claim coverage, fixed, re-verified 0/30 and 0/18 in the previously-failing sub-populations.
> Recorded explicitly: this is relocation-and-coverage of the failure surface, not attrition — no
> unchanged surface was rerun for a green word. R95's condition amended in place to its intent:
> mechanism clean under adversarial re-verify, residual classes enumerated and covered.
> Phase D: PASS, full.

*Closes a residual open since **R89** and carried through R92's amendment — the longest-lived
caveat of this lineage. Three things make the ruling more than a sign-off. **It amends R95's
condition "in place to its intent" rather than declaring the literal words met**: the condition
said "clean verdict" and no pass ever returned that word, so the operator restates what the
condition was *for* — mechanism clean under adversarial re-verify, residual classes enumerated
and covered — instead of pretending it was satisfied. **It names the failure mode it is NOT**:
"relocation-and-coverage … not attrition — no unchanged surface was rerun for a green word",
which is the precise thing the dispatcher refused to do across two passes and twice put to the
operator rather than resolve itself. And **it cites the coverage numbers as the grounds** (286
claims; 0/30 messages, 0/18 commit messages in the previously-failing sub-populations), so the
lift rests on a measurement rather than on fatigue. Phase D is **PASS, full** — the
PASS-WITH-DISCLOSURE label that R89 attached and R92 re-scoped is retired.*

## R98 — CARD-LINT-GATE ratified, owned by WP-R [INLINE]

> R98 — CARD-LINT-GATE RATIFIED (WP-R owns): curated rule set seeded from defect-catching rules
> (F601 class first), never the advisory backlog wholesale (CARD-LINT-TYPE's debt); no gate over
> a known-dirty baseline; mutation self-test mandatory. Rider: derive-or-delete for mechanical
> facts in rationale prose; gate-13 citation discipline where derivable.

*Ratifies the card the WPMINT dispatcher opened when ruff's **20 x F601** on the duplicated
registry block turned out to have been readable for four commits while no gate read it. The
constraints are the interesting half and each names a way a lint gate rots: **curated, seeded from
rules that CAUGHT a real defect here** (F601 first) rather than the backlog wholesale — the debt
`CARD-LINT-TYPE` already represents; **no gate over a known-dirty baseline**, because a gate that
starts red is a gate nobody can act on; and the **mutation self-test** every gate in this lineage
has had to pass. The **rider** generalises WPMINT's actual defect distribution: 17 of 17 residual
findings were mechanical facts stated in prose, so the standing answer is **derive-or-delete** —
a count or citation that can be derived should be derived (gate-13's pattern) and one that cannot
should not be asserted.*

## R99 — MERGE AUTHORIZED [INLINE]

> R99 — MERGE AUTHORIZED: 10-commit wpmint-scratch → dev, standard runbook (verify stack,
> quiet-box, double sweep, ledger/register appends, scratch deleted). The caveat ledger is empty;
> remaining pre-mint items are operator acts.

*R95's "MERGE AFTER DSV-2, not before" is discharged by R97. "The caveat ledger is empty" is the
load-bearing clause: no adjudication remains open against this stack — ADJ-18/19 (R92),
ADJ-20 (dissolved), ADJ-21 (R94), ADJ-22 (R95) are all closed, and the Phase D residual is lifted.
What remains before run5 mints is explicitly **operator acts**, not engineering.*

---


# R100–R101 — FIDELITY PATCH, appended 2026-07-29 (texts supplied retroactively by the operator)

**Fidelity note (dated 2026-07-29).** These two rulings were delivered in the WPBRIDGE-era
dispatch traffic and acted on (R100 by the WPTS push, R101 as WPBRIDGE's dispatch premise),
but their verbatim texts were never appended to this register by the sessions that received
them — a gap the WPTS close-out recorded under R107. The operator supplied both texts
verbatim on 2026-07-29; they are placed here in chronological position so the register
records both the texts and the gap they fill. Fidelity: **[INLINE]**.

## R100 — Push AUTHORIZED, proviso [INLINE]

> R100 — Push AUTHORIZED, proviso: origin is the private remote under
> operator control. R57 update: "remote exists" satisfied; "green Actions"
> remains a cutover-battery gate; Actions runs lint advisory-only per
> 4bc1c77 as intended.

## R101 — Correction on the record: one engineering item remained [INLINE]

> R101 — Correction on the record: one engineering item remained on the mint
> critical path — TD-4 / CARD-POOL-ENCODING-BRIDGE (mode PREFLIGHT cannot
> burst). WPBRIDGE dispatched: Phase T (fix, R64 posture law, production-path
> convergence, mutation-tested) + Phase R (dev-box rehearsal — a truthfully
> red rehearsal is a success; zero first-time code paths on the training
> box). After WPBRIDGE merges, the mint path is operator-only.
> [Superseded in part by R101's own correction history: TD-1 remained; see
> R102–R106. The mint path became operator-only at R107, by measured census.]

---


# R102–R105 — operator adjudication response, 2026-07-29 (WPBRIDGE)

Verbatim from the operator's ADJUDICATION RESPONSE to `wp/WPBRIDGE/ADJUDICATION_QUEUE.md`.
These close all three WPBRIDGE adjudications and close the WP itself. Fidelity: **[INLINE]**.

**Context these rulings answer.** WPBRIDGE landed TD-4 and then MEASURED that the dispatch's own
premise (R101 — *"the last engineering item on run5's critical path"*) was false: clearing the
pool/encoding bridge moved the boot forward without buying a completed burst, because
CARD-TRAINSTEP-ADAPTER (TD-1) is still live behind the warmup gate. R102–R105 route that finding
rather than absorbing it.

## R102 — ADJ-23: WP-TRAINSTEP, own WP, full pipeline [INLINE]

> R102 — ADJ-23: WP-TRAINSTEP, own WP, full pipeline (WPAX's QUEUE rationale governs — the
> learner's dispatch seam gets a DESIGN, not a follow-up grant). Scope: TD-1 through the typed
> representation route; scoped conformance gate on the coordinator↔trainer seam (mutation-tested;
> NOT the pyright backlog — that stays CARD-LINT-TYPE); behind-the-warmup-gate reachability
> census, fully adjudicated before IMPL.

*Takes WPBRIDGE's recommended option 1. The operative clause is **"a DESIGN, not a follow-up
grant"**: WPAX's ruling that landing the learner's missing half inside two loops is indefensible
is not weakened by the fact that TD-1 is now the frontier — it is the reason TD-1 gets its own
pipeline. Three scope constraints are new. (i) **"through the typed representation route"**
settles the dispatch question WPBRIDGE did not answer — `step.py:640`'s straight self-play arm
routes dense vs graph off the DECLARED representation, not off a buffer sniff, which is the same
LAW-11 posture WPBRIDGE just enforced one layer up. (ii) The **conformance gate is SCOPED** to the
coordinator↔trainer seam and must be mutation-tested (LAW-07); it is explicitly fenced off from
the pyright backlog so a type gate cannot grow into a lint WP. (iii) The **reachability census is
fully adjudicated BEFORE IMPL** — WPBRIDGE proved a CPU box cannot reach the warmup gate, so
WP-TRAINSTEP must settle how TD-1 will be OBSERVED before it writes the fix, not after.*

## R103 — ADJ-24 GRANTED: an armed smoke config is minted [INLINE]

> R103 — ADJ-24 GRANTED: smoke_preflight_armed.yaml minted via tooling (R1-legal),
> header-truth, live consumer = the second burst oracle; doubles as TD-1's end-to-end proof.

*WPBRIDGE measured that `_audit_paths` audits the NAMED config and every non-run5 config is
deliberately disarmed (R59), so run5 was the only config mode PREFLIGHT would burst — leaving no
cheap rehearsal target and one unreachable Phase-T oracle. The grant is narrow in the way that
matters: **minted via tooling**, so R1's "configs are minted, never hand-varied" holds and this
does not become a hand-armed second config; **header-truth**, so the file says what it is for; and
**a live consumer is named** (LAW-08) rather than the config being minted speculatively. The last
clause is the reason this is granted now instead of with WP-TRAINSTEP: a fast armed config is what
lets TD-1's fix be proven end-to-end at all, so it is infrastructure for R102, not a nicety.*

## R104 — ADJ-25: agreement-or-raise [INLINE]

> R104 — ADJ-25: agreement-or-raise. Disagreeing dual-shape configs are corrupt, not precedence
> questions. anchor.py + pretrain/validate.py converge onto the one resolver; LAW-11 arms
> re-pinned by oracle.

*Rejects the framing WPBRIDGE filed the card under. WPBRIDGE preserved flat-first precedence in
`resolve_from_config` because that is what the two deleted bridges did, and carded the
disagreement with `anchor.py` / `pretrain/validate.py` (both identity-first) as a latent
precedence split. The operator rules that **precedence is the wrong question**: a config declaring
an encoding in two shapes that DISAGREE is corrupt input, and the one authority must RAISE on it
rather than silently pick a winner — the same "refusing to silently pick a side" posture
`checkpoints.py:492` already takes. So the resolver keeps a single accept path, gains an
agreement check, and the two remaining private name-lifters converge onto it, finishing the
one-authority collapse WPBRIDGE started and scoped out. LAW-11's arms are re-pinned by oracle
because this narrows an accept-set, and a narrowing can lose a raise as easily as a widening can.*

## R105 — WPBRIDGE closes PARTIAL-RATIFIED; merge is authorized now [INLINE]

> R105 — WPBRIDGE closes PARTIAL-RATIFIED: Phase T full success, Phase R successful-by-definition
> (truthfully red). MERGE AUTHORIZED now, standard runbook, 1-commit stack. Good work does not
> wait on the next WP.

*Names the disposition precisely. **PARTIAL-RATIFIED**, not PASS-WITH-DISCLOSURE: Phase T is a
full success and Phase R met its own written success condition — the dispatch said *"a truthfully
red rehearsal is a success of this phase"*, and it was red truthfully, with every wall named and
nothing papered over. The partial is the WP's SCOPE (its dispatch asked for a completed burst that
TD-4 alone could never deliver), not its execution. **"Good work does not wait on the next WP"** is
the general clause: a stack that is gate-green and carries no open adjudication against itself
merges on its own merit, even when the finding it produced opens a larger WP. The three
adjudications are closed by R102–R104 and none of them is a finding AGAINST this stack — ADJ-23
and ADJ-24 are scope findings the WP itself raised, ADJ-25 is a latent card. Caveat ledger empty.*

---


# R106–R107 — operator adjudication response, 2026-07-29 (WPTS)

Verbatim from the operator's ADJUDICATION RESPONSE to `wp/WPTS/ADJUDICATION_QUEUE.md`. These
close the one queued WPTS adjudication and authorize the merge. Fidelity: **[INLINE]**.

## R106 — ADJ-26 RULED: CARD-PROTOCOL-COMPLETE, pre-cutover, NOT mint-blocking [INLINE]

> R106 — ADJ-26 RULED: CARD-PROTOCOL-COMPLETE, pre-cutover, NOT mint-blocking
> (called-and-undeclared with concretes present = type-system drift, zero
> runtime effect — distinct from TD-1's called-and-absent). Card: complete
> declarations against concretes; THEN widen the AST conformance gate to all
> *Like protocols (gate stays seam-scoped until the baseline is clean, per
> R98's no-lying-gate law). Design question, not mandate: a narrow read-side
> protocol for events.py vs growing WorkerPoolLike by nine. With this ruling
> the WPTS census is empty of unruled rows; the dispatch's SUCCESS clause is
> satisfied by adjudication.

## R107 — MERGE AUTHORIZED [INLINE]

> R107 — MERGE AUTHORIZED: 3-commit wpts-scratch → dev, standard runbook,
> then push per R100. Post-merge, run5's critical path is operator-only by
> measured census: box preflight → WP12-R decision → prereg → mint.

*Register note (fidelity): R100 and R101 were cited by dispatches but unregistered when R107
was received; the WPTS push acted on R107's explicit clause. DISCHARGED 2026-07-29 — the
operator supplied both texts verbatim and they now stand in chronological position above
(see the R100–R101 fidelity patch), R100's proviso consistent with the push as performed.*

---


# R108–R111 — operator adjudication response, 2026-07-29 (WPCLEAN close)

Verbatim from the operator's ADJUDICATION RESPONSE ("rulings R108(verbatim)–R111; R108
fills the FID gap"). The FID-R108 fidelity item is DISCHARGED: the [DISPATCH-DERIVED]
entry that stood here is replaced by the verbatim text, per its own note. R110 and R111
also arrived in a fuller prose form in the same dispatch; that prose is preserved in
`wp/WPCLEAN/DISPATCH_LOG.md`. Fidelity label on each is **[INLINE]**.

## R108 — WPCLEAN delegation; the R90 package renews [INLINE]

> R108 — WPCLEAN delegation: the R90 package renews unchanged. Phase-drop
> order when short: PC → LG/LT → residuals → R8 → NAME, dropping from the
> back. Fence: any deletion touching R20-protected dense surfaces (v6
> question included) is queue-with-recommendation, never auto-executed —
> that frame is operator-locked.

## R109 — HANDOFF-1 CLOSED; MIT confirmed [INLINE]

> R109 — HANDOFF-1 CLOSED: MIT confirmed (in place since 5bd1d70); the
> "hexo-mantis contributors" holder line is accepted as final unless the
> operator supplies a name (one-line commit riding the merge if so).

*No name was supplied with R111's merge dispatch; the default stands and no copyright
commit rides the merge.*

## R110 — queue dispositions [INLINE]

> R110 — Queue dispositions: CARD-PYRIGHT-STRICT accepted, post-cutover
> ratchet (one exclusion per commit, zero-error proof each). Q-NAME-1/2
> recommendations pre-approved iff naming-only or reachability-only AND
> v6w25/multicluster/dense infrastructure untouched; dense-surface deletion
> requires the operator's explicit line (R20 is operator-locked).
> Q-PFC-SPLIT + Q-PFC-R43: GROUND_PFC.md plan approved in stated order —
> flip-set pin regeneration first, then split + frozen hunk under S-3/R43 —
> as the opening phase of the next run.

## R111 — MERGE AUTHORIZED [INLINE]

> R111 — MERGE AUTHORIZED: 20-commit wpclean-scratch → dev, standard
> runbook, floor 2163, push per R100.

*Sequencing note, recorded with the rulings (operator, same dispatch): "run5 mints on
gnn_axis_v1; WP-AXIS2 lands post-mint; gnn_axis_v2 is run6's pre-registered baseline
change, measured against run5's Stage-0 evidence. v2 is never an in-run A/B."*

---


# R112 — host grant for the WPBOX run [INLINE]

**Fidelity note.** FID-R112 is DISCHARGED 2026-07-30: the operator supplied the verbatim
text below, replacing the [DISPATCH-DERIVED] reconstruction that stood here (cf. the
R100–R101 patch, FID-R108, for the same discharge pattern).

> R112 — Host access GRANTED for the WPBOX dispatch: this host only, via the operator's
> ssh alias; R31 satisfied by this explicit grant. Rule 7 absolute: provider name,
> alias, and host paths may appear in migration-workspace artifacts but NEVER in
> anything committed to hexo-mantis; bench provenance carries interpreter/numpy/rustc/
> CPU-model only. The wiped instance's freshly-pinned stack is THE prod baseline
> R18/R21 deferred to.

---


# R113–R117 — operator adjudication response, 2026-07-30 (WPBOX close)

Verbatim from the operator's ADJUDICATION RESPONSE closing WPBOX. These authorize the
merge, route the run5 GPU-OOM finding, open the run5 reachability frontier as its own WP,
rule on a batch of audit dispositions, and approve WP-LEAN-RENAME. Fidelity label on each
is **[INLINE]**.

## R113 — MERGE AUTHORIZED [INLINE]

> R113 — WPBOX merge authorized: 9-commit stack 45e6ee7…b482243, standard runbook, floor
> 2173, push (CB-4's green-Actions confirmation fires on it), scratch deleted. WPBOX
> closes ratified — the truthful OOM red, the first-ever GPU test executions, and the
> prereg-held battery are exactly what the box run existed to produce.

*DISCHARGED 2026-07-30: merged fast-forward onto `dev` (`aab268b..b482243`), full
cargo-workspace + pytest gate sweep green on the scratch tip pre-merge (2171 passed / 2
skipped, matching the 2173 floor), pushed to `origin/dev`, local `wpbox-scratch` deleted
(no remote copy existed).*

## R114 — CARD-RUN5-GPU-OOM routed to WP12-R [INLINE]

> R114 — CARD-RUN5-GPU-OOM lands in WP12-R (as the dispatcher already routed):
> diagnostic direction — trace the single ~12.8 GiB allocation event to its tensor
> provenance; prime suspects are an unbounded inference-server batch or a pathological
> graph exceeding the edge-cap's intended domain; the fix must be a capped, config-typed
> bound (rule 4), never a silent truncation (rule 5). Q-GAP-D-LATENCY and
> Q-GAP-C-EVAL-WALL ride its seam as recommended; Q-TRAIN-STEPS-FLOOR stays blocked
> behind it.

## R115 — CARD-RUN-MAIN opened; the run5 reachability frontier [INLINE]

> R115 — CARD-RUN-MAIN, new, mint-critical, own small WP, full pipeline. The audit's
> headline is the real frontier: every subsystem works in isolation and
> `python -m mantis.run` cannot start a run. The glue wires main() → build
> trainer/pool/buffer → compose_run → run_until_stopped, production posture, with the
> preflight boot child converging onto the same path (one composition authority — the
> preflight must boot what run5 boots, which it currently only approximates). I've been
> wrong twice about "last engineering item"; this one comes from a measured audit, and
> its WP still opens with a reachability re-census anyway.

## R116 — audit dispositions batch [INLINE]

> R116 — audit dispositions batch: dead TacticalSolver FFI export deleted (dead-weight
> law, rides any stack); glossary written as step 0 of WP-LEAN-RENAME; skeleton contract
> docs to the WP14 drift-gate's jurisdiction; honor-system laws (LAW-03/04/09/10/15) get
> a post-cutover enforcer card, the rest are process laws by nature; solver quiet-move
> body is Track B with F-35..40 as its mandate; unimplemented!() dense multi-window arms
> stay dormant-annotated (R20 frame).

## R117 — WP-LEAN-RENAME approved as post-mint work [INLINE]

> R117 — WP-LEAN-RENAME approved as post-mint work, combining the lean-4 migration with
> the tier-C registry renames per the audit's sequencing insight, under the audit's own
> constraints: LAW-12 forbids re-stamping (alias layer, not history rewrite), gate-3's
> floor handled by the ratchet-to-measured law, persist/load.rs's "v6" treated as the
> compat constant it is, and the R20 frame satisfied because v6_live2_ls lean-4 remains
> the dense control arm. Its DESIGN comes to you for the final deletion sign-off — that
> lock stays operator-held.


# R118–R121 — architect adjudication response, 2026-07-30 (WPMAIN Phase 0)

Verbatim. **R118 and R119 are a FIDELITY PATCH**: both were issued in chat when the WPMAIN
dispatch was authored and never delivered as register-ready text, so the dispatch cited two
rulings this register did not carry. WPMAIN's entry check caught it, halted before DESIGN,
and ran Phase 0 in a degraded mode — solo (no delegation, since the package is cited "per
R119") and with item 5's routing withheld (item 5 is "(R118 routing)") — under R115, which
mandates the re-census on its own. That is the FID-gap class R107 flagged; the miss is the
architect's. `wp/WPMAIN/ADJUDICATION_QUEUE.md` FID-R118 / FID-R119 close on this append.

**R120 and R121 are new**, ruling on the Phase 0 census: R120 resolves Q-EVALKEY, R121
takes the census's dispositions including its falsification of the audit's shape estimate.

Fidelity label on each is **[INLINE]**. Architect-authored per SESSION_HANDOFF_v2's role
clause; operator countersign at next review.

## R118 — WP11-A close-out RATIFIED in full [INLINE]

> R118 — WP11-A close-out RATIFIED in full. A-1 (wr_sealbot None until
> WP12-R): ratified; value populates in WP12-R Phase A. A-2 + A-3: ratified;
> run5 random-floor cadence and fresh-seed decision become named
> RUN5_MINT_PREREG rows. O-A..O-G: ratified — property strength preserved;
> O-G correct (protocol keyword name permitted, attribute reads banned).
> e2e option (b): ratified. Deviation #2 ratified; #3 revert ratified
> (fixture obeys the law); #5 document-the-unit ratified — elo_ci_lower_boot
> rename routes to WP-LEAN-RENAME. R-DRAIN-HARDCAP WIRE: ratified; R93's
> mutation-verified re-wiring confirmed it. Handoff rows: (a) encoding_spec
> blocker → WP12-R Phase B; (b) _DEFAULT_MAX_PLIES + run.py broad-except
> re-tightening + emit-coverage-of-eval-knobs → WPMAIN Phase-0 item 5,
> absorb-or-defer ruled in DESIGN; (c) F-RT2-2 → WP-R row; (d) eval_broken
> enum additions recorded. Discharged since issue: R23–R31 recording
> (register + ADDENDUM_A); run3_findings_v2 absence (arrived; R48/R54
> governed consumption).

## R119 — R90/R108 delegation renewed for WPMAIN + WP12-R [INLINE]

> R119 — The R90/R108 delegation package applies unchanged to the WPMAIN and
> WP12-R runs. HARD STOPS verbatim: new adjudication classes; any dev
> mutation; scope widening beyond carded scope; any change to run5's armed
> values (0.25 / 25000 / 50 — mint-prereg-only, R82/R85/R92).

## R120 — Q-EVALKEY: eval_enabled becomes a typed schema field [INLINE]

> R120 — eval_enabled is promoted to a typed RunConfig schema field. The
> code-side default True at the composition root dies (rule 1); every minted
> config carries the key explicitly (R1 completeness); live consumer = the
> one composition root, mutation-tested (LAW-08/LAW-07). R79 is not
> over-applied: eval_enabled is itself the fact, not a proxy beside a gating
> value. R64 unchanged — the preflight boots the config's own value and may
> never force False. run5 mints True (LAW-15: a promotion bar with eval off
> is unrepresentable as a decision). The dispatch's "expected ZERO new keys"
> is amended to exactly one: this key, with manifest impact stated in DESIGN.

## R121 — census dispositions: boot inversion accepted; run_until_stopped [INLINE]

> R121 — (a) The census's inversion is the design frame: lift _boot_main out
> of tools/ci_gates/preflight_mint.py into src/mantis (composition root
> territory), re-point BOTH callers onto it. A tool owning the only real boot
> is the one-authority violation this WP exists to end. The frozen-file exit
> (O-9/O-10/O-5 token census + O-1 parser census) is expected settled-class
> under R90a — the boot path IS the card's declared subject — auto-apply with
> S-3 records; any hunk that weakens an assertion rather than re-pointing it
> queues. (b) Defects 1 and 2 (dead signal handlers on every composed run;
> DiskGuard never constructed) are IN-SCOPE tree defects per R64 + the
> dispatch's LAW-16-at-the-root clause: the root installs handlers bound to
> the injected ShutdownState and constructs the disk guard, contract-tested
> at R64 posture (the census probe's faked build_run_safety is census-tier
> only; oracles fake nothing). (c) run_until_stopped: DESIGN rules
> wire-or-retire with grounds. Frame: compose_run → run_t is the live tested
> loop; name-truth (R73) + dead-weight law (R116) mean either
> run_until_stopped becomes the one loop or it is deleted, never documented
> around. Success criterion 1 is amended to "boots through the one composer
> into the live run loop, bounded, clean stop" — the chain in R115 named the
> audit's vocabulary, not a binding symbol. (d) No R20 surface is touched by
> any of a–c.


# R122–R124 — architect adjudication response, 2026-07-30 (WPMAIN DESIGN)

Verbatim. Rules the three open rows the WPMAIN DESIGN phase carried to adjudication:
Q-DISKGUARD-KEYS (granted as a family, amending R120's key budget), Q-RUNID-PARAM
(provisionally endorsed, with REVIEW-design charged), Q-MAXPLIES-DEFER (owner assigned).

Fidelity label on each is **[INLINE]**. Architect-authored per SESSION_HANDOFF_v2's role
clause; operator countersign at next review.

## R122 — Q-DISKGUARD-KEYS: granted as a FAMILY [INLINE]

> R122 — Disk-guard keys granted as ONE FAMILY under the R78/R80
> clarification pattern: one config block, one resolver, three typed leaves,
> minted at the dead literals 60/10/5 (the code's own stated intent; the
> behavior change is R121(b)'s mandated construction itself, not the values).
> R120's budget is amended to: one key (eval_enabled) + one family
> (disk_guard). No enable boolean — LAW-16 makes the guard always-constructed
> and R79 forbids a proxy beside gating values; off-semantics, if any leaf
> needs them, are explicit values, argued in DESIGN. Live consumers
> mutation-tested (LAW-08/LAW-07); manifest impact stated. Prereg note per
> R85 pattern: the three values are revisable at mint prereg once a real box
> boot observes guard behavior — the literals were dead, so nothing has ever
> measured them.

## R123 — Q-RUNID-PARAM: provisionally endorsed, REVIEW-design charged [INLINE]

> R123 — The run_id parameter deletion is provisionally endorsed on MF-1's
> own principle (no parameter carries a config fact) — same class as
> eval_enabled's removal, and it strengthens the R49-pattern property that
> forcing a divergent value is unrepresentable. REVIEW-design is CHARGED
> with three checks before this stands: (a) census where run_id is produced
> at HEAD and that it is genuinely a mint-time fact; (b) no LAW-12 stamp
> path breaks — filenames and stamps still carry run-id + content hash from
> the config-sourced value; (c) the deletion smuggles no default — an absent
> run_id RAISES named (R1/LAW-11 identity posture), never generates
> code-side. Any check fails → queue, deletion reverts to parameter pending
> adjudication.

## R124 — Q-MAXPLIES-DEFER: owner assigned [INLINE]

> R124 — _DEFAULT_MAX_PLIES schema promotion is CARD-MAXPLIES, owned by
> WP-R, pre-cutover, NOT mint-blocking. Grounds: code-side default present
> since WP11-A, no run5 dependency, and loading a mint-critical WP with it
> violates drop discipline. Not CARD-COORD-KNOBS — different seam (eval, not
> coordinator); that card's scope stays as R78/R80 fixed it.


# R125–R127 — architect adjudication response, 2026-07-30 (WPMAIN DESIGN loop-1 close)

Verbatim. Rules the three rows the DESIGN fix pass carried out of loop 1: the queued
weaken-class hunk (R125), the device-authority hole REVIEW-design surfaced (R126, which
adds the third and final item to the key budget and kills the `--device` flag), and the
strip-path placeholder countersign (R127).

Fidelity label on each is **[INLINE]**. Architect-authored per SESSION_HANDOFF_v2's role
clause; operator countersign at next review.

## R125 — Q-MF4-RC-PREDICATE: argued-deletion ADOPTED [INLINE]

> R125 — The except arm dies; MF-4's rc assertion goes with it per the
> pre-queued weaken hunk. Grounds: the arm is unreachable through the child
> CLI (Literal["grid","graph"] at schema/core.py:51 + registry cross-check;
> the drive reaches it only via SimpleNamespace) — keeping it alive to feed
> a test is R116 dead weight and the self-satisfying-test species this
> lineage kills on sight. The mapping alternative is REJECTED as recorded.
> Post-deletion posture: a RepresentationRouteError, if the closed enum is
> ever widened, propagates as an uncaught loud failure — that is LAW-14
> fail-loud, not a silent arm. Rider: LAW-11 means widening the
> representation enum is a deliberate design act; whoever widens it re-opens
> child-seam routing in that same design. S-3 record carries the
> unreachability measurement verbatim.

## R126 — Q-DEVICE-AUTHORITY: device is a CONFIG FACT; the flag dies [INLINE]

> R126 — Train device is promoted to a typed schema field: closed
> Literal["cpu","cuda"], required, no default, absent = named raise
> (R1/LAW-11 posture). The --device CLI flag dies on BOTH callers — MF-1's
> own principle; no parameter carries a config fact. Grounds: (a) the
> CLI-only flag leaves a posture-divergence hole at exactly the wall that
> killed the WPBOX burst — a cpu preflight against a cuda run5 false-clears
> the GPU memory wall, the LAW-03 instrument-that-cannot-false-clear
> corollary; (b) R64/R61: preflight boots the config's own value, production
> axes get production checks; (c) eval.worker_device already establishes
> device as config-domain. NOT a duplicate authority: train device and
> eval.worker_device are different facts (split topology is legitimate);
> REVIEW-design does not flag it as R79 class. DESIGN states the amp
> interaction: graph+cuda bf16 law untouched, cpu autocast semantics named.
> R122's budget amends to: eval_enabled + disk_guard family + device. All
> configs re-mint mechanically; per-config device values are mint decisions.
> Every prior artifact naming --device in the entry surface gets the R96
> correction pass.

## R127 — N-STRIPRESTAMP-PLACEHOLDER-R1: COUNTERSIGNED [INLINE]

> R127 — Countersigned as proposed: the strip path's synthetic config gains
> eval_enabled: True + disk_guard 60/10/5 (+ device per R126) at minted
> values, existing placeholder framing, stamp path untouched (LAW-12). No
> third default authority is created — stripped artifacts never boot runs.
> IMPL latitude: derive the values from schema defaults if mechanical;
> else literals + a comment citing R127.


# R128–R131 — architect adjudication response, 2026-07-30 (WPMAIN IMPL close)

Verbatim. Rules the four items IMPL carried out: the census false-negative and its R50-list
consequence (R128, which also lays down a standing census law), the unsatisfiable frozen
oracle (R129, an R43 grant + a new card), the CUDA-boot drives (R130), and the teardown
deviation (R131).

**Dispatcher note recorded at append time, R129's own trigger:** R129's grounds state "run5
mints checkpoint_interval > 0 so tail exposure is interval-bounded" and instruct that "run5
minting 0 escalates the card to mint-blocking on the spot." **Measured: `configs/run5.yaml:108`
mints `train.checkpoint_interval: 0`** (as do all six configs), and `trainer/core.py:487-489`
guards the periodic save with `if interval > 0`. **CARD-CLEANSTOP-SAVE therefore escalates to
MINT-BLOCKING by the ruling's own terms.** See `wp/WPMAIN/DISPATCH_LOG.md` for the full
save-path enumeration.

Fidelity label on each is **[INLINE]**. Architect-authored per SESSION_HANDOFF_v2's role
clause; operator countersign at next review.

## R128 — census correction + R50-list amendment; subprocess-shape census law [INLINE]

> R128 — CENSUS.md §4's false negative is corrected per R96 in every
> downstream artifact (DESIGN R50 list included). R50 list AMENDED,
> settled-class: tests/tools/test_preflight_mint_process.py:1586/:1596 added
> with disposition rewrite-to-pin-the-new-boot-law; the three CUDA-boot
> drives added with disposition per R130. IMPL's stop at the R50 bar was
> correct behavior — the bar worked; the list was wrong. STANDING LAW:
> reachability censuses over entry-point surfaces must cover subprocess /
> `-m` / console-script invocation shapes, never import shapes only — an
> entry point's consumers are by nature invisible to import greps.

## R129 — N-IMPL-OB1: frozen oracle re-pointed to truth; CARD-CLEANSTOP-SAVE [INLINE]

> R129 — R43 grant: O-B1 is re-pointed from "final checkpoint written" to
> the measured truth — clean bounded stop, rc 0, O2 arm sets running=False,
> checkpoints/ EMPTY because checkpoint_interval: 0, clean-vs-aborted
> distinction asserted intact. DESIGN §9's premise was false; the oracle
> asserts the positive truth, never goes vacuous. S-3 discipline + R81/R86
> mutation condition (not self-satisfying, no unrelated casualty).
> CARD-CLEANSTOP-SAVE opened: whether clean completion (stop_step reached)
> triggers a final save is a deliberate lifecycle design decision, owned
> pre-cutover, NOT mint-blocking — grounds: run5 mints checkpoint_interval
> > 0 so tail exposure is interval-bounded. VERIFY that value at mint
> prereg; run5 minting 0 escalates the card to mint-blocking on the spot.

## R130 — CUDA-boot drives: the bill converts to evidence [INLINE]

> R130 — The three rc-40 drives re-point to a minted CPU-device config
> preserving the rc-40 property (minted via tooling, R103 pattern — never
> hand-varied). run5.yaml's local boot evidence moves to the box preflight,
> where it belongs. NEW positive oracle required in the same pass: booting
> run5.yaml on a non-CUDA box fails LOUD in init_trainer, parent rc 33 —
> pinning R126 grounds (a) as a permanent regression oracle: the device
> false-clear is dead by construction and stays dead. rc 33 must trace to a
> named failure, not swallowed; if the raise is raw torch, that is
> acceptable-loud, recorded, not re-wrapped in-scope.

## R131 — N-IMPL-DESIGN8-SINKCLOSE countersigned; protocol debt routed [INLINE]

> R131 — Deviation COUNTERSIGNED as disclosed: §8's contract met, O-D2 pins
> it. The forcing cause is named as debt, not accepted as shape: seven
> off-list suites stand in SimpleNamespace sinks lacking stop()/close() —
> CARD-PROTOCOL-COMPLETE (R106) gains a row: complete the sink/watchdog
> protocol against concretes, then lift the arm restriction so teardown runs
> unconditionally. Production code contorting around under-implemented test
> fakes is the tail wagging the dog; it ends when that row closes.


# R132–R135 — architect adjudication response, 2026-07-31 (WPMAIN RED-TEAM close + merge)

Verbatim. Rules the two reserved rc-taxonomy findings (R132 disk-guard, R133 eval/GnnNet),
grants the frozen-header correction (R134), and authorizes the merge conditionally (R135).

Fidelity label on each is **[INLINE]**. Architect-authored per SESSION_HANDOFF_v2's role
clause; operator countersign at next review.

## R132 — Q-RT-DISKGUARD-RC0: bounded fix pass authorized, pre-merge [INLINE]

> R132 — A disk-guard abort returning rc 0 is R44-class evidence (a green
> that lies) on WPMAIN's own new subsystem; it does not merge as-is. One
> bounded fix pass, WP11-A F-RT2-1 precedent: disk-guard abort joins the
> fail-fast family with a registered exit code, wired through
> exit_code_for_abort (the one resolver WPMAIN just built), contract doc
> same commit (R9), manifest row, mutation test proving a fired guard is
> supervisor-distinguishable from a clean run — R84's template verbatim.
> This is a new abort CODE, not an armed-value change; the R119 hard stop
> is untouched. Targeted re-probe of that path only (RED-TEAM-3 pattern),
> then merge under R135.

## R133 — GnnNet call site: dedup into WP12-R Phase B; class-widened; rc rider [INLINE]

> R133 — CARD-GNNNET-NO-FORWARD DEDUPES into WP12-R Phase B (the WP11-A
> run5-mint blocker; same defect, second call site). Phase B's scope is
> WIDENED per R71 class-fix law: the class is "eval-side model invocation
> assumes dense" — census EVERY such call site (worker seam + inference_local
> + any sibling), route all through representation dispatch (LAW-11 closed
> enum), flip-set covers the class boundary, not the two demo sites.
> Rider, same phase: terminal-eval failure joins the rc taxonomy —
> eval_broken on the TERMINAL round is a registered nonzero rc (degraded
> completion; LAW-15: no promotion decision = deliverable incomplete);
> mid-run eval_broken stays non-fatal (rounds recur; persistent breakage is
> the watchdog's jurisdiction). Same R84 template as R132 — one taxonomy,
> two legs, two owners. Until it lands: WPMAIN's rc-0 boot evidence carries
> an R70-style caveat in the ledger row — "rc 0 does not certify eval
> health" — discharged by Phase B; the preflight's event-stream assertions
> are the interim instrument.

**MERGE APPEND (R228, 2026-08-04):** R133 caveat DISCHARGED — both halves, mechanism named
(R69). Cause half: Phase B (`29f304b`) — producer test `test_graph_round_encoding.py`; a
graph eval round now runs green in-tree. rc-taxonomy half: Phase O (`d0957f1`) — broken
terminal round exits **48** (7 `test_a_broken_terminal_round_exits_48[*]` pins, M-O8
two-sided, no-over-fire proven at `test_run_launcher.py:300`). Ratified **R167**
(`rulings_register.md:2429-2431`). Q-EVALBROKEN-RC0 (the rc-taxonomy row) also CLOSED by
R152's one-taxonomy landing.

## R134 — Q-R8-DISKGUARD-FROZEN-HEADER: granted [INLINE]

> R134 — R43 grant, settled-class (N-3 precedent: stale prose, no assertion
> depends): the one-line header correction rides the R132 fix-pass commit
> under S-3 discipline.

## R135 — WPMAIN MERGE AUTHORIZED, conditional [INLINE]

> R135 — Merge authorized on R132's fix pass + targeted re-probe green:
> 6-commit wpmain-scratch → dev, standard runbook (verify stack, quiet box,
> double sweep with RAM+disk, R42 timestamps, floor to measured, ledger +
> register appends, scratch deleted, push per R100). The R133 caveat rides
> the ledger row verbatim. WP12-R's entry gate then reads the new dev sha.

---


## Superseded

- **R33** (entropy knob as `{entropy_mode, entropy_value}` with `-0.005` minted) —
  SUPERSEDED IN FULL by R37. Its premise was falsified by Phase 0 target T-C; see
  `wp/WPSC/ADJUDICATION_QUEUE.md` ADJ-01 for the evidence.


# WPMAIN CLOSE — execution record, 2026-07-31 (appended by the dispatcher under R135)

R135's runbook executed in full. **dev = origin/dev = `49e9efa`** (from `b482243`), 7-commit
ff-merge, zero merge commits in range, `wpmain-scratch` deleted, pushed per R100, floor
2173 → 2297. Double serialized sweep GREEN and identical (2295 passed / 2 skipped both
runs), all 10 gate scripts + cargo + wasm rc 0, box state recorded before each sweep, R42
timestamps (07:13–07:29 +02:00) precede the merge.

**Disclosed deviation from R135's text:** the stack is **7 commits, not the 6 the ruling
names**. The R132 fix pass carried its own floor-ratchet chore commit, per the house
one-chunk-one-commit convention. Substance unchanged; disclosed rather than reshaped,
because rewriting history on a branch about to ff-merge would trade a true count for
altered shas (R52 exists to preserve them).

**R132 verified live by the dispatcher, not by oracle:** a rigged-threshold launch
(`fail_gb` above free space, nothing else varied) exits **47**; the unrigged control exits
**0**. A fired disk guard is now supervisor-distinguishable, which is R132's whole
deliverable.

**R133's caveat rides the ledger row verbatim** — "rc 0 does not certify eval health" —
and WP12-R Phase B discharges it.

**R129's escalation stands at close: CARD-CLEANSTOP-SAVE is MINT-BLOCKING.** Confirmed live
rather than inferred — after a clean 200-step completion `checkpoints/` held zero files.
run5 mints `train.checkpoint_interval: 0`, `trainer/core.py` guards the periodic save with
`interval > 0`, and the O2 clean-stop arm returns without saving, so a clean 25000-step run5
would write no checkpoint at all.


# R142–R146 — dispatcher-recorded WP12-R entry rulings (RENUMBERED from R136–R140)

**NUMBERING COLLISION, resolved 2026-07-31.** These five were appended by the WP12-R
dispatcher as `[DISPATCH-RECORDED]` operator selections at the WP12-R entry gate, taking
R136–R140. The operator subsequently issued their own **R136–R141** (WPMAIN close,
CARD-CLEANSTOP-SAVE, eval-decode Option A, opponent pinning, lean-4 sequencing, GAP-CENSUS)
for entirely different subjects. The operator's numbering is AUTHORITATIVE per
SESSION_HANDOFF_v2's document-authority order item 1, so the dispatcher's block is renumbered
here to R142–R146. No content is lost or altered; only the numbers move. Every downstream
artifact citing the old numbers was corrected in the same pass (DISPATCH_LOG.md,
ADJUDICATION_QUEUE.md, project memory) — the correction-propagation rule: withdraw or
renumber in the ARTIFACTS agents read, never only in session.

Cross-reference for anyone reading older text: dispatcher R136→**R142**, R137→**R143**,
R138→**R144**, R139→**R145**, R140→**R146**.

Operator rulings issued in-session at the WP12-R entry gate, answering the dispatcher's
entry questions (host grant, ladder pin source, CARD-GNNNET-NO-FORWARD routing) and the
follow-up pin-selection question after the dispatcher researched the SealBot upstream at
the operator's direction. Recorded per SESSION_HANDOFF_v2's document-authority order
item 1. Fidelity label **[DISPATCH-RECORDED]**: the operator selected among options the
dispatcher posed rather than supplying prose; operator words are quoted verbatim where
given. Operator countersign at next review.

## R142 — host grant for the WP12-R run (Phase D) [DISPATCH-RECORDED] *(was R136)*

> R136 — Host access GRANTED for the WP12-R dispatch, same box as WPBOX: the grant and
> mechanism of R112 carry over unchanged (this host only, via the operator's ssh alias).
> R31 is satisfied by this explicit per-dispatch grant, NOT by R112's precedent — R112
> was "this host only" and does not travel. Rule 7 remains ABSOLUTE: provider name,
> alias and host paths may appear in migration-workspace artifacts but NEVER in anything
> committed to hexo-mantis; bench provenance carries interpreter/numpy/rustc/CPU+GPU
> model only.

Phase D is UNBLOCKED. Recorded here BEFORE any ssh, per the dispatch's ordering clause.

## R143 — Phase A ladder: sealbot only, the rest authorized-skip [DISPATCH-RECORDED] *(was R137)*

**SUPERSEDED IN PART by the operator's R139**, which ratifies the same decision and adds the per-rung grounds (krakenbot: weights not cleanly accessible; strix-bot: actively changing) plus the sha-not-branch rider. Read R139 as the operative text; this row records the entry-gate decision that preceded it.

Operator, verbatim: *"for now look for sealbot to pin and skip the rest (op authorized)"*.

> R137 — Phase A resolves the SEALBOT rungs only. The dispatcher pins a sealbot source
> via vendor/pins.toml; the remaining four rungs (`kraken_raw`, `kraken_mcts200`,
> `strix_128`, `strix_256`) stay loud-skip with grounds recorded per rung —
> OPERATOR-AUTHORIZED, not a dispatcher shortfall. A-1 (`wr_sealbot` populates) is in
> scope because the sealbot rungs are.

Ladder target for run5: **2/6 rungs live** (`sealbot_d5`, `sealbot_d6`), 4/6
authorized-skip. EVAL_DECISION.md records the promotion bar on that basis, and must state
plainly that the ladder's Bradley-Terry fit rests on two rungs of one engine family.

## R144 — CARD-GNNNET-NO-FORWARD absorbed into Phase B as a symptom [DISPATCH-RECORDED] *(was R138)*

**Note the collision hazard:** the operator's **R138** is the eval-decode Option A adoption, a different subject entirely. Any older text reading "R138" in a GnnNet context means this row, R144.

> R138 — CARD-GNNNET-NO-FORWARD is NOT an independent defect and does not become its own
> card. The dispatcher's entry census established one chain, one root cause:
> `eval/worker.py:78,193` never thread `encoding_spec` → the engine binds the dense `v6`
> default (`inference_local.py:70-71`) → `_is_graph` is False → the dense arm calls
> `self.model(...)` (`:179`) → `GnnNet` has no `forward`. The card is absorbed into
> Phase B's declared subject and closes when Phase B's graph eval round runs green.
> **No `forward` is added to `GnnNet`** — `tests/selfplay/test_arm8_reachable_paths.py`
> :105-113 states in-tree that a `GnnNet.forward` would convert this loud failure into
> silent dense-shaped output from a graph net, and is itself R56's escalation trigger.
> No scope widening under R119: the card is a symptom on the chain Phase B already owns.

Discharges R133's routing and carries R133's caveat forward verbatim — *"rc 0 does not
certify eval health"* — which Phase B's producer test is the instrument to discharge.

## R145 — the SealBot pin is ramora0/SealBot master [DISPATCH-RECORDED] *(was R139)*

**CONFIRMED by the operator's R139**, which independently selects SealBot std (ramora0) and adds the binding rider that a pin is a COMMIT SHA, never a branch name — so this row's sha `c94749c…` is recorded as "sha c94749c21c16c3b072fff6da49762dd5f92f3986, master as of 2026-03-31". WP12-R Phase A verifies.

Issued after the operator directed the dispatcher to research the upstream online
("not sealbot perf do ramora0 sealbot", "check for latest version").

> R139 — The run5 ladder anchor is pinned to `https://github.com/ramora0/SealBot.git`
> at sha `c94749c21c16c3b072fff6da49762dd5f92f3986` (branch `master`, 2026-03-31) —
> the stable public default branch, NOT the newer experiment branches and NOT the
> private perf fork.

Grounds recorded with the ruling, all measured rather than assumed:

- **Lineage verified**: the local working copy carries an `upstream-master` branch whose
  tip is exactly `c94749c`, and `git merge-base --is-ancestor c94749c HEAD` returns true.
  The pinned commit is genuinely upstream of the operator's own perf work, not a
  divergent bot.
- **Rejected alternatives, with grounds**: `mixnet-repro` (`ef40a22`, 2026-07-16) and
  `nnue-eval` (`6892e5e`, 2026-07-15) are newer by date but are EXPERIMENT branches —
  anchoring a promotion bar to a branch that can move or be deleted is what LAW-10's
  anchor-matched requirement exists to prevent, and an NNUE eval would need weights
  `vendor/pins.toml` may not carry (repo_design §7, no loose weights).
  `[REDACTED:local:rule7_local_terms.txt:25:292d8813]/SealBot_perf` is the perf fork the local tree tracks; its master `88dd5739`
  is public but is not the upstream lineage, and the local tip `1be0ed03`
  (2026-06-02) is **unpushed** — an unreachable sha cannot be pinned at all.
- **LAW-15 representable at the pin**: upstream master exposes `max_depth` as a settable
  property (`current/minimax_bot.cpp:97`), so the `sealbot_d5`/`sealbot_d6` rungs get a
  REPRODUCIBLE fixed-depth bar. The constructor's only argument is `time_limit`
  (wall-clock, non-reproducible); the adapter MUST drive `max_depth` and neutralize the
  time cut, and DESIGN owns proving that the time limit cannot truncate below the
  configured depth.

Two consequences DESIGN must carry, both measured at entry:

1. **A vendor build step exists.** SealBot is C++/pybind11; the shipped artifacts are
   `cpython-314` while mantis runs 3.13.11, so the extension must be rebuilt from source.
   `make vendor` clones and patches — it does not build. The adapter must loud-skip when
   the extension is absent and name the build step (the existing vendor law: "features
   that need an unfetched vendor must skip loudly and name `make vendor`").
2. **The adapter's substance is a translation**, not a wrapper: `get_move(game)` consumes
   SealBot's own `HexGame`, not a mantis `Board`. Coordinate mapping, turn parity, and
   the 1-stone-first-turn / 2-stone-compound rule are the correctness surface, and the
   oracle inventory must pin them.

## R146 — ADJ-WP12R-5 measured BEFORE Phase A [DISPATCH-RECORDED] *(was R140)*

**DISCHARGED 2026-07-31**: the measurement ran and returned DEPLOY-MATCHING BROKEN; the operator ruled on it as **R138** (Option A adopted). The sequencing this row mandated is complete.

Operator sequencing ruling, WP12-R post-RED-TEAM, 2026-07-31.

> R140 — ADJ-WP12R-5 is MEASURED before Phase A opens. The deploy-matching
> question is upstream of the eval decision, not parallel to it: Phase A's
> EVAL_DECISION.md must state run5's promotion bar, and LAW-15 binds that bar
> to deploy-matched eval, so a bar written while the deploy-matching property
> is unresolved would be written on an unverified premise.

Grounds recorded with the ruling: RED-TEAM measured the eval decode path dropping a mean 609
legal moves per leaf on 840/844 leaves for `gnn_axis_v1` — run5's own encoding — against
`policy_logit_count = 362` on an unbounded board, while the registry row declares "per-legal-
node policy, no off-window drop". The open half is whether SELF-PLAY retains what eval drops;
if it does, the two paths do not share an action space and LAW-15 deploy-matching fails for
run5's encoding.

The measurement is record-only and explicitly NOT authorized to implement any remedy — it
returns a verdict plus costed options for the operator (R114's diagnose-before-fix pattern,
applied to an eval-semantics question rather than a perf one).

Phase order for the remainder of WP12-R becomes: **ADJ-5 measurement → A → D(+riders)**,
amending the dispatch's B → C → A → D. Recorded under R90d (phase reordering with grounds).


# R136–R141 — operator adjudication response, 2026-07-31 (WPMAIN close + WP12-R eval decode)

Verbatim from the operator. These ratify the WPMAIN close, escalate CARD-CLEANSTOP-SAVE,
ADOPT Option A on the eval decode (the WP12-R Phase-B/ADJ-5 measurement's recommendation),
record the opponent-pinning decision, hold WP-LEAN-RENAME's post-mint sequencing while
authorizing find-only prep, and authorize a gap census. Fidelity label on each: **[INLINE]**.

**Numbering note:** the WP12-R dispatcher had already taken R136–R140 for entry-gate rulings.
The operator's numbering is authoritative; the dispatcher's block was renumbered to R142–R146
in the same pass. See the R142–R146 header for the cross-reference.

## R136 — WPMAIN CLOSED, ratified in full [INLINE]

> R136 — WPMAIN close ratified: 7-commit deviation RATIFIED — a true count
> with stable shas beats a squash matching R135's stale text; that is R52's
> point. RT-1's property-level fix + the repo_design:101 producer noted as
> debt paid. The R133 caveat rides the ledger row until Phase B discharges.

*Phase B discharged the R133 caveat on 2026-07-31 (commits `29f304b` + `e950f6d` on
`wp12r-scratch`) — for its cause. `Q-EVALBROKEN-RC0` remains open for the rc-taxonomy defect.*

## R137 — CARD-CLEANSTOP-SAVE: mint-blocking, both legs [INLINE]

> R137 — R129's trigger fired on its own terms (run5 mints
> checkpoint_interval: 0; observed empty checkpoints/ on clean completion).
> Escalated MINT-BLOCKING. Own small card, full-but-light pipeline (Phase D
> pattern), pre-WP12-R-merge or riding its stack: (a) clean completion
> (stop_step reached) triggers a final save as its OWN semantic — a third
> taxonomy leg beside periodic and shutdown_save; clean-vs-aborted
> distinction untouched; LAW-12 stamp path; mutation test: clean 200-step
> run ends with exactly one final checkpoint. (b) RUN5_MINT_PREREG gains a
> checkpoint_interval row — 0 on a 25000-step run is not a legal production
> posture; the minted value is a prereg decision with grounds.

## R138 — eval decode: Option A ADOPTED; deploy-matching restored [INLINE]

> R138 — Option A: eval decode consumes the overflow the producer already
> returns; self-play semantics is THE authority (three independent sources
> agree; registry text right, eval decode wrong). Mint-blocking, lands in
> WP12-R on the eval seam. R71 class named: "eval-side consumption of the
> shared producer diverges from self-play consumption" — census EVERY
> divergence site between the two consumers (decode, sort_prior, child cap,
> any sibling), not the one demo line; flip-set covers the class boundary.
> Oracles: (i) parity — identical child sets, eval vs self-play, same
> positions, both sides of the FFI bound on one fixture (upgrades the
> in-tree-Rust-test caveat); (ii) the dispersed-position regression
> (radius-6, >361 legal) pinned within R7 fixture ceilings; (iii) ladder
> asymmetry dead: head samples the full legal set against RandomBot.
> Riders: Q-GAP-C-EVAL-WALL re-measured post-fix (eval cost will move);
> LAW-09 bench on the touched path; strength delta measured at box
> preflight, recorded to prereg (LAW-01), never gating the fix. Evidence
> statement: no historical promotion decision is contaminated — graph eval
> produced no pre-WPMAIN evidence, because it could not run.

**MERGE APPEND (R228, 2026-08-04):** R138 evidence statement DISCHARGED — mechanism:
`6de393c` + the cross-FFI parity and dispersed-regression oracles. No historical promotion
decision is contaminated: graph eval produced no pre-WPMAIN evidence because it could not
run. Riders remain open and non-gating: Q-GAP-C-EVAL-WALL re-measure; LAW-09 bench on the
touched path; strength delta measured at box preflight, recorded to prereg (LAW-01), never
gating the fix. Text authored at `PHASE_Q.md:291-293`.

## R139 — opponent pinning: operator decision recorded [INLINE]

> R139 — Ladder for run5: SealBot std (ramora0) + RandomBot floor; krakenbot
> SKIPPED (weights not cleanly accessible), strix-bot repo SKIPPED (actively
> changing) — loud-skip with these grounds per rung. Rider: a pin is a
> COMMIT SHA in vendor/pins.toml, never a branch name — "master" is recorded
> as "sha X, master as of date"; WP12-R Phase A verifies. CARD-SEALBOT-
> BRANCHES opened, deferred, NOT mint-relevant: evaluate ramora0 branches
> (nnue notably stronger) as a higher rung once SealBot std is consistently
> beaten in a real run; sha-pinned version decision then.

## R140 — lean-4 + renames: R117 sequencing HOLDS; prep runs parallel [INLINE]

> R140 — WP-LEAN-RENAME stays POST-MINT per R117. Grounds: run5 mints on
> gnn_axis_v1 — the dense path is not on the mint's critical path; pulling
> registry/encoding renames + dense-arch deletion forward collides with
> WP12-R mid-flight on the same seams and adds churn to frozen parity
> surfaces for zero mint value. AUTHORIZED NOW, find-only, zero tree
> mutation, parallel: (a) glossary (R117 step 0); (b) evidence census for
> the 18-plane retirement — assemble F-14/F-36 + the hexo_rl reasoning into
> the disposition doc DRAFT; the doc states the falsification record and
> mechanisms, never "worse/overengineered" as a vibe (R69/R98 derive-or-
> delete); (c) census what the current dense control arm actually is —
> "lean-4" is confirmed against the tree, not assumed. Deletion list comes
> back for your explicit sign-off line (R110/R117 — that lock stays yours).

## R141 — GAP-CENSUS authorized [INLINE]

> R141 — One find-only census dispatch, cheap tier, against HEAD 49e9efa +
> STATE §2/§3 + every open card and prereg row: produce MISSING.md — what
> stands between HEAD and mint, and between mint and cutover, each row with
> owner + blocking status. No proposals, no tree mutation.

## R147 — M-15: run5's RandomBot floor is ARMED; the config is wrong [DISPATCH-RECORDED]

Operator ruling, 2026-07-31, on the GAP-CENSUS (R141) finding M-15.

> R147 — run5's `eval.random_floor_games: 0` is WRONG. The RandomBot floor should be
> armed. The `4 -> 0` mint delta is superseded by R139, which names the RandomBot floor
> as part of run5's ladder.

Measured state at the time of ruling (`configs/run5.yaml`): `random_floor_games: 0` at `:20`,
carried as an explicit mint delta `# delta: eval.random_floor_games: 4 -> 0` at `:5`;
`random_model_sims: 96` at `:16` is already live.

**Consequences recorded, none executed** — `random_floor_games` is an armed value and
therefore MINT-PREREG ONLY (R119 hard stop; the dispatcher does not touch it):

1. The re-mint of `run5.yaml` with a nonzero `random_floor_games` becomes a named
   RUN5_MINT_PREREG row with grounds. The VALUE is the operator's at mint.
2. R138's third oracle ("ladder asymmetry dead: head samples the full legal set against
   RandomBot") now has a production subject in run5. The eval-decode DESIGN had recorded
   run5's exposure as sealbot-only on the strength of the `0`; that note is superseded.
3. `EVAL_DECISION.md` (Phase A) states a bar that includes the random floor.


# R148–R150 — operator adjudication response, 2026-07-31 (WP12-R second half)

Verbatim. These resolve the control-arm dispute, fix the second-half phase order, and set the
conditional merge bar. Fidelity label on each: **[INLINE]**.

## R148 — ADJ-11 RESOLVED: dense control arm = v6_live2_ls [INLINE]

> R148 — Operator confirms: the legacy production dense lineage is
> v6_live2_ls; it IS the R20 matched-FLOP control arm, consistent with R117
> ("v6_live2_ls lean-4 remains the dense control arm"). EVAL_DECISION and
> the R140(c) census consume this as ruled fact. Any tree evidence
> contradicting it → queue, never silent (R121 pattern).

*ADJ-WP12R-11 CLOSED as a dispute. Its measured tree evidence does not vanish — it REQUEUES
under this ruling's own clause as **ADJ-WP12R-18**, because the contradicting evidence is a
COMMITTED oracle this WP wrote (`tests/eval/test_graph_round_encoding.py:18,161,165,182`, all
four naming `v6`). Not silently re-pointed. The dispatcher's earlier report to the operator
that the tree contradicted R117 was accurate as measurement and is superseded as
interpretation: R148 rules the identification, the tree evidence becomes a naming defect to
correct under a grant rather than a fact in dispute.*

## R149 — second-half sequencing; CLEANSTOP rides the stack [INLINE]

> R149 — Order: Phase Q (queue clearance) → Phase A; Phase D parallel at
> will (independent surface, R90e; host per R142). CARD-CLEANSTOP-SAVE
> rides wp12r-scratch as its own phase per R137's "or riding its stack" —
> one merge event, one runbook. Unruled mint-relevant queue rows return to
> the architect in ONE batch, not serially.
>
> **Amendment, operator, same session:** Phase Q is NOT run by the WP12-R
> dispatcher in-session. The operator launches a SEPARATE DISPATCHER with
> CLEAN CONTEXT for it. The incumbent dispatcher's duty is to leave a
> self-contained handoff, not to execute Q.

*Recorded by the incumbent dispatcher on the operator's instruction. Grounds are sound and
worth stating: the WP12-R session that produced these 18 rows also produced the reasoning
behind them, and a queue-clearance pass benefits from a reader who has NOT already concluded
what each row means. Six claims in this WP were falsified by execution, several of them the
incumbent dispatcher's own; a clean-context triage is the same discipline as a fresh-context
REVIEW, applied to the ledger instead of the code. Handoff artifact:
`wp/WP12R/PHASE_Q_HANDOFF.md`.*

## R150 — WP12-R MERGE AUTHORIZED, conditional [INLINE]

> R150 — Merge authorized when ALL hold: Phases Q/A/CS/D green (D's success
> per R114+R138 riders: OOM class dead, burst past the death envelope,
> riders measured); adjudication queue EMPTY or every residual row
> explicitly non-mint with grounds; RUN5_MINT_PREREG.md holds every blocking
> row with owner + status; R133/R138 caveats discharged in the ledger.
> Standard runbook, floor to measured, push per R100. After merge the mint
> path is operator-only: box preflight both tiers → prereg → mint.


# R151–R154 — operator adjudication response, 2026-07-31 (WP12-R Phase Q architect batch)

Verbatim from the operator, answering Phase Q's four-row batch (R149's one-block rule).
These unblock Phase A, merge the observability question with the rc taxonomy, make the
training-target characterization mint-blocking, and authorize the dead-weight deletions.
Fidelity label on each: **[INLINE]**.

## R151 — ADJ-4: CARD-DENSE-EVAL-ADAPTER, pre-Stage-0 blocking, NOT mint [INLINE]

> R151 — The composed fact stands: run5's ruled control arm has never
> produced an eval result and cannot until wired. But nothing on the MINT
> path consumes dense eval: the promotion bar is graph candidate-vs-best
> (deploy-matched), the ladder is SealBot+RandomBot (R139). The first
> consumer is Stage 0 re-baseline — post-mint. CARD-DENSE-EVAL-ADAPTER
> opened: wire infer_batch_per_cluster into the deploy-head decode for
> no-drop specs via R138's expand_fn seam, per the recommendation; HARD
> GATE: Stage 0 cannot open until it lands + LAW-10 anchors re-measure on
> it. Not ridden here — R124's drop discipline; the seam is committed and
> tested, the adapter cost does not decay. Phase A UNBLOCKED: EVAL_DECISION
> states control arm = v6_live2_ls (R148) with eval wiring owed, card
> named, refused-loud-by-name today (covered/not_run honesty). Wiring the
> ruled control arm executes the R20 frame, not settles it — no lock
> touched.

*ADJ-WP12R-4 CLOSED as a queue row → **CARD-DENSE-EVAL-ADAPTER**, pre-Stage-0 blocking, NOT
mint-blocking. The dispatcher's composed finding is ratified as fact and then correctly
bounded: it blocks Stage 0, not the mint, because no mint-path consumer reads dense eval.*

## R152 — ADJ-7 + Q-EVALBROKEN-RC0: ONE taxonomy, rides this stack [INLINE]

> R152 — Merged as recommended: both are "the parent cannot distinguish
> eval failures." One small design, one authority: the eval_broken reason
> enum is the single source; every failure route produces a typed reason
> (LAW-18: logged in-run with fire-rate); the parent maps reason → rc on
> the TERMINAL round only, per R133's split (mid-run stays non-fatal,
> watchdog owns persistence). R84 template: mutation proves each reason
> class supervisor-distinguishable. No second boolean/proxy beside the
> enum (R79).

*`Q-EVALBROKEN-RC0` leaves architect-reserved status and merges with ADJ-WP12R-7. R133's
caveat "rc 0 does not certify eval health" is discharged in full when this lands.*

## R153 — ADJ-10: characterize first; documented semantics is authority [INLINE]

> R153 — Escalation was correct — training-target mass is not a
> dispatcher's to accept. Authority is already settled: the policy target
> is raw_visit_distribution (R34 recon) and the targets are documented as
> NOT inheriting the off-window skip — an export that drops off-window
> mass diverges from its documented authority, R138's exact class on the
> training seam. n=1 does not carry a verdict (LAW-01/LAW-04):
> characterization FIRST, this stack — pre-registered instrument, dropped-
> mass distribution over a representative real-game position sample,
> verdict rule pre-registered BEFORE measuring. Any systematic drop
> confirms the class defect regardless of size — size sets urgency, not
> guilt; a bug is not a semantics. Confirmed → fix rides this stack, R71
> flip-set over the class boundary, exported-target parity oracle (export
> == raw visit distribution on the fixture). MINT-BLOCKING until the
> measurement rules it: run5 training on silently truncated targets is
> R138's handicap moved from eval to learning.

## R154 — ADJ-19: deletion authorized, conditioned [INLINE]

> R154 — resolve_anchor_path + resolve_arch die under the dead-weight law
> (R116 precedent: deleted, not documented), riding this stack, PROVIDED
> the armed census's zero-ref evidence is recorded per deletion AND
> neither is an R20-protected dense surface (verify; if either is →
> queue-with-recommendation per R108, the named-list exclusion stands
> meanwhile). Anti-rot test survives the deletions — it is the law's
> enforcement, not the exclusions' registry.


# R155–R156 — operator adjudication response, 2026-07-31 (R153 leg 1 close, leg 2 mandate)

Verbatim. These mandate the legal-set leg before any other work, add a STANDING instrument
clause on what may clear an encoding, ratify leg 1's conduct, and bind the fix's flip-set to
the untested cap-boundary hypothesis. Fidelity label on each: **[INLINE]**.

## R155 — leg 2 mandated; measurement-path parity clause [INLINE]

> R155 — The legal-set leg runs BEFORE anything else: expand_and_backup_ls_at
> driven over the same 1440-position sample for gnn_axis_v1 (+ any other
> legal-set spec), prereg BEFORE running — verdict rule for "run5 exposure
> confirmed/refuted", aborts, and the instrument named. Instrument clause,
> standing for this card and its oracles: a drop measurement clears an
> encoding ONLY when driven through that encoding's PRODUCTION expand path
> (LAW-03 false-clear corollary made mechanical) — leg 1's gnn_axis_v1 zero
> stays labeled non-production forever. Mint-blocking stands until leg 2
> rules: refuted → mint unblocks on this axis, fix still rides (class
> confirmed, R153); confirmed → fix scope re-verifies run5's exporter and
> the prereg gains a dropped-mass row. Order after leg 2: fix and R152 in
> either order or parallel.

## R156 — leg-1 conduct ratified; hypothesis binds the flip-set [INLINE]

> R156 — Ratified: abort 1 fired and was honored (supersede, never merge —
> two samples under different generators are not one sample, LAW-04 spirit);
> the "0 affected cannot be read as clearance" framing is R69/R44 hygiene
> done right; the documented-semantics authority call stands (records.rs:481
> is the authority, policy.rs:166-168 describes it correctly — the export
> diverges). The cap-boundary hypothesis stays labeled UNTESTED and is not
> consumed as explanation — but it BINDS the fix's flip-set (R71/R72): rows
> at n_legal just above the 192 cap (the 193–235 band), sparse-coverage
> early plies, AND deep-tail dispersed rows are all mandatory boundary
> coverage, so the fix cannot pass on the demo region alone. If the fix's
> behavior confirms or kills the hypothesis in passing, record it (LAW-01);
> do not build an experiment for it.

*Operator note recorded with R156, on leg 1's value: leg 1 was worth its cost — it convicted
the class on `v6_live2_ls` with full attribution and settled the docs-vs-code authority
question empirically. **An instrument that measures the wrong path for one encoding AND SAYS
SO is evidence; one that does not say so is the R44 class.** The standing lesson is not "check
the path first" but "an instrument must publish which path it drove", which is what R155's
clause now makes mechanical.*


# R157–R159 — operator adjudication response, 2026-07-31 (R153 leg 2 close, fix mandate)

Verbatim. These consume leg 2's CONFIRMED verdict, name the degenerate rows as a separate
class, order a contamination census, set the fix's class boundary at the WHOLE target
pipeline, and pre-grant the frozen-oracle re-points. Fidelity label on each: **[INLINE]**.

## R157 — leg 2 CONFIRMED consumed; degenerate class named; contamination census [INLINE]

> R157 — Leg 2 ratified: prereg held, abort 4 is what makes the leg
> admissible, the inversion is real, leg 1's gnn_axis_v1 zero stays labeled
> non-production forever. RUN5 EXPOSURE CONFIRMED is consumed; R155's
> consequences are in force: mint BLOCKED on this axis, fix rides this
> stack, RUN5_MINT_PREREG.md gains the dropped-mass row with these grounds,
> exported-target parity oracle required. The 12 all-zero rows are a NAMED
> SEPARATE CLASS, not folded into the mass statistic: a policy target must
> be a valid distribution ALWAYS — the fix makes degenerate export
> UNREPRESENTABLE (correct by construction or fail loud, LAW-14), never a
> silently-shipped no-op row. R156's cap hypothesis: KILLED as recorded,
> in passing, no experiment built — flip-set roles reverse per the
> measurement, both bands stay mandatory. CONTAMINATION CENSUS rides the
> fix: no mantis training run exists (run5 unminted — statement recorded),
> but any committed fixture, corpus, bank, or golden generated through
> get_policy_ls carries the defect — enumerate, then regenerate or label
> VOID-AS-ANCHOR (STATE §6 pattern). hexo_rl history out of scope.

## R158 — fix scope: the WHOLE target pipeline is the class [INLINE]

> R158 — R71 class named: "consumer diverges from the documented no-drop
> producer contract" — and the class boundary is the ENTIRE target
> pipeline, not the Rust exporter alone. Census every stage: get_policy_ls
> export → wire/record format → replay buffer → Python batch assembly →
> loss. If the Python side consumes only the dense half, the fix merely
> moves the drop downstream — R138's lesson on the learning seam; each
> stage gets a producer-consumer parity check. Authority: records.rs:481
> semantics (the documented contract; policy.rs:166-168 describes it
> correctly). Semantics: export == raw visit distribution over the FULL
> child set (R34: raw_visit_distribution), mass sums to 1 within TOL,
> off-window mass carried, never renormalized-over-a-subset. Flip-set per
> R156 as sharpened by leg 2: deep-tail dispersed rows (high-magnitude
> region), 193–235 band (low edge), sparse-coverage early plies, plus a
> degenerate-row case that must fail loud. Parity oracle both sides of the
> FFI on one fixture (R138 pattern). LAW-09: the exporter is selfplay
> hot-path — IQR-gated bench vs the box floors, prereg bracket + abort.
> Q-GAP re-measures ride the next box session, not this fix.

## R159 — frozen-oracle pre-grant for the fix [INLINE]

> R159 — Settled-class R43 grant, pre-authorized per R88/R90a: frozen
> parity oracles and fixtures that pin the DROPPING behavior as expected
> (graph_child_parity.rs and child-parity fixtures are the named suspects)
> may be re-pointed to the no-drop law citing R157/R158 — S-3 discipline,
> enumerated hunks, verbatim extraction, before/after hashes, R81/R86
> mutation condition (not self-satisfying, no unrelated casualty). Any
> hunk that WEAKENS an assertion rather than re-pointing it to the new
> law, and any oracle whose subject is not target/parity semantics:
> QUEUE, never edit. Your stated posture — bring it back rather than
> re-point yourself — was right for an unruled surface; this ruling is
> the ruling, so settled-class hunks now apply-and-report.


# R160–R161 — operator adjudication response, 2026-07-31 (R153 sims-provenance; degenerate class)

Verbatim. These halt the prereg row pending a sims/export-regime provenance census, and ratify
the unconstructible-target disposition. Fidelity label on each: **[INLINE]**.

## R160 — sims/export-regime provenance; the table is not yet the grounds [INLINE]

> R160 — "50 sims = run5's actual" is UNVERIFIED and conflicts with the
> recorded PCR frame (600@10%/75@90%). Mandated census, before DESIGN
> freezes: (a) derive run5's sim regime from run5.yaml + the PCR/playout-cap
> config on disk — derive-or-delete (R98), never from memory of either side;
> (b) determine from the tree whether policy targets export from quick-arm
> moves at all, or full-arm only — the exporter's own gating, cited by
> file:line; (c) re-run the binding table at the TRUE export regime(s),
> both arms if both export. The prereg dropped-mass row cites ONLY
> production_path: true AND production_sims: true rows. The 50-sim table is
> retained as mechanism evidence under its actual provenance label. If the
> tree contradicts STATE's PCR frame: queue, authority order governs, STATE
> corrected per R96 — never silently reconciled.

## R161 — degenerate class superseded 12→37*; unconstructible ratified [INLINE]

> R161 — R157's count supersedes to the R160-verified regime's measurement
> (37 at 50 sims stands as mechanism evidence meanwhile). Ratified: masking
> degenerate rows REJECTED with the dilution grounds; the fix makes a
> degenerate target UNCONSTRUCTIBLE — a target that is not a valid
> distribution cannot be built (correct by construction or named raise,
> LAW-14). MAX_VISITS guard ratified as argued: needed, not binding at run5.


# R162–R164 — operator adjudication response, 2026-08-01 (Phase T ratification; ADJ-20/21)

Verbatim. These ratify Phase T in full with one record correction, record the sims-regime
recommendation without ruling it, and rule ADJ-21 as a LAW-18 failure riding R152.
Fidelity label on each: **[INLINE]**.

## R162 — Phase T CLOSED, ratified in full; the record corrected [INLINE]

> R162 — Phase T ratified: the entry judgement call (R154's two refusals ARE
> R154 executing as written — its conditions caught a defective zero-ref
> census and an R20 surface; the false Phase-Q evidence is logged as
> census-gap instance #5); STOP-1/STOP-2/DEV-1 as recorded; the in-card
> RED-TEAM rulings stand (R161 is constructor-quantified — every public
> constructor, the lens said so); FA-1..FA-3; the bench with its honest
> out-of-bracket intermediate; T-4's dispositions including both
> VOID-AS-ANCHOR labels. The 2c disclosure is RATIFIED: the dense
> value-only sentinel with a mask excluding degenerate rows from numerator
> AND denominator satisfies R161's no-dilution grounds; reversal point
> recorded. QN-1: deletion AUTHORIZED (outlawed semantics + zero consumers;
> confirmed non-R20-dense in the same commit line). The prereg dropped-mass
> row: APPEND as written. CORRECTION on the record per R96: "remaining path
> to mint is operator territory" is FALSE — Phases A and D, CS, and R152
> are owed engineering; the claim is struck in the dispatch log.

## R163 — ADJ-WP12R-20: sims regime is a prereg decision; recommendation recorded [INLINE]

> R163 — The flat-50 disk config vs STATE's PCR 600@10%/75@90% frame is an
> armed-value question: prereg-only, operator-only (R82/R90 hard-stop
> class). Architect recommendation, recorded not ruled: re-arm PCR 600/75
> per the recorded intent — compute-legal (mean ≈128 < the 150 hard cap),
> value-target-quality grounds stand, and the quick-arm policy mask means
> policy targets come from full-search moves. If re-armed: the binding
> table gains its 600-sim supplemental row and the prereg states the mask
> semantics explicitly. Flat 50 is acceptable ONLY as a deliberate prereg
> line with grounds — never as an accident of a zeroed block. The R96
> STATE correction stands either way.

## R164 — ADJ-WP12R-21: LAW-18 means in-run; rider on R152 [INLINE]

> R164 — Test-visible-only counters FAIL LAW-18 — the law's text is
> in-run, and its provenance (a null read unreadable without in-run
> counters) is exactly this case. Rider on R152: the Phase-T counter
> family including the §0b drift witness reaches the EVENT STREAM in
> R152's landing; one taxonomy + one observability commit family, R84
> mutation discipline. R152 is confirmed pre-merge (it was already owed;
> this makes it load-bearing, not just owed).


# R165–R167 — operator adjudication response, 2026-08-01 (sims regime; mutation-mechanism law; Phase O ratification)

Verbatim. These adopt the PCR sims regime for run5, raise the Phase-O mutation-mechanism
discipline into a STANDING law, and ratify Phase O in full including the queue-transfer
practice. Fidelity label on each: **[INLINE]**.

## R165 — sims regime: OPERATOR ADOPTS PCR 600/75; configurability confirmed [INLINE]

> R165 — Operator decision recorded: run5's sim regime is PCR re-armed at
> 600 full @ 10% / 75 quick @ 90% (mean ≈128, under the 150 hard cap),
> per R163's recommendation and the original intent. The prereg sims-regime
> line states this + the quick-arm policy-mask semantics; run5.yaml
> re-mints via tooling AT prereg authoring, not before (armed-value
> discipline). Configurability is ALREADY satisfied — the PCR block is
> schema-typed config with the V-PCR validator (R40); nothing new is
> built. Rider, pre-merge: the 600-sim supplemental binding-table row runs
> on the existing leg-2 instrument (CPU, cheap) so the prereg cites
> measured numbers at both arms. Flat-50 stays legal for smoke configs
> as a deliberate minted value. ADJ-WP12R-20 closes when the prereg line
> is written.

## R166 — mutation-mechanism law, standing from Phase CS [INLINE]

> R166 — Every mutation row carries a MECHANISM column: what the mutation
> reaches, and why the pin can see it — reachability stated, never
> assumed. Four instances in one phase (interned-int identity assert,
> call-site edit vs signature check, Rust rename done in Python,
> unfireable eleven-line branch) earn the law; the fifth instance being
> found exactly where the mid-phase fix was NOT back-propagated is the
> proof of the back-propagation clause: a discipline invented mid-phase
> applies to the phase's earlier tables in the same event. This is LAW-07
> producer-testing applied to mutations themselves.

## R167 — Phase O CLOSED, ratified [INLINE]

> R167 — Phase O ratified: three rows closed as stated, R133 caveat
> DISCHARGED with mechanism named (terminal exit 48, seven pins, M-O8
> two-sided, no-over-fire proven), ten non-mint rows transferred to the
> queue — the queue-transfer practice is RATIFIED AS STANDING (findings
> live in the adjudication queue, never only in impl notes; the ADJ-12/13
> lesson made law). The red team's two discoveries are logged as
> register-grade: 42 pre-registered mutations had never been executed by
> any prior stage (execution, not registration, is the evidence — R69
> applied to mutation banks), and a renamed pyo3 getter shipping a
> fabricated 0 through a green tree is the exact class R166 + the FFI
> parity oracles exist to kill.


# R168–R169 — operator adjudication response, 2026-08-01 (CS second-FAIL; ADJ-WP12R-24)

Verbatim. These adjudicate the Phase-CS two-FAIL stop and rule the ladder-liveness question.
Fidelity label on each: **[INLINE]**.

## R168 — CS second-FAIL adjudicated: bounded retry authorized; FAILs stand as history [INLINE]

> R168 — The two-FAIL stop rule targets an implementation that keeps
> failing; this implementation has survived two full adversarial passes
> with zero behavioral claims falsified, and FAIL-2 originated in the
> auditor's own transcription (11/13, not 12/13 — corrected at source per
> R96). Neither verdict is retroactively relabelled: the FAILs stand in
> the record; the retry DISCHARGES them. One bounded retry authorized at
> the enumerated scope EXACTLY: the four workspace-text items — the live
> M-1/OC-4b cell re-labelled to its true mechanism (the mutant dies at
> pytest.raises(OSError) before the zero-events assertion — the kill is
> real, the recorded mechanism was wrong), the wrong-mechanism-right-
> outcome cell corrected, the 11/13 figure propagated to every downstream
> artifact, grants doc already fixed. NO code, no sealed-file edit — any
> code need discovered mid-retry → the STOP stands, back to me. R166's
> back-propagation clause applies IN THIS RETRY: the mechanism column is
> verified across the ENTIRE CS mutation table, not the two named cells —
> this is the class's third instance in one card family and 26 lines is
> not a search radius. Targeted re-verify (RED-TEAM-3 pattern) of the
> corrected cells + the back-prop sweep only; clean → CS CLOSED, PASS
> with its honest FAIL→FAIL→discharge chain in the ledger row.

## R169 — ADJ-WP12R-24: liveness claims tie to their instrument; box rider approved [INLINE]

> R169 — "2/6 live rungs" is NOT claimable bare: a liveness claim with no
> runnable producer is the R69/LAW-07 class. Disposition: EVAL_DECISION.md
> states "2/6 resolve locally; liveness unverified in CI; verified at box
> preflight" — covered/not_run honesty, claim bound to its instrument.
> The liveness build RIDES the Phase-D box session (approved — cheap,
> batched, R142's grant covers it); its result upgrades the decision doc's
> line in place per R96 before the prereg cites it. No CI gate is built
> for it pre-mint (finish-line posture; card it WP-R if wanted later).


# Operator adjudication response, 2026-08-02 (R170 — dispatcher commit authority)

Verbatim. Clarifies the dispatcher's commit authority, supersedes the WP3-era template text,
retroactively ratifies the Phase O and CS commits, and carves the merge out as trigger-gated.
Fidelity label: **[INLINE]**.

## R170 — dispatcher commit authority clarified; template text superseded [INLINE]

> R170 — Standing: a dispatcher MAY commit to the wp-scoped scratch branch
> when (a) the verdict chain at that boundary is PASS or R70's six-part
> test is met, (b) gate evidence precedes the commit (R42), (c) the
> commit's content is subagent-authored EXCEPT transcribed measurements
> (the floor file is the named instance — a measurement, not a design
> decision), and (d) the authority is CITED in the ledger row, never
> inferred from precedent. dispatcher_template.md's "do NOT commit" is
> WP3-era text SUPERSEDED by R70/R47's practice — corrected in the
> template with a pointer to this ruling (workspace artifact, no repo
> commit needed). Both Phase O and CS commits are RATIFIED under this
> test retroactively; the disclosed sequence error (commit before
> authority derivation) is logged as derive-or-delete instance #10,
> caught by its own author — recorded, not punished. The MERGE is
> carved out: R150's runbook text is the plan, not the trigger; the
> ff-merge + scratch deletion + push execute only on a fresh go-line
> from the architect issued against the assembled evidence. "Authorized
> in a ruling" and "authorized right now" are different things for an
> irreversible action — the dispatcher's own framing, adopted verbatim
> as the standing rule for every future merge.


# Operator adjudication response, 2026-08-02 (R171 — Phase A bank freeze)

Verbatim. Ratifies the Phase-A oracle bank freeze and its three dispatcher rulings, names the
dispatcher's own defect pattern, and clears IMPL. Fidelity label: **[INLINE]**.

## R171 — Phase A oracle bank ratified; instance counts recorded [INLINE]

> R171 — Bank freeze (10/10) RATIFIED with all three recorded rulings:
> (a) F-A3 — the +48 band was a withdrawn figure carried from
> conversation instead of re-derived from PREREG_A REV 5 (+52…+86);
> corrected in the ARTIFACT per R96, derive-or-delete instance #11,
> dispatcher-owned, self-caught. The pattern is now named: both
> dispatcher instances are numbers taken from chat rather than the
> document — the standing answer is the one this lineage already has,
> derive from the artifact at point of use, and it applies to briefs
> exactly as to prose. (b) F-A2 — reachability class instance #9, first
> caught by MEASUREMENT at oracle-write (in-process rebind probe): the
> table amendment needs no grant, correctly reasoned — the frozen nodes
> are observers already redding correctly; the defect was the table's
> silence, and R166's mechanism column is what the amendment fills.
> (c) The refused weak-red is ratified as the bank's load-bearing
> property: 22/23 greened against a synthetic document, then seven
> mutations each redding exactly its named row — that is a bank that
> gates, not one that notices. The 0.60-vs-"0.6" comparator catch is
> logged as detector-self-test provenance (LAW-07 applied to the
> checker itself). IMPL proceeds against the frozen bank.


# Operator adjudication response, 2026-08-02 (R172 — Phase A close; the ADJ-22 hold)

Verbatim. Ratifies the Phase A close, makes the 7/3 manifest non-re-mint a STANDING rule, and
enforces the ADJ-WP12R-22 hold on Phase D. Fidelity label: **[INLINE]**.

**TRANSCRIPTION PROVENANCE (dispatch 5, 2026-08-03).** This ruling was ISSUED 2026-08-02 but never
reached the register — it was cited exactly once in the whole workspace
(`wp/WP12R/WP12R_dispatch_5.md:15`) and had no section here. Recovered verbatim from the operator
turn in the main session transcript (`6be40f85-…jsonl` line 1745, `type: user`,
`isSidechain: false`, ts `2026-08-02T20:31:44.684Z`, uuid `43949f7d-f54e-4e9d-bdd9-d3f3e4ab3e49`)
and transcribed by the dispatcher under dispatch 5's instruction to "verify R118–R177 exist
verbatim, fill gaps from the chat record before executing". The block quote is the operator's text
unaltered; the remainder of that turn was a one-off instruction about ADJ-22's verbatim text and is
recorded in the dispatch log, not as ruling text.

## R172 — Phase A CLOSED, ratified; the ADJ-WP12R-22 hold ENFORCED [INLINE]

> R172 — Phase A close ratified in full: the 7/3 manifest non-re-mint is
> CORRECT and is now the standing rule — pre-IMPL hashes are the evidence
> the implementation did not shape its tests; re-minting erases what the
> freeze exists to prove; grants docs are the key (Phases O/CS/A all ship
> this way). The liveness line shipping in STATE 1 under abort 10 is
> covered/not_run honesty done right — two agents declining an upgrade
> they had incentive to make is the instrument working. Commit text under
> abort 11 with R169's wording: correct, dispatcher-authored, disclosed.
> THE HOLD STANDS: Phase D does not open the box while ADJ-WP12R-22 is
> unread by the ruling authority. Five asks, five non-answers — the row
> now blocks by ruling, not by request.

**Status note:** the hold this ruling enforced was LIFTED by R174 once ADJ-WP12R-22 was resolved by
R173. The 7/3 non-re-mint standing rule survives unchanged and governs the freeze manifests at every
later phase.


# R173–R174 — operator adjudication response, 2026-08-03 (ADJ-WP12R-22 resolved; Phase D go-line)

Verbatim. These resolve the WP's last open mint-blocker against the dispatcher's recommendation,
amend R137 leg (b), open CARD-CS2, and sequence the Phase-D box session behind it.
Fidelity label on each: **[INLINE]**.

## R173 — ADJ-WP12R-22 RESOLVED: option (ii); R137 leg (b) amended; CARD-CS2 [INLINE]

> R173 — Leg (a) closes COMPLETION exposure only. What (i) leaves open on
> a multi-day 25000-step run: (1) crash-loss — SIGKILL/OOM-kill/power
> bypass shutdown_save; one hard fault loses the entire run, the
> lineage's founding trauma (R2/R12/LAW-14 provenance); (2) LAW-10 is
> unexecutable — the threat probe gates "each 5k-step checkpoint" and
> there are none; (3) eval/promotion candidates — mid-run candidate
> snapshots are what the gate evaluates (M-4's own blocker demonstrates
> it); (4) LAW-08/R1 — under (i), run5's minted config carries
> checkpoint_interval with ZERO live consumers on its own route: the
> exact dead-knob class R1 exists to kill, and "recorded as inert with
> grounds" is documentation of a violation, not a remedy. Therefore
> R137 leg (b) is AMENDED: the prereg row requires a nonzero value AND a
> live consumer on run5's route — CARD-CS2, riding this stack, scope:
> wire the interval read + periodic save into the graph step through ONE
> resolver (rule 3; the dense step converges onto the same seam — no
> second interval authority), reusing Phase CS's save machinery
> (LAW-12 stamp, LAW-14 fatality) unchanged; mutation test (interval N →
> checkpoints at N, 2N; interval 0 → none but the final); LAW-09 bench,
> prereg bracket, the write is interval-gated so the cost lands at the
> boundary step. Full-but-light pipeline. The graph-periodic gap does
> NOT become a post-mint card — it dies here. Prereg value stays
> operator's; architect recommendation recorded: 5000, grounds LAW-10
> cadence alignment. The dispatcher's queue-not-absorb call was exactly
> right (R119 hard stop honored); the (i) recommendation's error was
> scoping, and it was correctly submitted for ruling rather than acted
> on. Consumer-registry prose-map blindness: logged as the R93 class,
> row for the WP-R sweep. Stale cites: R39 discipline, already
> re-derived at 32ec7b9, disclosure ratified.

## R174 — Phase D go-line, sequenced [INLINE]

> R174 — The ADJ-WP12R-22 hold LIFTS. Order: CARD-CS2 lands on scratch
> first, then Phase D opens the box — so the 25001-step burst exercises
> the periodic save under production reality (R61: production-only axes
> get production-time checks; a checkpoint appearing mid-burst is now a
> burst assertion). D's four payloads unchanged, plus that assertion.
> Merge remains gated on my go-line per R170 with CS2 added to R150's
> roster.


# R175–R177 — operator adjudication response, 2026-08-03 (final-mile close-out; queue discipline; compute placement)

Verbatim. These ratify the close-out and rule its four decision items, set queue discipline ahead
of the R150 roster, and place compute for the remainder of the WP. Fidelity label on each:
**[INLINE]**.

## R175 — final-mile close-out ratified; four decision items ruled [INLINE]

> R175 — Close-out ratified incl. the refused one-shot-script findings
> (an instrument that runs once is not an instrument — correct). Items:
> (a) TERMINUS COLLISION: accepted AS PINNED — two stamped artifacts with
> distinct provenance (periodic vs clean-stop) are unambiguous under
> LAW-12; dedupe logic would add a branch to buy nothing; recorded.
> (b) LAW-10 PROBE ABSENT: CARD-THREAT-PROBE opened, NOT mint-blocking —
> run5's promotion bar is the live LAW-15 deploy-matched gate; the probe
> is a checkpoint canary. Owed PRE-STAGE-0 (anchors re-baseline there);
> the prereg gains an INSTRUMENT INVENTORY line recording its absence —
> a law without its instrument is stated, never implied healthy.
> (c) SEALBOT LICENSE: no license at the pinned sha = fetch-and-run
> locally for the operator's own eval only; SealBot source/artifacts
> NEVER enter hexo-mantis or anything distributed (the vendoring law
> already enforces this); pins.toml comment + ledger row record the
> status; operator flagged — contacting upstream is his call; not a
> run5 blocker. Not legal advice; the conservative posture is the point.
> (d) "rule 3" = CLAUDE.md hard rule 3, one-resolver-per-regime-knob —
> clarified in the register; both pinned readings were compatible with
> it, nothing moves.

## R176 — ADJ-WP12R-23 + queue discipline for the merge [INLINE]

> R176 — ADJ-23 was opened in the same header as ADJ-22 and its verbatim
> text has ALSO never reached the ruling authority — the -22 lesson
> applies before it can repeat: the next dispatcher's FIRST report opens
> with ADJ-23 verbatim. Provisional frame, pending the text: if run5 has
> NO resume path, buffer-not-saved is genuinely non-mint (crash = restart
> or cold-buffer resume behind the real prefill gate, stated in prereg);
> if resume-from-checkpoint is a claimed capability, it ESCALATES —
> CS2's weights without the buffer is half a resume. The ~25-row queue:
> audited to a classified table (mint-relevant must be zero or ruled)
> BEFORE the roster is evaluated — R150 already requires it; the audit
> is a parallel-track task, not a merge-morning discovery.

## R177 — compute placement + parallel tracks + the ast-derivation practice [INLINE]

> R177 — Standing for the rest of the WP: heavy compute (bursts, box-floor
> benches, liveness games, scored rounds) runs on the vast box; the local
> machine carries only the test/gate suite and sweeps (which stay
> serialized-quiet per R58/R68/R76 wherever they have always run). Box
> fitness check BEFORE the burst: disk/RAM/GPU recorded, quiet, stack
> pins verified against STACK_PINS — drift → record + provenance note on
> every floor-referenced bench, queue if material (R18/R21). TWO PARALLEL
> TRACKS authorized: BOX (Phase D's five payloads) and MERGE-PREP (local,
> no box: queue audit per R176, R150 roster pre-audit of every non-D
> cell, prereg row drafting incl. the checkpoint_interval replacement and
> instrument-inventory line, ledger/register verification R118–R177).
> The ast-derived firing-order practice is RECORDED as the reference
> mechanism for R166 compliance — recommended, not mandated, until it
> proves out once more in CS2/D evidence.


# R178–R180 — operator adjudication response, 2026-08-03 (ADJ-23 resolved; Phase D split; corrections ratified)

Verbatim. These resolve the WP's last MINT-RELEVANT-OPEN queue row, split Phase D and route its
fix to a dedicated dispatch, and ratify dispatch 5's entry-state corrections and the R172 register
recovery. Fidelity label on each: **[INLINE]**.

## R178 — ADJ-WP12R-23 RESOLVED: delete the dead key; posture documented; CARD-RESUME [INLINE]

> R178 — No resume path exists (measured at 982da03) — the non-mint branch
> holds, and with it the disposition: (a) DELETE train.buffer_save_interval
> + the no-op _try_save_buffer gate arms under R116/LAW-08 — a key minted
> into run5.yaml with zero reachable effect is the exact dead-knob class R1
> exists to kill, and it does not ship into the mint record; mechanical
> config re-mint rides the deletion. (b) The prereg gains one posture line:
> run5's buffer is deliberately non-persistent; crash = restart from
> scratch; warmup governed by the real prefill gate. (c) CARD-RESUME opened
> POST-MINT, owning the run.py:317-318 "owed S-2 work" as ONE design:
> weights + optimizer/scheduler + buffer persistence + launcher surface
> together — half-resumes are the trap this WP just spent a phase killing;
> nobody builds any piece of it separately. Schema-comment half: already
> discharged by CS, confirmed. Wire-now REJECTED: new scope pre-mint
> against the finish-line posture, for a capability with no consumer.

## R179 — Phase D SPLIT; the fix is its own dispatch; amplifier sequenced inside it [INLINE]

> R179 — D-DIAGNOSIS is CLOSED, PASS — its deliverable was the mechanism,
> and it delivered one that overturns the card's own premise (training
> step, not inference; E unbounded, batch_size bounds the wrong quantity;
> fp32 gather amplifier confirmed on-box to the observed gibibyte).
> R114's suspect list is corrected in the register by this ruling. D-FIX
> is its OWN dispatch, full pipeline — a config-typed edge cap with
> gradient-accumulating micro-batching is training-hot-path engineering
> with numerics semantics (loss normalization across micro-batches,
> determinism, LAW-18 fire-rate), not a session tail; cramming it is how
> the next Phase-T happens. Sequencing INSIDE D-fix, ruled now: the fp32
> gather DISPOSITION is DESIGN's first sub-decision — casting the gather
> to bf16 aligns with LAW-06's intent and roughly halves the dominant
> tensor, which changes what cap value the I-3 headroom curve yields; it
> gets its own prereg + parity oracle + commit, THEN the cap is sized
> once on the resulting envelope. Two changes, two commits, two benches,
> one sizing pass (confound discipline; no double-sizing). "The burst got
> further" is BANNED as evidence (death is stochastic 173/882/180 s) —
> success criteria are structural: peak-allocation bound proven at the
> cap, OOM unconstructible by cap semantics, burst survival as
> corroboration only. R150 amended: merge waits on D-fix + payloads 2–5
> (which ride D-fix's box session). Q-D-1 is MINT-RELEVANT: the preflight
> is a mint instrument and rc 40 where 46fc83e returned the truer 33 is a
> classification regression — root-caused and ruled before merge, rides
> D-fix or stands alone, dispatcher's routing call.

## R180 — corrections, recovery, and the probe lesson ratified [INLINE]

> R180 — Entry-state corrections accepted: 17 commits (my dispatch figure
> stale by the ratchet commits), freeze 47/66 with CS 0/2 drift authorized
> under G-CS2-1/2 — R51 correctly not fired. The R172 register recovery
> with provenance is RATIFIED and vindicates the verify-don't-trust
> handoff clause — the gap was real. Track 2 accepted in full: the 54-row
> audit (not ~25 — my figure, corrected), roster pre-audit with cell 4's
> gap register (AMBER acceptable: R150 requires rows with owners, values
> are prereg-authoring acts), R175a/b/c executions staged. The fitness
> probe disclosure is ratified with its lesson made standing: a fitness
> probe that can mutate the thing it measures is not a fitness probe —
> box probes run read-only (UV_NO_SYNC or equivalent) and assert the pins
> they found, and this rides STACK_PINS' own warning text.


# R181–R183 — operator adjudication response, 2026-08-03 (ADJ-26 disposition; BF1-2 hold; ADJ-25 routed)

Verbatim. These rule the D-FIX parity-oracle disposition, sustain the faster-than-envelope hold with
a bounded probe to close it, and assign R178(a)'s execution. Fidelity label on each: **[INLINE]**.

## R181 — ADJ-26: the STATISTIC is replaced; the band never moves [INLINE]

> R181 — Recommendation ADOPTED on its empirical grounds: an oracle whose
> max-form statistic reads up to 3.96e-1 on HEAD-vs-HEAD cannot witness
> F1 — it measures index_add_ atomics, not the change. Moving the band
> would green the tier and forfeit the witness: REJECTED, and recorded as
> the general rule — a band is never moved on a statistic that cannot
> distinguish its subject from nothing. Disposition: the parity oracle
> re-points to a statistic that is ZERO on identical code (the median
> form measures 0.000e+00 on all pairs — it is the discriminating base),
> bounded by a null-calibrated envelope derived from the MEASURED
> HEAD-vs-HEAD distribution with stated margin; the null-distribution
> measurement commits as a pinned artifact and is the envelope's cited
> grounds (R69). Frozen-oracle re-point: this IS the oracle-contradicts-
> design adjudication — granted under S-3 with the two-sided mutation
> condition: new statistic REDS under an injected real numerics change
> and GREENS on N fresh HEAD-vs-HEAD pairs. Where exactness matters,
> a deterministic-path (CPU) parity leg may assert equality outright —
> CUDA legs assert the calibrated bound, and each leg's label says which
> it is. F1 commits only after the re-pointed tier is green for real.

## R182 — BF1-2: HOLD SUSTAINED; one probe converts mechanism to root cause [INLINE]

> R182 — Correct hold — a plausible mechanism is not a root cause, and
> faster-than-envelope has burned this lineage before (LAW-15's
> wall-clock provenance). One targeted, budget-bounded box probe: vary
> bytes at fixed arithmetic (or read the profiler's memory-throughput
> bound on the gather directly) — wall tracking bytes confirms
> bandwidth-bound; confirmed → BF1-2 ACCEPTED with mechanism named and
> F1's wall envelope updated to measured reality; not confirmed →
> escalate with the probe data. Minutes of box, closes the row honestly.

## R183 — ADJ-25 routed; BF1-4 retired; the F3 question comes verbatim [INLINE]

> R183 — (a) ADJ-25: R178(a)'s deletion is ASSIGNED to this dispatcher —
> settled-class execution citing R178, own commit on the stack, config
> re-mint via tooling, lands before the merge evidence package.
> (b) BF1-4's premise is RETIRED BY MEASUREMENT: F1's effect shrinks at
> production in-degree (independent per-edge errors average down) — the
> review's finding-6 hold was right and is discharged in the opposite
> direction from expectation; the design doc gets the R96 correction.
> (c) The "Phase-B roster question from F3" reaches the architect
> VERBATIM in the next report — the ADJ-22/23 rule is now unconditional:
> no row is ruled from a mention.


# R184–R187 — operator adjudication response, 2026-08-03 (grant ratified; ratchet-down law; freeze authority; mint header)

Verbatim. These ratify the R178(a) grant and its overlap reconciliation, establish the standing law
for a downward test-count move, resolve the disagreeing freeze duplicate, and order the mint-header
defect fixed. Fidelity label on each: **[INLINE]**.

## R184 — G-R178A-1 ratified; grant-overlap reconciliation is the standing shape [INLINE]

> R184 — The grant's execution is ratified: throwaway-worktree authorship,
> sealed file byte-identical (hash-verified), and the patch makes the
> census STRICTLY STRICTER (4 reads/3 sites → 1/1; a same-spelled second
> reader now reds where the expectation previously absorbed it) — a
> re-point that tightens is the easy class, and it was still run through
> the full discipline. The explicit reconciliation against G-DFIX-2-C —
> overlap named, premise checked untouched, F2's implementer barred from
> citing this grant, both grant numbers on the twice-amended manifest —
> is RATIFIED AS THE STANDING SHAPE for overlapping grants on one file:
> reconciled in writing, never let pass. The blast census finding the two
> template sites the dispatcher missed is the census doing its job;
> both ride the deletion commit.

## R185 — ADJ-27: floor lowering authorized; the ratchet-down law [INLINE]

> R185 — The refusal to hide a ruled deletion inside F1's unrelated +16 is
> exactly right and is the reason this ruling exists: a floor that passes
> by coincidence of sequencing is a floor that lies (R44 class), and the
> record must be reproducible from a clean tree at every commit. STANDING
> LAW: the test-count floor may move DOWN only when ALL hold — (1) the
> deletions are RULED (register citation in the commit message), (2) the
> delta is attributed test-by-test, (3) the new floor is the MEASURED
> clean-tree count at that commit, (4) the lowering is its own visible
> act, never absorbed into an unrelated addition. ADJ-27 meets all four:
> floor 2532 → 2529 with grounds R178/R184, own commit line. The
> INSERT-ONLY precedent (teach the instrument, don't weaken it) is
> recorded beside it as the companion pattern: when a gate blocks ruled
> work, the gate learns the ruled class — it does not loosen.

## R186 — ADJ-28: one freeze authority; disagreeing duplicates become a red [INLINE]

> R186 — The tree-matching T freeze GOVERNS; the EVALDECODE row is
> corrected to T's sha with an S-3 record naming this ruling — a file
> frozen twice at disagreeing shas is not two freezes, it is zero
> (neither can be trusted until reconciled). The sharper half is the
> instrument: a freeze verification that reads clean over disagreeing
> duplicate rows trains readers to wave known-bad rows through — the
> checker is TAUGHT per the R185 companion pattern: duplicate path ⇒
> shas must agree or the verification REDS, with a mutation proving it.
> Sweep the freeze registers for further duplicates in the same pass;
> report the count either way.

## R187 — mint header defect: mint-relevant, fixed now [INLINE]

> R187 — mint_config.py writing Python str() into headers ("None" for
> None, failing validation) breaks minted-config provenance — the exact
> surface R1's "minted, never hand-varied" rests on, and the prereg
> authoring path will mint configs. MINT-RELEVANT: fixed in its own small
> commit riding this stack — proper serialization, a header round-trip
> test (mint → load → validate green, None handled), and a check that no
> EXISTING minted config carries a stringified-None header (if any does:
> re-mint mechanically, list them). Found-in-passing, fixed-in-full is
> the finish-line posture applied correctly.


# Operator adjudication response, 2026-08-03 (R188 — commit trio ratified; R185(4) clarified)

Verbatim. Fidelity label: **[INLINE]**.

## R188 — commit trio ratified; R185(4) clarified by direction; rulings upheld [INLINE]

> R188 — baade9c/842ead4/e7b0c1a ratified. (a) R185(4) CLARIFIED: "its own
> visible act" means never absorbed into UNRELATED work — a downward move
> rides the commit that CAUSES it, because the direction dictates the
> shape: splitting a downward ratchet leaves the intermediate commit red
> at its own floor, and a convention that only worked upward was a latent
> assumption until this exposed it. The dispatcher's ruling is adopted
> into R185's text. (b) Sequential falsification ACCEPTED as ruled: a
> claim true at its commit, narrowed by a later commit with an explicit
> cross-reference, is R9's amendment trail — pre-hedging against an
> unknown future would falsify the earlier record instead. (c) The PRE
> verification is the method lesson of the batch: two of four checks
> would have passed on a wrong PRE because the patch derives FROM PRE —
> the independent rebuild (green at the exact baseline its re-mint later
> turned red) is what made the verification real; the fifth byte-backup
> check and the total #-prefix discriminator are recorded with it.
> (d) R187's third sighting — the defective minter unable to regenerate
> its own defective output — is logged as the fix's own vindication:
> a provenance chain that cannot round-trip is exactly what R187 called
> mint-relevant. (e) The checker raising on its own author, and the
> drift trace living in the grants doc rather than as comments that
> would degrade sha256sum -c to warnings: both correct, second one is
> R88's actual ask. G-R178A-1 discharged; freeze registers at zero
> disagreements.


# R189–R190 — operator adjudication response, 2026-08-03 (R90c third loop; error ledger ratified)

Verbatim. Fidelity label on each: **[INLINE]**.

## R189 — R90c third loop GRANTED; MA-4's record corrected by ruling [INLINE]

> R189 — R90c's conditions are met on their face: loops converged
> (REVISE(7)→…→REVISE(1)), scope enumerated (two sentences, two
> documents, drafted at §L2-1, no code/measurement/band/oracle). GRANTED
> at that scope exactly. The correction's substance, so no reader
> re-litigates it: on the MA-4 path the RAISE is the detector —
> index_add_ raises on the Float/BFloat16 mismatch before _OpRecorder's
> assertion is ever reached (recorder calls the op before it records);
> the assertion is NOT the detector and the raise-path coverage is
> load-bearing, deletable by nobody on the strength of those two
> documents. PREREG_DFIX:280 + DESIGN_DFIX:1153-1154 adopt IMPL_NOTES §6's
> correct answer with a cross-reference. Non-blocking pair: both skip
> paths (unittest.SkipTest; marker-on-the-row) are AUTHORIZED as a
> settled-class fix with mutation proof each — own small commit or
> riding the loop-3 commit family, dispatcher's routing, disclosed;
> NOT inside the R90c scope, beside it.

## R190 — errors #8/#9 ratified as corrected; the notarisation clause [INLINE]

> R190 — #8: the G-R181-1 obligation error (correction misread as
> deletion) was caught before it re-created R186's disagreement, both
> rows updated, freeze-verify clean — and the stated argument is
> adopted into the record: BUILD INSTRUMENTS RATHER THAN RESOLVING TO BE
> CAREFUL; the morning's checker would have caught the evening's error
> regardless of anyone's diligence. #9 is the graver and the dispatcher's
> own reading is exactly right: repeating an unverified claim is the same
> violation as making it (R69's second clause), and a dispatcher's log
> NOTARISES — later phases treat it as settled, so adoption grants
> authority the claim never earned. Withdrawn in the artifact, correct.
> Standing emphasis, not new law: derive-at-point-of-use binds HARDEST on
> claims that flatter their maker and cost nothing to accept — that is
> the selection pressure that fills logs with convenient falsehoods.


# Operator adjudication response, 2026-08-03 (R191 — ADJ-29 resolved; the deterministic-mode class settled)

Verbatim. Fidelity label: **[INLINE]**.

## R191 — ADJ-WP12R-29 RESOLVED: deterministic-mode assertion; the class is settled [INLINE]

> R191 — FAIL upheld; recommendation ADOPTED. The CUDA parity leg runs
> under torch.use_deterministic_algorithms(True) and asserts EXACT
> equality on identical code (measured: 15/15 exact zeros under
> determinism) — an exact assertion beats any calibrated bound, is
> device-independent, and R181 already authorized the shape (CUDA-in-
> deterministic-mode IS the deterministic path where F1 does something).
> Determinism is TEST-SCOPE ONLY — production keeps its kernels; the leg
> says so in its name. The 5080 null artifact is re-labeled
> DEVICE-SPECIFIC (its "0.000e+00 on all pairs" was a property of that
> GPU's kernel majority-bit-identity, not of the statistic — the
> artifact's own columns prove it: >50% bit-identical there, 12–13% on
> the 4060); per-device envelope calibration is REJECTED as a treadmill.
> Instance #10 recorded: the dispatcher's first repro omitted DEPTH —
> the structure that produces the effect (nondeterminism compounds
> across four index_add_ calls) — and nearly refuted a correct FAIL;
> standing: a null probe must reproduce the compounding structure of
> the thing it nulls. Secondary finding BINDS F2: the median form reads
> exactly 0.0 against a 2.6e+3 defect confined to ≤50% of graphs —
> median is blind to minority-subset defects, so F2's oracle uses
> deterministic-mode exact/per-graph assertions, never inherits this
> statistic. STANDING SETTLED-CLASS from here: "oracle re-point from
> nondeterminism-contaminated statistic to deterministic-mode exact
> assertion" auto-applies under S-3 + the R81/R86 mutation condition
> (reds under an injected real numerics change, greens on fresh
> identical-code pairs, both devices where available) — no further
> architect round-trip for this class. What survived the attack is
> recorded as scoped: F1's memory result, reachability, loader
> refusals, the CPU-proxy verdict agreement 5/5 — the FAIL was the
> instrument's, never the change's.


# Operator adjudication response, 2026-08-03 (R192 — R191 execution ratified; C2 suspended)

Verbatim. Fidelity label: **[INLINE]**.

## R192 — R191 execution ratified; C2 SUSPENDED pending provenance; #11 recorded [INLINE]

> R192 — (a) Instance #11 accepted as classified: R190's notarisation
> class, two rulings later — a brief that instructs citing a specific
> column the artifact does not contain is the same act as notarising a
> false cite, and the agent CORRECTING THE DISPATCH rather than complying
> is the pipeline's proudest reflex; the entailment (median exactly zero
> ⟹ ≥50% bit-identity, by definition of a median) replaces the phantom
> column in the record. (b) The C2 disagreement: refusal to smooth is
> ratified, and the status is now RULED, not just recorded — C2 is
> SUSPENDED as load-bearing until the provenance hypothesis
> (untrained-randn vs shipped fixture) is VERIFIED by running both
> inputs through one probe on one device; whichever side reproduces,
> the loser's numbers get the R96 correction. The digit-exact
> alt_min match across two GPUs is noted as evidence the shipped
> fixture's behavior is stable — which is the property C2 exists to
> pin, and why the verification is cheap. Not on the merge path unless
> something re-makes C2 load-bearing; then it blocks. (c) The near-miss
> disclosure (wrong interpreter, one step from "no CUDA torch here")
> is logged with its method: ENUMERATE, don't guess — filesystem
> enumeration of torch/version.py is the derive-at-point-of-use of
> environments. (d) Declared +3, measured +4, reported as a miss: the
> discipline working as designed — the uncounted row carrying the
> central finding is exactly why declare-in-advance exists. (e) Both
> secondary closures ratified; the count-removal is the standing answer
> to self-staling numbers (a figure that must be re-edited on every
> sibling addition will be wrong and read as evidence — derive-or-
> delete, R98, applied to test prose).


# R193–R194 — operator adjudication response, 2026-08-03 (S3 ratified; F2 GO)

Verbatim. Fidelity label on each: **[INLINE]**.

## R193 — S3 ratified; the commit-message correction; cap recommendation recorded [INLINE]

> R193 — S3 CLOSED, PASS — one sizing pass as ruled, and the pass itself
> vindicated R179's sequencing with numbers: sizing on the synthetic
> would have over-sized the cap ~22% (the +6.93% real-vs-synthetic gap
> plus the 1.22-vs-1.06 fragmentation tax). (a) The 528eb37 correction is
> handled exactly per R188(b): the commit is immutable history, the
> correction lives in the log + prereg row with the cross-reference, and
> the ledger gains instance #12 — a synthetic-sourced number written
> unlabelled into a commit message; the standing rule it feeds is
> already on the books (R69/R98: a number states its provenance or
> doesn't ship). (b) The model-limit disclosure is recorded as the
> reference shape for honest fitting: peak-as-max-over-program-points is
> piecewise-planar, single-ray batches can't identify b and c
> separately, the off-ray grid supplies c — stated, costed (nothing, on
> run5's measured ray), bounded (±8.2% off-ray). (c) Architect
> recommendation to the prereg row, recorded not ruled:
> max_edges 4,500,000 / max_nodes 170,000 as proposed — margin sized for
> the two NAMED unknowns (the >1 GiB fragmentation swing, measured
> non-clean threshold 11.49-after-12.35; the co-resident eval child on
> worker_device: cuda absent from every number) rather than padded.
> Values remain the operator's at mint prereg per R179's armed-value
> clause. (d) The prereg row's negative claim is ratified verbatim:
> run5 still does not fit; the cap makes overshoot UNCONSTRUCTIBLE by
> splitting — it does not make the batch small. That sentence is the
> row's most important line; it survives every edit.

## R194 — F2 GO [INLINE]

> R194 — F2 implementation proceeds against the S3 sizing input under
> dispatch_6's F2 text as amended by R191 (deterministic-mode exact
> oracles, never the median statistic), R179 (structural success
> criteria, "got further" banned), and R193's values-as-recommendation.
> The remaining map after F2: F3 (Q-D-1), F4 box payloads, merge
> evidence package with the instance ledger, STOP for the go-line.


# R195–R197 — operator adjudication response, 2026-08-02 (F2 ratified; shakedown authorized; fresh dispatcher)

Verbatim. Fidelity label on each: **[INLINE]**.

**RECOVERY PROVENANCE (R172 precedent, second application).** These three were issued with the
dispatch-7 brief on 2026-08-02 and were **absent from this register** when the dispatch-7 dispatcher
verified R118–R197 on 2026-08-04 — the file ended at R194. The dispatcher **refused to reconstruct
them from the brief's paraphrase** (a brief's summary of a ruling is not the ruling, and three
manufactured `[INLINE]` rows would corrupt the exact property this register holds), opened
`Q-D7-1` as BLOCKING, and stopped. **The operator then supplied all three verbatim**, and they are
transcribed here from that supply. R195's arithmetic had already been independently corroborated by
measurement before its text arrived (`plan/freeze_verify.py` at `c04137d`: 48 OK / 16 MISMATCH /
0 MISSING across 64 unique paths, 2 duplicates agreeing — the quoted numbers exactly).

**Worth recording, because it is the near-miss:** R195's freeze SENTENCE already existed in the
tree, authored as a dispatcher disclosure at `wp/WP12R/DFIX_GRANTS.md:496`, which is why the gap
read as "in there in slightly different form". **It was not.** A working document containing the
substance a ruling later adopts is not the ruling — and R196 (which AUTHORIZES an action) and R197
(which SPECIFIES an artifact) had no pre-existing form anywhere in the workspace. The one of the
three whose content pre-existed is the only one that could be corroborated without its text; the two
that grant authority could not. **That asymmetry is the lesson: the rulings most dangerous to infer
are exactly the ones no artifact can corroborate.**

## R195 — F2 ratified; the freeze statement adopted; coupling pinned [INLINE]

> R195 — F2/67120a4 ratified: all three conjuncts of the bound now carry
> detectors (the reverting mutant that survived 131 passed is dead), the
> three-layer arming pin with measured residues (22×→4.16×→sized) stands,
> and the commit's refusal-to-claim ("run5 still doesn't fit; the cap
> makes overshoot unconstructible by splitting") is the honest line —
> R193(d) carried into the tree. The 1.20% headroom TIGHTNESS becomes a
> pinned coupling: a test or prereg note stating the constants move
> together with any re-measured budget, so nobody retunes one leg. The
> freeze statement is ADOPTED VERBATIM for the merge package: "48 of 64
> match, 16 pre-existing attributed drifts, one earlier unattributed
> drift absorbed into a granted row with provenance recorded" — never
> "48/64, up from 47"; a count that improves by absorption is not
> hygiene, and the three-legged trace is what makes the row honest.
> The cd/absolute-path note is logged in the command-hygiene family
> (a command that answers a different question than asked = the false-
> green shape).

## R196 — SHAKEDOWN RUN authorized; the quarantine law [INLINE]

> R196 — Inside S6, after BF2's bench clears its bracket, the burst MAY
> extend into a bounded shakedown run at run5's config shape.
> QUARANTINE, absolute: run-id prefix "shakedown-", outputs outside any
> evidential path, checkpoints throwaway (stamped per LAW-12 like
> everything, but never promotable), NO strength/learning claim ever
> cites it, and no number from it enters prereg grounds except
> OPERATIONAL measurements (throughput, memory, stall/watchdog/eval
> soak) explicitly labeled shakedown-sourced. What it buys is real:
> LAW-16 legs soaking for hours, CS2 checkpoints appearing on cadence,
> eval rounds firing live, the cap's fire-rate counter under production
> load — the exact operational unknowns the mint would otherwise meet
> first. What it can never buy is evidence about the NET. Sequencing
> unchanged: merge → box preflight → prereg → MINT run5 are unmoved;
> the shakedown runs beside the paperwork, not instead of it.

## R197 — fresh dispatcher for S6+S7; rollover at merge [INLINE]

> R197 — The D-fix session's context is spent; S6+S7 run under a fresh
> dispatcher context with a self-sufficient dispatch (below). Same
> verify-don't-trust entry (R118–R196), same instance-ledger duty
> (now 12+ entries, one line each, beside the roster at go-line).


# R198–R200 — operator adjudication response, 2026-08-04 (recovery ratified; ADJ-ZERO-GAMES framed)

Verbatim. Fidelity label on each: **[INLINE]**.

## R198 — recovery + withdrawals ratified; the corroboration asymmetry recorded [INLINE]

> R198 — The R195–R197 gap: my delivery error again (chat-issued, append
> assumed), caught by the same fidelity law, recovered under R172's
> provenance pattern — second application, ratified. The asymmetry is
> ADOPTED into the handoff lineage verbatim: rulings that authorize
> actions or specify artifacts have no pre-existing form and cannot be
> corroborated by measurement — they are the MOST dangerous to infer and
> the first to verify. Q-D7-2/Q-D7-3 withdrawals ratified as modeled:
> arithmetic on a dead producer is not a rate, and telemetry correctly
> reporting an empty buffer is not an alarm about the monitor. The ages
> map lesson (read the entry you need, then the other three) rides the
> ledger. LAW-16's first production firing — armed, registered code 42,
> snapshot written, on REAL staleness — is recorded as the run3-class
> defense proving itself live. The cap-survival non-claim is correct by
> R179's own symmetry: a death that wasn't a memory death refutes
> nothing.

## R199 — ADJ-ZERO-GAMES: mint-blocking, NOT merge-blocking; diagnosis framed [INLINE]

> R199 — The zero-games defect (inference_dispatch and selfplay_drain
> never tick, zero games, both shas) is MINT-BLOCKING — a run that plays
> no games cannot train — and NOT merge-blocking: Q-D7-4 exonerates the
> D-fix delta, and the 25-commit stack's verification is tree-level.
> Named for the diagnosis, because the report's own record contains the
> lever: at 982da03 REPRO A previously reached a training step
> attempting 9.18 GiB — a batch existed, so games existed — and the same
> sha now produces zero. Same tree, two behaviors ⟹ the delta lives
> OUTSIDE the tree: harness invocation, feed path (did earlier bursts
> feed through a path this boot doesn't?), box state, pins, or CUDA-in-
> worker. Prereg'd suspects, in order: (1) CUDA initialization inside
> spawned/forked selfplay inference workers (WPMAIN's local launch
> played games on CPU — the loop works; the box adds GPU-in-children,
> the classic fork/CUDA hang, and a dispatch thread that never ticks
> ONCE is an init hang's signature, not a slow loop's); (2) the
> invocation/environment diff between the Repro-A session and this one,
> derived from both sessions' recorded provenance. Diagnose-first, one
> discriminator per suspect, verdict before any fix. Dispositions:
> D's roster cell closes on its STRUCTURAL criteria (tree-level, R179's
> own design); production corroboration (burst under load, cap
> fire-rate) is OWED AT BOX PREFLIGHT, which ADJ-ZERO-GAMES now gates.
> Q-TRAIN-STEPS-FLOOR annotated blocked-on-this. S6.3–6.5 proceed where
> independent (eval-wall, liveness, M-4 don't need self-play games).
> Shakedown deferral RATIFIED — R196's purchase list is empty at zero
> games; it unblocks the moment this resolves. Merge path unchanged:
> S7 assembles, STOP for the go-line.

## R200 — zero-games: three discriminators, answerable before any fix [INLINE]

> R200 — Architect's suspect-(1) framing error recorded on the ledger
> (generic pattern over derived tree; the dispatcher's correct-then-test
> handling ratified). Next steps, all diagnosis, in order of information
> per minute: (1) STACK DUMP the hung burst — py-spy dump + native
> stacks (gdb/eu-stack) on the live process: where exactly
> collect_graph_data waits, where inference_dispatch's thread is, whether
> Rust worker threads EXIST. One attach answers the causal direction.
> (2) RE-READ Repro A's and S3's own recorded samples/artifacts: did
> inference_dispatch tick in THOSE runs? If it never ticked there either
> while training stepped, the batches were fed by burst-override or
> harness, zero-games is the box's permanent untested state, my Q-from-
> last-turn resolves "never", and the defect is re-framed from
> regression to first-contact. Derive the exact invocation (preflight
> burst mode vs plain mantis.run, config, env) of Repro A and both S3
> boots from their provenance records — this IS suspect (2), executable
> from artifacts without the box. (3) Runner startup census in the
> event stream of the hung arms: which start/registration events fired,
> in what order — an init-order deadlock or a blocking resource load
> (corpus/bank path) at first collect leaves a fingerprint there.
> Verdict rule stated before running, per house law. ADJ-ZERO-GAMES
> remains mint-blocking-not-merge-blocking; the R199 dispositions
> stand; S6.3–S7 proceed in parallel.


# R201–R202 — operator adjudication response, 2026-08-04 (first-contact ratified; localization without ptrace)

Verbatim. Fidelity label on each: **[INLINE]**.

## R201 — first-contact ratified; the shared error; S3's provenance question [INLINE]

> R201 — (a) Verdict NEVER consumed; ADJ-ZERO-GAMES re-framed to
> first-contact: "self-play has not yet been made to work on this box."
> Mint-blocking, not merge-blocking, unchanged. (b) The falsified lever
> was built by BOTH of us — I inferred games from the OOM in R199's own
> text; the dispatcher reported it as demonstrated; both are the Q-D7-2
> class one level up (a producer inferred from apparent output instead of
> its liveness record). Two ledger lines, one each. CARD-RUN5-GPU-OOM's
> diagnosis stands unshaken — the OOM was real, in the training step,
> which is what F1/F2 fixed. (c) MANDATED, cheap, from artifacts: what
> fed S3's buffers? If the burst-override feed, the cost model's
> bytes-per-edge physics is untouched (E→bytes doesn't care who made the
> batch) but the E-DISTRIBUTION grounds under the cap-value
> recommendation inherit a provenance label — the prereg row states its
> feed source either way. (d) The Repro-A staleness-disarmed delta:
> recorded, incidental, correct to not chase.

## R202 — localization without ptrace: the producer narrates its own startup [INLINE]

> R202 — Ptrace denial does not block this; it redirects it to the
> LAW-18 answer we should want anyway. Three moves, in order:
> (1) VERIFY the local premise — R199 asserted "the local launch played
> games on CPU"; derive it from WPMAIN's recorded event stream
> (game_complete > 0?) instead of trusting my sentence. If local is ALSO
> zero-games and trained anyway, the feed question re-opens everywhere.
> (2) THE CHEAP DISCRIMINATOR, no ptrace, no code: rerun the burst with
> OMP_NUM_THREADS=1 / torch.set_num_threads(1). Grounds: the tree
> already contains tests/train/test_heartbeat_watchdog.py's
> GIL-starvation subject ("seq freezes, supervisor declares it stale" —
> R46's flaky test), the box is a many-core machine where torch's
> default intra-op pool maximizes GIL contention, 29-of-34 threads sit
> in futex_wait, and the failure is identical on CUDA and CPU arms —
> device-independent, load-shaped. If games appear single-threaded,
> the defect is named and the fix is a bounded thread/GIL discipline,
> not a hunt. Verdict rule before running. (3) If (2) refutes:
> STARTUP NARRATION lands in-tree as PERMANENT instrumentation, not a
> throwaway patch — the runner emits registered events for its own
> lifecycle (started, workers spawned N, first game loop entered, first
> inference enqueued/served, first record drained), LAW-18/LAW-07 shape,
> mutation-tested; then one box run reads the fingerprint. That
> instrument should exist regardless — a producer whose startup is
> invisible is how this cost four sessions. Ptrace capability as a
> BOX-PROVISIONING option (cap-add or a different instance) is routed
> to the operator as fallback, not prerequisite.


# Operator adjudication response, 2026-08-04 (R203 — three-way split; reduced-latitude dispatcher mode)

Verbatim. Fidelity label: **[INLINE]**.

## R203 — three-way split; reduced-latitude dispatcher mode; channel/rename deferral confirmed [INLINE]

> R203 — Remaining work splits into three single-problem dispatches (8A
> zero-games, 8B box payloads, 8C evidence package) run under REDUCED-
> LATITUDE MODE for a non-Claude dispatcher: no new rulings, no
> paraphrased rulings (quote verbatim or STOP), no judgment calls beyond
> each dispatch's enumerated latitude, every ambiguity → STOP + report,
> queue rows verbatim always, forbidden-actions list binding. 8B may
> share 8A's box session once 8A's verdict lands; 8C runs parallel,
> local. Channel-count and architecture renames CONFIRMED post-mint
> (R117/R140; F-14's number is 8-of-18, not 4 — the doc derives from the
> register); 8C verifies the R140 prep artifacts exist and reports,
> nothing more.


# R204–R206 — operator adjudication response, 2026-08-04 (Step 2 REFUTED; defect GLOBAL; Step 3 GO local-first; CARD-EVAL-CORESIDGENCY opened)

Verbatim. Fidelity label on each: **[INLINE]**.

## R204 — Step 2 adjudicated REFUTED; the defect is GLOBAL; questions answered [INLINE]

> R204 — (a) Q1: REFUTED stands. The pre-written rule's subject was the
> tick, and it did not tick; the divergent failure mode does not rescue
> the GIL hypothesis because Step 1's LOCAL-GAMES-NO shows the defect on
> a CPU-only local machine where no CUDA pressure exists — the failure
> is device-independent BY the local evidence, and single-threading
> changed nothing about the producer. GIL-starvation closed.
> (b) Q2 resolved from the record: dispatch 7's CPU arm was a
> TOOLING-MINTED twin config, "differing from run5.yaml in exactly two
> keys by diff — train.device and eval.worker_device" (dispatch 7's own
> report, verbatim); no repro.sh flag existed because none was used.
> Moot now — no further CPU arm is needed under (a). The 8A dispatcher's
> refusal to mint one without sign-off was correct latitude discipline.
> (c) ADJ-ZERO-GAMES re-frames again: box-first-contact → GLOBAL
> first-contact — production self-play has never run in ANY
> python -m mantis.run invocation; only eval-path games have ever
> played. Still mint-blocking, still not merge-blocking (tree-level
> verification unaffected). The R42-compliant register append and the
> STOP conduct are ratified — reduced-latitude mode performed exactly
> as designed.

## R205 — Step 3 GO, re-scoped LOCAL-FIRST; the local-feed census rides it [INLINE]

> R205 — Q3 authorized with one change that makes it 10× faster: the
> defect reproduces LOCALLY, so the narration fingerprint is read on the
> LOCAL machine (CPU, smoke config, minutes per iteration) — the box is
> for CONFIRMATION after a fix, not for diagnosis. Phase 0 of Step 3,
> from artifacts before any code: what did local training actually do —
> training_step event counts and batch provenance in the WPMAIN launch
> and the CS 475s runs. If training_step == 0 locally, the "trains its
> 200 steps" claims get the R96 correction (loop-steps, not training)
> and the cold-buffer gate's behavior is documented; if > 0, name the
> feed. Then the narration lands per 8A Step 3's spec (registered
> lifecycle events, LAW-07/LAW-18, mutation-tested, house pipeline),
> fingerprint read locally, fix routed on the fingerprint. The
> instrumentation is permanent either way.

## R206 — CARD-EVAL-CORESIDENCY opened: measured, prereg-relevant [INLINE]

> R206 — Step 2 measured the co-resident eval child at 8.21 GiB on the
> 16 GiB box before trainer allocation — the exact named unknown R193's
> margin was sized around, now with a number that does not fit under
> that margin. Card scope: characterize eval-child steady VRAM (load
> spike vs resident), then the prereg decides with grounds — smaller
> co-residency, staggered scheduling, or eval.worker_device: cpu for
> run5 (LAW-15 note: deploy-matching binds the eval REGIME, sims and
> semantics — device changes wall, not verdicts; state it in the row).
> The EDGE-CAP row gains a co-residency line either way. Mint-blocking
> ONLY through the prereg row being honest; the measurement rides any
> box session.


# R207–R209 — operator adjudication response, 2026-08-04 (R204(c) FALSIFIED; phantom-beat blocker; Step 3 sequenced)

Verbatim. Fidelity label on each: **[INLINE]**.

## R207 — R204(c) FALSIFIED and corrected; both census errors ledgered [INLINE]

> R207 — My "GLOBAL first-contact / self-play has never run in ANY
> invocation" is FALSIFIED by measurement (13 games, corpus_selfplay_frac
> 1.0, 5 real training steps) and corrected per R96 wherever cited —
> architect ledger instance. Root of the error chain, both recorded:
> Step 1's LOCAL-GAMES-NO read the wrong counter (game_complete is never
> emitted on this path; games live in iteration_complete.games_total),
> and R199/R200's "inference_dispatch never ticked" signature was
> diagnostic garbage — the source never ticks in HEALTHY runs either.
> Both are instances of the same law: a census must first verify its
> counter is a live producer (LAW-07 applied to censuses). The
> dispatcher's STOP on contradicting a ruling premise: exactly right,
> third time running.

## R208 — MMINT-BLOCKER: phantom heartbeat sources under an armed abort [INLINE]

> R208 — inference_dispatch and selfplay_drain are registered heartbeat
> sources that no producer ever ticks on the production path, sitting
> under a staleness watchdog that, when armed, KILLED a burst at its
> 1800 s deadline (rc 34, code 42) — a false-positive abort that will
> execute every healthy run5 at minute 30. This is LAW-07's founding
> class (phantom gate input arming an abort chain) live in the tree.
> MINT-BLOCKING, own small card CARD-PHANTOM-BEAT, rides the stack:
> census EVERY registered heartbeat source against its actual producer
> (mutation-tested — tick the producer, see the age drop); each phantom
> source is either WIRED to a real producer or REMOVED from watchdog
> jurisdiction with grounds, never left registered-and-dead; the
> staleness watchdog's arming audit gains a producer-liveness conjunct
> so a source with no producer cannot be armed (the R79/LAW-07 shape).
> The rc-34 box death is re-labeled in the record: not evidence about
> games — evidence about the watchdog.

## R209 — the fourth cell; game_complete folds into narration; Step 3 sequenced [INLINE]

> R209 — Next action, LOCAL, before any instrumentation: mint the run5
> CPU twin via tooling (dispatch-7 precedent — exactly train.device +
> eval.worker_device flipped; R103 pattern, header-truth, twin is a
> diagnostic config not a new key) and run it locally ~20 min. Verdict
> rule now: GAMES (iteration_complete.games_total > 0) ⇒ box-specific —
> suspects become box assets/env (corpus or opening files present
> locally but absent on box FIRST — a sampler waiting on a missing
> asset blocks exactly like this and violates LAW-14's fail-loud);
> NO GAMES ⇒ config-shaped and locally reproducible ⇒ BISECT the
> smoke↔run5 config delta by minted intermediates (tooling only, R1),
> halving until the blocking key(s) are named. Either way the defect
> gets named before fixed. game_complete-per-game emission: YES, an
> instrumentation gap — folds into Step 3's narration scope (LAW-18)
> together with the lifecycle events, which land AFTER the delta
> verdict so the narration is built knowing what it must witness.
> Box stays untouched until a local verdict exists.


# R210–R213 — operator adjudication response, 2026-08-04 (Key-1 named; narration authorized; box measurement owed; R96/R167 compliance)

Verbatim. Fidelity label on each: **[INLINE]**.

## R210 — ADJ-ZERO-GAMES re-framed; Key-1 named; Step 3 narration authorized as the fix chunk [INLINE]

> R210 — The R209 bisect is accepted: the "zero-games" diagnosis on the CPU path was an INVISIBILITY artifact, not absence. Key 1 (train.log_interval=1000 gating iteration_complete, the sole carrier of games_total) is measured-confirmed: intermediate 1 produced actor_sync=13, learner_step=12 at log_interval=1000 with iteration_complete:0 — games WERE produced, the run was healthy but invisible. The entire R199/R200 signature (game_complete:0, training_step:0, inference_dispatch age≈wall, buffer_size:0 in warmup) is exactly what an invisible-but-healthy run5 looks like. Fix route (a) — decouple iteration_complete from log_interval (emit per coordinator step; training_step alerting stays gated) — is AUTHORIZED as the structural fix and is the head of the Step 3 narration chunk (R209: narration lands after the delta verdict; the delta verdict is done). The narration chunk is ONE DESIGN, scope: (i) decouple iteration_complete emission from _run_log_interval, (ii) registered lifecycle events (runner_started, workers_spawned, game_loop_entered, first_inference_enqueued/served, first_record_drained), (iii) per-game game_complete emission (R209 fold-in), all LAW-07/LAW-18, mutation-tested, house pipeline. Route (b) — lower run5's log_interval as the fix — REJECTED: it is a workaround that leaves the coupling defect in place; games_total is a per-iteration counter, not a training-logging event. The mint's log_interval value for logging/alerting cadence is a prereg consideration, not decided here.

## R211 — Key-2 (n_simulations=50 CPU-slow) closed [INLINE]

> R211 — selfplay.mcts.n_simulations=50 producing no completed games in 5 min on CPU is a CPU-only performance characteristic, not a production defect (run5 targets GPU). Closed. CPU repro of run5 self-play at production sims is not supported; do not attempt it. No mint action, no card.

## R212 — Box game-production must be MEASURED, not inferred [INLINE]

> R212 — The agent's claim "the box may have been producing games all along" is an INFERENCE about the box from a local CPU result — R199/R207 class: never infer a producer from apparent output. The local CPU twin producing games does NOT establish the box's GPU path produces games. 8B must run a diagnostic box burst (low log_interval=10, diagnostic config in /tmp-class, NOT the mint config — log_interval is a reporting knob, not an armed value per STATE §4, so no armed-value issue) to MEASURE whether GPU run5 produces games, reading iteration_complete.games_total and actor_sync. Box access under the existing R142 grant. Verdict rule fixed: games_total>0 OR actor_sync>0 ⇒ box produces games, ADJ-ZERO-GAMES closes as "invisible-games" on both paths; games_total=0 AND actor_sync=0 ⇒ box genuinely produces zero, a second distinct defect — re-bisect on box. The re-framing of the project history from "zero-games" to "invisible-games" is CONDITIONAL on this measurement and must not be written into artifacts as settled before it returns.

## R213 — R96/R167 compliance owed before any further work [INLINE]

> R213 — The agent reported findings in the dispatch report but did NOT append them to wp/WP12R/ADJUDICATION_QUEUE.md verbatim (R167: findings live in the queue, not only in reports) and did NOT correct downstream artifacts (R96: correct in EVERY downstream artifact). Required before the narration chunk begins: (i) append the bisect table + Key-1/Key-2 mechanism text verbatim to ADJUDICATION_QUEUE.md; (ii) add a falsified-claims ledger entry to docs/registers/falsified.md AND plan/ARCHITECT_STATE_2026-08-04.md §7 — "run5 CPU twin produces zero games (R207/R209 premise)" is FALSIFIED (R210: games produced, invisible); the box half stays marked UNRESOLVED pending R212; (iii) reframe STATE §2A "ADJ-ZERO-GAMES" to reflect: CPU path = invisibility (resolved by R210), box path = measurement-owed (R212), NOT "zero games (box, CUDA and CPU twin)"; (iv) report the corrections back. The instance goes on the instance ledger (what: zero-games misdiagnosis root cause = log_interval gating; who: dispatch 8A via R209 bisect; how caught: intermediate 1 actor_sync>0 with iteration_complete=0; where corrected: falsified.md, STATE §2A/§7, ADJUDICATION_QUEUE).


# Operator adjudication response, 2026-08-04 (R214 — routing: both lanes in parallel; narration DESIGN starts now)

Verbatim. Fidelity label: **[INLINE]**.

## R214 — routing: both lanes in parallel; narration DESIGN starts now [INLINE]

> R214 — The R213 compliance is accepted (register R210–R213 ×1 each; F-43 in falsified.md with box-half correctly marked UNRESOLVED; STATE §2A reframed; queue row verbatim). The "begin narration DESIGN or route R212 first" question is a false dichotomy — R210 already sequenced them parallel. BOTH proceed now: (a) Step 3 narration DESIGN chunk (R210 scope i–iii, house pipeline, fresh contexts, max 2 loops) starts immediately, local, no box dependency; (b) R212 box diagnostic burst routes to 8B in parallel under the R142 grant. The narration DESIGN is NOT gated on the box measurement: the iteration_complete decoupling fix is required on both paths, and R212 only determines whether a second distinct box defect exists alongside the invisibility defect. Rider: the narration DESIGN's ORACLE-WRITE stage must assert, mutation-tested, that iteration_complete emits on every coordinator step at run5's log_interval=1000 (i.e., the gate is removed for games_total/iteration_complete while training_step alerting stays gated) — this is the falsifying test for the fix itself. No new rulings, no paraphrase, REDUCED-LATITUDE holds.
> Dispatcher: append R214 verbatim, then mint two dispatch prompts in parallel — (a) narration DESIGN chunk, (b) R212 box burst to 8B.

# R215–R217 — operator adjudication response, 2026-08-04 (R207 mechanism corrected (R96); F-43 corrected; the production-visible games signal is actor_sync; narration chunk re-authorized with corrected premises; run.py:349 grant boundary)

Verbatim. Fidelity label: **[INLINE]**.

## R215 — R207 mechanism corrected (R96); F-43 corrected; the production-visible games signal is actor_sync [INLINE]

> R215 — R207's core falsification (self-play ran in every healthy invocation; the GLOBAL "never ran" claim is FALSE) STANDS, confirmed by actor_sync=13 at bisect intermediate 1. R207's two visibility-mechanism attributions are BOTH corrected, per R96, in every downstream artifact: (a) "game_complete is never emitted on this path" → FALSE: game_complete IS emitted at pool_drain.py:177 (golden-pinned, C-03 test_pool_drain_parity.py:332-352 + J-05 test_selfplay_census.py:398-421) but DROPPED in production because the production WorkerPool is constructed with sink=None at run.py:349; (b) "games live in iteration_complete.games_total" → FALSE as a production visibility claim: iteration_complete is gated by log_interval=1000 (step.py:576, R210) and does not fire before step 1000. The production-visible games signal is actor_sync (trainer stepping ⇒ buffer has data ⇒ games produced), NOT iteration_complete.games_total (gated) and NOT game_complete (dropped by sink=None). F-43's wording "game_complete is never emitted to the sink (R207)" is corrected to "game_complete IS emitted (pool_drain.py:177, golden-pinned) but dropped because pool._sink=None at run.py:349." R207 is not paraphrased or rewritten in the register — this ruling corrects its mechanism; the register entry stands verbatim with R215 as its correction of record. Ledger instance: what = R207 double mechanism misdiagnosis (game_complete "never emitted" + games "live in iteration_complete"); who = dispatch 8A narration DESIGN STOP; how caught = pool_drain.py:177 emit + golden pins contradict "never emitted", R210 contradicts "live in iteration_complete"; where corrected = this ruling, F-43, STATE §7, ADJUDICATION_QUEUE.

## R216 — Narration chunk re-authorized with corrected premises; parts (ii)+(iii) share run.py:349 sink injection [INLINE]

> R216 — The narration chunk (R210) is re-authorized with corrected premises. Scope, corrected: (i) decouple iteration_complete emission from _run_log_interval in coordinator/step.py — emit iteration_complete (carrying games_total) per coordinator step; training_step alerting stays gated by log_interval — train-side, independent, no run.py change. (ii)+(iii) share a single root: the production WorkerPool at run.py:349 is built with sink=None, so game_complete (already wired at pool_drain.py:177, golden-pinned) AND any selfplay lifecycle events are dropped. The fix for (ii) and (iii) is ONE change: inject a sink at run.py:349. game_complete needs NO re-wiring and must NOT break the C-03/J-05 goldens — the event is already correct, only undelivered. Part (ii) lifecycle events (runner_started, workers_spawned, game_loop_entered, first_inference_enqueued/_served, first_record_drained) register through the selfplay-local EventSink Protocol (pool_hooks.py:32-40) — NOT mantis.train.emit.EventSink; routing them through the train sink would create the forbidden selfplay→train import edge. The bridge/adapter, if one is needed, lives in mantis.run (the composition root, which already imports both sides); whether run_safety.sink is directly injectable (structural Protocol compatibility + payload-shape match) or needs a mantis.run-local adapter is the DESIGN stage's first decision, with the constraint FIXED: no src/mantis/selfplay/ file imports mantis.train.emit; the DAG edge stays absent. All parts LAW-07/LAW-18, mutation-tested. House pipeline: DESIGN → REVIEW-design → ORACLE-WRITE → IMPL → REVIEW-impl → RED-TEAM, fresh contexts, max 2 loops.

## R217 — run.py:349 grant boundary: narration owns sink=, R208 owns heartbeat= [INLINE]

> R217 — run.py:349 is a shared construction site for two distinct defects. The narration chunk's grant covers ONLY the sink= keyword argument at run.py:349 (and the pool_hooks.py/pool_drain.py/step.py edits named in R216). The narration chunk must NOT touch the heartbeat= keyword — that is R208's subject (CARD-PHANTOM-BEAT: selfplay_drain heartbeat dropped because pool._heartbeat=None at the same site). The two chunks land on wp12r-scratch sequentially (narration first, R208 after); each touches a different keyword, so no merge conflict. The narration chunk's IMPL notes must state explicitly that heartbeat= was left at None untouched, and RED-TEAM must probe that the narration change did not alter heartbeat behavior. R208 remains mint-blocking and rides the stack.
> Dispatcher: append R215–R217 verbatim; drive R96 corrections (R207 mechanism wording in falsified.md F-43, STATE §7, ADJUDICATION_QUEUE — game_complete "never emitted" → "emitted but dropped by sink=None"; iteration_complete.games_total "carries games" → "gated by log_interval, not production-visible"); then re-dispatch the narration DESIGN with the corrected R216 premises.

# Operator adjudication response, 2026-08-04 (R218 — REVIEW-design PASS-WITH-FIXES accepted; ORACLE-WRITE authorized; Q-O-TWO-POOL-READS collapse is a semantic change the oracle must assert)

Verbatim. Fidelity label: **[INLINE]**.

## R218 — REVIEW-design PASS-WITH-FIXES accepted; ORACLE-WRITE authorized; Q-O-TWO-POOL-READS collapse is a semantic change the oracle must assert INLINE

> R218 — The REVIEW-design verdict (PASS-WITH-FIXES) is accepted. All ten fixes (C1 test-impact inventory, C2 cost correction + Q-O-TWO-POOL-READS collapse, S1 O2/O3 exclusion, S2 access path (α), S3 bind() location, M1–M5) are verified present in DESIGN_NARRATION.md. The design is faithful to R210/R214/R215/R216/R217. ORACLE-WRITE is authorized to proceed (fresh context, house pipeline). Two riders: (1) the Q-O-TWO-POOL-READS collapse is a SEMANTIC CHANGE — the target_integrity snapshot and the mcts_mean_depth/cluster-stats snapshot become ONE atomic read instead of two microseconds-apart reads that could straddle a game boundary. The oracle for part (i) MUST assert that emit_iteration_complete_event uses the snapshot passed from _target_integrity_report (not its own pool.runner_stats() call), and a falsifying mutation that re-introduces the second runner_stats() call MUST turn RED. The design's wording "the collapse does NOT change the straddle" is corrected here: it ELIMINATES the straddle (both reads are now atomic on the same snapshot); this is more correct, not neutral, and the IMPL notes must state it as a behavior change, not a no-op. (2) C1's test test_log_interval_boundaries_are_evaluated_per_training_step — after the update, the test NAME describes only the training_step/monitor_gates arms (which stay log_interval-gated); the iteration_complete arm now asserts per-burst cadence. The IMPL docstring update must reflect this split; renaming the test is IMPL's call, not required. The first_record_drained (β) semantics (first non-empty drain) and the emit_training_events wrapper retention are accepted as designed.
> ORACLE-WRITE stage is authorized. Two oracles per R214 §7: (1) part (i) iteration_complete emits every coordinator step at log_interval=1000 with re-couple mutation turning RED; (2) parts (ii)+(iii) N games yield N delivered game_complete events with sink=None revert turning RED. Both mutations driven both ways. C-03/J-05 goldens stay GREEN. The Q-O-TWO-POOL-READS collapse oracle (rider 1) folds into oracle (1).

# Operator adjudication response, 2026-08-04 (R219 — ORACLE-WRITE accepted; IMPL authorized; O-N1b precondition-gate noted)

Verbatim. Fidelity label: **[INLINE]**.

## R219 — ORACLE-WRITE accepted; IMPL authorized; O-N1b precondition-gate noted [INLINE]

> R219 — The ORACLE-WRITE stage is accepted. Two oracle files written to hexo-mantis (tests/train/test_iteration_complete_decoupling.py, tests/selfplay/test_game_complete_delivery.py), 5 tests total, RED/GREEN at HEAD verified: 3 RED (O-N1 iteration_complete below log_interval, O-N1b collapse, O-N2a sink=None AST), 2 GREEN (O-N1 alerting-stays-gated conjunct, O-N2b N-games-yield-N-delivered with stub sink). C-03/J-05 goldens GREEN (34 passed). Floor 2690 → 2695, non-decreasing. R217 boundary holds (O-N2a inspects sink= only, never heartbeat=). Tree: M docs/registers/falsified.md + 2 untracked test files, no production code touched, no commit. IMPL is authorized (fresh context, house pipeline, max 2 loops). Rider: O-N1b's RED at HEAD is a PRECONDITION-gated RED (precondition: len(iter_events) == 1 fails because O-N1 hasn't landed), not the collapse assertion itself firing — this is a correct oracle pattern (can't test the collapse until the decoupling exists), but IMPL must verify that after landing O-N1, O-N1b's REAL assertion (one runner_stats() call per emit) fires and the falsifying mutation (re-introduce the second call) turns it RED. IMPL notes MUST state: (a) heartbeat= left at None untouched (R217), (b) the Q-O-TWO-POOL-READS collapse is a semantic change — eliminates the straddle, both reads atomic on one snapshot (R218 rider 1), (c) the test_log_interval_boundaries_are_evaluated_per_training_step update per DESIGN §3.6 (iteration_complete → [4,8,12,16,20], training_step/monitor_gates stay [5,10,15,20]). Scratch branch only; PASS verdict + gate evidence timestamped before any commit (R42).

# R220–R221 — operator adjudication response, 2026-08-04 (R212 verdict ratified; wedged-drain kill is operator's call, carded as lifecycle defect)

Verbatim. Fidelity label: **[INLINE]**.

## R220 — R212 verdict: box produces games (actor_sync=5); ADJ-ZERO-GAMES closes as invisible-games on both paths [INLINE]

> R220 — The R212 box measurement is COMPLETE. actor_sync=5 satisfies the fixed verdict rule (games_total>0 OR actor_sync>0 ⇒ box produces games). ADJ-ZERO-GAMES is CLOSED as an invisibility defect on BOTH paths (CPU: R210 log_interval gating; box: same + sink=None drop, R215/R216). The "zero-games" → "invisible-games" re-frame is now UNCONDITIONAL — the condition R212 placed on it (box measurement) is satisfied. The re-frame may be written into artifacts as settled. Falsified-claims ledger: the box half of F-43 ("BOX HALF UNRESOLVED pending R212") is now RESOLVED — update F-43 to drop the UNRESOLVED marker per R96. STATE §2A reframes to: ADJ-ZERO-GAMES CLOSED (R210/R215/R216/R220); the box was never zero-games, it was invisible-games (log_interval gating + sink=None drop). R213(ii) instance-ledger entry updates: the box half is no longer "pending R212" — it is closed by this measurement.

## R221 — wedged-drain kill is operator's call; finding carded as lifecycle defect [INLINE]

> R221 — The wedged drain (PID 160990, 16 min past SIGTERM, past its own final_eval_drain_timeout_sec=900 ceiling, holding 9798 MiB GPU; timeout 1200 PID 160985 has no --kill-after so it will not escalate to SIGKILL) is the same lifecycle-defect class as the orphan PID 160143 the operator previously authorized killing. The R212 measurement is immutable (actor_sync=5 in the JSONL); killing the process cannot change the verdict (R220). The kill decision is the OPERATOR's alone (HARD LIMITS: host/box access is per-dispatch; R142 covered the burst, not a separate process-kill). RECOMMENDATION: kill the chain (SIGKILL 160990 if SIGTERM has no effect, then 160985) — the drain has exceeded its own ceiling and is wedged, not draining; leaving it holds the GPU indefinitely for no measurement value. The finding is CARDED as a lifecycle defect (drain-exceeds-ceiling + timeout without --kill-after), adjacent to R208's scope but distinct (drain-hang, not phantom-heartbeat) — owner: post-mint queue, NOT the narration chunk (the narration chunk's scope is fixed by R210/R216; this is a shutdown-path defect, not an event-visibility defect). The agent's report must record: the kill (if authorized), the JSONL path (immutable evidence), the actor_sync=5 value, and the drain-exceeds-ceiling finding verbatim in ADJUDICATION_QUEUE.md.

# Operator adjudication response, 2026-08-04 (R222 — R212 lane-b COMPLETE; ADJ-ZERO-GAMES closed; one mint blocker down)

Verbatim. Fidelity label: **[INLINE]**.

## R222 — R212 lane-b COMPLETE; ADJ-ZERO-GAMES closed; one mint blocker down [INLINE]

> R222 — The R212 box measurement is ACCEPTED. actor_sync=5 > 0 fires the R212 verdict rule; ADJ-ZERO-GAMES is CLOSED as invisible-games on both paths (R220). The "zero-games → invisible-games" re-frame is UNCONDITIONAL — R96 corrections applied to F-43 (box-half RESOLVED), STATE §2A/§7 (CLOSED), ADJUDICATION_QUEUE (verbatim measurement + kill record). The R221 kill (operator-authorized, 16:49:51Z) is recorded; the drain-exceeds-ceiling + timeout-without---kill-after lifecycle defect is carded to the post-mint queue (NOT the narration chunk; shutdown-path, not event-visibility). iteration_complete=0 at log_interval=10 on the box CONFIRMS the R210 decoupling defect exists on both paths — the narration chunk's scope-(i) fix is required on both, exactly as R214 stated. STATE §3 item 1 (R209 fourth-cell → verdict → named blocker → fix) is COMPLETE. Mint blockers remaining: CARD-PHANTOM-BEAT (R208) only — the narration chunk must still land, then R208 rides the stack. No second distinct box defect.

# Operator adjudication response, 2026-08-04 (R223 — Narration chunk IMPL PASS; chunk ready for scratch-branch commit; R208 is the last mint blocker)

Verbatim. Fidelity label: **[INLINE]**.

## R223 — Narration chunk IMPL PASS; chunk ready for scratch-branch commit; R208 is the last mint blocker [INLINE]

> R223 — The WP12R Step 3 narration chunk IMPL is ACCEPTED (PASS). All three parts landed: (i) iteration_complete decoupled from _run_log_interval, emits per-burst at the O6 return, training_step alerting stays gated (R210); the Q-O-TWO-POOL-READS collapse passes the _target_integrity_report snapshot into emit_iteration_complete_event as the rstats kwarg — ONE atomic runner_stats() call per emit, eliminating the straddle (R218 rider 1, a semantic change correctly stated in IMPL). (ii)+(iii) _DeferredSink adapter injected at run.py:380 (was sink=None), bind() after build_run_safety at :549; heartbeat=None UNTOUCHED at :381 (R217 verified); six lifecycle events emitted through the selfplay-local EventSink Protocol (R216 DAG verified — no selfplay→train.emit import). game_complete not re-wired, C-03/J-05 goldens GREEN (34 passed). REVIEW-impl C1 (lifecycle producer tests — tests/selfplay/test_lifecycle_events.py, 6 tests) and S1 (first_record_drained OR-semantics: collected_rows or games_batch) fixed. RED-TEAM FAIL (3 tests — _Pool stubs missing runner_stats() after per-burst emit) fixed by upgrading stubs to PoolTelemetryLike. All falsifying mutations driven both ways: re-couple → O-N1 RED; un-collapse → O-N1b RED; revert sink=None → O-N2(a) RED; each restored GREEN. Gate evidence: 2701 collected (floor 2690 ✓), lint GREEN, pyright 0 errors (4 pre-existing _engine warnings), all affected suites GREEN. The chunk is READY for the scratch-branch commit (R42: PASS verdict + gate evidence timestamped before commit — operator authorizes the commit). STATE §3 item 3 (narration instrumentation) is COMPLETE. Mint blockers remaining: CARD-PHANTOM-BEAT (R208) ONLY — it rides the stack next, touching heartbeat= at the same run.py:381 site (different keyword, no merge conflict per R217).
> Operator: the narration chunk is ready to commit to wp12r-scratch — say the word and the agent stages + commits (R42: timestamp the gate evidence before the commit). After that, R208 (CARD-PHANTOM-BEAT) is the last mint blocker before the merge evidence package.

# Operator adjudication response, 2026-08-04 (R224 — dispatcher errors corrected)

Verbatim. Fidelity label: **[INLINE]**.

## R224 — dispatcher errors corrected (floor 2690 not 2701; 8C working directory + register file named) [INLINE]

> R224 — Both STOP reports are ACCEPTED — the agents did exactly right (R203: STOP on mismatch, never guess). The errors are MINE, not theirs. Three corrections: (1) FLOOR: the on-disk floor is 2690 (tools/ci_gates/test_count_floor.txt, set by commit c04137d). I wrote "Floor 2701" in both dispatches — I confused the collected count (2701, post-narration-chunk) with the floor file. R98 violation (number from memory, not derived from the file at point of use); corrected in both dispatch files. The gate-evidence condition for R208's IMPL is: collected count non-decreasing from 2701 (the post-cebe4cc collected count), floor file stays 2690 unless the IMPL agent ratchets it on PASS (precedent: c04137d ratcheted on F2 close). (2) 8C WORKING DIRECTORY: the 8C dispatch operates in mantis-migration/ (the migration workspace at [REDACTED:abs-home-path:b4672b47]/Work/Hexo/mantis-migration/), NOT in hexo-mantis/. All paths in the dispatch (plan/, wp/WP12R/, plan/rulings_register.md, plan/RUN5_MINT_PREREG.md, plan/freeze_verify.py, etc.) are relative to the migration workspace root. The hexo-mantis repo is a sibling directory used only for the falsified.md edit and code commits. (3) 8C REGISTER FILE: the R-numbered rulings register is plan/rulings_register.md in the migration workspace, NOT docs/registers/falsified.md (which is the F-numbered falsified-claims ledger in hexo-mantis). The grep "every ruling R118–R223 present exactly once" runs against plan/rulings_register.md. Instance ledger: the dispatch errors go on the ledger (what = floor 2690 misstated as 2701 + 8C working directory and register file unnamed; who = architect in dispatch authoring; how caught = 8C + R208 agents STOPPED at entry; where corrected = R224, both dispatch files corrected).

# R225–R226 — operator adjudication response, 2026-08-04 (R208 STOP adjudicated; 8C MERGE_EVIDENCE.md accepted)

Verbatim. Fidelity label: **[INLINE]**.

## R225 — R208 STOP adjudicated: conjunct is composition-root, not watchdog (option A); WIRED confirmed; test :471 stays GREEN [INLINE]

> R225 — The R208 STOP is ACCEPTED — the agent correctly identified the O-P2 / test :471 contradiction. Resolution: **option (A)**. The "producer-liveness conjunct" R208 names lives at the COMPOSITION ROOT (`run.py`), NOT inside `HeartbeatWatchdog.arm()`. The watchdog is correct as written: it trusts the `wired_sources` declaration from the root and treats declared-but-never-beaten sources as wedges (test :471, the F3 carve-out, is CORRECT and stays GREEN). The defect is at `run.py:113` — `_BASE_WIRED_SOURCES = ("train_step", "inference_dispatch", "selfplay_drain")` declares `inference_dispatch` and `selfplay_drain` as "wired unconditionally" but `heartbeat=None` at `run.py:381` means they are NOT wired. The root LIES to the watchdog. After the WIRED fix (inject `heartbeat=_DeferredHeartbeat()` at `run.py:381`, bind after `build_run_safety`), all four sources have real producers and the declaration becomes TRUE. The conjunct is a root-level assertion before `watchdog.start()` (`run.py:~671`): every source in `wired_sources` must have a real producer injection (the `_DeferredHeartbeat` is bound, not `None`). If someone reverts to `heartbeat=None`, the assertion raises → the watchdog cannot arm → O-P2's oracle RED. Existing unit tests (:471, F3) construct `HeartbeatWatchdog` directly with no composition root → no root assertion → stay GREEN. R208's wording "the staleness watchdog's arming audit gains a producer-liveness conjunct" is clarified: "arming audit" means the arming FLOW (root → `watchdog.start()` → `arm()`), and the conjunct lives at the root's invocation site, not inside the watchdog class. The watchdog's `wired_sources` trust model is unchanged and correct.

> Q2: CONFIRMED — the conjunct is arm/construction-time only. Mid-run producer death stays a 42 (a stage that beats then dies is in `beaten_sources()` → not phantom → the normal staleness fire at `:360` catches it — correct wedge detection). RED-TEAM answer: "no, the conjunct does not fire on mid-run death, and that is correct."

> Q3: WIRED CONFIRMED for both `inference_dispatch` and `selfplay_drain`. Both producers are real liveness signals (`inference_server.py:528,745`, `pool_drain.py:55-59`) — just undelivered because `heartbeat=None`. The `_DeferredHeartbeat` adapter mirrors the narration chunk's `_DeferredSink` (R217: separate adapter, `_DeferredSink` untouched, `sink=` at `:380` untouched). No source qualifies for REMOVED — `HEARTBEAT_SOURCES` is name-pinned at `tests/monitor/test_heartbeat.py:40` to the exact 4-tuple, and all four producers are real. The fix is: (1) `_DeferredHeartbeat` class in `run.py` (mirrors `_DeferredSink`), (2) inject at `run.py:381` (`heartbeat=_DeferredHeartbeat()` instead of `heartbeat=None`), (3) bind after `build_run_safety` at the `:549` area (same window as the sink bind), (4) root-level assertion before `watchdog.start()` at `:671` that every `wired_sources` entry has a bound (non-None) producer. Instance ledger: what = R208 O-P2/watchdog-:471 contradiction (conjunct location ambiguous in R208's wording); who = architect (R208 wording) + dispatch agent (caught the contradiction); how caught = test :471 asserts the conjunct's negation; where corrected = R225 (conjunct relocated to composition root), R208 clarified.

## R226 — 8C MERGE_EVIDENCE.md accepted; 2 missing prereg rows flagged [INLINE]

> R226 — The 8C MERGE_EVIDENCE.md deliverable is ACCEPTED. One mint blocker remains (CARD-PHANTOM-BEAT, WAITING(R208)). ADJ-ZERO-GAMES is CLOSED. Prereg: 13 of 15 rows present, 2 MISSING — buffer non-persistence (R178) and R206 co-residency line — listed not authored (correct per 8C's mandate). Freeze re-run: 47 OK / 17 MISMATCH / 0 MISSING (R195 statement quoted verbatim, numbers updated-by-re-run). R140 prep: all three artifacts verify. The 2 missing prereg rows are NOT merge-blocking (R178 is a written ruling already in the register; the R206 line is pending the 8B measurement) — they must be present at MINT authoring, not at the go-line. The architect's go-line (R170) can be issued once R208 lands; MERGE_EVIDENCE.md updates R208 from WAITING to CLOSED at that point.

# Operator adjudication response, 2026-08-04 (R229 — post-merge: 8B findings routed)

Verbatim. Fidelity label: **[INLINE]**.

## R229 — post-merge: three 8B findings routed for prereg authoring; R206 unbounded-growth flagged [INLINE]

> R229 — The WP12-R merge is COMPLETE (R228 go-line, dev == origin/dev == 45a0a81, scratch deleted, pushed per R100). The post-merge STATE doc is delivered (`ARCHITECT_STATE_2026-08-04_POST_MERGE.md`). 8B is DONE — 6 of 7 payloads executed, 1 STOPPED (payload 2, stale brief: ADJ-29 already settled C2 provenance, no winner/loser possible). Three 8B findings are routed for prereg authoring (NOT mint blockers — the merge is done; these are inputs the operator needs at MINT authoring):
>
> (1) **R206 eval-child VRAM (payload 6)** — the prior 8.21 GiB steady-state measurement is SUPERSEDED: at `deploy_sims=150`, eval-child co-resident VRAM grows unboundedly to 13.5 GiB (no-gate baseline 386 MiB). This is material — the operator's R206 co-residency prereg row must reflect the unbounded-growth finding, NOT the old 8.21 GiB figure. The 13.5 GiB vs the 5080's 16 GiB leaves ~2.5 GiB headroom — tight but feasible IF nothing else competes. Recorded in `MEASUREMENT_R206_coresidency.md`. This finding does NOT block MINT, but the operator must decide at prereg authoring whether the unbounded-growth profile is acceptable or whether a cap/conjunct is needed (that decision is OPERATOR-ONLY — armed-value territory). Flagged hardest: it changes the co-residency picture materially.
>
> (2) **best_model.pt unloadable (payload 5)** — M-4 scored round passed all six conditions (a)-(f), abort-10 discharged. `best_model.pt` won't load. Carded, NOT blocking. Owner: post-mint queue (likely CARD-RESUME territory — checkpoint format/loader). Recorded in the 8B report.
>
> (3) **Q-TRAIN-STEPS-FLOOR = 1 step (payload 7)** — post-cebe4cc decouple, the minimum training step floor is 1 (one coordinator step produces one iteration_complete emit). This is the input the prereg `floor cadence` row needs. Recorded in `MEASUREMENT_QTRAIN_FLOOR.md`.
>
> The operator handoff line stands verbatim (R228): **"box preflight both tiers → prereg authoring → MINT."** The three 8B findings feed prereg rows; R206 is the hardest flag. The next session opens on the post-merge STATE doc.


# R230–R232 — operator adjudication response, 2026-08-04 (post-merge 8A/8B: orphan workers; VRAM discrimination; perf authorization)

Verbatim. These rule the orphan-worker finding as MINT-RELEVANT with a fix card, adopt the VRAM
discrimination-before-fixing frame with a prereg fallback recorded, and authorize a non-evidential
perf characterization. Fidelity label on each: **[INLINE]**.

## R230 — preflight ratified; CARD-ORPHAN-WORKERS opened MINT-RELEVANT; box grant [INLINE]

> R230 — Step-1 preflight PASS ratified on its evidence (R210 fix live:
> iteration_complete per-burst; R208 fix live: all four sources beating;
> LAW-16 legs on SIGINT). File the orphan finding — it is ruled now:
> worker-pool children surviving parent SIGINT with CPU pinned, reproduced
> twice, is a LAW-16 defect (lifecycle is ONE subsystem; save-then-exit
> that leaves 694%-CPU orphans has not exited) and MINT-RELEVANT — a
> killed run5 must not leave the box poisoned for the next launch.
> CARD-ORPHAN-WORKERS: fix on a fresh scratch branch (wppre-scratch, R47),
> full pipeline; scope = pool teardown on signal (terminate → join with
> bounded timeout → kill escalation, the drain_or_kill pattern the eval
> seam already has), mutation test: SIGINT during active self-play leaves
> ZERO descendant processes. The uv silent torch-downgrade is the R180
> class on a new face: box invocations of uv carry UV_NO_SYNC=1 or
> --frozen ALWAYS — added to the box-hygiene standing text. The
> operator's request constitutes the per-dispatch box grant for 9A/9B
> (R31 satisfied); alias out-of-band as always.

## R231 — VRAM: discriminate before fixing; prereg fallback recorded [INLINE]

> R231 — CARD_VRAM_ACCUMULATION's frame is ADOPTED: Hypothesis A
> (allocator caching of variable-size graph batches) vs B (reference
> leak) is DISCRIMINATED BY MEASUREMENT before any fix — allocated-vs-
> reserved at boundaries, then the empty_cache probe; a blind
> empty_cache would mask a leak. Fix routes on the verdict (A: cache
> release at game boundaries with its sync cost benched; B: find the
> retained reference — tree/wire/batcher checklist per the card), full
> pipeline on wppre-scratch, then the co-residency curve RE-MEASURED at
> deploy_sims=150 and its number written into the R206 prereg row.
> FALLBACK recorded for the prereg if the investigation has no verdict
> at authoring time: eval.worker_device: cpu for run5 (option c) — the
> conservative line; deploy-matching binds regime and sims, not device;
> deploy_sims=150 is armed and untouchable throughout.

## R232 — perf characterization authorized, non-evidential [INLINE]

> R232 — One CUDA perf session at run5's real shape, AFTER the 9A fix
> lands (so eval co-residency is real): games/hour (self-play),
> training steps/hour, eval-round wall, steady VRAM both processes, GPU
> utilization, and the derived end-to-end estimate — wall time to 25000
> steps. R196's quarantine applies: operational numbers only, labeled,
> no net-evidence claim ever; outputs feed the prereg's estimates
> section and Stage-0 planning. Every number ships its command +
> provenance line.

## R233 — 9A interim ratified; location approved; torch-pin ruled [INLINE]

> R233 — (a) VERDICT-A consumed: allocator caching on variable-size graph
> batches; move-boundary necessity MEASURED (cache hits 8.4 GiB inside
> one game), not guessed — ratified. (b) Location deviation APPROVED:
> the defect class is "deploy-head graph eval inference", and
> DeployHeadPlayer is its grep-verified home; the dispatch's "eval
> worker" wording named the process, the design named the seam — the
> design is right. The torch.compile(reduce-overhead) incompatibility
> note rides the DESIGN as a recorded hazard for the post-mint compile
> card. (c) Torch CPU pin: NO pyproject change pre-mint — the CPU wheel
> index is the deliberate parity regime and touching it now is new scope
> against finish-line posture. Standing instead: the box bootstrap is a
> MECHANICAL documented step (workspace runbook), and every box session's
> fitness probe ASSERTS torch build == cu128 before any run — it already
> caught both incidents, which is the probe doing its job.
> CARD-TORCH-INDEX opened post-mint (conditional index / uv extra).

## R234 — F-9B-1 resolved: the row's sentence governs; the drift was mine [INLINE]

> R234 — The EDGE-CAP row's wording ("the cap's job is to make the
> overshoot unconstructible by splitting, not to make the batch small")
> is the ORIGINAL Phase-T sentence; R193's and the STATE doc's "verbatim
> requirement" quoted my own restatement of it. The original governs; no
> edit; F-9B-1 CLOSED; architect ledger line — a verbatim-requirement
> that cites a paraphrase as the verbatim is the R98 class pointed at
> myself. Phase P ratified in full, including the not-a-git-repo
> observation (the dispatch's commit instruction was inapplicable and
> correctly reported rather than improvised around).

## R235 — finishing order, bench bracket, and two watch items [INLINE]

> R235 — Order to done: LAW-09 bench (before/after moves-in-fixed-wall;
> pre-registered acceptance: move-wall cost ≤ +60% accepted outright,
> beyond that STOP with the numbers — eval wall is not training
> throughput, but a doubling needs eyes) → V-2 (round COMPLETES; steady
> eval-child ≤ ~2 GiB or STOP for the cpu fallback) → Phase L →
> close/sweep → go-line evidence → Phase X per 9B's spec. Two WATCH
> items ride X: (1) RECONCILE the 20.12 s/round eval-wall figure with
> V-0's zero-games-in-15-min at deploy_sims=150 — those two numbers
> cannot describe the same regime; find the 20.12 s measurement's
> config/sims provenance and label both correctly (the prereg must not
> carry an eval-wall number of unknown regime); (2) watch the TRAINER
> process's reserved-memory curve during the soak for the same
> variable-batch allocator signature on the self-play inference path —
> if it grows, NEW CARD with the curve, no ad-hoc fix.

## R236 — mint PAUSED by operator intake (posture amendment) [INLINE]

> R236 — Mint paused by operator intake (posture amendment). The
> operator's act of opening this intake amends the finish-line posture of
> record. The mint is PAUSED until the problem table below is
> dispositioned per STATE §2. No new scope beyond the dispositioned
> pre-mint bundle. Prereg authoring resumes only after the fix branch
> merges under R170 and box preflight re-runs green on the new dev.

## R237 — entry-verification record (gaps OPEN) [INLINE]

> R237 — Entry-verification record (gaps OPEN). (i) Register R118–R235:
> UNVERIFIED — register not attached to this session; verify-and-fill
> under recovery-provenance headers at next repo contact before any
> register write lands. (ii) 9C status (V-1 bench, V-2, Phase L,
> wppre-scratch merge, Phase X): UNVERIFIED. (iii) Fix-branch
> name/base/contents: UNVERIFIED, and the defect doc self-declares
> nothing implemented — contradiction with STATE §2 to be resolved by the
> operator. Rulings in this session are register-ready blocks pending (i).

## R238 — radius resurrection watch (LAW-02) [INLINE]

> R238 — Radius resurrection watch (LAW-02). Both new research docs carry
> radius-8 framing: the Copilot review's formal identification is built on
> d_hex ≤ 8; the defect doc's CNN-3 arithmetic uses "radius 5 or 8". The
> falsified ledger holds run5 radius = 6, never 8 (R26). Structural
> conclusions (state-dependent legal set, locality caveat on
> strategy-stealing) survive at r=6; any numeric claim citing 8 (span-leak
> margins, candidate-set sizes, L(S) growth) must be re-derived from
> registry.toml at point of use (R98) before it grounds a fix. No doc may
> enter the design record with an un-annotated radius-8 number.

## R239 — F-15 transfers as a design constraint on SYS-5 [INLINE]

> R239 — F-15 transfers as a design constraint on SYS-5. SYS-5
> (quiescence → proven MCTS-Solver ±∞ backup with subtree termination) is
> adjacent to falsified row F-15 (expansion-time forced-win short-circuit
> → net never sees near-win positions → no fork learning). Context
> transfers as a hazard, not a kill: subtree termination re-creates the
> starvation mechanism unless ExIt-style target injection is mandatory in
> the design. Any SYS-5 design without proof-as-training-target is
> rejected at DESIGN stage. Noted: both independent reviews' #1
> recommendation and the register's own F-38/F-39/F-40 convergent close
> point at the same lever — this is the rare case where new research and
> the falsified register agree.

## R240 — goal ordering of record (operator posture) [INLINE]

> R240 — Goal ordering of record (operator posture). Work proceeds in
> three tracks, strictly ordered for merges, interleaved for prep: (1)
> Correctness — the pre-mint fix bundle (problem-table rows A1–A15 + the
> LAW-07/18 debts) finishes on the existing fix branch; nothing
> architectural rides it. (2) Architecture & extensibility — the PLAN-0
> program (capability seam, conformance suite T1–T4, ragged-wire collapse,
> then arches) opens only after (1) merges and the intake's MEAS items
> report. (3) Performance — a standing LAW-09 track: profiling harness
> prepped during (1), box profile run gates the sims/lever prereg lines,
> per-hotspot preregs thereafter. Mint slot unchanged: after (1) +
> preflight + prereg.

## R241 — fix-branch adoption protocol [INLINE]

> R241 — Fix-branch adoption protocol. The unfinished fix branch is
> inventoried before extension: every existing commit/diff is mapped
> against A1–A15 and classified ADOPT (passes its item's oracle) / FINISH
> / REDO. No assumption that partial work is correct — R155 applies to
> anything claiming a fix. Inventory report lands in the adjudication
> queue before new implementation starts.

# R242–R248 — operator adjudication response, 2026-08-07 (dispatcher correctness-bundle findings ruled; perf strategy of record)

## R242 — ADJ-D12: gate cadence decoupled from narration cadence [INLINE]

> R242 — ADJ-D12: gate cadence decoupled from narration cadence. The
> dispatcher's finding is accepted and it is worse than the original item:
> draw-rate and SealBot-WR hard aborts cannot fire before step 1000 at
> run5's log_interval — armed machinery with a blind first kilometer, the
> F-43 class on the abort path itself. Mechanism ruling: a new explicit
> schema key (e.g. monitor.gate_interval), no default (R1), consumed by
> gate evaluation + abort sampling + monitor_gates emission;
> train.log_interval reverts to narration-only. Consec/threshold semantics
> are re-expressed in gate-interval units — the re-scaled armed values are
> operator prereg rows, not code. This supersedes R210's "training_step
> alerting stays gated" clause in scope: R210 governed games-visibility
> narration, and its clause must not be read as arming-cadence law. Full
> sub-pipeline (impl/review/red-team) mandatory — this touches armed
> aborts.

## R243 — ADJ-D13: item 6 disposition [INLINE]

> R243 — ADJ-D13: item 6 disposition. Correct halt. Final state:
> non-finite guard + counter + alert inclusion stand; the hard-abort arm
> stays exactly as the R56 pin has it (disarmed at 1e9). Arming NaN into
> the hard abort is a prereg row. Commit 0a2b238's overstating subject
> gets reworded by interactive rebase next session — branch is unpushed,
> clean-bisect rule applies.

## R244 — Per-item verification law (new standing law; instance to the ledger) [INLINE]

> R244 — Per-item verification law (new standing law; instance to the
> ledger). Three regressions surviving to EXIT proves directories-touched
> ≠ directories-that-pin. Henceforth: per-item verification runs (a) the
> item's declared pins, (b) the test files that reference any touched
> symbol (grep-derived, not guessed), and (c) a full default-tier
> checkpoint at minimum every 3 items and at EXIT. Ledger entry: what = 3
> regressions incl. 25 preflight failures from item 6; how caught = EXIT
> full tier only; where corrected = this ruling + dispatcher template.

## R245 — CNN-6 verdict: DROP confirmed; dense-arm augmentation is an operator decision [INLINE]

> R245 — CNN-6 verdict: DROP confirmed; dense-arm augmentation is an
> operator decision. Lossless augmentation group has order 4, not 12; 8/12
> elements drop ~25% of cells each — confirmed label noise on the dense
> path, mechanism per Balestriero. Two options to the operator: (a)
> restrict dense-arm augmentation to the order-4 subgroup pre-mint
> (correctness fix; changes the control arm's training regime vs run3), or
> (b) keep 12-fold for run3 comparability (comparability on this axis is
> already suspect — run3 post-mortem: 12-fold aug failed to produce
> invariance). My lean: (a); the control arm's job is to be a sound dense
> baseline, not a bug-compatible one. Adjudication row, operator rules.
> Graph path structurally unaffected (rotate_axial exact).

## R246 — CNN-9 confirmed → card, not mint-blocking (verify the exemption) [INLINE]

> R246 — CNN-9 confirmed → card, not mint-blocking (verify the
> exemption). 4/7 history planes carry opponent stones deterministically.
> Control arm v6_live2_ls keeps planes {0,8,16,17} — no history planes —
> so the mint is untouched if that plane set is confirmed at point of use
> from registry.toml (R98). Fix rides the cross-language-parity card
> (encoding F-01/F-02), post-mint.

## R247 — Bootstrap corpus intake (card + manifest pin) [INLINE]

> R247 — Bootstrap corpus intake (card + manifest pin). The HF corpus is
> human-only, rated, per-game Elo, sha256'd, encoding-free axial move
> lists, MIT — matches the F-06 canonical shape and is presumably the
> canonical corpus's published export. Adopt as a sha-pinned external
> bootstrap artifact (R7: outside repo, manifest-indexed), pending an
> audit: winner/coordinate convention mapping to mantis-core, dedupe
> overlap vs the in-repo corpus by game_hash, Elo/length distributions,
> and two recorded selection biases — decisive-only (zero draw mass →
> value-target bias, F-07 adjacency) and ≥20 moves (drops short tactical
> wins — exactly the near-win class F-15/F-38 care about). Whether run5's
> bootstrap points at it is a prereg row.

## R248 — Perf strategy of record (answers the ordering question) [INLINE]

> R248 — Perf strategy of record (answers your ordering question). Two
> tracks, split by what is being optimized:
>
> Track P-loop (now, mint-gating): the harness, not the net. Batch
> starvation is loop-level and architecture-independent. This work
> precedes the mint and does not wait for clean architecture — it is
> infrastructure correctness.
>
> Track P-arch (post-seam): the net. Kernel/arch-level optimization waits
> for PLAN-0 Stage 3, because today the K-reduction alone has four
> implementations (SYS-1) — optimizing duplicated paths is paid twice and
> measured never. PLAN-E's T6 makes µs/leaf a permanent conformance
> column, which is "development with perf in mind" made structural.

## R249 — ADJ-D32: phantom cluster metrics (F-10/LAW-07 class, live on run5's arm) [INLINE]

> R249 — ADJ-D32: phantom cluster metrics (F-10/LAW-07 class, live on
> run5's arm). Accepted. derived_mean_f64 hard-0.0 at zero count + never-
> passed variance atomics = a channel asserting "perfect agreement" from
> zero samples — and it retroactively voids Phase R's CNN-1 σ-pull (the
> pulled σ was the phantom, not a measurement; CNN-1's zero-compute test
> reverts to NOT-RUN). Fix mechanism: zero-count → None; emitter drops
> None fields rather than publishing them; producer + mutation tests.
> Combined with R250 below: on graph encodings these fields are absent,
> not None-as-0.

## R250 — encoding-conditional instruments (governs ADJ-D9 + item 10) [INLINE]

> R250 — Encoding-conditional instruments (governs ADJ-D9 + item 10). K,
> cluster variance, coverage, and uncovered_forced_win are dense/K-cluster-
> path concepts; K is structurally absent on gnn_axis_v1. Standing rule:
> an instrument for a mechanism an encoding does not have is absent from
> that encoding's event stream — never zero, never null-as-value. Item
> 10's two halves are implemented on the dense path, ticking only when a
> K-cluster encoding is active; they remain owed (LAW-07/18 debts on the
> control arm, prerequisites for the CNN-1/CNN-3 post-mint measurements).
> Dispatcher applies this principle to the drafted ADJ-D9 wiring
> alternatives and selects the matching one; if neither matches, HALT
> stands.

## R251 — ADJ-D22: cadence-based silent disarm closed [INLINE]

> R251 — ADJ-D22: cadence-based silent disarm closed. Correct catch —
> R242's defect class relocated onto its own knob. Mechanism: the armed-
> abort audit computes each armed row's earliest possible fire step from
> the live cadence keys and FAILS any row whose value exceeds a declared
> fraction of max_train_steps; the fraction is a schema constant with a
> live consumer (no code-side default, R1). Deliberate disarm keeps
> exactly one spelling — the existing explicit R56-style pin — and a large
> interval is never a sanctioned disarm. FULL pipeline.

## R252 — R244 rider ratified, with a process note [INLINE]

> R252 — R244 rider ratified, with a process note. The evidence-hygiene
> rider is accepted on its measured grounds (110 s tier vs the contention-
> inflated walls). Going forward, riders to rulings are drafted by
> dispatchers but enter the register only via an architect ruling — this
> one enters as R244-a by this act.

## R253 — Q-FIND-1 disposition [INLINE]

> R253 — Q-FIND-1 disposition. Sequence, binding: (1) box run completes →
> read batch_fill_pct against the pre-registered ≈1.56% prediction under
> its fixed falsification criterion; (2) prediction holds → batching fix
> (server-side multi-graph collation into the existing segment-batched
> forward) is authorized as a pre-mint item on this branch, FULL pipeline,
> one commit; (3) one IQR-gated before/after bench at matched config
> (LAW-09), sph + fill + util reported; (4) the resulting worker/batch/wait
> values become prereg rows. Prediction falsified → back to the
> flamegraph, no fix authorized. ADJ-D17's correction is accepted as
> stated (finding stands, armed surface = the two dense configs; the
> drawn-but-unapplied sym draw on graph gets a one-line determinism note
> in the queue).

## R254 — KLENT, re-affirmed [INLINE]

> R254 — KLENT, re-affirmed (answers your question). KLENT is a sample-
> efficiency lever, not a throughput lever — it reduces GPU-hours to a
> given strength (deflated to ~1.3–2.5× for our cost structure by the
> prior assessment), it does not fill batches or raise sph. The current
> binding constraint is throughput, where the levers are Q-FIND-1
> (harness), SYS-4/Gumbel low-n regimes (search budget — already a prereg
> line), and worker/batch config. KLENT ingredients stay post-mint Track B
> exactly as previously ruled (λ-returns first, entropy normalization by
> log|A(s)|, reverse-KL anchoring), for confound discipline. The 4×
> headline stays dead.

## R255 — ADJ-D34: the boot guard derives its bound from the config (mint-critical) [INLINE]

> R255 — ADJ-D34: the boot guard derives its bound from the config (mint-
> critical). MAX_VISITS = 128 is a literal tunable on an armed path — rule
> 4 violation with a mint-blocking consequence. Mechanism: the Phase-T
> guard's capacity is derived at composition time from the configured sims
> regime (max over PCR arms), with the schema validating the relation
> explicitly; a regime the guard cannot honor is a mint-time error, never
> a boot surprise. No new literal, no default. The 600/75 values
> themselves stay prereg rows. FULL pipeline — this is Phase-T's integrity
> machinery.

## R256 — ADJ-D37: R250's mapping was wrong; the dispatcher's halt was right [INLINE]

> R256 — ADJ-D37: R250's mapping was wrong; the dispatcher's halt was
> right. I ruled the forced-win-injection instrument onto the dense path;
> measurement shows the mechanism runs on run5's graph legal_set arm and
> not on the shipped dense grids. My error — landing it as ruled would
> have produced an instrument reading zero exactly where the drops happen,
> the F-27 canary shape. Corrected rule: an instrument attaches to the
> mechanism's measured live path, not to the encoding family it was first
> described under; R250's absence principle stands, the mapping is re-
> derived from code per instrument (R98). uncovered_forced_win lands on
> the graph path. Ledger instance recorded.

## R257 — Shrimp-Bot reference intake [INLINE]

> R257 — Shrimp-Bot reference intake. Adopted as a reference
> implementation with two hard fences: (i) rules divergence — Shrimp-Bot
> is the radius-8 game; mantis run5 is radius-6 (R26/R238). Architecture
> and loop patterns transfer; corpora, checkpoints, and any radius-
> dependent arithmetic do not. (ii) acting-scheme divergence — MantisNet
> acts search-free via the KLENT operator; "never search-free at deploy"
> is an operator lock. Their net is separable from their acting scheme —
> which is precisely PLAN-0's axis separation, demonstrated in the wild.

## R258 — KLENT speed, refined [INLINE]

> R258 — KLENT speed, refined (your question). Two distinct speed effects,
> and only one was in R254: (i) sample efficiency — fewer games to a given
> strength (~1.3–2.5× our cost structure, unchanged); (ii) acting cost —
> Shrimp-Bot's KLENT is search-free, so their cost per game is ~1 forward
> per placement vs our 75–600 sims. That second effect is where their
> speed lives, and it is fenced off for run5 by your own lock. The
> admissible middle path, post-mint Track B: KLENT-style improved-policy
> targets from a per-action Q head as training targets, shrinking the sims
> a good target needs (the Gumbel-low-n direction). run5 stays clean for
> confound discipline; if post-batching ETA is still unacceptable, pulling
> λ-returns forward is an operator prereg re-decision.

## R259 — SHAKEDOWN-1 (authorized run class, non-mint) [INLINE]

> R259 — SHAKEDOWN-1 (authorized run class, non-mint). A long graph-arm
> training run from the remediation build, explicitly complete config
> minted as shakedown_*.yaml, run-id unambiguously not run5. Purpose, in
> value order: (1) first-ever live soak of the just-fixed survivability
> machinery — checkpoint/resume, watchdog saves, promotion/anchor
> integrity, gate cadence — including one deliberate SIGTERM→save→resume
> cycle in hour one; (2) the R253 clause-1 readout — the shakedown's own
> event stream delivers batch_fill_pct, so the Q-FIND-1 gate opens itself;
> (3) the before/after vehicle for the batching fix (restarting a shakedown
> is free — no prereg integrity to protect; expect the regime-coupled ring
> refusal if sims change, that's correct, start fresh); (4) trajectory +
> ladder data and progressively better checkpoints, which the WP-AXIS2
> forward-only falsifiers need as input. Nothing from a shakedown is a
> strength claim beyond LAW-15's protocol rules; no shakedown result arms
> a prereg value by itself.

## R260 — absence-mode autopilot (the long-horizon dispatcher) [INLINE]

> R260 — Absence-mode autopilot (the long-horizon dispatcher you asked
> for). Your instinct is right and the mechanism already exists at
> architect level — scale it down: durable state on disk, ephemeral
> context per packet, never compact — restart. Protocol:
> mantis-migration/plan/autopilot/ with MISSION.md (immutable orders),
> STATE.md (rewritten at every packet boundary: sha, run status, current
> phase, next action), JOURNAL.md (append-only findings/decisions, daily
> digest), AUTHORIZATIONS.md (your signed grants). Each work packet =
> fresh session: read MISSION + STATE + JOURNAL tail → one bounded packet
> → update STATE → exit; an outer wrapper relaunches. The run itself never
> depends on agent liveness — LAW-16 makes it self-sufficient; the agent
> monitors the JSONL stream read-only and intervenes only per the runbook
> (bounded crash-restarts; never touches armed values). HALT-and-queue
> changes meaning in absence mode: HALT the item, continue the mission
> with the next unblocked item. All law discipline unchanged: FULL
> pipeline on hot/armed paths, one change = one commit = one bench, R244
> verification, falsified fences.

## R261 — go-package (operator-signed) [INLINE]

> R261 — Go-package (yours to sign before leaving; without it the mission
> shrinks to local-only work).
>
> - **R245 ruling — one word.** My recommendation stands: (c), dispatcher
>   implements the per-record gate during the mission (dense-arm only,
>   doesn't block the graph shakedown).
> - **Branch push grant:** push remediation (and child branches
>   remediation/*) to origin. dev stays frozen; no merges — unchanged R170.
> - **Box grant:** 8 days compute for shakedown + benches; box checkout
>   updated to mission HEAD; ssh path left working for the agent.
> - **Restart authority:** relaunch shakedown per runbook on crash/after
>   benched improvements; max N unexplained-crash restarts before the run
>   parks and the mission continues on non-run work.
> - **Standing forbiddens** (I've drafted them into the mission): no
>   merge/push to dev, no mint, no armed-value or falsified-register edits,
>   no host/config changes beyond the runbook, no force-push, no R20-surface
>   changes.

**Operator approval recorded 2026-08-07** (verbatim: "operator approves those") —
R259, R260, and the R261 go-package are signed. R245 is thereby RULED option (c)
(conditional per-record symmetry gate, dense arm).

## R263 — R253 Reading M adopted; Design A authorized [INLINE]

> R263 — R253 Reading M adopted; Design A authorized. The clause-(1) gate
> was always about the mechanism, not the literal w1 bracket: occupancy
> capped at exactly n_workers, collector threshold structurally
> unreachable, every pop burning its full wait — that is the starvation
> mechanism confirmed at w20 to the integer, matching its own w20
> prediction. Reading L would elevate a bracket's authoring context above
> the mechanism it predicted — rejected. Rider (LAW-09): the ~15 ms/pop
> unattributed overhead gets flamegraph attribution in the same packet as
> Design A's bench — it bounds Design A's ceiling and must not be
> discovered after the verdict. One change = one commit = one IQR bench
> stands; attribution is measurement, not change.

## R264 — controlled resume re-verification on the live burn: authorized [INLINE]

> R264 — Controlled resume re-verification on the live burn: authorized.
> Soaking survivability is the burn's stated purpose; a deliberate SIGTERM
> → resume on the current ring is the cheapest live proof of f9f9eee and
> restores restart-with-ring capability. If it fails: explained crash,
> fresh ring, root-cause packet — budget intact. Until it passes,
> fresh-rings-only stands.

## R265 — ADJ-D38 mechanism (generalizes D36/R251) [INLINE]

> R265 — ADJ-D38 mechanism (generalizes D36/R251). The WR-consec is
> unfireable because gate 12 audits fire-step in the training-step clock
> while WR samples arrive in the eval-round clock. Rule: the fireability
> audit computes each armed row's earliest possible fire in that axis's
> own sample clock (draw-rate: gate-interval steps; WR: eval rounds ×
> their real cadence), derived from live cadence keys — no axis is
> auditable in a clock it doesn't tick in. D36's derivation pattern
> extends; values stay blank. FULL pipeline.

## R266 — ratifications [INLINE]

> R266 — Ratifications. ADJ-D36 ratified as landed (bit-identical ≤32
> preserved). D35 closed per the S2-stands memo. F-R-P4-1 and F-R-P2B-2
> accepted as closed. R245(c) disclosures accepted; the LAW-18
> augmentation-group counter is owed before any dense-arm training,
> dispatcher-ownable.

# R268–R270 — operator adjudication response, 2026-08-16 [RECOVERY-PROVENANCE]

**Provenance (R172/R195 precedent).** Filled 2026-08-16 from the operator's own
message text, pasted verbatim in-session, discharging the R270 recovery directive
("verify each present exactly once; fill gaps under recovery-provenance headers").
Source is the operator message, not a dispatcher gloss — the block quotes below are
the ruling text as issued.

**R267 REMAINS A GAP.** No verbatim text was available to this fill. The only record
is the STATE_2026-08-16.md §5 digest line — "R267 eval posture mechanism inert,
values operator" — corroborated by `plan/autopilot/STATE.md` ("PKG-5 eval posture
(R267) COMPLETE, landed (inert)") and `plan/EVAL_POSTURE_OPTIONS.md`. It is NOT
reconstructed here: a digest line is not the ruling. Fill it from the exported chat
transcript. (Separately still open and NOT filled here: **R227** absent and **R228**
headerless — ADJ-D2, whose brief §0.3 directs "do NOT fill", operator-side; and
**R262**, whose operative text lives in `plan/autopilot/MISSION.md`.)

## R268 — F-816-1: no crash slot consumed [RECOVERED VERBATIM]

> R268 — F-816-1: no crash slot consumed. Accepted as recommended — last act
> was a normal step, no software error line, box-external cause (OOM-killer
> class) more likely than ours. Ring restarts are proven (PKG-2), so relaunch
> is cheap when wanted.

## R269 — F-816-6 is the headline; bootstrap path → MINT-CRITICAL [RECOVERED VERBATIM]

> R269 — F-816-6 is the real headline, and it reframes "the GNN is slow."
> Three findings in one row: (a) draw_rate pinned 1.000 — at
> bootstrap-from-scratch strength neither side ever completes six, so every
> game runs to the ply cap. That is the training-side twin of F-R-P2B-5, and
> it compounds the slowness arithmetic: capped games are maximum-length games,
> so every game costs cap × sims forwards for zero learning signal. Part of
> "extremely slow" is actually "degenerate." (b) ply 0.00 at every check
> cannot be literally true of cap-length games — suspected phantom channel,
> F-10/R249 class, investigate before trusting any ply-derived stat. (c) zero
> solver fires — reconcile against config disarms before reading it as a
> defect. Consequence: the bootstrap path is promoted to mint-critical — the
> R247 corpus (BC pretrain / corpus-mix warm-start) is no longer a
> nice-to-have, it's the mechanism that makes early self-play decisive; the
> alternative (ply-cap/draw-abort posture surgery via PKG-5's inert machinery)
> treats the symptom. And run5-as-minted aborts on draw rate ~step 25k either
> way — a prereg decision now with measured grounds.

## R270 — handoff; STATE_2026-08-16 supersedes STATE_2026-08-06 [RECOVERED VERBATIM]

> R270 — Handoff. This architect context is past its reliable horizon. The
> state snapshot below supersedes STATE_2026-08-06; one entry caution flagged
> inside: rulings R236–R270 were issued in-chat and may not all be in
> plan/rulings_register.md — the next session's first act is verify-and-fill
> with this chat transcript as the recovery source (the R172/R195 precedent).

# R271 — operator adjudication response, 2026-08-16 (register hygiene) [INLINE]

> R271 — Register hygiene: archive/index split. (a) `plan/rulings_register.md`
> is the append-only verbatim ARCHIVE — never compressed, rewritten, or pruned;
> only recovery-provenance fills and status annotations touch it. (b)
> `plan/RULINGS_ACTIVE.md` is the derived working index (standing laws, locks,
> fences, live pre-mint force, dispatch governance); sessions seed from ACTIVE +
> laws.md + CLAUDE.md, the register is consulted at point of use when a cited
> number needs its text (R98). Absence from ACTIVE claims no forward force and
> deletes nothing; ACTIVE is never authority — the register wins on conflict.
> (c) Graduation lifecycle: a ruling stating a durable rule is absorbed into
> docs/registers/laws.md or CLAUDE.md by normal amendment commit; its ACTIVE row
> collapses to a pointer, then drops at next curation. (d) ACTIVE is curated at
> session close by the register-pen holder; one log line per curation (ACTIVE
> §8). (e) The 14 single-ruling batch headers may be normalized in one mechanical
> docs commit — optional, zero substance.

# R262 — hierarchical autopilot orchestration [RECOVERY-PROVENANCE / SUMMARY-ONLY]

**Out of numeric position by design.** This is a foot append, not a splice: the
register is append-only for substance (R271(a)). R262's numeric neighbours R261 and
R263 are unmoved and unedited; read this section as if it sat between them.

**Provenance.** Source: `plan/autopilot/MISSION.md`, its closing section
`ORCHESTRATION (R262)` — the last paragraph of the file, immediately after PHASE M4.
MISSION.md's own header records that file as "operator-issued mission text, received
verbatim 2026-08-07 at session start (post R255/R256, HEAD 5bc84d7)" and immutable.
So the block below is operator-issued text, quoted verbatim from disk.

**Fidelity: [SUMMARY-ONLY], deliberately — NOT [RECOVERED VERBATIM].** The quoted
block is the MISSION's *application* of R262, addressed in the second person to MAIN,
and it CITES R262's own sections (§1–§6) as an external document rather than
reproducing them. It therefore carries R262's operative content — every section is
named with its substance — but it is not R262's own words, and R262's numbered text
does not exist anywhere on disk. Downstream artifacts corroborate the section
structure without reproducing it either: `plan/autopilot/LOCKS.md:1` ("serialization
convention (R262 §5)"), `plan/autopilot/TRACK_SD-FIX.md:1` ("sole committer, R262
§3"), `plan/autopilot/handoffs/SD-FIX/FP4_LEAF_BRIEFS.md:1` ("artifact-mediated
handoff per R262 §4"), `plan/autopilot/handoffs/SD-FIX/FINDINGS_F-P1.md:4` ("SD-FIX
creates no rulings (R262 §6)"). `plan/autopilot/JOURNAL.md:16` already recorded the
gap in these terms: "R262 has no register entry; operative text lives in MISSION.md."
The R267 precedent governs the label: a rendering is not the ruling. If the exported
2026-08 architect transcript is ever recovered, R262's own text supersedes this
section and should be appended beside it under a second recovery-provenance header.

> ORCHESTRATION (R262): You are MAIN, a Fable-class thin orchestrator.
> First act: write MISSION.md (this text), STATE.md, JOURNAL.md, and
> three track states under plan/autopilot/. Launch sub-dispatchers
> SD-RUN, SD-FIX, SD-DEV (Fable) with track briefs derived from phases
> M0–M4 per R262 §1. Leaves: Opus 5 (impl/design/diagnosis), Sonnet 5
> (mechanical/monitoring). Enforce: capability floor + cross-model
> review on FULL items (§2); SD-FIX sole code committer, leaves return
> diffs+transcripts only (§3); artifact-mediated hand-offs, pins re-run
> before acceptance (§4); ≤3 leaves/sub, tier+box lock file (§5);
> forbiddens bind all nodes, HALTs escalate inward and park, never stall
> the mission (§6). You hold no implementation context: orient every
> packet from MISSION+STATE+JOURNAL tail+repo at HEAD, rewrite STATE at
> packet end, exit and relaunch rather than compact (R260).

*Structure recoverable from the block, section by section: §1 track briefs derived
from mission phases M0–M4, sub-dispatchers SD-RUN/SD-FIX/SD-DEV at Fable class;
§2 capability floor + cross-model review on FULL items; §3 SD-FIX is the sole code
committer, leaves return diffs and transcripts only; §4 artifact-mediated hand-offs
with pins re-run before acceptance; §5 ≤3 leaves per sub-dispatcher, tier + box lock
file; §6 forbiddens bind every node, HALTs escalate inward and park rather than stall
the mission. The leaf-model floor — Opus 5 for impl/design/diagnosis, Sonnet 5 for
mechanical/monitoring — is stated in the block itself, not in a cited section.*

# R272 — operator adjudication response, 2026-08-16 (R271 execution ratified) [INLINE]

> R272 — R271 execution ratified in full. (a) Census, both foot appends,
> 14-header normalization, and seed-range corrections accepted as landed; the
> R262 [SUMMARY-ONLY] fidelity call is CORRECT and standing — a mission's
> application of a ruling is a rendering, not the ruling (R267 precedent);
> supersedable by the exported transcript only. (b) The R259 restore and the
> R23–R31 authority qualification are ratified, and the practice becomes part of
> R271(d): every curation spot-checks ≥5 index lines against verbatim register
> text; an index line may never claim more than its ruling — an overclaiming
> index line is the F-43 misstatement class in miniature. (c) The two
> config-cited BLOCKING values are pinned to the prereg batch with their
> evidence: checkpoint_interval (R137(b) am. R173; run5.yaml:111 mints 0, not a
> legal production posture) and random_floor_games (R147/A-2; run5.yaml:21 mints
> 0). Both rows authored, both VALUES operator-owed — they join eval
> posture/budgets, sims regime, gate-cadence consecs, NaN arming, corpus row.
> (d) The dispatcher's R119 restraint on R147 (armed value, not touched) was
> correct behavior, on the record.

**Accompanying operator direction, same message, NOT part of R272's numbered text**
(recorded because it carries forward force on how this workspace is versioned):

> Git init: yes — but one snapshot commit, not six.
>
> pro — this whole arc's pathology is unversioned load-bearing text: R227/R262/R267
> gaps exist because ruling texts lived only in chats. F-816-1 showed a host event
> wipes a box; an unversioned mantis-migration on one machine is the same exposure
> for the entire governance record.
> con of six-commit replay — the pre-R271 tree was never versioned, so replaying
> "the six commits" fabricates a history that never existed. R69/R98 spirit: don't
> manufacture provenance.
> Verdict: git init + one commit `docs(plan): snapshot at R271/R272 — register
> split landed, first versioned state`, then a private remote (rule-7: autopilot
> docs carry box specifics — never public), pushes operator-only per R170. From
> here on, register append-only becomes machine-checkable by diff.
>
> Owed, unchanged: R267 transcript · ADJ-D2 (R227/R228) text · prereg values incl.
> the two newly pinned · Design A bench readout (still deliverable 1).

# R273 — operator adjudication response, 2026-08-16 (R272 execution ratified; hygiene program CLOSED) [INLINE]

> R273 — R272 execution ratified. Git posture accepted as standing: whitelist
> ignore, sha-manifests-in / blobs-out, snapshot-not-replay history, repo-local
> identity. The register-hygiene program is CLOSED — remaining items are
> operator-input-gated, not work-gated (R267 transcript, ADJ-D2 text), plus the §4
> GRAD absorption queue, deferred as a batchable dispatcher packet, not session
> work. R240 ordering now governs: correctness → architecture → perf.

**Rider recorded from the same message, NOT part of R273's numbered text** — one
verification owed: the R160 index tightening (ACTIVE §4, v1.2) was ratified on the
grounds that it ran in the sanctioned direction, but the operator could not verify it
against the register from their context. **Owed: one verbatim check of R160's index
line when the register is next in the operator's context.** Grounds as stated: R272(b)
makes narrowing safe by construction — an underclaiming index line costs a register
lookup, an overclaiming one manufactures law.

# Pointer — R274's NOT-FILLED marker, SUPERSEDED by the foot append (2026-08-17)

**This is a pointer, not a ruling section.** A `[ABSENT — VERBATIM NOT RECOVERABLE THIS
SESSION]` marker stood here from 2026-08-17, appended by the R277 dispatcher, who recorded that
R274's verbatim text was not in that session's context and refused to promote the
`plan/ADJUDICATION_QUEUE.md` F-816-9 paraphrase to register text (R69/R98). **The marker is
DISCHARGED**: the architect supplied R274 verbatim in the R278 follow-up prompt the same day,
and it is filled as a foot append at the end of this file — `# R274 — architect adjudication
response, 2026-08-16 (VisitSlotsExceeded routing; landed 2026-08-17, …)`, out of numeric
position by design, on the R262 foot-append precedent (R271(a): append-only for substance).

The header was demoted to this pointer so that R274 has exactly ONE `# Rnnn —` section header
in the file and the census invariant survives the fill. R274's numeric neighbours R273 and R275
are unmoved and unedited; read the foot append as if it sat here. Recorded 2026-08-17 by the
dispatcher executing R278 Task 1.

# R275 — F-816-9 Phase A ratified; PRODUCER-BUG stands; Phase C GO [INLINE]

> R275 — F-816-9 Phase A ratified; PRODUCER-BUG stands; Phase C GO. (a) R255 capacity
> derivation CLEARED, regime-tagged: valid for the current visit-limited target construction;
> re-derives with both pins if completed-Q-on-graph is adopted at prereg (LAW-02). (b) Class
> refined to two conjuncts — seam (failed inference never reports completed; run-fatal loud,
> LAW-14; inference_failures_total reaches the event stream in-run, LAW-18/R164) and exporter
> (a zero-visit search cannot produce a target; refusal is loud and named). Flip-sets cover
> both conjuncts plus the healthy ply-cap game (R274(f)) and the exact-capacity boundary. (c)
> The sims prereg row is BLOCKED on this fix landing — the tripwire-sensitivity mechanism is
> the grounds; at 600/75 the defect is silent. Other prereg rows (checkpoint interval, random
> floor, NaN arming, gate consecs, corpus) are NOT blocked. (d) Trigger forensics ride the
> next box session: before anything is deleted, grep the five dead after-reps' logs for
> graph_inference_forward_failed / CUDA-OOM signatures. If GPU-failure-under-batch-fusion
> confirms, it routes to the CARD-RUN5-GPU-OOM class (R114) as its own item — the seam fix
> does not close it. (e) Bench re-run has two legitimate outcomes post-fix: numbers, or a loud
> named inference failure — the second is the trigger confirmation, not a failed errand. (f)
> The dispatcher's two retractions are ratified R69/R96-clean; corrections live in the report
> artifact. (g) Phase A's evidence quality — pre-registered discriminators, measurement over
> inference, precise retraction — is the standard; on the record.

# R276 — Phase C exit adjudication (merge gate, grants, OOM routing, ply cap) [INLINE]

> R276 — (a) f816-scratch APPROVED for merge conditional on the closure-typing cite;
> merge/push operator (R170). (b) Seam deviation ratified; sequential reviewer isolation
> adopted as an R262 rider — concurrent review voids the later verdict unless re-verified in
> an isolated worktree. (c) R43 per-event grant for the target_latch_propagation.rs edit
> granted retroactively, disclosed same-event, never precedent; queue row stands. (d) F-816-2
> telemetry-only reclassification ratified; R96 correction propagates to every artifact
> carrying "feeds an armed abort," architect's repetition recorded. (e) Capacity guard
> retained as defense-in-depth, direct-construction coverage, R275(a) re-arm clause. (f) OOM
> trigger CONFIRMED → F-816-10, CARD-RUN5-GPU-OOM class (R114). Design-first packet
> authorized: memory-bounded fusion (bucketed padding per the Q4 memo admissible), loud and
> counted — no silent catch-and-retry without a counter and a cap. Bench unit redefined and
> declared openly per LAW-09: parent vs (Design A + memory bound) as one deployable unit —
> Design A cannot run without the bound, so the bound is part of the change, not a confound.
> Attribution rides the bench itself. (g) — ply cap, below.
>
> R276(g) — Ply cap stays 128 through bootstrap-fix validation. The cap VALUE is a prereg
> row, decided jointly with the adjudication criterion, from the corpus audit's measured
> length distribution (plies, LAW-03) — not from a precautionary round number. Operator may
> override (it is his row); an override carries these grounds on the record. Rider: verify
> _DEFAULT_MAX_PLIES = 128 is schema-resident, not a code-side default (R1) — one hygiene
> line, next packet.

**Grounds recorded from the same message, NOT part of R276's numbered text** — the operator's
argument against raising the cap now: at draw_rate 1.000 the cap is the only thing bounding the
cost of a worthless game; 200/128 = 1.56x more forwards per degenerate game for zero additional
signal, which makes the F-816-6 headline strictly worse and does not move the 25k-step abort
collision by one step. The right number is derivable from the corpus audit's move-count
distribution (p99/p999, plies), and cap + adjudication posture are ONE decision — valuing the
cap before choosing the posture orders the decision backwards.

# R277 — operator adjudication response, 2026-08-17 (merge gate ratified + grants) [INLINE]

> R277 — (a) The merge-gate HALT is ratified as correct and load-bearing: the closure-typing
> hole was real, reachable, and repo-documented as intended behavior ("release blocked Rust
> waiters even if this thread exits unexpectedly") — the fix in 461728b (discriminator =
> runner kill-switch, ordering running=false before either close, pool stops runner before
> server) is APPROVED; the classifier reading no queue retires the wrong-queue class by
> construction. (b) Per-event operator grant (source: operator's message, this date): merge
> f816-scratch (6 commits through 461728b) → dev fast-forward, push origin — gates green
> first, grant expires on completion, never precedent. (c) Cleanup grant, containment-gated:
> delete only branches verified fully contained in dev by ancestor check (wppre-scratch per
> F-816-8; f816-scratch post-merge); prune local review worktrees only after verifying no
> unique commits; anything not fully contained is reported, never deleted. (d)
> _DEFAULT_MAX_PLIES registered as F-816-11, R255-MAX_VISITS class: arena/eval cap is an
> unconfigurable literal, adjudicator-coupled, divergence-silent against
> selfplay.max_game_moves. Fix direction fixed by hard rule 3 — one resolver, eval reads
> self-play's seam, a deliberate split requires its own prereg key — and it RIDES the ply-cap
> prereg packet (the row cannot be honoured without it). (e) Routing protocol adopted: every
> ruling ships with ROUTE + follow-up prompt. ROUTE: CURRENT SESSION — it holds the branch,
> verification state, and exit report; a fresh dispatcher would re-derive all of it for a
> mechanical merge.

# R274 — architect adjudication response, 2026-08-16 (VisitSlotsExceeded routing; landed 2026-08-17, text supplied verbatim by the architect) [INLINE]

**Out of numeric position by design.** Foot append, not a splice — the register is append-only
for substance (R271(a)), and this is the R262 precedent applied a second time. R274's numeric
neighbours R273 and R275 are unmoved and unedited; a pointer sits at the numeric position where
the R277 dispatcher's NOT-FILLED marker stood. Read this section as if it sat between them.

**Provenance and fidelity: [INLINE], verbatim.** Source: the architect's R278 follow-up prompt,
2026-08-17, which supplied R274's text under an explicit VERBATIM instruction. This is the R270
verify-and-fill act (R172/R195 precedent) and it discharges the R277-dispatcher's OWED item. The
`plan/ADJUDICATION_QUEUE.md` F-816-9 paraphrase is superseded by the text below and must not be
cited as ruling text from here on. R278(b) is the ruling that carries this fill, and its rider
makes same-event fills the standing shape: every ruling's follow-up prompt now carries the
ruling text verbatim, so this class of gap cannot re-open.

> R274 — VisitSlotsExceeded adjudication. (a) Registered as **F-816-9**,
> queue row from the exit-report evidence verbatim. EXPLAINED run-fatal
> defect — consumes NO unexplained-crash slot. (b) MINT-BLOCKING,
> correctness class; under R240 it outranks and blocks the bench. (c) The
> R255 check is not presumed wrong and the producer is not presumed right —
> that is what Phase A measures. Two protected outcomes: (i) capacity
> derived in the wrong axis/unit → fix the derivation, keep the check;
> (ii) the producer genuinely writes mass beyond the sims bound →
> target-integrity no-drop (R155/R157/R158) puts the fix on the producer.
> **A capacity raise without a measured mechanism is banned; deleting or
> softening the check is banned** — LAW-14 stays run-fatal. (d) Bench
> re-run protocol: after side pinned to 8ba2d0d against its parent, matched
> config, one change one bench; py-spy provisioning rides the bench re-run,
> not this packet. (e) F-816-2 rides the same packet as an independent
> card — one packet, separate commits, per-item verification (R244), no
> shared-commit bundling. (f) The fix ships with the test the tiers lack:
> a deterministic ply-cap-length game through the production record path.

# R278 — operator adjudication response, 2026-08-17 (R277 execution ratified; R274 fill, audit certification, branch adjudication) [INLINE]

> R278 — (a) R277 execution ratified; the R274 NOT-FILLED halt and the exit-code self-catch
> are the discipline working, on the record. (b) R274 filled from the architect's verbatim
> text (rides this follow-up); rider to R277(e): every ruling's follow-up prompt carries the
> ruling text verbatim for same-event register append. (c) Completed-Q-on-graph admissibility
> DOWNGRADED to ASSERTED (source: STATE digest; no register text); exercisable at prereg only
> by operator re-affirmation; R275(a)'s citation annotated to this disposition. (d) Audit
> tool: certified-before-cited — fix OPEN-2 (manifest shape), OPEN-5 (elo pair), OPEN-7
> (game_id absent), re-run; certified output supersedes the hand pass before prereg use. (e)
> The five worktree-agent-* branches get content-equivalence adjudication: per branch, diff
> tree against dev; subject-landed or tree-equivalent → tag archive/<name>, delete the ref
> (reversible); unique content → report, operator decides; a8dc82d6 inspected first and
> individually. (f) hexo-mantis origin switches to the SSH URL on this machine — host-local
> config, rule-7 clean, approved. (g) Evidence-citing reports verify cited files are tracked
> at commit time (git ls-files --error-unmatch); convention line lands in the packet-record
> doc. (h) Ply cap stays 128 now; the prereg row decides cap × adjudication criterion jointly
> on the certified distribution, with the tail-floor caveat attached and F-816-11 as
> precondition. ROUTE: CURRENT SESSION.

**Second-order catches recorded from the same message, NOT part of R278's numbered text** —
quoted verbatim; they are the grounds behind (c), (d) and (h):

> The R255 spot-check has a real consequence. "Completed-Q-on-graph is ADMISSIBLE as a prereg
> option" traces to a STATE digest line citing a "retirement clause" that R255's registered
> text doesn't contain — and my own R275(a) repeated the citation. The claim's true source is
> probably the unrecovered R267/R236–R270 chat. Under R160 that makes admissibility ASSERTED,
> not register-backed — downgraded accordingly; exercisable at prereg only by your
> re-affirmation. Third instance of the index-overreach class (R160, R259, R255); the
> spot-check law keeps paying.
>
> The audit numbers are a disclosed hand pass, and prereg grounds must be certified. The
> tool's contract is wrong in three named places; fix and re-run before those numbers enter a
> prereg row. Interim use ratified as disclosed.
>
> The corpus data updates the ply-cap picture without changing my verdict. 128 truncates 7.3%
> of decisive human games (and the tail is a floor — decisive-only filtering hides unresolved
> long games), 200 truncates 2.1%, p99 is 259. So your instinct wasn't baseless — but the
> cheap way to buy that signal back is arming ply_cap_adjudication (seat-neutral criterion),
> which converts capped games into decisions, not buying 56% more forwards per game in a
> regime that's still 100% degenerate. Cap value decides at prereg, jointly with the
> criterion, on F-816-11's unified resolver, re-derived from measured selfplay lengths
> post-run5.
>
> The git add -f near-miss and the tail-piped exit codes were both self-caught — that's R69
> working. Both graduate to convention lines.

# R279 — operator adjudication response, 2026-08-17 (R278 execution ratified; corpus CERTIFIED as prereg grounds; audit-merge, disposal, identity and push grants) [INLINE]

**Provenance: [INLINE], verbatim.** Text supplied in the architect's message accompanying the
R279 follow-up prompt, 2026-08-17, under the R278(b) rider (every ruling's follow-up carries
its ruling text verbatim, so the append happens in the same event). Operator forwarding of that
follow-up is what makes grants (e) and (f) effective; (e) is per-event and expires on completion.

> R279 — (a) R278 execution ratified; gap-vs-owed definitional split adopted into ACTIVE §1;
> the architect's "exactly six" expectation was the error, on the ledger. Certified nearest-rank
> values govern (p99 263, p999 523); off-ladder hand-pass Elo quartiles correctly withdrawn.
> (b) The corpus is now CERTIFIED GROUNDS for the bootstrap and ply-cap prereg rows (exit 0,
> sha match, 8698/8698 distinct, winner convention replay-verified, OPEN-2/5/7 closed with
> refusal-pinned amendments). (c) a8dc82d6: archive-tag + delete GRANTED — dev strictly
> supersedes by range-diff, the tag preserves the sha, evidence is on the record. (d)
> hexo-mantis gets a repo-local git identity (the derived one); host config otherwise
> untouched. (e) Merge grant, effective on operator forwarding of the follow-up:
> r278-audit-scratch → dev after gates (floor 3198), push over SSH. (f) Standing push grant
> proposed, effective on operator forwarding: mantis-migration main → its private remote,
> docs-only repo; hexo-mantis pushes remain per-event under R170. (g) Ply-cap prereg
> recommendation as stated above — cap × bootstrap × adjudication decided as one matrix, values
> operator's, F-816-11 single-resolver fix as implementation precondition, turn-boundary value
> derived at point of use. (h) ROUTE: housekeeping → CURRENT SESSION; F-816-10 → FRESH
> DISPATCHER; both prompts attached.

**On (g)'s "as stated above" — recorded because the antecedent is NOT in this register.** R279's
numbered text points back at a recommendation stated earlier in the same architect message; that
prose was not supplied to the dispatcher and is therefore not on the record here. What IS on the
record is the follow-up prompt's own specification of the same matrix, landed as GROUNDS on the
PLY-CAP row of `plan/RUN5_MINT_PREREG.md` (certified distribution cite + tail-floor caveat +
label-noise framing; the warm-start → ~256-ply-class vs posture-only → 128 conditional;
adjudication armed either way; F-816-11 named as precondition). Cite the prereg row's GROUNDS
block, never "R279(g) verbatim", for the reasoning — R279(g) itself registers only the SHAPE of
the decision (one matrix, operator-valued, F-816-11 precondition, turn-boundary derived at point
of use). No value is registered by R279.
**POINTER FORWARD (added 2026-08-18 under R280(b)):** the missing antecedent has since been
supplied SELF-CONTAINED as `R279(g)-ANNEX`, a foot section at the end of this register. The
annex REPLACES this stub's force and is R279(g)'s reasoning of record; this note stands
unedited because it records that the gap existed and was caught.

# R280 — architect adjudication response, 2026-08-18 (R279 execution ratified; F-816-10 ratified + merge grant; rule-7 tag scan; R43 ratification withheld) [INLINE]

**Provenance: [INLINE], verbatim.** Text supplied in the architect's message accompanying the
R280 follow-up prompt, 2026-08-18, under the R278(b) rider (a follow-up carries its ruling text
verbatim so the append happens in the same event). Operator forwarding of that follow-up is what
makes grants (c), (d) and (f) effective; each is per-event and expires on completion.

> R280 — (a) R279 execution ratified: prereg-row creation was R119-clean dispatcher work;
> full-branch-name tags correct (no alias); zero-headroom floor is the ratchet working as
> designed, R46 quarantine being the sanctioned skip path. (b) R279(g)-ANNEX (self-contained,
> supplied below) repairs the dangling antecedent; protocol rider to R277(e): register-bound
> ruling text carries no deictic references — an annex under the issuing ruling's number is the
> repair path. (c) Rule-7 scan of the five public archive tags (trees, messages, diffs) before
> any further push; hits → remote tag deleted, local archive kept, reported. (d) F-816-10
> ratified: pre-registered verdict chain, cross-model reviews, the equal-length transposition
> pin on the record as the packet's most valuable find; merge grant effective on operator
> forwarding — gates at f043dc7 (floor 3323), fast-forward → dev, push SSH. (e) The FROZEN-table
> edit: R43 ratification WITHHELD pending the two queue rows verbatim next exit; the edit stands
> meanwhile on the stated grounds. (f) Worktree prune grant for the three registered clean
> worktrees, verify-then-remove. (g) Mint path restated: merge → box sitting (STEP 0 config
> capture → budget → calibrate → MINT VALUE, operator → validation burst → LAW-09 bench) →
> bench verdict (architect) → prereg batch (matrix ready) → R61 preflight → MINT. ROUTE:
> CURRENT SESSION.

**Accompanying direction, NOT part of R280's numbered text** (recorded under the R272/R276
labelling shape). Two items the architect owed and one withheld, in the architect's own framing:
(1) the R279(g) dangling antecedent is named as the ARCHITECT's defect — "I wrote 'as stated
above' into a register-bound block, and the register correctly recorded a stub" — repaired by
the annex below; (2) on the section-header retitle question, **existing headers STAND** (the
provenance line inside each section carries the truth), and the convention GOING FORWARD is
"architect adjudication response, operator-ratified by forwarding" — forward-only, never a
retro-edit of landed headers; (3) the withheld item is R280(e): "disclosed-but-unread is not
granted" — the FROZEN-table edit stands on its stated grounds but is NOT ratified until the two
queue rows are read verbatim.

# R279(g)-ANNEX — the ply-cap × bootstrap matrix, self-contained [INLINE, annex to R279(g) under R280(b)]

**What this is and what it replaces.** R279(g)'s numbered text ends "recommendation as stated
above", and that antecedent was never supplied to the register — the R279 section records that
honestly as a stub (see the note under R279, which STANDS and now carries a pointer here). Under
**R280(b)** the architect supplies the reasoning SELF-CONTAINED, as an annex under the issuing
ruling's number. **The annex REPLACES the stub's force**: from here, this is R279(g)'s reasoning
of record. The stub's honesty note is left in place unedited, because deleting it would erase the
fact that the gap existed and was caught.

**Protocol rider adopted with it (R280(b), extending R277(e)):** register-bound ruling text
carries NO deictic references — no "as stated above", no "the foregoing", no pronoun whose
referent lives outside the registered text. An annex under the issuing ruling's number is the
sanctioned repair path when one slips through.

> R279(g)-ANNEX: The ply-cap and bootstrap prereg rows decide as one matrix. (i) Warm-start taken
> (BC-pretrain or corpus-mix) → self-play cap rises to the ~256-ply class, exact value on the
> engine's turn boundary derived at point of use, covering p99=263 of certified human decisive
> lengths and cutting truncation label-noise from 7.335% to ~1%; cost binds only on the tail once
> play is decisive. (ii) Posture-only → cap stays 128; in the degenerate regime every added ply
> is waste and the draw-abort bounds it. (iii) eval.ply_cap_adjudication arms in either branch;
> criterion and min_margin operator-valued, seat-neutrality asymmetry disclosed. (iv) The
> certified tail is a floor on the true tail (decisive-only bias). (v) F-816-11's single-resolver
> fix is the implementation precondition. (vi) Deploy/bridge cap is a separate knob; training cap
> ≥ deploy expectation. Values remain operator-owed.

**Still true after the annex, stated so the annex is not over-read.** No VALUE is registered
here: (i)'s "~256-ply class" is a CLASS, its exact value derived at point of use on the engine's
turn boundary (LAW-03), and every operator-valued item in (iii) stays operator-valued (R119).
R276(g)/R278(h) are unamended — **the cap STAYS 128 until the prereg row values it.**

# R281 — architect adjudication response, 2026-08-18 (push hold ratified; rule-7 SANITIZE-FORWARD; gate 17 authorized; F-816-12 ratified mint-critical; F-816-13 granted) [INLINE]

**Provenance: [INLINE], verbatim.** Text supplied in the architect's message accompanying the
R281 follow-up prompt, 2026-08-18, under the R278(b) rider (a follow-up carries its ruling text
verbatim so the append happens in the same event). Operator forwarding of that follow-up is what
makes (b)'s sanitize+push and (c)'s gate adoption effective; the grants are per-event and expire
on completion.

> R281 — (a) The push hold is RATIFIED as correct and on the record; the scan's
> premise-falsification was surfaced, not absorbed. (b) Rule-7 disposition: SANITIZE-FORWARD —
> the fixture's host paths neutralized in a forward commit (manifest sha updated; consuming test
> verified path-insensitive first, HALT if not); history rewrite REJECTED on provenance grounds
> — sha citations across the governance record are load-bearing; residual old-history exposure
> ACCEPTED as path-shape-only and recorded. The held merge pushes immediately after. (c) Gate 17
> authorized: the scan's pattern set becomes a repo-local CI gate on added/modified files, with
> one full-tree baseline run at adoption (the known blob exempted by sha until sanitized);
> convention line — fixture-capture tools normalize host paths at write time. (d) F-816-12
> RATIFIED, MINT-CRITICAL, CARD-RUN5-GPU-OOM class: (i) the two caps are ONE partition —
> box-sitting STEP 1c/3 fit train.microbatch_caps and inference.fused_graph_caps JOINTLY from
> the measured terms; two independent mints are not a legal posture; (ii) derivation comments
> carry sha + regime tags at re-mint (convention, LAW-02 shape); (iii) a boot-time partition
> assertion (measured card >= declared partition sum) is authorized within the box-procedure
> amendment — the F-27-on-bytes class needs one live check over the whole partition, not two
> green halves; (iv) no separate code packet — the procedure amendment plus joint mint close it.
> (e) F-816-13 GRANTED per R276(c) shape, per-event, never precedent; the table row cites its
> queue row. (f) The artifact-gate false-red disclosure, detached-HEAD reachability rigor, and
> the annex over-read guard: on the record. ROUTE: CURRENT SESSION.

**Accompanying direction, NOT part of R281's numbered text** (recorded under the R272/R276
labelling shape), in the architect's own framing. On **F-816-12**: "the trainer's 9.431 GiB
budget partitions a card against a self-play term measured one regime ago — pre-Design-A, one
graph in flight — while the guarding oracle measures the trainer alone and stays green. That is
the F-27 canary class applied to a byte budget, **fourth instance of premise-moved-under-a-green-
gate this project has caught** (F-27, F-43, the R255 tripwire-sensitivity, now this). The row's
own framing is correct: **a partition is one object; certifying its halves independently
certifies nothing.** The box OOM string is the live confirmation — 13.10 GiB already allocated
when a 1.72 GiB inference request arrived." On **F-816-13**: "now readable: granted. The loop
iterates plan-parts, not items; each iteration is a whole vectorized collate+forward+softmax;
and sequential parts with freed tensors is the bound's mechanism — 'all parts at once' is
definitionally the defect. The reasoning block in the table is exactly the R276(c) shape."

**Execution note on (c), recorded because it is a DEVIATION IN FORM from the literal text.**
R281(c) anticipates "the known blob exempted by sha until sanitized". The sanitation (b) and the
gate (c) landed in the SAME session, sanitation FIRST, so the exemption was never needed and
**gate 17's EXEMPT register ships EMPTY**. This is stronger than the text, not weaker — R98 asks
for a clean baseline and the gate has never been green over a dirty tree — and the exemption
MECHANISM is implemented (grounds + blob sha, self-expiring both ways) and pinned by a test, so
the path R281(c) named exists and is exercisable. Recorded rather than absorbed.

# R282 — architect adjudication response, 2026-08-18 (R281 execution ratified; delegation boundary set; BOX SITTING packet authorized) [INLINE]

**Provenance: [INLINE], verbatim — but carried ONE PACKET LATE, and that is on the ledger.**
R282 was issued in the architect's message accompanying the box-sitting dispatch, whose Task 0.1
instructed "append R282 [INLINE] verbatim" while the dispatch carried no R282 body. The sitting
HALTed at Task 0 on exactly that (`plan/F816_10_SITTING_RECORD.md` §1, BLOCKER 1). The text
below was supplied verbatim in the SUBSEQUENT dispatch (the R283 resume), which is where this
append happens. **R283(a) puts the un-carried body on the ledger as an architect defect against
the architect's own R278(b) rider**, and R283(b) converts the rider into a rule: a dispatch
citing an un-carried ruling number HALTs at Task 0 by rule, not by judgment.

> R282 — (a) R281 execution ratified: gate-17's false-clean self-catch, the
> regex-defect self-test kill, the empty-EXEMPT deviation (stronger than the
> text, R98), and the untracked-supplement design (tracked floor, untracked
> ceiling — the supplement gains the new username) all on the record; the
> pipe-status class is now a NAMED recurring class under
> PACKET_RECORD_CONVENTIONS §2. (b) Delegation boundary per operator's
> direction: architect decides routine adjudications/grants/dispositions
> after pros-cons-and-red-team, on the record; preserved operator-only — box
> grants, run5 mint authorization, judgment-valued prereg rows;
> measurement-derived cap values mint in-sitting under pre-registered
> acceptance (calibration falsifier PASS ∧ partition inequality holds ∧
> fitted pair in the tool's recommended form), any deviation HALTs with the
> numbers. (c) BOX SITTING packet authorized, fresh dispatcher, box grant
> activated by operator forwarding: STEP −1 box prep → STEP 0–6 per the
> amended procedure; all sitting artifacts land in mantis-migration only;
> bench verdict bands pre-registered from QFIND1_READOUT §2's banked bracket
> BEFORE any run; the burst reads actor_sync as the production-visible games
> signal (F-43); STEP 3 mints run5 AND shakedown, closing F-816-12.
> (d) Residual-exposure identity (blob 82785cf @ 528eb37, path-shape only)
> re-recorded as accepted. ROUTE: FRESH DISPATCHER.
>   [R283 note of record: R282(c)'s "band from QFIND1_READOUT §2" was
>   unsatisfiable as written — §2 carries no after-band and Row B's A5
>   forbids scaling. R283(c) supplies the authored band. R282(c) is
>   otherwise in force.]

# R283 — architect adjudication response, 2026-08-18 (box-sitting HALT ratified; R282 carried; bench pre-registration AUTHORED and binding; calibration falsifier substitution) [INLINE]

**Provenance: [INLINE], verbatim.** Text supplied in the architect's message accompanying the
box-sitting RESUME dispatch, 2026-08-18, under the R278(b) rider — this dispatch carries BOTH
its own body and R282's, which is the rider being honoured after the defect R283(a) ledgers.
The band in (c) is AUTHORED by the architect, not derived from a banked bracket: the sitting's
BLOCKER 2 established that no after-band existed to derive from, and R283(c) states that its
legitimacy comes from being written BEFORE the measurement, not from where the numbers descend.

> R283 — (a) HALT ratified on both blockers; three architect defects on the
> ledger: un-carried R282 body (violating the R278(b) rider), rung-2
> transcription error (the ladder says rung 1), instrument-shape mismatch in
> the before-side pin. (b) R282 lands via this prompt verbatim; protocol
> rider: a dispatch citing an un-carried ruling number HALTs at Task 0 — by
> rule. (c) The pre-registration is AUTHORED and binding: instrument =
> bench_side.sh burst 5×21 at matched config; before side = the banked
> burst-shaped [REDACTED:abs-root-path:0fe68e7a]bench_before; P1 mechanism-engaged = after
> occupancy.max > 20 (P1 false ⇒ defect investigation, no perf verdict);
> P4 verdict gate on median gph vs before 138.43 — BUILD ≥ 208 (1.5× floor,
> expected band 277–554), PROTOTYPE seam-ladder rung 1 at 166–208 with P1
> true, STOP/INVESTIGATE < 166 or any after-median regression; P2
> (queue_wait.mean < 10 ms) and P3 (fill ≥ 45%) recorded diagnostics, not
> gates; declared unit = the bundle f54be91 → 24ae93e + minted caps
> (R276(f)); one extra side at 8ba2d0d^ authorized only if the verdict
> hinges on attribution; burn §3 demoted to sustained-regime sanity anchor.
> (d) Calibration falsifier substitution ACCEPTED: synthetic-graph
> calibration (no ring on the box) with STEP 4 gaining the real-graph
> falsifier — recorded peak memory during the past-ply-120 burst must
> respect the minted budget with the design margin; substitution recorded,
> never silent. (e) Flamegraph wrap-launch only (CAP_SYS_PTRACE measured
> clear). (f) Gate-17 reporting fix authorized per-event: the green line
> prints tracked + local pattern counts, never local contents; mutation
> self-test extended to the local-supplement arm. (g) Governance push is an
> operator command this round. (h) F-816-12 remains OPEN until STEP 3.
> ROUTE: FRESH DISPATCHER, resume from the sitting record.

**Accompanying direction, NOT part of R283's numbered text** (recorded under the R272/R276
labelling shape), in the architect's own framing. On the three ledgered defects: "R282 never had
a landing vehicle. I wrote the R278(b) rider — *every ruling's follow-up prompt carries the
ruling text verbatim* — and then violated it one packet later: R282 lived in chat, the file only
cited it. The dispatcher's refusal to mint against a paraphrase of an unread authority is exactly
the discipline the register exists for." On the before-side pin: "I pinned the burn's n=8646
sustained numbers to a burst-shaped instrument. That's the **LAW-15 wall-clock-bar mistake in
miniature**: the comparison must be burst-vs-burst, same shape, same config — and the matched
before side already exists at `[REDACTED:abs-root-path:0fe68e7a]bench_before`. The burn numbers demote to a sustained-regime
sanity anchor." On the Task-1 HALT: "the finding I'm most glad exists: Row B's own A5 forbade
exactly the extrapolation a sloppier pipeline would have made."

**Grounds recorded for (c), because a pre-registration's grounds are the half that makes it
falsifiable.** The band is derived from the burn §3 measured mechanism, not from an envelope:
`occupancy.max` pinned at `n_workers` over 8,646 samples (starvation, confirmed to the integer);
`collate` ~22 ms/pop dominant over the 10 ms deadline; and Design A's mechanism converting
N pops × collate into ONE collate of N — which is where the throughput is expected to come from,
not merely from filling the batch.

---

# R284 — architect adjudication response, operator-ratified by forwarding, 2026-08-18 (LAW-09 bench verdict STOP/INVESTIGATE; perf packet targets ordered; minted caps stand; F-816-14 + R46 loop queued) [INLINE]

**Provenance: [INLINE], verbatim.** Text supplied in the architect's message accompanying the
P-MASK/P-CHECKS perf dispatch, 2026-08-18, under the R278(b) rider as made non-discretionary by
R283(b). Header title follows the R280 forward-only convention.

**One provenance note, recorded because the packet carried the text TWICE and the two copies are
not identical.** The dispatch opens with a `## RULING TEXT CARRIED (R283(b) rule)` block and
closes with a fuller block; the body quoted below is the CLOSING block, which is the
ruling-shaped carry (it runs to the ROUTE line and states each clause in full). The opening
block is a compressed restatement by the same hand. **They agree on every operative item** —
verdict, the four pre-registered readings, the target order, the caps disposition, the declined
escalation, the sims fork, both queued items, and the path. They differ in exactly one place, and
the difference runs in the register's favour: clause (a) of the closing block reads "the
latency-not-fill reading **above** is the verdict's mechanism of record", whose referent lives
outside the registered text and is therefore the **deictic shape R280(b) bans**; the opening
block spells the mechanism out instead ("latency-per-leaf-batch-round-trip, not GPU fill, is the
binding metric for fixed-worker MCTS self-play — the serve thread's per-batch Python overhead
raised latency and no fill compensates"). Both are recorded here so the register carries the
ruling AND the reading that survives R280(b); nothing downstream turns on the choice, because
the two say the same thing and the packet's own targets (b) are identical in both.

> R284 — LAW-09 bench verdict: STOP/INVESTIGATE as pre-registered (P1 TRUE, P2
> FAIL, P3 PASS, P4 0.911× non-overlapping IQRs). (a) Mechanism engaged,
> throughput regressed; the latency-not-fill reading above is the verdict's
> mechanism of record. (b) Investigation targets in order: P-MASK (gnn.py:153
> sync-free gather replacing boolean-mask indexing, output-parity-oracled),
> P-CHECKS (_check_structural to Rust per hard rule 12, or amortized to O(1) per
> batch with grounds), then the pure-forward microbench (STATE §2.4's queued
> measurement) only if the first two don't close it. (c) The minted caps STAND —
> they are correctness, not perf; zero OOM at peak 13.053/14.084 closes
> F-816-10's validation; the 3.30% split cost is accepted. (d) Attribution
> escalation DECLINED — the verdict is STOP/INVESTIGATE under every attribution
> split, so the 8ba2d0d^ side buys nothing. (e) Sims-row implication: PCR 600/75
> is unviable on measured numbers; the row waits for the re-bench or mints in the
> 50-sims class accepting current throughput — operator's fork, best taken after
> the re-bench. (f) Queued: F-816-14 eval-child orphan (survives parent SIGTERM
> holding 458 MiB — LAW-16 class, partition-threatening, fix before any long
> run); the R46 loop is ORDERED on the pre-existing preflight foreign-litter
> flake — a red tier at push time is not a legal steady state; ACTIVE
> version-stamp drift joins the conventions (stamp rides the §8 edit). (g) Path:
> perf packet → after-side re-bench against the SAME banked before side and the
> SAME R283(c) band → verdict → prereg → preflight → mint. ROUTE: FRESH
> DISPATCHER, packet carries R284 verbatim.

**Clause (a) as carried in the opening block, recorded because it is the R280(b)-clean
statement of the same mechanism** (NOT a second ruling, and NOT an annex — R284's numbered text
is the block above):

> (a) Mechanism engaged, throughput regressed; latency-per-leaf-batch-round-trip,
> not GPU fill, is the binding metric for fixed-worker MCTS self-play — the serve
> thread's per-batch Python overhead raised latency and no fill compensates.

**One citation in (b) does not resolve, recorded as a note of record and NOT repaired here.**
"hard rule 12" has no referent: `CLAUDE.md`'s hard rules run 1–9, and `docs/design/repo_design.md`
§12 is strength-claim + eval discipline. The rule the clause is reaching for is
`repo_design.md` §10 (Performance doctrine), first bullet, which names the division of labour
in terms that cover `_check_structural` exactly — "Rust owns every per-position / per-record /
per-leaf loop (board, legal moves, MCTS, graph build, **contract validation on marshaled
arrays**). … a Python-level per-item loop on a hot path is a review-blocking defect." The
P-CHECKS design cites §10, not "hard rule 12", and says so.

**ANNOTATION under R284's foot — appended 2026-08-18 by the R285 dispatcher under R285(c) and
R285(d). APPEND-ONLY: R284's numbered text above is untouched, and nothing above this line was
edited to accommodate it.**

R285(c) makes the citation note above a **correction OF RECORD**: R284(b)'s "hard rule 12"
referenced the **predecessor document**, and the current doctrine home is the **perf-doctrine
bullet** — `docs/design/repo_design.md` §10, first bullet, whose "contract validation on
marshaled arrays" is the clause that covers `_check_structural`. **Annotation, never repair**:
R284's text stands exactly as issued, and a reader who cites "hard rule 12" is directed here.

R285(d) settles the double-carriage recorded above. **R284's canonical text is the register's
appended block** — the closing, ruling-shaped carry quoted above. The chat variant is a
**SUPERSEDED DRAFT**, not a second authority and not an annex; the clause-(a) variant recorded
beside it keeps its stated status (the R280(b)-clean statement of the same mechanism, NOT a
second ruling). The double-carriage and the residual deixis in the superseded variant are
**architect defects on the ledger** — they are not dispatcher findings and were not treated as
such. **Forward rule, from R285(d): a ruling has ONE text — the packet-embedded one.**

# R285 — architect adjudication response, 2026-08-18 (R284 execution ratified; F-816-18 GRANTED; F-816-16 conditional; merge grant conditional; AUDIT-BEFORE-REBASELINE ordered) [INLINE]

**Provenance: [INLINE], verbatim.** Text supplied in the architect's follow-up packet routing
R285 execution to the current session, 2026-08-18, under the R283(b) carry rule as made
non-discretionary for this workspace. Header title follows the R280 forward-only convention.

**Carriage note — R285(d) applied to R285 itself, recorded because the same shape recurs.** The
packet carried this text TWICE: an opening `## RULING TEXT CARRIED (R283(b) rule)` block and a
closing paragraph. Unlike R284's two carriages, **these two are IDENTICAL** — verified
mechanically (whitespace-normalized string equality by script, not compared by eye), so the
line-wrapping differs and nothing else does. There is therefore ONE text and no variant to
choose between: the shape R285(d) deprecates is present, the defect it deprecates is not. The
block below is that text.

> R285 — (a) R284 packet ratified: parity-oracle discipline, the
> pre-registered no-local-signal prediction, the R58 orphan catch with
> re-measurement, Item 3's failed-first-drive honesty, Item 4's
> hypothesis-killed-by-experiment. The sync-migration trap is named of
> record: the 36.22% box frame may be misattributed pipeline wait; the
> re-bench's two falsifiable predictions are the decision instrument.
> (b) F-816-18 GRANTED — bandwidth-floor reasoning accepted; the doctrine's
> mechanism is satisfied; prediction rides the re-bench. (c) Citation
> correction of record: R284(b)'s "hard rule 12" referenced the predecessor
> document; the current doctrine home is the perf-doctrine bullet; register
> annotation, never repair. (d) R284's canonical text is the register's
> appended block; the chat variant is a superseded draft; double-carriage
> and residual deixis are architect defects on the ledger; henceforth a
> ruling has ONE text — the packet-embedded one. (e) F-816-16 conditionally
> GRANTED iff its subject is exactly the assertion-map delta (check 8
> reformulation, check 9 range guard, check 13 addition, zero removals);
> any other content → HALT before merge. F-816-17/-20: on-branch subjects
> HALT the merge; findings-only rows route onward. (f) Merge grant, operator
> forwarding, conditional on (e) resolving clean: gates at the branch head,
> fast-forward → dev, push. (g) F-816-15: AUDIT-BEFORE-REBASELINE ordered —
> all 39 reds classified before any manifest touch; own packet. (h) The
> supervisor-unarmed orphan row: PDEATHSIG-class fix ordered PRE-MINT; the
> re-bench proceeds meanwhile with orphan sweeps before and after the run.
> (i) Re-bench box event authorized on merge: the packet's request block +
> orphan sweep + the band read from the register. ROUTE: CURRENT SESSION.

**EXECUTION OUTCOME, recorded here for the archive reader and LABELLED NOT PART OF THE NUMBERED
TEXT (R276 shape).** Clause (e) resolved **MISMATCH → HALT** on the same day, so the clause-(f)
merge grant did not vest and the clause-(i) box event was not authorized. The grounds are
mechanical and are recorded in `plan/ADJUDICATION_QUEUE.md` (F-816-16's status block) and in
`plan/RULINGS_ACTIVE.md` §8: F-816-16's subject is the frozen EVAL-STUB signature edit
(`tests/eval/test_eval_selfplay_child_parity.py`, `ORACLE_FREEZE_EVALDECODE.sha256:2`), whereas
the assertion-map delta named by (e) lives in `src/mantis/selfplay/graph_collate.py`, which is
in NO freeze register — the two subjects are disjoint. Independently, a SECOND frozen path
(`tests/eval/test_rung_seat_off_window.py`, `ORACLE_FREEZE_A.sha256:6`) was edited on the branch
and F-816-16 declares it NOT frozen. Both directions of (e)'s "any other content" test are
therefore met. **Nothing was merged; the branch is unchanged and unpushed.**

# R286 — architect adjudication response, 2026-08-19 (F-816-16 HALT ratified; BOTH frozen edits GRANTED on their actual subjects; de-triplication ORDERED onto the F-816-15 audit; merge grant RE-VESTS; F-816-17/-20 routings ratified) [INLINE]

**Provenance: [INLINE], verbatim.** Text supplied in the architect's follow-up packet routing
R286 execution to the current session, 2026-08-19, under the R283(b) carry rule as made
non-discretionary for this workspace. Header title follows the R280 forward-only convention.

**Carriage note — the R285(d) duty, discharged BEFORE this append.** The packet carried this text
TWICE: an opening `## RULING TEXT CARRIED (R283(b) rule)` block and a closing paragraph. R285(d)
makes diffing the copies BY SCRIPT a mechanical duty rather than an impression, so it was done:
both copies were extracted, whitespace-normalized (`re.sub(r"\s+", " ", …).strip()`) and compared
for string equality — **1,875 characters each, EQUAL**. The line-wrapping differs and nothing
else does. There is therefore ONE text and no variant to choose between: as with R285, the shape
R285(d) deprecates is present, the defect it deprecates is not. The block below is that text.

> R286 — (a) The F-816-16 HALT is RATIFIED; the diff-not-description test
> caught two independent failures: the architect's mis-scoped conditional
> (R285(e) named a subject that is frozen nowhere — on the ledger) and the
> row's incomplete census (one of two frozen edits undisclosed; mechanism
> measured: single-register hand-read while the verifier is red on both
> paths). (b) Both frozen edits GRANTED on their actual subjects —
> test_eval_selfplay_child_parity.py (ORACLE_FREEZE_EVALDECODE:2) and
> test_rung_seat_off_window.py (ORACLE_FREEZE_A:6) — per-event, never
> precedent; grounds: the signature change is forced by the production seam,
> the equal-count argument is checkable and enforced by check 13, and no
> assertion, golden, or expectation moves. The grant is conditioned on the
> R43 row being amended to name both paths with the census-error mechanism
> recorded. (c) De-triplication ORDERED: one shared stub module, imported by
> all three eval test files, frozen once — riding the F-816-15 audit packet
> so freeze surgery happens exactly once; the audit's classification pass
> covers these two paths like the other 37. (d) The R285(f) merge grant
> RE-VESTS on (b): gates already green at 9c14368; fast-forward → dev, push,
> delete the scratch ref. (e) The re-bench box event vests on the merge per
> R285(i). The 0.16%-vs-0.31% prose discrepancy: originated in the R284 exit
> prose, repeated by the architect — both ledgers; the record's figure
> governs; nothing in the argument turns on it. (f) F-816-17's routing
> RATIFIED as filed: the dead legal_mask build rides the re-bench readout;
> if collate still shows, its removal is the named next change with the A4
> rows re-expressed against the gather. (g) F-816-20 routings RATIFIED:
> items 1–2 and the .tmp-litter half of item 3 → the PDEATHSIG packet; the
> gate-16-scope half → the cutover battery beside F-816-15.
> ROUTE: CURRENT SESSION.

**EXECUTION OUTCOME, recorded here for the archive reader and LABELLED NOT PART OF THE NUMBERED
TEXT (R276 shape).** All clauses executed the same day. (b)'s grant CONDITION was discharged
first — `plan/ADJUDICATION_QUEUE.md`, F-816-16, now names both frozen paths with their register
coordinates, carries the census-error mechanism, and reads **CLOSED-GRANTED**; the paragraph that
declared `test_rung_seat_off_window.py` unfrozen is corrected IN PLACE with its original wording
preserved verbatim beside the correction, because a laundered census is the defect rather than
the fix. (c) landed as a scope line on F-816-15. (d) then vested: `r284-perf-scratch` @ `9c14368`
fast-forwarded onto `dev` and was pushed to `origin/dev`, the scratch ref deleted, leaving `dev`
the only branch. (e) vested with it — `plan/R284_REBENCH_REQUEST.md` is forwardable and its §0.2
now names the post-merge `dev` head. (f)/(g) landed as ratification stamps on their rows.

**NOTE OF RECORD on (e)'s figure — the disposition is (e)'s, the provenance below is MEASURED and
is not what "a prose discrepancy" would predict.** R286(e) makes "the record's figure" governing,
and the record's figure is **0.31%**: 27 / 8,696 samples, derived in `plan/R284_PERF_RESULTS.md`
§1.2 (py-spy, 100 Hz, 60 iterations at 41,808 nodes). That value governs and is what
`plan/R284_REBENCH_REQUEST.md` §3 carries.

**But the 0.16% variant is not a mis-transcription of it, and the ledger should say so.** Traced
to source rather than assumed: `plan/R284_PERF_DESIGN.md` §0 records `gnn.py:153`
`emb[legal_mask]` at **0.16% self (4 / 2,505)** — a DIFFERENT profile run, taken BEFORE the work
to establish the "this host cannot rank Python-side costs" finding, with a different sample count
and a different node count. Each figure is **correct for its own run**; the DESIGN's own
"1/226th" is internally consistent with 0.16%, and the re-bench block's "1/117th" with 0.31%. So
what propagated into the R284 exit prose and then into R285's was a figure carried ACROSS
DOCUMENTS WITHOUT ITS PROVENANCE — the two profiles were never distinguished at the point of
carriage — rather than a number typed wrong. **The ledger entries stand as (e) directs**; this
note records what the defect actually is, because "repeated a wrong number" and "carried a right
number away from the run that produced it" have different fixes, and only the second one is true
here.

**What is verified mechanically, stated at the width the check supports.** Before this section
was appended, neither figure appeared anywhere in this register. As of R286 both appear — inside
R286(e)'s own quoted text, where they are named ONLY to identify the discrepancy. **No ruling's
numbered argument rests on either value**, which is (e)'s own point ("nothing in the argument
turns on it"): both put the local share two orders below the box's 36.22% frame
(`plan/F816_10_SITTING_RECORD.md` §12.5).

# R287 — architect adjudication response, operator-ratified by forwarding, 2026-08-19 (R286 execution ratified; MISSION CLEAN-SWEEP established; mission-scoped grants; perf lane of record; D1 = remote CI verified) [INLINE]

**Provenance: [INLINE], verbatim.** Text supplied in the MISSION CLEAN-SWEEP dispatch packet
(Appendix A, named by the packet as "the ONE canonical text"), operator-ratified by forwarding
per the R261 shape the packet itself invokes, 2026-08-19, under the R283(b) carry rule. Header
title follows the R280 forward-only convention.

**Carriage note — the R285(d) duty, discharged BEFORE this append.** The packet carried this text
TWICE: the `## Appendix A — R287 verbatim` block and a closing trailing paragraph. Both copies
were extracted, whitespace-normalized (`re.sub(r"\s+", " ", …).strip()`) and compared for string
equality by script — **1,698 characters each, EQUAL**. The line-wrapping differs and nothing else
does. There is therefore ONE text and no variant to choose between; that is now three consecutive
packets where the shape R285(d) deprecates is present and the defect it deprecates is not. The
block below is that text.

> R287 — (a) R286 execution ratified; the figure-provenance convention
> adopted from its founding case: every measured figure carried across
> documents names its producing run. (b) MISSION CLEAN-SWEEP established
> under R260/R262: an autonomous loop, durable state on disk, ephemeral
> packet contexts, subagent leaves with sequential isolated review,
> red-team-adjudicated decisions with pre-registered verdicts.
> Mission-scoped grants, operator-signed by forwarding: commits, gate-green
> merges to dev, and pushes on both repos; remote-CI repair including
> workflow files (gate logic stays in tools/ — the workflow only calls it).
> Excluded, always: armed values, self-granted frozen edits, box access
> (box blocks are staged for operator forwarding), architecture refactors,
> bootstrap/prereg decisions — these queue for the architect or operator
> and the mission continues past them. (c) Perf lane of record: consume the
> re-bench readout when it lands → trainer-side profile (the unmeasured
> half of steps/h) → conversion/pinned-H2D candidates with local mechanism
> proofs → Rust criterion at run5 shape → bucketed-compile design-only;
> graves fenced per LAW-05/LAW-02 item by item. (d) D1 of the
> definition-of-done is remote CI verified green on dev HEAD — checked, not
> assumed — and it is the first loop item. (e) The vested re-bench block
> forwards in parallel with the mission start; its readout is mission
> input; its verdict remains the architect's. (f) Runaway-scope fences:
> enumerated definition-of-done, item-size tripwire (a fix wanting more
> than its row's scope files a new row and halts the item), checkpoint
> report at every merge, operator stop at any time. ROUTE: FRESH DISPATCHER
> — the mission MAIN.


# R288 — architect adjudication response, operator-ratified by forwarding, 2026-08-20 (MISSION CLEAN-SWEEP ratified in full; RQ-1 GRANTED as a class; F-816-23 GRANTED diff-scoped — the D1 key; R286(c) re-sequenced; the F-17/F-19 scope ruling and candidate INCR-GRAPH; batch adjudication ordered) [INLINE]

**Provenance: [INLINE], verbatim.** Text supplied in the R288 execution follow-up packet, which
names its own opening block "the ONE canonical text", operator-ratified by forwarding per the
R261 shape, 2026-08-20, under the R283(b) carry rule. Header title follows the R280 forward-only
convention.

**Carriage note — the R285(d) duty, discharged BEFORE this append, and this time it FIRED.** The
packet carried this text TWICE: the `## RULING TEXT CARRIED (R283(b) rule)` block and a closing
trailing paragraph. Both copies were extracted, whitespace-normalized (`re.sub(r"\s+", " ",
…).strip()`) and compared by script — **1,680 characters vs 1,742: NOT EQUAL**. Three consecutive
packets carried this shape with no variant; the fourth has one. The delta is **two insertions,
both inside clause (f), 62 characters in total**, located by `difflib.SequenceMatcher` opcodes
rather than by eye:

    A[1587] -> B[1587]  insert  "(RQ-2..19 and the Q3_FIX_DESIGN set) "
    A[1655] -> B[1692]  insert  " — no per-row round trips"

**Which text governs, and why.** R285(d) fixes a ruling's canonical text as "the packet-embedded
one"; here BOTH copies are packet-embedded, so the tiebreak is the packet's own nomination — its
authority line reads *"Authority: R288 (carried verbatim below — the ONE canonical text)"* and
points at the first block. **The first block is therefore the numbered text below.** The variant's
extra words are recorded here in full, verbatim, and are **NOT part of the numbered text**: clause
(f) in the trailing copy reads *"all RQ rows (RQ-2..19 and the Q3_FIX_DESIGN set) surfaced verbatim
in ONE document for a single architect ruling pass — no per-row round trips."* **Nothing in
execution turns on the choice**: the same packet's Task 4 orders the row set re-derived from
`plan/mission_clean/STATE.md` §RULINGS-QUEUE *rather than* from any enumeration, and "one document,
one pass" already excludes per-row round trips. The variant is strictly more specific and strictly
consistent; it is preserved rather than adopted, because promoting the un-nominated copy is the
drift R285(d) was written against.

> R288 — (a) MISSION CLEAN-SWEEP ratified in full; on the record: the Q2
> Option-B falsifier firing pre-execution, the CI evidence-surface-first
> method, the twice-broken-twice-fixed PDEATHSIG cycle with measured rcs,
> and the first trainer profile in project history. (b) RQ-1 GRANTED as a
> class: the 34 REAL-DRIFT holds ratify held→OK where each path's
> annotation traces to a merged, reviewed packet (the held-at pins); any
> path whose citation fails to trace stays held — the tool's rc reflects
> reality either way. (c) F-816-23 GRANTED diff-scoped: the grant binds to
> the exact diff carried on the row, per-event, never precedent, with one
> condition — the diff touches no production config and no arming
> authority; if it does, HALT. This is the D1 key; one CI cycle follows.
> (d) R286(c) re-sequenced, not rescinded: de-triplication (F-816-21)
> proceeds AFTER RQ-1 lands, when consolidation consolidates ratified
> content instead of laundering unratified content — the Q2 hold was
> correct. (e) Falsified-register scope ruling: F-17/F-19 are scoped to
> their measured subject (legal-set maintenance on descent paths, dense-era
> regime); explicitly NOT covered: eval caching, root-level increments,
> transposition reuse, and incremental axis-graph construction from parent
> — the last registered as candidate INCR-GRAPH, gated on Q4d's box
> measurement, pre-registered falsifier = the F-19 inequality at run5's
> measured depth distribution. The LAW-02 valve is the re-open path for
> every grave; the fence stands as a re-litigation tax, never a
> prohibition. (f) Batch adjudication ordered: all RQ rows surfaced
> verbatim in ONE document for a single architect ruling pass.
> ROUTE: CURRENT SESSION.


# R289 — architect adjudication response, operator-ratified by forwarding, 2026-08-20 (the RQ batch ruling ordered by R288(f): 23 lettered clauses over 26 rows, premise-verification institutionalized, RQ-21/RQ-22 minted, three consolidated packets, and the ONE-TEXT protocol made permanent) [INLINE]

**Provenance: [INLINE], verbatim.** Text supplied in the R289 packet under its own heading
`## R289 — CANONICAL TEXT (the batch adjudication ordered by R288(f))`, operator-ratified by
forwarding per the R261 shape, 2026-08-20, under the R283(b) carry rule. Header title follows the
R280 forward-only convention.

**Carriage note — the R285(d) duty, and the first packet to satisfy (v) BY CONSTRUCTION.** R289 is
carried in this packet **exactly once**. There is no second copy to diff, so the mechanical
comparison that has run at every append since R285 has nothing to compare and is recorded as
NOT-APPLICABLE rather than as a pass. That is the outcome clause (v) legislates for permanently,
arriving in the same packet that legislates it. The chat exit report accompanying this execution
carries **no clause text**, per (v)'s second sentence.

**Architect-ledger defect, annotated and NOT repaired (R285(c) shape).** The packet's own preamble
reads *"This packet IS the ruling's one canonical home (R289(t) below ends the double-carriage
class)"*. **The clause that ends the double-carriage class is (v), not (t)** — (t) disposes of
RQ-b2 and RQ-c and says nothing about carriage. The misreference is in the packet's framing prose,
which is not part of the numbered text below, so nothing in the ruling is affected; it is recorded
here because a future reader chasing "(t)" for the ONE-TEXT rule would find a different subject and
conclude the register had drifted.

> R289 — Dispositions bind to each row's VERBATIM text in
> plan/RQ_BATCH_2026-08.md: before executing any lettered clause, the
> executor verifies the clause's stated premise against the row's own text;
> a mismatch HALTs that row back to the architect and the rest proceed
> (the R286 diff-not-description rule, institutionalized).
>
> (a) RQ-3: freeze pins reference COMMITTED content only — a freeze
> register row is minted in the same commit that lands the frozen file,
> pinning the staged blob; a register born red against every committed
> tree is a process defect. Practice amendment lands in the conventions
> doc and the ORACLE-WRITE leaf instructions.
> (b) RQ-4: a new freeze row may pin only content that is
> reviewed-and-merged via the full pipeline or explicitly granted; a
> born-green row over unratified content is the laundering class the Q2
> falsifier caught, and is banned by rule.
> (c) RQ-5: an independent fresh-context cross-model review of the
> freeze_verify.py mission diff is ORDERED, findings-only — the tool
> ratified 34 paths and is now load-bearing; load-bearing tools get the
> same review the code gets.
> (d) Q2 §7 q5 (assign id RQ-21): freeze_verify becomes CI GATE 18 —
> wired into the local gate battery and the workflow, every run, with its
> 12 self-test arms; owner = the gate family; a verifier without a
> trigger is disarmed by construction.
> (e) Q2 §7 q3 (assign id RQ-22): whether row 30 should be frozen at all
> is evaluated under (a)/(b)'s freeze policy inside the freeze-governance
> packet; freezes exist to protect adjudicated oracles, not to embalm
> convenient files — the packet states the criterion and applies it to
> every current freeze row, row 30 included.
> (f) RQ-6 and RQ-12: annotate, never repair, for the closed ledger
> (RQ-6); repo_design.md gets a normal amendment commit aligning its
> exit-code text to the derived census test, which is the authority
> (RQ-12, R9 shape).
> (g) RQ-7: SPLIT — the mechanism and floor derivation are the
> architect's: supervisor_kill_grace_sec must exceed the measured
> terminal-eval drain with margin (drain measured >320 s; the floor
> derivation states grace ≥ measured drain × a stated margin); the VALUE
> is a prereg row, operator-owed, grounds attached.
> (h) RQ-8: MonitorConfig migrates to schema-resident defaults per R1 —
> no code-side defaults; LAW-08 live-consumer checks ride; dispatcher
> packet.
> (i) RQ-9: wrapper depth ≥ 2 REFUSES with a named error; a legitimate
> deep-wrapper launch is made explicit via an env override that emits its
> own LAW-18 arming event — warn-and-continue is banned on this path.
> (j) RQ-10: the supervisor dies OF its signal — after save-then-exit
> drains complete, it re-raises and exits 128+n so every caller reads the
> truth; rc-substitution on caught signals is banned (LAW-16 alignment).
> (k) RQ-11: the string-constant module reference is replaced by a direct
> import or a registered symbol; if a circular import forced the string,
> the seam is misplaced and the packet says where it moves.
> (l) RQ-13: eval _work_dir becomes run_id-scoped; hygiene packet.
> (m) RQ-15: each Q1-R rider classifies BY ITS OWN TEXT into (i)
> adds-instruments-or-tests under existing laws → pre-approved as a
> class, execute; or (ii) changes a bar, threshold, or armed posture →
> prereg/architect, file and hold. Ambiguity halts the rider, not the
> batch.
> (n) RQ-16: the dead legal_mask field's removal proceeds NOW as LAW-08
> hygiene (zero production consumers, independently re-derived), with the
> A4 two-views rows re-expressed against the gather; its perf effect is
> measured at the next box sitting inside C1's before/after —
> subsumption accepted, one packet, separate commits.
> (o) RQ-17: bench floors are host-attested instruments; cross-host
> comparisons are inadmissible for verdicts (mechanism evidence only);
> the box re-attests its own floors; the mcts_bench floor waits for the
> box. LAW-15's reproducible-instrument doctrine applies to perf floors.
> (p) RQ-18: the compiled-arm parity criterion — (i) measure the
> eager-vs-eager output noise floor on the fixture battery under the
> production autocast regime, twice-run same weights; (ii) compiled
> outputs must sit within k× that measured floor with k pre-registered
> before the box run; (iii) argmax and top-k policy decisions identical
> on the battery; (iv) any breach fails the prototype — no post-hoc
> widening (LAW-09). This precedes Q4e box S2.
> (q) RQ-19: the mixed-batch/pretrained-buffer path is RESERVED, not
> dead — it is the corpus-mix candidate mechanism of the bootstrap prereg
> row; it gains a RESERVED marker citing that row and MUST NOT be
> deleted; cpu_budget.py relocates under tests/ or names a live consumer
> in one commit.
> (r) RQ-20: the premise assertion Q1R §3.4 required lands in the next
> engine commit, citing RQ-20 and the R288(c) grant; the same-commit
> requirement's violation is recorded as an artifact of the diff-scoped
> grant instruction — architect ledger, execution correct.
> (s) RQ-a: a CI-runtime packet is AUTHORIZED — session-scoped shared
> fixtures where isolation permits, tier sharding/parallelism in the
> workflow, with constraints: zero test-semantics changes, count floor
> preserved, process-global-state hazards (the F-816-4 thread-leak class)
> explicitly red-teamed, flake-rate watched across three consecutive CI
> runs before the packet closes. Target: restore ≥90 min headroom.
> RQ-b's class rule rides here: test-scope disarm of an armed value is
> legal ONLY with the OC-7 pattern — disclosure docstring (what is
> disarmed, why, scope), the armed rule's fire coverage cited in named
> other tests, and never in configs/.
> (t) RQ-b2: the host-independent fire witness is a synthetic-injection
> producer test at the gate's input seam — feed a synthetic pool crossing
> the threshold, assert the fire (LAW-07 shape); if the seam does not
> admit injection, that inadmissibility is the defect and the packet
> fixes the seam. RQ-c: min_step-class values classify in the
> armed_aborts.py manifest as instrument vs safety; the manifest is the
> single classification authority; misclassified rows migrate by
> amendment.
> (u) The lint gate refuses LOUD with a named cause on a missing
> interpreter (the mise/node shim case) — a self-test that cannot
> distinguish missing-interpreter from broken-config fails toward
> refusal, the gate-17 degrade-wide precedent.
> (v) Protocol, permanent: a ruling's text exists in EXACTLY ONE place —
> the packet that lands it. Chat summaries are labelled non-canonical and
> carry no clause text. This clause closes the double-carriage class
> (fourth instance recorded at R288's carriage).
> (w) Packet consolidation: ONE freeze-governance packet carries (a),
> (b), (c), (d), (e) and the owed de-triplication (F-816-21's work); ONE
> hygiene packet carries (h), (k), (l), (n), (q)'s relocation; the
> CI-runtime packet carries (s), (t)-witness, (u); (g)'s derivation and
> (p)'s floor measurement are architect/box items; (r) is one commit.
> ROUTE: CURRENT SESSION for bookkeeping, (r), (d)'s wiring, and packet
> assembly; FRESH DISPATCHERS for the three consolidated packets.


# R290 — architect adjudication response, operator-ratified by forwarding, 2026-08-20 (R289's execution ratified; RQ-8 ESCALATED and re-scoped with a fork; RQ-16's per-tensor census; the freeze-governance packet re-scoped into six ordered steps; the R43 same-act re-pin clarified) [INLINE]

**Provenance: [INLINE], verbatim.** Text supplied in the R290 follow-up packet under its own
heading `## R290 — CANONICAL TEXT`, operator-ratified by forwarding per the R261 shape,
2026-08-20, under the R283(b) carry rule. Header title follows the R280 forward-only convention.

**Carriage note — R285(d), second consecutive NOT-APPLICABLE.** The packet states in its own
preamble that it *"is R290's ONE canonical home (R289(v))"*, and it carries the ruling **exactly
once**. There is no second copy to diff, so the mechanical comparison is recorded as
NOT-APPLICABLE rather than as a pass. R289(v) is now producing the outcome it legislates for, in
the first packet that inherited it.

> R290 — (a) The R289 execution is ratified. The two premise-verification
> halts are the mechanism's proof of value: the RQ-16 halt prevented an
> architect-ordered deletion of code with live read-sites in
> model/gnn.py:198 and model/gine.py:171 — the second deletion stopped by
> diff-not-description. The RQ-12 no-churn resolution and the measured arm
> selections on clauses (k) and (t) are ratified as executed.
> (b) RQ-8 is ESCALATED and re-scoped: the row's original claim is
> falsified (all MonitorConfig fields are schema-resident); the real
> defect is a bare-MonitorConfig() construction path that bypasses the
> minted config and has silently disarmed an armed abort. Before any
> packet inherits this row: identify the disarmed abort by name, every
> bare-construction call site, and whether any PRODUCTION entry point
> (R128 reachability: subprocess, -m, console-script shapes included)
> reaches one. A production-reachable bypass is a MINT-BLOCKING
> correctness row filed immediately with the evidence; a test-only bypass
> routes to the hygiene packet. The duplicate-default-authority half
> routes to the hygiene packet in either fork (R79 single-authority
> shape: one construction path, bare construction made unrepresentable or
> loudly refused).
> (c) RQ-16's re-scope, binding on the hygiene packet: the dead-transfer
> class is adjudicated PER TENSOR with an R128-shape reachability census
> over production entry points — read-site grep alone proves nothing in
> either direction; a src/ path whose only reachable caller is a test is
> a LAW-08 finding, not a deletion license, until the census says so. The
> "A4 two-views" citation is traced to a surviving artifact or declared
> unresolvable; an unresolvable contract claim re-derives from code or is
> retired with a note — it is never enforced from memory.
> (d) The gate-18 HALT is ratified; the freeze-governance packet is
> RE-SCOPED into one coherent event, in order: (1) gate-17/rule-7 scan of
> the freeze registers and freeze_verify tool; (2) relocation of both
> into the engine repo (they pin engine files; colocation retires the
> cross-repo secret and the phantom-gate shape, and resolves RQ-4's
> placement question); (3) the freeze policy clauses R289(a)/(b) applied,
> with the ORACLE-WRITE leaf-instructions document CREATED there — the
> second landing site R289(a) named did not exist, architect ledger;
> (4) the owed de-triplication; (5) gate 18 wired natively in the engine
> repo's battery and workflow; (6) the RQ-22 row-30 evaluation under the
> stated freeze criterion.
> (e) R43 clarification of record: an architect order that names a frozen
> path carries the R43 grant for exactly the ordered change, and same-act
> re-pinning is a TERM of every such grant. The item-0 late re-pin is
> ratified as disclosed-not-backdated; its lesson stands verbatim: the
> same-act discipline does not survive being known, only being checked —
> which is gate 18's mandate.
> (f) The packet's (t)-for-(v) misreference and R289(a)'s phantom landing
> site are annotated in the register foot, append-only, architect ledger.
> (g) Operator item, two minutes, before any repo-visibility change:
> authenticate gh on the working host — unauthenticated access works only
> while the engine repo is public.
> (h) Routing: this follow-up executes (b)'s investigation and the
> bookkeeping; the freeze-governance packet dispatches fresh under (d)'s
> scope; the hygiene packet dispatches after (b)'s fork resolves, carrying
> (b)'s routed half and (c); the CI-runtime packet is unaffected and
> dispatches on operator forwarding as authored.
> ROUTE: CURRENT SESSION.

---

## REGISTER-FOOT ANNOTATIONS — ordered by R290(f), architect ledger, APPEND-ONLY

**Both items are recorded, neither is repaired.** Nothing above this line is edited; the two
annotated texts stand verbatim where they are. This block exists because a future reader
following either reference would land somewhere the register does not go, and would reasonably
conclude the register had drifted.

**ANNOTATION 1 — the (t)-for-(v) misreference in the R289 packet's framing prose.** The R289
packet's preamble reads *"This packet IS the ruling's one canonical home (R289(t) below ends the
double-carriage class)"*. **The clause that ends the double-carriage class is (v), not (t)** —
(t) disposes of RQ-b2 and RQ-c and says nothing whatever about carriage. The misreference is in
framing prose, not in the numbered text, so **no clause of R289 is affected**. R290(f) promotes
this from the R289 carriage note (where the executing session recorded it on the day) to the
register foot, which is what makes it findable by a reader who never opens R289's header block.

**ANNOTATION 2 — R289(a)'s phantom landing site.** R289(a) directs the freeze-practice amendment
to land in two places: *"the conventions doc and the ORACLE-WRITE leaf instructions"*. The
conventions doc exists (`plan/PACKET_RECORD_CONVENTIONS.md`). **Nothing called the ORACLE-WRITE
leaf instructions existed anywhere** — not in this workspace, not in the engine repo — when the
clause was written, so half the order named a file that could not be edited. R290(d)(3) resolves
it forward by ordering the document **CREATED**, in the engine repo, as a step of the re-scoped
freeze-governance event. Annotated here as an architect-ledger item per R290(f): the clause was
executable only after a document it presupposed was written.

**Neither annotation carries a correction to any numbered clause.** R285(c)'s shape: annotate,
never repair.

**ANNOTATION 3 — the two R300(b) figure corrections, LEDGERED here as a POINTER (R301(a)).** R301(a)
records that the `_check_structural` band overshoot and P2's median are **the architect's, on the
ledger**. Their text is the `**ANNOTATION under R300's foot**` block and stays there — this row is a
pointer, not a second copy (R285 ONE-TEXT). Added by the R301 landing session so that a reader who
comes to the architect ledger looking for the mission's ledgered items finds them, which is exactly
what R290(f) created this block to do. **Neither figure touches R300's verdict**, which turns on P4
against the 166 line and on neither pre-registered falsifier firing.

**On the sentence immediately above this one:** it reads "Neither annotation", having been written
when the ledger held two. It is left as written — this block is APPEND-ONLY and the line is a
past-tense statement about the items that preceded it. Its property holds for all three: **no
annotation in this block corrects a numbered clause.** Recorded rather than repaired, because a
tally embedded in prose has to be re-edited on every append and is then read as evidence — R8's
derive-or-delete rule, arriving in the register instead of a file header.

**ANNOTATION 4 — CARD-RUN5-GPU-OOM's site wording, CORRECTED by R302(c) from new measurement.**
R114 routes the card with a diagnostic direction that reads the OOM as an inference-side event:
*"trace the single ~12.8 GiB allocation event to its tensor provenance; prime suspects are an
unbounded inference-server batch or a pathological graph exceeding the edge-cap's intended
domain"*. **The measured site set is now wider.** The VAST-REBUILD provisioning burst on the
replacement host OOM'd in the **GNN training forward** — `src/mantis/model/gine.py:71`
(`msg = (xs.index_select(0, src) + e).relu()`), 4.77 GiB requested against 3.29 GiB free, peak
occupancy 14798/16303 MiB — reached through
`run_training_loop → coordinator.step → _run_training_step → _graph_step →
trainer.train_step_from_graph_batch → forward_batch`, 3.3 s after the first self-play game
completed. Self-play inference ran 95 s clean in the same drive and completed a game, so the
failure is on the **trainer's** term of the partition, not the inference server's.
**What this annotation does NOT do.** It does not withdraw R114's diagnostic direction: that
direction was written against the F-816-10 measurement of an inference-side allocation and
remains the right first question when the OOM is inference-side. It widens the site set the card
covers, so that a future reader who measures a trainer-forward OOM does not conclude the card is
about something else and open a second one.
**And the class reading has changed under R302(c).** An OOM at a site the caps were fitted for is
a **stale mint on a changed host**, not a code defect — the caps were fitted against a partition
measured on a container R301(c) destroyed. That is why this instance routes to a re-calibration
box event rather than to an engine packet. Evidence: `plan/BOX_PREP_RECORD_2026-08-21.md` §7 and
`plan/BOX_PREP_2026-08-21/burst_events.jsonl` + `burst_traceback.txt`.

**ANNOTATION 5 — R304(b)'s SPLIT IS TOO COARSE, and the defect is in its operative half. Corrected
here by the same session that wrote it, 2026-08-22.** R304(b) separated two things: an exact
**rules** fact (the board automorphism group) and a **model** property (invariance of a trained
net, measured FALSE for the GNN). It then ordered: *"Land the field named for the rules fact it can
support … and augmentation policy may read it."* **That order is wrong, and drafting the design
against real code is what exposed it.**

**There are THREE facts under one name, not two.** (i) the **board** automorphism group — a rules
fact, exact, twelve elements, identical for every arch and therefore useless as a *per-arch*
capability; (ii) whether the **ENCODING** commutes with a given automorphism, so that a transformed
position encodes to the correspondingly transformed encoding — a static, per-arch property; (iii)
whether the trained **NET** is invariant under it — measured, FALSE for the GNN, magnitude unmeasured.

**Augmentation validity needs (i) AND (ii). It does not need (iii).** R304(b) collapsed (ii) into
(i), so a field carrying the rules fact and read by augmentation policy would authorise augmenting
**every** record — including the ones the engine already refuses. `crates/mantis-selfplay/src/replay/
hexg/sample.rs:168-176` forces identity for 0-stone records and states its grounds in the code:
*"the empty-board fallback rectangle is not closed under 8 of the 12 D6 elements, and an empty
stone list carries no orientation to learn."* That is (ii), already implemented, already carved out,
with the reason written down. A capability field built on R304(b) as worded would have declared a
symmetry exact where the encoding is measurably not closed under it.

**What survives R304(b) unchanged**, and it is the clause's load-bearing half: a capability surface
must not carry a model-equivariance claim, and **the inverted reading — declared-exact therefore
augmentation unnecessary — remains the live hazard**, backwards for a non-equivariant net. The
correction is to the decomposition and to which fact the augmentation-facing field carries, not to
the holding.

**Corrected order, superseding R304(b)'s final two sentences on the field's naming:** the
augmentation-facing capability declares **(ii)** — the automorphisms under which THIS ARCH'S
ENCODING is exact — and its name must say `encoding`. (i) is a rules constant and belongs in the
rules layer, not in per-arch caps. (iii) is a measured property that belongs to a trained
checkpoint, never to an arch declaration, and the orbit probe R304(b) ordered remains ordered for
it. Carried into `plan/DESIGN_ARCHCAPS.md` §2 as a pre-registered constraint.

**ANNOTATION 6 — fact (ii)'s TYPING is CORRECTED: it is per-record and per-site, not "a static,
per-arch property". Ordered by R307(c), appended 2026-08-22 by the MISSION PLANC-SEAM-M1 MAIN
dispatcher; found by Leaf 0 and re-derived independently before it was credited.** ANNOTATION 5
types fact (ii) — whether THIS ARCH'S ENCODING commutes with a given automorphism — as *"a static,
per-arch property"*. **That typing is wrong at engine HEAD** (`dev` = `47b78f9`). The engine decides
losslessness **per record and per site, on both arms**, and never as a per-arch constant. The four
sites, each verified at this landing by the coordinates given (§0 premise check, symbol-located):

- **Graph arm, per record — 12 or 1.** `crates/mantis-selfplay/src/replay/hexg/sample.rs:172`:
  `let sym = if augment && !rec.stones.is_empty() { self.rng.random_range(0..N_SYMS) } else { 0 };`
- **Dense arm, per record — 12 or 4.** `crates/mantis-selfplay/src/replay/sym.rs:134-140`,
  `draw_record_sym(rng, compact)`: `compact` → full `N_SYMS`, else `draw_window_preserving_sym`.
- **Dense arm, per site — permanently 4.** `crates/mantis-selfplay/src/runner/game.rs:675-676`
  draws the flat window-preserving element, with the grounds stated **in its own comment at
  `:668-674`** as well as at `sym.rs:719-727`: the per-game sym is drawn before the first stone,
  *"whose compactness is unknowable here — there is no record yet to certify. So this IS the
  per-record gate, evaluated at the only moment it can be."*
- **Python mirror, per row — 12 or 4.** `src/mantis/data/augment.py:214-229`, `draw_record_syms`,
  gated on each row's `spread` flag.

The engine states the principle itself, at `sym.rs:129-131`: *"the gate is the per-record evaluation
of 'is this element lossless HERE', and the subgroup is the answer whenever the record cannot be
certified."*

**ANNOTATION 5's own citation for (ii) is what contradicts its typing of (ii).** ANNOTATION 5 cites
`hexg/sample.rs:168-176` as the evidence that (ii) is real and already carved out — and that site
**is** the per-record gate. The citation was read as proving the fact EXISTS; it equally proves the
fact is **not a per-arch constant**. Recorded plainly because this is the second time in two
annotations that drafting against code has corrected the same clause, and the reusable half is the
same both times: **a ruling about code is falsified by code, not by re-reading its own prose.**

**Consequence, ruled in R307(b), stated here so the annotation is not read as a naming tweak:**
`caps.exact_symmetries` is **DELETED** from the ArchCaps design. A frozen per-arch set over a
per-record mechanism admits exactly two readings and Leaf 0 broke both — the **union** authorises
augmentation the engine refuses (ANNOTATION 5's own stated failure, arriving through its own
corrected field), the **intersection** collapses to `{identity}` for `GnnArch`, which is the
**inverted reading** through the opposite door. **The per-record gate IS the symmetry authority,
singular**, and consumers derive symmetry facts through it at point of use.

**What SURVIVES, unchanged:** no capability surface carries a model-equivariance claim, and **the
inverted reading remains the named live hazard** — R307(b) closes its last door by removing the
field that could have carried it. ANNOTATION 5's three-fact decomposition stands; what moves is the
typing of (ii) and, with it, the field. As with ANNOTATION 5 itself: **the correction is to the
decomposition's typing, not to the holding.**


**ANNOTATION 9 — R257's "search-free" premise is QUALIFIED, not repaired. Ordered by R316(d),
appended 2026-08-28 by the R316 landing.** R257's second fence describes the reference bot's net
as acting *search-free via the KLENT operator*, without qualification. **That holds of its
TRAINING loop and not of the artifact it serves**: the deployed checkpoint searches — Gumbel
sequential halving at 16–128 sims per stone. The ground is
`plan/research/REFBOT_EVIDENCE_shrimp.md` §6 at `mantis-migration @ 0af92d7`, read at that
source's own `9c94b95`; the figures live there and are deliberately not restated here (R287(a):
cite the producing record, do not copy its numbers into a second home where they can drift).

**Nothing in R257 is edited and no clause moves.** The fence's premise is narrower than its text,
which is a statement about the PREMISE and not about the holding — and the direction is the
favourable one: an artifact that searches at deploy is independent corroboration of the R254/R258
deploy lock, not a challenge to it. R316(d) accepts REFBOT-SCAN-1 as evidence on disk and parks
every candidate on the rail; nothing enters the tree on this annotation.

**Why an annotation and not a repair, stated once for the class.** A landed ruling is read from
the register, never rewritten by a mission (R9/R289(v)). A premise that has narrowed since it was
written is exactly what the annotation form exists for: the reader who follows R257 to its fence
must find the qualification attached to it, and must not find a text quietly different from the
one the operator ratified.

**ANNOTATION 10 — R257's `[r8]` FENCE IS NARROWED IN SCOPE, not lifted, and R257 is NOT repaired.
Ordered by R322(b), appended 2026-08-30 by the R322 landing.** R257's fence (i) reads as a blanket:
*"Shrimp-Bot is the radius-8 game; mantis run5 is radius-6 (R26/R238). Architecture and loop
patterns transfer; corpora, checkpoints, and any radius-dependent arithmetic do not."* Read
blanket-wise — and it has been, in every derived doc that tags a reference quantity `[r8]` — it
says every quantity from a radius-8 reference must be re-derived before it grounds anything.

**What RESEARCH-SCOUT-1 measured, and it cuts the other way.** The strix curriculum is not a
board-size ladder; it walks `win_length` and `placement_radius` on an unbounded board, and its
**stage S5 is `win_length 6`, `placement_radius 6`, unbounded — this project's exact rule set.**
The ground is `plan/research/SCOUT_2026-08-30_CANDIDATES.md` §1 G6, which states the stage's own
knobs; our half of the match was re-derived at its producer this session and again at this landing:
`crates/mantis-encoding/src/registry.toml` `[encodings.gnn_axis_v1]` gives `win_length = 6`,
`graph_radius = 6`, `legal_move_radius = 6`. Per R287(a) the stage's figures are cited to that
record and deliberately not copied here.

**The narrowing, stated so it cannot be over-read.** The fence attaches to their **published
figures** — every one of which is taken at radius 8 or on synthetic planes — and to their
**non-S5 stages**. It does NOT attach to S5-stage material, which is stated at our rules and
transfers without a radius re-derivation. `[r6-MATCH]` is the label for that class. Three things
the annotation does not do: it does not lift the fence for any figure, because no strix figure is
published at S5; it does not touch fence (ii), the acting-scheme divergence, which remains an
operator lock (R254/R258, and ANNOTATION 9's qualification of its premise); and it licenses no
transfer that skips the seam — S5-stage material enters behind the contract with its witnesses,
like anything else.

**Why an annotation and not a repair, restated for this instance.** R257's holding is unchanged and
correct; what has changed is that a reference we knew only at radius 8 turns out to have a stage at
radius 6. That is a fact about the reference, not an error in the ruling, and the reader who
follows R257 to its fence must find the narrowing attached to it rather than a text quietly
different from the one the operator ratified (R9/R289(v), and ANNOTATION 9's own stated reason).

# R291 — architect adjudication response, operator-ratified by forwarding, 2026-08-20 (R290's execution ratified; F-816-24 RATIFIED MINT-BLOCKING with a fix packet ORDERED; the LAW-08 citation-prose defect fixed as a CLASS; R289(c) slotted; R289(v) matured into read-from-the-register) [INLINE]

**Provenance: [INLINE], verbatim.** Text supplied in the R291 follow-up packet under its own
heading `## R291 — CANONICAL TEXT`, operator-ratified by forwarding per the R261 shape,
2026-08-20, under the R283(b) carry rule. Header title follows the R280 forward-only convention.

**Carriage note — R285(d), third consecutive NOT-APPLICABLE, and the last time the check will mean
what it used to.** The packet states in its own preamble that it *"is R291's ONE canonical home
(R289(v))"* and carries the ruling **exactly once**, so the mechanical two-copy comparison has
nothing to compare. **Clause (h) then matures the rule past the check itself:** from R291 forward a
landed ruling is READ from this register at point of use and copied nowhere, so the failure mode
R285(d) was built to detect — two copies that differ — becomes unrepresentable rather than merely
undetected. The check is not retired here; it is left standing over a class that (h) empties.

**Structural note, so nothing reads as drift.** R290's section ends with a
`## REGISTER-FOOT ANNOTATIONS` block ordered by R290(f). Appending R291 below it moves that block
off the literal foot of the file. **Nothing is edited, moved or re-titled** — append-only holds
(R271(a)) — and the block stays findable by its own heading, which is what R290(f) points at. Its
header carries no R-number, so it disturbs neither the section count nor the distinct-number set.

> R291 — (a) The R290 execution is ratified. The architect's premise
> conflation ("both bypasses and disarmed" — measured as two disjoint
> paths) is on the architect's ledger; the fork survived the error because
> it was stated as a test, not a narrative — pre-registered tests
> generalize to investigation orders, on the record. The census method
> (grep AND AST, 31% grep over-report measured) is the standing method for
> construction censuses.
> (b) F-816-24 is RATIFIED MINT-BLOCKING with its re-stated severity: the
> supervisor entry surface (python -m mantis.monitor.supervise) reads no
> minted config, so RUN5_MINT_PREREG row 19's safety bound cannot
> discharge — config-authored grace values reach no process. A fix packet
> is ORDERED, fresh dispatcher, full R262 pipeline with cross-model
> review: (i) the supervisor takes a REQUIRED --config; absent = named
> error, never a default (R1/LAW-11 shape); (ii) monitor config resolves
> through the ONE resolver; the bare MonitorConfig() at that site becomes
> unrepresentable or loudly refused; (iii) LAW-07 producer test = a
> real-drive witness launching the supervisor via -m with a distinctive
> minted grace value and asserting the value live in the process, plus
> the refusal test; (iv) the Q3-landed signal posture and escalation
> ladder are untouchable — the fix rides the tested seam; (v) the packet's
> exit carries the 14-surface config-reachability census: every -m
> production surface either demonstrably loads the minted config or is
> recorded config-free with grounds; further config-blind surfaces become
> rows, and only then is a census gate considered.
> (c) The LAW-08 citation-prose defect is fixed as a CLASS (R71, second
> instance in one file): the consumer-citation checker verifies every
> arrow by symbol reference (the R244 grep-derived shape), never by prose;
> the neighbouring drain-block fix is the pattern; the class fix rides the
> F-816-24 packet as its own commit.
> (d) Gate 12's scope answer is accepted: boundary, not finding; its scope
> statement is truthful.
> (e) R289(c)'s independent freeze_verify review is SLOTTED: after
> freeze-governance step (2) relocation, before step (5) gate wiring —
> the tool is reviewed in its new home before it becomes a gate.
> (f) The CI-runtime packet's stale collision line and census stamp are
> refreshed by that packet's own Task 0 at dispatch, never churned before.
> (g) The "39 fields" figure correction (32 top-level / 37 leaf /
> MonitorConfig 29) and the CLAUDE.md console-scripts zero-members note
> are accepted as recorded; the load-bearing check (config − schema = ∅)
> stands.
> (h) Protocol maturation of R289(v): once a ruling is landed in the
> register, subsequent packets READ it from the register at point of use
> and carry NO copy — the register is the one place. Dispatches cite the
> number and the file, never the text.
> (i) Operator precondition, restated: gh auth login on the working host —
> freeze-governance step (2) waits on it.
> ROUTE: CURRENT SESSION for this landing; FRESH DISPATCHER for the
> F-816-24 packet, which reads R291 from the register.


# R292 — architect adjudication response, operator-ratified by forwarding, 2026-08-20 (R291's execution ratified; the R79 duplicate-authority half SPLIT between two packets; the hygiene stop LIFTED except one item; the overlap check adopted as a standing instrument) [INLINE]

**Provenance: [INLINE], verbatim.** Text supplied in the R292 follow-up packet under its own
heading `## R292 — CANONICAL TEXT`, operator-ratified by forwarding per the R261 shape,
2026-08-20, under the R283(b) carry rule. Header title follows the R280 forward-only convention.

**Carriage note — R285(d) NOT-APPLICABLE, fourth consecutive, and now by DESIGN rather than by
luck.** The packet carries the ruling exactly once and says so in its own preamble, citing both
R289(v) and R291(h). R292(d) rules the check **noted-not-retired**: it stands over the class R291(h)
empties, at zero cost, so that a regression of the protocol itself is caught by the same mechanism
that used to catch a double carriage. The verdict is recorded, not skipped.

**Operator item DISCHARGED DURING this landing, recorded because the clause restates it as open.**
R292(e) restates R291(i) *"without change"* — `gh auth login` on the working host, gating
freeze-governance step (2). **The operator ran it mid-session, before this append.** Measured
immediately after, on the working host: `gh auth status` reports `✓ Logged in to github.com account
[REDACTED:local:rule7_local_terms.txt:26:8f2fd2d5] (keyring)`, active account true, git protocol ssh, token scopes `gist`, `read:org`,
`repo`. **The gate is discharged**, and `repo` is the scope a private-repo read and any
visibility change both require, so step (2) is unblocked on this axis. Nothing in the clause is
amended — a clause restating a gate that closed while the packet was in flight is not wrong, it is
merely overtaken, and the discharge is evidence appended beside it rather than an edit to it.

> R292 — (a) The R291 execution is ratified. The packet-authorship overlap
> check (shared 7-word runs against the register, found 32 in the first
> draft, rewritten before landing) is ADOPTED as a standing check for
> every future packet — the anti-carriage discipline now has an
> instrument, and a packet's own commit subject records its result.
> (b) Scope ruling on the duplicate-authority (R79) half: the F-816-24
> fix packet owns the supervisor site — the required --config, the
> single-resolver path, and the refusal-or-unrepresentability mechanism
> as built at that site. The hygiene packet owns the class-wide
> extension — every remaining bare-construction site including the
> test-only coordinator site, the resolver-only construction authority
> for src/, and a stated rule for the test-side constructions — and
> takes that ONE item up only after the F-816-24 packet merges, with the
> fix packet holding right-of-way on monitor/supervise.py and on the
> refusal mechanism. THE HYGIENE STOP IS LIFTED for every other hygiene
> item, effective on this ruling's landing.
> (c) The F-B1 re-opening is ratified as DESIGN input for the F-816-24
> packet: a supervisor that reads its own config re-opens the
> parent-child same-file binding question; config_identity_sha256 is the
> named candidate binding; DESIGN answers it explicitly and RED-TEAM
> verifies the answer exists — the question must not be discovered late.
> (d) Accepted as recorded: the O-18 measurement with its
> re-derive-at-head instruction; the register-foot annotation position
> note (append-only holds, the block is findable by heading); R285(d)
> noted-not-retired — the check stands over the class R291(h) empties,
> at zero cost, catching regressions of the protocol itself.
> (e) Standing operator items, restated without change: gh auth on the
> working host gates freeze-governance step (2); the box sitting remains
> the critical-path item and is wholly operator-scheduled; the F-816-24
> packet dispatches on forwarding and reads R291/R292 from the register.
> ROUTE: CURRENT SESSION.


# R293 — architect adjudication response, operator-ratified by forwarding, 2026-08-20 (R292's execution ratified; the reconciliation-grep convention adopted; the gh precondition DISCHARGED; the architect desk declared EMPTY with four packets forwardable) [INLINE]

**Provenance: [INLINE], verbatim.** Text supplied in the R293 follow-up packet under its own
heading `## R293 — CANONICAL TEXT`, operator-ratified by forwarding per the R261 shape,
2026-08-20, under the R283(b) carry rule. Header title follows the R280 forward-only convention.

**Carriage note — R285(d) NOT-APPLICABLE, fifth consecutive.** One copy, nothing to diff. The check
remains armed per R292(d) over the class R291(h) empties; the verdict is recorded rather than
skipped, which is the whole point of keeping a zero-cost check pointed at an emptied class.

**A note on what this ruling is, because an empty desk is easy to misread.** R293(c) does not close
the migration, discharge a card, or lift an exclusion. It records that **nothing is waiting on the
architect** — every disposition is ruled, no row is halted, no adjudication is owed — and that the
next architect act is triggered by an operator event (the box readout), not by a queue item. **The
liabilities that were open before this ruling are open after it**: the owed texts R227 / R228 /
R267, and the prereg values. An empty desk is a statement about the *queue*, not about the *work*.

> R293 — (a) The R292 execution is ratified. On the record: the overlap
> check's first production use caught its own author (18 shared runs in
> the hygiene draft, 0 at landing) — the adoption argument, demonstrated;
> and the four stale cross-references found after the head replacement
> adopt one convention line: replacing a section carries a same-edit grep
> for references to the replaced content, reconciled in the same act.
> (b) The gh authentication evidence is accepted in the R285(c) shape —
> recorded beside the clause, no clause edited. Freeze-governance step
> (2)'s precondition is DISCHARGED; the packet dispatches on operator
> forwarding under its existing R290(d) authority — no new grant exists
> or is needed.
> (c) The architect desk is empty: no open dispositions, no halted rows,
> no owed adjudications. Forwardable now, in any order: the F-816-24
> packet (mint-blocking, local-only, independent of the box),
> PACKET_HYGIENE.md, PACKET_FREEZE_GOVERNANCE.md, PACKET_CI_RUNTIME.md.
> Operator-scheduled: the box sitting (BOX_BLOCK.md), whose readout
> triggers the next architect act — the re-bench verdict. The owed texts
> (R227/R228, R267) and the prereg values remain where they were.
> ROUTE: CURRENT SESSION.


# R294 — architect adjudication response, operator-ratified by forwarding, 2026-08-20 (R293's execution ratified with its scope stretch adopted as principle; re-check-at-dispatch ratified; the past-tense exclusion fixes the reconciliation grep's scope; the board unchanged) [INLINE]

**Provenance: [INLINE], verbatim.** Text supplied in the R294 follow-up packet under its own
heading `## R294 — CANONICAL TEXT`, operator-ratified by forwarding per the R261 shape,
2026-08-20, under the R283(b) carry rule. Header title follows the R280 forward-only convention.

**Carriage note — R285(d) NOT-APPLICABLE, sixth consecutive.** One copy, nothing to diff; armed per
R292(d) over the class R291(h) empties, recorded rather than skipped.

**Shape note: this ruling is the R293(a) convention's own first amendment, and all three of its
substantive clauses came out of executing it once.** (a) rules how far an adoption sweep reaches,
(b) rules how long a discharge is good for, and (c) rules what the sweep must not flag. A convention
that acquires its boundaries from its first use — rather than from a specification written in
advance — is the shape LAW-01 asks for; recorded because the pattern is worth recognising the next
time a convention is adopted.

> R294 — (a) The R293 execution is ratified, including the disclosed
> scope stretch: sites 2–4 lived outside the landing's own files, and
> repairing them in the same act was correct — adopted as principle, a
> convention lands together with the repairs its adoption sweep finds
> (the R98 clean-baseline rule applied to conventions; gate 17's empty
> exemption list is the precedent). (b) The re-check-at-dispatch
> instruction on discharged preconditions is ratified — a discharge is
> evidence at a moment, and dispatch re-verifies it. (c) The past-tense
> exclusion is ratified: "(was: …)" state history is provenance, never a
> stale live claim; the reconciliation grep's scope is live assertions
> only. (d) The desk remains empty; the board is unchanged from R293(c):
> four packets forwardable, the box sitting operator-scheduled, the
> re-bench verdict the next architect act. ROUTE: CURRENT SESSION.


# R295 — architect adjudication response, operator-ratified by forwarding, 2026-08-21 (R294's execution ratified with both conventions' first-run arguments put on the record; MISSION FINISH-LINE established over the four authored packets, the held hygiene item and three closing deliverables; the dependency order fixed; the R287(b) grant pattern renewed by forwarding and NO new authority issued) [INLINE]

**Provenance: [INLINE], verbatim.** Text supplied in the MISSION FINISH-LINE dispatch under its own
heading `## R295 — CANONICAL TEXT`, operator-ratified by forwarding per the R261 shape, 2026-08-21,
under the R283(b) carry rule. Header title follows the R280 forward-only convention.

**Carriage note — R285(d) NOT-APPLICABLE, seventh consecutive.** One copy in the dispatch, nothing
to diff. The mission preamble points at the canonical block by its heading instead of restating it,
which is R291(h)'s own discipline applied to a ruling that lands in the same act it authorizes.
Armed per R292(d) over the class R291(h) empties, recorded rather than skipped.

**Shape note: this is the first mission-establishing ruling that issues no grant of its own.**
R287 had to carry its mission grants explicitly. Here (b) renews the R287(b) pattern by forwarding,
(c) sequences work whose authority already lives in the packets, and (d) names deliverables rather
than permissions. The load-bearing sentence is inside (b) — *this mission adds sequencing, not
authority* — with the corollary stated twice over: each packet's own hard limits, pipelines and
exit criteria govern inside it, and (c)'s parallelism is re-measured at dispatch rather than
inherited. The distance between R287's explicit grant list and R295's empty one measures how much
of the authority structure has since moved into the packets themselves. Labelled as grounds, NOT
part of the numbered text.

> R295 — (a) The R294 execution is ratified. On the record with its
> first-run argument: the past-tense exclusion exists because an unbounded
> stale-claim grep would, over successive curations, consume the change
> log it exists to protect; and boards are pointed at, never copied
> forward — a restated unmoved board is how a stale board is eventually
> trusted. (b) MISSION FINISH-LINE is established: scope = the four
> authored packets (PACKET_F81624_SUPERVISOR_CONFIG, PACKET_HYGIENE,
> PACKET_FREEZE_GOVERNANCE, PACKET_CI_RUNTIME), the hygiene item held
> behind the F-816-24 merge, and three closing deliverables; grants = the
> R287(b) mission pattern renewed by operator forwarding; each packet's
> own hard limits, pipelines, and exit criteria govern inside it —
> this mission adds sequencing, not authority. (c) Dependency order is
> fixed: F-816-24 dispatches first among engine-touching work and holds
> right-of-way per R292(b); freeze-governance re-verifies the gh
> discharge at its own dispatch (R294(b)); CI-runtime executes its Task-0
> staleness refresh before anything else in it; the hygiene held item
> starts only after the F-816-24 merge lands on dev. Parallel execution
> is permitted exactly where the packets' collision tables say it is —
> re-measured at dispatch, never inherited. (d) Closing deliverables:
> (M3) RUN5_MINT_PREREG row 19's DISCHARGE-BLOCKED-ON note flips to
> DISCHARGEABLE when the F-816-24 real-drive witness is merged and green
> on CI — the witness is the clearing condition, as the note states;
> (M4) a fresh architect STATE snapshot superseding the 2026-08-16
> lineage plus a refreshed session-handoff seed, both swept under the
> stale-claim conventions; (M5) CLEAN REPORT 2, ending with the
> who-owes-what list — operator: the box sitting, the prereg values, the
> R227/R228 and R267 texts; architect: the re-bench verdict and any rows
> this mission queues. (e) The mission ends when every packet's exit
> criteria are met with remote CI green, the held item is done, and
> M3–M5 are delivered — or when the operator stops it.
> ROUTE: FRESH DISPATCHER — mission MAIN; all other ruling text is read
> from plan/rulings_register.md at point of use (R291(h)).


# R296 — architect adjudication response, operator-ratified by forwarding, 2026-08-21 (Checkpoint 3 and the Q1 exit ratified with all six decisions; F-816-25/-26 RULED into mission items, F-816-27 rides prereg row 19; gate 14's interpreter fixed by repo toolchain pin; STRUCTURE-NOT-TEXT adopted; the box sitting joins as Q6 under the renewed R282(c) grant; token riders bind prose, never checks) [INLINE]

**Provenance: [INLINE], verbatim.** Text supplied in the R296 follow-up packet under its own heading
`## R296 — CANONICAL TEXT`, operator-ratified by forwarding per the R261 shape, 2026-08-21, under
the R283(b) carry rule. Header title follows the R280 forward-only convention.

**Carriage note — R285(d) NOT-APPLICABLE, eighth consecutive.** One copy; the packet names it
"R296's ONE canonical home" in its own preamble. Armed per R292(d) over the class R291(h) empties.

**Shape note — this ruling closes the loop on a mission's own findings for the first time.** R295
established the mission; R296 rules the rows that mission FILED and turns three of them into
mission items. The pattern is worth naming: a mission that files rows to the architect and then
receives them back as scheduled work is doing what the rulings queue exists for, and (b)'s
sequencing clause — F-816-25 before any BC-pretrain — is the first time a filed row has come back
carrying an ordering constraint on work nobody has dispatched yet. Labelled as grounds, NOT part of
the numbered text.

> R296 — (a) Checkpoint 3 and the Q1 exit are ratified, all six recorded
> decisions included: the full-RunConfig load (the config_identity_sha256
> mechanical argument governs), the retained override flags with the
> re-argued justification on the record ("a fence risk that isn't real is
> worse than no argument — it gets quoted back later as though the fence
> had decided"), publish-don't-compare with F-816-26 filed, the declined
> census gate with its semantic-discriminator grounds, the DECLINED
> kill-grace ceiling confirmed as the correct refusal (it is half of an
> open prereg row), and fix-forward over revert. The five-reviewer
> sequential isolation result stands as the argument stated as a result.
> (b) F-816-25 (pretrain CLI shadows five minted train.* keys, three
> divergent at run5) is RULED: R79 duplicate-authority class; the
> shadowing flags are removed or become config-only; the fix is a mission
> item SEQUENCED BEFORE any BC-pretrain execution, because the pretrain
> CLI sits directly on the bootstrap path the operator may choose at
> prereg — a divergent shadow there would silently un-mint the bootstrap
> row's own values.
> (c) F-816-26 (parent/child config binding) is RULED: the child-side
> comparison of config_identity_sha256 against the parent-published value
> lands as a mission item, completing the F-B1 re-opening; mismatch is a
> named refusal, never a warning.
> (d) F-816-27 (kill-grace ceiling) RIDES prereg row 19 to the operator
> sitting, its three filed options attached; no agent authors either side
> of that inequality.
> (e) Gate 14's interpreter is fixed by REPO TOOLCHAIN PIN — the
> rust-toolchain.toml pattern applied to node: a committed, portable
> version pin that any mise host auto-provisions; no host config is
> touched; the gate's refusal arm stays armed for hosts without mise.
> (f) The STRUCTURE-NOT-TEXT convention is ADOPTED (third instance of the
> class: the equal-length FFI defense, the raw-text citation checker, the
> bare-call construction guard): verification mechanisms derive from
> structure — AST, types, reachability — never from text matching; every
> existing text-matching guard found by later work converts on contact.
> (g) THE BOX SITTING JOINS THE MISSION as Q6, dispatcher-run under the
> operator's forwarded grant (the R282(c) pattern, renewed here):
> executed after the Q2–Q4 merges so the box measures final dev, per
> BOX_BLOCK.md verbatim — raw tables, no verdict, the band read from the
> register at point of use, orphan sweeps both sides, evidence off-box.
> Its readout triggers the architect's re-bench verdict. Operator-only
> residue after this mission: the prereg values, the R227/R228 and R267
> texts, and the mint word.
> (h) TOKEN RIDERS, binding on prose and never on checks: checkpoint
> reports at most one screen; censuses run only on register-touching acts
> and print one line; spot-checks proportionate (≥3 for pointer-only
> curations, ≥5 for substantive); exit reports lead with deltas and
> findings, not restated state; leaves carry bulk, MAIN carries state.
> ROUTE: mission MAIN, current loop.


# R297 — architect adjudication response, operator-ratified by forwarding, 2026-08-21 (checkpoint 6 ratified with its disclosed mis-scope; the NEGATIVE-SEARCH COROLLARY adopted as R296(f)'s completion; RQ-16 dispositioned per field with node_coords deletable; the halt record annotated not rewritten; the handoff plan of record set) [INLINE]

**Provenance: [INLINE], verbatim.** Text supplied in the R297 follow-up packet under its own
heading `## R297 — CANONICAL TEXT`, operator-ratified by forwarding per the R261 shape,
2026-08-21, under the R283(b) carry rule. Header title follows the R280 forward-only convention.

**Carriage note — R285(d) NOT-APPLICABLE, ninth consecutive.** One copy, named as such by the
packet's own preamble. Armed per R292(d) over the class R291(h) empties.

**Shape note — this ruling completes a convention on the evidence of the convention failing.**
R296(f) adopted STRUCTURE-NOT-TEXT for verification mechanisms. Within the same day the same class
appeared in a *search*: a case-scoped, file-type-scoped negative grep recorded an absence as a
fact, and that fact became leg 1 of a halt's premise verification. (b) completes the convention by
extending it to the negative case, which the original clause did not reach — a guard that matches
text and a search that fails to match text are the same instrument read in opposite directions.
Labelled as grounds, NOT part of the numbered text.

> R297 — (a) Checkpoint 6 is ratified, the disclosed grep mis-scope
> included; the census leaf's claims stand re-verified. On the record: the
> structure-not-text class bit the checker of the class within days — a
> case-scoped negative search recorded an absence as a fact.
> (b) The NEGATIVE-SEARCH COROLLARY is ADOPTED as R296(f)'s completion: a
> negative text-search result is evidence about the pattern, never about
> the artifact, unless its scope and case posture are stated beside it;
> it joins the M5 conversion census as a second axis (negative-search
> sites audited alongside text-matching guards).
> (c) RQ-16 dispositions, binding on the hygiene packet: node_coords is
> GENUINELY DEAD and deletable — one commit, the bridge's versioned
> flat-array contract handled per the packet's own rules; legal_mask,
> policy_dst_slot, window_center, current_player are TEST-ONLY LAW-08
> findings resolved PER FIELD — retire with test re-expression where the
> now-resolved A4 contract covers it (legal_mask re-expresses against the
> gather), wire a live consumer only where one is legitimately owed, one
> commit per field, never a blanket act; the C1/H2D perf measurement of
> the removals rides Q6 on the box as already staged.
> (d) The R289 halt's premise-verification record is annotated, not
> rewritten: leg 1 falsified by the resolved citation (with the two
> independent miss mechanisms stated), leg 2 untouched and sharpened.
> (e) Handoff plan of record: when the mission delivers CLEAN REPORT 2
> and the Q6 sitting record, the architect returns IN ONE TURN the
> re-bench verdict plus the operator's requested handoff package — the
> fresh-session starter prompt for the architecture program (PLAN-0 seam,
> WP-AXIS2, SYS-5, KLENT ordering), the updated project instruction file
> codifying the operator's working preferences and the matured protocols
> (one-canonical-text, routing, token riders, agent-executes/operator-
> forwards), and the memory update — all seeded from the mission's M4
> STATE snapshot so the new session starts lean. Bigger development moves
> to that fresh session; this session closes after the verdict + handoff.
> ROUTE: mission MAIN, current loop.


# R298 — architect adjudication response, operator-ratified by forwarding, 2026-08-21 (checkpoint 7 ratified and both judgement calls adopted as the per-field template; DEFERRED RATIFICATION in force for the mission remainder with the exclusions absolute; early return only on total block; the final surfacing is CLEAN REPORT 2 + the Q6 sitting record answered in one turn) [INLINE]

**Provenance: [INLINE], verbatim.** Text supplied in the R298 follow-up packet under its own
heading `## R298 — CANONICAL TEXT`, operator-ratified by forwarding per the R261 shape,
2026-08-21, under the R283(b) carry rule. Header title follows the R280 forward-only convention.
The packet names itself the operator's LAST forward before the mission's final return.

**Carriage note — R285(d) NOT-APPLICABLE, tenth consecutive.** One copy, named as such by the
packet's own preamble. Armed per R292(d) over the class R291(h) empties.

**Shape note — this ruling changes the mission's CONTROL LOOP, not its scope.** Every prior ruling
in this mission adjudicated work; (b) adjudicates the *cadence of adjudication itself*, moving from
per-checkpoint architect turns to batch ratification from on-disk records. That is only safe
because of a property the mission has been building throughout and which (b) leans on explicitly:
the records are **re-derivable in full** — every figure names its producing run, every guard is
mutation-proved, every census is scripted. Deferred ratification over an un-re-derivable record
would be unratifiable, not merely deferred. The exclusions it restates (frozen edits without
grants, armed values, prereg decisions) are the three classes no record can make safe after the
fact, which is why they stay absolute rather than deferred. Labelled as grounds, NOT part of the
numbered text.

> R298 — (a) Checkpoint 7 is ratified. Both judgement calls are ADOPTED as
> the pattern for the remaining per-field work: retired-key goldens are
> kept as evidence with the absence asserted — a capture whose bytes are
> rewritten on code change has stopped being a capture, and a check that
> skips unmatched keys passes by not checking; and retired-set authority
> lives in one module with the stated reason derivation alone cannot
> carry. The two-referents catch (wire array alive, device tensor dead)
> is on the record as R290(c)'s class one layer deeper.
> (b) DEFERRED RATIFICATION is in force for the remainder of MISSION
> FINISH-LINE: checkpoint reports land in the governance repo and do NOT
> return to the operator; the mission continues without per-checkpoint
> architect turns. Judgement calls of the disclosed-and-reasoned class —
> disclosed same-act, mechanism stated, mutation-proved where a guard is
> involved — proceed under STANDING APPROVAL and are batch-ratified by
> the architect at mission end from the on-disk records, which are
> re-derivable in full. The rulings queue remains monotone: file, skip,
> continue. Frozen edits without existing grants, armed values, and
> prereg decisions remain absolute exclusions — filed, never taken.
> (c) EARLY RETURN occurs in exactly one case: every remaining queue item
> is blocked. Otherwise the mission's next and final surfacing is CLEAN
> REPORT 2 together with the Q6 sitting record, which the architect
> answers in one turn: batch ratification of all interim checkpoints, the
> re-bench verdict against the R283(c) band read from the register, and
> the operator's handoff package per R297(e).
> (d) For the remaining Q2 fields: one commit per field on the checkpoint-
> 7 template; the A4 re-expression rides legal_mask's commit as its own
> reviewed change. For Q6: BOX_BLOCK.md verbatim, raw tables, no verdict.
> ROUTE: mission MAIN, current loop — land this, then run silent to the
> finish.


# R299 — architect adjudication response, operator-ratified by forwarding, 2026-08-21 (MISSION FINISH-LINE batch-ratified from the on-disk records; F-816-29 GRANTED diff-scoped; F-816-28 ruled by principle; F-816-30 ruled and MECHANISM-NOT-PROXY adopted; the re-bench verdict NOT issued and still owed; the handoff placed and this session lineage closed) [INLINE]

**Provenance: [INLINE], verbatim.** Text supplied in the R299 follow-up packet under its own
heading `## R299 — CANONICAL TEXT`, operator-ratified by forwarding per the R261 shape,
2026-08-21, under the R283(b) carry rule. Header title follows the R280 forward-only convention.

**Carriage note — R285(d) NOT-APPLICABLE, eleventh consecutive.** One copy, named as such by the
packet's own preamble. Armed per R292(d) over the class R291(h) empties.

**Shape note — the first ruling in this lineage to ratify REFUSALS as work product.** (a) puts two
declines on the record as *correct*: refusing to produce bench numbers from an inadmissible host,
and refusing to infer box identity from an ssh configuration. Both are cases where the executable
action existed and was not taken, and neither produced an artifact — so without this clause they
would appear in the record only as absences. A process that ratifies only what was *made* teaches
that not-making is invisible; this one does not. (e) is the same shape at mission scale: the
re-bench verdict is **not issued**, and the clause says why in terms of the bench's own
pre-registered admissibility rules rather than treating the gap as an omission. Labelled as
grounds, NOT part of the numbered text.

> R299 — (a) BATCH RATIFICATION per R298(b): every interim checkpoint of
> MISSION FINISH-LINE is ratified from the on-disk records — Q1/M3, the
> Q2 items delivered, Q2b, M4/M5 — including all disclosed judgement
> calls under the standing-approval class. The gate-3b caveat is accepted
> exactly as stated: a claimed-green sibling set is not a completed
> integration tier; the completed run's result is recorded when it
> finishes and the record says which claim it upgrades. The two refusals
> — declining inadmissible-host bench numbers and declining to infer box
> identity from ssh config — are ratified as correct and on the record.
> (b) F-816-29 is GRANTED, diff-scoped to the row's one-keyword-argument
> diff, same-act re-pin as a term (R290(e)), per-event, never precedent.
> (c) F-816-28 is RULED by principle: the resolution preserves BOTH
> invariants — a single exit-code-constant authority AND util's
> import-free property; the option satisfying both is taken; if none
> does, the primitive does not move and the row records why. The rc
> census test remains the authority-keeper either way.
> (d) F-816-30 is RULED: a skip guard's discriminator must detect the
> MECHANISM its docstring names (the BLAS path in use), never a proxy —
> platform.machine() is true on every runner and sees nothing.
> Mechanism-not-proxy is adopted as the structure-not-text convention's
> sibling. The fix rides the Q4 CI-runtime packet; R46 governs if
> deflaking is chosen instead.
> (e) The RE-BENCH VERDICT is NOT ISSUED: no after side exists; Q6's
> structural block was correct under the admissibility rules the bench
> itself pre-registered. The verdict remains owed against the unchanged
> R283(c) band, triggered by the box sitting's readout, in whichever
> session receives it.
> (f) HANDOFF per R297(e): ARCHITECT_SESSION_PROMPT v3 supersedes v2;
> the ARCHITECTURE-ERA session starter is the seed for the successor
> session; Q3, Q4, F-816-28, F-816-29-execution, and F-816-30 are the
> successor's carry-over dispatch queue; the operator's residue is
> unchanged — prereg values, R227/R228 and R267 texts, the box sitting,
> the mint word. This session lineage closes after this landing.
> ROUTE: CURRENT SESSION.


# R300 — architect adjudication response, operator-ratified by forwarding, 2026-08-21 (the ARCH-ERA verification census RATIFIED and the numbering HALT resolved — the head is R299 at f9988bd; THE RE-BENCH VERDICT IS ISSUED, STOP/INVESTIGATE, with the 0.911× regression ERASED and both predictions CONFIRMED; the residual perf track SCOPED and explicitly NOT mint-blocking; the Q6 adjudications ruled and F-Q6-1..8 routed; the SECURITY disposition set with rotation as the remedy and no history rewrite; git-over-SSH fixed as an environment rule; the carry-over queue of record) [INLINE]

**Provenance: [INLINE], verbatim.** Text supplied in the ARCH-ERA bridge packet under its own
heading `## R300 — CANONICAL TEXT`, operator-ratified by forwarding per the R261 shape,
2026-08-21, under the R283(b) carry rule. Header title follows the R280 forward-only convention.

**Carriage note — R285(d) NOT-APPLICABLE, twelfth consecutive.** One copy, named as such by the
packet's own preamble ("R300's ONE canonical home"). Armed per R292(d) over the class R291(h)
empties.

**Premise verification, per R289's precondition, run BEFORE this landing.** (i) The head claim in
(a) is TRUE by measurement: `f9988bd` is `book(R299): ruling landed verbatim and ACTIVE to v3.4`,
and the pre-append census reads R23–R299, 270 sections, 270 distinct, 0 duplicates. The
project-knowledge snapshot the ARCH-ERA session halted on was a pre-append copy, exactly as the
packet says. (ii) Every figure in (b) was checked against `plan/Q6_BOX_EVIDENCE/P1_P4_TABLE.md`
and `GO_NO_GO_READINGS.md`, which are tracked. P1 20→64, P3 26.72→86.30, P4 138.4320 [138.1215,
140.6790] → 138.9557 [137.1952, 143.2682] with the IQRs overlapping, gather 36.34%→0.03%,
`_check_structural` 12.89%→3.65%, `repeat_interleave` 42.43%, M-3 pinned +178.2%/+181.7%, serve
idle-wait — all match. Two figures do NOT match the record exactly and are corrected BY
ANNOTATION below, never by repair (R285(c) shape); neither touches the verdict.

**Shape note — the first ruling in this lineage to ISSUE the verdict its four predecessors carried
as owed.** R283(c) authored the band; R284 read the before side against it; R285–R299 carried the
after-side verdict forward as a live liability, and R299(e) put on the record that it was *not
issued* because no after side existed. (b) closes that liability against the **unchanged** band,
which is the whole point of pre-registration: the band was written before the numbers and is read
against them, not adjusted to them. Note what the verdict does NOT do — it does not convert a
confirmed mechanism into a throughput win. Both predictions were CONFIRMED, neither falsifier
fired, the regression is ERASED, and P4 is still 138.96 against a 166 line. A mechanism working as
predicted and the system not getting faster are compatible findings, and (c) is what follows from
holding both at once. Labelled as grounds, NOT part of the numbered text.

> R300 — (a) The ARCH-ERA session's verification census is ratified in
> full: the numbering HALT was correct under R98/R160, the dormant-verdict
> posture was correct, and holding PLAN-C recon one turn rather than
> drafting from memory was the ledger's own discipline. The register head
> is R299, landed at f9988bd; numbering continues from this ruling.
> (b) THE RE-BENCH VERDICT, as pre-registered against R283(c), on the Q6
> sitting record (plan/Q6_BOX_EVIDENCE/, before side banked at f54be91,
> after side 18c934d, declared unit 91 commits — recorded, not
> re-litigated): P1 TRUE (occupancy 64 vs 20); P2 ~49 ms, diagnostic
> fail; P3 86.3%, diagnostic pass; P4 median 138.96 gph [137.20, 143.27]
> vs before 138.43 [138.12, 140.68] — below the 166 line: the verdict is
> STOP/INVESTIGATE by the band's own mapping. Readings of record: the
> 0.911× regression is ERASED (parity, IQRs overlap); both P-MASK and
> P-CHECKS predictions CONFIRMED (gather 36.34% → 0.03%; _check_structural
> 12.89% → 3.65%, inside its predicted 2–3 band; collate 17.72 → 7.18 ms);
> neither pre-registered falsifier fired. The prior investigation's
> targets are CLOSED.
> (c) The residual investigation is SCOPED by the after-side flamegraph
> and is NOT on the mint path: (i) the serve thread is 49.95% idle-wait —
> measure whether it is supply-limited before designing anything;
> (ii) repeat_interleave at gnn.py:60 (42.43% of remaining busy time) is
> the THIRD member of the size-from-contents sync class (nonzero,
> boolean-mask, tensor-counts) — candidate P-REPEAT: precomputed
> counts/offsets with the full P-MASK discipline (parity oracle, one
> change one bench); (iii) C1-pinned staging is LIVE — M-3 measured
> pinned +178%/+182% over pageable and the drop condition is unmet.
> MINT IS NOT PERF-BLOCKED: PCR 600/75 remains unviable on measured
> numbers; the 50-sims class is viable at parity throughput; the
> operator's sims fork is OPEN now, with the perf track continuing as its
> own thread.
> (d) Q6 adjudications: Q3-D1 PASS closes the PDEATHSIG program on GPU.
> F-816-14's split is ACCEPTED — the SIGKILL leg is closed (memory
> released in 5 s); the SIGTERM leg's question was mis-worded, the
> terminal-eval drain is designed behavior, and F-Q6-8 re-words it.
> Preflight 244 s vs 1800 recorded with the 447→244 attestation delta.
> Q4d floors CLEAN within ±4% on the attested host — the floor family is
> re-validated. F-Q6-1..8 route to the carry-over queue; F-Q6-1 (the
> flamegraph instrument's own 12.4 GiB orphan) joins the PDEATHSIG family
> as its instrument-side member.
> (e) SECURITY DISPOSITION: the captured live token is COMPROMISED by
> rule (it reached a remote); the operator rotates it as the first act;
> the forward redaction stands; NO history rewrite — the repo is private,
> recent shas are load-bearing, and rotation is the effective remedy.
> The incident and this disposition are the record. Evidence disposition:
> mantis-migration is the SANCTIONED home for box content — rule 7
> governs the public engine repo, and provider terms, IPs, and ssh
> invocations in this workspace are by design, not leaks. The pushed
> evidence STANDS; the uncommitted remainder is committed after a
> SECRET-class scan only (tokens, keys, credentials — never provider
> terms); a lightweight secret-scan convention (token/key patterns before
> any evidence commit) is adopted into PACKET_RECORD_CONVENTIONS.md.
> (f) ENVIRONMENT RULE, operator-supplied: mantis-migration remote
> operations use git-over-SSH ONLY, never the gh CLI — gh is
> authenticated to the engine-repo account, the migration remote belongs
> to the other account, and a gh 404 against it is expected and
> meaningless. Recorded so no future session re-derives the confusion.
> (g) The R299 session's two disclosed acts are ratified: deriving the
> register head in the v3 prompt instead of naming a number (a
> transcribed head is how v2 went stale), and authoring
> SESSION_STARTER_ARCH_ERA.md from the M4 snapshot with authorship
> declared — the starter STANDS.
> (h) CARRY-OVER QUEUE of record: Q3 freeze-governance · Q4 CI-runtime +
> F-816-30 · F-816-28 (two-invariant rule) · F-816-25 (before any
> BC-pretrain) · F-816-26 · F-Q6-1..8 routing · the perf track (idle-wait
> measurement, P-REPEAT, C1-pinned) as OPTIONAL and non-mint-blocking ·
> then the architecture program per the session starter. Operator
> residue: token rotation (immediate) · prereg values (sims fork now
> open) · R227/R228 + R267 texts · the mint word.
> ROUTE: the ARCH-ERA session lands this as its first act.

**ANNOTATION under R300's foot — two figures in (b) read against the sitting record, corrected
here and NOT in the text above (R285(c): correct by annotation, never by repair).** Filed by the
R300 landing session, 2026-08-21. Neither annotation touches the verdict, which turns on P4
against the 166 line and on whether either falsifier fired.

1. **`_check_structural` 3.65% is ABOVE its predicted band, not inside it.** (b) reads "3.65%,
   inside its predicted 2–3 band". `plan/Q6_BOX_EVIDENCE/GO_NO_GO_READINGS.md` records
   "**3.65 %** inclusive (self 0.95 %)" against "predicted band **2–3 %**", i.e. **3.65 > 3**.
   What is unaffected: P-CHECKS' own pre-registered falsifier is *"a flamegraph showing
   `_check_structural` still near 12 %"*, and 3.65 is not near 12 — so **CONFIRMED** stands, and
   the prediction's direction and magnitude are both right. The band overshoot is the honest
   residual and is recorded as such.
2. **P2's median is 50.25 ms, not ~49 ms.** (b) reads "P2 ~49 ms". `P1_P4_TABLE.md` records
   `queue_wait.mean_ms` **50.2493** [47.9263, 50.8743] after, against 23.2631 before. The tilde
   in the text carries the approximation; the record carries the number, and R283(c) assigns P2
   no gate either way — "diagnostic fail" is the correct characterization at either value.

Why annotate rather than let it pass: this register's own doctrine is that a landed number becomes
what future sessions cite, and the ONE-TEXT rule (R285) makes the register the place a citation
resolves to. A ruling's figure that is off by a band edge is exactly the class R8/G-DFIX-4 names —
a transcribed value read later as evidence. The text stands verbatim because it is the operator's;
the correction stands beside it because the record has to be right.


# R301 — architect adjudication response, operator-ratified by forwarding, 2026-08-21 (the R300 landing exit RATIFIED, both figure corrections ledgered as the architect's; Q-C0 RULED URGENT-DECAYING and the research copy ORDERED NOW under a secret scan; OPERATOR RULING OF RECORD — the migration remote is DELETED and never pushed to again, the vast instance DESTROYED AND RECREATED as the R300(e) rotation, so this workspace is LOCAL-COMMIT-ONLY effective immediately; two conventions adopted — process censuses REDACT token-bearing argv at capture time, and a rented box carries NO repository credentials; the seam-recon findings ACCEPTED and Q-C1..Q-C4 routed; the VAST-SETUP dispatch AUTHORIZED) [INLINE]

**Provenance: [INLINE], verbatim.** Text supplied in the R301 bridge packet under its own
heading `## R301 — CANONICAL TEXT`, operator-ratified by forwarding per the R261 shape,
2026-08-21, under the R283(b) carry rule. Header title follows the R280 forward-only convention.

**Carriage note — R285(d) NOT-APPLICABLE, thirteenth consecutive.** One copy, named as such by
the packet's own preamble ("R301's ONE canonical home; land it, continue from R302"). Armed per
R292(d) over the class R291(h) empties.

**Premise verification, per R289's precondition, run BEFORE this landing.**

1. **Head and census.** Pre-append census by script: **R23–R300, 271 sections, 271 distinct, 0
   duplicates**, missing-in-range `{24, 29, 32, 33, 227, 228, 267}` — identical to the figure
   v3.5 recorded, so the head is R300 and R301 is the next number. No numbering halt is owed.
2. **(a)'s two figure corrections exist and are correct.** `plan/Q6_BOX_EVIDENCE/GO_NO_GO_READINGS.md`
   reads "**3.65 %** inclusive (self 0.95 %)" against "predicted band **2–3 %**", and
   `P1_P4_TABLE.md` reads `queue_wait.mean_ms` **50.2493** [47.9263, 50.8743]. Both were already
   annotated under R300's foot by the R300 landing session; (a) ratifies an annotation that
   exists rather than ordering one, and **ledgers** it — see ANNOTATION 3 in the register-foot
   block, added by this landing as a POINTER, never a second copy (R285 ONE-TEXT).
3. **(a)'s scanner clause.** `python3 plan/secret_scan.py --self-test` → all eight controls
   **FIRES**, both negative controls silent, **rc 0** (read directly, not through a pipe —
   conventions §2). The retired gitignore probe is `.gitignore` §7, which records the interim
   `plan/Q6_BOX_EVIDENCE/` exclusion as removed under R300(e) and kept as a note rather than
   deleted silently.
4. **(b)'s subject, measured at this landing.** `hexo-mantis/tmp/research/` holds exactly three
   files — `architecture_defects_and_plan.md` 61,570 B · `copilot_neural_arch_literature_review.md`
   44,222 B · `hexo_architecture_review.md` 415,433 B — **521,225 bytes total**, gitignored by
   `/tmp/` at `hexo-mantis/.gitignore:24`, tracked in neither repo. `mantis-migration/plan/research/`
   did not exist. Q-C0's statement of its own subject is accurate.
5. **(c)'s remote claim, corroborated by measurement and stated with its limit.**
   `git ls-remote origin` **over SSH** (never `gh` — R300(f)) returns `ERROR: Repository not
   found.`, **rc 128**. What that establishes: the remote does not answer to this key today.
   What it does **not** establish: deletion specifically, since a revoked key produces the same
   string — R297(b)'s shape applied to a network probe. It **corroborates** the operator's
   statement; the operator's statement is what makes it a ruling of record.

**Shape note — this is the first ruling in the lineage whose operative half is an ENVIRONMENT
fact the workspace cannot itself change.** (c) does not order an act; it records that two things
the record has been assuming — a reachable remote, and a live box — no longer exist, and then
fixes what every standing "push the governance record" line now means. That is why (c) also
states the exposure it recreates *once*, with a recommendation and no order: single-copy risk is
the operator's to accept, and a ruling that quietly ordered a backup would be inventing an
authority the operator kept. Labelled as grounds, NOT part of the numbered text.

> R301 — (a) The R300 landing exit is ratified. The two R300(b) figure
> errors (3.65% sits ABOVE its predicted 2–3 band; P2's median is
> 50.25 ms) are the architect's, on the ledger, corrected by annotation
> and never by repair — neither touches the verdict's load-bearing legs
> (P4 against 166; neither falsifier fired). The secret scanner's
> positive-and-negative-control build and the retired gitignore probe are
> ratified.
> (b) Q-C0 is RULED URGENT-DECAYING and the copy is ORDERED NOW:
> hexo-mantis/tmp/research/ (the architecture program's ~520 KB source
> set plus siblings) copies to mantis-migration/plan/research/ through
> the secret scan and a rule-7-pattern read (provider terms are permitted
> in the destination; credentials are not); the copy commit states
> provenance (source path, file count, byte total). The tmp/ originals
> stay where they are — placement there remains one-way; the governance
> copy is the durable one.
> (c) OPERATOR RULING OF RECORD, verbatim intent: the remote
> [REDACTED:local:rule7_local_terms.txt:27:591f6616] repository is DELETED by the operator and is never
> pushed to again; the vast instance is DESTROYED AND RECREATED, which
> constitutes the R300(e) token rotation. Consequences: the governance
> workspace is LOCAL-COMMIT-ONLY effective immediately — every standing
> or convention line that says "governance push" is satisfied by the
> local commit until the operator provides a new remote; the
> single-copy exposure this recreates is stated once with the
> recommendation (a fresh private remote after rotation, or a periodic
> git bundle to a second disk) and the decision stays the operator's.
> (d) CONVENTION, the incident's class fix: process censuses REDACT
> token-bearing arguments at capture time (jupyter/syncthing/API-token
> argv patterns masked before the census is written) — the leak was the
> instrument faithfully recording a command line, so the instrument
> learns redaction; lands in PACKET_RECORD_CONVENTIONS with a
> positive-control test in secret_scan.py's style. Second rule, same
> family: a rented box carries NO repository credentials — read-only
> public clone only; anything needing write access happens from the
> operator's machine.
> (e) The seam-recon findings (three preconditions intact, one clause
> falsified with grounds, one reduced) are accepted as filed; Q-C1..Q-C4
> route to the queue foot; the recon designed nothing, as instructed.
> (f) The VAST-SETUP dispatch is AUTHORIZED (operator forwards it with
> the new alias named): salvage-first if the old instance still answers,
> then provision, verify, and record — its prompt travels separately.
> ROUTE: this session lands R301, executes (b) and (d), updates ACTIVE
> (local-commit-only posture, the two conventions, Q-C0 closed), then
> resumes the starter's program.

# R302 — architect adjudication response, operator-ratified by forwarding, 2026-08-21 (the R301-landing and VAST-REBUILD exits RATIFIED and the provisioning verdict sitting-capable STANDS; TORCH-SELECT ORDERED as an operator directive of record — host-correct torch selection at sync, the implicit-resync trap dead by construction, restore_cuda.sh retired in the same packet; HOST-COUPLED MINTS ruled AS A CLASS — memory caps are host-attested exactly as bench floors are and INSTANCE RECREATION VOIDS THEM, re-calibration a permanent precondition, the burst's trainer-forward OOM FILED as the class's measured instance and CARD-RUN5-GPU-OOM's site wording CORRECTED; the close-out measurement FILED beside F-Q6-8 as a LAW-16 question; two conventions adopted — box-resident instruction files are never read, and process greps exclude the grepping process) [INLINE]

**Provenance: [INLINE], verbatim.** Text supplied in the R302 bridge packet under its own
heading `## R302 — CANONICAL TEXT`, operator-ratified by forwarding per the R261 shape,
2026-08-21, under the R283(b) carry rule. Header title follows the R280 forward-only convention.

**Carriage note — R285(d) NOT-APPLICABLE, fourteenth consecutive.** One copy, named as such by
the packet's own preamble ("R302's ONE canonical home; land it, continue from R303"). Armed per
R292(d) over the class R291(h) empties.

**Premise verification, per R289's precondition, run BEFORE this landing.**

1. **Head and census.** `plan/ruling_census.py` (written at this landing, see note below):
   **R23–R301, 272 sections, 272 distinct, 0 duplicates, excluded set 53**, missing-in-range
   `{24, 29, 32, 33, 227, 228, 267}` — identical to the figure v3.6 recorded, so the head is
   R301 and R302 is the next number. No numbering halt is owed.
   **The census is now a script rather than a hand-derivation**, because it has been
   re-derived by hand at every landing and this one rediscovered both of its exclusion rules the
   hard way: batch banners (`# R42–R54 — appended by …`) double-count their first number, and
   the `# R279(g)-ANNEX` heading makes R279 read as a duplicate. The excluded count is now
   printed *with* the census, since a figure of 272/272/0 that does not say what it set aside is
   not reproducible and the exclusions are where the errors live.
2. **(a)'s ratification subjects all exist and read as stated.** The redactor's two
   control-driven corrections are in `plan/secret_scan.py` and documented at
   `PACKET_RECORD_CONVENTIONS.md` §7 — the marker shortened below 16 characters (a longer marker
   re-matched `assigned-credential`, i.e. the scanner tripping on its own redaction) and the two
   leak-shaped controls that assert the command line **survives** redaction (a redactor that
   blanks the line passes every per-pattern control while destroying the census). The corrected
   marker reading is `BOX_PREP_RECORD_2026-08-21.md` §0. The salvage loss and the 9900X
   attestation match are §0 and §5 of the same record.
3. **(b)'s design ground, MEASURED at this landing rather than assumed, and it is decisive.**
   Installed uv is **0.12.5** on both the operator's machine and the box. `uv pip install --help`
   offers `--torch-backend` with `auto`, `cpu` and **`cu128`** among its values and an
   `UV_TORCH_BACKEND` env binding — but **`uv sync` does NOT accept the flag**, so a CLI-only
   answer would not cover the bootstrap. **`torch-backend` IS a valid `[tool.uv]` pyproject
   key**, established with a control: an invented key (`definitely-not-a-real-key`) is rejected
   with `unknown field`, and uv enumerates the accepted set — `torch-backend` is in it. A
   pyproject-declared backend therefore governs `uv sync` **and** `uv run` alike, which is
   precisely what "dies by construction" requires. DESIGN still chooses, but it chooses with
   this measurement in hand.
4. **(c)'s membership, with ONE scope correction the clause's own wording does not carry.**
   `train.microbatch_caps` and `inference.fused_graph_caps` are live config keys (42 and 67 hits
   across `src/` + `configs/`, the latter with a single resolver read-path). **`_SIZING_BUDGET_GIB`
   is not a config key at all** — it is a literal, `9.431`, at
   `tests/train/test_graph_microbatch_bound.py:314`. It is host-attested in exactly the sense (c)
   means (STEP 1c measured it on the old box; `F816_10_EXIT.md` §235 already flags it as resting
   on a stale pre-Design-A figure), so it belongs to the class — but **voiding it edits a test,
   not a config**, which is a different act with a different review path. Checked and reported
   because it would otherwise surface as a surprise mid-re-calibration: the file is **NOT** in
   any of the 7 `wp/*/ORACLE_FREEZE*.sha256` registers, so the edit needs **no frozen-file
   grant**. The class has three members and two edit paths.
5. **(d)'s anchor exists.** `F-Q6-8` is `ADJUDICATION_QUEUE.md:4399`, inside the
   `F-Q6-1 .. F-Q6-8` pointer section routed by R300(d). Its subject — that `F-816-14`'s SIGTERM
   arm measures the terminal-eval park rather than a bare SIGTERM — is the same seam the new
   measurement lands on, so (d)'s "beside F-Q6-8" is a real adjacency and not a filing convenience.
6. **(f)'s trap is real and present.** `git remote -v` in `mantis-migration` still returns
   `origin [REDACTED:ssh-userhost:3de132cd]:[REDACTED:local:rule7_local_terms.txt:25:292d8813]/[REDACTED:local:rule7_local_terms.txt:27:591f6616].git` for fetch and push, against a repository
   R301(c) records as deleted. The VAST-REBUILD session declined to push and said so; the remote
   entry itself is what (f) orders removed.

**Shape note — (c) is the first ruling to give a MINTED VALUE the same lifetime rule as a bench
floor, and the consequence is larger than the clause looks.** A floor is void on a host change
because the number describes the host; (c) says a memory cap is void for the same reason and by
the same logic. What that buys is the reading of the provisioning burst: an OOM that would
otherwise be triaged as a code defect (and sent to a packet, and looked for in `gine.py`) is
instead a **stale mint on a new host** — the caps were fitted against a partition measured on a
container that no longer exists. The OOM is evidence the rule is right, not a bug report. This
is also why (c) cannot be discharged by raising a number: the joint fit is a partition, and
R281(d) already ruled both caps are minted in ONE act. Labelled as grounds, NOT part of the
numbered text.

> R302 — (a) The R301-landing and VAST-REBUILD exits are ratified: the
> redactor's two control-driven design corrections (a marker that
> re-matched its own scanner; leak-shaped controls asserting the line
> survives redaction), the corrected marker reading, the salvage loss
> accepted as pre-priced (medians banked), and the attestation-matching
> new host (9900X — floors admissible again). The provisioning verdict
> stands: the box is sitting-capable.
> (b) TORCH-SELECT is ORDERED (operator directive of record): an engine
> packet making torch selection host-correct at sync — GPU wheel where a
> compatible CUDA driver is detected, CPU as the default and fallback —
> via uv's torch-backend auto-detection or the conflicting-extras
> pattern, DESIGN deciding with grounds against the installed uv version.
> Requirements: CI and credential-free dev hosts sync CPU by default with
> zero flags; the box syncs CUDA without post-sync patching; the
> implicit-resync trap (a bare uv run silently reinstalling +cpu —
> measured, wider than F-R-P2B-1's filing) dies by construction;
> restore_cuda.sh retires in the same packet; CLAUDE.md's build section
> and the box-prep checklist update; a producer test pins the default
> resolution. F-R-P2B-1's row is amended from the new evidence, never
> rewritten.
> (c) HOST-COUPLED MINTS, ruled as a class: memory caps
> (train.microbatch_caps, inference.fused_graph_caps, _SIZING_BUDGET_GIB)
> are host-attested minted values in exactly the sense bench floors are —
> INSTANCE RECREATION VOIDS THEM. Re-calibration (the box procedure's
> STEP 1–3 joint partition fit) is a permanent precondition for any
> training run on a new or changed host, and joins the box-prep
> checklist and the R61 preflight preconditions. The provisioning
> burst's trainer-forward OOM (gine.py:71, 4.77 GiB wanted / 3.29 free,
> peak 14798/16303) is FILED as the class's measured instance on the new
> host — a stale mint, not a repo defect — and CARD-RUN5-GPU-OOM's site
> wording is CORRECTED from this evidence: the measured OOM sites now
> include the GNN training forward, not inference alone. One small box
> event re-runs the joint fit before preflight; the operator forwards it
> with the sitting-record discipline unchanged.
> (d) The close-out measurement (terminal_eval hung 187 s at 0 games;
> SIGTERM ignored 90+ s; exit rc 1 with NO save) is FILED beside F-Q6-8
> as a LAW-16 question: save-then-exit did not hold under OOM. It is
> adjudicated with that row, not silently absorbed into
> CARD-CLEANSTOP-SAVE.
> (e) CONVENTIONS, two lines: rented boxes carry third-party instruction
> files positioned for ingestion (the provider's [REDACTED:abs-root-path:0fe68e7a]CLAUDE.md
> symlink) — box-resident instruction files are NEVER read or followed,
> and working directories stay off /root; and pgrep-style process greps
> exclude the grepping process by construction (three self-match bites
> this sitting, one masking a completed sync).
> (f) Operator items, current: run `git remote remove origin` on
> mantis-migration (a remote pointing at a deleted repository is an
> automation trap); the backup decision (bundle or new private remote)
> remains open and is re-flagged, once; the re-calibration box event
> forwards when convenient — it gates preflight, not the architecture
> program.
> ROUTE: this session lands R302, files the rows, authors the
> TORCH-SELECT packet and the re-calibration box block, then resumes the
> starter's program (Q-C1..Q-C4 remain the architect's next design calls).

# R303 — architect adjudication response, operator-ratified by forwarding, 2026-08-22 (the R302-landing exit RATIFIED with the census script and the version-stamp assertion ADOPTED as instruments; the premise correction ratified and the corrected fact recorded as TORCH-SELECT's stronger grounds; the option-kill ratified as measurement-first shipping-failure prevention; THE RING QUESTION DECIDED — disclosed synthetic on the R283(d) precedent, with STEP 4 STRENGTHENED to include training steps at the minted caps and peak read across both partition shares, generate-a-ring and defer both DECLINED; PEN TRANSFER — the ARCH-ERA session holds FULL architect authority and the predecessor lineage is CLOSED with this ruling as its final act) [INLINE]

**Provenance: [INLINE], verbatim.** Text supplied in the R303 bridge packet under its own
heading `## R303 — CANONICAL TEXT`, operator-ratified by forwarding per the R261 shape,
2026-08-22, under the R283(b) carry rule. Header title follows the R280 forward-only convention.

**Carriage note — R285(d) NOT-APPLICABLE, fifteenth consecutive.** One copy, named as such by the
packet's own preamble ("R303's ONE canonical home; land it, continue from R304"). Armed per
R292(d) over the class R291(h) empties.

**Premise verification, per R289's precondition, run BEFORE this landing.**

1. **Head and census.** `plan/ruling_census.py`: **R23–R302, 273 sections, 273 distinct, 0
   duplicates, excluded set 53**, missing-in-range `{24, 29, 32, 33, 227, 228, 267}` — identical to
   the figure v3.7 recorded, so the head is R302 and R303 is the next number. No numbering halt.
2. **(a)'s two adoptions — one existed, one is built by this landing.** `plan/ruling_census.py`
   was landed at R302. The **version-stamp assertion is new here**: `--stamp` derives the stamp
   from §8's last curation entry and compares, `--self-test` proves it fires. **All four controls
   fire** — clean PASSES; a drifted stamp FAILS; a missing stamp FAILS; **§8 entries that do not
   parse FAIL rather than pass vacuously**, which is the phantom-gate refusal (LAW-07) and the
   half most likely to be omitted. The drift control is not hypothetical: it reproduces the exact
   v3.5-header-against-v3.7-entry state found at the R302 landing. Live: `STAMP OK`.
3. **(a)'s ledger item exists.** The "wider than the filing" correction is on the record as the
   append-only ANNOTATION under `F-R-P2B-1`'s foot in `plan/ADJUDICATION_QUEUE.md`, which states
   both halves: the `uv run` behaviour was on file from 2026-08-07 (`wp/WPBOX/STACK_PINS.md:22-24`,
   quoted inside the row itself), and the new fact is that the mitigation reached no part of the
   engine repository (`UV_NO_SYNC`/`--no-sync`, zero hits across `CLAUDE.md`, `Makefile`,
   `pyproject.toml`, `docs/`, `tools/`).
4. **(b)'s measurement stands as cited.** `[tool.uv] torch-backend` is schema-valid (control: an
   invented key is rejected with `unknown field` and uv enumerates the accepted set) and **inert
   for the project workflow** — `auto`, `cpu` and `cu128` each resolve torch to
   `registry = "https://pypi.org/simple"`, `UV_TORCH_BACKEND` likewise, measured `--no-cache`
   against a no-env control; `uv sync --dry-run` under `cu128` would install PyPI's CUDA-**13**
   bundle. uv 0.12.5 on both hosts.
5. **(c)'s precedent is REAL and quoted, and (c) is more than its application.** R283(d) reads:
   *"Calibration falsifier substitution ACCEPTED: synthetic-graph calibration (no ring on the box)
   with STEP 4 gaining the real-graph falsifier — recorded peak memory during the past-ply-120
   burst must respect the minted budget with the design margin; substitution recorded, never
   silent."* **Note what the bare precedent would have left open.** R283(d)'s falsifier is a
   *games* burst; `F816_10_BOX_PROCEDURE.md` STEP 4 still reads "One game past ply 120 on the GPU,
   zero OOM, counters live." The OOM this sitting must clear is in the **GNN training forward**
   (`F-R302-1`), which a games-only burst never reaches — the VAST-REBUILD drive completed a game
   at t+95.3 s and died at t+98.6 s on the trainer's first step. So (c)'s strengthening is not
   belt-and-braces: **without it the substitution would have been accepted against a falsifier
   incapable of firing on the failure it exists to catch.**
6. **(d)'s prompt exists, under a filename that does not say so.** `plan/ARCHITECT_SESSION_PROMPT.md`
   self-identifies in its header as *"v3, 2026-08-21 — SUPERSEDES v2"* per R299(f), with v2
   preserved unmodified beside it. **There is no file whose name contains `v3`**, so a reader
   grepping for one concludes it is missing; recorded here rather than renamed, because the
   supersession is deliberately a pointer and renaming would break the citations that already
   resolve.
7. **(d)'s residue list carries SEVEN items; ACTIVE's v3.7 stamp carried SIX.** Remote removal ·
   backup decision · re-calibration forward · prereg values · R227/R228 + R267 texts · the mint
   word were all stamped at v3.7. **`ssh-only next box` was NOT on the residue** — it lived as a
   recommendation in `BOX_PREP_RECORD_2026-08-21.md` §8 and in the re-calibration block's §3.
   (d) promotes it, and v3.8 executes that.

**Shape note — this ruling both ENDS a lineage and creates the authority that continues it, and
those two halves have different evidentiary weight.** (a)–(c) are ordinary adjudication and are
verified above in the ordinary way. (d) is constitutive: it cannot be checked against a prior fact
because it *makes* the fact. What can be checked, and is, is that the authority (d) confers is
bounded — the operator's residue is enumerated, R282(b)'s delegation boundary is unchanged, and
the prompt (d) names is on disk and readable. **An architect session's first duty under (d) is
therefore to keep verifying its own premises exactly as it did when it was executing someone
else's**, because the mechanism that catches an architect's error is the same one that caught the
dispatcher's, and (d) removes the second reader who used to run it. Labelled as grounds, NOT part
of the numbered text.

> R303 — (a) The R302-landing exit is ratified. ADOPTED: the census
> script (plan/ruling_census.py, encoding §1's own exclusion rules beside
> the figure) and the version-stamp assertion (the stamp derives from
> §8's last entry and is asserted by script — the third recurrence of the
> drift proved the R284(f) convention is not self-enforcing; instruments
> are). The premise correction is ratified with the architect's "wider
> than the filing" claim on the ledger; the corrected fact — the
> UV_NO_SYNC mitigation never reached the engine repo, so a box built
> from the repo's own bootstrap walks in blind — is the stronger grounds
> for TORCH-SELECT and is recorded as such.
> (b) The TORCH-SELECT option-kill is ratified as measurement-first
> shipping-failure prevention: [tool.uv] torch-backend is valid-but-inert
> for locked project workflows, established by cache-bypassed controls
> before the design could recommend it. The conflicting-extras/group
> pattern with a box-local UV_CONFIG_FILE proceeds to DESIGN with
> grounds, as packeted.
> (c) The RING QUESTION is DECIDED: the re-calibration runs on
> DISCLOSED SYNTHETIC graphs, the R283(d) precedent — with STEP 4
> STRENGTHENED as this decision's price: the validation burst includes
> training steps at the minted caps, not games alone, because the
> measured OOM site is the GNN training forward and a games-only burst
> cannot clear a trainer-share bound; peak memory is read across both
> phases against both partition shares. Generate-a-ring is DECLINED
> (sitting cost without commensurate falsifier value over
> synthetic-plus-strengthened-STEP-4); defer is DECLINED (it blocks
> preflight). The block amends accordingly and forwards on the
> operator's word.
> (d) PEN TRANSFER, effective on this ruling's landing: the ARCH-ERA
> session holds FULL architect authority under ARCHITECT_SESSION_PROMPT
> v3 and R282(b) — Q-C1..Q-C4 and every subsequent adjudication,
> ratification, grant, and verdict are its calls, made the way the
> prompt says: decided after pros, cons, and an adversarial pass, on the
> record, with the operator holding exactly the residue ACTIVE stamps
> (remote removal · backup decision · re-calibration forward ·
> prereg values · R227/R228 + R267 texts · the mint word · ssh-only next
> box). Exit reports route to the ARCH-ERA session directly. The
> predecessor session lineage is CLOSED with this ruling as its final
> act; its remaining value is historical and lives in the register it
> spent three hundred rulings building.
> ROUTE: the ARCH-ERA session lands R303, amends the re-calibration
> block per (c), adopts the stamp assertion into the census script, and
> proceeds as the architect it now is.

# R304 — architect ruling, AUTHORED by the ARCH-ERA session under R303(d), 2026-08-22 (Q-C1..Q-C4 RULED, closing the PLAN-C seam recon's open calls: the capability surface lands AFTER the mint as its own thread; caps.exact_symmetries is SPLIT because the conflict it inherits is a conflation of a rules fact with a model property, and the inverted reading is named as the real risk; §1.3 is NOT a site and is CLOSED on PLAN-C's own rule plus compile-time exhaustiveness; validate_against_state_dict and ShapeMismatchError are DELETED on a zero-caller census that includes tests; the ArchCaps DESIGN is AUTHORIZED to draft now and sequenced to land after the mint) [AUTHORED]

**Provenance: AUTHORED, not carried.** This is the first ruling written by the architect session
itself under the R303(d) pen transfer, rather than landed from an operator-forwarded packet. There
is no canonical block to diff against, so the R285(d) two-copy check and the ONE-TEXT byte-diff
are **NOT-APPLICABLE by construction** rather than by absence — recorded because sixteen
consecutive NOT-APPLICABLE notes have meant "the packet carried one copy", and this one means
something different. The text below IS the canonical text; it exists nowhere else (R289(v)).

**Premise verification, run BEFORE this ruling — and one recon citation was checked and HOLDS.**

1. **Head and census.** `plan/ruling_census.py`: **R23–R303, 274/274/0, excluded set 53** — head is
   R303, R304 is next.
2. **`R284(g)` exists and says what the recon says it says.** Checked because a first pass over
   R284's blockquote found only clause `(a)` at line-start and appeared to falsify the citation;
   the clause letters are **wrapped mid-line**, and `(g)` reads: *"Path: perf packet → after-side
   re-bench against the SAME banked before side and the SAME R283(c) band → verdict → prereg →
   preflight → mint."* **The apparent falsification was an artifact of the grep, not a fact about
   the register** — recorded because a negative text search is evidence about the pattern, not the
   artifact (R297(b)), and this is that rule catching its own author.
3. **Current position on R284(g)'s path.** Verdict **ISSUED** (R300(b)). Remaining: **prereg**
   (operator-owed values) → **preflight** (blocked by R302(c)'s re-calibration) → **mint**.
   `ArchCaps` appears at no step of that path.
4. **Q-C4's census re-run WIDER than the recon's.** The recon measured `src/`. Re-measured across
   `src/`, `tests/`, `tools/` and `crates/`: `validate_against_state_dict` has **zero callers
   anywhere, including zero tests**. Its only raised type, `ShapeMismatchError`, is raised at
   `resolvers.py:641` and `:655` — **both inside that one function** — and is otherwise unused.
5. **Q-C2's falsity note read at source.** `GNN_DEFECT_VERIFICATION_2026-08-07.md` §§85-110:
   `configs/run5.yaml:128` mints `train.augment: false`, so D6 augmentation is **off entirely** and
   `sym` is always 0 (`sample.rs:172-176`); separately, the GNN carries **no equivariance
   machinery** (measured: `lib.rs:546-549`, `gine.py` full-file read), untied axis weights make
   rotated positions yield different edge embeddings (`gine.py:94,53`), and **the magnitude of the
   output non-invariance is UNMEASURED**.

> R304 — (a) Q-C1 RULED: the capability surface lands AFTER the mint,
> as its own thread. R284(g) fixes the mint path — perf packet,
> re-bench, verdict, prereg, preflight, mint — and ArchCaps is at no
> step of it; the verdict is issued and only prereg and preflight
> remain. R300(c)'s posture governs a track that is real and not on the
> mint path, and this is one. "Validated at mint" is a requirement on
> mints that happen AFTER the surface exists; it does not retroactively
> invalidate a mint taken before it, because the compatibility it
> checks is already enforced today by the identity keys (LAW-11), the
> schema, and the registry handshake at CI gate 8. That is also the
> binding condition on landing: ArchCaps must REPLACE or DERIVE FROM
> those checks, never sit beside them — a second authority over the
> same relation is the duplicated-default class R1 exists to kill, and
> adding one mid-flight is precisely how F-816-24 and the MonitorConfig
> chain happened.
> (b) Q-C2 RULED: caps.exact_symmetries does not land as one field,
> because the conflict it inherits is a CONFLATION, not a
> contradiction. Two different facts wear one name. The board
> automorphism group is a RULES fact and is exact — twelve elements,
> pure integer axial lattice math (sym.rs:19,60), measured and holding.
> Model invariance under that group is a MEASURED PROPERTY OF A
> TRAINED NET, and for the GNN at head it is FALSE with its magnitude
> unmeasured. A capability surface may declare the first. It may never
> be read as the second. Land the field named for the rules fact it can
> support (board_automorphisms, or any name that cannot be misread as a
> model claim), and augmentation policy may read it — augmentation
> applies board automorphisms to DATA, whose validity does not depend
> on the net being equivariant. THE REAL RISK IS THE INVERTED READING:
> a field called "exact symmetries" invites the conclusion that the
> model already handles them and augmentation is unnecessary, which for
> this net is exactly backwards — non-equivariance is the reason to
> augment, not the reason not to. The orbit probe
> (GNN_DEFECT_VERIFICATION's own cheapest falsifier) is ORDERED as the
> precondition for any MODEL-side symmetry claim, and is explicitly NOT
> a blocker on augmentation policy. run5 minting train.augment: false
> is a separate matter and stays the operator's prereg row.
> (c) Q-C3 RULED: §1.3 is NOT a site, and it CLOSES. SYS-2's "enforced
> globally" clause is falsified at head; what survives — the invariant
> re-asserted per arm rather than derived from a declared capability —
> is what PLAN-C's own closing rule endorses: one place, branches
> growing linearly, each stating its grounds. The stronger ground is
> mechanical rather than textual: the match at validate.rs:148 is
> EXHAUSTIVE with no wildcard, so a third representation FAILS TO
> COMPILE until someone writes its arm and its justification, whereas
> derivation from a capability would accept a third representation on a
> default. Replacing a compile-time refusal with a runtime derivation
> is a downgrade, and this site is therefore left exactly as it is.
> (d) Q-C4 RULED: validate_against_state_dict and ShapeMismatchError
> are DELETED, with their exports and docstring lines. The census is
> zero callers across src/, tests/, tools/ and crates/ — no test either,
> so nothing breaks and nothing is left unproven that was proven
> before. LAW-08 is NOT the grounds and is not stretched to reach this:
> it governs config keys and registered encodings, and this is neither.
> The grounds are that an exported validator whose probe sets are CNN
> key names silently no-ops on a graph state-dict, and the production
> representation is graph — so its green means "I could not see the
> thing I check", which is R299(d)'s mechanism-not-proxy shape sitting
> in the open waiting for its first caller to be misled. A false green
> is worse than an absent check. If a shape cross-check is wanted
> later it is written fail-closed against the representation it
> validates, which is a different function, not this one revived.
> Routing: its own commit on the TORCH-SELECT branch — a four-line
> deletion does not justify its own scratch branch, and a separate
> commit keeps one-change-one-commit intact.
> (e) The ArchCaps DESIGN is AUTHORIZED to be drafted NOW, and
> sequenced to land after the mint per (a). Drafting costs no mint-path
> time and the recon has already cut the work down: three of PLAN-C's
> four structural preconditions are satisfied at head, so what remains
> is the capability surface and the sites that consume it — of which
> (c) removes one, (b) reshapes one, and 1.5 is reduced to an audit
> tool. The design carries (a)'s no-second-authority condition and
> (b)'s naming split as pre-registered constraints, not as review
> comments. It does not touch code until the operator forwards it.
> ROUTE: CURRENT-SESSION. The architect drafts the ArchCaps design and
> files (d) onto the TORCH-SELECT packet; the orbit probe under (b) is
> queued behind the re-calibration, not ahead of it.

# R305 — architect record of operator decisions, ARCH-ERA session, 2026-08-22 (the superseding-bridge appendix recorded: backup DECLINED-RETIRED, origin removal VERIFIED, routing restated, torch confirmed as the operator's; residue 7 → 5) [INLINE]

**Provenance: [INLINE], authored.** Text originates in the ARCH-ERA architect session under
R303(d), the R304 precedent; its one canonical home is
`PACKET_R305_OPERATOR_RECORD.md` §2, and this append was byte-diffed against that block.
The operator decisions recorded in (a) carry the operator's own authority; R305 records
them, it does not create them. The P5 measurement (`git remote -v` empty) is quoted in the
landing report and referenced here, not restated.

> R305 — (a) The operator decisions carried by the superseding R303
> bridge's appendix are RECORDED, each an operator decision of record:
> (1) BACKUP — DECLINED. The governance workspace stays single-copy on
> one disk by the operator's choice, and the flag RETIRES PERMANENTLY —
> no session, packet, or curation re-raises it; the single-copy exposure
> statement R301(c) made once remains made exactly once. (2) ORIGIN —
> `git remote remove origin` on mantis-migration was executed by the
> operator and is VERIFIED at this landing: `git remote -v` returns no
> remotes. The v3.7 automation-trap listing is discharged by the
> measurement. (3) ROUTING — all agent exit reports and questions route
> to the ARCH-ERA architect session directly; this restates R303(d)'s
> consequence and adds no authority. (4) TORCH — the GPU-detect /
> CPU-fallback directive is confirmed as the operator's own; it is
> already the R302(b) order and no new instruction issues. (b) The
> operator residue REDUCES from seven to five: forwarding the
> re-calibration block · the prereg values · the R227/R228 and R267
> texts · the mint word · ssh-only next box. Struck: origin (verified
> removed) and backup (declined and retired).
> ROUTE: FRESH DISPATCHER — this packet's own landing; verify origin,
> append, census, curate ACTIVE to v3.11, commit locally, report to the
> ARCH-ERA session.

# R306 — architect adjudication, ARCH-ERA session, 2026-08-22 (the R305 landing RATIFIED with the architect's stamp-model error on the ledger; instrument-over-prediction and de-mark-before-normalize ADOPTED; MISSION PLANC-SEAM-M1 established under deferred ratification; dispatch order of record set, Q4 first) [INLINE]

**Provenance: [INLINE], authored.** Text originates in the ARCH-ERA architect session under
R303(d); one canonical home: `PACKET_PLANC_SEAM_M1.md` §0. Byte-diffed on append
(de-marked, whitespace-normalized, line-by-line; report the mechanism).

> R306 — (a) The R305 landing is RATIFIED with all four disclosures
> accepted. The packet's §3 stamp-assertion prediction was WRONG and the
> error is the architect's, on the ledger: check_stamp compares the
> ACTIVE header against §8's last entry — two facts internal to one file
> — and no register append can move it. ADOPTED as standing practice: an
> instrument's measured behavior outranks any packet's prediction of it;
> the executor proceeds and reports, halting only on the packet's named
> halt triggers. ADOPTED for all substring spot-checks: strip markdown
> emphasis and blockquote markers BEFORE whitespace normalization, and a
> five-of-five failure indicts the checker before the text. The census
> runs from plan/ (path fact, recorded). The reconstructed-not-observed
> rewording is ratified as figure-provenance discipline applied to a
> process claim.
> (b) MISSION PLANC-SEAM-M1 is ESTABLISHED per its packet: Leaf 0 the
> ArchCaps design review, then the conformance-suite pipeline in the
> R262 shape — DESIGN, REVIEW-design, ORACLE, IMPL, REVIEW-impl,
> RED-TEAM — sequential, isolated worktrees, fix loops after each
> review. DEFERRED RATIFICATION is in force for this mission in the
> R298(b) shape and lapses with it: checkpoint records land on disk and
> do not return; disclosed-and-reasoned judgement calls proceed under
> standing approval and are batch-ratified at mission end from records
> re-derivable in full. Frozen edits without existing grants, armed
> values, and prereg decisions remain ABSOLUTE exclusions — filed,
> never taken. Early return only if every remaining item is blocked.
> (c) Binding constraints on the mission, ranking above anything its
> design derives: nothing from this mission touches the mint path or
> lands a capability surface before the mint (R304, Q-C1); anything
> depending on ArchCaps sequences with ArchCaps; the R257 fences hold —
> Shrimp-Bot patterns transfer, its rules, radius and search-free
> posture do not; radius derives from registry.toml at point of use
> (R26); box measurements are STAGED as box-block deliverables for
> operator forwarding, never run by the mission — box grants are
> operator-only; a µs/leaf figure is host-attested or it is mechanism
> evidence, never a verdict.
> (d) Dispatch order of record: PACKET_CI_RUNTIME (Q4, carrying
> F-816-30) forwards FIRST — the flaky oracles and the shrinking
> headroom are the substrate every other packet runs on; then
> PACKET_FREEZE_GOVERNANCE (Q3, gh re-checked at dispatch). This
> mission runs in parallel exactly where collision tables re-measured
> at dispatch permit (R295(c)), never by inheritance. F-816-25 keeps
> its before-any-BC-pretrain precedence; F-816-26 and F-816-28 queue
> behind Q3/Q4.
> ROUTE: FRESH DISPATCHER — mission MAIN per PACKET_PLANC_SEAM_M1.md;
> land this at Task 0, census expect R23–R306, curate ACTIVE to v3.12,
> local commits only in the governance workspace.

# R307 — architect adjudication, ARCH-ERA session, 2026-08-22 (checkpoint D1/Leaf-0 RATIFIED; Q-C5 RULED — caps.exact_symmetries DELETED, the per-record gate is the sole symmetry authority; ANNOTATION 6 ordered; two bounds on the F2–F9 repairs; T2 retired for re-derivation; PHASE 2 WP-AXIS2 in scope with depth-to-done) [INLINE]

**Provenance: [INLINE], authored.** Text originates in the ARCH-ERA architect session under
R303(d); one canonical home: `PACKET_PLANC_SEAM_M1_AMEND1.md` §1. Byte-diff on append
(de-marked, whitespace-normalized, line-by-line, per R306(a); report the mechanism). The
engine coordinates in (b) are as verified by this landing's §0, which supersedes the
checkpoint's reading if they drifted.

> R307 — (a) Checkpoint D1 / Leaf 0 is RATIFIED: the SOUND-WITH-REPAIRS
> verdict stands; the dispatcher's independent re-derivation of five
> findings before crediting a leaf that indicts architect text is
> ACCEPTED as the standard for exactly that case; the Q-C5 routing was
> correct — a mission FILES against architect text, it never repairs it
> — and the tier-granular HALT (T2 alone, the leaf proceeding) is the
> mission discipline applied at the right size. F2–F9 are the DESIGN
> leaf's to fix, under the two bounds in (d).
> (b) Q-C5 is RULED: `caps.exact_symmetries` is DELETED from the
> ArchCaps design. Grounds: fact (ii) is not a static per-arch
> property — the engine decides losslessness per record and per site,
> on both arms (hexg/sample.rs:172 gating on stone presence, 12-or-1;
> sym.rs:135-140 gating on compact, 12-or-4; game.rs:675-676
> permanently 4 with its unknowable-at-draw-time grounds;
> augment.py:214-229 the per-row mirror; sym.rs:128-131 stating the
> gate outright as the per-record evaluation of losslessness HERE). A
> frozen per-arch set over that mechanism admits exactly two readings
> and the review broke both: the union authorises augmentation the
> engine refuses; the intersection collapses to identity for GnnArch —
> the inverted reading through the opposite door. The per-record gate
> IS the symmetry authority, singular. A capability field beside it —
> as a set, a summary, or a gate pointer — is the second authority
> R304(a) exists to abolish; routing to the gate is already carried by
> the representation type's exhaustive match (R304, Q-C3 grounds), so
> even a pointer duplicates. Consumers needing symmetry facts derive
> them through the gate at point of use, never from the arch. This
> extends R304(a)'s derive-not-store from two fields to the CLASS:
> where the engine holds a per-record authority, no per-arch field
> summarises it.
> (c) ANNOTATION 6 is ORDERED under R304's foot, append-only, in the
> established annotation shape: fact (ii)'s typing "a static, per-arch
> property" is CORRECTED to "per-record and per-site, decided by the
> engine's own gate"; the annotation cites the four sites in (b) as
> its evidence and states plainly that ANNOTATION 5's own citation for
> (ii) — hexg/sample.rs:168-176 — is the per-record gate and
> contradicts the typing it was cited to support. ANNOTATION 5's
> surviving content SURVIVES: no surface carries a model-equivariance
> claim, and the inverted reading remains the named live hazard — (b)
> closes its last door by removing the field that could carry it.
> (d) Two BINDING BOUNDS on the F2–F9 repairs, ranking above the
> design's own text: (i) per F4 — no exit criterion replaces or
> re-keys the seven-name ban. The ban's subject includes non-head
> knobs (entropy_reg_weight is a regularizer term, trainer/core.py:
> 79-80) and the nonzero-entropy graph-ban is an operator LOCK (R37,
> ACTIVE §2); any caps-derived rule is ADDITIVE beside the ban, and a
> future replacement requires its own ruling carrying a
> mutation-proved producer test that fires on every banned name, not
> one. (ii) per F9 — LAW-08 is STRUCK from the design's grounds
> wherever cited, per R304(d)'s explicit refusal of that stretch; the
> design stands on R304(a) and on this ruling.
> (e) T2 as written is RETIRED with the field. The DESIGN leaf
> re-derives the tier from (b) and decides with grounds, acceptance
> pre-registered before IMPL either way: EITHER T2 retargets to the
> structural absence — no conformant package carries a symmetry claim
> of any type, and augmentation is reachable only through the engine
> gate, both checked by AST/type/reachability (R296(f)), with a
> planted-break for each (a package smuggling a symmetry field; an
> augmentation call path bypassing the gate) — OR its content folds
> into T1's no-second-authority checks and the suite ships as
> T1/T3/T4/T6. A tier that would test the deleted field's presence in
> any form is the wrong answer by construction.
> (f) MISSION SCOPE EXTENDED — PHASE 2, WP-AXIS2, depth-to-done. After
> the suite completes D2–D4 the mission proceeds WITHOUT RETURNING:
> WP-AXIS2 DESIGN derived from the gnn_axis_v2 memo v1 at point of use
> (mean+max readout · tied axis weights · drop norm_q/r · dummy-node
> normalization — the memo's text governs over this parenthetical),
> through REVIEW-design and ORACLE in the R262 shape; IMPL proceeds in
> the same mission IF AND ONLY IF the suite has landed green on remote
> CI and the collision table re-measured at that moment is clear —
> the new arch lands BEHIND the seam and is exercised by the suite as
> its first non-trivial conformance subject, which is the seam earning
> its existence. Depth-to-done: the mission runs each item to finished
> or genuinely-blocked, never to a courtesy checkpoint. HARD
> EXCLUSIONS unchanged and restated because Phase 2 walks near them:
> NO training or eval run on any host — box events are
> operator-forwarded and R302(c) voids the caps until the
> re-calibration sitting mints; NO armed value, NO prereg decision, NO
> frozen edit without an existing grant; NOTHING touches the mint
> path; the R257 fences and R26 hold. The item-size tripwire applies
> per leaf; deferred ratification (R306(b)) covers Phase 2 in full.
> ROUTE: mission MAIN, current loop — land R307 with ANNOTATION 6 in
> the same act, census expect R23–R307, curate ACTIVE to v3.13, local
> commits only in the governance workspace, then continue the DESIGN
> leaf with (b)/(d)/(e) as inputs and Phase 2 on the mission board.

# R308 — architect adjudication, ARCH-ERA session, 2026-08-22 (RE-CALIBRATION sitting RATIFIED AS SUCCESSFUL HALT — caps remain void, two mechanism F-rows ordered; MINT-ON-BRANCH adopted; worktree repair ordered; Q-C9 RULED derive-or-delete; B(i) shape DECIDED; stamp constraint set; frozen-golden grant procedure; RECAL-PREP ordered) [INLINE]

**Provenance: [INLINE], authored.** Text originates in the ARCH-ERA architect session under
R303(d); one canonical home: `PACKET_R308_POSTHALT.md` §1. Byte-diff on append per R306(a)
(de-marked, whitespace-normalized, line-by-line; extraction by tooling, never retyped —
the R307 landing's awk mechanism is the standard).

> R308 — (a) The RE-CALIBRATION sitting on the replacement instance is
> RATIFIED AS A SUCCESSFUL HALT. The mint was legal when made — all
> three R282(b) conjuncts held — and is VOID now that STEP 4 falsified
> the partition on three clauses; R262 was honoured and the falsifier
> did not move. Consequences stand exactly as the exit states them:
> ACTIVE §5's voided-caps row does NOT expire; F-R302-1 is EXPLAINED,
> not closed — its measured upstream mechanism is allocator reservation
> fragmentation under DEFAULT posture (14.8047 GiB reserved against
> 10.0997 allocated in pure inference, 8.76×; joint peak 15816 MiB,
> 19.6 MiB from the card) — and the architect does not close it on this
> evidence; R61's precondition remains unsatisfied; no training run
> preflights or mints on this host. Two mechanism findings are ORDERED
> filed as F-rows from the sitting record, IDs per the record
> conventions: (i) the eval-child budget term is ROUND-DEPENDENT, not
> constant — 0.881 GiB at 41 samples, 1.186 at 709, 3.529 with rounds
> allowed to complete — so the four-term budget model is falsified in
> STRUCTURE on this host, not merely in values; STEP 1d as written
> cannot measure a growing term, and the strengthened STEP 4 catching
> it is R303(c)'s price purchasing exactly what it was set to purchase.
> (ii) inference-phase reservation fragmentation as the upstream
> mechanism of F-R302-1. The in-sitting refusal to re-size shares on
> arithmetic is RATIFIED — inference at 2.09 GiB of a 13.92 GiB bound
> is not the term that failed.
> (b) MINT-ON-BRANCH is ADOPTED as procedure, the sitting's own
> proposal: STEP 3 lands on a branch and dev fast-forwards ONLY after
> STEP 4 passes; F816_10_BOX_PROCEDURE.md amends accordingly at next
> touch (RECAL-PREP carries it). The concurrent-worktree contamination
> is ORDERED repaired by mission MAIN: the planc-conformance worktree
> resets onto current dev — which no longer carries the halted pair —
> before its next commit, ancestry-verified after the reset. The
> sitting's refusal to touch another session's live branch is RATIFIED.
> (c) Q-C9 is RULED, derive-or-delete applied inside the gate: the
> silent-encoding gate's ENCODINGS DERIVES from
> crates/mantis-encoding/registry.toml at point of use — never a
> transcribed tuple — and the gate's own floor reasoning extends to the
> set it scans for: the gate asserts the derived set is non-empty and
> matches the registry's encoding count, so scanned-no-encodings can
> never pass, by the same argument MIN_SCANNED_FILES already makes
> about scanned-no-files one line away. The planted-break is an ORACLE
> obligation: an encoding added to the registry and deliberately
> omitted from the gate's reach must turn the gate red. If the gate
> file is frozen in any manifest, THIS ruling names
> tools/ci_gates/silent_encoding_gate.py and is the R43 grant for
> exactly that change, same-act re-pin a term of it (R290(e)).
> (d) The B(i) shape is DECIDED for Phase-2 IMPL: forward-time
> symmetrization — the shape that leaves the tensor and the state-dict
> key set untouched and therefore preserves BC transfer.
> nn.utils.parametrize is REJECTED for this use: it renames keys, and
> key stability is LAW-12's subject, not a style preference. ORACLE may
> overturn only on a disqualifying fact, pre-registered before IMPL.
> (e) The build.py stamp defect (net.arch assigned post-hoc, so a
> mis-dispatched v1 net would stamp itself v2 — LAW-12's subject
> exactly) carries a BINDING constraint into Phase-2 DESIGN/ORACLE: no
> post-hoc arch assignment anywhere on the build path; a net's arch
> stamp issues at construction inside the arch's own build path and
> requires a structural witness at stamp time — the probe-hash or
> key-set the design already carries; a net whose structure does not
> witness the stamp RAISES with a named error, it never stamps. The
> planted-break (mis-dispatched net must raise) is an ORACLE
> obligation.
> (f) The frozen-golden question is ANSWERED as procedure, and the
> leaf's refusal is RATIFIED as this mission's third correct
> exclusion-refusal: no grant issues on a description (R286/R289).
> When Phase-2 IMPL reaches slot-2 it presents the exact diff and the
> verbatim frozen rows for tests/fixtures/manifest.toml; the grant
> issues then, naming the path, same-act re-pin. The same procedure
> pre-answers the re-sit: the two frozen files transcribing the
> outgoing cap pair present their diffs at the re-sit's mint and the
> grant issues against those diffs in the same act.
> (g) RECAL-PREP is ORDERED as its own small dispatch, engine-side, no
> box contact: (i) the allocator posture becomes a MINTED REGIME KNOB —
> one resolver (hard rule 3, the amp-dtype precedent), explicit in
> every config with no code-side default (hard rule 1), asserted at
> boot beside the R281(d)(iii) partition assertion, armed-abort
> covered — killing the governance objection under which DEFAULT was
> kept while expandable_segments measured 11.36 vs 14.98 GiB card peak
> on the live drive; the posture VALUE mints at the re-sit,
> measurement-derived under R282(b), never in this dispatch. (ii)
> eval-child instrumentation so the re-sit measures the round-growth
> to plateau or bound, and the STEP 1d amendment that makes a growing
> term measurable where it grows. (iii) BOX_BLOCK_RECALIBRATION.md
> amends with Δ5 (posture candidate expandable_segments:True, minted
> in-sitting under the new knob), Δ6 (the eval-child term in
> round-dependent form, with the budget model's structural amendment),
> and the mint-on-branch procedure per (b). The re-sit is
> operator-forwarded thereafter, alias named, R299 shape unchanged.
> run5's n_workers has no prereg row and makes the cap unbiteable at
> its current value — that row is FILED to the operator's prereg
> matrix, judgment-valued, never minted by a sitting.
> ROUTE: mission MAIN lands this ruling — census expect R23–R308,
> ACTIVE v3.14, local commits only — and executes (b)'s worktree
> repair in the same act; (c)/(d)/(e)/(f) are Phase-2 ORACLE and IMPL
> inputs; (g) is the FRESH-DISPATCHER packet the operator forwards.

**ANNOTATION 7 — R308(c)'s registry PATH is CORRECTED: the file is
`crates/mantis-encoding/src/registry.toml`, not `crates/mantis-encoding/registry.toml`. Ordered by
R309(b), appended 2026-08-22 by the MISSION PLANC-SEAM-M1 MAIN dispatcher. The error is the
ARCHITECT'S and is recorded on the architect's ledger as the SECOND occurrence of the identical
string.** R308(c) orders the silent-encoding gate's `ENCODINGS` to derive "from
crates/mantis-encoding/registry.toml at point of use". **That path does not exist at engine HEAD**
(`dev` = `47b78f9`), verified at this landing by mechanism rather than by reading: `ls
crates/mantis-encoding/registry.toml` fails; `crates/mantis-encoding/src/registry.toml` is a
10 180-byte regular file. **Every executor of R308(c) derives from the latter.** Nothing else in
(c) moves: the derive-or-delete rule, the non-empty-and-matches-the-registry-count assertion, the
planted-break obligation and the R43 grant over `tools/ci_gates/silent_encoding_gate.py` all stand
exactly as written.

**WHY IT IS ON THE LEDGER AND NOT ON THE MISSION'S.** Mission PLANC-SEAM-M1 had already filed this
exact correction once — checkpoint 04, finding R-3, against the same wrong string in the
PLANC-SEAM-M1 packet §3 — and it then reappeared inside a landed ruling. A correction that has to
be made twice is not a transcription slip; it is a source the architect keeps reading. **The source
was measured at this landing, and R309(b)'s own account of it is imprecise in a way worth recording
rather than repeating:** CLAUDE.md's **hard-rule-4 line carries no path at all** (hard rule 4 is
"R4 gates. No gate/monitor input without a producer test; every registered encoding has a live
consumer."). The only `registry.toml` mention in CLAUDE.md that a reader turns into a path is the
**Map** bullet at `CLAUDE.md:16` — `- crates/mantis-encoding — registry.toml + spec + validators +
dense encode kernels.` — which carries **no directory segment**, so it does not state the wrong path
so much as invite it: a reader resolving that bullet against the crate root lands on exactly the
string that has now been written twice. Corrected in the same act per R309(b), one engine commit,
because the durable-rules file is where the reader reads.

**AND THE FIX WAS ITSELF CAUGHT WRONG, TWICE, BY A GATE — recorded because it is the whole
argument for fixing it at the reader.** The first correction wrote `src/registry.toml`; CI gate
10 (`tools/ci_gates/check_tracked_refs.py`) RED, because a bare `src/registry.toml` resolves from
the repo root and does not exist either. The second wrote the whole path but SPELLED THE WRONG
ONE beside it as a caution; gate 10 RED again, because it cannot tell a cautionary quote from a
reference. **Three wrong paths in a row on one line, each caught by mechanism** — which is the
measured case for why this correction belongs in a file a gate reads, and why the wrong string
is named HERE, in the register, and deliberately not in CLAUDE.md.

**Coordinate check, both directions, so this annotation does not itself become a source.** R308(e)
names `build.py`'s stamp defect and carries **no line number**, so the Phase-2 ORACLE's second
coordinate correction (`net.arch = arch` is `build.py:54`, not `:53`) has **no subject in the
register** and no annotation is owed for it. Verified at HEAD: `src/mantis/model/build.py:54`.

# R309 — architect adjudication, ARCH-ERA session, 2026-08-22 (Phase-2 ORACLE + RECAL-PREP exit RATIFIED; ANNOTATION 7 ordered — R308(c) path corrected, architect's ledger; Q-C10 RULED behavioral-witness-or-no-landing; detector dilemma DECIDED — GnnNetV2 new class with mandatory detectors; the R308(f) grant ISSUED against the presented diff; n_workers prereg row RECORDED from the operator; WORKER-SWEEP ordered as Phase W of the re-sit) [INLINE]

**Provenance: [INLINE], authored.** Text originates in the ARCH-ERA architect session under
R303(d); one canonical home: `PACKET_R309_GRANT_AND_WITNESS.md` §1. Byte-diff on append per R306(a)
(de-marked, whitespace-normalized, line-by-line; extraction by tooling, never retyped —
the R307 landing's awk mechanism is the standard).

> R309 — (a) Phase-2 ORACLE and the RECAL-PREP exit are RATIFIED in
> full: the six-candidate census with its 33 constructible breaks; the
> unapplied-diff refusal as this mission's FOURTH correct
> exclusion-refusal; the red team's catch of the dispatcher's own
> case-normalization against c10's exact-token toBool; the four
> run-only regressions fixed at mechanism with no guard relaxed; the
> growth RE-ATTRIBUTION — 3.529 GiB is a within-move demand peak, not
> retained cache — filed with file and line, which leaves Δ6's
> round-dependent budget form standing on the demand mechanism; the
> eval-side boot assertion REMOVED rather than edit a frozen oracle,
> residual disclosed in the resolver's docstring; and the dispatcher's
> own-error correction of its "mostly correct" cardinality framing,
> measured false at ORACLE — six of the eleven reds are bare counts a
> rename walks past and none is a refusal — corrected at the source
> the next reader reads. The R308(d) strengthening is RECORDED: at
> HEAD edge_proj applies once and each conv's lin operates
> post-projection, so forward-time symmetrization is a complete tying
> at one site with the key set untouched.
> (b) ANNOTATION 7 is ORDERED under R308's foot, append-only: R308(c)
> reads crates/mantis-encoding/registry.toml; the path at HEAD is
> crates/mantis-encoding/src/registry.toml, and every executor of (c)
> derives from the latter. The error is the architect's, on the
> ledger, SECOND occurrence of the identical string — checkpoint 04
> corrected it in the original packet and the architect re-introduced
> it from CLAUDE.md's own hard-rule-4 line. If CLAUDE.md carries the
> wrong path at HEAD it is corrected in the same act, one engine
> commit: the durable-rules file misdocumenting the single source of
> truth is the misinformation-where-the-reader-reads class, and the
> fix belongs where the reader is.
> (c) Q-C10 is RULED: structural unfalsifiability CONVERTS the
> verification obligation; it never waives it. A candidate no
> structural check can witness lands ONLY with a pre-registered
> BEHAVIORAL witness — an input or property pair whose observed output
> must change in a stated direction if and only if the candidate is
> present. For Candidate A: a discriminating input on which max and
> mean readouts provably differ, so the composed readout's output
> separates v2 from v1 by construction. For C(i): the symmetry-pair
> check at the tied site — sigma-paired inputs must agree at that site
> after the change where they measurably disagreed before it, the
> orbit-spread instrument narrowed to one site. Both witnesses are
> ORACLE rows before IMPL, red first. And the principle carries its
> edge: where neither a structural nor a behavioral witness is
> constructible, the candidate DOES NOT LAND — a change nothing can
> detect cannot be verified to exist, and this tree does not carry
> unverifiable changes; the plan's own verified-not-trusted clause
> generalizes to exactly this.
> (d) The detector dilemma is DECIDED: GnnNetV2 lands as a NEW arch;
> in-place modification of GnnNet is REJECTED — its loud goldens are
> accidental detectors purchased by silently re-typing every existing
> checkpoint's subject. The accidental detectors are replaced by
> DELIBERATE ones, ordered: (i) the R308(e) witness-stamp is a
> CONSTRUCTION PRECONDITION for v2 — no GnnNetV2 constructs anywhere
> before mis-dispatch raises with a named error; (ii) v2 mints its own
> golden at slot-2 under the R308(f) grant-on-diff procedure, and the
> flip-one-byte demonstration is RATIFIED as the standard evidentiary
> form for a frozen-edit grant argument; (iii)
> tests/test_fixtures_manifest.py gains the reverse check — required
> list and present set equal in BOTH directions — so an unlisted
> golden reds as loudly as a missing one; (iv) the six bare-count
> gates convert on contact: a count that gates names its subjects by
> set equality against the registry, never by cardinality — the
> seventh application of derive-or-delete this mission.
> (e) THE GRANT: the R308(f) procedure is satisfied and the grant
> ISSUES against the exact diff presented unapplied in
> RECAL_PREP_EXIT.md §6 for the frozen row
> test_booting_run5_on_a_non_CUDA_box_fails_LOUD_in_init_trainer,
> path named by the diff itself, same-act re-pin a term of the grant
> (R290(e)). One verification obligation attaches, a check and not a
> diff change: RED-TEAM verifies the row's verdict on a CPU host under
> BOTH posture tokens, and the posture-coupling the exit disclosed is
> recorded beside the row. Execution: apply the diff verbatim, re-pin
> in the same act, local suite green, PUSH dev — the two RECAL-PREP
> commits ride the same push — and confirm remote CI green. The
> knowingly-red-head refusal to push before the grant is RATIFIED.
> (f) The n_workers prereg row is RECORDED as the operator's, verbatim
> in substance: n_workers = 1 is REJECTED as far too low; the operator
> bracket is greater-than-1 through 14, with extension past 14
> PERMITTED while throughput gains persist and memory discipline
> holds, bounded above by the box's measured thread count. The POINT
> VALUE is measurement-derived inside that bracket under a
> pre-registered selection rule — the knee rule: the smallest rung
> within 95 percent of the best PASSING rung's throughput. A rung
> PASSES only with a PLATEAU memory verdict under the stopping-rule
> discipline (REFUSED is never a verdict; GROWING fails the rung); an
> OOM at a rung is data that fails the rung and stops the ladder's
> extension, never a sitting failure. No post-hoc movement of any of
> it.
> (g) WORKER-SWEEP is ORDERED as PHASE W of the re-sit, entering the
> box block AHEAD of STEP 1: the sweep runs SELF-PLAY ONLY — no
> trainer step executes before the mint, so the voided-caps row is not
> crossed — walks the pre-registered ladder 2, 4, 8, 12, 14 with
> extension per (f), picks the point value under the knee rule, writes
> it into the run configs on the sitting's branch, and STEP 1's four
> terms are then measured AT THAT GEOMETRY. That ordering is the cure
> for the unbiteable-cap finding: caps fit at the config that will
> run, or they are stale at birth. Prep is engine-side, fresh
> dispatcher, no box contact: the sweep driver with per-rung
> throughput and memory verdicts in the RECAL-PREP instrument's
> discipline, and the block's Phase W amendment. The re-sit forwards
> ONCE, alias named, carrying Phase W then STEP 1 through 4.
> ROUTE: mission MAIN lands this ruling and ANNOTATION 7, executes
> (e), census expect R23–R309, ACTIVE v3.15, governance commits local
> only; (g)'s prep is the FRESH-DISPATCHER packet the operator
> forwards; (c) and (d) bind Phase-2 IMPL.

# R310 — architect adjudication, ARCH-ERA session, 2026-08-27 (F-R309-1 DISPOSED — repair (1) TWO LIVE ARMS granted against a MEASURED diff and executed; R309(e)'s push clause COMPLETED; red-by-design REFUSED as a posture; WORKER-SWEEP Phase-W prep RATIFIED; the Phase-W ordering conflict RESOLVED against the architect's own text; grant-on-a-measured-diff ADOPTED; three prereg matters ROUTED to the operator) [INLINE]

**Provenance: [INLINE], authored.** Text originates in the ARCH-ERA architect session under
R303(d); one canonical home: `PACKET_R310_FR309_1_DISPOSITION.md` §1. Byte-diff on append per
R306(a) (de-marked, whitespace-normalized, line-by-line; extraction by tooling, never retyped —
the R307 landing's awk mechanism is the standard).

> R310 — (a) F-R309-1 is DISPOSED on its FIRST ranked repair, and the
> frozen edit is GRANTED: path tests/tools/test_preflight_mint_process.py,
> row test_booting_run5_on_a_non_CUDA_box_fails_LOUD_in_init_trainer,
> against the exact diff presented at plan/FR309_1_REPAIR_DIFF.md §2,
> same-act re-pin a term of the grant (R290(e)). The row branches on the
> CONFIG'S OWN STATE, asked of declared_allocator_posture — the module
> that owns the vocabulary — and BOTH arms assert. On the R119
> placeholder: the shipped preflight refuses with
> UncalibratedAllocatorPostureError and does NOT reach init_trainer,
> which is R308(g)(i)'s ORDERING measured END TO END through the real
> process and asserted nowhere else in this repo. On a minted token: the
> original device subject, unchanged, in the posture read from the config
> through the resolver. The repair is SELF-EXPIRING — the sitting's mint
> retires the placeholder arm with no act of remembering. Repair (2), the
> skipif extension, is REJECTED: it would switch off R126's only witness,
> the instrument that cannot false-clear, for the length of a box
> sitting, and buying CI quiet with a dead instrument is the trade this
> register has refused every time it has been offered. Leaving the row
> red is REJECTED for the finding's own reason — a correct refusal that
> never expires is a standing outage.
> (b) THE RULE THAT GENERALIZES IT, binding forward: a row whose SUBJECT
> is unavailable at HEAD carries a LIVE ARM at HEAD or a DECLARED skip.
> RED-BY-DESIGN IS NOT A POSTURE THIS TREE CARRIES. A red that is
> expected teaches every reader to expect red, and the next red — the
> real one — then arrives into an audience that has already learned to
> look away; that is the same mechanism as a stale line count and a
> phantom gate input, one layer up. And the rule carries its edge, so it
> cannot be satisfied by an arm that asserts nothing: where the
> unavailable subject leaves a witness gap, the arm covering the
> available state must assert SOMETHING NO OTHER ROW ASSERTS, or the
> honest answer is the declared skip. The repair granted in (a) meets
> that test on its own terms, and plan/FR309_1_REPAIR_DIFF.md §6 and §7
> state both halves — what it adds, and what stays dormant until the
> mint.
> (c) R309(e)'s execution clause is COMPLETED. With (a) applied the local
> suite is green, and the PUSH is AUTHORIZED and taken. The grant is over
> dev and not over a commit list: R309(e) named the two RECAL-PREP
> commits as RIDERS on a push of the branch, never as its contents, so
> the head that pushes carries them, ANNOTATION 7's engine fix, R309(g)'s
> Phase-W prep, and (a). The WORKER-SWEEP exit's withholding was CORRECT
> and it named F-R309-1 as its reason; the reason dies here and the hold
> dies with it. Remote CI green on the pushed head is the confirming
> measurement and is recorded with its run id, not asserted.
> (d) The WORKER-SWEEP Phase-W prep is RATIFIED in full — driver, plan
> file committed before the sitting, oracle inventory, the block's Phase-W
> amendment, and the CPU floor whose honest answer was rc 2 with every
> rung REFUSED and PICK = none. THREE ADVERSARIAL ROUNDS each found
> defects under the last, and the ordering is the result worth carrying:
> the design review found a LIVE red on a shipped oracle; the red team,
> against a GREEN 109-oracle build, found four defects each capable of
> minting a wrong n_workers that no reader of the artifact could catch;
> the impl review, against the red team's own fix pass, found a
> non-terminating loop that pass had INTRODUCED and that the packet's
> most-argued closure had never been shown to reject anything. F-WS-12 is
> ADOPTED as a standing instrument note: a suite that passes is evidence
> about the suite, not about the artifact. What produced findings, every
> time, was mutating the implementation and re-running, or asking how
> this produces a plausible WRONG NUMBER — never reading the code and
> agreeing with it.
> (e) The Phase-W ORDERING CONFLICT is RESOLVED, and the ARCHITECT'S TEXT
> is the half that moves. R309(g) put Phase W "ahead of STEP 1" against a
> tree in which every config carries the R119 posture placeholder,
> runtime-refused on any cuda process, and the posture is decided INSIDE
> STEP 1, at 1b — so the ordering as written is UNRUNNABLE, and the
> packet was right to surface it rather than resolve it quietly. ADOPTED:
> STEP 0 (the posture A/B, RELOCATED out of STEP 1b, unchanged in
> content) → PHASE W → STEP 1's four terms at the picked geometry.
> R309(g)'s intent is preserved exactly — caps fit at the geometry that
> will run, or they are stale at birth — and only the step numbering
> moves. Recorded on the ARCHITECT'S LEDGER as a text-vs-tree defect: a
> clause whose order was authored against a tree it was not checked
> against.
> (f) GRANT-ON-A-MEASURED-DIFF. R308(f)'s present-the-diff procedure
> gains ONE conjunct, and F-R309-1 paid for it: a presented diff is a
> MEASURED diff. The R309(e) grant was correct on its face, issued on its
> face, and bought a head that was red for a week and would have blocked
> every unrelated push — because nobody had RUN it. Presentation
> therefore carries the drive: the diff applied in a working tree, the
> row driven, a planted break per new assertion showing each one bites,
> and the tree restored or the diff captured. Before the grant, not
> after. The exclusion is UNCHANGED and is on LANDING a frozen edit
> without a grant; measuring one in an uncommitted tree is how a grant
> becomes decidable, and a session that cannot show the measurement has
> presented a description (R286/R289).
> (g) The THREE PRE-REGISTRATION MATTERS are ROUTED TO THE OPERATOR,
> UNMOVED, and the executor's refusal to move them is RATIFIED as this
> mission's FIFTH correct exclusion-refusal. F-WS-2 (does an OOM stop the
> whole ladder or only its EXTENSION), F-WS-7 (what happens when Phase
> W's pick does not survive STEP 2 — on the 2026-08-22 measurements the
> MODAL outcome, not a corner) and F-WS-4 (the Phase W mint silently
> RE-SCOPES an armed draw-rate abort whose N_pool_min 50 was
> pre-registered at n_workers 1 under ADJ-08) are judgment-valued prereg
> rows, and R282(b) reserves every one of them. They are answered BEFORE
> the re-sit forwards, or they are answered at the box by whoever is
> standing there — which is the definition of post-hoc. The architect
> puts a RECOMMENDATION on each so the operator's decision is informed,
> and decides none of them: F-WS-2, keep the register's narrower reading;
> F-WS-7, adopt the proposed fallback (down the PASSING rungs, largest
> first, to the largest rung at which the partition closes; HALT with the
> numbers if none does); F-WS-4, the two stale in-tree sentences take a
> same-act correction at the mint. A recommendation is not a
> pre-registration: the operator's ADOPTION of one, before the sitting, is
> what pre-registers it.
> ROUTE: mission MAIN lands this ruling, executes (a) and (c), census
> expect R23–R310, ACTIVE v3.16, governance commits local only; (g) is
> SURFACED to the operator as residue, and the re-sit does not forward
> until it is answered; (b) and (f) bind FORWARD.

**ANNOTATION 8 — R310(c)'s REMOTE-CI CONFIRMATION IS WAIVED BY THE OPERATOR. Recorded 2026-08-27,
same day, by the ARCH-ERA architect session at the operator's direction; appended under R310's foot
on the ANNOTATION 7 precedent, append-only.** R310(c) ends *"Remote CI green on the pushed head is
the confirming measurement and is recorded with its run id, not asserted."* **The operator
overruled that, in their own words: *"forget the remote ci / I overrule watching it as operator /
focus on local specific tests when needed"*, with the grounds stated as a question — whether a push
whose only new edit is a TEST needs a CI round at all.** The grounds are measured and they hold:
this landing's own diff is **one file, `tests/tools/test_preflight_mint_process.py`**, and the whole
push set touches **ZERO paths under `crates/`** (`git diff --name-only origin/dev..dev | grep -c
'^crates/'` = 0), so no Rust changed and the cargo half is unaffected. **What replaces it is not
nothing, and that is why this is an annotation and not a deletion:** the confirming measurement is
now the LOCAL tiers, both run to completion on this host at the pushed tree — **integration 39
passed / 10 skipped / 0 failed, default 3730 passed / 4 skipped / 0 failed, twelve local gates rc
0** — and they are recorded in the ACTIVE §8 v3.16 entry exactly as the run id would have been.
**WHAT IS GIVEN UP, stated because a waiver that hides its cost is worth less than the round it
saved:** the local host is not the CI host, and the fresh-clone `uv sync` leg (gate 1) is the one
check no local run reproduces. **Scope: this push only.** R310(c)'s sentence stands as written for
every future push; it is an operator call per push, and R282(b) reserves exactly this to them.

# R311 — architect ruling, 2026-08-22 (R310 ratified; field-ruling conditions; remote CI suspended by operator decision — local green is the gate; docs-follow-reality; symbol-not-line packet law; plain comms adopted; CLAUDE.md refreshed; planc-conformance push granted; the three worker preregs ADOPTED by the operator) [INLINE]

**Provenance: [INLINE], authored.** Text originates in the ARCH-ERA architect session under
R303(d); one canonical home: `PACKET_R311_VELOCITY_v2.md` §1. Byte-diff on append per R306(a)
(de-marked, whitespace-normalized, line-by-line; extraction by tooling, never retyped — the
R307 landing's awk mechanism is the standard). **This landing's packet was SUPERSEDED IN
FLIGHT:** a v1 of the same packet was extracted and appended earlier in the same session, and
the architect replaced it with v2 before the landing committed. The v1 append was REVERTED from
the working tree (never committed, so the register's history carries no superseded text) and
v1's file was removed so exactly ONE canonical text of R311 exists on disk (R285's ONE-TEXT
rule). The v1→v2 delta is three clauses and is stated in the §8 v3.17 entry.

> R311 — (a) R310 is RATIFIED in full, including its repair, its
> five-drive measurement, ANNOTATION 8, and the correction of the
> stale ACTIVE claim. Its shape becomes the rule: an execution
> session MAY issue a field ruling when all four hold — the item
> blocks progress; the evidence is complete and on disk; the ruling
> is disclosed as a field ruling in the act itself; and it is filed
> for ratification by the architect session. Operator locks (armed
> values, prereg rows, box grants, the mint word, owed texts) are
> never field-ruled. "Measured before granted" (R310(f)) binds every
> future grant: a grant executes against measured behavior, not
> against its face.
> (b) CI AND TEST CADENCE. REMOTE CI IS SUSPENDED, an operator
> decision of record, until the operator re-enables it; no push or
> merge waits on it and no clause elsewhere requiring remote green
> binds while the suspension holds — D3 and R307(f) are AMENDED to
> full local green. The working standard everywhere is the full
> local gate set. While iterating inside a leg: targeted tests,
> smallest relevant first; the full local sweep runs at leg exit and
> before any push, not per edit. Commits touching only documents or
> governance need no gates at all. Gates are never weakened by this
> ruling — it changes when they run, never what they check. The
> known cost is on the record: the fresh-clone leg is the one thing
> no local run reproduces (ANNOTATION 8's grounds), and it is
> accepted by the operator for the suspension's duration. Re-enable
> is the operator's word and will be recorded when given.
> (c) DOCS FOLLOW REALITY. A non-canonical working document that
> disagrees with verified repo or register state is repaired in
> place by whoever finds the mismatch, recorded in one checkpoint
> line, with no loop and no surfacing. Register text still corrects
> only by annotation. A documentation loop that survives one repair
> pass is a finding about the document, not a task to keep doing.
> (d) SYMBOL, NOT LINE. Packets and rulings reference code by symbol
> and mechanism. Line numbers may appear only as hints marked as
> hints. Every figure, path, or claim carried into a packet is
> marked verify-at-HEAD and the executor re-derives it before use.
> (e) PLAIN COMMS. plan/COMMS_STYLE.md (created by this landing) is
> adopted for the architect and every dispatcher: plain language to
> the operator; one short screen per report with full detail on
> disk; no ceremony; no restating of state that lives in files.
> (f) CLAUDE.md is REFRESHED per this packet's §3 spec in one
> commit, kept lean: the adapted external code-quality rules, the
> Rust equivalents, the (b) cadence, and a one-line pointer to (c).
> (g) The planc-conformance push is GRANTED under (b): push when its
> own full local sweep is green.
> (h) The three worker-sitting preregs are ADOPTED BY THE OPERATOR,
> 2026-08-22, and are pre-registered as of that adoption: F-WS-2 —
> an OOM during the ladder stops the ladder's extension only, never
> the whole sweep; F-WS-7 — when Phase W's pick does not survive the
> fit, fall back down the passing rungs largest first, and HALT with
> the numbers if none closes; F-WS-4 — the two stale draw-rate
> sentences are corrected in the same act as the Phase W mint, and
> N_pool_min does not move. The re-sit is FORWARDABLE once this
> ruling lands.
> ROUTE: land this, create COMMS_STYLE.md per §2, apply §3 to
> CLAUDE.md, census expect R23–R311, ACTIVE to its next version,
> then continue Phase-2 under the new cadence.

# R312 — architect ruling, 2026-08-27 (R311 landing RATIFIED with two architect errors ledgered; the mirror becomes a REDACTED DERIVATIVE and verbatim public mirroring is refused; --check held out of the gate set behind Q4; the conformance branch repair ORDERED; gate 17's worktree weakness CLOSED loud) [INLINE]

**Provenance: [INLINE], authored.** Text originates in the ARCH-ERA architect session under
R303(d) and arrived as an operator-forwarded amendment to `PACKET_R311_VELOCITY_v2`; one
canonical home: `PACKET_R312_POST_R311_ADJUDICATION.md` §1. Byte-diff on append per R306(a)
(de-marked, whitespace-normalized, line-by-line; extraction by tooling, never retyped). **The
paste carried the placeholder `R{next}`**, substituted to `R312` after verifying the head is
R311 — census `R23-R311, 282/282/0` at entry, no `R312` string under `plan/`, and the re-sit's
exit is a running DRAFT that files no field ruling. **The substitution is the only edit.**

> R312 — (a) The R311 landing is RATIFIED including the v1-revert-before-commit, the
> four found-not-told corrections, and the mirror leak near-miss handled by reset-before-
> push with the defense moved into sync_governance.py. On the architect's ledger: the
> rustfmt gate claim and the stale re-sit prompt were the architect's errors; supersessions
> of in-flight packets ship as labeled amendments from now on, never replacements.
> (b) The mirror is a REDACTED DERIVATIVE: sync_governance.py redacts via gate 17's own
> scan with stable placeholders before writing; --check verifies against the redacted
> transform; the header names it derivative and points at canonical. Verbatim mirroring
> into a public repo is refused. The operator may override by making the repo private.
> (c) --check stays OUT of the gate set until Q4's gate protocol lands; it runs at every
> governance landing per the standing line.
> (d) The conformance branch repair is ORDERED: reset onto dev per R308(b), apply the
> granted one-line gate-17 escape fix, re-derive the floor on the reset branch, full local
> sweep, push under R311(g).
> (e) Gate 17's worktree weakness is CLOSED loud: absent local-terms file prints an
> OPERATOR-TERM ARM SKIPPED banner, and sync_governance.py refuses to write any mirror
> without the full arm present.
> ROUTE: land this, census, ACTIVE next version, implement (b)/(e), execute (d), one plain
> screen back.

# R313 — architect ruling, 2026-08-27 (R312 landing RATIFIED; a scan passing is not the output being clean; the independent grep PROMOTED into sync_governance.py; PIPE-EXIT LAW; SHARED-TREE AMEND LAW; both executing-session errors ledgered as disclosed with their repairs ratified) [INLINE]

**Provenance: [INLINE], authored.** Text originates in the ARCH-ERA architect session under
R303(d) and arrived as an operator-forwarded amendment; one canonical home:
`PACKET_R313_INDEPENDENT_CHECK.md` §1. Byte-diff on append per R306(a) (de-marked,
whitespace-normalized, line-by-line; extraction by tooling, never retyped). **The paste carried
the placeholder `R{next}`**, substituted to `R313` after verifying the head is R312 — census
`R23-R312, 283/283/0` at entry, no `R313` string under `plan/`, and the re-sit session's further
commits file no field ruling. **The substitution is the only edit.**

> R313 — (a) The R312 landing is RATIFIED, including the redaction-insufficiency
> repair: the gate-17-scan redaction passed while still carrying three account handles
> and a private repo name; an independent grep caught it, not the gate; fixed at the
> designed extension point (operator supplement 3→6 terms). Recorded as principle: a
> scan passing is not the output being clean — the independent check authored this catch.
> (b) sync_governance.py gains a second, structurally different assertion: after
> writing, it greps every supplement term against every written mirror and refuses on
> any hit — the independent grep promoted into the tool. With the existing
> refuse-without-arm rule this makes mirror generation main-tree-only by construction;
> the supplement's single-copy nature is recorded as designed, not accidental.
> (c) PIPE-EXIT LAW: no gate, scanner, or verifier ever sits upstream of a pipe
> without pipefail or an explicit PIPESTATUS capture — an exit code never dies in a
> pipe. The masked secret_scan rc 1 behind `| tail -1` is this law's second producing
> instance; the box procedure's PIPESTATUS correction was its first. One-time sweep of
> gate wrappers and plan scripts for the pattern; fix on contact.
> (d) SHARED-TREE AMEND LAW: --amend only after re-reading HEAD in the same breath and
> verifying it is your own commit; in a shared working tree prefer a new commit. The
> byte-identical restoration of the foreign commit is RATIFIED.
> (e) Both errors stand on the executing session's ledger as disclosed; both repairs
> chose the correct side — rename the local, never weaken the scanner; restore the
> foreign commit, never absorb it. The conformance push proceeds under R311(g) when
> the sweep is green; nothing else is blocked.
> ROUTE: land this, census, ACTIVE next version, implement (b), run (c)'s one-time
> sweep, one plain screen back.

# R314 — architect ruling, 2026-08-28 (R313 execution RATIFIED; the shared-tree law GENERALIZES to every commit and the detached-worktree recovery is named; sync_governance --check RE-TARGETS to the dev ref and the filed finding closes) [INLINE]

**Provenance: [INLINE], authored.** Text originates in the ARCH-ERA architect session under
R303(d) and arrived as an operator-forwarded amendment; one canonical home:
`PACKET_R314_DEV_REF_MIRROR.md` §1. Byte-diff on append per R306(a) (de-marked,
whitespace-normalized, line-by-line; extraction by tooling, never retyped; the extraction and
diff ran under `set -o pipefail`, which is R313(c) applied to its own successor). **The paste
carried the placeholder `R{next}`**, substituted to `R314` after verifying the head is R313 —
census `R23-R313, 284/284/0` at entry, no `R314` string under `plan/`, and the re-sit session's
completed exit files a self-correction but no field ruling. **The substitution is the only edit.**

> R314 — (a) The R313 execution is RATIFIED: both conformance reds fixed at cause,
> R308(b) discharged, the audit-written assertion sharing no code with the redaction,
> and the pipe-exit sweep's gate-14 catch — a type baseline that could announce itself
> unmeasured is precisely the class the law was cut for.
> (b) The shared-tree law GENERALIZES, third instance in one family: verify branch AND
> tip before ANY commit in a shared worktree, not only before an amend. Concurrent
> sessions commit from their own worktrees (R276(b) extended beyond mission leaves to
> every concurrent session); the main checkout belongs to the session running the box
> event. The detached-worktree cherry-pick with byte-identical restoration of the
> foreign tip is RATIFIED as the named recovery pattern.
> (c) sync_governance.py --check RE-TARGETS to the dev ref: mirrors are compared
> against git show dev:docs/governance/<name>, never the working tree, because the
> mirror contract is about what dev carries. Writing stays main-tree-only per R313(b).
> One new control: with the worktree parked on a non-dev branch, --check must report
> dev's truth. The filed finding closes with this.
> ROUTE: land this, census, ACTIVE next version, implement (c) with its control, one
> plain screen back.

# R315 — architect ruling, 2026-08-28 (the re-sit RATIFIED as a SUCCESSFUL HALT — blocked by the instrument, not the card; the allocator posture SETTLED as expandable_segments on a pre-declared mechanism criterion; RESIT-PREP-2 ORDERED engine-side; the third sitting gated behind it and a ratified margin floor) [INLINE]

**Provenance: [INLINE], authored.** Text originates in the ARCH-ERA architect session under
R303(d) and arrived as an operator-forwarded amendment; one canonical home:
`PACKET_R315_RESIT_RATIFICATION.md` §1. Byte-diff on append per R306(a) (de-marked,
whitespace-normalized, line-by-line; extraction by tooling, never retyped; extraction and diff
under `set -o pipefail`, R313(c)). **The paste carried the placeholder `R{next}`**, substituted
to `R315` after verifying the head is R314 — census `R23-R314, 285/285/0` at entry, stamp v3.20,
no `R315` string under `plan/`. **The substitution is the only edit.**

> R315 — (a) The re-sit is RATIFIED AS A SUCCESSFUL HALT — blocked by the
> instrument, not the card — and the R314 landing is RATIFIED with it. Every
> August FAIL clause now passes under expandable_segments; the REFUSED eval
> verdict is the tool working. Nothing vests: caps stay void, F-R302-1 stays open
> (mechanism now fully explained as a posture artefact; it closes at a standing
> mint), R61 unsatisfied.
> (b) The allocator posture is SETTLED: expandable_segments, on the sitting's
> pre-declared mechanism criterion — the fragmentation divisor the partition
> arithmetic divides by is wrong 3.77× under DEFAULT and within 0.6% under
> expandable. It mints with the caps at the next sitting.
> (c) RESIT-PREP-2 is ORDERED, engine-side, no box, reading RESIT_FINDINGS.md at
> point of use: (i) F-RESIT-10 — worker_sweep calls seed_everything; a
> determinism control (same rung, same seed, twice, sub-1% apart) becomes a
> planted-break; (ii) F-RESIT-14 — derive the eval-round timeout mechanism from
> the findings, fix at cause, and close the gate hole: a timeout exit must fire
> LAW-15's gate, never bypass it; (iii) Δ8 is REWRITTEN against the measured
> nine-act mint — RESIT_MINT_MEASURED_DIFF.patch is the base and the block names
> all ten files and both frozen rows correctly; (iv) F-RESIT-7 — conjunct 2
> gains a pre-registered MARGIN FLOOR proposed from the two sittings' measured
> peaks and ratified by the architect before any forwarding; a partition that
> passes by a rounding error licenses nothing, twice measured.
> (d) The third sitting forwards after (c) lands and the margin floor is
> ratified: seeded sweep, eval term to plateau or bound, then measure-fit-mint-
> burst per the corrected block. Intent of record: the instrument is now
> complete; this is the sitting that mints or produces a card-level fact.
> ROUTE: land this, census, ACTIVE next version, then RESIT-PREP-2 as a fresh
> dispatch the operator forwards; one plain screen back.

# R316 — architect ruling, ARCH-ERA session, 2026-08-28 (RESIT-PREP-2 RATIFIED in full; the conjunct-2 MARGIN FLOOR ratified at M = 0.35 GiB with its operative form pinned as a subtrahend BEFORE the fit; the frozen-file GRANT for the reason rename scoped to the one measured row; REFBOT-SCAN-1 accepted and R257 ANNOTATED not repaired; COMMENT STYLE adopted as an operator direction; F-RESIT-6 defaulted; the third sitting cleared to forward on the operator's arm) [INLINE]

**Provenance: [INLINE], authored.** Text originates in the ARCH-ERA architect session under R303(d); one canonical home: `PACKET_R316_RESIT_PREP2_RATIFICATION.md` §1. Byte-diff on append per R306(a) (de-marked, whitespace-normalized, line-by-line; extraction by tooling, never retyped). **The paste carried the placeholder `R{next}`**, substituted to `R316` after verifying the head is R315 — census R23–R315, 286 sections / 286 distinct / 0 duplicates, excluded set 53, missing-in-range {24, 29, 32, 33, 227, 228, 267}; `--stamp` reported `STAMP OK: stamp v3.21 == §8 last entry v3.21`. The substitution is disclosed as the only edit to the canonical text.

**Precondition note, recorded because the check as worded can no longer be satisfied by any landing.** The packet asks for *no `R{next}` string under `plan/`*. Ten files carry one, including this register and `RULINGS_ACTIVE.md` — every occurrence is the PROSE each landing writes to disclose its own substitution (*"the paste carried the placeholder `R{next}`, substituted to R31x"*). There is no LIVE unsubstituted placeholder, which is what the check means; the literal form has been self-falsifying since R312 wrote the first disclosure. Verified by reading all ten, not by counting them.

> R316 — (a) RESIT-PREP-2 is RATIFIED in full. (i) The seeded sweep with its
> determinism control (0.5821% against a 1% band, AGREE; three planted breaks
> bite) and the seed_everything move to mantis.util.determinism, on its measured
> ground: any import under mantis.train pulls eight training modules into
> sys.modules, so a carve-out would have been a hole. (ii) The mechanism
> correction of record — the eval-round failure is a progress-budget escalation
> mislabelled JOIN_TIMEOUT, not a join timeout — and the gate hole CLOSED: a
> broken round now fires LAW-15's gate with reason eval_round_broken, planted
> break 8 rows red on revert. (iii) The Δ8 rewrite, membership derived from the
> measured patch, never counted. The session's three self-caught errors stand on
> its own ledger as the gates working.
> (b) The MARGIN FLOOR is RATIFIED: M = 0.35 GiB, derived from the single
> measured under-prediction — 2026-08-22's joint peak exceeded its declared
> budget by 0.3352 GiB — rounded up. OPERATIVE FORM, pinned: M is a subtrahend
> in the budget derivation BEFORE the fit, budget = (usable − trainer −
> eval_child − M) ÷ frag, and conjunct 2 then asserts headroom ≥ M. A floor
> checked without the derivation subtracting it refuses every sitting
> mechanically: the derivation consumes headroom to ≈0.107 GiB by construction,
> twice measured. LIMITS CARRIED: the evidence is one measurement; conjunct-2
> headroom cannot separate a passing burst from a failing one (0.0012 GiB apart
> across sittings whose bursts differ by 3.075 GiB), so M is a forcing term,
> never a predictor; and M is denominated in the CURRENT partition arithmetic,
> the 0.26 GiB double-count included — correcting that double-count VOIDS M and
> re-derives it. THE ARMED VALUE IS THE OPERATOR'S: his forwarding of a
> third-sitting block carrying M = 0.35 is the value lock.
> (c) FROZEN-FILE GRANT for the reason rename. Scope: exactly the measured one
> row — test_eval_broken_reason_enum.py's set-equality pin — plus the source
> symbol it pins; the applied-driven-reverted probe satisfies the flip-one-byte
> standard. Executes as RESIT-PREP-2b: one commit, own worktree (R314(b)), full
> local gates at leg exit. Ground: an instrument label naming a mechanism that
> is not the cause is the stale-text defect class inside an instrument, and it
> does not ride into the third sitting.
> (d) REFBOT-SCAN-1 is ACCEPTED as evidence on disk. R257 is ANNOTATED at the
> register foot, never repaired: its "search-free" premise qualifies to
> Shrimp-Bot's TRAINING loop; the served artifact searches at deploy — Gumbel
> sequential halving, 16–128 sims — which corroborates the R254/R258 deploy
> lock. The candidates PARK on the rail: C-GUMBEL routes to the run6 sims-fork
> prereg adjudication (operator lock); C-RECOMPUTE queues as a post-mint seam
> candidate carrying its written witness, its upstream figure standing only as
> a prior because the author repudiated the record it comes from; C-PROVEN-LEAF
> parks to the SYS-5+ExIt arc with F-15's ExIt-injection term binding. Nothing
> enters the tree now.
> (e) COMMENT STYLE, operator direction, standard on contact via CLAUDE.md:
> comments only where needed; public APIs carry docstrings; no file-top banner
> comments; no narrative comment blocks — a needed comment states its
> non-obvious fact in one line. CARVE-OUT: load-bearing in-source markers —
> pinned bands, planted-break markers, armed-value provenance, license-required
> attribution — are mechanism, not commentary, and stay. Applied on contact,
> never as a cleanup pass.
> (f) F-RESIT-6 DEFAULT: the corrected block mints shakedown's n_workers to the
> Phase W pick with the other six configs. The operator's forwarding of the
> block confirms the default by silence; preserving a different value anywhere
> is his word before it.
> (g) The third sitting FORWARDS when (c)'s commit lands and the operator arms
> M — R315(d) is then satisfied. Intent unchanged: it mints or produces a
> card-level fact.
> ROUTE: land this, census, ACTIVE next version; annotate R257 in the same
> landing; execute (c) engine-side from its own worktree; add (e) to CLAUDE.md;
> one plain screen back.

# R317 — architect ruling, ARCH-ERA session, 2026-08-28 (the RECAL-SITTING-3 HALT ratified in full; the determinism control's defect diagnosed AT CAUSE as a wall-clock throughput proxy under a cross-regime carried band; the control RE-SPECIFIED on exact observables — net-parameter-hash GATE, move-sequence-hash DIAGNOSTIC, throughput REPORTED with no band; a measured noise floor with a strictly-conservative knee-rule amendment; the sitting RESUMES in the same session under the same grant) [INLINE]

**Provenance: [INLINE], authored.** Text originates in the ARCH-ERA architect session under R303(d), routed to the waiting RECAL-SITTING-3 session — same session, same grant, no new lock consumed; one canonical home: `PACKET_R317_DETERMINISM_RESPEC.md` §1. Byte-diff on append per R306(a) (de-marked, whitespace-normalized, line-by-line; extraction by tooling, never retyped; extraction and diff under `set -o pipefail`, R313(c)). **The paste carried the placeholder `R{next}`**, substituted to `R317` after verifying the head is R316 — census R23–R316, 287 sections / 287 distinct / 0 duplicates, excluded set 53, missing-in-range {24, 29, 32, 33, 227, 228, 267}; `--stamp` reported `STAMP OK: stamp v3.22 == §8 last entry v3.22`. The substitution is disclosed as the only edit to the canonical text.

> R317 — (a) The RECAL-SITTING-3 HALT is RATIFIED on every clause:
> halt-on-surprise applied to the instrument's own DIVERGED verdict; no
> mid-sitting instrument change improvised; the background ladder banked as
> evidence only; nothing touched. The conduct is the apparatus working.
> (b) THE DEFECT, AT CAUSE. The determinism control certifies through a PROXY —
> wall-clock moves_per_min conflates what the seed controls (the net, the
> trajectories) with what the machine controls (timing) — and its 1% band is a
> CROSS-REGIME CARRY: calibrated from one quiet engine-side measurement
> (0.5821%, n=1) and asked to certify a live-GPU drive whose own within-drive
> round noise is ~6% peak-to-peak. R302(c)'s voiding logic extends across
> regimes; the architect ratified past it at R316(a)(i), and that error is on
> the architect's ledger. The 0.5821% figure stays true in its regime; the 1%
> band is SUPERSEDED by (c).
> (c) THE CONTROL IS RE-SPECIFIED on exact observables, pre-registered here
> before any measurement it will judge: (i) GATE — net-parameter hash equality:
> the constructed net hashes EQUAL across both control drives AND across every
> ladder rung; any inequality is DIVERGED and HALTs. This tests exactly what
> the F-RESIT-10 repair claimed — same seed, same net — with no band. (ii)
> DIAGNOSTIC, non-gating — per-drive move-sequence hash: equality means the
> throughput spread is timing; inequality is a live-pipeline nondeterminism
> fact (batching/kernel class) recorded as a host-regime finding. This decides
> the sitting's two readings by measurement, not argument. (iii) REPORTED —
> the throughput spread, no band, beside the measured noise floor.
> (d) NOISE FLOOR, measured in the certifying regime: FOUR fresh same-seed
> drives at rung 4 on the box, after (c) lands; σ = the empirical std of the
> four drive means (each the mean of its 5 measured rounds, matching the
> ladder's unit), disclosed with the stated assumption that rung-4 relative
> noise carries across rungs. The knee rule is AMENDED in the only safe
> post-hoc direction — strictly conservative: the within set expands to every
> rung with metric ≥ (knee threshold − 3σ·best); the pick remains the smallest
> member. The expansion can only pull the pick toward fewer workers. No
> criterion here is fitted to a number that will be used: the banked ladder
> stays evidence-only and the pick comes from a fresh seeded ladder under
> (c)/(d).
> (e) THE SITTING RESUMES in the same session under the same grant. Order:
> land this ruling; implement (c)/(d) engine-side from an own worktree
> (R314(b)), full local gates, merge to dev, push, re-sync the box to the new
> tip and stamp it; then Phase W under (c)/(d); then the block unchanged —
> eval term to PLATEAU or DECLARED BOUND, measure-fit-mint-burst, M = 0.35
> binding as armed. The Phase W screen (pick, both hashes, σ, the within set)
> is WRITTEN INTO THE SITTING RECORD, not surfaced — RUN-TO-DONE holds.
> Intent unchanged: mint or a card-level fact.
> ROUTE: land this, census, ACTIVE next version, then execute (e) to done;
> one plain exit screen at the end.

# R318 — architect ruling, ARCH-ERA session, 2026-08-28 (the eval-stall investigation RATIFIED on every leg; the deploy head ADOPTS self-play's leaf batching, k read from the same leaf_batch_size knob, as a train/deploy ALIGNMENT rather than a concession; the deploy-lock word given by operator relay for exactly this change; the per-move cache release KEPT as the eval term's bounding mechanism; a falsifiable ~8x prediction rides as a surprise tripwire; the sitting resumes RUN-TO-DONE) [INLINE]

**Provenance: [INLINE], authored.** Text originates in the ARCH-ERA architect session under R303(d), routed to the waiting RECAL-SITTING-3 session — same session, same grant, the launcher's operator acts carrying; one canonical home: `PACKET_R318_DEPLOY_LEAF_BATCHING.md` §1. Byte-diff on append per R306(a) (de-marked, whitespace-normalized, line-by-line; extraction by tooling, never retyped; extraction and diff under `set -o pipefail`, R313(c)). **The paste carried the placeholder `R{next}`**, substituted to `R318` after verifying the head is R317 — census R23–R317, 288 sections / 288 distinct / 0 duplicates, excluded set 53, missing-in-range {24, 29, 32, 33, 227, 228, 267}; `--stamp` reported `STAMP OK: stamp v3.23 == §8 last entry v3.23`; last register header `R317`, no `R318` heading present, and no LIVE placeholder under `plan/` (every surviving occurrence is a landed disclosure's own prose, R316's recorded precondition note). The substitution is disclosed as the only edit to the canonical text, and the packet's PREAMBLE reference to the placeholder was deliberately LEFT UNSUBSTITUTED — R316's own over-broad-substitution near-miss, applied as the standing caution.

> R318 — (a) The eval-stall investigation is RATIFIED on every leg: the
> mechanism named at cause on three independent grounds (identical frames with
> a hot core and an idle card; arithmetic that reproduces games_total 0 from
> first principles; structural provenance to 075ae35, killing every regression
> hypothesis by the code's own history); the geometry bisect with its
> limitation disclosed rather than smoothed; the Phase W pick acquitted; the
> fix withheld because it moves the quantity STEP 1d measures and sits inside
> the deploy lock. The record's in-place correction of its own stale line and
> the flagged non-ship of R317(c)(ii)'s diagnostic under the escape clause are
> both accepted; the diagnostic stays on the books as non-blocking debt.
> (b) THE RULING. The deploy head adopts leaf batching: select_leaves(k) with
> k read from the SAME config knob self-play reads (leaf_batch_size; run5
> value 8) — no new constant enters the tree. GROUNDS: (i) the defect is the
> round-trip count and only batching reaches it; (ii) the net's policy and
> value targets are generated by k=8 leaf-batched search, so k=1 at deploy was
> the unexamined train/deploy mismatch — this change ALIGNS the shipped search
> with the regime the model is trained under, and eval remains deploy-matched
> by construction because eval runs the deploy head; (iii) the promotion
> protocol stays fair: both sides of every gate and of run6-vs-run5 search
> under one regime at fixed nodes. The per-move release_cuda_cache STAYS — it
> is the eval term's bounding mechanism; the term is measured with it, at
> k = leaf_batch_size, as production will run.
> (c) LOCK. This is a change within the R254/R258 locked surface. The
> operator's relay of this packet is his word on the lock for this change,
> named in §0. The lock itself is untouched: search at deploy remains of
> record; search-free deploy remains ruled out.
> (d) A FALSIFIABLE PREDICTION rides with the fix, as a surprise tripwire and
> not a gate: round-trips fall ~8× and the first re-run gate block completes
> WELL inside eval.round_timeout_sec at the minted geometry. If round 1 does
> not complete inside its budget, the mechanism statement was incomplete —
> halt on surprise, nothing measured around.
> (e) SEQUENCE. Implement (b) engine-side from an own worktree, full local
> gates, merge dev, push, re-sync the box and stamp the new tip. Re-run the
> battery at the minted geometry: PASS = games complete inside the budget AND
> the eval-term instrument returns a real series at last. Then STEP 1d at
> k = leaf_batch_size with the release in place, then STEP 2–4 per the block,
> M = 0.35 binding as armed. If the measured term refuses the partition at
> STEP 2 under M, that is a HALT and a card-level fact — no in-sitting knob
> search. RUN-TO-DONE; mint or card-level fact.
> (f) On PASS of (e)'s battery, the standing WPMAIN caveat "rc 0 does not
> certify eval health" is DISCHARGED by annotation at the next landing — its
> referent now has a mechanism, a fix, and a passing battery attached. Until
> that PASS it stands.
> ROUTE: land this, census, ACTIVE next version, then execute (e) to done;
> one plain exit screen at the end.

# R319 — architect ruling, ARCH-ERA session, 2026-08-28 (the R318(d) tripwire HALT RATIFIED and the retraction ACCEPTED as exemplary conduct; the CARD-LEVEL FACT banked — the pre-registered gate geometry cannot fit its own budget on this card class under any implementation; the EVAL-CHILD TERM DECOUPLED from gate geometry on a measured invariance ground with a two-point probe carrying its own falsifier; gate-geometry-vs-budget ROUTED to the operator with run6 held behind it; two field-scoped orders — no default readable as a measurement, progress_path goes live) [INLINE]

**Provenance: [INLINE], authored.** Text originates in the ARCH-ERA architect session under R303(d), routed to the RECAL-SITTING-3 session (same grants re-signed by the operator's forwarding); one canonical home: `DISPATCH_RECAL_SITTING3_RESUME.md` §2 — which SUPERSEDES the never-forwarded `PACKET_R319_TERM_DECOUPLE.md` (verified ABSENT under `plan/`, so no competing text exists and R285's ONE-TEXT rule is satisfied by construction rather than by cleanup). Byte-diff on append per R306(a) (de-marked, whitespace-normalized, line-by-line; extraction by tooling, never retyped; extraction and diff under `set -o pipefail`, R313(c)). **The paste carried the placeholder `R{next}`**, substituted to `R319` after verifying the head is R318 — census R23–R318, 289 sections / 289 distinct / 0 duplicates, excluded set 53, missing-in-range {24, 29, 32, 33, 227, 228, 267}; `--stamp` reported `STAMP OK: stamp v3.24 == §8 last entry v3.24`; last register header `R318`, no `R319` heading present, and no LIVE placeholder under `plan/`. The substitution is disclosed as the only edit to the canonical text; the dispatch's §1 reference to the placeholder was deliberately left unsubstituted, R316's over-broad-substitution near-miss applied as a standing caution.

**A STALE SHA IN THE FORWARDED TEXT, repaired in place under R311(c) rather than executed against.** §1 states `recal-mint-20260828 @ 0e566d9`; the branch is at **`28fcad1`**. `0e566d9` was the PRE-REBASE mint commit — the R318 landing rebased so `dev` carries only the fix and the mint branch is `dev` plus the single mint commit, with content verified identical (`git diff` between old and new tips: EMPTY). The dispatch's own `28fcad1` box-HEAD line is the correct one; the two lines describe the same tree. Noted in one line in the dispatch, no loop.

> R319 — (a) The R318(d) tripwire HALT is RATIFIED and the fix stands
> VERIFIED independently of it: the oracle proves ~8× fewer round-trips at a
> byte-exact identical node budget, both planted breaks bite. What was
> incomplete was the SIZE estimate, not the mechanism. The RETRACTION is
> ACCEPTED into the record as exemplary conduct: games_total 0 was a hardcoded
> broken-path literal read as a measurement — a default wearing a measurement's
> clothes, this project's top defect class, caught by the session's own record
> discipline. Every previously ratified conclusion survives on the positive
> observation that always carried it: the phase marker is emitted only after
> the gate block returns and was absent in every drive. Nothing in R317 or
> R318 moves.
> (b) THE CARD-LEVEL FACT, banked by this sitting regardless of what follows:
> the pre-registered gate block (screen_games 80 × deploy_sims 150, ~1.5M leaf
> evaluations; confirm 128 on escalation) costs on the order of an hour of
> this card's FULL measured self-play throughput, and multiples of that beside
> training — the executor re-derives the throughput figure from the ladder
> artifacts at point of use, verify-at-HEAD. round_timeout_sec 3600 cannot
> hold that geometry on this card class under ANY implementation. The residual
> defect is the pre-registered geometry/budget pair, not the code path.
> (c) THE EVAL-CHILD TERM IS DECOUPLED from gate geometry, on a MEASURED
> ground, pre-registered here before the measurement: the child plays games
> sequentially, so peak device demand is per-game-structured and invariant to
> screen_games; the term is denominated in (deploy_sims, leaf_batch_size,
> model/config shape) only. STEP 1d therefore runs a diagnostic eval geometry
> — production sims and k, throwaway generous timeout, disclosed — as a
> TWO-POINT INVARIANCE PROBE: a 4-game round, then an 8-game round. From the
> first round's observed per-game peak variance the session pre-states a
> tolerance BEFORE driving the second; the two round peaks must agree within
> it. Agreement: the larger point is the term, at least three diagnostic
> rounds supply the round-dependence series to the existing PLATEAU/GROWING/
> REFUSED instrument, and STEP 2–4 proceed unchanged with M = 0.35 binding.
> Disagreement FALSIFIES the invariance ground and HALTs — the probe is an
> instrument, not an assumption. The minted term SURVIVES any later
> re-preregistration of gate geometry that preserves sims, k, and model shape.
> (d) GATE GEOMETRY vs BUDGET is ROUTED TO THE OPERATOR as a prereg
> adjudication, prepared by the architect after this sitting: raise
> round_timeout to measured need with margin; reduce screen/confirm with the
> power loss stated; or a sequential (SPRT-class) gate per the REFBOT
> evidence, expected games typically well under fixed-N at equal error rates.
> Nothing in this sitting touches those rows. RUN6 DOES NOT START until this
> adjudication lands — a promotion gate that cannot complete inside its own
> budget is the defect class this era exists to end.
> (e) TWO FIELD-SCOPED ORDERS land with (c)'s implementation: (i) the
> broken-round path stops reporting a valid-looking games_total — absent or an
> explicit sentinel, pinned by a test; a default must not be readable as a
> measurement again. (ii) RoundSpec.progress_path goes LIVE minimally: the
> child writes per-game progress (game index, move count, timestamp), the
> parent logs it, escalation semantics UNCHANGED this sitting — observability
> first, policy later. A declared field with no consumer does not remain in
> the tree.
> (f) VERIFY THE HISTORY: whether any production eval round has EVER completed
> (search run5-era artifacts for a success-path games_total). The answer lands
> in the sitting record either way; if none, run5 ran effectively ungated,
> recorded plainly, and (d)'s urgency is thereby measured, not argued.
> (g) SEQUENCE: implement (c)+(e) engine-side from an own worktree, full local
> gates, merge dev, push, re-sync the box; STEP 1d per (c); STEP 2–4 per the
> block; RUN-TO-DONE. Mint or card-level fact — noting (b) already banks one.
> ROUTE: land this, census, ACTIVE next version, then execute to done; one
> plain exit screen at the end.

# R320 — architect ruling, ARCH-ERA session, 2026-08-29 (PERF-TRANCHE-1 RATIFIED with A3's refutation accepted as evidence and the 7.2% pre-control discrepancy left OPEN as instrument hygiene; the 2026-08-29 gate-geometry adjudication WITHDRAWN as fatally defective with the error placed at the architect's desk, its harness check RATIFIED at mechanism — the eval outcome channel is a CONSTANT, so 0/13 is not a slow gate but no gate at all; the SEQUENCING INVERTED, ply-cap adjudication made the PRECONDITION of gate geometry and the EVAL_POSTURE_OPTIONS §4.4 posture ADOPTED FOR MEASUREMENT for one round; the screen's 84.3% unconditional-escalation property RECORDED so sitting 4 does not inherit it) [INLINE]

**Provenance: [INLINE], authored.** Text originates in the ARCH-ERA architect session under R303(d), routed to this EVAL-CHANNEL-1 dispatcher (the operator's forwarding is the box grant and his signature on the §4.4 measurement posture); one canonical home: `DISPATCH_EVAL_CHANNEL_1.md` §2, written to disk from the forwarded paste before extraction, and no competing text exists under `plan/`. Byte-diff on append per R306(a) (de-marked, whitespace-normalized, line-by-line; extraction by `awk` over the canonical home, never retyped; extraction and diff under `set -o pipefail`, R313(c)). **The paste carried the placeholder `R{next}`**, substituted to `R320` after verifying the head is R319 — census R23–R319, 290 sections / 290 distinct / 0 duplicates, excluded set 53, missing-in-range {24, 29, 32, 33, 227, 228, 267}; `--stamp` reported `STAMP OK: stamp v3.25 == §8 last entry v3.25`; `--self-test` all controls fire; last register header `R319`, no `R320` heading present, and no LIVE `R{next}` placeholder under `plan/` (the ten files carrying the string carry it as a REFERENCE to the convention, none inside a quoted canonical block — checked as `^> *R{next}`, zero hits). The substitution is disclosed as the only edit to the canonical text, and it was applied to the canonical home as well as to this entry so the two are byte-identical; §1.3's "head must be R319" is a precondition statement, not a placeholder, and was left untouched.

**Clause (a)'s referents verified at point of use before landing, not transcribed from the paste.** `plan/research/PERF_TRANCHE1_RESULTS.md`: A3 `REFUTED` at :80 with the F-10 correction at :109 and :190-192; A4 `NOT LANDED` at :81 with the 0.157 ms/sim figure and the non-attempt's grounds at :155-165; the 7.2% pre-control-vs-ledger gap at :11-12 and :223. **Clause (b)'s harness chain re-derived at HEAD** (`hexo-mantis` @ `3d4636a`), each link at its source: `src/mantis/arena/match.py:122-125` (`plies >= max_plies` and `adjudicator is None` ⇒ `winner = "draw"`, `_DEFAULT_MAX_PLIES = 128` at :66); all **seven** committed configs ship `eval.ply_cap_adjudication: null` (`grep -rn` over `configs/`); `src/mantis/eval/aggregate.py:62-63` scores a draw `0.5`; `should_escalate` at :205 and `gate_promotion_decision` at :211-217; `configs/run5.yaml:34-35` `promotion_winrate: 0.55`, `screen_confirm_lo: 0.44`; and the ~350 s/game, every-game-at-the-128-ply-cap measurement at `RECAL_SITTING3_RECORD_2026-08-28.md` §10.3. **Clause (d)'s 84.3% recomputed** by exact binomial tail — P(X >= 36 | n=80, p=0.5) = 0.842847 — together with the withdrawn text's two corrected figures (screen-80 false-pass 0.217021; end-to-end power 0.9716 × 0.4954 = 0.4813). The withdrawal itself is on the record at `plan/ADJUDICATION_EVAL_GATE_GEOMETRY.md` (committed `fa73a01`), reviewed against `plan/EVAL_POSTURE_OPTIONS.md` §2/§4.2/§4.4 and the pinned constant-channel test `tests/eval/test_strength_floor_gate.py::test_an_all_ply_cap_probe_reads_a_healthy_half_on_the_WIN_RATE_axis_alone` before it was committed.

> R320 — (a) PERF-TRANCHE-1 is RATIFIED: witness verdicts as tabled; A3's
> refutation accepted as evidence correcting F-10; the A4 non-attempt ENDORSED
> (correctness exposure outranks its 0.157 ms/sim); Appendix A's in-place
> corrections ratified; the in-tranche defect disclosure accepted. The 7.2%
> pre-control/ledger discrepancy stands OPEN as instrument hygiene: ledger
> absolute levels are not quotable without re-measurement.
> (b) The gate-geometry adjudication of 2026-08-29 is WITHDRAWN as fatally
> defective, error at the architect's desk: authored without reading
> plan/EVAL_POSTURE_OPTIONS.md, whose §2 already states the outcome channel is
> a constant; its arithmetic additionally erred in the flattering direction.
> The harness check is RATIFIED at mechanism: every game runs to the 128-ply
> cap, every config ships ply_cap_adjudication null, a capped game with no
> adjudicator scores draw, a draw scores 0.5 — so WR ≡ 0.500 with zero
> variance, the gate escalates unconditionally and returns constant False.
> 0/13 is explained: it is not a slow gate; it is not a gate.
> (c) SEQUENCING INVERTED: ply-cap adjudication is the PRECONDITION of gate
> geometry. The EVAL_POSTURE_OPTIONS §4.4 posture is ADOPTED FOR MEASUREMENT:
> longest_run_margin, seat-neutral, min_margin 1, wired for one measurement
> round whose tally reads the margin distribution nobody has measured — how
> capped games decide, at what margins, with what seat balance and decisive
> rate. Production adjudication values and the ply-cap matrix remain operator
> prereg, grounded by that distribution.
> (d) RECORDED: even on a healthy channel the shipped screen never screens —
> P(escalate | truly-50%) ≈ 84.3%. Sitting 4 does not inherit the screen
> geometry unexamined; geometry is re-adjudicated AFTER (c)'s distribution
> lands, a sequential gate the likely destination, draw increments defined
> then. The sealbot rung repair rides unchanged.
> ROUTE: land this, census, ACTIVE next version; execute the measurement
> round; one plain exit screen.

# R321 — architect ruling, ARCH-ERA session, 2026-08-30 (the universal model contract ADOPTED as the design of record, GnnNetV2 re-cast as its first tenant rather than its purpose; two carried perf figures CORRECTED on the record, both found by the verify-at-HEAD pass and one of them favourable; lane B0 ORDERED — the conformance suite is 33 ahead / 35 behind dev and nothing in the seam lane has a base until it lands; the identity primitive's promotion out of diagnostics made a PRECONDITION of the identity part; the aux-head ban HELD until its replacement rule ships; the staged eval gate pre-registered with bootstrap ELEVATED) [INLINE]

**Provenance: [INLINE], authored.** Text originates in the ARCH-ERA architect session under R303(d), routed to this SEAM-B0 dispatcher, whose authorization is the operator's forwarding; one canonical home: `PACKET_R321_SEAM_V1.md` §1, and no competing text exists under `plan/`. Byte-diff on append per R306(a) (de-marked, whitespace-normalized, line-by-line; extraction by `awk` over the canonical home, never retyped; extraction and diff under `set -o pipefail`, R313(c)). **The packet carried the placeholder `R{next}`**, substituted to `R321` after verifying the head is R320 — census R23–R320, 291 sections / 291 distinct / 0 duplicates, excluded set 53, missing-in-range {24, 29, 32, 33, 227, 228, 267}; `--stamp` reported `STAMP OK: stamp v3.26 == §8 last entry v3.26`; `--self-test` all four controls fire; last register header `R320`, no `R321` heading present. **The LIVE-placeholder check is `^> *R{next}` under `plan/` and it returns EXACTLY ONE hit — this packet's own canonical block**, which is the condition the packet's §0 states for its own landability; a zero-hit reading would refuse the very file being landed, and that is the R312-onward correction applied rather than re-discovered. The substitution is the ONLY edit to the canonical text and was applied to the canonical home as well as to this entry, so the two are byte-identical; the `R321` occurrences in the packet's filename and preamble are REFERENCES, not canonical text, and were left untouched.

**The ruling's referents re-derived at point of use before landing, not transcribed from the packet.** Clause (a)'s design of record is `plan/SEAM_V1_DESIGN.md`, committed `d0c3321`. **Clause (b)(i)** against `plan/research/PERF_TRANCHE1_RESULTS.md:144` — floor **1.494**, post-tranche served **2.805**, so 1.494 ÷ 2.805 = **53.26 %**; the superseded 44 % sits at `plan/research/PERF_RESEARCH.md:828` against the pre-tranche **3.396** card, and 1.494 ÷ 3.396 = **43.99 %** confirms the stale figure was correct for its own era and stale only by a tranche. **Clause (b)(ii)** against `plan/research/PERF_BASELINE_LEDGER.md:474,511` — **12.7 %** mean GPU utilisation on the 25-move arm, **14.5 %** mean with **96.8 %** of samples below 25 % on the 64-move arm, i.e. ~85–87 % idle. **Clause (c)'s divergence measured, not carried:** `git rev-list --left-right --count dev...planc-conformance` reports `35 33` at landing — **33 ahead / 35 behind**, the packet's figure exactly — and the suite is **nine** files under `tests/model/conformance/`, **seven** of them test modules. **Clause (d)'s observable** is `_net_param_hash` at `src/mantis/diagnostics/worker_sweep.py:901`, its one in-module caller at `:1031`, and the R81 oracle naming it at `tests/diagnostics/test_worker_sweep_determinism.py:57`. **Clause (e)'s tripwire** is `GRAPH_FORBIDDEN_NONZERO_WEIGHTS` at `src/mantis/train/trainer/core.py:81`, raised at `:547-552` — one named site to retire, as (e) states.

> R321 — (a) THE UNIVERSAL MODEL CONTRACT is ADOPTED as the design of
> record: `plan/SEAM_V1_DESIGN.md`, its seven parts, its governing principle
> (contract, not call graph — boundaries at batch level, hot paths compiled
> per-arch with zero runtime indirection, conformance proven offline). PLANC
> Phase-2 is RE-SCOPED to it. GnnNetV2 is the first TENANT; the product is the
> CONTRACT it proves. The accept bar for the whole mission is one sentence and
> it is testable: after V1, adding a model kind means writing the arch behind
> the contract, letting the suite prove it, and training it after the mint —
> if a second arch still needs edits to the trainer, the server, the arena, or
> the config schema outside its own scope, the seam did not ship.
> (b) TWO CARRIED FIGURES ARE CORRECTED ON THE RECORD, both found by the
> verify-at-HEAD pass that preceded the design and neither by a reviewer.
> (i) "the 44% floor" was stale by one tranche: 44% was the floor's share of a
> 3.396 ms/sim card; PERF-TRANCHE-1 moved the served level to 2.805, so the
> floor is 53% of the card. The correction runs FAVOURABLE and strengthens the
> fused-kernel row — the lever now addresses a majority of self-play cost.
> (ii) "~80% idle card" understated its own measurement: eval-path GPU
> utilisation is 12.7% / 14.5% mean with 96.8% of samples below 25%, so the
> card is ~85-87% idle. The 80% figure is the operator's own conservative
> phrasing, so no decision built on it moves. Both are recorded rather than
> quietly restated, per the dispatcher-correction discipline: a figure that
> travels is corrected in the artifact readers open.
> (c) LANE B0 IS ORDERED, and it is the first act of the seam lane: the
> conformance suite exists ONLY on `planc-conformance`, measured 33 ahead /
> 35 behind `dev`. Rebase it onto `dev`, run the full local gate set, land it.
> Suite v2 has no base until this does. The design brief omitted this and the
> omission is on the architect's ledger. Standing conjuncts are UNCHANGED and
> are re-measured at the moment the work starts, never inherited (R295(c)):
> the collision table clear, full local green, floor ratcheted in its own
> commit.
> (d) IDENTITY (part 6) carries a PRECONDITION, not a consequence: R317(c)(i)'s
> exact observable lives at `_net_param_hash` in
> `src/mantis/diagnostics/worker_sweep.py`. A determinism gate, a mint
> denomination and a provenance primitive do not live in a diagnostic module.
> Promote it into the model contract with its callers re-pointed, as bounded
> work, BEFORE part 6 is specified against it.
> (e) THE AUX-HEAD BAN IS HELD. `GRAPH_FORBIDDEN_NONZERO_WEIGHTS`
> (`src/mantis/train/trainer/core.py`) is a trainer-side tripwire standing in
> for a property; the property is the refbot evidence's zero-serve-cost rule —
> train-only heads are free behind the seam because the serving wire is fixed
> by contract. Replace the assertion WITH the rule, in one act. Retiring the
> ban ahead of its replacement is NOT licensed by this ruling, and a seam that
> cannot yet fix the serving wire by contract has not earned the retirement.
> (f) THE STAGED EVAL GATE is pre-registered as in `SEAM_V1_DESIGN.md` §7,
> values operator-held. EVAL-CHANNEL-1's finding stands as its ground and is
> restated for the record: both shipped criteria are degenerate at this
> maturity, so no `min_margin` and no choice inside the closed criterion set
> decides anything. R320(c)'s sequencing inverts once more and the inversion
> HOLDS: adjudication is necessary but NOT sufficient; the binding
> precondition is a checkpoint mature enough to produce a decidable position.
> BOOTSTRAP BC-PRETRAIN IS THEREBY ELEVATED — it is both the early-strength
> fix and the earliest path to a calibratable gate. Its own precondition is
> unchanged and is the operator's: F-816-25's fix precedes any BC-pretrain.
> The adjudication doc is the architect's next deliverable.
> (g) THE 500-1000 g/h LEDGER is ADOPTED AS A LEDGER AND NOTHING MORE. Every
> row is DERIVED until its witness lands; the product ~5-15x makes the goal
> feasible ON PAPER and promises nothing. Strength is held by fixed-node
> gates. THE BINDING CONSTRAINT IS THE STRENGTH-AT-FEWER-SIMS WITNESS, which
> is what (f) makes measurable — so no row that trades search for speed is
> acceptable before the gate can tell two checkpoints apart. Sequencing
> follows from that, not from convenience.
> (h) SEQUENCE. Lane A (box) and lane B (no box) run in parallel and do not
> collide. A: `strength_floor` arming values proposed by the architect, then
> the sealbot rung repair, then sitting 4 re-fits and mints ONCE — terms,
> geometry and gate stage-1 values in one act. B: B0 per (c), then the
> contract spec, then the suite v2 sections, then GnnNetV2 as the proving
> tenant. Perf tranche-2 slots after A's geometry decision, per the operator's
> WAIT-until-geometry ruling; Lane C after tranche-2 re-baselines. RUN6 WAITS
> ON LANE B — standing direction, unchanged.
> ROUTE: land this, census, ACTIVE next version, then B0 as its own dispatch;
> one plain exit screen.

# R322 — architect ruling, ARCH-ERA session, 2026-08-30 (SEAM-B0's extras and SEAM-B1 RATIFIED in full, W-C1 named as the leg's finding; R257's [r8] fence NARROWED by annotation on strix stage S5; the five scout-flagged reference-row items corrected in place; SEAM-B2 ADOPTED as three legs — the eight red config-partition rows die by repair with candidate D beside them, the dense-lineage migration derives reachability structurally, and two components land UNARMED behind the contract because LANDING IS NOT ARMING) [INLINE]

**Provenance: [INLINE], authored.** Text originates in the ARCH-ERA architect session under R303(d), routed to this SEAM-B2 dispatcher, whose authorization is the operator's forwarding; one canonical home: `PACKET_R322_SEAM_B2.md` §1, and no competing text exists under `plan/`. Byte-diff on append per R306(a) (de-marked, whitespace-normalized, line-by-line; extraction by `awk` over the canonical home, never retyped; extraction and diff under `set -o pipefail`, R313(c)). **The packet carried the placeholder `R{next}`**, substituted to `R322` after verifying the head is R321 — census `R23–R321, 292 sections / 292 distinct / 0 duplicates, excluded set 53`, missing-in-range `{24, 29, 32, 33, 227, 228, 267}`; `--stamp` reported `STAMP OK: stamp v3.27 == §8 last entry v3.27`; `--self-test` all four controls fire; last register header `R321`, no `R322` heading present. **The LIVE-placeholder check `grep -rn '^> *R{next}' plan/` returns EXACTLY ONE hit — this packet's own canonical block** (R321's correction, applied rather than re-discovered). The substitution is the ONLY edit to the canonical text and was applied to the canonical home as well as to this entry, so the two are byte-identical; the `R322` occurrences in the packet's filename and preamble are REFERENCES, not canonical text, and were left untouched.

**The ruling's referents re-derived at point of use before landing, not transcribed from the packet.** **Clause (a)'s** race fix is `4ee60d6` on `dev` — *"the graph ring excludes by mutex, not by a PyRefMut held across the GIL release"*, 4 files, `tests/bridge/test_gil_release_on_native_calls.py` added; SEAM-B1's three sections are `9ce0a06` (T7), `5269074` (T8), `cf096e6` (T9), GnnNetV2 is `c5c25cb`, and the floor ratchet `4043 → 4117` is `2e25633`, the branch tip. **W-C1's figures are read at their producer**, `plan/SEAM_B1_TASKS.md:136-140`: node counts (16, 128, 1024) — an **8× and then a 64×** node increase — V1 `‖agg[dummy]‖` 4.11 → 47.5 → 322.9 (**78.5×**), V2 0.46 → 0.82 → 1.14 (**2.49×**). **Clause (b)'s** subject: R257's text is `plan/rulings_register.md:3882` and its fence (i) names *"radius-8 game … radius-dependent arithmetic do not [transfer]"*; the narrowing ground is `plan/research/SCOUT_2026-08-30_CANDIDATES.md:143` and `:922`, which measures stage **S5 = `win_length 6`, `placement_radius 6`, unbounded**. **Clause (c)'s** five items are enumerated at `plan/research/SCOUT_2026-08-30_CANDIDATES.md:899-928`; four land in `plan/SEAM_V1_DESIGN.md` §6.4's reference row (`:268-317`) and its neighbour list, the fifth in `inputs/klent_assessment.md:91-93`. Item **4 is R257's fence and is discharged by clause (b)'s annotation, not by an in-place edit** — the two clauses partition the five, they do not overlap. **Clause (d)'s "eight red rows" is DERIVED, not counted from prose**: `tests/model/conformance/test_config_partition_shared_vs_arch_scoped.py` declares `ARCH_SCOPED` with **four** `ArchScopedKey` rows (`train.microbatch_caps.{max_edges,max_nodes}`, `inference.fused_graph_caps.{max_fused_edges,max_fused_nodes}`) and `DECLARED_RED_ROWS` is their product with the **two** red classes — 4 × 2 = **8**. **Candidate D's premise** is B1's own disclosed limitation at `plan/SEAM_B1_TASKS.md:120-123` (*"V2 is not yet selectable from a minted config"*), and the site is `arch_from_spec_and_config` in `src/mantis/model/arch.py`, which returns `CnnArch` or `GnnArch` and can return `GnnArchV2` on no input. **Clause (d)'s "§4 policy"** is `plan/SEAM_V1_DESIGN.md:173-186` — *"Migrate what we will ablate against. Archive the rest with their goldens and a one-line grave note."* **Leg 3's two components** are the scout's λ-return CODEC row (`plan/research/SCOUT_2026-08-30_CANDIDATES.md:288`, whose witness sketch is the pure-function golden plus the mover-sign pair) and card 1 `S-HLGAUSS` (`:551-598`); the census is card 6 `S-CODEBOOK`'s witness (i) (`:790-844`).

> R322 — (a) RATIFIED: SEAM-B0's extras — the run-fatal PyRefMut race fix
> (4ee60d6) with its mutation self-test pattern commended into practice — and
> SEAM-B1 in full: the three suite sections with their planted breaks, GnnNetV2
> proven behind the contract, the witness results including W-A1's disclosed
> instrument defect, and W-C1 as the leg's finding: GNN-3's size-generalization
> hazard is measured (V1 dummy-aggregation norm 78.5× over a 64× node increase;
> V2 2.49×). RESEARCH-SCOUT-1 is ACCEPTED as evidence on disk.
> (b) R257 is ANNOTATED at the register foot, never repaired: strix stage S5 is
> win_length 6 / placement radius 6 / unbounded — this project's exact rule
> set. The [r8] fence NARROWS to their published figures and non-S5 stages;
> S5-stage material transfers directly, still behind the seam with witnesses.
> (c) The five reference-row items the scout flagged are corrected IN PLACE per
> docs-follow-reality, one disclosed line each; the scout was right to flag and
> not fix.
> (d) B2 SCOPE ADOPTED, three legs. LEG 1: T9's eight red rows die by repair —
> the graph-only cap keys become properly arch-scoped and leave the grid
> configs; the ratchet rows must be deleted by the fix, never widened; a repair
> that would touch a MINTED row is a HALT, not a judgment call. Candidate D
> lands: an arch selector makes V2 selectable and suite-proven via a throwaway
> diagnostic config; every shipped production config still selects its current
> arch, byte-unchanged. LEG 2: migration per the §4 policy — derive
> reachability structurally (structure, not text): a dense-lineage arch with no
> production config selecting it and no non-test consumer reaching it is
> ARCHIVED (goldens captured, grave note written, code fenced from build_net's
> dispatch); anything load-bearing is SURFACED with its consumers named, not
> archived. Redundancy harvest is BOUNDED to what repair and archive expose;
> larger unifications are surfaced with measured scope for a later packet.
> LEG 3 (severable): two components land UNARMED behind the contract with the
> scout's witnesses — the λ-return value-target CODEC (pure-function golden +
> mover-sign pair; no checkpoint, no box) and S-HLGAUSS re-binning into dist65.
> Proven by the suite, selected by nothing, armed by nothing: LANDING IS NOT
> ARMING, and arming is the operator's run6 prereg. The S-CODEBOOK reachability
> census (3^11 patterns at run5 density, no training) runs as a measurement and
> updates its card in place.
> (e) Nothing trains, the box is untouched, graves stay dead. ROUTE: land this,
> census, ACTIVE next version; execute the legs; one plain exit screen.
