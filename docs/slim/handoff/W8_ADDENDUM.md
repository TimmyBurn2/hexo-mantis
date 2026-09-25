# W8 ADDENDUM — the close (read after WAVE_BRIEF.md and HANDOFF.md)

W8 closes SLIM-FIX (R368). It runs in this order, and every step lands as its own commit:
1. W8-D, the operator grants.
2. The run10 MATCH, taken while the handoff tools still exist.
3. The AFTER figure and the exit record.
4. The STATE paragraph.
5. `git rm -r docs/slim`.
6. REVIEW-W8.
7. The full `make gates.exit`.

The wave then STOPS at READY-TO-MERGE. It does not fast-forward dev and it does not push. The operator
enacts the merge after reading the exit record.

## W8-D — operator grants (2026-09-25; execute FIRST, before the AFTER figure and the final sweep)

On 2026-09-25 the operator decided the points W6 and W7 left open. The decision was delegated to the
original session with the words "go with the architecturally cleanest decisions". Rules for every item:
- Each item lands as ONE commit. The subject names the grant, for example
  `test(config): … (operator grant 2026-09-25)`.
- Each item is recorded in `docs/audits/SLIM_FIX_EXIT_<date>.md`, with its sha and what was run.
- REVIEW-W8 checks that each item was executed EXACTLY as scoped, no wider.

1. **run6.yaml STAYS in configs/.** The operator wants it kept as a record, but nothing may bind it BY NAME.
   - (a) Re-point the R310-frozen `tests/tools/test_preflight_mint_process.py` oracle (`RUN5 =`, :50, plus
     its other by-name reads). Use the EQUALITY-KEEPING shape: derive the forced leaves from the base config
     the oracle runs on, and add the heldout_gap leaves only when that base carries the block, so
     `differing == FORCED_TWIN_LEAVES` survives. Do NOT use the filed subset-narrowing diff
     (`halt_twin_heldout.diff`, which REVIEW-W6 #7 showed NARROWS the oracle). The grant covers the
     frozen-file edit. Plant a twin that fails to replace a forced leaf; the oracle must red.
   - (b) DELETE `tests/config/test_eval_config_remint.py::test_the_ruled_deploy_sims_are_pinned`. A minted
     value is provenance, not an expectation. The 160 stays recorded in R346(b) and in run6.yaml. Lower
     test_count_floor in the same commit.
   - (c) The other by-name readers re-point to the census under R368(e)'s standing order ("tests binding
     them re-point to the census"). Those are `tests/tools/test_preflight_mint.py` (:69–72 comments, :299,
     :602, :621, :638, :656) and any others `git grep` finds. If a reader has no census form, the row HALTs.
   - Exit check: `git grep -n 'run6\.yaml' -- src tools tests` returns nothing but history prose. The
     armed_aborts owner/note text is item 2's. run6 then remains an ordinary census member.
2. **`src/mantis/config/armed_aborts.py` MANIFEST PROSE.** Only the owner, note and RESIDUAL text moves.
   The `required`, `config_path` and `source_pin` fields never move. Correct the prose to current facts:
   - `policy_loss_trough` and `ply_cap_attractor` lose the run-named and stale "run7's mint proposes"
     wording. Make it true, e.g. "no production config has armed it yet; a mint that pre-registers {…}
     flips this row REQUIRED" (REVIEW-W6 #3).
   - `terminal_eval_broken`'s RESIDUAL cites the test file that actually witnesses it (the census row
     c79b4e79 landed), not the non-existent `test_minted_config_remint.py`.
   - Gate 12 (`tools/ci_gates/preflight_mint.py --audit-only`) is green before and after, and
     `tests/config/test_armed_abort_manifest.py` stays green.
3. **`src/mantis/eval/pipeline.py`'s round-completion catch-all** follows CLAUDE.md: the
   `logger.exception` message no longer repeats `repr(exc)`.
   - The O-30 witness,
     `tests/eval/test_eval_broken_reason_routes.py::test_the_round_completion_route_logs_a_traceback_and_the_detail`,
     re-points its detail assertion to the record's `exc_info`. The detail is still logged, so this is
     not a narrowing.
   - Planted break: drop `exc_info`, and the witness must red. Quote the red line.
4. **REVIEW-W6 #12.** Annotate R368's Amends line about R336(e) at the register foot, as the register
   requires: R336(e)'s on-contact clause is the rustfmt one. Then drop the "comment-application half"
   gloss from CLAUDE.md's Rust bullet.
5. **`docs/governance/archive/README.md`.** Keep ONE tombstone line (what was deleted, plus the commit)
   and drop the RULINGS_ACTIVE.md narrative paragraph.
6. **`Trainer.load_checkpoint`** (`src/mantis/train/trainer/core.py`, a classmethod with zero callers):
   DELETE it. LAW-12 allows one loader, `mantis.train.checkpoints.load_checkpoint`.
   - Run the resume and checkpoint conformance tests before and after: tests/train/test_checkpoint*,
     the resume tests, and tests/model/conformance/** read-only.
   - run10 MATCH after.
7. **CARD-POISON-STANCE stays CARDED.** It is a runner error-behaviour change beyond slimming, and it
   lands after run10 STARTs. Record the recommendation in the exit record: `into_inner` for
   latch/stop/finalize/spawn, and a named error for the drain faces. Record the ground with it.
8. **REVIEW-W2 #3** (par.rs fan-out without a LAW-09 bench) stays a recorded deviation, in the exit
   record.
9. **run7.yaml and run8.yaml stay deleted** per R368(e). History keeps them, and every run dir has its
   `resolved_config.yaml`. Record this in the exit record.

10. **Carried from W7 (on contact, not granted work; record them in the exit record, or land each as one
    commit if it is trivially safe):**
    - The three refusal STRING literals in `crates/mantis-encoding/src/spec/validate.rs` (≈ :135–151) that
      still name the deleted `sym_tables_for` and D6 chain tables (REVIEW-W7 #9). A refusal-text edit is
      code; if you land it, `registry_census.rs` and `axis_pin.rs` must stay green, and a test that reads the
      text must not red.
    - The style verifier's method limits (REVIEW-W7 #18: a literal-exact Rust pass, a pragma grep, a
      heredoc-range guard). They matter only if a later style pass reuses the verifier; record it in the
      exit record.

After W8-D: run10 MATCH, `comment_lint` GREEN, and gates 7/10/12/13/15/17 rc 0. Fold the floor moves.
`docs/slim/handoff/integrate.sh` still exists at this point.

## W8-E — the AFTER figure and the exit record (BEFORE docs/slim goes)

- Measure the AFTER figure at the post-W8-D tip with `docs/slim/00_MAP.md` §1's two commands (per top dir
  and per package). Set it beside §1's BEFORE figure (1 071 files / 419 648 text lines at 1e8d6d6), with
  Δ per row.
- Write `docs/audits/SLIM_FIX_EXIT_<date>.md`. It carries everything that dies with docs/slim:
  - the BEFORE and AFTER figures, both commands verbatim;
  - the per-wave ledger summary: W0–W8, rows done, refuted-at-contact and still-C with grounds, from
    PROGRESS and LEDGER §6;
  - every HALT and how it closed (the W6 run6 HALTs close by W8-D item 1);
  - the operator-owed decisions and their W8-D dispositions;
  - the review list: REVIEW_W1..W8 paths plus each one's must/should/note counts and closure;
  - the model-per-leg record, from each wave's dispatch log;
  - the KNOWN-RED's account and cure (cdbc8000);
  - W8-D's nine items with their shas.
- Move `docs/slim/handoff/halt_twin_heldout.diff` to `docs/audits/SLIM_FIX_halt_twin_heldout.diff`
  (`git mv`) BEFORE the rm. It is the rejected shape, and REVIEW-W6 #7 and the grant refer to it.
- STATE.md gets ONE exit paragraph: SLIM-FIX exited, the range, the exit record's path, and
  READY-TO-MERGE pending the operator's merge.

## W8-F — dissolve docs/slim

- `git rm -r docs/slim`, then `test ! -e docs/slim`.
- Any doc gate 10 scans (Makefile, README.md, CLAUDE.md, docs/contracts, docs/governance; RULINGS.md is
  SCAN_EXEMPT) that still cites a docs/slim path gets a `DISSOLVED_PATHS` row in
  `tools/ci_gates/check_tracked_refs.py`: `"docs/slim/": "dissolved at SLIM-FIX's close (R368); the exit
  record is docs/audits/SLIM_FIX_EXIT_<date>.md"`. Do not rewrite those docs.
- Gate 10 rc 0 after. A comment-only edit to a gate is fine; the value line is the order.
- `integrate.sh` and run10_resolved.py die with the directory. For commits after this, fold floors by
  hand (measure, write the floor, `--amend`) and take the run10 MATCH from the exit record's pinned
  procedure. Copy `run10_resolved.py` into the exit record's appendix, or check it before the rm.

## W8 exit

1. A fresh read-only REVIEW-W8 (opus) over W8's range. It checks:
   - W8-D executed exactly as scoped;
   - the exit record against PROGRESS and LEDGER;
   - the AFTER figure recomputed;
   - R367(a), R368(b)/(c)/(i) and correctness.
   File it verbatim as `docs/audits/REVIEW_W8_<date>.md` with §5. Two fix loops max, then HALT.
2. The FULL `make gates.exit` (`run_all.sh --with-slow`, ≈ 2 h+) in `.wt/gates`, run as a systemd user
   unit exactly as W7's sweep was (see the exit record's procedure). It must print ALL GREEN. A red you
   cannot attribute to the wave is recorded, not fixed.
3. STOP at READY-TO-MERGE. Do not fast-forward and do not push; the operator enacts ENACTS 4. Remove
   every w8-* worktree and branch.
