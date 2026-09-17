# repo_design.md — the contract-#5 amendment chronicle, v1 → v8 (ARCHIVED 2026-09-17)

Moved verbatim out of `docs/design/repo_design.md` §4 by REPAIR-A4 step 11 (R355(f)). These six
AMENDMENT sections record how the run-config schema contract grew from v1 to v8 and how the
"deferred doc half" was discharged; `docs/contracts/run_config_schema.md` carries every version row
since and is the authority. Frozen: never edited, corrected only by a note in the live file.

### AMENDMENT — contract #5 v1 → v2, with the doc half of the same-commit clause DEFERRED

**WPAX Phase S (card CARD-SMOKE-SEAM), ADJ-09 Option B.** Recorded here rather than left as
silent drift (R9).

1. **The bump.** `train.max_train_steps` is a NEW **required** field on `RunConfig` — the
   run-length authority, resolved to `StepCoordinatorConfig.stop_step`, which is the real
   stop condition. This is an *incompatible* change: a config file lacking the key fails to
   load. That is strictly more than the precedent the contract doc's own status line records
   for WP8 (*"additive founding growth … no key changed incompatibly; the config's own
   `schema_version` stays 1"*), so contract #5's row above moves **v1 → v2**. The config
   files' own `schema_version:` key is a *file-format* pin and is unchanged at `1`; it is
   not this contract's version.
2. **The doc half is OWED and BLOCKED, and the owner is WP14.**
   `docs/contracts/run_config_schema.md` must gain the key and the bump. Phase S is barred
   from touching that file (rule R11: it carries another work package's uncommitted work —
   an unrecoverable loss if disturbed), so the same-commit clause above is **half-kept**: the
   version bumped here, the contract doc deferred to **WP14**. Stated outright rather than
   left to be discovered.
3. **run5-mint checklist item.** Before the next `configs/run5.yaml` mint, check that
   `docs/contracts/run_config_schema.md` carries `train.max_train_steps`. A stale contract doc
   *for the very config being minted* is what misleads at mint time, so the owed update is
   recorded where the minter will read it.
4. **Nothing enforces contract #5's doc.** MEASURED at this commit: `grep -rn
   "run_config_schema" .` over the whole repo returns **no hits at all outside this
   amendment** — no test, no tool, no Makefile target, no CI gate names the contract file
   (not even the file itself). Two consequences: the drift
   recorded above **cannot make any gate lie** (nothing reads the doc, so nothing can report
   green over it), and this clause was **already weaker than it reads** — kept by manual
   discipline alone. That missing handshake is its own defect, recorded for WP14 / WP-R
   rather than fixed here.

**Precedence, recorded so it is not re-derived.** R11 protects another WP's uncommitted work
(unrecoverable); this clause's doc half protects a document with no consumer (recoverable
staleness). The reading applied here is that **R11 yields nothing and the doc clause defers**.
That is a dispatcher reading pending operator ratification, not a settled rule.

### AMENDMENT — contract #5 v2 → v3, same shape, same deferred doc half

**WPAX Phase D (card CARD-DRAWRATE-KEY), R65 as re-scoped by R80.** Recorded here rather than
left as silent drift (R9); the S-4 amendment above is the precedent this follows verbatim.

1. **The bump.** `train.draw_rate_abort` is a NEW **required** field on `RunConfig` — a nested
   block (`threshold` `gt=0, le=1`, `min_step` `ge=1`, `N_pool_min` `ge=1`) or `null`, which
   is the EXPLICIT disarmed posture. Incompatible for the same
   reason S-4's was: a config lacking the key fails to load. Contract #5's row moves
   **v2 → v3**; the config files' own `schema_version:` file-format pin is unchanged at `1`.
   A second cross-field rule joins `RunConfig`'s validators: `train.draw_rate_abort.min_step`
   must be `< train.max_train_steps`, the twin of the actor-lag rule and the same defect class
   ("armed in the config, absent in effect").

   **WPMINT Phase DS amendment (operator ruling R92), same contract row.** The block's third
   key was `min_samples` (`ge=1, le=DRAW_RATE_WINDOW`) and is now `N_pool_min` (`ge=1`),
   because the gated STATISTIC changed: it is the pooled count-weighted rate
   `Σ draws / Σ completed` over the union of worker windows, and insufficient evidence
   (`Σ completed < N_pool_min`) is a NO OBSERVATION rather than a healthy `0.0`. Two
   validators replace the retired `le=` bound, one at each end of the same defect class:
   `RunConfig._draw_rate_evidence_bar_within_configured_capacity` (`N_pool_min <= DRAW_RATE_WINDOW *
   selfplay.n_workers` — a cross-SECTION rule, which is why it cannot be a field bound) and
   `DrawRateAbortConfig._one_drawn_game_cannot_fire_the_abort` (`1/N_pool_min < threshold`).
   The block stays required and `null` stays the explicit disarmed posture, so the contract
   row does not move again.
2. **The doc half is OWED and BLOCKED, and the owner is still WP14.** R11 bars this phase from
   `docs/contracts/run_config_schema.md` exactly as it barred Phase S, so the same-commit
   clause is half-kept again: version bumped here, contract doc deferred. The run5-mint
   checklist item above now covers **two** owed keys — `train.max_train_steps` and
   `train.draw_rate_abort`.
3. **Measured consequence, recorded because it moves an operator-facing number.** With run5
   armed at `min_step: 25000`, the mint preflight's minimum legal `--burst-steps` for
   `configs/run5.yaml` moves from **101 to 25001**: the burst override shortens
   `train.max_train_steps`, and every fire-floor in the config must stay inside the run. The
   tool now enumerates each binding rule with its own floor rather than reporting only the
   maximum. See IMPL_NOTES_D's STOP-2.
4. **The 101→25001 tension is resolved by DISCLOSURE, not by a shorter burst** (WPMINT Phase B
   / CARD-D-BURST-FLOOR). The floor cannot be shrunk — `min_step` is a run5 armed value and is
   mint-prereg-only (R82/R85) — and a shorter burst that pretended to cover the draw-rate axis
   is barred by R64. The preflight's evidence report therefore carries a `tier` block naming
   which mint tier the burst it ACCEPTED belongs to (`none` / `sync_lag` / `full`, derived from
   `_burst_floors`) and, in words, what that tier does NOT prove. **Both `sync_lag` and `full`
   are required for a mint, and `full` covers `sync_lag`** — one green `full` run discharges
   both. This deviates from the two-SEPARATE-RUNS shape the card presumed, on a measured
   ground: a `PRODUCTION_CONFIGS` row must arm `draw_rate_collapse` (gate 12 assertion (c)), an
   armed row puts `min_step + 1` in the floor set, and the override refuses anything below the
   max at rc 11 — so on a production config tier `sync_lag` is UNREACHABLE, and the only route
   to it is disarming the row the mint exists to arm. Measured, HEAD: run5's floor is 25001 and
   every other `configs/` entry's is 101, because only run5 arms the abort. Also measured: no
   burst of any length has ever run here — the boot child dies at TD-4 before `compose_run` —
   so `covered` is `[]` and both tiers stay OWED on every report the tool can currently write.
   The cost of the `full` tier is a published LOWER BOUND (`>= 1041.5 s` from WP10's 41.66
   ms/train-step floor) whose missing term — game-bound self-play generation for `>= 25001`
   completed games on one worker — is named rather than estimated.

### AMENDMENT — contract #5 v3 → v4: the step-coordinator knobs are CONFIG

**WPMINT Phase K-B (card `CARD-COORD-KNOBS`), R78 as clarified by R80, method bound by R93.**
Recorded here rather than left as silent drift (R9); the S-4 and Phase D amendments above are
the precedent this follows verbatim.

1. **The bump.** Twenty NEW **required** leaves on `RunConfig`, so contract #5's row moves
   **v3 → v4**; the config files' own `schema_version:` file-format pin is unchanged at `1`.
   Nineteen are flat `train.*` keys — `eval_interval`, `log_interval`, `buffer_save_interval`,
   `min_buf_size`, `replay_capacity`, `replay_capacity_schedule`, `training_steps_per_game`,
   `max_train_burst`, `batch_size`, `augment`, `recency_weight`, `mixing_initial_w`,
   `mixing_min_w`, `mixing_decay_steps`, `hard_gn_threshold`, `hard_gn_min_steps`,
   `terminal_eval_enabled`, `bot_batch_share`, `selfplay_stall_timeout_sec` — read by ONE
   resolver, `mantis.config.resolve.coordinator.resolve_coordinator_knobs`. (`buffer_save_
   interval` was DELETED at WP12-R by R178(a); see the v5 → v6 amendment below. This list is
   the historical record of what Phase K-B authored.) The twentieth is
   `train.draw_rate_abort.consec`, which joins the abort family's block because R80 says its
   terms travel together. Incompatible for the same reason S-4's and Phase D's were: a config
   lacking any of them fails to load. `mantis.run._step_coordinator_config` now holds **zero
   literals** — every `StepCoordinatorConfig` field arrives from a resolver.

   **They are FLAT `train.*` keys, not a `train.coordinator` block, and that is a decision.**
   `train.step_coordinator.*` was RULED AGAINST at Phase D (§2, pinned verbatim by
   `tests/config/test_drawrate_arming_authority.py`) on the ground that a config block named
   after a dataclass is named after the wrong thing; these are training hyperparameters, and
   the coordinator is the object that reads them, not the fact they express. An OPTIONAL block
   was independently barred: `_leaf_paths` counts a `Block | None` as one leaf, so ~20 keys
   would have bought a cheap registry count by hiding from the LAW-08 bijection that justifies
   them.

   **Three names differ from their runtime fields, each for a measured reason** — two, since
   the v5 → v6 amendment below deleted the first of them with its key.
   `buffer_save_interval` -> `checkpoint_interval` (the coordinator's is the REPLAY-BUFFER
   save cadence; `train.checkpoint_interval` is the already-authored TRAINER cadence, and two
   config keys with one spelling is the duplicated-authority class R1 exists to kill);
   `replay_capacity` -> `capacity` and `replay_capacity_schedule` -> `buffer_schedule` (a bare
   `train.capacity` names nothing on its own).

2. **SIX coordinator fields are DELETED, not authored** (adjudication call K-a):
   `composition_interval`, `value_probe_interval`, `soft_ew_threshold`, `soft_ew_min_pts`,
   `instrumentation_enabled`, `bot_corpus_path` had no reader anywhere in `src/`, re-verified
   at HEAD by grep AND by recording every attribute read on a live `StepCoordinatorConfig`
   across the whole test tier. R1 requires a live consumer per key, so typing them in would
   have CREATED the violation the card meant to close.

3. **One value moves, and it is a correction, not a change.** `train.batch_size` is minted at
   **256**, not the dead field's `8`: WPMINT Phase K-A measured that
   `coordinator/step.py::_run_training_step` read
   `train_cfg.get("batch_size", full_config.get("batch_size", 256))`, that both lookups miss on
   the production path, and that the run's real batch size was therefore the literal `256`.
   Every other authored value is the value the code already used, proven by rebuilding
   `StepCoordinatorConfig` from all five minted configs at the parent commit and comparing
   field for field.

4. **`grad_norm_hard_abort` joins the armed-abort manifest as a DEFERRED row**
   (adjudication call K-c), with a new `Mechanism.CONFIG_THRESHOLD_BELOW_CEILING` and a
   `ceiling_path` naming `monitor.alert_grad_norm_max`. DEFERRED because a REQUIRED row would
   gate run5's mint on a grad-norm threshold nobody pre-registered — the class R84 refused.
   The mechanism is upper-bounded because `CONFIG_THRESHOLD_GT_ZERO` reads the shipped `1e9`
   as ARMED, which is "armed in the config, absent in effect".

5. **R78's first design question is ANSWERED YES and implemented.** The mint preflight's
   evidence JSON now carries a `coordinator` block — the resolved knobs, drain caps,
   `stop_step` and draw-rate terms, DERIVED from the shipped resolvers and never restated —
   in both AUDIT and PREFLIGHT mode. It witnesses the config -> resolver -> builder seam and
   explicitly NOT a child that was handed something else.

6. **The doc half is OWED and BLOCKED, and the owner is still WP14 / Phase W.** R11 bars this
   phase from `docs/contracts/run_config_schema.md` exactly as it barred Phase S and Phase D,
   so the same-commit clause is half-kept a third time: version bumped here, contract doc
   deferred. The run5-mint checklist item now covers **three** owed entries —
   `train.max_train_steps`, `train.draw_rate_abort` (including `consec`) and the nineteen
   `train.*` coordinator knobs (eighteen since the v5 → v6 amendment below).

### AMENDMENT — the deferred doc half is DISCHARGED; contract #10 is added

**WPMINT Phase W (WP14), operator rulings R66 and R91.** The three amendments above each
half-kept the same-commit clause and deferred the doc to WP14. This is that commit.

1. **`docs/contracts/run_config_schema.md` is committed and current at v4.** It carries the
   version history v1 -> v4, the run-length key, the draw-rate block including `consec` and the
   `min_samples` -> `N_pool_min` swap, the nineteen coordinator knobs with their three
   schema-vs-runtime renames (eighteen and two since the v5 → v6 amendment below), the
   `ReplayCapacityStage` sub-block, and every cross-field
   validator by name. The run5-mint checklist item raised in the v2 amendment is **discharged**:
   all three owed entries are in the doc.
2. **The curation was TRUTH-CHECKED, not merely extended.** Every claim inherited from the WP8
   working copy was grepped against the tree and a claim that could not be verified was removed
   or corrected rather than softened. Five were FALSE at HEAD: the status line's "checkpoints
   land in WP10" (WP10 landed at `b29f0bc`), an assertion row calling the legal-move radius
   schedule a REQUIRED field (the field and its resolver module are RETIRED), the resolver list
   naming `radius`, an emit payload described as 9 knobs (it is 8), and a regime-parity row
   whose radius arm now asserts the field's ABSENCE. The corrections are in the doc; the
   evidence is in `wp/WPMINT/IMPL_NOTES_W.md`.
3. **R11 is discharged and it never had repo-side enforcement.** Re-measured at this commit:
   `git grep R11` over `tools/ src/ tests/ Makefile .github/ docs/` returns only this file's
   own prose (plus one unrelated `§f-R11` fixture id). The rule lived entirely in the migration
   workspace, so "every gate asserting the exception drops it" was a no-op in the tree — the
   deferral was kept by discipline alone, exactly as the v2 amendment's item 4 predicted. That
   item's measured claim (*"`grep -rn "run_config_schema" .` returns no hits outside this
   amendment"*) is now **superseded**: item 4 below is its remedy.
4. **CI gate 13 closes the missing handshake** (R91's design question, answered by BUILDING).
   `tools/ci_gates/contract_doc_gate.py` refuses a contract doc that cites a config key or a
   `mantis.*` symbol the shipped schema does not have, or that states a leaf-key-path count
   other than the live one. It follows the gate-12 pattern strictly: every check is answered by
   importing `RunConfig` and the module tree, so a schema change ALONE reds it — pinned by a
   mutation that adds a leaf to `RunConfig` and leaves the doc untouched. The "deliberately
   absent" section is checked in REVERSE rather than exempted, so a retired key returning is
   also a red. Its one transcription — the leaf walker — is self-defending: the two
   consumer-registry copies assert `170` against the same schema, so a walker that diverged
   here would disagree with the doc and red this gate.
5. **`docs/contracts/checkpoint_envelope.md` is filled** (R11's second named item): status
   SKELETON -> LIVE, with the v2 filename/payload/metadata shape, the eleven assertions and
   their pinning tests. The subsystem landed at WP10; only the contract half was outstanding.
6. **Contract #10 is ADDED — the mint preflight evidence report** (Phase B's F-B3). The
   artifact has carried the version string `preflight-mint-v1` since WPAX and Phase B added a
   top-level `tier` field to it, while nothing under `docs/contracts/` described it. A
   versioned artifact with no contract is a version string that means nothing, so it is
   written rather than deferred a fourth time. It documents the shape, the always-written rule,
   the derived-from-what-the-run-DID discipline and the measured F-B1 gap.
7. **What this amendment does NOT close.** Gate 13 is a CITATION check: a claim the doc omits
   entirely is invisible to it, and it does not read prose for truth. Stated here rather than
   left for a reader to discover, because a gate whose real reach is narrower than its name is
   the class §9 keeps closing.

### AMENDMENT — contract #5 v5 → v6: `train.buffer_save_interval` is DELETED (a knob count goes DOWN)

**WP12-R, operator ruling R178(a), assigned for execution by R183(a); grounds R116 + LAW-08 +
R1.** Recorded here rather than left as silent drift (R9). This is the FIRST amendment in which
a required leaf is REMOVED, and the shape is worth naming because every prior one added.

   **A gap in this log, recorded not laundered:** `docs/contracts/run_config_schema.md` is at
   **v5** (WPMAIN's five promotions — `eval_enabled`, the `monitor.disk_guard` family,
   `train.device`), and no v4 → v5 amendment was written here; item 1 of the amendment above
   still says the doc is "current at v4". That bump belongs to WPMAIN and is not this
   amendment's to author, so it is flagged rather than back-filled. This amendment is
   therefore **v5 → v6**, counted from the contract doc's own version table.

1. **The deletion.** `train.buffer_save_interval` is gone. It was one of the nineteen flat
   `train.*` step-coordinator keys the v3 → v4 amendment above authored, and it was the only
   one whose consumer chain ended nowhere: it fed `StepCoordinatorConfig.checkpoint_interval`,
   whose sole reader was `coordinator/step.py`'s D4 `_try_save_buffer` arm, and WP12-R Phase CS
   MEASURED (F-CS-2) that `buffer_persist.try_save_buffer` returns immediately unless
   `mixing_cfg["buffer_persist"]` is truthy while the production composition root passes
   `mixing_cfg={}` and nothing in `src/` ever sets that key. A key minted into `run5.yaml` with
   zero reachable effect is the dead-knob class R1 exists to kill, and R178(a) refused to ship
   it into the mint record. **`RunConfig`'s leaf count moves 175 → 174** (measured, not
   transcribed — item 4 of the amendment above states `170`, which was already stale).
2. **What went with it.** The rename seam retires whole: `CoordinatorKnobsSpec.checkpoint_
   interval`, `StepCoordinatorConfig.checkpoint_interval` and both no-op `_try_save_buffer`
   call sites (D4's cadence arm and O3's shutdown-signal arm). **Three schema-vs-runtime
   renames become two** — the surviving pair is `replay_capacity` -> `capacity` and
   `replay_capacity_schedule` -> `buffer_schedule`. The v3 → v4 amendment's third rename
   existed only because `train.checkpoint_interval` (the TRAINER's periodic save, which is
   untouched and still live) would otherwise have collided with a same-named coordinator key;
   with the coordinator field deleted there is no collision left to disambiguate. The
   nineteen coordinator knobs are now **eighteen**; the enumerations above are left as the
   historical record of what Phase K-B authored, which is what an amendment log is for.
3. **`train/buffer_persist.py` SURVIVES, deliberately.** Only the arms are deleted, not the
   helper, and that is R178(c)'s doing: buffer persistence returns, if at all, as ONE design
   under `CARD-RESUME` (post-mint) covering weights + optimizer/scheduler + buffer + launcher
   together. Half-resumes are the trap the WP had just finished killing, so nobody builds a
   piece of it separately. Consequence to state plainly rather than discover later: with both
   arms gone, `buffer_persist.buffer_save_errors_total` — published in the LAW-18
   `monitor_gates` payload — has no production producer. It had none before either (both arms
   returned early on every production leg), so the deletion changes reachability by nothing;
   the gauge is now visibly, rather than invisibly, pinned at zero until CARD-RESUME lands.
4. **Zero production behaviour moves.** Every committed config minted the interval at `0`, so
   the D4 arm's own gate (`cfg.checkpoint_interval > 0`) never opened even before the helper's
   early return, and the O3 arm's call was a no-op by the same measurement. The re-mint is a
   pure one-line deletion in each of the six configs, verified by replaying each config's own
   header-recorded template + deltas through `tools/mint_config.py` and diffing: header lines
   byte-identical, no other leaf moved.
5. **The one instrument this forced open, and how.** `tests/config/test_minted_config_remint.py`
   required the live configs to be INSERT-ONLY against the frozen `b482243` baseline. The
   baseline was NOT re-cut — `tests/fixtures/manifest.toml` is a frozen oracle and the
   directory name is a commit pin, and a re-cut baseline would make every assertion in that
   file vacuous. Instead the instrument learned ONE named deletion: closed one-element
   `_REMOVED_LEAVES` / `_REMOVED_LINES` sets, set EQUALITY on both halves, `replace` still
   forbidden outright (narrowed once since, at R187 — item 6), plus a guard test asserting the
   allowance cannot widen without a ruling. Every mutation class the pre-R178 version caught
   still reds — value drift, armed-value drift, a dropped header line, a second deleted leaf, a
   reformat, a reorder, a rewritten comment, and a same-shape deletion of a different line.
6. **The minted header is now REPLAYABLE, and two configs were re-minted for it (R187).**
   `tools/mint_config.py` stamped its `# delta:` values with Python `str()`, so a `None` came
   out as the six characters `None` and read back as the STRING `"None"` — measured on
   `configs/smoke_preflight_armed.yaml`'s `eval.ladder.rungs` delta, whose replay through its
   own minter failed schema validation. R1's "minted, never hand-varied" is only checkable
   because a minted config's header replays to the config, so the defect sits under the rule.
   The renderer is now `yaml.safe_dump` — the inverse of the `yaml.safe_load` that parses
   `--set`, which is what makes the header's value domain and the tool's own parser the same
   language — with a per-value round-trip CHECK and a loud rc-2 refusal for the two values in
   that domain it cannot invert (`!!omap`, `!!pairs`). `configs/run5.yaml` and
   `configs/smoke_preflight_armed.yaml` were re-minted mechanically: bodies byte-identical,
   five header lines re-rendered, run5's armed `0.25 / 25000 / 50` untouched (R119) and proven
   so by the structural half of the instrument above staying green through the re-mint. That
   re-mint is the `replace` the instrument had to learn: a closed five-element
   `_REHEADERED_DELTAS`, and a tolerance that PROVES each change is rendering-only by
   reproducing the baseline slot from the live slot through the old `str()`. New producer:
   `tests/config/test_mint_header_roundtrip.py` (LAW-07).

### AMENDMENT — contract #5 v7 → v8: `monitor.gate_interval` is ADDED (arming cadence leaves narration cadence)

**Pre-mint correctness bundle, operator ruling R242 (ADJ-D12); grounds R1 + LAW-18 + the F-43
class.** Recorded here rather than left as silent drift (R9).

   **A gap in this log, recorded not laundered — same shape as the v5 → v6 block above.**
   `docs/contracts/run_config_schema.md` reached **v7** under WP12-R (R179) and no v6 → v7
   amendment was written here. That bump belongs to WP12-R and is not this amendment's to
   author, so it is flagged rather than back-filled. This amendment is therefore **v7 → v8**,
   counted from the contract doc's own version table. Item 5 of the table in §4 above still
   reads `v4` and is corrected to `v8` by this amendment — it has been stale since v5.

1. **The addition.** `monitor.gate_interval` (`int`, `ge=1`, **no default**) is a new required
   leaf. **`RunConfig`'s leaf count moves 176 → 177** (derived from `_leaf_paths(RunConfig)`,
   not transcribed; gate 13 asserts the contract doc's stated count against the live one, so
   this number cannot rot silently).
2. **The defect it closes, measured.** Everything downstream of `_run_log_interval`'s
   `self._train_step % cfg.log_interval != 0` guard was `train.log_interval`-gated —
   including the LIVE-producer `draw_rate_collapse` hard abort and the LAW-18 `monitor_gates`
   summary. At run5's minted `log_interval: 1000` **no draw-rate abort could fire and no
   `monitor_gates` event existed before training step 1000**: armed machinery with a blind
   first kilometre, which is the F-43 "armed in the config, absent in effect" class landing on
   the abort path itself. R242's words: "it is worse than the original item".
3. **Why a key and not a constant.** `consec` counts consecutive OBSERVATIONS, and an
   observation is ATTEMPTED once per gate boundary — a boundary whose producer is absent or
   whose evidence is below `N_pool_min` is SKIP-counted and appends nothing, so it neither
   advances nor resets `consec`. The sampling stride is therefore the gate's time constant
   only as a LOWER BOUND on elapsed steps, never an equality. That is precisely why the
   cadence needed to be a settable fact: running the gates per step without a knob would have
   re-scaled every armed abort's window by `log_interval`, and run5's armed
   `train.draw_rate_abort.consec: 3` would silently have meant 3 consecutive STEPS rather than
   3 consecutive gate boundaries — a 1000× change to a pre-registered armed value. Armed
   values are operator-only, so the cadence became explicit rather than a side effect of a
   logging knob.

      **Correction of record (R96 — corrected in the artifact, not only in the log).** The
      first draft of this item stated "an observation is appended per gate run" and said the
      re-scaling would have meant "50 STEPS instead of 50 boundaries". Both were wrong and
      both were caught by the RED-TEAM pass on this commit: the append claim is falsified by a
      measured drive (an observation, an 8-boundary evidence blackout, then two observations →
      the abort fires with `consec` satisfied ACROSS the gap), and the `50` was
      `N_pool_min` — a completed-game evidence bar — not `consec`, which is `3`. The same
      false append-claim was found in four places in `coordinator/step.py` and one in
      `schema/train.py`; all are corrected, and `monitor/rules.py::check_draw_rate_collapse`
      had it right all along.
4. **Zero production behaviour moves, and that is enforced, not asserted.** Every one of the
   six configs mints `gate_interval` EQUAL to its own `train.log_interval` (1000 ×5; 10 for
   `smoke_preflight_armed`), so the shipped cadence, every `consec` meaning and every armed
   value are unchanged in effect. The re-scaled values the operator actually wants are prereg
   rows, not code. `tests/train/test_gate_interval_decoupling.py::test_p6b` pins the equality
   across all six and names itself as the row the prereg ruling RE-POINTS rather than deletes.
   Verified by replaying each config's own header-recorded template + deltas through
   `tools/mint_config.py` and diffing: byte-identical on all six, pure insertion, no other
   leaf moved.
5. **What `train.log_interval` becomes.** Narration only — the `training_step` payload, the 4
   WARN rules, the axis distribution. Its `ge=1` survives on a NARROWER ground: WPMINT DR-7's
   measurement (that `log_interval <= 0` killed the hard-abort family and the instrument that
   would have shown it) moved WITH the gates to `monitor.gate_interval`, and what remains here
   is that there is no legitimate "never narrate" posture. Both the schema field comment and
   `resolve/coordinator.py` are re-pointed accordingly; leaving them would have told an
   operator choosing `log_interval` that it still arms the aborts.
6. **Scope boundary held.** R210's "training_step alerting stays gated" clause is superseded
   IN SCOPE by R242 — R210 governed games-visibility narration and its clause is not
   arming-cadence law — while `iteration_complete`'s decoupling (R210's actual subject) is
   untouched and still pinned. The `grad_norm_hard_abort` row and every `source_pin` in
   `src/mantis/config/armed_aborts.py` are UNTOUCHED: that abort stays disarmed at its
   unauthored `1e9` per R56/R243, and arming it is a separate prereg row.
7. **Carrier route, and why it is not a new shape.** Schema `monitor.gate_interval` → a THIRD
   enumerated drop in `resolve_monitor_config` → `compose_run` → `StepCoordinatorConfig.
   gate_interval`. That is exactly the route the four `monitor.drain.*` keys already take
   (monitor-schema-scoped, resolver-dropped by name, composition-root-threaded), so no new
   kind of edge is introduced. The drop is enumerated one name per line, never a filter —
   `tests/config/test_disk_guard_keys.py` pins that, for the DR-11 reason it always did.
