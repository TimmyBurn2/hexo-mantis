# PROGRESS — SLIM-FIX (R368), the durable resume point

Resume from THIS file after any stop, never from memory. Updated at every leg exit.

## Where the run is

- Branch: `claude/slim-fix-r368`, cut from `origin/claude/slim-scout-census-v3i2hj` @ 1e8d6d6 (base dev 69e1532).
- Host: the operator's desktop (not a Claude environment; `CLAUDE_CODE_ENVIRONMENT_NAME` unset). AMD Ryzen 7
  3700X, 16 threads, 46 GiB, flags `avx2` only (no `avx512_bf16`, no `amx`). torch 2.11.0+cpu, `mantis._engine` OK.
- origin/dev = 69e1532 at W0 entry (no moved commits; no rebase).
- Wave: **W7** (the style pass) IN PROGRESS (see `## W7`) — work order `docs/slim/handoff/W7_ADDENDUM.md`. W6 EXITED (see `## W6`). W5 EXITED (see `## W5`). W4 EXITED: W3 EXITED. REVIEW-W3 filed (`docs/audits/REVIEW_W3_2026-09-24.md`, 0 must-fix beyond a
  ruff red in the handoff helper, 2 should-fix, 4 notes, every row group HELD, run10 MATCH re-verified); its fixes:
  ec0ac144 (ruff), cb4a5dd1 (`_is_graph` flag), bf22b4c3 (sweep docstring), plus 09fa8e6b — the exit sweep caught
  `tests/eval/test_graph_round_encoding.py`'s recorder pinning the deleted `_is_graph`; re-pointed to the
  constructor's own closed predicate (the finding-2 fix's miss; the sweep is what caught it). Exit sweep `make
  gates` in `.wt/gates` at aad7eb6d: 18 of 19 green, gate 3a RED on exactly that witness plus one TRANSIENT
  (`test_supervisor_signal_posture`'s subprocess SIGINT assert — the dispatcher ran two worktree `uv sync`s
  concurrently with 3a; investigated to ground below), 3a re-run at the fixed tip 09fa8e6b: only that one red
  remains. 3b 34 passed/4 skipped in 3050 s; 3c collected 5041;
  14 GREEN pyright 247 files 0 errors; 8 ARMED+PASS; 12 rc 0; 15 166 justified 0 stale. W1 EXITED earlier
  (cadcc367), W2 EXITED (3db6ab5a).
  **The KNOWN-RED (recorded, not fixed — the loop's step-5 rule):** `tests/monitor/test_supervisor_signal_posture.py
  ::test_the_stop_handlers_are_installed_by_main_and_never_at_import` reds in FULL default-tier runs on this host
  (4/4: the sweep at aad7eb6d, two re-runs at 09fa8e6b, one at the W3 BASE 8b75f984 with a fresh venv — so NOT
  the wave's; W2's sweep at 3db6ab5a was green earlier the same day) and passes standalone (6+/6 in .wt/gates
  and the main checkout) and in every partial batch tried (tests/monitor whole; tests/util+victim;
  arena..eval+victim; fixtures+model+victim). Mechanism, narrowed by instrumented runs: the pytest process's
  SIGINT disposition is SIG_IGN at the moment the test spawns its import-probe child, and CPython preserves an
  inherited SIG_IGN across exec (demonstrated directly), so the child's getsignal assert fails; the flip is NOT
  made through Python's signal module (a sitecustomize wrapper on signal.signal fires only for pytest's own
  startup install); a polling thread observed in-process flips between train/lifecycle's
  `install_signal_handlers._stop` handler and SIG_IGN around root/train tests, but GIL starvation makes that
  timeline unreliable and the exact C-level, timing-dependent setter is unidentified. Every later wave's exit
  sweep and W8's gates.exit will show this one red until the operator rules on it.
  W4 EXITED. 10 implementation commits 47dd9a05..51eddda5 (net −410): the sealbot vendor side
  (60755d15, R368(e), eval_instrument v3, meta_ci target set updated); TOOLS-2-07 (71903a8d, the seven
  by-path loaders → mantis.util.loadpkg); TOOLS-2-10 (44678400, run_shard_paths the one shard
  enumeration + producer test); TOOLS-2-08 (a9528a77, AMENDED-with-ground: the constants read the
  engine exports and are pinned value-and-order, the win-line scan STAYS LOCAL — the engine's
  find_winning_line has a sorted-stones fallback that would mask the completing-stone contract
  (LAW-03), confirmed independently by REVIEW-W4 in the Rust); TOOLS-2-09 (e87e3686, board.js the
  one renderer, parity-harnessed); TOOLS-2-15 (6adbeebf); TOOLS-1-01 (622893b6); gate 14's two new
  gated measures (5fc33aa1, defect 23: ruling_cite_lines 1210 = comments+docstrings+rust-docs+
  text-format cites with bare R10-and-up, textfile_comment_excess_lines 466, both self-tested and
  producer-tested, the old ungated counter retired, CLAUDE.md names them); gate 16 tree-wide zero
  (51eddda5, 203 sites fixed with explicit utf-8, os.open skipped by mechanism).
  REVIEW-W4 filed (docs/audits/REVIEW_W4_2026-09-24.md, e6215240): 0 must-fix, 0 should-fix, 5 notes
  (all recorded: the TOOLS-2-08 amendment above; one reclaimable cite line in the gate's own
  docstring — W7/W8; CARD-STYLE-BACKLOG's stale measure figure — W6; a brief slip; PROGRESS
  staleness — this entry). The ci_gates surface was exactly the five authorized files. Collected
  5035 = floor; all seven comment measures at floor; run10 MATCH. The known-red passed REVIEW-W4's own full default tier and one of
  the dispatcher's — flaky as recorded; the operator's row stands. W5's addendum is committed
  (docs/slim/handoff/W5_ADDENDUM.md, b0e959c3). Exit sweep `make gates` at e6215240 in .wt/gates:
  ALL GREEN except 3a's single failure = the KNOWN-RED flake (it passed REVIEW-W4's own full tier
  half an hour earlier at the same commit; 2a 1145 s; 3b 34 passed/4 skipped in 3397 s; 3c
  collected 5035 = floor; 7/8/9/11/12/13/14/15/16/17 green; pyright 248 files 0 errors); the
  post-review docs commits (b0e959c3 + this one) re-checked with gates 10/13/14/15/17 green.
  NEXT: W5 per its addendum (the 2026-09-24 dispatcher-session scout inventory is folded into it);
  then W6, W7, W8 per HANDOFF.md.

## W7 — the style pass (IN PROGRESS)

**Dispatch log** (models per leg named in each row):

- Entry at d18dc5bc: collected 4928 = floor; comment_lint GREEN at floor (cite 1163, comment_excess 3124, textfile
  466); run10 MATCH; tree clean. Per-package measures re-derived with the addendum's method: identical to its table.
- Every leg commit is checked by a scratch verifier (Python: AST equal with docstrings stripped; Rust: comment-stripped
  token text equal; text formats: own-line `#` lines dropped) before integration.
- Launched in parallel, each in `.wt/w7-<leg>` from d18dc5bc: L1 crates/mantis-selfplay (**opus**), L2 crates/mantis-search
  + crates/mantis-encoding (**opus**), L3 tools/ci_gates (**opus**), L4 crates/mantis-core + -bridge + -graph (**sonnet**).
- L1 LANDED 2ce26192, 39fa4868 (**opus**; one package split src | tests+benches): selfplay cite 78→3, comment_excess
  871→701 (−260 lines net); rust_doc_excess fell 2486→2320 as a side effect. Kept: CARD-RING-SAMPLER-SEED, the
  atomic.rs planted-break cite, queue_fuse_pin's LAW-07 self-test marker, the `rotation-free` text, R8 headers.
  rustfmt's code reflow in 5 base-unformatted test files was reverted (comment edits only). Verifier comment-only.
- L2 LANDED f0766683 (search), 6282b66f (encoding) (**opus**): search cite 9→0, comment_excess 559→446; encoding
  13→0, 105→72. registry.toml untouched, registry gate ARMED+PASS; selection.rs's root comment corrected (it
  claimed every node selects by PUCT, false under Gumbel). Verifier comment-only; run10 MATCH; gates 10/12/13/15/17 rc 0.
- L5 tests/train+config+tools+selfplay (**sonnet**) and L6 the remaining tests/ packages (**sonnet**; tests/model/
  conformance/** excluded under R368(h)) launched in the freed slots.
- L3 LANDED 463bba16 (**opus**): tools/ci_gates cite 67→9, comment_excess 92→9, textfile 252→38 (+184/−483, 17
  files). Kept: the two registers (comment_length_floor.txt, tier_declaration.txt), CARD-LINT-GATE, a LAW-07 token
  inside registry_gate.sh's Python heredoc (data), CARD-POOL-ENCODING-BRIDGE in preflight_mint.py's docstring (a
  test asserts it in `__doc__`), every cite inside a string/refusal text. REVIEW-W4 N2 DONE (comment_lint's docstring
  says "two-digit-and-up"). CARD-STAMP-FLOOR's docstring cite dropped; CARDS' row re-pointed to the file that still
  cites it (preflight_mint_parent.py, a string). Every touched gate's self-test green; 786 tools/preflight tests green.
- L4 LANDED f03b4d08 (core), 3c6bf3af (bridge), e6b56e5d (graph) (**sonnet**): cite 5→0 each; comment_excess core
  207→198, bridge 189→189 (its excess is `///` API docs and R8 headers — docstring length is on contact), graph
  112→71. SAFETY blocks and the `!Sync` invariant kept; rustfmt skipped on base-unformatted files; check.wasm green.
- L7 src train+config+eval+root (**opus**) and L8 the remaining src packages, tools/ root + dashboard/ladder/probe1/
  viewer and the Makefile (**sonnet**) launched.

## W6 — docs + configs (EXITED 2026-09-25)

**Exit facts.**
- Range 4e8c663a..0c8b9176 (plus the record commits after it): 61 commits at exit record, 80 files, +1 408/−9 785.
  src, crates and tools: 5 files, +8/−7, all prose. The armed_aborts.py MANIFEST text is back at its base bytes.
- Collected 4927 at W6 entry → **4928** = floor. Every comment measure is unchanged at its floor.
- configs/ at exit: dev_example, run10, run6 (HALT), smoke_preflight_armed. run10, dev_example, smoke and run6 are
  byte-identical to 69e15329. **run10 MATCH** at the tip.
- REVIEW-W6 (`docs/audits/REVIEW_W6_2026-09-25.md`): 1 MUST-FIX, 7 SHOULD-FIX, 10 NOTE. All eight were fixed in ONE
  loop; the reviewer's closure check found them CLOSED with 0 new must/should, and its 3 notes were fixed.
- Exit sweep `make gates` at 0c8b9176 in .wt/gates, user unit mantis-gates-w6 (01:32–02:37 UTC): **ALL GREEN,
  19 green**. Walls: 2a 1056 s, 3a 336 s (4868 passed, 17 skipped, 0 failed), 3b 2492 s (34 passed, 4 skipped).
  pyright 0 errors; 8 ARMED+PASS; 12 rc 0. Gate 17 ran at the tracked floor in the worktree; the main tree's
  `--full-tree` run with the local supplement is rc 0. Gate 1 and the slow tier were not run, by design (W8).
- Operator asks: see the OPERATOR ASKS row below. NEXT: W7 per `docs/slim/handoff/W7_ADDENDUM.md` (8e5eab3e).

**Dispatch log** (models per leg are named in each row):

- Entry at 4e8c663a: collected 4927 = floor, comment_lint GREEN at floor, run10 MATCH, tree clean.
- Legs launched in parallel, each in its own worktree (branch = worktree name):
  - `.wt/w6-configs`, **opus**: addendum legs 1+2 (by-name readers onto the census, run6–8 deleted).
  - `.wt/w6-cards`, **sonnet**: legs 3+4 (RULINGS_ACTIVE.md retired, CARDS closed rows and drift).
  - `.wt/w6-mech`, **sonnet**: leg 5 minus the register annotations (CLAUDE.md, toolchain files,
    repo_design §3, Cargo profile, SHAKEDOWN7G, .gitattributes, analyzer_design, README).
    LANDED 8db24400..6db4f09a (9 commits): CLAUDE.md (R368(g)'s one rule, floors count derived,
    digest -> LAWS.md, viewer + analyzer admitted), mise/rust-toolchain (count gone, pin ->
    `[provenance].rustc`), repo_design §3 R9 amendment (defect 43), `[profile.profiling]` deleted
    (DOCS-4-11, no builder), SHAKEDOWN7G deleted (DOCS-3-01), .gitattributes (DOCS-4-04/-05),
    analyzer_design Status (DOCS-3-06), README STATUS (DOCS-4-09), MSRV list (DOCS-4-10). Makefile
    floor count REFUTED at contact (no such text). Gates 10/13/15/17 green; measures unchanged.
  - `.wt/w6-reg`, **opus**: the falsified.md F-43/F-04 and RULINGS.md defect-47 annotations.
    LANDED 4f9cd33c (F-43 by symbol), 174a9180 (F-04: the min pin left at 3dd20b49), b1e91385 (defect 47:
    the coverage pair one high, "Four" names five). Append-only, +36/-0; gates 10/13/17 green.
- RESUMED 2026-09-25 by a second W6 dispatcher (the first, plus the configs and cards
  implementers, stopped on a usage limit). Both cut-off diffs were read and KEPT for finishing:
  - `.wt/w6-configs` (**opus**): 60cabe25 + 19 uncommitted tests/config re-points, sound in
    direction; the implementer verifies, commits, and re-points the other test dirs.
  - `.wt/w6-configs-b` (**opus**, new, from 98fb255a): tests/tools + tests/diagnostics re-points.
  - `.wt/w6-cards` (**sonnet**): the archive delete + F-816-34/35/36 carry kept; leg 4 to finish.
  - The run7/run8 delete follows once both configs halves land, with gate 10 `DISSOLVED_PATHS` rows.
- configs-b LANDED c7e5cbba, 9650fd10, e97609be, 92156f4e (tests/tools + tests/diagnostics onto the
  census; collected 4969 while run6–8 exist, since the census rows parametrize). Main-checkout run of the
  touched files: 532 passed, 1 CUDA skip.
- **HALT, configs/run6.yaml STAYS.** tests/tools/test_preflight_mint_process.py is R310-frozen (00_MAP
  §5), and its `RUN5` binds run6. A path-only re-point to run10 reds all 4 real-boot integration rows (rc
  33): run10's `train.heldout_gap.ring` names an untracked checkpoint ring, which run6 lacks. The measured
  fix, applied and driven 6/6 green, edits the oracle:
  - `train.heldout_gap{,.ring,.ring_sha256,.batches,.seed,.interval}` join `FORCED_TWIN_LEAVES`;
  - the twin drops the `train.heldout_gap=` header delta;
  - `differing == FORCED_TWIN_LEAVES` becomes `differing <= FORCED_TWIN_LEAVES and "train.device" in differing`.

  That needs a grant (R43/R310). The diff NARROWS the frozen oracle (REVIEW-W6 #7): the equality becomes a
  subset check, so a twin that silently failed to replace a forced leaf would pass. A non-narrowing shape keeps
  `==` by deriving the forced set from the base config (the heldout_gap leaves only when the base carries the
  block); the grant should name which shape. Operator: grant it, then delete run6.yaml. The implementer's
  oracle-body commit ad29e624 was DROPPED, and so was 3334eda1's test_preflight_mint.py half (also
  R310-frozen). With run6 kept, neither is needed.
- cards LANDED 33e32a16..5edc954a (9 commits, **sonnet**): RULINGS_ACTIVE.md deleted after its F-816-34/35/36
  text was carried into CARDS and the remaining `A:` coordinate cites were dropped (each paragraph names its
  ruling). CARDS 1028 -> 730 lines: the closed, landed and spent cards are removed, and half-open ones are
  condensed to their residue. DOCS-1-04: the markers section was re-derived, 11 real markers. DOCS-1-05:
  OC7 DISCHARGED, MAX_CHILDREN 1024, the pyright cite. REVIEW-W4 N3: measures named from the floor file.
  Defect 48 (MINPIN) done. Defect 51 plus W5's residue: CARD-MECHANISM-SWEEP members and CARD-W5-RESIDUE.
  Gates 10/13/17 green.
  - Kept, with grounds: CARD-GUMBEL-HEAD-RESIDUE (still carded), CARD-WARMSTART-CONTROL (unread),
    CARD-E1-RULER-R6 (self-retaining).
- Queued: leg 6, the STATE rewrite (**opus**), then
  REVIEW-W6 (**opus**).
- RESUMED 2026-09-25 by a third W6 dispatcher (the second hit a forced hand-back). The HALT's measured
  oracle diff is filed at `docs/slim/handoff/halt_twin_heldout.diff` for the operator (NOT applied).
  `.wt/w6-configs` found with a pytest absent-configs check still running (run6-8 deleted uncommitted);
  waiting for it to go quiet before integrating 60cabe25..2aa476f4.
- configs A LANDED 16e1a912..1a651436 (8 commits, **opus**; implementer SHAs 60cabe25..2aa476f4): src/tools prose
  generic (armed_aborts owner text, later REVERTED by e57e52d9; pipeline comment, select_balanced_book, strength_frontier, probe1 `--run-id`
  required); tests/config, eval, train, selfplay, run, arena, encoding readers onto the census (sweeps on
  `discovered_config_paths`, production-only laws on `production_configs`); run-named symbols renamed
  (RUN5/_RUN5/_PRODUCTION, test_run3_parity_values_pinned -> test_the_gate_parity_values_are_pinned, the PZ-1
  drawrate RUN5_PREREG -> DRAW_RATE_PREREG with values unchanged). PZ-1 extensions: the deploy-matched coincidence
  gained a census row (plant c_puct += 0.5 reds it), f32 launch pin census-parametrized, q_rescale re-expressed as
  a key flip on one census config (never-set-key plant reds three rows). Pin REMOVED with grounds: run6's
  `screen_confirm_lo == 0.44` (no ruling names it; minted values are
  provenance; run7's re-mint had already moved it to 0.5). The implementer's "inherited by-name run10 pin"
  removal does not appear in the diff (REVIEW-W6 #9): the one run10 by-name reader was re-pointed. Absent-configs check (run6-8 moved out): 498 passed, 1 failed = the HALT row below only.
  Main-checkout run of every W6-touched test file: 915 passed, 4 skipped. Collected 5086 with run6-8 present.
  - **HALT 2 (same operator grant):** tests/config/test_eval_config_remint.py::test_the_ruled_deploy_sims_are_pinned
    binds `deploy_sims == 160` (R346(b)/R348(e)) to configs/run6.yaml by path; no census member carries 160.
- STATE rewrite LANDED 14f86534, 57279306 (**opus**, implementer 37b25df3/14173647): 726 -> 90 lines, the
  "dropped class -> where it lives" table; landed BEFORE the config delete so gate 10 needed no
  DISSOLVED_PATHS row (the old STATE was the only scanned citer of run7/run8.yaml).
- 8b00b4dd: configs/run7.yaml + run8.yaml deleted (run6 stays on the HALT). Gates 7, 10, 12, 13 rc 0; tests/config
  + census 797 passed; W6-touched files 774 passed, 3 skipped; run10 MATCH; dev_example, smoke_preflight_armed,
  run10, run6 byte-identical to 69e15329. Collected 5086 -> **4928** (floor folded). tools/ci_gates/** untouched.
- Residue (on contact / W7 or card, not W6's): run-named symbols in non-reader files — test_gate_parity
  `..._matches_run3`, test_drawrate_pooled_statistic RUN5_* (PZ-1), test_checkpoint_conformance `run8_shaped`,
  test_gate_interval_decoupling `_RUN5_LOG_INTERVAL`, test_steps_budget_carry `run8s`, the `_run5()` helper; and (REVIEW-W6 #10) test_ring_audit's
  `..._run8_bands` pair, test_resolvers_nested_identity `..._run5_declares`, test_deploy_matched_hparam_coincidence's
  `..._equals_run5` and `..._run5_declares` (renames only, not assertion edits);
  armed_aborts.py policy_loss_trough owner text "run7's mint" (PZ-6 row, needs a ruling-named re-point).
- REVIEW-W6 (**opus**, fresh, read-only) LAUNCHED over 4e8c663a..8eec45a9. W7_ADDENDUM committed 8e5eab3e
  (exit step 5 taken early so a stop cannot lose it; its measures are derived at 8eec45a9).
- REVIEW-W6 FILED 56e90642 (`docs/audits/REVIEW_W6_2026-09-25.md`): 1 MUST-FIX, 7 SHOULD-FIX, 10 NOTE.
- Fix loop 1 LANDED caeaaf85..80cc7779 (8 commits, **opus**): #1 the two open cards restored; #2+#11 STATE's
  pointer rows re-pointed to the commits that hold each leg's facts (verified per commit, several reviewer
  candidates corrected), wave status left to PROGRESS; #3 armed_aborts.py REVERTED to its 4e8c663a bytes (the
  wave's reword touched a PZ-6 manifest row without a ruling and stated a false ground); #4 six CLOSED marker
  rows plus CARD-GAME-RECORD-STATUS-DRIFT; #5 three condensed rows' meaning restored; #6 profiling recipe adds
  `CARGO_PROFILE_RELEASE_STRIP=none`, amendments re-keyed `AMENDMENT (R368, 2026-09-25)`; #8 the D-15 pairs are one
  constant (node ids identical; a fifth-pair plant reds the census row); #14 F-816-34/35 name R338. #7, #9, #10
  recorded at 2ac9a886. Measures unchanged; collected 4928.
- REVIEW-W6 notes left with grounds: #12 (R368's Amends line vs R336(e), operator annotation), #13 and #15
  (on contact), #16 (acceptable), #17 (none), #18 (DISSOLVED_PATHS run9 reason stale; tools/ci_gates off-limits,
  next ruling-named gate edit).
- OPERATOR ASKS from W6: (1) the run6 grant (HALT 1 + HALT 2; the diff narrows, see above); (2) ONE ruling for
  armed_aborts.py MANIFEST text: policy_loss_trough's note/owner ("run7's mint"; three mints since kept it null),
  ply_cap_attractor's owner/note, terminal_eval_broken's RESIDUAL cite of the missing test file; (3) REVIEW-W6 #12.
- Closure check (the reviewer, read-only) at 80cc7779: every routed finding CLOSED, 0 new MUST/SHOULD, 3 notes
  (N1 fixed 4d9f4c09, N2/N3 fixed aea3a34c); §5 filed 0c8b9176.
- Exit sweep `make gates` LAUNCHED at 0c8b9176 in .wt/gates as user unit mantis-gates-w6 (01:32 UTC).
  Worktrees and branches w6-cards, w6-configs, w6-configs-b, w6-state, w6-fix REMOVED (each branch's content is on
  main, less the deliberately dropped frozen-oracle commits ad29e624 and 3334eda1).

## W5 — tests + fixtures (EXITED 2026-09-25)

**Exit facts.**
- Range 9839becd..8297c74d: 108 commits, 306 files, +3 302/−138 142. The bulk is fixture bytes; src/crates
  are 7 files, +46/−124, all unreachable dense code, docstrings and test-only cfg.
- Collected 5035 at W5 entry → **4927** = floor.
- Comment measures only fell: comment_excess 3169→3124, docstring_excess 12239→11908, private_docstring
  1378→1283, ruling_cite 1210→1163. banner 0, rust_doc 2486 and textfile 466 are unchanged.
- **run10 MATCH** at the tip. configs/, templates and mint are byte-unchanged.
- REVIEW-W5 filed (`docs/audits/REVIEW_W5_2026-09-24.md`): 3 MUST-FIX, 5 SHOULD-FIX, 9 NOTES. All eight
  fixed in ONE loop (4e18a0c8..3a94133d, opus); the reviewer verified the closure with its own plants.
- Exit sweep `make gates` in .wt/gates at b832d041, as user unit mantis-gates-w5:
  - 17 green, 2 RED. Both were the wave's own: one slashed slice name in the filed review matched gate
    17's abs-root-path pattern and redded gate 17, and 3a through test_gate_vacuity's wide scan.
  - Fixed at 8297c74d (the only change: the review doc). Re-run there: **3a 4869 passed, 15 skipped,
    0 failed**; gates 17/10/13/14/15/3c rc 0.
  - Walls: 2a 1061 s, 3a 339 s (re-run 335 s), 3b 2512 s (34 passed, 4 skipped).
  - pyright 0 errors; 8 ARMED+PASS; 12 rc 0.
- **The KNOWN-RED is gone.** cdbc8000's SIG_DFL preexec reset cures the inherited-SIG_IGN flake. The
  reviewer reproduced the old red under `trap '' TERM INT`, saw the new file pass under the same trap,
  and found the import-time plants still red, so the row is NOT narrowed. It passed in the sweep, its
  re-run, the review's full tier and the dispatcher's two pre-review tiers. The operator's pending row can
  close.

**Dispatchers and models per leg.**
- First W5 dispatcher (usage-limit stop after 74 commits, 9839becd..9d520676): slice legs root, eval,
  model, cfgtools/cfgfin, selfplay, train-a, train-b.
- Second W5 dispatcher (2026-09-24/25), models chosen per leg:
  - AQ-CARD-FAKES finisher: **opus**.
  - Drain re-base plus follow-up: **opus**.
  - Train helper hoists: **opus**.
  - Residue2 (Rust + config): **sonnet**.
  - Misc (eval/model/tools rows): **sonnet**.
  - L20–L34 inventory and W6 docs inventory: **sonnet**, read-only.
  - REVIEW-W5 and its closure check: **opus**.
  - Fix loop 1: **opus**.

**Landed by the second dispatcher** (sha: rows):
- w5-residue: 876f8353 (the four flat globs; the census choice was corrected by 15c2b6e8, see REVIEW #3),
  e5d236dc (REVIEW-W1 note 10, the fused-cap helper), d3419b30.
- 905cf259 AQ-CARD-FAKES: the cut-off 26-file diff was COMPLETED, not discarded. Equivalence was argued
  per family; node ids 273 = 273 over the touched files; two planted run.py breaks red; −1419.
- Leg 8, the DRAIN RE-BASE:
  - 15912645: the suite reads graph goldens re-captured from current code. No generator ever existed; a
    scratch capture asserted the arm-independent fields equal the old dense oracle first. REVIEW #10:
    acceptable.
  - a670e1dc: push_dense, the dense arm, `_is_graph`, PoolDims and `_feat/_chain/_pol_len` are gone; the
    dense goldens and collect_data_input.npz are deleted; line-endings floor 41→39.
  - Planted drain defects red before and after the deletion. push_dense_many never existed.
- Drain follow-up: 85cb2e9f L-DUP-22, fd6b28fc TESTS-5-03 residual, 0f7787f2 dense residue
  (recent_buffer, _board_size, _trunk_size, the pool_derived golden blocks).
- Residue2: 4300b691 REVIEW-W1 note 11 config half; 41e4271f atomic.rs TempDir guard; 2f002d9d L-DUP-04.
  For L-DUP-04 the dispatcher authorized widening from_stones' gate to `any(test, feature)`; the release
  build is unchanged.
- Misc: 0ae15928 TESTS-5-05 + 6-10a; 5ee831ef L-SEAM-05; 108279cc L-SEAM-06; 7eca14b3 L-DUP-15 residual;
  3b752fed L-DUP-28 residual.
- Train hoists: e5436147 TESTS-1-05, 209451ee TESTS-1-06/L-DUP-13, a13d0cc6 L-DUP-25, 7360de68 L-DUP-26
  residual. Node ids are equal per row, with planted breaks per family.
- 80ba1f6e: STATE.md's F2 line re-pointed (gate 10 red on the deleted _coordinator_pool.py).
- Fix loop 1: 4e18a0c8 (two falsely-subsumed rows restored), c79b4e79 (eval_enabled census value pin),
  15c2b6e8 (sweeps over discovered_config_paths, no hard count), 6d02eaa5, 06cd974e, 7a5173b0, 3a94133d.

**REFUTED at contact, with grounds.**
- TESTS-5-17: a planted row drop reds only j01, so it is kept.
- TESTS-1-14: 6c28c1c9 already drives pretrain; a plant reds 6.
- TESTS-6-06: sealbot is gone, and the strix env-key row has no subsumer.

**Still C / KEEP, with grounds.**
- `_model_samples` pair: R43/R310 frozen oracle.
- The receipt pair: KEEP-by-choice (REVIEW #15; open to a later hoist).
- The 13 `train_step_from_tensors` stubs: the REPROBE verdict.
- The drawrate trio's private fakes: PZ-1, with differing semantics.
- `test_coordinator_knobs_wiring._real_graph_ring`: tests/config cannot import tests/train.
- L-STYLE-04 (subprocess `text=True`, 79 sites; needs a gate-16 ruling) and L-STYLE-10 (function-scope
  imports, on contact): tree-wide, touching src and tools/ci_gates.
- TESTS-3-NEW-2, L-SEAM-NEW-2/-NEW-3, L-STYLE-NEW-2: closed-unverifiable. They are Δ0, and their text lived
  in a removed scratchpad.

**Records.**
- REVIEW #9: the e4018c6a floor lag (+12 carried by f4c23467); no history rewrite.
- REVIEW #12's subject inaccuracies are recorded as is.
- The drain capture procedure: drive the base harness's `run_stats_loop` on the graph arm, dump the
  recorded fields, and assert arm-independent equality with dense_5s_crossed before writing.
- OPERATOR items:
  - armed_aborts.py's `terminal_eval_broken` RESIDUAL cites the non-existent
    test_minted_config_remint.py. The live holder is c79b4e79's census row. It needs a ruling-named
    re-point.
  - pipeline.py's repr row is still pending from W3.

Worktrees: every w5-* worktree and branch is removed, and /tmp/base_check is gone. .wt/gates is detached at
8297c74d. NEXT: W6 per `docs/slim/handoff/W6_ADDENDUM.md`.

## W0 — entry

| step | state |
|---|---|
| env name printed | unset (operator's desktop) |
| `uv sync` + `import torch, mantis._engine` | GREEN |
| git fetch / origin/dev | 69e1532, nothing to rebase |
| host record | above |
| BEFORE sweep (`make gates.exit`, own worktree at 1e8d6d6, own venv) | ALL GREEN, 20 gates incl. slow. Walls (s, under concurrent agent load): 2a 2817, 2b 7, 4 0, 5 31, 3a 384, 3b 3373, slow 5, 3c 4, 7 1, 8 0, 9 0, 11 0, 12 0, 13 2, 14 11, 15 1, 16 1, 6 0, 10 0, 17 1 (≈ 1 h 50 min). The slow tier is 5 s: it stays here at W8 |
| R368 landed | 9ee503d7 |
| test_count_floor 4862 → 5112 (collected at base) | 1fbddf17 |
| re-probe of the 30 PENDING-PROBE rows | DONE: 29 GREEN, 1 AMENDED (TESTS-5-06 Δ −48), 0 RED — `docs/slim/REPROBE.md`. CORE-1-08 must edit `tests/train/test_trainer_seam_conformance.py`'s `GridRouteBufferLike` import + SEAM_MATRIX row in the same commit |

## W1 — correctness (01_DEFECTS 1–39, 42)

Rows a later wave resolves, named here per the packet:
- 15 (TrainerLike.train_step_from_tensors phantom) → W3, AQ-PHANTOM by R368(d).
- 16 (InferenceServer.submit_and_wait reshapes to a None shape) → W3 with CORE-2-03 (the grid arm); fixed there
  if the method survives.
- 17 (corpus_metrics creates reports/ at import) → W3, CORE-2-01 deletes the module.
- 23 (comment_lint misses bare R-cites and docstrings) → W4, where gate 14 learns both classes.
- 24 (test_count_floor 4862 vs 5112) → DONE in W0 (1fbddf17).

W1 rows integrated (defect → commit subject on the branch; each proven by a planted break, evidence in the leg's
agent reports kept for the review):
- 3 DONE (search_levers LAW-18 rows, event contract v2); 4+6 DONE (producer-less fields out, contract v3);
  5 DONE; 21 DONE (stub vs runtime test).
- 18, 19, 20, 38, 39 DONE (workspace builds with 0 warnings).
- 1, 12, 13, 22, 25, 26, 28 DONE; 42 DONE (STATE sanitized forward). 28's residue: the same flat glob at 7 in-scope
  sites (tests/arena/test_book_geometry_pairing.py, tests/test_run_eval_enabled_authority.py,
  tests/train/{test_arch_stamp_authority,test_bc_graph_reroute,test_ema_lever_is_reachable,
  test_pretrain_cli_states_no_training_knob}.py) → W5; two conformance sites are out of scope (R368(h)).
- 7, 8, 9, 10, 11, 14, 27, 29, 30, 35, 36 DONE (27 repaired as no-defaults MonitorConfig: importing the schema
  from monitor would cycle config ↔ monitor).
- 2, 31, 32, 33, 34 DONE; 37 REFUTED-AT-CONTACT (no mark on any non-test helper, AST scan of tests/; the six
  `@_LINUX_ONLY` marks sit on test functions and deselect under a forced platform).
- Residue for later waves: `v6_live2_ls` outside tests/eval (src/mantis/encoding/audit_sections.py,
  tests/config/test_resolve_encoding.py, tests/arena/test_deploy_head.py, tests/data/_frozen_games.py,
  tests/fixtures/selfplay/pool/encoding_resolve.json) → W3/W5 grid residue; `buffer_persist.try_save_buffer` and
  its counter now fully unread → CORE-1-07 (W3).

## W2 — Rust

Integrated (commit subjects on the branch carry the row IDs):
- RUST-3-16 DONE (one stub at the wheel path; pyright `stubPath`; the root wheel no longer ships a second
  `_engine.pyi`). RUST-3-01/02/03/04/05/06/07/08/09/10/11/12/13/14/17/18 DONE; RUST-3-15 done-at-contact (W1);
  TESTS-5-13 DONE with RUST-3-02; RUST-3-NEW-2 DONE. RUST-2-NEW-2 (spawn_mock_graph_games) still C: KEEP — four
  Python tests use it as the sole mock producer (TESTS-5-27 rule).
- RUST-2-01/03/04/05/06/07/08/09/10(helpers)/11/12/13/15 DONE; RUST-2-16 DONE via the RUST-3-10 family;
  RUST-2-02 WIRED (positions_dropped can fire on the graph arm — LAW-18 row, event contract v4);
  RUST-2-NEW-1 done-at-contact (W1); REVIEW-W1 note 3 DONE (the runner's forced-child pre-validation removed).
  Still C: RUST-2-14 (sole caller is the oracle bank's o4b test), RUST-2-17 feature_len/policy_len (the inv19/inv23
  pin subject), RUST-2-10's wide_board (oracle-bank feature).
- Search group (8 commits, 0c3ec191..c3231390; 4 from the interrupted agent's diff, verified before commit):
  RUST-1-01/02/03/04/06/07/16(rem)/17/NEW-1 DONE; -05 C→B DONE (its only callers were tests, both now drive the
  production `_ls_at`); -08 C→B DONE (dense `get_policy`; the only Python caller was a test leg); -09 C→B DONE
  (R155/R157 label leg 1 non-production and direct no file; r153_leg2 now runs v1 AND r8 through the production
  expand with aborts 1/3/4 — planted r8-only drop → red); -10 C→B PARTIAL (2 of 4 deleted; the cm 16/20/115 and
  cm 7 tests kept, no golden row pins them); -11 C→B PARTIAL (the two dirichlet property tests deleted,
  dirichlet_parity asserts both; the non-uniform blend test kept); -12 C→B DONE (implied by
  search_kind_conformance); -18 DONE; S-L-DUP-03 C→B DONE; -13/-14 CARDED, -15 stays C (R368(h)).
- REVIEW-W2 finding 3 (the shared `par::map_in_order` fan-out, 6c30e948): kept, not reverted. Ground: it dedups
  the thread-scheduling skeleton around the graph build, not the build — same static chunking, same join into the
  named error, same in-order flatten, monomorphic generic; `leaf_graph_parallel_parity` and
  `hexg_sample_parallel_parity` are bit-identical and green. No LAW-09 bench was taken; recorded as a DEVIATION
  from the letter of R368(h) (a reversal would re-grow the comment measures past floors the wave already lowered).
- Coverage dropped by RUST-2-13 (R368(k)): the pre-A1 fuse oracle's degenerate-array and dst-half offset-walk
  cases; the shipped fuse stays pinned by queue_fuse_pin's 3-graph frozen input + reconstruction + mutation test.
- Residue noted for contact: core `splitmix64_next` is `pub` in the production lib for tests (gate behind the
  `test-fixtures` feature); selfplay `WeightSchedule` is a one-field wrapper around f16 1.0 (collapse to a const);
  `Board.cluster_threshold` write-only (CARD-CLUSTER-THRESHOLD-RESIDUE).
- unwrap/expect: production sites fixed in encoding (registry parser), bridge (runner ctor), selfplay (HEXG loader,
  version_range, opening draw). CARD owed: hot-loop sites (core `Board::check_win`; selfplay queues/graph.rs 13
  lock/condvar-poison expects, graph.rs position(), search_drive.rs 261/265/724) and a POISON STANCE for 9
  lock-poison expects in selfplay runner (finalize.rs, mod.rs latch/stop/drain faces, spawn.rs, search_drive.rs:88).

## W3 — Python src

Integrated (commit subjects carry the row IDs):
- eval/arena/bots/diagnostics/monitor/util: CORE-3-01/02/03/04/05/06/07/08/09/10/11/12/15/16/17, CORE-2-04,
  CORE-3-NEW-1 (C→B: a duplicate, now called), TESTS-6-01 DONE; the sealbot adapter's src side GONE (R368(e),
  find_vendor_root moved to bots/strix.py first; eval_instrument.md v2); CORE-3-13 refuted-at-contact (W1);
  CORE-3-18/-19 KEEP (every diagnostics module has a caller; deploy/ reserved by CLAUDE.md). REVIEW-W2 note 6 DONE.
  OPERATOR QUESTION: pipeline.py's round-completion catch-all repeats `repr(exc)` in its logger.exception message
  (CLAUDE.md's style rule says it should not) but the protected O-30 witness
  `test_the_round_completion_route_logs_a_traceback_and_the_detail` pins the repr — kept (a witness is never
  narrowed); the rule vs the witness is the operator's call.
- train/model/encoding: CORE-1-01/03/04/05/06/07/08/09/10/11..16/17/18/19/24/25/26/28/29/31/32/NEW-1/NEW-2 DONE;
  CORE-1-22 anchor half DONE (resume_state half stays C: write_resume_state is PZ-1); CORE-1-30 partial (kept params
  are PZ-1/seam/golden-pinned); CORE-1-27 stays C (touches _registry_sha_handshake, PZ-2, no ruling names it);
  CORE-1-02 done at W1. 01_DEFECTS 15 RESOLVED (TrainerLike + GridRouteBufferLike gone; the production Trainer IS a
  TrainerLike, pinned). Newly found: Trainer.load_checkpoint (classmethod) has zero callers — left (LAW-12 caution).
- selfplay/config/data/env/run.py: CORE-2-01/02/03/05/06/07/08/10/11/12/15/16(run.py half)/17/18/19/20/21/22,
  CORE-2-NEW-1, L-DUP-05/06/08/09 DONE; the old corpus pipeline and mantis.env GONE (uv.lock −465: matplotlib, rich
  out of the `analysis` extra); 01_DEFECTS 16 RESOLVED (submit_and_wait was grid-only, deleted), 17 RESOLVED (module
  gone + a new import-side-effect witness, tests/test_import_side_effects.py). Still C: CORE-2-13 (the nsims resolver
  is a contract-cited seam member; its sealbot name goes with the residue agent), CORE-2-16 resolver modules,
  CORE-2-23 (the dense drain arm is the drain-parity oracle's driven path → W5 with the drain re-base).
- sha256 leg: AQ-SHA DONE (L-DUP-11, CORE-1-20/21, CORE-3-14 sha half, TOOLS-2-12a/b, defect 51) — every file sha256
  in src/ and tools/ reads `mantis.util.hashing.sha256_file` (streamed; digests identical; witness
  tests/util/test_hashing.py against every manifest-pinned sha). Kept apart with grounds: tools/ci_gates'
  preflight_mint_parent `_sha256` (tools/ci_gates/** needs a ruling naming it), audit_bootstrap_corpus (mantis-free by
  design), registry_gate.sh (independent oracle), books.py (hashes the bytes it parses), checkpoint_state_sha256
  (a net-param hash, PZ-1). W3 residue DONE: GameRecorder.latest_replay_path, util.constants.HISTORY_LEN, run.py's
  three grid IfExps + test_the_grid_arm_is_the_serial_width (S-A-TESTS-6-11 run.py half), grid prose, the sealbot
  opponent name in nsims (CORE-2-13's resolver stays C). Card candidate: arena/match.py::_trajectory_hash ≡
  eval/aggregate.py::_traj_key (PZ-1).
- broad-except: fixed eval/child_memory make_probe, worker_sweep CLI refusals now log tracebacks, train stamp
  resolver + parent-death arm + disk_guard loop, selfplay spearman read; the rest classified as top-level handlers,
  record-and-surface contracts, PZ-1 dump-on-fire guards or re-raises (agent reports). Card candidates: the
  worker_sweep sampler thread, the inference_server graph-loop fail paths, the signals.py:325 force-exit reap.

## Ledger (row → done / refuted-at-contact / halted)

### W4 (tools + the gate-learning orders)

- DONE: TOOLS-2-07 (71903a8d), TOOLS-2-08 AMENDED-with-ground (a9528a77 — constants deduped, scan
  kept; see REVIEW-W4 finding 1), TOOLS-2-09 (e87e3686), TOOLS-2-10 (44678400), TOOLS-2-15
  (6adbeebf), TOOLS-1-01 (622893b6), the sealbot vendor side incl. eval_instrument v3 (60755d15),
  defect 23 / R368(g)'s two gate-14 measures (5fc33aa1), gate 16 tree-wide zero (51eddda5),
  TOOLS-1-12 riding the named comment_lint edit (the _excess cap parameter).
- DONE in an earlier wave, recorded: TOOLS-2-12 (W3 84a4eef0), L-STYLE-03 (W4's gate-16 widening).
- KEEP by ruling: TOOLS-1-14 and TOOLS-1-15 (R368(k) — the workflow is the operator's switch);
  TOOLS-1-13 (jscpd 0 clones, divergent semantics); TOOLS-2-01 (the R247 certify stage of a live
  pipeline), TOOLS-2-02 `proofs` (falsified.md F-53's named instrument; netread/spread stay),
  TOOLS-2-04 (fixture provenance).
- Still C, grounds: TOOLS-1-02..06, 08, 09, 10, 11, 16 (R368(b): "The same holds for …
  tools/ci_gates/**" — no ruling names those edits; config_templates is the brief's do-not-edit);
  TOOLS-1-NEW-2 RESOLVED by R368(b) itself (record-only); TOOLS-2-03 (pre-R362 record rendering,
  operator's call); TOOLS-2-13/-14/-16/-17 (PZ homes: the checkpoints loader + build_net,
  pipeline/aggregate, the PZ-4 minter, a contract-doc cite — the swa site died with W3).
- Residue for later waves: REVIEW-W4 N2 (one reclaimable ruling_cite_lines floor line, W7/W8),
  N3 (CARD-STYLE-BACKLOG's stale measure figure, W6).
