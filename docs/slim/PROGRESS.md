# PROGRESS — SLIM-FIX (R368), the durable resume point

Resume from THIS file after any stop, never from memory. Updated at every leg exit.

## Where the run is

- Branch: `claude/slim-fix-r368`, cut from `origin/claude/slim-scout-census-v3i2hj` @ 1e8d6d6 (base dev 69e1532).
- Host: the operator's desktop (not a Claude environment; `CLAUDE_CODE_ENVIRONMENT_NAME` unset). AMD Ryzen 7
  3700X, 16 threads, 46 GiB, flags `avx2` only (no `avx512_bf16`, no `amx`). torch 2.11.0+cpu, `mantis._engine` OK.
- origin/dev = 69e1532 at W0 entry (no moved commits; no rebase).
- Wave: **W5** (tests + fixtures) IN FLIGHT — see `## W5` below. W4 EXITED: W3 EXITED. REVIEW-W3 filed (`docs/audits/REVIEW_W3_2026-09-24.md`, 0 must-fix beyond a
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

## W5 — tests + fixtures (IN FLIGHT; resume here)

- Range starts at 9839becd. 74 commits integrated by the first W5 dispatcher (usage-limit stop) through
  9d520676; the second dispatcher resumed 2026-09-24: w5-residue integrated (876f8353 four flat globs →
  census, e5d236dc the fused-cap helper off dev_example, d3419b30 conftest cite) — collected 4906 → 4902
  (the census drops the two exempt configs from four parametrizations). Integrated branches' worktrees and
  branches removed (w5-root/-eval/-model/-cfgfin/-cfgtools/-selfplay/-train-a/-train-b/-residue,
  /tmp/base_check).
- In flight: w5-fakes (AQ-CARD-FAKES hoist; 26 files were left uncommitted mid-leg; rebased onto d3419b30,
  finisher model opus); w5-drain (leg 8, the dense drain oracle re-base + src deletion; opus); a read-only
  inventory of L20–L34 (sonnet) to name the remaining rows.
- Models per leg so far (second dispatcher): fakes finisher opus, drain re-base opus, inventory scout sonnet.
- LANDED leg 8 (drain re-base, opus): 15912645 (suite on graph goldens re-captured from current code; no generator
  existed — scratch capture, arm-independent fields asserted equal to the dense oracle first) + a670e1dc (push_dense,
  the dense arm, _is_graph, PoolDims, _feat/_chain/_pol_len gone; dense goldens + collect_data_input.npz deleted;
  line-endings floor 41 → 39). Planted drain defects (row drop; per-row game id) RED before and after the deletion.
  push_dense_many never existed in Rust. run10 MATCH. Collected 4899.
- Inventory (sonnet scout): ~128 rows DONE, 11 in flight, 20 UNDONE + 4 unverifiable Δ0 NEW rows (TESTS-3-NEW-2,
  L-SEAM-NEW-2/-NEW-3, L-STYLE-NEW-2: descriptions lived in a removed scratchpad). Policy rows L-STYLE-04
  (subprocess text=True encoding) and L-STYLE-10 (function-scope imports) are NOT W5 legs: tree-wide, touch src and
  tools/ci_gates (R368(b)) — carried to the HANDOFF as still-C.
- LANDED (second dispatcher, model per leg): 905cf259 AQ-CARD-FAKES finished (opus; the cut-off 26-file diff
  completed, not discarded: equivalence argued per family, 273 = 273 node ids over the touched files, two planted
  run.py breaks red; −1419); drain follow-up (opus) 85cb2e9f L-DUP-22, fd6b28fc TESTS-5-03 residual, 0f7787f2 dense
  residue (recent_buffer/_board_size/_trunk_size, pool_derived goldens), TESTS-5-17 REFUTED (planted row drop reds
  only j01); residue2 (sonnet) 4300b691 note 11 config half (census-parametrized, +21 rows), 41e4271f atomic.rs
  TempDir guard, 2f002d9d L-DUP-04 (dispatcher authorized widening from_stones' gate to any(test, feature)),
  TESTS-1-14 REFUTED (6c28c1c9 already drives pretrain; plant reds 6); misc (sonnet) 0ae15928 TESTS-5-05+6-10a,
  5ee831ef L-SEAM-05, 108279cc L-SEAM-06 (subsumer the GRAVES row; flatten plant reds), 7eca14b3 L-DUP-15 residual,
  3b752fed L-DUP-28 residual (2 of 4 pairs; `_model_samples` is the R43/R310 frozen pair → C; the receipt pair
  differs by a default arg → KEEP); TESTS-6-06 REFUTED (sealbot gone; the strix env-key row has no subsumer).
  Collected 4918.
- Assigned: w5-residue2 (sonnet: run6-bound _run5 → census, atomic.rs tempdir guard, L-DUP-04 fwm_board, TESTS-1-14);
  w5-misc (sonnet: TESTS-5-05, 6-10a, L-SEAM-05/-06, TESTS-6-06, L-DUP-15/-28 residuals); drain agent follow-up
  (opus: L-DUP-22, TESTS-5-03/-17 residuals, dense residue recent_buffer/_trunk_size/_board_size); after fakes lands:
  train hoists TESTS-1-05/-06, L-DUP-25, L-DUP-26 residual.

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
