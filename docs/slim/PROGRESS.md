# PROGRESS — SLIM-FIX (R368), the durable resume point

Resume from THIS file after any stop, never from memory. Updated at every leg exit.

## Where the run is

- Branch: `claude/slim-fix-r368`, cut from `origin/claude/slim-scout-census-v3i2hj` @ 1e8d6d6 (base dev 69e1532).
- Host: the operator's desktop (not a Claude environment; `CLAUDE_CODE_ENVIRONMENT_NAME` unset). AMD Ryzen 7
  3700X, 16 threads, 46 GiB, flags `avx2` only (no `avx512_bf16`, no `amx`). torch 2.11.0+cpu, `mantis._engine` OK.
- origin/dev = 69e1532 at W0 entry (no moved commits; no rebase).
- Wave: **W3** (Python src). W2 EXITED: REVIEW-W2 filed (`docs/audits/REVIEW_W2_2026-09-24.md`, 0 must-fix, one
  loop); exit sweep `make gates` on 3db6ab5a ALL GREEN, 19 gates (2a 1068 s, 3a 356 s, 3b 3005 s); the post-review
  text commits on 8b75f984 re-checked (gates 10/13/14/15/17, workspace build 0 warnings); run10 resolved MATCH at
  3db6ab5a. W1 EXITED earlier (cadcc367, 19 gates green after a quota-voided first run). W3: the three package
  groups INTEGRATED (collected 5147 → 5039); the sha256 consolidation + W3 residue agent running in `.wt/w3-sha`.

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
- broad-except: fixed eval/child_memory make_probe, worker_sweep CLI refusals now log tracebacks, train stamp
  resolver + parent-death arm + disk_guard loop, selfplay spearman read; the rest classified as top-level handlers,
  record-and-surface contracts, PZ-1 dump-on-fire guards or re-raises (agent reports). Card candidates: the
  worker_sweep sampler thread, the inference_server graph-loop fail paths, the signals.py:325 force-exit reap.

## Ledger (row → done / refuted-at-contact / halted)

(filled per wave)
