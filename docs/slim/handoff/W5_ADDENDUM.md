# W5 ADDENDUM — the tests + fixtures wave (read after WAVE_BRIEF.md)

Test slimming (R368(d)/(e)/(k), the AQ-* dispositions) plus residue. A deletion of a dead test is
proven by the collected count dropping in its own commit + the file's neighbours green; a WITNESS
change needs its planted break. Slimming never grows lines net per commit. This wave is LARGE:
work the legs in the order below, one commit per finding class, and integrate after each leg —
the floors fold per commit and a long un-integrated tail risks conflicts.

## Leg order (topological; LEDGER §4 L20–L34)
1. **R368(e) fixture deletions FIRST** (they move two floors and a manifest):
   - `tests/fixtures/graph_parity/wpa_positions.json` (966,939 B) + its `manifest.tsv` F-row and
     the `corpus_file`/`corpus_sha256` header lines; the outer `tests/fixtures/manifest.toml`
     row for `graph_parity/manifest.tsv` is RE-PINNED in the same commit with a comment naming
     the grant (R368(e)) and the commit the file lived in (`69e15329`); precedent: the
     completed_q/golden_bits.txt row's grant comment. `.gitattributes:5` cites the file — reword.
     Only reader is the Rust manifest loop (graph_parity.rs `manifest_and_fixture_files_valid`);
     no Python test touches it.
   - `tests/fixtures/value_probes/`: delete `decoded_v.npz`, `metrics.json`, `probe_set_v1.jsonl`,
     `negatives_v1.jsonl`, `CHANGELOG.md` + their manifest.toml rows AND the six-line
     `CORRECTED IN PLACE (R311(c))` note (:73–78 — R368(e) overrides the recorded revisit).
     KEEP the live goldens `dist65_golden.json`, `forward/small_gnn.pt`, `statedict_keys/`
     (readers: test_dist65, test_forward_parity, test_build_net/train conftest).
     `tests/test_fixtures_manifest.py` must stay green; `tests/test_line_endings.py`'s ≥40-row
     floor moves in the same commit (46 → 41; 39 if 9-06 lands).
2. **Dead-test deletions (L27, L28)** before any hoist — the hoists' Δ and conflict surface
   shrink. Per-row lists in LEDGER; the collected count and tier_declaration.txt rows move with
   each deletion. Already gone (record only): TESTS-5-13 (W2), 1-02/2-13 (W3), 4-17 (W3 pack),
   6-11 run.py half (W3).
3. **Row deletions (L32, L33)** — subsumed/duplicate rows; contract-doc rows that name a deleted
   test move in the SAME commit (gate 13/10).
4. **The AQ-* hoists (R368(k))** — a card/citation protects a file's existence and assertion,
   NOT its helpers:
   - AQ-CARD-FAKES: the `_Pool/_Buffer/_RunnerStats/_fake_run_safety` families (7 root
     composition files → `tests/_drivable.py::DrivablePoolStub`, reconciled with
     tests/train/_coordinator_pool.py::CoordinatorPoolStub — ONE shared double, not two);
     drain_caps_wiring vs coordinator_knobs_wiring; the `_Buffer` trio in train wiring tests;
     cluster_* stub attrs; TESTS-2-07's four families.
   - AQ-LIFECYCLE (K20 carve): hoist ONLY the true twins (`_alive` ×3, `_ppid_of` ≡
     `_ppid_of_pid`, `_events` ×2). `_spawn_supervisor`/`_reap` differ deliberately — keep both.
   - AQ-BF16: merge `deterministic_algorithms` (tests/model/_bf16_parity.py ⇄
     tests/train/_microbatch_harness.py) and `_graph_step` (test_finite_gradient_guard ⇄
     test_nonfinite_guard) WITH a planted fp16 swap reding BEFORE and AFTER the merge.
     CONSTRAINT: tests/model/test_bf16_parity_nulldist.py (PZ) must not be edited — keep one copy
     importable as `bp.deterministic_algorithms`.
   - AQ-P2: print set-equality of the two CONSUMER_REGISTRY dicts
     (test_every_key_has_consumer.py ⇄ _p2.py, 103 identical strings) in a witness, then retire
     the p2 copy AND tests/config/test_regime_parity_p2.py + test_resolved_config_emit_p2.py;
     docs/contracts/run_config_schema.md:175's row moves in the same commit.
   - AQ-POOL-HPARAMS (K15): delete test_pool_hparams_arms.py's killed-knob test + its
     `_RecordingRunnerConfig` + `assemble` fixture; the survivor is test_pool_hparams.py's
     killed-and-relocated test + the runner_config_goldens row (R38 already annotated).
5. **L-SEAM-05/-06, L20/L21/L23/L34 doc+simplicity rows** (small; comment floors fall per commit).
6. **The stubs verdict**: the 13 `train_step_from_tensors` stubs STAY (REPROBE: nothing reds with
   the protocol member gone). Only TESTS-2-15 (test_train_step_dispatch's dense-entry test) and
   TESTS-7-07 (`_ExplodingTrainer`'s override + `_Buffer.sample_batch_with_pos`) act.
7. **Residue**: defect-28's six still-flat `glob("*.yaml")` sites re-point to
   `mantis.config.census.production_configs(repo_root)` (the ONE discovery authority, pinned by
   test_config_discovery_authority; the 7th site died with W3's bd5ac4ed; the two conformance
   sites stay out, R368(h)). REVIEW-W1 notes 9 (`tests/_monitor_config.py::REQUIRED_TEST_VALUES`
   ⇄ test_monitor_schema's VALID_MONITOR_SCALARS → one factory), 10 (the 37-file `57149441` caps
   literal → one fixture read off dev_example's resolved section), 11 (test_pool_encoding_bridge
   `_run5_dump`, test_resolved_config_emit `_run5` — re-point; run6/8 die in W6, so point at the
   census, not a named config). The `/tmp/mantis-*` leaks: test_abort_exit_signal.py:139 and
   test_run_root_lifecycle.py:132 (mkdtemp → tmp_path/TemporaryDirectory teardown),
   crates/mantis-selfplay/src/replay/atomic.rs:147-152 (Rust: remove_dir_all or a Drop guard).
8. **THE DRAIN RE-BASE LAST (R368(d))**: point tests/selfplay/test_pool_drain_parity.py at the
   GRAPH goldens (graph_pushed.npz + the graph variant; `collect_data_input.npz` is NOT
   dense-only — both factories read it), PLANT a drain defect and show the suite reds, then
   delete: `pool_push.push_dense` + its `__all__` entry, `pool_drain.run_stats_loop`'s dense
   else-arm and the `"representation": "graph" if pool._is_graph else "dense"` ternary (CORE-2-23),
   `pool._is_graph` (hparams.py:123), and the dense PoolDims fields (feat_len/chain_len/pol_len,
   hparams.py:213-218 — goldens pin PoolDims(0,0,362); the fixtures/selfplay/pool/
   runner_config_goldens.json `"_is_graph": true` rows move with the flag). Witnesses that must
   stay green/extended: test_pool_drain_parity, test_pool_drain_arms, test_drain_row_shape_parity
   (`_FAKE_FILES` pins the fake paths), test_game_complete_absence (reads dense_5s_crossed
   events — re-point), the manifest re-pin, the line-endings floor. The dense goldens
   (dense_pushed.npz + the 4 dense variants) go only after the suite no longer reads them.

## Hard limits (beyond WAVE_BRIEF's)
- tests/model/conformance/** stays out (R368(h)); test_bf16_parity_nulldist.py unedited (PZ).
- The two conformance `glob("*.yaml")` sites stay flat (R368(h)).
- Frozen-oracle files: a byte-frozen test is edited ONLY by the freeze's owner row; the R43/R310
  pairs stay C (re-lane table records the ground).
- Manifest edits: schema is exactly {path, sha256, added_by} rows + the tsv's F-rows; the checker
  reds on any drift — that is the point.
- tools/ci_gates/**: only tier_declaration.txt row removals for deleted marked tests (the brief's
  standing permission); NOTHING else.
- Do NOT hand-edit configs/*.yaml; run6–8.yaml deletions are W6's, not this wave's.

## Checks (each commit)
The touched tests + every test that greps a removed name (`git grep -n <name> -- src tools tests
crates docs`); `pytest --collect-only -q -m ''` count (dispatcher folds); comment_lint GREEN;
r8_header_gate on touched files (a file crossing under 300 drops its header — the gate says so);
encoding_io_gate (now tree-wide); gate 10; gate 12 `--audit-only`; `ruff check .`; for any
src/mantis/{config,train,model}/ touch: the run10 resolved check.

## Report additions
Per commit: rows, Δlines net, tests run, collected count, floor moves. The re-lane table (ID | old
| new | ground) for every TESTS lane-C row in LEDGER §6 the wave touches or re-lanes — the W5
scout inventory (2026-09-24, dispatcher session) already lists each with its recorded ground;
verify at contact, do not trust blindly.
