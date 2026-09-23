# 00_MAP — slimming census, the BEFORE figure and the partition

Tree: `origin/dev` = `69e153296c6d8245456d08ab5ddb6a0749e47c67` (2026-09-21 22:53 +0200), 1 071 tracked paths.
Census branch: `claude/slim-scout-census-v3i2hj` (the platform's branch; the packet's `slim` is this one).
Register at HEAD: `### R358` present; the head entry is R367 (the packet snapshot said R358 — verified newer).

## 1. Size (BEFORE)

Unit: text lines from `git diff --numstat <empty-tree> HEAD` (git's own count; binary files count 0 and are
tallied separately). `wc -l` is NOT used: it counts newline bytes inside the 33 binary fixtures.

Command (per top dir):

```
E=$(git hash-object -t tree /dev/null)
git diff --numstat $E HEAD | awk -F'\t' '{p=$3; top=(index(p,"/")?substr(p,1,index(p,"/")-1):"<root>"); F[top]++; if($1=="-")B[top]++; else L[top]+=$1} END{for(k in F) printf "%-10s %5d %4d %7d\n",k,F[k],B[k]+0,L[k]}' | sort -k4 -n -r
```

| top dir | files | binary | text lines | of which .py | .rs | .md | other |
|---|---:|---:|---:|---:|---:|---:|---:|
| tests | 546 | 33 | 279 440 | 97 375 | 0 | 36 | 182 029 (fixture data) |
| docs | 67 | 0 | 40 710 | 0 | 0 | 40 710 | 0 |
| crates | 145 | 0 | 40 012 | 0 | 39 019 | 0 | 993 |
| src | 193 | 0 | 38 133 | 37 201 | 0 | 0 | 932 |
| tools | 98 | 0 | 16 362 | 14 337 | 0 | 29 | 1 996 |
| (root files) | 13 | 0 | 3 646 | 0 | 0 | 303 | 3 343 |
| configs | 6 | 0 | 1 055 | | | | 1 055 |
| vendor | 2 | 0 | 149 | | | | 149 |
| .github | 1 | 0 | 141 | | | | 141 |
| **all** | **1 071** | **33** | **419 648** | 148 913 | 39 019 | 41 078 | 190 638 |

Command (per package): the same pipe with the key `src/mantis/<pkg>`, `crates/<crate>`, `tests/<pkg>`,
`tools/<pkg>`, `docs/<dir>` (files directly under a top dir keyed `<top>/<files>`):

```
git diff --numstat $E HEAD | awk -F'\t' '{p=$3; c=split(p,a,"/"); if(c==1)k="<root>"; else if(a[1]=="src"&&c>=4)k=a[1]"/"a[2]"/"a[3]; else if(a[1]=="src")k="src/mantis/<files>"; else if((a[1]=="crates"||a[1]=="tests"||a[1]=="tools"||a[1]=="docs")&&c>=3)k=a[1]"/"a[2]; else k=a[1]"/<files>"; F[k]++; if($1=="-")B[k]++; else L[k]+=$1} END{for(k in F) printf "%-32s %5d %4d %7d\n",k,F[k],B[k]+0,L[k]}' | sort
```

```
package                          files  bin   lines
.github/<files>                      1    0     141
<root>                              13    0    3646
configs/<files>                      6    0    1055
crates/mantis-bridge                13    0    5176
crates/mantis-core                  17    0    3855
crates/mantis-encoding               9    0    1647
crates/mantis-graph                  8    0    2837
crates/mantis-search                35    0   11382
crates/mantis-selfplay              63    0   15115
docs/audits                          5    0    3096
docs/contracts                       9    0    1502
docs/design                         43    0   12314
docs/governance                     10    0   23798
src/mantis/<files>                   4    0    1695
src/mantis/arena                     9    0     934
src/mantis/bots                      6    0     860
src/mantis/config                   45    0    5273
src/mantis/data                     14    0    2529
src/mantis/deploy                    1    0       1
src/mantis/diagnostics              11    0    4154
src/mantis/encoding                  7    0    1514
src/mantis/env                       2    0      84
src/mantis/eval                     11    0    3110
src/mantis/model                    10    0    1358
src/mantis/monitor                  12    0    2623
src/mantis/selfplay                 14    0    4356
src/mantis/train                    38    0    9157
src/mantis/util                      9    0     485
tests/<files>                       24    0    6186
tests/arena                         11    0    1556
tests/bots                           5    0    1543
tests/bridge                        16    0    1683
tests/config                        56    0   11050
tests/data                           7    0    1099
tests/diagnostics                   19    0    4274
tests/encoding                      15    0    1437
tests/env                            1    0     113
tests/eval                          47    0    8851
tests/fixtures                      64   33  182119
tests/model                         32    0    7917
tests/monitor                       18    0    3905
tests/selfplay                      57    0   10339
tests/tools                         73    0   14286
tests/train                         95    0   22428
tests/util                           6    0     654
tools/<files>                       23    0    4972
tools/analyzer                      13    0    1305
tools/ci_gates                      21    0    5569
tools/config_templates               1    0     247
tools/dashboard                     17    0    2144
tools/ladder                        10    0     996
tools/probe1                         8    0     750
tools/viewer                         5    0     379
vendor/<files>                       2    0     149
```

## 2. Partition (area slices)

Every `git ls-files` path is matched against the 21 slice regexes below; exactly one must match.
The awk program is this block, verbatim; the command extracts it from this file and runs it:

```
sed -n '/^```awk$/,/^```$/p' docs/slim/00_MAP.md | sed '1d;$d' > /tmp/partition.awk
git diff --numstat $(git hash-object -t tree /dev/null) HEAD | awk -F'\t' -f /tmp/partition.awk
```

```awk
BEGIN {
  n = 0
  S[++n] = "R1";  P["R1"] = "^crates/mantis-search/"
  S[++n] = "R2";  P["R2"] = "^crates/mantis-selfplay/"
  S[++n] = "R3";  P["R3"] = "^crates/mantis-(core|encoding|graph|bridge)/"
  S[++n] = "C1";  P["C1"] = "^src/mantis/(train|model|encoding)/"
  S[++n] = "C2";  P["C2"] = "^src/mantis/((selfplay|data|config|env)/|[^/]+$)"
  S[++n] = "C3";  P["C3"] = "^src/mantis/(eval|arena|bots|diagnostics|monitor|util|deploy)/"
  S[++n] = "L1";  P["L1"] = "^(tools/(ci_gates|config_templates)/|Makefile$|\\.github/)"
  S[++n] = "L2";  P["L2"] = "^tools/([^/]+$|analyzer/|dashboard/|ladder/|probe1/|viewer/)"
  S[++n] = "T1";  P["T1"] = "^tests/train/([^t]|t[^e]|te[^s]|tes[^t]|test[^_]|test_[a-g])"
  S[++n] = "T2";  P["T2"] = "^tests/train/test_[h-z]"
  S[++n] = "T3";  P["T3"] = "^tests/tools/"
  S[++n] = "T4";  P["T4"] = "^tests/(config|data|encoding|env|util)/"
  S[++n] = "T5";  P["T5"] = "^tests/(selfplay|bridge|arena)/"
  S[++n] = "T6";  P["T6"] = "^tests/(eval|bots|diagnostics)/"
  S[++n] = "T7";  P["T7"] = "^tests/([^/]+$|monitor/)"
  S[++n] = "T8";  P["T8"] = "^tests/model/"
  S[++n] = "T9";  P["T9"] = "^tests/fixtures/"
  S[++n] = "D1";  P["D1"] = "^docs/(governance/[^/]+$|audits/|contracts/|slim/)"
  S[++n] = "D2";  P["D2"] = "^docs/governance/archive/"
  S[++n] = "D3";  P["D3"] = "^docs/design/"
  S[++n] = "D4";  P["D4"] = "^(configs/|vendor/|(\\.gitattributes|\\.gitignore|CLAUDE\\.md|Cargo\\.lock|Cargo\\.toml|LICENSE|README\\.md|mise\\.toml|pyproject\\.toml|rust-toolchain\\.toml|rustfmt\\.toml|uv\\.lock)$)"
}
{
  add = $1; path = $3; hits = 0; who = ""
  for (i = 1; i <= n; i++) if (path ~ P[S[i]]) { hits++; who = S[i] }
  if (hits == 0) { unassigned++; print "UNASSIGNED " path > "/dev/stderr"; next }
  if (hits > 1) { double++; print "DOUBLE " path > "/dev/stderr"; next }
  F[who]++
  if (add == "-") B[who]++; else L[who] += add
}
END {
  printf "%-4s %6s %6s %8s  %s\n", "id", "files", "binary", "lines", "pattern"
  for (i = 1; i <= n; i++) { s = S[i]; tot += L[s]; ft += F[s]; printf "%-4s %6d %6d %8d  %s\n", s, F[s], B[s], L[s], P[s] }
  printf "ALL  %6d %8s %8d\n", ft, "", tot
  printf "partition check: unassigned %d, double-assigned %d\n", unassigned + 0, double + 0
}
```

Result at `69e1532`:

```
id    files binary    lines  pattern
R1       35      0    11382  ^crates/mantis-search/
R2       63      0    15115  ^crates/mantis-selfplay/
R3       47      0    13515  ^crates/mantis-(core|encoding|graph|bridge)/
C1       55      0    12029  ^src/mantis/(train|model|encoding)/
C2       79      0    13937  ^src/mantis/((selfplay|data|config|env)/|[^/]+$)
C3       59      0    12167  ^src/mantis/(eval|arena|bots|diagnostics|monitor|util|deploy)/
L1       24      0     6043  ^(tools/(ci_gates|config_templates)/|Makefile$|\.github/)
L2       76      0    10546  ^tools/([^/]+$|analyzer/|dashboard/|ladder/|probe1/|viewer/)
T1       46      0    12031  ^tests/train/([^t]|t[^e]|te[^s]|tes[^t]|test[^_]|test_[a-g])
T2       49      0    10397  ^tests/train/test_[h-z]
T3       73      0    14286  ^tests/tools/
T4       85      0    14353  ^tests/(config|data|encoding|env|util)/
T5       84      0    13578  ^tests/(selfplay|bridge|arena)/
T6       71      0    14668  ^tests/(eval|bots|diagnostics)/
T7       42      0    10091  ^tests/([^/]+$|monitor/)
T8       32      0     7917  ^tests/model/
T9       64     33   182119  ^tests/fixtures/
D1       20      0    11180  ^docs/(governance/[^/]+$|audits/|contracts/|slim/)
D2        4      0    17216  ^docs/governance/archive/
D3       43      0    12314  ^docs/design/
D4       20      0     4764  ^(configs/|vendor/|(\.gitattributes|\.gitignore|CLAUDE\.md|Cargo\.lock|Cargo\.toml|LICENSE|README\.md|mise\.toml|pyproject\.toml|rust-toolchain\.toml|rustfmt\.toml|uv\.lock)$)
ALL    1071            419648
partition check: unassigned 0, double-assigned 0
```

Slice notes:
- Split rule (~15k): R2 (15 115) is one crate and within tolerance. T9 (182 119) is over by DATA: 167 362 of
  its lines are three fixture files (`graph_parity/wpa_positions.json` 124 769, `graph_parity/manifest.tsv`
  23 771, `mctx_parity/mctx_parity_v1.json` 18 822) inside indivisible packages; its code/manifest part is small.
  D2 (17 216) is four frozen files in one directory. tests/train had no sub-package, so T1/T2 split it at the
  alphabetical midpoint (`test_[a-g]*` + underscore helpers | `test_[h-z]*`) — a deviation from "package
  boundaries", recorded.
- `docs/slim/` is placed in D1 so the partition still holds after this census commits.

| slice | scout file | area | content |
|---|---|---|---|
| R1 | S-A-RUST-1.md | A-RUST | crates/mantis-search |
| R2 | S-A-RUST-2.md | A-RUST | crates/mantis-selfplay |
| R3 | S-A-RUST-3.md | A-RUST | crates/mantis-core, -encoding, -graph, -bridge |
| C1 | S-A-CORE-1.md | A-CORE | src/mantis/train, model, encoding |
| C2 | S-A-CORE-2.md | A-CORE | src/mantis/selfplay, data, config, env + package root files |
| C3 | S-A-CORE-3.md | A-CORE | src/mantis/eval, arena, bots, diagnostics, monitor, util, deploy |
| L1 | S-A-TOOLS-1.md | A-TOOLS | tools/ci_gates, tools/config_templates, Makefile, .github |
| L2 | S-A-TOOLS-2.md | A-TOOLS | tools/ top-level modules, analyzer, dashboard, ladder, probe1, viewer |
| T1 | S-A-TESTS-1.md | A-TESTS | tests/train a–g + helpers |
| T2 | S-A-TESTS-2.md | A-TESTS | tests/train h–z |
| T3 | S-A-TESTS-3.md | A-TESTS | tests/tools |
| T4 | S-A-TESTS-4.md | A-TESTS | tests/config, data, encoding, env, util |
| T5 | S-A-TESTS-5.md | A-TESTS | tests/selfplay, bridge, arena |
| T6 | S-A-TESTS-6.md | A-TESTS | tests/eval, bots, diagnostics |
| T7 | S-A-TESTS-7.md | A-TESTS | tests/ root modules + tests/monitor |
| T8 | S-A-TESTS-8.md | A-TESTS | tests/model (incl. the conformance suite) |
| T9 | S-A-TESTS-9.md | A-TESTS | tests/fixtures |
| D1 | S-A-DOCS-1.md | A-DOCS | docs/governance (non-archive), docs/audits, docs/contracts |
| D2 | S-A-DOCS-2.md | A-DOCS | docs/governance/archive (frozen) |
| D3 | S-A-DOCS-3.md | A-DOCS | docs/design/** |
| D4 | S-A-DOCS-4.md | A-DOCS | configs/, vendor/, root files |
| — | S-L-DUP.md, S-L-SEAM.md, S-L-STYLE.md | LENS | whole tree |

## 3. Protected zone (lane C only)

Resolved by a read-only map agent against HEAD (every row carries its evidence command in the source table); the dispatcher did not re-derive it. Headings demoted one level.


`gg` = `git grep -n`. Symbols are `path::symbol`, and nothing below uses a line number as a citation.
Source of the list: docs/governance/LAWS.md "The protected set" = RULINGS.md R346(d) verbatim (`sed -n 45,60p LAWS.md`).

### PZ-1 LAWS protected set (11 items)

| item | implementing path::symbol | pinning test(s) | evidence cmd → result |
|---|---|---|---|
| net-param hash on the warm-start | src/mantis/model/identity.py::net_param_hash, ::state_dict_param_hash; src/mantis/train/warmstart.py::apply_bc_warm_start (hash inequality raises WarmStartIdentityError), ::resolve_bc_warm_start, ::maybe_warmstart_gnn_from_bc, ::WARM_START_ROW; src/mantis/train/anchor.py::verify_launch_anchor_pin, ::checkpoint_state_sha256 (F-32 launch pin, derived from identity.warm_start); src/mantis/config/schema/core.py::WarmStartConfig | tests/train/test_bc_warm_start_entry.py (::test_the_warm_started_net_carries_the_declared_checkpoints_hash, ::test_a_checkpoint_that_is_NOT_the_declared_net_is_REFUSED); tests/model/test_net_param_hash_promotion.py; tests/train/test_f32_launch_pin_wiring.py | `gg net_param_hash -- src` → hits in warmstart.py (source hash, then the live hash), identity.py def, anchor.py, worker_sweep.py, acceptance_witness.py |
| served-sims exactness | crates/mantis-selfplay/src/runner/search_drive.rs::run_mcts_search (PUCT leaf-batch clamp); crates/mantis-search/src/mcts/seq_halving.rs::considered_visits_sequence (Gumbel: sequence length == num_simulations); runner counter `max_sims_per_search` (crates/mantis-selfplay/src/runner/{mod.rs,stats.rs}) | crates/mantis-selfplay/tests/served_sims_exact.rs (6 tests: r6/r8 at 50 and 600, both kinds at 64 and 320); seq_halving.rs `#[cfg(test)]`; tests/bridge/test_runner_derived_means.py | `gg -i "served.sims" -- crates` → served_sims_exact.rs header "53.46 served sims/move against n_simulations: 50"; seq_halving.rs header "49 of 50 and 599 of 600" |
| the suite's conformance sections | tests/model/conformance/** (11 test modules + _corpus.py + conftest.py: T1 window-frame, T6/T7 perf floor, memory envelope, T9 config partition, T11 graves (test_arch_reachability_and_graves), selector, symmetry-claim, roster guard, codecs-unarmed, legal-move coverage, no-second-arch-kind-table, leaf-forward harness); crates/mantis-search/tests/search_kind_conformance.rs (header "SUITE V2 SECTION") | self-pinning; 156 `def test_` in tests/model/conformance, 11 `#[test]` in search_kind_conformance.rs | `git ls-files \| grep -i conformance` → 15 paths. AMBIGUOUS, conformance-named but outside "the suite": tests/train/test_checkpoint_conformance.py (Suite A, T-CK-01..33), tests/train/test_trainer_seam_conformance.py (SEAM_MATRIX), tests/train/test_no_phantom_seam_member.py |
| 1-in-1 collate checks | src/mantis/selfplay/graph_collate.py::_canary_should_run, ::reset_semantic_canary, ::collate_graph_batch, ::_check_structural, ::EdgeGeometryCheck, ::GraphContractError (+ ::EdgeAttrGeometryMismatch etc.); `collate_check_period=1` call sites in src/mantis/selfplay/pool.py (InferenceServer build) and src/mantis/eval/worker.py (2 engines: ::run_round candidate, ::_play_gate_block best-anchor); src/mantis/selfplay/inference_server.py::InferenceServer (collate_check_period), ::_EdgeGeometryChecker; crates/mantis-bridge/src/graph_contract.rs (verify_edge_geometry pyfunction); crates/mantis-graph/src/lib.rs::verify_contract | tests/eval/test_f816_37_instrument.py::test_every_collate_path_asks_for_one_in_one, ::test_period_one_runs_the_semantic_layer_on_every_batch; tests/selfplay/test_edge_geometry_checker_thread.py; tests/selfplay/test_graph_collate_adv.py | `gg -i "1-in-1" -- src tests` → pool.py, inference_server.py, acceptance_witness.py, test_f816_37_instrument.py, test_edge_geometry_checker_thread.py |
| arena legality | src/mantis/arena/match.py::_play_one_game ("THE LEGALITY BOUNDARY": is_legal checked before apply_move, an illegal move forfeits), ::IllegalOpeningError (a book move that will not replay is fatal), ::play_paired_match; src/mantis/arena/adjudicate.py::TERMINAL_FORFEIT, ::TERMINAL_REASONS; crates/mantis-bridge/src/board.rs::is_legal | tests/arena/test_legality_boundary.py (8 tests) | `gg -i legal -- src/mantis/arena` → match.py IllegalOpeningError + the boundary comment |
| finite-gradient guard | src/mantis/train/losses.py::clip_and_step (a non-finite pre-clip norm runs zero_grad and skips optimizer.step); src/mantis/train/trainer/core.py::Trainer.train_step_from_graph_batch (nonfinite_loss_microbatches / nonfinite_grad_steps counters, refusal reasons nonfinite_gradient / no_contributing_microbatch); src/mantis/monitor/rules.py::check_nonfinite_loss; the isfinite filters in src/mantis/train/coordinator/step.py | tests/train/test_finite_gradient_guard.py; tests/train/test_nonfinite_guard.py | `gg -i nonfinite -- src/mantis/train` → trainer/core.py counters, losses.py clip_and_step |
| resume bundle round-trip | src/mantis/train/bundle.py::publish_bundle, ::read_manifest, ::verify_bundle, ::complete_bundles, ::newest_complete_bundle, ::prune_bundles, ::BundleManifest, ::atomic_write; src/mantis/train/resume_state.py::ResumeState, ::RingRef, ::write_resume_state, ::load_resume_state, ::verify_ring, ::capture_rng_streams, ::restore_rng_streams; src/mantis/run.py::_restore_resume_state; src/mantis/train/coordinator/step.py (publishes through bundle.publish_bundle); src/mantis/train/checkpoints.py::resume_trainer, ::load_checkpoint; src/mantis/train/orchestrator.py::build_resume_config_overrides | tests/train/test_resume_bundle.py, test_periodic_bundle.py, test_resume_ring_roundtrip.py, test_resume_state.py, test_resume_semantics.py (+ fixture train/resume_goldens.json), test_firing_halt_resume_witness.py, test_resume_owned_paths.py, test_checkpoint_conformance.py | `gg -i "resume.bundle" -- src` → bundle.py docstring "The resume BUNDLE — checkpoint + ring + sidecar, published by a manifest last" |
| gate pair statistics | src/mantis/eval/aggregate.py::pair_units, ::pair_bootstrap_wr_ci, ::_pair_key, ::_unit_key, ::_traj_key, ::_distinct_per_pair, ::aggregate_gate, ::GateAggregate, ::gate_promotion_decision, ::aggregate_sequential_gate; src/mantis/arena/books.py::round_openings | tests/eval/test_gate_pair_statistics.py; tests/eval/test_gate_sequential.py | test header: "The gate's CI must resample opening PAIRS, not games"; run_config_schema.md v31: "The gate pair statistic named in the protected set is unchanged in meaning" |
| F-816-37 dump-on-fire | src/mantis/selfplay/collate_dump.py::write_collate_dump; src/mantis/selfplay/inference_server.py::InferenceServer._dump_collate_failure; src/mantis/selfplay/pool.py::_collate_dump_target; src/mantis/eval/worker.py::_collate_dump_target; src/mantis/train/coordinator/dispatch.py::_dump_train_collate; src/mantis/diagnostics/f816_37_rate_bar.py (rate-bar CLI over collate_dumps) | tests/eval/test_f816_37_instrument.py; tests/train/test_f816_37_train_path_dump.py; tests/diagnostics/test_f816_37_rate_bar.py; tests/tools/test_run_dashboard.py (dump panel) | `gg -l "collate_dump\|write_collate_dump" -- tests`; CARDS.md F-816-37 row "1-in-1 eval-path instrument with dump-on-fire (protected set)" |
| strength_floor | src/mantis/eval/floor_gate.py::evaluate_strength_floor, ::probe_measurements, ::StrengthFloorVerdict, ::FLOOR_PROBE_VARIANT; src/mantis/eval/worker.py::run_round (PHASE 0 floor_probe), ::_play_floor_probe; src/mantis/eval/pipeline.py::emit_strength_floor; src/mantis/config/resolve/eval_posture.py::resolve_strength_floor, ::StrengthFloorSpec; src/mantis/config/schema/core.py::StrengthFloorConfig | tests/eval/test_strength_floor_gate.py, test_strength_floor_refuses_the_round.py, test_strength_floor_verdict_on_the_routed_mapping.py; tests/diagnostics/test_acceptance_witness.py | `gg -l "evaluate_strength_floor\|floor_probe" -- tests` |
| draw-rate abort | src/mantis/train/coordinator/step.py::StepCoordinator._run_hard_abort_gates (draw arm), ::_sample; src/mantis/train/coordinator/config.py::pooled_draw_rate, ::DrawRateAbortLike; src/mantis/monitor/rules.py::check_draw_rate_collapse; src/mantis/config/resolve/draw_rate.py::resolve_draw_rate_abort, ::DrawRateAbortSpec; src/mantis/config/schema/train.py::DrawRateAbortConfig; src/mantis/config/schema/core.py::RunConfig._draw_rate_evidence_bar_within_configured_capacity; src/mantis/config/armed_aborts.py::MANIFEST row `draw_rate_collapse` (REQUIRED); pooled_draw_counts in src/mantis/selfplay/pool.py and instrumentation.py | tests/train/test_drawrate_gate_branch_flipset.py, test_drawrate_abort_threading.py, test_drawrate_gate_capacity.py, test_draw_rate_is_a_fraction.py; tests/selfplay/test_drawrate_pooled_statistic.py; tests/config/test_drawrate_arming_authority.py, test_drawrate_schema_range.py; tests/tools/test_drawrate_arming_surface_named_failure.py | `gg draw_rate -- src/mantis/train/coordinator/step.py` → spec, _sample, pooled_draw_rate, check_draw_rate_collapse, _fire_hard_abort |

### PZ-2 Seam (R321 UNIVERSAL MODEL CONTRACT)

The design of record, `plan/SEAM_V1_DESIGN.md` @ d0c3321, is **NOT IN THIS REPO**. `git ls-files | grep -i seam` returns only test files, and docs/governance/archive/README.md §"Sitting records and plan/" says: "Neither has ever been tracked in this repository". The contract text survives here only in RULINGS.md R321 and in quotes in docs/governance/archive/{rulings_register,RULINGS_ACTIVE}.md. R321's accept bar: a second arch must need NO edits to the trainer, server, arena or config schema outside its own scope.

| member | path(s) / symbol | role |
|---|---|---|
| arch contract / registry | src/mantis/model/arch.py::GnnArch, ::GnnArchV2, ::GnnArchV2SoftPolicy, ::ModelArch, ::ARCH_KINDS, ::ARCH_KINDS_BY_REPRESENTATION, ::INCUMBENT_ARCH_KIND, ::ARCH_KIND_ROW, ::declared_arch_kind, ::arch_from_spec_and_config, ::select_arch, ::UnknownArchKind, ::RepresentationMismatch; the re-exports in src/mantis/model/__init__.py | the ONE arch-kind vocabulary and selector; config key `identity.arch_kind` (src/mantis/config/schema/core.py::IdentityConfig) |
| build dispatch | src/mantis/model/build.py::build_net | the dispatch census that T6/T7 and the envelope tests parse (BUILD_SOURCE); the arch handle travels on net.arch |
| identity | src/mantis/model/identity.py::net_param_hash, ::state_dict_param_hash | R321 part-6 precondition (hash promoted out of diagnostics) |
| stamp / loader | src/mantis/train/checkpoints.py::_ARCH_KINDS, ::_arch_to_dict, ::_arch_from_dict, ::stamped_arch_kind, ::save_checkpoint, ::load_checkpoint; src/mantis/eval/snapshot.py::_ARCH_TYPES | the arch travels with the artifact; one loader (LAW-12); config-less call sites resolve from the stamp (R330) |
| per-arch modules | src/mantis/model/gnn.py::GnnNet, ::GnnDist65ValueHead, ::load_from_bc; gnn_v2.py::GnnNetV2, ::GnnNetV2SoftPolicy, ::RepresentationNetworkV2; gine.py::_GINEConv, ::RepresentationNetwork, ::PolicyHead; dist65.py; value_targets.py::lambda_return_targets, ::scalar_to_hl_gauss (codecs landed UNARMED, R322/R323); amp.py::amp_dtype_for (LAW-06) | the tenants behind the contract |
| caps (config side) | src/mantis/config/schema/core.py::ARCH_SCOPED_KEYS; src/mantis/config/resolve/arch_scope.py::refuse_outside_its_arch; resolve/microbatch.py::resolve_microbatch_caps; resolve/fused_graph_caps.py::resolve_fused_graph_caps; resolve/gnn_widths.py::resolve_gnn_widths; tools/bench_floors.toml | the measured caps (train.microbatch_caps.*, inference.fused_graph_caps.*) are arch-scoped keys |
| caps (declarations) | tests/model/conformance/test_arch_states_its_memory_envelope.py::registered_envelopes, ::MemoryEnvelope, ::REQUIRED_TERMS; test_arch_states_its_perf_floor.py::registered_probes, ::arch_kinds_dispatched, ::check_floor_manifest; test_leaf_forward_throughput_harness.py | each arch states its envelope and floor, derived and never transcribed. The declarations live in the TESTS, not in src |
| encoding registry | crates/mantis-encoding/src/registry.toml ([encodings.gnn_axis_v1], [encodings.gnn_axis_r8]); crates/mantis-encoding/src/registry/mod.rs::lookup, ::lookup_or_panic, ::all_specs, ::parse_encoding_toml, ::registry_sha, ::registry_sha_hex; registry/parse.rs; spec/mod.rs::RegistrySpec; spec/validate.rs; src/mantis/encoding/__init__.py::_registry_sha_handshake; src/mantis/encoding/registry.py::lookup, ::all_specs | identity key; gate 8 (tools/ci_gates/registry_gate.sh); pinned by crates/mantis-encoding/tests/{axis_pin.rs,registry_census.rs}. **`manifests.toml` DOES NOT EXIST**: `git ls-files \| grep -c manifests.toml` → 0 |
| graph wire | src/mantis/selfplay/graph_collate.py::GraphWirePayload, ::GraphBatch, ::graph_wire_from_rust, ::collate_graph_batch, ::GraphContractError family; crates/mantis-graph/src/lib.rs::build_axis_graph, ::verify_contract, ::NODE_FEAT_DIM, ::EDGE_FEAT_DIM; crates/mantis-bridge/src/graph_contract.rs; docs/contracts/graph_wire.md | the batch-level boundary |
| trainer/server protocols | src/mantis/train/coordinator/config.py::TrainerLike, ::ReplayBufferLike, ::GraphRouteBufferLike, ::GridRouteBufferLike, ::RecentBufferLike, ::WorkerPoolLike, ::EvalPipelineLike, ::ClockLike; src/mantis/train/events.py::PoolTelemetryLike | policed by tests/train/test_trainer_seam_conformance.py::SEAM_MATRIX and test_no_phantom_seam_member.py |
| search kind | crates/mantis-search (SearchKind, MAX_CHILDREN_PER_NODE, MAX_ARMED_SIMS*); src/mantis/config/resolve/search.py | suite-v2 section crates/mantis-search/tests/search_kind_conformance.rs; tests/eval/test_search_kind_is_one_selector.py |
| conformance suite | tests/model/conformance/** + search_kind_conformance.rs | see PZ-1 row 3 |
| witnesses | src/mantis/diagnostics/acceptance_witness.py (R327 BC acceptance witness) with tests/diagnostics/test_acceptance_witness.py; tests/model/test_gnn_v2_witnesses.py; tests/model/test_arch_v2_dispatch.py; search_kind_conformance.rs WITNESS block; src/mantis/train/heldout.py::HeldoutSlice (held-out witness, R366(c), ratified by R367) | the proofs that the contract holds |
| detectors | the search_kind_conformance.rs "DETECTOR" block; tests/model/conformance/test_arch_declares_no_symmetry_claim.py; test_no_second_arch_kind_table.py; src/mantis/encoding/resolvers.py (unified state-dict detector; tests/encoding/test_no_shape_sniff_dispatch.py limits it to ONE caller) | these catch drift in the kind or arch |
| goldens | tests/fixtures/** (45 `[[required]]` sha rows in tests/fixtures/manifest.toml, checked by tests/test_fixtures_manifest.py; graph_parity/{inputs.bin,raw/*} are covered by graph_parity/manifest.tsv); Rust: crates/mantis-core/tests/golden_replay.rs, crates/mantis-search/src/mcts/{golden_tests.rs,parity_tests.rs}, crates/mantis-search/tests/{mctx_parity.rs,temperature_parity_golden.rs}, crates/mantis-graph/tests/{graph_parity.rs,fixture_selftest.rs,common/mod.rs}, crates/mantis-selfplay/tests/{graph_child_parity.rs,queue_fuse_pin.rs,replay_hexg.rs,target_export_parity.rs,worker_output_pin.rs}; the grave tests/fixtures/model_graves/hexonet_grave_v1.json (T11) | byte/bit parity |
| contract docs | docs/contracts/*.md (9 files; repo_design §4 "Seam contracts") | a version bump, the doc and its tests change in one commit; gate 13 checks run_config_schema.md |

### PZ-3 Freeze

**There is no freeze manifest and no freeze_verify at HEAD.**
- `git ls-files | grep -i -E "freeze|frozen"` → only tests/data/_frozen_games.py (committed input games; not a manifest).
- `git log --all --diff-filter=A --name-only | grep -i freeze` → only src/mantis/train/pretrain/freeze.py (a layer-freeze helper, deleted with the grid path in 3dd20b4).
- CARDS.md RQ-21 says: "no gate-18 script exists in tools/ci_gates/". The ORACLE_FREEZE*.sha256 registers lived in `wp/` in the migration workspace and were never tracked here (see the archive README and the archive register's R332 landing note).
- R332 LIFTED R118/A-1 (eval/rounds.py). Its guard, tests/eval/test_wr_sealbot_config_only.py, is absent at HEAD.

Still frozen by standing text:

| member | frozen by |
|---|---|
| docs/governance/archive/** | R346(e); archive/README.md "FROZEN … never edited" |
| docs/design/archive/** | R355(f); its README "FROZEN records" |
| docs/audits/archive/** | R355(f); its README "FROZEN" |
| tests/fixtures/** (the manifest rows plus graph_parity/manifest.tsv) | repo_design §8: the manifest test FAILS on drift. A re-mint needs a grant (the manifest's "Freeze amendment FA-1" note) |
| src/mantis/arena/books/{manifest.toml,book_v1_s20260625_p4.json,book_v2_pool_s20260915_p4.json} | LAW-15. The manifest says "FROZEN + sha-pinned", and books.py::paired_openings verifies the sha at load |
| crates/mantis-graph/src/lib.rs | its header: "verbatim single-file port of the frozen axis-graph builder; splitting is barred while the byte-parity gate stands". The gate is crates/mantis-graph/tests/graph_parity.rs |
| crates/mantis-search/src/mcts/policy.rs::get_improved_policy_ls | "The completed-Q math is FROZEN" (golden_bits.txt) |
| crates/mantis-search/src/mcts/golden_tests.rs seeded RNG const | "FROZEN — the golden was captured with" it |
| crates/mantis-selfplay oracle-bank feature (Cargo.toml [features]): tests/target_integrity_postfix.rs + gated legs in target_export_parity.rs / target_wire_carry.rs | "the byte-frozen post-fix oracle bank" |
| crates/mantis-selfplay/src/records.rs | "verbatim behaviour-exact port of the frozen game_runner/records.rs, kept single-file for 1:1 audit" |
| minted configs (PZ-4) | R322: "a repair touching a MINTED row is a HALT"; R327: a row edit needs a FROZEN-FILE GRANT |

Provenance only ("port of the frozen X"), no live freeze rule: crates/mantis-bridge/src/{inference,runner}.rs, crates/mantis-selfplay/src/runner/*, queues/*.

### PZ-4 Minted configs

| file | census | identity (encoding / arch_kind / warm_start) |
|---|---|---|
| configs/run6.yaml | production | gnn_axis_r8 / GnnArchV2 / checkpoints/bc/run6_00006500_ca1afb71.ckpt, reinit [value_head] |
| configs/run7.yaml | production | gnn_axis_r8 / GnnArchV2 / the same BC ckpt, reinit [] |
| configs/run8.yaml | production | gnn_axis_r8 / GnnArchV2 / run7_00042000_46fdb931.ckpt |
| configs/run10.yaml | production | gnn_axis_r8 / GnnArchV2SoftPolicy / run8_00045000_3bdedf76.ckpt, reinit [aux_policy_head] |
| configs/dev_example.yaml | EXEMPT | gnn_axis_v1. Grounds: the gate-12 red-capability demo (LAW-07, ADJ-13 M1 row) |
| configs/smoke_preflight_armed.yaml | EXEMPT | gnn_axis_v1. Grounds: the armed preflight burst oracle (tests/tools/test_preflight_armed_smoke.py) |

- Evidence: `.venv/bin/python -c "from mantis.config.census import production_configs, exempt_config_paths; …"` → `['run10.yaml','run6.yaml','run7.yaml','run8.yaml']`, exempt `['configs/dev_example.yaml','configs/smoke_preflight_armed.yaml']`.
- The census is src/mantis/config/census.py::EXEMPT_CONFIGS, ::production_configs, ::discovered_config_paths, ::exempt_config_paths, ::ConfigCensusError. Discovery goes through src/mantis/config/loader.py::discover_configs.
- Census consumers: src/mantis/config/armed_aborts.py, tools/ci_gates/preflight_mint.py (gate 12), tools/ci_gates/validate_configs.py (gate 7), tests/config/test_config_census.py, and about 12 more pins (`gg -l "production_configs\|EXEMPT_CONFIGS" -- tests`).
- The minter is tools/mint_config.py::main, ::MintRowError, ::HeaderRenderError, ::_delta_line, with template tools/config_templates/dev.yaml. Every config starts with `# minted-by: tools/mint_config.py` and a set of `# delta:` lines. Tests: tests/config/test_mint_row.py, test_mint_and_diff.py, test_mint_header_roundtrip.py, test_minted_values_are_provenance_not_expectation.py, test_eval_config_remint.py.
- configs/run9.yaml was DELETED by R367 (tools/ci_gates/check_tracked_refs.py carries the note).

### PZ-5 Governance (docs/governance/**)

RULINGS.md is canonical from R346, corrects only by annotation, and the next ruling is R368. archive/ is FROZEN.
- docs/governance/LAWS.md, RULINGS.md, STATE.md (rewritten in place), CARDS.md, falsified.md, COMMS_STYLE.md
- docs/governance/archive/README.md, RULINGS_ACTIVE.md, laws.md, rulings_register.md

### PZ-6 Other ruling-named code (touch only with a ruling)

- src/mantis/train/orchestrator.py::RESUME_CHECKPOINT_OWNED_KEYS / ::RESUME_CHECKPOINT_OWNED_PATHS: a golden-pinned contract row (CARD-STYLE-BACKLOG, B-14).
- src/mantis/train/checkpoints.py::apply_config_overrides_f1, ::resume_trainer(declared_keys), `declared_lr`, ::RESUME_DIRECTIVE_KEYS: the F1 defer path. No production caller passes these; handle on contact only (B-14).
- src/mantis/selfplay/pool_push.py::push_dense (called from pool_drain.py): the instrumentation oracle for the drain-parity suite. Deleting it means re-basing 6 oracles and re-pinning drain fixtures (B-15).
- B-15 "on contact": src/mantis/selfplay/utils.py::get_temperature; graph_collate.py::collate_graph_batch(device=None); "the two segment softmaxes" (only graph_collate.py::segment_softmax resolves by name; the second is UNRESOLVED); the double torch.load.
- src/mantis/model/arch.py::INCUMBENT_ARCH_KIND / ::ARCH_KIND_ROW: R323 incumbent pin against every minted config, plus the R336 selector row.
- src/mantis/train/anchor.py::verify_launch_anchor_pin: F-32 SHAPE A, with the pin derived from identity.warm_start (R334).
- src/mantis/train/lifecycle/heartbeat_watchdog.py::poll_once: F-11 SHAPE A producer-liveness (R334).
- src/mantis/config/armed_aborts.py::MANIFEST: gate 12 rows. CARD-EXEMPT-CONFIGS-OPERATOR-CONFIRM awaits a ruling.
- src/mantis/eval/sequential.py (GSPRT) + src/mantis/eval/rounds.py::GateSpec.sequential: contract v31, the promotion rule beside the protected pair statistic.
- crates/mantis-search/src/mcts/mod.rs::MAX_CHILDREN_PER_NODE (=1024): R346(c) interior cap K.
- crates/mantis-selfplay/src/replay/hexg/mod.rs::MAX_STONES + crates/mantis-bridge/src/hexg.rs::max_stones: one Rust owner (R329/R331(ii)).
- src/mantis/bots/sealbot.py::BUILD_ABSENT_MARKER: R324 mechanism marker, read by src/mantis/bots/resolve.py.
- R367 "RATIFIED as landed": Trainer._aux_soft_policy_terms and GnnNetV2SoftPolicy (the soft target); Trainer._policy_head_grad_norms (the weight-envelope witness); src/mantis/config/resolve/aux_soft_policy.py; src/mantis/train/heldout.py and src/mantis/config/resolve/heldout_gap.py.
- src/mantis/model/amp.py::amp_dtype_for: LAW-06 bf16, pinned by tests/model/test_bf16_parity_nulldist.py and test_one_amp_dtype_authority.py.
- src/mantis/eval/errors.py::EvalBrokenReason: the ONE authority for why a round broke (repo_design §11).
- load_pretrained_buffer is DELETED (R326). The absence oracle is tests/train/test_bc_graph_reroute.py (`_BURIED`); do not resurrect it or drop the oracle.
- tools/ci_gates/** incl. test_count_floor.txt and comment_length_floor.txt: the ratchets move one way only (gate 3c, gate 14 / R346(f)).
- tools/bench_floors.toml + rust-toolchain.toml: the 28 floors are attested to rustc 1.97.1, so a toolchain bump invalidates them (LAW-09).

### Glob list
```
src/mantis/model/**
src/mantis/train/warmstart.py
src/mantis/train/anchor.py
src/mantis/train/bundle*.py
src/mantis/train/resume_state.py
src/mantis/train/checkpoints.py
src/mantis/train/orchestrator.py
src/mantis/train/losses.py
src/mantis/train/heldout.py
src/mantis/train/trainer/core.py
src/mantis/train/coordinator/step.py
src/mantis/train/coordinator/config.py
src/mantis/train/coordinator/dispatch.py
src/mantis/train/events.py
src/mantis/train/lifecycle/heartbeat_watchdog.py
src/mantis/run.py
src/mantis/arena/match.py
src/mantis/arena/adjudicate.py
src/mantis/arena/books.py
src/mantis/arena/books/**
src/mantis/eval/aggregate.py
src/mantis/eval/floor_gate.py
src/mantis/eval/sequential.py
src/mantis/eval/worker.py
src/mantis/eval/pipeline.py
src/mantis/eval/snapshot.py
src/mantis/eval/rounds.py
src/mantis/eval/errors.py
src/mantis/selfplay/graph_collate.py
src/mantis/selfplay/collate_dump.py
src/mantis/selfplay/inference_server.py
src/mantis/selfplay/pool.py
src/mantis/selfplay/pool_push.py
src/mantis/selfplay/instrumentation.py
src/mantis/monitor/rules.py
src/mantis/diagnostics/acceptance_witness.py
src/mantis/diagnostics/f816_37_rate_bar.py
src/mantis/encoding/**
src/mantis/config/census.py
src/mantis/config/armed_aborts.py
src/mantis/config/schema/core.py
src/mantis/config/schema/train.py
src/mantis/config/resolve/draw_rate.py
src/mantis/config/resolve/eval_posture.py
src/mantis/config/resolve/arch_scope.py
src/mantis/config/resolve/microbatch.py
src/mantis/config/resolve/fused_graph_caps.py
src/mantis/config/resolve/gnn_widths.py
src/mantis/config/resolve/aux_soft_policy.py
src/mantis/config/resolve/heldout_gap.py
src/mantis/config/resolve/search.py
src/mantis/bots/sealbot.py
crates/mantis-encoding/**
crates/mantis-graph/src/lib.rs
crates/mantis-graph/tests/**
crates/mantis-bridge/src/graph_contract.rs
crates/mantis-bridge/src/board.rs
crates/mantis-bridge/src/hexg.rs
crates/mantis-search/src/mcts/seq_halving.rs
crates/mantis-search/src/mcts/policy.rs
crates/mantis-search/src/mcts/mod.rs
crates/mantis-search/src/mcts/*_tests.rs
crates/mantis-search/tests/**
crates/mantis-selfplay/src/runner/search_drive.rs
crates/mantis-selfplay/src/runner/mod.rs
crates/mantis-selfplay/src/runner/stats.rs
crates/mantis-selfplay/src/records.rs
crates/mantis-selfplay/src/replay/hexg/mod.rs
crates/mantis-selfplay/tests/**
crates/mantis-core/tests/golden_replay.rs
tests/model/conformance/**
tests/model/test_net_param_hash_promotion.py
tests/model/test_gnn_v2_witnesses.py
tests/model/test_arch_v2_dispatch.py
tests/model/test_bf16_parity_nulldist.py
tests/model/test_one_amp_dtype_authority.py
tests/fixtures/**
tests/test_fixtures_manifest.py
tests/arena/test_legality_boundary.py
tests/eval/test_f816_37_instrument.py
tests/eval/test_gate_pair_statistics.py
tests/eval/test_gate_sequential.py
tests/eval/test_strength_floor_*.py
tests/train/test_bc_warm_start_entry.py
tests/train/test_f32_launch_pin_wiring.py
tests/train/test_*finite*_guard.py
tests/train/test_resume_*.py
tests/train/test_periodic_bundle.py
tests/train/test_firing_halt_resume_witness.py
tests/train/test_checkpoint_conformance.py
tests/train/test_trainer_seam_conformance.py
tests/train/test_no_phantom_seam_member.py
tests/train/test_f816_37_train_path_dump.py
tests/train/test_draw*.py
tests/train/test_bc_graph_reroute.py
tests/selfplay/test_edge_geometry_checker_thread.py
tests/selfplay/test_graph_collate_adv.py
tests/selfplay/test_drawrate_pooled_statistic.py
tests/config/test_drawrate_*.py
tests/config/test_config_census.py
tests/config/test_mint_*.py
tests/tools/test_drawrate_arming_surface_named_failure.py
tests/diagnostics/test_acceptance_witness.py
tests/diagnostics/test_f816_37_rate_bar.py
tests/bridge/test_runner_derived_means.py
tests/data/_frozen_games.py
configs/**
tools/mint_config.py
tools/config_templates/**
tools/ci_gates/**
tools/bench_floors.toml
docs/governance/**
docs/contracts/**
docs/design/archive/**
docs/audits/archive/**
```

## 4. Live-caller checklist and the caller-surface index

Every DEAD claim ticks each item (a grep non-result alone is not evidence, R297(b)):

- AST imports
- pyproject entry points
- `python -m` targets in the Makefile, shell scripts, docs and STATE procedures
- subprocess command strings
- importlib, getattr, registry-by-name lookups
- conftest files and pytest plugins
- pyo3 exports used from Python
- config keys read by name
- every gate's tool path
- box and operator-machine procedures named in STATE (dashboard refresh, puller, strix follower, shakedown)

The index below (a read-only map agent, verified at HEAD, headings demoted) is where each channel is searched.


Purpose: before calling anything DEAD, a scout checks every surface below. Items are `path::symbol`, never line numbers.
Env: `.venv/bin/python` (no torch). Gates that import producers (12, 13, the monitor manifest verifier) give FALSE reds here (`No module named 'torch'`);
gates 9, 10, 15, tier census and the key-consumer tests run clean without torch (verified: rc 0 / 10 passed).

### 0. Scout quick checklist (a symbol/file X is DEAD only if ALL of these come back empty)
1. `git grep -n -w X` over the WHOLE tree (not only *.py): docs/, Makefile, *.sh, *.toml, *.yaml, *.service, crates/.
2. String-keyed lookups: `git grep -n -E "[\"']X[\"']"` (getattr-by-string, event names, registry keys, test-name tables).
3. Path strings: `git grep -n -F "<basename of file>"` and the Path-segment form `git grep -n -E "[\"']<dir>[\"']\s*/\s*[\"']<name>"`.
4. Section 7 tables + section 10 CONSUMER_REGISTRY strings + section 11 pins (producer_manifest.yaml, armed_aborts source_pin, tier_declaration.txt, bench_floors.toml).
5. If a TEST is deleted: gate 3c floor (collected 5112 vs floor 4862; below needs `tools/ci_gates/test_count_ratchet_down.txt`), and tier_declaration.txt (a declared test that vanishes = STALE row = red).

### 1. pyproject entry points
Cmd: `sed` over pyproject.toml + crates/mantis-bridge/pyproject.toml; `git grep -E "pytest11|pytest_plugins|entry-points"`.
- `[project.scripts]`: NONE. `[project.entry-points]`: NONE. pytest11 / plugin registrations: NONE. No setup.py/setup.cfg/tox.ini/pytest.ini.
- `[tool.hatch.build.targets.wheel] packages = ["src/mantis"]`; uv workspace member `crates/mantis-bridge` (dist `mantis-engine`).
- maturin: `module-name = "mantis._engine"`, `python-source = "python"`, feature `extension-module` -> ships `crates/mantis-bridge/python/mantis/_engine.pyi` + the .so.
- `src/mantis/__init__.py` uses `pkgutil.extend_path` so `mantis._engine` (wheel) and the editable src package merge.
- pytest: `testpaths=["tests"]`, `addopts="-ra --strict-markers -m 'not integration and not slow'"`, markers `integration`, `slow`. No `-p` plugin anywhere.
- pyright: include `src`,`tools`; EXCLUDE `tools/strix_driver.py` (runs inside vendored strix venv).
- mise.toml: `node = 26.7.0` (sole consumer: pyright in gate 14). rust-toolchain.toml pins 1.97.1.
=> Every CLI is reached via `python -m <module>` or `python tools/<script>.py` (sections 2-5), never a console script.

### 2. Makefile (86 lines, 19 targets; exact target set pinned by tests/test_meta_ci.py::test_the_makefile_dispatches_exactly_the_declared_target_set)
Cmd: `cat Makefile`.
| target | runs |
|---|---|
| build | `uv sync` |
| build.cuda | `uv sync --extra cuda --no-group cpu` |
| build.native | `RUSTFLAGS="-C target-cpu=native" uv sync --reinstall-package mantis-engine` |
| test | `UV_NO_SYNC=1 uv run pytest -m "not integration and not slow"`; `cargo test --workspace --locked` |
| test.integration | `UV_NO_SYNC=1 uv run pytest -m integration` |
| lint | `bash tools/ci_gates/lint_gate.sh --self-test` |
| lint.rust | `cargo clippy --workspace --all-targets --locked -- -D clippy::all` |
| gates / gates.exit | `bash tools/ci_gates/run_all.sh` / `... --with-slow` |
| dashboard | `uv run python tools/run_dashboard.py --events E --out O [--ladder-state L] [--external-points D...]` |
| viewer | `uv run python tools/game_viewer.py --run R... --out O --title T` |
| analyzer | `uv run python tools/position_analyzer.py serve --checkpoints D... [--strix] [--port] [--device] [--threads]` |
| bench / bench.baseline | `cargo bench -p mantis-core --bench smoke_bench --locked -- ...` (baseline adds `--save-baseline local`) |
| check.wasm | `cargo check -p mantis-graph --target wasm32-unknown-unknown --locked` |
| vendor / vendor.sealbot / vendor.strix | `bash tools/vendor_fetch.sh` / `bash tools/vendor_build_sealbot.sh` / `bash tools/vendor_build_strix.sh` |
| clean | `cargo clean`; `rm -rf dist` |
No `python -m` in the Makefile. tools/ paths named: run_dashboard.py, game_viewer.py, position_analyzer.py, ci_gates/lint_gate.sh, ci_gates/run_all.sh, vendor_fetch.sh, vendor_build_sealbot.sh, vendor_build_strix.sh.

### 3. Shell scripts (git ls-files '*.sh' = 8) + CI workflow
- tools/ci_gates/run_all.sh -> every row below (UV_NO_SYNC=1; nothing short-circuits):
  | gate | invocation |
  |---|---|
  | 1 (opt-in `--with-fresh-sync`) | `bash tools/ci_gates/gate_01_fresh_sync.sh` |
  | 2a | `cargo test --workspace --locked` |
  | 2b | `cargo clippy --workspace --all-targets --locked -- -D clippy::all` |
  | 4 | `make check.wasm` |
  | 5 | `make bench` |
  | 3a / 3b | `uv run pytest -m "not integration and not slow"` / `uv run pytest -m integration` |
  | slow (opt-in `--with-slow`) | `uv run pytest -m slow` |
  | 3c | `bash tools/ci_gates/test_count_gate.sh` |
  | 7 | `uv run python tools/ci_gates/validate_configs.py` |
  | 8 | `bash tools/ci_gates/registry_gate.sh` |
  | 9 | `uv run python tools/check_import_dag.py src/mantis` |
  | 11 | `uv run python tools/ci_gates/silent_encoding_gate.py` |
  | 12 | `uv run python tools/ci_gates/preflight_mint.py --audit-only` |
  | 13 | `uv run python tools/ci_gates/contract_doc_gate.py` |
  | 14 | `bash tools/ci_gates/lint_gate.sh --self-test` |
  | 15 | `uv run python tools/ci_gates/r8_header_gate.py` |
  | 16 | `uv run python tools/ci_gates/encoding_io_gate.py` |
  | 6 | `python3 tools/ci_gates/artifact_gate.py --base $BASE_REF` |
  | 10 | `python3 tools/ci_gates/check_tracked_refs.py` |
  | 17 | `python3 tools/ci_gates/rule7_gate.py --base $BASE_REF` |
- gate_01_fresh_sync.sh -> git clone . tmp; `uv sync --locked`; inline python: `_engine.__doc__`, `_engine.registry_sha_hex()`, `_engine.Board()`.
- registry_gate.sh -> inline python: `mantis._engine.registry_sha`, `mantis.encoding::{EncodingRegistryError,_registry_sha_handshake,_resolve_registry_toml}` (private names used ONLY by string-embedded code here — AST scans of .py miss them); echoes the deferred `python -m mantis.encoding audit` sub-check.
- lint_gate.sh -> `uv run python tools/ci_gates/comment_lint.py` (FIRST), `uv run ruff check .`, `uv run pyright --outputjson`, inline `python -c json` parsers; self-test pipes into ruff with `--stdin-filename src/mantis/_lint_gate_selftest.py`.
- test_count_gate.sh -> `uv run pytest --collect-only -q -m ''`; reads tools/ci_gates/test_count_floor.txt (4862) and optional test_count_ratchet_down.txt (absent at HEAD); `uv run python tools/ci_gates/tier_census.py` (second arm).
- vendor_fetch.sh -> inline python3 (tomllib vendor/pins.toml; git clone/fetch/checkout; `git apply` vendor/patches/sealbot.patch).
- vendor_build_sealbot.sh -> python3 tomllib pin read; `uv run --no-project --with pybind11 --with setuptools python setup.py build_ext --inplace` in vendor/external/sealbot/current.
- vendor_build_strix.sh -> `uv sync --group cpu --all-packages --extra train` in vendor/external/hexo-strix; venv python `-c 'import hexo_rs, hexo_a0.model, torch_geometric'`.
- .github/workflows/ci.yml (remote CI SUSPENDED, file still parsed by tests/tools/test_local_gate_runner.py + tests/test_meta_ci.py): same gate rows as run_all.sh PLUS `uv run python tools/ci_gates/pytest_step_summary.py <junit> --title ...` (ci.yml is that tool's ONLY caller; 0 tests, 0 docs).
- Non-.sh launcher: tools/ladder/vps/ladder-bot@.service -> `ExecStart=/opt/mantis/.venv/bin/python tools/ladder_bot.py --backend %i --server ... --work-dir ... --threads ...`.

### 4. `python -m` targets (cmd: `git grep -n -E "python[0-9.]* -m [a-z_.]+"` over docs tools src tests configs Makefile .github README CLAUDE; plus `"-m", "<mod>"` list forms)
| module | exists at HEAD | citing files (list-form subprocess callers marked *) |
|---|---|---|
| mantis.run | yes | STATE.md, CARDS.md, RULINGS.md, RUN8/RUN10_PREREG, archive docs, src/mantis/run.py, tools/ci_gates/rule7_gate.py (fixture string), tests/test_run_launcher.py*, tests/tools/test_preflight_mint_process.py*, tests/diagnostics/test_worker_sweep_reachability.py*, tests/test_run_one_authority.py, tests/monitor/test_the_one_logging_handler_is_installed.py, tests/tools/test_rule7_gate.py |
| mantis.monitor.supervise | yes | repo_design.md, archive docs, src/mantis/monitor/supervise.py, src/mantis/train/lifecycle/heartbeat_watchdog.py, tests/monitor/test_supervisor_config_witness.py*, tests/monitor/test_supervisor_signal_posture.py* |
| mantis.train.lifecycle.arm_exec | yes | src/mantis/monitor/supervise.py::spawn_child* via `mantis.monitor.heartbeat::PARENT_DEATH_ARM_EXEC_MODULE`; tests/monitor/test_arm_exec_trampoline.py* |
| mantis.encoding (`__main__` -> audit.main) | yes | docs/contracts/registry.md, repo_design.md, src/mantis/encoding/{__main__,audit}.py, tools/ci_gates/registry_gate.sh (echo), tools/hardcode_scan.py |
| mantis.eval.worker | yes | tools/strength_frontier.py::run_cell* (`[python,"-m","mantis.eval.worker",spec,result]`), STRENGTH_FRONTIER_1 doc |
| mantis.diagnostics.fusion_calibrate | yes | src/mantis/config/armed_aborts.py, src/mantis/config/resolve/fused_graph_caps.py, tools/config_templates/dev.yaml (comment), tests/diagnostics/test_fusion_calibrate_refusals.py* |
| mantis.diagnostics.ring_audit | yes | RUN8_PREREG, RUN10_PREREG, STATE.md (no `python -m` prefix there) |
| mantis.diagnostics.ring_reader | yes | RUN8_PREREG, STATE.md |
| mantis.diagnostics.worker_sweep | yes | self, tools/worker_sweep_plan.toml |
| mantis.diagnostics.eval_child_memory | yes | docs/contracts/event_manifest.md, self |
| mantis.diagnostics.cuda_build_guard | yes | self, tests/diagnostics/test_cuda_build_guard.py* |
| mantis.diagnostics.mirror_receipts / acceptance_witness | yes | self only |
| mantis.data.bootstrap_encode | yes | self, src/mantis/train/pretrain/graph_route.py |
| mantis.data.corpus_analysis | yes | self only |
| mantis.train.pretrain (`__main__`) | yes | pretrain/{__init__,__main__,cli}.py, AUDIT_2026-09-09 (archive) |
| mantis.dash | NO | docs/design/observatory_design.md, archive rulings_register (historical) |
| mantisnet.klent.run | NO (foreign) | docs/audits/archive/AUDIT_2026-09-09.md |
| mantis.train | no `__main__` | tests/monitor/test_supervisor.py (fake child_argv, never executed) |
| pytest | - | tests/test_bare_pytest_tier.py* |
Runnable-but-uncited: src/mantis/diagnostics/f816_37_rate_bar.py has a `__main__` guard, no `python -m` citation (lib consumers: tools/dashboard/health.py, its test). src/mantis/train/pretrain/cli.py has its own guard (reached via package `__main__`).
Script-path form (`python tools/X.py`) citations exist for every tools/*.py except hardcode_scan.py (loaded by path, sec 7) — see sec 11 table.

### 5. STATE.md procedures (docs/governance/STATE.md + its pointer RUN10_PREREG_2026-09-21.md §6)
Cmd: code-span extraction from STATE.md; each path checked vs `git ls-files`, each `mantis.*` via `importlib.util.find_spec` + `git grep "^def <attr>"`.
| procedure | names | resolves at HEAD |
|---|---|---|
| dashboard refresh (box, every 10 min, loopback 8766) | `tools/run_dashboard.py` + `tools/dashboard/`, `tools/game_viewer.py` + `tools/viewer/`, `make dashboard/viewer` | yes (the 10-min loop + server are box-local, not in tree) |
| puller (operator machine, systemd user unit `mantis-puller-run8.service`, 600 s) | `tools/mirror_pull.py` (--source --mirror --run-id --mirror-id --once --interval-sec); receipts via src/mantis/util/mirror_receipts.py | yes (unit file NOT in tree) |
| strix follower | `tools/strix_follower.py --follow --no-promotions --cadence 36000 --unit equal_work --poll-sec 300`; `--once <ckpt> --unit net_only`; loads tools/strength_frontier.py by path -> `python -m mantis.eval.worker`; strix via tools/strix_driver.py in the vendored venv (`make vendor`, `make vendor.strix`) | yes; wrappers `/workspace/oc7/chain_follower_run8.sh`, `box_follower_run8.sh` NOT in tree (box-local) |
| shakedown | `run_shakedown.sh` (3 h cap), `shakedown8.yaml` minted via tools/mint_config.py + `tools/config_diff.py --expect run_id`, run = `mantis.run` (under `mantis.monitor.supervise`) | run_shakedown.sh + shakedown8.yaml NOT in tree (box-local); tools yes |
| preflight (MANUAL mint gate) | `tools/ci_gates/preflight_mint.py --config --burst-steps --out-dir --timeout-sec` (+ `preflight_mint_parent.py` by path); stamp src/mantis/config/preflight_stamp.py | yes |
| twin | `python -m mantis.run --inherit-preflight <run>.yaml` | yes (src/mantis/run.py argparse `--inherit-preflight`) |
| witness | `tests/selfplay/test_qsigma_rescale_reaches_target.py` (QSigma pin) + ring stats; `mantis.diagnostics.ring_audit <ring> --bands <prereg md> --events <jsonl>` | yes |
| mint | `tools/mint_config.py --template dev [--set k=v] [--mint-row k=v] --out`; `tools/config_diff.py --from-header / --expect` | yes |
| bench_server (admission bench) | `tools/bench_server.py` (B 64, 200 s, `--device cuda`) | yes |
| ring_audit / ring_reader | `mantis.diagnostics.ring_audit`, `mantis.diagnostics.ring_reader`, `.tactics` | yes (all three) |
| probe1 | `tools/probe1.py netread` + `tools/probe1/` (subcmds decompose/netread/gap/proofs/spread/swa) | yes |
| analyzer | `make analyzer` -> `tools/position_analyzer.py serve` + `tools/analyzer/` | yes |
| ladder | `tools/ladder_bot.py`, `tools/ladder/{openings,receipt,session,backends}.py`, systemd `tools/ladder/vps/ladder-bot@.service` | yes |
| build/vendor on box | `make build.cuda`, `make vendor`, `make vendor.strix`, `make gates.exit` | yes |
| other STATE names | `mantis.config.census.production_configs`, `mantis.train.mixing._steps_budget` | yes (both defs present) |
| NAMED AS DELETED | `mantis.eval.{ladder,bt,channel_health}` (R362), `configs/run9.yaml` (R367; whitelisted DISSOLVED in gate 10) | absent — correct |
| box-local only | `chain_tt_ab.sh`, `chain_strix_15k.sh`, `box_resume_run7.sh`, `aux_burst.py` (RUN10 prereg, kept off-tree by design) | absent — by design |

### 6. subprocess / exec targets (cmd: `git grep -n -E "subprocess\.|Popen\(|check_call|check_output|os\.system|sys\.executable"` src tools tests)
src/ (9 sites):
- src/mantis/bots/strix.py::DriverTransport -> `<strix venv python> tools/strix_driver.py` (path tuple `_DRIVER=("tools","strix_driver.py")`); same class reused by tools/analyzer/strix.py (`DRIVER`) and tools/ladder/backends.py.
- src/mantis/monitor/supervise.py::spawn_child -> `sys.executable -m mantis.train.lifecycle.arm_exec -- <argv>`; arm_exec.py `os.execvp(child)` (argv after `--` on the supervise CLI, i.e. `python -m mantis.run ...`).
- src/mantis/eval/pipeline.py::EvalPipeline -> multiprocessing `get_context(spawn|forkserver).Process(target=_worker_entry)` -> lazy `mantis.eval.worker.worker_main` (reachability via string-free AST, but import is LAZY inside the function).
- git rev-parse: data/bootstrap_encode.py::encode_corpus, data/corpus_io.py::_resolve_git_commit, diagnostics/fusion_calibrate.py::_git_head, diagnostics/worker_sweep.py (git helper), util/git.py.
tools/:
- tools/strength_frontier.py::run_cell -> `python -m mantis.eval.worker`; tools/strix_follower.py loads strength_frontier by path and calls `run_cell` (python=sys.executable).
- tools/ci_gates/preflight_mint.py::_child_argv -> re-execs ITSELF (`sys.executable abspath(__file__) --_boot`); git rev-parse.
- tools/mirror_pull.py::_rsync -> `rsync -a --partial`.
- git only: ci_gates/{artifact_gate,check_tracked_refs,comment_lint,rule7_gate}.py, vendor_fetch.sh.
tests/ (43 files; 117 call lines), grouped by executed target:
- tools/ci_gates/preflight_mint.py: config/test_allocator_posture_authority, tools/test_preflight_{armed_smoke,child_convergence,mint,mint_process,pfc_cards}; preflight_mint_parent.py + validate_configs.py: tools/test_preflight_mint_process.
- tools/mint_config.py: config/test_{config_diff_from_header,mint_and_diff,mint_header_roundtrip,mint_row}, monitor/test_supervisor_config_witness, tools/test_preflight_mint_process; tools/config_diff.py: config/test_{config_diff_from_header,mint_and_diff,mint_row}.
- bash gates: run_all.sh (tools/test_local_gate_runner), test_count_gate.sh (tools/test_test_count_gate, tools/test_tier_census), registry_gate.sh (tools/test_registry_gate), vendor_build_sealbot.sh (bots/test_sealbot_vendored, tools/test_vendor_build_sealbot), vendor_fetch.sh (tools/test_vendor_fetch_idempotent, tools/test_vendor_pins_sealbot).
- python gates: artifact_gate (tools/test_artifact_gate), contract_doc_gate (tools/test_contract_doc_gate), tier_census (tools/test_tier_census), check_import_dag (tools/test_check_import_dag, diagnostics/test_worker_sweep_reachability), mint_opening_book (arena/test_books, tools/test_mint_opening_book).
- `-m` modules: see sec 4 (*). `python -c <code>` snippets that import by STRING: config/test_resolve_amp, monitor/test_monitor_census, monitor/test_supervisor_spawn_contract, test_launch_smoke, test_run_pdeathsig, train/test_parent_death_signal, bots/test_sealbot_vendored, monitor/test_arm_exec_trampoline, config/test_monitor_config_single_authority, test_run_one_authority (18 files embed `from mantis... import` in strings: `git grep -n -E "[\"'](import mantis|from mantis[a-zA-Z_.]* import)" tests`).

### 7. Dynamic lookup (cmd: `git grep -n -E "import_module|spec_from_file_location|find_spec|getattr\("` src tools; AST scan for str-keyed dicts of callables)
Import-by-name / by-path:
- src/mantis/monitor/manifest.py::_resolve_dotted / _module_source -> every `module`/`symbol`/`feeds_from`/`event_literal` in src/mantis/monitor/producer_manifest.yaml (16 rows: draw_rate_collapse grad_norm_hard_abort heartbeat.{train_step,inference_dispatch,selfplay_drain,eval_round} persist_fatal selfplay_stall actor_lag resolved_config disk_guard warn.training_step_alerts target_integrity_counters batch_fill_pct inference_batch_timing terminal_eval_broken; modules cited: mantis.eval.{errors,pipeline}, mantis.monitor.{heartbeat,sink}, mantis.run, mantis.selfplay.{inference_server,pool,pool_drain,pool_hooks}, mantis.train.{actor_sync,checkpoints,events}, mantis.train.coordinator.{config,drain,step}, mantis.train.lifecycle.disk_guard) + each row's `producer_test: tests/..::test_fn`.
- tools/ci_gates/contract_doc_gate.py::_symbol_exists -> every `mantis.*` token in docs/contracts/run_config_schema.md.
- src/mantis/encoding/audit.py -> loads tools/hardcode_scan.py by path (`_section_hardcode`).
- src/mantis/bots/sealbot.py -> vendored game module + `minimax_cpp*.so` by path.
- tools/ci_gates/preflight_mint.py -> preflight_mint_parent.py as `_preflight_mint_parent`.
- tools/{run_dashboard,game_viewer,position_analyzer,probe1,ladder_bot}.py -> load package tools/{dashboard,viewer,analyzer,probe1,ladder} by path, then `import_module("<pkg>.cli")` (+ `dashboard.health`).
- tools/strix_follower.py -> tools/strength_frontier.py by path.
- tests/tools/conftest.py::load_tools_package("dashboard"|"viewer"|"ladder"|"analyzer"), `_load_puller` (tools/mirror_pull.py), `importlib.import_module("analyzer.engines")`; ~60 tests/tools/test_*.py load tools/*.py via spec_from_file_location (top targets: preflight_mint.py 19, mint_config.py 8, strength_frontier.py 6, validate_configs.py 6).
- tests/config/test_radius_removed.py -> `exec(f"from mantis.config import {name}")`.
Name-keyed tables (value set -> where defined):
- src/mantis/model/arch.py::ARCH_KINDS {GnnArch, GnnArchV2, GnnArchV2SoftPolicy} <- config `identity.arch_kind` + checkpoint stamps; aliases eval/snapshot.py::_ARCH_TYPES, train/checkpoints.py::_ARCH_KINDS; ARCH_KINDS_BY_REPRESENTATION, INCUMBENT_ARCH_KIND {graph: GnnArch}; schema/core.py::SOFT_POLICY_ARCH_KINDS; checkpoints.py::_LEGACY_BY_REPRESENTATION, _SYNTH_ARCH_SCOPED.
- Encodings: src/mantis/encoding/registry.py::_REGISTRY_CACHE from `_engine.all_specs()` <- crates/mantis-encoding/src/registry.toml rows {gnn_axis_v1, gnn_axis_r8}; configs name them in `identity.encoding`.
- src/mantis/bots/resolve.py::_KNOWN_KINDS {random, sealbot, strix}, SKIP_REASON_MARKERS; src/mantis/config/resolve/search.py::SEARCH_KINDS {puct, gumbel} (-> Rust via MCTSTree.configure_search / SelfPlayRunnerConfig.search_kind).
- src/mantis/selfplay/buffers.py::_RAW_FOR {GRAPH: HexgBuffer}; src/mantis/model/amp.py::_STRING_TO_TORCH {fp16,bf16}; src/mantis/monitor/rules.py::WARN_RULE_INPUTS; src/mantis/diagnostics/ring_audit.py::_OPS (band ops parsed from prereg markdown).
- getattr-by-string counters: src/mantis/selfplay/pool_hooks.py (15 SelfPlayRunner counters incl. worker_panics, export_offwindow_mass_moves, inference_failures_total), src/mantis/train/coordinator/step.py (_TRAINER_COUNTERS, rstats names), src/mantis/monitor/supervise.py::_OVERRIDABLE, src/mantis/config/armed_aborts.py::_dotted (row `config_path` strings), config/schema/core.py (section/field names).
- tools: ladder_bot `--backend {mantis,strix}` (systemd `%i`), analyzer cli {serve, once}, probe1 cli {decompose, netread, gap, proofs, spread, swa}, strix_follower `--unit {equal_work, net_only}`.
- Event-name consumers (string keys): tools/dashboard via `rec.rows/last/series("<event>")`: disk_alert disk_free eval_channel_health eval_round_complete eval_round_device_memory eval_strength_floor game_complete hard_abort iteration_complete monitor_gates resolved_config resume_state_persisted run_boot_identity run_segment_started trainer_step training_alert training_step; plus dashboard/reader.py::HEAVY_FIELDS. An "unused" event emission may be read here.

### 8. conftest files and pytest plugins (cmd: AST walk of `git ls-files '**/conftest.py'`)
No `pytest_plugins` anywhere; no `-p` in addopts/Makefile/CI; one hook: tests/conftest.py::pytest_report_header (prints TIER:).
- tests/conftest.py: fixtures _reseed(autouse), _restore_signal_dispositions(autouse), seeded_libs, smoke_run_config, mk_graph_buffer, preflight_stamped, synthetic_run_dir; helpers _deep_merge, make_run_config_from_minted.
- tests/bridge/conftest.py: panic_exception(session).
- tests/config/conftest.py: production_config.
- tests/model/conformance/conftest.py: derived.
- tests/monitor/conftest.py: spy_sink, fake_clock, exit_spy, snapshot_spy, mutable_counter.
- tests/selfplay/conftest.py (session): collate_expectations, drain_goldens, encoding_resolve_golden, runner_config_goldens, pure_function_battery, _payload_bank, collated_golden, hotpath_golden, wire_geometry, collect_data_input, dense_pushed, graph_pushed, graph_rows_input; payload_fields; helper _load_npz.
- tests/tools/conftest.py: preflight_budget_sec, preflight_harness_ceiling_sec, _preflight_probe_path_is_not_left_in_the_tree(autouse,session), local_puller(module), dashboard, viewer, ladder, analyzer, mint_stamp, mantis_engine, positions; helpers load_tools_package, load_dashboard_package, mint_analyzer_stamp, _load_puller, _run_dirs_under, _sweep.
- tests/train/conftest.py: spy_sink, fake_clock, tiny_arch, tiny_net, optim_scaler_sched, full_graph_net(session), full_graph_state, full_train_hparams, valid_config, invalid_config, metadata_kwargs, mk_config, mk_meta, mk_optim, resume_goldens/legacy_shapes/anchor_key_set(session); helpers make_tiny_arch, make_run_config, make_full_train_hparams, make_metadata_kwargs, make_optim_scaler_sched, _make_*_block.
Test helper modules imported by bare name (pytest rootdir insertion): tests/_drivable.py, tests/config/_consumer_resolver.py, tests/data/_frozen_games.py, tests/model/_bf16_parity.py, tests/model/conformance/_corpus.py, tests/selfplay/{_fused_graph_harness,_retired_batch_fields,_wire_geometry}.py, tests/tools/_ladder_stub.py, tests/train/{_coordinator_pool,_microbatch_harness,_warmstart_config}.py, tests/util/_cpu_budget.py.
Fixture names are resolved by pytest by PARAMETER NAME — grep `def test_.*\b<fixture>\b` before calling a fixture dead.

### 9. PyO3 exports (mantis._engine)
Cmd: registrations `grep add_class|add_function|m.add(` in crates/mantis-bridge/src/*.rs; runtime `dir(mantis._engine)` from the built .so (28 names); per-name `git grep -l -P "\b<name>\b" -- src tools tests '*.py'` (methods: `\.<name>\b`, then `-w` recheck for zeros).
Module-level (28 at runtime) — files referencing:
Board 78 · HexgBuffer 72 · all_specs 30 · MCTSTree 16 · RegistrySpec 15 · InferenceBatcher 13 · SelfPlayRunnerConfig 11 · DEFAULT_CLUSTER_THRESHOLD 10 · registry_sha 8 · SelfPlayRunner 7 · derived_hexg_visit_capacity 7 · verify_edge_geometry 7 · HEX_AXES 6 · WIN_LENGTH 6 · max_stones 6 · GraphWire 5 · TacticalSolver 4 · registry_sha_hex 4 (+2 .sh gates) · GraphTargets 3 (tests only: bridge/test_pyclass_roundtrips, bridge/test_surface, train/test_resume_ring_roundtrip) · WireAlreadyConsumed 3 (selfplay/graph_collate.py + 2 tests) · graph_row_outcome 2 (data/bootstrap_encode.py + test) · SelectionDesync 1 (tests/selfplay/test_selection_desync_is_named.py) · mcts_max_armed_sims 1 / mcts_max_armed_sims_gumbel 1 (config/schema/selfplay.py) · mcts_pool_overflow_count 1 / take_mcts_pool_overflow_count 1 (tests/bridge/test_surface.py only)
**ZERO Python references (candidate-dead, not concluded):** `mcts_omitted_prior_stats`, `take_mcts_omitted_prior_stats` (crates/mantis-bridge/src/utils.rs; Rust side `mantis_search::omitted_prior_stats` is used by crates/mantis-search/tests/search_kind_conformance.rs).
Methods with ZERO Python refs (word-level, any string): Board.{terminal_value_to_move, threat_moves, get_threats}, RegistrySpec.builder_impl_required, MCTSTree.{last_search_stats, run_simulations_cpu_only, root_raw_value}, SelfPlayRunner.positions_dropped.
Test-only / string-only methods: Board.set_legal_move_radius (only a `grep crates/` test), MCTSTree.apply_dirichlet_to_root (string in tests/config/test_no_bridge_default_shadows_a_config_key.py), RegistrySpec.{plane_layout, has_pass_slot, sym_table_id, n_cells, chain_stride, aux_stride, n_source_planes, n_actions} (tests/encoding/test_inv22_spec_parity.py by string; some in tools/hardcode_scan.py), plus several InferenceBatcher/SelfPlayRunner/MCTSTree accessors used only by tests/bridge/*.
Stub drift (both twins, pinned identical by tests/bridge/test_engine_stub_twins_agree.py, NOT checked vs runtime): pyi declares `MY_STONE_PLANE, OPP_STONE_PLANE, MOVES_REMAINING_PLANE, PLY_PARITY_PLANE` which the runtime does NOT export (retired R346(f)); runtime exports RegistrySpec.{cluster_threshold, cluster_window_size} and SelfPlayRunner.{inference_failures_total, max_sims_per_search} which the pyi omits (all four are used from Python).

### 10. Config keys read by name
Mechanism: `mantis.config.schema.leaf_paths(RunConfig)` (160 leaves; doc states 160) is bijected against CONSUMER_REGISTRY in tests/config/test_every_key_has_consumer.py AND the deliberate twin tests/config/test_every_key_has_consumer_p2.py; each value string is token-resolved against every name defined in src/ + crates/ by tests/config/_consumer_resolver.py::defined_names (so a symbol CITED in a registry string is a live dependency: deleting it reds `test_every_registry_string_names_symbols_that_exist`). tests/config/test_consumer_citation_arrows.py checks process-entry arrows. Gate 13 checks docs/contracts/run_config_schema.md key/symbol citations; gate 12 checks src/mantis/config/armed_aborts.py rows' `config_path` against production configs (census = configs/ minus src/mantis/config/census.py EXEMPT rows).
Scout commands:
- key -> cited reader: `grep -n '"<dotted.key>"' tests/config/test_every_key_has_consumer.py`
- reader actually reads it: `git grep -n -E "\.<leaf>\b" -- src/mantis crates tools`
- symbol cited by a registry string: `git grep -n -w <symbol> -- tests/config/test_every_key_has_consumer*.py src/mantis/monitor/producer_manifest.yaml src/mantis/config/armed_aborts.py docs/contracts/run_config_schema.md`
- run (torch-free, ~1 s): `.venv/bin/python -m pytest -q -p no:cacheprovider tests/config/test_every_key_has_consumer.py tests/config/test_every_key_has_consumer_p2.py tests/config/test_consumer_citation_arrows.py`
- list leaves: `.venv/bin/python -c "from mantis.config.schema import RunConfig, leaf_paths; print('\n'.join(leaf_paths(RunConfig)))"`

### 11. Other non-AST callers
Rust:
- include_str!: crates/mantis-encoding/src/registry/mod.rs::REGISTRY_TOML <- src/registry.toml; crates/mantis-search/src/mcts/golden_tests.rs::GOLDEN_BITS <- tests/fixtures/search_golden/completed_q/golden_bits.txt (cfg(test)); crates/mantis-selfplay/src/replay/sym.rs (test) <- hexg/sample.rs; crates/mantis-selfplay/tests/rotation_parity.rs::SEARCH <- src/runner/search_drive.rs (source-text pin). NOTE: crates/mantis-bridge/pyproject.toml comment still says `manifests.toml` is include_str!-ed — that file is gone (AUDIT-1 F-36), comment stale.
- No build.rs anywhere. No [[bin]]/[[example]]/[[test]] tables (tests auto-discovered: 66 files under crates/*/tests/). cdylib: mantis-bridge `[lib] name=mantis_bridge`. Features: mantis-bridge `extension-module`; mantis-core `test-fixtures`; mantis-graph `default=["native"]`; mantis-selfplay `default=["phase_t_postfix"]`.
- [[bench]] (harness=false, 6 targets): mantis-core {smoke_bench, board_bench}, mantis-graph {build_bench}, mantis-search {mcts_bench}, mantis-selfplay {graph_build_bench, queue_fuse_bench}. (Makefile/run_all comments say "eight"/"seven non-smoke" — only 6 exist.) Compiled only by gate 2b `--all-targets`; smoke_bench by `make bench`.
- tools/bench_floors.toml: 23 `[floor.*]` rows keyed by `<bench>_<group>_<fn>` (board_bench 19, build_bench 2, graph_build_bench 1, smoke_bench 1) — CLAUDE.md says 28; liveness/names checked by tests/tools/test_bench_floors.py, also tests/model/conformance/test_leaf_forward_throughput_harness.py. Renaming a criterion group/fn reds these.
- Rust tests/benches read tests/fixtures/* by path: board/board_replay_golden_v1.json, eval_selfplay_parity/target_parity_v1.json, graph_parity/inputs.bin, mctx_parity/mctx_parity_v1.json (generator tools/gen_mctx_parity_fixtures.py), replay/, search_golden/{completed_q,temperature}/, worker/.
Pinned-count / name files:
- tools/ci_gates/test_count_floor.txt = 4862 (gate 3c; ratchet-down via absent test_count_ratchet_down.txt).
- tools/ci_gates/tier_declaration.txt (99 rows `file<TAB>test_fn|<module><TAB>marker`) — every slow/skip/skipif/integration/importorskip must be declared; deleting a declared test leaves a STALE row -> red (tools/ci_gates/tier_census.py; currently "79 deselected, all declared").
- tools/ci_gates/comment_length_floor.txt: comment_excess 3385, banner 0, docstring_excess 13079, private_docstring_excess 1485, rust_doc_excess 2676 (ratchet: may only fall).
- src/mantis/config/armed_aborts.py rows {actor_lag, draw_rate_collapse, policy_loss_trough, ply_cap_attractor, grad_norm_hard_abort, fused_graph_caps_calibrated, allocator_posture_minted}; `source_pin=(path, code-substring)` tamper-scanned by gate 12 over src/mantis/run.py (5), config/resolve/{allocator_posture,fused_graph_caps}.py, train/coordinator/step.py — editing the pinned line text reds gate 12.
- src/mantis/monitor/producer_manifest.yaml (sec 7): `event_literal` rows require the literal string to stay a CODE constant in the named module.
Path-string consumers:
- Gate 10 tools/ci_gates/check_tracked_refs.py: any `src|tests|tools|configs|crates|docs|vendor/...` token in Makefile, README.md, CLAUDE.md, docs/contracts/*.md, docs/governance/*.md (RULINGS.md exempt; docs/design exempt) must be tracked -> DELETING a file cited there reds gate 10 unless added to DISSOLVED_PATHS.
- Gate 13 contract_doc_gate: docs/contracts/run_config_schema.md `mantis.*` symbols + keys must resolve (reverse under "Deliberately absent").
- Gate 17 rule7_gate: content scan only (host strings), not a caller.
- tests/test_meta_ci.py (Makefile target set, ci.yml wiring), tests/tools/test_local_gate_runner.py (every ci.yml `gate N:` has a run_all.sh row).
- Source-text pins: ~90 test files open `src/mantis/...` as text (Path form `"src" / "mantis"`: tests/train 13, tests/config 11, tests/eval 11, tests/model 7, tests/selfplay 7 ...), and ~70 open `tools/...`; they assert tokens exist/absent — `git grep -n -F "<file basename>" tests` before moving/deleting a file or renaming a token.
- src -> tools path strings: bots/strix.py (`_DRIVER`, BUILD_SCRIPT tools/vendor_build_strix.sh), bots/sealbot.py (BUILD_SCRIPT tools/vendor_build_sealbot.sh), encoding/audit.py (tools/hardcode_scan.py), error-message strings naming tools/mint_config.py (config/resolve/{allocator_posture,composition,fused_graph_caps}.py, diagnostics/fusion_calibrate.py, monitor/supervise.py), diagnostics/worker_sweep.py (tools/worker_sweep_plan.toml).
- src/mantis/encoding/__init__.py::_REGISTRY_TOML_REL = "crates/mantis-encoding/src/registry.toml" (runtime path handshake, gate 8).
- Book data: src/mantis/arena/books/manifest.toml names book_v1_s20260625_p4.json / book_v2_pool_s20260915_p4.json (+ minter provenance strings tools/mint_opening_book.py, tools/select_balanced_book.py).
- vendor/pins.toml -> vendor/patches/sealbot.patch; strix checkpoint name/sha (consumed by src/mantis/bots/strix.py, tools/vendor_build_strix.sh).
- No pre-commit config; no other workflow than .github/workflows/ci.yml.
- `__all__` re-exports: 91 src files carry `__all__`; a symbol may be live only via a package re-export (grep the bare name, not the module path). Tests pinning export sets: tests/encoding/test_no_dead_resolver_export.py, tests/train/test_no_phantom_seam_member.py, tests/config/test_config_discovery_authority.py, tests/model/conformance/test_arch_reachability_and_graves.py (graves = names that must stay ABSENT).
Per-tool citation counts (file-level `git grep -l -F <basename>`; docs/tests/code): audit_bootstrap_corpus 2/2/0 · bench_server 10/1/0 · mirror_pull 4/2/0 · pytest_step_summary 0/0/ci.yml only · hardcode_scan 0/1/2(src encoding audit) · gen_mctx_parity_fixtures 0/1/2(crates) · select_balanced_book 0/1/2 · probe1.py 4/0(loaded as package)/2 · position_analyzer 2/0/Makefile · all ci_gates/*.py have a run_all.sh or lint_gate.sh/test_count_gate.sh caller except preflight_mint_parent.py (loaded by preflight_mint.py) and pytest_step_summary.py (ci.yml only).
