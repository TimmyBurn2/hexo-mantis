# Contract: run config schema

- version: v13
- owner: mantis.config.schema
- status: LIVE since scaffold (WP0). Twelve growth steps since (v1 -> v13). Each through v6 is
  recorded as a named amendment in docs/design/repo_design.md §4; v7, v8 and v9 are NOT, and
  that is stated rather than implied — v7 landed without one, v8 (R242/ADJ-D12) inherited that
  gap rather than back-filling somebody else's amendment, and v9 records the same gap for
  itself instead of quietly closing it. v2 through v5 and v7 through v9 are
  *incompatible* — a config lacking any of the added keys fails to load — and v6 is
  incompatible in the other
  direction: a config still CARRYING the deleted key fails `extra="forbid"`. The config files' own `schema_version:`
  key is a FILE-FORMAT pin and is unchanged at `1`; it is not this contract's version.

## Summary
pydantic models, `extra="forbid"`, `frozen=True`, `strict=True` (silent scalar coercions
rejected); `schema_version` pinned in every file; NO code-side default anywhere — a default
lives only in the schema field; one resolver per regime knob; the loader rejects duplicate
YAML keys and enumerates the audit root name-agnostically.

## Version history

| ver | what moved | landed by |
|---|---|---|
| v1 | founding shape; additive growth in WP8 (strict mode, the resolver family, the duplicate-key loader, the consumer bijection) — no key changed incompatibly | WP0 scaffold, WP8 |
| v2 | `train.max_train_steps` becomes a required leaf: the RUN-LENGTH authority, distinct from `train.total_steps`, which is only the LR-scheduler horizon | WPAX Phase S (ADJ-09 Option B) |
| v3 | `train.draw_rate_abort` becomes a required leaf — a block or `null`, where `null` is the EXPLICIT disarmed posture. Its third key was `min_samples` (`ge=1, le=DRAW_RATE_WINDOW`) and is now `N_pool_min` (`ge=1`), because the gated statistic changed to the pooled count-weighted rate | WPAX Phase D; third-key swap WPMINT Phase DS (R92) |
| v4 | twenty new required leaves: nineteen flat `train.*` step-coordinator knobs plus `train.draw_rate_abort.consec`. Six sibling coordinator fields were DELETED rather than authored (no reader in `src/`), and `train.batch_size` is minted at the value the code actually used | WPMINT Phase K-B |
| v6 | one required leaf REMOVED — the `train` section's `buffer_save_interval`, the first leaf this contract has ever dropped. Its only consumer chain ended in `coordinator/step.py`'s D4 `_try_save_buffer` arm, which WP12-R Phase CS MEASURED (F-CS-2) production-dead on every leg, so a key minted into `run5.yaml` had zero reachable effect (R116/LAW-08/R1). The `buffer_save_interval` -> `checkpoint_interval` rename seam and both no-op `_try_save_buffer` arms go with it. `extra="forbid"` makes this incompatible in reverse: a config still carrying the key fails to load | WP12-R (R178(a), assigned by R183(a)) |
| v5 | five new required leaves, all promotions of authority OUT of code: `eval_enabled` (top-level bool — was a `compose_run` parameter with a code-side default `True`, R120), the `monitor.disk_guard` family `{interval_sec, warn_gb, fail_gb}` (was four dead `dict.get` literals in a function with zero callers, R122) and `train.device` (`Literal["cpu","cuda"]` — was a `--device` CLI flag on BOTH callers, which let a cpu-flagged preflight false-clear a cuda-minted run, R126) | WPMAIN (CARD-RUN-MAIN) |
| v8 | one new required leaf: `monitor.gate_interval` — the ARMING cadence, split off the NARRATION cadence `train.log_interval`. Until this version the live hard-abort gates and the LAW-18 `monitor_gates` summary both ran inside `_run_log_interval`, so at run5's minted `log_interval: 1000` the draw-rate abort could take no observation and no `monitor_gates` event could exist before training step 1000 — armed machinery with a blind first kilometre, and the instrument that would have shown it switched off by the same knob. `ge=1` and NO off value, for the reason `log_interval` carries the same bound (WPMINT DR-7): a non-positive stride kills the whole hard-abort family AND its visibility together. Every committed config mints it EQUAL to its own `train.log_interval`, so NO armed value's meaning moves as this lands; the re-scaled stride and the `consec` re-derived in gate-interval units are mint-prereg rows. Consumer: `mantis.run.compose_run` names it directly and threads it into `StepCoordinatorConfig.gate_interval`, read by `_run_gate_interval` — a scalar with one leaf and no shape needs no `resolve_*` module, but it carries no default and no fallback to `log_interval` anywhere | remediation bundle (R242, ADJ-D12) |
| v7 | two new required leaves, ONE block: `train.microbatch_caps.{max_edges, max_nodes}` — the GRAPH training step's memory bound. `train.batch_size` bounds the number of GRAPHS and bounds neither quantity that drives memory (E and N are SUMS over the sampled graphs; CARD-RUN5-GPU-OOM was one unbounded `[E, hidden]` allocation, measured at `E = 18 735 930`). A nested block and not two flat keys because the members are sized TOGETHER from ONE measured cost model against ONE budget — `DrawRateAbortConfig`'s grounds applied to a different fact. `ge=1` on both and NO off value: the schema cannot express "uncapped", because an uncapped graph step is the defect the block exists to make unconstructible (R79). GRAPH-ROUTE-SCOPED consumer: the resolver's PROVIDER is threaded to `dispatch.py::_graph_step` alone and `_grid_step` is not given it, so a grid run structurally cannot read the block | WP12-R dispatch 6 phase F2 (R179) |
| v10 | two new required leaves, ONE block: `inference.fused_graph_caps.{max_fused_edges, max_fused_nodes}` — the GRAPH INFERENCE forward's memory bound, the training cap's partner over the same card. `inference_batch_size` bounds the number of GRAPHS in a fused pop and bounds neither quantity that drives memory: E and N are SUMS over the fused graphs, and the fuse had NO bound at all, so the training cap's budget — which subtracted a self-play term measured when the inference forward carried ONE graph — was fitted against a partner that then moved by a large factor. A nested block and not two flat keys for `MicrobatchCapsConfig`'s recorded grounds: the members are sized TOGETHER from ONE measured fit against ONE budget, and two keys would be two authorities over one byte budget. The members are deliberately NOT spelled `max_edges`/`max_nodes` — the train-side OF2-9 census freezes those names, and distinct names keep the two budgets unconfusable in a grep. Shape is `int \| None` with `ge=1` and NO "uncapped" sentinel (R79): the off state is unrepresentable, because a disable sentinel is a switch for turning the fix off. `null` IS NOT AN OFF STATE — it is R119's placeholder, schema-VALID so gate 7 stays green and the repo ships a complete config, and runtime-REFUSED (`UncalibratedFusedGraphCapsError`) so a graph run on an uncalibrated production config CANNOT CONSTRUCT ITS INFERENCE SERVER; the two production configs mint it, the five others mint values that are non-binding by construction and derived from each template's own geometry. GRAPH-ROUTE-SCOPED consumer: `mantis.config.resolve.fused_graph_caps.resolve_fused_graph_caps` -> the GRAPH branch of `InferenceServer.__init__` (eagerly, at construction) -> `_run_graph_loop`'s `plan_fused_forwards` partition, and -> `RoundSpec.fused_graph_caps` -> `LocalInferenceEngine` in the eval child, which has its own CUDA context and which no in-process bound can see | F-816-10 (R276(f)) |
| v9 | five new required leaves, TWO blocks, both minted `null` in every committed config AT THE TIME v9 LANDED — **no longer true at HEAD**, and the row is corrected in place under R311(c) rather than left standing: `configs/run5.yaml` and `configs/shakedown_20260807.yaml` now ARM `eval.strength_floor` (`probe_games: 4, min_decisive_rate: 0.25, min_winrate: 0.0`) as a mint act, so the disarmed-everywhere claim below describes v9's landing state and not the shipped tree (AUDIT-1 F-05; `mantis.eval.floor_gate` carried the same stale sentence and is repaired with it). `eval.ply_cap_adjudication` IS still `null` in all seven. The leaves: `eval.ply_cap_adjudication.{criterion, min_margin}` and `eval.strength_floor.{probe_games, min_decisive_rate, min_winrate}`. Grounds are measured, not hypothetical (F-R-P2B-5): a terminal eval round at training step 33 completed ZERO of its spec'd games inside its full 4 h hard cap with the worker healthy throughout, and the same burn measured `draw_rate` 1.0 at the arena's 128-move ply cap — so the eval instrument was simultaneously unable to finish a round and, on the games it did finish, reporting one constant. The first block makes a ply-capped game resolvable by a declared criterion instead of collapsing to `"draw"`; the second gates the expensive gate-block/ladder behind a cheap probe against the round's cheapest opponent. Both are `Block | None` on the R79 shape — `null` is the EXPLICIT disarmed posture and is the IDENTITY value: with both `null` the arena's capped-game label, the round's phase order, the sidecar result JSON's key set and the event stream are byte-identical to v8. The five VALUES are mint-prereg rows the operator owns; no default exists anywhere in code and none is proposed here. Consumers: `mantis.config.resolve.eval_posture.{resolve_ply_cap_adjudication, resolve_strength_floor}` -> `RoundSpec` -> `mantis.arena.match.play_paired_match` / `mantis.eval.floor_gate.evaluate_strength_floor` | eval-posture bundle (F-R-P2B-5) |
| v11 | one new required leaf: `allocator_posture` — a TOP-LEVEL `Literal["default","expandable_segments"] | None` naming the CUDA caching allocator's REGIME. Grounds are measured, not hypothetical: the 2026-08-22 re-calibration sitting measured **14.98 GiB of card high-water under the DEFAULT posture against 11.36 under `expandable_segments:True`**, same config, same host, same duration — and kept DEFAULT anyway, because a cap fitted under the better posture would have depended on an environment variable that no config minted, no gate checked and no `armed_aborts` row covered. That is a minted value with an unminted precondition (R1's silent-authority class), and this leaf is its removal. TOP-LEVEL for `eval_enabled`'s recorded grounds — a root-composition fact spanning more than one section's surface, with three consumers in two processes. A CLOSED TOKEN SET rather than the raw env string, because `expandable_segments:True,max_split_size_mb:128` is a regime nobody fitted and a free-text key would admit it while reading as minted. `null` IS NOT AN OFF STATE — it is R119's placeholder, schema-VALID so gate 7 stays green and the repo ships complete configs, and runtime-REFUSED (`UncalibratedAllocatorPostureError`) so a CUDA process on an unminted regime cannot boot. EVERY committed config mints `null`: R308(g)(i) reserves the VALUE for the re-calibration sitting under R282(b), so no value is proposed here and none exists in code. CUDA-ROUTE-SCOPED consumer: `mantis.config.resolve.allocator_posture` -> the boot assertion in `mantis.run.build_run_collaborators` (before the first CUDA allocation), and -> `RoundSpec.allocator_posture` -> `mantis.eval.worker.run_round`'s first statement, because the eval child is a second allocator in its own process whose ENVIRONMENT the parent's assertion says nothing about. A cpu-device process is not enforced and says so. Audited by the DEFERRED `allocator_posture_minted` armed-abort row | RECAL-PREP (R308(g)(i)) |
| v12 | NO leaf added or removed — the SHAPE of two existing blocks changes, and it is the first time this contract has made a key's REQUIREDNESS depend on another key. `train.microbatch_caps` and `inference.fused_graph_caps` become ARCH-SCOPED: REQUIRED on `identity.representation == "graph"`, REFUSED on any other. Until v12 `RunConfig` was `extra="forbid"` with every key required, so a GRID config was not merely allowed to carry two graph-only cap blocks counted in EDGES and NODES — it was REQUIRED to, and `tools/mint_config.py` had to write a number for a quantity a grid run has none of. The call sites were already arch-gated by hand (`mantis.run` resolves the fused caps only on the graph branch; `train/coordinator/step.py` hands the microbatch caps to the graph arm as a lazy thunk) and everything BELOW them was not: the schema demanded the key and the resolver served it to a config of either arch. The scoping is DECLARED in one place, `mantis.config.schema.core.ARCH_SCOPED_KEYS`, and enforced in two: `RunConfig` refuses a config whose representation does not match, and each block's ONE resolver refuses a foreign config BY NAME before it looks for the block. PRESENCE IS READ OFF `model_fields_set`, not off the value, so an explicit `null` is CARRYING the key — which keeps this distinct from the R119 placeholder that lives on the two blocks' MEMBERS and means "minted but uncalibrated". `RunConfig` also gains a `model_serializer` that DROPS an absent arch-scoped block from `model_dump()`, because a dump has no `model_fields_set` and the round trip `RunConfig.model_validate(config.model_dump())` is load-bearing on every resolver call site. The two GRID configs lose both blocks; NO production config changes, and no minted value moves | SEAM-B2 Leg 1 (R322(d)) |
| v13 | one new OPTIONAL leaf, the first this contract has ever carried with a schema default: `identity.arch_kind` (`str \| None`, default `null`) — THE ARCH-SELECTOR ROW (R330(e), candidate D of R322(d)). R323(b) rules that the key enters production configs ONLY as a minted row at run6's mint, so the schema must accept its absence today and every committed config omits it; the two prior shapes (a REQUIRED key, which puts a row into the two production configs; no key, which makes the mint an engine change) are both excluded by that ruling. The `null` is not a fallback carrying a guess: `mantis.model.arch.arch_from_spec_and_config` hands a PRESENT value to `select_arch`, which refuses an unknown kind or one the representation does not admit BY NAME at the first net built, and resolves an ABSENT row to `INCUMBENT_ARCH_KIND[representation]` — what every config minted before the row existed has always built, pinned against the real minted files by `tests/model/conformance/test_arch_selector_makes_v2_selectable.py`. The schema cannot validate the value against the vocabulary itself: `mantis.model.amp` already imports `mantis.config`, so a `config -> model` import would be a gate-9 cycle. The exemption from the all-fields-required pin is enumerated by name in `tests/config/test_schema.py`, so a THIRD optional leaf is still a red. CONFIG-LESS call sites never read this row: a checkpoint's legacy read, `strip_and_restamp` and the pretrain validator resolve an artifact's arch from its STAMP through the one function `mantis.train.checkpoints.stamped_arch_kind` | FINISH-1 (R330(e)) |

## Shape

Eleven top-level fields; **186 leaf key-paths** under the walker that descends nested blocks
(including optional ones) and counts a `list[SubModel]` field as ONE leaf.

| section | leaves | models |
|---|---|---|
| `schema_version` | 1 | int, pinned to `SCHEMA_VERSION` |
| `run_id` | 1 | str, `^[a-z0-9][a-z0-9_\-]*$` |
| `seed` | 1 | int |
| `eval_enabled` | 1 | bool |
| `allocator_posture` | 1 | `Literal["default","expandable_segments"] \| None` |
| `identity` | 3 | `IdentityConfig` |
| `eval` | 35 | `EvalConfig`, `GateConfig`, `LadderConfig`, `LadderRung`, `PlyCapAdjudicationConfig`, `StrengthFloorConfig` |
| `train` | 52 | `TrainConfig`, `DrawRateAbortConfig`, `ReplayCapacityStage`, `MicrobatchCapsConfig` |
| `selfplay` | 44 | `SelfplayConfig`, `MctsConfig`, `PlayoutCapConfig` |
| `inference` | 10 | `InferenceConfig`, `FusedGraphCapsConfig` |
| `monitor` | 37 | `MonitorSchemaConfig`, `DrainCapsConfig`, `DiskGuardConfig` |

**The per-section rows above are HAND-VERIFIED, and gate 13 does not check them.** The gate
derives the document's TOTAL from `RunConfig` and compares it; no code path parses this table,
so a section row can drift undetected while the total stays green. F-816-10 re-derived all ten
rows against the gate's own `_leaf_paths` walker and found `train` stating **53** against a
live **52** — stale since v6 removed the `train` section's dead replay-buffer-save leaf and
decremented the total but not the row. Corrected here. Whoever edits a section row next re-derives all ten the same way,
or writes the check (ORACLE-WRITE noted it is oracle-able: derive per-section counts from
`RunConfig`, compare to this table) and retires the gap.

`mantis.config.schema` is a package, not a module: `_base` carries the ONE `StrictModel` every
section subclasses, and `core`/`train`/`selfplay`/`monitor` carry the sections. The split is
what keeps the package's internal import graph a DAG (CI gate 9) — `core` imports `train`, so
`train` cannot import `StrictModel` from `core`.

## Who asserts what where

| assertion | where |
|---|---|
| missing key = hard error; unknown key = hard error (top-level AND nested); values immutable | mantis.config.schema._base (`StrictModel`: `extra="forbid"`, `frozen=True`) |
| silent scalar coercion rejected (str->int, float->int, bool->int) | mantis.config.schema._base (`StrictModel`: `strict=True`) |
| duplicate YAML key = hard error (the frozen loader silently last-won) | mantis.config.loader (`_UniqueKeyLoader` -> `DuplicateKeyError`) |
| `schema_version` pinned to the exact current version (`SCHEMA_VERSION = 1`) | mantis.config.schema.core (`RunConfig._pin_schema_version`) |
| identity keys (encoding, representation) have no terminal defaults; representation is the closed Literal {grid, graph}; `identity.arch_kind` is the ONE optional identity leaf — the arch-selector row, absent in every committed config until run6's mint writes it (R323(b)), read only by `mantis.model.arch.arch_from_spec_and_config`, which hands a present value to `select_arch` (unknown/non-admitted kinds refused by name at construction) and resolves absence to the representation's pinned incumbent; artifacts never read it | mantis.config.schema.core (`IdentityConfig`), mantis.model.arch |
| a declared representation that disagrees with the encoding registry is REJECTED at load — a mismatch would bypass the LAW-06 amp pin | mantis.config.schema.core (`IdentityConfig._representation_matches_registry`) |
| eval opponent sims are REQUIRED fields (no code default); the resolver reads the config value | mantis.config.schema.core (`EvalConfig`) + mantis.config.resolve.nsims |
| the two early-strength eval postures are REQUIRED keys whose DISARMED value is the explicit `null`, not an absent key — arming is a property of the value (R79), and both ship `null` so the mechanism is inert until mint | mantis.config.schema.core (`PlyCapAdjudicationConfig`, `StrengthFloorConfig`) + mantis.config.resolve.eval_posture |
| absent encoding + no stamp = error (no "v6" terminal default, LAW-11) | mantis.config.resolve.encoding (`AbsentEncodingError`) |
| graph autocast dtype pinned to bf16 in code (LAW-06) | mantis.config.resolve.amp (a string token; never a `torch.dtype`) |
| one resolver per regime knob — eval reads the same seam self-play does | mantis.config.resolve (`actor_sync`, `amp`, `bootstrap`, `composition`, `coordinator`, `drain`, `draw_rate`, `encoding`, `eval_posture`, `monitor`, `nsims`, `run_length`) |
| resolved-config event payload: 7 schema leaves (`source="file"`) + the derived `amp_dtype` = 8 knobs, no merge provenance | mantis.config.emit (`ResolvedConfig.to_event_payload`) |
| the audit root is enumerated name-agnostically, so a file the loader ACCEPTS is a file the audit SEES | mantis.config.loader (`discover_configs`; the shared-authority invariant) |
| configs are complete and minted, never hand-varied; delta stamped in header | tools/mint_config.py |
| two configs differ exactly where claimed | tools/config_diff.py `--expect` |
| a committed config's stamped header cannot lie about its delta | tools/config_diff.py `--from-header` |
| every committed config schema-validates (empty set = gate failure) | CI gate 7 (tools/ci_gates/validate_configs.py) |
| every schema leaf key has a live consumer (185-entry bijection, LAW-08), in two independently-maintained copies | tests/config/test_every_key_has_consumer.py + tests/config/test_every_key_has_consumer_p2.py |
| every `required` armed-abort row is armed in every production config | CI gate 12 (tools/ci_gates/preflight_mint.py `--audit-only`) |

## Cross-field rules (the invariants no single field can carry)

A bound that spans two SECTIONS lives on `RunConfig`, the one model that sees both. Each rule
carries its own name: a rule hidden inside a validator named for a different axis is a false
name at the moment it fires.

| validator | model | rule |
|---|---|---|
| `_policy_target_completed_q_consistency` | `RunConfig` | `train.policy_target`, `train.completed_q_values` and `selfplay.completed_q_values` are ONE decision with two consumers — all three must agree |
| `_actor_lag_threshold_exceeds_sync_cadence` | `RunConfig` | `monitor.actor_lag_threshold_steps` > `train.actor_sync_cadence_steps`: a threshold at or below the cadence fires under healthy operation |
| `_actor_sync_knobs_fit_inside_the_run` | `RunConfig` | the sync cadence, the lag threshold and `train.draw_rate_abort.min_step` must each be < `train.max_train_steps` — a guard the run never reaches is "armed in the config, absent in effect" |
| `_graph_sims_regime_fits_the_hexg_record_format` | `RunConfig` | graph runs only (R255/ADJ-D34): the HEXG visit-slot capacity is DERIVED from `selfplay.mcts.n_simulations` + `selfplay.playout_cap.*` + `selfplay.leaf_batch_size` (+ `selfplay.completed_q_values`) by the ONE engine authority (`mantis._engine.derived_hexg_visit_capacity`, the same function the boot guard calls) — a regime the record format cannot honor REDs here at mint, never as a boot surprise |
| `_draw_rate_evidence_bar_within_configured_capacity` | `RunConfig` | `train.draw_rate_abort.N_pool_min` <= `DRAW_RATE_WINDOW * selfplay.n_workers`, the measured ceiling of the pooled window sum; the bound that replaced the retired `min_samples: le=DRAW_RATE_WINDOW` pin. **A CAPACITY check, not a reachability one (R95/ADJ-22)** — whether the bar is actually met depends on how many workers report, which load time cannot witness; an unmet bar surfaces at runtime as an absence of observations (R92), never as a healthy `0.0` |
| `_one_drawn_game_cannot_fire_the_abort` | `DrawRateAbortConfig` | `1 / N_pool_min` < `threshold`: below that a SINGLE drawn game meets the bar and fires a hard abort |
| `_entropy_sign` | `TrainConfig` | `train.entropy_reg_weight` >= 0, raised as the NAMED sign-law error rather than a bare bound message |
| `_mixing_floor_is_below_its_start` | `TrainConfig` | `train.mixing_min_w` <= `train.mixing_initial_w`, else the floor wins at every step and two sibling keys decide nothing while still reading as the schedule's terms |
| `_stages_are_strictly_increasing` | `TrainConfig` | `train.replay_capacity_schedule` steps strictly increase; the consumer's cursor only moves forward, so an out-of-order stage is applied at the wrong boundary |
| `_validate_ladder` | `LadderConfig` | non-empty rungs, unique rung names, `0 < activation_wr_lower_ci <= graduation_wr_lower_ci < 1`, and two `>= 1` cadence floors |
| `_mutual_exclusion` | `PlayoutCapConfig` | `fast_prob` and `full_search_prob` are mutually exclusive; a configured quick/full pair must differ and its probability must sit in `(0, 1)` |

TWO v4 keys are spelled differently in the schema and in the runtime object they reach, each
for a measured reason: `train.replay_capacity` -> `capacity` and
`train.replay_capacity_schedule` -> `buffer_schedule` (a config key spelled only `capacity`
names nothing on its own). The rename happens at the schema and does not propagate into the
runtime object. A THIRD rename stood here until v6 and is listed under *Deliberately absent*
below: it existed only because `train.checkpoint_interval` (the TRAINER's periodic save, which
is untouched and still live) would otherwise have collided with a same-named coordinator key,
and with the coordinator field deleted there is no collision left to disambiguate.

## Pinning tests

| test | file |
|---|---|
| example config validates; every committed config validates; schema round-trip; all fields required (no code-side defaults, `model_fields` introspection) | tests/config/test_schema.py |
| top-level / nested unknown key rejected; missing top-level / identity / eval / selfplay key rejected; wrong `schema_version` rejected | tests/config/test_schema.py |
| representation `"dense"` (outside the closed set) rejected, `"grid"` accepted; a graph encoding declared `grid` (and the converse) rejected at validate | tests/config/test_schema.py |
| strict coercion rejection: str->int, float->int, bool->int | tests/config/test_schema_strict.py |
| duplicate-key loader rejection + mutation self-test | tests/config/test_loader_duplicate_key.py |
| discovery/loader shared authority: a file discovery skips is a file the loader refuses | tests/config/test_config_discovery_authority.py |
| encoding reconcile decision-equivalence + absent -> raise | tests/config/test_resolve_encoding.py |
| eval `model_sims` resolver + unknown-opponent / `None` raise | tests/config/test_resolve_nsims.py |
| amp dtype token graph->bf16 / grid->fp16 + the DAG no-torch guard | tests/config/test_resolve_amp.py, tests/config/test_resolve_amp_dtype.py |
| bootstrap path resolver | tests/config/test_resolve_bootstrap.py |
| resolved-config emit: 8-knob payload, death-of-merge census, and no `train`/`selfplay`/`monitor` leaf threaded into it | tests/config/test_resolved_config_emit.py, tests/config/test_resolved_config_emit_p2.py |
| one-key diff; mint output validates; header stamped; unknown delta key exits 2; diff exit 0 on an exactly-claimed diff, exit 1 otherwise | tests/config/test_mint_and_diff.py |
| lying-header `--from-header` self-check + mutation self-test | tests/config/test_config_diff_from_header.py |
| regime parity per LAW knob (sims, amp, encoding) and the radius knob's ABSENCE from every production config | tests/config/test_regime_parity.py, tests/config/test_regime_parity_p2.py |
| every-key-has-consumer bijection, the 177 count, the walker's descent into an OPTIONAL block, and a mutation self-test in both copies | tests/config/test_every_key_has_consumer.py, tests/config/test_every_key_has_consumer_p2.py |
| the radius field is removed everywhere: no schedule on the schema, no resolver module, no symbol in either `__all__` | tests/config/test_radius_removed.py |
| `train` section bounds and required-field census | tests/config/test_train_schema.py |
| `train.entropy_reg_weight` sign law; `policy_target`/`completed_q_values` cross-section consistency | tests/config/test_train_entropy.py, tests/config/test_train_policy_value_target_consistency.py |
| the eighteen coordinator knobs are read by ONE resolver and each moves the behaviour it names | tests/config/test_coordinator_knobs_wiring.py |
| the four `monitor.drain.*` keys each move the join bound the eval pipeline uses; the builder takes them as a required keyword-only parameter | tests/config/test_drain_caps_wiring.py |
| `train.draw_rate_abort` bounds, the evidence-bar CAPACITY rule, the one-drawn-game rule, and every config stating its posture explicitly | tests/config/test_drawrate_schema_range.py |
| the arming authority is the block and only the block — no second enable flag | tests/config/test_drawrate_arming_authority.py |
| `eval` numeric bounds (out-of-domain rejected, in-domain boundary accepted) including the `inf`/`nan` sweep | tests/config/test_eval_schema_bounds.py |
| `selfplay` / `mcts` / `playout_cap` bounds and the PCR mutual-exclusion rules | tests/config/test_selfplay_schema.py, tests/config/test_mcts_playout_cap_schema.py, tests/config/test_selfplay_playout_cap_mutual_exclusion.py |
| `monitor` section field-for-field parity with the runtime `MonitorConfig` | tests/config/test_monitor_schema.py |
| `train.actor_sync_cadence_steps` bounds and its reachability inside `train.max_train_steps` | tests/config/test_actor_sync_schema.py |
| the eval section survives a re-mint byte-identically | tests/config/test_eval_config_remint.py |
| `train.microbatch_caps` bounds (`ge=1` on both members, both required, no third member), the ONE-authority reader census, the graph-route-only consumer on run5's own config, and the five smoke configs' caps proven NON-BINDING on their own batches | tests/config/test_train_schema.py, tests/train/test_graph_microbatch_authority.py |
| the micro-batch split is the un-split step: partition properties, slice fidelity, the exact normalisation identity, one optimizer step per training step, the LAW-18 fire-rate counter, and the named out-of-domain raise | tests/train/test_graph_microbatch.py, tests/train/test_graph_microbatch_bound.py |

## Deliberately absent

- **`selfplay.legal_move_radius` / `selfplay.legal_move_radius_schedule`.** The encoding
  registry is the SOLE radius authority; nothing on the build path reads a config-level
  override, so the field would be a consumer-less knob (R1/LAW-08). `RadiusStage` and the
  `mantis.config.resolve.radius` module are RETIRED, not merely unused, and the emit payload
  gained no replacement leaf. Pinned by tests/config/test_radius_removed.py.
- **`eval.gate.screen_confirm_hi`.** Stored-but-never-read in run3; a dead schema key would
  violate LAW-08/R1, and `extra="forbid"` rejects a minted one.
- **Six step-coordinator fields** — `composition_interval`, `value_probe_interval`,
  `soft_ew_threshold`, `soft_ew_min_pts`, `instrumentation_enabled`, `bot_corpus_path`. No
  reader anywhere in `src/`, so v4 deleted them rather than authoring them: typing a dead knob
  in would have CREATED the R1/LAW-08 violation the section exists to prevent.
- **A boolean enable beside the draw-rate abort block.** The block's value already gates its
  own check; a flag would be a second authority over one fact.
- **The `train` section's `buffer_save_interval`** and, with it,
  `StepCoordinatorConfig.checkpoint_interval` and both no-op `_try_save_buffer` arms. v6 DELETED the key under R178(a): the replay-buffer
  save it paced is production-dead on every leg (WP12-R Phase CS, F-CS-2 — the helper returns
  unless `mixing_cfg["buffer_persist"]` is truthy and the composition root passes
  `mixing_cfg={}`), so the key had zero reachable effect while shipping into the run5 mint
  record. The helper module `train/buffer_persist.py` itself SURVIVES: persistence returns,
  if at all, as ONE design under CARD-RESUME (R178(c), post-mint) covering weights,
  optimizer/scheduler, buffer and launcher together. Do not read this row as "buffer
  persistence is banned" — it says the dead CONFIG KEY is, and that no piece of resume is
  built separately. The TRAINER's own periodic save, spelled `checkpoint_interval` in the same
  `train` section, is a DIFFERENT key and is LIVE — it is deliberately not named in dotted
  form here, because everything under this heading is checked in reverse and that key must
  resolve.

Everything named under this heading is checked in REVERSE by CI gate 13: a key or module listed
here that comes BACK reds the gate, which is the direction that matters.

## Disclosed residuals

Recorded here because a contract that lists only its closed edges misleads at mint time. Each
is also written at the field it belongs to.

- `train.hard_gn_threshold` and `train.hard_gn_min_steps` have OPEN upper halves: the shipped
  `1e9` threshold is finite, positive and unreachable by any real gradient norm, and a very
  large `min_steps` disarms the gate without touching the threshold. No honest ceiling is
  derivable from either field alone, so none is invented; the gate joins the armed-abort
  manifest as a DEFERRED row instead, whose ceiling is read off `monitor.alert_grad_norm_max`.
- `train.draw_rate_abort.consec`'s upper half is CLOSED (ADJ-D36), and closed by derivation
  rather than by a ceiling: the gate's history ring is sized BY the minted `consec` at the
  point of use (`_run_hard_abort_gates`'s draw-rate arm trims to `spec.consec`; the literal
  depth constant is deleted), so every schema-legal value is fireable and "unfireable while
  it audits ARMED" is no longer a reachable state on this key. Kept in this list as a
  closure record: this bullet used to call its upper half open "for the same reason" as the
  grad-norm pair above — that reason still stands for them and no longer stands here. Driven
  by tests/train/test_drawrate_gate_capacity.py.
- `monitor.wr_collapse_consecutive_evals` and `monitor.wr_rolling_consecutive_evals` have
  their upper halves CLOSED by the same derivation, one gate over (R265 / ADJ-D38): the
  sealbot-WR ring's capacity is now the max of the two minted consec keys and rule B's own
  peak window (`mantis.monitor.rules.WR_PEAK_WINDOW_EVALS`), derived at the point of use, so
  no schema-legal value is armed-in-the-config and unfireable-in-effect. The peak window is
  NAMED rather than derived away because the deleted ring depth was carrying two jobs at
  once, and widening rule B's peak with the ring would have moved an armed rule's decision.
  Driven by tests/train/test_wr_gate_capacity.py, bit-identical below the old depth.
- Their LOWER halves stay OPEN and are an operator question, not a closed edge: `ge=0` admits
  `0`, which does not disable a trigger — `history[-0:]` is the whole ring in Python — so `0`
  arms a weaker-evidence variant that fires on however many evals the ring holds. ADJ-D38
  raises "a rule that needs zero observations is not a rule" (the posture
  `train.draw_rate_abort.consec` already takes with `ge=1`) and R265 does not rule it, so no
  bound moved. Every committed config mints 2 and 3.
- `train.draw_rate_abort.threshold` still admits a hair-trigger below `1 / N_pool_min`;
  `_one_drawn_game_cannot_fire_the_abort` closes the part of it that matters.
- `train.bot_batch_share` is half-wired by design: its sibling `bot_corpus_path` was dead and
  is deleted, so only an injecting caller can supply a bot buffer, and `0.0` is the only value
  the composition root can honour.
- `N_pool_min`'s minted value rests on a continuity argument, not on a measured early-run draw
  distribution; it is revisable at mint prereg.
