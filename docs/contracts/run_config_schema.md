# Contract: run config schema

- version: v8
- owner: mantis.config.schema
- status: LIVE since scaffold (WP0). Seven growth steps since (v1 -> v8). Each through v6 is
  recorded as a named amendment in docs/design/repo_design.md §4; v7 and v8 are NOT, and that
  is stated rather than implied — v7 landed without one and v8 (R242/ADJ-D12) inherits that
  gap rather than back-filling somebody else's amendment. v2 through v5, v7 and v8 are
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

## Shape

Ten top-level fields; **177 leaf key-paths** under the walker that descends nested blocks
(including optional ones) and counts a `list[SubModel]` field as ONE leaf.

| section | leaves | models |
|---|---|---|
| `schema_version` | 1 | int, pinned to `SCHEMA_VERSION` |
| `run_id` | 1 | str, `^[a-z0-9][a-z0-9_\-]*$` |
| `seed` | 1 | int |
| `eval_enabled` | 1 | bool |
| `identity` | 2 | `IdentityConfig` |
| `eval` | 30 | `EvalConfig`, `GateConfig`, `LadderConfig`, `LadderRung` |
| `train` | 53 | `TrainConfig`, `DrawRateAbortConfig`, `ReplayCapacityStage`, `MicrobatchCapsConfig` |
| `selfplay` | 44 | `SelfplayConfig`, `MctsConfig`, `PlayoutCapConfig` |
| `inference` | 8 | `InferenceConfig` |
| `monitor` | 37 | `MonitorSchemaConfig`, `DrainCapsConfig`, `DiskGuardConfig` |

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
| identity keys (encoding, representation) have no terminal defaults; representation is the closed Literal {grid, graph} | mantis.config.schema.core (`IdentityConfig`) |
| a declared representation that disagrees with the encoding registry is REJECTED at load — a mismatch would bypass the LAW-06 amp pin | mantis.config.schema.core (`IdentityConfig._representation_matches_registry`) |
| eval opponent sims are REQUIRED fields (no code default); the resolver reads the config value | mantis.config.schema.core (`EvalConfig`) + mantis.config.resolve.nsims |
| absent encoding + no stamp = error (no "v6" terminal default, LAW-11) | mantis.config.resolve.encoding (`AbsentEncodingError`) |
| graph autocast dtype pinned to bf16 in code (LAW-06) | mantis.config.resolve.amp (a string token; never a `torch.dtype`) |
| one resolver per regime knob — eval reads the same seam self-play does | mantis.config.resolve (`actor_sync`, `amp`, `bootstrap`, `composition`, `coordinator`, `drain`, `draw_rate`, `encoding`, `monitor`, `nsims`, `run_length`) |
| resolved-config event payload: 7 schema leaves (`source="file"`) + the derived `amp_dtype` = 8 knobs, no merge provenance | mantis.config.emit (`ResolvedConfig.to_event_payload`) |
| the audit root is enumerated name-agnostically, so a file the loader ACCEPTS is a file the audit SEES | mantis.config.loader (`discover_configs`; the shared-authority invariant) |
| configs are complete and minted, never hand-varied; delta stamped in header | tools/mint_config.py |
| two configs differ exactly where claimed | tools/config_diff.py `--expect` |
| a committed config's stamped header cannot lie about its delta | tools/config_diff.py `--from-header` |
| every committed config schema-validates (empty set = gate failure) | CI gate 7 (tools/ci_gates/validate_configs.py) |
| every schema leaf key has a live consumer (177-entry bijection, LAW-08), in two independently-maintained copies | tests/config/test_every_key_has_consumer.py + tests/config/test_every_key_has_consumer_p2.py |
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
