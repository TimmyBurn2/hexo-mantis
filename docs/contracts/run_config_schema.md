# Contract: run config schema

- version: v4
- owner: mantis.config.schema
- status: LIVE since scaffold (WP0). Four growth steps since (v1 -> v4), each recorded as a
  named amendment in docs/design/repo_design.md §4; the last three are *incompatible* — a
  config lacking any of the added keys fails to load. The config files' own `schema_version:`
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

## Shape

Nine top-level fields; **170 leaf key-paths** under the walker that descends nested blocks
(including optional ones) and counts a `list[SubModel]` field as ONE leaf.

| section | leaves | models |
|---|---|---|
| `schema_version` | 1 | int, pinned to `SCHEMA_VERSION` |
| `run_id` | 1 | str, `^[a-z0-9][a-z0-9_\-]*$` |
| `seed` | 1 | int |
| `identity` | 2 | `IdentityConfig` |
| `eval` | 30 | `EvalConfig`, `GateConfig`, `LadderConfig`, `LadderRung` |
| `train` | 50 | `TrainConfig`, `DrawRateAbortConfig`, `ReplayCapacityStage` |
| `selfplay` | 44 | `SelfplayConfig`, `MctsConfig`, `PlayoutCapConfig` |
| `inference` | 8 | `InferenceConfig` |
| `monitor` | 33 | `MonitorSchemaConfig`, `DrainCapsConfig` |

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
| every schema leaf key has a live consumer (170-entry bijection, LAW-08), in two independently-maintained copies | tests/config/test_every_key_has_consumer.py + tests/config/test_every_key_has_consumer_p2.py |
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
| `_draw_rate_evidence_bar_within_configured_capacity` | `RunConfig` | `train.draw_rate_abort.N_pool_min` <= `DRAW_RATE_WINDOW * selfplay.n_workers`, the measured ceiling of the pooled window sum; the bound that replaced the retired `min_samples: le=DRAW_RATE_WINDOW` pin. **A CAPACITY check, not a reachability one (R95/ADJ-22)** — whether the bar is actually met depends on how many workers report, which load time cannot witness; an unmet bar surfaces at runtime as an absence of observations (R92), never as a healthy `0.0` |
| `_one_drawn_game_cannot_fire_the_abort` | `DrawRateAbortConfig` | `1 / N_pool_min` < `threshold`: below that a SINGLE drawn game meets the bar and fires a hard abort |
| `_entropy_sign` | `TrainConfig` | `train.entropy_reg_weight` >= 0, raised as the NAMED sign-law error rather than a bare bound message |
| `_mixing_floor_is_below_its_start` | `TrainConfig` | `train.mixing_min_w` <= `train.mixing_initial_w`, else the floor wins at every step and two sibling keys decide nothing while still reading as the schedule's terms |
| `_stages_are_strictly_increasing` | `TrainConfig` | `train.replay_capacity_schedule` steps strictly increase; the consumer's cursor only moves forward, so an out-of-order stage is applied at the wrong boundary |
| `_validate_ladder` | `LadderConfig` | non-empty rungs, unique rung names, `0 < activation_wr_lower_ci <= graduation_wr_lower_ci < 1`, and two `>= 1` cadence floors |
| `_mutual_exclusion` | `PlayoutCapConfig` | `fast_prob` and `full_search_prob` are mutually exclusive; a configured quick/full pair must differ and its probability must sit in `(0, 1)` |

Three v4 keys are spelled differently in the schema and in the runtime object they reach, each
for a measured reason: `train.buffer_save_interval` -> `checkpoint_interval` (the coordinator's
is the REPLAY-BUFFER save cadence, while `train.checkpoint_interval` is the already-authored
TRAINER cadence, and two config keys with one spelling is the duplicated-authority class R1
kills), `train.replay_capacity` -> `capacity` and `train.replay_capacity_schedule` ->
`buffer_schedule` (a config key spelled only `capacity` names nothing on its own). The rename
happens at the schema and does not propagate into the runtime object.

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
| every-key-has-consumer bijection, the 170 count, the walker's descent into an OPTIONAL block, and a mutation self-test in both copies | tests/config/test_every_key_has_consumer.py, tests/config/test_every_key_has_consumer_p2.py |
| the radius field is removed everywhere: no schedule on the schema, no resolver module, no symbol in either `__all__` | tests/config/test_radius_removed.py |
| `train` section bounds and required-field census | tests/config/test_train_schema.py |
| `train.entropy_reg_weight` sign law; `policy_target`/`completed_q_values` cross-section consistency | tests/config/test_train_entropy.py, tests/config/test_train_policy_value_target_consistency.py |
| the nineteen coordinator knobs are read by ONE resolver and each moves the behaviour it names | tests/config/test_coordinator_knobs_wiring.py |
| the four `monitor.drain.*` keys each move the join bound the eval pipeline uses; the builder takes them as a required keyword-only parameter | tests/config/test_drain_caps_wiring.py |
| `train.draw_rate_abort` bounds, the evidence-bar CAPACITY rule, the one-drawn-game rule, and every config stating its posture explicitly | tests/config/test_drawrate_schema_range.py |
| the arming authority is the block and only the block — no second enable flag | tests/config/test_drawrate_arming_authority.py |
| `eval` numeric bounds (out-of-domain rejected, in-domain boundary accepted) including the `inf`/`nan` sweep | tests/config/test_eval_schema_bounds.py |
| `selfplay` / `mcts` / `playout_cap` bounds and the PCR mutual-exclusion rules | tests/config/test_selfplay_schema.py, tests/config/test_mcts_playout_cap_schema.py, tests/config/test_selfplay_playout_cap_mutual_exclusion.py |
| `monitor` section field-for-field parity with the runtime `MonitorConfig` | tests/config/test_monitor_schema.py |
| `train.actor_sync_cadence_steps` bounds and its reachability inside `train.max_train_steps` | tests/config/test_actor_sync_schema.py |
| the eval section survives a re-mint byte-identically | tests/config/test_eval_config_remint.py |

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
- `train.draw_rate_abort.consec` has an open upper half for the same reason — the only honest
  ceiling is the abort history's own depth, a runtime constant in a module `mantis.config` must
  not import. Above that depth the abort is unfireable while it still audits ARMED.
- `train.draw_rate_abort.threshold` still admits a hair-trigger below `1 / N_pool_min`;
  `_one_drawn_game_cannot_fire_the_abort` closes the part of it that matters.
- `train.bot_batch_share` is half-wired by design: its sibling `bot_corpus_path` was dead and
  is deleted, so only an injecting caller can supply a bot buffer, and `0.0` is the only value
  the composition root can honour.
- `N_pool_min`'s minted value rests on a continuity argument, not on a measured early-run draw
  distribution; it is revisable at mint prereg.
