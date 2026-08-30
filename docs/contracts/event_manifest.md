# Contract: event manifest

- version: v1
- owner: `mantis.monitor` (`manifest.py` + `producer_manifest.yaml`)
- status: v1 — filled by the run-safety subsystem port (WP13-A); the eval-pipeline rows
  (`sealbot_wr_warn` producer, `eval_round` heartbeat, `eval_round_wall`/`eval_broken`/
  `eval_rung_skipped`) landed at WP11-A

## Summary

Every headless gate/monitor input cites a LIVE producer and a NAMED producer test. The
manifest (`src/mantis/monitor/producer_manifest.yaml`) is the data; the checker
(`mantis.monitor.manifest.verify_manifest`) resolves both halves of every row and raises
`ManifestError` naming the offending row; the checker itself carries LAW-07 mutation
self-tests proving it BITES. A gate input whose producer was renamed, deleted or never
ported is therefore un-shippable — which is exactly the F-10 failure (a silently-unported
feature left a gate fed by nothing for 16,880 steps) made structurally impossible.

Scope: **headless only**. Display panels do NOT appear here — the old
`monitoring/dashboard_manifest.yaml` panel half died with the dashboards; what survives is
its producer law. There is exactly ONE channel: `channel: jsonl_event_sink`, the
`JsonlEventSink` JSONL stream (the old dual `emit_event`/structlog split died with
structlog).

### Row shape

```yaml
version: 1
channel: jsonl_event_sink
gates:
  - id: <gate id>                     # unique; the ManifestError message names it
    input: <input name>               # optional prose label for the consumed signal
    producer:
      kind: symbol | event_literal | seam
      module: <importable module>
      symbol: <dotted attr>           # symbol / seam
      literal: <event name>           # event_literal
      feeds_from: <dotted attr>       # optional upstream producer, also resolved
    also: {kind: symbol, module: …, symbol: …}   # optional second producer, also resolved
    pending: <WP id>                  # REQUIRED on `kind: seam`
    producer_test: <path>::<test_fn>  # REQUIRED on every row
```

### Resolution rules (what "resolves" means)

| kind | resolves iff |
|---|---|
| `symbol` | `module` imports AND the dotted attribute resolves. An INSTANCE attribute assigned as `self.<attr>` in the owning class's source counts (an instance counter is a real producer). |
| `event_literal` | the **QUOTED** literal (regex `["']<literal>["']`) appears in the named module's SOURCE. Quoted-only on purpose: an identifier substring such as `self._train_step` can NEVER satisfy a `train_step` row. |
| `seam` | the named attr resolves AND the row carries `pending: <WP>` naming the WP that owes the concrete producer. A pending gate with no owner is the silently-dead-forever class. |
| every row | `producer_test` = `<path>::<test_fn>`; the file exists under the repo root and an ast walk finds that function. A stale node id is un-shippable. |

An EMPTY gate list is a FAILURE, never a vacuous pass (R4: a gate surface with zero
producers is a phantom-armed abort chain waiting to happen).

## Who asserts what where

| fact | asserted where | pinning test |
|---|---|---|
| the shipped manifest loads and EVERY row resolves (producer + producer_test) | `mantis/monitor/manifest.py::verify_manifest` over `producer_manifest.yaml` | `tests/monitor/test_manifest_contract.py::test_shipped_manifest_every_row_resolves` (O-01) |
| a dead producer symbol raises `ManifestError` naming the row | `manifest.py::_verify_symbol` / `_resolve_dotted` | `::test_dead_producer_symbol_bites` (O-02, LAW-07 arm 1) |
| a missing producer_test node raises `ManifestError` naming the row | `manifest.py::_verify_producer_test` (ast) | `::test_missing_producer_test_bites` (O-02, LAW-07 arm 2) |
| an event-literal row needs a QUOTED literal (no substring match) | `manifest.py::_verify_event_literal` | `::test_event_literal_substring_does_not_falsely_resolve` |
| a `pending:` seam row must name a WP | `manifest.py::_verify_seam` | `::test_pending_seam_row_requires_a_wp_name` |
| an empty manifest is a failure | `manifest.py::verify_manifest` | `::test_empty_manifest_is_a_failure` |
| gate fire/skip/warn counts are visible IN RUN (LAW-18), on the ARMING cadence `monitor.gate_interval` and NOT on the narration cadence `train.log_interval` (R242/ADJ-D12 — under the old coupling run5's minted `log_interval: 1000` meant no `monitor_gates` event existed before training step 1000) | `train/coordinator/step.py::_run_gate_interval` -> `_emit_monitor_gates` (`monitor_gates` event) | `tests/train/test_gate_interval_decoupling.py::test_p1_gates_are_visible_and_advancing_far_below_the_log_interval_boundary` |
| the booted process publishes its OWN config identity first (`run_boot_identity`, F-B1) — consumed by the mint preflight's parent, which verdicts match/mismatch/unwitnessed and reds rc 14 on mismatch | `mantis/run.py::compose_run` (`config_identity_sha256`, the ONE authority) | `tests/test_run_composition.py::test_compose_run_publishes_its_boot_identity_first_through_the_one_authority` + `tests/tools/test_preflight_config_identity.py` |
| sealbot-WR ships WARN-ONLY (operator G-3): a sustained collapse emits a visible `sealbot_wr_warn` and does NOT stop the run | `train/coordinator/step.py::on_eval_round_complete` (`sealbot_wr_warn`) | `tests/train/test_coordinator_gates.py::test_sealbot_default_is_warn_only_and_does_not_shut_down` |
| the sealbot hard-abort remains a one-field capability (`wr_hard_abort_enabled=True`) | `monitor/rules.py::check_sealbot_wr_hard_abort` | `tests/train/test_coordinator_gates.py::test_sealbot_hard_abort_capability_when_enabled` |
| a heartbeat source nothing wired is loud, never a stall abort | `train/lifecycle/heartbeat_watchdog.py` (`heartbeat_source_unwired`; `heartbeat_watchdog_armed.unwired_sources`) | `tests/train/test_heartbeat_watchdog.py::test_undeclared_never_beaten_source_warns_instead_of_firing` |
| a teardown wedge is bounded (close-out deadline) | `heartbeat_watchdog.py::_check_close_out_overrun` (`close_out_timeout`) | `tests/train/test_heartbeat_watchdog.py::test_close_out_overrun_fires_after_the_teardown_budget` |
| the fire path reaches `exit_fn` even if a BOUNDED effect (snapshot, sink close) hangs — the two fire emits are exception-safe but hang-unbounded, supervisor-covered (`_fire` docstring, N3) | `heartbeat_watchdog.py::_bounded` (`heartbeat_watchdog_fire_complete`) | `tests/train/test_heartbeat_watchdog.py::test_hung_snapshot_still_exits_within_a_bounded_time` |
| an eval-result shape the seam cannot consume is recorded | `train/coordinator/drain.py::_route_eval_result` (`eval_result_unroutable`) | `tests/train/test_run_safety_wiring.py::test_an_unroutable_eval_result_is_recorded_loudly` |
| an inert gate is loud, not silent (skip counter) | `step.py::on_eval_round_complete` (`sealbot_wr_gate_skipped`) | `tests/train/test_coordinator_gates.py::test_sealbot_absent_key_skips_and_counts` |
| a gate skipped because the round BROKE is distinguishable from one skipped because the round carried no number (F-RESIT-14) | `step.py::on_eval_round_complete` (`sealbot_wr_gate_skipped` with `reason: eval_round_broken` + the typed `eval_broken_reason`) | `tests/train/test_eval_broken_reaches_the_gate.py::test_a_broken_round_reaches_the_gate_and_is_NAMED_there` |
| a gate skipped because the strength floor REFUSED the round is distinguishable from both of the above (F-RESIT-14 in a third form, R324(d)) | `step.py::on_eval_round_complete` (`sealbot_wr_gate_skipped` with `reason: strength_floor_refused` + `strength_floor_failed_bars`) | `tests/train/test_strength_floor_refusal_reaches_the_gate.py::test_a_floor_refused_round_is_NAMED_at_the_gate` |

## Shipped rows (WP13-A)

| id | kind | producer | producer test |
|---|---|---|---|
| `draw_rate_collapse` | symbol | `train.coordinator.config.pooled_draw_rate` ← `selfplay.pool.WorkerPool.pooled_draw_counts` | `tests/train/test_coordinator_gates.py::test_draw_rate_gate_fires_on_live_producer` |
| `sealbot_wr_warn` | symbol (+`also`) | `eval.rounds.build_round_result` (field `wr_sealbot`) + `train.coordinator.step.StepCoordinator.on_eval_round_complete` | `tests/eval/test_wr_sealbot_handshake.py::test_round_result_always_carries_wr_sealbot` |
| `grad_norm_hard_abort` | symbol | `train.coordinator.step.StepCoordinator._run_training_step` | `tests/train/test_coordinator_gates.py::test_grad_norm_gate_fires_with_the_uniform_contract` |
| `heartbeat.train_step` | event_literal | `train.coordinator.step` / `train_step` | `tests/train/test_coordinator_gates.py::test_step_loop_beats` |
| `heartbeat.inference_dispatch` | event_literal | `selfplay.inference_server` / `inference_dispatch` | `tests/selfplay/test_inference_server.py::test_graph_loop_emits_one_heartbeat_per_batch` |
| `heartbeat.selfplay_drain` | event_literal | `selfplay.pool_drain` / `selfplay_drain` | `tests/selfplay/test_pool_drain_parity.py::test_heartbeat_emission_at_drain` |
| `heartbeat.eval_round` | event_literal | `eval.pipeline` / `eval_round` (the persistent poller thread; beats every tick, idle or active) | `tests/train/test_eval_heartbeat.py::test_poller_thread_beats_eval_round` |
| `persist_fatal` | symbol (+`also`) | `train.checkpoints.persist_errors_total` + `monitor.sink.JsonlEventSink.persist_errors_total` | `tests/monitor/test_persist_fatal.py::test_sink_failure_counts_and_aborts` |
| `selfplay_stall` | symbol | `selfplay.pool.WorkerPool.games_completed` | `tests/train/test_lifecycle_contract.py::test_watchdog_fires_after_timeout` |
| `disk_guard` | symbol | `train.lifecycle.disk_guard.DiskGuard.check_once` | `tests/train/test_lifecycle_contract.py::test_disk_guard_emits_free_event` |
| `warn.training_step_alerts` | event_literal | `train.events` / `training_step` (4 rules: `entropy_collapse`, `selfplay_entropy_collapse`, `grad_norm_spike`, `loss_increase_window`) | `tests/train/test_coordinator_gates.py::test_log_interval_emits_training_step` |
| `eval_round_wall` | event_literal (+`also`) | `eval.pipeline` / `eval_round_started` + `eval_round_complete` | `tests/eval/test_round_events.py::test_round_emits_start_and_complete_wall_events` |
| `eval_round_progress` | event_literal | `eval.pipeline` / `eval_round_complete` (field `progress`) + `eval.worker._RoundProgress` (the writer) — R319(e). **Two joined changes, and the contract needs both.** (1) `games_total` is `int | None`, and **`None` is the BROKEN-round value**: it was a hardcoded `0` on every broken path while the success path summed the real count, so the two were indistinguishable and a killed round read as a round that played nothing. RECAL-SITTING-3 published exactly that and retracted it — a default wearing a measurement's clothes. A consumer must treat `None` as NO MEASUREMENT, never as zero. (2) `progress` carries the child's LAST per-game row, `{game_index, phase, plies, t_wall, terminal, winner, candidate_color, margin}`, or `null` when the child wrote none — so a broken round reports HOW FAR it got instead of nothing. `game_index` runs monotonically ACROSS phases (`floor_probe`/`gate_screen`/`gate_confirm`/`rung`/`random_floor`). Counters, LABELS and a timestamp ONLY: no moves, no positions, no trajectory hash, so the redaction discipline holds BY CONSTRUCTION rather than by filtering. **The four outcome fields are R320(c)** — `terminal` is `mantis.arena.adjudicate.TERMINAL_REASONS`, `winner` is `candidate`/`opponent`/`draw`, `candidate_color` is the SEAT (`1`/`-1`), and `margin` is the adjudicator's SIGNED candidate-minus-opponent measurement. They exist because the adjudicator's own tally is four counters, which reads the decisive RATE and says nothing about its SHAPE, and R320(c)'s measurement round is asked for the margin DISTRIBUTION and the seat balance. **`margin: null` is NO MEASUREMENT** (no adjudicator armed, or the game never reached the cap) and **`margin: 0` is a MEASURED exact tie** — the residual-draw bin at `min_margin: 1`; a consumer that reads `null` as `0` inflates that bin, which is (1)'s defect one field over. A record shape the writer does not recognise writes nulls rather than raising, on the same grounds as the `OSError` arm. **REPORTING ONLY this sitting** — escalation semantics are explicitly unchanged (R319(e)(ii)) and a structural test refuses any branch on `read_progress`. The writer disables itself after one loud stderr line on `OSError` and never raises: deliberately NOT LAW-14's persistence-is-fatal posture, because this file is diagnostic and raising would let a progress line kill a healthy round | `tests/eval/test_eval_round_observability.py::test_a_broken_round_now_CARRIES_how_far_it_got` (+ the sentinel's own structural row, `::test_the_broken_call_site_passes_None_and_no_literal_zero`; the R320(c) outcome fields at `::test_each_row_carries_the_outcome_facts_the_margin_distribution_needs`, `::test_a_ZERO_margin_and_an_ABSENT_one_do_not_collide` and `::test_the_verdict_margin_the_writer_reads_is_the_ADJUDICATORS_OWN`) |
| `eval_broken` | event_literal | `eval.pipeline` / `eval_broken` — payload `reason` is a `mantis.eval.errors.EvalBrokenReason` member, the ONE authority (WP12-R Phase O / R152): `join_timeout`, `killed`, `exit_nonzero`, `result_missing`, `result_invalid`, `ladder_persist_failed`, `round_completion_error` — seven members, seven censused routes, wire spellings byte-identical to the bare literals they replace. `phase` stays on the payload as a FUNCTION of the reason (`drain` / `worker_exit` / `ladder_persist` / `round_completion`). The two exception-bearing routes carry a `detail`, and they carry DIFFERENT ones: `round_completion_error` carries `detail` = `repr(exc)` AND an `exception_class` key; `ladder_persist_failed` carries `detail` ONLY — a persistence message naming the ladder-state path, not `repr(exc)` — and NO `exception_class` key at all (the emitter adds each extra iff it was passed, `pipeline.py:499-501`). The other five routes carry neither. The routed round result carries the same value as `eval_broken_reason` (`None` IS the clean state — the `eval_broken` bool and the `error` key are DELETED, R79) plus `eval_broken_detail`, which is PROSE nobody branches on | `tests/eval/test_eval_broken.py::test_killed_worker_yields_eval_broken_and_clean_drain` (+ reason `round_completion_error` — RED-TEAM-FIX WP11-A F1 layer 2, ANY uncaught exception in round completion/scheduling converts to a delivered `eval_broken` rather than killing the poller thread silently — `tests/eval/test_round_completion_error.py::test_poller_thread_survives_an_uncaught_exception_in_round_completion`) |
| `eval_rung_skipped` | event_literal | `eval.pipeline` / `eval_rung_skipped` | `tests/eval/test_rung_loud_skip.py::test_unresolvable_rung_emits_skip_event_and_log` |
| `eval_strength_floor` | event_literal | `eval.pipeline` / `eval_strength_floor` — the ladder strength floor's ONE channel (F-R-P2B-5). ARMED-ONLY: it is emitted iff the worker's result payload carries a `strength_floor` key, which it does iff `eval.strength_floor` is a block. Every committed config mints `null`, so on the tree as shipped this literal never reaches the stream and its two counters never move. Payload = `{event, round_id, step}` + the verdict (`passed`, `games`, `decisive_games`, `decisive_rate`, `wins`, `draws`, `winrate`, `min_decisive_rate`, `min_winrate`, `failed_bars`) + `checked_total` / `skipped_total`. ONE event and not a pass channel plus a skip channel, because they are one fact; BOTH totals ride it because LAW-18 wants a fire RATE — a skip count with no denominator cannot separate "the floor never fires" from "the floor never ran". **Not a `producer_manifest.yaml` monitor input**: no rule consumes it, and registering it would create a producer-without-consumer (the exclusion `eval_rung_skip_class` already carries) | `tests/eval/test_eval_posture_inert.py::test_an_armed_failing_round_emits_the_floor_event_with_BOTH_totals` (+ the inert arm, `::test_a_disarmed_round_emits_no_posture_event_and_moves_no_counter`) |
| `eval_ply_cap_adjudication` | event_literal | `eval.pipeline` / `eval_ply_cap_adjudication` — the ply-cap criterion's in-run fire rate (F-R-P2B-5, LAW-18). ARMED-ONLY on the same terms as the row above, keyed off the worker result's `ply_cap_adjudication` key. Payload = `{event, round_id, step, criterion, min_margin, adjudicated, candidate, opponent, draw}`. The tally is the ADJUDICATOR's own count of the capped games it saw and how each resolved, carried up from the child rather than re-derived from the aggregates — an aggregate records a winner, never whether a rule or a criterion produced it. Same producer-without-consumer exclusion | `tests/eval/test_eval_posture_inert.py::test_an_armed_adjudication_round_emits_its_own_tally_event` (+ the adjudicator-side counter, `tests/arena/test_ply_cap_adjudication.py::test_the_adjudicator_counts_its_own_fires`) |
| `eval_round_device_memory` | event_literal | `eval.pipeline` / `eval_round_device_memory` — the eval CHILD's own device-memory readout, per round (RECAL-PREP, R308(g)(ii)). UNCONDITIONAL, unlike the two armed-only rows above, and the difference is the point: this is a MEASUREMENT the re-calibration sitting needs from every round, including the ones with no CUDA counters (`available: false`, every counter `null`) and the ones that stopped early. Dropping the unmeasured rounds would bias the series a growth verdict is taken over, and "this round had no counters" is itself a finding about a drive. Payload = `{event, round_id, step, device_memory}`, where `device_memory` carries `{available, device, round_id, phases[], round_peak_allocated_bytes, round_peak_reserved_bytes}` and each phase entry carries `{phase, t_mono_sec, max_memory_allocated_bytes, max_memory_reserved_bytes, memory_allocated_bytes, memory_reserved_bytes}`. Emitted from the CHILD's payload, never from a parent-side reading: the term is the child's, in the child's own process and CUDA context, and a parent-side substitute is exactly what under-measured it three times (0.881 -> 1.186 -> 3.529 GiB). **Not a `producer_manifest.yaml` monitor input**: no rule consumes it — its reader is the manual `python -m mantis.diagnostics.eval_child_memory`, the same posture `fusion_calibrate` has — so registering it would create the producer-without-consumer the two rows above are already excluded for | `tests/eval/test_eval_child_memory.py::test_cm01_phases_are_recorded_in_the_order_they_were_marked` (+ the reader's own refusals, `tests/diagnostics/test_eval_child_memory_reader.py::test_rd02_a_stream_with_no_events_of_the_expected_kind_refuses_by_name`) |
| `batch_fill_pct` | symbol (+`also`) | `selfplay.pool_hooks.batch_fill_pct` ← `selfplay.inference_server.InferenceServer.total_requests` / `.forward_count` | `tests/selfplay/test_inference_batch_timing.py::test_batch_fill_pct_reaches_the_sink_from_the_live_inference_counters` |
| `inference_batch_timing` | symbol (+`also`) | `selfplay.inference_server.InferenceServer.batch_timing_snapshot` + `selfplay.pool_hooks.inference_batch_timing` | `tests/selfplay/test_inference_batch_timing.py::test_the_batching_block_reaches_the_sink_on_iteration_complete` |
| `eval_rung_skip_class` | event_literal | `eval.pipeline` / `eval_rung_skip_class` — the fourth skip channel (WP12-R Phase A, LAW-18/R164: in-run FIRE RATE, not "a log line somewhere"). One event per skipped rung, emitted alongside its `eval_rung_skipped` through the same injected sink; payload `{event, round_id, rung, reason_class, class_count}`, where `class_count` is the running count for that class WITHIN the round. `reason_class` is drawn from the CLOSED partition `mantis.eval.pipeline.SKIP_REASON_CLASSES` = `operator_authorized` (the kraken/strix skip R139 ruled) / `vendor_absent` / `build_absent` / `load_failed` (the three ways a sealbot rung fails to resolve). Classification imports `mantis.bots.resolve.SKIP_REASON_MARKERS`, the same literals the refusal strings are built from, so the wording cannot drift out of the classifier's reach silently. A reason matching no marker (or more than one) emits NO class event and logs `eval_rung_skip_class_unclassified` at ERROR — a loud failure, never a fifth bucket invented at emission time. **Not a `producer_manifest.yaml` monitor input**: it is an event-stream counter with no consuming rule, and registering it as one would create a producer-without-consumer | `tests/eval/test_rung_skip_class_counter.py::test_each_skip_reason_class_counts_itself_in_run` (+ the closed-set / no-over-fire conjunct, `::test_the_class_set_is_closed_and_a_repeated_class_does_not_over_fire`) |

## WP-UNFREEZE rows (actor-sync ⊥ deploy-gate split; LAW-18)

| event | producer | producer test |
|---|---|---|
| `actor_sync` | `train.actor_sync.ActorSync.maybe_sync` — one event per successful sync; payload `{event, step, actor_ckpt_step, lag_steps_pre_sync, cadence_steps, sync_count, duration_ms}` (the cadence is a lever under test — it logs its own fire rate in-run) | `tests/train/test_actor_sync.py::test_actor_sync_event_carries_lever_fire_rate_fields` |
| `actor_lag_exceeded` | `train.lifecycle.heartbeat_watchdog.HeartbeatWatchdog._check_actor_lag` — armed: the `heartbeat_watchdog_fired` payload with `reason=actor_lag_exceeded`, exit code 45; disarmed: ONE `actor_lag_exceeded {armed: false}` event per exceedance episode (latched) | `tests/train/test_actor_lag_watchdog.py::test_rigged_lag_over_threshold_disarmed_emits_event_and_never_aborts` |
| `actor_lag_negative` | same check — a negative lag is a wiring bug reported loudly once, never a fire | `tests/train/test_actor_lag_watchdog.py::test_negative_lag_reports_wiring_bug_event_once` |
| `actor_lag_sample` | same check — the HEALTHY-path reading, emitted before either fire arm; payload `{event, seq, learner_step, actor_ckpt_step, lag_steps, threshold_steps}`, the SAME `detail` dict the fire path uses, so a sample can never disagree with the reading that fires. Gated on the interval already in the object (`file_interval_sec`), so one config fact never enters the ctor twice under two names. Before it, a healthy run published NOTHING about the lag reading and no observer could tell a live reading from a frozen 0 (LAW-18: a lever under test logs its own fire rate in-run) | `tests/train/test_actor_lag_sample_emission.py::test_a_healthy_poll_emits_an_actor_lag_sample_carrying_the_live_reading` |

The `heartbeat_watchdog_armed` payload additionally gains one key, `actor_lag`: either
`{armed: bool, threshold_steps: int}` or the string `"absent"` when no spec was injected —
a disabled or unwired lag check is loud at arm time, never silent
(`tests/train/test_actor_lag_watchdog.py::test_arm_event_names_actor_lag_posture`).

Coverage state at WP11-A landing (recorded, not silent — the intended operator posture):
WP13-A shipped **ZERO new default-active hard-aborts**; WP11-A lands the mid-run eval-
RESULT producer that row `sealbot_wr_warn` was pending on.
- `sealbot_wr_warn` ships **WARN-ONLY** (operator G-3): a sustained collapse emits a visible
  `sealbot_wr_warn` and does NOT stop the run; the one-field flip `wr_hard_abort_enabled=True`
  restores the A/B/C hard-abort as a capability. The producer LANDED at WP11-A
  (`eval.rounds.build_round_result` unconditionally sets `wr_sealbot`); an eval round that
  never resolves a sealbot rung (0/6 census verdict at HEAD — no adapters installed) still
  routes `wr_sealbot: None`, which the coordinator skip-counts loudly
  (`sealbot_wr_gate_skipped`), never silently.
- `draw_rate_collapse` is armed **by the config** (WPAX Phase D, R65/R80): the
  `train.draw_rate_abort` block — `threshold` / `min_step` / `N_pool_min` / `consec`, one
  block and one resolver — is the sole authority, and `null` is the EXPLICIT off posture
  (R79: arming is a property of the resolved value; there is no boolean beside it).
  `configs/run5.yaml` arms it at `0.25 / 25000 / 50 / 3` (R82/R85/R92, pre-registered at mint
  prereg); the four non-production configs carry `null` (R59). **WPMINT Phase K-B (R78/R80,
  call K-b) authored the FOURTH term**: `consec` was the coordinator's own code-side default
  `draw_rate_consec = 3` and is now `train.draw_rate_abort.consec`, inside the block because a
  term of a DISARMED abort is not a fact. Its value is unchanged, so nothing an observer reads
  moves; what moved is who says it. The `monitor_gates` event's `draw_rate_threshold` field keeps
  its name and now carries `null` on a disarmed run rather than the retired `0.0` spelling.
  **WPMINT Phase DS (R92) replaced the gated statistic** and with it the block's third key
  (`min_samples` -> `N_pool_min`): the metric is the POOLED COUNT-WEIGHTED rate
  `Sum(draws)/Sum(completed)` over the union of worker windows, and an interval with fewer
  than `N_pool_min` completed games yields NO OBSERVATION — skip-counted in the gate's own
  `monitor_gates` counters, never appended to the abort history as a healthy `0.0`. An
  observer therefore reads insufficient evidence as a rising `skips` count, not as a healthy
  reading.
  **WPMINT Phase X (CARD-ABORT-EXIT / R84) makes a fired abort supervisor-distinguishable.**
  Until it landed, `shutdown.running = False` was written by four sites in
  `train/coordinator/step.py` — `stop()`, the O2 iteration limit, the O3 shutdown-save and
  `_fire_hard_abort` — and nothing recorded which one fired: a collapsed run and a completed
  run were the same observable, in state and in exit status. `_fire_hard_abort` now records the
  RULE NAME on `ShutdownState.abort_rule` beside the stop (`None` on all three clean stops), and
  `mantis.config.armed_aborts.exit_code_for_abort` resolves that name to the manifest row's
  `exit_code` at a process boundary. The `draw_rate_collapse` row's `exit_code` is **46**
  (`monitor.heartbeat.DRAW_RATE_COLLAPSE_EXIT_CODE`, the fail-fast family's cooperative
  member — see repo_design §11). The `hard_abort` event is unchanged and still carries
  `rule` / `message` / `step`: the event stream was never the gap, the PROCESS was, and a
  supervisor that reads only exit statuses now sees 46 instead of 0. Delivery stays
  cooperative — the run still unwinds through `close_out`, the terminal-eval drain and the
  shutdown checkpoint — so the abort's own evidence survives the abort.
  A rule with no manifest row (`grad_norm_hard_abort`, `sealbot_wr_abort`) resolves to `None`:
  truthful, and no code is invented for an abort nobody pre-registered.
- `disk_space_exhausted` is the manifest's THIRD row and its second authored code, **47**
  (`monitor.heartbeat.DISK_SPACE_EXHAUSTED_EXIT_CODE`), REQUIRED, arming surface
  `monitor.disk_guard.fail_gb` (WPMAIN RED-TEAM RT-2 / R132). WPMAIN constructed the disk
  guard for the first time in any run (R121(b)/R122) and thereby armed LAW-16 leg 3; the
  RED-TEAM then measured what that armed. `DiskGuard.check_once` SIGTERMs its own pid below
  `fail_gb`, the handler sets `shutdown_save`/`running` and **never** `abort_rule`, and
  `mantis.run.main` read `abort_rule is None` and returned **0** — a run the disk guard killed
  reported success, and a supervisor reading only the rc relaunches into the same full volume.
  The event stream was NOT the gap: `disk_alert {level: "critical"}` was always emitted and is
  unchanged. The PROCESS was. Two things changed, and neither is an event: the guard latches
  its critical arm (it re-fired every `interval_sec` and so supplied LAW-16's two-press
  force-exit itself, `sys.exit(1)` mid-save — the critical `disk_alert` still emits every tick
  because the condition persists, only the SIGTERM is once-per-run), and `compose_run`'s
  teardown reads that latch after the guard thread is joined and records the rule through
  `ShutdownState.record_abort`, which is now THE one writer of that field for both fire paths
  (set-once, first fire wins). The guard never names the rule itself: the name is a manifest
  row's, `mantis.train` may not import the manifest, so the composition root — which already
  reads `exit_code_for_abort` — imports `DISK_SPACE_ABORT_RULE` and does the naming. Delivery
  stays cooperative for 46's reason exactly: an `os._exit(47)` would discard the save the
  guard fires to protect. Residual, disclosed: an OPERATOR's SIGTERM still resolves to rc 0 —
  nothing records a rule for a signal the process did not send itself, and R132's scope is the
  guard.
- `terminal_eval_broken` is the manifest's FOURTH REQUIRED row and its third authored code,
  **48** (`monitor.heartbeat.TERMINAL_EVAL_BROKEN_EXIT_CODE`), arming surface
  `train.terminal_eval_enabled` (WP12-R Phase O / R152, discharging R133's caveat "rc 0 does
  not certify eval health"). The event stream was not the gap here either: `eval_broken` was
  always emitted and is unchanged apart from its reason becoming typed. The PROCESS was.
  `drain.close_out` computed the terminal round's result, routed it, and DISCARDED the return
  value one frame below `ShutdownState`, so a terminal battery that was killed, returned
  garbage or could not persist its ladder state exited **0** and a supervisor recorded a clean
  finish (LAW-15: no promotion decision = deliverable incomplete). `drain.run_terminal_eval`
  now latches the routed result's own `eval_broken_reason` on the coordinator — set-once, ONE
  writer in `src/`, reachable only from the one function that passes `ignore_stride=True`,
  which is what keeps R133's mid-run/terminal split structural: a MID-RUN broken round still
  exits 0, because rounds recur and persistent breakage stays the watchdog's jurisdiction. The
  composition root re-parses the latched string through `EvalBrokenReason` (an unregistered
  spelling is a loud `ValueError`, never a silent rc 0) and records the rule AFTER the
  disk-guard read, so first-fire-wins keeps the root cause: 47 beats 48, and a mid-loop 46
  keeps its name. ONE code for seven reason classes by decision — the family is one number per
  OUTCOME with the cause in the payload — so the seven stay pairwise-distinguishable in THIS
  channel and never at the rc. Delivery is cooperative and is the cleanest of the three: the
  terminal eval is the LAST action of `close_out`, so there is nothing left for an `os._exit`
  to discard; `main` returns the number. Residual, disclosed: an exception raised BEFORE or
  DURING the terminal eval leaves the latch unset and exits rc 1 — loud, but indistinguishable
  from a composition wall (`Q-RT-RC1-COLLISION`).
- `iteration_complete.target_integrity` carries the three WP12-R Phase-T target-integrity
  counters IN-RUN (R164 / LAW-18): `export_offwindow_mass_moves`, `gridls_zero_policy_rows`
  and `target_integrity_defects` — plus, from R275(b), `inference_failures_total` — each
  `{total, delta, per_position}`, beside the `positions_delta` denominator. `PREREG_T §0b` names the first as THE in-run witness
  attributing the expected game-shape drift, and until Phase O it was readable only by a test
  calling `runner_stats(pool)` — a witness a live run cannot read is not a witness, and
  LAW-18's own text is that a post-hoc offline probe cannot distinguish "starved" from
  "ineffective". LAW-03, stated because the unit is not obvious: `per_position` is fires per
  RECORDED POSITION — not per game and not per ply — so `gridls_zero_policy_rows` can
  legitimately exceed 1.0 (one position contributes many cluster rows) and it is a RATE, never
  a fraction; no per-MOVE denominator is published, so a per-move rate is NOT re-derivable
  from this payload. `per_position` is `None`, never a fabricated `0.0`, when no position was
  recorded in the interval (the convention below, applied). An IDLE lever stays VISIBLE at 0:
  `target_integrity_defects` reads 0 in every run that survives to emit, because its latch is
  run-fatal — that permanent zero is the posture, not an unproduced field.
  `inference_failures_total` (R275(b)) shares that posture and that latch: it counts leaf
  inferences that FAILED on an OPEN queue, and a drain shutdown — which closes both queues
  before the waiters wake — does NOT count. It is the SEAM conjunct of the class the other
  three guard at the exporter, and the two are meant to be read together: advanced with
  `target_integrity_defects` at 0 says the run died at the seam before any target was built.
  It is published on EVERY encoding — both `infer_and_expand` arms carry a failure leg, so
  the R250 absence rule does not apply (mapping re-derived from code, R256). A DECREASE is
  emitted as measured and never clamped: the atomics are monotonic, so a negative delta is a
  wiring bug and a `max(0, …)` would hide it (the `actor_lag_negative` precedent below).
- `iteration_complete.inference_batching` is the Q3 in-run batching instrument (LAW-18),
  and `iteration_complete.batch_fill_pct` gains the manifest row it shipped without.
  `batch_fill_pct` has published a mean batch occupancy on every `iteration_complete` since
  WP13-A, fed by `InferenceServer._total_requests` / `._forward_count` against
  `._batch_size`, with nothing making either counter's disappearance un-shippable. The new
  block carries what the ratio cannot resolve: `queue_wait` (the wall time inside
  `InferenceBatcher.next_graph_batch` — EXACTLY the Rust collector's
  `batch_size / 2`-or-`max_wait_ms` wait, `crates/mantis-selfplay/src/queues/graph.rs`),
  `collate` (the `collate_graph_batch` cost) and `occupancy` (count/total/mean plus
  min/max and a power-of-two histogram, because a mean cannot separate "always 1" from
  "sometimes 64, sometimes 0"). Each timing sub-block publishes RAW `count` + `total_ms`
  beside the derived `mean_ms`, so a consumer differences two consecutive events to get an
  INTERVAL mean; min/max are run-cumulative extremes and do not difference. **Graph path
  only** — the dense loop is not instrumented, so a grid run's block carries `None` for
  every derived reading (the unproduced-field convention below, applied), while
  `empty_polls` stays VISIBLE at 0 on the producing path.
  The block carries a `fusion` SUB-BLOCK (F-816-10, LAW-18/R164) — the graph inference
  forward's memory bound reporting its own fire rate in-run: `caps` (the two minted members
  of `inference.fused_graph_caps`), `fusion_parts` (GPU forwards actually run),
  `fusion_splits` (**pops that split** — the lever's own fire rate, not a cut count),
  `fusion_bound_hits` (`{edges, nodes}`: which member forced each cut, so the reading says
  which member to re-fit) and `fused_batch_nodes`/`fused_batch_edges` (count/total/mean plus
  min/max and a power-of-two-lower-bound histogram). **The unit is the PART, not the pop** —
  the part is what the GPU sees and what the cap bounds, and a pop's total is recoverable as
  the sum over its parts while the reverse is not. Two adjacent readings move as this lands,
  both intended: `collate.count` becomes the part count (`sum(M)`) where it previously
  equalled `queue_wait.count`, and `batch_fill_pct`'s denominator `_forward_count` stays ONE
  PER POP, because that metric is an occupancy (requests per pop against
  `inference_batch_size`) and not a GPU-forward count. **`fusion` is `None` on a grid run**,
  key present and value null: the dense batch is a fixed-shape tensor already bounded by
  `inference_batch_size`, so the grid path never reads the caps and never plans a split — a
  zeroed block would read as "the lever ran and never fired", which is the opposite
  statement (the unproduced-field convention below, applied). `graph_build_time` is
  deliberately NOT built: it lives per-leaf in Rust (`mantis-graph::build_axis_graph`) and
  would be a LAW-09 hot-path change owing re-run parity oracles and an IQR-gated bench.
  Registration follows the `target_integrity_counters` precedent (R164) rather than the
  `eval_rung_skip_class` exclusion above; those two rulings disagree about whether an
  in-run instrument with no consuming gate rule belongs here, and an operator ruling
  collapsing them is owed.
- `stride5_spam` was **REMOVED** at close-out (operator directive B — a dead artifact of bad
  hyperparams that never occurs under current recipes).
- `eval_round` joins the heartbeat sources at WP11-A (4th source): the eval pipeline's
  persistent poller thread beats it every tick, in or out of an active round.
- `eval_round_complete`'s routed `gate.elo_ci_lower_boot` field (fed from
  `eval.aggregate.aggregate_gate`, consumed by `gate_promotion_decision`'s `ci_lo_boot`
  parameter) is NOT an Elo-scale bootstrap bound (deviation #5, FIX-PASS document-the-unit
  ruling): it is the pooled distinct-game draw-aware WR bootstrap's LOWER bound,
  RE-CENTERED to the Elo zero-point — `wr_lower_boot - 0.5`, so its range is `[-0.5, 0.5]`,
  not Elo points. It is DECISION-EQUIVALENT to a literal per-resample BT/Elo bootstrap for
  the promotion test: for the 2-entity candidate-vs-best comparison, any monotone
  transform commutes with quantiles, so `wr_lower_boot - 0.5 > 0 ⟺ Elo_lower > 0` — the
  `ci_lo_boot > 0.0` truth-table cell `gate_promotion_decision` reads is bit-identical
  either way. The field KEEPS its historical name (`elo_ci_lower_boot`) for run3-parity
  continuity; a future consumer must read the VALUE as a re-centered win-rate bound, never
  as Elo points.

`grad_norm_hard_abort` has an armed-abort manifest row since **WPMINT Phase K-B** (call K-c),
and it is **DEFERRED**: its threshold is `train.hard_gn_threshold`, authored by the same phase
and minted at `1e9`, which no finite gradient norm reaches — so the gate is LIVE, fires through
the same `_fire_hard_abort` contract as every other, and is effectively OFF. The row makes that
visible instead of silent: gate 12 prints it loudly on every run and gates nothing, because a
REQUIRED row would demand a number nobody pre-registered (R84's class). Its predicate is
`Mechanism.CONFIG_THRESHOLD_BELOW_CEILING`, which reads its ceiling off
`monitor.alert_grad_norm_max` — a hard abort set orders of magnitude above the line the run
already WARNS at is not a hard abort. Its `exit_code` is `None`, truthfully: R84 authored a
code for the draw-rate family only. Closing the row is a mint-prereg value plus a one-field
flip to REQUIRED.

The one gate LIVE the moment a coordinator runs is `grad_norm_hard_abort`. The heartbeat
watchdog, persist-fatal and the heartbeat file are code-complete and oracle-tested but are
CONSTRUCTED ONLY by `train.subsystems.build_run_safety`, which has no caller yet (the full-run
launch entry is not WP13-A's property) — they arm when the run wiring lands. Every warn/inert
state is named per-gate in each `monitor_gates` event (checks/fires/skips/warns), so nothing
is silently disabled.

## Event stream conventions

- Every event is one JSON object per line; the event NAME travels under the `"event"` key;
  `ts` (wall clock) is stamped by the sink iff the producer did not supply one.
- The first line of every segment is `run_segment_started` carrying
  `{run_id, segment, pid, created_utc, contract: "event-manifest-v1"}`.
- Log identity: one segment file per process start
  (`events_<run_id>_seg<NNNN>.jsonl`) — a JSONL file NEVER spans two run segments
  (§11 rotation-on-resume). The segment is claimed ATOMICALLY (`O_CREAT|O_EXCL` over
  `max(existing)+1`, with a bounded re-scan on collision), so N concurrent starts claim N
  distinct files; `run_id` is validated where the filename is built (empty / path
  separator / `..` / control chars are rejected LOUD, `RunIdError`), because the schema
  pattern is enforced by a caller that does not exist yet.
- A missing `"event"` key is a producer BUG (loud `ValueError`), not a persistence failure.
- Persistence failures are COUNTED (`persist_errors_total`) and made run-fatal by the
  heartbeat watchdog (exit 43) — never swallowed (LAW-14). The rule is the literal
  `persist_errors_total > 0`, evaluated on every poll from the LIVE module attribute: there
  is no baseline and no forgiveness window, so an error raised before the watchdog was armed
  still aborts at the first poll.
- **An unproduced field carries `None`, never a fabricated value.** A constant `0` in the
  ONE channel reads as a real measurement and is the F-10 class in miniature. Currently
  unproduced: `training_step.quiescence_fires_per_step` (the solver-delta half is
  DEFER/ARCH; the key is retained for schema stability and travels as `None` = NOT
  MEASURED). A consumer must treat `None` as "no producer", never as zero.
- **The R250 absence family (this rule's ONLY exception): a key whose MECHANISM the active
  encoding does not have is ABSENT from that encoding's stream — never zero, never
  `null`-as-value.** Every `iteration_complete` block subtracted on these grounds (the
  numbered rows below) is keyed on the SAME authority (`mantis.train.events.is_graph_run`,
  which reads the run's declared `identity.representation`), so they cannot disagree about
  which arm the run is on. The `None` convention above still governs every OTHER key,
  including these blocks when their mechanism EXISTS but has no producer wired.
- **(1) ADJ-D32 (R249 + R250): the cluster block.** Its three keys — `cluster_value_std_mean`,
  `cluster_policy_disagreement_mean`, `cluster_variance_sample_count` — have three arms:
  - **GRAPH representation** — all three OMITTED. The cluster-variance accumulators are
    structurally unreachable on that arm (the search drive returns into the graph inference
    path before any variance code runs, and the atomics are not passed to it), so there is
    no producer to report `None` for. Absence rather than `None` because these keys had
    already shipped as hard `0.0`s for a whole run: a key that has carried a number is read
    as one, and `null` is what a JSON consumer coerces back to zero most readily.
  - **PUCT + grid** — `cluster_variance_sample_count` ALWAYS present (a raw counter,
    truthful at 0, and the evidence for the drop); each derived mean present only when the
    producer supplied one, DROPPED per-field when the bridge returns `None` at zero samples.
  - **Gumbel + grid** — the CONFRES S2 convention is RETAINED: all four regime-gated keys
    (the three above plus `mcts_root_concentration`) stay present carrying `None`, so the
    payload shape is regime-stable. That `None` is a REGIME-`None` ("no such instrument
    under this descent"), not R249's zero-count `None`. Whether R249's drop should extend to
    it is UNDER ADJUDICATION; until ruled, S2 stands.
  `mcts_root_concentration` is NOT a cluster field — it is accumulated once per search,
  path-independently — and stays on both representations, subject only to the S2 regime gate.
- **(2) Item 10(b) (R250): `iteration_complete.k_cluster_histogram`.** The LAW-18 fire-rate
  log for the K-cluster lever: a mapping from K (cluster views per recorded position) to the
  cumulative count of positions recorded at that K, over buckets `"1"`..`"8"` plus a `">8"`
  guard for any K outside the registry's `k_max`. Three arms:
  - **GRAPH representation** — the key is OMITTED. The only writer is
    `record_position` on the dense arm; `record_position_graph_dispatch` does not take the
    histogram as a parameter, so a graph run's buckets are zero for want of a producer. A
    histogram of zeros is a WORSE fabrication than a scalar zero, because it has shape and
    therefore reads as a measured distribution.
  - **No producer** (an engine build predating the getter) — keyed, carrying `None`, per the
    unproduced-field convention. This is the case the R250 subtraction is NOT.
  - **Grid** — the bucket mapping, cumulative since pool start. The labels are derived from
    the vector's LENGTH, so widening the bucket array in Rust relabels the payload with no
    Python edit. LAW-03: the unit is RECORDED POSITIONS; the buckets sum to the dense
    `record_position` call count, so the distribution is self-normalising and no separate
    denominator ships beside it.
- **(3) R256/ADJ-D37: `iteration_complete.uncovered_forced_win`.** The LAW-18 fire-rate
  log for the forced-win coverage clip: per-lever DROP EVENTS — a proven forced win
  swallowed by the K-cluster WINDOW criterion while the injecting lever (O1
  `forced_win_policy_weight` or the solver's `solver_visit_weight`) was armed. LAW-03 unit
  note: the unit is lever-drop events, not distinct wins — a move with BOTH levers armed
  can contribute two ticks (each armed lever independently dropped an injection).
  Disclosure: every shipped config disarms both levers (`forced_win_policy_enabled:
  false`, `solver_enabled: false`), so on minted runs this reads a truthful 0 until a
  prereg row arms one — the instrument pre-positions for that re-arm (R163's
  recommendation), per LAW-18. Producer: the ONE counted helper
  `records::apply_forced_win_one_hot_ls_counted` (both mechanism sites route through it;
  producer + mutation self-tests in `records::ls_tests`, Python seam pins in
  `tests/train/test_uncovered_forced_win.py`). Three arms — the K histogram's gate,
  INVERTED, on the same `is_graph_run` authority (R256: the instrument attaches to the
  mechanism's measured live path, the LS target path, live on run5's graph arm):
  - **GRAPH representation** — cumulative `{"total", "per_position"}`; `total` is a raw
    counter, truthful at 0; `per_position` is the rate over the snapshot's cumulative
    `positions_generated`, `None` before any position is recorded (a rate over zero
    samples is not a measurement, R249).
  - **Dense (grid) representation** — the key is OMITTED. Publishing it would resurrect
    the ADJ-D37 arm-(i) trap: a `{total: 0}` reading zero on arms whose forced-win drops a
    DIFFERENT mechanism owns (`v6`/`v6w25` take the Dense target arm). Disclosed:
    `v6_live2_ls` is itself LS, so its Rust-side counter can tick while emission stays
    graph-scoped per R256's explicit landing — the dense-LS stream gap is an adjudication-
    queue disclosure, not an oversight.
  - **No producer** (an engine build predating the getter) — keyed, carrying `None`, per
    the unproduced-field convention.
- **(4) R266/F-P1/N1 (fdc6f09/R245(c)): `training_step.symmetry_draws`.** The LAW-18
  fire-rate log for the per-record compact/spread symmetry gate: a record that is
  window-lossless under every D6 element draws from the FULL 12-element group, one that
  is not draws only from `sym::WINDOW_PRESERVING_SYMS` (4 elements) — previously a silent
  restriction with no in-run reading of how often each arm fires. Producer: the ONE
  counted call site both sample cores route through
  (`crates/mantis-selfplay/src/replay/sample.rs::record_symmetry_draw`; producer +
  mutation self-tests in `crates/mantis-selfplay/tests/replay_compact_gate.rs`, Python
  seam pins in `tests/train/test_symmetry_draws.py`). Ticked ONLY on an `augment=True`
  draw (the b349ec4/R249 disarmed-lever posture: an unaugmented draw never consults
  `compact`, so counting it would fabricate a reading for a lever never exercised). Three
  arms — the K histogram's gate (item (2)), NOT inverted (this mechanism, like the K
  histogram, is DENSE-only — the graph arm has no window and keeps the full group
  unconditionally, `sym::WINDOW_PRESERVING_SYMS`'s own doc):
  - **DENSE (grid) representation** — cumulative `{"compact", "spread", "compact_fraction"}`;
    the two raw counts are truthful at 0 (R249); `compact_fraction` is `None` until at
    least one augmented draw has landed (a rate over zero samples is not a measurement).
  - **GRAPH representation** — the key is OMITTED. The mechanism has no subject there
    (publishing a keyed zero or `None` would both read as "measured" to a stream
    consumer, the same D37/10(b) arm-(i) trap).
  - **No producer** (an engine build predating the getters) — keyed, carrying `None`, per
    the unproduced-field convention.
- Known counter overlap (debt **R-QUARANTINE-COUNTER**): `checkpoints.persist_errors_total`
  is incremented BOTH by a fatal write failure and by a deliberately survivable quarantine
  write (`checkpoints.py::_write_quarantine`, the §6/R3 survive-run clause). Under the
  literal `> 0` rule a quarantine write therefore aborts the run at 43. Splitting the
  counter (fatal-write vs survivable-quarantine) is owed before the run5 mint; until then
  an unstampable save is treated as run-fatal, which is the safe direction (it is itself a
  provenance defect).

## Pinning tests

- `tests/monitor/test_manifest_contract.py` — O-01 (shipped manifest resolves) + O-02
  (both LAW-07 mutation arms, the quoted-literal arm and the pending-WP arm).
- `tests/monitor/test_monitor_census.py` — the import/torch/swallow/phantom-token censuses
  around the manifest's owning package.
- `tests/monitor/test_sink.py`, `tests/monitor/test_rotation_on_resume.py` — the channel's
  own contract (line atomicity, `ts` stamping, segment rotation-on-resume).
- `tests/train/test_coordinator_gates.py`, `tests/monitor/test_persist_fatal.py`,
  `tests/train/test_lifecycle_contract.py` — the producer tests the rows cite.

## Trainer-side narration literals (delivered since F-R-P2B-2; catalog note, not rows)

The production Trainer emits through the composed sink since the F-P4 sink threading
(`run.py` builds it with a `_DeferredSink` bound to the channel beside the pool's). Three
literals ride the stream from the trainer itself; none is a gate/monitor INPUT, so none
gets a `gates:` row — their producer tests live in pytest (LAW-07's half that applies):

- `periodic_checkpoint_save` — one row per crossed `train.checkpoint_interval` boundary,
  emitted AFTER the write with the writer's returned path (R173 seam;
  `tests/train/test_periodic_checkpoint.py` + the composed-stream drive in
  `tests/test_run_launcher.py`).
- `trainer_step` — the trainer's OWN per-step diagnostic row (dense and graph tails,
  `trainer/core.py`). Deliberately a DISTINCT literal from `training_step`: the
  coordinator's `log_interval`-gated narration above owns that name and its one documented
  shape. `trainer_step` is per-step and ungated — at `max_train_steps: 1e6` that is ~1e6
  rows into one segment; the volume/retention posture is an open operator row
  (FINDINGS_F-P4), recorded here so a reader of this catalog knows the cadence is a
  decision still owed, not a contract.
- `aux_chain_loss` — the chain-loss lever's per-step fire-rate leg (LAW-18), emitted only
  when the lever is armed (`train.aux_chain_weight > 0`).
