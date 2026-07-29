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
| gate fire/skip/warn counts are visible IN RUN (LAW-18) | `train/coordinator/step.py::_emit_monitor_gates` (`monitor_gates` event) | `tests/train/test_coordinator_gates.py::test_log_interval_emits_training_step` |
| sealbot-WR ships WARN-ONLY (operator G-3): a sustained collapse emits a visible `sealbot_wr_warn` and does NOT stop the run | `train/coordinator/step.py::on_eval_round_complete` (`sealbot_wr_warn`) | `tests/train/test_coordinator_gates.py::test_sealbot_default_is_warn_only_and_does_not_shut_down` |
| the sealbot hard-abort remains a one-field capability (`wr_hard_abort_enabled=True`) | `monitor/rules.py::check_sealbot_wr_hard_abort` | `tests/train/test_coordinator_gates.py::test_sealbot_hard_abort_capability_when_enabled` |
| a heartbeat source nothing wired is loud, never a stall abort | `train/lifecycle/heartbeat_watchdog.py` (`heartbeat_source_unwired`; `heartbeat_watchdog_armed.unwired_sources`) | `tests/train/test_heartbeat_watchdog.py::test_undeclared_never_beaten_source_warns_instead_of_firing` |
| a teardown wedge is bounded (close-out deadline) | `heartbeat_watchdog.py::_check_close_out_overrun` (`close_out_timeout`) | `tests/train/test_heartbeat_watchdog.py::test_close_out_overrun_fires_after_the_teardown_budget` |
| the fire path reaches `exit_fn` even if an effect HANGS | `heartbeat_watchdog.py::_bounded` (`heartbeat_watchdog_fire_complete`) | `tests/train/test_heartbeat_watchdog.py::test_hung_snapshot_still_exits_within_a_bounded_time` |
| an eval-result shape the seam cannot consume is recorded | `train/coordinator/drain.py::_route_eval_result` (`eval_result_unroutable`) | `tests/train/test_run_safety_wiring.py::test_an_unroutable_eval_result_is_recorded_loudly` |
| an inert gate is loud, not silent (skip counter) | `step.py::on_eval_round_complete` (`sealbot_wr_gate_skipped`) | `tests/train/test_coordinator_gates.py::test_sealbot_absent_key_skips_and_counts` |

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
| `eval_broken` | event_literal | `eval.pipeline` / `eval_broken` | `tests/eval/test_eval_broken.py::test_killed_worker_yields_eval_broken_and_clean_drain` (+ reason `round_completion_error` — RED-TEAM-FIX WP11-A F1 layer 2, ANY uncaught exception in round completion/scheduling converts to a delivered `eval_broken` rather than killing the poller thread silently — `tests/eval/test_round_completion_error.py::test_poller_thread_survives_an_uncaught_exception_in_round_completion`) |
| `eval_rung_skipped` | event_literal | `eval.pipeline` / `eval_rung_skipped` | `tests/eval/test_rung_loud_skip.py::test_unresolvable_rung_emits_skip_event_and_log` |

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
