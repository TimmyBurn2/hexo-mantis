# STATE — where mantis actually is

Rewritten in place, never appended to. Every value below was read from the tree at the commit
named under "Provenance", not copied from a register. Where a register disagreed with the tree,
the tree won and the disagreement is recorded in the last section.

## Current phase — R352 LANDED; run7 RE-MINTED (run6's Gumbel trainer, every head, the ply-cap halt, the sealbot GIL release, round timeout 14 400) and STAMPED on `15109ac3`; VIEWER-1 and RUNG-2 LANDED and measured; START is HELD BY THE OPERATOR

**R352 is landed verbatim** (`60b752ad`), the operator's forward of the run7-kind packet (dated
2026-09-14): shakedown7 falsified PUCT self-play at the minted regime (**F-52**); run7 is run6's
Gumbel-320/64 trainer with exactly one change — every head loaded — plus the instrument fixes
(PUCT-512 reader, 3 000/288 cadence, `eval.rung_concurrency 8` RATIFIED), the value warm-up
dropped to 0, the ply-cap HALT ordered, RUNG-2 (strix) and VIEWER-1 ordered. R351's status line
marks (c)'s PUCT arm and (d)'s warm-up superseded for run7.

**The tree at the exit (`dev`, pushed):** `60b752ad` R352/F-52/cards · `d13bc7c3` the ply-cap
halt + the re-mint (contract v29) · `2d227ed6` VIEWER-1 · `56b51004` RUNG-2 · `4e0fd430` floors ·
`9719f586` stamp record · `de536a2d` timeout 10 800 · `53cd235c` CARD-SEALBOT-GIL-SERIAL ·
`66fabaf3` STATE (checkpoint) · `86335a0d` the grave-guard and Makefile-census fixes ·
`1b44d442` GAME-QUALITY · `180ba3bd` CARD-SEALBOT-TT-SEAT · `01390697` STRIX-RUNG step 0 ·
`2177c926` **the sealbot GIL release** (patch hunk 3) + SHAKEDOWN7G · `15109ac3` timeout 14 400 ·
this STATE. Full local gate set on `2177c926`: **ALL GREEN, 20 gates, slow tier ran** (3a 4 743
passed, 3b 47 passed in 50 min); the one commit above it is a mint + docs, checked by the config
tests (806 passed) and gates 7/12/13/10/17.

**What the leg measured, in order:**

1. **The re-minted trainer IS run6's regime, faster** (`SHAKEDOWN7G_2026-09-14.md`): 3 938 steps in
   3 h (1 630 steps/h before its round, 1 310 beside it; the block: 1 163), 22–37-ply games, 0.7 %
   draws, the ply-cap window ≤ 0.015 against a halt at 0.5 (LAW-18: `monitor_gates` carries it
   at every boundary), step 3 000 at +1.98 h, the step-3 000 net 0.70 vs its step-0 anchor.
2. **The eval round did not fit its bound under contention:** the twin's escalated round
   projected to ≈ 3.2 h; the cause was the vendored sealbot holding the GIL through its search
   (48 idle rung minutes for 288 games of 0–3 s sealbot moves). ON THE OPERATOR'S DIRECTION
   ("do the cleanest thing", 14:35 UTC) the GIL release landed as the tracked patch's third hunk
   (`2177c926`; no move, score or depth receipt changes — the vendored suite's 21 tests pass on
   the new build, the release pinned by a tick-rate probe RED on the old one), rebuilt here and on
   the box. **The rung fell from 48 to 21 min** (0.23 → 0.12 s/ply); the gate block's per-ply cost
   rose 0.20 → 0.30 s on an idle card, unexplained, so `eval.round_timeout_sec` was minted at
   **14 400** (the drain's hard cap) rather than 10 800: a long round delays the next kick, a
   killed one loses its reading (prereg §7–§8).
3. **Four stamps of the re-mint, every one PASS and MIRRORED**, `sealbot_d5` at PUCT-512 after
   101 Gumbel steps on every head: 0.507 (`d13bc7c3`), 0.597 (`de536a2d`), 0.642 (the twin),
   0.635 (`2177c926`), **0.635 (`15109ac3`, THE START STAMP: `preflight_run7_20260914T161609Z.json`,
   config identity `0ad6b333…`, workspace MIRRORED to `mantis-mirror/run7-preflight`, checkpoint
   `run7_00000101_fcedc217.ckpt`; its round 2 582 s: screen 21 min at 0.41 — no escalation —
   the rung 21.4 min, 183/288)**. The PUCT-mint net read 0.271 on the same construction.
4. **RUNG-2 at step 0** (`STRIX_RUNG_2026-09-14.md`): the BC net 0.083 [0.052, 0.122] vs strix
   as-shipped (128/m16, ours PUCT-512), 0.073 at equal work (256/256), 0 fence findings in 576
   games, 12 % seat-decided pairs (sealbot cells: 37–57 %), 7–8 s/game at 8 in flight — the
   instrument has ≈ 90 pp of range where sealbot has ≈ 35; the 15 000-step cadence is confirmed.
5. **GAME-QUALITY** (`GAME_QUALITY_2026-09-14.md`, the operator's question from the viewer): run6
   self-play is tactically honest (0 missed wins, 3.6 % unanswered fours vs humans' 14 %); "PUCT
   exploits sealbot" is not supported — every strong player beats sealbot d5 by continuous fours
   past its horizon (strix-128: 30–2); the Gumbel DEPLOY head leaves 3 of 4 opponent fours
   standing (CARD-GUMBEL-HEAD-RESIDUE's symptom). Two instrument findings became cards:
   **CARD-SEALBOT-TT-SEAT** (the engine's transposition table persists across games and is keyed
   without the root player — every sealbot reading on record went through it; size unmeasured)
   and the mate-claim non-proofs.
6. The LAW-16 save did NOT fire under `timeout` on the twin (an eval child in flight); it had on
   shakedown7 — CARD-SHAKEDOWN-TIMEOUT-STOP's shape again; START's supervisor path is the clean one.

**START IS HELD BY THE OPERATOR** (2026-09-14 16:20 UTC: "you do not start run7 yet"). Everything
for it is staged: the box `/workspace/hexo-mantis` on branch `r352d` at `15109ac3` (`make
build.cuda`, never a bare `uv sync`; the vendored sealbot rebuilt with the release; strix vendored
with its venv and the checkpoint), `/workspace/runs/run7` ABSENT (a clean start),
`/workspace/oc7/box_start_run7.sh` (the supervisor over `mantis.run`, out-dir `/workspace/runs/run7`),
the puller `tools/mirror_pull.py --source <box-alias>:/workspace/runs/run7 --mirror
<mirror-root>/run7 --run-id run7 --interval-sec 600` under a persistent unit on the operator's
machine (a session-bound puller dies with the session). The first in-run round comes
at step 3 000 ≈ 2 h after START; its wall against 14 400 is the first live confirmation of §2.

**Dispatcher state for a fresh session.** Open, in order: (1) START on the operator's word, as
staged; (2) the strix point at 15 000-step intervals on the mirrored checkpoints
(`/workspace/oc7/cells_strix_step0.json`'s shape with the checkpoint path as `candidate`,
`/workspace/oc7/run_cells.sh`); (3) the balanced opening book (`book_v2`, judge-balanced, for
strix cells and the next run — run7's rung stays on v1 for comparability with run6's curve);
(4) CARD-SEALBOT-TT-SEAT's A/B (a ruling); (5) CARD-GUMBEL-HEAD-RESIDUE. Owed still: the α = 1.0
three-row reconstruction, `CARD-SELFPLAY-SEARCH-STATS`, `CARD-DRAIN-POLLER-RACE`. run6's record
stays under `/workspace/runs/run6/`; the viewer over run6, shakedown7 and the run7 stamp games is
`mantis-mirror/viewer/` (serve the mirror directory; `?g=run/game_id&ply=N&heat=1` links); the
dashboards `mantis-mirror/dashboard/`.

## PERF-A4 — the serving levers, merged at `41a5fea8` (branch `perf-a4`, red-teamed)

Record: `docs/design/measurements/PERF_A4_2026-09-11.md`. Four levers landed: check 14 with the GIL
released (+17 %), pinned `non_blocking` H2D + `output_size=`, `inference.compile_trunk` (+17–21 %),
the two-thread software pipeline (+52 %). Combined at 32 workers: leaves/s 1,831 → 3,804
(+108 %). The block measured **1 163 steps/h whole** at 32 workers on the box
(`RUN6_BLOCK_2026-09-12.md`). The edge codebook is NOT landed (device-dependent numerics).

## Hold 1 — the stamp trap and run6's burst floor: DECIDED by matrix, landed at `652b9f02`

`compose_run(burst_stop_step=)` is the eighth census parameter; a production config stamps at
`sync_lag` from a ≈ 100-step burst (≈ 6 min on the box under the Gumbel mint, 12 under PUCT).
`CARD-STAMP-FLOOR` carries the rejected options. Phase W is NOT re-run before a start
(`CARD-PHASE-W-AT-GUMBEL`).

## Hold 2 — the completed-Q target in decided positions: RULED by R350(e)

All 25 α = 1.0 rows of the START-path burst sat at `moves_remaining == 1` in LOST positions
(F-45); the block read 4.98 per 1 000, flat. R350(e) EXCLUDES them from the policy loss from this
tree on (`exclude_alpha_full_rows`, before the denominator; `trainer_step.policy_rows_excluded_alpha_full`).
The three-row reconstruction R349(c) ordered is still OWED. The re-mint's burst emitted 19
`alpha_full_row` events in 101 steps.

## R349(b) — the mirror arm, landed and proven

`tools/mirror_pull.py` on the operator's machine, receipts beside each artifact on the box, the
preflight's rc 16 reading them. run6's mirror is `~/Work/HeXO/mantis-mirror/run6` (36 bundles,
153 shards); shakedown7's is beside it; the game shards of both are mirrored (the viewer's
subjects). The run7 puller runs under a session for the stamps; for the run it is the operator's
tmux/systemd unit.

## Integration tier: CUDA where a card exists (R349(a))

fp32 on `train.device: cpu` is the ONE carve-out to LAW-06 (`tests/train/test_law06_cpu_carveout.py`).
The dev box runs the CPU wheel; the tier's authority is the box.

## Minted values — `configs/run7.yaml` (and `configs/run6.yaml`, a finished run's record)

`run7.yaml`: **38 deltas** from the `dev` template, replayable (`tools/config_diff.py --from-header`
MATCH). `selfplay.search.kind: gumbel` (320/64 at p 0.25, `c_visit` 50, `c_scale` 1.0, `q_rescale`
true, `gumbel_m` 16, `fast_policy_weight` 0.0, `temp_min` 0.5), `deploy.search.kind: puct`,
`train.policy_target: completed_improved_policy`, `identity.warm_start: {checkpoints/bc/run6_00006500_ca1afb71.ckpt,
2e72abd4…, reinit: []}`, `policy_loss_weight_schedule.warmup_steps` 0, **`train.ply_cap_abort {rate:
0.5, window_games: 600, min_step: 3000}`**, `train.eval_interval` 3000 = `checkpoint_interval`,
`eval.gate.deploy_sims` 512, `eval.sealbot_model_sims` 512, the sealbot rung at `games_max` 288 with
`round_games` 288 and calibration 288 every round, `random_floor_games` 20, `gate.stride` 1,
`eval.concurrency` 8, `eval.rung_concurrency` 8, **`eval.round_timeout_sec` 14 400**,
`policy_loss_trough_abort: null`, `draw_rate_abort {0.25, 25000, 50, 3}`, `supervisor_kill_grace_sec`
120, `n_workers` 32, the PERF-A4 serving rows, `actor_lag_abort_enabled` true, `seed` 20260914.
Gate 12 is GREEN with `policy_loss_trough` and `ply_cap_attractor` printed DEFERRED. `run6.yaml`
gained only the template's `ply_cap_abort: null` row (no delta; it stays the truth of what run6 ran).

## Protected set, laws, cards

Protected set as listed in `docs/governance/LAWS.md`; seventeen laws (LAW-06 amended by R349(a));
no gate number added (still 17). Cards (`docs/governance/CARDS.md`): opened by R352 —
`CARD-PUCT-ATTRACTOR`, `CARD-GUMBEL-HEAD-RESIDUE`, `CARD-SEALBOT-GIL-SERIAL`; opened by R350 and
still open — `CARD-DRAIN-POLLER-RACE`, `CARD-SELFPLAY-SEARCH-STATS`; `CARD-STOP-DRAIN-VS-GRACE`
fixed in code (the grace relation is minted); `CARD-WARMSTART-CONTROL` closed on the frontier.
STRENGTH-FRONTIER-1 is COMPLETE; INVESTIGATION-1's trough item is measured; TEST-1 is not
dispatched.

## Exit facts — the R352 packet, 2026-09-14

- Ruling: R352 at `60b752ad`, verbatim; numbering continues at R353. R351's status line marks
  its superseded clauses. Cards opened: CARD-PUCT-ATTRACTOR (answered by GAME-QUALITY; closes by
  ruling), CARD-GUMBEL-HEAD-RESIDUE, CARD-SEALBOT-GIL-SERIAL (the lever applied; closes on run7's
  first in-run round wall), CARD-SEALBOT-TT-SEAT.
- Full local gate set (`tools/ci_gates/run_all.sh --with-slow`, dev box, CPU wheel) on
  `2177c926`: ALL GREEN, 20 gates; gate 1 not run (the accepted cost).
- Collected tests: 4 729 → **4 804** (the floor follows). Comment ratchet: 3 534 / 23 / **13 510**.
- Contract: `run_config_schema.md` v29 (164 leaf key-paths, gate 13 green); `event_manifest.md`
  carries the three `monitor_gates` ply-cap keys; `eval_instrument.md` the strix rung and the
  patch's third hunk; `repo_design.md` the viewer's amendment and exit code 50; `vendor/pins.toml`
  the strix pin (commit + two sha256s; the unsupplied `config.toml` disclosed in the pin).
- Vendored patch: `vendor/patches/sealbot.patch` gained its third hunk (the GIL release around the
  search); nothing is pushed to the SealBot repository, ever.
- The box: `r352d` at `15109ac3`; run7 stamped (identity `0ad6b333…`), START HELD; the game
  shards of run6, shakedown7, shakedown7g, the run7 stamps and the strix cells are mirrored.
- Commits on this line: 17, one line each, empty bodies, zero trailers; `dev` pushed to `origin`
  at the leg's exit on the operator's approval.

## Provenance

Derived 2026-09-14 on `dev` at the leg's exit commit, from `configs/run7.yaml`, `configs/run6.yaml`,
the box's `/workspace/runs/run7-preflight/preflight_run7_20260914T161609Z.json` and its mirror
(`mantis-mirror/run7-preflight`), the four stamps' `r000001_101_terminal_progress.txt`, the
shakedown7g record,
`tools/ci_gates/test_count_floor.txt`, `tools/ci_gates/comment_length_floor.txt` and
`docs/governance/LAWS.md`. Ruling texts: `docs/governance/RULINGS.md`.
