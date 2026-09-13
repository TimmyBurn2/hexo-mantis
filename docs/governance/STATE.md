# STATE — where mantis actually is

Rewritten in place, never appended to. Every value below was read from the tree at the commit
named under "Provenance", not copied from a register. Where a register disagreed with the tree,
the tree won and the disagreement is recorded in the last section.

## Current phase — R351 LANDED; the four σ cells COMPLETE (no pair alive); run7 MINTED (`configs/run7.yaml`, PUCT/PUCT); the preflight stamp, the shakedown and START are next

**R351 is landed verbatim** (`487d8661`), the operator's forward of the frontier-verdict packet
(dated 2026-09-14, forwarded 2026-09-13): the Gumbel deploy head that read every screen of run6
scored its 16 candidates by a min-max-rescaled Q × `c_scale` 1.0 — a 50–200-nat value term
against a few nats of prior, a pair neither the paper nor Mctx ran. R347(b) is ANNOTATED under
its foot; the packet's "F-46" is filed as **F-50** (F-46–F-49 were already taken; the resolution
is noted in the entry, as `R{next}` → R350 was). `dashboard-v2` merged ff at `3295cd8c`.

**The four σ cells (R351(b)) are COMPLETE — no pair is alive.** On the 18k net vs `sealbot_d5`,
288 paired games each, tree `646ca237` (record §F of
`docs/design/measurements/STRENGTH_FRONTIER_1_2026-09-13.md`; raw cells `/workspace/frontier/phase3/`
on the box and mirrored to the operator's `mantis-mirror/frontier/`, all three phases, 98 MB):

| σ pair | 128 | 512 |
|---|---|---|
| (rescale, 0.1) — Mctx's default | 0.167 [0.125, 0.208] | 0.056 [0.031, 0.083] |
| (no rescale, 1.0) — the paper's Go pair | 0.219 [0.174, 0.264] | 0.073 [0.045, 0.104] |
| (rescale, 1.0) — run6's mint | 0.233 | 0.066 |
| PUCT | 0.568 | 0.774 |

Both fall from 128 to 512 and both sit ~70 pp below PUCT-512's CI. By R351(c)'s rule run7's
self-play is PUCT (320/64 at p 0.25, full-arm policy rows, τ 0.5); deploy and eval PUCT-512.
**F-51 is filed:** σ's scale is NOT the Gumbel head's failure — all three pairs read the same net
within 7 pp at 128 and 2 pp at 512 and every one halves per doubling of sims; the class R351(e)
hoped would vanish with the scale does not. What is left is the head itself (m 16 over a ≈ 355-move
legal set, the halving schedule, the interior selector) or a defect the parity pins do not reach —
a card, not a run7 question.

**Landed on dev this leg (one line each, the full local gate set run at `5f05026f`, see the Exit
facts):** `646ca237` — the σ rescale switch: `QSigma { c_visit, c_scale, rescale }` through
`mantis-search`/`mantis-selfplay`/the bridge, the bridge's root calls reading the tree's ONE
configured σ (`MCTSTree.search_sigma` reads it back), `selfplay.q_rescale` REQUIRED (Mctx's
`rescale_values`; every config mints `true`, the arm it ran), per-cell `c_scale`/`q_rescale` in
`tools/strength_frontier.py`, contract v26. `5d014130` — **`search.kind` SPLIT** (R351(c)):
`selfplay.search.kind` and `deploy.search.kind`, two resolvers with one reader per key
(`resolve_selfplay_search_kind` feeds the pool, the HEXG capacity, the policy-target validator and
the resume guard; `resolve_deploy_search_kind` feeds the eval pipeline), the node-pool ceiling
checking eval sims under the deploy kind, `RETIRED_STAMP_SECTIONS` so a stamp carrying the retired
top-level `search` loads as provenance (run6's 36 bundles and the BC strip still load; the resume
guard reads a pre-split stamp's kind through the retired path), contract v27, repo_design amended.
`5f05026f` — the policy-loss trough is the dashboard's eighth health input (WARN, never a halt;
the prereg's {0.2 nats, 3 rows, step 5000} read off `trainer_step.policy_loss`), R351(d).

**run7 is MINTED: `configs/run7.yaml`** (34 header deltas, `config_diff --from-header` MATCH,
gates 7/12 green, `PRODUCTION_CONFIGS` carries it). Rows: PUCT self-play (320/64 at p 0.25,
`fast_policy_weight` 0.0, `temp_min` 0.5) and PUCT deploy/eval at 512/512; `reinit: []` on the
existing strip (net hash `2e72abd4…`, every tensor); `eval_interval` = `checkpoint_interval` 3000;
the sealbot point 288 paired games with `round_games` 288 AND `calibration_games` 288 /
`calibration_every_k_rounds` 1 (a sealbot-only rung that saturates would otherwise drop to 8-game
calibration and end the watch); `random_floor_games` 20; `gate.stride` 1; value warm-up 2000;
`policy_loss_trough_abort: null` (demoted — the manifest row stays DEFERRED);
`supervisor_kill_grace_sec` 120 as the relation; `seed` 20260914 (a dispatcher's value, the
packet's date). NOT minted: `monitor.drain` at its defaults — `config_diff --from-header` reads a
row minted at the template's own value as a lying header; the boot's `resolved_config.yaml`
records the four caps (900 × 3, 14 400, 14 400). The prereg's §5 carries every as-minted row.

**Dispatcher state for a fresh session.** The box `/workspace/hexo-mantis` is on branch `r351` at
`646ca237` (the σ switch; the split, the dashboard warning and the mint are NOT there yet — carry
`646ca237..dev` over by bundle before the preflight); CUDA torch was restored with
`make build.cuda` after a bare `uv sync` reverted it (do not run a bare `uv sync` on the box).
Open, in order: (1) bundle → box → `make build.cuda`; (2) the preflight stamp:
`/workspace/oc7/box_preflight.sh configs/run7.yaml /workspace/runs/run7-preflight <burst-steps>
<receipt-wait-sec>` with the operator's puller cycling against the out-dir
(`tools/mirror_pull.py --source <box-alias>:/workspace/runs/run7-preflight --mirror
<mirror-root>/run7-preflight --run-id run7 --interval-sec 60`); (3) the 4 h shakedown
twin (a config whose ONLY delta from `run7.yaml` is `run_id: shakedown`, verified by
`config_diff --expect run_id`; `/workspace/run_shakedown.sh <cfg> /workspace/runs/shakedown7
14400` — its end by `timeout` is rc 124 and CARD-SHAKEDOWN-TIMEOUT-STOP applies); (4) START on
the operator's word, `/workspace/oc7/box_start_run6.sh`'s shape with `run7`, the puller under
tmux/systemd on the operator's machine. Owed items (`CARDS.md`): the α = 1.0 three-row
reconstruction, `CARD-SELFPLAY-SEARCH-STATS`, `CARD-DRAIN-POLLER-RACE`; and a NEW card is
owed for F-51's residue (which part of the Gumbel head loses to the most-visited child at
every σ). run6's record stays under `/workspace/runs/run6/`, the R342-era logs under
`/workspace/r342/`.

## PERF-A4 — the serving levers, merged at `41a5fea8` (branch `perf-a4`, red-teamed)

Record: `docs/design/measurements/PERF_A4_2026-09-11.md`. Four levers landed: check 14 with the GIL
released (+17 %), pinned `non_blocking` H2D + `output_size=`, `inference.compile_trunk` (+17–21 %),
the two-thread software pipeline (+52 %). Combined at 32 workers: leaves/s 1,831 → 3,804
(+108 %). The block measured **1 163 steps/h whole** at 32 workers on the box
(`RUN6_BLOCK_2026-09-12.md`). The edge codebook is NOT landed (device-dependent numerics).

## Hold 1 — the stamp trap and run6's burst floor: DECIDED by matrix, landed at `652b9f02`

`compose_run(burst_stop_step=)` is the eighth census parameter; a production config stamps at
`sync_lag` from a ≈ 100-step burst (≈ 12 min on the box). `CARD-STAMP-FLOOR` carries the rejected
options. Phase W is NOT re-run before a start (`CARD-PHASE-W-AT-GUMBEL`).

## Hold 2 — the completed-Q target in decided positions: RULED by R350(e)

All 25 α = 1.0 rows of the START-path burst sat at `moves_remaining == 1` in LOST positions
(F-45); the block read 4.98 per 1 000, flat. R350(e) EXCLUDES them from the policy loss from this
tree on (`exclude_alpha_full_rows`, before the denominator; `trainer_step.policy_rows_excluded_alpha_full`).
The three-row reconstruction R349(c) ordered is still OWED.

## R349(b) — the mirror arm, landed and proven

`tools/mirror_pull.py` on the operator's machine, receipts beside each artifact on the box, the
preflight's rc 16 reading them. run6's mirror is `~/Work/HeXO/mantis-mirror/run6` (36 bundles,
153 shards, the final bundle receipted at the stop). The puller service is STOPPED; the run is.

## Integration tier: CUDA where a card exists (R349(a))

fp32 on `train.device: cpu` is the ONE carve-out to LAW-06 (`tests/train/test_law06_cpu_carveout.py`).
The dev box runs the CPU wheel; the tier's authority is the box.

## Minted values — `configs/run7.yaml` (and `configs/run6.yaml`, a finished run's record)

`run7.yaml`: 34 deltas from the `dev` template, replayable (`tools/config_diff.py --from-header`
MATCH). `selfplay.search.kind: puct`, `deploy.search.kind: puct`, `train.policy_target:
raw_visit_distribution`, `selfplay.c_scale` 1.0 / `q_rescale` true (template values, inert under
PUCT), `identity.warm_start: {checkpoints/bc/run6_00006500_ca1afb71.ckpt, 2e72abd4…, reinit: []}`,
`train.eval_interval` 3000 = `checkpoint_interval`, `eval.gate.deploy_sims` 512,
`eval.sealbot_model_sims` 512, the sealbot rung at `games_max` 288 with `round_games` 288 and
calibration 288 every round, `policy_loss_weight_schedule.warmup_steps` 2000,
`policy_loss_trough_abort: null`, `supervisor_kill_grace_sec` 120, `n_workers` 32, the PERF-A4
serving rows, `draw_rate_abort {0.25, 25000, 50, 3}`, `actor_lag_abort_enabled` true. Gate 12 is
GREEN with `policy_loss_trough` printed DEFERRED. `run6.yaml` re-minted through its own header
twice this leg (+`q_rescale: true`; `search.kind` → the two split keys, both `gumbel`) — the truth
of what it ran; its 32 original deltas replay.

## Protected set, laws, cards

Protected set as listed in `docs/governance/LAWS.md`; seventeen laws (LAW-06 amended by R349(a));
no gate number added (still 17). Cards (`docs/governance/CARDS.md`): opened by R350 —
`CARD-STOP-DRAIN-VS-GRACE` (fixed in code; the grace relation is a run7 mint row),
`CARD-WARMSTART-CONTROL` (closes on the frontier's `bc_tp`/`bc_full` pair — landed: 0.038 vs
0.413), `CARD-DRAIN-POLLER-RACE`, `CARD-SELFPLAY-SEARCH-STATS`; closed — `CARD-EVAL-CADENCE`
(R350(d)), `CARD-ALPHA-TARGET-FORM` (R350(e)); folded — `CARD-SEALBOT-HORIZON` into the frontier.
STRENGTH-FRONTIER-1 is RUNNING; INVESTIGATION-1's trough item is measured (record above); TEST-1
is not dispatched.

## Exit facts — the frontier-verdict packet (R351), 2026-09-13

- Ruling: R351 at `487d8661`, verbatim; the packet's `R{next}` → R351 and its "F-46" → F-50
  resolved in the entry; numbering continues at R352. R347(b) annotated under its foot.
- Commits on this line: `3295cd8c` (dashboard-v2 ff), `487d8661` (R351, F-50), `646ca237` (the σ
  switch), `5d014130` (the kind split), `5f05026f` (the trough warning), then the mint + records
  commit. One line each, empty body, zero trailers.
- The four σ cells: 18:10 → 19:20 UTC on the box, 4 cells at parallel 4, 1 152 games, 0 failed,
  11.9–14.6 s/game; verdict NO PAIR ALIVE; F-51 filed.
- The box: branch `r351` at `646ca237`; CUDA torch restored by `make build.cuda`.
- **Collected tests: 4 656 → 4 729** (the floor file follows; gate 3c property 2 holds against
  `origin/dev`'s 4 614). Comment ratchet: 3 545 / 23 / 13 527 → **3 535 / 23 / 13 511** (the
  dashboard merge had taken the tree below the floor without lowering it; lowered here).
- Contract: `docs/contracts/run_config_schema.md` v26–v27 (160 leaf key-paths, gate 13 green);
  `docs/design/repo_design.md` carries the split's amendment (R9).
- Gates on the committed tree (`tools/ci_gates/run_all.sh --with-slow`, dev box, CPU wheel):
  see the gate log line the mint commit's message cites; the default tier alone read
  **4 669 passed, 8 skipped** at `5d014130` and the config-reading tests 876 passed with
  `run7.yaml` declared.

## Provenance

Derived 2026-09-13 on `dev` at the exit commit of this leg, from `configs/run7.yaml`,
`configs/run6.yaml`, the box's `/workspace/frontier/phase3/summary.jsonl` (complete, 19:20 UTC)
and its mirror, `tools/ci_gates/test_count_floor.txt`, `tools/ci_gates/comment_length_floor.txt`
and `docs/governance/LAWS.md`. Ruling texts: `docs/governance/RULINGS.md`.
