# STATE — where mantis actually is

Rewritten in place, never appended to. Every value below was read from the tree at the commit
named under "Provenance", not copied from a register. Where a register disagreed with the tree,
the tree won and the disagreement is recorded in the last section.
Repaired in place 2026-09-17 (R311(c), REPAIR-A4 step 10, ledger C-1/C-2/C-5): the minted-values
block, the OPEN-card line, the exit block's box and push facts, dispatcher items (5) and (6); everything
else is the 2026-09-15 rewrite and reads as of that date.

## Current phase — **R365 (2026-09-21, item 17): run9 is NOT STARTED — a 288-game cell resolves ≥ 4 pp against a +0.7 pp/15k slope (E4), so its mint (`61bd2fd1`) and the budget-carry fix stand as RUN10'S BASE, un-armed; run8 STOPPED at 55 170 (08:23:44 UTC, `shutdown_save`), the box's last work RAN (PERF-3 step 3: no knee, the CPU launch stage owns every B; the E1 ruler-r6 cell: 0.080 [0.049, 0.115] against the r8 unit's 0.142 — "strix @ r8" is a unit qualifier) and the instance is RELEASED, the mirror the record; PROBE-1 is READ (`PROBE1_2026-09-21.md`, `tools/probe1.py`): no tail-only rows on any run8 ring and a 1.8× one-hot asymmetry mr 1 vs mr 2, the value head calibrated equally in both halves (E3 CLOSED), the value head's held-out gap 0.13 of 0.51 on all 18 nets beside a flat under-fit policy (REPORTED), proof-as-target DEAD (2.3 %, strict novelty 0/112, F-53), the soft-policy head ADMITTED (KL 6.6 ≫ 0.1), ARCH-D6 PARKED (the median spread falls 20 % while strix rises, the tail does not), the SWA net for the EMA cell BUILT for run10's preflight window; R366 composes run10 under R365(c).** Before it: R364 (item 16) had ARMED run9 on `dev` and granted run8's stop; R364 (2026-09-21, item 16): run8's STOP is GRANTED to the box session (it was still LIVE at ≈ 53k at this writing, 8 h past its last pre-registered read) and run9 is ARMED on `dev` — `configs/run9.yaml` minted over run8's header (parent run8@45k, the DATA REGIME: window 500 000 / `training_steps_per_game` 2.4 in the envelope [2.0, 3.0] / burst 8, the gate a REGRESSION GUARD at H0 0.42 / H1 0.52 with the new `at_max_pairs: promote` leaf, cadence 36 000 steps = 15 000 games, `dirichlet_enabled: false` with its pin), the budget's remainder now CARRIED (the envelope's mechanism, verified quantised at HEAD), the prereg FILLED, the parameter-distance test READ, CARD-ARCH-D6 opened unarmed; the box order (STOP → PERF-3 step 3 → preflight → twin → witness → START → follower `--cadence 36000`) is the box session's on the pushed tip.** Before it: run8 LIVE since 2026-09-18 16:55:34 UTC on the vested stamp (`c5d9e2fe…`, tree `86308bc7`; R358: σ + `train.augment: true` from the 42k parent), the shakedown witness PASSED and the ring audit PASSED all five bands with the two new rows READ; the strix follower chain runs beside it (parent solver ON → parent net_only → `--follow`), CONTENDED (dispatcher item 10 below); R359 landed the readings and the sourced queue (item 11); R360 re-aimed PERF-3 — steps 1–2 DONE on dev + mirror, step 3 is run9's preflight window — and made the twin's preflight inheritance code (item 12); **R361 (2026-09-19) withdrew the per-promotion strix trigger — the follower was relaunched `--no-promotions` at 06:15:06 UTC, run8 step 8 851 — and the EVAL CENSUS is READ (item 13): promotion selects nothing in the self-play loop, rounds cost the trainer 24–32 % while they run and strix cells 60–64 %, run7 spent 59 % of its wall in a round; **R362 (2026-09-19, item 14) RULED the rows: run8 runs to 30k regardless of the 15k reading, the sealbot rung is DELETED from the tree (contract v33), run9's eval rows are FIXED (cadence 15 000, gate 256/GSPRT unchanged, the gate's rule fields on the stream) and run9's ONE training swap and parent are ruled at 30k; the box's run8 stamp is untouched and still carries the retired rows, which the loader now tolerates**; **R363 (2026-09-20, item 15): the 30k point is INCONCLUSIVE (0.135 [0.097, 0.177] against 0.111) and run8 runs to 45k, the run9 PARENT RULE is PRE-STATED before that cell (the follower series monotone → 45k; else the highest whose CI holds the other two; the gate's best_model only without a strix triple), the LADDER unit is FIXED as code (`book_v1_s20260625_p4` paired, opening index = match index), ANALYZER-1 is admitted under R9, and the CPU deploy head is PROFILED (per-leaf forward ≈ 20 ms flat in batch size: 5.0 s per stone at 256 sims, 8 threads, idle); **the 45k point read 2026-09-21: 0.142 [0.104, 0.181], the series monotone, so the PARENT RULE names run8@45k — R364 is the operator's; the run continues (52 986 at 05:50 UTC), next cell 60k****. run7 STOPPED 2026-09-18 06:25 UTC at 83 482. The 2026-09-15 resume record follows as history

**The leg on the record:** the R353 packet landed in full (below), then the operator's 2026-09-15
questions — the run's checkup, WHY the eval is slow, sealbot's share, the book's ceiling — became
`RUN7_EVAL_COST_2026-09-15.md` and the independent review `docs/design/eval_gate_memo_2026-09-15.md`,
and the operator's decisions on both became the resume re-mint.

**Tree at the exit (`dev`; pushed at `ba51fd46` once the resume below was approved):** R353 (`c31fd74e`) · the sealbot TT
fix (`748f5c47`) · the dashboard's note (`136ab7b5`, discharged at `5bc28070`) · the census
(`416c95b4`) · the arm producer (`af47a8ab`) · CARD-SEALBOT-GIL-SERIAL closed (`f33ea87c`) · the A/B
record and CARD-SEALBOT-TT-SEAT closed (`992cd519`, `7e0a424c`) · `eval.max_plies` (`79d832cd`,
contract v30) · run6's truthful 256 and the memo (`eb0da661`) · **the GSPRT and the re-mint**
(`ce0a8ff6`, contract v31) · ratchet trims (`d3592315`) · `RUN7_EVAL_COST` (`bca5db17`) · this
STATE. Full local gate set on `ce0a8ff6`+trims: **19 GREEN + 3a green in the foreground** (4 798
passed; the background launch's `SIGINT=SIG_IGN` artefact is recorded once above and not re-derived);
gate 1 not run (the accepted cost).

**The run (read 15:10–18:30 UTC, `RUN7_EVAL_COST_2026-09-15.md`):**

1. START at 17:27:46 UTC 2026-09-14 on the vested stamp; supervisor pid 3094932; 32 workers; the
   puller (600 s) is a systemd user unit on the operator's machine; the dashboard and viewer are
   generated ON THE BOX from the live run dir every 10 min (`/workspace/obs/`, loopback 8766,
   reached by `ssh -N -L 8766:127.0.0.1:8766 vast` or the vast portal's cloudflared quick tunnel,
   whose hostname changes on every restart).
2. Rounds: r1 @3k 8 131 s promoted 0.733 · r2 @6k 12 816 s (under the A/B) not promoted 0.688 ·
   r3 @9k 12 483 s promoted 0.670 · **r4 @12k and r5 @18k KILLED at the 14 400 s bound, readings
   lost; the 15k and 21k kicks skipped as busy.** Cause: the gate block (208 games, PUCT-512 vs
   PUCT-512, games 55 → 95 plies, every screen escalating at 0.44) is 64–85 % of the wall and the
   per-ply cost beside the trainer is 2× idle; sealbot's own compute is ≈ 0 of it.
3. Trainer: 1 430 steps/h alone in hour 1–2, ≈ 900 since — game-length bound (self-play 45 → 63
   plies, one step per game), the eval child second-order. Losses: value 0.56 → 0.49, policy
   2.33 → 2.69 → 2.65 (the trough shape). Self-play decisive (ply-cap window 0.005 vs the halt's
   0.5), balanced, lengthening.
4. **Strength:** sealbot 0.733 → 0.688 → 0.670 (old adapter; the A/B read the levels as they
   stand); **strix cell A: BC net 0.083 [0.052, 0.118] → the promoted 9k anchor 0.097 [0.066, 0.132]
   → the 15k trainer net 0.045 [0.021, 0.073]** — the trainer peaked near 9k and regressed below
   its start by 15k; the gate held the 9k net (r4/r5 read 0.44–0.46 against it).
5. The book: 42–47 % of rung pairs are seat-decided → a paired-WR ceiling ≈ 0.78; the readings sit
   near the BOOK's ceiling (R353(e): a power loss; `book_v2` is the fix, carded). [R365(e), 2026-09-21:
   0.78 is the sealbot RUNG population's ceiling, not the book's — annotated at `RUN7_EVAL_COST` §B.]
6. **Will it stop at 25k? No.** `max_train_steps` 1 000 000; 25 000 is only where two armed aborts
   go live. Stopping is one SIGTERM to the supervisor (save-then-exit); the block is a minimum.

**The resume re-mint (`configs/run7.yaml` at `ce0a8ff6`, identity changed → a fresh stamp):**
`eval.max_plies` 256 (its own row; the operator's first ask of 128 was measured a weak lever and
run6/the frontier ran at 256 — `79d832cd`'s "128 = run6's cap" corrected in place), gate
`deploy_sims` 256 and `sealbot_model_sims` 256 (the rung series will step DOWN 6–13 pp at the
resume for the same strength; the frontier's two columns bridge it), `screen_confirm_lo` 0.5, and
`eval.gate.sequential` armed (GSPRT 0.52/0.62, α .05, β .10, 16 then every 8 to 104 pairs, the
LLR's sign at the maximum; `null` is the old rule, what run6 ran). Expected round ≈ 4 500 s
typical / ≈ 8 800 s worst against 12 500 / > 14 400; eval duty ≈ 40 % of wall. Concurrency 8/8,
cadence 3 000 and the book unchanged. The values CANNOT be live-patched (the parent's config is in
memory; the tree is stamp-bound). Resume mechanics verified on the real 18k checkpoint (launch
config wins outside the checkpoint-owned set; the pre-row stamp loads with
`checkpoint_config_predates_schema`).

**The resume, DONE (operator: "you may also push and start again", 2026-09-15):** `dev` pushed
at `ba51fd46`; run7 stopped by SIGTERM at 19:24:50 UTC with no round in flight (r5 had died at
17:23, the 24k kick was minutes away) — save-then-exit wrote `run7_00023829_4a302f88.ckpt` + bundle
+ sidecar + ring (100 000 positions, round_counter 5), child rc 0; the worktree and the run tree
moved to `ba51fd46` and rebuilt (`make build.cuda`, the extension now the arm-producer's); the
preflight ran from the worktree on the idle card and PASSED (`run7-preflight-resume`, mirrored):
its terminal round was 360 games in **1 229 s** — the GSPRT rejected the 101-step burst net against
its anchor at 24 pairs (LLR −3.14, WR 0.33), the rung at 256 read 0.441 [0.385, 0.500]; the run
resumed under the supervisor at 20:03:10 UTC (`box_resume_run7.sh`): stamp accepted on the new tree,
the stop checkpoint's stamp tolerated (`missing=['eval.max_plies', 'eval.gate.sequential']`),
`resume_state_restored step=23829`, four RNG streams, 32 workers, events in `seg0002`; both refresh
scripts follow the newest segment. The mirror puller runs on. CARD-EVAL-ROUND-OVERRUN closes when
the resumed run's first three rounds land inside the bound.

## The SEALBOT-TT A/B — read; every sealbot level on the record STANDS

`SEALBOT_TT_AB_2026-09-14.md`: four cells, 1 152 games, old tree vs fixed worktree. Concurrency 8
(run7's shape): BC net PUCT-150 0.469 → 0.462 (Δ −0.007 [−0.062, +0.049]), ck18k PUCT-512 0.764 →
0.792 (+0.028 [−0.021, +0.080]), run7 stamp net PUCT-512 0.618 → 0.604 (−0.014 [−0.062, +0.038]).
Serial (run6's rounds' and the frontier's shape): BC net PUCT-150 0.458 → 0.458 (+0.000 [−0.042,
+0.045]). Ratio 1.00; in no game did sealbot's move differ before the candidate's. The defect is
real in the engine and inert as a bias against our nets (their moves are not sealbot's, so the
next game never revisits the table's nodes); the fix stays for reproducibility. The dashboard's
note now states the finding (`SEALBOT_TT_NOTE`). CARD-SEALBOT-TT-SEAT CLOSED. Observation left
unruled: at concurrency 8 the CANDIDATE side is not run-to-run reproducible game-for-game (first
divergence at the candidate's move in 72–78 % of games, median ply 9–19); the level is stable.

## PERF-A4 — the serving levers, merged at `41a5fea8` (branch `perf-a4`, red-teamed)

Record: `docs/design/measurements/PERF_A4_2026-09-11.md`. Four levers landed; combined at 32
workers: leaves/s 1,831 → 3,804 (+108 %). The edge codebook is NOT landed.

## Hold 1 — the stamp trap and run6's burst floor: DECIDED by matrix, landed at `652b9f02`

`compose_run(burst_stop_step=)` is the eighth census parameter; a production config stamps at
`sync_lag` from a ≈ 100-step burst. `CARD-STAMP-FLOOR` carries the rejected options.

## Hold 2 — the completed-Q target in decided positions: RULED by R350(e)

α = 1.0 rows are EXCLUDED from the policy loss (`exclude_alpha_full_rows`); the three-row
reconstruction R349(c) ordered is still OWED. run7's burst emitted 19 `alpha_full_row` events in
101 steps; the run emits them at the same rate.

## R349(b) — the mirror arm, landed and proven

`tools/mirror_pull.py` on the operator's machine, receipts beside each artifact on the box. run7's
puller is a systemd user unit (above); run6's and shakedown7's mirrors stand beside it.

## Integration tier: CUDA where a card exists (R349(a))

fp32 on `train.device: cpu` is the ONE carve-out to LAW-06. The dev box runs the CPU wheel.

## Minted values — `configs/run7.yaml` (the running config, as re-minted 2026-09-15)

**41 deltas** from the `dev` template, replayable (derived: the `# delta:` header lines).
`selfplay.search.kind: gumbel` (320/64 at p 0.25, `c_scale` 1.0, `q_rescale` true),
`deploy.search.kind: puct`, `identity.warm_start` the LAW-12 strip with `reinit: []`,
`train.ply_cap_abort {0.5, 600, 3000}`, `train.eval_interval` 3000 = `checkpoint_interval`, the gate
at `deploy_sims` 256 and the sealbot rung at `sealbot_model_sims` 256 since the resume re-mint (512
before it — the two units are labelled on the dashboard; RULINGS annotations under R351/R352), the
rung 288 at `rung_concurrency` 8, `eval.max_plies` 256, the GSPRT armed (`eval.gate.sequential`),
`eval.round_timeout_sec` 14 400, `n_workers` 32, `seed` 20260914. `selfplay.search_stats_every` 8
(contract v32) is in the template and the file; the resumed run's stamp predates the row and the
loader tolerates it (`checkpoint_config_predates_schema`). Gate 12 rc 0 with FOUR rows DEFERRED
(`policy_loss_trough`, `ply_cap_attractor`, `grad_norm_hard_abort`, `sealbot_wr_abort`; the
`actor_lag` row was RETIRED by B-1, CARD-SERVER-OWNED-COPY). **Since R362(c), 2026-09-19:** the
committed `configs/run7.yaml` and `configs/run8.yaml` are RE-MINTED without the rung rows (35 and 37
deltas; the sealbot rung, its 288-game point and `rung_concurrency` are gone from the tree), while
the BOX's run8 stamp still carries them — `RETIRED_STAMP_PATHS` is what lets the tree load that
stamp; gate 12 now has THREE deferred rows.

## Protected set, laws, cards

Seventeen laws; no gate number added (still 17). Cards (`docs/governance/CARDS.md`): CLOSED this
leg — CARD-PUCT-ATTRACTOR (R353(e)), CARD-SEALBOT-GIL-SERIAL (round 1's wall), CARD-SEALBOT-TT-SEAT (the A/B's reading); OPEN —
every open card is in `CARDS.md` and is derived there, never enumerated here (this line once named
three of ~thirty). Opened by R355 since: CARD-SERVER-OWNED-COPY, CARD-STYLE-BACKLOG;
CARD-SELFPLAY-SEARCH-STATS landed 2026-09-16. Owed still: the α = 1.0 three-row reconstruction.

## Dispatcher state for a fresh session

run7 is LIVE; stopping it is ONE SIGTERM to the supervisor (save-then-exit). Open, in order:
(1) the resumed run's rounds — r6 @24k is the re-mint's first live round; read its wall and
the GSPRT's `pairs_played`/`stopped`, and expect the rung ≈ 6–13 pp below the 512 series;
(2) the strix cadence — 15k read 0.045 and the 9k anchor 0.097; the 2026-09-17 cells (`STRIX_RUN7_60K_2026-09-17.md`) read 24k 0.139 · 30k 0.170 · 42k 0.142 · 45k 0.094 · 48k 0.163 · 60k 0.111 — a plateau 6–9 pp above the start, two dips both instruments see; the next cadence point is 75k; (3) the
balanced opening book (`book_v2`); (4) CARD-GUMBEL-HEAD-RESIDUE; (5) OBSERVATORY — DECIDED
2026-09-17 under R355(f) by the operator: `tools/run_dashboard.py` + `tools/dashboard/` and
`tools/game_viewer.py` + `tools/viewer/` are the ONE implementation (they produce the box page
through the refresh scripts); the phase-1 reader layer (the observatory's `readers/` package, commits
`3a563574..4678537d`, 39 tests) is RETIRED from the tree with nothing consuming it. The design
(`docs/design/observatory_design.md`, `observatory_research.md`) stays as DASH-2's record; its
measured reader advantage (run6: 3.0 s / 84 MB against the dashboard's 5.0 s / 729 MB) is on the
DASH-2 card so phase 1 is revived from history, not rewritten, when the server is built. The
worktree `/workspace/mantis-tt` exists only for the A/B's fixed arm.
(6) REPAIR-A4 (R355) LANDED in three plans, `d44f3459..` up to and including the review-fix
commits of 2026-09-17 (the tip is `git log dev`; 43 commits at this writing):
A-2 + A-1 + the search-stats producer; the in-run repairs B-1..B-9, B-11, B-19, A-3, C-3; steps
10–12 — the doc repairs, the §3 deletes and archives (`docs/design/archive/`, `docs/audits/archive/`),
`mantis.diagnostics.ring_reader`/`.tactics`, gate 15's stale-header rule, two ratchet measures,
CARD-STYLE-BACKLOG; then the observatory decision of item (5). (7) R356 (2026-09-17): run8's
mint LANDED on `dev` (`configs/run8.yaml` = run7's resume mint + `selfplay.q_rescale: false` + the
warm start `run7_00042000_46fdb931.ckpt`, the gate's best_model; prereg
`docs/design/measurements/RUN8_PREREG_2026-09-17.md`); the 30k stop is SPENT (run7 at 72k, r22 in
flight at the mint). Run-ops owes, on the box under R356's grant: the stamp, run7's stop after it is
vested (one SIGTERM), the 3 h shakedown + the entropy witness, START, the follower's first cell
(the parent at 256/256), then the five R355(c) cells beside run8. (8) R357 (2026-09-18): the
witness is CORRECTED (A1 under R356's foot — 0.002 was the census MEDIAN, run7's full-arm mean reads
0.105–0.111): three-part, the QSigma pin on dev (`tests/selfplay/test_qsigma_rescale_reaches_target.py`
GREEN at HEAD: run8's config softens the Δq ≈ 0.01 row, run7's makes it one-hot, both red under a
planted stuck switch), then the shakedown ring's full-arm MEDIAN H(explicit) > 0.02 nats AND one-hot
share < 25 %; `mantis.diagnostics.ring_audit <ring> --bands <prereg>` is the standing pre-START gate
(exit 1 on a miss; run7's 23829 / 81000 rings read counter-threat 2.9 / 3.1 %, residue 0/275 and
0/184, full-arm one-hot share 65 / 62 % — three of the five run8 bands MISSED, the pre-fix baseline).
**run7 STOPPED 2026-09-18 06:25:23 UTC** by operator direction ("stop run7 if we don't need it any
more" — nothing on the record needed it: flat by its own gate since 42k, r13–r25 @48k–81k all
`promoted: false`, r22 @72k completed), BEFORE the stamp rather than after it (R356(b)'s grant, one
SIGTERM to supervisor pid 3127628, executed out of R357's ROUTE order on that direction): `shutdown_save
step=83482`, `run7_00083482_01e6df4b.ckpt` + bundle + resume + ring written, `terminal_eval_skipped
reason=signal_stop`, child rc 0, no round in flight (r25 @81k was the last, wall 4 211 s, rung 0.674;
r26 never fired); the puller receipted the 83482 bundle on cycle 292. The card is idle. Run-ops on
the box (fresh session, alias `vast`, R356 §5 7–13 + the four R357 deltas, the STOP step spent):
preflight → parent copy + sha → stamp → 3 h shakedown → witness (§3) + audit (§3a) on the same ring →
START → the follower's first cell. RESEARCH-STRENGTH runs in a web-enabled session beside it.
(9) R358 (2026-09-18, `docs/design/research/STRENGTH_RESEARCH_2026-09-18.md` read): run8 is
RE-MINTED in place (`configs/run8.yaml`, `f06c3234`) = run7's resume mint + `selfplay.q_rescale:
false` + **`train.augment: true`** + the 42k warm start, `config_diff --expect` MATCH on exactly
those leaves (warm_start as its two changed leaves), header MATCH, gate 12 rc 0, the count still 5;
the `2a2837b8` mint is superseded (never stamped — run7 stopped before it, so `run_id` stays run8).
Landed BEFORE the mint commit, as the arming condition: `iteration_complete.sym_draws` (12 D6 bins
+ empty-board skips off the ring's `draw_syms`, cumulative since boot; `tests/train/
test_augment_sym_counter.py` — augment true fills 12 bins with bin 0 ≤ 2× the mean, false puts
every draw in bin 0, a planted stuck RNG reds the band), `iteration_complete.samples_consumed_total`
+ `positions_produced_total` (one row), `mantis.diagnostics.ring_audit <ring> --events <jsonl>`
reading `replay_ratio` (Δsamples ÷ Δpositions over the ring's span, wall hours printed; run7's
derived 3.6 vs strix's 8.0) and `sym_bin0_over_mean` (1.0 uniform, 12 stuck) — both REPORTED, not
banded, the five run8 bands unchanged; the dashboard's throughput panel carries the draw line;
`crates/mantis-graph/tests/d6_lossless.rs` pins Appendix A (24 positions × 12 elements, ≈ 27 s in
debug). The NET-ONLY cell tooling: `tools/strix_driver.py` passes `disable_forcing_solver`
(default False = every reading on record; verified against the vendored strix, both variants load
and select), the adapter's `<stem>:net_only` variant (a distinct rung by name, regime key and bot
name `…_nosolver`), `tools/strix_follower.py --once <ckpt> --unit net_only` →
`<ckpt>.strix256_nosolver.json` (`strix.solver: off` in the receipt; the dashboard labels the
series "solver OFF"). Box order under R358 §6 (alias `vast`, the grant standing): preflight +
`make build.cuda` + parent copy/sha may run NOW; stamp only once the mint commit is on `dev` →
3 h shakedown, one contended round → witness (R357 §3(b), pin attached) → `ring_audit --bands
<prereg> --events <events_run8_seg0001.jsonl>` on the same ring, five bands + the two new rows READ
→ START → follower `--once <parent> ` (solver ON) → `--once <parent> --unit net_only` →
`--follow`; the cell's pre-stated reading is on CARD-STRIX-NET-ONLY; then PERF-3's packet.
(10) THE BOX BLOCK, DONE 2026-09-18 (alias `vast`, the run tree `/workspace/hexo-mantis` on branch `r358`,
carried by bundle — `dev` was NOT pushed at the time): `86308bc7` + `make build.cuda` (torch 2.11.0+cu128,
CUDA True, the rebuilt bridge serves `sym_draw_counts`); the parent copied to `checkpoints/run7/` and
sha-verified (`5750cca4…`); **run8 preflight PASS** 10:13–11:54 UTC (101-step burst, tier `sync_lag`,
(a) sync 51/51 (b) lag pass, `cuda_build` PASS, workspace MIRRORED; terminal round 520 games / 5 520 s on
the idle card, the burst net promoted over its anchor, rung 0.771 [0.722, 0.819] vs sealbot at 256 —
a preflight reading, not a series point); `shakedown8.yaml` minted from run8's header (`config_diff
--expect run_id` MATCH) and preflighted (PASS 11:58–13:18, stamp `fb17d30c…`); **shakedown8**
13:19:22–16:19:25 UTC (`run_shakedown.sh`, 3 h cap, rc 124): 1 004 steps/h, 1 026 games/h, 39 866
turns/h, mean game 38.8 turns, leaves 3 196/s, draw rate 1.7 %, ply-cap rate 2.7 % over the 600-game
window, card peak 11.67 GiB; step 3 000 at 16:18:41 — the gate round at 3k STARTED 44 s before the cap
and was CUT (its wall is not read); the 3k bundle + ring were written. **WITNESS PASS**: the QSigma pin
5/5 on the box tree; on the step-3000 ring's 24 970 full-arm rows median H(explicit) **0.270 nats**
(> 0.02; run7 0.0000), one-hot share **22.1 %** (< 25 %; run7 62–65 %), mean 0.458. **AUDIT PASS (5
bands)**: counter_threat_share 0.013 % (< 0.5 %; 3 of 23 245 block rows), quiescence_residue 0 of 6,
h_full_median 0.270, one_hot_share_full 0.221, cap_rate 2.2 % (< 5 %; 1 179 games); READ:
**replay_ratio 3.02** over the ring's span (1.13 h, positions 170 515 → 270 493, samples 467 712 →
769 536), **sym_bin0_over_mean 0.998** (bins 62 780–63 868, 9 479 empty-board skips; augmentation is ON
and uniform). **START 16:55:34 UTC**: supervisor pid 3182641 over `mantis.run` pid 3182650, out-dir
`/workspace/runs/run8`, `bc_warmstart_loaded` net hash `a9a46c55…` verified over 50 tensors, 32 workers;
the puller `mantis-puller-run8.service` (600 s) replaces run7's on the operator's machine. The follower
chain (`/workspace/oc7/chain_follower_run8.sh`, work `/workspace/oc8/strix_follow`): the parent at 256/256
solver ON, then `--unit net_only`, then `--follow`; its first launch was killed after 90 s because the
regime read IDLE beside the live trainer — a follower bug (heartbeat ages keyed by FILENAME; the stale
`run8-preflight/logs/heartbeat_run8.json` overwrote the live one), fixed at `5e25e40c` (keyed by
`<run>/logs/<file>`, test pinned), carried by bundle (tools-only over the stamped `86308bc7`; the run's
tree HEAD is `5e25e40c`, `src/` and the extension byte-identical) and relaunched 17:00:59 UTC reading
CONTENDED. run8's `dirichlet_*` rows are inert by code on the Gumbel arm (`search_drive.rs` applies
Dirichlet only under `SearchKind::Puct`; the deploy head has none) — no double noise; minting them
`false` is a run9 cosmetic. **THE TWO CELLS, READ** (CONTENDED beside run8, 288 paired games each, 0 fence
findings): the parent at 256/256 with strix's solver **ON 0.111 [0.073, 0.149]** (18:11:57 UTC, wall
4 256 s) and **OFF 0.115 [0.076, 0.153]** (19:46:41 UTC, wall 5 682 s) — **Δ +0.3 pp, inside both
CIs: the solver is not the gap** (CARD-STRIX-NET-ONLY's pre-stated rule; search-in-the-loop leaves the
run9 queue until new evidence). The 256/256 solver-ON point is run8's baseline for R356(c)'s reading
(the parent's as-shipped 512/128 point was 0.142). `--follow` is up (pid 3183447, 300 s polls): every
15 000-step save and every promotion fires one equal-work cell. run8 under the two cells ran at 369
steps/h / 394 games/h (step 1 057 at 19:47 UTC, 2 h 52 min in) — the CONTENDED price, ≈ 35 % of the
shakedown's rate; alone it should return to ≈ 1 000 steps/h. PERF-3 and R359 are the architect's next.
(11) R359 (2026-09-18, dev only; no box hours, run8 untouched): the readings LANDED — R359 verbatim
in the register (next R360), A1 under CARD-STRIX-NET-ONLY (the cell read the PLAY-TIME solver; the
training-side proof-target hypothesis is untested, not refuted), the prereg's §4 numeric line
(baseline 0.111 solver ON; SUCCESS 30k ≥ 0.161; FALSIFIED 15k ≤ 0.111 AND 30k ≤ 0.111; else read 45k)
and §3a's rule (only the pre-START audit halts; a 15k-ring miss is reported), CARD-RUN9-QUEUE sourced
from KataGo's methods with (vi) policy-surprise weighting and (vii) the auxiliary soft-policy head
and the order (i), (vii), (ii), (vi), (iv), (v), nothing armed; `dirichlet_*` inert on the Gumbel arm
written at the field and in the contract's residuals; the 4 h twin in the prereg. **PERF-3 (R359(f))
is ORDERED and STOPPED AT ITS OWN STEP 1 by the record, nothing built:** the §2 design (collate k+1
under forward k, double-buffered) IS PERF-A4's pipeline, `c888c3b7` on `dev` since 2026-09-11, +52 %
over serial at 32 workers (pre-reg +40–60 % met), and run7/run8 are minted on it — shakedown8's
3.19k leaves/s is that regime; the carried GPU 52.8 % / "never overlap" ratios are F-47's
pre-pipeline tree, and at HEAD the record reads GPU 85–95 % busy with the server thread's CPU stage
the bound (`PERF_A4_2026-09-11.md` §5). This workstation has no CUDA device and F-47's harness is not
in the tree, so step 1 cannot run here either. CARD-PERF-3 carries the numbers; the re-aim and the
register annotation under R359(f) landed as R360 (item 12).
Also OWED as code (CARDS.md): R359(e)'s twin-inherits-its-run's-preflight rule — the launch trap
keys the stamp by an identity that hashes `run_id`, so a twin is refused without its own stamp.
(12) R360 (2026-09-18, dev + mirror; no box hours, run8 untouched): the register at R360 (next
R361), A1 under R359's foot. **Twin inheritance is CODE** (`ee6e7abe`): `python -m mantis.run
--inherit-preflight <run>.yaml` launches a twin (the run's rows but `run_id`) on the run's VESTED
stamp — `require_preflight_stamp(inherit_from=)` diffs the two configs leaf by leaf, refuses any
difference beyond `run_id` by name (`PreflightStampTwinMismatchError`, pinned with a planted
`seed`), writes the twin's stamp with `inherited_from` + "preflight inherited from <sha>"; a stamp
of the twin's own wins; the parent's refusals propagate. **PERF-3 steps 1–2 DONE**
(`PERF3_2026-09-18.md`; CARD-PERF-3 carries the numbers): alone run8 serves 3 208 leaves/s at
B 46 with the CPU stage (11.46 of a 14.44-ms cycle) the bound and 85 % of pops waking at the
32-leaf threshold; beside a strix cell 1 222; shakedown8's 3 196 vs A4's 3 662 is one unit, two
regimes (88- vs 32-ply games). `tools/bench_server.py` is in the tree (the real server per B,
budget + probe witnesses); the workstation CPU ratio reads collate/graph flat-to-falling in B and
the supply cap B ≈ workers × leaf_batch / 2. **Step 3 is the box's, in run9's preflight window**
(≈ 19 min, the command in the record); the architect designs after it. Nothing in run9 armed.
(13) R361 (2026-09-19, dev + mirror + ONE tools-only box action; run8's stamp untouched): the register at
R361 (next R362), A2 under R356's foot (the per-promotion cell's cost; one annotation already stood
there), R356/R359/R360's Status lines amended. **THE BOX ACTION:** `tools/strix_follower.py --follow`
gained `--promotions/--no-promotions` (default on = R356(a), pinned + mutation-checked; `0d6ee0ee`),
cherry-picked onto the stamped lineage as the box branch `r361` (`08805676` = `5e25e40c` + that
commit, `src/`+`crates/` byte-identical to `86308bc7`) and carried by bundle; the old `--follow` (pid
3183447) stopped 06:14:58 UTC with no cell in flight, the new one (pid **3211513**, `--follow
--no-promotions --cadence 15000 --poll-sec 300`) started **06:15:06 UTC at run8 step 8 851**; the
launch script `/workspace/oc7/box_follower_run8.sh` carries the flag (its R358 copy kept as `.r358`).
The two promotion cells already fired stand as receipts: **run8@3000 0.056 [0.031, 0.083], run8@6000
0.0625 [0.038, 0.090]** (256/256, CONTENDED) — both below the parent's 0.111; the R356(c) reading is
15k/30k. **THE CENSUS** (`docs/design/measurements/EVAL_COST_2026-09-19.md`; CARD-EVAL-REDESIGN
carries the per-direction numbers): (i) self-play serves the LEARNER's weights every 2 steps
(`ActorSync` → `InferenceServer.load_state_dict_safe`); promotion writes `best_model.pt` + the
in-memory anchor, whose only consumers are the next round's `_best.pt` (verified by tensor hash:
r2's best == r1's candidate), the resume path and the next run's parent — the gate is an instrument;
(ii) run7 25 rounds = 50.0 h eval wall (gate 49 %, rung 34 %, two 4-h timeouts 16 %), gate game
38 s at 93 plies, rung game 9.3 s at 63; GSPRT rejects 36 pairs mean; run8 2 rounds 4.35 h, both
accepted (88, 48 pairs), gate game 40 s at 91 plies; the sealbot rung 0.657 ± 0.045 flat over run7;
(iii) the GSPRT's long end is the (μ0+μ1)/2 = 0.57 midpoint — H0 0.50 buys −7 % pairs for 2 → 5 %
false promotion of an equal candidate; (iv) rounds in flight 59 % of run7's wall, 33 % of run8's
plus 39 % strix cells (72 % together); run8's trainer 999–1 069 steps/h alone, 682–724 in a round,
356–415 in a cell. ETA (ESTIMATE): 15k ≈ 13:30–14:00 UTC 09-19, 30k ≈ 08:30–09:30 UTC 09-20. Gate
sims 64 UNMEASURED. Two cards opened (CARD-EVAL-REDESIGN, CARD-EVAL-GATE-FIELDS-IN-STREAM); CARD-PERF-3
carries R361(d)'s contended arm. No config touched; nothing in run9 armed.
(14) R362 (2026-09-19, dev + the two READ-ONLY box checks §0(4) owed; no box hours, run8's stamp
untouched): **THE BOX, read 10:02:49 UTC after two banner drops** — the follower pid 3211513 is
ALIVE (etime 3:47:43, `--follow --no-promotions --cadence 15000 --poll-sec 300`, HEAD `08805676`),
its chain log ends at the R361(a) relaunch line with no cell fired since (15k not reached); the
puller mirror is FRESH (`events_run8_seg0001.jsonl` at 09:42 UTC, run8 at **step 11 842**; r3 @9000
NOT promoted, sealbot 0.693, wall 12 152 s; the 9000 resume bundle + ring receipted 07:07 UTC).
Nothing relaunched. The register at R362 (next R363), A1 under R361's foot (the H0 clause: the zero-drift point is the 0.57
midpoint), R361's Status line amended. **THE DELETION, one commit:** the sealbot rung is gone from
the tree — `eval.ladder`, `eval.sealbot_model_sims`, `eval.rung_concurrency`, the nine `monitor.wr_*`
leaves (contract v33; every committed config RE-MINTED through its own header minus those rows,
byte-identical elsewhere; the `dev` template stripped), `mantis.eval.{ladder,bt,channel_health}`,
the coordinator's `on_eval_round_complete` WR gate (a completed round routes straight to promotion
through `drain._route_eval_result`), the `sealbot_wr_abort` manifest row with the `EVAL_ROUND`
clock and `Cadence.EVAL_ROUND_CONSEC`, the `sealbot_wr_warn` producer row, the A/B/C predicates,
`eval_channel_health`, `wr_sealbot*` on `eval_round_complete`, the rung-skip channels, the resume
sidecar's `last_p_hat` (a pre-R362 sidecar is read with it ignored, version unbumped). The rung
machinery (`RoundSpec.rung_jobs`, `RungJob` now carrying its own bootstrap terms,
`worker._play_rung_block`) survives for the strix cell alone; the frontier tool's sealbot cell and
its default opponent are gone (a cell names `strix` or a snapshot); the sealbot ADAPTER stays as a
vendored opponent with no production caller. `RETIRED_STAMP_PATHS` widens the loader's tolerance to
these nested paths (pinned) so run9 boots from a run8 stamp. The dashboard draws the rung as a stated
gap on a record with no reading. Gate 12: THREE rows DEFERRED now (`policy_loss_trough`,
`ply_cap_attractor`, `grad_norm_hard_abort`). **THE STREAM GAINS** `eval_round_complete.gate` =
`{rule, pairs_played, stopped, llr, wr_confirm, n_pooled, promoted, wall_sec}` (`GATE_STREAM_FIELDS`;
`wall_sec` the gate block's own, measured in the child), `null` when no gate ran, on the success and
the A-3 partial routes, producer test + planted break; CARD-EVAL-GATE-FIELDS-IN-STREAM CLOSED,
CARD-EVAL-REDESIGN moved (gate sims 64 as a measured cell is what stays open on it). **RUN9:**
`docs/design/measurements/RUN9_PREREG_2026-09-19.md` is a SKELETON — the eval rows fixed
(`--set train.eval_interval=15000`; the P6b pin reads `monitor.gate_interval == train.log_interval`
and is unmoved, its count line is widened BY NAME when run9's config lands), the parent, the seed and
the ONE training swap BLANK BY DESIGN, "ruled at 30k"; run9's config file is NOT minted (a
placeholder warm start is a hand-varied identity row, R1/LAW-11). Repo_design carries the R362(c)
amendment; `eval_instrument.md`, `event_manifest.md`, `checkpoint_envelope.md` and
`run_config_schema.md` (v33, 150 leaves, the twelve section rows re-derived) are repaired in the
same commit. The collected-test count is 4 863 against the 4 862 floor (no ratchet-down record
needed); the R346(f) comment floors follow the deletion down (3 405 / 13 105 / 1 506). **The full
local gate set on `ff5a43d8` (`run_all.sh --with-slow --base 671bf12c`, 11:43 UTC): 18 GREEN —
2a cargo test (2 001 s), 2b, 4, 5, 3b integration (48 passed, 3 848 s), the slow tier (5 passed),
3c, 7, 8, 9, 11, 12, 13, 14 (pyright 230 files, 0 errors), 15, 16, 6, 17; gate 1 not run (the
accepted cost). Two reds with ONE cause, this file's own uncommitted draft naming run9's unminted
config as a path: gate 10 (rc 1 on that line) and 3a's single failure, the gate-10 vacuity test
(4 799 passed otherwise); the line is reworded and both re-read GREEN before this commit.**
**THE 15k POINT, read off the mirror 16:19 UTC 2026-09-19 (recorded, NOT acted on — R362(b)):** the
cadence cell `run8_00015000_5777cb58.ckpt.strix256.json` (equal-work 256/256, solver ON, CONTENDED,
13:00:07 → 14:34 UTC, 5 639 s, 19.6 s/game, median 43.5 plies) reads **0.104 [0.069, 0.139]**, 288 games,
30 W / 258 L / 0 D — at the parent's 0.111 [0.073, 0.149], so the FALSIFIED line's 15k half
(≤ 0.111) is MET by 0.7 pp inside both CIs; the 30k half decides. The 15k gate round r5 PROMOTED
(GSPRT accept at 80 pairs, LLR +4.55, pooled 0.641 over 160 games, wall 9 553 s — gate 7 309 s, rung
2 206 s); sealbot 0.799 [0.753, 0.844]; random floor 20/20; r3 @9000 and r4 @12000 NOT promoted
(0.693, 0.778). The 15k RING AUDIT (`ring_audit … --bands RUN8_PREREG`, rc 1): counter-threat 0.031 %
PASS, residue 0/8 PASS, `h_full_median` 0.209 PASS (mean 0.465), cap rate 0.88 % PASS,
**`one_hot_share_full` 0.2745 MISS (band < 0.25; the shakedown read 22.1 %)** — a live-run miss is
REPORTED and stops nothing (R359(b)); replay_ratio 3.10, sym bin0/mean 1.000.
**THE 30k POINT, read off the mirror 06:38 UTC 2026-09-20 — the R356(c) line reads INCONCLUSIVE:** the
cadence cell `run8_00030000_6e45edb0.ckpt.strix256.json` (equal-work 256/256, solver ON, CONTENDED,
05:04:06 → 06:29:07 UTC, 5 100 s, 17.7 s/game, median 39 plies, 0 fence findings) reads
**0.135 [0.097, 0.177]**, 288 games, 39 W / 249 L / 0 D. Against the parent's 0.111 [0.073, 0.149]:
NOT SUCCESS (< 0.161), NOT FALSIFIED (> 0.111 — the 15k half was met, the 30k half is not) →
**INCONCLUSIVE, whose pre-stated action is READ 45k under the three-cell rule** (R356(c), R359(b));
the run9 parent and training-swap ruling R362(e) hangs at 30k therefore waits on 45k unless the
operator rules otherwise — an OPEN decision, not this session's. The series: 3k 0.056 · 6k 0.0625 ·
15k 0.104 · 30k 0.135 (+2.4 pp over the parent, inside both CIs). The rounds since 15k ALL REJECTED
by the GSPRT against the 15k anchor — r6 @18k 32 pairs 0.422, r7 @21k 40 pairs 0.469, r8 @24k 24 pairs
0.458, r9 @27k 80 pairs 0.537, r10 @30k 32 pairs 0.391 (A-3 partial; its rung in flight at 06:38) —
while the sealbot rung rose 0.764 → 0.752 → 0.806 → 0.830 (the census's finding, live). The 30k
RING AUDIT (rc 1): counter-threat 0.009 % PASS, residue 0/4 PASS, `h_full_median` 0.200 PASS (mean
0.463), cap rate 0.22 % PASS, **`one_hot_share_full` 0.2761 MISS** (band < 0.25; 15k read 0.2745,
the shakedown 22.1 %) — reported, stops nothing (R359(b)); replay_ratio 3.61 (span 1.92 h), sym
bin0/mean 0.999. Trainer at 30 699 (06:37 UTC), 943 steps/h over 12 h; the follower's next cadence
cell is 45k; ETA 45k ≈ 21:40–22:40 UTC 2026-09-20, its cell's reading ≈ 1.4 h after.
(15) R363 (2026-09-20, dev only; the box UNTOUCHED, run8 runs to 45k — the mirror read run8 at step 30 871 at
06:49 UTC): the register at R363 (next R364), R362's Status line amended ((c)'s roles line and "GSPRT bounds
unchanged" QUALIFIED, (e)'s "ruled at 30k" MOVED to 45k); the repo_design ANALYZER-1 amendment carries R363
where it said `R<nnn>` (the design doc's quoted copy too). **THE 45k PARENT RULE IS PRE-STATED** in
`RUN9_PREREG_2026-09-19.md` §1c, verbatim from R363(b), with the two points already on the record (15k 0.104,
30k 0.135, the parent 0.111) — the 45k cell reads against it; the run8 prereg's §4 carries the 30k reading as
INCONCLUSIVE. **THE LADDER UNIT IS CODE** (`tools/ladder/openings.py`, receipt schema v2): `book_v1_s20260625_p4`
in FILE order, pair `m` (the m-th pair of `gameStart`s against one opponent in the session, both bots
counting the same events, `--match-offset` for a resumed series) plays opening `m` translated onto the
server's auto-placed origin — the second player's first compound turn is the book's plies 2–3 unsearched,
the first player's is ply 4 plus ONE searched stone, a board off the prefix is searched and the receipt says
where (`opening.off_book_at`); `book_stones` rides every move, `--replay` re-derives the forced stones from
(opening, position) and names a disagreement (`book_misses`), the below-budget line is per searched stone;
`--preset play` is the 64-sim row R363 §0(5) allows, on `search.preset` of every receipt, never a unit
reading (in the profile below its argmax agreed with 256's in 58 of 101 positions — a different player, not
a faster one). CARD-LADDER-RUNG carries the unit and the admission test (one 288-game IDLE cell beside the
follower's parent cell); CARD-LADDER-SERVER-ASKS records the six server items as the operator's (the four
spec-file lags, the racy `moves[]`, house bots refusing bot challenges — the opening-variety finding is not
among them, R363(c) having answered it on our side). **THE CPU-HEAD PROFILE, READ** (`CPU_HEAD_PROFILE_2026-09-20.md`;
the ladder's `MantisBackend` on run8@30k, `configs/run8.yaml`'s deploy seam — PUCT 256, `leaf_batch_size` 8,
the batcher at 64 / 10 ms, eager — 101 stones over four book openings, plies 4–30, this workstation IDLE,
8 torch threads): **5.0 s per stone median (10.0 s per compound turn)** and 1.23 s at 64 sims; the engine-
boundary call is 99.3 % of the wall; the leaf-batch histogram is 8 leaves in 61 % of calls (81 % of leaves),
1 leaf in 17 % (2.8 % of leaves; 728 cold-tree/root calls cost 16 s of 481); **the per-leaf forward is FLAT
in batch size — 22.5 ms at 1 leaf, 19.8 ms per leaf at 8** (16.9 → 20.5 ms/leaf from 4–9 to 20–29 stones
as the graphs grow: ≈ 470 nodes and ≈ 12 k edges per leaf under `gnn_axis_r8`) and **NOT moved by torch
threads** (opening 0's 26 stones: 8 threads 4 551 ms per stone, 4 threads +16 %, 16 threads +20 %), so the
head is bound by the forward itself, not by batching, fill or core count; the batcher's 10 ms deadline
costs 7.98 ms per pop = 7.2 % of the wall (one submitter never reaches the 32-leaf wake), collate 0.32 ms
per pop. Numbers, not a design: nothing is ordered on them.
**THE 45k POINT, read off the mirror 05:50 UTC 2026-09-21 — the PARENT RULE names 45k:** the cadence cell
`run8_00045000_3bdedf76.ckpt.strix256.json` (equal-work 256/256, solver ON, CONTENDED, 20:29:07 → 22:05:24 UTC
09-20, 5 776 s, 20.1 s/game, median 43 plies, 0 fence findings) reads **0.142 [0.104, 0.181]**, 288 games, 41 W /
247 L / 0 D. The series 15k **0.104** → 30k **0.135** → 45k **0.142** is monotone non-decreasing on point estimates,
so under R363(b) as pre-stated in `RUN9_PREREG_2026-09-19.md` §1c **the run9 parent is run8@45k** (checkpoint sha256
`c4990d03…`, net `3aef7883…`) — the rule's output, recorded here; the ruling (R364: the parent, the ONE training
swap, run9's gate H1 from the measured per-15k gain: +3.1 pp then +0.7 pp) is the operator's. Against run7's 42k
parent (0.111 [0.073, 0.149]) the 45k point is +3.1 pp with the CIs overlapping; R356(c)'s SUCCESS line (≥ 0.161)
is not reached by any point. The 45k gate round r15 read 0.500 over 40 pairs against the 15k anchor and REJECTED;
r11–r17 (33k–51k) all rejected (the anchor is frozen at 15k, R363(b)); r16 @48k ran to the GSPRT's 104-pair cap
undecided at 0.495 (208 games, wall 10 507 s). The sealbot rung on the box's pre-R362 tree: 0.809 · 0.826 · 0.781 ·
0.792 · 0.833 · 0.814 · 0.774 (r11–r17). The 45k RING AUDIT (rc 1): counter-threat 0.009 % PASS, residue 0/6 PASS,
`h_full_median` 0.286 PASS (mean 0.594; 15k 0.209, 30k 0.200), cap rate 0.16 % PASS, **`one_hot_share_full` 0.2705
MISS** (15k 0.2745, 30k 0.2761, 51k 0.2754 — flat at ≈ 27 % against the < 0.25 band; reported, stops nothing);
replay_ratio 3.19, sym bin0/mean 1.000. **The run was NOT stopped at 45k** (nothing on the record ordered it): trainer
at **52 986** (05:50 UTC), 871 steps/h averaged over the 60.8 h since START (alone ≈ 1 000–1 070, in a round
680–770), lr 9.97e-4, loss 2.62 (policy 2.12, value 0.50) at 51k against 3.19 / 2.54 / 0.64 at step 1, no abort,
one narration alert (`loss_increase_window` at 50 000); self-play 53 084 games, mean 79 plies = 40.6 compound turns,
draws 0.5 %, ply-cap 0.53 %, served 3 193 leaves/s, batch fill 75 %; the 17 rounds' walls sum to 31.0 h = 51 % of the
run's wall in flight (R361's census, live). The follower's next cadence cell is 60k: ETA 60k ≈ 13:15–14:00 UTC
2026-09-21, its cell ≈ 1.6 h after; the puller mirror is fresh (05:50 UTC), the box itself was NOT contacted.

(16) R364 (2026-09-21, dev + mirror; the box UNTOUCHED by this session — run8 read LIVE at 52 986 at 05:50 UTC
and its stop is the box session's first act): the register at R364 (next R365), A1 under R363's foot ("read 45k"
lacked "then stop"; the architect's; 8 h), R362's and R363's Status lines amended. **THE PARAMETER-DISTANCE TEST,
READ** (`PARAM_DISTANCE_2026-09-21.md`; §0(7)): eighteen checkpoints off the mirror (the run7@42k parent as step 0,
then 3k … 51k), 302 470 parameters in float64 — the 3k-step displacement ‖Δ‖ FLAT at 18.7–24.3 (the LR reads 9.97e-4
at 51k, effectively flat), successive displacements nearly orthogonal (cos 0.13–0.20 from 6k to 30k, falling to
0.03 by 51k), the cumulative path k^0.71 (between a random walk's 86 and a straight line's 353 at k = 17: 142),
the distance to the running mean growing monotonically 10 → 78 (a drifting centre, not a noise ball, its growth
slowing 6.3 → 1.3 per segment late), the weight norm +49 % (156.6 → 233.8; `representation` +19 %, `policy_head`
+62 %, **`value_head` +96 %**) — beside strix +4.8 pp (3k → 15k) then +3.1 then +0.7 and the gate's r6–r17 band
0.31–0.54. What is ruled out: a small step, a settled centre. What is left open, the operator's for run10's LR:
noise-spending LR vs the 100k-window's disjoint-data signature (which run9's window changes first). **THE MINT:**
`configs/run9.yaml` from run8's header through `mint_config.py`, `config_diff --expect` MATCH on exactly eleven
leaves (`run_id`, the warm start's two, `train.replay_capacity` 500 000, `train.training_steps_per_game` 2.4,
`train.max_train_burst` 8, `train.eval_interval` 36 000, `eval.gate.sequential.mu0` 0.42 / `mu1` 0.52 /
`at_max_pairs` promote, `selfplay.mcts.dirichlet_enabled` false), `--from-header` MATCH; the seed byte-equal.
The parent's file sha256 `c4990d03…` (mirror copy), net `3aef7883…` read through the LAW-12 loader. **Why 2.4 and
not 2.5:** run8's 45k ring reads replay_ratio 3.19 at 1.0 (256 samples per 80.3 positions per game), so 2.4
predicts 7.7 and stays inside [7, 9] over run8's whole observed base-rate spread (3.02–3.61 → 7.2–8.7) while 2.5
reads 7.6–9.0 and its 37 500-step cadence is not a checkpoint step; 2.4 × 15 000 = 36 000 exactly. The
dispatcher's pick inside the envelope; the shakedown's `replay_ratio` says whether it stands (prereg §1b: outside
[7, 9] → re-pick inside the envelope with its own preflight; unreachable → HALT). **THE KEY'S SEMANTICS AT HEAD,
VERIFIED (R364 §0(3)):** `_steps_budget` was `min(max(1, round(games × ratio)), burst)` per burst with no remainder
carried; run8's stream reads ONE new game on 98.0 % of its 52 141 bursts (two on 1.9 %, three on 0.05 %), so 2.5
would have realised `round(2.5)` = 2 steps per game (≈ 6.4 reuse) and 2.6 three (≈ 9.6) — the envelope's target
sat in a hole. `mantis.train.mixing._steps_budget` now carries the fraction (`floor(carry + games × ratio)`, the
ceiling drops its excess, integer ratios unchanged so run8's 1.0 realises as before; in-memory, a resume restarts
it at 0), pinned by `tests/train/test_steps_budget_carry.py` (+5). A correctness fix, not a second swap; the
operator can veto it before the stamp (one commit). **THE GATE RULE (R364(c)) IS A LEAF:** `eval.gate.sequential.
at_max_pairs` (`sign` | `promote`, contract v34, leaf count 151) — `gsprt_decision(..., at_max_pairs=)` promotes an
undecided candidate at the 104-pair cap under `promote`, the bounds still decide before it, a reject is still a
reject (`tests/eval/test_gate_sequential.py` +2 rules, the spec refusal widened); `run7.yaml` and `run8.yaml`
RE-MINTED through their own headers with `sign` (byte-identical elsewhere; the box's run8 stamp predates the leaf
and the loader tolerates it). Under 0.52/0.62 the two run8 cap rounds (r3 0.536, r16 0.524) were rejected by the
sign; under run9's rule they promote. **R359(d)'s PIN LANDED:** `dirichlet_root_fires` on `RunnerStatsSnapshot`
(counted at the PUCT arm's mix-in site; not a bridge field), `crates/mantis-selfplay/tests/dirichlet_inert_on_gumbel.rs`
drives one worker with the rows ARMED — Gumbel 0 fires over a fully served 64-sim search, PUCT 2 — the second
half proves the counter sees the lever. **THE PINS BY NAME:** `PRODUCTION_CONFIGS` + the gate-12 process test,
the P6b count 5 → 6, the fused-caps / draw-rate / strength-floor / arch-kind / microbatch production sets, the
undeclared-plant stem `run9` → `run10`. Gate 12 rc 0 (three rows deferred), gate 7 every config OK, gate 13 rc 0
at v34. **THE PREREG** (`RUN9_PREREG_2026-09-19.md`, filled in place): the eleven rows with their why, the
envelope and the dispatcher's rule, the reading at EQUAL GAMES (SUCCESS 30k-games ≥ 0.192; FALSIFIED 15k ≤ 0.142
AND 30k ≤ 0.142; else read 45k-games THEN STOP; steps 36 000 / 72 000 / 108 000), the STOP LAW, box-hours beside
every point, the run9 ring bands with **`one_hot_share_full` RE-STATED at < 0.30** (the parent's own rings read
0.2705–0.2761 flat — at 0.25 the twin of a 27 % parent HALTs by construction; run7's 62–65 % collapse is still
refused twofold) and **`replay_ratio ≥ 7.0` banded** (the ≤ 9 half read; the tool bands one operator per row) —
the block PARSES and reads run8's 45k ring as expected (five PASS, `replay_ratio` 3.06 MISS, rc 1); the 4 h twin
runs NO gate round (36 000 steps is 15 h away) and the regression-guard rule is exercised by the PREFLIGHT's
terminal round. run8's prereg §4 carries the 45k reading and the stop. **CARDS:** CARD-ARCH-D6 OPENED (not armed;
witness = the analyzer's symmetry spread per checkpoint, detector + conformance before any training),
CARD-RUN9-STOP-LAW opened, CARD-RUN9-QUEUE (i) rides run9 / (ii) waits / (iv) queued, CARD-EVAL-REDESIGN moved
(the regression guard), CARD-PERF-3 step 3's window fixed. **THE BOX ORDER** (prereg §5, the box session's, on
the pushed tip): run8 STOP (one SIGTERM; the follower stopped with no cell in flight) → PERF-3 step 3 (≈ 40 min,
B alone then the contended arm; numbers before design) → `make build.cuda` + the parent copied and sha-verified
into `checkpoints/run8/` → run9 preflight → the stamp → the 4 h twin `--inherit-preflight` → the witness (QSigma
pin, the six bands, `replay_ratio` 7–9 reported with the value that stands, sym bins, entropy) → START →
`strix_follower.py --follow --no-promotions --cadence 36000`. Box-hours ESTIMATE before the shakedown: ≈ 16 h per
15 000 games if games/h holds (2.4× the trainer's steps per game; the shakedown's games/h replaces it).

(17) R365 (2026-09-21, dev + mirror + the box's LAST session — run8 STOPPED and the instance RELEASED): the
register at R365 (next R366), A2 under R364's foot (run9 not started, the mint = run10's base), A1 under R362's
foot (the "matches run7's dip" join), R362's and R364's Status lines amended; F-44 annotated by regime (run6's
92 % is unread at run8's mint and at 2.4 steps/game; its three forward uses pointed), the "≈ 0.78 book ceiling"
annotated by population at `RUN7_EVAL_COST` §B (the rung's 42–47 %; strix cells read 0.86–0.92) with its uses at
`STRIX_RUN7_60K` and the 2026-09-15 block pointed; `PARAM_DISTANCE` re-pinned (sha256 `60ca27c4…`, `203e7060`)
in the register and in the research packet, which is now ON `dev` (`75f3627f`, one-file cherry-pick from
`research-strength-2`); `RUN9_PREREG` carries the not-started header, E4's resolution statement in its unit
section and the `one_hot_share_full` band STRUCK (five bands parse; run8's 45k ring reads them as before).
**THE BOX, in order (branch `r365` = the `dev` tip `d4804e58` by bundle, `make build.cuda`, CUDA torch
2.11+cu128):** run8's follower (pid 3211513, no cell in flight) SIGTERM 08:23:40 → run8's supervisor SIGTERM
08:23:43 → `shutdown_save step=55170`, `terminal_eval_skipped reason=signal_stop`, `run8_00055170_34ff6c4e.ckpt`
+ bundle + resume + ring written and RECEIPTED by the puller (round r18 @54k was in flight and abandoned with
the run; the record closed at 45k by R365(a)) → **PERF-3 step 3** (`PERF3_2026-09-18.md` §step 3): alone at
200-s cells 2 236 / 2 394 / **2 466** / 2 055 / 2 091 leaves/s at B 16 … 256, the CPU launch stage 0.34–0.39
ms/leaf at EVERY B, no knee, D-1 dead by its own falsifier (1.10×), the probe exact across B; contended beside
the E1 cell 593 / 984 / 1 326 / 1 101 / 1 039 (27–54 %, the CPU stage doubled per leaf); the "128 sims" arm
NOT MEASURED (no producer in the tool; P-B3's twin in run10's window) → **E1** (`tools/strix_follower.py --once
… --unit ruler_r6`, the variant `<stem>:r6` landed at `d4804e58`): **0.080 [0.049, 0.115]**, 23–265–0, median
41 plies, 2 804 s, IDLE, 6 690 fence findings and 0 out-of-fence — against the r8 unit's 0.142 [0.104, 0.181] a
−6.2 pp move with disjoint CIs, so "strix @ r8" IS a unit qualifier on every follower point (the series and the
parent choice stand, every point shares the unit; strix at its trained radius is the stronger ruler) → the mirror
verified (rsync dry-run: run8 and run7 complete; the cell dirs, the bench JSONs, the scripts and every other
run dir copied to `mantis-mirror/box-final-2026-09-21/`) → `vastai destroy instance 35883053` at 10:17 UTC, the
host unreachable a minute later; the local puller unit stopped. Box-hours since
the stop: 1.11 measured (0.33 bench alone + 0.78 E1 with the contended bench inside) of the packet's 1.6.
**PROBE-1 READ** (`docs/design/measurements/PROBE1_2026-09-21.md`; the instrument `tools/probe1.py` +
`tools/probe1/` on the production sample + collate, the cadence search, the bridged solver and the analyzer's
sweep, pinned by `tests/tools/test_probe1.py`; `ring_audit.reconstruct_moves` exposes the ONE cadence
sequence): (1) DECOMPOSER — ZERO tail-only rows on all 19 run8 rings (the conflation P5 posited is run7's,
not run8's), the pooled share 25.7–28.4 % flat since 15k, **mr 1 rows one-hot 30–37 % vs mr 2 15–20 %** on
every ring; (2) E3 — MAE ratio mr1/mr2 1.002 [0.983, 1.019], the reliability curves on each other bin for bin:
CLOSED, one scalar suffices; (3) C3-1 — policy gap 0.09 ± 0.035 on a flat 2.1-nat loss, VALUE gap 0.13 (0.05–0.19,
positive on all 18 nets) of 0.51, total 0.22: BETWEEN, reported, the value gap on the next ring named as run10's
in-run reuse witness; (4) P-B2 — proof 2.30 % [1.88, 2.72] of 4 877 roots, 112/112 z-agreement, 4.5 ms/root,
budget never binding, **strict novelty 0 of 112**: DEAD, F-53 filed; (5) P-B1 — median KL(prior‖target) 6.58
(4.49 on supported rows; KL(target‖prior) 1.21; KL(target‖target^¼) 0.76) over 5 081 full-arm rows: ADMITTED
(queue (vii)) under R365(c); (6) SPREAD — the median 0.122 → 0.097 first-six to last-six, −0.008 per 10k, the
translation exact, four positions at 0.4–0.5 on every checkpoint: ARCH-D6 STAYS PARKED; (7) the SWA net
`run8swa_00051000_ae478dee.ckpt` (net `61b09ba8…`, 8 sources 30k–51k, a fresh weights-only stamp with its
derivation beside it, in the mirror root) BUILT for the EMA cell in run10's preflight window — the cell itself
NOT run today (the packet's placement and budget). CARDS: CARD-PROBE-1, CARD-RUN10-RULE, CARD-E1-RULER-R6 opened
(E1 READ), CARD-ARCH-D6 / RUN9-STOP-LAW / RUN9-QUEUE / PERF-3 moved. **What R366 owes:** run10's composition
under R365(c) — the soft-policy head admitted, proof-as-target and the mid-turn value premise dead, the gap's
witness pre-stated, the band's re-derivation (pooled < 0.30 or per-mr), the follower's unit (r8 stays or r6
re-reads the series at ≈ 3 × 1.2 bh), the EMA cell in run10's window, LR by that cell and the drift reading. **Exit:** `make gates.exit` on the leg's tip (`23003bb9`) ALL GREEN 2026-09-21 12:00 UTC (1 h 42 min: default tier
4 983 passed, integration 49 passed in 63 min, slow 5 passed; 5 047 collected against the 4 862 floor; pyright 0 errors on 258
files, the comment ratchet AT its floor on every measure); `dev` is seven commits ahead of `origin/dev`, UNPUSHED by this
session (a push is the operator's).

## Exit facts — the R353 packet, 2026-09-14

- Ruling: R353 at `c31fd74e`, verbatim; numbering continues at R354. The 2026-09-15 eval-cost
  decisions are OPERATOR DIRECTION in the session (recorded in `RUN7_EVAL_COST_2026-09-15.md` §D
  and contract rows v30/v31), not a numbered ruling; the memo names the protected-set item
  ("gate pair statistics") a ruling should name.
- Full local gate set on the leg's code: 19 GREEN + 3a green in the foreground; gate 1 not run.
- Collected tests: **4 988** collected (2026-09-17) against the committed floor 4 862. Comment
  ratchet: 3 451 / 0 / 13 387, plus private-docstring 1 530 and Rust-doc 2 677 since 2026-09-17
  (every floor lowered, none raised).
- Contracts: `game_record.md` gained `move_sims` / `move_arms` (self-play only); `run_config_schema.md`
  v30 (`eval.max_plies`) and v31 (`eval.gate.sequential`); `event_manifest.md` the gate's rule fields
  and the `gate_sequential` phase; the fixtures manifest re-pinned `drain_goldens.json` (the row's ninth field).
- Vendored patch: unchanged (three hunks); the TT fix is in the adapter, not the vendor tree.
- The box: run7 LIVE since the 2026-09-15 resume on `ba51fd46` (events in `seg0002`); the worktree
  `mantis-tt` at `748f5c47` exists only for the A/B's fixed arm; the chains `chain_tt_ab.sh` and
  `chain_strix_15k.sh` are spent.
- Ledger (R357 §0(6)): the exit sweep on `b4ddb21c` is ACCEPTED as partial — the full
  `run_all.sh --with-slow` ran on `8224b967` (19 green + the count-pin red), the fix and 3a re-gated on
  the tip, cargo unchanged; no re-run; the next full sweep is at the R357 leg's exit.
- R357 leg exit (2026-09-18): `run_all.sh --with-slow` on `9eaa2277` — 19 GREEN + 3a RED (one
  undeclared module-scope `importorskip` in the new `tests/diagnostics/test_ring_audit.py`, refused by
  the tier census, fixed at `a693a0a4`), 2a 1 976 s, 3b 49 passed / 2 skipped in 6 144 s (1 h 42 min on
  this host), the slow tier 5 passed; 3a re-run on the tip `a693a0a4`: 4 943 passed, 8 skipped, 56
  deselected, 341 s; 3c collected 5 007 against the floor 4 862, 77 deselected tests all declared;
  gate 1 not run (the accepted cost). `dev` PUSHED at `a693a0a4` (`b4ddb21c..a693a0a4`, 8 commits,
  the RESEARCH-STRENGTH-1 doc `03dba760` fast-forwarded in by the operator) on the operator's grant.
- R358 leg exit (2026-09-18, `dev` tip = this commit's parent chain `b8ad5ff0..`): gate 3a on the
  tip 4 966 passed / 8 skipped / 56 deselected (337 s); 3c collected 5 030 against the floor 4 862, 77
  deselected all declared; gates 4, 6, 7, 8, 9, 10, 11, 12 (rc 0, four rows deferred), 13, 14 (ruff
  + pyright 0 + the comment ratchet AT its floor on all five measures), 15, 16, 17 green; `cargo test`
  on the three touched crates (graph, selfplay, bridge) 297 passed, workspace clippy `-D clippy::all`
  clean; 2a on the whole workspace, 3b and the slow tier NOT run (untouched crates, no integration-
  marked test touched); gate 1 not run (the accepted cost). Nine gate-runner tests need a writable
  `UV_CACHE_DIR` in a sandboxed shell (`uv run` fails on a read-only `~/.cache/uv`); they pass with one.
  `dev` NOT pushed — the operator pushes (R358 §6 step 1).
- R359 leg exit (2026-09-18): docs and governance only, plus ONE one-line field comment
  (`schema/selfplay.py`); gate 13 rc 0 on the contract doc, gates 14/15/16/17 and 3c re-run on the
  tip (the field comment is the one source touch); `dev` NOT pushed — the operator pushes.
- R360 leg exit (2026-09-18): code in `preflight_stamp.py`, `run.py` (the flag), the new
  `tools/bench_server.py`; tests in `tests/config/test_preflight_stamp.py` (+4),
  `tests/test_run_main_authority.py` (+1), `tests/tools/test_bench_server.py` (+3); the launcher's
  pinned flag SET widened by name to admit `--inherit-preflight` (`tests/test_run_launcher.py`);
  five trap stubs widened (`test_run_pdeathsig.py`, `test_survivability.py`). Gates 10, 13, 14
  (ruff, pyright 0, the ratchet AT its floor on every measure after three folds), 15, 16, 17 (over
  every file changed since `9dbc6667`) green; 3c collected 5 039 against the floor 4 862, 77
  deselected all declared; 3a on the tip green with a writable `UV_CACHE_DIR` (the nine
  gate-runner tests that shell out to `uv run` red in a sandboxed shell without one, as recorded
  at the R358 exit). `dev` NOT pushed — the operator pushes.
- R361 leg exit (2026-09-19): code in `tools/strix_follower.py` (the switch) and two one-line
  docstring folds in `selfplay/pool.py` + `pool_hooks.py` (the sync hook said "a promoted state_dict";
  it is the learner's — the docstring floor lowered by the one line folded); tests in
  `tests/tools/test_strix_follower.py` (+2, one mutation-checked). Gates on the tip: 3a 4 977 passed /
  8 skipped / 56 deselected (334 s, writable `UV_CACHE_DIR`); 3c collected 5 041 against the floor
  4 862; 6, 7, 8, 9, 10, 11, 12 (rc 0, four rows deferred), 13, 14 (ruff, pyright 0, the ratchet AT
  its floor on every measure after the one fold), 15, 16, 17 (over every file changed since
  `8379073e`) green; 2a/2b/4/5, 3b and the slow tier NOT run (no crate touched, no integration- or
  slow-marked test touched); gate 1 not run (the accepted cost). `dev` PUSHED on the operator's
  in-session grant.
- R363 leg exit (2026-09-20): code in `tools/ladder/` (new `openings.py`; `backends.py`, `receipt.py`,
  `session.py`, `__init__.py`) and `tools/ladder_bot.py`, the VPS README; tests in `tests/tools/` (new
  `test_ladder_openings.py` +7; backends +3, receipt +4 plants +1, session +1, bot +1). Gates on the tip:
  3a 4 952 passed / 8 skipped / 56 deselected (341 s, writable `UV_CACHE_DIR`); 3c collected 5 016 against
  the floor 4 862, 79 deselected all declared; 6, 7, 8, 9, 10, 11, 12 (rc 0, three rows deferred), 13, 14
  (ruff, pyright 249 files 0 errors, the ratchet AT its floor on every measure), 15, 16, 17 green; 2a/2b/4/5,
  3b and the slow tier NOT run (no crate touched, no integration- or slow-marked test touched); gate 1 not
  run (the accepted cost). `dev` PUSHED on the packet's direction ("push so the record is one thing").
- Commits on this line: one line each, empty bodies, zero trailers; interleaved with the
  OBSERVATORY session's (linear: their branch was rebased on `35c89657` and fast-forwarded);
  `dev` was pushed at `ba51fd46` for the resume, and the REPAIR-A4 leg (`d44f3459..` this commit)
  is pushed under the operator's grant of 2026-09-17.

## Provenance

The 45k read derived 2026-09-21 on `dev` from the mirror alone (`events_run8_seg0001.jsonl` at 05:50 UTC, the 45k/51k rings and `.strix256.json`, the game records under `logs/games/` for the per-round gate counts, `eval_ladder_state.json` for the rung), no box contact. Item (15) derived 2026-09-20 on `dev` from the R363 packet, the mirror (`events_run8_seg0001.jsonl` at 06:49 UTC, run8 at 30 871), the three profile JSONs in that session's scratchpad (8 / 16 / 4 threads) and the gate logs of the session; NO box contact. Item (14) derived 2026-09-19 on `dev` at `ff5a43d8` from the commits named in it, the gate logs of that session, the mirror (`events_run8_seg0001.jsonl` pulled 09:42 UTC) and one read-only ssh to the box at 10:02 UTC (`ps -p 3211513`, `/workspace/oc7/chain_follower_run8.log`). Item (13) derived 2026-09-19 on `dev` from the box (`/workspace/oc7/chain_follower_run8.log`, `ps`, the
`r361` branch), the mirror's `events_run{7,8}_seg*.jsonl` (run8 pulled 06:05 UTC), the rounds'
`eval_spool.work/<run>/*_{result.json,progress.txt}` copied off the box's disk, the spool `.pt`
tensor hashes and the source files named in the record. Item (12) derived 2026-09-18 on `dev` from the R360 commits, `PERF3_2026-09-18.md` and the mirror's
`events_run8_seg0001.jsonl` (pulled 20:54 UTC). Item (11) derived 2026-09-18 on `dev` from `docs/design/measurements/PERF_A4_2026-09-11.md`, `git log`
(`c888c3b7`, `41a5fea8`), `configs/run8.yaml`, `src/mantis/config/preflight_stamp.py` and this host
(`torch.cuda.is_available()` False). Item (10) derived 2026-09-18 on the box (`/workspace/oc7/preflight_run8.log`, `preflight_shakedown8.log`,
`shakedown8.launch.log`, `witness_shakedown8.log`, `run8_supervisor.log`, `chain_follower_run8.log`,
`/workspace/runs/shakedown8/logs/events_shakedown8_seg0001.jsonl`). Item (9) and the R358 exit facts derived 2026-09-18 on `dev` at the R358 leg's exit commit, from
`configs/run8.yaml`, `tools/config_diff.py`, the gate logs of that session and `docs/governance/RULINGS.md`.
Everything below the heading otherwise: derived 2026-09-14 on `dev` at the leg's exit commit, from `configs/run7.yaml`, the box's
`/workspace/runs/run7/logs/events_run7_seg0001.jsonl` and
`eval_spool.work/run7/r000001_3000_progress.txt`, `/workspace/oc7/{run7_supervisor,chain_tt_ab}.log`,
`tools/ci_gates/test_count_floor.txt`, `tools/ci_gates/comment_length_floor.txt` and
`docs/governance/LAWS.md`. Ruling texts: `docs/governance/RULINGS.md`.
