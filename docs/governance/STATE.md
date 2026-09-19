# STATE — where mantis actually is

Rewritten in place, never appended to. Every value below was read from the tree at the commit
named under "Provenance", not copied from a register. Where a register disagreed with the tree,
the tree won and the disagreement is recorded in the last section.
Repaired in place 2026-09-17 (R311(c), REPAIR-A4 step 10, ledger C-1/C-2/C-5): the minted-values
block, the OPEN-card line, the exit block's box and push facts, dispatcher items (5) and (6); everything
else is the 2026-09-15 rewrite and reads as of that date.

## Current phase — **run8 LIVE since 2026-09-18 16:55:34 UTC** on the vested stamp (`c5d9e2fe…`, tree `86308bc7`; R358: σ + `train.augment: true` from the 42k parent), the shakedown witness PASSED and the ring audit PASSED all five bands with the two new rows READ; the strix follower chain runs beside it (parent solver ON → parent net_only → `--follow`), CONTENDED (dispatcher item 10 below); R359 landed the readings and the sourced queue (item 11); R360 re-aimed PERF-3 — steps 1–2 DONE on dev + mirror, step 3 is run9's preflight window — and made the twin's preflight inheritance code (item 12); **R361 (2026-09-19) withdrew the per-promotion strix trigger — the follower was relaunched `--no-promotions` at 06:15:06 UTC, run8 step 8 851 — and the EVAL CENSUS is READ (item 13): promotion selects nothing in the self-play loop, rounds cost the trainer 24–32 % while they run and strix cells 60–64 %, run7 spent 59 % of its wall in a round; **R362 (2026-09-19, item 14) RULED the rows: run8 runs to 30k regardless of the 15k reading, the sealbot rung is DELETED from the tree (contract v33), run9's eval rows are FIXED (cadence 15 000, gate 256/GSPRT unchanged, the gate's rule fields on the stream) and run9's ONE training swap and parent are ruled at 30k; the box's run8 stamp is untouched and still carries the retired rows, which the loader now tolerates**. run7 STOPPED 2026-09-18 06:25 UTC at 83 482. The 2026-09-15 resume record follows as history

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
   near the BOOK's ceiling (R353(e): a power loss; `book_v2` is the fix, carded).
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
REPORTED and stops nothing (R359(b)); replay_ratio 3.10, sym bin0/mean 1.000. Trainer at 17 119
(16:15 UTC), 1 099 steps/h alone; ETA 30k ≈ 05:45–07:15 UTC 2026-09-20, its cell's reading
≈ 1.5 h after.

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
- Commits on this line: one line each, empty bodies, zero trailers; interleaved with the
  OBSERVATORY session's (linear: their branch was rebased on `35c89657` and fast-forwarded);
  `dev` was pushed at `ba51fd46` for the resume, and the REPAIR-A4 leg (`d44f3459..` this commit)
  is pushed under the operator's grant of 2026-09-17.

## Provenance

Item (14) derived 2026-09-19 on `dev` at `ff5a43d8` from the commits named in it, the gate logs of that session, the mirror (`events_run8_seg0001.jsonl` pulled 09:42 UTC) and one read-only ssh to the box at 10:02 UTC (`ps -p 3211513`, `/workspace/oc7/chain_follower_run8.log`). Item (13) derived 2026-09-19 on `dev` from the box (`/workspace/oc7/chain_follower_run8.log`, `ps`, the
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
