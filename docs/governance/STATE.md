# STATE — where mantis actually is

Rewritten in place, never appended to. Every value below was read from the tree at the commit
named under "Provenance", not copied from a register. Where a register disagreed with the tree,
the tree won and the disagreement is recorded in the last section.
Repaired in place 2026-09-17 (R311(c), REPAIR-A4 step 10, ledger C-1/C-2/C-5): the minted-values
block, the OPEN-card line, the exit block's box and push facts, dispatcher items (5) and (6); everything
else is the 2026-09-15 rewrite and reads as of that date.

## Current phase — run7 RESUMED 2026-09-15 20:03 UTC from step 23 829 on the re-minted config (tree `ba51fd46`, gate/rung 256, GSPRT armed); the strix anchor reads the run peaked near 9k; the first resumed round (@24k) is the re-mint's live confirmation

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
`actor_lag` row was RETIRED by B-1, CARD-SERVER-OWNED-COPY).

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
Run-ops on the box (fresh session, alias `vast`, R356 §5 7–13 + the four R357 deltas): preflight →
parent copy + sha → stamp → run7 STOP at once (record the final step and r22's state) → 3 h
shakedown → witness (§3) + audit (§3a) on the same ring → START → the follower's first cell,
CONTENDED if the twin is up. RESEARCH-STRENGTH runs in a web-enabled session beside it.

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
- Commits on this line: one line each, empty bodies, zero trailers; interleaved with the
  OBSERVATORY session's (linear: their branch was rebased on `35c89657` and fast-forwarded);
  `dev` was pushed at `ba51fd46` for the resume, and the REPAIR-A4 leg (`d44f3459..` this commit)
  is pushed under the operator's grant of 2026-09-17.

## Provenance

Derived 2026-09-14 on `dev` at the leg's exit commit, from `configs/run7.yaml`, the box's
`/workspace/runs/run7/logs/events_run7_seg0001.jsonl` and
`eval_spool.work/run7/r000001_3000_progress.txt`, `/workspace/oc7/{run7_supervisor,chain_tt_ab}.log`,
`tools/ci_gates/test_count_floor.txt`, `tools/ci_gates/comment_length_floor.txt` and
`docs/governance/LAWS.md`. Ruling texts: `docs/governance/RULINGS.md`.
