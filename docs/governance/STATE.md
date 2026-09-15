# STATE — where mantis actually is

Rewritten in place, never appended to. Every value below was read from the tree at the commit
named under "Provenance", not copied from a register. Where a register disagreed with the tree,
the tree won and the disagreement is recorded in the last section.

## Current phase — R353 LANDED; run7 STARTED 2026-09-14 17:27 UTC on `15109ac3` and PROMOTED at its first in-run round; SEALBOT-TT fixed in-repo, its A/B read on frozen nets; the GAME-QUALITY CENSUS answered "wins more"; VIEWER-1 labels every move's arm

**R353 is landed verbatim** (`c31fd74e`), the operator's forward of the START packet (dated
2026-09-14): run7 STARTS on the vested stamp; SEALBOT-TT is an instrument defect (the fourth on
record) fixed and A/B'd on three frozen nets; the GAME-QUALITY CENSUS over games that already
exist; VIEWER-1 labels every move's arm and sims; CARD-PUCT-ATTRACTOR closes on GAME-QUALITY's
reading. The START hold of 16:20 UTC is lifted by R353(a).

**The tree at the exit (`dev`):** `c31fd74e` R353 + the card closure · `748f5c47` **the sealbot fix**
(`new_game()` replaces a searched engine) · `136ab7b5` the dashboard's PROVISIONAL note ·
`416c95b4` the census record · `af47a8ab` **the arm producer** (Rust row → drain → record →
viewer) · `2f6328f1` CARD-SEALBOT-TT-SEAT updated · `51d4d5ea` test floor 4 811 · `35c89657`
the golden's manifest re-pin · `f33ea87c` CARD-SEALBOT-GIL-SERIAL closed · the A/B record, the dashboard note's
discharge, CARD-SEALBOT-TT-SEAT closed, this STATE. Full local gate set (`run_all.sh --with-slow`) on the code at `35c89657`: **19 GREEN
+ gate 3a green in a foreground re-run** (4 786 passed; the background launch had handed the
whole run `SIGINT=SIG_IGN`, which the supervisor posture probe correctly refuses — a launch
artefact, recorded here so nobody re-derives it); gate 1 not run (the accepted cost).

**What the leg did, in the ROUTE's order:**

1. **START.** `/workspace/oc7/box_start_run7.sh` at 17:27:46 UTC on stamp
   `preflight_run7_20260914T161609Z.json` (identity `0ad6b333…`, tree `15109ac3`, box branch
   `r352d`): `preflight_stamp_accepted`, `bc_warmstart_loaded loaded_keys=50 reinit=[]`, 32
   workers, the supervisor pid 3094932 over `mantis.run`. On the operator's machine the puller
   (`mantis-puller-run7.service`, 600 s) and the dashboard + run7 viewer refresh
   (`mantis-dashboard-run7.timer`, 10 min) are systemd USER units; the mirror is
   `mantis-mirror/run7`, the pages `mantis-mirror/dashboard/run7.html` and `viewer-run7/`.
   Trainer rate alone on the card ≈ 1 500–1 750 steps/h, ≈ 1 400 beside a round.
2. **The first in-run round** `r000001_3000` (19:26–21:42 UTC): **wall 8 131 s against 14 400**,
   PROMOTED (screen 54/80 → confirm 95/128), `sealbot_d5` rung 211/288 = 0.733 (old adapter,
   which the A/B read as transferring at ratio 1.00; the stamp's step-101 net read 0.635, run6's 3k net 0.63), phases probe 6 s · screen
   2 154 s (0.49 s/ply) · confirm 3 800 s (0.51 s/ply) · **rung 2 142 s (0.19 s/ply)** · floor 28 s.
   CARD-SEALBOT-GIL-SERIAL CLOSED on that wall: the rung is 1.7× its idle released wall where the
   GIL-held build's idle wall alone was 2 880 s; the gate blocks are the round's cost now.
3. **SEALBOT-TT** (`748f5c47`): the fix is in-repo — `SealBotAdapter.new_game()` swaps in a fresh
   `MinimaxBot` once the current one has searched (5 ms per game; per-search cost unchanged, 522
   vs 499 ms mean over 24 colour-alternating games, so TT growth moves nothing and the gate
   block's 0.20 → 0.30 s/ply drift — a block with no sealbot in it — cannot be TT at all). Pinned
   by a Tier-1 construction count and a Tier-2 colour-swapped pair, both RED on the old adapter
   (3 of the book's first 4 pairs diverge at the swapped game's first move). The root-player-keyed
   patch variant was refused: under `rung_concurrency 8` it would make a game depend on the
   scheduling of its thread's earlier games. The dashboard's strength panel carries
   `SEALBOT_TT_NOTE` on every record (the finding, since the A/B landed). **The A/B**: three frozen nets × old tree
   (`15109ac3`, run7's) vs the fixed worktree (`/workspace/mantis-tt` at `748f5c47`, Python 3.11,
   byte-identical sealbot build), 288 games each at concurrency 8, chained after round 1 —
   READ IN `docs/design/measurements/SEALBOT_TT_AB_2026-09-14.md` (see the A/B section below).
4. **GAME-QUALITY CENSUS** (`GAME_QUALITY_CENSUS_2026-09-14.md`, 13 389 games): PUCT-512's level is
   "wins more" — 0.3 % of must-block turns leave a four standing, 99.7 % of its wins are forced
   double threats after a ≥ 2-turn forcing run, its games are the same length won or lost (median
   41 / 41); the Gumbel head leaves 75.9 % of fours standing (the pre-stated 3-of-4), 79 % of its
   losses are its own missed blocks, and it loses FAST (23 plies). Blunder rate is a property of
   the head, not the net or the sims; line-extension share is not a strength axis.
5. **VIEWER-1's arm label** (`af47a8ab`): `GameResultRow` gained one `(sims, is_full_search)` per
   move from the runner's own draw; the record writes `move_sims` / `move_arms`
   (`opening`/`full`/`fast`, contract doc updated); the viewer names the arm and sims of the stone
   just placed and draws fast-arm stones dashed; a record from before the producer — run7's own
   shards, its tree being frozen — says "arm not recorded". Verified on a rendered page and
   through the real FFI.
6. **The strix point at 15k** is CHAINED on the box (`/workspace/oc7/chain_strix_15k.sh` waits for
   `run7_00015000_*.ckpt`, then cells A/B into `/workspace/oc7/strix_15k/`); it reads ≈ 11 h after
   START and is the next session's to record.

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

## Minted values — `configs/run7.yaml` (unchanged this leg; the running config)

**38 deltas** from the `dev` template, replayable. `selfplay.search.kind: gumbel` (320/64 at p 0.25,
`c_scale` 1.0, `q_rescale` true), `deploy.search.kind: puct`, `identity.warm_start` the LAW-12
strip with `reinit: []`, `train.ply_cap_abort {0.5, 600, 3000}`, `train.eval_interval` 3000 =
`checkpoint_interval`, deploy and eval PUCT-512, the rung 288 at `rung_concurrency` 8,
`eval.round_timeout_sec` 14 400, `n_workers` 32, `seed` 20260914. Gate 12 GREEN with two rows
DEFERRED.

## Protected set, laws, cards

Seventeen laws; no gate number added (still 17). Cards (`docs/governance/CARDS.md`): CLOSED this
leg — CARD-PUCT-ATTRACTOR (R353(e)), CARD-SEALBOT-GIL-SERIAL (round 1's wall), CARD-SEALBOT-TT-SEAT (the A/B's reading); OPEN —
CARD-GUMBEL-HEAD-RESIDUE,
CARD-DRAIN-POLLER-RACE, CARD-SELFPLAY-SEARCH-STATS. Owed still: the α = 1.0 three-row
reconstruction.

## Dispatcher state for a fresh session

run7 is LIVE; stopping it is ONE SIGTERM to the supervisor (save-then-exit). Open, in order:
(1) run7's rounds as they land (round 2, step 6 000, ran under the A/B's load and is not a wall
reading); (2) the strix point at 15k (`/workspace/oc7/strix_15k/`); (3) the
balanced opening book (`book_v2`); (4) CARD-GUMBEL-HEAD-RESIDUE; (5) the OBSERVATORY packet the
operator is drafting (merge the dashboard and the viewer — analysis first, a server is a
repo_design deviation). The worktree `/workspace/mantis-tt` exists only for the A/B's fixed arm.

## Exit facts — the R353 packet, 2026-09-14

- Ruling: R353 at `c31fd74e`, verbatim; numbering continues at R354.
- Full local gate set on the leg's code: 19 GREEN + 3a green in the foreground; gate 1 not run.
- Collected tests: 4 804 → **4 811** (the floor follows). Comment ratchet: 3 534 / 23 / 13 510
  (held: the leg's docstrings were trimmed to the floor, none raised).
- Contracts: `game_record.md` gained `move_sims` / `move_arms` (self-play only); `event_manifest.md`
  unchanged; the fixtures manifest re-pinned `drain_goldens.json` (the row's ninth field).
- Vendored patch: unchanged (three hunks); the TT fix is in the adapter, not the vendor tree.
- The box: run7 LIVE on `r352d` at `15109ac3`; the worktree `mantis-tt` at `748f5c47` for the
  A/B's fixed arm; the chains `chain_tt_ab.sh` and `chain_strix_15k.sh`.
- Commits on this line: one line each, empty bodies, zero trailers; `dev` NOT pushed (no
  operator approval this session).

## Provenance

Derived 2026-09-14 on `dev` at the leg's exit commit, from `configs/run7.yaml`, the box's
`/workspace/runs/run7/logs/events_run7_seg0001.jsonl` and
`eval_spool.work/run7/r000001_3000_progress.txt`, `/workspace/oc7/{run7_supervisor,chain_tt_ab}.log`,
`tools/ci_gates/test_count_floor.txt`, `tools/ci_gates/comment_length_floor.txt` and
`docs/governance/LAWS.md`. Ruling texts: `docs/governance/RULINGS.md`.
