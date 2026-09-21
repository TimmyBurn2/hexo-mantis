# MEASUREMENT — run7's eval rounds outlast their cadence; the strix points at 9k and 15k (2026-09-15)

Read off the live run at 15:10–18:30 UTC (run7 21–25 h in, step 19 500–22 700, box `vast`, tree
`15109ac3`). The instrument is the run's own record: `events_run7_seg0001.jsonl`, the rounds'
`eval_spool.work/run7/<round>_progress.txt` rows (one per game with `t_wall`, `plies`, `phase`),
the mirrored game shards (pair structure), and three strix cells run through
`tools/strength_frontier.py` on the box beside the run.

## A. The rounds

| round | started | wall | verdict | screen (80g) | confirm (128g) | rung (288g) | sealbot WR |
|---|---|---|---|---|---|---|---|
| r1 @3k | 19:26 | 8 131 s | promoted | 2 154 s (0.49 s/ply, 55 plies/g) | 3 800 s (0.51, 58) | 2 142 s (0.19, 38) | 0.733 [0.68, 0.79] |
| r2 @6k | 22:51 | 12 816 s (under the SEALBOT-TT A/B) | not promoted | 5 191 s (0.84, 77) | 5 128 s (0.59, 68) | 2 456 s (0.20, 42) | 0.688 [0.64, 0.74] |
| r3 @9k | 03:03 | 12 483 s | promoted | 3 273 s (0.61, 67) | 4 729 s (0.56, 66) | 4 432 s (0.25, 61) | 0.670 [0.62, 0.72] |
| r4 @12k | 06:40 | **14 400 s — killed** (`eval_broken round_timeout`) | reading lost | 4 273 s (0.61, 87; p90 191) | 8 038 s (0.66, 95; p90 225, 8 at the 256 cap) | ≈ 2 080 s in, unfinished | — |
| @15k | — | `eval_round_skipped_busy` (r4 in flight) | — | | | | |
| r5 @18k | 13:21 | **14 400 s — killed** | reading lost | 4 036 s (0.60, 85; p90 173) | in flight at the bound | — | — |
| @21k | — | skipped busy (r5 in flight) | | | | | |

Every screen escalated under `screen_confirm_lo` 0.44 (0.675, 0.525, 0.637, 0.469, 0.463), so the
gate was a fixed 208-game block, 64–85 % of a round's wall; per-ply cost beside the trainer is
2× the idle stamp's (0.31 → 0.5–0.84 s), and the gate's games lengthened round by round (55 → 95
plies a game) because both sides block every four (`GAME_QUALITY_CENSUS_2026-09-14.md`). Sealbot's
own compute is ≈ 0 of the wall: a rung ply costs 0.19–0.25 s against a gate ply's 0.5–0.84 with two
searching sides, i.e. ≈ 0.25 s per searching side, and the rung is the candidate's 512-sim searches.

The trainer: 1 430 steps/h in the first 2 h, ≈ 900 from round 2 on; the decline tracks self-play
game length (median 45 → 63 plies, +40 %; one training step per game), the eval child's cost on it
is second-order (its sims/s median is the same beside a round as without). The A/B's two extra cells
beside a round read 470 steps/h — a second box job beside a round is the one thing that halves it.

## B. Pair structure (the book's ceiling)

Seat-decided pairs (the same seat wins both legs of an opening): rung 44 / 47 / 42 %; gate screen
45 / 60 / 38 / 68 / 55 %, confirm 33 / 48 / 50 / 55 %; the p2 seat takes 52–60 % of rung games.
With a fraction s of pairs decided by the seat a paired WR cannot exceed 1 − s/2 ≈ 0.78 at s 0.44;
run7's 0.67–0.73 sits near the book's ceiling, not sealbot's. Against strix 12 % of pairs are
seat-decided (a stronger side converts the bad seat too) — R353(e)'s "power loss, not a bias".

> ANNOTATION (R365(e), 2026-09-21; the paragraph above is unedited). "≈ 0.78" is a POPULATION number,
> not a constant of the book: 1 − s/2 is arithmetic on the sealbot RUNG's measured seat-decided
> fraction (42–47 % over run7's rung pairs on `book_v1`). The same book reads other ceilings on other
> populations — `STRIX_RUN7_60K_2026-09-17.md` §B.4's sealbot cells span 37–57 % seat-decided (a
> ceiling 0.72–0.82) and its strix cells 16–28 % (0.86–0.92; this section's own 12 % reads 0.94). Quoted
> as "the ≈ 0.78 book ceiling" at `STRIX_RUN7_60K` §B.4 and `STATE.md` (the 2026-09-15 rewrite, point 5)
> it is the rung population's ceiling carried as the book's; each population's ceiling is its own
> measured fraction (the audit: `STRENGTH_RESEARCH_2_2026-09-21.md` P5 List 2(a)).

## C. The strix points (cell A, as-shipped: ours PUCT-512 vs strix 128/m16, 288 games, the same book)

| net | WR | CI | wins | plies med | s/game (beside the run) |
|---|---|---|---|---|---|
| step 0, the BC net (`STRIX_RUNG_2026-09-14.md`) | 0.083 | [0.052, 0.118] | 24 | 39 | 7.0 |
| **step 9 000, the promoted anchor** (`best_model.pt.provenance.json`) | **0.097** | **[0.066, 0.132]** | 28 | 41 | 13.2 |
| step 15 000, the trainer's net | 0.045 | [0.021, 0.073] | 13 | 37 | 13.6 |
| step 15 000, equal work (cell B, 256/256) | 0.017 | [0.003, 0.035] | 5 | 35 | 10.2 |

Read together with the gate (r4 and r5 read 0.469 and 0.463 against the 9k anchor before their
rounds died) and the rung (0.733 → 0.688 → 0.670): the trainer's net peaked near 9k on the
external anchor and had regressed below the BC start by 15k; the promotion gate held the 9k net as
the deploy tag, which is what the gate is for. The trainer's losses over the same span: value
0.56 → 0.49 (down), policy 2.33 → 2.69 at 9–10k → 2.65 (up 0.3 nats, run6's trough shape,
R350(b)(iv)).

## D. What was changed for the resume (operator decisions, 2026-09-15; the memo they rest on: `docs/design/eval_gate_memo_2026-09-15.md`)

`configs/run7.yaml` re-minted at `ce0a8ff6` (contracts v30 + v31): `eval.max_plies` 256 — its own
row, no longer a copy of `selfplay.max_game_moves`; the 128 the operator first asked for was
measured a weak lever (3–5 % of plies in a normal round, 13–18 % in r4's, −0.02…+0.03 on the
reading) and run6/the frontier had run at 256; `eval.gate.deploy_sims` 512 → 256 and
`eval.sealbot_model_sims` 512 → 256 (the gate block halves; the rung series will read 6–13 pp
lower for the same strength — the frontier's two columns are the bridge: ck18k 0.774 → 0.646,
ck25k 0.689 → 0.601, ck3k 0.628 → 0.542, ck13k 0.599 → 0.538); `eval.gate.screen_confirm_lo` 0.44 →
0.5; and `eval.gate.sequential` armed — the GSPRT over pairs (H0 0.52 / H1 0.62, α 0.05, β 0.10,
16 pairs then every 8 to 104, the LLR's sign at the maximum), which on run7's pair model reproduces
the old rule's error curve within 0.02 with 26–50 % fewer games. Expected round: ≈ 4 500 s typical,
≈ 8 800 s worst (r4's lengths at the 104-pair maximum), against 12 500 / > 14 400 today; eval duty
≈ 40 % of wall instead of ≥ 100 %. Concurrency stays 8/8 (unmeasured lever), the cadence 3 000,
the book v1 until `book_v2` is measured. None of it reaches the live run: the parent holds its
config in memory and the tree is bound by the stamp, so the values arrive by stop → preflight
stamp → resume (`--resume-from`, launch config wins outside the checkpoint-owned set; verified on
the real 18k checkpoint, whose stamp predates `eval.max_plies` and loads with
`checkpoint_config_predates_schema`).

## E. What this record does not say

- Nothing about why the trainer regressed after 9k; the strix points are three, one per net.
- The GSPRT's live behaviour: simulated, and pinned through one real in-process round; its first
  live round is the resume's preflight terminal round.
- The trainer-rate gain from the shorter eval duty cycle is an estimate (+5–15 %); the one
  contention datapoint is the A/B's.

## Sources

`/workspace/runs/run7/logs/events_run7_seg0001.jsonl`, `eval_spool.work/run7/r00000{1..5}_*_progress.txt`,
`/workspace/oc7/strix_{step0,15k,best9k}/*/cell.json` (mirrored to `mantis-mirror/strix_*`),
`~/Work/HeXO/mantis-mirror/run7/logs/games` (pair structure), `/workspace/oc7/run7_check.py` and
`round_cost.py` (this session's scratch), `docs/design/eval_gate_memo_2026-09-15.md`.
