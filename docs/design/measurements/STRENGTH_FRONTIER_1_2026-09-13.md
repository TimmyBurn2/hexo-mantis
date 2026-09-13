# MEASUREMENT — STRENGTH-FRONTIER-1 on the stopped run6 (R350(c), 2026-09-13)

Instrument: `tools/strength_frontier.py` (driver sha `1e586293055d` as run on the box; the tree
there is `0f20896e`, the block's own, so the eval child, the deploy head, the book and the sealbot
build are exactly run6's) composing one `RoundSpec` per cell from `configs/run6.yaml` through the
same resolvers `mantis.run` uses for its eval pipeline, and spawning `python -m mantis.eval.worker`
on it — nothing here plays a game the run's own eval worker would not play. Every cell keeps
`round_index 0`, so all cells draw the SAME 144-opening window from `book_v1_s20260625_p4` with
`seed_base 20260625`; cells are therefore paired across regimes, not only within. The candidate
plays each opening once per colour (288 games); the opponent is `sealbot_d5` (depth 5) except for
the two head-to-heads, which run the GATE block's screen at the stated deploy sims with the
confirm phase made unreachable. WR is draw-aware; the CI is a 2 000-resample bootstrap over
OPENING PAIRS (both legs of an opening are one observation) at 95 %; `s/game` is the cell's wall
over its games with 8–12 cells sharing the card, so it is a throughput reading of the box under
that load, not of one game alone. Box: RTX 5080, 24 cores; cells ran 4 (phase 1, from 08:27 UTC)
then 8 (phase 2, from 08:30 UTC) at a time, CPU ≈ 81 % busy, GPU ≈ 53 %.

The nets: `bc_full` = every tensor of the BC checkpoint of record (`run6_00006500_ca1afb71.ckpt`,
net hash `2e72abd4…`, 50 tensors — its value head IS trained on the human corpus' outcomes,
`train.value_target: pure_outcome_z` through the same train step as self-play; R350's landing note);
`bc_tp` = the same checkpoint through run6's seam (`representation.*` + `policy_head.*`, the value
head the seeded fresh init — what run6 booted with, up to the RNG position at build);
`ck3k/13k/18k/25k` = run6's frozen bundles at those steps; `ck35k` = the run's final net (step
35 084, the stop). Search kinds are HEAD's two deploy heads: `puct` plays the most-visited root
child; `gumbel` runs Sequential Halving with `gumbel_m 16`, `c_visit 50`, `c_scale 1.0`.

## A. The two cells the ruling asked for first, and the control pair beside them

| cell | WR | 95 % CI (pairs) | W–L–D | s/game | median plies |
|---|---|---|---|---|---|
| `bc_full` PUCT-150 | **0.413** | [0.358, 0.469] | 119–169–0 | 12.6 | 31 |
| `bc_full` Gumbel-160/m16 | **0.177** | [0.135, 0.219] | 51–237–0 | 14.5 | 25 |
| `bc_tp` PUCT-150 | **0.038** | [0.017, 0.062] | 11–277–0 | 13.0 | 31 |
| `bc_tp` Gumbel-160/m16 | **0.014** | [0.003, 0.028] | 4–284–0 | 15.7 | 19 |

Two facts, each with disjoint CIs. **The kind:** the same net reads 24 pp lower under the Gumbel
deploy head than under PUCT at equal sims — R350(c)'s first cell, decided: "if Gumbel deploy
reads far below PUCT's 53 %, the search implementation or its scale is the failure". **The head
set:** the BC checkpoint's OWN value head is worth 0.413 vs 0.038 at PUCT-150 — the seam that
threw it away cost run6 its warm start, and a BC prior searched over a random value head is
WORSE than the prior alone (the R342-era 7/32 and 17/32 readings of the same construction were
taken with a head that no longer exists; §D). Under Gumbel the two nets are 0.177 vs 0.014.

## B. The grid — frozen checkpoints × sims × kind vs `sealbot_d5`

Filled as cells land (WR [95 % CI over pairs]; W–L–D, s/game and median plies in the cell's
`cell.json` on the box). Blank = still running at the last read (10:02 UTC).

| net | kind | 128 | 256 | 512 |
|---|---|---|---|---|
| ck3k | puct | 0.431 [0.372, 0.483] (124–164–0, 33 plies) | | |
| ck3k | gumbel | 0.205 [0.160, 0.250] (59–229–0, 25 plies) | | |
| ck13k | puct | | | |
| ck13k | gumbel | **0.031** [0.014, 0.052] (9–279–0, 27 plies) | | |
| ck18k | puct | **0.568** [0.510, 0.625] (163–124–1, 49 plies) | | |
| ck18k | gumbel | 0.233 [0.191, 0.278] (67–221–0, 31 plies) | | |
| ck25k | puct | 0.497 [0.444, 0.549] (143–145–0, 45 plies) | | |
| ck25k | gumbel | 0.167 [0.128, 0.208] (48–240–0, 27 plies) | | |
| ck35k | gumbel | | — | — |

At 128 sims, read so far: under PUCT the block's nets are AT OR ABOVE the BC net (`bc_full`
PUCT-150 0.413): 0.431 at 3k → 0.568 at 18k → 0.497 at 25k — the block trained a net that
PUCT plays 15 pp better than its warm start at 18k. Under the Gumbel head the same nets read
0.205 → 0.031 (13k) → 0.233 → 0.167: run6's own screens, reproduced — and the 13k "trough" is a
3 % Gumbel reading of a net PUCT reads in the forties (interim 42 % at 237/288 games). The
Gumbel deploy head's reading fell where the value head's calibration moved (the block record:
value loss 0.53 → 0.57 at 4k–6k → 0.48 by 21k), which is what Sequential Halving's completed-Q
root pick depends on and PUCT's most-visited pick does not. Median plies under PUCT lengthen
with training (33 → 49): the net holds longer games against the depth-5 reader.

## C. The 25k net against its own prior at equal search (gate screen, 288 games)

| cell | WR (25k as candidate) | 95 % CI | W–L–D | s/game |
|---|---|---|---|---|
| ck25k vs `bc_full`, Gumbel-160/m16 | | | | |
| ck25k vs `bc_full`, PUCT-150 | | | | |

## D. What the R340 control was, read from the box

R350(a) states the control as "the R340 burst — same BC net, ALL heads, PUCT-50 self-play,
PUCT-150 deploy — read 53 % at step 0 and 79 % at step 2,004" vs `sealbot_d5`. Three things on
the record say otherwise, and the ruling's §2 asked for exactly this read before anything else:

1. **The head set.** The R342 bursts on the rebuilt box (2026-09-07, `/workspace/r342/burst/burst.log`
   and `/workspace/r342/g4/g4.log`) boot with `bc_warmstart_loaded … loaded_keys=46 verified_tensors=46`
   from `checkpoints/bc/run6_00006500_5191bd09.ckpt` and the seam's own warning
   `bc_warmstart_source_has_value_head … the value head stays fresh either way` — the same
   `BC_TRANSFER_PREFIXES` seam run6 booted with, one day after the R340 run 2 whose driver log
   died with the box. Every burst of that era started with a FRESH value head.
2. **The instrument.** Until `6ee52ca7` (2026-09-09, three days before run6's START) the deploy
   head was "PUCT tree + transformed-Q root pick" — `log(prior) + (c_visit + max_n)·c_scale·q`
   at the root, no Sequential Halving — a third algorithm that commit deleted. Neither of HEAD's
   two heads is the one that produced the control.
3. **The denominators.** 53.1 % = 17/32, the sitting-10 step-0 witness's `sealbot_d5` RUNG (32
   games, archive v3.47: "17/32, Elo +21.7 CI [−89, +163] — INDISTINGUISHABLE"); 72.5 % = 58/80
   and 78.8 % = 63/80 are 80-game fractions, and 80 is the GATE SCREEN's size — the candidate
   against the step-0 ANCHOR, its own starting net. The "monotone series vs sealbot_d5" is one
   32-game sealbot reading followed by two readings of the candidate beating its own step-0
   self. The R342 g4 round's rung at step 10 (same construction, old head, 32 games,
   `/workspace/r342/g4/run/logs/eval_spool.work/r342_g4/r000001_10_progress.txt`) reads
   **7/32 = 22 %** with its gate screen at 59/80 = 74 % against the anchor.

So the control is: the BC prior with a fresh value head, under a deleted head, read 17/32 once
and 7/32 once against `sealbot_d5`; nothing on the record put that net at 79 % against the bot.
Section A measures the same net under HEAD's heads at 288 games; the block's 16 % → 2.5 % → 16 %
was never a fall from 53 %.

## E. Findings

## Sources
