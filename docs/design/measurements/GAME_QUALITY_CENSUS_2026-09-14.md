# MEASUREMENT — GAME-QUALITY CENSUS: is PUCT-512's 0.77 "wins more" or "loses later" (2026-09-14)

Ordered by R353(c) over games that already exist: the frontier's 33 cells, the two strix step-0
cells, shakedown7's self-play and the run7 stamp's external games (13 389 games, 10 368 of them
eval-channel, every game read whole). No box time. The question it answers, as the ruling put
it: whether PUCT-512's 0.77 against sealbot_d5 is "wins more" or "loses later"; the pre-stated
reading (R353(c)) was that if PUCT's wins are mostly forced sequences and its fours-left-standing
rate is near zero while Gumbel's is near 0.75, PUCT is the better player and the longer games are
competent defence on both sides.

## Verdict

**"Wins more."** Pooled over the five PUCT-512 sealbot cells (1 440 games, WR 0.640): the
candidate leaves an opponent four standing in **0.3 %** of its must-block turns (sealbot 0.0 %),
**99.7 %** of its wins end in a forced double threat after a forcing run the loser could not step
out of (median 5 loser turns, p10 4), and its losses are the same motif pointed back (87 % forced by
sealbot, 1.7 % its own missed block). Its games are the same length whether it wins or loses
(median 41 plies both; p90 91 vs 95). The Gumbel deploy head at 512 sims on the same nets (WR
0.037) leaves **75.9 %** of fours standing — the 3-of-4 the ruling pre-stated — and 79 % of its
losses are its own missed blocks; its losses are SHORT (median 23 plies vs 33 for its wins),
because a side that leaves fours standing loses fast. So the longer PUCT games are two sides that
both block every four and win only by building an unhittable threat set; "beats fast" is what
happened to the Gumbel head, and the pre-stated reading holds on every column.

The instrument's own check: sealbot_d5 leaves 0.0 % of fours standing in every cell it plays, and
every side's missed-win rate is 0.0 % except shakedown7's (3.5 % of its wins came from the other
side missing a win in-turn).

## Definitions (per turn, on 6-cell windows along the three hex axes; instrument 1 of `GAME_QUALITY_2026-09-14.md`, validated 0 disagreements against the engine on 18 190 positions)

- **four** of P: a window with ≥ 4 P stones and no opponent stone. **CHECK**: the mover starts its
  turn with ≥ 1 opponent four on the board and some ≤ 2-cell set hits every one. **LOST1**: the
  opponent's fours have no ≤ 2-cell hitting set (a threat set two stones cannot answer — the
  "double threat" of a two-stone game). **WIN1**: the mover can complete six this turn.
- **fours left standing / CHECK %** ("blunders"): of the mover's CHECK turns, the share after which
  an opponent four still stands (the mover did not block and did not win).
- **fours made / game**: turns after which the mover has ≥ 1 four it did not have before.
- **win by forced double threat**: the loser's last turn started LOST1 and the loser had been in
  CHECK or LOST1 at ≥ 2 consecutive turn starts before the end (a forcing run); **direct double
  threat**: LOST1 reached from a QUIET turn (no run); **by opponent missed block**: the loser's
  last turn started in CHECK with a hitting set available; **by opponent missed win**: the loser
  had WIN1 and did not take it.
- **line-extension share**: of a side's stones placed while it had ≥ 1 stone, the share hex-adjacent
  to a cell of the side's longest contiguous own-stone run(s) at the moment of placement.
- **plies at win / at loss**: median game length over the candidate's wins and its losses.

## A. Per kind × sims × opponent (candidate nets pooled: BC net, run6 ck3k/13k/18k/25k/35k; the σ-variant cells of phase 3 are shown per cell only)

| cell (n cells) | games | cand WR | cand fours left standing / CHECK % | opp fours left standing / CHECK % | cand fours made / game | opp fours made / game | cand line-ext % | opp line-ext % | plies at cand win (med) | plies at cand loss (med) |
|---|---|---|---|---|---|---|---|---|---|---|
| gumbel-128 vs sealbot_d5 (5) | 1440 | 0.157 | 72.3 | 0.0 | 2.06 | 2.10 | 33.4 | 44.8 | 33 | 25 |
| gumbel-160 vs bc_full (1) | 288 | 0.438 | 76.4 | 70.8 | 1.24 | 1.46 | 36.7 | 50.0 | 19 | 19 |
| gumbel-160 vs sealbot_d5 (2) | 576 | 0.095 | 86.7 | 0.0 | 1.58 | 2.00 | 43.2 | 54.5 | 31 | 19 |
| gumbel-256 vs sealbot_d5 (4) | 1152 | 0.076 | 78.5 | 0.0 | 1.42 | 2.12 | 29.2 | 49.9 | 33 | 23 |
| gumbel-512 vs sealbot_d5 (4) | 1152 | 0.037 | 75.9 | 0.0 | 1.09 | 2.20 | 27.7 | 53.5 | 33 | 23 |
| puct-128 vs sealbot_d5 (4) | 1152 | 0.490 | 0.5 | 0.0 | 5.62 | 4.35 | 31.4 | 29.6 | 43 | 47 |
| puct-150 vs bc_full (1) | 288 | 0.672 | 0.2 | 0.8 | 9.85 | 7.19 | 27.1 | 28.3 | 53 | 47 |
| puct-150 vs sealbot_d5 (2) | 576 | 0.226 | 1.9 | 0.0 | 2.95 | 3.56 | 36.7 | 45.8 | 31 | 33 |
| puct-256 vs sealbot_d5 (4) | 1152 | 0.582 | 0.2 | 0.0 | 5.80 | 4.21 | 32.0 | 29.0 | 44 | 49 |
| puct-256 vs strix@256 (1) | 288 | 0.073 | 1.3 | 0.7 | 3.76 | 4.69 | 38.0 | 37.2 | 31 | 43 |
| puct-512 vs sealbot_d5 (5) | 1440 | 0.640 | 0.3 | 0.0 | 5.39 | 3.75 | 35.0 | 29.3 | 41 | 41 |
| puct-512 vs strix@128 (1) | 288 | 0.083 | 2.2 | 1.3 | 3.62 | 4.62 | 36.7 | 36.2 | 31 | 41 |

| cell | cand wins | forced double-threat % | direct double-threat % | opp missed block % | opp missed win % | opp wins | forced % | direct % | cand missed block % | cand missed win % |
|---|---|---|---|---|---|---|---|---|---|---|
| gumbel-128 vs sealbot_d5 | 226 | 100.0 | 0.0 | 0.0 | 0.0 | 1214 | 3.7 | 6.9 | 89.4 | 0.0 |
| gumbel-160 vs bc_full | 126 | 0.0 | 0.0 | 100.0 | 0.0 | 162 | 0.0 | 0.0 | 100.0 | 0.0 |
| gumbel-160 vs sealbot_d5 | 55 | 100.0 | 0.0 | 0.0 | 0.0 | 521 | 1.3 | 17.7 | 81.0 | 0.0 |
| gumbel-256 vs sealbot_d5 | 88 | 100.0 | 0.0 | 0.0 | 0.0 | 1064 | 2.6 | 11.7 | 85.6 | 0.0 |
| gumbel-512 vs sealbot_d5 | 43 | 100.0 | 0.0 | 0.0 | 0.0 | 1109 | 4.1 | 16.5 | 79.4 | 0.0 |
| puct-128 vs sealbot_d5 | 563 | 99.8 | 0.0 | 0.2 | 0.0 | 587 | 86.5 | 10.7 | 2.7 | 0.0 |
| puct-150 vs bc_full | 187 | 87.2 | 2.7 | 10.2 | 0.0 | 88 | 88.6 | 9.1 | 2.3 | 0.0 |
| puct-150 vs sealbot_d5 | 130 | 100.0 | 0.0 | 0.0 | 0.0 | 446 | 82.3 | 13.2 | 4.5 | 0.0 |
| puct-256 vs sealbot_d5 | 670 | 99.6 | 0.1 | 0.3 | 0.0 | 482 | 86.1 | 12.2 | 1.7 | 0.0 |
| puct-256 vs strix@256 | 21 | 66.7 | 0.0 | 33.3 | 0.0 | 267 | 91.4 | 4.5 | 4.1 | 0.0 |
| puct-512 vs sealbot_d5 | 920 | 99.7 | 0.2 | 0.1 | 0.0 | 518 | 87.1 | 11.2 | 1.7 | 0.0 |
| puct-512 vs strix@128 | 24 | 50.0 | 4.2 | 45.8 | 0.0 | 264 | 88.6 | 4.5 | 6.8 | 0.0 |


Forced-run lengths (loser turns in continuous CHECK/LOST1 before the end), pooled sealbot cells:
PUCT-512 candidate wins median 5 (p10 4), its losses median 3; Gumbel-512 candidate wins median 5,
its losses median **1** (it is mated from a quiet turn or walks into a four). Game length p90:
PUCT-512 wins 91 / losses 95; Gumbel-512 wins 41 / losses 35.

## B. Per cell

| cell | games | cand WR | cand FLS/CHECK % | opp FLS/CHECK % | cand fours/game | opp fours/game | cand line-ext % | opp line-ext % | plies win | plies loss |
|---|---|---|---|---|---|---|---|---|---|---|
| bc_full_gumbel160 | 288 | 0.177 | 79.8 | 0.0 | 2.54 | 1.98 | 50.2 | 46.7 | 31 | 23 |
| bc_full_puct150 | 288 | 0.413 | 3.7 | 0.0 | 4.09 | 3.14 | 46.6 | 39.1 | 31 | 33 |
| bc_tp_gumbel160 | 288 | 0.014 | 94.6 | 0.0 | 0.62 | 2.03 | 33.8 | 64.5 | 34 | 19 |
| bc_tp_puct150 | 288 | 0.038 | 0.3 | 0.0 | 1.81 | 3.99 | 26.4 | 52.5 | 31 | 31 |
| ck13k_gumbel128 | 288 | 0.031 | 71.1 | 0.0 | 0.87 | 2.30 | 19.0 | 48.8 | 37 | 27 |
| ck13k_gumbel256 | 288 | 0.014 | 75.5 | 0.0 | 0.55 | 2.25 | 17.0 | 53.3 | 29 | 25 |
| ck13k_gumbel512 | 288 | 0.007 | 71.9 | 0.0 | 0.42 | 2.29 | 15.6 | 55.3 | 37 | 25 |
| ck13k_puct128 | 288 | 0.464 | 0.4 | 0.0 | 5.64 | 5.19 | 25.0 | 28.3 | 63 | 49 |
| ck13k_puct256 | 288 | 0.538 | 0.1 | 0.0 | 5.84 | 5.15 | 24.7 | 27.6 | 65 | 55 |
| ck13k_puct512 | 288 | 0.599 | 0.1 | 0.0 | 5.70 | 5.23 | 24.6 | 27.7 | 69 | 47 |
| ck18k_gumbel128 | 288 | 0.233 | 73.8 | 0.0 | 2.60 | 1.96 | 37.1 | 41.1 | 31 | 27 |
| ck18k_gumbel256 | 288 | 0.111 | 81.8 | 0.0 | 1.84 | 2.01 | 33.3 | 47.0 | 32 | 25 |
| ck18k_gumbel512 | 288 | 0.066 | 75.7 | 0.0 | 1.58 | 2.16 | 32.2 | 49.2 | 33 | 23 |
| ck18k_puct128 | 288 | 0.568 | 0.6 | 0.0 | 6.55 | 4.15 | 33.2 | 27.5 | 45 | 52 |
| ck18k_puct256 | 288 | 0.646 | 0.2 | 0.1 | 6.78 | 4.34 | 32.1 | 26.3 | 41 | 63 |
| ck18k_puct512 | 288 | 0.774 | 0.0 | 0.0 | 6.38 | 3.28 | 37.7 | 25.7 | 41 | 57 |
| ck25k_gumbel128 | 288 | 0.167 | 71.0 | 0.0 | 2.76 | 2.11 | 39.1 | 42.9 | 33 | 25 |
| ck25k_gumbel256 | 288 | 0.056 | 77.5 | 0.0 | 1.80 | 2.18 | 33.3 | 49.3 | 35 | 23 |
| ck25k_gumbel512 | 288 | 0.014 | 76.5 | 0.0 | 1.30 | 2.24 | 31.6 | 53.8 | 35 | 21 |
| ck25k_puct128 | 288 | 0.497 | 0.6 | 0.0 | 6.29 | 4.05 | 34.1 | 28.7 | 41 | 51 |
| ck25k_puct256 | 288 | 0.601 | 0.3 | 0.0 | 6.39 | 3.67 | 35.9 | 28.9 | 45 | 51 |
| ck25k_puct512 | 288 | 0.689 | 0.4 | 0.0 | 6.34 | 3.55 | 37.3 | 27.8 | 43 | 49 |
| ck25k_vs_bcfull_gumbel160 | 288 | 0.438 | 76.4 | 70.8 | 1.24 | 1.46 | 36.7 | 50.0 | 19 | 19 |
| ck25k_vs_bcfull_puct150 | 288 | 0.672 | 0.2 | 0.8 | 9.85 | 7.19 | 27.1 | 28.3 | 53 | 47 |
| ck35k_gumbel128 | 288 | 0.149 | 67.6 | 0.0 | 1.93 | 2.18 | 31.0 | 45.5 | 33 | 25 |
| ck3k_gumbel128 | 288 | 0.205 | 79.8 | 0.0 | 2.12 | 1.93 | 41.6 | 46.0 | 31 | 23 |
| ck3k_gumbel256 | 288 | 0.125 | 79.8 | 0.0 | 1.50 | 2.02 | 33.2 | 50.2 | 32 | 23 |
| ck3k_gumbel512 | 288 | 0.062 | 80.0 | 0.0 | 1.05 | 2.11 | 32.1 | 56.0 | 33 | 21 |
| ck3k_puct128 | 288 | 0.431 | 0.3 | 0.1 | 4.01 | 4.00 | 35.2 | 35.0 | 33 | 33 |
| ck3k_puct256 | 288 | 0.542 | 0.4 | 0.1 | 4.16 | 3.69 | 38.5 | 35.0 | 33 | 33 |
| ck3k_puct512 | 288 | 0.628 | 0.9 | 0.1 | 4.55 | 3.21 | 42.4 | 32.7 | 33 | 31 |
| ck18k_gumbel128_raw_c1 | 288 | 0.219 | 67.5 | 0.0 | 2.55 | 2.06 | 35.7 | 41.9 | 33 | 27 |
| ck18k_gumbel128_rescale_c01 | 288 | 0.167 | 72.5 | 0.0 | 2.19 | 2.08 | 32.5 | 43.5 | 33 | 27 |
| ck18k_gumbel512_raw_c1 | 288 | 0.073 | 81.2 | 0.0 | 1.45 | 2.09 | 30.6 | 51.6 | 33 | 23 |
| ck18k_gumbel512_rescale_c01 | 288 | 0.056 | 80.1 | 0.0 | 1.33 | 2.11 | 29.3 | 51.0 | 35 | 23 |
| strix_A_asshipped_bc | 288 | 0.083 | 2.2 | 1.3 | 3.62 | 4.62 | 36.7 | 36.2 | 31 | 41 |
| strix_B_equalwork_bc | 288 | 0.073 | 1.3 | 0.7 | 3.76 | 4.69 | 38.0 | 37.2 | 31 | 43 |
| shakedown7_selfplay | 2421 | 0.844 | 15.5 | (self) | 6.63 | (self) | 16.5 | (self) | 91 | draw 256 (755) |
| run7_stamp_external | 288 | 0.507 | 0.3 | 0.0 | 3.98 | 3.50 | 37.6 | 35.1 | 31 | 37 |

| cell | cand wins | forced DT % | direct DT % | opp missed block % | opp missed win % | opp wins | forced % | direct % | cand missed block % | cand missed win % |
|---|---|---|---|---|---|---|---|---|---|---|
| bc_full_gumbel160 | 51 | 100.0 | 0.0 | 0.0 | 0.0 | 237 | 2.5 | 8.9 | 88.6 | 0.0 |
| bc_full_puct150 | 119 | 100.0 | 0.0 | 0.0 | 0.0 | 169 | 78.1 | 11.2 | 10.7 | 0.0 |
| bc_tp_gumbel160 | 4 | 100.0 | 0.0 | 0.0 | 0.0 | 284 | 0.4 | 25.0 | 74.6 | 0.0 |
| bc_tp_puct150 | 11 | 100.0 | 0.0 | 0.0 | 0.0 | 277 | 84.8 | 14.4 | 0.7 | 0.0 |
| ck13k_gumbel128 | 9 | 100.0 | 0.0 | 0.0 | 0.0 | 279 | 5.0 | 10.4 | 84.6 | 0.0 |
| ck13k_gumbel256 | 4 | 100.0 | 0.0 | 0.0 | 0.0 | 284 | 2.5 | 12.7 | 84.9 | 0.0 |
| ck13k_gumbel512 | 2 | 100.0 | 0.0 | 0.0 | 0.0 | 286 | 3.8 | 18.2 | 78.0 | 0.0 |
| ck13k_puct128 | 133 | 100.0 | 0.0 | 0.0 | 0.0 | 154 | 88.3 | 9.1 | 2.6 | 0.0 |
| ck13k_puct256 | 155 | 100.0 | 0.0 | 0.0 | 0.0 | 133 | 83.5 | 15.8 | 0.8 | 0.0 |
| ck13k_puct512 | 172 | 99.4 | 0.6 | 0.0 | 0.0 | 115 | 87.0 | 13.0 | 0.0 | 0.0 |
| ck18k_gumbel128 | 67 | 100.0 | 0.0 | 0.0 | 0.0 | 221 | 3.6 | 5.9 | 90.5 | 0.0 |
| ck18k_gumbel256 | 32 | 100.0 | 0.0 | 0.0 | 0.0 | 256 | 3.1 | 12.5 | 84.4 | 0.0 |
| ck18k_gumbel512 | 19 | 100.0 | 0.0 | 0.0 | 0.0 | 269 | 4.8 | 16.4 | 78.8 | 0.0 |
| ck18k_puct128 | 163 | 100.0 | 0.0 | 0.0 | 0.0 | 124 | 84.7 | 11.3 | 4.0 | 0.0 |
| ck18k_puct256 | 186 | 99.5 | 0.0 | 0.5 | 0.0 | 102 | 95.1 | 2.9 | 2.0 | 0.0 |
| ck18k_puct512 | 223 | 100.0 | 0.0 | 0.0 | 0.0 | 65 | 92.3 | 7.7 | 0.0 | 0.0 |
| ck25k_gumbel128 | 48 | 100.0 | 0.0 | 0.0 | 0.0 | 240 | 3.8 | 2.5 | 93.8 | 0.0 |
| ck25k_gumbel256 | 16 | 100.0 | 0.0 | 0.0 | 0.0 | 272 | 2.6 | 6.2 | 91.2 | 0.0 |
| ck25k_gumbel512 | 4 | 100.0 | 0.0 | 0.0 | 0.0 | 284 | 3.5 | 14.1 | 82.4 | 0.0 |
| ck25k_puct128 | 143 | 100.0 | 0.0 | 0.0 | 0.0 | 145 | 86.2 | 10.3 | 3.4 | 0.0 |
| ck25k_puct256 | 173 | 100.0 | 0.0 | 0.0 | 0.0 | 115 | 87.0 | 11.3 | 1.7 | 0.0 |
| ck25k_puct512 | 198 | 100.0 | 0.0 | 0.0 | 0.0 | 89 | 87.6 | 10.1 | 2.2 | 0.0 |
| ck25k_vs_bcfull_gumbel160 | 126 | 0.0 | 0.0 | 100.0 | 0.0 | 162 | 0.0 | 0.0 | 100.0 | 0.0 |
| ck25k_vs_bcfull_puct150 | 187 | 87.2 | 2.7 | 10.2 | 0.0 | 88 | 88.6 | 9.1 | 2.3 | 0.0 |
| ck35k_gumbel128 | 43 | 100.0 | 0.0 | 0.0 | 0.0 | 245 | 3.7 | 6.9 | 89.4 | 0.0 |
| ck3k_gumbel128 | 59 | 100.0 | 0.0 | 0.0 | 0.0 | 229 | 2.2 | 8.3 | 89.5 | 0.0 |
| ck3k_gumbel256 | 36 | 100.0 | 0.0 | 0.0 | 0.0 | 252 | 2.4 | 15.9 | 81.7 | 0.0 |
| ck3k_gumbel512 | 18 | 100.0 | 0.0 | 0.0 | 0.0 | 270 | 4.1 | 17.4 | 78.5 | 0.0 |
| ck3k_puct128 | 124 | 99.2 | 0.0 | 0.8 | 0.0 | 164 | 86.6 | 12.2 | 1.2 | 0.0 |
| ck3k_puct256 | 156 | 98.7 | 0.6 | 0.6 | 0.0 | 132 | 81.1 | 16.7 | 2.3 | 0.0 |
| ck3k_puct512 | 181 | 99.4 | 0.0 | 0.6 | 0.0 | 107 | 86.0 | 9.3 | 4.7 | 0.0 |
| ck18k_gumbel128_raw_c1 | 63 | 100.0 | 0.0 | 0.0 | 0.0 | 225 | 5.3 | 6.2 | 88.4 | 0.0 |
| ck18k_gumbel128_rescale_c01 | 48 | 100.0 | 0.0 | 0.0 | 0.0 | 240 | 3.3 | 6.7 | 90.0 | 0.0 |
| ck18k_gumbel512_raw_c1 | 21 | 100.0 | 0.0 | 0.0 | 0.0 | 267 | 4.1 | 13.5 | 82.4 | 0.0 |
| ck18k_gumbel512_rescale_c01 | 16 | 100.0 | 0.0 | 0.0 | 0.0 | 272 | 3.3 | 18.4 | 78.3 | 0.0 |
| strix_A_asshipped_bc | 24 | 50.0 | 4.2 | 45.8 | 0.0 | 264 | 88.6 | 4.5 | 6.8 | 0.0 |
| strix_B_equalwork_bc | 21 | 66.7 | 0.0 | 33.3 | 0.0 | 267 | 91.4 | 4.5 | 4.1 | 0.0 |
| shakedown7_selfplay | 1666 | 17.2 | 13.3 | 65.9 | 3.5 |  |  |  |  |  |
| run7_stamp_external | 146 | 99.3 | 0.7 | 0.0 | 0.0 | 142 | 85.2 | 13.4 | 1.4 | 0.0 |

`FLS` = fours left standing; `DT` = double threat. `shakedown7_selfplay` is self-play (both sides
the same net; "cand" columns pool both sides; its `cand WR` is the decisive share, 755 of 2 421 games
drew at the 256-ply cap); `run7_stamp_external` is the start stamp's 288 games (PUCT-512, step 101,
`RUN7_STAMP2_2026-09-14.md`). `bc_tp_*` cells are the BC net through the trunk+policy seam (fresh
value head) — R350's control shape, not a reading of the BC net.

## C. What the columns say beyond the verdict

- **Blunder rate is a property of the HEAD, not the net or the sims.** Every PUCT cell reads
  0.0–3.7 % fours left standing (3.7 % only for the BC net at 150 sims); every Gumbel cell reads
  67.5–94.6 %, at 128, 256 and 512 sims alike, on the BC net and on every run6 checkpoint, under
  all three σ pairs (phase 3: 67.5 / 72.5 / 81.2 / 80.1 %). Sims move the WR inside each head
  (PUCT 0.49 → 0.58 → 0.64 pooled) without moving the blunder rate — the contest between two sides
  that block everything is decided by search depth in the forcing run.
- **Fours made** follows the same split: PUCT makes 5.4–5.8 fours per game to sealbot's 3.8–4.4
  against it; the Gumbel head makes 1.1–2.1 to sealbot's 2.1–2.2. Against strix (128 / 256) the BC
  net at PUCT makes 3.6–3.8 to strix's 4.6–4.7 and wins 0.07–0.08: strix out-builds it, and 89–91 %
  of strix's wins are forced runs; of the BC net's 45 wins, 33–46 % came from a strix missed block
  (strix-128/256 leaves 0.7–1.3 % of fours standing — not tactically clean either).
- **Line-extension share** is NOT a strength axis: PUCT 25–42 %, the Gumbel head 16–50 %, sealbot
  26–65 % depending on who it plays, strix 36–37 %. Sealbot's share rises against the Gumbel head
  (49–65 %) and falls against PUCT (26–33 %) — it extends its own line when nothing forces it to
  block, so the column reads the OPPONENT's pressure as much as the mover's style.
- **The promotion gate at Gumbel-160 vs the BC snapshot** (`ck25k_vs_bcfull_gumbel160`): 100 % of
  both sides' wins are the other side's missed block, 0 forced. The same pairing at PUCT-150:
  87 % forced, 10 % missed block. A Gumbel-vs-Gumbel gate compared two blind heads.
- **shakedown7's PUCT self-play** (τ 0.5 throughout, 320/64) is the one PUCT set that blunders
  (15.5 % fours left standing, 66 % of decisive games end by a missed block): the head is not the
  regime, and CARD-PUCT-ATTRACTOR's closing line stands.

## D. What this record does not say

- Nothing here is an Elo; the columns are one-turn tactics and endings, and "loses later" was
  answered by game length AND ending class, not by length alone.
- The frontier's sealbot cells were played through the pre-R353(b) adapter (CARD-SEALBOT-TT-SEAT):
  the WR column is PROVISIONAL until the A/B re-derives the level; the blunder and ending
  columns are read off the moves themselves and do not depend on the bar's level.
- The forced run is measured from the loser's side (how long it was forced), not proved from the
  attacker's (the pure-fours solver of `GAME_QUALITY_2026-09-14.md` §D.5 proves 2–5 attacker turns
  within budget).
- run6's Gumbel self-play (the TRAINER) is not in this census; `GAME_QUALITY_2026-09-14.md` §B read
  it at 3.6 % fours left standing and 81 % constructed endings.

## Sources

Games: `mantis-mirror/frontier/phase{1,2,3}/<cell>/games` (run_id `frontier1`, 33 cells, the
`cell.json` of each for kind/sims/opponent; the 4 random-floor games per cell are excluded, the
`ck25k_vs_bcfull_*` cells' games are the gate block's `promotion` channel), `mantis-mirror/strix_step0/
strix_{A,B}_*/games`, `mantis-mirror/shakedown7/logs/games` (2 421 self-play), `mantis-mirror/
run7-stamp-d13bc7c3/games` (288 external). Read with `mantis.monitor.game_record.iter_run_games`.

Scripts (this session's scratchpad, `census/`, untracked, dev-only): `tactics.py` (the validated
one-turn instrument, copied unchanged from the GAME-QUALITY session), `census.py` (per game:
autopsy rows, fours made, line extension, ending class; 14 processes), `agg.py` (the tables);
outputs `out/census_games.jsonl` (13 389 rows), `out/census_tables.md`, `out/census_tables.json`.
