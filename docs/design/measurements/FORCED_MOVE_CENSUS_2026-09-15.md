# MEASUREMENT — FORCED-MOVE CENSUS: is the forced block inside the Gumbel-16 target's support? (2026-09-15)

Ordered by R354(b) over the run record as it exists: hypothesis (b) says the self-play head
(Gumbel, m = 16 over the legal set) leaves a forced block un-sampled, so the completed-Q target
puts its mass elsewhere and the prior is trained AWAY from blocking exactly where blocking is
forced. The order asked for the census over the self-play shards with search stats (1-in-8);
that sample has no producer (CARD-SELFPLAY-SEARCH-STATS: the shards carry the move list, the
result and — from `af47a8ab`, i.e. segment 7 onward — `move_sims`/`move_arms`, and no
per-position search stats), so the census reads the replay RINGS instead: every
`run7_<step>_<hash>.ckpt.ring.bin` in the mirror holds the last 100 000 training rows before
that checkpoint with the stones, the side to move, the stones remaining this turn, the arm,
the SPARSE target (the 16 visited root children with their improved-policy masses) and the
tail mass α. Seven run7 rings (3000, 6000, 9000, 15000, 18000, 21000, 23829; the 12000 ring is
not in the mirror, and a 24000 ring landed while this ran and is shown for completeness —
88.7 % of its rows are games the 23829 ring already holds) and eight run6 rings (3000, 8000,
12000, 14000, 18000, 25000, 30000, 35000), 1.5 M rows, every row read. No box time, no net
loaded. The pre-stated reading: a forced-move miss rate above ~10 % in the full arm, rising
with step, confirms (b); below 2 % refutes it and the search moves to the target's sharpness
alone.

## Verdict

**Refuted, on every ring, in both arms, at every step.** The forced move is inside the
target's explicit support — one of the 16 Gumbel-sampled, Sequential-Halving-visited root
children — on **194 985 of 194 995** forced rows across run7's seven distinct rings (**miss
0.005 %**; per ring 0.00–0.01 %; full arm 0.00–0.02 %, quick arm 0.00–0.01 %) and on
216 067 of 216 092 in run6's eight (**0.012 %**). Nothing rises with step: run7's full arm
reads 0.00, 0.01, 0.00, 0.00, 0.00, 0.00, 0.02 % from 3k to 23.8k, and run6's reads 0.00 %
at 25k, 30k and 35k, the steps of its decline. The refute threshold is 2 %; the record sits
two orders of magnitude under it. The mechanism (b) names — the sampler never seeing the
block — does not occur: with `MAX_ROOT_CHILDREN = u16::MAX` the Gumbel root expands the full
legal set (median 505–675 cells in these rings, not ≈ 355) and the prior puts the forced cell
in the top 16 draws essentially always.

What the target then does with the block is the second reading the ruling named, and the
record answers it: the target is a one-hot. Its explicit mass on the forced set has median
1.000 and mean 0.96–0.98 on every ring; 77–78 % of ALL rows carry one explicit cell above
0.9; the entropy of the explicit masses has median 0.002–0.003 nats (mean 0.18–0.20, the mean
carried by the ~22 % of rows that are not one-hot). The residue the census does find is not a
sampling miss but a search verdict: on **2.5–4.4 % of forced-BLOCK rows** the block was among
the 16 searched candidates and the one-hot went to a NON-blocking cell (`mass(F) < 0.1`;
run7 k = 1 rows, where the block is unambiguous, 2.5–4.0 %; the full 320-sim arm is no better
than the quick 64-sim arm — 3.3–4.3 % vs 2.1–4.0 %). Of 435 such k = 1 rows in the 23829
ring, the one-hot cell makes a mover four on 151 and a mover five on 16 (a counter-threat that
loses to the opponent's completion under two-stone turns) and is hex-adjacent to a threat cell
on 167. WIN rows are essentially perfect (0.00–0.04 % off the win). So the prior is trained
toward a losing move on ~3 % of forced blocks by the completed-Q argmax itself, with the block
in hand, at 64 and at 320 sims alike; that is the target's sharpness, not the sampler's reach.

Two corrections to the brief this census had to make: (i) run6 was NOT a PUCT self-play run —
its `resolved_config.yaml` says `search.kind: gumbel`, `gumbel_m: 16`, and every run6 ring
from step 1 000 on has 16 visit slots (a PUCT ring would carry 327) — so no ring in the
mirror holds a visit-distribution target and the PUCT-vs-Gumbel comparison of step 8 is not
available from the record; the run6 rows are the same head at an earlier run, and they read
the same. (ii) quick-arm rows ARE pushed: `is_full_search = 1` on 24.7–25.3 % of the rows of
every ring in both runs (`full_search_prob 0.25`), contrary to INVESTIGATION1_TROUGH's
"every row is `full_search`".

## Definitions

Per ring row (a search-root position with its stored target). Windows are the 6-cell segments
along the three axial axes (1,0), (0,1), (1,−1); `mantis._engine.HEX_AXES`.

- **four** of P: a window with ≥ 4 P stones and no opponent stone (a five is a four with one
  empty). Its empty set E has 1 or 2 cells; the opponent completes it next turn with ≤ 2 stones.
- **k**: the mover's stones remaining this turn, read from the row's `moves_remaining`
  (k = 2: first stone of the turn; k = 1: second stone, the first already on the board).
- **W1**: cells completing six for the mover now (a mover window with ≥ 5 stones, its empties).
  **W2**: cells starting a two-stone completion (a mover window with exactly 4 stones and 2
  empties; both empties). **W** = W1 ∪ W2 at k = 2, W1 at k = 1.
- **CHECK**: ≥ 1 opponent four. **B(k)** (the ruling's ACCEPTABLE block set, LITERAL): cells c
  such that the opponent fours NOT hit by c have a hitting set of size ≤ k − 1. At k = 1 this
  is the common cell of all fours. At k = 2 a cell hitting nothing qualifies whenever the fours
  already share a cell — then every legal cell is in B (44–60 % of k = 2 BLOCK rows; the target
  is scored as in-support with mass 1 − α). **B_strict(k = 2)** additionally requires c to hit
  ≥ 1 four; reported separately. **LOST1**: B(k) empty (the unanswerable double threat);
  excluded from the miss rate and counted.
- **F** = W if W is non-empty else B (a win beats a block). **FORCED** row: F non-empty.
  **WIN∧CHECK**: rows in both classes, scored as WIN.
- **support**: the row's explicit cells — the root children with ≥ 1 visit, which under
  Sequential Halving with m = 16 are the 16 Gumbel-Top-k draws (`n_visits = 16` on every row).
  **MISS**: F ∩ support = ∅. **PARTIAL**: F ∩ support ≠ ∅ and the explicit mass on F < 0.5.
  **mass(F) < 0.1**: the one-hot went elsewhere. **α**: the row's tail mass — the target mass
  on the unvisited legal cells, which the trainer spreads as α · prior; a missed forced cell can
  receive at most α.
- **arm**: `is_full_search` — full = 320 sims, quick = 64 sims (`n_sims_full`/`n_sims_quick`,
  drawn per move at `full_search_prob 0.25`; `fast_prob = 0`, so there is no fast-game arm and
  the ruling's "full / fast" is read as full / quick).
- **|legal|**: empty cells within hex distance 8 of any stone (`gnn_axis_r8`,
  `legal_move_radius = 8`; 25 on the empty board).
- **H(explicit)**: entropy of the 16 explicit masses renormalised by 1 − α. This is not the
  trough record's H(target), which the trainer rebuilt with the tail spread over the net's own
  prior; the two are not comparable to the decimal.

Validation of the tactics instrument: against the engine on 5 000 ring rows (3 000 from the
3000 ring, seed 7; 2 000 from the 23829 ring, seed 11) replayed into
`Board.with_encoding_name("gnn_axis_r8")` — 0 disagreements on side/k, legal-set size,
`winning_moves(mover)` = W1, `winning_moves(opp)` = opponent fives, `threat_moves(mover)` = W2;
`forced_win_move(2)` disagreed on 14 rows, every one a gapped five the engine's own
`winning_moves` reports but its `has_player_long_run(_, 5)` pre-gate skips (an engine-side
narrowing, not a tactics difference). 17 hand-built cases (a four with two empties, a five, a
capped five, a mover five and four, two disjoint fives = LOST1 at k = 1 and blockable at k = 2,
two overlapping fives hit by one cell, win-beats-block, the legal-set sizes) all pass
(`tactics.py` self-test). The ring parser was checked against the bridge's own loader
(`HexgBuffer.load_from_path`: same 100 000 rows, 0 `game_id_at` mismatches on 2 000 sampled
rows, explicit + α = 1 within 2e-5 on every row, the file consumed to its last byte).

## run7 — base rates (n = rows per ring)


| step | rows | full-arm % | k=1 % | FORCED % | WIN % | CHECK % | BLOCK-forced % | LOST1 % | WIN∧CHECK % | median |legal| | mean fours/CHECK row |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3000 | 100000 | 25.1 | 50.9 | 29.88 | 3.63 | 31.29 | 26.24 | 2.78 | 2.27 | 505 | 2.09 |
| 6000 | 100000 | 25.3 | 50.7 | 29.15 | 2.89 | 30.43 | 26.26 | 2.25 | 1.91 | 539 | 1.93 |
| 9000 | 100000 | 24.9 | 50.7 | 27.32 | 2.79 | 28.39 | 24.53 | 2.09 | 1.77 | 621 | 1.93 |
| 15000 | 100000 | 24.8 | 50.7 | 26.79 | 2.91 | 28.18 | 23.88 | 2.31 | 1.99 | 546 | 1.95 |
| 18000 | 100000 | 25.1 | 50.7 | 28.05 | 2.68 | 29.41 | 25.37 | 2.25 | 1.79 | 541 | 1.90 |
| 21000 | 100000 | 25.1 | 50.7 | 27.59 | 2.79 | 29.05 | 24.80 | 2.40 | 1.85 | 517 | 1.92 |
| 23829 | 100000 | 24.9 | 50.7 | 26.21 | 2.78 | 27.71 | 23.43 | 2.37 | 1.91 | 512 | 1.94 |
| 24000 | 100000 | 24.8 | 50.7 | 26.33 | 2.82 | 27.84 | 23.51 | 2.37 | 1.96 | 512 | 1.95 |


## run7 — the forced-move miss rate (n = forced rows; α columns are over the MISSED rows only)


| step | n forced | MISS % | PARTIAL % | mass(F)<0.1 % | mass(F) mean | mass(F) median | α(missed) mean | α(missed) median | α(missed) p90 |
|---|---|---|---|---|---|---|---|---|---|
| 3000 | 29877 | 0.00 | 3.71 | 3.45 | 0.962 | 1.000 | nan | nan | nan |
| 6000 | 29153 | 0.00 | 2.88 | 2.69 | 0.971 | 1.000 | 0.0000 | 6.0e-35 | 6.0e-35 |
| 9000 | 27321 | 0.01 | 2.92 | 2.76 | 0.970 | 1.000 | 0.0000 | 5.7e-22 | 1.0e-21 |
| 15000 | 26794 | 0.01 | 3.04 | 2.81 | 0.969 | 1.000 | 0.6667 | 1.0e+00 | 1.0e+00 |
| 18000 | 28053 | 0.00 | 2.48 | 2.25 | 0.975 | 1.000 | 0.0000 | 8.7e-11 | 8.7e-11 |
| 21000 | 27591 | 0.00 | 2.58 | 2.30 | 0.974 | 1.000 | nan | nan | nan |
| 23829 | 26206 | 0.01 | 2.82 | 2.58 | 0.971 | 1.000 | 0.0000 | 8.7e-25 | 8.9e-23 |
| 24000 | 26331 | 0.01 | 2.90 | 2.68 | 0.971 | 1.000 | 0.0000 | 8.7e-25 | 8.9e-23 |

## run7 — by arm

| step | arm | n forced | MISS % | PARTIAL % | mass(F)<0.1 % | mass(F) mean | mass(F) median | α(missed) mean | α(missed) median | α(missed) p90 |
|---|---|---|---|---|---|---|---|---|---|---|
| 3000 | full 320 | 7509 | 0.00 | 3.46 | 3.32 | 0.965 | 1.000 | nan | nan | nan |
| 3000 | quick 64 | 22368 | 0.00 | 3.79 | 3.49 | 0.962 | 1.000 | nan | nan | nan |
| 6000 | full 320 | 7321 | 0.01 | 3.81 | 3.69 | 0.962 | 1.000 | 0.0000 | 6.0e-35 | 6.0e-35 |
| 6000 | quick 64 | 21832 | 0.00 | 2.57 | 2.35 | 0.974 | 1.000 | nan | nan | nan |
| 9000 | full 320 | 6742 | 0.00 | 3.95 | 3.83 | 0.960 | 1.000 | nan | nan | nan |
| 9000 | quick 64 | 20579 | 0.01 | 2.58 | 2.41 | 0.974 | 1.000 | 0.0000 | 5.7e-22 | 1.0e-21 |
| 15000 | full 320 | 6648 | 0.00 | 3.97 | 3.81 | 0.960 | 1.000 | nan | nan | nan |
| 15000 | quick 64 | 20146 | 0.01 | 2.74 | 2.49 | 0.973 | 1.000 | 0.6667 | 1.0e+00 | 1.0e+00 |
| 18000 | full 320 | 7007 | 0.00 | 2.70 | 2.58 | 0.973 | 1.000 | nan | nan | nan |
| 18000 | quick 64 | 21046 | 0.00 | 2.41 | 2.14 | 0.976 | 1.000 | 0.0000 | 8.7e-11 | 8.7e-11 |
| 21000 | full 320 | 7034 | 0.00 | 2.63 | 2.42 | 0.974 | 1.000 | nan | nan | nan |
| 21000 | quick 64 | 20557 | 0.00 | 2.56 | 2.26 | 0.974 | 1.000 | nan | nan | nan |
| 23829 | full 320 | 6563 | 0.02 | 3.05 | 2.93 | 0.969 | 1.000 | 0.0000 | 1.1e-22 | 1.1e-22 |
| 23829 | quick 64 | 19643 | 0.01 | 2.74 | 2.47 | 0.972 | 1.000 | 0.0000 | 5.0e-25 | 8.0e-25 |
| 24000 | full 320 | 6544 | 0.02 | 3.18 | 3.07 | 0.968 | 1.000 | 0.0000 | 1.1e-22 | 1.1e-22 |
| 24000 | quick 64 | 19787 | 0.01 | 2.80 | 2.55 | 0.972 | 1.000 | 0.0000 | 5.0e-25 | 8.0e-25 |

## run7 — by k


| step | k | n forced | MISS % | PARTIAL % | mass(F)<0.1 % | mass(F) mean | mass(F) median | α(missed) mean | α(missed) median | α(missed) p90 |
|---|---|---|---|---|---|---|---|---|---|---|
| 3000 | k=2 first stone | 16293 | 0.00 | 3.52 | 3.31 | 0.964 | 1.000 | nan | nan | nan |
| 3000 | k=1 second stone | 13584 | 0.00 | 3.94 | 3.61 | 0.960 | 1.000 | nan | nan | nan |
| 6000 | k=2 first stone | 16146 | 0.01 | 1.91 | 1.74 | 0.981 | 1.000 | 0.0000 | 6.0e-35 | 6.0e-35 |
| 6000 | k=1 second stone | 13007 | 0.00 | 4.07 | 3.86 | 0.959 | 1.000 | nan | nan | nan |
| 9000 | k=2 first stone | 15007 | 0.01 | 1.87 | 1.78 | 0.981 | 1.000 | 0.0000 | 5.7e-22 | 1.0e-21 |
| 9000 | k=1 second stone | 12314 | 0.00 | 4.20 | 3.95 | 0.957 | 1.000 | nan | nan | nan |
| 15000 | k=2 first stone | 14703 | 0.01 | 1.98 | 1.81 | 0.980 | 1.000 | 0.0000 | 5.5e-13 | 5.5e-13 |
| 15000 | k=1 second stone | 12091 | 0.02 | 4.33 | 4.04 | 0.956 | 1.000 | 1.0000 | 1.0e+00 | 1.0e+00 |
| 18000 | k=2 first stone | 15348 | 0.01 | 1.79 | 1.63 | 0.982 | 1.000 | 0.0000 | 8.7e-11 | 8.7e-11 |
| 18000 | k=1 second stone | 12705 | 0.00 | 3.31 | 3.01 | 0.967 | 1.000 | nan | nan | nan |
| 21000 | k=2 first stone | 15098 | 0.00 | 2.42 | 2.18 | 0.976 | 1.000 | nan | nan | nan |
| 21000 | k=1 second stone | 12493 | 0.00 | 2.78 | 2.45 | 0.972 | 1.000 | nan | nan | nan |
| 23829 | k=2 first stone | 14334 | 0.02 | 2.11 | 1.97 | 0.979 | 1.000 | 0.0000 | 8.7e-25 | 8.9e-23 |
| 23829 | k=1 second stone | 11872 | 0.00 | 3.66 | 3.33 | 0.963 | 1.000 | nan | nan | nan |
| 24000 | k=2 first stone | 14388 | 0.02 | 2.21 | 2.06 | 0.978 | 1.000 | 0.0000 | 8.7e-25 | 8.9e-23 |
| 24000 | k=1 second stone | 11943 | 0.00 | 3.73 | 3.42 | 0.962 | 1.000 | nan | nan | nan |

## run7 — by class


| step | class | n forced | MISS % | PARTIAL % | mass(F)<0.1 % | mass(F) mean | mass(F) median | α(missed) mean | α(missed) median | α(missed) p90 |
|---|---|---|---|---|---|---|---|---|---|---|
| 3000 | WIN | 3633 | 0.00 | 0.03 | 0.00 | 0.999 | 1.000 | nan | nan | nan |
| 3000 | BLOCK | 26244 | 0.00 | 4.22 | 3.92 | 0.957 | 1.000 | nan | nan | nan |
| 6000 | WIN | 2893 | 0.00 | 0.00 | 0.00 | 1.000 | 1.000 | nan | nan | nan |
| 6000 | BLOCK | 26260 | 0.00 | 3.19 | 2.98 | 0.968 | 1.000 | 0.0000 | 6.0e-35 | 6.0e-35 |
| 9000 | WIN | 2795 | 0.00 | 0.00 | 0.00 | 1.000 | 1.000 | nan | nan | nan |
| 9000 | BLOCK | 24526 | 0.01 | 3.25 | 3.07 | 0.967 | 1.000 | 0.0000 | 5.7e-22 | 1.0e-21 |
| 15000 | WIN | 2914 | 0.00 | 0.00 | 0.00 | 1.000 | 1.000 | nan | nan | nan |
| 15000 | BLOCK | 23880 | 0.01 | 3.41 | 3.16 | 0.966 | 1.000 | 0.6667 | 1.0e+00 | 1.0e+00 |
| 18000 | WIN | 2681 | 0.00 | 0.04 | 0.04 | 0.999 | 1.000 | nan | nan | nan |
| 18000 | BLOCK | 25372 | 0.00 | 2.74 | 2.49 | 0.972 | 1.000 | 0.0000 | 8.7e-11 | 8.7e-11 |
| 21000 | WIN | 2788 | 0.00 | 0.00 | 0.00 | 1.000 | 1.000 | nan | nan | nan |
| 21000 | BLOCK | 24803 | 0.00 | 2.87 | 2.56 | 0.971 | 1.000 | nan | nan | nan |
| 23829 | WIN | 2779 | 0.00 | 0.04 | 0.04 | 0.999 | 1.000 | nan | nan | nan |
| 23829 | BLOCK | 23427 | 0.01 | 3.15 | 2.89 | 0.968 | 1.000 | 0.0000 | 8.7e-25 | 8.9e-23 |
| 24000 | WIN | 2818 | 0.00 | 0.04 | 0.04 | 0.999 | 1.000 | nan | nan | nan |
| 24000 | BLOCK | 23513 | 0.01 | 3.24 | 2.99 | 0.967 | 1.000 | 0.0000 | 8.7e-25 | 8.9e-23 |

## run7 — by arm × class


| step | arm | class | n forced | MISS % | PARTIAL % | mass(F)<0.1 % | mass(F) mean | mass(F) median | α(missed) mean | α(missed) median | α(missed) p90 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3000 | full | WIN | 931 | 0.00 | 0.11 | 0.00 | 0.998 | 1.000 | nan | nan | nan |
| 3000 | full | BLOCK | 6578 | 0.00 | 3.94 | 3.79 | 0.961 | 1.000 | nan | nan | nan |
| 3000 | quick | WIN | 2702 | 0.00 | 0.00 | 0.00 | 1.000 | 1.000 | nan | nan | nan |
| 3000 | quick | BLOCK | 19666 | 0.00 | 4.31 | 3.97 | 0.956 | 1.000 | nan | nan | nan |
| 6000 | full | WIN | 709 | 0.00 | 0.00 | 0.00 | 1.000 | 1.000 | nan | nan | nan |
| 6000 | full | BLOCK | 6612 | 0.02 | 4.22 | 4.08 | 0.958 | 1.000 | 0.0000 | 6.0e-35 | 6.0e-35 |
| 6000 | quick | WIN | 2184 | 0.00 | 0.00 | 0.00 | 1.000 | 1.000 | nan | nan | nan |
| 6000 | quick | BLOCK | 19648 | 0.00 | 2.85 | 2.61 | 0.971 | 1.000 | nan | nan | nan |
| 9000 | full | WIN | 711 | 0.00 | 0.00 | 0.00 | 1.000 | 1.000 | nan | nan | nan |
| 9000 | full | BLOCK | 6031 | 0.00 | 4.41 | 4.28 | 0.956 | 1.000 | nan | nan | nan |
| 9000 | quick | WIN | 2084 | 0.00 | 0.00 | 0.00 | 1.000 | 1.000 | nan | nan | nan |
| 9000 | quick | BLOCK | 18495 | 0.01 | 2.87 | 2.68 | 0.971 | 1.000 | 0.0000 | 5.7e-22 | 1.0e-21 |
| 15000 | full | WIN | 701 | 0.00 | 0.00 | 0.00 | 1.000 | 1.000 | nan | nan | nan |
| 15000 | full | BLOCK | 5947 | 0.00 | 4.44 | 4.25 | 0.956 | 1.000 | nan | nan | nan |
| 15000 | quick | WIN | 2213 | 0.00 | 0.00 | 0.00 | 1.000 | 1.000 | nan | nan | nan |
| 15000 | quick | BLOCK | 17933 | 0.02 | 3.07 | 2.79 | 0.969 | 1.000 | 0.6667 | 1.0e+00 | 1.0e+00 |
| 18000 | full | WIN | 734 | 0.00 | 0.00 | 0.00 | 1.000 | 1.000 | nan | nan | nan |
| 18000 | full | BLOCK | 6273 | 0.00 | 3.01 | 2.89 | 0.970 | 1.000 | nan | nan | nan |
| 18000 | quick | WIN | 1947 | 0.00 | 0.05 | 0.05 | 0.999 | 1.000 | nan | nan | nan |
| 18000 | quick | BLOCK | 19099 | 0.01 | 2.65 | 2.36 | 0.973 | 1.000 | 0.0000 | 8.7e-11 | 8.7e-11 |
| 21000 | full | WIN | 713 | 0.00 | 0.00 | 0.00 | 1.000 | 1.000 | nan | nan | nan |
| 21000 | full | BLOCK | 6321 | 0.00 | 2.93 | 2.69 | 0.971 | 1.000 | nan | nan | nan |
| 21000 | quick | WIN | 2075 | 0.00 | 0.00 | 0.00 | 1.000 | 1.000 | nan | nan | nan |
| 21000 | quick | BLOCK | 18482 | 0.00 | 2.85 | 2.52 | 0.971 | 1.000 | nan | nan | nan |
| 23829 | full | WIN | 686 | 0.00 | 0.15 | 0.15 | 0.998 | 1.000 | nan | nan | nan |
| 23829 | full | BLOCK | 5877 | 0.02 | 3.39 | 3.25 | 0.966 | 1.000 | 0.0000 | 1.1e-22 | 1.1e-22 |
| 23829 | quick | WIN | 2093 | 0.00 | 0.00 | 0.00 | 1.000 | 1.000 | nan | nan | nan |
| 23829 | quick | BLOCK | 17550 | 0.01 | 3.07 | 2.76 | 0.969 | 1.000 | 0.0000 | 5.0e-25 | 8.0e-25 |
| 24000 | full | WIN | 687 | 0.00 | 0.15 | 0.15 | 0.998 | 1.000 | nan | nan | nan |
| 24000 | full | BLOCK | 5857 | 0.02 | 3.53 | 3.41 | 0.964 | 1.000 | 0.0000 | 1.1e-22 | 1.1e-22 |
| 24000 | quick | WIN | 2131 | 0.00 | 0.00 | 0.00 | 0.999 | 1.000 | nan | nan | nan |
| 24000 | quick | BLOCK | 17656 | 0.01 | 3.14 | 2.85 | 0.968 | 1.000 | 0.0000 | 5.0e-25 | 8.0e-25 |

## run7 — k = 2 BLOCK rows under the STRICT rule (first stone must hit a four; n = k = 2 BLOCK rows)


| step | strict LOST1 among k=2 CHECK rows | n forced | MISS % | PARTIAL % | mass(F)<0.1 % | mass(F) mean | mass(F) median | α(missed) mean | α(missed) median | α(missed) p90 |
|---|---|---|---|---|---|---|---|---|---|---|
| 3000 | 0.00 % of 14457 | 14457 | 0.14 | 30.93 | 28.28 | 0.689 | 1.000 | 0.0000 | 1.9e-14 | 1.2e-11 |
| 6000 | 0.00 % of 14689 | 14689 | 0.25 | 37.10 | 34.07 | 0.626 | 1.000 | 0.0000 | 1.7e-14 | 1.4e-08 |
| 9000 | 0.00 % of 13598 | 13598 | 0.28 | 38.26 | 35.43 | 0.614 | 1.000 | 0.0000 | 3.3e-17 | 3.5e-09 |
| 15000 | 0.00 % of 13223 | 13223 | 0.37 | 40.24 | 37.56 | 0.594 | 1.000 | 0.0003 | 1.8e-14 | 1.2e-07 |
| 18000 | 0.00 % of 13984 | 13984 | 0.33 | 42.38 | 39.47 | 0.573 | 0.999 | 0.0000 | 3.5e-17 | 2.9e-10 |
| 21000 | 0.00 % of 13676 | 13676 | 0.29 | 42.85 | 39.78 | 0.569 | 0.999 | 0.0000 | 9.7e-17 | 3.6e-11 |
| 23829 | 0.00 % of 12911 | 12911 | 0.38 | 41.89 | 38.92 | 0.578 | 1.000 | 0.0204 | 1.1e-16 | 5.9e-08 |
| 24000 | 0.00 % of 12948 | 12948 | 0.37 | 41.54 | 38.53 | 0.582 | 1.000 | 0.0000 | 6.5e-17 | 3.8e-08 |

## run7 — by |legal| bucket (rings pooled)


| |legal| | n forced | MISS % | PARTIAL % | mass(F)<0.1 % | mass(F) mean | mass(F) median | α(missed) mean | α(missed) median | α(missed) p90 |
|---|---|---|---|---|---|---|---|---|---|
| 1–100 | 0 | – | – | – | – | – | – | – | – |
| 101–200 | 0 | – | – | – | – | – | – | – | – |
| 201–300 | 0 | – | – | – | – | – | – | – | – |
| 301–400 | 4514 | 0.00 | 2.50 | 2.28 | 0.975 | 1.000 | nan | nan | nan |
| 401–∞ | 216812 | 0.01 | 2.93 | 2.71 | 0.970 | 1.000 | 0.1538 | 1.1e-22 | 8.0e-01 |

## run7 — the target's sharpness (n = all rows per ring)


| step | H(explicit) mean | H(explicit) median | rows with one cell > 0.9 % | α mean | α median | α p90 | α = 1 per 1000 |
|---|---|---|---|---|---|---|---|
| 3000 | 0.181 | 0.003 | 77.9 | 0.0091 | 3.6e-13 | 1.5e-05 | 4.2 |
| 6000 | 0.182 | 0.003 | 78.0 | 0.0033 | 5.9e-13 | 1.3e-05 | 1.2 |
| 9000 | 0.187 | 0.003 | 77.7 | 0.0041 | 4.5e-13 | 1.1e-05 | 1.8 |
| 15000 | 0.199 | 0.002 | 77.1 | 0.0054 | 2.5e-13 | 1.1e-05 | 2.3 |
| 18000 | 0.201 | 0.002 | 77.2 | 0.0059 | 2.3e-13 | 1.5e-05 | 2.4 |
| 21000 | 0.193 | 0.002 | 77.4 | 0.0068 | 1.5e-13 | 1.4e-05 | 3.0 |
| 23829 | 0.190 | 0.002 | 77.6 | 0.0066 | 1.6e-13 | 1.3e-05 | 2.7 |
| 24000 | 0.190 | 0.002 | 77.6 | 0.0063 | 1.7e-13 | 1.4e-05 | 2.6 |

#### Sharpness on FORCED rows vs QUIET rows, pooled (run7)

| rows | n | H(explicit) mean | one cell > 0.9 % | mass(F) mean |
|---|---|---|---|---|
| forced | 221326 | 0.173 | 77.8 | 0.970 |
| quiet | 559857 | 0.197 | 77.7 | – |
| lost1 | 18817 | 0.201 | 71.1 | – |

## The ten run7 misses and where the k = 1 one-hots go

Every run7 miss (3000: 0; 6000: 1; 9000: 2; 15000: 3; 18000: 1; 21000: 0; 23829: 3; the
24000 ring's 3 are the same three rows) is a BLOCK row; 8 are k = 2 rows facing 3–4 fours with
a 2–4-cell B and α between 1e-35 and 1e-10 (the missed cells' target mass is ≤ α ≈ 0), and 2
are k = 1 rows with α = 1 — the all-tail row the loss excludes (R350(e)), where the explicit
support carries no mass at all. run6's 25 misses have the same shape (22 k = 2, 3 α = 1).

On the 23829 ring's 435 k = 1 BLOCK rows with mass(F) < 0.5 (the block searched, the one-hot
elsewhere): the one-hot cell makes a mover four on 151, a mover five on 16, and is hex-adjacent
to an opponent threat cell on 167; 0 are zero-mass rows. run6 35000: 301 rows — 64 make a
four, 15 a five, 187 adjacent, 1 zero-mass.

## run6 — the same head, the earlier run (n as marked)


| step | rows | full-arm % | k=1 % | FORCED % | WIN % | CHECK % | BLOCK-forced % | LOST1 % | WIN∧CHECK % | median |legal| | mean fours/CHECK row |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3000 | 100000 | 24.8 | 50.9 | 32.80 | 3.53 | 34.53 | 29.27 | 2.99 | 2.27 | 498 | 2.01 |
| 8000 | 100000 | 24.8 | 50.7 | 26.91 | 2.73 | 28.72 | 24.18 | 2.44 | 2.10 | 599 | 1.97 |
| 12000 | 100000 | 25.0 | 50.6 | 26.21 | 2.51 | 27.82 | 23.70 | 2.23 | 1.89 | 675 | 1.96 |
| 14000 | 100000 | 25.0 | 50.6 | 26.02 | 2.60 | 27.66 | 23.42 | 2.35 | 1.89 | 634 | 1.96 |
| 18000 | 100000 | 24.9 | 50.7 | 27.81 | 2.91 | 29.54 | 24.91 | 2.51 | 2.11 | 516 | 1.95 |
| 25000 | 100000 | 25.2 | 50.6 | 25.80 | 2.51 | 26.94 | 23.29 | 2.17 | 1.48 | 523 | 1.85 |
| 30000 | 100000 | 24.7 | 50.6 | 25.80 | 2.47 | 27.09 | 23.34 | 2.26 | 1.50 | 524 | 1.86 |
| 35000 | 100000 | 25.1 | 50.6 | 24.75 | 2.44 | 25.93 | 22.31 | 2.18 | 1.44 | 526 | 1.86 |

#### Forced-move miss rate per ring, all forced rows (run6)

| step | n forced | MISS % | PARTIAL % | mass(F)<0.1 % | mass(F) mean | mass(F) median | α(missed) mean | α(missed) median | α(missed) p90 |
|---|---|---|---|---|---|---|---|---|---|
| 3000 | 32797 | 0.00 | 2.27 | 1.99 | 0.977 | 1.000 | 1.0000 | 1.0e+00 | 1.0e+00 |
| 8000 | 26905 | 0.02 | 2.31 | 2.09 | 0.976 | 1.000 | 0.2000 | 3.9e-18 | 6.0e-01 |
| 12000 | 26205 | 0.03 | 2.48 | 2.29 | 0.975 | 1.000 | 0.0000 | 5.6e-18 | 7.3e-14 |
| 14000 | 26021 | 0.03 | 2.38 | 2.11 | 0.976 | 1.000 | 0.1429 | 2.1e-17 | 4.0e-01 |
| 18000 | 27814 | 0.01 | 2.74 | 2.50 | 0.972 | 1.000 | 0.0000 | 2.8e-15 | 7.5e-13 |
| 25000 | 25797 | 0.00 | 2.75 | 2.57 | 0.972 | 1.000 | 0.0000 | 1.2e-09 | 1.2e-09 |
| 30000 | 25803 | 0.00 | 2.43 | 2.21 | 0.975 | 1.000 | nan | nan | nan |
| 35000 | 24750 | 0.00 | 2.63 | 2.40 | 0.973 | 1.000 | nan | nan | nan |

#### By arm (run6)

| step | arm | n forced | MISS % | PARTIAL % | mass(F)<0.1 % | mass(F) mean | mass(F) median | α(missed) mean | α(missed) median | α(missed) p90 |
|---|---|---|---|---|---|---|---|---|---|---|
| 3000 | full 320 | 8113 | 0.01 | 1.42 | 1.33 | 0.986 | 1.000 | 1.0000 | 1.0e+00 | 1.0e+00 |
| 3000 | quick 64 | 24684 | 0.00 | 2.56 | 2.21 | 0.974 | 1.000 | nan | nan | nan |
| 8000 | full 320 | 6727 | 0.00 | 2.01 | 1.86 | 0.980 | 1.000 | nan | nan | nan |
| 8000 | quick 64 | 20178 | 0.02 | 2.41 | 2.17 | 0.975 | 1.000 | 0.2000 | 3.9e-18 | 6.0e-01 |
| 12000 | full 320 | 6522 | 0.00 | 2.21 | 2.12 | 0.978 | 1.000 | nan | nan | nan |
| 12000 | quick 64 | 19683 | 0.04 | 2.57 | 2.35 | 0.973 | 1.000 | 0.0000 | 5.6e-18 | 7.3e-14 |
| 14000 | full 320 | 6467 | 0.03 | 2.15 | 1.99 | 0.978 | 1.000 | 0.5000 | 5.0e-01 | 9.0e-01 |
| 14000 | quick 64 | 19554 | 0.03 | 2.45 | 2.15 | 0.975 | 1.000 | 0.0000 | 2.1e-17 | 2.8e-11 |
| 18000 | full 320 | 6947 | 0.00 | 2.61 | 2.46 | 0.974 | 1.000 | nan | nan | nan |
| 18000 | quick 64 | 20867 | 0.01 | 2.78 | 2.52 | 0.971 | 1.000 | 0.0000 | 2.8e-15 | 7.5e-13 |
| 25000 | full 320 | 6461 | 0.00 | 2.46 | 2.35 | 0.975 | 1.000 | nan | nan | nan |
| 25000 | quick 64 | 19336 | 0.01 | 2.84 | 2.64 | 0.971 | 1.000 | 0.0000 | 1.2e-09 | 1.2e-09 |
| 30000 | full 320 | 6402 | 0.00 | 2.34 | 2.16 | 0.977 | 1.000 | nan | nan | nan |
| 30000 | quick 64 | 19401 | 0.00 | 2.45 | 2.23 | 0.975 | 1.000 | nan | nan | nan |
| 35000 | full 320 | 6156 | 0.00 | 2.57 | 2.42 | 0.974 | 1.000 | nan | nan | nan |
| 35000 | quick 64 | 18594 | 0.00 | 2.66 | 2.39 | 0.973 | 1.000 | nan | nan | nan |

#### By k (run6)

| step | k | n forced | MISS % | PARTIAL % | mass(F)<0.1 % | mass(F) mean | mass(F) median | α(missed) mean | α(missed) median | α(missed) p90 |
|---|---|---|---|---|---|---|---|---|---|---|
| 3000 | k=2 first stone | 18065 | 0.00 | 2.97 | 2.70 | 0.970 | 1.000 | nan | nan | nan |
| 3000 | k=1 second stone | 14732 | 0.01 | 1.43 | 1.13 | 0.985 | 1.000 | 1.0000 | 1.0e+00 | 1.0e+00 |
| 8000 | k=2 first stone | 14888 | 0.03 | 2.14 | 1.96 | 0.978 | 1.000 | 0.0000 | 2.0e-18 | 6.7e-17 |
| 8000 | k=1 second stone | 12017 | 0.01 | 2.52 | 2.25 | 0.974 | 1.000 | 1.0000 | 1.0e+00 | 1.0e+00 |
| 12000 | k=2 first stone | 14480 | 0.06 | 2.24 | 2.09 | 0.977 | 1.000 | 0.0000 | 5.6e-18 | 7.3e-14 |
| 12000 | k=1 second stone | 11725 | 0.00 | 2.77 | 2.54 | 0.971 | 1.000 | nan | nan | nan |
| 14000 | k=2 first stone | 14248 | 0.04 | 2.01 | 1.83 | 0.979 | 1.000 | 0.0000 | 1.0e-17 | 2.3e-11 |
| 14000 | k=1 second stone | 11773 | 0.01 | 2.82 | 2.45 | 0.971 | 1.000 | 1.0000 | 1.0e+00 | 1.0e+00 |
| 18000 | k=2 first stone | 15200 | 0.02 | 2.61 | 2.43 | 0.973 | 1.000 | 0.0000 | 2.8e-15 | 7.5e-13 |
| 18000 | k=1 second stone | 12614 | 0.00 | 2.90 | 2.59 | 0.970 | 1.000 | nan | nan | nan |
| 25000 | k=2 first stone | 14119 | 0.01 | 2.68 | 2.54 | 0.973 | 1.000 | 0.0000 | 1.2e-09 | 1.2e-09 |
| 25000 | k=1 second stone | 11678 | 0.00 | 2.83 | 2.60 | 0.970 | 1.000 | nan | nan | nan |
| 30000 | k=2 first stone | 14187 | 0.00 | 2.21 | 2.07 | 0.978 | 1.000 | nan | nan | nan |
| 30000 | k=1 second stone | 11616 | 0.00 | 2.69 | 2.38 | 0.972 | 1.000 | nan | nan | nan |
| 35000 | k=2 first stone | 13638 | 0.00 | 2.57 | 2.40 | 0.974 | 1.000 | nan | nan | nan |
| 35000 | k=1 second stone | 11112 | 0.00 | 2.72 | 2.40 | 0.972 | 1.000 | nan | nan | nan |

#### By class (run6)

| step | class | n forced | MISS % | PARTIAL % | mass(F)<0.1 % | mass(F) mean | mass(F) median | α(missed) mean | α(missed) median | α(missed) p90 |
|---|---|---|---|---|---|---|---|---|---|---|
| 3000 | WIN | 3531 | 0.00 | 0.00 | 0.00 | 1.000 | 1.000 | nan | nan | nan |
| 3000 | BLOCK | 29266 | 0.00 | 2.55 | 2.23 | 0.974 | 1.000 | 1.0000 | 1.0e+00 | 1.0e+00 |
| 8000 | WIN | 2727 | 0.00 | 0.00 | 0.00 | 1.000 | 1.000 | nan | nan | nan |
| 8000 | BLOCK | 24178 | 0.02 | 2.57 | 2.32 | 0.974 | 1.000 | 0.2000 | 3.9e-18 | 6.0e-01 |
| 12000 | WIN | 2508 | 0.00 | 0.00 | 0.00 | 1.000 | 1.000 | nan | nan | nan |
| 12000 | BLOCK | 23697 | 0.03 | 2.74 | 2.53 | 0.972 | 1.000 | 0.0000 | 5.6e-18 | 7.3e-14 |
| 14000 | WIN | 2603 | 0.00 | 0.00 | 0.00 | 1.000 | 1.000 | nan | nan | nan |
| 14000 | BLOCK | 23418 | 0.03 | 2.64 | 2.35 | 0.973 | 1.000 | 0.1429 | 2.1e-17 | 4.0e-01 |
| 18000 | WIN | 2907 | 0.00 | 0.10 | 0.10 | 0.999 | 1.000 | nan | nan | nan |
| 18000 | BLOCK | 24907 | 0.01 | 3.05 | 2.78 | 0.969 | 1.000 | 0.0000 | 2.8e-15 | 7.5e-13 |
| 25000 | WIN | 2506 | 0.00 | 0.00 | 0.00 | 1.000 | 1.000 | nan | nan | nan |
| 25000 | BLOCK | 23291 | 0.00 | 3.04 | 2.85 | 0.969 | 1.000 | 0.0000 | 1.2e-09 | 1.2e-09 |
| 30000 | WIN | 2468 | 0.00 | 0.04 | 0.00 | 0.999 | 1.000 | nan | nan | nan |
| 30000 | BLOCK | 23335 | 0.00 | 2.68 | 2.44 | 0.972 | 1.000 | nan | nan | nan |
| 35000 | WIN | 2440 | 0.00 | 0.00 | 0.00 | 1.000 | 1.000 | nan | nan | nan |
| 35000 | BLOCK | 22310 | 0.00 | 2.92 | 2.66 | 0.970 | 1.000 | nan | nan | nan |

#### k=2 BLOCK rows, STRICT variant (first stone must itself hit a four) (run6)

| step | strict LOST1 among k=2 CHECK rows | n forced | MISS % | PARTIAL % | mass(F)<0.1 % | mass(F) mean | mass(F) median | α(missed) mean | α(missed) median | α(missed) p90 |
|---|---|---|---|---|---|---|---|---|---|---|
| 3000 | 0.00 % of 16281 | 16281 | 0.27 | 38.06 | 35.48 | 0.615 | 1.000 | 0.0000 | 1.8e-15 | 3.3e-09 |
| 8000 | 0.00 % of 13494 | 13494 | 0.49 | 36.43 | 33.35 | 0.632 | 1.000 | 0.0000 | 8.7e-16 | 8.9e-08 |
| 12000 | 0.00 % of 13198 | 13198 | 0.40 | 37.13 | 34.40 | 0.625 | 1.000 | 0.0000 | 1.6e-16 | 8.4e-11 |
| 14000 | 0.00 % of 12918 | 12918 | 0.32 | 38.44 | 35.75 | 0.613 | 1.000 | 0.0007 | 2.1e-16 | 1.8e-08 |
| 18000 | 0.00 % of 13715 | 13715 | 0.28 | 41.34 | 38.66 | 0.583 | 1.000 | 0.0004 | 2.9e-13 | 1.1e-08 |
| 25000 | 0.00 % of 12844 | 12844 | 0.47 | 47.13 | 44.05 | 0.525 | 0.824 | 0.0000 | 4.2e-16 | 4.7e-07 |
| 30000 | 0.00 % of 12919 | 12919 | 0.22 | 46.85 | 43.88 | 0.529 | 0.878 | 0.0000 | 7.8e-17 | 1.3e-10 |
| 35000 | 0.00 % of 12386 | 12386 | 0.21 | 46.48 | 43.40 | 0.534 | 0.942 | 0.0000 | 3.9e-16 | 1.7e-10 |

#### Target sharpness per ring (run6; n = all rows)

| step | H(explicit) mean | H(explicit) median | rows with one cell > 0.9 % | α mean | α median | α p90 | α = 1 per 1000 |
|---|---|---|---|---|---|---|---|
| 3000 | 0.196 | 0.005 | 75.9 | 0.0080 | 4.6e-13 | 2.5e-05 | 3.9 |
| 8000 | 0.187 | 0.003 | 77.2 | 0.0077 | 4.0e-13 | 2.2e-05 | 3.9 |
| 12000 | 0.202 | 0.005 | 76.3 | 0.0073 | 3.6e-13 | 2.0e-05 | 3.6 |
| 14000 | 0.196 | 0.003 | 76.8 | 0.0072 | 3.3e-13 | 2.1e-05 | 3.5 |
| 18000 | 0.197 | 0.002 | 76.9 | 0.0081 | 2.6e-13 | 2.1e-05 | 3.2 |
| 25000 | 0.203 | 0.002 | 76.9 | 0.0078 | 9.4e-14 | 1.6e-05 | 3.4 |
| 30000 | 0.208 | 0.002 | 76.4 | 0.0091 | 7.9e-14 | 1.6e-05 | 5.0 |
| 35000 | 0.198 | 0.002 | 77.1 | 0.0086 | 7.6e-14 | 1.3e-05 | 4.5 |

#### Sharpness on FORCED rows vs QUIET rows, pooled (run6)

| rows | n | H(explicit) mean | one cell > 0.9 % | mass(F) mean |
|---|---|---|---|---|
| forced | 216092 | 0.184 | 76.3 | 0.974 |
| quiet | 564775 | 0.201 | 77.2 | – |
| lost1 | 19133 | 0.282 | 64.4 | – |

## What the record cannot answer

- **The σ-pair arm** ((rescale, 1.0) as run, (raw, 1.0), (rescale, 0.1)): NOT answerable. The
  HEXG v2 row stores, per explicit cell, the improved-policy MASS
  softmax(log π + σ(completed Q)) as one f32, and one scalar α; it stores no visit counts, no
  Q values, no raw root value and no prior (`persist.rs` layout: `n_stones, n_visits,
  current_player, moves_remaining, ply_index, is_full_search, value_valid, outcome, game_length,
  game_id, weight, tail_mass, stones × (q, r, p), visits × (q, r, prob)`). A second σ cannot be
  recomputed from a softmax output. The eval-channel shards carry root visits and a root value
  but not the completed Q, and they are the deploy head, not the self-play head.
- **The 1-in-8 self-play search-stats sample**: no producer. run7's 24 189 self-play games in
  the mirror carry moves and result; the 245 games of segment 7 (post-resume, `af47a8ab`
  landed) add `move_sims`/`move_arms` per ply and still no per-position stats. So the arm split
  above is the RING's `is_full_search`, not the shards'.
- **prior(F) on the missed rows**: not run — no net was loaded. With 10 misses in run7 the
  quantity is moot; the bound the record gives is target mass on a missed cell ≤ α, and α is
  ≤ 1e-10 on 8 of the 10 (the other 2 are α = 1 rows the loss excludes).
- **A PUCT visit-distribution target for comparison**: no ring in the mirror holds one (both
  runs are Gumbel m = 16). The comparison stands only across the eval-channel game records of
  the earlier census, which are the deploy head at 128–512 sims.
- **The 12000 run7 ring**: only its `.ckpt` is in the mirror; the step window 9000–15000 is
  read from those two rings.
- **k = 2 "forced"** is weaker than k = 1 by construction (a first stone need not block when
  the fours share a hitting cell); the strict rule is reported beside it and the k = 1 rows,
  where the block is unambiguous, carry the verdict on their own.
- What this census does not separate: whether the ~3 % non-block one-hots are the value head
  mis-ranking a searched block (completed-Q with a wrong v) or the search's own shallow reading;
  both are "the target's sharpness", neither is (b).

## Sources

Scripts (this session's scratchpad `census/`, stdlib + numpy; the engine only in the
validation): `ring_reader.py` (HEXG v2 parser, LE, validated against
`mantis._engine.HexgBuffer`), `tactics.py` (windows, W1/W2/fours/B/B_strict/LOST1/legal count;
`python tactics.py` runs the 17 hand cases), `validate_engine.py RING N SEED` (the engine
cross-check), `census.py RING OUT_DIR` (one npz of per-row columns per ring), `tables.py
OUT_DIR --run run7|run6` (every table above), `run_all.sh` (the 16-ring chain, ≈ 45 s per ring
on the dev box). Environment: `/home/tom/Work/HeXO/hexo-mantis/.venv/bin/python` at HEAD
`83667687`, read-only.

Commands: `python ring_reader.py <ring>` (header + field histograms);
`python validate_engine.py …run7_00003000_d525bc55.ckpt.ring.bin 3000 7` and
`… run7_00023829_4a302f88.ckpt.ring.bin 2000 11`; `./run_all.sh` → `out/*.npz`;
`python tables.py out --run run7` / `--run run6`.

Mirror files read (`~/Work/HeXO/mantis-mirror/`, every ring HEXG v2, `gnn_axis_r8`,
`max_visits 16`, `capacity 100000`, `size 100000`):
`run7/checkpoints/run7_00003000_d525bc55`, `run7_00006000_f3c1cf05`, `run7_00009000_ce6ff519`,
`run7_00015000_40e13c80`, `run7_00018000_146446a5`, `run7_00021000_f82e3839`,
`run7_00023829_4a302f88`, `run7_00024000_efa82bb2` (`.ckpt.ring.bin`, 100 000 rows each;
1 423 / 1 441 games in the last two, 1 269 shared); `run6/checkpoints/run6_00003000_aa3120ba`,
`run6_00008000_d3168649`, `run6_00012000_5f0d09ac`, `run6_00014000_82773719`,
`run6_00018000_c46fa529`, `run6_00025000_c8bf16a5`, `run6_00030000_3b9e0a5c`,
`run6_00035000_6ab40069` (100 000 rows each); `run7/resolved_config.yaml` and
`run6/resolved_config.yaml` (`search.kind: gumbel`, `gumbel_m: 16`, `full_search_prob 0.25`,
`n_sims_quick 64`, `n_sims_full 320`, `fast_prob 0.0`); `run7/logs/games/` (45 shards: segment
1 = 27 shards, 23 944 self-play games, 0 with `move_arms`; segments 2–6 = 16 shards, 0
self-play games; segment 7 = 1 shard, 245 self-play games, 245 with `move_arms`; segment 8 =
1 shard, 0 self-play). Repo sources for the layout and semantics:
`crates/mantis-selfplay/src/replay/hexg/persist.rs`, `push.rs`, `mod.rs`
(`HEXG_GUMBEL_M_MAX`), `crates/mantis-selfplay/src/records.rs` (`record_position_graph`: the
explicit support + tail), `crates/mantis-selfplay/src/runner/search_drive.rs` (the arm draw,
`visited_root_child_cells` as the support, `get_improved_policy_ls` as the target),
`crates/mantis-search/src/mcts/mod.rs` (`MAX_ROOT_CHILDREN`), `completed_q.rs`,
`gumbel_mctx.rs`, `crates/mantis-core/src/board/moves.rs` (`winning_moves`, `threat_moves`,
`forced_win_move`, the radius-8 legal ball), `crates/mantis-encoding/src/registry.toml`
(`gnn_axis_r8.legal_move_radius = 8`), `docs/contracts/replay_persist.md`,
`docs/governance/CARDS.md` (CARD-SELFPLAY-SEARCH-STATS),
`docs/design/measurements/GAME_QUALITY_CENSUS_2026-09-14.md` (the four/CHECK/LOST1/WIN1
definitions reused), `INVESTIGATION1_TROUGH_2026-09-13.md`.
