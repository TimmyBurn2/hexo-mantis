# MEASUREMENT — A-2 falsifier: the census residue and the quiescence fire rates under the corrected threat unit (2026-09-16)

Ordered by R355(a): "FALSIFIER, offline before any box hour: the residue on one ring under the
fixed backup." Ring: `run7_00023829_4a302f88.ckpt.ring.bin` (100 000 rows). Engines: the parent of
the fix (`30198f05`, the one-stone rule) and the fix (`2865cd3a`), both through the real backup
path (`MCTSTree` root expanded with a uniform prior and NN = 0, one forced descent into the
child, `get_root_children_info` Q in the root's view) — the red team's own probe
(`redteamA/census_rows.py`), over EVERY residue row rather than 20.

## The residue (the census's 435 k = 1 BLOCK rows with mass(F) < 0.5)

Of the 435, the 275 whose one-hot is OUTSIDE the block set B(1) reach the backup (the other 160
are the census's partial-mass rows, the one-hot on a block cell). Every reconstruction succeeded
and every one-hot and block cell was a root child.

| engine | rows through the backup | counter-threat child >= block child |
|---|---|---|
| one-stone rule (`30198f05`) | 275 | **275 (100.0 %)** |
| corrected unit (`2865cd3a`) | 275 | **0 (0.0 %)** |

Reading: the veto now fires on every residue row — the counter-threat child (the opponent to
move with two stones and an open window of two empties) reads −1 from the root, the block child
the net's value. What this does NOT show: whether the ~3 % one-hot residue falls IN-RUN — that is
run8's ring at 15k and 30k (R355(d)'s pre-registered reading); the offline instrument shows only
that the mechanism red team A reproduced is closed.

## Fire rates over 5 000 random rows (the LAW-18 number)

4 439 rows reconstructed (561 skipped: fewer than 8 stones, or a stone set no legal-cadence
interleaving reproduces). Each row expanded alone as a leaf with NN = 0; the verdict read off the
root value and `quiescence_fire_count`.

| engine | +1 | −1 | blend | none |
|---|---|---|---|---|
| one-stone rule | 0 (0.00 %) | 0 (0.00 %) | 73 (1.64 %) | 4 366 (98.36 %) |
| corrected unit, as SHIPPED | 120 (2.70 %) | 121 (2.73 %) | 15 (0.34 %) | 4 183 (94.23 %) |
| corrected unit + "whole turn forced" blend (NOT shipped) | 120 (2.70 %) | 121 (2.73 %) | 294 (6.62 %) | 3 904 (87.95 %) |

Reading: the old rule's ±1 never fired on a ring row (≥ 3 one-stone completions does not occur
at these depths) and its blend fired on 1.6 %, mixing "mover has two win cells" (+0.3, now a +1
proof) with "opponent has two five-cells" (−0.3). The corrected unit proves 5.4 % of leaves
(2.7 % each way — a mover win-in-turn, an unblockable opponent threat set) and blends 0.3 %: two
opponent fives against a two-stone turn, the pre-A-2 rule's one surviving heuristic case.

The third row is the variant the plan first carried (D1): a blend whenever the mover's whole
two-stone turn goes to blocking, which also covers an OPEN FOUR at k = 2. It fires on 6.6 % of
leaves — four times the old rule's whole fire rate — and R355(d) frames A-2 as "correctness, not
a swap", so it is NOT shipped: an open four faced with two stones is left to the net, as before.
The number is here so arming it later is a ruling with its rate known, not a guess.

## The hot-path cost (LAW-09, `mcts_bench`, this host, criterion vs `pre_a2`, 3 s windows)

| bench | change (95 % CI) | verdict |
|---|---|---|
| mcts_sims_cpu_only/100 | −0.17 % [−0.37, +0.04] | no change |
| mcts_sims_cpu_only/400 | +0.22 % [−0.23, +0.84] | no change |
| mcts_sims_cpu_only/800 | −0.69 % [−1.03, −0.39] | noise-level |
| expand_leaf/dense/413 (8 stones) | +4.08 % [+1.59, +6.47] | regressed |
| expand_leaf/ls_graph/413 (8 stones) | +1.53 % [−0.80, +4.08] | no change |
| expand_leaf/dense/1071 (32 stones) | +3.74 % [+1.86, +5.51] | regressed |
| expand_leaf/ls_graph/1071 (32 stones) | +7.70 % [+6.18, +9.15] | regressed |

Reading: the sims groups run from the empty board and the ply gate (`< 7`) short-circuits; the
expand groups pay the scan — one 11-cell line read per stone per axis (≈ 30 lookups a stone) in
place of `has_player_long_run`'s pre-gate — ≈ 3 µs on a 70 µs expansion at 32 stones. On a
production leaf (50–150 stones) the scan is 1.5–4.5 k lookups against a legal-set rebuild of
≈ 200 cells per stone, so the expected in-run cost is a few percent of the expansion and a
smaller share of the leaf's wall (inference dominates). The self-play leaves/s number is read at
run8's shakedown; a cheaper structure (a per-stone axis-neighbour gate) is a separate measured
commit if that number says so.

## Sources
`~/Work/HeXO/repair-a4-2026-09-16/a2_falsifier.py` (the scratchpad copy is volatile);
`~/Work/HeXO/inv1-scratch-2026-09-15/census/{ring_reader,tactics}.py` + `out/run7_00023829_4a302f88.npz`;
`~/Work/HeXO/inv1-scratch-2026-09-15/inv1/redteamA/census_rows.py` (the probe this generalises);
`crates/mantis-search/benches/mcts_bench.rs`; `crates/mantis-core/src/board/threats.rs`;
`crates/mantis-search/src/mcts/backup.rs::apply_quiescence`.
