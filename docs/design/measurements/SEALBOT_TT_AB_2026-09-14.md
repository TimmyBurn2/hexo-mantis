# MEASUREMENT — SEALBOT-TT A/B: the seat-contaminated adapter vs a fresh engine per game (2026-09-14/15)

Ordered by R353(b). The defect (CARD-SEALBOT-TT-SEAT): the vendored engine's transposition table
persists across `get_move` calls, is keyed by position ⊕ side-to-move ⊕ stones-left — not the
root player — while its scores are root-relative, so an adapter that played seat A reads its own
entries with the wrong sign when its next game puts it on seat B. The fix (`748f5c47`, in-repo):
`SealBotAdapter.new_game()` replaces a searched engine, so every game starts from an empty table.
The question the A/B answers: how much did the contamination move the sealbot LEVELS on the
record, so a ratio can re-derive them.

## Design

Three frozen nets, the same book (`book_v1_s20260625_p4`, seed 20260625, 144 paired openings),
288 games per cell, both arms on the box during run7 (the trainer stepping at ≈ 1 400 steps/h
throughout; round 2 overlapped the last two cells): the OLD arm from run7's own tree
`/workspace/hexo-mantis` at `15109ac3` (the adapter every reading on the record went through),
the FIXED arm from the worktree `/workspace/mantis-tt` at `748f5c47` (same driver sha, same Python
3.11, byte-identical `minimax_cpp` build — sha `a65294e4…` in both trees, six book positions
agree move-for-move across the trees). Cells: the BC net (`bc_full`) at PUCT-150, run6's
`ck18k` at PUCT-512, run7's start-stamp net (`run7_00000101_fcedc217`) at PUCT-512; concurrency 8
(the rung's shape since R351). Readout: WR with the record's own pair-bootstrap CI, the paired
delta over the 144 openings with a bootstrap CI, per-pair agreement, identical games, walls.

## A. The rung's shape (concurrency 8)

| cell | old WR [CI] | fixed WR [CI] | delta (fixed − old) [paired CI] | pairs same outcome | plies old / fixed | wall old / fixed (s) |
|---|---|---|---|---|---|---|
| bc_full_puct150 | 0.469 [0.406, 0.524] | 0.462 [0.403, 0.521] | −0.007 [−0.062, +0.049] | 95/144 | 33 / 33 | 1 003 / 1 017 |
| ck18k_puct512 | 0.764 [0.712, 0.809] | 0.792 [0.740, 0.833] | +0.028 [−0.021, +0.080] | 96/144 | 46 / 45 | 5 117 / 5 227 |
| run7s0_puct512 | 0.618 [0.562, 0.670] | 0.604 [0.545, 0.656] | −0.014 [−0.062, +0.038] | 94/144 | 33 / 33 | 3 271 / 3 263 |

Pooled over 864 games: old 0.617, fixed 0.619, **ratio 1.004**; every paired CI includes 0.
Identical games move-for-move: 63 / 71 / 81 of 288. **In every one of the 649 games that differ,
the first move to differ is the CANDIDATE's** (median first-divergence ply 12 / 19 / 9), never
sealbot's: given the same position, the old and the fixed adapter chose the same move in all
864 games. At this shape the contamination changed no sealbot move at all.

Why the shape matters: `play_paired_match` hands the two games of an opening — `(o, +1)` and
`(o, −1)`, consecutive slots — to whichever threads are free, so with eight thread-local pairs
the same adapter almost never plays both games of a pair back to back, and a cross-seat hit then
needs a transposition between games of DIFFERENT random 4-ply openings, which does not happen.
The local reproduction that made the Tier-2 test RED (3 of 4 pairs diverge at the swapped game's
first move) is the SERIAL shape: one adapter, both games of the pair in a row.

## B. The serial shape (concurrency 1) — run6's rounds (`rung_concurrency 1`) and the frontier's 33 cells were taken this way

One cell, the cheapest, both arms at `concurrency 1` (00:25–01:38 UTC, beside run7's round 2):

| cell | old WR [CI] | fixed WR [CI] | delta (fixed − old) [paired CI] | pairs same outcome | plies old / fixed | wall old / fixed (s) |
|---|---|---|---|---|---|---|
| bc_full_puct150_serial | 0.458 [0.403, 0.514] | 0.458 [0.399, 0.517] | **+0.000 [−0.042, +0.045]** | 108/144 | 33 / 33 | 4 378 / 4 264 |

Identical games 98/288; in the 190 that differ the first move to differ is the candidate's in
190, sealbot's in 0 (median first-divergence ply 11.5). The frontier's own serial reading of this
cell (`STRENGTH_FRONTIER_1_2026-09-13.md`, 2026-09-13, an earlier tree) was 0.413 [0.358, 0.472];
both arms here read 0.458 — the same within the CIs, and the same as each other.

**Why the Tier-2 test is RED on the old adapter while the A/B reads zero:** the test's opponent is
a fresh SEALBOT, so game 2's positions are exactly the nodes game 1's search stored (both sides
play the deterministic engine's moves), and the wrong-signed hits land at low remaining depth;
against our nets the opponent's moves are not sealbot's, game 2's tree does not revisit game 1's
nodes, and the table's entries are never read. The defect is real in the engine and inert as a
bias on this instrument's readings. "Sealbot first to differ: 0" is a lower bound (a sealbot
divergence after the candidate's is invisible), which is why the LEVEL, not the move count, is
the reading: Δ 0.000 / −0.007 / +0.028 / −0.014 over 1 152 games, every CI including 0.

## C. TT growth as a cost — tested and refused

- Per-search cost, locally (16-core box, depth 5, 24 colour-alternating games per arm against a
  fresh sealbot opponent): one engine across games 522 ms mean / 102 ms median per search, a
  fresh engine per game 499 / 101 ms; node counts 531k vs 519k mean; both track the position,
  not the number of games the engine has played (per-6-game bands 453 → 569 → 657 → 431 ms
  reused, 429 → 538 → 636 → 430 fresh). The rebuild is 5 ms per game (≈ 42 MB table, the old
  one freed on replacement).
- Cell walls above: old and fixed within 2 % on every cell.
- The 0.20 → 0.30 s/ply drift the ruling names is the GATE block's (`RUN7_STAMP2_2026-09-14.md`
  §E): candidate vs anchor, both deploy heads, no sealbot instance in the process while it runs
  (the phases are sequential: gate, then rung). TT growth cannot explain it by construction; the
  observation stays unexplained and is not this record's.

## D. What the A/B says for the record

1. Every sealbot level on the record stands as read — run7's stamps and rounds (the rung's
   shape, §A) and run6's rounds and the frontier's cells (the serial shape, §B): the ratio is
   1.00 [≈ 0.95, 1.06] on four cells and 1 152 games, and no sealbot move was seen to differ.
   The fix is kept because it makes every game a deterministic function of its own moves (an
   offline replay reproduces it), which the old adapter did not.
2. The dashboard's PROVISIONAL note discharges into this finding: readings through the old
   adapter transfer at ratio 1.00 (`tools/dashboard/tier2.py::SEALBOT_TT_NOTE`).
3. An OBSERVATION, not ruled here: at concurrency 8 the CANDIDATE side is not run-to-run
   reproducible game-for-game — two runs of the same cell diverge at the candidate's move in
   72–78 % of games (median ply 9–19), presumably the inference batch composition under eight
   games in flight (bf16 numerics change with the batch). The level is stable (this record);
   the games are distinct (LAW-04 counts them as such). A cell-level reproducibility statement
   for the rung would need an old-vs-old control, which was not run.

## E. What this record does not say

- Nothing about depth-6 or the strix rung (unaffected by construction: the strix adapter is not
  the vendored engine).
- The serial cell (§B) is ONE cell at 150 sims; run6's 35 rounds were at 32 games each and are
  not individually re-read — the serial delta is the ratio that transfers to them.
- The cost of the A/B to run7: the trainer read ≈ 470 steps/h while round 2 and two 512-sim
  cells shared the card (22:51–00:22 UTC), against ≈ 1 400 beside a round alone.

## Sources

Box: `/workspace/oc7/tt_ab/{old,fixed}/<cell>/{cell.json,result.json,progress.txt,games/}`,
`/workspace/oc7/chain_tt_ab.log`, `tt_ab_{old,fixed}.log`, `chain_tt_ab_serial.log`; mirrored to
the operator's `mantis-mirror/tt_ab/`. Trees: `/workspace/hexo-mantis` (`15109ac3`) and the
worktree `/workspace/mantis-tt` (`748f5c47`, branch `r353`, venv rebuilt on Python 3.11 —
`/workspace/oc7/rebuild_mantis_tt_py311.sh`). Runner: `/workspace/oc7/run_cells_tree.sh <tree>
<cells> <work> <parallel>` over `tools/strength_frontier.py`. Readout: this session's scratchpad
`tt/ab_analyze.py` (pair scores, bootstrap CIs, identical games) and the first-mover script in the
session transcript; the local timing `tt/tt_growth.py` → `tt_growth.json`.
