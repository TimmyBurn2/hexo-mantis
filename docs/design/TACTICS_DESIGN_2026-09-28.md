# TACTICS — the solver's kind and its wiring: measured, then designed (2026-09-28)

Status: **ACCEPTED by R377** (2026-09-28), which rules §12's questions; where its entry in
`docs/governance/RULINGS.md` and this text differ, the ruling governs. **R378** (2026-09-29) rules §12 Q5 by (c),
amends §6's proven-root target form by (e) and extends §11's twin witness by (f); the same precedence holds.
**R379** (2026-09-30) replaces (e)'s median rule and the all-vetoed root's empty row in §6 by (b); the same precedence
holds. **R380** (2026-09-30) retracts R379(b)'s re-search (F1) at deploy and in self-play and keeps the row-wise mixing
(F2) under test; the same precedence holds. **R381** (2026-10-01) kills F2 by screen: the feed of record is arm A's,
no re-search and no mixing, and F2's code leaves the tree; the same precedence holds.
Was: **DESIGN** (the TACTICS-DESIGN packet under R376(d); CARD-TACTICS-LANE re-aimed to it). No code, config
or mint change rides this document. It chooses the kind of the ONE exact tactics module, specifies its API and
its wiring at deploy and in self-play, and scopes TACTICS-DEPLOY and TACTICS-SELFPLAY. The readings it stands on
were measured on the desktop (CPU and the 3070) against Six's pinned source and run8's mirrored saves. Their
records are local, held for `mantis-records/tactics-design/`: rules and amendments hashed before each reading,
drivers, raw rows. §0 gives the readings, §1–§2 the census and the register, §3 the choice, §4–§11 the design, §12 the
open questions.

## 0. What was measured (T2–T4) — the numbers this design stands on

**T2, the same positions through both solvers.**
- Sets:
  - A: the 3 555 turn starts before S in D1's 233 losses.
  - B: the 30 turn starts where D1 proved one of our own wins (in 24 losses), a subset of A.
  - C: the 10 069 turn starts of Six's 240 fixture games.
  - D: 2 000 random turn starts from run8@45k's ring.
  - Supplementary, outside the verdict: S, D1's 232 opponent turn starts where the proved run begins.
- Both boards agreed on the mover and the stones left at every position; none was refused.
- Every position is classed on Six's board: W1 (the mover finishes a window now), DEF (the opponent holds threat
  windows; LOST when the cover needs more than two stones, else BLOCK), or QUIET (neither side holds a window with
  four stones). The PRIMARY set is the 8 966 QUIET positions of A ∪ C ∪ D. 69 of them repeat an earlier
  position (by stone sets, mostly in C's fixture openings), and none is dropped.

| solver, config (turns T, nodes N) | primary WINs | A (2 582) | B (30) | C (5 010) | D (1 374) | S (232) |
|---|---|---|---|---|---|---|
| Six `ThreatSolver`, T 2 / N 64 (its leaf budget) | 2 119 | 5 | 3 | 2 001 | 113 (8.2 %) | 51 |
| Six, T 8 / N 64 | 2 450 | 15 | 11 | 2 283 | 152 | 96 |
| Six, T 8 / N 2 000 | 3 445 | 44 | 28 | 3 180 | 221 | 223 |
| Six, T 8 / N 20 000 (its root budget; D1's instrument) | **3 506** | 49 | 30 | 3 225 | 232 (16.9 %) | 232 |
| ours `TacticalSolver` (F-53's config), every config, depth 4T + 1 plies | **0** | 0 | 0 | 0 | 0 | 0 |
| ours widened (`neighbor_dist` 2), 300-position subsample, every config | **0** (300/300 exhausted) | | | | | |

- Outside QUIET, ours proves every W1 (3 735 of 3 735) and 23 of the 195 LOST positions as LOSSes. Six's
  `analyze` decides both exactly from window counts, before its solver is called.
- Contradictions (a WIN against a LOSS on one position): 0.
- Cost, µs per solve, contended: another session held the desktop at load 30–60 (the solve run 30–38, the
  verifier stages 46–60), and the 8-worker run shows it.
  - On QUIET positions, Six's search-only median / p99 is 73 / 2 064 at T 2 / N 64 and 129 / 227 117 at T 8 /
    N 20 000. With its per-solve clear of an 8 MiB table, a median solve costs ≈ 3.7 ms.
  - Ours costs ≈ 3.4 ms a call, contended, whether or not it proves: the per-call allocation of its 8 MiB proof
    table.
  - The widened arm costs ≈ 100–170 µs a node (dev 100, contended 166): 3.3 s median at 20 000 nodes. That is
    3–980x our own per-call cost and 78–45 900x Six's search-only median, config by config.
  - A single-process re-time on 400 positions per class (RULES A3; load 25 → 6):
    - Six's search-only median / p99 on QUIET is 30 / 568 µs at T 2 / N 64 and 70 µs / 110 ms at T 8 / N 20 000
      (max 222 ms), ≈ 7.3 µs a node. Its 8 MiB clear costs ≈ 0.45 ms.
    - Ours costs ≈ 0.5 ms a call at one node (the table allocation) and ≈ 280 µs a node where it searches (BLOCK
      positions: 4.4 ms median, 66 ms p99).
- **Soundness.** Every one of the 7 473 WIN claims was verified two ways:
  - CHECK 1: the claimer re-solving at every attacker turn, against EVERY non-losing defence, enumerated by a
    brute-force window scan independent of `coveringPairs`.
  - CHECK 2: a play-out against Six gen 455 @64 as the defender.
  - Controls: the planted-unsound copy of Six's `attack()` (the defender cut to its first covering pair) was
    REFUTED on 87 of its 87 extra claims; all 1 460 one-turn claims were CONFIRMED.
  - Outcome: **0 refuted.**
    - CHECK 1: 3 737 of Six's 3 738 claims CONFIRMED; the one INCONCLUSIVE is an 8-turn claim in S whose tree
      passed the 2 000 re-solve cap. All 3 735 of ours CONFIRMED.
    - CHECK 2: all 7 473 CONFIRMED, no forfeit.
    - No defender node had more than 4 non-losing replies, so CHECK 1 enumerated every defence.
  - Determinism: 50 of 50 re-solves per solver identical in a fresh process (verdict, turns, first turn, nodes).

**T3, D1's specificity.**
- Method: 40 of D1's 106 avoidable losses, each "safe" alternative re-solved at 12 turns / 60 000 nodes; 20 played
  out with Six gen 455 @64 on both seats.
- Re-solve: 39 of the 40 alternatives stay SAFE, 0 turn UNSAFE, and 1 is UNRESOLVED (exhausted). **f_corr =
  0.455 [0.385, 0.515]** (0.444 [0.372, 0.505] with the unresolved one counted unsafe). D1's "safe" turns hold
  against a deeper STRICT search.
- Play-outs: 16 of 20 are lost within 13 opponent turns (most within 4–8), so a turn that removes the strict win
  mostly does NOT save the game against a strong opponent.
  - The pre-stated second reading counts those 16 as unsafe and the 20 UNPLAYED alternatives as safe:
    **0.273 [0.196, 0.351]**.
  - On the 20 played alone, 4 held: s 0.20 and f **0.091 [0.032, 0.188]**, wholly below 0.30.
- By the letter, neither pre-stated reading lies wholly below 0.30, so the lane's order stands on T3's band. The
  played-only reading says the defence audit's likely yield is small. The offence (§3) does not depend on it.
  What the audit CONVERTS is TACTICS-DEPLOY's A/B to read (§12, Q2).

**T4, the starvation witness.**
- The exam: 261 QUIET radius-8 positions, each a strict win of 2–6 turns for the mover, proved by Six at
  20 000 / 8 and CONFIRMED by both soundness checks.
  - 214 come from D1's losses and run8's ring; 47 are topped up from Six's fixtures where A and D ran short:
    7 at 3 turns, 29 at 5 and 11 at 6.
  - Frozen by sha256 `2b2eb7f5…`.
- run8's panel saves 36k–54k read it as:
  - the prior on the proof's first turn (the two cells summed): mean **0.196–0.280** (panel mean 0.237, SD 0.028,
    per-save SE ≈ 0.013); median rank 2–3;
  - the value: mean **0.327–0.477** (panel mean 0.406, SD 0.050); 78–84 % of positions read above zero;
  - by proof length, the prior falls from ≈ 0.32 at 2 turns to ≈ 0.16 at 4, then rises again at 5 (≈ 0.24),
    whose stratum is half C.
  - The random-init floor reads 0.003 and −0.010, and every save sits far above it.
  - This is the twin's baseline and band.

## 1. The census (T1): the kinds, side by side

Read at source by symbol: ours at `897d3a06`, Six at the pin `f2b5ec2`, strix's vendored clone at `5a771e5`.

| | ours: `mantis_search::TacticalSolver` | Six: `ThreatSolver` + `mcts.cpp::analyze` | strix: `forcing::solve_wide` |
|---|---|---|---|
| node | one STONE (`tactics::search::solve`); per-stone negamax α-β, PVS/LMR, aspiration, iterative deepening in plies | one ATTACKER TURN (`ThreatSolver::Impl::attack`); iterative deepening in attacking turns, depth-first, TT | one turn (`atk_within` / `def_within`); IDTT |
| attacker moves | `ordering::candidates`: stones making a FIVE (`Board::threat_moves`) + the opponent's five cells; quiet cells only with `neighbor_dist` | `doubleThreats`: pairs whose new fours need >= 2 blockers, from three-cells (<= 48), two-windows and free partners (<= 4), ranked by cover then fours; cover >= 3 wins | (hot, partner) pairs with B >= 2; `wide` adds quiet builders |
| defender moves | per stone: the five's block + counter-fives (in check); a not-in-check node cannot conclude a LOSS | `coveringPairs`: EVERY pair hitting every threat window. At cover 2 that is every non-losing reply, so each proof is exhaustive over defences | every minimum 2-cover |
| strictly forcing | only through fives (the defender must be in check at both stones) | yes: "every attacking turn needs both defender stones to block" | yes (B >= 2) |
| proves | WIN and LOSS | WIN (LOSS only as lost-on-cover in `analyze`) | WIN |
| budgets | depth in plies, nodes = stone expansions; an 8 MiB TT (65 536 buckets) allocated per call | turns (excl. the completing turn), nodes = attacker turns; an 8 MiB TT, cleared by the caller | turns (incl. the completing turn); per-solve TT |
| on the search path | NO (offline probes via the bridge) | root 20 000 / 8, leaf 64 / 2, lost-on-cover, the forced-block child restriction | root 2 000 / 6 at mr >= 1; leaf off |
| board data | rescans the legal set and the stones per call | incremental window counts with per-side threat / three / two lists | an incremental window kernel |

- **Our leaf quiescence** (`MCTSTree::apply_quiescence`):
  - An exact ONE-turn value override: an own finishable window gives +1, an uncoverable opponent set gives −1,
    and the two-fives blend subtracts 0.3.
  - It moves values only. It saves no GPU evaluation and shortens no descent.
- **Our core threats** (`Board::open_windows`, `min_hitting_stones`, `winning_moves`, `threat_moves`) scan per
  call. `Board::find_winning_line` is terminal bookkeeping only.
- **Why ours reaches nothing QUIET.** HTTT's threat currency is the FOUR, a window a side finishes with its two
  stones. Our not-in-check candidate set is `threat_moves(stm) ∪ threat_moves(opp)`: stones that turn a four
  into a five. With no four on the board that set is EMPTY, and a QUIET root returns UNKNOWN at one node. The
  census predicted this before the run, and T2 read it: 0 of 8 966 at every config.

## 2. The register (LAW-02): F-15, F-38, F-39, F-53, R239

| row | what was measured, with which instrument | what transfers to an 8-turn strictly forcing solver | what does not |
|---|---|---|---|
| F-15 | an expansion-time forced-win short-circuit in the predecessor's self-play: the net never evaluated near-win positions, so no fork learning | the mechanism, and stronger with depth: leaf terminals and a root short-circuit IN SELF-PLAY keep proven positions from the net, and the new kind proves far more positions (T2 D: 16.9 % of QUIET self-play turn starts at 20 000 / 8) | deploy: nothing trains, so nothing starves |
| F-38 | our per-stone prover in LOSS mode, 6–8 turns, 33 "value-blind" losses: all proved; "the lever is search-in-the-loop" | the direction: D1 found a strict opponent run in 232 of 233 losses (1–8 turns, mode 4) and T2 re-proves all 232 | reach, cost and deploy value: F-38 is an existence proof on 33 chosen positions, with no strictness condition |
| F-39 | the same prover as a "one-primitive" TSS at deploy budgets: 3/38 traps flipped (all mate-in-2); broadening regressed to 0/38 at ~100x cost; "a competent pattern-guided minimax is required" | the per-stone five-primitive kind is weak, and broadening kills its depth. T2 re-reads both: ours 0 of 8 966 at every config; widened, 300/300 exhausted with 0 proofs at 3–980x our own per-call cost | the kind: Six's generator is the pattern-guided turn-level search F-39 asked for (2 119 proofs at 64 nodes). DECIDE-1 B4 put its worth at ≈ 1 logit |
| F-53 | our bridged solver at 6/12 plies / 2 000 nodes on 4 877 full-arm ring roots: proof rate 2.30 %, strict novelty 0/112 | the METHOD, which is the falsifier proof-as-target needs. TACTICS-SELFPLAY re-runs it with the new kind before arming the target | the numbers: that solver proves only standing fives. On the same ring the new kind proves 232 of 2 000 random turn starts (11.6 %; 16.9 % of the QUIET ones), against F-53's 2.30 % of full-arm roots (another population: both stones, the full arm). Its novelty is unmeasured |
| R239 | subtree termination re-creates F-15 unless ExIt target injection is mandatory; a SYS-5 design without proof-as-target is rejected at design stage | directly: leaf terminals and the root short-circuit ARE subtree termination. Self-play injects the proof as the policy target (§6), and T4's witness guards it (R376(d)) | deploy (no training) |

Nothing in the register kills the kind. F-38 names the lane, and F-39 names the kind it needs. F-15 and R239 bind
the self-play half only.

## 3. The choice: the turn-level strictly forcing solver is the core

- The packet's rule: "the kind with the higher reach at equal cost and zero refuted WIN claims is the core."
- On the primary set, Six's kind proves 2 119 positions at its cheapest config and 3 506 at 20 000 / 8. Ours
  proves 0 at every config, and so does its widened body at 3–980x its own per-call cost. At every cost where both kinds
  have a configuration, Six's reach is the higher.
- The pre-registered operationalisation of "equal cost" (RULES §4) is a Pareto rule with a paired test. Its
  LETTER returns NO VERDICT, on an artefact the rule did not foresee:
  - Both sides pay an 8 MiB table on every call: Six's `clear()`, and our `prove`'s fresh `ProofTt` (65 536 x
    2 x 64 B).
  - In the contended run, our zero-reach configs cost 31.8 s over the primary set against Six's cheapest 34.5 s.
    - Six's clear alone totals 32.9–33.1 s, ≈ 3 % above our whole call.
    - The other ≈ 1.5 s is Six actually searching, where ours stops at one node.
  - The rule has no null configuration, so a config that proves nothing at a lower cost blocks "dominance".
  - The re-time's medians (Six 504 µs, ours 516 µs at T 2 / N 64) show the floors are alike. It measured medians,
    not totals, and A3 kept it out of the verdict.
- Zero WIN claims are refuted for either kind (§0), so condition (a) holds for both. The letter's NO VERDICT
  rests on the missing null configuration and a floor both sides pay.
- The architect rules on the letter (§12, Q1). This design proceeds on Six's kind.
- **What the kind buys, on record.**
  - Offence: in 24 of S1's 233 losses, a strict win of OUR OWN stood at a turn start (30 positions, 2–8 turns).
    The root solver at 20 000 / 8 proves all 30, and both checks confirm every one: S1 left 24 proven wins on the
    table. S1 is the S ruler's cell. X's analogue is unmeasured.
  - The leaf wiring's reach: 8.2 % of run8's QUIET self-play turn starts hold a strict win of <= 2 turns that Six's
    64-node leaf budget proves.
  - Defence: T3's reading (§0) is the caution. A strict audit removes the proven blunder; whether it holds the
    game is TACTICS-DEPLOY's A/B.

"Six's kind" is the kind, not Six's code. The module below is our own Rust implementation of a turn-level strictly
forcing search, checked against Six's verdicts as goldens (§10). Six's source stays a pinned reference, never
built into the tree.

## 4. The module: `mantis_search::tactics`, rewritten around the turn

ONE implementation per thing:
- The lane's module REPLACES the per-stone `TacticalSolver`. Its brute-force fuzz and red-team tests are rewritten
  for the new kind, and its commit stays the provenance of F-38/F-39/F-53 (repo_design §5 kept it only as their
  instrument).
- `apply_quiescence` reads its one-turn facts from the module in both modes (§4.2). With tactics on, its exact
  branches become terminals (§5.1).

### 4.1 The API (deterministic, net-free)

```rust
pub struct TurnSolver { /* table, scratch */ }
impl TurnSolver {
    pub fn new(table_entries: usize) -> Self;          // a fixed-size table, allocated once
    pub fn clear(&mut self);                           // a generation bump: O(1), no memset
    pub fn solve(&mut self, board: &Board, turns: u8, nodes: u64) -> Solved;
    pub fn line(&self, board: &Board) -> Vec<(i32, i32)>;   // the proof's main line, from the table
}
pub struct Solved { pub verdict: Verdict, pub nodes: u64 }
pub struct Turn { stones: [(i32, i32); 2], len: u8 }  // one or two stones, no heap
pub enum Verdict {
    /// A strictly forcing win for the side to move within `turns` attacking turns; the six lands on the next.
    Win { first: Turn, turns: u8 },
    /// The side to move cannot stop a six this turn: lost on cover.
    Loss,
    /// Neither proven. `exhausted` when the node budget ran out.
    Unknown { exhausted: bool },
}
pub fn analyze(board: &Board) -> LeafTactics;          // exact one-turn tactics from window counts
pub struct LeafTactics { pub terminal: Option<Terminal>, pub forced: Vec<(i32, i32)> }
pub enum Terminal { Win, Loss }                        // for the side to move
```

- **What `solve` searches.** It runs `analyze` first:
  - a finishable window is `Win` with `turns` 0;
  - lost-on-cover is `Loss`;
  - a forced block is `Unknown`, since the strict search needs a QUIET root.
  - Only a QUIET root with two stones left is searched, Six's precondition.
- **Budgets.** `turns` (attacking turns, Six's unit: the completing turn is not counted) and `nodes` (attacker
  turn starts, `attack` calls). No clock: a deadline makes a verdict depend on the host, and a promotion bar or a
  self-play record must replay.
- **Determinism.**
  - Every generated list is sorted: attack turns by (cover desc, fours desc, packed pair asc), covering pairs by
    cell order.
  - The table is keyed by a 128-bit Zobrist key built from `mantis_core::board::zobrist`'s own table and stored
    whole. strix's 2026-07-14 soundness bug was a non-injective key derivation: 2 732 single-key collisions
    within ±64 (strix `docs/research/2026-07-14-solver-zobrist-collision-and-wl4-guard.md`). The key derivation
    is part of what the goldens pin.
    - Amended in place (TACTICS-DEPLOY, L1's review): core's out-of-table `ZobristTable::get_for_pos` is that bug's
      class — a cell and its reflection through the origin share a key in 1 of 8 cells beyond ±9 (2 680 pairs within
      ±64), and a table keyed by it carried a proven win to the reflected position. The solver keys stones by its
      own injective derivation (`tactics::grid::stone_key`), pinned by a distinctness test over ±256 and a
      reflected-pair regression test. Core's `Board::zobrist_hash` keeps the defect (CARD-ZOBRIST-REFLECTION).
  - `clear` bumps a generation, so a stale entry reads as empty.
  - Same board, same budgets, same table state: the same `Solved`, nodes included.
- **The table.** Proven entries carry `proven <= turns_left`; "searched without a proof" entries carry the depth.
  It is a completeness cache only: a Win is concluded only after every covering reply is refuted. The deploy
  root clears it per search, and the leaf solver shares it across one search's leaves, clearing it at
  `MCTSTree::new_game`.
- **Loss.** The kind proves wins. `Loss` means lost on cover, decided by `analyze`. A stronger Loss (every
  covering reply of ours loses to a strict opponent win) is an AND node over our covering pairs; it is cheap only
  at cover 2, and it stays out of v1 (§12, Q4).
- **The line.** `line` walks the table from the root: the attacker's stored turn, the first covering reply, the
  next stored turn. Proof trees are trees; the line is a display and golden artefact, never a verdict.

### 4.2 Immediate wins and forced blocks from window counts (`analyze`)
In order, as Six's `analyze` and our `apply_quiescence` both decide it:
1. The mover holds a window it can finish with its stones left, and every gap is legal → `Terminal::Win`.
2. The opponent holds threat windows (>= 4 stones, none of the mover's):
   - their cover needs more stones than the mover has left → `Terminal::Loss`;
   - else `forced` = every empty cell of those windows. "Some stone of this turn must block; the order of a turn's
     stones doesn't matter, so block first."
3. Else nothing: the position is QUIET for the solver.

This is ONE implementation of the one-turn facts, in BOTH modes.
- `apply_quiescence` is rebased on `analyze`: its +1 / −1 branches and the two-fives blend read the facts from it.
- A parity pin on the leaf corpus holds its values bit for bit against today's `open_windows` +
  `min_hitting_stones`. Quiescence is on in `configs/run10.yaml`, so tactics-off stays today's search value for
  value.
- `analyze` also reports the cover of the opponent's FIVES (windows needing one stone), which the blend reads.
- `mantis.diagnostics.tactics` (the census vocabulary in Python) is carded to call it through the bridge.

### 4.3 The incremental data: a local grid, not a Board change
- `analyze` (every leaf) needs no grid. It scans the stones' windows sparsely, as `open_windows` does today:
  ≈ 30 lookups a stone. Its cost is H1's.
- The SOLVER owns a dense 256 x 256 grid (Six's width), allocated once with the `TurnSolver` and reused:
  - at each solve, the position's stones are placed, O(stones x 18) window updates, centred on the stones' bounding
    box; they are removed when the solve returns;
  - there is no per-solve memset;
  - inside the search, each stone placed or undone is 18 window updates.
- A placement within 16 cells of the grid's edge ends the solve as `Unknown { exhausted: false }` and counts
  `grid_overflow`: a named row, no recentring in v1. A game that wide is a finding (Six's engine refuses 1 of run8's
  62 481 games at its 191-cell span).
- The grid holds:
  - per-window packed counts per axis;
  - per-side lists of threat (>= 4), three (== 3) and two (== 2) windows with no opponent stone, with O(1)
    removal;
  - an incremental 128-bit key.
- `mantis_core::Board` is NOT changed. Its benches (`tools/bench_floors.toml`) and every consumer stay as they
  are. The module reads `cells_iter`, `current_player`, `moves_remaining` and `legal_move_radius`.
- Legality needs no coverage bookkeeping: every generated cell is a gap of a window holding a stone, so it lies
  within 5 of a stone. `MCTSTree::configure_tactics` refuses a tree whose board radius is below 5 with a named
  error (production is 8). The solver never meets such a board.
- If the leaf-analysis cost (§9, H1) demands it, the grid can later ride the MCTS descent (applied in
  `select_one_leaf`, undone in `rewind`). That is a separate, benched commit.

## 5. Deploy wiring (lands first)

The tree owns the module: `MCTSTree::configure_tactics(TacticsConfig)`, off by default. The Python
`DeployHeadPlayer` and the Rust self-play `play_one_move` both drive the same `MCTSTree` methods, so the code at
deploy and in self-play is one code (LAW-15).

### 5.1 Leaf terminals
- **The root is always expanded** (Six's rule, `mcts.cpp` `gather`: "The search root is always expanded, even
  when lost").
  - `analyze` runs on EVERY new leaf, the root included. At the root its `forced` set restricts the children
    (§5.2), but its `terminal` is never applied: the root's decided cases belong to the offence, which runs
    before the search (§5.3).
  - A lost-on-cover root is expanded over every legal child: nothing holds, and the audit's best hold (§5.3)
    picks the longest resistance.
  - With `root_nodes` 0, a QUIET root is searched like any other: no terminal.
  - Without this rule, a decided root is never expanded. Self-play then returns `RootExpansionFailed` →
    `MoveOutcome::Continue` and re-searches the same position until the ply cap; the deploy head raises "no root
    children".
- In `MCTSTree::select_leaves` and `select_leaves_forced`, after `descend` returns a NEW leaf and before
  `queue_leaf`:
  - run `analyze` on the leaf board;
  - at a NON-ROOT leaf that is QUIET with two stones left, and `leaf_nodes > 0`, run `TurnSolver::solve(leaf,
    leaf_turns, leaf_nodes)`.
- At a non-root leaf, a Win or a Loss makes the node terminal with ±1 for its side to move (the CF-1 convention
  `backup` already uses). It is backed up inline, counted as a DESCENT and a SOLVER TERMINAL, and never queued: no GPU
  evaluation.
- A re-selected terminal node (won, lost on cover, or solver-proven) is backed up inline and COUNTED as a descent.
  - Today a won leaf's first visit is queued for the net, whose output the expansion then discards
    (`check_win` in `expand_leaf_with`).
  - Its re-selections depend on the kind. Under PUCT (`select_leaves`) they ride the TT-hit path, uncounted.
    Under Gumbel (`select_leaves_forced`, which has no TT path) they are queued again and counted as served
    leaves.
  - Amended in place 2026-09-29 (TACTICS-SELFPLAY P0, R378(c)): the PUCT table path now counts every descent
    toward `n`, a terminal there as an inline descent (it backs up its own value) and any other as a table hit.
  - With tactics on, the first visit is decided before the queue, and every revisit is a counted descent. A search
    whose visits pile onto a proven child therefore still spends exactly its budget, instead of starving
    `select_leaves` of countable leaves.
- With tactics off, the terminals, the restriction and the leaf solver are skipped. `analyze` still feeds
  `apply_quiescence` (§4.2), and the parity pin keeps today's values bit for bit.
- The leaf's `forced` set rides its `MCTSTree::pending` entry to its expansion (§5.2).
- `apply_quiescence`'s exact branches become the terminals above. Its heuristic blend (−0.3 for two fives against
  a two-stone turn) is decided by its own open question (§12, Q6).

### 5.2 The forced-block child restriction
- At expansion (`expand_leaf_with` → `pick_topk_children` / `pick_topk_children_ls`), a non-empty `forced` set
  replaces the legal set: the children are the forced cells, and the priors are renormalised over them.
- It is exact: a non-blocking stone loses on the opponent's next turn. The mover's own immediate win is decided
  first (§4.2), so no winning alternative is cut.
- At the root it shrinks Gumbel's child set, and the draw is over the forced cells. `m` clamps to the child count
  as today.

### 5.3 The root audit
The deploy head decides per stone with a fresh tree per call (`DeployHeadPlayer.select_move` → `tree.new_game`),
so the audit spans two calls. It is two `MCTSTree` methods, called by both drivers:
- `root_offence`, BEFORE the search, as in Six's `searchStone`;
- `root_audit`, AFTER the search and before the move is returned.

- **Offence.** At a root with two stones left:
  - `solve(root, root_turns, root_nodes)`; a Win plays `first`'s first stone WITHOUT a search.
  - With two stones in `first`, the tree stores `next_proof_stone = (key after it, the second)`. It is named apart
    from `MCTSTree::pending`, the leaf queue `new_game` clears. It survives `new_game` and is valid only on that
    exact key; the runner clears it at a game's start, so no state crosses games. Amended in place 2026-09-29:
    nothing clears it between games; it is taken at its next use and filtered by key, so a stale one is dropped.
  - A proof stone, the next proof stone and a finishing cell are NOT root children. Each is checked against the
    board's legal set before it is returned. An illegal one is a named error (`TacticsError::ProofStoneIllegal`),
    counted, and the search's own move is played instead (§8).
  - At the next call (one stone left) the stored stone is played, again without a search.
  - `analyze`'s Win (a finishable window, one or two stones) is played at either call, without a search.
  - **Self-play's branch point.** `play_one_move` calls `root_offence` before `run_mcts_search`. A decided root
    skips the search AND `records::refuse_zero_visit_export`, which would latch an unexpanded root run-fatal. It
    writes the proof target (§6) through its own record branch, with its own test. The guard stays on every
    searched root. Amended in place 2026-09-29 (R378(e)): a decided self-play root IS searched, plays its stone and
    records the search's own target, so there is no separate branch and the guard covers it.
- **Defence at one stone left** (the turn completes):
  - After the search picks d, audit `post = root + d` (the opponent to move with two stones).
  - `analyze(post)` Win for the opponent (we left a finishable window) → veto. On QUIET, `solve(post,
    audit_turns, audit_nodes)` Win → veto.
  - On a veto, walk the root's children in the kind's own order (PUCT: visits, then prior; Gumbel: Sequential
    Halving's final ranking) and take the first whose `post` is not a proven opponent win. `Unknown` holds.
  - After `audit_k` candidates with no hold, take the BEST HOLD: the candidate whose opponent proof needs the most
    turns, ties to the kind's order. The loss is delayed as long as the proof allows.
- **Defence at two stones left** (the first stone):
  - The chosen c is HOLDABLE if one of c's top `audit_m` second stones holds: its visited children by visits, then
    the forced cells.
  - The holding second stone d2 is stored as `next_hold_stone = (key after c, d2)`. At the next call (a fresh
    tree), a vetoed pick tries d2 first, before walking the candidates. Without it, the second call could miss the
    hold the first call proved and fall back to a best hold, which is a proven loss.
  - If c is not holdable, try the next root candidates the same way, up to `audit_k`; with none, take the best
    hold.
  - D1: 100 of 106 avoidable losses turned on our LAST turn before the opponent's run, and a pair decides it, so
    the audit must see both stones.
  - Amended in place (TACTICS-DEPLOY, L3, corrected after its review): the position after a first stone is decided
    by `analyze` first — a finish holds on its stone, a lost cover is a loss in 0 turns (no second stone helps).
    Otherwise its top `audit_m` second stones are its children in the tree's own order (visits, then prior), then
    the forced blocks after it. One with nothing to try (unexpanded, quiet after it) holds unproven, as an `Unknown`
    post does, and an alternative returned that way is counted (`audit_unvetted`). The first text read an
    unexpanded lost-cover first stone as "nothing to try" and let it hold.
- **Cost bounds.**
  - The audit's calls share one total budget, `audit_total_nodes`. At exhaustion the kind's own choice stands,
    counted.
  - The audit uses the search's own candidates and order, never a new enumeration. The D1 driver's all-pairs
    scan (1.27 M alternatives on 138 turns) is an instrument, not a deploy path.

## 6. Self-play wiring (default off until the twin)

- **One schema block, two homes.**
  - `SearchConfig` (`src/mantis/config/schema/search.py`) gains a REQUIRED key `tactics: TacticsConfig | None`,
    with `null` the explicit off (the house idiom).
  - Its leaves: `leaf_nodes`, `leaf_turns`, `root_nodes`, `root_turns`, `forced_block`, and an `audit` block
    (`nodes`, `turns`, `k`, `m`, `total_nodes`).
  - `deploy.search.tactics` arms the deploy head and `selfplay.search.tactics` the workers: the same Rust code
    behind both.
  - The template and every production config mint `null`. `configs/run10.yaml` stays the config the instruments
    read (R376(c)).
  - Consumers (LAW-08): `SelfPlayRunnerConfig` → `MCTSTree::configure_tactics`, and the stamp's `deploy.search`
    → `DeployHeadPlayer` → the same call.
- **Proof as the target (R239).** At a root the offence plays without a search, the recorded policy target is the
  proof:
  - two-hot 0.5 / 0.5 when `first` holds two stones, one-hot when it holds one, and one-hot on the stored stone at
    the next call (strix's form);
  - the row is marked `proven_root`, with a LAW-18 fire rate (as built: no row marker; the runner's
    `proven_root_rows` counts it);
  - the value target stays the game's z.
  - Six records no policy target at such a root (`decided`). This design takes strix's form, because R239 makes
    injection mandatory and the target is what trains. §12 Q3 asks the architect.
  - AMENDED by R378(e), in form, not rule: the target at a proven self-play root follows A9's reading. It is the
    searched target where its proof-set mass reads ≥ 0.7 at the median, else the α = 0.5 mixture; never the bare
    two-hot above.
  - REPLACED by R379(b), row by row: the target mixes the proof at α = 0.5 only where the searched mass on it reads
    below 0.5; elsewhere it is the searched target.
  - KILLED by R381(a), by screen (A″ − A −0.48 [−0.80, −0.16] on the shipped head): the feed of record is arm A's.
    The F2 bullets below record what TACTICS-SELFPLAY-2 built; HYGIENE-1 took the code out of the tree, its
    feed byte-equal again to TACTICS-SELFPLAY's (`368cbad0`), and `mixed_rows` with it.
  - Built (TACTICS-SELFPLAY-2 F2). `root_offence` names the proof (`MCTSTree::last_root_proof`): a finish's window
    cells, a found proof's pair (or its one stone), or the stored stone; only its root-legal stones, since a pair's
    second stone can be legal only after its first. The searched mass on it sums its cells in either order.
    `mix_proof` sets the target to 0.5·searched + 0.5·proof, the proof's stones in equal shares, at every decided row
    whatever its arm; `mixed_rows` counts the mixed rows among `proven_root_rows` (full draws). A Gumbel row stores
    the proof's cells explicitly (`fitted_support`), taking slots from the lowest-mass candidates, since an unvisited
    proof cell would otherwise sit in the prior-shaped tail.
  - Stated, not changed: a finish's proof is one window, so a target split over two finishing windows is mixed toward
    one; a PUCT row at `leaf_batch_size` 1 could need one slot past its derived capacity for two unvisited proof
    stones (every minted config batches 8).
- **A vetoed cell outside the searched candidates (as built).** A Gumbel row stores its searched candidates only
  (`gumbel_m` slots), so such a cell is zeroed in the target but keeps its training-tail share; `vetoed_target_rows`
  counts only rows whose every veto holds no mass.
  - SEALED by HYGIENE-1 under R377(f) (the audit's F7): the training tail spreads over every unstored legal cell by the
    prior, so a sparse row now stores every veto at zero within its slots, and the tail renormalises over the rest. A
    root restricted to its forced blocks, past its slots, leaves unvisited blocks the row cannot name, and a veto with
    no slot cannot be stored: in either case the tail folds into the stored cells (`tail_leak_rows` counts every row
    that would have leaked; a fold with no stored mass records no policy, `tail_emptied_rows`). A written row with a
    policy whose tail still leaks is the run-fatal `TailLeak`.
- **All-vetoed roots (added as built).** When every unit of the searched target sits on vetoed moves, the target is
  left as searched and the row records no policy target, as a lost root's does (`emptied_target_rows`).
  REPLACED by R379(b), at deploy and in self-play alike: such a root is re-searched over the non-vetoed set at the
  same budget; that search's improved policy is the row's target and its winner the played move.
  RETRACTED by R380(b), at deploy and in self-play alike: such a root records no policy target and plays the audit's
  best hold, as arm A did, R378(e)'s decided root by another route. The F1 bullets below record what TACTICS-SELFPLAY-2
  built and ran; TACTICS-SELFPLAY-3 lands the retraction in code (R380(f)).
  Landed by TACTICS-SELFPLAY-3's L1 (2026-09-30): no re-search in either home and no `research_*` row; the code is
  arm A's again, F2 beside it.
  - Built (TACTICS-SELFPLAY-2 F1, as its review amended it). The letter, "every searched move is vetoed", cannot
    fire under Gumbel (the audit walks k + 1 = 5 of m = 16 candidates), so the build reads it in two parts, stated to
    the operator with the exit:
    - The move: the re-search's winner is played only where the audit held on nothing (its pick is itself a veto) and
      the kind's untempered target sits wholly on the vetoes (`all_vetoed`, tolerance 1e-4) —
      `MCTSTree::searched_all_vetoed`, one predicate both homes read, so both play the same move on the same tree. An
      audited hold is never replaced, so a stored `next_hold_stone` pair survives.
    - The target (self-play only): a row whose own target sits wholly on the vetoes (PUCT's at the move's temperature;
      the card's 8 366 emptied rows) is searched again for its target whatever the move; where the audit held, the
      hold stays the move and the re-search supplies only the target (`research_over_hold`).
  - `MCTSTree::begin_research` re-arms the root without the vetoes and carries the first search's rows. It refuses a
    lost root (its row carries no policy), a root with no legal move left, and a last stone whose every legal forced
    block is vetoed: every block is proven lost and every other stone leaves the opponent a finish, so it is a lost
    root too (`research_refused_lost`, counted among `decided_lost_rows`), and the best hold plays. At two stones left
    a non-block first stone is searched, since a block may follow it.
  - The winner is the kind's own pick (`select_move` in self-play, the head's `_search` at deploy), not audited again.
  - The re-searched row's sparse support stores the vetoes at zero (`fitted_support`), taking slots from the
    lowest-mass candidates, whose mass joins the tail: a vetoed cell outside the support would take the tail's
    prior-shaped share. The pins keep the audit's walk order; with k + 2 ≥ gumbel_m they can take every slot (the
    block of record is k 4, m 16).
  - Rows: `research_count`, `research_over_hold` and `research_refused_lost` (`TacticsCounters`, both homes); the move
    row `research_rows` counts re-searched full rows holding every veto at zero. `emptied_target_rows` counts only an
    all-vetoed row no re-search could take. Under PUCT self-play `research_count` also holds target-only re-searches at a
    vetoed best hold whose tempered row empties while the untempered read does not, so the winner-played rate is not
    read off it there; under Gumbel the two reads coincide.
- **Lost roots in self-play.** A lost-on-cover root is searched (the root is always expanded), but every child
  loses. Following Six (`searchStone` marks it `decided`), its row carries NO policy target: counted as
  `decided_lost`, value target z. §12 Q3.
- **Audit vetoes in self-play.** The played move is the audited one. The vetoed candidates' mass in the searched
  target is zeroed and the target renormalised: the proven-losing move is taught as a zero, the same principle as
  the proof target. §12 Q3.
- **Leaf terminals in self-play** are F-15's hazard: proven leaves never reach the net. The T4 witness reads the
  net's prior and value on a frozen exam of verified strict wins at every twin save. The band is §0 T4's. A save
  below the panel's mean − 3 SD on mean prior (< 0.154) or mean value (< 0.255) is a starvation signal (R376(d);
  §11 gives its false-alarm rate and power).
- **F-53 before the target arms.** TACTICS-SELFPLAY re-runs PROBE-1's reading 4 with the new kind (proof rate,
  strict novelty, ms per root) on run8@45k's ring, with a pre-stated line, before `selfplay.search.tactics` is
  minted non-null.

## 7. Accounting (R376(e); LAW-18 rows)

- A SIMULATION is a DESCENT that backs up a value. The budget loops count descents: `run_mcts_search`,
  `DeployHeadPlayer._drive_puct` and `._drive_gumbel`.
- Three units:
  - SERVED LEAVES are leaves answered by the inference seam, eval-cache hits included: today's unit,
    `max_sims_per_search`.
  - GPU EVALUATIONS are the eval cache's misses (the self-play runner arms `EVAL_CACHE_CAPACITY`). They are
    their own row.
  - INLINE DESCENTS are backed up without the seam: solver terminals and terminal revisits, with tactics on.
- `infer_and_expand_graph` returns `(served_leaves, inline_descents)`. `select_leaves(n)` /
  `select_leaves_forced` count inline descents toward their n, so returned boards + inline descents <= n and a
  batch never overspends.
- Amended in place 2026-09-29 (P0, R378(c)): a fourth unit, TABLE HITS, the descents backed up with the TT's value
  (`last_tt_hits`), counts toward n as well; a terminal on the table path is an inline descent, with tactics off
  too. `infer_and_expand_graph` returns `(served, inline + table)`, and every loop counts served + inline + table.
- **A named deviation from R376(e)'s letter, awaiting a ruling (§12 Q5).** A TT-hit expansion of a non-terminal
  leaf is a descent that backs up a value, so R376(e) counts it. Today it is uncounted (PUCT only; Gumbel has no
  TT path).
  - This design keeps it uncounted in both modes. Counting it would change the tactics-off search wherever PUCT
    transposes, and void the replay witness TACTICS-DEPLOY's plain arm relies on (§11).
  - Amended in place (TACTICS-DEPLOY, L2's review): the deviation also STARVES. A select call whose 4n attempts are
    all TT hits returns nothing, and the budget loops stop short, tactics on or off. It is counted, not fixed: the
    runner's `starved_searches` / `starved_descents` rows and the head's `last_sims`, and the witness's tactics-on
    cases pin served + inline + starved == N a search (CARD-TT-HIT-STARVATION). A ruling on Q5 removes it.
  - RULED by R378(c): a TT hit is a counted descent, as are a net leaf and a solver terminal. The plain head's early
    end is a defect, fixed with a planted-break test before the twin (CARD-SIMS-ACCOUNTING). The witness pins all
    three cases, and TACTICS-DEPLOY's A/B stands.
- Rows per search, summed per worker into the runner's stats snapshot and emitted as manifest rows; per
  `select_move` on the head as `last_tactics`:
  - `descents`, `gpu_evals`, `solver_terminals` (split `win1`, `lost_on_cover`, `strict_win`), `terminal_revisits`;
  - `forced_restrictions`, `leaf_solver_calls`, `leaf_solver_exhausted`;
  - `root_proofs_found`, `proof_stones_played`, `finishes_played`, `root_solver_exhausted`, `decided_lost`;
  - `root_vetoes` (the chosen turn allowed a proven opponent win), `audit_swaps` (the audit changed the move),
    `best_holds`, `audit_unvetted`, `audit_calls`, `audit_exhausted`;
  - `proven_root_targets` (self-play).
  - Amended in place 2026-09-29: `table_hits` joins the rows (`TacticsCounters::rows` is the one name list), and a
    self-play move adds `proven_root_rows`, `decided_lost_rows`, `vetoed_target_rows` and `emptied_target_rows`
    in `proven_root_targets`' place; the runner sums them all (`SelfPlayRunner.tactics_totals`).
  - Amended in place 2026-09-30 (TACTICS-SELFPLAY-3): `mixed_rows` (F2) and `vetoed_all_rows`, every all-vetoed root
    whatever its row holds, join the move rows; `emptied_target_rows` equals it while each such row has no policy.
    Amended 2026-10-01 (HYGIENE-1): `mixed_rows` left the tree with F2; `vetoed_all_rows` stays.
    `tail_emptied_rows` (a kind) and `tail_leak_rows` (a cross-count) join them with the tail seal.
    Amended 2026-10-06 (R386(d)): `unsearched_decided_rows`, the quick draws' decided roots played unsearched, joins
    them, the one move row a quick draw adds.
- The served-sims witness pins DESCENTS per SEARCHED root: `descents == n_sims`, tactics on or off.
  - With tactics on: `served_leaves + solver_terminals + terminal_revisits == descents`, and
    `gpu_evals <= served_leaves`.
  - With tactics off: `served_leaves == descents` (today's pin; the deploy head's test counts infer calls).
  - A root the offence decides with no search serves 0 descents and is carved out of the pin. Three cases: a
    proof found, the next proof stone played, or a finishable window played. Each has its own counter, so no
    proof is counted twice.
  - Amended in place 2026-09-29 (P0): with tactics on, `served_leaves + solver_terminals + terminal_revisits +
    table_hits == descents`; with tactics off, served + inline + table == descents, and the deploy head's test reads
    root visits, not infer calls. In self-play a decided root is searched (R378(e)), so it spends its budget too.
  - Amended in place 2026-10-06 (R386(d)): only on the full arm. A quick draw's decided root plays its stone with
    no search and serves 0 descents, its row value-only (no explicit entry, no tail, root value +1), counted on
    `unsearched_decided_rows`; a quick row trains no policy, so the search could change nothing it carries.
  - Amended in place (TACTICS-SELFPLAY-2 F1): a re-searched root runs two searches, each spending exactly its budget
    (`max_sims_per_search` stays the budget), and both count, so a move's `descents` is `(1 + research_count) × n`;
    the deploy head's `last_sims` sums both. RETRACTED by R380(b): since TACTICS-SELFPLAY-3's L1 a move runs one
    search, its `descents` is `n` and `last_sims` is that search's.
- Every row gets a producer test (LAW-07) and an entry in `docs/contracts/event_manifest.md`.

## 8. Protected contact

| protected item (LAWS.md) | pinning tests | touched how | kept green how |
|---|---|---|---|
| served-sims exactness, which pins descents (R376(e)) | `served_sims_exact.rs::both_kinds_serve_exactly_sixty_four`, `::r8_at_fifty_sims_serves_exactly_fifty_per_search`, `test_deploy_head_budget_spent.py::test_every_kind_spends_exactly_its_budget` | the budget loops count descents; solver terminals are descents without a GPU call | tactics off is the default and leaves every count as it is. Each test gains a tactics-on case asserting descents == n_sims and the §7 identity. The names stay; the module doc's "N leaves of network work" becomes "N descents; network leaves are their own row", citing R376(e) |
| the search-kind conformance | `search_kind_conformance.rs::a_gumbel_round_is_exactly_the_halving_phase_wide` | a Gumbel round may hold solver terminals, and a forced root shrinks the child set | a round still spends exactly its width in descents. The test gains a tactics-on case; `m` clamps as today |
| arena legality | `test_legality_boundary.py::test_a_candidate_playing_off_the_legal_set_forfeits_and_the_move_is_not_applied`, `::test_an_opening_that_does_not_replay_is_a_fatal_corpus_error` | the audit substitutes a root child; the restriction narrows children; the offence returns a proof stone, the next proof stone or a finishing cell, which are NOT root children | root children are legal. The three solver-supplied stones (and the audit's stored hold stone, a second stone the first call tried, checked against the legal set and dropped if not in it) are checked against the legal set before return, with `TacticsError::ProofStoneIllegal` counted and the search's move played; a new test plants an illegal proof stone and asserts the refusal. The arena's own check is untouched |
| gate pair statistics | `test_gate_pair_statistics.py::test_the_two_legs_of_an_opening_are_one_unit`, `::test_the_gate_ci_and_eff_n_are_both_over_pairs` | TACTICS-DEPLOY's A/B reads paired openings | untouched |
| every other item (warm-start hash, conformance roster, 1-in-1 collate, finite-gradient guard, resume bundle, F-816-37, strength_floor, draw-rate abort) | as listed in LAWS.md | not touched | — |

## 9. Cost: the LAW-09 bench plan
Every hotspot is pre-registered with its expected bracket and its abort line. The numbers derive from T2's costs
(§0: contended, and the single-process re-time). One change is one commit is one IQR-gated bench.

| hotspot | bench | expected | abort |
|---|---|---|---|
| H1 leaf analysis (`analyze`'s sparse scan) on every new leaf, and the solver grid's place/remove at each solve | `mcts_bench` new group `tactics_leaf/{off,on}` on the graph leaf corpus | <= +10 µs a leaf at 100 stones (A2's quiescence scan cost ≈ 3 µs at 32 stones; place/remove is O(stones x 18), no memset) | > +40 µs a leaf, or > +15 % on `expand_leaf/ls_graph` |
| H2 the leaf solver at 64 nodes / 2 turns on QUIET two-stone leaves | the same group, `tactics_leaf/solver64` | a search-only median of 30 µs and p99 of 568 µs per QUIET two-stone leaf, max 1.2 ms (the re-time; contended: 73 / 2 064), plus ≈ 0 at non-QUIET leaves (the precondition fails in < 1 µs) | self-play descents/s on the box's `tools/bench_server.py` below 0.8x tactics-off, or GPU evaluations/s below 0.75x |
| H3 the root solver at 20 000 / 8 | a root corpus bench | a median of 70 µs, p99 110 ms and max 222 ms per root (the re-time; contended: 129 µs / 227 ms) | p99 > 500 ms, or the deploy head's per-move wall up by > 25 % |
| H4 the root audit | the deploy head on a mid-game corpus | <= `audit_k` x `audit_m` calls within `audit_total_nodes`; p99 <= 250 ms | > 25 % of the per-move wall (Six's `rootSolverShare` is 25 %) |

- **In-run.** Descents/s and GPU evaluations/s are read beside each other: solver terminals are descents that
  cost no GPU. The box baseline is PERF-ADA's loop (4 505 leaves/s, GPU 89 %). The expected band for descents/s
  is [0.9x, 1.1x] of tactics-off.
- Amended in place (TACTICS-DEPLOY, L5): H2's named instrument cannot read H2. `tools/bench_server.py` benches the
  inference server alone and runs no search. L5 read H2 on the REAL self-play pool at the run's settings instead,
  with the runner's leaf block armed through a scratch-only bridge patch (the self-play path stays unarmed in the
  tree). It read descents/s 1.080x and GPU evaluations/s 0.928x at 64/2.

## 10. Tests
- **Goldens from T2.** A fixture of ~600 positions stratified over A, C and D and over verdicts, each with Six's
  `(found, turns, first turn, nodes)` at 64 / 2, 2 000 / 8 and 20 000 / 8.
  - The port must reproduce `found`, `turns` and the first turn exactly wherever Six finished well inside its
    budget: not exhausted, and nodes <= N / 2.
    - The table is always-replace: 20 000 stores into 262 144 slots overwrite ≈ 760 entries. Our 128-bit keys
      index differently and collide differently, so a verdict can flip only where the budget binds.
    - Budget-bound positions (the 98 exhausted at 20 000 / 8, and the near-budget ones) are read within a band,
      and `nodes` everywhere within a band.
  - The fixture is data (`tests/fixtures/`), generated once by a driver that stays outside the tree.
- **Soundness fuzz.** Random QUIET positions at radius 8 (production; §4.3 needs >= 5), 8–24 stones.
  - Soundness needs exhaustive DEFENCE, not exhaustive attack. Every solver Win is re-verified by the CHECK-1
    verifier ported to `mantis_core::Board`: every non-losing defence enumerated by a brute-force window scan,
    the attacker re-solving at each turn, the six checked on the board. 0 refutations.
  - Completeness at depth 1: over every pair of cells within distance 2 of a stone (brute force), the solver at
    1 turn must find every pair of cover >= 3, on positions with at most 48 three-cells.
    - The port keeps Six's `kMaxThreeCells` cap (48), so the goldens stay exact.
    - Above the cap, a cover-3 pair may go unseen. That is a named incompleteness, counted as `three_cells_capped`,
      and a lever for later.
  - All attacker pairs to 2 turns would be ≈ 10^9 nodes a position, so the fuzz does not attempt it.
- **The verifier as a test.** T2's CHECK 1, ported: every Win on the goldens is replayed against every non-losing
  defence (a brute-force window scan), the attacker re-solving. 0 refutations. The port closes three gaps T2's
  driver left, none reached in T2 (at most 4 replies a node, a 256-wide board):
  - its singles branch paired x only with cells playable before x;
  - a cover <= 2 node with no legal reply returned without checking the six;
  - both checks read Six's board.
- **Planted breaks** (LAW-07; each must red its test, then be reverted):
  1. Covering pairs cut to the first: the fuzz and CHECK 1 red. T2's control REFUTED 87 of 87.
  2. A leaf terminal's sign flipped: the terminal-value test reds.
  3. A solver terminal not counted as a descent: the served-sims tactics-on case reds.
  4. The audit keeping a vetoed move: the audit test reds.
  5. The forced restriction dropping one forced cell: the restriction test reds.
- **Determinism.** Two runs over the goldens with the table cleared are identical, and so are two self-play
  searches at a fixed seed with tactics on.

## 11. The follow-up packets

Amended in place 2026-09-28 per R377, which ruled §12: the A/B gains an audit-off arm and reads X's floor first
(R377(e)), and TACTICS-SELFPLAY's targets, control and band are ruled (R377(f)).

**TACTICS-DEPLOY: the implementation and the A/B.**
- **Build.** On the desktop: §4–§8, the goldens and the fuzz, then the benches H1–H4, every tactics row null
  except in the A/B's composition. One leg per hotspot commit, each closed by a fresh review.
- **The A/B.** At equal descents (`deploy_sims` 256, PUCT, the stamp's own config), THREE arms of the parent on X
  (Six gen 30 @16, cache 0) and S (strix @ r8), over three panel saves (39k, 45k, 51k), so the read is of the
  lever and not of one net (R375(d)): the plain parent, the full module (`deploy.search.tactics` with the audit)
  and the audit-off module (the same block with the audit off), R377(e). X's floor is read first.
  - The tactics-off tree must replay DECIDE-1's panel cells: a replay witness of one cell per ruler, >= 90 % of
    games identical. Then the plain arm is the panel's own readings, and only the tactics-on cells run.
- **Criteria and power lines** (R376(d) reads both rulers). Each packet restates them with its bands
  (CARD-PACKET-POWER-LINE).
  - **X.** A cell's SE is ≈ 0.128 logit (DECIDE-1), so a paired difference is ≈ 0.181 and the three-net mean
    ≈ 0.105.
    - PASS on X iff the three-net mean lift > 0.172 logit: one-sided α 0.05, a false pass of 0.05 at zero.
    - Power 0.77 at +0.25 logit, 0.985 at +0.40 and 0.998 at +0.47. One net alone would have 0.71 at +0.40,
      below 0.8, so three nets are read.
    - Six's own solver is worth ≈ 1 logit against the parent (B4). The size of our lever on X is unmeasured.
  - **S.** A cell's SE is ≈ 0.194 logit, so the three-net mean ≈ 0.158. S's own criterion: the three-net mean
    lift > 0.261 logit (one-sided α 0.05).
    - Power 0.47 at +0.25, 0.81 at +0.40 and 0.91 at +0.47.
    - +0.47 is what S1's 24 missed proven wins alone would add on S: 55/288 → 79/288.
  - **The lever PASSES iff X passes** (R376(a): recipe decisions read X).
    - S is read with its own criterion and reported PASS or FAIL.
    - X passing with S's three-net point estimate at or below zero goes to the architect as a ruler
      disagreement.
  - **The audit's own share** (R377(e)): full − audit-off on X. The defence audit lands only if its arm earns its
    cost; otherwise the module ships with the audit off by default and the audit is carded. The packet states
    this rule, its power and its false-pass rate before the first cell.
- **Known-bad** (LAW-19). Against the ruler's measured floor: the audit INVERTED (it keeps only candidates that
  ALLOW a proven opponent win when one exists), one X cell on the parent.
  - It must read below the plain parent's X by more than 0.298 logit (one-sided α 0.05 on one paired cell).
  - Power 0.80 at −0.45 logit and 0.997 at −0.80. A deliberate blunder whenever one is available should cost
    far more than −0.45.
  - X's random-init floor is unmeasured (CARD-SIX-RUNG). TACTICS-DEPLOY reads it first (one cell), so the
    known-bad has a floor to stand against.
- **Box-h.**

  | item | box-h |
  |---|---|
  | tactics-on X cells (3 x ≈ 0.33) | ≈ 1.0 |
  | tactics-on S cells (3 x ≈ 0.74) | ≈ 2.2 |
  | audit-off X and S cells (R377(e); 3 x ≈ 0.33 + 3 x ≈ 0.74) | ≈ 3.2 |
  | replay witnesses (one per ruler) | ≈ 1.1 |
  | known-bad + X floor | ≈ 0.7 |
  | build + bench_server on the box | ≈ 0.5 |
  | **total** | **≈ 8.7** (was ≈ 5.5 before the audit-off arm) |

**TACTICS-SELFPLAY: the twin with the witness.** Amended by R378 (2026-09-29): the proven-root target by (e) (§6); the
witness gains an enrichment ceiling, the defence exam and ring-composition bands, and T4 reads by proof length (f);
the in-run gate is deploy-matched, candidate and anchor on the same block (d). The packet carries the bands.
- **Twin.** A 4 h twin from the parent, `selfplay.search.tactics` armed as TACTICS-DEPLOY's read sets it, proof as
  the target (§6), audit vetoes as zeros. R377(f) rules the targets: a proven root plays and records its proof, a
  vetoed move gets zero target mass, and a lost root records no policy target.
- **Before it runs.**
  - F-53's reading re-run with the new kind (proof rate, strict novelty, ms per root), with a pre-stated line.
  - The T4 exam and the panel's numbers carried over from this packet, unchanged; the starvation rule below is a
    proposal.
- **During the run.**
  - LAW-18 rows for every §7 counter.
  - The witness read at every save: mean prior on the winning first turn, mean value.
  - Starvation, RULED by R377(f) (was PROPOSED for TACTICS-SELFPLAY's own pre-registration): a save below the
    panel's mean − 3 SD, i.e. mean prior < 0.154 or mean value < 0.255.
    - False alarm ≈ 0.13 % per row per save at the panel's spread.
    - A true drop of 0.08 / 0.10 / 0.12 in mean prior is flagged with probability 0.45 / 0.73 / 0.91 per save.
    - This departs from RULES §7, which named "below the panel's minimum". That rule's false-alarm rate, ≈ 7 %
      (prior) and ≈ 6 % (value) per row per save, was computed after T4's reading. The exam and the panel's
      numbers carry over unchanged; the rule is the twin packet's to pre-register.
- **Its strength read.** At its 4 h save, on X, against a plain twin of the same length: R377(f) makes the plain
  twin the control. Was the architect's question (§12, Q7): a plain twin doubles the cost, and a panel prior was
  the alternative.
- **Box-h.** ≈ 4 (twin) + 0.5 (preflight) + ≈ 0.7 (two X cells) ≈ 5.2, and ≈ 9.5 with the plain twin R377(f)
  orders.

## 12. Open questions for the architect — ruled by R377, and Q5 by R378

Amended in place 2026-09-28: each ruled question leads with its ruling and keeps its question as "Was".
1. **The verdict's letter.** RULED by R377(a): TACTICS-DESIGN is accepted. The head-to-head's null configuration
   is "no solver", and a solver with zero reach is dominated at any cost; Six's kind is the core, and our per-stone
   solver is retired when the module lands. Was: RULES §4(b) returns NO VERDICT because a zero-reach config at a
   lower table floor blocks dominance (§3). The packet's sentence ("higher reach at equal cost") is met by Six's
   kind: 2 119–3 506 proofs against 0, with 0 refuted claims on either side. Rule on the letter, or confirm Six's
   kind as the core.
2. **T3 and the audit's value.** RULED by R377(e): TACTICS-DEPLOY reads the full module and an audit-off arm
   against the plain parent on both rulers, X's floor first; the defence audit lands only if its arm earns its
   cost (§11). Was: against a deeper strict search, D1's "safe" turns are safe (0 of 40 unsafe; f stays 0.455).
   Played out with Six on both seats, 16 of 20 still lose within 13 opponent turns (f read that way is 0.273
   [0.196, 0.351], or 0.091 [0.032, 0.188] on the 20 played alone). Does the lane go first on the offence and the
   leaves (S1's 24 missed proven wins)? The rows give fire rates, not strength. Reading the audit's own share needs
   a second tactics arm with the audit off (≈ +3.2 box-h). Buy it, or read the bundle?
3. **The self-play targets.** RULED by R377(f), proof as the target: a proven root plays and records its proof, a
   vetoed move gets zero target mass, and a lost root records no policy target. Was:
   - Proven roots: strix's two-hot proof target (this design), or Six's no-target `decided` rows?
   - Lost-on-cover roots: no policy target (Six's rule; this design), or the searched target over children that
     all lose?
   - Do audit vetoes zero the vetoed moves' target mass?
4. **A strict Loss in v1?** RULED by R377(b): v1 proves wins and can't-cover losses, Six's set; deeper loss proofs
   are a later lever (CARD-TACTICS-DEEP-LOSS). Was: at cover-2 positions, every covering reply losing to a strict
   opponent win is a proof of Loss at bounded cost. It would make BLOCK leaves terminal too. In v1 or later?
5. **TT-hit descents: a named deviation. RULED by R378(c): a descent that backs up a value is a simulation,
   whatever backed it (the net, the table or the solver). The plain head's early end is a defect, fixed with a
   planted-break test before the twin; the served-sims witness pins all three cases, and the A/B stands.** Was: NOT
   RULED by R377, which named no item for it. R376(e)'s letter
   counts a TT-hit expansion, a descent that backs up a value; today's PUCT code does not. This design keeps it
   uncounted (§7), because counting it changes the tactics-off search wherever PUCT transposes and voids the
   replay witness the A/B's plain arm relies on. Rule the deviation, or order the change and re-read the plain
   arm: its six cells replace the two replay witnesses, ≈ +2.1 box-h net.
6. **`apply_quiescence`'s blend.** RULED by R377(d): the quiescence override and its blend stay in v1 behind the
   shared analysis function; retiring the blend is its own leg (CARD-QUIESCENCE-BLEND-RETIRE). Was: the −0.3
   heuristic for two fives against a two-stone turn is not exact. Keep it with tactics on, or retire it with the
   exact branches?
7. **TACTICS-SELFPLAY's control.** RULED by R377(f): a plain twin is the control. Was: a plain twin of equal
   length (≈ +4.3 box-h), or the panel as the prior?
