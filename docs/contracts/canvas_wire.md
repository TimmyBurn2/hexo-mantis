# Contract: canvas wire (#12)

- version: v1
- owner: crate mantis-graph (`build_canvas`, the per-leaf canvas); the batch wire's owner is the serving path
- status: v1 — the per-leaf canvas lands with its builder (DENSE-1 L1, `docs/design/DENSE1_DESIGN_2026-10-11.md`
  §2.3 and §5). The batch layout, its collate checks, its decode and the root rule's exits are built with the serving
  path (DENSE-1 L2), which extends this doc; until then no path serves or samples a canvas: every path that builds an
  axis graph refuses a `hexcanvas` row by name (`resolve_geometry`, `is_graph_representation`).

## Summary

The canvas is the hex-conv kind's input for one position. It covers exactly the graph's node set, the FENCE (every
stone and every legal cell, the union of the stones' hex balls of radius `legal_move_radius`), masked inside the
fence's smallest axial box. No centre, window or crop edge exists: the canvas's origin is the fence's minimum corner,
and every cell outside the fence is zero on every plane, so a net that masks every op to the fence sees the same input
whatever container it is served in. The reply to the search stays the graph's (`LegalSetPolicy`'s trunk-19 window and
overflow map), so the canvas carries the graph builder's legal order and reply slots unchanged.

## The per-leaf canvas (`mantis_graph::Canvas`)

| field | type | meaning |
|---|---|---|
| `axis_pair` | `u8` | the axial pair the rows and columns run along: 0 = (q, r), 1 = (r, s), 2 = (s, q), with s = −q − r |
| `height`, `width` | `usize` | the container: the region's extent along the pair's first and second coordinate, each rounded up to a multiple of `canvas_side_step` and at least `canvas_min_side` |
| `origin` | `(i32, i32)` | the pair coordinates of cell (0, 0): the region's minimum corner |
| `far` | `bool` | the region is the far one (below) |
| `fit_side` | `usize` | the smallest of the three pairs' square buckets: the position's size class, invariant under D6 |
| `planes` | `Vec<u8>` | `[6, height, width]` row-major, each 0 or 1, in `CANVAS_PLANES` order |
| `legal` | `Vec<(i32, i32)>` | every legal cell `(q, r)`, lexicographic: the graph builder's legal nodes in its order |
| `legal_gather` | `Vec<i32>` | per legal cell, its cell index `row * width + col`, or `FAR_CELL` (−1) for a cell outside a far region |
| `policy_dst_slot` | `Vec<i32>` | per legal cell, its slot in the trunk-sized reply window, `OFF_WINDOW_SLOT` (−1) off it: the graph's `policy_scatter_index` |
| `window_center` | `(i32, i32)` | the reply window's centre: the graph's bbox midpoint |

`CANVAS_PLANES` (the registry row's `plane_layout`, single-sourced from the builder):

| # | plane | 1 on |
|---|---|---|
| 0 | `own` | the side to move's stones |
| 1 | `opponent` | the other side's stones |
| 2 | `legal` | the legal cells inside the region |
| 3 | `fence` | the region; also the mask |
| 4 | `stones_left_1` | the whole region when one placement is left this turn (a second stone, or ply 0) |
| 5 | `stones_left_2` | the whole region when two are left |

Every plane is a function of what the order-free eval-cache key holds (stones, side, stones left), and the canvas
carries no history.

## The rules

- **Region.** The fence: stones ∪ the cells within `legal_move_radius` of a stone. The empty board's region is the
  engine's 5 × 5 first-stone box, q, r ∈ [−2, 2].
- **Boxes and buckets.** A pair's box is the region's extent along its two coordinates. A side's container side is
  `max(canvas_min_side, ⌈x / canvas_side_step⌉ · canvas_side_step)`; a pair's square bucket is its box's larger side's.
  `fit_side` is the smallest square bucket of the three.
- **Orientation, from the position alone.** The served pair is the lowest-id pair whose square bucket is at most
  `fit_side + canvas_side_step`. It reads the stone set only (not their order, the side to move or the stones left), so
  the order-free cache key stays valid and the trainer's draw (among the D6 elements whose square bucket equals the
  served one, L2b) and the server agree. The band makes flips between consecutive plies rarer than a strict minimum.
- **Far mode.** When `fit_side` exceeds `canvas_max_side` (S_max), the region becomes the stones ∪ the cells within
  `far_halo` of a stone, and the boxes, orientation and container are recomputed on it. A legal cell outside it keeps its
  place in `legal` with `legal_gather = FAR_CELL`: every legal cell has a prior in every mode (the far logit's
  `far − log n_far` decode is the serving path's). The trigger reads `fit_side`, not the served bucket, because a D6
  element permutes the three buckets: a trigger on the served one would put one orientation of a position in far mode
  and another not.
- **Exactness, bit for bit on the builder (a fixed-algorithm CPU path).**
  - D6: for every stone-bearing position S and each of the 12 elements g, `build(g·S)` holds at cell g·c exactly the
    planes `build(S)` holds at c, the legal set mapped by g, and the same `far` and `fit_side`; the container and pair
    may differ, being chosen per position. The empty board's box is closed under its 4-element subgroup only, which is
    why the sampler keeps an empty record at identity (as for the graph).
  - Translation: `build(S + t)` equals `build(S)` in every field but `legal` (shifted by t) and the reply frame.
  - Inertness: every cell outside the `fence` plane is 0 on every plane; `own`, `opponent` and `legal` lie inside it;
    the region touches row 0 and column 0.
  - The reply frame is the graph's: `legal` and `policy_dst_slot` equal the graph builder's legal nodes and policy
    scatter index for the same position, radius and trunk.
  - The net-output tolerances (1e-4 in fp32 across containers; 5e-3 in served bf16 policy probability across tiers and
    batch sizes, R394(c)) need a net and are the serving leg's pins.

## The root rule (`far_max_side`, S_far)

A ROOT whose canvas is in far mode with `fit_side` above `far_max_side` is refused; a leaf never is, and the serving
budget covers `far_max_side` plus the search's depth margin (a leaf exceeds its root by at most 8 cells a placement).
The cap OUTCOME exists on both sides — self-play's `ply_cap` termination with no z, and the arena's `TERMINAL_PLY_CAP`,
a draw unless `eval.ply_cap_adjudication` is armed — but no path reaches it from a refused root, since both loops end at
the cap by ply count alone (`runner/game.rs`, `arena/match.py`). The design, built with the serving path:

- self-play: the runner ends the game at that ply through its cap exit (termination `ply_cap`, its rows value-invalid,
  no z), and counts it in a LAW-18 row;
- arena and cells: the canvas player refuses the root by a named error, `_play_one_game` ends the game as
  `TERMINAL_PLY_CAP` scored as the cap's outcome, and the refusal is counted.

## Who asserts what where

| fact | asserted where | pinning test |
|---|---|---|
| the region, orientation, container, planes, gather and reply frame; far mode at `fit_side > canvas_max_side` | `crates/mantis-graph/src/canvas.rs` | `crates/mantis-graph/tests/canvas.rs`: `the_canvas_bytes_at_radius_eight_hash_to_the_pinned_digest`, `the_band_serves_the_lowest_pair_within_one_step_of_the_smallest`, `far_mode_keeps_every_stone_and_flags_the_legal_cells_outside_the_halo` |
| D6 exactness, the empty board's subgroup | `canvas.rs` | `the_canvas_of_the_rotated_board_is_the_rotation_of_the_canvas_under_all_twelve_elements`, `the_empty_board_canvas_is_invariant_under_exactly_the_order_four_subgroup` |
| container invariance: translation, inertness, the origin | `canvas.rs` | `a_translated_board_builds_the_same_canvas`, `every_cell_outside_the_fence_is_zero_on_every_plane` |
| the orientation reads the stone set alone | `canvas.rs` | `the_orientation_is_a_function_of_the_stone_set_alone` |
| the reply frame is the graph builder's | `canvas.rs` | `the_canvas_carries_the_graphs_reply_frame` |
| the legal set is mantis-core's, as a set and a count, across two producers that cannot collude | `_engine.build_canvas` (bridge) | `tests/model/conformance/test_legal_move_coverage_boundary.py` (the canvas arm) |
| the row's keys and their relations | `crates/mantis-encoding/src/spec/validate.rs` | `crates/mantis-encoding/tests/registry_census.rs` |
| no graph-building path admits a canvas row | `mantis-selfplay` `runner/params.rs::resolve_geometry`, `mantis.selfplay.hparams.is_graph_representation` | `crates/mantis-selfplay/src/runner/params.rs` (tests), `tests/selfplay/test_pool_encoding_arms.py` |
