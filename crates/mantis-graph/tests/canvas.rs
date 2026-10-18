// Exceeds the 300-line soft cap (R8): every pin of the builder reads one corpus through one cell map and one inverse
// layout; split, each file would re-derive the map its assertions compare against.

//! The canvas builder's pins (contract #12): goldens, D6 and container exactness, the reply frame, the band, far mode.

#[path = "common/positions.rs"]
mod positions;

use std::collections::{BTreeMap, BTreeSet};

use mantis_graph::{
    build_axis_graph, build_canvas, pair_coords, BuildParams, Canvas, CanvasParams, EmptyEdges,
    StoneList, CANVAS_PLANES, FAR_CELL,
};
use sha2::{Digest, Sha256};

/// The `hexcanvas_r8` row's keys: radius 8, the trunk-19 reply frame, sides ×8 from 24, S_max 112, halo 4.
const R8: CanvasParams = CanvasParams {
    radius: 8,
    trunk_size: 19,
    side_step: 8,
    min_side: 24,
    max_side: 112,
    far_halo: 4,
    current_player: 1,
    moves_remaining: 2,
};
/// The same row at S_max 24, so every position wider than a lone stone's fence builds in far mode.
const FAR: CanvasParams = CanvasParams { max_side: 24, ..R8 };
const N_SYMS: usize = 12;
/// The elements under which the empty board's 5×5 axial square is invariant.
const EMPTY_BOARD_INVARIANT: [usize; 4] = [0, 3, 6, 9];
const N_PLANES: usize = CANVAS_PLANES.len();

/// sha256 over every recorded position's canvas bytes in file order, at R8 and at FAR.
const R8_CANVAS_SHA256: &str = "3fb13f45414036be1a3cb419869fe4ee5120d7fc7c780e4802ed4dde1779351c";
const FAR_CANVAS_SHA256: &str = "3d7b5e715f5f889a7b71a0834b947e13c18e18cb87abdeb7c3cd5fa3c970c005";

/// The same 12 elements as `mantis_selfplay::replay::sym::rotate_axial` (reflect-then-rotate).
fn rotate_axial(q: i32, r: i32, s: usize) -> (i32, i32) {
    let (mut q, mut r) = if s >= 6 { (r, q) } else { (q, r) };
    for _ in 0..(s % 6) {
        let nq = -r;
        let nr = q + r;
        q = nq;
        r = nr;
    }
    (q, r)
}

fn hexd(a: (i32, i32), b: (i32, i32)) -> i32 {
    let (dq, dr) = (b.0 - a.0, b.1 - a.1);
    dq.abs().max(dr.abs()).max((dq + dr).abs())
}

/// The axial cell of a pair coordinate, the inverse of `pair_coords`.
fn axial(pair: u8, a: i32, b: i32) -> (i32, i32) {
    match pair {
        0 => (a, b),
        1 => (-a - b, a),
        2 => (b, -a - b),
        p => panic!("axis pair {p} is not one of 0, 1, 2"),
    }
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }
    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() as i64 % (hi - lo))
    }
}

/// Seeded positions: compact, spread (band cases) and wide (far mode at R8 too).
fn random_positions() -> Vec<Vec<(i32, i32, i8)>> {
    let mut rng = Lcg(20_261_018);
    let mut out = Vec::new();
    for spread in [6i64, 14, 30, 60] {
        for _ in 0..12 {
            let n = rng.range(1, 41) as usize;
            let mut set = BTreeSet::new();
            while set.len() < n {
                set.insert((
                    rng.range(-spread, spread + 1) as i32,
                    rng.range(-spread, spread + 1) as i32,
                ));
            }
            out.push(
                set.iter()
                    .enumerate()
                    .map(|(i, &(q, r))| (q, r, if i.is_multiple_of(2) { 1 } else { -1 }))
                    .collect(),
            );
        }
    }
    out
}

/// A recorded position: its stones, the side to move, the stones it still places this turn.
type Recorded = (Vec<(i32, i32, i8)>, i8, u8);

fn recorded() -> Vec<Recorded> {
    positions::read_positions_r8()
        .unwrap_or_else(|e| panic!("{e}"))
        .into_iter()
        .map(|p| (p.stones, p.to_move, p.moves_remaining))
        .collect()
}

fn build(stones: &[(i32, i32, i8)], params: &CanvasParams) -> Canvas {
    build_canvas(
        &StoneList {
            stones: stones.to_vec(),
        },
        params,
    )
}

/// Every fence cell's six plane values keyed by its axial coordinate; panics on a lit cell outside the fence.
fn cell_map(c: &Canvas) -> BTreeMap<(i32, i32), [u8; N_PLANES]> {
    let area = c.height * c.width;
    assert_eq!(
        c.planes.len(),
        N_PLANES * area,
        "planes are [6, height, width]"
    );
    let mut m = BTreeMap::new();
    for i in 0..area {
        let v: [u8; N_PLANES] = std::array::from_fn(|k| c.planes[k * area + i]);
        if v[3] == 1 {
            let (row, col) = ((i / c.width) as i32, (i % c.width) as i32);
            m.insert(axial(c.axis_pair, c.origin.0 + row, c.origin.1 + col), v);
        } else {
            assert_eq!(
                v, [0; N_PLANES],
                "cell {i} outside the fence is not zero on every plane"
            );
        }
    }
    m
}

fn canvas_bytes(c: &Canvas) -> Vec<u8> {
    let mut v = vec![c.axis_pair, u8::from(c.far)];
    for x in [c.height, c.width, c.fit_side, c.legal.len()] {
        v.extend((x as u32).to_le_bytes());
    }
    for x in [c.origin.0, c.origin.1, c.window_center.0, c.window_center.1] {
        v.extend(x.to_le_bytes());
    }
    v.extend(&c.planes);
    for &(q, r) in &c.legal {
        v.extend(q.to_le_bytes());
        v.extend(r.to_le_bytes());
    }
    for &x in c.legal_gather.iter().chain(&c.policy_dst_slot) {
        v.extend(x.to_le_bytes());
    }
    v
}

fn digest(params: &CanvasParams) -> String {
    let mut h = Sha256::new();
    for (stones, to_move, left) in recorded() {
        let p = CanvasParams {
            current_player: to_move,
            moves_remaining: left,
            ..*params
        };
        h.update(canvas_bytes(&build(&stones, &p)));
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn the_canvas_bytes_at_radius_eight_hash_to_the_pinned_digest() {
    assert_eq!(recorded().len(), positions::POSITIONS_R8_COUNT);
    assert_eq!(
        digest(&R8),
        R8_CANVAS_SHA256,
        "the canvas builder's bytes moved at S_max 112"
    );
    assert_eq!(
        digest(&FAR),
        FAR_CANVAS_SHA256,
        "the canvas builder's far-mode bytes moved"
    );
}

/// Per stone-bearing position and element: the rotated board's canvas holds at g·c what the board's holds at c.
fn assert_d6_exact(stones: &[(i32, i32, i8)], params: &CanvasParams, label: &str) {
    let c0 = build(stones, params);
    let (m0, l0) = (
        cell_map(&c0),
        c0.legal.iter().copied().collect::<BTreeSet<_>>(),
    );
    let far0: BTreeSet<(i32, i32)> = c0
        .legal
        .iter()
        .zip(&c0.legal_gather)
        .filter(|&(_, &g)| g == FAR_CELL)
        .map(|(&c, _)| c)
        .collect();
    for s in 0..N_SYMS {
        let rotated: Vec<_> = stones
            .iter()
            .map(|&(q, r, p)| {
                let (a, b) = rotate_axial(q, r, s);
                (a, b, p)
            })
            .collect();
        let c1 = build(&rotated, params);
        let want: BTreeMap<_, _> = m0
            .iter()
            .map(|(&(q, r), &v)| (rotate_axial(q, r, s), v))
            .collect();
        assert_eq!(
            cell_map(&c1),
            want,
            "{label} sym {s}: the planes are not the rotated planes"
        );
        let l1: BTreeSet<_> = c1.legal.iter().copied().collect();
        assert_eq!(
            l1,
            l0.iter().map(|&(q, r)| rotate_axial(q, r, s)).collect(),
            "{label} sym {s}: legal set"
        );
        let far1: BTreeSet<(i32, i32)> = c1
            .legal
            .iter()
            .zip(&c1.legal_gather)
            .filter(|&(_, &g)| g == FAR_CELL)
            .map(|(&c, _)| c)
            .collect();
        assert_eq!(
            far1,
            far0.iter().map(|&(q, r)| rotate_axial(q, r, s)).collect(),
            "{label} sym {s}: far cells"
        );
        assert_eq!(
            (c1.far, c1.fit_side),
            (c0.far, c0.fit_side),
            "{label} sym {s}: far mode or size class moved"
        );
    }
}

#[test]
fn the_canvas_of_the_rotated_board_is_the_rotation_of_the_canvas_under_all_twelve_elements() {
    let (mut far_cases, mut band_cases) = (0usize, 0usize);
    let mut all = recorded()
        .into_iter()
        .map(|(st, _, _)| st)
        .collect::<Vec<_>>();
    all.extend(random_positions());
    for (i, stones) in all.iter().filter(|st| !st.is_empty()).enumerate() {
        for (params, mode) in [(&R8, "r8"), (&FAR, "far")] {
            assert_d6_exact(stones, params, &format!("position {i} {mode}"));
            let c = build(stones, params);
            far_cases += usize::from(c.far);
            band_cases += usize::from(c.height.max(c.width) > c.fit_side);
        }
    }
    assert!(
        far_cases > 0 && band_cases > 0,
        "no far ({far_cases}) or band ({band_cases}) case was built"
    );
}

#[test]
fn the_empty_board_canvas_is_invariant_under_exactly_the_order_four_subgroup() {
    let c = build(&[], &R8);
    let m = cell_map(&c);
    assert_eq!(
        (m.len(), c.legal.len(), c.height, c.width),
        (25, 25, 24, 24),
        "the 5×5 first-stone box"
    );
    let invariant: Vec<usize> = (0..N_SYMS)
        .filter(|&s| {
            m.keys()
                .map(|&(q, r)| rotate_axial(q, r, s))
                .collect::<BTreeSet<_>>()
                == m.keys().copied().collect()
        })
        .collect();
    assert_eq!(invariant, EMPTY_BOARD_INVARIANT.to_vec());
}

#[test]
fn a_translated_board_builds_the_same_canvas() {
    for stones in random_positions()
        .iter()
        .chain(recorded().iter().map(|(st, _, _)| st).take(64))
    {
        for params in [&R8, &FAR] {
            let c0 = build(stones, params);
            for t in [(1, 0), (-3, 7), (11, -5), (-40, -41)] {
                let moved: Vec<_> = stones
                    .iter()
                    .map(|&(q, r, p)| (q + t.0, r + t.1, p))
                    .collect();
                let c1 = build(&moved, params);
                let shift = pair_coords(t.0, t.1)[usize::from(c0.axis_pair)];
                assert_eq!(
                    (c1.axis_pair, c1.height, c1.width, c1.far, c1.fit_side),
                    (c0.axis_pair, c0.height, c0.width, c0.far, c0.fit_side),
                    "translation by {t:?} moved the container"
                );
                assert_eq!(
                    c1.origin,
                    (c0.origin.0 + shift.0, c0.origin.1 + shift.1),
                    "origin under {t:?}"
                );
                assert_eq!(
                    (&c1.planes, &c1.legal_gather),
                    (&c0.planes, &c0.legal_gather),
                    "arrays under {t:?}"
                );
                let legal: Vec<_> = c0.legal.iter().map(|&(q, r)| (q + t.0, r + t.1)).collect();
                assert_eq!(c1.legal, legal, "legal cells under {t:?}");
            }
        }
    }
}

#[test]
fn every_cell_outside_the_fence_is_zero_on_every_plane() {
    let mut all = recorded();
    all.extend(random_positions().into_iter().map(|st| (st, -1, 1)));
    all.push((Vec::new(), 1, 1));
    for (stones, to_move, left) in &all {
        for params in [&R8, &FAR] {
            let p = CanvasParams {
                current_player: *to_move,
                moves_remaining: *left,
                ..*params
            };
            let c = build(stones, &p);
            let m = cell_map(&c);
            let fence_rows = m
                .keys()
                .map(|&(q, r)| pair_coords(q, r)[usize::from(c.axis_pair)]);
            let (min_a, min_b) =
                fence_rows.fold((i32::MAX, i32::MAX), |(a, b), (x, y)| (a.min(x), b.min(y)));
            assert_eq!(
                (min_a, min_b),
                c.origin,
                "the fence's minimum corner is not cell (0, 0)"
            );
            assert!(
                c.height.is_multiple_of(8)
                    && c.width.is_multiple_of(8)
                    && c.height >= 24
                    && c.width >= 24,
                "container sides"
            );
            for (&cell, v) in &m {
                let stone = stones.iter().find(|s| (s.0, s.1) == cell);
                let own = stone.is_some_and(|s| s.2 == if *to_move == 1 { 1 } else { -1 });
                assert_eq!(
                    (v[0], v[1]),
                    (u8::from(own), u8::from(stone.is_some() && !own)),
                    "{cell:?} stones"
                );
                assert_eq!(
                    (v[4], v[5]),
                    (u8::from(*left == 1), u8::from(*left == 2)),
                    "{cell:?} stones left"
                );
                assert!(
                    stone.is_some() || v[2] == 1,
                    "{cell:?} is in the fence but neither a stone nor legal"
                );
            }
            for (&(q, r), &g) in c.legal.iter().zip(&c.legal_gather) {
                let near = stones
                    .iter()
                    .map(|s| hexd((q, r), (s.0, s.1)))
                    .min()
                    .unwrap_or(1);
                assert_eq!(
                    g == FAR_CELL,
                    c.far && near > 4,
                    "{:?}: far flag at distance {near}",
                    (q, r)
                );
                if g != FAR_CELL {
                    let (row, col) = (g as usize / c.width, g as usize % c.width);
                    let at = axial(
                        c.axis_pair,
                        c.origin.0 + row as i32,
                        c.origin.1 + col as i32,
                    );
                    assert_eq!(at, (q, r), "legal_gather points at another cell");
                    assert_eq!(
                        m.get(&at).map(|v| v[2]),
                        Some(1),
                        "{at:?} is gathered but not legal"
                    );
                }
            }
            let lit_legal = m.values().filter(|v| v[2] == 1).count();
            assert_eq!(
                lit_legal,
                c.legal_gather.iter().filter(|&&g| g != FAR_CELL).count(),
                "legal plane count"
            );
        }
    }
}

#[test]
fn the_canvas_carries_the_graphs_reply_frame() {
    for (stones, to_move, left) in recorded() {
        let gp = BuildParams {
            radius: 8,
            current_player: to_move,
            moves_remaining: left,
            empty_edges: EmptyEdges::Kept,
            ..BuildParams::V1_GEOMETRY
        };
        let g = build_axis_graph(
            &StoneList {
                stones: stones.clone(),
            },
            &gp,
        );
        let c = build(
            &stones,
            &CanvasParams {
                current_player: to_move,
                moves_remaining: left,
                ..R8
            },
        );
        let graph_legal: Vec<(i32, i32)> = g
            .legal_node_gather
            .iter()
            .map(|&n| {
                (
                    g.node_coords[2 * n as usize],
                    g.node_coords[2 * n as usize + 1],
                )
            })
            .collect();
        assert_eq!(
            c.legal, graph_legal,
            "the canvas's legal order is not the graph's"
        );
        assert_eq!(
            c.policy_dst_slot, g.policy_scatter_index.0,
            "the reply slots are not the graph's"
        );
        assert_eq!(
            c.window_center, g.window_center,
            "the reply window's centre is not the graph's"
        );
    }
}

#[test]
fn the_orientation_is_a_function_of_the_stone_set_alone() {
    for stones in random_positions() {
        let c0 = build(&stones, &R8);
        let mut reversed = stones.clone();
        reversed.reverse();
        for (st, player, left) in [(stones.clone(), -1, 1), (reversed, 1, 2)] {
            let c1 = build(
                &st,
                &CanvasParams {
                    current_player: player,
                    moves_remaining: left,
                    ..R8
                },
            );
            assert_eq!(
                (
                    c1.axis_pair,
                    c1.height,
                    c1.width,
                    c1.origin,
                    &c1.legal_gather
                ),
                (
                    c0.axis_pair,
                    c0.height,
                    c0.width,
                    c0.origin,
                    &c0.legal_gather
                ),
                "the container moved with the stone order, the side to move or the stones left"
            );
            let area = c0.height * c0.width;
            let (f0, f1) = (
                &c0.planes[3 * area..4 * area],
                &c1.planes[3 * area..4 * area],
            );
            assert_eq!(f0, f1, "the fence moved");
        }
    }
}

#[test]
fn the_band_serves_the_lowest_pair_within_one_step_of_the_smallest() {
    let shape = |c: &Canvas| (c.axis_pair, c.height, c.width, c.fit_side, c.far);
    // A lone stone: a 17-wide fence in every pair, served in (q, r) at the minimum side.
    assert_eq!(shape(&build(&[(0, 0, 1)], &R8)), (0, 24, 24, 24, false));
    // Extents q 27, r 37, s 27: buckets 40, 40, 32. Pair 0 is one step over the smallest, inside the band.
    assert_eq!(
        shape(&build(&[(0, 0, 1), (10, -20, -1)], &R8)),
        (0, 32, 40, 32, false)
    );
    // Extents q 63, r 40, s 40: buckets 64, 40, 64. Pair 0 is three steps over: pair 1 (r, s) serves.
    assert_eq!(
        shape(&build(&[(0, 0, 1), (46, -23, -1)], &R8)),
        (1, 40, 40, 40, false)
    );
}

#[test]
fn far_mode_keeps_every_stone_and_flags_the_legal_cells_outside_the_halo() {
    // Extents q 29, r 17, s 29: the fence's smallest bucket is 32, past FAR's S_max 24.
    let stones = [(0, 0, 1), (12, 0, -1)];
    let (near, far) = (build(&stones, &R8), build(&stones, &FAR));
    assert!(
        !near.far && far.far,
        "far mode fires past S_max and not below it"
    );
    assert_eq!(
        (far.axis_pair, far.height, far.width, far.fit_side),
        (0, 24, 24, 24),
        "the halo's box"
    );
    let m = cell_map(&far);
    assert!(
        stones
            .iter()
            .all(|s| m.get(&(s.0, s.1)).is_some_and(|v| v[0] + v[1] == 1)),
        "a stone left the canvas"
    );
    assert_eq!(
        far.legal, near.legal,
        "far mode dropped a legal cell from the reply"
    );
    let in_halo = far
        .legal
        .iter()
        .filter(|&&c| stones.iter().any(|s| hexd(c, (s.0, s.1)) <= 4))
        .count();
    assert_eq!(
        far.legal_gather.iter().filter(|&&g| g != FAR_CELL).count(),
        in_halo,
        "the halo's legal cells"
    );
    assert!(
        far.legal_gather.contains(&FAR_CELL),
        "no legal cell lies outside the halo"
    );
}
