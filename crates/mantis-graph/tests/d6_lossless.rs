//! R358(b): build(rot_s(board)) IS rot_s(build(board)) under all 12 D6 elements at radius 8 (RESEARCH-STRENGTH-1 App. A, pinned).

use std::collections::{BTreeMap, BTreeSet};

use mantis_graph::{
    build_axis_graph, AxisGraph, BuildParams, StoneList, EDGE_FEAT_DIM, NODE_FEAT_DIM,
};

const R8: BuildParams = BuildParams {
    win_length: 6,
    radius: 8,
    current_player: 1,
    moves_remaining: 2,
    trunk_size: 19,
};
const N_SYMS: usize = 12;
/// The elements under which the empty board's fixed 5×5 axial square is invariant.
const EMPTY_BOARD_INVARIANT: [usize; 4] = [0, 3, 6, 9];
const N_POS: usize = 24;

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
    let dq = (b.0 - a.0).abs();
    let dr = (b.1 - a.1).abs();
    let ds = (b.0 + b.1 - a.0 - a.1).abs();
    dq.max(dr).max(ds)
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

fn coords(g: &AxisGraph) -> Vec<(i32, i32)> {
    (0..g.num_nodes())
        .map(|i| (g.node_coords[2 * i], g.node_coords[2 * i + 1]))
        .collect()
}

fn stone_set(g: &AxisGraph) -> BTreeSet<(i32, i32, i8)> {
    let c = coords(g);
    (0..g.n_stones as usize)
        .map(|i| {
            let own = g.node_feat.0[i * NODE_FEAT_DIM] == 1.0;
            (c[i].0, c[i].1, if own { 1 } else { -1 })
        })
        .collect()
}

fn legal_set(g: &AxisGraph) -> BTreeSet<(i32, i32)> {
    let c = coords(g);
    (0..g.num_nodes())
        .filter(|&i| g.legal_mask[i])
        .map(|i| c[i])
        .collect()
}

type EdgeMap = BTreeMap<((i32, i32), (i32, i32)), (usize, i32, i32)>;

/// Real edges keyed by (src coord, dst coord) → (axis, signed_dist, src_player); dummies skipped.
fn edge_map(g: &AxisGraph) -> EdgeMap {
    let c = coords(g);
    let mut m = BTreeMap::new();
    for e in 0..g.edge_index.src.len() {
        let (s, d) = (g.edge_index.src[e] as usize, g.edge_index.dst[e] as usize);
        let a = &g.edge_attr.0[e * EDGE_FEAT_DIM..(e + 1) * EDGE_FEAT_DIM];
        let Some(axis) = (0..3).find(|&k| a[k] == 1.0) else {
            continue;
        };
        m.insert((c[s], c[d]), (axis, a[3] as i32, a[4] as i32));
    }
    m
}

fn rotate_stones(stones: &[(i32, i32, i8)], s: usize) -> Vec<(i32, i32, i8)> {
    stones
        .iter()
        .map(|&(q, r, p)| {
            let (a, b) = rotate_axial(q, r, s);
            (a, b, p)
        })
        .collect()
}

fn random_position(rng: &mut Lcg) -> Vec<(i32, i32, i8)> {
    let n_st = rng.range(1, 61) as usize;
    let mut set = BTreeSet::new();
    while set.len() < n_st {
        set.insert((rng.range(-14, 15) as i32, rng.range(-14, 15) as i32));
    }
    set.iter()
        .enumerate()
        .map(|(i, &(q, r))| (q, r, if i.is_multiple_of(2) { 1 } else { -1 }))
        .collect()
}

fn axis_spread(st: &[(i32, i32, i8)]) -> f64 {
    let n = st.len() as f64;
    let cq = st.iter().map(|s| f64::from(s.0)).sum::<f64>() / n;
    let cr = st.iter().map(|s| f64::from(s.1)).sum::<f64>() / n;
    st.iter()
        .map(|s| (f64::from(s.0) - cq).abs().max((f64::from(s.1) - cr).abs()))
        .fold(0.0, f64::max)
        .max(1.0)
}

#[test]
fn build_of_the_rotated_board_is_the_rotation_of_the_build_under_all_twelve_elements() {
    let mut rng = Lcg(20_260_918);
    let mut axis_maps: Vec<BTreeSet<(usize, usize, i32)>> = vec![BTreeSet::new(); N_SYMS];
    let mut total_edges = 0usize;
    for _ in 0..N_POS {
        let stones = random_position(&mut rng);
        let g0 = build_axis_graph(
            &StoneList {
                stones: stones.clone(),
            },
            &R8,
        );
        let (s0, l0, e0) = (stone_set(&g0), legal_set(&g0), edge_map(&g0));
        total_edges += e0.len();
        for (s, axis_map) in axis_maps.iter_mut().enumerate() {
            let g1 = build_axis_graph(
                &StoneList {
                    stones: rotate_stones(&stones, s),
                },
                &R8,
            );
            let s0r: BTreeSet<_> = s0
                .iter()
                .map(|&(q, r, p)| {
                    let (a, b) = rotate_axial(q, r, s);
                    (a, b, p)
                })
                .collect();
            assert_eq!(
                stone_set(&g1),
                s0r,
                "sym {s}: stone node set is not the rotated stone set"
            );
            let l0r: BTreeSet<_> = l0.iter().map(|&(q, r)| rotate_axial(q, r, s)).collect();
            assert_eq!(
                legal_set(&g1),
                l0r,
                "sym {s}: the fence is not the rotated fence — a cell dropped or added"
            );
            assert_eq!(g1.num_nodes(), g0.num_nodes(), "sym {s}: node count");
            let e1 = edge_map(&g1);
            assert_eq!(e1.len(), e0.len(), "sym {s}: edge count");
            for (&(a, b), &(ax, sd, sp)) in &e0 {
                let key = (rotate_axial(a.0, a.1, s), rotate_axial(b.0, b.1, s));
                let &(ax1, sd1, sp1) = e1
                    .get(&key)
                    .unwrap_or_else(|| panic!("sym {s}: edge {a:?}->{b:?} missing after rotation"));
                assert_eq!(sd.abs(), sd1.abs(), "sym {s}: |signed_dist| changed");
                assert_eq!(sp, sp1, "sym {s}: src_player changed");
                assert_eq!(
                    hexd(a, b),
                    hexd(key.0, key.1),
                    "sym {s}: hex distance changed"
                );
                axis_map.insert((ax, ax1, sd.signum() * sd1.signum()));
            }
        }
    }
    assert!(
        total_edges > 0,
        "the positions built no edge; the test measured nothing"
    );
    for (s, map) in axis_maps.iter().enumerate() {
        // ONE permutation of the three axes, each with ONE sign: three entries, distinct in and out.
        assert_eq!(
            map.len(),
            3,
            "sym {s}: the axis map is not a single signed permutation: {map:?}"
        );
        let ins: BTreeSet<_> = map.iter().map(|m| m.0).collect();
        let outs: BTreeSet<_> = map.iter().map(|m| m.1).collect();
        assert_eq!((ins.len(), outs.len()), (3, 3), "sym {s}: {map:?}");
    }
    assert_eq!(
        axis_maps[0].iter().collect::<Vec<_>>(),
        vec![&(0, 0, 1), &(1, 1, 1), &(2, 2, 1)],
        "the identity element must map every axis to itself with sign +1"
    );
}

#[test]
fn the_coordinate_spread_is_equivariant_under_the_order_four_subgroup_only() {
    // Not a loss (deploy sees the same features on the rotated board) but why the net must LEARN
    // the other eight elements: the `norm_q`/`norm_r` normaliser changes under them.
    let mut rng = Lcg(20_260_918);
    let mut changed = [0usize; N_SYMS];
    for _ in 0..N_POS {
        let stones = random_position(&mut rng);
        let sp0 = axis_spread(&stones);
        for (s, count) in changed.iter_mut().enumerate() {
            if (axis_spread(&rotate_stones(&stones, s)) - sp0).abs() > 1e-9 {
                *count += 1;
            }
        }
    }
    for s in EMPTY_BOARD_INVARIANT {
        assert_eq!(
            changed[s], 0,
            "sym {s} is in the exact subgroup and changed the spread"
        );
    }
    let others: Vec<usize> = (0..N_SYMS)
        .filter(|s| !EMPTY_BOARD_INVARIANT.contains(s))
        .collect();
    assert!(
        others.iter().all(|&s| changed[s] > 0),
        "an element outside the subgroup never changed the spread over {N_POS} positions: {changed:?}"
    );
}

#[test]
fn the_empty_board_fence_is_invariant_under_exactly_the_order_four_subgroup() {
    let le = legal_set(&build_axis_graph(&StoneList::default(), &R8));
    assert_eq!(le.len(), 25, "the empty board's fixed axial square");
    let invariant: Vec<usize> = (0..N_SYMS)
        .filter(|&s| {
            le.iter()
                .map(|&(q, r)| rotate_axial(q, r, s))
                .collect::<BTreeSet<_>>()
                == le
        })
        .collect();
    assert_eq!(invariant, EMPTY_BOARD_INVARIANT.to_vec());
}
