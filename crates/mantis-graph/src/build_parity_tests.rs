//! The builder against the two-pass algorithm it replaced, every field byte for byte, and its verify against the per-edge verify.
//! >300 lines: the replaced algorithm is the reference, kept whole beside the tests that compare against it.

use std::panic::{catch_unwind, AssertUnwindSafe};

use super::*;

#[path = "../tests/common/positions.rs"]
mod positions;

/// The replaced builder: a hash legal set, a nearest-stone scan per legal cell, emit-then-dedup.
fn two_pass_build(stones_in: &StoneList, params: &BuildParams) -> AxisGraph {
    assert!(
        (1..=32).contains(&params.win_length),
        "BuildParams: win_length {} outside supported 1..=32 (threat cells buffer bound)",
        params.win_length
    );
    assert!(
        params.trunk_size >= 1,
        "BuildParams: trunk_size {} < 1",
        params.trunk_size
    );
    let win_length = params.win_length as usize;
    let radius = i32::from(params.radius);
    let window = (params.win_length - 1) as usize;
    let cur = params.current_player;

    let mut stone_map: FnvMap<i64, i8> =
        FnvMap::with_capacity_and_hasher(stones_in.stones.len(), BuildHasherDefault::default());
    for &(q, r, p) in &stones_in.stones {
        stone_map.insert(pack(q, r), p);
    }
    let mut stones: Vec<(i32, i32, i8)> = stone_map
        .iter()
        .map(|(&k, &p)| ((k >> 32) as i32, k as u32 as i32, p))
        .collect();
    stones.sort_unstable_by_key(|&(q, r, _)| (q, r));
    let legal = legal_moves_from_stones(&stone_map, &stones, radius);

    let player_feat: f32 = if cur == 1 { 1.0 } else { -1.0 };
    let own_is_p1 = player_feat > 0.0;
    let moves_feat: f32 = (f64::from(params.moves_remaining) / 2.0) as f32;
    let to_move: i8 = if cur == 1 { 1 } else { -1 };

    let n_stones = stones.len();
    let n_legal = legal.len();
    let n_real = n_stones + n_legal;
    let n = n_real + 1;
    assert!(
        n < (1 << 30),
        "node count {n} exceeds the 30-bit dedup key budget"
    );
    let dummy_idx = n_real as u32;
    let fdim = NODE_FEAT_DIM;

    let mut coords: Vec<i32> = Vec::with_capacity(n * 2);
    let mut node_kind: Vec<Kind> = Vec::with_capacity(n_real);
    let mut coord_to_idx: FnvMap<i64, u32> =
        FnvMap::with_capacity_and_hasher(n_real, BuildHasherDefault::default());
    for (i, &(q, r, p)) in stones.iter().enumerate() {
        coords.push(q);
        coords.push(r);
        coord_to_idx.insert(pack(q, r), i as u32);
        node_kind.push(Kind::Stone(p));
    }
    for (j, &(q, r)) in legal.iter().enumerate() {
        let idx = (n_stones + j) as u32;
        coords.push(q);
        coords.push(r);
        coord_to_idx.insert(pack(q, r), idx);
        node_kind.push(Kind::Empty);
    }
    coords.push(0);
    coords.push(0);
    let coord_index = CoordIndex::build(&coords, n_real);

    let (cq, cr, spread): (f64, f64, f64) = if n_stones > 0 {
        let mut sq = 0f64;
        let mut sr = 0f64;
        for &(q, r, _) in &stones {
            sq += f64::from(q);
            sr += f64::from(r);
        }
        let cq = sq / n_stones as f64;
        let cr = sr / n_stones as f64;
        let mut max_dev = 0f64;
        for &(q, r, _) in &stones {
            max_dev = max_dev.max((f64::from(q) - cq).abs().max((f64::from(r) - cr).abs()));
        }
        (cq, cr, max_dev.max(1.0))
    } else {
        (0.0, 0.0, 1.0)
    };

    let mut features = vec![0f32; n * fdim];
    let (l_own, l_opp, l_empty, l_moves, l_norm_q, l_norm_r, l_inv) = (0, 1, 2, 3, 4, 5, 6);
    let set_coords = |features: &mut [f32], base: usize, q: i32, r: i32| {
        features[base + l_norm_q] = ((f64::from(q) - cq) / spread) as f32;
        features[base + l_norm_r] = ((f64::from(r) - cr) / spread) as f32;
    };
    for (i, &(q, r, p)) in stones.iter().enumerate() {
        let base = i * fdim;
        let col = if (p == 1) == own_is_p1 { l_own } else { l_opp };
        features[base + col] = 1.0;
        features[base + l_moves] = moves_feat;
        set_coords(&mut features, base, q, r);
    }
    for (j, &(q, r)) in legal.iter().enumerate() {
        let idx = n_stones + j;
        let base = idx * fdim;
        features[base + l_empty] = 1.0;
        features[base + l_moves] = moves_feat;
        set_coords(&mut features, base, q, r);
        let min_d = if stones.is_empty() {
            1
        } else {
            let mut m = i32::MAX;
            for &(sq, sr, _) in &stones {
                m = m.min(hex_distance((q, r), (sq, sr)));
            }
            m
        };
        features[base + l_inv] = (1.0f64 / f64::from(min_d.max(1))) as f32;
    }
    let dummy_base = (dummy_idx as usize) * fdim;
    features[dummy_base + l_moves] = moves_feat;

    let mut legal_mask = vec![false; n];
    let mut stone_mask = vec![false; n];
    for m in stone_mask.iter_mut().take(n_stones) {
        *m = true;
    }
    for j in 0..n_legal {
        legal_mask[n_stones + j] = true;
    }

    let cap = n_real * 3 * 2 * window * 2 + n_real * 2;
    let mut edge_src: Vec<u32> = Vec::with_capacity(cap);
    let mut edge_dst: Vec<u32> = Vec::with_capacity(cap);
    let mut edge_attr: Vec<f32> = Vec::with_capacity(cap * EDGE_FEAT_DIM);
    let mut edge_key: Vec<u32> = Vec::with_capacity(cap);
    for i in 0..n_real {
        let iq = coords[i * 2];
        let ir = coords[i * 2 + 1];
        let i_kind = node_kind[i];
        let src_i = i_kind.player_feat();
        for (axis_idx, &(dq, dr)) in WIN_AXES.iter().enumerate() {
            for sign in [1i32, -1i32] {
                let sdq = dq * sign;
                let sdr = dr * sign;
                for d in 1..=(window as i32) {
                    let tq = iq + sdq * d;
                    let tr = ir + sdr * d;
                    let hit = match coord_index {
                        Some(ref ix) => ix.get(tq, tr),
                        None => coord_to_idx.get(&pack(tq, tr)).copied(),
                    };
                    let Some(j) = hit else {
                        break;
                    };
                    let j_kind = node_kind[j as usize];
                    let src_j = j_kind.player_feat();
                    let signed_dist = (d * sign) as f32;
                    edge_src.push(i as u32);
                    edge_dst.push(j);
                    push_attr(&mut edge_attr, axis_idx, signed_dist, src_i);
                    edge_src.push(j);
                    edge_dst.push(i as u32);
                    push_attr(&mut edge_attr, axis_idx, -signed_dist, src_j);
                    let sbit = u32::from(sign < 0);
                    let wd = window as u32;
                    let base = axis_idx as u32 * 2 * wd + (d - 1) as u32;
                    edge_key.push(i as u32 * DEDUP_STRIDE_AXES * wd + base + sbit * wd);
                    edge_key.push(j * DEDUP_STRIDE_AXES * wd + base + (1 - sbit) * wd);
                    let should_stop = match i_kind {
                        Kind::Stone(ip) => matches!(j_kind, Kind::Stone(jp) if jp != ip),
                        Kind::Empty => matches!(j_kind, Kind::Stone(_)),
                    };
                    if should_stop {
                        break;
                    }
                }
            }
        }
    }
    two_pass_dedup(
        &mut edge_src,
        &mut edge_dst,
        &mut edge_attr,
        &edge_key,
        n_real,
        window,
    );
    for i in 0..n_real as u32 {
        edge_src.push(dummy_idx);
        edge_dst.push(i);
        edge_attr.extend_from_slice(&[0.0; EDGE_FEAT_DIM]);
        edge_src.push(i);
        edge_dst.push(dummy_idx);
        edge_attr.extend_from_slice(&[0.0; EDGE_FEAT_DIM]);
    }

    let stone_index = StoneIndex::build(&stones, i32::from(params.win_length) - 1);
    for idx in 0..n_real {
        let c = (coords[idx * 2], coords[idx * 2 + 1]);
        let tf = match stone_index {
            Some(ref sidx) => node_threat_features_indexed(sidx, c, to_move, win_length),
            None => node_threat_features(&stone_map, c, to_move, win_length),
        };
        let base = idx * fdim + BASE_DIM;
        for (k, &v) in tf.iter().enumerate() {
            features[base + k] = v as f32;
        }
    }

    let wc = window_center(&stones);
    let mut policy_scatter_index: Vec<i32> = Vec::with_capacity(n_legal);
    let mut legal_node_gather: Vec<u32> = Vec::with_capacity(n_legal);
    for (j, &(q, r)) in legal.iter().enumerate() {
        legal_node_gather.push((n_stones + j) as u32);
        policy_scatter_index.push(window_flat_idx(q, r, wc.0, wc.1, params.trunk_size));
    }

    let g = AxisGraph {
        node_feat: NodeFeat(features),
        edge_index: EdgeIndex {
            src: edge_src,
            dst: edge_dst,
        },
        edge_attr: EdgeAttr(edge_attr),
        legal_mask,
        stone_mask,
        policy_scatter_index: PolicyScatterIndex(policy_scatter_index),
        node_coords: coords,
        legal_node_gather,
        n_stones: n_stones as u16,
        n_nodes_checksum: n as u32,
        window_center: wc,
        current_player: to_move,
        builder_impl: BUILDER_IMPL_NATIVE,
    };
    per_edge_verify(&g, n_stones, n_legal, params);
    g
}

/// The replaced dedup pass: keep each key's first occurrence, compacting the three arrays in place.
fn two_pass_dedup(
    src: &mut Vec<u32>,
    dst: &mut Vec<u32>,
    attr: &mut Vec<f32>,
    key_of: &[u32],
    n_real: usize,
    window: usize,
) {
    let e = src.len();
    let bits = n_real
        .saturating_mul(DEDUP_STRIDE_AXES as usize)
        .saturating_mul(window.max(1));
    let mut seen = vec![0u64; bits.div_ceil(64)];
    let mut w = 0usize;
    for rd in 0..e {
        let k = key_of[rd] as usize;
        let (word, bit) = (k >> 6, 1u64 << (k & 63));
        if seen[word] & bit == 0 {
            seen[word] |= bit;
            if w != rd {
                src[w] = src[rd];
                dst[w] = dst[rd];
                attr.copy_within(
                    rd * EDGE_FEAT_DIM..rd * EDGE_FEAT_DIM + EDGE_FEAT_DIM,
                    w * EDGE_FEAT_DIM,
                );
            }
            w += 1;
        }
    }
    src.truncate(w);
    dst.truncate(w);
    attr.truncate(w * EDGE_FEAT_DIM);
}

/// The oracle's `axis_idx_of(a)`: the one-hot axis of an attr.
fn axis_idx_of(a: &[f32]) -> u8 {
    if a[0] > 0.5 {
        0
    } else if a[1] > 0.5 {
        1
    } else {
        2
    }
}

/// The replaced producer verify: the axis read from the row's one-hot, then each attr checked against it.
#[allow(clippy::float_cmp)]
fn per_edge_verify(g: &AxisGraph, n_stones: usize, n_legal: usize, params: &BuildParams) {
    let trunk_sz = params.trunk_size;
    let n = g.num_nodes();
    let n_edges = g.num_edges();
    assert!(
        g.n_nodes_checksum as usize == n && n == n_stones + n_legal + 1,
        "NodeCountChecksum: declared {} vs N {} (stones {} + legal {} + 1)",
        g.n_nodes_checksum,
        n,
        n_stones,
        n_legal
    );
    assert!(
        g.node_feat.0.len() == n * NODE_FEAT_DIM && g.node_coords.len() == 2 * n,
        "NodeFeatDimMismatch: node_feat len {} (want {}), node_coords len {} (want {})",
        g.node_feat.0.len(),
        n * NODE_FEAT_DIM,
        g.node_coords.len(),
        2 * n
    );
    assert!(
        g.legal_mask.len() == n && g.stone_mask.len() == n,
        "NodeFeatDimMismatch: mask lens {}/{} != N {}",
        g.legal_mask.len(),
        g.stone_mask.len(),
        n
    );
    assert!(
        g.edge_attr.0.len() == EDGE_FEAT_DIM * n_edges,
        "EdgeAttrDimMismatch: edge_attr len {} != 5*E {}",
        g.edge_attr.0.len(),
        EDGE_FEAT_DIM * n_edges
    );
    assert!(
        g.builder_impl == BUILDER_IMPL_NATIVE,
        "NonNativeSampleBuilder: impl tag != 1"
    );
    assert!(
        n_legal > 0 || n_stones == 0,
        "EmptyLegalSet: {n_stones} stones but zero legal nodes"
    );

    let dummy_idx = (n - 1) as u32;
    let cur_f = f32::from(g.current_player);
    let window = i32::from(params.win_length) - 1;
    for e in 0..n_edges {
        let s = g.edge_index.src[e];
        let d = g.edge_index.dst[e];
        assert!(
            (s as usize) < n && (d as usize) < n,
            "EdgeIndexOutOfBounds: edge {e} = ({s},{d}), N {n}"
        );
        let a = &g.edge_attr.0[e * EDGE_FEAT_DIM..e * EDGE_FEAT_DIM + EDGE_FEAT_DIM];
        if s == dummy_idx || d == dummy_idx {
            assert!(
                a.iter().all(|&x| x == 0.0),
                "EdgeAttrGeometryMismatch: dummy edge {e} has non-zero attrs {a:?}"
            );
            continue;
        }
        let axis = axis_idx_of(a) as usize;
        let onehot_ok = (0..3).all(|k| a[k] == if k == axis { 1.0 } else { 0.0 });
        let dist = a[3];
        let di = dist as i32;
        let (aq, ar) = WIN_AXES[axis];
        let dq = g.node_coords[d as usize * 2] - g.node_coords[s as usize * 2];
        let dr = g.node_coords[d as usize * 2 + 1] - g.node_coords[s as usize * 2 + 1];
        let src_player = if g.stone_mask[s as usize] {
            if g.node_feat.0[s as usize * NODE_FEAT_DIM] == 1.0 {
                cur_f
            } else {
                -cur_f
            }
        } else {
            0.0
        };
        assert!(
            onehot_ok
                && f64::from(dist) == f64::from(di)
                && di != 0
                && di.abs() <= window
                && dq == di * aq
                && dr == di * ar
                && a[4] == src_player,
            "EdgeAttrGeometryMismatch: edge {e} ({s}->{d}) attrs {a:?} vs geometry \
             delta ({dq},{dr}), expected src_player {src_player}"
        );
    }

    assert!(
        g.legal_node_gather.len() == n_legal && g.policy_scatter_index.0.len() == n_legal,
        "GatherNotLegalNode: gather/slot lens {}/{} != n_legal {}",
        g.legal_node_gather.len(),
        g.policy_scatter_index.0.len(),
        n_legal
    );
    let half = (trunk_sz - 1) / 2;
    let n_slots = (trunk_sz * trunk_sz) as usize;
    let mut slot_seen = vec![false; n_slots];
    for (i, &row) in g.legal_node_gather.iter().enumerate() {
        assert!(
            (row as usize) >= n_stones && (row as usize) < n_stones + n_legal,
            "GatherNotLegalNode: gather[{i}] = {row} outside [{}, {})",
            n_stones,
            n_stones + n_legal
        );
        let q = g.node_coords[row as usize * 2];
        let r = g.node_coords[row as usize * 2 + 1];
        let slot = g.policy_scatter_index.0[i];
        if slot == OFF_WINDOW_SLOT {
            let wq = q - g.window_center.0 + half;
            let wr = r - g.window_center.1 + half;
            assert!(
                wq < 0 || wq >= trunk_sz || wr < 0 || wr >= trunk_sz,
                "ScatterSlotCanonicalMismatch: slot[{i}] sentinel but coord ({q},{r}) in-window"
            );
        } else {
            assert!(
                slot >= 0 && (slot as usize) < n_slots,
                "ScatterSlotOutOfBounds: slot[{i}] = {slot}, trunk² = {n_slots}"
            );
            let wq = q - g.window_center.0 + half;
            let wr = r - g.window_center.1 + half;
            assert!(
                wq >= 0 && wq < trunk_sz && wr >= 0 && wr < trunk_sz && slot == wq * trunk_sz + wr,
                "ScatterSlotCanonicalMismatch: slot[{i}] = {slot} vs canonical for ({q},{r})"
            );
            assert!(
                !slot_seen[slot as usize],
                "ScatterSlotAliasing: slot {slot} claimed by two legal nodes"
            );
            slot_seen[slot as usize] = true;
        }
    }
}

/// A panic's message, or a placeholder naming a payload that carries none.
fn panic_text(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
        .unwrap_or_else(|| "<non-string panic payload>".to_string())
}

/// What a call returned, or the message it panicked with.
fn outcome<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    catch_unwind(AssertUnwindSafe(f)).map_err(|p| panic_text(&*p))
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// `Ok` when every field is byte-equal, floats by bit pattern, else the first field that differs.
fn same_bytes(a: &AxisGraph, b: &AxisGraph) -> Result<(), &'static str> {
    // Exhaustive on purpose: a field added to `AxisGraph` fails to compile here until it is compared.
    let AxisGraph {
        node_feat,
        edge_index,
        edge_attr,
        legal_mask,
        stone_mask,
        policy_scatter_index,
        node_coords,
        legal_node_gather,
        n_stones,
        n_nodes_checksum,
        window_center,
        current_player,
        builder_impl,
    } = a;
    let checks: [(&'static str, bool); 14] = [
        ("node_feat", bits(&node_feat.0) == bits(&b.node_feat.0)),
        ("edge_index.src", edge_index.src == b.edge_index.src),
        ("edge_index.dst", edge_index.dst == b.edge_index.dst),
        ("edge_attr", bits(&edge_attr.0) == bits(&b.edge_attr.0)),
        ("legal_mask", *legal_mask == b.legal_mask),
        ("stone_mask", *stone_mask == b.stone_mask),
        (
            "policy_scatter_index",
            *policy_scatter_index == b.policy_scatter_index,
        ),
        ("node_coords", *node_coords == b.node_coords),
        (
            "legal_node_gather",
            *legal_node_gather == b.legal_node_gather,
        ),
        ("n_stones", *n_stones == b.n_stones),
        ("n_nodes_checksum", *n_nodes_checksum == b.n_nodes_checksum),
        ("window_center", *window_center == b.window_center),
        ("current_player", *current_player == b.current_player),
        ("builder_impl", *builder_impl == b.builder_impl),
    ];
    match checks.iter().find(|(_, ok)| !ok) {
        Some(&(name, _)) => Err(name),
        None => Ok(()),
    }
}

/// The geometries each position is built at: both shipped radii, the golden set's alternate, the smallest window, a wider one.
fn variants(to_move: i8, moves_remaining: u8) -> [BuildParams; 6] {
    let flipped = (-to_move, 3 - moves_remaining.min(2));
    let at =
        |win_length, radius, trunk_size, (current_player, moves_remaining): (i8, u8)| BuildParams {
            win_length,
            radius,
            current_player,
            moves_remaining,
            trunk_size,
        };
    [
        at(6, 8, 19, (to_move, moves_remaining)),
        at(6, 8, 19, flipped),
        at(6, 6, 19, (to_move, moves_remaining)),
        at(5, 4, 19, flipped),
        at(2, 1, 7, (to_move, moves_remaining)),
        at(8, 12, 25, flipped),
    ]
}

/// Builds `stones` through both algorithms and names the first disagreement, panics compared by message.
fn compare(stones: &StoneList, params: &BuildParams) -> Result<(), String> {
    match (
        outcome(|| build_axis_graph(stones, params)),
        outcome(|| two_pass_build(stones, params)),
    ) {
        (Ok(a), Ok(b)) => same_bytes(&a, &b).map_err(|f| format!("field {f} differs")),
        (Err(a), Err(b)) if a == b => Ok(()),
        (a, b) => Err(format!(
            "outcomes differ: builder {:?} vs reference {:?}",
            a.map(|_| "a graph"),
            b.map(|_| "a graph")
        )),
    }
}

#[test]
fn every_field_matches_the_two_pass_reference_on_the_recorded_positions() {
    let recorded = positions::read_positions_r8().unwrap_or_else(|e| panic!("{e}"));
    let mut mismatches = Vec::new();
    let mut compared = 0usize;
    let mut off_grid = 0usize;
    for (i, p) in recorded.iter().enumerate() {
        let stones = StoneList {
            stones: p.stones.clone(),
        };
        for (v, params) in variants(p.to_move, p.moves_remaining).iter().enumerate() {
            if let Err(e) = compare(&stones, params) {
                mismatches.push(format!("position {i} variant {v}: {e}"));
            }
            compared += 1;
        }
        off_grid += usize::from(BallGrid::build(&p.stones, 8).is_none());
    }
    assert_eq!(compared, positions::POSITIONS_R8_COUNT * 6);
    assert_eq!(
        off_grid, 0,
        "real positions must take the legal grid, or the change is inert"
    );
    assert!(
        mismatches.is_empty(),
        "{} of {compared} builds differ:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}

/// Boards real play never reaches: empty, one stone, overlapping and far-apart clusters, a duplicate coord, walls.
fn edge_cases() -> Vec<StoneList> {
    let mut cluster: Vec<(i32, i32, i8)> = Vec::new();
    for q in -3i32..=3 {
        for r in -3i32..=3 {
            if (q + r).abs() <= 3 {
                let player = if (q * 7 + r * 3).rem_euclid(2) == 0 {
                    1
                } else {
                    -1
                };
                cluster.push((q, r, player));
            }
        }
    }
    let mut cluster_and_far = cluster.clone();
    cluster_and_far.push((5000, -7, 1));
    let mut walls = Vec::new();
    for k in 0..5 {
        walls.push((k, 0, 1i8));
        walls.push((k, 2, -1i8));
        walls.push((-k, k, if k % 2 == 0 { 1 } else { -1 }));
    }
    walls.sort_unstable();
    walls.dedup_by_key(|s| (s.0, s.1));
    [
        vec![],
        vec![(0, 0, 1)],
        vec![(3, -2, -1)],
        vec![(0, 0, 1), (1, 0, -1), (0, 1, 1), (2, -1, -1), (3, 0, 1)],
        vec![(-9, -4, 1), (-11, -7, -1), (-6, 2, 1), (-30, 17, -1)],
        vec![(0, 0, 1), (700, -3, -1), (701, -3, 1)],
        vec![(0, 0, 1), (300_000, 5, -1)],
        vec![(0, 0, 1), (2, 0, -1), (0, 0, -1)],
        cluster,
        cluster_and_far,
        walls,
    ]
    .into_iter()
    .map(|stones| StoneList { stones })
    .collect()
}

#[test]
fn every_field_matches_the_two_pass_reference_on_the_edge_cases() {
    let mut mismatches = Vec::new();
    let (mut dense_lookup, mut grid) = ([0usize; 2], [0usize; 2]);
    for (i, stones) in edge_cases().iter().enumerate() {
        for (cp, mr) in [(1i8, 1u8), (-1, 2)] {
            let radius_zero = BuildParams {
                radius: 0,
                ..variants(cp, mr)[0]
            };
            for (v, params) in variants(cp, mr).iter().chain([&radius_zero]).enumerate() {
                if let Err(e) = compare(stones, params) {
                    mismatches.push(format!("case {i} ({cp}, {mr}) variant {v}: {e}"));
                }
                let radius = i32::from(params.radius);
                grid[usize::from(BallGrid::build(&stones.stones, radius).is_some())] += 1;
                if let Ok(g) = outcome(|| build_axis_graph(stones, params)) {
                    let dense = axis_index_is_dense(&g.node_coords, g.num_nodes() - 1);
                    dense_lookup[usize::from(dense)] += 1;
                }
            }
        }
    }
    assert!(
        mismatches.is_empty(),
        "{} builds differ:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
    // Vacuity control: the cases must reach both arms of the legal grid and of the coordinate lookup.
    assert!(
        dense_lookup.iter().chain(&grid).all(|&n| n > 0),
        "lookup arms hash/dense taken {dense_lookup:?}, legal arms hash/grid {grid:?}"
    );
}

/// The one test in this binary that skips the verify, so the skip count's deltas are exact.
#[test]
fn a_build_that_skips_the_producer_verify_is_byte_identical_and_counted() {
    let recorded = positions::read_positions_r8().unwrap_or_else(|e| panic!("{e}"));
    let mut cases: Vec<(StoneList, BuildParams)> = Vec::new();
    for p in &recorded {
        for params in variants(p.to_move, p.moves_remaining) {
            let stones = StoneList {
                stones: p.stones.clone(),
            };
            cases.push((stones, params));
        }
    }
    cases.extend(edge_cases().into_iter().map(|s| (s, variants(-1, 2)[0])));
    let before = unverified_builds();
    let mut mismatches = Vec::new();
    for (c, (stones, params)) in cases.iter().enumerate() {
        let verified = build_axis_graph(stones, params);
        let skipped =
            build_axis_graph_verified_by(stones, params, ProducerVerify::ConsumerEveryBatch);
        if let Err(field) = same_bytes(&verified, &skipped) {
            mismatches.push(format!("case {c}: field {field} differs"));
        }
    }
    assert!(
        mismatches.is_empty(),
        "{} of {} builds differ:\n{}",
        mismatches.len(),
        cases.len(),
        mismatches.join("\n")
    );
    assert_eq!(
        unverified_builds() - before,
        cases.len() as u64,
        "each skipping build counts once and no verifying build counts"
    );
}

/// One way to corrupt a built graph at edge `e`.
type Mutation = fn(&mut AxisGraph, usize);

fn attr(g: &mut AxisGraph, e: usize) -> &mut [f32] {
    &mut g.edge_attr.0[e * EDGE_FEAT_DIM..(e + 1) * EDGE_FEAT_DIM]
}

/// Corruptions of a row's attrs, its endpoints, its endpoints' wire columns and the payload's shape.
fn mutations() -> Vec<(&'static str, Mutation)> {
    vec![
        ("none", |_, _| {}),
        ("dist sign", |g, e| attr(g, e)[3] = -attr(g, e)[3]),
        ("dist zero", |g, e| attr(g, e)[3] = 0.0),
        ("dist plus one", |g, e| attr(g, e)[3] += 1.0),
        ("dist plus half", |g, e| attr(g, e)[3] += 0.5),
        ("dist nan", |g, e| attr(g, e)[3] = f32::NAN),
        ("dist past window", |g, e| attr(g, e)[3] = 40.0),
        ("dist past i32", |g, e| attr(g, e)[3] = 3.0e9),
        ("player negated", |g, e| attr(g, e)[4] = -attr(g, e)[4]),
        ("player two", |g, e| attr(g, e)[4] = 2.0),
        ("player nan", |g, e| attr(g, e)[4] = f32::NAN),
        ("onehot rotated", |g, e| attr(g, e)[..3].rotate_right(1)),
        ("onehot halved", |g, e| {
            attr(g, e)[..3].iter_mut().for_each(|x| *x *= 0.5);
        }),
        ("onehot doubled", |g, e| attr(g, e)[..3].fill(1.0)),
        ("onehot negative zeros", |g, e| {
            attr(g, e)[..3]
                .iter_mut()
                .filter(|x| **x == 0.0)
                .for_each(|x| *x = -0.0);
        }),
        ("ends swapped", |g, e| {
            let (s, d) = (g.edge_index.src[e], g.edge_index.dst[e]);
            g.edge_index.src[e] = d;
            g.edge_index.dst[e] = s;
        }),
        ("retargeted", |g, e| {
            g.edge_index.dst[e] = (g.edge_index.dst[e] + 1) % g.num_nodes() as u32;
        }),
        ("self loop", |g, e| {
            g.edge_index.dst[e] = g.edge_index.src[e];
        }),
        ("self loop with a zero-step row", |g, e| {
            g.edge_index.dst[e] = g.edge_index.src[e];
            let player = attr(g, e)[4];
            attr(g, e).copy_from_slice(&edge_attr_row(0, 0.0, player));
        }),
        ("out of bounds", |g, e| {
            g.edge_index.dst[e] = g.num_nodes() as u32;
        }),
        ("to the dummy", |g, e| {
            g.edge_index.dst[e] = (g.num_nodes() - 1) as u32;
        }),
        ("source moved", |g, e| {
            g.node_coords[g.edge_index.src[e] as usize * 2] += 1;
        }),
        ("target moved", |g, e| {
            g.node_coords[g.edge_index.dst[e] as usize * 2 + 1] -= 1;
        }),
        ("source own column flipped", |g, e| {
            let c = &mut g.node_feat.0[g.edge_index.src[e] as usize * NODE_FEAT_DIM];
            *c = 1.0 - *c;
        }),
        ("source stone mask flipped", |g, e| {
            let m = &mut g.stone_mask[g.edge_index.src[e] as usize];
            *m = !*m;
        }),
        ("attr row dropped", |g, _| {
            let len = g.edge_attr.0.len();
            g.edge_attr.0.truncate(len - EDGE_FEAT_DIM);
        }),
        ("checksum", |g, _| g.n_nodes_checksum += 1),
        ("gather to row zero", |g, _| g.legal_node_gather[0] = 0),
        ("slot shifted", |g, _| g.policy_scatter_index.0[0] += 1),
    ]
}

/// The edges worth corrupting: the first and last axis edge, one per source kind, sign and axis, two dummies.
#[allow(clippy::float_cmp)] // the rows hold exact constants the builder wrote
fn probe_edges(g: &AxisGraph) -> Vec<usize> {
    let n_axis = g.num_edges() - 2 * (g.num_nodes() - 1);
    let row = |e: usize| &g.edge_attr.0[e * EDGE_FEAT_DIM..(e + 1) * EDGE_FEAT_DIM];
    let mut picks = vec![0, n_axis - 1, n_axis, g.num_edges() - 1];
    picks.extend((0..n_axis).find(|&e| row(e)[4] != 0.0));
    picks.extend((0..n_axis).find(|&e| row(e)[4] == 0.0));
    picks.extend((0..n_axis).find(|&e| row(e)[3] < 0.0));
    picks.extend((0..n_axis).find(|&e| row(e)[1] == 1.0));
    picks.extend((0..n_axis).find(|&e| row(e)[2] == 1.0));
    picks
}

#[test]
fn the_verify_accepts_and_refuses_exactly_what_the_per_edge_verify_does() {
    let recorded = positions::read_positions_r8().unwrap_or_else(|e| panic!("{e}"));
    let mut cases: Vec<(StoneList, BuildParams)> = Vec::new();
    for p in recorded.iter().step_by(64) {
        let vs = variants(p.to_move, p.moves_remaining);
        for params in [vs[0], vs[3], vs[4]] {
            let stones = StoneList {
                stones: p.stones.clone(),
            };
            cases.push((stones, params));
        }
    }
    cases.extend(edge_cases().into_iter().map(|s| (s, variants(-1, 2)[0])));
    let mut disagreements = Vec::new();
    let mut named = std::collections::BTreeMap::<String, usize>::new();
    let mut passes = 0usize;
    for (c, (stones, params)) in cases.iter().enumerate() {
        // Built by the reference, so a verify that refuses a sound graph is a disagreement, not a skipped case.
        let built = two_pass_build(stones, params);
        // A window narrowed under the built graph is the one way a sound row fails on its length alone.
        for win_length in [params.win_length - 1, params.win_length + 1] {
            let p = BuildParams {
                win_length,
                ..*params
            };
            let (ns, nl) = (built.n_stones as usize, built.legal_node_gather.len());
            let fast = outcome(|| verify_contract(&built, ns, nl, &p));
            let slow = outcome(|| per_edge_verify(&built, ns, nl, &p));
            if fast != slow {
                disagreements.push(format!(
                    "case {c} at window {win_length}: {fast:?} vs {slow:?}"
                ));
            }
        }
        for e in probe_edges(&built) {
            for (name, mutate) in mutations() {
                let mut g = built.clone();
                mutate(&mut g, e);
                let (ns, nl) = (g.n_stones as usize, g.legal_node_gather.len());
                let fast = outcome(|| verify_contract(&g, ns, nl, params));
                let slow = outcome(|| per_edge_verify(&g, ns, nl, params));
                match &slow {
                    Ok(()) => passes += 1,
                    Err(m) => {
                        let check = m.split(':').next().unwrap_or("").to_string();
                        *named.entry(check).or_default() += 1;
                    }
                }
                if fast != slow {
                    disagreements.push(format!("case {c} edge {e} {name}: {fast:?} vs {slow:?}"));
                }
            }
        }
    }
    assert!(
        disagreements.is_empty(),
        "{} disagreements:\n{}",
        disagreements.len(),
        disagreements.join("\n")
    );
    // Vacuity controls: both verdicts occur, and the per-edge checks are among the refusals.
    assert!(
        passes > 0,
        "no corrupted graph passed: the comparison saw only refusals"
    );
    for check in [
        "EdgeAttrGeometryMismatch",
        "EdgeIndexOutOfBounds",
        "EdgeAttrDimMismatch",
        "NodeCountChecksum",
    ] {
        assert!(named.contains_key(check), "{check} never fired: {named:?}");
    }
}
