//! The parallel ring rebuild is BIT-IDENTICAL to the serial one.
//!
//! Determinism has two halves, tested separately. The GENERATOR: identically seeded buffers must
//! draw the same symmetries in the same order on one thread or many — a hoist that changed the
//! draw ORDER or COUNT would still produce valid training data, differently augmented, silently.
//! The REASSEMBLY: `policy_target` is ONE flat concatenation segmented by the graphs' own legal
//! counts, so results returned in completion order mis-pair every target while every length
//! still checks out.

use mantis_graph::AxisGraph;
use mantis_selfplay::replay::hexg::{GraphRecord, GraphTargets, HexgBuffer};
use mantis_selfplay::replay::sym::N_SYMS;
use rand::RngExt;

const CAP: usize = 512;
const VISIT_CAP: usize = 128;
const SEED: u64 = 0x5AFE_0B1F_0000_0001;

fn splitmix64(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A ring of deterministic positions wide enough that a chunked split is non-trivial.
fn filled_buffer(n_records: usize) -> HexgBuffer {
    let mut buf = HexgBuffer::new(CAP, "gnn_axis_v1", VISIT_CAP).expect("graph buffer");
    // The buffer seeds its generator from ENTROPY, so two buffers never agree by construction;
    // pinning it makes serial-vs-parallel a comparison of the REBUILD, not of two samples.
    buf.seed_sampler(0xB1_0000_0001);
    let mut s = SEED;
    for i in 0..n_records {
        // A compact line of stones plus two legal cells at its ends, so the aligned mass equals
        // the stored mass and `mass_drop_check` does not refuse the record. Every seventh record
        // is EMPTY: the hoist's guard forces the identity symmetry on a stoneless board, and a
        // corpus with no empty record cannot see the draw COUNT shift while every value matches.
        let empty = i % 7 == 3;
        let n_stones = if empty {
            0
        } else {
            6 + (splitmix64(&mut s) % 18) as i16
        };
        let stones: Vec<(i16, i16, i8)> = (0..n_stones)
            .map(|q| (q, (q % 3) - 1, if q % 2 == 0 { 1 } else { -1 }))
            .collect();
        // An empty board's legal set is the 5x5 origin fallback, so its visits sit there.
        let visits = if empty {
            vec![(0i16, 0i16, 0.6f32), (1, 0, 0.4)]
        } else {
            vec![(-1i16, 0i16, 0.6f32), (n_stones, 0, 0.4)]
        };
        let rec = GraphRecord {
            stones,
            visits,
            tail_mass: 0.0,
            current_player: if i % 2 == 0 { 1 } else { -1 },
            moves_remaining: 2,
            ply_index: (i % 50) as u16,
            is_full_search: true,
            outcome: if i % 3 == 0 { 1.0 } else { -1.0 },
            value_valid: true,
            game_length: 40,
            game_id: -1,
        };
        buf.push_record_impl(&rec, (10 + i) as i64).expect("push");
    }
    buf
}

fn assert_graphs_identical(a: &[AxisGraph], b: &[AxisGraph], case: &str) {
    assert_eq!(a.len(), b.len(), "{case}: graph count");
    for (i, (ga, gb)) in a.iter().zip(b).enumerate() {
        assert_eq!(
            ga.node_feat.0, gb.node_feat.0,
            "{case}: graph {i} node_feat"
        );
        assert_eq!(
            ga.node_coords, gb.node_coords,
            "{case}: graph {i} node_coords"
        );
        assert_eq!(
            ga.edge_index.src, gb.edge_index.src,
            "{case}: graph {i} edge src"
        );
        assert_eq!(
            ga.edge_index.dst, gb.edge_index.dst,
            "{case}: graph {i} edge dst"
        );
        assert_eq!(
            ga.edge_attr.0, gb.edge_attr.0,
            "{case}: graph {i} edge_attr"
        );
        assert_eq!(
            ga.legal_node_gather, gb.legal_node_gather,
            "{case}: graph {i} gather"
        );
        assert_eq!(
            ga.policy_scatter_index.0, gb.policy_scatter_index.0,
            "{case}: graph {i} scatter"
        );
        assert_eq!(
            ga.n_nodes_checksum, gb.n_nodes_checksum,
            "{case}: graph {i} checksum"
        );
        assert_eq!(
            ga.window_center, gb.window_center,
            "{case}: graph {i} window_center"
        );
        assert_eq!(
            ga.current_player, gb.current_player,
            "{case}: graph {i} current_player"
        );
    }
}

fn assert_targets_identical(a: &GraphTargets, b: &GraphTargets, case: &str) {
    assert_eq!(a.policy_target, b.policy_target, "{case}: policy_target");
    assert_eq!(a.outcomes, b.outcomes, "{case}: outcomes");
    assert_eq!(a.value_valid, b.value_valid, "{case}: value_valid");
    assert_eq!(a.is_full_search, b.is_full_search, "{case}: is_full_search");
    assert_eq!(a.argmax_q, b.argmax_q, "{case}: argmax_q");
    assert_eq!(a.argmax_r, b.argmax_r, "{case}: argmax_r");
    assert_eq!(a.argmax_valid, b.argmax_valid, "{case}: argmax_valid");
}

/// The headline: same seed, same ring, serial vs parallel — every byte equal.
#[test]
fn parallel_rebuild_is_bit_identical_to_serial() {
    for &(records, batch, threads) in &[
        (64usize, 32usize, 4usize),
        (256, 128, 8),
        // More threads than items: the chunker must not spawn empty chunks or reorder.
        (16, 5, 12),
        // One item, many threads — the degenerate split.
        (16, 1, 8),
    ] {
        for &augment in &[false, true] {
            let mut serial = filled_buffer(records);
            let mut parallel = filled_buffer(records);
            let (gs, ts) = serial
                .sample_graph_batch_impl(batch, augment, 0.0, 1)
                .unwrap();
            let (gp, tp) = parallel
                .sample_graph_batch_impl(batch, augment, 0.0, threads)
                .unwrap();
            let case = format!("records={records} batch={batch} threads={threads} aug={augment}");
            assert_eq!(gs.len(), batch, "{case}: batch size");
            assert_graphs_identical(&gs, &gp, &case);
            assert_targets_identical(&ts, &tp, &case);
        }
    }
}

/// The generator half: the hoist must not change what is drawn, only when.
///
/// Sampled repeatedly, because a hoist that consumed a different number of draws diverges on the
/// SECOND sample even where the first agreed.
#[test]
fn the_rng_hoist_leaves_the_draw_stream_unchanged_across_repeats() {
    let mut serial = filled_buffer(128);
    let mut parallel = filled_buffer(128);
    for round in 0..3 {
        let (gs, ts) = serial.sample_graph_batch_impl(48, true, 0.0, 1).unwrap();
        let (gp, tp) = parallel.sample_graph_batch_impl(48, true, 0.0, 8).unwrap();
        let case = format!("augmented repeat {round}");
        assert_graphs_identical(&gs, &gp, &case);
        assert_targets_identical(&ts, &tp, &case);
    }
}

/// The reassembly half: order is INDEX order, not completion order, so the assertion is
/// per-item and positional, never a multiset.
#[test]
fn results_come_back_in_index_order() {
    let mut serial = filled_buffer(200);
    let mut parallel = filled_buffer(200);
    let (gs, ts) = serial.sample_graph_batch_impl(64, false, 0.0, 1).unwrap();
    let (gp, tp) = parallel.sample_graph_batch_impl(64, false, 0.0, 7).unwrap();

    // The batch is genuinely heterogeneous, or a permutation would be invisible here.
    let distinct: std::collections::HashSet<usize> = gs.iter().map(AxisGraph::num_nodes).collect();
    assert!(
        distinct.len() > 3,
        "the corpus must vary in size for an order test to bite; saw {} distinct node counts",
        distinct.len()
    );
    for (i, (a, b)) in gs.iter().zip(&gp).enumerate() {
        assert_eq!(
            a.num_nodes(),
            b.num_nodes(),
            "graph {i} landed out of order"
        );
        assert_eq!(
            a.num_edges(),
            b.num_edges(),
            "graph {i} landed out of order"
        );
    }
    assert_targets_identical(&ts, &tp, "index order");
}

/// The hoist's PREDICATE, against a transcription of the original inline draw.
///
/// The parity tests above run BOTH arms through the hoist, so a changed draw COUNT moves them
/// together and passes. Only this row sees it. The corpus carries empty-board records
/// deliberately: with none, `n_stones != 0` and `!stones.is_empty()` never disagree.
#[test]
fn the_hoisted_draw_matches_the_original_inline_predicate() {
    let indices: Vec<usize> = (0..200usize).map(|i| (i * 7) % 128).collect();
    let empty_seen = {
        let buf = filled_buffer(128);
        indices
            .iter()
            .filter(|&&i| buf.record_at(i).stones.is_empty())
            .count()
    };
    assert!(
        empty_seen > 0,
        "the index set must reach at least one empty-board record, or the predicate this \
         test exists to pin is never exercised"
    );

    for augment in [false, true] {
        let mut hoisted_buf = filled_buffer(128);
        let mut inline_buf = filled_buffer(128);
        let hoisted = hoisted_buf.draw_syms(&indices, augment);

        // The original body, transcribed: the predicate reads the MATERIALISED record and the
        // draw happens inside the per-index step rather than in a pass of its own.
        let inline: Vec<usize> = indices
            .iter()
            .map(|&idx| {
                let rec = inline_buf.record_at(idx);
                if augment && !rec.stones.is_empty() {
                    inline_buf.rng.random_range(0..N_SYMS)
                } else {
                    0
                }
            })
            .collect();
        assert_eq!(
            hoisted, inline,
            "augment={augment}: the hoisted draw stream diverged from the original inline \
             one. Either the predicate changed meaning or the number of draws did — both \
             silently re-augment every future batch."
        );
    }
}
