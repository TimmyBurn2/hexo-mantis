//! ⊕ R335(e) Leg 2 (`S-PREFUSE`) — the parity harness, BANKED with a verdict.
//!
//! WHAT THE CARD TURNS ON. `S-PREFUSE` moves `GraphWire::from_axis_graphs` off the server
//! thread and into the workers, so the queue carries fused wire. SCOUT §5 P2 risk 3 names the
//! condition the whole card depends on: *"fusing per worker fuses per worker's SLICE, so a pop
//! that draws from k workers must concatenate k wires; if that concatenation is done on the
//! server it re-introduces a share of the term the card removes. The design must fuse to a
//! worker-local wire that the server can concatenate BY OFFSET ARITHMETIC ALONE, or the card
//! is worth less than it looks."*
//!
//! THIS FILE ANSWERS THAT, and it needs no GPU to do it. `concat_by_offset` below re-fuses
//! NOTHING: it never touches an `AxisGraph`, only the already-fused wires, and it does exactly
//! the four things a server-side concatenation is allowed to do — copy bulk arrays, add a
//! running node base to the two index arrays, add running bases to the three offset arrays,
//! and splice `edge_index`'s two halves. If its output is byte-identical to a whole-batch
//! `from_axis_graphs`, offset arithmetic alone suffices and the card's seam is viable.
//!
//! THE SPLICE IS THE NON-OBVIOUS PART, and it is why this is a test and not an assertion in a
//! design note. `edge_index` is `[src_global (E) ‖ dst_global (E)]` — ALL srcs then ALL dsts —
//! so concatenating two wires is not an append. It is a four-way interleave: A.src, B.src+base,
//! A.dst, B.dst+base. An implementation that appends would produce a wire that still passes
//! every length and offset check and is silently wrong about every edge.
//!
//! WHAT THIS FILE DOES NOT CLAIM. It does not measure the card. The bar R335(e) sets for
//! `S-PREFUSE` is `ms/sim` on the CONTENDED CARD, and there is no card in this workspace —
//! see `PERF_TRANCHE3_RESULTS.md` §3. The leg is banked; this is the half of it that could be
//! finished honestly, and it is the half that carried the risk.

#[path = "common/mod.rs"]
mod common;

use common::{concat_by_offset, corpus, fuse};
use mantis_selfplay::queues::GraphWireArrays;

fn assert_field_equal(a: &GraphWireArrays, b: &GraphWireArrays, ctx: &str) {
    assert_eq!(a.contract_version, b.contract_version, "{ctx}: contract_version");
    assert_eq!(a.builder_impl, b.builder_impl, "{ctx}: builder_impl");
    assert_eq!(a.n_graphs, b.n_graphs, "{ctx}: n_graphs");
    assert_eq!(a.node_feat, b.node_feat, "{ctx}: node_feat");
    assert_eq!(a.node_coords, b.node_coords, "{ctx}: node_coords");
    assert_eq!(a.edge_index, b.edge_index, "{ctx}: edge_index");
    assert_eq!(a.edge_attr, b.edge_attr, "{ctx}: edge_attr");
    assert_eq!(a.node_offsets, b.node_offsets, "{ctx}: node_offsets");
    assert_eq!(a.edge_offsets, b.edge_offsets, "{ctx}: edge_offsets");
    assert_eq!(a.legal_offsets, b.legal_offsets, "{ctx}: legal_offsets");
    assert_eq!(a.legal_node_gather, b.legal_node_gather, "{ctx}: legal_node_gather");
    assert_eq!(a.policy_dst_slot, b.policy_dst_slot, "{ctx}: policy_dst_slot");
    assert_eq!(a.n_nodes_checksum, b.n_nodes_checksum, "{ctx}: n_nodes_checksum");
    assert_eq!(a.n_stones, b.n_stones, "{ctx}: n_stones");
    assert_eq!(a.window_center, b.window_center, "{ctx}: window_center");
    assert_eq!(a.current_player, b.current_player, "{ctx}: current_player");
}

/// The card's precondition, at both radii and at several worker splits: a wire concatenated
/// from k worker-local wires by offset arithmetic alone is byte-identical, EVERY FIELD, to the
/// wire the server builds today from the raw graphs.
#[test]
fn offset_concatenation_is_byte_identical_to_the_whole_batch_fuse() {
    for (radius, seed) in [(6u16, 0x0A1F_6675_7365_0001u64), (8, 0x0A1F_6675_7365_0002)] {
        // 40 graphs is the ledger's measured pop; the splits are the plausible worker mixes.
        let graphs = corpus(40, radius, seed);
        let whole = fuse(&graphs);
        for split in [vec![40], vec![20, 20], vec![8; 5], vec![1; 40], vec![13, 1, 26]] {
            let mut parts = Vec::new();
            let mut at = 0usize;
            for k in &split {
                parts.push(fuse(&graphs[at..at + k]));
                at += k;
            }
            assert_eq!(at, graphs.len(), "the split must cover the pop");
            let joined = concat_by_offset(parts);
            assert_field_equal(&joined, &whole, &format!("r{radius} split {split:?}"));
        }
    }
}

/// PLANTED BREAK, and it plants the mistake the layout actually invites: appending
/// `edge_index` wholesale instead of splicing its two halves. The result has the right length
/// and the right offsets — every structural check the collate runs would pass it — and it is
/// wrong about every edge in every graph after the first. If this test ever stops failing, the
/// parity assertion above has stopped comparing `edge_index`.
#[test]
fn a_naive_edge_index_append_is_caught() {
    let graphs = corpus(6, 6, 0x0A1F_6675_7365_0003);
    let whole = fuse(&graphs);
    let parts = [fuse(&graphs[..3]), fuse(&graphs[3..])];
    let base = *parts[0].node_offsets.last().expect("node offsets");
    let mut naive = concat_by_offset(vec![fuse(&graphs[..3]), fuse(&graphs[3..])]);
    // Rebuild edge_index the WRONG way: A whole, then B whole, each merely re-offset.
    naive.edge_index = parts[0]
        .edge_index
        .iter()
        .copied()
        .chain(parts[1].edge_index.iter().map(|&v| v + base))
        .collect();
    assert_eq!(
        naive.edge_index.len(),
        whole.edge_index.len(),
        "the naive append has the RIGHT LENGTH — which is what makes it dangerous"
    );
    assert_ne!(
        naive.edge_index, whole.edge_index,
        "the naive append must NOT match; if it does, `[src ‖ dst]` is no longer the layout \
         and this file's splice is testing nothing"
    );
}
