//! The position key misses on a seeded drive exactly the leaves the SHA-256 key over the built graph missed.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use fxhash::FxHashSet;
use mantis_core::{Board, BoardGeometry};
use mantis_graph::AxisGraph;
use mantis_search::{MCTSTree, VIRTUAL_LOSS_PENALTY};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use sha2::{Digest, Sha256};

use super::{infer_and_expand_graph, InferContext, LeafSelection};
use crate::queues::{GraphQueue, EVAL_CACHE_CAPACITY, EVAL_CACHE_SHARDS};

const SEED: u64 = 0x5045_5246_0003_0002;
const SIMS: usize = 64;
const LEAF_BATCH: usize = 8;
const LEAVES: usize = 10_000;
const PLY_CAP: usize = 40;
/// A net version lasts this many searches, so the drive crosses many bumps.
const SEARCHES_PER_VERSION: usize = 5;

/// The cache's key before the position key, verbatim: SHA-256 over every field of the built graph.
fn sha_key(g: &AxisGraph) -> [u8; 32] {
    fn words(buf: &mut Vec<u8>, ws: impl ExactSizeIterator<Item = u32>) {
        buf.extend_from_slice(&(ws.len() as u64).to_le_bytes());
        for w in ws {
            buf.extend_from_slice(&w.to_le_bytes());
        }
    }
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
    } = g;
    let mut buf = Vec::new();
    words(&mut buf, node_feat.0.iter().map(|x| x.to_bits()));
    words(&mut buf, edge_index.src.iter().copied());
    words(&mut buf, edge_index.dst.iter().copied());
    words(&mut buf, edge_attr.0.iter().map(|x| x.to_bits()));
    words(&mut buf, legal_mask.iter().map(|&b| u32::from(b)));
    words(&mut buf, stone_mask.iter().map(|&b| u32::from(b)));
    words(&mut buf, policy_scatter_index.0.iter().map(|&x| x as u32));
    words(&mut buf, node_coords.iter().map(|&x| x as u32));
    words(&mut buf, legal_node_gather.iter().copied());
    buf.extend_from_slice(&n_stones.to_le_bytes());
    buf.extend_from_slice(&n_nodes_checksum.to_le_bytes());
    buf.extend_from_slice(&window_center.0.to_le_bytes());
    buf.extend_from_slice(&window_center.1.to_le_bytes());
    buf.extend_from_slice(&current_player.to_le_bytes());
    buf.push(*builder_impl);
    Sha256::digest(&buf).into()
}

/// What one drive read: per `infer_and_expand_graph` call, the net version and the SHA of each graph served.
struct Drive {
    calls: Vec<(u64, Vec<[u8; 32]>)>,
    served_leaves: u64,
    gpu_evals: u64,
}

/// A deterministic server: a prior decaying from the legal centroid and a value hashed from the legal set.
fn answer(spec_logits: usize, g: &AxisGraph) -> (Vec<f32>, f32) {
    let coords: Vec<(i32, i32)> = g
        .legal_node_gather
        .iter()
        .map(|&r| {
            (
                g.node_coords[r as usize * 2],
                g.node_coords[r as usize * 2 + 1],
            )
        })
        .collect();
    let n = coords.len().max(1) as f32;
    let cq = coords.iter().map(|c| c.0 as f32).sum::<f32>() / n;
    let cr = coords.iter().map(|c| c.1 as f32).sum::<f32>() / n;
    let raw: Vec<f32> = coords
        .iter()
        .map(|&(q, r)| {
            let (dq, dr) = (q as f32 - cq, r as f32 - cr);
            (-1.5 * dq.abs().max(dr.abs()).max((dq + dr).abs())).exp()
        })
        .collect();
    let total: f32 = raw.iter().sum();
    let mut s = coords
        .iter()
        .fold(coords.len() as u64 ^ spec_logits as u64, |h, &(q, r)| {
            h.wrapping_mul(31)
                .wrapping_add(((q as u64) << 32) ^ (r as u32 as u64))
        });
    let value = (mantis_core::board::zobrist::splitmix64_next(&mut s) % 1801) as f32 / 1000.0 - 0.9;
    (raw.into_iter().map(|x| x / total).collect(), value)
}

fn drive(cache_capacity: usize) -> Drive {
    let spec = mantis_encoding::lookup_or_panic("gnn_axis_r8");
    let geometry = crate::runner::params::resolve_geometry(spec).expect("a registry graph spec");
    let queue = GraphQueue::with_eval_cache(1, LEAF_BATCH, cache_capacity);
    let received: Arc<Mutex<Vec<[u8; 32]>>> = Arc::default();
    let producer = {
        let (queue, received) = (queue.clone(), Arc::clone(&received));
        thread::spawn(move || loop {
            let batch = queue.pop_graph_batch(LEAF_BATCH, 5);
            if batch.is_empty() {
                if queue.is_closed() {
                    break;
                }
                continue;
            }
            let mut ids = Vec::with_capacity(batch.len());
            let mut results = Vec::with_capacity(batch.len());
            for (id, g) in batch {
                received.lock().expect("unpoisoned").push(sha_key(&g));
                let coords: Vec<(i32, i32)> = g
                    .legal_node_gather
                    .iter()
                    .map(|&r| {
                        (
                            g.node_coords[r as usize * 2],
                            g.node_coords[r as usize * 2 + 1],
                        )
                    })
                    .collect();
                let (probs, value) = answer(spec.policy_logit_count, &g);
                ids.push(id);
                results.push(
                    crate::records::assemble_ls_from_gnn_probs(
                        spec.policy_logit_count,
                        &probs,
                        &g.policy_scatter_index.0,
                        &coords,
                    )
                    .map(|ls| (ls, value)),
                );
            }
            queue.submit_graph_results(&ids, results);
        })
    };
    let (version, running) = (AtomicU64::new(0), AtomicBool::new(true));
    let (served, gpu, inline, table) = (
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
    );
    let infer = InferContext {
        graph_queue: &queue,
        spec,
        model_version: &version,
        running: &running,
        win_length: geometry.win_length,
        graph_radius: geometry.graph_radius,
        served_leaves: &served,
        gpu_evals: &gpu,
        inline_descents: &inline,
        tt_hits: &table,
    };
    let board_geometry = BoardGeometry {
        legal_move_radius: spec.legal_move_radius as i32,
        cluster_window_size: spec.cluster_window_size.unwrap_or(spec.board_size),
    };
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut tree = MCTSTree::new_full(1.5, VIRTUAL_LOSS_PENALTY, 0.25);
    let mut calls: Vec<(u64, Vec<[u8; 32]>)> = Vec::new();
    let mut searches = 0usize;
    let mut call = |tree: &mut MCTSTree, batch: usize| -> usize {
        let before = received.lock().expect("unpoisoned").len();
        let v = version.load(Ordering::Acquire);
        let (n, unserved) = infer_and_expand_graph(
            tree,
            LeafSelection::Batch(batch),
            geometry.agg_trunk_sz,
            infer,
        )
        .unwrap_or_else(|e| panic!("the drive's leaves serve: {e}"));
        calls.push((v, received.lock().expect("unpoisoned")[before..].to_vec()));
        n + unserved
    };
    'games: while served.load(Ordering::Relaxed) < LEAVES as u64 {
        let mut board = Board::with_geometry(board_geometry);
        for _ in 0..PLY_CAP {
            if searches.is_multiple_of(SEARCHES_PER_VERSION) {
                version.fetch_add(1, Ordering::AcqRel);
            }
            searches += 1;
            tree.new_game(board.clone());
            let mut sims = call(&mut tree, 1);
            while sims < SIMS {
                let n = call(&mut tree, LEAF_BATCH.min(SIMS - sims));
                if n == 0 {
                    break;
                }
                sims += n;
            }
            let top = tree.get_top_visits(3);
            if top.is_empty() {
                continue 'games;
            }
            let (q, r) = top[rng.random_range(0..top.len())].0;
            board.apply_move(q, r).expect("a searched child is legal");
            if board.winner().is_some() {
                continue 'games;
            }
        }
    }
    queue.close();
    producer.join().expect("the producer exits");
    Drive {
        calls,
        served_leaves: served.load(Ordering::Relaxed),
        gpu_evals: gpu.load(Ordering::Relaxed),
    }
}

/// Cache off, every leaf reaches the producer and the oracle replays the SHA key; cache on, the misses do. Legal
/// play fixes side and moves_remaining by the stone count, so those and off-table cells are the unit pins'.
#[test]
fn the_position_key_misses_exactly_the_leaves_the_sha_key_missed() {
    let every = drive(0);
    let keyed = drive(EVAL_CACHE_CAPACITY);
    assert_eq!(
        every.calls.len(),
        keyed.calls.len(),
        "the two drives diverged"
    );
    assert_eq!(
        every.served_leaves, keyed.served_leaves,
        "the two drives diverged"
    );
    assert!(
        every.served_leaves >= LEAVES as u64,
        "{} leaves",
        every.served_leaves
    );
    assert_eq!(
        every.gpu_evals, every.served_leaves,
        "the cache-off drive served every leaf"
    );
    // The oracle: a graph misses unless its SHA was stored under this version before this call.
    let mut stored: FxHashSet<(u64, [u8; 32])> = FxHashSet::default();
    let mut per_version = std::collections::HashMap::<u64, usize>::new();
    let mut oracle: Vec<Vec<[u8; 32]>> = Vec::with_capacity(every.calls.len());
    for (v, all) in &every.calls {
        let want: Vec<[u8; 32]> = all
            .iter()
            .filter(|k| !stored.contains(&(*v, **k)))
            .copied()
            .collect();
        *per_version.entry(*v).or_default() += want.len();
        stored.extend(all.iter().map(|k| (*v, *k)));
        oracle.push(want);
    }
    // Below one shard's capacity per version, no entry is ever evicted: eviction order is shard-local.
    let shard = EVAL_CACHE_CAPACITY / EVAL_CACHE_SHARDS;
    assert!(
        per_version.values().all(|&m| m < shard),
        "a version could evict: {per_version:?}"
    );
    for (i, ((want, (v, _)), (kv, got))) in oracle
        .iter()
        .zip(&every.calls)
        .zip(&keyed.calls)
        .enumerate()
    {
        assert_eq!(v, kv, "call {i}: another net version");
        assert_eq!(
            want, got,
            "call {i}: the position key's misses are not the SHA key's"
        );
    }
    let oracle_misses: u64 = oracle.iter().map(|w| w.len() as u64).sum();
    assert_eq!(keyed.gpu_evals, oracle_misses);
    assert!(oracle_misses < every.served_leaves, "the cache never fired");
    println!(
        "{} leaves over {} calls, {} misses ({:.1} % hits), {} versions",
        every.served_leaves,
        every.calls.len(),
        oracle_misses,
        100.0 * (1.0 - oracle_misses as f64 / every.served_leaves as f64),
        per_version.len()
    );
}
