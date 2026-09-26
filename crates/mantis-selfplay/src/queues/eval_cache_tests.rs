//! The eval cache's key completeness, version, capacity and byte-budget arms.

use super::*;
use crate::queues::build_leaf_graph;

fn eval(v: f32, dense: usize) -> CachedEval {
    CachedEval {
        policy: LegalSetPolicy {
            dense: vec![v; dense],
            ..Default::default()
        },
        value: v,
        center: (1, 2),
    }
}

fn graph(stones: &[(i64, i64, i64)], player: i64, moves_remaining: i64) -> AxisGraph {
    build_leaf_graph(stones, player, moves_remaining, 6, 6, 19).expect("a legal test position")
}

fn key(n: u128) -> GraphKey {
    GraphKey(n)
}

#[test]
fn the_same_encoded_input_is_the_same_key_in_any_stone_order() {
    let a = graph(&[(0, 0, 1), (1, 0, -1), (0, 1, -1)], 1, 2);
    let b = graph(&[(0, 1, -1), (0, 0, 1), (1, 0, -1)], 1, 2);
    assert_eq!(GraphKey::of(&a), GraphKey::of(&b));
}

/// The planted-incomplete-key control: every field of the encoded input must move the key.
#[test]
fn every_field_of_the_encoded_input_moves_the_key() {
    let base = graph(&[(0, 0, 1), (1, 0, -1), (0, 1, -1)], 1, 2);
    let k = GraphKey::of(&base);
    let mut arms: Vec<(&str, AxisGraph)> = Vec::new();
    let mut g = base.clone();
    g.node_feat.0[0] += 1.0;
    arms.push(("node_feat", g));
    let mut g = base.clone();
    g.edge_index.src[0] ^= 1;
    arms.push(("edge_index.src", g));
    let mut g = base.clone();
    g.edge_index.dst[0] ^= 1;
    arms.push(("edge_index.dst", g));
    let mut g = base.clone();
    g.edge_attr.0[0] += 1.0;
    arms.push(("edge_attr", g));
    let mut g = base.clone();
    g.legal_mask[0] = !g.legal_mask[0];
    arms.push(("legal_mask", g));
    let mut g = base.clone();
    g.stone_mask[0] = !g.stone_mask[0];
    arms.push(("stone_mask", g));
    let mut g = base.clone();
    g.policy_scatter_index.0[0] += 1;
    arms.push(("policy_scatter_index", g));
    let mut g = base.clone();
    g.node_coords[0] += 1;
    arms.push(("node_coords", g));
    let mut g = base.clone();
    g.legal_node_gather[0] += 1;
    arms.push(("legal_node_gather", g));
    let mut g = base.clone();
    g.n_stones += 1;
    arms.push(("n_stones", g));
    let mut g = base.clone();
    g.n_nodes_checksum += 1;
    arms.push(("n_nodes_checksum", g));
    let mut g = base.clone();
    g.window_center.0 += 1;
    arms.push(("window_center.q", g));
    let mut g = base.clone();
    g.window_center.1 += 1;
    arms.push(("window_center.r", g));
    let mut g = base.clone();
    g.current_player = -g.current_player;
    arms.push(("current_player", g));
    let mut g = base.clone();
    g.builder_impl ^= 1;
    arms.push(("builder_impl", g));
    for (field, g) in &arms {
        assert_ne!(GraphKey::of(g), k, "{field} does not move the key");
    }
}

#[test]
fn positions_one_stone_or_one_side_apart_are_different_keys() {
    let stones = [(0, 0, 1), (1, 0, -1), (0, 1, -1)];
    let k = GraphKey::of(&graph(&stones, 1, 2));
    assert_ne!(k, GraphKey::of(&graph(&stones, 1, 1)));
    assert_ne!(k, GraphKey::of(&graph(&stones[..2], 1, 2)));
}

/// The stale-version control: an entry is served only under the version that computed it.
#[test]
fn a_hit_needs_the_same_key_and_version() {
    let cache = EvalCache::new(64, EVAL_CACHE_BYTES);
    cache.put(key(7), 3, eval(0.5, 4));
    assert_eq!(cache.get(key(7), 3).map(|e| e.value), Some(0.5));
    assert!(
        cache.get(key(7), 4).is_none(),
        "another net version must miss"
    );
    assert!(cache.get(key(8), 3).is_none());
}

#[test]
fn a_newer_version_drops_the_older_entries() {
    let cache = EvalCache::new(64, EVAL_CACHE_BYTES);
    cache.put(key(1), 1, eval(0.1, 4));
    cache.put(key(1), 2, eval(0.2, 4));
    assert!(cache.get(key(1), 1).is_none());
    assert_eq!(cache.get(key(1), 2).map(|e| e.value), Some(0.2));
    cache.put(key(1), 1, eval(0.9, 4));
    assert_eq!(
        cache.get(key(1), 2).map(|e| e.value),
        Some(0.2),
        "a stale put must not overwrite"
    );
}

#[test]
fn the_entry_count_never_exceeds_the_capacity() {
    let cache = EvalCache::new(SHARDS * 4, EVAL_CACHE_BYTES);
    for i in 0..10_000u128 {
        cache.put(key(i), 0, eval(0.0, 4));
    }
    assert!(cache.len() <= SHARDS * 4, "held {} entries", cache.len());
    assert!(!cache.is_empty());
}

#[test]
fn the_byte_budget_is_enforced_at_insert() {
    let one = eval(0.0, 362).bytes();
    let budget = SHARDS * one * 3;
    let cache = EvalCache::new(EVAL_CACHE_CAPACITY, budget);
    for i in 0..10_000u128 {
        cache.put(key(i), 0, eval(0.0, 362));
    }
    assert!(
        cache.bytes() <= budget,
        "held {} bytes over {budget}",
        cache.bytes()
    );
    assert_eq!(cache.bytes(), cache.len() * one);
    cache.put(key(u128::MAX), 0, eval(0.0, budget));
    assert!(
        cache.get(key(u128::MAX), 0).is_none(),
        "an entry past a shard's whole budget is refused"
    );
}

#[test]
fn a_zero_capacity_or_budget_cache_is_off() {
    for cache in [EvalCache::new(0, EVAL_CACHE_BYTES), EvalCache::new(64, 0)] {
        cache.put(key(1), 0, eval(0.3, 4));
        assert!(cache.get(key(1), 0).is_none() && cache.is_empty());
    }
}
