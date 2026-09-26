//! The per-net evaluation cache: a leaf whose ENCODED INPUT the same net version already evaluated
//! replays that evaluation instead of queueing a GPU forward.
//!
//! The key is the net version plus a 128-bit digest of every field of the built `AxisGraph`, the
//! exact input the net reads, so two leaves share an entry only when the net would see identical
//! tensors; a replay then differs from a fresh forward by no more than the served path's own
//! batch-size spread (the entry was computed in another pop). Bounded by an entry count AND a byte
//! budget, both enforced at insert and evicted first-in first-out; a shard seeing a newer version
//! drops everything it holds.

use std::collections::VecDeque;
use std::sync::Mutex;

use fxhash::FxHashMap;
use mantis_graph::AxisGraph;
use mantis_search::LegalSetPolicy;
use sha2::{Digest, Sha256};

use crate::poison::lock_or_recover;

/// Entries across all shards.
pub const EVAL_CACHE_CAPACITY: usize = 32_768;
/// Bytes across all shards, counted per entry by [`CachedEval::bytes`].
pub const EVAL_CACHE_BYTES: usize = 256 << 20;
const SHARDS: usize = 16;

/// The identity of one encoded input: a SHA-256 prefix over every `AxisGraph` field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GraphKey(u128);

impl GraphKey {
    /// The digest of every field of `g`, each length-prefixed so no two layouts share bytes.
    #[must_use]
    pub fn of(g: &AxisGraph) -> Self {
        let mut buf: Vec<u8> = Vec::with_capacity(
            4 * (g.node_feat.0.len() + g.edge_attr.0.len() + 2 * g.edge_index.src.len())
                + 4 * (g.policy_scatter_index.0.len() + g.node_coords.len())
                + 4 * g.legal_node_gather.len()
                + g.legal_mask.len()
                + g.stone_mask.len()
                + 128,
        );
        put_words(&mut buf, g.node_feat.0.iter().map(|x| x.to_bits()));
        put_words(&mut buf, g.edge_index.src.iter().copied());
        put_words(&mut buf, g.edge_index.dst.iter().copied());
        put_words(&mut buf, g.edge_attr.0.iter().map(|x| x.to_bits()));
        put_words(&mut buf, g.legal_mask.iter().map(|&b| u32::from(b)));
        put_words(&mut buf, g.stone_mask.iter().map(|&b| u32::from(b)));
        put_words(&mut buf, g.policy_scatter_index.0.iter().map(|&x| x as u32));
        put_words(&mut buf, g.node_coords.iter().map(|&x| x as u32));
        put_words(&mut buf, g.legal_node_gather.iter().copied());
        buf.extend_from_slice(&g.n_stones.to_le_bytes());
        buf.extend_from_slice(&g.n_nodes_checksum.to_le_bytes());
        buf.extend_from_slice(&g.window_center.0.to_le_bytes());
        buf.extend_from_slice(&g.window_center.1.to_le_bytes());
        buf.extend_from_slice(&g.current_player.to_le_bytes());
        buf.push(g.builder_impl);
        let digest = Sha256::digest(&buf);
        let mut prefix = [0u8; 16];
        prefix.copy_from_slice(&digest[..16]);
        Self(u128::from_le_bytes(prefix))
    }

    /// The key as 32 lowercase hex digits.
    #[must_use]
    pub fn hex(self) -> String {
        format!("{:032x}", self.0)
    }

    fn shard(self) -> usize {
        (self.0 % SHARDS as u128) as usize
    }
}

/// Appends a length prefix and then each word little-endian.
fn put_words(buf: &mut Vec<u8>, words: impl ExactSizeIterator<Item = u32>) {
    buf.extend_from_slice(&(words.len() as u64).to_le_bytes());
    for w in words {
        buf.extend_from_slice(&w.to_le_bytes());
    }
}

/// One served evaluation: the policy row, the value and the builder's window centre.
#[derive(Clone, Debug)]
pub struct CachedEval {
    pub policy: LegalSetPolicy,
    pub value: f32,
    pub center: (i32, i32),
}

impl CachedEval {
    /// An upper bound on the heap and inline bytes this entry holds, key and queue slot included.
    #[must_use]
    pub fn bytes(&self) -> usize {
        let overflow_slot = std::mem::size_of::<((i32, i32), f32)>() + 1;
        std::mem::size_of::<Self>()
            + 2 * std::mem::size_of::<GraphKey>()
            + self.policy.dense.capacity() * std::mem::size_of::<f32>()
            + self.policy.overflow.capacity() * overflow_slot
    }
}

#[derive(Default)]
struct Shard {
    version: u64,
    bytes: usize,
    map: FxHashMap<GraphKey, CachedEval>,
    order: VecDeque<GraphKey>,
}

impl Shard {
    fn evict_oldest(&mut self) {
        if let Some(old) = self.order.pop_front() {
            if let Some(gone) = self.map.remove(&old) {
                self.bytes -= gone.bytes();
            }
        }
    }
}

/// The sharded cache the runner's workers share through the graph queue.
pub struct EvalCache {
    shards: Vec<Mutex<Shard>>,
    per_shard: usize,
    per_shard_bytes: usize,
}

impl EvalCache {
    /// A cache of at most `capacity` entries and `byte_budget` bytes; `0` for either never stores.
    #[must_use]
    pub fn new(capacity: usize, byte_budget: usize) -> Self {
        Self {
            shards: (0..SHARDS).map(|_| Mutex::new(Shard::default())).collect(),
            per_shard: capacity / SHARDS,
            per_shard_bytes: byte_budget / SHARDS,
        }
    }

    fn off(&self) -> bool {
        self.per_shard == 0 || self.per_shard_bytes == 0
    }

    /// The evaluation `key` received under net `version`, if this cache holds it.
    #[must_use]
    pub fn get(&self, key: GraphKey, version: u64) -> Option<CachedEval> {
        if self.off() {
            return None;
        }
        let shard = lock_or_recover(&self.shards[key.shard()], None);
        if shard.version != version {
            return None;
        }
        shard.map.get(&key).cloned()
    }

    /// Stores `eval` for `key` under net `version`; a newer version clears the shard first, and an
    /// entry larger than a shard's whole budget is never stored.
    pub fn put(&self, key: GraphKey, version: u64, eval: CachedEval) {
        let size = eval.bytes();
        if self.off() || size > self.per_shard_bytes {
            return;
        }
        let mut shard = lock_or_recover(&self.shards[key.shard()], None);
        if version < shard.version {
            return;
        }
        if version > shard.version {
            shard.map.clear();
            shard.order.clear();
            shard.bytes = 0;
            shard.version = version;
        }
        if shard.map.contains_key(&key) {
            return;
        }
        while shard.order.len() >= self.per_shard || shard.bytes + size > self.per_shard_bytes {
            shard.evict_oldest();
        }
        shard.bytes += size;
        shard.order.push_back(key);
        shard.map.insert(key, eval);
    }

    /// Entries held right now, across all shards.
    #[must_use]
    pub fn len(&self) -> usize {
        self.shards
            .iter()
            .map(|s| lock_or_recover(s, None).map.len())
            .sum()
    }

    /// Whether the cache holds nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bytes held right now, as [`CachedEval::bytes`] counts them.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.shards
            .iter()
            .map(|s| lock_or_recover(s, None).bytes)
            .sum()
    }
}

#[cfg(test)]
mod tests {
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

    /// The planted-incomplete-key control: a key that skipped any one field would pass the arm
    /// that perturbs it, so every field of the encoded input must move the key.
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
}
