//! The per-net eval cache: a leaf whose hashed `AxisGraph` this net version already evaluated replays it.

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
        // Destructured, so a field added to the encoded input fails to compile until it is keyed.
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
        let mut buf: Vec<u8> = Vec::with_capacity(
            4 * (node_feat.0.len() + edge_attr.0.len() + 2 * edge_index.src.len())
                + 4 * (policy_scatter_index.0.len() + node_coords.len() + legal_node_gather.len())
                + legal_mask.len()
                + stone_mask.len()
                + 128,
        );
        put_words(&mut buf, node_feat.0.iter().map(|x| x.to_bits()));
        put_words(&mut buf, edge_index.src.iter().copied());
        put_words(&mut buf, edge_index.dst.iter().copied());
        put_words(&mut buf, edge_attr.0.iter().map(|x| x.to_bits()));
        put_words(&mut buf, legal_mask.iter().map(|&b| u32::from(b)));
        put_words(&mut buf, stone_mask.iter().map(|&b| u32::from(b)));
        put_words(&mut buf, policy_scatter_index.0.iter().map(|&x| x as u32));
        put_words(&mut buf, node_coords.iter().map(|&x| x as u32));
        put_words(&mut buf, legal_node_gather.iter().copied());
        buf.extend_from_slice(&n_stones.to_le_bytes());
        buf.extend_from_slice(&n_nodes_checksum.to_le_bytes());
        buf.extend_from_slice(&window_center.0.to_le_bytes());
        buf.extend_from_slice(&window_center.1.to_le_bytes());
        buf.extend_from_slice(&current_player.to_le_bytes());
        buf.push(*builder_impl);
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
        let cap = self.policy.overflow.capacity();
        let buckets = if cap == 0 {
            0
        } else {
            (cap * 8 / 7 + 1).next_power_of_two()
        };
        let overflow_slot = std::mem::size_of::<((i32, i32), f32)>() + 1;
        std::mem::size_of::<Self>()
            + 2 * std::mem::size_of::<GraphKey>()
            + self.policy.dense.capacity() * std::mem::size_of::<f32>()
            + buckets * overflow_slot
            + 16
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

    /// Stores `eval` under `version`; a newer version clears the shard, an entry past its budget is refused.
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
#[path = "eval_cache_tests.rs"]
mod tests;
