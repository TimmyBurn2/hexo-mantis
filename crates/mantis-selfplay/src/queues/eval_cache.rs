//! The per-net eval cache: a leaf whose position this net version already evaluated replays it, unbuilt.

use std::collections::VecDeque;
use std::sync::Mutex;

use fxhash::FxHashMap;
use mantis_core::board::zobrist::splitmix64_next;
use mantis_graph::{EmptyEdges, BUILDER_IMPL_NATIVE};
use mantis_search::LegalSetPolicy;

use crate::poison::lock_or_recover;
use crate::queues::graph::check_leaf_request;

/// Entries across all shards.
pub const EVAL_CACHE_CAPACITY: usize = 32_768;
/// Bytes across all shards, counted per entry by [`CachedEval::bytes`].
pub const EVAL_CACHE_BYTES: usize = 256 << 20;
/// Shards the cache splits its entries and bytes over, by key.
pub const EVAL_CACHE_SHARDS: usize = 16;

/// The identity of one leaf's evaluation: a Zobrist key over every input `build_leaf_graph` reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LeafKey(u128);

impl LeafKey {
    /// The key of `build_leaf_graph`'s request after its seam guards; `Err` is the builder's own refusal, verbatim.
    pub fn of(
        stones: &[(i64, i64, i64)],
        current_player: i64,
        moves_remaining: i64,
        win_length: u8,
        radius: u16,
        trunk_size: i32,
        empty_edges: EmptyEdges,
    ) -> Result<Self, String> {
        check_leaf_request(stones, current_player, moves_remaining, radius)?;
        let geometry = u64::from(win_length)
            | (u64::from(radius) << 8)
            | (u64::from(trunk_size as u32) << 24)
            | (u64::from(BUILDER_IMPL_NATIVE) << 56);
        let mut key = word(SIDE_TAG, current_player as u64)
            ^ word(LEFT_TAG, moves_remaining as u64)
            ^ word(GEOMETRY_TAG, geometry)
            ^ word(EDGES_TAG, u64::from(empty_edges == EmptyEdges::Pruned));
        // An XOR over the stones: order-free; the stones must be distinct cells, as a `Board`'s are (a pair cancels).
        for &(q, r, p) in stones {
            key ^= stone_word(q, r, p);
        }
        Ok(Self(key))
    }

    /// The key as 32 lowercase hex digits.
    #[must_use]
    pub fn hex(self) -> String {
        format!("{:032x}", self.0)
    }

    fn shard(self) -> usize {
        (self.0 % EVAL_CACHE_SHARDS as u128) as usize
    }
}

/// One stone's word, injective by construction: the low half a bijection of the cell, the high half of cell and owner.
fn stone_word(q: i64, r: i64, player: i64) -> u128 {
    let cell = (u64::from(q as u32) << 32) | u64::from(r as u32);
    let (mut low, mut high) = (cell, cell ^ if player == 1 { P1_TAG } else { P2_TAG });
    (u128::from(splitmix64_next(&mut high)) << 64) | u128::from(splitmix64_next(&mut low))
}

const P1_TAG: u64 = 0x5031_5354_4f4e_4501;
const P2_TAG: u64 = 0x5032_5354_4f4e_4502;
const SIDE_TAG: u64 = 0x5349_4445_0000_0001;
const LEFT_TAG: u64 = 0x4c45_4654_0000_0002;
const GEOMETRY_TAG: u64 = 0x4745_4f4d_0000_0003;
const EDGES_TAG: u64 = 0x4544_4745_0000_0004;

/// A 128-bit word for `value` under `tag`, from two draws of one splitmix64 stream.
fn word(tag: u64, value: u64) -> u128 {
    let mut state = tag ^ value.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    let lo = splitmix64_next(&mut state);
    let hi = splitmix64_next(&mut state);
    (u128::from(hi) << 64) | u128::from(lo)
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
            + 2 * std::mem::size_of::<LeafKey>()
            + self.policy.dense.capacity() * std::mem::size_of::<f32>()
            + buckets * overflow_slot
            + 16
    }
}

#[derive(Default)]
struct Shard {
    version: u64,
    bytes: usize,
    map: FxHashMap<LeafKey, CachedEval>,
    order: VecDeque<LeafKey>,
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
            shards: (0..EVAL_CACHE_SHARDS)
                .map(|_| Mutex::new(Shard::default()))
                .collect(),
            per_shard: capacity / EVAL_CACHE_SHARDS,
            per_shard_bytes: byte_budget / EVAL_CACHE_SHARDS,
        }
    }

    fn off(&self) -> bool {
        self.per_shard == 0 || self.per_shard_bytes == 0
    }

    /// The evaluation `key` received under net `version`, if this cache holds it.
    #[must_use]
    pub fn get(&self, key: LeafKey, version: u64) -> Option<CachedEval> {
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
    pub fn put(&self, key: LeafKey, version: u64, eval: CachedEval) {
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
