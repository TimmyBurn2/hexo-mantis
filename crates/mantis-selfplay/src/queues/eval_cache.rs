//! The exact per-net evaluation cache: a served leaf whose position was already evaluated by the
//! same net version replays that evaluation instead of queueing a GPU forward.
//!
//! EXACT by construction: the key is the whole position (the sorted stone set, the side to move
//! and the moves remaining), compared in full by the map, never a hash alone; an entry is served
//! only under the net version it was computed by, and a shard seeing a newer version drops
//! everything it holds. Bounded: at most `capacity` entries, evicted first-in first-out.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use fxhash::FxHashMap;
use mantis_search::LegalSetPolicy;

use crate::poison::lock_or_recover;

/// Entries across all shards. A `LegalSetPolicy` row is bounded by the spec's trunk window,
/// so the cache's memory is bounded by this count times that row.
pub const EVAL_CACHE_CAPACITY: usize = 32_768;
const SHARDS: usize = 16;

/// The identity of a position: the stones in coordinate order, the side to move, the moves left.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PositionKey {
    stones: Box<[(i32, i32, i8)]>,
    current_player: i8,
    moves_remaining: u8,
}

impl PositionKey {
    /// Canonicalises `stones` (any order) into the key.
    #[must_use]
    pub fn new(mut stones: Vec<(i32, i32, i8)>, current_player: i8, moves_remaining: u8) -> Self {
        stones.sort_unstable();
        Self {
            stones: stones.into_boxed_slice(),
            current_player,
            moves_remaining,
        }
    }

    fn shard(&self) -> usize {
        let mut h = u64::from(self.moves_remaining) ^ (u64::from(self.current_player as u8) << 8);
        for &(q, r, c) in self.stones.iter() {
            h = h.rotate_left(7)
                ^ (q as u32 as u64)
                ^ ((r as u32 as u64) << 32)
                ^ u64::from(c as u8);
        }
        (h % SHARDS as u64) as usize
    }
}

/// One served evaluation: the policy row, the value and the builder's window centre.
#[derive(Clone, Debug)]
pub struct CachedEval {
    pub policy: LegalSetPolicy,
    pub value: f32,
    pub center: (i32, i32),
}

#[derive(Default)]
struct Shard {
    version: u64,
    map: FxHashMap<Arc<PositionKey>, CachedEval>,
    order: VecDeque<Arc<PositionKey>>,
}

/// The sharded cache the runner's workers share through the graph queue.
pub struct EvalCache {
    shards: Vec<Mutex<Shard>>,
    per_shard: usize,
}

impl EvalCache {
    /// A cache of at most `capacity` entries; `0` never stores and never hits.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            shards: (0..SHARDS).map(|_| Mutex::new(Shard::default())).collect(),
            per_shard: capacity / SHARDS,
        }
    }

    /// The evaluation `key` received under net `version`, if this cache holds it.
    #[must_use]
    pub fn get(&self, key: &PositionKey, version: u64) -> Option<CachedEval> {
        if self.per_shard == 0 {
            return None;
        }
        let shard = lock_or_recover(&self.shards[key.shard()], None);
        if shard.version != version {
            return None;
        }
        shard.map.get(key).cloned()
    }

    /// Stores `eval` for `key` under net `version`; a newer version clears the shard first.
    pub fn put(&self, key: PositionKey, version: u64, eval: CachedEval) {
        if self.per_shard == 0 {
            return;
        }
        let mut shard = lock_or_recover(&self.shards[key.shard()], None);
        if version < shard.version {
            return;
        }
        if version > shard.version {
            shard.map.clear();
            shard.order.clear();
            shard.version = version;
        }
        if shard.map.contains_key(&key) {
            return;
        }
        while shard.order.len() >= self.per_shard {
            if let Some(old) = shard.order.pop_front() {
                shard.map.remove(&old);
            }
        }
        let key = Arc::new(key);
        shard.order.push_back(Arc::clone(&key));
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval(v: f32) -> CachedEval {
        CachedEval {
            policy: LegalSetPolicy {
                dense: vec![v; 4],
                ..Default::default()
            },
            value: v,
            center: (1, 2),
        }
    }

    #[test]
    fn a_hit_needs_the_same_position_and_version() {
        let cache = EvalCache::new(64);
        let key = PositionKey::new(vec![(1, 0, 1), (0, 0, -1)], 1, 2);
        cache.put(key.clone(), 3, eval(0.5));
        let same_in_other_order = PositionKey::new(vec![(0, 0, -1), (1, 0, 1)], 1, 2);
        assert_eq!(
            cache.get(&same_in_other_order, 3).map(|e| e.value),
            Some(0.5)
        );
        assert!(
            cache.get(&key, 4).is_none(),
            "another net version must miss"
        );
        assert!(cache
            .get(&PositionKey::new(vec![(1, 0, 1), (0, 0, -1)], -1, 2), 3)
            .is_none());
        assert!(cache
            .get(&PositionKey::new(vec![(1, 0, 1), (0, 0, -1)], 1, 1), 3)
            .is_none());
    }

    #[test]
    fn a_newer_version_drops_the_older_entries() {
        let cache = EvalCache::new(64);
        let key = PositionKey::new(vec![(0, 0, 1)], -1, 1);
        cache.put(key.clone(), 1, eval(0.1));
        cache.put(key.clone(), 2, eval(0.2));
        assert!(cache.get(&key, 1).is_none());
        assert_eq!(cache.get(&key, 2).map(|e| e.value), Some(0.2));
        cache.put(key.clone(), 1, eval(0.9));
        assert_eq!(
            cache.get(&key, 2).map(|e| e.value),
            Some(0.2),
            "a stale put must not overwrite"
        );
    }

    #[test]
    fn the_entry_count_never_exceeds_the_capacity() {
        let cache = EvalCache::new(SHARDS * 4);
        for i in 0..10_000 {
            cache.put(PositionKey::new(vec![(i, -i, 1)], 1, 1), 0, eval(0.0));
        }
        assert!(cache.len() <= SHARDS * 4, "held {} entries", cache.len());
        assert!(!cache.is_empty());
    }

    #[test]
    fn a_zero_capacity_cache_is_off() {
        let cache = EvalCache::new(0);
        let key = PositionKey::new(vec![], 1, 1);
        cache.put(key.clone(), 0, eval(0.3));
        assert!(cache.get(&key, 0).is_none() && cache.is_empty());
    }
}
