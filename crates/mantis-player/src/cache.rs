//! The per-game exact cache, `GameEvalCache` ported: a leaf served once in a game is served from here after.

use std::collections::{HashMap, VecDeque};

use mantis_search::LegalSetPolicy;
use mantis_selfplay::queues::eval_cache::LeafKey;

/// A game's entries at most; past it the oldest goes and a miss is served again (`eval_cache.MAX_ENTRIES`).
pub const CACHE_ENTRIES: usize = 8192;

/// One served leaf, exactly as the net returned it.
#[derive(Debug, Clone)]
pub struct Entry {
    pub policy: LegalSetPolicy,
    pub value: f32,
    pub center: (i32, i32),
}

/// The cache's rows, cumulative over the player's life: `positions = hits + served + in_batch_repeats`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheRows {
    pub calls: u64,
    pub positions: u64,
    pub hits: u64,
    pub served: u64,
    pub in_batch_repeats: u64,
    pub all_hit_calls: u64,
    pub evicted: u64,
}

/// Where one leaf of a call gets its outputs: the cache, or the call's `i`-th fresh evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Hit,
    Fresh(usize),
}

/// One game's entries in insertion order, so the oldest is the one evicted and a hit refreshes nothing.
#[derive(Debug, Default)]
pub struct GameCache {
    entries: HashMap<LeafKey, Entry>,
    order: VecDeque<LeafKey>,
    rows: CacheRows,
}

impl GameCache {
    /// Empty the entries: no game reads another's evaluations. The rows stay cumulative.
    pub fn new_game(&mut self) {
        self.entries.clear();
        self.order.clear();
    }

    #[must_use]
    pub fn rows(&self) -> CacheRows {
        self.rows
    }

    /// Each key's source and the fresh keys in first-seen order, a key repeated in the call evaluated once.
    #[must_use]
    pub fn plan(&self, keys: &[LeafKey]) -> (Vec<Source>, Vec<LeafKey>) {
        let mut fresh: Vec<LeafKey> = Vec::new();
        let mut index: HashMap<LeafKey, usize> = HashMap::new();
        let sources = keys
            .iter()
            .map(|key| {
                if self.entries.contains_key(key) {
                    return Source::Hit;
                }
                Source::Fresh(*index.entry(*key).or_insert_with(|| {
                    fresh.push(*key);
                    fresh.len() - 1
                }))
            })
            .collect();
        (sources, fresh)
    }

    /// The call's outputs in leaf order, then `fresh` remembered and the rows counted, as `infer_batch_ls` does.
    #[must_use]
    pub fn finish(
        &mut self,
        keys: &[LeafKey],
        sources: &[Source],
        fresh_keys: &[LeafKey],
        fresh: Vec<Entry>,
    ) -> Vec<Entry> {
        let hits = sources.iter().filter(|s| **s == Source::Hit).count();
        let out = keys
            .iter()
            .zip(sources)
            .filter_map(|(key, source)| match source {
                Source::Hit => self.entries.get(key).cloned(),
                Source::Fresh(i) => fresh.get(*i).cloned(),
            })
            .collect();
        for (key, entry) in fresh_keys.iter().zip(fresh) {
            if self.entries.len() >= CACHE_ENTRIES {
                if let Some(oldest) = self.order.pop_front() {
                    self.entries.remove(&oldest);
                    self.rows.evicted += 1;
                }
            }
            self.entries.insert(*key, entry);
            self.order.push_back(*key);
        }
        let rows = &mut self.rows;
        rows.calls += 1;
        rows.positions += keys.len() as u64;
        rows.hits += hits as u64;
        rows.served += fresh_keys.len() as u64;
        rows.in_batch_repeats += (keys.len() - hits - fresh_keys.len()) as u64;
        rows.all_hit_calls += u64::from(!keys.is_empty() && fresh_keys.is_empty());
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(n: i64) -> LeafKey {
        LeafKey::of(
            &[(n, 0, 1)],
            -1,
            2,
            6,
            8,
            19,
            mantis_graph::EmptyEdges::Kept,
        )
        .expect("a valid request")
    }

    fn entry(v: f32) -> Entry {
        Entry {
            policy: LegalSetPolicy::default(),
            value: v,
            center: (0, 0),
        }
    }

    #[test]
    fn a_call_serves_hits_from_the_cache_and_each_unseen_key_once() {
        let mut cache = GameCache::default();
        let (a, b) = (key(1), key(2));
        let (sources, fresh) = cache.plan(&[a, b, a]);
        assert_eq!(
            (sources, fresh.clone()),
            (
                vec![Source::Fresh(0), Source::Fresh(1), Source::Fresh(0)],
                vec![a, b]
            )
        );
        let out = cache.finish(
            &[a, b, a],
            &[Source::Fresh(0), Source::Fresh(1), Source::Fresh(0)],
            &fresh,
            vec![entry(0.1), entry(0.2)],
        );
        assert_eq!(
            out.iter().map(|e| e.value).collect::<Vec<_>>(),
            vec![0.1, 0.2, 0.1]
        );
        let (sources, fresh) = cache.plan(&[b]);
        assert_eq!((sources.clone(), fresh.len()), (vec![Source::Hit], 0));
        assert_eq!(
            cache.finish(&[b], &sources, &fresh, Vec::new())[0].value,
            0.2
        );
        let rows = cache.rows();
        assert_eq!(
            (
                rows.calls,
                rows.positions,
                rows.hits,
                rows.served,
                rows.in_batch_repeats,
                rows.all_hit_calls
            ),
            (2, 4, 1, 2, 1, 1)
        );
    }

    #[test]
    fn the_oldest_entry_goes_at_capacity_and_a_hit_does_not_refresh_it() {
        let mut cache = GameCache::default();
        for i in 0..CACHE_ENTRIES as i64 {
            let k = key(i + 10);
            let (s, f) = cache.plan(&[k]);
            let _ = cache.finish(&[k], &s, &f, vec![entry(0.0)]);
        }
        let first = key(10);
        let (s, f) = cache.plan(&[first]);
        let _ = cache.finish(&[first], &s, &f, Vec::new()); // a hit on the oldest
        let late = key(-1);
        let (s, f) = cache.plan(&[late]);
        let _ = cache.finish(&[late], &s, &f, vec![entry(0.0)]);
        assert_eq!(cache.rows().evicted, 1);
        assert_eq!(
            cache.plan(&[first]).0,
            vec![Source::Fresh(0)],
            "the oldest went though it was just hit"
        );
        cache.new_game();
        assert_eq!(
            cache.plan(&[late]).0,
            vec![Source::Fresh(0)],
            "a new game reads none of the last one's entries"
        );
    }
}
