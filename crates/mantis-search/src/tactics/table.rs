//! The solver's table: fixed size, always-replace, the 128-bit key stored whole, cleared by a generation bump.
//!
//! A completeness cache only: a win is concluded only after every covering reply is refuted.

#[derive(Clone, Copy, Default)]
struct Entry {
    key: u128,
    a: u32,
    b: u32,
    generation: u32,
    proven: u8,
    searched: u8,
}

/// A stored attacker node: its turn (`a`, `b` packed cells), turns to win (0 unproven), turns searched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Stored {
    pub(crate) a: u32,
    pub(crate) b: u32,
    pub(crate) proven: u8,
    pub(crate) searched: u8,
}

pub(crate) struct Table {
    entries: Vec<Entry>,
    mask: usize,
    generation: u32,
}

impl Table {
    /// A table of `entries` slots rounded down to a power of two (at least one).
    pub(crate) fn new(entries: usize) -> Self {
        let size = if entries <= 1 {
            1
        } else {
            1usize << (usize::BITS - 1 - entries.leading_zeros())
        };
        Table {
            entries: vec![Entry::default(); size],
            mask: size - 1,
            generation: 1,
        }
    }

    /// Forget every entry in O(1): a stale generation reads as empty.
    pub(crate) fn clear(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.entries.fill(Entry::default());
            self.generation = 1;
        }
    }

    pub(crate) fn probe(&self, key: u128) -> Option<Stored> {
        let e = &self.entries[key as usize & self.mask];
        (e.generation == self.generation && e.key == key).then_some(Stored {
            a: e.a,
            b: e.b,
            proven: e.proven,
            searched: e.searched,
        })
    }

    pub(crate) fn store(&mut self, key: u128, stored: Stored) {
        self.entries[key as usize & self.mask] = Entry {
            key,
            a: stored.a,
            b: stored.b,
            generation: self.generation,
            proven: stored.proven,
            searched: stored.searched,
        };
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cleared_table_forgets_and_a_stored_key_reads_back_whole() {
        let mut t = Table::new(1000);
        assert_eq!(t.len(), 512, "rounded down to a power of two");
        let s = Stored {
            a: 7,
            b: 9,
            proven: 3,
            searched: 4,
        };
        let key = (1u128 << 100) | 5;
        t.store(key, s);
        assert_eq!(t.probe(key), Some(s));
        assert_eq!(
            t.probe(5),
            None,
            "same slot, another key: the whole key is compared"
        );
        t.clear();
        assert_eq!(t.probe(key), None);
    }

    #[test]
    fn the_generation_wraps_to_a_fresh_table() {
        let mut t = Table::new(4);
        t.generation = u32::MAX;
        let key = 3u128;
        t.store(
            key,
            Stored {
                a: 1,
                b: 2,
                proven: 1,
                searched: 1,
            },
        );
        t.clear();
        assert_eq!(t.generation, 1);
        assert_eq!(t.probe(key), None);
    }
}
