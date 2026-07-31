//! Ragged legal-set policy — the ls-policy math MCTS consumes.
//!
//! `LegalSetPolicy` + `is_covered` are the only two items the MCTS reads out of
//! the record family; they cross into this crate so the search never reaches
//! back into the selfplay record builders (which stay with the dense/ls-agg
//! producers). No sorted-Vec representation, no incremental-delta coverage:
//! coverage is rebuilt from a `FxHashSet` legal set with inline dedup.

use fxhash::FxHashMap;
use mantis_core::board::Board;

/// Ragged legal-set policy: the in-global-window slots in `dense` (keyed by
/// `window_flat_idx`, fast array path), plus off-global-window cells COVERED by
/// some cluster in `overflow` (keyed by board coord). Re-projected per-cluster
/// into dense rows by the (selfplay-resident) ls-aggregation producers.
#[derive(Clone, Debug, Default)]
pub struct LegalSetPolicy {
    pub dense: Vec<f32>,
    pub overflow: FxHashMap<(i32, i32), f32>,
}

impl LegalSetPolicy {
    /// Read the prior/target mass for board coord `(q, r)`. In-global-window
    /// cells read `dense` (identical to the dense path); any off-window cell
    /// present in `overflow` reads its entry — on the EXPORT path (WP12-R
    /// Phase T no-drop law) uncovered off-window cells carry overflow entries
    /// too, so coverage no longer bounds the overflow key set; an off-window
    /// cell ABSENT from `overflow` reads `floor` (the no-coverage prior, a
    /// prior-producer convention). `(bcq, bcr)` is the global window centre,
    /// `trunk_sz`/`half` the spec-derived geometry.
    #[inline]
    #[allow(clippy::too_many_arguments)] // VERBATIM signature (scalar coord + geometry + floor)
    pub fn get(&self, q: i32, r: i32, bcq: i32, bcr: i32, trunk_sz: i32, half: i32, floor: f32) -> f32 {
        let flat = Board::window_flat_idx_at_geom(q, r, bcq, bcr, trunk_sz, half);
        if flat < self.dense.len() {
            self.dense[flat]
        } else {
            self.overflow.get(&(q, r)).copied().unwrap_or(floor)
        }
    }
}

/// Coverage predicate: is `(q, r)` inside >=1 cluster window? Byte-identical to
/// the aggregation bound test (same `wq = q - cq + half in [0, trunk_sz)`). The
/// PRIOR producers (`aggregate_policy_ls` — a grid-ls prior structurally has no
/// mass outside cluster windows) use this to scope their ragged set, so no
/// uncovered key leaks into a PRIOR's `overflow`; the EXPORT path (WP12-R
/// Phase T) is coverage-free — an exported target's `overflow` may carry
/// uncovered keys. The O1/solver injection gates also read it. Widened to `pub`
/// so the selfplay record producers can import it alongside `LegalSetPolicy`.
#[inline]
pub fn is_covered(q: i32, r: i32, centers: &[(i32, i32)], trunk_sz: i32, half: i32) -> bool {
    centers.iter().any(|&(cq, cr)| {
        let wq = q - cq + half;
        let wr = r - cr + half;
        wq >= 0 && wq < trunk_sz && wr >= 0 && wr < trunk_sz
    })
}
