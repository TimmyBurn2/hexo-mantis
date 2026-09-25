//! Ragged legal-set policy — the ls-policy math MCTS consumes.
//!
//! `LegalSetPolicy` is the one item the MCTS reads out of the record family; it
//! crosses into this crate so the search never reaches back into the selfplay
//! record builders.

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
    /// Read the prior/target mass for board coord `(q, r)`. In-global-window cells read `dense`;
    /// an off-window cell present in `overflow` reads its entry (on the export path uncovered
    /// off-window cells carry entries too, so coverage does not bound the key set); an absent
    /// off-window cell reads `floor`, the no-coverage prior. `(bcq, bcr)` is the global window
    /// centre, `trunk_sz`/`half` the spec-derived geometry.
    #[inline]
    #[allow(clippy::too_many_arguments)] // VERBATIM signature (scalar coord + geometry + floor)
    pub fn get(
        &self,
        q: i32,
        r: i32,
        bcq: i32,
        bcr: i32,
        trunk_sz: i32,
        half: i32,
        floor: f32,
    ) -> f32 {
        let flat = Board::window_flat_idx_at_geom(q, r, bcq, bcr, trunk_sz, half);
        if flat < self.dense.len() {
            self.dense[flat]
        } else {
            self.overflow.get(&(q, r)).copied().unwrap_or(floor)
        }
    }
}
