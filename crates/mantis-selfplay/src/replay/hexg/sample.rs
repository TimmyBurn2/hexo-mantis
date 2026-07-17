//! HEXG sample path — rebuild-at-sample via the native builder.
//!
//! Weighted-sample record indices, then per sampled record: draw a uniform D6
//! element, coord-rotate the stored stones AND the visit-map keys by it, rebuild
//! via `mantis_graph::build_axis_graph` (which stamps `builder_impl = 1`), and
//! align the rotated visit-keys to the built legal nodes by coord → the
//! per-legal-node policy target.
//!
//! Ported from the predecessor engine's `replay_buffer/hexg/sample.rs` with the
//! FFI + graph-wire strip: the terminal block-diagonal graph-wire fuse
//! (`from_axis_graphs`, whose wire type lives in the predecessor inference-bridge
//! module → WP6) is deferred, so the core returns the buffer-owned
//! `(Vec<AxisGraph>, GraphTargets)` — for the single-graph oracle local == global,
//! so the fuse changes no computed value (R-1). Error type → `Result<_, String>`.

use std::collections::HashSet;

use fxhash::FxHashMap;
use mantis_graph::{build_axis_graph, AxisGraph, BuildParams, StoneList};
use rand::RngExt;

use super::super::sym::{rotate_axial, N_SYMS};
use super::{GraphTargets, HexgBuffer};

/// Compare aligned visit mass against the mass stored at push time; LOUD-fail,
/// naming the record's `game_id`/`ply`, when they diverge beyond tolerance.
/// Pure (no bindings) so it is directly unit-testable.
pub fn mass_drop_check(game_id: i64, ply_idx: u16, stored_mass: f32, aligned_mass: f32) -> Result<(), String> {
    const REL_TOL: f32 = 1e-4;
    const ABS_FLOOR: f32 = 1e-6;
    let dropped = stored_mass - aligned_mass;
    let tripped = if stored_mass.abs() > ABS_FLOOR {
        (dropped.abs() / stored_mass.abs()) > REL_TOL
    } else {
        aligned_mass.abs() > ABS_FLOOR
    };
    if tripped {
        Err(format!(
            "HEXG sample: visit mass dropped at sample-align (illegal/off-window \
             visit coord not in the rebuilt legal-node set) for game_id={game_id} \
             ply={ply_idx}: stored={stored_mass:.6} aligned={aligned_mass:.6} \
             dropped={dropped:.6} (tolerance rel={REL_TOL})"
        ))
    } else {
        Ok(())
    }
}

impl HexgBuffer {
    /// Sample a single index by weighted rejection (32-attempt cap, then
    /// unconditional accept). Identical to `ReplayBuffer::weighted_sample_one`.
    #[inline]
    pub fn weighted_sample_one(&mut self) -> usize {
        const MAX_REJECT: usize = 32;
        for _ in 0..MAX_REJECT {
            let idx = self.rng.random_range(0..self.size);
            let w = half::f16::from_bits(self.weights[idx]).to_f32();
            if w >= 1.0 || self.rng.random::<f32>() < w {
                return idx;
            }
        }
        self.rng.random_range(0..self.size)
    }

    /// Newest-slots window size for the `recent_frac` selection: head-relative
    /// `[head - window, head)` mod capacity, clamped to `size`.
    #[inline]
    #[must_use]
    pub fn recent_window(&self) -> usize {
        self.size.min(usize::max(256, self.capacity / 2))
    }

    /// Draw `n` indices uniformly (with replacement) from the newest-slots window.
    pub fn sample_recent_indices(&mut self, n: usize) -> Vec<usize> {
        let window = self.recent_window();
        debug_assert!(window > 0, "recent_window must be >0 when size>0");
        let start = (self.head + self.capacity - window) % self.capacity;
        (0..n)
            .map(|_| {
                let offset = self.rng.random_range(0..window);
                (start + offset) % self.capacity
            })
            .collect()
    }

    /// Sample `batch_size` slot indices, deduping by `game_id` (untagged -1 slots
    /// skip the guard). `recent_frac == 0.0` is byte-identical to the full-ring
    /// weighted sample; `> 0.0` draws `round(batch_size * recent_frac)` from the
    /// newest slots and the remainder weighted-uniform.
    pub fn sample_indices(&mut self, batch_size: usize, recent_frac: f32) -> Vec<usize> {
        const MAX_RETRIES: usize = 8;
        let mut indices: Vec<usize> = if recent_frac > 0.0 && self.size > 0 {
            let n_recent = ((batch_size as f32) * recent_frac).round() as usize;
            let n_recent = n_recent.min(batch_size);
            let mut idx = self.sample_recent_indices(n_recent);
            idx.extend((n_recent..batch_size).map(|_| self.weighted_sample_one()));
            idx
        } else {
            (0..batch_size).map(|_| self.weighted_sample_one()).collect()
        };
        let mut seen: HashSet<i64> = HashSet::with_capacity(batch_size);
        for _ in 0..MAX_RETRIES {
            seen.clear();
            let mut all_unique = true;
            for idx in &mut indices {
                let gid = self.game_ids[*idx];
                if gid == -1 || seen.insert(gid) {
                    continue;
                }
                all_unique = false;
                let mut candidate = self.weighted_sample_one();
                for _ in 0..16 {
                    let cgid = self.game_ids[candidate];
                    if cgid == -1 || !seen.contains(&cgid) {
                        break;
                    }
                    candidate = self.weighted_sample_one();
                }
                *idx = candidate;
                let cgid = self.game_ids[candidate];
                if cgid != -1 {
                    seen.insert(cgid);
                }
            }
            if all_unique {
                break;
            }
        }
        indices
    }

    /// Rebuild + align `batch_size` sampled records. Returns the buffer-owned
    /// `(Vec<AxisGraph>, GraphTargets)`; the block-diagonal graph-wire fuse
    /// (`from_axis_graphs`) is deferred to WP6 (R-1). See module docs.
    pub fn sample_graph_batch_impl(
        &mut self,
        batch_size: usize,
        augment: bool,
        recent_frac: f32,
    ) -> Result<(Vec<AxisGraph>, GraphTargets), String> {
        if self.size == 0 {
            return Err("Cannot sample from an empty HEXG buffer".to_string());
        }
        let indices = self.sample_indices(batch_size, recent_frac);

        let mut graphs = Vec::with_capacity(batch_size);
        let mut policy_target: Vec<f32> = Vec::new();
        let mut outcomes: Vec<f32> = Vec::with_capacity(batch_size);
        let mut value_valid: Vec<u8> = Vec::with_capacity(batch_size);
        let mut is_full_search: Vec<u8> = Vec::with_capacity(batch_size);
        let mut argmax_q: Vec<i32> = Vec::with_capacity(batch_size);
        let mut argmax_r: Vec<i32> = Vec::with_capacity(batch_size);
        let mut argmax_valid: Vec<u8> = Vec::with_capacity(batch_size);

        let params_base = BuildParams {
            win_length: self.win_length,
            radius: self.radius,
            current_player: 1, // overwritten per record
            moves_remaining: 2,
            trunk_size: self.trunk_size,
        };

        for &idx in &indices {
            let rec = self.record_at(idx);
            let game_id = self.game_ids[idx];
            let ply_idx = rec.ply_index;

            // D6 element for this sample (uniform per-sample). sym 0 = identity
            // when augmentation is OFF. Force identity for a 0-stone record: the
            // empty-board fallback rectangle is not closed under 8 of the 12 D6
            // elements, and an empty stone list carries no orientation to learn.
            let sym = if augment && !rec.stones.is_empty() {
                self.rng.random_range(0..N_SYMS)
            } else {
                0
            };

            // Rotate stones by the element (axial lattice automorphism).
            let mut stones: Vec<(i32, i32, i8)> = Vec::with_capacity(rec.stones.len());
            for &(q, r, p) in &rec.stones {
                let (rq, rr) = rotate_axial(i32::from(q), i32::from(r), sym);
                stones.push((rq, rr, p));
            }
            let params = BuildParams {
                current_player: rec.current_player,
                moves_remaining: rec.moves_remaining,
                ..params_base
            };
            // Rebuild — the builder emits the correctly re-indexed graph and
            // stamps builder_impl=1. edge_index is NEVER cached across aug.
            let g = build_axis_graph(&StoneList { stones }, &params);

            // Rotate the visit-map KEYS by the SAME element, so the policy target
            // follows each cell to its new location.
            let mut vmap: FxHashMap<(i32, i32), f32> = FxHashMap::default();
            let stored_mass: f32 = rec.visits.iter().map(|&(_, _, prob)| prob).sum();
            for &(q, r, prob) in &rec.visits {
                let (rq, rr) = rotate_axial(i32::from(q), i32::from(r), sym);
                vmap.insert((rq, rr), prob);
            }

            // Align to the built legal nodes (gather order == the segment order):
            // target[i] = rotated_visit_map[coord_of_legal_node_i] or 0. No
            // off-window drop — every legal node gets its coord's mass.
            let mut best_prob = f32::NEG_INFINITY;
            let mut best_coord: Option<(i32, i32)> = None;
            let mut aligned_mass = 0.0f32;
            for &row in &g.legal_node_gather {
                let cq = g.node_coords[row as usize * 2];
                let cr = g.node_coords[row as usize * 2 + 1];
                let prob = vmap.get(&(cq, cr)).copied().unwrap_or(0.0);
                aligned_mass += prob;
                policy_target.push(prob);
                if prob > best_prob {
                    best_prob = prob;
                    best_coord = Some((cq, cr));
                }
            }

            // ALWAYS-ON contract check: compare aligned mass against the mass
            // stored at push time and LOUD-fail, naming the record, when they
            // diverge beyond tolerance. `rotate_axial` is exact integer-lattice
            // math, so a legit producer's mass survives the align bit-for-bit.
            mass_drop_check(game_id, ply_idx, stored_mass, aligned_mass)?;

            match best_coord {
                Some((q, r)) if best_prob > 0.0 => {
                    argmax_q.push(q);
                    argmax_r.push(r);
                    argmax_valid.push(1);
                }
                // all-zero target (value-only / quick-search row): no argmax cell.
                _ => {
                    argmax_q.push(0);
                    argmax_r.push(0);
                    argmax_valid.push(0);
                }
            }

            outcomes.push(rec.outcome);
            value_valid.push(u8::from(rec.value_valid));
            is_full_search.push(u8::from(rec.is_full_search));
            graphs.push(g);
        }

        let targets = GraphTargets {
            policy_target,
            outcomes,
            value_valid,
            is_full_search,
            argmax_q,
            argmax_r,
            argmax_valid,
        };
        Ok((graphs, targets))
    }
}
