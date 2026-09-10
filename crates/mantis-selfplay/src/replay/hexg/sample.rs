// Exceeds the 300-line soft cap (R8): the sample path is ONE unit — the weighted index draw, the
// RNG hoist, the parallel rebuild that consumes it and the per-record align. The hoist and the
// rebuild are a determinism pair: split apart, a reader can change the draw order in one file
// and the parallel reassembly in another and see neither break the other.
//! HEXG sample path — rebuild-at-sample via the native builder.
//!
//! Weighted-sample record indices, then per sampled record: draw a uniform D6 element,
//! coord-rotate the stored stones AND the visit-map keys by it, rebuild via
//! `mantis_graph::build_axis_graph`, and align the rotated visit-keys to the built legal nodes
//! by coord → the per-legal-node policy target. The terminal graph-wire fuse is deferred.

use std::collections::HashSet;

use fxhash::FxHashMap;
use mantis_graph::{build_axis_graph, AxisGraph, BuildParams, StoneList};
use rand::RngExt;

use super::super::sym::{rotate_axial, N_SYMS};
use super::{GraphRecord, GraphTargets, HexgBuffer};

/// Compare aligned visit mass against the mass stored at push time; LOUD-fail, naming the
/// record, when they diverge beyond tolerance. Pure, so it is unit-testable.
pub fn mass_drop_check(
    game_id: i64,
    ply_idx: u16,
    stored_mass: f32,
    aligned_mass: f32,
) -> Result<(), String> {
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
    /// Sample one index by weighted rejection, 32-attempt cap then unconditional accept.
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

    /// Newest-slots window for `recent_frac`: `[head - window, head)` mod capacity, clamped.
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

    /// Remember WHAT this batch was made of, for the trainer's per-batch line: rows per game
    /// says whether the dedupe guard is doing anything, and age in rows back from the newest is
    /// what says a ring has stopped being fed. Stored rather than returned, so the hot sample
    /// path keeps its signature.
    fn record_batch_composition(&mut self, indices: &[usize]) {
        let mut per_game: FxHashMap<i64, u32> = FxHashMap::default();
        let mut ages: Vec<u32> = Vec::with_capacity(indices.len());
        for &idx in indices {
            let gid = self.game_ids[idx];
            if gid != -1 {
                *per_game.entry(gid).or_insert(0) += 1;
            }
            // `head` points one past the newest, so the newest slot is `head - 1` mod capacity.
            let newest = (self.head + self.capacity - 1) % self.capacity;
            ages.push(((newest + self.capacity - idx) % self.capacity) as u32);
        }
        ages.sort_unstable();
        self.last_batch_distinct_games = per_game.len() as u32;
        self.last_batch_max_rows_per_game = per_game.values().copied().max().unwrap_or(0);
        self.last_batch_untagged_rows =
            indices.iter().filter(|&&i| self.game_ids[i] == -1).count() as u32;
        self.last_batch_age_quantiles = if ages.is_empty() {
            [0, 0, 0]
        } else {
            let at = |q: f64| ages[(((ages.len() - 1) as f64) * q).round() as usize];
            [at(0.5), at(0.9), at(0.99)]
        };
    }

    /// Sample `batch_size` slot indices, deduping by `game_id` (untagged -1 slots skip the
    /// guard). `recent_frac == 0.0` is byte-identical to the full-ring weighted sample.
    pub fn sample_indices(&mut self, batch_size: usize, recent_frac: f32) -> Vec<usize> {
        const MAX_RETRIES: usize = 8;
        let mut indices: Vec<usize> = if recent_frac > 0.0 && self.size > 0 {
            let n_recent = ((batch_size as f32) * recent_frac).round() as usize;
            let n_recent = n_recent.min(batch_size);
            let mut idx = self.sample_recent_indices(n_recent);
            idx.extend((n_recent..batch_size).map(|_| self.weighted_sample_one()));
            idx
        } else {
            (0..batch_size)
                .map(|_| self.weighted_sample_one())
                .collect()
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

    /// The per-sample D6 draws for one batch, in INDEX ORDER — the RNG hoist.
    ///
    /// The parallel rebuild cannot share `&mut self.rng`, and hoisting the draw consumes the
    /// generator in exactly the order and count the serial loop did, which is what makes the
    /// parallel path BIT-IDENTICAL. `n_stones[idx] != 0` stands in for `!rec.stones.is_empty()`
    /// so no record is materialised first, an equivalence pinned by the parity test.
    pub fn draw_syms(&mut self, indices: &[usize], augment: bool) -> Vec<usize> {
        indices
            .iter()
            .map(|&idx| {
                if augment && self.n_stones[idx] != 0 {
                    self.rng.random_range(0..N_SYMS)
                } else {
                    0
                }
            })
            .collect()
    }

    pub fn sample_graph_batch_impl(
        &mut self,
        batch_size: usize,
        augment: bool,
        recent_frac: f32,
        n_threads: usize,
    ) -> Result<(Vec<AxisGraph>, GraphTargets), String> {
        if self.size == 0 {
            return Err("Cannot sample from an empty HEXG buffer".to_string());
        }
        let indices = self.sample_indices(batch_size, recent_frac);
        self.record_batch_composition(&indices);

        let params_base = BuildParams {
            win_length: self.win_length,
            radius: self.radius,
            current_player: 1, // overwritten per record
            moves_remaining: 2,
            trunk_size: self.trunk_size,
        };

        let syms = self.draw_syms(&indices, augment);

        // Materialise the records BEFORE the parallel section: `record_at` borrows `&self`,
        // and the section must not hold a borrow of the buffer while threads run.
        let items: Vec<(GraphRecord, i64, usize)> = indices
            .iter()
            .zip(&syms)
            .map(|(&idx, &sym)| (self.record_at(idx), self.game_ids[idx], sym))
            .collect();

        let per_item = build_and_align_batch(&items, &params_base, n_threads)?;

        let mut graphs = Vec::with_capacity(batch_size);
        let mut policy_target: Vec<f32> = Vec::new();
        let mut explicit_mask: Vec<u8> = Vec::new();
        let mut tail_mass: Vec<f32> = Vec::with_capacity(batch_size);
        let mut outcomes: Vec<f32> = Vec::with_capacity(batch_size);
        let mut value_valid: Vec<u8> = Vec::with_capacity(batch_size);
        let mut is_full_search: Vec<u8> = Vec::with_capacity(batch_size);
        let mut argmax_q: Vec<i32> = Vec::with_capacity(batch_size);
        let mut argmax_r: Vec<i32> = Vec::with_capacity(batch_size);
        let mut argmax_valid: Vec<u8> = Vec::with_capacity(batch_size);

        // Reassembly is IN INDEX ORDER, not completion order: `policy_target` is one flat
        // concatenation whose segment boundaries the collate derives from the legal counts.
        for out in per_item {
            policy_target.extend_from_slice(&out.policy_target);
            explicit_mask.extend_from_slice(&out.explicit_mask);
            tail_mass.push(out.tail_mass);
            argmax_q.push(out.argmax_q);
            argmax_r.push(out.argmax_r);
            argmax_valid.push(out.argmax_valid);
            outcomes.push(out.outcome);
            value_valid.push(out.value_valid);
            is_full_search.push(out.is_full_search);
            graphs.push(out.graph);
        }

        Ok((
            graphs,
            GraphTargets {
                policy_target,
                explicit_mask,
                tail_mass,
                outcomes,
                value_valid,
                is_full_search,
                argmax_q,
                argmax_r,
                argmax_valid,
            },
        ))
    }
}

/// One sampled record's rebuilt graph and its aligned per-legal-node targets.
struct SampleOut {
    graph: AxisGraph,
    policy_target: Vec<f32>,
    explicit_mask: Vec<u8>,
    tail_mass: f32,
    argmax_q: i32,
    argmax_r: i32,
    argmax_valid: u8,
    outcome: f32,
    value_valid: u8,
    is_full_search: u8,
}

/// Rebuild + align every sampled record, across at most `n_threads` OS threads, returning the
/// results IN INDEX ORDER. Measured on the run5 shape, `sample_ring` splits 1 221 ms of
/// `build_axis_graph` against 163 ms of fuse and 2 ms of align — a serial loop over an
/// embarrassingly parallel rebuild whose items touch only their own record. `std::thread::scope`
/// with a chunked split rather than a work-stealing pool, because rayon is absent and adding it
/// is a pins event; `n_threads <= 1` runs the serial path here, the exact-parity control.
fn build_and_align_batch(
    items: &[(GraphRecord, i64, usize)],
    params_base: &BuildParams,
    n_threads: usize,
) -> Result<Vec<SampleOut>, String> {
    if items.is_empty() {
        return Ok(Vec::new());
    }
    let threads = n_threads.max(1).min(items.len());
    if threads == 1 {
        return items
            .iter()
            .map(|it| build_and_align_one(it, params_base))
            .collect();
    }
    let chunk = items.len().div_ceil(threads);
    let mut per_chunk: Vec<Result<Vec<SampleOut>, String>> = Vec::new();
    std::thread::scope(|scope| {
        let handles: Vec<_> = items
            .chunks(chunk)
            .map(|slice| {
                scope.spawn(move || {
                    slice
                        .iter()
                        .map(|it| build_and_align_one(it, params_base))
                        .collect()
                })
            })
            .collect();
        for h in handles {
            // A panicking worker becomes the NAMED error the caller already handles, never a
            // panic that would cross the FFI.
            per_chunk.push(h.join().unwrap_or_else(|_| {
                Err("HEXG sample: a rebuild worker thread panicked".to_string())
            }));
        }
    });
    let mut out = Vec::with_capacity(items.len());
    for chunk_result in per_chunk {
        out.extend(chunk_result?);
    }
    Ok(out)
}

/// The per-record body: rotate, rebuild, align the visit map, check the mass.
fn build_and_align_one(
    item: &(GraphRecord, i64, usize),
    params_base: &BuildParams,
) -> Result<SampleOut, String> {
    let (rec, game_id, sym) = (&item.0, item.1, item.2);
    let ply_idx = rec.ply_index;
    let mut policy_target: Vec<f32> = Vec::new();
    let mut explicit_mask: Vec<u8> = Vec::new();

    // Rotate stones by the element (axial lattice automorphism).
    let mut stones: Vec<(i32, i32, i8)> = Vec::with_capacity(rec.stones.len());
    for &(q, r, p) in &rec.stones {
        let (rq, rr) = rotate_axial(i32::from(q), i32::from(r), sym);
        stones.push((rq, rr, p));
    }
    let params = BuildParams {
        current_player: rec.current_player,
        moves_remaining: rec.moves_remaining,
        ..*params_base
    };
    // The builder re-indexes and stamps builder_impl=1; edge_index is NEVER cached across aug.
    let g = build_axis_graph(&StoneList { stones }, &params);

    // Rotate the visit-map KEYS by the SAME element, so the target follows each cell.
    let mut vmap: FxHashMap<(i32, i32), f32> = FxHashMap::default();
    let stored_mass: f32 = rec.visits.iter().map(|&(_, _, prob)| prob).sum();
    for &(q, r, prob) in &rec.visits {
        let (rq, rr) = rotate_axial(i32::from(q), i32::from(r), sym);
        vmap.insert((rq, rr), prob);
    }

    // Align to the built legal nodes (gather order == segment order); no off-window drop.
    let mut best_prob = f32::NEG_INFINITY;
    let mut best_coord: Option<(i32, i32)> = None;
    let mut aligned_mass = 0.0f32;
    for &row in &g.legal_node_gather {
        let cq = g.node_coords[row as usize * 2];
        let cr = g.node_coords[row as usize * 2 + 1];
        // The mask is membership in the STORED map, not `prob > 0`: an explicit entry that
        // underflowed would otherwise be reclassified as tail and get the prior's shape.
        let stored = vmap.get(&(cq, cr)).copied();
        let prob = stored.unwrap_or(0.0);
        aligned_mass += prob;
        policy_target.push(prob);
        explicit_mask.push(u8::from(stored.is_some()));
        if prob > best_prob {
            best_prob = prob;
            best_coord = Some((cq, cr));
        }
    }

    // ALWAYS-ON contract check: `rotate_axial` is exact integer math, so a legitimate
    // producer's mass survives the align bit-for-bit.
    mass_drop_check(game_id, ply_idx, stored_mass, aligned_mass)?;

    let (argmax_q, argmax_r, argmax_valid) = match best_coord {
        Some((q, r)) if best_prob > 0.0 => (q, r, 1u8),
        // all-zero target (value-only / quick-search row): no argmax cell.
        _ => (0, 0, 0u8),
    };

    Ok(SampleOut {
        graph: g,
        policy_target,
        explicit_mask,
        tail_mass: rec.tail_mass,
        argmax_q,
        argmax_r,
        argmax_valid,
        outcome: rec.outcome,
        value_valid: u8::from(rec.value_valid),
        is_full_search: u8::from(rec.is_full_search),
    })
}
