//! Storage layout, ring capacity, and weight-schedule configuration for
//! `ReplayBuffer`. Ported verbatim from the predecessor engine's
//! `replay_buffer/storage.rs` with the FFI-binding strip (the error type becomes
//! `Result<_, String>`; error strings preserved).

use half::f16;
use std::sync::atomic::Ordering;

use super::schedule::{WeightBracket, WeightSchedule};
use super::ReplayBuffer;

impl ReplayBuffer {
    /// Return `(size, capacity, weight_histogram)` for dashboard display.
    #[must_use]
    pub fn get_buffer_stats(&self) -> (usize, usize, Vec<u64>) {
        let histogram = vec![
            self.weight_buckets[0].load(Ordering::Relaxed),
            self.weight_buckets[1].load(Ordering::Relaxed),
            self.weight_buckets[2].load(Ordering::Relaxed),
        ];
        (self.size, self.capacity, histogram)
    }

    /// R266/F-P1/N1 — cumulative count of sampled records drawn from the FULL
    /// 12-element D6 group (the record was window-lossless under every
    /// element). See `ReplayBuffer::compact_draws`'s field doc.
    #[must_use]
    pub fn compact_draws(&self) -> u64 {
        self.compact_draws.load(Ordering::Relaxed)
    }

    /// R266/F-P1/N1 — cumulative count of sampled records restricted to
    /// `sym::WINDOW_PRESERVING_SYMS`. See `ReplayBuffer::spread_draws`'s field
    /// doc.
    #[must_use]
    pub fn spread_draws(&self) -> u64 {
        self.spread_draws.load(Ordering::Relaxed)
    }

    /// Return a fresh position ID and advance the internal counter.
    pub fn next_game_id(&mut self) -> i64 {
        let id = self.next_game_id;
        self.next_game_id += 1;
        id
    }

    /// Grow the buffer to `new_capacity` positions, preserving all existing data.
    /// The ring buffer is linearised in-place (oldest entry → slot 0) before
    /// extending. Errors if `new_capacity <= self.capacity`.
    pub fn resize(&mut self, new_capacity: usize) -> Result<(), String> {
        if new_capacity <= self.capacity {
            return Err(format!(
                "resize: new_capacity ({}) must be greater than current capacity ({})",
                new_capacity, self.capacity,
            ));
        }

        let state_stride = self.encoding.state_stride();
        let chain_stride = self.encoding.chain_stride();
        let policy_stride = self.encoding.policy_stride();
        let aux_stride = self.encoding.aux_stride();

        // Linearise the ring buffer when it has wrapped around.
        if self.size == self.capacity && self.head != 0 {
            self.states[..self.capacity * state_stride].rotate_left(self.head * state_stride);
            self.chain_planes[..self.capacity * chain_stride].rotate_left(self.head * chain_stride);
            self.policies[..self.capacity * policy_stride].rotate_left(self.head * policy_stride);
            self.outcomes[..self.capacity].rotate_left(self.head);
            self.game_ids[..self.capacity].rotate_left(self.head);
            self.weights[..self.capacity].rotate_left(self.head);
            self.ownership[..self.capacity * aux_stride].rotate_left(self.head * aux_stride);
            self.winning_line[..self.capacity * aux_stride].rotate_left(self.head * aux_stride);
            self.is_full_search[..self.capacity].rotate_left(self.head);
            self.value_target_valid[..self.capacity].rotate_left(self.head);
            self.position_indices[..self.capacity].rotate_left(self.head);
            // R245(c): the losslessness flag is a per-slot column and MUST ride
            // the same rotation as the row it describes — otherwise linearising
            // would pair every flag with a different record.
            self.compact[..self.capacity].rotate_left(self.head);
        }

        // Extend storage to new capacity.
        let default_w = f16::from_f32(1.0).to_bits();
        self.states.resize(new_capacity * state_stride, 0u16);
        self.chain_planes.resize(new_capacity * chain_stride, 0u16);
        self.policies.resize(new_capacity * policy_stride, 0.0f32);
        self.outcomes.resize(new_capacity, 0.0f32);
        self.game_ids.resize(new_capacity, -1i64);
        self.weights.resize(new_capacity, default_w);
        self.ownership.resize(new_capacity * aux_stride, 1u8); // 1 = empty
        self.winning_line.resize(new_capacity * aux_stride, 0u8);
        self.is_full_search.resize(new_capacity, 1u8); // 1 = full-search default
        self.value_target_valid.resize(new_capacity, 1u8); // 1 = supervise value default
        self.position_indices.resize(new_capacity, 0u16);
        self.compact.resize(new_capacity, 0u8); // 0 = spread/uncertified (R245(c))

        self.head = self.size;
        self.capacity = new_capacity;
        Ok(())
    }

    /// Count valid outcomes in the half-open interval `[lo, hi)`. Reads only the
    /// live prefix (`self.size` slots).
    #[must_use]
    pub fn outcome_in_range_count(&self, lo: f32, hi: f32) -> usize {
        self.outcomes[..self.size]
            .iter()
            .filter(|&&v| v >= lo && v < hi)
            .count()
    }

    /// Set the game-length weight schedule.
    ///
    /// Args:
    ///     thresholds: exclusive upper bounds (sorted ascending)
    ///     weights: f32 weights, same length as thresholds
    ///     default_weight: weight for games >= all thresholds (typically 1.0)
    pub fn set_weight_schedule(
        &mut self,
        thresholds: Vec<u16>,
        weights: Vec<f32>,
        default_weight: f32,
    ) -> Result<(), String> {
        if thresholds.len() != weights.len() {
            return Err("thresholds and weights must have the same length".to_string());
        }
        let brackets: Vec<WeightBracket> = thresholds
            .iter()
            .zip(weights.iter())
            .map(|(&t, &w)| WeightBracket {
                max_moves: t,
                weight: f16::from_f32(w).to_bits(),
            })
            .collect();
        self.weight_schedule = WeightSchedule {
            brackets,
            default_weight: f16::from_f32(default_weight).to_bits(),
        };
        Ok(())
    }
}
