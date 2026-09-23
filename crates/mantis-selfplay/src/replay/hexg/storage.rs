//! HEXG ring storage mechanics — resize (linearise+extend) and dashboard stats on the graph SoA
//! strides; the error type is `Result<_, String>`.

use std::sync::atomic::Ordering;

use half::f16;

use super::{HexgBuffer, MAX_STONES};

impl HexgBuffer {
    /// `(size, capacity, weight_histogram)` for dashboard display.
    #[must_use]
    pub fn get_buffer_stats_impl(&self) -> (usize, usize, Vec<u64>) {
        let histogram = vec![
            self.weight_buckets[0].load(Ordering::Relaxed),
            self.weight_buckets[1].load(Ordering::Relaxed),
            self.weight_buckets[2].load(Ordering::Relaxed),
        ];
        (self.size, self.capacity, histogram)
    }

    /// Grow the ring to `new_capacity`, preserving all records (linearise + extend).
    pub fn resize_impl(&mut self, new_capacity: usize) -> Result<(), String> {
        if new_capacity <= self.capacity {
            return Err(format!(
                "resize: new_capacity ({}) must be greater than current capacity ({})",
                new_capacity, self.capacity,
            ));
        }

        let sstride = MAX_STONES * 2; // stones_qr per-record stride
        let vcap = self.visit_capacity;
        let vstride = vcap * 2; // visit_qr per-record stride

        if self.size == self.capacity && self.head != 0 {
            self.stones_qr[..self.capacity * sstride].rotate_left(self.head * sstride);
            self.stone_players[..self.capacity * MAX_STONES].rotate_left(self.head * MAX_STONES);
            self.n_stones[..self.capacity].rotate_left(self.head);
            self.visit_qr[..self.capacity * vstride].rotate_left(self.head * vstride);
            self.visit_probs[..self.capacity * vcap].rotate_left(self.head * vcap);
            self.n_visits[..self.capacity].rotate_left(self.head);
            self.tail_mass[..self.capacity].rotate_left(self.head);
            self.current_player[..self.capacity].rotate_left(self.head);
            self.moves_remaining[..self.capacity].rotate_left(self.head);
            self.ply_index[..self.capacity].rotate_left(self.head);
            self.is_full_search[..self.capacity].rotate_left(self.head);
            self.outcomes[..self.capacity].rotate_left(self.head);
            self.value_valid[..self.capacity].rotate_left(self.head);
            self.game_length[..self.capacity].rotate_left(self.head);
            self.game_ids[..self.capacity].rotate_left(self.head);
            self.weights[..self.capacity].rotate_left(self.head);
        }

        let default_w = f16::from_f32(1.0).to_bits();
        self.stones_qr.resize(new_capacity * sstride, 0i16);
        self.stone_players.resize(new_capacity * MAX_STONES, 0i8);
        self.n_stones.resize(new_capacity, 0u16);
        self.visit_qr.resize(new_capacity * vstride, 0i16);
        self.visit_probs.resize(new_capacity * vcap, 0.0f32);
        self.n_visits.resize(new_capacity, 0u16);
        self.tail_mass.resize(new_capacity, 0.0f32);
        self.current_player.resize(new_capacity, 1i8);
        self.moves_remaining.resize(new_capacity, 2u8);
        self.ply_index.resize(new_capacity, 0u16);
        self.is_full_search.resize(new_capacity, 1u8);
        self.outcomes.resize(new_capacity, 0.0f32);
        self.value_valid.resize(new_capacity, 1u8);
        self.game_length.resize(new_capacity, 0u16);
        self.game_ids.resize(new_capacity, -1i64);
        self.weights.resize(new_capacity, default_w);

        self.head = self.size;
        self.capacity = new_capacity;
        Ok(())
    }
}
