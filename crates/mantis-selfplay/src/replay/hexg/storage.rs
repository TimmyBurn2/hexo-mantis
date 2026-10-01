//! HEXG ring storage mechanics — dashboard stats on the graph SoA strides.

use std::sync::atomic::Ordering;

use super::HexgBuffer;

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
}
