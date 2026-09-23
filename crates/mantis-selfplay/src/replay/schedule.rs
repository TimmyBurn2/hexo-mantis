//! The per-slot replay weight: uniform, since no path sets a game-length schedule.

use half::f16;

/// The per-slot sampling weight the HEXG v2 record stores: uniform 1.0.
#[derive(Clone, Debug)]
pub struct WeightSchedule {
    /// f16 bits of every slot's weight.
    pub default_weight: u16,
}

impl WeightSchedule {
    pub fn uniform() -> Self {
        WeightSchedule {
            default_weight: f16::from_f32(1.0).to_bits(),
        }
    }

    /// The weight (as f16 bits) of a pushed slot.
    #[inline]
    #[must_use]
    pub fn weight_for(&self) -> u16 {
        self.default_weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// O-31 (partial): uniform schedule accepts all positions equally.
    #[test]
    fn test_uniform_schedule_all_weight_one() {
        let schedule = WeightSchedule::uniform();
        let w = f16::from_bits(schedule.weight_for()).to_f32();
        assert!((w - 1.0).abs() < 0.01);
    }
}
