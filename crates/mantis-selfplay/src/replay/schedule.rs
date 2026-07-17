//! Game-length weight schedule for sampling — the weight half of the
//! `sym_tables.rs` 3-way split, moved NEAR sampling. Ported verbatim from the
//! predecessor engine's `replay_buffer/sym_tables.rs`.

use half::f16;

/// A single threshold bracket: games with length < `max_moves` get `weight`.
/// Brackets are evaluated in order; the first match wins.
#[derive(Clone, Debug)]
pub struct WeightBracket {
    /// Exclusive upper bound (`game_length < max_moves`).
    pub max_moves: u16,
    /// f16-as-u16 bits.
    pub weight: u16,
}

/// Config-driven weight schedule for game-length-based sampling.
/// Default: all positions have weight 1.0 (uniform sampling).
#[derive(Clone, Debug)]
pub struct WeightSchedule {
    pub brackets: Vec<WeightBracket>,
    /// f16 bits for weight when no bracket matches.
    pub default_weight: u16,
}

impl WeightSchedule {
    pub fn uniform() -> Self {
        WeightSchedule {
            brackets: Vec::new(),
            default_weight: f16::from_f32(1.0).to_bits(),
        }
    }

    /// Look up the weight (as f16 bits) for a given game length.
    #[inline]
    #[must_use]
    pub fn weight_for(&self, game_length: u16) -> u16 {
        for b in &self.brackets {
            if game_length < b.max_moves {
                return b.weight;
            }
        }
        self.default_weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// O-31 (partial): weight_for returns the correct bracket.
    #[test]
    fn test_weight_schedule_lookup() {
        let schedule = WeightSchedule {
            brackets: vec![
                WeightBracket { max_moves: 10, weight: f16::from_f32(0.15).to_bits() },
                WeightBracket { max_moves: 25, weight: f16::from_f32(0.50).to_bits() },
            ],
            default_weight: f16::from_f32(1.0).to_bits(),
        };

        let w5 = f16::from_bits(schedule.weight_for(5)).to_f32();
        let w10 = f16::from_bits(schedule.weight_for(10)).to_f32();
        let w15 = f16::from_bits(schedule.weight_for(15)).to_f32();
        let w25 = f16::from_bits(schedule.weight_for(25)).to_f32();
        let w40 = f16::from_bits(schedule.weight_for(40)).to_f32();

        assert!((w5 - 0.15).abs() < 0.01, "game_length=5 → {w5}");
        assert!((w10 - 0.50).abs() < 0.01, "game_length=10 → {w10}");
        assert!((w15 - 0.50).abs() < 0.01, "game_length=15 → {w15}");
        assert!((w25 - 1.0).abs() < 0.01, "game_length=25 → {w25}");
        assert!((w40 - 1.0).abs() < 0.01, "game_length=40 → {w40}");
    }

    /// O-31 (partial): uniform schedule accepts all positions equally.
    #[test]
    fn test_uniform_schedule_all_weight_one() {
        let schedule = WeightSchedule::uniform();
        let w = f16::from_bits(schedule.weight_for(5)).to_f32();
        assert!((w - 1.0).abs() < 0.01);
        let w = f16::from_bits(schedule.weight_for(100)).to_f32();
        assert!((w - 1.0).abs() < 0.01);
    }
}
