//! Ply / Turn vocabulary newtypes (measurement-unit discipline: one Ply = one
//! stone placed; one Turn = one compound move). Never infer turn phase from
//! ply parity — the turn hand-off is driven by `Board::moves_remaining`.

/// Half-move counter: one Ply = one stone placed. Ply 0 is the opening single.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Ply(u32);

/// Compound-turn counter: Turn 0 = the 1-stone opening turn; every later turn
/// places 2 stones (turn != ply; never infer phase from ply parity).
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Turn(u32);

impl Ply {
    pub const ZERO: Ply = Ply(0);

    /// Explicit constructor — no `From<u32>` by design (vocabulary type).
    #[must_use]
    pub const fn new(index: u32) -> Ply {
        Ply(index)
    }

    /// Explicit accessor — no `Into<u32>` / `Deref` by design.
    #[must_use]
    pub const fn index(self) -> u32 {
        self.0
    }

    /// The successor ply (+1); used by `apply_move`.
    #[must_use]
    pub const fn next(self) -> Ply {
        Ply(self.0 + 1)
    }

    /// The compound turn this ply counter maps to: ply 0 -> turn 0;
    /// plies 2k-1, 2k -> turn k, i.e. `Turn((index + 1) / 2)`
    /// (written as `div_ceil(2)`, which is identical for u32 and cannot
    /// overflow at `u32::MAX`).
    #[must_use]
    pub const fn turn(self) -> Turn {
        Turn(self.0.div_ceil(2))
    }
}

impl Turn {
    /// Explicit constructor — no `From<u32>` by design.
    #[must_use]
    pub const fn new(index: u32) -> Turn {
        Turn(index)
    }

    /// Explicit accessor — no `Into<u32>` / `Deref` by design.
    #[must_use]
    pub const fn index(self) -> u32 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;

    /// The Ply->Turn mapping is derived from the ENGINE, not prose: walking a
    /// real game, the turn index must increment exactly on the applies where
    /// `apply_move` flips `current_player` (the moves_remaining-driven
    /// hand-off), never on ply parity.
    #[test]
    fn turn_mapping_derived_from_engine() {
        let mut b = Board::new();
        assert_eq!(b.ply.turn(), Turn::new(0), "empty board is in the opening turn");
        let mut expected_turn = 0u32;
        let mut seed = 0x00c0_ffee_1234_5678u64;
        for _ in 0..41 {
            let legal = b.legal_moves();
            assert!(!legal.is_empty());
            // splitmix64 draw
            seed = seed.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^= z >> 31;
            let (q, r) = legal[(z as usize) % legal.len()];
            let before = b.current_player;
            b.apply_move(q, r).unwrap();
            if b.current_player != before {
                expected_turn += 1;
            }
            assert_eq!(
                b.ply.turn().index(),
                expected_turn,
                "turn index must increment exactly when apply_move flips current_player (ply {})",
                b.ply.index()
            );
        }
    }

    #[test]
    fn ply_turn_round_trips_and_next() {
        assert_eq!(Ply::ZERO, Ply::new(0));
        assert_eq!(Ply::new(7).index(), 7);
        assert_eq!(Ply::new(7).next(), Ply::new(8));
        assert_eq!(Turn::new(3).index(), 3);
        // Pinned mapping values: ply 0 -> turn 0; plies 2k-1, 2k -> turn k.
        assert_eq!(Ply::new(0).turn(), Turn::new(0));
        assert_eq!(Ply::new(1).turn(), Turn::new(1));
        assert_eq!(Ply::new(2).turn(), Turn::new(1));
        assert_eq!(Ply::new(3).turn(), Turn::new(2));
        assert_eq!(Ply::new(4).turn(), Turn::new(2));
        assert_eq!(Ply::new(5).turn(), Turn::new(3));
    }

    #[test]
    fn repr_transparent_size_pin() {
        assert_eq!(std::mem::size_of::<Ply>(), std::mem::size_of::<u32>());
        assert_eq!(std::mem::size_of::<Turn>(), std::mem::size_of::<u32>());
    }
}
