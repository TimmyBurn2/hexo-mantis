//! Ply vocabulary newtype (measurement-unit discipline: one Ply = one stone placed, never a
//! compound turn). Never infer turn phase from ply parity — the turn hand-off is driven by
//! `Board::moves_remaining`.

/// Half-move counter: one Ply = one stone placed. Ply 0 is the opening single.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Ply(u32);

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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ply_round_trips_and_next() {
        assert_eq!(Ply::ZERO, Ply::new(0));
        assert_eq!(Ply::new(7).index(), 7);
        assert_eq!(Ply::new(7).next(), Ply::new(8));
    }

    #[test]
    fn repr_transparent_size_pin() {
        assert_eq!(std::mem::size_of::<Ply>(), std::mem::size_of::<u32>());
    }
}
