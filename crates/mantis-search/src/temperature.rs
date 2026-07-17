//! Quarter-cosine self-play temperature schedule (pure std math).
//!
//! Ported from the self-play worker loop's inline temperature helper. It carries
//! NONE of the loop's coupling — the symmetry/rotation helpers stay in the
//! self-play layer; only this pure `f32` math lives here so the cross-language
//! CSV golden (`tests/temperature_parity_golden.rs`) can pin it inside this crate.

/// Quarter-cosine temperature schedule used by the self-play worker loop.
///
/// Returns 1.0 at compound_move=0, decays via cos(π/2·progress) toward
/// temp_min, then clamps at temp_min for compound_move ≥ temp_threshold.
///
/// # Arguments
/// * `compound_move`  — zero-indexed compound move number in the current game
/// * `temp_threshold` — compound move at which the floor kicks in. Config-driven;
///   default `0` = schedule OFF (returns a constant `temp_min` at every move).
/// * `temp_min`       — minimum temperature floor. Config-driven; default `0.5`.
pub fn compute_move_temperature(
    compound_move: usize,
    temp_threshold: usize,
    temp_min: f32,
) -> f32 {
    if compound_move < temp_threshold {
        let progress = compound_move as f32 / temp_threshold as f32;
        f32::max(temp_min, (std::f32::consts::FRAC_PI_2 * progress).cos())
    } else {
        temp_min
    }
}

/// The ply→compound-move clock the temperature golden pins as contract:
/// `cm = 0 if ply == 0 else div_ceil(ply, 2)` (the ply-0 special case + the
/// ply1/ply2 same-cm boundary). Kept pure (takes `usize`) so callers pass
/// `board.ply.index() as usize`.
pub fn ply_to_compound_move(ply: usize) -> usize {
    if ply == 0 {
        0
    } else {
        ply.div_ceil(2)
    }
}
