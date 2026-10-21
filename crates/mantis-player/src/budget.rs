//! The per-stone budget set on the device: the turn's sims from its clock and the device's measured costs, ρ = 1.

use mantis_search::MAX_ARMED_SIMS;

/// The share of the turn's clock the search may spend.
pub const MARGIN: f64 = 0.8;
/// The budget rule's floor, and the floor of the clocks an arena may offer a device.
pub const FLOOR: u32 = 32;
/// The CPU path's floor under a clock below `min_turn_ms`: a weak turn beats a forfeit.
pub const CPU_FLOOR: u32 = 16;

/// `(0.8·T − root) / (stones · leaf)`, uncapped: sims a stone at ρ = 1, a descent charged a whole served leaf.
#[must_use]
pub fn quotient(turn_ms: f64, root_ms: f64, leaf_ms: f64, stones: u32) -> f64 {
    (MARGIN * turn_ms - root_ms) / (f64::from(stones.max(1)) * leaf_ms)
}

/// The turn's sims a stone: the quotient floored, held to `[floor, cap]`; `cap` is never below `floor` by construction.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // held to [0, cap] before the cast
pub fn sims_for_turn(
    turn_ms: f64,
    root_ms: f64,
    leaf_ms: f64,
    stones: u32,
    floor: u32,
    cap: u32,
) -> u32 {
    let q = quotient(turn_ms, root_ms, leaf_ms, stones);
    let whole = if q.is_nan() {
        0.0
    } else {
        q.floor().clamp(0.0, f64::from(cap))
    };
    (whole as u32).max(floor).min(cap)
}

/// The shortest turn clock at which `FLOOR` sims a stone fit two stones.
#[must_use]
pub fn min_turn_ms(root_ms: f64, leaf_ms: f64) -> f64 {
    (2.0 * f64::from(FLOOR) * leaf_ms + root_ms) / MARGIN
}

/// The sims a game's pool is sized for: one doubling over the two-stone quotient, held to `[FLOOR, 976]`.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // held to [0, 976] before the cast
pub fn pool_sims(turn_ms: f64, root_ms: f64, leaf_ms: f64) -> u32 {
    let ceiling = u32::try_from(MAX_ARMED_SIMS).unwrap_or(u32::MAX);
    let q = 2.0 * quotient(turn_ms, root_ms, leaf_ms, 2);
    let whole = if q.is_nan() {
        0.0
    } else {
        q.floor().clamp(0.0, f64::from(ceiling))
    };
    (whole as u32).clamp(FLOOR, ceiling)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_research_guaranteed_column_reproduces() {
        // RESEARCH-WEB Q5 at T 10 s, root 93 ms, two stones, ρ = 1: desk WebGPU and one CPU thread.
        assert_eq!(sims_for_turn(10_000.0, 93.0, 8.84, 2, FLOOR, 976), 447);
        assert_eq!(sims_for_turn(10_000.0, 93.0, 58.71, 2, FLOOR, 976), 67);
        assert!((min_turn_ms(93.0, 58.71) - 4_813.05).abs() < 0.01);
    }

    #[test]
    fn the_rule_holds_its_floor_its_cap_and_its_order() {
        assert_eq!(sims_for_turn(1_000.0, 93.0, 58.71, 2, FLOOR, 976), FLOOR);
        assert_eq!(
            sims_for_turn(1_000.0, 93.0, 58.71, 2, CPU_FLOOR, 976),
            CPU_FLOOR
        );
        assert_eq!(sims_for_turn(600_000.0, 93.0, 8.84, 2, FLOOR, 976), 976);
        assert_eq!(
            sims_for_turn(600_000.0, 93.0, 8.84, 2, FLOOR, 134),
            134,
            "a pool's cap binds"
        );
        assert_eq!(
            sims_for_turn(10_000.0, 93.0, 58.71, 1, FLOOR, 976),
            134,
            "one stone takes the turn's whole share"
        );
        assert_eq!(
            sims_for_turn(10_000.0, 93.0, 0.0, 2, FLOOR, 976),
            976,
            "a zero cost reads the cap, never past it"
        );
        let mut last = 0;
        for t in (2_000..60_000).step_by(1_000) {
            let s = sims_for_turn(f64::from(t), 93.0, 58.71, 2, FLOOR, 976);
            assert!(s >= last, "sims never fall as the clock grows");
            last = s;
        }
    }

    #[test]
    fn a_pool_is_sized_one_doubling_over_its_clock_and_never_below_the_floor() {
        assert_eq!(pool_sims(10_000.0, 93.0, 58.71), 134);
        assert_eq!(pool_sims(10_000.0, 93.0, 8.84), 894);
        assert_eq!(pool_sims(1_000.0, 93.0, 58.71), FLOOR);
        assert_eq!(pool_sims(600_000.0, 93.0, 8.84), 976);
    }
}
