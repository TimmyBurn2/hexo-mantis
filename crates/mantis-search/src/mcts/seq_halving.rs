//! Sequential-Halving visit schedule, ported from Mctx `_src/seq_halving.py`.
//!
//! Mctx does not allocate Sequential Halving as "phases × sims-per-candidate".
//! It precomputes, for the whole search, the sequence of CONSIDERED VISIT COUNTS
//! — one entry per simulation — and at simulation `i` the root may only descend
//! into a candidate whose current visit count equals `sequence[i]`. Halving is
//! then implicit: once every candidate at level `v` has been visited the eligible
//! set shrinks, and the schedule moves on.
//!
//! WHY THE SHAPE MATTERS AND NOT ONLY THE RATIO. The sequence has length exactly
//! `num_simulations`, so the budget is consumed exactly. The prior driver derived
//! `sims_per = remaining_budget / (remaining_phases * candidates)` per phase and
//! never allocated the integer-division remainder — measured at 49 of 50 and 599
//! of 600 (`mantis-selfplay/tests/served_sims_exact.rs`).

/// Mctx's floor on an eligible candidate's score. Keeps the argmax finite when a
/// log-prior underflows, without letting a real score reach `-inf`.
const LOW_LOGIT: f32 = -1e9;

/// Sequence of considered visit counts, one entry per simulation.
///
/// Verbatim port of `get_sequence_of_considered_visits`. `m <= 1` degenerates to
/// `0..num_simulations` — a single candidate is visited once per simulation, so
/// its considered count rises every step.
///
/// Mctx computes `int(num_simulations / (log2max * num_considered))` as a float
/// divide truncated to int; this uses integer division, which agrees for every
/// value representable here (both operands are small positive integers) and does
/// not depend on binary64 rounding.
// `num_simulations` is bounded by `MAX_ARMED_SIMS` (~1302) at runner construction, so
// the index never approaches `u32::MAX`.
#[allow(clippy::cast_possible_truncation)]
#[must_use]
pub fn considered_visits_sequence(m: usize, num_simulations: usize) -> Vec<u32> {
    if m <= 1 {
        return (0..num_simulations as u32).collect();
    }
    let log2max = m.next_power_of_two().trailing_zeros().max(1) as usize;
    let mut sequence: Vec<u32> = Vec::with_capacity(num_simulations);
    let mut visits = vec![0u32; m];
    let mut num_considered = m;
    while sequence.len() < num_simulations {
        let num_extra_visits = (num_simulations / (log2max * num_considered)).max(1);
        for _ in 0..num_extra_visits {
            sequence.extend_from_slice(&visits[..num_considered]);
            for v in visits.iter_mut().take(num_considered) {
                *v += 1;
            }
        }
        num_considered = (num_considered / 2).max(2);
    }
    sequence.truncate(num_simulations);
    sequence
}

/// Mctx `score_considered`, as a per-candidate score plus an eligibility verdict.
///
/// Returns `None` for a candidate whose visit count is not the considered one —
/// Mctx's `-inf` penalty, kept as an `Option` so a caller cannot accidentally
/// carry `-inf` into an arithmetic that would produce `NaN`.
///
/// `max_logit` is the caller's `max` over the candidate log-priors; Mctx
/// subtracts it (`logits - max(logits)`) before scoring. The subtraction is a
/// constant shift and cannot change an argmax, but it is kept because it is what
/// puts the score on Mctx's scale and hence in range of the `low_logit` floor.
#[inline]
#[must_use]
pub fn score_considered(
    considered_visit: u32,
    visit_count: u32,
    gumbel: f32,
    log_prior: f32,
    max_logit: f32,
    normalized_q: f32,
) -> Option<f32> {
    if visit_count != considered_visit {
        return None;
    }
    Some((gumbel + (log_prior - max_logit) + normalized_q).max(LOW_LOGIT))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_schedule_consumes_exactly_its_budget() {
        for m in [1usize, 2, 4, 8, 16] {
            for n in [2usize, 8, 32, 50, 96] {
                assert_eq!(
                    considered_visits_sequence(m, n).len(),
                    n,
                    "m={m} n={n}: the schedule must have one entry per simulation — a \
                     shorter one leaves budget unallocated, which is the 49-of-50 defect"
                );
            }
        }
    }

    #[test]
    fn a_schedule_never_skips_a_visit_level() {
        // Every candidate reaching level v+1 must have passed through v: the
        // sequence is non-decreasing in the count each successive entry considers
        // for a FIXED candidate, which shows up as a non-decreasing max.
        let seq = considered_visits_sequence(8, 96);
        let mut running_max = 0;
        for &v in &seq {
            assert!(
                v <= running_max + 1,
                "considered visit {v} jumps past {running_max}+1 — a level with no \
                 eligible candidate stalls the root selector"
            );
            running_max = running_max.max(v);
        }
    }

    #[test]
    fn a_single_candidate_is_visited_once_per_simulation() {
        assert_eq!(considered_visits_sequence(1, 5), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn score_is_none_off_the_considered_level() {
        assert!(score_considered(2, 3, 0.0, 0.0, 0.0, 0.0).is_none());
        assert!(score_considered(2, 2, 0.0, 0.0, 0.0, 0.0).is_some());
    }

    #[test]
    fn score_floors_at_the_low_logit() {
        let s = score_considered(0, 0, -1e30, 0.0, 0.0, 0.0).expect("eligible at level 0");
        assert_eq!(s, -1e9, "the floor is Mctx's low_logit, not the raw sum");
    }
}
