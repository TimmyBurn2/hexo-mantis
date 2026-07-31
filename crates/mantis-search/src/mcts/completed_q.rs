//! Shared Gumbel completed-Q math (Danihelka et al., ICLR 2022 §4, Eq. 33).
//!
//! The completed-Q improved-policy computation was duplicated byte-for-byte
//! across `policy.rs::get_improved_policy` (S1, dense) and
//! `get_improved_policy_ls` (S2, ragged legal-set). This module hosts the ONE
//! copy. The two sites differ ONLY in off-window handling + output container —
//! that divergence stays in each caller's scatter stage. The shared fns return
//! per-child masses in the SAME order the caller supplied its `CqChild` slice;
//! the caller scatters by its own coord/flat-index side-table.
//!
//! NUMERIC CONTRACT (golden-pinned, `golden_tests.rs`): the FMA forms
//! `sigma_scale.mul_add(completed_q, log_prior)` and
//! `(sum_n_f / visited_prior_sum).mul_add(policy_weighted_q, v_hat)` are FROZEN.
//! Do NOT rewrite as `a*b+c` — an FMA→mul change is sub-ULP but nonzero on x86
//! and the bit-exact goldens reject it.
//!
//! NOT a home for S3 (`gumbel.rs::score`): that is `gumbel + log_prior +
//! sigma(q_hat)` with `q_hat=0` unvisited and NO v_mix — a different rule.

/// One root child's completed-Q inputs, pre-extracted by the caller's single
/// child scan. `q_val` is already in ROOT perspective (caller applies the
/// `mr==1 ? -1 : 1` q_sign flip when reading `w_value`).
#[derive(Clone, Copy)]
pub(super) struct CqChild {
    pub visits: u32,
    pub prior: f32,
    /// Root-perspective Q = q_sign * w_value / visits (0.0 when unvisited).
    pub q_val: f32,
}

/// Per-root aggregates accumulated in the SAME caller child scan.
#[derive(Clone, Copy)]
pub(super) struct CqAgg {
    pub sum_n: u32,
    pub max_n: u32,
    pub visited_prior_sum: f32,
    pub policy_weighted_q: f32,
    /// Root value estimate W/N (`root.w_value / root.n_visits`).
    pub v_hat: f32,
}

/// v_mix: mixed value estimate for unvisited actions (paper Eq. 33).
///
/// FROZEN FMA form matching the two callers. `visited_prior_sum <= 1e-8`
/// falls back to raw `v_hat` (else-branch).
#[inline]
pub(super) fn v_mix(agg: &CqAgg) -> f32 {
    if agg.visited_prior_sum > 1e-8 {
        let sum_n_f = agg.sum_n as f32;
        // `v_hat + (sum_n_f / visited_prior_sum) * policy_weighted_q`
        // → fused FMA on the inner mul-add.
        (1.0 / (1.0 + sum_n_f))
            * (sum_n_f / agg.visited_prior_sum).mul_add(agg.policy_weighted_q, agg.v_hat)
    } else {
        agg.v_hat
    }
}

/// Completed-Q improved-policy MASSES, one per `CqChild` in input order.
///
/// Body extracted verbatim from the S1/S2 callers with the scatter removed: the
/// caller scatters the returned masses into its own container (dense `Vec<f32>`
/// vs ragged `LegalSetPolicy`).
///
/// Degenerate guards return an EMPTY vec (caller emits its empty container,
/// byte-identical to the old early `return policy;` / `return LegalSetPolicy`):
/// - `children` empty,
/// - `max_logit == -inf` (no finite logit),
/// - `sum_exp <= 0.0`.
///
/// PRECONDITION: caller has already handled the `sum_n == 0` prior-fallback case
/// (see `prior_fallback_masses`) — this fn assumes `agg.sum_n > 0`.
#[inline]
pub(super) fn improved_policy_masses(
    children: &[CqChild],
    agg: &CqAgg,
    c_visit: f32,
    c_scale: f32,
) -> Vec<f32> {
    if children.is_empty() {
        return Vec::new();
    }

    let v_mix = v_mix(agg);
    let sigma_scale = (c_visit + agg.max_n as f32) * c_scale;

    // Pass 1: max_logit over children (illegal slots never enter `children`).
    let mut max_logit = f32::NEG_INFINITY;
    for ch in children {
        let completed_q = if ch.visits > 0 {
            ch.q_val.clamp(-1.0, 1.0)
        } else {
            v_mix.clamp(-1.0, 1.0)
        };
        let log_prior = (ch.prior.max(1e-8)).ln();
        // `log_prior + sigma_scale * completed_q` → fused FMA.
        let l = sigma_scale.mul_add(completed_q, log_prior);
        if l > max_logit {
            max_logit = l;
        }
    }
    if max_logit == f32::NEG_INFINITY {
        return Vec::new();
    }

    // Pass 2: sum-exp over children.
    let mut sum_exp = 0.0f32;
    for ch in children {
        let completed_q = if ch.visits > 0 {
            ch.q_val.clamp(-1.0, 1.0)
        } else {
            v_mix.clamp(-1.0, 1.0)
        };
        let log_prior = (ch.prior.max(1e-8)).ln();
        let l = sigma_scale.mul_add(completed_q, log_prior);
        sum_exp += (l - max_logit).exp();
    }
    if sum_exp <= 0.0 {
        return Vec::new();
    }

    // Pass 3: softmax mass per child (caller scatters by its side-table).
    let mut masses = Vec::with_capacity(children.len());
    for ch in children {
        let completed_q = if ch.visits > 0 {
            ch.q_val.clamp(-1.0, 1.0)
        } else {
            v_mix.clamp(-1.0, 1.0)
        };
        let log_prior = (ch.prior.max(1e-8)).ln();
        let l = sigma_scale.mul_add(completed_q, log_prior);
        masses.push((l - max_logit).exp() / sum_exp);
    }
    masses
}

/// `sum_n == 0` prior-fallback masses: normalized priors, one per `CqChild` in
/// input order. Both S1 and S2 normalize by the same `total_prior`. When
/// `total_prior == 0` the raw (unnormalized) priors pass through unchanged —
/// byte-identical to the old behaviour where the divide is skipped.
#[inline]
pub(super) fn prior_fallback_masses(children: &[CqChild]) -> Vec<f32> {
    let mut masses: Vec<f32> = children.iter().map(|ch| ch.prior).collect();
    // WP12-R Phase T: the normalizer accumulates in f64 (cast once to f32).
    // A sequential f32 sum drifts ~2e-6 relative at the 192-child cap, so the
    // zero-visit fallback shipped a "distribution" missing unity by more than
    // the target-integrity oracles tolerate. Bit-identical on every committed
    // golden fixture (S1/S2_RED3 all-unvisited verified byte-equal); the
    // divisions below stay f32 — no formula change, accumulation only.
    let total_prior = masses.iter().map(|&m| f64::from(m)).sum::<f64>() as f32;
    if total_prior > 0.0 {
        for m in &mut masses {
            *m /= total_prior;
        }
    }
    masses
}
