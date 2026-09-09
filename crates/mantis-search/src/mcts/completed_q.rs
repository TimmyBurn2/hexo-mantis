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
    /// The value BACKED UP at root expansion — Mctx's `tree.raw_values[root]`.
    /// Read only by the Mctx arm; the legacy arm's `v_mix` uses `v_hat`.
    pub raw_value: f32,
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

// ── Mctx arm (GUMBEL-REPAIR-1) ───────────────────────────────────────────────
//
// `qtransform_completed_by_mix_value` from `mctx/_src/qtransforms.py`, kept
// APART from the legacy functions above rather than folded into them. Two
// reasons, both load-bearing: the legacy path is byte-pinned by
// `golden_tests.rs` and a shared body would put a branch inside frozen
// arithmetic; and the two arms disagree about their own aggregates — Mctx floors
// every prior at the dtype's tiny value before summing, which the legacy
// accumulation deliberately does not do. Recomputing here from `children` keeps
// each arm's guards its own.

/// Mctx's completed Q-values: mixed-value completion off the RAW root value,
/// min-max rescaled, then scaled by `(c_visit + max_visits) * c_scale`.
///
/// `c_visit` is Mctx's `maxvisit_init` and `c_scale` is its `value_scale` — the
/// same slot, not a second knob (see `SelfplayConfig`'s docstring).
///
/// THE DEVIATION THIS FUNCTION EXISTS FOR is `raw_value`. The legacy `v_mix`
/// takes the root's BACKED-UP mean `W/N`; Mctx takes `tree.raw_values[root]`,
/// the value the network produced for the root before any child statistic
/// entered it. Every other term of the mixed value already agreed.
///
/// All-unvisited is not special-cased: every completed value is then `v_mix`,
/// the rescale maps a constant vector to zeros, and the caller's
/// `softmax(log_prior + 0)` is the prior. Verified against Mctx's own output for
/// that case rather than reasoned about.
pub(super) fn mctx_completed_qvalues(
    children: &[CqChild],
    raw_value: f32,
    c_visit: f32,
    c_scale: f32,
) -> Vec<f32> {
    /// Mctx's `epsilon` for the rescale denominator.
    const EPSILON: f32 = 1e-8;

    if children.is_empty() {
        return Vec::new();
    }

    let mut sum_n: u32 = 0;
    let mut max_n: u32 = 0;
    let mut sum_probs = 0.0f32;
    let mut prior_weighted_q = 0.0f32;
    for ch in children {
        sum_n += ch.visits;
        max_n = max_n.max(ch.visits);
        if ch.visits > 0 {
            // Mctx: `prior_probs = maximum(finfo.tiny, prior_probs)` BEFORE the sum,
            // so a visited child with a zero prior cannot make the denominator zero.
            let p = ch.prior.max(f32::MIN_POSITIVE);
            sum_probs += p;
            prior_weighted_q += p * ch.q_val;
        }
    }

    let sum_n_f = sum_n as f32;
    let weighted_q = if sum_probs > 0.0 {
        prior_weighted_q / sum_probs
    } else {
        0.0
    };
    let v_mix = sum_n_f.mul_add(weighted_q, raw_value) / (sum_n_f + 1.0);

    let mut completed: Vec<f32> = children
        .iter()
        .map(|ch| if ch.visits > 0 { ch.q_val } else { v_mix })
        .collect();

    // Rescale over the COMPLETED vector — Mctx's `_rescale_qvalues` takes min/max
    // across all actions AFTER completion, not across the visited ones only.
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    for &v in &completed {
        min_v = min_v.min(v);
        max_v = max_v.max(v);
    }
    let span = (max_v - min_v).max(EPSILON);
    let visit_scale = (c_visit + max_n as f32) * c_scale;
    for v in &mut completed {
        *v = (*v - min_v) / span * visit_scale;
    }
    completed
}

/// `softmax(log_prior + completed_q)` — Mctx's `action_weights`, one mass per
/// `CqChild` in input order. Empty in for empty out.
pub(super) fn mctx_improved_policy_masses(
    children: &[CqChild],
    raw_value: f32,
    c_visit: f32,
    c_scale: f32,
) -> Vec<f32> {
    let completed = mctx_completed_qvalues(children, raw_value, c_visit, c_scale);
    if completed.is_empty() {
        return Vec::new();
    }
    let mut logits: Vec<f32> = children
        .iter()
        .zip(&completed)
        .map(|(ch, &q)| (ch.prior.max(1e-8)).ln() + q)
        .collect();
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max_logit.is_finite() {
        return Vec::new();
    }
    let mut sum_exp = 0.0f32;
    for l in &mut logits {
        *l = (*l - max_logit).exp();
        sum_exp += *l;
    }
    if sum_exp <= 0.0 {
        return Vec::new();
    }
    for l in &mut logits {
        *l /= sum_exp;
    }
    logits
}
