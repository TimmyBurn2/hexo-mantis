//! Gumbel completed-Q math (Danihelka et al., ICLR 2022 §4; `mctx/_src/qtransforms.py`).
//!
//! ONE copy of the arithmetic, shared by the three surfaces that need it: the dense
//! improved-policy export (`policy.rs::get_improved_policy`), the ragged legal-set export
//! (`get_improved_policy_ls`) and the root/interior selectors. The two exporters differ
//! ONLY in off-window handling + output container, and that divergence stays in each
//! caller's scatter stage; the shared fns return per-child masses in the SAME order the
//! caller supplied its `CqChild` slice.
//!
//! The legacy (pre-Mctx) completion — mixed value off the root's running mean `W/N`,
//! sigma-scaled against raw Q in [-1, 1] — is DELETED with its dialect. What remains is
//! `qtransform_completed_by_mix_value`: the mixed value off the root's RAW network value,
//! min-max rescaled, scaled by `(c_visit + max_visits) * c_scale`.

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

/// Zero-visit prior-fallback masses: normalized priors, one per `CqChild` in input
/// order. When `total_prior == 0` the raw (unnormalized) priors pass through unchanged.
///
/// The completed-Q exporters do NOT use this — their own arithmetic already returns the
/// normalized prior when nothing is visited (see `mctx_completed_qvalues`). It is the
/// VISIT-COUNT exporter's zero-visit arm (`get_policy_ls`), which has no other way to
/// answer, so the ls seam keeps ONE fallback authority instead of two.
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

// ── Completion ───────────────────────────────────────────────────────────────

/// Mctx's completed Q-values: mixed-value completion off the RAW root value,
/// min-max rescaled, then scaled by `(c_visit + max_visits) * c_scale`.
///
/// `c_visit` is Mctx's `maxvisit_init` and `c_scale` is its `value_scale` — the
/// same slot, not a second knob (see `SelfplayConfig`'s docstring).
///
/// `raw_value` is `tree.raw_values[node]` — the value the network produced for the node
/// before any child statistic entered it, NOT the running mean `W/N`. The distinction is
/// the one the deleted legacy completion got wrong.
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

/// Mctx `_prepare_argmax_input`: `softmax(log_prior + completed_q) - visits / (1 + Σvisits)`,
/// one score per child in input order. The caller argmaxes it.
///
/// A PURE FUNCTION over three slices rather than a method on the tree, so the parity test
/// can pin it against Mctx's own `interior_argmax_input` without building a synthetic tree
/// — the alternative was pinning only the argmax, which many wrong score vectors share.
///
/// Empty in, empty out. Empty also on a degenerate softmax (no finite logit, or a sum-exp
/// that underflows to zero), so the caller can tell "no answer" from "answer 0".
pub(super) fn mctx_interior_argmax_input(
    priors: &[f32],
    completed: &[f32],
    visits: &[u32],
) -> Vec<f32> {
    let n = priors.len().min(completed.len()).min(visits.len());
    if n == 0 {
        return Vec::new();
    }
    let logit = |j: usize| priors[j].max(1e-8).ln() + completed[j];
    let max_logit = (0..n).map(logit).fold(f32::NEG_INFINITY, f32::max);
    if !max_logit.is_finite() {
        return Vec::new();
    }
    let sum_exp: f32 = (0..n).map(|j| (logit(j) - max_logit).exp()).sum();
    if sum_exp <= 0.0 {
        return Vec::new();
    }
    let denom = 1.0 + visits[..n].iter().sum::<u32>() as f32;
    (0..n)
        .map(|j| (logit(j) - max_logit).exp() / sum_exp - visits[j] as f32 / denom)
        .collect()
}
