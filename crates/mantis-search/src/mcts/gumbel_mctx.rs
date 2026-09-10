//! Gumbel root state (`SearchKind::Gumbel`).
//!
//! NO CANDIDATE SET IS KEPT: eligibility is recomputed every simulation from the schedule's
//! considered visit count, and halving is what that eligibility DOES rather than a step the
//! driver takes. BATCHED BY ROUND, exactly equivalent at the root — consecutive simulations at
//! one considered level land on DIFFERENT children and the schedule spends exactly
//! `num_considered` entries per level, so one round trip reproduces the sequential loop's root
//! visit counts. Pinned against Mctx's own outputs in `tests/fixtures/mctx_parity/`.

use rand::{RngExt, SeedableRng};

use super::seq_halving::{considered_visits_sequence, score_considered};
use super::MCTSTree;

/// Per-search Mctx root state: the Gumbel draw and the halving schedule.
pub struct MctxRootState {
    /// Gumbel(0,1), one per root child, drawn ONCE per search over the whole child set — a
    /// node's children here ARE its legal set, so there is nothing to mask.
    pub gumbel_values: Vec<f32>,
    /// `ln(prior)` per root child; Mctx carries logits and this repo probabilities, differing
    /// by a constant that `max_logit` removes.
    pub log_priors: Vec<f32>,
    /// `max(log_priors)` — Mctx's `logits - max(logits)` normalisation.
    pub max_logit: f32,
    /// Considered visit count per simulation index. Length IS the budget.
    pub schedule: Vec<u32>,
    /// Pool index of the root's first child.
    pub first_child: u32,
}

impl MctxRootState {
    /// Draw the Gumbel noise and build the schedule; call once, after the root is expanded.
    /// `m` is `max_num_considered_actions`, clamped to the root's child count.
    pub fn new(
        tree: &MCTSTree,
        m: usize,
        num_simulations: usize,
        rng: &mut impl rand::Rng,
    ) -> Self {
        let root = &tree.pool[0];
        let n_children = root.n_children as usize;
        let first_child = root.first_child;

        let gumbel_values: Vec<f32> = (0..n_children)
            .map(|_| {
                // Gumbel(0,1) = -log(-log(U)); the clamp keeps both logs finite at the ends.
                let u: f32 = rng.random::<f32>().clamp(1e-10, 1.0 - 1e-7);
                -(-u.ln()).ln()
            })
            .collect();

        let log_priors: Vec<f32> = (0..n_children)
            .map(|j| tree.pool[first_child as usize + j].prior.max(1e-8).ln())
            .collect();
        let max_logit = log_priors.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        MctxRootState {
            gumbel_values,
            log_priors,
            max_logit,
            schedule: considered_visits_sequence(m.min(n_children).max(1), num_simulations),
            first_child,
        }
    }

    /// `new`, from an explicit seed rather than a caller-held RNG: a promotion bar must be a
    /// reproducible instrument and the Gumbel draw is the deploy head's one stochastic term. It
    /// lives here, not in the bridge, so the RNG choice stays one authority.
    #[must_use]
    pub fn new_seeded(tree: &MCTSTree, m: usize, num_simulations: usize, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        MctxRootState::new(tree, m, num_simulations, &mut rng)
    }

    /// Mctx's `simulation_index`, read from the TREE because that is what Mctx indexes the
    /// schedule with — a driver-side counter would disagree the first time a simulation reached
    /// no child.
    #[must_use]
    pub fn simulation_index(&self, tree: &MCTSTree) -> usize {
        let n = tree.pool[0].n_children as usize;
        (0..n)
            .map(|j| tree.pool[self.first_child as usize + j].n_visits as usize)
            .sum()
    }

    /// The root child this simulation must descend into; `None` on an exhausted schedule or a
    /// desynchronised search, where the caller stops rather than descending arbitrarily.
    #[must_use]
    pub fn select(&self, tree: &MCTSTree, c_visit: f32, c_scale: f32) -> Option<u32> {
        let sim = self.simulation_index(tree);
        let considered = *self.schedule.get(sim)?;
        let completed = tree.root_completed_qvalues(c_visit, c_scale);
        self.argmax_at(tree, considered, &completed)
    }

    /// EVERY root child this halving round must descend into, as pool indices.
    ///
    /// THE UNIT IS THE ROUND, read off the schedule: the run of consecutive entries at the SAME
    /// considered visit level is one pass over the candidates alive at that level, so the batch
    /// is `m` leaves, then `m/2` — sized by the algorithm rather than a knob. Ordered by SCORE
    /// descending, and EMPTY on an exhausted schedule or a desynchronised search.
    #[allow(clippy::cast_possible_truncation)] // j < n_children, itself a u16
    #[must_use]
    pub fn round_batch(&self, tree: &MCTSTree, c_visit: f32, c_scale: f32) -> Vec<u32> {
        let sim = self.simulation_index(tree);
        let Some(&considered) = self.schedule.get(sim) else {
            return Vec::new();
        };
        // The round's own length, capped by what the schedule has left, so the LAST round of a
        // truncated schedule is short and the budget is still consumed exactly.
        let run = self.schedule[sim..]
            .iter()
            .take_while(|&&v| v == considered)
            .count();

        let completed = tree.root_completed_qvalues(c_visit, c_scale);
        let mut scored: Vec<(u32, f32)> = Vec::new();
        for (j, ((&q, &gumbel), &log_prior)) in completed
            .iter()
            .zip(&self.gumbel_values)
            .zip(&self.log_priors)
            .enumerate()
        {
            let visits = tree.pool[self.first_child as usize + j].n_visits;
            if let Some(score) =
                score_considered(considered, visits, gumbel, log_prior, self.max_logit, q)
            {
                scored.push((self.first_child + j as u32, score));
            }
        }
        // Descending by score, ties broken by pool index for determinism — the same tie-break
        // `argmax_at`'s strict `>` gives.
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        scored.truncate(run);
        scored.into_iter().map(|(idx, _)| idx).collect()
    }

    /// Mctx's final action: the highest-scoring of the MOST-VISITED children, which is
    /// Sequential Halving's answer rather than a visit-count sample.
    #[must_use]
    pub fn best_action(&self, tree: &MCTSTree, c_visit: f32, c_scale: f32) -> Option<u32> {
        let n = tree.pool[0].n_children as usize;
        let max_visits = (0..n)
            .map(|j| tree.pool[self.first_child as usize + j].n_visits)
            .max()?;
        let completed = tree.root_completed_qvalues(c_visit, c_scale);
        self.argmax_at(tree, max_visits, &completed)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn argmax_at(&self, tree: &MCTSTree, considered: u32, completed: &[f32]) -> Option<u32> {
        let mut best: Option<(u32, f32)> = None;
        for (j, ((&q, &gumbel), &log_prior)) in completed
            .iter()
            .zip(&self.gumbel_values)
            .zip(&self.log_priors)
            .enumerate()
        {
            let visits = tree.pool[self.first_child as usize + j].n_visits;
            let Some(score) =
                score_considered(considered, visits, gumbel, log_prior, self.max_logit, q)
            else {
                continue;
            };
            // Strict `>` so the first of equal scores wins, matching `argmax`.
            if best.is_none_or(|(_, b)| score > b) {
                best = Some((self.first_child + j as u32, score));
            }
        }
        best.map(|(idx, _)| idx)
    }
}
