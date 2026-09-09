//! Gumbel root state (`SearchKind::Gumbel`).
//!
//! NO CANDIDATE SET IS KEPT. Eligibility is recomputed every simulation from the
//! schedule's considered visit count, and halving is what that eligibility DOES rather
//! than a step the driver takes. The deleted legacy dialect drew a top-m candidate set
//! once and then allocated a phase budget across it, which is a different algorithm.
//!
//! WHAT IT FOLLOWS: `mctx/_src/action_selection.py::gumbel_muzero_root_action_selection`
//! and `policies.py::gumbel_muzero_policy`'s final action. Both are pinned against
//! Mctx's own outputs — see `tests/fixtures/mctx_parity/`.
//!
//! BATCHED BY ROUND, not one simulation at a time, and the equivalence is exact at the
//! root. Consecutive simulations at one considered level land on DIFFERENT children — a
//! child leaves the eligible set the moment its visit count passes the level — and the
//! schedule spends exactly `num_considered` entries at each level, so a ROUND visits every
//! eligible candidate once. `round_batch` returns that set, and issuing its descents
//! together produces the same root visit counts Mctx's sequential loop would, one round
//! trip instead of `num_considered` of them. Forcing a whole batch into ONE child would be
//! a different algorithm; this is not that.

use rand::{RngExt, SeedableRng};

use super::seq_halving::{considered_visits_sequence, score_considered};
use super::MCTSTree;

/// Per-search Mctx root state: the Gumbel draw and the halving schedule.
pub struct MctxRootState {
    /// Gumbel(0,1), one per root child, drawn ONCE per search over the whole
    /// child set — Mctx draws over every action and masks the illegal ones; a
    /// node's children here ARE its legal set, so the draw is over all of them.
    pub gumbel_values: Vec<f32>,
    /// `ln(prior)` per root child. Mctx carries logits and this repo carries
    /// probabilities; the two differ by a constant that `max_logit` removes.
    pub log_priors: Vec<f32>,
    /// `max(log_priors)` — Mctx's `logits - max(logits)` normalisation.
    pub max_logit: f32,
    /// Considered visit count per simulation index. Length IS the budget.
    pub schedule: Vec<u32>,
    /// Pool index of the root's first child.
    pub first_child: u32,
}

impl MctxRootState {
    /// Draw the Gumbel noise and build the schedule. Call once, after the root is
    /// expanded.
    ///
    /// `m` is `max_num_considered_actions`; Mctx clamps it to the number of valid
    /// actions, which here is the root's child count.
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
                // Gumbel(0,1) = -log(-log(U)). The clamp keeps both logs finite at
                // the ends of the unit interval.
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

    /// `new`, from an explicit seed rather than a caller-held RNG.
    ///
    /// EXISTS FOR THE DEPLOY HEAD, and the seed is the reason. A promotion bar has to be a
    /// reproducible instrument (LAW-15) and the Gumbel draw is the head's one stochastic
    /// term; a caller that seeds per (game, ply) gets a bar that replays exactly. It lives
    /// HERE rather than in the bridge so the RNG choice stays one authority — a
    /// bridge-side `StdRng` would be a second stream nobody could compare against this
    /// one.
    #[must_use]
    pub fn new_seeded(tree: &MCTSTree, m: usize, num_simulations: usize, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        MctxRootState::new(tree, m, num_simulations, &mut rng)
    }

    /// Mctx's `simulation_index`: the sum of the root children's visit counts.
    ///
    /// Read from the TREE rather than counted by the driver, because that is what
    /// Mctx indexes the schedule with — a driver-side counter would disagree the
    /// first time a simulation failed to reach a child.
    #[must_use]
    pub fn simulation_index(&self, tree: &MCTSTree) -> usize {
        let n = tree.pool[0].n_children as usize;
        (0..n)
            .map(|j| tree.pool[self.first_child as usize + j].n_visits as usize)
            .sum()
    }

    /// The root child this simulation must descend into, as a pool index.
    ///
    /// `None` when the schedule is exhausted, or when no child sits at the
    /// considered visit level — the second case is a desynchronised search rather
    /// than a legal state, and the caller stops rather than descending somewhere
    /// arbitrary.
    #[must_use]
    pub fn select(&self, tree: &MCTSTree, c_visit: f32, c_scale: f32) -> Option<u32> {
        let sim = self.simulation_index(tree);
        let considered = *self.schedule.get(sim)?;
        let completed = tree.root_completed_qvalues(c_visit, c_scale);
        self.argmax_at(tree, considered, &completed)
    }

    /// EVERY root child this halving round must descend into, as pool indices.
    ///
    /// THE UNIT IS THE ROUND, and the round is read off the schedule rather than counted by
    /// the driver: from the current simulation index, the run of consecutive entries at the
    /// SAME considered visit level is exactly one pass over the candidates alive at that
    /// level. So the batch is `m` leaves in the first phase, `m/2` in the next, and so on —
    /// the Gumbel analogue of `leaf_batch_size`, sized by the algorithm instead of by a knob.
    ///
    /// Ordered by SCORE, descending, so a run that is truncated by the end of the schedule
    /// spends its last entries on the candidates Sequential Halving would have kept.
    ///
    /// EMPTY when the schedule is exhausted, or when no child sits at the considered level —
    /// the second is a desynchronised search rather than a legal state, and the caller stops
    /// rather than descending somewhere arbitrary.
    #[allow(clippy::cast_possible_truncation)] // j < n_children, itself a u16
    #[must_use]
    pub fn round_batch(&self, tree: &MCTSTree, c_visit: f32, c_scale: f32) -> Vec<u32> {
        let sim = self.simulation_index(tree);
        let Some(&considered) = self.schedule.get(sim) else {
            return Vec::new();
        };
        // The round's own length: how many consecutive entries stay at this level. Capped
        // by what the schedule has left, so the LAST round of a truncated schedule is short
        // and the budget is still consumed exactly.
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
        // Descending by score, ties broken by pool index so the order is deterministic —
        // the same tie-break `argmax_at`'s strict `>` gives.
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        scored.truncate(run);
        scored.into_iter().map(|(idx, _)| idx).collect()
    }

    /// Mctx's final action: `considered_visit = max(visit_counts)`, then the same
    /// score. The winner is the highest-scoring of the MOST-VISITED children, which
    /// is what makes it Sequential Halving's answer and not a visit-count sample.
    #[must_use]
    pub fn best_action(&self, tree: &MCTSTree, c_visit: f32, c_scale: f32) -> Option<u32> {
        let n = tree.pool[0].n_children as usize;
        let max_visits = (0..n)
            .map(|j| tree.pool[self.first_child as usize + j].n_visits)
            .max()?;
        let completed = tree.root_completed_qvalues(c_visit, c_scale);
        self.argmax_at(tree, max_visits, &completed)
    }

    /// The three per-child vectors are ZIPPED rather than indexed: they are built from
    /// the root's child count at construction and `completed` from it at call time, so a
    /// zip truncates to the shortest instead of panicking if the two ever disagree.
    #[allow(clippy::cast_possible_truncation)] // j < n_children, itself a u16
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
