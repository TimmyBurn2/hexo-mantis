//! Mctx-dialect Gumbel root state (`GumbelVariant::Mctx`).
//!
//! HELD APART FROM `gumbel.rs` ON PURPOSE. The legacy `GumbelSearchState::score`
//! is golden site S3 and its bits are frozen (`golden_tests.rs`: *"the completed-Q
//! refactor must NOT touch S3"*). The two dialects also do genuinely different
//! things: legacy draws a top-m candidate SET once and then allocates a phase
//! budget across it, while Mctx keeps no candidate set at all — eligibility is
//! recomputed every simulation from the schedule's considered visit count, and
//! halving is what that eligibility does rather than a step the driver takes.
//!
//! WHAT IT FOLLOWS: `mctx/_src/action_selection.py::gumbel_muzero_root_action_selection`
//! and `policies.py::gumbel_muzero_policy`'s final action. Both are pinned against
//! Mctx's own outputs — see `tests/fixtures/mctx_parity/`.
//!
//! ONE SIMULATION AT A TIME, and it is not an oversight. Mctx re-derives the root
//! choice after every backup, and consecutive simulations at one considered level
//! deliberately land on DIFFERENT children (a child leaves the eligible set the
//! moment its visit count passes the level). Forcing a whole leaf batch into one
//! child would be a different algorithm. Per-worker batching is what is given up,
//! not device batching: N workers each submitting one leaf still hand the
//! inference server N leaves to fuse.

use rand::RngExt;

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
