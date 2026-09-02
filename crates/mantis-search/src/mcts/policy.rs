// Exceeds the 300-line soft cap: the dense + legal-set policy extractors, root
// Dirichlet, and their in-src tests port as one unit (share completed_q + setup).
//! Policy extraction for MCTSTree — temperature-applied policy,
//! Gumbel completed-Q improved policy, root children info,
//! Dirichlet noise application at root, top-visits selection.

use super::{completed_q, MCTSTree};
use crate::legal_set::LegalSetPolicy;
use fxhash::FxHashMap;

impl MCTSTree {
    /// `n_actions` is the policy stride supplied by the caller (=
    /// `spec.policy_stride()`). 19-window with a pass slot = bs²+1; a no-pass
    /// encoding = bs². The caller passes the correct stride for its encoding.
    pub fn get_policy(&self, temperature: f32, n_actions: usize) -> Vec<f32> {
        let mut policy = vec![0.0f32; n_actions];

        let root = &self.pool[0];
        if !root.is_expanded() {
            return policy;
        }

        let first = root.first_child as usize;
        let n_ch  = root.n_children  as usize;

        if temperature == 0.0 {
            if let Some(best) = (first..first + n_ch)
                .max_by_key(|&i| self.pool[i].n_visits)
            {
                let val = self.pool[best].action_idx;
                let q = (val >> 16) as i32 - 32768;
                let r = (val & 0xFFFF) as i32 - 32768;
                let action = self.root_board.window_flat_idx(q, r);

                if action < n_actions {
                    policy[action] = 1.0;
                }
            }
        } else {
            let visits: Vec<f32> = (first..first + n_ch)
                .map(|i| (self.pool[i].n_visits as f32).powf(1.0 / temperature))
                .collect();
            let total: f32 = visits.iter().sum();
            if total > 0.0 {
                for (j, &v) in visits.iter().enumerate() {
                    let val = self.pool[first + j].action_idx;
                    let q = (val >> 16) as i32 - 32768;
                    let r = (val & 0xFFFF) as i32 - 32768;
                    let action = self.root_board.window_flat_idx(q, r);

                    if action < n_actions {
                        policy[action] = v / total;
                    }
                }
            }
        }

        policy
    }

    /// Compute improved policy targets using Gumbel completed Q-values
    /// (Danihelka et al., Gumbel AlphaZero, ICLR 2022 §4, Appendix D Eq. 33).
    ///
    /// `n_actions` is the policy stride supplied by the caller (=
    /// `spec.policy_stride()`); see `get_policy` for the rationale.
    ///
    /// `child_data` already carries (action, visits, prior, q_val) and the
    /// softmax is sparse (only entries in `child_data` are non-zero before exp).
    /// The max-logit / sum-exp passes iterate `child_data` directly instead of
    /// scanning a full n_actions-wide sentinel vector, and the final exp/scatter
    /// writes straight into `policy[action]`.
    ///
    /// Returns an `n_actions`-dim probability distribution that incorporates
    /// MCTS Q-values into the prior, giving useful policy signal even at
    /// low simulation counts.
    pub fn get_improved_policy(
        &self,
        n_actions: usize,
        c_visit: f32,
        c_scale: f32,
    ) -> Vec<f32> {
        let mut policy = vec![0.0f32; n_actions];

        let root = &self.pool[0];
        if !root.is_expanded() {
            return policy;
        }

        let first = root.first_child as usize;
        let n_ch = root.n_children as usize;

        // Children store w_value in their own player-to-move perspective (backup.rs negamax).
        // When root.moves_remaining==1 the children belong to the opponent, so negate their Q
        // to bring them into root's perspective before computing completed-Q targets.
        let q_sign: f32 = if self.pool[0].moves_remaining == 1 { -1.0 } else { 1.0 };

        // The completed-Q math lives in `super::completed_q`. The ONE S1↔S2
        // divergence (off-window handling + output container) stays here in the
        // scatter: S1 drops `action >= n_actions`, scatters into a dense
        // `Vec<f32>`. `actions[i]` is the flat index for `children[i]`.
        let mut children: Vec<completed_q::CqChild> = Vec::with_capacity(n_ch);
        let mut actions: Vec<usize> = Vec::with_capacity(n_ch);
        let mut agg = completed_q::CqAgg {
            sum_n: 0, max_n: 0, visited_prior_sum: 0.0, policy_weighted_q: 0.0, v_hat: 0.0,
        };

        for j in 0..n_ch {
            let child = &self.pool[first + j];
            let val = child.action_idx;
            let q = (val >> 16) as i32 - 32768;
            let r = (val & 0xFFFF) as i32 - 32768;
            let action = self.root_board.window_flat_idx(q, r);
            if action >= n_actions {
                continue;
            }

            let visits = child.n_visits;
            let prior = child.prior;
            let q_val = if visits > 0 {
                q_sign * child.w_value / visits as f32
            } else {
                0.0
            };

            agg.sum_n += visits;
            if visits > agg.max_n {
                agg.max_n = visits;
            }
            if visits > 0 {
                agg.visited_prior_sum += prior;
                agg.policy_weighted_q += prior * q_val;
            }

            children.push(completed_q::CqChild { visits, prior, q_val });
            actions.push(action);
        }

        // Edge case: no visits at all — return prior distribution.
        if agg.sum_n == 0 {
            for (action, mass) in actions
                .iter()
                .zip(completed_q::prior_fallback_masses(&children))
            {
                policy[*action] = mass;
            }
            return policy;
        }

        // v_hat: root value estimate (W/N from root node).
        agg.v_hat = root.w_value / root.n_visits as f32;

        // Shared softmax(log_prior + sigma(completedQ)); empty ⇒ degenerate
        // guard fired (byte-identical to the old early `return policy`).
        let masses = completed_q::improved_policy_masses(&children, &agg, c_visit, c_scale);
        for (action, mass) in actions.iter().zip(masses) {
            policy[*action] = mass;
        }

        // Note: policy pruning is applied only in the Python training loop
        // to avoid double-pruning.

        policy
    }

    /// Legal-set counterpart of `get_policy`. Keys each root child by board
    /// coord into a ragged `LegalSetPolicy` (in-window → `dense`, off-window →
    /// `overflow`). NO-DROP law (authority records.rs:468-479; R34/R153/R156,
    /// WP12-R Phase T): EVERY root child is routed — in- AND off-window,
    /// covered or not — so `Σdense + Σoverflow == 1` by construction whenever
    /// any child was visited. A zero-visit root returns the prior-fallback
    /// distribution over the FULL child set — the improved exporters' exact
    /// semantics (`prior_fallback_masses`), so the ls seam has ONE fallback
    /// authority. The export never reads cluster-coverage geometry.
    pub fn get_policy_ls(&self, temperature: f32, n_actions: usize) -> LegalSetPolicy {
        let mut dense = vec![0.0f32; n_actions];
        let mut overflow: FxHashMap<(i32, i32), f32> = FxHashMap::default();

        let root = &self.pool[0];
        if !root.is_expanded() {
            return LegalSetPolicy { dense, overflow };
        }
        let first = root.first_child as usize;
        let n_ch = root.n_children as usize;

        if temperature == 0.0 {
            if let Some(best) = (first..first + n_ch).max_by_key(|&i| self.pool[i].n_visits) {
                let val = self.pool[best].action_idx;
                let q = (val >> 16) as i32 - 32768;
                let r = (val & 0xFFFF) as i32 - 32768;
                let flat = self.root_board.window_flat_idx(q, r);
                if flat < n_actions {
                    dense[flat] = 1.0;
                } else {
                    overflow.insert((q, r), 1.0);
                }
            }
            return LegalSetPolicy { dense, overflow };
        }

        let visits: Vec<f32> = (first..first + n_ch)
            .map(|i| (self.pool[i].n_visits as f32).powf(1.0 / temperature))
            .collect();
        let total: f32 = visits.iter().sum();
        if total > 0.0 {
            for (j, &v) in visits.iter().enumerate() {
                let val = self.pool[first + j].action_idx;
                let q = (val >> 16) as i32 - 32768;
                let r = (val & 0xFFFF) as i32 - 32768;
                let flat = self.root_board.window_flat_idx(q, r);
                if flat < n_actions {
                    dense[flat] = v / total;
                } else {
                    overflow.insert((q, r), v / total);
                }
            }
            return LegalSetPolicy { dense, overflow };
        }

        // total == 0 (zero-visit root, sims = 1 / inference-failure regime):
        // the prior-fallback distribution over the FULL child set — byte-equal
        // to the improved exporters' `sum_n == 0` arm (ONE fallback semantics;
        // DESIGN_T §3.1 closes the silent all-zero arm).
        let mut children: Vec<completed_q::CqChild> = Vec::with_capacity(n_ch);
        let mut coords: Vec<(i32, i32, usize)> = Vec::with_capacity(n_ch);
        for j in 0..n_ch {
            let child = &self.pool[first + j];
            let val = child.action_idx;
            let q = (val >> 16) as i32 - 32768;
            let r = (val & 0xFFFF) as i32 - 32768;
            let flat = self.root_board.window_flat_idx(q, r);
            children.push(completed_q::CqChild {
                visits: child.n_visits,
                prior: child.prior,
                q_val: 0.0,
            });
            coords.push((q, r, flat));
        }
        for (&(q, r, flat), mass) in coords
            .iter()
            .zip(completed_q::prior_fallback_masses(&children))
        {
            if flat < n_actions {
                dense[flat] = mass;
            } else {
                overflow.insert((q, r), mass);
            }
        }
        LegalSetPolicy { dense, overflow }
    }

    /// Legal-set counterpart of `get_improved_policy`. The completed-Q softmax
    /// math is FROZEN; the differences are (1) EVERY off-window child is
    /// retained (keyed into `overflow`) — the NO-DROP law (records.rs:468-479
    /// authority; the pre-Phase-T uncovered drop subset-renormalized the
    /// remaining masses, the authority's second forbidden mode) — and (2) the
    /// output is the ragged `LegalSetPolicy`. The child SET the frozen math
    /// runs over is the full root-child set; the formulas are untouched.
    pub fn get_improved_policy_ls(
        &self,
        n_actions: usize,
        c_visit: f32,
        c_scale: f32,
    ) -> LegalSetPolicy {
        let mut dense = vec![0.0f32; n_actions];
        let mut overflow: FxHashMap<(i32, i32), f32> = FxHashMap::default();

        let root = &self.pool[0];
        if !root.is_expanded() {
            return LegalSetPolicy { dense, overflow };
        }
        let first = root.first_child as usize;
        let n_ch = root.n_children as usize;
        let q_sign: f32 = if self.pool[0].moves_remaining == 1 { -1.0 } else { 1.0 };

        // The completed-Q math is shared with S1 via `super::completed_q`. The
        // ONE S1↔S2 divergence stays here: the ragged scatter (every child kept;
        // no coverage read). `coords[i] = (q, r, flat)` for `children[i]`;
        // flat >= n_actions ⇒ off-window (→ overflow).
        let mut children: Vec<completed_q::CqChild> = Vec::with_capacity(n_ch);
        let mut coords: Vec<(i32, i32, usize)> = Vec::with_capacity(n_ch);
        let mut agg = completed_q::CqAgg {
            sum_n: 0, max_n: 0, visited_prior_sum: 0.0, policy_weighted_q: 0.0, v_hat: 0.0,
        };

        for j in 0..n_ch {
            let child = &self.pool[first + j];
            let val = child.action_idx;
            let q = (val >> 16) as i32 - 32768;
            let r = (val & 0xFFFF) as i32 - 32768;
            let flat = self.root_board.window_flat_idx(q, r);

            let visits = child.n_visits;
            let prior = child.prior;
            let q_val = if visits > 0 {
                q_sign * child.w_value / visits as f32
            } else {
                0.0
            };

            agg.sum_n += visits;
            if visits > agg.max_n {
                agg.max_n = visits;
            }
            if visits > 0 {
                agg.visited_prior_sum += prior;
                agg.policy_weighted_q += prior * q_val;
            }

            children.push(completed_q::CqChild { visits, prior, q_val });
            coords.push((q, r, flat));
        }

        // Edge case: no visits — return prior distribution (all children included).
        if agg.sum_n == 0 {
            for (&(q, r, flat), mass) in coords
                .iter()
                .zip(completed_q::prior_fallback_masses(&children))
            {
                if flat < n_actions {
                    dense[flat] = mass;
                } else {
                    overflow.insert((q, r), mass);
                }
            }
            return LegalSetPolicy { dense, overflow };
        }

        agg.v_hat = root.w_value / root.n_visits as f32;

        // Shared softmax(log_prior + sigma(completedQ)); empty ⇒ degenerate
        // guard fired (byte-identical to the old early returns).
        let masses = completed_q::improved_policy_masses(&children, &agg, c_visit, c_scale);
        for (&(q, r, flat), mass) in coords.iter().zip(masses) {
            if flat < n_actions {
                dense[flat] = mass;
            } else {
                overflow.insert((q, r), mass);
            }
        }

        LegalSetPolicy { dense, overflow }
    }

    /// Returns (child_pool_index, prior) for each root child.
    /// Used by Gumbel MCTS to build the candidate list after root expansion.
    pub fn get_root_children_info(&self) -> Vec<(u32, f32)> {
        let root = &self.pool[0];
        if !root.is_expanded() {
            return Vec::new();
        }
        let first = root.first_child as usize;
        let n_ch = root.n_children as usize;
        (first..first + n_ch)
            .map(|i| (i as u32, self.pool[i].prior))
            .collect()
    }

    // The loop indexes both `self.pool[first + j]` and `noise[j]`, so the range
    // form is intentional; the FROZEN FMA blend below is byte-verbatim (S4 golden).
    #[allow(clippy::needless_range_loop)]
    pub fn apply_dirichlet_to_root(&mut self, noise: &[f32], epsilon: f32) {
        let root = &self.pool[0];
        if !root.is_expanded() {
            return;
        }
        let n_ch = root.n_children as usize;
        if noise.len() < n_ch {
            return;
        }
        let first = root.first_child as usize;

        for j in 0..n_ch {
            let child = &mut self.pool[first + j];
            // §F2: `(1.0 - epsilon) * prior + epsilon * noise` → fused FMA.
            child.prior = epsilon.mul_add(noise[j], (1.0 - epsilon) * child.prior);
        }
    }

    /// Top-N children of root by visit count.
    /// Returns Vec<((q, r), visits, prior, q_value)> sorted by visits descending.
    ///
    /// Returns raw `(i32, i32)` axial coords; Python callers format with
    /// f-strings at the call site.
    pub fn get_top_visits(&self, n: usize) -> Vec<((i32, i32), u32, f32, f32)> {
        let root = &self.pool[0];
        if !root.is_expanded() {
            return Vec::new();
        }
        let first = root.first_child as usize;
        let n_ch = root.n_children as usize;

        let mut children: Vec<(usize, u32)> = (first..first + n_ch)
            .map(|i| (i, self.pool[i].n_visits))
            .collect();
        children.sort_unstable_by_key(|&(_, visits)| std::cmp::Reverse(visits));
        children.truncate(n);

        let q_sign: f32 = if root.moves_remaining == 1 { -1.0 } else { 1.0 };
        children.into_iter().map(|(i, visits)| {
            let node = &self.pool[i];
            let val = node.action_idx;
            let q = (val >> 16) as i32 - 32768;
            let r = (val & 0xFFFF) as i32 - 32768;
            let q_value = if visits > 0 { q_sign * node.w_value / visits as f32 } else { 0.0 };
            ((q, r), visits, node.prior, q_value)
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::tests::{setup_two_child_tree, setup_expanded_root};
    use mantis_core::board::{Board, BOARD_SIZE};
    use crate::mcts::node::Node;

    #[test]
    fn test_get_policy_proportional_to_visits() {
        let (mut tree, child_a, child_b) = setup_two_child_tree(1.5);
        tree.pool[0].n_visits = 10;
        tree.pool[child_a as usize].n_visits = 7;
        tree.pool[child_b as usize].n_visits = 3;

        let policy = tree.get_policy(1.0, BOARD_SIZE * BOARD_SIZE + 1);
        let pa = policy[180];
        let pb = policy[181];
        assert!((pa - 0.7).abs() < 1e-5, "action 0 should get 70%: {pa}");
        assert!((pb - 0.3).abs() < 1e-5, "action 1 should get 30%: {pb}");
        assert!((pa + pb - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_get_policy_argmax_temperature_zero() {
        let (mut tree, child_a, child_b) = setup_two_child_tree(1.5);
        tree.pool[0].n_visits = 10;
        tree.pool[child_a as usize].n_visits = 7;
        tree.pool[child_b as usize].n_visits = 3;

        let policy = tree.get_policy(0.0, BOARD_SIZE * BOARD_SIZE + 1);
        assert_eq!(policy[180], 1.0);
        assert_eq!(policy[181], 0.0);
    }

    #[test]
    fn test_policy_sums_to_one_after_search() {
        let mut tree = MCTSTree::new(1.5);
        let board = Board::new();
        tree.new_game(board);

        let n_sims = 10;
        let uniform = vec![1.0 / (BOARD_SIZE * BOARD_SIZE + 1) as f32; BOARD_SIZE * BOARD_SIZE + 1];
        for _ in 0..n_sims {
            let leaves = tree.select_leaves(1)
        .expect("select_leaves: no desync in this fixture");
            let n = leaves.len();
            let policies: Vec<Vec<f32>> = (0..n).map(|_| uniform.clone()).collect();
            let values = vec![0.0f32; n];
            tree.expand_and_backup(&policies, &values);
        }

        let policy = tree.get_policy(1.0, BOARD_SIZE * BOARD_SIZE + 1);
        let sum: f32 = policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "policy should sum to 1.0, got {sum}");
    }

    #[test]
    fn test_dirichlet_ignored_before_root_expanded() {
        let mut tree = MCTSTree::new(1.5);
        tree.new_game(Board::new());
        tree.apply_dirichlet_to_root(&[0.5, 0.5], 0.25);
        assert_eq!(tree.root_n_children(), 0);
    }

    #[test]
    fn test_dirichlet_mixes_priors_correctly() {
        let (mut tree, child_a, child_b) = setup_two_child_tree(1.5);
        assert!(tree.pool[0].is_expanded());

        let noise   = [0.9f32, 0.1f32];
        let epsilon = 0.25f32;
        tree.apply_dirichlet_to_root(&noise, epsilon);

        let expected_a = (1.0 - epsilon) * 0.7 + epsilon * 0.9;
        let expected_b = (1.0 - epsilon) * 0.3 + epsilon * 0.1;

        let prior_a = tree.pool[child_a as usize].prior;
        let prior_b = tree.pool[child_b as usize].prior;

        assert!((prior_a - expected_a).abs() < 1e-6,
            "child_a prior: expected {expected_a:.6}, got {prior_a:.6}");
        assert!((prior_b - expected_b).abs() < 1e-6,
            "child_b prior: expected {expected_b:.6}, got {prior_b:.6}");
    }

    #[test]
    fn test_get_root_children_info_returns_correct_count() {
        let tree = setup_expanded_root();
        let info = tree.get_root_children_info();
        assert_eq!(info.len(), tree.root_n_children());
        // All priors should be > 0 (from uniform policy over full action space).
        for &(idx, prior) in &info {
            assert!(prior > 0.0, "prior for child {idx} should be > 0");
        }
        assert!(!info.is_empty(), "root should have children after expansion");
    }

    // ── get_improved_policy tests ────────────────────────────────────────────

    /// Helper: set up a tree with N root children having specified visits, w_values, and priors.
    /// Each child is placed at action (0, j) for j in 0..N.
    fn setup_improved_policy_tree(
        children: &[(u32, f32, f32)], // (visits, w_value, prior)
    ) -> MCTSTree {
        let mut tree = MCTSTree::new(1.5);
        let board = Board::new();
        tree.root_board = board;
        let n = children.len();
        tree.pool[0].first_child = 1;
        tree.pool[0].n_children = n as u16;
        tree.pool[0].moves_remaining = 2;

        let mut total_visits = 0u32;
        let mut total_w = 0.0f32;
        for (j, &(visits, w_value, prior)) in children.iter().enumerate() {
            let q = 0i32;
            let r = j as i32;
            let action_idx = ((q as u32).wrapping_add(32768) << 16) | (r as u32).wrapping_add(32768);
            tree.pool[1 + j] = Node {
                parent: 0,
                action_idx,
                n_visits: visits,
                w_value,
                prior,
                first_child: u32::MAX,
                n_children: 0,
                moves_remaining: 1,
                is_terminal: false,
                terminal_value: 0.0,
                virtual_loss_count: 0,
            };
            total_visits += visits;
            total_w += w_value;
        }
        tree.pool[0].n_visits = total_visits;
        tree.pool[0].w_value = total_w;
        tree.next_free = 1 + n as u32;
        tree
    }

    #[test]
    fn test_improved_policy_sums_to_one() {
        // Three children with different visits and Q values.
        let tree = setup_improved_policy_tree(&[
            (10, 5.0, 0.5),   // Q=0.5
            (8, -2.0, 0.3),   // Q=-0.25
            (2, 0.4, 0.2),    // Q=0.2
        ]);
        let policy = tree.get_improved_policy(BOARD_SIZE * BOARD_SIZE + 1, 50.0, 1.0);
        let sum: f32 = policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "policy should sum to 1.0, got {sum}");
    }

    #[test]
    fn test_improved_policy_no_visits_returns_prior() {
        // All children unvisited — should return normalized priors.
        let tree = setup_improved_policy_tree(&[
            (0, 0.0, 0.6),
            (0, 0.0, 0.4),
        ]);
        let policy = tree.get_improved_policy(BOARD_SIZE * BOARD_SIZE + 1, 50.0, 1.0);
        let sum: f32 = policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "prior fallback should sum to 1.0, got {sum}");

        // The two non-zero entries should roughly reflect priors.
        let nonzero: Vec<f32> = policy.iter().copied().filter(|&p| p > 0.0).collect();
        assert_eq!(nonzero.len(), 2);
        assert!(nonzero[0] > nonzero[1], "higher prior should get higher prob");
    }

    #[test]
    fn test_improved_policy_q_ordering() {
        // Two children: one clearly winning (Q=+0.9), one losing (Q=-0.9).
        // Equal priors — the improved policy should favor the winning child.
        let tree = setup_improved_policy_tree(&[
            (50, 45.0, 0.5),   // Q=+0.9
            (50, -45.0, 0.5),  // Q=-0.9
        ]);
        let policy = tree.get_improved_policy(BOARD_SIZE * BOARD_SIZE + 1, 50.0, 1.0);

        // Find the two non-zero actions.
        let (cq, cr) = tree.root_board.window_center();
        let idx_good = Board::window_flat_idx_at(0, 0, cq, cr);
        let idx_bad = Board::window_flat_idx_at(0, 1, cq, cr);

        assert!(policy[idx_good] > policy[idx_bad],
            "Q=+0.9 child should get more probability than Q=-0.9: {} vs {}",
            policy[idx_good], policy[idx_bad]);
    }

    // Note: policy_prune_frac test removed — pruning now lives only in
    // Python's training loop to avoid double-pruning.

    #[test]
    fn test_improved_policy_illegal_actions_stay_zero() {
        let tree = setup_improved_policy_tree(&[
            (10, 5.0, 0.7),
            (5, 1.0, 0.3),
        ]);
        let policy = tree.get_improved_policy(BOARD_SIZE * BOARD_SIZE + 1, 50.0, 1.0);

        // Only 2 actions should be non-zero out of board_size*board_size+1.
        let nonzero_count = policy.iter().filter(|&&p| p > 0.0).count();
        assert_eq!(nonzero_count, 2, "only legal actions should have non-zero prob, got {nonzero_count}");
    }
}
