//! Sequential Halving's final ranking, the order the deploy audit walks under Gumbel: `best_action` leads it.

#[allow(dead_code)]
mod common;

use common::{gumbel_search_with_state, r8_board, stub_policy, C_VISIT};
use mantis_search::QSigma;

#[test]
fn the_ranking_leads_with_best_action_and_holds_every_root_child_once_by_visits() {
    let sigma = QSigma {
        c_visit: C_VISIT,
        c_scale: 1.0,
        rescale: true,
    };
    for seed in 0..6u64 {
        let (tree, state) =
            gumbel_search_with_state(&r8_board(), &stub_policy(), seed, 64, 16, 1.0);
        let ranking = state.ranking(&tree, sigma);
        assert_eq!(
            ranking.first().copied(),
            state.best_action(&tree, sigma),
            "seed {seed}"
        );
        let visits: Vec<u32> = ranking
            .iter()
            .map(|&i| tree.pool[i as usize].n_visits)
            .collect();
        assert!(
            visits.windows(2).all(|w| w[0] >= w[1]),
            "seed {seed}: more visits first"
        );
        let mut all = ranking.clone();
        all.sort_unstable();
        all.dedup();
        assert_eq!(
            all.len(),
            tree.root_n_children(),
            "seed {seed}: every root child once"
        );
    }
}
