//! The pending-loss frame only reaches a batched PUCT selection: a Gumbel search plays the same tree in either frame.

#[allow(dead_code)]
mod common;

use common::{gumbel_search_in_frame, r8_board, stub_policy, C_VISIT};
use mantis_search::QSigma;

#[test]
fn a_gumbel_search_is_bit_identical_in_either_pending_loss_frame() {
    let sigma = QSigma {
        c_visit: C_VISIT,
        c_scale: 1.0,
        rescale: true,
    };
    for seed in 0..4u64 {
        let (old, old_state) =
            gumbel_search_in_frame(&r8_board(), &stub_policy(), seed, 64, 16, 1.0, false);
        let (new, new_state) =
            gumbel_search_in_frame(&r8_board(), &stub_policy(), seed, 64, 16, 1.0, true);
        let stats = |t: &mantis_search::MCTSTree| -> Vec<(u32, u32)> {
            let root = &t.pool[0];
            (root.first_child..root.first_child + u32::from(root.n_children))
                .map(|i| {
                    (
                        t.pool[i as usize].n_visits,
                        t.pool[i as usize].w_value.to_bits(),
                    )
                })
                .collect()
        };
        assert_eq!(
            stats(&old),
            stats(&new),
            "seed {seed}: the root's children moved"
        );
        assert_eq!(
            old_state.ranking(&old, sigma),
            new_state.ranking(&new, sigma),
            "seed {seed}"
        );
    }
}
