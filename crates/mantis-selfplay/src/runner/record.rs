// >300 justify (R8): the record phase is ONE dispatch with two mutually exclusive arms, and
// the pair only reads correctly side by side — the dense arm's K-cluster scatter loop and the
// graph arm's whole-board single push are the two answers to the same question, and which one
// runs is the `is_graph` branch in the caller. It crossed the cap when item 10(b) added the K
// histogram, whose entire correctness argument is that ONE of these two functions takes the
// counter and the other structurally cannot: splitting them would put that argument's two
// halves in different files, where the absence emitted on a graph run stops being a one-diff
// read. The tests live here because `record_position` is `pub(crate)` and no integration test
// can reach it.
//! Record phase (WP6 D1/D14) — `record_position` /
//! `record_position_graph_dispatch` (frozen `worker_loop/inner.rs:1507/1590`,
//! dispatch branch `:1398`), called BEFORE `apply_move` so the pre-move
//! `board.ply` is the row's `ply_index` (LAW-03 measurement-unit: never reframed
//! in ply-parity units).
//!
//! Consumes the WP6 `records.rs` producers (dense + ls + graph) and applies the
//! per-game D6 record-time rotation over the WP5 `replay/sym.rs` tables via the
//! [`super::rotate`] wrappers. The dense encode kernels live in `mantis_encoding`
//! (WP3 relocated them off `Board`): `encode_state_to_buffer_channels` +
//! `encode_chain_planes` are free functions taking the board.

use std::sync::atomic::{AtomicU64, Ordering};

use mantis_core::{Board, Player};
use mantis_encoding::{encode_chain_planes, encode_state_to_buffer_channels};

use crate::records::{self, TargetIntegrityError};
use crate::replay::hexg::GraphRecord;
use crate::replay::sym::SymTables;

use super::rotate::{rotate_chain_inplace, rotate_policy_inplace, rotate_state_inplace};
use super::search_drive::MovePolicy;

/// Per-position record tuple pushed into `records_vec` (frozen `inner.rs:123`).
/// Field order: `(feat, chain, projected_policy, player, center_q, center_r,
/// is_full_search, ply_index)`. `ply_index` (CF-4) is the 0-based pre-move ply,
/// shared across the K cluster rows of one decision.
pub(crate) type RecordTuple = (Vec<f32>, Vec<f32>, Vec<f32>, Player, i32, i32, bool, u16);

/// Width of the in-run K histogram (item 10(b), LAW-18): index `i` in `0..8`
/// counts a recorded position that expanded into exactly `i + 1` cluster views,
/// and the LAST index is a guard bucket.
///
/// 8 real buckets because `k_max` is 8 on every registered encoding that has more
/// than one view (`registry.toml`; `k_max_never_exceeds_the_real_bucket_range`
/// below derives that bound from `all_specs()` rather than restating it, so a
/// registry bump reds on the commit that raises it).
pub(crate) const K_CLUSTER_HISTOGRAM_BUCKETS: usize = 9;

/// Bucket index for a K of `k`: `k - 1` inside `1..=8`, the guard bucket otherwise.
///
/// The guard catches BOTH ends and is deliberately not attributed to any real K.
/// `k > 8` is the registry past its declared `k_max`; `k == 0` is
/// `get_cluster_views` returning no centre at all, a call that records NO row
/// (the scatter loop does not execute). Folding either into the `K = 1` bucket
/// would publish a measurement of a position that was never expanded that way —
/// the fabrication class R250 exists to forbid, one layer down from the event.
#[inline]
pub(crate) const fn k_cluster_bucket(k: usize) -> usize {
    if k >= 1 && k < K_CLUSTER_HISTOGRAM_BUCKETS {
        k - 1
    } else {
        K_CLUSTER_HISTOGRAM_BUCKETS - 1
    }
}

/// Per-move position recorder (frozen `inner.rs:1507`; warm path).
///
/// Encodes per-cluster state/chain/policy buffers, forward-scatters under the
/// per-game symmetry, and pushes one record per cluster view into `records_vec`.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::fn_params_excessive_bools)] // ports per-move flag locals from caller; bundle would re-pack
pub(crate) fn record_position(
    board: &Board,
    kept_planes: &'static [usize],
    n_cells: usize,
    agg_trunk_sz: i32,
    is_fast_game: bool,
    completed_q_values: bool,
    policy_stride: usize,
    has_pass_slot: bool,
    target_policy: &MovePolicy,
    sym_idx: usize,
    sym_tables: &'static SymTables,
    move_is_full_search: bool,
    records_vec: &mut Vec<RecordTuple>,
    gridls_zero_policy_rows: &AtomicU64,
    k_cluster_histogram: &[AtomicU64; K_CLUSTER_HISTOGRAM_BUCKETS],
) {
    let (views, centers) = board.get_cluster_views();
    // Item 10(b) / LAW-18 (R250): the K-cluster lever's own in-run fire-rate log.
    // K — how many cluster views this decision expands into — is known HERE and
    // nowhere else, and a run could previously only be told K_avg after the fact,
    // which cannot separate "K is 1 everywhere, the lever is dead" from "K is
    // spread and the multi-window path is doing work". ONE relaxed RMW on a
    // 9-element array, no allocation and no branch past the bucket clamp, on a
    // path that already heap-allocates K feature buffers per call.
    //
    // The GRAPH record path cannot reach this: `record_position_graph_dispatch`
    // does not take the histogram as a parameter, so its absence on a graph run
    // is a compile-time property of the seam, not a runtime gate anyone can flip.
    k_cluster_histogram[k_cluster_bucket(centers.len())].fetch_add(1, Ordering::Relaxed);
    // §P11: hoist legal_moves once across the K cluster scatters.
    let record_legal_moves = board.legal_moves();
    for (k, center) in centers.iter().enumerate() {
        let mut feat = vec![0.0f32; kept_planes.len() * n_cells];
        encode_state_to_buffer_channels(board, &views[k], &mut feat, kept_planes, n_cells);
        // Compute Q13 chain-length planes separately (not in state).
        let mut chain = vec![0.0f32; 6 * n_cells];
        encode_chain_planes(
            &views[k][..n_cells],
            &views[k][n_cells..2 * n_cells],
            &mut chain,
            n_cells,
            agg_trunk_sz,
        );
        // Fast games: zero-policy marks value-only targets (unless completed
        // Q-values are enabled, which give signal even at 50 sims).
        let mut projected_policy = if is_fast_game && !completed_q_values {
            vec![0.0; policy_stride]
        } else {
            match target_policy {
                MovePolicy::Dense(t) => records::aggregate_policy_to_local(
                    policy_stride, has_pass_slot, agg_trunk_sz, board, center, t, &record_legal_moves,
                ),
                MovePolicy::Ls(ls) => {
                    let row = records::aggregate_policy_to_local_ls(
                        policy_stride, has_pass_slot, agg_trunk_sz, board, center, ls, &record_legal_moves,
                    );
                    // LAW-18 (DESIGN_T §3.6): count each §3.5 zero-row fill —
                    // a cluster window that saw zero visit mass records the
                    // value-only sentinel row instead of a fabricated uniform.
                    if row.iter().all(|&p| p == 0.0) {
                        gridls_zero_policy_rows.fetch_add(1, Ordering::Relaxed);
                    }
                    row
                }
            }
        };
        // §130: forward-scatter the recorded state, chain, and policy into the
        // rotated frame.
        if sym_idx != 0 {
            rotate_state_inplace(&mut feat, sym_idx, sym_tables);
            rotate_chain_inplace(&mut chain, sym_idx, sym_tables);
            rotate_policy_inplace(&mut projected_policy, sym_idx, sym_tables, n_cells);
        }
        // CF-4: record this decision's 0-based ply index (pre-move `board.ply`).
        records_vec.push((
            feat,
            chain,
            projected_policy,
            board.current_player,
            center.0,
            center.1,
            move_is_full_search,
            board.ply.index() as u16,
        ));
    }
}

/// Graph sibling of `record_position` (frozen `inner.rs:1590`). Whole-board (NO
/// K-cluster loop, NO dense planes) — pushes ONE compact `GraphRecord` via the
/// `records::record_position_graph` primitive.
///
/// `target_policy` is guaranteed `MovePolicy::Ls` whenever this is called: a
/// graph spec forces `legal_set = true` (D2), and the search drive only builds
/// `MovePolicy::Dense` when `legal_set == false`. The `Dense` arm is therefore
/// structurally unreachable — an always-on `unreachable!()` is the correct
/// die-loud response.
///
/// # Errors
/// WP12-R Phase T (DESIGN_T §3.3/§3.4): forwards `record_position_graph`'s
/// typed [`TargetIntegrityError`] to the caller, which latches it run-fatal
/// (LAW-14) — the record that would carry a degenerate target cannot be built.
#[cold]
#[inline(never)]
pub(crate) fn record_position_graph_dispatch(
    board: &Board,
    target_policy: &MovePolicy,
    trunk_sz: i32,
    move_is_full_search: bool,
    graph_records_vec: &mut Vec<GraphRecord>,
    visit_capacity: usize,
) -> Result<(), TargetIntegrityError> {
    let ls = match target_policy {
        MovePolicy::Ls(ls) => ls,
        MovePolicy::Dense(_) => unreachable!(
            "record_position_graph_dispatch: target_policy must be MovePolicy::Ls for a \
             graph spec — D2 forces legal_set=true whenever spec.representation is Graph"
        ),
    };
    let current_player = board.current_player as i8;
    let moves_remaining = board.moves_remaining;
    let ply_index = board.ply.index() as u16;
    let rec = records::record_position_graph(
        board,
        ls,
        trunk_sz,
        current_player,
        moves_remaining,
        ply_index,
        move_is_full_search,
        visit_capacity,
    )?;
    graph_records_vec.push(rec);
    Ok(())
}


#[cfg(test)]
mod k_histogram_tests {
    use super::{
        k_cluster_bucket, record_position, record_position_graph_dispatch,
        K_CLUSTER_HISTOGRAM_BUCKETS,
    };
    use crate::replay::hexg::GraphRecord;
    use crate::replay::sym::sym_tables_for;
    use crate::runner::search_drive::MovePolicy;
    use mantis_core::Board;
    use mantis_encoding::{all_specs, lookup_or_panic, RegistrySpec};
    use mantis_search::LegalSetPolicy;
    use fxhash::FxHashMap;
    use std::sync::atomic::{AtomicU64, Ordering};

    type Hist = [AtomicU64; K_CLUSTER_HISTOGRAM_BUCKETS];

    fn fresh_hist() -> Hist {
        std::array::from_fn(|_| AtomicU64::new(0))
    }

    fn read(h: &Hist) -> Vec<u64> {
        h.iter().map(|s| s.load(Ordering::Relaxed)).collect()
    }

    /// A board with `groups` well-separated stone groups — `get_cluster_views`
    /// returns one centre per group, so K is chosen by construction rather than
    /// asserted from a transcribed constant.
    fn board_with_groups(groups: usize) -> Board {
        let mut b = Board::new();
        for g in 0..groups {
            let base = g as i32 * 40;
            b.apply_move(base, base).expect("group stone must be placeable");
        }
        b
    }

    fn drive_dense(board: &Board, spec: &'static RegistrySpec, hist: &Hist) {
        let mut records = Vec::new();
        let zero_rows = AtomicU64::new(0);
        record_position(
            board,
            spec.kept_plane_indices,
            spec.n_cells(),
            spec.trunk_size as i32,
            true,  // is_fast_game — value-only rows; the K count is what is under test
            false, // completed_q_values
            spec.policy_stride(),
            spec.has_pass_slot,
            &MovePolicy::Dense(vec![0.0; spec.policy_stride()]),
            0, // sym_idx 0 — no rotation
            sym_tables_for(spec),
            true,
            &mut records,
            &zero_rows,
            hist,
        );
    }

    /// The bucket map, end to end including BOTH guard ends.
    ///
    /// FALSIFYING MUTATION: change the clamp in `k_cluster_bucket` (drop the
    /// `k >= 1` arm, or make the upper test `<=`). MUST turn this RED.
    #[test]
    fn k_cluster_bucket_gives_every_real_k_its_own_slot_and_guards_both_ends() {
        for k in 1..K_CLUSTER_HISTOGRAM_BUCKETS {
            assert_eq!(k_cluster_bucket(k), k - 1, "K={k} must own bucket {}", k - 1);
        }
        let guard = K_CLUSTER_HISTOGRAM_BUCKETS - 1;
        assert_eq!(k_cluster_bucket(0), guard, "K=0 records NO row and is not a K=1 position");
        assert_eq!(k_cluster_bucket(K_CLUSTER_HISTOGRAM_BUCKETS), guard, "K past k_max guards");
        assert_eq!(k_cluster_bucket(usize::MAX), guard, "no K may index out of the array");
    }

    /// The 8 real buckets are only honest while no registered encoding declares a
    /// larger `k_max`. DERIVED from the registry (R192(e)) so a bump reds on the
    /// commit that raises it instead of silently routing real Ks into the guard.
    #[test]
    fn no_registered_encoding_declares_a_k_max_past_the_real_buckets() {
        let real_buckets = (K_CLUSTER_HISTOGRAM_BUCKETS - 1) as u32;
        for spec in all_specs() {
            assert!(
                spec.k_max <= real_buckets,
                "{} declares k_max={} but the histogram has {real_buckets} real buckets — \
                 widen K_CLUSTER_HISTOGRAM_BUCKETS, or every position at that K lands in the \
                 guard and the distribution stops being readable",
                spec.name,
                spec.k_max,
            );
        }
    }

    /// LAW-07 producer test + the increment's MUTATION pin: a recorded position is
    /// counted in the bucket for the K it actually expanded into, once per call.
    ///
    /// K is read off the SAME `get_cluster_views` the recorder uses, so the
    /// expectation is derived rather than transcribed and the test still bites if
    /// the clustering rules move.
    ///
    /// FALSIFYING MUTATION: delete the `k_cluster_histogram[..].fetch_add(1, ..)`
    /// line in `record_position`. MUST turn this RED.
    #[test]
    fn record_position_counts_each_position_in_its_own_k_bucket() {
        let spec = lookup_or_panic("v6");
        for groups in [1usize, 2, 3] {
            let board = board_with_groups(groups);
            let k = board.get_cluster_views().1.len();
            assert_eq!(k, groups, "premise: {groups} separated groups must give K={groups}");

            let hist = fresh_hist();
            drive_dense(&board, spec, &hist);

            let mut expected = vec![0u64; K_CLUSTER_HISTOGRAM_BUCKETS];
            expected[k_cluster_bucket(k)] = 1;
            assert_eq!(
                read(&hist),
                expected,
                "one recorded position at K={k} must add exactly one count to bucket {} and \
                 touch no other bucket",
                k_cluster_bucket(k),
            );

            // Cumulative, not set-once: a second call at the same K adds a second count.
            drive_dense(&board, spec, &hist);
            expected[k_cluster_bucket(k)] = 2;
            assert_eq!(read(&hist), expected, "the histogram must accumulate across calls");
        }
    }

    /// R250 at the PRODUCER: the graph record path leaves the histogram at zero.
    ///
    /// This is why the emitter may omit the field on a graph run rather than
    /// publish its zeros — the zeros are "no producer", not "K was never 1..=8".
    /// The absence is structural: `record_position_graph_dispatch` does not take
    /// the histogram as a parameter at all, so no runtime gate has to hold. This
    /// test drives the real function so that a future edit which THREADS the
    /// histogram into the graph arm (and thereby makes the emitted absence a lie)
    /// reds here rather than shipping.
    #[test]
    fn the_graph_record_path_never_touches_the_k_histogram() {
        let mut b = Board::new();
        b.apply_move(0, 0).expect("stone");
        b.apply_move(2, 0).expect("stone");
        b.apply_move(0, 2).expect("stone");
        let (bcq, bcr) = b.window_center();
        let legal = b.legal_moves();
        let mut dense = vec![0.0f32; 362];
        for (cell, mass) in [(legal[0], 0.6f32), (legal[1], 0.4)] {
            let idx = Board::window_flat_idx_at_geom(cell.0, cell.1, bcq, bcr, 19, 9);
            assert!(idx < 362, "premise: the chosen legal cell must be in-window");
            dense[idx] = mass;
        }
        let ls = MovePolicy::Ls(LegalSetPolicy { dense, overflow: FxHashMap::default() });

        let hist = fresh_hist();
        let mut graph_records: Vec<GraphRecord> = Vec::new();
        record_position_graph_dispatch(&b, &ls, 19, true, &mut graph_records, 128)
            .expect("a full-mass target must record");

        assert_eq!(graph_records.len(), 1, "premise: the graph path really recorded a position");
        assert_eq!(
            read(&hist),
            vec![0u64; K_CLUSTER_HISTOGRAM_BUCKETS],
            "R250: the graph record path must leave every K bucket at zero — there is no K on \
             that arm, and a count here would make the emitted absence a fabrication"
        );
    }
}
