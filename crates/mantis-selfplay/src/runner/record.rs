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
use crate::replay::hexg::{GraphRecord, MAX_VISITS};
use crate::replay::sym::SymTables;

use super::rotate::{rotate_chain_inplace, rotate_policy_inplace, rotate_state_inplace};
use super::search_drive::MovePolicy;

/// Per-position record tuple pushed into `records_vec` (frozen `inner.rs:123`).
/// Field order: `(feat, chain, projected_policy, player, center_q, center_r,
/// is_full_search, ply_index)`. `ply_index` (CF-4) is the 0-based pre-move ply,
/// shared across the K cluster rows of one decision.
pub(crate) type RecordTuple = (Vec<f32>, Vec<f32>, Vec<f32>, Player, i32, i32, bool, u16);

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
) {
    let (views, centers) = board.get_cluster_views();
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
        MAX_VISITS,
    )?;
    graph_records_vec.push(rec);
    Ok(())
}
