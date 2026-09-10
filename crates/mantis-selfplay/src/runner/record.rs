//! Record phase (WP6 D14) — `record_position_graph_dispatch` (frozen
//! `worker_loop/inner.rs:1590`, dispatch branch `:1398`), called BEFORE `apply_move` so the
//! pre-move `board.ply` is the row's `ply_index` (LAW-03 measurement-unit: never reframed in
//! ply-parity units). The dense K-cluster recorder and the in-run K histogram beside it went
//! with the grid path (R346(f)).

use mantis_core::Board;

use crate::records::{self, TargetIntegrityError};
use crate::replay::hexg::GraphRecord;

use super::search_drive::MovePolicy;

/// Push ONE whole-board graph record for this decision.
///
/// # Errors
/// WP12-R Phase T (DESIGN_T §3.3/§3.4): forwards `record_position_graph`'s
/// typed [`TargetIntegrityError`] to the caller, which latches it run-fatal
/// (LAW-14) — the record that would carry a degenerate target cannot be built.
// `#[cold]`/`#[inline(never)]` are DELETED with the dense recorder they were paired against.
pub(crate) fn record_position_graph_dispatch(
    board: &Board,
    target_policy: &MovePolicy,
    trunk_sz: i32,
    move_is_full_search: bool,
    graph_records_vec: &mut Vec<GraphRecord>,
    visit_capacity: usize,
    explicit_support: Option<&fxhash::FxHashSet<(i32, i32)>>,
) -> Result<(), TargetIntegrityError> {
    let ls = match target_policy {
        MovePolicy::Ls(ls) => ls,
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
        explicit_support,
    )?;
    graph_records_vec.push(rec);
    Ok(())
}
