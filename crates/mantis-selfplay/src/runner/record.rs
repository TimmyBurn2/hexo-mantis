//! Record phase — `record_position_graph_dispatch`, called BEFORE `apply_move` so the pre-move
//! `board.ply` is the row's `ply_index` (never reframed in ply-parity units).

use mantis_core::Board;
use mantis_search::LegalSetPolicy;

use crate::records::{self, TargetIntegrityError};
use crate::replay::hexg::GraphRecord;

/// Push ONE whole-board graph record for this decision, carrying `root_value`, its search's (or proof's) root value.
///
/// # Errors
/// Forwards `record_position_graph`'s typed [`TargetIntegrityError`] to the caller, which
/// latches it run-fatal — the record that would carry a degenerate target cannot be built.
#[allow(clippy::too_many_arguments)] // the row's inputs: the target, its support and the search's root value
pub(crate) fn record_position_graph_dispatch(
    board: &Board,
    target_policy: &LegalSetPolicy,
    trunk_sz: i32,
    move_is_full_search: bool,
    graph_records_vec: &mut Vec<GraphRecord>,
    visit_capacity: usize,
    explicit_support: Option<&fxhash::FxHashSet<(i32, i32)>>,
    root_value: f32,
) -> Result<(), TargetIntegrityError> {
    let current_player = board.current_player as i8;
    let moves_remaining = board.moves_remaining;
    let ply_index = board.ply.index() as u16;
    let mut rec = records::record_position_graph(
        board,
        target_policy,
        trunk_sz,
        current_player,
        moves_remaining,
        ply_index,
        move_is_full_search,
        visit_capacity,
        explicit_support,
    )?;
    rec.root_value = root_value;
    rec.root_value_valid = true;
    graph_records_vec.push(rec);
    Ok(())
}
