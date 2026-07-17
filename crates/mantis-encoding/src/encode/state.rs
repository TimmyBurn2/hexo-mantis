//! Dense state-tensor kernels (free fns; bodies byte-identical to the old
//! `impl Board` methods — only the signature, spec threading, and the
//! `board.ply.index() % 2` accessor differ).

use mantis_core::board::{Board, Cell, Player, BOARD_SIZE, TOTAL_CELLS};

use super::{MOVES_REMAINING_PLANE, MY_STONE_PLANE, OPP_STONE_PLANE, PLY_PARITY_PLANE};
use crate::spec::RegistrySpec;

/// Encode the 18-plane state tensor into `out` from a 2-plane cluster view.
///
/// Layout:
///   plane  0:    current player's stones (from `planes_2[0..TOTAL_CELLS]`)
///   planes 1-7:  zero (no Python history on the Rust self-play path)
///   plane  8:    opponent's stones (from `planes_2[TOTAL_CELLS..2*TOTAL_CELLS]`)
///   planes 9-15: zero (no Python history)
///   plane 16:    moves_remaining == 2 broadcast
///   plane 17:    ply parity broadcast
///
/// `out` must have length `18 * TOTAL_CELLS`. Callers zero-init the buffer before
/// calling; this function writes planes 0, 8, 16, 17 but leaves 1..7 / 9..15
/// untouched (the self-play path zero-inits the pooled buffers). Chain-length
/// planes are computed separately via `encode_chain_planes`.
pub fn encode_state_to_buffer(board: &Board, planes_2: &[f32], out: &mut [f32]) {
    // Plane 0: my stones
    out[..TOTAL_CELLS].copy_from_slice(&planes_2[..TOTAL_CELLS]);
    // Plane 8: opp stones
    for i in 0..TOTAL_CELLS {
        out[OPP_STONE_PLANE * TOTAL_CELLS + i] = planes_2[TOTAL_CELLS + i];
    }
    // Plane 16: moves_remaining == 2 ? 1.0 : 0.0
    let mr_val = if board.moves_remaining == 2 { 1.0 } else { 0.0 };
    for i in 0..TOTAL_CELLS {
        out[MOVES_REMAINING_PLANE * TOTAL_CELLS + i] = mr_val;
    }
    // Plane 17: ply % 2
    let ply_val = (board.ply.index() % 2) as f32;
    for i in 0..TOTAL_CELLS {
        out[PLY_PARITY_PLANE * TOTAL_CELLS + i] = ply_val;
    }
    debug_assert_eq!(
        out.len(),
        18 * TOTAL_CELLS,
        "encode_state_to_buffer output length mismatch — expected 18 planes × {TOTAL_CELLS} cells"
    );
}

/// Public alias for `encode_state_to_buffer`. Preserved as a named entry point
/// for callers outside this module.
#[inline]
pub fn encode_planes_to_buffer(board: &Board, planes_2: &[f32], out: &mut [f32]) {
    encode_state_to_buffer(board, planes_2, out);
}

/// Encode a *subset* of the 18 wire planes selected by `channels`, in the order
/// given. Used by sweep variants whose model in_channels < 18.
///
/// Plane semantics match `encode_state_to_buffer` (see header comment).
/// Channels 0/8 carry the only non-zero stone information on the Rust self-play
/// path; channels 16/17 are scalar broadcasts; 1–7 / 9–15 are zero on this path
/// (history filled by Python tensor assembly). `channels.iter().any(|c| c >= 18)`
/// panics in debug; release silently skips out-of-range entries.
///
/// `n_cells` is caller-supplied (= `spec.n_cells()` from the registry).
/// `planes_2.len()` must equal `2 * n_cells`; `out.len()` must equal
/// `channels.len() * n_cells`. The 19×19 single-window path passes `TOTAL_CELLS`
/// (361); the 25×25 window path passes `625`.
#[inline]
pub fn encode_state_to_buffer_channels(
    board: &Board,
    planes_2: &[f32],
    out: &mut [f32],
    channels: &[usize],
    n_cells: usize,
) {
    let n = channels.len();
    debug_assert_eq!(
        out.len(),
        n * n_cells,
        "encode_state_to_buffer_channels output length mismatch — \
         expected {n} planes × {n_cells} cells"
    );
    let mr_val = if board.moves_remaining == 2 { 1.0 } else { 0.0 };
    let ply_val = (board.ply.index() % 2) as f32;
    for (slot, &ch) in channels.iter().enumerate() {
        let dst = &mut out[slot * n_cells..(slot + 1) * n_cells];
        // Address planes by the named plane consts, not bare literals — the
        // history-plane pretrain↔selfplay drift guard.
        match ch {
            MY_STONE_PLANE => {
                dst.copy_from_slice(&planes_2[0..n_cells]);
            }
            OPP_STONE_PLANE => {
                dst.copy_from_slice(&planes_2[n_cells..2 * n_cells]);
            }
            MOVES_REMAINING_PLANE => {
                for v in dst.iter_mut() {
                    *v = mr_val;
                }
            }
            PLY_PARITY_PLANE => {
                for v in dst.iter_mut() {
                    *v = ply_val;
                }
            }
            c if c < 18 => {
                // History planes 1..7 / 9..15 are zero on the Rust self-play
                // path; clear in case the caller did not zero-init.
                for v in dst.iter_mut() {
                    *v = 0.0;
                }
            }
            _ => {
                debug_assert!(false, "channel index {ch} out of range [0, 18)");
                for v in dst.iter_mut() {
                    *v = 0.0;
                }
            }
        }
    }
}

/// `to_planes` variant emitting only the listed channels, in the listed order.
/// Length = `channels.len() * board_size²`. See `encode_state_to_buffer_channels`
/// for plane semantics.
///
/// Multi-window guard: panics if `spec.is_multi_window`. Multi-window self-play
/// is α-deferred; use `get_cluster_views()` for those encodings instead.
pub fn to_planes_channels(board: &Board, spec: &RegistrySpec, channels: &[usize]) -> Vec<f32> {
    if spec.is_multi_window {
        unimplemented!("multi-window selfplay deferred to alpha; route via get_cluster_views()");
    }
    let mut planes_2 = vec![0.0f32; 2 * TOTAL_CELLS];
    let (my_cell, opp_cell) = match board.current_player {
        Player::One => (Cell::P1, Cell::P2),
        Player::Two => (Cell::P2, Cell::P1),
    };
    for (&(q, r), &cell) in board.cells_iter() {
        let flat = board.window_flat_idx(q, r);
        if flat < TOTAL_CELLS {
            if cell == my_cell {
                planes_2[flat] = 1.0;
            } else if cell == opp_cell {
                planes_2[TOTAL_CELLS + flat] = 1.0;
            }
        }
    }
    let mut out = vec![0.0f32; channels.len() * TOTAL_CELLS];
    // This path is single-window only (the guard above bails for multi-window);
    // single-window uses the 19×19 wire layout (TOTAL_CELLS).
    encode_state_to_buffer_channels(board, &planes_2, &mut out, channels, TOTAL_CELLS);
    out
}

/// Encode the board as a flat f32 array of length `18 * board_size²`
/// representing shape `[18, board_size, board_size]` (18 history+scalar planes).
///
/// `board_size` comes from `spec.board_size`. Chain-length planes are computed
/// separately via `encode_chain_planes`. Stones outside the current 19×19 window
/// are silently omitted.
///
/// Multi-window guard: panics if `spec.is_multi_window`. Single-window self-play
/// must route through `get_cluster_views` for those encodings; the silent shape
/// corruption `to_planes` used to produce was the plane-export blocker.
/// Multi-window self-play is α-deferred.
pub fn to_planes(board: &Board, spec: &RegistrySpec) -> Vec<f32> {
    if spec.is_multi_window {
        unimplemented!("multi-window selfplay deferred to alpha; route via get_cluster_views()");
    }
    let board_size = spec.board_size;
    let total_cells = board_size * board_size;

    let mut planes_2 = vec![0.0f32; 2 * TOTAL_CELLS];
    let (my_cell, opp_cell) = match board.current_player {
        Player::One => (Cell::P1, Cell::P2),
        Player::Two => (Cell::P2, Cell::P1),
    };
    for (&(q, r), &cell) in board.cells_iter() {
        let flat = board.window_flat_idx(q, r);
        if flat < TOTAL_CELLS {
            if cell == my_cell {
                planes_2[flat] = 1.0;
            } else if cell == opp_cell {
                planes_2[TOTAL_CELLS + flat] = 1.0;
            }
        }
    }

    // Output buffer sized by the encoding's board_size. board_size == 19 is
    // bit-identical to the canonical single-window path.
    let mut out = vec![0.0f32; 18 * total_cells];
    if board_size == BOARD_SIZE {
        encode_state_to_buffer(board, &planes_2, &mut out);
    } else {
        // The board_size != 19 single-window fallback: emit the 18-plane wire
        // layout into the top-left 361 cells of each larger plane, then
        // broadcast the two scalar planes over the full plane. Unreachable for
        // the registered set (no registered single-window grid encoding has
        // board_size != 19; the multi-window survivors panic first) — kept
        // byte-verbatim and diff-gated.
        for i in 0..TOTAL_CELLS {
            out[MY_STONE_PLANE * total_cells + i] = planes_2[i];
            out[OPP_STONE_PLANE * total_cells + i] = planes_2[TOTAL_CELLS + i];
        }
        // Plane 16: moves_remaining == 2 broadcast over full plane.
        let mr_val = if board.moves_remaining == 2 { 1.0 } else { 0.0 };
        for i in 0..total_cells {
            out[MOVES_REMAINING_PLANE * total_cells + i] = mr_val;
        }
        // Plane 17: ply parity broadcast over full plane.
        let ply_val = (board.ply.index() % 2) as f32;
        for i in 0..total_cells {
            out[PLY_PARITY_PLANE * total_cells + i] = ply_val;
        }
    }
    out
}
