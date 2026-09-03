//! Q13 chain-length plane encoding (already free functions on the old side;
//! ported verbatim, only the `HEX_AXES` import path changes).
//!
//! Output layout: 6 planes × `n_cells` f32 entries (caller-supplied), written
//! into the caller's buffer slice at offset 0. Values are /6.0 normalized.
//! Plane order: [axis0_cur, axis0_opp, axis1_cur, axis1_opp, axis2_cur, axis2_opp].

use mantis_core::board::HEX_AXES;

/// Saturation cap for chain length — the 6-in-a-row win target.
// AUDIT-1 F-42. The cap IS the win length, and `mantis_core::board::WIN_LENGTH` owns it;
// a second 6 here goes stale the day the rule changes.
const CHAIN_CAP: i32 = mantis_core::board::WIN_LENGTH as i32;
/// Normalisation denominator; derived from `CHAIN_CAP` (one const, not two) so
/// the cap and its normalizer cannot drift apart.
const CHAIN_NORM: f32 = CHAIN_CAP as f32;

#[inline]
fn flat_idx(q: i32, r: i32, trunk_sz: i32, half: i32) -> usize {
    ((q + half) as usize) * (trunk_sz as usize) + (r + half) as usize
}

#[inline]
fn in_window(q: i32, r: i32, half: i32) -> bool {
    q >= -half && q <= half && r >= -half && r <= half
}

/// Walk +step * (dq, dr) from (q, r) counting consecutive `own` cells. Stops at
/// window edge or first non-own cell. Max count = `CHAIN_CAP - 1`.
#[inline]
fn count_run(own: &[f32], q: i32, r: i32, dq: i32, dr: i32, trunk_sz: i32, half: i32) -> i32 {
    let mut c = 0i32;
    for k in 1..CHAIN_CAP {
        let qk = q + dq * k;
        let rk = r + dr * k;
        if !in_window(qk, rk, half) {
            break;
        }
        let idx = flat_idx(qk, rk, trunk_sz, half);
        if own[idx] > 0.5 {
            c += 1;
        } else {
            break;
        }
    }
    c
}

/// Write one chain-length plane (single axis, single player) into `out`.
/// `out` must have length `n_cells` (= trunk_sz²).
#[inline]
fn encode_chain_plane_one(
    own: &[f32],
    opp: &[f32],
    dq: i32,
    dr: i32,
    out: &mut [f32],
    trunk_sz: i32,
    half: i32,
) {
    for q in -half..=half {
        for r in -half..=half {
            let idx = flat_idx(q, r, trunk_sz, half);
            if opp[idx] > 0.5 {
                out[idx] = 0.0;
                continue;
            }
            let pos_run = count_run(own, q, r, dq, dr, trunk_sz, half);
            let neg_run = count_run(own, q, r, -dq, -dr, trunk_sz, half);
            let is_own = own[idx] > 0.5;
            if !is_own && pos_run == 0 && neg_run == 0 {
                out[idx] = 0.0;
                continue;
            }
            let mut v = 1 + pos_run + neg_run;
            if v > CHAIN_CAP {
                v = CHAIN_CAP;
            }
            out[idx] = (v as f32) / CHAIN_NORM;
        }
    }
}

/// Write all 6 chain-length planes into `out` (length `6 * n_cells`).
///
/// `cur_mask` and `opp_mask` are `n_cells`-sized f32 masks with 1.0 at stone
/// positions and 0.0 elsewhere. `n_cells` and `trunk_sz` are caller-supplied;
/// `n_cells` must equal `trunk_sz * trunk_sz`.
#[inline]
pub fn encode_chain_planes(
    cur_mask: &[f32],
    opp_mask: &[f32],
    out: &mut [f32],
    n_cells: usize,
    trunk_sz: i32,
) {
    let half = (trunk_sz - 1) / 2;
    debug_assert_eq!(n_cells, (trunk_sz as usize) * (trunk_sz as usize));
    debug_assert_eq!(cur_mask.len(), n_cells);
    debug_assert_eq!(opp_mask.len(), n_cells);
    debug_assert_eq!(out.len(), 6 * n_cells);

    for (axis_idx, &(dq, dr)) in HEX_AXES.iter().enumerate() {
        let cur_base = 2 * axis_idx * n_cells;
        let opp_base = (2 * axis_idx + 1) * n_cells;
        // Split into two mutable slices so we can borrow disjoint regions.
        let (head, tail) = out.split_at_mut(opp_base);
        encode_chain_plane_one(
            cur_mask,
            opp_mask,
            dq,
            dr,
            &mut head[cur_base..cur_base + n_cells],
            trunk_sz,
            half,
        );
        encode_chain_plane_one(
            opp_mask,
            cur_mask,
            dq,
            dr,
            &mut tail[0..n_cells],
            trunk_sz,
            half,
        );
    }
}
