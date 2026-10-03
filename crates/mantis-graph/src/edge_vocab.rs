//! Every `edge_attr` row the builder can emit, by code: 0 the dummy's zeros, then axis × `±1..=±(w-1)` × player `-1, 0, +1`.

use crate::EDGE_FEAT_DIM;

/// The row an axis edge carries: the axis one-hot, the signed distance, the source node's player.
#[must_use]
pub fn edge_attr_row(axis_idx: usize, signed_dist: f32, src_player: f32) -> [f32; EDGE_FEAT_DIM] {
    let mut a = [0.0f32; EDGE_FEAT_DIM];
    a[axis_idx] = 1.0;
    a[3] = signed_dist;
    a[4] = src_player;
    a
}

/// The signed distances an axis walk of `win_length` reaches, in code order.
fn distances(win_length: u8) -> impl Iterator<Item = i32> {
    let reach = i32::from(win_length) - 1;
    (-reach..=-1).chain(1..=reach)
}

/// Every row in code order, or `None` when `win_length` leaves no axis edge or more codes than a byte holds.
#[must_use]
pub fn edge_vocabulary(win_length: u8) -> Option<Vec<[f32; EDGE_FEAT_DIM]>> {
    let size = 1 + 3 * 2 * (usize::from(win_length).checked_sub(1)?) * 3;
    if win_length < 2 || size > 256 {
        return None;
    }
    let mut rows = Vec::with_capacity(size);
    rows.push([0.0; EDGE_FEAT_DIM]);
    for axis in 0..3 {
        for d in distances(win_length) {
            for p in [-1.0, 0.0, 1.0] {
                rows.push(edge_attr_row(axis, d as f32, p));
            }
        }
    }
    Some(rows)
}

/// `row`'s index in `edge_vocabulary(win_length)`, or `None` for any row outside it (by value: -0.0 codes as 0.0).
#[must_use]
#[allow(clippy::float_cmp)] // the rows hold exact small integers the builder wrote; a tolerance would admit others
pub fn edge_code(row: &[f32], win_length: u8) -> Option<u8> {
    let &[a0, a1, a2, d, p] = row else {
        return None;
    };
    if [a0, a1, a2, d, p] == [0.0; EDGE_FEAT_DIM] {
        return Some(0);
    }
    let axis = match (a0, a1, a2) {
        (1.0, 0.0, 0.0) => 0,
        (0.0, 1.0, 0.0) => 1,
        (0.0, 0.0, 1.0) => 2,
        _ => return None,
    };
    let reach = i32::from(win_length) - 1;
    let di = distances(win_length).position(|v| v as f32 == d)?;
    let pi = [-1.0, 0.0, 1.0].iter().position(|&v| v == p)?;
    let code = 1 + (axis * 2 * usize::try_from(reach).ok()? + di) * 3 + pi;
    u8::try_from(code).ok()
}

#[cfg(test)]
#[path = "edge_vocab_tests.rs"]
mod tests;
