//! Rotation helpers (WP6 D-seam-3) — the per-game D6 rotation wrappers
//! (`inv_sym_idx` + the in-place `rotate_state/chain/policy/aux` scatters), ported
//! from the frozen `worker_loop/rotate.rs:54/68/80/92/115` as thin wrappers over
//! the WP5 `replay/sym.rs` tables (`SymTables` / `apply_symmetry_state` /
//! `apply_chain_symmetry`, single-source D6).
//!
//! The frozen `compute_move_temperature` copy is KILLED (D15 — the driver uses
//! `mantis_search::{compute_move_temperature, ply_to_compound_move}`); only the
//! four scatter wrappers + `inv_sym_idx` port here. Each helper allocates a
//! temporary buffer and `mem::swap`s it in (cheap relative to inference);
//! `sym_idx == 0` is the identity scatter and the callers short-circuit it.

use crate::replay::sample::{apply_chain_symmetry, apply_symmetry_state};
use crate::replay::sym::SymTables;

/// Inverse of dihedral element `s` parameterized as reflect-then-rotate^n
/// (frozen `rotate.rs:54`). Pure rotations (`s ∈ 0..6`): inverse is rotation by
/// `(6 - s) % 6`. Reflective elements (`s ∈ 6..12`): self-inverse.
#[inline]
pub(crate) fn inv_sym_idx(s: usize) -> usize {
    if s < 6 {
        (6 - s) % 6
    } else {
        s
    }
}

/// Forward-scatter a state buffer in place under `sym_idx` (frozen `rotate.rs:68`).
/// Plane-count-generic — `apply_symmetry_state` deduces `n_planes` from
/// `buf.len() / n_cells`.
#[inline]
pub(crate) fn rotate_state_inplace(buf: &mut Vec<f32>, sym_idx: usize, tables: &SymTables) {
    let mut tmp = vec![0.0f32; buf.len()];
    apply_symmetry_state::<f32>(buf, &mut tmp, sym_idx, tables);
    std::mem::swap(buf, &mut tmp);
}

/// Forward-scatter a 6-plane chain buffer in place under `sym_idx` (frozen
/// `rotate.rs:80`). Includes the axis-plane remap (chain planes encode
/// hex-axis-specific data).
#[inline]
pub(crate) fn rotate_chain_inplace(buf: &mut Vec<f32>, sym_idx: usize, tables: &SymTables) {
    let mut tmp = vec![0.0f32; buf.len()];
    apply_chain_symmetry::<f32>(buf, &mut tmp, sym_idx, tables);
    std::mem::swap(buf, &mut tmp);
}

/// Forward-scatter a single policy buffer in place (frozen `rotate.rs:92`). The
/// pass-action slot (at index `n_cells`) is a global identity — it stays put.
#[inline]
pub(crate) fn rotate_policy_inplace(
    buf: &mut Vec<f32>,
    sym_idx: usize,
    tables: &SymTables,
    n_cells: usize,
) {
    let mut tmp = vec![0.0f32; buf.len()];
    let scatter = &tables.scatter[sym_idx];
    for &(sc, dc) in scatter {
        tmp[dc as usize] = buf[sc as usize];
    }
    if buf.len() > n_cells {
        tmp[n_cells] = buf[n_cells];
    }
    std::mem::swap(buf, &mut tmp);
}

/// Forward-scatter the combined `aux_u8` buffer (ownership ‖ winning_line) in
/// place (frozen `rotate.rs:115`). Ownership default is 1 (empty); winning_line
/// default is 0 (no win mask).
#[inline]
pub(crate) fn rotate_aux_inplace(
    buf: &mut Vec<u8>,
    sym_idx: usize,
    tables: &SymTables,
    n_cells: usize,
) {
    let mut tmp = vec![0u8; buf.len()];
    tmp[..n_cells].fill(1); // ownership default = empty
    let scatter = &tables.scatter[sym_idx];
    for &(sc, dc) in scatter {
        tmp[dc as usize] = buf[sc as usize];
        tmp[n_cells + dc as usize] = buf[n_cells + sc as usize];
    }
    std::mem::swap(buf, &mut tmp);
}

// ── P-06 in-src parity pins: the pub(crate) rotation wrappers the integration test
//    (`tests/rotation_parity.rs`) cannot reach ──────────────────────────────────
#[cfg(test)]
mod parity_tests {
    use super::{inv_sym_idx, rotate_aux_inplace, rotate_policy_inplace, rotate_state_inplace};
    use crate::replay::sym::{SymTables, N_SYMS};

    fn flat(q: i32, r: i32, bs: i32, half: i32) -> usize {
        ((q + half) * bs + (r + half)) as usize
    }

    /// `inv_sym_idx` is the dihedral group inverse: pure rotations invert to
    /// `(6 - s) % 6`, reflective elements are self-inverse.
    #[test]
    fn inv_sym_idx_is_the_group_inverse() {
        assert_eq!(inv_sym_idx(0), 0);
        assert_eq!(inv_sym_idx(1), 5);
        assert_eq!(inv_sym_idx(2), 4);
        assert_eq!(inv_sym_idx(3), 3);
        assert_eq!(inv_sym_idx(4), 2);
        assert_eq!(inv_sym_idx(5), 1);
        for s in 6..N_SYMS {
            assert_eq!(inv_sym_idx(s), s, "reflective elements are self-inverse");
        }
    }

    /// Site-1 forward (`rotate_state_inplace`) then site-2 inverse
    /// (`rotate_policy_inplace` with `inv_sym_idx`) restores the interior — the MCTS
    /// tree always operates in the canonical frame. f32 bits compared for exactness.
    #[test]
    fn state_forward_then_policy_inverse_restores_interior() {
        let t = SymTables::new();
        let bs = t.board_size as i32;
        let half = (bs - 1) / 2;
        let n = t.n_cells;
        let coords: Vec<(i32, i32)> = (-3..=3)
            .flat_map(|q| (-3..=3).map(move |r| (q, r)))
            .collect();
        let mut base = vec![0.0f32; n];
        for (k, &(q, r)) in coords.iter().enumerate() {
            base[flat(q, r, bs, half)] = (k as f32) + 1.0;
        }
        for s in 1..N_SYMS {
            let mut buf = base.clone();
            rotate_state_inplace(&mut buf, s, &t);
            rotate_policy_inplace(&mut buf, inv_sym_idx(s), &t, n);
            for &(q, r) in &coords {
                let i = flat(q, r, bs, half);
                assert_eq!(
                    buf[i].to_bits(),
                    base[i].to_bits(),
                    "sym {s}: forward∘inverse must restore interior cell ({q},{r})",
                );
            }
        }
    }

    /// `rotate_aux_inplace` fills the ownership plane default = 1 (empty) and keeps
    /// the window centre (a fixed point of every rotation) verbatim across planes.
    #[test]
    fn aux_rotation_defaults_ownership_and_fixes_center() {
        let t = SymTables::new();
        let bs = t.board_size as i32;
        let half = (bs - 1) / 2;
        let n = t.n_cells;
        let center = flat(0, 0, bs, half);
        let mut aux = vec![0u8; 2 * n];
        aux[..n].fill(1);
        aux[center] = 2; // P1 ownership sentinel at the centre
        aux[n + center] = 1; // winning-line mask at the centre
        rotate_aux_inplace(&mut aux, 1, &t, n);
        assert_eq!(aux[center], 2, "centre ownership preserved (fixed point)");
        assert_eq!(aux[n + center], 1, "centre winning-line mask preserved");
        let edge = flat(0, 8, bs, half);
        assert_eq!(aux[edge], 1, "ownership default is 1 (empty), never 0");
    }
}
