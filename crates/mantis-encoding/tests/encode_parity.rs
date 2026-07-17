//! O-3 — full encode-kernel byte parity across the surviving DENSE encodings
//! {v6, v6w25, v6_live2_ls}. Every case is rebuilt through the ported FREE-FN
//! kernels and byte-compared (raw bytes + sha256 over canonical LE bytes) against
//! the ⊕ goldens captured from the FROZEN old kernels (capture-first, mtime-proven).
//! `panic_case`s assert the free fn panics `unimplemented!`; the out-of-range
//! channel case asserts the debug `debug_assert!` fires. Zero tolerance.

#[allow(dead_code)]
mod common;

use std::panic;
use std::path::Path;

use common::{
    build_output, first_diff_offset, parse_manifest, read_blob, read_inputs_bin, sha256_hex,
    verify_file_row, verify_fixture_root, CaseInput, CLASS_OOR_SKIP, CLASS_PANIC,
};
use mantis_core::Board;
use mantis_encoding::{
    encode_state_to_buffer, encode_state_to_buffer_channels, lookup_or_panic, MOVES_REMAINING_PLANE,
    MY_STONE_PLANE, OPP_STONE_PLANE, PLY_PARITY_PLANE,
};

/// Run `build_output` catching the expected panic; returns true iff it panicked.
fn panicked(c: &CaseInput) -> bool {
    let prev = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let r = panic::catch_unwind(panic::AssertUnwindSafe(|| build_output(c)));
    panic::set_hook(prev);
    r.is_err()
}

#[test]
fn encode_kernel_byte_parity() {
    let root = common::fixture_root();
    verify_fixture_root(&root).unwrap_or_else(|e| panic!("{e}"));
    let manifest = parse_manifest(&root).unwrap_or_else(|e| panic!("{e}"));

    // (1) Every fixture FILE verified before use (LAW-07: corrupt byte → loud).
    assert!(!manifest.files.is_empty(), "manifest has no F-rows");
    for f in &manifest.files {
        verify_file_row(&root, f).unwrap_or_else(|e| panic!("F-row: {e}"));
    }

    // (2) No ARCH/KILL encoding may appear (O-16 belt-and-braces).
    assert_eq!(manifest.header_value("encodings"), Some("v6,v6w25,v6_live2_ls"));
    for row in &manifest.cases {
        assert!(
            matches!(row.encoding.as_str(), "v6" | "v6w25" | "v6_live2_ls"),
            "forbidden encoding {:?} in a case row",
            row.encoding
        );
    }

    let cases = read_inputs_bin(&root.join("inputs.bin")).unwrap_or_else(|e| panic!("{e}"));
    let panic_ids = manifest.header_ids("panic_cases").unwrap();

    let mut byte_checked = 0usize;
    let mut panic_checked = 0usize;
    let mut oor_checked = 0usize;

    for c in &cases {
        if c.class == CLASS_PANIC {
            assert!(panic_ids.contains(&c.case_id), "case {} not in panic_cases header", c.case_id);
            assert!(panicked(c), "panic_case {} did NOT panic (multi-window guard lost)", c.case_id);
            // The blob carries no payload for a panic case.
            let blob = read_blob(&root.join(format!("raw/case_{:05}.bin", c.case_id))).unwrap();
            assert!(blob.payload.is_none(), "panic_case {} blob unexpectedly has payload", c.case_id);
            panic_checked += 1;
            continue;
        }

        if c.class == CLASS_OOR_SKIP {
            // Failure-mode fidelity: the ≥18 channel arm `debug_assert!`s. Default
            // `cargo test` is a debug build, so the kernel must panic here; the
            // captured release bytes (zero in that slot) are checked only under a
            // release build.
            if cfg!(debug_assertions) {
                assert!(panicked(c), "out-of-range channel case {} did not debug_assert", c.case_id);
            } else {
                let out = build_output(c);
                compare_case(&root, &manifest, c, &out);
            }
            oor_checked += 1;
            continue;
        }

        let out = build_output(c);
        compare_case(&root, &manifest, c, &out);
        byte_checked += 1;
    }

    assert_eq!(panic_checked, panic_ids.len(), "panic_case count mismatch");
    assert!(byte_checked >= 20, "too few byte cases checked ({byte_checked})");
    println!(
        "PARITY OK: {} cases ({byte_checked} byte, {panic_checked} panic, {oor_checked} oor), 0 mismatches",
        cases.len()
    );
}

/// Byte-compare (raw payload + sha256 over canonical LE bytes) one case against
/// its golden. Zero tolerance — any diff is a loud, offset-naming failure.
fn compare_case(root: &Path, manifest: &common::Manifest, c: &CaseInput, out: &[u8]) {
    let blob = read_blob(&root.join(format!("raw/case_{:05}.bin", c.case_id))).unwrap();
    let payload = blob
        .payload
        .as_ref()
        .unwrap_or_else(|| panic!("case {} has no golden payload", c.case_id));

    // Raw byte-for-byte against the committed golden.
    if let Some(off) = first_diff_offset(out, payload) {
        panic!(
            "case {} kernel {} enc {}: BYTE MISMATCH at offset {off} (new_len={}, golden_len={})",
            c.case_id,
            c.kernel,
            c.encoding_id,
            out.len(),
            payload.len()
        );
    }

    // sha256 over canonical LE bytes == manifest C-row (Python-hashlib-written).
    let row = manifest
        .case_row(c.case_id)
        .unwrap_or_else(|| panic!("case {} missing C-row", c.case_id));
    let got = sha256_hex(out);
    assert_eq!(got, row.sha256, "case {} sha256 drift vs manifest C-row", c.case_id);
    assert_eq!(out.len() as u64, row.nbytes, "case {} nbytes drift", c.case_id);
}

// ── relocated in-src kernel unit tests (moved out of encode.rs per repo_design §8) ──

/// Channel-select drift guard (re-targeted from the ARCHed v6_live2 to the
/// registered v6_live2_ls, which carries the IDENTICAL kept set [0,8,16,17]).
/// A synthetic decoupled state (mr_val=1, ply_val=0) gives a MOVES↔PLY swap
/// teeth. Exercises the relocated plane consts + the free-fn kernels.
#[test]
fn channel_select_matches_registry_kept_set_v6_live2_ls() {
    let spec = lookup_or_panic("v6_live2_ls");
    let kept = spec.kept_plane_indices;
    assert_eq!(
        kept,
        [MY_STONE_PLANE, OPP_STONE_PLANE, MOVES_REMAINING_PLANE, PLY_PARITY_PLANE].as_slice(),
        "v6_live2_ls kept_plane_indices drifted from the named semantic plane offsets"
    );

    const TOTAL_CELLS: usize = 361;
    let mut b = Board::new();
    b.apply_move(0, 0).unwrap();
    b.apply_move(1, 0).unwrap();
    b.apply_move(0, 1).unwrap();
    b.moves_remaining = 2; // mr_val = 1.0
    b.ply = mantis_core::Ply::new(4); // ply_val = 0.0 (decoupled from mr_val)

    let mut planes_2 = vec![0.0f32; 2 * TOTAL_CELLS];
    planes_2[0] = 1.0;
    planes_2[5] = 1.0;
    planes_2[TOTAL_CELLS + 7] = 1.0;
    planes_2[TOTAL_CELLS + 11] = 1.0;

    let mut full = vec![0.0f32; 18 * TOTAL_CELLS];
    encode_state_to_buffer(&b, &planes_2, &mut full);
    let mut sel = vec![0.0f32; kept.len() * TOTAL_CELLS];
    encode_state_to_buffer_channels(&b, &planes_2, &mut sel, kept, TOTAL_CELLS);
    for (slot, &ch) in kept.iter().enumerate() {
        let lhs = &full[ch * TOTAL_CELLS..(ch + 1) * TOTAL_CELLS];
        let rhs = &sel[slot * TOTAL_CELLS..(slot + 1) * TOTAL_CELLS];
        assert_eq!(lhs, rhs, "v6_live2_ls kept plane {ch} (slot {slot}) encoder drift");
    }
}
