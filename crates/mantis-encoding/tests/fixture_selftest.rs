//! O-3 self-test — fixture-checker mutation self-tests (LAW-07: prove the gate
//! bites before trusting it). BINDING: every test calls the IDENTICAL public
//! `common::` helpers encode_parity.rs uses (manifest parser, F-row verifier,
//! blob reader, output builder, fixture-root resolver) — no parallel checker.

#[allow(dead_code)]
mod common;

use std::path::{Path, PathBuf};

use common::CLASS_PANIC;

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("mantis_encode_selftest_{}_{name}", std::process::id()));
    if d.exists() {
        std::fs::remove_dir_all(&d).unwrap();
    }
    std::fs::create_dir_all(d.join("raw")).unwrap();
    d
}

/// Copy manifest.tsv + the first byte-case raw blob into `dst`; return (id, rel).
fn copy_first_byte_case(src: &Path, dst: &Path) -> (u32, String) {
    let m = common::parse_manifest(src).unwrap();
    let cid = m.cases[0].case_id; // C-rows exist only for byte cases
    let rel = format!("raw/case_{cid:05}.bin");
    std::fs::copy(src.join("manifest.tsv"), dst.join("manifest.tsv")).unwrap();
    std::fs::copy(src.join(&rel), dst.join(&rel)).unwrap();
    (cid, rel)
}

#[test]
fn sha256_fips_vectors() {
    assert_eq!(
        common::sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        common::sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        common::sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
}

#[test]
fn corrupt_one_byte_detected() {
    let src = common::fixture_root();
    common::verify_fixture_root(&src).unwrap_or_else(|e| panic!("{e}"));
    let tmp = scratch("corrupt");
    let (_cid, rel) = copy_first_byte_case(&src, &tmp);

    // Rebuild the golden output (the SAME build the parity gate does), then plant
    // a flipped byte inside the blob's output payload.
    let pristine = common::read_blob(&tmp.join(&rel)).unwrap();
    let rebuilt = common::build_output(&pristine.input);
    let flip_at_payload = 7usize;
    let flip_at_file = pristine.payload_offset + flip_at_payload;
    let mut bytes = std::fs::read(tmp.join(&rel)).unwrap();
    bytes[flip_at_file] ^= 0xFF;
    std::fs::write(tmp.join(&rel), &bytes).unwrap();

    // (a) the F-row verifier reports THAT file (sha drift).
    let m = common::parse_manifest(&tmp).unwrap();
    let frow = m.files.iter().find(|f| f.path == rel).unwrap();
    let err = common::verify_file_row(&tmp, frow).unwrap_err();
    assert!(err.contains(&rel), "F-row error must name the file: {err}");

    // (b) the rebuilt (correct) output vs the corrupted golden → offset pinned.
    let corrupt = common::read_blob(&tmp.join(&rel)).unwrap();
    let off = common::first_diff_offset(&rebuilt, corrupt.payload.as_ref().unwrap());
    assert_eq!(off, Some(flip_at_payload), "offset diagnosis must locate the planted corruption");
    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn truncated_fixture_detected() {
    let src = common::fixture_root();
    common::verify_fixture_root(&src).unwrap_or_else(|e| panic!("{e}"));
    let tmp = scratch("truncate");
    let (_cid, rel) = copy_first_byte_case(&src, &tmp);

    let mut bytes = std::fs::read(tmp.join(&rel)).unwrap();
    bytes.truncate(bytes.len() - 10);
    std::fs::write(tmp.join(&rel), &bytes).unwrap();

    let err = common::read_blob(&tmp.join(&rel)).unwrap_err();
    assert!(
        err.contains(&rel) && err.contains("truncated"),
        "truncation must be a loud error naming the file: {err}"
    );
    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn absent_fixtures_fail_not_skip() {
    let tmp = scratch("absent");
    let missing = tmp.join("no_such_fixture_root");
    let err = common::verify_fixture_root(&missing).unwrap_err();
    assert!(
        err.contains("fixtures absent") && err.contains(missing.to_str().unwrap()),
        "absence error must name the path: {err}"
    );
    let err = common::verify_fixture_root(&tmp).unwrap_err();
    assert!(err.contains("manifest") && err.contains(tmp.to_str().unwrap()), "{err}");
    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn manifest_unknown_dtype_rejected() {
    let src = common::fixture_root();
    common::verify_fixture_root(&src).unwrap_or_else(|e| panic!("{e}"));
    let tmp = scratch("dtype");
    let text = std::fs::read_to_string(src.join("manifest.tsv")).unwrap();
    let planted = text.replacen("\tf32\t", "\tf64\t", 1);
    assert_ne!(planted, text, "plant must apply");
    std::fs::write(tmp.join("manifest.tsv"), planted).unwrap();
    let err = common::parse_manifest(&tmp).unwrap_err();
    assert!(err.contains("unknown dtype") && err.contains("f64"), "dtype error: {err}");
    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn manifest_unknown_row_kind_rejected() {
    let src = common::fixture_root();
    common::verify_fixture_root(&src).unwrap_or_else(|e| panic!("{e}"));
    let tmp = scratch("rowkind");
    let mut text = std::fs::read_to_string(src.join("manifest.tsv")).unwrap();
    text.push_str("Z\tbogus\trow\n");
    std::fs::write(tmp.join("manifest.tsv"), text).unwrap();
    let err = common::parse_manifest(&tmp).unwrap_err();
    assert!(err.contains("unknown row kind") && err.contains('Z'), "row-kind error: {err}");
    std::fs::remove_dir_all(&tmp).ok();
}

/// Sanity: the fixture set really does contain the panic_cases the parity gate
/// asserts on (belt-and-braces that the header is not empty by accident).
#[test]
fn panic_cases_present_in_fixtures() {
    let src = common::fixture_root();
    common::verify_fixture_root(&src).unwrap_or_else(|e| panic!("{e}"));
    let m = common::parse_manifest(&src).unwrap();
    let panic_ids = m.header_ids("panic_cases").unwrap();
    assert!(!panic_ids.is_empty(), "panic_cases header must list the multi-window to_planes cases");
    let cases = common::read_inputs_bin(&src.join("inputs.bin")).unwrap();
    for id in &panic_ids {
        let c = cases.iter().find(|c| c.case_id == *id).unwrap();
        assert_eq!(c.class, CLASS_PANIC, "panic_case {id} must carry the panic class");
    }
}
