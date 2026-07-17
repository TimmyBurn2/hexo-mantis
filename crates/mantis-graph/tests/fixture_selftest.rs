//! Fixture-checker mutation self-tests (LAW-07: prove the gate bites before
//! trusting it). BINDING: every test calls the IDENTICAL public `common::`
//! helpers graph_parity.rs uses (manifest parser, F-row verifier, blob
//! readers, fixture-root resolver), parametrized ONLY by the fixture-root
//! path — no parallel/re-implemented checker.

#[allow(dead_code)]
mod common;

use std::path::{Path, PathBuf};

/// Fresh scratch dir for a planted-corruption fixture copy.
fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("mantis_graph_selftest_{}_{name}", std::process::id()));
    if d.exists() {
        std::fs::remove_dir_all(&d).unwrap();
    }
    std::fs::create_dir_all(d.join("raw")).unwrap();
    d
}

/// Copy manifest.tsv + one raw blob from the REAL fixture root into `dst`;
/// returns the blob's manifest-relative path. Uses the first raw-subset case.
fn copy_fixture_subset(src_root: &Path, dst_root: &Path) -> (u32, String) {
    let m = common::parse_manifest(src_root).unwrap();
    let cid = m.header_ids("raw_subset").unwrap()[0];
    let rel = format!("raw/case_{cid:05}.bin");
    std::fs::copy(src_root.join("manifest.tsv"), dst_root.join("manifest.tsv")).unwrap();
    std::fs::copy(src_root.join(&rel), dst_root.join(&rel)).unwrap();
    (cid, rel)
}

#[test]
fn sha256_fips_vectors() {
    // FIPS 180-4 test vectors: empty, "abc", and the 448-bit vector.
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
    let (cid, rel) = copy_fixture_subset(&src, &tmp);

    // Plant: flip one byte inside the node_feat PAYLOAD.
    let pristine = common::read_blob(&tmp.join(&rel)).unwrap();
    let nf = pristine.fields.iter().find(|f| f.name == "node_feat").unwrap();
    let flip_at_payload = 3usize;
    let flip_at_file = nf.payload_offset + flip_at_payload;
    let mut bytes = std::fs::read(tmp.join(&rel)).unwrap();
    bytes[flip_at_file] ^= 0xFF;
    std::fs::write(tmp.join(&rel), &bytes).unwrap();

    // (a) the F-row verifier reports THAT file.
    let m = common::parse_manifest(&tmp).unwrap();
    let frow = m.files.iter().find(|f| f.path == rel).unwrap();
    let err = common::verify_file_row(&tmp, frow).unwrap_err();
    assert!(err.contains(&rel), "F-row error must name the file: {err}");

    // (b) the blob-vs-C-rows check reports THAT field.
    let corrupt = common::read_blob(&tmp.join(&rel)).unwrap();
    let err = common::check_blob_against_case_rows(&corrupt, &m.case_rows(cid)).unwrap_err();
    assert!(err.contains("node_feat") && err.contains("sha"), "field error must name field: {err}");

    // (c) first_diff_offset pinpoints the planted byte.
    let cf = corrupt.fields.iter().find(|f| f.name == "node_feat").unwrap();
    assert_eq!(
        common::first_diff_offset(&cf.payload, &nf.payload),
        Some(flip_at_payload),
        "offset diagnosis must locate the planted corruption"
    );
    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn truncated_fixture_detected() {
    let src = common::fixture_root();
    common::verify_fixture_root(&src).unwrap_or_else(|e| panic!("{e}"));
    let tmp = scratch("truncate");
    let (_cid, rel) = copy_fixture_subset(&src, &tmp);

    // Plant: cut the blob mid-payload (drop the last 10 bytes).
    let mut bytes = std::fs::read(tmp.join(&rel)).unwrap();
    let cut = bytes.len() - 10;
    bytes.truncate(cut);
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
    // Point the SAME fixture-root resolver the gate uses at an absent dir:
    // the result must be a loud error naming the path (FAIL semantics; the
    // gate tests panic on this error — they never skip).
    let tmp = scratch("absent");
    let missing = tmp.join("no_such_fixture_root");
    let err = common::verify_fixture_root(&missing).unwrap_err();
    assert!(
        err.contains("fixtures absent") && err.contains(missing.to_str().unwrap()),
        "absence error must name the path: {err}"
    );
    // Present dir but no manifest.tsv: also a named FAIL.
    let err = common::verify_fixture_root(&tmp).unwrap_err();
    assert!(
        err.contains("manifest") && err.contains(tmp.to_str().unwrap()),
        "manifest-absence error must name the path: {err}"
    );
    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn manifest_unknown_dtype_rejected() {
    let src = common::fixture_root();
    common::verify_fixture_root(&src).unwrap_or_else(|e| panic!("{e}"));
    let tmp = scratch("dtype");
    let text = std::fs::read_to_string(src.join("manifest.tsv")).unwrap();
    // Plant: first f32 C-row dtype becomes f64 (not a registered dtype).
    let planted = text.replacen("\tf32\t", "\tf64\t", 1);
    assert_ne!(planted, text, "plant must apply");
    std::fs::write(tmp.join("manifest.tsv"), planted).unwrap();
    let err = common::parse_manifest(&tmp).unwrap_err();
    assert!(err.contains("unknown dtype") && err.contains("f64"), "dtype error: {err}");
    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn manifest_shape_mismatch_detected() {
    let src = common::fixture_root();
    common::verify_fixture_root(&src).unwrap_or_else(|e| panic!("{e}"));
    let tmp = scratch("shape");
    let (cid, rel) = copy_fixture_subset(&src, &tmp);

    // Plant: swap the node_feat C-row dims (same nbytes, wrong shape).
    let blob = common::read_blob(&tmp.join(&rel)).unwrap();
    let nf = blob.fields.iter().find(|f| f.name == "node_feat").unwrap();
    let old_shape = format!("{}x{}", nf.dims[0], nf.dims[1]);
    let new_shape = format!("{}x{}", nf.dims[1], nf.dims[0]);
    let text = std::fs::read_to_string(tmp.join("manifest.tsv")).unwrap();
    let needle = format!("C\t{cid}\tnode_feat\tf32\t{old_shape}\t");
    let planted = text.replacen(&needle, &format!("C\t{cid}\tnode_feat\tf32\t{new_shape}\t"), 1);
    assert_ne!(planted, text, "plant must apply");
    std::fs::write(tmp.join("manifest.tsv"), planted).unwrap();

    let m = common::parse_manifest(&tmp).unwrap(); // still well-formed
    let err = common::check_blob_against_case_rows(&blob, &m.case_rows(cid)).unwrap_err();
    assert!(
        err.contains("shape mismatch") && err.contains("node_feat"),
        "shape edit must be reported, not silence: {err}"
    );
    std::fs::remove_dir_all(&tmp).ok();
}
