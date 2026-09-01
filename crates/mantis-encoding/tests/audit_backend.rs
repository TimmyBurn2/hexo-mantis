//! O-12 — audit backend unit layer (through the public API). Census row shape +
//! cross-table INV-1..6 verdicts + severity → exit-code mapping. The e2e audit
//! (real .pt/.npz on disk) waits for WP7's bridge + CLI.

use mantis_encoding::audit::{census, cross_table, exit_code, CkptEntry, CorpusEntry, Severity};

fn ck(name: Option<&str>, sha: Option<&str>, has_meta: bool) -> CkptEntry {
    CkptEntry {
        display: "m.pt".to_string(),
        name: name.map(str::to_string),
        corpus_sha: sha.map(str::to_string),
        has_meta,
    }
}
fn co(name: Option<&str>, sha: Option<&str>) -> CorpusEntry {
    CorpusEntry {
        display: "c.npz".to_string(),
        name: name.map(str::to_string),
        sha: sha.map(str::to_string),
    }
}

#[test]
fn census_rows_shape() {
    let rows = census();
    assert_eq!(rows.len(), 5);
    let names: Vec<&str> = rows.iter().map(|r| r[0].as_str()).collect();
    assert_eq!(names, ["gnn_axis_r8", "gnn_axis_v1", "v6", "v6_live2_ls", "v6w25"]);
    // v6 row: name, board_size, n_planes, policy_logits, multi_window, schema_v
    let v6 = rows.iter().find(|r| r[0] == "v6").unwrap();
    assert_eq!(v6[1], "19");
    assert_eq!(v6[2], "8");
    assert_eq!(v6[3], "362");
    assert_eq!(v6[4], "false");
    assert_eq!(v6[5], "3");
}

#[test]
fn inv5_ok_info_exit0() {
    let f = cross_table(&[ck(Some("v6"), Some("aa"), true)], &[co(Some("v6"), Some("aa"))], false);
    assert_eq!(f.iter().map(|x| x.severity).collect::<Vec<_>>(), [Severity::Info]);
    assert_eq!(exit_code(&f), 0);
}

#[test]
fn inv1_enc_mismatch_error_exit2() {
    let f = cross_table(&[ck(Some("v6"), Some("aa"), true)], &[co(Some("v6w25"), Some("aa"))], false);
    assert!(f.iter().any(|x| x.severity == Severity::Error));
    assert_eq!(exit_code(&f), 2);
}

#[test]
fn inv2_orphan_sha_error() {
    let f = cross_table(&[ck(Some("v6"), Some("dead"), true)], &[co(Some("v6"), Some("aa"))], false);
    assert!(f.iter().any(|x| x.severity == Severity::Error && x.message.contains("matches no corpus")));
    assert_eq!(exit_code(&f), 2);
}

#[test]
fn inv3_no_corpus_sha_warn() {
    let f = cross_table(&[ck(Some("v6"), None, true)], &[], false);
    assert_eq!(f.iter().map(|x| x.severity).collect::<Vec<_>>(), [Severity::Warn]);
    assert_eq!(exit_code(&f), 1);
}

#[test]
fn inv4_no_meta_warn() {
    let f = cross_table(&[ck(None, None, false)], &[], false);
    assert_eq!(f.iter().map(|x| x.severity).collect::<Vec<_>>(), [Severity::Warn]);
    assert!(f[0].message.contains("no metadata"));
}

#[test]
fn inv6_orphan_corpus_info_lax_warn_strict() {
    let co = [co(Some("v6"), Some("aa"))];
    assert_eq!(exit_code(&cross_table(&[], &co, false)), 0); // info in lax
    assert_eq!(exit_code(&cross_table(&[], &co, true)), 1); // warn in strict
}
