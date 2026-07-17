//! O-11 — manifests.toml loud-parse + tamper. The embedded manifest exposes the
//! 4 registered corpus/anchor pins; a mutated copy (unknown key / missing
//! required / held-out∩corpus sha collision / malformed sha) is a loud named
//! error, never a silent accept.

use mantis_encoding::manifests::parse_manifests;
use mantis_encoding::{
    anchor_path, assert_not_heldout, corpus_path, corpus_sha_pin, held_out_shas,
};

const V6_LIVE2_LS_SHA: &str = "3813edc2fb10a7c5ab976a0293e38cbba0fd6b84e5295630f339ca421b345c97";
const S5_HELDOUT_SHA: &str = "88f99c2b5fea7495484e4e9cc1af831d1e053221dc7e0f9c8f5d3ab6f27aa69e";

// ── the embedded, validated manifest ─────────────────────────────────────────
#[test]
fn embedded_manifest_registered_pins() {
    // The 4 registered blocks present; ARCH/KILL blocks ABSENT.
    let mut names = mantis_encoding::manifests::pinned_encoding_names();
    names.sort_unstable();
    assert_eq!(names, ["gnn_axis_v1", "v6", "v6_live2_ls", "v6w25"]);
    for absent in ["v6_live2", "v7full", "v8", "v8_canvas_realness", "v7mw", "v6tp"] {
        assert!(corpus_path(absent).is_none(), "{absent} pin block must be ABSENT");
    }

    assert_eq!(corpus_path("v6"), Some("data/bootstrap_corpus.npz"));
    assert_eq!(anchor_path("v6"), Some("checkpoints/bootstrap_model_v6.pt"));

    // The v6_live2_ls anchor PATH is the file bootstrap_model_v6_live2.pt
    // (consumed AS v6_live2_ls, no-reshape — the §a.8 caution).
    assert_eq!(
        anchor_path("v6_live2_ls"),
        Some("checkpoints/bootstrap_model_v6_live2.pt"),
        "the anchor file is bootstrap_model_v6_live2.pt (consumed as v6_live2_ls)"
    );
    assert_eq!(corpus_sha_pin("v6_live2_ls"), Some(V6_LIVE2_LS_SHA), "the sole launch pin");
    assert_eq!(corpus_sha_pin("v6"), None, "v6 unpinned");
    assert_eq!(corpus_sha_pin("gnn_axis_v1"), None, "gnn deliberately unpinned");

    assert!(held_out_shas().contains(&S5_HELDOUT_SHA), "s5 held-out set present");
    assert_eq!(mantis_encoding::manifests::held_out_sizes(), vec![12_872_280]);

    assert!(assert_not_heldout(V6_LIVE2_LS_SHA).is_ok(), "a corpus pin is not held-out");
    let err = assert_not_heldout(S5_HELDOUT_SHA).unwrap_err();
    assert!(err.contains("held-out"), "a held-out sha must be rejected from a load: {err}");
}

// ── tamper discipline ─────────────────────────────────────────────────────────
fn valid_manifest() -> String {
    format!(
        r#"
schema_version = 1

[encodings.v6]
corpus_path = "data/c.npz"
anchor_path = "checkpoints/a.pt"

[encodings.v6_live2_ls]
corpus_path = "data/ls.npz"
anchor_path = "checkpoints/bootstrap_model_v6_live2.pt"
corpus_sha256 = "{V6_LIVE2_LS_SHA}"

[[held_out]]
label = "s5"
sha256 = "{S5_HELDOUT_SHA}"
size_bytes = 123
"#
    )
}

#[test]
fn valid_manifest_parses() {
    parse_manifests(&valid_manifest()).expect("a well-formed manifest must parse");
}

#[test]
fn tamper_unknown_key_rejected() {
    let m = valid_manifest().replace(
        "corpus_path = \"data/c.npz\"",
        "corpus_path = \"data/c.npz\"\nbogus_key = 7",
    );
    let err = parse_manifests(&m).unwrap_err();
    assert!(err.contains("unknown key") && err.contains("bogus_key"), "{err}");
}

#[test]
fn tamper_missing_required_rejected() {
    let m = valid_manifest().replace("anchor_path = \"checkpoints/a.pt\"\n", "");
    let err = parse_manifests(&m).unwrap_err();
    assert!(err.contains("anchor_path"), "missing required must be named: {err}");
}

#[test]
fn tamper_malformed_sha_rejected() {
    let m = valid_manifest().replace(V6_LIVE2_LS_SHA, "not_a_real_sha");
    let err = parse_manifests(&m).unwrap_err();
    assert!(err.contains("malformed sha256"), "{err}");
}

#[test]
fn tamper_heldout_corpus_sha_collision_rejected() {
    // Make the held-out sha equal the corpus pin → collision must be loud.
    let m = valid_manifest().replace(S5_HELDOUT_SHA, V6_LIVE2_LS_SHA);
    let err = parse_manifests(&m).unwrap_err();
    assert!(err.contains("collides"), "held-out ∩ corpus pin must be a loud error: {err}");
}
