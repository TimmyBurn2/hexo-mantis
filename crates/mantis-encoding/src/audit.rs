//! Registry-shape audit backend (the WP3 half of the encoding audit).
//!
//! Ports the two REGISTRY-SHAPE sections of the old Python audit — §1 registered
//! census and §6 cross-table ckpt↔corpus reconciliation (INV-1..6) — plus the
//! severity → exit-code mapping. Both are pure data (no filesystem / torch /
//! npz): the census reads `all_specs()`, the cross-table joins synthetic entry
//! vectors. The filesystem / torch / npz / grep CLI sections are WP7.

use crate::registry::all_specs;

/// Finding severity. `exit_code` maps info/warn/error → 0/1/2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Info,
    Warn,
    Error,
}

impl Severity {
    #[must_use]
    pub fn code(self) -> i32 {
        match self {
            Severity::Info => 0,
            Severity::Warn => 1,
            Severity::Error => 2,
        }
    }
}

/// One audit finding.
#[derive(Debug, Clone)]
pub struct Finding {
    pub severity: Severity,
    pub section: &'static str,
    pub message: String,
}

/// A checkpoint's audit-relevant metadata (torch load is WP7; this is the pure
/// shape the cross-table joins on).
#[derive(Debug, Clone)]
pub struct CkptEntry {
    pub display: String,
    pub name: Option<String>,
    pub corpus_sha: Option<String>,
    pub has_meta: bool,
}

/// A corpus's audit-relevant metadata (npz sidecar + file sha are WP7).
#[derive(Debug, Clone)]
pub struct CorpusEntry {
    pub display: String,
    pub name: Option<String>,
    pub sha: Option<String>,
}

/// §1 — registered census. One row per registered encoding (sorted by name):
/// `[name, board_size, n_planes, policy_logits, multi_window, schema_v]`.
#[must_use]
pub fn census() -> Vec<[String; 6]> {
    let mut rows: Vec<[String; 6]> = all_specs()
        .map(|s| {
            [
                s.name.to_string(),
                s.board_size.to_string(),
                s.n_planes.to_string(),
                s.policy_logit_count.to_string(),
                s.is_multi_window.to_string(),
                s.schema_version.to_string(),
            ]
        })
        .collect();
    rows.sort_by(|a, b| a[0].cmp(&b[0]));
    rows
}

/// §6 — cross-table consistency (ckpts ↔ corpora via sha256). Emits INV-1..6:
///   INV-1 enc-mismatch (ckpt name != corpus name)     → error
///   INV-2 orphan-sha (ckpt sha matches no corpus)      → error
///   INV-3 no-corpus-sha (metadata lacks corpus_sha256) → warn
///   INV-4 no-meta (checkpoint has no metadata)         → warn
///   INV-5 ok (ckpt ↔ corpus name agree)               → info
///   INV-6 orphan-corpus (sha unreferenced by any ckpt) → info (warn in strict)
#[must_use]
pub fn cross_table(ckpts: &[CkptEntry], corpora: &[CorpusEntry], strict: bool) -> Vec<Finding> {
    let mut findings = Vec::new();

    let by_sha: std::collections::HashMap<&str, &CorpusEntry> = corpora
        .iter()
        .filter_map(|c| c.sha.as_deref().map(|s| (s, c)))
        .collect();

    let mut referenced: std::collections::HashSet<String> = std::collections::HashSet::new();

    for ck in ckpts {
        if !ck.has_meta {
            findings.push(Finding {
                severity: Severity::Warn,
                section: "§6",
                message: format!("{}: no metadata — cannot cross-reference corpus", ck.display),
            });
            continue;
        }
        let Some(sha) = ck.corpus_sha.as_deref() else {
            findings.push(Finding {
                severity: Severity::Warn,
                section: "§6",
                message: format!("{}: metadata lacks corpus_sha256", ck.display),
            });
            continue;
        };
        let Some(matched) = by_sha.get(sha) else {
            findings.push(Finding {
                severity: Severity::Error,
                section: "§6",
                message: format!(
                    "{}: corpus_sha256={}… matches no corpus",
                    ck.display,
                    &sha[..sha.len().min(12)]
                ),
            });
            continue;
        };
        if let Some(msha) = matched.sha.as_deref() {
            referenced.insert(msha.to_string());
        }
        if matched.name != ck.name {
            findings.push(Finding {
                severity: Severity::Error,
                section: "§6",
                message: format!(
                    "{}: ckpt encoding={:?} but corpus encoding={:?}",
                    ck.display, ck.name, matched.name
                ),
            });
        } else {
            findings.push(Finding {
                severity: Severity::Info,
                section: "§6",
                message: format!("{} ↔ {}: {:?}", ck.display, matched.display, ck.name),
            });
        }
    }

    // INV-6 — orphan corpora (sha unreferenced by any stamped ckpt).
    for co in corpora {
        if let Some(sha) = co.sha.as_deref() {
            if !referenced.contains(sha) {
                findings.push(Finding {
                    severity: if strict { Severity::Warn } else { Severity::Info },
                    section: "§6",
                    message: format!(
                        "{}: orphan corpus — unused by any stamped checkpoint",
                        co.display
                    ),
                });
            }
        }
    }

    findings
}

/// Severity aggregation → process exit code (0 info / 1 warn / 2 error).
#[must_use]
pub fn exit_code(findings: &[Finding]) -> i32 {
    findings.iter().map(|f| f.severity.code()).max().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ck(display: &str, name: Option<&str>, sha: Option<&str>, has_meta: bool) -> CkptEntry {
        CkptEntry {
            display: display.to_string(),
            name: name.map(str::to_string),
            corpus_sha: sha.map(str::to_string),
            has_meta,
        }
    }
    fn co(display: &str, name: Option<&str>, sha: Option<&str>) -> CorpusEntry {
        CorpusEntry {
            display: display.to_string(),
            name: name.map(str::to_string),
            sha: sha.map(str::to_string),
        }
    }
    fn statuses(fs: &[Finding]) -> Vec<Severity> {
        fs.iter().map(|f| f.severity).collect()
    }

    #[test]
    fn census_shape_and_rows() {
        let rows = census();
        assert_eq!(rows.len(), 5, "pruned registered set = 5");
        // sorted by name
        let names: Vec<&str> = rows.iter().map(|r| r[0].as_str()).collect();
        assert_eq!(names, ["gnn_axis_r8", "gnn_axis_v1", "v6", "v6_live2_ls", "v6w25"]);
        let v6 = rows.iter().find(|r| r[0] == "v6").unwrap();
        assert_eq!(v6, &[
            "v6".to_string(),
            "19".to_string(),
            "8".to_string(),
            "362".to_string(),
            "false".to_string(),
            "3".to_string(),
        ]);
        let gnn = rows.iter().find(|r| r[0] == "gnn_axis_v1").unwrap();
        assert_eq!(gnn[2], "0"); // n_planes
        assert_eq!(gnn[5], "4"); // schema_version
    }

    #[test]
    fn inv5_ok() {
        let ck = [ck("m.pt", Some("v6"), Some("aa"), true)];
        let co = [co("c.npz", Some("v6"), Some("aa"))];
        let f = cross_table(&ck, &co, false);
        assert_eq!(statuses(&f), [Severity::Info]);
        assert_eq!(exit_code(&f), 0);
    }

    #[test]
    fn inv1_enc_mismatch_error() {
        let ck = [ck("m.pt", Some("v6"), Some("aa"), true)];
        let co = [co("c.npz", Some("v6w25"), Some("aa"))];
        let f = cross_table(&ck, &co, false);
        assert!(f.iter().any(|x| x.severity == Severity::Error && x.message.contains("v6w25")));
        assert_eq!(exit_code(&f), 2);
    }

    #[test]
    fn inv2_orphan_sha_error() {
        let ck = [ck("m.pt", Some("v6"), Some("deadbeef"), true)];
        let co = [co("c.npz", Some("v6"), Some("aa"))];
        let f = cross_table(&ck, &co, false);
        // INV-2 error on the ckpt; INV-6 info on the unreferenced corpus.
        assert!(f.iter().any(|x| x.severity == Severity::Error && x.message.contains("matches no corpus")));
        assert_eq!(exit_code(&f), 2);
    }

    #[test]
    fn inv3_no_corpus_sha_warn() {
        let ck = [ck("m.pt", Some("v6"), None, true)];
        let f = cross_table(&ck, &[], false);
        assert_eq!(statuses(&f), [Severity::Warn]);
        assert_eq!(exit_code(&f), 1);
    }

    #[test]
    fn inv4_no_meta_warn() {
        let ck = [ck("m.pt", None, None, false)];
        let f = cross_table(&ck, &[], false);
        assert_eq!(statuses(&f), [Severity::Warn]);
        assert!(f[0].message.contains("no metadata"));
    }

    #[test]
    fn inv6_orphan_corpus_info_then_warn_in_strict() {
        let co = [co("c.npz", Some("v6"), Some("aa"))];
        let lax = cross_table(&[], &co, false);
        assert_eq!(statuses(&lax), [Severity::Info]);
        assert_eq!(exit_code(&lax), 0);
        let strict = cross_table(&[], &co, true);
        assert_eq!(statuses(&strict), [Severity::Warn]);
        assert_eq!(exit_code(&strict), 1);
    }

    #[test]
    fn exit_code_takes_max_severity() {
        let f = vec![
            Finding { severity: Severity::Info, section: "§6", message: String::new() },
            Finding { severity: Severity::Error, section: "§6", message: String::new() },
            Finding { severity: Severity::Warn, section: "§6", message: String::new() },
        ];
        assert_eq!(exit_code(&f), 2);
        assert_eq!(exit_code(&[]), 0);
    }
}
