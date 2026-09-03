//! The graph-parity fixture gate: the ported builder must reproduce the
//! captured old-builder goldens byte-exactly (floats included, NO tolerance)
//! over all 1,696 fixture cases x 14 fields. Fixture absence or drift FAILS —
//! never skips (loud-shrinkage rule; repo_design §8).

#[allow(dead_code)]
mod common;

use std::collections::BTreeMap;

/// Fixture root or die loud with the named path (FAIL, not skip).
fn root_or_die() -> std::path::PathBuf {
    let root = common::fixture_root();
    common::verify_fixture_root(&root).unwrap_or_else(|e| panic!("{e}"));
    root
}

#[test]
fn manifest_and_fixture_files_valid() {
    let root = root_or_die();
    let m = common::parse_manifest(&root).unwrap_or_else(|e| panic!("{e}"));
    for row in &m.files {
        common::verify_file_row(&root, row).unwrap_or_else(|e| panic!("{e}"));
    }
    assert_eq!(m.header_value("endianness"), Some("little"));
    assert_eq!(m.header_value("total_cases"), Some("1696"));
    let cases = common::read_inputs_bin(&root.join("inputs.bin")).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(cases.len(), 1696, "inputs.bin case count");
    // class splits must match the header census
    let mut counts = [0usize; 6];
    for c in &cases {
        counts[c.class as usize] += 1;
    }
    let got = format!(
        "deg_empty:{},deg_1stone:{},base:{},sweep:{},trunc:{},altgeom:{}",
        counts[0], counts[1], counts[2], counts[3], counts[4], counts[5]
    );
    assert_eq!(m.header_value("class_counts"), Some(got.as_str()), "class split drift");
    assert!(!m.header_ids("sentinel_cases").unwrap_or_else(|e| panic!("{e}")).is_empty());
    assert!(!m.header_ids("raw_subset").unwrap_or_else(|e| panic!("{e}")).is_empty());
    assert_eq!(m.cases.len(), 1696 * 14, "C-row count");
}

#[test]
fn all_1696_cases_byte_parity() {
    let root = root_or_die();
    let m = common::parse_manifest(&root).unwrap_or_else(|e| panic!("{e}"));
    let cases = common::read_inputs_bin(&root.join("inputs.bin")).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(cases.len(), 1696);
    let raw_subset = m.header_ids("raw_subset").unwrap_or_else(|e| panic!("{e}"));

    // Index C-rows once (manifest order is per-case field order).
    let mut rows_by_case: BTreeMap<u32, Vec<&common::CaseRow>> = BTreeMap::new();
    for row in &m.cases {
        rows_by_case.entry(row.case_id).or_default().push(row);
    }

    let mut mismatches: Vec<String> = Vec::new();
    for case in &cases {
        let rows = rows_by_case
            .get(&case.case_id)
            .unwrap_or_else(|| panic!("case {}: no C-rows in manifest", case.case_id));
        assert_eq!(rows.len(), 14, "case {}: C-row count", case.case_id);
        let g = common::build_case(case);
        let built = common::canonical_fields(&g);
        for (bf, row) in built.iter().zip(rows.iter()) {
            assert_eq!(bf.name, row.field, "case {} field order", case.case_id);
            let mut bad = Vec::new();
            if bf.dtype != row.dtype {
                bad.push(format!("dtype {} != {}", bf.dtype, row.dtype));
            }
            if bf.dims != row.shape {
                bad.push(format!("shape {:?} != {:?}", bf.dims, row.shape));
            }
            if bf.payload.len() as u64 != row.nbytes {
                bad.push(format!("nbytes {} != {}", bf.payload.len(), row.nbytes));
            }
            let sha = common::sha256_hex(&bf.payload);
            if sha != row.sha256 {
                bad.push(format!("sha {sha} != {}", row.sha256));
            }
            if bad.is_empty() {
                continue;
            }
            let diag = if raw_subset.contains(&case.case_id) {
                let blob_path = root.join(format!("raw/case_{:05}.bin", case.case_id));
                match common::read_blob(&blob_path) {
                    Ok(blob) => {
                        let gold = &blob.fields.iter().find(|f| f.name == bf.name).unwrap().payload;
                        match common::first_diff_offset(&bf.payload, gold) {
                            Some(off) => format!(
                                "first diff at payload offset {off}: built {:?} vs golden {:?}",
                                bf.payload.get(off),
                                gold.get(off)
                            ),
                            None => "payloads byte-equal vs raw golden (manifest row drift?)".to_string(),
                        }
                    }
                    Err(e) => format!("raw golden unreadable for offset diff: {e}"),
                }
            } else {
                "hash-only case: regenerate the full golden blob workspace-side (single \
                 capture-script run against the frozen predecessor tree) for offline \
                 offset diffing"
                    .to_string()
            };
            mismatches.push(format!(
                "case {} field {}: {} — {}",
                case.case_id,
                bf.name,
                bad.join("; "),
                diag
            ));
        }
    }
    assert!(
        mismatches.is_empty(),
        "BYTE PARITY FAILED — {} mismatches:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
    println!("PARITY OK: 1696 cases, 14 fields, 0 mismatches");
}

#[test]
fn raw_subset_deep_byte_parity() {
    let root = root_or_die();
    let m = common::parse_manifest(&root).unwrap_or_else(|e| panic!("{e}"));
    let raw_subset = m.header_ids("raw_subset").unwrap_or_else(|e| panic!("{e}"));
    let cases = common::read_inputs_bin(&root.join("inputs.bin")).unwrap_or_else(|e| panic!("{e}"));
    assert!(!raw_subset.is_empty());
    for &cid in &raw_subset {
        let blob_path = root.join(format!("raw/case_{cid:05}.bin"));
        let blob = common::read_blob(&blob_path).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(blob.case_id, cid);
        // blob internal consistency vs its manifest C-rows (the same check the
        // self-tests mutate)
        let rows = m.case_rows(cid);
        common::check_blob_against_case_rows(&blob, &rows).unwrap_or_else(|e| panic!("{e}"));
        // embedded input block == the inputs.bin record
        let case = &cases[cid as usize];
        assert_eq!(case.case_id, cid);
        assert_eq!(
            blob.input_block,
            common::serialize_input_record(case),
            "case {cid}: embedded input block != inputs.bin record"
        );
        // byte-compare EVERY field payload (exercises the offset path routinely)
        let g = common::build_case(case);
        let built = common::canonical_fields(&g);
        for (bf, gf) in built.iter().zip(blob.fields.iter()) {
            assert_eq!(bf.name, gf.name);
            if let Some(off) = common::first_diff_offset(&bf.payload, &gf.payload) {
                panic!(
                    "case {cid} field {}: first diff at payload offset {off}: built {:?} vs golden {:?}",
                    bf.name,
                    bf.payload.get(off),
                    gf.payload.get(off)
                );
            }
            assert_eq!(bf.dims, gf.dims, "case {cid} field {} dims", bf.name);
        }
    }
}

#[test]
fn sentinel_present_and_roundtrips() {
    let root = root_or_die();
    let m = common::parse_manifest(&root).unwrap_or_else(|e| panic!("{e}"));
    let sentinel_cases = m.header_ids("sentinel_cases").unwrap_or_else(|e| panic!("{e}"));
    // A fixture set without a sentinel is a FAILURE (hole), never a pass.
    assert!(!sentinel_cases.is_empty(), "sentinel-free fixture set — hole, not a pass");
    let cases = common::read_inputs_bin(&root.join("inputs.bin")).unwrap_or_else(|e| panic!("{e}"));
    for &cid in &sentinel_cases {
        let g = common::build_case(&cases[cid as usize]);
        assert!(
            g.policy_scatter_index.0.contains(&mantis_graph::OFF_WINDOW_SLOT),
            "case {cid}: manifest lists a sentinel but the NEW builder produced none"
        );
    }
    // >= 1 committed raw golden shows the 0xFFFFFFFF LE pattern at a sentinel slot.
    let raw_subset = m.header_ids("raw_subset").unwrap_or_else(|e| panic!("{e}"));
    let mut verified = 0usize;
    for &cid in &raw_subset {
        if !sentinel_cases.contains(&cid) {
            continue;
        }
        let blob = common::read_blob(&root.join(format!("raw/case_{cid:05}.bin")))
            .unwrap_or_else(|e| panic!("{e}"));
        let psi = &blob.fields.iter().find(|f| f.name == "policy_scatter_index").unwrap().payload;
        let idx = psi
            .chunks_exact(4)
            .position(|c| i32::from_le_bytes(c.try_into().unwrap()) == mantis_graph::OFF_WINDOW_SLOT)
            .unwrap_or_else(|| panic!("case {cid}: listed sentinel case has no -1 slot in the golden"));
        assert_eq!(
            &psi[idx * 4..idx * 4 + 4],
            &[0xFF, 0xFF, 0xFF, 0xFF],
            "case {cid}: sentinel byte pattern not 0xFFFFFFFF LE"
        );
        verified += 1;
    }
    assert!(verified >= 1, "raw subset contains no sentinel case — selector hole");
}

#[test]
fn window_boundary_slots_honest() {
    // Adversarial construction (no fixtures, no behavior change): a wide bbox
    // whose 19x19 policy window both CONTAINS legal cells exactly at the
    // window edge and EXCLUDES others. In-window edge cells must carry the
    // canonical slot; out-of-window cells must carry the -1 sentinel; the
    // always-on verify_contract inside the builder accepted the graph (the
    // build returning at all proves it).
    let trunk = 19i32;
    let half = (trunk - 1) / 2;
    let stones = mantis_graph::StoneList { stones: vec![(0, 0, 1), (28, 0, -1)] };
    let params = mantis_graph::BuildParams::V1_GEOMETRY; // wl 6 / radius 6 / trunk 19
    let g = mantis_graph::build_axis_graph(&stones, &params);
    let (cq, cr) = g.window_center;
    assert_eq!((cq, cr), (14, 0), "bbox midpoint");
    let mut n_sentinel = 0usize;
    let mut n_edge_cell = 0usize;
    for (i, &row) in g.legal_node_gather.iter().enumerate() {
        let q = g.node_coords[row as usize * 2];
        let r = g.node_coords[row as usize * 2 + 1];
        let wq = q - cq + half;
        let wr = r - cr + half;
        let in_window = wq >= 0 && wq < trunk && wr >= 0 && wr < trunk;
        let slot = g.policy_scatter_index.0[i];
        if in_window {
            assert_eq!(slot, wq * trunk + wr, "in-window legal cell ({q},{r}) must get the canonical slot");
            if wq == 0 || wq == trunk - 1 || wr == 0 || wr == trunk - 1 {
                n_edge_cell += 1;
            }
        } else {
            assert_eq!(slot, mantis_graph::OFF_WINDOW_SLOT, "off-window legal cell ({q},{r}) must get -1");
            n_sentinel += 1;
        }
    }
    assert!(n_sentinel > 0, "construction must produce off-window legal cells");
    assert!(n_edge_cell > 0, "construction must produce in-window cells exactly at the window edge");
}
