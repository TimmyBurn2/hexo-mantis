//! Exceeds the 300-line soft cap (R8): single shared
//! fixture-verification module — the ONE code path both parity gates, the
//! self-tests, and the bench loader use (splitting it would fork the gate,
//! defeating the LAW-07 binding).
//!
//! Shared helpers for the graph-parity fixture gate: SHA-256 hex (the locked
//! `sha2` crate, checked against the manifest's Python-hashlib hashes on every
//! F-row), the manifest.tsv parser, the inputs.bin / MGPB blob readers, and the
//! canonical field serializer (the mirror of the capture writer). Every checker is
//! parametrized ONLY by the fixture-root path so the mutation self-tests
//! exercise the IDENTICAL functions the parity gate calls.

#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]

use std::path::{Path, PathBuf};

use mantis_graph::AxisGraph;
use sha2::{Digest, Sha256};

/// Canonical field order (capture manifest C-row order).
pub const FIELD_ORDER: [&str; 14] = [
    "node_feat",
    "edge_src",
    "edge_dst",
    "edge_attr",
    "legal_mask",
    "stone_mask",
    "node_coords",
    "policy_scatter_index",
    "legal_node_gather",
    "n_stones",
    "n_nodes_checksum",
    "window_center",
    "current_player",
    "builder_impl",
];

/// inputs.bin `class` codes.
pub const CLASS_BASE: u8 = 2;

const DTYPE_NAMES: [&str; 6] = ["f32", "i32", "u32", "u16", "u8", "i8"];

/// Lowercase hex SHA-256 of `data`, the manifest's hash format.
#[must_use]
pub fn sha256_hex(data: &[u8]) -> String {
    Sha256::digest(data).iter().map(|b| format!("{b:02x}")).collect()
}

/// First index at which `a` and `b` differ (or where one ends); `None` if equal.
#[must_use]
pub fn first_diff_offset(a: &[u8], b: &[u8]) -> Option<usize> {
    if let Some(i) = a.iter().zip(b.iter()).position(|(x, y)| x != y) {
        return Some(i);
    }
    if a.len() != b.len() {
        return Some(a.len().min(b.len()));
    }
    None
}

// fixture root

/// The repo's graph-parity fixture root (path computation only).
#[must_use]
pub fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/graph_parity")
}

/// FAIL-not-skip presence check: the fixture directory and manifest.tsv must
/// exist. The error names the missing path and the regeneration pointer.
pub fn verify_fixture_root(root: &Path) -> Result<(), String> {
    if !root.is_dir() {
        return Err(format!(
            "fixtures absent — restore {} (see its manifest; capture is regenerable \
             from the migration workspace)",
            root.display()
        ));
    }
    let m = root.join("manifest.tsv");
    if !m.is_file() {
        return Err(format!(
            "fixtures absent — manifest missing: {} (restore the graph_parity fixture \
             set; capture is regenerable from the migration workspace)",
            m.display()
        ));
    }
    Ok(())
}

// manifest.tsv

#[derive(Debug)]
pub struct FileRow {
    pub path: String,
    pub nbytes: u64,
    pub sha256: String,
}

#[derive(Debug)]
pub struct CaseRow {
    pub case_id: u32,
    pub field: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub nbytes: u64,
    pub sha256: String,
}

#[derive(Debug)]
pub struct Manifest {
    pub header: Vec<(String, String)>,
    pub files: Vec<FileRow>,
    pub cases: Vec<CaseRow>,
}

impl Manifest {
    #[must_use]
    pub fn header_value(&self, key: &str) -> Option<&str> {
        self.header
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }
    /// Comma-joined id list header (e.g. `sentinel_cases`, `raw_subset`).
    pub fn header_ids(&self, key: &str) -> Result<Vec<u32>, String> {
        let v = self
            .header_value(key)
            .ok_or_else(|| format!("manifest: header key '{key}' missing"))?;
        v.split(',')
            .map(|s| {
                s.trim()
                    .parse::<u32>()
                    .map_err(|e| format!("manifest: header '{key}' id '{s}': {e}"))
            })
            .collect()
    }
    /// The 14 C-rows of one case, in manifest order.
    #[must_use]
    pub fn case_rows(&self, case_id: u32) -> Vec<&CaseRow> {
        self.cases.iter().filter(|c| c.case_id == case_id).collect()
    }
}

fn parse_shape(s: &str, line_no: usize) -> Result<Vec<u64>, String> {
    s.split('x')
        .map(|d| {
            d.parse::<u64>()
                .map_err(|e| format!("manifest line {line_no}: malformed shape '{s}': {e}"))
        })
        .collect()
}

/// Parse `<root>/manifest.tsv`. Unknown row kind, unknown dtype, malformed
/// shape/number, or `schema != 1` is a loud error — never skipped.
pub fn parse_manifest(root: &Path) -> Result<Manifest, String> {
    let path = root.join("manifest.tsv");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("manifest unreadable: {}: {e}", path.display()))?;
    let mut m = Manifest { header: Vec::new(), files: Vec::new(), cases: Vec::new() };
    for (i, line) in text.lines().enumerate() {
        let line_no = i + 1;
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix("# ") {
            let (k, v) = rest
                .split_once(" = ")
                .ok_or_else(|| format!("manifest line {line_no}: malformed header '{line}'"))?;
            m.header.push((k.to_string(), v.to_string()));
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        match cols[0] {
            "F" => {
                if cols.len() != 4 {
                    return Err(format!("manifest line {line_no}: F-row wants 4 columns, got {}", cols.len()));
                }
                m.files.push(FileRow {
                    path: cols[1].to_string(),
                    nbytes: cols[2]
                        .parse()
                        .map_err(|e| format!("manifest line {line_no}: nbytes: {e}"))?,
                    sha256: cols[3].to_string(),
                });
            }
            "C" => {
                if cols.len() != 7 {
                    return Err(format!("manifest line {line_no}: C-row wants 7 columns, got {}", cols.len()));
                }
                if !DTYPE_NAMES.contains(&cols[3]) {
                    return Err(format!("manifest line {line_no}: unknown dtype '{}'", cols[3]));
                }
                m.cases.push(CaseRow {
                    case_id: cols[1]
                        .parse()
                        .map_err(|e| format!("manifest line {line_no}: case_id: {e}"))?,
                    field: cols[2].to_string(),
                    dtype: cols[3].to_string(),
                    shape: parse_shape(cols[4], line_no)?,
                    nbytes: cols[5]
                        .parse()
                        .map_err(|e| format!("manifest line {line_no}: nbytes: {e}"))?,
                    sha256: cols[6].to_string(),
                });
            }
            kind => return Err(format!("manifest line {line_no}: unknown row kind '{kind}'")),
        }
    }
    let schema = m
        .header_value("schema")
        .ok_or_else(|| "manifest: header key 'schema' missing".to_string())?;
    if schema != "1" {
        return Err(format!("manifest: schema {schema} != 1"));
    }
    Ok(m)
}

/// Verify one F-row: the file exists under `root`, has the declared nbytes,
/// and sha256-matches. The error names the file.
pub fn verify_file_row(root: &Path, row: &FileRow) -> Result<(), String> {
    let path = root.join(&row.path);
    let bytes = std::fs::read(&path)
        .map_err(|e| format!("fixture file absent/unreadable: {}: {e}", path.display()))?;
    if bytes.len() as u64 != row.nbytes {
        return Err(format!(
            "fixture file size drift: {} is {} bytes, manifest says {}",
            path.display(),
            bytes.len(),
            row.nbytes
        ));
    }
    let got = sha256_hex(&bytes);
    if got != row.sha256 {
        return Err(format!(
            "fixture file sha drift: {} sha256 {got} != manifest {}",
            path.display(),
            row.sha256
        ));
    }
    Ok(())
}

// binary readers

struct Cur<'a> {
    b: &'a [u8],
    off: usize,
    name: String,
}

impl<'a> Cur<'a> {
    fn take(&mut self, n: usize, what: &str) -> Result<&'a [u8], String> {
        if self.off + n > self.b.len() {
            return Err(format!(
                "{}: truncated — short read of {what} at offset {} (need {n} bytes, {} left)",
                self.name,
                self.off,
                self.b.len() - self.off
            ));
        }
        let s = &self.b[self.off..self.off + n];
        self.off += n;
        Ok(s)
    }
    fn u8v(&mut self, what: &str) -> Result<u8, String> {
        Ok(self.take(1, what)?[0])
    }
    fn i8v(&mut self, what: &str) -> Result<i8, String> {
        Ok(self.take(1, what)?[0] as i8)
    }
    fn u16v(&mut self, what: &str) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2, what)?.try_into().unwrap()))
    }
    fn u32v(&mut self, what: &str) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4, what)?.try_into().unwrap()))
    }
    fn i32v(&mut self, what: &str) -> Result<i32, String> {
        Ok(i32::from_le_bytes(self.take(4, what)?.try_into().unwrap()))
    }
    fn u64v(&mut self, what: &str) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8, what)?.try_into().unwrap()))
    }
}

#[derive(Debug)]
pub struct CaseInput {
    pub case_id: u32,
    pub class: u8,
    pub base_idx: i32,
    pub win_length: u8,
    pub radius: u16,
    pub current_player: i8,
    pub moves_remaining: u8,
    pub trunk_size: i32,
    pub stones: Vec<(i32, i32, i8)>,
}

fn read_input_record(cur: &mut Cur) -> Result<CaseInput, String> {
    let case_id = cur.u32v("case_id")?;
    let class = cur.u8v("class")?;
    let base_idx = cur.i32v("base_idx")?;
    let win_length = cur.u8v("win_length")?;
    let radius = cur.u16v("radius")?;
    let current_player = cur.i8v("current_player")?;
    let moves_remaining = cur.u8v("moves_remaining")?;
    let trunk_size = cur.i32v("trunk_size")?;
    let n_stones = cur.u32v("n_stones")? as usize;
    let mut stones = Vec::with_capacity(n_stones);
    for _ in 0..n_stones {
        let q = cur.i32v("stone q")?;
        let r = cur.i32v("stone r")?;
        let p = cur.i8v("stone p")?;
        stones.push((q, r, p));
    }
    Ok(CaseInput {
        case_id,
        class,
        base_idx,
        win_length,
        radius,
        current_player,
        moves_remaining,
        trunk_size,
        stones,
    })
}

/// The exact byte image of one inputs.bin record (mirror of the capture writer).
#[must_use]
pub fn serialize_input_record(c: &CaseInput) -> Vec<u8> {
    let mut b = Vec::with_capacity(22 + c.stones.len() * 9);
    b.extend_from_slice(&c.case_id.to_le_bytes());
    b.push(c.class);
    b.extend_from_slice(&c.base_idx.to_le_bytes());
    b.push(c.win_length);
    b.extend_from_slice(&c.radius.to_le_bytes());
    b.extend_from_slice(&c.current_player.to_le_bytes());
    b.push(c.moves_remaining);
    b.extend_from_slice(&c.trunk_size.to_le_bytes());
    b.extend_from_slice(&(c.stones.len() as u32).to_le_bytes());
    for &(q, r, p) in &c.stones {
        b.extend_from_slice(&q.to_le_bytes());
        b.extend_from_slice(&r.to_le_bytes());
        b.extend_from_slice(&p.to_le_bytes());
    }
    b
}

/// Read inputs.bin (magic MGPI, version 1). Short read = loud named error.
pub fn read_inputs_bin(path: &Path) -> Result<Vec<CaseInput>, String> {
    let bytes =
        std::fs::read(path).map_err(|e| format!("inputs.bin unreadable: {}: {e}", path.display()))?;
    let mut cur = Cur { b: &bytes, off: 0, name: path.display().to_string() };
    let magic = cur.take(4, "magic")?;
    if magic != b"MGPI" {
        return Err(format!("{}: bad magic {magic:?}", path.display()));
    }
    let version = cur.u32v("version")?;
    if version != 1 {
        return Err(format!("{}: version {version} != 1", path.display()));
    }
    let n_cases = cur.u32v("n_cases")? as usize;
    let mut cases = Vec::with_capacity(n_cases);
    for _ in 0..n_cases {
        cases.push(read_input_record(&mut cur)?);
    }
    if cur.off != bytes.len() {
        return Err(format!("{}: trailing bytes after {n_cases} records", path.display()));
    }
    Ok(cases)
}

#[derive(Debug)]
pub struct BlobField {
    pub name: String,
    pub dtype: String,
    pub dims: Vec<u64>,
    pub payload: Vec<u8>,
    /// Absolute offset of the payload inside the blob file.
    pub payload_offset: usize,
}

#[derive(Debug)]
pub struct Blob {
    pub case_id: u32,
    /// The embedded inputs.bin record, raw bytes.
    pub input_block: Vec<u8>,
    pub input: CaseInput,
    pub fields: Vec<BlobField>,
}

/// Read one MGPB raw golden blob. Any short read (declared nbytes vs EOF) is a
/// loud error naming the file.
pub fn read_blob(path: &Path) -> Result<Blob, String> {
    let bytes =
        std::fs::read(path).map_err(|e| format!("blob absent/unreadable: {}: {e}", path.display()))?;
    let mut cur = Cur { b: &bytes, off: 0, name: path.display().to_string() };
    let magic = cur.take(4, "magic")?;
    if magic != b"MGPB" {
        return Err(format!("{}: bad magic {magic:?}", path.display()));
    }
    let version = cur.u32v("version")?;
    if version != 1 {
        return Err(format!("{}: version {version} != 1", path.display()));
    }
    let case_id = cur.u32v("case_id")?;
    let ib_start = cur.off;
    let input = read_input_record(&mut cur)?;
    if input.case_id != case_id {
        return Err(format!("{}: embedded input case_id {} != header {case_id}", path.display(), input.case_id));
    }
    let input_block = bytes[ib_start..cur.off].to_vec();
    let n_fields = cur.u8v("n_fields")?;
    if n_fields != 14 {
        return Err(format!("{}: n_fields {n_fields} != 14", path.display()));
    }
    let mut fields = Vec::with_capacity(14);
    for _ in 0..14 {
        let name_len = cur.u8v("field name_len")? as usize;
        let name = std::str::from_utf8(cur.take(name_len, "field name")?)
            .map_err(|e| format!("{}: field name not ASCII: {e}", path.display()))?
            .to_string();
        let dtype_code = cur.u8v("dtype code")?;
        if !(1..=6).contains(&dtype_code) {
            return Err(format!("{}: field {name}: unknown dtype code {dtype_code}", path.display()));
        }
        let dtype = DTYPE_NAMES[(dtype_code - 1) as usize].to_string();
        let ndim = cur.u8v("ndim")? as usize;
        let mut dims = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            dims.push(u64::from(cur.u32v("dim")?));
        }
        let nbytes = cur.u64v("payload_nbytes")? as usize;
        let payload_offset = cur.off;
        let payload = cur.take(nbytes, &format!("field {name} payload"))?.to_vec();
        fields.push(BlobField { name, dtype, dims, payload, payload_offset });
    }
    if cur.off != bytes.len() {
        return Err(format!("{}: trailing bytes", path.display()));
    }
    let names: Vec<&str> = fields.iter().map(|f| f.name.as_str()).collect();
    if names != FIELD_ORDER {
        return Err(format!("{}: field order drift: {names:?}", path.display()));
    }
    Ok(Blob { case_id, input_block, input, fields })
}

/// Verify a blob's 14 fields against its manifest C-rows (dtype, shape, nbytes,
/// payload sha). First mismatch is a loud error naming the case and field.
pub fn check_blob_against_case_rows(blob: &Blob, rows: &[&CaseRow]) -> Result<(), String> {
    if rows.len() != 14 {
        return Err(format!("case {}: {} C-rows, want 14", blob.case_id, rows.len()));
    }
    for (f, row) in blob.fields.iter().zip(rows.iter()) {
        if f.name != row.field {
            return Err(format!(
                "case {}: field order mismatch: blob '{}' vs manifest '{}'",
                blob.case_id, f.name, row.field
            ));
        }
        if f.dtype != row.dtype {
            return Err(format!(
                "case {} field {}: dtype mismatch: blob {} vs manifest {}",
                blob.case_id, f.name, f.dtype, row.dtype
            ));
        }
        if f.dims != row.shape {
            return Err(format!(
                "case {} field {}: shape mismatch: blob {:?} vs manifest {:?}",
                blob.case_id, f.name, f.dims, row.shape
            ));
        }
        if f.payload.len() as u64 != row.nbytes {
            return Err(format!(
                "case {} field {}: nbytes mismatch: blob {} vs manifest {}",
                blob.case_id,
                f.name,
                f.payload.len(),
                row.nbytes
            ));
        }
        let got = sha256_hex(&f.payload);
        if got != row.sha256 {
            return Err(format!(
                "case {} field {}: payload sha mismatch: blob {got} vs manifest {}",
                blob.case_id, f.name, row.sha256
            ));
        }
    }
    Ok(())
}

// canonical serializer (mirror of the capture writer)

#[derive(Debug)]
pub struct BuiltField {
    pub name: &'static str,
    pub dtype: &'static str,
    pub dims: Vec<u64>,
    pub payload: Vec<u8>,
}

fn f32s(v: &[f32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for &x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}
fn u32s(v: &[u32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for &x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}
fn i32s(v: &[i32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for &x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}
fn bools(v: &[bool]) -> Vec<u8> {
    v.iter().map(|&b| u8::from(b)).collect()
}

/// The canonical little-endian serialization of a built graph — the 14 fields
/// in manifest order, byte-identical to what the capture writer emitted from
/// the old builder for equal graphs.
#[must_use]
pub fn canonical_fields(g: &AxisGraph) -> Vec<BuiltField> {
    let n = g.num_nodes() as u64;
    let e = g.num_edges() as u64;
    let l = g.legal_node_gather.len() as u64;
    vec![
        BuiltField { name: "node_feat", dtype: "f32", dims: vec![n, 11], payload: f32s(&g.node_feat.0) },
        BuiltField { name: "edge_src", dtype: "u32", dims: vec![e], payload: u32s(&g.edge_index.src) },
        BuiltField { name: "edge_dst", dtype: "u32", dims: vec![e], payload: u32s(&g.edge_index.dst) },
        BuiltField { name: "edge_attr", dtype: "f32", dims: vec![e, 5], payload: f32s(&g.edge_attr.0) },
        BuiltField { name: "legal_mask", dtype: "u8", dims: vec![n], payload: bools(&g.legal_mask) },
        BuiltField { name: "stone_mask", dtype: "u8", dims: vec![n], payload: bools(&g.stone_mask) },
        BuiltField { name: "node_coords", dtype: "i32", dims: vec![2 * n], payload: i32s(&g.node_coords) },
        BuiltField {
            name: "policy_scatter_index",
            dtype: "i32",
            dims: vec![l],
            payload: i32s(&g.policy_scatter_index.0),
        },
        BuiltField {
            name: "legal_node_gather",
            dtype: "u32",
            dims: vec![l],
            payload: u32s(&g.legal_node_gather),
        },
        BuiltField { name: "n_stones", dtype: "u16", dims: vec![1], payload: g.n_stones.to_le_bytes().to_vec() },
        BuiltField {
            name: "n_nodes_checksum",
            dtype: "u32",
            dims: vec![1],
            payload: g.n_nodes_checksum.to_le_bytes().to_vec(),
        },
        BuiltField {
            name: "window_center",
            dtype: "i32",
            dims: vec![2],
            payload: i32s(&[g.window_center.0, g.window_center.1]),
        },
        BuiltField {
            name: "current_player",
            dtype: "i8",
            dims: vec![1],
            payload: g.current_player.to_le_bytes().to_vec(),
        },
        BuiltField { name: "builder_impl", dtype: "u8", dims: vec![1], payload: vec![g.builder_impl] },
    ]
}

/// Build the graph for one fixture case input.
#[must_use]
pub fn build_case(c: &CaseInput) -> AxisGraph {
    let params = mantis_graph::BuildParams {
        win_length: c.win_length,
        radius: c.radius,
        current_player: c.current_player,
        moves_remaining: c.moves_remaining,
        trunk_size: c.trunk_size,
    };
    mantis_graph::build_axis_graph(&mantis_graph::StoneList { stones: c.stones.clone() }, &params)
}
