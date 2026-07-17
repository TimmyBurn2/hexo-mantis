//! Exceeds the 300-line soft cap (R8): single shared, dep-free
//! fixture-verification module — the ONE code path both parity gates, the
//! self-tests, and the bench loader use (splitting it would fork the gate,
//! defeating the LAW-07 binding).
//!
//! Shared helpers for the graph-parity fixture gate: hand-rolled FIPS-180-4
//! SHA-256 (cross-validated against the manifest's Python-hashlib hashes on
//! every F-row plus FIPS vectors in fixture_selftest.rs), the manifest.tsv
//! parser, the inputs.bin / MGPB blob readers, and the canonical field
//! serializer (the mirror of the capture writer). Every checker is
//! parametrized ONLY by the fixture-root path so the mutation self-tests
//! exercise the IDENTICAL functions the parity gate calls.

#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]

use std::path::{Path, PathBuf};

use mantis_graph::AxisGraph;

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

// ── SHA-256 (FIPS 180-4, dep-free) ───────────────────────────────────────────

const K: [u32; 64] = [
    0x428a_2f98, 0x7137_4491, 0xb5c0_fbcf, 0xe9b5_dba5, 0x3956_c25b, 0x59f1_11f1, 0x923f_82a4,
    0xab1c_5ed5, 0xd807_aa98, 0x1283_5b01, 0x2431_85be, 0x550c_7dc3, 0x72be_5d74, 0x80de_b1fe,
    0x9bdc_06a7, 0xc19b_f174, 0xe49b_69c1, 0xefbe_4786, 0x0fc1_9dc6, 0x240c_a1cc, 0x2de9_2c6f,
    0x4a74_84aa, 0x5cb0_a9dc, 0x76f9_88da, 0x983e_5152, 0xa831_c66d, 0xb003_27c8, 0xbf59_7fc7,
    0xc6e0_0bf3, 0xd5a7_9147, 0x06ca_6351, 0x1429_2967, 0x27b7_0a85, 0x2e1b_2138, 0x4d2c_6dfc,
    0x5338_0d13, 0x650a_7354, 0x766a_0abb, 0x81c2_c92e, 0x9272_2c85, 0xa2bf_e8a1, 0xa81a_664b,
    0xc24b_8b70, 0xc76c_51a3, 0xd192_e819, 0xd699_0624, 0xf40e_3585, 0x106a_a070, 0x19a4_c116,
    0x1e37_6c08, 0x2748_774c, 0x34b0_bcb5, 0x391c_0cb3, 0x4ed8_aa4a, 0x5b9c_ca4f, 0x682e_6ff3,
    0x748f_82ee, 0x78a5_636f, 0x84c8_7814, 0x8cc7_0208, 0x90be_fffa, 0xa450_6ceb, 0xbef9_a3f7,
    0xc671_78f2,
];

/// Hand-rolled SHA-256 (test-side only; cross-validated on every manifest
/// F-row against Python hashlib plus FIPS vectors in fixture_selftest.rs).
#[must_use]
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h: [u32; 8] = [
        0x6a09_e667, 0xbb67_ae85, 0x3c6e_f372, 0xa54f_f53a, 0x510e_527f, 0x9b05_688c,
        0x1f83_d9ab, 0x5be0_cd19,
    ];
    let bitlen = (data.len() as u64).wrapping_mul(8);
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bitlen.to_be_bytes());
    let mut w = [0u32; 64];
    for chunk in msg.chunks_exact(64) {
        for (i, word) in w.iter_mut().take(16).enumerate() {
            *word = u32::from_be_bytes([chunk[4 * i], chunk[4 * i + 1], chunk[4 * i + 2], chunk[4 * i + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    let mut out = [0u8; 32];
    for (i, word) in h.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

#[must_use]
pub fn sha256_hex(data: &[u8]) -> String {
    let mut s = String::with_capacity(64);
    for b in sha256(data) {
        s.push_str(&format!("{b:02x}"));
    }
    s
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

// ── fixture root ─────────────────────────────────────────────────────────────

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

// ── manifest.tsv ─────────────────────────────────────────────────────────────

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

// ── binary readers ───────────────────────────────────────────────────────────

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

// ── canonical serializer (mirror of the capture writer) ──────────────────────

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
