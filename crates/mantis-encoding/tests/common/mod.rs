//! Exceeds the 300-line soft cap (R8): single shared, dep-free
//! fixture-verification module — the ONE code path both the parity gate, the
//! mutation self-tests, and the bench-adjacent loader use (splitting it would
//! fork the gate, defeating the LAW-07 binding).
//!
//! Shared helpers for the encode-parity fixture gate: hand-rolled FIPS-180-4
//! SHA-256 (cross-validated against the manifest's Python-hashlib hashes on every
//! F/C-row plus FIPS vectors in fixture_selftest.rs), the manifest.tsv parser,
//! the MEPI inputs.bin / MEPB blob readers, and the NEW-side kernel output
//! builder (drives the ported free-fn kernels over each case's input block).

#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]

use std::path::{Path, PathBuf};

use mantis_core::{Board, Ply};
use mantis_encoding::{
    encode_state_to_buffer, encode_state_to_buffer_channels, encode_chain_planes, lookup_or_panic,
    to_planes, to_planes_channels,
};

// ── kernel / class codes (mirror capture_dump.rs + capture_goldens.py) ───────
pub const K_STATE: u8 = 1;
pub const K_CHANNELS: u8 = 2;
pub const K_CHAIN: u8 = 3;
pub const K_TO_PLANES: u8 = 4;
pub const K_TO_PLANES_CHANNELS: u8 = 5;

pub const CLASS_OOR_SKIP: u8 = 3;
pub const CLASS_PANIC: u8 = 4;

const DTYPE_NAMES: [&str; 1] = ["f32"];

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

#[must_use]
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h: [u32; 8] = [
        0x6a09_e667, 0xbb67_ae85, 0x3c6e_f372, 0xa54f_f53a, 0x510e_527f, 0x9b05_688c, 0x1f83_d9ab,
        0x5be0_cd19,
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
            *word = u32::from_be_bytes([
                chunk[4 * i],
                chunk[4 * i + 1],
                chunk[4 * i + 2],
                chunk[4 * i + 3],
            ]);
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
#[must_use]
pub fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/encode_parity")
}

/// FAIL-not-skip presence check: the fixture dir + manifest.tsv must exist.
pub fn verify_fixture_root(root: &Path) -> Result<(), String> {
    if !root.is_dir() {
        return Err(format!(
            "fixtures absent — restore {} (regenerable via wp/WP3/capture/capture_goldens.py)",
            root.display()
        ));
    }
    let m = root.join("manifest.tsv");
    if !m.is_file() {
        return Err(format!(
            "fixtures absent — manifest missing: {} (restore the encode_parity fixture set)",
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
    pub kernel: String,
    pub encoding: String,
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
    /// Comma-joined id list header (e.g. `panic_cases`, `raw_subset`, `oor_cases`).
    pub fn header_ids(&self, key: &str) -> Result<Vec<u32>, String> {
        let v = self
            .header_value(key)
            .ok_or_else(|| format!("manifest: header key '{key}' missing"))?;
        if v.trim().is_empty() {
            return Ok(Vec::new());
        }
        v.split(',')
            .map(|s| {
                s.trim()
                    .parse::<u32>()
                    .map_err(|e| format!("manifest: header '{key}' id '{s}': {e}"))
            })
            .collect()
    }
    #[must_use]
    pub fn case_row(&self, case_id: u32) -> Option<&CaseRow> {
        self.cases.iter().find(|c| c.case_id == case_id)
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
                    return Err(format!(
                        "manifest line {line_no}: F-row wants 4 columns, got {}",
                        cols.len()
                    ));
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
                if cols.len() != 8 {
                    return Err(format!(
                        "manifest line {line_no}: C-row wants 8 columns, got {}",
                        cols.len()
                    ));
                }
                if !DTYPE_NAMES.contains(&cols[4]) {
                    return Err(format!("manifest line {line_no}: unknown dtype '{}'", cols[4]));
                }
                m.cases.push(CaseRow {
                    case_id: cols[1]
                        .parse()
                        .map_err(|e| format!("manifest line {line_no}: case_id: {e}"))?,
                    kernel: cols[2].to_string(),
                    encoding: cols[3].to_string(),
                    dtype: cols[4].to_string(),
                    shape: parse_shape(cols[5], line_no)?,
                    nbytes: cols[6]
                        .parse()
                        .map_err(|e| format!("manifest line {line_no}: nbytes: {e}"))?,
                    sha256: cols[7].to_string(),
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

/// Verify one F-row: file exists under `root`, declared nbytes, sha256 matches.
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

// ── binary reader ────────────────────────────────────────────────────────────
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
    fn u32v(&mut self, what: &str) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4, what)?.try_into().unwrap()))
    }
    fn i32v(&mut self, what: &str) -> Result<i32, String> {
        Ok(i32::from_le_bytes(self.take(4, what)?.try_into().unwrap()))
    }
    fn u64v(&mut self, what: &str) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8, what)?.try_into().unwrap()))
    }
    fn f32v(&mut self, n: usize, what: &str) -> Result<Vec<f32>, String> {
        let raw = self.take(4 * n, what)?;
        Ok(raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }
}

/// One decoded inputs.bin / blob-embedded case record.
#[derive(Debug, Clone)]
pub struct CaseInput {
    pub case_id: u32,
    pub kernel: u8,
    pub encoding_id: u8,
    pub class: u8,
    pub mr: u8,
    pub ply: u32,
    pub n_cells: usize,
    pub trunk_sz: i32,
    pub channels: Vec<usize>,
    pub planes_2: Vec<f32>,
    pub cur: Vec<f32>,
    pub opp: Vec<f32>,
    pub moves: Vec<(i32, i32)>,
}

fn read_case_record(cur: &mut Cur) -> Result<CaseInput, String> {
    let case_id = cur.u32v("case_id")?;
    let kernel = cur.u8v("kernel")?;
    let encoding_id = cur.u8v("encoding_id")?;
    let class = cur.u8v("class")?;
    let mut c = CaseInput {
        case_id,
        kernel,
        encoding_id,
        class,
        mr: 0,
        ply: 0,
        n_cells: 0,
        trunk_sz: 0,
        channels: Vec::new(),
        planes_2: Vec::new(),
        cur: Vec::new(),
        opp: Vec::new(),
        moves: Vec::new(),
    };
    match kernel {
        K_STATE => {
            c.mr = cur.u8v("mr")?;
            c.ply = cur.u32v("ply")?;
            c.n_cells = cur.u32v("n_cells")? as usize;
            c.planes_2 = cur.f32v(2 * c.n_cells, "planes_2")?;
        }
        K_CHANNELS => {
            c.mr = cur.u8v("mr")?;
            c.ply = cur.u32v("ply")?;
            c.n_cells = cur.u32v("n_cells")? as usize;
            let nch = cur.u32v("n_channels")? as usize;
            for _ in 0..nch {
                c.channels.push(cur.u32v("channel")? as usize);
            }
            c.planes_2 = cur.f32v(2 * c.n_cells, "planes_2")?;
        }
        K_CHAIN => {
            c.n_cells = cur.u32v("n_cells")? as usize;
            c.trunk_sz = cur.i32v("trunk_sz")?;
            c.cur = cur.f32v(c.n_cells, "cur")?;
            c.opp = cur.f32v(c.n_cells, "opp")?;
        }
        K_TO_PLANES => {
            let nm = cur.u32v("n_moves")? as usize;
            for _ in 0..nm {
                c.moves.push((cur.i32v("q")?, cur.i32v("r")?));
            }
        }
        K_TO_PLANES_CHANNELS => {
            let nm = cur.u32v("n_moves")? as usize;
            for _ in 0..nm {
                c.moves.push((cur.i32v("q")?, cur.i32v("r")?));
            }
            let nch = cur.u32v("n_channels")? as usize;
            for _ in 0..nch {
                c.channels.push(cur.u32v("channel")? as usize);
            }
        }
        other => return Err(format!("unknown kernel code {other}")),
    }
    Ok(c)
}

/// Read inputs.bin (magic MEPI, version 1). Short read = loud named error.
pub fn read_inputs_bin(path: &Path) -> Result<Vec<CaseInput>, String> {
    let bytes = std::fs::read(path)
        .map_err(|e| format!("inputs.bin unreadable: {}: {e}", path.display()))?;
    let mut cur = Cur { b: &bytes, off: 0, name: path.display().to_string() };
    if cur.take(4, "magic")? != b"MEPI" {
        return Err(format!("{}: bad magic", path.display()));
    }
    if cur.u32v("version")? != 1 {
        return Err(format!("{}: bad version", path.display()));
    }
    let n = cur.u32v("n_cases")? as usize;
    let mut cases = Vec::with_capacity(n);
    for _ in 0..n {
        cases.push(read_case_record(&mut cur)?);
    }
    if cur.off != bytes.len() {
        return Err(format!("{}: trailing bytes", path.display()));
    }
    Ok(cases)
}

/// The 'out' field of one raw blob (`None` for a panic_case, n_fields=0).
#[derive(Debug)]
pub struct BlobOut {
    pub case_id: u32,
    pub input: CaseInput,
    pub payload: Option<Vec<u8>>,
    /// Absolute offset of the payload inside the file (for corruption diagnosis).
    pub payload_offset: usize,
}

/// Read one MEPB raw golden blob. Any short read is a loud error naming the file.
pub fn read_blob(path: &Path) -> Result<BlobOut, String> {
    let bytes = std::fs::read(path)
        .map_err(|e| format!("blob absent/unreadable: {}: {e}", path.display()))?;
    let mut cur = Cur { b: &bytes, off: 0, name: path.display().to_string() };
    if cur.take(4, "magic")? != b"MEPB" {
        return Err(format!("{}: bad magic", path.display()));
    }
    if cur.u32v("version")? != 1 {
        return Err(format!("{}: bad version", path.display()));
    }
    let case_id = cur.u32v("case_id")?;
    let input = read_case_record(&mut cur)?;
    if input.case_id != case_id {
        return Err(format!(
            "{}: embedded case_id {} != header {case_id}",
            path.display(),
            input.case_id
        ));
    }
    let n_fields = cur.u8v("n_fields")?;
    let payload = if n_fields == 0 {
        None
    } else {
        let name_len = cur.u8v("field name_len")? as usize;
        cur.take(name_len, "field name")?;
        cur.u8v("dtype")?;
        let ndim = cur.u8v("ndim")? as usize;
        for _ in 0..ndim {
            cur.u32v("dim")?;
        }
        let nbytes = cur.u64v("payload_nbytes")? as usize;
        let off = cur.off;
        let p = cur.take(nbytes, "payload")?.to_vec();
        return finish_blob(cur, &bytes, path, case_id, input, Some(p), off);
    };
    finish_blob(cur, &bytes, path, case_id, input, payload, 0)
}

fn finish_blob(
    cur: Cur,
    bytes: &[u8],
    path: &Path,
    case_id: u32,
    input: CaseInput,
    payload: Option<Vec<u8>>,
    payload_offset: usize,
) -> Result<BlobOut, String> {
    if cur.off != bytes.len() {
        return Err(format!("{}: trailing bytes", path.display()));
    }
    Ok(BlobOut { case_id, input, payload, payload_offset })
}

// ── NEW-side kernel output builder ───────────────────────────────────────────
fn enc_name(id: u8) -> &'static str {
    match id {
        0 => "v6",
        1 => "v6w25",
        2 => "v6_live2_ls",
        _ => panic!("unknown encoding id {id}"),
    }
}

fn state_board(mr: u8, ply: u32) -> Board {
    let mut b = Board::new();
    b.moves_remaining = mr;
    b.ply = Ply::new(ply);
    b
}

fn board_with_moves(moves: &[(i32, i32)]) -> Board {
    let mut b = Board::new();
    for &(q, r) in moves {
        b.apply_move(q, r).expect("golden move sequence must be legal");
    }
    b
}

fn f32_le(v: &[f32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for &x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

/// Drive the NEW ported free-fn kernel for one case and return the canonical
/// little-endian output bytes. PANICS for the multi-window `to_planes*` cases
/// (class PANIC) and, in a debug build, for the out-of-range-channel case
/// (class OOR_SKIP, `debug_assert!`) — callers guard those via `catch_unwind`.
#[must_use]
pub fn build_output(c: &CaseInput) -> Vec<u8> {
    match c.kernel {
        K_STATE => {
            let board = state_board(c.mr, c.ply);
            let mut out = vec![0.0f32; 18 * c.n_cells];
            encode_state_to_buffer(&board, &c.planes_2, &mut out);
            f32_le(&out)
        }
        K_CHANNELS => {
            let board = state_board(c.mr, c.ply);
            let mut out = vec![0.0f32; c.channels.len() * c.n_cells];
            encode_state_to_buffer_channels(&board, &c.planes_2, &mut out, &c.channels, c.n_cells);
            f32_le(&out)
        }
        K_CHAIN => {
            let mut out = vec![0.0f32; 6 * c.n_cells];
            encode_chain_planes(&c.cur, &c.opp, &mut out, c.n_cells, c.trunk_sz);
            f32_le(&out)
        }
        K_TO_PLANES => {
            let board = board_with_moves(&c.moves);
            let spec = lookup_or_panic(enc_name(c.encoding_id));
            f32_le(&to_planes(&board, spec))
        }
        K_TO_PLANES_CHANNELS => {
            let board = board_with_moves(&c.moves);
            let spec = lookup_or_panic(enc_name(c.encoding_id));
            f32_le(&to_planes_channels(&board, spec, &c.channels))
        }
        other => panic!("unknown kernel code {other}"),
    }
}
