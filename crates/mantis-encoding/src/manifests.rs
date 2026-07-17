//! Corpus / anchor / held-out pins — loud-parsed sibling TOML.
//!
//! Tamper discipline: the loader hard-errors on unknown key / missing required /
//! malformed sha / a held-out sha colliding with a corpus pin — never a silent
//! accept. Parsing collects ALL violations into one message (mirrors the
//! registry validator).

use std::collections::BTreeMap;
use std::sync::LazyLock;

use toml::Value;

static MANIFESTS_TOML: &str = include_str!("manifests.toml");

static MANIFESTS: LazyLock<Manifests> = LazyLock::new(|| {
    parse_manifests(MANIFESTS_TOML)
        .unwrap_or_else(|e| panic!("manifests.toml: parse/validation failed:\n{e}"))
});

/// One encoding's artifact pins.
#[derive(Debug, Clone)]
pub struct EncodingPins {
    pub corpus_path: String,
    pub anchor_path: String,
    pub corpus_sha256: Option<String>,
}

/// One held-out set entry (never enters a training load).
#[derive(Debug, Clone)]
pub struct HeldOut {
    pub label: String,
    pub sha256: String,
    pub size_bytes: u64,
}

/// Parsed manifests.
#[derive(Debug, Clone)]
pub struct Manifests {
    pub schema_version: u32,
    pub encodings: BTreeMap<String, EncodingPins>,
    pub held_out: Vec<HeldOut>,
}

fn is_hex64(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Loud parser: `toml::from_str` → collect-all-errors. Public so a tamper test
/// can feed a mutated copy and confirm a loud named error (never a silent accept).
pub fn parse_manifests(text: &str) -> Result<Manifests, String> {
    let root: Value = toml::from_str(text).map_err(|e| format!("TOML parse error: {e}"))?;
    let table = root
        .as_table()
        .ok_or_else(|| "top-level is not a table".to_string())?;

    let mut errs: Vec<String> = Vec::new();

    // Unknown top-level keys.
    const TOP_KEYS: [&str; 3] = ["schema_version", "encodings", "held_out"];
    for key in table.keys() {
        if !TOP_KEYS.contains(&key.as_str()) {
            errs.push(format!("unknown top-level key {key:?}"));
        }
    }

    let schema_version = match table.get("schema_version").and_then(Value::as_integer) {
        Some(v) if v >= 0 => v as u32,
        Some(v) => {
            errs.push(format!("schema_version must be >= 0; got {v}"));
            0
        }
        None => {
            errs.push("missing or non-integer key \"schema_version\"".to_string());
            0
        }
    };

    let mut encodings: BTreeMap<String, EncodingPins> = BTreeMap::new();
    match table.get("encodings").map(Value::as_table) {
        Some(Some(enc_table)) => {
            for (name, body) in enc_table {
                match parse_encoding_block(name, body) {
                    Ok(pins) => {
                        encodings.insert(name.clone(), pins);
                    }
                    Err(mut e) => errs.append(&mut e),
                }
            }
        }
        Some(None) => errs.push("\"encodings\" is not a table".to_string()),
        None => errs.push("missing required key \"encodings\"".to_string()),
    }

    let mut held_out: Vec<HeldOut> = Vec::new();
    match table.get("held_out") {
        Some(Value::Array(arr)) => {
            for (i, body) in arr.iter().enumerate() {
                match parse_held_out(i, body) {
                    Ok(h) => held_out.push(h),
                    Err(mut e) => errs.append(&mut e),
                }
            }
        }
        Some(_) => errs.push("\"held_out\" is not an array".to_string()),
        None => {} // held_out is optional
    }

    // Overlap invariant: a held-out sha must NOT collide with any corpus pin.
    let corpus_pins: std::collections::BTreeSet<&str> = encodings
        .values()
        .filter_map(|e| e.corpus_sha256.as_deref())
        .collect();
    for h in &held_out {
        if corpus_pins.contains(h.sha256.as_str()) {
            errs.push(format!(
                "held-out sha {} ({}) collides with a registered corpus pin",
                h.sha256, h.label
            ));
        }
    }

    if errs.is_empty() {
        Ok(Manifests { schema_version, encodings, held_out })
    } else {
        Err(errs
            .iter()
            .map(|e| format!("  - {e}"))
            .collect::<Vec<_>>()
            .join("\n"))
    }
}

fn parse_encoding_block(name: &str, body: &Value) -> Result<EncodingPins, Vec<String>> {
    let mut errs = Vec::new();
    let table = match body.as_table() {
        Some(t) => t,
        None => return Err(vec![format!("[encodings.{name}]: not a table")]),
    };
    const KEYS: [&str; 3] = ["corpus_path", "anchor_path", "corpus_sha256"];
    for key in table.keys() {
        if !KEYS.contains(&key.as_str()) {
            errs.push(format!("[encodings.{name}]: unknown key {key:?}"));
        }
    }
    let corpus_path = match table.get("corpus_path").and_then(Value::as_str) {
        Some(s) => Some(s.to_string()),
        None => {
            errs.push(format!("[encodings.{name}]: missing or non-string key \"corpus_path\""));
            None
        }
    };
    let anchor_path = match table.get("anchor_path").and_then(Value::as_str) {
        Some(s) => Some(s.to_string()),
        None => {
            errs.push(format!("[encodings.{name}]: missing or non-string key \"anchor_path\""));
            None
        }
    };
    let corpus_sha256 = match table.get("corpus_sha256") {
        None => None,
        Some(Value::String(s)) if is_hex64(s) => Some(s.clone()),
        Some(Value::String(s)) => {
            errs.push(format!(
                "[encodings.{name}].corpus_sha256: malformed sha256 {s:?} (want 64 lowercase hex)"
            ));
            None
        }
        Some(_) => {
            errs.push(format!("[encodings.{name}].corpus_sha256: must be a string"));
            None
        }
    };

    if errs.is_empty() {
        Ok(EncodingPins {
            corpus_path: corpus_path.unwrap(),
            anchor_path: anchor_path.unwrap(),
            corpus_sha256,
        })
    } else {
        Err(errs)
    }
}

fn parse_held_out(i: usize, body: &Value) -> Result<HeldOut, Vec<String>> {
    let mut errs = Vec::new();
    let table = match body.as_table() {
        Some(t) => t,
        None => return Err(vec![format!("[[held_out]][{i}]: not a table")]),
    };
    const KEYS: [&str; 3] = ["label", "sha256", "size_bytes"];
    for key in table.keys() {
        if !KEYS.contains(&key.as_str()) {
            errs.push(format!("[[held_out]][{i}]: unknown key {key:?}"));
        }
    }
    let label = match table.get("label").and_then(Value::as_str) {
        Some(s) => Some(s.to_string()),
        None => {
            errs.push(format!("[[held_out]][{i}]: missing or non-string key \"label\""));
            None
        }
    };
    let sha256 = match table.get("sha256") {
        Some(Value::String(s)) if is_hex64(s) => Some(s.clone()),
        Some(Value::String(s)) => {
            errs.push(format!("[[held_out]][{i}].sha256: malformed sha256 {s:?}"));
            None
        }
        _ => {
            errs.push(format!("[[held_out]][{i}]: missing or malformed key \"sha256\""));
            None
        }
    };
    let size_bytes = match table.get("size_bytes").and_then(Value::as_integer) {
        Some(v) if v >= 0 => Some(v as u64),
        Some(v) => {
            errs.push(format!("[[held_out]][{i}].size_bytes must be >= 0; got {v}"));
            None
        }
        None => {
            errs.push(format!("[[held_out]][{i}]: missing or non-integer key \"size_bytes\""));
            None
        }
    };
    if errs.is_empty() {
        Ok(HeldOut {
            label: label.unwrap(),
            sha256: sha256.unwrap(),
            size_bytes: size_bytes.unwrap(),
        })
    } else {
        Err(errs)
    }
}

// ── public accessors (over the embedded, validated manifest) ─────────────────

/// Corpus path for a registered encoding (repo-relative).
#[must_use]
pub fn corpus_path(name: &str) -> Option<&'static str> {
    MANIFESTS.encodings.get(name).map(|e| e.corpus_path.as_str())
}

/// Anchor checkpoint path for a registered encoding (repo-relative).
#[must_use]
pub fn anchor_path(name: &str) -> Option<&'static str> {
    MANIFESTS.encodings.get(name).map(|e| e.anchor_path.as_str())
}

/// Corpus sha256 launch-pin for a registered encoding (`None` = unpinned).
#[must_use]
pub fn corpus_sha_pin(name: &str) -> Option<&'static str> {
    MANIFESTS
        .encodings
        .get(name)
        .and_then(|e| e.corpus_sha256.as_deref())
}

/// The registered encoding names carrying pins (sorted).
#[must_use]
pub fn pinned_encoding_names() -> Vec<&'static str> {
    MANIFESTS.encodings.keys().map(String::as_str).collect()
}

/// All held-out set shas.
#[must_use]
pub fn held_out_shas() -> Vec<&'static str> {
    MANIFESTS.held_out.iter().map(|h| h.sha256.as_str()).collect()
}

/// All held-out set byte sizes.
#[must_use]
pub fn held_out_sizes() -> Vec<u64> {
    MANIFESTS.held_out.iter().map(|h| h.size_bytes).collect()
}

/// Assert a sha is NOT a held-out set (a training load must never touch one).
///
/// # Errors
/// Returns a named error if `sha` matches any held-out set entry.
pub fn assert_not_heldout(sha: &str) -> Result<(), String> {
    if let Some(h) = MANIFESTS.held_out.iter().find(|h| h.sha256 == sha) {
        Err(format!(
            "sha {sha} is the held-out set {:?} — it must never enter a training load",
            h.label
        ))
    } else {
        Ok(())
    }
}
