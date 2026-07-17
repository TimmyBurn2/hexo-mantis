//! Encoding registry — TOML parser + LazyLock lookup + sha handshake.
//!
//! `registry.toml` is embedded at compile time via `include_str!`; the first
//! call to `lookup`/`all_specs` parses it, validates every entry, and builds a
//! `HashMap<&'static str, &'static RegistrySpec>` whose values live (leaked) for
//! the process lifetime.
//!
//! Parse failures panic with a multi-line diagnostic listing every offending
//! field. Init-time panic is acceptable — registry parse failure is
//! unrecoverable (a Board cannot be constructed without a valid encoding).

use std::collections::HashMap;
use std::sync::LazyLock;

use sha2::{Digest, Sha256};
use toml::Value;

use crate::spec::RegistrySpec;

mod parse;
use parse::parse_one;

/// Canonical registry source. Embedded at compile time so the binary is
/// self-contained — runtime never reads from disk.
static REGISTRY_TOML: &str = include_str!("../registry.toml");

static REGISTRY: LazyLock<HashMap<&'static str, &'static RegistrySpec>> = LazyLock::new(load);

static REGISTRY_SHA_HEX: LazyLock<String> = LazyLock::new(|| {
    let mut s = String::with_capacity(64);
    for b in registry_sha() {
        s.push_str(&format!("{b:02x}"));
    }
    s
});

/// Look up an encoding by name. Returns `None` if unknown.
#[must_use]
pub fn lookup(name: &str) -> Option<&'static RegistrySpec> {
    REGISTRY.get(name).copied()
}

/// Look up an encoding by name, panicking with a helpful message on miss.
#[must_use]
pub fn lookup_or_panic(name: &str) -> &'static RegistrySpec {
    if let Some(s) = lookup(name) {
        s
    } else {
        let mut known: Vec<&str> = REGISTRY.keys().copied().collect();
        known.sort_unstable();
        panic!("encoding registry: unknown encoding {name:?}; registered: {known:?}");
    }
}

/// Iterate all registered specs (order is HashMap-arbitrary).
pub fn all_specs() -> impl Iterator<Item = &'static RegistrySpec> {
    REGISTRY.values().copied()
}

/// Parse + validate ONE `[encodings.<name>]` block from a TOML fragment (the
/// body key/value pairs as a top-level table). Public so tooling and the
/// red-team parser tests can validate an encoding block without going through
/// the embedded registry. Runs the SAME `parse_one` + `validate()` path as the
/// registry loader (unknown-key reject, representation-required, collect-all).
///
/// # Errors
/// Returns the named parse error (or, if parse succeeds, the validator error).
pub fn parse_encoding_toml(name: &str, body_toml: &str) -> Result<RegistrySpec, String> {
    let value: Value = toml::from_str(body_toml).map_err(|e| format!("TOML parse error: {e}"))?;
    let spec = parse_one(name, &value)?;
    spec.validate()?;
    Ok(spec)
}

/// SHA-256 of the embedded `registry.toml` bytes. Deterministic over the
/// compiled source; the bridge re-exports it as `_engine.registry_sha()` for the
/// dev/test on-disk-vs-compiled handshake.
#[must_use]
pub fn registry_sha() -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(REGISTRY_TOML.as_bytes());
    hasher.finalize().into()
}

/// Lowercase hex of `registry_sha()` (stable `&'static str` convenience).
#[must_use]
pub fn registry_sha_hex() -> &'static str {
    REGISTRY_SHA_HEX.as_str()
}

// --------------------------------------------------------------------------
// TOML parsing — runs once via LazyLock.
// --------------------------------------------------------------------------

fn load() -> HashMap<&'static str, &'static RegistrySpec> {
    let root: Value = toml::from_str(REGISTRY_TOML)
        .unwrap_or_else(|e| panic!("encoding registry: TOML parse error: {e}"));

    let encodings = root
        .get("encodings")
        .and_then(Value::as_table)
        .unwrap_or_else(|| panic!("encoding registry: missing top-level [encodings] table"));

    let mut errors: Vec<String> = Vec::new();
    let mut map: HashMap<&'static str, &'static RegistrySpec> = HashMap::new();

    for (name, body) in encodings {
        match parse_one(name, body) {
            Ok(spec) => {
                // SAFETY: allocated by Box::leak in registry::load();
                // stable for process lifetime — registry is one-shot init.
                let leaked: &'static RegistrySpec = Box::leak(Box::new(spec));
                if let Err(e) = leaked.validate() {
                    errors.push(e);
                    continue;
                }
                map.insert(leaked.name, leaked);
            }
            Err(e) => errors.push(e),
        }
    }

    assert!(
        errors.is_empty(),
        "encoding registry: parse/validation failed for {} entries:\n{}",
        errors.len(),
        errors
            .iter()
            .map(|e| format!("  * {e}"))
            .collect::<Vec<_>>()
            .join("\n")
    );

    map
}

fn leak_str(s: &str) -> &'static str {
    // SAFETY: allocated by Box::leak in registry::load();
    // stable for process lifetime — registry is one-shot init.
    Box::leak(s.to_string().into_boxed_str())
}

/// Parse a TOML field that is either an integer or the string `"none"`.
/// `Ok(Some(int))`, `Ok(None)` for `"none"`, or `Err(msg)` if missing / wrong shape.
fn parse_int_or_none(v: Option<&Value>) -> Result<Option<usize>, String> {
    match v {
        Some(Value::Integer(i)) => {
            if *i < 0 {
                Err(format!("integer must be >= 0; got {i}"))
            } else {
                Ok(Some(*i as usize))
            }
        }
        Some(Value::String(s)) if s == "none" => Ok(None),
        Some(Value::String(s)) => Err(format!(
            "string value must be \"none\" sentinel; got {s:?}"
        )),
        Some(other) => Err(format!(
            "must be integer or string \"none\"; got {:?}",
            other.type_str()
        )),
        None => Err("missing key".to_string()),
    }
}
