//! The net a manifest binds: its deploy block, its encoding, and `RustForward` over the weights file its sha256 names.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use serde_json::Value;
use sha2::{Digest, Sha256};

use mantis_player::cache::CACHE_ENTRIES;
use mantis_player::forward::{RustForward, TensorRow};
use mantis_player::player::{DeployBlock, EARLY_STOP, TREE_REUSE};
use mantis_search::mcts::{AuditConfig, AuditMode, TacticsConfig};
use mantis_search::VIRTUAL_LOSS_PENALTY;

use crate::CliError;

const FORMAT: &str = "mantis-web-manifest-v2";

fn hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .fold(String::new(), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        })
}

fn at<'a>(m: &'a Value, path: &str) -> Result<&'a Value, CliError> {
    m.pointer(path)
        .ok_or_else(|| CliError::Manifest(format!("no `{path}`")))
}

fn num(m: &Value, path: &str) -> Result<f64, CliError> {
    at(m, path)?
        .as_f64()
        .ok_or_else(|| CliError::Manifest(format!("`{path}` is not a number")))
}

fn whole<T: TryFrom<u64>>(m: &Value, path: &str) -> Result<T, CliError> {
    let bad = || CliError::Manifest(format!("`{path}` is not a count this engine holds"));
    at(m, path)?
        .as_u64()
        .and_then(|n| T::try_from(n).ok())
        .ok_or_else(bad)
}

fn text<'a>(m: &'a Value, path: &str) -> Result<&'a str, CliError> {
    at(m, path)?
        .as_str()
        .ok_or_else(|| CliError::Manifest(format!("`{path}` is not a string")))
}

fn flag(m: &Value, path: &str) -> Result<bool, CliError> {
    at(m, path)?
        .as_bool()
        .ok_or_else(|| CliError::Manifest(format!("`{path}` is not a boolean")))
}

#[allow(clippy::cast_possible_truncation)] // the manifest's floats were written from f32
fn block(m: &Value) -> Result<DeployBlock, CliError> {
    let r = "/deploy/resolved";
    if text(m, &format!("{r}/search_kind"))? != "puct" {
        return Err(CliError::Manifest(
            "the engine carries PUCT only".to_owned(),
        ));
    }
    let t = format!("{r}/tactics");
    let tactics = if at(m, &t)?.is_null() {
        None
    } else {
        if text(m, &format!("{t}/kind"))? != "strict_turn" {
            return Err(CliError::Manifest(
                "tactics kind is not strict_turn".to_owned(),
            ));
        }
        let a = format!("{t}/audit");
        let audit = if at(m, &a)?.is_null() {
            None
        } else {
            let mode = match text(m, &format!("{a}/mode"))? {
                "hold" => AuditMode::Hold,
                "inverted" => AuditMode::Inverted,
                other => return Err(CliError::Manifest(format!("audit mode {other:?}"))),
            };
            Some(AuditConfig {
                turns: whole(m, &format!("{a}/turns"))?,
                nodes: whole(m, &format!("{a}/nodes"))?,
                k: whole(m, &format!("{a}/k"))?,
                m: whole(m, &format!("{a}/m"))?,
                total_nodes: whole(m, &format!("{a}/total_nodes"))?,
                mode,
            })
        };
        Some(TacticsConfig {
            leaf_turns: whole(m, &format!("{t}/leaf_turns"))?,
            leaf_nodes: whole(m, &format!("{t}/leaf_nodes"))?,
            root_turns: whole(m, &format!("{t}/root_turns"))?,
            root_nodes: whole(m, &format!("{t}/root_nodes"))?,
            audit,
        })
    };
    Ok(DeployBlock {
        c_puct: num(m, &format!("{r}/c_puct"))? as f32,
        fpu_reduction: num(m, &format!("{r}/fpu_reduction"))? as f32,
        quiescence_enabled: flag(m, &format!("{r}/quiescence_enabled"))?,
        quiescence_blend_2: num(m, &format!("{r}/quiescence_blend_2"))? as f32,
        leaf_batch_size: whole(m, &format!("{r}/leaf_batch_size"))?,
        tactics,
    })
}

/// The net the manifest binds: its block, its encoding and the forward over the weights file its sha256 names.
pub struct Net {
    pub stem: String,
    pub manifest_sha: String,
    pub weights_sha: String,
    pub encoding: String,
    pub block: DeployBlock,
    pub forward: RustForward,
}

#[allow(clippy::float_cmp)] // the manifest carries the engine's own constant, written exactly
pub fn load(path: &Path) -> Result<Net, CliError> {
    let bytes =
        std::fs::read(path).map_err(|e| CliError::Io(format!("{}: {e}", path.display())))?;
    let m: Value = serde_json::from_slice(&bytes).map_err(|e| CliError::Manifest(e.to_string()))?;
    if text(&m, "/format")? != FORMAT {
        return Err(CliError::Manifest(format!("the format is not {FORMAT}")));
    }
    if text(&m, "/encoding/registry_sha")? != mantis_encoding::registry_sha_hex() {
        return Err(CliError::Manifest(
            "exported for another encoding registry".to_owned(),
        ));
    }
    let constants = [
        (
            "early_stop",
            flag(&m, "/deploy/player/early_stop")? == EARLY_STOP,
        ),
        (
            "tree_reuse",
            flag(&m, "/deploy/player/tree_reuse")? == TREE_REUSE,
        ),
        (
            "cache_entries",
            whole::<usize>(&m, "/deploy/player/cache_entries")? == CACHE_ENTRIES,
        ),
        (
            "virtual_loss",
            num(&m, "/deploy/player/virtual_loss")? == f64::from(VIRTUAL_LOSS_PENALTY),
        ),
    ];
    if let Some((name, _)) = constants.iter().find(|(_, same)| !same) {
        return Err(CliError::Manifest(format!(
            "`deploy.player.{name}` is not this engine's"
        )));
    }
    let encoding = text(&m, "/encoding/name")?.to_owned();
    let files = at(&m, "/files")?.as_array().map_or(&[][..], Vec::as_slice);
    let file = files.iter().find(|f| f.get("tensors").is_some());
    let file = file.ok_or_else(|| CliError::Manifest("no weights file".to_owned()))?;
    let name = text(file, "/name")?;
    if !flag(file, "/passes_its_line")? {
        return Err(CliError::Weights(format!("{name} misses its witness line")));
    }
    let wpath = path.with_file_name(name);
    let raw =
        std::fs::read(&wpath).map_err(|e| CliError::Io(format!("{}: {e}", wpath.display())))?;
    let weights_sha = hex(&raw);
    if weights_sha != text(file, "/sha256")? || raw.len() != whole::<usize>(file, "/bytes")? {
        return Err(CliError::Weights(format!(
            "{name} is not the file the manifest binds"
        )));
    }
    let table = at(file, "/tensors")?
        .as_array()
        .map_or(&[][..], Vec::as_slice)
        .iter();
    let table = table
        .map(|t| {
            Ok(TensorRow {
                name: text(t, "/name")?.to_owned(),
                rows: whole(t, "/rows")?,
                cols: whole(t, "/cols")?,
                offset: whole(t, "/offset")?,
            })
        })
        .collect::<Result<Vec<_>, CliError>>()?;
    let floats: Vec<f32> = raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let forward = RustForward::load(&table, &floats, whole(&m, "/encoding/node_feat_dim")?)
        .map_err(CliError::Forward)?;
    let file_name = path
        .file_name()
        .map(|f| f.to_string_lossy().into_owned())
        .unwrap_or_default();
    Ok(Net {
        stem: file_name.trim_end_matches(".manifest.json").to_owned(),
        manifest_sha: hex(&bytes),
        weights_sha,
        encoding,
        block: block(&m)?,
        forward,
    })
}

/// The one `*.manifest.json` beside the executable, as the release tarball lays it out.
pub fn beside_binary() -> Result<PathBuf, CliError> {
    let exe = std::env::current_exe()?;
    let dir = exe
        .parent()
        .ok_or_else(|| CliError::Usage("no directory holds the binary".to_owned()))?;
    let found: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.to_string_lossy().ends_with(".manifest.json"))
        .collect();
    match found.as_slice() {
        [one] => Ok(one.clone()),
        _ => Err(CliError::Usage(format!(
            "{} manifests beside the binary; name one with --net",
            found.len()
        ))),
    }
}
