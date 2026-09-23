//! Helpers shared by the integration tests and benches; a subdirectory, so cargo builds no
//! binary for it, and a bench pulls it in with `#[path]`.

#![allow(dead_code)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use mantis_selfplay::queues::GraphQueue;
use mantis_selfplay::records::assemble_ls_from_gnn_probs;

/// One SplitMix64 step: the deterministic stream every seeded corpus here draws from.
pub fn splitmix64(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A uniform-prior mock inference server on the graph queue, through the PRODUCTION
/// `assemble_ls_from_gnn_probs`: every leaf is answered, popping up to `pop_max` per batch.
pub fn spawn_uniform_producer(
    queue: GraphQueue,
    n_actions: usize,
    served: Arc<AtomicUsize>,
    pop_max: usize,
) -> JoinHandle<()> {
    thread::spawn(move || loop {
        let batch = queue.pop_graph_batch(pop_max, 5);
        if batch.is_empty() {
            if queue.is_closed() {
                break;
            }
            continue;
        }
        let mut ids = Vec::with_capacity(batch.len());
        let mut results = Vec::with_capacity(batch.len());
        for (id, g) in batch {
            let coords: Vec<(i32, i32)> = g
                .legal_node_gather
                .iter()
                .map(|&row| {
                    (
                        g.node_coords[row as usize * 2],
                        g.node_coords[row as usize * 2 + 1],
                    )
                })
                .collect();
            let n = coords.len();
            let probs = vec![1.0f32 / n.max(1) as f32; n];
            ids.push(id);
            results.push(
                assemble_ls_from_gnn_probs(n_actions, &probs, &g.policy_scatter_index.0, &coords)
                    .map(|ls| (ls, 0.0f32)),
            );
        }
        served.fetch_add(ids.len(), Ordering::Relaxed);
        queue.submit_graph_results(&ids, results);
    })
}

/// An `eval_selfplay_parity` fixture, minted FLAT (numbers, short strings, flat arrays) so
/// this reader needs no JSON dependency.
pub fn fixture_text(name: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/eval_selfplay_parity")
        .join(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("fixture {} unreadable: {e}", path.display()))
}

/// The raw text of the value that follows `"key":`. Panics (never defaults) on absence —
/// a fixture that lost a key must fail loudly, not silently read a zero.
pub fn value_of<'a>(src: &'a str, key: &str) -> &'a str {
    let needle = format!("\"{key}\":");
    let at = src
        .find(&needle)
        .unwrap_or_else(|| panic!("fixture key {key:?} absent"));
    let rest = src[at + needle.len()..].trim_start();
    if let Some(inner) = rest.strip_prefix('[') {
        let end = inner
            .find(']')
            .unwrap_or_else(|| panic!("unterminated array for {key:?}"));
        &inner[..end]
    } else if let Some(inner) = rest.strip_prefix('"') {
        let end = inner
            .find('"')
            .unwrap_or_else(|| panic!("unterminated string for {key:?}"));
        &inner[..end]
    } else {
        let end = rest.find([',', '\n', '}']).unwrap_or(rest.len());
        rest[..end].trim_end()
    }
}

pub fn ints(src: &str, key: &str) -> Vec<i64> {
    value_of(src, key)
        .split(',')
        .map(|t| {
            t.trim()
                .parse::<i64>()
                .unwrap_or_else(|e| panic!("fixture {key:?}: {t:?} is not an integer ({e})"))
        })
        .collect()
}

pub fn floats(src: &str, key: &str) -> Vec<f64> {
    value_of(src, key)
        .split(',')
        .map(|t| {
            t.trim()
                .parse::<f64>()
                .unwrap_or_else(|e| panic!("fixture {key:?}: {t:?} is not a float ({e})"))
        })
        .collect()
}

pub fn scalar(src: &str, key: &str) -> i64 {
    let raw = value_of(src, key);
    raw.trim()
        .parse::<i64>()
        .unwrap_or_else(|e| panic!("fixture {key:?}: {raw:?} is not an integer ({e})"))
}

pub fn pairs(flat: &[i64]) -> Vec<(i32, i32)> {
    assert_eq!(flat.len() % 2, 0, "coord array must be pairs");
    flat.chunks_exact(2)
        .map(|c| (c[0] as i32, c[1] as i32))
        .collect()
}
