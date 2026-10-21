// >300 justify (R8): the one module that may carry #[wasm_bindgen] (a census test pins it), so the browser surface
// cannot be split without a second bindgen home.

//! The browser engine's one FFI surface (wasm32 only): the step machine and `RustForward` behind `#[wasm_bindgen]`.

use serde_json::{json, Value};
use wasm_bindgen::prelude::*;

use mantis_search::mcts::{AuditConfig, AuditMode, TacticsConfig};

use crate::batch::{build_batch, Position, WebBatch};
use crate::cache::CacheRows;
use crate::forward::{NetOutputs, RustForward, TensorRow};
use crate::player::{
    Budget, Costs, DeployBlock, GameClock, Player, PlayerError, Pull, Rows, StoneResult, TurnResult,
};

fn named(name: &str, message: &str) -> JsValue {
    let error = js_sys::Error::new(message);
    error.set_name(name);
    error.into()
}

fn refused(e: &PlayerError) -> JsValue {
    named(e.name(), &e.to_string())
}

fn cells(flat: &[i32]) -> Result<Vec<(i32, i32)>, JsValue> {
    if !flat.len().is_multiple_of(2) {
        return Err(named("IllegalStone", "cells come as (q, r) pairs"));
    }
    Ok(flat.chunks(2).map(|c| (c[0], c[1])).collect())
}

fn finite(v: f32) -> Value {
    if v.is_finite() {
        json!(v)
    } else {
        Value::Null
    }
}

fn stone_json(s: &StoneResult) -> Value {
    json!({
        "cell": [s.cell.0, s.cell.1], "sims": s.sims, "stopped": s.stopped, "decided": s.decided,
        "inherited": s.inherited, "root_visits": s.root_visits, "root_value": s.root_value.map_or(Value::Null, finite),
        "top": s.top.iter().map(|c| json!({"cell": [c.cell.0, c.cell.1], "visits": c.visits,
            "prior": finite(c.prior), "q": finite(c.q)})).collect::<Vec<_>>(),
        "tactics": s.tactics.map_or(Value::Null, |t| t.rows().iter().map(|&(k, v)| (k.to_owned(), json!(v))).collect()),
    })
}

fn turn_json(t: &TurnResult) -> Value {
    json!({
        "stones": t.stones.iter().map(stone_json).collect::<Vec<_>>(), "sims_set": t.sims_set,
        "floor_over_clock": t.floor_over_clock, "budget_pool_capped": t.budget_pool_capped, "wall_ms": t.wall_ms, "served_leaves": t.served_leaves,
    })
}

fn rows_json(r: &Rows, c: &CacheRows) -> Value {
    json!({
        "stop_fired": r.stop_fired, "stop_saved": r.stop_saved, "select_calls": r.select_calls,
        "select_overlaps": r.select_overlaps, "select_network_leaves": r.select_network_leaves,
        "reuse_reroots": r.reuse_reroots, "reuse_inherited": r.reuse_inherited, "cache_calls": c.calls,
        "cache_positions": c.positions, "cache_hits": c.hits, "cache_served": c.served,
        "cache_in_batch_repeats": c.in_batch_repeats, "cache_all_hit_calls": c.all_hit_calls,
        "cache_evicted": c.evicted,
    })
}

/// One engine for one board: the host pulls its batches, runs a net on them, and pushes the outputs back.
#[wasm_bindgen]
pub struct Engine {
    player: Player,
    forward: Option<RustForward>,
    outputs: Option<NetOutputs>,
    turn: Option<TurnResult>,
}

#[wasm_bindgen]
impl Engine {
    /// `deploy.resolved`, flat: `tactics` `[leaf_turns, leaf_nodes, root_turns, root_nodes]`, `audit` `[turns, nodes, k, m, total]`.
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        encoding: &str,
        cpu_path: bool,
        c_puct: f32,
        fpu_reduction: f32,
        quiescence_enabled: bool,
        quiescence_blend_2: f32,
        leaf_batch_size: u32,
        tactics: Option<Vec<u32>>,
        audit: Option<Vec<u32>>,
        audit_mode: Option<String>,
    ) -> Result<Engine, JsValue> {
        let bad = |why: &str| named("Refused", &format!("deploy block: {why}"));
        let turns = |n: u32| u8::try_from(n).map_err(|_| bad("a turn count past 255"));
        let audit = match (audit.as_deref(), audit_mode.as_deref()) {
            (None, None) => None,
            (Some(&[t, nodes, k, m, total]), Some(mode)) => Some(AuditConfig {
                turns: turns(t)?,
                nodes: u64::from(nodes),
                k,
                m,
                total_nodes: u64::from(total),
                mode: match mode {
                    "hold" => AuditMode::Hold,
                    "inverted" => AuditMode::Inverted,
                    _ => return Err(bad("an audit mode neither hold nor inverted")),
                },
            }),
            _ => return Err(bad("an audit needs five numbers and a mode")),
        };
        let tactics = match tactics.as_deref() {
            None if audit.is_none() => None,
            Some(&[lt, ln, rt, rn]) => Some(TacticsConfig {
                leaf_turns: turns(lt)?,
                leaf_nodes: u64::from(ln),
                root_turns: turns(rt)?,
                root_nodes: u64::from(rn),
                audit,
            }),
            _ => return Err(bad("tactics need four numbers, and an audit needs tactics")),
        };
        let block = DeployBlock {
            c_puct,
            fpu_reduction,
            quiescence_enabled,
            quiescence_blend_2,
            leaf_batch_size: leaf_batch_size as usize,
            tactics,
        };
        let player = Player::new(block, encoding, cpu_path).map_err(|e| refused(&e))?;
        Ok(Engine {
            player,
            forward: None,
            outputs: None,
            turn: None,
        })
    }

    /// `RustForward` over the weights file: `names` one a line, `dims` `(rows, cols, offset)` for each.
    pub fn load_weights(
        &mut self,
        names: &str,
        dims: &[u32],
        floats: &[f32],
        node_feat_dim: u32,
    ) -> Result<(), JsValue> {
        let names: Vec<&str> = names.lines().collect();
        if dims.len() != 3 * names.len() {
            return Err(named(
                "Weights",
                "a tensor table needs three numbers a name",
            ));
        }
        let table: Vec<TensorRow> = names
            .iter()
            .zip(dims.chunks(3))
            .map(|(name, d)| TensorRow {
                name: (*name).to_owned(),
                rows: d[0] as usize,
                cols: d[1] as usize,
                offset: d[2] as usize,
            })
            .collect();
        let forward = RustForward::load(&table, floats, node_feat_dim as usize)
            .map_err(|e| named("Weights", &e.to_string()))?;
        self.forward = Some(forward);
        Ok(())
    }

    pub fn set_costs(&mut self, leaf_ms: f64, root_ms: f64) -> Result<(), JsValue> {
        self.player
            .set_costs(Costs { leaf_ms, root_ms })
            .map_err(|e| refused(&e))
    }

    #[wasm_bindgen(getter)]
    pub fn min_turn_ms(&self) -> Option<f64> {
        self.player.min_turn_ms()
    }

    /// A new game at `moves` (flat `q, r` pairs), its pool sized for `sims`, `turn_ms` or neither.
    pub fn new_game(
        &mut self,
        moves: &[i32],
        sims: Option<u32>,
        turn_ms: Option<f64>,
    ) -> Result<(), JsValue> {
        let clock = match (sims, turn_ms) {
            (Some(n), None) => GameClock::Sims(n),
            (None, Some(t)) => GameClock::TurnMs(t),
            (None, None) => GameClock::Unlimited,
            _ => return Err(named("Refused", "new_game: sims or turn_ms, not both")),
        };
        self.player
            .new_game(&cells(moves)?, clock)
            .map_err(|e| refused(&e))
    }

    pub fn play(&mut self, stones: &[i32]) -> Result<(), JsValue> {
        self.player.play(&cells(stones)?).map_err(|e| refused(&e))
    }

    pub fn begin_turn(
        &mut self,
        now_ms: f64,
        sims: Option<u32>,
        turn_ms: Option<f64>,
    ) -> Result<(), JsValue> {
        let budget = match (sims, turn_ms) {
            (Some(n), None) => Budget::Sims(n),
            (None, Some(t)) => Budget::Clock { turn_ms: t },
            (None, None) => Budget::Unlimited,
            _ => return Err(named("Refused", "begin_turn: sims or turn_ms, not both")),
        };
        self.turn = None;
        self.player
            .begin_turn(budget, now_ms)
            .map_err(|e| refused(&e))
    }

    /// `true` when a batch waits for the net; `false` when the turn ended and `turn_json` holds it.
    pub fn next_batch(&mut self, now_ms: f64) -> Result<bool, JsValue> {
        self.outputs = None;
        match self.player.next_batch(now_ms).map_err(|e| refused(&e))? {
            Pull::Batch(_) => Ok(true),
            Pull::Turn(result) => {
                self.turn = Some(result);
                Ok(false)
            }
        }
    }

    fn batch(&self) -> Result<&WebBatch, JsValue> {
        self.player
            .pending()
            .ok_or_else(|| named("OutOfStep", "no batch is outstanding"))
    }

    /// A copy of the pending batch, for a host backend.
    pub fn pending_batch(&self) -> Result<Batch, JsValue> {
        Ok(Batch {
            inner: self.batch()?.clone(),
        })
    }

    /// `RustForward` on any batch: the on-device witness and W2 read it.
    pub fn forward_batch(&self, batch: &Batch) -> Result<Outputs, JsValue> {
        let forward = self
            .forward
            .as_ref()
            .ok_or_else(|| named("Weights", "no weights loaded"))?;
        let out = forward
            .forward(&batch.inner)
            .map_err(|e| named("Weights", &e.to_string()))?;
        Ok(Outputs { inner: out })
    }

    /// `RustForward` on the pending batch in place; `submit_pending` hands its outputs to the search.
    pub fn forward_pending(&mut self) -> Result<(), JsValue> {
        let forward = self
            .forward
            .as_ref()
            .ok_or_else(|| named("Weights", "no weights loaded"))?;
        let outputs = forward
            .forward(self.batch()?)
            .map_err(|e| named("Weights", &e.to_string()))?;
        self.outputs = Some(outputs);
        Ok(())
    }

    pub fn submit_pending(&mut self, now_ms: f64) -> Result<(), JsValue> {
        let out = self
            .outputs
            .take()
            .ok_or_else(|| named("OutOfStep", "no forward ran on the pending batch"))?;
        self.player
            .submit(&out.probs, &out.value, now_ms)
            .map_err(|e| refused(&e))
    }

    /// A host backend's `probs [L]` and `value [G]` for the pending batch.
    pub fn submit(&mut self, probs: &[f32], values: &[f32], now_ms: f64) -> Result<(), JsValue> {
        self.player
            .submit(probs, values, now_ms)
            .map_err(|e| refused(&e))
    }

    pub fn abort(&mut self) {
        self.outputs = None;
        self.player.abort();
    }

    /// The last turn's result as JSON, or `undefined` before one ends.
    pub fn turn_json(&self) -> Option<String> {
        self.turn.as_ref().map(|t| turn_json(t).to_string())
    }

    pub fn rows_json(&self) -> String {
        let (r, c) = self.player.rows();
        rows_json(&r, &c).to_string()
    }
}

/// The module's linear memory in bytes: its high-water mark, since a wasm memory never shrinks.
#[wasm_bindgen]
#[must_use]
pub fn memory_bytes() -> f64 {
    core::arch::wasm32::memory_size(0) as f64 * 65_536.0
}

/// Contract #13's batch in JS: built from positions by the engine, copied from the pending one, or given as arrays.
#[wasm_bindgen]
pub struct Batch {
    inner: WebBatch,
}

#[wasm_bindgen]
impl Batch {
    /// A batch from its arrays; the forward checks them before it reads one.
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        x: Vec<f32>,
        src: Vec<i32>,
        dst: Vec<i32>,
        code: Vec<i32>,
        legal: Vec<i32>,
        node_graph: Vec<i32>,
        legal_graph: Vec<i32>,
        divisor: Vec<i32>,
        real: Vec<i32>,
        mean_mask: Vec<i32>,
        mean_den: Vec<i32>,
    ) -> Batch {
        let n_graphs = mean_den.len();
        let inner = WebBatch {
            n_graphs,
            x,
            src,
            dst,
            code,
            legal,
            node_graph,
            legal_graph,
            divisor,
            real,
            mean_mask,
            mean_den,
            leaves: Vec::new(),
        };
        Batch { inner }
    }

    #[wasm_bindgen(getter)]
    pub fn n_graphs(&self) -> usize {
        self.inner.n_graphs
    }

    pub fn x(&self) -> Vec<f32> {
        self.inner.x.clone()
    }

    /// One int32 array by its contract #13 name.
    pub fn i32(&self, name: &str) -> Result<Vec<i32>, JsValue> {
        let b = &self.inner;
        Ok(match name {
            "src" => &b.src,
            "dst" => &b.dst,
            "code" => &b.code,
            "legal" => &b.legal,
            "node_graph" => &b.node_graph,
            "legal_graph" => &b.legal_graph,
            "divisor" => &b.divisor,
            "real" => &b.real,
            "mean_mask" => &b.mean_mask,
            "mean_den" => &b.mean_den,
            _ => return Err(named("Refused", &format!("no batch array {name:?}"))),
        }
        .clone())
    }
}

/// The engine's own batch of `positions`: `stones` flat `(q, r, p)` triples, `counts` stones a position.
#[wasm_bindgen]
pub fn web_batch(
    encoding: &str,
    stones: &[i32],
    counts: &[u32],
    sides: &[i32],
    lefts: &[i32],
) -> Result<Batch, JsValue> {
    let spec = mantis_encoding::lookup(encoding)
        .ok_or_else(|| named("Refused", &format!("unknown encoding {encoding:?}")))?;
    if !stones.len().is_multiple_of(3) || counts.len() != sides.len() || counts.len() != lefts.len()
    {
        return Err(named(
            "Batch",
            "stones come as (q, r, p) triples, one count, side and left a position",
        ));
    }
    let mut triples = stones
        .chunks(3)
        .map(|s| (i64::from(s[0]), i64::from(s[1]), i64::from(s[2])));
    let mut positions: Vec<Position> = Vec::with_capacity(counts.len());
    for ((&n, &side), &left) in counts.iter().zip(sides).zip(lefts) {
        let own: Vec<_> = triples.by_ref().take(n as usize).collect();
        if own.len() != n as usize {
            return Err(named(
                "Batch",
                "the counts ask for more stones than were given",
            ));
        }
        positions.push((own, i64::from(side), i64::from(left)));
    }
    if triples.next().is_some() {
        return Err(named(
            "Batch",
            "more stones were given than the counts ask for",
        ));
    }
    let inner = build_batch(&positions, spec).map_err(|e| named("Batch", &e.to_string()))?;
    Ok(Batch { inner })
}

/// The net's outputs for one batch.
#[wasm_bindgen]
pub struct Outputs {
    inner: NetOutputs,
}

#[wasm_bindgen]
impl Outputs {
    pub fn logits(&self) -> Vec<f32> {
        self.inner.logits.clone()
    }

    pub fn probs(&self) -> Vec<f32> {
        self.inner.probs.clone()
    }

    pub fn value(&self) -> Vec<f32> {
        self.inner.value.clone()
    }
}

/// The compiled encoding registry's sha256: a manifest naming another was exported for another engine.
#[wasm_bindgen]
#[must_use]
pub fn registry_sha() -> String {
    mantis_encoding::registry_sha_hex().to_owned()
}

/// The player's own constants as the manifest's `deploy.player` names them, as JSON.
#[wasm_bindgen]
#[must_use]
pub fn player_constants() -> String {
    json!({
        "early_stop": crate::player::EARLY_STOP, "tree_reuse": crate::player::TREE_REUSE,
        "cache_entries": crate::cache::CACHE_ENTRIES, "virtual_loss": mantis_search::VIRTUAL_LOSS_PENALTY,
    })
    .to_string()
}
