//! The web player's Python surface: `web_batch` and `RustForward` for the export's witness, `Player` for W3.

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use mantis_player::batch::WebBatch;
use mantis_player::batch::{build_batch, Position};
use mantis_player::cache::CACHE_ENTRIES;
use mantis_player::forward::{RustForward, TensorRow};
use mantis_player::player::{
    Budget, Costs, DeployBlock, GameClock, Player, PlayerError, Pull, StoneResult, EARLY_STOP,
    TREE_REUSE,
};

use mantis_search::VIRTUAL_LOSS_PENALTY;

use crate::mcts::tactics_config_of;

/// Contract #13's batch of `positions` under `encoding`: `x` f32 flat, every other array int32, and `n_graphs`; `ValueError` on an unknown or refused row, a seam guard or a pack refusal.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn web_batch<'py>(
    py: Python<'py>,
    positions: Vec<Position>,
    encoding: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let spec = mantis_encoding::lookup(encoding).ok_or_else(|| {
        PyValueError::new_err(format!("web_batch: unknown encoding {encoding:?}"))
    })?;
    let batch = py
        .detach(|| build_batch(&positions, spec))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    batch_dict(py, &batch)
}

fn batch_dict<'py>(
    py: Python<'py>,
    batch: &mantis_player::batch::WebBatch,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("n_graphs", batch.n_graphs)?;
    out.set_item("x", batch.x.clone().into_pyarray(py))?;
    for (name, array) in [
        ("src", &batch.src),
        ("dst", &batch.dst),
        ("code", &batch.code),
        ("legal", &batch.legal),
        ("node_graph", &batch.node_graph),
        ("legal_graph", &batch.legal_graph),
        ("divisor", &batch.divisor),
        ("real", &batch.real),
        ("mean_mask", &batch.mean_mask),
        ("mean_den", &batch.mean_den),
    ] {
        out.set_item(name, array.clone().into_pyarray(py))?;
    }
    Ok(out)
}

fn refused(e: &PlayerError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn leaf<'py>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    d.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("deploy block: no `{key}`")))
}

fn stone_dict<'py>(py: Python<'py>, stone: &StoneResult) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("cell", stone.cell)?;
    d.set_item("sims", stone.sims)?;
    d.set_item("stopped", stone.stopped)?;
    d.set_item("decided", stone.decided)?;
    d.set_item("inherited", stone.inherited)?;
    d.set_item("root_visits", stone.root_visits)?;
    d.set_item("root_value", stone.root_value)?;
    let top: Vec<_> = stone
        .top
        .iter()
        .map(|c| (c.cell, c.visits, c.prior, c.q))
        .collect();
    d.set_item("top", top)?;
    match stone.tactics {
        Some(t) => {
            let rows = PyDict::new(py);
            for (name, value) in t.rows() {
                rows.set_item(name, value)?;
            }
            d.set_item("tactics", rows)?;
        }
        None => d.set_item("tactics", py.None())?,
    }
    Ok(d)
}

/// The Rust deploy driver: one board at a time, its turns pulled as batches and pushed back as outputs.
#[pyclass(name = "Player", module = "mantis._engine", unsendable)]
pub struct PyPlayer {
    inner: Player,
}

#[pymethods]
impl PyPlayer {
    /// A player for `encoding` under the manifest's `deploy.resolved` block. Raises: ValueError (a block that is not PUCT, a missing or bad key, a refused row).
    #[new]
    #[pyo3(signature = (deploy, encoding, *, cpu_path))]
    fn new(deploy: &Bound<'_, PyDict>, encoding: &str, cpu_path: bool) -> PyResult<Self> {
        let kind: String = leaf(deploy, "search_kind")?.extract()?;
        if kind != "puct" {
            return Err(PyValueError::new_err(format!(
                "deploy block: the player carries PUCT only, not {kind:?}"
            )));
        }
        let tactics = match deploy.get_item("tactics")? {
            Some(t) if !t.is_none() => Some(tactics_config_of(&t.cast_into::<PyDict>()?)?),
            _ => None,
        };
        let block = DeployBlock {
            c_puct: leaf(deploy, "c_puct")?.extract()?,
            fpu_reduction: leaf(deploy, "fpu_reduction")?.extract()?,
            quiescence_enabled: leaf(deploy, "quiescence_enabled")?.extract()?,
            quiescence_blend_2: leaf(deploy, "quiescence_blend_2")?.extract()?,
            leaf_batch_size: leaf(deploy, "leaf_batch_size")?.extract()?,
            tactics,
        };
        Ok(PyPlayer {
            inner: Player::new(block, encoding, cpu_path).map_err(|e| refused(&e))?,
        })
    }

    /// Set the device's measured costs.
    fn set_costs(&mut self, leaf_ms: f64, root_ms: f64) {
        self.inner.set_costs(Costs { leaf_ms, root_ms });
    }

    /// The shortest clock at which the floor fits two stones, once costs are set.
    #[getter]
    fn min_turn_ms(&self) -> Option<f64> {
        self.inner.min_turn_ms()
    }

    /// A new game at the position `moves` reach, its pool sized for `sims`, `turn_ms` or neither (unlimited). Raises: ValueError.
    #[pyo3(signature = (moves, *, sims=None, turn_ms=None))]
    #[allow(clippy::needless_pass_by_value)]
    fn new_game(
        &mut self,
        moves: Vec<(i32, i32)>,
        sims: Option<u32>,
        turn_ms: Option<f64>,
    ) -> PyResult<()> {
        let clock = match (sims, turn_ms) {
            (Some(n), None) => GameClock::Sims(n),
            (None, Some(t)) => GameClock::TurnMs(t),
            (None, None) => GameClock::Unlimited,
            _ => return Err(PyValueError::new_err("new_game: sims or turn_ms, not both")),
        };
        self.inner.new_game(&moves, clock).map_err(|e| refused(&e))
    }

    /// The other side's stones. Raises: ValueError.
    #[allow(clippy::needless_pass_by_value)]
    fn play(&mut self, stones: Vec<(i32, i32)>) -> PyResult<()> {
        self.inner.play(&stones).map_err(|e| refused(&e))
    }

    /// Begin the turn under `sims`, `turn_ms` or neither (unlimited), at `now_ms`. Raises: ValueError.
    #[pyo3(signature = (now_ms, *, sims=None, turn_ms=None))]
    fn begin_turn(&mut self, now_ms: f64, sims: Option<u32>, turn_ms: Option<f64>) -> PyResult<()> {
        let budget = match (sims, turn_ms) {
            (Some(n), None) => Budget::Sims(n),
            (None, Some(t)) => Budget::Clock { turn_ms: t },
            (None, None) => Budget::Unlimited,
            _ => {
                return Err(PyValueError::new_err(
                    "begin_turn: sims or turn_ms, not both",
                ))
            }
        };
        self.inner
            .begin_turn(budget, now_ms)
            .map_err(|e| refused(&e))
    }

    /// The misses as one batch (contract #13's arrays and `n_graphs`), or `{"turn": [stones], ...}` once the turn ends. Raises: ValueError.
    fn next_batch<'py>(&mut self, py: Python<'py>, now_ms: f64) -> PyResult<Bound<'py, PyDict>> {
        match self.inner.next_batch(now_ms).map_err(|e| refused(&e))? {
            Pull::Batch(batch) => batch_dict(py, batch),
            Pull::Turn(result) => {
                let d = PyDict::new(py);
                let stones = result
                    .stones
                    .iter()
                    .map(|s| stone_dict(py, s))
                    .collect::<PyResult<Vec<_>>>()?;
                d.set_item("turn", stones)?;
                d.set_item("sims_set", result.sims_set)?;
                d.set_item("floor_over_clock", result.floor_over_clock)?;
                d.set_item("budget_pool_capped", result.budget_pool_capped)?;
                d.set_item("wall_ms", result.wall_ms)?;
                Ok(d)
            }
        }
    }

    /// The outstanding batch's `probs [L]` and `value [G]`. Raises: ValueError (a wrong length, a refused expand).
    fn submit(
        &mut self,
        probs: PyReadonlyArray1<'_, f32>,
        values: PyReadonlyArray1<'_, f32>,
        now_ms: f64,
    ) -> PyResult<()> {
        self.inner
            .submit(probs.as_slice()?, values.as_slice()?, now_ms)
            .map_err(|e| refused(&e))
    }

    /// Drop the turn in flight; the board is as before it.
    fn abort(&mut self) {
        self.inner.abort();
    }

    /// The search's and the cache's rows, cumulative, named as the Python deploy head's `search_rows`.
    fn rows<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let (r, c) = self.inner.rows();
        let d = PyDict::new(py);
        for (name, value) in [
            ("stop_fired", r.stop_fired),
            ("stop_saved", r.stop_saved),
            ("select_calls", r.select_calls),
            ("select_overlaps", r.select_overlaps),
            ("select_network_leaves", r.select_network_leaves),
            ("reuse_reroots", r.reuse_reroots),
            ("reuse_inherited", r.reuse_inherited),
            ("cache_calls", c.calls),
            ("cache_positions", c.positions),
            ("cache_hits", c.hits),
            ("cache_served", c.served),
            ("cache_in_batch_repeats", c.in_batch_repeats),
            ("cache_all_hit_calls", c.all_hit_calls),
            ("cache_evicted", c.evicted),
        ] {
            d.set_item(name, value)?;
        }
        Ok(d)
    }
}

fn int32s(batch: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<i32>> {
    Ok(leaf(batch, key)?
        .extract::<PyReadonlyArray1<'_, i32>>()?
        .as_slice()?
        .to_vec())
}

/// The export's served forward in Rust over its weights file: W2's native row.
#[pyclass(name = "RustForward", module = "mantis._engine", frozen)]
pub struct PyRustForward {
    inner: RustForward,
}

#[pymethods]
impl PyRustForward {
    /// The manifest's tensor table over the weights file's floats. Raises: ValueError (a table the file does not hold).
    #[new]
    #[allow(clippy::needless_pass_by_value)]
    fn new(
        table: Vec<Bound<'_, PyDict>>,
        weights: PyReadonlyArray1<'_, f32>,
        node_feat_dim: usize,
    ) -> PyResult<Self> {
        let rows = table
            .iter()
            .map(|r| {
                Ok(TensorRow {
                    name: leaf(r, "name")?.extract()?,
                    rows: leaf(r, "rows")?.extract()?,
                    cols: leaf(r, "cols")?.extract()?,
                    offset: leaf(r, "offset")?.extract()?,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let inner = RustForward::load(&rows, weights.as_slice()?, node_feat_dim)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyRustForward { inner })
    }

    /// `logits`, `probs` and `value` for contract #13's arrays (`x` flat f32, the rest int32). Raises: ValueError, TypeError.
    fn forward<'py>(
        &self,
        py: Python<'py>,
        batch: &Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let mean_den = int32s(batch, "mean_den")?;
        let web = WebBatch {
            n_graphs: mean_den.len(),
            x: leaf(batch, "x")?
                .extract::<PyReadonlyArray1<'_, f32>>()?
                .as_slice()?
                .to_vec(),
            src: int32s(batch, "src")?,
            dst: int32s(batch, "dst")?,
            code: int32s(batch, "code")?,
            legal: int32s(batch, "legal")?,
            node_graph: int32s(batch, "node_graph")?,
            legal_graph: int32s(batch, "legal_graph")?,
            divisor: int32s(batch, "divisor")?,
            real: int32s(batch, "real")?,
            mean_mask: int32s(batch, "mean_mask")?,
            mean_den,
            leaves: Vec::new(),
        };
        let out = py
            .detach(|| self.inner.forward(&web))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let d = PyDict::new(py);
        d.set_item("logits", out.logits.into_pyarray(py))?;
        d.set_item("probs", out.probs.into_pyarray(py))?;
        d.set_item("value", out.value.into_pyarray(py))?;
        Ok(d)
    }
}

/// The player's own constants, each pinned equal to the source it mirrors.
#[pyfunction]
fn player_constants(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("early_stop", EARLY_STOP)?;
    d.set_item("tree_reuse", TREE_REUSE)?;
    d.set_item("cache_entries", CACHE_ENTRIES)?;
    d.set_item("virtual_loss", VIRTUAL_LOSS_PENALTY)?;
    Ok(d)
}

/// Register `web_batch`, `Player`, `RustForward` and `player_constants` into `_engine`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(web_batch, m)?)?;
    m.add_function(wrap_pyfunction!(player_constants, m)?)?;
    m.add_class::<PyPlayer>()?;
    m.add_class::<PyRustForward>()?;
    Ok(())
}
