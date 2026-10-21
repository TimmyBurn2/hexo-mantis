// >300 justify (R8): the per-stone loop is one state machine — the stone's root phase, its drive, the cache step between
// them and the turn they add up to — so a split would pass half a stone's state across a module boundary.

//! The deploy head's per-stone loop as a pull/push step machine; the engine reads no clock and calls no net.

use std::fmt;

use mantis_core::{Board, BoardGeometry, Cell, Player as Side};
use mantis_encoding::RegistrySpec;
use mantis_search::mcts::{SelectionDesync, TacticsConfig, TacticsCounters, TacticsError};
use mantis_search::{
    armed_sims_for, LegalSetPolicy, MCTSTree, MAX_ARMED_SIMS, MAX_CHILDREN_PER_NODE,
    VIRTUAL_LOSS_PENALTY,
};
use mantis_selfplay::queues::eval_cache::LeafKey;
use mantis_selfplay::records::assemble_ls_from_gnn_probs;
use mantis_selfplay::runner::params::{resolve_geometry, WorkerGeometry};

use crate::batch::{build_batch, BatchError, Position, WebBatch};
use crate::budget::{min_turn_ms, pool_sims, quotient, sims_for_turn, CPU_FLOOR, FLOOR};
use crate::cache::{CacheRows, Entry, GameCache, Source};

/// The early stop is on, as the ladder ships it (`LADDER_EARLY_STOP`).
pub const EARLY_STOP: bool = true;
/// Within-turn tree reuse is on, as the ladder ships it (`LADDER_TREE_REUSE`).
pub const TREE_REUSE: bool = true;

/// The deploy block resolved from the stamp's config (the manifest's `deploy.resolved`); the search is PUCT.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeployBlock {
    pub c_puct: f32,
    pub fpu_reduction: f32,
    pub quiescence_enabled: bool,
    pub quiescence_blend_2: f32,
    pub leaf_batch_size: usize,
    pub tactics: Option<TacticsConfig>,
}

/// The device's measured costs: wall ms a served leaf and a turn's root phases.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Costs {
    pub leaf_ms: f64,
    pub root_ms: f64,
}

/// What a game's pool is sized for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GameClock {
    Sims(u32),
    TurnMs(f64),
    Unlimited,
}

/// What a turn spends a stone.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Budget {
    Sims(u32),
    Clock { turn_ms: f64 },
    Unlimited,
}

/// One stone as played: `decided` is the root solver's unsearched stone, `inherited` the visits a re-root kept.
#[derive(Debug, Clone, PartialEq)]
pub struct StoneResult {
    pub cell: (i32, i32),
    pub sims: u32,
    pub stopped: bool,
    pub decided: bool,
    pub inherited: u32,
    pub root_visits: u32,
    pub root_value: Option<f32>,
    pub top: Vec<Child>,
    pub tactics: Option<TacticsCounters>,
}

/// A root child as the search left it: its visits, its prior and its Q from the mover's side.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Child {
    pub cell: (i32, i32),
    pub visits: u32,
    pub prior: f32,
    pub q: f32,
}

/// A turn: its stones, the sims it set a stone, its flags, its wall time on the host's clock and its served leaves.
#[derive(Debug, Clone, PartialEq)]
pub struct TurnResult {
    pub stones: Vec<StoneResult>,
    pub sims_set: u32,
    pub floor_over_clock: bool,
    pub budget_pool_capped: bool,
    pub wall_ms: f64,
    /// The leaves the net served this turn: its cost a leaf is `wall_ms / served_leaves`, root phases folded in.
    pub served_leaves: u64,
}

/// The search's rows, cumulative over the player's life, named as the Python head's `search_rows`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Rows {
    pub stop_fired: u64,
    pub stop_saved: u64,
    pub select_calls: u64,
    pub select_overlaps: u64,
    pub select_network_leaves: u64,
    pub reuse_reroots: u64,
    pub reuse_inherited: u64,
}

/// What `next_batch` hands out: the misses to evaluate, or the finished turn.
#[derive(Debug)]
pub enum Pull<'a> {
    Batch(&'a WebBatch),
    Turn(TurnResult),
}

/// Why the player refused; a mid-turn refusal rolls the turn back and the player stays usable.
#[derive(Debug, Clone, PartialEq)]
pub enum PlayerError {
    OutOfStep(&'static str),
    OutputShape {
        what: &'static str,
        got: usize,
        want: usize,
    },
    IllegalStone {
        cell: (i32, i32),
        reason: &'static str,
    },
    NoRootChildren,
    Refused(String),
    Expand(String),
    Selection(SelectionDesync),
    Tactics(TacticsError),
    Batch(BatchError),
    Leaf(String),
    NoCosts,
    BudgetPastPool {
        sims: u32,
        cap: u32,
    },
    NonFinite(&'static str),
}

impl PlayerError {
    /// The variant's name: a host's error `name`.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::OutOfStep(_) => "OutOfStep",
            Self::OutputShape { .. } => "OutputShape",
            Self::IllegalStone { .. } => "IllegalStone",
            Self::NoRootChildren => "NoRootChildren",
            Self::Refused(_) => "Refused",
            Self::Expand(_) => "Expand",
            Self::Selection(_) => "Selection",
            Self::Tactics(_) => "Tactics",
            Self::Batch(_) => "Batch",
            Self::Leaf(_) => "Leaf",
            Self::NoCosts => "NoCosts",
            Self::BudgetPastPool { .. } => "BudgetPastPool",
            Self::NonFinite(_) => "NonFinite",
        }
    }
}

impl fmt::Display for PlayerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfStep(why) => write!(f, "out of step: {why}"),
            Self::OutputShape { what, got, want } => {
                write!(f, "{what} holds {got} values, the batch {want}")
            }
            Self::IllegalStone { cell, reason } => write!(f, "stone {cell:?} is illegal: {reason}"),
            Self::NoRootChildren => write!(
                f,
                "the search produced no root children, so there is no stone to pick"
            ),
            Self::Refused(why) | Self::Expand(why) | Self::Leaf(why) => write!(f, "{why}"),
            Self::Selection(e) => write!(f, "{e}"),
            Self::Tactics(e) => write!(f, "{e}"),
            Self::Batch(e) => write!(f, "{e}"),
            Self::NoCosts => write!(f, "a clock budget needs the device's measured costs"),
            Self::BudgetPastPool { sims, cap } => {
                write!(f, "{sims} sims a stone is past this game's pool ({cap})")
            }
            Self::NonFinite(what) => write!(f, "{what} holds a value that is not a finite number"),
        }
    }
}

impl std::error::Error for PlayerError {}

enum Phase {
    Start,
    Drive,
    Done,
}

struct Stone {
    phase: Phase,
    sims_done: u32,
    inherited: u32,
}

struct Turn {
    start_board: Board,
    sims: u32,
    stones_left: u8,
    started_ms: f64,
    served: u64,
    flags: (bool, bool),
    stones: Vec<StoneResult>,
    stone: Stone,
}

struct Pending {
    leaves: Vec<Board>,
    keys: Vec<LeafKey>,
    sources: Vec<Source>,
    fresh: Vec<LeafKey>,
    batch: WebBatch,
    unserved: u32,
}

enum Advance {
    Net,
    Done,
}

/// One engine for one board at a time: a game, its turns, and the pull/push between them.
pub struct Player {
    spec: &'static RegistrySpec,
    block: DeployBlock,
    cpu_path: bool,
    geometry: WorkerGeometry,
    board_geometry: BoardGeometry,
    tree: MCTSTree,
    cache: GameCache,
    board: Board,
    rows: Rows,
    costs: Option<Costs>,
    turn: Option<Turn>,
    pending: Option<Pending>,
}

fn side(board: &Board) -> i64 {
    match board.current_player {
        Side::One => 1,
        Side::Two => -1,
    }
}

/// A board as the served path requests it, its stones as the bridge's `get_stones` lists them.
fn position_of(board: &Board) -> Position {
    let stones = board
        .cells_iter()
        .map(|(&(q, r), &cell)| {
            let p = match cell {
                Cell::Empty => 0,
                Cell::P1 => 1,
                Cell::P2 => -1,
            };
            (i64::from(q), i64::from(r), p)
        })
        .collect();
    (stones, side(board), i64::from(board.moves_remaining))
}

fn count(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

/// The bridge's pre-expand checks (arity, stride, centre, trunk), the player's interim copy held to them by error parity.
pub fn expand_refusal(
    leaves: &[Board],
    outputs: &[Entry],
    stride: usize,
    trunk: i32,
) -> Result<(), PlayerError> {
    if outputs.len() != leaves.len() {
        return Err(PlayerError::Expand(format!(
            "{} outputs for {} pending leaves",
            outputs.len(),
            leaves.len()
        )));
    }
    for (i, (leaf, out)) in leaves.iter().zip(outputs).enumerate() {
        if out.policy.dense.len() != stride {
            return Err(PlayerError::Expand(format!(
                "leaf {i}: dense half has {} slots != policy stride {stride}",
                out.policy.dense.len()
            )));
        }
        if leaf.window_center() != out.center {
            return Err(PlayerError::Expand(format!(
                "leaf {i}: builder window centre {:?} != the board's {:?}",
                out.center,
                leaf.window_center()
            )));
        }
        if i32::try_from(leaf.cluster_window_size()).ok() != Some(trunk) {
            return Err(PlayerError::Expand(format!(
                "leaf {i}: trunk {trunk} != the board's window {}",
                leaf.cluster_window_size()
            )));
        }
    }
    Ok(())
}

impl Player {
    /// A player for `encoding`'s graph row; `cpu_path` arms the CPU floor. Errors: `Refused` (a row, a block).
    pub fn new(block: DeployBlock, encoding: &str, cpu_path: bool) -> Result<Player, PlayerError> {
        if block.leaf_batch_size < 1 {
            return Err(PlayerError::Refused(
                "leaf_batch_size must be at least 1".to_owned(),
            ));
        }
        if let Some(t) = block.tactics {
            t.check()
                .map_err(|why| PlayerError::Refused(format!("tactics: {why}")))?;
        }
        let spec = mantis_encoding::lookup(encoding)
            .ok_or_else(|| PlayerError::Refused(format!("unknown encoding {encoding:?}")))?;
        let geometry = resolve_geometry(spec).map_err(|e| PlayerError::Refused(e.to_string()))?;
        let board_geometry = BoardGeometry {
            legal_move_radius: i32::try_from(spec.legal_move_radius).unwrap_or(i32::MAX),
            cluster_window_size: spec.cluster_window_size.unwrap_or(spec.board_size),
        };
        let mut player = Player {
            spec,
            block,
            cpu_path,
            geometry,
            board_geometry,
            tree: MCTSTree::with_pool(block.c_puct, VIRTUAL_LOSS_PENALTY, block.fpu_reduction, 1),
            cache: GameCache::default(),
            board: Board::with_geometry(board_geometry),
            rows: Rows::default(),
            costs: None,
            turn: None,
            pending: None,
        };
        player.tree = player.fresh_tree(4 * MAX_CHILDREN_PER_NODE * FLOOR as usize);
        Ok(player)
    }

    fn fresh_tree(&self, max_nodes: usize) -> MCTSTree {
        let b = self.block;
        let mut tree =
            MCTSTree::with_pool(b.c_puct, VIRTUAL_LOSS_PENALTY, b.fpu_reduction, max_nodes);
        tree.configure_quiescence(b.quiescence_enabled, b.quiescence_blend_2);
        tree.configure_tactics(b.tactics);
        tree
    }

    /// Set the device's measured costs. Errors: `Refused` unless `leaf_ms` is finite above 0 and `root_ms` finite, not below 0.
    pub fn set_costs(&mut self, costs: Costs) -> Result<(), PlayerError> {
        if !(costs.leaf_ms.is_finite()
            && costs.leaf_ms > 0.0
            && costs.root_ms.is_finite()
            && costs.root_ms >= 0.0)
        {
            return Err(PlayerError::Refused(format!(
                "costs {costs:?} are not a device's"
            )));
        }
        self.costs = Some(costs);
        Ok(())
    }

    #[must_use]
    pub fn costs(&self) -> Option<Costs> {
        self.costs
    }

    /// The shortest clock at which the floor fits two stones on this device, once its costs are known.
    #[must_use]
    pub fn min_turn_ms(&self) -> Option<f64> {
        self.costs.map(|c| min_turn_ms(c.root_ms, c.leaf_ms))
    }

    #[must_use]
    pub fn rows(&self) -> (Rows, CacheRows) {
        (self.rows, self.cache.rows())
    }

    /// The batch a `next_batch` handed out and no `submit` has taken yet.
    #[must_use]
    pub fn pending(&self) -> Option<&WebBatch> {
        self.pending.as_ref().map(|p| &p.batch)
    }

    #[must_use]
    pub fn board(&self) -> &Board {
        &self.board
    }

    fn costs_or_refuse(&self) -> Result<Costs, PlayerError> {
        self.costs.ok_or(PlayerError::NoCosts)
    }

    /// A fresh tree and cache at `moves`, pooled for `clock`. Errors: `OutOfStep`, `IllegalStone`, `NoCosts`.
    pub fn new_game(&mut self, moves: &[(i32, i32)], clock: GameClock) -> Result<(), PlayerError> {
        if self.turn.is_some() {
            return Err(PlayerError::OutOfStep("a turn is in play"));
        }
        let mut board = Board::with_geometry(self.board_geometry);
        for &cell in moves {
            board
                .apply_move(cell.0, cell.1)
                .map_err(|reason| PlayerError::IllegalStone { cell, reason })?;
        }
        let ceiling = count(MAX_ARMED_SIMS);
        let sims = match clock {
            GameClock::Sims(n) => n.clamp(FLOOR, ceiling),
            GameClock::TurnMs(t) => {
                let c = self.costs_or_refuse()?;
                pool_sims(t, c.root_ms, c.leaf_ms)
            }
            GameClock::Unlimited => ceiling,
        };
        let b = self.block; // the old pool and solver go before the new ones are allocated
        self.tree = MCTSTree::with_pool(b.c_puct, VIRTUAL_LOSS_PENALTY, b.fpu_reduction, 1);
        self.tree = self.fresh_tree(4 * MAX_CHILDREN_PER_NODE * sims as usize);
        self.cache.new_game();
        self.board = board;
        Ok(())
    }

    /// The other side's stones, all or none. Errors: `OutOfStep` mid-turn, `IllegalStone`.
    pub fn play(&mut self, stones: &[(i32, i32)]) -> Result<(), PlayerError> {
        if self.turn.is_some() {
            return Err(PlayerError::OutOfStep("a turn is in play"));
        }
        let mut board = self.board.clone();
        for &cell in stones {
            board
                .apply_move(cell.0, cell.1)
                .map_err(|reason| PlayerError::IllegalStone { cell, reason })?;
        }
        self.board = board;
        Ok(())
    }

    /// Begin the side to move's turn. Errors: `OutOfStep` mid-turn, `NoCosts`, `BudgetPastPool`.
    pub fn begin_turn(&mut self, budget: Budget, now_ms: f64) -> Result<(), PlayerError> {
        if self.turn.is_some() {
            return Err(PlayerError::OutOfStep("a turn is in play"));
        }
        let cap = count(armed_sims_for(self.tree.pool.len()).min(MAX_ARMED_SIMS));
        let stones = u32::from(self.board.moves_remaining.max(1));
        let (sims, flags) = match budget {
            Budget::Sims(n) if n > cap => return Err(PlayerError::BudgetPastPool { sims: n, cap }),
            Budget::Sims(n) => (n, (false, false)),
            Budget::Unlimited => (cap, (false, false)),
            Budget::Clock { turn_ms } => {
                let c = self.costs_or_refuse()?;
                let short = turn_ms < min_turn_ms(c.root_ms, c.leaf_ms);
                let floor = if short && self.cpu_path {
                    CPU_FLOOR
                } else {
                    FLOOR
                };
                let capped = quotient(turn_ms, c.root_ms, c.leaf_ms, stones) > f64::from(cap);
                (
                    sims_for_turn(turn_ms, c.root_ms, c.leaf_ms, stones, floor, cap),
                    (short, capped),
                )
            }
        };
        self.turn = Some(Turn {
            start_board: self.board.clone(),
            sims,
            stones_left: self.board.moves_remaining.max(1),
            started_ms: now_ms,
            served: 0,
            flags,
            stones: Vec::new(),
            stone: Stone {
                phase: Phase::Start,
                sims_done: 0,
                inherited: 0,
            },
        });
        Ok(())
    }

    /// Run the turn to its next miss batch or its end. Errors: `OutOfStep`, or a search refusal (the turn rolls back).
    pub fn next_batch(&mut self, now_ms: f64) -> Result<Pull<'_>, PlayerError> {
        if self.pending.is_some() {
            return Err(PlayerError::OutOfStep("a batch is outstanding"));
        }
        let mut turn = self
            .turn
            .take()
            .ok_or(PlayerError::OutOfStep("no turn begun"))?;
        match self.advance(&mut turn) {
            Ok(Advance::Net) => {
                self.turn = Some(turn);
                self.pending
                    .as_ref()
                    .map(|p| Pull::Batch(&p.batch))
                    .ok_or(PlayerError::OutOfStep("no batch to hand out"))
            }
            Ok(Advance::Done) => Ok(Pull::Turn(self.close(turn, now_ms))),
            Err(e) => {
                self.roll_back(turn.start_board);
                Err(e)
            }
        }
    }

    /// The batch's `probs [L]` and `value [G]`. Errors: `OutOfStep`, `OutputShape`, `Expand` (the turn rolls back).
    pub fn submit(
        &mut self,
        probs: &[f32],
        values: &[f32],
        _now_ms: f64,
    ) -> Result<(), PlayerError> {
        let pending = self
            .pending
            .take()
            .ok_or(PlayerError::OutOfStep("no batch is outstanding"))?;
        let mut turn = self
            .turn
            .take()
            .ok_or(PlayerError::OutOfStep("no turn begun"))?;
        match self.apply(&mut turn, &pending, probs, values) {
            Ok(()) => {
                self.turn = Some(turn);
                Ok(())
            }
            Err(e) => {
                self.roll_back(turn.start_board);
                Err(e)
            }
        }
    }

    /// Drop the turn in flight; the board is as it was before the turn.
    pub fn abort(&mut self) {
        self.pending = None;
        if let Some(turn) = self.turn.take() {
            self.roll_back(turn.start_board);
        }
    }

    /// The board as before the turn, under a tree reset to it: no stale virtual loss survives into a later re-root.
    fn roll_back(&mut self, start_board: Board) {
        self.tree.new_game(start_board.clone());
        self.board = start_board;
    }

    fn close(&mut self, turn: Turn, now_ms: f64) -> TurnResult {
        let wall_ms = now_ms - turn.started_ms;
        if turn.served > 0 && wall_ms.is_finite() && wall_ms > 0.0 {
            // The engine sees no time inside a call, so a turn's root phases fold into its cost a leaf: the safe side.
            #[allow(clippy::cast_precision_loss)] // a turn serves far fewer than 2^52 leaves
            let leaf_ms = wall_ms / turn.served as f64;
            self.costs = Some(Costs {
                leaf_ms,
                root_ms: 0.0,
            });
        }
        TurnResult {
            stones: turn.stones,
            sims_set: turn.sims,
            floor_over_clock: turn.flags.0,
            budget_pool_capped: turn.flags.1,
            wall_ms,
            served_leaves: turn.served,
        }
    }

    fn advance(&mut self, turn: &mut Turn) -> Result<Advance, PlayerError> {
        loop {
            match turn.stone.phase {
                Phase::Done => return Ok(Advance::Done),
                Phase::Start => {
                    if self.start_stone(turn)? {
                        return Ok(Advance::Net);
                    }
                }
                Phase::Drive => {
                    if self.drive(turn)? {
                        return Ok(Advance::Net);
                    }
                }
            }
        }
    }

    /// A stone's root phase: the origin on an empty board, the re-root, the root solver, the root step.
    fn start_stone(&mut self, turn: &mut Turn) -> Result<bool, PlayerError> {
        if self.board.cells_iter().next().is_none() {
            let result = StoneResult {
                cell: (0, 0),
                sims: 0,
                stopped: false,
                decided: false,
                inherited: 0,
                root_visits: 0,
                root_value: None,
                top: Vec::new(),
                tactics: None,
            };
            self.place(turn, result)?;
            return Ok(false);
        }
        self.tree
            .check_tactics_board(&self.board)
            .map_err(PlayerError::Tactics)?;
        let inherited = if TREE_REUSE {
            self.tree.reroot_or_new_game(self.board.clone())
        } else {
            self.tree.new_game(self.board.clone());
            0
        };
        if self.block.tactics.is_some() {
            match self.tree.root_offence() {
                Ok(Some(cell)) => {
                    let result = StoneResult {
                        cell,
                        sims: 0,
                        stopped: false,
                        decided: true,
                        inherited,
                        root_visits: self.tree.root_visits(),
                        root_value: None,
                        top: Vec::new(),
                        tactics: Some(self.tree.tactics_counters()),
                    };
                    self.place(turn, result)?;
                    return Ok(false);
                }
                Ok(None) | Err(TacticsError::ProofStoneIllegal { .. }) => {}
                Err(e) => return Err(PlayerError::Tactics(e)),
            }
        }
        if inherited > 0 {
            self.rows.reuse_reroots += 1;
            self.rows.reuse_inherited += u64::from(inherited);
        }
        turn.stone = Stone {
            phase: Phase::Drive,
            sims_done: 0,
            inherited,
        };
        let leaves = self.tree.select_leaves(1).map_err(PlayerError::Selection)?;
        let unserved = self.unserved();
        if leaves.is_empty() {
            turn.stone.sims_done += unserved;
            return Ok(false);
        }
        self.serve(turn, leaves, unserved)
    }

    fn unserved(&self) -> u32 {
        count(self.tree.last_inline_descents() + self.tree.last_tt_hits())
    }

    /// One drive step: the budget's end, the early stop, or a batch of leaves through the cache.
    fn drive(&mut self, turn: &mut Turn) -> Result<bool, PlayerError> {
        let done = turn.stone.sims_done;
        if done >= turn.sims {
            self.finish_searched(turn, false)?;
            return Ok(false);
        }
        let left = turn.sims - done;
        if EARLY_STOP && self.leader_fixed(left) {
            self.rows.stop_fired += 1;
            self.rows.stop_saved += u64::from(left);
            self.finish_searched(turn, true)?;
            return Ok(false);
        }
        let leaves = self
            .tree
            .select_leaves_filled(self.block.leaf_batch_size, left as usize)
            .map_err(PlayerError::Selection)?;
        let unserved = self.unserved();
        if leaves.is_empty() && unserved == 0 {
            self.finish_searched(turn, false)?;
            return Ok(false);
        }
        if leaves.is_empty() {
            turn.stone.sims_done += unserved;
            return Ok(false);
        }
        self.serve(turn, leaves, unserved)
    }

    /// No runner-up can reach the visit leader with `left` descents.
    fn leader_fixed(&self, left: u32) -> bool {
        match self.tree.get_top_visits(2).as_slice() {
            [_] => true,
            [first, second] => first.1.saturating_sub(second.1) > left,
            _ => false,
        }
    }

    /// The leaves through the cache: `true` when the misses wait for the net as the pending batch.
    fn serve(
        &mut self,
        turn: &mut Turn,
        leaves: Vec<Board>,
        unserved: u32,
    ) -> Result<bool, PlayerError> {
        let g = self.geometry;
        let positions: Vec<Position> = leaves.iter().map(position_of).collect();
        let keys = positions
            .iter()
            .map(|(stones, player, left)| {
                LeafKey::of(
                    stones,
                    *player,
                    *left,
                    g.win_length,
                    g.graph_radius,
                    g.agg_trunk_sz,
                    g.empty_edges,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(PlayerError::Leaf)?;
        let (sources, fresh) = self.cache.plan(&keys);
        if fresh.is_empty() {
            let outputs = self.cache.finish(&keys, &sources, &fresh, Vec::new());
            self.expand(turn, &leaves, &outputs, unserved)?;
            return Ok(false);
        }
        let first: Vec<Position> = fresh
            .iter()
            .filter_map(|key| {
                keys.iter()
                    .position(|k| k == key)
                    .map(|i| positions[i].clone())
            })
            .collect();
        let batch = build_batch(&first, self.spec).map_err(PlayerError::Batch)?;
        self.pending = Some(Pending {
            leaves,
            keys,
            sources,
            fresh,
            batch,
            unserved,
        });
        Ok(true)
    }

    fn apply(
        &mut self,
        turn: &mut Turn,
        pending: &Pending,
        probs: &[f32],
        values: &[f32],
    ) -> Result<(), PlayerError> {
        let batch = &pending.batch;
        if probs.len() != batch.legal.len() {
            return Err(PlayerError::OutputShape {
                what: "probs",
                got: probs.len(),
                want: batch.legal.len(),
            });
        }
        if values.len() != batch.n_graphs {
            return Err(PlayerError::OutputShape {
                what: "value",
                got: values.len(),
                want: batch.n_graphs,
            });
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(PlayerError::NonFinite("value"));
        }
        let mut offset = 0;
        let mut fresh = Vec::with_capacity(batch.n_graphs);
        for (meta, &value) in batch.leaves.iter().zip(values) {
            let rows = meta.policy_dst_slot.len();
            let slice = probs
                .get(offset..offset + rows)
                .ok_or(PlayerError::OutputShape {
                    what: "probs",
                    got: probs.len(),
                    want: offset + rows,
                })?;
            let policy = assemble_ls_from_gnn_probs(
                self.spec.policy_stride(),
                slice,
                &meta.policy_dst_slot,
                &meta.legal_coords,
            )
            .map_err(PlayerError::Expand)?;
            fresh.push(Entry {
                policy,
                value,
                center: meta.center,
            });
            offset += rows;
        }
        turn.served += fresh.len() as u64;
        let outputs = self
            .cache
            .finish(&pending.keys, &pending.sources, &pending.fresh, fresh);
        self.expand(turn, &pending.leaves, &outputs, pending.unserved)
    }

    /// The bridge's `expand_and_backup_ls_graph` checks, then the expand; the leaves and their descents count.
    fn expand(
        &mut self,
        turn: &mut Turn,
        leaves: &[Board],
        outputs: &[Entry],
        unserved: u32,
    ) -> Result<(), PlayerError> {
        let trunk = self.geometry.agg_trunk_sz;
        expand_refusal(leaves, outputs, self.spec.policy_stride(), trunk)?;
        let policies: Vec<LegalSetPolicy> = outputs.iter().map(|o| o.policy.clone()).collect();
        let values: Vec<f32> = outputs.iter().map(|o| o.value).collect();
        let centers: Vec<(i32, i32)> = outputs.iter().map(|o| o.center).collect();
        self.tree
            .expand_and_backup_ls_at(&policies, &values, &centers, trunk);
        turn.stone.sims_done += count(leaves.len()) + unserved;
        Ok(())
    }

    /// The searched stone: the select rows, the most visited child, the audit, the placement.
    fn finish_searched(&mut self, turn: &mut Turn, stopped: bool) -> Result<(), PlayerError> {
        let counters = self.tree.select_counters();
        self.rows.select_calls += counters.calls;
        self.rows.select_overlaps += counters.overlaps;
        self.rows.select_network_leaves += counters.network_leaves;
        let top: Vec<Child> = self
            .tree
            .get_top_visits(2)
            .iter()
            .map(|&(cell, visits, prior, q)| Child {
                cell,
                visits,
                prior,
                q,
            })
            .collect();
        let mut cell = top
            .first()
            .map(|c| c.cell)
            .ok_or(PlayerError::NoRootChildren)?;
        let (root_visits, root_value) = (self.tree.root_visits(), self.tree.root_value());
        let tactics = if self.block.tactics.is_some() {
            let order = self.tree.audit_order(None);
            cell = self.tree.root_audit(cell, order.as_deref());
            Some(self.tree.tactics_counters())
        } else {
            None
        };
        let result = StoneResult {
            cell,
            sims: turn.stone.sims_done,
            stopped,
            decided: false,
            inherited: turn.stone.inherited,
            root_visits,
            root_value: Some(root_value),
            top,
            tactics,
        };
        self.place(turn, result)
    }

    fn place(&mut self, turn: &mut Turn, result: StoneResult) -> Result<(), PlayerError> {
        let cell = result.cell;
        self.board
            .apply_move(cell.0, cell.1)
            .map_err(|reason| PlayerError::IllegalStone { cell, reason })?;
        turn.stones.push(result);
        turn.stones_left = turn.stones_left.saturating_sub(1);
        let phase = if self.board.check_win() || turn.stones_left == 0 {
            Phase::Done
        } else {
            Phase::Start
        };
        turn.stone = Stone {
            phase,
            sims_done: 0,
            inherited: 0,
        };
        Ok(())
    }
}
