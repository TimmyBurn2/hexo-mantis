//! `TurnSolver`: Six's turn-level strictly forcing search, by iterative deepening in attacking turns.
//!
//! Every attacking turn needs both defender stones to block, so the defender never gets a free stone and every win
//! found is sound. Deterministic: same board, budgets and table state give the same `Solved`, nodes included.

use mantis_core::board::Board;

use super::analyze::{analyze, Terminal};
use super::gen::{covering_pairs, double_threats, GenScratch, ThreatTurn};
use super::grid::{Grid, Kind, Side};
use super::table::{Stored, Table};

/// Mixed into the table key by the attacker's side, so two stages can never share an entry.
const STAGE: [u128; 2] = [
    0x6a09_e667_f3bc_c908_b2fb_1366_ea95_7d3e,
    0xbb67_ae85_84ca_a73b_3c6e_f372_fe94_f82b,
];

/// One or two stones of a turn, in play order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Turn {
    stones: [(i32, i32); 2],
    len: u8,
}

impl Turn {
    pub(crate) fn of(stones: &[(i32, i32)]) -> Self {
        let mut t = Turn {
            stones: [(0, 0); 2],
            len: stones.len().min(2) as u8,
        };
        t.stones[..t.len as usize].copy_from_slice(&stones[..t.len as usize]);
        t
    }

    #[must_use]
    pub fn stones(&self) -> &[(i32, i32)] {
        &self.stones[..self.len as usize]
    }
}

/// What a solve proved for the side to move.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// A strictly forcing win within `turns` attacking turns; the six lands on the turn after the last.
    /// `turns` 0 is a finish this turn.
    Win { first: Turn, turns: u8 },
    /// The side to move cannot stop a six this turn: lost on cover.
    Loss,
    /// Neither proven; `exhausted` when the node budget ran out.
    Unknown { exhausted: bool },
}

/// A solve's verdict and the attacker nodes it spent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Solved {
    pub verdict: Verdict,
    pub nodes: u64,
}

fn pack(cell: (i32, i32)) -> u32 {
    (((cell.0 + 32768) as u32) << 16) | ((cell.1 + 32768) as u32 & 0xFFFF)
}

fn unpack(packed: u32) -> (i32, i32) {
    (
        (packed >> 16) as i32 - 32768,
        (packed & 0xFFFF) as i32 - 32768,
    )
}

/// The turn-level strictly forcing solver over its own grid and table, both allocated once.
pub struct TurnSolver {
    grid: Grid,
    table: Table,
    gen: GenScratch,
    attacks: Vec<Vec<ThreatTurn>>,
    defences: Vec<Vec<(u32, u32)>>,
    attacker: Side,
    nodes: u64,
    budget: u64,
    aborted: bool,
    overflow: bool,
    root: Option<(u32, u32)>,
    grid_overflows: u64,
}

impl TurnSolver {
    /// A solver whose table holds `table_entries` slots, rounded down to a power of two.
    #[must_use]
    pub fn new(table_entries: usize) -> Self {
        TurnSolver {
            grid: Grid::new(),
            table: Table::new(table_entries),
            gen: GenScratch::default(),
            attacks: Vec::new(),
            defences: Vec::new(),
            attacker: Side::P1,
            nodes: 0,
            budget: 0,
            aborted: false,
            overflow: false,
            root: None,
            grid_overflows: 0,
        }
    }

    /// Forget the table in O(1).
    pub fn clear(&mut self) {
        self.table.clear();
    }

    /// Solves that stopped at the grid's edge, cumulative.
    #[must_use]
    pub fn grid_overflows(&self) -> u64 {
        self.grid_overflows
    }

    /// Generations that paired only the first 48 three-cells, cumulative (a named incompleteness).
    #[must_use]
    pub fn three_cells_capped(&self) -> u64 {
        self.gen.three_cells_capped
    }

    /// Prove a strictly forcing win for the side to move within `turns` attacking turns and `nodes` attacker nodes.
    /// A finish this turn is `Win` with `turns` 0 and a lost cover `Loss`, both from `analyze`; only a quiet
    /// position with two stones to place is searched.
    pub fn solve(&mut self, board: &Board, turns: u8, nodes: u64) -> Solved {
        let unknown = |exhausted| Solved {
            verdict: Verdict::Unknown { exhausted },
            nodes: 0,
        };
        if board.check_win() {
            return unknown(false);
        }
        let facts = analyze(board);
        match facts.terminal {
            Some(Terminal::Win) => {
                return Solved {
                    verdict: Verdict::Win {
                        first: Turn::of(&facts.finish),
                        turns: 0,
                    },
                    nodes: 0,
                };
            }
            Some(Terminal::Loss) => {
                return Solved {
                    verdict: Verdict::Loss,
                    nodes: 0,
                };
            }
            None => {}
        }
        if !facts.forced.is_empty() {
            return unknown(false);
        }
        self.search_quiet(board, turns, nodes)
    }

    /// `solve` less its `analyze`, for a caller that has read `board` quiet already (the leaf wiring).
    pub fn search_quiet(&mut self, board: &Board, turns: u8, nodes: u64) -> Solved {
        let unknown = |exhausted| Solved {
            verdict: Verdict::Unknown { exhausted },
            nodes: 0,
        };
        if board.moves_remaining != 2 || turns == 0 || nodes == 0 {
            return unknown(false);
        }
        if self.grid.load(board).is_err() {
            self.grid_overflows += 1;
            return unknown(false);
        }
        self.attacker = Side::of(board.current_player);
        self.nodes = 0;
        self.budget = nodes;
        self.aborted = false;
        self.overflow = false;
        self.root = None;
        let depth = usize::from(turns) + 1;
        if self.attacks.len() < depth {
            self.attacks.resize_with(depth, Vec::new);
            self.defences.resize_with(depth, Vec::new);
        }
        let mut proven = 0;
        for limit in 1..=turns {
            proven = self.attack(limit, 0);
            if proven > 0 || self.aborted {
                break;
            }
        }
        self.grid.unload();
        let verdict = match (proven, self.root) {
            (p, Some((a, b))) if p > 0 => Verdict::Win {
                first: Turn::of(&[unpack(a), unpack(b)]),
                turns: p,
            },
            _ if self.overflow => {
                self.grid_overflows += 1;
                Verdict::Unknown { exhausted: false }
            }
            _ => Verdict::Unknown {
                exhausted: self.aborted,
            },
        };
        Solved {
            verdict,
            nodes: self.nodes,
        }
    }

    /// The proof's main line after a `Win`: each stored attacker turn and the first covering reply, from the table.
    /// A display artefact, never a verdict.
    pub fn line(&mut self, board: &Board) -> Vec<(i32, i32)> {
        let facts = analyze(board);
        if facts.terminal == Some(Terminal::Win) {
            return facts.finish;
        }
        let mut out = Vec::new();
        if board.moves_remaining != 2 || self.grid.load(board).is_err() {
            return out;
        }
        let me = Side::of(board.current_player);
        let mut replies = Vec::new();
        while let Some(stored) = self.table.probe(self.grid.key() ^ STAGE[me as usize]) {
            if stored.proven == 0 {
                break;
            }
            let (a, b) = (unpack(stored.a), unpack(stored.b));
            out.extend([a, b]);
            let (Some(ia), Some(ib)) = (self.grid.index_of(a.0, a.1), self.grid.index_of(b.0, b.1))
            else {
                break;
            };
            if stored.proven == 1 || self.grid.at(ia) != 0 || self.grid.at(ib) != 0 {
                break;
            }
            self.grid.place(ia, me);
            self.grid.place(ib, me);
            covering_pairs(&self.grid, me, &mut self.gen, &mut replies);
            let Some(&(x, y)) = replies.first() else {
                break;
            };
            out.extend([self.grid.cell_at(x), self.grid.cell_at(y)]);
            self.grid.place(x, me.other());
            self.grid.place(y, me.other());
        }
        self.grid.unload();
        out
    }

    /// Attacking turns the side to move needs within `turns_left`, 0 when none was found (Six's `attack`).
    fn attack(&mut self, turns_left: u8, ply: usize) -> u8 {
        self.nodes += 1;
        if self.nodes > self.budget {
            self.aborted = true;
            return 0;
        }
        let me = self.attacker;
        let key = self.grid.key() ^ STAGE[me as usize];
        if let Some(e) = self.table.probe(key) {
            if e.proven > 0 && e.proven <= turns_left {
                if ply == 0 {
                    self.root = Some((e.a, e.b));
                }
                return e.proven;
            }
            if e.proven == 0 && e.searched >= turns_left {
                return 0;
            }
        }
        let mut turns = std::mem::take(&mut self.attacks[ply]);
        double_threats(&self.grid, me, &mut self.gen, &mut turns);
        let mut proven = 0u8;
        let mut best = None;
        if let Some(t) = turns.iter().find(|t| t.cover >= 3) {
            if Grid::placeable(t.a) && Grid::placeable(t.b) {
                proven = 1;
                best = Some(*t);
            } else {
                self.overflow = true;
                self.aborted = true;
            }
        }
        if !self.aborted && proven == 0 && turns_left > 1 {
            for t in &turns {
                if !(Grid::placeable(t.a) && Grid::placeable(t.b)) {
                    self.overflow = true;
                    self.aborted = true;
                    break;
                }
                self.grid.place(t.a, me);
                self.grid.place(t.b, me);
                let slowest = self.defend(turns_left, ply);
                self.grid.undo();
                self.grid.undo();
                if self.aborted {
                    break;
                }
                if let Some(slowest) = slowest {
                    proven = 1 + slowest;
                    best = Some(*t);
                    break;
                }
            }
        }
        self.attacks[ply] = turns;
        if self.aborted {
            return 0;
        }
        let cells = best.map_or((0, 0), |t| {
            (pack(self.grid.cell_at(t.a)), pack(self.grid.cell_at(t.b)))
        });
        self.table.store(
            key,
            Stored {
                a: cells.0,
                b: cells.1,
                proven,
                searched: turns_left,
            },
        );
        if ply == 0 && proven > 0 {
            self.root = Some(cells);
        }
        proven
    }

    /// The slowest proven win over every covering reply to the attacker's turn on the grid, or `None` when a reply
    /// refutes it: no reply exists, a reply gives the defender a threat window, or the attacker finds no win.
    fn defend(&mut self, turns_left: u8, ply: usize) -> Option<u8> {
        let me = self.attacker;
        let opp = me.other();
        let mut replies = std::mem::take(&mut self.defences[ply]);
        covering_pairs(&self.grid, me, &mut self.gen, &mut replies);
        let mut slowest = if replies.is_empty() { None } else { Some(0) };
        for &(x, y) in &replies {
            if !(Grid::placeable(x) && Grid::placeable(y)) {
                self.overflow = true;
                self.aborted = true;
                slowest = None;
                break;
            }
            self.grid.place(x, opp);
            self.grid.place(y, opp);
            let needed = if self.grid.list(opp, Kind::Threat).is_empty() {
                self.attack(turns_left - 1, ply + 1)
            } else {
                0
            };
            self.grid.undo();
            self.grid.undo();
            slowest = match (slowest, needed) {
                (_, 0) => None,
                (Some(s), n) => Some(s.max(n)),
                (None, _) => None,
            };
            if slowest.is_none() {
                break;
            }
        }
        self.defences[ply] = replies;
        slowest
    }
}
