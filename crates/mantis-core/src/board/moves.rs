// Exceeds the 300-line soft cap: legal-move/win/threat rules port as one
// line-auditable unit together with their in-file oracle test suite.
use std::cell::RefCell;

use fxhash::FxHashSet;
use super::state::{Board, Cell, Player, HEX_AXES, hex_distance};

/// Per-thread reusable scratch buffers for the `get_clusters()` BFS partition, replacing three
/// per-call Vec allocations. Thread-local because `Board::clone` is itself on the search hot
/// path, so a per-Board scratch would need skip-on-clone treatment and survive nothing.
struct ClusterScratch {
    stones: Vec<(i32, i32)>,
    visited: Vec<bool>,
    queue: Vec<usize>,
}

impl ClusterScratch {
    fn new() -> Self {
        Self {
            stones: Vec::new(),
            visited: Vec::new(),
            queue: Vec::new(),
        }
    }
}

thread_local! {
    static CLUSTER_SCRATCH_TLS: RefCell<ClusterScratch> = RefCell::new(ClusterScratch::new());
}

/// Stones in a row required to win; re-exported so search says `WIN_LENGTH - 1`, not a bare 5.
pub const WIN_LENGTH: usize = 6;

/// Default maximum hex distance at which a new stone may be placed. The official rule is 8;
/// the cap is 5 because self-play with early bootstrap nets fragments the board past the
/// 19×19 view window, while real games never exceed radius 5 between consecutive plies.
pub const DEFAULT_LEGAL_MOVE_RADIUS: i32 = 5;

/// Cells in a closed hex ball of the given radius, centre included: `3r² + 3r + 1`. ONE home
/// for the arithmetic, so tests say `hex_ball_cells(R) - 1` rather than a transcribed `90`.
#[must_use]
pub const fn hex_ball_cells(radius: i32) -> usize {
    (3 * radius * radius + 3 * radius + 1) as usize
}

/// Default maximum hex distance between stones sharing a cluster, held equal to the
/// legal-move radius so the cluster and move windows cannot mismatch.
pub const DEFAULT_CLUSTER_THRESHOLD: i32 = 5;

impl Board {
    /// Shared reference to the lazily-maintained legal move set, rebuilt on demand from a hex
    /// ball of radius `legal_move_radius` around every stone. Prefer it over `legal_moves()`
    /// in search expansion. The returned reference borrows `*self`, so invalidating the cache
    /// while it is live is statically rejected (INV-3, `legal_cache` in `state::core`):
    ///
    /// ```compile_fail,E0502
    /// use mantis_core::board::Board;
    ///
    /// let mut b = Board::new();
    /// let s = b.legal_moves_set(); // shared borrow of `b`, held live below
    /// b.mark_cache_dirty();        // &mut b while `s` is live — E0502
    /// s.len();                     // held reference used AFTER the &mut call
    /// ```
    pub fn legal_moves_set(&self) -> &FxHashSet<(i32, i32)> {
        if self.cache_is_dirty() {
            // SAFETY: INV-1..INV-4 (see the `legal_cache` field doc in
            // state::core) — exclusive access: dirty==true means a `&mut self`
            // event happened after the last `legal_moves_set` return (INV-2),
            // so no previously returned shared reference is still live
            // (INV-3); this `&mut` is dropped before the shared return borrow
            // below is created (INV-1); and the rebuild block calls NO Board
            // method and reads nothing through the cache (INV-4 — adding any
            // call here is a review failure), so no aliasing borrow can be
            // created while it lives.
            let cache = unsafe { &mut *self.legal_cache.get() };
            cache.clear();
            if self.cells.is_empty() {
                // Empty board: 5×5 region, same as Board::new() init.
                for dq in -2i32..=2 {
                    for dr in -2i32..=2 {
                        cache.insert((dq, dr));
                    }
                }
            } else {
                // For every placed stone, emit all empty cells within `legal_move_radius`.
                // The hex ball in axial coords: |dq| ≤ R, |dr| ≤ R, |dq + dr| ≤ R.
                let r = self.legal_move_radius;
                // Bbox-based upper bound on the legal-move set size: the min of the axial
                // bbox area (tight late-game) and cells.len() × hex-ball area (tight early).
                // A proven bound lets the insert loop grow the table in ONE allocation.
                let ru = r.max(0) as usize;
                let w_q = (self.max_q.saturating_sub(self.min_q) as usize)
                    .saturating_add(1)
                    .saturating_add(2 * ru);
                let w_r = (self.max_r.saturating_sub(self.min_r) as usize)
                    .saturating_add(1)
                    .saturating_add(2 * ru);
                let bbox_area = w_q.saturating_mul(w_r);
                // Hex ball area in axial coords = 3r² + 3r + 1.
                let ball_area = 3 * ru * ru + 3 * ru + 1;
                let combo_bound = self.cells.len().saturating_mul(ball_area);
                let upper_bound = bbox_area.min(combo_bound);
                // reserve relative to current capacity — no-op if already sized.
                cache.reserve(upper_bound.saturating_sub(cache.len()));
                for &(sq, sr) in self.cells.keys() {
                    for dq in -r..=r {
                        let dr_min = (-r).max(-r - dq);
                        let dr_max = r.min(r - dq);
                        for dr in dr_min..=dr_max {
                            let pos = (sq + dq, sr + dr);
                            if !self.cells.contains_key(&pos) {
                                cache.insert(pos);
                            }
                        }
                    }
                }
            }
            self.clear_cache_dirty();
        }
        // SAFETY: shared read; INV-2/INV-3 — no exclusive borrow can be
        // created while this reference is live (every dirty-true transition,
        // hence every rebuild, requires `&mut self`).
        unsafe { &*self.legal_cache.get() }
    }

    /// All legal moves as a sorted Vec; use `legal_moves_set()` on performance-critical paths.
    pub fn legal_moves(&self) -> Vec<(i32, i32)> {
        let mut moves_vec: Vec<(i32, i32)> = self.legal_moves_set().iter().copied().collect();
        moves_vec.sort_unstable();
        moves_vec
    }

    /// Number of legal moves — O(1) when the cache is clean, O(n²+bbox) after a mutation.
    pub fn legal_move_count(&self) -> usize {
        self.legal_moves_set().len()
    }

    /// Returns true if either player has 6 in a row (checks last move only).
    pub fn check_win(&self) -> bool {
        match self.last_move {
            None => false,
            Some((q, r)) => {
                // `apply_move` atomically inserts the cell and sets `last_move`, so when
                // `last_move == Some((q, r))` the cell is present and `.unwrap()` is sound.
                let cell = *self.cells.get(&(q, r)).unwrap();
                self.count_in_line(q, r, cell) >= WIN_LENGTH
            }
        }
    }

    /// Returns the winning player, if any.
    pub fn winner(&self) -> Option<Player> {
        if self.player_wins(Player::One) {
            Some(Player::One)
        } else if self.player_wins(Player::Two) {
            Some(Player::Two)
        } else {
            None
        }
    }

    /// Returns true if `player` has 6 stones in a row along any hex axis.
    pub fn player_wins(&self, player: Player) -> bool {
        let cell = match player {
            Player::One => Cell::P1,
            Player::Two => Cell::P2,
        };
        // Fast path: only the player who just moved can have just won.
        if let Some((lq, lr)) = self.last_move {
            if self.cells.get(&(lq, lr)).copied() == Some(cell) {
                return self.count_in_line(lq, lr, cell) >= WIN_LENGTH;
            }
        }
        // Fallback: scan all stones of this player (reached when player != last mover).
        for (&(q, r), &c) in &self.cells {
            if c == cell && self.count_in_line(q, r, cell) >= WIN_LENGTH {
                return true;
            }
        }
        false
    }

    /// Maximum consecutive run through (q, r) for stones of type `cell`, over all three axes.
    fn count_in_line(&self, q: i32, r: i32, cell: Cell) -> usize {
        let mut best = 0;
        for &(dq, dr) in &HEX_AXES {
            let count = 1
                + self.count_direction(q, r, dq, dr, cell)
                + self.count_direction(q, r, -dq, -dr, cell);
            if count > best {
                best = count;
            }
        }
        best
    }

    /// Count consecutive stones of `cell` starting from (q, r) in direction
    /// (dq, dr), not counting (q, r) itself.
    pub(crate) fn count_direction(&self, mut q: i32, mut r: i32, dq: i32, dr: i32, cell: Cell) -> usize {
        let mut count = 0;
        loop {
            q += dq;
            r += dr;
            if self.cells.get(&(q, r)).copied() != Some(cell) {
                break;
            }
            count += 1;
        }
        count
    }

    /// CF-1 terminal value from the side-to-move's perspective at a `check_win` leaf, and the
    /// single engine-owned surface for that sign. `apply_move` flips the player ONLY on a
    /// turn-final stone, so `moves_remaining == 1` means the winner is still to move (**+1.0**)
    /// and `moves_remaining == 2` means it flipped to the loser (**-1.0**).
    #[inline]
    pub fn terminal_value_to_move(&self) -> f32 {
        if self.moves_remaining == 1 { 1.0 } else { -1.0 }
    }

    /// True if `player` has at least `min_len` consecutive stones along any hex axis — a cheap
    /// O(stones × 3 × avg_run) pre-check before the O(legal_moves) `count_winning_moves`.
    pub fn has_player_long_run(&self, player: Player, min_len: usize) -> bool {
        let cell = match player {
            Player::One => Cell::P1,
            Player::Two => Cell::P2,
        };
        for (&(q, r), &c) in &self.cells {
            if c != cell {
                continue;
            }
            for &(dq, dr) in &HEX_AXES {
                let run = 1
                    + self.count_direction(q, r, dq, dr, cell)
                    + self.count_direction(q, r, -dq, -dr, cell);
                if run >= min_len {
                    return true;
                }
            }
        }
        false
    }

    /// Count how many empty cells, if occupied by `player`, would complete a 6-in-a-row, by
    /// testing each legal cell's run length along all three axes without placing a stone. At
    /// `count >= 3` the side to move has a forced win: the opponent blocks at most 2 per turn.
    pub fn count_winning_moves(&self, player: Player) -> u32 {
        let cell = match player {
            Player::One => Cell::P1,
            Player::Two => Cell::P2,
        };

        let legal = self.legal_moves_set();
        let mut count = 0u32;

        for &(q, r) in legal.iter() {
            for &(dq, dr) in &HEX_AXES {
                let run = 1
                    + self.count_direction(q, r, dq, dr, cell)
                    + self.count_direction(q, r, -dq, -dr, cell);
                if run >= WIN_LENGTH {
                    count += 1;
                    break; // count each cell at most once
                }
            }
        }

        count
    }

    /// All empty legal cells that complete a 6-in-a-row for `player`, sorted — the cells a
    /// player can play to win NOW, and equivalently the cells the opponent must deny.
    pub fn winning_moves(&self, player: Player) -> Vec<(i32, i32)> {
        let cell = match player {
            Player::One => Cell::P1,
            Player::Two => Cell::P2,
        };
        let legal = self.legal_moves_set();
        let mut wins: Vec<(i32, i32)> = Vec::new();
        for &(q, r) in legal.iter() {
            for &(dq, dr) in &HEX_AXES {
                let run = 1
                    + self.count_direction(q, r, dq, dr, cell)
                    + self.count_direction(q, r, -dq, -dr, cell);
                if run >= WIN_LENGTH {
                    wins.push((q, r));
                    break; // each cell at most once
                }
            }
        }
        wins.sort_unstable();
        wins
    }

    /// Cells that, if `player` plays them, give `player` ≥1 immediate winning move afterward —
    /// the threat-CREATING move set behind a threat-space search.
    ///
    /// Candidates come FROM player-stone neighborhoods rather than a scan of all legal cells:
    /// per stone, the 6 length-6 windows per axis, a window with exactly 4 player stones + 2
    /// empties and no opponent yielding 2 candidates. Output is IDENTICAL to the linear scan
    /// (asserted by `threat_moves_equivalence_fuzz`). Sorted and deterministic.
    pub fn threat_moves(&self, player: Player) -> Vec<(i32, i32)> {
        let pcell = match player {
            Player::One => Cell::P1,
            Player::Two => Cell::P2,
        };
        let legal = self.legal_moves_set();

        let mut candidates: FxHashSet<(i32, i32)> = FxHashSet::default();

        for (&(sq, sr), &sc) in &self.cells {
            if sc != pcell {
                continue;
            }
            // Per axis, the 6 windows containing (sq, sr): offset s puts the stone at (-s).
            for &(dq, dr) in &HEX_AXES {
                for s in -5..=0i32 {
                    let mut pcount = 0usize; // existing player stones in window
                    let mut ecount = 0usize; // empty cells in window
                    let mut dead = false;
                    let mut empties = [(0i32, 0i32); 2];
                    let wq = sq + s * dq;
                    let wr = sr + s * dr;
                    for i in 0..6i32 {
                        let (q, r) = (wq + i * dq, wr + i * dr);
                        match self.cells.get(&(q, r)).copied() {
                            Some(c) if c == pcell => pcount += 1,
                            None => {
                                if ecount < 2 {
                                    empties[ecount] = (q, r);
                                }
                                ecount += 1;
                            }
                            _ => { dead = true; break; } // opponent or over-2-empties kill
                        }
                    }
                    // Exactly 4 existing player stones + 2 empties: playing either empty
                    // gives 5 + 1 empty = a win-in-1. `pcount` counts existing stones only.
                    if !dead && pcount == 4 && ecount == 2 {
                        for &e in &empties {
                            if legal.contains(&e) {
                                candidates.insert(e);
                            }
                        }
                    }
                }
            }
        }

        let mut out: Vec<(i32, i32)> = candidates.into_iter().collect();
        out.sort_unstable();
        out
    }

    /// Oracle implementation of `threat_moves` — a linear scan, used ONLY in tests.
    #[cfg(test)]
    pub fn threat_moves_ref(&self, player: Player) -> Vec<(i32, i32)> {
        let pcell = match player {
            Player::One => Cell::P1,
            Player::Two => Cell::P2,
        };
        let mut legal: Vec<(i32, i32)> = self.legal_moves_set().iter().copied().collect();
        legal.sort_unstable();
        let mut out = Vec::new();
        for &(cq, cr) in &legal {
            let mut is_threat = false;
            'axis: for &(dq, dr) in &HEX_AXES {
                for s in -5..=0i32 {
                    let mut pcount = 0;
                    let mut empties = 0;
                    let mut dead = false;
                    for i in 0..6i32 {
                        let (q, r) = (cq + (s + i) * dq, cr + (s + i) * dr);
                        let occ = if (q, r) == (cq, cr) {
                            Some(pcell)
                        } else {
                            self.cells.get(&(q, r)).copied()
                        };
                        match occ {
                            Some(c) if c == pcell => pcount += 1,
                            None => empties += 1,
                            _ => { dead = true; break; }
                        }
                    }
                    if !dead && pcount == 5 && empties == 1 {
                        is_threat = true;
                        break 'axis;
                    }
                }
            }
            if is_threat {
                out.push((cq, cr));
            }
        }
        out
    }

    /// The lexicographically-first empty legal cell completing a 6-in-a-row for `player`, or
    /// `None`. Deterministic; the primitive behind the forced-win one-hot POLICY target.
    pub fn first_winning_move(&self, player: Player) -> Option<(i32, i32)> {
        let cell = match player {
            Player::One => Cell::P1,
            Player::Two => Cell::P2,
        };
        let legal = self.legal_moves_set();
        let mut cells: Vec<(i32, i32)> = legal.iter().copied().collect();
        cells.sort_unstable();
        for (q, r) in cells {
            for &(dq, dr) in &HEX_AXES {
                let run = 1
                    + self.count_direction(q, r, dq, dr, cell)
                    + self.count_direction(q, r, -dq, -dr, cell);
                if run >= WIN_LENGTH {
                    return Some((q, r));
                }
            }
        }
        None
    }

    /// The immediate move (for the SIDE TO MOVE) that proves a within-turn forced win, or
    /// `None` — the forced-win one-hot POLICY target detector.
    ///
    /// * `depth >= 1`: a move completing 6-in-a-row now, at any turn-phase.
    /// * `depth >= 2` AND `moves_remaining == 2`: a first placement leaving the SAME player an
    ///   immediate win on the turn's second stone. Turn-phase comes from `moves_remaining`,
    ///   never ply parity; at `mr == 1` the opponent replies first, so depth-2 is suppressed.
    ///
    /// Pre-gated by `has_player_long_run` (≥5 for a one-move win, ≥4 for two), which skips
    /// rare two-gap setups such as `XX__XX`: precision over recall.
    pub fn forced_win_move(&self, depth: u8) -> Option<(i32, i32)> {
        if depth == 0 {
            return None;
        }
        let player = self.current_player;

        // depth-1: an immediate 6-completing move (valid at any moves_remaining).
        if self.has_player_long_run(player, WIN_LENGTH - 1) {
            if let Some(mv) = self.first_winning_move(player) {
                return Some(mv);
            }
        }

        // depth-2 only at mr == 2 — otherwise the opponent replies before the second stone.
        if depth >= 2
            && self.moves_remaining == 2
            && self.has_player_long_run(player, WIN_LENGTH - 2)
        {
            let legal = self.legal_moves_set();
            let mut cells: Vec<(i32, i32)> = legal.iter().copied().collect();
            cells.sort_unstable();
            for (q, r) in cells {
                let mut probe = self.clone();
                if probe.apply_move(q, r).is_err() {
                    continue;
                }
                // mr 2→1, no flip: a win available now means the same player completes 6.
                if probe.has_player_long_run(player, WIN_LENGTH - 1)
                    && probe.first_winning_move(player).is_some()
                {
                    return Some((q, r));
                }
            }
        }

        None
    }

    /// The cells forming the winning 6-in-a-row, or an empty Vec if no win. Fast path from the
    /// last placed stone, fallback scan over all stones — the fallback must agree with
    /// `player_wins`'s, since the first stone of a turn can complete a line the second misses.
    pub fn find_winning_line(&self) -> Vec<(i32, i32)> {
        if let Some((lq, lr)) = self.last_move {
            if let Some(&cell) = self.cells.get(&(lq, lr)) {
                for &(dq, dr) in &HEX_AXES {
                    let pos_count = self.count_direction(lq, lr, dq, dr, cell) as i32;
                    let neg_count = self.count_direction(lq, lr, -dq, -dr, cell) as i32;
                    let total = (1 + pos_count + neg_count) as usize;
                    if total >= WIN_LENGTH {
                        let mut line = Vec::with_capacity(total);
                        for i in -neg_count..=pos_count {
                            line.push((lq + dq * i, lr + dr * i));
                        }
                        return line;
                    }
                }
            }
        }
        // Stones sorted by (q, r) so the returned line is deterministic across map orders.
        let mut stones: Vec<((i32, i32), Cell)> =
            self.cells.iter().map(|(&k, &v)| (k, v)).collect();
        stones.sort_unstable_by_key(|&((q, r), _)| (q, r));
        for &((q, r), cell) in &stones {
            for &(dq, dr) in &HEX_AXES {
                // Only count from the start of a run (no predecessor of same colour).
                if self.cells.get(&(q - dq, r - dr)).copied() == Some(cell) {
                    continue;
                }
                let pos_count = self.count_direction(q, r, dq, dr, cell) as i32;
                let total = (1 + pos_count) as usize;
                if total >= WIN_LENGTH {
                    let mut line = Vec::with_capacity(total);
                    for i in 0..=pos_count {
                        line.push((q + dq * i, r + dr * i));
                    }
                    return line;
                }
            }
        }
        vec![]
    }

    /// Partition all placed stones into clusters where two stones share one iff their
    /// `hex_distance` is at most `self.cluster_threshold`. Consumed by `get_cluster_views()`.
    pub fn get_clusters(&self) -> Vec<Vec<(i32, i32)>> {
        let mut clusters: Vec<Vec<(i32, i32)>> = Vec::new();
        if self.cells.is_empty() {
            return clusters;
        }

        let threshold = self.cluster_threshold;

        CLUSTER_SCRATCH_TLS.with(|scratch| {
            let mut s = scratch.borrow_mut();
            let ClusterScratch { stones, visited, queue } = &mut *s;

            // Refill the thread-local scratch from this Board's stones.
            // `clear` + `extend` reuses the existing allocation when capacity
            // suffices; `visited.resize(.., false)` zeroes only the in-use
            // prefix (still O(n) but no allocation when n ≤ capacity).
            stones.clear();
            stones.extend(self.cells.keys().copied());
            visited.clear();
            visited.resize(stones.len(), false);
            queue.clear();

            for i in 0..stones.len() {
                if visited[i] { continue; }
                let mut cluster = Vec::new();
                queue.push(i);
                visited[i] = true;

                while let Some(curr) = queue.pop() {
                    cluster.push(stones[curr]);
                    for j in 0..stones.len() {
                        if !visited[j] && hex_distance(stones[curr].0, stones[curr].1, stones[j].0, stones[j].1) <= threshold {
                            visited[j] = true;
                            queue.push(j);
                        }
                    }
                }
                clusters.push(cluster);
            }
        });

        clusters
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ply::Ply;

    #[test]
    fn test_count_winning_moves_empty_board() {
        let board = Board::new();
        assert_eq!(board.count_winning_moves(Player::One), 0);
        assert_eq!(board.count_winning_moves(Player::Two), 0);
    }

    #[test]
    fn test_count_winning_moves_five_in_row() {
        // P1 has 5 in a row on the E axis (q=0..4, r=0), so q=-1 and q=5 both complete 6.
        let mut board = Board::new();
        board.apply_move(0, 0).unwrap(); // P1 ply0 (single)
        // Place 4 more P1 stones (need P2 filler moves between each pair)
        // P2 fillers go far away; we just need P1 to have 5 in a row.
        board.apply_move(0, 9).unwrap(); board.apply_move(0, 8).unwrap(); // P2 turn
        board.apply_move(1, 0).unwrap(); board.apply_move(2, 0).unwrap(); // P1 turn
        board.apply_move(0, 7).unwrap(); board.apply_move(0, 6).unwrap(); // P2 turn
        board.apply_move(3, 0).unwrap(); board.apply_move(4, 0).unwrap(); // P1 turn

        let p1_wins = board.count_winning_moves(Player::One);
        assert_eq!(p1_wins, 2, "5-in-a-row should have exactly 2 winning moves");
    }

    #[test]
    fn test_count_winning_moves_five_blocked_one_end() {
        // P1 5 in a row at r=0 with a P2 blocker at q=-1, so only q=5 wins.
        let mut board = Board::new();
        board.apply_move(0, 0).unwrap(); // P1
        board.apply_move(-1, 0).unwrap(); board.apply_move(0, 9).unwrap(); // P2 (blocker + filler)
        board.apply_move(1, 0).unwrap(); board.apply_move(2, 0).unwrap(); // P1
        board.apply_move(0, 8).unwrap(); board.apply_move(0, 7).unwrap(); // P2
        board.apply_move(3, 0).unwrap(); board.apply_move(4, 0).unwrap(); // P1

        let p1_wins = board.count_winning_moves(Player::One);
        assert_eq!(p1_wins, 1, "one end blocked → 1 winning move");
    }

    #[test]
    fn test_count_winning_moves_zero_when_early_game() {
        let mut board = Board::new();
        board.apply_move(0, 0).unwrap(); // P1
        board.apply_move(3, 3).unwrap(); board.apply_move(4, 4).unwrap(); // P2
        board.apply_move(0, 5).unwrap(); board.apply_move(5, 0).unwrap(); // P1

        assert_eq!(board.count_winning_moves(Player::One), 0);
        assert_eq!(board.count_winning_moves(Player::Two), 0);
    }

    #[test]
    fn test_count_winning_moves_three_independent_winning_cells() {
        // P1 has three separate 5-in-a-row threats, each with one open end: E axis r=0 (win
        // at q=5), NE axis q=0 (win at r=5), NW axis (win at (-5,5)); other ends P2-blocked.

        let mut board = Board::new();

        for q in 0..5i32 {
            board.cells.insert((q, 0), Cell::P1);
        }
        // Blocker for E-axis west end
        board.cells.insert((-1, 0), Cell::P2);

        for r in 1..5i32 {  // r=0 already placed above
            board.cells.insert((0, r), Cell::P1);
        }
        // Blocker for NE-axis south end
        board.cells.insert((0, -1), Cell::P2);

        for i in 1..5i32 {  // (0,0) already placed
            board.cells.insert((-i, i), Cell::P1);
        }
        // Blocker for NW-axis south end
        board.cells.insert((1, -1), Cell::P2);

        board.has_stones = true;
        board.mark_cache_dirty();

        let p1_wins = board.count_winning_moves(Player::One);
        // E-axis: q=5 (1 cell); NE-axis: (0,5) (1 cell); NW-axis: (-5,5) (1 cell)
        assert!(p1_wins >= 3,
            "expected ≥3 winning moves for three blocked 5-in-a-row threats, got {p1_wins}");
    }

    #[test]
    fn test_has_player_long_run_empty_board() {
        let board = Board::new();
        assert!(!board.has_player_long_run(Player::One, 5));
        assert!(!board.has_player_long_run(Player::Two, 5));
    }

    #[test]
    fn test_has_player_long_run_detects_five_in_row() {
        let mut board = Board::new();
        for q in 0..5i32 {
            board.cells.insert((q, 0), Cell::P1);
        }
        board.has_stones = true;
        board.mark_cache_dirty();
        assert!(board.has_player_long_run(Player::One, 5),
            "5 consecutive P1 stones should be detected");
        assert!(!board.has_player_long_run(Player::Two, 5),
            "P2 has no long run");
    }

    #[test]
    fn test_has_player_long_run_returns_false_for_scattered() {
        let mut board = Board::new();
        board.cells.insert((0, 0), Cell::P1);
        board.cells.insert((5, 0), Cell::P1);
        board.cells.insert((0, 5), Cell::P1);
        board.has_stones = true;
        board.mark_cache_dirty();
        assert!(!board.has_player_long_run(Player::One, 3));
    }

    // `forced_win_move(depth)`: depth-1 completes 6 now; depth-2 (only at mr==2) is a first
    // placement leaving the SAME player an immediate win on the second stone.

    /// Build a static position with explicit side-to-move + turn-phase, bbox included.
    fn fwm_board(stones: &[((i32, i32), Cell)], player: Player, mr: u8) -> Board {
        let mut b = Board::new();
        let (mut lq, mut hq, mut lr, mut hr) = (i32::MAX, i32::MIN, i32::MAX, i32::MIN);
        for &((q, r), c) in stones {
            b.cells.insert((q, r), c);
            lq = lq.min(q); hq = hq.max(q); lr = lr.min(r); hr = hr.max(r);
        }
        b.has_stones = true;
        b.min_q = lq; b.max_q = hq; b.min_r = lr; b.max_r = hr;
        b.mark_cache_dirty();
        b.current_player = player;
        b.moves_remaining = mr;
        b.ply = Ply::new(stones.len() as u32);
        b
    }

    #[test]
    fn test_first_winning_move_returns_completing_cell() {
        // P1 5-in-a-row (q=0..4, r=0): (-1,0) and (5,0) each complete 6.
        let stones: Vec<_> = (0..5).map(|q| ((q, 0), Cell::P1)).collect();
        let b = fwm_board(&stones, Player::One, 2);
        let mv = b.first_winning_move(Player::One).expect("a winning move exists");
        let mut b2 = b.clone();
        b2.apply_move(mv.0, mv.1).unwrap();
        assert!(b2.check_win(), "first_winning_move must complete 6, got {mv:?}");
        assert_eq!(b.first_winning_move(Player::Two), None, "P2 has no winning move");
    }

    #[test]
    fn test_forced_win_move_depth1_completes_six() {
        let stones: Vec<_> = (0..5).map(|q| ((q, 0), Cell::P1)).collect();
        let b = fwm_board(&stones, Player::One, 2);
        let mv = b.forced_win_move(1).expect("depth-1 forced win exists");
        let mut b2 = b.clone();
        b2.apply_move(mv.0, mv.1).unwrap();
        assert!(b2.check_win(), "depth-1 move must complete 6, got {mv:?}");
    }

    #[test]
    fn test_forced_win_move_depth1_fires_at_mr1() {
        // Depth-1 (immediate win) is valid regardless of turn-phase (mr==1 too).
        let stones: Vec<_> = (0..5).map(|q| ((q, 0), Cell::P1)).collect();
        let b = fwm_board(&stones, Player::One, 1);
        let mv = b.forced_win_move(2).expect("depth-1 must fire at mr==1");
        let mut b2 = b.clone();
        b2.apply_move(mv.0, mv.1).unwrap();
        assert!(b2.check_win(), "got {mv:?}");
    }

    #[test]
    fn test_forced_win_move_depth2_sets_up_within_turn_win() {
        // P1 4-in-a-row: no single move wins, but at mr==2 a first placement leaves P1 a win.
        let stones: Vec<_> = (0..4).map(|q| ((q, 0), Cell::P1)).collect();
        let b = fwm_board(&stones, Player::One, 2);
        assert_eq!(b.forced_win_move(1), None, "4-in-a-row has no immediate win");

        let mv = b.forced_win_move(2).expect("depth-2 forced win exists");
        let mut b2 = b.clone();
        b2.apply_move(mv.0, mv.1).unwrap();
        // After the first placement the SAME player is still to move (mr 2→1, no
        // flip) and now has an immediate win — that is the within-turn forced win.
        assert_eq!(b2.current_player, Player::One,
            "first placement of a 2-move turn keeps the same player");
        assert!(b2.first_winning_move(Player::One).is_some(),
            "after the depth-2 setup P1 must have an immediate win, setup={mv:?}");
    }

    #[test]
    fn test_forced_win_move_depth2_guarded_at_mr1() {
        // Same 4-in-a-row at mr==1: the opponent moves next, so depth-2 must be suppressed.
        let stones: Vec<_> = (0..4).map(|q| ((q, 0), Cell::P1)).collect();
        let b = fwm_board(&stones, Player::One, 1);
        assert_eq!(b.forced_win_move(2), None,
            "depth-2 must be guarded off at mr==1 (opponent blocks before 2nd stone)");
    }

    #[test]
    fn test_forced_win_move_targets_side_to_move_only() {
        let stones: Vec<_> = (0..5).map(|q| ((q, 0), Cell::P2)).collect();
        let b = fwm_board(&stones, Player::One, 2);
        assert_eq!(b.forced_win_move(2), None, "only the side-to-move's wins count");
    }

    #[test]
    fn test_forced_win_move_none_without_threat() {
        let stones = [((0, 0), Cell::P1), ((5, 0), Cell::P1), ((0, 5), Cell::P1)];
        let b = fwm_board(&stones, Player::One, 2);
        assert_eq!(b.forced_win_move(2), None);
    }

    #[test]
    fn test_terminal_value_to_move_cf1_sign() {
        // Engine-owned CF-1 sign: mr==1 (winner still to move) ⇒ +1.0, mr==2 ⇒ -1.0.
        let stones: Vec<_> = (0..5).map(|q| ((q, 0), Cell::P1)).collect();
        let b1 = fwm_board(&stones, Player::One, 1);
        assert_eq!(b1.terminal_value_to_move(), 1.0, "mr==1 ⇒ +1.0");
        let b2 = fwm_board(&stones, Player::One, 2);
        assert_eq!(b2.terminal_value_to_move(), -1.0, "mr==2 ⇒ -1.0");
    }

    // Pins fast `threat_moves` == oracle `threat_moves_ref` over a broad corpus. That
    // equivalence IS the soundness guarantee behind the loss guard and candidate completeness.

    /// Build a random board by playing `n` legal moves chosen by a splitmix64 PRNG.
    fn random_board(seed: u64, n: usize) -> Board {
        let mut s = seed;
        let mut board = Board::new();
        for _ in 0..n {
            let legal = board.legal_moves();
            if legal.is_empty() { break; }
            // splitmix64
            s = s.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^= z >> 31;
            let idx = (z as usize) % legal.len();
            let (q, r) = legal[idx];
            let _ = board.apply_move(q, r);
        }
        board
    }

    #[test]
    #[cfg_attr(miri, ignore)] // 200-board runtime — excluded set per prereg
    fn threat_moves_equivalence_fuzz() {
        // 200 random boards × 2 players = 400 checks, over stone counts 5..=50.
        for seed in 0u64..200 {
            let n = 5 + ((seed * 7 + 13) % 46) as usize; // 5..50 stones
            let board = random_board(seed.wrapping_mul(0xdeadbeef), n);

            for player in [Player::One, Player::Two] {
                let fast = board.threat_moves(player);
                let reference = board.threat_moves_ref(player);
                assert_eq!(
                    fast, reference,
                    "threat_moves diverged from threat_moves_ref on seed={seed} n={n} player={player:?}\n  fast={fast:?}\n  ref={reference:?}"
                );
            }
        }
    }

    #[test]
    fn threat_moves_five_in_row_open_at_both_ends() {
        // P1 4-in-a-row: q=4 and q=-1 each make a 5-stone line with one empty end.
        let mut board = Board::new();
        for q in 0..4i32 {
            board.cells.insert((q, 0), Cell::P1);
        }
        board.has_stones = true;
        board.mark_cache_dirty();
        // Rebuild min/max so legal_moves_set works correctly.
        board.min_q = 0; board.max_q = 3; board.min_r = 0; board.max_r = 0;

        let fast = board.threat_moves(Player::One);
        let reference = board.threat_moves_ref(Player::One);
        assert_eq!(fast, reference, "fast and ref must agree on 4-in-a-row position");
        // The threat cells are those that extend to 5 stones with 1 open end:
        // placing at q=-1 → 4 player +1 more = window (-1,0..4,0) if open, etc.
        // Just verify equivalence here; exact cells depend on legal set.
        assert!(!fast.is_empty(), "4-in-a-row should produce threat moves");
    }

    #[test]
    fn threat_moves_empty_board_is_empty() {
        let board = Board::new();
        assert_eq!(board.threat_moves(Player::One), board.threat_moves_ref(Player::One));
        assert_eq!(board.threat_moves(Player::Two), board.threat_moves_ref(Player::Two));
        assert!(board.threat_moves(Player::One).is_empty());
        assert!(board.threat_moves(Player::Two).is_empty());
    }

    // The dirty-flag-while-borrowed trigger is statically unwritable: the flag is private to
    // `state::core` and its sole crate-visible true-setter takes `&mut self`.

    #[test]
    fn threat_moves_opponent_blocked_window_excluded() {
        // A P2 stone at q=4 blocks the east end; the reference oracle is ground truth.
        let mut board = Board::new();
        for q in 0..4i32 {
            board.cells.insert((q, 0), Cell::P1);
        }
        board.cells.insert((4, 0), Cell::P2);
        board.has_stones = true;
        board.min_q = 0; board.max_q = 4; board.min_r = 0; board.max_r = 0;
        board.mark_cache_dirty();

        let fast = board.threat_moves(Player::One);
        let reference = board.threat_moves_ref(Player::One);
        assert_eq!(fast, reference, "blocked end must be excluded consistently");
    }
}
