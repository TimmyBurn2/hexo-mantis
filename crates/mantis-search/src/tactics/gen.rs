// >300 justify (R8): doubleThreats, coveringPairs and the scratch and tie-breaks they share port as one unit.
//! Six's strictly forcing move generator on grid indices (`(q, r)` order, so its tie-breaks carry over exactly).

use super::grid::{Grid, Kind, Side};

/// Cells that turn a three into a four are paired from at most this many (Six's `kMaxThreeCells`).
const MAX_THREE_CELLS: usize = 48;
/// A stone making a double threat alone gets this many second stones that build new threes.
const FREE_PARTNERS: usize = 4;
const NO_CELL: u32 = u32::MAX;

/// A two-stone attacking turn (`a` < `b`) and the fewest blockers of the fours it makes (3 means more than two).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ThreatTurn {
    pub(crate) a: u32,
    pub(crate) b: u32,
    pub(crate) cover: u8,
}

/// A window about to hold four or five stones, and the one or two empties it keeps.
#[derive(Clone, Copy)]
pub(crate) struct Four([u32; 2]);

impl Four {
    fn hits(&self, cell: u32) -> bool {
        self.0[0] == cell || self.0[1] == cell
    }
}

/// Fewest cells touching every four: 0 with none, 3 when two are not enough.
fn cover_of(fours: &[Four]) -> u8 {
    let Some((first, rest)) = fours.split_first() else {
        return 0;
    };
    // Every cover holds a cell of the first four.
    for &x in &first.0 {
        if x != NO_CELL && rest.iter().all(|f| f.hits(x)) {
            return 1;
        }
    }
    for &x in &first.0 {
        if x == NO_CELL {
            continue;
        }
        let Some(miss) = rest.iter().position(|f| !f.hits(x)) else {
            continue;
        };
        for &y in &rest[miss].0 {
            if y != NO_CELL && rest[miss + 1..].iter().all(|f| f.hits(x) || f.hits(y)) {
                return 2;
            }
        }
    }
    3
}

fn pair_key(x: u32, y: u32) -> u64 {
    (u64::from(x.min(y)) << 32) | u64::from(x.max(y))
}

#[derive(Clone, Copy)]
struct TwoPair {
    key: u64,
    rest: Four,
}

/// Reused buffers, so a generated turn list costs no allocation in the steady state.
#[derive(Default)]
pub(crate) struct GenScratch {
    threes: Vec<[u32; 3]>,
    cell_threes: Vec<(u32, u32)>,
    three_cells: Vec<u32>,
    two_pairs: Vec<TwoPair>,
    two_cells: Vec<u32>,
    partners: Vec<(i64, u32)>,
    candidates: Vec<u64>,
    fours: Vec<Four>,
    ranked: Vec<(u8, usize, u64)>,
    cover_fours: Vec<Four>,
    cover_cells: Vec<u32>,
    /// Generations that met more three-cells than `MAX_THREE_CELLS` (a named incompleteness).
    pub(crate) three_cells_capped: u64,
}

impl GenScratch {
    /// The fours `me`'s three windows through `cell` make with `other` also filled, skipping windows through `skip`.
    fn add_three_fours(&mut self, cell: u32, other: u32, skip: u32) {
        let from = self.cell_threes.partition_point(|&(c, _)| c < cell);
        for &(c, id) in &self.cell_threes[from..] {
            if c != cell {
                break;
            }
            let empties = self.threes[id as usize];
            if skip != NO_CELL && empties.contains(&skip) {
                continue;
            }
            let mut f = [NO_CELL; 2];
            let mut n = 0;
            for e in empties {
                if e != cell && e != other {
                    f[n] = e;
                    n += 1;
                }
            }
            self.fours.push(Four(f));
        }
    }
}

/// The empty cells of window `id`, in window order.
fn window_empties(grid: &Grid, id: u32) -> impl Iterator<Item = u32> + '_ {
    Grid::window_cells(id)
        .into_iter()
        .filter(|&c| grid.at(c) == 0)
}

/// `me`'s turns whose fours need two or more blockers, ranked by (cover desc, fours desc, pair asc).
/// Empty when `me` already holds a threat window (it can finish instead).
pub(crate) fn double_threats(grid: &Grid, me: Side, s: &mut GenScratch, out: &mut Vec<ThreatTurn>) {
    out.clear();
    if !grid.list(me, Kind::Threat).is_empty() {
        return;
    }
    s.threes.clear();
    s.cell_threes.clear();
    for &id in grid.list(me, Kind::Three) {
        let mut empties = [NO_CELL; 3];
        for (slot, cell) in empties.iter_mut().zip(window_empties(grid, id)) {
            *slot = cell;
        }
        let three = s.threes.len() as u32;
        s.threes.push(empties);
        s.cell_threes.extend(empties.iter().map(|&e| (e, three)));
    }
    s.cell_threes.sort_unstable();
    s.three_cells.clear();
    for &(cell, _) in &s.cell_threes {
        if s.three_cells.last() != Some(&cell) {
            s.three_cells.push(cell);
        }
    }

    s.two_pairs.clear();
    s.two_cells.clear();
    for &id in grid.list(me, Kind::Two) {
        let mut e = [NO_CELL; 4];
        for (slot, cell) in e.iter_mut().zip(window_empties(grid, id)) {
            *slot = cell;
        }
        for i in 0..4 {
            s.two_cells.push(e[i]);
            for j in i + 1..4 {
                let mut rest = [NO_CELL; 2];
                let mut n = 0;
                for (k, &cell) in e.iter().enumerate() {
                    if k != i && k != j {
                        rest[n] = cell;
                        n += 1;
                    }
                }
                s.two_pairs.push(TwoPair {
                    key: pair_key(e[i], e[j]),
                    rest: Four(rest),
                });
            }
        }
    }
    s.two_pairs.sort_unstable_by_key(|p| p.key);

    // Candidates: every new four had three stones and gains one, or had two and gains both.
    s.candidates.clear();
    let pairable = s.three_cells.len().min(MAX_THREE_CELLS);
    if s.three_cells.len() > MAX_THREE_CELLS {
        s.three_cells_capped += 1;
    }
    for i in 0..pairable {
        for j in i + 1..pairable {
            s.candidates
                .push(pair_key(s.three_cells[i], s.three_cells[j]));
        }
    }
    // A pair filling only two windows needs a second four: another such window, or a three through either cell.
    let mut i = 0;
    while i < s.two_pairs.len() {
        let key = s.two_pairs[i].key;
        let mut j = i;
        while j < s.two_pairs.len() && s.two_pairs[j].key == key {
            j += 1;
        }
        let (lo, hi) = ((key >> 32) as u32, key as u32);
        if j - i >= 2
            || s.three_cells.binary_search(&lo).is_ok()
            || s.three_cells.binary_search(&hi).is_ok()
        {
            s.candidates.push(key);
        }
        i = j;
    }
    // A stone whose own fours already need two blockers leaves the second stone free to start new threes.
    if !s.two_cells.is_empty() {
        s.two_cells.sort_unstable();
        s.partners.clear();
        let mut i = 0;
        while i < s.two_cells.len() {
            let cell = s.two_cells[i];
            let mut j = i;
            while j < s.two_cells.len() && s.two_cells[j] == cell {
                j += 1;
            }
            if s.three_cells.binary_search(&cell).is_err() {
                s.partners.push((-((j - i) as i64), cell));
            }
            i = j;
        }
        let best = FREE_PARTNERS.min(s.partners.len());
        s.partners.sort_unstable();
        for i in 0..pairable {
            let a = s.three_cells[i];
            s.fours.clear();
            s.add_three_fours(a, NO_CELL, NO_CELL);
            if cover_of(&s.fours) < 2 {
                continue;
            }
            for p in 0..best {
                s.candidates.push(pair_key(a, s.partners[p].1));
            }
        }
    }
    s.candidates.sort_unstable();
    s.candidates.dedup();

    s.ranked.clear();
    for c in 0..s.candidates.len() {
        let key = s.candidates[c];
        let (a, b) = ((key >> 32) as u32, key as u32);
        s.fours.clear();
        s.add_three_fours(a, b, NO_CELL);
        s.add_three_fours(b, a, a);
        let from = s.two_pairs.partition_point(|p| p.key < key);
        for p in &s.two_pairs[from..] {
            if p.key != key {
                break;
            }
            s.fours.push(p.rest);
        }
        let cover = cover_of(&s.fours);
        if cover >= 2 {
            s.ranked.push((cover, s.fours.len(), key));
        }
    }
    s.ranked
        .sort_unstable_by(|x, y| y.0.cmp(&x.0).then(y.1.cmp(&x.1)).then(x.2.cmp(&y.2)));
    out.extend(s.ranked.iter().map(|&(cover, _, key)| ThreatTurn {
        a: (key >> 32) as u32,
        b: key as u32,
        cover,
    }));
}

/// Every pair of cells (`x` < `y`) blocking all of `attacker`'s threat windows, in cell order.
pub(crate) fn covering_pairs(
    grid: &Grid,
    attacker: Side,
    s: &mut GenScratch,
    out: &mut Vec<(u32, u32)>,
) {
    out.clear();
    s.cover_fours.clear();
    s.cover_cells.clear();
    for &id in grid.list(attacker, Kind::Threat) {
        let mut f = [NO_CELL; 2];
        for (slot, cell) in f.iter_mut().zip(window_empties(grid, id)) {
            *slot = cell;
            if !s.cover_cells.contains(&cell) {
                s.cover_cells.push(cell);
            }
        }
        s.cover_fours.push(Four(f));
    }
    s.cover_cells.sort_unstable();
    for i in 0..s.cover_cells.len() {
        for j in i + 1..s.cover_cells.len() {
            let (x, y) = (s.cover_cells[i], s.cover_cells[j]);
            if s.cover_fours.iter().all(|f| f.hits(x) || f.hits(y)) {
                out.push((x, y));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn four(e: &[u32]) -> Four {
        let mut f = [NO_CELL; 2];
        f[..e.len()].copy_from_slice(e);
        Four(f)
    }

    #[test]
    fn cover_of_counts_the_fewest_blockers_up_to_three() {
        assert_eq!(cover_of(&[]), 0);
        assert_eq!(cover_of(&[four(&[1, 2])]), 1);
        assert_eq!(cover_of(&[four(&[1, 2]), four(&[2, 3])]), 1);
        assert_eq!(cover_of(&[four(&[1, 2]), four(&[3, 4])]), 2);
        assert_eq!(cover_of(&[four(&[1, 2]), four(&[3, 4]), four(&[5, 6])]), 3);
        // An open four's three windows: {-2,-1}, {-1,4}, {4,5} need two stones.
        assert_eq!(
            cover_of(&[four(&[10, 11]), four(&[11, 16]), four(&[16, 17])]),
            2
        );
        assert_eq!(cover_of(&[four(&[7]), four(&[8]), four(&[9])]), 3);
    }
}
