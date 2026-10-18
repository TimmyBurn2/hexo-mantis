//! The hex-canvas builder (contract #12): the fence, stones ∪ the legal set, masked inside its smallest axial box.

use super::{legal_cells, stone_table, window_center, window_flat_idx, StoneList};

/// The canvas planes in wire order; a `hexcanvas` registry row's `plane_layout` is validated against it.
pub const CANVAS_PLANES: [&str; 6] = [
    "own",
    "opponent",
    "legal",
    "fence",
    "stones_left_1",
    "stones_left_2",
];

/// `legal_gather` of a legal cell outside a far region: it has no cell, and the far logit carries it.
pub const FAR_CELL: i32 = -1;

const OWN: usize = 0;
const OPPONENT: usize = 1;
const LEGAL: usize = 2;
const FENCE: usize = 3;
const STONES_LEFT_1: usize = 4;

/// The builder's inputs: the row's legal radius, reply trunk and canvas keys, then the position's side and stones left.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CanvasParams {
    pub radius: u16,
    pub trunk_size: i32,
    pub side_step: u16,
    pub min_side: u16,
    /// S_max: past this smallest square bucket the region is the far one.
    pub max_side: u16,
    pub far_halo: u16,
    /// +1 / −1 side to move (anything else is −1, as the graph builder treats it).
    pub current_player: i8,
    pub moves_remaining: u8,
}

/// One built canvas: contract #12's per-leaf fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Canvas {
    /// The axis pair rows and columns run along: 0 (q, r), 1 (r, s), 2 (s, q), with s = −q − r.
    pub axis_pair: u8,
    pub height: usize,
    pub width: usize,
    /// The pair coordinates of cell (0, 0): the region's minimum corner.
    pub origin: (i32, i32),
    pub far: bool,
    /// The smallest of the three pairs' square buckets: the position's size class, invariant under D6.
    pub fit_side: usize,
    /// `[CANVAS_PLANES.len(), height, width]` row-major, each 0 or 1.
    pub planes: Vec<u8>,
    /// Every legal cell `(q, r)`, lexicographic: the graph builder's legal nodes in its order.
    pub legal: Vec<(i32, i32)>,
    /// Per legal cell, its cell index `row * width + col`, or `FAR_CELL` outside a far region.
    pub legal_gather: Vec<i32>,
    /// Per legal cell, its slot in the trunk-sized reply window, `OFF_WINDOW_SLOT` off it.
    pub policy_dst_slot: Vec<i32>,
    pub window_center: (i32, i32),
}

/// A cell's coordinates in the three axis pairs, by id.
#[inline]
#[must_use]
pub fn pair_coords(q: i32, r: i32) -> [(i32, i32); 3] {
    let s = -q - r;
    [(q, r), (r, s), (s, q)]
}

/// An extent rounded up to the side quantum, at least the minimum side.
#[inline]
fn container_side(extent: usize, step: usize, min: usize) -> usize {
    extent.div_ceil(step).saturating_mul(step).max(min)
}

/// Each pair's minimum corner and extents over `cells`, and its square bucket.
struct Boxes {
    lo: [(i32, i32); 3],
    extent: [(usize, usize); 3],
    bucket: [usize; 3],
}

impl Boxes {
    fn of(cells: &[(i32, i32)], step: usize, min: usize) -> Self {
        let mut lo = [(i32::MAX, i32::MAX); 3];
        let mut hi = [(i32::MIN, i32::MIN); 3];
        for &(q, r) in cells {
            for (p, (a, b)) in pair_coords(q, r).into_iter().enumerate() {
                lo[p] = (lo[p].0.min(a), lo[p].1.min(b));
                hi[p] = (hi[p].0.max(a), hi[p].1.max(b));
            }
        }
        let extent: [(usize, usize); 3] = std::array::from_fn(|p| {
            (
                (hi[p].0 - lo[p].0 + 1) as usize,
                (hi[p].1 - lo[p].1 + 1) as usize,
            )
        });
        let bucket = extent.map(|(ea, eb)| container_side(ea.max(eb), step, min));
        Self { lo, extent, bucket }
    }

    fn fit(&self) -> usize {
        self.bucket.iter().copied().min().unwrap_or(usize::MAX)
    }

    /// The lowest-id pair within one step of the smallest bucket: a function of the region alone.
    fn served(&self, step: usize) -> usize {
        let band = self.fit().saturating_add(step);
        (0..3).find(|&p| self.bucket[p] <= band).unwrap_or(0)
    }
}

/// Build one canvas: the fence (the empty board's 5x5 first-stone box with no stone), or past `max_side` the stones and their `far_halo`.
#[must_use]
#[allow(clippy::missing_panics_doc)] // panics ARE the contract: a parameter outside its domain dies loud
pub fn build_canvas(stones_in: &StoneList, params: &CanvasParams) -> Canvas {
    assert!(params.side_step >= 1, "CanvasParams: side_step 0");
    assert!(
        params.trunk_size >= 1,
        "CanvasParams: trunk_size {} < 1",
        params.trunk_size
    );
    assert!(
        matches!(params.moves_remaining, 1 | 2),
        "CanvasParams: moves_remaining {} lights no stones-left plane",
        params.moves_remaining
    );
    assert!(
        params.far_halo >= 1 && params.max_side >= params.min_side,
        "CanvasParams: far_halo {} or max_side {} below min_side {} leaves an empty far region",
        params.far_halo,
        params.max_side,
        params.min_side
    );
    let (step, min) = (usize::from(params.side_step), usize::from(params.min_side));
    let to_move: i8 = if params.current_player == 1 { 1 } else { -1 };

    let (stone_map, stones) = stone_table(stones_in);
    let (legal, legal_dist) = legal_cells(&stone_map, &stones, i32::from(params.radius));

    let fence: Vec<(i32, i32)> = stones
        .iter()
        .map(|&(q, r, _)| (q, r))
        .chain(legal.iter().copied())
        .collect();
    let mut boxes = Boxes::of(&fence, step, min);
    let far = boxes.fit() > usize::from(params.max_side);
    let halo = i32::from(params.far_halo);
    let in_region = |d: i32| !far || d <= halo;
    let region: Vec<(i32, i32)> = if far {
        let near = legal
            .iter()
            .zip(&legal_dist)
            .filter(|&(_, &d)| d <= halo)
            .map(|(&c, _)| c);
        let cells: Vec<(i32, i32)> = stones.iter().map(|&(q, r, _)| (q, r)).chain(near).collect();
        boxes = Boxes::of(&cells, step, min);
        cells
    } else {
        fence
    };

    let pair = boxes.served(step);
    let (height, width) = (
        container_side(boxes.extent[pair].0, step, min),
        container_side(boxes.extent[pair].1, step, min),
    );
    let origin = boxes.lo[pair];
    let cell = |q: i32, r: i32| {
        let (a, b) = pair_coords(q, r)[pair];
        ((a - origin.0) as usize) * width + (b - origin.1) as usize
    };
    let area = height * width;
    let mut planes = vec![0u8; CANVAS_PLANES.len() * area];
    let left_plane = match params.moves_remaining {
        1 => Some(STONES_LEFT_1),
        2 => Some(STONES_LEFT_1 + 1),
        _ => None,
    };
    for &(q, r) in &region {
        let i = cell(q, r);
        planes[FENCE * area + i] = 1;
        if let Some(k) = left_plane {
            planes[k * area + i] = 1;
        }
    }
    for &(q, r, p) in &stones {
        let side = if p == to_move { OWN } else { OPPONENT };
        planes[side * area + cell(q, r)] = 1;
    }
    let mut legal_gather = Vec::with_capacity(legal.len());
    for (&(q, r), &d) in legal.iter().zip(&legal_dist) {
        if in_region(d) {
            let i = cell(q, r);
            planes[LEGAL * area + i] = 1;
            legal_gather.push(i as i32);
        } else {
            legal_gather.push(FAR_CELL);
        }
    }
    let wc = window_center(&stones);
    let policy_dst_slot = legal
        .iter()
        .map(|&(q, r)| window_flat_idx(q, r, wc.0, wc.1, params.trunk_size))
        .collect();
    Canvas {
        axis_pair: pair as u8,
        height,
        width,
        origin,
        far,
        fit_side: boxes.fit(),
        planes,
        legal,
        legal_gather,
        policy_dst_slot,
        window_center: wc,
    }
}
