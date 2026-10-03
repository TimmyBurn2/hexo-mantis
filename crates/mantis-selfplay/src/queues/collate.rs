// >300 justify (R8): the structural checks 4-13 and the pack they guard are one pass over one wire, kept beside
// each other so a check and the copy it licenses cannot drift apart.

//! `pack_wire`: checks 4–13 of `docs/contracts/graph_wire.md`, in order and by the reader's class names, fused with the pack.

use std::fmt;

use mantis_graph::{edge_code, edge_vocabulary, EDGE_FEAT_DIM, OFF_WINDOW_SLOT};

/// Policy slots `[0, POLICY_SLOTS)`; `OFF_WINDOW_SLOT` is the one negative slot the contract allows.
const POLICY_SLOTS: i32 = 362;
/// Below this many edges the copy runs on the calling thread: a spawn costs more than the share it would copy.
const PARALLEL_MIN_EDGES: usize = 1 << 16;

/// The structural contract errors, named as the Python reader's classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructuralKind {
    BatchCountMismatch,
    OffsetsNonMonotonic,
    NodeCountChecksum,
    EdgeIndexOutOfBounds,
    EdgeCrossesGraphBoundary,
    ScatterGatherCrossesGraph,
    ScatterSlotOutOfBounds,
    ScatterSlotAliasing,
    EmptyLegalSet,
    GatherNotStrictlyIncreasing,
    /// Check 14's class: the coded pack refuses a row outside the edge vocabulary once checks 4–13 hold.
    EdgeAttrGeometryMismatch,
}

impl StructuralKind {
    /// The Python class this kind is raised as.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::BatchCountMismatch => "BatchCountMismatch",
            Self::OffsetsNonMonotonic => "OffsetsNonMonotonic",
            Self::NodeCountChecksum => "NodeCountChecksum",
            Self::EdgeIndexOutOfBounds => "EdgeIndexOutOfBounds",
            Self::EdgeCrossesGraphBoundary => "EdgeCrossesGraphBoundary",
            Self::ScatterGatherCrossesGraph => "ScatterGatherCrossesGraph",
            Self::ScatterSlotOutOfBounds => "ScatterSlotOutOfBounds",
            Self::ScatterSlotAliasing => "ScatterSlotAliasing",
            Self::EmptyLegalSet => "EmptyLegalSet",
            Self::GatherNotStrictlyIncreasing => "GatherNotStrictlyIncreasing",
            Self::EdgeAttrGeometryMismatch => "EdgeAttrGeometryMismatch",
        }
    }
}

/// A wire the contract refuses, or a call the caller shaped wrong (its out slices, its dims).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PackError {
    Contract {
        kind: StructuralKind,
        message: String,
    },
    Caller(String),
}

impl fmt::Display for PackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Contract { kind, message } => write!(f, "{}: {message}", kind.name()),
            Self::Caller(message) => write!(f, "pack_wire: {message}"),
        }
    }
}

impl std::error::Error for PackError {}

fn refuse<T>(kind: StructuralKind, message: String) -> Result<T, PackError> {
    Err(PackError::Contract { kind, message })
}

/// The wire's flat arrays, read-only, as `GraphWire::take` yields them.
pub struct WireRef<'a> {
    pub n_graphs: usize,
    pub node_feat: &'a [f32],
    /// `[src (E) ‖ dst (E)]`.
    pub edge_index: &'a [i64],
    pub edge_attr: &'a [f32],
    pub node_offsets: &'a [i64],
    pub edge_offsets: &'a [i64],
    pub legal_offsets: &'a [i64],
    pub legal_node_gather: &'a [i64],
    pub policy_dst_slot: &'a [i32],
    pub n_nodes_checksum: &'a [u32],
    pub n_stones: &'a [u16],
    pub window_center: &'a [i32],
    pub current_player: &'a [i8],
}

/// Where the edges' features go: the rows themselves, or each row's code in `mantis_graph::edge_vocabulary`.
pub enum EdgeOut<'a> {
    Attr(&'a mut [f32]),
    Code { codes: &'a mut [u8], win_length: u8 },
}

/// The seven device arrays, each sized as its wire array (codes: one per edge); `n_stones` is widened to `i64`.
pub struct PackOut<'a> {
    pub x: &'a mut [f32],
    pub edge_index: &'a mut [i64],
    pub edges: EdgeOut<'a>,
    pub legal_offsets: &'a mut [i64],
    pub legal_node_gather: &'a mut [i64],
    pub node_offsets: &'a mut [i64],
    pub n_stones: &'a mut [i64],
}

/// Per non-empty graph: `(graph, src_lo, src_hi, dst_lo, dst_hi)` over its edge range.
type EdgeSpan = (usize, i64, i64, i64, i64);

/// One thread's spans and its first row outside the vocabulary.
type EdgeShare = (Vec<EdgeSpan>, Option<String>);

fn caller<T>(message: impl Into<String>) -> Result<T, PackError> {
    Err(PackError::Caller(message.into()))
}

fn at(offsets: &[i64], i: usize) -> Result<usize, PackError> {
    offsets
        .get(i)
        .and_then(|&v| usize::try_from(v).ok())
        .map_or_else(
            || caller(format!("offset {i} unreadable after the offsets check")),
            Ok,
        )
}

/// Checks 4–13, then the arrays into `out` alike at any `threads`; a coded off-vocabulary row is refused last (check 14's class).
pub fn pack_wire(
    w: &WireRef<'_>,
    out: &mut PackOut<'_>,
    node_feat_dim: usize,
    edge_feat_dim: usize,
    threads: usize,
) -> Result<(), PackError> {
    if node_feat_dim == 0 || edge_feat_dim == 0 {
        return caller("zero feature dim");
    }
    if let EdgeOut::Code { win_length, .. } = out.edges {
        if edge_feat_dim != EDGE_FEAT_DIM || edge_vocabulary(win_length).is_none() {
            return caller(format!(
                "no edge vocabulary codes {edge_feat_dim}-wide rows at win_length {win_length}"
            ));
        }
    }
    if !w.node_feat.len().is_multiple_of(node_feat_dim)
        || !w.edge_attr.len().is_multiple_of(edge_feat_dim)
    {
        return caller("feature arrays not divisible by their dims (checks 1-2 precede the pack)");
    }
    let n = w.node_feat.len() / node_feat_dim;
    let e = w.edge_attr.len() / edge_feat_dim;
    if w.edge_index.len() != 2 * e {
        return caller("edge_index is not 2E (check 2 precedes the pack)");
    }
    let shapes = [
        (out.x.len(), w.node_feat.len()),
        (out.edge_index.len(), w.edge_index.len()),
        match &out.edges {
            EdgeOut::Attr(attr) => (attr.len(), w.edge_attr.len()),
            EdgeOut::Code { codes, .. } => (codes.len(), e),
        },
        (out.legal_offsets.len(), w.legal_offsets.len()),
        (out.legal_node_gather.len(), w.legal_node_gather.len()),
        (out.node_offsets.len(), w.node_offsets.len()),
        (out.n_stones.len(), w.n_stones.len()),
    ];
    if shapes.iter().any(|(o, i)| o != i) {
        return caller(format!("out slices sized {shapes:?} (out, wire)"));
    }
    check_counts(w)?;
    let lg = w.legal_node_gather.len();
    check_offsets(w, n, e, lg)?;
    check_checksums(w)?;
    let miss = if e > 0 {
        pack_edges(w, out, e, edge_feat_dim, n, threads)?
    } else {
        None
    };
    check_and_pack_gather(w, out, n)?;
    if let Some(message) = miss {
        return refuse(StructuralKind::EdgeAttrGeometryMismatch, message);
    }
    out.x.copy_from_slice(w.node_feat);
    out.legal_offsets.copy_from_slice(w.legal_offsets);
    out.node_offsets.copy_from_slice(w.node_offsets);
    for (dst, &s) in out.n_stones.iter_mut().zip(w.n_stones) {
        *dst = i64::from(s);
    }
    Ok(())
}

/// Check 4: every per-graph array is sized by `B`, and the slots by `Lg`.
fn check_counts(w: &WireRef<'_>) -> Result<(), PackError> {
    let b = w.n_graphs;
    let (Some(b1), Some(b2)) = (b.checked_add(1), b.checked_mul(2)) else {
        return refuse(
            StructuralKind::BatchCountMismatch,
            format!("B={b} sizes no array"),
        );
    };
    for (name, len, want) in [
        ("node_offsets", w.node_offsets.len(), b1),
        ("edge_offsets", w.edge_offsets.len(), b1),
        ("legal_offsets", w.legal_offsets.len(), b1),
        ("n_nodes_checksum", w.n_nodes_checksum.len(), b),
        ("n_stones", w.n_stones.len(), b),
        ("current_player", w.current_player.len(), b),
        ("window_center", w.window_center.len(), b2),
    ] {
        if len != want {
            return refuse(
                StructuralKind::BatchCountMismatch,
                format!("len({name})={len} != {want} (B={b})"),
            );
        }
    }
    let lg = w.legal_node_gather.len();
    if w.policy_dst_slot.len() != lg {
        return refuse(
            StructuralKind::BatchCountMismatch,
            format!(
                "len(policy_dst_slot)={} != Lg={lg}",
                w.policy_dst_slot.len()
            ),
        );
    }
    Ok(())
}

/// Check 5: each offset array starts at 0, ends at its total and never decreases.
fn check_offsets(w: &WireRef<'_>, n: usize, e: usize, lg: usize) -> Result<(), PackError> {
    for (name, off, total) in [
        ("node_offsets", w.node_offsets, n),
        ("edge_offsets", w.edge_offsets, e),
        ("legal_offsets", w.legal_offsets, lg),
    ] {
        let (first, last) = (off[0], off[off.len() - 1]);
        if first != 0 {
            return refuse(
                StructuralKind::OffsetsNonMonotonic,
                format!("{name}[0]={first} != 0"),
            );
        }
        if usize::try_from(last).ok() != Some(total) {
            return refuse(
                StructuralKind::OffsetsNonMonotonic,
                format!("{name}[B]={last} != total {total}"),
            );
        }
        if off.windows(2).any(|p| p[1] < p[0]) {
            return refuse(
                StructuralKind::OffsetsNonMonotonic,
                format!("{name} not non-decreasing"),
            );
        }
    }
    Ok(())
}

/// Check 6: each graph's node count is its checksum, and holds its stones plus the dummy.
fn check_checksums(w: &WireRef<'_>) -> Result<(), PackError> {
    let counts = w.node_offsets.windows(2).map(|p| p[1] - p[0]);
    if counts
        .zip(w.n_nodes_checksum)
        .any(|(c, &s)| c != i64::from(s))
    {
        return refuse(
            StructuralKind::NodeCountChecksum,
            "per-graph node count != n_nodes_checksum".into(),
        );
    }
    if w.n_stones
        .iter()
        .zip(w.n_nodes_checksum)
        .any(|(&s, &c)| u32::from(s) + 1 > c)
    {
        return refuse(
            StructuralKind::NodeCountChecksum,
            "n_stones + 1 > n_nodes_checksum for some graph".into(),
        );
    }
    Ok(())
}

/// Checks 7–8 over the copy (in `[0, N)`, then in its graph); coded, returns the first off-vocabulary row for the caller.
fn pack_edges(
    w: &WireRef<'_>,
    out: &mut PackOut<'_>,
    n_edges: usize,
    edge_feat_dim: usize,
    n_nodes: usize,
    threads: usize,
) -> Result<Option<String>, PackError> {
    let threads = if n_edges < PARALLEL_MIN_EDGES {
        1
    } else {
        threads.max(1)
    };
    let groups = edge_groups(w.edge_offsets, w.n_graphs, n_edges, threads)?;
    let (mut src_out, mut dst_out) = out.edge_index.split_at_mut(n_edges);
    let (mut feat_out, win_length) = match &mut out.edges {
        EdgeOut::Attr(attr) => (EdgeChunk::Attr(attr), 0),
        EdgeOut::Code { codes, win_length } => (EdgeChunk::Code(codes), *win_length),
    };
    let mut jobs = Vec::with_capacity(groups.len());
    for &(g0, g1) in &groups {
        let (e0, e1) = (at(w.edge_offsets, g0)?, at(w.edge_offsets, g1)?);
        let (src, src_rest) = std::mem::take(&mut src_out).split_at_mut(e1 - e0);
        let (dst, dst_rest) = std::mem::take(&mut dst_out).split_at_mut(e1 - e0);
        let (feat, feat_rest) = feat_out.split_at(e1 - e0, edge_feat_dim);
        (src_out, dst_out, feat_out) = (src_rest, dst_rest, feat_rest);
        jobs.push((g0, g1, e0, src, dst, feat));
    }
    let run = |job: ShareJob<'_>| pack_share(w, job, edge_feat_dim, win_length);
    let run = &run;
    let mut jobs = jobs.into_iter();
    let first = jobs.next();
    // The first share runs on the calling thread; every other is joined before any error propagates.
    let shares: Vec<Result<EdgeShare, PackError>> = std::thread::scope(|scope| {
        let handles: Vec<_> = jobs
            .map(|job| std::thread::Builder::new().spawn_scoped(scope, move || run(job)))
            .collect();
        let mut shares = vec![first.map_or_else(|| Ok((Vec::new(), None)), run)];
        for handle in handles {
            shares.push(match handle {
                Ok(h) => h
                    .join()
                    .unwrap_or_else(|_| Err(PackError::Caller("a pack thread panicked".into()))),
                Err(e) => Err(PackError::Caller(format!(
                    "a pack thread could not start: {e}"
                ))),
            });
        }
        shares
    });
    let mut spans: Vec<EdgeSpan> = Vec::new();
    let mut misses: Vec<Option<String>> = Vec::new();
    for share in shares {
        let (s, m) = share?;
        spans.extend(s);
        misses.push(m);
    }
    let n = i64::try_from(n_nodes).map_err(|_| PackError::Caller("N past i64".into()))?;
    let lo = spans.iter().map(|s| s.1.min(s.3)).min().unwrap_or(0);
    let hi = spans.iter().map(|s| s.2.max(s.4)).max().unwrap_or(0);
    if lo < 0 || hi >= n {
        return refuse(
            StructuralKind::EdgeIndexOutOfBounds,
            format!("edge_index out of [0,{n})"),
        );
    }
    for &(g, slo, shi, dlo, dhi) in &spans {
        let (seg_lo, seg_hi) = (w.node_offsets[g], w.node_offsets[g + 1]);
        if slo < seg_lo || shi >= seg_hi || dlo < seg_lo || dhi >= seg_hi {
            return refuse(
                StructuralKind::EdgeCrossesGraphBoundary,
                "an edge endpoint is outside its own graph's node range".into(),
            );
        }
    }
    Ok(misses.into_iter().flatten().next())
}

/// One thread's graphs `[g0, g1)`, their first edge, and its slices of the edge outputs.
type ShareJob<'a> = (
    usize,
    usize,
    usize,
    &'a mut [i64],
    &'a mut [i64],
    EdgeChunk<'a>,
);

/// Copy (or code) one share of the edges and read each of its non-empty graphs' endpoint spans.
fn pack_share(
    w: &WireRef<'_>,
    (g0, g1, base, src, dst, feat): ShareJob<'_>,
    edge_feat_dim: usize,
    win_length: u8,
) -> Result<EdgeShare, PackError> {
    let n_edges = w.edge_index.len() / 2;
    let (src_in, dst_in) = w.edge_index.split_at(n_edges);
    let end = base + src.len();
    let mut miss: Option<String> = None;
    src.copy_from_slice(&src_in[base..end]);
    dst.copy_from_slice(&dst_in[base..end]);
    let rows = &w.edge_attr[base * edge_feat_dim..end * edge_feat_dim];
    match feat {
        EdgeChunk::Attr(attr) => attr.copy_from_slice(rows),
        EdgeChunk::Code(codes) => {
            for (k, (code, row)) in codes.iter_mut().zip(rows.chunks(edge_feat_dim)).enumerate() {
                let Some(c) = edge_code(row, win_length) else {
                    miss = Some(format!(
                        "edge {} attr {row:?} is outside the edge vocabulary (win_length {win_length})",
                        base + k
                    ));
                    break;
                };
                *code = c;
            }
        }
    }
    let mut spans: Vec<EdgeSpan> = Vec::new();
    for g in g0..g1 {
        let (lo, hi) = (at(w.edge_offsets, g)?, at(w.edge_offsets, g + 1)?);
        if hi > lo {
            let (slo, shi) = min_max(&src_in[lo..hi]);
            let (dlo, dhi) = min_max(&dst_in[lo..hi]);
            spans.push((g, slo, shi, dlo, dhi));
        }
    }
    Ok((spans, miss))
}

/// One thread's share of the edge-feature output.
enum EdgeChunk<'a> {
    Attr(&'a mut [f32]),
    Code(&'a mut [u8]),
}

impl EdgeChunk<'_> {
    /// The first `edges` edges' share, and the rest.
    fn split_at(self, edges: usize, edge_feat_dim: usize) -> (Self, Self) {
        match self {
            Self::Attr(a) => {
                let (head, tail) = a.split_at_mut(edges * edge_feat_dim);
                (Self::Attr(head), Self::Attr(tail))
            }
            Self::Code(c) => {
                let (head, tail) = c.split_at_mut(edges);
                (Self::Code(head), Self::Code(tail))
            }
        }
    }
}

/// Contiguous graph ranges of about `E / threads` edges each, at most `threads` of them, each holding a graph.
fn edge_groups(
    edge_offsets: &[i64],
    b: usize,
    e: usize,
    threads: usize,
) -> Result<Vec<(usize, usize)>, PackError> {
    let want = threads.min(b).max(1);
    let mut groups = Vec::with_capacity(want);
    let mut g0 = 0;
    for k in 1..want {
        let target = e * k / want;
        let mut g1 = g0;
        while g1 < b && at(edge_offsets, g1 + 1)? <= target {
            g1 += 1;
        }
        if g1 > g0 {
            groups.push((g0, g1));
            g0 = g1;
        }
    }
    groups.push((g0, b));
    Ok(groups)
}

fn min_max(xs: &[i64]) -> (i64, i64) {
    xs.iter()
        .fold((i64::MAX, i64::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)))
}

/// Checks 9–13 over the gather and its slots, then the gather's copy.
fn check_and_pack_gather(
    w: &WireRef<'_>,
    out: &mut PackOut<'_>,
    n: usize,
) -> Result<(), PackError> {
    let gather = w.legal_node_gather;
    let lg = gather.len();
    let n = i64::try_from(n).map_err(|_| PackError::Caller("N past i64".into()))?;
    if lg > 0 {
        let (lo, hi) = min_max(gather);
        if lo < 0 || hi >= n {
            return refuse(
                StructuralKind::ScatterGatherCrossesGraph,
                format!(
                    "legal_node_gather outside [0,{n}): [{lo}, {hi}] — a row that is in no graph at all, \
                     not merely in the wrong one"
                ),
            );
        }
        for g in 0..w.n_graphs {
            let rows = &gather[at(w.legal_offsets, g)?..at(w.legal_offsets, g + 1)?];
            let (seg_lo, seg_hi) = (w.node_offsets[g], w.node_offsets[g + 1]);
            if rows.iter().any(|&r| r < seg_lo || r >= seg_hi) {
                return refuse(
                    StructuralKind::ScatterGatherCrossesGraph,
                    "legal_node_gather points into another graph".into(),
                );
            }
        }
    }
    if w.policy_dst_slot
        .iter()
        .any(|&s| s >= POLICY_SLOTS || (s < 0 && s != OFF_WINDOW_SLOT))
    {
        return refuse(
            StructuralKind::ScatterSlotOutOfBounds,
            "policy_dst_slot out of [0,362) and not the -1 sentinel".into(),
        );
    }
    if lg > 0 {
        let mut seen = [false; POLICY_SLOTS as usize];
        for g in 0..w.n_graphs {
            seen.fill(false);
            for &s in &w.policy_dst_slot[at(w.legal_offsets, g)?..at(w.legal_offsets, g + 1)?] {
                let Ok(slot) = usize::try_from(s) else {
                    continue;
                };
                if std::mem::replace(&mut seen[slot], true) {
                    return refuse(
                        StructuralKind::ScatterSlotAliasing,
                        "two legal nodes in one graph map to the same slot".into(),
                    );
                }
            }
        }
    }
    if w.legal_offsets.windows(2).any(|p| p[1] == p[0]) {
        return refuse(
            StructuralKind::EmptyLegalSet,
            "a graph has an empty legal set".into(),
        );
    }
    if let Some(i) = gather.windows(2).position(|p| p[1] <= p[0]) {
        return refuse(
            StructuralKind::GatherNotStrictlyIncreasing,
            format!(
                "legal_node_gather not strictly increasing at i={}: {} -> {}",
                i + 1,
                gather[i],
                gather[i + 1]
            ),
        );
    }
    out.legal_node_gather.copy_from_slice(gather);
    Ok(())
}

#[cfg(test)]
#[path = "collate_tests.rs"]
mod tests;
