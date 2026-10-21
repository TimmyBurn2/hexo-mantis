// >300 justify (R8): the derivation and the tests that pin each derived array against the offsets it came from read one
// table (contract #13's batch), kept in one file so an array and its pin cannot drift apart.

//! The batch both net backends read (contract #13): the served path's pack, every array beyond `x` as int32.

use std::fmt;

use mantis_encoding::RegistrySpec;
use mantis_selfplay::queues::collate::{pack_wire, EdgeOut, PackError, PackOut, WireRef};
use mantis_selfplay::queues::graph::build_leaf_graph;
use mantis_selfplay::queues::wire::GraphWire;
use mantis_selfplay::runner::params::{resolve_geometry, GeometryError};

/// One leaf as the served path requests it: stones `(q, r, player)`, the side to move, its stones left this turn.
pub type Position = (Vec<(i64, i64, i64)>, i64, i64);

/// `n_graphs` leaves block-diagonally, rows global; N nodes, E edges, L legal rows (contract #13's batch table).
#[derive(Debug, Clone, PartialEq)]
pub struct WebBatch {
    pub n_graphs: usize,
    /// `[N, node_feat_dim]` row-major, as packed.
    pub x: Vec<f32>,
    pub src: Vec<i32>,
    pub dst: Vec<i32>,
    /// Each edge's row in the encoding's edge vocabulary.
    pub code: Vec<i32>,
    /// The legal cells' node rows, strictly ascending.
    pub legal: Vec<i32>,
    pub node_graph: Vec<i32>,
    pub legal_graph: Vec<i32>,
    /// 1 on a real node (a stone or legal), else max(in-degree, 1).
    pub divisor: Vec<i32>,
    /// 1 on a real node, else 0.
    pub real: Vec<i32>,
    /// 1 on a stone, or on every node of a graph with no stones: the value head's mean.
    pub mean_mask: Vec<i32>,
    /// `[G]`: each graph's `mean_mask` count.
    pub mean_den: Vec<i32>,
}

/// Why a batch could not be built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchError {
    Empty,
    Geometry(GeometryError),
    MissingKey(&'static str),
    Leaf(String),
    WireConsumed,
    Pack(PackError),
    TooLarge(&'static str),
}

impl fmt::Display for BatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "web batch: no positions"),
            Self::Geometry(e) => write!(f, "web batch: {e}"),
            Self::MissingKey(key) => write!(f, "web batch: the encoding row has no `{key}`"),
            Self::Leaf(reason) => write!(f, "web batch: {reason}"),
            Self::WireConsumed => write!(f, "web batch: the fused wire was read twice"),
            Self::Pack(e) => write!(f, "web batch: {e}"),
            Self::TooLarge(what) => write!(f, "web batch: a {what} does not fit int32"),
        }
    }
}

impl std::error::Error for BatchError {}

fn narrow(value: i64, what: &'static str) -> Result<i32, BatchError> {
    i32::try_from(value).map_err(|_| BatchError::TooLarge(what))
}

fn row(value: i64, what: &'static str) -> Result<usize, BatchError> {
    usize::try_from(value).map_err(|_| BatchError::TooLarge(what))
}

/// The batch of `positions` under `spec`'s geometry, or the [`BatchError`] that names why not.
#[allow(clippy::missing_errors_doc)]
pub fn build_batch(
    positions: &[Position],
    spec: &'static RegistrySpec,
) -> Result<WebBatch, BatchError> {
    derive(pack(positions, spec)?)
}

/// The coded collate's seven arrays, as `pack_wire` leaves them.
struct Packed {
    n_graphs: usize,
    x: Vec<f32>,
    edge_index: Vec<i64>,
    codes: Vec<u8>,
    legal_offsets: Vec<i64>,
    gather: Vec<i64>,
    node_offsets: Vec<i64>,
    n_stones: Vec<i64>,
}

fn pack(positions: &[Position], spec: &'static RegistrySpec) -> Result<Packed, BatchError> {
    if positions.is_empty() {
        return Err(BatchError::Empty);
    }
    let geometry = resolve_geometry(spec).map_err(BatchError::Geometry)?;
    let node_feat_dim = spec
        .node_feat_dim
        .ok_or(BatchError::MissingKey("node_feat_dim"))?;
    let edge_feat_dim = spec
        .edge_feat_dim
        .ok_or(BatchError::MissingKey("edge_feat_dim"))?;
    let contract_version = spec
        .contract_version
        .ok_or(BatchError::MissingKey("contract_version"))?;
    let graphs = positions
        .iter()
        .map(|(stones, player, left)| {
            build_leaf_graph(
                stones,
                *player,
                *left,
                geometry.win_length,
                geometry.graph_radius,
                geometry.agg_trunk_sz,
                geometry.empty_edges,
            )
            .map_err(BatchError::Leaf)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let fused = GraphWire::from_axis_graphs(&graphs, contract_version)
        .take()
        .map_err(|_| BatchError::WireConsumed)?;

    let (n_graphs, n_edges) = (fused.n_graphs, fused.edge_index.len() / 2);
    let n_legal = fused.legal_node_gather.len();
    let mut x = vec![0.0_f32; fused.node_feat.len()];
    let mut edge_index = vec![0_i64; 2 * n_edges];
    let mut codes = vec![0_u8; n_edges];
    let (mut legal_offsets, mut node_offsets) =
        (vec![0_i64; n_graphs + 1], vec![0_i64; n_graphs + 1]);
    let (mut gather, mut n_stones) = (vec![0_i64; n_legal], vec![0_i64; n_graphs]);
    let wire = WireRef {
        n_graphs,
        node_feat: &fused.node_feat,
        node_coords: &fused.node_coords,
        edge_index: &fused.edge_index,
        edge_attr: &fused.edge_attr,
        node_offsets: &fused.node_offsets,
        edge_offsets: &fused.edge_offsets,
        legal_offsets: &fused.legal_offsets,
        legal_node_gather: &fused.legal_node_gather,
        policy_dst_slot: &fused.policy_dst_slot,
        n_nodes_checksum: &fused.n_nodes_checksum,
        n_stones: &fused.n_stones,
        window_center: &fused.window_center,
        current_player: &fused.current_player,
    };
    let mut out = PackOut {
        x: &mut x,
        edge_index: &mut edge_index,
        edges: EdgeOut::Code {
            codes: &mut codes,
            win_length: geometry.win_length,
        },
        legal_offsets: &mut legal_offsets,
        legal_node_gather: &mut gather,
        node_offsets: &mut node_offsets,
        n_stones: &mut n_stones,
    };
    // One thread: the pack's single share runs on the calling thread, as wasm32 requires.
    pack_wire(
        &wire,
        &mut out,
        node_feat_dim,
        edge_feat_dim,
        geometry.agg_trunk_sz,
        1,
        None,
    )
    .map_err(BatchError::Pack)?;
    Ok(Packed {
        n_graphs,
        x,
        edge_index,
        codes,
        legal_offsets,
        gather,
        node_offsets,
        n_stones,
    })
}

/// The pack's checks hold here: offsets ascend within N and L, edges and gathers stay in their graphs (4–13), and a graph's legal count is its own (15).
fn derive(packed: Packed) -> Result<WebBatch, BatchError> {
    let Packed {
        n_graphs,
        x,
        edge_index,
        codes,
        legal_offsets,
        gather,
        node_offsets,
        n_stones,
    } = packed;
    let (n_edges, n_legal) = (codes.len(), gather.len());
    let n_nodes = row(node_offsets[n_graphs], "node row")?;
    let src = edge_index[..n_edges]
        .iter()
        .map(|&v| narrow(v, "node row"))
        .collect::<Result<Vec<_>, _>>()?;
    let dst = edge_index[n_edges..]
        .iter()
        .map(|&v| narrow(v, "node row"))
        .collect::<Result<Vec<_>, _>>()?;
    let legal = gather
        .iter()
        .map(|&v| narrow(v, "node row"))
        .collect::<Result<Vec<_>, _>>()?;
    let code = codes.iter().map(|&c| i32::from(c)).collect();
    let (mut node_graph, mut legal_graph) = (vec![0_i32; n_nodes], vec![0_i32; n_legal]);
    let (mut real, mut mean_mask) = (vec![0_i32; n_nodes], vec![0_i32; n_nodes]);
    let mut mean_den = vec![0_i32; n_graphs];
    for gi in 0..n_graphs {
        let graph = i32::try_from(gi).map_err(|_| BatchError::TooLarge("graph index"))?;
        let (n0, n1) = (
            row(node_offsets[gi], "node row")?,
            row(node_offsets[gi + 1], "node row")?,
        );
        let (l0, l1) = (
            row(legal_offsets[gi], "legal row")?,
            row(legal_offsets[gi + 1], "legal row")?,
        );
        let stones = row(n_stones[gi], "stone count")?;
        node_graph[n0..n1].fill(graph);
        legal_graph[l0..l1].fill(graph);
        // The builder lays a graph's rows out [stones | legal | dummy].
        real[n0..n0 + stones].fill(1);
        let mask_end = if stones > 0 { n0 + stones } else { n1 };
        mean_mask[n0..mask_end].fill(1);
        mean_den[gi] =
            i32::try_from(mask_end - n0).map_err(|_| BatchError::TooLarge("node count"))?;
    }
    for &r in &gather {
        real[row(r, "node row")?] = 1;
    }
    let mut in_degree = vec![0_i32; n_nodes];
    for &d in &edge_index[n_edges..] {
        in_degree[row(d, "node row")?] += 1;
    }
    let divisor = real
        .iter()
        .zip(&in_degree)
        .map(|(&r, &d)| if r == 1 { 1 } else { d.max(1) })
        .collect();
    Ok(WebBatch {
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
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOLDENS: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/graph_build/positions_r8.txt"
    );

    fn goldens(n: usize) -> Vec<Position> {
        let text =
            std::fs::read_to_string(GOLDENS).expect("the golden positions fixture is present");
        text.lines()
            .filter_map(|line| line.strip_prefix("S "))
            .take(n)
            .map(|fields| {
                let v: Vec<i64> = fields
                    .split_whitespace()
                    .map(|f| f.parse().expect("an integer field"))
                    .collect();
                let count = usize::try_from(v[2]).expect("a stone count");
                (
                    (0..count)
                        .map(|i| (v[3 + 3 * i], v[4 + 3 * i], v[5 + 3 * i]))
                        .collect(),
                    v[0],
                    v[1],
                )
            })
            .collect()
    }

    fn spec(name: &str) -> &'static RegistrySpec {
        mantis_encoding::lookup(name).expect("a registered row")
    }

    #[test]
    fn every_derived_array_agrees_with_the_offsets_it_came_from() {
        let positions = goldens(16);
        let b = build_batch(&positions, spec("gnn_axis_r8")).expect("the goldens build");
        assert_eq!(b.n_graphs, 16);
        let n = b.node_graph.len();
        assert_eq!(b.x.len(), n * 11);
        assert!(b.node_graph.windows(2).all(|w| w[0] <= w[1]) && b.node_graph[n - 1] == 15);
        assert!(b.legal.windows(2).all(|w| w[0] < w[1]), "legal rows ascend");
        for (j, &r) in b.legal.iter().enumerate() {
            let r = usize::try_from(r).expect("a row");
            assert_eq!(
                b.node_graph[r], b.legal_graph[j],
                "a legal row stays in its graph"
            );
            assert_eq!((b.real[r], b.divisor[r]), (1, 1));
        }
        for (s, d) in b.src.iter().zip(&b.dst) {
            let (s, d) = (
                usize::try_from(*s).expect("a row"),
                usize::try_from(*d).expect("a row"),
            );
            assert_eq!(
                b.node_graph[s], b.node_graph[d],
                "an edge stays in its graph"
            );
        }
        for (g, (stones, _, _)) in positions.iter().enumerate() {
            let rows = b
                .node_graph
                .iter()
                .enumerate()
                .filter(|&(_, &gi)| gi == i32::try_from(g).expect("g"));
            let mask: Vec<i32> = rows.map(|(i, _)| b.mean_mask[i]).collect();
            assert_eq!(mask.iter().sum::<i32>(), b.mean_den[g]);
            assert_eq!(
                usize::try_from(b.mean_den[g]).expect("a count"),
                stones.len(),
                "stones are the mean's rows"
            );
        }
        let mut in_degree = vec![0; n];
        for &d in &b.dst {
            in_degree[usize::try_from(d).expect("a row")] += 1;
        }
        let legal: std::collections::HashSet<i32> = b.legal.iter().copied().collect();
        for (row, (&graph, &degree)) in b.node_graph.iter().zip(&in_degree).enumerate() {
            let first = b
                .node_graph
                .iter()
                .position(|&g| g == graph)
                .expect("its first row");
            let stones = positions[usize::try_from(graph).expect("a graph")].0.len();
            let real = row - first < stones || legal.contains(&i32::try_from(row).expect("a row"));
            assert_eq!(
                b.real[row],
                i32::from(real),
                "row {row}: real is a stone or a legal row"
            );
            let want = if real { 1 } else { degree.max(1) };
            assert_eq!(
                b.divisor[row], want,
                "row {row}: a dummy divides by its in-degree"
            );
        }
        assert!(b.real.contains(&0), "the goldens hold dummy rows to pin");
    }

    #[test]
    fn an_empty_board_takes_the_mean_over_every_node() {
        let b = build_batch(&[(Vec::new(), 1, 1)], spec("gnn_axis_r8"))
            .expect("the empty board builds");
        assert!(b.mean_mask.iter().all(|&m| m == 1));
        assert_eq!(
            usize::try_from(b.mean_den[0]).expect("a count"),
            b.node_graph.len()
        );
    }

    #[test]
    fn the_pruned_row_builds_fewer_edges_on_the_same_nodes() {
        let positions = goldens(4);
        let kept = build_batch(&positions, spec("gnn_axis_r8")).expect("kept builds");
        let pruned = build_batch(&positions, spec("gnn_axis_r8_pruned")).expect("pruned builds");
        assert_eq!(kept.x, pruned.x);
        assert!(pruned.src.len() < kept.src.len());
    }

    #[test]
    fn a_canvas_row_and_an_empty_request_are_refused_by_name() {
        let positions = goldens(1);
        assert!(matches!(
            build_batch(&positions, spec("hexcanvas_r8")),
            Err(BatchError::Geometry(_))
        ));
        assert_eq!(
            build_batch(&[], spec("gnn_axis_r8")),
            Err(BatchError::Empty)
        );
    }
}
