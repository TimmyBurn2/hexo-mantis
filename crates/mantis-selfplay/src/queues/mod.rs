//! Inference queues — the pure-Rust (pyo3/numpy-STRIPPED) half of the frozen
//! `inference_bridge.rs` (WP6 D4/D5/D6). Two DISJOINT modules
//! (ADJUDICATION L61 forbids dense/graph ring unification): the dense
//! `Vec<f32>` batcher ([`dense::DenseQueue`]) and the parallel graph
//! `AxisGraph` batcher ([`graph::GraphQueue`]); the graph batcher never touches
//! the dense pool. [`wire::GraphWire`] is the block-diagonal fuse tensor with a
//! single-read `take()`.
//!
//! WP6 exposes PURE-RUST producer/consumer APIs only (LOCKED DECISION 4): the
//! consumer face is `submit_batch_and_wait` / `submit_graph_and_wait`; the
//! producer face is `pop_batch` + `submit_results` (dense) and
//! `pop_graph_batch` + `submit_graph_results` (graph). The NN itself + the pyo3
//! producer face (`next_*_batch`, numpy marshaling, GIL-release) DEFER to WP7;
//! tests + benches drive a MOCK producer over these plain-Rust APIs.

pub mod dense;
pub mod graph;
pub mod wire;

pub use dense::DenseQueue;
pub use graph::{build_leaf_graph, saturation_threshold, GraphQueue};
pub use wire::{GraphWire, GraphWireArrays, WireAlreadyConsumed};
