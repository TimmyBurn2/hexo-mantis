//! Inference queues — the pure-Rust (pyo3/numpy-STRIPPED) half of the frozen
//! `inference_bridge.rs` (WP6 D4/D5/D6). The parallel graph `AxisGraph` batcher
//! ([`graph::GraphQueue`]); [`wire::GraphWire`] is the block-diagonal fuse tensor with a
//! single-read `take()`. The dense `Vec<f32>` batcher went with the grid path (R346(f)).
//!
//! PURE-RUST producer/consumer APIs only (LOCKED DECISION 4): the consumer face is
//! `submit_graph_and_wait`; the producer face is `pop_graph_batch` +
//! `submit_graph_results`.

pub mod graph;
pub mod wire;

pub use graph::{
    build_leaf_graph, build_leaf_graphs_batch, saturation_threshold, GraphQueue, LeafRequest,
};
pub use wire::{GraphWire, GraphWireArrays, WireAlreadyConsumed};
