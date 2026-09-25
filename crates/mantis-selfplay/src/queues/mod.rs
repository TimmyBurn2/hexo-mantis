//! Inference queues, pure Rust (no pyo3/numpy): the graph batcher [`graph::GraphQueue`] (consumer
//! `submit_graph_and_wait`, producer `pop_graph_batch` + `submit_graph_results`) and
//! [`wire::GraphWire`], the block-diagonal fuse tensor with a single-read `take()`.

pub mod eval_cache;
pub mod graph;
pub mod wire;

pub use graph::{
    build_leaf_graph, build_leaf_graphs_batch, saturation_threshold, GraphQueue, LeafRequest,
};
pub use eval_cache::{CachedEval, EvalCache, PositionKey, EVAL_CACHE_CAPACITY};
pub use wire::{GraphWire, GraphWireArrays, WireAlreadyConsumed};
