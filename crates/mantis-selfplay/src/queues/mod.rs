//! Inference queues, pure Rust (no pyo3/numpy): the graph batcher [`graph::GraphQueue`] (consumer
//! `submit_graph_and_wait`, producer `pop_graph_batch` + `submit_graph_results`) and
//! [`wire::GraphWire`], the block-diagonal fuse tensor with a single-read `take()`.

pub mod collate;
pub mod eval_cache;
pub mod graph;
pub mod wire;

pub use eval_cache::{
    CachedEval, EvalCache, LeafKey, EVAL_CACHE_BYTES, EVAL_CACHE_CAPACITY, EVAL_CACHE_SHARDS,
};
pub use graph::{
    build_leaf_graph, build_leaf_graphs_batch, check_leaf_request, saturation_threshold,
    GraphQueue, LeafRequest,
};
pub use wire::{GraphWire, GraphWireArrays, WireAlreadyConsumed};
