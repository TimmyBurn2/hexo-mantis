"""The graph coordinator-drive rig: one ring builder, one sampling buffer double and one
non-binding drive declaration for every tests/train harness that fakes a replay buffer."""
from __future__ import annotations

from typing import Any

from mantis._engine import HexgBuffer


def filled_hexg(n_records: int = 8, capacity: int = 64) -> HexgBuffer:
    """A real graph ring the coordinator stubs sample through, fed through the real push path."""
    hb = HexgBuffer(capacity, "gnn_axis_v1", 128)
    for i in range(n_records):
        stones = [(0, 0, 1), (1, 0, -1), (0, 1, 1)][: 2 + (i % 2)]
        hb.push_graph_position(stones, [(2, 0, 0.6), (1, 1, 0.4)], 1, 30, 2 + i, True,
                               1.0 if i % 2 == 0 else -1.0, True, 10 + i)
    return hb


class GraphSampleBuffer:
    """The replay-buffer double; sampling delegates to a real `HexgBuffer` so the dispatcher collates the real wire."""

    def __init__(self, size: int = 1000, capacity: int = 100_000) -> None:
        self.size = size
        self.capacity = capacity
        self._hexg = filled_hexg()

    def resize(self, n: int) -> None:
        self.capacity = n

    def save_to_path(self, p: Any) -> None: ...

    def sample_graph_batch(self, n: int, *, augment: bool = False, recent_frac: float = 0.0,
                           n_threads: int = 1) -> Any:
        return self._hexg.sample_graph_batch(n, augment=augment, recent_frac=recent_frac,
                                             n_threads=n_threads)


#: The declaration a `StepCoordinator` reads on the graph route: the dispatch identity plus
#: the sections its resolvers read. The caps are the NON-BINDING pair — nothing here
#: exercises a split.
GRAPH_FULL_CONFIG: dict = {
    "identity": {"encoding": "gnn_axis_v1", "representation": "graph"},
    "train": {"microbatch_caps": {"max_edges": 100_000_000, "max_nodes": 4_000_000},
              "fast_policy_weight": 0.0},
    "selfplay": {"n_workers": 1},
}
