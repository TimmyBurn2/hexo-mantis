"""⊕ WP12-R Phase EVALDECODE — C-9: the THIRD consumer refuses a graph spec by name.

D-18 of the R138 census. `SelfPlayWorker` (`selfplay/worker.py:108,165-166`) is a third
consumer of the graph producer with the IDENTICAL drop: `infer_batch` -> the dense
`expand_and_backup`. Its own module docstring says it is **NOT on the training data path**,
which is exactly why it is silently wrong the moment it is handed a graph spec — nothing
downstream notices. DESIGN §g.3 Option A (recommended, file-plan item 8) closes the class's
third arm with one raise, and discloses its cost: after the fix `_infer_batch_graph` has
zero production consumers.

Pre-registered HEAD verdict (PREREG §1.4 / §5): **RED** — at HEAD the constructor SUCCEEDS,
building a graph `LocalInferenceEngine` and starting its `InferenceServer`, so this oracle
fails with `DID NOT RAISE` and the engine it leaked is closed in the `finally` below.

CONDITIONAL: this file and its +1 collected test exist only if the operator takes §g.3
Option A. If D-18 is queued instead, C-9 and this file are dropped TOGETHER and the
pre-registered count becomes 2334 rather than 2335.

The raise's exception CLASS is deliberately not pinned: DESIGN fixes that the refusal
exists and is by name, not its wording, and pinning prose no design fixed would red a
correct fix (that is an R43 adjudication, not an oracle). What is pinned is that the
failure is DELIBERATE — the encoding is named in the message, and an incidental
`TypeError`/`AttributeError` does not satisfy it.
"""
from __future__ import annotations

import pytest
import torch

from mantis.encoding import lookup
from mantis.selfplay.worker import SelfPlayWorker


def test_selfplay_worker_refuses_a_graph_encoding_spec() -> None:
    built = None
    try:
        with pytest.raises(Exception) as excinfo:  # noqa: B017 — class deliberately unpinned
            built = SelfPlayWorker(
                torch.nn.Identity(),
                {"mcts": {"n_simulations": 2}},
                torch.device("cpu"),
                encoding_spec=lookup("gnn_axis_v1"),
            )
        assert not isinstance(excinfo.value, (TypeError, AttributeError)), (
            f"the refusal must be deliberate, not an incidental "
            f"{type(excinfo.value).__name__}: {excinfo.value}"
        )
        assert "gnn_axis_v1" in str(excinfo.value), str(excinfo.value)
    finally:
        if built is not None:
            built._engine.close()
