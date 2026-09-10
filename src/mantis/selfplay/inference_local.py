"""Local inference engine — the synchronous face over the batched seam.

>300 justify: one class, FOUR decode contracts that must be read together — the dense
`infer_batch` scatter-max/min-pool decode, the graph leg riding the ONE `InferenceServer`, the
RAW per-cluster grid decode (`infer_batch_per_cluster`), and the no-drop GRAPH decode
(`infer_batch_ls`). Splitting them separates each decode from the docstring stating what it
drops. Representation dispatch reads the BOUND SPEC, never the live model object.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

# Canonical stub-exported location — `torch.amp` itself does not re-export for type checkers.
from mantis._engine import Board
from mantis.config.resolve.fused_graph_caps import FusedGraphCapsSpec
from mantis.config.resolve.inference_batching import InferenceBatchingSpec
from mantis.encoding import EncodingSpec
from mantis.selfplay.hparams import is_graph_representation


class LocalInferenceEngine:
    """Wrap a grid or graph net and handle the full inference pipeline.

    Dense (grid): build (K, C, trunk, trunk) tensors, run one forward, map per-cluster policy
    outputs to one global vector per board, min-pool the values. Graph: reuse the production
    graph seam end to end rather than re-implementing the encoding.

    `encoding_spec`, `fused_graph_caps` and `inference_batching` are REQUIRED and keyword-only.
    This class hand-builds its `InferenceServer` config from a dict literal with no `RunConfig`,
    so a default here would be a value nobody minted on the one path with nothing to mint it
    from. Measured: at the single-stream deploy head (supply 8 against a collector threshold of
    32) the collector's own deadline is 1.76 of the eval path's 5.30 ms/sim, 33 %. GRID callers
    pass `None` EXPLICITLY. `max_in_flight` is the most graphs the caller can have in flight, and
    the collector's saturation threshold derives from it.

    Raises:
        ValueError: a graph engine was constructed with `inference_batching=None`.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        *,
        encoding_spec: EncodingSpec,
        fused_graph_caps: FusedGraphCapsSpec | None,
        inference_batching: InferenceBatchingSpec | None,
        max_in_flight: int,
        collate_check_period: int | None = None,
        collate_dump: tuple[str, Callable[[], dict[str, Any]]] | None = None,
        leaf_build_threads: int = 1,
    ) -> None:
        self.model = model
        self.device = device
        self.encoding_spec: EncodingSpec = encoding_spec
        # Representation comes from the BOUND SPEC; the frozen original's model-object sniff is
        # deleted repo-wide.
        self._is_graph = is_graph_representation(self.encoding_spec)
        # `1` is the SERIAL path and the exact-parity control: this layer must not invent a host
        # reservation, and a self-play worker is already one of `n_workers` threads.
        self._leaf_build_threads = max(1, int(leaf_build_threads))
        # DEFAULTED, unlike `leaf_batch_size` above: a wrong value there changes what the run
        # measures, while the worst a default does here is keep the status-quo check rate, which
        # `None` names and which cannot express "off".
        self._collate_check_period = collate_check_period
        self._collate_dump = collate_dump
        self._graph_batcher = None
        self._graph_server = None
        if self._is_graph:
            from mantis._engine import InferenceBatcher
            from mantis.selfplay.inference_server import InferenceServer

            if inference_batching is None:
                raise ValueError(
                    "LocalInferenceEngine: a GRAPH engine was constructed with "
                    "`inference_batching=None`. `None` is the GRID arm — it means 'this "
                    "route builds no graph server' — and there is no literal to fall back "
                    "to here (R1/LAW-11). Resolve it in the parent through "
                    "`mantis.config.resolve.inference_batching` and thread it in."
                )

            # THREADED, never guessed: the collector's saturation threshold derives from it.
            self._graph_batcher = InferenceBatcher(
                encoding_spec=self.encoding_spec, max_in_flight=max_in_flight)
            # `InferenceHParams.from_config` reads `config["inference"]` and this caller has no
            # `RunConfig`, so the rest are dataclass defaults handed explicitly.
            self._graph_server = InferenceServer(
                model, device,
                {"inference": {
                    # THREADED, never hardcoded — the batching geometry, same rule as the caps.
                    "inference_batch_size": inference_batching.inference_batch_size,
                    "inference_max_wait_ms": inference_batching.inference_max_wait_ms,
                }},
                batcher=self._graph_batcher, encoding_spec=self.encoding_spec,
                # THREADED, never hardcoded: a cap written here would be a SECOND authority over
                # one byte budget, on the one construction path with no config to be the first.
                fused_graph_caps=fused_graph_caps,
                # Threaded for the caps' reason: the rate and dump target are properties of the
                # PATH this engine serves, which the server cannot know.
                collate_check_period=self._collate_check_period,
                collate_dump=self._collate_dump,
            )
            self._graph_server.start()

    def close(self) -> None:
        """Stop the graph `InferenceServer` thread; a no-op for a dense engine. Idempotent, and
        also invoked best-effort from `__del__`."""
        if self._graph_server is not None:
            self._graph_server.stop()
            self._graph_server.join(timeout=5.0)
            self._graph_server = None
            self._graph_batcher = None

    def __del__(self) -> None:
        # The ONE sanctioned swallow in this package (census-allowlisted): a raising `__del__`
        # is unraisable at GC time, so it can only be printed, and it can fire during interpreter
        # shutdown when the globals `close()` needs are already torn down.
        try:
            self.close()
        except Exception:  # noqa: BLE001 — best-effort GC-time cleanup, never raise
            pass

    @torch.inference_mode()
    def infer(self, board: Board) -> tuple[list[float], float]:
        """Single-board convenience wrapper around `infer_batch`."""
        policies, values = self.infer_batch([board])
        return policies[0], values[0]

    @torch.inference_mode()
    def infer_batch(self, boards: list[Board]) -> tuple[list[list[float]], list[float]]:
        """Run inference on a list of boards.

        Returns:
            policies: global policy vectors (length `spec.policy_logit_count` each).
            values:   min-pooled scalar values, or the dist65-decoded value on the graph leg.
        """
        if not boards:
            return [], []

        return self._infer_batch_graph(boards)

    def _infer_batch_graph(
        self, boards: list[Board]
    ) -> tuple[list[list[float]], list[float]]:
        """Graph-representation leg of `infer_batch`, reusing the production graph seam. The
        dense half of each legal-set policy is returned and the coord-keyed overflow is DROPPED,
        which is the dense single-window branch's own contract.

        NO PRODUCTION CONSUMER REACHES THIS METHOD: the eval deploy head refuses a graph encoding
        and the eval worker's graph arm goes through `infer_batch_ls`. Retained because the
        census pins it and its drop contract is what `infer_batch_ls` is defined against.
        """
        positions = [
            (list(board.get_stones()), int(board.current_player), int(board.moves_remaining))
            for board in boards
        ]
        batcher = self._graph_batcher
        if batcher is None:
            # Set on every graph __init__; None only for a dense engine or after close().
            raise RuntimeError(
                "LocalInferenceEngine._infer_batch_graph: graph batcher is gone — the "
                "engine was closed (or constructed dense) before this inference call."
            )
        results = batcher.submit_graphs_and_wait(positions)
        policies = [dense for dense, _overflow, _value in results]
        values = [float(value) for _dense, _overflow, value in results]
        return policies, values

    @torch.inference_mode()
    def infer_ls(self, board: Board) -> tuple[
        list[float], list[tuple[tuple[int, int], float]], float, tuple[int, int]
    ]:
        """Single-board door onto `infer_batch_ls`: ONE delegation, so the refusal cannot drift."""
        dense, overflow, values, centers = self.infer_batch_ls([board])
        return dense[0], overflow[0], values[0], centers[0]

    @torch.inference_mode()
    def infer_batch_ls(self, boards: list[Board]) -> tuple[
        list[list[float]],
        list[list[tuple[tuple[int, int], float]]],
        list[float],
        list[tuple[int, int]],
    ]:
        """The NO-DROP graph decode: BOTH halves of what the shared producer returns.

        `_infer_batch_graph` keeps only the dense half and throws the `overflow` away — measured
        at 53.2% of legal moves at run5's geometry. This keeps both and returns the BUILDER's
        window centre, because the consumer must read priors in the frame their slots were baked
        in and `Board` does not expose that centre to Python.

        Returns:
            dense:    the in-window half per board (length `spec.policy_logit_count`).
            overflow: the off-window half per board, `((q, r), prob)` entries. The wire ORDER
                      is an artifact of map iteration; the Rust consumer rebuilds a map.
            values:   scalar value per board.
            centers:  the builder's `(cq, cr)` window centre per board.

        """
        if not boards:
            return [], [], [], []

        positions = [
            (list(board.get_stones()), int(board.current_player), int(board.moves_remaining))
            for board in boards
        ]
        batcher = self._graph_batcher
        if batcher is None:
            # Set on every graph __init__; None only after close().
            raise RuntimeError(
                "LocalInferenceEngine.infer_batch_ls: graph batcher is gone — the engine "
                "was closed before this inference call."
            )
        results = batcher.submit_graphs_and_wait_ls(positions, self._leaf_build_threads)
        dense = [d for d, _overflow, _value, _center in results]
        overflow = [list(o) for _dense, o, _value, _center in results]
        values = [float(v) for _dense, _overflow, v, _center in results]
        centers = [(int(c[0]), int(c[1])) for _dense, _overflow, _value, c in results]
        return dense, overflow, values, centers
