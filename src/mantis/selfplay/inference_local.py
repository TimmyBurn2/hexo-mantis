"""Local inference engine — the synchronous face over the batched seam.

>300 justify: one class, three decode contracts that must be read together — the dense
`infer_batch` scatter-max/min-pool decode, the graph leg riding the ONE `InferenceServer`,
and the RAW per-cluster decode (`infer_batch_per_cluster`) — splitting them would separate
each decode from the docstring stating what it drops. WPSC Phase 2 SC-A2's explicit
8-field `InferenceHParams`-default dict literal (replacing the old `{"selfplay": {}}`
fallback) is what pushed this file from 292 to 303 lines; WPCLEAN Phase LT's
type-visibility guards (batcher None-guard, canonical autocast import) took it to 323.
MEASURED size now 330 lines (`wc -l`): WP12-R Phase C deleted the `lookup("v6")` ternary
(gate 11's arm 8) and re-pointed the class docstring, which had asserted a dense default
that no longer exists.

One class, three decode contracts that must be read together: the dense `infer_batch`
scatter-max/min-pool decode, the graph leg that rides the ONE server, and the RAW
per-cluster decode (`infer_batch_per_cluster` — deliberately NO scatter-max, NO
off-window drop, NO min-pool). Each decode lives next to the docstring stating what it
drops.

`LocalInferenceEngine` batches boards through the network and returns global policy
vectors + min-pooled scalar values. It is the Python-side path used by bot/eval callers
that do not go through the Rust self-play runner. It is multi-window aware via
`GameState.to_tensor()`, which returns a list of cluster centers; for single-window
encodings the centers loop runs with K=1 and degenerates to the trivial mapping.

Representation dispatch reads the BOUND SPEC (`spec.representation`), never the live
model object: inferring architecture by reading attributes off an `nn.Module` is banned
repo-wide. The graph leg reuses the production graph seam — it constructs and rides an
`InferenceServer` rather than re-implementing a second graph loop.
"""
from __future__ import annotations

import numpy as np
import torch

# Canonical stub-exported location — `torch.amp` itself does not re-export for type checkers.
from torch.amp.autocast_mode import autocast

from mantis._engine import Board
from mantis.encoding import EncodingSpec
from mantis.env.game_state import GameState
from mantis.selfplay.hparams import is_graph_representation


class LocalInferenceEngine:
    """Wraps a grid or graph net and handles the full inference pipeline.

    Dense (grid) representation:
      1. Build (K, C, trunk, trunk) tensors for a batch of boards.
      2. Run a single forward pass.
      3. Map per-cluster local policy outputs → one global policy vector per board.
      4. Aggregate per-cluster values via min-pooling.

    Graph representation: `infer_batch` reuses the production graph inference seam
    (`InferenceBatcher.submit_graphs_and_wait` → a background `InferenceServer` graph loop
    → `collate_graph_batch` → `GnnNet.forward_batch` → segment-softmax → the Rust legal-set
    assemble). Single-source reuse, not a reimplementation of the graph encoding.

    The caller passes `encoding_spec`; it is the AUTHORITY for the representation
    dispatch, and it is REQUIRED and keyword-only — there is no default to inherit
    (WP12-R Phase C closed gate 11's arm 8: the old `else lookup("v6")` ternary bound a
    dense spec for every caller who said nothing, so a graph caller silently
    misconfigured the graph batcher). LAW-11 says an absent encoding is an error; a
    required parameter makes absent UNCONSTRUCTIBLE, which pyright catches before a
    worker ever spawns. Handing a graph-built model a dense spec (or the inverse) is a
    wiring error and fails loudly rather than decoding garbage.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        *,
        encoding_spec: EncodingSpec,
    ) -> None:
        self.model = model
        self.device = device
        self.encoding_spec: EncodingSpec = encoding_spec
        # Representation comes from the BOUND SPEC. The frozen original sniffed the live
        # model object here and preferred it over the spec when the two disagreed; that
        # sniff is deleted repo-wide (arch travels on declared metadata, never on an
        # `nn.Module`). Every production graph caller passes its spec, so this is
        # value-identical on every reachable input — and a genuine model/spec
        # disagreement now fails loudly instead of silently decoding down the other arm.
        self._is_graph = is_graph_representation(self.encoding_spec)
        self._graph_batcher = None
        self._graph_server = None
        if self._is_graph:
            from mantis._engine import InferenceBatcher
            from mantis.selfplay.inference_server import InferenceServer

            self._graph_batcher = InferenceBatcher(encoding_spec=self.encoding_spec)
            # WPSC Phase 2 SC-A2: InferenceHParams.from_config reads config["inference"]
            # directly (no top-level-namespace fallback) — this standalone caller has no
            # RunConfig to draw from, so it hands the `InferenceHParams` dataclass defaults
            # explicitly (zero-behavior-change: these are the same values `{"selfplay": {}}`
            # used to resolve to via the old `.get(k, default)` fallback chain).
            self._graph_server = InferenceServer(
                model, device,
                {"inference": {
                    "inference_batch_size": 64, "inference_max_wait_ms": 10,
                    "trace_inference": True, "compile_inference": False,
                    "compile_inference_mode": "default", "compile_inference_dynamic": True,
                    "perf_timing": False, "perf_sync_cuda": False,
                # WPSC Phase 3 SC-B3: InferenceServer hard-reads config["train"]
                # ["amp_dtype"] unconditionally (R30b, no fallback) — inert here (this
                # branch is always graph, LAW-06 bf16-pinned regardless of the value).
                }, "train": {"amp_dtype": "bf16"}},
                batcher=self._graph_batcher, encoding_spec=self.encoding_spec,
            )
            self._graph_server.start()

    def close(self) -> None:
        """Stop the graph `InferenceServer` thread (no-op for a dense engine).

        Callers that construct a graph-representation engine should call this when done.
        Also invoked best-effort from `__del__`. Idempotent.
        """
        if self._graph_server is not None:
            self._graph_server.stop()
            self._graph_server.join(timeout=5.0)
            self._graph_server = None
            self._graph_batcher = None

    def __del__(self) -> None:
        # The ONE sanctioned swallow in this package (census-allowlisted). A raising
        # `__del__` is a Python-semantics hazard: the exception is unraisable at GC time,
        # so it cannot be handled, only printed — and it can fire during interpreter
        # shutdown when the module globals `close()` needs are already torn down. Every
        # other `except: pass` in mantis.selfplay is a defect.
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
            values:   scalar values, one per board (min-pooled over clusters, or the
                      dist65-decoded value on the graph leg).
        """
        if not boards:
            return [], []

        if self._is_graph:
            return self._infer_batch_graph(boards)

        spec = self.encoding_spec
        board_size = spec.board_size
        n_actions = spec.policy_logit_count
        half = (board_size - 1) // 2

        all_tensors = []
        board_info: list[tuple[int, list[tuple[int, int]]]] = []

        for board in boards:
            state = GameState.from_board(board)
            tensor, centers = state.to_tensor()
            if tensor.shape[1] != spec.n_planes:
                # Slice the full wire tensor to THIS encoding's kept planes. The plane
                # count comes from the bound spec, never from a module attribute: a
                # module-level constant here would be pinned to one encoding and would
                # feed the wrong plane count into any other.
                tensor = tensor[:, list(spec.kept_plane_indices)]
            all_tensors.append(torch.from_numpy(tensor))
            board_info.append((len(centers), centers))

        # Single batched forward pass over all clusters from all boards.
        batch_tensor = torch.cat(all_tensors, dim=0).to(self.device)

        self.model.eval()
        with autocast(
            device_type=self.device.type,
            enabled=(self.device.type in ("cuda", "mps")),
        ):
            log_policy, value, _v_logit = self.model(batch_tensor.float())

        policies_np = log_policy.exp().cpu().float().numpy()  # (TotalK, n_actions)
        values_np = value.squeeze(-1).cpu().float().numpy()  # (TotalK,)

        results_p: list[list[float]] = []
        results_v: list[float] = []

        cursor = 0
        for i, board in enumerate(boards):
            k, centers = board_info[i]
            board_policies = policies_np[cursor:cursor + k]
            board_values = values_np[cursor:cursor + k]
            cursor += k

            # Min-pool over clusters: treat the worst window as the board value.
            v = float(board_values.min())

            # Map each legal move to the highest probability across all windows.
            global_policy = np.zeros(n_actions, dtype=np.float64)
            for q, r in board.legal_moves():
                mcts_idx = board.to_flat(q, r)
                if mcts_idx >= n_actions - 1:
                    continue
                max_prob = 0.0
                for k_idx, (cq, cr) in enumerate(centers):
                    wq = q - cq + half
                    wr = r - cr + half
                    if 0 <= wq < board_size and 0 <= wr < board_size:
                        local_idx = wq * board_size + wr
                        if board_policies[k_idx, local_idx] > max_prob:
                            max_prob = board_policies[k_idx, local_idx]
                global_policy[mcts_idx] = max_prob

            total = global_policy.sum()
            if total > 1e-9:
                global_policy /= total
            else:
                global_policy.fill(1.0 / n_actions)

            results_p.append(global_policy.tolist())
            results_v.append(v)

        return results_p, results_v

    def _infer_batch_graph(
        self, boards: list[Board]
    ) -> tuple[list[list[float]], list[float]]:
        """Graph-representation leg of `infer_batch`.

        Reuses the production graph inference seam (`submit_graphs_and_wait` → the
        background `InferenceServer` graph loop → `collate_graph_batch` →
        `GnnNet.forward_batch` → segment-softmax → the Rust legal-set assemble): a native
        axis graph is built once per board from its live stones by the same seam the
        self-play leaf builder runs, never a hand-rolled Python graph encode.

        The dense half of each assembled legal-set policy is returned as the policy
        vector; the coord-keyed overflow (off-window legal moves the whole-board graph's
        single window does not cover) is DROPPED here — exactly the drop contract the
        dense single-window branch above already applies (`mcts_idx >= n_actions - 1:
        continue`). This is the existing `infer_batch` contract, not a new approximation;
        the no-drop decode is `infer_batch_per_cluster`, which has no graph analogue.
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
    def infer_batch_per_cluster(
        self, boards: list[Board]
    ) -> tuple[list[list[float]], list[float], list[int]]:
        """RAW per-cluster policy/value vectors for the Rust legal-set expand path.

        Unlike `infer_batch` (which scatter-max collapses K clusters into ONE dense global
        vector AND DROPS off-window moves where `mcts_idx >= n_actions-1`), this returns
        the per-cluster outputs RAW — NO scatter-max, NO drop, NO min-pool. The Rust
        legal-set expand does the aggregation and value min-pool so the deploy head pools
        BYTE-IDENTICALLY to the self-play worker, retaining off-window cells covered by
        some cluster.

        Center order: `GameState.from_board` reads the board's cluster views and the Rust
        expand RECOMPUTES centers from the same call on the pending board — so the
        per-cluster rows align by construction.

        Returns:
            policies: FLAT list of per-cluster prob vectors (length
                      `spec.policy_logit_count` each), leaf-major then cluster order.
            values:   FLAT list of per-cluster scalar values, same order.
            leaf_k:   K (cluster count) per board, aligned with `boards`.

        Raises:
            NotImplementedError: the model is graph-representation. The no-drop legal-set
                decode has no graph analogue (the graph net is whole-board, no K-cluster);
                die loud here instead of an `AttributeError` two lines down.
        """
        if not boards:
            return [], [], []
        if self._is_graph:
            raise NotImplementedError(
                "infer_batch_per_cluster: no legal-set/no-drop decode exists for a graph "
                "model — the graph net is whole-board (no K-cluster). Use infer_batch "
                "instead."
            )

        spec = self.encoding_spec

        all_tensors = []
        leaf_k: list[int] = []
        for board in boards:
            state = GameState.from_board(board)
            tensor, centers = state.to_tensor()
            if tensor.shape[1] != spec.n_planes:
                tensor = tensor[:, list(spec.kept_plane_indices)]
            all_tensors.append(torch.from_numpy(tensor))
            leaf_k.append(len(centers))

        batch_tensor = torch.cat(all_tensors, dim=0).to(self.device)

        self.model.eval()
        with autocast(
            device_type=self.device.type,
            enabled=(self.device.type in ("cuda", "mps")),
        ):
            log_policy, value, _v_logit = self.model(batch_tensor.float())

        policies_np = log_policy.exp().cpu().float().numpy()  # (TotalK, n_actions)
        values_np = value.squeeze(-1).cpu().float().numpy()  # (TotalK,)

        policies = [policies_np[i].tolist() for i in range(policies_np.shape[0])]
        values = [float(v) for v in values_np]
        return policies, values, leaf_k


__all__ = ["LocalInferenceEngine"]
