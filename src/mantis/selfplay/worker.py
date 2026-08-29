"""Single-process self-play helper — the MCTS + policy-sampling glue for bot play.

The full self-play training loop lives in Rust (the engine's `SelfPlayRunner`, driven
from `mantis.selfplay.pool`); this module is **NOT on the training data path**. Its
consumers are the bot/eval side: an MCTS search over a board plus the sampling rule that
turns the resulting visit distribution into a move.

Retained behaviour:
  - Dirichlet noise at the root (self-play only, disabled for evaluation), applied once
    per full compound turn — never at an intermediate ply of a 2-stone turn.
  - Temperature resolved per-move by `get_temperature` (`utils.py`): compound-turn
    quarter-cosine for "training", tau=0 for "evaluation" — the same schedule shape the
    Rust training path uses.

The encoding spec is the authority for board geometry and action-space size; nothing here
reads geometry off the model.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from mantis._engine import Board, MCTSTree
from mantis.encoding import EncodingSpec as RegistrySpec
from mantis.encoding import lookup, resolve_from_config
from mantis.selfplay.hparams import is_graph_representation
from mantis.selfplay.inference_local import LocalInferenceEngine
from mantis.selfplay.utils import get_temperature

# Back-compat for callers that import `get_temperature` from this module.
__all__ = ["SelfPlayWorker", "get_temperature"]


def _to_registry_spec(spec: RegistrySpec | Any) -> RegistrySpec:
    """Adapter — return the registry-form `EncodingSpec`.

    Accepts the registry dataclass directly, or any spec-like object exposing `.name`
    (which is re-looked-up); anything else is a `TypeError`.
    """
    if isinstance(spec, RegistrySpec):
        return spec
    if hasattr(spec, "name"):
        return lookup(spec.name)
    raise TypeError(
        f"_to_registry_spec: cannot adapt {type(spec).__name__!r}; "
        "expected mantis.encoding.EncodingSpec"
    )


class SelfPlayWorker:
    """MCTS + inference wrapper used by the model-backed bot.

    Args:
        model:  trained (or random) net.
        config: config dict. Used keys: `mcts.n_simulations`, `mcts.c_puct`,
                `mcts.dirichlet_alpha`, `mcts.epsilon`. Within-game temperature is
                resolved per-move by `get_temperature` (which honours the legacy eval
                alias `temperature_threshold_ply`).
        device: torch device.
        encoding_spec: optional explicit registry `EncodingSpec` (or any spec-like object
                with `.name`); resolved from config when omitted.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: dict[str, Any],
        device: torch.device,
        encoding_spec: RegistrySpec | Any | None = None,
    ) -> None:
        self.config = config
        self.device = device

        if encoding_spec is None:
            self.encoding_spec: RegistrySpec = resolve_from_config(config)
        else:
            self.encoding_spec = _to_registry_spec(encoding_spec)
        if is_graph_representation(self.encoding_spec):
            # D-18 of R138's census. This worker is the THIRD consumer of the graph
            # producer and it drops the same half the eval seam did: `_infer_batch` ->
            # `infer_batch` (dense half only) -> the dense `expand_and_backup`. Because
            # it is NOT on the training data path (module docstring), nothing downstream
            # would ever notice — which is exactly why it must not be a silent arm.
            # Refuse by name rather than wire a brand-new seam into a non-data-path
            # consumer for zero mint value (DESIGN §g.3 Option A).
            raise NotImplementedError(
                f"SelfPlayWorker does not implement the graph decode: encoding "
                f"{self.encoding_spec.name!r} declares "
                f"representation={self.encoding_spec.representation!r}. This worker "
                f"expands through LocalInferenceEngine.infer_batch, whose graph leg keeps "
                f"only the dense half of the producer's legal-set policy. The no-drop "
                f"graph path is DeployHeadPlayer(expand_fn=...) over infer_batch_ls "
                f"(mantis.eval.worker), or the Rust self-play runner."
            )
        self._board_size: int = self.encoding_spec.board_size
        self._n_actions: int = self.encoding_spec.policy_logit_count

        mcts_cfg = config.get("mcts", config)
        self.n_sims = int(mcts_cfg.get("n_simulations", config.get("n_simulations", 50)))
        self.c_puct = float(mcts_cfg.get("c_puct", 1.5))
        # Within-game temperature is resolved per-move by `get_temperature` from
        # `self.config`; there is no cached threshold field.
        self.dirichlet_alpha = float(mcts_cfg.get("dirichlet_alpha", 0.3))
        # ⚠ FIELD NAME != CONFIG KEY. The exploration-noise weight is read from
        # `mcts.epsilon`, NOT from `mcts.dirichlet_eps`/`mcts.dirichlet_epsilon`. Reading
        # the field's own spelling returns None on every config and silently substitutes
        # the 0.25 default for the operator's value — the same silently-disabled-knob
        # class that bites the temperature threshold. Traced to the frozen read site.
        self.dirichlet_eps = float(mcts_cfg.get("epsilon", 0.25))

        # Pass the resolved spec into the engine so the Python-side path sizes its
        # global-policy vector from `spec.policy_logit_count`.
        self._engine = LocalInferenceEngine(
            model, device, encoding_spec=self.encoding_spec,
            # EXPLICIT `None`, and it is UNCONDITIONALLY correct here rather than a default
            # this site happens to get away with: the graph representation is REFUSED by name
            # above (`NotImplementedError`, D-18 of R138's census), so this constructor cannot
            # be reached with a graph spec and the engine it builds is always the dense one,
            # which has no fused graph forward to bound. Written out so a reader sees the
            # decision instead of a silence (F-816-10 D-1).
            fused_graph_caps=None,
            # EXPLICIT `None` / `0` on the same grounds as the line above: the graph
            # representation is refused by name at this constructor, so this engine opens no
            # graph collector and there is no batching geometry or supply for it to carry.
            inference_batching=None,
            max_in_flight=0,
        )
        self.tree = MCTSTree(c_puct=self.c_puct)

        # Keep a direct reference for callers that access `worker.model`.
        self.model = model

    # ── Inference (thin delegation) ─────────────────────────────────────────────
    @torch.no_grad()
    def _infer(self, board: Board) -> tuple[list[float], float]:
        return self._engine.infer(board)

    @torch.no_grad()
    def _infer_batch(self, boards: list[Board]) -> tuple[list[list[float]], list[float]]:
        return self._engine.infer_batch(boards)

    # ── MCTS search ─────────────────────────────────────────────────────────────
    def _run_mcts(
        self,
        board: Board,
        use_dirichlet: bool = True,
        temperature: float | None = None,
    ) -> np.ndarray:
        return self._run_mcts_with_sims(
            board, n_sims=self.n_sims,
            use_dirichlet=use_dirichlet, temperature=temperature,
        )

    def _run_mcts_with_sims(
        self,
        board: Board,
        n_sims: int,
        use_dirichlet: bool = True,
        temperature: float | None = None,
        batch_size: int = 8,
    ) -> np.ndarray:
        """Run `n_sims` MCTS simulations from `board` using batched inference.

        Returns `MCTSTree.get_policy`'s dense policy vector as the ndarray it already is
        (consumers index it numerically; no list copy is taken)."""
        self.tree.new_game(board)
        # Dirichlet noise only at the start of a full compound turn, not at an
        # intermediate ply (the second stone of a 2-stone turn). Ply 0 is the opening
        # single stone — that IS a full turn, so noise applies there.
        is_intermediate_ply = board.moves_remaining == 1 and board.ply > 0
        dirichlet_applied = is_intermediate_ply  # skip noise if mid-turn
        effective_batch = max(1, int(batch_size))

        sims_done = 0
        while sims_done < n_sims:
            current_batch = min(effective_batch, n_sims - sims_done)
            try:
                leaves = self.tree.select_leaves(current_batch)
            except BaseException as exc:
                # Native MCTS can occasionally panic during batched leaf reconstruction.
                # Recover by restarting at root in batch_size=1 mode.
                if current_batch > 1 and "cell already occupied" in str(exc):
                    self.tree.new_game(board)
                    dirichlet_applied = False
                    effective_batch = 1
                    leaves = self.tree.select_leaves(1)
                else:
                    raise

            if not leaves:
                break

            policies, values = self._engine.infer_batch(leaves)
            self.tree.expand_and_backup(policies, values)
            sims_done += current_batch

            # Apply Dirichlet noise to the root priors after the first expansion.
            if use_dirichlet and not dirichlet_applied:
                n_ch = self.tree.root_n_children()
                if n_ch > 0:
                    noise = np.random.dirichlet(
                        [self.dirichlet_alpha] * n_ch
                    ).tolist()
                    self.tree.apply_dirichlet_to_root(noise, self.dirichlet_eps)
                    dirichlet_applied = True

        if temperature is None:
            temperature = get_temperature(
                ply=int(board.ply),
                mode="evaluation" if not use_dirichlet else "training",
                config=self.config,
            )

        return self.tree.get_policy(
            temperature=temperature, board_size=self._board_size
        )

    # ── Action sampling ─────────────────────────────────────────────────────────
    def _sample_action(
        self,
        policy: list[float],
        legal_moves: list[tuple[int, int]],
        board: Board,
    ) -> tuple[int, int]:
        """Sample a move from the MCTS policy, restricted to legal moves.

        Falls back to uniform sampling over the legal moves if MCTS assigns zero
        probability to all of them (degenerate case). The action-space size comes from
        the registry-resolved spec, never from a module-level constant.
        """
        n_actions = self._n_actions
        legal_flat = [board.to_flat(q, r) for q, r in legal_moves]
        probs = np.array(
            [policy[i] if i < n_actions else 0.0 for i in legal_flat],
            dtype=np.float64,
        )
        total = probs.sum()
        if total < 1e-9:
            probs = np.ones(len(legal_moves)) / len(legal_moves)
        else:
            probs /= total
        idx = np.random.choice(len(legal_moves), p=probs)
        return legal_moves[idx]
