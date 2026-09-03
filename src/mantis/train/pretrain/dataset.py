"""Pretrain dataset + augmentation collate (WP10 §a.7 IMPROVE — the v8 dataset is KILLED).

Ports the old `bootstrap/pretrain_dataset.py` behaviour-exact for every REACHABLE encoding, with
the ratified WP10 amendment: **the v8 dispatch key + the v8 augment branch are SEVERED**
(v8 never crosses — the registry has no v8; §a.7/§e). What remains is the v6-family (grid) path:

  - ``AugmentedBootstrapDataset`` — torch ``Dataset`` over flat (state, policy, outcome) triples.
  - ``make_augmented_collate`` — a collate_fn that batches the triples and applies hex
    augmentation, drawn PER ROW over the group that is lossless for that row (R245(c) — this is
    a window-clamped DENSE site; see ``mantis.data.augment.spread_mask``), via the Rust
    ``mantis._engine.apply_symmetries_batch`` binding (19×19 fast path) or a pure-numpy scatter
    (25×25 K-cluster, e.g. v6w25). Chain planes are recomputed post-augment from the augmented
    stone planes.
  - (``_game_winner_from_replay`` was DELETED at AUDIT-1 F-34 — see the note below.)

Imports the WP9-relocated ``mantis.data`` seams (train → data is DAG-legal): the numpy replayer
``mantis.data.replay.replay_game_to_triples`` + ``mantis.data.augment.get_policy_scatters`` /
``spread_mask`` / ``draw_record_syms``.
"""
from __future__ import annotations

import numpy as np
import torch

from mantis._engine import apply_symmetries_batch
from mantis.data.augment import draw_record_syms, get_policy_scatters, spread_mask
from mantis.data.replay import replay_game_to_triples
from mantis.encoding import lookup as _lookup_encoding
from mantis.encoding import opp_stone_slot
from mantis.env.game_state import N_CHAIN_PLANES, _compute_chain_planes

# 19×19 is the Rust `apply_symmetries_batch` fast-path board size (the binding hardcodes it); a
# larger K-cluster trunk (e.g. v6w25 at 25×25) falls to the pure-numpy scatter below.
_RUST_AUG_BOARD_SIZE = 19

__all__ = [
    "AugmentedBootstrapDataset",
    "make_augmented_collate",
    "replay_game_to_triples",
]

# AUDIT-1 F-34. `_game_winner_from_replay` IS DELETED. It built an identity-blind `Board()` —
# engine defaults, radius 5 — and then wrapped every `apply_move` in `except Exception: break`,
# so a game whose moves are illegal at radius 5 truncated silently and the function returned a
# winner read off the TRUNCATED position. At radius 6 that refusal is 34.76 % of the bootstrap
# corpus (R327), which is exactly the measurement this would have hidden. It was exported and
# had no caller anywhere in `src/` or `tests/`, so nothing is re-pointed.


class AugmentedBootstrapDataset(torch.utils.data.Dataset):
    """Pretrain dataset yielding raw (state, policy, outcome) triples.

    Hex augmentation (drawn per row over that row's lossless group, R245(c)) is applied in
    ``make_augmented_collate`` via the Rust ``apply_symmetries_batch`` binding (the same scatter
    kernel the ReplayBuffer uses at sample time). Chain planes (6, the Q13 aux head target) are
    recomputed in collate from the augmented stone planes — recomputing post-augment is
    self-consistent and avoids an axis-perm remap.

    Args:
        states:    (N, C, H, W) float16 array (mmap-compatible).
        policies:  (N, n_actions) float32 array.
        outcomes:  (N,) float32 array, values in {-1, 0, +1}.
    """

    def __init__(
        self,
        states: np.ndarray,
        policies: np.ndarray,
        outcomes: np.ndarray,
    ) -> None:
        self.states = states
        self.policies = policies
        self.outcomes = outcomes

    def __len__(self) -> int:
        return len(self.outcomes)

    def __getitem__(self, idx: int):
        # Copy out of the (possibly mmapped) backing store so collate can batch-concat safely.
        return (
            self.states[idx].copy(),
            self.policies[idx].copy(),
            float(self.outcomes[idx]),
        )


def make_augmented_collate(augment: bool, encoding: str):
    """Return a collate_fn that batches triples and applies hex augmentation.

    19×19 grid (v6 family): Rust ``apply_symmetries_batch`` state scatter (one PyO3 hop) + numpy
    policy scatter via precomputed index tables. Larger K-cluster trunk (e.g. v6w25 at 25×25):
    pure-numpy state + policy scatter (the Rust binding hardcodes 19×19). ``augment=False``: no
    scatter. Chain planes are recomputed from augmented stone planes 0 (cur) / opp-slot (opp)
    — the opp t0 slot is derived from the registry spec (4 for the v6 family, 1 for v6_live2)."""
    _enc_spec = _lookup_encoding(encoding)
    has_pass = _enc_spec.has_pass_slot
    board_size = _enc_spec.trunk_size
    _opp_slot = opp_stone_slot(_enc_spec)
    scatters_np = get_policy_scatters(board_size, has_pass=has_pass) if augment else None

    def _collate(batch):
        n = len(batch)
        states = np.stack([b[0] for b in batch], axis=0)
        policies = np.stack([b[1] for b in batch], axis=0)
        outcomes = np.asarray([b[2] for b in batch], dtype=np.float32)

        if augment and scatters_np is not None:
            # R245(c): window-clamped DENSE site — the draw is gated per row on whether that
            # row fits the window (mantis.data.augment.spread_mask). A row a dropping element
            # would clip gets only WINDOW_PRESERVING_SYMS; a window-fitting one gets all 12.
            # Only the two channels this closure SCATTERS are certified — the chain planes are
            # recomputed below from the augmented stone planes, so they are induced by states.
            sym_indices = draw_record_syms(
                spread_mask(board_size, states=states, policies=policies)
            ).astype(np.int64)

            if board_size == _RUST_AUG_BOARD_SIZE:
                states_f32 = states.astype(np.float32, copy=False)
                states_f32 = apply_symmetries_batch(
                    states_f32, sym_indices.astype(np.uint64).tolist()
                )
                scattered = np.empty_like(policies)
                for i in range(n):
                    scattered[i] = policies[i][scatters_np[int(sym_indices[i])]]
                policies = scattered
                states = states_f32.astype(np.float16, copy=False)
            else:
                # Larger K-cluster trunk — pure-numpy scatter, batched per-sym.
                C = states.shape[1]
                spatial = board_size * board_size
                states_flat = states.reshape(n, C, spatial)
                augmented = np.empty_like(states_flat)
                policy_aug = np.empty_like(policies)
                for sym in range(12):
                    mask_idx = np.where(sym_indices == sym)[0]
                    if mask_idx.size == 0:
                        continue
                    sc = scatters_np[sym]
                    state_scatter = sc[:spatial] if has_pass else sc
                    augmented[mask_idx] = states_flat[mask_idx][:, :, state_scatter]
                    policy_aug[mask_idx] = policies[mask_idx][:, sc]
                states = augmented.reshape(n, C, board_size, board_size)
                policies = policy_aug

        # Chain planes — recomputed post-augment from stone planes 0 (cur) / opp-slot (opp).
        chain_np = np.zeros((n, N_CHAIN_PLANES, board_size, board_size), dtype=np.float16)
        for i in range(n):
            chain_np[i] = _compute_chain_planes(
                states[i, 0].astype(np.float32),
                states[i, _opp_slot].astype(np.float32),
            ).astype(np.float16) / 6.0

        return (
            torch.from_numpy(states),
            torch.from_numpy(chain_np),
            torch.from_numpy(policies),
            torch.from_numpy(outcomes),
        )

    return _collate


