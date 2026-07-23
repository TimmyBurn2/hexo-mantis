"""Python-side ring buffer for recent self-play positions (WP10 §a.4 PORT;
old training/recency_buffer.py).

Maintains the last ``capacity`` positions in pre-allocated NumPy arrays, used
alongside the Rust replay buffer for recency-weighted batch sampling (biasing
training toward newer self-play data). ``push()`` / ``sample()`` are lock-guarded
so the pool stats thread and the training loop can call them concurrently.
Behaviour-exact; the only change is the encoding import (`hexo_rl` → `mantis`).
"""
from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from mantis.encoding import lookup as _lookup_encoding

_V6 = _lookup_encoding("v6")
BOARD_SIZE: int = _V6.board_size
BUFFER_CHANNELS: int = _V6.n_planes
NUM_CELLS: int = _V6.n_cells


class RecentBuffer:
    """Rolling-window ring buffer over recent self-play positions with aux columns.

    Args:
        capacity:    max positions; oldest overwritten once full (ring semantics).
        state_shape: shape of one state tensor, default (8, 19, 19) — HEXB v6.
        policy_len:  policy logits per position (default 362).
        aux_stride:  flat length of one ownership/winning_line plane (default 361).
    """

    def __init__(
        self,
        capacity: int,
        state_shape: tuple[int, ...] = (BUFFER_CHANNELS, BOARD_SIZE, BOARD_SIZE),
        policy_len: int = NUM_CELLS + 1,
        aux_stride: int = NUM_CELLS,
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self.capacity = capacity
        self._states = np.zeros((capacity, *state_shape), dtype=np.float16)
        self._chain_planes = np.zeros((capacity, 6, *state_shape[1:]), dtype=np.float16)
        self._policies = np.zeros((capacity, policy_len), dtype=np.float32)
        self._outcomes = np.zeros(capacity, dtype=np.float32)
        # ownership default 1 ("empty"), winning_line default 0 — neutral fallback.
        self._ownership = np.ones((capacity, aux_stride), dtype=np.uint8)
        self._winning_line = np.zeros((capacity, aux_stride), dtype=np.uint8)
        # default is_full_search=1 / value_target_valid=1 so an unpopulated slot is
        # never mistakenly masked out of policy / value loss if ever sampled.
        self._is_full_search = np.ones(capacity, dtype=np.uint8)
        self._value_target_valid = np.ones(capacity, dtype=np.uint8)
        self._head = 0
        self._size = 0
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        with self._lock:
            return self._size

    def push(
        self,
        state: np.ndarray,
        chain_planes: Optional[np.ndarray] = None,
        policy: Optional[np.ndarray] = None,
        outcome: float = 0.0,
        ownership: Optional[np.ndarray] = None,
        winning_line: Optional[np.ndarray] = None,
        is_full_search: bool = True,
        value_target_valid: bool = True,
    ) -> None:
        """Add one position; overwrites the oldest when full."""
        with self._lock:
            self._states[self._head] = state
            self._chain_planes[self._head] = chain_planes if chain_planes is not None else 0
            self._policies[self._head] = policy if policy is not None else 0
            self._outcomes[self._head] = float(outcome)
            self._ownership[self._head] = ownership if ownership is not None else 1
            self._winning_line[self._head] = winning_line if winning_line is not None else 0
            self._is_full_search[self._head] = 1 if is_full_search else 0
            self._value_target_valid[self._head] = 1 if value_target_valid else 0
            self._head = (self._head + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

    def sample(
        self, n: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Uniform random sample of ``n`` positions (8-tuple: states, chain_planes,
        policies, outcomes, ownership, winning_line, is_full_search,
        value_target_valid). Fancy-indexed returns are freshly allocated (not aliased
        to the ring storage), so callers may mutate them freely."""
        with self._lock:
            if self._size == 0:
                raise ValueError("Cannot sample from empty RecentBuffer")
            indices = np.random.randint(0, self._size, n)
            return (
                self._states[indices],
                self._chain_planes[indices],
                self._policies[indices],
                self._outcomes[indices],
                self._ownership[indices],
                self._winning_line[indices],
                self._is_full_search[indices],
                self._value_target_valid[indices],
            )

    def save_to_path(self, path: str) -> int:
        """Save valid entries to a compressed .npz. Returns the count saved."""
        with self._lock:
            size = self._size
            if size == 0:
                return 0
            if size < self.capacity:
                idx = np.arange(size)
            else:
                idx = np.arange(self._head, self._head + size) % self.capacity
            np.savez_compressed(
                path,
                states=self._states[idx],
                chain_planes=self._chain_planes[idx],
                policies=self._policies[idx],
                outcomes=self._outcomes[idx],
                ownership=self._ownership[idx],
                winning_line=self._winning_line[idx],
                is_full_search=self._is_full_search[idx],
                value_target_valid=self._value_target_valid[idx],
            )
            return size

    def load_from_path(self, path: str) -> int:
        """Load entries from a .npz saved by `save_to_path`. Returns the count loaded."""
        data = np.load(path)
        with self._lock:
            n = min(len(data["states"]), self.capacity)
            self._states[:n] = data["states"][:n]
            self._chain_planes[:n] = data["chain_planes"][:n]
            self._policies[:n] = data["policies"][:n]
            self._outcomes[:n] = data["outcomes"][:n]
            self._ownership[:n] = data["ownership"][:n]
            self._winning_line[:n] = data["winning_line"][:n]
            self._is_full_search[:n] = data["is_full_search"][:n]
            if "value_target_valid" in data:
                self._value_target_valid[:n] = data["value_target_valid"][:n]
            else:
                self._value_target_valid[:n] = 1
            self._head = n % self.capacity
            self._size = n
        return n
