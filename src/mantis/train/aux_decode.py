"""Auxiliary target decoding: u8 buffer columns → float32 training tensors (WP10 §a.3 PORT).

Canonical home for ownership + winning_line decode so `train/trainer/core.py` stays focused
on forward / backward / optim / scheduler / save / load. Behaviour-exact.
"""
from __future__ import annotations

import numpy as np
import torch

from mantis.encoding import lookup as _lookup_encoding

BOARD_SIZE: int = _lookup_encoding("v6").board_size


def decode_ownership(arr: np.ndarray, device: torch.device, board_size: int = BOARD_SIZE) -> torch.Tensor:
    """Decode u8 ownership {0=P2, 1=empty, 2=P1} → float32 {-1, 0, +1} on `device`.

    Ships u8 to device first (4x smaller H2D transfer vs fp32) then converts + shifts in-place.
    `arr`: uint8 (B, board_size, board_size) or (B, board_size**2).
    """
    c = np.ascontiguousarray(arr)
    if c.ndim == 2:
        c = c.reshape(-1, board_size, board_size)
    return torch.from_numpy(c).to(device, non_blocking=True).float().sub_(1.0)


def decode_winning_line(arr: np.ndarray, device: torch.device, board_size: int = BOARD_SIZE) -> torch.Tensor:
    """Decode u8 winning_line → float32 {0.0, 1.0}, moved to `device`."""
    c = np.ascontiguousarray(arr)
    if c.ndim == 2:
        c = c.reshape(-1, board_size, board_size)
    return torch.from_numpy(c).to(device, non_blocking=True).float()


def mask_aux_rows(
    pred: torch.Tensor, target: torch.Tensor, n_pretrain: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice `[n_pretrain:]` to exclude corpus rows from aux loss.

    Corpus rows carry dummy aux (ownership=1→0.0, winning_line=0) that must not contribute to
    spatial head losses. When n_pretrain==0, returns the tensors unchanged (no copy).
    """
    if n_pretrain == 0:
        return pred, target
    return pred[n_pretrain:], target[n_pretrain:]
