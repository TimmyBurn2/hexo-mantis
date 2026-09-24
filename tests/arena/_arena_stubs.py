"""The arena suites' shared opening stub: a frozen (id, moves) pair."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Opening:
    opening_id: str
    moves: list
