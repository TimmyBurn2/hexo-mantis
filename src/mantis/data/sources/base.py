"""CorpusSource ABC and GameRecord dataclass — the pipeline's unit of consumption."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GameRecord:
    """A complete game record ready for the corpus pipeline.

    Attributes:
        game_id_str: Opaque source-specific identifier.
                     Human: UUID from JSON filename.
                     Hybrid: "{seed_uuid}:c{continuation_idx}".
        moves:       Ordered (q, r) axial coordinate sequence for the full game.
        winner:      +1 if player 1 won, -1 if player 2 won.
        source:      Source label: "human" | "hybrid" | "api" | …
        metadata:    Optional extra fields; the pipeline ignores this dict but
                     preserves it for diagnostics and metrics.
    """

    game_id_str: str
    moves: list[tuple[int, int]]
    winner: int
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)  # pyright: ignore[reportUnknownVariableType]


class CorpusSource(ABC):
    """Abstract base for all corpus game sources.

    A source yields :class:`GameRecord` objects one at a time via ``__iter__``.
    It is responsible only for producing complete, valid game records.
    """

    @abstractmethod
    def name(self) -> str:
        """Short label used in log fields and metrics tags."""

    @abstractmethod
    def __iter__(self) -> Iterator[GameRecord]:
        """Yield one :class:`GameRecord` per completed game."""

    # No base `__len__`: the old `__len__ -> int | None` returning None made `len(source)`
    # a runtime TypeError anyway (PLE0303 — len() rejects non-int), just a misleading one.
    # A source that knows its length declares its own int-returning `__len__`
    # (`HumanSource` does); on the rest, `len()` fails loud and honestly (WPCLEAN Phase LT).
