"""The ladder's opening unit (R363(c)): `book_v1_s20260625_p4` PAIRED, opening index = match index, a convention between OUR two bots — the server's challenge carries no opening field, so both bots play the same book prefix from the server's auto-placed origin and search only past it."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from mantis.arena.books import book_openings

#: The follower's unit; the pairing is match.py's law there and the challenger's alternating `firstPlayer` here.
BOOK_ID = "book_v1_s20260625_p4"

Cell = tuple[int, int]


@dataclass(frozen=True)
class LadderOpening:
    """Opening `index` of `book` in FILE order, its stones RELATIVE to the first one (the server places that one)."""

    book: str
    index: int
    opening_id: str
    relative: tuple[Cell, ...]

    def on_origin(self, origin: Cell) -> tuple[Cell, ...]:
        """The opening translated onto the server's origin stone (the geometry is translation-invariant)."""
        return tuple((q + origin[0], r + origin[1]) for q, r in self.relative)

    def to_record(self) -> dict[str, Any]:
        return {"book": self.book, "index": self.index, "opening_id": self.opening_id,
                "relative": [[int(q), int(r)] for q, r in self.relative]}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> LadderOpening:
        return cls(book=str(record["book"]), index=int(record["index"]), opening_id=str(record["opening_id"]),
                   relative=tuple((int(q), int(r)) for q, r in record["relative"]))


def ladder_opening(match_index: int) -> LadderOpening:
    """Opening `match_index` of BOOK_ID in file order — no seed, no permutation. Raises: IndexError past the book (a series longer than the book is not this unit); BookError when the book does not verify."""
    openings = book_openings(BOOK_ID)
    if not 0 <= match_index < len(openings):
        raise IndexError(f"match index {match_index} is outside {BOOK_ID} ({len(openings)} openings, 0-based); "
                         "the unit does not wrap")
    opening = openings[match_index]
    q0, r0 = opening.moves[0]
    return LadderOpening(book=BOOK_ID, index=match_index, opening_id=opening.opening_id,
                         relative=tuple((q - q0, r - r0) for q, r in opening.moves))


def forced_stones(opening: LadderOpening, cells: Sequence[Cell]) -> tuple[Cell, ...] | None:
    """The book stones the next compound turn plays on a board whose stones in placement order are `cells`: up to two while the book runs, `()` once it is exhausted, `None` when the board has left the book (off-book — search)."""
    if not cells:
        return None
    full = opening.on_origin(cells[0])
    n = len(cells)
    if tuple(cells[: len(full)]) != full[:n]:
        return None
    return full[n:n + 2]


__all__ = ["BOOK_ID", "Cell", "LadderOpening", "forced_stones", "ladder_opening"]
