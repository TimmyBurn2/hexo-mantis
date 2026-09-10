"""An opening book only replays under the geometry it was minted at.

`book_v1_s20260625_p4` is minted against `gnn_axis_v1` at `legal_move_radius = 6` and its
openings scatter up to six hex-steps apart. Under a radius-5 encoding 292 of its 512 openings
contain a move outside the board's legal set, and the arena used to play them anyway.

This is a PAIRING check, not a book check: the book is correct for its declared minting
geometry, and nothing anywhere related a config's `identity.encoding` to the `opening_book`
its eval blocks name.

The numbers are derived, never transcribed: the required radius is computed from the book's
own moves and the encodings' radii are read from the registry.
"""
from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest
import yaml

from mantis.encoding import lookup

_REPO = Path(__file__).resolve().parents[2]
_BOOKS_DIR = _REPO / "src" / "mantis" / "arena" / "books"
_CONFIGS = _REPO / "configs"

#: Configs whose `identity.encoding` cannot replay the book their eval blocks name. Not a
#: waiver but an inventory, asserted EXACTLY below, so closing a gap reds this suite and the
#: row must be removed rather than outliving its reason.
_KNOWN_UNPLAYABLE_PAIRINGS: set[str] = set()


def _hex_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    dq, dr = a[0] - b[0], a[1] - b[1]
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


def _required_radius(moves: list[tuple[int, int]]) -> int:
    """Return the smallest `legal_move_radius` under which `moves` replays.

    Each move after the first must fall inside the radius ball of some already-placed stone,
    so one move's requirement is its distance to the nearest earlier stone and the sequence's
    is the largest of those.
    """
    return max(
        (min(_hex_distance(moves[i], moves[j]) for j in range(i)) for i in range(1, len(moves))),
        default=0,
    )


def _book_openings(book_id: str) -> list[list[tuple[int, int]]]:
    with (_BOOKS_DIR / "manifest.toml").open("rb") as handle:
        entry = tomllib.load(handle)["books"][book_id]
    payload = json.loads((_BOOKS_DIR / entry["file"]).read_text(encoding="utf-8"))
    return [[tuple(m) for m in o["moves"]] for o in payload["openings"]]


def _books_named_by(config: dict) -> set[str]:
    """Return every `opening_book` value anywhere in a config, found structurally."""
    found: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "opening_book" and isinstance(value, str):
                    found.add(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(config)
    return found


def _config_files() -> list[Path]:
    files = sorted(_CONFIGS.glob("*.yaml"))
    assert files, "no configs found — a pairing check with nothing to check is a phantom gate"
    return files


def test_the_book_declares_its_geometry_by_construction() -> None:
    """Prove the book's required radius, derived from its own moves, is its minting encoding's."""
    # Read out of the minter's source rather than imported: `tools/` is not importable and
    # `sys.path` may not move, so the value is still derived from the tool, not transcribed.
    source = (_REPO / "tools" / "mint_opening_book.py").read_text(encoding="utf-8")
    match = re.search(r'^_MINT_ENCODING\s*=\s*"([^"]+)"', source, re.M)
    assert match is not None, (
        "`_MINT_ENCODING` is no longer a module-level string literal in "
        "tools/mint_opening_book.py — REFUSING to report clean on an unparseable minter"
    )
    mint_encoding = match.group(1)

    openings = _book_openings("book_v1_s20260625_p4")
    assert len(openings) == 512
    required = max(_required_radius(mv) for mv in openings)
    minted_radius = lookup(mint_encoding).legal_move_radius
    assert required <= minted_radius, (
        f"the book needs radius {required} but was minted against {mint_encoding} at "
        f"radius {minted_radius} — the minter and the artifact disagree"
    )


@pytest.mark.parametrize("radius,expected_playable", [(5, False), (6, True), (8, True)])
def test_the_radius_5_gap_is_real_and_the_radius_6_one_is_not(
    radius: int, expected_playable: bool
) -> None:
    """Measure the gap: radius 5 cannot replay the book, radius 6 and 8 can."""
    openings = _book_openings("book_v1_s20260625_p4")
    unplayable = [mv for mv in openings if _required_radius(mv) > radius]
    assert (not unplayable) == expected_playable, (
        f"at radius {radius}, {len(unplayable)} of {len(openings)} openings do not replay"
    )


def test_every_shipped_config_pairs_its_encoding_with_a_replayable_book() -> None:
    """Prove every shipped config's encoding radius covers every book its eval blocks name."""
    broken: dict[str, str] = {}
    for path in _config_files():
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        encoding = config["identity"]["encoding"]
        radius = lookup(encoding).legal_move_radius
        for book_id in _books_named_by(config):
            unplayable = [
                mv for mv in _book_openings(book_id) if _required_radius(mv) > radius
            ]
            if unplayable:
                broken[path.name] = (
                    f"{encoding} (radius {radius}) cannot replay {len(unplayable)} of "
                    f"{len(_book_openings(book_id))} openings in {book_id}"
                )
    assert set(broken) == _KNOWN_UNPLAYABLE_PAIRINGS, (
        f"the unplayable-pairing inventory moved. Found: {broken}; recorded: "
        f"{sorted(_KNOWN_UNPLAYABLE_PAIRINGS)}. A config that GAINED a broken pairing is a "
        f"defect; one that LOST it means the row above is stale and must be deleted."
    )


def test_run6_is_not_one_of_them() -> None:
    """Prove run6's encoding replays every book it names, stated separately so it is visible."""
    config = yaml.safe_load((_CONFIGS / "run6.yaml").read_text(encoding="utf-8"))
    radius = lookup(config["identity"]["encoding"]).legal_move_radius
    for book_id in _books_named_by(config):
        assert all(_required_radius(mv) <= radius for mv in _book_openings(book_id)), (
            f"run6 names {book_id}, which its encoding cannot replay"
        )
