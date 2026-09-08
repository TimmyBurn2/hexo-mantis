"""R345(b)(2) finding — an opening book only replays under the geometry it was minted at.

WHAT THE LEGALITY BOUNDARY FOUND. `book_v1_s20260625_p4` is minted against `gnn_axis_v1`
(`tools/mint_opening_book.py::_MINT_ENCODING`, `legal_move_radius = 6`), and its openings are
uniform-random draws from that board's legal set — so they scatter up to six hex-steps apart.
Replayed under a radius-5 encoding, 292 of its 512 openings (57.03%) contain a move that is
not in the board's legal set, and until R345(b)(2) the arena played them anyway: every one of
those games started from a position the rules cannot reach, and the promotion bar read the
result.

WHY THIS IS A PAIRING TEST AND NOT A BOOK TEST. The book is not wrong — it is correct for its
declared minting geometry, and `tests/arena/test_books.py` already pins that it reproduces
from its minter args. What had no check at all was the PAIRING: nothing anywhere related a
config's `identity.encoding` to the `opening_book` its eval blocks name. That is the LAW-08
shape one level up — a registered artifact with a consumer that cannot use it.

THE NUMBERS BELOW ARE DERIVED, NEVER TRANSCRIBED (R192(e)). The required radius is computed
from the book's own moves and the encodings' radii are read from the registry, so a re-minted
book or a moved registry row changes what this suite asserts rather than making it stale.
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

#: Configs whose `identity.encoding` cannot replay the book their eval blocks name. NOT a
#: waiver — an inventory, asserted EXACTLY below, so closing the gap reds this suite and the
#: row must then be removed rather than quietly outliving its reason (the reverse-check shape
#: gate 13 uses on its "deliberately absent" section).
#:
#: `sustained_kcluster.yaml` is a legacy dense/k-cluster config on no run6 path. Re-minting it
#: — onto a radius >= 6 encoding, or onto a book minted at radius 5 the repo does not yet ship
#: — is a mint act and the operator's (R1: configs are minted, never hand-varied), so it is
#: recorded here rather than repaired in passing.
_KNOWN_UNPLAYABLE_PAIRINGS = {"sustained_kcluster.yaml"}


def _hex_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    dq, dr = a[0] - b[0], a[1] - b[1]
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


def _required_radius(moves: list[tuple[int, int]]) -> int:
    """The smallest `legal_move_radius` under which `moves` replays.

    Each move after the first must fall inside the radius ball of SOME already-placed stone
    (`Board::legal_moves_set` is that union), so the requirement of one move is its distance
    to the NEAREST earlier stone, and the requirement of the sequence is the largest of those.
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
    """Every `opening_book` value anywhere in a config, found structurally."""
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


# ── the measurement ─────────────────────────────────────────────────────────────────────
def test_the_book_declares_its_geometry_by_construction() -> None:
    """The book's required radius, derived from its own moves, is its minting encoding's."""
    # Read out of the minter's SOURCE, not imported: `tools/` is not an importable package
    # and R5 bars the `sys.path` write that would make it one, so the established shape here
    # (tests/arena/test_books.py) is to reach the tool by path. The value is still derived
    # from the tool rather than transcribed into this file.
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
    """The finding itself, as a measurement rather than a sentence."""
    openings = _book_openings("book_v1_s20260625_p4")
    unplayable = [mv for mv in openings if _required_radius(mv) > radius]
    assert (not unplayable) == expected_playable, (
        f"at radius {radius}, {len(unplayable)} of {len(openings)} openings do not replay"
    )


# ── the standing pairing check ──────────────────────────────────────────────────────────
def test_every_shipped_config_pairs_its_encoding_with_a_replayable_book() -> None:
    """`identity.encoding`'s radius must cover every book the config's eval blocks name."""
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
    """Stated separately because it is the fact the packet's hold turns on.

    Folded into the inventory above it would be one absent dict key — true, and invisible.
    """
    config = yaml.safe_load((_CONFIGS / "run6.yaml").read_text(encoding="utf-8"))
    radius = lookup(config["identity"]["encoding"]).legal_move_radius
    for book_id in _books_named_by(config):
        assert all(_required_radius(mv) <= radius for mv in _book_openings(book_id)), (
            f"run6 names {book_id}, which its encoding cannot replay"
        )
