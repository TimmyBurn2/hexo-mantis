"""AUDIT-1 F-34 — no production path builds a `Board()` that ignores the run's identity.

THE DEFECT. `mantis._engine.Board()` calls `Board::new()`, which takes
`DEFAULT_LEGAL_MOVE_RADIUS = 5` and `DEFAULT_CLUSTER_THRESHOLD = 5` — the ENGINE's defaults,
not the encoding's. Six Python sites constructed one: both `data/replay.py` replayers,
`data/generate.py::_play_one_game`, both `data/corpus_metrics.py` analysers, and
`train/pretrain/dataset.py::_game_winner_from_replay`. So generated bot games were
radius-5 constrained regardless of the identity they feed, and cluster counts on a `v6w25`
corpus — registry threshold 8 — were computed under a rule that corpus never played by.

THE THREADED FORM (`Board.with_encoding_name`) is what the surviving sites use; the census
below is the standing guard that no new site reverts to the blind one.

FOUR OF THE SIX SITES ARE NOW DELETED OUTRIGHT (R346(f)): `data/replay.py`'s two replayers,
`data/corpus_metrics.py` and `train/pretrain/dataset.py` went with the grid path, and the
three grid encodings this file used to measure geometry against (`v6`, `v6w25`,
`v6_live2_ls`) left the registry with them. The per-replayer rows and the
`_game_winner_from_replay` grave are retired for the same reason — a grave over a module
that no longer exists is not a guard, it is an import error waiting for a reader. What
remains is the property that outlived the subject: the census, its self-test, and the
positive half re-pointed onto the geometry the graph encodings DO carry.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src" / "mantis"


def _no_arg_board_calls() -> list[str]:
    """Every `Board()` with no arguments under `src/mantis/`, by AST — a text search would
    also hit `Board.with_encoding_name(...)` and every mention in a comment."""
    found: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id == "Board" and not node.args and not node.keywords:
                found.append(f"{path.relative_to(_SRC)}:{node.lineno}")
    return found


def test_no_production_module_builds_an_identity_blind_board() -> None:
    offenders = _no_arg_board_calls()
    assert not offenders, (
        f"a no-arg `Board()` under src/: {offenders}. It takes the ENGINE defaults (radius 5, "
        "cluster threshold 5) whatever the encoding says. Use "
        "`Board.with_encoding_name(<the resolved identity>)` (AUDIT-1 F-34)."
    )


def test_the_census_can_see_a_planted_one(tmp_path: Path) -> None:
    """LAW-07 self-test — a census that matches nothing would pass forever."""
    planted = tmp_path / "mut.py"
    planted.write_text("def f():\n    return Board()\n", encoding="utf-8")
    tree = ast.parse(planted.read_text(encoding="utf-8"))
    hits = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "Board" and not n.args and not n.keywords
    ]
    assert len(hits) == 1, "the finder's own predicate does not match a planted no-arg Board()"


def test_an_identity_bound_board_carries_the_encodings_geometry() -> None:
    """The positive half: the threaded form actually differs from the blind one, so the repair
    is not cosmetic.

    Measured on the RADIUS now rather than the cluster threshold: the graph encodings mint
    `cluster_threshold = "none"`, so the field the grid rows differed in no longer exists to
    differ. `gnn_axis_r8` mints radius 8 and `gnn_axis_v1` mints 6, both against the engine's
    default 5 — one identity-bound board, one blind one, and the two disagree.
    """
    from mantis._engine import Board
    from mantis.encoding import lookup

    r8 = Board.with_encoding_name("gnn_axis_r8")
    assert r8.legal_move_radius() == lookup("gnn_axis_r8").legal_move_radius == 8, (
        "the identity-bound board does not carry the encoding's legal-move radius — the "
        "repair would be a no-op and this row would prove nothing"
    )
    assert Board().legal_move_radius() != r8.legal_move_radius(), (
        "the blind board and the identity-bound one agree, so the whole census above is "
        "measuring a distinction with no difference"
    )
