"""AUDIT-1 F-34 — no production path builds a `Board()` that ignores the run's identity.

THE DEFECT. `mantis._engine.Board()` calls `Board::new()`, which takes
`DEFAULT_LEGAL_MOVE_RADIUS = 5` and `DEFAULT_CLUSTER_THRESHOLD = 5` — the ENGINE's defaults,
not the encoding's. Six Python sites constructed one: both `data/replay.py` replayers,
`data/generate.py::_play_one_game`, both `data/corpus_metrics.py` analysers, and
`train/pretrain/dataset.py::_game_winner_from_replay`. So generated bot games were
radius-5 constrained regardless of the identity they feed, and cluster counts on a `v6w25`
corpus — registry threshold 8 — were computed under a rule that corpus never played by.

THE WORST OF THEM IS DELETED. `_game_winner_from_replay` wrapped every `apply_move` in
`except Exception: break` on top of the blind board, so a game whose moves are illegal at
radius 5 truncated SILENTLY and it returned a winner read off the truncated position. At
radius 6 that refusal is 34.76 % of the bootstrap corpus (R327) — the exact measurement it
would have hidden. It was exported and had no caller anywhere.

The threaded form already existed (`Board.with_encoding_name`, used by `bootstrap_encode`,
`eval/worker` and `acceptance_witness`); these sites simply were not using it.
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


def test_the_truncating_winner_helper_stays_deleted() -> None:
    """A grave (R-graves discipline): the symbol is gone from the module AND from the package
    export, so re-adding it is a visible act rather than a re-import."""
    from mantis.train import pretrain
    from mantis.train.pretrain import dataset

    assert not hasattr(dataset, "_game_winner_from_replay")
    assert not hasattr(pretrain, "_game_winner_from_replay")
    assert "_game_winner_from_replay" not in pretrain.__all__


def test_an_identity_bound_board_carries_the_encodings_geometry() -> None:
    """The positive half: the threaded form actually differs from the blind one, so the repair
    is not cosmetic. `v6_live2_ls` mints radius 5 / threshold 5; `v6w25` mints threshold 8."""
    from mantis._engine import Board
    from mantis.encoding import lookup

    w25 = Board.with_encoding_name("v6w25")
    assert w25.cluster_threshold() == lookup("v6w25").cluster_threshold == 8, (
        "the identity-bound board does not carry the encoding's cluster threshold — the "
        "repair would be a no-op and this row would prove nothing"
    )


@pytest.mark.parametrize("name", ["v6", "v6_live2_ls"])
def test_each_replayer_binds_the_encoding_it_is_named_for(name: str) -> None:
    """`replay.py`'s two replayers are per-encoding by construction, so each names its own
    identity rather than taking whatever `Board()` hands back."""
    import inspect

    from mantis.data import replay

    fn = {"v6": replay.replay_game_to_triples_v6,
          "v6_live2_ls": replay.replay_game_to_triples_ls}[name]
    src = inspect.getsource(fn)
    assert f'Board.with_encoding_name("{name}")' in src, (
        f"the {name} replayer no longer binds its own encoding"
    )
