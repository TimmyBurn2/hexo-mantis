"""AUDIT-1 F-15 — the eval arena caps at the RUN's `selfplay.max_game_moves`, not at a constant.

THE DEFECT. `arena/match.py::DEFAULT_MAX_PLIES = 128` defaulted `_play_one_game` and
`play_paired_match`, and its own comment said it *"mirrors the production self-play default
(`SelfPlayRunnerConfig.max_moves_per_game=128`)"* — a copy of a bridge signature default, which
is itself a copy of the minted `selfplay.max_game_moves`. Three copies of one number. And the
eval side passed none of them: `eval/worker.py`, `eval/pipeline.py` and `run.py` never set
`max_plies`, so every eval game capped at the module constant.

WHY IT MATTERS RATHER THAN BEING TIDY. The ply cap is half of the ply-cap x adjudication matrix
that is an operator-owed prereg row. The moment `max_game_moves` is re-minted, eval would keep
capping at 128 with NO config diff, and the draw channel — a capped game is a draw when no
adjudicator is armed — would change meaning underneath the bar LAW-15 calls deploy-matched.

THE REPAIR is threading, not a new number: `max_plies` is REQUIRED on both arena entry points
and on `RoundSpec`, resolved once in the parent from `config.selfplay.max_game_moves`. The
constant survives as a TEST fixture cap and as nothing else.
"""
from __future__ import annotations

from typing import Any

import pytest

from mantis.arena.match import _play_one_game
from mantis.arena.regime import RegimeKey
from mantis.eval.rounds import RoundSpec


class _FirstLegalBot:
    """Plays the first legal move forever — so the game can only end at the cap."""

    def name(self) -> str:
        return "first_legal"

    def new_game(self) -> None: ...

    def select_move(self, board: Any) -> tuple[int, int]:
        return board.legal_moves()[0]


def _board_factory() -> Any:
    from mantis._engine import Board

    return Board.with_encoding_name("v6_live2_ls")


@pytest.mark.parametrize("cap", [7, 37, 64])
def test_a_game_ends_at_the_cap_it_was_given(cap: int) -> None:
    """The audit's own pin, generalised to three values. ONE value would pass against a
    surviving 128-literal if it happened to be 128; three cannot."""
    winner, plies, moves, terminal, _adj = _play_one_game(
        _FirstLegalBot(), _FirstLegalBot(), [],
        candidate_color=1, board_factory=_board_factory, max_plies=cap,
    )
    assert plies == cap, f"asked for a {cap}-ply cap, played {plies}"
    assert len(moves) == cap
    assert winner == "draw" and terminal == "ply_cap", (
        "a capped game with no adjudicator is a draw — the channel whose meaning would have "
        "moved silently when `max_game_moves` was re-minted"
    )


def test_neither_arena_entry_point_carries_a_ply_cap_DEFAULT() -> None:
    """The structural half. A default here is how the eval side capped at a constant while
    never mentioning one: every call site simply omitted the argument."""
    import inspect

    from mantis.arena.match import play_paired_match

    for fn in (_play_one_game, play_paired_match):
        param = inspect.signature(fn).parameters["max_plies"]
        assert param.default is inspect.Parameter.empty, (
            f"{fn.__name__} has a `max_plies` default again — a caller that omits it then "
            "silently caps at a number nobody minted (AUDIT-1 F-15)"
        )


def test_the_round_spec_carries_the_cap_across_the_process_seam() -> None:
    """`RoundSpec` is how anything reaches the eval CHILD. A field that is not on it is a
    value the child cannot have, which is why the cap lives here beside `leaf_batch_size`."""
    assert "max_plies" in RoundSpec.__dataclass_fields__
    assert RoundSpec.__dataclass_fields__["max_plies"].default is __import__(
        "dataclasses"
    ).MISSING, "the seam field acquired a default — the same defect, one layer out"


def test_the_pipeline_takes_the_cap_from_the_config_and_hands_it_on() -> None:
    """The parent half, end to end: a config's own `selfplay.max_game_moves` is what the
    pipeline carries, so re-minting the key moves the eval cap with it."""
    import inspect

    from mantis.eval.pipeline import build_eval_pipeline
    from mantis.run import compose_run  # noqa: F401  (imported to prove the module loads)

    param = inspect.signature(build_eval_pipeline).parameters["max_plies"]
    assert param.default is inspect.Parameter.empty, (
        "the pipeline builder defaulted the cap — the parent would then supply a number the "
        "operator did not mint"
    )
    src = inspect.getsource(__import__("mantis.run", fromlist=["run"]).compose_run)
    assert "max_plies=config.selfplay.max_game_moves" in src, (
        "the composition root no longer threads the RUN's cap into the eval pipeline; the "
        "eval bar and the self-play regime have drifted apart again (LAW-15)"
    )


def test_the_regime_key_is_constructible() -> None:
    """Vacuity guard for the parametrised rows above: they exercise a real arena game, so the
    arena's own vocabulary has to be importable for their result to mean anything."""
    assert RegimeKey is not None
