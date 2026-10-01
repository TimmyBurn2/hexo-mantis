"""The row-outcome authority masks exactly the no-winner ends and refuses a winner that disagrees with its reason."""
from __future__ import annotations

import pytest

from mantis._engine import graph_row_outcome


@pytest.mark.parametrize(("winner", "reason", "want"), [
    (1, 0, (1.0, 1)), (-1, 0, (-1.0, 1)), (1, 1, (1.0, 1)), (0, 2, (0.0, 0)), (0, 3, (0.0, 0)),
])
def test_a_decided_game_trains_its_result_and_a_game_with_no_winner_trains_none(winner: int, reason: int,
                                                                                   want: tuple[float, int]) -> None:
    assert graph_row_outcome(1, winner, reason) == want


@pytest.mark.parametrize(("winner", "reason"), [(0, 0), (0, 1), (1, 2), (-1, 3), (1, 4), (0, 4)])
def test_a_winner_that_disagrees_with_its_reason_is_refused(winner: int, reason: int) -> None:
    """PLANTED BREAK: drop the agreement check and a no-winner game on reason 0 trains a draw at 0."""
    with pytest.raises(ValueError, match="disagree"):
        graph_row_outcome(1, winner, reason)
