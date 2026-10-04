"""The runner refuses a quick arm wider than the full arm's m at boot, whatever the schema let through."""
from __future__ import annotations

import pytest

from mantis._engine import SelfPlayRunner, SelfPlayRunnerConfig


def _config(m: int, m_quick: int) -> SelfPlayRunnerConfig:
    return SelfPlayRunnerConfig(n_workers=1, q_rescale=True, search_stats_every=0, gumbel_m=m, gumbel_m_quick=m_quick,
                                encoding_name="gnn_axis_r8")


@pytest.mark.parametrize("m_quick", [0, 5])
def test_a_quick_m_outside_one_to_m_is_refused_at_boot(m_quick: int) -> None:
    with pytest.raises(ValueError, match="gumbel_m_quick"):
        SelfPlayRunner(_config(4, m_quick))


def test_a_quick_m_up_to_m_boots() -> None:
    assert SelfPlayRunner(_config(4, 4)).games_completed == 0
    assert SelfPlayRunner(_config(4, 1)).games_completed == 0
