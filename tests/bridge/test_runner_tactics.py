"""The self-play runner config arms a tactics block, None disarms, a bad one is refused, and the totals name every row."""
from __future__ import annotations

import pytest

from mantis._engine import SelfPlayRunner, SelfPlayRunnerConfig
from mantis.config.resolve.tactics import tactics_block

_BLOCK = tactics_block({"kind": "strict_turn", "leaf_turns": 3, "leaf_nodes": 256, "root_turns": 8, "root_nodes": 20000,
                        "audit": {"turns": 8, "nodes": 2000, "k": 4, "m": 4, "total_nodes": 40000}})


def _config() -> SelfPlayRunnerConfig:
    return SelfPlayRunnerConfig(n_workers=1, q_rescale=True, search_stats_every=0, gumbel_m_quick=16, encoding_name="gnn_axis_r8")


def test_a_block_arms_the_runner_config_and_none_disarms_it() -> None:
    cfg = _config()
    assert not cfg.tactics_armed
    cfg.configure_tactics(_BLOCK)
    assert cfg.tactics_armed
    cfg.configure_tactics(None)
    assert not cfg.tactics_armed


def test_a_malformed_block_is_refused_by_name() -> None:
    assert _BLOCK is not None
    cfg = _config()
    cfg.configure_tactics(_BLOCK)
    with pytest.raises(ValueError, match="tactics block"):
        cfg.configure_tactics({**_BLOCK, "kind": "per_stone"})
    assert not cfg.tactics_armed, "a refused block leaves the config disarmed, not on its old block"


def test_the_runner_names_every_tactics_row_and_reads_zero_before_it_searches() -> None:
    cfg = _config()
    cfg.configure_tactics(_BLOCK)
    totals = SelfPlayRunner(cfg).tactics_totals()
    for row in ("descents", "root_proofs_found", "root_vetoes", "decided_lost", "proven_root_rows",
                "decided_lost_rows", "vetoed_target_rows", "emptied_target_rows", "vetoed_all_rows",
                "tail_emptied_rows", "tail_leak_rows", "unsearched_decided_rows"):
        assert totals[row] == 0, row
    assert "mixed_rows" not in totals, "the proof mixture's row left the tree with the mixture"
