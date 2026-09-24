"""The three per-boundary events publish exactly the key rosters the event contract lists."""
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mantis.monitor.sink import EVENT_CONTRACT
from mantis.selfplay.pool_hooks import RunnerStats
from mantis.train.coordinator.step import StepCoordinator
from mantis.train.events import emit_iteration_complete_event, emit_training_step_event
from _spy import SpyEventSink

_DOC = Path(__file__).resolve().parents[2] / "docs" / "contracts" / "event_manifest.md"
_ROSTER = re.compile(r"^- `(?P<event>\w+)` \([^)]*\): (?P<keys>.+)$")


def _doc_rosters() -> dict[str, set[str]]:
    """The `Published field rosters` section, as event name to its listed keys."""
    text = _DOC.read_text(encoding="utf-8")
    section = text.split("## Published field rosters", 1)[1].split("\n## ", 1)[0]
    rosters: dict[str, set[str]] = {}
    for line in section.splitlines():
        match = _ROSTER.match(line)
        if match:
            rosters[match["event"]] = set(re.findall(r"`(\w+)`", match["keys"]))
    return rosters


def _training_step() -> dict[str, Any]:
    return emit_training_step_event(
        0, {"loss": 1.0, "policy_loss": 0.6, "value_loss": 0.4, "grad_norm": 0.1, "lr": 1e-3,
            "policy_entropy": 2.0, "policy_entropy_selfplay": 2.0}, SpyEventSink())


def _iteration_complete() -> dict[str, Any]:
    sink = SpyEventSink()
    rstats = RunnerStats(
        games_completed=1, positions_generated=4, x_wins=1, o_wins=0, draws=0, model_version=0,
        mcts_quiescence_fires=0, mcts_mean_depth=1.0, mcts_mean_root_concentration=0.5,
        pcr_full_moves=4, pcr_quick_moves=0, gumbel_round_leaves=8, gumbel_rounds=4)
    pool = SimpleNamespace(search_kind="gumbel", avg_game_length=4.0, x_winrate=1.0,
                           o_winrate=0.0, draw_rate=0.0, sims_per_sec=None, batch_fill_pct=0.0)
    emit_iteration_complete_event(
        1, 1, 0, pool, SimpleNamespace(size=4, capacity=8), lambda: None, None, {},
        rstats, sink, search_levers={})
    return sink.events[0]


def _monitor_gates() -> dict[str, Any]:
    sink = SpyEventSink()
    coord = SimpleNamespace(_train_step=1, _gate_stats={}, _policy_loss_reference=None,
                            _policy_loss_window_means=[], _ply_cap_rate=None,
                            _watchdog_counters=lambda: None)
    cfg = SimpleNamespace(draw_rate_abort=None, policy_loss_trough_abort=None, ply_cap_abort=None)
    StepCoordinator._emit_monitor_gates(coord, cfg, sink)  # type: ignore[arg-type]
    return sink.events[0]


_BUILDERS = {"training_step": _training_step, "iteration_complete": _iteration_complete,
             "monitor_gates": _monitor_gates}


def test_the_contract_lists_a_roster_for_every_checked_event() -> None:
    assert set(_doc_rosters()) == set(_BUILDERS), f"doc rosters: {sorted(_doc_rosters())}"


@pytest.mark.parametrize("event", sorted(_BUILDERS))
def test_the_published_keys_equal_the_contract_roster(event: str) -> None:
    """A key with no roster row is undocumented; a row no builder publishes is a phantom."""
    published = set(_BUILDERS[event]())
    listed = _doc_rosters()[event]
    extra, phantom = sorted(published - listed), sorted(listed - published)
    assert extra == [], f"{event} publishes keys the contract omits: {extra}"
    assert phantom == [], f"{event}: the contract lists keys nobody publishes: {phantom}"


def test_the_contract_version_is_the_one_the_stream_stamps() -> None:
    version = re.search(r"^- version: (v\d+)$", _DOC.read_text(encoding="utf-8"), re.MULTILINE)
    assert version is not None and EVENT_CONTRACT == f"event-manifest-{version[1]}"
