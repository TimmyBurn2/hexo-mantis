"""⊕ WP11-A DESIGN §a.3/§c.2 — the G-2 `wr_sealbot` handshake (WP13-A handoff). `build_round_
result` UNCONDITIONALLY sets `wr_sealbot` (float | None — the routed result's key is ALWAYS
present, never absent), populated from the FIRST sealbot-kind rung (in ladder order) that
recorded >=1 game this round; absent/no-sealbot-games routes `None`, which drives the REAL
`StepCoordinator.on_eval_round_complete`'s existing skip counter (step.py:472-524, read this
session — already exists at HEAD).

RED-at-import (file-level): the top-level `from mantis.eval.rounds import build_round_result`
anchor below makes THIS WHOLE FILE fail collection today (`mantis.eval` does not exist yet —
mirrors the house convention in tests/train/test_coordinator_gates.py). Per-test provenance,
for the record (all currently unreachable behind the same collection error):
  * 4 of 6 tests need `mantis.eval.rounds.build_round_result` directly — genuinely new-RED.
  * `test_absent_sealbot_rounds_route_none_and_coordinator_skip_counts` would ALSO collect and
    PASS today in isolation (it only needs `mantis.train.coordinator.step`/`.config`, already
    existing, and hand-constructs the routed result dict) — kept in this file because it is
    the OTHER half of the G-2 handshake this suite pins as one unit; the shared anchor import
    is what makes it RED-at-import here rather than a silent pre-existing-green outlier.
  * `test_manifest_sealbot_row_is_now_a_resolving_symbol` / `_mutation_bites_on_a_renamed_
    symbol` use `mantis.monitor.manifest` (ALREADY EXISTS) and would be RED-by-assertion on
    their own (the shipped manifest's `sealbot_wr_warn` row is still `kind: seam` /
    `pending: WP11A` today) — also swept behind the shared anchor here.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mantis.eval.rounds import build_round_result  # noqa: F401 — RED-at-import anchor

_REPO = Path(__file__).resolve().parents[2]
_MANIFEST = _REPO / "src" / "mantis" / "monitor" / "producer_manifest.yaml"


def _rung_cfg(name: str, bot: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, bot=bot)


_LADDER_ORDER = [
    _rung_cfg("sealbot_d5", "sealbot"),
    _rung_cfg("kraken_raw", "kraken"),
    _rung_cfg("sealbot_d6", "sealbot"),
]


def _rung_result(*, games: int, wr) -> dict:
    return {"games": games, "wins": 0, "losses": 0, "draws": 0, "wr": wr,
            "wr_ci_lower": None, "wr_ci_upper": None, "eff_n": games,
            "regime_key": "k", "status": "active"}


def _base_kwargs(**overrides) -> dict:
    base = dict(
        step=1000, round_id="r000001_1000", rungs_config=_LADDER_ORDER, rung_results={},
        gate_result=None, skipped_rungs=[], bt={"ratings": {}, "p_hat": {}},
        schedule_next={}, eval_round_wall_sec=1.0, eval_broken=False, error=None,
        random_wr=None,
    )
    base.update(overrides)
    return base


def test_round_result_always_carries_wr_sealbot() -> None:
    from mantis.eval.rounds import build_round_result

    success = build_round_result(**_base_kwargs(rung_results={}))
    assert "wr_sealbot" in success and success["wr_sealbot"] is None

    broken = build_round_result(**_base_kwargs(eval_broken=True, error="killed"))
    assert "wr_sealbot" in broken and (broken["wr_sealbot"] is None or isinstance(broken["wr_sealbot"], float))

    all_skip = build_round_result(**_base_kwargs(
        rung_results={}, skipped_rungs=[{"rung": r.name, "reason": "no adapter"} for r in _LADDER_ORDER],
    ))
    assert "wr_sealbot" in all_skip and all_skip["wr_sealbot"] is None


def test_wr_sealbot_populates_from_first_sealbot_rung_with_games() -> None:
    from mantis.eval.rounds import build_round_result

    rung_results = {
        "kraken_raw": _rung_result(games=8, wr=0.5),      # NOT sealbot — must be ignored
        "sealbot_d5": _rung_result(games=6, wr=0.75),     # FIRST sealbot rung with games
        "sealbot_d6": _rung_result(games=6, wr=0.10),     # a later sealbot rung — must be ignored
    }
    result = build_round_result(**_base_kwargs(rung_results=rung_results))
    assert result["wr_sealbot"] == pytest.approx(0.75), (
        "wr_sealbot must come from the FIRST sealbot-kind rung (ladder order) with games, "
        "never a non-sealbot rung and never a LATER sealbot rung"
    )


def test_sealbot_rung_with_zero_games_this_round_is_skipped_for_the_handshake() -> None:
    from mantis.eval.rounds import build_round_result

    rung_results = {
        "sealbot_d5": _rung_result(games=0, wr=None),     # zero games this round -> not a source
        "sealbot_d6": _rung_result(games=4, wr=0.60),     # this one has games -> the real source
    }
    result = build_round_result(**_base_kwargs(rung_results=rung_results))
    assert result["wr_sealbot"] == pytest.approx(0.60)


def test_absent_sealbot_rounds_route_none_and_coordinator_skip_counts() -> None:
    """The OTHER half of the handshake — the REAL, already-existing coordinator consumer
    (step.py:472-524). Hand-constructed result dict (no mantis.eval import needed for this
    one test)."""
    import dataclasses

    from mantis.config.loader import load_config
    from mantis.config.resolve.drain import resolve_drain_caps
    from mantis.monitor.config import MonitorConfig
    from mantis.run import _step_coordinator_config
    from mantis.train.coordinator.step import StepCoordinator
    from mantis.train.lifecycle.signals import ShutdownState

    class _Sink:
        def __init__(self) -> None:
            self.events: list[dict] = []

        def emit(self, event) -> None:
            self.events.append(dict(event))

        def named(self, name: str) -> list[dict]:
            return [e for e in self.events if e.get("event") == name]

    class _Pool:
        games_completed = 0
        n_workers = 1

        def start(self) -> None: ...
        def stop(self) -> None: ...
        def buffer_composition(self) -> dict: return {}
        def pooled_draw_counts(self) -> tuple[int, int]: return (0, 0)
        def current_stride5_p90(self) -> int: return 1
        def check_producer_health(self) -> None: ...
        def update_checkpoint_step(self, step: int) -> None: ...

    class _Trainer:
        step = 0
        model = object()

        def train_step_from_tensors(self, *a, **k) -> dict: return {}
        def save_checkpoint(self, loss_info) -> None: ...

    class _Buffer:
        size = 1000
        capacity = 100_000

        def resize(self, n: int) -> None: ...
        def save_to_path(self, p) -> None: ...

    sink = _Sink()
    # WPMINT Phase K-A stage 0: DERIVED from the production builder, never a hand-written
    # 24-kwarg census (which is why a new coordinator knob costs this file no edit). `None`
    # is the EXPLICIT disarmed draw-rate posture — this harness is not about that abort —
    # and the four drain caps come from a MINTED `monitor.drain` block (R93/DR-11).
    config = dataclasses.replace(
        _step_coordinator_config(
            stop_step=10**9, draw_rate_abort=None,
            drain_caps=resolve_drain_caps(load_config(_REPO / "configs" / "dev_example.yaml").monitor),
        ),
        eval_interval=1, log_interval=1, min_buf_size=10,
    )
    coord = StepCoordinator(
        trainer=_Trainer(), buffer=_Buffer(), pretrained_buffer=None, recent_buffer=None,
        pool=_Pool(), eval_pipeline=None, subsystems=SimpleNamespace(gpu_monitor=None),
        anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
        shutdown=ShutdownState(), eval_model=object(), bufs=None,
        config=config, full_config={}, train_cfg={}, mixing_cfg={},
        sink=sink, heartbeat=None, monitor_cfg=MonitorConfig(),
    )

    coord.on_eval_round_complete({"step": 5000, "wr_sealbot": None})

    skip_events = sink.named("sealbot_wr_gate_skipped")
    assert len(skip_events) == 1
    assert skip_events[0]["reason"] == "wr_sealbot_absent"
    assert skip_events[0]["skipped_total"] == 1


def test_manifest_sealbot_row_is_now_a_resolving_symbol() -> None:
    from mantis.monitor.manifest import load_manifest

    doc = load_manifest(_MANIFEST)
    rows = {row["id"]: row for row in doc["gates"]}
    assert "sealbot_wr_warn" in rows, "the sealbot_wr_warn row must still exist"
    row = rows["sealbot_wr_warn"]
    producer = row["producer"]
    assert producer.get("kind") == "symbol", (
        "the sealbot_wr_warn row must flip from kind:'seam'/pending:'WP11A' to a resolving "
        f"kind:'symbol' row once the producer lands (got kind={producer.get('kind')!r})"
    )
    assert producer.get("module") == "mantis.eval.rounds"
    assert producer.get("symbol") == "build_round_result"
    assert "pending" not in row, "a resolved producer must not still carry a 'pending' WP tag"


def test_manifest_mutation_bites_on_a_renamed_symbol(tmp_path) -> None:
    """LAW-07 mutation self-test companion: once the row points at
    `mantis.eval.rounds.build_round_result`, renaming the symbol in a manifest copy must make
    `verify_manifest` raise (never silently pass a dead producer)."""
    from mantis.monitor.manifest import ManifestError, verify_manifest

    live_test_node = (
        "tests/monitor/test_manifest_contract.py::test_shipped_manifest_every_row_resolves"
    )
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(
        "version: 1\nchannel: jsonl_event_sink\ngates:\n"
        "  - id: sealbot_wr_warn\n"
        "    producer: {kind: symbol, module: mantis.eval.rounds, "
        "symbol: build_round_result_RENAMED_MUTANT}\n"
        f"    producer_test: {live_test_node}\n"
    )
    with pytest.raises(ManifestError) as ei:
        verify_manifest(mutated, _REPO)
    assert "sealbot_wr_warn" in str(ei.value)
