"""Prove the draw-rate ring's capacity is the minted `consec`, derived rather than a constant.

A literal history depth beside a rule that refuses on `len(history) < consec` made every `consec`
above that depth permanently unfireable — armed in the config, absent in effect — while the
cadence audit published a finite fire step for it.

Killers: a resurrected trim literal of any value; a trim keyed to the wrong spec field; no trim at
all; a trim on the SKIP path; `>=` -> `>` on the rule's length gate.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

from mantis.config.armed_aborts import Cadence
from mantis.config.loader import load_config
from mantis.config.resolve.coordinator import resolve_coordinator_knobs
from mantis.config.resolve.drain import resolve_drain_caps
from mantis.config.resolve.draw_rate import DrawRateAbortSpec
from mantis.monitor.config import MonitorConfig
from mantis.run import _step_coordinator_config
from mantis.train.coordinator.step import StepCoordinator
from mantis.train.lifecycle.signals import ShutdownState

#: One above the deleted history-depth literal of 32 — the first value that was unfireable. A
#: test INPUT, not an authority: no shipped code may know the number 32 ever existed.
_ABOVE_OLD_DEPTH = 33

#: `min_step=0` keeps the fire in reach of direct drives; `N_pool_min=10` is deliberately
#: unequal to every `consec` here and below the 12-observation drive, so a wrong-field trim shows.
_SPEC = DrawRateAbortSpec(threshold=0.4, min_step=0, N_pool_min=10, consec=3)


class _Pool:
    """Serve the pool surface the draw-rate gate touches, with mutable counts so one
    coordinator can be driven through observation and blackout boundaries alike."""

    games_completed = 0

    def __init__(self, counts: tuple[int, int]) -> None:
        self._counts = (int(counts[0]), int(counts[1]))
        self.counts_calls = 0

    def set_counts(self, counts: tuple[int, int]) -> None:
        self._counts = (int(counts[0]), int(counts[1]))

    def pooled_draw_counts(self) -> tuple[int, int]:
        self.counts_calls += 1
        return self._counts


class _Buffer:
    size, capacity = 1000, 100_000

    def save_to_path(self, path) -> None:
        return None


class _SpySink:
    def __init__(self) -> None:
        self.events: list = []

    def emit(self, event) -> None:
        self.events.append(event)


_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "dev_example.yaml"
_DRAIN_CAPS = resolve_drain_caps(load_config(_CONFIG_PATH).monitor)
_KNOBS = resolve_coordinator_knobs(load_config(_CONFIG_PATH).train)
_GATE_INTERVAL = load_config(_CONFIG_PATH).monitor.gate_interval


def _coordinator(*, spec, pool):
    config = dataclasses.replace(
        _step_coordinator_config(stop_step=10**9, draw_rate_abort=spec,
                                 drain_caps=_DRAIN_CAPS, gate_interval=_GATE_INTERVAL,
                                 knobs=_KNOBS),
        log_interval=1, gate_interval=1, eval_interval=1, min_buf_size=1,
        terminal_eval_enabled=False,
    )
    shutdown = ShutdownState()
    coord = StepCoordinator(
        trainer=SimpleNamespace(step=0), buffer=_Buffer(), pretrained_buffer=None,
        recent_buffer=None, pool=pool, eval_pipeline=None,
        subsystems=SimpleNamespace(gpu_monitor=None),
        anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
        shutdown=shutdown, eval_model=object(), bufs=None, config=config,
        full_config={}, train_cfg={}, mixing_cfg={}, sink=_SpySink(),
        heartbeat=None, monitor_cfg=MonitorConfig(),
    )
    return SimpleNamespace(coord=coord, pool=pool, shutdown=shutdown, config=config)


def test_a_consec_above_the_old_depth_fires_at_the_consec_th_observation() -> None:
    """Prove a `consec` above the old depth fires at exactly its consec-th observation; the
    not-fired-before half pins the other direction, so an early fire is caught too."""
    spec = dataclasses.replace(_SPEC, consec=_ABOVE_OLD_DEPTH)
    h = _coordinator(spec=spec, pool=_Pool((90, 100)))  # rate 0.9 >= threshold 0.4
    for boundary in range(1, _ABOVE_OLD_DEPTH):
        assert h.coord._run_hard_abort_gates(h.config) is False, (
            f"fired at observation {boundary} < consec={_ABOVE_OLD_DEPTH}: the rule must "
            "wait for the consec-th observation"
        )
    assert h.coord._run_hard_abort_gates(h.config) is True, (
        f"consec={_ABOVE_OLD_DEPTH} did not fire at its consec-th observation: the ring "
        "is being clipped below the minted `consec` — the ADJ-D36 fifth face is back"
    )


def test_the_ring_capacity_is_the_minted_consec_not_a_constant() -> None:
    """Prove the ring holds exactly `consec` entries after 12 observations, at two `consec`s: any
    constant capacity yields the same length for both and reds at least one assertion."""
    for consec in (2, 5):
        h = _coordinator(spec=dataclasses.replace(_SPEC, consec=consec),
                         pool=_Pool((0, 100)))  # rate 0.0 < threshold: observe, never fire
        for _ in range(12):
            assert h.coord._run_hard_abort_gates(h.config) is False
        assert len(h.coord._draw_rate_history) == consec, (
            f"ring holds {len(h.coord._draw_rate_history)} entries for consec={consec}: "
            "capacity must BE the minted `consec` (one authority, derived at the point of "
            "use), not a constant and not another spec field"
        )


def test_a_skipped_boundary_neither_appends_nor_resets_above_the_old_depth() -> None:
    """Prove a skipped boundary neither appends to nor resets the ring, above the old depth:
    `consec=34` fires at the 34th OBSERVATION, which is the 35th boundary."""
    spec = dataclasses.replace(_SPEC, consec=34)
    pool = _Pool((90, 100))
    h = _coordinator(spec=spec, pool=pool)
    for _ in range(20):
        assert h.coord._run_hard_abort_gates(h.config) is False
    ring_before = list(h.coord._draw_rate_history)
    pool.set_counts((9, 9))  # completed 9 < N_pool_min 10: NO OBSERVATION (R92)
    assert h.coord._run_hard_abort_gates(h.config) is False
    assert h.coord._draw_rate_history == ring_before, (
        "a blackout boundary touched the ring: a skipped boundary must neither append "
        "nor trim nor reset (R92/BUG-1)"
    )
    pool.set_counts((90, 100))
    for observation in range(21, 34):
        assert h.coord._run_hard_abort_gates(h.config) is False, (
            f"fired at observation {observation} < consec=34"
        )
    assert h.coord._run_hard_abort_gates(h.config) is True, (
        "consec=34 did not fire at its 34th observation across a blackout: either the "
        "skip reset the ring or the capacity is clipped below the minted `consec`"
    )


def test_the_published_earliest_fire_step_is_deliverable_above_the_old_depth() -> None:
    """Match the cadence audit's published earliest fire step against a REAL fire.

    At `gate_interval=1` one direct gate call is one boundary, so the published arithmetic must
    equal the measured fire boundary exactly.
    """
    spec = dataclasses.replace(_SPEC, consec=_ABOVE_OLD_DEPTH)
    h = _coordinator(spec=spec, pool=_Pool((90, 100)))
    # Read off the harness config, never re-typed: a hand-copied interval would fork the one
    # authority the published number and the machine are supposed to share.
    interval = h.config.gate_interval
    # The interval is the row's sample-clock PERIOD, not an operand — the same place production
    # supplies it from.
    published = Cadence.GATE_INTERVAL_CONSEC.earliest_fire_step(
        (spec.consec, spec.min_step), period_steps=interval)
    assert published == float(interval * _ABOVE_OLD_DEPTH), (
        f"cadence arithmetic answered {published!r} for interval={interval}, "
        f"consec={spec.consec}, min_step=0 — expected the consec-th boundary in steps"
    )
    fired_at = None
    for boundary in range(1, _ABOVE_OLD_DEPTH + 1):
        if h.coord._run_hard_abort_gates(h.config):
            fired_at = boundary
            break
    assert fired_at is not None and float(interval * fired_at) == published, (
        f"the audit publishes earliest fire step {published!r} but the machine fired at "
        f"boundary {fired_at!r} (interval {interval}): the published number must be "
        "deliverable by the code that evaluates the row"
    )
