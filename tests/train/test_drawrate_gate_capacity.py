"""ADJ-D36 oracle — the draw-rate ring's capacity is the minted `consec`, derived, never
a constant, so NO schema-legal `consec` is unfireable and the R251 cadence audit's
published `earliest_fire_step` is deliverable by construction.

WHAT WENT WRONG. `step.py::_sample` trimmed the gate history to a literal
`_GATE_HISTORY_DEPTH = 32` while `rules.py::check_draw_rate_collapse` refuses on
`len(history) < consec`: any schema-legal `consec >= 33` (`ge=1`, no upper bound) was
PERMANENTLY unfireable — armed in the config, absent in effect (the schema's own "fifth
face") — while `Cadence.GATE_INTERVAL_CONSEC` computed a finite fire step for it and
PUBLISHED that number as though the run could deliver it (gate 12 audited such a row
GREEN). The fix derives the capacity from the ONE authority both sides already read:
`train.draw_rate_abort.consec`. No literal, no new config key, no import-DAG edge.

THE DRIVES BELOW STATE THE MUTATIONS THAT RED THEM:

* a resurrected trim literal (any value: 32, or a "generous" 64) — the capacity pin
  measures `len(ring) == consec` for consec 2 and 5 after 12 observations, and the
  above-depth fire drive reds for any literal below its `consec=33`;
* trim keyed to the WRONG spec field (`N_pool_min=10` here, deliberately != consec and
  < the 12-observation drive) — the capacity pin reds on `len == 10`;
* no trim at all (unbounded ring) — the capacity pin reds on `len == 12`;
* trim on the SKIP path (a boundary that observes nothing must never touch the ring,
  R92/BUG-1) — the blackout drive reds if the skip shrinks or resets the ring;
* `>=` -> `>` on the rule's length gate — the fire drives red one observation late.

The coordinator config comes from the production builder (`mantis.run.
_step_coordinator_config`), and `_run_hard_abort_gates` is called DIRECTLY so one call is
one gate boundary — same posture as `test_drawrate_gate_branch_flipset.py`, whose fakes
are duplicated locally per R5 (no cross-test import). R7 / gate 6: nothing here writes a
file.
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

#: One above the DELETED `_GATE_HISTORY_DEPTH = 32` — the exact first value ADJ-D36 names
#: as unfireable on the clipped code. A test INPUT, not an authority: the fix must make
#: this value fireable without any shipped code knowing the number 32 ever existed.
_ABOVE_OLD_DEPTH = 33

#: `min_step=0` keeps the fire in reach of direct drives (the harness never advances
#: `_train_step`); `N_pool_min=10` is deliberately != every `consec` used here AND below
#: the 12-observation capacity drive, so a trim keyed to the wrong field is measurable.
_SPEC = DrawRateAbortSpec(threshold=0.4, min_step=0, N_pool_min=10, consec=3)


# ── local fakes (R5: no cross-test import) ────────────────────────────────────────────
class _Pool:
    """The pool surface the draw-rate gate touches, and only that — counts are MUTABLE
    (`set_counts`) so ONE coordinator can be driven through observation AND blackout
    boundaries, which the blackout drive needs and a fixed-counts fake cannot do."""

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


# ── the fifth face, closed: an above-the-old-depth `consec` FIRES ─────────────────────
def test_a_consec_above_the_old_depth_fires_at_the_consec_th_observation() -> None:
    """`consec = 33` fires at exactly the 33rd observed boundary.

    REDs on the clipped code (`del history[:-32]` caps `len(history)` at 32 forever, so
    `len(history) >= 33` is unsatisfiable and the abort NEVER fires — the fifth face),
    and reds on any resurrected literal below 33. The not-fired-through-32 half pins the
    other direction: a change that fires EARLY (a widened tail, or length-gate drift) is
    caught here too, not just the unfireable defect."""
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
    """After 12 healthy observations the ring holds EXACTLY `consec` entries — for
    consec=2 AND consec=5, same drive.

    The two-value drive is what makes this a DERIVATION pin rather than a size pin: any
    constant capacity (32 resurrected, 64 "to be safe", or the wrong spec field —
    `N_pool_min=10` sits between 5 and 12 on purpose) yields the same length for both
    coordinators and reds at least one of the two assertions; no-trim-at-all yields 12
    and reds both."""
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
    """BUG-1's neither-advances-nor-resets contract, driven ABOVE the old depth.

    20 observations, one below-`N_pool_min` blackout boundary, then 14 more observations:
    `consec=34` fires at the 34th OBSERVATION, which is the 35th BOUNDARY. Reds if the
    skip path touches the ring (trim-on-skip, reset-on-skip: the ring would shrink below
    a satisfiable 34-tail, or the fire would need 34 fresh observations), and reds on the
    clipped code, where 34 can never fire at all. Extends `test_gate_interval_decoupling.
    py::test_p10_a_skipped_boundary_neither_advances_nor_resets_consec` past the deleted
    depth, where the old code's behaviour DIVERGES from the contract instead of agreeing
    with it."""
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


# ── the audit tie: the published earliest fire step is deliverable ────────────────────
def test_the_published_earliest_fire_step_is_deliverable_above_the_old_depth() -> None:
    """`Cadence.GATE_INTERVAL_CONSEC`'s published number, matched against a REAL fire at
    a `consec` the old code could never satisfy — the ADJ-D36 false affirmative, killed.

    At `gate_interval=1` a direct `_run_hard_abort_gates` call is one boundary, so the
    member's `interval * max(consec, ceil(min_step/interval))` must equal the measured
    fire boundary EXACTLY. On the clipped code this arithmetic answers 33.0 while the
    machine never fires — the audit publishing a number the run structurally cannot
    deliver is precisely what R251 exists to refuse, one knob over (LAW-07: the audit
    input cites its live producer, and this drive is the citation)."""
    spec = dataclasses.replace(_SPEC, consec=_ABOVE_OLD_DEPTH)
    h = _coordinator(spec=spec, pool=_Pool((90, 100)))
    # The interval operand is READ OFF THE HARNESS CONFIG, never re-typed: this drive's
    # whole point is that the published number and the machine share one authority, and a
    # hand-copied interval would be that authority forked (the harness pins it to 1 so a
    # direct `_run_hard_abort_gates` call is one boundary).
    interval = h.config.gate_interval
    published = Cadence.GATE_INTERVAL_CONSEC.earliest_fire_step(
        (interval, spec.consec, spec.min_step))
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
