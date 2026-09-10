# >300 justify (R8): one ring, one claim, and the claim has two halves that only mean something
# together — the capacity must GROW with the minted consec keys and rule B's peak window must NOT
# grow with it. Split across two files a reader could satisfy either half alone, and the
# bit-identity sweep would sit across an import from the drives that establish what it matches.
"""The sealbot-WR ring's capacity is DERIVED — from the two minted consec keys and from rule B's
own peak window — so no schema-legal consec is unfireable, and rule B's peak window does NOT
widen with the deeper ring.

WHAT WENT WRONG: `on_eval_round_complete` trimmed the ring to a literal depth of 5 while all
three triggers refuse on `len(history) >= their consec`, so any schema-legal consec >= 6 was
PERMANENTLY unfireable while the abort was armed. WHY THIS IS NOT THE DRAW-RATE FIX COPIED: the
ring served TWO masters, the consec tails AND `peak_wr`, so deriving the capacity alone would
raise rule B's bar — a behavioural change no ruling authorizes.

Mutations the drives red on: a resurrected trim literal; a capacity keyed to one consec knob
only; a capacity not floored by the peak window; a whole-ring `peak_wr`; no trim at all; a trim
on the SKIP path; and `>=` -> `>` on any length gate.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

from mantis.config.armed_aborts import Cadence
from mantis.config.loader import load_config
from mantis.config.resolve.coordinator import resolve_coordinator_knobs
from mantis.config.resolve.drain import resolve_drain_caps
from mantis.monitor.config import MonitorConfig
from mantis.monitor.rules import WR_PEAK_WINDOW_EVALS, sealbot_wr_trajectory_alert
from mantis.run import _step_coordinator_config
from mantis.train.coordinator.step import StepCoordinator
from mantis.train.lifecycle.signals import ShutdownState

#: The DELETED trim depth — a test INPUT and a historical fact, not an authority. It is asserted
#: equal to rule B's surviving window, because those were one literal.
_OLD_DEPTH = 5


class _Buffer:
    size, capacity = 1000, 100_000

    def save_to_path(self, path) -> None:
        return None


class _SpySink:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, event) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict]:
        return [event for event in self.events if event.get("event") == name]


_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "dev_example.yaml"
_DEV = load_config(_CONFIG_PATH)
_DRAIN_CAPS = resolve_drain_caps(_DEV.monitor)
_KNOBS = resolve_coordinator_knobs(_DEV.train)
_GATE_INTERVAL = _DEV.monitor.gate_interval


def _monitor_cfg(**overrides) -> MonitorConfig:
    """The shipped `MonitorConfig` with named deltas — never a re-typed field census."""
    return dataclasses.replace(MonitorConfig(), **overrides)


def _coordinator(*, monitor_cfg: MonitorConfig, eval_interval: int = 1000):
    config = dataclasses.replace(
        _step_coordinator_config(stop_step=10**9, draw_rate_abort=None,
                                 drain_caps=_DRAIN_CAPS, gate_interval=_GATE_INTERVAL,
                                 knobs=_KNOBS),
        log_interval=1, gate_interval=1, eval_interval=eval_interval, min_buf_size=1,
        terminal_eval_enabled=False,
    )
    shutdown = ShutdownState()
    sink = _SpySink()
    coord = StepCoordinator(
        trainer=SimpleNamespace(step=0), buffer=_Buffer(), pretrained_buffer=None,
        recent_buffer=None, pool=SimpleNamespace(games_completed=0), eval_pipeline=None,
        subsystems=SimpleNamespace(gpu_monitor=None),
        anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
        shutdown=shutdown, eval_model=object(), bufs=None, config=config,
        full_config={}, train_cfg={}, mixing_cfg={}, sink=sink,
        heartbeat=None, monitor_cfg=monitor_cfg,
    )
    return SimpleNamespace(coord=coord, shutdown=shutdown, config=config, sink=sink,
                           monitor_cfg=monitor_cfg)


def _round(h, index: int, wr: float) -> None:
    """Route ONE completed eval round, stamped at the step its round index lands on."""
    h.coord.on_eval_round_complete(
        {"step": index * h.config.eval_interval, "wr_sealbot": wr})


def test_a_wr_consec_above_the_old_depth_fires_at_the_consec_th_eval_round() -> None:
    """`wr_collapse_consecutive_evals = 8` fires at exactly the 8th routed eval round. REDs on the
    clipped code, where `len(history)` caps at 5 forever and the abort never fires; the
    not-fired-through-7 half catches a drift that fires EARLY. The rolling knob is deliberately
    HIGHER, so the measured round is the collapse pair's."""
    cfg = _monitor_cfg(wr_hard_abort_enabled=True,
                       wr_collapse_consecutive_evals=8, wr_rolling_consecutive_evals=9,
                       wr_early_death_min_step=0, wr_collapse_min_step=0,
                       wr_rolling_min_step=0)
    h = _coordinator(monitor_cfg=cfg)
    for index in range(1, 8):
        _round(h, index, 0.01)
        assert h.shutdown.running is True, (
            f"fired at eval round {index} < consec 8: no trigger may fire before its own "
            "consec-th observation"
        )
    _round(h, 8, 0.01)
    assert h.shutdown.running is False, (
        "consec 8 did not fire at its 8th eval round: the WR ring is being clipped below the "
        "minted consec — ADJ-D38's unfireable-in-effect defect is back"
    )
    assert h.sink.named("hard_abort"), "a fired WR abort must announce itself in the stream"


def test_the_wr_ring_capacity_is_derived_from_BOTH_consec_keys_and_the_peak_window() -> None:
    """After 14 healthy rounds the ring holds EXACTLY `max(peak window, collapse, rolling)`,
    measured for three pairs giving three DIFFERENT answers — a DERIVATION pin, not a size pin.
    `(2, 3)` answers 5, `(8, 3)` answers 8, `(2, 11)` answers 11; any constant gives one answer
    for all three and no trim gives 14."""
    for collapse, rolling in ((2, 3), (8, 3), (2, 11)):
        cfg = _monitor_cfg(wr_collapse_consecutive_evals=collapse,
                           wr_rolling_consecutive_evals=rolling)
        h = _coordinator(monitor_cfg=cfg)
        for index in range(1, 15):
            _round(h, index, 0.5)          # healthy: above every threshold, nothing fires
        assert h.shutdown.running is True, "the healthy drive must not fire anything"
        assert len(h.coord._wr_history) == max(WR_PEAK_WINDOW_EVALS, collapse, rolling), (
            f"ring holds {len(h.coord._wr_history)} entries for collapse={collapse}, "
            f"rolling={rolling}: the capacity must BE the max of the two minted consec keys "
            "and rule B's own peak window, derived at the point of use — not a constant, and "
            "not one of the two keys alone"
        )


def test_a_round_carrying_no_WR_neither_appends_nor_trims_the_ring() -> None:
    """The skip path must not touch the ring: a round with no `wr_sealbot` appends nothing, and if
    it also TRIMMED, an above-depth consec would need a fresh unbroken run after every blackout.
    Driven ABOVE the old depth, where a wrong answer is observable."""
    cfg = _monitor_cfg(wr_collapse_consecutive_evals=9, wr_rolling_consecutive_evals=9)
    h = _coordinator(monitor_cfg=cfg)
    for index in range(1, 8):
        _round(h, index, 0.5)
    before = list(h.coord._wr_history)
    h.coord.on_eval_round_complete({"step": 8000})            # no `wr_sealbot` — a SKIP
    assert h.coord._wr_history == before, (
        "a round carrying no WR touched the ring: a skipped observation must neither append "
        "nor trim nor reset it"
    )
    assert h.sink.named("sealbot_wr_gate_skipped"), "the skip must be visible (LAW-18)"


def test_rule_B_takes_its_peak_over_its_OWN_window_never_over_the_whole_ring() -> None:
    """The ring depth was ALSO a semantic constant of rule B, so a derived capacity must not carry
    the peak window with it. The 8-eval history is chosen so the two answers DIFFER: the
    whole-ring peak (0.80) puts rule B's bar at 0.40 and the trailing 0.20s clear it, while the
    windowed peak (0.30) puts it at 0.15. Both quantities are DERIVED in the drive, so the pin
    fails only if the WINDOW moves, and the positive control fires from inside the window."""
    cfg = _monitor_cfg()                                   # ratio 0.5, consec 3, mins 25000
    history = [(23000, 0.80), (24000, 0.80), (25000, 0.80), (26000, 0.30),
               (27000, 0.30), (28000, 0.20), (29000, 0.20), (30000, 0.20)]
    assert len(history) > WR_PEAK_WINDOW_EVALS, "the pin needs a history DEEPER than the window"
    whole_peak = max(wr for _, wr in history)
    windowed_peak = max(wr for _, wr in history[-WR_PEAK_WINDOW_EVALS:])
    current = history[-1][1]
    assert current < whole_peak * cfg.wr_collapse_from_peak_ratio, (
        "premise: under a WHOLE-RING peak this history clears rule B's bar and the trigger "
        "fires — without that this test cannot tell the two windows apart"
    )
    assert current >= windowed_peak * cfg.wr_collapse_from_peak_ratio, (
        "premise: under the WINDOWED peak the same history does NOT clear the bar"
    )
    assert sealbot_wr_trajectory_alert(history, 30000, cfg) is None, (
        "rule B fired on a peak taken outside its own window: deriving the ring capacity "
        "must not widen the peak window with it — that is a behavioural change to an armed "
        "rule, which ADJ-D38 says is a ruling and not a rider on the capacity fix"
    )
    inside = [(26000, 0.80), (27000, 0.80), (28000, 0.20), (29000, 0.20), (30000, 0.20)]
    fired = sealbot_wr_trajectory_alert(inside, 30000, cfg)
    assert fired is not None and "trigger B" in fired, (
        "…and rule B must still fire when the peak IS inside its window, or the drive above "
        f"proves only that the rule is dead; got {fired!r}"
    )


#: Sequences exercising a fire and a non-fire on each trigger, a recovering dip, and a run
#: longer than the old ring, so replica and coordinator are compared across evictions.
_CORPUS: tuple[tuple[str, tuple[float, ...]], ...] = (
    ("healthy", (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)),
    ("early_death", (0.01, 0.01, 0.01, 0.01, 0.01, 0.01)),
    ("collapse_from_peak", (0.8, 0.8, 0.3, 0.12, 0.12, 0.12, 0.12)),
    ("recovering_dip", (0.3, 0.3, 0.02, 0.3, 0.3, 0.3)),
    ("rolling", (0.08, 0.09, 0.08, 0.5, 0.08, 0.08)),
    ("long_mixed", (0.6, 0.05, 0.4, 0.02, 0.35, 0.09, 0.3, 0.04, 0.5, 0.01, 0.2, 0.02)),
)


def _pre_d38_replica(sequence, cfg, eval_interval):
    """The PRE-D38 machine replayed. The shipped rule can stand in for the pre-D38 one HERE only,
    and the assert says why: with the ring clipped to five, the windowed peak and the old
    whole-ring peak are the same number by construction."""
    ring: list[tuple[int, float]] = []
    fired: list[bool] = []
    for index, wr in enumerate(sequence, start=1):
        step = index * eval_interval
        ring.append((step, float(wr)))
        del ring[:-_OLD_DEPTH]
        assert len(ring) <= WR_PEAK_WINDOW_EVALS
        fired.append(sealbot_wr_trajectory_alert(ring, step, cfg) is not None)
    return fired, list(ring)


def test_the_old_depth_and_rule_Bs_window_were_ONE_literal() -> None:
    """The premise the bit-identity claim rests on: the number rule B kept is the number the ring
    was clipped to."""
    assert WR_PEAK_WINDOW_EVALS == _OLD_DEPTH, (
        "rule B's peak window is no longer the deleted ring depth, so the pre-D38 replica "
        f"below is replaying a machine that never existed; got {WR_PEAK_WINDOW_EVALS}"
    )


def test_every_consec_at_or_below_the_old_depth_is_BIT_IDENTICAL_to_the_clipped_ring() -> None:
    """The behavioural-equivalence proof over the region the change was allowed to leave alone:
    both consec knobs from 0 through the old depth, six sequences, compared per round on ring
    CONTENTS and on whether the rule fired. `consec = 0` is included deliberately, since
    `history[-0:]` is the WHOLE ring in Python."""
    checked = 0
    for collapse in range(_OLD_DEPTH + 1):
        for rolling in range(_OLD_DEPTH + 1):
            cfg = _monitor_cfg(wr_collapse_consecutive_evals=collapse,
                               wr_rolling_consecutive_evals=rolling)
            for label, sequence in _CORPUS:
                h = _coordinator(monitor_cfg=cfg)
                live_fired: list[bool] = []
                for index, wr in enumerate(sequence, start=1):
                    warns_before = len(h.sink.named("sealbot_wr_warn"))
                    _round(h, index, wr)
                    live_fired.append(len(h.sink.named("sealbot_wr_warn")) > warns_before)
                want_fired, want_ring = _pre_d38_replica(
                    sequence, cfg, h.config.eval_interval)
                assert list(h.coord._wr_history) == want_ring, (
                    f"{label} at collapse={collapse}, rolling={rolling}: the ring diverged "
                    f"from the pre-D38 machine — {h.coord._wr_history} vs {want_ring}"
                )
                assert live_fired == want_fired, (
                    f"{label} at collapse={collapse}, rolling={rolling}: the trajectory "
                    f"DECISION diverged from the pre-D38 machine — {live_fired} vs "
                    f"{want_fired}. Below the old depth this change must move nothing"
                )
                checked += 1
    assert checked == (_OLD_DEPTH + 1) ** 2 * len(_CORPUS), (
        "the sweep must cover every consec pair at or below the old depth against every "
        f"sequence, or the equivalence claim is narrower than it reads; ran {checked}"
    )


def test_the_published_earliest_fire_round_is_deliverable_above_the_old_depth() -> None:
    """`Cadence.EVAL_ROUND_CONSEC`'s published number, matched against a REAL coordinator firing
    at a consec the old ring could never satisfy. The operands and the period are READ OFF the
    harness, never re-typed, because the audit's number and the machine must share one authority.
    On the clipped code the arithmetic answers 8 rounds while the machine never fires at all."""
    cfg = _monitor_cfg(wr_hard_abort_enabled=True,
                       wr_collapse_consecutive_evals=8, wr_rolling_consecutive_evals=9,
                       wr_early_death_min_step=0, wr_collapse_min_step=0,
                       wr_rolling_min_step=0)
    h = _coordinator(monitor_cfg=cfg)
    operands = (cfg.wr_collapse_consecutive_evals, cfg.wr_early_death_min_step,
                cfg.wr_collapse_min_step, cfg.wr_rolling_consecutive_evals,
                cfg.wr_rolling_min_step)
    period = h.config.eval_interval
    published_rounds = Cadence.EVAL_ROUND_CONSEC.earliest_fire_samples(
        operands, period_steps=period)
    published_step = Cadence.EVAL_ROUND_CONSEC.earliest_fire_step(
        operands, period_steps=period)
    assert published_rounds == 8.0 and published_step == float(8 * period), (
        f"the eval-round arithmetic answered {published_rounds!r} rounds / "
        f"{published_step!r} steps for {operands} at period {period}"
    )
    fired_at = None
    for index in range(1, 9):
        _round(h, index, 0.01)
        if not h.shutdown.running:
            fired_at = index
            break
    assert fired_at is not None and float(fired_at) == published_rounds, (
        f"the audit publishes an earliest fire of {published_rounds!r} eval round(s) but the "
        f"machine fired at round {fired_at!r}: the published number must be deliverable by "
        "the code that evaluates the row"
    )
    stamped = [event["step"] for event in h.sink.named("hard_abort")]
    assert stamped == [published_step], (
        f"…and it must land on the published TRAINING STEP, which is where an operator reads "
        f"it; got {stamped} against {published_step}"
    )
