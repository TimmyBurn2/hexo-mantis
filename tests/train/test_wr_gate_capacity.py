# >300 justify (R8): one ring, one claim, and the claim has two halves that only mean
# something together — the capacity must GROW with the minted consec keys and rule B's peak
# window must NOT grow with it. Split across two files, a reader could satisfy either half
# alone (that is exactly the naive D36 extension ADJ-D38 warns against), and the bit-identity
# sweep would sit on the other side of an import from the drives that establish what it is
# identical TO. The pre-D38 replica, the corpus it replays and the live-coordinator harness
# are one apparatus and belong in one unit.
"""R265 / ADJ-D38 oracle — the sealbot-WR ring's capacity is DERIVED (from the two minted
consec keys and from rule B's own peak window), never a literal, so NO schema-legal consec is
unfireable; and rule B's peak window does NOT widen with the deeper ring.

WHAT WENT WRONG. `step.py::on_eval_round_complete` trimmed the WR ring to a literal
`WR_HISTORY_DEPTH = 5` while all three triggers in `monitor/rules.py::
sealbot_wr_trajectory_alert` refuse on `len(history) >= their consec`: any schema-legal
`monitor.wr_collapse_consecutive_evals` or `monitor.wr_rolling_consecutive_evals` >= 6
(`ge=0`, no upper bound) was PERMANENTLY unfireable while `monitor.wr_hard_abort_enabled`
armed the abort — ADJ-D36's class, one gate over, on the axis LAW-15/F-30 names as the one
that actually kills runs. Worse on the audit side: the axis had no `ArmedAbort` row at all,
so gate 12 could not compute even a FALSE affirmative for it.

WHY THIS IS NOT ADJ-D36's FIX COPIED. The WR ring served TWO masters: the consec tails AND
`peak_wr`, which rule B took over the WHOLE ring. Deriving the capacity alone would have
widened rule B's peak window with it — a peak over more evals is a HIGHER bar for
`wr < peak * ratio`, i.e. a behavioural change to an armed rule that no ruling authorizes. So
the capacity DERIVES and the window is NAMED (`monitor/rules.py::WR_PEAK_WINDOW_EVALS`),
beside the predicate that reads it.

THE DRIVES BELOW STATE THE MUTATIONS THAT RED THEM:

* a resurrected trim literal (5, or a "generous" 64) — the capacity pin measures three
  DIFFERENT lengths for three consec pairs, so no constant satisfies it, and the
  above-depth fire drive reds for any literal below its consec;
* capacity keyed to only ONE of the two consec knobs — the capacity pin's third pair moves
  only the rolling knob;
* capacity keyed to the consecs but NOT floored by the peak window — the bit-identity sweep
  reds on every pair whose max consec is below 5 (the ring would shrink under rule B);
* a whole-ring `peak_wr` (the naive D36 extension) — the peak-window drive reds, because the
  8-eval history it uses fires trigger B under a whole-ring peak and must not under the
  windowed one;
* no trim at all — the capacity pin reds on the drive length;
* a trim on the SKIP path — the absent-WR drive reds;
* `>=` -> `>` on any trigger's length gate — the above-depth fire drive reds one eval late.

The coordinator config comes from the production builder (`mantis.run.
_step_coordinator_config`) and `on_eval_round_complete` is called DIRECTLY, so one call is
one routed eval round — the same posture `tests/train/test_drawrate_gate_capacity.py` takes
for its own gate, with the fakes duplicated locally per R5 (no cross-test import). R7 /
gate 6: nothing here writes a file.
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

#: The DELETED `WR_HISTORY_DEPTH`. A test INPUT and a historical fact, not an authority: the
#: fix must make every value above it fireable without any shipped code knowing the number.
#: It is written here as the old depth AND asserted equal to rule B's surviving window,
#: because those were the same literal and the window is the half that must not move.
_OLD_DEPTH = 5


# ── local fakes (R5: no cross-test import) ────────────────────────────────────────────
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


# ── the D36 class on the WR axis, closed: an above-the-old-depth consec FIRES ──────────
def test_a_wr_consec_above_the_old_depth_fires_at_the_consec_th_eval_round() -> None:
    """`wr_collapse_consecutive_evals = 8` fires at exactly the 8th routed eval round.

    REDs on the clipped code — `del ring[:-5]` caps `len(history)` at 5 forever, so
    `len(history) >= 8` is unsatisfiable and the abort NEVER fires however long the collapse
    runs. The not-fired-through-7 half pins the other direction: a widened tail or length-gate
    drift that fires EARLY is caught here too, not only the unfireable defect.

    `wr_rolling_consecutive_evals = 9` is deliberately HIGHER, so the first satisfiable
    trigger is the collapse pair's and the measured round is 8 rather than the rolling rule's.
    """
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
    """After 14 healthy rounds the ring holds EXACTLY `max(peak window, collapse, rolling)`
    — measured for three pairs that give three DIFFERENT answers.

    That is what makes this a DERIVATION pin rather than a size pin. `(2, 3)` answers 5 (the
    peak window floors it, which is the half that keeps rule B's window intact); `(8, 3)`
    answers 8 (the collapse knob); `(2, 11)` answers 11 (the rolling knob ALONE — a capacity
    keyed to only the collapse knob reds here and nowhere else). Any constant gives one
    answer for all three; no trim at all gives 14 for all three.
    """
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
    """The skip path must not touch the ring — the R92/BUG-1 contract on this axis.

    A routed round whose result has no `wr_sealbot` is skip-counted and appends nothing; if
    it also TRIMMED, a deep ring would be clipped by an evidence blackout and an
    above-the-old-depth consec would need a fresh unbroken run of observations after every
    absent WR. Driven ABOVE the old depth, where a wrong answer is observable at all.
    """
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


# ── the peak window: preserved EXACTLY, and it is the half D36's pattern would have broken ─
def test_rule_B_takes_its_peak_over_its_OWN_window_never_over_the_whole_ring() -> None:
    """The constraint ADJ-D38 names by hand: the ring depth was ALSO a semantic constant of
    rule B, so a derived capacity must not carry the peak window with it.

    The 8-eval history below is chosen so the two answers DIFFER, which is the only way to
    witness which one is live: the whole-ring peak (0.80) puts rule B's bar at 0.40 and the
    trailing 0.20s all clear it — trigger B WOULD fire — while the windowed peak (0.30 over
    the last five) puts the bar at 0.15, which they do not. The rule must return None.

    Both discriminating quantities are DERIVED from the history in the drive rather than
    transcribed, so the pin survives an edit to the numbers and fails only if the WINDOW
    moves. The positive control below it fires the same trigger on a history whose peak IS
    inside the window, so this is not merely "rule B never fires".
    """
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


# ── bit-identity for every consec at or below the old depth ───────────────────────────
#: Sequences chosen to exercise a fire and a non-fire on each trigger, plus a recovering dip
#: (the §175/L34 asymmetry) and a run longer than the old ring, so the replica and the live
#: coordinator are compared across evictions rather than only on short histories.
_CORPUS: tuple[tuple[str, tuple[float, ...]], ...] = (
    ("healthy", (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)),
    ("early_death", (0.01, 0.01, 0.01, 0.01, 0.01, 0.01)),
    ("collapse_from_peak", (0.8, 0.8, 0.3, 0.12, 0.12, 0.12, 0.12)),
    ("recovering_dip", (0.3, 0.3, 0.02, 0.3, 0.3, 0.3)),
    ("rolling", (0.08, 0.09, 0.08, 0.5, 0.08, 0.08)),
    ("long_mixed", (0.6, 0.05, 0.4, 0.02, 0.35, 0.09, 0.3, 0.04, 0.5, 0.01, 0.2, 0.02)),
)


def _pre_d38_replica(sequence, cfg, eval_interval):
    """The PRE-D38 machine, replayed: append, `del ring[:-5]`, then the trajectory rule.

    The shipped rule can stand in for the pre-D38 one HERE and only here, and the assert says
    why: with the ring clipped to five entries `history[-WR_PEAK_WINDOW_EVALS:]` is the whole
    history, so the windowed peak and the old whole-ring peak are the same number by
    construction. That is exactly the claim "bit-identical for every consec <= the old depth"
    reduces to, which is why this replica is honest rather than circular.
    """
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
    """The premise the whole bit-identity claim rests on, stated rather than assumed: the
    number rule B kept is the number the ring was clipped to. If they ever differ, "identical
    for consec <= the old depth" stops meaning what this file says it means."""
    assert WR_PEAK_WINDOW_EVALS == _OLD_DEPTH, (
        "rule B's peak window is no longer the deleted ring depth, so the pre-D38 replica "
        f"below is replaying a machine that never existed; got {WR_PEAK_WINDOW_EVALS}"
    )


def test_every_consec_at_or_below_the_old_depth_is_BIT_IDENTICAL_to_the_clipped_ring() -> None:
    """The behavioural-equivalence proof, driven over the whole legal region the change was
    allowed to leave alone: both consec knobs from 0 through the old depth, six sequences.

    Compared per round, not just at the end: the ring CONTENTS and whether the trajectory
    rule fired. Every committed config sits inside this region (collapse 3, rolling 2), so a
    single mismatch here is a shipped behavioural change — which this ruling does not grant.

    `consec = 0` is included deliberately. It does NOT disable a trigger — `history[-0:]` is
    the WHOLE ring in Python — so it arms a weaker-evidence variant, and the sweep pins that
    the variant behaves identically on both sides rather than quietly moving with the ring.
    """
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


# ── the audit tie: the published earliest fire is deliverable IN EVAL ROUNDS ───────────
def test_the_published_earliest_fire_round_is_deliverable_above_the_old_depth() -> None:
    """`Cadence.EVAL_ROUND_CONSEC`'s published number, matched against a REAL coordinator
    firing at a consec the old ring could never satisfy — R265's half of ADJ-D38, and the
    reason the two halves ship together.

    The operands and the period are READ OFF the harness, never re-typed: the whole point is
    that the audit's number and the machine share one authority, and a hand-copied operand
    would be that authority forked (the correction D36's own audit-tie drive took).

    On the clipped code this arithmetic answers 8 rounds while the machine never fires at all
    — an audit publishing a number the run structurally cannot deliver is precisely what R251
    exists to refuse, and it is what gate 12 would have started doing for this axis the
    moment the row was added without the ring fix.
    """
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
