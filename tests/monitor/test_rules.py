"""The pure stateless rule functions: decision-parity plus the run-safety hard-aborts on
LIVE-shaped inputs.

An ORACLE-FIRST file — the top-level `import mantis.monitor.rules` raises before any port code
exists. Decision-parity is asserted against the old-side semantics, where the code IS the spec:
the warn rules at their registered boundaries, including the non-finite grad-norm pin and the 3-window
strictly-increasing rule, and the headless emitter routing one `training_alert` per fired rule
through the injected sink in rule order; and draw-rate collapse over `pooled_draw_rate` history.

`check_draw_rate_collapse` takes explicit `threshold`/`consec`/`min_step` kwargs rather than a
`StepCoordinatorConfig`, so `monitor/**` keeps zero `train` import; the coordinator passes the
threshold in.
"""
from __future__ import annotations

import math

import pytest

# LIVE producer for the draw-rate rule's input (torch-free stdlib module).
from mantis.train.coordinator.config import pooled_draw_rate

from mantis.monitor.config import MonitorConfig
from mantis.monitor.rules import (
    check_draw_rate_collapse,
    check_entropy_collapse,
    check_grad_norm_spike,
    check_loss_increase_window,
    check_selfplay_entropy_collapse,
    emit_training_step_alerts,
)


def test_entropy_collapse_boundary() -> None:
    """O-21 — combined-stream entropy fires strictly BELOW alert_entropy_min (1.0)."""
    cfg = MonitorConfig()
    assert check_entropy_collapse({"policy_entropy": 0.99}, cfg) is not None
    assert check_entropy_collapse({"policy_entropy": 1.0}, cfg) is None   # not < 1.0
    assert check_entropy_collapse({"policy_entropy": 1.01}, cfg) is None
    assert check_entropy_collapse({}, cfg) is None                        # absent → no fire


def test_selfplay_entropy_collapse_boundary_and_nonfinite_guard() -> None:
    """O-21 — selfplay entropy fires below collapse_threshold_nats (1.5); NaN/inf are ignored (isfinite guard)."""
    cfg = MonitorConfig()
    assert check_selfplay_entropy_collapse({"policy_entropy_selfplay": 1.49}, cfg) is not None
    assert check_selfplay_entropy_collapse({"policy_entropy_selfplay": 1.5}, cfg) is None
    assert check_selfplay_entropy_collapse({"policy_entropy_selfplay": float("nan")}, cfg) is None
    assert check_selfplay_entropy_collapse({"policy_entropy_selfplay": float("inf")}, cfg) is None


def test_grad_norm_spike_boundary_and_nonfinite_fires() -> None:
    """O-21 — fires strictly ABOVE alert_grad_norm_max (10.0), and on any NON-FINITE norm."""
    cfg = MonitorConfig()
    assert check_grad_norm_spike({"grad_norm": 10.01}, cfg) is not None
    assert check_grad_norm_spike({"grad_norm": 10.0}, cfg) is None      # not > 10.0
    assert check_grad_norm_spike({"grad_norm": float("nan")}, cfg) is not None
    assert check_grad_norm_spike({"grad_norm": float("inf")}, cfg) is not None
    assert check_grad_norm_spike({}, cfg) is None, "absence is not a fire"
    assert not (float("nan") > 10.0), (
        "sanity: NaN comparisons are False — which is WHY the old `gn > max` test could "
        "never fire on a NaN and an explicit isfinite check is required"
    )


def test_loss_increase_window_strictly_increasing() -> None:
    """O-21 — fires only when the last (window+1) losses are all strictly increasing; a window of exactly `n` (3) samples is too short to fire."""
    cfg = MonitorConfig()  # alert_loss_increase_window == 3
    assert check_loss_increase_window([1.0, 2.0, 3.0], cfg) is None          # len == n
    assert check_loss_increase_window([1.0, 2.0, 3.0, 4.0], cfg) is not None  # 4 strictly up
    assert check_loss_increase_window([1.0, 2.0, 2.0, 4.0], cfg) is None      # a plateau breaks it
    assert check_loss_increase_window([4.0, 3.0, 2.0, 1.0], cfg) is None      # decreasing


def test_headless_emitter_routes_training_alert_events_in_rule_order() -> None:
    """O-21 — the headless emitter fires the 4 warn rules and routes ONE `training_alert` event per fired rule through the INJECTED sink (structlog is dead), rule order preserved, and returns the fired messages."""
    cfg = MonitorConfig()
    sink = _RecordingSink()
    loss_window: list[float] = [1.0, 2.0, 3.0]  # caller-owned deque tail
    payload = {
        "event": "training_step", "step": 500,
        "policy_entropy": 0.5,                       # < 1.0 → entropy_collapse
        "policy_entropy_selfplay": 1.0,              # < 1.5 → selfplay_entropy_collapse
        "grad_norm": 25.0,                           # > 10.0 → grad_norm_spike
        "loss_total": 4.0,                           # window → 1,2,3,4 strictly up
    }
    fired = emit_training_step_alerts(payload, cfg, loss_window, sink=sink)
    assert len(fired) == 4, f"all four warn rules should fire, got {fired}"
    alert_events = [e for e in sink.events if e.get("event") == "training_alert"]
    assert len(alert_events) == 4, "one training_alert event per fired rule, through the sink"
    rules_in_order = [e["rule"] for e in alert_events]
    assert rules_in_order == [
        "entropy_collapse", "selfplay_entropy_collapse", "grad_norm_spike", "loss_increase_window",
    ], f"rule order must be preserved, got {rules_in_order}"


def test_headless_emitter_nonfinite_grad_norm_fires_through_the_sink() -> None:
    """Item 6 — a NaN grad_norm must reach the event stream as a grad_norm_spike alert."""
    cfg = MonitorConfig()
    sink = _RecordingSink()
    payload = {"event": "training_step", "step": 1, "grad_norm": float("nan"),
               "policy_entropy": 5.0}
    fired = emit_training_step_alerts(payload, cfg, [], sink=sink)
    assert any("grad" in m for m in fired), f"no grad alert fired; got {fired}"
    assert any(e.get("rule") == "grad_norm_spike"
               for e in sink.events if e.get("event") == "training_alert")


def test_a_nonfinite_loss_fires_its_own_rule_and_stays_out_of_the_window() -> None:
    """Item 6 — the two halves together, and they are in tension by design."""
    cfg = MonitorConfig()
    sink = _RecordingSink()
    window: list[float] = []
    payload = {"event": "training_step", "step": 1, "loss_total": float("nan"),
               "policy_entropy": 5.0}
    fired = emit_training_step_alerts(payload, cfg, window, sink=sink)
    assert window == [], "a non-finite loss must not poison the loss window"
    assert any(e.get("rule") == "nonfinite_loss"
               for e in sink.events if e.get("event") == "training_alert"), (
        f"the excluded value was not reported; fired={fired}"
    )


def test_a_finite_loss_does_not_fire_the_nonfinite_rule() -> None:
    """Mutation self-test: the rule must be silent on healthy training, or it reports nothing."""
    cfg = MonitorConfig()
    sink = _RecordingSink()
    window: list[float] = []
    emit_training_step_alerts(
        {"event": "training_step", "step": 1, "loss_total": 1.5, "policy_entropy": 5.0},
        cfg, window, sink=sink)
    assert window == [1.5], "a finite loss must still enter the window"
    assert not any(e.get("rule") == "nonfinite_loss"
                   for e in sink.events if e.get("event") == "training_alert")


# The stride5-spam rule was REMOVED at close-out: it is a dead artifact of bad hyperparams that
# never occurs under current recipes. The selfplay-owned `current_stride5_p90()` producer stays.


def test_pooled_draw_rate_below_the_bar_is_no_observation() -> None:
    """O-03, RE-POINTED by WPMINT Phase DS (R92)."""
    assert pooled_draw_rate((0, 0), N_pool_min=50) is None
    assert pooled_draw_rate((3, 3), N_pool_min=50) is None, (
        "three drawn games is not evidence: a 1.0 here would be a total-collapse abort on "
        "evidence the operator declared insufficient"
    )
    assert pooled_draw_rate((0, 50), N_pool_min=50) == 0.0, (
        "a MEASURED zero over sufficient evidence is still a real healthy reading"
    )
    assert pooled_draw_rate((25, 50), N_pool_min=50) == pytest.approx(0.5)


def test_draw_rate_collapse_fires_on_sustained_high_rate_past_min_step() -> None:
    """O-03 / P-03 — fires iff the last `consec` samples are all >= threshold AND current_step >= min_step."""
    history = [pooled_draw_rate((45, 100), N_pool_min=50) for _ in range(3)]  # 0.45 each
    assert check_draw_rate_collapse(history, 50000, threshold=0.4, consec=3, min_step=20000) is not None


def test_draw_rate_collapse_below_min_step_no_fire() -> None:
    """O-03 — before min_step the gate is silent even on a high sustained draw rate."""
    history = [0.9, 0.9, 0.9]
    assert check_draw_rate_collapse(history, 10000, threshold=0.4, consec=3, min_step=20000) is None


def test_draw_rate_collapse_below_consec_no_fire() -> None:
    """O-03 — a single high sample among low ones is not sustained collapse."""
    assert check_draw_rate_collapse([0.1, 0.1, 0.9], 50000,
                                    threshold=0.4, consec=3, min_step=0) is None


def test_draw_rate_collapse_threshold_nonpositive_disables() -> None:
    """O-03 — threshold <= 0 disables the gate (ships OFF at WP13-A landing, §f R9)."""
    assert check_draw_rate_collapse([0.9, 0.9, 0.9], 50000,
                                    threshold=0.0, consec=3, min_step=0) is None


def test_draw_rate_collapse_measured_healthy_zero_never_fires() -> None:
    """O-03 — a history of MEASURED 0.0 never fires even when configured."""
    history = [pooled_draw_rate((0, 50), N_pool_min=50) for _ in range(5)]
    assert all(x == 0.0 for x in history)
    assert check_draw_rate_collapse(history, 50000, threshold=0.4, consec=3, min_step=0) is None


class _RecordingSink:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, event) -> None:
        self.events.append(dict(event))


def test_math_import_available_for_finite_guards() -> None:
    """Sanity that the finite guards the rules rely on behave as asserted (documents intent)."""
    assert not math.isfinite(float("nan"))
    assert not math.isfinite(float("inf"))
