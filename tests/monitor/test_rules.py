"""⊕ O-03/O-04/O-05 + O-21 — the pure stateless rule functions (decision-parity + the
three run-safety hard-aborts on LIVE-shaped inputs).

RED-at-import until IMPL writes `mantis.monitor.rules` + `mantis.monitor.config`. This is
an ORACLE-FIRST (⊕) file: the top-level `import mantis.monitor.rules` raises
ModuleNotFoundError before any port code exists.

Decision-parity is asserted against the OLD-side semantics in
`hexo_rl/monitoring/alert_rules.py` (the code IS the spec):
  * O-05 sealbot-WR triggers A/B/C with the "N consecutive, not a single dip" guards
    (old `config.py` values verbatim: 0.10/2/20000; peak×0.5/3/25000; 0.05/3/15000);
  * O-21 warn rules at the registered boundaries (1.0 / 1.5 nats, 10.0 gn incl. the NaN
    `gn == gn` pin, 3-window strictly-increasing), and the headless emitter routing one
    `training_alert` event per fired rule through the injected sink, in rule order;
  * O-03 draw-rate collapse over `recent_pool_draw_rate` history (LIVE producer — the NaN
    `draw_target_fraction` phantom is never keyed on).

(O-04 stride5-spam was REMOVED at close-out per operator directive B.)

`check_draw_rate_collapse` takes explicit `threshold`/`consec`/`min_step` kwargs (NOT a
`StepCoordinatorConfig` object) so `monitor/**` keeps zero `train` import (DAG §2). The
draw-rate threshold lives in `StepCoordinatorConfig` (WP10); the coordinator passes it in
(see ORACLE_NOTES: IMPL API constraint).
"""
from __future__ import annotations

import math

import pytest

# LIVE producer for the draw-rate rule's input (torch-free stdlib module).
from mantis.train.coordinator.config import recent_pool_draw_rate

from mantis.monitor.config import MonitorConfig
from mantis.monitor.rules import (
    check_draw_rate_collapse,
    check_entropy_collapse,
    check_grad_norm_spike,
    check_loss_increase_window,
    check_sealbot_wr_hard_abort,
    check_selfplay_entropy_collapse,
    emit_training_step_alerts,
    sealbot_wr_trajectory_alert,
)


# ══ O-05 sealbot-WR decision parity ═══════════════════════════════════════════════════
# Each row: (label, history[(step,wr)], current_step, expect_fire). Values chosen against
# the old alert_rules.check_sealbot_wr_hard_abort semantics with the default MonitorConfig.
_SEALBOT_BATTERY: list[tuple[str, list[tuple[int, float]], int, bool]] = [
    ("empty_history_no_fire", [], 30000, False),
    ("healthy_high_wr_no_fire",
     [(30000, 0.5), (31000, 0.5), (32000, 0.5)], 32000, False),
    # Trigger C — early death: last 3 all < 0.05, past step 15000.
    ("trigger_C_early_death_fires",
     [(16000, 0.04), (17000, 0.03), (18000, 0.02)], 18000, True),
    # Trigger B — collapse from peak: last 3 all < peak(0.30)×0.5=0.15, past step 25000.
    ("trigger_B_collapse_from_peak_fires",
     [(26000, 0.30), (27000, 0.12), (28000, 0.12), (29000, 0.12)], 29000, True),
    # Trigger A — rolling: last 2 both < 0.10, past step 20000, len==2 (< the 3 that B/C need).
    ("trigger_A_rolling_fires",
     [(21000, 0.08), (22000, 0.09)], 22000, True),
    # Recovering single dip — one 2% eval after strong evals must NOT fire (§175/L34).
    ("single_recovering_dip_no_fire",
     [(30000, 0.30), (31000, 0.30), (32000, 0.02)], 32000, False),
    # B transient: only one low eval below peak×0.5 → not 3 consecutive → no fire.
    ("B_transient_single_low_no_fire",
     [(26000, 0.30), (27000, 0.30), (28000, 0.12)], 28000, False),
    # A below min_step: last 2 low but current_step <= 20000 → no fire.
    ("A_below_min_step_no_fire",
     [(19000, 0.08), (19500, 0.09)], 19500, False),
    # A below consecutive: only one sample → len < wr_rolling_consecutive_evals(2) → no fire.
    ("A_below_consec_no_fire",
     [(21000, 0.08)], 21000, False),
]


@pytest.mark.parametrize("label,history,step,expect_fire",
                         _SEALBOT_BATTERY, ids=[r[0] for r in _SEALBOT_BATTERY])
def test_sealbot_wr_decision_parity(label, history, step, expect_fire) -> None:
    """O-05 / P-05 — 100% decision match with the old-side triggers A/B/C incl. the
    consecutive-N guards. A message (truthy) = fire; None = no fire (the de-diagnosis of the
    MESSAGE text happens at the instrument, §0 — parity here is the DECISION only).

    The DECISION is a property of the trajectory, INDEPENDENT of the abort disposition, so
    it is asserted on `sealbot_wr_trajectory_alert` (which ignores the flag). The
    default-posture flip (warn-only) is a coordinator/disposition concern, tested at the
    hard-abort wrapper below and in test_coordinator_gates."""
    cfg = MonitorConfig()
    msg = sealbot_wr_trajectory_alert(list(history), step, cfg)
    assert (msg is not None) is expect_fire, f"{label}: fire={msg is not None}, want {expect_fire}"


def test_hard_abort_disposition_requires_the_enabled_flag() -> None:
    """O-05 / operator G-3 — the DEFAULT `MonitorConfig()` ships `wr_hard_abort_enabled=False`
    (warn-only), so `check_sealbot_wr_hard_abort` returns None on a collapse; the SAME
    trajectory yields a message from `sealbot_wr_trajectory_alert`, and setting the flag True
    restores the hard-abort disposition (the A/B/C capability is unchanged — only the default
    disposition moved)."""
    collapse = [(16000, 0.01), (17000, 0.01), (18000, 0.01)]
    default_cfg = MonitorConfig()
    assert default_cfg.wr_hard_abort_enabled is False, "shipped default is warn-only"
    assert check_sealbot_wr_hard_abort(collapse, 18000, default_cfg) is None
    # The trajectory FACT is still produced (the warn path has content — never silent).
    fact = sealbot_wr_trajectory_alert(collapse, 18000, default_cfg)
    assert fact is not None and "Objective-A" in fact and "Objective-B" in fact
    # Flipping the one field restores the hard-abort message.
    hard = check_sealbot_wr_hard_abort(collapse, 18000, MonitorConfig(wr_hard_abort_enabled=True))
    assert hard is not None and "HARD-ABORT" in hard and "Objective-A" in hard


# ══ O-21 warn-rule decision parity ════════════════════════════════════════════════════
def test_entropy_collapse_boundary() -> None:
    """O-21 — combined-stream entropy fires strictly BELOW alert_entropy_min (1.0)."""
    cfg = MonitorConfig()
    assert check_entropy_collapse({"policy_entropy": 0.99}, cfg) is not None
    assert check_entropy_collapse({"policy_entropy": 1.0}, cfg) is None   # not < 1.0
    assert check_entropy_collapse({"policy_entropy": 1.01}, cfg) is None
    assert check_entropy_collapse({}, cfg) is None                        # absent → no fire


def test_selfplay_entropy_collapse_boundary_and_nonfinite_guard() -> None:
    """O-21 — selfplay entropy fires below collapse_threshold_nats (1.5); NaN/inf are ignored
    (isfinite guard); the canonical field wins over the legacy fallback."""
    cfg = MonitorConfig()
    assert check_selfplay_entropy_collapse({"selfplay_model_entropy_batch": 1.49}, cfg) is not None
    assert check_selfplay_entropy_collapse({"selfplay_model_entropy_batch": 1.5}, cfg) is None
    assert check_selfplay_entropy_collapse({"selfplay_model_entropy_batch": float("nan")}, cfg) is None
    assert check_selfplay_entropy_collapse({"selfplay_model_entropy_batch": float("inf")}, cfg) is None
    # legacy fallback field
    assert check_selfplay_entropy_collapse({"policy_entropy_selfplay": 1.0}, cfg) is not None


def test_grad_norm_spike_boundary_and_nan_ignored() -> None:
    """O-21 — grad-norm fires strictly ABOVE alert_grad_norm_max (10.0); a NaN grad_norm is
    IGNORED (the `gn == gn` pin — a NaN must never trip the instability abort)."""
    cfg = MonitorConfig()
    assert check_grad_norm_spike({"grad_norm": 10.01}, cfg) is not None
    assert check_grad_norm_spike({"grad_norm": 10.0}, cfg) is None      # not > 10.0
    assert check_grad_norm_spike({"grad_norm": float("nan")}, cfg) is None
    assert not (float("nan") > 10.0), "sanity: NaN comparisons are False, so the guard is real"


def test_loss_increase_window_strictly_increasing() -> None:
    """O-21 — fires only when the last (window+1) losses are all strictly increasing; a window
    of exactly `n` (3) samples is too short to fire."""
    cfg = MonitorConfig()  # alert_loss_increase_window == 3
    assert check_loss_increase_window([1.0, 2.0, 3.0], cfg) is None          # len == n
    assert check_loss_increase_window([1.0, 2.0, 3.0, 4.0], cfg) is not None  # 4 strictly up
    assert check_loss_increase_window([1.0, 2.0, 2.0, 4.0], cfg) is None      # a plateau breaks it
    assert check_loss_increase_window([4.0, 3.0, 2.0, 1.0], cfg) is None      # decreasing


def test_headless_emitter_routes_training_alert_events_in_rule_order() -> None:
    """O-21 — the headless emitter fires the 4 warn rules and routes ONE `training_alert`
    event per fired rule through the INJECTED sink (structlog is dead), rule order preserved,
    and returns the fired messages. Bites an alert path with no live sink producer (LAW-07)."""
    cfg = MonitorConfig()
    sink = _RecordingSink()
    loss_window: list[float] = [1.0, 2.0, 3.0]  # caller-owned deque tail
    payload = {
        "event": "training_step", "step": 500,
        "policy_entropy": 0.5,                       # < 1.0 → entropy_collapse
        "selfplay_model_entropy_batch": 1.0,         # < 1.5 → selfplay_entropy_collapse
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


def test_headless_emitter_nan_grad_norm_does_not_fire() -> None:
    """O-21 — a NaN grad_norm must not produce a grad_norm_spike alert through the emitter."""
    cfg = MonitorConfig()
    sink = _RecordingSink()
    payload = {"event": "training_step", "step": 1, "grad_norm": float("nan"),
               "policy_entropy": 5.0}
    fired = emit_training_step_alerts(payload, cfg, [], sink=sink)
    assert all("grad" not in m for m in fired)
    assert not any(e.get("rule") == "grad_norm_spike"
                   for e in sink.events if e.get("event") == "training_alert")


# ══ (O-04 stride5-spam rule REMOVED at close-out, operator directive B — stride5-spam is a
#     dead artifact of bad hyperparams that never occurs under current recipes. The
#     selfplay-owned `WorkerPool.current_stride5_p90()` producer is unrelated and stays.) ══


# ══ O-03 draw-rate collapse rule (on the LIVE producer) ═══════════════════════════════
def test_recent_pool_draw_rate_empty_map_is_zero() -> None:
    """O-03 — the LIVE producer returns 0.0 for an empty worker map, so the gate can never
    fire on empty signal (never the NaN `draw_target_fraction` phantom)."""
    assert recent_pool_draw_rate({}) == 0.0
    assert recent_pool_draw_rate({0: 0.4, 1: 0.6}) == pytest.approx(0.5)


def test_draw_rate_collapse_fires_on_sustained_high_rate_past_min_step() -> None:
    """O-03 / P-03 — fires iff the last `consec` samples are all >= threshold AND
    current_step >= min_step. Empty/zero history never fires."""
    history = [recent_pool_draw_rate({0: 0.4, 1: 0.5}) for _ in range(3)]  # ~0.45 each
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


def test_draw_rate_collapse_empty_signal_never_fires() -> None:
    """O-03 — a history of 0.0 (no worker has a game yet) never fires even when configured."""
    history = [recent_pool_draw_rate({}) for _ in range(5)]
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
