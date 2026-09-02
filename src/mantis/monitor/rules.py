# >300 justify (R8): one decision surface. Every function here is a PURE predicate over one
# `training_step`-shaped payload, and they are read together — the WARN rules fire as an
# ordered tuple through a single emitter (`WARN_RULE_NAMES` / `emit_training_step_alerts`),
# and the hard-abort rules are the same predicates at abort severity. Splitting the WARN set
# from the emitter, or the warns from the aborts, would separate the rule ORDER from the code
# that depends on it and put a rule's threshold in a different file from the one place that
# reads it. The decision-parity citations to `hexo_rl/monitoring/alert_rules.py` are
# load-bearing and must sit beside the predicate each one certifies.
"""Pure stateless run-safety rules + the headless training-step alert emitter (§c.5/§c.6).

Ports the surviving ~55% of `hexo_rl/monitoring/alert_rules.py` with DECISION PARITY (the
old code is the spec) and drops the rest:

* IN — the 4 training-step WARN rules, the sealbot-WR trajectory instrument
  (`sealbot_wr_trajectory_alert` + the `check_sealbot_wr_hard_abort` disposition wrapper —
  triggers A/B/C with the "N consecutive, not a single dip" guards, §175/L34), plus
  `check_draw_rate_collapse` (the WP10-deferred sustained-draw gate).
* OUT — `check_value_spread_canary` (F-27 IS its falsification: it stayed green through a
  33%→5% WR collapse) and the whole `check_strength_*`/`check_robustness_*`/
  `check_objective_a_coverage` family (fed by the KILLED phantom `strength_aggregate`).
* REMOVED at close-out (operator directive B): `check_stride5_spam` — stride5-spam is a dead
  artifact of bad hyperparams that never occurs under current recipes. Its selfplay-owned
  producer `WorkerPool.current_stride5_p90()` and the `stride5_run_p90` telemetry field are
  UNRELATED to the gate and stay; only the run-safety GATE is gone.

Every function is stateless — the caller owns the history/window ring. The sealbot message
is DE-DIAGNOSED (§0): it reports the trajectory fact and names BOTH candidate mechanisms,
never asserting the strength regression the old text asserted (that misread misdirected six
investigations; the STOP decision was still right, the diagnosis was not).

DAG note: the draw-rate threshold lives in `StepCoordinatorConfig` (WP10), but this module
must never import `mantis.train` — the coordinator passes the numbers in as explicit keyword
arguments.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, MutableSequence, Sequence
from typing import Any

from mantis.monitor.config import MonitorConfig

# The instrument-level fix for the §D-FOUNDING misread: the fired message states the
# trajectory FACT and hands diagnosis back to the operator (§0).
_DEDIAGNOSIS = (
    "trajectory fact only — the cause is EITHER off-distribution exploitability "
    "(Objective-A) OR strength regression (Objective-B); diagnosis is the operator's"
)

#: Rule B's PEAK WINDOW, in eval rounds — how many most-recent evals `sealbot_wr_trajectory_
#: alert` takes its `peak_wr` over (R265 / ADJ-D38).
#:
#: THIS NUMBER IS NOT NEW AND ITS MEANING HAS NOT MOVED. It was the sealbot-WR ring's depth
#: (`step.py::WR_HISTORY_DEPTH = 5`, old `step_coordinator.py`'s `pop(0)` past 5), and rule B
#: read its peak over the whole ring — so "peak over the last five evals" is what the shipped
#: instrument has always decided on. What ADJ-D38 found is that ONE literal was carrying TWO
#: independent jobs: the ring's CAPACITY (how much evidence the trajectory rules can see) and
#: rule B's peak WINDOW (how far back "peak" reaches). Deriving the capacity from the minted
#: consec keys — which is what makes every schema-legal consec fireable — would have silently
#: widened the peak window with it, and a peak taken over more evals is a HIGHER bar for
#: `wr < peak * ratio`: a behavioural change to an armed rule that no ruling authorizes.
#:
#: So the capacity is DERIVED and the window is NAMED, here, beside the predicate that reads
#: it. Rule B's decision is bit-identical to the pre-D38 instrument for every history the old
#: ring could hold, and stays bit-identical on a deeper ring because the window does not move
#: with the ring. Driven by `tests/train/test_wr_gate_capacity.py`.
#:
#: NOT a config key and NOT a code-side default (R1): R1 forbids a default DUPLICATING a
#: schema field's authority, and no schema field has ever expressed this quantity — it is the
#: semantic of a rule, the way `check_loss_increase_window`'s `<= n` boundary is. Making it a
#: key would MINT a new armed value, which is precisely what this change may not do.
WR_PEAK_WINDOW_EVALS: int = 5

# The rule-name tokens the emitted `training_alert` events carry (manifest + gate keys).
WARN_RULE_NAMES: tuple[str, ...] = (
    "entropy_collapse",
    "selfplay_entropy_collapse",
    "grad_norm_spike",
    "loss_increase_window",
    # Item 6. A non-finite loss is excluded from the loss window BY DESIGN (a NaN poisons
    # every later comparison), and that exclusion was silently also excluding it from every
    # alert. This rule is what makes the excluded value visible.
    "nonfinite_loss",
)

#: The payload key(s) each WARN rule's verdict is a function of. AUDIT-1 F-29, and LAW-18:
#: a rule whose INPUT is absent DID NOT RUN, which is a different fact from a rule that ran
#: and found nothing wrong — and the two were one observable (`None`, no event, no counter).
#: `check_selfplay_entropy_collapse` has been the first for the whole life of the run:
#: `selfplay_model_entropy_batch` has no producer anywhere in `src/`, and its fallback
#: `policy_entropy_selfplay` has none either (it travelled as `NaN`, now as `None` — F-01).
#: So the rule has never once been able to fire, and its silence read as health.
#: `loss_increase_window` reads the CALLER's window rather than the payload, so it has no
#: payload input and is never counted absent.
WARN_RULE_INPUTS: dict[str, tuple[str, ...]] = {
    "entropy_collapse": ("policy_entropy",),
    "selfplay_entropy_collapse": ("selfplay_model_entropy_batch", "policy_entropy_selfplay"),
    "grad_norm_spike": ("grad_norm",),
    "loss_increase_window": (),
    "nonfinite_loss": ("loss_total",),
}

#: Per-rule count of the steps at which a rule could not run for want of its input. Read live
#: as a MODULE ATTRIBUTE by the coordinator's `monitor_gates` emit (the counter-binding rule:
#: never a from-imported int), so "this rule is quiet" and "this rule has never been able to
#: speak" are two numbers instead of one silence.
WARN_RULE_SKIPS: dict[str, int] = dict.fromkeys(WARN_RULE_NAMES, 0)


def rule_input_absent(name: str, payload: Mapping[str, Any]) -> bool:
    """True when NONE of `name`'s declared payload inputs carries a value this step.

    A rule with no declared input (`loss_increase_window`) is never absent — it reads state
    the emitter owns, so there is nothing about the payload that could stop it running.
    """
    keys = WARN_RULE_INPUTS.get(name, ())
    if not keys:
        return False
    return all(payload.get(key) is None for key in keys)


# ── the 4 training-step WARN rules (decision-parity ports) ────────────────────────────
def check_entropy_collapse(payload: Mapping[str, Any], cfg: MonitorConfig) -> str | None:
    """Combined-stream entropy below ``alert_entropy_min`` (strictly below; absent = no fire)."""
    ent = payload.get("policy_entropy")
    if ent is not None and ent < float(cfg.alert_entropy_min):
        return f"policy entropy {ent:.2f} — possible mode collapse"
    return None


def check_selfplay_entropy_collapse(
    payload: Mapping[str, Any], cfg: MonitorConfig
) -> str | None:
    """Selfplay-stream entropy below ``collapse_threshold_nats``.

    Prefers the canonical ``selfplay_model_entropy_batch``; falls back to the legacy
    ``policy_entropy_selfplay``. Non-finite values are IGNORED (the `isfinite` guard) —
    a NaN entropy is a missing measurement, not a collapse.
    """
    ent_sp = payload.get(
        "selfplay_model_entropy_batch", payload.get("policy_entropy_selfplay")
    )
    if (
        ent_sp is not None
        and isinstance(ent_sp, (int, float))
        and math.isfinite(ent_sp)
        and ent_sp < float(cfg.collapse_threshold_nats)
    ):
        return f"selfplay entropy {ent_sp:.2f} — selfplay mode collapse"
    return None


def check_grad_norm_spike(payload: Mapping[str, Any], cfg: MonitorConfig) -> str | None:
    """Grad norm strictly above ``alert_grad_norm_max``, OR non-finite.

    NON-FINITE FIRES (item 6). This rule used to carry a `gn == gn` NaN filter, described as
    "a NaN must never trip the instability alert" — old-side parity, preserved verbatim. It
    is backwards. A NaN grad norm is not a missing reading, it is the most severe instability
    there is: `clip_and_step` has scaled by a NaN coefficient and every weight is now NaN
    (falsified row F-11's cascade). Under the old filter that step alerted on NOTHING, while
    a merely large-but-finite norm alerted — so the alert was quietest exactly when the model
    had just been destroyed. `inf` fires for the same reason. An ABSENT `grad_norm` is still
    no fire: absence is a missing reading, which is a different thing from a bad one.
    """
    gn = payload.get("grad_norm")
    if gn is None:
        return None
    if not math.isfinite(gn):
        return f"grad norm {gn} — NON-FINITE, weights are corrupt"
    if gn > float(cfg.alert_grad_norm_max):
        return f"grad norm {gn:.1f} — instability"
    return None


def check_nonfinite_loss(payload: Mapping[str, Any], cfg: MonitorConfig) -> str | None:
    """A non-finite ``loss_total`` fires (item 6).

    The loss window deliberately does NOT accept a non-finite value — appending one poisons
    every later comparison in `check_loss_increase_window`, since NaN comparisons are False.
    But "not in the window" was silently becoming "not reported at all": a run whose loss
    went NaN dropped out of the loss-increase rule and triggered nothing else, so the trace
    showed a healthy-looking flat window while training was already dead. Keeping the value
    out of the window and firing a rule ON it is what separates those two.
    """
    del cfg  # threshold-free: non-finite is not a matter of degree
    loss = payload.get("loss_total")
    if loss is None or isinstance(loss, bool) or not isinstance(loss, (int, float)):
        return None
    if math.isfinite(loss):
        return None
    return f"loss_total {loss} — NON-FINITE, training step produced no usable gradient"


def check_loss_increase_window(
    window: Sequence[float], cfg: MonitorConfig
) -> str | None:
    """``alert_loss_increase_window`` consecutive STRICTLY increasing losses.

    ``window`` is the caller-owned tail of recent ``loss_total`` values; a window of
    exactly ``n`` samples is too short to fire (parity: ``len(window) <= n``).
    """
    n = int(cfg.alert_loss_increase_window)
    if len(window) <= n:
        return None
    tail = list(window)[-n - 1:]
    if all(tail[i] < tail[i + 1] for i in range(len(tail) - 1)):
        return f"loss increased {n} consecutive steps"
    return None


def emit_training_step_alerts(
    payload: Mapping[str, Any],
    cfg: MonitorConfig,
    loss_window: MutableSequence[float],
    *,
    sink: Any,
) -> list[str]:
    """Run the 4 WARN rules over one ``training_step`` payload and route each fired rule
    through the INJECTED sink as one ``training_alert`` event (structlog is dead — the
    event stream is the ONE channel).

    Appends this step's ``loss_total`` to the caller-owned ``loss_window`` first (so the
    window rule sees the current step), then fires in the pinned `WARN_RULE_NAMES` order.
    Returns the fired messages.
    """
    loss = payload.get("loss_total")
    if isinstance(loss, (int, float)) and not isinstance(loss, bool) and math.isfinite(loss):
        loss_window.append(float(loss))
    step = payload.get("step")
    results = (
        check_entropy_collapse(payload, cfg),
        check_selfplay_entropy_collapse(payload, cfg),
        check_grad_norm_spike(payload, cfg),
        check_loss_increase_window(loss_window, cfg),
        check_nonfinite_loss(payload, cfg),
    )
    fired: list[str] = []
    for name, message in zip(WARN_RULE_NAMES, results, strict=True):
        # AUDIT-1 F-29: count the steps a rule could not run at, BEFORE deciding it is quiet.
        if rule_input_absent(name, payload):
            WARN_RULE_SKIPS[name] += 1
        if message is None:
            continue
        fired.append(message)
        sink.emit(
            {"event": "training_alert", "rule": name, "message": message, "step": step}
        )
    return fired


# ── hard-abort rules ─────────────────────────────────────────────────────────────────
def sealbot_wr_trajectory_alert(
    wr_history: Sequence[tuple[int, float]],
    current_step: int,
    cfg: MonitorConfig,
) -> str | None:
    """The sustained sealbot-WR collapse FACT (triggers A/B/C, de-diagnosed) — computed
    REGARDLESS of the abort disposition.

    F-27 mandates exactly this instrument (a WR trajectory gate) after the value-spread
    canary held green through a 33%→5% collapse. Triggers (any fires):
      C. WR below ``wr_early_death_threshold`` for ``wr_collapse_consecutive_evals``
         consecutive evals past ``wr_early_death_min_step``;
      B. WR below ``peak × wr_collapse_from_peak_ratio`` for the same consecutive count
         past ``wr_collapse_min_step``, where ``peak`` is the highest WR of the last
         ``WR_PEAK_WINDOW_EVALS`` evals — rule B's own window, named rather than inherited
         from the caller's ring depth since R265/ADJ-D38 (the ring is now sized by the
         minted consec keys, and the peak must not widen with it);
      A. WR below ``wr_rolling_threshold`` for ``wr_rolling_consecutive_evals``
         consecutive evals past ``wr_rolling_min_step``.
    The consecutive-N guards are the §175/L34 asymmetry: a single self-correcting dip once
    aborted a RECOVERING run, and a missed abort is the cheaper error.

    Stateless — the caller owns the (step, wr) ring. Returns a DE-DIAGNOSED message (§0)
    naming the trigger and BOTH candidate mechanisms, or None. The DISPOSITION — hard-abort
    vs warn-only — is the CALLER's: `check_sealbot_wr_hard_abort` gates it on
    ``wr_hard_abort_enabled`` (default False = warn-only, operator G-3); the coordinator's
    warn path emits this fact verbatim on a `sealbot_wr_warn` event. The DECISION (which
    trigger, fire/no-fire) is unchanged from the old-side instrument — only the default
    disposition moved.
    """
    if not wr_history:
        return None

    history = list(wr_history)
    current_wr = history[-1][1]
    # R265 / ADJ-D38 — rule B's peak is taken over its own WINDOW, not over whatever the
    # caller's ring happens to hold. Until D38 those were the same thing by accident: the
    # ring was clipped to a literal 5 and this read the whole ring. The ring's capacity now
    # DERIVES from the minted consec keys so no schema-legal consec is unfireable, and a
    # whole-ring peak would have silently widened this window with it — a higher bar for
    # `wr < peak * ratio`, i.e. a behavioural change to an armed rule that no ruling
    # authorizes. The slice is a no-op for every history the old ring could hold, which is
    # what makes the fix bit-identical for every consec <= the old depth.
    peak_wr = max(wr for _, wr in history[-WR_PEAK_WINDOW_EVALS:])
    n_consec_collapse = int(cfg.wr_collapse_consecutive_evals)

    if (
        current_step > cfg.wr_early_death_min_step
        and len(history) >= n_consec_collapse
        and all(wr < cfg.wr_early_death_threshold for _, wr in history[-n_consec_collapse:])
    ):
        return (
            f"sealbot-WR trigger C (early death): WR {current_wr:.1%} < "
            f"{cfg.wr_early_death_threshold:.0%} for {n_consec_collapse} consecutive evals "
            f"past step {cfg.wr_early_death_min_step:,} — {_DEDIAGNOSIS}"
        )

    if (
        current_step > cfg.wr_collapse_min_step
        and peak_wr > 0.0
        and len(history) >= n_consec_collapse
        and all(
            wr < peak_wr * cfg.wr_collapse_from_peak_ratio
            for _, wr in history[-n_consec_collapse:]
        )
    ):
        return (
            f"sealbot-WR trigger B (collapse from peak): WR {current_wr:.1%} < "
            f"peak {peak_wr:.1%} × {cfg.wr_collapse_from_peak_ratio:.0%} for "
            f"{n_consec_collapse} consecutive evals past step {cfg.wr_collapse_min_step:,} "
            f"— {_DEDIAGNOSIS}"
        )

    n_consec = int(cfg.wr_rolling_consecutive_evals)
    if current_step > cfg.wr_rolling_min_step and len(history) >= n_consec:
        tail = history[-n_consec:]
        if all(wr < cfg.wr_rolling_threshold for _, wr in tail):
            mean_wr = sum(wr for _, wr in tail) / len(tail)
            return (
                f"sealbot-WR trigger A (rolling): mean WR {mean_wr:.1%} < "
                f"{cfg.wr_rolling_threshold:.0%} for {n_consec} consecutive evals past "
                f"step {cfg.wr_rolling_min_step:,} — {_DEDIAGNOSIS}"
            )

    return None


def check_sealbot_wr_hard_abort(
    wr_history: Sequence[tuple[int, float]],
    current_step: int,
    cfg: MonitorConfig,
) -> str | None:
    """The HARD-ABORT disposition of the sealbot-WR trajectory — gated on
    ``wr_hard_abort_enabled``.

    Default posture is warn-only (`wr_hard_abort_enabled=False`, operator G-3): with the
    flag False this returns None (no hard abort) and the coordinator emits a `sealbot_wr_warn`
    instead — the trajectory instrument is never silent, it just does not stop the run.
    Setting the flag True restores the A/B/C hard-abort exactly (the triggers are unchanged;
    only the default disposition moved). Enforcement (``shutdown.running=False``) is the
    caller's.
    """
    if not cfg.wr_hard_abort_enabled:
        return None
    alert = sealbot_wr_trajectory_alert(wr_history, current_step, cfg)
    return f"HARD-ABORT ({alert})" if alert is not None else None


def check_draw_rate_collapse(
    history: Sequence[float],
    current_step: int,
    *,
    threshold: float,
    consec: int,
    min_step: int,
) -> str | None:
    """Sustained self-play draw-rate collapse past ``min_step``.

    ``history`` is the caller-owned series of ``pooled_draw_rate(pool.pooled_draw_counts(),
    N_pool_min=…)`` samples — the LIVE producer, and by R92 (WPMINT Phase DS) it carries ONLY
    real observations: an interval with less than ``N_pool_min`` completed games yields
    ``None`` at the producer, is skip-counted, and never enters this series. So ``consec``
    counts consecutive OBSERVATIONS here, not consecutive gate runs over an unchanging
    reading. The NaN draw-target phantom
    input at `pool_push.py:135` is NEVER keyed on here (O-15 grep-ban, which bans even the
    token). ``threshold <= 0`` is a LIBRARY guard with its own tests; it is unreachable
    from production since WPAX Phase D, where arming became a property of the resolved
    `train.draw_rate_abort` value (`None` = off) and the caller stopped passing a
    disabling number (R65/R79/R80).
    """
    if threshold <= 0 or consec <= 0 or len(history) < consec:
        return None
    if current_step < min_step:
        return None
    tail = [float(v) for v in list(history)[-consec:]]
    if all(value >= threshold for value in tail):
        return (
            f"HARD-ABORT (draw-rate collapse): pool draw rate {tail[-1]:.2f} >= "
            f"{threshold:.2f} for {consec} consecutive checks past step {min_step:,} "
            f"— self-play has collapsed into drawn games"
        )
    return None
