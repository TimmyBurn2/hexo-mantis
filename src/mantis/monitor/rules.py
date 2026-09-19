"""Pure stateless run-safety rules plus the headless training-step alert emitter.

Ports the surviving part of the predecessor's alert rules with DECISION PARITY — the old code is
the spec. IN: the four training-step WARN rules and `check_draw_rate_collapse`. OUT: the
value-spread canary, which stayed green through a 33%→5% WR collapse, the strength/robustness
family, fed by a killed phantom input, and (R362(c)) the sealbot-WR trajectory instrument, whose
producer — the sealbot rung — is deleted.

Every function is stateless — the caller owns the history/window ring. The draw-rate
threshold lives in `StepCoordinatorConfig`, but this module must never import `mantis.train`, so
the coordinator passes the numbers in as explicit keyword arguments.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, MutableSequence, Sequence
from typing import Any

from mantis.monitor.config import MonitorConfig

# The rule-name tokens the emitted `training_alert` events carry (manifest + gate keys).
WARN_RULE_NAMES: tuple[str, ...] = (
    "entropy_collapse",
    "selfplay_entropy_collapse",
    "grad_norm_spike",
    "loss_increase_window",
    # A non-finite loss is excluded from the loss window BY DESIGN, since a NaN poisons every later
    # comparison, and that exclusion was silently also excluding it from every alert.
    "nonfinite_loss",
)

#: The payload key(s) each WARN rule's verdict is a function of. LAW-18: a rule whose INPUT is
#: absent DID NOT RUN, which is a different fact from one that ran and found nothing wrong, and the
#: two were one observable. Both entropy rules were the first for the whole life of the run until
#: R355(e) (B-4): the graph step now publishes `policy_entropy` and `policy_entropy_selfplay`.
WARN_RULE_INPUTS: dict[str, tuple[str, ...]] = {
    "entropy_collapse": ("policy_entropy",),
    "selfplay_entropy_collapse": ("selfplay_model_entropy_batch", "policy_entropy_selfplay"),
    "grad_norm_spike": ("grad_norm",),
    "loss_increase_window": (),
    "nonfinite_loss": ("loss_total",),
}

#: Per-rule count of the steps at which a rule could not run for want of its input. Read live as a
#: MODULE ATTRIBUTE by the coordinator's emit, never a from-imported int, so "this rule is quiet"
#: and "this rule has never been able to speak" are two numbers instead of one silence.
WARN_RULE_SKIPS: dict[str, int] = dict.fromkeys(WARN_RULE_NAMES, 0)


def rule_input_absent(name: str, payload: Mapping[str, Any]) -> bool:
    """True when NONE of `name`'s declared payload inputs carries a value this step. A rule with no
    declared input is never absent — it reads state the emitter owns, so nothing about the payload
    could stop it running."""
    keys = WARN_RULE_INPUTS.get(name, ())
    if not keys:
        return False
    return all(payload.get(key) is None for key in keys)


def check_entropy_collapse(payload: Mapping[str, Any], cfg: MonitorConfig) -> str | None:
    """Combined-stream entropy below ``alert_entropy_min`` (strictly below; absent = no fire)."""
    ent = payload.get("policy_entropy")
    if ent is not None and ent < float(cfg.alert_entropy_min):
        return f"policy entropy {ent:.2f} — possible mode collapse"
    return None


def check_selfplay_entropy_collapse(
    payload: Mapping[str, Any], cfg: MonitorConfig
) -> str | None:
    """Selfplay-stream entropy below ``collapse_threshold_nats``, preferring the canonical
    ``selfplay_model_entropy_batch`` over the legacy ``policy_entropy_selfplay``. Non-finite values
    are IGNORED: a NaN entropy is a missing measurement, not a collapse."""
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
    """Grad norm strictly above ``alert_grad_norm_max``, OR non-finite. The old `gn == gn` NaN
    filter was backwards: a NaN grad norm means `clip_and_step` has scaled by a NaN coefficient and
    every weight is now NaN, so the alert was quietest exactly when the model had just been
    destroyed. An ABSENT `grad_norm` is still no fire — absence is a missing reading, not a bad
    one."""
    gn = payload.get("grad_norm")
    if gn is None:
        return None
    if not math.isfinite(gn):
        return f"grad norm {gn} — NON-FINITE, weights are corrupt"
    if gn > float(cfg.alert_grad_norm_max):
        return f"grad norm {gn:.1f} — instability"
    return None


def check_nonfinite_loss(payload: Mapping[str, Any], cfg: MonitorConfig) -> str | None:
    """A non-finite ``loss_total`` fires. The loss window deliberately does NOT accept one, since
    appending it poisons every later comparison, but "not in the window" was silently becoming "not
    reported at all"."""
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
    """``alert_loss_increase_window`` consecutive STRICTLY increasing losses. ``window`` is the
    caller-owned tail of recent ``loss_total`` values, and a window of exactly ``n`` samples is too
    short to fire."""
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
    """Run the 4 WARN rules over one ``training_step`` payload and route each fired rule through the
    INJECTED sink as one ``training_alert`` event. Appends this step's ``loss_total`` to the
    caller-owned window first, then fires in the pinned `WARN_RULE_NAMES` order."""
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


def check_policy_loss_trough(
    history: Sequence[float],
    current_step: int,
    *,
    reference: float,
    delta_nats: float,
    consec: int,
    max_step: int,
) -> str | None:
    """R350(b)(iv): the last ``consec`` window means all sit ``delta_nats`` above ``reference``
    while ``current_step <= max_step``; silent past it (the signature is an EARLY overwrite)."""
    if delta_nats <= 0 or consec <= 0 or len(history) < consec:
        return None
    if current_step > max_step:
        return None
    tail = [float(v) for v in list(history)[-consec:]]
    bar = float(reference) + float(delta_nats)
    if all(value >= bar for value in tail):
        return (
            f"HARD-ABORT (policy-loss trough): policy loss {tail[-1]:.3f} >= {bar:.3f} "
            f"(reference window {float(reference):.3f} + {float(delta_nats):.2f} nats) for "
            f"{consec} consecutive windows inside the first {max_step:,} steps — the "
            "warm-started prior is being overwritten"
        )
    return None


def check_ply_cap_attractor(
    observed_rate: float,
    current_step: int,
    *,
    rate: float,
    window_games: int,
    min_step: int,
) -> str | None:
    """R352(c): the last ``window_games`` games' cap fraction STRICTLY above ``rate`` past ``min_step``."""
    if rate <= 0 or window_games <= 0 or current_step < min_step:
        return None
    if float(observed_rate) > float(rate):
        return (
            f"HARD-ABORT (ply-cap attractor): {float(observed_rate):.2f} of the last "
            f"{window_games} self-play games ended at the ply cap, above {float(rate):.2f} past "
            f"step {min_step:,} — self-play is drifting into cap draws (F-02/F-22/F-52)"
        )
    return None


def check_draw_rate_collapse(
    history: Sequence[float],
    current_step: int,
    *,
    threshold: float,
    consec: int,
    min_step: int,
) -> str | None:
    """Sustained self-play draw-rate collapse past ``min_step``.

    ``history`` carries ONLY real observations: an interval with fewer than ``N_pool_min`` completed
    games yields ``None`` at the producer and never enters the series, so ``consec`` counts
    consecutive OBSERVATIONS rather than gate runs over an unchanging reading. ``threshold <= 0`` is
    a LIBRARY guard, unreachable from production, where arming is a property of the resolved value.
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
