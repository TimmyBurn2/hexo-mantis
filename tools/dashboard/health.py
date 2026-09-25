"""The health badge: the worst of nine inputs, and an absent input is never green."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mantis.diagnostics.f816_37_rate_bar import (
    MAX_IN_WINDOW,
    Firing,
    scan_dumps,
    scan_logs,
    worst_window,
)

from .reader import Record

#: This many retained bundles without receipts is the mirror WARNING (never a halt).
MIRROR_LAG_WARN_BUNDLES = 2

_RANK = {"bad": 3, "warn": 2, "unmeasured": 1, "ok": 0}

#: The trough signature run7's prereg armed as a halt, DEMOTED to this warning: policy
#: loss >= its first reading + delta on `consec` consecutive `trainer_step` rows by `max_step`.
TROUGH_DELTA_NATS = 0.2
TROUGH_CONSEC = 3
TROUGH_MAX_STEP = 5000

#: The minted ply-cap halt terms, read when the record's `monitor_gates` rows carry none
#: (a run that did not arm the halt is still read for the attractor at these terms).
PLY_CAP_RATE = 0.5
PLY_CAP_WINDOW_GAMES = 600

_WATCHDOG_FIRES = ("heartbeat_watchdog_fired", "selfplay_stall_watchdog")
_HARD_ABORTS = ("hard_abort", "hard_abort_after_stop")


@dataclass(frozen=True)
class HealthInput:
    name: str
    state: str
    reason: str


@dataclass(frozen=True)
class HealthReading:
    state: str
    inputs: list[HealthInput]
    firings: list[Firing]


def assess(rec: Record) -> HealthReading:
    """Read the nine health inputs off the record and rank them into one badge."""
    inputs = [_gates(rec), _hard_aborts(rec), _watchdogs(rec)]
    fired, firings = _firings(rec)
    inputs += [fired, _mirror(rec), _alerts(rec), _disk(rec), _policy_loss_trough(rec),
               _ply_cap(rec)]
    state = max((i.state for i in inputs), key=lambda s: _RANK[s])
    return HealthReading(state=state, inputs=inputs, firings=firings)


def _gates(rec: Record) -> HealthInput:
    last = rec.last("monitor_gates")
    if last is None:
        return HealthInput("gate fires", "unmeasured", "no monitor_gates row in this record")
    gates: dict[str, Any] = last.get("gates") or {}
    fired = {n: s.get("fires") for n, s in gates.items()
             if isinstance(s, dict) and isinstance(s.get("fires"), int) and s["fires"] > 0}
    if fired:
        return HealthInput("gate fires", "bad", "fired: " + ", ".join(
            f"{n} ×{c}" for n, c in sorted(fired.items())) + f" (monitor_gates at step {last.get('step')})")
    return HealthInput("gate fires", "ok",
                       f"{len(gates)} gate(s), no fires at step {last.get('step')}")


def _hard_aborts(rec: Record) -> HealthInput:
    if not rec.rows("run_segment_started"):
        return HealthInput("hard aborts", "unmeasured",
                           "no run_segment_started row, so the lifecycle producer is unproven")
    aborts = [r for name in _HARD_ABORTS for r in rec.rows(name)]
    if aborts:
        first = aborts[0]
        return HealthInput("hard aborts", "bad",
                           f"{len(aborts)} hard abort(s); first {first.get('rule')} at step "
                           f"{first.get('step')}")
    return HealthInput("hard aborts", "ok", "none emitted by an armed lifecycle")


def _watchdogs(rec: Record) -> HealthInput:
    armed = [r for name in ("heartbeat_watchdog_armed", "selfplay_stall_watchdog_armed")
             for r in rec.rows(name) if r.get("enabled") is True]
    if not armed:
        return HealthInput("watchdogs", "unmeasured", "no *_watchdog_armed row with enabled=true")
    fires = {name: len(rec.rows(name)) for name in _WATCHDOG_FIRES if rec.rows(name)}
    if fires:
        return HealthInput("watchdogs", "bad", "fired: " + ", ".join(
            f"{n} ×{c}" for n, c in sorted(fires.items())))
    return HealthInput("watchdogs", "ok", f"{len(armed)} armed, none fired")


def _firings(rec: Record) -> tuple[HealthInput, list[Firing]]:
    name = "F-816-37 firings"
    if rec.record_dir is None:
        return HealthInput(name, "unmeasured",
                           "no --record-dir, so collate_dumps/ and the logs were not read"), []
    dumps, logs = scan_dumps(rec.record_dir), scan_logs(rec.record_dir)
    worst, _ = worst_window(dumps)
    if logs or worst > MAX_IN_WINDOW:
        return HealthInput(name, "bad", f"in-wire {len(dumps)}, out-of-wire {len(logs)}, worst 12 h "
                           f"window {worst} against a bar of {MAX_IN_WINDOW} — condemns under "
                           "R342(b)(iv)"), dumps + logs
    return HealthInput(name, "ok", f"{len(dumps)} in-wire, 0 out-of-wire, worst 12 h window "
                       f"{worst} of {MAX_IN_WINDOW} allowed"), dumps + logs


def _mirror(rec: Record) -> HealthInput:
    rows = [r for r in rec.rows("resume_state_persisted")
            if isinstance(r.get("unreceipted_bundles"), list)]
    if not rows:
        return HealthInput("mirror receipts", "unmeasured", "no resume_state_persisted row")
    lag = len(rows[-1]["unreceipted_bundles"])
    if lag >= MIRROR_LAG_WARN_BUNDLES:
        return HealthInput("mirror receipts", "warn",
                           f"{lag} retained bundle(s) without a receipt at step "
                           f"{rows[-1].get('step')} — the puller missed at least "
                           f"{MIRROR_LAG_WARN_BUNDLES} intervals (R349(b), not a halt)")
    return HealthInput("mirror receipts", "ok",
                       f"{lag} unreceipted at step {rows[-1].get('step')} (warning at "
                       f"{MIRROR_LAG_WARN_BUNDLES})")


def _alerts(rec: Record) -> HealthInput:
    alerts = rec.rows("training_alert")
    if alerts:
        rules = sorted({str(a.get("rule")) for a in alerts})
        return HealthInput("training alerts", "warn",
                           f"{len(alerts)} alert(s): {', '.join(rules)}")
    if rec.last("monitor_gates") is None:
        return HealthInput("training alerts", "unmeasured",
                           "no monitor_gates row, so the alert rules are unproven to have run")
    return HealthInput("training alerts", "ok", "none")


def _disk(rec: Record) -> HealthInput:
    free = rec.series("disk_free", "ts", "disk_free_gb")
    if not free:
        return HealthInput("disk", "unmeasured", "no disk_free row")
    alerts = rec.rows("disk_alert")
    if alerts:
        return HealthInput("disk", "warn", f"{len(alerts)} disk_alert row(s); last free "
                           f"{free[-1][1]:.1f} GB")
    return HealthInput("disk", "ok", f"last free {free[-1][1]:.1f} GB")


def _policy_loss_trough(rec: Record) -> HealthInput:
    series = rec.series("trainer_step", "step", "policy_loss")
    if len(series) < 2:
        return HealthInput("policy-loss trough", "unmeasured",
                           f"{len(series)} trainer_step.policy_loss row(s); the signature needs "
                           "a reference and at least one later row")
    reference = series[0][1]
    bar = reference + TROUGH_DELTA_NATS
    run = 0
    for step, loss in series[1:]:
        if step > TROUGH_MAX_STEP:
            break
        run = run + 1 if loss >= bar else 0
        if run >= TROUGH_CONSEC:
            return HealthInput("policy-loss trough", "warn",
                               f"policy loss {loss:.2f} >= {reference:.2f} + {TROUGH_DELTA_NATS} "
                               f"nats on {run} consecutive rows by step {step:.0f} — the trough "
                               "signature, a WARNING not a halt (R351(d): the block's policy "
                               "loss rose 0.6 nats while the net improved)")
    return HealthInput("policy-loss trough", "ok",
                       f"no {TROUGH_CONSEC}-row rise of {TROUGH_DELTA_NATS} nats over the first "
                       f"reading {reference:.2f} inside step {TROUGH_MAX_STEP}")


def ply_cap_terms(rec: Record) -> tuple[float, int, bool]:
    """`(rate, window_games, armed)`: the run's own terms off `monitor_gates`, else the minted ones."""
    for row in reversed(rec.rows("monitor_gates")):
        rate, window = row.get("ply_cap_abort_rate"), row.get("ply_cap_window_games")
        if isinstance(rate, (int, float)) and isinstance(window, int) and not isinstance(rate, bool) \
                and not isinstance(window, bool) and window > 0:
            return float(rate), int(window), True
    return PLY_CAP_RATE, PLY_CAP_WINDOW_GAMES, False


def ply_cap_windowed(games: list[dict[str, Any]], window: int) -> list[tuple[float, float]]:
    """`(game ordinal, cap fraction of the `window` games ending there)` for every full window."""
    flags = [1 if g.get("terminal_reason") == "ply_cap" else 0 for g in games]
    if window <= 0 or len(flags) < window:
        return []
    out: list[tuple[float, float]] = []
    running = sum(flags[:window])
    out.append((float(window), running / window))
    for i in range(window, len(flags)):
        running += flags[i] - flags[i - window]
        out.append((float(i + 1), running / window))
    return out


def _ply_cap(rec: Record) -> HealthInput:
    rate, window, armed = ply_cap_terms(rec)
    terms = (f"rate {rate:g} over {window} games" + ("" if armed else
             " (R352(c)'s minted terms — the halt was not armed in this run, so the record is read at them)"))
    series = ply_cap_windowed(rec.rows("game_complete"), window)
    if not series:
        return HealthInput("ply-cap attractor", "unmeasured",
                           f"{len(rec.rows('game_complete'))} game_complete row(s), fewer than the "
                           f"{window}-game window — no windowed cap rate exists yet ({terms})")
    peak_x, peak = max(series, key=lambda p: p[1])
    if peak > rate:
        return HealthInput("ply-cap attractor", "bad",
                           f"windowed cap rate peaked at {peak:.2f} at game {peak_x:.0f}, above "
                           f"{terms} — the ply-cap attractor (F-02/F-22/F-52), the signature "
                           "R352(c)'s halt fires on")
    return HealthInput("ply-cap attractor", "ok",
                       f"windowed cap rate peaked at {peak:.2f} (game {peak_x:.0f}), last "
                       f"{series[-1][1]:.2f}, never above {terms}")
