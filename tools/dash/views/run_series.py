"""Is training stable, is self-play healthy: small multiples of the reduced series, each saying which way is good."""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence

from ..readers.record import RunSnapshot
from .charts import HIGHER, LOWER, RUN_CLASSES, Goal, Key, figure, smoothed
from .fmt import esc, num, pct, short, sig
from .stats import wilson
from .svg import Band, Chart, Line, Ref

#: Plain names for the trainer's alert rules; an unknown rule is shown by its own name.
_ALERT = {"grad_norm_spike": "gradient-norm spike", "loss_increase_window": "loss rise"}
#: The rolling window the side and cap shares are read over, in games.
WINDOW = 2000
_TRAIN = (("value_loss", "Value loss", "Smoothed. Band: range per bucket.", LOWER, "trainer_step"),
          ("policy_loss", "Policy loss", "Against the search target.", LOWER, "trainer_step"),
          ("policy_entropy", "Policy entropy", "Nats, on training rows.", Goal("watch", "falls slowly, a cliff is collapse"),
           "trainer_step"),
          ("grad_norm", "Gradient norm", "Before clipping. ▲ marks a spike warning.", Goal("watch", "steady is good, spikes are warnings"),
           "trainer_step"),
          ("lr", "Learning rate, ×10⁻³", "", Goal("watch", "follows the schedule"), "trainer_step"),
          ("avg_game_length", "Game length", "Mean stones per self-play game.", Goal("watch", "no target"), "iteration_complete"))
_PLAY = (("games_per_hour", "Games per hour", "", HIGHER), ("positions_per_hour", "Positions per hour", "All workers.", HIGHER),
         ("steps_per_hour", "Trainer steps per hour", "", HIGHER))


def x_span(snaps: Sequence[RunSnapshot]) -> tuple[float, float]:
    """Every step-axis chart on a page runs from zero to the furthest run's last step, so the charts line up."""
    return 0.0, float(max((x.events.steps_max or 0) for x in snaps)) or 1.0


def _multiple(snaps: Sequence[RunSnapshot], event: str, key: str, title: str, definition: str, goal: Goal,
              fmt: Callable[[float], str] = sig, from_zero: bool = False, marks: Sequence[tuple[float, str]] = ()) -> str:
    lines: list[Line] = []
    bands: list[Band] = []
    reach = 0.0
    keys = []
    for i, x in enumerate(snaps):
        pairs = x.events.series(event, "step", key).pairs()
        keys.append(Key(x.label, RUN_CLASSES[i], off="" if pairs else "not recorded"))
        if not pairs:
            continue
        line, band, half = smoothed(x.label, RUN_CLASSES[i], pairs, band=len(snaps) == 1)
        lines.append(line)
        reach = max(reach, half)
        if band is not None:
            bands.append(band)
    if not lines:
        return figure(title, None, definition, goal=goal, gap=("Not recorded in this run.", f"No {event}.{key} rows."))
    ends = {ln.name: ln.pts[-1][1] for ln in lines if ln.pts}
    nows = [(fmt(ends[x.label]) if x.label in ends else "", RUN_CLASSES[i]) for i, x in enumerate(snaps)]
    y_domain = (0.0, max(y for ln in lines for _, y in ln.pts) * 1.2) if from_zero else None
    chart = Chart(title, lines=lines, bands=bands, reach=reach, y_fmt=fmt, y_domain=y_domain, x_domain=x_span(snaps), marks=marks)
    return figure(title, chart, definition, goal=goal, now=nows[0][0], nows=nows, keys=keys if len(snaps) > 1 else ())


def _rolling(snap: RunSnapshot, flag: Callable[[int], int | None]) -> list[tuple[float, float]]:
    """The share over the last `WINDOW` games at each game's step, thinned to about 160 points; a None flag is not counted."""
    g = snap.events.games
    window: deque[int] = deque()
    total = 0
    out: dict[float, float] = {}
    for i in range(g.count):
        value = flag(i)
        if value is None:
            continue
        window.append(value)
        total += value
        if len(window) > WINDOW:
            total -= window.popleft()
        if len(window) == WINDOW and g.step[i] >= 0:
            out[float(g.step[i])] = total / WINDOW
    pts = sorted(out.items())
    stride = max(1, len(pts) // 160)
    return pts[::stride] + ([pts[-1]] if pts and (len(pts) - 1) % stride else [])


def _share_chart(snaps: Sequence[RunSnapshot], title: str, definition: str, goal: Goal,
                 flag_of: Callable[[RunSnapshot], Callable[[int], int | None]], refs: Sequence[Ref] = ()) -> tuple[str, list[tuple[float, float]]]:
    series = [(x, _rolling(x, flag_of(x)), RUN_CLASSES[i]) for i, x in enumerate(snaps)]
    lines = [Line(x.label, cls, pts) for x, pts, cls in series if pts]
    first = series[0][1]
    if not lines:
        return figure(title, None, definition, goal=goal, gap=(f"Fewer than {num(WINDOW)} games so far.",
                                                               "The share needs a full window.")), first
    chart = Chart(title, lines=lines, refs=refs, y_fmt=lambda v: pct(v, 1), y_floor=0.0, x_domain=x_span(snaps))
    nows = [(pct(pts[-1][1], 1) if pts else "", cls) for _x, pts, cls in series]
    keys = [Key(x.label, cls, off="" if pts else "too few games") for x, pts, cls in series] if len(snaps) > 1 else []
    return figure(title, chart, definition, goal=goal, now=nows[0][0], nows=nows, keys=keys), first


def _last_smoothed(snap: RunSnapshot, event: str, key: str) -> float | None:
    """A series' last smoothed value, the one its chart ends on and its header shows."""
    pairs = snap.events.series(event, "step", key).pairs()
    return smoothed(key, "c1", pairs, band=False)[0].pts[-1][1] if pairs else None


def _ends(snap: RunSnapshot, key: str) -> tuple[float, float] | None:
    """A trainer series' first and last smoothed values, the ones its chart starts and ends on."""
    pairs = snap.events.series("trainer_step", "step", key).pairs()
    if len(pairs) < 2:
        return None
    line, _band, _half = smoothed(key, "c1", pairs, band=False)
    return (line.pts[0][1], line.pts[-1][1]) if line.pts else None


def training(snaps: Sequence[RunSnapshot]) -> tuple[str, str, str]:
    head = snaps[0].events
    aborts = [r for name in ("hard_abort", "hard_abort_after_stop") for r in head.rows(name)]
    alerts = head.rows("training_alert")
    words = f"{len(aborts)} abort{'s' if len(aborts) != 1 else ''}" if aborts else "No aborts"
    kinds: dict[str, int] = {}
    for a in alerts:
        kind = _ALERT.get(str(a.get("rule")), str(a.get("rule") or "warning").replace("_", " "))
        kinds[kind] = kinds.get(kind, 0) + 1
    if alerts:
        named = [short(a.get("step")) for a in alerts[:6]]
        steps = (", ".join(named[:-1]) + " and " + named[-1] if len(named) > 1 else named[0]) + (
            f" and {len(alerts) - 6} more" if len(alerts) > 6 else "")
        what = next(iter(kinds)) if len(kinds) == 1 else "warning"
        sentence = f"{words}. <strong>{len(alerts)} {esc(what)}{'s' if len(alerts) != 1 else ''}</strong>: {steps}."
    else:
        sentence = f"{words}, no warnings."
    notes = ", ".join(f"{esc(k)} ×{n}" if n > 1 else esc(k) for k, n in kinds.items()) if len(kinds) > 1 else ""
    trend = ""
    value, policy = _ends(snaps[0], "value_loss"), _ends(snaps[0], "policy_loss")
    if value is not None and policy is not None:
        trend = f"Smoothed, value loss {sig(value[0])} → {sig(value[1])} and policy loss {sig(policy[0])} → {sig(policy[1])}.<br>"
    aside = (f"Warnings: {notes}.<br>" if notes else "") + trend + "Losses track the fit to moving targets, not strength."
    for i, x in enumerate(snaps[1:], 1):
        theirs = x.events.rows("training_alert")
        stops = sum(len(x.events.rows(n)) for n in ("hard_abort", "hard_abort_after_stop"))
        said = f"{stops} abort{'s' if stops != 1 else ''}" if stops else "no aborts"
        said += f", {len(theirs)} warning{'s' if len(theirs) != 1 else ''}: " + ", ".join(short(a.get("step")) for a in theirs[:6]) if theirs else ", no warnings"
        aside += f'<br><span class="{RUN_CLASSES[i]}">{esc(x.label)}: {said}.</span>'
    spikes = [(float(a["step"]), f"gradient-norm spike at {num(a['step'])}") for a in alerts
              if a.get("rule") == "grad_norm_spike" and isinstance(a.get("step"), int)]
    panels = "".join(_multiple(snaps, event, key, title, definition, goal,
                               fmt=(lambda v: f"{v * 1e3:.2f}") if key == "lr" else sig, from_zero=key == "lr",
                               marks=spikes if key == "grad_norm" else ())
                     for key, title, definition, goal, event in _TRAIN)
    return sentence, aside, panels


def selfplay(snaps: Sequence[RunSnapshot]) -> tuple[str, str, str]:
    head = snaps[0].events
    first_html, first = _share_chart(
        snaps, "First player wins", f"Last {num(WINDOW)} decided games.", Goal("near", "best near 50 %"),
        lambda x: (lambda i: None if x.events.games.winner[i] not in (0, 1) else int(x.events.games.winner[i] == 0)),
        refs=(Ref(0.5),))
    cap_html, cap = _share_chart(snaps, "Games ending at the cap", f"Last {num(WINDOW)} games.", LOWER,
                                 lambda x: (lambda i: int(x.events.games.cap[i])))
    gph = _last_smoothed(snaps[0], "iteration_complete", "games_per_hour")
    parts = [f"<strong>{num(gph)} games an hour</strong>" if gph is not None else "No throughput row yet"]
    decided = [int(w == 0) for w in head.games.winner if w in (0, 1)][-WINDOW:]
    if len(decided) == WINDOW:
        lo, hi = wilson(sum(decided), WINDOW)
        share = sum(decided) / WINDOW
        slight = "slightly " if abs(share - 0.5) < 0.05 else ""
        parts.append("balanced sides" if lo <= 0.5 <= hi else f"the {'first' if lo > 0.5 else 'second'} player {slight}favoured")
    if cap:
        parts.append(f"{pct(cap[-1][1], 1)} of games at the cap")
    sentence = ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1] + "."
    aside = "" if first else f"The side and cap shares need {num(WINDOW)} games."
    for i, x in enumerate(snaps[1:], 1):
        other = _last_smoothed(x, "iteration_complete", "games_per_hour")
        if other is not None:
            aside = f'<span class="{RUN_CLASSES[i]}">{esc(x.label)}: {num(other)} games an hour.</span><br>{aside}'
    panels = first_html + cap_html + "".join(
        _multiple(snaps, "iteration_complete", key, title, definition, goal,
                  fmt=(lambda v: short(v)) if "positions" in key else (lambda v: num(v)))
        for key, title, definition, goal in _PLAY)
    return sentence, aside.removesuffix("<br>"), panels
