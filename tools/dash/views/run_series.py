"""Is training stable, is self-play healthy: small multiples of the reduced series, each saying which way is good."""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence

from ..readers.record import RunSnapshot
from .charts import HIGHER, LOWER, RUN_CLASSES, Goal, Key, figure, smoothed
from .fmt import esc, kilo, num, pct, short, sig
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
    y_domain = (0.0, max(y for ln in lines for _, y in ln.pts) * 1.2 or 1.0) if from_zero else None
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


def _alerts_said(aborts: list[dict], alerts: list[dict], *, bold: bool) -> tuple[str, str]:
    """Aborts and warnings in a sentence (one kind is named, several are counted), and the kinds when several."""
    words = f"{len(aborts)} abort{'s' if len(aborts) != 1 else ''}" if aborts else "No aborts"
    kinds: dict[str, int] = {}
    for a in alerts:
        kind = _ALERT.get(str(a.get("rule")), str(a.get("rule") or "warning").replace("_", " "))
        kinds[kind] = kinds.get(kind, 0) + 1
    if not alerts:
        return f"{words if bold else words[0].lower() + words[1:]}, no warnings.", ""
    named = [short(a["step"]) for a in alerts if isinstance(a.get("step"), int)][:6]
    rest = len(alerts) - len(named)
    if rest:
        steps = ", ".join(named) + f" and {rest} more" if named else f"{rest} with no step"
    else:
        steps = ", ".join(named[:-1]) + " and " + named[-1] if len(named) > 1 else named[0]
    what = f"{len(alerts)} {next(iter(kinds)) if len(kinds) == 1 else 'warning'}{'s' if len(alerts) != 1 else ''}"
    notes = ", ".join(f"{esc(k)} ×{n}" if n > 1 else esc(k) for k, n in kinds.items()) if len(kinds) > 1 else ""
    if not bold:
        words = words[0].lower() + words[1:]
    return f"{words}. {f'<strong>{esc(what)}</strong>' if bold else esc(what)}: {steps}.", notes


def _ends(snap: RunSnapshot, event: str, key: str) -> tuple[float, float] | None:
    """A series' first and last smoothed values, the ones its chart starts and ends on and its header shows."""
    pairs = snap.events.series(event, "step", key).pairs()
    line = smoothed(key, "c1", pairs, band=False)[0] if pairs else None
    return (line.pts[0][1], line.pts[-1][1]) if line is not None and line.pts else None


def _last_smoothed(snap: RunSnapshot, event: str, key: str) -> float | None:
    ends = _ends(snap, event, key)
    return ends[1] if ends else None


def training(snaps: Sequence[RunSnapshot]) -> tuple[str, str, str]:
    head = snaps[0].events
    alerts = head.rows("training_alert")
    sentence, notes = _alerts_said(head.rows("hard_abort") + head.rows("hard_abort_after_stop"), alerts, bold=True)
    trend = ""
    value, policy = _ends(snaps[0], "trainer_step", "value_loss"), _ends(snaps[0], "trainer_step", "policy_loss")
    if value is not None and policy is not None:
        trend = f"Smoothed, value loss {sig(value[0])} → {sig(value[1])} and policy loss {sig(policy[0])} → {sig(policy[1])}.<br>"
    aside = (f"Warnings: {notes}.<br>" if notes else "") + trend + "Losses track the fit to moving targets, not strength."
    for i, x in enumerate(snaps[1:], 1):
        theirs, _ = _alerts_said(x.events.rows("hard_abort") + x.events.rows("hard_abort_after_stop"),
                                 x.events.rows("training_alert"), bold=False)
        aside += f'<br><span class="{RUN_CLASSES[i]}">{esc(x.label)}: {theirs}</span>'
    spikes = [(float(a["step"]), f"{snaps[0].label}: gradient-norm spike at {num(a['step'])}") for a in alerts
              if a.get("rule") == "grad_norm_spike" and isinstance(a.get("step"), int)]
    panels = "".join(_multiple(snaps, event, key, title, definition, goal,
                               fmt=(lambda v: f"{v * 1e3:.2f}") if key == "lr" else sig, from_zero=key == "lr",
                               marks=spikes if key == "grad_norm" else ())
                     for key, title, definition, goal, event in _TRAIN)
    return sentence, aside, panels


def _play_said(snap: RunSnapshot, *, bold: bool) -> str:
    """One run's self-play in a sentence: throughput, which side the interval favours, the share at the cap."""
    gph = _last_smoothed(snap, "iteration_complete", "games_per_hour")
    rate = f"{num(gph)} games an hour" if gph is not None else "No throughput row yet"
    parts = [f"<strong>{rate}</strong>" if bold else rate]
    decided = [int(w == 0) for w in snap.events.games.winner if w in (0, 1)][-WINDOW:]
    if len(decided) == WINDOW:
        lo, hi = wilson(sum(decided), WINDOW)
        slight = "slightly " if abs(sum(decided) / WINDOW - 0.5) < 0.05 else ""
        parts.append("balanced sides" if lo <= 0.5 <= hi else f"the {'first' if lo > 0.5 else 'second'} player {slight}favoured")
    cap = _rolling(snap, lambda i: int(snap.events.games.cap[i]))
    if cap:
        parts.append(f"{pct(cap[-1][1], 1)} of games at the cap")
    said = ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1] + "."
    return said


def selfplay(snaps: Sequence[RunSnapshot]) -> tuple[str, str, str]:
    first_html, first = _share_chart(
        snaps, "First player wins", f"Last {num(WINDOW)} decided games.", Goal("near", "best near 50 %"),
        lambda x: (lambda i: None if x.events.games.winner[i] not in (0, 1) else int(x.events.games.winner[i] == 0)),
        refs=(Ref(0.5),))
    cap_html, cap = _share_chart(snaps, "Games ending at the cap", f"Last {num(WINDOW)} games.", LOWER,
                                 lambda x: (lambda i: int(x.events.games.cap[i])))
    sentence = _play_said(snaps[0], bold=True)
    aside = "" if first else f"The side and cap shares need {num(WINDOW)} games."
    for i, x in enumerate(snaps[1:], 1):
        aside += f'<br><span class="{RUN_CLASSES[i]}">{esc(x.label)}: {_play_said(x, bold=False)}</span>'
    panels = first_html + cap_html + "".join(
        _multiple(snaps, "iteration_complete", key, title, definition, goal,
                  fmt=(lambda v: kilo(v)) if "positions" in key else (lambda v: num(v)))
        for key, title, definition, goal in _PLAY)
    return sentence, aside.removeprefix("<br>"), panels
