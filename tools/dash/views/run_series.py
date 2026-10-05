"""Is training stable, is self-play healthy: small multiples of the reduced series; losses shown, never scored."""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence

from ..readers.record import RunSnapshot
from .charts import RUN_CLASSES, Key, figure, smoothed
from .fmt import esc, num, pct, short, sig
from .stats import wilson
from .svg import Band, Chart, Line, Ref

#: The rolling window the side and cap shares are read over, in games.
WINDOW = 2000
_TRAIN = (("value_loss", "Value loss", "Training value loss, smoothed; the band is each bucket's range.", "trainer_step"),
          ("policy_loss", "Policy loss", "Cross-entropy against the search target.", "trainer_step"),
          ("policy_entropy", "Policy entropy", "Entropy of the net's policy on training rows, nats.", "trainer_step"),
          ("grad_norm", "Gradient norm", "Before clipping.", "trainer_step"),
          ("lr", "Learning rate, ×10⁻³", "The schedule as applied; the axis starts at zero.", "trainer_step"),
          ("avg_game_length", "Game length", "Mean stones per self-play game.", "iteration_complete"))
_PLAY = (("games_per_hour", "Games per hour", "Self-play throughput."),
         ("positions_per_hour", "Positions per hour", "Positions produced per hour across all workers."),
         ("steps_per_hour", "Trainer steps per hour", "Optimiser steps."))


def x_span(snaps: Sequence[RunSnapshot]) -> tuple[float, float]:
    """Every step-axis chart on a page runs from zero to the furthest run's last step, so the charts line up."""
    return 0.0, float(max((x.events.steps_max or 0) for x in snaps)) or 1.0


def _multiple(snaps: Sequence[RunSnapshot], event: str, key: str, title: str, definition: str,
              fmt: Callable[[float], str] = sig, from_zero: bool = False) -> str:
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
        return figure(title, None, definition, gap=("Not recorded in this run.", f"No {event}.{key} row in the record."))
    last = snaps[0].events.series(event, "step", key).last()
    y_domain = (0.0, max(y for ln in lines for _, y in ln.pts) * 1.2) if from_zero else None
    chart = Chart(title, lines=lines, bands=bands, reach=reach, y_fmt=fmt, y_domain=y_domain, x_domain=x_span(snaps))
    return figure(title, chart, definition, now=fmt(last) if last is not None else "", keys=keys if len(snaps) > 1 else ())


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


def _share_chart(snaps: Sequence[RunSnapshot], title: str, definition: str, flag_of: Callable[[RunSnapshot], Callable[[int], int | None]],
                 refs: Sequence[Ref] = ()) -> tuple[str, list[tuple[float, float]]]:
    series = [(x, _rolling(x, flag_of(x)), RUN_CLASSES[i]) for i, x in enumerate(snaps)]
    lines = [Line(x.label, cls, pts) for x, pts, cls in series if pts]
    first = series[0][1]
    if not lines:
        return figure(title, None, definition, gap=(f"Fewer than {num(WINDOW)} games in the record.",
                                                    "The share is read over a full window only.")), first
    chart = Chart(title, lines=lines, refs=refs, y_fmt=lambda v: pct(v, 1), y_floor=0.0, x_domain=x_span(snaps))
    return figure(title, chart, definition, now=pct(first[-1][1], 1) if first else ""), first


def training(snaps: Sequence[RunSnapshot]) -> tuple[str, str, str]:
    head = snaps[0].events
    aborts = [r for name in ("hard_abort", "hard_abort_after_stop") for r in head.rows(name)]
    alerts = head.rows("training_alert")
    words = f"{len(aborts)} abort{'s' if len(aborts) != 1 else ''} fired" if aborts else "No abort fired"
    if alerts:
        steps = ", ".join(num(a.get("step")) for a in alerts[:4]) + (" …" if len(alerts) > 4 else "")
        sentence = f"{words}; <strong>{len(alerts)} warning{'s' if len(alerts) != 1 else ''}</strong>, at {steps}."
    else:
        sentence = f"{words} and no warning."
    notes = "; ".join(esc(str(a.get("message") or a.get("rule"))) for a in alerts[:3])
    vl, pl = head.series("trainer_step", "step", "value_loss"), head.series("trainer_step", "step", "policy_loss")
    trend = ""
    if len(vl) > 1 and len(pl) > 1:
        trend = (f"Value loss {sig(vl.y[0])} → {sig(vl.last())}, policy loss {sig(pl.y[0])} → {sig(pl.last())}. ")
    aside = (notes + ". " if notes else "") + trend + "Losses are shown, never scored: lower is not stronger."
    panels = "".join(_multiple(snaps, event, key, title, definition,
                               fmt=(lambda v: f"{v * 1e3:.2f}") if key == "lr" else sig, from_zero=key == "lr")
                     for key, title, definition, event in _TRAIN)
    return sentence, aside, panels


def selfplay(snaps: Sequence[RunSnapshot]) -> tuple[str, str, str]:
    head = snaps[0].events
    first_html, first = _share_chart(
        snaps, "First mover wins", f"Share of the last {num(WINDOW)} decided games won by the side that placed the opening stone.",
        lambda x: (lambda i: None if x.events.games.winner[i] not in (0, 1) else int(x.events.games.winner[i] == 0)),
        refs=(Ref(0.5, label="even"),))
    cap_html, cap = _share_chart(snaps, "Games ending at the cap", f"Share of the last {num(WINDOW)} games that hit the ply cap.",
                                 lambda x: (lambda i: int(x.events.games.cap[i])))
    gph = head.series("iteration_complete", "step", "games_per_hour").last()
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
    aside = (f"First mover wins {pct(first[-1][1])} of the last {num(WINDOW)} decided games." if first
             else f"The side and cap shares need {num(WINDOW)} games.")
    panels = first_html + cap_html + "".join(
        _multiple(snaps, "iteration_complete", key, title, definition, fmt=(lambda v: short(v)) if "positions" in key else (lambda v: num(v)))
        for key, title, definition in _PLAY)
    return sentence, aside, panels
