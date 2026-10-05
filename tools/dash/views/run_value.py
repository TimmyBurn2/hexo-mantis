"""Is the value head learning: the monitor's instrument at every save rules; the knowledge horizon is the search's value beside it."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ..readers.horizon import K_MAX, REACH_SHARE, Curve, Horizon
from ..readers.record import RunSnapshot
from ..readers.saves import PLY_BANDS, Records, Save
from .charts import HIGHER, LOWER, RUN_CLASSES, Goal, Key, figure, table
from .fmt import esc, num, pct, short, signed
from .run_series import x_span
from .svg import Chart, Line, Ref

_BAND_NAMES = {"plies_0_10": "0–10", "plies_11_40": "11–40", "plies_41_up": "41+"}


def _series(saves: Sequence[Save], attr: str) -> list[tuple[float, float]]:
    return [(float(s.step), v) for s in saves if (v := getattr(s, attr)) is not None]


def gaps(saves: Sequence[Save]) -> list[tuple[float, float]]:
    """The train/held-out gap at the save it belongs to: each record's lagged read is of the save before it."""
    return sorted({float(s.lagged.step): s.lagged.gap_cf_ce for s in saves
                   if s.lagged is not None and s.lagged.gap_cf_ce is not None}.items())


def _halt(rec: Records) -> str | None:
    halt = rec.halt
    if halt is None:
        return None
    rows = "; ".join(esc(r) for r in halt.get("halting_rows") or [])
    raw_signal = halt.get("signal")
    signal: dict[str, Any] = raw_signal if isinstance(raw_signal, dict) else {}
    sent = signal.get("sent")
    how = ("The monitor signalled the run to save and stop." if halt.get("armed") is True and sent is True
           else f"The monitor was armed but sent no signal ({esc(signal.get('reason') or 'no reason given')})."
           if halt.get("armed") is True and sent is False
           else "The monitor was not armed to signal." if halt.get("armed") is False
           else "It was the run's final save." if halt.get("final_save") else "No signal is on record.")
    return f"<strong>Halt at {num(halt.get('step'))}</strong>: {rows}. {how}"


def _verdict(rec: Records) -> tuple[str, str]:
    """The sentence from the latest save: a halt first, then the reading; the lagged read's interval gates its word."""
    last, first = rec.saves[-1], rec.saves[0]
    halted = _halt(rec)
    if last.cf_ce is None:
        lead = f"The {short(last.step)} save has no held-out reading ({esc(last.gen_note or 'not measured')})."
    else:
        lead = f"Held-out cross-entropy <strong>{last.cf_ce:.3f}</strong> at {short(last.step)}"
        lead += f" ({first.cf_ce:.3f} at {short(first.step)})." if first is not last and first.cf_ce is not None else "."
    exams = last.exams
    if not exams or all(e.holds is None for e in exams.values()):
        exam_line = "Exams not measured at this save."
    elif all(e.holds for e in exams.values()):
        exam_line = "All exams above their floors."
    else:
        missed = ", ".join(esc(k) for k, e in exams.items() if e.holds is False)
        exam_line = f"<strong>{missed} below the floor</strong>" + (" (armed: the next miss halts)." if last.armed_floors else ".")
    aside = ""
    lag = next((s.lagged for s in reversed(rec.saves) if s.lagged is not None and s.lagged.ci is not None), None)
    if lag is not None and lag.ci is not None and lag.diff is not None:
        lo, hi = lag.ci
        if hi < 0:
            word = f"beats its lagged net by {-lag.diff:.3f} nats on the games after its save (95 % interval {-hi:.3f} to {-lo:.3f})"
        elif lo > 0:
            word = f"trails its lagged net by {lag.diff:.3f} nats on the games after its save (95 % interval {lo:.3f} to {hi:.3f})"
        else:
            word = (f"shows no clear difference from its lagged net on the games after its save (cross-entropy "
                    f"{signed(lag.diff, 3)} nats, 95 % interval {signed(lo, 3)} to {signed(hi, 3)})")
        aside = f"{short(lag.step)} {word}.<br>"
    series = gaps(rec.saves)
    line = next((s.gap_line for s in reversed(rec.saves) if s.gap_line is not None), None)
    if series and line is not None:
        aside += f"Train/held-out gap {signed(series[-1][1], 3)} at {short(series[-1][0])} (rule: two saves above {line:.2f}).<br>"
    if rec.gap_rule is not None:
        over = " and ".join(short(s) for s in rec.gap_rule.get("over") or [])
        lead += f" <strong>Gap rule fired at {short(rec.gap_rule.get('step'))}.</strong>"
        aside = f"{over} above the gap line: a learning-rate re\u2011mint is due.<br>{exam_line}<br>" + aside
    else:
        lead += f" {exam_line}"
    return (halted + " " + lead if halted else lead), aside.removesuffix("<br>")


def _metric(snaps: Sequence[RunSnapshot], runs: list[tuple[RunSnapshot, Records, str]], title: str, attr: str,
            definition: str, goal: Goal, why: Callable[[Records], str]) -> str:
    lines = [Line(x.label, cls, pts) for x, rec, cls in runs if (pts := _series(rec.saves, attr))]
    keys = [Key(x.label, cls, off="" if _series(rec.saves, attr) else "not measured") for x, rec, cls in runs] if len(runs) > 1 else []
    if not lines:
        return figure(title, None, definition, goal=goal, gap=("Not measured at any save yet.", why(runs[0][1])))
    nows = [(num(v, 3) if (v := getattr(rec.saves[-1], attr)) is not None else "", cls) for _x, rec, cls in runs]
    return figure(title, Chart(title, lines=lines, x_domain=x_span(snaps)), definition, goal=goal, now=nows[0][0], nows=nows,
                  keys=keys)


def _no_temperature(rec: Records) -> str:
    return "No save fitted a temperature yet. A net with no skill calibrates nothing, so the exams wait."


def _instrument(snaps: Sequence[RunSnapshot]) -> list[str]:
    runs = [(x, x.records, RUN_CLASSES[i]) for i, x in enumerate(snaps) if x.records is not None and x.records.saves]
    head = snaps[0]
    if not runs:
        gap = (("No save read yet.", "The monitor writes one record per save. None is on file yet.")
               if head.records is not None else
               ("No monitor records for this run.", "Start the server with --records to read the value head per save."))
        return [figure("Value head", None, gap=gap)]
    first = runs[0][1]
    failed = "The held-out read failed at every save so far."
    out = [_metric(snaps, runs, "Held-out cross-entropy", "cf_ce", "Calibrated, on the held-out ring, nats.", LOWER, lambda r: failed),
           _metric(snaps, runs, "AUC", "auc", "Chance the head ranks a won position above a lost one.", HIGHER, lambda r: failed),
           _metric(snaps, runs, "Temperature", "temperature", "Fitted per save. Above 1, the head is overconfident.",
                   Goal("near", "best near 1"), _no_temperature)]
    band_lines = [Line(_BAND_NAMES[b], "c1", [(float(s.step), v) for s in first.saves if (v := s.bands.get(b)) is not None],
                       dash=i == 1, faint=i == 0) for i, b in enumerate(PLY_BANDS)]
    drawn = any(ln.pts for ln in band_lines)
    only = head.label if len(snaps) > 1 else ""
    band_chart = Chart("by ply band", lines=band_lines, x_domain=x_span(snaps)) if drawn else None
    out.append(figure("Held-out cross-entropy by ply", band_chart, goal=LOWER, only=only,
                      keys=[Key(_BAND_NAMES[b], "c1", ("faint", "dash", "line")[i]) for i, b in enumerate(PLY_BANDS)],
                      gap=None if drawn else ("Not measured at any save yet.", "No band was read.")))
    line = next((s.gap_line for s in reversed(first.saves) if s.gap_line is not None), None)
    refs = [Ref(line, marked=True)] if line is not None else []
    gap_lines = [Line(x.label, cls, pts) for x, rec, cls in runs if (pts := gaps(rec.saves))]
    gap_chart = Chart("gap", lines=gap_lines, refs=refs, x_domain=x_span(snaps)) if gap_lines else None
    nows = [(signed(g[-1][1], 3) if (g := gaps(rec.saves)) else "", cls) for _x, rec, cls in runs]
    out.append(figure("Train/held-out gap", gap_chart, "Read on the games after each save.",
                      goal=Goal("down", f"lower is better, stay under {line:.2f}" if line is not None else "lower is better"),
                      now=nows[0][0], nows=nows, keys=[Key(x.label, cls) for x, _r, cls in runs] if len(runs) > 1 else [],
                      gap=None if gap_lines else ("No gap read yet.", "A save's gap is read with the save after it.")))
    out.extend(_exams(snaps, first, only))
    return out


def _exams(snaps: Sequence[RunSnapshot], rec: Records, only: str) -> list[str]:
    """One small chart per exam, each with its own floor; the table twin rides on the last."""
    names = sorted({k for s in rec.saves for k in s.exams})
    goal = Goal("up", "higher is better, above the floor")

    def state(e: Any) -> str:
        return "not measured" if e is None or e.holds is None else "holds" if e.holds else "misses"
    rows = [[num(s.step), *(num(s.exams[e].calibrated_mean, 3) if e in s.exams else "—" for e in names),
             *(state(s.exams.get(e)) for e in names)] for s in rec.saves]
    twin = table(["step", *(f"{e} mean" for e in names), *names], rows)
    if not names:
        return [figure("Exams", None, goal=goal, only=only, gap=("No exam measured at any save yet.", _no_temperature(rec)))]
    out = []
    for i, exam in enumerate(names):
        pts = [(float(s.step), e.calibrated_mean) for s in rec.saves
               if (e := s.exams.get(exam)) is not None and e.calibrated_mean is not None]
        floor = next((s.exams[exam].floor for s in reversed(rec.saves) if exam in s.exams and s.exams[exam].floor is not None), None)
        refs = [Ref(floor, label="floor", marked=True, end=True)] if floor is not None else []
        chart = Chart(exam, lines=[Line(exam, "c1", pts)], refs=refs, x_domain=x_span(snaps)) if pts else None
        out.append(figure(f"{exam} exam", chart, "Calibrated mean. A break is an unmeasured save.", goal=goal, only=only,
                          now=num(pts[-1][1], 3) if pts else "", twin=twin if i == len(names) - 1 else "",
                          gap=None if pts else ("Not measured at any save yet.", _no_temperature(rec))))
    return out


def _curve(c: Curve, name: str, cls: str, *, dash: bool, faint: bool) -> Line:
    return Line(name, cls, [(float(k), v) for k, v in enumerate(c.share) if v is not None], dash=dash, faint=faint)


def _span(steps: tuple[int, int] | None) -> str:
    return f"steps {short(steps[0])}–{short(steps[1])}" if steps else "steps not recorded"


def _horizon(h: Horizon | None, only: str) -> str:
    title = "How far ahead the search sees"
    definition = f"Share of positions where the search's value has the right sign, by full turns to the end (up to {K_MAX})."
    goal = HIGHER
    if h is None:
        return figure(title, None, definition, goal=goal, only=only, gap=("No sampled search values in this run's games.",
                                                               "Self-play records carry them once the sampler is on."))
    lines = [_curve(h.early_won, "won, first fifth", "cf", dash=False, faint=True),
             _curve(h.early_lost, "lost, first fifth", "cf", dash=True, faint=True),
             _curve(h.late_won, "side to move won", "c1", dash=False, faint=False),
             _curve(h.late_lost, "side to move lost", "c1", dash=True, faint=False)]

    def cell(c: Curve, k: int) -> str:
        return pct(c.share[k], 1) if c.share[k] is not None else "—"
    rows = [[str(k), cell(h.early_won, k), cell(h.early_lost, k), cell(h.late_won, k), cell(h.late_lost, k),
             num(h.late_won.n[k] + h.late_lost.n[k])] for k in range(K_MAX + 1)]
    return figure(title, Chart(title, lines=lines, y_fmt=lambda v: pct(v), x_fmt=lambda v: num(v), x_name="turns to the end", x_unit="", x_title="full turns to the end",
                               x_domain=(0.0, float(K_MAX)), y_ceil=1.0, height=160),
                  f"{definition} {num(h.games)} sampled games, {num(h.window_games)} per fifth.", goal=goal, only=only,
                  keys=[Key(f"last fifth, {_span(h.late_steps)}", "c1"), Key(f"first fifth, {_span(h.early_steps)}"),
                        Key("dashed: side to move lost", "cf", "dash")],
                  twin=table(["turns to the end", "won, first", "lost, first", "won, last", "lost, last", "positions, last"], rows))


def _reach(h: Horizon | None) -> str:
    if h is None:
        return ""

    if h.late_reach is None and h.early_reach is None:
        return f"The search never calls the winner at {pct(REACH_SHARE)}."
    early = "never" if h.early_reach is None else str(h.early_reach)
    if h.late_reach is None:
        out = "never" if h.early_reach is None else f"{h.early_reach} turn{'s' if h.early_reach != 1 else ''} out"
        return f"The search never calls the winner at {pct(REACH_SHARE)} in the last fifth of sampled games; {out} in the first."
    return (f"The search calls the winner {h.late_reach} turn{'s' if h.late_reach != 1 else ''} out at {pct(REACH_SHARE)} "
            f"in the last fifth of sampled games, and {early} in the first.")


def section(snaps: Sequence[RunSnapshot]) -> tuple[str, str, str]:
    """The verdict from the instrument when it has a save, else from the horizon; the panels."""
    head = snaps[0]
    reach = _reach(head.horizon)
    if head.records is not None and head.records.saves:
        sentence, aside = _verdict(head.records)
        aside = "<br>".join(x for x in (aside, reach) if x)
    elif head.records is not None:
        sentence, aside = "No save read yet.", reach
    elif reach:
        sentence, aside = reach, "No monitor records for this run."
    else:
        sentence, aside = "No value reading yet.", "Neither monitor records nor sampled search values."
    for i, x in enumerate(snaps[1:], 1):
        rec = x.records
        if rec is not None and rec.saves and rec.saves[-1].cf_ce is not None:
            line = f"{esc(x.label)}: held-out cross-entropy {rec.saves[-1].cf_ce:.3f} at {short(rec.saves[-1].step)}."
            if rec.gap_rule is not None:
                line += f" Gap rule fired at {short(rec.gap_rule.get('step'))}."
            aside = "<br>".join(t for t in (aside, f'<span class="{RUN_CLASSES[i]}">{line}</span>') if t)
    return sentence, aside, "".join(_instrument(snaps)) + _horizon(head.horizon, head.label if len(snaps) > 1 else "")
