"""Is the value head learning: the monitor's instrument at every save rules; the knowledge horizon is the search's value beside it."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ..readers.horizon import K_MAX, REACH_SHARE, Curve, Horizon
from ..readers.record import RunSnapshot
from ..readers.saves import PLY_BANDS, Records, Save
from .charts import RUN_CLASSES, Key, figure, table
from .fmt import esc, num, pct, short, signed
from .run_series import x_span
from .svg import Chart, Line, Ref

_BAND_NAMES = {"plies_0_10": "plies 0–10", "plies_11_40": "plies 11–40", "plies_41_up": "plies 41 and up"}


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
    how = ("and the monitor signalled the run to save and stop" if halt.get("armed") is True and sent is True
           else f"and the monitor was armed, but its signal was not sent ({esc(signal.get('reason') or 'no reason given')})"
           if halt.get("armed") is True and sent is False
           else "but the monitor was not armed to signal the run" if halt.get("armed") is False
           else "at the run's own final save" if halt.get("final_save") else "and no signal is on record")
    return f"<strong>A halting row fired</strong> at {num(halt.get('step'))}, {how}: {rows}."


def _verdict(rec: Records) -> tuple[str, str]:
    """The sentence from the latest save: a halt first, then the reading; the lagged read's interval gates its word."""
    last, first = rec.saves[-1], rec.saves[0]
    halted = _halt(rec)
    if last.cf_ce is None:
        lead = f"The save at {short(last.step)} was not measured on the held-out set ({esc(last.gen_note or 'no reading')})."
    else:
        lead = f"Held-out cross-entropy <strong>{last.cf_ce:.3f}</strong> at {short(last.step)}"
        lead += f"; it read {first.cf_ce:.3f} at {short(first.step)}." if first is not last and first.cf_ce is not None else "."
    exams = last.exams
    if not exams or all(e.holds is None for e in exams.values()):
        lead += " The exams were not measured at this save."
    elif all(e.holds for e in exams.values()):
        lead += " Every exam holds its floor."
    else:
        missed = ", ".join(esc(k) for k, e in exams.items() if e.holds is False)
        lead += f" <strong>{missed} below the floor</strong>" + (" (armed: the next miss halts)." if last.armed_floors else ".")
    aside = ""
    lag = next((s.lagged for s in reversed(rec.saves) if s.lagged is not None and s.lagged.ci is not None), None)
    if lag is not None and lag.ci is not None and lag.diff is not None:
        lo, hi = lag.ci
        word = "better than" if hi < 0 else "worse than" if lo > 0 else "not separable from"
        aside = (f"On the games after it, the save at {short(lag.step)} reads {word} its lagged net "
                 f"({signed(lag.diff, 3)} nats, interval {signed(lo, 3)} to {signed(hi, 3)}). ")
    series = gaps(rec.saves)
    line = next((s.gap_line for s in reversed(rec.saves) if s.gap_line is not None), None)
    if series and line is not None:
        aside += f"The train/held-out gap at {short(series[-1][0])} is {signed(series[-1][1], 3)} against the line at {line:.2f}. "
    if rec.gap_rule is not None:
        over = ", ".join(short(s) for s in rec.gap_rule.get("over") or [])
        aside += f"The gap rule fired at {num(rec.gap_rule.get('step'))} (saves {over} above the line): a rate re-mint is owed. "
    return (halted + " " + lead if halted else lead), aside + "The temperature is fitted on the held-out set at each save."


def _metric(snaps: Sequence[RunSnapshot], runs: list[tuple[RunSnapshot, Records, str]], title: str, attr: str,
            definition: str, why: Callable[[Records], str]) -> str:
    lines = [Line(x.label, cls, pts) for x, rec, cls in runs if (pts := _series(rec.saves, attr))]
    keys = [Key(x.label, cls, off="" if _series(rec.saves, attr) else "not measured") for x, rec, cls in runs] if len(runs) > 1 else []
    if not lines:
        return figure(title, None, definition, gap=("Not measured at any save yet.", why(runs[0][1])))
    last = getattr(runs[0][1].saves[-1], attr)
    return figure(title, Chart(title, lines=lines, x_domain=x_span(snaps)), definition, now=num(last, 3), keys=keys)


def _no_temperature(rec: Records) -> str:
    return "Every GEN read so far fitted no temperature: a net with no skill calibrates nothing, so its exams stay unread."


def _instrument(snaps: Sequence[RunSnapshot]) -> list[str]:
    runs = [(x, x.records, RUN_CLASSES[i]) for i, x in enumerate(snaps) if x.records is not None and x.records.saves]
    head = snaps[0]
    if not runs:
        gap = (("No save read yet.", "The monitor writes one record per save; none is on file under the records directory.")
               if head.records is not None else
               ("No monitor records for this run.", "A run read by no monitor has no per-save value reading; the server "
                "takes the monitor's records with --records."))
        return [figure("Value instrument", None, "The instrument of record for the value head, read at every save.", gap=gap)]
    first = runs[0][1]
    out = [_metric(snaps, runs, "Held-out cross-entropy", "cf_ce", "Calibrated value cross-entropy on the held-out ring, nats.",
                   lambda r: "The GEN read failed at every save on record."),
           _metric(snaps, runs, "AUC", "auc", "How often the head ranks a won position over a lost one on the held-out ring.",
                   lambda r: "The GEN read failed at every save on record."),
           _metric(snaps, runs, "Temperature", "temperature",
                   "The calibration temperature fitted at each save; above 1 the head is overconfident.", _no_temperature)]
    band_lines = [Line(_BAND_NAMES[b], "c1", [(float(s.step), v) for s in first.saves if (v := s.bands.get(b)) is not None],
                       dash=i == 1, faint=i == 0) for i, b in enumerate(PLY_BANDS)]
    out.append(figure("Held-out cross-entropy by ply band", Chart("by ply band", lines=band_lines, x_domain=x_span(snaps))
                      if any(ln.pts for ln in band_lines) else None, "Faint: plies 0–10; dashed: 11–40; solid: 41 and up.",
                      keys=[Key(_BAND_NAMES[b], "c1", "dash" if i == 1 else "line") for i, b in enumerate(PLY_BANDS)],
                      gap=None if any(ln.pts for ln in band_lines) else ("Not measured at any save yet.", "No band was read.")))
    series = gaps(first.saves)
    line = next((s.gap_line for s in reversed(first.saves) if s.gap_line is not None), None)
    refs = [Ref(line, label="the gap rule's line", marked=True, end=True)] if line is not None else []
    out.append(figure("Train/held-out gap", Chart("gap", lines=[Line("gap", "c1", series)], refs=refs, x_domain=x_span(snaps))
                      if series else None,
                      "The memorisation gap of each save, read on the games after it; two saves above the line fire the rule.",
                      now=signed(series[-1][1], 3) if series else "",
                      gap=None if series else ("No gap read yet.", "A save's gap is read with the save after it.")))
    out.append(_exams(snaps, first))
    return out


def _exams(snaps: Sequence[RunSnapshot], rec: Records) -> str:
    names = sorted({k for s in rec.saves for k in s.exams})
    lines, refs = [], []
    for i, exam in enumerate(names):
        pts = [(float(s.step), e.calibrated_mean) for s in rec.saves
               if (e := s.exams.get(exam)) is not None and e.calibrated_mean is not None]
        if pts:
            lines.append(Line(exam, "c1", pts, dash=i == 1))
        floor = next((s.exams[exam].floor for s in reversed(rec.saves) if exam in s.exams and s.exams[exam].floor is not None), None)
        if floor is not None:
            refs.append(Ref(floor, label=f"{exam} floor", marked=True, end=i == 0))

    def state(e: Any) -> str:
        return "not measured" if e is None or e.holds is None else "holds" if e.holds else "misses"
    rows = [[num(s.step), *(num(s.exams[e].calibrated_mean, 3) if e in s.exams else "—" for e in names),
             *(state(s.exams.get(e)) for e in names)] for s in rec.saves]
    definition = "Each exam's calibrated mean against its floor; an unmeasured save is a gap in the line, never a zero."
    twin = table(["step", *(f"{e} mean" for e in names), *names], rows)
    if not lines:
        return figure("Exams, calibrated", None, definition, twin=twin,
                      gap=("No exam measured at any save yet.", _no_temperature(rec)))
    return figure("Exams, calibrated", Chart("exams", lines=lines, refs=refs, x_domain=x_span(snaps)), definition,
                  keys=[Key(e, "c1", "dash" if i == 1 else "line") for i, e in enumerate(names)], twin=twin)


def _curve(c: Curve, name: str, cls: str, *, dash: bool, faint: bool) -> Line:
    return Line(name, cls, [(float(k), v) for k, v in enumerate(c.share) if v is not None], dash=dash, faint=faint)


def _span(steps: tuple[int, int] | None) -> str:
    return f"steps {short(steps[0])}–{short(steps[1])}" if steps else "steps not recorded"


def _horizon(h: Horizon | None) -> str:
    title = "Knowledge horizon"
    definition = ("Share of positions where the sign of the search's root value matches the result, by full turns to the "
                  f"end (up to {K_MAX}); each turn's first stone of the sampled self-play games.")
    if h is None:
        return figure(title, None, definition, gap=("No sampled search values in this run's game records.",
                                                    "Self-play records carry them for one game in N once the sampler is on."))
    lines = [_curve(h.early_won, "won, first fifth", "cf", dash=False, faint=True),
             _curve(h.early_lost, "lost, first fifth", "cf", dash=True, faint=True),
             _curve(h.late_won, "side to move won", "c1", dash=False, faint=False),
             _curve(h.late_lost, "side to move lost", "c1", dash=True, faint=False)]

    def cell(c: Curve, k: int) -> str:
        return pct(c.share[k], 1) if c.share[k] is not None else "—"
    rows = [[str(k), cell(h.early_won, k), cell(h.early_lost, k), cell(h.late_won, k), cell(h.late_lost, k),
             num(h.late_won.n[k] + h.late_lost.n[k])] for k in range(K_MAX + 1)]
    return figure(title, Chart(title, lines=lines, y_fmt=lambda v: pct(v), x_fmt=lambda v: num(v), x_name="turns to the end",
                               x_domain=(0.0, float(K_MAX)), y_ceil=1.0, height=160),
                  f"{definition} {num(h.games)} sampled games; each window is the first or last {num(h.window_games)}.",
                  keys=[Key(f"side to move won, last fifth ({_span(h.late_steps)})", "c1"),
                        Key("side to move lost", "c1", "dash"), Key(f"first fifth ({_span(h.early_steps)})")],
                  twin=table(["turns to the end", "won, first", "lost, first", "won, last", "lost, last", "positions, last"], rows))


def _reach(h: Horizon | None) -> str:
    if h is None:
        return ""
    def says(r: int | None) -> str:
        return "no turn" if r is None else f"{r} turn{'s' if r != 1 else ''}"
    return (f"In the last fifth of the sampled games the search calls the winner {says(h.late_reach)} out at {pct(REACH_SHARE)}; "
            f"in the first fifth, {says(h.early_reach)}. That is the search's value, not the head.")


def section(snaps: Sequence[RunSnapshot]) -> tuple[str, str, str]:
    """The verdict from the instrument when it has a save, else from the horizon; the panels."""
    head = snaps[0]
    reach = _reach(head.horizon)
    if head.records is not None and head.records.saves:
        sentence, aside = _verdict(head.records)
        aside += (" " + reach) if reach else ""
    elif head.records is not None:
        sentence, aside = "No save has been read yet.", ("The monitor's records are given; its first save record is not on file. " + reach).strip()
    elif reach:
        sentence, aside = reach.split(";")[0] + ".", reach + " No monitor records were given for this run."
    else:
        sentence, aside = "No value reading in this run's record yet.", "Neither a monitor record nor a sampled search value."
    return sentence, aside, "".join(_instrument(snaps)) + _horizon(head.horizon)
