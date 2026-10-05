"""Is the value head learning: the monitor's instrument at every save rules; the knowledge horizon is the search's value beside it."""
from __future__ import annotations

from collections.abc import Sequence

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


def _verdict(rec: Records) -> tuple[str, str]:
    """The sentence from the latest save: a halt first, then the lagged read's interval, the exams and the gap rule."""
    last = rec.saves[-1]
    if rec.halt is not None:
        rows = "; ".join(esc(r) for r in rec.halt.get("halting_rows") or [])
        return (f"<strong>A halting row fired</strong> at {num(rec.halt.get('step'))}: {rows}.",
                "The monitor's halt record is on file; the run is the operator's to resume or end.")
    first = rec.saves[0]
    if last.cf_ce is None:
        lead = f"The save at {short(last.step)} was not measured on the held-out set ({esc(last.gen_note or 'no reading')})."
    elif first is not last and first.cf_ce is not None:
        lead = (f"Held-out cross-entropy <strong>{last.cf_ce:.3f}</strong> at {short(last.step)}, "
                f"{'down' if last.cf_ce < first.cf_ce else 'up'} from {first.cf_ce:.3f} at {short(first.step)}.")
    else:
        lead = f"Held-out cross-entropy <strong>{last.cf_ce:.3f}</strong> at {short(last.step)}."
    exams = last.exams
    if not exams or all(e.holds is None for e in exams.values()):
        lead += " The exams were not measured at this save."
    elif all(e.holds for e in exams.values()):
        lead += " Every exam holds its floor."
    else:
        missed = ", ".join(esc(k) for k, e in exams.items() if e.holds is False)
        lead += f" <strong>{missed} below the floor</strong>" + (" (armed)." if last.armed_floors else ".")
    lag = last.lagged
    aside = ""
    if lag is not None and lag.ci is not None and lag.diff is not None:
        lo, hi = lag.ci
        word = "better than" if hi < 0 else "worse than" if lo > 0 else "not separable from"
        aside = (f"On the games after it, the save at {short(lag.step)} reads {word} its lagged net "
                 f"({signed(lag.diff, 3)} nats, interval {signed(lo, 3)} to {signed(hi, 3)}). ")
    if last.gap is not None and last.gap_line is not None:
        aside += (f"The train/held-out gap is {signed(last.gap, 3)} against the line at {last.gap_line:.2f}"
                  + (", and the gap rule has fired." if last.gap_fired else ".") + " ")
    return lead, aside + "Calibrated reads: the temperature is fitted on the held-out set at each save."


def _instrument(snaps: Sequence[RunSnapshot]) -> list[str]:
    runs = [(x, x.records, RUN_CLASSES[i]) for i, x in enumerate(snaps) if x.records is not None and x.records.saves]
    if not runs:
        return [figure("Value instrument", None, "The instrument of record for the value head, read at every save.",
                       gap=("No monitor records for this run.", "Pass the monitor's records directory with --records; "
                            "a run read by no monitor has no per-save value reading."))]
    keys = [Key(x.label, cls, "line", "" if x.records and x.records.saves else "not recorded")
            for x, cls in zip(snaps, RUN_CLASSES, strict=False)] if len(snaps) > 1 else []
    out = []
    for title, attr, definition in (
            ("Held-out cross-entropy", "cf_ce", "Calibrated value cross-entropy on the held-out ring, nats; lower reads better."),
            ("AUC", "auc", "How often the head ranks a won position over a lost one on the held-out ring."),
            ("Temperature", "temperature", "The calibration temperature fitted at each save; above 1 the head is overconfident.")):
        lines = [Line(x.label, cls, _series(rec.saves, attr)) for x, rec, cls in runs]
        last = getattr(runs[0][1].saves[-1], attr)
        out.append(figure(title, Chart(title, lines=lines, x_domain=x_span(snaps)), definition, now=num(last, 3), keys=keys))
    head = runs[0][1]
    band_lines = [Line(_BAND_NAMES[b], "c1", [(float(s.step), v) for s in head.saves if (v := s.bands.get(b)) is not None],
                       dash=i == 1, faint=i == 0) for i, b in enumerate(PLY_BANDS)]
    out.append(figure("Held-out cross-entropy by ply band", Chart("by ply band", lines=band_lines, x_domain=x_span(snaps)),
                      "Faint: plies 0–10; dashed: 11–40; solid: 41 and up.",
                      keys=[Key(_BAND_NAMES[b], "c1", "dash" if i == 1 else "line") for i, b in enumerate(PLY_BANDS)]))
    gaps = [(float(s.step), s.gap) for s in head.saves if s.gap is not None]
    line = next((s.gap_line for s in reversed(head.saves) if s.gap_line is not None), None)
    out.append(figure("Train/held-out gap", Chart("gap", lines=[Line("gap", "c1", gaps)],
                                                  refs=[Ref(line, label="the gap rule's line", marked=True, end=True)]
                                                  if line else [], x_domain=x_span(snaps)),
                      "The memorisation gap read on the games after each save; two saves above the line fire the rule.",
                      now=signed(gaps[-1][1], 3) if gaps else "",
                      gap=None if gaps else ("No gap read yet.", "The first lagged read lands with the second save.")))
    exams = sorted({k for s in head.saves for k in s.exams})
    lines, refs = [], []
    for i, exam in enumerate(exams):
        lines.append(Line(exam, "c1", [(float(s.step), e.calibrated_mean) for s in head.saves
                                       if (e := s.exams.get(exam)) is not None and e.calibrated_mean is not None], dash=i == 1))
        floor = next((s.exams[exam].floor for s in reversed(head.saves) if exam in s.exams and s.exams[exam].floor), None)
        if floor is not None:
            refs.append(Ref(floor, label=f"{exam} floor", marked=True, end=i == 0))
    rows = [[num(s.step), *(num(s.exams[e].calibrated_mean, 3) if e in s.exams else "—" for e in exams),
             *("holds" if s.exams.get(e) and s.exams[e].holds else "not measured" if not s.exams.get(e)
               or s.exams[e].holds is None else "misses" for e in exams)] for s in head.saves]
    out.append(figure("Exams, calibrated", Chart("exams", lines=lines, refs=refs, x_domain=x_span(snaps)),
                      "Each exam's calibrated mean against its floor; an unmeasured save is a gap in the line, never a zero.",
                      keys=[Key(e, "c1", "dash" if i == 1 else "line") for i, e in enumerate(exams)],
                      twin=table(["step", *(f"{e} mean" for e in exams), *exams], rows)))
    return out


def _curve(c: Curve, name: str, cls: str, *, dash: bool, faint: bool) -> Line:
    return Line(name, cls, [(float(k), v) for k, v in enumerate(c.share) if v is not None], dash=dash, faint=faint)


def _horizon(h: Horizon | None) -> str:
    title = "Knowledge horizon"
    definition = ("Share of positions where the sign of the search's root value matches the result, by full turns to the "
                  "end; each turn's first stone of the sampled self-play games.")
    if h is None:
        return figure(title, None, definition, gap=("No sampled search values in this run's game records.",
                                                    "Self-play records carry them for one game in N once the sampler is on."))
    lines = [_curve(h.early_won, "won, early", "cf", dash=False, faint=True),
             _curve(h.early_lost, "lost, early", "cf", dash=True, faint=True),
             _curve(h.late_won, "side to move won", "c1", dash=False, faint=False),
             _curve(h.late_lost, "side to move lost", "c1", dash=True, faint=False)]
    early = f"early in the run (to {short(h.early_steps[1])})" if h.early_steps else "early in the run"
    rows = [[str(k), pct(h.early_won.share[k], 1) if h.early_won.share[k] is not None else "—",
             pct(h.late_won.share[k], 1) if h.late_won.share[k] is not None else "—",
             pct(h.late_lost.share[k], 1) if h.late_lost.share[k] is not None else "—",
             num(h.late_won.n[k] + h.late_lost.n[k])] for k in range(K_MAX + 1)]
    return figure(title, Chart(title, lines=lines, y_fmt=lambda v: pct(v), x_fmt=lambda v: num(v), x_name="turns to the end",
                               x_domain=(0.0, float(K_MAX)), y_ceil=1.0, height=160),
                  f"{definition} {num(h.games)} sampled games; each window is {num(h.window_games)}.",
                  keys=[Key("side to move won", "c1"), Key("side to move lost", "c1", "dash"), Key(early)],
                  twin=table(["turns to the end", "won, early", "won, late", "lost, late", "positions, late"], rows))


def section(snaps: Sequence[RunSnapshot]) -> tuple[str, str, str]:
    """The verdict from the instrument when it has a save, else from the horizon; the panels."""
    head = snaps[0]
    h = head.horizon
    reach = ""
    if h is not None and h.late_reach is not None:
        how = "as" if h.early_reach == h.late_reach else f"up from {h.early_reach}" if (h.early_reach or 0) < h.late_reach \
            else f"down from {h.early_reach}"
        reach = (f"The search calls the winner {h.late_reach} turn{'s' if h.late_reach != 1 else ''} out at "
                 f"{pct(REACH_SHARE)}, {how}{' early in the run' if how == 'as' else ''}.")
    if head.records is not None and head.records.saves:
        sentence, aside = _verdict(head.records)
        aside += (" " + reach + " That is the search's value, not the head.") if reach else ""
    elif reach:
        sentence, aside = reach, "That is the search's value, not the head; no monitor records were given for this run."
    else:
        sentence, aside = "No value reading in this run's record yet.", "Neither a monitor record nor a sampled search value."
    return sentence, aside, "".join(_instrument(snaps)) + _horizon(h)
