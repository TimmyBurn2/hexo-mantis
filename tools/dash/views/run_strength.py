"""Is it getting stronger: each ruler in win rate over its own parent's band (the bar on the rule's), every ruler in logit."""
from __future__ import annotations

from collections.abc import Sequence

from ..readers.ladder import Change
from ..readers.record import RunSnapshot
from ..readers.sidecars import CELLS, LINE_LOGIT, Cell, Ruler
from .charts import HIGHER, RUN_CLASSES, Goal, Key, figure, table
from .fmt import at, esc, num, pct, short, signed
from .stats import expit, separation
from .svg import Chart, Dots, Line, Ref

_WORD = {1: "beats", -1: "trails", 0: "shows no clear difference from"}
#: Colour classes for the rulers on the logit chart, in the order they are listed.
_RULER_CLASSES = ("c1", "c3", "cf", "cf")


def _role(r: Ruler, declared: bool) -> str:
    if r.rule:
        return "the rule"
    if r.rule_until is not None:
        return f"the rule before {short(r.rule_until)}"
    return "report-only" if declared else "no rule declared"


def _rule_note(snap: RunSnapshot) -> str:
    """A declared rule that names no unit or several is stated, and its going-forward read withheld."""
    if snap.rule is None or snap.rule_matches == 1:
        return ""
    if snap.rule_matches == 0:
        return f" The rule reads {esc(snap.rule)}, which has no cell yet."
    return (f" The rule's unit {esc(snap.rule)} matches {snap.rule_matches} series (a pin, tactics block or sims changed), "
            "so the going-forward read is withheld.")


def _lead(rulers: Sequence[Ruler]) -> Ruler | None:
    """The ruler the verdict reads: the rule's, else the first listed (Six before Strix)."""
    return next((r for r in rulers if r.rule), rulers[0] if rulers else None)


def _parent_name(c: Cell) -> str:
    return at(c.run_id, c.step)


def verdict(snap: RunSnapshot) -> tuple[str, str]:
    """The sentence and its aside; each word gated by its interval, the going-forward read only on the rule's ruler."""
    lead = _lead(snap.rulers)
    if lead is None or not lead.line:
        return "No strength reading yet.", "The first cell sidecar starts the line."
    last, first = lead.line[-1], lead.line[0]
    parts = []
    if lead.parent is not None:
        vs = separation(last.wr, last.n, lead.parent.wr, lead.parent.n)
        parts.append(f"{short(last.step)} <strong>{_WORD[vs.sign]}</strong> its parent")
    if len(lead.line) > 1:
        own = separation(last.wr, last.n, first.wr, first.n)
        parts.append(f"{_WORD[own.sign]} {short(first.step)}" if parts else f"{short(last.step)} {_WORD[own.sign]} {short(first.step)}")
    role = f" ({_role(lead, True)})" if snap.rule is not None else ""
    sentence = (" and ".join(parts) + f" on {esc(lead.name)}{role}.") if parts else \
        f"{short(last.step)} wins <strong>{pct(last.wr, 1)}</strong> on {esc(lead.name)}{role}. No parent cell to compare."
    lines = [f"{short(last.step)}: {pct(last.wr, 1)} of {num(last.n)} games"
             + (f", {num(last.forfeits)} Six forfeits left out." if last.forfeits else ".")
             + (f" Parent {esc(_parent_name(lead.parent))}: {pct(lead.parent.wr, 1)}." if lead.parent is not None else "")]
    gf = lead.going_forward if lead.rule else None
    if gf is not None:
        mean, used = gf
        lines.append(f"Mean of the last {used} cell{'s' if used > 1 else ''}: {signed(mean)} logit over the parent, "
                     f"{'above' if mean > LINE_LOGIT else 'below'} the {signed(LINE_LOGIT)} bar"
                     + "." + ("" if used >= CELLS else f" The rule needs {CELLS} cells; {used} {'is' if used == 1 else 'are'} read."))
    lines += [f"{esc(r.name)} at {short(r.line[-1].step)}: {pct(r.line[-1].wr, 1)}"
              + (f" ({num(r.line[-1].forfeits)} Six forfeits left out)" if r.line[-1].forfeits else "")
              + f", parent {pct(p.wr, 1)}, {signed(r.line[-1].logit - p.logit)} logit over the parent."
              for r, p in ((r, r.parent) for r in snap.rulers if r is not lead and r.line) if p is not None]
    lines += [f"The rule moved from {esc(c.frm)} to {esc(c.to)} at {short(c.step)}{_bridge(c, pairs)}" for c, pairs in snap.switches]
    lines += [x for x in (_rule_note(snap).strip(), _ladder_text(snap).strip()) if x]
    return sentence, "<br>".join(lines)


def _twin(snap: RunSnapshot, unit: tuple[str, ...]) -> Ruler | None:
    """The run's ruler on the same unit, when it has read a cell there."""
    return next((r for r in snap.rulers if r.unit == unit and r.line), None)


def _compared(snaps: Sequence[RunSnapshot], lead: Ruler) -> str:
    """Each compared run's latest cell on the lead ruler's unit, one line apiece."""
    out = ""
    for x in snaps[1:]:
        twin = _twin(x, lead.unit)
        if twin is not None:
            c, first = twin.line[-1], twin.line[0]
            if len(twin.line) > 1:
                said = (f"{esc(x.label)}: {short(c.step)} {_WORD[separation(c.wr, c.n, first.wr, first.n).sign]} {short(first.step)}, "
                        f"{pct(c.wr, 1)} of {num(c.n)} games")
            else:
                said = f"{esc(x.label)} at {short(c.step)}: {pct(c.wr, 1)} of {num(c.n)} games"
            said += f", {num(c.forfeits)} Six forfeits left out." if c.forfeits else "."
            out += f'<br><span class="{RUN_CLASSES[snaps.index(x)]}">{said}</span>'
    return out


def _ladder_text(snap: RunSnapshot) -> str:
    rungs = snap.ladder
    if rungs is None:
        return ""
    if not rungs.note.startswith("read"):
        return f" Ladder: {esc(rungs.note)}."
    text = f" Ladder ruler: {esc(rungs.current or 'not named')}" + (
        f", streak {rungs.streak} toward the next rung" if rungs.streak is not None else "") + "."
    if rungs.note != "read":
        text += f" The ladder file {esc(rungs.note[len('read, but '):])}."
    for change, pairs in snap.bridges:
        text += f" Moved up from {esc(change.frm)} to {esc(change.to)} at {short(change.step)}{_bridge(change, pairs)}"
    return text


def _bridge(change: Change, pairs: tuple[tuple[Cell, Cell], ...]) -> str:
    """The latest checkpoint read on both units of a change, or that none is yet."""
    if not pairs:
        return "; no checkpoint read on both units yet."
    a, b = pairs[-1]
    return f"; bridge {short(a.step)}: {pct(a.wr)} on {esc(change.frm)}, {pct(b.wr)} on {esc(change.to)}."


def _marks(snap: RunSnapshot) -> list[tuple[float, str]]:
    marks = [(float(r["step"]), f"gate promotion at {num(r['step'])}") for r in snap.events.rows("eval_round_complete")
             if r.get("promoted") is True and isinstance(r.get("step"), int)]
    if snap.ladder is not None:
        marks += [(float(c.step), f"ruler {c.frm} → {c.to} at {num(c.step)}") for c in snap.ladder.changes]
    marks += [(float(c.step), f"rule {c.frm} → {c.to} at {num(c.step)}") for c, _ in snap.switches]
    return marks


def _winrate_panel(snaps: Sequence[RunSnapshot], ruler: Ruler, xmax: float, lead: bool) -> str:
    """One ruler in win rate with its own parent's band, every compared run's twin overlaid; the bar only on the rule's."""
    head = snaps[0]
    title = f"Win rate against {ruler.name}"
    twins = [ruler, *(_twin(x, ruler.unit) for x in snaps[1:])]
    dots = [Dots(x.label, RUN_CLASSES[i], [(float(c.step), c.wr, c.lo, c.hi) for c in t.line])
            for i, (x, t) in enumerate(zip(snaps, twins, strict=True)) if t is not None]
    keys = [Key(x.label, RUN_CLASSES[i], "dot", off="" if t is not None else "not read on this ruler")
            for i, (x, t) in enumerate(zip(snaps, twins, strict=True))]
    refs, lines = [], []
    if ruler.parent is not None:
        refs.append(Ref(ruler.parent.wr, ruler.parent.lo, ruler.parent.hi, f"parent {_parent_name(ruler.parent)}"))
        keys.append(Key("parent, 95 % band", glyph="band"))
        gf = ruler.going_forward if ruler.rule else None
        if gf is not None:
            mean, used = gf
            refs.append(Ref(expit(ruler.parent.logit + LINE_LOGIT), label="bar", marked=True, end=True))
            last = ruler.line[-used:]
            level = expit(ruler.parent.logit + mean)
            lines.append(Line(f"mean of the last {used}", "c1", [(float(last[0].step), level), (float(last[-1].step), level)], dash=True))
            keys += [Key(f"bar, parent {signed(LINE_LOGIT)} logit", "warn", "dash"), Key(f"mean of the last {used}", "c1", "dash")]
    marks = _marks(head)
    if marks:
        keys.append(Key("promotion or ruler change", glyph="mark"))
    chart = Chart(title, lines=lines, dots=dots, refs=refs, marks=marks, y_fmt=lambda v: pct(v), width=400, height=220,
                  y_floor=0.0, y_ceil=1.0, x_domain=(0.0, xmax))
    role = _role(ruler, head.rule is not None)
    lead_in = "Report-only. " if role == "report-only" else "" if ruler.rule or ruler.rule_until is None else f"T{role[1:]}. "
    definition = lead_in + f"{ruler.label}. Whiskers: 95 % interval." + (
        " Beats or trails only when the difference's interval excludes 0." if lead else "")
    nows = [(pct(t.line[-1].wr) if t is not None else "", RUN_CLASSES[i]) for i, t in enumerate(twins)]
    return figure(title, chart, definition, goal=HIGHER, now=pct(ruler.line[-1].wr), nows=nows, keys=keys)


def _logit_panel(head: RunSnapshot, xmax: float, only: str) -> str:
    declared = head.rule is not None
    dots, keys = [], []
    drawn = [(r, r.parent) for r in head.rulers if r.parent is not None and r.line]
    for i, (r, parent) in enumerate(drawn):
        if parent is None:
            continue
        cls, base, hw_p = _RULER_CLASSES[min(i, len(_RULER_CLASSES) - 1)], parent.logit, parent.logit_half_width or 0.0
        pts = []
        for c in r.line:
            d, hw_c = c.logit - base, c.logit_half_width
            hw = None if hw_c is None else (hw_c * hw_c + hw_p * hw_p) ** 0.5
            pts.append((float(c.step), d, None if hw is None else d - hw, None if hw is None else d + hw))
        dots.append(Dots(r.name, cls, pts, dx=(i - (len(drawn) - 1) / 2) * 6))
        keys.append(Key(f"{r.name}, {_role(r, declared)}", cls, "dot"))
    refs = [Ref(0.0, label="parent")]
    if any(r.rule for r in head.rulers):
        refs.append(Ref(LINE_LOGIT, label=f"bar {signed(LINE_LOGIT)}", marked=True))
    missing = [r.name for r in head.rulers if r.parent is None] if any(r.parent is not None for r in head.rulers) else []
    definition = "Logit of the win rate minus the parent's on the same ruler, so rulers of different strength share one axis."
    if missing:
        definition += (f" No parent cell on {', '.join(missing)}, so not drawn here." if len(missing) <= 3
                       else f" {len(missing)} rulers have no parent cell, so are not drawn here.")
    goal = Goal("up", "higher is better, 0 is the parent")
    rows = [[r.name, _role(r, declared), num(c.step), pct(c.wr, 1), f"{pct(c.lo, 1)} to {pct(c.hi, 1)}",
             num(c.n) + (f" ({num(c.forfeits)} forfeits left out)" if c.forfeits else ""),
             signed(c.logit - r.parent.logit) if r.parent is not None else "—", c.regime]
            for r in head.rulers for c in r.line]
    rows += [[r.name, "parent", num(r.parent.step), pct(r.parent.wr, 1), f"{pct(r.parent.lo, 1)} to {pct(r.parent.hi, 1)}",
              num(r.parent.n), signed(0.0), r.parent.regime] for r in head.rulers if r.parent is not None]
    twin = table(["ruler", "role", "step", "win rate", "95 % interval", "games", "logit over parent", "host load"], rows)
    if not dots:
        return figure("Every ruler against its parent", None, definition, goal=goal, only=only, twin=twin,
                      gap=("No ruler has a parent cell yet.", "Read the parent's save on a ruler to draw it."))
    chart = Chart("every ruler", dots=dots, refs=refs, marks=_marks(head), y_fmt=lambda v: signed(v), width=400, height=220,
                  x_domain=(0.0, xmax))
    return figure("Every ruler against its parent", chart, definition, goal=goal, only=only, keys=keys, twin=twin)


def section(snaps: Sequence[RunSnapshot]) -> tuple[str, str, str]:
    """The verdict, the aside, the rule's win rate, every ruler in logit, then each other ruler's win rate; no cell is one gap."""
    head = snaps[0]
    sentence, aside = verdict(head)
    lead = _lead(head.rulers)
    if lead is None:
        return sentence, aside, figure("Win rate against the rulers", None, goal=HIGHER,
                                       gap=("No ruler reading for this run.", "No cell sidecar under the --cells directories."))
    xmax = max([float(x.events.live_steps or 0) for x in snaps] + [float(c.step) for r in head.rulers for c in r.line])
    aside += _compared(snaps, lead)
    # Every other ruler worth a line gets its own win-rate chart; a lone screen cell with no parent stays in the table.
    others = "".join(_winrate_panel(snaps, r, xmax, lead=False) for r in head.rulers
                     if r is not lead and r.line and (r.parent is not None or len(r.line) > 1))
    return (sentence, aside, _winrate_panel(snaps, lead, xmax, lead=True) + _logit_panel(head, xmax, head.label if len(snaps) > 1 else "")
            + others)
