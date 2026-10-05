"""Is it getting stronger: the Six cells as the primary panel, Strix where a run has strix cells, the parent band, the going-forward band."""
from __future__ import annotations

from collections.abc import Sequence

from ..readers.record import RunSnapshot
from ..readers.sidecars import CELLS, LINE_LOGIT, Cell, Strength
from .charts import RUN_CLASSES, Key, figure, table
from .fmt import esc, num, pct, short, signed
from .stats import expit, separation
from .svg import Chart, Dots, Line, Ref

_FAMILY = {"six": "Six", "strix": "Strix"}
_WORD = {1: "stronger than", -1: "weaker than", 0: "not separable from"}
_UNIT = {"six": "Six generation 30 at 16 nodes, the ruler unit, paired openings.",
         "strix": "Strix at its pinned net and simulations, paired openings."}


def _promotions(snap: RunSnapshot) -> list[tuple[float, str]]:
    return [(float(r["step"]), f"gate promotion at {num(r['step'])}") for r in snap.events.rows("eval_round_complete")
            if r.get("promoted") is True and isinstance(r.get("step"), int)]


def verdict(s: Strength) -> tuple[str, str]:
    """The sentence and its aside, each word gated by its interval; the going-forward read is stated as the rule reads it."""
    if not s.line:
        return "No strength reading in this run's record yet.", "A cell sidecar for one of the run's checkpoints would start the line."
    last, first = s.line[-1], s.line[0]
    parts = []
    if s.parent is not None:
        vs = separation(last.wr, last.n, s.parent.wr, s.parent.n)
        parts.append(f"{short(last.step)} is <strong>{_WORD[vs.sign]}</strong> its parent{' yet' if vs.sign == 0 else ''}")
    if len(s.line) > 1:
        own = separation(last.wr, last.n, first.wr, first.n)
        lead = "and " if parts else f"{short(last.step)} is "
        word = {1: "clearly stronger than", -1: "weaker than", 0: "not separable from"}[own.sign]
        parts.append(f"{lead}{word} its own {short(first.step)}")
    sentence = (", ".join(parts) + ".") if parts else f"{short(last.step)} is the run's only cell, and it has no parent to read against."
    aside = f"Against {_FAMILY[s.family]} it wins {pct(last.wr, 1)} of {num(last.n)} games"
    if s.parent is not None:
        aside += f"; the parent ({esc(s.parent.run_id)} at {short(s.parent.step)}) won {pct(s.parent.wr, 1)}"
    aside += ". A word needs the 95 % interval on the difference to exclude zero."
    gf = s.going_forward
    if gf is not None:
        mean, used = gf
        side = "above" if mean > LINE_LOGIT else "below"
        aside += (f" Going forward: the mean of the last {used} cell{'s' if used > 1 else ''} is {signed(mean)} logit over the "
                  f"parent, {side} the line at {signed(LINE_LOGIT)}" + ("" if used >= CELLS else f" ({used} of {CELLS} cells so far)") + ".")
    return sentence, aside


def _dots(cells: Sequence[Cell], name: str, cls: str) -> Dots:
    return Dots(name, cls, [(float(c.step), c.wr, c.lo, c.hi) for c in cells])


def panel(snaps: Sequence[RunSnapshot], family: str, *, primary: bool) -> str:
    """One family's chart over the runs on the page; the first run's parent and going-forward band are drawn on it."""
    head = snaps[0]
    s: Strength = getattr(head, family)
    title = f"Win rate against {_FAMILY[family]}"
    if not any(getattr(x, family).line for x in snaps):
        return figure(title, None, _UNIT[family],
                      gap=(f"No {_FAMILY[family]} reading in this run's record.",
                           "The follower writes one sidecar per cell beside the checkpoint; none was found under the cell "
                           "directories given."))
    dots = [_dots(getattr(x, family).line, x.label, RUN_CLASSES[i]) for i, x in enumerate(snaps)
            if getattr(x, family).line]
    refs: list[Ref] = []
    lines: list[Line] = []
    keys = [Key(x.label, RUN_CLASSES[i], "dot", "" if getattr(x, family).line else "not measured")
            for i, x in enumerate(snaps)]
    if s.parent is not None:
        refs.append(Ref(s.parent.wr, s.parent.lo, s.parent.hi, f"parent {s.parent.run_id} at {short(s.parent.step)}"))
        keys.append(Key("parent, 95 % band", glyph="band"))
        if primary and s.going_forward is not None:
            bar = expit(s.parent.logit + LINE_LOGIT)
            refs.append(Ref(bar, label=f"parent {signed(LINE_LOGIT)} logit", marked=True, end=True))
            mean, used = s.going_forward
            last = s.line[-used:]
            level = expit(s.parent.logit + mean)
            lines.append(Line(f"mean of the last {used}", "c1", [(float(last[0].step), level), (float(last[-1].step), level)],
                              dash=True))
            keys += [Key("going-forward line", glyph="dash"), Key(f"mean of the last {used} cells", "c1", "dash")]
    marks = _promotions(head)
    if marks:
        keys.append(Key("gate promotion", glyph="mark"))
    unit = s.line[0].label if s.line else ""
    xmax = max([float(x.events.steps_max or 0) for x in snaps] + [p[0] for d in dots for p in d.pts])
    chart = Chart(title, lines=lines, dots=dots, refs=refs, marks=marks, y_fmt=lambda v: pct(v), width=400, height=220,
                  y_floor=0.0, y_ceil=1.0, x_domain=(0.0, xmax))
    rows = [[x.label, num(c.step), pct(c.wr, 1), f"{pct(c.lo, 1)} – {pct(c.hi, 1)}", num(c.n), c.regime, "line"]
            for x in snaps for c in getattr(x, family).line]
    rows += [[head.label, num(c.step), pct(c.wr, 1), f"{pct(c.lo, 1)} – {pct(c.hi, 1)}", num(c.n), c.regime,
              f"other unit: {c.label}"] for c in s.other]
    if s.parent is not None:
        rows.append(["parent", num(s.parent.step), pct(s.parent.wr, 1), f"{pct(s.parent.lo, 1)} – {pct(s.parent.hi, 1)}",
                     num(s.parent.n), s.parent.regime, s.parent.stem])
    definition = (f"{unit}. Whiskers are each cell's 95 % interval over distinct games. A cell read in another unit is "
                  "in the table as \"other unit\", never on the line.")
    if s.parent is None and s.parent_other is not None:
        definition += f" The parent's cell ({s.parent_other.stem}) was read in another unit, so no band is drawn."
    return figure(title, chart, definition, now=pct(s.line[-1].wr) if s.line else "", keys=keys,
                  twin=table(["run", "step", "win rate", "95 % interval", "games", "regime", "unit"], rows))


def section(snaps: Sequence[RunSnapshot]) -> tuple[str, str, str]:
    """The verdict, the aside and the panels: Six always, Strix only where a run on the page has a strix cell."""
    head = snaps[0]
    lead = head.six if head.six.line or not head.strix.line else head.strix
    sentence, aside = verdict(lead)
    panels = panel(snaps, "six", primary=True)
    if any(x.strix.line or x.strix.other for x in snaps):
        panels += panel(snaps, "strix", primary=False)
    return sentence, aside, panels
