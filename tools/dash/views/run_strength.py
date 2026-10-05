"""Is it getting stronger: the rule's ruler in win rate with its parent and going-forward bands, every ruler in logit over its own parent."""
from __future__ import annotations

from collections.abc import Sequence

from ..readers.record import RunSnapshot
from ..readers.sidecars import CELLS, LINE_LOGIT, Cell, Ruler, logit
from .charts import RUN_CLASSES, Key, figure, table
from .fmt import esc, num, pct, short, signed
from .stats import expit, separation
from .svg import Chart, Dots, Line, Ref

_WORD = {1: "stronger than", -1: "weaker than", 0: "not separable from"}
_OWN = {1: "clearly stronger than", -1: "weaker than", 0: "not separable from"}
#: Colour classes for the rulers on the logit chart, in the order they are listed.
_RULER_CLASSES = ("c1", "c2", "c3", "cf")


def _role(r: Ruler, any_rule: bool) -> str:
    return "the rule" if r.rule else "report-only" if any_rule else "no rule declared"


def _lead(rulers: Sequence[Ruler]) -> Ruler | None:
    """The ruler the verdict reads: the rule's, else the first listed (Six before Strix)."""
    return next((r for r in rulers if r.rule), rulers[0] if rulers else None)


def _parent_name(c: Cell) -> str:
    return f"{c.run_id} at {short(c.step)}"


def verdict(snap: RunSnapshot) -> tuple[str, str]:
    """The sentence and its aside; each word gated by its interval, the going-forward read only on the rule's ruler."""
    lead = _lead(snap.rulers)
    if lead is None or not lead.line:
        return "No strength reading in this run's record yet.", "A cell sidecar for one of the run's checkpoints would start the line."
    last, first = lead.line[-1], lead.line[0]
    parts = []
    if lead.parent is not None:
        vs = separation(last.wr, last.n, lead.parent.wr, lead.parent.n)
        parts.append(f"{short(last.step)} is <strong>{_WORD[vs.sign]}</strong> its parent{' yet' if vs.sign == 0 else ''}")
    if len(lead.line) > 1:
        own = separation(last.wr, last.n, first.wr, first.n)
        parts.append(f"{'and ' if parts else short(last.step) + ' is '}{_OWN[own.sign]} its own {short(first.step)}")
    sentence = (", ".join(parts) + f" on {esc(lead.name)}.") if parts else \
        f"{short(last.step)} is the run's only cell on {esc(lead.name)}, and it has no parent there to read against."
    aside = f"On {esc(lead.name)} ({_role(lead, any(r.rule for r in snap.rulers))}) it wins {pct(last.wr, 1)} of {num(last.n)} games"
    if lead.parent is not None:
        aside += f"; the parent ({esc(_parent_name(lead.parent))}) won {pct(lead.parent.wr, 1)}"
    aside += ". A word needs the 95 % interval on the difference to exclude zero."
    gf = lead.going_forward if lead.rule else None
    if gf is not None:
        mean, used = gf
        aside += (f" Going forward: the mean of the last {used} cell{'s' if used > 1 else ''} is {signed(mean)} logit over the parent, "
                  f"{'above' if mean > LINE_LOGIT else 'below'} the line at {signed(LINE_LOGIT)}"
                  + ("" if used >= CELLS else f" ({used} of {CELLS} cells so far)") + ".")
    others = [(r, r.parent) for r in snap.rulers if r is not lead and r.line and r.parent is not None]
    if others:
        aside += " Report-only: " + "; ".join(
            f"{esc(r.name)} {signed(r.line[-1].logit - p.logit)} logit over its parent at {short(r.line[-1].step)}"
            for r, p in others if p is not None) + "."
    return sentence, aside + _ladder_text(snap)


def _ladder_text(snap: RunSnapshot) -> str:
    rungs = snap.ladder
    if rungs is None:
        return ""
    if rungs.note != "read":
        return f" The ruler ladder: {esc(rungs.note)}."
    text = f" The comparison ruler is {esc(rungs.current or 'not named')}" + (
        f", streak {rungs.streak}" if rungs.streak is not None else "") + "."
    for change, pairs in snap.bridges:
        text += f" It escalated from {esc(change.frm)} to {esc(change.to)} at {short(change.step)}"
        if pairs:
            a, b = pairs[-1]
            text += f"; bridge: {short(a.step)} read {pct(a.wr)} on {esc(change.frm)} and {pct(b.wr)} on {esc(change.to)}."
        else:
            text += "; no checkpoint is read on both rungs yet."
    return text


def _marks(snap: RunSnapshot) -> list[tuple[float, str]]:
    marks = [(float(r["step"]), f"gate promotion at {num(r['step'])}") for r in snap.events.rows("eval_round_complete")
             if r.get("promoted") is True and isinstance(r.get("step"), int)]
    if snap.ladder is not None:
        marks += [(float(c.step), f"ruler {c.frm} → {c.to} at {num(c.step)}") for c in snap.ladder.changes]
    return marks


def _rule_panel(snaps: Sequence[RunSnapshot], lead: Ruler, xmax: float) -> str:
    head = snaps[0]
    title = f"Win rate against {lead.name}"
    dots = [Dots(head.label, "c1", [(float(c.step), c.wr, c.lo, c.hi) for c in lead.line])]
    for i, other in enumerate(snaps[1:], 1):
        twin = next((r for r in other.rulers if r.unit == lead.unit), None)
        if twin is not None:
            dots.append(Dots(other.label, RUN_CLASSES[i], [(float(c.step), c.wr, c.lo, c.hi) for c in twin.line]))
    keys = [Key(x.label, RUN_CLASSES[i], "dot") for i, x in enumerate(snaps)]
    refs, lines = [], []
    if lead.parent is not None:
        refs.append(Ref(lead.parent.wr, lead.parent.lo, lead.parent.hi, f"parent {_parent_name(lead.parent)}"))
        keys.append(Key("parent, 95 % band", glyph="band"))
        gf = lead.going_forward if lead.rule else None
        if gf is not None:
            mean, used = gf
            refs.append(Ref(expit(lead.parent.logit + LINE_LOGIT), label=f"parent {signed(LINE_LOGIT)} logit", marked=True, end=True))
            last = lead.line[-used:]
            level = expit(lead.parent.logit + mean)
            lines.append(Line(f"mean of the last {used}", "c1", [(float(last[0].step), level), (float(last[-1].step), level)], dash=True))
            keys += [Key("going-forward line", glyph="dash"), Key(f"mean of the last {used} cell{'s' if used > 1 else ''}", "c1", "dash")]
    marks = _marks(head)
    if marks:
        keys.append(Key("promotion or ruler change", glyph="mark"))
    chart = Chart(title, lines=lines, dots=dots, refs=refs, marks=marks, y_fmt=lambda v: pct(v), width=400, height=220,
                  y_floor=0.0, y_ceil=1.0, x_domain=(0.0, xmax))
    definition = (f"{lead.label}; {_role(lead, lead.rule)}. Whiskers are each cell's 95 % interval over distinct games; "
                  "the table names each cell's host load.")
    return figure(title, chart, definition, now=pct(lead.line[-1].wr), keys=keys)


def _logit_panel(head: RunSnapshot, xmax: float) -> str:
    any_rule = any(r.rule for r in head.rulers)
    dots, keys = [], []
    drawn = [(r, r.parent) for r in head.rulers if r.parent is not None and r.line]
    for i, (r, parent) in enumerate(drawn):
        if parent is None:
            continue
        cls, base = _RULER_CLASSES[min(i, len(_RULER_CLASSES) - 1)], parent.logit
        dots.append(Dots(r.name, cls, [(float(c.step), c.logit - base, None if c.lo is None else logit(c.lo) - base,
                                         None if c.hi is None else logit(c.hi) - base) for c in r.line]))
        keys.append(Key(f"{r.name}, {_role(r, any_rule)}", cls, "dot"))
    refs = [Ref(0.0, label="each ruler's parent")]
    if any_rule:
        refs.append(Ref(LINE_LOGIT, label=f"the rule's line {signed(LINE_LOGIT)}", marked=True, end=True))
    missing = [r.name for r in head.rulers if r.parent is None]
    definition = ("Each ruler's cells as logit(win rate) minus the logit of the parent's win rate on the same ruler, so "
                  "rulers of different strength read on one axis; rulers are never pooled or joined.")
    if missing:
        definition += f" No parent cell on {', '.join(missing)}, so it is in the table only."
    rows = [[r.name, _role(r, any_rule), num(c.step), pct(c.wr, 1), f"{pct(c.lo, 1)} – {pct(c.hi, 1)}", num(c.n),
             signed(c.logit - r.parent.logit) if r.parent is not None else "—", c.regime]
            for r in head.rulers for c in r.line]
    rows += [[r.name, "parent", num(r.parent.step), pct(r.parent.wr, 1), f"{pct(r.parent.lo, 1)} – {pct(r.parent.hi, 1)}",
              num(r.parent.n), "+0.00", r.parent.regime] for r in head.rulers if r.parent is not None]
    twin = table(["ruler", "role", "step", "win rate", "95 % interval", "games", "logit over parent", "host load"], rows)
    if not dots:
        return figure("Every ruler, over its parent", None, definition, twin=twin,
                      gap=("No ruler has a parent cell yet.", "Each ruler needs the parent's checkpoint read on it."))
    chart = Chart("every ruler", dots=dots, refs=refs, marks=_marks(head), y_fmt=lambda v: signed(v), width=400, height=220,
                  x_domain=(0.0, xmax))
    return figure("Every ruler, over its parent", chart, definition, keys=keys, twin=twin)


def section(snaps: Sequence[RunSnapshot]) -> tuple[str, str, str]:
    """The verdict, the aside and the two panels; a run with no cell at all is one stated gap."""
    head = snaps[0]
    sentence, aside = verdict(head)
    lead = _lead(head.rulers)
    if lead is None:
        return sentence, aside, figure("Win rate against the rulers", None, "Six and Strix cells, each unit its own series.",
                                       gap=("No ruler reading in this run's record.", "The follower writes one sidecar per cell "
                                            "beside the checkpoint; none was found under the cell directories given."))
    xmax = max([float(x.events.live_steps or 0) for x in snaps] + [float(c.step) for r in head.rulers for c in r.line])
    return sentence, aside, _rule_panel(snaps, lead, xmax) + _logit_panel(head, xmax)
