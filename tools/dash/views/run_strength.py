"""Is it getting stronger: the rule's ruler in win rate beside a list of every other reading, each opened on demand."""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from ..readers.record import RunSnapshot
from ..readers.sidecars import CELLS, LINE_LOGIT, Cell, Ruler
from .charts import RUN_CLASSES, Goal, Key, figure, table
from .fmt import at, esc, num, pct, short, signed
from .stats import expit, separation
from .svg import Chart, Dots, Ref

_WORD = {1: "beats", -1: "trails", 0: "shows no clear difference from"}
_AHEAD = {1: "ahead", -1: "behind", 0: "level"}
_WIN_RATE = Goal("up", "win rate, higher is better")


@dataclass(frozen=True)
class Reading:
    """One row of the readings list and the panel it opens."""

    key: str
    title: str
    value: str
    note: str
    panel: str
    wide: bool = False


def _role(r: Ruler, declared: bool) -> str:
    if r.rule:
        return "the rule"
    if r.rule_until is not None:
        return f"the rule before {short(r.rule_until)}"
    return "report-only" if declared else "no rule declared"


def _lead(rulers: Sequence[Ruler]) -> Ruler | None:
    """The ruler the verdict reads: the rule's, else a former rule's, else the first listed (Six before Strix)."""
    return next((r for r in rulers if r.rule), rulers[0] if rulers else None)


def titles_of(snaps: Sequence[RunSnapshot]) -> dict[str, str]:
    """Every unit any served run has read, by name and by unit field, to its ruler's title."""
    return {k: r.title for x in snaps for r in x.rulers for k in (r.unit_field, r.name)}


def _parent_name(c: Cell) -> str:
    return at(c.run_id, c.step)


def _since(line: Sequence[Cell]) -> Cell:
    """The cell the latest is set against: the first of the last `CELLS` on this instrument, else the first."""
    return line[-CELLS:][0]


def verdict(snap: RunSnapshot, titles: dict[str, str] | None = None) -> tuple[str, str]:
    """The sentence and its aside; each word gated by its interval, the going-forward read only on the rule's ruler."""
    names = titles or titles_of([snap])
    lead = _lead(snap.rulers)
    switch = _switch_text(snap, lead, names)
    if lead is None or not lead.line:
        return "No strength reading yet.", switch or "The first cell sidecar starts the line."
    last, before = lead.line[-1], _since(lead.line)
    parts = []
    if lead.parent is not None:
        vs = separation(last.wr, last.n, lead.parent.wr, lead.parent.n)
        parts.append(f"{short(last.step)} <strong>{_WORD[vs.sign]}</strong> its parent")
    if before is not last:
        own = separation(last.wr, last.n, before.wr, before.n)
        parts.append(f"{_WORD[own.sign]} {short(before.step)}" if parts else f"{short(last.step)} {_WORD[own.sign]} {short(before.step)}")
    role = f" ({_role(lead, True)})" if snap.rule is not None else ""
    sentence = (" and ".join(parts) + f" against {esc(lead.title)}{role}.") if parts else \
        f"{short(last.step)} wins <strong>{pct(last.wr, 1)}</strong> against {esc(lead.title)}{role}. No parent cell to compare."
    lines = [f"{short(last.step)}: {pct(last.wr, 1)} of {num(last.n)} games"
             + (f", {num(last.forfeits)} Six forfeits left out." if last.forfeits else ".")
             + (f" Parent {esc(_parent_name(lead.parent))}: {pct(lead.parent.wr, 1)}." if lead.parent is not None else "")]
    gf = lead.going_forward if lead.rule else None
    if gf is not None:
        mean, used = gf
        since = "" if lead.rule_since is None else f" since {short(lead.rule_since)}"
        lines.append(f"Mean of the last {used} cell{'s' if used > 1 else ''}{since}: {signed(mean)} logit over the parent, "
                     f"{'above' if mean > LINE_LOGIT else 'below'} the {signed(LINE_LOGIT)} bar"
                     + "." + ("" if used >= CELLS else f" The rule needs {CELLS} cells; {used} {'is' if used == 1 else 'are'} read."))
    lines += [x for x in (switch, _rule_note(snap, names), _ladder_text(snap).strip()) if x]
    return sentence, "<br>".join(lines)


def _rule_note(snap: RunSnapshot, names: dict[str, str]) -> str:
    """A declared rule that names no unit or several is stated (a switch's line already says an unread rule has no cell)."""
    if snap.rule is None or snap.rule_matches == 1 or (snap.rule_matches == 0 and snap.switches):
        return ""
    if snap.rule_matches == 0:
        return f"The rule reads {esc(names.get(snap.rule, snap.rule))}, which has no cell yet."
    return (f"The rule's unit {esc(snap.rule)} matches {snap.rule_matches} series (a pin, tactics block or sims changed), "
            "so the going-forward read is withheld.")


def _switch_text(snap: RunSnapshot, lead: Ruler | None, names: dict[str, str]) -> str:
    """The latest rule switch, and the latest save read on both of its rulers."""
    if not snap.switches:
        return ""
    change, pairs = snap.switches[-1]
    old, new = esc(names.get(change.frm, change.frm)), esc(names.get(change.to, change.to))
    if lead is not None and lead.rule and change.to in (lead.unit_field, lead.name):
        moved = f"The rule moved here from {old} at {short(change.step)}"
    else:
        moved = f"The rule moved from {old} to {new} at {short(change.step)}"
    if snap.rule_matches == 0:
        return f"{moved}; the run has no cell on it yet."
    if not pairs:
        return f"{moved}; no save is read on both yet."
    a, b = pairs[-1]
    return f"{moved}; the {short(a.step)} save reads {pct(a.wr)} on the old ruler and {pct(b.wr)} on the new."


def _twin(snap: RunSnapshot, unit: tuple[str, ...]) -> Ruler | None:
    """The run's ruler on the same unit, when it has read a cell there."""
    return next((r for r in snap.rulers if r.unit == unit and r.line), None)


def _matched(mine: Ruler, theirs: Ruler | None) -> list[tuple[Cell, Cell]]:
    """The saves both runs read on one ruler, by step."""
    by_step = {c.step: c for c in theirs.line} if theirs is not None else {}
    return [(c, by_step[c.step]) for c in mine.line if c.step in by_step]


def _gap(pairs: Sequence[tuple[Cell, Cell]]) -> str:
    """Two runs at the saves both read: every logit gap of the last `CELLS`, the latest's interval and its gated word."""
    mine, theirs = pairs[-1]
    word = _AHEAD[separation(mine.wr, mine.n, theirs.wr, theirs.n).sign]
    d = mine.logit - theirs.logit
    hw = ((mine.logit_half_width or 0.0) ** 2 + (theirs.logit_half_width or 0.0) ** 2) ** 0.5
    if len(pairs) == 1:
        return f"{pct(mine.wr, 1)} against {pct(theirs.wr, 1)}, {signed(d)} logit ({signed(d - hw)} to {signed(d + hw)}), {word}"
    shown = pairs[-CELLS:]
    gaps = ", ".join(signed(a.logit - b.logit) for a, b in shown)
    return (f"{gaps} logit, {short(shown[0][0].step)} to {short(mine.step)} (the latest {signed(d - hw)} to {signed(d + hw)}), "
            f"{word}")


def _compared(snaps: Sequence[RunSnapshot], lead: Ruler) -> str:
    """Each compared run against this one at the saves both read: on the lead ruler, else on the ruler shared latest (unpaired)."""
    out = ""
    for i, x in enumerate(snaps[1:], start=1):
        pairs, where = _matched(lead, _twin(x, lead.unit)), ""
        if not pairs:
            shared = [(p, r) for r in snaps[0].rulers if (p := _matched(r, _twin(x, r.unit)))]
            if shared:
                pairs, other = max(shared, key=lambda pr: pr[0][-1][0].step)
                lead_name = "the rule's ruler" if lead.rule else "the ruler charted here"
                where = f" on {esc(other.title)} (it has none on {lead_name} yet)"
        if not pairs:
            said = f"{esc(x.label)} shares no save with this run on any ruler yet."
        else:
            saves = f"the same save, {short(pairs[0][0].step)}" if len(pairs) == 1 else f"the {len(pairs)} saves both read"
            said = f"Against {esc(x.label)}{where}, at {saves}: {_gap(pairs)}; unpaired."
        out += f'<br><span class="{RUN_CLASSES[i]}">{said}</span>'
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
        text += f" Moved up from {esc(change.frm)} to {esc(change.to)} at {short(change.step)}"
        if pairs:
            a, b = pairs[-1]
            text += f"; bridge {short(a.step)}: {pct(a.wr)} on {esc(change.frm)}, {pct(b.wr)} on {esc(change.to)}."
        else:
            text += "; no checkpoint read on both rungs yet."
    return text


def _marks(snap: RunSnapshot) -> list[tuple[float, str]]:
    marks = [(float(r["step"]), f"gate promotion at {num(r['step'])}") for r in snap.events.rows("eval_round_complete")
             if r.get("promoted") is True and isinstance(r.get("step"), int)]
    if snap.ladder is not None:
        marks += [(float(c.step), f"rung {c.frm} → {c.to} at {num(c.step)}") for c in snap.ladder.changes]
    return marks


def _tables(snaps: Sequence[RunSnapshot], ruler: Ruler) -> str:
    """The ruler's own cells, then each compared run against it at every save both read."""
    out = table(["step", "win rate", "95 % interval", "games", "host load"],
                [[num(c.step), pct(c.wr, 1), f"{pct(c.lo, 1)} to {pct(c.hi, 1)}",
                  num(c.n) + (f" ({num(c.forfeits)} forfeits left out)" if c.forfeits else ""), c.regime] for c in ruler.line])
    for x in snaps[1:]:
        rows = []
        for mine, theirs in _matched(ruler, _twin(x, ruler.unit)):
            sep = separation(mine.wr, mine.n, theirs.wr, theirs.n)
            rows.append([num(mine.step), pct(mine.wr, 1), pct(theirs.wr, 1), f"{signed(sep.d * 100, 1)} pts",
                         f"{signed(sep.lo * 100, 1)} to {signed(sep.hi * 100, 1)}", signed(mine.logit - theirs.logit)])
        out += table(["step", snaps[0].label, x.label, f"{snaps[0].label} − {x.label}", "95 % interval", "logit"], rows,
                     f"Head to head with {x.label}, {len(rows)} save{'s' if len(rows) != 1 else ''} (unpaired)")
    return out


def _now(r: Ruler, step: int) -> str:
    """A ruler's win rate at `step` when it read that save, else its latest with the step named."""
    c = next((c for c in r.line if c.step == step), r.line[-1])
    return pct(c.wr) + ("" if c.step == step else f" at {short(c.step)}")


def _winrate_panel(snaps: Sequence[RunSnapshot], ruler: Ruler, xmax: float, *, lead: bool, latest: int) -> str:
    """One ruler in win rate with its own parent's band, every compared run's twin overlaid; the bar only on the rule's."""
    head = snaps[0]
    title = ruler.title
    twins = [ruler, *(_twin(x, ruler.unit) for x in snaps[1:])]
    spread = (len(snaps) - 1) / 2
    dots = [Dots(x.label, RUN_CLASSES[i], [(float(c.step), c.wr, c.lo, c.hi) for c in t.line], dx=(i - spread) * 5)
            for i, (x, t) in enumerate(zip(snaps, twins, strict=True)) if t is not None]
    keys = [Key(x.label, RUN_CLASSES[i], "dot", off="" if t is not None else "not read on this ruler")
            for i, (x, t) in enumerate(zip(snaps, twins, strict=True))] if len(snaps) > 1 else []
    refs = []
    if ruler.parent is not None:
        refs.append(Ref(ruler.parent.wr, ruler.parent.lo, ruler.parent.hi, f"parent {_parent_name(ruler.parent)}"))
        keys.append(Key("parent, 95 % band", glyph="band"))
        if ruler.rule:
            refs.append(Ref(expit(ruler.parent.logit + LINE_LOGIT), label="bar", marked=True, end=True))
            keys.append(Key(f"bar, parent {signed(LINE_LOGIT)} logit", "warn", "dash"))
    marks = _marks(head)
    switches = [(float(c.step), f"rule switch at {num(c.step)}") for c, _ in head.switches]
    keys += [Key("rule switch", glyph="switch")] if switches else []
    if marks:
        keys.append(Key("promotion or rung change" if head.ladder is not None and head.ladder.changes else "gate promotion",
                        glyph="mark"))
    chart = Chart(title, dots=dots, refs=refs, marks=marks, switches=switches, y_fmt=lambda v: pct(v), width=400, height=220,
                  y_floor=0.0, y_ceil=1.0, x_domain=(0.0, xmax))
    role = _role(ruler, head.rule is not None)
    lead_in = "Report-only. " if role == "report-only" else "" if ruler.rule or ruler.rule_until is None else f"T{role[1:]}. "
    definition = lead_in + f"{ruler.label}. Unit {ruler.name}. Whiskers: 95 % interval." + (
        " Beats or trails only when the difference's interval excludes 0." if lead else "")
    own = ruler.line[-1].step
    nows = [(_now(t, latest if t is ruler else own) if t is not None else "", RUN_CLASSES[i]) for i, t in enumerate(twins)]
    return figure(title, chart, definition, goal=_WIN_RATE, now=_now(ruler, latest), nows=nows, keys=keys,
                  twin=_tables(snaps, ruler))


def _natural(text: str) -> list[int | str]:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", text)]


def _oneoffs_panel(rulers: Sequence[Ruler]) -> str:
    """Single reads with no parent, no second save and no twin to set them against: listed, never charted."""
    rows = [[r.title, num(c.step), pct(c.wr, 1), f"{pct(c.lo, 1)} to {pct(c.hi, 1)}", num(c.n)]
            for r in sorted(rulers, key=lambda r: _natural(r.title)) for c in r.line]
    return ('<div class="sheet"><h3>One-off reads</h3>'
            + table(["ruler", "step", "win rate", "95 % interval", "games"], rows, "Cells", opened=True)
            + "<p>A single cell with nothing to set it against: a screen or a sweep.</p></div>")


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _readings_list(label: str, items: Sequence[Reading]) -> str:
    """Each further reading as a link to its panel; the script turns the links into toggles that open one panel each."""
    rows = "".join(f'<a class="reading" href="#sp-{r.key}" data-key="{r.key}"><span class="t">{esc(r.title)}</span>'
                   f'<span class="v">{esc(r.value)}</span>' + (f'<span class="n">{esc(r.note)}</span>' if r.note else "") + "</a>"
                   for r in items)
    return (f'<div class="readings" data-run="{esc(label)}" role="group" aria-labelledby="readings-title">'
            f'<h3 id="readings-title">More readings</h3>{rows}</div>')


def section(snaps: Sequence[RunSnapshot]) -> tuple[str, str, str]:
    """The verdict, the aside, the lead ruler's chart beside the readings list, then the opened panels; no cell is one gap."""
    head = snaps[0]
    names = titles_of(snaps)
    sentence, aside = verdict(head, names)
    lead = _lead(head.rulers)
    if lead is None:
        return sentence, aside, figure("Win rate against the rulers", None, goal=_WIN_RATE,
                                       gap=("No ruler reading for this run.", "No cell sidecar under the --cells directories."))
    xmax = max([float(x.events.live_steps or 0) for x in snaps] + [float(c.step) for r in head.rulers for c in r.line])
    aside += _compared(snaps, lead)
    declared, latest = head.rule is not None, lead.line[-1].step
    others = [r for r in head.rulers if r is not lead and r.line]
    charted = [r for r in others if r.rule_until is not None or r.parent is not None or len(r.line) > 1
               or any(_matched(r, _twin(x, r.unit)) for x in snaps[1:])]
    items = []
    for r in charted:
        notes = [_role(r, declared)] if declared else []
        if r.parent is not None:
            c, p = r.line[-1], r.parent
            notes.append(f"{_WORD[separation(c.wr, c.n, p.wr, p.n).sign]} its parent, {signed(c.logit - p.logit)} logit")
        items.append(Reading(_key(r.name), r.title, _now(r, latest), ", ".join(notes),
                             _winrate_panel(snaps, r, xmax, lead=False, latest=latest)))
    oneoffs = [r for r in others if r not in charted]
    if oneoffs:
        items.append(Reading("one-off", "One-off reads", f"{len(oneoffs)} cell{'s' if len(oneoffs) != 1 else ''}",
                             "screens and sweeps", _oneoffs_panel(oneoffs), wide=True))
    main = _winrate_panel(snaps, lead, xmax, lead=True, latest=latest)
    if not items:
        return sentence, aside, f'<div class="lead">{main}</div>'
    panels = "".join(f'<div class="sp{" full" if r.wide else ""}" id="sp-{r.key}">{r.panel}</div>' for r in items)
    return (sentence, aside, f'<div class="lead">{main}{_readings_list(head.label, items)}</div>'
            f'<div class="sp-grid" data-run="{esc(head.label)}">{panels}</div>')
