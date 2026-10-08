"""The Run view: the status line and four questions, each a verdict, an aside and its evidence; the record details closed."""
from __future__ import annotations

from collections.abc import Sequence

from ..readers.record import RunSnapshot
from ..readers.sidecars import Cell, Ruler
from . import run_series, run_strength, run_value
from .charts import table
from .fmt import esc, kilo, num, pct, short, sig, when
from .page import Shell, render

_STATE = {"live": "Live", "stale": "Stale", "stopped": "Stopped", "unknown": "No heartbeat", "frozen": "Frozen",
          "unread": "Read failed"}


def status(snap: RunSnapshot, now: float | None, failure: str | None = None) -> str:
    """Live, stale or stopped by the heartbeat's age (Frozen in a frozen file, Read failed after a failed read), then the facts."""
    ev = snap.events
    state = "frozen" if now is None else "unread" if failure else snap.beat.state(now)
    saves = [r.get("step") for r in ev.rows("periodic_checkpoint_save") if isinstance(r.get("step"), int)]
    alerts = len(ev.rows("training_alert"))
    seg = ev.live_segment
    seg_fact = ""
    if seg is not None:
        since = when(seg.started.get("ts") if seg.started else seg.first_ts)
        seg_fact = f'<span class="fact">segment <b>{seg.number}</b>since {esc(since)}</span>'
    beat = "heartbeat" if state == "live" else "last heartbeat"
    beat_ts = snap.beat.wall_ts if now is None else snap.beat.beat_ts
    fail = (f'<span class="fact warn">last read failed (<b>{esc(failure)}</b>), showing the one before</span>'
            if failure else "")
    return (f'<div class="status"><span class="state {state}"><span class="dot"></span>{_STATE[state]}</span>'
            f'<span class="fact"><b>{num(ev.live_steps)}</b>steps</span>'
            f'<span class="fact"><b>{num(ev.games.count)}</b>games</span>'
            f'<span class="fact"><b>{short(saves[-1]) if saves else "none"}</b>last save</span>{seg_fact}'
            f'<span class="fact">{beat} <b>{esc(when(beat_ts))}</b></span>{fail}'
            f'<span class="fact{" warn" if alerts else ""}"><b>{alerts or "No"}</b>{"warning" if alerts == 1 else "warnings"}'
            "</span></div>")


def _section(key: str, question: str, sentence: str, aside: str, panels: str, layout: str) -> str:
    return (f'<section class="q" id="{key}"><div class="say"><h2>{question}</h2><p class="verdict">{sentence}</p>'
            f'<p class="aside">{aside}</p></div><div class="panels {layout}">{panels}</div></section>')


def _outcome(promoted: object) -> str:
    return "promoted" if promoted is True else "not promoted" if promoted is False else "no decision"


def _wall(sec: object) -> str:
    if not isinstance(sec, (int, float)) or isinstance(sec, bool):
        return "—"
    return f"{num(sec)} s" if sec < 120 else f"{num(sec / 60)} min"


def _ruler(snap: RunSnapshot, unit: str | None) -> tuple[Ruler | None, str]:
    """The one series `unit` names (an exact name first, else a unique unit field), or None and the cell's word for why."""
    if unit is None:
        return None, "not declared"
    named = ([r for r in snap.rulers if r.name.split(" #")[0] == unit]
             or [r for r in snap.rulers if r.unit_field == unit])
    if len(named) > 1:
        return None, f"ambiguous ({len(named)} series)"
    return (named[0], "not read") if named else (None, "not read")


def _ruled_at(snap: RunSnapshot, step: int) -> str | None:
    """The unit the rule read at `step`: the first unit before its first switch, then each switch's unit from its step."""
    unit = snap.switches[0][0].frm if snap.switches else snap.rule
    for change, _ in snap.switches:
        unit = change.to if step >= change.step else unit
    return unit


def _read(found: tuple[Ruler | None, str], step: int) -> str:
    ruler, why = found
    c: Cell | None = next((c for c in ruler.line if c.step == step), None) if ruler is not None else None
    if c is None:
        return why
    return pct(c.wr) if c.lo is None or c.hi is None else f"{pct(c.wr)} [{num(100.0 * c.lo)}–{pct(c.hi)}]"


def saves(snap: RunSnapshot) -> str:
    """One row a save, newest first: the rule's and the second ruler's readings, the monitor's rows, the rate, the gate round."""
    if snap.records is None or not snap.records.saves:
        return ""
    second = _ruler(snap, snap.ladder.second if snap.ladder is not None else None)
    rounds: dict[int, list[str]] = {}
    for r in snap.events.rows("eval_round_complete"):
        if isinstance(r.get("step"), int):
            rounds.setdefault(r["step"], []).append(f"{r.get('round_id', '')} {_outcome(r.get('promoted'))}".strip())
    rows = []
    for s in reversed(snap.records.saves):
        held = sum(e.holds is True for e in s.exams.values())
        rows.append([num(s.step), _read(_ruler(snap, _ruled_at(snap, s.step)), s.step), _read(second, s.step), sig(s.cf_ce),
                     num(s.temperature, 2), sig(s.gap), f"{held} of {len(s.exams)}" if s.exams else "—",
                     kilo(s.positions_per_h) if s.positions_per_h is not None else "not measured",
                     "; ".join(rounds.get(s.step, [])) or "—"])
    (lead, _), (other, _) = _ruler(snap, snap.rule), second
    caption = (f"The ruler of record is {lead.title if lead else snap.rule or 'not declared'}; the second ruler "
               f"{other.title if other else (snap.ladder.second if snap.ladder else None) or 'not declared'}. Newest save "
               "first; a gate round shows at the save it ran on")
    head = ["save", "ruler of record", "second ruler", "GEN cf CE", "T", "gap", "exams held", "positions/h save to save", "gate round"]
    return (f'<details class="record" open><summary>Saves <span>{len(rows)} saves</span></summary>'
            f'<div class="record-body"><div>{table(head, rows, caption, opened=True)}</div></div></details>')


def details(snap: RunSnapshot) -> str:
    """Gate rounds, alerts and stops, the segments, the inputs' notes and the event inventory; nothing else survives."""
    ev = snap.events
    rounds = [[str(r.get("round_id", "")), num(r.get("step")), _outcome(r.get("promoted")), _wall(r.get("wall_sec"))]
              for r in ev.rows("eval_round_complete")]
    stops = [[name, num(r.get("step")), str(r.get("rule") or r.get("message") or "")]
             for name in ("training_alert", "hard_abort", "hard_abort_after_stop", "shutdown_save", "clean_stop_save")
             for r in ev.rows(name)]
    segs = [[str(s.number), str((s.started or {}).get("pid", "—")), when(s.first_ts), when(s.last_ts), num(s.rows)]
            for s in ev.segments]
    identity = ev.last("run_boot_identity") or {}
    notes = [["config sha256", str(identity.get("config_sha256") or "not recorded")[:16]],
             ["parent", snap.parent_stem or "none"], ["games indexed", num(snap.games_indexed)],
             ["records", "none given" if snap.records is None else f"{len(snap.records.saves)} saves read"]]
    notes += [["cell skipped", n] for n in snap.cells_skipped]
    if snap.records is not None:
        notes += [["save skipped", n] for n in snap.records.skipped]
    inventory = sorted(ev.counts.items(), key=lambda kv: -kv[1])
    blocks = [("Gate rounds", ["round", "step", "outcome", "wall"], rounds),
              ("Alerts and stops", ["event", "step", "rule"], stops),
              ("Segments", ["segment", "pid", "first row", "last row", "rows"], segs),
              ("Inputs", ["input", "reading"], notes),
              ("Event inventory", ["event", "rows"], [[k, num(v)] for k, v in inventory])]
    body = "".join(f"<div><h4>{esc(title)}</h4>{table(head, rows) if rows else '<p class=muted>none</p>'}</div>"
                   for title, head, rows in blocks)
    def count(n: int, word: str) -> str:
        return f"{n} {word}{'' if n == 1 else 's'}"
    summary = f"{count(len(rounds), 'gate round')}, {count(len(ev.segments), 'segment')}, {count(len(ev.counts), 'event kind')}"
    return (f'<details class="record"><summary>Record details <span>{esc(summary)}</span></summary>'
            f'<div class="record-body">{body}</div></details>')


def page(snaps: Sequence[RunSnapshot], *, runs: tuple[str, ...], now: float | None, failure: str | None = None) -> str:
    """The whole Run view for the first snapshot, any second one overlaid; `now` None renders the frozen form."""
    head = snaps[0]
    frozen = now is None
    strength = run_strength.section(snaps)
    value = run_value.section(snaps)
    train = run_series.training(snaps)
    play = run_series.selfplay(snaps)
    source = esc(when(head.events.last_ts))
    foot = (f"Frozen copy of {esc(head.label)}, record as of {source}." if frozen
            else ", ".join(f"{esc(x.label)} record as of {esc(when(x.events.last_ts))}" for x in snaps) + ".")
    body = (f'<main class="wrap">{status(head, now, failure)}'
            + _section("strength", "Is it getting stronger?", *strength, layout="stack")
            + _section("value", "Is the value head learning?", *value, layout="grid")
            + _section("training", "Is training stable?", *train, layout="grid")
            + _section("selfplay", "Is self-play healthy?", *play, layout="grid")
            + saves(head) + details(head) + f'</main><footer class="wrap">{foot}</footer>')
    shell = Shell("run", runs, head.label, snaps[1].label if len(snaps) > 1 else None, frozen)
    title = f"mantis {head.label} at {short(head.events.live_steps)}"
    return render(title, shell, body, scripts=("charts.js",))
