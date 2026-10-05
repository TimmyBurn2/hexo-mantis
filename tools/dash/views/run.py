"""The Run view: the status line and four questions, each a verdict, an aside and its evidence; the record details closed."""
from __future__ import annotations

from collections.abc import Sequence

from ..readers.record import RunSnapshot
from . import run_series, run_strength, run_value
from .charts import table
from .fmt import esc, num, short, when
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
            else f"{esc(head.label)} record as of {source}.")
    body = (f'<main class="wrap">{status(head, now, failure)}'
            + _section("strength", "Is it getting stronger?", *strength, layout="two")
            + _section("value", "Is the value head learning?", *value, layout="grid")
            + _section("training", "Is training stable?", *train, layout="grid")
            + _section("selfplay", "Is self-play healthy?", *play, layout="grid")
            + details(head) + f'</main><footer class="wrap">{foot}</footer>')
    shell = Shell("run", runs, head.label, snaps[1].label if len(snaps) > 1 else None, frozen)
    title = f"mantis {head.label} at {short(head.events.live_steps)}"
    return render(title, shell, body, scripts=("charts.js",))
