# >300 justify (R8): the panel table, the readers that fill it and the renderer are ONE unit.
# A panel is a producer claim plus the arithmetic over it plus how absence is drawn, and the
# whole point of this tool is that those three never disagree — a panel whose producer is
# declared in one file and drawn in another can be drawn from a number the declaration does
# not cover, which is the invented-number class R333(d) forbids.
"""The run dashboard: one command, one self-contained HTML file, no server and no new producer.

R333(d). This reads a run record that already exists — the JSONL event stream, and optionally
the eval ladder's state file — and writes ONE HTML file with no external references: no CDN, no
JavaScript, no image files beside it. Nothing here emits an event, opens a socket, or touches a
running process.

THE RULE THAT SHAPES EVERY PANEL: **absent is not zero, and it applies to pixels.** A panel whose
producer does not exist at HEAD is declared BANKED in `BANKED_PANELS` below, named with the
producer that would fill it, and drawn as a stated gap. A panel whose producer exists but whose
series is empty in THIS record is drawn as "no rows", naming the event that would carry them.
Neither is ever drawn as a zero, an empty axis, or a flat line at the bottom of a chart — those
are the shapes a reader mistakes for a measurement.

REFUSAL, NOT A BLANK PAGE. An unreadable or event-less record raises `EmptyRunRecord`. A tool
that renders a clean-looking page from nothing is the phantom-gate shape in a different medium.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class EmptyRunRecord(RuntimeError):
    """The record carried no parseable events — refuse rather than render an empty page."""


class UnknownPanel(RuntimeError):
    """A panel was requested that the panel table does not declare."""


# --------------------------------------------------------------------------------------- #
# The panel table — the producer claim, per panel, checked against the record
# --------------------------------------------------------------------------------------- #
#: Panels whose producer DOES NOT EXIST at HEAD. Each row names what would fill it. These are
#: findings, carried into the page so a reader sees the gap rather than an empty chart.
#: Measured in REPAIR-3 Leg 4's producer census (`plan/REPAIR3_WORKING_NOTES.md`).
BANKED_PANELS: dict[str, str] = {
    "average sims/move":
        "NO PRODUCER. `SelfPlayHParams.effective_sims_per_move` is derived in-process and used "
        "to BILL `sims_per_sec` (selfplay/pool_drain.py), and is emitted by nothing. It is "
        "recoverable only as sims_per_sec ÷ positions-per-second — and those two are measured "
        "over DIFFERENT windows (a rolling drain rate against a run average), so the quotient "
        "is not the quantity. Filling this panel needs the field on `iteration_complete`.",
    "held-out loss":
        "NO PRODUCER in a run record. `train/pretrain/heldout.py::HeldOutMonitor.counters()` is "
        "LAW-18-shaped but sits on the BC PRETRAIN path and reports through a LOGGER line "
        "(`graph_route.py`, `bc_graph_pretrain_stop`), never through the JSONL sink. A "
        "self-play run emits no held-out anything. Filling this panel needs the counters "
        "emitted as an event.",
}


@dataclass
class Panel:
    """One rendered block: a title, the events it reads, and its body."""

    title: str
    reads: str
    body: str
    note: str = ""


@dataclass
class Record:
    """A parsed run record. `events` keeps insertion order; `by` indexes by event name."""

    events: list[dict[str, Any]]
    ladder: dict[str, Any] | None = None
    source: str = ""
    by: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        index: dict[str, list[dict[str, Any]]] = {}
        for row in self.events:
            index.setdefault(str(row.get("event", "?")), []).append(row)
        self.by = index

    def rows(self, name: str) -> list[dict[str, Any]]:
        return self.by.get(name, [])


def load_record(events_path: Path, ladder_path: Path | None = None) -> Record:
    """Parse a JSONL event stream, and the ladder state file when one is given.

    Args:
        events_path: the run's JSONL event stream.
        ladder_path: the run's `eval_ladder_state.json`, when it has one.

    Returns:
        The parsed record.

    Raises:
        EmptyRunRecord: the stream parsed to zero events.
        OSError: the stream could not be read.
    """
    events: list[dict[str, Any]] = []
    unparseable = 0
    for line in events_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            unparseable += 1
            continue
        if isinstance(row, dict):
            events.append(row)
    if not events:
        raise EmptyRunRecord(
            f"{events_path} carried no parseable events ({unparseable} unparseable line(s)). "
            "Refusing to render: a page built from nothing looks exactly like a page built "
            "from a clean run, and that is the one mistake this tool must not make."
        )
    ladder = None
    if ladder_path is not None and ladder_path.exists():
        try:
            ladder = json.loads(ladder_path.read_text(encoding="utf-8"))
        except ValueError:
            ladder = None
    return Record(events=events, ladder=ladder, source=str(events_path))


# --------------------------------------------------------------------------------------- #
# Drawing — inline SVG only
# --------------------------------------------------------------------------------------- #
def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _finite(values: list[Any]) -> list[float]:
    out: list[float] = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def no_rows(event_name: str) -> str:
    """The ONE way an empty series is drawn: named, never as a zero or a flat line."""
    return (f'<p class="absent">No rows in this record. This panel is drawn from '
            f'<code>{_esc(event_name)}</code>; the record carries none, which is an ABSENCE '
            f'and is not a zero.</p>')


def sparkline(series: list[tuple[float, float]], *, label: str, height: int = 120,
              width: int = 640, color: str = "#3b6ea5") -> str:
    """One series as an inline SVG polyline with its own min/max/last stated in text."""
    if len(series) < 2:
        return (f'<p class="absent">{_esc(label)}: {len(series)} point(s) — too few to draw a '
                "series. The value(s) are in the table beside this panel.</p>")
    xs = [p[0] for p in series]
    ys = [p[1] for p in series]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    span_x = (x1 - x0) or 1.0
    span_y = (y1 - y0) or 1.0
    pad = 8
    pts = " ".join(
        f"{pad + (x - x0) / span_x * (width - 2 * pad):.2f},"
        f"{height - pad - (y - y0) / span_y * (height - 2 * pad):.2f}"
        for x, y in series
    )
    return (
        f'<figure><svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'role="img" aria-label="{_esc(label)}">'
        f'<polyline fill="none" stroke="{color}" stroke-width="1.6" points="{pts}"/>'
        f'</svg><figcaption>{_esc(label)} — min {y0:.6g}, max {y1:.6g}, last {ys[-1]:.6g} '
        f'over {len(series)} point(s), x from {x0:.6g} to {x1:.6g}</figcaption></figure>'
    )


def table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return ""
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in row) + "</tr>" for row in rows
    )
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


# --------------------------------------------------------------------------------------- #
# The panels
# --------------------------------------------------------------------------------------- #
def panel_throughput(rec: Record) -> Panel:
    rows = rec.rows("iteration_complete")
    if not rows:
        return Panel("Throughput", "iteration_complete", no_rows("iteration_complete"))
    def series(key: str) -> list[tuple[float, float]]:
        out = []
        for r in rows:
            step, val = r.get("step"), r.get(key)
            if isinstance(step, (int, float)) and isinstance(val, (int, float)):
                if math.isfinite(float(val)):
                    out.append((float(step), float(val)))
        return out
    charts = "".join(
        sparkline(series(k), label=f"{k} (per step)")
        for k in ("games_per_hour", "steps_per_hour", "positions_per_hour", "sims_per_sec")
        if series(k)
    )
    last = rows[-1]
    summary = table(
        ["field", "last value"],
        [[k, last.get(k)] for k in ("games_total", "games_this_iter", "avg_game_length",
                                    "buffer_size", "buffer_capacity", "batch_fill_pct",
                                    "corpus_selfplay_frac", "draw_rate")
         if k in last],
    )
    return Panel("Throughput", "iteration_complete", charts + summary)


def panel_sims_per_move(_rec: Record) -> Panel:
    return Panel("Average sims/move", "(none)", banked_block("average sims/move"))


def panel_memory(rec: Record) -> Panel:
    steps = rec.rows("trainer_step")
    body = ""
    if steps:
        rows = []
        for r in steps:
            e, ce = r.get("edges"), r.get("caps_max_edges")
            n, cn = r.get("nodes"), r.get("caps_max_nodes")
            mb = r.get("microbatches") or 1
            if not all(isinstance(v, (int, float)) for v in (e, ce, n, cn)):
                continue  # a row missing a term is skipped, never coerced to a zero
            e, ce, n, cn = float(e), float(ce), float(n), float(cn)  # type: ignore[arg-type]
            if ce <= 0 or cn <= 0:
                continue  # a zero cap is not a 100% share; it is an unusable row
            # per-MICROBATCH share: the cap bounds one microbatch, the event totals the step.
            rows.append((float(r.get("step", 0)), (e / mb) / ce, (n / mb) / cn))
        if rows:
            body += sparkline([(s, ee) for s, ee, _ in rows],
                              label="edges per microbatch ÷ caps_max_edges", color="#a5533b")
            body += sparkline([(s, nn) for s, _, nn in rows],
                              label="nodes per microbatch ÷ caps_max_nodes", color="#3b8a5a")
            body += table(
                ["metric", "max share over the run"],
                [["edges / caps_max_edges", f"{max(e for _, e, _ in rows):.4f}"],
                 ["nodes / caps_max_nodes", f"{max(n for _, _, n in rows):.4f}"]],
            )
        else:
            body += no_rows("trainer_step (edges/nodes/caps_max_*)")
    else:
        body += no_rows("trainer_step")

    mem = rec.rows("eval_round_device_memory")
    if mem:
        rows = []
        for r in mem:
            dm = r.get("device_memory") or {}
            rows.append([r.get("round_id"), r.get("step"), dm.get("available"),
                         dm.get("max_memory_allocated_bytes"),
                         dm.get("max_memory_reserved_bytes")])
        body += "<h4>Eval child, card bytes</h4>" + table(
            ["round", "step", "available", "max allocated", "max reserved"], rows)
    else:
        body += ("<h4>Eval child, card bytes</h4>"
                 + no_rows("eval_round_device_memory"))
    return Panel("Memory shares vs minted caps",
                 "trainer_step, eval_round_device_memory", body,
                 note="SCOPE: the microbatch caps (minted, and carried in the same payload as "
                      "the value they bound) and the eval child's card counters. The trainer's "
                      "GiB budget is a mint decision that reaches no event, so it is not here.")


def panel_losses(rec: Record) -> Panel:
    steps = rec.rows("trainer_step")
    body = ""
    if steps:
        for key, color in (("loss", "#3b6ea5"), ("policy_loss", "#7a3ba5"),
                           ("value_loss", "#a5843b"), ("grad_norm", "#a5533b")):
            ser = [(float(r["step"]), float(r[key])) for r in steps
                   if isinstance(r.get("step"), (int, float))
                   and isinstance(r.get(key), (int, float)) and math.isfinite(float(r[key]))]
            if ser:
                body += sparkline(ser, label=f"trainer_step.{key}", color=color)
    else:
        body += no_rows("trainer_step")
    ts = rec.rows("training_step")
    if ts:
        last = ts[-1]
        body += "<h4>training_step, last row</h4>" + table(
            ["field", "value"],
            [[k, last.get(k)] for k in sorted(last) if k not in ("event", "ts")],
        )
    else:
        body += "<h4>training_step</h4>" + no_rows("training_step")
    return Panel("Training losses", "trainer_step, training_step", body)


def panel_heldout(_rec: Record) -> Panel:
    return Panel("Held-out loss", "(none)", banked_block("held-out loss"))


def panel_gates(rec: Record) -> Panel:
    gates = rec.rows("monitor_gates")
    body = ""
    if gates:
        last = gates[-1].get("gates") or {}
        body += table(
            ["gate", "checks", "fires", "skips", "warns"],
            [[name, s.get("checks"), s.get("fires"), s.get("skips"), s.get("warns")]
             for name, s in sorted(last.items())],
        )
        body += ('<p class="note">Last <code>monitor_gates</code> row of '
                 f'{len(gates)} in this record. A gate with <code>checks 0</code> ran on no '
                 "step here — an absence, not a pass.</p>")
    else:
        body += no_rows("monitor_gates")
    aborts = rec.rows("hard_abort")
    body += "<h4>Hard aborts</h4>"
    body += (table(["rule", "step", "message"],
                   [[a.get("rule"), a.get("step"), a.get("message")] for a in aborts])
             if aborts else
             '<p class="absent">None in this record. That is an absence of the EVENT, and it is '
             'the expected shape for a run that did not abort.</p>')
    floors = rec.rows("eval_strength_floor")
    body += "<h4>Strength-floor refusals</h4>"
    if floors:
        body += table(
            ["round", "step", "passed", "checked_total", "skipped_total"],
            [[f.get("round_id"), f.get("step"), f.get("passed"),
              f.get("checked_total"), f.get("skipped_total")] for f in floors])
    else:
        body += (no_rows("eval_strength_floor")
                 + '<p class="note">The producer EXISTS and emits only on an ARMED round, so an '
                   "empty series here means the posture was disarmed for this run — which is a "
                   "different fact from the floor never refusing anything.</p>")
    return Panel("Gate outcomes and floor refusals",
                 "monitor_gates, hard_abort, eval_strength_floor", body)


def panel_strength(rec: Record) -> Panel:
    body = ""
    if rec.ladder:
        rungs = rec.ladder.get("rungs") or {}
        drew = False
        for name in sorted(rungs):
            history = (rungs[name] or {}).get("history") or []
            wr = [(float(h["round_idx"]), float(h["wr"])) for h in history
                  if isinstance(h.get("round_idx"), (int, float))
                  and isinstance(h.get("wr"), (int, float))]
            if wr:
                drew = True
                body += sparkline(wr, label=f"{name}: win rate by round")
            body += table(
                ["round", "games", "wr", "ci_lo"],
                [[h.get("round_idx"), h.get("games"), h.get("wr"), h.get("ci_lo")]
                 for h in history],
            ) or f'<p class="absent">{_esc(name)}: no history rows.</p>'
        if not rungs:
            body += ('<p class="absent">The ladder state carries no rungs. That is an ABSENCE '
                     "of ladder history, not a strength of zero.</p>")
        elif not drew:
            body += ('<p class="note">No rung has two or more scored rounds, so no series is '
                     "drawn; the per-round rows are above.</p>")
    else:
        body += ('<p class="absent">No <code>eval_ladder_state.json</code> was given or it did '
                 "not parse. The per-round, per-rung <code>(games, wr, ci_lo)</code> series "
                 "lives in that state file — the EVENT stream carries only the transitions "
                 "(<code>eval_rung_activated</code>, <code>eval_rung_graduated</code>), so this "
                 "panel cannot be drawn from the stream alone.</p>")
    rounds = rec.rows("eval_round_complete")
    body += "<h4>Rounds</h4>"
    body += (table(["round", "step", "wall_sec", "games_total", "promoted", "wr_sealbot"],
                   [[r.get("round_id"), r.get("step"), r.get("wall_sec"), r.get("games_total"),
                     r.get("promoted"), r.get("wr_sealbot")] for r in rounds])
             if rounds else no_rows("eval_round_complete"))
    if rounds:
        body += ('<p class="note"><code>games_total: null</code> is a BROKEN round (killed '
                 "before it could report), not a round that played nothing; "
                 "<code>promoted: null</code> is no promotion decision taken, which is not "
                 "<code>false</code>.</p>")
    for name in ("eval_rung_activated", "eval_rung_graduated", "eval_ladder_zero_game_round"):
        got = rec.rows(name)
        if got:
            body += f"<h4>{_esc(name)}</h4>" + table(
                sorted({k for r in got for k in r if k not in ("event", "ts")}),
                [[r.get(k) for k in sorted({k for x in got for k in x
                                            if k not in ("event", "ts")})] for r in got])
    return Panel("Strength vs external rungs, with CIs",
                 "eval_ladder_state.json, eval_round_complete, eval_rung_*", body)


def panel_determinism(rec: Record) -> Panel:
    games = rec.rows("game_complete")
    if not games:
        return Panel("Determinism hash", "game_complete", no_rows("game_complete"))
    hashes = [g.get("game_id_byte_hash") for g in games if g.get("game_id_byte_hash")]
    counts = Counter(hashes)
    repeats = [(h, n) for h, n in counts.items() if n > 1]
    body = table(
        ["metric", "value"],
        [["games with a byte hash", f"{len(hashes)} of {len(games)}"],
         ["DISTINCT trajectories", len(counts)],
         ["repeated trajectories", len(repeats)],
         ["effective-n (LAW-04 dedupe key)", len(counts)]],
    )
    if repeats:
        body += "<h4>Repeated trajectories</h4>" + table(
            ["hash", "times"], [[h, n] for h, n in sorted(repeats, key=lambda p: -p[1])[:20]])
        body += ('<p class="note">A repeat is byte-identical play. LAW-04 counts DISTINCT games, '
                 "so the effective-n above is the row a strength CI must be taken over.</p>")
    return Panel("Determinism hash", "game_complete.game_id_byte_hash", body)


def panel_health(rec: Record) -> Panel:
    segments = rec.rows("run_segment_started")
    body = "<h4>Segments (restarts)</h4>"
    body += (table(["segment", "run_id", "pid", "created_utc"],
                   [[s.get("segment"), s.get("run_id"), s.get("pid"), s.get("created_utc")]
                    for s in segments])
             if segments else no_rows("run_segment_started"))
    if len(segments) > 1:
        body += (f'<p class="note">{len(segments)} segments — the run restarted '
                 f"{len(segments) - 1} time(s).</p>")

    watch = ["heartbeat_watchdog_armed", "heartbeat_watchdog_fired",
             "heartbeat_watchdog_fire_complete", "heartbeat_watchdog_staleness_disarmed",
             "heartbeat_source_unwired", "selfplay_stall_watchdog",
             "selfplay_stall_watchdog_armed", "selfplay_stall_watchdog_save_failed",
             "eval_broken", "eval_round_skipped_busy", "eval_result_unroutable",
             "eval_rung_skipped", "actor_lag_exceeded", "actor_lag_negative",
             "disk_guard_error", "hard_abort", "hard_abort_after_stop", "clean_stop_save",
             "shutdown_save"]
    body += "<h4>Stalls, refusals and lifecycle</h4>" + table(
        ["event", "count", "in this record"],
        [[name, len(rec.rows(name)),
          "yes" if rec.rows(name) else "ABSENT (the event is not emitted; not a zero)"]
         for name in watch],
    )
    alerts = rec.rows("disk_alert")
    body += "<h4>Disk</h4>"
    if alerts:
        body += table(["level", "free GB"],
                      [[a.get("level"), a.get("disk_free_gb")] for a in alerts])
    free = [(float(i), float(r["disk_free_gb"])) for i, r in enumerate(rec.rows("disk_free"))
            if isinstance(r.get("disk_free_gb"), (int, float))]
    body += sparkline(free, label="disk_free_gb by sample") if free else no_rows("disk_free")
    ta = rec.rows("training_alert")
    body += "<h4>Training alerts</h4>"
    if ta:
        byrule = Counter(str(a.get("rule")) for a in ta)
        body += table(["rule", "count", "first message"],
                      [[rule, n, next(a.get("message") for a in ta if a.get("rule") == rule)]
                       for rule, n in byrule.most_common()])
    else:
        body += no_rows("training_alert")
    return Panel("Health", "run_segment_started, watchdogs, disk_*, training_alert", body)


#: The page's panels, in order. The names are the census's names.
PANELS = (
    panel_throughput, panel_sims_per_move, panel_memory, panel_losses, panel_heldout,
    panel_gates, panel_strength, panel_determinism, panel_health,
)


def banked_block(name: str) -> str:
    """A banked panel's body: the finding, never a chart.

    Raises:
        UnknownPanel: `name` is not a declared banked panel.
    """
    if name not in BANKED_PANELS:
        raise UnknownPanel(f"{name!r} is not in BANKED_PANELS; declare it or draw it")
    return (f'<p class="banked"><strong>BANKED — no producer at HEAD.</strong> '
            f'{_esc(BANKED_PANELS[name])}</p>')


# --------------------------------------------------------------------------------------- #
# The page
# --------------------------------------------------------------------------------------- #
_CSS = """
:root { color-scheme: light dark; --fg:#1b1b1b; --bg:#fdfdfc; --muted:#5d5d5d;
        --rule:#d8d5cf; --card:#ffffff; --absent:#8a6d1f; --banked:#8a2f2f; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e7e5e0; --bg:#141413; --muted:#a3a09a; --rule:#33322f; --card:#1c1b1a;
          --absent:#d9b45a; --banked:#e08a8a; } }
* { box-sizing: border-box; }
body { margin:0; padding:24px; background:var(--bg); color:var(--fg);
       font:14px/1.55 -apple-system, "Segoe UI", Roboto, "Helvetica Neue", sans-serif; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:17px; margin:0 0 2px; }
h4 { font-size:13px; margin:18px 0 6px; color:var(--muted);
     text-transform:uppercase; letter-spacing:.04em; }
.sub { color:var(--muted); margin:0 0 24px; }
section { background:var(--card); border:1px solid var(--rule); border-radius:8px;
          padding:16px 18px; margin:0 0 18px; }
.reads { color:var(--muted); font-size:12px; margin:0 0 12px; }
.note { color:var(--muted); font-size:12px; margin:8px 0 0; }
.absent { color:var(--absent); font-size:13px; margin:8px 0; }
.banked { color:var(--banked); font-size:13px; margin:4px 0; }
figure { margin:12px 0; } figcaption { color:var(--muted); font-size:12px; margin-top:2px; }
svg { display:block; background:transparent; }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; font-size:12.5px; min-width:100%; }
th, td { text-align:left; padding:3px 12px 3px 0; border-bottom:1px solid var(--rule);
         white-space:nowrap; }
th { color:var(--muted); font-weight:600; }
code { font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px; }
footer { color:var(--muted); font-size:12px; margin-top:24px; }
"""


def render(rec: Record, title: str) -> str:
    """The whole page, as one self-contained HTML string."""
    panels = [fn(rec) for fn in PANELS]
    counts = Counter(str(r.get("event", "?")) for r in rec.events)
    boot = (rec.rows("run_boot_identity") or [{}])[0]
    # The source is shown by NAME, not by path. A rendered page is an artifact someone will
    # attach to a record, and an absolute path under a home directory is exactly what CI gate 17
    # (rule 7) exists to keep out of a public repo — a report should not be the thing that
    # smuggles one in. The `config_sha256` below is a config hash, not a secret, and is the
    # provenance a reader needs to know WHICH run this is.
    header_rows = [
        ["source", Path(rec.source).name],
        ["events", f"{len(rec.events)} rows, {len(counts)} distinct types"],
        ["run_id", boot.get("run_id", "— (no run_boot_identity row)")],
        ["config sha256", boot.get("config_sha256", "— (no run_boot_identity row)")],
        ["ladder state", "loaded" if rec.ladder else "not given / unparseable"],
    ]
    body = "".join(
        f"<section><h2>{_esc(p.title)}</h2>"
        f'<p class="reads">reads: <code>{_esc(p.reads)}</code></p>'
        f"{p.body}"
        + (f'<p class="note">{_esc(p.note)}</p>' if p.note else "")
        + "</section>"
        for p in panels
    )
    inventory = table(["event", "rows"], [[k, v] for k, v in counts.most_common()])
    return (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{_esc(title)}</title><style>{_CSS}</style></head><body>"
        f"<h1>{_esc(title)}</h1>"
        '<p class="sub">Regenerated from an existing run record. No server, no producer, no '
        "live connection. A panel with no producer is drawn as a stated gap — absent is not "
        "zero.</p>"
        f"<section><h2>Record</h2>{table(['field', 'value'], header_rows)}"
        f"<h4>Event inventory</h4>{inventory}</section>"
        f"{body}"
        "<footer>mantis run dashboard — <code>tools/run_dashboard.py</code> (R333(d)). "
        "Every number on this page is read from the record; none is derived across differently "
        "windowed producers.</footer></body></html>"
    )


def self_test() -> int:
    """Prove the two refusals fire: an empty record, and a banked panel that is not declared."""
    import tempfile

    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        empty = Path(tmp) / "empty.jsonl"
        empty.write_text("\n  \nnot json\n", encoding="utf-8")
        try:
            load_record(empty)
            print("  [SELF-TEST FAILED] an EMPTY record rendered instead of refusing")
            bad += 1
        except EmptyRunRecord:
            print("  [OK] an empty/unparseable record REFUSES")

        one = Path(tmp) / "one.jsonl"
        one.write_text('{"event": "run_boot_identity", "run_id": "x"}\n', encoding="utf-8")
        page = render(load_record(one), "self-test")
        if "BANKED — no producer at HEAD" not in page:
            print("  [SELF-TEST FAILED] a banked panel did not announce itself")
            bad += 1
        else:
            print("  [OK] a banked panel announces itself as a gap")
        if "absent" not in page:
            print("  [SELF-TEST FAILED] an empty series did not render as an absence")
            bad += 1
        else:
            print("  [OK] an empty series renders as an ABSENCE, not a zero")
        # A chart drawn from nothing is the invented number in its most convincing form.
        if "<figure>" in page:
            print("  [SELF-TEST FAILED] a CHART was drawn from a record with no series")
            bad += 1
        else:
            print("  [OK] no chart is drawn from a record with no series")
        # A zero may appear ONLY as an occurrence count that says ABSENT in the same row.
        naked = [row for row in re.findall(r"<tr>.*?</tr>", page)
                 if "<td>0</td>" in row and "ABSENT" not in row]
        if naked:
            print(f"  [SELF-TEST FAILED] {len(naked)} zero(s) drawn without an absence label: "
                  f"{naked[:2]}")
            bad += 1
        else:
            print("  [OK] every zero on the page is labelled an ABSENCE in its own row")

    try:
        banked_block("a panel nobody declared")
        print("  [SELF-TEST FAILED] an undeclared banked panel rendered")
        bad += 1
    except UnknownPanel:
        print("  [OK] an UNDECLARED banked panel refuses")
    print("self-test: all controls fire" if not bad else f"self-test: {bad} control(s) DID NOT FIRE")
    return bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", type=Path, help="the run's JSONL event stream")
    ap.add_argument("--ladder-state", type=Path, default=None,
                    help="the run's eval_ladder_state.json (the per-rung wr/ci series)")
    ap.add_argument("--out", type=Path, help="the HTML file to write")
    ap.add_argument("--title", default=None, help="page title (default: the events file's name)")
    ap.add_argument("--self-test", action="store_true", help="prove the refusals fire")
    args = ap.parse_args(argv)

    if args.self_test:
        return 1 if self_test() else 0
    if not args.events or not args.out:
        ap.error("--events and --out are required (or pass --self-test)")

    rec = load_record(args.events, args.ladder_state)
    title = args.title or f"mantis run record — {args.events.name}"
    args.out.write_text(render(rec, title), encoding="utf-8")
    banked = len(BANKED_PANELS)
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes) from {len(rec.events)} events; "
          f"{len(PANELS)} panels, {banked} BANKED (no producer at HEAD): "
          f"{', '.join(sorted(BANKED_PANELS))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
