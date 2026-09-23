# >300 justify (R8): the collapsed tier is one roster — every panel the old dashboard drew that
# is not a chart now lives here, and the roster test reads them as one set.
"""Tier 3: the collapsed detail panels and the absence notes every short marker points at."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .fmt import esc, num, pct
from .health import MIRROR_LAG_WARN_BUNDLES, HealthReading
from .model import Gaps, Panel, no_rows, table
from .reader import Record
from .strength import RoundPoint
from .svg import Series, envelope_chart


class UnknownPanel(RuntimeError):
    """A dropped panel was named that the roster does not declare."""


#: Panels dropped from the default view, each row naming what would fill it (R333(d): a gap
#: with no owner is a complaint, not a finding).
DROPPED: dict[str, str] = {
    "average sims/move":
        "NO PRODUCER. `SelfPlayHParams.effective_sims_per_move` is derived in-process and used "
        "to bill `sims_per_sec` (selfplay/pool_drain.py) and is emitted by nothing; the "
        "quotient sims_per_sec ÷ positions-per-second spans two windows and is not the "
        "quantity. Filling this panel needs `iteration_complete.avg_sims_per_move`.",
    "held-out loss":
        "NO PRODUCER in a run record. `train/pretrain/heldout.py::HeldOutMonitor` sits on the "
        "BC pretrain path and reports through a logger line, never the JSONL sink; a self-play "
        "run emits no held-out anything. Filling this panel needs the counters as an event.",
    "replay ratio":
        "NO PRODUCER for the pair. Samples consumed and positions produced never ride the same "
        "`iteration_complete` row, and a quotient across two producers' windows is not the "
        "ratio. Filling this panel needs `iteration_complete.samples_consumed_total` beside "
        "`positions_produced_total`, cumulative, on one row.",
    "policy entropy / accuracy":
        "NO PRODUCER for accuracy: no trainer tail measures a value accuracy, so `training_step` "
        "carries none. Filling this panel needs the trainer to measure it.",
}


def dropped_note(name: str) -> str:
    """The stated gap for a dropped panel.

    Raises:
        UnknownPanel: `name` is not declared in `DROPPED`.
    """
    if name not in DROPPED:
        raise UnknownPanel(f"{name!r} is not in DROPPED; declare it or draw it")
    return f'<li><b>{esc(name)}</b> — {esc(DROPPED[name])}</li>'


def provenance(rec: Record) -> Panel:
    counts = Counter(str(r.get("event", "?")) for r in rec.events)
    boot = rec.last("run_boot_identity") or {}
    dropped = ", ".join(f"{k}: {v} field(s)" for k, v in sorted(rec.dropped_fields.items())) or "none"
    rows = [
        ["source", Path(rec.source).name],
        ["events", f"{num(len(rec.events))} rows, {len(counts)} distinct types"],
        ["run_id", boot.get("run_id", "— (no run_boot_identity row)")],
        ["config sha256", boot.get("config_sha256", "— (no run_boot_identity row)")],
        ["ladder state", rec.ladder_note],
        ["record directory", "given" if rec.record_dir is not None else "not given (--record-dir)"],
        ["heavy per-game fields dropped at parse", dropped],
    ]
    return Panel("Provenance", "run_boot_identity, the file names given on the command line",
                 table(["field", "value"], rows), "provenance")


def inventory(rec: Record) -> Panel:
    counts = Counter(str(r.get("event", "?")) for r in rec.events)
    return Panel("Event inventory", "every row's `event` field",
                 table(["event", "rows"], [[k, v] for k, v in counts.most_common()]), "inventory")


def memory(rec: Record) -> Panel:
    steps = rec.rows("trainer_step")
    body = ""
    if steps:
        rows = []
        for r in steps:
            e, ce, n, cn = r.get("edges"), r.get("caps_max_edges"), r.get("nodes"), r.get("caps_max_nodes")
            mb = r.get("microbatches") or 1
            if not all(isinstance(v, (int, float)) for v in (e, ce, n, cn)):
                continue
            e, ce, n, cn = float(e), float(ce), float(n), float(cn)  # type: ignore[arg-type]
            if ce <= 0 or cn <= 0:
                continue
            rows.append((float(r.get("step", 0)), (e / mb) / ce, (n / mb) / cn))
        if rows:
            body += envelope_chart([Series("edges per microbatch ÷ caps_max_edges",
                                           [(s, ee) for s, ee, _ in rows], "s4")], x_label="step")
            body += envelope_chart([Series("nodes per microbatch ÷ caps_max_nodes",
                                           [(s, nn) for s, _, nn in rows], "s5")], x_label="step")
            body += table(["metric", "max share over the run"],
                          [["edges / caps_max_edges", f"{max(e for _, e, _ in rows):.4f}"],
                           ["nodes / caps_max_nodes", f"{max(n for _, _, n in rows):.4f}"]])
        else:
            body += no_rows("trainer_step (edges/nodes/caps_max_*)")
    else:
        body += no_rows("trainer_step")
    return Panel("Memory shares vs minted caps", "trainer_step", body, "memory",
                 note="The microbatch caps are minted and carried in the same payload as the "
                      "value they bound. The trainer's GiB budget reaches no event.")


def device_memory(rec: Record) -> Panel:
    mem = rec.rows("eval_round_device_memory")
    if not mem:
        return Panel("Eval child device memory", "eval_round_device_memory",
                     no_rows("eval_round_device_memory"), "device-memory")
    rows = []
    for r in mem:
        dm = r.get("device_memory") or {}
        rows.append([r.get("round_id"), r.get("step"), dm.get("available"),
                     dm.get("max_memory_allocated_bytes"), dm.get("max_memory_reserved_bytes")])
    return Panel("Eval child device memory", "eval_round_device_memory",
                 table(["round", "step", "available", "max allocated", "max reserved"], rows),
                 "device-memory")


def determinism(rec: Record) -> Panel:
    games = rec.rows("game_complete")
    if not games:
        return Panel("Determinism hash", "game_complete.game_id_byte_hash",
                     no_rows("game_complete"), "determinism")
    hashes = [g.get("game_id_byte_hash") for g in games if g.get("game_id_byte_hash")]
    counts = Counter(hashes)
    repeats = [(h, n) for h, n in counts.items() if n > 1]
    body = table(["metric", "value"],
                 [["games with a byte hash", f"{len(hashes)} of {len(games)}"],
                  ["distinct trajectories", len(counts)],
                  ["repeated trajectories", f"{len(repeats)}" + ("" if repeats else " (none)")],
                  ["effective-n (LAW-04 dedupe key)", len(counts)]])
    if repeats:
        body += "<h4>Repeated trajectories</h4>" + table(
            ["hash", "times"], [[h, n] for h, n in sorted(repeats, key=lambda p: -p[1])[:20]])
    return Panel("Determinism hash", "game_complete.game_id_byte_hash", body, "determinism",
                 note="A repeat is byte-identical play; LAW-04 counts distinct games, so the "
                      "effective-n is the row a strength CI must be taken over.")


_WATCH = ("heartbeat_watchdog_armed", "heartbeat_watchdog_fired",
          "heartbeat_watchdog_fire_complete", "heartbeat_watchdog_staleness_disarmed",
          "heartbeat_source_unwired", "selfplay_stall_watchdog", "selfplay_stall_watchdog_armed",
          "selfplay_stall_watchdog_save_failed", "eval_broken", "eval_round_skipped_busy",
          "eval_result_unroutable", "eval_rung_skipped", "actor_lag_exceeded",
          "actor_lag_negative", "disk_guard_error", "hard_abort", "hard_abort_after_stop",
          "clean_stop_save", "shutdown_save")


def lifecycle(rec: Record) -> Panel:
    segments = rec.rows("run_segment_started")
    body = "<h4>Segments (restarts)</h4>"
    body += (table(["segment", "run_id", "pid", "created_utc"],
                   [[s.get("segment"), s.get("run_id"), s.get("pid"), s.get("created_utc")]
                    for s in segments]) if segments else no_rows("run_segment_started"))
    if len(segments) > 1:
        body += f'<p class="note">{len(segments)} segments — the run restarted {len(segments) - 1} time(s).</p>'
    body += "<h4>Stalls, refusals and lifecycle</h4>" + table(
        ["event", "count", "in this record"],
        [[name, len(rec.rows(name)), "yes" if rec.rows(name) else "absent (not emitted; not a zero)"]
         for name in _WATCH])
    alerts = rec.rows("disk_alert")
    body += "<h4>Disk alerts</h4>"
    body += (table(["level", "free GB"], [[a.get("level"), a.get("disk_free_gb")] for a in alerts])
             if alerts else no_rows("disk_alert"))
    return Panel("Lifecycle", "run_segment_started, the watchdog and refusal events, disk_alert",
                 body, "lifecycle")


def mirror(rec: Record) -> Panel:
    reads = "resume_state_persisted.unreceipted_bundles"
    rows = [r for r in rec.rows("resume_state_persisted")
            if isinstance(r.get("unreceipted_bundles"), list)]
    if not rows:
        return Panel("Mirror receipts", reads, no_rows("resume_state_persisted"), "mirror")
    last = rows[-1]
    lag = len(last["unreceipted_bundles"])
    body = table(["published at step", "unreceipted bundles (steps)", "count"],
                 [[r.get("step"), ", ".join(str(x) for x in r["unreceipted_bundles"]) or "none",
                   len(r["unreceipted_bundles"])] for r in rows[-12:]])
    if lag >= MIRROR_LAG_WARN_BUNDLES:
        body += (f'<p class="warn"><b>Warning</b> — {lag} retained bundle(s) carry no receipt at '
                 f"the last publication (step {last.get('step')}): the puller has missed at "
                 f"least {MIRROR_LAG_WARN_BUNDLES} intervals. Not a halt (R349(b)).</p>")
    return Panel("Mirror receipts", reads, body, "mirror",
                 note=f"last publication: {lag} unreceipted (warning at {MIRROR_LAG_WARN_BUNDLES})")


def alpha_full(rec: Record) -> Panel:
    title, reads = "alpha = 1.0 rows per 1,000", "iteration_complete.gumbel_alpha_full"
    blocks = [(r.get("step"), r.get("gumbel_alpha_full")) for r in rec.rows("iteration_complete")]
    measured = [(step, b) for step, b in blocks
                if isinstance(b, dict) and isinstance(b.get("per_1000"), (int, float))]
    if not measured:
        return Panel(title, reads, no_rows("iteration_complete.gumbel_alpha_full"), "alpha")
    series = [(float(step if step is not None else i), float(b["per_1000"]))
              for i, (step, b) in enumerate(measured)]
    last_step, last = measured[-1]
    body = envelope_chart([Series("alpha = 1.0 rows per 1,000 graph rows (cumulative)", series)],
                          x_label="step")
    body += table(["step", "rows", "graph rows", "per 1,000"],
                  [[step, b.get("rows"), b.get("graph_rows"), b.get("per_1000")]
                   for step, b in measured[-8:]])
    return Panel(title, reads, body, "alpha",
                 note=f"at step {last_step}: {last.get('rows')} of {last.get('graph_rows')} rows "
                      f"({last.get('per_1000')} per 1,000) since boot")


def gates(rec: Record) -> Panel:
    rows = rec.rows("monitor_gates")
    body = ""
    if rows:
        last = rows[-1].get("gates") or {}
        body += table(["gate", "checks", "fires", "skips", "warns"],
                      [[name, s.get("checks"), s.get("fires"), s.get("skips"), s.get("warns")]
                       for name, s in sorted(last.items())])
        body += (f'<p class="note">Last <code>monitor_gates</code> row of {len(rows)}. A gate with '
                 "<code>checks 0</code> ran on no step here — an absence, not a pass.</p>")
    else:
        body += no_rows("monitor_gates")
    aborts = rec.rows("hard_abort")
    body += "<h4>Hard aborts</h4>"
    body += (table(["rule", "step", "message"],
                   [[a.get("rule"), a.get("step"), a.get("message")] for a in aborts]) if aborts
             else '<p class="absent">None in this record — an absence of the event, the expected '
                  'shape for a run that did not abort.</p>')
    floors = rec.rows("eval_strength_floor")
    body += "<h4>Strength-floor refusals</h4>"
    if floors:
        body += table(["round", "step", "passed", "checked_total", "skipped_total"],
                      [[f.get("round_id"), f.get("step"), f.get("passed"), f.get("checked_total"),
                        f.get("skipped_total")] for f in floors])
    else:
        body += no_rows("eval_strength_floor") + (
            '<p class="note">The producer emits only on an armed round, so an empty series '
            'means the posture was disarmed for this run.</p>')
    return Panel("Gate outcomes and floor refusals", "monitor_gates, hard_abort, eval_strength_floor",
                 body, "gates")


def firings(health: HealthReading, rec: Record) -> Panel:
    title, reads = "F-816-37 firings", "collate_dumps/*.json + *.log under --record-dir (files, not events)"
    if rec.record_dir is None:
        return Panel(title, reads, '<p class="absent">Absent — no run-record directory was '
                     'supplied, so collate_dumps/ and the logs were not read. This is not a '
                     'report of zero firings.</p>', "firings")
    rows = [[f.channel, f.where, f"{f.when:.0f}", f.detail] for f in health.firings]
    body = table(["channel", "location", "epoch", "detail"], rows) if rows else (
        '<p class="absent">No firings in this record — and the directory was read, which is what '
        'makes this a zero rather than an absence.</p>')
    reason = next(i.reason for i in health.inputs if i.name == "F-816-37 firings")
    return Panel(title, reads, body, "firings", note=reason)


def rounds(rec: Record, series: dict[str, list[RoundPoint]]) -> Panel:
    rows = rec.rows("eval_round_complete")
    body = ""
    if rows:
        body += table(["round", "step", "wall s", "games (all opponents)", "promoted", "wr", "CI lo", "CI hi"],
                      [[r.get("round_id"), r.get("step"), num(r.get("wall_sec")), r.get("games_total"),
                        r.get("promoted"), _rate(r.get("wr_sealbot")),
                        _rate(r.get("wr_sealbot_ci_lower")), _rate(r.get("wr_sealbot_ci_upper"))]
                       for r in rows])
        body += ('<p class="note"><code>games_total: null</code> is a broken round (killed before '
                 "it could report), not a round that played nothing; <code>promoted: null</code> "
                 "is no decision taken, which is not <code>false</code>. The CI is the record's "
                 "own: a 95 % pair-bootstrap over distinct games (LAW-04).</p>")
    else:
        body += no_rows("eval_round_complete")
    for name, points in series.items():
        played = [p for p in points if p.games]
        if played:
            body += f"<h4>{esc(name)} — games per round from the ladder file</h4>" + table(
                ["round", "step", "games", "wr", "Wilson lo", "Wilson hi"],
                [[p.round_idx, p.step, p.games, pct(p.wr, 2),
                  pct(p.wilson[0], 1) if p.wilson else "—", pct(p.wilson[1], 1) if p.wilson else "—"]
                 for p in played])
    health = rec.last("eval_channel_health")
    if health:
        body += "<h4>Channel health, last row</h4>" + table(
            ["field", "value"], [[k, health.get(k)] for k in sorted(health) if k not in ("event", "ts")])
    for name in ("eval_rung_activated", "eval_rung_graduated", "eval_ladder_zero_game_round"):
        got = rec.rows(name)
        if got:
            keys = sorted({k for r in got for k in r if k not in ("event", "ts")})
            body += f"<h4>{esc(name)}</h4>" + table(keys, [[r.get(k) for k in keys] for r in got])
    return Panel("Rounds", "eval_round_complete, eval_ladder_state.json, eval_channel_health, eval_rung_*",
                 body, "rounds")


def _rate(value: Any) -> str:
    """A win rate or CI bound as a percentage; a null stays a visible `None`."""
    return pct(float(value), 2) if isinstance(value, (int, float)) else str(value)


def learning_details(rec: Record) -> Panel:
    ts = rec.last("training_step")
    body = "<h4>training_step, last row</h4>"
    body += (table(["field", "value"], [[k, ts.get(k)] for k in sorted(ts) if k not in ("event", "ts")])
             if ts else no_rows("training_step"))
    return Panel("Learning details", "training_step", body, "learning",
                 note="Null fields on that row are unmeasured, never zero.")


def absence_notes(gaps: Gaps) -> Panel:
    items = "".join(f"<li><b>{esc(panel)}</b> — {statement}</li>" for panel, statement in gaps.items)
    dropped = "".join(dropped_note(name) for name in DROPPED)
    body = ("<h4>Stated gaps on this page</h4>"
            + (f"<ul>{items}</ul>" if items else '<p class="note">None: every panel found its rows.</p>')
            + "<h4>Dropped from the default view</h4>"
            + f'<ul class="dropped">{dropped}</ul>')
    return Panel("Absence notes", "every panel above", body, "notes",
                 note="Every short \"not measured\" marker on the page points here.")


def panels(rec: Record, health: HealthReading, series: dict[str, list[RoundPoint]],
           gaps: Gaps) -> list[Panel]:
    """The tier-3 roster, in page order; the absence notes come last so they see every gap."""
    return [provenance(rec), rounds(rec, series), learning_details(rec), gates(rec),
            firings(health, rec), mirror(rec), lifecycle(rec), memory(rec), device_memory(rec),
            determinism(rec), alpha_full(rec), inventory(rec), absence_notes(gaps)]
