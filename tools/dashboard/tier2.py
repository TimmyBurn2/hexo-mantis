"""Tier 2: the chart panels, every series drawn through the per-pixel envelope."""
from __future__ import annotations

from .envelope import windows
from .external import external_panel
from .fmt import esc, num, pct
from .health import ply_cap_terms, ply_cap_windowed
from .ladder import ladder_chart
from .model import Gaps, Panel, no_rows
from .reader import Record
from .stats import quantile
from .strength import RUNG_RETIRED_NOTE, RoundPoint, sealbot_readings_present
from .svg import (
    WIDTH,
    QuantileWindow,
    Series,
    bar_chart,
    envelope_chart,
    quantile_chart,
    tick_chart,
)
from .throughput import throughput

_LOSS_KEYS = (("value_loss", "value loss", "s3"), ("policy_loss", "policy loss", "s2"),
              ("loss", "total loss", "s1"), ("grad_norm", "grad norm", "s4"), ("lr", "learning rate", "s6"))


#: The record carries no witness of which adapter played its rung, so the A/B's finding
#: is stated on every record (docs/design/measurements/SEALBOT_TT_AB_2026-09-14.md).
SEALBOT_TT_NOTE = (
    "Sealbot readings through the pre-748f5c47 adapter (its transposition table persisted across "
    "games, CARD-SEALBOT-TT-SEAT, R353(b)) transfer at ratio 1.00: the A/B on four cells and "
    "1 152 games read \u0394 0.000 / \u22120.007 / +0.028 / \u22120.014, every CI including 0 "
    "(SEALBOT_TT_AB_2026-09-14.md). The record does not say which adapter played its rung; the "
    "strix rung is unaffected."
)


def strength(series: dict[str, list[RoundPoint]], rec: Record, gaps: Gaps) -> Panel:
    reads = ("eval_ladder_state.json rungs[*].history (games, wr) joined by round index to "
             "eval_round_complete (step, promoted, games_total, wr_sealbot, wr_sealbot_ci_*)")
    if not rec.rows("eval_round_complete"):
        body = no_rows("eval_round_complete") + gaps.mark(
            "Strength ladder", "no <code>eval_round_complete</code> row, so no round is drawn")
        return Panel("Strength ladder", reads, body, "ladder", SEALBOT_TT_NOTE)
    if not sealbot_readings_present(rec):
        return Panel("Strength ladder", reads, gaps.mark("Strength ladder", RUNG_RETIRED_NOTE),
                     "ladder", SEALBOT_TT_NOTE)
    body = ladder_chart(series)
    if not rec.rungs():
        body += gaps.mark("Strength ladder",
                          f"the ladder file was not read ({esc(rec.ladder_note)}); games per "
                          "rung and the Wilson band come from it, so only the record's own "
                          "bootstrap CI is drawn")
    return Panel("Strength ladder", reads, body, "ladder", SEALBOT_TT_NOTE)


def losses(rec: Record, gaps: Gaps) -> Panel:
    reads = "trainer_step (step, value_loss, policy_loss, loss, grad_norm, lr)"
    if not rec.rows("trainer_step"):
        return Panel("Losses", reads, no_rows("trainer_step") + gaps.mark(
            "Losses", "no <code>trainer_step</code> row"), "losses")
    ceiling = _knob(rec, "monitor.alert_grad_norm_max")
    charts = []
    for key, label, cls in _LOSS_KEYS:
        pts = rec.series("trainer_step", "step", key)
        rules = [(ceiling, f"monitor.alert_grad_norm_max = {num(ceiling)}")] \
            if key == "grad_norm" and ceiling is not None else None
        if pts:
            charts.append(f'<div class="multiple"><h3>{esc(label)}</h3>'
                          + envelope_chart([Series(label, pts, cls)], x_label="step", height=110,
                                           rules=rules) + "</div>")
        else:
            charts.append(f'<div class="multiple"><h3>{esc(label)}</h3>' + gaps.mark(
                "Losses", f"<code>trainer_step.{key}</code> carries no finite value") + "</div>")
    note = "" if ceiling is not None else gaps.mark(
        "Losses", "the grad-norm abort ceiling <code>monitor.alert_grad_norm_max</code> is not "
        "among the knobs the <code>resolved_config</code> event carries, so no rule is drawn "
        "on the grad-norm chart")
    return Panel("Losses", reads, f'<div class="multiples">{"".join(charts)}</div>' + note, "losses")


def _knob(rec: Record, name: str) -> float | None:
    """A numeric knob off the `resolved_config` event, or `None` when it does not ride it."""
    row = rec.last("resolved_config")
    knob = ((row or {}).get("knobs") or {}).get(name)
    value = knob.get("value") if isinstance(knob, dict) else None
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _share(games: list[dict], key: str, width: int) -> list[tuple[float, float]]:
    pts = [(float(i), 1.0 if key == "cap" and g.get("terminal_reason") == "ply_cap"
            else 1.0 if key == "draw" and g.get("winner") == -1
            else 1.0 if key == "first" and g.get("winner") == 0 else 0.0)
           for i, g in enumerate(games, start=1)]
    return [(w.x, sum(w.values) / len(w.values)) for w in windows(pts, width=width)]


def quality(rec: Record, gaps: Gaps) -> Panel:
    reads = ("game_complete (moves = plies, winner, terminal_reason); x = game ordinal since boot; "
             "monitor_gates (ply_cap_abort_rate, ply_cap_window_games) for the halt's window")
    games = rec.rows("game_complete")
    if not games:
        marks = "".join(f'<div class="multiple"><h3>{esc(t)}</h3>' + gaps.mark(
            "Self-play quality", f"{t}: no <code>game_complete</code> row") + "</div>"
            for t in ("game length (plies)", "share ended by ply cap", "draw rate", "first-mover win rate",
                      "ply-cap share over the halt's window"))
        return Panel("Self-play quality", reads, no_rows("game_complete")
                     + f'<div class="multiples">{marks}</div>', "quality")
    lengths = [(float(i), float(g["moves"])) for i, g in enumerate(games, start=1)
               if isinstance(g.get("moves"), (int, float))]
    qwins = []
    for w in windows(lengths, width=WIDTH):
        vals = sorted(w.values)
        qwins.append(QuantileWindow(w.x, quantile(vals, 0.1), quantile(vals, 0.5), quantile(vals, 0.9)))
    charts = ['<div class="multiple"><h3>game length, plies</h3>'
              + quantile_chart(qwins, x_label="game", label="plies") + "</div>"]
    n = len(games)
    caps = sum(1 for g in games if g.get("terminal_reason") == "ply_cap")
    draws = sum(1 for g in games if g.get("winner") == -1)
    first = sum(1 for g in games if g.get("winner") == 0)
    for key, label, cls, total in (("cap", "share ended by ply cap", "s4", caps),
                                   ("draw", "draw rate", "s6", draws),
                                   ("first", "first-mover win rate", "s5", first)):
        chart = envelope_chart([Series(label, _share(games, key, WIDTH), cls)], x_label="game",
                               height=110, floor_zero=True)
        charts.append(f'<div class="multiple"><h3>{esc(label)}</h3>{chart}'
                      f'<p class="note">whole record: {num(total)} of {num(n)} games, {pct(total / n, 2)}</p></div>')
    charts.append(_ply_cap_window(rec, games, gaps))
    note = ('<p class="note">Windows are one pixel column of games each; the share is the mean '
            'over the window. <code>moves</code> counts plies (LAW-03); one compound turn is two.</p>')
    return Panel("Self-play quality", reads, f'<div class="multiples">{"".join(charts)}</div>' + note,
                 "quality")


def _ply_cap_window(rec: Record, games: list[dict], gaps: Gaps) -> str:
    """The ply-cap halt as the record reads it: the cap share over the halt's OWN window."""
    rate, window, armed = ply_cap_terms(rec)
    title = f"ply-cap share, {num(window)}-game window"
    terms = (f"halt rate {rate:g}" + ("" if armed else
             " (R352(c)'s minted terms — the halt was not armed in this run)"))
    series = ply_cap_windowed(games, window)
    if not series:
        mark = gaps.mark("Self-play quality",
                         f"{title}: {num(len(games))} games, fewer than the {num(window)}-game "
                         f"window, so no windowed rate exists yet ({terms})")
        return f'<div class="multiple"><h3>{esc(title)}</h3>{mark}<p class="note">{esc(terms)}</p></div>'
    chart = envelope_chart([Series("cap share", series, "s4")], x_label="game", height=110,
                           floor_zero=True, rules=[(rate, f"halt rate {rate:g}")])
    peak_x, peak = max(series, key=lambda p: p[1])
    return (f'<div class="multiple"><h3>{esc(title)}</h3>{chart}'
            f'<p class="note">{esc(terms)}; peak {peak:.2f} at game {peak_x:.0f}, last '
            f'{series[-1][1]:.2f} — the series the halt fires on when it exceeds the rate at or past '
            f'its step floor</p></div>')


def economy(rec: Record, gaps: Gaps) -> Panel:
    reads = "iteration_complete (step, buffer_size ÷ buffer_capacity); the rates are the Throughput panel's"
    if not rec.rows("iteration_complete"):
        return Panel("Data economy", reads, no_rows("iteration_complete") + gaps.mark(
            "Data economy", "no <code>iteration_complete</code> row"), "economy")
    charts = []
    fill = [(float(r["step"]), r["buffer_size"] / r["buffer_capacity"])
            for r in rec.rows("iteration_complete")
            if isinstance(r.get("step"), (int, float)) and isinstance(r.get("buffer_size"), (int, float))
            and isinstance(r.get("buffer_capacity"), (int, float)) and r["buffer_capacity"] > 0]
    charts.append('<div class="multiple"><h3>buffer fill (size ÷ capacity)</h3>'
                  + (envelope_chart([Series("buffer fill", fill, "s6")], x_label="step", height=110,
                                    floor_zero=True) if fill
                     else gaps.mark("Data economy", "buffer_size / buffer_capacity absent")) + "</div>")
    note = gaps.mark("Data economy", "replay ratio (samples consumed ÷ positions produced) needs "
                     "<code>samples_consumed_total</code> and <code>positions_produced_total</code> "
                     "on one <code>iteration_complete</code> row; neither is emitted")
    return Panel("Data economy", reads, f'<div class="multiples">{"".join(charts)}</div>' + note, "economy")


def health_timeline(rec: Record, gaps: Gaps) -> Panel:
    reads = ("eval_round_complete (step, wall_sec), disk_free (ts, disk_free_gb), training_alert (step), "
             "monitor_gates rows where a gate's fires rose, hard_abort / watchdog fires (step)")
    charts = []
    walls = [(float(r["step"]), float(r["wall_sec"])) for r in rec.rows("eval_round_complete")
             if isinstance(r.get("step"), (int, float)) and isinstance(r.get("wall_sec"), (int, float))]
    charts.append('<div class="multiple"><h3>eval round wall time, s</h3>'
                  + (bar_chart(walls, x_label="step", label="wall s") if len(walls) >= 2
                     else gaps.mark("Health timeline", "fewer than two rounds carry wall_sec")) + "</div>")
    stamps = [t for t in (r.get("ts") for r in rec.events) if isinstance(t, (int, float))]
    free = rec.series("disk_free", "ts", "disk_free_gb")
    if free and stamps:
        t0 = float(stamps[0])
        chart = envelope_chart([Series("disk free GB", [((t - t0) / 3600.0, gb) for t, gb in free], "s5")],
                               x_label="hours since first event", height=110, floor_zero=True)
    else:
        chart = gaps.mark("Health timeline", "no <code>disk_free</code> row")
    charts.append(f'<div class="multiple"><h3>disk free, GB</h3>{chart}</div>')
    ticks: list[tuple[float, str]] = [(float(a["step"]), "alert") for a in rec.rows("training_alert")
                                      if isinstance(a.get("step"), (int, float))]
    previous: dict[str, int] = {}
    for row in rec.rows("monitor_gates"):
        for name, s in (row.get("gates") or {}).items():
            fires = s.get("fires") if isinstance(s, dict) else None
            if isinstance(fires, int) and fires > previous.get(name, 0) and isinstance(row.get("step"), (int, float)):
                ticks.append((float(row["step"]), "fire"))
            if isinstance(fires, int):
                previous[name] = fires
    for name in ("hard_abort", "heartbeat_watchdog_fired", "selfplay_stall_watchdog"):
        ticks += [(float(r["step"]), "abort") for r in rec.rows(name) if isinstance(r.get("step"), (int, float))]
    steps = [float(r["step"]) for r in rec.events if isinstance(r.get("step"), (int, float))]
    if steps:
        chart = tick_chart(ticks, x_range=(0.0, max(steps)), x_label="step", label="alerts, gate fires, aborts")
    else:
        chart = gaps.mark("Health timeline", "no row carries a step, so the tick axis has no range")
    charts.append(f'<div class="multiple"><h3>ticks</h3>{chart}</div>')
    return Panel("Health timeline", reads, f'<div class="multiples">{"".join(charts)}</div>', "health")


def panels(rec: Record, series: dict[str, list[RoundPoint]], gaps: Gaps) -> list[Panel]:
    """The tier-2 roster in the contract's order."""
    return [strength(series, rec, gaps), external_panel(rec.external, rec.external_note, gaps),
            losses(rec, gaps), quality(rec, gaps), throughput(rec, gaps), economy(rec, gaps),
            health_timeline(rec, gaps)]
