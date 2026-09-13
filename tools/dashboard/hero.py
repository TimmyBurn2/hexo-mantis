"""Tier 1: the seven hero cells, each read from one row and each stating what it could not."""
from __future__ import annotations

from .fmt import esc, hours, num, pct, signed
from .health import HealthReading
from .model import GAP_MARKER, Gaps, HeroCell
from .reader import Record
from .stats import elo_of_wr
from .strength import RoundPoint, primary_rung, promotion_reading, trend


def strength_now(rec: Record, series: dict[str, list[RoundPoint]], gaps: Gaps) -> HeroCell:
    rung = primary_rung(rec)
    played = [p for p in series.get(rung, []) if not p.broken and p.wr is not None]
    label = f"Strength now vs {rung}"
    if not played:
        gaps.mark("Strength now", "no completed <code>eval_round_complete</code> row")
        return HeroCell(label, "—", GAP_MARKER, "absent")
    p = max(played, key=lambda q: q.step)
    wr = p.wr if p.wr is not None else 0.0
    parts = []
    if p.games:
        parts.append(f"{p.games} games")
    else:
        gaps.mark("Strength now", "games per rung are on the ladder file's row, not the round "
                  "event; without the ladder no Wilson interval can be formed")
        parts.append("n not on the round row — see notes")
    if p.wilson:
        parts.append(f"95 % Wilson {pct(p.wilson[0])}–{pct(p.wilson[1])}")
    if p.ci:
        parts.append(f"record's own CI {pct(p.ci[0])}–{pct(p.ci[1])}")
    parts.append(f"≈ {num(round(elo_of_wr(wr)))} Elo")
    parts.append(f"round {esc(p.round_id)} at step {num(p.step)}")
    return HeroCell(label, pct(wr, 2), " · ".join(parts), "neutral")


def trend_cell(rec: Record, series: dict[str, list[RoundPoint]], gaps: Gaps) -> HeroCell:
    label = "Trend, Elo / 1k steps"
    fit = trend(series.get(primary_rung(rec), []))
    completed = len([p for p in series.get(primary_rung(rec), []) if not p.broken])
    if fit is None:
        gaps.mark("Trend", f"the OLS Elo slope needs three completed rounds; {completed} in the record")
        return HeroCell(label, "—", f"{completed} rounds in window — {GAP_MARKER}", "absent")
    state = ("ok" if fit.lo > 0 else "bad") if fit.excludes_zero else "neutral"
    detail = (f"95 % CI {signed(fit.lo)} … {signed(fit.hi)} "
              f"({'excludes' if fit.excludes_zero else 'includes'} 0) · {fit.n} rounds in window "
              "(every completed round; R334(f) witness (iii) shape, 0 % rounds clamped by half a game)")
    return HeroCell(label, signed(fit.slope), detail, state)


def promotion_cell(rec: Record) -> HeroCell:
    decided, latest = promotion_reading(rec)
    if latest is None:
        return HeroCell("Last promotion decision", "no round", "no eval_round_complete row", "absent")
    detail = f"latest round {esc(latest.round_id)} at step {num(latest.step)}"
    if decided is not None:
        detail += f" · last decision: {decided.state} at step {num(decided.step)} ({esc(decided.round_id)})"
    else:
        detail += " · no round has taken a decision"
    state = {"promoted": "ok", "not promoted": "neutral"}.get(latest.state, "neutral")
    return HeroCell("Last promotion decision", latest.state, detail, state)


def progress_cell(rec: Record, gaps: Gaps) -> HeroCell:
    steps = [r["step"] for r in rec.events if isinstance(r.get("step"), int)]
    last_it = rec.last("iteration_complete")
    games = last_it.get("games_total") if last_it else None
    if games is None:
        gaps.mark("Progress", "games since boot reads <code>iteration_complete.games_total</code>; absent")
    wall = rec.wall_hours()
    if wall is None:
        gaps.mark("Progress", "wall hours need two timestamped rows; absent")
    detail = (f"{num(games) if games is not None else '—'} games since boot · "
              f"{hours(wall)} wall since the first event")
    return HeroCell("Progress", f"step {num(max(steps))}" if steps else "—", detail,
                    "neutral" if steps else "absent")


def health_cell(health: HealthReading) -> HeroCell:
    unmeasured = [i.name for i in health.inputs if i.state == "unmeasured"]
    value = health.state
    if health.state == "unmeasured":
        value = "unmeasured for " + ", ".join(unmeasured)
    reasons = "".join(f'<li class="{i.state}"><b>{esc(i.name)}</b>: {esc(i.reason)}</li>'
                      for i in health.inputs if i.state != "ok")
    if not reasons:
        reasons = "<li>every input read and clean</li>"
    return HeroCell("Health", value, f'<ul class="reasons">{reasons}</ul>', health.state)


def throughput_cell(rec: Record, gaps: Gaps) -> HeroCell:
    last = rec.last("iteration_complete")
    if last is None or not isinstance(last.get("games_per_hour"), (int, float)):
        gaps.mark("Throughput now", "<code>iteration_complete.games_per_hour</code> absent on the last row")
        return HeroCell("Throughput now", "—", GAP_MARKER, "absent")
    detail = (f"{num(last.get('steps_per_hour'))} steps / h · {num(last.get('positions_per_hour'))} "
              f"positions / h · {num(last.get('sims_per_sec'))} sims / s · at step {num(last.get('step'))}")
    return HeroCell("Throughput now", f"{num(last['games_per_hour'])} games / h", detail)


def losses_cell(rec: Record, gaps: Gaps) -> HeroCell:
    last = rec.last("trainer_step")
    if last is None:
        gaps.mark("Losses now", "no <code>trainer_step</code> row")
        return HeroCell("Losses now (value / policy)", "—", GAP_MARKER, "absent")
    detail = (f"total {num(last.get('loss'))} · grad norm {num(last.get('grad_norm'))} · "
              f"lr {num(last.get('lr'), 2)} · at step {num(last.get('step'))}")
    return HeroCell("Losses now (value / policy)",
                    f"{num(last.get('value_loss'))} / {num(last.get('policy_loss'))}", detail)


def cells(rec: Record, series: dict[str, list[RoundPoint]], health: HealthReading,
          gaps: Gaps) -> list[HeroCell]:
    """The seven cells in the contract's order."""
    return [strength_now(rec, series, gaps), trend_cell(rec, series, gaps), promotion_cell(rec),
            progress_cell(rec, gaps), health_cell(health), throughput_cell(rec, gaps),
            losses_cell(rec, gaps)]
