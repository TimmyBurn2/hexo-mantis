"""Throughput: games/h, plies/h, turns/h, leaves/s and steps/h side by side, a unit on every label and axis."""
from __future__ import annotations

from typing import Any

from .fmt import esc, num
from .model import Gaps, Panel, no_rows
from .reader import Record
from .svg import WIDTH, Series, envelope_chart

#: `iteration_complete` rates on the step axis, each with the unit its producer measures in.
RATE_KEYS = (
    ("games_per_hour", "games / h", "s1"),
    ("positions_per_hour", "turns / h (positions_per_hour: compound turns, ≈ 2 plies each)", "s2"),
    ("sims_per_sec", "leaves / s (sims_per_sec: positions × effective sims per move, billed)", "s5"),
    ("steps_per_hour", "steps / h", "s3"),
)
PLIES_LABEL = "plies / h (Σ game_complete.moves per window ÷ the window's wall)"
#: A window narrower than this many games has a wall span of a few seconds; the rate is noise.
MIN_GAMES_PER_WINDOW = 8


def plies_per_hour(games: list[dict[str, Any]], *, width: int = WIDTH) -> list[tuple[float, float]]:
    """Per window of games (x = game ordinal): Σ `moves` (plies) ÷ the window's own wall span in hours."""
    rows = [(i, float(g["ts"]), float(g["moves"])) for i, g in enumerate(games, start=1)
            if isinstance(g.get("ts"), (int, float)) and isinstance(g.get("moves"), (int, float))]
    if not rows:
        return []
    n = len(games)
    windows = max(1, min(width, n // MIN_GAMES_PER_WINDOW))
    columns: dict[int, list[tuple[float, float]]] = {}
    for ordinal, ts, plies in rows:
        col = min(int((ordinal - 1) / n * windows), windows - 1)
        columns.setdefault(col, []).append((ts, plies))
    out: list[tuple[float, float]] = []
    for col in sorted(columns):
        span_h = (max(ts for ts, _ in columns[col]) - min(ts for ts, _ in columns[col])) / 3600.0
        # One timestamp shared by every game of the window is no span: skipped, never ÷ 0.
        if span_h > 0.0:
            out.append((float((col + 1) * n / windows), sum(p for _, p in columns[col]) / span_h))
    return out


def _plies_chart(rec: Record, gaps: Gaps) -> str:
    games = rec.rows("game_complete")
    series = plies_per_hour(games) if games else []
    if len(series) < 2:
        return f'<div class="multiple"><h3>{esc(PLIES_LABEL)}</h3>' + gaps.mark(
            "Throughput", "plies / h needs two windows of <code>game_complete</code> rows with "
            f"<code>ts</code> and <code>moves</code>; {num(len(games))} game(s) in the record") + "</div>"
    chart = envelope_chart([Series("plies / h", series, "s4")], x_label="game", height=110,
                           floor_zero=True)
    return (f'<div class="multiple"><h3>{esc(PLIES_LABEL)}</h3>{chart}'
            '<p class="note"><code>moves</code> counts PLIES — one compound turn places two stones '
            "(LAW-03); x = game ordinal since boot, the window one pixel column of games.</p></div>")


def _rate_chart(rec: Record, key: str, label: str, cls: str, gaps: Gaps) -> str:
    pts = rec.series("iteration_complete", "step", key)
    chart = (envelope_chart([Series(label.split(" (")[0], pts, cls)], x_label="step", height=110,
                            floor_zero=True) if len(pts) >= 2
             else gaps.mark("Throughput", f"<code>iteration_complete.{key}</code> carries fewer than "
                            "two finite values"))
    return f'<div class="multiple"><h3>{esc(label)}</h3>{chart}</div>'


def sym_draw_line(rec: Record, gaps: Gaps) -> str:
    """The D6 draw bins off the last `iteration_complete.sym_draws`, as a share per element."""
    rows = [r for r in rec.rows("iteration_complete") if isinstance(r.get("sym_draws"), dict)]
    block: dict[str, Any] = rows[-1]["sym_draws"] if rows else {}
    bins = block.get("bins")
    total = sum(bins) if isinstance(bins, list) and bins and all(isinstance(b, int) for b in bins) else 0
    if not bins or total == 0:
        return gaps.mark("Throughput", "<code>iteration_complete.sym_draws</code> carries no draw: the "
                         "augmentation counter has no producer on this record, or the run has not sampled yet")
    shares = " · ".join(f"{i}: {b / total * 100:.1f} %" for i, b in enumerate(bins))
    return (f'<p class="note"><code>iteration_complete.sym_draws</code> — D6 element share of '
            f"{num(total)} draws since boot (uniform = {100 / len(bins):.1f} % each; bin 0 is the identity, "
            f"all of it under <code>train.augment: false</code>): {esc(shares)}; "
            f"{num(block.get('empty_skipped'))} empty-board rows left unrotated.</p>")


def throughput(rec: Record, gaps: Gaps) -> Panel:
    """The five rates side by side; every label carries its unit and its producer, every chart its axis."""
    reads = ("iteration_complete (step, games_per_hour, positions_per_hour, sims_per_sec, steps_per_hour, "
             "sym_draws); game_complete (ts, moves) for plies / h")
    if not rec.rows("iteration_complete") and not rec.rows("game_complete"):
        return Panel("Throughput", reads, no_rows("iteration_complete") + gaps.mark(
            "Throughput", "no <code>iteration_complete</code> and no <code>game_complete</code> row"),
            "throughput")
    games, rest = RATE_KEYS[0], RATE_KEYS[1:]
    charts = [_rate_chart(rec, *games, gaps), _plies_chart(rec, gaps)]
    charts += [_rate_chart(rec, key, label, cls, gaps) for key, label, cls in rest]
    note = ("Games / h understates the work when games lengthen (run7: 60 → 73 plies over the run); "
            "plies / h and leaves / s do not. leaves / s is the pool's bill — positions × the config's "
            "effective sims per move over the drain interval — not a served-leaf count.")
    return Panel("Throughput", reads, f'<div class="multiples">{"".join(charts)}</div>'
                 f'<p class="note">{esc(note)}</p>{sym_draw_line(rec, gaps)}', "throughput")
