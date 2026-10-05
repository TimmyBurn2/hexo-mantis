"""The figure around a chart: title and latest value, legend, the plot, the definition line, a table twin; or a stated gap."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .fmt import esc
from .svg import Band, Chart, Line, bucket, median_half_range, smooth

#: Colour classes by a run's place on the page: the first run blue, the second orange, the third aqua.
RUN_CLASSES = ("c1", "c2", "c3")


@dataclass(frozen=True)
class Key:
    """One legend entry: its label, colour class, glyph (`line`, `dash`, `dot`, `band`, `mark`, `faint`) and an off-reason."""

    label: str
    cls: str = "cf"
    glyph: str = "line"
    off: str = ""


@dataclass(frozen=True)
class Goal:
    """Which way a chart is good: `down`, `up`, `near` a target, or `watch` (no better way), and the words for it."""

    way: str
    words: str


LOWER = Goal("down", "lower is better")
HIGHER = Goal("up", "higher is better")
_GLYPH = {"down": "↓", "up": "↑", "near": "◎", "watch": "∿"}


def legend(keys: Sequence[Key]) -> str:
    """The legend row; an entry with an off-reason is drawn disabled with the reason beside it."""
    if not keys:
        return ""
    parts = []
    for k in keys:
        glyph = {"line": "", "dash": "dash", "dot": "dot", "band": "band", "mark": "mk", "faint": "faint"}[k.glyph]
        off = f" ({esc(k.off)})" if k.off else ""
        parts.append(f'<span class="{k.cls}{" off" if k.off else ""}"><i class="{glyph}"></i>'
                     f'<span class="lbl">{esc(k.label)}{off}</span></span>')
    return f'<div class="legend">{"".join(parts)}</div>'


def table(head: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """A table twin, closed by default; cells arrive already formatted and are escaped here."""
    if not rows:
        return ""
    th = "".join(f"<th>{esc(h)}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f'<details><summary>Table</summary><table class="data"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></details>'


def figure(title: str, chart: Chart | None, definition: str = "", *, goal: Goal | None = None, now: str = "",
           nows: Sequence[tuple[str, str]] = (), only: str = "", keys: Sequence[Key] = (), twin: str = "",
           gap: tuple[str, str] | None = None) -> str:
    """One figure; `nows` gives each compared run's latest value, `only` names the one run a panel draws, a `gap` replaces the plot."""
    if gap is not None:
        plot = f'<div class="gap"><strong>{esc(gap[0])}</strong>{esc(gap[1])}</div>'
    elif chart is None:
        plot = '<div class="gap"><strong>Nothing to draw.</strong>No series reached this chart.</div>'
    else:
        plot = chart.render()
    scope = f'<span class="only">Only {esc(only)} is drawn.</span>' if only else ""
    aim = (f'<p class="goal {goal.way}"><span class="g" aria-hidden="true">{_GLYPH[goal.way]}</span>{esc(goal.words)}</p>'
           if goal is not None else "")
    caption = f"<figcaption>{esc(definition)}{' ' if definition and only else ''}{scope}</figcaption>" if definition or only else ""
    many = len(nows) > 1
    latest = "".join(f'<b class="{cls}">{esc(v)}</b>' for v, cls in nows if v) if many else esc(now)
    return (f'<figure class="chart"><div class="head{" many" if many else ""}"><h3 title="{esc(title)}">{esc(title)}</h3>'
            f'<span class="now">{latest}</span></div>'
            f'{aim}{legend(keys)}<div class="plot">{plot}</div>{caption}{twin}</figure>')


def smoothed(name: str, cls: str, pairs: Sequence[tuple[float, float]], *, band: bool) -> tuple[Line, Band | None, float]:
    """A noisy per-step series as its bucket means smoothed, the buckets' min–max range, and the median half-range."""
    buckets = bucket(pairs)
    line = Line(name, cls, list(zip([b[0] for b in buckets], smooth([b[1] for b in buckets]), strict=True)))
    rng = Band(cls, [(b[0], b[2], b[3]) for b in buckets]) if band else None
    return line, rng, median_half_range(buckets)
