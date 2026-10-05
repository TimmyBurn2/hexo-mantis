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
    """One legend entry: its label, colour class, glyph (`line`, `dash`, `dot`, `band`, `mark`) and an off-reason."""

    label: str
    cls: str = "cf"
    glyph: str = "line"
    off: str = ""


def legend(keys: Sequence[Key]) -> str:
    if not keys:
        return ""
    parts = []
    for k in keys:
        glyph = {"line": "", "dash": "dash", "dot": "dot", "band": "band", "mark": "mk"}[k.glyph]
        off = f" — {esc(k.off)}" if k.off else ""
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


def figure(title: str, chart: Chart | None, definition: str, *, now: str = "", keys: Sequence[Key] = (),
           twin: str = "", gap: tuple[str, str] | None = None) -> str:
    """One figure; a `gap` replaces the plot with its two sentences, never an empty axis."""
    if gap is not None:
        plot = f'<div class="gap"><strong>{esc(gap[0])}</strong>{esc(gap[1])}</div>'
    elif chart is None:
        plot = '<div class="gap"><strong>Nothing to draw.</strong>No series reached this chart.</div>'
    else:
        plot = chart.render()
    return (f'<figure class="chart"><div class="head"><h3>{esc(title)}</h3><span class="now">{esc(now)}</span></div>'
            f'{legend(keys)}<div class="plot">{plot}</div><figcaption>{esc(definition)}</figcaption>{twin}</figure>')


def smoothed(name: str, cls: str, pairs: Sequence[tuple[float, float]], *, band: bool) -> tuple[Line, Band | None, float]:
    """A noisy per-step series as its bucket means smoothed, the buckets' min–max range, and the median half-range."""
    buckets = bucket(pairs)
    line = Line(name, cls, list(zip([b[0] for b in buckets], smooth([b[1] for b in buckets]), strict=True)))
    rng = Band(cls, [(b[0], b[2], b[3]) for b in buckets]) if band else None
    return line, rng, median_half_range(buckets)
