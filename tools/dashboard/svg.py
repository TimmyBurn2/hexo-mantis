"""Inline-SVG charts: geometry in the SVG, every label as HTML text that keeps its pixel size."""
from __future__ import annotations

from dataclasses import dataclass

from .envelope import envelope
from .fmt import esc, num

#: Chart width in viewBox units; also the number of pixel columns the envelope is bucketed to.
WIDTH = 640
PAD = 2.0


@dataclass(frozen=True)
class Series:
    label: str
    points: list[tuple[float, float]]
    cls: str = "s1"


@dataclass(frozen=True)
class QuantileWindow:
    x: float
    p10: float
    p50: float
    p90: float


@dataclass(frozen=True)
class Scale:
    x0: float
    x1: float
    y0: float
    y1: float
    height: float

    def x(self, value: float) -> float:
        span = (self.x1 - self.x0) or 1.0
        return PAD + (value - self.x0) / span * (WIDTH - 2 * PAD)

    def y(self, value: float) -> float:
        span = (self.y1 - self.y0) or 1.0
        return self.height - PAD - (value - self.y0) / span * (self.height - 2 * PAD)


def _scale(xs: list[float], ys: list[float], height: float,
           floor_zero: bool = False) -> Scale:
    y0, y1 = min(ys), max(ys)
    if floor_zero:
        y0 = min(0.0, y0)
    if y0 == y1:
        y0, y1 = y0 - 0.5, y1 + 0.5
    return Scale(min(xs), max(xs), y0, y1, height)


def _pt(x: float, y: float) -> str:
    return f"{x:.0f},{y:.1f}"


def figure(svg_body: str, scale: Scale, *, height: float, x_label: str, caption: str,
           legend: str = "", data: str = "") -> str:
    """The HTML around one drawing: y labels left, x labels below, the caption last."""
    y_mid = (scale.y0 + scale.y1) / 2
    x_mid = (scale.x0 + scale.x1) / 2
    return (
        '<figure class="chart">'
        f'<div class="y-labels"><span>{num(scale.y1)}</span><span>{num(y_mid)}</span>'
        f'<span>{num(scale.y0)}</span></div>'
        f'<svg viewBox="0 0 {WIDTH} {height:.0f}" preserveAspectRatio="xMidYMid meet" '
        f'role="img" data-x0="{scale.x0:g}" data-x1="{scale.x1:g}" data-y0="{scale.y0:g}" '
        f'data-y1="{scale.y1:g}"{data}>{svg_body}</svg>'
        f'<div class="x-labels"><span>{num(scale.x0)}</span><span>{esc(x_label)} {num(x_mid)}'
        f'</span><span>{num(scale.x1)}</span></div>'
        f'{legend}<figcaption>{caption}</figcaption></figure>'
    )


def too_few(label: str, count: int) -> str:
    return (f'<p class="absent">{esc(label)}: {count} point(s) — too few to draw a series; '
            'the value is in the table beside this panel.</p>')


def envelope_chart(series: list[Series], *, x_label: str, height: float = 140,
                   rules: list[tuple[float, str]] | None = None,
                   floor_zero: bool = False) -> str:
    """Each series as its per-pixel min–max band and its `last` line; extremes exact."""
    drawn = [s for s in series if len(s.points) >= 2]
    if not drawn:
        s = series[0] if series else Series("(none)", [])
        return too_few(s.label, len(s.points))
    envs = {s.label: envelope(s.points, width=WIDTH) for s in drawn}
    xs = [b.x for e in envs.values() for b in e]
    ys = [v for e in envs.values() for b in e for v in (b.lo, b.hi)]
    ys += [r[0] for r in (rules or [])]
    sc = _scale(xs, ys, height, floor_zero)
    body, captions, data = [], [], ""
    for s in drawn:
        buckets = envs[s.label]
        if any(b.lo != b.hi for b in buckets):
            top = " ".join(_pt(sc.x(b.x), sc.y(b.hi)) for b in buckets)
            bottom = " ".join(_pt(sc.x(b.x), sc.y(b.lo)) for b in reversed(buckets))
            body.append(f'<polygon class="band {s.cls}" points="{top} {bottom}"/>')
        line = " ".join(_pt(sc.x(b.x), sc.y(b.last)) for b in buckets)
        body.append(f'<polyline class="line {s.cls}" data-label="{esc(s.label)}" points="{line}"/>')
        full = [y for _, y in s.points]
        lo, hi, last = min(full), max(full), full[-1]
        captions.append(f'<b class="{s.cls}">{esc(s.label)}</b> min {num(lo)}, max {num(hi)}, '
                        f'last {num(last)} over {num(len(s.points))} points')
        if len(drawn) == 1:
            data = (f' data-min="{min(b.lo for b in buckets):g}" '
                    f'data-max="{max(b.hi for b in buckets):g}" data-last="{buckets[-1].last:g}"')
    for value, label in rules or []:
        y = sc.y(value)
        body.append(f'<line class="rule" x1="{PAD}" y1="{y:.1f}" x2="{WIDTH - PAD}" y2="{y:.1f}"/>')
        captions.append(f'rule: {esc(label)}')
    return figure("".join(body), sc, height=height, x_label=x_label,
                  caption=" · ".join(captions), data=data)


def quantile_chart(windows: list[QuantileWindow], *, x_label: str, label: str,
                   height: float = 140) -> str:
    """A p10–p90 band with the median as the line, one window per pixel column."""
    if len(windows) < 2:
        return too_few(label, len(windows))
    sc = _scale([w.x for w in windows], [w.p10 for w in windows] + [w.p90 for w in windows],
                height)
    top = " ".join(_pt(sc.x(w.x), sc.y(w.p90)) for w in windows)
    bottom = " ".join(_pt(sc.x(w.x), sc.y(w.p10)) for w in reversed(windows))
    line = " ".join(_pt(sc.x(w.x), sc.y(w.p50)) for w in windows)
    body = (f'<polygon class="band s1" points="{top} {bottom}"/>'
            f'<polyline class="line s1" data-label="{esc(label)} median" points="{line}"/>')
    last = windows[-1]
    caption = (f'<b class="s1">{esc(label)}</b> band p10–p90, line median · last window: p10 '
               f'{num(last.p10)}, median {num(last.p50)}, p90 {num(last.p90)} · '
               f'{len(windows)} windows')
    return figure(body, sc, height=height, x_label=x_label, caption=caption)


def bar_chart(points: list[tuple[float, float]], *, x_label: str, label: str,
              height: float = 120) -> str:
    """One bar per point, from zero; for sparse series such as rounds."""
    if len(points) < 2:
        return too_few(label, len(points))
    sc = _scale([x for x, _ in points], [y for _, y in points], height, floor_zero=True)
    w = max(2.0, (WIDTH - 2 * PAD) / (len(points) * 1.5))
    body = "".join(
        f'<rect class="bar s1" x="{sc.x(x) - w / 2:.1f}" y="{sc.y(y):.1f}" width="{w:.1f}" '
        f'height="{max(0.0, sc.y(0.0) - sc.y(y)):.1f}"/>' for x, y in points)
    ys = [y for _, y in points]
    caption = (f'<b class="s1">{esc(label)}</b> min {num(min(ys))}, max {num(max(ys))}, '
               f'last {num(ys[-1])} over {len(points)} bars')
    return figure(body, sc, height=height, x_label=x_label, caption=caption)


def tick_chart(events: list[tuple[float, str]], *, x_range: tuple[float, float],
               x_label: str, label: str, height: float = 40) -> str:
    """A tick per event on a shared x range; the class names the kind."""
    sc = Scale(x_range[0], x_range[1], 0.0, 1.0, height)
    body = "".join(
        f'<line class="tick {esc(kind)}" x1="{sc.x(x):.1f}" y1="{PAD}" x2="{sc.x(x):.1f}" '
        f'y2="{height - PAD:.1f}"/>' for x, kind in events)
    kinds: dict[str, int] = {}
    for _, kind in events:
        kinds[kind] = kinds.get(kind, 0) + 1
    caption = f'<b>{esc(label)}</b> ' + (", ".join(f"{esc(k)} ×{n}" for k, n in sorted(kinds.items()))
                                        or "none in this record")
    return figure(body, sc, height=height, x_label=x_label, caption=caption)
