"""The chart primitives: one axis, 2 px lines over a faint range, whiskered points, reference bands; drawn server-side as SVG."""
from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .fmt import esc, num, script_json, short, sig

Fmt = Callable[[float], str]
_M = {"l": 40.0, "r": 10.0, "t": 8.0, "b": 22.0}


@dataclass(frozen=True)
class Line:
    """A polyline in a run's or engine's colour class; `faint` draws it thin and dimmed (an earlier window)."""

    name: str
    cls: str
    pts: Sequence[tuple[float, float]]
    dash: bool = False
    faint: bool = False


@dataclass(frozen=True)
class Band:
    """A filled range `(x, lo, hi)` under a line, at 10 % opacity."""

    cls: str
    pts: Sequence[tuple[float, float, float]]


@dataclass(frozen=True)
class Dots:
    """Points with 95 % whiskers `(x, y, lo, hi)`; `dx` nudges them sideways so two series at one step stay apart."""

    name: str
    cls: str
    pts: Sequence[tuple[float, float, float | None, float | None]]
    dx: float = 0.0


@dataclass(frozen=True)
class Ref:
    """A horizontal reference: a line at `y`, an optional band `lo..hi`, a label; `marked` draws it dashed in the warn tone."""

    y: float
    lo: float | None = None
    hi: float | None = None
    label: str = ""
    marked: bool = False
    end: bool = False


@dataclass(frozen=True)
class Chart:
    """One chart's marks and axes; `render()` returns the SVG and the crosshair's data."""

    title: str
    lines: Sequence[Line] = ()
    bands: Sequence[Band] = ()
    dots: Sequence[Dots] = ()
    refs: Sequence[Ref] = ()
    marks: Sequence[tuple[float, str]] = ()
    x_domain: tuple[float, float] | None = None
    y_domain: tuple[float, float] | None = None
    y_floor: float | None = None
    y_ceil: float | None = None
    y_fmt: Fmt = sig
    x_fmt: Fmt = short
    x_name: str = "step"
    x_unit: str = "steps"
    x_title: str = ""
    width: float = 250.0
    height: float = 140.0
    reach: float = 0.0
    extra: list[str] = field(default_factory=list)

    def domains(self) -> tuple[float, float, float, float]:
        xs = [p[0] for s in self.lines for p in s.pts] + [p[0] for d in self.dots for p in d.pts]
        x0, x1 = self.x_domain or ((min(xs), max(xs)) if xs else (0.0, 1.0))
        if self.y_domain is not None:
            return x0, x1, *self.y_domain
        ys = [p[1] for s in self.lines for p in s.pts] + [v for d in self.dots for p in d.pts for v in p[1:]
                                                           if v is not None]
        ys += [v for r in self.refs for v in (r.y, r.lo, r.hi) if v is not None]
        y0, y1 = (min(ys) - self.reach, max(ys) + self.reach) if ys else (0.0, 1.0)
        pad = (y1 - y0) * 0.08 or 0.05
        y0, y1 = y0 - pad, y1 + pad
        if self.y_floor is not None:
            y0 = max(self.y_floor, y0)
        if self.y_ceil is not None:
            y1 = min(self.y_ceil, y1)
        return x0, x1, y0, y1

    def render(self) -> str:
        """The SVG followed by the crosshair's JSON (values only; the browser maps them, it never derives them)."""
        w, h = self.width, self.height
        m = {**_M, "b": _M["b"] + (13.0 if self.x_title else 0.0)}
        iw, ih = w - m["l"] - m["r"], h - m["t"] - m["b"]
        x0, x1, y0, y1 = self.domains()

        def px(v: float) -> float:
            return _M["l"] + (v - x0) / ((x1 - x0) or 1.0) * iw

        def py(v: float) -> float:
            return _M["t"] + (1 - (v - y0) / ((y1 - y0) or 1.0)) * ih

        clip = "k" + hashlib.sha1(repr((self.title, x0, x1, y0, y1)).encode()).hexdigest()[:10]
        wide = ' class="wide"' if w > 300 else ""
        out = [f'<svg viewBox="0 0 {w:.0f} {h:.0f}"{wide} role="img" aria-label="{esc(self.title)}">'
               f'<defs><clipPath id="{clip}"><rect x="{_M["l"]}" y="{_M["t"]}" width="{iw:.1f}" height="{ih:.1f}"/>'
               "</clipPath></defs>"]
        yt = ticks(y0, y1, 4)
        y_text = shared_decimals(yt) if self.y_fmt is sig else [self.y_fmt(v) for v in yt]
        for v, label in zip(yt, y_text, strict=True):
            out.append(f'<line class="gridline" x1="{_M["l"]}" x2="{w - _M["r"]}" y1="{py(v):.1f}" y2="{py(v):.1f}"/>'
                       f'<text class="tick" x="{_M["l"] - 7}" y="{py(v) + 3.5:.1f}" text-anchor="end">{esc(label)}</text>')
        for r in self.refs:
            # A dashed rule line carries its value on the axis when no tick already sits there.
            if r.marked and y0 <= r.y <= y1 and all(abs(py(r.y) - py(v)) > 9 for v in yt):
                label = shared_decimals([*yt, r.y])[-1] if self.y_fmt is sig else self.y_fmt(r.y)
                out.append(f'<text class="tick ref" x="{_M["l"] - 7}" y="{py(r.y) + 3.5:.1f}" text-anchor="end">{esc(label)}</text>')
        out.append(f'<line class="axisline" x1="{_M["l"]}" x2="{w - _M["r"]}" y1="{_M["t"] + ih:.1f}" y2="{_M["t"] + ih:.1f}"/>')
        xt = ticks(x0, x1, 4)
        for i, v in enumerate(xt):
            last = i == len(xt) - 1 and px(v) > w - _M["r"] - 40
            label = f"{self.x_fmt(v)} {self.x_unit}" if i == len(xt) - 1 and self.x_unit else self.x_fmt(v)
            out.append(f'<text class="tick" x="{w - _M["r"] if last else px(v):.1f}" y="{h - 6 - (m["b"] - _M["b"]):.1f}" '
                       f'text-anchor="{"end" if last else "middle"}">{esc(label)}</text>')
        if self.x_title:
            out.append(f'<text class="tick" x="{_M["l"] + iw / 2:.1f}" y="{h - 3:.1f}" text-anchor="middle">{esc(self.x_title)}</text>')
        out.append(f'<g clip-path="url(#{clip})">')
        for r in self.refs:
            if r.lo is not None and r.hi is not None:
                out.append(f'<rect class="refband" x="{_M["l"]}" width="{iw:.1f}" y="{py(r.hi):.1f}" '
                           f'height="{max(1.0, py(r.lo) - py(r.hi)):.1f}"/>')
            out.append(f'<line class="{"refmark" if r.marked else "ref"}" x1="{_M["l"]}" x2="{w - _M["r"]}" '
                       f'y1="{py(r.y):.1f}" y2="{py(r.y):.1f}"/>')
        for b in self.bands:
            if b.pts:
                up = " ".join(f"{px(x):.1f},{py(hi):.1f}" for x, _lo, hi in b.pts)
                dn = " ".join(f"{px(x):.1f},{py(lo):.1f}" for x, lo, _hi in reversed(b.pts))
                out.append(f'<polygon class="band {b.cls}" points="{up} {dn}"/>')
        for s in self.lines:
            if len(s.pts) > 1:
                pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in s.pts)
                cls = f"ln {s.cls}" + (" dash" if s.dash else "") + (" faint" if s.faint else "")
                out.append(f'<polyline class="{cls}" points="{pts}"/>')
            elif s.pts:
                out.append(f'<circle class="dot {s.cls}" cx="{px(s.pts[0][0]):.1f}" cy="{py(s.pts[0][1]):.1f}" r="3.5"/>')
        out.append("</g>")
        for r in self.refs:
            if r.label and r.end:
                out.append(f'<text class="reflabel" x="{w - _M["r"]:.1f}" y="{py(r.y) - 5:.1f}" text-anchor="end">'
                           f'{esc(r.label)}</text>')
            elif r.label:
                out.append(f'<text class="reflabel" x="{_M["l"] + 6}" y="{py(r.y) + 11:.1f}">{esc(r.label)}</text>')
        for d in self.dots:
            for x, y, lo, hi in d.pts:
                cx = px(x) + d.dx
                if lo is not None and hi is not None:
                    out.append(f'<line class="whisk {d.cls}" x1="{cx:.1f}" x2="{cx:.1f}" y1="{py(lo):.1f}" y2="{py(hi):.1f}"/>')
                out.append(f'<circle class="dot {d.cls}" cx="{cx:.1f}" cy="{py(y):.1f}" r="4.5"/>')
        for x, label in self.marks:
            out.append(f'<path class="mark" d="M{px(x) - 4:.1f},{_M["t"] + ih:.1f} l4,-6 l4,6z"><title>{esc(label)}</title></path>')
        out.extend(self.extra)
        out.append("</svg>")
        series = [{"name": s.name, "cls": s.cls, "pts": [[_r(x), _r(y)] for x, y in s.pts]} for s in self.lines if not s.faint]
        series += [{"name": d.name, "cls": d.cls, "dots": True, "pts": [[_r(v) for v in p] for p in d.pts]} for d in self.dots]
        data = {"x": [x0, x1], "y": [y0, y1], "w": w, "h": h, "m": m, "xname": self.x_name, "series": series}
        blob = script_json(data)
        return "".join(out) + f'<script type="application/json" class="xh">{blob}</script>'


def shared_decimals(values: Sequence[float]) -> list[str]:
    """Axis labels at the fewest shared decimals that print every tick exactly: 0 / 5 / 10, 0.025 / 0.050."""
    places = next((p for p in range(7) if all(abs(round(v, p) - v) < 1e-9 for v in values)), 6)
    return [num(v, places) for v in values]


def _r(v: float | None) -> float | None:
    """Five significant figures: the readout's precision, not the record's."""
    return None if v is None else float(f"{v:.5g}")


def ticks(lo: float, hi: float, n: int) -> list[float]:
    """Round tick values covering `lo..hi`, about `n` of them and never fewer than two."""
    span = (hi - lo) or abs(hi) or 1.0
    mag = 10 ** math.floor(math.log10(span / n))
    steps = [m * mag for m in (0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10)]
    i = next(k for k, step in enumerate(steps) if step >= span / n)
    while True:
        step = steps[i]
        first = math.ceil(lo / step - 1e-9) * step
        out = [round(first + k * step, 12) for k in range(int((hi - first) / step + 1e-9) + 1)]
        if len(out) >= 2 or i == 0:
            return out
        i -= 1


def bucket(pairs: Sequence[tuple[float, float]], n: int = 120) -> list[tuple[float, float, float, float]]:
    """`(x, mean, min, max)` per bucket of equal x width; an empty bucket is absent, never a zero."""
    if not pairs:
        return []
    x0, x1 = pairs[0][0], pairs[-1][0]
    width = ((x1 - x0) / n) or 1.0
    acc: dict[int, list[float]] = {}
    xs: dict[int, list[float]] = {}
    for x, y in pairs:
        k = min(n - 1, int((x - x0) / width))
        acc.setdefault(k, []).append(y)
        xs.setdefault(k, []).append(x)
    return [(sum(xs[k]) / len(xs[k]), sum(v) / len(v), min(v), max(v)) for k, v in sorted(acc.items())]


def smooth(values: Sequence[float], k: int = 5) -> list[float]:
    """A centred rolling mean over `k` buckets, shrinking at the ends."""
    half = k // 2
    return [sum(values[max(0, i - half):i + half + 1]) / len(values[max(0, i - half):i + half + 1])
            for i in range(len(values))]


def median_half_range(buckets: Sequence[tuple[float, float, float, float]]) -> float:
    """The median of half of each bucket's range: the y axis fits the line plus this, so a spike clips."""
    halves = sorted((b[3] - b[2]) / 2 for b in buckets)
    return halves[len(halves) // 2] if halves else 0.0
