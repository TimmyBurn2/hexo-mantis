"""The strength ladder chart: WR per rung over steps, bands, whiskers, markers, broken rounds."""
from __future__ import annotations

import math

from .fmt import esc, num, pct
from .strength import RoundPoint
from .svg import PAD, WIDTH, Scale, figure

_HEIGHT = 200.0
_CLASSES = ("s1", "s2", "s3", "s4", "s5", "s6")


def ladder_chart(series: dict[str, list[RoundPoint]]) -> str:
    """One figure for every rung; returns an absence paragraph when no rung has a point."""
    points = [p for pts in series.values() for p in pts]
    if len([p for p in points if not p.broken]) < 1:
        return ('<p class="absent">No completed round in this record, so no strength series '
                'is drawn.</p>')
    xs = [float(p.step) for p in points]
    tops = [p.wr or 0.0 for p in points] + [b[1] for p in points for b in (p.wilson, p.ci) if b]
    y_top = min(1.0, max(0.6, math.ceil((max(tops) + 0.05) * 10) / 10))
    sc = Scale(min(xs), max(xs), 0.0, y_top, _HEIGHT)
    if sc.x0 == sc.x1:
        sc = Scale(sc.x0 - 500.0, sc.x1 + 500.0, 0.0, y_top, _HEIGHT)
    body = [f'<line class="half" x1="{PAD}" y1="{sc.y(0.5):.1f}" x2="{WIDTH - PAD}" y2="{sc.y(0.5):.1f}"/>']
    legend = []
    for (name, pts), cls in zip(series.items(), _CLASSES * 3, strict=False):
        played = [p for p in pts if not p.broken and p.wr is not None]
        banded = [p for p in played if p.wilson]
        if len(banded) >= 2:
            top = " ".join(f"{sc.x(p.step):.0f},{sc.y(p.wilson[1]):.1f}" for p in banded if p.wilson)
            bottom = " ".join(f"{sc.x(p.step):.0f},{sc.y(p.wilson[0]):.1f}"
                              for p in reversed(banded) if p.wilson)
            body.append(f'<polygon class="band {cls}" points="{top} {bottom}"/>')
        if len(played) >= 2:
            line = " ".join(f"{sc.x(p.step):.0f},{sc.y(p.wr):.1f}" for p in played if p.wr is not None)
            body.append(f'<polyline class="line {cls}" data-label="{esc(name)} win rate" points="{line}"/>')
        for p in played:
            x, y = sc.x(p.step), sc.y(p.wr if p.wr is not None else 0.0)
            if p.ci:
                body.append(f'<line class="whisker {cls}" x1="{x:.1f}" y1="{sc.y(p.ci[0]):.1f}" '
                            f'x2="{x:.1f}" y2="{sc.y(p.ci[1]):.1f}"/>')
            kind = "promoted" if p.promoted is True else "rejected" if p.promoted is False else "point"
            body.append(f'<circle class="marker {kind}" cx="{x:.1f}" cy="{y:.1f}" r="5"/>')
        legend.append(f'<li><i class="swatch {cls}"></i>{esc(name)}: line = win rate per round, '
                      "band = 95 % Wilson from (wr, games) of the ladder row"
                      + (", whiskers = the record's own bootstrap CI" if any(p.ci for p in played) else "")
                      + "</li>")
    broken = [p for p in points if p.broken]
    for p in broken:
        body.append(f'<circle class="marker broken" cx="{sc.x(p.step):.1f}" cy="{sc.y(0.0):.1f}" r="5"/>')
    legend.append('<li><i class="swatch promoted"></i>promoted · <i class="swatch rejected"></i>'
                  'not promoted · plain = no decision taken</li>')
    if broken:
        legend.append(f'<li><i class="swatch broken"></i>broken round (games_total: null), '
                      f'{len(broken)} — drawn on the axis, not omitted</li>')
    last = [p for p in points if not p.broken and p.wr is not None]
    last_p = max(last, key=lambda p: p.step)
    caption = (f"last completed round {esc(last_p.round_id)} at step {num(last_p.step)}: "
               f"{pct(last_p.wr)}" + (f" over {last_p.games} games" if last_p.games else "")
               + f" · {len(last)} completed, {len(broken)} broken · the dashed rule is 50 %")
    return figure("".join(body), sc, height=_HEIGHT, x_label="step", caption=caption,
                  legend=f'<ul class="legend">{"".join(legend)}</ul>')
