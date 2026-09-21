"""External points (R356(d)): the strix follower's sidecars as a series with CIs, unit and regime on the axis, the gap to strix as a number."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .fmt import esc, num, pct
from .model import Gaps, Panel, table
from .stats import elo_of_wr
from .svg import PAD, WIDTH, Scale, figure

#: The sidecar suffixes the follower writes: `<ckpt>.strix256.json` (equal-work), `.strix512.json`, `.strix256_nosolver.json`.
SIDECAR_GLOB = "*.strix*.json"
_HEIGHT = 200.0
_CLASSES = ("s2", "s3", "s4", "s5", "s6", "s1")
PRODUCER = ("tools/strix_follower.py sidecars (<ckpt>.strix256.json per cadence checkpoint and "
            "promotion; .strix512.json for the as-shipped cell)")


@dataclass(frozen=True)
class ExternalPoint:
    """One sidecar: the checkpoint's step and net, the unit it was read in, its regime, its reading."""

    run_id: str
    step: int
    unit: str
    ours_sims: int
    strix_sims: int
    regime: str
    trigger: str
    wr: float
    ci: tuple[float, float] | None
    eff_n: int | None
    games: int | None
    net_hash: str
    checkpoint: str
    path: str
    #: `"off"` when the sidecar says strix's root VCF solver was disabled (R358(a)'s net-only cell).
    solver: str = "on"
    #: strix's placement_radius when the sidecar names one (R365 E1's ruler-r6 cell); None = the driver's 8.
    radius: int | None = None

    @property
    def unit_label(self) -> str:
        label = f"{self.run_id} · {self.unit}: ours PUCT-{self.ours_sims} vs strix {self.strix_sims} sims"
        label += ", solver OFF" if self.solver == "off" else ""
        return label + ("" if self.radius is None else f", strix @ r{self.radius}")


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _int(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def parse_sidecar(path: Path, raw: Any) -> ExternalPoint | None:
    """A point from one sidecar's JSON, or `None` when a field the series needs is absent."""
    if not isinstance(raw, dict):
        return None
    step, wr = _int(raw.get("step")), _num(raw.get("wr"))
    ours, strix = raw.get("ours") or {}, raw.get("strix") or {}
    ours_sims, strix_sims = _int(ours.get("sims")), _int(strix.get("sims"))
    if step is None or wr is None or ours_sims is None or strix_sims is None:
        return None
    lo, hi = _num(raw.get("wr_ci_lower")), _num(raw.get("wr_ci_upper"))
    return ExternalPoint(
        run_id=str(raw.get("run_id", "?")), step=step, unit=str(raw.get("unit", "?")),
        ours_sims=ours_sims, strix_sims=strix_sims,
        regime=str(raw.get("regime", "?")), trigger=str(raw.get("trigger", "?")), wr=wr,
        ci=(lo, hi) if lo is not None and hi is not None else None,
        eff_n=_int(raw.get("eff_n")), games=_int(raw.get("games")),
        net_hash=str(raw.get("net_hash", "?")), checkpoint=str(raw.get("checkpoint", path.name)),
        path=str(path), solver=str(strix.get("solver", "on")), radius=_int(strix.get("radius")),
    )


def load_external_points(specs: list[Path] | None) -> tuple[list[ExternalPoint], str]:
    """Every sidecar under each spec (a directory, or one file) as points, plus a note on what was skipped."""
    if not specs:
        return [], "no --external-points given"
    paths: list[Path] = []
    for spec in specs:
        paths += sorted(spec.rglob(SIDECAR_GLOB)) if spec.is_dir() else [spec] if spec.is_file() else []
    if not paths:
        return [], f"{', '.join(str(s) for s in specs)} holds no {SIDECAR_GLOB} sidecar"
    points: list[ExternalPoint] = []
    skipped: list[str] = []
    for path in paths:
        if path.name.endswith(".failed.json"):
            skipped.append(f"{path.name} (a failed cell, not a receipt)")
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            skipped.append(f"{path.name} ({exc.__class__.__name__})")
            continue
        point = parse_sidecar(path, raw)
        if point is None:
            skipped.append(f"{path.name} (missing step, wr or the unit's sims)")
            continue
        points.append(point)
    note = f"{len(points)} sidecar(s) read"
    if skipped:
        note += "; skipped " + ", ".join(skipped)
    return sorted(points, key=lambda p: (p.run_id, p.unit, p.step)), note


def series_by_unit(points: list[ExternalPoint]) -> dict[str, list[ExternalPoint]]:
    """Points grouped by unit label, each unit its own instrument — never merged."""
    out: dict[str, list[ExternalPoint]] = {}
    for p in points:
        out.setdefault(p.unit_label, []).append(p)
    return {k: sorted(v, key=lambda p: p.step) for k, v in out.items()}


def gap_statement(p: ExternalPoint) -> str:
    """The gap to strix as a NUMBER: percentage points below parity and the Elo it implies."""
    below = (0.5 - p.wr) * 100.0
    sign = "below" if below >= 0 else "above"
    ci = f", CI {pct(p.ci[0])}–{pct(p.ci[1])}" if p.ci else ""
    return (f"step {num(p.step)}: WR {pct(p.wr)} vs strix{ci} — {abs(below):.1f} pp {sign} parity "
            f"(≈ {num(round(elo_of_wr(p.wr)))} Elo), {p.regime}, n = {num(p.eff_n)}")


def external_chart(units: dict[str, list[ExternalPoint]]) -> str:
    """One figure, one series per unit: whiskers = the pair-level CI, marker filled when CONTENDED."""
    points = [p for pts in units.values() for p in pts]
    xs = [float(p.step) for p in points]
    tops = [p.wr for p in points] + [p.ci[1] for p in points if p.ci]
    # Parity stays on the chart: the gap to strix is the distance to the dashed rule.
    y_top = min(1.0, max(0.55, round(max(tops) + 0.05, 1)))
    sc = Scale(min(xs), max(xs), 0.0, y_top, _HEIGHT)
    if sc.x0 == sc.x1:
        sc = Scale(sc.x0 - 500.0, sc.x1 + 500.0, 0.0, y_top, _HEIGHT)
    body = [f'<line class="half" x1="{PAD}" y1="{sc.y(0.5):.1f}" x2="{WIDTH - PAD}" y2="{sc.y(0.5):.1f}"/>']
    legend = []
    for (label, pts), cls in zip(units.items(), _CLASSES * 3, strict=False):
        if len(pts) >= 2:
            line = " ".join(f"{sc.x(p.step):.0f},{sc.y(p.wr):.1f}" for p in pts)
            body.append(f'<polyline class="line {cls}" data-label="{esc(label)}" points="{line}"/>')
        for p in pts:
            x, y = sc.x(p.step), sc.y(p.wr)
            if p.ci:
                body.append(f'<line class="whisker {cls}" x1="{x:.1f}" y1="{sc.y(p.ci[0]):.1f}" '
                            f'x2="{x:.1f}" y2="{sc.y(p.ci[1]):.1f}"/>')
            kind = "point" if p.regime == "CONTENDED" else "idle"
            body.append(f'<circle class="marker {kind} {cls}" cx="{x:.1f}" cy="{y:.1f}" r="5"/>')
        legend.append(f'<li><i class="swatch {cls}"></i>{esc(label)}: {len(pts)} point(s), '
                      "whiskers = the pair-level bootstrap CI over distinct games (LAW-04)</li>")
    legend.append('<li><i class="swatch"></i>filled = CONTENDED (a live trainer shared the card) · '
                  '<i class="swatch idle"></i>hollow = IDLE</li>')
    latest = max(points, key=lambda p: p.step)
    caption = ("y = WR vs strix in the unit the legend names · x = step · the dashed rule is parity "
               f"(50 %) · latest: {esc(gap_statement(latest))}")
    return figure("".join(body), sc, height=_HEIGHT, x_label="step", caption=caption,
                  legend=f'<ul class="legend">{"".join(legend)}</ul>')


def external_panel(points: list[ExternalPoint], note: str, gaps: Gaps) -> Panel:
    """The tier-2 panel: a chart per unit, the gap table, and a stated gap when no sidecar was given."""
    reads = f"{PRODUCER}: step, unit (ours.sims, strix.sims), regime, wr, wr_ci_lower/upper, eff_n"
    title = "External anchor: strix"
    if not points:
        body = ('<p class="absent">No external point in this record: the strix series is read from '
                "the follower's sidecars, not from the event stream, and none was given.</p>"
                + gaps.mark(title, f"no strix sidecar read ({esc(note)}); the producer is "
                            f"<code>{esc(PRODUCER)}</code>"))
        return Panel(title, reads, body, "external")
    units = series_by_unit(points)
    rows = [[p.run_id, num(p.step), p.unit_label, p.regime, p.trigger, pct(p.wr),
             f"{pct(p.ci[0])}–{pct(p.ci[1])}" if p.ci else "—", num(p.eff_n),
             f"{(0.5 - p.wr) * 100:.1f} pp", p.net_hash[:12], p.checkpoint]
            for p in sorted(points, key=lambda p: (p.step, p.unit))]
    body = external_chart(units) + table(
        ["run", "step", "unit", "regime", "trigger", "WR vs strix", "95 % CI (pairs)", "eff_n",
         "gap to parity", "net hash", "checkpoint"], rows)
    gap_lines = "".join(f"<li>{esc(label)} — {esc(gap_statement(pts[-1]))}</li>"
                        for label, pts in units.items())
    body += f'<ul class="legend">{gap_lines}</ul>'
    note_text = (f"{note}. Each (run, unit) is its own series and each unit its own instrument (the "
                 "two sims pairs are never one series; another run's point — a parent's bridge cell — sits "
                 "on THAT run's step axis); "
                 "a point's regime travels with it because the wall changes with it and the WR did "
                 "not (STRIX_RUN7_60K_2026-09-17.md). Strix's absolute level is not stated: it is a "
                 "fixed external reference (R352(e)).")
    return Panel(title, reads, body, "external", note_text)
