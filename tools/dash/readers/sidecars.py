"""Strength from the follower's cell sidecars: one series per unit, each with its own parent, never pooled or joined across units."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # the ladder tool loads this file alone, outside the package
    from .ladder import Change

#: The sidecars the follower writes beside a checkpoint: `<ckpt>.six30_16[.arm].json`, `<ckpt>.strix256[_r6].json`,
#: `<ckpt>.ladder455_n16[.arm].json`; a test holds every follower unit to these.
GLOBS = ("*.six*.json", "*.strix*.json", "*.ladder*.json")
#: The going-forward read: the mean logit of the last `CELLS` cells against the parent's logit plus `LINE_LOGIT`.
CELLS = 4
LINE_LOGIT = 0.17


Z95 = 1.96


class CellRefused(ValueError):
    """A sidecar whose reading cannot be stated honestly; the message says why."""


def wilson(k: float, n: int) -> tuple[float, float]:
    """The Wilson 95 % interval of k successes in n (a draw counts half); (0, 1) for n = 0."""
    if n <= 0:
        return 0.0, 1.0
    p, z2 = k / n, Z95 * Z95
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = Z95 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def logit(p: float, n: int | None = None) -> float:
    """log(p / (1 − p)); with `n` games a reading of 0 or 1 is moved half a game in, else it is clamped at 1e-6."""
    edge = 0.5 / n if n else 1e-6
    q = min(max(p, edge), 1 - edge)
    return math.log(q / (1 - q))


@dataclass(frozen=True)
class Cell:
    """One sidecar: whose checkpoint, at which step, the unit it was read in (`name` for the page, `unit` for identity), its reading."""

    run_id: str
    stem: str
    step: int
    family: str
    unit_field: str
    name: str
    unit: tuple[str, ...]
    label: str
    wr: float
    lo: float | None
    hi: float | None
    n: int
    regime: str
    forfeits: int = 0
    title: str = ""

    @property
    def logit(self) -> float:
        """The win rate's logit, half a game in from 0 and 1."""
        return logit(self.wr, self.n)

    @property
    def logit_half_width(self) -> float | None:
        """Half the interval's width in logit (each bound half a game in), or None without both bounds."""
        if self.lo is None or self.hi is None:
            return None
        return (logit(self.hi, self.n) - logit(self.lo, self.n)) / 2


def _int(v: Any) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def _tactics(raw: dict[str, Any]) -> tuple[str | None, str]:
    """The tactics arm and its block's hash: two blocks under one arm name are two instruments."""
    t = raw.get("tactics")
    if not isinstance(t, dict):
        return None, "no tactics"
    block = t.get("block")
    digest = "" if block is None else hashlib.sha256(json.dumps(block, sort_keys=True).encode()).hexdigest()[:8]
    arm = str(t.get("arm") or "config")
    return arm, f"{arm} {digest}".strip()


def parse(path: Path, raw: Any) -> Cell | None:
    """A cell from one sidecar's JSON, or None when a field the series needs is absent (distinct games included)."""
    if not isinstance(raw, dict):
        return None
    step, wr, ours, n = _int(raw.get("step")), _num(raw.get("wr")), raw.get("ours") or {}, _int(raw.get("eff_n"))
    six, strix = raw.get("six"), raw.get("strix")
    if step is None or wr is None or n is None or not isinstance(ours, dict):
        return None
    arm, tactics = _tactics(raw)
    field, book = str(raw.get("unit")), raw.get("opening_book")
    # The openings are part of the instrument: one unit name over two books is two series.
    common = (field, str(ours.get("search_kind")), str(ours.get("sims")), tactics, str(raw.get("opening_book_sha256") or book))
    openings = f", {str(book).split('_')[0]} openings" if book else ""
    if isinstance(six, dict):
        family = "six"
        unit = ("six", *common, str(six.get("commit")), str(six.get("net_sha256")), str(six.get("nodes")))
        label = (f"Ours: {str(ours.get('search_kind')).upper()}, {ours.get('sims')} sims, "
                 f"{tactics if arm is None else 'tactics ' + tactics}. Six: gen {six.get('generation')}, {six.get('nodes')} nodes")
        title = f"Six gen {six.get('generation')}, {six.get('nodes')} nodes{openings}"
    elif isinstance(strix, dict):
        family = "strix"
        unit = ("strix", *common, str(strix.get("commit")), str(strix.get("checkpoint_sha256")), str(strix.get("sims")),
                str(strix.get("solver", "on")), str(strix.get("radius")), str(strix.get("device", "cpu")))
        label = (f"Ours: {str(ours.get('search_kind')).upper()}, {ours.get('sims')} sims. Strix: {strix.get('sims')} sims, solver "
                 f"{strix.get('solver', 'on')}" + ("" if strix.get("radius") is None else f", radius {strix.get('radius')}"))
        title = f"Strix {strix.get('sims')} sims{openings}"
    else:
        return None
    if book:
        label += f". On {book} openings"
    stem = str(raw.get("checkpoint") or path.name.split(".ckpt")[0]).split(".ckpt")[0]
    lo, hi = _num(raw.get("wr_ci_lower")), _num(raw.get("wr_ci_upper"))
    findings = raw.get("six_findings") if family == "six" else None
    forfeits = (_int(findings.get("count")) if isinstance(findings, dict) else None) or 0
    if forfeits:
        wr, lo, hi, n = _without_forfeits(raw, wr, n, forfeits)
    return Cell(run_id=str(raw.get("run_id")), stem=stem, step=step, family=family, unit_field=field,
                name=field if arm is None else f"{field}.{arm}", unit=unit, label=label, wr=wr, lo=lo, hi=hi, n=n,
                regime=str(raw.get("regime", "?")), forfeits=forfeits, title=title)


def _without_forfeits(raw: dict[str, Any], wr: float, n: int, forfeits: int) -> tuple[float, float, float, int]:
    """The reading over the real games: the sidecar counts each Six forfeit as our win. Raises: CellRefused (no real game, or counts that do not line up)."""
    games = _int(raw.get("games"))
    if games is not None and games != n:
        raise CellRefused(f"{forfeits} Six forfeits counted over {games} games, the reading over {n} distinct games")
    if forfeits >= n:
        raise CellRefused(f"every one of its {n} games is a Six forfeit")
    wins, draws = _int(raw.get("wins")), _int(raw.get("draws")) or 0
    score = (wins - forfeits + draws / 2) if wins is not None else wr * n - forfeits
    real = n - forfeits
    lo, hi = wilson(max(0.0, score), real)
    return max(0.0, score) / real, lo, hi, real


def load(dirs: Iterable[Path]) -> tuple[list[Cell], list[str]]:
    """Every sidecar under the directories (failed cells skipped and named), and the skip notes."""
    cells: list[tuple[Cell, str, str]] = []
    skipped: list[str] = []
    seen: set[Path] = set()
    for root in dirs:
        if not root.is_dir():
            skipped.append(f"a --cells directory named {root.name} does not exist")
            continue
        for path in sorted({p for g in GLOBS for p in root.rglob(g)}):
            if path in seen:
                continue
            seen.add(path)
            if path.name.endswith(".failed.json"):
                skipped.append(f"{path.name}: a failed cell")
                continue
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                cell = parse(path, raw)
            except CellRefused as exc:
                skipped.append(f"{path.name}: {exc}")
                continue
            except (OSError, ValueError) as exc:
                skipped.append(f"{path.name}: {type(exc).__name__}")
                continue
            if cell is None:
                skipped.append(f"{path.name}: no step, win rate, distinct games or opponent")
            else:
                cells.append((cell, str(raw.get("finished_utc") or ""), str(path)))
    # One checkpoint read twice on one unit: the reading over more games stands (a screen or smoke never hides a full cell),
    # then the later-finished one, then the path, so the pick never depends on the order the directories were walked.
    unique: dict[tuple[str, tuple[str, ...]], tuple[Cell, str, str]] = {}
    for entry in cells:
        c, held = entry[0], unique.get((entry[0].stem, entry[0].unit))
        if held is None or (c.n, entry[1], entry[2]) > (held[0].n, held[1], held[2]):
            unique[(c.stem, c.unit)] = entry
    return sorted((e[0] for e in unique.values()), key=lambda c: (c.run_id, c.step, c.name)), skipped


@dataclass(frozen=True)
class Ruler:
    """One unit's series for one run: its cells by step, its parent's cell, whether the rule reads it, when a switch moved it off."""

    name: str
    unit_field: str
    family: str
    unit: tuple[str, ...]
    label: str
    line: tuple[Cell, ...]
    parent: Cell | None
    rule: bool
    title: str = ""
    rule_until: int | None = None
    rule_since: int | None = None

    @property
    def going_forward(self) -> tuple[float, int] | None:
        """`(mean logit of the last cells since the rule moved here − the parent's logit, cells used)`, or None."""
        ruled = [c for c in self.line if self.rule_since is None or c.step >= self.rule_since]
        if self.parent is None or not ruled:
            return None
        last = ruled[-CELLS:]
        return sum(c.logit for c in last) / len(last) - self.parent.logit, len(last)


def _tag(unit: tuple[str, ...]) -> str:
    return hashlib.sha256("|".join(unit).encode()).hexdigest()[:6]


def rulers(cells: list[Cell], run_id: str, parent_stem: str | None, rule: str | None,
           switches: tuple[Change, ...] = ()) -> tuple[tuple[Ruler, ...], int]:
    """Every unit as its own series (same-named ones told apart by a hash), and how many units the rule names: it marks only one."""
    by_unit: dict[tuple[str, ...], list[Cell]] = {}
    for c in cells:
        if c.run_id == run_id:
            by_unit.setdefault(c.unit, []).append(c)
    names: dict[str, int] = {}
    titles: dict[str, int] = {}
    for own in by_unit.values():
        names[own[0].name] = names.get(own[0].name, 0) + 1
        titles[own[0].title] = titles.get(own[0].title, 0) + 1

    def named(u: str | None) -> list[tuple[str, ...]]:
        return [k for k, own in by_unit.items() if u is not None and u in (own[0].name, own[0].unit_field)]

    matched = named(rule)
    until = {former: s.step for s in switches for former in named(s.frm)}
    out = []
    for unit, own in by_unit.items():
        head = own[0]
        parent = next((c for c in cells if parent_stem is not None and c.stem == parent_stem and c.unit == unit), None)
        name = head.name if names[head.name] == 1 else f"{head.name} #{_tag(unit)}"
        title = head.title if titles[head.title] == 1 else f"{head.title} ({name})"
        out.append(Ruler(name=name, unit_field=head.unit_field, family=head.family, unit=unit, label=head.label, title=title,
                         line=tuple(sorted(own, key=lambda c: c.step)), parent=parent,
                         rule=len(matched) == 1 and unit == matched[0], rule_until=until.get(unit),
                         rule_since=switches[-1].step if switches and unit in matched else None))
    return tuple(sorted(out, key=lambda r: (not r.rule, -(r.rule_until or -1), r.family != "six", r.name))), len(matched)


def bridges(cells: list[Cell], run_id: str, units: tuple[str, str]) -> list[tuple[Cell, Cell]]:
    """Checkpoints of the run read on both units (by name or unit field): the same net on two rungs of the ladder."""
    def on(u: str) -> dict[str, Cell]:
        return {c.stem: c for c in cells if c.run_id == run_id and u in (c.name, c.unit_field)}
    a, b = on(units[0]), on(units[1])
    return [(a[s], b[s]) for s in sorted(set(a) & set(b), key=lambda s: a[s].step)]
