"""The run monitor's records: one value-instrument read per save, the halt and the gap rule; an unread field stays None, never zero."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SAVE_RE = re.compile(r"^(?P<step>\d{8})\.json$")
PLY_BANDS = ("plies_0_10", "plies_11_40", "plies_41_up")
#: The record keys the Run view reads, `*` for every exam; pinned against `tools/run_monitor`'s real output.
PINNED: tuple[str, ...] = (
    "step", "saved_ts", "final", "gen.cf_ce", "gen.temperature", "gen.auc", "gen.policy_ce",
    *(f"gen.bands.{b}.cf_ce" for b in PLY_BANDS), "exams.*.calibrated_mean", "exams.*.floor", "exams.*.holds",
    "ring_bands.rows", "ring_bands.misses", "lagged_of.step", "lagged_of.diff", "lagged_of.ci", "lagged_of.games",
    "lagged_of.worse", "lagged_of.current.gap.cf_ce", "rates", "halting_rows", "reported_rows", "armed_floors",
    "floors_live", "gap_rule.gap", "gap_rule.line", "gap_rule.fired",
)
#: The keys the page reads off `HALT.json` and `GAP_RULE.json`, pinned against the monitor's real output too.
PINNED_HALT: tuple[str, ...] = ("step", "halting_rows", "armed", "final_save")
PINNED_GAP_RULE: tuple[str, ...] = ("step", "gap", "line", "over", "fired")


def missing_keys(raw: dict[str, Any], pinned: tuple[str, ...] = PINNED) -> list[str]:
    """The pinned keys a record lacks (a present key holding null is present: null is "not measured")."""
    out: list[str] = []
    for path in pinned:
        head, _, rest = path.partition(".*.")
        nodes = [(head, raw)] if not rest else [(f"{head}.{k}", v) for k, v in _dict(raw.get(head)).items()] or [
            (f"{head}.?", {})]
        for label, node in nodes:
            parts = (rest or path).split(".")
            here: Any = node
            for part in parts:
                if not isinstance(here, dict) or part not in here:
                    out.append(f"{label}.{rest}" if rest else path)
                    break
                here = here[part]
    return out


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def _dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


@dataclass(frozen=True)
class Exam:
    """One exam at one save: its calibrated mean, its floor, whether it holds (None = not measured) and why not."""

    calibrated_mean: float | None
    floor: float | None
    holds: bool | None
    not_measured: str | None


@dataclass(frozen=True)
class Lagged:
    """The previous save read against its lagged net on the games after it: the difference, its interval, the gap."""

    step: int
    diff: float | None
    ci: tuple[float, float] | None
    games: int | None
    gap_cf_ce: float | None
    worse: bool | None
    note: str | None


@dataclass(frozen=True)
class Save:
    """One save's reading of record; every number is None when the monitor did not measure it."""

    step: int
    saved_ts: float | None
    final: bool
    cf_ce: float | None
    temperature: float | None
    auc: float | None
    policy_ce: float | None
    bands: dict[str, float | None]
    gen_note: str | None
    exams: dict[str, Exam]
    ring_misses: tuple[str, ...]
    ring_note: str | None
    lagged: Lagged | None
    halting_rows: tuple[str, ...]
    reported_rows: tuple[str, ...]
    armed_floors: tuple[str, ...]
    floors_live: bool | None
    gap: float | None
    gap_line: float | None
    gap_fired: bool
    positions_per_h: float | None = None
    rates_note: str | None = None


def _exam(raw: Any) -> Exam:
    row = _dict(raw)
    holds = row.get("holds")
    return Exam(calibrated_mean=_num(row.get("calibrated_mean")), floor=_num(row.get("floor")),
                holds=holds if isinstance(holds, bool) else None,
                not_measured=str(row["not_measured"]) if row.get("not_measured") else None)


def _lagged(raw: Any) -> Lagged | None:
    row = _dict(raw)
    step = row.get("step")
    if not isinstance(step, int) or isinstance(step, bool):
        return None
    ci = row.get("ci")
    pair = (float(ci[0]), float(ci[1])) if isinstance(ci, list) and len(ci) == 2 and all(
        _num(x) is not None for x in ci) else None
    gap = _dict(_dict(row.get("current")).get("gap"))
    worse, games = row.get("worse"), row.get("games")
    return Lagged(step=step, diff=_num(row.get("diff")), ci=pair,
                  games=games if isinstance(games, int) and not isinstance(games, bool) else None,
                  gap_cf_ce=_num(gap.get("cf_ce")), worse=worse if isinstance(worse, bool) else None,
                  note=str(row["note"]) if row.get("note") else None)


def parse_save(raw: dict[str, Any]) -> Save:
    """A save record as `tools/run_monitor` writes it. Raises: KeyError, when the record carries no step."""
    gen, bands, rule, rates = (_dict(raw.get(k)) for k in ("gen", "ring_bands", "gap_rule", "rates"))
    per_band = _dict(gen.get("bands"))
    return Save(
        step=int(raw["step"]), saved_ts=_num(raw.get("saved_ts")), final=bool(raw.get("final")),
        cf_ce=_num(gen.get("cf_ce")), temperature=_num(gen.get("temperature")), auc=_num(gen.get("auc")),
        policy_ce=_num(gen.get("policy_ce")), bands={b: _num(_dict(per_band.get(b)).get("cf_ce")) for b in PLY_BANDS},
        gen_note=str(gen["not_measured"]) if gen.get("not_measured") else None,
        exams={str(k): _exam(v) for k, v in _dict(raw.get("exams")).items()},
        ring_misses=tuple(str(m) for m in bands.get("misses") or []),
        ring_note=str(bands["not_measured"]) if bands.get("not_measured") else None,
        lagged=_lagged(raw.get("lagged_of")),
        halting_rows=tuple(str(r) for r in raw.get("halting_rows") or []),
        reported_rows=tuple(str(r) for r in raw.get("reported_rows") or []),
        armed_floors=tuple(str(r) for r in raw.get("armed_floors") or []),
        floors_live=raw["floors_live"] if isinstance(raw.get("floors_live"), bool) else None,
        gap=_num(rule.get("gap")), gap_line=_num(rule.get("line")), gap_fired=rule.get("fired") is True,
        positions_per_h=_num(rates.get("positions_per_h")), rates_note=str(rates["note"]) if rates.get("note") else None)


@dataclass(frozen=True)
class Records:
    """Every save read, in step order, the halt and the gap rule's first firing when on record, and what was skipped."""

    saves: tuple[Save, ...]
    halt: dict[str, Any] | None
    gap_rule: dict[str, Any] | None
    skipped: tuple[str, ...]


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load(records_dir: Path) -> Records:
    """The monitor's records under `records_dir`; an unparseable save is skipped and named, never read as zeros. Raises: OSError."""
    saves: list[Save] = []
    skipped: list[str] = []
    folder = records_dir / "saves"
    for path in sorted(folder.iterdir()) if folder.is_dir() else []:
        if SAVE_RE.match(path.name) is None:
            continue
        raw = _json(path)
        try:
            saves.append(parse_save(raw if isinstance(raw, dict) else {}))
        except (KeyError, TypeError, ValueError):
            skipped.append(path.name)
    halt, gap = _json(records_dir / "HALT.json"), _json(records_dir / "GAP_RULE.json")
    return Records(saves=tuple(sorted(saves, key=lambda s: s.step)), halt=halt if isinstance(halt, dict) else None,
                   gap_rule=gap if isinstance(gap, dict) else None, skipped=tuple(skipped))
