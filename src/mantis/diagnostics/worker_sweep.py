# >300 justify (R8): ONE claim — the worker count this card wants, the rule that picked it, and
# every number the rule ran on; the walk, the readings, the verdicts and the knee are not separable.
"""`python -m mantis.diagnostics.worker_sweep` — what should `selfplay.n_workers` BE on this box?

Walks a pre-registered worker ladder, verdicts each rung's memory series, and picks the knee.
SELF-PLAY ONLY, and that is structural: no trainer step may execute, so this module builds its
own collaborators and never imports `mantis.run`.

THE RULE IS NOT IN THIS FILE. Every threshold is read from an explicit plan file
(`tools/worker_sweep_plan.toml`); missing and unknown keys both raise. The stopping rule is
`eval_child_memory.classify` imported, not re-written, and REFUSED is never a verdict.

TWO SINKS, LARGER GOVERNS, AND THE REPORT SAYS WHICH ONE DID: `card` is a LEVEL, `allocator` a
per-round DEMAND peak. They disagreed by 3.62 GiB at matched config on one host, and the
disagreement is a finding.

Game counts come from the Rust runner's counters, never events (falsified F-43: `game_complete`
is dropped in production). `moves` is `positions_generated`, one per APPLIED COMPOUND TURN, not
per ply. Every figure carries its sampling limit; unmeasured rounds are excluded BY NAME.

`--config` + `--plan` DRIVES; `--select-only` re-derives a pick from a written report and drives
nothing. Naming an input a mode does not read is itself a refusal.

    0   a pick was made
    1   no rung PASSED — every measurable rung was GROWING or OOM
    2   REFUSED — the plan, the config, the report or the ladder cannot be answered about

`rc 1` is "NO PICK", not "the sitting failed". It NEVER MINTS AND NEVER WRITES A CONFIG.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import tomllib
from dataclasses import dataclass
from math import isfinite, sqrt
from pathlib import Path
from statistics import fmean, median, stdev
from typing import Any

import torch

from mantis.config.loader import load_config
from mantis.config.resolve.allocator_posture import (
    assert_allocator_posture,
    declared_allocator_posture,
    governs_device,
    read_live_allocator_conf,
)
from mantis.config.resolve.coordinator import resolve_coordinator_knobs
from mantis.diagnostics.eval_child_memory import (
    GROWING,
    PLATEAU,
    RC_REFUSED,
    InsufficientRoundsError,
    classify,
)
from mantis.eval.child_memory import make_probe
from mantis.model.identity import net_param_hash
from mantis.selfplay.buffers import BufferKind
from mantis.selfplay.hparams import resolve_pool_encoding
from mantis.selfplay.pool import WorkerPool
from mantis.selfplay.pool_hooks import runner_stats
from mantis.util.determinism import seed_everything
from mantis.util.device import (
    cuda_counters_available,
    cuda_device_total_bytes,
    cuda_device_used_bytes,
    reset_cuda_peak_counters,
)

TOOL = "mantis.diagnostics.worker_sweep"
MARKER = "MANTIS_WORKER_SWEEP"
GIB = 1024 ** 3

#: Rung verdicts beyond the two `classify` returns. REFUSED keeps `eval_child_memory`'s meaning
#: exactly — not a verdict, a statement that no verdict is available.
REFUSED = "REFUSED"
OOM = "OOM"
#: A rung that failed for a reason this tool does not model. NOT a tool failure: `panic = "unwind"`
#: makes a Rust panic cross the FFI as an exception, so it is a named RUNG failure, not a lost ladder.
RUNG_ERROR = "RUNG_ERROR"
#: The pool's sole producer died (`WorkerPool.check_producer_health` raises it into view). Without
#: that call the Rust counters climb while nothing reaches the buffer: throughput plus a flat series.
PRODUCER_DEAD = "PRODUCER_DEAD"

#: The three series a rung is verdicted on. The two instruments are verdicted SEPARATELY because
#: `max()` resolves them into `governing` before the stopping rule ever runs.
_SINK_FIELDS = {
    "governing": "governing_peak_bytes",
    "card": "sampled_peak_bytes",
    "allocator": "allocator_peak_bytes",
}

#: The closed token set for `[selection].metric`. Both figures are recorded for every rung
#: whichever is chosen; only the RANKING is single-valued.
METRICS = ("moves_per_min", "games_per_min")

#: The ruling's own constants, pinned here and NOT plan knobs: the plan STATES them so the report
#: echoes what it ran under, and the loader REFUSES any other value.
RULED_RUNGS = (2, 4, 8, 12, 14)
RULED_KNEE_PCT = 95.0

#: The ranking metric is pinned too: `games_per_min` reads 0 for a HEALTHY rung whenever a round
#: is shorter than a game (measured, 22 moves and ZERO games at 2 workers over 20 s).
PREREG_METRIC = "moves_per_min"

#: The determinism control's FORMER band, superseded and kept only because old reports cite it:
#: 1% held engine-side at 0.5821%, then a live box measured 3.9258%. It gates nothing now.
RULED_DETERMINISM_BAND_PCT = 1.0

#: The band's UPPER bound: at `band_pct = 500` every rung PLATEAUs and the memory gate is off.
MAX_BAND_PCT = 5.0
#: A window of one round makes PLATEAU mean "the final round did not exceed the max of all before
#: it", which is not a convergence test at all.
MIN_PLATEAU_ROUNDS = 2

#: The plan's exact shape; missing section, missing key and unknown key are each a named `ValueError`.
PLAN_SHAPE: dict[str, tuple[str, ...]] = {
    "provenance": ("prereg_ruling", "prereg_recorded", "authored_by", "note"),
    "ladder": ("rungs", "extension_step", "extension_max", "min_gain_pct"),
    "rounds": ("warmup_rounds", "measured_rounds", "round_sec", "sampler_interval_sec"),
    "stopping_rule": ("plateau_rounds", "band_pct"),
    "selection": ("knee_pct", "metric"),
}


class SweepRefusal(Exception):
    """A named refusal that exits `RC_REFUSED` and emits no pick, so every refusal takes ONE exit
    path with one code and one destination."""


class _Discard:
    """Swallow the eval probe's own `MANTIS_EVAL_MEM` channel, reusing everything else it does: a
    sweep log carrying eval markers would be read by `eval_child_memory` as an eval drive."""

    def write(self, _text: str) -> int:
        return 0

    def flush(self) -> None:
        return None


@dataclass(frozen=True)
class SweepPlan:
    """The pre-registered rule, whole. Frozen: a run-scoped constant a consumer could rebind is a
    second authority with extra steps."""

    rungs: tuple[int, ...]
    extension_step: int
    extension_max: int
    min_gain_pct: float
    warmup_rounds: int
    measured_rounds: int
    round_sec: float
    sampler_interval_sec: float
    plateau_rounds: int
    band_pct: float
    knee_pct: float
    metric: str
    provenance: dict[str, Any]

    @property
    def rounds_per_rung(self) -> int:
        return self.warmup_rounds + self.measured_rounds


def _require(section: str, keys: tuple[str, ...], block: Any, path: Path) -> dict[str, Any]:
    if not isinstance(block, dict):
        raise ValueError(f"{path}: [{section}] is missing or is not a table")
    missing = [k for k in keys if k not in block]
    unknown = [k for k in block if k not in keys]
    if missing:
        raise ValueError(
            f"{path}: [{section}] is missing required key(s) {missing}. This plan has no "
            "defaults: a threshold nobody wrote down is a rule nobody chose, applied to a "
            "measurement a mint will rest on."
        )
    if unknown:
        raise ValueError(
            f"{path}: [{section}] carries unknown key(s) {unknown}. Refusing rather than "
            "ignoring them — a silently-ignored key is a rule someone believed was in force."
        )
    return dict(block)


def load_plan(path: str | Path) -> SweepPlan:
    """Load and validate the sweep plan. Every refusal names what is wrong and why it matters."""
    plan_path = Path(path)
    raw = tomllib.loads(plan_path.read_text(encoding="utf-8"))
    unknown_sections = [k for k in raw if k not in PLAN_SHAPE]
    if unknown_sections:
        raise ValueError(f"{plan_path}: unknown section(s) {unknown_sections}")
    blocks = {name: _require(name, keys, raw.get(name), plan_path)
              for name, keys in PLAN_SHAPE.items()}
    ladder, rounds = blocks["ladder"], blocks["rounds"]
    rule, selection = blocks["stopping_rule"], blocks["selection"]
    # TYPES ARE NOT COERCED: `float("95.0")` and `int(2.9)` both succeed, so a plan could STATE one
    # thing and RUN another — `rungs = [2.9, ...]` truncating to the ruled ladder is the instance.
    for section, key, value, want in (
        ("ladder", "rungs", tuple(ladder["rungs"]), int),
        ("ladder", "extension_step", ladder["extension_step"], int),
        ("ladder", "extension_max", ladder["extension_max"], int),
        ("ladder", "min_gain_pct", ladder["min_gain_pct"], float),
        ("rounds", "warmup_rounds", rounds["warmup_rounds"], int),
        ("rounds", "measured_rounds", rounds["measured_rounds"], int),
        ("rounds", "round_sec", rounds["round_sec"], float),
        ("rounds", "sampler_interval_sec", rounds["sampler_interval_sec"], float),
        ("stopping_rule", "plateau_rounds", rule["plateau_rounds"], int),
        ("stopping_rule", "band_pct", rule["band_pct"], float),
        ("selection", "knee_pct", selection["knee_pct"], float),
    ):
        values = value if isinstance(value, tuple) else (value,)
        for item in values:
            ok = isinstance(item, int) and not isinstance(item, bool) if want is int else (
                isinstance(item, (int, float)) and not isinstance(item, bool))
            if not ok:
                raise ValueError(
                    f"{plan_path}: [{section}].{key} must be written as "
                    f"{'an integer' if want is int else 'a number'}, got {item!r} "
                    f"({type(item).__name__}). Nothing here is coerced: a plan that states one "
                    "value and runs another is not a pre-registration."
                )
    rungs = tuple(int(r) for r in ladder["rungs"])
    if not rungs:
        raise ValueError(f"{plan_path}: [ladder].rungs is empty — there is no ladder to walk")
    if any(r < 2 for r in rungs):
        raise ValueError(
            f"{plan_path}: [ladder].rungs contains a rung below 2 ({sorted(rungs)}). R309(f) "
            "REJECTS n_workers = 1 as far too low; the rule cannot be un-made by editing the "
            "file it is written in."
        )
    if list(rungs) != sorted(set(rungs)):
        raise ValueError(f"{plan_path}: [ladder].rungs must be strictly increasing: {rungs}")
    if rungs != RULED_RUNGS:
        raise ValueError(
            f"{plan_path}: [ladder].rungs is {list(rungs)}; R309(g) names the base ladder "
            f"{list(RULED_RUNGS)} and R309(f) closes with 'No post-hoc movement of any of it'. "
            "Extension past the top of that ladder is what the ruling permits, and it is the "
            "EXTENSION keys that carry it — not an edit to the base ladder."
        )
    if float(selection["knee_pct"]) != RULED_KNEE_PCT:
        raise ValueError(
            f"{plan_path}: [selection].knee_pct is {selection['knee_pct']}; R309(f) fixes the "
            f"knee rule at {RULED_KNEE_PCT:g} percent of the best PASSING rung's throughput. "
            "This is the ruling's own constant, in the same sentence as 'No post-hoc movement "
            "of any of it', and it is not a plan knob."
        )
    if int(ladder["extension_max"]) < max(rungs):
        raise ValueError(
            f"{plan_path}: [ladder].extension_max {ladder['extension_max']} is below the "
            f"highest base rung {max(rungs)} — the ceiling would refuse the ladder itself"
        )
    if int(rounds["measured_rounds"]) <= int(rule["plateau_rounds"]):
        raise ValueError(
            f"{plan_path}: [rounds].measured_rounds {rounds['measured_rounds']} is not GREATER "
            f"than [stopping_rule].plateau_rounds {rule['plateau_rounds']}. Fewer would make a "
            "verdict unreachable; EQUAL is worse, because it is reachable and wrong: `classify` "
            "then takes its degenerate branch (`running = peaks[0]`, the window covering the "
            "whole series) and the rule silently becomes 'no round exceeds the FIRST by more "
            "than the band', with no history outside the window at all. A plan that cannot "
            "reach a HONEST verdict is a refusal at LOAD, not a surprise hours into a sitting."
        )
    for section, key, value in (
        ("rounds", "round_sec", rounds["round_sec"]),
        ("rounds", "sampler_interval_sec", rounds["sampler_interval_sec"]),
        ("ladder", "extension_step", ladder["extension_step"]),
    ):
        if float(value) <= 0:
            raise ValueError(f"{plan_path}: [{section}].{key} must be positive, got {value}")
    # THE TWO KNOBS THAT DECIDE PASS/FAIL FOR EVERY RUNG, bounded on BOTH sides: `band_pct = 500`
    # turns the memory gate off entirely and `plateau_rounds = 1` is not a convergence test.
    if not 0.0 <= float(rule["band_pct"]) <= MAX_BAND_PCT:
        raise ValueError(
            f"{plan_path}: [stopping_rule].band_pct must be in [0, {MAX_BAND_PCT:g}], got "
            f"{rule['band_pct']}. Above that bound the memory conjunct R309(f) makes the PASS "
            "condition is off, and every rung plateaus by construction."
        )
    if int(rule["plateau_rounds"]) < MIN_PLATEAU_ROUNDS:
        raise ValueError(
            f"{plan_path}: [stopping_rule].plateau_rounds must be at least "
            f"{MIN_PLATEAU_ROUNDS}, got {rule['plateau_rounds']}. A window of one round is not a "
            "convergence test — and `classify` refuses 0 or less with a DIFFERENT error, hours "
            "into a sitting, after the first rung has been driven."
        )
    if not 0.0 < float(selection["knee_pct"]) <= 100.0:
        raise ValueError(
            f"{plan_path}: [selection].knee_pct must be in (0, 100], got {selection['knee_pct']}"
        )
    if float(rule["band_pct"]) < 0.0 or float(ladder["min_gain_pct"]) < 0.0:
        raise ValueError(f"{plan_path}: band_pct and min_gain_pct may not be negative")
    if selection["metric"] not in METRICS:
        raise ValueError(
            f"{plan_path}: [selection].metric {selection['metric']!r} is not one of {METRICS}"
        )
    if selection["metric"] != PREREG_METRIC:
        raise ValueError(
            f"{plan_path}: [selection].metric is {selection['metric']!r}; the ranking metric is "
            f"PRE-REGISTERED as {PREREG_METRIC!r} (DESIGN amendment A1, on a measurement taken "
            "before any box number existed). Both figures are recorded and printed for every "
            "rung either way — only the RANKING is single-valued, and it is not a plan knob. "
            "With rounds shorter than a game, `games_per_min` ranks every rung at zero and the "
            "knee then picks the smallest rung at rc 0 while the moves column disagrees."
        )
    if int(rounds["warmup_rounds"]) < 0:
        raise ValueError(f"{plan_path}: [rounds].warmup_rounds may not be negative")
    # At a sampler interval at or above the round length a round collects zero or one card sample,
    # so the rung would PASS on the allocator series alone. Refused at LOAD, not an hour in.
    if float(rounds["sampler_interval_sec"]) >= float(rounds["round_sec"]):
        raise ValueError(
            f"{plan_path}: [rounds].sampler_interval_sec "
            f"({rounds['sampler_interval_sec']}) is not shorter than round_sec "
            f"({rounds['round_sec']}), so a round collects at most one card sample and the "
            "card high-water is not a series. The card column would read card_samples<=1 on "
            "every rung while the rung still PASSED on the allocator series alone."
        )
    return SweepPlan(
        rungs=rungs,
        extension_step=int(ladder["extension_step"]),
        extension_max=int(ladder["extension_max"]),
        min_gain_pct=float(ladder["min_gain_pct"]),
        warmup_rounds=int(rounds["warmup_rounds"]),
        measured_rounds=int(rounds["measured_rounds"]),
        round_sec=float(rounds["round_sec"]),
        sampler_interval_sec=float(rounds["sampler_interval_sec"]),
        plateau_rounds=int(rule["plateau_rounds"]),
        band_pct=float(rule["band_pct"]),
        knee_pct=float(selection["knee_pct"]),
        metric=str(selection["metric"]),
        provenance=blocks["provenance"],
    )


def emit_marker(record: dict[str, Any], *, out: Any) -> None:
    """Write one `MANTIS_WORKER_SWEEP` line, flushed: a sweep can be killed at a rung, and a
    buffered marker is a measurement that did not survive the thing it was measuring."""
    print(f"{MARKER} {json.dumps(record, sort_keys=True)}", file=out, flush=True)


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Refuse a duplicate key rather than taking the last, which is what `json.loads` does. This
    reader is the recovery path for a sweep killed at a rung — a hand-repaired log."""
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(
                f"{MARKER} payload carries the key {key!r} twice. Refusing rather than taking "
                "the last one: a concatenated or hand-repaired log is exactly what this reader "
                "is for."
            )
        seen[key] = value
    return seen


def _no_non_finite(token: str) -> Any:
    """Refuse `NaN`/`Infinity`/`-Infinity`, which `json.loads` accepts as values. A peak of `NaN`
    is not a measurement and must not travel as one."""
    raise ValueError(f"{MARKER} payload carries the non-finite constant {token!r}")


def parse_sweep_markers(text: str) -> list[dict[str, Any]]:
    """Recover the marker records from a captured sweep log, FAILING CLOSED on no markers and on
    an unreadable payload: a reader that guesses produces a number indistinguishable from data."""
    records: list[dict[str, Any]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith(MARKER):
            continue
        body = line[len(MARKER):].strip()
        try:
            parsed = json.loads(body, object_pairs_hook=_no_duplicate_keys,
                                parse_constant=_no_non_finite)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{MARKER} line carries no readable json payload ({exc}); refusing to guess at "
                f"it: {body[:120]!r}"
            ) from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"{MARKER} payload must be a json object, got {type(parsed).__name__}")
        records.append(parsed)
    if not records:
        raise ValueError(
            f"no {MARKER} lines found. This reader does not fall back to guessing at a file's "
            "shape: the substitute reading would be indistinguishable from a measurement."
        )
    return records


class CardSampler:
    """Background running-maximum of CARD used bytes — the second instrument, not a second view.
    Runs only where the counters exist; `peak()`/`samples()` are per-WINDOW, opened by `reset()`."""

    def __init__(self, device: str, interval_sec: float) -> None:
        self._device = device
        self._interval = interval_sec
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._peak: int | None = None
        self._samples = 0
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                used = cuda_device_used_bytes(self._device)
                with self._lock:
                    self._peak = used if self._peak is None else max(self._peak, used)
                    self._samples += 1
                self._stop.wait(self._interval)
        except Exception as exc:  # noqa: BLE001 — recorded and surfaced, never swallowed
            # A DAEMON THREAD THAT DIES SILENTLY IS A MEASUREMENT THAT STOPPED: `window()` would keep
            # returning the open window's last peak and then `(None, 0)`, switching instrument mid-flight.
            with self._lock:
                self._error = exc

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True, name="sweep-card-sampler")
        self._thread.start()

    def reset(self) -> None:
        with self._lock:
            self._peak = None
            self._samples = 0

    def window(self) -> tuple[int | None, int]:
        with self._lock:
            return self._peak, self._samples

    def error(self) -> BaseException | None:
        """The sampler thread's cause of death, or `None` while it is alive."""
        with self._lock:
            return self._error

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None


@dataclass(frozen=True)
class RoundReading:
    """One measurement round. `available` False means the host had no CUDA counters — the round is
    still LISTED and COUNTED, and is excluded from the verdict by name."""

    index: int
    warmup: bool
    wall_sec: float
    games: int
    moves: int
    available: bool
    sampled_peak_bytes: int | None
    allocator_peak_bytes: int | None
    card_samples: int

    @property
    def governing_peak_bytes(self) -> int | None:
        """The LARGER of the two sinks — the box block's standing rule, which needs both."""
        peaks = [p for p in (self.sampled_peak_bytes, self.allocator_peak_bytes) if p is not None]
        return max(peaks) if peaks else None

    @property
    def governing_sink(self) -> str | None:
        """WHICH sink governed, reported because the two are different KINDS of quantity.

        `card` is `total - free`, a LEVEL unaffected by `reset_peak_memory_stats`; `allocator` is
        a per-round DEMAND peak the round boundary zeroes. `max()` is the right GATE but silently
        changes what the series MEANS depending on which side wins.
        """
        if self.governing_peak_bytes is None:
            return None
        return ("card" if self.sampled_peak_bytes is not None
                and self.governing_peak_bytes == self.sampled_peak_bytes else "allocator")

    @property
    def sink_disagreement_bytes(self) -> int | None:
        """How far apart the two instruments were — the block's rule ends "and the disagreement is
        a finding", which needs the difference reported to be actionable."""
        if self.sampled_peak_bytes is None or self.allocator_peak_bytes is None:
            return None
        return abs(self.sampled_peak_bytes - self.allocator_peak_bytes)

    @property
    def moves_per_min(self) -> float:
        return (self.moves / self.wall_sec) * 60.0 if self.wall_sec > 0 else 0.0

    @property
    def games_per_min(self) -> float:
        return (self.games / self.wall_sec) * 60.0 if self.wall_sec > 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "warmup": self.warmup, "wall_sec": round(self.wall_sec, 3),
            "games": self.games, "moves": self.moves, "available": self.available,
            "sampled_peak_bytes": self.sampled_peak_bytes,
            "allocator_peak_bytes": self.allocator_peak_bytes,
            "governing_peak_bytes": self.governing_peak_bytes,
            "governing_sink": self.governing_sink,
            "sink_disagreement_bytes": self.sink_disagreement_bytes,
            "card_samples": self.card_samples,
            "moves_per_min": round(self.moves_per_min, 4),
            "games_per_min": round(self.games_per_min, 4),
        }


@dataclass(frozen=True)
class RungResult:
    """One rung's whole record — the verdict AND everything the verdict was taken on."""

    n_workers: int
    verdict: str
    rounds: tuple[RoundReading, ...]
    refusal: str | None
    produced_by: str
    #: The constructed net's parameter hash, post-seed pre-play. `None` when the rung never reached
    #: a built net; the gate excludes those rather than reading an absence as agreement.
    net_param_hash: str | None = None

    def measured_peaks(self) -> list[int]:
        """The governing peaks of the measured rounds, oldest first. The narrowing lives here, not
        at each call site: re-deriving `measured`'s own guarantee is three places to get it wrong."""
        return [peak for r in self.measured if (peak := r.governing_peak_bytes) is not None]

    @property
    def measured(self) -> tuple[RoundReading, ...]:
        return tuple(r for r in self.rounds
                     if not r.warmup and r.available and r.governing_peak_bytes is not None)

    @property
    def scored(self) -> tuple[RoundReading, ...]:
        """Rounds a THROUGHPUT figure is taken over: non-warm-up, measured or not. Throughput does
        not need CUDA counters and is not thrown away for want of them."""
        return tuple(r for r in self.rounds if not r.warmup)

    def throughput(self, metric: str) -> float:
        wall = sum(r.wall_sec for r in self.scored)
        if wall <= 0:
            return 0.0
        total = sum(r.moves if metric == "moves_per_min" else r.games for r in self.scored)
        return (total / wall) * 60.0

    def spread(self, metric: str) -> dict[str, float | None]:
        """Min / median / max of the per-round rate across the scored rounds. Reported, never
        gating: the knee band is 5 percent and a reader who cannot see the spread cannot tell a
        5 percent difference from noise."""
        values = sorted(getattr(r, metric) for r in self.scored)
        if not values:
            return {"min": None, "median": None, "max": None, "mean": None, "rel_se": None,
                    "n_rounds": 0}
        mean = fmean(values)
        # THIS RUNG'S OWN NOISE. `None` when it cannot be stated (one scored round, or a non-positive
        # mean); `select_knee` REFUSES such a rung rather than reading None as zero.
        rel_se = ((stdev(values) / sqrt(len(values))) / mean
                  if len(values) >= 2 and mean > 0 else None)
        return {"min": round(values[0], 4), "median": round(median(values), 4),
                "max": round(values[-1], 4), "mean": round(mean, 4), "rel_se": rel_se,
                "n_rounds": len(values)}

    def series(self, sink: str) -> list[int]:
        """One sink's own per-round peaks, oldest first, warm-up and unmeasured rounds excluded."""
        key = _SINK_FIELDS[sink]
        return [value for r in self.measured if (value := getattr(r, key)) is not None]

    def sink_verdicts(self, *, plateau_rounds: int, band_pct: float) -> dict[str, str]:
        """The stopping rule applied to EACH sink independently, plus the governing composite.

        `max()` resolves the pair BEFORE the rule runs, and a card level RATCHETS, so a rung whose
        allocator DEMAND grows 3.3x under a flat 15.4 GiB card level verdicts PLATEAU on the composite
        while the allocator series says GROWING. Growth on ANY sink fails the rung.
        """
        verdicts: dict[str, str] = {}
        for sink in _SINK_FIELDS:
            series = self.series(sink)
            if not series:
                continue
            try:
                verdicts[sink] = classify(series, plateau_rounds=plateau_rounds,
                                          band_pct=band_pct)
            except InsufficientRoundsError:
                verdicts[sink] = REFUSED
        return verdicts

    def trailing_rise_pct(self, sink: str, plateau_rounds: int) -> float | None:
        """Percentage change from the first to the last round of the trailing window. DISCLOSED,
        NEVER GATING: `classify` compares against the running maximum BEFORE the window, so a
        raised baseline can hide the climb this reports."""
        series = self.series(sink)
        if len(series) < plateau_rounds or plateau_rounds < 2 or series[-plateau_rounds] <= 0:
            return None
        window = series[-plateau_rounds:]
        return (window[-1] - window[0]) / window[0] * 100.0

    def as_dict(self, metric: str, *, plateau_rounds: int | None = None,
                band_pct: float | None = None) -> dict[str, Any]:
        governing = self.measured_peaks()
        disagreements = [r.sink_disagreement_bytes for r in self.measured
                         if r.sink_disagreement_bytes is not None]
        sinks = [r.governing_sink for r in self.measured]
        return {
            "n_workers": self.n_workers, "verdict": self.verdict, "refusal": self.refusal,
            "produced_by": self.produced_by, "net_param_hash": self.net_param_hash,
            "rounds_total": len(self.rounds), "rounds_measured": len(self.measured),
            "rounds_unmeasured": len(self.scored) - len(self.measured),
            "wall_sec": round(sum(r.wall_sec for r in self.rounds), 3),
            "moves_per_min": round(self.throughput("moves_per_min"), 4),
            "games_per_min": round(self.throughput("games_per_min"), 4),
            "moves_per_min_spread": self.spread("moves_per_min"),
            "games_per_min_spread": self.spread("games_per_min"),
            "ranking_metric": metric,
            "rung_peak_bytes": max(governing) if governing else None,
            "governing_sink_counts": {sink: sinks.count(sink) for sink in set(sinks)},
            "max_sink_disagreement_bytes": max(disagreements) if disagreements else None,
            "sink_verdicts": (
                {} if plateau_rounds is None or band_pct is None
                else self.sink_verdicts(plateau_rounds=plateau_rounds, band_pct=band_pct)),
            "trailing_rise_pct": (
                {} if plateau_rounds is None
                else {sink: value for sink in _SINK_FIELDS
                      if (value := self.trailing_rise_pct(sink, plateau_rounds)) is not None}),
            "rounds": [r.as_dict() for r in self.rounds],
        }


def thread_bound() -> tuple[int, str]:
    """The box's MEASURED thread count and which call answered. `sched_getaffinity` is honest
    inside a container with a CPU mask; WHICH one answered is stamped because the two can differ."""
    getter = getattr(os, "sched_getaffinity", None)
    if getter is not None:
        return len(getter(0)), "os.sched_getaffinity(0)"
    return int(os.cpu_count() or 1), "os.cpu_count()"


def _select_sweep_buffer(config: Any, spec: Any, capacity: int) -> Any:
    """Select the replay buffer for a trainer-free pool through `BufferKind.from_spec`.

    A CLOSED match that raises on an unknown representation. A SECOND SITE, disclosed: the run's
    own `mantis.run::_select_buffer` is unreachable from here for the reason that makes this tool
    trainer-free. No encoding literal sits in a default position; visit geometry is DERIVED.
    """
    # The refusal is the BUFFER LAYER's, not the boot's: a diagnostics tool must not put a
    # run-fatal route's exception on a path no run takes.
    kind = BufferKind.from_spec(spec)
    if kind is BufferKind.GRAPH:
        from mantis._engine import HexgBuffer, derived_hexg_visit_capacity

        sp = config.selfplay
        pc = sp.playout_cap
        visit_capacity = derived_hexg_visit_capacity(
            n_simulations=sp.mcts.n_simulations, standard_sims=pc.standard_sims,
            fast_prob=pc.fast_prob, fast_sims=pc.fast_sims,
            full_search_prob=pc.full_search_prob, n_sims_quick=pc.n_sims_quick,
            n_sims_full=pc.n_sims_full, leaf_batch_size=sp.leaf_batch_size,
            gumbel_m=sp.gumbel_m, search_kind=config.search.kind,
        )
        return HexgBuffer(capacity, config.identity.encoding, visit_capacity)


def build_sweep_net(config: Any, arch: Any, device: torch.device) -> Any:
    """SEED, then build this rung's network. The seeding is the point, and it is a REPAIR.

    An unseeded build raced a DIFFERENT random net per rung, and on an unbounded board the policy
    decides how far stones spread, hence graph size, hence forward cost. Measured: at a FIXED
    worker count throughput varied 1.60x on the draw alone against 2.39x for the whole ladder.
    This process is not a run — it builds MANY pools and its output is a COMPARISON — so each
    starts from the same RNG state, seeded from the config's own `seed` before `build_net`.
    """
    from mantis.model import build_net

    seed_everything(int(config.seed))
    return build_net(arch).to(device)


def _hash_gate(pairs: list[tuple[str, str | None]]) -> dict[str, Any]:
    """Gate on net-parameter hash equality, no band, over whichever drives built a net. A drive
    that never reached a built net is excluded rather than counted as agreement or divergence."""
    present = {label: h for label, h in pairs if h is not None}
    if len(present) < 2:
        return {"verdict": REFUSED, "hashes": present,
                "reason": f"only {len(present)} drive(s) built a net to hash; R317(c)(i) needs "
                          "at least two to gate on"}
    distinct = sorted(set(present.values()))
    if len(distinct) > 1:
        return {"verdict": DIVERGED, "hashes": present,
                "reason": f"net-parameter hashes differ across drives (R317(c)(i)): "
                          f"{len(distinct)} distinct value(s) over {len(present)} drive(s) — "
                          "the same seed did not build the same net"}
    return {"verdict": AGREE, "hashes": present,
            "reason": f"net-parameter hashes agree across all {len(present)} drive(s) that "
                      "built one (R317(c)(i))"}


def build_sweep_pool(config: Any, *, n_workers: int, device: torch.device) -> WorkerPool:
    """Build the self-play collaborators for ONE rung — model, buffer, pool. No trainer.

    `n_workers` arrives through `WorkerPool`'s own override seam, never by editing a config on
    disk: varying it is the measurement."""
    from mantis.model import arch_from_spec_and_config

    raw = config.model_dump()
    resolved = resolve_pool_encoding(raw, arch=None)
    arch = arch_from_spec_and_config(resolved.registry_spec, raw)
    model = build_sweep_net(config, arch, device)
    capacity = int(resolve_coordinator_knobs(config.train).capacity)
    buffer = _select_sweep_buffer(config, resolved.registry_spec, capacity)
    return WorkerPool(model=model, config=raw, device=device, replay_buffer=buffer,
                      n_workers=n_workers, arch=arch)


def _verdict_for(rounds: tuple[RoundReading, ...], plan: SweepPlan) -> tuple[str, str | None]:
    """Apply the imported stopping rule to EVERY sink, failing the rung on growth in ANY of them.
    REFUSED IS NEVER A VERDICT: too few measured rounds is a refusal about the drive."""
    stub = RungResult(n_workers=0, verdict="", rounds=rounds, refusal=None, produced_by="")
    peaks = stub.series("governing")
    try:
        classify(peaks, plateau_rounds=plan.plateau_rounds, band_pct=plan.band_pct)
    except InsufficientRoundsError as exc:
        return REFUSED, str(exc)
    verdicts = stub.sink_verdicts(plateau_rounds=plan.plateau_rounds, band_pct=plan.band_pct)
    growing = sorted(sink for sink, v in verdicts.items() if v == GROWING)
    if growing:
        return GROWING, (
            f"growth seen on: {', '.join(growing)} (all sink verdicts: {verdicts}). The block's "
            "rule decides which FIGURE governs; it does not decide which SERIES the stopping "
            "rule is asked about, and a rung that grows on either instrument has not converged."
        )
    return PLATEAU, None


def drive_rung(
    config: Any, plan: SweepPlan, *, n_workers: int, device: torch.device, label: str,
    out: Any, sleep: Any = time.sleep,
) -> RungResult:
    """Run one rung: build a fresh pool, walk the rounds, verdict the series, tear down.

    FRESH POOL PER RUNG, or the rung measures the ladder's history. EVERY FAILURE IS A RUNG
    VERDICT, NOT A TRACEBACK: an escaping exception reaches the interpreter as shell rc 1, which
    this tool reserves for "no rung PASSED".
    """
    device_str = str(device)
    counters = cuda_counters_available(device_str)
    readings: list[RoundReading] = []
    pool: Any = None
    sampler: CardSampler | None = None
    started = False
    teardown_note: str | None = None
    net_hash: str | None = None

    def _finish(verdict: str, refusal: str | None) -> RungResult:
        note = refusal if teardown_note is None else f"{refusal or ''} | {teardown_note}".strip()
        return RungResult(n_workers=n_workers, verdict=verdict, rounds=tuple(readings),
                          refusal=note or None, produced_by=label, net_param_hash=net_hash)

    emit_marker({"phase": "rung_start", "n_workers": n_workers, "counters": counters,
                 "produced_by": label}, out=out)
    try:
        try:
            # The BUILD is inside the guard: a rung can OOM constructing the model or the ring, and
            # that is the same data as a rung that OOMs mid-drive.
            pool = build_sweep_pool(config, n_workers=n_workers, device=device)
            # Hashed POST-SEED, PRE-PLAY — before `pool.start()` lets any worker touch the net, so
            # a later divergence cannot be blamed on this read.
            net_hash = net_param_hash(pool.model)
            sampler = CardSampler(device_str, plan.sampler_interval_sec) if counters else None
            pool.start()
            started = True
            if sampler is not None:
                sampler.start()
            for index in range(plan.rounds_per_rung):
                warmup = index < plan.warmup_rounds
                probe = make_probe(device_str, round_id=f"w{n_workers}r{index}", out=_Discard())
                if counters:
                    reset_cuda_peak_counters(device_str)
                if sampler is not None:
                    sampler.reset()
                before = runner_stats(pool)
                start = time.monotonic()
                emit_marker({"phase": f"round_start:{index}", "n_workers": n_workers,
                             "warmup": warmup, "produced_by": label,
                             **probe.mark(f"round_start:{index}")}, out=out)
                sleep(plan.round_sec)
                # THE POOL'S OWN FAIL-FAST HOOK, at every round boundary: `_stats_loop` stores the
                # drain loop's exception and does NOT raise, so a dead feeder reads as a flat
                # memory series while `runner_stats` keeps climbing off the Rust counters.
                pool.check_producer_health()
                if sampler is not None and (sampler_error := sampler.error()) is not None:
                    raise SweepRefusal(
                        f"the card sampler thread died during round {index}: {sampler_error}. "
                        "The rung's series would change instrument mid-flight, and a figure that "
                        "FELL for that reason reads as memory released."
                    )
                after = runner_stats(pool)
                elapsed = time.monotonic() - start
                end_mark = probe.mark(f"round_end:{index}")
                sampled_peak, samples = sampler.window() if sampler is not None else (None, 0)
                # The Rust counters are monotone, so a NEGATIVE delta means they were reset or the
                # runner replaced mid-rung and the rate spans two programs. Clamping to 0 hid that.
                games_delta = after.games_completed - before.games_completed
                moves_delta = after.positions_generated - before.positions_generated
                if games_delta < 0 or moves_delta < 0:
                    raise SweepRefusal(
                        f"round {index} measured a NEGATIVE counter delta (games "
                        f"{games_delta}, moves {moves_delta}): the runner's counters are "
                        "monotone, so this means they were reset or the runner was replaced "
                        "mid-rung. The rung's rate would be taken across two programs; no "
                        "reading is recorded."
                    )
                reading = RoundReading(
                    index=index, warmup=warmup, wall_sec=elapsed,
                    games=games_delta,
                    moves=moves_delta,
                    available=bool(end_mark.get("max_memory_allocated_bytes") is not None),
                    sampled_peak_bytes=sampled_peak,
                    allocator_peak_bytes=end_mark.get("max_memory_allocated_bytes"),
                    card_samples=samples,
                )
                readings.append(reading)
                emit_marker({"phase": f"round_end:{index}", "n_workers": n_workers,
                             "produced_by": label, **reading.as_dict(), **end_mark}, out=out)
        finally:
            # TEARDOWN IN ITS OWN GUARD: a raise here REPLACES the return value, so a join failure
            # — likeliest exactly when a rung has just OOM'd — erased the OOM finding and killed
            # the ladder. `started` is the predicate; the try/except is for every other cause.
            if sampler is not None:
                sampler.stop()
            if started and pool is not None:
                try:
                    pool.stop()
                except Exception as exc:  # noqa: BLE001 — recorded on the rung, never swallowed
                    teardown_note = f"teardown failed after this rung: {exc!r}"
            emit_marker({"phase": "rung_end", "n_workers": n_workers, "produced_by": label,
                         "teardown_note": teardown_note}, out=out)
    except torch.OutOfMemoryError as exc:
        # DATA, not a sitting failure: the rung fails and the EXTENSION stops.
        emit_marker({"phase": "rung_oom", "n_workers": n_workers, "produced_by": label}, out=out)
        return _finish(OOM, f"CUDA out of memory at {n_workers} workers: {exc}")
    except SweepRefusal as exc:
        return _finish(REFUSED, str(exc))
    except KeyboardInterrupt:
        # AN EXPLICIT DECISION: at ~70 minutes for the base ladder on a rented box, losing every
        # measured rung to a Ctrl-C is the expensive outcome. Named as interrupted and re-raised.
        emit_marker({"phase": "rung_interrupted", "n_workers": n_workers, "produced_by": label},
                    out=out)
        raise
    except Exception as exc:  # noqa: BLE001 — a rung failure with a NAME, never a lost ladder
        emit_marker({"phase": "rung_error", "n_workers": n_workers, "produced_by": label,
                     "error": repr(exc)}, out=out)
        return _finish(RUNG_ERROR, f"{n_workers} workers failed with {exc!r}")

    if (producer_exc := getattr(pool, "_producer_exc", None)) is not None:
        return _finish(PRODUCER_DEAD, f"the self-play buffer feeder died: {producer_exc!r}")
    scored = [r for r in readings if not r.warmup]
    if scored and sum(r.moves for r in scored) == 0:
        return _finish(REFUSED,
                       f"{n_workers} workers generated NO moves across {len(scored)} measured "
                       f"round(s) totalling {sum(r.wall_sec for r in scored):.1f}s — that is a "
                       "sampling limit, not a throughput of zero, and this tool will not report "
                       "it as one")
    verdict, refusal = _verdict_for(tuple(readings), plan)
    return _finish(verdict, refusal)


def walk_ladder(plan: SweepPlan, *, runner: Any, label: str) -> tuple[list[RungResult], str]:
    """Walk the base ladder WHOLE, then extend while gains persist and discipline holds.

    THE BASE BRACKET IS PRE-REGISTERED AND IS WALKED IN FULL — skipping rungs above the measured
    thread count once cut `2, 4, 8, 12, 14` to `2, 4, 8`; the bound attaches to the EXTENSION.
    An OOM fails its own rung, the base rungs above it are STILL WALKED, and only the extension
    closes. EXTENSION STARTS ABOVE THE LAST RUNG RUN, never above the best PASSING one, so the
    walk terminates; keying on an enumerated verdict set re-drove one rung forever.

    Returns the results and the STATED reason the walk stopped.
    """
    bound, bound_source = thread_bound()
    ceiling = min(plan.extension_max, bound)
    results: list[RungResult] = []
    oom_at: int | None = None
    for rung in plan.rungs:
        result = runner(rung)
        results.append(result)
        if result.verdict == OOM and oom_at is None:
            oom_at = rung

    while True:
        highest = results[-1]
        if oom_at is not None:
            return results, (
                f"rung {oom_at} OOM'd; R309(f) stops the ladder's EXTENSION there. The base "
                "rungs above it were still walked, because that is what the clause says")
        if highest.verdict != PLATEAU:
            return results, (f"the highest rung run ({highest.n_workers}) did not PASS "
                             f"({highest.verdict}); no extension")
        passing = [r for r in results if r.verdict == PLATEAU]
        prior = [r for r in passing if r.n_workers != highest.n_workers]
        if prior:
            prior_best = max(r.throughput(plan.metric) for r in prior)
            gain_pct = (((highest.throughput(plan.metric) - prior_best) / prior_best * 100.0)
                        if prior_best > 0 else 0.0)
            if gain_pct <= plan.min_gain_pct:
                return results, (
                    f"rung {highest.n_workers} gained {gain_pct:.2f}% on {plan.metric} over the "
                    f"best prior passing rung, at or below the {plan.min_gain_pct:g}% floor — "
                    "gains no longer persist (R309(f))")
        nxt = highest.n_workers + plan.extension_step
        if nxt > ceiling:
            which = ("[ladder].extension_max" if plan.extension_max <= bound
                     else f"the measured thread bound {bound} ({bound_source})")
            return results, f"the next extension rung {nxt} is above {which}"
        result = runner(nxt)
        results.append(result)
        if result.verdict == OOM:
            oom_at = nxt


AGREE = "AGREE"
DIVERGED = "DIVERGED"


def determinism_verdict(first: dict[str, Any], second: dict[str, Any], *,
                        metric: str) -> dict[str, Any]:
    """Two drives of the SAME rung under the SAME seed: did they build the SAME net?

    THE GATE IS NET-PARAMETER-HASH EQUALITY, NO BAND. The former throughput band was measured
    FALSE — 0.5821% AGREE engine-side against 3.9258% DIVERGED on a live box at the same rung —
    because `moves_per_min` conflates what a seed controls with wall-clock scheduling. The spread
    is still carried, REPORTED and NON-GATING. Pure over the two rung rows.

    Raises:
        ValueError: either row is missing `n_workers` or the ranking column, the two rows are
            not the SAME rung (which would make the comparison a ladder step, not a control),
            or a ranking value is non-finite.
    """
    rows = (first, second)
    for row in rows:
        if metric not in row:
            raise ValueError(f"a determinism-control row carries no {metric!r} column")
        value = row[metric]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(value):
            raise ValueError(
                f"a determinism-control row's {metric} is {value!r}. NaN and +/-inf are values "
                "to `json.loads`, and a control that ranks one would report agreement about a "
                "number that is not a measurement"
            )
    if first.get("n_workers") != second.get("n_workers"):
        raise ValueError(
            f"the determinism control compares ONE rung with itself; got n_workers "
            f"{first.get('n_workers')!r} and {second.get('n_workers')!r}. Two different rungs "
            "would make this a ladder step wearing the control's name"
        )
    values = [float(row[metric]) for row in rows]
    undecidable = [row.get("verdict") for row in rows
                   if row.get("verdict") in (REFUSED, OOM, RUNG_ERROR, PRODUCER_DEAD)]
    block = {
        "n_workers": first.get("n_workers"), "metric": metric,
        "first": values[0], "second": values[1],
        "verdicts": [row.get("verdict") for row in rows],
    }
    gate = _hash_gate([("first", first.get("net_param_hash")),
                       ("second", second.get("net_param_hash"))])
    if undecidable or min(values) <= 0:
        return {**block, "spread_pct": None, "net_hash_gate": gate, "verdict": REFUSED,
                "reason": (f"a drive is not a measurement to compare: verdicts {block['verdicts']}"
                           f", {metric} {values}. REFUSED is never a verdict about determinism")}
    spread = abs(values[1] - values[0]) / min(values) * 100.0
    return {**block, "spread_pct": spread, "net_hash_gate": gate, "verdict": gate["verdict"],
            "reason": (f"{gate['reason']}; throughput spread {spread:.4f}% on {metric} "
                       "REPORTED with no band (R317(c)(iii))")}


def select_knee(rows: list[dict[str, Any]], *, knee_pct: float, metric: str) -> dict[str, Any]:
    """Apply the ruled knee rule as a pure function over the report's own rung rows.

    PURE, so `--select-only` re-derives the pick through THIS function and not a second copy, and
    the returned block carries every input the rule ran on. THE NOISE TERM IS PER-RUNG: the
    widening is `3 × max(rel_se over the PASSING rungs) × best`, which can only ADD rungs, and the
    pick is the SMALLEST member of `within` — so noise can only move the pick toward FEWER workers.

    Raises:
        ValueError: a passing rung carries no finite, non-negative `rel_se`, in addition to
            every pre-existing validation this function performs.
    """
    if knee_pct != RULED_KNEE_PCT:
        raise ValueError(
            f"knee_pct is {knee_pct}; R309(f) fixes the knee rule at {RULED_KNEE_PCT:g} percent. "
            "This is checked in BOTH modes, from SOURCE — a report is a file, and a file can be "
            "edited between the drive that wrote it and the reader that quotes it."
        )
    if metric != PREREG_METRIC:
        raise ValueError(
            f"metric is {metric!r}; the ranking metric is pre-registered as {PREREG_METRIC!r} "
            "(DESIGN amendment A1). Both figures are recorded for every rung; only the RANKING "
            "is single-valued."
        )
    passing = []
    for row in rows:
        if row.get("verdict") != PLATEAU:
            continue
        # THE ROWS ARE VALIDATED, because in `--select-only` they are whatever a file says: a
        # three-key hand-written dict once printed `PICK = 1`, the one value the rule REJECTS.
        n_workers = row.get("n_workers")
        if not isinstance(n_workers, int) or isinstance(n_workers, bool) or n_workers < 2:
            raise ValueError(
                f"rung row carries n_workers={n_workers!r}. R309(f) REJECTS n_workers = 1 and "
                "this reader will not print a pick it would have refused to measure."
            )
        if metric not in row:
            raise ValueError(f"rung {n_workers} carries no {metric!r} column")
        value = row[metric]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(value):
            raise ValueError(
                f"rung {n_workers}'s {metric} is {value!r}. NaN and +/-inf are values to "
                "`json.loads`: a NaN row used to vanish from the knee set with no refusal, and "
                "an inf row used to capture the pick."
            )
        spread = row.get(f"{metric}_spread")
        rel_se = spread.get("rel_se") if isinstance(spread, dict) else None
        if rel_se is None:
            raise ValueError(
                f"rung {n_workers} carries no measured rel_se for {metric!r}: its rounds cannot "
                "state their own noise (fewer than two scored rounds, or a report written before "
                "R330(d)). A pick is not selected under a noise term nobody measured, and None is "
                "not read as zero."
            )
        if not isinstance(rel_se, (int, float)) or isinstance(rel_se, bool) \
                or not isfinite(rel_se) or rel_se < 0:
            raise ValueError(
                f"rung {n_workers}'s rel_se is {rel_se!r}; R330(d) requires a finite, non-negative "
                "relative standard error measured from the rung's own rounds."
            )
        passing.append({"n_workers": n_workers, "value": float(value), "rel_se": float(rel_se)})
    passing.sort(key=lambda p: p["n_workers"])
    notes = sorted({str(r.get("verdict")) for r in rows
                    if r.get("verdict") in (OOM, RUNG_ERROR, PRODUCER_DEAD, REFUSED)})
    if not rows:
        return {"knee_pct": knee_pct, "metric": metric, "passing": [], "best": None,
                "threshold": None, "per_rung_rel_se": {}, "noise_rel_se_max": None,
                "noise_source_rung": None, "noise_adjustment": None, "adjusted_threshold": None,
                "within": [], "picked": None, "notes": notes,
                "reason": "the report carries NO RUNGS at all — this is a statement about the "
                          "document, not about the card's memory"}
    if not passing:
        return {"knee_pct": knee_pct, "metric": metric, "passing": [], "best": None,
                "threshold": None, "per_rung_rel_se": {}, "noise_rel_se_max": None,
                "noise_source_rung": None, "noise_adjustment": None, "adjusted_threshold": None,
                "within": [], "picked": None, "notes": notes,
                "reason": "no rung PASSED (a PLATEAU memory verdict is required, R309(f))"}
    best = max(passing, key=lambda p: p["value"])
    if best["value"] <= 0:
        raise ValueError(
            f"every passing rung ranks at {best['value']} on {metric}: the ranking column cannot "
            "order the ladder, so the knee rule has nothing to apply. Refusing rather than "
            "picking the smallest rung off an identically-zero table."
        )
    threshold = best["value"] * (knee_pct / 100.0)
    # The MAX rel-SE over the candidate (passing) set, and the ONLY safe post-hoc direction:
    # subtracting from the threshold can only ADD rungs, and the pick is the SMALLEST member.
    noise_source = max(passing, key=lambda p: p["rel_se"])
    adjustment = 3.0 * noise_source["rel_se"] * best["value"]
    adjusted_threshold = threshold - adjustment
    within = [p for p in passing if p["value"] >= adjusted_threshold]
    picked = min(within, key=lambda p: p["n_workers"])
    return {
        "knee_pct": knee_pct, "metric": metric, "passing": passing, "best": best,
        "threshold": threshold,
        "per_rung_rel_se": {str(p["n_workers"]): p["rel_se"] for p in passing},
        "noise_rel_se_max": noise_source["rel_se"], "noise_source_rung": noise_source["n_workers"],
        "noise_adjustment": adjustment, "adjusted_threshold": adjusted_threshold,
        "within": within, "picked": picked["n_workers"],
        "notes": notes,
        "reason": (f"the smallest passing rung within {knee_pct:g}% of the best passing rung's "
                   f"{metric}, widened by R330(d)'s 3-sigma per-rung noise term (max rel-SE "
                   f"{noise_source['rel_se']:.4%} at rung {noise_source['n_workers']}, "
                   f"-{adjustment:.4f} off the threshold)"),
    }


def _sha256(path: Path | str) -> str | None:
    """The config's REAL SHA-256. `git hash-object` returns git's blob hash — SHA-1 over
    `blob <len>\0<content>` — which under a field named `config_sha256` makes a later `sha256sum`
    look like a changed config."""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def provenance(config: Any, config_path: Path, *, device: str, label: str) -> dict[str, Any]:
    """Record what produced every figure in this report.

    NO HOST IDENTIFIERS: GPU model and CPU count are regime facts; a hostname, home path or
    provider name is never read here. The live allocator conf is read through the ONE authority
    and its SOURCE VARIABLE is stamped beside it, since only one of the two c10 reads may be set.
    """
    live = read_live_allocator_conf()
    bound, bound_source = thread_bound()
    cuda = torch.cuda.is_available()
    commit = _git("rev-parse", "HEAD")
    porcelain = _git("status", "--porcelain")
    return {
        "tool": TOOL,
        "produced_by": label,
        # CAPTURE-TIME REDACTION, and it has to be here rather than in a later scan: this report
        # lands in a governance workspace and a sitting record, which CI gate 17 does not scan.
        "config_name": Path(config_path).name,
        "config_sha256": _sha256(config_path),
        "git_commit": commit,
        # `None`, not `False`, when git could not answer: `bool(None)` is a POSITIVE CLAIM OF
        # CLEANLINESS. Keyed on `porcelain`, since keying on `commit` published clean when it failed.
        "git_dirty": None if porcelain is None else bool(porcelain),
        "run_id": getattr(config, "run_id", None),
        # THE SEED EVERY RUNG'S NETWORK WAS BUILT FROM, carried so a reader can SEE that the
        # ranking column is a function of `n_workers` alone rather than having to trust it.
        "seed": int(config.seed),
        "encoding": config.identity.encoding,
        "representation": config.identity.representation,
        "device": device,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": cuda,
        "cuda_counters_available": cuda_counters_available(device),
        "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
        # The capacity every peak here is implicitly measured against: without it a reader cannot
        # size a figure, and the downstream partition is an inequality against exactly this number.
        "card_total_bytes": (cuda_device_total_bytes(device)
                             if cuda_counters_available(device) else None),
        "thread_bound": bound,
        "thread_bound_source": bound_source,
        "declared_allocator_posture": declared_allocator_posture(config.model_dump()),
        "allocator_posture_governs_device": governs_device(device),
        "live_allocator_conf": live.raw,
        "live_allocator_conf_source_var": live.source_var,
    }


def build_report(*, plan: SweepPlan, prov: dict[str, Any], results: list[RungResult],
                 stopped: str) -> dict[str, Any]:
    rows = [r.as_dict(plan.metric, plateau_rounds=plan.plateau_rounds,
                      band_pct=plan.band_pct) for r in results]
    # The ladder-wide gate: every rung shares one seed and one config, so every net any rung built
    # must hash equal — a divergence means the ranking column is not comparable across rungs.
    gate = _hash_gate([(str(r.n_workers), r.net_param_hash) for r in results])
    selection = select_knee(rows, knee_pct=plan.knee_pct, metric=plan.metric)
    if gate["verdict"] == DIVERGED:
        selection = {**selection, "picked": None,
                     "reason": f"VOIDED by R317(c)(i): {gate['reason']} — the ranking column is "
                               "not comparable across rungs built from different nets"}
    return {
        "tool": TOOL,
        "prereg": dict(plan.provenance),
        "plan": {
            "rungs": list(plan.rungs), "extension_step": plan.extension_step,
            "extension_max": plan.extension_max, "min_gain_pct": plan.min_gain_pct,
            "warmup_rounds": plan.warmup_rounds, "measured_rounds": plan.measured_rounds,
            "round_sec": plan.round_sec, "sampler_interval_sec": plan.sampler_interval_sec,
            "plateau_rounds": plan.plateau_rounds, "band_pct": plan.band_pct,
            "knee_pct": plan.knee_pct, "metric": plan.metric,
        },
        "provenance": prov,
        "ladder_stopped_because": stopped,
        "rungs": rows,
        "net_hash_gate": gate,
        "selection": selection,
    }


def rc_for(report: dict[str, Any]) -> int:
    """0 a pick · 1 measurable but nothing PASSED · 2 nothing was measurable at all. A DIVERGED
    net_hash_gate REFUSES: it is a statement about the INSTRUMENT, where an all-GROWING ladder is
    data."""
    if report.get("net_hash_gate", {}).get("verdict") == DIVERGED:
        return RC_REFUSED
    if report["selection"]["picked"] is not None:
        return 0
    decisive = [r for r in report["rungs"] if r["verdict"] in (PLATEAU, GROWING, OOM)]
    return 1 if decisive else RC_REFUSED


def _gib(value: Any) -> str:
    return "unmeasured" if value is None else f"{value / GIB:.4f} GiB"


def render(report: dict[str, Any], out: Any) -> None:
    """The human screen. Every figure beside its sampling limit and its producing run."""
    prov = report["provenance"]
    plan = report["plan"]
    print(f"{TOOL} — produced_by={prov['produced_by']}", file=out)
    if not prov["cuda_counters_available"]:
        print(
            "MECHANISM EVIDENCE ONLY — this host has no CUDA counters. No figure below is a\n"
            "floor, a bound, or a comparison point for any other host. Cross-host numbers are\n"
            "mechanism evidence and nothing else.", file=out,
        )
    print(f"prereg: {report['prereg'].get('prereg_ruling')} "
          f"(recorded {report['prereg'].get('prereg_recorded')})", file=out)
    print(f"config: {prov['config_name']} sha256={prov['config_sha256']} "
          f"commit={prov['git_commit']} dirty={prov['git_dirty']}", file=out)
    print(f"regime: device={prov['device']} torch={prov['torch_version']} "
          f"gpu={prov['gpu_name']} card_total={_gib(prov.get('card_total_bytes'))} "
          f"posture={prov['declared_allocator_posture']!r} "
          f"live_alloc_conf={prov['live_allocator_conf']!r} "
          f"(from {prov['live_allocator_conf_source_var']})", file=out)
    oversubscribed = [row["n_workers"] for row in report["rungs"]
                      if row["n_workers"] > prov["thread_bound"]]
    print(f"bound:  {prov['thread_bound']} threads via {prov['thread_bound_source']}"
          + (f"  — rungs {oversubscribed} are OVER-SUBSCRIBED and were walked anyway: R309(f) "
             "bounds the EXTENSION by the thread count, and the base bracket is pre-registered"
             if oversubscribed else ""), file=out)
    print(f"rule:   rounds={plan['warmup_rounds']}w+{plan['measured_rounds']}m of "
          f"{plan['round_sec']:g}s · plateau_rounds={plan['plateau_rounds']} "
          f"band_pct={plan['band_pct']:g} · rank on {plan['metric']} · "
          f"knee_pct={plan['knee_pct']:g}", file=out)
    print("", file=out)
    for row in report["rungs"]:
        print(f"  n_workers={row['n_workers']:>3}  {row['verdict']:<18} "
              f"moves/min={row['moves_per_min']:>10.3f}  games/min={row['games_per_min']:>8.3f}  "
              f"peak={_gib(row['rung_peak_bytes'])}", file=out)
        print(f"      sample: rounds_measured={row['rounds_measured']} "
              f"rounds_unmeasured={row['rounds_unmeasured']} wall_sec={row['wall_sec']:.1f} "
              f"— that count and that wall time are the limit, not a bound "
              f"(produced_by={row['produced_by']})", file=out)
        spread = row[f"{row['ranking_metric']}_spread"]
        if spread["n_rounds"]:
            print(f"      spread ({row['ranking_metric']}, n={spread['n_rounds']}): "
                  f"min={spread['min']} median={spread['median']} max={spread['max']} "
                  f"— the knee band is a percentage; read it against this, not against the "
                  f"point estimate alone", file=out)
        if row["governing_sink_counts"]:
            print(f"      sinks: governed by {row['governing_sink_counts']} · "
                  f"max disagreement {_gib(row['max_sink_disagreement_bytes'])} — where the two "
                  f"instruments disagree the larger governs AND the disagreement is a finding",
                  file=out)
        if row["sink_verdicts"]:
            print(f"      verdicts per sink: {row['sink_verdicts']} — the stopping rule is asked "
                  f"of EACH series, not of the pair `max()` already resolved; growth on either "
                  f"instrument fails the rung", file=out)
        rises = {sink: f"{value:+.2f}%" for sink, value in row["trailing_rise_pct"].items()}
        risen = [s for s, v in row["trailing_rise_pct"].items() if v > plan["band_pct"]]
        if rises:
            print(f"      trailing-window rise: {rises}"
                  + (f"  ⚠ {risen} rose more than the band across the window while the verdict "
                     "was taken against the running maximum BEFORE it — disclosed, non-gating: "
                     "this is the shape the stopping rule cannot see" if risen else ""),
                  file=out)
        for reading in row["rounds"]:
            flag = "" if reading["available"] else "  [unmeasured]"
            tag = "warmup" if reading["warmup"] else "      "
            print(f"        r{reading['index']} {tag} moves={reading['moves']:>7} "
                  f"games={reading['games']:>4} "
                  f"card={_gib(reading['sampled_peak_bytes'])} "
                  f"alloc={_gib(reading['allocator_peak_bytes'])} "
                  f"governing={_gib(reading['governing_peak_bytes'])} "
                  f"card_samples={reading['card_samples']}{flag}", file=out)
        if row["refusal"]:
            print(f"      REFUSAL/NOTE: {row['refusal']}", file=out)
    print("", file=out)
    print(f"ladder stopped because: {report['ladder_stopped_because']}", file=out)
    render_selection(report["selection"], out)


def render_determinism_control(control: dict[str, Any], out: Any) -> None:
    """The control's own screen: the net-hash gate, the reported spread with no band, and the
    verdict."""
    spread = control["spread_pct"]
    shown = "unmeasurable" if spread is None else f"{spread:.4f}%"
    gate = control.get("net_hash_gate", {})
    print("", file=out)
    print(f"DETERMINISM CONTROL — rung {control['n_workers']} driven twice in one process, "
          "same seed", file=out)
    print(f"  drive 1        {control['first']:12.4f} {control['metric']}", file=out)
    print(f"  drive 2        {control['second']:12.4f} {control['metric']}", file=out)
    print(f"  net hashes     {gate.get('hashes', {})}", file=out)
    print(f"  spread         {shown:>12}   REPORTED, no band (R317(c)(iii))", file=out)
    print(f"  rung verdicts  {control['verdicts']}", file=out)
    print(f"  VERDICT        {control['verdict']:>12}   {control['reason']}", file=out)


def render_selection(selection: dict[str, Any], out: Any) -> None:
    """The knee arithmetic WITH its inputs — the derivation, not just the answer."""
    print(f"selection: knee_pct={selection['knee_pct']:g} on {selection['metric']}", file=out)
    if not selection["passing"]:
        print(f"  PICK = none — {selection['reason']}", file=out)
        # THE NOTES PRINT HERE TOO, and this is the run where the reader most needs them; the line
        # once sat BELOW an early return, unreachable in exactly the case it was written for.
        if selection.get("notes"):
            print(f"  what the ladder DID return: {', '.join(selection['notes'])}", file=out)
        return
    print("  passing rungs (PLATEAU only): "
          + ", ".join(f"{p['n_workers']}@{p['value']:.3f}" for p in selection["passing"]),
          file=out)
    print(f"  best passing:                 {selection['best']['n_workers']} "
          f"@ {selection['best']['value']:.3f}", file=out)
    print(f"  threshold = {selection['best']['value']:.3f} * "
          f"{selection['knee_pct'] / 100.0:g} = {selection['threshold']:.3f}", file=out)
    print("  per-rung rel-SE (each rung's own rounds): "
          + ", ".join(f"{n}@{v:.4%}" for n, v in selection["per_rung_rel_se"].items()), file=out)
    print(f"  R330(d) noise term: max rel-SE {selection['noise_rel_se_max']:.4%} at rung "
          f"{selection['noise_source_rung']} x 3 -> -{selection['noise_adjustment']:.3f}, "
          f"adjusted threshold = {selection['adjusted_threshold']:.3f}", file=out)
    print("  at or above adjusted threshold:"
          + ", ".join(f"{p['n_workers']}@{p['value']:.3f}" for p in selection["within"]),
          file=out)
    print(f"  PICK = {selection['picked']}   ({selection['reason']})", file=out)
    if selection.get("notes"):
        print(f"  NOT in the passing set: {', '.join(selection['notes'])} — a reader who quotes "
              "the arithmetic alone would not otherwise see that the ladder had a failing rung",
              file=out)


def run_determinism_control(*, config_path: Path, plan_path: Path, n_workers: int,
                            out: Any) -> dict[str, Any]:
    """Drive ONE rung TWICE in one process and report whether the two built the SAME net.

    `build_sweep_net` seeds from the config's own `seed` before every pool build, so two drives
    must build the same network and hash equal. It lives in the shipped tool rather than a
    sitting's script, and runs in ONE process — the residual was measured position-independent
    (0.58% apart, network held fixed), so a difference here is about the seeding.

    Raises:
        AllocatorPostureMismatchError: the live allocator conf does not match the minted posture.
        SweepRefusal: the plan file is unreadable, incomplete, or moves a ruled constant.
    """
    plan = load_plan(plan_path)
    config = load_config(config_path)
    device = torch.device(config.train.device)
    assert_allocator_posture(config.model_dump(), device_type=device.type)
    label = f"{getattr(config, 'run_id', 'run')}@{_git('rev-parse', '--short', 'HEAD') or 'no-git'}"

    rows: list[dict[str, Any]] = []
    for attempt in (1, 2):
        result = drive_rung(config, plan, n_workers=n_workers, device=device,
                            label=f"{label}#determinism-{attempt}", out=out)
        row = result.as_dict(plan.metric, plateau_rounds=plan.plateau_rounds,
                             band_pct=plan.band_pct)
        row["drive"] = attempt
        rows.append(row)
    control = determinism_verdict(rows[0], rows[1], metric=plan.metric)
    prov = provenance(config, Path(config_path), device=str(device), label=label)
    return {"tool": TOOL, "mode": "determinism_control", "prereg": dict(plan.provenance),
            "provenance": prov, "drives": rows, "control": control}


def run_sweep(*, config_path: Path, plan_path: Path, out: Any) -> dict[str, Any]:
    """Load, assert the regime, walk the ladder, build the report. Writes no config, ever."""
    plan = load_plan(plan_path)
    config = load_config(config_path)
    device = torch.device(config.train.device)
    # The SAME authority `mantis.run` calls, imported and not copied: a CUDA process boots on a
    # MINTED regime or not at all. On a non-CUDA device the resolver's route scoping skips it.
    assert_allocator_posture(config.model_dump(), device_type=device.type)
    label = f"{getattr(config, 'run_id', 'run')}@{_git('rev-parse', '--short', 'HEAD') or 'no-git'}"

    def runner(n_workers: int) -> RungResult:
        return drive_rung(config, plan, n_workers=n_workers, device=device, label=label, out=out)

    results, stopped = walk_ladder(plan, runner=runner, label=label)
    prov = provenance(config, Path(config_path), device=str(device), label=label)
    return build_report(plan=plan, prov=prov, results=results, stopped=stopped)


def read_report(path: str | Path) -> dict[str, Any]:
    """Load a report this tool wrote, and REFUSE anything else — identity, not only readability:
    a reader that asks nothing about whose output a document is will print a pick off a
    hand-written dict at rc 0."""
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"{path}: not a report object ({type(report).__name__})")
    if report.get("tool") != TOOL:
        raise ValueError(
            f"{path}: `tool` is {report.get('tool')!r}, not {TOOL!r}. This reader answers about "
            "its own output and refuses to answer about a document it cannot vouch for."
        )
    for key in ("rungs", "plan", "provenance"):
        if not isinstance(report.get(key), (dict, list)):
            raise ValueError(f"{path}: the report carries no `{key}` block")
    stated = report["plan"]
    if stated.get("knee_pct") != RULED_KNEE_PCT or stated.get("metric") != PREREG_METRIC:
        raise ValueError(
            f"{path}: the report states knee_pct={stated.get('knee_pct')!r} "
            f"metric={stated.get('metric')!r}; the pre-registered rule is "
            f"{RULED_KNEE_PCT:g} / {PREREG_METRIC!r}. A report whose stated rule is not the "
            "ruling's was produced under a rule this tool cannot apply, or has been edited."
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=f"python -m {TOOL}", description=__doc__)
    parser.add_argument("--config", help="a MINTED RunConfig — the run's own, otherwise unchanged")
    parser.add_argument("--plan", help="the pre-registered sweep plan (tools/worker_sweep_plan.toml)")
    parser.add_argument("--out", help="report destination; stdout when absent")
    parser.add_argument("--select-only",
                        help="re-derive the knee from a report this tool already wrote")
    parser.add_argument("--determinism-control", type=int, metavar="N",
                        help="drive rung N TWICE in one process and report whether the two built "
                             "the SAME net, by hash equality (R317(c)(i)); walks no ladder and "
                             "picks nothing")
    args = parser.parse_args(argv)

    if args.determinism_control is not None:
        # Same discipline `--select-only` carries: naming an input a mode does not read describes
        # a run that did not happen, so it is refused BY NAME rather than ignored.
        if args.select_only:
            print("REFUSED: --determinism-control drives a rung and --select-only reads a written "
                  "report; they are different modes and naming both describes neither",
                  file=sys.stderr)
            return RC_REFUSED
        if not args.config or not args.plan:
            print("REFUSED: --determinism-control needs both --config and --plan, for the reason "
                  "the ladder does: a config this tool picked would measure a program nobody "
                  "asked about, and a plan it picked would be a pre-registration nobody wrote",
                  file=sys.stderr)
            return RC_REFUSED
        if args.determinism_control < 2:
            print(f"REFUSED: --determinism-control {args.determinism_control}; R309(f) REJECTS "
                  "n_workers = 1 and this tool will not drive a rung it would refuse to pick",
                  file=sys.stderr)
            return RC_REFUSED
        try:
            report = run_determinism_control(config_path=Path(args.config),
                                             plan_path=Path(args.plan),
                                             n_workers=args.determinism_control, out=sys.stderr)
        except KeyboardInterrupt:
            print("REFUSED: interrupted; no control was completed", file=sys.stderr)
            return RC_REFUSED
        except Exception as exc:  # noqa: BLE001 — ONE refusal path, and rc 1 is not it
            print(f"REFUSED: {exc!r}", file=sys.stderr)
            return RC_REFUSED
        if args.out is not None:
            Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True),
                                      encoding="utf-8")
            print(f"{TOOL}: determinism control written to {args.out}", file=sys.stderr)
        else:
            print(json.dumps(report, indent=2, sort_keys=True))
        render_determinism_control(report["control"], sys.stderr)
        verdict = report["control"]["verdict"]
        return 0 if verdict == AGREE else (1 if verdict == DIVERGED else RC_REFUSED)

    if args.select_only:
        # EVERY input the mode does not read is refused BY NAME. `--out` was silently ignored here
        # while `--config`/`--plan` were refused; one of the two behaviours had to be wrong.
        unread = [name for name, value in (("--config", args.config), ("--plan", args.plan),
                                           ("--out", args.out)) if value]
        if unread:
            print(f"REFUSED: --select-only re-derives a pick from a written report and runs no "
                  f"drive; {', '.join(unread)} name inputs it does not read", file=sys.stderr)
            return RC_REFUSED
        try:
            report = read_report(args.select_only)
            # The noise term is read off each rung's OWN row; a report predating the mechanism
            # refuses inside `select_knee` by rung, never re-derives under a zero.
            selection = select_knee(report["rungs"], knee_pct=RULED_KNEE_PCT,
                                    metric=PREREG_METRIC)
            produced_by = report["provenance"].get("produced_by", "<unstated>")
        except (OSError, AttributeError, KeyError, TypeError, ValueError) as exc:
            print(f"REFUSED: cannot re-derive a pick from {args.select_only}: {exc}",
                  file=sys.stderr)
            return RC_REFUSED
        print(f"{TOOL} --select-only {args.select_only}")
        print(f"produced_by={produced_by}")
        render_selection(selection, sys.stdout)
        report["selection"] = selection
        return rc_for(report)

    if not args.config or not args.plan:
        print("REFUSED: both --config and --plan are required. Neither is defaulted: a config "
              "this tool picked would measure a program nobody asked about, and a plan it "
              "picked would be a pre-registration nobody wrote.", file=sys.stderr)
        return RC_REFUSED
    if args.out is not None:
        # PROBED BEFORE THE FIRST RUNG: the write used to sit outside the guard and BEFORE the
        # render, so an unwritable path after a seventy-minute ladder lost the whole screen.
        try:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text("", encoding="utf-8")
        except OSError as exc:
            print(f"REFUSED: --out {args.out} is not writable ({exc}); refusing BEFORE the "
                  "ladder rather than after it", file=sys.stderr)
            return RC_REFUSED
    try:
        report = run_sweep(config_path=Path(args.config), plan_path=Path(args.plan),
                           out=sys.stderr)
    except KeyboardInterrupt:
        print("REFUSED: interrupted; no complete ladder was measured", file=sys.stderr)
        return RC_REFUSED
    except Exception as exc:  # noqa: BLE001 — ONE refusal path, and rc 1 is not it
        # An escaping exception exits rc 1, which this tool reserves for "no rung PASSED". A
        # malformed input must never present as a measured memory result.
        print(f"REFUSED: {exc!r}", file=sys.stderr)
        return RC_REFUSED

    payload = json.dumps(report, indent=2, sort_keys=True)
    render(report, sys.stderr)          # the screen FIRST: the numbers survive a failed write
    if args.out:
        try:
            Path(args.out).write_text(payload + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"REFUSED: could not write {args.out} ({exc}); the report follows on stdout so "
                  "the drive is not lost", file=sys.stderr)
            print(payload)
            return RC_REFUSED
    else:
        print(payload)
    return rc_for(report)


__all__ = [
    "GROWING",
    "MARKER",
    "METRICS",
    "OOM",
    "PLATEAU",
    "REFUSED",
    "RC_REFUSED",
    "PRODUCER_DEAD",
    "RUNG_ERROR",
    "CardSampler",
    "RoundReading",
    "RungResult",
    "SweepPlan",
    "SweepRefusal",
    "build_report",
    "build_sweep_pool",
    "drive_rung",
    "emit_marker",
    "load_plan",
    "main",
    "parse_sweep_markers",
    "provenance",
    "AGREE",
    "DIVERGED",
    "build_sweep_net",
    "determinism_verdict",
    "render_determinism_control",
    "run_determinism_control",
    "rc_for",
    "read_report",
    "render",
    "render_selection",
    "run_sweep",
    "select_knee",
    "thread_bound",
    "walk_ladder",
]

if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
