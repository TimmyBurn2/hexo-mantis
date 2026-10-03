"""One run's monitor: every save read on the value instrument, the calibrated exams and the ring bands, its halting rows decided."""
# >300 justify (R8): the save's reads, the lagged pairing across saves, the rates over the monitor's idle windows and
# the halt are one record per save; split, a halting row could be decided on a reading the record does not hold.
from __future__ import annotations

import importlib
import json
import logging
import os
import shutil
import signal
import time
from argparse import Namespace
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mantis.diagnostics.ring_audit import audit, check_bands, event_rows, load_bands
from mantis.diagnostics.ring_reader import load_ring
from mantis.util.hashing import sha256_file
from mantis.util.loadpkg import load_tools_package

from .exams import load_positions, read_values
from .rings import max_game_id, unseen_ring

load_tools_package("value_instrument")
_vi = importlib.import_module("value_instrument.cli")

_LOG = logging.getLogger(__name__)

#: The events the monitor reads: the saves, the rates' counters and the run's own stops.
EVENTS = ("periodic_checkpoint_save", "iteration_complete", "hard_abort", "hard_abort_after_stop", "shutdown_save",
          "clean_stop_save")
_STOPS = ("shutdown_save", "clean_stop_save")
#: Ring copies kept in the work dir: the save being read and the one before it, which the lagged read needs.
_RING_COPIES = 2


@dataclass(frozen=True)
class Setup:
    """What one monitor reads and where: every member from the command line, none defaulted."""

    run_dir: Path
    run_id: str
    out: Path
    gen_ring: Path
    gen_sha256: str
    exams: Path
    exams_sha256: str
    bands: Path
    floors: dict[str, float]
    parent: Path
    batches: int
    device: str
    threads: int
    halt: bool


@dataclass(frozen=True)
class Readers:
    """The reads a save takes, injected so the decision logic is tested without a net."""

    value: Callable[[Path, Path, Path, Path | None], dict[str, Any]]
    exams: Callable[[Path, Path, dict[str, float]], dict[str, Any]]
    bands: Callable[[Path, Path | None, Path], dict[str, Any]]


@dataclass
class _Save:
    step: int
    ckpt: Path
    ring: Path | None
    max_gid: int | None


@dataclass
class State:
    """The monitor's memory across saves: the last read save, its ring copy, the busy windows and the rate counters."""

    last: _Save | None = None
    last_saved_ts: float | None = None
    lagged_net: Path | None = None
    busy: list[tuple[float, float]] = field(default_factory=list)
    counters: list[dict[str, float]] = field(default_factory=list)
    halted: bool = False
    #: A stop's step, set by its event: the run's final save lands after the event that announces it.
    final_step: int | None = None
    ended: bool = False


def value_read(setup: Setup) -> Callable[[Path, Path, Path, Path | None], dict[str, Any]]:
    """The instrument of record's `read` at the setup's draws; `(ckpt, held-out ring, out, train ring) -> body`."""
    def read(ckpt: Path, ring: Path, out: Path, train: Path | None) -> dict[str, Any]:
        out.parent.mkdir(parents=True, exist_ok=True)
        return _vi.read(Namespace(ckpt=ckpt, heldout=ring, train=train, out=out, seed=_vi.HELDOUT_SEED,
                                  train_seed=_vi.TRAIN_SEED, fold_seed=_vi.FOLD_SEED, batches=setup.batches,
                                  threads=setup.threads, device=setup.device))
    return read


def exams_read(setup: Setup) -> Callable[[Path, Path, dict[str, float]], dict[str, Any]]:
    """Each exam's values calibrated at the save's GEN temperature against its floor; `(ckpt, gen_read, floors) -> rows`."""
    positions = load_positions(setup.exams)

    def read(ckpt: Path, gen_read: Path, floors: dict[str, float]) -> dict[str, Any]:
        values = read_values(ckpt, positions, threads=setup.threads)
        out: dict[str, Any] = {}
        for exam, floor in floors.items():
            rows = gen_read.parent / f"{gen_read.stem}.{exam}.jsonl"
            rows.write_text("".join(json.dumps({"value": v["value"]}) + "\n" for v in values if v["exam"] == exam),
                            encoding="utf-8")
            out[exam] = _vi.exams(Namespace(read=gen_read, rows=rows, field="value", net=None, floor=floor))
        return out
    return read


def bands_read(ring: Path, events: Path | None, bands: Path) -> dict[str, Any]:
    """The ring audit's rows on a save's ring and the band misses; a banded row the ring cannot measure is a miss."""
    loaded = load_ring(ring)
    rows = audit(loaded) + event_rows(events, ring_size=int(loaded.header.size))
    misses, unknown = check_bands(rows, load_bands(bands))
    return {"rows": {r.key: r.value for r in rows}, "misses": misses + [f"{k}: no such audit row" for k in unknown]}


def halting_rows(exams: dict[str, Any], bands: dict[str, Any]) -> list[str]:
    """Every halting row that fired: an exam's calibrated mean below its floor, a ring band outside."""
    fired = [f"{exam} calibrated {row['calibrated_mean']:.4f} below the floor {row['floor']}"
             for exam, row in exams.items() if not row["holds"]]
    return fired + [f"ring band {miss}" for miss in bands["misses"]]


def rates(counters: list[dict[str, float]], since: float, until: float,
          busy: list[tuple[float, float]]) -> dict[str, Any]:
    """Games, positions and steps per hour between two saves, over the whole span and over the monitor's idle windows."""
    rows = [c for c in counters if since <= c["ts"] <= until]
    out: dict[str, Any] = {"span_h": (until - since) / 3600.0, "rows": len(rows)}
    if len(rows) < 2:
        return {**out, "note": "NOT MEASURED: fewer than two iteration rows in the span"}
    keys = ("games_total", "positions_produced_total", "step")
    total = {k: rows[-1][k] - rows[0][k] for k in keys}
    hours = (rows[-1]["ts"] - rows[0]["ts"]) / 3600.0
    idle = dict.fromkeys(keys, 0.0)
    idle_h = 0.0
    for a, b in zip(rows, rows[1:], strict=False):
        if any(a["ts"] < hi and b["ts"] > lo for lo, hi in busy):
            continue
        idle_h += (b["ts"] - a["ts"]) / 3600.0
        for k in keys:
            idle[k] += b[k] - a[k]
    names = {"games_total": "games_per_h", "positions_produced_total": "positions_per_h", "step": "steps_per_h"}
    out.update({names[k]: total[k] / hours if hours > 0 else None for k in keys})
    out["idle_h"] = idle_h
    out.update({f"idle_{names[k]}": idle[k] / idle_h if idle_h > 0 else None for k in keys})
    return out


def halt_run(run_dir: Path, run_id: str) -> dict[str, Any]:
    """ONE SIGTERM to the run's process (its heartbeat's pid): the run saves, then exits; a second would tear it down. Raises: OSError, KeyError, ValueError."""
    beat = json.loads((run_dir / "logs" / f"heartbeat_{run_id}.json").read_text(encoding="utf-8"))
    pid = int(beat["pid"])
    os.kill(pid, signal.SIGTERM)
    return {"pid": pid, "signal": "SIGTERM", "ts": time.time()}


class Monitor:
    """Reads each save once, in step order, and pairs it with its lagged net on the games produced after it."""

    def __init__(self, setup: Setup, readers: Readers, state: State | None = None) -> None:
        self.setup = setup
        self.readers = readers
        self.state = state or State(lagged_net=setup.parent)
        for sub in ("saves", "reads", "work"):
            (setup.out / sub).mkdir(parents=True, exist_ok=True)

    def _log(self, row: dict[str, Any]) -> None:
        with (self.setup.out / "monitor.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, allow_nan=False) + "\n")

    def _events_file(self) -> Path | None:
        found = sorted((self.setup.run_dir / "logs").glob(f"events_{self.setup.run_id}_seg*.jsonl"))
        return found[-1] if found else None

    def on_events(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Fold a batch of event rows in: counters kept, every save read, every stop recorded; returns the save records."""
        done = []
        for row in rows:
            event = row.get("event")
            if event == "iteration_complete" and isinstance(row.get("ts"), int | float):
                self.state.counters.append({k: float(row.get(k) or 0) for k in
                                            ("ts", "games_total", "positions_produced_total", "step")})
            elif event in ("hard_abort", "hard_abort_after_stop"):
                self._log({"event": "run_hard_abort", "rule": row.get("rule"), "step": row.get("step"),
                           "message": row.get("message")})
            elif event == "periodic_checkpoint_save" and isinstance(row.get("path"), str) and not self.state.halted:
                done.append(self.read_save(int(row["step"]), Path(row["path"]), float(row.get("ts") or time.time())))
            elif event in _STOPS:
                self._log({"event": f"run_{event}", "step": row.get("step")})
                if isinstance(row.get("step"), int):
                    self.state.final_step = int(row["step"])
        return done

    def final_save(self, timeout_s: float, poll_s: float = 5.0) -> dict[str, Any] | None:
        """Read the stop's save once its ring lands (a save already read is not re-read); `None` past `timeout_s`."""
        step = self.state.final_step
        self.state.ended = True
        if step is None or self.state.halted or (self.state.last is not None and self.state.last.step == step):
            return None
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            found = sorted((self.setup.run_dir / "checkpoints").glob(f"{self.setup.run_id}_{step:08d}_*.ckpt"))
            if found and Path(f"{found[-1]}.ring.bin").is_file():
                return self.read_save(step, found[-1], time.time())
            time.sleep(poll_s)
        self._log({"event": "final_save_missing", "step": step, "timeout_s": timeout_s})
        return None

    def read_save(self, step: int, ckpt: Path, saved_ts: float) -> dict[str, Any]:
        """One save: the GEN read, the exams at its temperature, the bands, the rates, the previous save's lagged read; may halt."""
        s, out = self.setup, self.setup.out
        record_path = out / "saves" / f"{step:08d}.json"
        t0 = time.time()
        ring = Path(f"{ckpt}.ring.bin")
        copy = out / "work" / f"ring_{step:08d}.bin"
        have_ring = ring.is_file()
        if have_ring:
            shutil.copyfile(ring, copy)
        gen_read = out / "reads" / f"gen_{step:08d}.json"
        gen = self.readers.value(ckpt, s.gen_ring, gen_read, None)
        exams = self.readers.exams(ckpt, gen_read, s.floors)
        bands = self.readers.bands(copy, self._events_file(), s.bands) if have_ring else {
            "rows": {}, "misses": ["NOT MEASURED: the save carries no ring"]}
        record: dict[str, Any] = {
            "step": step, "ckpt": str(ckpt), "ckpt_sha256": sha256_file(ckpt), "saved_ts": saved_ts,
            "gen": {"cf_ce": gen["heldout"]["overall"]["cf_ce"], "temperature": gen["heldout"]["overall"]["temperature"],
                    "auc": gen["heldout"]["overall"]["auc"], "policy_ce": gen["heldout"]["policy_ce"],
                    "bands": {b: gen["heldout"][b] for b in ("plies_0_10", "plies_11_40", "plies_41_up")},
                    "read": str(gen_read)},
            "exams": exams, "ring_bands": bands,
        }
        if self.state.last is not None and have_ring:
            record["lagged_of"] = self._lagged(self.state.last, copy)
        since = self.state.last_saved_ts
        if since is None:
            since = self.state.counters[0]["ts"] if self.state.counters else saved_ts
        record["rates"] = rates(self.state.counters, since, saved_ts, self.state.busy)
        record["halting_rows"] = fired = halting_rows(exams, bands)
        self.state.busy.append((t0, time.time()))
        record["monitor_busy_s"] = round(time.time() - t0, 1)
        record_path.write_text(json.dumps(record, indent=1, allow_nan=False), encoding="utf-8")
        self._log({"event": "save_read", "step": step, "halting_rows": fired, "record": str(record_path)})
        self._advance(_Save(step, ckpt, copy if have_ring else None,
                            max_game_id(copy) if have_ring else None), saved_ts)
        if fired:
            self._halt(step, fired)
        return record

    def _lagged(self, prev: _Save, later_ring: Path) -> dict[str, Any]:
        """The previous save and ITS lagged net, both read on the games produced after the previous save."""
        if prev.max_gid is None or self.state.lagged_net is None:
            return {"note": "NOT MEASURED: the previous save carried no ring"}
        unseen = self.setup.out / "work" / f"unseen_{prev.step:08d}.bin"
        meta = unseen_ring(later_ring, prev.max_gid, unseen)
        cur = self.setup.out / "reads" / f"unseen_{prev.step:08d}_current.json"
        lag = self.setup.out / "reads" / f"unseen_{prev.step:08d}_lagged.json"
        own = self.readers.value(prev.ckpt, unseen, cur, prev.ring)
        self.readers.value(self.state.lagged_net, unseen, lag, None)
        paired = _vi.lagged(Namespace(current=cur, lagged=lag, effect=0.012))
        unseen.unlink(missing_ok=True)
        ov = own["heldout"]["overall"]
        return {"step": prev.step, "lagged_net": str(self.state.lagged_net), "unseen": meta,
                "current": {"cf_ce": ov["cf_ce"], "temperature": ov["temperature"], "auc": ov["auc"],
                            "policy_ce": own["heldout"]["policy_ce"], "gap": own.get("gap")},
                "diff": paired["diff"], "ci": paired["ci"], "se_game": paired["se_game"], "games": paired["games"],
                "detection": paired["detection"], "worse": paired["worse"],
                "bands": {b: {k: v.get(k) for k in ("diff", "ci", "n")} for b, v in paired["bands"].items()}}

    def _advance(self, save: _Save, saved_ts: float) -> None:
        if self.state.last is not None:
            self.state.lagged_net = self.state.last.ckpt
        self.state.last = save
        self.state.last_saved_ts = saved_ts
        copies = sorted((self.setup.out / "work").glob("ring_*.bin"))
        for old in copies[:-_RING_COPIES]:
            old.unlink()

    def _halt(self, step: int, fired: list[str]) -> None:
        self.state.halted = True
        body: dict[str, Any] = {"step": step, "halting_rows": fired, "armed": self.setup.halt}
        if self.setup.halt:
            body["sent"] = halt_run(self.setup.run_dir, self.setup.run_id)
        (self.setup.out / "HALT.json").write_text(json.dumps(body, indent=1), encoding="utf-8")
        self._log({"event": "halt", **body})
        _LOG.error("halting row fired at step %s: %s", step, fired)


def inputs_pinned(setup: Setup) -> None:
    """Refuse a GEN ring or exam set that is not the pinned one. Raises: ValueError."""
    for path, want in ((setup.gen_ring, setup.gen_sha256), (setup.exams, setup.exams_sha256)):
        got = sha256_file(path)
        if got != want:
            raise ValueError(f"{path} hashes {got}, not the pinned {want}")


__all__ = ["EVENTS", "Monitor", "Readers", "Setup", "State", "bands_read", "exams_read", "halt_run", "halting_rows",
           "inputs_pinned", "rates", "value_read"]
