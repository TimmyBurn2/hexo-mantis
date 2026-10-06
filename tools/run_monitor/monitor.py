"""One run's monitor: every save read on the value instrument, the calibrated exams and the ring bands, its halting rows decided."""
# >300 justify (R8): the save's reads, the lagged pairing across saves, the persisted state a restart resumes from and
# the halt are one record per save; split, a halting row could be decided on a reading the record does not hold.
from __future__ import annotations

import importlib
import json
import logging
import shutil
import time
from argparse import Namespace
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from mantis.diagnostics.ring_audit import audit, check_bands, event_rows, load_bands
from mantis.diagnostics.ring_reader import load_ring
from mantis.util.hashing import sha256_file
from mantis.util.loadpkg import load_tools_package

from .exams import load_positions, read_values
from .rings import max_game_id, unseen_ring
from .rules import band_trends, counter_row, gap_rule, halt_run, is_run, rates, run_pid, verdict

load_tools_package("value_instrument")
_vi = importlib.import_module("value_instrument.cli")

_LOG = logging.getLogger(__name__)

#: The events the monitor reads: the saves, the rates' counters, the run's own stops and each new life of the run.
EVENTS = ("periodic_checkpoint_save", "iteration_complete", "hard_abort", "hard_abort_after_stop", "shutdown_save",
          "clean_stop_save", "run_segment_started")
_STOPS = ("shutdown_save", "clean_stop_save")
#: Ring copies kept in the work dir: the save being read and the one before it, which the lagged read needs.
_RING_COPIES = 2
_PLY_BANDS = ("plies_0_10", "plies_11_40", "plies_41_up")


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
    bands_sha256: str
    floors: dict[str, float]
    line: float
    parent: Path
    batches: int
    device: str
    threads: int
    halt: bool
    gap_line: float
    floors_from_first_pass: bool
    bands_from_step: int
    two_read_bands: frozenset[str]
    #: The step of a halt on record that a run resumed past: archived once, never cleared for any other step.
    resume_past_halt: int | None


@dataclass(frozen=True)
class Readers:
    """The reads a save takes, injected so the decision logic is tested without a net."""

    value: Callable[[Path, Path, Path, Path | None], dict[str, Any]]
    exams: Callable[[Path, Path, dict[str, float]], dict[str, Any]]
    bands: Callable[[Path, Path | None, Path], dict[str, Any]]


@dataclass
class State:
    """What a restarted monitor resumes from (`state.json`): the last read save, its lagged net, the busy windows."""

    last_step: int | None = None
    last_ckpt: str | None = None
    last_ring: str | None = None
    last_max_gid: int | None = None
    last_saved_ts: float | None = None
    lagged_net: str | None = None
    busy: list[list[float]] = field(default_factory=list)
    halted: bool = False
    aborts: list[str] = field(default_factory=list)
    armed_floors: list[str] = field(default_factory=list)
    floors_live: bool | None = None
    gap_over: list[int] = field(default_factory=list)
    armed_bands: list[str] = field(default_factory=list)


def value_read(setup: Setup) -> Callable[[Path, Path, Path, Path | None], dict[str, Any]]:
    """The instrument of record's `read` at the setup's draws, the card's cache released after it; `(ckpt, held-out, out, train) -> body`."""
    def read(ckpt: Path, ring: Path, out: Path, train: Path | None) -> dict[str, Any]:
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            return _vi.read(Namespace(ckpt=ckpt, heldout=ring, train=train, out=out, seed=_vi.HELDOUT_SEED,
                                      train_seed=_vi.TRAIN_SEED, fold_seed=_vi.FOLD_SEED, batches=setup.batches,
                                      threads=setup.threads, device=setup.device))
        finally:
            if setup.device.startswith("cuda"):
                torch.cuda.empty_cache()
    return read


def exams_read(setup: Setup) -> Callable[[Path, Path, dict[str, float]], dict[str, Any]]:
    """Each exam's values calibrated at the save's GEN temperature against its floor; `(ckpt, gen_read, floors) -> rows`. Raises: ValueError, OSError (the positions file)."""
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
    """The ring audit's rows on a save's ring and the band misses; a banded row the ring cannot measure is a miss. Raises: ValueError, OSError."""
    loaded = load_ring(ring)
    rows = audit(loaded) + event_rows(events, ring_size=int(loaded.header.size))
    misses, unknown = check_bands(rows, load_bands(bands))
    return {"rows": {r.key: r.value for r in rows}, "misses": misses + [f"{k}: no such audit row" for k in unknown]}


class Monitor:
    """Reads each save once, in step order, pairs it with its lagged net on the games produced after it, and survives a restart."""

    def __init__(self, setup: Setup, readers: Readers) -> None:
        self.setup = setup
        self.readers = readers
        for sub in ("saves", "reads", "work"):
            (setup.out / sub).mkdir(parents=True, exist_ok=True)
        path = setup.out / "state.json"
        self.state = State(**json.loads(path.read_text(encoding="utf-8"))) if path.is_file() else State(
            lagged_net=str(setup.parent))
        if self.state.floors_live is None:
            self.state.floors_live = not setup.floors_from_first_pass
        if setup.resume_past_halt is not None:
            self._past_halt(setup.resume_past_halt)
        # A halt on record is never re-sent, whatever the state file says.
        self.state.halted = self.state.halted or (setup.out / "HALT.json").is_file()
        self.counters: list[dict[str, float]] = []
        self.final_step: int | None = None
        self.segment_pid: int | None = None

    def _past_halt(self, step: int) -> None:
        """Archive the halt at `step` as `HALT_<step>.json` and follow on, every arm kept; a restart finds it archived. Raises: ValueError, OSError."""
        halt, archived = self.setup.out / "HALT.json", self.setup.out / f"HALT_{step:08d}.json"
        if halt.is_file():
            on_record = json.loads(halt.read_text(encoding="utf-8")).get("step")
            if on_record != step:
                raise ValueError(f"the halt on record is at step {on_record}, not {step}: it is not cleared")
            if self.state.last_step is None or self.state.last_step < step:
                raise ValueError(f"the state never advanced past the halted save {step}: following on would re-read it")
            # The state first: a stop between the two leaves HALT.json in place, and the restart archives it again.
            self.state.halted = False
            self._persist()
            halt.replace(archived)
            self._log({"event": "halt_resumed_past", "step": step, "archived": archived.name})
        elif not archived.is_file():
            raise ValueError(f"no halt at step {step} on record in {self.setup.out}")

    def _log(self, row: dict[str, Any]) -> None:
        with (self.setup.out / "monitor.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": time.time(), **row}, allow_nan=False) + "\n")

    def _persist(self) -> None:
        tmp = self.setup.out / "state.json.tmp"
        tmp.write_text(json.dumps(asdict(self.state), indent=1), encoding="utf-8")
        tmp.replace(self.setup.out / "state.json")

    def beat(self) -> None:
        """The monitor's own liveness, for the watcher: written every poll."""
        (self.setup.out / "monitor_alive.json").write_text(json.dumps(
            {"ts": time.time(), "last_step": self.state.last_step, "halted": self.state.halted}), encoding="utf-8")

    def run_alive(self) -> bool:
        """Whether the run is up: its heartbeat's process, or the one its newest segment started under; before a heartbeat, up."""
        beat = self.setup.run_dir / "logs" / f"heartbeat_{self.setup.run_id}.json"
        return (not beat.is_file() or run_pid(self.setup.run_dir, self.setup.run_id) is not None
                or (self.segment_pid is not None and is_run(self.segment_pid, self.setup.run_id)))

    def _events_file(self) -> Path | None:
        found = sorted((self.setup.run_dir / "logs").glob(f"events_{self.setup.run_id}_seg*.jsonl"))
        return found[-1] if found else None

    def on_events(self, rows: list[dict[str, Any]]) -> None:
        """Fold a batch of event rows in: counters kept, every save not yet read read once, every stop recorded."""
        for row in rows:
            event = row.get("event")
            if event == "iteration_complete":
                counters = counter_row(row)
                if counters is not None:
                    self.counters.append(counters)
            elif event in ("hard_abort", "hard_abort_after_stop"):
                key = f"{event}:{row.get('rule')}:{row.get('step')}"
                if key not in self.state.aborts:
                    self.state.aborts.append(key)
                    self._persist()
                    self._log({"event": "run_hard_abort", "rule": row.get("rule"), "step": row.get("step"),
                               "message": row.get("message")})
            elif event == "periodic_checkpoint_save" and isinstance(row.get("path"), str):
                step = int(row["step"])
                if not self.state.halted and (self.state.last_step is None or step > self.state.last_step):
                    self.read_guarded(step, Path(row["path"]), float(row.get("ts") or time.time()), stopping=False)
            elif event in _STOPS and isinstance(row.get("step"), int):
                # A stop at or before the last read save ended an earlier life of the run, already read.
                if self.state.last_step is None or int(row["step"]) > self.state.last_step:
                    self.final_step = int(row["step"])
            elif event == "run_segment_started":
                self.final_step = None
                self.segment_pid = row["pid"] if isinstance(row.get("pid"), int) else None

    def final_save(self, timeout_s: float, poll_s: float = 5.0) -> None:
        """After a stop or a dead run: read the newest save past the last one read once its ring lands; it is never signalled."""
        if self.state.halted:
            return
        deadline = time.time() + timeout_s
        while True:
            newest = self._newest_unread()
            if newest is not None and (Path(f"{newest[1]}.ring.bin").is_file() or time.time() >= deadline):
                self.read_guarded(newest[0], newest[1], time.time(), stopping=True)
                return
            if time.time() >= deadline:
                self._log({"event": "final_save_missing", "step": self.final_step, "timeout_s": timeout_s})
                return
            time.sleep(poll_s)

    def _newest_unread(self) -> tuple[int, Path] | None:
        found = [(step, p) for p in (self.setup.run_dir / "checkpoints").glob(f"{self.setup.run_id}_*_*.ckpt")
                 if (step := _step_of(p)) is not None and (self.state.last_step is None or step > self.state.last_step)]
        return max(found) if found else None

    def read_guarded(self, step: int, ckpt: Path, saved_ts: float, *, stopping: bool) -> dict[str, Any] | None:
        """`read_save`, with a failure recorded and the follow kept alive: one bad read must not end the halting guard."""
        try:
            return self.read_save(step, ckpt, saved_ts, stopping=stopping)
        except Exception:  # noqa: BLE001 — a failed read is recorded; ending here would end the halting guard
            _LOG.exception("save %s: the read failed", step)
            self._log({"event": "save_read_failed", "step": step, "ckpt": str(ckpt)})
            copy = self.setup.out / "work" / f"ring_{step:08d}.bin"
            try:
                self._advance(step, ckpt, copy if copy.is_file() else None, saved_ts)
            except Exception:  # noqa: BLE001 — the same guard: the next save still gets read
                _LOG.exception("save %s: advancing past the failed read failed", step)
            return None

    def read_save(self, step: int, ckpt: Path, saved_ts: float, *, stopping: bool) -> dict[str, Any]:
        """One save: the GEN read, the exams at its temperature, the bands, the rates, the previous save's lagged read; may halt. Raises: OSError, KeyError."""
        s, out = self.setup, self.setup.out
        t0 = time.time()
        ring = Path(f"{ckpt}.ring.bin")
        copy = out / "work" / f"ring_{step:08d}.bin"
        have_ring = ring.is_file() and _copy(ring, copy)
        gen, exams = self._gen_and_exams(step, ckpt)
        bands = self._bands(step, copy) if have_ring else {
            "rows": {}, "misses": [], "not_measured": "the save's ring was gone before the monitor read it"}
        record: dict[str, Any] = {
            "step": step, "ckpt": str(ckpt), "ckpt_sha256": sha256_file(ckpt), "saved_ts": saved_ts, "final": stopping,
            "gen": gen, "exams": exams, "ring_bands": bands,
        }
        if self.state.last_step is not None and have_ring:
            record["lagged_of"] = self._lagged_guarded(copy)
        since = self.state.last_saved_ts
        if since is None:
            since = self.counters[0]["ts"] if self.counters else saved_ts
        record["rates"] = rates(self.counters, since, saved_ts, [(a, b) for a, b in self.state.busy])
        decided = verdict(exams, bands, armed=self.state.armed_floors, floors_live=bool(self.state.floors_live),
                          bands_live=step >= s.bands_from_step, two_read_bands=s.two_read_bands,
                          armed_bands=self.state.armed_bands)
        lagged = record.get("lagged_of") or {}
        gap = (lagged.get("current") or {}).get("gap") or {}
        rule = gap_rule(gap.get("cf_ce"), self.state.gap_over, int(lagged.get("step", step)), s.gap_line)
        fired = decided["fired"]
        # Before the record: a record that fails to write must not cost a floor or a band its arm.
        self.state.armed_floors, self.state.floors_live = decided["armed"], decided["floors_live"]
        self.state.armed_bands, self.state.gap_over = decided["armed_bands"], rule["over"]
        self.state.busy.append([t0, time.time()])
        if fired:  # first: a record that fails to write must not cost the guard its signal
            self._halt(step, fired, stopping=stopping)
        record.update({"halting_rows": fired, "reported_rows": decided["reported"], "armed_floors": decided["armed"],
                       "floors_live": decided["floors_live"], "armed_bands": decided["armed_bands"],
                       "band_trends": band_trends(bands["rows"], self._last_rows(), self.state.last_step,
                                                  s.two_read_bands),
                       "gap_rule": rule, "monitor_busy_s": round(time.time() - t0, 1)})
        (out / "saves" / f"{step:08d}.json").write_text(json.dumps(record, indent=1, allow_nan=False), encoding="utf-8")
        self._log({"event": "save_read", "step": step, "final": stopping, "halting_rows": fired,
                   "reported_rows": decided["reported"], "armed_floors": decided["armed"],
                   "armed_bands": decided["armed_bands"],
                   "unread_floors": sorted(exam for exam, row in exams.items() if row["holds"] is None)})
        if rule["fired"]:
            self._gap_fired(step, rule)
        self._advance(step, ckpt, copy if have_ring else None, saved_ts)
        return record

    def _last_rows(self) -> dict[str, Any] | None:
        """The last read save's band rows off its own record; `None` when there is none (no save read yet, its read failed)."""
        if self.state.last_step is None:
            return None
        try:
            body = json.loads((self.setup.out / "saves" / f"{self.state.last_step:08d}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        rows = (body.get("ring_bands") or {}).get("rows") if isinstance(body, dict) else None
        return rows if isinstance(rows, dict) else None

    def _gen_and_exams(self, step: int, ckpt: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        """The GEN read and the exams at its temperature; a failed read, or a read with no temperature, leaves every floor unread."""
        s = self.setup
        gen_read = s.out / "reads" / f"gen_{step:08d}.json"
        try:
            body = self.readers.value(ckpt, s.gen_ring, gen_read, None)
        except Exception:  # noqa: BLE001 — a failed GEN read leaves the floors unread; the bands must still decide
            _LOG.exception("save %s: the GEN read failed", step)
            return {"not_measured": "the GEN read failed"}, _unread(s.floors, "the GEN read failed")
        ov, held = body["heldout"]["overall"], body["heldout"]
        gen = {"cf_ce": ov["cf_ce"], "temperature": ov["temperature"], "auc": ov["auc"], "policy_ce": held["policy_ce"],
               "bands": {b: held[b] for b in _PLY_BANDS}, "read": str(gen_read)}
        if ov["temperature"] is None:
            return gen, _unread(s.floors, "the GEN read fit no temperature: a net with no skill calibrates nothing")
        try:
            return gen, self.readers.exams(ckpt, gen_read, s.floors)
        except Exception:  # noqa: BLE001 — the same: the floors stay unread and the bands still decide
            _LOG.exception("save %s: the exams failed", step)
            return gen, _unread(s.floors, "the exams read failed")

    def _bands(self, step: int, ring: Path) -> dict[str, Any]:
        try:
            return self.readers.bands(ring, self._events_file(), self.setup.bands)
        except Exception:  # noqa: BLE001 — a failed audit leaves the bands unread; the floors must still decide
            _LOG.exception("save %s: the ring audit failed", step)
            return {"rows": {}, "misses": [], "not_measured": "the ring audit failed"}

    def _lagged_guarded(self, later_ring: Path) -> dict[str, Any]:
        try:
            return self._lagged(later_ring)
        except Exception:  # noqa: BLE001 — the lagged read only reports; it must not take the halting rows down
            _LOG.exception("the lagged read of save %s failed", self.state.last_step)
            return {"step": int(self.state.last_step or 0), "note": "NOT MEASURED: the lagged read failed"}

    def _lagged(self, later_ring: Path) -> dict[str, Any]:
        """The last read save and ITS lagged net, both read on the games produced after that save."""
        prev, ring = int(self.state.last_step or 0), self.state.last_ring
        if self.state.last_max_gid is None or ring is None or self.state.lagged_net is None:
            return {"step": prev, "note": "NOT MEASURED: the previous save carried no ring"}
        unseen = self.setup.out / "work" / f"unseen_{prev:08d}.bin"
        try:
            meta = unseen_ring(later_ring, self.state.last_max_gid, unseen)
        except ValueError as exc:
            return {"step": prev, "note": f"NOT MEASURED: {exc}"}
        cur = self.setup.out / "reads" / f"unseen_{prev:08d}_current.json"
        lag = self.setup.out / "reads" / f"unseen_{prev:08d}_lagged.json"
        try:
            own = self.readers.value(Path(str(self.state.last_ckpt)), unseen, cur, Path(ring))
            self.readers.value(Path(self.state.lagged_net), unseen, lag, None)
            paired = _vi.lagged(Namespace(current=cur, lagged=lag, effect=self.setup.line))
        finally:
            unseen.unlink(missing_ok=True)
        ov = own["heldout"]["overall"]
        return {"step": prev, "lagged_net": self.state.lagged_net, "unseen": meta,
                "current": {"cf_ce": ov["cf_ce"], "temperature": ov["temperature"], "auc": ov["auc"],
                            "policy_ce": own["heldout"]["policy_ce"], "gap": own.get("gap")},
                "diff": paired["diff"], "ci": paired["ci"], "se_game": paired["se_game"], "games": paired["games"],
                "detection": paired["detection"], "worse": paired["worse"],
                "bands": {b: {k: v.get(k) for k in ("diff", "ci", "n")} for b, v in paired["bands"].items()}}

    def _advance(self, step: int, ckpt: Path, ring: Path | None, saved_ts: float) -> None:
        if self.state.last_ckpt is not None:
            self.state.lagged_net = self.state.last_ckpt
        self.state.last_step, self.state.last_ckpt, self.state.last_saved_ts = step, str(ckpt), saved_ts
        self.state.last_ring = None if ring is None else str(ring)
        self.state.last_max_gid = None if ring is None else max_game_id(ring)
        for old in sorted((self.setup.out / "work").glob("ring_*.bin"))[:-_RING_COPIES]:
            old.unlink()
        self._persist()

    def _gap_fired(self, step: int, rule: dict[str, Any]) -> None:
        """The gap rule's verdict for the operator, its first firing kept: a rate re-mint is owed; the run is never signalled."""
        first = self.setup.out / "GAP_RULE.json"
        if not first.is_file():
            first.write_text(json.dumps({"step": step, **rule}, indent=1), encoding="utf-8")
        self._log({"event": "gap_rule_fired", "step": step, **rule})
        _LOG.warning("the gap rule fired at step %s: saves %s above %s", step, rule["over"], rule["line"])

    def _halt(self, step: int, fired: list[str], *, stopping: bool) -> None:
        self.state.halted = True
        self._persist()
        body: dict[str, Any] = {"step": step, "halting_rows": fired, "armed": self.setup.halt, "final_save": stopping}
        if self.setup.halt and not stopping:
            body["signal"] = halt_run(self.setup.run_dir, self.setup.run_id)
        (self.setup.out / "HALT.json").write_text(json.dumps(body, indent=1), encoding="utf-8")
        self._log({"event": "halt", **body})
        _LOG.error("halting row fired at step %s: %s", step, fired)


def _unread(floors: dict[str, float], why: str) -> dict[str, Any]:
    return {exam: {"floor": floor, "calibrated_mean": None, "holds": None, "not_measured": why} for exam, floor in floors.items()}


def _copy(src: Path, dst: Path) -> bool:
    shutil.copyfile(src, dst)
    return True


def _step_of(ckpt: Path) -> int | None:
    parts = ckpt.name.rsplit("_", 2)
    return int(parts[-2]) if len(parts) == 3 and parts[-2].isdigit() else None


def inputs_pinned(setup: Setup) -> None:
    """Refuse a GEN ring, exam set or bands file that is not the pinned one, or a two-read band it does not band. Raises: ValueError, OSError."""
    for path, want in ((setup.gen_ring, setup.gen_sha256), (setup.exams, setup.exams_sha256),
                       (setup.bands, setup.bands_sha256)):
        got = sha256_file(path)
        if got != want:
            raise ValueError(f"{path} hashes {got}, not the pinned {want}")
    unbanded = sorted(setup.two_read_bands - set(load_bands(setup.bands)))
    if unbanded:
        raise ValueError(f"--two-read-bands names {unbanded}, which {setup.bands} does not band")


__all__ = ["EVENTS", "Monitor", "Readers", "Setup", "State", "bands_read", "exams_read", "inputs_pinned", "value_read"]
