"""The strix follower: the equal-work strix cell (or the Six ruler's) on every 15 000-step checkpoint and every promotion."""
# >300 justify (R8): one unit — the triggers (the EVENT STREAM, never polled filenames), the tail, the regime
# read, the cell (`tools/strength_frontier.py`'s) and the sidecar receipt (the stamp never touched) are one contract.
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mantis.bots.six import CACHE_ENTRIES as SIX_CACHE_ENTRIES
from mantis.bots.six import pin_record as six_pin_record
from mantis.bots.strix import _pin as strix_pin_record
from mantis.config.resolve.tactics import ARMS, arm_from_file
from mantis.util.hashing import sha256_file

EQUAL_WORK = "equal_work"
AS_SHIPPED = "as_shipped"
NET_ONLY = "net_only"  # equal work with strix's root VCF solver OFF
RULER_R6 = "ruler_r6"  # equal work with strix at its TRAINED placement_radius 6 (the driver's default is 8)
SIX30_16 = "six30_16"  # ours PUCT-256 vs the Six ruler, gen 30 at 16 nodes
#: unit -> (our sims, the opponent's sims or nodes, sidecar suffix). A strix suffix names OUR sims: the 256 series.
UNITS: dict[str, tuple[int, int, str]] = {EQUAL_WORK: (256, 256, "strix256"),
                                          AS_SHIPPED: (512, 128, "strix512"),
                                          NET_ONLY: (256, 256, "strix256_nosolver"),
                                          RULER_R6: (256, 256, "strix256_r6"),
                                          SIX30_16: (256, 16, "six30_16")}
SOLVER_OFF_UNITS = frozenset({NET_ONLY})  # every other unit is the rung on record
RADIUS_UNITS: dict[str, int] = {RULER_R6: 6}  # every other unit rides the driver's default radius
SIX_UNITS: dict[str, tuple[str, int]] = {SIX30_16: ("gen0030", 30)}  # unit -> (the pinned network, its generation)
FOLLOW_UNITS = (EQUAL_WORK, SIX30_16)  # the rulers; every other unit is a --once cell
_PLAYED_BYTES = ("engine_sha256", "net_sha256", "runtime_sha256")  # what a six receipt names as played
TRIGGER_EVENTS = ("periodic_checkpoint_save", "eval_round_complete")
#: A heartbeat younger than this at cell start names a live run in the evidence.
HEARTBEAT_LIVE_SEC = 300.0
#: The playing host is CONTENDED when any GPU's utilisation is at or above this (an idle desktop reads single digits).
GPU_BUSY_PCT = 10
#: ... or when its 1-minute load per logical CPU is at or above this (a quarter of the cores already busy).
LOAD_BUSY_PER_CPU = 0.25
#: 2: the regime is the playing host's load (`regime_evidence.host`); 1 read it off the run's heartbeat.
SIDECAR_SCHEMA_VERSION = 2
_REPO = Path(__file__).resolve().parents[1]
_STEP_IN_NAME = re.compile(r"_(\d{8})_[0-9a-f]{8}\.ckpt$")


@dataclass(frozen=True)
class Trigger:
    """One reason to read a checkpoint: `kind` is `cadence` or `promotion`; `path` when the event named it."""

    step: int
    kind: str
    path: str | None


def triggers_from_rows(rows: Iterator[Mapping[str, Any]], cadence: int, *,
                       promotions: bool = True) -> list[Trigger]:
    """The triggers in `rows`: a cadence-multiple periodic save, or (when `promotions`) a promoted round."""
    out: list[Trigger] = []
    for row in rows:
        event, step = row.get("event"), row.get("step")
        if not isinstance(step, int) or isinstance(step, bool):
            continue
        if event == "periodic_checkpoint_save" and cadence > 0 and step % cadence == 0:
            path = row.get("path")
            out.append(Trigger(step, "cadence", path if isinstance(path, str) else None))
        elif promotions and event == "eval_round_complete" and row.get("promoted") is True:
            out.append(Trigger(step, "promotion", None))
    return out


class EventTail:
    """Reads only the NEW lines of every `events_<run_id>_seg*.jsonl` in `logs/`, in segment order."""

    def __init__(self, run_dir: Path, run_id: str) -> None:
        self.logs = run_dir / "logs"
        self.prefix = f"events_{run_id}_seg"
        self.offsets: dict[Path, int] = {}

    def read_new(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in sorted(self.logs.glob(f"{self.prefix}*.jsonl")):
            start = self.offsets.get(path, 0)
            with path.open("rb") as fh:
                fh.seek(start)
                data = fh.read()
            # A partial last line (the writer mid-append) is left for the next read.
            end = data.rfind(b"\n") + 1
            self.offsets[path] = start + end
            for line in data[:end].splitlines():
                if not any(name.encode() in line for name in TRIGGER_EVENTS):
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict) and row.get("event") in TRIGGER_EVENTS:
                    rows.append(row)
        return rows


def resolve_checkpoint(run_dir: Path, run_id: str, trigger: Trigger) -> Path | None:
    """The `.ckpt` a trigger names: the event's own path when it exists, else the step's file."""
    if trigger.path and Path(trigger.path).is_file():
        return Path(trigger.path)
    found = sorted((run_dir / "checkpoints").glob(f"{run_id}_{trigger.step:08d}_*.ckpt"))
    return found[0] if found else None


def sidecar_path(checkpoint: Path, unit: str, arm: str | None = None) -> Path:
    """The receipt beside the checkpoint; an A/B arm's names the arm, so no arm shadows another or the config's."""
    tag = "" if arm is None else f".{arm}"
    return checkpoint.with_name(f"{checkpoint.name}.{UNITS[unit][2]}{tag}.json")


@dataclass(frozen=True)
class HostLoad:
    """The load of the host that plays the cell, read at cell start; `gpu_util_pct` is `None` without nvidia-smi."""

    load_1m: float
    cpu_count: int
    gpu_util_pct: tuple[int, ...] | None


def _gpu_util_pct() -> tuple[int, ...] | None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, encoding="utf-8", timeout=10, check=True).stdout
        return tuple(int(line) for line in out.split())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _usable_cpus() -> int:
    affinity = getattr(os, "sched_getaffinity", None)
    return len(affinity(0)) if affinity is not None else (os.cpu_count() or 1)


def read_host_load() -> HostLoad:
    """This machine's 1-minute load average, the CPUs this process may use, and per-GPU utilisation."""
    return HostLoad(load_1m=os.getloadavg()[0], cpu_count=_usable_cpus(), gpu_util_pct=_gpu_util_pct())


def regime(run_dir: Path, run_id: str, now: float, host: HostLoad) -> tuple[str, dict[str, Any]]:
    """CONTENDED when the host playing the cell is busy at cell start (a GPU or the CPUs at their line); a mirrored heartbeat is evidence, never the label."""
    own = run_dir / "logs" / f"heartbeat_{run_id}.json"
    ages: dict[str, float | None] = {}
    for beat in sorted({own, *run_dir.parent.glob("*/logs/heartbeat_*.json")}):
        key = "/".join(beat.parts[-3:])  # never the bare filename: a stale `<run>-preflight/` beat shares it
        try:
            ages[key] = round(now - float(json.loads(beat.read_text(encoding="utf-8"))["wall_ts"]), 1)
        except (OSError, ValueError, TypeError, KeyError):
            ages[key] = None
    live = sorted(name for name, age in ages.items() if age is not None and age < HEARTBEAT_LIVE_SEC)
    per_cpu = host.load_1m / max(host.cpu_count, 1)
    busy = per_cpu >= LOAD_BUSY_PER_CPU or any(pct >= GPU_BUSY_PCT for pct in host.gpu_util_pct or ())
    gpus = None if host.gpu_util_pct is None else list(host.gpu_util_pct)
    return ("CONTENDED" if busy else "IDLE"), {
        "heartbeat_age_sec": ages, "live": live, "heartbeat_age_sec_self": ages.get("/".join(own.parts[-3:])),
        "host": {"load_1m": host.load_1m, "cpu_count": host.cpu_count, "load_per_cpu": round(per_cpu, 4),
                 "gpu_util_pct": gpus}}


def compose_cell(checkpoint: Path, *, unit: str, step: int, games: int, concurrency: int,
                 label: str) -> dict[str, Any]:
    """The frontier cell for one checkpoint in one unit: PUCT ours, strix at its sims or Six at its nodes, paired games."""
    ours, theirs, _suffix = UNITS[unit]
    if unit in SIX_UNITS:
        return {"label": label, "candidate": str(checkpoint), "search_kind": "puct", "sims": ours, "opponent": "six",
                "six_net": SIX_UNITS[unit][0], "six_nodes": theirs, "games": games, "step": step,
                "concurrency": concurrency}
    cell = {"label": label, "candidate": str(checkpoint), "search_kind": "puct", "sims": ours,
            "opponent": "strix", "strix_sims": theirs, "games": games, "step": step,
            "concurrency": concurrency}
    # A solver-ON, default-radius cell carries neither key, so it is byte-identical to every receipt on record.
    if unit in SOLVER_OFF_UNITS:
        return {**cell, "strix_solver": False}
    return {**cell, "strix_radius": RADIUS_UNITS[unit]} if unit in RADIUS_UNITS else cell


def opponent_pin(unit: str) -> dict[str, Any]:
    """The unit's opponent as its pin declares it: strix's commit and checkpoint, or Six's commit, engine and network."""
    if unit in SIX_UNITS:
        return six_pin_record(SIX_UNITS[unit][0])
    pin = strix_pin_record() or {}
    return {"commit": pin.get("sha"), "checkpoint": pin.get("checkpoint"),
            "checkpoint_sha256": pin.get("checkpoint_sha256")}


def played_bytes_error(record: Mapping[str, Any], pin: Mapping[str, Any]) -> str | None:
    """Why a six cell's engines did not all play the pin's engine, network and runtime, or None when they did."""
    engines = dict(record.get("six_engine") or {})
    for key in _PLAYED_BYTES:
        if engines.get(key) != [pin.get(key)]:
            return f"the engines played {key} {engines.get(key)}, the pin says {pin.get(key)}"
    return None


def sidecar_record(checkpoint: Path, *, unit: str, trigger: str, record: Mapping[str, Any],
                   regime_name: str, regime_evidence: Mapping[str, Any], run_id: str,
                   started: float, finished: float, pin: Mapping[str, Any]) -> dict[str, Any]:
    """The receipt: what was read, in which unit and regime, by which net, and what it read."""
    ours, theirs, _suffix = UNITS[unit]
    readout = dict(record.get("readout") or {})
    candidate = dict((record.get("provenance") or {}).get("candidate") or {})
    cell = dict(record.get("cell") or {})
    if unit in SIX_UNITS:
        engines = dict(record.get("six_engine") or {})
        # The hashes the engines re-verified at start, when each is one value; `played_bytes_error` says otherwise.
        played = {k: v[0] for k in _PLAYED_BYTES if len(v := engines.get(k) or []) == 1}
        opponent: dict[str, Any] = {"six": {**dict(pin), **played, "generation": SIX_UNITS[unit][1], "nodes": theirs,
                                            "cache_entries": SIX_CACHE_ENTRIES, "provider": engines.get("provider"),
                                            "engine_starts": engines.get("starts"),
                                            "searches": engines.get("searches"),
                                            "stale_pending": engines.get("stale_pending")},
                                    "six_findings": record.get("six_findings")}
    else:
        opponent = {"strix": {**dict(pin), "sims": theirs, "solver": "off" if unit in SOLVER_OFF_UNITS else "on",
                              **({"radius": RADIUS_UNITS[unit]} if unit in RADIUS_UNITS else {})},
                    "strix_findings": record.get("strix_findings")}
    return {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "run_id": run_id, "checkpoint": checkpoint.name, "checkpoint_sha256": sha256_file(checkpoint),
        "step": cell.get("step"), "net_hash": candidate.get("net_hash"),
        "unit": unit, "ours": {"search_kind": "puct", "sims": ours}, **opponent,
        "trigger": trigger, "regime": regime_name, "regime_evidence": dict(regime_evidence),
        "games": readout.get("games"), "eff_n": readout.get("eff_n"), "pairs": readout.get("pairs"),
        "wins": readout.get("wins"), "losses": readout.get("losses"), "draws": readout.get("draws"),
        "wr": readout.get("wr"), "wr_ci_lower": readout.get("wr_ci_lower"),
        "wr_ci_upper": readout.get("wr_ci_upper"), "median_plies": readout.get("median_plies"),
        "sec_per_game": readout.get("sec_per_game"), "wall_sec": record.get("wall_sec"),
        "started_utc": time.strftime("%FT%TZ", time.gmtime(started)),
        "finished_utc": time.strftime("%FT%TZ", time.gmtime(finished)),
        "concurrency": cell.get("concurrency"), "label": record.get("label"), "rc": record.get("rc"),
    }


RunCell = Callable[[Mapping[str, Any]], Mapping[str, Any]]


class Follower:
    """Reads triggers, plays one cell per (checkpoint, unit), writes the receipt; skips a receipted one."""

    def __init__(self, *, run_dir: Path, run_id: str, run_cell: RunCell, unit: str = EQUAL_WORK,
                 cadence: int = 15_000, promotions: bool = True, games: int = 288,
                 concurrency: int = 8, pin: Mapping[str, Any] | None = None,
                 clock: Callable[[], float] = time.time, log: Callable[[str], None] = print,
                 host_load: Callable[[], HostLoad] = read_host_load, arm: str | None = None,
                 tactics: Mapping[str, Any] | None = None) -> None:
        self.run_dir, self.run_id, self.run_cell = run_dir, run_id, run_cell
        self.unit, self.cadence, self.promotions = unit, cadence, promotions
        self.games, self.concurrency = games, concurrency
        self.pin = dict(pin) if pin is not None else {}
        self.clock, self.log, self.host_load = clock, log, host_load
        #: The candidate's A/B arm (`None`: the config's own block) and the bridge block it resolved to.
        self.arm, self.tactics = arm, None if tactics is None else dict(tactics)
        self.tail = EventTail(run_dir, run_id)
        self.pending: dict[int, Trigger] = {}
        self.fired: list[Path] = []

    def read_one(self, checkpoint: Path, *, trigger: str) -> tuple[str, Path]:
        """Play the cell unless its receipt exists: `(receipted|written|failed, path)`; Raises: ValueError — see below."""
        out = sidecar_path(checkpoint, self.unit, self.arm)
        if out.exists():
            prior = json.loads(out.read_text(encoding="utf-8")).get("tactics") or {}
            if self.arm is not None and prior.get("block") != self.tactics:
                raise ValueError(f"{out.name} was read with another tactics block: another block is another arm, "
                                 "never a re-read of this one")
            self.log(f"follower: {out.name} exists — receipted, not re-read")
            return "receipted", out
        m = _STEP_IN_NAME.search(checkpoint.name)
        step = int(m.group(1)) if m else 0
        label = f"{self.unit}_{self.run_id}_{step}" + ("" if self.arm is None else f"_{self.arm}")
        cell = compose_cell(checkpoint, unit=self.unit, step=step, games=self.games,
                            concurrency=self.concurrency, label=label)
        if self.arm is not None:
            cell = {**cell, "tactics": self.tactics, "tactics_arm": self.arm}
        started = self.clock()
        regime_name, evidence = regime(self.run_dir, self.run_id, started, self.host_load())
        self.log(f"follower: {trigger} → {checkpoint.name} in {self.unit} ({regime_name})")
        record = self.run_cell(cell)
        finished = self.clock()
        body = sidecar_record(checkpoint, unit=self.unit, trigger=trigger, record=record,
                              regime_name=regime_name, regime_evidence=evidence, run_id=self.run_id,
                              started=started, finished=finished, pin=self.pin)
        if self.arm is not None or "tactics" in record:
            body["tactics"] = {"arm": self.arm, "block": self.tactics, **(record.get("tactics") or {})}
        error = record.get("error")
        if self.unit in SIX_UNITS and record.get("rc") == 0 and "readout" in record:
            error = played_bytes_error(record, self.pin)
        elif error is None and (record.get("rc") != 0 or "readout" not in record):
            error = f"the cell exited rc {record.get('rc')} with no readout and no error of its own; its child.log says why"
        if record.get("rc") != 0 or "readout" not in record or error is not None:
            failed = out.with_name(out.name.replace(".json", ".failed.json"))
            failed.write_text(json.dumps({**body, "error": error}, indent=1),
                              encoding="utf-8")
            self.log(f"follower: FAILED rc={record.get('rc')} — {failed.name}, no receipt: {error}")
            return "failed", failed
        out.write_text(json.dumps(body, indent=1), encoding="utf-8")
        self.fired.append(out)
        self.log(f"follower: {out.name} wr {body['wr']} [{body['wr_ci_lower']}, {body['wr_ci_upper']}]")
        return "written", out

    def poll(self) -> list[Path]:
        """One pass: new triggers join the pending set; every pending one whose checkpoint exists is read."""
        rows = iter(self.tail.read_new())
        for trig in triggers_from_rows(rows, self.cadence, promotions=self.promotions):
            self.pending.setdefault(trig.step, trig)
        written: list[Path] = []
        for step in sorted(self.pending):
            trig = self.pending[step]
            checkpoint = resolve_checkpoint(self.run_dir, self.run_id, trig)
            if checkpoint is None:
                continue
            del self.pending[step]
            status, out = self.read_one(checkpoint, trigger=trig.kind)
            if status == "written":
                written.append(out)
        return written

    def follow(self, poll_sec: float, max_polls: int | None = None) -> None:
        polls = 0
        while max_polls is None or polls < max_polls:
            self.poll()
            polls += 1
            if max_polls is None or polls < max_polls:
                time.sleep(poll_sec)


def _frontier() -> Any:
    """`tools/strength_frontier.py` by path — `tools/` is not a package and `sys.path` is not touched."""
    path = _REPO / "tools" / "strength_frontier.py"
    spec = importlib.util.spec_from_file_location("strength_frontier", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("strength_frontier", module)
    spec.loader.exec_module(module)
    return module


def _real_run_cell(config_path: str, work_dir: Path) -> RunCell:
    from mantis.config.loader import load_config
    from mantis.config.resolve.allocator_posture import governs_device

    frontier = _frontier()
    config = load_config(config_path)
    work_dir.mkdir(parents=True, exist_ok=True)
    base = frontier.base_round_spec(config, work_dir=work_dir)
    env = dict(os.environ)
    if governs_device(config.eval.worker_device) and base.allocator_posture == "expandable_segments":
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    def run(cell: Mapping[str, Any]) -> Mapping[str, Any]:
        return frontier.run_cell(cell, config=config, base=base, work_dir=work_dir,
                                 python=sys.executable, env=env)
    return run


def main(argv: list[str] | None = None) -> int:
    """`--follow` tails the run's events; `--once <ckpt>` reads one checkpoint (the parent's bridge cell)."""
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--config", required=True, help="the run's config (the eval seam the cell composes from)")
    ap.add_argument("--run-dir", type=Path, required=True, help="the run dir: logs/events_*, checkpoints/")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--work-dir", type=Path, required=True, help="where cells play (games, child logs)")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--follow", action="store_true")
    mode.add_argument("--once", type=Path, metavar="CKPT")
    ap.add_argument("--unit", choices=sorted(UNITS), default=EQUAL_WORK,
                    help=f"--follow reads a ruler unit ({', '.join(FOLLOW_UNITS)}); every unit plays --once")
    ap.add_argument("--cadence", type=int, default=15_000)
    ap.add_argument("--promotions", action=argparse.BooleanOptionalAction, default=True,
                    help="--follow: a cell on every promoted round too (R356(a)); "
                         "--no-promotions reads the cadence points only (R361(a))")
    ap.add_argument("--games", type=int, default=288)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--poll-sec", type=float, default=300.0)
    ap.add_argument("--arm", choices=ARMS, default=None,
                    help="the candidate's A/B arm over --tactics-block, named in the receipt; absent, the config's own")
    ap.add_argument("--tactics-block", type=Path, default=None, help="a search.tactics block (JSON)")
    args = ap.parse_args(argv)
    if args.follow and args.unit not in FOLLOW_UNITS:
        ap.error(f"--follow reads the ruler units {list(FOLLOW_UNITS)} only; every other unit is a --once cell")
    armed = None
    if args.arm is not None:
        try:
            armed = arm_from_file(args.arm, args.tactics_block)
        except (ValueError, OSError) as exc:
            ap.error(f"--arm {args.arm}: {exc}")
    elif args.tactics_block is not None:
        ap.error("--tactics-block names the block an --arm overlays; give the arm")
    follower = Follower(run_dir=args.run_dir, run_id=args.run_id,
                        run_cell=_real_run_cell(args.config, args.work_dir), unit=args.unit,
                        cadence=args.cadence, promotions=args.promotions, games=args.games,
                        concurrency=args.concurrency, pin=opponent_pin(args.unit), arm=args.arm, tactics=armed)
    if args.once is not None:
        status, _path = follower.read_one(args.once.resolve(), trigger="once")
        return 1 if status == "failed" else 0
    follower.follow(args.poll_sec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
