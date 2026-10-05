"""`follow` a run's event stream and read every save, or read `once` one checkpoint; a refusal prints its reason and returns 2."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

from mantis.monitor.event_tail import EventTail

from .monitor import (
    EVENTS,
    Monitor,
    Readers,
    Setup,
    bands_read,
    exams_read,
    inputs_pinned,
    value_read,
)


def _floors(text: str) -> dict[str, float]:
    """`EXAM=FLOOR,EXAM=FLOOR` as a mapping. Raises: ValueError."""
    out = {}
    for part in text.split(","):
        name, _, value = part.partition("=")
        if not name or not value:
            raise ValueError(f"--floors: {part!r} is not EXAM=FLOOR")
        out[name.strip()] = float(value)
    return out


def setup_of(a: argparse.Namespace) -> Setup:
    """The command line as a `Setup`. Raises: ValueError (malformed floors, a line that is not positive, a negative step)."""
    if not a.line > 0 or not a.gap_line > 0:
        raise ValueError(f"--line and --gap-line must be positive, got {a.line} and {a.gap_line}")
    if a.bands_from_step < 0:
        raise ValueError(f"--bands-from-step must be at least 0, got {a.bands_from_step}")
    return Setup(run_dir=a.run_dir, run_id=a.run_id, out=a.out, gen_ring=a.gen_ring, gen_sha256=a.gen_sha256,
                 exams=a.exams, exams_sha256=a.exams_sha256, bands=a.bands, bands_sha256=a.bands_sha256,
                 floors=_floors(a.floors), line=a.line, parent=a.parent, batches=a.batches, device=a.device,
                 threads=a.threads, halt=a.halt, gap_line=a.gap_line,
                 floors_from_first_pass=a.floors_from == "first-pass", bands_from_step=a.bands_from_step)


def follow(monitor: Monitor, tail: EventTail, poll_s: float, final_timeout_s: float) -> int:
    """Read every save as its event lands until the run stops or stays dead for the final timeout (0), or a row halts (3)."""
    dead_since: float | None = None
    while True:
        monitor.on_events(tail.read_new())
        monitor.beat()
        if monitor.state.halted:
            return 3
        # A run looking dead may be a resume starting under a stale heartbeat: its new segment's pid revives it.
        dead_since = None if monitor.run_alive() else (dead_since or time.time())
        if monitor.final_step is not None or (dead_since is not None and time.time() - dead_since >= final_timeout_s):
            monitor.final_save(final_timeout_s if monitor.final_step is not None else 0.0)
            monitor.beat()
            return 3 if monitor.state.halted else 0
        time.sleep(poll_s)


def main(argv: list[str] | None = None) -> int:
    """The command line."""
    ap = argparse.ArgumentParser(prog="run_monitor")
    ap.add_argument("mode", choices=("follow", "once"))
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", type=Path, required=True, help="the run's records directory")
    ap.add_argument("--gen-ring", type=Path, required=True)
    ap.add_argument("--gen-sha256", required=True)
    ap.add_argument("--exams", type=Path, required=True)
    ap.add_argument("--exams-sha256", required=True)
    ap.add_argument("--bands", type=Path, required=True, help="the pre-registration carrying [ring_audit.bands]")
    ap.add_argument("--bands-sha256", required=True)
    ap.add_argument("--floors", required=True, help="EXAM=FLOOR,... on the calibrated exam means")
    ap.add_argument("--line", type=float, required=True, help="the value line of the lagged read, nats")
    ap.add_argument("--parent", type=Path, required=True, help="the first save's lagged net")
    ap.add_argument("--batches", type=int, required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--gpu-mem-fraction", type=float, required=True, help="the monitor's cap on the card (cuda)")
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--halt", action="store_true", help="send the run ONE SIGTERM when a halting row fires")
    ap.add_argument("--gap-line", type=float, required=True, help="the memorisation gap's line, nats")
    ap.add_argument("--floors-from", choices=("start", "first-pass"), required=True,
                    help="the exam floors halt from the first save, or from the first save that passes them all")
    ap.add_argument("--bands-from-step", type=int, required=True, help="ring band misses before this step only report")
    ap.add_argument("--poll-sec", type=float, default=30.0)
    ap.add_argument("--final-timeout-sec", type=float, default=1800.0)
    ap.add_argument("--ckpt", type=Path, default=None, help="once: the checkpoint to read")
    ap.add_argument("--step", type=int, default=None, help="once: its step")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    try:
        setup = setup_of(a)
        inputs_pinned(setup)
    except (ValueError, OSError) as exc:
        print(f"run_monitor: {exc}", file=sys.stderr)
        return 2
    if setup.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(a.gpu_mem_fraction)
    monitor = Monitor(setup, Readers(value=value_read(setup), exams=exams_read(setup), bands=bands_read))
    (setup.out / "setup.json").write_text(json.dumps({k: str(v) for k, v in vars(a).items()}, indent=1), encoding="utf-8")
    if a.mode == "once":
        if a.ckpt is None or a.step is None:
            print("run_monitor: once needs --ckpt and --step", file=sys.stderr)
            return 2
        record = monitor.read_save(a.step, a.ckpt, time.time(), stopping=True)
        print(json.dumps({"step": a.step, "halting_rows": record["halting_rows"], "reported_rows": record["reported_rows"],
                          "exams": record["exams"]}, indent=1))
        return 3 if record["halting_rows"] else 0
    return follow(monitor, EventTail(setup.run_dir, setup.run_id, EVENTS), a.poll_sec, a.final_timeout_sec)


__all__ = ["follow", "main", "setup_of"]
