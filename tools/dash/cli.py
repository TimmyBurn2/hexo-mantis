"""`python tools/dash.py [serve|freeze] …` — serve is the default; freeze writes the Run view as one self-contained file."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import analyzer_routes
from . import desk as desk_mod
from .readers.events import EmptyRunRecord
from .readers.record import RunRecord
from .routes import GET
from .serve import run_server
from .views import run as run_view

DEFAULT_BIND, DEFAULT_PORT, DEFAULT_POLL_SEC = "127.0.0.1", 8765, 30.0


def _pairs(values: list[str] | None, flag: str) -> dict[str, Path]:
    """`ID=DIR` pairs. Raises: SystemExit, on a value without `=`."""
    out: dict[str, Path] = {}
    for raw in values or []:
        label, sep, path = raw.partition("=")
        if not sep or not label or not path:
            raise SystemExit(f"{flag} wants ID=DIR, got {raw!r}")
        out[label] = Path(path)
    return out


def _inputs(p: argparse.ArgumentParser) -> None:
    p.add_argument("--run", action="append", metavar="ID=DIR", help="a run directory (logs/, checkpoints/, "
                   "resolved_config.yaml) or its mirror, under a label; repeatable, the first is the default")
    p.add_argument("--records", action="append", metavar="ID=DIR", help="the run monitor's records for that label")
    p.add_argument("--cells", action="append", type=Path, default=[], metavar="DIR",
                   help="a directory searched for cell sidecars (<ckpt>.six30_16*.json, <ckpt>.strix*.json, "
                        "<ckpt>.ladder455_n16*.json) at every poll; repeatable")
    p.add_argument("--ladder", action="append", metavar="ID=FILE",
                   help="that run's units file from the records, re-read at every poll: the rule (UNIT[,UNIT@STEP…]), the "
                        "second ruler, the legacy units and the rung state; with none, no rule is declared")


def build_parser() -> argparse.ArgumentParser:
    """The CLI: `serve` binds loopback by default (`--bind 0.0.0.0` is unsafe: no auth, no TLS); `freeze` writes a file."""
    ap = argparse.ArgumentParser(prog="python tools/dash.py", description=__doc__)
    sub = ap.add_subparsers(dest="cmd")
    serve = sub.add_parser("serve", help="the read-only server (the default)")
    _inputs(serve)
    serve.add_argument("--checkpoints", action="append", type=Path, default=[], metavar="DIR",
                       help="stamped checkpoints for the Analyzer; omitted, the Analyzer says how to start it")
    serve.add_argument("--strix", action="store_true", help="also offer the vendored strix pin to the Analyzer")
    serve.add_argument("--device", default="cpu", help="torch device for the Analyzer's nets (default cpu)")
    serve.add_argument("--threads", type=int, default=None, help="torch threads for the Analyzer's nets")
    serve.add_argument("--bind", default=DEFAULT_BIND)
    serve.add_argument("--port", type=int, default=DEFAULT_PORT)
    serve.add_argument("--poll-sec", type=float, default=DEFAULT_POLL_SEC, help="how often the record is re-read")
    once = sub.add_parser("once", help="one position read by one engine, the record printed as JSON")
    once.add_argument("--checkpoints", action="append", type=Path, required=True, metavar="DIR")
    once.add_argument("--strix", action="store_true")
    once.add_argument("--device", default="cpu")
    once.add_argument("--threads", type=int, default=None)
    once.add_argument("--engine", required=True, help="an engine id (the checkpoint's stem)")
    once.add_argument("--moves", default="", help="`q,r;q,r;…`")
    once.add_argument("--sims", type=int, default=0, help="0 = the net only")
    once.add_argument("--symmetry", action="store_true")
    freeze = sub.add_parser("freeze", help="the Run view as one self-contained HTML file")
    _inputs(freeze)
    freeze.add_argument("--compare", default=None, metavar="ID", help="a second run label to overlay")
    freeze.add_argument("--out", type=Path, required=True)
    return ap


def records_of(args: argparse.Namespace) -> list[RunRecord]:
    """One `RunRecord` per `--run`, in order. Raises: SystemExit (no run, a malformed pair), EmptyRunRecord, OSError."""
    runs = _pairs(args.run, "--run")
    if not runs:
        raise SystemExit("give at least one --run ID=DIR")
    monitor, ladders = _pairs(args.records, "--records"), _pairs(args.ladder, "--ladder")
    for flag, given in (("--records", monitor), ("--ladder", ladders)):
        unknown = sorted(set(given) - set(runs))
        if unknown:
            raise SystemExit(f"{flag} names no served run: {', '.join(unknown)}")
    return [RunRecord(label, path, monitor.get(label), tuple(args.cells), ladder_file=ladders.get(label))
            for label, path in runs.items()]


def freeze(args: argparse.Namespace) -> int:
    """Write the Run view of the first run (the `--compare` one overlaid) to `--out`; 2 on a refused record. Raises: OSError."""
    try:
        records = records_of(args)
        snaps = {r.label: r.poll() for r in records}
    except EmptyRunRecord as exc:
        print(f"dash: refused: {exc}", file=sys.stderr)
        return 2
    first = records[0].label
    chosen = [snaps[first]] + ([snaps[args.compare]] if args.compare in snaps and args.compare != first else [])
    html = run_view.page(chosen, runs=tuple(snaps), now=None)
    args.out.write_text(html, encoding="utf-8")
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")
    return 0


def once(args: argparse.Namespace) -> int:
    """Read one position on one engine and print the record; 2 on a refusal (printed to stderr). Raises: OSError."""
    from .engine.dispatch import Dispatcher  # the engine layer (torch) is the optional dependency
    from .engine.engines import discover
    disp = Dispatcher(discover(args.checkpoints), device=args.device, threads=args.threads, strix=args.strix)
    try:
        out = disp.handle({"op": "analyze", "engine": args.engine, "moves": args.moves, "sims": args.sims,
                           "symmetry": args.symmetry, "seq": 0})
    finally:
        disp.close()
    if out["status"] != 200:
        print(f"refused ({out['status']}): {out['body']['refused']}", file=sys.stderr)
        return 2
    print(json.dumps(out["body"]["record"], indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the CLI; `serve` is the default subcommand."""
    args_in = list(sys.argv[1:] if argv is None else argv)
    if not args_in or (args_in[0].startswith("-") and args_in[0] not in ("-h", "--help")):
        args_in.insert(0, "serve")
    args = build_parser().parse_args(args_in)
    if args.cmd == "freeze":
        return freeze(args)
    if args.cmd == "once":
        return once(args)
    try:
        records = records_of(args)
    except EmptyRunRecord as exc:
        print(f"dash: refused: {exc}", file=sys.stderr)
        return 2
    desk = desk_mod.LazyDesk(args.checkpoints, strix=args.strix, device=args.device, threads=args.threads) \
        if args.checkpoints or args.strix else None
    extra = {**GET, "analyzer": analyzer_routes.page(desk), "api/engines": analyzer_routes.engines(desk)}
    return run_server(args, records, get_extra=extra, post_of=lambda hub: analyzer_routes.post(hub, desk),
                      on_stop=desk.close if desk is not None else None)


__all__ = ["DEFAULT_BIND", "DEFAULT_PORT", "build_parser", "freeze", "main", "records_of"]
