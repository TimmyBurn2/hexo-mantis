"""`python tools/position_analyzer.py [serve|once] …` — serve is the default; once prints one record as JSON."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .dispatch import Dispatcher
from .engines import discover

DEFAULT_BIND, DEFAULT_PORT, DEFAULT_TIMEOUT = "127.0.0.1", 8766, 120.0


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--checkpoints", action="append", required=True, type=Path, metavar="DIR",
                   help="a directory of stamped checkpoints, or a mirror root (its */checkpoints are read); repeatable")
    p.add_argument("--strix", action="store_true", help="also offer the vendored strix pin (needs `make vendor.strix`)")
    p.add_argument("--device", default="cpu", help="torch device for the nets (default cpu; the box passes cuda)")
    p.add_argument("--threads", type=int, default=None, help="torch threads (absent = torch's own default)")


def build_parser() -> argparse.ArgumentParser:
    """The CLI: `serve` binds loopback by default; `--bind 0.0.0.0` is UNSAFE (no auth, no TLS) and never the default."""
    ap = argparse.ArgumentParser(prog="python tools/position_analyzer.py", description=__doc__)
    sub = ap.add_subparsers(dest="cmd")
    serve = sub.add_parser("serve", help="the loopback server + page (the default)")
    _common(serve)
    serve.add_argument("--bind", default=DEFAULT_BIND, help="0.0.0.0 exposes the engines to the LAN: unsafe, documented")
    serve.add_argument("--port", type=int, default=DEFAULT_PORT)
    serve.add_argument("--timeout-sec", type=float, default=DEFAULT_TIMEOUT, help="the analyst's per-request wait")
    once = sub.add_parser("once", help="one position → the record on stdout")
    _common(once)
    once.add_argument("--engine", required=True, help="an id from /engines (the checkpoint's stem)")
    once.add_argument("--moves", default="", help="`q,r;q,r;…`")
    once.add_argument("--sims", type=int, default=0, help="0 = the net only")
    once.add_argument("--symmetry", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    """Run the CLI; 0 on success, 2 on a refusal (printed to stderr)."""
    args_in = list(sys.argv[1:] if argv is None else argv)
    if not args_in or args_in[0].startswith("-"):
        args_in.insert(0, "serve")
    args = build_parser().parse_args(args_in)
    infos = discover(args.checkpoints)
    if args.cmd == "once":
        disp = Dispatcher(infos, device=args.device, threads=args.threads, strix=args.strix)
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
    print("serve: the server lands with the page (ANALYZER-1 phase 2); `once` is available", file=sys.stderr)
    return 2


__all__ = ["DEFAULT_BIND", "DEFAULT_PORT", "DEFAULT_TIMEOUT", "build_parser", "main"]
