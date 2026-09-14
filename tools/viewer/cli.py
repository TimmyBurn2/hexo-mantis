"""The command line: `--run RUN_ID=GAMES_DIR` (repeatable) `--out DIR [--title T]`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .html import data_path, render, shard_js
from .reader import EmptyGameRecord, RunData, build_run


def _parse_run(spec: str) -> tuple[str, Path]:
    run_id, sep, games_dir = spec.partition("=")
    if not sep or not run_id or not games_dir:
        raise argparse.ArgumentTypeError(f"--run wants RUN_ID=GAMES_DIR, got {spec!r}")
    return run_id, Path(games_dir)


def main(argv: list[str] | None = None) -> int:
    """Write `<out>/index.html` and `<out>/data/<run>/<shard>.js` for every run; 2 on a refusal."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="append", required=True, type=_parse_run, metavar="RUN_ID=GAMES_DIR",
                    help="a run's id and its GAME-RECORD-1 directory (<run>/logs/games); repeatable")
    ap.add_argument("--out", type=Path, required=True, help="the viewer directory to write")
    ap.add_argument("--title", default="mantis game viewer", help="page title")
    args = ap.parse_args(argv)
    runs: list[RunData] = []
    for run_id, games_dir in args.run:
        try:
            runs.append(build_run(games_dir, run_id))
        except EmptyGameRecord as exc:
            print(f"game_viewer: refused: {exc}", file=sys.stderr)
            return 2
    args.out.mkdir(parents=True, exist_ok=True)
    total = 0
    for run in runs:
        for shard in run.shards:
            path = args.out / data_path(run.run_id, shard.name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(shard_js(run.run_id, shard.name, shard.games), encoding="utf-8")
            total += path.stat().st_size
    page = args.out / "index.html"
    page.write_text(render(runs, args.title), encoding="utf-8")
    games = sum(len(r.index) for r in runs)
    print(f"wrote {page} ({page.stat().st_size:,} bytes) and {sum(len(r.shards) for r in runs)} "
          f"shard file(s) ({total:,} bytes) for {games:,} games")
    for run in runs:
        for finding in run.findings:
            print(f"FINDING {finding}", file=sys.stderr)
        if run.skipped_lines:
            print(f"{run.run_id}: {run.skipped_lines} unparseable line(s) skipped", file=sys.stderr)
    return 0
