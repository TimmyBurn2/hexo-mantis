"""LADDER-1 (CARD-LADDER-RUNG): one process per registered bot on a HeXO server's bot API — hold the stream, answer every move request through ONE backend (`mantis` or `strix`) from the book prefix R363(c) fixed (`book_v1_s20260625_p4` paired, opening index = match index), write a receipt per game; EVAL only, nothing here writes a ring; the token comes from `HEXO_TOKEN` alone; `--replay RECEIPT` is the determinism + budget witness."""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]


def _load_ladder() -> Any:
    """`tools/ladder` by path under its own name (`sys.path` untouched, R5); once per process."""
    if "ladder" in sys.modules:
        return sys.modules["ladder"]
    pkg = _REPO / "tools" / "ladder"
    spec = importlib.util.spec_from_file_location("ladder", pkg / "__init__.py", submodule_search_locations=[str(pkg)])
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {pkg}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ladder"] = module
    spec.loader.exec_module(module)
    return module


def parse_time_control(text: str) -> dict[str, Any]:
    """`unlimited` | `turn:<ms>` | `match:<main_ms>:<increment_ms>` -> the server's TimeControl union. Raises: ValueError on any other spelling."""
    parts = text.split(":")
    try:
        if parts == ["unlimited"]:
            return {"mode": "unlimited"}
        if len(parts) == 2 and parts[0] == "turn":
            return {"mode": "turn", "turnTimeMs": int(parts[1])}
        if len(parts) == 3 and parts[0] == "match":
            return {"mode": "match", "mainTimeMs": int(parts[1]), "incrementMs": int(parts[2])}
    except ValueError:
        pass
    raise ValueError(f"time control {text!r}: use unlimited, turn:<ms> or match:<main_ms>:<increment_ms>")


@dataclass
class ReplayReport:
    """What `--replay` found: the recorded moves and budgets reproduced or not (`mismatches`, `budget_misses`), the book stones re-derived from the receipt's opening or not (`book_misses`), and the turns recorded BELOW the configured sims for their searched stones (`below_budget`: a decided position — strix's root VCF solver answers in a few visits, mantis's PUCT tree stops once every path hits a terminal; measured live 2026-09-19 at 3 and 427 of 512 — reported, never a failure on its own; the replay must reproduce the count)."""

    game_id: str
    backend: str
    moves: int = 0
    mismatches: list[dict[str, Any]] = field(default_factory=list)
    budget_misses: list[dict[str, Any]] = field(default_factory=list)
    book_misses: list[dict[str, Any]] = field(default_factory=list)
    below_budget: list[dict[str, Any]] = field(default_factory=list)
    ms_per_turn: list[float] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.mismatches and not self.budget_misses and not self.book_misses


def replay_receipt(receipt: dict[str, Any], backend: Any) -> ReplayReport:
    """Rebuild every position the receipt answered from the server's own move list and answer it again. Raises: ValueError when the receipt carries no `moves_full` (the server kept no record)."""
    ladder = _load_ladder()
    full = receipt.get("moves_full")
    if not full:
        raise ValueError(f"{receipt['game_id']}: no moves_full on the receipt; the positions cannot be rebuilt")
    report = ReplayReport(game_id=str(receipt["game_id"]), backend=str(receipt["bot"]["backend"]))
    opening = ladder.openings.LadderOpening.from_record(receipt["opening"])
    backend.new_game(str(receipt["game_id"]))
    for move in receipt["moves"]:
        stones = int(move["stones"])
        cells = [{"q": q, "r": r, "p": side} for q, r, side in full[:stones]]
        to_move = receipt["side"]
        board = ladder.wire.board_from_wire({"to_move": to_move, "cells": cells}, encoding=backend.encoding)
        # The book stones come from the receipt's OPENING and the position, never from the recorded placements.
        forced = ladder.openings.forced_stones(opening, [(int(q), int(r)) for q, r, _side in full[:stones]]) or ()
        recorded_book = [list(c) for c in move["placements"][: int(move["book_stones"])]]
        if len(forced) != int(move["book_stones"]) or [list(c) for c in forced] != recorded_book:
            report.book_misses.append({"request_id": move["request_id"], "recorded": recorded_book,
                                       "derived": [list(c) for c in forced]})
        turn = backend.select_turn(board, forced)
        report.moves += 1
        report.ms_per_turn.append(float(turn.ms))
        replayed = [[q, r] for q, r in turn.placements]
        if replayed != move["placements"]:
            report.mismatches.append({"request_id": move["request_id"], "recorded": move["placements"],
                                      "replayed": replayed})
        if int(turn.sims) != int(move["sims"]):
            report.budget_misses.append({"request_id": move["request_id"], "sims": int(turn.sims),
                                         "recorded": int(move["sims"])})
        configured = (2 - len(forced)) * int(receipt["sims_configured"])
        if int(move["sims"]) < configured:
            report.below_budget.append({"request_id": move["request_id"], "sims": int(move["sims"]),
                                        "configured": configured})
    return report


def ladder_presets() -> dict[str, int | None]:
    """The presets the backends know (`unit`, `play`), read off the package so the CLI cannot drift from it."""
    return dict(_load_ladder().backends.PRESET_SIMS)


def _open_backend(args: argparse.Namespace, ladder: Any) -> Any:
    if args.backend == "mantis":
        from mantis.config.loader import load_config

        if args.config is None or args.checkpoint is None:
            raise SystemExit("--backend mantis needs --config (the run's eval seam) and --checkpoint")
        return ladder.backends.open_mantis(load_config(args.config), Path(args.checkpoint), threads=args.threads,
                                           preset=args.preset)
    return ladder.backends.open_strix(sims=args.sims, threads=args.threads, preset=args.preset)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--backend", choices=("mantis", "strix"), required=True)
    ap.add_argument("--server", required=True, help="the HeXO server's origin, e.g. https://host")
    ap.add_argument("--work-dir", type=Path, required=True, help="receipts land under <work-dir>/receipts/<net_hash8>/")
    ap.add_argument("--config", help="mantis: the run config (deploy kind, deploy_sims, σ, batching are its own)")
    ap.add_argument("--checkpoint", help="mantis: the stamped .ckpt to play")
    ap.add_argument("--sims", type=int, default=256, help="strix: sims per stone (the unit on record is 256)")
    ap.add_argument("--threads", type=int, default=None, help="torch threads for the backend (default: torch's)")
    ap.add_argument("--preset", choices=sorted(ladder_presets()), default="unit",
                    help="unit: the config's deploy_sims / --sims (the reading on record); play: 64 sims, R363 §0(5), "
                         "labelled on every receipt, never a unit reading")
    ap.add_argument("--match-offset", type=int, default=0,
                    help="the pair index the first pair against an opponent plays (a resumed series names where it left off)")
    ap.add_argument("--no-open", action="store_true", help="hold the stream without taking challenges")
    ap.add_argument("--accept-from", help="comma-separated profile ids whose challenges are accepted; default any")
    ap.add_argument("--challenge", metavar="PROFILE_ID", help="issue paired challenges to this bot")
    ap.add_argument("--games", type=int, default=None, help="--challenge: how many games (challenges) to issue")
    ap.add_argument("--time-control", default=None, help="--challenge: unlimited | turn:<ms> | match:<main>:<inc>")
    ap.add_argument("--challenge-timeout-sec", type=float, default=900.0)
    ap.add_argument("--no-reconnect", action="store_true", help="exit when the stream ends instead of reconnecting")
    ap.add_argument("--replay", type=Path, metavar="RECEIPT", help="the witness: replay this receipt and exit")
    args = ap.parse_args(argv)
    ladder = _load_ladder()

    if args.replay is not None:
        receipt = ladder.receipt.read_receipt(args.replay)
        backend = _open_backend(args, ladder)
        try:
            if backend.net_hash != receipt["bot"]["net_hash"]:
                print(f"replay: the receipt was played by {receipt['bot']['name']}, this backend is {backend.name}",
                      file=sys.stderr)
                return 2
            report = replay_receipt(receipt, backend)
        finally:
            backend.close()
        mean_ms = sum(report.ms_per_turn) / len(report.ms_per_turn) if report.ms_per_turn else 0.0
        print(f"replay {report.game_id} ({report.backend}): {report.moves} turn(s), {len(report.mismatches)} "
              f"mismatch(es), {len(report.budget_misses)} budget miss(es), {len(report.book_misses)} book miss(es), "
              f"{len(report.below_budget)} below budget, {mean_ms / 1000.0:.2f} s/turn -> "
              f"{'PASS' if report.passed else 'FAIL'}")
        for row in report.mismatches + report.budget_misses + report.book_misses + report.below_budget:
            print(f"  {row}")
        return 0 if report.passed else 1

    token = os.environ.get("HEXO_TOKEN")
    if not token:
        print("ladder_bot: set HEXO_TOKEN to the bot account's token (never on the command line)", file=sys.stderr)
        return 2
    plan = None
    if args.challenge is not None:
        if args.games is None or args.time_control is None:
            ap.error("--challenge needs --games and --time-control")
        plan = ladder.session.ChallengePlan(profile_id=args.challenge, games=int(args.games),
                                            time_control=parse_time_control(args.time_control))
    accept = None if args.accept_from is None else frozenset(p.strip() for p in args.accept_from.split(",") if p.strip())
    options = ladder.session.SessionOptions(
        server=args.server, work_dir=args.work_dir, open_for_challenges=not args.no_open, accept_from=accept,
        challenge=plan, reconnect=not args.no_reconnect, challenge_timeout_sec=float(args.challenge_timeout_sec),
        match_offset=int(args.match_offset))
    backend = _open_backend(args, ladder)
    client = ladder.client.LadderClient(args.server, token)
    try:
        summary = ladder.session.Session(client, backend, options=options,
                                         log=lambda line: print(line, file=sys.stderr, flush=True)).run()
    finally:
        backend.close()
    print(f"ladder_bot: {summary.games} game(s): {summary.wins} won, {summary.losses} lost, {summary.aborted} aborted; "
          f"{summary.rejections} rejection(s), {summary.challenges_issued} challenge(s) issued, "
          f"{len(summary.receipts)} receipt(s) under {args.work_dir / 'receipts'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
