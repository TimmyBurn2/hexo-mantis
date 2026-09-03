"""The BC acceptance witness — R328(f)'s two halves, measured through the production seam.

Over the 300-line soft cap (R8) and kept as ONE unit: the file is a single instrument end to end — the
arm vocabulary, the seeded build, the engine replay, the readout arithmetic and the CLI that
emits it — and its two repaired halves are only checkable against each other. Splitting the
measurement away from the arm construction would put the seeded control in one file and the
number it is the baseline for in another, which is the arrangement that let the first witness
ship a comparison against a fresh draw.

R328(f) pre-registers ONE acceptance test for a behaviour-cloned checkpoint: post-BC play
against the random bot must clear the armed `strength_floor` decisive rate, AND the longest-run
distribution must show contested play. The first half is a bar the repo already owns
(`mantis.eval.floor_gate`); the second half is a description of the board, and describing the
board is where the first witness went wrong twice.

**IT LIVES HERE RATHER THAN AS A BOX SCRIPT BECAUSE ITS FIRST VERSION DID NOT.** Sitting 6's
witness ran as an ad-hoc file on the box. It returned a passing verdict and two defects rode
inside it, neither visible in its output:

1. **The control was a DRAW, not a baseline.** The "fresh net" arm called `build_net` with
   whatever global RNG state the process happened to hold, so the number the BC arm was
   compared against came from an uncontrolled sample of initialisations. This is F-RESIT-10's
   defect one path over — the same defect `mantis.diagnostics.worker_sweep` was repaired for,
   with the same repair: seed through `mantis.util.determinism` immediately before the ONE RNG
   consumer, from the config's own `seed`, never a literal.
2. **Stone colour was re-derived from PLY PARITY.** The script assigned
   `1 if (i // 2) % 2 == 0 else -1` over the move list. The engine hands the first stone to
   player 1 and then alternates in PAIRS (LAW-03; `tests/arena/test_ply_cap_adjudication.py`
   states the same order), so that expression is off by one stone from ply 1 onward and splits
   most lines between the two sides. It was not a subtle wrongness: the BC arm reported twenty
   wins — every one of which requires a six-in-a-row — with a longest run of 4.

Both repairs are the same repair in the end: **do not re-derive what a live authority already
holds.** Colour comes from replaying the record through the engine and asking
`mantis.arena.adjudicate.longest_run`; decisive/wins/draws come from
`mantis.eval.floor_gate.probe_measurements`; the player comes from
`mantis.eval.worker.build_candidate_player`, so the witness searches at the same width and
through the same decode entrance an eval round does.

**WHAT THIS MODULE DOES NOT OWN.** No threshold: the floor's three terms arrive resolved from
`eval.strength_floor`, and a config that leaves it `null` gets `floor_verdict: null` and a
readout with no verdict rather than a witness that quietly passes itself (LAW-07/LAW-08). No
ply cap of its own: it plays at the arena's `DEFAULT_MAX_PLIES`, the cap the eval worker plays
at, passed explicitly at the call site so a reader sees which cap the numbers are under.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from mantis._engine import Board
from mantis.arena.adjudicate import longest_run
from mantis.arena.books import paired_openings
from mantis.arena.match import DEFAULT_MAX_PLIES, play_paired_match
from mantis.arena.regime import RegimeKey
from mantis.bots.resolve import resolve_bot
from mantis.config.loader import load_config
from mantis.config.resolve.eval_posture import resolve_strength_floor
from mantis.config.resolve.fused_graph_caps import resolve_fused_graph_caps
from mantis.config.resolve.inference_batching import resolve_inference_batching
from mantis.encoding import lookup
from mantis.eval.floor_gate import evaluate_strength_floor, probe_measurements
from mantis.eval.worker import build_candidate_player
from mantis.model import arch_from_spec_and_config, build_net
from mantis.model.identity import net_param_hash
from mantis.selfplay.hparams import is_graph_representation
from mantis.selfplay.inference_local import LocalInferenceEngine
from mantis.train.checkpoints import load_checkpoint
from mantis.util.determinism import seed_everything

#: The label reserved for the seeded fresh-initialisation arm. `--arm control=CONTROL` is how
#: an operator asks for it; every other arm value is a checkpoint path.
CONTROL_CHECKPOINT = "CONTROL"

#: The witness's own regime variant. It plays the floor probe's opponent at the floor probe's
#: width, so its records are the one set that could plausibly be pooled with a probe's — and
#: `aggregate_rung`'s MixedRegimeError exists precisely so two differently-purposed sets never
#: are. A distinct variant is what keeps that impossible rather than merely unlikely.
WITNESS_VARIANT = "acceptance_witness"


class WitnessArmError(ValueError):
    """An arm specification this module cannot build. Raised, never defaulted past."""


@dataclass(frozen=True)
class ArmSpec:
    """One side of the witness: a label and the weights behind it.

    `checkpoint is None` is the SEEDED CONTROL — a fresh net at the config's seed. It is a
    fixed baseline, reproducible from the config alone, which is the whole point of the arm:
    the BC number means nothing beside a comparison that resamples itself every run.
    """

    label: str
    checkpoint: Path | None

    @classmethod
    def parse(cls, raw: str) -> ArmSpec:
        """Parse a `label=checkpoint` / `label=CONTROL` CLI argument.

        Raises:
            WitnessArmError: no `=`, or an empty label.
        """
        label, sep, value = raw.partition("=")
        if not sep or not label:
            raise WitnessArmError(
                f"arm {raw!r} is not 'label=<checkpoint>' or 'label={CONTROL_CHECKPOINT}'"
            )
        return cls(label=label, checkpoint=None if value == CONTROL_CHECKPOINT else Path(value))


def seeded_net(arch: Any, *, seed: int) -> torch.nn.Module:
    """Build `arch` from a KNOWN RNG state — the control arm's fixed baseline.

    `seed_everything` is documented idempotent and is called here immediately before
    `build_net`, the ONE RNG consumer on this path, exactly as
    `mantis.diagnostics.worker_sweep._seeded_net` does and for the same measured reason: an
    unseeded comparison arm is a fresh draw per invocation, so the quantity the other arm is
    judged against moves on its own. R30a's one-boot-site rule is not crossed — this process
    is not a run, it is a COMPARISON between nets, so each one must start from the same state
    or the comparison is not one.

    Args:
        arch: the resolved `ModelArch` to build.
        seed: the run config's own `seed`, threaded — never a literal here.

    Returns:
        The built net, on CPU, before any `.to(device)`.
    """
    seed_everything(int(seed))
    return build_net(arch)


def replay_board(moves: Sequence[tuple[int, int]], *, encoding_name: str) -> Any:
    """Rebuild a finished game's board by replaying its moves through the ENGINE.

    `GameRecord` carries the move list and not the board, and the move list alone does not say
    which stone is whose: the engine gives player 1 the first stone and then alternates in
    PAIRS, so ply index and colour are different alternations (LAW-03). Replaying is how the
    colour question gets answered by the authority that owns it instead of by arithmetic on
    the index — the arithmetic is what the first witness got wrong.

    Args:
        moves: the record's `moves`, in play order.
        encoding_name: the registered encoding the game was played under; it fixes the board's
            geometry, so a board built under a different one is a different board.

    Returns:
        The engine board after the last recorded move.
    """
    board = Board.with_encoding_name(encoding_name)
    for q, r in moves:
        board.apply_move(int(q), int(r))
    return board


def record_runs(record: Any, *, encoding_name: str) -> tuple[int, int]:
    """`(candidate, opponent)` longest run for one game, both colours from the engine.

    Args:
        record: an arena `GameRecord`.
        encoding_name: the encoding the game was played under.

    Returns:
        The two longest lines, in cells.
    """
    board = replay_board(record.moves, encoding_name=encoding_name)
    candidate_color = int(record.colors["candidate"])
    ceiling = len(record.moves)
    return (
        longest_run(board, candidate_color, ceiling=ceiling),
        longest_run(board, -candidate_color, ceiling=ceiling),
    )


def _positive_games(raw: str) -> int:
    """`--games` must be at least 1 (AUDIT-1 F-28/A08).

    Raises:
        argparse.ArgumentTypeError: the value is not an integer, or is below 1.
    """
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--games must be an integer, got {raw!r}") from exc
    if value < 1:
        raise argparse.ArgumentTypeError(
            f"--games must be at least 1, got {value}: a witness over zero games measures "
            "nothing and would print a verdict about it"
        )
    return value


def measure_arm(records: Sequence[Any], *, encoding_name: str, floor: Any) -> dict[str, Any]:
    """The witness readout for one arm: the floor half and the contested-play half.

    Decisive/wins/draws are `probe_measurements`' — the same arithmetic the armed floor gates
    on, not a second definition beside it. The run distribution is reported by SEAT and by
    OUTCOME: `loser_runs` is the half that answers "contested", because a winner's line is 6
    by definition in a won game and says nothing about whether anyone contested it.

    Args:
        records: the arm's `GameRecord`s.
        encoding_name: the encoding the games were played under.
        floor: the resolved `StrengthFloorSpec`, or `None` when `eval.strength_floor` is off.

    Returns:
        A JSON-ready mapping. `floor_verdict` is `None` on the unarmed posture — an unarmed
        witness reports numbers and NO verdict rather than a pass it did not measure.
    """
    games, decisive, wins, draws = probe_measurements(records)
    runs = [record_runs(rec, encoding_name=encoding_name) for rec in records]
    winner_runs, loser_runs = [], []
    for rec, (cand_run, opp_run) in zip(records, runs, strict=True):
        if rec.winner == "candidate":
            winner_runs.append(cand_run)
            loser_runs.append(opp_run)
        elif rec.winner == "opponent":
            winner_runs.append(opp_run)
            loser_runs.append(cand_run)
    verdict = evaluate_strength_floor(records, floor) if floor is not None else None
    return {
        "games": games,
        "decisive_games": decisive,
        # AUDIT-1 F-28/A08. A rate over ZERO games is not a rate of zero — and this tool's
        # whole subject is a decisive-rate bar, so `decisive_rate 0.0` over no games reads as
        # the strongest possible refusal evidence when nothing was played at all.
        "decisive_rate": (decisive / games) if games else None,
        "wins": wins,
        "draws": draws,
        "winrate": (wins / games) if games else None,
        "terminals": dict(Counter(rec.terminal for rec in records)),
        "mean_plies": (sum(rec.plies for rec in records) / games) if games else None,
        "runs_candidate_opponent": [list(pair) for pair in runs],
        "longest_run_max": max((max(pair) for pair in runs), default=None),
        "winner_runs": winner_runs,
        "loser_runs": loser_runs,
        "loser_run_histogram": dict(sorted(Counter(loser_runs).items())),
        "trajectory_hashes": [rec.trajectory_hash for rec in records],
        "floor_verdict": None if verdict is None else verdict.as_payload(),
    }


def witness_regime(*, encoding_name: str, model_sims: int, opening_book: str) -> RegimeKey:
    """The witness's regime tag — the floor probe's shape under `WITNESS_VARIANT`."""
    return RegimeKey(
        bot="random", variant=WITNESS_VARIANT, model_sims=model_sims,
        opponent_spec="random:uniform", opening_book=opening_book,
        deploy_matched=True, encoding=encoding_name,
    )


def play_arm(
    candidate: Any, opponent: Any, openings: Sequence[Any], *,
    regime_key: RegimeKey, encoding_name: str, max_plies: int, games: int,
) -> list[Any]:
    """Play one arm's games and return at most `games` records.

    No adjudicator is passed: the witness reads decisiveness off the arena's own `terminal`
    field, and an armed ply-cap adjudicator would convert capped non-results into wins, which
    is precisely the signal the witness exists to see.
    """
    records = play_paired_match(
        candidate, opponent, openings, regime_key=regime_key,
        board_factory=lambda: Board.with_encoding_name(encoding_name),
        record_sink=None, max_plies=max_plies, adjudicator=None,
    )
    return list(records[:games])


def _arm_engine(arm: ArmSpec, *, cfg: Any, dump: dict[str, Any], spec: Any,
                device: torch.device) -> LocalInferenceEngine:
    """Build one arm's inference engine: seeded net, then the arm's weights if it has any.

    Raises:
        CheckpointStampError: the checkpoint is not a v2 envelope, or its stamp disagrees with
            the declared encoding.
    """
    net = seeded_net(arch_from_spec_and_config(spec, dump), seed=cfg.seed)
    if arm.checkpoint is not None:
        net.load_state_dict(
            load_checkpoint(arm.checkpoint, declared_encoding=spec.name).model_state
        )
    # GRID passes `None` for both EXPLICITLY — "this route has no fused graph forward to bound"
    # and "this route builds no graph collector" — which is the engine's own stated contract
    # and not a fallback: resolving either on a grid config raises by arch scope (R322(d)).
    graph = is_graph_representation(spec)
    return LocalInferenceEngine(
        net.to(device).eval(), device, encoding_spec=spec,
        fused_graph_caps=resolve_fused_graph_caps(dump) if graph else None,
        inference_batching=resolve_inference_batching(dump) if graph else None,
        max_in_flight=cfg.selfplay.leaf_batch_size,
        # AUDIT-1 F-31: the declared autocast dtype, from the config this witness already
        # holds. `amp_dtype_for` resolves it (LAW-06); this site names no dtype.
        amp_dtype=cfg.train.amp_dtype,
    )


def run_witness(config_path: Path, arms: Sequence[ArmSpec], *, games: int,
                device: torch.device) -> dict[str, Any]:
    """Play every arm against the random bot and return the full readout.

    The encoding is the CONFIG's (`identity.encoding`) with no override: LAW-11 makes the
    identity a declared key, and a witness that could re-point it at the command line would be
    able to report a strength number for a geometry the checkpoint was never trained at.
    """
    cfg = load_config(config_path)
    dump = cfg.model_dump()
    spec = lookup(cfg.identity.encoding)
    floor = resolve_strength_floor(cfg.eval)
    sims = cfg.eval.random_model_sims
    leaf_batch_size = cfg.selfplay.leaf_batch_size
    out: dict[str, Any] = {
        "config": str(config_path),
        "encoding": spec.name,
        "seed": cfg.seed,
        "random_model_sims": sims,
        "leaf_batch_size": leaf_batch_size,
        "max_plies": DEFAULT_MAX_PLIES,
        "games_per_arm": games,
        "armed_floor": None if floor is None else {
            "probe_games": floor.probe_games,
            "min_decisive_rate": floor.min_decisive_rate,
            "min_winrate": floor.min_winrate,
        },
        "arms": {},
    }
    for arm in arms:
        engine = _arm_engine(arm, cfg=cfg, dump=dump, spec=spec, device=device)
        try:
            records = play_arm(
                build_candidate_player(engine, sims, spec=spec,
                                       leaf_batch_size=leaf_batch_size),
                resolve_bot("random", depth=None, opponent_sims=sims)(
                    seed=cfg.eval.gate.seed_base),
                paired_openings(cfg.eval.gate.opening_book, n_pairs=max(games // 2, 1),
                                seed=cfg.eval.gate.seed_base),
                regime_key=witness_regime(encoding_name=spec.name, model_sims=sims,
                                          opening_book=cfg.eval.gate.opening_book),
                encoding_name=spec.name, max_plies=DEFAULT_MAX_PLIES, games=games,
            )
            measured = measure_arm(records, encoding_name=spec.name, floor=floor)
            measured["net_param_hash"] = net_param_hash(engine.model)
            measured["checkpoint"] = None if arm.checkpoint is None else str(arm.checkpoint)
            out["arms"][arm.label] = measured
        finally:
            engine.close()
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entrance: `python -m mantis.diagnostics.acceptance_witness`."""
    parser = argparse.ArgumentParser(description="R328(f) BC acceptance witness")
    parser.add_argument("--config", required=True, type=Path)
    # `--games 0` measured nothing and then printed a witness about it (F-28/A08).
    parser.add_argument("--games", required=True, type=_positive_games)
    parser.add_argument("--device", required=True)
    parser.add_argument("--arm", action="append", required=True,
                        metavar=f"LABEL=CKPT|LABEL={CONTROL_CHECKPOINT}")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    readout = run_witness(args.config, [ArmSpec.parse(a) for a in args.arm],
                          games=args.games, device=torch.device(args.device))
    text = json.dumps(readout, indent=1, sort_keys=True)
    if args.out is not None:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
