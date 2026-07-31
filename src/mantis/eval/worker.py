"""mantis.eval.worker — CHILD-ONLY module (design §a.3 worker.py).

Entry: `python -m mantis.eval.worker <spec.json> <result.json>` / spawn target
`worker_main`. Loads snapshots, builds nets on `spec.worker_device`, plays: (1) the gate
block (candidate vs the best anchor, deploy-matched, screen -> confirm escalation), (2)
resolved ladder-rung blocks (`RungUnresolvable` per rung is RECORDED, never fatal), (3)
the random floor. Writes the sidecar result JSON ATOMICALLY (tmp + os.replace). Imports
`mantis.selfplay.inference_local` for leaf inference — the ONE parent-side-excepted
inference surface (isolation law 1: this module is the out-of-process leg).

>300 justify, stated at this file's MEASURED size of 328 lines (`wc -l`, re-measured after
WP12-R Phase B and its RED-TEAM close; 284 before it): one child-process entry point owning
the gate block, ladder-rung blocks, and the random floor — each phase shares the candidate
DeployHeadPlayer/inference engine/book-loading machinery; splitting them would duplicate
that setup three times and let the phases drift out of the SAME-process-covers-the-round
contract this module exists to keep (isolation law: exactly one worker process per round).
WP12-R added the ONE encoding resolution and the decode-capability guard that the same
one-process contract requires be made once, here, for every block below.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from mantis.arena.books import paired_openings
from mantis.arena.deploy_head import DeployHeadPlayer
from mantis.arena.match import play_paired_match
from mantis.arena.regime import RegimeKey
from mantis.bots.protocol import RungUnresolvable
from mantis.bots.resolve import resolve_bot
from mantis.encoding import EncodingSpec, lookup, normalize_encoding_name
from mantis.eval.aggregate import aggregate_gate, aggregate_rung
from mantis.eval.errors import EvalDecodeUnsupportedError
from mantis.eval.rounds import RoundSpec, RungJob
from mantis.eval.snapshot import load_model_snapshot
from mantis.selfplay.inference_local import LocalInferenceEngine

#: confirm-phase opening seed offset (deploy_strength_eval.py:519 parity) — the confirm
#: block draws a DIFFERENT slice of the book than the screen block.
_CONFIRM_SEED_OFFSET = 7919

#: Policy-pool values the eval decode ENTRANCE actually implements. Every block routes
#: through `_build_candidate_player` -> `DeployHeadPlayer(infer_fn=engine.infer)` ->
#: `LocalInferenceEngine.infer_batch`, whose dense arm scatter-MAXes over cluster windows
#: and DROPS off-window cells (`if mcts_idx >= n_actions - 1: continue`,
#: inference_local.py:207-208). "none" is the single-window / graph case, no pooling.
#: A CLOSED SET, not a blocklist: a registry row that later declares `scatter_mean` fails
#: here until this seam implements it, rather than being silently max-pooled.
_DECODE_IMPLEMENTED_POLICY_POOLS = frozenset({"none", "scatter_max"})


def _assert_decode_implements_declared_pooling(spec: EncodingSpec) -> None:
    """Refuse a round whose DECLARED policy pooling this worker's decode cannot honour."""
    if spec.policy_pool in _DECODE_IMPLEMENTED_POLICY_POOLS:
        return
    raise EvalDecodeUnsupportedError(
        f"encoding {spec.name!r} declares policy_pool={spec.policy_pool!r}, which this eval "
        f"worker's decode entrance does not implement: DeployHeadPlayer reaches the net "
        f"through LocalInferenceEngine.infer_batch, whose dense arm scatter-maxes and DROPS "
        f"off-window cells. The no-drop decode (infer_batch_per_cluster) exists but is not "
        f"wired to the deploy head (ADJ-WP12R-4). Refusing to report an eval result pooled "
        f"differently from the encoding's own declaration."
    )


def _agg_record(game_record: Any) -> dict[str, Any]:
    """Arena `GameRecord` -> the aggregate.py record convention `{p1,p2,winner,moves,
    regime_key}` (p1 == candidate always, by this worker's own construction)."""
    winner = {"candidate": "p1", "opponent": "p2", "draw": "draw"}[game_record.winner]
    return {
        "p1": "cand", "p2": "opponent", "winner": winner,
        "moves": [list(m) for m in game_record.moves],
        "regime_key": game_record.regime_key.canonical(),
        "trajectory_hash": game_record.trajectory_hash,
    }


def _model_sims_for_kind(spec: RoundSpec, kind: str) -> int:
    return {
        "sealbot": spec.sealbot_model_sims,
        "kraken": spec.kraken_model_sims,
        "strix": spec.strix_model_sims,
        "random": spec.random_model_sims,
    }[kind]


def _build_candidate_player(engine: LocalInferenceEngine, n_sims: int) -> DeployHeadPlayer:
    return DeployHeadPlayer(infer_fn=engine.infer, n_sims=n_sims)


def _play_gate_block(
    spec: RoundSpec,
    candidate_engine: LocalInferenceEngine,
    board_factory,
    *,
    encoding_spec: EncodingSpec,
) -> dict | None:
    """The gate block: candidate vs the best anchor, deploy-matched, screen -> confirm
    escalation (`should_escalate`, the SINGLE lower-bound test). Returns the raw
    `{"screen": [...], "confirm": [...]}` record lists, or None when there is no best
    anchor to play against yet (run3 `run(best_model=None)` parity)."""
    if spec.best_snapshot is None or not spec.gate.run_gate:
        return None

    best_model = load_model_snapshot(spec.best_snapshot, device=spec.worker_device)
    best_engine = LocalInferenceEngine(
        best_model, _device(spec.worker_device), encoding_spec=encoding_spec
    )
    try:
        candidate = _build_candidate_player(candidate_engine, spec.gate.deploy_sims)
        opponent = _build_candidate_player(best_engine, spec.gate.deploy_sims)

        regime_key = RegimeKey(
            bot="best_anchor", variant="deploy", model_sims=spec.gate.deploy_sims,
            opponent_spec="best_anchor:deploy_matched", opening_book=spec.gate.opening_book,
            deploy_matched=True, encoding=spec.encoding,
        )
        screen_openings = paired_openings(
            spec.gate.opening_book, n_pairs=max(spec.gate.screen_games // 2, 1),
            seed=spec.gate.seed_base,
        )
        screen_records = play_paired_match(
            candidate, opponent, screen_openings, regime_key=regime_key,
            board_factory=board_factory, record_sink=None,
        )
        screen_agg = [_agg_record(r) for r in screen_records]

        wr_screen = _draw_aware_wr(screen_agg)
        escalate = wr_screen is not None and wr_screen >= spec.gate.screen_confirm_lo
        confirm_agg: list[dict[str, Any]] = []
        if escalate:
            confirm_openings = paired_openings(
                spec.gate.opening_book, n_pairs=max(spec.gate.confirm_games // 2, 1),
                seed=spec.gate.seed_base + _CONFIRM_SEED_OFFSET,
            )
            confirm_records = play_paired_match(
                candidate, opponent, confirm_openings, regime_key=regime_key,
                board_factory=board_factory, record_sink=None,
            )
            confirm_agg = [_agg_record(r) for r in confirm_records]
        return {"screen": screen_agg, "confirm": confirm_agg}
    finally:
        best_engine.close()


def _draw_aware_wr(records: list[dict[str, Any]]) -> float | None:
    if not records:
        return None
    wins = sum(1 for r in records if r["winner"] == "p1")
    draws = sum(1 for r in records if r["winner"] == "draw")
    return (wins + 0.5 * draws) / len(records)


def _play_rung_block(
    spec: RoundSpec, rung_job: RungJob, candidate_engine: LocalInferenceEngine, board_factory
) -> list[dict[str, Any]]:
    bot_factory = resolve_bot(
        rung_job.bot, depth=rung_job.depth,
        opponent_sims=_model_sims_for_kind(spec, rung_job.bot),
    )
    opponent = bot_factory()
    # M-3: the candidate plays rung games at the resolved PER-KIND *_model_sims — the same
    # value the RegimeKey below stamps — never at gate.deploy_sims (that value is reserved
    # for the deploy-matched GATE block, LAW-15). Playing deploy_sims here while stamping a
    # different regime value was an A3 mislabel (the record must describe the regime it
    # actually played).
    candidate = _build_candidate_player(candidate_engine, _model_sims_for_kind(spec, rung_job.bot))
    regime_key = RegimeKey(
        bot=rung_job.bot, variant=rung_job.variant, model_sims=_model_sims_for_kind(spec, rung_job.bot),
        opponent_spec=f"{rung_job.bot}:{rung_job.variant}", opening_book=rung_job.opening_book,
        deploy_matched=rung_job.deploy_matched, encoding=spec.encoding,
    )
    openings = paired_openings(
        rung_job.opening_book, n_pairs=max(rung_job.games // 2, 1), seed=spec.seed_base,
    )
    records = play_paired_match(
        candidate, opponent, openings, regime_key=regime_key,
        board_factory=board_factory, record_sink=None,
    )
    return [_agg_record(r) for r in records[: rung_job.games]]


def _play_random_floor(spec: RoundSpec, candidate_engine: LocalInferenceEngine, board_factory) -> list[dict[str, Any]]:
    if spec.random_floor_games <= 0:
        return []
    bot_factory = resolve_bot("random", depth=None, opponent_sims=spec.random_model_sims)
    opponent = bot_factory(seed=spec.seed_base)
    # M-3: same fix as the rung block — the random floor plays at the resolved
    # random_model_sims, not gate.deploy_sims.
    candidate = _build_candidate_player(candidate_engine, spec.random_model_sims)
    regime_key = RegimeKey(
        bot="random", variant="raw", model_sims=spec.random_model_sims,
        opponent_spec="random:uniform", opening_book=spec.gate.opening_book,
        deploy_matched=True, encoding=spec.encoding,
    )
    openings = paired_openings(
        spec.gate.opening_book, n_pairs=max(spec.random_floor_games // 2, 1), seed=spec.seed_base,
    )
    records = play_paired_match(
        candidate, opponent, openings, regime_key=regime_key,
        board_factory=board_factory, record_sink=None,
    )
    return [_agg_record(r) for r in records[: spec.random_floor_games]]


def _device(name: str):
    import torch

    return torch.device(name)


def run_round(spec: RoundSpec) -> dict[str, Any]:
    """Play the full round described by `spec`; return the RAW (validate_worker_result-
    shaped) result dict. Never raises for a resolver failure (per-rung skipped, never
    fatal) — an unrecoverable failure (model load, engine build) is allowed to propagate
    so the parent's join/exit-code path classifies it (isolation law 2)."""
    from mantis._engine import Board

    # ONE resolution of the round's DECLARED encoding. Board geometry and the inference
    # decode are sized from the SAME resolved value, so they cannot diverge — before this,
    # `board_factory` read `spec.encoding` while the engine bound a `"v6"` constant.
    enc_name = normalize_encoding_name(spec.encoding)
    enc_spec: EncodingSpec = lookup(enc_name)
    _assert_decode_implements_declared_pooling(enc_spec)

    def board_factory():
        return Board.with_encoding_name(enc_name)

    candidate_model = load_model_snapshot(spec.candidate_snapshot, device=spec.worker_device)
    candidate_engine = LocalInferenceEngine(
        candidate_model, _device(spec.worker_device), encoding_spec=enc_spec
    )

    try:
        gate_records = _play_gate_block(
            spec, candidate_engine, board_factory, encoding_spec=enc_spec
        )
        gate_result: dict | None = None
        if gate_records is not None:
            gate_agg = aggregate_gate(gate_records["screen"], gate_records["confirm"], spec.gate)
            gate_result = {
                "wr_screen": gate_agg.wr_screen, "wr_confirm": gate_agg.wr_confirm,
                "n_screen": gate_agg.n_screen, "n_confirm": gate_agg.n_confirm,
                "n_pooled": gate_agg.n_pooled, "escalated": gate_agg.escalated,
                "elo_ci_lower_boot": gate_agg.elo_ci_lower_boot, "low_power": gate_agg.low_power,
                "eff_n": gate_agg.eff_n, "reason": "", "deploy_matched": True,
                "promoted": gate_agg.promoted,
            }

        rungs_result: dict[str, Any] = {}
        skipped_rungs: list[dict[str, str]] = []
        for rung_job in spec.rung_jobs:
            if rung_job.games <= 0:
                continue
            try:
                records = _play_rung_block(spec, rung_job, candidate_engine, board_factory)
            except RungUnresolvable as exc:
                skipped_rungs.append({"rung": rung_job.name, "reason": exc.reason})
                continue
            # M-2: thread eval.ladder.bootstrap_{resamples,ci_level,seed} through — never
            # fall back to aggregate.py's own signature defaults, which had no live
            # consumer and made a minted `eval.ladder.bootstrap_resamples` silently inert.
            agg = aggregate_rung(
                records,
                bootstrap_resamples=spec.ladder_bootstrap_resamples,
                bootstrap_ci_level=spec.ladder_bootstrap_ci_level,
                bootstrap_seed=spec.ladder_bootstrap_seed,
            )
            rungs_result[rung_job.name] = {
                "games": agg.games, "wins": agg.wins, "losses": agg.losses, "draws": agg.draws,
                "wr": agg.wr, "wr_ci_lower": agg.wr_ci_lower, "wr_ci_upper": agg.wr_ci_upper,
                "eff_n": agg.eff_n, "regime_key": agg.regime_key, "status": "active",
            }

        random_records = _play_random_floor(spec, candidate_engine, board_factory)
        random_agg = (
            aggregate_rung(
                random_records,
                bootstrap_resamples=spec.ladder_bootstrap_resamples,
                bootstrap_ci_level=spec.ladder_bootstrap_ci_level,
                bootstrap_seed=spec.ladder_bootstrap_seed,
            )
            if random_records
            else None
        )
        random_result = (
            {"games": 0, "wr": None}
            if random_agg is None
            else {"games": random_agg.games, "wr": random_agg.wr}
        )

        return {
            "step": spec.step,
            "gate": gate_result,
            "rungs": rungs_result,
            "skipped_rungs": skipped_rungs,
            "random": random_result,
            "worker_pid": os.getpid(),
        }
    finally:
        candidate_engine.close()


def worker_main(spec_path: str | Path, result_path: str | Path) -> None:
    """The spawn-ctx `Process` target. Writes the result ATOMICALLY (tmp + os.replace)."""
    spec = RoundSpec.from_dict(json.loads(Path(spec_path).read_text()))
    result = run_round(spec)
    target = Path(result_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(result))
    tmp.replace(target)


def _main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print("usage: python -m mantis.eval.worker <spec.json> <result.json>", file=sys.stderr)
        return 2
    worker_main(argv[0], argv[1])
    return 0


if __name__ == "__main__":
    sys.exit(_main())
