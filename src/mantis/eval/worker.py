"""mantis.eval.worker — CHILD-ONLY module.

Entry: `python -m mantis.eval.worker <spec.json> <result.json>` / spawn target `worker_main`.
Loads snapshots, builds nets on `spec.worker_device`, then plays the strength floor probe (when
armed), the gate block, the resolved ladder rungs (a per-rung `RungUnresolvable` is RECORDED,
never fatal) and the random floor, writing the sidecar result JSON ATOMICALLY.

>300 justify (R8): one entry point owning all four blocks, which share the candidate player,
inference engine, book loading, the ONE encoding resolution, the decode-capability guard and
the graph decode+expand collaborator. Splitting would duplicate that setup and let the phases
drift out of the one-worker-process-per-round contract.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mantis.arena.adjudicate import PlyCapAdjudicator
from mantis.arena.books import round_openings
from mantis.arena.deploy_head import DeployHeadPlayer
from mantis.arena.match import play_paired_match
from mantis.arena.regime import RegimeKey
from mantis.bots.protocol import RungUnresolvable
from mantis.bots.resolve import resolve_bot
from mantis.config.resolve.allocator_posture import assert_posture_token
from mantis.encoding import EncodingSpec, lookup, normalize_encoding_name
from mantis.eval.aggregate import aggregate_gate, aggregate_rung
from mantis.eval.child_memory import make_probe
from mantis.eval.errors import EvalDecodeUnsupportedError
from mantis.eval.floor_gate import FLOOR_PROBE_VARIANT, evaluate_strength_floor
from mantis.eval.rounds import GameRecordTarget, RoundSpec, RungJob
from mantis.eval.snapshot import load_model_snapshot
from mantis.monitor.game_record import (
    GameRecordError,
    GameRecordWriter,
    eval_record,
    seat_result,
)
from mantis.monitor.sink import RunIdError
from mantis.selfplay.inference_local import LocalInferenceEngine

#: Confirm-phase opening seed offset, so the confirm block draws a DIFFERENT book slice.
_CONFIRM_SEED_OFFSET = 7919

#: Policy-pool values the eval decode ENTRANCE actually implements: the grid arm's dense
#: `infer_batch` scatter-MAXes and DROPS off-window cells, the graph arm drops nothing. A
#: CLOSED SET, so a registry row declaring `scatter_mean` is refused, not silently max-pooled.
_DECODE_IMPLEMENTED_POLICY_POOLS = frozenset({"none", "scatter_max"})

#: Value-pool values the eval decode ENTRANCE implements, on the same CLOSED-SET discipline:
#: `value_pool` has NO Python consumer (the grid arm hardcodes `.min()`, the graph arm pools
#: nothing), so a row declaring `"mean"` would otherwise be silently min-pooled.
_DECODE_IMPLEMENTED_VALUE_POOLS = frozenset({"none", "min"})


def _assert_decode_implements_declared_pooling(spec: EncodingSpec) -> None:
    """Refuse a round whose DECLARED pooling this worker's decode cannot honour.

    Policy channel first — the SHIPPED order, so no already-refused encoding changes message.
    """
    _assert_policy_pool_implemented(spec)
    _assert_value_pool_implemented(spec)


def _assert_policy_pool_implemented(spec: EncodingSpec) -> None:
    if spec.policy_pool in _DECODE_IMPLEMENTED_POLICY_POOLS:
        return
    raise EvalDecodeUnsupportedError(
        f"encoding {spec.name!r} declares policy_pool={spec.policy_pool!r}, which this eval "
        f"worker's decode entrance does not implement: on the GRID arm DeployHeadPlayer "
        f"reaches the net through LocalInferenceEngine.infer_batch, whose dense arm "
        f"scatter-maxes and DROPS off-window cells. The grid no-drop decode "
        f"(infer_batch_per_cluster + the Rust expand_and_backup_ls) exists but is not wired "
        f"to the deploy head (ADJ-WP12R-4). Refusing to report an eval result pooled "
        f"differently from the encoding's own declaration."
    )


def _assert_value_pool_implemented(spec: EncodingSpec) -> None:
    if spec.value_pool in _DECODE_IMPLEMENTED_VALUE_POOLS:
        return
    raise EvalDecodeUnsupportedError(
        f"encoding {spec.name!r} declares value_pool={spec.value_pool!r}, which this eval "
        f"worker's decode entrance does not implement: nothing in the Python decode READS "
        f"the field. The grid arm hardcodes a min-reduction over cluster windows "
        f"(LocalInferenceEngine.infer_batch: 'v = float(board_values.min())') and the graph "
        f"arm performs no reduction at all. Implemented: "
        f"{sorted(_DECODE_IMPLEMENTED_VALUE_POOLS)}. Refusing to report an eval result whose "
        f"value channel was pooled differently from the encoding's own declaration "
        f"(ADJ-WP12R-6)."
    )


class _RoundProgress:
    """Per-game progress, written by the CHILD as the round plays.

    PLAIN COUNTERS, LABELS AND TIMESTAMPS ONLY — no moves, no positions, no trajectory hash, so
    the redaction discipline holds by construction. `margin` is `None` when no adjudicator was
    armed or the cap was not reached — never `0`, which is a MEASURED margin.

    A write error is reported ONCE on stderr and disables further writes, never raising:
    deliberately NOT LAW-14's posture, because this file is diagnostic.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._games = 0
        self._disabled = False
        #: The phase whose sink was built most recently, recorded here rather than threaded
        #: through a sixth parameter. Read only by the dump-on-fire context.
        self.current_phase = "before_first_block"

    def sink(self, phase: str) -> Callable[[Any], None]:
        """Return a `play_paired_match` sink for one phase; records `current_phase`."""
        self.current_phase = phase

        def _record(game_record: Any) -> None:
            self._games += 1
            adjudication = getattr(game_record, "adjudication", None)
            colors = getattr(game_record, "colors", None) or {}
            self._write({
                "game_index": self._games,
                "phase": phase,
                # A record shape carrying no ply count must write NULLS, not a ply-ZERO game.
                "plies": (None if getattr(game_record, "plies", None) is None
                          else int(game_record.plies)),
                "t_wall": round(time.time(), 3),
                "terminal": getattr(game_record, "terminal", None),
                "winner": getattr(game_record, "winner", None),
                "candidate_color": colors.get("candidate"),
                "margin": None if adjudication is None else int(adjudication.margin),
            })
        return _record

    def _write(self, row: dict[str, Any]) -> None:
        if self._disabled:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        except OSError as exc:
            self._disabled = True
            print(
                f"eval progress writes DISABLED after {exc!r} on {self._path} — the round "
                f"continues; only its progress visibility is lost (R319(e)(ii))",
                file=sys.stderr, flush=True,
            )


class _RoundGameRecords:
    """The eval channels' GAME-RECORD-1 producer, separate from `_RoundProgress` because that
    class satisfies its no-positions discipline by construction and moves are what a game record
    is for. `None` for `target` is the no-op arm.

    A CONSTRUCTION FAILURE IS NOT FATAL HERE, unlike the trainer's arm: killing a round over an
    unwritable directory trades a lost record for a skipped gate. Reported once on stderr.
    """

    def __init__(self, target: GameRecordTarget | None, *, round_id: str, step: int) -> None:
        self._writer: GameRecordWriter | None = None
        if target is not None:
            try:
                self._writer = GameRecordWriter(
                    record_dir=target.record_dir, run_id=target.run_id)
            except (OSError, GameRecordError, RunIdError) as exc:
                print(
                    f"eval game records DISABLED for round {round_id} after {exc!r} on "
                    f"{target.record_dir} — the round continues and its promotion decision "
                    f"is unaffected; only this round's games are unrecorded",
                    file=sys.stderr, flush=True,
                )
        self._run_id = "" if target is None else target.run_id
        self._round_id = round_id
        self._step = int(step)
        self._games = 0

    def sink(
        self, phase: str, *, channel: str, rung: str, served_sims: int, seed: int
    ) -> Callable[[Any], None]:
        """Return a `play_paired_match` sink for one block of the round.

        `channel` is passed rather than derived from `phase`, which would be a second authority.
        """
        def _record(game_record: Any) -> None:
            if self._writer is None:
                return
            self._games += 1
            colour = int(game_record.colors["candidate"])
            self._writer.write(eval_record(
                game_id=f"{self._round_id}_{phase}_{self._games:05d}",
                run_id=self._run_id, step=self._step, channel=channel, rung=rung,
                phase=phase, game_index=self._games,
                moves=game_record.moves,
                result=seat_result(game_record.winner, colour),
                plies=int(game_record.plies), termination=str(game_record.terminal),
                candidate_color=colour, seed=seed, served_sims=served_sims,
                trajectory_hash=getattr(game_record, "trajectory_hash", None),
                search_stats=getattr(game_record, "search_stats", None),
            ))
        return _record

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()

    @property
    def games_written(self) -> int:
        return 0 if self._writer is None else self._writer.games_written


def _both(*sinks: Callable[[Any], None]) -> Callable[[Any], None]:
    """Fan one `record_sink` out to several; `play_paired_match` calls its sink in LOOP ORDER."""
    def _fan(record: Any) -> None:
        for sink in sinks:
            sink(record)
    return _fan


def _agg_record(game_record: Any) -> dict[str, Any]:
    """Convert an arena `GameRecord` to aggregate.py's record convention (p1 == candidate)."""
    winner = {"candidate": "p1", "opponent": "p2", "draw": "draw"}[game_record.winner]
    return {
        "p1": "cand", "p2": "opponent", "winner": winner,
        "moves": [list(m) for m in game_record.moves],
        "regime_key": game_record.regime_key.canonical(),
        "trajectory_hash": game_record.trajectory_hash,
        # The SEAT. `trajectory_hash` covers the MOVE LIST ALONE, so two legs of a colour pair
        # with coinciding moves hash identically and LAW-04's dedupe collapsed them to ONE game
        # — dropping a result whose outcome is typically the OPPOSITE of the leg it kept.
        "candidate_color": game_record.colors["candidate"],
        # The PAIR the legs belong to: `candidate_color` keeps them DISTINCT for the dedupe,
        # this keeps them RELATED for a bootstrap that must resample openings, not games.
        "opening_id": game_record.opening_id,
    }


def _model_sims_for_kind(spec: RoundSpec, kind: str) -> int:
    return {
        "sealbot": spec.sealbot_model_sims,
        "random": spec.random_model_sims,
    }[kind]


def _graph_expand_fn(engine: LocalInferenceEngine, spec: EncodingSpec):
    """Decode and expand one graph leaf through `expand_and_backup_ls_at` — the same producer,
    expand and frame self-play uses, with the builder's window centre threaded from the producer
    so the bridge's leaf/policy alignment cross-check is possible."""
    def _expand(tree, leaves) -> None:
        dense, overflow, values, centers = engine.infer_batch_ls(leaves)
        tree.expand_and_backup_ls_graph(
            dense, overflow, values, centers, spec.policy_logit_count, spec.trunk_size,
        )

    return _expand


def build_candidate_player(
    engine: LocalInferenceEngine, n_sims: int, *, spec: EncodingSpec, leaf_batch_size: int,
    c_visit: float, c_scale: float, search_kind: str, gumbel_m: int, gumbel_seed: int,
) -> DeployHeadPlayer:
    """Build the candidate player by a CLOSED match on the DECLARED representation.

    Never on a model attribute and never falling through to a dense arm: an unregistered
    representation must not silently become a dropping decode. `leaf_batch_size` is THREADED.
    """
    if spec.representation == "graph":
        return DeployHeadPlayer(expand_fn=_graph_expand_fn(engine, spec), n_sims=n_sims,
                                leaf_batch_size=leaf_batch_size,
                                c_visit=c_visit, c_scale=c_scale,
                                search_kind=search_kind, gumbel_m=gumbel_m,
                                gumbel_seed=gumbel_seed)
    if spec.representation == "grid":
        return DeployHeadPlayer(infer_fn=engine.infer, n_sims=n_sims,
                                c_visit=c_visit, c_scale=c_scale,
                                leaf_batch_size=leaf_batch_size,
                                search_kind=search_kind, gumbel_m=gumbel_m,
                                gumbel_seed=gumbel_seed)
    raise EvalDecodeUnsupportedError(
        f"encoding {spec.name!r} declares representation={spec.representation!r}, which "
        f"this eval worker's decode entrance does not implement. The implemented arms are "
        f"'grid' (infer_batch) and 'graph' (infer_batch_ls). Refusing to fall through to "
        f"either arm — a decode chosen by fallthrough is the defect this match exists to "
        f"prevent."
    )


def _build_adjudicator(spec: RoundSpec) -> PlyCapAdjudicator | None:
    """Build the round's ONE ply-cap adjudicator, or None on the disarmed posture.

    Shared by every phase so the fire tally covers the whole round rather than one block.
    """
    posture = spec.ply_cap_adjudication
    if posture is None:
        return None
    return PlyCapAdjudicator(posture.criterion, posture.min_margin)


def _collate_dump_target(spec: RoundSpec, progress: _RoundProgress) -> tuple[str, Any]:
    """Return where a graph-contract failure is dumped, and what context rides with it. The
    directory is DERIVED from the round's progress path, so no second path authority exists; the
    context is a CALLABLE because `phase` is read at the moment of the fire."""
    def _context() -> dict[str, Any]:
        phase = progress.current_phase
        return {
            "round_id": spec.round_id,
            "step": spec.step,
            "encoding": spec.encoding,
            "phase": phase,
            # THE CONCURRENCY IN FORCE: only the gate block runs concurrently.
            "concurrency": spec.concurrency if phase.startswith("gate_") else 1,
            "gate_concurrency_armed": spec.concurrency,
            "leaf_build_threads": spec.leaf_build_threads,
            "worker_device": spec.worker_device,
        }

    return str(Path(spec.progress_path).parent), _context


def _play_floor_probe(
    spec: RoundSpec, probe_games: int, candidate_engine: LocalInferenceEngine, board_factory,
    *, encoding_spec: EncodingSpec, adjudicator: PlyCapAdjudicator | None,
    progress: _RoundProgress,
    games: _RoundGameRecords,
) -> list:
    """Play the strength-floor probe: `probe_games` games against the CHEAPEST opponent.

    Same opponent, sims and book as the random floor, but `RegimeKey.variant` is
    `FLOOR_PROBE_VARIANT` so the sets cannot pool. Returns arena `GameRecord`s, whose
    `terminal` field the floor's decisiveness bar reads.
    """
    bot_factory = resolve_bot("random", depth=None, opponent_sims=spec.random_model_sims)
    opponent = bot_factory(seed=spec.seed_base)
    candidate = build_candidate_player(
        candidate_engine, spec.random_model_sims, spec=encoding_spec,
        leaf_batch_size=spec.leaf_batch_size,
        c_visit=spec.c_visit, c_scale=spec.c_scale,
        search_kind=spec.search_kind, gumbel_m=spec.gumbel_m, gumbel_seed=spec.seed_base,
    )
    regime_key = RegimeKey(
        bot="random", variant=FLOOR_PROBE_VARIANT, model_sims=spec.random_model_sims,
        opponent_spec="random:uniform", opening_book=spec.gate.opening_book,
        deploy_matched=True, encoding=spec.encoding,
    )
    openings = round_openings(
        spec.gate.opening_book, n_pairs=max(probe_games // 2, 1),
        seed_base=spec.seed_base, round_index=spec.round_index,
    )
    records = play_paired_match(
        candidate, opponent, openings, regime_key=regime_key,
        board_factory=board_factory, record_sink=_both(progress.sink("floor_probe"), games.sink("floor_probe", channel="random_floor", rung="random", served_sims=spec.random_model_sims, seed=spec.seed_base)), adjudicator=adjudicator, max_plies=spec.max_plies,
    )
    return list(records[:probe_games])


def _play_gate_block(
    spec: RoundSpec,
    candidate_engine: LocalInferenceEngine,
    board_factory,
    *,
    encoding_spec: EncodingSpec,
    adjudicator: PlyCapAdjudicator | None,
    progress: _RoundProgress,
    games: _RoundGameRecords,
) -> dict | None:
    """Play the gate block: candidate vs the best anchor, deploy-matched, screen -> confirm
    escalation. Returns the raw `{"screen": [...], "confirm": [...]}` lists, or None with no
    best anchor yet.

    THE ROUND'S ONLY CONCURRENT BLOCK — 93 % of the round's wall. DISCLOSED at G > 1:
    `record_sink` is called in loop order AFTER the block completes, so the progress file goes
    quiet and then fills; at `concurrency == 1` the arena's serial loop is byte-exact.
    """
    if spec.best_snapshot is None or not spec.gate.run_gate:
        return None

    best_model = load_model_snapshot(spec.best_snapshot, device=spec.worker_device)
    best_engine = LocalInferenceEngine(
        best_model, _device(spec.worker_device), encoding_spec=encoding_spec,
        # The parent's bound crosses the process seam on the spec: this child has its OWN CUDA
        # context and allocator, invisible to the parent's in-process bound.
        fused_graph_caps=spec.fused_graph_caps,
        inference_batching=spec.inference_batching,
        max_in_flight=spec.leaf_batch_size,
        leaf_build_threads=spec.leaf_build_threads,
        collate_check_period=1,
        collate_dump=_collate_dump_target(spec, progress),
    )
    try:
        def _pair() -> tuple[Any, Any]:
            return (
                build_candidate_player(
                    candidate_engine, spec.gate.deploy_sims, spec=encoding_spec,
                    leaf_batch_size=spec.leaf_batch_size,
                    c_visit=spec.c_visit, c_scale=spec.c_scale,
                    search_kind=spec.search_kind, gumbel_m=spec.gumbel_m,
                    gumbel_seed=spec.seed_base,
                ),
                build_candidate_player(
                    best_engine, spec.gate.deploy_sims, spec=encoding_spec,
                    leaf_batch_size=spec.leaf_batch_size,
                    c_visit=spec.c_visit, c_scale=spec.c_scale,
                    search_kind=spec.search_kind, gumbel_m=spec.gumbel_m,
                    gumbel_seed=spec.seed_base,
                ),
            )

        candidate, opponent = _pair()

        regime_key = RegimeKey(
            bot="best_anchor", variant="deploy", model_sims=spec.gate.deploy_sims,
            opponent_spec="best_anchor:deploy_matched", opening_book=spec.gate.opening_book,
            deploy_matched=True, encoding=spec.encoding,
        )
        # A per-ROUND window over a seed_base-seeded permutation, confirm offset by
        # `_CONFIRM_SEED_OFFSET`, so consecutive rounds and phases do not replay games.
        screen_openings = round_openings(
            spec.gate.opening_book, n_pairs=max(spec.gate.screen_games // 2, 1),
            seed_base=spec.gate.seed_base, round_index=spec.round_index,
        )
        screen_records = play_paired_match(
            candidate, opponent, screen_openings, regime_key=regime_key,
            board_factory=board_factory, record_sink=_both(progress.sink("gate_screen"), games.sink("gate_screen", channel="promotion", rung="anchor", served_sims=spec.gate.deploy_sims, seed=spec.gate.seed_base)), adjudicator=adjudicator, max_plies=spec.max_plies,
            player_factory=_pair, concurrency=spec.concurrency,
        )
        screen_agg = [_agg_record(r) for r in screen_records]

        wr_screen = _draw_aware_wr(screen_agg)
        escalate = wr_screen is not None and wr_screen >= spec.gate.screen_confirm_lo
        confirm_agg: list[dict[str, Any]] = []
        if escalate:
            confirm_openings = round_openings(
                spec.gate.opening_book, n_pairs=max(spec.gate.confirm_games // 2, 1),
                # The offset stays on the SEED. On the ROUND INDEX it is not equivalent:
                # screen and confirm draw windows of DIFFERENT widths from the SAME permutation,
                # so an index offset collides on a schedule — MEASURED at run6's 40/64 widths,
                # round 2's confirm drew ALL FORTY of the screen's openings. On its own
                # permutation the worst overlap is 8 of 40 against ~5 expected by chance.
                seed_base=spec.gate.seed_base + _CONFIRM_SEED_OFFSET,
                round_index=spec.round_index,
            )
            confirm_records = play_paired_match(
                candidate, opponent, confirm_openings, regime_key=regime_key,
                board_factory=board_factory, record_sink=_both(progress.sink("gate_confirm"), games.sink("gate_confirm", channel="promotion", rung="anchor", served_sims=spec.gate.deploy_sims, seed=spec.gate.seed_base + _CONFIRM_SEED_OFFSET)), adjudicator=adjudicator, max_plies=spec.max_plies,
                player_factory=_pair, concurrency=spec.concurrency,
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
    spec: RoundSpec, rung_job: RungJob, candidate_engine: LocalInferenceEngine, board_factory,
    *, encoding_spec: EncodingSpec, adjudicator: PlyCapAdjudicator | None,
    progress: _RoundProgress,
    games: _RoundGameRecords,
) -> list[dict[str, Any]]:
    bot_factory = resolve_bot(
        rung_job.bot, depth=rung_job.depth,
        opponent_sims=_model_sims_for_kind(spec, rung_job.bot),
    )
    opponent = bot_factory()
    # Rung games play at the resolved PER-KIND *_model_sims the RegimeKey stamps, never at
    # gate.deploy_sims, which is reserved for the deploy-matched GATE block.
    candidate = build_candidate_player(
        candidate_engine, _model_sims_for_kind(spec, rung_job.bot), spec=encoding_spec,
        leaf_batch_size=spec.leaf_batch_size,
        c_visit=spec.c_visit, c_scale=spec.c_scale,
        search_kind=spec.search_kind, gumbel_m=spec.gumbel_m, gumbel_seed=spec.seed_base,
    )
    regime_key = RegimeKey(
        bot=rung_job.bot, variant=rung_job.variant, model_sims=_model_sims_for_kind(spec, rung_job.bot),
        opponent_spec=f"{rung_job.bot}:{rung_job.variant}", opening_book=rung_job.opening_book,
        deploy_matched=rung_job.deploy_matched, encoding=spec.encoding,
    )
    openings = round_openings(
        rung_job.opening_book, n_pairs=max(rung_job.games // 2, 1),
        seed_base=spec.seed_base, round_index=spec.round_index,
    )
    records = play_paired_match(
        candidate, opponent, openings, regime_key=regime_key,
        board_factory=board_factory, record_sink=_both(progress.sink("rung"), games.sink("rung", channel="external", rung=rung_job.name, served_sims=_model_sims_for_kind(spec, rung_job.bot), seed=spec.seed_base)), adjudicator=adjudicator, max_plies=spec.max_plies,
    )
    return [_agg_record(r) for r in records[: rung_job.games]]


def _play_random_floor(
    spec: RoundSpec, candidate_engine: LocalInferenceEngine, board_factory,
    *, encoding_spec: EncodingSpec, adjudicator: PlyCapAdjudicator | None,
    progress: _RoundProgress,
    games: _RoundGameRecords,
) -> list[dict[str, Any]]:
    if spec.random_floor_games <= 0:
        return []
    bot_factory = resolve_bot("random", depth=None, opponent_sims=spec.random_model_sims)
    opponent = bot_factory(seed=spec.seed_base)
    # The random floor plays at the resolved random_model_sims, not gate.deploy_sims.
    candidate = build_candidate_player(
        candidate_engine, spec.random_model_sims, spec=encoding_spec,
        leaf_batch_size=spec.leaf_batch_size,
        c_visit=spec.c_visit, c_scale=spec.c_scale,
        search_kind=spec.search_kind, gumbel_m=spec.gumbel_m, gumbel_seed=spec.seed_base,
    )
    regime_key = RegimeKey(
        bot="random", variant="raw", model_sims=spec.random_model_sims,
        opponent_spec="random:uniform", opening_book=spec.gate.opening_book,
        deploy_matched=False, encoding=spec.encoding,
    )
    openings = round_openings(
        spec.gate.opening_book, n_pairs=max(spec.random_floor_games // 2, 1),
        seed_base=spec.seed_base, round_index=spec.round_index,
    )
    records = play_paired_match(
        candidate, opponent, openings, regime_key=regime_key,
        board_factory=board_factory, record_sink=_both(progress.sink("random_floor"), games.sink("random_floor", channel="random_floor", rung="random", served_sims=spec.random_model_sims, seed=spec.seed_base)), adjudicator=adjudicator, max_plies=spec.max_plies,
    )
    return [_agg_record(r) for r in records[: spec.random_floor_games]]


def _device(name: str):
    import torch

    return torch.device(name)


def _round_result(
    spec: RoundSpec, *, gate_result: dict | None, rungs_result: dict[str, Any],
    skipped_rungs: list[dict[str, str]], random_result: dict[str, Any],
    floor_payload: dict[str, Any] | None, adjudicator: PlyCapAdjudicator | None,
    device_memory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build THE sidecar result — one shape, one place, both exit paths. `_REQUIRED_RESULT_KEYS`
    are unconditional; a posture key is attached iff it was armed, so a disarmed run's result
    JSON stays byte-identical."""
    result: dict[str, Any] = {
        "step": spec.step,
        "gate": gate_result,
        "rungs": rungs_result,
        "skipped_rungs": skipped_rungs,
        "random": random_result,
        "worker_pid": os.getpid(),
    }
    if floor_payload is not None:
        result["strength_floor"] = floor_payload
    if adjudicator is not None:
        result["ply_cap_adjudication"] = {
            "criterion": adjudicator.criterion,
            "min_margin": adjudicator.min_margin,
            **adjudicator.tally(),
        }
    if device_memory is not None:
        # UNCONDITIONAL on both exit paths: a round that stopped at the strength floor is
        # exactly a round whose term is small, so dropping it would bias the series.
        result["device_memory"] = device_memory
    return result


def run_round(spec: RoundSpec) -> dict[str, Any]:
    """Play the full round described by `spec` and return the RAW result dict. Never raises for
    a resolver failure (per-rung skipped); model-load and engine-build failures propagate so the
    parent classifies them."""
    from mantis._engine import Board

    # The eval child asserts the allocator posture FOR ITSELF, first statement: it is a SECOND
    # allocator on the same card in its own process. A cuda worker_device with no token RAISES,
    # so the seam's `None` default can fail an assertion but never excuse one.
    assert_posture_token(spec.allocator_posture, device_type=spec.worker_device)
    probe = make_probe(spec.worker_device, round_id=spec.round_id)
    progress = _RoundProgress(spec.progress_path)
    # The round's game records, constructed here so a claim failure is loud at round START.
    games = _RoundGameRecords(spec.game_record, round_id=spec.round_id, step=spec.step)
    probe.mark("round_start")

    # ONE resolution of the DECLARED encoding, so geometry and decode cannot diverge.
    enc_name = normalize_encoding_name(spec.encoding)
    enc_spec: EncodingSpec = lookup(enc_name)
    _assert_decode_implements_declared_pooling(enc_spec)

    def board_factory():
        return Board.with_encoding_name(enc_name)

    candidate_model = load_model_snapshot(spec.candidate_snapshot, device=spec.worker_device)
    candidate_engine = LocalInferenceEngine(
        candidate_model, _device(spec.worker_device), encoding_spec=enc_spec,
        fused_graph_caps=spec.fused_graph_caps,
        inference_batching=spec.inference_batching,
        max_in_flight=spec.leaf_batch_size,
        leaf_build_threads=spec.leaf_build_threads,
        collate_check_period=1,
        collate_dump=_collate_dump_target(spec, progress),
    )

    adjudicator = _build_adjudicator(spec)

    try:
        # PHASE 0 — the strength floor runs BEFORE the gate block, the round's most expensive
        # phase, which the floor exists not to spend. Disarmed, this branch is not taken.
        floor_payload: dict[str, Any] | None = None
        if spec.strength_floor is not None:
            probe_records = _play_floor_probe(
                spec, spec.strength_floor.probe_games, candidate_engine, board_factory,
                encoding_spec=enc_spec, adjudicator=adjudicator, progress=progress, games=games,
            )
            verdict = evaluate_strength_floor(probe_records, spec.strength_floor)
            floor_payload = verdict.as_payload()
            probe.mark("floor_probe")
            if not verdict.passed:
                # The round STOPS and says so; the absent gate result is what blocks promotion.
                probe.mark("round_end")
                return _round_result(
                    spec, gate_result=None, rungs_result={}, skipped_rungs=[],
                    random_result={"games": 0, "wr": None},
                    floor_payload=floor_payload, adjudicator=adjudicator,
                    device_memory=probe.payload(),
                )

        gate_records = _play_gate_block(
            spec, candidate_engine, board_factory, encoding_spec=enc_spec,
            adjudicator=adjudicator, progress=progress, games=games,
        )
        # The one phase putting a SECOND model and engine on the card, skipped WHOLE with no
        # anchor. Marked whichever branch it took.
        probe.mark("gate_block")
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
                records = _play_rung_block(
                    spec, rung_job, candidate_engine, board_factory, encoding_spec=enc_spec,
                    adjudicator=adjudicator, progress=progress, games=games,
                )
            except RungUnresolvable as exc:
                skipped_rungs.append({"rung": rung_job.name, "reason": exc.reason})
                probe.mark(f"rung_skipped:{rung_job.name}")
                continue
            probe.mark(f"rung:{rung_job.name}")
            # Thread the bootstrap knobs through: aggregate.py's signature defaults had no
            # live consumer and made a minted value silently inert.
            agg = aggregate_rung(
                records,
                bootstrap_resamples=spec.ladder_bootstrap_resamples,
                bootstrap_ci_level=spec.ladder_bootstrap_ci_level,
                bootstrap_seed=spec.ladder_bootstrap_seed,
            )
            rungs_result[rung_job.name] = {
                "games": agg.games, "wins": agg.wins, "losses": agg.losses, "draws": agg.draws,
                "wr": agg.wr, "wr_ci_lower": agg.wr_ci_lower, "wr_ci_upper": agg.wr_ci_upper,
                # The CHILD has no `LadderState`, so a constant "active" mislabelled SATURATED
                # rungs; the parent stamps the real status, read BEFORE `record_round`.
                "eff_n": agg.eff_n, "regime_key": agg.regime_key,
            }

        random_records = _play_random_floor(
            spec, candidate_engine, board_factory, encoding_spec=enc_spec,
            adjudicator=adjudicator, progress=progress, games=games,
        )
        probe.mark("random_floor")
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

        probe.mark("round_end")
        return _round_result(
            spec, gate_result=gate_result, rungs_result=rungs_result,
            skipped_rungs=skipped_rungs, random_result=random_result,
            floor_payload=floor_payload, adjudicator=adjudicator,
            device_memory=probe.payload(),
        )
    finally:
        # The shard is closed and INDEXED on every exit path OUT OF THIS TRY: one left open by
        # a broken round is indistinguishable from "no games played". A raise BEFORE the try
        # leaves it unindexed, but `iter_run_games` scans shards rather than trusting the index.
        games.close()
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
    # The eval CHILD is its own process and installed no handler, so its INFO went nowhere.
    from mantis.monitor.logging_setup import configure_logging

    configure_logging()
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print("usage: python -m mantis.eval.worker <spec.json> <result.json>", file=sys.stderr)
        return 2
    worker_main(argv[0], argv[1])
    return 0


if __name__ == "__main__":
    sys.exit(_main())
