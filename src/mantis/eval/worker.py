"""mantis.eval.worker — CHILD-ONLY module (design §a.3 worker.py).

Entry: `python -m mantis.eval.worker <spec.json> <result.json>` / spawn target
`worker_main`. Loads snapshots, builds nets on `spec.worker_device`, plays: (0) the strength
floor probe, ONLY when `eval.strength_floor` is armed — a failing probe returns the round
here, before any expensive phase; (1) the gate block (candidate vs the best anchor,
deploy-matched, screen -> confirm escalation), (2) resolved ladder-rung blocks
(`RungUnresolvable` per rung is RECORDED, never fatal), (3) the random floor. Writes the
sidecar result JSON ATOMICALLY (tmp + os.replace). Imports
`mantis.selfplay.inference_local` for leaf inference — the ONE parent-side-excepted
inference surface (isolation law 1: this module is the out-of-process leg).

>300 justify (R8): one entry point
owning the gate block, ladder-rung blocks, and the random floor — each phase shares the
candidate DeployHeadPlayer/inference engine/book-loading machinery; splitting them would
duplicate that setup three times and let the phases drift out of the
SAME-process-covers-the-round contract this module exists to keep (isolation law: exactly
one worker process per round). WP12-R added the ONE encoding resolution and the
decode-capability guard that the same one-process contract requires be made once, here,
for every block below; Phase EVALDECODE (R138) added the CLOSED representation match and
the graph decode+expand collaborator, which all four blocks must reach through the same
single resolution — that is precisely why they cannot be split apart.
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

#: confirm-phase opening seed offset (deploy_strength_eval.py:519 parity) — the confirm
#: block draws a DIFFERENT slice of the book than the screen block.
_CONFIRM_SEED_OFFSET = 7919

#: Policy-pool values the eval decode ENTRANCE actually implements. `build_candidate_player`
#: matches CLOSED on `spec.representation` and the two arms drop DIFFERENTLY, which is why
#: this guard names the grid one:
#:   - GRID  -> `DeployHeadPlayer(infer_fn=engine.infer)` -> `LocalInferenceEngine.infer_batch`,
#:     whose dense arm scatter-MAXes over cluster windows and DROPS off-window cells
#:     (`if mcts_idx >= n_actions - 1: continue`, inference_local.py:207-208).
#:   - GRAPH -> `DeployHeadPlayer(expand_fn=...)` -> `infer_batch_ls` ->
#:     `expand_and_backup_ls_graph`, which drops NOTHING (WP12-R Phase EVALDECODE, R138).
#: "none" is the single-window / graph case, no pooling. A CLOSED SET, not a blocklist: a
#: registry row that later declares `scatter_mean` fails here until this seam implements it,
#: rather than being silently max-pooled.
_DECODE_IMPLEMENTED_POLICY_POOLS = frozenset({"none", "scatter_max"})

#: Value-pool values the eval decode ENTRANCE actually implements (ADJ-WP12R-6). Same
#: CLOSED-SET discipline as the policy pools above, and for the same reason: `value_pool`
#: has no Python consumer that reads it: `LocalInferenceEngine.infer_batch` HARDCODES the
#: cluster reduction (`v = float(board_values.min())`, inference_local.py:203) and the graph
#: arm does no pooling at all (one whole-board window; the dist65 head's value is returned
#: as-is). So the two implemented values are exactly:
#:   - "min"  -> the grid arm's hardcoded `.min()` over cluster windows.
#:   - "none" -> single-window grid, or graph: no reduction to perform.
#: A registry row later declaring `value_pool="mean"` or `"max"` would pass every existing
#: check and then be SILENTLY min-pooled, because nothing reads the field. This guard is
#: what converts that silent wrong answer into a refused round. It is the value-channel
#: half of the same defect class R138 named on the policy channel.
_DECODE_IMPLEMENTED_VALUE_POOLS = frozenset({"none", "min"})


def _assert_decode_implements_declared_pooling(spec: EncodingSpec) -> None:
    """Refuse a round whose DECLARED pooling this worker's decode cannot honour.

    Checks BOTH channels, policy first — that ordering is the SHIPPED one and is preserved
    deliberately, so no already-refused encoding changes which message it fails with. Each
    raise names its own channel, so the two are never confusable.
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
    """R319(e)(ii) — per-game progress, written by the CHILD as the round plays.

    THE DEFECT THIS CLOSES. `RoundSpec.progress_path` was constructed, threaded across the
    spawn seam and declared on the dataclass, and NOTHING wrote or read it — a declared field
    with no consumer. So an eval round was observable only as *started* and *finished/killed*,
    with no way to tell "most of the way through the screen block" from "stuck on game 1".
    RECAL-SITTING-3 spent two 3600 s drives unable to make that distinction, and the gap is
    what let a hardcoded `games_total: 0` be read as a measurement (§8.1 of the sitting record).

    PLAIN COUNTERS, LABELS AND TIMESTAMPS ONLY — no moves, no positions, no trajectory hash.
    The redaction discipline is satisfied by construction rather than by filtering: nothing
    here can carry a position, so there is nothing to redact.

    THE OUTCOME FIELDS (R320(c)). `terminal`, `winner`, `candidate_color` and `margin` ride the
    same row because the round R320 sends to measure asks for the margin DISTRIBUTION — at what
    margins capped games decide, and with what seat balance — and the adjudicator's own tally
    is four counters (`adjudicated`/`candidate`/`opponent`/`draw`), which reads the decisive
    RATE and nothing about its shape. Every one of these four is already in hand at the sink;
    before this they were dropped on the floor and no consumer could recover them. `margin` is
    `None` whenever no adjudicator was armed or the game did not reach the cap — a real absence,
    never a `0`, because `0` is a MEASURED margin (the two sides exactly level) and the two must
    not collide (R319(e)'s no-default-readable-as-a-measurement order, one field over).

    OBSERVABILITY MUST NOT BECOME A NEW FAILURE MODE. A write error is reported ONCE on stderr
    and then disables further writes; it never raises. That is deliberately NOT LAW-14's
    persistence-is-fatal posture, and the distinction is that this file is diagnostic: losing
    it costs visibility, while raising here would turn a progress line into a way to kill a
    round that was otherwise healthy.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._games = 0
        self._disabled = False
        #: R339(c): the phase whose sink was built most recently, i.e. the block about to play
        #: or playing. Recorded here rather than threaded through a sixth parameter because
        #: `sink(phase)` is already evaluated at each block's call site immediately before that
        #: block runs — so this costs no new production call site and cannot fall out of step
        #: with the blocks. Read only by the dump-on-fire context.
        self.current_phase = "before_first_block"

    def sink(self, phase: str) -> Callable[[Any], None]:
        """A `play_paired_match(record_sink=...)` callable for one phase of the round.

        SIDE EFFECT, deliberate: records `current_phase`. See the attribute's own note.
        """
        self.current_phase = phase

        def _record(game_record: Any) -> None:
            self._games += 1
            adjudication = getattr(game_record, "adjudication", None)
            colors = getattr(game_record, "colors", None) or {}
            self._write({
                "game_index": self._games,
                "phase": phase,
                # AUDIT-1 F-28/B03. `int(getattr(..., "plies", 0))` published a game that
                # ended at ply ZERO for a record shape carrying no ply count, and
                # `event_manifest.md`'s row for this writer already says a shape the writer
                # does not recognise writes NULLS rather than raising.
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
    """The eval channels' GAME-RECORD-1 producer (R344(b)).

    SEPARATE FROM `_RoundProgress` ON PURPOSE, and the separation is a property of that class
    rather than a preference: its docstring states *"PLAIN COUNTERS, LABELS AND TIMESTAMPS
    ONLY — no moves, no positions, no trajectory hash … nothing here CAN carry a position, so
    there is nothing to redact."* Moves are exactly what a game record is for, so putting
    them in that file would delete a discipline it satisfies by construction. Two files, two
    contracts.

    `None` for `target` is the no-op arm — the state every test-built `RoundSpec` is in.

    A CONSTRUCTION FAILURE IS NOT FATAL HERE, and that is the opposite of the trainer's arm on
    purpose. In `mantis.run` an un-openable store raises, because the run has not started and a
    run that cannot write its games should say so before it plays 25 000 of them. Inside a
    round the calculus inverts: the round produces the promotion decision the run gates on, and
    killing it over an unwritable directory would convert a lost record into a broken round, a
    skipped gate and a `eval_broken` the operator has to read. So the failure is reported ONCE
    on stderr and recording is off for this round — `_RoundProgress`'s posture under
    R319(e)(ii), for R319(e)(ii)'s reason.
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
        """A `play_paired_match(record_sink=...)` callable for one block of the round.

        `channel` is the ruling's own vocabulary and is passed rather than derived from
        `phase`: the gate's two phases are one channel and the floor's two are another, so a
        phase->channel map here would be a second place the round's shape is written down.
        """
        def _record(game_record: Any) -> None:
            if self._writer is None:
                return
            self._games += 1
            colour = int(game_record.colors["candidate"])
            self._writer.write(eval_record(
                # DETERMINISTIC, unlike self-play's uuid4: an eval game is identified by
                # where it sits in a round, so a reader who has the round's logs can name
                # the game those logs are about.
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
    """One `record_sink` that feeds several. `play_paired_match` calls its sink in LOOP
    ORDER under every concurrency, so both consumers see the same games in the same order —
    which is what lets a progress row and a game record be matched up after the fact."""
    def _fan(record: Any) -> None:
        for sink in sinks:
            sink(record)
    return _fan


def _agg_record(game_record: Any) -> dict[str, Any]:
    """Arena `GameRecord` -> the aggregate.py record convention `{p1,p2,winner,moves,
    regime_key}` (p1 == candidate always, by this worker's own construction)."""
    winner = {"candidate": "p1", "opponent": "p2", "draw": "draw"}[game_record.winner]
    return {
        "p1": "cand", "p2": "opponent", "winner": winner,
        "moves": [list(m) for m in game_record.moves],
        "regime_key": game_record.regime_key.canonical(),
        "trajectory_hash": game_record.trajectory_hash,
        # The SEAT (item 5(b)). `play_paired_match` plays every opening twice with
        # `candidate_color` in (1, -1), but `trajectory_hash` is a sha256 over the MOVE LIST
        # ALONE — so the two legs of a colour pair whose move sequences coincide hash
        # identically, and this record used to carry nothing else to tell them apart
        # (`p1`/`p2` are the constants "cand"/"opponent" by this worker's own construction).
        # LAW-04's dedupe then collapsed the pair to ONE game, dropping a real result whose
        # outcome is typically the OPPOSITE of the leg it kept — a WR biased toward whichever
        # leg arrived first, on half the eff_n. Carrying the colour keeps both legs distinct.
        "candidate_color": game_record.colors["candidate"],
        # R345(b)(4). The PAIR the two legs above belong to. `candidate_color` keeps the legs
        # DISTINCT for LAW-04's dedupe; this keeps them RELATED for the bootstrap, which must
        # resample openings rather than games because the two legs of one opening start from
        # the same position and their outcomes are correlated. The arena has always stamped
        # `opening_id` on the record and nothing downstream ever read it.
        "opening_id": game_record.opening_id,
    }


def _model_sims_for_kind(spec: RoundSpec, kind: str) -> int:
    return {
        "sealbot": spec.sealbot_model_sims,
        "random": spec.random_model_sims,
    }[kind]


def _graph_expand_fn(engine: LocalInferenceEngine, spec: EncodingSpec):
    """The graph arm's decode+expand collaborator (WP12-R Phase EVALDECODE, R138).

    Consumes BOTH halves the shared producer returns and expands through
    `expand_and_backup_ls_at` — the same producer, the same expand and the same frame
    self-play uses (`search_drive.rs:373 -> :397 -> :421`). Nothing is reimplemented here;
    the builder's window centre is threaded from the producer rather than re-derived,
    which is what makes the bridge's leaf/policy alignment cross-check possible.
    """
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
    """CLOSED match on the DECLARED representation — never on a model attribute, and
    never with a dense arm as the fallthrough. An unregistered representation is the
    exact input that must not silently become a dropping decode (R138's class).

    `leaf_batch_size` is THREADED, never defaulted (R318(b)): it is the round spec's copy of
    the config's `selfplay.leaf_batch_size`, so both arms search at the width the net's
    policy/value targets were generated at.
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
    """The round's ONE ply-cap adjudicator, or None on the disarmed posture.

    Built once and shared by every phase so the LAW-18 fire tally covers the whole round
    rather than one block of it. `None` is what every committed config produces
    (`eval.ply_cap_adjudication: null`), and on that arm `play_paired_match` takes the same
    branch it took before this parameter existed.
    """
    posture = spec.ply_cap_adjudication
    if posture is None:
        return None
    return PlyCapAdjudicator(posture.criterion, posture.min_margin)


def _collate_dump_target(spec: RoundSpec, progress: _RoundProgress) -> tuple[str, Any]:
    """R339(c): where a graph-contract failure is dumped, and what context rides with it.

    The directory is DERIVED from the round's own progress path — the eval work dir under the
    run record — rather than named here, so the dump lands wherever the round's other artifacts
    already do and no second path authority exists. Untracked by R7, like everything else there.

    The context is a CALLABLE because its interesting half moves during the round: `phase` is
    read at the moment of the fire, so a dump says whether the failure happened inside the one
    CONCURRENT block or in one of the three serial ones. That distinction is what R339(c)'s
    halt condition turns on, and a dict snapshotted at construction could not carry it.
    """
    def _context() -> dict[str, Any]:
        phase = progress.current_phase
        return {
            "round_id": spec.round_id,
            "step": spec.step,
            "encoding": spec.encoding,
            "phase": phase,
            # THE CONCURRENCY IN FORCE, not the round's arming: only the gate block runs
            # concurrently, so every other phase is serial by construction and says 1.
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

    Same opponent kind, sims and book as the random floor — the point of the probe is that it
    is the cheapest measurement already in the round's vocabulary, not a new regime. Its
    `RegimeKey.variant` is `FLOOR_PROBE_VARIANT` rather than the floor's `"raw"` so the two
    sets can never pool through `aggregate_rung`'s single-regime rule.

    Returns arena `GameRecord`s, NOT `_agg_record` dicts: the floor's decisiveness bar reads
    `GameRecord.terminal`, which the aggregate convention does not carry.
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
    """The gate block: candidate vs the best anchor, deploy-matched, screen -> confirm
    escalation (`should_escalate`, the SINGLE lower-bound test). Returns the raw
    `{"screen": [...], "confirm": [...]}` record lists, or None when there is no best
    anchor to play against yet (run3 `run(best_model=None)` parity).

    THE ROUND'S ONLY CONCURRENT BLOCK (R339(b)). It is 93 % of the round's wall, which is why
    the key reaches here and nowhere else. At `spec.concurrency == 1` — the schema default and
    every config that predates the row — `play_paired_match` never calls `_pair` and runs the
    same serial loop on the same two objects, byte-exact. DISCLOSED at G > 1: the arena calls
    `record_sink` in loop order AFTER the block completes rather than as each game lands, so
    the R319(e)(ii) progress file goes quiet for the block's duration and then fills. Nothing
    branches on it — `pipeline.read_progress` is reporting only, and its docstring forbids a
    caller from branching on the value — but a reader watching the file will see a gap.
    """
    if spec.best_snapshot is None or not spec.gate.run_gate:
        return None

    best_model = load_model_snapshot(spec.best_snapshot, device=spec.worker_device)
    best_engine = LocalInferenceEngine(
        best_model, _device(spec.worker_device), encoding_spec=encoding_spec,
        # The parent's resolved bound, carried across the process seam on the spec (D-1).
        # This child has its OWN CUDA context and its own allocator, so the in-process bound
        # the parent's server carries is blind to it — this is how the cap gets here.
        fused_graph_caps=spec.fused_graph_caps,
        # `None` on a grid round is CORRECT and travels: the engine refuses it only on
        # the graph branch, which is the one place that knows the route.
        inference_batching=spec.inference_batching,
        max_in_flight=spec.leaf_batch_size,
        # AUDIT-1 F-31: the declared autocast dtype, threaded on the spec for the same reason
        # as the three above — the dense decode had none at all.
        # NIGHTRUN-1 E1, same seam and same reason as the three above: the child has no
        # `RunConfig` to derive a host reservation from, and a serial leaf build is 95 % of
        # this path's measured cost.
        leaf_build_threads=spec.leaf_build_threads,
        # R339(c): 1-in-1 on the eval path. `F-816-37`'s only occurrence was here, and the
        # production rate was 1-in-64, so the rate at which it could have been caught is
        # unknown. ~93 games a round is what makes every-collate affordable on this path.
        collate_check_period=1,
        collate_dump=_collate_dump_target(spec, progress),
    )
    try:
        def _pair() -> tuple[Any, Any]:
            # R339(b): ONE construction expression for both the serial pair and every
            # concurrent thread's pair. Two copies would be two authorities over the search
            # regime the deploy-matched bar is read at (LAW-15).
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
        # R345(b)(4): a per-ROUND window over a seed_base-seeded permutation, so round N and
        # round N+1 do not play the same games. The confirm phase offsets its round index by
        # `_CONFIRM_SEED_OFFSET` so an escalation draws openings the screen did not, which is
        # what the old `seed_base + offset` was doing on the seed axis.
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
                # The offset stays on the SEED, which is where it always was and where it has
                # to be. Moving it to the ROUND INDEX (the first cut of R345(b)(4)) looked
                # equivalent and was not: screen and confirm draw windows of DIFFERENT widths
                # from the SAME permutation, so an index offset only shifts where each lands
                # and they collide on a schedule. MEASURED at run6's own 40/64 pair widths:
                # round 2's confirm block drew ALL FORTY of the screen's openings, rounds 1
                # and 3 drew 24 and 32 — a confirm phase that mostly REPLAYS the screen, with
                # deterministic argmax players producing the same games, for 128 games of
                # wall-clock and no new evidence. On its own permutation the overlap is
                # incidental: worst 8 of 40 over twelve rounds against ~5 expected by chance.
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
    # M-3: the candidate plays rung games at the resolved PER-KIND *_model_sims — the same
    # value the RegimeKey below stamps — never at gate.deploy_sims (that value is reserved
    # for the deploy-matched GATE block, LAW-15). Playing deploy_sims here while stamping a
    # different regime value was an A3 mislabel (the record must describe the regime it
    # actually played).
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
    # M-3: same fix as the rung block — the random floor plays at the resolved
    # random_model_sims, not gate.deploy_sims.
    candidate = build_candidate_player(
        candidate_engine, spec.random_model_sims, spec=encoding_spec,
        leaf_batch_size=spec.leaf_batch_size,
        c_visit=spec.c_visit, c_scale=spec.c_scale,
        search_kind=spec.search_kind, gumbel_m=spec.gumbel_m, gumbel_seed=spec.seed_base,
    )
    regime_key = RegimeKey(
        bot="random", variant="raw", model_sims=spec.random_model_sims,
        opponent_spec="random:uniform", opening_book=spec.gate.opening_book,
        # AUDIT-1 F-28/B02, and it is M-3's mislabel one block over: the random floor plays
        # at `random_model_sims` against a uniform bot, which is not the deploy-matched
        # comparison LAW-15 reserves the term for. Only the GATE block is deploy-matched —
        # there both sides play at `spec.gate.deploy_sims`, so its `True` is derived from the
        # construction and stays.
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
    """THE sidecar result builder — one shape, one place, both exit paths.

    The six `_REQUIRED_RESULT_KEYS` are unconditional. The two posture keys are attached IFF
    their posture was armed, which is the discipline `_broken_result`'s `detail` /
    `exception_class` extras already follow and it is what makes the disarmed run's result
    JSON byte-identical to the pre-change tree — not merely equivalent, identical, including
    the key set a future consumer might iterate.
    """
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
        # RECAL-PREP / R308(g)(ii). UNCONDITIONAL on both exit paths, unlike the two posture
        # keys above: those are attached iff their posture was armed, and this one is a
        # MEASUREMENT the re-sit needs from every round including the ones that stopped early.
        # A round that returned at the strength floor is exactly a round whose term is small,
        # and dropping it would bias the series the growth verdict is taken over.
        result["device_memory"] = device_memory
    return result


def run_round(spec: RoundSpec) -> dict[str, Any]:
    """Play the full round described by `spec`; return the RAW (validate_worker_result-
    shaped) result dict. Never raises for a resolver failure (per-rung skipped, never
    fatal) — an unrecoverable failure (model load, engine build) is allowed to propagate
    so the parent's join/exit-code path classifies it (isolation law 2)."""
    from mantis._engine import Board

    # RECAL-PREP / R308(g)(i): the eval child asserts the allocator posture FOR ITSELF, first
    # statement, before any model is loaded or any engine is built. The child is a SECOND
    # allocator on the same card in its own process; the parent's boot assertion says nothing
    # about the environment a hand-launched `python -m mantis.eval.worker` runs in, and a
    # posture is a property of THIS process. A cuda worker_device with no token RAISES —
    # `assert_posture_token`'s own refusal — so the seam's `None` default can fail an
    # assertion but never excuse one.
    assert_posture_token(spec.allocator_posture, device_type=spec.worker_device)
    probe = make_probe(spec.worker_device, round_id=spec.round_id)
    # R319(e)(ii): per-game progress, written as the round plays. Constructed HERE, beside the
    # memory probe, because both are round-scoped observability the phases below share.
    progress = _RoundProgress(spec.progress_path)
    # R344(b): the round's game records, beside the progress file and deliberately not in it
    # (see `_RoundGameRecords`). Constructed here so a claim failure is loud at round START
    # rather than on the round's first finished game.
    games = _RoundGameRecords(spec.game_record, round_id=spec.round_id, step=spec.step)
    probe.mark("round_start")

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
        candidate_model, _device(spec.worker_device), encoding_spec=enc_spec,
        fused_graph_caps=spec.fused_graph_caps,
        # `None` on a grid round is CORRECT and travels: the engine refuses it only on
        # the graph branch, which is the one place that knows the route.
        inference_batching=spec.inference_batching,
        max_in_flight=spec.leaf_batch_size,
        # AUDIT-1 F-31 — see the gate block's site for the reason.
        # NIGHTRUN-1 E1 — see the gate block's site for the reason.
        leaf_build_threads=spec.leaf_build_threads,
        # R339(c) — see the gate block's site. Both of the round's engines are armed: the
        # failing collate came off whichever engine was serving the position, and arming one
        # would leave half the round's forwards at 1-in-64.
        collate_check_period=1,
        collate_dump=_collate_dump_target(spec, progress),
    )

    adjudicator = _build_adjudicator(spec)

    try:
        # PHASE 0 — the strength floor. It runs BEFORE the gate block because the gate block
        # is the round's most expensive phase and the whole point of the floor is not to
        # spend it (F-R-P2B-5: 0 games in a full 4 h cap, gate-block-first). On the disarmed
        # posture — every committed config — this branch is not taken and the round begins at
        # the gate block exactly as it did before, same phase order, same seeds, same games.
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
                # The round STOPS here, and says so. Nothing is fabricated: no gate result,
                # no rung results, no random-floor number — the absence of a gate result is
                # what keeps `promote.apply_gate_decision` from promoting, by the same
                # `gate_result is None` route a round with no anchor already takes.
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
        # The one phase that puts a SECOND model and a SECOND engine on the card, and the one
        # that is skipped WHOLE while there is no anchor. Marked whichever branch it took, so
        # the series carries the round's posture rather than only its number.
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
                # AUDIT-1 F-28/B02. `"status": "active"` was a constant here: the CHILD has
                # no `LadderState`, so it labelled a SATURATED rung's calibration games
                # "active" every round. The parent stamps the real status in
                # `EvalPipeline._success_result`, read BEFORE `record_round` so it is the
                # status the rung was PLAYED under.
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
        # The shard is closed and INDEXED on every exit path OUT OF THIS TRY — the two early
        # returns above and any raise inside it. A shard left open by a round that broke is a
        # shard the index never names, which is the one state a reader cannot distinguish from
        # "this round played no games". It is NOT every exit path of the function: a raise
        # BEFORE the try (engine construction) leaves the shard open and unindexed, and that
        # is stated rather than claimed away — the records are still readable, because
        # `iter_run_games` scans shards and does not trust the index to enumerate them.
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
    # AUDIT-1 F-08: the eval CHILD is its own process and installed no handler either, so every
    # INFO it emitted — including its allocator-posture assertion — went nowhere.
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
