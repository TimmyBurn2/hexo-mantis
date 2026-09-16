# Exceeds the 300-line soft cap (R8): the round contract and its rehydration are ONE unit — the
# fields carried across the eval process seam, the table saying which must be rebuilt as a
# dataclass on the far side, and the result-shape validation the child answers with. A field
# split from its rehydration row arrives as a raw mapping and fails at the first attribute read.
"""RoundSpec (PATHS AND PRIMITIVES ONLY: no live model crosses the process seam), `build_round_result`
(sets `wr_sealbot` unconditionally) and `resolve_ladder_rungs` (records a `RungUnresolvable`, never fails the round)."""
from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mantis.bots.protocol import RungUnresolvable
from mantis.config.resolve.eval_posture import PlyCapAdjudicationSpec, StrengthFloorSpec
from mantis.config.resolve.fused_graph_caps import FusedGraphCapsSpec
from mantis.config.resolve.inference_batching import InferenceBatchingSpec
from mantis.eval.errors import EvalBrokenReason, ResultContractError

_LOG = logging.getLogger(__name__)

#: The contract-doc / schema-census name of the gate-block concurrency row: it lives beside the
#: spec that carries it, because the name belongs with the consumer, not with the test.
EVAL_CONCURRENCY_ROW = "eval.concurrency"
#: The rung block's own row (R351): same idiom, its one reader is `worker._play_rung_block`.
EVAL_RUNG_CONCURRENCY_ROW = "eval.rung_concurrency"

__all__ = [
    "EVAL_CONCURRENCY_ROW",
    "EVAL_RUNG_CONCURRENCY_ROW",
    "GameRecordTarget",
    "RoundSpec",
    "build_round_result",
    "resolve_ladder_rungs",
    "validate_worker_result",
]


def resolve_ladder_rungs(
    rungs: Sequence[Any], resolve_bot_fn: Callable[..., Any]
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Resolve each rung's bot; a `RungUnresolvable` is CAUGHT and appended to `skipped`, never
    raised further. Re-evaluated fresh every call."""
    resolved: dict[str, Any] = {}
    skipped: list[dict[str, str]] = []
    for rung in rungs:
        try:
            resolved[rung.name] = resolve_bot_fn(
                rung.bot, depth=rung.depth, opponent_sims=rung.opponent_sims
            )
        except RungUnresolvable as exc:
            skipped.append({"rung": rung.name, "reason": exc.reason})
    return resolved, skipped


def _rehydrate(cls: Any, payload: Any) -> Any:
    """Rebuild an optional posture spec from its JSON mapping; `None` stays `None`, and
    already-typed values pass through so an in-process spec and one read back from the worker's
    spec file take the same path — two construction routes are how a field ends up meaning one
    thing in-process and another across the seam."""
    if payload is None or isinstance(payload, cls):
        return payload
    return cls(**payload)


@dataclass(frozen=True)
class GameRecordTarget:
    """Where the eval CHILD writes its game records: resolved once in the parent and carried as
    data. `None` means this round records none — the state every test-constructed spec is in and
    one production must never be in. The child claims its OWN shard under `record_dir` with
    `O_CREAT|O_EXCL`, so a round child and the live trainer can never share a file."""

    record_dir: str
    run_id: str


#: The resolver-produced specs `from_dict` must REHYDRATE, as DATA rather than transcribed
#: statements: a field added to `RoundSpec` and forgotten here arrives in the child as a raw
#: mapping and fails at its first attribute read, in a subprocess whose stderr nobody reads.
_REHYDRATED_SPEC_FIELDS: tuple[tuple[str, Any], ...] = (
    ("ply_cap_adjudication", PlyCapAdjudicationSpec),
    ("strength_floor", StrengthFloorSpec),
    ("fused_graph_caps", FusedGraphCapsSpec),
    ("inference_batching", InferenceBatchingSpec),
    ("game_record", GameRecordTarget),
)


@dataclass(frozen=True)
class RungJob:
    name: str
    bot: str
    variant: str
    depth: int | None
    opponent_sims: int | None
    opening_book: str
    deploy_matched: bool
    games: int


@dataclass(frozen=True)
class GateSpec:
    stride: int
    screen_games: int
    confirm_games: int
    promotion_winrate: float
    screen_confirm_lo: float
    deploy_sims: int
    opening_book: str
    bootstrap_resamples: int
    min_distinct_per_pair: int
    seed_base: int
    run_gate: bool
    #: `eval.gate.sequential` as a plain mapping (`None` = screen/confirm). NOT defaulted: a spec
    #: silently carrying `None` while the config minted the block is the silently-disabled class.
    sequential: dict[str, Any] | None


@dataclass(frozen=True)
class RoundSpec:
    """PATHS AND PRIMITIVES ONLY — a torch module is not representable here."""

    round_id: str
    #: The round's ordinal within the run, monotone and resume-restored; it is what makes each
    #: round's opening subset non-overlapping with the last. Parsing it back out of `round_id`
    #: would be transcription — the id's format is a display decision, not a data contract.
    round_index: int
    step: int
    candidate_snapshot: str
    best_snapshot: str | None
    best_step: int | None
    encoding: str
    worker_device: str
    gate: GateSpec
    rung_jobs: list[RungJob]
    random_floor_games: int
    random_model_sims: int
    sealbot_model_sims: int
    seed_base: int
    round_timeout_sec: float
    result_path: str
    progress_path: str
    # The three ladder bootstrap keys threaded to the live aggregation path; before this they
    # had NO live consumer, the worker silently using the aggregator's signature defaults.
    ladder_bootstrap_resamples: int
    ladder_bootstrap_ci_level: float
    ladder_bootstrap_seed: int
    #: Where this round's games are WRITTEN, or `None` for a round that records none. Same shape
    #: and reason as the postures below: resolved once in the parent, carried as data.
    game_record: GameRecordTarget | None
    #: The two early-strength eval postures, resolved ONCE in the parent and carried as frozen
    #: dataclasses that round-trip without a schema import. `None` is the ARMED=NO posture.
    ply_cap_adjudication: PlyCapAdjudicationSpec | None
    strength_floor: StrengthFloorSpec | None
    #: The graph inference forward's memory bound: the eval worker is a SECOND allocator on the
    #: same card that no in-process bound can see, and its engine builds a server from a
    #: hand-made dict with no `RunConfig`. `None` is the GRID arm and must round-trip as `None`.
    fused_graph_caps: FusedGraphCapsSpec | None
    #: The EVAL leaf-graph build's width, derived in the parent because the child has no
    #: `RunConfig`. `1` is the serial path and the exact-parity control.
    leaf_build_threads: int
    #: The deploy head's MCTS leaf-batch width, so the eval child searches under the SAME regime
    #: the net's targets came from. NOT defaulted: a default silently restores the k=1 mismatch.
    leaf_batch_size: int
    #: The run's `eval.max_plies` (its own row since 2026-09-15; before that a copy of
    #: `selfplay.max_game_moves`, and before AUDIT-1 F-15 a hardcoded 128).
    max_plies: int
    #: The deploy head's completed-Q sigma terms — REQUIRED schema keys the eval head never
    #: received, so the deploy-matched bar searched at a regime nobody minted.
    c_visit: float
    c_scale: float
    q_rescale: bool
    #: The run's `deploy.search.kind` and `selfplay.gumbel_m`. The deploy head used to run a regime in
    #: NO config at all — a PUCT tree with a Gumbel-scored root pick — so the kind is resolved by
    #: the SAME authority `SelfPlayHParams.from_config` reads. NOT defaulted.
    search_kind: str
    gumbel_m: int
    #: The graph collector's batching geometry, resolved in the parent. These two knobs were
    #: LITERALS in the child's hand-made server dict, and a literal wrong for the route cost 33%
    #: of the eval path's ms/sim in the collector's own deadline. `None` is the GRID arm.
    inference_batching: InferenceBatchingSpec | None
    #: The allocator REGIME the caps were fitted under, as the config's minted token: a posture
    #: is a property of the PROCESS's environment, so the parent's assertion says nothing about
    #: the child's. Its default is safe because the consumer REQUIRES a token under cuda.
    #: The gate-block concurrency. NOT defaulted: a spec silently carrying `1` while the config
    #: minted `4` is the silently-disabled-knob class.
    concurrency: int
    #: The rung block's games in flight (`eval.rung_concurrency`). NOT defaulted, for the same reason.
    rung_concurrency: int
    allocator_posture: str | None = None
    #: The candidate's sims against a strix rung (RUNG-2): `None` on every production round (strix
    #: is not in the ladder), set by the frontier tool's strix cells; a strix job without it is refused by name.
    strix_model_sims: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RoundSpec:
        payload = dict(payload)
        payload["gate"] = GateSpec(**payload["gate"])
        payload["rung_jobs"] = [RungJob(**job) for job in payload["rung_jobs"]]
        for field, spec_cls in _REHYDRATED_SPEC_FIELDS:
            payload[field] = _rehydrate(spec_cls, payload[field])
        return cls(**payload)


_REQUIRED_RESULT_KEYS = (
    "step", "gate", "rungs", "skipped_rungs", "random", "worker_pid",
)


def partial_gate_path(result_path: str) -> Path:
    """The gate phase's PARTIAL sidecar beside `result_path`, written when the gate block ends (A-3)."""
    return Path(result_path + ".gate.partial.json")


def write_partial_gate(result_path: str, *, step: int, gate_result: Mapping[str, Any]) -> Path | None:
    """Persist the gate verdict atomically; `None` on an OSError (logged: the round survives without it)."""
    target = partial_gate_path(result_path)
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps({"step": int(step), "phase": "gate", "gate": dict(gate_result)}),
                       encoding="utf-8")
        tmp.replace(target)
    except OSError:
        _LOG.warning("partial_gate_not_written path=%s", target, exc_info=True)
        return None
    return target


def read_partial_gate(result_path: str, *, step: int) -> dict[str, Any] | None:
    """The partial gate verdict for THIS step, or `None` (absent, unreadable, another step's)."""
    try:
        raw = json.loads(partial_gate_path(result_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict) or raw.get("step") != step:
        return None
    gate = raw.get("gate")
    if not isinstance(gate, dict) or "promoted" not in gate:
        return None
    return gate


def validate_worker_result(raw: Any) -> dict[str, Any]:
    """The sidecar result JSON's shape contract. Any missing key -> named
    `ResultContractError` (never a silent partial read)."""
    if not isinstance(raw, dict):
        raise ResultContractError(f"worker result must be a JSON object, got {type(raw).__name__}")
    missing = [key for key in _REQUIRED_RESULT_KEYS if key not in raw]
    if missing:
        raise ResultContractError(f"worker result missing required key(s): {missing}")
    return raw


def _first_sealbot_wr(
    rungs_config: Sequence[Any], rung_results: Mapping[str, Mapping[str, Any]]
) -> tuple[float | None, str | None, int | None, float | None, float | None]:
    """`(wr, rung_name, games, ci_lower, ci_upper)` for the FIRST sealbot-kind rung with >= 1 game
    this round, all-`None` if none; the identity and the CI travel with the value out of the SAME
    walk, because a saturated rung draws 0 games off-cadence and the number would silently
    become the next rung's."""
    for rung in rungs_config:
        if getattr(rung, "bot", None) != "sealbot":
            continue
        info = rung_results.get(rung.name)
        if info is None:
            continue
        games = int(info.get("games", 0))
        if games <= 0:
            continue
        return (info.get("wr"), rung.name, games,
                info.get("wr_ci_lower"), info.get("wr_ci_upper"))
    return None, None, None, None, None


def _gate_result_to_mapping(gate_result: Any) -> dict[str, Any] | None:
    if gate_result is None:
        return None
    if dataclasses.is_dataclass(gate_result) and not isinstance(gate_result, type):
        payload = dataclasses.asdict(gate_result)
    elif isinstance(gate_result, Mapping):
        payload = dict(gate_result)
    else:
        raise TypeError(f"build_round_result: unsupported gate_result type {type(gate_result)!r}")
    payload.setdefault("reason", "")
    payload.setdefault("deploy_matched", True)
    return payload


def _gate_result_promoted(gate_result: Any) -> bool:
    if gate_result is None:
        return False
    if hasattr(gate_result, "promoted"):
        return bool(gate_result.promoted)
    if isinstance(gate_result, Mapping):
        return bool(gate_result.get("promoted", False))
    return False


def build_round_result(
    *,
    step: int,
    round_id: str,
    rungs_config: Sequence[Any],
    rung_results: Mapping[str, Mapping[str, Any]],
    gate_result: Any,
    skipped_rungs: Sequence[Mapping[str, str]],
    bt: Mapping[str, Any],
    schedule_next: Mapping[str, int],
    eval_round_wall_sec: float,
    reason: EvalBrokenReason | None,
    detail: str | None,
    random_wr: float | None,
    worker_pid: int | None = None,
    candidate_snapshot_path: str | None = None,
    strength_floor: Mapping[str, Any] | None = None,
    gate_verdict_partial: bool = False,
) -> dict[str, Any]:
    """Assemble the coordinator-facing round-result mapping, with `wr_sealbot` UNCONDITIONALLY
    present — success, broken and all-skip rounds alike.

    ONE authority for "did this round break": the typed `reason`, where `None` IS the clean
    state, with no defaulted boolean survivor beside it. `detail` is PROSE and nothing under
    `src/` may branch on it. `gate_verdict_partial`: the verdict is the child's partial sidecar (A-3).
    """
    promoted = (reason is None or gate_verdict_partial) and _gate_result_promoted(gate_result)
    _sealbot_reading = _first_sealbot_wr(rungs_config, rung_results)
    result: dict[str, Any] = {
        "step": step,
        "round_id": round_id,
        "promoted": promoted,
        "promoted_step": step if promoted else None,
        "wr_sealbot": _sealbot_reading[0],
        "wr_sealbot_rung": _sealbot_reading[1],
        "wr_sealbot_games": _sealbot_reading[2],
        # The ROUND CI, beside the win rate it belongs to and out of the same walk that
        # selected both: a bare win rate invites a reader to treat 32 games as a point estimate.
        "wr_sealbot_ci_lower": _sealbot_reading[3],
        "wr_sealbot_ci_upper": _sealbot_reading[4],
        "wr_random": random_wr,
        "eval_round_wall_sec": eval_round_wall_sec,
        "eval_broken_reason": reason,
        "eval_broken_detail": detail,
        "gate_verdict_partial": bool(gate_verdict_partial),
        "gate": _gate_result_to_mapping(gate_result),
        "rungs": dict(rung_results),
        "skipped_rungs": list(skipped_rungs),
        "bt": dict(bt),
        "schedule_next": dict(schedule_next),
    }
    if worker_pid is not None:
        result["worker_pid"] = worker_pid
    if candidate_snapshot_path is not None:
        result["candidate_snapshot_path"] = candidate_snapshot_path
    # PRESENCE is the arming evidence, as on the worker payload this copies from: a disarmed
    # round produces no key, and a key written unconditionally as `None` would report "armed, and
    # it passed nothing".
    if strength_floor is not None:
        result["strength_floor"] = dict(strength_floor)
    return result
