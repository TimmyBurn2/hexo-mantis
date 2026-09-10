# Exceeds the 300-line soft cap (R8): the round contract and its rehydration are ONE unit — the
# fields carried across the eval process seam, the table saying which must be rebuilt as a
# dataclass on the far side, and the result-shape validation the child answers with. A field
# split from its rehydration row arrives as a raw mapping and fails at the first attribute read.
"""RoundSpec + build_round_result + resolve_ladder_rungs.

`RoundSpec` is PATHS AND PRIMITIVES ONLY — the type surface cannot carry a live model across the
process seam. `build_round_result` UNCONDITIONALLY sets `wr_sealbot`, and `resolve_ladder_rungs`
CATCHES a `RungUnresolvable` and records it rather than failing the round.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from mantis.bots.protocol import RungUnresolvable
from mantis.config.resolve.eval_posture import PlyCapAdjudicationSpec, StrengthFloorSpec
from mantis.config.resolve.fused_graph_caps import FusedGraphCapsSpec
from mantis.config.resolve.inference_batching import InferenceBatchingSpec
from mantis.eval.errors import EvalBrokenReason, ResultContractError

#: The contract-doc / schema-census name of the gate-block concurrency row: it lives beside the
#: spec that carries it, because the name belongs with the consumer, not with the test.
EVAL_CONCURRENCY_ROW = "eval.concurrency"

__all__ = [
    "EVAL_CONCURRENCY_ROW",
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
    #: The run's `selfplay.max_game_moves`. A hardcoded 128 capped every eval game as a copy of a
    #: copy of the minted key, so a re-mint left eval capping silently and the draw channel
    #: changing meaning with no config diff.
    max_plies: int
    #: The deploy head's completed-Q sigma terms — REQUIRED schema keys the eval head never
    #: received, so the deploy-matched bar searched at a regime nobody minted.
    c_visit: float
    c_scale: float
    #: The run's `search.kind` and `selfplay.gumbel_m`. The deploy head used to run a regime in
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
    allocator_posture: str | None = None

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
    """Return `(wr, rung_name, games, ci_lower, ci_upper)` for the FIRST sealbot-kind rung with
    >= 1 game this round; all-`None` if none recorded a game.

    Once a sealbot rung saturates it draws 0 games off-cadence and the reported number silently
    becomes the next rung's, so the identity and the CI travel with the value out of the SAME
    walk that selects it.
    """
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
) -> dict[str, Any]:
    """Assemble the coordinator-facing round-result mapping, with `wr_sealbot` UNCONDITIONALLY
    present — success, broken and all-skip rounds alike.

    ONE authority for "did this round break": the typed `reason`, where `None` IS the clean
    state, with no defaulted boolean survivor beside it. `detail` is PROSE and nothing under
    `src/` may branch on it.
    """
    promoted = (reason is None) and _gate_result_promoted(gate_result)
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
