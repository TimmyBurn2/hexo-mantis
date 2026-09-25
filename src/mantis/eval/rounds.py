# Exceeds the 300-line soft cap (R8): the round contract, its far-side rehydration table and the
# result-shape validation are ONE unit; a field split from its rehydration row arrives raw.
"""RoundSpec (PATHS AND PRIMITIVES ONLY: no live model crosses the process seam) and
`build_round_result`, the coordinator-facing round-result mapping."""
from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mantis.config.resolve.eval_posture import PlyCapAdjudicationSpec, StrengthFloorSpec
from mantis.config.resolve.fused_graph_caps import FusedGraphCapsSpec
from mantis.config.resolve.inference_batching import InferenceBatchingSpec
from mantis.eval.errors import EvalBrokenReason, ResultContractError

_LOG = logging.getLogger(__name__)

#: The contract-doc / schema-census name of the gate-block concurrency row: it lives beside the
#: spec that carries it, because the name belongs with the consumer, not with the test.
EVAL_CONCURRENCY_ROW = "eval.concurrency"

__all__ = [
    "EVAL_CONCURRENCY_ROW",
    "GATE_STREAM_FIELDS",
    "GameRecordTarget",
    "RoundSpec",
    "build_round_result",
    "gate_stream_fields",
    "validate_worker_result",
]

#: The gate's rule fields riding `eval_round_complete.gate` (CARD-EVAL-GATE-FIELDS-IN-STREAM): how
#: it stopped, read and cost; names are the child's own, and `wall_sec` is the gate block's.
GATE_STREAM_FIELDS: tuple[str, ...] = (
    "rule", "pairs_played", "stopped", "llr", "wr_confirm", "n_pooled", "promoted", "wall_sec",
)


def gate_stream_fields(gate: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The `eval_round_complete.gate` projection of a gate mapping; `None` when no gate ran.

    Every field is read with `.get`, never a subscript: the A-3 partial sidecar and an older
    result mapping may lack one, and an absent field must read as `None`, never kill the poller.
    """
    if gate is None:
        return None
    return {name: gate.get(name) for name in GATE_STREAM_FIELDS}


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


#: The resolver-produced specs `from_dict` must REHYDRATE; a `RoundSpec` field forgotten here
#: arrives in the child raw and fails at its first attribute read, in an unread stderr.
_REHYDRATED_SPEC_FIELDS: tuple[tuple[str, Any], ...] = (
    ("ply_cap_adjudication", PlyCapAdjudicationSpec),
    ("strength_floor", StrengthFloorSpec),
    ("fused_graph_caps", FusedGraphCapsSpec),
    ("inference_batching", InferenceBatchingSpec),
    ("game_record", GameRecordTarget),
)


@dataclass(frozen=True)
class RungJob:
    """One external-opponent block the child plays beside the gate. No production round
    carries one; the strix cells compose theirs in `tools/strength_frontier.py`, so the
    block's pair-bootstrap terms ride the job."""

    name: str
    bot: str
    variant: str
    opponent_sims: int | None
    opening_book: str
    deploy_matched: bool
    games: int
    bootstrap_resamples: int
    bootstrap_ci_level: float
    bootstrap_seed: int


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
    #: The round's ordinal, monotone and resume-restored, keeping each round's openings off the
    #: last's; never parsed from `round_id`, whose format is display, not contract.
    round_index: int
    step: int
    candidate_snapshot: str
    best_snapshot: str | None
    best_step: int | None
    encoding: str
    worker_device: str
    gate: GateSpec
    #: `[]` on every production round; a strix cell carries exactly one.
    rung_jobs: list[RungJob]
    random_floor_games: int
    random_model_sims: int
    seed_base: int
    round_timeout_sec: float
    result_path: str
    progress_path: str
    #: Where this round's games are WRITTEN, or `None` for a round that records none. Same shape
    #: and reason as the postures below: resolved once in the parent, carried as data.
    game_record: GameRecordTarget | None
    #: The two early-strength eval postures, resolved ONCE in the parent and carried as frozen
    #: dataclasses that round-trip without a schema import. `None` is the ARMED=NO posture.
    ply_cap_adjudication: PlyCapAdjudicationSpec | None
    strength_floor: StrengthFloorSpec | None
    #: The graph forward's memory bound for the worker, a SECOND allocator on the card whose
    #: server is built with no `RunConfig`. `None` must round-trip as `None`.
    fused_graph_caps: FusedGraphCapsSpec | None
    #: The EVAL leaf-graph build's width, derived in the parent because the child has no
    #: `RunConfig`. `1` is the serial path and the exact-parity control.
    leaf_build_threads: int
    #: The deploy head's MCTS leaf-batch width, so the eval child searches under the SAME regime
    #: the net's targets came from. NOT defaulted: a default silently restores the k=1 mismatch.
    leaf_batch_size: int
    #: The run's `eval.max_plies`, its own row since 2026-09-15.
    max_plies: int
    #: The deploy head's completed-Q sigma terms — REQUIRED schema keys the eval head never
    #: received, so the deploy-matched bar searched at a regime nobody minted.
    c_visit: float
    c_scale: float
    q_rescale: bool
    #: The run's `deploy.search.kind` and `selfplay.gumbel_m`, resolved by the SAME authority
    #: `SelfPlayHParams.from_config` reads. NOT defaulted.
    search_kind: str
    gumbel_m: int
    #: The graph collector's batching geometry, resolved in the parent: a wrong child literal
    #: cost 33% of the eval path's ms/sim in the collector's own deadline.
    inference_batching: InferenceBatchingSpec | None
    #: The gate-block concurrency. NOT defaulted: a spec silently carrying `1` while the config
    #: minted `4` is the silently-disabled-knob class.
    concurrency: int
    #: A rung job's games in flight — the cell's own `concurrency` (no config row).
    rung_concurrency: int
    #: The minted allocator REGIME, asserted by the child for its own process; `None` is safe
    #: because a cuda consumer REQUIRES a token.
    allocator_posture: str | None = None
    #: The candidate's sims against a strix rung (RUNG-2): `None` on every production round, set
    #: by the frontier tool's strix cells; a strix job without it is refused by name.
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
    gate_result: Any,
    eval_round_wall_sec: float,
    reason: EvalBrokenReason | None,
    detail: str | None,
    random_wr: float | None,
    worker_pid: int | None = None,
    candidate_snapshot_path: str | None = None,
    strength_floor: Mapping[str, Any] | None = None,
    gate_verdict_partial: bool = False,
) -> dict[str, Any]:
    """Assemble the coordinator-facing round-result mapping — success and broken rounds alike.

    ONE authority for "did this round break": the typed `reason`, where `None` IS the clean
    state, with no defaulted boolean survivor beside it. `detail` is PROSE and nothing under
    `src/` may branch on it. `gate_verdict_partial`: the verdict is the child's partial sidecar (A-3).
    """
    promoted = (reason is None or gate_verdict_partial) and _gate_result_promoted(gate_result)
    result: dict[str, Any] = {
        "step": step,
        "round_id": round_id,
        "promoted": promoted,
        "promoted_step": step if promoted else None,
        "wr_random": random_wr,
        "eval_round_wall_sec": eval_round_wall_sec,
        "eval_broken_reason": reason,
        "eval_broken_detail": detail,
        "gate_verdict_partial": bool(gate_verdict_partial),
        "gate": _gate_result_to_mapping(gate_result),
    }
    if worker_pid is not None:
        result["worker_pid"] = worker_pid
    if candidate_snapshot_path is not None:
        result["candidate_snapshot_path"] = candidate_snapshot_path
    # PRESENCE is the arming evidence: a disarmed round writes no key, since `None` would read
    # "armed, and it passed nothing".
    if strength_floor is not None:
        result["strength_floor"] = dict(strength_floor)
    return result
