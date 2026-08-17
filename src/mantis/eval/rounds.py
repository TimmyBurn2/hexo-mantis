"""RoundSpec + build_round_result + resolve_ladder_rungs (design §a.3 rounds.py).

`RoundSpec` is PATHS AND PRIMITIVES ONLY — a torch module is not a representable field
(isolation law: the type surface cannot carry a live model across the process seam).
`build_round_result` assembles the coordinator-facing round-result mapping (§c.2);
it UNCONDITIONALLY sets `wr_sealbot` — the concrete producer the WP13-A G-2 prereg flip
points at. `resolve_ladder_rungs` is the parent-side per-rung resolution unit: a
`RungUnresolvable` is CAUGHT and recorded, never fatal to the round.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from mantis.bots.protocol import RungUnresolvable
from mantis.config.resolve.eval_posture import PlyCapAdjudicationSpec, StrengthFloorSpec
from mantis.config.resolve.fused_graph_caps import FusedGraphCapsSpec
from mantis.eval.errors import EvalBrokenReason, ResultContractError

__all__ = [
    "RoundSpec",
    "build_round_result",
    "resolve_ladder_rungs",
    "validate_worker_result",
]


# ── the parent-side per-rung resolver unit ──────────────────────────────────────────────
def resolve_ladder_rungs(
    rungs: Sequence[Any], resolve_bot_fn: Callable[..., Any]
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Attempt `resolve_bot_fn(rung.bot, depth=rung.depth, opponent_sims=rung.opponent_sims)`
    per rung; a `RungUnresolvable` is CAUGHT and appended to `skipped` — NEVER raised
    further (never fatal to the round). Re-evaluated fresh every call (never sticky)."""
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


# ── the worker round spec (JSON-(de)serializable; paths + primitives only) ─────────────
def _rehydrate(cls: Any, payload: Any) -> Any:
    """Rebuild an optional posture spec from its JSON mapping; `None` stays `None`.

    Already-typed values pass through unchanged so a spec built in-process (the terminal
    round never round-trips through JSON on the parent side) and one read back from the
    worker's spec file take the same path — the alternative, two construction routes, is how
    a field ends up meaning one thing in-process and another across the seam.
    """
    if payload is None or isinstance(payload, cls):
        return payload
    return cls(**payload)


#: The optional resolver-produced specs `from_dict` must REHYDRATE, as DATA rather than as
#: three transcribed statements. One loop over one table is what keeps the set closed: a field
#: added to `RoundSpec` and forgotten here arrives in the child as a raw mapping and fails at
#: the child's first attribute read — which, on `fused_graph_caps`, would be at the moment it
#: tries to bound a forward, in a subprocess whose stderr nobody is reading.
_REHYDRATED_SPEC_FIELDS: tuple[tuple[str, Any], ...] = (
    ("ply_cap_adjudication", PlyCapAdjudicationSpec),
    ("strength_floor", StrengthFloorSpec),
    ("fused_graph_caps", FusedGraphCapsSpec),
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
    kraken_model_sims: int
    strix_model_sims: int
    seed_base: int
    round_timeout_sec: float
    result_path: str
    progress_path: str
    # M-2: eval.ladder.bootstrap_{resamples,ci_level,seed} threaded to the live rung
    # aggregation path (worker.py's `aggregate_rung` calls) — these three schema fields
    # had NO live consumer before (worker.py silently used aggregate.py's own signature
    # defaults instead), which is exactly the silently-disabled-knob class R1/LAW-08 exist
    # to kill.
    ladder_bootstrap_resamples: int
    ladder_bootstrap_ci_level: float
    ladder_bootstrap_seed: int
    #: The two early-strength eval postures (F-R-P2B-5), resolved ONCE in the parent by
    #: `mantis.config.resolve.eval_posture` and carried across the process seam as plain
    #: dataclasses — paths-and-primitives still holds, since both are frozen records of
    #: scalars that `dataclasses.asdict`/`from_dict` round-trip without a schema import in
    #: the child. `None` is the ARMED=NO posture every committed config mints, and on that
    #: arm neither the arena loop nor `run_round` takes a new branch.
    ply_cap_adjudication: PlyCapAdjudicationSpec | None
    strength_floor: StrengthFloorSpec | None
    #: The graph inference forward's memory bound (F-816-10, D-1), in the SAME shape and for
    #: the same reason as the two postures above: resolved ONCE in the parent by
    #: `mantis.config.resolve.fused_graph_caps` and carried across the process seam as a plain
    #: frozen dataclass. It is here because the eval worker is a SECOND allocator on the same
    #: card — `eval.worker_device: cuda` plus a spawn-context subprocess — that no in-process
    #: bound can see, and its `LocalInferenceEngine` builds its graph server from a hand-made
    #: dict with no `RunConfig` to resolve against. `None` is the GRID arm: a grid eval round
    #: has no fused graph forward to bound, and `None` must round-trip as `None` rather than
    #: as a rehydration failure.
    fused_graph_caps: FusedGraphCapsSpec | None

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


# ── worker result-contract validation ───────────────────────────────────────────────────
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


# ── the round-result builder (§c.2) ─────────────────────────────────────────────────────
def _first_sealbot_wr(
    rungs_config: Sequence[Any], rung_results: Mapping[str, Mapping[str, Any]]
) -> float | None:
    """`wr_sealbot` = the FIRST sealbot-kind rung (ladder order) with >=1 game this round;
    None if no sealbot rung ever recorded a game (skip-counted at the coordinator, G-2)."""
    for rung in rungs_config:
        if getattr(rung, "bot", None) != "sealbot":
            continue
        info = rung_results.get(rung.name)
        if info is None:
            continue
        if int(info.get("games", 0)) <= 0:
            continue
        return info.get("wr")
    return None


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
) -> dict[str, Any]:
    """Assemble the coordinator-facing round-result mapping (§c.2). `wr_sealbot` is
    UNCONDITIONALLY present — success, broken, and all-skip rounds alike.

    WP12-R Phase O (R152/R79): ONE authority for "did this round break" — the typed
    `reason`, where `None` IS the clean state. The `eval_broken: bool` parameter and the
    `error: str | None` parameter are DELETED rather than defaulted: a defaulted survivor
    is a MIGRATED authority, not an absent one (`run.py:366-372`, MF-2 Attack B), and
    `eval_broken=True, error=None` was constructible while both existed. `detail` is PROSE
    beside the reason (`repr(exc)`, a persistence message) and never a reason spelling —
    nothing under `src/` may branch on it
    (`tests/eval/test_round_result_reason_shape.py::test_no_module_under_src_branches_on_the_eval_broken_detail`).
    """
    promoted = (reason is None) and _gate_result_promoted(gate_result)
    result: dict[str, Any] = {
        "step": step,
        "round_id": round_id,
        "promoted": promoted,
        "promoted_step": step if promoted else None,
        "wr_sealbot": _first_sealbot_wr(rungs_config, rung_results),
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
    return result
