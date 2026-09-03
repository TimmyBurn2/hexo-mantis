# Exceeds the 300-line soft cap (R8): the round contract and its rehydration are ONE
# unit — every field carried across the eval process seam, the table that says which of
# them must be rebuilt as a dataclass on the far side, and the result-shape validation
# the child answers with. A field split from its rehydration row arrives in the child as
# a raw mapping and fails at the first attribute read, in a subprocess nobody is reading.
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
from mantis.config.resolve.inference_batching import InferenceBatchingSpec
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
    ("inference_batching", InferenceBatchingSpec),
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
    #: The EVAL leaf-graph build's width (NIGHTRUN-1 E1), derived ONCE in the parent by
    #: `mantis.config.resolve.leaf_build_threads` and carried here for `fused_graph_caps`'
    #: reason: the child has no `RunConfig` to derive a host reservation from. A plain int,
    #: so it needs no `_REHYDRATED_SPEC_FIELDS` row — `asdict`/`from_dict` round-trip it as
    #: itself. `1` is the serial path and the exact-parity control.
    leaf_build_threads: int
    #: The deploy head's MCTS leaf-batch width (R318(b)) — the config's OWN
    #: `selfplay.leaf_batch_size`, carried across the process seam so the eval child searches
    #: under the SAME regime the net's policy/value targets were generated in. Here for
    #: `fused_graph_caps`' reason: the child has no `RunConfig` to resolve against. NOT
    #: defaulted, unlike `allocator_posture` below — no consumer can refuse a bad value on this
    #: axis the way `assert_posture_token` refuses a missing posture, so a default would
    #: silently restore the k=1 train/deploy mismatch this field exists to close. A plain int,
    #: so it round-trips through `asdict`/`from_dict` with no rehydration entry.
    leaf_batch_size: int
    #: The run's declared `train.amp_dtype`, carried across the process seam for
    #: `fused_graph_caps`' reason (AUDIT-1 F-31). The eval child's `LocalInferenceEngine` has no
    #: `RunConfig` to resolve against, and its DENSE decodes carried no `dtype=` on their
    #: autocast at all — so the deploy-matched forward ran at torch's device default while the
    #: run declared something else, on the one path LAW-15 reads a promotion bar off. Resolved
    #: through `amp_dtype_for` (the ONE authority, LAW-06) inside the engine; threaded, never
    #: named at the construction site.
    amp_dtype: str
    #: The run's `selfplay.max_game_moves`, carried across the process seam for `amp_dtype`'s
    #: reason (AUDIT-1 F-15). `arena/match.py::DEFAULT_MAX_PLIES = 128` defaulted every eval
    #: game, and its own comment said it "mirrors the production self-play default" — a copy of
    #: a bridge signature default, itself a copy of the minted key. So the moment
    #: `max_game_moves` is re-minted, eval keeps capping at 128 silently and the draw channel
    #: changes meaning with no config diff, on the bar LAW-15 reads deploy-matched.
    max_plies: int
    #: The deploy head's completed-Q sigma terms, `selfplay.{c_visit, c_scale}` — REQUIRED
    #: schema keys that the eval head never received (AUDIT-1 F-39). `DeployHeadPlayer`
    #: defaulted them to `50.0` / `1.0` on its own signature and `eval/worker.py` constructed it
    #: with only `n_sims` and `leaf_batch_size`, so the deploy-matched bar searched at a regime
    #: nobody minted — and the moment `c_visit` is re-minted, LAW-15's "deploy-matched" claim
    #: quietly stops being true. Threaded here for `leaf_batch_size`' reason exactly.
    c_visit: float
    c_scale: float
    #: The graph collector's batching geometry — pop width and pop deadline — resolved ONCE in
    #: the parent by `mantis.config.resolve.inference_batching` and carried across the process
    #: seam, for `fused_graph_caps`' reason: the child's `LocalInferenceEngine` builds its graph
    #: server from a hand-made dict with no `RunConfig` to resolve against, and before this
    #: field those two knobs were LITERALS in that dict. The ledger measured the cost of a
    #: literal that is wrong for the route (F-2): at the single-stream deploy head, 33 % of the
    #: eval path's ms/sim was the collector's own deadline. NOT defaulted, for
    #: `leaf_batch_size`' reason — no consumer can refuse a bad value on this axis, so a
    #: default would silently restore the hardcode. `None` is the GRID arm: a grid eval round
    #: builds no graph server, so there is no collector geometry to carry.
    inference_batching: InferenceBatchingSpec | None
    #: The CUDA caching allocator REGIME the round's caps were fitted under (RECAL-PREP,
    #: R308(g)(i)), as the config's own minted token. Here for `fused_graph_caps`' reason and
    #: on the same seam: the child is a SECOND allocator on the same card, in its own process,
    #: and a posture is a property of the PROCESS's environment — so the parent's boot
    #: assertion says nothing about the child's, and the child has no `RunConfig` to resolve
    #: against. `None` is the NOT-CUDA arm: a cpu eval child has no caching allocator to
    #: govern. It is the one field on this dataclass carrying a DEFAULT, and the default is
    #: safe for the reason `ArmedAbort.ceiling_path`'s is: the consumer REQUIRES a token
    #: whenever `worker_device` is cuda and raises without one, so `None` can neither excuse
    #: an assertion nor pass for a posture. What it buys is that a round spec built by a test
    #: that has no opinion about allocators does not have to state one.
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
) -> tuple[float | None, str | None, int | None]:
    """`(wr, rung_name, games)` for the FIRST sealbot-kind rung (ladder order) with >=1 game
    this round; `(None, None, None)` if no sealbot rung recorded a game (skip-counted at the
    coordinator, G-2).

    AUDIT-1 F-14, producer half, completed under R332(b) — the R118/A-1 freeze on this file
    is LIFTED. The WR alone is not a series: once `sealbot_d5` saturates it draws 0 games
    off-cadence and the reported number silently becomes `sealbot_d6`'s, so a trajectory rule
    testing `wr < peak * ratio` compares two opponents. The identity travels with the value
    out of the SAME walk that selects it, so the two cannot drift.
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
        return info.get("wr"), rung.name, games
    return None, None, None


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
    _sealbot_reading = _first_sealbot_wr(rungs_config, rung_results)
    result: dict[str, Any] = {
        "step": step,
        "round_id": round_id,
        "promoted": promoted,
        "promoted_step": step if promoted else None,
        "wr_sealbot": _sealbot_reading[0],
        "wr_sealbot_rung": _sealbot_reading[1],
        "wr_sealbot_games": _sealbot_reading[2],
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
    # R324(d): PRESENCE is the arming evidence, exactly as it is on the worker payload this
    # copies from. A disarmed round — every committed config — produces no key here, so the
    # routed mapping is byte-identical to what it was before the floor existed. A key
    # written unconditionally as `None` would report "armed, and it passed nothing" for a
    # round the worker never applied the floor to.
    if strength_floor is not None:
        result["strength_floor"] = dict(strength_floor)
    return result
