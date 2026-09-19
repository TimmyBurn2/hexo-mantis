"""mantis.eval — pipeline, promotion gate, aggregation.

Public API surface: `build_eval_pipeline`/`EvalPipeline`/`DrainCaps` (pipeline.py),
`apply_gate_decision`/`DeployTagHooks` (promote.py),
`aggregate_gate`/`aggregate_rung`/`gate_promotion_decision`/
`should_escalate` (aggregate.py), `build_round_result` (rounds.py),
`evaluate_strength_floor`/`probe_measurements`/`StrengthFloorVerdict` (floor_gate.py),
the error taxonomy (errors.py), and the snapshot write/load pair (snapshot.py).
"""
from __future__ import annotations

from mantis.eval.aggregate import (
    GateAggregate,
    RungAggregate,
    aggregate_gate,
    aggregate_rung,
    gate_promotion_decision,
    pair_bootstrap_wr_ci,
    should_escalate,
)
from mantis.eval.errors import (
    BookError,
    EvalBrokenError,
    MixedRegimeError,
    ResultContractError,
    RungUnresolvable,
)
from mantis.eval.floor_gate import (
    StrengthFloorVerdict,
    evaluate_strength_floor,
    probe_measurements,
)
from mantis.eval.pipeline import DrainCaps, EvalPipeline, build_eval_pipeline
from mantis.eval.promote import DeployTagHooks, apply_gate_decision
from mantis.eval.rounds import RoundSpec, build_round_result
from mantis.eval.snapshot import load_model_snapshot, write_model_snapshot

__all__ = [
    "BookError",
    "DrainCaps",
    "EvalBrokenError",
    "EvalPipeline",
    "GateAggregate",
    "MixedRegimeError",
    "DeployTagHooks",
    "ResultContractError",
    "RoundSpec",
    "RungAggregate",
    "RungUnresolvable",
    "StrengthFloorVerdict",
    "aggregate_gate",
    "aggregate_rung",
    "apply_gate_decision",
    "build_eval_pipeline",
    "build_round_result",
    "evaluate_strength_floor",
    "gate_promotion_decision",
    "load_model_snapshot",
    "pair_bootstrap_wr_ci",
    "probe_measurements",
    "should_escalate",
    "write_model_snapshot",
]
