"""mantis.eval — pipeline, promotion gate, opponent ladder, Bradley-Terry, aggregation.

Public API surface: `build_eval_pipeline`/`EvalPipeline`/`DrainCaps` (pipeline.py),
`apply_gate_decision`/`DeployTagHooks` (promote.py), `LadderState` (ladder.py),
`fit_bt`/`predict_p` (bt.py), `aggregate_gate`/`aggregate_rung`/`gate_promotion_decision`/
`should_escalate` (aggregate.py), `build_round_result`/`resolve_ladder_rungs` (rounds.py),
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
from mantis.eval.bt import fit_bt, predict_p
from mantis.eval.errors import (
    BookError,
    EvalBrokenError,
    LadderStateError,
    MixedRegimeError,
    ResultContractError,
    RungUnresolvable,
)
from mantis.eval.ladder import LadderState
from mantis.eval.pipeline import DrainCaps, EvalPipeline, build_eval_pipeline
from mantis.eval.promote import DeployTagHooks, apply_gate_decision
from mantis.eval.rounds import RoundSpec, build_round_result, resolve_ladder_rungs
from mantis.eval.snapshot import load_model_snapshot, write_model_snapshot

__all__ = [
    "BookError",
    "DrainCaps",
    "EvalBrokenError",
    "EvalPipeline",
    "GateAggregate",
    "LadderState",
    "LadderStateError",
    "MixedRegimeError",
    "DeployTagHooks",
    "ResultContractError",
    "RoundSpec",
    "RungAggregate",
    "RungUnresolvable",
    "aggregate_gate",
    "aggregate_rung",
    "apply_gate_decision",
    "build_eval_pipeline",
    "build_round_result",
    "fit_bt",
    "gate_promotion_decision",
    "load_model_snapshot",
    "pair_bootstrap_wr_ci",
    "predict_p",
    "resolve_ladder_rungs",
    "should_escalate",
    "write_model_snapshot",
]
