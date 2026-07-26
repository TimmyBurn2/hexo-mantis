"""SC-A1 oracle — `TrainConfig.entropy_reg_weight` sign law (R37; DESIGN_P2.md §2 /
PREREG_P2.md suite #2). Split out of test_train_schema.py per R8 (300-line soft cap).

RED-at-import until IMPL lands `TrainConfig`. A negative `entropy_reg_weight` must raise a
NAMED `ValueError` whose message states the R37.3 sign law ("positive coefficient" /
"subtracted entropy bonus") — not a bare pydantic bound message — because the SIGN LAW is
the point (ADJ-01: the historical `-0.005` was a sign-leak, never a real value in this
tree). `GRAPH_FORBIDDEN_NONZERO_WEIGHTS` (core.py) is untouched by the schema (R37.4 /
LAW-07: no duplicate authority) — grep-gated here.

The valid-payload dict is duplicated (not imported) from test_train_schema.py, matching
the repo's own cross-file idiom (`_valid_eval_block()` in tests/config/test_schema.py /
test_schema_strict.py) — R5 forbids a `tests` package (no `__init__.py`), so cross-test-
file imports are not available.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from mantis.config.schema import TrainConfig

REPO_ROOT = Path(__file__).resolve().parents[2]

# Duplicated verbatim from test_train_schema.py::VALID_TRAIN_PAYLOAD (zero-behavior-change
# mint values, DESIGN_P2.md §1.1/§2).
VALID_TRAIN_PAYLOAD: dict = {
    "lr": 1e-3, "weight_decay": 1e-4, "grad_clip": 1.0, "fp16": True, "amp_dtype": "fp16",
    "lr_schedule": "cosine", "total_steps": 1_000_000, "scheduler_t_max": None,
    "eta_min": 5e-4, "min_lr": None, "checkpoint_interval": 0,
    "actor_sync_cadence_steps": 1, "completed_q_values": False,
    "value_target": "pure_outcome_z", "policy_target": "raw_visit_distribution",
    "draw_reward": -0.5, "ply_cap_value": -0.5, "policy_prune_frac": 0.0,
    "entropy_reg_weight": 0.0, "aux_opp_reply_weight": 0.0, "uncertainty_weight": 0.0,
    "ownership_weight": 0.0, "threat_weight": 0.0, "aux_chain_weight": 0.0,
    "ply_index_weight": 0.0, "threat_pos_weight": 1.0,
}


def _payload(**over: object) -> dict:
    out = dict(VALID_TRAIN_PAYLOAD)
    out.update(over)
    return out


def test_negative_entropy_weight_raises_named_sign_error():
    with pytest.raises(ValueError) as exc:
        TrainConfig.model_validate(_payload(entropy_reg_weight=-0.005))
    message = str(exc.value)
    assert "positive coefficient" in message or "subtracted entropy bonus" in message, (
        f"expected the R37.3 sign-law message, got: {message!r}"
    )


def test_negative_entropy_weight_is_not_a_bare_pydantic_bound_message():
    with pytest.raises(ValidationError) as exc:
        TrainConfig.model_validate(_payload(entropy_reg_weight=-1.0))
    # a bare pydantic ge=0 rejection reads "greater than or equal to 0" with no sign-law
    # prose — the NAMED error must say more than that.
    assert "greater than or equal to 0" not in str(exc.value) or (
        "positive coefficient" in str(exc.value) or "subtracted entropy bonus" in str(exc.value)
    )


def test_entropy_weight_zero_constructs_cleanly():
    cfg = TrainConfig.model_validate(_payload(entropy_reg_weight=0.0))
    assert cfg.entropy_reg_weight == 0.0


def test_entropy_weight_positive_constructs_cleanly():
    cfg = TrainConfig.model_validate(_payload(entropy_reg_weight=0.25))
    assert cfg.entropy_reg_weight == 0.25


def test_schema_module_does_not_duplicate_graph_forbidden_nonzero_weights():
    # R37.4/LAW-07 producer test: the schema module must NOT import or re-declare the
    # `core.py:69-72` GRAPH_FORBIDDEN_NONZERO_WEIGHTS tuple — the graph-nonzero ban stays a
    # single train-step-time authority, never duplicated at load time.
    train_schema_src = (REPO_ROOT / "src" / "mantis" / "config" / "schema" / "train.py")
    text = train_schema_src.read_text()
    assert "GRAPH_FORBIDDEN_NONZERO_WEIGHTS" not in text
    assert not re.search(r"aux_opp_reply_weight.*uncertainty_weight.*ownership_weight", text)
