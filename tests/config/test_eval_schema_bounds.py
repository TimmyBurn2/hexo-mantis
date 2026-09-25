"""Numeric-bounds validation on the eval/gate schema fields.

Pre-fix, `random_model_sims=-5` and `gate.promotion_winrate=2.0` loaded SILENTLY — crashing
`np.quantile` inside a worker or disabling promotion forever. Every case is parametrized: one out-of-domain
value raises a `ValidationError` naming the field, one in-domain boundary value loads clean.

The two timeout fields were floor-only bounds that admitted a REAL `.inf` YAML literal end to
end, which reproduced a silent poller death through `Process.join(float("inf"))`. They now carry
`allow_inf_nan=False` and a finite ceiling. (The `eval.ladder` rows this file also bounded were
deleted with the sealbot rung.)
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from mantis.config.schema import SCHEMA_VERSION, RunConfig, _EVAL_TIMEOUT_CEILING_SEC
from _schema_blocks import inference_block, monitor_block, selfplay_block, train_block


def _gate(**overrides: Any) -> dict:
    base = dict(
        stride=1, screen_games=80, confirm_games=128, promotion_winrate=0.55,
        screen_confirm_lo=0.44, deploy_sims=150, opening_book="book_v1_s20260625_p4",
        bootstrap_resamples=1000, min_distinct_per_pair=10, seed_base=20260625, sequential=None,
    )
    base.update(overrides)
    return base


def _payload(**eval_overrides: Any) -> dict:
    eval_block = dict(
        random_model_sims=96, random_floor_games=4, worker_device="cuda",
        round_timeout_sec=3600.0, worker_kill_grace_sec=10.0, gate=_gate(),
        ply_cap_adjudication=None, strength_floor=None, max_plies=128,
    )
    eval_block.update(eval_overrides)
    return {
        "schema_version": SCHEMA_VERSION,
        "eval_enabled": True,
        # A REQUIRED top-level leaf; `null` is the placeholder, refused at boot on a cuda process.
        "allocator_posture": None,
        "run_id": "unit_test",
        "seed": 1,
        "identity": {"encoding": "gnn_axis_v1", "representation": "graph"},
        "model": {"gnn": {"hidden": 128, "num_layers": 4}, "aux_soft_policy": None},
        "eval": eval_block,
        "train": train_block(),
        "deploy": {"search": {"kind": "puct"}},
        "selfplay": selfplay_block(),
        "inference": inference_block(),
        "monitor": monitor_block(),
    }


def _set_path(payload: dict, path: "tuple[str, ...]", value: Any) -> dict:
    """Deep-set a value at `path`, which is relative to `eval`."""
    payload = copy.deepcopy(payload)
    node = payload["eval"]
    for key in path[:-1]:
        node = node[key]
    last = path[-1]
    node[last] = value
    return payload


def _validate(payload: dict) -> RunConfig:
    return RunConfig.model_validate(payload)


def test_random_model_sims_negative_is_rejected_not_silently_loaded() -> None:
    """`eval.random_model_sims = -5` previously loaded with zero error."""
    payload = _payload(random_model_sims=-5)
    with pytest.raises(ValidationError) as ei:
        _validate(payload)
    assert "random_model_sims" in str(ei.value)


def test_promotion_winrate_above_one_is_rejected_not_silently_loaded() -> None:
    """`promotion_winrate = 2.0` loaded clean and disabled promotion forever."""
    payload = _payload()
    payload["eval"]["gate"]["promotion_winrate"] = 2.0
    with pytest.raises(ValidationError) as ei:
        _validate(payload)
    assert "promotion_winrate" in str(ei.value)


# Every bounded numeric field, out-of-domain + in-domain. Paths are relative to `eval`.
_OUT_OF_DOMAIN_CASES = [
    # EvalConfig
    (("random_model_sims",), 0, "eval.random_model_sims"),
    (("random_model_sims",), -5, "eval.random_model_sims"),
    (("random_floor_games",), -1, "eval.random_floor_games"),
    (("round_timeout_sec",), 0.0, "eval.round_timeout_sec"),
    (("round_timeout_sec",), -1.0, "eval.round_timeout_sec"),
    (("worker_kill_grace_sec",), -1.0, "eval.worker_kill_grace_sec"),
    # Non-finite and above-ceiling on the two timeout fields, plus the third floor-only float.
    (("round_timeout_sec",), float("inf"), "eval.round_timeout_sec"),
    (("round_timeout_sec",), float("-inf"), "eval.round_timeout_sec"),
    (("round_timeout_sec",), float("nan"), "eval.round_timeout_sec"),
    (("round_timeout_sec",), _EVAL_TIMEOUT_CEILING_SEC + 1.0, "eval.round_timeout_sec"),
    (("worker_kill_grace_sec",), float("inf"), "eval.worker_kill_grace_sec"),
    (("worker_kill_grace_sec",), float("-inf"), "eval.worker_kill_grace_sec"),
    (("worker_kill_grace_sec",), float("nan"), "eval.worker_kill_grace_sec"),
    (("worker_kill_grace_sec",), _EVAL_TIMEOUT_CEILING_SEC + 1.0, "eval.worker_kill_grace_sec"),
    # GateConfig
    (("gate", "stride"), 0, "eval.gate.stride"),
    (("gate", "screen_games"), 0, "eval.gate.screen_games"),
    (("gate", "confirm_games"), 0, "eval.gate.confirm_games"),
    (("gate", "promotion_winrate"), 2.0, "eval.gate.promotion_winrate"),
    (("gate", "promotion_winrate"), -0.1, "eval.gate.promotion_winrate"),
    (("gate", "screen_confirm_lo"), 1.1, "eval.gate.screen_confirm_lo"),
    (("gate", "screen_confirm_lo"), -0.1, "eval.gate.screen_confirm_lo"),
    (("gate", "deploy_sims"), 0, "eval.gate.deploy_sims"),
    (("gate", "bootstrap_resamples"), 0, "eval.gate.bootstrap_resamples"),
    (("gate", "min_distinct_per_pair"), 0, "eval.gate.min_distinct_per_pair"),
    # LadderConfig
]


@pytest.mark.parametrize("path,bad_value,field_hint", _OUT_OF_DOMAIN_CASES,
                        ids=[f"{'.'.join(p)}={v!r}" for p, v, _ in _OUT_OF_DOMAIN_CASES])
def test_out_of_domain_value_raises_named_validation_error(
    path: "tuple[str, ...]", bad_value: Any, field_hint: str,
) -> None:
    payload = _set_path(_payload(), path, bad_value)
    with pytest.raises(ValidationError) as ei:
        _validate(payload)
    field_name = path[-1]
    assert field_name in str(ei.value), (
        f"expected a named ValidationError mentioning {field_name!r} for {field_hint}="
        f"{bad_value!r}, got: {ei.value}"
    )


_IN_DOMAIN_BOUNDARY_CASES = [
    (("random_model_sims",), 1),
    (("random_floor_games",), 0),
    (("round_timeout_sec",), 0.001),
    (("worker_kill_grace_sec",), 0.0),
    # The exact ceiling must still load: the fix rejects only non-finite/above-ceiling values.
    (("round_timeout_sec",), _EVAL_TIMEOUT_CEILING_SEC),
    (("worker_kill_grace_sec",), _EVAL_TIMEOUT_CEILING_SEC),
    (("gate", "stride"), 1),
    (("gate", "screen_games"), 1),
    (("gate", "confirm_games"), 1),
    (("gate", "promotion_winrate"), 0.0),
    (("gate", "promotion_winrate"), 1.0),
    (("gate", "screen_confirm_lo"), 0.0),
    (("gate", "screen_confirm_lo"), 1.0),
    (("gate", "deploy_sims"), 1),
    (("gate", "bootstrap_resamples"), 1),
    (("gate", "min_distinct_per_pair"), 1),
]


@pytest.mark.parametrize("path,value", _IN_DOMAIN_BOUNDARY_CASES,
                        ids=[f"{'.'.join(p)}={v!r}" for p, v in _IN_DOMAIN_BOUNDARY_CASES])
def test_in_domain_boundary_value_loads_clean(path: "tuple[str, ...]", value: Any) -> None:
    payload = _set_path(_payload(), path, value)
    _validate(payload)  # must not raise


def test_valid_payload_still_loads_after_bounds_added() -> None:
    """Sanity anchor: the bounds must never reject a legitimate, already-shipped config shape."""
    cfg = RunConfig.model_validate(_payload())
    assert cfg.eval.gate.promotion_winrate == 0.55


    # The ORIGINAL repro shape: a genuine YAML document, not a hand-constructed Python float.
def _yaml_doc_with_eval_override(field: str, yaml_literal: str) -> dict:
    """A full payload whose `eval.<field>` is parsed from a REAL YAML literal, not `float(...)`."""
    doc_text = f"eval_override_value: {yaml_literal}\n"
    parsed_value = yaml.safe_load(doc_text)["eval_override_value"]
    payload = _payload()
    payload["eval"][field] = parsed_value
    return payload


@pytest.mark.parametrize(
    "field,yaml_literal",
    [
        ("worker_kill_grace_sec", ".inf"),
        ("worker_kill_grace_sec", "-.inf"),
        ("worker_kill_grace_sec", ".nan"),
        ("round_timeout_sec", ".inf"),
        ("round_timeout_sec", "-.inf"),
        ("round_timeout_sec", ".nan"),
    ],
    ids=["worker_kill_grace_sec=.inf", "worker_kill_grace_sec=-.inf", "worker_kill_grace_sec=.nan",
         "round_timeout_sec=.inf", "round_timeout_sec=-.inf", "round_timeout_sec=.nan"],
)
def test_original_f_rt2_1_repro_real_yaml_document_now_rejected(field: str, yaml_literal: str) -> None:
    """The exact reproduction: `worker_kill_grace_sec: .inf` as a genuine YAML document loaded
    SILENTLY and killed a poller through `Process.join(float('inf'))`."""
    payload = _yaml_doc_with_eval_override(field, yaml_literal)
    assert not math.isfinite(payload["eval"][field])  # confirm the injected value IS non-finite
    with pytest.raises(ValidationError) as ei:
        _validate(payload)
    assert field in str(ei.value), (
        f"expected a named ValidationError mentioning {field!r} for the real YAML literal "
        f"{yaml_literal!r}, got: {ei.value}"
    )


def test_original_f_rt2_1_repro_ceiling_boundary_still_loads() -> None:
    """The non-`.inf` half of the repro: a grace at the finite ceiling is legitimate."""
    payload = _payload()
    payload["eval"]["worker_kill_grace_sec"] = _EVAL_TIMEOUT_CEILING_SEC
    cfg = _validate(payload)
    assert cfg.eval.worker_kill_grace_sec == _EVAL_TIMEOUT_CEILING_SEC


def test_max_plies_is_required_and_positive() -> None:
    """`eval.max_plies` (2026-09-15): every eval game's ply cap, its OWN row; absent is an error naming it, 0 refused."""
    payload = _payload()
    del payload["eval"]["max_plies"]
    with pytest.raises(ValidationError, match="max_plies"):
        _validate(payload)
    with pytest.raises(ValidationError, match="max_plies"):
        _validate(_payload(max_plies=0))
    assert _validate(_payload(max_plies=128)).eval.max_plies == 128


def test_max_plies_above_the_engines_stone_ceiling_is_refused_on_the_graph_path() -> None:
    """An eval cap above the engine's `max_stones()` is refused at mint, like the self-play cap."""
    from mantis._engine import max_stones

    with pytest.raises(ValidationError, match="eval.max_plies"):
        _validate(_payload(max_plies=max_stones() + 1))
    assert _validate(_payload(max_plies=max_stones())).eval.max_plies == max_stones()
