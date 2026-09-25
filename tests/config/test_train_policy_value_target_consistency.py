"""RunConfig cross-section validator: `train.policy_target` must name the target `selfplay.search.kind` produces.

The rule replaced a three-way agreement between `policy_target` and a `completed_q_values`
boolean on each of two sections, so a disagreement is now expressible in exactly one shape
instead of seven. The surviving key is the one a CHECKPOINT STAMP carries, so a resume still has
something to compare a restored ring against.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from mantis.config.schema import RunConfig, SCHEMA_VERSION
from _schema_blocks import eval_block, inference_block, monitor_block, train_block


def _selfplay_block(*, n_simulations: int = 50) -> dict:
    return {
        "n_workers": 1, "leaf_batch_size": 8, "max_game_moves": 128,
        "c_visit": 50.0, "c_scale": 1.0, "q_rescale": True, "gumbel_m": 16,
        "gumbel_explore_moves": 10, "search_stats_every": 8,
        "results_queue_cap": 10_000, "random_opening_plies": 0,
        "log_investigation_metrics": True,
        "mcts": {"n_simulations": n_simulations, "c_puct": 1.5, "fpu_reduction": 0.25,
                 "quiescence_enabled": True, "quiescence_blend_2": 0.3,
                 "dirichlet_alpha": 0.3, "dirichlet_epsilon": 0.25, "dirichlet_enabled": True},
        "playout_cap": {"fast_sims": 50, "fast_prob": 0.0, "standard_sims": 0,
                        "full_search_prob": 0.0, "n_sims_quick": 0, "n_sims_full": 0,
                        "temperature_threshold_compound_moves": 0, "temp_min": 0.5},
    }


def _payload(
    *,
    train_over: dict | None = None,
    search_kind: str = "puct",
    n_simulations: int = 50,
    selfplay_over: dict | None = None,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "eval_enabled": True,
        # A REQUIRED top-level leaf; `null` is the placeholder, refused at boot on a cuda process.
        "allocator_posture": None,
        "run_id": "unit_test",
        "seed": 1,
        "identity": {"encoding": "gnn_axis_v1", "representation": "graph"},
        "model": {"gnn": {"hidden": 128, "num_layers": 4}, "aux_soft_policy": None},
        "deploy": {"search": {"kind": search_kind}},
        "eval": eval_block(),
        "train": train_block(**(train_over or {})),
        "selfplay": {**_selfplay_block(n_simulations=n_simulations), "search": {"kind": search_kind},
                     **(selfplay_over or {})},
        "inference": inference_block(),
        "monitor": monitor_block(),
    }


def test_the_shipped_combo_constructs_cleanly():
    cfg = RunConfig.model_validate(_payload())
    assert cfg.selfplay.search.kind == "puct"
    assert cfg.train.policy_target == "raw_visit_distribution"


def test_the_completed_target_under_puct_is_refused():
    """A PUCT search exports the visit distribution; declaring the completed target trains
    the KL loss on visit-count rows."""
    with pytest.raises(ValidationError, match="policy_target"):
        RunConfig.model_validate(
            _payload(train_over={"policy_target": "completed_improved_policy"})
        )


def test_the_raw_target_under_gumbel_is_refused():
    """A Gumbel search exports the completed-Q improved policy, so scoring it as a visit
    distribution applies the wrong loss to every row."""
    # Exercised on the graph identity, which is legal under `gumbel`: the sparse row carries the
    # tail mass, so the record-format refusal that used to fire first is gone.
    payload = _payload(search_kind="gumbel")
    with pytest.raises(ValidationError, match="policy_target"):
        RunConfig.model_validate(payload)


def test_the_gumbel_kind_and_the_completed_target_agree():
    payload = _payload(
        search_kind="gumbel",
        train_over={"policy_target": "completed_improved_policy"},
    )
    cfg = RunConfig.model_validate(payload)
    assert cfg.selfplay.search.kind == "gumbel"
    assert cfg.train.policy_target == "completed_improved_policy"


def test_the_gumbel_kind_on_a_graph_run_mints_at_the_minted_slot_bound():
    """A graph run under `gumbel` mints at a slot bound of m, not the sims regime.

    Only `selfplay.gumbel_m` candidates are ever visited; every other legal action's target is the
    recording prior times one scalar, so the row stores m entries plus that scalar.
    """
    cfg = RunConfig.model_validate(
        _payload(
            search_kind="gumbel",
            train_over={"policy_target": "completed_improved_policy"},
            n_simulations=192,
        )
    )
    assert cfg.selfplay.search.kind == "gumbel"
    assert cfg.identity.representation == "graph"


def test_a_gumbel_m_past_the_minted_bound_is_refused_on_a_graph_run():
    """The bound is m, not the sims regime."""
    with pytest.raises(ValidationError, match="gumbel_m"):
        RunConfig.model_validate(
            _payload(
                search_kind="gumbel",
                train_over={"policy_target": "completed_improved_policy"},
                n_simulations=192,
                selfplay_over={"gumbel_m": 17},
            )
        )


def test_a_config_still_carrying_the_retired_value_target_is_refused():
    """`train.value_target` is RETIRED: a config carrying it fails `extra="forbid"`, only a stamp may."""
    with pytest.raises(ValidationError, match="value_target"):
        RunConfig.model_validate(_payload(train_over={"value_target": "pure_outcome_z"}))


def test_out_of_enum_policy_target_rejected_by_literal_before_cross_section_validator():
    with pytest.raises(ValidationError, match="policy_target"):
        RunConfig.model_validate(_payload(train_over={"policy_target": "completed_q_policy"}))
