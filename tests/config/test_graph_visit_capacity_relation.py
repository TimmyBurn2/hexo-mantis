"""The graph sims regime must fit the HEXG record format at mint, not at boot.

Visit capacity is derived from the sims regime (max over armed PCR arms + ``leaf_batch_size``
− 1) by the one authority ``mantis._engine.derived_hexg_visit_capacity``, and the schema
validates the relation: a derived capacity past the ``u16`` visit-count ceiling of 65535 reds
at config validation. The boot guard is defense-in-depth, pinned Rust-side in
``target_boot_guards.rs``.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

_PCR_600_75 = {
    "playout_cap": {
        "full_search_prob": 0.10,
        "n_sims_quick": 75,
        "n_sims_full": 600,
    }
}


def test_the_600_75_prereg_shape_validates_clean(smoke_run_config) -> None:
    """Prove a 600/75 sims regime on the graph arm validates clean."""
    config = smoke_run_config("run6.yaml", selfplay=_PCR_600_75)
    assert config.selfplay.playout_cap.n_sims_full == 600


#: A leaf-batch wide enough to push the derived capacity past the u16 ceiling on its own.
#: The sims axis cannot be used: the tighter `MAX_ARMED_SIMS` pool bound refuses a wild
#: `n_sims_full` first. `leaf_batch_size` carries no such bound, so the ceiling stays reachable.
_LEAF_BATCH_OVER_THE_CEILING = 70_000


def test_a_regime_over_the_record_format_ceiling_reds_at_mint(smoke_run_config) -> None:
    """Prove a regime no capacity can honor is refused by validation, naming the ceiling."""
    with pytest.raises(ValidationError, match="65535"):
        smoke_run_config(
            "run6.yaml",
            selfplay={
                "playout_cap": {
                    "full_search_prob": 0.10,
                    "n_sims_quick": 75,
                    "n_sims_full": 600,
                },
                "leaf_batch_size": _LEAF_BATCH_OVER_THE_CEILING,
            },
        )


def test_the_sims_axis_is_bounded_EARLIER_by_the_node_pool(smoke_run_config) -> None:
    """Prove a wild sims regime reds against the node-pool bound, not the record format."""
    with pytest.raises(ValidationError) as excinfo:
        smoke_run_config(
            "run6.yaml",
            selfplay={
                "playout_cap": {
                    "full_search_prob": 0.10,
                    "n_sims_quick": 75,
                    "n_sims_full": 70_000,
                }
            },
        )
    assert "n_sims_full" in str(excinfo.value)


def test_the_refusal_names_the_governing_config_keys(smoke_run_config) -> None:
    """Prove the refusal names the config keys the capacity is derived from."""
    with pytest.raises(ValidationError, match="leaf_batch_size"):
        smoke_run_config(
            "run6.yaml",
            selfplay={
                "playout_cap": {
                    "full_search_prob": 0.10,
                    "n_sims_quick": 75,
                    "n_sims_full": 600,
                },
                "leaf_batch_size": _LEAF_BATCH_OVER_THE_CEILING,
            },
        )


def test_the_relation_has_no_grid_arm_left_to_be_scoped_against() -> None:
    """Prove the registry has no non-graph representation, so the relation is unconditional.

    The scoping half retired with the grid path; a re-introduced representation reds here
    rather than silently re-opening an unscoped arm.
    """
    from mantis.encoding import all_specs

    reps = {str(spec.representation) for spec in all_specs()}
    assert reps == {"graph"}, (
        f"a non-graph representation is registered again ({sorted(reps)}); the visit-capacity "
        "relation was SCOPED to graph and that scoping was deleted with the grid path"
    )

def test_every_minted_graph_config_satisfies_the_relation(smoke_run_config) -> None:
    """Prove every shipped graph config satisfies the derivation (50-sims/leaf-8 -> capacity 57)."""
    for name in ("run6.yaml", "run6.yaml", "smoke_preflight_armed.yaml", "dev_example.yaml"):
        config = smoke_run_config(name)
        assert config.identity.representation == "graph"
