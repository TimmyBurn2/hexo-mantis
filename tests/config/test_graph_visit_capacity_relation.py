"""R255/ADJ-D34 — the graph sims regime must fit the HEXG record format AT MINT.

The Phase-T guard's capacity is DERIVED from the configured sims regime
(max over armed PCR arms + ``leaf_batch_size`` − 1) by the ONE authority
``mantis._engine.derived_hexg_visit_capacity`` (Rust: ``replay::hexg``), and the
schema validates the relation explicitly: a regime the record format cannot
honor (derived capacity past the ``u16`` visit-count ceiling, 65535) REDs at
config validation — mint time — never as a boot surprise. ADJ-D34's defect was
the inversion: a ``MAX_VISITS = 128`` literal made the prereg'd PCR 600/75 row
un-bootable while every config validated clean.

The dispatch's two pins live here: (1) a 600/75-shaped config validates clean;
(2) the mutation — a regime exceeding what any capacity can honor is refused by
the SCHEMA, with the boot guard demoted to defense-in-depth (its own pin is
Rust-side, ``target_boot_guards.rs``).
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
    """The SIMS-REGIME prereg row (R160/R163/R165) must be mintable: 600/75 on the
    graph arm validates — the exact shape ADJ-D34 measured as un-bootable."""
    config = smoke_run_config("run5.yaml", selfplay=_PCR_600_75)
    assert config.selfplay.playout_cap.n_sims_full == 600


def test_a_regime_over_the_record_format_ceiling_reds_at_mint(smoke_run_config) -> None:
    """The dispatch's mutation pin: no capacity can honor a 70_000-sim arm (the
    per-record visit count is u16), so validation itself refuses — naming the
    ceiling and saying it is a mint-time error."""
    with pytest.raises(ValidationError, match="65535"):
        smoke_run_config(
            "run5.yaml",
            selfplay={
                "playout_cap": {
                    "full_search_prob": 0.10,
                    "n_sims_quick": 75,
                    "n_sims_full": 70_000,
                }
            },
        )


def test_the_refusal_names_the_governing_config_keys(smoke_run_config) -> None:
    """An operator reading the mint error must be able to act on it: the message
    names the sims-regime keys the capacity is derived from."""
    with pytest.raises(ValidationError, match="leaf_batch_size"):
        smoke_run_config(
            "run5.yaml",
            selfplay={
                "playout_cap": {
                    "full_search_prob": 0.10,
                    "n_sims_quick": 75,
                    "n_sims_full": 70_000,
                }
            },
        )


def test_the_relation_is_graph_scoped(smoke_run_config) -> None:
    """Dense-362 records carry no HEXG visit slot: the SAME wild regime on a grid
    config validates clean (the relation attaches to the record format, not to
    the sims keys per se — R250's absence principle, mint-side)."""
    config = smoke_run_config(
        "sustained_kcluster.yaml",
        selfplay={
            "playout_cap": {
                "full_search_prob": 0.10,
                "n_sims_quick": 75,
                "n_sims_full": 70_000,
            }
        },
    )
    assert config.identity.representation == "grid"


def test_every_minted_graph_config_satisfies_the_relation(smoke_run_config) -> None:
    """Gate-7 invariant, asserted here so a future re-mint cannot regress it
    silently: all shipped graph configs pass the derivation (their regimes are
    50-sims/leaf-8 → capacity 57). `shakedown_20260807.yaml` joins at F-P2B (R259) —
    same regime as run5, and it is the config the R255 relation actually gates next."""
    for name in ("run5.yaml", "shakedown_20260807.yaml", "smoke_gnn.yaml", "dev_example.yaml"):
        config = smoke_run_config(name)
        assert config.identity.representation == "graph"
