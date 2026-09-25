"""⊕ Arena — RegimeKey construction (A3; design §a.2, §b arena/test_regime_key.py).

Every eval game record carries a
canonical RegimeKey = (bot, variant, model_sims, opponent_spec, opening_book,
deploy_matched, encoding); the aggregator (tests/eval/test_aggregate_regime.py) raises on
a mixed set of these keys. This suite pins construction + the canonical form only.
"""
from __future__ import annotations

import pytest

from mantis.arena.regime import RegimeKey


def _key(**overrides) -> RegimeKey:
    base = dict(
        bot="sealbot",
        variant="d5",
        model_sims=150,
        opponent_spec="sealbot:depth=5",
        opening_book="book_v1_s20260625_p4",
        deploy_matched=True,
        encoding="gnn_axis_v1",
    )
    base.update(overrides)
    return RegimeKey(**base)


def test_canonical_is_the_field_tuple_joined_in_order():
    assert _key().canonical() == "sealbot|d5|150|sealbot:depth=5|book_v1_s20260625_p4|1|gnn_axis_v1"
    with pytest.raises(ValueError, match="separator"):
        _key(variant="d|5").canonical()


def test_any_field_change_changes_key():
    base = _key()
    base_canonical = base.canonical()
    field_overrides = [
        {"variant": "d6"},
        {"model_sims": 128},
        {"opponent_spec": "sealbot:depth=6"},
        {"opening_book": "book_v2"},
        {"deploy_matched": False},
        {"encoding": "gnn_axis_r8"},
    ]
    for overrides in field_overrides:
        changed = _key(**overrides)
        assert changed != base, f"changing {overrides} must change the RegimeKey"
        assert changed.canonical() != base_canonical, (
            f"changing {overrides} must change the canonical form"
        )
