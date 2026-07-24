"""⊕ WP11-A arena — RegimeKey construction (A3; design §a.2, §b arena/test_regime_key.py).

RED-at-import until IMPL writes `mantis.arena.regime`. Every eval game record carries a
canonical RegimeKey = (bot, variant, model_sims, opponent_spec, opening_book,
deploy_matched, encoding); the aggregator (tests/eval/test_aggregate_regime.py) raises on
a mixed set of these keys. This suite pins construction + canonical round-trip only.
"""
from __future__ import annotations

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


def test_canonical_roundtrip():
    key = _key()
    canonical = key.canonical()
    assert isinstance(canonical, str)
    restored = RegimeKey.from_canonical(canonical)
    assert restored == key
    assert restored.canonical() == canonical


def test_any_field_change_changes_key():
    base = _key()
    base_canonical = base.canonical()
    field_overrides = [
        {"bot": "kraken"},
        {"variant": "d6"},
        {"model_sims": 128},
        {"opponent_spec": "sealbot:depth=6"},
        {"opening_book": "book_v2"},
        {"deploy_matched": False},
        {"encoding": "v6_live2_ls"},
    ]
    for overrides in field_overrides:
        changed = _key(**overrides)
        assert changed != base, f"changing {overrides} must change the RegimeKey"
        assert changed.canonical() != base_canonical, (
            f"changing {overrides} must change the canonical form"
        )


def test_key_equality_is_full_tuple():
    a = _key()
    b = _key()
    assert a == b
    # Equality must consider EVERY field, not a subset (e.g. not just bot+variant).
    c = _key(deploy_matched=False)
    assert a != c
    d = _key(model_sims=151)
    assert a != d
