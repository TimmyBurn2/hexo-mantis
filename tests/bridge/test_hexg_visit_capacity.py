"""R255/ADJ-D34 — producer + mutation self-tests for the capacity derivation's FFI face.

``mantis._engine.derived_hexg_visit_capacity`` is the schema validator's INPUT
(LAW-07: every gate input cites a live producer with a mutation self-test), and
it is the SAME Rust authority the boot guard calls — so these pins hold the two
surfaces to one formula. The ``HexgBuffer`` ctor leg pins that the composed
buffer's slot geometry really is the derived value, not a 128 literal: a
130-visit record — over the deleted cap — must survive push/sample intact.
"""
from __future__ import annotations

import pytest

from mantis import _engine
from mantis.encoding.registry import lookup


def _derive(**over):
    args = dict(
        n_simulations=50,
        standard_sims=0,
        fast_prob=0.0,
        fast_sims=50,
        full_search_prob=0.0,
        n_sims_quick=0,
        n_sims_full=0,
        leaf_batch_size=8,
        completed_q_values=False,
    )
    args.update(over)
    return _engine.derived_hexg_visit_capacity(**args)


def test_derivation_is_max_armed_plus_leaf_overshoot() -> None:
    assert _derive() == 57  # run5 shape: 50 + 8 - 1
    assert _derive(standard_sims=40) == 47  # standard_sims wins when set
    assert _derive(full_search_prob=0.10, n_sims_quick=75, n_sims_full=600) == 607
    assert _derive(fast_prob=0.5, fast_sims=500) == 507


def test_an_unarmed_arm_never_enters_the_max() -> None:
    assert _derive(fast_sims=500) == 57  # fast_prob == 0.0 → inert
    assert _derive(n_sims_quick=70_000, n_sims_full=70_000) == 57  # prob 0.0 → inert


def test_a_regime_over_the_ceiling_raises_naming_it() -> None:
    with pytest.raises(ValueError, match="65535"):
        _derive(full_search_prob=0.10, n_sims_quick=75, n_sims_full=70_000)


def test_completed_q_refusal_tracks_the_derived_capacity() -> None:
    # Below MAX_CHILDREN_PER_NODE (192): child-count-wide support cannot fit.
    with pytest.raises(ValueError, match="192"):
        _derive(completed_q_values=True)
    # A 600/75 regime derives 607 >= 192: the refusal would be vacuous — admits.
    assert (
        _derive(completed_q_values=True, full_search_prob=0.10, n_sims_quick=75, n_sims_full=600)
        == 607
    )


# ── the composed buffer's slots ARE the derived capacity ────────────────────────


def test_buffer_ctor_requires_an_explicit_visit_capacity() -> None:
    """No default (R255: 'no literal, no default') — the two-arg form is gone."""
    with pytest.raises(TypeError):
        _engine.HexgBuffer(8, "gnn_axis_v1")


def _hex_dist(q: int, r: int, q2: int, r2: int) -> int:
    dq, dr = q - q2, r - r2
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


def test_buffer_carries_and_honors_the_composed_capacity() -> None:
    hb = _engine.HexgBuffer(8, "gnn_axis_v1", 607)
    assert hb.visit_capacity == 607
    # 130 visit cells — over the deleted 128 literal — push AND sample intact.
    # The cells are drawn from the legal set (within `legal_move_radius` of a stone,
    # unoccupied). AUDIT-1 F-41: that radius was typed `<= 6` here under a comment naming the
    # registry as the authority, so the premise restated the very fact it cited. It is READ
    # off the row this buffer is built for, and the search box is derived from it, so a row
    # whose radius moves moves this test instead of leaving it asserting a stale disk.
    radius = lookup("gnn_axis_v1").legal_move_radius
    stones = [(0, 0, 1), (1, 0, -1)]
    occupied = {(0, 0), (1, 0)}
    cells = sorted(
        {
            (q, r)
            for q in range(-radius - 1, radius + 3)
            for r in range(-radius - 1, radius + 2)
            if (q, r) not in occupied
            and any(_hex_dist(q, r, sq, sr) <= radius for sq, sr in occupied)
        }
    )[:130]
    assert len(cells) == 130, "premise: the legal disk must hold 130 target cells"
    visits = [(q, r, 1.0 / 130.0) for q, r in cells]
    hb.push_graph_position(stones, visits, 1, 2, 0, True, 1.0, True, 4)
    assert hb.size == 1
    _wire, targets = hb.sample_graph_batch(1, False, 0.0)
    assert len(targets.policy_target) >= 130


def test_over_composed_capacity_push_dies_loud() -> None:
    hb = _engine.HexgBuffer(8, "gnn_axis_v1", 57)
    stones = [(0, 0, 1)]
    visits = [(2 + i, -i, 1.0 / 58.0) for i in range(58)]
    with pytest.raises(ValueError):
        hb.push_graph_position(stones, visits, 1, 2, 0, True, 1.0, True, 4)


def test_ctor_refuses_a_capacity_the_format_cannot_store() -> None:
    with pytest.raises(ValueError):
        _engine.HexgBuffer(8, "gnn_axis_v1", 0)
    with pytest.raises(ValueError):
        _engine.HexgBuffer(8, "gnn_axis_v1", 65_536)
