# >300 justify (R8): one function's oracle — every row drives `_select_buffer`, so its routing
# and its seeding stay one diff to read.
"""Oracle for `mantis.run._select_buffer`: representation routing, its named refusal, and
sampler seeding.

The unknown-representation arm uses a `SimpleNamespace` because a validated `RunConfig` cannot
carry one: `Literal["grid", "graph"]` plus the registry cross-check make it unrepresentable.
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from mantis.run import _select_buffer
from mantis.train.coordinator.dispatch import RepresentationRouteError

_CAPACITY = 64


def _derived(config) -> int:
    """Return the capacity the ONE formula derives for this config, through the bridge export."""
    from mantis._engine import derived_hexg_visit_capacity

    sp = config.selfplay
    pc = sp.playout_cap
    return derived_hexg_visit_capacity(
        n_simulations=sp.mcts.n_simulations,
        standard_sims=pc.standard_sims,
        fast_prob=pc.fast_prob,
        fast_sims=pc.fast_sims,
        full_search_prob=pc.full_search_prob,
        n_sims_quick=pc.n_sims_quick,
        n_sims_full=pc.n_sims_full,
        leaf_batch_size=sp.leaf_batch_size,
        gumbel_m=sp.gumbel_m,
        search_kind=config.search.kind,
    )


def test_a_graph_config_selects_the_graph_buffer(smoke_run_config) -> None:
    """A graph representation selects the REAL `HexgBuffer`, carrying the declared encoding.

    MUTATION THAT REDS IT: return a `ReplayBuffer` on both arms — a token census cannot see it.
    """
    from mantis._engine import HexgBuffer

    config = smoke_run_config("smoke_preflight_armed.yaml")
    buffer = _select_buffer(config, _CAPACITY)
    assert isinstance(buffer, HexgBuffer), (
        f"a graph run gets the graph buffer, off the DECLARATION and nothing else; got "
        f"{type(buffer).__name__}"
    )
    # The dense arm and its seeding twin are gone; the surviving routing claim is the refusal
    # below — an unknown representation never falls through to the one remaining arm.


@pytest.mark.parametrize("representation", ["", "dense", "GRAPH", None])
def test_an_unknown_representation_is_a_named_route_error_that_quotes_law_11(
    representation,
) -> None:
    """An unknown or absent representation is a NAMED route error, never a silent default.

    The message assertions are part of the oracle: the refusal must name the law and quote the
    refused value, or a boot failure cannot tell a typo from a missing key.

    MUTATION THAT REDS IT: a silent `ReplayBuffer` default.
    """
    config = SimpleNamespace(
        identity=SimpleNamespace(representation=representation, encoding="gnn_axis_v1"))
    with pytest.raises(RepresentationRouteError) as exc_info:
        _select_buffer(config, _CAPACITY)
    message = str(exc_info.value)
    assert "LAW-11" in message, (
        f"the refusal must cite the law it enforces; got {message!r}"
    )
    assert repr(representation) in message or str(representation) in message, (
        f"…and quote the value it refused; got {message!r}"
    )


def test_the_route_error_carries_no_tool_exit_code(smoke_run_config) -> None:
    """The route error is a `TypeError` subclass and carries no CI-tool exit code.

    MUTATION THAT REDS IT: give `RepresentationRouteError` an `rc`, or map it onto
    `PreflightConfigError(rc=10)`.
    """
    config = SimpleNamespace(
        identity=SimpleNamespace(representation="widened_later", encoding="gnn_axis_v1"))
    with pytest.raises(RepresentationRouteError) as exc_info:
        _select_buffer(config, _CAPACITY)
    assert isinstance(exc_info.value, TypeError), (
        "the route error stays a TypeError subclass — a wiring error, not a data error"
    )
    assert not hasattr(exc_info.value, "rc"), (
        "a `src/` exception must not carry a CI-tool exit code: that layering is the defect "
        "this WP ends, and the taxonomy the deleted predicate asserted lives in the CLASS"
    )
    source = inspect.getsource(_select_buffer)
    assert "PreflightConfigError" not in source, (
        "the lifted selector must not import or raise the tool's error class — the lift is "
        "out of `tools/`, not a copy of it"
    )


def test_the_graph_buffer_is_composed_with_the_derived_visit_capacity(
    smoke_run_config,
) -> None:
    """The composed buffer's visit capacity is the DERIVATION's output, on both sims regimes.

    It compares against the bridge-exported derivation rather than a hand-typed number, which
    would silently encode one minted sims regime.

    MUTATION THAT REDS IT: compose `HexgBuffer` with any fixed capacity.
    """
    # Under the minted Gumbel kind a sparse row's capacity is m whatever the sims, so the two
    # shapes that must DIFFER drive the full-vector puct kind; the minted config is the third.
    full_vector = {"search": {"kind": "puct"}, "train": {"policy_target": "raw_visit_distribution"}}
    pcr = smoke_run_config(
        "run6.yaml",
        selfplay={
            "playout_cap": {
                "full_search_prob": 0.10,
                "n_sims_quick": 75,
                "n_sims_full": 600,
            }
        },
        **full_vector,
    )
    assert _select_buffer(pcr, _CAPACITY).visit_capacity == _derived(pcr)
    puct_minted_sims = smoke_run_config("run6.yaml", **full_vector)
    assert _select_buffer(puct_minted_sims, _CAPACITY).visit_capacity == _derived(puct_minted_sims)
    assert _derived(pcr) != _derived(puct_minted_sims), (
        "the two sims regimes now derive the same capacity, so this test can no longer tell a "
        "derivation from a constant — the whole point of driving both shapes"
    )

    minted = smoke_run_config("run6.yaml")
    assert _select_buffer(minted, _CAPACITY).visit_capacity == _derived(minted)


def _fill_graph_ring(buffer, n_records: int = 32) -> None:
    """Push `n_records` positions with UNIQUE outcomes, so a sample's outcome vector is a
    faithful transcript of which slots it drew and in what order.
    """
    for i in range(n_records):
        n_stones = 6 + (i % 7)
        stones = [(q, (q % 3) - 1, 1 if q % 2 == 0 else -1) for q in range(n_stones)]
        buffer.push_graph_position(
            stones, [(-1, 0, 0.6), (n_stones, 0, 0.4)],
            1 if i % 2 == 0 else -1, 2, i % 50, True,
            -1.0 + 2.0 * i / (n_records - 1), True, 40, 10 + i,
        )


def test_the_graph_arm_seeds_its_sampler_from_config_seed(smoke_run_config) -> None:
    """Two rings built from the SAME config draw the SAME batch.

    The ring's sampler is a Rust `StdRng` seeded from OS entropy at construction, so without
    the explicit seeding two launches of one config cannot agree.

    MUTATION THAT REDS IT: delete `buffer.seed_sampler(config.seed)` from the graph arm — the
    buffers still construct, still sample, and disagree.
    """
    config = smoke_run_config("smoke_preflight_armed.yaml")

    first = _select_buffer(config, _CAPACITY)
    _fill_graph_ring(first)
    second = _select_buffer(config, _CAPACITY)
    _fill_graph_ring(second)

    _, targets_a = first.sample_graph_batch(24)
    _, targets_b = second.sample_graph_batch(24)
    assert list(targets_a.outcomes) == list(targets_b.outcomes), (
        "two rings built from one config drew different batches — the sampler is not being "
        "seeded from config.seed, so the run is not reproducible across launches"
    )


def test_a_different_config_seed_moves_the_graph_draw(smoke_run_config) -> None:
    """The control for arm 1: a different `config.seed` moves the draw.

    MUTATION THAT REDS IT: seed from a literal instead of `config.seed`.
    """
    config = smoke_run_config("smoke_preflight_armed.yaml")
    other = smoke_run_config("smoke_preflight_armed.yaml", seed=config.seed + 1)

    baseline = _select_buffer(config, _CAPACITY)
    _fill_graph_ring(baseline)
    moved = _select_buffer(other, _CAPACITY)
    _fill_graph_ring(moved)

    assert list(baseline.sample_graph_batch(24)[1].outcomes) != list(
        moved.sample_graph_batch(24)[1].outcomes
    ), "changing config.seed did not change the batch stream"
