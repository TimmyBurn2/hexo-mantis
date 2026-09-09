# >300 justify (R8): one function's oracle. Every row here drives `_select_buffer` — which
# buffer each representation selects, that an unknown one is a named refusal, and that both
# arms seed their sampler. Splitting the seeding rows out would put two claims about one
# function in two files, where a change to its routing and a change to its seeding stop
# being one diff to read.
"""⊕ WPMAIN ORACLE — the buffer selector at its new home (DESIGN §1.2 item 5 / §9 O-F1).

RED-at-import until IMPL lands `mantis.run._select_buffer` — the lift of
`preflight_mint.py:823-856`'s `_build_buffer` out of a CI GATE and into the composition root
(D-1/D-2, R121(a)).

Why the move matters beyond tidiness, measured: the raise this function owns sits at
`tools/`, and gate 11's `SCAN_ROOTS = ("src", "crates")` (`silent_encoding_gate.py:63`) does
not scan `tools/`. So the repo's own silent-encoding-fallback gate has never been able to
see the one LAW-11 raise on the buffer route. The lift brings it under the gate. (Gate 11
stays quiet on the moved code: its patterns all require a REGISTERED-ENCODING literal in a
default position, and the selector contains no encoding literal at all — it passes
`config.identity.encoding` affirmatively. Measured, DESIGN §1.2 item 5.)

The MF-4 drives (`tests/tools/test_preflight_mint_process.py:892-925`) are this oracle's
predecessors; DESIGN §4 re-points them here. One predicate does NOT come along: MF-4's
`assert caught.value.rc == 10`. `RepresentationRouteError` is a `TypeError` subclass
(`coordinator/dispatch.py:35`) and carries no `rc`, correctly — a `src/` error carrying a
CI-tool exit code is the layering defect this WP ends. R125 ruled the argued-deletion and
REJECTED the child-seam mapping alternative (a dead except-arm kept alive to feed a test,
R116). The last test below is the successor for what that predicate was really protecting:
the error must not acquire a tool taxonomy on its way into `src/`.

Fakes: one, enumerated. The unknown-representation arm uses a `SimpleNamespace` stand-in
because a validated `RunConfig` CANNOT carry an unknown representation — `Literal["grid",
"graph"]` (`schema/core.py:51`) plus the registry cross-check (`:54-62`) make it
unrepresentable. That measurement is R125's own grounds, and it is why the arm exists at all:
the route must stay loud for the day someone widens the enum.
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from mantis.run import _select_buffer  # RED-at-import anchor
from mantis.train.coordinator.dispatch import RepresentationRouteError

_CAPACITY = 64


def _derived(config) -> int:
    """The capacity the ONE formula derives for this config, through the bridge export.

    `mantis._engine.derived_hexg_visit_capacity` delegates VERBATIM to
    `mantis_selfplay::replay::hexg::derived_visit_capacity` — the same function the schema
    validator and the runner's boot guard call. Asking it here is the difference between
    "the composition called the derivation" and "the composition produced the number I typed".
    """
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
        search_kind=config.search.kind,
    )


def test_a_graph_config_selects_the_graph_buffer(smoke_run_config) -> None:
    """O-F1, arm 1. `identity.representation == "graph"` -> the REAL `HexgBuffer`, carrying
    the config's declared encoding.

    MUTATION THAT REDS IT: return a `ReplayBuffer` on both arms — the dense-by-default
    defect. It is invisible to a token census (all four class names still appear in the
    file), which is what the tree already MEASURED about O-9's instrument."""
    from mantis._engine import HexgBuffer

    config = smoke_run_config("smoke_gnn.yaml")
    buffer = _select_buffer(config, _CAPACITY)
    assert isinstance(buffer, HexgBuffer), (
        f"a graph run gets the graph buffer, off the DECLARATION and nothing else; got "
        f"{type(buffer).__name__}"
    )


def test_a_grid_config_selects_the_dense_buffer(smoke_run_config) -> None:
    """O-F1, arm 2. The other declared route, driven from a MINTED grid config
    (`smoke_radius_curriculum.yaml`) rather than a hand-built one — the axis is varied, not
    re-pinned.

    MUTATION THAT REDS IT: route both representations to `HexgBuffer` (the inverse of arm
    1's mutation, and equally invisible to a token census)."""
    from mantis._engine import ReplayBuffer

    config = smoke_run_config("smoke_radius_curriculum.yaml")
    buffer = _select_buffer(config, _CAPACITY)
    assert isinstance(buffer, ReplayBuffer), (
        f"a grid run gets the dense buffer; got {type(buffer).__name__}"
    )


@pytest.mark.parametrize("representation", ["", "dense", "GRAPH", None])
def test_an_unknown_representation_is_a_named_route_error_that_quotes_law_11(
    representation,
) -> None:
    """O-F1, arm 3 — the raise MF-I4 recorded as having NO producer.

    Its own docstring says so: O-9 asserted only that the TOKENS `HexgBuffer`,
    `ReplayBuffer`, `identity`, `representation` appear, and all four survive replacing this
    raise with a silent `ReplayBuffer` default — measured green at full tier (RR-12).

    MUTATION THAT REDS IT: exactly that silent default. The message assertions are part of
    the oracle, not decoration: an absent or unknown representation must be an ERROR that
    NAMES the law and QUOTES the refused value, or the operator reading a boot failure
    cannot tell a typo from a missing key.

    The error family is `RepresentationRouteError` — reused, not invented: the SAME axis
    already raises it for the train-step route, and `dispatch.py:9` cites "the `_build_buffer`
    posture" by name. One error family per axis (§1.2 item 5)."""
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
    """O-F1, arm 4 — the successor for MF-4's deleted `rc == 10` predicate (R125).

    What that predicate really protected was a taxonomy claim: the refusal is a distinct,
    identifiable failure and not an anonymous crash. What it did BADLY was express that as a
    CI tool's exit code living on a `src/` exception. So the claim is re-stated where it
    belongs: the error is the named route error, it is a `TypeError` (a wiring error, not a
    data error — `BufferKindMismatch`'s posture), and it carries no `rc` attribute.

    MUTATION THAT REDS IT: give `RepresentationRouteError` an `rc` — or re-introduce the
    child-seam mapping `RepresentationRouteError -> PreflightConfigError(rc=10)`, which R125
    rejected by name. Post-deletion posture, ruled: a widened representation enum makes this
    error propagate through the child as an UNCAUGHT loud failure with a full named
    traceback (LAW-14 fail-loud), never a silent arm.

    Rider recorded, and IMPL lands it in `_select_buffer`'s docstring so the next reader
    finds it in-tree: LAW-11 makes widening the representation enum a deliberate design act,
    and whoever widens it re-opens child-seam routing in that same design."""
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


# ── R255/ADJ-D34: the composed graph buffer carries the DERIVED visit capacity ──


def test_the_graph_buffer_is_composed_with_the_derived_visit_capacity(
    smoke_run_config,
) -> None:
    """R255: 'derived at composition time from the configured sims regime'. The composed
    buffer's slot geometry must be the DERIVATION's output, on both shapes.

    AUDIT-1 F-49: the two expectations were `== 607` and `== 57`, the formula's answers typed
    by hand under a docstring that spells the arithmetic out. `target_boot_guards.rs` pins the
    FORMULA; this file's job is that the composition CALLS it, so it compares against the one
    bridge-exported derivation rather than against a re-computed number. A hand-typed 57 also
    silently encodes run5's minted sims regime, which is the class-6 shape: re-mint and this
    test reds with "57 != N" and nothing says where 57 came from.

    MUTATION THAT REDS IT: compose `HexgBuffer` with any fixed capacity (the old 128, or a new
    constant) instead of calling the derivation."""
    pcr = smoke_run_config(
        "run5.yaml",
        selfplay={
            "playout_cap": {
                "full_search_prob": 0.10,
                "n_sims_quick": 75,
                "n_sims_full": 600,
            }
        },
    )
    assert _select_buffer(pcr, _CAPACITY).visit_capacity == _derived(pcr)

    minted = smoke_run_config("run5.yaml")
    assert _select_buffer(minted, _CAPACITY).visit_capacity == _derived(minted)
    assert _derived(pcr) != _derived(minted), (
        "the two sims regimes now derive the same capacity, so this test can no longer tell a "
        "derivation from a constant — the whole point of driving both shapes"
    )


def _fill_graph_ring(buffer, n_records: int = 32) -> None:
    """Push `n_records` distinguishable positions so a draw has a readable signature.

    Every record carries a UNIQUE `outcome`, which makes `GraphTargets.outcomes` a faithful
    transcript of WHICH slots a sample drew and in what order — the observable the seed is
    supposed to determine. A shared outcome value would collapse different draws onto equal
    vectors and the assertions below would pass on a broken sampler.
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
    """R344(a), arm 1. Two rings built from the SAME config draw the SAME batch.

    The ring's sampler is a Rust `StdRng` seeded from OS entropy at construction, so before
    this it was structurally impossible for two launches of one config to agree — which is
    the gap R343(c)'s determinism-seam witness uncovered and which no Python-side
    `seed_everything` could close.

    MUTATION THAT REDS IT: delete the `buffer.seed_sampler(config.seed)` line from the graph
    arm. The buffers still construct, still carry the right encoding, still sample — and
    disagree, which is exactly the silence this pins. A source grep for the call would not
    survive the line moving to a caller that forgets it; this asserts the BEHAVIOUR at the
    one construction site.
    """
    config = smoke_run_config("smoke_gnn.yaml")

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
    """R344(a), arm 2 — the control WITHOUT which arm 1 is vacuous.

    Arm 1 would pass on a sampler that ignored its seed and happened to be deterministic
    (a fixed constructor seed, say, or a draw that stopped consuming the generator). Only
    this arm shows the draw is a function OF `config.seed`.

    MUTATION THAT REDS IT: seed from a literal instead of `config.seed`."""
    config = smoke_run_config("smoke_gnn.yaml")
    other = smoke_run_config("smoke_gnn.yaml", seed=config.seed + 1)

    baseline = _select_buffer(config, _CAPACITY)
    _fill_graph_ring(baseline)
    moved = _select_buffer(other, _CAPACITY)
    _fill_graph_ring(moved)

    assert list(baseline.sample_graph_batch(24)[1].outcomes) != list(
        moved.sample_graph_batch(24)[1].outcomes
    ), "changing config.seed did not change the batch stream"


def _fill_dense_ring(buffer, encoding: str, n_records: int = 32) -> None:
    """Dense twin of `_fill_graph_ring`: unique `outcome` per record, geometry from the
    registry spec rather than from literals (the same authority the buffer itself was
    built through)."""
    import numpy as np

    from mantis._engine import RegistrySpec

    spec = RegistrySpec.from_registry(encoding)
    size = spec.board_size
    state = np.zeros((8, size, size), dtype=np.float16)
    chain = np.zeros((6, size, size), dtype=np.float16)
    policy = np.zeros(spec.policy_stride, dtype=np.float32)
    policy[0] = 1.0
    ownership = np.ones(spec.n_cells, dtype=np.uint8)
    winning_line = np.zeros(spec.n_cells, dtype=np.uint8)
    for i in range(n_records):
        buffer.push(state, chain, policy, -1.0 + 2.0 * i / (n_records - 1),
                    ownership, winning_line, 10 + i)


def test_the_dense_arm_seeds_its_sampler_too(smoke_run_config) -> None:
    """R344(a), arm 3. The grid route carries the same contract — asserted rather than
    assumed, because the two arms are two `return`s and a repair applied to only one of them
    is a shape this file already has precedent for (arms 1 and 2 above each pin one route
    because routing them both to one buffer type was the measured defect).

    `sample_batch`'s element 3 is `outcomes` (`SampleBatch`'s field order: states, chain,
    policies, outcomes, …), so it is the same draw transcript the graph arms read.

    MUTATION THAT REDS IT: delete the seeding line from the grid arm only."""
    config = smoke_run_config("smoke_radius_curriculum.yaml")
    encoding = config.identity.encoding

    first = _select_buffer(config, _CAPACITY)
    _fill_dense_ring(first, encoding)
    second = _select_buffer(config, _CAPACITY)
    _fill_dense_ring(second, encoding)

    assert list(first.sample_batch(24, False)[3]) == list(second.sample_batch(24, False)[3]), (
        "two dense rings built from one config drew different batches — the grid arm is not "
        "seeding its sampler"
    )
