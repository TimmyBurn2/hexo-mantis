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
    """R255: 'derived at composition time from the configured sims regime'. The
    composed buffer's slot geometry must be the derivation's output — under the
    600/75 PCR shape that is max(50, 75, 600) + 8 − 1 = 607, and under the minted
    run5 shape 50 + 8 − 1 = 57. A literal anywhere on this path reds one of the two.

    MUTATION THAT REDS IT: compose `HexgBuffer` with any fixed capacity (the old
    128, or a new constant) instead of calling the derivation."""
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
    assert _select_buffer(pcr, _CAPACITY).visit_capacity == 607

    minted = smoke_run_config("run5.yaml")
    assert _select_buffer(minted, _CAPACITY).visit_capacity == 57
