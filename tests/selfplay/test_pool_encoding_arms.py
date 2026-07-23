"""Suite D (encoding arms) — D-02 … D-05, D-17 (`mantis.selfplay.hparams`).

IMPL-written (non-⊕) per DESIGN §b: D-01 is the ⊕ golden in `test_pool_encoding_resolve.py`
and this file carries the arms ORACLE-WRITE deliberately left to IMPL, whose capture rows
live in `wp/WPSP/oldside/data/c3a_c3d_report.json` (`C3a_resolve_encoding_for_pool`,
sections `v8` / `v8_canvas_realness` / `unregistered` / `_model_board_size_crosscheck`).

Two rows pin DECLARED deviations rather than old behaviour, and say so:

  * D-02 — the frozen resolver carried its own `v8` guard raising `NotImplementedError`
    BEFORE the registry was consulted. `v8` is not registered on this side at all, so the
    registry itself hard-errors first (DV-7). Loud failure is preserved; the class and the
    message change, and that is the point of the pin.
  * D-03/D-04 — the cross-check reads the DECLARED arch, never a live module attribute
    (DV-5). A graph arch declares no board size and therefore passes vacuously, which is
    exactly what the frozen `getattr(model, "board_size", spec.board_size)` did for a
    graph net.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from mantis.encoding import EncodingRegistryError, all_specs, lookup
from mantis.model import CnnArch, GnnArch, RepresentationMismatch
from mantis.selfplay.hparams import is_graph_representation, resolve_pool_encoding

# The v8 family: KILLed, and its guard died with the registry entry (DESIGN §e).
_V8_FAMILY = ("v8", "v8_canvas_realness")


@dataclass
class _SpecStub:
    """Spec-shaped stub carrying only what the representation dispatch reads."""

    representation: str
    name: str = "spec_stub"


# ── D-02 — an unregistered encoding fails LOUD, naming the registered set ─────────
@pytest.mark.parametrize("name", [*_V8_FAMILY, "definitely_not_registered"])
def test_unregistered_encoding_loud(name: str) -> None:
    """D-02 — PASS iff resolving an unregistered encoding raises `EncodingRegistryError`
    whose message names BOTH the rejected name and the registered set.

    DECLARED DEVIATION (DV-7): old-side the v8 family reached a hand-written
    `NotImplementedError` guard inside the resolver; new-side v8 is not in the registry,
    so `lookup` hard-errors one layer earlier. The behaviour that matters — self-play
    refuses to start on a v8 config, loudly — is preserved; the error class and text are
    not. FAIL = a silent fall-through to some default encoding, which would train a run
    against a wire format nobody asked for."""
    with pytest.raises(EncodingRegistryError) as exc:
        resolve_pool_encoding({"encoding": name}, arch=None)

    message = str(exc.value)
    assert name in message, "the rejected name must appear in the error"
    for registered in ("v6", "v6w25", "v6_live2_ls", "gnn_axis_v1"):
        assert registered in message, (
            f"the error must name the registered set (missing {registered!r}) — an "
            "operator reading it should not have to grep the registry"
        )


# ── D-03 / D-04 — the arch ↔ encoding canvas cross-check ─────────────────────────
@pytest.mark.parametrize(
    "encoding,arch_board,expected_spec_board",
    [("v6", 25, 19), ("v6w25", 19, 25)],
)
def test_arch_board_size_mismatch_raises(
    encoding: str, arch_board: int, expected_spec_board: int
) -> None:
    """D-03 — PASS iff a `CnnArch` whose declared board size disagrees with the resolved
    encoding raises `ValueError` citing BOTH numbers, before any Rust runner exists.
    FAIL = a mis-paired checkpoint and config route planes through wrong-shaped buffers
    and produce silently-wrong training data instead of a crash."""
    arch = CnnArch(board_size=arch_board, in_channels=8, filters=16, res_blocks=1)
    with pytest.raises(ValueError) as exc:
        resolve_pool_encoding({"encoding": encoding}, arch=arch)

    message = str(exc.value)
    assert str(arch_board) in message and str(expected_spec_board) in message, (
        f"both sizes must be in the message, got: {message}"
    )
    assert encoding in message


@pytest.mark.parametrize("encoding,arch_board", [("v6", 19), ("v6w25", 25)])
def test_arch_board_size_match_passes(encoding: str, arch_board: int) -> None:
    """D-04 — PASS iff a matching declared board size resolves cleanly to the captured
    canvas geometry. FAIL = the cross-check rejects a correct pairing (which would make
    the guard unusable and invite its removal)."""
    arch = CnnArch(board_size=arch_board, in_channels=8, filters=16, res_blocks=1)
    resolved = resolve_pool_encoding({"encoding": encoding}, arch=arch)
    assert resolved.board_size == arch_board
    assert resolved.encoding_name == encoding


def test_graph_arch_has_no_board_size_and_passes_vacuously() -> None:
    """D-04 (graph arm) — PASS iff a `GnnArch`, which declares NO board size, passes the
    cross-check for the graph encoding. This mirrors the frozen `getattr(..., default)`
    behaviour exactly. FAIL = the port invented a board-size requirement for graph nets,
    which would make every graph pool unconstructable."""
    spec = lookup("gnn_axis_v1")
    arch = GnnArch(in_dim=spec.node_feat_dim, edge_dim=spec.edge_feat_dim,
                   hidden=32, num_layers=2)
    assert not hasattr(arch, "board_size"), (
        "GnnArch must declare no board_size — the vacuous pass depends on it"
    )
    resolved = resolve_pool_encoding({"encoding": "gnn_axis_v1"}, arch=arch)
    assert resolved.encoding_name == "gnn_axis_v1"
    assert resolved.n_kept_planes == 0


# ── D-05 — the grid/graph classification covers the registry EXACTLY ─────────────
def test_every_registered_encoding_classified() -> None:
    """D-05 — PASS iff every spec in the registry classifies into exactly one of the two
    arms, and both arms are non-empty. FAIL = a registered encoding the pool can neither
    dispatch to the dense path nor the graph path — i.e. a spec that would reach a
    default arm. There is no default arm, so this is the test that keeps the closed set
    honest as the registry grows."""
    specs = list(all_specs())
    assert specs, "the registry must not be empty"

    graph, grid = [], []
    for spec in specs:
        (graph if is_graph_representation(spec) else grid).append(spec.name)

    assert len(graph) + len(grid) == len(specs), "a spec was classified twice or not at all"
    assert grid, "no grid encoding classified — the dense arm would be dead"
    assert graph, "no graph encoding classified — the graph arm would be dead"
    assert not set(graph) & set(grid)


@pytest.mark.parametrize("name", ["v6", "v6w25", "v6_live2_ls", "gnn_axis_v1"])
def test_pool_resolve_classification_matches_the_spec_representation(name: str) -> None:
    """D-05 (agreement arm) — PASS iff the pool's classification agrees with the spec's
    own `representation` field for every registered encoding. FAIL = the pool dispatches
    on something other than the registry's declared representation."""
    resolved = resolve_pool_encoding({"encoding": name}, arch=None)
    spec = resolved.registry_spec
    assert is_graph_representation(spec) == (str(spec.representation) == "graph")


# ── D-17 — AM-1: an unknown representation raises, never defaults to dense ───────
@pytest.mark.parametrize("rep", ["hex_soup", "GRID", "", "dense"])
def test_hparams_unknown_representation_raises(rep: str) -> None:
    """D-17 — PASS iff an unrecognised `representation` raises `RepresentationMismatch`
    (amendment AM-1). The frozen code read `getattr(spec, "is_graph", False)`, so ANY
    spec object that did not answer the attribute silently took the DENSE arm.
    FAIL = the dense-by-default arm is back: a future third representation would be fed
    through the CNN wire format and produce corrupt data with no error (LAW-11)."""
    with pytest.raises(RepresentationMismatch):
        is_graph_representation(_SpecStub(representation=rep))


def test_hparams_absent_representation_raises() -> None:
    """D-17 (absent-attribute arm) — PASS iff a spec object with NO `representation`
    attribute raises rather than defaulting. This is the exact shape the frozen
    `getattr(..., False)` silently swallowed."""
    class _NoRepresentation:
        name = "no_representation"

    with pytest.raises(RepresentationMismatch):
        is_graph_representation(_NoRepresentation())
