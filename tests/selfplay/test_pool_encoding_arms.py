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
from mantis.model import GnnArch, RepresentationMismatch
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
    for registered in (spec.name for spec in all_specs()):
        assert registered in message, (
            f"the error must name the registered set (missing {registered!r}) — an "
            "operator reading it should not have to grep the registry"
        )


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
    """D-05 — PASS iff every spec in the registry classifies onto the graph arm, and the arm
    is non-empty. FAIL = a registered encoding the pool cannot dispatch — i.e. a spec that
    would reach a default arm. There is no default arm, so this is the test that keeps the
    closed set honest as the registry grows. The dense arm it used to partition against went
    with the grid path (R346(f)); what survives is that no spec falls through."""
    specs = list(all_specs())
    assert specs, "the registry must not be empty"

    graph = [spec.name for spec in specs if is_graph_representation(spec)]

    assert len(graph) == len(specs), "a spec did not classify onto the graph arm"


@pytest.mark.parametrize("name", [spec.name for spec in all_specs()])
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
