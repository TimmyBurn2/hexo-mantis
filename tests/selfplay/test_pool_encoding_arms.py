"""The encoding arms of the self-play pool resolver (`mantis.selfplay.hparams`).

Two rows pin deviations from the old resolver rather than its behaviour: an unregistered
encoding now hard-errors in the registry instead of at a hand-written guard, and the board-size
cross-check reads the DECLARED arch, so a graph arch passes it vacuously.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from mantis.encoding import EncodingRegistryError, all_specs, lookup
from mantis.model import GnnArch, RepresentationMismatch
from mantis.selfplay.hparams import is_graph_representation, resolve_pool_encoding

# The v8 family is retired; its guard died with its registry entry.
_V8_FAMILY = ("v8", "v8_canvas_realness")


@dataclass
class _SpecStub:
    """Spec-shaped stub carrying only what the representation dispatch reads."""

    representation: str
    name: str = "spec_stub"


@pytest.mark.parametrize("name", [*_V8_FAMILY, "definitely_not_registered"])
def test_unregistered_encoding_loud(name: str) -> None:
    """Prove an unregistered encoding raises, naming both the rejected name and the registered set.

    A silent fall-through to a default encoding would train a run against a wire format
    nobody asked for.
    """
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
    """Prove a `GnnArch`, which declares no board size, passes the cross-check vacuously."""
    spec = lookup("gnn_axis_v1")
    arch = GnnArch(in_dim=spec.node_feat_dim, edge_dim=spec.edge_feat_dim,
                   hidden=32, num_layers=2)
    assert not hasattr(arch, "board_size"), (
        "GnnArch must declare no board_size — the vacuous pass depends on it"
    )
    resolved = resolve_pool_encoding({"encoding": "gnn_axis_v1"}, arch=arch)
    assert resolved.encoding_name == "gnn_axis_v1"
    assert resolved.n_kept_planes == 0


def test_every_registered_encoding_classified() -> None:
    """Prove every registered spec classifies onto the graph arm; there is no default arm to
    fall through to.
    """
    specs = list(all_specs())
    assert specs, "the registry must not be empty"

    graph = [spec.name for spec in specs if is_graph_representation(spec)]

    assert len(graph) == len(specs), "a spec did not classify onto the graph arm"


@pytest.mark.parametrize("name", [spec.name for spec in all_specs()])
def test_pool_resolve_classification_matches_the_spec_representation(name: str) -> None:
    """Prove the pool classifies on the registry's declared representation, not on something else."""
    resolved = resolve_pool_encoding({"encoding": name}, arch=None)
    spec = resolved.registry_spec
    assert is_graph_representation(spec) == (str(spec.representation) == "graph")


@pytest.mark.parametrize("rep", ["hex_soup", "GRID", "", "dense"])
def test_hparams_unknown_representation_raises(rep: str) -> None:
    """Prove an unrecognised representation raises rather than taking a default arm.

    A dense-by-default arm would feed a future third representation through the wrong wire
    format and produce corrupt data with no error.
    """
    with pytest.raises(RepresentationMismatch):
        is_graph_representation(_SpecStub(representation=rep))


def test_hparams_absent_representation_raises() -> None:
    """Prove a spec object with no `representation` attribute raises rather than defaulting."""
    class _NoRepresentation:
        name = "no_representation"

    with pytest.raises(RepresentationMismatch):
        is_graph_representation(_NoRepresentation())
