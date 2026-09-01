"""R328(b)(c) — the run6 graph identity `gnn_axis_r8`, and the marker branch it forced open.

WHY A NEW REGISTRY ROW AND NOT AN EDIT TO `gnn_axis_v1`. `configs/run5.yaml` names
`gnn_axis_v1`, and R26 holds run5's radius registry-derived at 6 — so mutating that row would
move run5's radius to 8 while typing no `8` into any run5 file, and nothing could detect it:
`CheckpointMetadata` stamps `encoding_name` and NOT the geometry. This suite pins the
consequence rather than the intention: **the two rows differ in exactly one knob, in two
places, and `gnn_axis_v1` still reads 6.**
"""
from __future__ import annotations

import pytest

import mantis.encoding as encoding
from mantis.encoding.resolvers import (
    AmbiguousGraphMarkerError,
    detect_encoding_from_state_dict,
)
from mantis.encoding._probes import GNN_GRAPH_MARKER_KEY

R8 = "gnn_axis_r8"
V1 = "gnn_axis_v1"


def _spec(name: str):
    return encoding.lookup(name)


def test_r328b_01_the_run6_row_exists_and_carries_radius_8_at_BOTH_sites() -> None:
    """Two keys, one geometry — the registry's own comment calls `legal_move_radius` the one
    that *matches graph_radius*, so a row that moved only one would be half-changed."""
    spec = _spec(R8)
    assert spec.representation == "graph"
    assert spec.legal_move_radius == 8, "legal_move_radius is the Board's authority"
    assert spec.graph_radius == 8, "graph_radius is the builder's authority"


def test_r328b_02_run5s_identity_is_untouched() -> None:
    """The whole ground for a new row rather than an edit. If this reds, R26 is broken."""
    v1 = _spec(V1)
    assert v1.legal_move_radius == 6
    assert v1.graph_radius == 6


def test_r328b_03_exactly_one_knob_differs_between_the_two_graph_rows() -> None:
    """DERIVED field-by-field, never a typed list of "the fields that should match".

    A typed list is edited in the same commit as the drift it would have caught. This walks
    every field the spec surface exposes and asserts the difference SET is exactly the two
    radius keys — so a stray `win_length` or `policy_logit_count` change reds here even though
    nobody thought to assert that field."""
    a, b = _spec(V1), _spec(R8)
    fields = [f for f in dir(a) if not f.startswith("_") and not callable(getattr(a, f))]
    assert len(fields) > 10, f"the spec surface collapsed to {fields!r}; this test is vacuous"
    differing = {f for f in fields if getattr(a, f) != getattr(b, f)}
    assert differing == {"name", "legal_move_radius", "graph_radius", "notes"}, (
        f"the two graph rows differ in {sorted(differing)}. This encoding exists to move ONE "
        "knob so the r6/r8 comparison is a comparison; any other difference makes it two.")


def test_r328b_04_the_roster_carries_both_graph_rows() -> None:
    """The conformance suite parametrises over `all_specs()`, so this is what makes every
    tier run at radius 8 without a tier edit."""
    graph = sorted(s.name for s in encoding.all_specs()
                   if getattr(s, "representation", "grid") == "graph")
    assert graph == [R8, V1], f"graph roster is {graph}"


# ═══ the marker branch the second graph row forced open ══════════════════════════════════
def test_r328c_05_an_unstamped_graph_checkpoint_now_REFUSES_instead_of_guessing() -> None:
    """THE DEFECT THE IDENTITY CHANGE EXPOSED, and the reason it is code and not a gate.

    `detect_encoding_from_state_dict` used to answer `lookup("gnn_axis_v1")` for any state
    dict carrying the graph marker. That was under-determined all along — the marker says
    GRAPH, never WHICH graph — and invisible while one graph row existed. Gate 11 says in its
    own docstring that it cannot catch this shape (affirmative dispatch), so the refusal has
    to live here."""
    state = {GNN_GRAPH_MARKER_KEY: object()}
    with pytest.raises(AmbiguousGraphMarkerError) as excinfo:
        detect_encoding_from_state_dict(state, "unstamped_graph.pt")
    message = str(excinfo.value)
    assert R8 in message and V1 in message, (
        f"the refusal must NAME the candidates it refused to choose between: {message!r}")


def test_r328c_06_a_STAMPED_graph_checkpoint_still_resolves_both_ways() -> None:
    """The positive control. Row 05 would pass on a function that raised unconditionally."""
    for name in (V1, R8):
        state = {GNN_GRAPH_MARKER_KEY: object(), "metadata": {"encoding_name": name}}
        assert detect_encoding_from_state_dict(state, "stamped.pt").name == name


def test_r328c_07_the_refusal_is_derived_from_the_registry_not_from_a_count(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Pruning back to ONE graph row re-arms the marker branch with no edit.

    Pins that the branch reads the live registry rather than a hard-coded `2`, which is what
    keeps this from becoming a row that has to be remembered."""
    from mantis.encoding import resolvers
    monkeypatch.setattr(resolvers, "_graph_specs", lambda: [_spec(V1)])
    got = detect_encoding_from_state_dict({GNN_GRAPH_MARKER_KEY: object()}, "unstamped.pt")
    assert got.name == V1
