"""Arm 8 (`LocalInferenceEngine` encoding default) is bounded on every reachable path.

R56 rider. Arm 8 is the one silent-encoding fallback WPUF-2 left OPEN — it is registered in
gate 11's `KNOWN_DEBT` under WP12-R rather than closed, because `src/mantis/eval/worker.py`
constructs the engine positionally and closing the default would change eval-worker
behaviour that is WP12-R's decision to make (WP11-A handoff, run5-mint blocker).

R56 requires proof that, while it stays open, every reachable path either **passes the
encoding explicitly** or **fails loud**. That is what this file pins. If these tests cannot
hold, R56 escalates arm 8 to a hard run5-mint blocker — so a failure here is not a broken
test, it is the escalation trigger.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from mantis.encoding import lookup
from mantis.selfplay.hparams import is_graph_representation
from mantis.selfplay.inference_local import LocalInferenceEngine

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src" / "mantis"

# The construction sites that deliberately do NOT pass a spec. This set is the arm-8
# exposure, and it must not grow silently: a new positional construction is a new
# unbounded path, which is exactly what gate 11 registered the debt to prevent.
KNOWN_UNTHREADED = {"eval/worker.py"}


def _construction_sites() -> list[tuple[str, int, bool]]:
    """(relpath, lineno, passes_encoding_spec) for every `LocalInferenceEngine(...)` call."""
    sites: list[tuple[str, int, bool]] = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
            if name != "LocalInferenceEngine":
                continue
            threaded = any(kw.arg == "encoding_spec" for kw in node.keywords) or len(
                node.args
            ) >= 3
            sites.append((str(path.relative_to(SRC)), node.lineno, threaded))
    return sites


def test_every_construction_site_is_censused():
    """The census must find something — a silent AST miss would make this file vacuous."""
    assert _construction_sites(), "no LocalInferenceEngine construction found in src/"


def test_no_new_unthreaded_construction_site_appears():
    """Arm 8's exposure is exactly the known WP12-R sites and nothing else."""
    unthreaded = {rel for rel, _line, threaded in _construction_sites() if not threaded}
    assert unthreaded <= KNOWN_UNTHREADED, (
        f"NEW unthreaded LocalInferenceEngine construction(s): "
        f"{sorted(unthreaded - KNOWN_UNTHREADED)}. Arm 8's exposure may not grow while it "
        f"is registered-open; either thread encoding_spec or escalate per R56."
    )


def test_the_selfplay_path_threads_its_spec_explicitly():
    """The self-play actor — the high-volume path — never relies on the default."""
    threaded = {rel for rel, _line, t in _construction_sites() if t}
    assert "selfplay/worker.py" in threaded


def test_the_default_is_a_grid_spec_and_that_is_why_graph_callers_break_loudly():
    """Pins WHAT the default is, so a change to it is visible rather than inferred.

    The default being a *grid* spec is the entire mechanism by which the unthreaded eval
    path fails loud on graph regimes instead of quietly decoding garbage: representation
    dispatch reads the bound spec, so a graph model bound to a dense spec takes the dense
    arm and blows up rather than producing plausible-looking wrong numbers.
    """
    default_spec = lookup("v6")
    assert not is_graph_representation(default_spec)


def test_a_graph_model_with_the_defaulted_dense_spec_fails_loud_not_silent():
    """The R56 property itself, on the reachable mismatch.

    A graph-built net handed the defaulted (dense) spec must RAISE. Silently running the
    dense arm over a graph model would be the failure R56 exists to rule out — plausible
    output from the wrong pipeline.
    """
    torch = pytest.importorskip("torch")
    from mantis.model import arch_from_spec_and_config, build_net

    graph_spec = lookup("gnn_axis_v1")
    graph_net = build_net(arch_from_spec_and_config(graph_spec, {}))

    # Construct with NO encoding_spec — exactly what eval/worker.py does today.
    engine = LocalInferenceEngine(graph_net, torch.device("cpu"))
    try:
        # It bound the dense default, so it took the dense arm despite a graph net.
        assert engine.encoding_spec.name == "v6"
        assert engine._is_graph is False

        # The specific loudness, asserted rather than a bare `Exception`: the dense arm
        # calls `model(...)`, and `GnnNet` implements `forward_batch` but NOT `forward`.
        #
        # That mechanism is INCIDENTAL and this assertion is deliberately tight because of
        # it: if `GnnNet` ever gains a `forward`, this mismatch stops raising and starts
        # returning dense-shaped output from a graph net — silent, plausible and wrong.
        # This test going red in that way is R56's escalation trigger for arm 8, not a
        # test to relax.
        with pytest.raises(NotImplementedError, match="forward"):
            engine.infer_batch([_a_board()])
    finally:
        engine.close()


def _a_board():
    from mantis._engine import Board

    return Board()
