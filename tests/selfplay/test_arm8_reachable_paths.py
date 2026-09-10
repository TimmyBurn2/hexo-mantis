"""Arm 8 (`LocalInferenceEngine`'s encoding default) is CLOSED, and stays closed.

WP12-R Phase B threads the round's DECLARED encoding through `mantis.eval.worker` — the
last production construction site that relied on the default — and Phase C deletes the
`encoding_spec if ... else lookup(<a constant>)` ternary outright, making `encoding_spec` a
REQUIRED keyword-only parameter. Absent is then UNCONSTRUCTIBLE rather than defaulted,
which is what LAW-11 asks for.

This file was arm 8's R56 rider while the arm sat registered-open in gate 11's
`KNOWN_DEBT`, pinning that every reachable path either threaded the spec or failed loud.
With the arm closed it becomes the REOPEN GUARD: no default may come back, no construction
site may omit the spec, and the one mismatch that is still constructible must still fail
loud. (R56's escalation trigger has no subject after this card — ADJ-WP12R-3.)

⊕ WP12-R oracles O-4 and O-5 (PREREG §1), RED at HEAD by pre-registration. O-6 — the one
still-constructible mismatch, a graph net bound to a DENSE spec — has no subject since
R346(f) deleted the dense arm: there is no second representation to bind the wrong one of.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from mantis.encoding import lookup
from mantis.selfplay.inference_local import LocalInferenceEngine

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src" / "mantis"


def _construction_sites() -> list[tuple[str, int, bool]]:
    """(relpath, lineno, passes_encoding_spec) for every `LocalInferenceEngine(...)` call.

    Keyword detection ONLY. `encoding_spec` is keyword-only after Phase C, so a third
    positional argument cannot thread it — the old `len(node.args) >= 3` heuristic could
    now only mislabel an unrelated positional as threaded.
    """
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
            threaded = any(kw.arg == "encoding_spec" for kw in node.keywords)
            sites.append((str(path.relative_to(SRC)), node.lineno, threaded))
    return sites


def test_every_construction_site_is_censused():
    """The census must find something — a silent AST miss would make this file vacuous."""
    assert _construction_sites(), "no LocalInferenceEngine construction found in src/"


def test_no_construction_site_omits_the_spec():
    """⊕ O-4. NO production construction may omit the spec. No allowlist — an empty one
    is a weaker statement than none at all, and the constant it replaced is deleted.

    HEAD: RED, reporting exactly `{"eval/worker.py"}` (its two sites, `:78` and `:193`).
    """
    unthreaded = {rel for rel, _line, threaded in _construction_sites() if not threaded}
    assert unthreaded == set(), (
        f"LocalInferenceEngine construction(s) with no encoding_spec: {sorted(unthreaded)}. "
        f"Arm 8 is CLOSED: every construction states its encoding, because the decode and "
        f"the board geometry must be sized from the SAME declared value."
    )


def test_the_selfplay_path_threads_its_spec_explicitly():
    """The eval worker — the seam WP12-R exists to close — threads it.

    Named rather than merely counted: this is the seam a regression would silently re-point
    at a constant. The self-play worker was the other named consumer; it went with the dense
    path (R346(f)), and `test_no_construction_site_omits_the_spec` above is what covers any
    site that replaces it.
    """
    threaded = {rel for rel, _line, t in _construction_sites() if t}
    assert {"eval/worker.py"} <= threaded


def test_there_is_no_default_encoding_spec():
    """⊕ O-5. There is no default to fall back to, and none can be supplied positionally.

    The default dies AT THE SIGNATURE rather than behind a raise: LAW-11 says an absent
    encoding is an error, and a required keyword-only parameter makes absent
    unconstructible — checked by pyright before a worker ever spawns, where a runtime
    `if encoding_spec is None: raise` would be unreachable code behind it.

    HEAD: RED on both arms — measured `default: None kind: POSITIONAL_OR_KEYWORD`.
    """
    torch = pytest.importorskip("torch")

    parameter = inspect.signature(LocalInferenceEngine.__init__).parameters["encoding_spec"]
    assert parameter.default is inspect.Parameter.empty
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY

    with pytest.raises(TypeError):
        LocalInferenceEngine(torch.nn.Identity(), torch.device("cpu"))


def _a_board():
    from mantis._engine import Board

    return Board()
