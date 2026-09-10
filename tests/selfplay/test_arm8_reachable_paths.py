"""`LocalInferenceEngine`'s encoding default is CLOSED, and stays closed.

`encoding_spec` is a required keyword-only parameter, so an absent encoding is
unconstructible rather than defaulted. This file is the reopen guard: no default may come
back, and no construction site may omit the spec.
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
    """Return `(relpath, lineno, passes_encoding_spec)` for every `LocalInferenceEngine(...)` call.

    Keyword detection only: `encoding_spec` is keyword-only, so a positional argument
    cannot thread it and counting positionals could only mislabel one as threaded.
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
    """Prove the census finds construction sites; a silent AST miss would make this file vacuous."""
    assert _construction_sites(), "no LocalInferenceEngine construction found in src/"


def test_no_construction_site_omits_the_spec():
    """Prove no production construction omits the spec; there is no allowlist."""
    unthreaded = {rel for rel, _line, threaded in _construction_sites() if not threaded}
    assert unthreaded == set(), (
        f"LocalInferenceEngine construction(s) with no encoding_spec: {sorted(unthreaded)}. "
        f"Arm 8 is CLOSED: every construction states its encoding, because the decode and "
        f"the board geometry must be sized from the SAME declared value."
    )


def test_the_selfplay_path_threads_its_spec_explicitly():
    """Prove the eval worker threads its spec, named rather than merely counted.

    A regression would silently re-point this seam at a constant; the census above covers
    any site that replaces it.
    """
    threaded = {rel for rel, _line, t in _construction_sites() if t}
    assert {"eval/worker.py"} <= threaded


def test_there_is_no_default_encoding_spec():
    """Prove there is no default encoding spec and none can be supplied positionally.

    The default dies at the signature rather than behind a raise, so pyright catches an
    absent encoding before a worker ever spawns.
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
