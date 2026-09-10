"""Every production `forward_batch` call passes the GATHER, not a mask.

When the 4th parameter changed from a dense bool mask to the wire's `legal_node_gather`, one call
site was missed and nothing in the tree could see it: the tool is CUDA-only and its measuring arm
has no executing test, and `GraphBatch.legal_mask` is declared `Any`, so pyright accepts a bool
tensor for a `Tensor`-annotated parameter.

The class is "a production call site hands `forward_batch` the wrong view of the legal set", so
the boundary is every `forward_batch` call under `src/`. An AST scan with an ALLOWLIST rather than
a denylist: a newly-invented wrong name fails too, which a `!= "legal_mask"` check would wave through.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src" / "mantis"

#: What a call may pass as the legal-set argument: the `GraphBatch` field, or the parameter name
#: where a local already holds it.
_ALLOWED = {"legal_node_gather", "legal_index"}

#: Position of the legal-set argument in `forward_batch(x, edge_index, edge_attr, <here>, ...)`.
_ARG_POS = 3


def _call_sites() -> list[tuple[str, int, str]]:
    """Return (relative path, lineno, source text of the legal-set argument) per call."""
    found: list[tuple[str, int, str]] = []
    for path in sorted(_SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "forward_batch" not in text:
            continue
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            if name != "forward_batch":
                continue
            kw = {k.arg: k.value for k in node.keywords}
            if "legal_index" in kw:
                arg = kw["legal_index"]
            elif len(node.args) > _ARG_POS:
                arg = node.args[_ARG_POS]
            else:
                pytest.fail(f"{path}:{node.lineno}: forward_batch call has no legal-set argument")
            found.append((str(path.relative_to(_SRC)), node.lineno, ast.unparse(arg)))
    return found


def test_the_scan_finds_the_call_sites_it_claims_to_guard() -> None:
    """A scan that matches nothing passes vacuously. The count is a FLOOR, not a pin: a new
    consumer of the graph forward obeys the rule without editing this number."""
    sites = _call_sites()
    # The floor moves down when a call site is deleted: a floor above what the tree can reach is
    # a claim about code that is not there.
    assert len(sites) >= 3, f"the scan found only {len(sites)} call site(s): {sites}"
    files = {s[0] for s in sites}
    for expected in ("selfplay/inference_server.py", "train/trainer/core.py",
                     "diagnostics/fusion_calibrate.py"):
        assert expected in files, f"{expected} is a known call site the scan did not reach"


def test_every_production_forward_batch_call_passes_the_gather() -> None:
    wrong = [(f, ln, src) for f, ln, src in _call_sites()
             if src.rsplit(".", 1)[-1] not in _ALLOWED]
    assert not wrong, (
        "forward_batch's 4th argument is the wire's `legal_node_gather` (int64 rows), not a "
        f"dense bool mask. Offending call site(s): {wrong}"
    )
