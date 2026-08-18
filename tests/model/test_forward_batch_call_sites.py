"""Every production `forward_batch` call passes the GATHER, not a mask (R71 class-fix).

WHY THIS FILE EXISTS. R284's P-MASK changed `GnnNet.forward_batch`'s 4th parameter from
`legal_mask` (dense bool) to `legal_index` (the wire's `legal_node_gather`). The IMPL converted
four production call sites and missed a fifth —
`mantis/diagnostics/fusion_calibrate.py::_measure_point` — and **nothing in the tree could see
it**:

* the tool is CUDA-ONLY by design and refuses on a host without a device, so its measuring arm
  has no test that executes it (`tests/diagnostics/test_fusion_calibrate_refusals.py` covers the
  refusal, `--shapes-only`, and the budget arguments — never the forward);
* `GraphBatch.legal_mask` is declared `Any` (`graph_collate.py`), so pyright and gate 14 accept
  a bool tensor for a `Tensor`-annotated parameter without complaint.

The miss would have surfaced as an `AssertionError` on the box, on the first sweep point, of the
one tool that is named as the sanctioned way to re-derive `inference.fused_graph_caps` — a
MINTED, mint-critical value (`config/armed_aborts.py`, `config/resolve/fused_graph_caps.py`,
`config/schema/selfplay.py`, `tools/ci_gates/preflight_mint.py` all point at it as the remedy).

R71 says a fix names its class and the flip-set covers the CLASS BOUNDARY, not the demo input.
The class is "a production call site hands `forward_batch` the wrong view of the legal set", and
the boundary is every `forward_batch` call under `src/`. So this is an AST scan with an ALLOWLIST
rather than a denylist: a new call passing a newly-invented wrong name fails too, which a
`!= "legal_mask"` check would wave through.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src" / "mantis"

#: What a call may pass as the legal-set argument. `legal_node_gather` is the field on
#: `GraphBatch`; `legal_index` is the parameter name, used where a local already holds it.
_ALLOWED = {"legal_node_gather", "legal_index"}

#: Position of the legal-set argument in `forward_batch(x, edge_index, edge_attr, <here>, ...)`.
_ARG_POS = 3


def _call_sites() -> list[tuple[str, int, str]]:
    """(relative path, lineno, the source text of the legal-set argument) per call."""
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
    """LAW-07's first half: a scan that matches nothing passes vacuously. The count is a FLOOR,
    not a pin — a new production consumer of the graph forward must not have to edit this
    number, only to obey the rule."""
    sites = _call_sites()
    assert len(sites) >= 4, f"the scan found only {len(sites)} call site(s): {sites}"
    files = {s[0] for s in sites}
    for expected in ("selfplay/inference_server.py", "train/trainer/core.py",
                     "train/subsystems.py", "diagnostics/fusion_calibrate.py"):
        assert expected in files, f"{expected} is a known call site the scan did not reach"


def test_every_production_forward_batch_call_passes_the_gather() -> None:
    wrong = [(f, ln, src) for f, ln, src in _call_sites()
             if src.rsplit(".", 1)[-1] not in _ALLOWED]
    assert not wrong, (
        "forward_batch's 4th argument is the wire's `legal_node_gather` (int64 rows), not a "
        f"dense bool mask. Offending call site(s): {wrong}"
    )
