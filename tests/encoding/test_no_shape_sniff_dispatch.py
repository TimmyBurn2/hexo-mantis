"""AUDIT-1 F-20 — no read surface executes pickle, and no DISPATCH path sniffs a shape.

TWO DEFECTS, one finding.

**The pickle-exec hole.** `docs/contracts/checkpoint_envelope.md` asserts *"every read surface
is `torch.load(weights_only=True)`; there is no pickle-exec fallback"*. That was FALSE at HEAD
in two places: `encoding/resolvers.py::resolve_from_checkpoint` (reachable from the pretrain
CLI's `--resume` without `--encoding`) and `encoding/audit_sections.py`'s §2 checkpoint scan.
Both loaded with `weights_only=False`, which executes arbitrary pickle.

**The shape sniffer on a dispatch path.** With no stamp, `resolve_from_checkpoint` fell through
to `detect_encoding_from_state_dict`: first an ARCH-STRUCTURAL marker key
(`representation.input_proj.weight`), then conv / policy-fc shape matching across the
registered grid set. A V3 graph arch that renames `input_proj` is classified GRID — silently,
with a `DeprecationWarning` that advises the only correct action while proceeding without it.
`checkpoints.load_legacy_weights` refuses to shape-sniff and says so; two postures on one
question is the duplicate-authority class.

WHAT WAS KEPT AND WHY. The detector still exists for ONE caller: the audit CLI's §2
declared-vs-inferred reconciliation, which REPORTS to an operator and selects no behaviour.
Deleting it would remove a real diagnostic; letting it dispatch is what F-20 is about. This
file is the line between those.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src" / "mantis"
#: The ONE module allowed to call the shape detector — it reports, it does not dispatch.
_REPORTING_CALLER = "encoding/audit_sections.py"
#: The detector and its thin name-returning wrapper live here; a definition is not a call.
_DETECTOR_HOMES = {"encoding/resolvers.py", "encoding/compat.py"}


def _calls_named(names: set[str]) -> list[tuple[str, int, str]]:
    found: list[tuple[str, int, str]] = []
    for path in sorted(_SRC.rglob("*.py")):
        rel = str(path.relative_to(_SRC))
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else fn.id if isinstance(fn, ast.Name) else None
            if name in names:
                found.append((rel, node.lineno, name or ""))
    return found


def test_every_torch_load_in_src_is_weights_only() -> None:
    """Contract #4, asserted rather than stated. `weights_only=False` executes pickle; an
    OMITTED `weights_only` is the same hazard with less evidence of intent."""
    offenders: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        rel = str(path.relative_to(_SRC))
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            is_torch_load = (
                isinstance(fn, ast.Attribute) and fn.attr == "load"
                and isinstance(fn.value, ast.Name) and fn.value.id == "torch"
            )
            if not is_torch_load:
                continue
            kw = next((k for k in node.keywords if k.arg == "weights_only"), None)
            if kw is None:
                offenders.append(f"{rel}:{node.lineno} — no `weights_only=`")
            elif not (isinstance(kw.value, ast.Constant) and kw.value.value is True):
                offenders.append(f"{rel}:{node.lineno} — weights_only is not True")
    assert not offenders, (
        f"a torch.load read surface that is not weights-only: {offenders}. "
        "`docs/contracts/checkpoint_envelope.md` asserts every read surface is "
        "`weights_only=True` with no pickle-exec fallback (AUDIT-1 F-20)."
    )


def test_the_census_has_a_subject() -> None:
    """Vacuity guard: the walk must actually reach `torch.load` calls."""
    loads = [
        (rel, line) for rel, line, _ in _calls_named({"load"})
    ]
    assert len(loads) >= 5, f"only {len(loads)} `load(` call(s) reached — the walk is broken"


def test_only_the_audit_calls_the_shape_detector() -> None:
    """The detector REPORTS; it must never dispatch. A second caller is how a report becomes a
    decision without anyone deciding."""
    callers = {
        rel for rel, _line, _n in _calls_named(
            {"detect_encoding_from_state_dict", "infer_encoding_from_state_dict"}
        )
    } - _DETECTOR_HOMES
    assert callers <= {_REPORTING_CALLER}, (
        f"the state-dict shape detector is called from {sorted(callers - {_REPORTING_CALLER})}. "
        "It dispatches on an arch-structural key and on conv widths, so a renamed graph trunk "
        "reads as grid — which is fine in a report and not fine in a resolution (AUDIT-1 F-20)."
    )


def test_an_unstamped_checkpoint_RAISES_instead_of_being_guessed(tmp_path: Path) -> None:
    """The behaviour change, pinned. This used to warn and return a guess; the warning's advice
    ('stamp metadata explicitly') was the only correct action, and proceeding anyway is what
    let unstamped artifacts stay unstamped."""
    import torch

    from mantis.encoding import resolve_from_checkpoint
    from mantis.encoding.registry import EncodingRegistryError

    path = tmp_path / "unstamped.pt"
    torch.save({"model_state": {"trunk.0.weight": torch.zeros(8, 8, 3, 3)}}, path)
    with pytest.raises(EncodingRegistryError, match="no metadata"):
        resolve_from_checkpoint(path)


def test_a_stamped_checkpoint_still_resolves(tmp_path: Path) -> None:
    """The control: the stamp path is untouched, and it is the only path."""
    import torch

    from mantis.encoding import resolve_from_checkpoint

    path = tmp_path / "stamped.pt"
    torch.save({"model_state": {}, "metadata": {"encoding_name": "v6_live2_ls"}}, path)
    assert resolve_from_checkpoint(path).name == "v6_live2_ls"
