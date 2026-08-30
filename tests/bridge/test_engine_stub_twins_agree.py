"""The two `_engine.pyi` stubs are TWINS, and nothing checked that they still were.

THE DEFECT, MEASURED. Both files' own headers say *"Keep both stubs identical when the
bridge API changes"* — a rule enforced by memory, which is the enforcement rule 7 was under
when a scan found 101 committed box paths (gate 17's history). Measured at NIGHTRUN-1 Leg 1,
the two had already drifted: `InferenceBatcher.lock_recoveries` — a real export, a live
LAW-18 counter — was declared in the wheel-shipped copy and ABSENT from
`src/mantis/_engine.pyi`, which is the copy pyright actually reads. So the type checker
would have refused a correct read of a shipped counter, and the copy that ships in the wheel
was not the copy CI type-checks against.

WHY THE HEADERS ARE THE ONLY LEGITIMATE DIFFERENCE. Each file's module docstring names the
OTHER one and says what its own copy is for, so they cannot be byte-identical whole. The
comparison therefore starts after each file's own docstring and is byte-exact from there —
never a normalised or fuzzy match, because a stub that differs by a default value or an
argument is exactly the drift this exists to catch.

THIS IS A DUPLICATE-AUTHORITY GUARD (R79), not a style rule. The single authority for the
FFI surface is the Rust `#[pymethods]` block; both stubs are transcriptions of it. Two
transcriptions with no check between them is one transcription plus an unverified copy.
"""
from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src" / "mantis" / "_engine.pyi"
_WHEEL = _REPO / "crates" / "mantis-bridge" / "python" / "mantis" / "_engine.pyi"


def _body(path: Path) -> str:
    """Everything after the file's own module docstring — the part that must be identical."""
    text = path.read_text(encoding="utf-8")
    assert text.startswith('"""'), f"{path} does not open with a module docstring"
    end = text.index('"""', 3)
    return text[end + 3:]


def test_both_stub_files_exist() -> None:
    """A vacuity guard. A comparison of two files one of which was renamed away would pass
    on an exception nobody sees; here it fails on the file, by name."""
    assert _SRC.is_file(), f"{_SRC} is missing"
    assert _WHEEL.is_file(), f"{_WHEEL} is missing"


def test_the_two_stubs_declare_the_SAME_ffi_surface() -> None:
    src_body, wheel_body = _body(_SRC), _body(_WHEEL)
    if src_body != wheel_body:
        import difflib
        diff = "\n".join(
            difflib.unified_diff(
                src_body.splitlines(), wheel_body.splitlines(),
                fromfile=str(_SRC.relative_to(_REPO)),
                tofile=str(_WHEEL.relative_to(_REPO)), lineterm="", n=2,
            )
        )
        raise AssertionError(
            "the two `_engine.pyi` stubs have drifted. The src copy is what pyright reads; "
            "the wheel copy is what ships. Both transcribe the SAME Rust `#[pymethods]` "
            f"surface, so a difference is one of them being wrong:\n{diff}"
        )


def test_each_header_names_the_other_copy() -> None:
    """The headers carry the rule this file now enforces, and each must point at its twin —
    otherwise a reader who finds one stub has no way to learn the other exists."""
    src_head = _SRC.read_text(encoding="utf-8")[:_SRC.read_text(encoding="utf-8").index('"""', 3)]
    wheel_head = (_WHEEL.read_text(encoding="utf-8")
                  [:_WHEEL.read_text(encoding="utf-8").index('"""', 3)])
    assert "crates/mantis-bridge/python/mantis/_engine.pyi" in src_head
    assert "src/mantis/_engine.pyi" in wheel_head
