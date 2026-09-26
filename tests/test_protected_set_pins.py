"""Every invariant of LAWS.md's protected set names pinning tests, and every named test exists."""
from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_LAWS = _REPO / "docs" / "governance" / "LAWS.md"
_PIN = re.compile(r"`([\w./-]+\.(?:py|rs))::(\w+)`")
_INVARIANTS = 11


def _protected_bullets(text: str) -> list[str]:
    """The protected set's bullets, each joined with its continuation lines."""
    section = text.split("## The protected set", 1)[1].split("\n## ", 1)[0]
    bullets: list[str] = []
    for line in section.splitlines():
        if line.startswith("- "):
            bullets.append(line)
        elif bullets and line.startswith("  "):
            bullets[-1] += " " + line.strip()
    return bullets


def _missing_pins(text: str, root: Path) -> list[str]:
    """Each pin whose file or test function does not exist, and each bullet that names none."""
    missing: list[str] = []
    for bullet in _protected_bullets(text):
        pins = _PIN.findall(bullet)
        if not pins:
            missing.append(f"no pin: {bullet}")
        for rel, name in pins:
            path = root / rel
            define = rf"^\s*(?:async\s+)?def {name}\(" if rel.endswith(".py") else rf"\bfn {name}\s*\("
            if not path.is_file() or not re.search(define, path.read_text(encoding="utf-8"), re.M):
                missing.append(f"{rel}::{name}")
    return missing


def test_the_protected_set_has_all_its_invariants() -> None:
    assert len(_protected_bullets(_LAWS.read_text(encoding="utf-8"))) == _INVARIANTS


def test_every_named_pinning_test_exists() -> None:
    assert _missing_pins(_LAWS.read_text(encoding="utf-8"), _REPO) == []


def test_a_renamed_pin_and_an_unpinned_invariant_are_both_caught() -> None:
    text = _LAWS.read_text(encoding="utf-8")
    first = _PIN.search(text.split("## The protected set", 1)[1])
    assert first is not None
    renamed = text.replace(f"::{first.group(2)}`", f"::{first.group(2)}_renamed`", 1)
    assert _missing_pins(renamed, _REPO) == [f"{first.group(1)}::{first.group(2)}_renamed"]
    unpinned = text.replace("- arena legality —", "- arena legality\n- a new invariant —", 1)
    assert any(m.startswith("no pin: - arena legality") for m in _missing_pins(unpinned, _REPO))
