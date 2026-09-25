"""ONE resolver of the live-consumer strings for the consumer registry."""
from __future__ import annotations

import ast
import functools
import importlib.util
import re
from pathlib import Path


_REPO = Path(__file__).resolve().parents[2]

__all__ = ["defined_names", "unresolved_tokens"]
#: A code token: a dotted name, or a snake_case name with an underscore (prose has none).
_TOKEN = re.compile(r"(?<![\w.])([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+|[a-z_][a-z0-9]*_[a-z0-9_]+)(?![\w.])")
_RUST_ITEM = re.compile(r"\b(?:fn|struct|enum|const|static|mod|type)\s+([A-Za-z_]\w*)")
_RUST_FIELD = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?([a-z_]\w*):\s", re.M)
_IDENT = re.compile(r"^[A-Za-z_]\w*$")
_NOT_CODE_SUFFIX = (".py", ".yaml", ".toml", ".md", ".json", ".txt", ".sh")


@functools.cache
def defined_names() -> frozenset[str]:
    """Every name DEFINED in src/ and crates/ (defs, targets, params, identifier strings, Rust items)."""
    names: set[str] = set()
    for path in (_REPO / "src").rglob("*.py"):
        names.update(path.relative_to(_REPO / "src").with_suffix("").parts)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    names.update(_target_names(target))
            elif isinstance(node, ast.AnnAssign):
                names.update(_target_names(node.target))
            elif isinstance(node, ast.arg):
                names.add(node.arg)
            elif isinstance(node, ast.keyword) and node.arg:
                names.add(node.arg)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and _IDENT.match(node.value):
                names.add(node.value)
    for path in (_REPO / "crates").rglob("*.rs"):
        text = path.read_text(encoding="utf-8")
        names.update(_RUST_ITEM.findall(text))
        names.update(_RUST_FIELD.findall(text))
    return frozenset(names)


def _target_names(target: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(target):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
    return out


def unresolved_tokens(text: str, defined: frozenset[str]) -> list[str]:
    """Tokens with a segment defined nowhere; importable modules, ALL-CAPS and file names pass."""
    dead: list[str] = []
    for tok in _TOKEN.findall(text):
        if tok.endswith(_NOT_CODE_SUFFIX) or tok.startswith(("http", "docs.", "tests.", "tools.")):
            continue
        segs = tok.split(".")
        if segs[0] not in defined and importlib.util.find_spec(segs[0]) is not None:
            continue
        if any(seg not in defined and not seg.isupper() for seg in segs):
            dead.append(tok)
    return dead

