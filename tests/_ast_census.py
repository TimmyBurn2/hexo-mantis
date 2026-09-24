"""The AST census helpers the run-authority suites share; roots/repo are parameters."""
from __future__ import annotations

import ast
import tokenize
from pathlib import Path


def production_sources(*roots: Path) -> list[Path]:
    """Every shipped `.py` under the given roots, sorted. `tests/` is deliberately OUT: a test
    may compose freely — the one-authority law is about what SHIPS."""
    return sorted(p for root in roots for p in root.rglob("*.py"))


def rel(path: Path, repo: Path) -> str:
    return str(path.relative_to(repo))


def code_text(path: Path) -> str:
    """Source with COMMENT / STRING / f-string-literal tokens removed, including the 3.11-floor
    guard: FSTRING_MIDDLE is 3.12+, and on 3.11 f-strings lex as STRING."""
    skip = {tokenize.COMMENT, tokenize.STRING, getattr(tokenize, "FSTRING_MIDDLE", -1)}
    with path.open("rb") as handle:
        return "\n".join(tok.string for tok in tokenize.tokenize(handle.readline)
                         if tok.type not in skip)


def enclosing_defs(tree: ast.AST) -> dict[ast.AST, str]:
    """Map every node to the name of the nearest enclosing `def`, so a census can report
    WHERE a call sits rather than only that it exists."""
    owner: dict[ast.AST, str] = {}

    def walk(node: ast.AST, name: str) -> None:
        for child in ast.iter_child_nodes(node):
            child_name = child.name if isinstance(
                child, ast.FunctionDef | ast.AsyncFunctionDef) else name
            owner[child] = child_name
            walk(child, child_name)

    walk(tree, "<module>")
    return owner


def called_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def root_name(node: ast.AST) -> str | None:
    """The base `Name` of an attribute/subscript/call chain: `config.train.device` -> `config`."""
    while True:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute | ast.Subscript):
            node = node.value
        elif isinstance(node, ast.Call):
            node = node.func
        else:
            return None


def func_def(tree: ast.AST, name: str, *, where: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"no `def {name}` found in {where}")


def body_without_docstring(fn: ast.FunctionDef) -> list[ast.stmt]:
    body = list(fn.body)
    if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        return body[1:]
    return body


def call_sites(symbol: str, *, roots: tuple[Path, ...], repo: Path) -> set[str]:
    sites: set[str] = set()
    for path in production_sources(*roots):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        owner = enclosing_defs(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and called_name(node) == symbol:
                sites.add(f"{rel(path, repo)}::{owner.get(node, '<module>')}")
    return sites
