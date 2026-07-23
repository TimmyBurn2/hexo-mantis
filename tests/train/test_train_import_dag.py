"""O-DAG — `train/*` has no top-level import of `mantis.eval` / `mantis.monitor` /
`mantis.arena` (WP10 §b O-DAG; repo_design §2).

CI gate 9 (`tools/check_import_dag.py`) proves the whole `src/mantis` graph is acyclic, but
a `train → eval` / `train → monitor` edge is only a CYCLE if the reverse edge also exists —
so the FORBIDDEN-edge invariant needs its own oracle. eval/monitor/arena are reached via
injected callables / the `EventSink` Protocol, never a hard import edge; `init_trainer`
lazily imports `Trainer` (inside its body — not a top-level edge). This test mirrors the
check_import_dag semantics: it inspects module TOP-LEVEL imports only (function/method-body
and `if TYPE_CHECKING` imports are lazy by definition and are NOT edges).
"""
from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN = ("mantis.eval", "mantis.monitor", "mantis.arena")
TRAIN_ROOT = Path(__file__).resolve().parents[2] / "src" / "mantis" / "train"


def _top_level_imports(tree: ast.Module) -> list[str]:
    """Dotted targets of module TOP-LEVEL import statements only (no function/class body,
    no `if TYPE_CHECKING`)."""
    targets: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            targets.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            targets.append(node.module)
            targets.extend(f"{node.module}.{alias.name}" for alias in node.names)
    return targets


def _forbidden_hit(target: str) -> str | None:
    for pkg in FORBIDDEN:
        if target == pkg or target.startswith(pkg + "."):
            return pkg
    return None


def test_train_has_no_top_level_eval_monitor_arena_import():
    train_files = sorted(TRAIN_ROOT.rglob("*.py"))
    assert train_files, f"no train/ modules found under {TRAIN_ROOT}"
    violations: list[str] = []
    for path in train_files:
        tree = ast.parse(path.read_text(), filename=str(path))
        for target in _top_level_imports(tree):
            hit = _forbidden_hit(target)
            if hit is not None:
                violations.append(f"{path.relative_to(TRAIN_ROOT.parents[2])} -> {target} ({hit})")
    assert not violations, (
        "train/* must not top-level import eval/monitor/arena (reach them via injected "
        "callables / the EventSink Protocol):\n  " + "\n  ".join(violations)
    )
