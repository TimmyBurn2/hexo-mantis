"""O-DAG — `train/*` has no top-level import of `mantis.eval` / `mantis.arena`
(WP10 §b O-DAG; repo_design §2).

CI gate 9 (`tools/check_import_dag.py`) proves the whole `src/mantis` graph is acyclic, but
a `train → eval` edge is only a CYCLE if the reverse edge also exists — so the
FORBIDDEN-edge invariant needs its own oracle. eval/arena are reached via injected
callables, never a hard import edge; `init_trainer` lazily imports `Trainer` (inside its
body — not a top-level edge). This test mirrors the check_import_dag semantics: it inspects
module TOP-LEVEL imports only (function/method-body and `if TYPE_CHECKING` imports are lazy
by definition and are NOT edges).

WP13-A: `mantis.monitor` LEAVES this blanket-ban tuple — `train → monitor` is a legal §2
edge and the run-safety wiring uses exactly three of them. It is now policed by a STRICTER
oracle with an exact allowlist rather than a ban:
`tests/monitor/test_monitor_census.py::test_train_to_monitor_import_sites_are_exactly_the_pinned_set`
(O-19: {coordinator/step.py, subsystems.py, lifecycle/heartbeat_watchdog.py} — a fourth
site fails, and so does a missing one). The `mantis.eval` ban is untouched (L-A, O-23).
"""
from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN = ("mantis.eval", "mantis.arena")
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
        "train/* must not top-level import eval/arena (reach them via injected "
        "callables / the EventSink Protocol; the monitor edge is allowlisted by O-19):\n  "
        + "\n  ".join(violations)
    )
