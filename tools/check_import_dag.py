"""Import-DAG check (CI gate 9): top-level import cycles fail the build.

CLI: python tools/check_import_dag.py [PACKAGE_ROOT]   (default src/mantis; package
name = basename of PACKAGE_ROOT). Exit 0 = acyclic; 1 = cycle found (each cycle printed
as `CYCLE: a -> b -> a`); 2 = usage error or unparseable file (parse failures are loud,
never skipped).

Algorithm (repo_design §2): AST scan of module TOP-LEVEL imports only. Function/method/
class-body imports are lazy by definition and are NOT edges ("top-level imports only;
lazy imports need a stated reason"); imports nested under top-level `if` (e.g.
TYPE_CHECKING) are also not edges. Two digraphs are checked: module-level (file -> file)
and condensed first-level-subpackage (self-edges dropped); a cycle in EITHER exits 1 —
the module graph catches intra-subpackage knots, the condensed graph catches the
historical training<->eval-class cycles even when no single module pair cycles.
"""
import ast
import sys
from pathlib import Path


def _module_name(pkg: str, root: Path, path: Path) -> str:
    parts = list(path.relative_to(root).parts)
    parts[-1] = parts[-1][: -len(".py")]
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join([pkg, *parts]) if parts else pkg


def _import_targets(mod: str, is_pkg: bool, node: ast.Import | ast.ImportFrom) -> list[str]:
    """Dotted-name targets of one top-level import statement (pre-resolution)."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if node.level == 0:
        base = node.module or ""
    else:
        parts = mod.split(".")
        if not is_pkg:
            parts = parts[:-1]
        strip = node.level - 1
        if strip >= len(parts):
            return []
        if strip:
            parts = parts[:-strip]
        base = ".".join(parts)
        if node.module:
            base = f"{base}.{node.module}" if base else node.module
    return [f"{base}.{alias.name}" if base else alias.name for alias in node.names]


def _resolve(target: str, modules: dict[str, Path]) -> str | None:
    """Nearest existing module node for a dotted target (pkg.a.b else its package)."""
    parts = target.split(".")
    while parts:
        cand = ".".join(parts)
        if cand in modules:
            return cand
        parts.pop()
    return None


def _find_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    """Iterative three-color DFS; returns every back-edge cycle path."""
    color = dict.fromkeys(graph, 0)  # 0=white 1=gray 2=black
    cycles: list[list[str]] = []
    for start in sorted(graph):
        if color[start]:
            continue
        color[start] = 1
        path = [start]
        iters = [iter(sorted(graph[start]))]
        while iters:
            nxt = next(iters[-1], None)
            if nxt is None:
                color[path.pop()] = 2
                iters.pop()
                continue
            if nxt not in color:
                continue
            if color[nxt] == 1:
                cycles.append(path[path.index(nxt):] + [nxt])
            elif color[nxt] == 0:
                color[nxt] = 1
                path.append(nxt)
                iters.append(iter(sorted(graph[nxt])))
    return cycles


def main(argv: list[str]) -> int:
    if len(argv) > 2:
        print("usage: check_import_dag.py [PACKAGE_ROOT]", file=sys.stderr)
        return 2
    root = Path(argv[1]) if len(argv) == 2 else Path("src/mantis")
    if not root.is_dir():
        print(f"usage error: package root is not a directory: {root}", file=sys.stderr)
        return 2
    pkg = root.resolve().name

    modules: dict[str, Path] = {
        _module_name(pkg, root, f): f for f in sorted(root.rglob("*.py"))
    }
    is_package = {m: p.name == "__init__.py" for m, p in modules.items()}

    parse_failed = False
    module_graph: dict[str, set[str]] = {m: set() for m in modules}
    for mod, path in modules.items():
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError as exc:
            print(f"PARSE ERROR: {path}:{exc.lineno}: {exc.msg}", file=sys.stderr)
            parse_failed = True
            continue
        for node in tree.body:  # module top level ONLY
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            for target in _import_targets(mod, is_package[mod], node):
                if target != pkg and not target.startswith(pkg + "."):
                    continue
                resolved = _resolve(target, modules)
                if resolved is not None and resolved != mod:
                    module_graph[mod].add(resolved)
    if parse_failed:
        return 2

    def condense(name: str) -> str:
        parts = name.split(".")
        return ".".join(parts[:2]) if len(parts) >= 2 else pkg

    condensed_graph: dict[str, set[str]] = {condense(m): set() for m in modules}
    for src, dsts in module_graph.items():
        for dst in dsts:
            a, b = condense(src), condense(dst)
            if a != b:
                condensed_graph[a].add(b)

    seen: set[str] = set()
    for graph in (module_graph, condensed_graph):
        for cycle in _find_cycles(graph):
            line = "CYCLE: " + " -> ".join(cycle)
            if line not in seen:
                seen.add(line)
                print(line)
    if seen:
        return 1
    print(f"OK: no import cycles ({len(modules)} modules)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
