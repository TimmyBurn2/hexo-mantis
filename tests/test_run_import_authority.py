"""The one-authority census at the PROPERTY, not the spelling.

A census keyed on the name as SPELLED is walked past by an alias, a module-level rebind, or a
new module calling an uncensused boot symbol; all three shipped green through the full tier and
every gate. To call any of `mantis.run`'s boot functions under any spelling a module must first
NAME `mantis.run` in an import, so this census is over IMPORTS and aliasing cannot evade it. It
is also the producer for `repo_design.md`'s "NOTHING imports `mantis.run`" claim, which gate 9
does not check — it checks cycles.

Fakes: NONE. Every assertion is a static census over the shipped `src/` + `tools/`.
"""
from __future__ import annotations

import ast
from pathlib import Path

from mantis.run import compose_run, launch_run  # noqa: F401  (the live objects under census)
from _ast_census import call_sites, called_name, production_sources, rel

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
_TOOLS = _REPO / "tools"
_RUN_MODULE = "mantis.run"

#: The ONLY shipped module that may bind anything out of `mantis.run` — the preflight child,
#: whose import is function-local so the tool stays importable without torch.
_SANCTIONED_IMPORTERS = {"tools/ci_gates/preflight_mint.py"}

#: `launch_run` — the boot symbol no call-site census named — and its one production caller.
_LAUNCH_SITES = {"src/mantis/run.py::main"}

#: The boot symbols a module-level rebind could re-spell inside the two sanctioned files.
_BOOT_SYMBOLS = frozenset({"build_run_collaborators", "compose_run", "launch_run", "main"})

#: Call shapes that can import by STRING and so evade an `Import`/`ImportFrom` census.
_DYNAMIC_IMPORTERS = frozenset({"import_module", "__import__", "find_spec", "load_module"})


def _own_package(path: Path) -> str | None:
    """The dotted package a module's RELATIVE imports resolve against, or `None` when the file
    is not inside an installed package (everything under `tools/`)."""
    if _SRC not in path.parents:
        return None
    # The module's own last component is dropped: a package's `__init__` resolves relatives
    # against the package itself, so it and a sibling module yield the same package.
    parts = list(path.relative_to(_SRC).with_suffix("").parts)[:-1]
    return ".".join(parts)


def _absolute_module(node: ast.ImportFrom, path: Path) -> str | None:
    """`from ..run import x` inside `mantis/train/foo.py` -> `mantis.run`; relative imports are
    the guise a name census over `"mantis.run"` misses entirely."""
    if node.level == 0:
        return node.module
    package = _own_package(path)
    if package is None:
        return None
    parts = package.split(".") if package else []
    ascend = node.level - 1
    base = parts[:len(parts) - ascend] if ascend else parts
    return ".".join([*base, *([node.module] if node.module else [])])


def _binds_the_run_module(tree: ast.AST, path: Path) -> list[str]:
    """Every way this module could get its hands on `mantis.run`, as human-readable reasons.

    Four shapes: reading only `ImportFrom(module="mantis.run")` cannot support the negative."""
    reasons: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == _RUN_MODULE or alias.name.startswith(f"{_RUN_MODULE}."):
                    reasons.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = _absolute_module(node, path)
            if module == _RUN_MODULE or (module or "").startswith(f"{_RUN_MODULE}."):
                reasons.append(f"from {module} import "
                               f"{', '.join(alias.name for alias in node.names)}")
            elif module == "mantis":
                for alias in node.names:
                    if alias.name == "run":
                        reasons.append("from mantis import run")
        elif isinstance(node, ast.Call) and called_name(node) in _DYNAMIC_IMPORTERS:
            for argument in [*node.args, *[kw.value for kw in node.keywords]]:
                if isinstance(argument, ast.Constant) and argument.value == _RUN_MODULE:
                    reasons.append(f"{called_name(node)}({_RUN_MODULE!r})")
    return reasons


def test_no_shipped_module_but_the_preflight_child_imports_the_composition_root() -> None:
    """Only the preflight child may import `mantis.run`.

    Alias-proof by construction: any second boot path must NAME the module first, whatever it
    calls the symbols afterwards, and the relative and dynamic import shapes are covered too."""
    importers = {
        rel(path, _REPO): reasons
        for path in production_sources(_SRC, _TOOLS)
        if (reasons := _binds_the_run_module(
            ast.parse(path.read_text(encoding="utf-8")), path))
    }
    assert set(importers) == _SANCTIONED_IMPORTERS, (
        "`mantis.run` is a SOURCE-ONLY DAG node (docs/design/repo_design.md; `run.py`'s own "
        "module docstring): the ONLY shipped module that may name it is the preflight child, "
        "which binds the SAME two functions the launcher calls. Any other importer is a "
        "SECOND BOOT PATH — it can compose a run from a config nobody typed, and it is "
        "invisible to every name-keyed census in the tree (RED-TEAM RT-1, variants A1/A2/"
        f"A4/A5). Got { {k: v for k, v in sorted(importers.items())} }"
    )


def test_the_source_only_node_claim_is_still_the_one_this_census_produces_for() -> None:
    """The design doc still states the source-only-node claim this file produces for; a
    substring check, not a line number, because the paragraph is reflowed by every amendment."""
    design = (_REPO / "docs" / "design" / "repo_design.md").read_text(encoding="utf-8")
    assert "NOTHING imports" in design and "`mantis.run`" in design, (
        "docs/design/repo_design.md no longer states the source-only-DAG-node property this "
        "file is the producer for; the claim and its producer move together (R9: an "
        "amendment commit, never silent drift)"
    )


def test_launch_run_has_exactly_one_production_call_site() -> None:
    """`launch_run` has exactly one production call site. Name-keyed, so an aliased
    `launch_run as _go` walks past it — the alias-proof property is the import census above."""
    assert call_sites("launch_run", roots=(_SRC, _TOOLS), repo=_REPO) == _LAUNCH_SITES, (
        "`launch_run` is called by `main` and by nothing else — the preflight child calls "
        "the two functions BENEATH it (build + compose) so its two sanctioned instruments "
        f"can sit between them; got {sorted(call_sites('launch_run', roots=(_SRC, _TOOLS), repo=_REPO))}"
    )


def test_no_sanctioned_module_re_spells_a_boot_symbol_through_a_module_level_rebind() -> None:
    """Neither sanctioned file may re-spell a boot symbol through a module-level rebind: every
    call-site census in the tree reads the SPELLING, so a rebind is a free second call site."""
    offenders: list[str] = []
    for name in sorted({*_SANCTIONED_IMPORTERS, "src/mantis/run.py"}):
        tree = ast.parse((_REPO / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Name):
                continue
            if node.value.id in _BOOT_SYMBOLS:
                targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
                offenders.append(f"{name}: {targets} = {node.value.id}")
    assert not offenders, (
        "a boot symbol was re-bound to a second name; every call-site census in the tree "
        f"reads the SPELLING, so the rebind is a free second call site: {offenders}"
    )
