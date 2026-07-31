"""⊕ WPMAIN — the one-authority census at the PROPERTY, not the spelling (RED-TEAM RT-1).

`tests/test_run_one_authority.py` is BYTE-FROZEN from `7c28536` and its instrument
(`_call_sites`) matches `ast.Call` nodes by the name **as spelled at the call site**. RED-TEAM
drove that blind spot to a green tier:

| variant | shape | result |
|---|---|---|
| A1 | new module under `src/mantis/`, `from mantis.run import compose_run as _go`, call `_go(...)` | 23/23 authority rows GREEN |
| A2 | module-level rebind `_go = compose_run`, call `_go(...)` | 10/10 GREEN |
| A3 | same module, UNALIASED `compose_run(...)` | RED (the control — the census works on the spelling) |
| A4/A5 | new module calling **`launch_run`** — censused NOWHERE — with an `os.environ` device override | **full tier 2278 passed / 2 skipped, all 8 gates rc 0, lint GREEN** |

A4/A5 is the fourth instance of the census-gap species (R128's standing law, on the BINDING
axis): `compose_run_v2` was the rename, `main()` was the uncensused body, `-m mantis.run` was
the invocation shape, and this is a census that names a SYMBOL where the claim is about a
PROPERTY. A second boot path in `src/mantis/` calls no censused symbol directly, is not
`main`, and ships green.

**This file closes the whole family with one property**: to call ANY of `mantis.run`'s boot
functions — under any alias, any rebind, any spelling — a module must first NAME `mantis.run`
in an import. So the census is over IMPORTS, and aliasing is structurally irrelevant to it.

It is also the missing producer for a structural claim the tree has asserted in prose since
WPMAIN and nowhere else: `docs/design/repo_design.md`'s *"NOTHING imports `mantis.run` — it is
a source-only DAG node"*, re-stated in `run.py`'s own module docstring. RED-TEAM measured gate
9 (`check_import_dag.py`) rc 0 with a probe module importing `mantis.run`: gate 9 checks
CYCLES, not this. A claimed structural property with no producer is R4/LAW-07's exact shape,
and this file is the row that closes it.

Why a NEW file: R43 queues a frozen-oracle edit REGARDLESS OF DIRECTION, and the strengthening
this needs is inside `test_run_one_authority.py`. R5 bars cross-test imports, so the ~35 lines
of AST helper below are re-derived rather than shared — disclosed here, and self-checking in
the one way that matters: both copies parse the same shipped tree.

Fakes: NONE. Every assertion is a static census over the shipped `src/` + `tools/`.
"""
from __future__ import annotations

import ast
from pathlib import Path

from mantis.run import compose_run, launch_run  # noqa: F401  (the live objects, RED anchor)

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
_TOOLS = _REPO / "tools"
_RUN_MODULE = "mantis.run"

#: The ONLY shipped module that may bind anything out of `mantis.run` — the preflight child,
#: whose function-local `from mantis.run import build_run_collaborators, compose_run`
#: (DESIGN §1.4, kept function-local so the tool stays importable without torch) is the whole
#: of success criterion 2. `src/mantis/run.py` itself is not here because it does not import
#: itself; a self-import would be a new shape and must be argued, not typed.
_SANCTIONED_IMPORTERS = {"tools/ci_gates/preflight_mint.py"}

#: `launch_run` — the symbol RED-TEAM found censused NOWHERE — and its one production caller.
_LAUNCH_SITES = {"src/mantis/run.py::main"}

#: The boot symbols a module-level rebind could re-spell inside the two sanctioned files.
_BOOT_SYMBOLS = frozenset({"build_run_collaborators", "compose_run", "launch_run", "main"})

#: Call shapes that can import by STRING and so evade an `Import`/`ImportFrom` census.
_DYNAMIC_IMPORTERS = frozenset({"import_module", "__import__", "find_spec", "load_module"})


def _production_sources() -> list[Path]:
    """Every shipped `.py` under `src/` and `tools/`. `tests/` is deliberately OUT: a test
    may import and compose freely — the one-authority law is about what SHIPS."""
    return sorted([*_SRC.rglob("*.py"), *_TOOLS.rglob("*.py")])


def _rel(path: Path) -> str:
    return str(path.relative_to(_REPO))


def _enclosing_defs(tree: ast.AST) -> dict[ast.AST, str]:
    owner: dict[ast.AST, str] = {}

    def walk(node: ast.AST, name: str) -> None:
        for child in ast.iter_child_nodes(node):
            child_name = child.name if isinstance(
                child, ast.FunctionDef | ast.AsyncFunctionDef) else name
            owner[child] = child_name
            walk(child, child_name)

    walk(tree, "<module>")
    return owner


def _called_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _own_package(path: Path) -> str | None:
    """The dotted package a module's RELATIVE imports resolve against, or `None` when the
    file is not inside an installed package (everything under `tools/`)."""
    if _SRC not in path.parents:
        return None
    # Drop the module's own last component: for `mantis/train/loop.py` that leaves
    # `mantis.train`, and for `mantis/train/__init__.py` it leaves `mantis.train` too —
    # a package's `__init__` resolves relatives against the package itself.
    parts = list(path.relative_to(_SRC).with_suffix("").parts)[:-1]
    return ".".join(parts)


def _absolute_module(node: ast.ImportFrom, path: Path) -> str | None:
    """`from ..run import x` inside `mantis/train/foo.py` -> `mantis.run`.

    Relative imports are the guise a name census over `"mantis.run"` misses entirely, and
    every module under `src/mantis/` can reach the root with one — which is exactly the
    place a second boot path would be written."""
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

    Four shapes, because a census that reads only `ImportFrom(module="mantis.run")` is an
    instrument that cannot support the negative it asserts (R128's standing law applied to
    this census's own method)."""
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
        elif isinstance(node, ast.Call) and _called_name(node) in _DYNAMIC_IMPORTERS:
            for argument in [*node.args, *[kw.value for kw in node.keywords]]:
                if isinstance(argument, ast.Constant) and argument.value == _RUN_MODULE:
                    reasons.append(f"{_called_name(node)}({_RUN_MODULE!r})")
    return reasons


def _call_sites(symbol: str) -> set[str]:
    sites: set[str] = set()
    for path in _production_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        owner = _enclosing_defs(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _called_name(node) == symbol:
                sites.add(f"{_rel(path)}::{owner.get(node, '<module>')}")
    return sites


# ══ the property: only the sanctioned module may NAME the composition root ════════════
def test_no_shipped_module_but_the_preflight_child_imports_the_composition_root() -> None:
    """RT-1's one-assertion closure, and `repo_design.md`'s missing producer.

    Alias-proof by construction: `from mantis.run import compose_run as _go` (A1),
    `_go = compose_run` after any import (A2) and a module calling `launch_run` (A4/A5) all
    require the importing module to appear here, whatever it calls the symbols afterwards.
    A second boot path that does NOT import `mantis.run` cannot call its functions at all.

    MUTATION THAT REDS IT (driven — this is RED-TEAM's own A4/A5, restored verbatim): add
    `src/mantis/_rt_probe_launcher.py` containing `from mantis.run import launch_run` and a
    function that loads a config, re-points `train.device` from `os.environ` through
    `model_dump`/`model_validate`, and launches. That module was green through the FULL tier
    and all eight gates; the set below names it.

    The four relative/dynamic shapes are covered too, and each is the cheap way past a naive
    version of this census: `from ..run import compose_run` from anywhere under
    `src/mantis/`, `from mantis import run`, `import mantis.run as _r`, and
    `importlib.import_module("mantis.run")`."""
    importers = {
        _rel(path): reasons
        for path in _production_sources()
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
    """The LAW-07 half: a producer test is only a producer while the claim it produces for
    still exists and still says what the producer proves.

    MUTATION THAT REDS IT: delete or reword `repo_design.md`'s source-only-node sentence
    while leaving this file green — a producer for a claim nobody makes any more, which is
    the phantom-gate class inverted (LAW-07's own origin, F-10). This is deliberately a
    SUBSTRING check on the claim's load-bearing words, not a line number: the paragraph is
    reflowed by every design amendment."""
    design = (_REPO / "docs" / "design" / "repo_design.md").read_text(encoding="utf-8")
    assert "NOTHING imports" in design and "`mantis.run`" in design, (
        "docs/design/repo_design.md no longer states the source-only-DAG-node property this "
        "file is the producer for; the claim and its producer move together (R9: an "
        "amendment commit, never silent drift)"
    )


# ══ `launch_run` joins the call-site census it was never on ═══════════════════════════
def test_launch_run_has_exactly_one_production_call_site() -> None:
    """RT-1's second half. O-A1 censuses `compose_run`, `build_run_collaborators` and the
    five irreducible steps — and NOT `launch_run`, so RED-TEAM's second launcher needed no
    alias at all to stay invisible.

    MUTATION THAT REDS IT: any second production caller of `launch_run` — a `mantis.deploy`
    entry point, a `tools/` convenience wrapper, a resume shim. Each is a boot posture that
    can drift from run5's, which is the failure this WP is named for; adding one is a design
    decision that edits this set, never an edit that happens to pass.

    Note what this row does NOT claim: it is name-keyed, exactly like the frozen census, and
    an aliased `launch_run as _go` walks past it. The alias-proof property is the import
    census above; this row is the readable instance."""
    assert _call_sites("launch_run") == _LAUNCH_SITES, (
        "`launch_run` is called by `main` and by nothing else — the preflight child calls "
        "the two functions BENEATH it (build + compose) so its two sanctioned instruments "
        f"can sit between them; got {sorted(_call_sites('launch_run'))}"
    )


def test_no_sanctioned_module_re_spells_a_boot_symbol_through_a_module_level_rebind() -> None:
    """A2's in-module residue. The import census stops a module that has to IMPORT the root;
    inside the two files that legitimately name it, a module-level `_go = compose_run`
    followed by `_go(...)` would still walk past every call-site census, frozen or not.

    MUTATION THAT REDS IT: write that rebind in either sanctioned file. There is no
    legitimate use — an alias for a function called once in the file it is defined in is a
    re-spelling and nothing else."""
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
