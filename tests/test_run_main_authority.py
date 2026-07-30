"""⊕ WPMAIN — `main()`'s body census: the O-A2 twin the ORACLE-WRITE set left open.

REVIEW-impl condition C-5 / finding F-5. `tests/test_run_one_authority.py` applies its
`_body_without_docstring` instrument to `launch_run` (O-A2, `:274`) and to the preflight
child's `_boot_main` (O-A4, `:443`) — and to nothing else. `main`, the launcher an operator
actually types (`python -m mantis.run --config <path> --out-dir <path>`), had NO body census
at all, so this three-line edit was green through the whole tree:

    config = load_config(args.config)
    config.train.device = os.environ.get("MANTIS_DEVICE", config.train.device)
    handles = launch_run(config=config, out_dir=args.out_dir)

Green through O-A1 (one composer, two call sites — unchanged), O-A2 (`launch_run`'s body is
untouched), O-A3/O-A4 (builder and child untouched), O-A5 (no `or` behind a config fact, and
an `os.environ.get` roots at `os`, not at `config`), O-B2 (the flag surface is unchanged — no
`--device` comes back), and `compose_run`'s `revalidate_run_config` (a schema-valid `"cpu"`
re-validates clean). That is precisely the device false-clear R126 exists to make
unrepresentable — one level ABOVE the layer R126 closed — and `run.py:641-644` claims it is
already closed ("no invocation can point either caller somewhere else"). True today,
structurally undefended until this file.

O-A2's own docstring supplies the argument verbatim: *"a launcher that 'adjusts' the config
before composing is a divergent path wearing the one-authority name."* `main` IS a launcher.
This file is its missing twin.

Why a NEW file and not two more lines inside `test_run_one_authority.py`: that file is
BYTE-FROZEN from the ORACLE-WRITE commit `7c28536`, and R43 queues a frozen-oracle edit
REGARDLESS OF DIRECTION — a strengthening edit queues too. The one grant spent on this branch
is R129's, on `tests/test_run_launcher.py`. The price of the freeze is the ~40 lines of AST
instrument re-derived below (R5 bars cross-test imports); it is disclosed here rather than
hidden, and the duplication is self-checking in the one way that matters — both copies parse
the SAME `src/mantis/run.py`, so a divergence between them shows up as one file red and the
other green on the same tree.

Fakes: NONE. Every assertion is a static census over the shipped `src/mantis/run.py`.
"""
from __future__ import annotations

import ast
from pathlib import Path

from mantis.run import launch_run, main  # noqa: F401  (the live objects the census is about)

_REPO = Path(__file__).resolve().parents[1]
_RUN_PY = _REPO / "src" / "mantis" / "run.py"

#: The ONE loader (`mantis.config.loader.load_config`) and the ONE launch path. `main` is
#: allowed to name exactly these two and to put nothing between them.
_LOADER = "load_config"
_LAUNCHER = "launch_run"

#: Names a config-carrying local may not be called under any circumstances without being the
#: loader's own direct result. These are the two the tree already uses for a `RunConfig`
#: (`config` in `run.py`, `cfg` in the resolvers), and O-A5 bans `or`/`.get` fallbacks behind
#: the same two roots — the ban here is the ASSIGNMENT half of that same posture.
_CONFIG_ROOTS = ("config", "cfg")


def _tree() -> ast.Module:
    return ast.parse(_RUN_PY.read_text(encoding="utf-8"))


def _func(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"no `def {name}` found in src/mantis/run.py")


def _called_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _root_name(node: ast.AST) -> str | None:
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


def _binding_targets(fn: ast.FunctionDef) -> list[tuple[ast.AST, ast.stmt]]:
    """Every node `fn` can BIND to, paired with the statement that binds it.

    Deliberately wider than `ast.Assign`: an augmented assignment, an annotated assignment, a
    walrus, a loop variable and a `with ... as` are all writes, and a census that only reads
    `ast.Assign` is an instrument that cannot support the negative it asserts (R128's standing
    law, applied to this census's own method)."""
    found: list[tuple[ast.AST, ast.stmt]] = []

    def add(target: ast.AST, stmt: ast.stmt) -> None:
        if isinstance(target, ast.Tuple | ast.List):
            for element in target.elts:
                add(element, stmt)
        elif isinstance(target, ast.Starred):
            add(target.value, stmt)
        else:
            found.append((target, stmt))

    for stmt in ast.walk(fn):
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                add(target, stmt)
        elif isinstance(stmt, ast.AugAssign | ast.AnnAssign):
            add(stmt.target, stmt)
        elif isinstance(stmt, ast.NamedExpr):
            add(stmt.target, stmt)
        elif isinstance(stmt, ast.For | ast.AsyncFor):
            add(stmt.target, stmt)
        elif isinstance(stmt, ast.With | ast.AsyncWith):
            for item in stmt.items:
                if item.optional_vars is not None:
                    add(item.optional_vars, stmt)
    return found


def _launch_call(fn: ast.FunctionDef) -> ast.Call:
    calls = [node for node in ast.walk(fn)
             if isinstance(node, ast.Call) and _called_name(node) == _LAUNCHER]
    assert len(calls) == 1, (
        f"`main` must call {_LAUNCHER}() exactly once — a second launch in the entry point is "
        f"a second boot with the first one's rc thrown away; found {len(calls)}"
    )
    return calls[0]


def _config_binding(fn: ast.FunctionDef) -> tuple[ast.expr, str | None, ast.stmt | None]:
    """What `main` hands to the launcher: the expression, the local name carrying it (or
    `None` when the loader's result is forwarded inline), and the statement that bound it."""
    kwargs = {kw.arg: kw.value for kw in _launch_call(fn).keywords}
    node = kwargs.get("config")
    assert node is not None, (
        f"{_LAUNCHER}(config=...) must be passed by KEYWORD at the seam this census reads "
        f"(the same posture O-A2 pins one layer down); got {sorted(kwargs)}"
    )
    if isinstance(node, ast.Call):
        return node, None, None
    assert isinstance(node, ast.Name), (
        "config= must be either the loader's own call or a plain local name bound from it — "
        f"an expression here is the transform this census exists to forbid; got "
        f"{ast.dump(node)[:160]}"
    )
    binds = [stmt for target, stmt in _binding_targets(fn)
             if isinstance(target, ast.Name) and target.id == node.id]
    assert len(binds) == 1, (
        f"the config local {node.id!r} must be bound EXACTLY once in `main`; found "
        f"{len(binds)} bindings — a rebind is a second config the launcher never saw"
    )
    return node, node.id, binds[0]


# ══ the loaded config reaches the launcher UNMODIFIED ═════════════════════════════════
def test_main_hands_the_launcher_the_loaders_own_result_and_nothing_else() -> None:
    """The O-A2 twin, flow half. What `main` passes as `config=` must be the direct result of
    `load_config(...)` — inline, or via a local bound once from it and used nowhere else.

    MUTATION THAT REDS IT: `launch_run(config=_adjust(load_config(args.config)), ...)`, or a
    local that is loaded, touched and then launched. Both keep every other census in the tree
    green (§F-5), and both are a boot the one-authority property does not cover."""
    fn = _func(_tree(), "main")
    node, name, binding = _config_binding(fn)

    value = node if binding is None else getattr(binding, "value", None)
    assert isinstance(value, ast.Call) and _called_name(value) == _LOADER, (
        f"`main` must launch what `{_LOADER}()` returned, with nothing between the load and "
        f"the launch (R126: the device — and every other config fact — is the CONFIG's, and "
        f"the entry point may not re-decide it); got {ast.dump(value)[:160]}"
    )

    if name is not None:
        uses = [n for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id == name]
        assert len(uses) == 2, (
            f"the config local {name!r} may appear exactly twice in `main` — once bound from "
            f"{_LOADER}(), once forwarded to {_LAUNCHER}(config=). A third occurrence is a "
            f"read, a mutation or a hand-off this census cannot vouch for; found {len(uses)}"
        )


# ══ nothing in `main` writes to the config it launches with ═══════════════════════════
def test_nothing_in_main_assigns_onto_the_config_it_launches_with() -> None:
    """The O-A2 twin, mutation half — and the one that reds F-5's exact three-line edit.

    MUTATION THAT REDS IT (driven, REVIEW-impl F-5): insert
    `config.train.device = os.environ.get("MANTIS_DEVICE", config.train.device)` between the
    load and the launch. Every existing oracle stays green; this assertion does not."""
    fn = _func(_tree(), "main")
    _, name, binding = _config_binding(fn)
    banned = {*_CONFIG_ROOTS, *([name] if name else [])}

    for target, stmt in _binding_targets(fn):
        root = _root_name(target)
        if root not in banned:
            continue
        assert stmt is binding and isinstance(target, ast.Name), (
            f"`main` writes to {root!r} at `{ast.dump(target)[:80]}`. The entry point loads "
            "the config and launches it; it may not ADJUST it first. A device coercion, an "
            "env-var override or a resume flag written here is the divergent boot path "
            "wearing the one-authority name (O-A2's argument, applied to the launcher an "
            "operator actually types) — and it is invisible to every other census in the tree"
        )


# ══ …nor re-points the parsed CLI inputs ══════════════════════════════════════════════
def test_main_does_not_re_point_the_arguments_it_parsed() -> None:
    """The same class one step earlier: mutating `args.config` chooses a DIFFERENT config file
    rather than adjusting the loaded one, and `run.py:641-644` claims neither is possible ("no
    invocation can point either caller somewhere else").

    MUTATION THAT REDS IT: `args.config = os.environ.get("MANTIS_CONFIG", args.config)` after
    `parse_args`. argparse's own required-flag census (`test_run_launcher.py:100-133`) reads
    the PARSER, so it sees nothing at all."""
    fn = _func(_tree(), "main")
    parsed = [(target, stmt) for target, stmt in _binding_targets(fn)
              if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call)
              and _called_name(stmt.value) == "parse_args"]
    assert len(parsed) == 1 and isinstance(parsed[0][0], ast.Name), (
        "`main` must bind `parse_args(...)` exactly once, to a plain name; found "
        f"{len(parsed)} bindings"
    )
    args_name = parsed[0][0].id

    for target, stmt in _binding_targets(fn):
        if _root_name(target) != args_name:
            continue
        assert stmt is parsed[0][1], (
            f"nothing may write to `{args_name}` after argparse produced it: re-pointing "
            "`--config` or `--out-dir` inside the launcher is the same false-clear class as "
            "adjusting the loaded config, one step earlier in the same function"
        )
