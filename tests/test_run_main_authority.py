# >300 justify (R8): four rows of ONE census family over ONE parse of `src/mantis/run.py`,
# sharing the AST instrument that makes the claim checkable. R5 bars cross-test imports, so a
# split forks that instrument into copies that drift apart while both stay green.
"""`main()`'s body census — the twin of the `launch_run` body oracle.

`main`, the launcher an operator actually types, had no body census at all, so inserting a
device coercion between the load and the launch was green through every other oracle in the
tree. These rows pin that `main` launches the loader's own result unmodified, writes to neither
the config nor the parsed namespace, and reads both run inputs off the arguments it parsed.

Fakes: NONE. Every assertion is a static census over the shipped `src/mantis/run.py`.
"""
from __future__ import annotations

import ast
from pathlib import Path

from mantis.run import launch_run, main  # noqa: F401  (the live objects the census is about)

_REPO = Path(__file__).resolve().parents[1]
_RUN_PY = _REPO / "src" / "mantis" / "run.py"

#: The ONE loader and the ONE launch path; `main` may name exactly these two, with nothing
#: between them.
_LOADER = "load_config"
_LAUNCHER = "launch_run"

#: Names a config-carrying local may not take unless it is the loader's own direct result.
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

    Wider than `ast.Assign` on purpose: augmented and annotated assignment, walrus, loop
    variable and `with ... as` are all writes a narrower census could not support."""
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
    """What `main` hands the launcher: the expression, the local carrying it (`None` when the
    loader's result is forwarded inline), and the statement that bound it."""
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


def test_main_hands_the_launcher_the_loaders_own_result_and_nothing_else() -> None:
    """What `main` passes as `config=` must be `load_config(...)`'s direct result.

    MUTATION THAT REDS IT: `config=_adjust(load_config(args.config))`, or a local that is
    loaded, touched, then launched — both keep every other census green."""
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


def test_nothing_in_main_assigns_onto_the_config_it_launches_with() -> None:
    """Nothing in `main` may write to the config it launches with.

    MUTATION THAT REDS IT: `config.train.device = os.environ.get("MANTIS_DEVICE", …)` between
    the load and the launch — every other oracle in the tree stays green."""
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


def _parsed_args_name(fn: ast.FunctionDef) -> tuple[str, ast.stmt]:
    """The local `main` binds `parse_args(...)` to, and the statement that binds it."""
    parsed = [(target, stmt) for target, stmt in _binding_targets(fn)
              if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call)
              and _called_name(stmt.value) == "parse_args"]
    assert len(parsed) == 1 and isinstance(parsed[0][0], ast.Name), (
        "`main` must bind `parse_args(...)` exactly once, to a plain name; found "
        f"{len(parsed)} bindings"
    )
    return parsed[0][0].id, parsed[0][1]


def _sole_argument(call: ast.Call, *, label: str) -> ast.expr:
    """The ONE value a single-input call was handed, positional or keyword."""
    values = [*call.args, *[kw.value for kw in call.keywords]]
    assert len(values) == 1, (
        f"{label} must be called with exactly one argument at the seam this census reads; "
        f"got {len(values)}"
    )
    return values[0]


def test_main_reads_both_run_inputs_off_the_arguments_it_parsed() -> None:
    """Both run inputs must be attribute reads off the `parse_args` binding.

    The object-level rows leave the loader's own ARGUMENT and `out_dir=` unread, so an
    env-defaulted path boots a config nobody typed while every census stays green.
    MUTATION THAT REDS IT: `os.environ.get("MANTIS_CONFIG", args.config)` inside
    `load_config(...)`, or an env-defaulted `out_dir=`. The instrument is SHAPE, not spelling,
    so any wrapper fails it whatever it is named."""
    fn = _func(_tree(), "main")
    args_name, _ = _parsed_args_name(fn)

    loads = [node for node in ast.walk(fn)
             if isinstance(node, ast.Call) and _called_name(node) == _LOADER]
    assert len(loads) == 1, (
        f"`main` must call {_LOADER}() exactly once; found {len(loads)} — two loads is two "
        "configs and the launcher only ever sees one of them"
    )
    loaded_path = _sole_argument(loads[0], label=f"{_LOADER}()")
    assert (isinstance(loaded_path, ast.Attribute)
            and isinstance(loaded_path.value, ast.Name)
            and loaded_path.value.id == args_name), (
        f"the path handed to {_LOADER}() must be `{args_name}.<flag>` — the value argparse "
        "produced, read directly. Anything wrapped around it is a route to a file the "
        f"operator did not type; got {ast.dump(loaded_path)[:160]}"
    )

    out_dir = {kw.arg: kw.value for kw in _launch_call(fn).keywords}.get("out_dir")
    assert out_dir is not None, (
        "`launch_run(out_dir=...)` must be passed by KEYWORD at the seam this census reads"
    )
    assert (isinstance(out_dir, ast.Attribute)
            and isinstance(out_dir.value, ast.Name)
            and out_dir.value.id == args_name), (
        f"`out_dir=` must be `{args_name}.<flag>` too — a run that writes its logs and "
        "checkpoints somewhere other than the directory named on the command line is the "
        f"same false-clear class as loading a different config; got {ast.dump(out_dir)[:160]}"
    )


def test_main_does_not_re_point_the_arguments_it_parsed() -> None:
    """Nothing may re-point `args.config` / `args.out_dir` after argparse produced them.

    MUTATION THAT REDS IT: `args.config = os.environ.get("MANTIS_CONFIG", args.config)` after
    `parse_args` — argparse's own required-flag census reads the PARSER and sees nothing."""
    fn = _func(_tree(), "main")
    args_name, parse_stmt = _parsed_args_name(fn)

    for target, stmt in _binding_targets(fn):
        if _root_name(target) != args_name:
            continue
        assert stmt is parse_stmt, (
            f"nothing may write to `{args_name}` after argparse produced it: re-pointing "
            "`--config` or `--out-dir` inside the launcher is the same false-clear class as "
            "adjusting the loaded config, one step earlier in the same function"
        )
