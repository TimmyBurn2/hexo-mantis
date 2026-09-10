# >300 justify (R8): O-A1..O-A5 are ONE census family making ONE claim — that exactly one
# composition path exists — and they share the instrument that makes it checkable. R5 bars
# cross-test imports, so a split forks that helper set, and two instruments drift apart while
# both stay green. The src-side census (O-A2) and the child-side census (O-A4) are twins and
# have to be readable side by side.
"""THE ONE COMPOSITION AUTHORITY — oracles O-A1..O-A5.

What this file exists to stop: two boot paths. `python -m mantis.run` used to validate a config
and exit while a CI tool owned the only real composition, so "the preflight boots what run5
boots" was a claim with no producer on either side.

O-A1 refuses a second composer or builder anywhere in `src/`/`tools/` and a child that binds its
composer from elsewhere. O-A2 refuses `launch_run` growing a third step or transforming what it
forwards. O-A3 refuses the builder that stops BUILDING — a token census was MEASURED
insufficient (a silent-default mutation left 1773 tests green), so it asserts a bound CALL and
is paired with behavioural drives. O-A4 is the CHILD side, which had no producer at all. O-A5
catches the smuggled default in its one uncensused guise, `config.run_id or "run"`.

Fakes: NONE — static censuses over the shipped source, plus one identity read of live modules.
"""
from __future__ import annotations

import ast
import sys
import tokenize
from pathlib import Path

# RED-at-import anchor: `build_run_collaborators` / `launch_run` do not exist yet.
from mantis.run import build_run_collaborators, compose_run, launch_run  # noqa: F401

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src" / "mantis"
_TOOLS = _REPO / "tools"
_RUN_PY = _SRC / "run.py"
_TOOL_PY = _TOOLS / "ci_gates" / "preflight_mint.py"

#: The two production callers, and the ONLY two (DESIGN §1.1). Sites are
#: `<repo-relative path>::<enclosing def>`.
_SANCTIONED_SITES = {
    "src/mantis/run.py::launch_run",
    "tools/ci_gates/preflight_mint.py::_boot_main",
}

#: The re-cut composer's parameter tuple, as the CHILD must pass it. `resume_state` joined it
#: at R343(c) and the child's inclusion is the point: a parameter the launcher passes and the
#: child omits is a divergent boot wearing the one-authority name.
_COMPOSE_KWARGS = (
    "config", "trainer", "pool", "buffer", "log_dir", "checkpoint_dir", "resume_state",
)

#: Where a `WorkerPool` may be CONSTRUCTED in shipped code. TWO entries, the second argued
#: rather than typed — an allowlist that grows by edit is not an allowlist. The rule is one pool
#: construction ON ANY BOOT PATH, weaker than the original claim, and the assertion message says
#: the weaker thing.
#:
#: `mantis.diagnostics.worker_sweep` is not a boot path, and checkably so: it composes no
#: trainer, no `compose_run`, no `StepCoordinator`, no run-safety triple and no lifecycle, and
#: every one of those absences is asserted above by the same census. A third entry is a design
#: decision that edits this set, and a planted-break row proves the allowlist can still say no.
_SANCTIONED_POOL_SITES = {
    "src/mantis/run.py::build_run_collaborators",
    "src/mantis/diagnostics/worker_sweep.py::build_sweep_pool",
}


def _production_sources() -> list[Path]:
    """Every shipped `.py` under `src/` and `tools/`. `tests/` is deliberately OUT: a test may
    compose freely — the one-authority law is about what SHIPS."""
    return sorted([*(_SRC.rglob("*.py")), *(_TOOLS.rglob("*.py"))])


def _rel(path: Path) -> str:
    return str(path.relative_to(_REPO))


def _code_text(path: Path) -> str:
    """Source with COMMENT / STRING / f-string-literal tokens removed, including the 3.11-floor
    guard: FSTRING_MIDDLE is 3.12+, and on 3.11 f-strings lex as STRING."""
    skip = {tokenize.COMMENT, tokenize.STRING, getattr(tokenize, "FSTRING_MIDDLE", -1)}
    with path.open("rb") as handle:
        return "\n".join(tok.string for tok in tokenize.tokenize(handle.readline)
                         if tok.type not in skip)


def _enclosing_defs(tree: ast.AST) -> dict[ast.AST, str]:
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


def _called_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _call_sites(symbol: str) -> set[str]:
    sites: set[str] = set()
    for path in _production_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        owner = _enclosing_defs(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _called_name(node) == symbol:
                sites.add(f"{_rel(path)}::{owner.get(node, '<module>')}")
    return sites


def _definition_sites(symbol: str) -> set[str]:
    sites: set[str] = set()
    for path in _production_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == symbol:
                sites.add(_rel(path))
    return sites


def _func(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"no `def {name}` found — the one-authority shape is not in place")


def _body_without_docstring(fn: ast.FunctionDef) -> list[ast.stmt]:
    body = list(fn.body)
    if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        return body[1:]
    return body


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


def test_the_tree_holds_exactly_one_composer_and_exactly_one_collaborator_builder() -> None:
    """O-A1, definition half. MUTATION THAT REDS IT: define a second `compose_run` or
    `build_run_collaborators` anywhere under `src/` or `tools/` — which is precisely how the
    preflight's approximate boot came to exist, one helper at a time."""
    assert _definition_sites("compose_run") == {"src/mantis/run.py"}, (
        "exactly ONE `def compose_run` may exist in the shipped tree; found "
        f"{sorted(_definition_sites('compose_run'))}"
    )
    assert _definition_sites("build_run_collaborators") == {"src/mantis/run.py"}, (
        "the collaborator builder is the composition root's, not a tool's — the whole card "
        f"is that a CI gate owned the only real boot; found "
        f"{sorted(_definition_sites('build_run_collaborators'))}"
    )


def test_the_only_production_composition_call_sites_are_the_launcher_and_the_preflight_child(
) -> None:
    """O-A1, call-site half. MUTATION THAT REDS IT: any new production caller of the builder or
    composer — each is a boot path that can drift from run5's, so adding one is a design
    decision that edits this set."""
    assert _call_sites("compose_run") == _SANCTIONED_SITES, (
        "compose_run may be called from exactly two production sites (DESIGN §1.1) — "
        f"got {sorted(_call_sites('compose_run'))}"
    )
    assert _call_sites("build_run_collaborators") == _SANCTIONED_SITES, (
        "the builder has the same two callers and no others — a third would be a third "
        f"boot posture; got {sorted(_call_sites('build_run_collaborators'))}"
    )


def test_no_second_composer_exists_under_any_name() -> None:
    """O-A1, the half a NAME census cannot cover: a second composer called something else.

    `def compose_run_v2(...)` defeats a definition census, a name-keyed call-site census and the
    child's import census at once — but it cannot compose a run without driving the loop,
    building the run-safety triple and constructing the coordinator, so those three call sites
    ARE the composer whatever it is named."""
    for symbol in ("run_training_loop", "build_run_safety", "StepCoordinator"):
        assert _call_sites(symbol) == {"src/mantis/run.py::compose_run"}, (
            f"{symbol} is one of composition's irreducible steps: a second caller is a "
            f"second composer under another name; got {sorted(_call_sites(symbol))}"
        )
    assert _call_sites("init_trainer") == {"src/mantis/run.py::build_run_collaborators"}, (
        "init_trainer must be constructed in the composition root's builder and nowhere "
        "else — a tool that builds its own is the D-1 inversion; got "
        f"{sorted(_call_sites('init_trainer'))}"
    )
    assert _call_sites("WorkerPool") == _SANCTIONED_POOL_SITES, (
        "WorkerPool may be constructed on the composition root's builder and at the ONE "
        "filed non-boot site (see `_SANCTIONED_POOL_SITES`); got "
        f"{sorted(_call_sites('WorkerPool'))}"
    )


def test_the_preflight_child_binds_its_composer_from_mantis_run_itself() -> None:
    """O-A1, identity half: the child's two functions must BE `mantis.run`'s objects, bound by an
    `ImportFrom` naming `mantis.run` exactly. MUTATION THAT REDS IT: re-point the child's import
    at a shim, or re-declare either function in the tool. Asserted at SOURCE level because the
    import is function-local (the tool must import without torch), so the `ImportFrom` node IS
    the binding."""
    tool_tree = ast.parse(_TOOL_PY.read_text(encoding="utf-8"))
    boot = _func(tool_tree, "_boot_main")
    imports = [node for node in ast.walk(boot) if isinstance(node, ast.ImportFrom)]
    bound = {alias.name: node.module for node in imports for alias in node.names}
    for symbol in ("build_run_collaborators", "compose_run"):
        assert bound.get(symbol) == "mantis.run", (
            f"the preflight child must bind {symbol} from `mantis.run` and from nowhere "
            f"else (one authority, R121(a)); got {bound.get(symbol)!r}"
        )
    assert callable(build_run_collaborators) and callable(compose_run), (
        "…and the objects that name resolves to must be the live composition root's"
    )


def test_launch_run_is_exactly_build_then_compose_with_nothing_in_between() -> None:
    """O-A2: `launch_run`'s body is EXACTLY two statements — one bound builder call, one
    `return compose_run(...)` forwarding its fields and the SAME config. MUTATION THAT REDS IT:
    a third step flips the statement count; a different config to the composer flips the
    `ast.Name` equality. Both are invisible on a green tier, hence a structural instrument."""
    tree = ast.parse(_RUN_PY.read_text(encoding="utf-8"))
    body = _body_without_docstring(_func(tree, "launch_run"))
    assert len(body) == 2, (
        "launch_run is the pass-through, and only the pass-through: one build, one compose. "
        f"Found {len(body)} statements: {[type(s).__name__ for s in body]}"
    )
    build_stmt, return_stmt = body
    assert isinstance(build_stmt, ast.Assign) and isinstance(build_stmt.value, ast.Call), (
        "statement 1 must BIND the builder's result (a bare call would throw the "
        "collaborators away)"
    )
    assert _called_name(build_stmt.value) == "build_run_collaborators"
    assert isinstance(return_stmt, ast.Return) and isinstance(return_stmt.value, ast.Call)
    assert _called_name(return_stmt.value) == "compose_run"

    build_kwargs = {kw.arg: kw.value for kw in build_stmt.value.keywords}
    compose_kwargs = {kw.arg: kw.value for kw in return_stmt.value.keywords}
    assert not build_stmt.value.args and not return_stmt.value.args, (
        "both calls are keyword-only at the seam the census reads (DESIGN §1.1)"
    )
    assert set(build_kwargs) == {"config", "out_dir", "checkpoint_path"}, (
        "the builder takes the validated config, the out-dir and the optional resume "
        "target — and NOT a device parameter (R126/MF-1: no parameter carries a config "
        f"fact); got {sorted(build_kwargs)}"
    )
    assert set(compose_kwargs) == set(_COMPOSE_KWARGS), (
        f"the composer's call must pass exactly {list(_COMPOSE_KWARGS)}; got "
        f"{sorted(compose_kwargs)}"
    )
    for name, node in list(build_kwargs.items()) + [("config", compose_kwargs["config"])]:
        assert isinstance(node, ast.Name), (
            f"{name}= must forward the parameter UNMODIFIED — a transform here is the "
            f"divergent path; got {ast.dump(node)[:120]}"
        )
    assert compose_kwargs["config"].id == build_kwargs["config"].id, (
        "the composer must receive THE SAME config object the collaborators were built "
        "from; a second name here is a two-config boot"
    )
    collab = [target.id for target in build_stmt.targets if isinstance(target, ast.Name)]
    assert len(collab) == 1, "the builder's result binds to exactly one name"
    for field in _COMPOSE_KWARGS[1:]:
        node = compose_kwargs[field]
        assert (isinstance(node, ast.Attribute) and node.attr == field
                and isinstance(node.value, ast.Name) and node.value.id == collab[0]), (
            f"{field}= must be `{collab[0]}.{field}` — the collaborator the BUILDER made, "
            f"never a value the launcher computed; got {ast.dump(node)[:120]}"
        )


def test_the_builder_calls_the_real_trainer_and_pool_constructors_and_binds_them() -> None:
    """O-A3, half 1: `build_run_collaborators` must CALL `init_trainer(` and `WorkerPool(` and
    bind each result. Token presence survives a stand-in mutation (measured); a bound CALL does
    not. Equal-or-stronger is claimed for the PAIR with the behavioural drives."""
    tree = ast.parse(_RUN_PY.read_text(encoding="utf-8"))
    builder = _func(tree, "build_run_collaborators")
    bound: dict[str, str] = {}
    for node in ast.walk(builder):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            name = _called_name(node.value)
            if name and isinstance(node.targets[0], ast.Name):
                bound[name] = node.targets[0].id
    for constructor in ("init_trainer", "WorkerPool", "_select_buffer"):
        assert constructor in bound, (
            f"build_run_collaborators must construct the REAL {constructor} and bind the "
            f"result (R64: every wall the real boot hits is a tree defect, not a thing to "
            f"route around); bound constructions found: {sorted(bound)}"
        )


def test_the_buffer_selector_routes_the_declared_representation_to_a_real_engine_buffer() -> None:
    """O-A3, half 2: `_select_buffer` must name the real engine buffer and read the DECLARED
    representation. The `ReplayBuffer` arm went with the dense path, so what still matters is
    the THIRD arm — an absent or unknown representation RAISES rather than falling through to
    the one buffer left."""
    tree = ast.parse(_RUN_PY.read_text(encoding="utf-8"))
    selector = _func(tree, "_select_buffer")
    called = {_called_name(node) for node in ast.walk(selector) if isinstance(node, ast.Call)}
    assert "HexgBuffer" in called, (
        f"_select_buffer must construct HexgBuffer on its own arm; got {sorted(called)}"
    )
    assert "ReplayBuffer" not in called, (
        "_select_buffer names ReplayBuffer, which R346(f) deleted from the engine — a dense "
        f"arm cannot be re-added without the buffer it returns; got {sorted(called)}"
    )
    attributes = {node.attr for node in ast.walk(selector) if isinstance(node, ast.Attribute)}
    assert {"representation", "encoding"} <= attributes, (
        "the selection reads `config.identity.representation` and passes "
        "`config.identity.encoding` affirmatively — never sniffed, never defaulted (LAW-11)"
    )
    raises = [node for node in ast.walk(selector) if isinstance(node, ast.Raise)]
    assert raises, "the third arm RAISES; an absent representation is an ERROR (LAW-11)"


def test_the_composition_root_contains_no_stand_in_for_a_production_object() -> None:
    """O-A3, half 3 — the stand-in ban, applied to the module that owns the boot. MUTATION THAT
    REDS IT: a `MagicMock`, a `monkeypatch` or a `setattr(` re-point smuggled in to get past a
    wall, which is a TREE DEFECT and is fixed or queued instead.

    Scanned over CODE with comment/string tokens removed, because a raw-text census flags the
    module's prose. The `SimpleNamespace` carve-out is ENUMERATED: HEAD's root already
    constructs two, so the bound is a COUNT of CONSTRUCTIONS and a third must be argued."""
    code = _code_text(_RUN_PY)
    for token in ("MagicMock", "unittest.mock", "mock.patch", "monkeypatch", "setattr("):
        assert token not in code, (
            f"src/mantis/run.py contains {token!r}: the composition root must contain no "
            "stand-in for a production object (O-2's posture, R64)"
        )
    constructions = [node for node in ast.walk(ast.parse(_RUN_PY.read_text(encoding="utf-8")))
                     if isinstance(node, ast.Call) and _called_name(node) == "SimpleNamespace"]
    assert len(constructions) <= 2, (
        "only the two DISCLOSED SimpleNamespace constructions may exist in the root (the "
        f"anchor seed and the coordinator `subsystems=` stand-in); found {len(constructions)}"
    )


def test_the_preflight_child_boots_through_one_builder_and_one_composer_only() -> None:
    """O-A4, the O-A2 twin: the child side, which had NO producer at all. Three mutation vectors
    are each RED — a different config object handed to the composer than the builder got, a
    `collab` mutated between the two calls, and a third composition step inserted between them.
    Every one leaves rc 0 and every existing preflight assertion green."""
    tree = ast.parse(_TOOL_PY.read_text(encoding="utf-8"))
    builder_calls = [node for node in ast.walk(tree)
                     if isinstance(node, ast.Call)
                     and _called_name(node) == "build_run_collaborators"]
    compose_calls = [node for node in ast.walk(tree)
                     if isinstance(node, ast.Call) and _called_name(node) == "compose_run"]
    assert len(builder_calls) == 1, f"exactly one builder call in the tool; got {len(builder_calls)}"
    assert len(compose_calls) == 1, f"exactly one compose call in the tool; got {len(compose_calls)}"

    build_config = {kw.arg: kw.value for kw in builder_calls[0].keywords}.get("config")
    compose_kwargs = {kw.arg: kw.value for kw in compose_calls[0].keywords}
    assert isinstance(build_config, ast.Name) and isinstance(compose_kwargs.get("config"), ast.Name)
    assert compose_kwargs["config"].id == build_config.id, (
        "the child must compose the SAME config object it built collaborators from — a "
        "second name here is exactly how a burst-overridden boot and an un-overridden "
        f"compose diverge; got {compose_kwargs['config'].id!r} vs {build_config.id!r}"
    )
    assert set(compose_kwargs) == set(_COMPOSE_KWARGS), (
        "the child passes the 6-tuple and nothing else — no `eval_enabled=`, no `run_id=`: "
        f"the CONFIG governs both (R120/R123); got {sorted(compose_kwargs)}"
    )

    boot = _func(tree, "_boot_main")
    body = _body_without_docstring(boot)
    build_index = next(i for i, stmt in enumerate(body)
                       if any(node is builder_calls[0] for node in ast.walk(stmt)))
    compose_index = next(i for i, stmt in enumerate(body)
                         if any(node is compose_calls[0] for node in ast.walk(stmt)))
    assert build_index < compose_index, "the builder runs BEFORE the composer"
    for stmt in body[build_index + 1:compose_index]:
        if isinstance(stmt, ast.ImportFrom) and stmt.module == "mantis.run":
            continue
        assert isinstance(stmt, ast.If) and all(
            isinstance(inner, ast.Raise) for inner in stmt.body), (
            "only the §4.2 resumed-trainer REFUSAL may sit between the builder and the "
            "composer — it is a read-only instrument, and anything else there is a second "
            f"composition step in disguise; found {ast.dump(stmt)[:160]}"
        )

    collab_name = next(
        target.id for stmt in body if isinstance(stmt, ast.Assign)
        and any(node is builder_calls[0] for node in ast.walk(stmt))
        for target in stmt.targets if isinstance(target, ast.Name)
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                assert not (isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == collab_name), (
                    f"nothing may assign onto `{collab_name}`: a collaborator swapped after "
                    "the build is a boot the composer never agreed to"
                )
    for field in _COMPOSE_KWARGS[1:]:
        node = compose_kwargs[field]
        assert (isinstance(node, ast.Attribute) and node.attr == field
                and isinstance(node.value, ast.Name) and node.value.id == collab_name), (
            f"{field}= must be `{collab_name}.{field}` — the child composes what the ONE "
            f"builder made; got {ast.dump(node)[:120]}"
        )


def test_no_or_fallback_or_dict_get_stands_behind_a_config_fact_at_the_root() -> None:
    """O-A5. `compose_run`'s `run_id: str = "run"` default dies with the parameter, and this is
    the check that it does not survive in another guise: `config.run_id or "run"`, or
    `dump.get("run_id", "run")`. Nothing else sees it — the `getattr(config` census reads a
    different idiom, the consumer bijection sees a key that IS consumed, and gate 11 wants a
    registered-encoding literal. Also pinned: `config.run_id` is the EXACT expression in the
    composer, so the published identity and the declared one cannot be two things."""
    tree = ast.parse(_RUN_PY.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            root = _root_name(node.values[0])
            assert root not in ("config", "cfg"), (
                "an `or` whose left operand is a config read is a code-side default for a "
                f"config fact (R1): {ast.dump(node)[:160]}"
            )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "get":
            root = _root_name(node.func.value)
            assert root not in ("config", "cfg"), (
                "a `.get(...)` on a config-derived object smuggles the default the schema "
                f"is supposed to own (R1/LAW-11): {ast.dump(node)[:160]}"
            )
    composer = _func(tree, "compose_run")
    reads_run_id = any(
        isinstance(node, ast.Attribute) and node.attr == "run_id"
        and isinstance(node.value, ast.Name) and node.value.id == "config"
        for node in ast.walk(composer)
    )
    assert reads_run_id, (
        "compose_run must read `config.run_id` — the exact expression, from the validated "
        "config (`core.py:237`), with no parameter and no fallback behind it (R123)"
    )


def test_the_pool_allowlist_BITES_on_a_third_construction_site(tmp_path) -> None:
    """LAW-07 self-test: an allowlist rule never shown to reject anything is indistinguishable
    from one that accepts everything. The plant is a THIRD construction site in a temp tree, the
    same shape the second entry was granted for, so if that grant ever becomes a blanket
    exemption this row fails first."""
    planted_src = tmp_path / "src" / "mantis" / "somewhere"
    planted_src.mkdir(parents=True)
    (planted_src / "new_booter.py").write_text(
        "from mantis.selfplay.pool import WorkerPool\n"
        "def boot():\n"
        "    return WorkerPool(model=None, config={}, device=None, replay_buffer=None,\n"
        "                      arch=None)\n",
        encoding="utf-8",
    )
    (tmp_path / "tools").mkdir()
    module = sys.modules[__name__]
    saved = (module._SRC, module._TOOLS, module._REPO)
    module._SRC = tmp_path / "src" / "mantis"
    module._TOOLS = tmp_path / "tools"
    module._REPO = tmp_path
    try:
        offenders = _call_sites("WorkerPool") - _SANCTIONED_POOL_SITES
    finally:
        module._SRC, module._TOOLS, module._REPO = saved
    assert offenders == {"src/mantis/somewhere/new_booter.py::boot"}, (
        f"a planted third WorkerPool construction site was not rejected: {sorted(offenders)}"
    )
