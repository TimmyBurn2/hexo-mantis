# >300 justify (R8): O-A1..O-A5 are ONE census family making ONE claim — that exactly one
# composition path exists — and they share the instrument that makes the claim checkable
# (`_call_sites` / `_enclosing_defs` / `_body_without_docstring` / `_root_name` /
# `_code_text`, ~90 lines). R5 bars cross-test imports, so a split forks that helper set into
# two copies, and the helpers ARE the instrument: two copies is two instruments that drift
# apart while both stay green. The src-side census (O-A2) and the child-side census (O-A4)
# are also deliberately readable side by side — they are twins, and a reviewer checking that
# the two halves say the same thing about the same shape has to see them together.
"""⊕ WPMAIN ORACLE — the ONE composition authority (DESIGN §1/§9, oracles O-A1..O-A5).

RED-at-import until IMPL lands `mantis.run.build_run_collaborators` / `mantis.run.launch_run`
(the import below is the RED anchor; every oracle in this file rides on it). Written
ORACLE-FIRST: the boot path does not exist in `src/` yet — it lives in a CI GATE
(`tools/ci_gates/preflight_mint.py::_boot_main`), which is the one-authority violation
CARD-RUN-MAIN exists to end (R121(a)).

What this file exists to stop, in one sentence: **two boot paths.** The audit headline is
that `python -m mantis.run` validates a config and exits (`run.py:341`) while a CI tool owns
the only real composition — so "the preflight boots what run5 boots" was a claim with no
producer on EITHER side. These five oracles are that producer.

The defect each one is the only witness to:

- O-A1 — a SECOND composer, or a second builder, anywhere in `src/` or `tools/`; and a
  child that binds its composer from somewhere other than `mantis.run`.
- O-A2 — `launch_run` growing a third composition step, or transforming what it forwards
  (a launcher that "adjusts" the config before composing is a divergent path wearing the
  one-authority name).
- O-A3 — the builder that stops BUILDING: O-9's token census over the tool asserted only
  that `init_trainer`/`WorkerPool`/`HexgBuffer`/`ReplayBuffer` appear as tokens, and the
  tree itself MEASURED that insufficient (`test_preflight_mint_process.py:894-898`: a
  silent-default mutation left 1773 tests green). This is O-9's builder-token half at its
  new home, strengthened from "token present" to "call, with the result bound", and paired
  with the behavioural drives O-F1/O-B1 (DESIGN §4, C-1a: the pair is the equal-or-stronger
  successor, never the census alone).
- O-A4 — the CHILD side of one-authority, which REVIEW-design found had no producer at all.
  Three mutation vectors are each RED here: a different config object handed to the composer
  than the one the collaborators were built from; a `collab` mutated between the two calls;
  a third composition step inserted in the child.
- O-A5 — the smuggled default in its one uncensused guise: `config.run_id or "run"`, or
  `config.model_dump().get("run_id", "run")`. The `getattr(config` census
  (`test_run_strict_composition.py:445-465`), the consumer bijection and CI gate 11 all miss
  an `or`/`.get` fallback (R123 check (c)).

Fakes: NONE. Every oracle here is a static census over the shipped source, plus one identity
read of live module objects.
"""
from __future__ import annotations

import ast
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

#: DESIGN §1.5 — the re-cut composer's parameter tuple, as the CHILD must pass it.
_COMPOSE_KWARGS = ("config", "trainer", "pool", "buffer", "log_dir", "checkpoint_dir")


def _production_sources() -> list[Path]:
    """Every shipped `.py` under `src/` and `tools/`. `tests/` is deliberately OUT: a test
    may compose freely — the one-authority law is about what SHIPS."""
    return sorted([*(_SRC.rglob("*.py")), *(_TOOLS.rglob("*.py"))])


def _rel(path: Path) -> str:
    return str(path.relative_to(_REPO))


def _code_text(path: Path) -> str:
    """Source with COMMENT / STRING / f-string-literal tokens removed — the house instrument
    (`test_preflight_mint.py`'s `_code_text`), including its 3.11-floor guard: FSTRING_MIDDLE
    is 3.12+ (PEP 701), and on 3.11 f-strings lex as STRING."""
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


# ══ O-A1 — one composer, one builder, one binding ═════════════════════════════════════
def test_the_tree_holds_exactly_one_composer_and_exactly_one_collaborator_builder() -> None:
    """O-A1, definition half. MUTATION THAT REDS IT: define a second `compose_run` (or a
    second `build_run_collaborators`) anywhere under `src/` or `tools/` — which is precisely
    how the preflight's approximate boot came to exist in the first place, one helper at a
    time. A second definition is a second authority no signature census can see."""
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
    """O-A1, call-site half — the repo census the dispatch's minimum set names.

    MUTATION THAT REDS IT: any new production caller of the builder or the composer — a
    `mantis.deploy` launcher, a second entry point, a convenience wrapper in `tools/`. Every
    such caller is a boot path that can drift from run5's, which is the failure this WP is
    named for. Adding one is a design decision that edits this set, never an edit that
    happens to pass."""
    assert _call_sites("compose_run") == _SANCTIONED_SITES, (
        "compose_run may be called from exactly two production sites (DESIGN §1.1) — "
        f"got {sorted(_call_sites('compose_run'))}"
    )
    assert _call_sites("build_run_collaborators") == _SANCTIONED_SITES, (
        "the builder has the same two callers and no others — a third would be a third "
        f"boot posture; got {sorted(_call_sites('build_run_collaborators'))}"
    )


def test_no_second_composer_exists_under_any_name() -> None:
    """O-A1, the half a NAME census cannot cover — and the mutation that walks past every
    other assertion in this file: a second composer called something else.

    `def compose_run_v2(...)` defeats a definition census, a call-site census keyed on the
    name, and the child's import census all at once. What it CANNOT do is compose a run
    without driving the loop, building the run-safety triple and constructing the
    coordinator. Those three calls are the boot's irreducible steps, so their call sites ARE
    the composer, whatever it is named.

    Measured at `b482243`: each of the three has exactly one call site in shipped code, and
    it is `compose_run`. Green today, and load-bearing the moment a second boot path is
    written.

    The builder half is RED today for the reason the whole card exists: `init_trainer` and
    `WorkerPool` are constructed in `tools/ci_gates/preflight_mint.py::_boot_main` — a CI
    gate owning the only real collaborator build (D-1)."""
    for symbol in ("run_training_loop", "build_run_safety", "StepCoordinator"):
        assert _call_sites(symbol) == {"src/mantis/run.py::compose_run"}, (
            f"{symbol} is one of composition's irreducible steps: a second caller is a "
            f"second composer under another name; got {sorted(_call_sites(symbol))}"
        )
    for symbol in ("init_trainer", "WorkerPool"):
        assert _call_sites(symbol) == {"src/mantis/run.py::build_run_collaborators"}, (
            f"{symbol} must be constructed in the composition root's builder and nowhere "
            f"else — a tool that builds its own is the D-1 inversion; got "
            f"{sorted(_call_sites(symbol))}"
        )


def test_the_preflight_child_binds_its_composer_from_mantis_run_itself() -> None:
    """O-A1, identity half. The child's `build_run_collaborators` / `compose_run` must BE
    `mantis.run`'s objects — bound by an `ImportFrom` naming `mantis.run` exactly.

    MUTATION THAT REDS IT: re-point the child's import at a shim module
    (`from mantis.run_compat import compose_run`), or re-declare either function inside the
    tool. Both keep every existing preflight assertion green while the child boots something
    else — the divergence route RED-TEAM's "can any flag make preflight boot a different
    path" lens is pre-registered against.

    Why the binding is asserted at SOURCE level and not by attribute identity: DESIGN §1.4
    keeps the import FUNCTION-LOCAL inside `_boot_main` (the tool must stay importable
    without pulling torch), so there is no module attribute to compare. The `ImportFrom`
    node IS the binding."""
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


# ══ O-A2 — launch_run is build -> compose, pass-through ═══════════════════════════════
def test_launch_run_is_exactly_build_then_compose_with_nothing_in_between() -> None:
    """O-A2 (DESIGN §1.1): `launch_run`'s body is EXACTLY two statements — one builder call
    whose result is bound, one `return compose_run(...)` forwarding that result's fields and
    the SAME config object.

    MUTATION THAT REDS IT: (i) insert any third step (a config transform, a device coercion,
    an `if resume:` branch) — the statement count flips; (ii) pass a DIFFERENT config to the
    composer than the builder got (`compose_run(config=adjusted, ...)`) — the `ast.Name`
    equality flips. Both mutations are behaviourally invisible on a green CI tier, which is
    why the instrument is structural. This is the SRC-side twin of O-A4."""
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
    assert set(build_kwargs) == {"config", "out_dir"}, (
        "the builder takes the validated config and the out-dir — and NOT a device "
        "parameter (R126/MF-1: no parameter carries a config fact); got "
        f"{sorted(build_kwargs)}"
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


# ══ O-A3 — the builder really builds (O-9's builder-token successor) ══════════════════
def test_the_builder_calls_the_real_trainer_and_pool_constructors_and_binds_them() -> None:
    """O-A3, half 1 (DESIGN §9, C-1a). `build_run_collaborators` must CALL `init_trainer(`
    and `WorkerPool(` and bind each result to a name.

    MUTATION THAT REDS IT: replace either construction with a stand-in, a `None`, or a
    lazily-deferred handle. Token presence — what O-9 asserted over the tool — survives
    exactly that mutation (measured, `test_preflight_mint_process.py:894-898`); a bound CALL
    does not. Equal-or-stronger is claimed for the PAIR of this census and the behavioural
    drives (O-B1 boots through it; O-F1 drives the selector), never for the census alone."""
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


def test_the_buffer_selector_routes_graph_and_grid_to_the_real_engine_buffers() -> None:
    """O-A3, half 2. `_select_buffer` must name both real engine buffers and read the
    DECLARED representation.

    MUTATION THAT REDS IT: collapse the two arms into one (`return ReplayBuffer(...)`) — the
    dense-by-default defect LAW-11 bans, and the mutation that survived O-9's token census.
    The behavioural producer is O-F1; this is its structural half, and it is the one that
    also sees a route deleted from a still-token-bearing file."""
    tree = ast.parse(_RUN_PY.read_text(encoding="utf-8"))
    selector = _func(tree, "_select_buffer")
    called = {_called_name(node) for node in ast.walk(selector) if isinstance(node, ast.Call)}
    for buffer_class in ("HexgBuffer", "ReplayBuffer"):
        assert buffer_class in called, (
            f"_select_buffer must construct {buffer_class} on its own arm; got {sorted(called)}"
        )
    attributes = {node.attr for node in ast.walk(selector) if isinstance(node, ast.Attribute)}
    assert {"representation", "encoding"} <= attributes, (
        "the selection reads `config.identity.representation` and passes "
        "`config.identity.encoding` affirmatively — never sniffed, never defaulted (LAW-11)"
    )
    raises = [node for node in ast.walk(selector) if isinstance(node, ast.Raise)]
    assert raises, "the third arm RAISES; an absent representation is an ERROR (LAW-11)"


def test_the_composition_root_contains_no_stand_in_for_a_production_object() -> None:
    """O-A3, half 3 — O-2's stand-in ban, applied to the module that now owns the boot.

    MUTATION THAT REDS IT: a `MagicMock`, a `monkeypatch`, a `setattr(` re-point smuggled
    into the root to get past a wall. R64: a wall the real boot hits is a TREE DEFECT and is
    fixed or queued, never papered over in the composer.

    Scanned over CODE with comment/string tokens removed — O-2's own instrument, for O-2's
    own stated reason: a raw-text census flags the module's prose, which is the false
    positive that teaches people to word comments around a gate. `run.py` is half rationale
    by design (its R8 header says so), and that rationale must stay writable.

    The `SimpleNamespace` carve-out is ENUMERATED, not waived (the P-11 trap): HEAD's own
    root already CONSTRUCTS two — the `resolved_anchor` seed (`run.py:254`) and the
    coordinator's `subsystems=` stand-in (`run.py:285`, disclosed in DESIGN §7 leg 3 as out
    of scope). A blanket ban would red on shipped code, so the bound is a COUNT of
    CONSTRUCTIONS: two, the two that exist. Constructions rather than textual occurrences,
    because the `from types import SimpleNamespace` line is not a stand-in and counting it
    would make the bound a lie. A third construction is a new stand-in and must be argued,
    not typed."""
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


# ══ O-A4 — the child-side twin (success criterion 2's producer) ═══════════════════════
def test_the_preflight_child_boots_through_one_builder_and_one_composer_only() -> None:
    """O-A4 (DESIGN §9, the O-A2 twin; REVIEW-design C-2/F-2 found the child side had NO
    producer at all — this is it).

    Three mutation vectors, each RED here:
      (i)  hand `compose_run` a different config object than the builder got — the
           `ast.Name` identity check flips (this is the two-config boot the burst override
           makes reachable: `booted` vs `config`);
      (ii) mutate `collab` between the two calls (`collab.trainer = something_else`) — the
           attribute-assignment ban flips;
      (iii) insert a third composition step between them — the between-statements check
           flips (only the §4.2 refusal and the `mantis.run` import may sit there).
    Every one of the three leaves rc 0 and every existing preflight assertion green."""
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


# ══ O-A5 — the guise no existing census sees ══════════════════════════════════════════
def test_no_or_fallback_or_dict_get_stands_behind_a_config_fact_at_the_root() -> None:
    """O-A5 / R123-O1. `compose_run`'s `run_id: str = "run"` default (`run.py:173`) dies with
    the parameter (§1.5) — check (c) of R123 is that it does not survive the move in another
    guise, and the guise is `config.run_id or "run"` or `dump.get("run_id", "run")`.

    MUTATION THAT REDS IT: write either fallback. Nothing else in the tree sees it — the
    `getattr(config` substring census reads a different idiom, the consumer bijection sees a
    key that IS consumed, gate 11 requires a REGISTERED-ENCODING literal in a default
    position, and the signature census sees a parameter that is genuinely gone. This oracle
    is the only witness, which is why R123 made it a rider.

    Also pinned: `config.run_id` appears as the EXACT expression in the composer, so the
    identity the run publishes (`run_boot_identity`, `run.py:210-219`) and the identity the
    config declares cannot be two things."""
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
