# >300 justify (R8): the gate identity, the sink specification, the enumeration criterion and
# every control that shows each of them can reject are ONE unit. A discriminator this narrow is
# only trustworthy alongside the negative controls that prove it does NOT fire on the two
# production loops; separating them would let the discriminator be widened until it flags them,
# with nothing in the same file going red.
"""T2b — every per-row D6 element applied in `src/mantis/` Python is obtained through the gate.

WHY THE NAME CARRIES `python`. `plan/DESIGN_ARCHCAPS.md` exit criterion 5(b)'s subject is FOUR
gate sites, THREE of them Rust (`hexg/sample.rs:172`, `sym.rs:134-140`, `runner/game.rs:676`).
This census is `src/mantis/`, `*.py`, case-sensitive, only — so it cannot see three quarters of
its criterion's subject. **Criterion 5 is PARTIALLY discharged**; the three Rust sites are a
named residue that this suite does not schedule. The property is true of them; this tier does
not check them, and its name says so.

ONE PRODUCER — one census over one tree. A green means "unchanged, in Python".

CLAUSE (i) IS A REACHABILITY PROPERTY AND ITS APPROXIMATION IS STATED, not implied. What is
implemented is an INTRA-MODULE reaching-definitions walk: for each sink, the applied index
expression is resolved backwards through local assignments within the enclosing module, through
pure element-wise transforms (`.astype`, `.tolist`, `int(...)`, `np.asarray`, a subscript of
it), to its definition, which must be a call to `draw_record_syms` or
`draw_window_preserving_syms`. NOT COVERED, stated: an index arriving as a parameter from
another module, one round-tripped through a container the walk does not model, one
reconstructed from a file/config/checkpoint, and anything outside `src/mantis/**/*.py`.

THE GATE IS IDENTIFIED BY ITS IMPORT EDGE TO `mantis.data.augment`, NEVER BY BARE NAME.
Without the edge any module could define `def draw_record_syms(...): return
np.random.randint(0, 12, n)` and pass; the decoy control below plants exactly that.

THE SINK IS AN ARGUMENT POSITION THAT RECEIVES A PER-ROW D6 ELEMENT — three of them, structural:
`apply_symmetries_batch` argument 2 (both the bare-`Name` and the `Attribute` call forms are
live at HEAD), a subscript of a `get_policy_scatters(...)` result whose index is per-row, and
`sample_graph_batch(augment=True)`. THE THIRD KIND IS RECORDED, NOT CHECKED, and cannot be:
its draw happens in Rust (`hexg/sample.rs:172`), one of this tier's three named residues. It is
therefore unfalsifiable by construction — it can never contribute an ungated row — and its
count is reported SEPARATELY from the checked sinks so that a kind which checks nothing cannot
swell the number a reader takes for coverage. Zero such sinks exist at HEAD.

THE GROUP-ENUMERATION FORM IS EXCLUDED BY A STATED CRITERION, NOT BY TUNING, and the criterion
is STRICTLY STRONGER than "skip enumeration loops". In that form the subscript index is the
group counter and the row→element association is carried by the ROW MASK, so the criterion is a
conjunction and this tier checks its second half: a subscript whose index is the target of an
iteration over `range(K)` is admissible ONLY IF, in the same loop body, the rows it applies to
are selected by a mask derived from a comparison against a GATE-DRAWN array. An enumeration
loop masked against a non-gate array fires; an enumeration loop applied to all rows
unconditionally fires. Both are required controls below, and the two production loops are
required NEGATIVE controls — a tier that reds on them is measuring the proxy, not the mechanism.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from _corpus import ConformanceRefusal

SRC = Path(__file__).resolve().parents[3] / "src" / "mantis"
GATE_MODULE = "mantis.data.augment"
GATE_FUNCTIONS = ("draw_record_syms", "draw_window_preserving_syms")
SCATTER_FACTORY = "get_policy_scatters"
ENGINE_SINK = "apply_symmetries_batch"
#: The engine's own parameter name for the index operand
#: (`crates/mantis-bridge/src/utils.rs:46`). It is here because the operand is a POSITION only
#: until someone passes it by keyword, which the engine accepts and which ruff and pyright
#: accept: `apply_symmetries_batch(states, sym_indices=np.random.randint(0, 12, size=n))` is a
#: working ungated draw that a positional-only reader records as no sink at all.
ENGINE_SINK_INDEX_PARAM = "sym_indices"
ENGINE_SINK_KIND = "apply_symmetries_batch.index_operand"
GRAPH_SINK = "sample_graph_batch"
#: Arch / capability symbols that may not be imported into the augmentation path (clause ii).
ARCH_SYMBOLS = frozenset({"ModelArch", "CnnArch", "GnnArch", "ArchCaps", "arch_from_spec_and_config"})
ARCH_MODULES = ("mantis.model.arch",)

#: `sample_graph_batch(augment=True)` hands the draw to Rust (`hexg/sample.rs:172`), which this
#: PYTHON census cannot read. The sink is still RECORDED, because the census's job is to name
#: every place a D6 element is applied; but the `gated` flag it is recorded with means "NOT
#: CHECKED HERE", not "checked and found gated". Such a row can never be ungated, so it can
#: never falsify anything, and folding it into the gated total would let a kind that checks
#: nothing raise the number that reads as coverage. It is counted separately at the gate for
#: exactly that reason.
_RUST_SIDE_NOT_CHECKED_HERE = True
GRAPH_SINK_KIND = "sample_graph_batch.augment"

_TRANSFORM_METHODS = frozenset(
    {"astype", "tolist", "ravel", "flatten", "reshape", "copy", "item", "squeeze"}
)
_TRANSFORM_FUNCTIONS = frozenset({"int", "list", "tuple", "asarray", "array", "int64", "intp"})

#: The module-function form of an in-place write, where the target is argument 1. Enumerated
#: and NON-EXHAUSTIVE — the other two arms below are general, and this one is stated so the
#: residue is a named list rather than an implied one.
_INPLACE_FUNCTIONS = frozenset({"copyto", "put", "putmask", "place"})
#: numpy's general in-place channel, and it is a keyword rather than a family.
_OUT_PARAM = "out"


class UngatedSymmetryDraw(ConformanceRefusal):
    """A D6 element reaches a symmetry application without coming through the per-record gate."""


class UnclassifiableSink(ConformanceRefusal):
    """A symmetry application was found whose index operand this census cannot locate."""


class EmptyAugmentationCensus(ConformanceRefusal):
    """A census this tier quantifies over is empty, so its clause is vacuously true."""


class ArchImportInAugmentationPath(ConformanceRefusal):
    """A module on the augmentation path imports an arch or capability symbol."""


class ModuleFacts:
    """Everything one module's AST says about draws, sinks and the gate's identity in it."""

    def __init__(self, path: Path, tree: ast.AST) -> None:
        self.path = path
        self.tree = tree
        self.gate_names: set[str] = set()
        self.gate_module_aliases: set[str] = set()
        self.engine_sink_names: set[str] = set()
        self.arch_imports: list[str] = []
        self.definitions: dict[str, list[ast.expr]] = {}
        self.range_loops: dict[str, ast.For] = {}
        self.scatter_names: set[str] = set()
        self._collect()

    def _collect(self) -> None:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    bound = alias.asname or alias.name
                    if module == GATE_MODULE and alias.name in GATE_FUNCTIONS:
                        self.gate_names.add(bound)
                    if alias.name == ENGINE_SINK:
                        self.engine_sink_names.add(bound)
                    if module in ARCH_MODULES or alias.name in ARCH_SYMBOLS:
                        self.arch_imports.append(f"{module}.{alias.name}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == GATE_MODULE:
                        self.gate_module_aliases.add(alias.asname or alias.name)
                    if alias.name in ARCH_MODULES:
                        self.arch_imports.append(alias.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    self._record_definition(target, node.value)
            elif isinstance(node, ast.AnnAssign):
                if node.value is not None:
                    self._record_definition(node.target, node.value)
            elif isinstance(node, ast.AugAssign):
                self._record_definition(node.target, node.value)
            elif isinstance(node, ast.Call):
                self._record_inplace_writes(node)
            elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
                if isinstance(node.iter, ast.Call) and _callee_name(node.iter.func) == "range":
                    self.range_loops[node.target.id] = node
        for name, values in self.definitions.items():
            if any(self._is_scatter_factory_call(v) for v in values):
                self.scatter_names.add(name)

    def _record_definition(self, target: ast.expr, value: ast.expr) -> None:
        """Record `value` as a definition of the name this store writes THROUGH.

        A STORE IS NOT ONLY A REBINDING. `sym_indices[:] = np.random.randint(0, 12, size=n)`
        overwrites every element of a gate-drawn array while leaving the name bound to the gate
        call, so a walk that records only `ast.Name` targets keeps resolving the name to the
        gate and every downstream row receives a uniform-over-12 element — the exact defect this
        tier describes, in its cheapest spelling, and one the stated residues do not name.
        A subscript store is therefore recorded as an ADDITIONAL definition of its base name, as
        is an `AugAssign`, so `resolves_to_gate`'s "every definition, not merely one" sees it.

        An `Attribute` store is deliberately NOT recorded: an attribute expression never
        resolves to the gate on the read side either, so recording one under its base name
        would attribute a write to the wrong subject.
        """
        if isinstance(target, ast.Name):
            self.definitions.setdefault(target.id, []).append(value)
            return
        base = target
        while isinstance(base, ast.Subscript):
            base = base.value
        if isinstance(base, ast.Name) and base is not target:
            self.definitions.setdefault(base.id, []).append(value)

    def _record_inplace_writes(self, node: ast.Call) -> None:
        """Record a CALL that may overwrite a name IN PLACE as a definition of that name.

        `_record_definition` records stores — `x = …`, `x[:] = …`, `x += …` — so the subscript
        spelling of an in-place overwrite is seen. A mutation performed by a CALL leaves the
        name bound to the gate call and no store exists at all, so the walk kept resolving the
        name to the gate: `np.copyto(sym_indices, np.random.randint(0, 12, size=n))` planted
        one line after the real draw in `train/batch_assembly.py` left the whole suite green.
        The F3 fix's own argument was that the subscript store is "the cheapest spelling of the
        defect there is"; `np.copyto` is cheaper, and `.fill(3)` is cheaper still.

        THE METHOD ARM IS DEFAULT-UNSAFE ON PURPOSE. A method this census does not model may
        write through its receiver, so anything outside the PURE transform set it already
        models counts as a write. Over-recording costs a false ungated report, which is a red
        someone reads; under-recording costs a green that means nothing.
        """
        for keyword in node.keywords:
            if keyword.arg == _OUT_PARAM and isinstance(keyword.value, ast.Name):
                self.definitions.setdefault(keyword.value.id, []).append(node)
        func = node.func
        if isinstance(func, ast.Attribute):
            if func.attr in _INPLACE_FUNCTIONS and node.args:
                target = node.args[0]
                if isinstance(target, ast.Name):
                    self.definitions.setdefault(target.id, []).append(node)
            elif (
                isinstance(func.value, ast.Name)
                and func.attr not in _TRANSFORM_METHODS
                and func.attr not in GATE_FUNCTIONS
            ):
                self.definitions.setdefault(func.value.id, []).append(node)
        elif isinstance(func, ast.Name) and func.id in _INPLACE_FUNCTIONS and node.args:
            target = node.args[0]
            if isinstance(target, ast.Name):
                self.definitions.setdefault(target.id, []).append(node)

    def _is_scatter_factory_call(self, node: ast.expr) -> bool:
        inner = node
        if isinstance(inner, ast.IfExp):  # `f(...) if augment else None`
            return self._is_scatter_factory_call(inner.body) or self._is_scatter_factory_call(
                inner.orelse
            )
        return isinstance(inner, ast.Call) and _callee_name(inner.func) == SCATTER_FACTORY

    def is_gate_call(self, node: ast.expr) -> bool:
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        if isinstance(func, ast.Name):
            return func.id in self.gate_names
        if isinstance(func, ast.Attribute) and func.attr in GATE_FUNCTIONS:
            base = func.value
            return isinstance(base, ast.Name) and base.id in self.gate_module_aliases
        return False


def _engine_sink_operand(node: ast.Call) -> ast.expr | None:
    """The per-row index operand of an engine sink call, by POSITION or by KEYWORD.

    `len(node.args) >= 2` used to be a precondition of recording the sink at all, so a call
    with one positional and one keyword argument recorded NO sink — the census cardinality went
    6 to 5 and nothing compared it, while the call itself works at runtime.
    """
    for keyword in node.keywords:
        if keyword.arg == ENGINE_SINK_INDEX_PARAM:
            return keyword.value
    positional = node.args
    if len(positional) >= 2 and not any(isinstance(a, ast.Starred) for a in positional[:2]):
        return positional[1]
    return None


def _callee_name(func: ast.expr) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _strip_transforms(node: ast.expr) -> ast.expr:
    """Peel pure element-wise transforms off an expression: `.astype`, `.tolist`, `int(...)`."""
    current = node
    while True:
        if isinstance(current, ast.Call):
            func = current.func
            if isinstance(func, ast.Attribute) and func.attr in _TRANSFORM_METHODS:
                current = func.value
                continue
            if _callee_name(func) in _TRANSFORM_FUNCTIONS and current.args:
                current = current.args[0]
                continue
        break
    return current


def resolves_to_gate(node: ast.expr, facts: ModuleFacts, depth: int = 0) -> bool:
    """Backwards reaching-definitions: does this expression come from a gate call?

    EVERY definition of a name must resolve to the gate, not merely one — a name whose second
    definition is an ungated draw is an escape hatch, not an ambiguity.
    """
    if depth > 12:
        return False
    current = _strip_transforms(node)
    if facts.is_gate_call(current):
        return True
    if isinstance(current, ast.Subscript):
        return resolves_to_gate(current.value, facts, depth + 1)
    if isinstance(current, ast.Name):
        values = facts.definitions.get(current.id)
        if not values:
            return False
        return all(resolves_to_gate(v, facts, depth + 1) for v in values)
    if isinstance(current, ast.Call) and isinstance(current.func, ast.Attribute):
        if current.func.attr == "where" and len(current.args) == 1:
            return resolves_to_gate(current.args[0], facts, depth + 1)
    return False


def _is_per_row_index(node: ast.expr, facts: ModuleFacts) -> bool:
    """A per-row element, as opposed to a group counter: a subscript of an array, or any
    expression that is not the target of an iteration over `range(K)`."""
    stripped = _strip_transforms(node)
    if isinstance(stripped, ast.Name) and stripped.id in facts.range_loops:
        return False
    return True


def _mask_is_gate_derived(loop: ast.For, loop_var: str, facts: ModuleFacts) -> bool:
    """The enumeration criterion's SECOND half: in this loop body, rows are selected by a mask
    derived from a comparison against a GATE-DRAWN array."""
    for node in ast.walk(loop):
        if not isinstance(node, ast.Compare) or len(node.comparators) != 1:
            continue
        if not isinstance(node.ops[0], ast.Eq):
            continue
        right = node.comparators[0]
        if not (isinstance(right, ast.Name) and right.id == loop_var):
            continue
        if resolves_to_gate(node.left, facts):
            return True
    return False


def module_sinks(facts: ModuleFacts) -> list[tuple[str, int, str, bool]]:
    """`(kind, line, description, gated)` for every per-row D6 sink in one module."""
    sinks: list[tuple[str, int, str, bool]] = []
    for node in ast.walk(facts.tree):
        if isinstance(node, ast.Call):
            callee = node.func
            is_engine_sink = (
                isinstance(callee, ast.Name) and callee.id in facts.engine_sink_names
            ) or (isinstance(callee, ast.Attribute) and callee.attr == ENGINE_SINK)
            if is_engine_sink:
                operand = _engine_sink_operand(node)
                if operand is None:
                    raise UnclassifiableSink(
                        f"{facts.path}:{node.lineno}: a call to {ENGINE_SINK} was found whose "
                        f"`{ENGINE_SINK_INDEX_PARAM}` operand this census cannot locate (a "
                        "splat, or a spelling not modelled here). The census REFUSES rather "
                        "than skipping it: a sink dropped for being unreadable is a sink that "
                        "disappears from the number a reader takes for coverage, which is "
                        "exactly what a positional-only reader did with the keyword form."
                    )
                sinks.append(
                    (ENGINE_SINK_KIND, node.lineno, "engine batch symmetry apply",
                     resolves_to_gate(operand, facts))
                )
                continue
            if _callee_name(callee) == GRAPH_SINK:
                augment_on = any(
                    kw.arg == "augment" and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True
                    for kw in node.keywords
                )
                if augment_on:
                    sinks.append(
                        (GRAPH_SINK_KIND, node.lineno, "rust-side graph draw (NOT CHECKED by "
                         "this census: the draw is Rust-side)", _RUST_SIDE_NOT_CHECKED_HERE)
                    )
                continue
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if node.value.id not in facts.scatter_names:
                continue
            index = node.slice
            if _is_per_row_index(index, facts):
                sinks.append(
                    ("policy_scatter.per_row", node.lineno, f"{node.value.id}[per-row]",
                     resolves_to_gate(index, facts))
                )
            else:
                loop_var = _strip_transforms(index).id
                loop = facts.range_loops[loop_var]
                sinks.append(
                    ("policy_scatter.enumeration", node.lineno,
                     f"{node.value.id}[{loop_var}] under for {loop_var} in range(...)",
                     _mask_is_gate_derived(loop, loop_var, facts))
                )
    return sinks


def census(root: Path) -> tuple[list[tuple[str, int, str]], list[tuple[str, int, str, bool]]]:
    """`(draws, sinks)` over every `*.py` under `root`. Root-parameterised so every planted
    break below is constructible against the SAME walk the gate runs."""
    draws: list[tuple[str, int, str]] = []
    sinks: list[tuple[str, int, str, bool]] = []
    for path in sorted(root.rglob("*.py")):
        facts = ModuleFacts(path, ast.parse(path.read_text(encoding="utf-8")))
        rel = path.relative_to(root).as_posix()
        for node in ast.walk(facts.tree):
            if facts.is_gate_call(node):
                draws.append((rel, node.lineno, _callee_name(node.func)))
        for kind, line, desc, gated in module_sinks(facts):
            sinks.append((f"{rel}:{line}", line, f"{kind} — {desc}", gated))
    return draws, sinks


def augmentation_path_modules(root: Path) -> tuple[str, ...]:
    """Modules carrying a draw or a sink — the "augmentation path", derived, never listed."""
    draws, sinks = census(root)
    return tuple(sorted({d[0] for d in draws} | {s[0].rsplit(":", 1)[0] for s in sinks}))


def arch_imports_on_path(root: Path) -> list[str]:
    offenders: list[str] = []
    for rel in augmentation_path_modules(root):
        path = root / rel
        facts = ModuleFacts(path, ast.parse(path.read_text(encoding="utf-8")))
        offenders += [f"{rel}: {sym}" for sym in facts.arch_imports]
    return offenders


def require_non_empty(items: list, what: str) -> int:
    if not items:
        raise EmptyAugmentationCensus(
            f"the {what} census is EMPTY. 'every drawn element traces to a gate' is vacuously "
            "true over zero elements, which is the guard-with-nothing-to-check failure this "
            "tier's counters exist to refuse."
        )
    return len(items)


def require_no_arch_imports(root: Path) -> int:
    """Clause (ii)'s refusal, behind a helper so the planted break DRIVES it (R-O1).

    The raise used to be inline in the gate, which meant the control could only assert that the
    offender list came back non-empty — the tier's actual refusal was never reached by any
    break, and "this reds the tier" was an inference.
    """
    offenders = arch_imports_on_path(root)
    if offenders:
        raise ArchImportInAugmentationPath(
            "the augmentation path imports arch/capability symbols: " + ", ".join(offenders)
        )
    return len(augmentation_path_modules(root))


def require_all_gated(sinks: list[tuple[str, int, str, bool]]) -> None:
    ungated = [f"{s[0]} ({s[2]})" for s in sinks if not s[3]]
    if ungated:
        raise UngatedSymmetryDraw(
            "a D6 element reaches a symmetry application outside the per-record gate at: "
            + ", ".join(ungated)
            + ". The gate restricts a spread row to WINDOW_PRESERVING_SYMS; an ungated draw is "
            "uniform over 12 and injects label noise rather than augmenting."
        )


# --------------------------------------------------------------------------------------- #
# The gate half
# --------------------------------------------------------------------------------------- #
def test_every_per_row_D6_element_applied_in_src_comes_through_the_gate(derived):
    draws, sinks = census(SRC)
    derived("t2b.draw_sites", [f"{d[0]}:{d[1]}" for d in draws])
    derived("t2b.draw_census.cardinality", require_non_empty(draws, "draw"))
    derived("t2b.sink_sites", [f"{s[0]} {s[2]}" for s in sinks])
    derived("t2b.sink_census.cardinality", require_non_empty(sinks, "sink"))
    kinds: dict[str, int] = {}
    for kind, _line, _desc, _gated in ((s[2].split(" — ")[0], s[1], s[2], s[3]) for s in sinks):
        kinds[kind] = kinds.get(kind, 0) + 1
    derived("t2b.sink_kinds", kinds)
    unchecked = [s for s in sinks if s[2].split(" — ")[0] == GRAPH_SINK_KIND]
    derived("t2b.sink_census.rust_side_NOT_CHECKED", len(unchecked))
    derived("t2b.sink_census.checked_here", len(sinks) - len(unchecked))
    require_all_gated(sinks)


def test_the_augmentation_path_module_set_is_non_empty_and_imports_no_arch_symbol(derived):
    """Clause (ii). "No module in the augmentation path imports an arch symbol" is vacuous if
    the path resolves to zero modules, so the module set is pinned as a derived output first."""
    modules = augmentation_path_modules(SRC)
    derived("t2b.augmentation_path_modules", modules)
    derived("t2b.augmentation_path.cardinality", require_non_empty(list(modules), "module-set"))
    require_no_arch_imports(SRC)


# --------------------------------------------------------------------------------------- #
# Planted breaks and controls — all DEFAULT tier
# --------------------------------------------------------------------------------------- #
def _tree(tmp_path: Path, body: str, name: str = "mod.py") -> Path:
    root = tmp_path / "mantis"
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(body, encoding="utf-8")
    return root


_GATE_IMPORT = "from mantis.data.augment import draw_record_syms, get_policy_scatters\n"


def test_an_EMPTY_sink_census_is_refused(tmp_path):
    """PB-16. Clause (i) is vacuously true over zero applications, and only the DRAW census was
    ever required to be non-empty. Both counters are asserted and pinned now."""
    root = _tree(tmp_path, "x = 1\n")
    draws, sinks = census(root)
    assert (draws, sinks) == ([], [])
    with pytest.raises(EmptyAugmentationCensus, match="sink"):
        require_non_empty(sinks, "sink")


def test_an_ALIASED_engine_sink_fed_an_UNGATED_index_still_fires(tmp_path):
    """PB-17. Both call forms are live at HEAD — a bare `Name` under `from mantis._engine import
    apply_symmetries_batch` and an `Attribute` call — so the resolver binds import aliases."""
    root = _tree(
        tmp_path,
        "import numpy as np\n"
        "from mantis._engine import apply_symmetries_batch as _asb\n"
        "def go(states, n):\n"
        "    sym = np.random.randint(0, 12, size=n)\n"
        "    return _asb(states, sym.tolist())\n",
    )
    _draws, sinks = census(root)
    assert [s[2] for s in sinks] == [f"{ENGINE_SINK_KIND} — engine batch symmetry apply"]
    with pytest.raises(UngatedSymmetryDraw):
        require_all_gated(sinks)


def test_the_engine_sink_whose_index_is_passed_by_KEYWORD_is_STILL_a_sink(tmp_path):
    """F-RT-4. The form was planted in the real `train/batch_assembly.py`, ran correctly against
    the engine, passed ruff and pyright, and the whole conformance suite reported 101 passed
    while the sink census went 6 to 5. A sink that is not recorded is not a sink that is gated;
    it is one nothing was asked about."""
    root = _tree(
        tmp_path,
        "import numpy as np\n"
        "from mantis._engine import apply_symmetries_batch as _asb\n"
        "def go(states, n):\n"
        "    return _asb(states, sym_indices=np.random.randint(0, 12, size=n).tolist())\n",
    )
    _draws, sinks = census(root)
    assert [s[2] for s in sinks] == [f"{ENGINE_SINK_KIND} — engine batch symmetry apply"]
    with pytest.raises(UngatedSymmetryDraw):
        require_all_gated(sinks)


def test_a_GATED_index_passed_by_KEYWORD_does_NOT_fire(tmp_path):
    """The negative half. A reader widened until every keyword call fires is measuring the
    spelling, not the provenance."""
    root = _tree(
        tmp_path,
        _GATE_IMPORT + "from mantis._engine import apply_symmetries_batch as _asb\n"
        "def go(states, mask):\n"
        "    sym = draw_record_syms(mask)\n"
        "    return _asb(states, sym_indices=sym.tolist())\n",
    )
    _draws, sinks = census(root)
    assert len(sinks) == 1 and sinks[0][3] is True, sinks


def test_an_engine_sink_whose_INDEX_OPERAND_cannot_be_LOCATED_is_REFUSED(tmp_path):
    """Derive-or-refuse. The alternative — skipping the call — is how the keyword form vanished
    without a single assertion noticing."""
    root = _tree(
        tmp_path,
        "from mantis._engine import apply_symmetries_batch as _asb\n"
        "def go(states, args):\n"
        "    return _asb(states, *args)\n",
    )
    with pytest.raises(UnclassifiableSink, match="cannot locate"):
        census(root)


def test_a_DECOY_gate_defined_locally_does_NOT_satisfy_the_rule(tmp_path):
    """PB-18. If the gate were matched by NAME, any module could define its own
    `draw_record_syms` returning a uniform draw and pass. The gate's identity is its import
    edge to `mantis.data.augment`."""
    root = _tree(
        tmp_path,
        "import numpy as np\n"
        "from mantis._engine import apply_symmetries_batch\n"
        "def draw_record_syms(spread):\n"
        "    return np.random.randint(0, 12, len(spread))\n"
        "def go(states, spread):\n"
        "    sym = draw_record_syms(spread)\n"
        "    return apply_symmetries_batch(states, sym.tolist())\n",
    )
    draws, sinks = census(root)
    assert draws == [], "a locally defined decoy was counted as a gate draw"
    with pytest.raises(UngatedSymmetryDraw):
        require_all_gated(sinks)


def test_an_UNGATED_draw_into_each_sink_FIRES(tmp_path):
    """PB-19, the positive control. `int(rng.integers(0, 12))` and a typed literal, fed straight
    into the engine sink and into a `get_policy_scatters(...)` subscript."""
    root = _tree(
        tmp_path,
        "import numpy as np\n"
        "from mantis._engine import apply_symmetries_batch\n"
        "from mantis.data.augment import get_policy_scatters\n"
        "def go(states, policies, rng, n):\n"
        "    scatters = get_policy_scatters(19)\n"
        "    sym = int(rng.integers(0, 12))\n"
        "    out = policies[:, scatters[sym]]\n"
        "    return apply_symmetries_batch(states, [sym] * n), out\n",
    )
    _draws, sinks = census(root)
    assert len(sinks) == 2, sinks
    assert all(not s[3] for s in sinks), sinks
    with pytest.raises(UngatedSymmetryDraw):
        require_all_gated(sinks)


def test_an_IN_PLACE_OVERWRITE_of_the_gate_drawn_array_FIRES(tmp_path):
    """PB-19b, and the cheapest spelling of the defect there is: the name stays bound to the
    gate call while every element it carries is replaced by a uniform draw. Both halves are
    here, because the discriminator is only meaningful if the SAME module without the overwrite
    line passes — otherwise this is a test that the gate call was spelled at all."""
    body = (
        "import numpy as np\n" + _GATE_IMPORT +
        "from mantis.data.augment import spread_mask\n"
        "from mantis._engine import apply_symmetries_batch\n"
        "def go(states, board_size, n):\n"
        "    sym_indices = draw_record_syms(spread_mask(board_size, states=states))\n"
        "{overwrite}"
        "    return apply_symmetries_batch(states, sym_indices.tolist())\n"
    )
    clean = _tree(tmp_path / "clean", body.format(overwrite=""))
    _draws, sinks = census(clean)
    assert len(sinks) == 1, sinks
    require_all_gated(sinks)  # must NOT raise: the same module, minus one line

    overwritten = _tree(
        tmp_path / "overwritten",
        body.format(overwrite="    sym_indices[:] = np.random.randint(0, 12, size=n)\n"),
    )
    _draws, sinks = census(overwritten)
    assert len(sinks) == 1, sinks
    with pytest.raises(UngatedSymmetryDraw):
        require_all_gated(sinks)


#: One planted line per in-place-through-a-CALL spelling. Each leaves the name bound to the
#: gate call, so a walk that records only STORES keeps resolving it to the gate.
_INPLACE_CALL_PLANTS: tuple[tuple[str, str], ...] = (
    ("np-copyto", "    np.copyto(sym_indices, np.random.randint(0, 12, size=n))\n"),
    ("method-fill", "    sym_indices.fill(3)\n"),
    ("np-put", "    np.put(sym_indices, np.arange(n), np.random.randint(0, 12, size=n))\n"),
    ("out-keyword", "    np.mod(sym_indices, 7, out=sym_indices)\n"),
    ("method-sort", "    sym_indices.sort()\n"),
)


@pytest.mark.parametrize(
    "label,line", _INPLACE_CALL_PLANTS, ids=[row[0] for row in _INPLACE_CALL_PLANTS]
)
def test_an_IN_PLACE_OVERWRITE_through_a_CALL_FIRES(label, line, tmp_path):
    """F-RT-5. `np.copyto` planted one line after the real draw in `train/batch_assembly.py`
    left the entire suite green — the store-only walk saw no store, so the name still resolved
    to the gate. Every row here is a line a numpy-heavy diff writes without thinking about it,
    and the clean half of the same module is required to pass so this is a discriminator and
    not a test that the gate call was spelled at all."""
    body = (
        "import numpy as np\n" + _GATE_IMPORT +
        "from mantis.data.augment import spread_mask\n"
        "from mantis._engine import apply_symmetries_batch\n"
        "def go(states, board_size, n):\n"
        "    sym_indices = draw_record_syms(spread_mask(board_size, states=states))\n"
        "{overwrite}"
        "    return apply_symmetries_batch(states, sym_indices.tolist())\n"
    )
    clean = _tree(tmp_path / f"clean_{label}", body.format(overwrite=""))
    _draws, sinks = census(clean)
    require_all_gated(sinks)  # the SAME module minus one line must pass

    planted = _tree(tmp_path / f"planted_{label}", body.format(overwrite=line))
    _draws, sinks = census(planted)
    assert len(sinks) == 1, sinks
    with pytest.raises(UngatedSymmetryDraw):
        require_all_gated(sinks)


def test_a_PURE_TRANSFORM_on_the_gate_drawn_array_does_NOT_fire(tmp_path):
    """The negative half of the default-unsafe method arm. `.astype`/`.tolist`/`.reshape` are
    the transforms this census already models as pure; a rule widened until they counted as
    writes would red the two production loops, which is the proxy this tier refuses."""
    root = _tree(
        tmp_path,
        "import numpy as np\n" + _GATE_IMPORT +
        "from mantis.data.augment import spread_mask\n"
        "from mantis._engine import apply_symmetries_batch\n"
        "def go(states, board_size, n):\n"
        "    sym_indices = draw_record_syms(spread_mask(board_size, states=states))\n"
        "    flat = sym_indices.astype(np.int64).reshape(-1).copy()\n"
        "    return apply_symmetries_batch(states, flat.tolist())\n",
    )
    _draws, sinks = census(root)
    assert len(sinks) == 1 and sinks[0][3] is True, sinks


def test_the_TWO_PRODUCTION_LOOPS_do_NOT_fire(tmp_path):
    """PB-20, the NEGATIVE control, and a landing condition rather than a nicety.

    `for sym in range(12):` with a gate-drawn row mask is live and CORRECT at two production
    sites. A discriminator that reds on them is measuring "a 12-literal near a scatter" — the
    proxy — instead of "obtains a D6 element to apply", and would then be weakened until it
    fired on nothing at all.
    """
    root = _tree(
        tmp_path,
        "import numpy as np\n" + _GATE_IMPORT +
        "from mantis.data.augment import spread_mask\n"
        "def go(states_flat, policies, board_size, has_pass, n, spatial):\n"
        "    scatters = get_policy_scatters(board_size, has_pass=has_pass)\n"
        "    sym_indices = draw_record_syms(spread_mask(board_size, states=states_flat))\n"
        "    augmented = np.empty_like(states_flat)\n"
        "    for sym in range(12):\n"
        "        mask_idx = np.where(sym_indices == sym)[0]\n"
        "        if mask_idx.size == 0:\n"
        "            continue\n"
        "        sc = scatters[sym]\n"
        "        augmented[mask_idx] = states_flat[mask_idx][:, :, sc[:spatial]]\n"
        "    scattered = np.empty_like(policies)\n"
        "    for i in range(n):\n"
        "        scattered[i] = policies[i][scatters[int(sym_indices[i])]]\n"
        "    return augmented, scattered\n",
    )
    draws, sinks = census(root)
    assert len(draws) == 1, draws
    assert len(sinks) == 2, sinks
    require_all_gated(sinks)  # must NOT raise


def test_an_enumeration_loop_masked_against_a_NON_GATE_array_FIRES(tmp_path):
    """The enumeration criterion's first dangerous variant. "It is an enumeration loop" does
    not excuse it: the row→element association is carried by the mask, and this mask is not
    gate-derived."""
    root = _tree(
        tmp_path,
        "import numpy as np\n" + _GATE_IMPORT +
        "def go(states_flat, board_size, n):\n"
        "    scatters = get_policy_scatters(board_size)\n"
        "    rng_drawn = np.random.randint(0, 12, size=n)\n"
        "    out = np.empty_like(states_flat)\n"
        "    for sym in range(12):\n"
        "        mask_idx = np.where(rng_drawn == sym)[0]\n"
        "        sc = scatters[sym]\n"
        "        out[mask_idx] = states_flat[mask_idx][:, sc]\n"
        "    return out\n",
    )
    _draws, sinks = census(root)
    assert [s[3] for s in sinks] == [False], sinks
    with pytest.raises(UngatedSymmetryDraw):
        require_all_gated(sinks)


def test_an_enumeration_loop_applied_to_ALL_ROWS_unconditionally_FIRES(tmp_path):
    """The second dangerous variant: no row mask at all, so every row receives every element."""
    root = _tree(
        tmp_path,
        "import numpy as np\n" + _GATE_IMPORT +
        "def go(states_flat, board_size):\n"
        "    scatters = get_policy_scatters(board_size)\n"
        "    out = np.empty_like(states_flat)\n"
        "    for sym in range(12):\n"
        "        sc = scatters[sym]\n"
        "        out = states_flat[:, sc]\n"
        "    return out\n",
    )
    _draws, sinks = census(root)
    assert [s[3] for s in sinks] == [False], sinks
    with pytest.raises(UngatedSymmetryDraw):
        require_all_gated(sinks)


@pytest.mark.parametrize(
    "chain",
    [
        "sym.astype(np.int64)",
        "sym.astype(np.uint64).tolist()",
        "sym.tolist()",
        "[int(sym[i]) for i in range(n)][0]",
    ],
    ids=["astype", "astype_tolist", "tolist", "int_subscript"],
)
def test_each_TRANSFORM_CHAIN_shape_fires_on_a_NON_GATE_definition(tmp_path, chain):
    """PB-21. Production needs each of these chains modelled in both directions; a chain the
    walk cannot follow is an escape hatch, not a limitation, so each is planted with an
    ungated definition and must fire."""
    root = _tree(
        tmp_path,
        "import numpy as np\n"
        "from mantis._engine import apply_symmetries_batch\n"
        "def go(states, n):\n"
        "    sym = np.random.randint(0, 12, size=n)\n"
        f"    return apply_symmetries_batch(states, {chain})\n",
    )
    _draws, sinks = census(root)
    assert len(sinks) == 1, sinks
    with pytest.raises(UngatedSymmetryDraw):
        require_all_gated(sinks)


def test_an_ARCH_IMPORT_planted_on_the_augmentation_path_FIRES(tmp_path):
    """PB-22. Clause (ii)'s break — the "no capability route" half in its structural form."""
    root = _tree(
        tmp_path,
        "import numpy as np\n" + _GATE_IMPORT +
        "from mantis.model.arch import GnnArch\n"
        "from mantis.data.augment import spread_mask\n"
        "def go(states, board_size):\n"
        "    scatters = get_policy_scatters(board_size)\n"
        "    sym_indices = draw_record_syms(spread_mask(board_size, states=states))\n"
        "    return scatters[int(sym_indices[0])], GnnArch\n",
    )
    assert augmentation_path_modules(root) == ("mod.py",)
    assert arch_imports_on_path(root) == ["mod.py: mantis.model.arch.GnnArch"]
    with pytest.raises(ArchImportInAugmentationPath, match="mantis.model.arch.GnnArch"):
        require_no_arch_imports(root)
