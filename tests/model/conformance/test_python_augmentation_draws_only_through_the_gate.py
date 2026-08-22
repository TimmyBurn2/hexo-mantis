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
`sample_graph_batch(augment=True)`.

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
GRAPH_SINK = "sample_graph_batch"
#: Arch / capability symbols that may not be imported into the augmentation path (clause ii).
ARCH_SYMBOLS = frozenset({"ModelArch", "CnnArch", "GnnArch", "ArchCaps", "arch_from_spec_and_config"})
ARCH_MODULES = ("mantis.model.arch",)

_TRANSFORM_METHODS = frozenset(
    {"astype", "tolist", "ravel", "flatten", "reshape", "copy", "item", "squeeze"}
)
_TRANSFORM_FUNCTIONS = frozenset({"int", "list", "tuple", "asarray", "array", "int64", "intp"})


class UngatedSymmetryDraw(ConformanceRefusal):
    """A D6 element reaches a symmetry application without coming through the per-record gate."""


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
                    if isinstance(target, ast.Name):
                        self.definitions.setdefault(target.id, []).append(node.value)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.value is not None:
                    self.definitions.setdefault(node.target.id, []).append(node.value)
            elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
                if isinstance(node.iter, ast.Call) and _callee_name(node.iter.func) == "range":
                    self.range_loops[node.target.id] = node
        for name, values in self.definitions.items():
            if any(self._is_scatter_factory_call(v) for v in values):
                self.scatter_names.add(name)

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
            if is_engine_sink and len(node.args) >= 2:
                sinks.append(
                    ("apply_symmetries_batch.arg2", node.lineno, "engine batch symmetry apply",
                     resolves_to_gate(node.args[1], facts))
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
                        ("sample_graph_batch.augment", node.lineno, "rust-side graph draw", True)
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
    require_all_gated(sinks)


def test_the_augmentation_path_module_set_is_non_empty_and_imports_no_arch_symbol(derived):
    """Clause (ii). "No module in the augmentation path imports an arch symbol" is vacuous if
    the path resolves to zero modules, so the module set is pinned as a derived output first."""
    modules = augmentation_path_modules(SRC)
    derived("t2b.augmentation_path_modules", modules)
    derived("t2b.augmentation_path.cardinality", require_non_empty(list(modules), "module-set"))
    offenders = arch_imports_on_path(SRC)
    if offenders:
        raise ArchImportInAugmentationPath(
            "the augmentation path imports arch/capability symbols: " + ", ".join(offenders)
        )


# --------------------------------------------------------------------------------------- #
# Planted breaks and controls — all DEFAULT tier
# --------------------------------------------------------------------------------------- #
def _tree(tmp_path: Path, body: str, name: str = "mod.py") -> Path:
    root = tmp_path / "mantis"
    root.mkdir(exist_ok=True)
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
    assert [s[2] for s in sinks] == ["apply_symmetries_batch.arg2 — engine batch symmetry apply"]
    with pytest.raises(UngatedSymmetryDraw):
        require_all_gated(sinks)


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
