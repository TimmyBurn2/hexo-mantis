# >300 justify (R8): the union reader, the per-member resolver that finds each member where it
# is DEFINED, and the controls that show each of them can reject are ONE unit. A resolver whose
# controls live in another file can be narrowed back to a single-file walk with nothing in the
# same file going red — which is exactly how a member in another module went uninspected while
# an aggregate vacuity guard reported the walk as measuring something.
"""T2a — no member of the `ModelArch` union declares a symmetry claim, of any type.

AUTHORITY AND SCOPE. R307(b) DELETED `caps.exact_symmetries`; this tier is the PARTIAL
implementation of `plan/DESIGN_ARCHCAPS.md` exit criterion 5(a). It is partial because 5(a)
also covers `ArchCaps`' own fields, which do not exist at HEAD — when `ArchCaps` lands the same
AST walk extends to it with NO rule change. **Criterion 5 is therefore recorded as PARTIALLY
discharged with its residue named**, never as discharged: a criterion recorded as satisfied by
a check that cannot see part of its subject is the overclaiming class.

ONE PRODUCER. One subject (`ModelArch`), one structural predicate. A green means "no symmetry
claim has appeared", which is exactly what the module name says and not one word more.

EVERY MEMBER IS INSPECTED WHERE IT IS DEFINED, AND THE VACUITY GUARD IS PER-MEMBER. The walk
used to parse the union's own file and match a `ClassDef` by name there, so a member imported
from another module contributed ZERO inspected declarations while the aggregate `inspected > 0`
guard was satisfied by its siblings — a member carrying `exact_symmetries` passed. A new arch
in its own module imported into the union is the ordinary way this file grows, so the resolver
follows the import edge (`from … import X`, relative or absolute, re-export chains, and the
`module.X` spelling of a union operand) to the file that defines the member and walks the
`ClassDef` THERE. A member that cannot be located as a class definition is refused BY NAME
rather than silently contributing nothing. What the per-member guard asserts is that each
member's body was WALKED — not that it declared at least one name: a genuinely empty member
declares no claim, and refusing it would be a false red rather than a stronger check.

MECHANISM IS AST, NEVER REGEX (R296(f)). The adjacent `tests/model/test_arch_ban.py` guards a
DIFFERENT subject (the arch-off-module sniff) with a regex; this tier shares no subject with it
and deliberately does not inherit its mechanism.

CASE POSTURE, STATED (R297(b)): matching is CASE-INSENSITIVE over the declared name, so
`D6_MAP` and `Symmetries` fire. The family is ENUMERATED and NON-EXHAUSTIVE — `symmetr*`,
`automorphism*`, `equivarian*`, `d6`, `p6m`, `sym_*` — and it is a NAME family, so a
value-level claim on an innocuously named field (`augmentation_policy: Literal["d6", …]`) is
outside this mechanism. That residue is real and is not papered over; the near-miss negative
control below exists to stop the family being quietly widened until it fires on ordinary
fields.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from _corpus import ConformanceRefusal

ARCH_MODULE = Path(__file__).resolve().parents[3] / "src" / "mantis" / "model" / "arch.py"
UNION_NAME = "ModelArch"
#: Import-edge hops followed while locating a member. A re-export chain is finite; a cycle is
#: broken by the visited set below, and this only bounds pathological depth.
_MAX_IMPORT_HOPS = 8

#: The enumerated, non-exhaustive symmetry-name family. Case-folded before matching.
_FAMILY: tuple[re.Pattern[str], ...] = (
    re.compile(r"symmetr"),
    re.compile(r"automorphism"),
    re.compile(r"equivarian"),
    re.compile(r"(^|_)d6($|_)"),
    re.compile(r"(^|_)p6m($|_)"),
    re.compile(r"(^|_)sym($|_)"),
)


class SymmetryClaimOnArchDeclaration(ConformanceRefusal):
    """A member of the arch union declares a symmetry-named field, property or method."""


class ArchUnionUnresolved(ConformanceRefusal):
    """The union's member set could not be resolved, so the walk inspected zero classes."""


class ArchMemberNotLocated(ConformanceRefusal):
    """A resolved union member was never found as a class definition, so nothing on it was
    inspected — while the other members satisfied an aggregate vacuity guard."""


def is_symmetry_named(name: str) -> bool:
    folded = name.lower()
    return any(pattern.search(folded) for pattern in _FAMILY)


def union_members(path: Path) -> tuple[str, ...]:
    """The member class names of `ModelArch`, READ off the PEP-604 `BinOp`, never transcribed.

    `ModelArch = CnnArch | GnnArch` (`src/mantis/model/arch.py:73`) is a `BinOp(BitOr)`. A walk
    that fails to resolve it inspects zero classes and passes — which is why the member set is
    returned and asserted rather than used silently.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    members: list[str] = []
    for node in ast.walk(tree):
        targets = (
            node.targets if isinstance(node, ast.Assign)
            else [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        if not any(isinstance(t, ast.Name) and t.id == UNION_NAME for t in targets):
            continue
        stack = [node.value]
        while stack:
            item = stack.pop()
            if isinstance(item, ast.BinOp) and isinstance(item.op, ast.BitOr):
                stack.extend([item.left, item.right])
            elif isinstance(item, ast.Name):
                members.append(item.id)
            elif isinstance(item, ast.Attribute):
                # `orbit.OrbitArch` — recorded with its dotted prefix rather than DROPPED, which
                # is what an operand-shape the reader ignores does: it removes a member from the
                # subject without removing it from the union.
                dotted = _dotted(item)
                if dotted:
                    members.append(dotted)
            elif isinstance(item, ast.Constant) and isinstance(item.value, str):
                members.append(item.value)  # a string forward reference is still a member
            elif isinstance(item, ast.Subscript):  # Union[...] / Annotated[...] spellings
                stack.append(item.slice)
            elif isinstance(item, ast.Tuple):
                stack.extend(item.elts)
    return tuple(sorted(set(members)))


def _dotted(node: ast.expr) -> str:
    """`a.b.C` for an attribute chain over plain names; `""` for anything else."""
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return ""
    parts.append(current.id)
    return ".".join(reversed(parts))


def package_root(path: Path) -> tuple[Path, str]:
    """`(import root, dotted package)` for a module, derived by walking `__init__.py` upwards.

    Derived rather than typed so every control below can drive the resolver against a temp tree
    with its own package layout — a resolver whose root is hard-coded is one no planted break
    can reach.
    """
    parts: list[str] = []
    directory = path.parent
    while (directory / "__init__.py").is_file():
        parts.append(directory.name)
        directory = directory.parent
    return directory, ".".join(reversed(parts))


def _module_file(root: Path, dotted: str) -> Path | None:
    if not dotted:
        return None
    base = root.joinpath(*dotted.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _absolute_module(node: ast.ImportFrom, package: str) -> str:
    """The dotted module an `ImportFrom` names, with a relative import resolved against its
    own package — `from .orbit import X` inside `mantis.model` is `mantis.model.orbit`."""
    if not node.level:
        return node.module or ""
    parts = package.split(".") if package else []
    kept = parts[: len(parts) - (node.level - 1)] if node.level > 1 else parts
    return ".".join([*kept, node.module]) if node.module else ".".join(kept)


def locate_member(path: Path, member: str, _seen: frozenset[Path] = frozenset()) -> (
    tuple[Path, ast.ClassDef] | None
):
    """`(defining file, its ClassDef)` for one union member, or `None` if it cannot be located.

    Resolution order: a `ClassDef` of that name in `path`; else the import edge in `path` that
    binds the name, followed into the file it names; else, for a dotted member, the module that
    its prefix binds. THE POINT IS THAT THE SECOND CASE EXISTS: matching a `ClassDef` by name in
    the union's own file returns nothing for an imported member, and returning nothing is
    indistinguishable from "declares no symmetry claim".
    """
    if path in _seen or len(_seen) > _MAX_IMPORT_HOPS or not path.is_file():
        return None
    tree = ast.parse(path.read_text(encoding="utf-8"))
    root, package = package_root(path)
    if "." in member:
        prefix, _, leaf = member.rpartition(".")
        target = _module_file(root, _module_alias_target(tree, prefix))
        return locate_member(target, leaf, _seen | {path}) if target is not None else None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == member:
            return (path, node)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        for alias in node.names:
            if (alias.asname or alias.name) != member:
                continue
            target = _module_file(root, _absolute_module(node, package))
            if target is not None:
                return locate_member(target, alias.name, _seen | {path})
    return None


def _module_alias_target(tree: ast.AST, prefix: str) -> str:
    """The dotted module a member's `a.b` prefix refers to, through `import`/`from … import`."""
    head, _, rest = prefix.partition(".")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if (alias.asname or alias.name.split(".")[0]) == head:
                    base = alias.name if alias.asname else head
                    return f"{base}.{rest}" if rest else base
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if (alias.asname or alias.name) == head:
                    module = _absolute_module(node, "")
                    base = f"{module}.{alias.name}" if module else alias.name
                    return f"{base}.{rest}" if rest else base
    return prefix


def declared_names(path: Path, member: str) -> tuple[str, ...]:
    """Every field, `ClassVar`, property and method name declared on one member class.

    `path` is the union's file; the member is walked at the file that DEFINES it, which is not
    the same file in general. An unlocatable member returns `()` here and is refused by name at
    the gate — never treated as a member that declares nothing.
    """
    located = locate_member(path, member)
    return () if located is None else declarations_of(located[1])


def declarations_of(node: ast.ClassDef) -> tuple[str, ...]:
    names: list[str] = []
    for stmt in node.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            names.append(stmt.target.id)
        elif isinstance(stmt, ast.Assign):
            names.extend(t.id for t in stmt.targets if isinstance(t, ast.Name))
        elif isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef):
            names.append(stmt.name)
    return tuple(names)


def symmetry_claims(path: Path, members: tuple[str, ...]) -> tuple[str, ...]:
    """`Member.name` for every symmetry-named declaration on any union member."""
    return tuple(
        f"{member}.{name}"
        for member in members
        for name in declared_names(path, member)
        if is_symmetry_named(name)
    )


def require_union_resolved(members: tuple[str, ...], path: Path) -> int:
    if not members:
        raise ArchUnionUnresolved(
            f"the {UNION_NAME} union in {path} resolved to ZERO members, so the symmetry walk "
            "inspected zero classes and would pass for the wrong reason."
        )
    return len(members)


def require_every_member_located(path: Path, members: tuple[str, ...]) -> dict[str, str]:
    """THE PER-MEMBER VACUITY GUARD. Returns `member -> defining file`; refuses by name.

    The guard it replaces was `sum(len(declared_names(...)) for member in members) > 0`, an
    AGGREGATE: two members with fields satisfied it while a third contributed nothing because
    the walk could not see its file at all. An aggregate vacuity guard over a subject whose
    members are inspected independently reports "this measured something" when it measured a
    proper subset — the overclaiming class, inside the tier written to refuse it.
    """
    sites: dict[str, str] = {}
    unlocated: list[str] = []
    for member in members:
        located = locate_member(path, member)
        if located is None:
            unlocated.append(member)
        else:
            sites[member] = located[0].name
    if unlocated:
        raise ArchMemberNotLocated(
            f"union members {sorted(unlocated)} resolved into {UNION_NAME} but were never "
            f"located as a class definition reachable from {path.name}, so ZERO of their "
            "declarations were inspected. A member the walk cannot see is not a member that "
            "declares no symmetry claim; it is a member this tier has no evidence about."
        )
    return sites


def require_no_symmetry_claim(offenders: tuple[str, ...]) -> None:
    if offenders:
        raise SymmetryClaimOnArchDeclaration(
            f"arch declarations carry symmetry claims: {list(offenders)}. R307(b) deleted "
            "`caps.exact_symmetries` because a per-arch symmetry claim is a per-position fact "
            "wearing an architecture-level constant; a callable or a gate pointer is barred by "
            "the same rule as a plain field."
        )


def test_no_member_of_the_arch_union_declares_a_symmetry_claim(derived):
    members = union_members(ARCH_MODULE)
    derived("t2a.union_members", members)
    derived("t2a.union_cardinality", require_union_resolved(members, ARCH_MODULE))
    derived("t2a.member_defining_files", require_every_member_located(ARCH_MODULE, members))
    per_member = {m: len(declared_names(ARCH_MODULE, m)) for m in members}
    derived("t2a.declarations_inspected_per_member", per_member)
    derived("t2a.declarations_inspected", sum(per_member.values()))
    require_no_symmetry_claim(symmetry_claims(ARCH_MODULE, members))


def test_an_UNRESOLVED_union_is_refused(tmp_path):
    """PB-11. The walk reports the member set it inspected, and an empty one FAILS."""
    stub = tmp_path / "arch.py"
    stub.write_text("CnnArch = object\nModelArch = 3\n", encoding="utf-8")
    assert union_members(stub) == ()
    with pytest.raises(ArchUnionUnresolved, match="ZERO members"):
        require_union_resolved(union_members(stub), stub)


def test_a_THIRD_union_member_carrying_a_symmetry_field_is_caught(tmp_path):
    """PB-12. Proves the union is READ, not transcribed: a hard-coded two-member walk passes."""
    stub = tmp_path / "arch.py"
    stub.write_text(
        "class CnnArch:\n    board_size: int\n\n"
        "class GnnArch:\n    in_dim: int\n\n"
        "class OrbitArch:\n    exact_symmetries: tuple[int, ...]\n\n"
        "ModelArch = CnnArch | GnnArch | OrbitArch\n",
        encoding="utf-8",
    )
    members = union_members(stub)
    assert members == ("CnnArch", "GnnArch", "OrbitArch"), members
    with pytest.raises(SymmetryClaimOnArchDeclaration, match="OrbitArch.exact_symmetries"):
        require_no_symmetry_claim(symmetry_claims(stub, members))


def _package(tmp_path: Path, modules: dict[str, str]) -> Path:
    """A temp import tree — `dotted module name -> source` — under a package root, so the
    resolver is driven over a real import edge rather than a stubbed one. Returns `arch.py`."""
    root = tmp_path / "src"
    for dotted, body in modules.items():
        target = root.joinpath(*dotted.split("."))
        target.parent.mkdir(parents=True, exist_ok=True)
        for level in range(len(dotted.split(".")) - 1):
            init = root.joinpath(*dotted.split(".")[: level + 1]) / "__init__.py"
            init.parent.mkdir(parents=True, exist_ok=True)
            init.touch()
        target.with_suffix(".py").write_text(body, encoding="utf-8")
    return root / "mantis" / "model" / "arch.py"


def test_a_member_defined_in_ANOTHER_MODULE_is_inspected_where_it_is_DEFINED(tmp_path):
    """F-RT-1. The member is correctly resolved INTO the union and its declarations were
    inspected zero times, so a `exact_symmetries` on it passed. This is the ordinary shape of a
    new arch: its own module, imported into the union."""
    arch = _package(
        tmp_path,
        {
            "mantis.model.arch": (
                "from mantis.model.orbit import OrbitArch\n\n"
                "class CnnArch:\n    board_size: int\n\n"
                "class GnnArch:\n    in_dim: int\n\n"
                "ModelArch = CnnArch | GnnArch | OrbitArch\n"
            ),
            "mantis.model.orbit": (
                "class OrbitArch:\n    exact_symmetries: tuple[int, ...] = ()\n"
            ),
        },
    )
    members = union_members(arch)
    assert members == ("CnnArch", "GnnArch", "OrbitArch"), members
    assert require_every_member_located(arch, members)["OrbitArch"] == "orbit.py"
    assert declared_names(arch, "OrbitArch") == ("exact_symmetries",)
    with pytest.raises(SymmetryClaimOnArchDeclaration, match="OrbitArch.exact_symmetries"):
        require_no_symmetry_claim(symmetry_claims(arch, members))


def test_a_RELATIVE_import_and_a_DOTTED_operand_resolve_to_the_same_definition(tmp_path):
    """The two other spellings of the same edge. `from .orbit import X` and `orbit.OrbitArch`
    as a union operand both have to reach `orbit.py`; the dotted operand additionally has to
    survive the union reader, which used to drop a non-`Name` operand without a word."""
    arch = _package(
        tmp_path,
        {
            "mantis.model.arch": (
                "from mantis.model import orbit\n"
                "from .halo import HaloArch\n\n"
                "class CnnArch:\n    board_size: int\n\n"
                "ModelArch = CnnArch | orbit.OrbitArch | HaloArch\n"
            ),
            "mantis.model.orbit": "class OrbitArch:\n    d6_orbit_table: int = 0\n",
            "mantis.model.halo": "class HaloArch:\n    hidden: int = 8\n",
        },
    )
    members = union_members(arch)
    assert members == ("CnnArch", "HaloArch", "orbit.OrbitArch"), members
    sites = require_every_member_located(arch, members)
    assert sites["orbit.OrbitArch"] == "orbit.py" and sites["HaloArch"] == "halo.py"
    with pytest.raises(SymmetryClaimOnArchDeclaration, match="d6_orbit_table"):
        require_no_symmetry_claim(symmetry_claims(arch, members))


def test_a_member_that_cannot_be_LOCATED_is_refused_BY_NAME(tmp_path):
    """The per-member vacuity guard, driven where the aggregate one is satisfied: two members
    with fields and one whose module does not exist. The old aggregate `inspected > 0` passes
    on this input, which is the whole finding."""
    arch = _package(
        tmp_path,
        {
            "mantis.model.arch": (
                "from mantis.model.nowhere import GhostArch\n\n"
                "class CnnArch:\n    board_size: int\n\n"
                "class GnnArch:\n    in_dim: int\n\n"
                "ModelArch = CnnArch | GnnArch | GhostArch\n"
            ),
        },
    )
    members = union_members(arch)
    assert sum(len(declared_names(arch, m)) for m in members) > 0, (
        "the aggregate guard this replaces is SATISFIED here — that is why it is not the guard"
    )
    with pytest.raises(ArchMemberNotLocated, match="GhostArch"):
        require_every_member_located(arch, members)


def test_the_LOCATOR_does_NOT_fire_on_the_real_union():
    """Negative control. A locator that cannot find the shipped union's members would red the
    tier for its own reason rather than for its subject's. It asserts that every member was
    LOCATED — not WHERE: which file a member lives in is the thing this tier stopped caring
    about, and pinning it here would red on the ordinary refactor the fix exists to follow."""
    members = union_members(ARCH_MODULE)
    sites = require_every_member_located(ARCH_MODULE, members)
    assert set(sites) == set(members)
    assert all(site.endswith(".py") for site in sites.values()), sites


def test_a_symmetry_claim_fires_as_a_VALUE_a_CALLABLE_and_a_GATE_POINTER(tmp_path):
    """PB-13. R307(b) bars the claim of ANY type, so all three spellings must fire."""
    stub = tmp_path / "arch.py"
    stub.write_text(
        "class CnnArch:\n"
        "    exact_symmetries: tuple[int, ...] = ()\n"
        "    equivariance_check: Callable[[int], bool] | None = None\n"
        "    def d6_gate(self):\n        return None\n\n"
        "ModelArch = CnnArch\n",
        encoding="utf-8",
    )
    members = union_members(stub)
    offenders = symmetry_claims(stub, members)
    assert set(offenders) == {
        "CnnArch.exact_symmetries", "CnnArch.equivariance_check", "CnnArch.d6_gate",
    }, offenders
    with pytest.raises(SymmetryClaimOnArchDeclaration):
        require_no_symmetry_claim(offenders)


def test_the_stated_CASE_posture_is_pinned_by_its_own_control():
    """PB-14. Case-insensitive is a POSTURE, not an accident: `D6_ORBIT: ClassVar[int]` walks
    straight through a case-sensitive `d6` match, and that is the shape the family is for."""
    assert is_symmetry_named("D6_MAP")
    assert is_symmetry_named("Symmetries")
    assert is_symmetry_named("EQUIVARIANCE_TOL")
    assert is_symmetry_named("sym_table_id")


def test_the_family_does_NOT_fire_on_a_NEAR_MISS():
    """PB-15, a negative control and as binding as any positive one. A tier that fires on the
    near miss is measuring a proxy, and the family would then be widened until it flagged
    ordinary fields — which is how a guard's green stops meaning its name."""
    for benign in ("dihedral_order", "res_blocks", "policy_hidden", "n_value_bins", "system"):
        assert not is_symmetry_named(benign), benign
