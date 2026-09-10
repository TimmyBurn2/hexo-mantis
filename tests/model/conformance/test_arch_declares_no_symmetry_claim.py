# >300 justify (R8): the union reader, the per-member resolver that finds each member where it
# is DEFINED, and the controls that show each of them can reject are ONE unit.
"""No member of the `ModelArch` union declares a symmetry claim, of any type.

PARTIAL: `ArchCaps`' own fields do not exist at HEAD, and the same AST walk extends to them.

Every member is inspected WHERE IT IS DEFINED and the vacuity guard is PER-MEMBER: matching a
`ClassDef` by name in the union's own file inspected nothing for an imported member while an
aggregate guard was satisfied by its siblings. Matching is CASE-INSENSITIVE and the name family
is ENUMERATED and NON-EXHAUSTIVE, so a value-level claim on a benign name is out of scope.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from _corpus import ConformanceRefusal

ARCH_MODULE = Path(__file__).resolve().parents[3] / "src" / "mantis" / "model" / "arch.py"
UNION_NAME = "ModelArch"
#: Import-edge hops followed while locating a member; a cycle is broken by the visited set, so
#: this only bounds pathological depth.
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
    """A resolved union member was never found as a class definition, so nothing was inspected."""


def is_symmetry_named(name: str) -> bool:
    folded = name.lower()
    return any(pattern.search(folded) for pattern in _FAMILY)


def union_members(path: Path) -> tuple[str, ...]:
    """The member class names of `ModelArch`, READ off the PEP-604 `BinOp`, never transcribed."""
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
                # Recorded with its dotted prefix rather than DROPPED: an ignored operand shape
                # removes a member from the subject, not from the union.
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
    """`(import root, dotted package)`, derived by walking `__init__.py` upwards so the controls
    can drive the resolver against a temp tree."""
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
    """The dotted module an `ImportFrom` names, relative imports resolved against their package."""
    if not node.level:
        return node.module or ""
    parts = package.split(".") if package else []
    kept = parts[: len(parts) - (node.level - 1)] if node.level > 1 else parts
    return ".".join([*kept, node.module]) if node.module else ".".join(kept)


def locate_member(path: Path, member: str, _seen: frozenset[Path] = frozenset()) -> (
    tuple[Path, ast.ClassDef] | None
):
    """`(defining file, its ClassDef)` for one union member, or `None` if it cannot be located.

    Resolution order: a `ClassDef` of that name in `path`; else the import edge that binds the
    name; else, for a dotted member, the module its prefix binds. The second case is the point:
    matching by name in the union's own file returns nothing for an imported member, which is
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
    """Every name declared on one member class, walked at the file that DEFINES it; an
    unlocatable member returns `()` and is refused by name at the gate."""
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

    The aggregate guard it replaces was satisfied by two members with fields while a third
    contributed nothing because the walk could not see its file at all.
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
    """The walk reports the member set it inspected, and an empty one FAILS."""
    stub = tmp_path / "arch.py"
    stub.write_text("CnnArch = object\nModelArch = 3\n", encoding="utf-8")
    assert union_members(stub) == ()
    with pytest.raises(ArchUnionUnresolved, match="ZERO members"):
        require_union_resolved(union_members(stub), stub)


def test_a_THIRD_union_member_carrying_a_symmetry_field_is_caught(tmp_path):
    """The union is READ, not transcribed: a hard-coded two-member walk passes this."""
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
    """A temp import tree under a package root, so the resolver runs over a real import edge."""
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
    """A member defined in another module is inspected there — the ordinary shape of a new arch,
    and the one whose declarations used to be inspected zero times."""
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
    """Both other spellings of the edge reach `orbit.py`; the dotted operand must also survive
    the union reader, which used to drop a non-`Name` operand without a word."""
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
    """The per-member guard driven where the aggregate one is satisfied."""
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
    """Negative control: every member was LOCATED, not WHERE — pinning the file would red on the
    ordinary refactor this resolver exists to follow."""
    members = union_members(ARCH_MODULE)
    sites = require_every_member_located(ARCH_MODULE, members)
    assert set(sites) == set(members)
    assert all(site.endswith(".py") for site in sites.values()), sites


def test_a_symmetry_claim_fires_as_a_VALUE_a_CALLABLE_and_a_GATE_POINTER(tmp_path):
    """The claim is barred whatever its type, so all three spellings must fire."""
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
    """Case-insensitive is a POSTURE: `D6_ORBIT` walks straight through a case-sensitive match."""
    assert is_symmetry_named("D6_MAP")
    assert is_symmetry_named("Symmetries")
    assert is_symmetry_named("EQUIVARIANCE_TOL")
    assert is_symmetry_named("sym_table_id")


def test_the_family_does_NOT_fire_on_a_NEAR_MISS():
    """A negative control, as binding as any positive one: firing on the near miss measures a
    proxy."""
    for benign in ("dihedral_order", "res_blocks", "policy_hidden", "n_value_bins", "system"):
        assert not is_symmetry_named(benign), benign
