"""T2a — no member of the `ModelArch` union declares a symmetry claim, of any type.

AUTHORITY AND SCOPE. R307(b) DELETED `caps.exact_symmetries`; this tier is the PARTIAL
implementation of `plan/DESIGN_ARCHCAPS.md` exit criterion 5(a). It is partial because 5(a)
also covers `ArchCaps`' own fields, which do not exist at HEAD — when `ArchCaps` lands the same
AST walk extends to it with NO rule change. **Criterion 5 is therefore recorded as PARTIALLY
discharged with its residue named**, never as discharged: a criterion recorded as satisfied by
a check that cannot see part of its subject is the overclaiming class.

ONE PRODUCER. One subject (`ModelArch`), one structural predicate. A green means "no symmetry
claim has appeared", which is exactly what the module name says and not one word more.

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
            elif isinstance(item, ast.Subscript):  # Union[...] / Annotated[...] spellings
                stack.append(item.slice)
            elif isinstance(item, ast.Tuple):
                stack.extend(item.elts)
    return tuple(sorted(set(members)))


def declared_names(path: Path, member: str) -> tuple[str, ...]:
    """Every field, `ClassVar`, property and method name declared on one member class."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != member:
            continue
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
    inspected = sum(len(declared_names(ARCH_MODULE, m)) for m in members)
    derived("t2a.declarations_inspected", inspected)
    assert inspected > 0, "the walk inspected zero declarations — it is measuring nothing"
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
