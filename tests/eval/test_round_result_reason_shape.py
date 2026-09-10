"""The round-result mapping carries ONE authority for "did this round break", and the detail
beside it is PROSE nobody branches on.

No import anchor is used, deliberately: the oracles must fail because the SHAPE is wrong, not
because a module is missing. At HEAD `build_round_result` takes BOTH `eval_broken: bool` and
`error: str | None` and writes both into the mapping, so `(eval_broken=True, error=None)` is a
legal call producing a round that is broken and says nothing about why. The answer is one field,
`eval_broken_reason` with `None` as the clean state, plus `eval_broken_detail` for detail ONLY.
The legacy names must be DELETED, not defaulted — a defaulted parameter is a MIGRATED authority,
not an absent one — and the detail census is paired with a non-vacuity premise, since a census
over a name nothing produces passes for free. MUTATIONS THAT RED IT: re-add either legacy name,
or branch on `result["eval_broken_detail"]` anywhere under `src/`.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

from mantis.eval.rounds import build_round_result

_SRC = Path(__file__).resolve().parents[2] / "src" / "mantis"

#: The two names R79 deletes. Named here, not derived, so a rename cannot satisfy the pin.
_DELETED_NAMES = ("eval_broken", "error")
#: The two MAPPING KEYS that replace them — one authority for the fact, one slot for the
#: prose. Read against the routed result, never against the signature (see below).
_REQUIRED_NAMES = ("eval_broken_reason", "eval_broken_detail")
#: The two PARAMETER names — the SIGNATURE half's expectation, deliberately NOT the mapping-key
#: tuple: no signature can declare `eval_broken_reason`/`eval_broken_detail` as required
#: parameters and also accept the `reason=`/`detail=` call below.
_REQUIRED_PARAMS = ("reason", "detail")


def _clean_round_result() -> dict:
    """A CLEAN round built through the real builder. `reason=None` is the clean state and needs
    no enum member, so this oracle stays independent of `mantis.eval.errors`."""
    return build_round_result(
        step=1000, round_id="r000001_1000", rungs_config=[], rung_results={},
        gate_result=None, skipped_rungs=[], bt={"ratings": {}, "p_hat": {}},
        schedule_next={}, eval_round_wall_sec=1.5, reason=None, detail=None, random_wr=None,
    )


def test_build_round_result_declares_no_eval_broken_bool_and_no_error_string() -> None:
    """Signature half: deleted, never defaulted. `eval_broken: bool = False` would keep every
    call site compiling and `(eval_broken=True, error=None)` constructible; broken-ness is a
    property of the resolved reason, so `reason is None` ⟺ clean with nothing to disagree."""
    params = inspect.signature(build_round_result).parameters
    present = [name for name in _DELETED_NAMES if name in params]
    assert present == [], (
        f"build_round_result still declares {present} — R79: one fact, one field. A "
        "DEFAULTED survivor is a migrated authority, not an absent one (MF-2 Attack B). "
        f"Full parameter list: {list(params)}"
    )
    # `_REQUIRED_PARAMS`, not `_REQUIRED_NAMES`: the signature and the mapping are two surfaces
    # with two vocabularies, and this half owns the parameter one.
    missing = [name for name in _REQUIRED_PARAMS if name not in params]
    assert missing == [], (
        f"build_round_result must take {list(_REQUIRED_PARAMS)}; missing {missing}. "
        f"Full parameter list: {list(params)}"
    )
    for name in _REQUIRED_PARAMS:
        assert params[name].default is inspect.Parameter.empty, (
            f"{name!r} carries a default ({params[name].default!r}) — a defaulted reason "
            "lets a caller build a round result without ever deciding whether it broke"
        )


def test_the_round_result_mapping_carries_the_reason_and_neither_legacy_key() -> None:
    """Mapping half: the signature and the payload are two surfaces and either can drift alone —
    a builder could take `reason=` and still write `"eval_broken"` into the mapping for
    "compatibility". Driven through the REAL builder, not asserted about its source text."""
    result = _clean_round_result()
    leftover = [key for key in _DELETED_NAMES if key in result]
    assert leftover == [], (
        f"the routed mapping still carries {leftover} — the wire shape is the half that "
        f"`promote.py` and the train side actually read. Keys: {sorted(result)}"
    )
    for key in _REQUIRED_NAMES:
        assert key in result, (
            f"the routed mapping must carry {key!r} UNCONDITIONALLY (clean rounds included: "
            f"an ABSENT reason must be an error at the consumer, never 'assume clean'). "
            f"Keys: {sorted(result)}"
        )
    assert result["eval_broken_reason"] is None, (
        "a clean round's reason is None — that IS the clean state, and there is no second "
        f"boolean saying so; got {result['eval_broken_reason']!r}"
    )
    assert result["promoted"] is False, (
        "premise: this fixture built a non-promoted clean round (no gate result), so the "
        "reason assertion above is not riding on a promotion path"
    )


def _modules_reading(name: str) -> list[str]:
    """Every module under `src/mantis` that READS `name` off a mapping. AST, not grep: a
    subscript and a `.get(...)` are READS while a dict-literal key is the PRODUCER and must not
    count itself — the distinction a text grep cannot make."""
    offenders: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            reads = False
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) \
                    and node.slice.value == name:
                reads = True
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr in ("get", "pop") \
                    and any(isinstance(arg, ast.Constant) and arg.value == name
                            for arg in node.args):
                reads = True
            if reads:
                offenders.append(str(path.relative_to(_SRC.parent)))
                break
    return offenders


def test_no_module_under_src_branches_on_the_eval_broken_detail() -> None:
    """The detail is PROSE — `repr(exc)`, a path, a message — and prose is not a decision
    surface: branching on its text makes the detail a second authority beside the reason. The
    premise assertion comes first, since a census over a name nothing produces is a free green."""
    assert "eval_broken_detail" in _clean_round_result(), (
        "PREMISE — the key must be PRODUCED before a census over it means anything; without "
        "this the assertion below is a green over an empty search"
    )
    offenders = _modules_reading("eval_broken_detail")
    assert offenders == [], (
        f"{offenders} branch on `eval_broken_detail`. The detail is prose beside the ONE "
        "authority (the typed reason); reading it to decide anything re-creates the second "
        "decision surface R152 deleted. Read `eval_broken_reason` instead"
    )
