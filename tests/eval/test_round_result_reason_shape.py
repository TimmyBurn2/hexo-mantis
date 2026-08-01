"""⊕ WP12-R Phase O / O-11, O-13 (R152/R79) — the round-result mapping carries ONE authority
for "did this round break", and the detail beside it is PROSE nobody branches on.

RED-at-HEAD on its OWN mechanism (⊕): no import anchor is used here deliberately — the two
oracles below must fail because the SHAPE is wrong at HEAD, not because a module is missing.
That makes their red evidence about the subject rather than about an import line.

The R79 defect, measured at HEAD: `build_round_result` takes BOTH `eval_broken: bool` and
`error: str | None`, and writes both into the mapping (`rounds.py:187,204,205`). Two fields
for one fact is two authorities that can disagree, and the disagreement is CONSTRUCTIBLE
today — `build_round_result(..., eval_broken=True, error=None)` is a legal call that
produces a round which is broken and says nothing about why. The `error` value is worse than
redundant: for `_broken_result` it is the reason string VERBATIM (`pipeline.py:497`), so the
same fact is written twice under two names, and for the catch-all it is
`f"round_completion_error: {detail}"` — a reason spelling glued to prose, which is what
forces every reader to parse rather than compare.

R152's answer is one field: `eval_broken_reason: EvalBrokenReason | None`, where `None` IS
the clean state, plus `eval_broken_detail: str | None` carrying detail ONLY.

The oracles, and the defect each is the ONLY witness to:

- O-11 (2 nodes) — the two legacy names are DELETED from the signature and from the emitted
  mapping, not defaulted. A defaulted parameter is a MIGRATED authority, not an absent one
  (`run.py:366-372`, MF-2 Attack B): `eval_broken: bool = False` would keep every old call
  site compiling and keep the contradiction constructible. Sole witness — nothing else in
  the tree censuses this builder's parameter set.
  MUTATION THAT REDS IT (M-O11): re-add either name to the signature or to the mapping.

- O-13 — the census that keeps `eval_broken_detail` PROSE. The moment a module branches on
  its text, the detail becomes a second decision surface and the reason stops being the one
  authority — the exact shape R152 deletes, re-grown one field over. The census is paired
  with a NON-VACUITY premise (the key must actually be produced) because a census over a
  name nothing produces passes for free, which is how a census stops being an instrument.
  MUTATION THAT REDS IT (M-O13): add `if result["eval_broken_detail"].startswith(...)`
  anywhere under `src/`.
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
#: The two PARAMETER names, i.e. the SIGNATURE half's expectation — `DESIGN_O §b.3` verbatim
#: (`build_round_result(*, reason: EvalBrokenReason | None, detail: str | None, …)`), and
#: what `_clean_round_result` below already calls.
#:
#: G-12 (WP12-R Phase O, dispatcher grant, post-IMPL). This tuple did not exist: the
#: signature half at `:82-91` iterated `_REQUIRED_NAMES`, the MAPPING-KEY tuple, and no
#: signature can both declare `eval_broken_reason`/`eval_broken_detail` as required
#: parameters AND accept the `reason=`/`detail=` call three lines below — so the file was
#: unsatisfiable by any honest implementation (`**kwargs`, `*args`-naming and a faked
#: `__signature__` all "satisfy" it only by adding the silent alias R79 forbids). It was
#: invisible at ORACLE-WRITE because the assert at `:77` fails first at HEAD, so the line
#: below never executed. RE-POINT, not a weakening: the strength is unchanged — both
#: parameters PRESENT, both UNDEFAULTED, and both mapping KEYS still asserted present
#: unconditionally at `:107`. Do not "restore" the single tuple; conflating the two
#: surfaces is the defect.
_REQUIRED_PARAMS = ("reason", "detail")


def _clean_round_result() -> dict:
    """A CLEAN round built through the real builder. `reason=None` is the clean state — the
    whole point of the shape, and it needs no enum member, so this oracle stays independent
    of `mantis.eval.errors`."""
    return build_round_result(
        step=1000, round_id="r000001_1000", rungs_config=[], rung_results={},
        gate_result=None, skipped_rungs=[], bt={"ratings": {}, "p_hat": {}},
        schedule_next={}, eval_round_wall_sec=1.5, reason=None, detail=None, random_wr=None,
    )


# ══ O-11 — the two legacy authorities are gone from the SIGNATURE ══════════════════════
def test_build_round_result_declares_no_eval_broken_bool_and_no_error_string() -> None:
    """O-11, signature half. Deleted, never defaulted.

    `eval_broken: bool = False` would be the tempting migration: every existing call site
    keeps compiling and the diff shrinks. It is also exactly the defect — the bool survives
    beside the reason and `(eval_broken=True, error=None)` stays constructible. R79's text
    is "arming is a property of the resolved value", and broken-ness is a property of the
    resolved reason: `reason is None` ⟺ clean, with nothing beside it to disagree.
    """
    params = inspect.signature(build_round_result).parameters
    present = [name for name in _DELETED_NAMES if name in params]
    assert present == [], (
        f"build_round_result still declares {present} — R79: one fact, one field. A "
        "DEFAULTED survivor is a migrated authority, not an absent one (MF-2 Attack B). "
        f"Full parameter list: {list(params)}"
    )
    # G-12: `_REQUIRED_PARAMS`, not `_REQUIRED_NAMES` — the signature and the mapping are two
    # surfaces with two vocabularies, and this half owns the parameter one. Both assertions
    # below are the ones this oracle always made (PRESENT, and UNDEFAULTED); only the
    # spelling they look for is corrected.
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
    """O-11, mapping half. The signature and the payload are two surfaces and either can
    drift alone: a builder could take `reason=` and still write `"eval_broken": ...` into
    the mapping for "compatibility", which is the second authority back with a new spelling.

    Driven through the REAL builder rather than asserted about its source text.
    """
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


# ══ O-13 — the detail stays prose ══════════════════════════════════════════════════════
def _modules_reading(name: str) -> list[str]:
    """Every module under `src/mantis` that READS `name` off a mapping.

    AST, not grep: a subscript (`x["eval_broken_detail"]`) and a `.get("eval_broken_detail")`
    are READS; a dict-literal key is the PRODUCER and must not count itself, which is
    precisely the distinction a text grep cannot make. Reported as paths relative to the
    package so a failure names the offender.
    """
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
    """O-13. The detail is PROSE — `repr(exc)`, a persistence path, a message — and prose is
    not a decision surface. A module that branches on its text has made the detail a second
    authority beside the reason, which is R152's defect one field over, and it does it in
    the one place a reader would never look for a decision.

    The premise assertion is not decoration: a census over a name nothing produces is a
    green over nothing (the F-10 class in miniature), and it is exactly what this oracle
    would degrade into if the key were renamed. It is asserted first, so the census can
    never pass for the wrong reason.
    """
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
