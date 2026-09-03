"""A manifest producer must be CALLABLE from production, not merely importable.

AUDIT-1 F-33's PIN, landed on its own. `verify_manifest` resolves a `kind: symbol` row by
importing the module and resolving the dotted attribute — an EXISTENCE check. It cannot tell a
live producer from one whose every production call site has been deleted, and that is not
hypothetical: `buffer_persist.try_save_buffer` is the only incrementer of
`buffer_save_errors_total`, which `coordinator/step.py` publishes in the `monitor_gates`
payload, while BOTH its call sites were removed by R178(a)/R116. The counter's own mutation
self-test calls the function DIRECTLY, so it stays green for a field production cannot move.

**F-33's DELETION half is BANKED and that state stands** — `docs/design/repo_design.md`'s v5→v6
amendment item 3 rules the helper survives on R178(c)'s ground ("buffer persistence returns, if
at all, as ONE design under CARD-RESUME; nobody builds any piece of it separately") and DISCLOSES
the consequence in as many words: the gauge "is now visibly, rather than invisibly, pinned at
zero until CARD-RESUME lands." Overturning a recorded contract decision is the architect's call,
not a repair leg's. So the pin lands and the deletion does not — and the pin is what stops the
NEXT one of these being invisible. It does not fire on `buffer_save_errors_total`, which carries
no manifest row.

WHAT THIS ADDS: for every `kind: symbol` row, the named function must have at least one caller
outside `tests/`. A row whose producer is only ever called by its own test is the shape above.

WHAT IT DELIBERATELY DOES NOT DO: it does not check that the caller is REACHED at run time —
that is a whole-program question no static census answers, and claiming it would be the wider
false certainty this repo keeps closing. It checks the one thing that is checkable and was false.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from mantis.monitor.manifest import DEFAULT_MANIFEST_PATH, load_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Producer symbols whose only non-test caller is legitimately absent, with grounds. Asserted
#: for EQUALITY, so a row that stops being true reds as loudly as a new offender.
DECLARED_CALLERLESS: dict[str, str] = {}


def _leaf_name(symbol: str) -> str:
    """The attribute a dotted producer symbol ends in — the name a caller would write."""
    return symbol.rsplit(".", 1)[-1]


def _called_names(root: Path) -> set[str]:
    """Every name REACHED as a callable anywhere under `root`, by three routes.

    THREE, and each was found by this census firing on a live producer before it was widened —
    which is the same shape as the audit's own name-scoped census missing a fifth walker:

      1. CALL POSITION — `f(...)`, `obj.f(...)`.
      2. AN IMPORT ALIAS — `from m import batch_fill_pct as _batch_fill_pct` then
         `_batch_fill_pct(self)`. The manifest names the ORIGINAL; the call site writes the
         alias, and a census that reads only call names calls a live producer dead.
      3. A `getattr` STRING — `getattr(coord, "on_eval_round_complete", None)`. The seam that
         routes eval results to the sealbot-WR consumer is reached exactly this way, so a
         census blind to it would have retired the gate's only feed path.
    """
    out: set[str] = set()
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
            continue
        alias_of: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for a in node.names:
                    if a.asname:
                        alias_of[a.asname] = a.name
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if isinstance(fn, ast.Name):
                out.add(fn.id)
                out.add(alias_of.get(fn.id, fn.id))
            elif isinstance(fn, ast.Attribute):
                out.add(fn.attr)
            if isinstance(fn, ast.Name) and fn.id == "getattr" and node.args[1:]:
                target = node.args[1]
                if isinstance(target, ast.Constant) and isinstance(target.value, str):
                    out.add(target.value)
    return out


def _symbol_rows() -> list[tuple[str, str]]:
    """`(row id, dotted symbol)` for every `kind: symbol` producer in the shipped manifest."""
    doc = load_manifest(DEFAULT_MANIFEST_PATH)
    rows: list[tuple[str, str]] = []
    for gate in doc.get("gates") or []:
        for key in ("producer", "also"):
            producer = gate.get(key)
            if isinstance(producer, dict) and producer.get("kind") == "symbol":
                symbol = producer.get("symbol")
                if isinstance(symbol, str):
                    rows.append((str(gate.get("id")), symbol))
    return rows


def test_every_symbol_producer_has_a_caller_outside_tests():
    rows = _symbol_rows()
    assert len(rows) > 3, (
        f"only {len(rows)} symbol rows parsed out of the shipped manifest — the parse broke and "
        "this census would pass vacuously"
    )
    production_calls = _called_names(REPO_ROOT / "src") | _called_names(REPO_ROOT / "tools")
    assert len(production_calls) > 100, "the call census returned almost nothing; it is broken"

    orphans = {
        f"{row_id}: {symbol}"
        for row_id, symbol in rows
        # An ATTRIBUTE producer (`self.x`) is a value, not a callable — the leaf is the last
        # segment either way, and a value that is never "called" is not evidence of anything.
        if _leaf_name(symbol) not in production_calls
        and _leaf_name(symbol) in _called_names(REPO_ROOT / "tests")
    }
    orphans -= set(DECLARED_CALLERLESS)
    assert orphans == set(), (
        f"manifest producer(s) called ONLY from tests: {sorted(orphans)}. A monitor input whose "
        "producer cannot run in production is LAW-07's phantom-input class in the event stream: "
        "the row resolves, the mutation self-test calls the producer directly, and the field is "
        "published for a value nothing can move (AUDIT-1 F-33). Re-wire the caller, retire the "
        "row, or declare it in DECLARED_CALLERLESS with grounds."
    )
    stale = set(DECLARED_CALLERLESS) - {f"{i}: {s}" for i, s in rows}
    assert stale == set(), f"DECLARED_CALLERLESS names rows the manifest no longer has: {stale}"


def test_the_census_FIRES_on_a_producer_whose_only_caller_is_a_test(tmp_path: Path):
    """LAW-07 positive control: build the exact shape F-33 found and prove it is caught."""
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "prod.py").write_text(
        "def a_dead_producer():\n    return 1\n", encoding="utf-8")
    (tmp_path / "tests" / "test_it.py").write_text(
        "from prod import a_dead_producer\n\n\ndef test_x():\n    a_dead_producer()\n",
        encoding="utf-8")
    prod = _called_names(tmp_path / "src")
    tests = _called_names(tmp_path / "tests")
    assert "a_dead_producer" not in prod
    assert "a_dead_producer" in tests, "the control's own fixture does not have the shape"


def test_the_census_does_NOT_fire_on_a_producer_with_a_real_caller(tmp_path: Path):
    """The negative control, driven on all THREE reach routes.

    A census that fires on a live producer trains readers to skip it — and this one DID fire on
    three live producers before it was widened, once per route. Each arm below is the exact
    shape that made it fire.
    """
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "direct.py").write_text(
        "def a_live_producer():\n    return 1\n\n\ndef caller():\n    return a_live_producer()\n",
        encoding="utf-8")
    (tmp_path / "src" / "aliased.py").write_text(
        "from direct import a_live_producer as _renamed\n\n\n"
        "def caller():\n    return _renamed()\n",
        encoding="utf-8")
    (tmp_path / "src" / "dynamic.py").write_text(
        'def route(obj):\n    return getattr(obj, "a_handler_name", None)\n',
        encoding="utf-8")
    reached = _called_names(tmp_path / "src")
    assert "a_live_producer" in reached, "a direct call is not seen"
    assert "_renamed" in reached and "a_live_producer" in reached, (
        "an import ALIAS is not resolved to the name the manifest uses — this is what made the "
        "census call `pool_hooks.batch_fill_pct` dead while `pool.py` calls it as "
        "`_batch_fill_pct`"
    )
    assert "a_handler_name" in reached, (
        "a `getattr` STRING is not seen — this is what made the census call "
        "`on_eval_round_complete` dead while `drain.py` routes the sealbot gate's only feed "
        "through exactly that call"
    )


def test_the_call_census_refuses_to_report_clean_on_an_empty_tree(tmp_path: Path):
    """Vacuity control: an empty scan yields an empty set, which satisfies any subset test for
    free. The real assertion carries a floor on the call count, and this records why."""
    assert _called_names(tmp_path) == set()
    with pytest.raises(AssertionError):
        assert len(_called_names(tmp_path)) > 100
