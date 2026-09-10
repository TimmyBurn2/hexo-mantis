"""A manifest producer must be CALLABLE from production, not merely importable.

`verify_manifest` resolves a `kind: symbol` row by importing the module and resolving the
dotted attribute — an existence check that cannot tell a live producer from one whose every
production call site has been deleted. This census adds: every `kind: symbol` row's function
has at least one caller outside `tests/`.

It deliberately does NOT check that the caller is REACHED at run time — a whole-program
question no static census answers.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from mantis.monitor.manifest import DEFAULT_MANIFEST_PATH, load_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Producer symbols whose only non-test caller is legitimately absent, with grounds. Asserted
#: for equality, so a stale row reds as loudly as a new offender.
DECLARED_CALLERLESS: dict[str, str] = {}


def _leaf_name(symbol: str) -> str:
    """Return the attribute a dotted producer symbol ends in: the name a caller would write."""
    return symbol.rsplit(".", 1)[-1]


def _called_names(root: Path) -> set[str]:
    """Return every name reached as a callable under `root`, by three routes.

      1. Call position — `f(...)`, `obj.f(...)`.
      2. An import alias — the manifest names the original, the call site writes the alias.
      3. A `getattr` string — `getattr(coord, "on_eval_round_complete", None)` is how the
         sealbot-WR consumer's only feed path is reached.
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
    """Return `(row id, dotted symbol)` for every `kind: symbol` producer in the manifest."""
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
        # An attribute producer (`self.x`) is a value, not a callable, so never "called".
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
    """Positive control: a producer whose only caller is a test is caught."""
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
    """Negative control, driven on all three reach routes: each once made the census fire on a
    live producer."""
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
    free, so the real assertion carries a floor on the call count."""
    assert _called_names(tmp_path) == set()
    with pytest.raises(AssertionError):
        assert len(_called_names(tmp_path)) > 100
