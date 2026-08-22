"""`repo_design.md` §2's `diagnostics` row, made enforceable rather than prose (R9 amendment).

THE DEFECT THIS CLOSES IS THAT NOTHING WAS CHECKING. §2 listed `diagnostics` as a DAG LEAF, and
the row was already false at `c92bafc`: `fusion_calibrate.py` reaches `config`, `encoding`,
`model`, `selfplay` and `_engine`; `eval_child_memory.py` reaches `eval.child_memory`. CI gate 9
(`tools/check_import_dag.py`) checks CYCLES ONLY — it says so in its own docstring — so the drift
was invisible for as long as it existed. WORKER-SWEEP (R309(g)) deepens the same edge, which is
what forced the row to be corrected instead of quietly widened again.

WHAT THE CORRECTED ROW CLAIMS, and it is a DIRECTION rather than a leaf: `diagnostics` is a SINK.
It may import anything below `run`, because a readout that does not measure the same program the
run executes measures a different program — `fusion_calibrate.py`'s own docstring makes that
argument for the calibration path and it is the same argument here. **Nothing may import it.**
That single direction is what keeps every cycle through the layer unrepresentable, and it is what
this census checks.

A CENSUS AND NOT A COMMENT (R296(f)): the amendment says which check enforces it, because an
amended row with no mechanism is the same invisible drift with a newer date on it.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_PKG = Path(__file__).resolve().parents[1] / "src" / "mantis"


def _imports_at_any_scope(path: Path) -> set[str]:
    """Every dotted import target in `path`, at EVERY scope — a lazy import is still an edge for
    this question, since the ban is about direction and not about load order."""
    targets: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            targets.add(node.module)
            targets.update(f"{node.module}.{alias.name}" for alias in node.names)
    return targets


def test_nothing_outside_diagnostics_imports_mantis_diagnostics() -> None:
    offenders: list[str] = []
    for path in sorted(_PKG.rglob("*.py")):
        if path.relative_to(_PKG).parts[0] == "diagnostics":
            continue
        if any(t == "mantis.diagnostics" or t.startswith("mantis.diagnostics.")
               for t in _imports_at_any_scope(path)):
            offenders.append(str(path.relative_to(_PKG.parent.parent)))
    assert not offenders, (
        "`repo_design.md` §2: `diagnostics` is a SINK — it may import anything below `run` and "
        f"NOTHING may import it. These do: {offenders}. An importer turns the readout layer "
        "into a dependency of the thing it reads, which is how a measurement starts changing "
        "the program it measures."
    )


def test_the_census_can_see_an_importer_it_is_meant_to_reject(tmp_path: Path) -> None:
    """LAW-07. A direction ban never shown to reject anything is indistinguishable from one that
    accepts everything — and the FIRST cut of this row called the HELPER
    (`_imports_at_any_scope`) on a planted file rather than the census, so BLINDING the census
    (replacing its file filter with `continue`) kept both rows green. A self-test that cannot
    tell a working census from a blind one does not establish the property it claims.

    The pattern is `test_the_pool_allowlist_BITES_on_a_third_construction_site`'s: rebind the
    module's own root global to a temp tree and require the offender to be NAMED."""
    planted = tmp_path / "mantis" / "somewhere"
    planted.mkdir(parents=True)
    (planted / "new_consumer.py").write_text(
        "def go():\n"
        "    from mantis.diagnostics.worker_sweep import select_knee\n"
        "    return select_knee\n",
        encoding="utf-8",
    )
    (planted / "innocent.py").write_text("import json\n", encoding="utf-8")
    module = sys.modules[__name__]
    saved = module._PKG
    module._PKG = tmp_path / "mantis"
    try:
        with pytest.raises(AssertionError) as caught:
            test_nothing_outside_diagnostics_imports_mantis_diagnostics()
    finally:
        module._PKG = saved
    assert "new_consumer.py" in str(caught.value), (
        f"the census fired without naming the offender: {caught.value}"
    )
    assert "innocent.py" not in str(caught.value), "the census named a file that is clean"


def test_the_census_actually_walks_the_package_it_is_pointed_at(tmp_path: Path) -> None:
    """The other half of I-6: a BLINDED census (one that inspects nothing) passes the live row
    trivially. This row requires the walk to see files."""
    empty = tmp_path / "mantis"
    empty.mkdir()
    module = sys.modules[__name__]
    saved = module._PKG
    module._PKG = empty
    try:
        test_nothing_outside_diagnostics_imports_mantis_diagnostics()  # vacuously clean
    finally:
        module._PKG = saved
    assert sum(1 for _ in _PKG.rglob("*.py")) > 50, (
        "the real package root resolves to almost nothing — the census would be blind"
    )
