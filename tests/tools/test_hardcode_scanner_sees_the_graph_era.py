"""AUDIT-1 F-43 — the copy-detector scans the values the repo is actually built from.

THE DEFECT, four ways:

1. **A frozen dense-era target list.** `\\b(19|25|361|5|8)\\b`. The graph-era values were never
   scanned at ALL — `6` (graph_radius / win_length / n_chain_planes), `11` (node_feat_dim),
   `362` and `626` (policy_logit_count), `3` (win_axes), `16`/`17`/`18` (plane indices,
   n_source_planes). Meanwhile `8` had quietly acquired a SECOND meaning (`graph_radius` on
   `gnn_axis_r8`) that the list still read as a plane count.
2. **A name-keyed exemption that inverted the tool.** `_CANONICAL_DEFINE_RE` exempted ANY line
   defining a name in the canonical set, ANYWHERE — so a COPY that reused the canonical
   spelling was exempt because it reused it. That is the inverse of a copy detector, and F-42
   found the shape six times over.
3. **A fixed world-shared dump path** (`/tmp/encoding_audit_hardcode_hits.txt`) written under
   `except OSError: pass` — two users on one host race for it, and a failed write is silent.
4. **A warning that always fires**: §4 warned `configs/variants` was missing on every run. The
   directory does not exist and is not supposed to (R1 retired hand-varied configs). An
   assertion that always fires trains its readers to wave assertions through — R186's exact
   corrosion argument, which this repo has already paid for once.

The target set is now DERIVED from `all_specs()`, so a registry row minted tomorrow is scanned
the day it lands, and nobody has to remember to widen a literal.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "tools"))

import hardcode_scan as H  # noqa: E402


def test_the_target_set_is_derived_from_the_live_registry() -> None:
    """Not a literal list. Every registry geometry value >= 3 is a target."""
    from mantis.encoding import all_specs

    targets = {int(v) for v in H._HARDCODE_TARGETS}
    for spec in all_specs():
        for field in ("board_size", "policy_logit_count", "n_chain_planes",
                      "node_feat_dim", "edge_feat_dim", "win_length", "graph_radius"):
            value = getattr(spec, field, None)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 3:
                assert value in targets, (
                    f"{spec.name}.{field} = {value} is a registry geometry value the scanner "
                    "does not look for — the dense-era blind spot, back"
                )


@pytest.mark.parametrize("value", [6, 11, 362, 626, 3])
def test_the_graph_era_values_are_scanned_at_all(value: int) -> None:
    """The four the frozen list could not see, named. `6` is graph_radius AND win_length AND
    n_chain_planes; `11` is node_feat_dim; `362`/`626` are policy widths; `3` is win_axes."""
    assert str(value) in H._HARDCODE_TARGETS, f"{value} is not in the scanned set"


def test_a_planted_copy_in_a_NON_OWNING_file_is_caught(tmp_path: Path) -> None:
    """The audit's own pin. `const NODE_FEAT_DIM: usize = 11;` in a crate that does not own it
    is a COPY — and the old name-keyed exemption waved it through precisely BECAUSE it used
    the canonical name."""
    planted = tmp_path / "not_the_owner.rs"
    planted.write_text("pub const NODE_FEAT_DIM: usize = 11;\n", encoding="utf-8")
    hits = H._scan_file(planted)
    assert hits, (
        "a canonical-named copy in a non-owning file was exempted — the name-keyed exemption "
        "is back, and it is the inverse of a copy detector (AUDIT-1 F-43)"
    )


def test_the_OWNER_of_a_canonical_constant_is_still_exempt(tmp_path: Path) -> None:
    """The control, and it is what stops the row above from being a gate that fires on correct
    code: the file that OWNS the constant must define it without being flagged."""
    owner = tmp_path / "lib.rs"
    owner.write_text("pub const NODE_FEAT_DIM: usize = 11;\n", encoding="utf-8")
    assert not H._scan_file(owner), (
        "the owning file's own definition is flagged — every canonical constant would now be "
        "a permanent finding, which is how a scanner stops being read"
    )


def test_a_planted_copy_in_a_TEST_is_caught(tmp_path: Path) -> None:
    """The audit's second plant: `edge_dim = 5` in a test. `tests/` was skipped WHOLESALE, so
    a fixture could pin a stale geometry and nothing would say."""
    planted = tmp_path / "some_module.py"
    planted.write_text("edge_dim = 5\n", encoding="utf-8")
    assert H._scan_file(planted), "a bare registry-owned value is not detected at all"


def test_the_dump_path_is_a_parameter_and_not_a_shared_tmp_name() -> None:
    """No fixed `/tmp` filename, and no `except OSError: pass` around the write."""
    import inspect

    assert H._DEFAULT_HITS_DUMP is None, (
        "the scanner has a default dump path again — a fixed name is world-shared on a "
        "multi-user host"
    )
    assert "hits_dump" in inspect.signature(H._section_hardcode).parameters
    # STRUCTURE, not text: the comment recording the removed handler contains the removed
    # handler, so a substring search cannot tell the record from the thing (the same trap the
    # F-30 pin hit).
    import ast

    tree = ast.parse(inspect.getsource(H._section_hardcode))
    swallowed = [
        h.lineno for node in ast.walk(tree) if isinstance(node, ast.Try)
        for h in node.handlers
        if isinstance(h.type, ast.Name) and h.type.id == "OSError"
        and len(h.body) == 1 and isinstance(h.body[0], ast.Pass)
    ]
    assert not swallowed, (
        f"the dump write swallows OSError again at line(s) {swallowed} — a dump the operator "
        "asked for and did not get is a fact, not a silence"
    )


def test_the_variants_section_no_longer_warns_about_an_expected_absence(tmp_path: Path) -> None:
    """R186's corrosion argument: an assertion that always fires teaches its readers to wave
    assertions through. `configs/variants/` does not exist and is not supposed to."""
    from mantis.encoding.audit import AuditReport
    from mantis.encoding.audit_sections import _section_variants

    report = AuditReport()
    _section_variants(report, tmp_path / "definitely_absent")
    warns = [f for f in report.findings if f.severity == "warn" and f.section == "§4"]
    assert not warns, (
        f"§4 still warns about a directory that is expected to be absent: {warns}"
    )
