"""The copy-detector scans the values the repo is actually built from.

Four defects it closes: a frozen dense-era target list that never scanned the graph-era
values; a name-keyed exemption that waved through any copy reusing the canonical spelling,
which is the inverse of a copy detector; a fixed world-shared `/tmp` dump path written under
a swallowed `OSError`; and a §4 warning about a directory that is expected to be absent.

The target set is now DERIVED from `all_specs()`, so a registry row minted tomorrow is
scanned the day it lands.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "tools"))

import hardcode_scan as H  # noqa: E402


def test_the_target_set_is_derived_from_the_live_registry() -> None:
    """Prove the target set is derived from the live registry: every geometry value >= 3 is a target."""
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


@pytest.mark.parametrize("value", [6, 11, 362, 3])
def test_the_graph_era_values_are_scanned_at_all(value: int) -> None:
    """Prove the graph-era values are scanned: 6 is graph_radius/win_length, 11 node_feat_dim,
    362 the policy width, 3 win_axes."""
    assert str(value) in H._HARDCODE_TARGETS, f"{value} is not in the scanned set"


def test_a_planted_copy_in_a_NON_OWNING_file_is_caught(tmp_path: Path) -> None:
    """Prove a canonical-named copy in a non-owning file is caught, not exempted for its name."""
    planted = tmp_path / "not_the_owner.rs"
    planted.write_text("pub const NODE_FEAT_DIM: usize = 11;\n", encoding="utf-8")
    hits = H._scan_file(planted)
    assert hits, (
        "a canonical-named copy in a non-owning file was exempted — the name-keyed exemption "
        "is back, and it is the inverse of a copy detector (AUDIT-1 F-43)"
    )


def test_the_OWNER_of_a_canonical_constant_is_still_exempt(tmp_path: Path) -> None:
    """Control: the file that owns a canonical constant may define it without being flagged."""
    owner = tmp_path / "lib.rs"
    owner.write_text("pub const NODE_FEAT_DIM: usize = 11;\n", encoding="utf-8")
    assert not H._scan_file(owner), (
        "the owning file's own definition is flagged — every canonical constant would now be "
        "a permanent finding, which is how a scanner stops being read"
    )


def test_a_planted_copy_in_a_TEST_is_caught(tmp_path: Path) -> None:
    """Prove a planted copy in a test is caught; `tests/` used to be skipped wholesale."""
    planted = tmp_path / "some_module.py"
    planted.write_text("edge_dim = 5\n", encoding="utf-8")
    assert H._scan_file(planted), "a bare registry-owned value is not detected at all"


def test_the_dump_path_is_a_parameter_and_not_a_shared_tmp_name() -> None:
    """Prove the dump path is a parameter with no fixed `/tmp` name and no swallowed OSError."""
    import inspect

    assert H._DEFAULT_HITS_DUMP is None, (
        "the scanner has a default dump path again — a fixed name is world-shared on a "
        "multi-user host"
    )
    assert "hits_dump" in inspect.signature(H._section_hardcode).parameters
    # Structure, not text: a comment recording the removed handler contains the handler, so
    # a substring search cannot tell the record from the thing.
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
    """Prove §4 no longer warns about `configs/variants/`, a directory expected to be absent."""
    from mantis.encoding.audit import AuditReport
    from mantis.encoding.audit_sections import _section_variants

    report = AuditReport()
    _section_variants(report, tmp_path / "definitely_absent")
    warns = [f for f in report.findings if f.severity == "warn" and f.section == "§4"]
    assert not warns, (
        f"§4 still warns about a directory that is expected to be absent: {warns}"
    )
