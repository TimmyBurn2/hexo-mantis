"""Audit CLI e2e — the command→exit-code contract (0 clean / 1 warn / 2 error).

Exercised over the TORCH-FREE sections (§1 registered, §3 corpora, §4 variants,
§5 hardcodes, §6 cross-table). The §2 (checkpoints) leg needs torch to build/read
a `.pt`; it SKIPs-with-reason when torch is absent (tracked-not-silent) — the
exit-code contract itself is proven torch-free.
"""
from __future__ import annotations

import importlib.util

import pytest

from mantis.encoding.audit import (
    AuditReport,
    audit,
    main,
)

_HAVE_TORCH = importlib.util.find_spec("torch") is not None


# ── AuditReport.exit_code contract (unit) ───────────────────────────────────


def test_exit_code_clean_is_zero() -> None:
    r = AuditReport()
    r.add_finding("info", "§1", "all good")
    assert r.exit_code() == 0


def test_exit_code_warn_is_one() -> None:
    r = AuditReport()
    r.add_finding("info", "§1", "x")
    r.add_finding("warn", "§3", "missing")
    assert r.exit_code() == 1


def test_exit_code_error_is_two_and_takes_max() -> None:
    r = AuditReport()
    r.add_finding("warn", "§3", "w")
    r.add_finding("error", "§6", "e")
    r.add_finding("info", "§1", "i")
    assert r.exit_code() == 2


# ── CLI exit-code contract e2e (torch-free) ─────────────────────────────────


def _empty_dirs(tmp_path):
    ck = tmp_path / "checkpoints"
    co = tmp_path / "data"
    va = tmp_path / "variants"
    root = tmp_path / "root"
    for d in (ck, co, va, root):
        d.mkdir()
    return ck, co, va, root


def test_cli_clean_returns_zero(tmp_path) -> None:
    ck, co, va, root = _empty_dirs(tmp_path)
    rc = main([
        "audit",
        "--checkpoints-dir", str(ck),
        "--corpora-dir", str(co),
        "--variants-dir", str(va),
        "--repo-root", str(root),
    ])
    assert rc == 0


def test_cli_missing_dir_returns_warn(tmp_path) -> None:
    ck, co, va, root = _empty_dirs(tmp_path)
    missing = tmp_path / "nope"  # not created → §2 warns
    rc = main([
        "audit",
        "--checkpoints-dir", str(missing),
        "--corpora-dir", str(co),
        "--variants-dir", str(va),
        "--repo-root", str(root),
    ])
    assert rc == 1


def test_cli_error_returns_two_via_bad_variant(tmp_path) -> None:
    ck, co, va, root = _empty_dirs(tmp_path)
    # A variant yaml whose top-level is not a mapping → §4 error.
    (va / "broken.yaml").write_text("- a\n- b\n")
    rc = main([
        "audit",
        "--checkpoints-dir", str(ck),
        "--corpora-dir", str(co),
        "--variants-dir", str(va),
        "--repo-root", str(root),
    ])
    assert rc == 2


# ── §5 hardcodes — --hardcodes-only runs the scan ───────────────────────────


def test_hardcodes_only_clean_tree_is_zero(tmp_path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "clean.py").write_text("x = 'no bare geometry here'\n")
    rc = main(["audit", "--hardcodes-only", "--repo-root", str(tmp_path)])
    assert rc == 0


def test_hardcodes_only_flags_a_bare_literal(tmp_path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "hot.py").write_text("board_dim = 19\n")  # bare geometry literal
    # Non-strict → warn.
    assert main(["audit", "--hardcodes-only", "--repo-root", str(tmp_path)]) == 1
    # Strict → error.
    assert main(["audit", "--hardcodes-only", "--strict", "--repo-root", str(tmp_path)]) == 2


# ── §6 cross-table INV logic (reproduces the WP3 Rust reference) ─────────────


def test_cross_table_inv1_mismatch_is_error(tmp_path) -> None:
    from mantis.encoding.audit import CheckpointEntry, CorpusEntry
    from mantis.encoding.audit_sections import _section_cross_table

    report = AuditReport()
    # corpora_dir points at a non-empty/absent path so the empty-dir skip
    # heuristic does not fire — the join runs over the entry lists.
    corpora_dir = tmp_path / "absent_corpora"
    ck = [CheckpointEntry(tmp_path / "m.pt", "v6", "aa", True)]
    co = [CorpusEntry(tmp_path / "c.npz", "v6w25", "aa", True)]
    _section_cross_table(report, ck, co, corpora_dir)
    assert report.exit_code() == 2
    assert any("v6w25" in f.message for f in report.findings if f.severity == "error")


def test_cross_table_inv5_ok_is_info(tmp_path) -> None:
    from mantis.encoding.audit import CheckpointEntry, CorpusEntry
    from mantis.encoding.audit_sections import _section_cross_table

    report = AuditReport()
    corpora_dir = tmp_path / "absent_corpora"
    ck = [CheckpointEntry(tmp_path / "m.pt", "v6", "aa", True)]
    co = [CorpusEntry(tmp_path / "c.npz", "v6", "aa", True)]
    _section_cross_table(report, ck, co, corpora_dir)
    assert report.exit_code() == 0


# ── §1 registered reads the compiled registry ───────────────────────────────


def test_section_registered_lists_all_specs(tmp_path) -> None:
    ck, co, va, root = _empty_dirs(tmp_path)
    report = audit(ck, co, va, repo_root=root)
    sect = report.sections["§1"]
    names = {row[0] for row in sect.rows}
    from mantis.encoding import all_specs

    assert names == {s.name for s in all_specs()}


# ── §2 checkpoints (torch) — skip-with-reason when torch absent ──────────────


@pytest.mark.skipif(not _HAVE_TORCH, reason="torch not installed (model/train are later WPs)")
def test_section_checkpoints_declared_equals_inferred(tmp_path) -> None:
    import torch

    ck, co, va, root = _empty_dirs(tmp_path)
    # A stamped v6-shaped checkpoint: declared==inferred → OK (info).
    state = {
        "trunk.input_conv.weight": torch.zeros(64, 8, 3, 3),
        "policy_fc.weight": torch.zeros(362, 64),
    }
    torch.save({"model_state": state, "metadata": {"encoding_name": "v6"}}, ck / "m.pt")
    report = audit(ck, co, va, repo_root=root)
    # §2 (checkpoints) is the leg under test: the v6 match is reported and is clean (info).
    # The global exit code is dominated by §6's unrelated "no corpora to join against" warn,
    # so this test asserts on the §2 section directly, not on report.exit_code().
    s2 = [f for f in report.findings if f.section == "§2"]
    assert any("declared==inferred (v6)" in f.message for f in s2)
    assert all(f.severity == "info" for f in s2)
