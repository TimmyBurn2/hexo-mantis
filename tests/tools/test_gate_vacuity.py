"""Five gates that could pass over a scope nobody chose.

Every row is the same defect in a different gate: the check ran, found nothing in front of it,
and printed green, so silence and cleanliness were one observable.

* gate 17 used `--diff-filter=AM`, dropping a moved-AND-edited file, and scanned zero files on
  an empty diff;
* gate 6 silently returned `"HEAD~1"` for an empty, all-zeros or unresolvable `--base`, so a
  first push inspected one commit;
* gate 10 filtered its scope on `is_file()`, so a renamed directory or wrong CWD scanned
  nothing at rc 0;
* gate 13 skipped every citation whose root section is unknown, which is how a renamed section
  escapes the key check;
* gate 14 read `errorCount` alone, and `errorCount: 0` over zero analysed files means nothing
  was checked.
"""
from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GATES = REPO_ROOT / "tools" / "ci_gates"


def _load(name: str, rel: str) -> object:
    spec = importlib.util.spec_from_file_location(name, GATES / rel)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RULE7 = _load("_f26_rule7", "rule7_gate.py")
ARTIFACT = _load("_f26_artifact", "artifact_gate.py")
TRACKED = _load("_f26_tracked", "check_tracked_refs.py")
CONTRACT = _load("_f26_contract", "contract_doc_gate.py")


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    run = lambda *a: subprocess.run(["git", "-C", str(root), *a], check=True,  # noqa: E731
                                    capture_output=True, text=True)
    run("init", "-q", "-b", "dev")
    run("config", "user.email", "t@example.invalid")
    run("config", "user.name", "T")
    (root / "seed.txt").write_text("nothing\n", encoding="utf-8")
    run("add", "-A")
    run("commit", "-q", "-m", "seed")
    return root


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True).stdout


def test_a_moved_AND_edited_file_is_in_the_scan_scope(tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    """Prove a moved-and-edited file is in the scan scope; `--diff-filter=AM` dropped it."""
    root = _repo(tmp_path)
    (root / "a.txt").write_text("harmless\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "add a")
    base = _git(root, "rev-parse", "HEAD").strip()
    _git(root, "mv", "a.txt", "b.txt")
    (root / "b.txt").write_text("harmless\nand now edited\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "move and edit")

    monkeypatch.chdir(root)
    monkeypatch.setattr(RULE7, "REPO_ROOT", root)
    scope = RULE7.target_files(base)
    assert "b.txt" in scope, f"the moved-and-edited file is not in the scope: {scope}"
    assert "a.txt" not in scope, "the OLD path is not in the tree and must not be scanned"


def test_a_plain_add_and_a_plain_modify_are_still_in_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prove widening the filter did not lose the plain adds and modifies it already caught."""
    root = _repo(tmp_path)
    base = _git(root, "rev-parse", "HEAD").strip()
    (root / "added.txt").write_text("new\n", encoding="utf-8")
    (root / "seed.txt").write_text("changed\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "add and modify")

    monkeypatch.chdir(root)
    monkeypatch.setattr(RULE7, "REPO_ROOT", root)
    scope = set(RULE7.target_files(base))
    assert {"added.txt", "seed.txt"} <= scope, scope


def test_an_empty_diff_degrades_WIDE_rather_than_printing_green() -> None:
    """Prove an empty diff degrades to a wide scan rather than printing green."""
    proc = subprocess.run(
        ["python3", str(GATES / "rule7_gate.py"), "--base", "HEAD"],
        capture_output=True, text=True, cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "degrading WIDE" in proc.stdout, proc.stdout
    assert "tracked tree" in proc.stdout, "the wide scan did not actually run"


@pytest.mark.parametrize(
    "candidate",
    ["", "0" * 40, "definitely-not-a-rev"],
    ids=["empty", "all-zeros", "unresolvable"],
)
def test_a_fallback_base_is_NAMED_and_widens_past_one_commit(candidate: str) -> None:
    """Prove a fallback base is named and widens past one commit."""
    base, why = ARTIFACT._resolve_base(candidate)
    assert why != "given"
    assert base != "HEAD~1", (
        f"{candidate!r} narrowed the scan to one commit; origin/dev resolves in this tree"
    )
    assert base in ARTIFACT._WIDE_FALLBACKS


def test_a_resolvable_base_is_used_verbatim() -> None:
    """Prove a resolvable base is used verbatim."""
    base, why = ARTIFACT._resolve_base("HEAD")
    assert (base, why) == ("HEAD", "given")


def test_the_gate_PRINTS_the_base_it_chose() -> None:
    proc = subprocess.run(["python3", str(GATES / "artifact_gate.py")],
                          capture_output=True, text=True, cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.startswith("gate 6: base="), proc.stdout


def test_a_shrunken_scope_RAISES_rather_than_reporting_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prove a shrunken scope raises; `is_file()` filtering meant a wrong CWD scanned nothing."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="named scope is incomplete"):
        TRACKED._scope_files()


def test_the_glob_scope_has_a_floor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Prove the glob scope has a floor: the named files can exist while a renamed dir empties it."""
    for name in TRACKED.SCOPE:
        (tmp_path / name).write_text("x\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="yielded 0 file"):
        TRACKED._scope_files()


def test_one_full_glob_dir_cannot_carry_an_empty_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prove the floor is per-directory: one full glob dir must not carry an empty one."""
    for name in TRACKED.SCOPE:
        (tmp_path / name).write_text("x\n", encoding="utf-8")
    full, empty = list(TRACKED.GLOB_SCOPE)
    for i in range(sum(TRACKED.GLOB_SCOPE.values()) + 1):
        (tmp_path / full).mkdir(parents=True, exist_ok=True)
        (tmp_path / full / f"d{i}.md").write_text("x\n", encoding="utf-8")
    (tmp_path / empty).mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match=re.escape(empty)):
        TRACKED._scope_files()


def test_a_dissolved_path_that_comes_back_refuses_its_own_whitelist() -> None:
    """Prove a dissolved path that is tracked again refuses its own whitelist entry."""
    assert TRACKED.DISSOLVED_PATHS, "the whitelist is the mechanism under test"
    revived = next(iter(TRACKED.DISSOLVED_PATHS)) + "back.md"
    with pytest.raises(FileNotFoundError, match="tracked again"):
        TRACKED._check_dissolved({revived})


def test_the_real_scope_clears_the_floor_and_is_reported() -> None:
    proc = subprocess.run(["python3", str(GATES / "check_tracked_refs.py")],
                          capture_output=True, text=True, cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "gate 10: scanning" in proc.stdout, proc.stdout


def test_the_governance_docs_are_actually_in_the_scan_scope() -> None:
    """Prove the governance docs are really in the scan scope, rather than assumed from a glob."""
    scanned = {str(p) for p in TRACKED._scope_files()}
    assert "docs/governance/LAWS.md" in scanned, sorted(scanned)
    assert "docs/governance/CARDS.md" in scanned, sorted(scanned)
    assert "docs/governance/STATE.md" in scanned, sorted(scanned)
    assert not (scanned & set(TRACKED.SCAN_EXEMPT)), "an exempt file reached the scan"


def test_a_citation_whose_SECTION_is_gone_is_a_stale_citation(tmp_path: Path) -> None:
    """Prove a citation whose section is gone reads as stale, not as prose."""
    doc = tmp_path / "doc.md"
    doc.write_text("A row citing `gone_section.some_key` that no longer exists.\n",
                   encoding="utf-8")
    failures = [f for f in CONTRACT.check(doc) if "root section" in f]
    assert any("gone_section" in f for f in failures), failures


def test_a_declared_non_config_root_is_still_prose(tmp_path: Path) -> None:
    """Prove a declared non-config root is still prose: shape cannot separate it from a key path."""
    doc = tmp_path / "doc.md"
    doc.write_text("The dtype is a `torch.dtype`, resolved by `mantis.model.amp`.\n",
                   encoding="utf-8")
    # The synthetic doc has none of the gate's other required structure, so only the
    # unknown-root arm is read out.
    unknown_root = [f for f in CONTRACT.check(doc) if "root section" in f]
    assert unknown_root == [], unknown_root
    assert "torch" in CONTRACT._NON_CONFIG_ROOTS and "mantis" in CONTRACT._NON_CONFIG_ROOTS


def test_gate_14_refuses_a_green_over_zero_analysed_files() -> None:
    """Prove gate 14 refuses (rc 2, not red) a green over zero analysed files."""
    body = (GATES / "lint_gate.sh").read_text(encoding="utf-8")
    assert "filesAnalyzed" in body
    assert "PYRIGHT_MIN_FILES" in body
    assert 'exit "$PYRIGHT_REFUSED_RC"' in body


def test_gate_14_reports_what_it_analysed_on_the_GREEN_path_too() -> None:
    """Prove gate 14 reports what it analysed on the green path too, so its scope is auditable."""
    body = (GATES / "lint_gate.sh").read_text(encoding="utf-8")
    assert "pyright analysed ${FILES} file(s)" in body
