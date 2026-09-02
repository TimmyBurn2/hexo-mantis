"""AUDIT-1 F-26 — five gates that could pass over a scope nobody chose.

Each row here is the same defect in a different gate: the check ran, found nothing, and
printed green — because there was nothing in front of it. Silence and cleanliness were one
observable.

* **gate 17** used `--diff-filter=AM`, so a moved-AND-edited file (`R0xx`) was dropped: `git mv`
  a fixture, append a box path, and the leak gate scanned nothing. `artifact_gate.py` records
  this exact lesson for itself (WP0 RED-TEAM row A) and it never propagated. It also scanned
  ZERO files on an empty diff and printed green.
* **gate 6** silently returned `"HEAD~1"` for an empty, all-zeros or unresolvable `--base` — so
  a first push or force-push, exactly when the base IS all zeros, inspected the last commit
  only, with no line saying so.
* **gate 10** filtered its scope on `is_file()`: a renamed directory or a wrong CWD shrank the
  scan to nothing, rc 0, no output.
* **gate 13** skipped every citation whose root section is unknown — which is how a dotted
  module path escapes the key check, and equally how a citation whose SECTION was renamed
  escapes it.
* **gate 14** read `errorCount` alone. `errorCount: 0` over ZERO analysed files is the most
  convincing green this gate can print, and it means nothing was checked.
"""
from __future__ import annotations

import importlib.util
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


# ── gate 17: renames and empty scopes ─────────────────────────────────────────────────

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
    """THE PIN (gate 17). `--diff-filter=AM` dropped this file entirely."""
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
    """The control: widening the filter must not lose what it already caught."""
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
    """A leak gate fails toward over-scanning — the posture its own unresolvable-base arm
    already took, now on the empty-scope arm too."""
    proc = subprocess.run(
        ["python3", str(GATES / "rule7_gate.py"), "--base", "HEAD"],
        capture_output=True, text=True, cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "degrading WIDE" in proc.stdout, proc.stdout
    assert "tracked tree" in proc.stdout, "the wide scan did not actually run"


# ── gate 6: the base it actually used ─────────────────────────────────────────────────

@pytest.mark.parametrize(
    "candidate",
    ["", "0" * 40, "definitely-not-a-rev"],
    ids=["empty", "all-zeros", "unresolvable"],
)
def test_a_fallback_base_is_NAMED_and_widens_past_one_commit(candidate: str) -> None:
    """THE PIN (gate 6). All three returned a bare `"HEAD~1"` and printed nothing, so a first
    push inspected the last commit only under a line that said the gate had run."""
    base, why = ARTIFACT._resolve_base(candidate)
    assert why != "given"
    assert base != "HEAD~1", (
        f"{candidate!r} narrowed the scan to one commit; origin/dev resolves in this tree"
    )
    assert base in ARTIFACT._WIDE_FALLBACKS


def test_a_resolvable_base_is_used_verbatim() -> None:
    """The control."""
    base, why = ARTIFACT._resolve_base("HEAD")
    assert (base, why) == ("HEAD", "given")


def test_the_gate_PRINTS_the_base_it_chose() -> None:
    proc = subprocess.run(["python3", str(GATES / "artifact_gate.py")],
                          capture_output=True, text=True, cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.startswith("gate 6: base="), proc.stdout


# ── gate 10: the empty scope ──────────────────────────────────────────────────────────

def test_a_shrunken_scope_RAISES_rather_than_reporting_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE PIN (gate 10). `is_file()` filtering meant a wrong CWD scanned nothing at rc 0."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="named scope is incomplete"):
        TRACKED._scope_files()


def test_the_glob_scope_has_a_floor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The named files can exist while a RENAMED `docs/` leaves the globs empty."""
    for name in TRACKED.SCOPE:
        (tmp_path / name).write_text("x\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="glob scope yielded"):
        TRACKED._scope_files()


def test_the_real_scope_clears_the_floor_and_is_reported() -> None:
    proc = subprocess.run(["python3", str(GATES / "check_tracked_refs.py")],
                          capture_output=True, text=True, cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "gate 10: scanning" in proc.stdout, proc.stdout


# ── gate 13: the unknown root ─────────────────────────────────────────────────────────

def test_a_citation_whose_SECTION_is_gone_is_a_stale_citation(tmp_path: Path) -> None:
    """THE PIN (gate 13). `if root not in sections: continue` let a renamed section's keys
    read as prose — which is precisely the drift this gate exists to catch."""
    doc = tmp_path / "doc.md"
    doc.write_text("A row citing `gone_section.some_key` that no longer exists.\n",
                   encoding="utf-8")
    failures = [f for f in CONTRACT.check(doc) if "root section" in f]
    assert any("gone_section" in f for f in failures), failures


def test_a_declared_non_config_root_is_still_prose(tmp_path: Path) -> None:
    """The control, and the reason the roots are DECLARED: shape cannot separate
    `train.gone_away` from `torch.dtype`, so the legitimate namespaces are named."""
    doc = tmp_path / "doc.md"
    doc.write_text("The dtype is a `torch.dtype`, resolved by `mantis.model.amp`.\n",
                   encoding="utf-8")
    # The synthetic doc has none of this gate's other required structure, so only the
    # unknown-root arm is read out — the other failures are about missing headings and a leaf
    # count, none of which this row is about.
    unknown_root = [f for f in CONTRACT.check(doc) if "root section" in f]
    assert unknown_root == [], unknown_root
    assert "torch" in CONTRACT._NON_CONFIG_ROOTS and "mantis" in CONTRACT._NON_CONFIG_ROOTS


# ── gate 14: files analysed ───────────────────────────────────────────────────────────

def test_gate_14_refuses_a_green_over_zero_analysed_files() -> None:
    """THE PIN (gate 14). `errorCount: 0` over nothing is the most convincing green a type
    gate can print. The floor makes it a REFUSAL (rc 2), not a red — nothing was measured."""
    body = (GATES / "lint_gate.sh").read_text(encoding="utf-8")
    assert "filesAnalyzed" in body
    assert "PYRIGHT_MIN_FILES" in body
    assert 'exit "$PYRIGHT_REFUSED_RC"' in body


def test_gate_14_reports_what_it_analysed_on_the_GREEN_path_too() -> None:
    """A check whose scope is visible only on failure is a check nobody can audit for
    vacuity — the shape MF-3's `source_pins_ok: True` literal took."""
    body = (GATES / "lint_gate.sh").read_text(encoding="utf-8")
    assert "pyright analysed ${FILES} file(s)" in body
