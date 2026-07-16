"""Test the tester: the import-DAG gate must bite on planted cycles (LAW-07 pattern)."""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "check_import_dag.py"
SRC_MANTIS = REPO_ROOT / "src" / "mantis"


def _run(target: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(target)], capture_output=True, text=True, check=False
    )


def _make_pkg(root: Path, files: dict[str, str]) -> Path:
    pkg = root / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    for name, body in files.items():
        (pkg / name).write_text(body)
    return pkg


def test_real_tree_acyclic():
    res = _run(SRC_MANTIS)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "OK: no import cycles" in res.stdout


def test_planted_two_cycle_detected(tmp_path):
    pkg = _make_pkg(tmp_path, {"a.py": "import pkg.b\n", "b.py": "import pkg.a\n"})
    res = _run(pkg)
    assert res.returncode == 1
    cycle_lines = [line for line in res.stdout.splitlines() if line.startswith("CYCLE:")]
    assert cycle_lines, res.stdout
    assert any("pkg.a" in line and "pkg.b" in line for line in cycle_lines)


def test_acyclic_synthetic_passes(tmp_path):
    pkg = _make_pkg(tmp_path, {"a.py": "import pkg.b\n", "b.py": "X = 1\n"})
    res = _run(pkg)
    assert res.returncode == 0, res.stdout + res.stderr


def test_function_local_import_is_not_an_edge(tmp_path):
    pkg = _make_pkg(
        tmp_path, {"a.py": "def f():\n    import pkg.b\n", "b.py": "import pkg.a\n"}
    )
    res = _run(pkg)
    assert res.returncode == 0, res.stdout + res.stderr


def test_syntax_error_exits_2(tmp_path):
    pkg = _make_pkg(tmp_path, {"broken.py": "def broken(:\n"})
    res = _run(pkg)
    assert res.returncode == 2
    assert "broken.py" in res.stderr
