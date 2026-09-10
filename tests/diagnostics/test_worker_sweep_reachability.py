"""P1 — NO TRAINER STEP IS REACHABLE FROM THE WORKER SWEEP, checked structurally.

>300 justify (R8): ONE CLAIM, three arms and their planted breaks. Each arm covers an escape the
others cannot see — an import walk misses a subprocess, a `sys.modules` witness misses a source
edge that never executes, a process census misses an import — so the claim is their CONJUNCTION
and a split would let one arm be weakened without the argument for the other two being made
anywhere. The closure walkers are shared by the live rows and their planted-break rows so the
predicate and the proof it can fire move together.

Phase W is SELF-PLAY ONLY: no trainer step may execute before the mint, and a comment saying so
is not a mechanism. STATIC walks the import closure of `mantis.diagnostics.worker_sweep` at
EVERY scope (`tools/check_import_dag.py` is top-level-only by design; it answers cycles).
BEHAVIOURAL imports the module in a fresh subprocess and reads `sys.modules` — IMPORT TIME ONLY.
PROCESS censuses the AST for a launched Python entry point, which both import arms pass green.

Both import arms are also run against a temp-tree copy carrying a PLANTED BREAK, a train
import, and must red:
a ban never shown to reject anything is indistinguishable from one that accepts everything.
"""
from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src"
_PKG_ROOT = _SRC / "mantis"
_ENTRY = "mantis.diagnostics.worker_sweep"

#: Reaching either makes a trainer step constructible; `mantis.run` imports the orchestrator.
_BANNED_PREFIXES = ("mantis.train",)
_BANNED_EXACT = ("mantis.run",)


def _module_table(pkg_root: Path) -> dict[str, Path]:
    table: dict[str, Path] = {}
    for path in pkg_root.rglob("*.py"):
        parts = list(path.relative_to(pkg_root).parts)
        parts[-1] = parts[-1][: -len(".py")]
        if parts[-1] == "__init__":
            parts.pop()
        table[".".join(["mantis", *parts])] = path
    return table


def _import_targets(mod: str, is_pkg: bool, node: ast.Import | ast.ImportFrom) -> list[str]:
    """Dotted targets of one import statement, relative imports resolved.

    Lifted in SHAPE from `tools/check_import_dag.py` and not imported from it: that tool walks
    top-level statements only, and this walk must not."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if node.level == 0:
        base = node.module or ""
    else:
        parts = mod.split(".")
        if not is_pkg:
            parts = parts[:-1]
        strip = node.level - 1
        if strip >= len(parts):
            return []
        if strip:
            parts = parts[:-strip]
        base = ".".join(parts)
        if node.module:
            base = f"{base}.{node.module}" if base else node.module
    return [f"{base}.{alias.name}" if base else alias.name for alias in node.names]


def _resolve(target: str, table: dict[str, Path]) -> str | None:
    parts = target.split(".")
    while parts:
        candidate = ".".join(parts)
        if candidate in table:
            return candidate
        parts.pop()
    return None


def import_closure(entry: str, pkg_root: Path) -> set[str]:
    """Every `mantis.*` module reachable from `entry` by an import at ANY scope."""
    table = _module_table(pkg_root)
    if entry not in table:
        raise AssertionError(
            f"{entry} does not exist under {pkg_root}. This oracle is the reachability proof "
            "for the worker sweep; with no module to walk it proves nothing, so it refuses "
            "rather than passing vacuously."
        )
    seen: set[str] = set()
    stack = [entry]
    while stack:
        mod = stack.pop()
        if mod in seen:
            continue
        seen.add(mod)
        path = table[mod]
        tree = ast.parse(path.read_text(encoding="utf-8"))
        is_pkg = path.name == "__init__.py"
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for target in _import_targets(mod, is_pkg, node):
                    resolved = _resolve(target, table)
                    if resolved is not None:
                        stack.append(resolved)
    return seen


def banned_members(closure: set[str]) -> list[str]:
    return sorted(
        module for module in closure
        if module in _BANNED_EXACT or module.startswith(_BANNED_PREFIXES)
    )


def test_the_sweep_cannot_reach_the_trainer_by_an_import_at_any_scope() -> None:
    closure = import_closure(_ENTRY, _PKG_ROOT)
    found = banned_members(closure)
    assert not found, (
        f"{_ENTRY} reaches {found} — R309(g) makes Phase W SELF-PLAY ONLY so that no trainer "
        "step executes before the mint and the voided-caps row is not crossed. An import is "
        "reachability; a comment is not."
    )
    # A closure collapsed to the entry module alone would pass the row above proving nothing.
    assert len(closure) > 5, (
        f"the closure of {_ENTRY} is {sorted(closure)} — too small to be the real one; the "
        "walk is not seeing the imports it is supposed to be walking"
    )


def test_importing_the_sweep_does_not_put_the_trainer_into_sys_modules() -> None:
    program = textwrap.dedent(
        f"""
        import importlib, sys
        importlib.import_module({_ENTRY!r})
        bad = sorted(m for m in sys.modules
                     if m in {_BANNED_EXACT!r} or m.startswith({_BANNED_PREFIXES!r}))
        print("BANNED:" + ",".join(bad))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, (
        f"importing {_ENTRY} in a fresh interpreter failed:\n{proc.stderr[-2000:]}"
    )
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("BANNED:")]
    assert line, f"the witness printed no verdict line; stdout was {proc.stdout[-500:]!r}"
    reached = [m for m in line[-1][len("BANNED:"):].split(",") if m]
    assert not reached, (
        f"importing {_ENTRY} pulled {reached} into sys.modules. The static walk can miss an "
        "`importlib` call no AST names; this arm is why."
    )


@pytest.fixture()
def planted_tree(tmp_path: Path) -> Path:
    """A copy of `src/mantis` whose sweep module imports the trainer inside a function."""
    root = tmp_path / "mantis"
    shutil.copytree(_PKG_ROOT, root, ignore=shutil.ignore_patterns("__pycache__"))
    target = root / "diagnostics" / "worker_sweep.py"
    target.write_text(
        target.read_text(encoding="utf-8")
        + "\n\ndef _planted_break():\n"
        + "    from mantis.train.orchestrator import init_trainer\n"
        + "    return init_trainer\n",
        encoding="utf-8",
    )
    return root


def test_the_static_walk_reds_on_a_planted_lazy_trainer_import(planted_tree: Path) -> None:
    """The plant is a FUNCTION-BODY import — the shape a top-level-only walk waves through."""
    found = banned_members(import_closure(_ENTRY, planted_tree))
    assert found, (
        "the static walk did not see a function-scope `from mantis.train...` import planted "
        "in a copy of the module. A ban never shown to reject anything is indistinguishable "
        "from one that accepts everything."
    )


# arm 3 — no process creation of a Python entry point: the driver already imports `subprocess`
# for its `git` provenance, so one `subprocess.run([sys.executable, "-m", "mantis.run", ...])`
# keeps both import arms green and steps a trainer anyway. THE PREDICATE: `sys.executable` may
# not appear; `os.exec*`/`os.spawn*`/`os.system`/`multiprocessing.Process` may not be called;
# every `subprocess.*` argv must be a literal list whose program is on a one-entry allowlist,
# because an argv built at runtime is a program this census cannot read.

_PROCESS_APIS = ("run", "Popen", "call", "check_call", "check_output", "system", "Process",
                 "popen", "fork", "forkpty", "spawn", "spawn_main", "run_module", "run_path",
                 "ProcessPoolExecutor", "startfile", "import_module")
_EXEC_PREFIXES = ("exec", "spawn", "posix_spawn")
#: The ONE program the driver may launch: `git`, for the report's own commit provenance.
_ALLOWED_PROGRAMS = {"git"}
#: Process/import modules: none may be imported BY NAME, and only `subprocess` and `os` at all.
_PROCESS_MODULES = {"subprocess", "multiprocessing", "runpy", "concurrent", "importlib", "pty"}
_MODULE_OWNERS = {"subprocess", "os", "multiprocessing", "mp", "runpy", "importlib",
                  "concurrent", "futures", "pty"}


def process_creation_offenders(path: Path) -> list[str]:
    """Every process-creation site in `path` that this census cannot certify as non-Python."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders: list[str] = []
    for node in ast.walk(tree):
        # (0) `from subprocess import run` leaves an `ast.Name` callee with no owner to key on.
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] \
                in _PROCESS_MODULES:
            offenders.append(
                f"line {node.lineno}: `from {node.module} import ...` — import the module, not "
                "its callables, or this census cannot see the call")
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _PROCESS_MODULES and root != "subprocess":
                    offenders.append(f"line {node.lineno}: imports {alias.name}")
        if isinstance(node, ast.Attribute) and node.attr == "executable" and \
                isinstance(node.value, ast.Name) and node.value.id == "sys":
            offenders.append(f"line {node.lineno}: sys.executable")
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # (1) A bare name is a by-name import or a local shadow; the callee is then unreadable,
        # and an unreadable callee is refused rather than guessed at.
        if isinstance(func, ast.Name) and func.id in ("__import__", *_PROCESS_APIS):
            offenders.append(f"line {node.lineno}: bare-name call to {func.id}")
            continue
        # (2) A dynamic attribute fetch on a process module is a callee spelled at runtime.
        if isinstance(func, ast.Name) and func.id == "getattr" and len(node.args) >= 2 and \
                isinstance(node.args[0], ast.Name) and node.args[0].id in _MODULE_OWNERS:
            attr = node.args[1]
            readable = isinstance(attr, ast.Constant) and isinstance(attr.value, str)
            if not readable:
                offenders.append(
                    f"line {node.lineno}: getattr on {node.args[0].id} with a computed attribute "
                    "name — the callee is spelled at runtime and this census cannot read it")
                continue
            if attr.value in _PROCESS_APIS or attr.value.startswith(_EXEC_PREFIXES):
                offenders.append(f"line {node.lineno}: getattr({node.args[0].id}, {attr.value!r})")
            continue
        if isinstance(func, ast.Attribute):
            owner = (func.value.id if isinstance(func.value, ast.Name)
                     else func.value.attr if isinstance(func.value, ast.Attribute) else "")
            if owner == "os" and func.attr.startswith(_EXEC_PREFIXES):
                offenders.append(f"line {node.lineno}: os.{func.attr}")
                continue
            if func.attr not in _PROCESS_APIS:
                continue
            if owner not in _MODULE_OWNERS:
                continue
            if not node.args:
                offenders.append(f"line {node.lineno}: {owner}.{func.attr} with no literal argv")
                continue
            argv = node.args[0]
            # Only the PROGRAM — `argv[0]` — is decidable, so the head must be a string
            # constant; the tail may be anything (the driver splats `*args` into it).
            if not (isinstance(argv, ast.List) and argv.elts
                    and isinstance(argv.elts[0], ast.Constant)
                    and isinstance(argv.elts[0].value, str)):
                offenders.append(
                    f"line {node.lineno}: {owner}.{func.attr} argv head is not a string "
                    "constant — this census cannot read the program it launches")
                continue
            program = argv.elts[0].value
            if program not in _ALLOWED_PROGRAMS:
                offenders.append(f"line {node.lineno}: {owner}.{func.attr} launches {program!r}")
    return offenders


def test_the_sweep_cannot_launch_a_python_entry_point_as_a_subprocess() -> None:
    offenders = process_creation_offenders(_PKG_ROOT / "diagnostics" / "worker_sweep.py")
    assert not offenders, (
        "the worker sweep may launch nothing but `git`. R309(g) makes Phase W self-play only "
        f"so that NO TRAINER STEP EXECUTES before the mint; offenders: {offenders}"
    )


def test_the_process_census_reds_on_a_planted_mantis_run_subprocess(tmp_path: Path) -> None:
    """LAW-07. The plant is the exact escape the two import arms cannot see."""
    planted = tmp_path / "worker_sweep.py"
    planted.write_text(
        "import subprocess, sys\n"
        "def _planted_break(cfg, out):\n"
        "    return subprocess.run([sys.executable, '-m', 'mantis.run', '--config', cfg])\n",
        encoding="utf-8",
    )
    offenders = process_creation_offenders(planted)
    assert any("sys.executable" in o for o in offenders), offenders
    assert any("launches" in o or "cannot read" in o for o in offenders), offenders
    assert "mantis.run" in planted.read_text(encoding="utf-8"), "the plant must be the real shape"


def test_the_process_census_reds_on_a_runtime_built_argv(tmp_path: Path) -> None:
    """A runtime-built argv is refused, not waved through — rejecting only the literal escape
    teaches the next author to build the argv one line earlier."""
    planted = tmp_path / "worker_sweep.py"
    planted.write_text(
        "import subprocess\n"
        "def _planted_break(cmd):\n"
        "    return subprocess.run(cmd)\n",
        encoding="utf-8",
    )
    assert process_creation_offenders(planted), "a runtime-built argv was certified as safe"


def test_the_process_census_accepts_the_git_provenance_call_it_is_shaped_around(
    tmp_path: Path,
) -> None:
    """The `git` shape the ban is built around; a red here means census and tool have parted."""
    planted = tmp_path / "worker_sweep.py"
    planted.write_text(
        "import subprocess\n"
        "def _git(*args):\n"
        "    return subprocess.run(['git', *args], capture_output=True, check=True)\n",
        encoding="utf-8",
    )
    assert process_creation_offenders(planted) == []


def test_the_runtime_witness_reds_on_a_planted_trainer_import(planted_tree: Path) -> None:
    """Arm 2's own planted break. The temp package is reached through `PYTHONPATH` on the
    CHILD's environment — never a `sys.path` mutation, which R5 bans repo-wide."""
    program = textwrap.dedent(
        f"""
        import importlib, sys
        mod = importlib.import_module({_ENTRY!r})
        mod._planted_break()
        bad = sorted(m for m in sys.modules
                     if m in {_BANNED_EXACT!r} or m.startswith({_BANNED_PREFIXES!r}))
        print("BANNED:" + ",".join(bad))
        """
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(planted_tree.parent) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True,
                          check=False, env=env)
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("BANNED:")]
    assert line, f"the planted witness produced no verdict line: {proc.stderr[-1500:]}"
    reached = [m for m in line[-1][len("BANNED:"):].split(",") if m]
    assert reached, (
        "the runtime witness did not see a trainer import that actually executed — arm 2 is "
        "then indistinguishable from a check that always passes"
    )


@pytest.mark.parametrize(
    ("label", "source"),
    [
        ("importlib by string",
         "import importlib\ndef go():\n"
         "    return importlib.import_module('mantis.train.orchestrator')\n"),
        ("runpy", "import runpy\ndef go():\n"
                  "    return runpy.run_module('mantis.run', run_name='__main__')\n"),
        ("subprocess by name",
         "from subprocess import run\nimport shutil\ndef go(c):\n"
         "    return run([shutil.which('python3'), '-m', 'mantis.run', '--config', c])\n"),
        ("os.popen", "import os\ndef go(c):\n"
                     "    return os.popen('python3 -m mantis.run --config ' + c)\n"),
        ("multiprocessing by name",
         "from multiprocessing import Process\ndef go(f):\n    return Process(target=f)\n"),
        ("fork + computed exec",
         "import os\ndef go():\n    if os.fork() == 0:\n"
         "        getattr(os, 'exec' + 'vp')('python3', ['python3', '-m', 'mantis.run'])\n"),
        ("dunder import",
         "def go():\n"
         "    return __import__('mantis.train.orchestrator', fromlist=['init_trainer'])\n"),
        ("process pool",
         "import concurrent.futures\ndef go(f):\n"
         "    return concurrent.futures.ProcessPoolExecutor().submit(f)\n"),
    ],
)
def test_the_process_census_reds_on_every_escape_the_red_team_walked_past(
    tmp_path: Path, label: str, source: str,
) -> None:
    """Every escape the first cut of this census walked past must red.

    That cut keyed on the CALL SHAPE — an `ast.Attribute` callee owned by one of four modules —
    and all eight passed, because all three planted breaks used that one shape."""
    planted = tmp_path / "worker_sweep.py"
    planted.write_text(source, encoding="utf-8")
    assert process_creation_offenders(planted), f"escape {label!r} was certified as safe"
