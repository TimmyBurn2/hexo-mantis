"""Mechanical censuses over `src/mantis/monitor/**`.

STATIC/subprocess only (AST + a torch-blocked subprocess import): nothing here imports
`mantis.monitor.*` at module top, so this file collects cleanly. Each checker carries a
mutation self-test proving it BITES. The censuses: monitor imports with torch blocked and
holds zero torch tokens; its `mantis.*` imports stay within {util, encoding, monitor};
train files importing `mantis.monitor` are exactly `_EXPECTED_TRAIN_MONITOR_SITES`; zero
`except ...: pass` in `monitor/**`; zero `draw_target_fraction` in monitor and coordinator
gate code.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_MONITOR = _REPO / "src" / "mantis" / "monitor"
_TRAIN = _REPO / "src" / "mantis" / "train"
_COORD = _TRAIN / "coordinator"

_ALLOWED_MONITOR_MANTIS_IMPORTS = ("mantis.util", "mantis.encoding", "mantis.monitor")
# The pinned train->monitor import-site inventory. Paths relative to src/mantis.
_EXPECTED_TRAIN_MONITOR_SITES = {
    "train/coordinator/step.py",
    "train/subsystems.py",
    # `train/pretrain/cli.py` top-level imports `monitor.logging_setup.configure_logging`
    # because a process entry with no root handler drops every `logger.info` it emits; it
    # replaced `logging.basicConfig`, so it removes a second sink bootstrap, not adds a first.
    "train/pretrain/cli.py",
    "train/lifecycle/heartbeat_watchdog.py",
    # The stall watchdog routes its two saves through `monitor.best_effort`, the same
    # sanctioned optional-effect seam as its sibling `heartbeat_watchdog.py` above.
    "train/lifecycle/watchdog.py",
    # `signals.py` imports `PARENT_DEATH_PPID_ENV` from `monitor.heartbeat`: the supervisor
    # stamps its pid into the child's environment under that name and the run's arming gate
    # reads it, so both sides must key off ONE constant.
    "train/lifecycle/signals.py",
}


def _top_level_imports(tree: ast.Module) -> list[str]:
    targets: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            targets.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            targets.append(node.module)
            targets.extend(f"{node.module}.{a.name}" for a in node.names)
    return targets


def _swallow_sites(root: Path) -> list[str]:
    """`except …: pass` handlers whose body is exactly pass (the J-04 pattern)."""
    sites: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.body \
                    and all(isinstance(s, ast.Pass) for s in node.body):
                sites.append(f"{path.relative_to(root)}:{node.lineno}")
    return sites


def _grep(root: Path, token: str, patterns: tuple[str, ...] = ("*.py",)) -> list[str]:
    hits: list[str] = []
    paths = sorted({p for pattern in patterns for p in root.rglob(pattern)})
    for path in paths:
        for i, line in enumerate(path.read_text().splitlines(), start=1):
            if token in line:
                hits.append(f"{path.relative_to(root)}:{i}: {line.strip()}")
    return hits


def test_monitor_imports_without_torch_subprocess() -> None:
    """A fresh interpreter imports `mantis.monitor` and every submodule under a meta-path
    torch blocker with rc 0 — a liveness babysitter must never need torch to load."""
    script = (
        "import sys, importlib, pkgutil\n"
        "class _B:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'torch' or name.startswith('torch.'):\n"
        "            raise ImportError('torch blocked by the no-torch monitor census')\n"
        "        return None\n"
        "sys.meta_path.insert(0, _B())\n"
        "import mantis.monitor\n"
        "for m in pkgutil.walk_packages(mantis.monitor.__path__, mantis.monitor.__name__ + '.'):\n"
        "    importlib.import_module(m.name)\n"
        "print('OK')\n"
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert proc.returncode == 0, (
        f"mantis.monitor must import with torch blocked; stderr:\n{proc.stderr}"
    )
    assert "OK" in proc.stdout


def test_no_torch_import_token_in_monitor_sources() -> None:
    """Static backstop: no `import torch` / `from torch` token in `monitor/**`, which a lazy
    in-function import would use to evade the subprocess walk."""
    offenders: list[str] = []
    for path in sorted(_MONITOR.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(a.name.split(".")[0] == "torch"
                                                    for a in node.names):
                offenders.append(f"{path.name}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom) and node.module \
                    and node.module.split(".")[0] == "torch":
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == [], f"monitor/** must be torch-free, found: {offenders}"


def test_monitor_mantis_imports_are_within_the_allowed_set() -> None:
    """Every `mantis.*` import in `monitor/**` is under {util, encoding, monitor}, so a hidden
    hard edge into the headless core bites."""
    violations: list[str] = []
    for path in sorted(_MONITOR.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for target in _top_level_imports(tree):
            if target.startswith("mantis.") and not any(
                target == pkg or target.startswith(pkg + ".")
                for pkg in _ALLOWED_MONITOR_MANTIS_IMPORTS
            ):
                violations.append(f"{path.name} -> {target}")
    assert violations == [], f"monitor/** mantis-imports must be within the allowed set: {violations}"


def test_train_to_monitor_import_sites_are_exactly_the_pinned_set() -> None:
    """The ONLY train files that top-level import `mantis.monitor` are the seams pinned below;
    an extra edge, or a missing one, is DAG drift. This census, not
    `tests/train/test_train_import_dag.py`'s FORBIDDEN set, polices that edge."""
    sites: set[str] = set()
    for path in sorted(_TRAIN.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for target in _top_level_imports(tree):
            if target == "mantis.monitor" or target.startswith("mantis.monitor."):
                sites.add(str(path.relative_to(_TRAIN.parent)))
                break
    assert sites == _EXPECTED_TRAIN_MONITOR_SITES, (
        f"train→monitor import sites must be exactly the pinned set "
        f"{sorted(_EXPECTED_TRAIN_MONITOR_SITES)}; got {sorted(sites)}"
    )


def test_no_swallow_sites_in_monitor() -> None:
    """ZERO `except ...: pass` in `monitor/**`, no allowlist: every optional effect goes
    through `best_effort` (counted) or fails loud."""
    sites = _swallow_sites(_MONITOR)
    assert sites == [], f"monitor/** must have zero except-pass swallow sites, found: {sites}"


def test_swallow_census_bites_planted_swallow(tmp_path: Path) -> None:
    """The detector reports exactly 1 site for a planted swallow, so the census is not
    vacuous."""
    (tmp_path / "planted.py").write_text(
        "def f():\n    try:\n        g()\n    except Exception:\n        pass\n"
    )
    assert len(_swallow_sites(tmp_path)) == 1, "census must bite a planted except-pass"


def test_draw_target_fraction_absent_from_monitor_and_coordinator_gates() -> None:
    """Zero `draw_target_fraction` references in `monitor/**` or `train/coordinator/**`: the
    draw-rate gate keys on the LIVE `pooled_draw_rate`, never the NaN phantom.

    The scan covers `*.yaml` too, because `producer_manifest.yaml` is the file that DECLARES
    gate inputs and leaving it out would let a future row key a gate on the phantom.
    """
    hits = (_grep(_MONITOR, "draw_target_fraction", patterns=("*.py", "*.yaml", "*.yml"))
            + _grep(_COORD, "draw_target_fraction", patterns=("*.py", "*.yaml", "*.yml")))
    assert hits == [], f"draw_target_fraction is a phantom input and is banned here: {hits}"


def test_draw_target_fraction_ban_bites_planted_reference(tmp_path: Path) -> None:
    """A planted `draw_target_fraction` read is detected."""
    (tmp_path / "planted.py").write_text('x = pool_stats["draw_target_fraction"]\n')
    assert _grep(tmp_path, "draw_target_fraction"), "census must bite a planted phantom reference"
