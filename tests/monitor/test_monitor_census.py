"""O-15 / O-18 / O-19 / O-20 — mechanical censuses over `src/mantis/monitor/**` (+ the
train→monitor import-site inventory + the draw_target_fraction ban).

These are STATIC/subprocess censuses (AST + a torch-blocked subprocess import): they do NOT
`import mantis.monitor.*` at module top, so this file collects cleanly while the ⊕ oracle
files are RED-at-import. Each checker carries a LAW-07 mutation self-test proving it BITES.

  O-18 / P-18 — `import mantis.monitor` + a walk of every submodule succeeds under a meta-path
    torch blocker (rc 0); zero `torch` import tokens in `monitor/**` (§2 L94 headless law).
  O-19 / P-19 — `monitor/**` mantis-imports ⊆ {util, encoding, monitor}; train files importing
    `mantis.monitor` are EXACTLY {coordinator/step.py, subsystems.py, lifecycle/heartbeat_watchdog.py}.
  O-20 / P-20 — zero `except …: pass` swallow sites in `monitor/**`, NO allowlist; a planted
    swallow ⇒ detected (the J-04 pattern, reused).
  O-15 / P-15 — zero `draw_target_fraction` references in `monitor/**` + `train/coordinator/**`
    gate code (the NaN phantom-input landmine, `pool_push.py:135`).
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
# The pinned train→monitor import-site inventory (O-19 / P-19). Paths relative to src/mantis.
_EXPECTED_TRAIN_MONITOR_SITES = {
    "train/coordinator/step.py",
    "train/subsystems.py",
    "train/lifecycle/heartbeat_watchdog.py",
}


# ── AST helpers ───────────────────────────────────────────────────────────────────────
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


# ── O-18 no torch in monitor ──────────────────────────────────────────────────────────
def test_monitor_imports_without_torch_subprocess() -> None:
    """O-18 / P-18 — a fresh interpreter imports `mantis.monitor` and every submodule under a
    meta-path torch blocker with rc 0. A liveness babysitter must never need torch to load."""
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
    """O-18 — a static backstop: no `import torch` / `from torch` token anywhere in
    `monitor/**` (a lazy in-function torch import would evade the subprocess walk)."""
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


# ── O-19 import-DAG census ────────────────────────────────────────────────────────────
def test_monitor_mantis_imports_are_within_the_allowed_set() -> None:
    """O-19 / P-19 — every `mantis.*` import in `monitor/**` is under {util, encoding, monitor}.
    Bites a hidden hard edge (e.g. `mantis.train` or `mantis.selfplay`) into the headless core."""
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


def test_train_to_monitor_import_sites_are_exactly_the_pinned_three() -> None:
    """O-19 / P-19 — the ONLY train files that top-level import `mantis.monitor` are the three
    declared seams. Any extra edge (or a missing one) is DAG drift. NOTE for IMPL: the existing
    `tests/train/test_train_import_dag.py` FORBIDDEN set must drop `mantis.monitor` — this
    census with its exact allowlist takes over policing that edge (ORACLE_NOTES)."""
    sites: set[str] = set()
    for path in sorted(_TRAIN.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for target in _top_level_imports(tree):
            if target == "mantis.monitor" or target.startswith("mantis.monitor."):
                sites.add(str(path.relative_to(_TRAIN.parent)))
                break
    assert sites == _EXPECTED_TRAIN_MONITOR_SITES, (
        f"train→monitor import sites must be exactly the pinned three; got {sorted(sites)}"
    )


# ── O-20 except-pass census ───────────────────────────────────────────────────────────
def test_no_swallow_sites_in_monitor() -> None:
    """O-20 / P-20 — ZERO `except …: pass` swallow sites in `monitor/**`, NO allowlist. Every
    optional effect goes through `best_effort` (counted) or fails loud."""
    sites = _swallow_sites(_MONITOR)
    assert sites == [], f"monitor/** must have zero except-pass swallow sites, found: {sites}"


def test_swallow_census_bites_planted_swallow(tmp_path: Path) -> None:
    """O-20 / P-20 (LAW-07) — the detector reports exactly 1 site for a planted swallow, proving
    the census is not vacuous."""
    (tmp_path / "planted.py").write_text(
        "def f():\n    try:\n        g()\n    except Exception:\n        pass\n"
    )
    assert len(_swallow_sites(tmp_path)) == 1, "census must bite a planted except-pass"


# ── O-15 draw_target_fraction ban ─────────────────────────────────────────────────────
def test_draw_target_fraction_absent_from_monitor_and_coordinator_gates() -> None:
    """O-15 / P-15 — zero `draw_target_fraction` references in `monitor/**` or
    `train/coordinator/**`. The draw-rate gate keys on the LIVE `pooled_draw_rate`, never
    the NaN `draw_target_fraction` phantom (`pool_push.py:135`).

    The scan covers `*.yaml` as well as `*.py` (REVIEW-impl F-8): `producer_manifest.yaml` is
    the file that DECLARES gate inputs, so leaving it outside the ban would let a future row
    key a gate on the phantom without biting."""
    hits = (_grep(_MONITOR, "draw_target_fraction", patterns=("*.py", "*.yaml", "*.yml"))
            + _grep(_COORD, "draw_target_fraction", patterns=("*.py", "*.yaml", "*.yml")))
    assert hits == [], f"draw_target_fraction is a phantom input and is banned here: {hits}"


def test_draw_target_fraction_ban_bites_planted_reference(tmp_path: Path) -> None:
    """O-15 (LAW-07) — a planted `draw_target_fraction` read is detected."""
    (tmp_path / "planted.py").write_text('x = pool_stats["draw_target_fraction"]\n')
    assert _grep(tmp_path, "draw_target_fraction"), "census must bite a planted phantom reference"
