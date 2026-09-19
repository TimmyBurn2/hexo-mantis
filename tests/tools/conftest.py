"""Sweep the preflight oracles' probe paths before and after the session (a failing guard would poison every later run), and load the `tools/` packages by path."""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
#: The probe paths from the out-dir and symlink oracles, kept as literals rather than imported
#: so this file is not a consumer of a frozen oracle's internals.
PROBES = (REPO_ROOT / "_preflight_oracle_outdir",
          REPO_ROOT / "_preflight_symlink_probe")


def _sweep() -> None:
    """Remove each probe path loudly, the symlink arm FIRST: `is_dir()` follows symlinks, `rmtree` refuses them (195 collection errors when it ran second)."""
    for probe in PROBES:
        if probe.is_symlink():
            try:
                probe.unlink()
            except OSError as exc:
                raise RuntimeError(
                    f"could not unlink the preflight probe symlink {probe}: {exc}. Remove it "
                    "by hand — while it exists the probe path is not usable by the oracle."
                ) from exc
            continue
        if not probe.is_dir():
            continue
        try:
            shutil.rmtree(probe)
        except OSError as exc:
            raise RuntimeError(
                f"could not sweep the preflight probe path {probe}: {exc}. It must be removed "
                "by hand — while it exists, `test_an_out_dir_inside_the_repo_is_refused` fails "
                "on its PRECONDITION rather than on the guard it exists to witness, and the "
                "tree carries an untracked artifact directory (R7 / gate 6)."
            ) from exc


# THE one preflight wall-clock budget: a fixed bar on a real boot must clear the slowest host
# that runs it, so it is derived from completed drives rather than transcribed per test.
# Measured: dev host 161.6 s idle / 200.1 s under 698% load (2026-08-18, 16 cores); migration box
# 447 s; CI runner 1205 s (2026-08-19, the slowest budget-governed row, after 900 was measured
# red there at a 915.4 s truncation). 1800 covers the slowest measured row at ~1.5x.
PREFLIGHT_BUDGET_SEC = 1800.0

#: Derived, never transcribed: if `subprocess.run(timeout=...)` fires first the tool never writes
#: the rc-40 report the tests read, and the harness timeout hides the tool's verdict.
PREFLIGHT_HARNESS_CEILING_SEC = PREFLIGHT_BUDGET_SEC + 120.0


@pytest.fixture(scope="session")
def preflight_budget_sec() -> float:
    """Serve the one preflight wall-clock budget to sibling test modules."""
    return PREFLIGHT_BUDGET_SEC


@pytest.fixture(scope="session")
def preflight_harness_ceiling_sec() -> float:
    """Serve the harness ceiling, always above the tool budget."""
    return PREFLIGHT_HARNESS_CEILING_SEC


@pytest.fixture(scope="session", autouse=True)
def _preflight_probe_path_is_not_left_in_the_tree():
    """Sweep before as well as after, so an earlier session's leftovers cannot fail the guard's oracle."""
    _sweep()
    yield
    _sweep()


def _load_puller():
    spec = importlib.util.spec_from_file_location("_conftest_mirror_pull",
                                                  REPO_ROOT / "tools" / "mirror_pull.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_dirs_under(base: Path) -> list[tuple[Path, str]]:
    """`(run dir, run_id)` for every preflight run dir under `base`, by checkpoint or manifest."""
    from mantis.train.bundle_receipts import CHECKPOINT_NAME_RE

    found: dict[Path, str] = {}
    for ckpt in base.rglob("checkpoints/*.ckpt"):
        match = CHECKPOINT_NAME_RE.match(ckpt.name)
        if match is not None:
            found.setdefault(ckpt.parents[1], match.group("run_id"))
    for manifest in base.rglob("checkpoints/*.bundle.json"):
        try:
            run_id = str(json.loads(manifest.read_text(encoding="utf-8"))["run_id"])
        except (OSError, ValueError, KeyError):
            continue
        found.setdefault(manifest.parents[1], run_id)
    return sorted(found.items())


@pytest.fixture(scope="module")
def local_puller(tmp_path_factory) -> Iterator[Path]:
    """The R349(b) loop in miniature: the REAL puller as a LOCAL loop over every preflight run dir under pytest's tmp base."""
    puller = _load_puller()
    base = Path(tmp_path_factory.getbasetemp())
    mirrors = tmp_path_factory.mktemp("mirror")
    stop = threading.Event()

    def loop() -> None:
        cycle = 0
        while not stop.is_set():
            cycle += 1
            for run_dir, run_id in _run_dirs_under(base):
                if mirrors in run_dir.parents:
                    continue
                mirror = mirrors / f"{run_dir.parent.name}_{run_dir.name}"
                try:
                    puller.run_cycle(str(run_dir), mirror, run_id, cycle=cycle,
                                     mirror_id="local_puller")
                except puller.MirrorTransportError:
                    continue
            stop.wait(1.0)

    thread = threading.Thread(target=loop, name="local_puller", daemon=True)
    thread.start()
    try:
        yield mirrors
    finally:
        stop.set()
        thread.join(timeout=30)


def load_tools_package(name: str):
    """Load `tools/<name>` by path under that name, once per process; `sys.path` untouched (R5)."""
    if name in sys.modules:
        return sys.modules[name]
    pkg_dir = REPO_ROOT / "tools" / name
    spec = importlib.util.spec_from_file_location(
        name, pkg_dir / "__init__.py", submodule_search_locations=[str(pkg_dir)],
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_dashboard_package():
    """The `tools/dashboard` package, by path."""
    return load_tools_package("dashboard")


@pytest.fixture(scope="session")
def dashboard():
    """The `tools/dashboard` package, with its submodules importable as `dashboard.<name>`."""
    return load_dashboard_package()


@pytest.fixture(scope="session")
def viewer():
    """The `tools/viewer` package (VIEWER-1), with its submodules importable as `viewer.<name>`."""
    return load_tools_package("viewer")


@pytest.fixture(scope="session")
def ladder():
    """The `tools/ladder` package (LADDER-1), with its submodules importable as `ladder.<name>`."""
    return load_tools_package("ladder")


@pytest.fixture(scope="session")
def analyzer():
    """The `tools/analyzer` package (ANALYZER-1), with its submodules importable as `analyzer.<name>`."""
    return load_tools_package("analyzer")


def mint_analyzer_stamp(directory: Path, *, run_id: str = "an1", step: int = 7, deploy_kind: str | None = None) -> Path:
    """A stamped checkpoint of a tiny GnnArchV2 over dev_example's config (test_arch_stamp_authority's recipe)."""
    from mantis.config.loader import load_config
    from mantis.encoding import lookup
    from mantis.model import GnnArchV2, build_net
    from mantis.train.checkpoints import save_checkpoint

    cfg = load_config(REPO_ROOT / "configs" / "dev_example.yaml").model_dump()
    if deploy_kind is not None:
        cfg["deploy"]["search"]["kind"] = deploy_kind
    spec = lookup(cfg["identity"]["encoding"])
    arch = GnnArchV2(in_dim=int(spec.node_feat_dim), edge_dim=int(spec.edge_feat_dim), hidden=8, num_layers=1,
                     policy_hidden=8, value_hidden=8)
    return save_checkpoint(
        model=build_net(arch), optimizer=None, scaler=None, scheduler=None, step=step, config=cfg,
        metadata_kwargs={"encoding_name": cfg["identity"]["encoding"], "run_id": run_id, "arch": arch},
        checkpoint_dir=directory, kind="weights")


@pytest.fixture(scope="session")
def mint_stamp():
    """The stamp minter as a fixture, so analyzer tests share one recipe without importing each other."""
    return mint_analyzer_stamp
