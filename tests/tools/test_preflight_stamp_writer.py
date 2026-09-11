"""R348(c): the preflight is the ONE stamp writer, and it writes only after its verdict."""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mantis.config.loader import config_identity_sha256, load_config
from mantis.config.preflight_stamp import (
    PreflightStampMalformedError,
    require_preflight_stamp,
    stamp_path,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "ci_gates" / "preflight_mint.py"


def _tool() -> Any:
    spec = importlib.util.spec_from_file_location("_stamp_writer_tool", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = _tool()
_CONFIG_PATH = REPO_ROOT / "configs" / "run6.yaml"


@pytest.fixture
def state_home(monkeypatch, tmp_path) -> Path:
    home = tmp_path / "state"
    monkeypatch.setenv("XDG_STATE_HOME", str(home))
    return home


def _report(**halts) -> dict:
    report = TOOL._new_report("preflight")
    report["override"] = {"booted_config_sha256": "booted-sha"}
    report.update(halts)
    return report


def test_the_stamp_the_tool_writes_is_the_stamp_the_launcher_accepts(
    state_home: Path, tmp_path: Path,
) -> None:
    config = load_config(_CONFIG_PATH)
    report = _report(workspace={"verdict": "MIRRORED", "run_dir": "/x",
                                "bundle": {"step": 1, "files": {}},
                                "shard": {"name": "s", "sha256": ""}},
                     cuda_build={"verdict": "not_run", "reason": "cpu"})
    args = SimpleNamespace(burst_steps=16)
    TOOL._stamp_pass(config, _CONFIG_PATH, args, report, tmp_path / "out")

    stamp = require_preflight_stamp(config, tree_root=REPO_ROOT)
    assert stamp["halts"] == {"workspace": report["workspace"], "cuda_build": report["cuda_build"]}
    assert stamp["booted_config_sha256"] == "booted-sha" and stamp["burst_steps"] == 16
    assert stamp["report"].startswith(str(tmp_path / "out"))
    assert report["preflight_stamp"] == str(stamp_path(config_identity_sha256(config)))


def test_a_report_missing_a_start_halt_reading_writes_no_stamp(
    state_home: Path, tmp_path: Path,
) -> None:
    """Mutation: a tool that stopped recording a halt cannot mint a stamp that hides it."""
    config = load_config(_CONFIG_PATH)
    report = _report(workspace={"verdict": "DURABLE"})
    with pytest.raises(KeyError):
        TOOL._stamp_pass(config, _CONFIG_PATH, SimpleNamespace(burst_steps=1), report, tmp_path)
    assert not stamp_path(config_identity_sha256(config)).exists()
    with pytest.raises(PreflightStampMalformedError):
        TOOL.write_stamp(config=config, config_path=_CONFIG_PATH, tree_root=REPO_ROOT,
                         halts={"workspace": {}}, booted_config_sha256="x", burst_steps=1,
                         report_path=tmp_path)


def _statement_index(body: list[ast.stmt], callee: str) -> int:
    """Index of the first top-level statement in `body` that calls `callee`."""
    for i, stmt in enumerate(body):
        for node in ast.walk(stmt):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == callee):
                return i
    raise AssertionError(f"`_run_preflight` never calls {callee}")


def test_the_stamp_is_written_after_the_verdict_and_cleared_before_the_boot() -> None:
    """Source order in `_run_preflight`: clear < cuda halt < child < verdict < mirror halt <
    stamp — a stamp can never precede the receipts it certifies (R349(b))."""
    tree = ast.parse(TOOL_PATH.read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_run_preflight")
    order = [_statement_index(fn.body, name) for name in
             ("clear_stamp", "_assert_cuda_build_halt", "_run_child", "_verdict_exit",
              "_assert_mirror_receipts_halt", "_stamp_pass")]
    assert order == sorted(order) and len(set(order)) == 6, (
        f"statement order (clear, cuda halt, child, verdict, mirror halt, stamp) = {order}")


def test_audit_mode_never_touches_the_stamp_store() -> None:
    """Gate 12 runs `--audit-only` per commit; it must neither write nor clear a stamp."""
    tree = ast.parse(TOOL_PATH.read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_run_audit")
    calls = {node.func.id for node in ast.walk(fn)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert not calls & {"clear_stamp", "write_stamp", "_stamp_pass"}, calls
