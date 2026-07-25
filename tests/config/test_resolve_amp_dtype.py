"""WPSC Phase 3 SC-B3 — `resolve_amp_dtype(representation, declared_amp_dtype)` string-
level mirror of `tests/model/test_amp_dtype_p3.py` (torch-free; R30b, DESIGN_P3.md §4.3/
§4.4). New file — no prior resolvers-focused amp test at this signature exists.

RED at HEAD (`507c23b`): `resolve_amp_dtype` still takes exactly 1 positional arg
(`representation`) — every 2-arg call below raises TypeError.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from mantis.config.resolve.amp import resolve_amp_dtype

_AMP_MODULE = (
    Path(__file__).resolve().parents[2] / "src" / "mantis" / "config" / "resolve" / "amp.py"
)


@pytest.mark.parametrize("declared_amp_dtype", ["fp16", "bf16"])
def test_graph_ignores_declared_value_always_bf16(declared_amp_dtype: str) -> None:
    assert resolve_amp_dtype("graph", declared_amp_dtype) == "bf16"


def test_grid_returns_declared_value() -> None:
    assert resolve_amp_dtype("grid", "fp16") == "fp16"
    assert resolve_amp_dtype("grid", "bf16") == "bf16"


def test_grid_invalid_declared_value_raises() -> None:
    with pytest.raises(ValueError):
        resolve_amp_dtype("grid", "garbage")


def test_unknown_representation_raises() -> None:
    with pytest.raises(ValueError):
        resolve_amp_dtype("bogus_representation", "fp16")


def test_dag_purity_amp_module_imports_no_torch() -> None:
    """DAG-purity regression: `mantis.config` must not pull torch (the module's own
    docstring/design constraint) — a narrower, single-module source-scan than the
    whole-package subprocess guard `test_resolve_amp.py::test_o4b_...` already owns."""
    text = _AMP_MODULE.read_text()
    assert not re.search(r"^\s*(import torch|from torch)", text, re.M), (
        "config/resolve/amp.py must not import torch"
    )
