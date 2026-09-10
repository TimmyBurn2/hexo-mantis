"""O4 — amp dtype resolver (resolve/amp.resolve_amp_dtype) + O4b DAG no-torch guard.

graph->bf16 is a pinned code constant (LAW-06 / F-11: fp16 GINE overflow -> NaN); the token
is a STRING, never a torch.dtype (config -> encoding, util only; the model maps token->dtype).
The declared-value argument went with `train.amp_dtype` (R346(f)) — the resolver takes the
representation alone, so there is no longer a config spelling that could disagree with LAW-06.
"""
import re
import subprocess
import sys
from pathlib import Path

import pytest

from mantis.config.resolve.amp import resolve_amp_dtype

_AMP_MODULE = (
    Path(__file__).resolve().parents[2] / "src" / "mantis" / "config" / "resolve" / "amp.py"
)


def test_graph_is_bf16_pinned_law06():
    assert resolve_amp_dtype("graph") == "bf16"


def test_unknown_representation_raises():
    with pytest.raises(ValueError):
        resolve_amp_dtype("bogus_representation")


def test_returns_string_token_never_torch_dtype():
    assert isinstance(resolve_amp_dtype("graph"), str)


def test_dag_purity_amp_module_imports_no_torch() -> None:
    """DAG-purity regression: `mantis.config` must not pull torch (the module's own
    docstring/design constraint) — a narrower, single-module source-scan than the
    whole-package subprocess guard `test_o4b_...` below already owns."""
    text = _AMP_MODULE.read_text(encoding="utf-8")
    assert not re.search(r"^\s*(import torch|from torch)", text, re.M), (
        "config/resolve/amp.py must not import torch"
    )


def test_o4b_importing_config_package_never_pulls_torch():
    # DAG guard: importing the whole config package must not pull torch. Runs in a FRESH
    # interpreter — popping an installed torch from sys.modules and re-importing re-executes
    # torch's C++ init (double TORCH_LIBRARY(triton) registration), which would poison the
    # shared pytest session. The subprocess proves the guard without touching this process.
    code = (
        "import importlib, pkgutil, sys\n"
        "import mantis.config as cfg\n"
        "for mod in pkgutil.walk_packages(cfg.__path__, prefix='mantis.config.'):\n"
        "    importlib.import_module(mod.name)\n"
        "leaked = sorted(m for m in sys.modules if m == 'torch' or m.startswith('torch.'))\n"
        "assert not leaked, leaked\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
