"""O4 — amp dtype resolver (resolve/amp.resolve_amp_dtype) + O4b DAG no-torch guard.

graph->bf16 is a pinned code constant (LAW-06 / F-11: fp16 GINE overflow -> NaN); the token
is a STRING, never a torch.dtype (config -> encoding, util only; the model maps token->dtype).
"""
import importlib
import pkgutil
import sys

from mantis.config.resolve.amp import resolve_amp_dtype


def test_graph_is_bf16_pinned_law06():
    assert resolve_amp_dtype("graph") == "bf16"


def test_grid_is_fp16_historical():
    assert resolve_amp_dtype("grid") == "fp16"


def test_returns_string_token_never_torch_dtype():
    assert isinstance(resolve_amp_dtype("graph"), str)
    assert isinstance(resolve_amp_dtype("grid"), str)


def test_o4b_importing_config_package_never_pulls_torch():
    # Import the whole config package + every submodule; torch must stay absent (DAG guard).
    sys.modules.pop("torch", None)
    import mantis.config as cfg_pkg

    importlib.import_module("mantis.config")
    for mod in pkgutil.walk_packages(cfg_pkg.__path__, prefix="mantis.config."):
        importlib.import_module(mod.name)
    assert "torch" not in sys.modules
