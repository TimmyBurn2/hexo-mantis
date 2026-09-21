"""A schema-valid `RunConfig` mapping for the warm-start suite's `save_checkpoint` calls.

`save_checkpoint` validates its `config` against the live schema before writing, so a fixture
cannot hand it a stub. This borrows the one complete example config the repo ships rather than
transcribing 180-odd keys — a transcription would be a second copy of the schema that goes
stale silently.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]


def minimal_config(arch: Any = None) -> dict[str, Any]:
    """The shipped graph example config as a plain mapping; with `arch`, `model.gnn` at the arch's own widths (v35)."""
    from mantis.config.loader import load_config

    config = load_config(_REPO / "configs" / "smoke_preflight_armed.yaml").model_dump()
    if arch is not None:
        config["model"]["gnn"] = {"hidden": int(arch.hidden), "num_layers": int(arch.num_layers)}
    return config
