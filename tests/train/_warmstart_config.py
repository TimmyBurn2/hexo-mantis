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


def minimal_config() -> dict[str, Any]:
    """The shipped graph example config, as a plain mapping."""
    from mantis.config.loader import load_config

    return load_config(_REPO / "configs" / "smoke_gnn.yaml").model_dump()
