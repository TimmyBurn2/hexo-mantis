"""The mirrored parent checkpoints named in `MANTIS_PARENT_CHECKPOINTS` (`os.pathsep`-separated; LOUD skip if unset) load under the HEAD schema, stamps verbatim."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from mantis.train.checkpoints import load_checkpoint

_ENV = "MANTIS_PARENT_CHECKPOINTS"
_PATHS = [Path(p) for p in os.environ.get(_ENV, "").split(os.pathsep) if p]


@pytest.mark.skipif(not _PATHS, reason=f"LOUD SKIP — no parent stamps named; set {_ENV}")
@pytest.mark.parametrize("path", _PATHS, ids=[p.name for p in _PATHS])
def test_a_mirrored_parent_checkpoint_loads_with_its_stamp_verbatim(path: Path) -> None:
    """The parent loads, and the returned config is byte-for-byte the stamped one (never repaired)."""
    assert path.is_file(), f"{_ENV} names {path}, which is not a file"
    stamped = torch.load(path, weights_only=True, map_location="cpu")["config"]
    ck = load_checkpoint(path)
    assert ck.model_state, f"{path.name}: loaded with no weights"
    assert ck.config == stamped, f"{path.name}: the loader repaired the stamp"
