"""The PUCT constants a minted config states, so a test builds its trees the way a run does instead of restating them."""
from __future__ import annotations

from pathlib import Path

from mantis.config.loader import load_config
from mantis.config.resolve.puct import resolve_puct_constants

#: `configs/dev_example.yaml`'s `selfplay.mcts` constants, through the one resolver.
MINTED_PUCT = resolve_puct_constants(load_config(Path(__file__).resolve().parents[1] / "configs" / "dev_example.yaml"))
