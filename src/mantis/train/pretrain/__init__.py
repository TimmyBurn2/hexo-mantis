"""Bootstrap pretrain package (WP10 §a.7) — corpus → net pretraining for `mantis.train`.

Folds the old `bootstrap/pretrain.py` re-export shim: the public surface is `pretrain` (the CLI
main), `BootstrapTrainer`, `AugmentedBootstrapDataset` / `make_augmented_collate`, `validate`,
`_apply_finetune_freeze`. Entry point: `python -m mantis.train.pretrain` (see `__main__.py`).

KILLED (do not cross): the v8 dataset + the v8 augment branch (v8 never crosses); the raw-JSON
legacy `load_corpus` fallback (0 config consumers).
"""
from __future__ import annotations

from mantis.train.pretrain.dataset import (
    AugmentedBootstrapDataset,
    make_augmented_collate,
)
from mantis.train.pretrain.freeze import _apply_finetune_freeze
from mantis.train.pretrain.trainer import BootstrapTrainer
from mantis.train.pretrain.validate import validate

__all__ = [
    "AugmentedBootstrapDataset",
    "BootstrapTrainer",
    "_apply_finetune_freeze",
    "make_augmented_collate",
    "pretrain",
    "validate",
]


def pretrain(argv: list[str] | None = None) -> None:
    """The CLI main (lazy import of `cli` so the package imports without argparse side effects)."""
    from mantis.train.pretrain.cli import pretrain as _pretrain

    _pretrain(argv)
