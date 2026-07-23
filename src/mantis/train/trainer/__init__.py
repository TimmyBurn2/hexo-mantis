"""The Trainer package (WP10 §a.4). `core.Trainer` owns one training step + checkpoint IO."""
from __future__ import annotations

from mantis.train.trainer.core import (
    Trainer,
    TrainHParams,
    build_param_groups,
)

__all__ = ["Trainer", "TrainHParams", "build_param_groups"]
