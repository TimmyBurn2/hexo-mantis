"""Bootstrap pretrain package (WP10 §a.7) — corpus -> net pretraining for `mantis.train`.

The public surface is `pretrain` (the CLI main). The dense dataset, the dense
`BootstrapTrainer`, the CNN-attribute freeze helper and the dense validator went with the
grid path (R346(f)); the graph BC route is `mantis.train.pretrain.graph_route`.
Entry point: `python -m mantis.train.pretrain` (see `__main__.py`).
"""
from __future__ import annotations

__all__ = ["pretrain"]


def pretrain(argv: list[str] | None = None) -> None:
    """The CLI main (lazy import of `cli` so the package imports without argparse side effects).

    Args:
        argv: command-line arguments, or `None` to read `sys.argv`.
    """
    from mantis.train.pretrain.cli import pretrain as _pretrain

    _pretrain(argv)
