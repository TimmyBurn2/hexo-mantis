"""Training bootstrap/resume checkpoint resolver (PORT of frozen resolve/bootstrap.py).

One resolver for the ``--checkpoint`` / ``BOOTSTRAP`` training bootstrap: validate the resolved
path exists at LAUNCH (before torch.load) so a stale path fails loudly + early. CLI arg, not a
config-file key → no schema field. ``exists`` is injectable (no torch import).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable


class BootstrapNotFoundError(FileNotFoundError):
    """Resolved bootstrap/resume checkpoint path does not exist.

    Subclasses FileNotFoundError so existing handlers still catch it — it just fires EARLIER
    (at launch, in the resolver) and names the override knob.
    """


@dataclass(frozen=True)
class ResolvedBootstrap:
    """The resolved training bootstrap. ``path is None`` ⇒ a fresh run (no checkpoint)."""

    path: str | None
    source: str  # "cli" (a checkpoint path was given) | "none" (fresh run)


def resolve_bootstrap(
    cli_checkpoint: str | None,
    *,
    exists: Callable[[str], bool] = os.path.exists,
) -> ResolvedBootstrap:
    """Resolve + validate the training bootstrap/resume checkpoint.

    ``None`` → ResolvedBootstrap(None, "none") (fresh run). A provided path is validated to exist
    (probe called exactly once) and returns ResolvedBootstrap(path, "cli"); a missing path raises
    BootstrapNotFoundError naming the path + the BOOTSTRAP knob — at launch, not a late torch.load.
    """
    if cli_checkpoint is None:
        return ResolvedBootstrap(path=None, source="none")
    if not exists(cli_checkpoint):
        raise BootstrapNotFoundError(
            f"bootstrap/resume checkpoint {cli_checkpoint!r} does not exist. "
            "Set BOOTSTRAP=<path> (make targets) or --checkpoint <path> to an existing file "
            "(validated at launch, not a late torch.load failure)."
        )
    return ResolvedBootstrap(path=cli_checkpoint, source="cli")
