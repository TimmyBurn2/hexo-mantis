"""Training bootstrap/resume checkpoint resolver (PORT of frozen resolve/bootstrap.py).

One resolver for `mantis.run`'s ``--resume-from``: validate the resolved path exists at LAUNCH
(before torch.load) so a stale path fails loudly + early. A CLI arg, not a config-file key → no
schema field; `exists` is injectable so this module needs no torch import.

WIRED, LATE (AUDIT-1 F-47). This resolver had ZERO callers: `run.main` passed `--resume-from`
straight through to `launch_run`, so a mistyped path surfaced as whatever `torch.load` says
about a missing file, deep inside `init_trainer`'s resume branch, after the composition root
had already built a run. The guard existed and the flag existed and nothing joined them. Its
error text also named a `BOOTSTRAP=<path>` make target and a `--checkpoint` flag that do not
exist — a message pointing at knobs the operator cannot use, corrected here to the one that
does. It is called now, at `run.main`, before `launch_run`.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass


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
            "Pass `--resume-from <path>` to an existing file, or omit it for a fresh run "
            "(validated at launch, not a late torch.load failure)."
        )
    return ResolvedBootstrap(path=cli_checkpoint, source="cli")
