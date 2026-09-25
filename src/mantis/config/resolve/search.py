"""The two `search.kind` resolvers — ONE reader per key, and each wire reads its own.

Each returns its key unchanged; they exist so "the workers run `selfplay.search.kind`" and "the
deploy head runs `deploy.search.kind`" are true BY CONSTRUCTION, not by call sites agreeing.
NO DEFAULT: a bar that cannot say which search it ran is not a bar.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: The kinds this build implements. NOT a value default — the value lives in the config;
#: this is the name authority, the same posture as `resolve/nsims.py`'s opponent tuple.
SEARCH_KINDS: tuple[str, ...] = ("puct", "gumbel")


class MissingSearchKindError(ValueError):
    """The key is absent from the mapping, or is not a kind this build implements."""


def _section(config: Mapping[str, Any] | Any, name: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def _resolve(config: Mapping[str, Any] | Any, section: str) -> str:
    home = _section(config, section)
    search = _section(home, "search") if home is not None else None
    kind = _section(search, "kind") if search is not None else None
    dotted = f"{section}.search.kind"
    if kind is None:
        raise MissingSearchKindError(
            f"{dotted} is required and has no default (R1/LAW-11). The search regime "
            "decides the root mechanism, the interior selector and the exported target's "
            "semantics; the deploy key is what makes the promotion bar (LAW-15) a claim "
            "rather than a coincidence, and the self-play key is what a stored ring's rows mean."
        )
    kind = str(kind)
    if kind not in SEARCH_KINDS:
        raise MissingSearchKindError(
            f"{dotted}={kind!r} is not a search kind this build implements; known: "
            f"{list(SEARCH_KINDS)}. REFUSED rather than defaulted — a typo silently "
            "becoming 'puct' is the silent-fallback class LAW-11 closes."
        )
    return kind


def resolve_selfplay_search_kind(config: Mapping[str, Any] | Any) -> str:
    """The workers' kind, `selfplay.search.kind`, from a `RunConfig` or its `model_dump()`.

    Raises:
        MissingSearchKindError: the key is absent or not a kind this build implements — never
            defaulted, and never read from the deploy key (the training search would then
            follow what the deploy head plays).
    """
    return _resolve(config, "selfplay")


def resolve_deploy_search_kind(config: Mapping[str, Any] | Any) -> str:
    """The deploy head's kind, `deploy.search.kind` — the bar's and every external cell's.

    Raises:
        MissingSearchKindError: the key is absent or not a kind this build implements — never
            defaulted, and never read from the self-play key (the bar is matched to what will
            be deployed).
    """
    return _resolve(config, "deploy")
