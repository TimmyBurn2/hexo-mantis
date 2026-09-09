"""`search.kind` resolver — THE one selector both the self-play pool and the deploy head read.

WHY A RESOLVER FOR A ONE-LEAF SECTION. Not to transform anything: `resolve_search_kind`
returns the key unchanged. It exists so that "the deploy head runs the same search as
self-play" is true BY CONSTRUCTION rather than by two call sites happening to agree. The
head that decides a promotion (LAW-15's deploy-matched bar) and the workers that generate
the targets must read ONE authority; before this, the eval head's search regime was not
read from the config at all — it was `DeployHeadPlayer`'s own constructor signature, and
the run's regime and the bar's regime were only ever equal by coincidence (AUDIT-1 F-39
found the same class on `c_visit`/`c_scale`).

NO DEFAULT (R1/LAW-11). A mapping without `search.kind` raises: a bar that cannot say
which search it ran is not a bar.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: The kinds this build implements. NOT a value default — the value lives in the config;
#: this is the name authority, the same posture as `resolve/nsims.py`'s opponent tuple.
SEARCH_KINDS: tuple[str, ...] = ("puct", "gumbel")


class MissingSearchKindError(ValueError):
    """`search.kind` is absent from the mapping, or is not a kind this build implements."""


def resolve_search_kind(config: Mapping[str, Any] | Any) -> str:
    """The run's search kind, from a validated `RunConfig` or its `model_dump()` mapping.

    Args:
        config: a `RunConfig`, or any mapping shaped like one (`{"search": {"kind": ...}}`).

    Returns:
        The kind's config spelling — one of `SEARCH_KINDS`.

    Raises:
        MissingSearchKindError: the mapping carries no `search.kind`, or carries a value
            this build does not implement. Never defaulted: a silent `puct` would let a
            config that never declared its search regime still boot one, and would let the
            deploy head disagree with the self-play workers without a config diff to show
            for it.
    """
    section: Any
    if isinstance(config, Mapping):
        section = config.get("search")
        kind = section.get("kind") if isinstance(section, Mapping) else None
    else:
        section = getattr(config, "search", None)
        kind = getattr(section, "kind", None)
    if kind is None:
        raise MissingSearchKindError(
            "search.kind is required and has no default (R1/LAW-11). The search regime "
            "decides the root mechanism, the interior selector and the exported target's "
            "semantics, and it is the key that makes the deploy-matched promotion bar "
            "(LAW-15) a claim rather than a coincidence."
        )
    kind = str(kind)
    if kind not in SEARCH_KINDS:
        raise MissingSearchKindError(
            f"search.kind={kind!r} is not a search kind this build implements; known: "
            f"{list(SEARCH_KINDS)}. REFUSED rather than defaulted — a typo silently "
            "becoming 'puct' is the silent-fallback class LAW-11 closes."
        )
    return kind
