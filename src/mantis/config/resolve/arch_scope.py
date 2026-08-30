"""`refuse_outside_its_arch` — the read-path half of the schema's arch partition (R322(d)).

`mantis.config.schema.core.ARCH_SCOPED_KEYS` is the ONE authority on which config blocks
belong to which representation, and `RunConfig` refuses a config that carries a block its arch
does not have. That closes the SCHEMA half of `SEAM_V1_DESIGN` §3's red row. This module closes
the READ-PATH half: an arch-scoped key's one resolver refuses a config of the wrong arch BY
NAME, instead of answering.

WHY BOTH HALVES, when the schema already refuses. The two catch different things and the
conformance suite's T9 section executes them as two separate red classes. The schema half binds
what a validated `RunConfig` may contain; this half binds what the resolver does when it is
handed a plain mapping — which is what every resolver signature actually takes, because
`train/coordinator/` is the DAG-clean seam layer and nothing there imports a schema class. A
resolver that refuses only on ABSENCE is green today by accident: it is green because the key
happens not to be there, and it turns red again the moment anyone hand-adds the block back.

WHY THE CHECK IS CONDITIONAL ON A READABLE REPRESENTATION, and why that is not a hole. The
subject of these resolvers is the BLOCK, and a mapping with no readable `identity.representation`
makes no arch claim for this function to contradict — the block-presence refusals below it still
fire, so absence is still an error and never a default. Every config that came through
`mantis.config.loader.load_config` carries `identity.representation` by construction, and R1
makes that the only legitimate source of a production config; the conformance suite pins that
premise directly rather than leaving it implied, so this conditional cannot go quietly vacuous.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mantis.config.schema.core import ARCH_SCOPED_KEYS


class ArchScopedKeyOutsideItsArchError(ValueError):
    """An arch-scoped block's read path was handed a config of a different representation.

    A `ValueError` for `MissingEncodingError`'s reason: reading a graph-only cap on a grid run
    is a configuration ERROR, not a condition to recover from.
    """


def declared_representation(full_config: Any) -> str | None:
    """`identity.representation` if the mapping carries one readably, else `None`.

    Returns:
        The declared representation, or `None` when the config is not a mapping, carries no
        `identity` section, or carries one that is not a mapping — i.e. whenever the config
        makes no arch claim this module could contradict.
    """
    if not isinstance(full_config, Mapping):
        return None
    identity = full_config.get("identity")
    if not isinstance(identity, Mapping):
        return None
    representation = identity.get("representation")
    return representation if isinstance(representation, str) else None


def refuse_outside_its_arch(full_config: Any, section: str, field: str) -> None:
    """Raise if `full_config` declares a representation that `section.field` does not belong to.

    Args:
        full_config: the resolver's own argument, a plain mapping.
        section: the `RunConfig` section holding the block.
        field: the block's field name inside that section.

    Raises:
        ArchScopedKeyOutsideItsArchError: the config declares a representation and the key is
            arch-scoped to a different one.
        KeyError: `section.field` is not in `ARCH_SCOPED_KEYS` — a caller asking this module to
            police a key the partition does not place is asking about a rule that does not
            exist, and answering "fine" would be the phantom-gate shape (LAW-07).
    """
    key = next(
        (k for k in ARCH_SCOPED_KEYS if k.section == section and k.field == field), None
    )
    if key is None:
        raise KeyError(
            f"{section}.{field} is not in ARCH_SCOPED_KEYS, so there is no arch scope to "
            "enforce; place it in the partition or do not ask"
        )
    declared = declared_representation(full_config)
    if declared is not None and declared != key.arch:
        raise ArchScopedKeyOutsideItsArchError(
            f"{section}.{field} is ARCH-SCOPED to representation={key.arch!r} and this config "
            f"declares identity.representation={declared!r}. The read path REFUSES rather than "
            f"answering: {key.grounds}. (R322(d), `SEAM_V1_DESIGN` §3 — an arch-scoped key "
            "reachable outside its arch is a red row.)"
        )


__all__ = [
    "ArchScopedKeyOutsideItsArchError",
    "declared_representation",
    "refuse_outside_its_arch",
]
