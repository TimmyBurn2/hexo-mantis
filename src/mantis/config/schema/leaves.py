"""THE schema leaf walker — the one derivation of `RunConfig`'s leaf key-paths.

WHY ONE. AUDIT-1 F-44 counted four hand-mirrored copies of this walk; the census at REPAIR-3's
landing found FIVE, because it was scoped to the name `_leaf_paths` and the fifth is called
`live_leaf_paths`. The five did not agree: gate 13 and the consumer-registry bijection walked to
191 leaves, `test_eval_config_remint.py`'s pre-DR-6 copy to 182 (it stopped at `Block | None`,
the exact blindness R93 fixed, while its docstring claimed to mirror the others), and the
conformance partition's copy to 199 (it descends `list[SubModel]`, which the other four treat as
one leaf). Three answers to one question, each asserted as the schema's leaf set.

THE TWO MODES ARE A PARAMETER, NOT A COPY. The 191-answer and the 199-answer are both wanted:
the contract doc and the consumer registry hand out key-paths a config file can WRITE, and
`eval.ladder.rungs` is one such path whose members are list elements; the arch-vocabulary probe
wants every field name a future key could hide an architecture in, including inside a rung. So
the walk takes `descend_containers`, each call site says which question it is asking, and the
divergence is one keyword argument instead of two implementations.

WHAT IS A NESTED BLOCK. A bare `BaseModel`, or a union (`Block | None`, the house arming idiom
under R79) whose non-`None` arms name exactly ONE `BaseModel` — DR-6/R93: an optional block is
DESCENDED, because a fourth unconsumed key inside `DrawRateAbortConfig` once passed the full tier
and gates 7 and 12 green. A union naming two different blocks is a leaf: there is no single
key-path to hand out and guessing one is worse than stopping.

WHAT IS NOT, BY DEFAULT. A generic container — `list[SubModel]`, `dict[str, Block]` — is ONE
leaf (NIT-3). Its members are addressed through an index or a runtime key, so `eval.ladder.rungs.bot`
is not a path any config writes. `descend_containers=True` opts into the member walk for a
consumer that wants field NAMES rather than writable paths.
"""
from __future__ import annotations

import typing
from types import UnionType
from typing import TypeGuard, Union

from pydantic import BaseModel


def _is_block(annotation: object) -> TypeGuard[type[BaseModel]]:
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def nested_block(annotation: object, *, descend_containers: bool = False) -> type[BaseModel] | None:
    """The single nested config BLOCK an annotation names, or None if the field is a leaf.

    Args:
        annotation: the field annotation, as pydantic resolved it.
        descend_containers: also descend a generic container whose args name exactly one
            block (`list[SubModel]`, `dict[str, Block]`). Off by default — see the module
            docstring for which question each mode answers.

    Returns:
        The nested `BaseModel` subclass, or None when the field is a leaf.
    """
    if _is_block(annotation):
        return annotation
    origin = typing.get_origin(annotation)
    if origin is None:
        return None
    if origin in (Union, UnionType) or descend_containers:
        blocks = [arm for arm in typing.get_args(annotation) if _is_block(arm)]
        return blocks[0] if len(blocks) == 1 else None
    return None


def leaf_paths(model: type[BaseModel], prefix: str = "", *,
               descend_containers: bool = False) -> tuple[str, ...]:
    """Every leaf key-path of `model`, dotted, in declaration order.

    Args:
        model: the schema model to walk — `RunConfig` for the shipped run config.
        prefix: the dotted prefix already consumed; callers pass nothing.
        descend_containers: see `nested_block`.

    Returns:
        The leaf key-paths, in the order the fields are declared.
    """
    out: list[str] = []
    for name, field in model.model_fields.items():
        path = f"{prefix}{name}" if not prefix else f"{prefix}.{name}"
        nested = nested_block(field.annotation, descend_containers=descend_containers)
        if nested is not None:
            out.extend(leaf_paths(nested, path, descend_containers=descend_containers))
        else:
            out.append(path)
    return tuple(out)


__all__ = ["leaf_paths", "nested_block"]
