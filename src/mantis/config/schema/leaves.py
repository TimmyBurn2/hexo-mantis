"""THE schema leaf walker — the one derivation of `RunConfig`'s leaf key-paths.

Five hand-mirrored copies of this walk once disagreed on the leaf count, each asserted as the
schema's leaf set. The two wanted answers are a PARAMETER, not a copy: `descend_containers` says
whether the caller wants writable key-paths or every field name.

A nested block is a bare `BaseModel`, or a union whose non-`None` arms name exactly ONE
`BaseModel` — an optional block is DESCENDED, because an unconsumed key inside an optional block
once passed the full tier and gates 7 and 12 green. A union naming two different blocks is a leaf:
there is no single key-path to hand out. A generic container (`list[SubModel]`, `dict[str, Block]`)
is one leaf by default, since its members are addressed by index or runtime key.
"""
from __future__ import annotations

import typing
from types import UnionType
from typing import TypeGuard, Union

from pydantic import BaseModel


def _is_block(annotation: object) -> TypeGuard[type[BaseModel]]:
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def nested_block(annotation: object, *, descend_containers: bool = False) -> type[BaseModel] | None:
    """Return the single nested config BLOCK an annotation names, or None if the field is a leaf.

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
    """Return every leaf key-path of `model`, dotted, in declaration order.

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
