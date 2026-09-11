"""THE one read path for R347(e)'s lever: where check 14 runs; absence is a named raise."""
from collections.abc import Mapping
from typing import Any

_KEY = "inference"
_MEMBER = "edge_geometry_check"
#: The closed set of postures; the schema's `Literal` is the same set, and this is its consumer.
EDGE_GEOMETRY_CHECK_MODES: tuple[str, ...] = ("inline", "checker_thread")


class MissingEdgeGeometryCheckError(ValueError):
    """The lever's key is absent or not one of its postures. Names the missing LEVEL."""


def resolve_edge_geometry_check(full_config: Any) -> str:
    """Return the posture token for `inference.edge_geometry_check`.

    Raises:
        MissingEdgeGeometryCheckError: no `inference` section, no member, or an unknown posture.
    """
    if not isinstance(full_config, Mapping) or _KEY not in full_config:
        raise MissingEdgeGeometryCheckError(
            f"{_KEY}.{_MEMBER}: the config carries no `{_KEY}` section, so the lever's posture "
            "cannot be read — a code-side fallback would report as configured (R1/LAW-11)")
    section = full_config[_KEY]
    if not isinstance(section, Mapping) or _MEMBER not in section:
        raise MissingEdgeGeometryCheckError(
            f"{_KEY}.{_MEMBER} is absent. The schema defaults it, so a config reaching here "
            "without it was not built through the one loader")
    token = section[_MEMBER]
    if token not in EDGE_GEOMETRY_CHECK_MODES:
        raise MissingEdgeGeometryCheckError(
            f"{_KEY}.{_MEMBER}={token!r} is not one of {EDGE_GEOMETRY_CHECK_MODES}")
    return str(token)


__all__ = [
    "EDGE_GEOMETRY_CHECK_MODES",
    "MissingEdgeGeometryCheckError",
    "resolve_edge_geometry_check",
]
