"""THE one read path for A4-3's lever: whether the serving trunk is compiled; absence raises."""
from collections.abc import Mapping
from typing import Any

_KEY = "inference"
_MEMBER = "compile_trunk"


class MissingCompileTrunkError(ValueError):
    """The lever's key is absent or not a bool. Names the missing LEVEL."""


def resolve_compile_trunk(full_config: Any) -> bool:
    """Return `inference.compile_trunk`.
    Raises:
        MissingCompileTrunkError: no `inference` section, no member, or a non-bool value."""
    if not isinstance(full_config, Mapping) or _KEY not in full_config:
        raise MissingCompileTrunkError(
            f"{_KEY}.{_MEMBER}: the config carries no `{_KEY}` section, so the lever's posture "
            "cannot be read — a code-side fallback would report as configured (R1/LAW-11)")
    section = full_config[_KEY]
    if not isinstance(section, Mapping) or _MEMBER not in section:
        raise MissingCompileTrunkError(
            f"{_KEY}.{_MEMBER} is absent. The schema defaults it, so a config reaching here "
            "without it was not built through the one loader")
    value = section[_MEMBER]
    if not isinstance(value, bool):
        raise MissingCompileTrunkError(f"{_KEY}.{_MEMBER}={value!r} is not a bool")
    return value


__all__ = ["MissingCompileTrunkError", "resolve_compile_trunk"]
