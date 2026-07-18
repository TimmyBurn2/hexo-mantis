"""Config loader: yaml.safe_load (duplicate-key-rejecting) -> schema validation. No merge,
no defaults, no env expansion. Missing key and unknown key both raise pydantic.ValidationError
(loud, listing every error). A duplicate YAML key (frozen loader silently last-won) is a
HARD error here (judgment #8)."""
from pathlib import Path

import yaml

from mantis.config.schema import RunConfig


class DuplicateKeyError(ValueError):
    """A YAML mapping declared the same key twice (silent last-wins is banned)."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that HARD-ERRORS on a duplicate key at any nesting depth.

    The frozen utils/config.py merely LOGGED a warning and kept the last value; a duplicate
    key is a copy-paste hazard, so it becomes a load-time error. Safe construction is preserved
    (SafeLoader subclass — no arbitrary object construction).
    """

    def construct_mapping(self, node, deep=False):
        seen: set = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise DuplicateKeyError(
                    f"duplicate key {key!r} at {key_node.start_mark}"
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def load_config(path: str | Path) -> RunConfig:
    """Load and schema-validate one complete config file (duplicate keys rejected)."""
    raw = yaml.load(Path(path).read_text(), Loader=_UniqueKeyLoader)
    if not isinstance(raw, dict):
        raise TypeError(f"{path}: config root must be a mapping")
    return RunConfig.model_validate(raw)
