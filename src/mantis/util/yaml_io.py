"""THE YAML parser for this repo's config-shaped files — one loader, one decode.

AUDIT-1 F-45. Four YAML read paths existed over the same files and they did not agree.
`config.loader.load_config` used a duplicate-key-refusing loader and read with no
`encoding=`; `encoding.audit_sections` §4 — the section whose stated job is to report on
"whatever `load_config` accepts" — used a bare `yaml.safe_load`, which is LAST-WINS on a
duplicate key, so it could report a file clean that the loader refuses; `tools/mint_config.py`
read utf-8 through yet another call. A reader strictly more permissive than the loader it
reports on is not an audit.

It lives in `mantis.util` (a DAG leaf) rather than in `mantis.config` because
`mantis.encoding` is one of its consumers and `mantis.config` imports `mantis.encoding` —
putting it in the config package makes an import cycle, which CI gate 9 refuses.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class DuplicateKeyError(ValueError):
    """A YAML mapping declared the same key twice (silent last-wins is banned)."""


class UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that HARD-ERRORS on a duplicate key at any nesting depth.

    The frozen utils/config.py merely LOGGED a warning and kept the last value; a duplicate
    key is a copy-paste hazard, so it becomes a load-time error. Safe construction is
    preserved (SafeLoader subclass — no arbitrary object construction).
    """

    def construct_mapping(self, node: Any, deep: bool = False) -> dict[Any, Any]:
        """Build the mapping, refusing a repeated key.

        Raises:
            DuplicateKeyError: the same key appears twice in one mapping.
        """
        seen: set[Any] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise DuplicateKeyError(f"duplicate key {key!r} at {key_node.start_mark}")
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def parse_config_yaml(path: str | Path) -> Any:
    """Parse one YAML file with THE config parser — duplicate keys refused, UTF-8 decoded.

    Returns the raw parsed object; callers decide what shape they require.

    Raises:
        DuplicateKeyError: a mapping declared the same key twice, at any depth.
        yaml.YAMLError: the file is not well-formed YAML.
        OSError: the path cannot be read (`IsADirectoryError` for a real directory).
        UnicodeDecodeError: the file is not valid UTF-8.
    """
    return yaml.load(Path(path).read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
