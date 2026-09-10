"""Config loader: duplicate-key-rejecting yaml.safe_load -> schema validation. No merge, no
defaults, no env expansion; missing key, unknown key and duplicate key all raise.

Also carries `discover_configs`, the one enumeration gate 7 and gate 12 share, under the
invariant: `load_config(p)` succeeds => `p in discover_configs(root)` for every p under root.
Equivalently, a file discovery skips must be a file the loader refuses. Consequence, and it is
the point: `configs/` may hold ONLY complete configs — a stray `README.md`, `.gitkeep` or
`run5.yaml.bak` is a red gate, because the only way to spare one is a name filter.
`discover_configs` takes the directory as an argument and resolves no repo root.
"""
import hashlib
import json
from pathlib import Path

from mantis.config.schema import RunConfig
from mantis.util.yaml_io import DuplicateKeyError, parse_config_yaml

__all__ = [
    "DuplicateKeyError",
    "config_identity_sha256",
    "discover_configs",
    "load_config",
    "parse_config_yaml",
]


def discover_configs(configs_dir: str | Path) -> list[Path]:
    """Return EVERY path under `configs_dir` that is not a real directory, recursively, sorted.

    The one enumeration gate 7 and gate 12 both consume, and name-agnostic on purpose:
    `load_config` decides by content, so a suffix filter here leaves its complement
    launchable-and-invisible. `is_dir() and not is_symlink()` is the one exclusion — a real
    directory provably raises `IsADirectoryError` in the loader and `rglob` recurses through it,
    while a symlink to a directory is KEPT because `rglob` will not recurse through it and
    dropping it would hide a whole loadable subtree from both gates.
    """
    return sorted(path for path in Path(configs_dir).rglob("*")
                  if not (path.is_dir() and not path.is_symlink()))


def load_config(path: str | Path) -> RunConfig:
    """Load and schema-validate one complete config file (duplicate keys rejected).

    Shape-agnostic: a run may be launched from a path of any name. What keeps that safe is the
    invariant `discover_configs` holds — whatever this accepts under the audit root, the audit sees.

    Raises:
        DuplicateKeyError: a mapping declared the same key twice, at any depth.
        TypeError: the config root is not a mapping.
        yaml.YAMLError: the file is not well-formed YAML.
        OSError: the path cannot be read.
        UnicodeDecodeError: the file is not valid UTF-8.
        pydantic.ValidationError: the config fails the schema.
    """
    raw = parse_config_yaml(path)
    if not isinstance(raw, dict):
        raise TypeError(f"{path}: config root must be a mapping")
    return RunConfig.model_validate(raw)


def config_identity_sha256(config: RunConfig) -> str:
    """Return the ONE canonical identity hash of an in-memory config.

    sha256 over the sorted-keys JSON of `model_dump()`. Boot and the mint preflight must hash
    through this one function, or a child that read a different file stays invisible.
    """
    return hashlib.sha256(
        json.dumps(config.model_dump(), sort_keys=True, default=str).encode()
    ).hexdigest()
