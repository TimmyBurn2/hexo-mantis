"""Config loader: yaml.safe_load -> schema validation. No merge, no defaults, no env
expansion. Missing key and unknown key both raise pydantic.ValidationError (loud,
listing every error — pydantic collects all)."""
from pathlib import Path

import yaml

from mantis.config.schema import RunConfig


def load_config(path: str | Path) -> RunConfig:
    """Load and schema-validate one complete config file."""
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise TypeError(f"{path}: config root must be a mapping")
    return RunConfig.model_validate(raw)
