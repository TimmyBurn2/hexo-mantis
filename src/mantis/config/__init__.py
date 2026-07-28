"""Run-config schema + loader + resolver family + emit (live since scaffold, grown in WP8).

Imports only pydantic + yaml + mantis.encoding/util (DAG: config → encoding, util) — never
torch (resolve_amp_dtype returns a string token; the model maps token → torch.dtype).
"""
from mantis.config.emit import ResolvedConfig, ResolvedKnob, resolve_config
from mantis.config.loader import (
    CONFIG_SUFFIXES,
    ConfigSuffixError,
    DuplicateKeyError,
    discover_configs,
    is_config_path,
    load_config,
)
from mantis.config.resolve import (
    AbsentEncodingError,
    BootstrapNotFoundError,
    EncodingConflictError,
    reconcile_encoding,
    resolve_amp_dtype,
    resolve_bootstrap,
    resolve_eval_model_sims,
)
from mantis.config.schema import (
    SCHEMA_VERSION,
    EvalConfig,
    IdentityConfig,
    RunConfig,
    SelfplayConfig,
    StrictModel,
    TrainConfig,
)

__all__ = [
    "CONFIG_SUFFIXES",
    "SCHEMA_VERSION",
    "AbsentEncodingError",
    "BootstrapNotFoundError",
    "ConfigSuffixError",
    "DuplicateKeyError",
    "EncodingConflictError",
    "EvalConfig",
    "IdentityConfig",
    "ResolvedConfig",
    "ResolvedKnob",
    "RunConfig",
    "SelfplayConfig",
    "StrictModel",
    "TrainConfig",
    "discover_configs",
    "is_config_path",
    "load_config",
    "reconcile_encoding",
    "resolve_amp_dtype",
    "resolve_bootstrap",
    "resolve_config",
    "resolve_eval_model_sims",
]
