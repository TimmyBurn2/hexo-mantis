"""Run-config schema + loader + resolver family + emit (live since scaffold, grown in WP8).

Imports only pydantic + yaml + mantis.encoding/util (DAG: config → encoding, util) — never
torch (resolve_amp_dtype returns a string token; the model maps token → torch.dtype).
"""
from mantis.config.emit import ResolvedConfig, ResolvedKnob, resolve_config
from mantis.config.loader import DuplicateKeyError, load_config
from mantis.config.resolve import (
    AbsentEncodingError,
    BootstrapNotFoundError,
    EncodingConflictError,
    OfflineRadiusUnresolvableError,
    reconcile_encoding,
    require_offline_radius,
    resolve_amp_dtype,
    resolve_bootstrap,
    resolve_eval_model_sims,
    resolve_radius_from_schedule,
)
from mantis.config.schema import (
    SCHEMA_VERSION,
    EvalConfig,
    IdentityConfig,
    RadiusStage,
    RunConfig,
    SelfplayConfig,
    StrictModel,
)

__all__ = [
    "SCHEMA_VERSION",
    "AbsentEncodingError",
    "BootstrapNotFoundError",
    "DuplicateKeyError",
    "EncodingConflictError",
    "EvalConfig",
    "IdentityConfig",
    "OfflineRadiusUnresolvableError",
    "RadiusStage",
    "ResolvedConfig",
    "ResolvedKnob",
    "RunConfig",
    "SelfplayConfig",
    "StrictModel",
    "load_config",
    "reconcile_encoding",
    "require_offline_radius",
    "resolve_amp_dtype",
    "resolve_bootstrap",
    "resolve_config",
    "resolve_eval_model_sims",
    "resolve_radius_from_schedule",
]
