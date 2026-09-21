"""Run-config schema package (contract run-config-schema v1) — WPSC Phase 2 §10 split; it re-exports
the full pre-split public surface, so every `from mantis.config.schema import X` site is unaffected."""
from mantis.config.schema.core import (
    _EVAL_TIMEOUT_CEILING_SEC,
    ARCH_SCOPED_KEYS,
    OPERATIONAL_DEFAULT_KEYS,
    SCHEMA_VERSION,
    SOFT_POLICY_ARCH_KINDS,
    ArchScopedKey,
    EvalConfig,
    GateConfig,
    IdentityConfig,
    PlyCapAdjudicationConfig,
    RunConfig,
    StrengthFloorConfig,
    StrictModel,
    WarmStartConfig,
    operational_default_fields,
)
from mantis.config.schema.leaves import leaf_paths, nested_block
from mantis.config.schema.model import AuxSoftPolicyConfig, GnnWidthsConfig, ModelConfig
from mantis.config.schema.monitor import (
    DiskGuardConfig,
    DrainCapsConfig,
    MonitorSchemaConfig,
)
from mantis.config.schema.search import DeployConfig, SearchConfig
from mantis.config.schema.selfplay import (
    InferenceConfig,
    MctsConfig,
    PlayoutCapConfig,
    SelfplayConfig,
)
from mantis.config.schema.train import EmaConfig, TrainConfig

__all__ = [
    "ARCH_SCOPED_KEYS",
    "OPERATIONAL_DEFAULT_KEYS",
    "SCHEMA_VERSION",
    "SOFT_POLICY_ARCH_KINDS",
    "ArchScopedKey",
    "AuxSoftPolicyConfig",
    "DiskGuardConfig",
    "DrainCapsConfig",
    "EvalConfig",
    "GateConfig",
    "IdentityConfig",
    "InferenceConfig",
    "GnnWidthsConfig",
    "MctsConfig",
    "ModelConfig",
    "MonitorSchemaConfig",
    "PlayoutCapConfig",
    "PlyCapAdjudicationConfig",
    "RunConfig",
    "DeployConfig",
    "SearchConfig",
    "SelfplayConfig",
    "StrengthFloorConfig",
    "StrictModel",
    "EmaConfig",
    "TrainConfig",
    "WarmStartConfig",
    "leaf_paths",
    "nested_block",
    "operational_default_fields",
    "_EVAL_TIMEOUT_CEILING_SEC",
]
