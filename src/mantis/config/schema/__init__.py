"""Run-config schema package (contract run-config-schema v1) — WPSC Phase 2 §10 split.

`schema.py` outgrew the 300-line soft cap once TrainConfig/SelfplayConfig/MonitorConfig were
promoted to first-class schema fields; this package keeps each concern's field census under
the cap while re-exporting the full pre-split public surface so every existing
`from mantis.config.schema import X` call site is unaffected by the split.
"""
from mantis.config.schema.core import (
    _EVAL_TIMEOUT_CEILING_SEC,
    ARCH_SCOPED_KEYS,
    SCHEMA_VERSION,
    ArchScopedKey,
    EvalConfig,
    GateConfig,
    IdentityConfig,
    LadderConfig,
    LadderRung,
    PlyCapAdjudicationConfig,
    RunConfig,
    StrengthFloorConfig,
    StrictModel,
)
from mantis.config.schema.monitor import (
    DiskGuardConfig,
    DrainCapsConfig,
    MonitorSchemaConfig,
)
from mantis.config.schema.selfplay import (
    InferenceConfig,
    MctsConfig,
    PlayoutCapConfig,
    SelfplayConfig,
)
from mantis.config.schema.train import TrainConfig

__all__ = [
    "ARCH_SCOPED_KEYS",
    "SCHEMA_VERSION",
    "ArchScopedKey",
    "DiskGuardConfig",
    "DrainCapsConfig",
    "EvalConfig",
    "GateConfig",
    "IdentityConfig",
    "InferenceConfig",
    "LadderConfig",
    "LadderRung",
    "MctsConfig",
    "MonitorSchemaConfig",
    "PlayoutCapConfig",
    "PlyCapAdjudicationConfig",
    "RunConfig",
    "SelfplayConfig",
    "StrengthFloorConfig",
    "StrictModel",
    "TrainConfig",
    "_EVAL_TIMEOUT_CEILING_SEC",
]
